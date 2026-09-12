"""User preferences, and the measured speed that makes the estimate honest.

Follows local-upscaler's pattern, which took it from soundboard: a dataclass, a
loader that tolerates anything on disk, an atomic save. Values are clamped
rather than rejected — a hand-edited step count of 0 should become 1, not crash
the next run — and unknown keys survive the round trip so a newer version's
settings are not destroyed by an older one.

`Calibration` is the part worth explaining, and it needed rethinking for this
app. The upscaler stored one seconds-per-megapixel figure per model, because an
ESRGAN pass is very nearly linear in pixels. Diffusion is not: attention is
quadratic in the token count, so an image with four times the pixels costs far
more than four times as much. A single rate per recipe would therefore be
learned at whatever size the user last worked at and then be wrong everywhere
else — and confidently so, which is worse than having no measurement.

So the key carries three things, each of which genuinely changes the answer:

* the **recipe**, obviously;
* a coarse **megapixel bucket**, quantised to powers of two, which brackets the
  quadratic term without demanding a measurement at every exact size the user
  might pick;
* the **grade** from `hardware`, because the same recipe at the same size runs
  at roughly a third of the speed when its weights are streaming from disk
  rather than sitting in VRAM. Sharing a key between those would let one
  offloaded run poison the estimate for every resident one.

Measured on this machine, FLUX.2 Klein 4B at 512x512 samples at 22.3 s/step —
so the stored unit is seconds per *sampling step*, with load and text encoding
excluded. See `runner.throughput` for why that separation is not pedantic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import paths
from .engine import binary, catalog, hardware

#: Output size choices. None means "match the source image".
SIZE_CHOICES = (None, 512, 640, 768, 1024, 1280, 1536)
DEFAULT_SIZE = 768

#: How the memory flags are chosen. "auto" defers to `hardware.verdict`, which
#: is right almost always; the rest are escape hatches for when it is not.
MEMORY_AUTO = "auto"
MEMORY_VRAM = "vram"
MEMORY_RAM = "ram"
MEMORY_DISK = "disk"
MEMORY_MODES = (MEMORY_AUTO, MEMORY_VRAM, MEMORY_RAM, MEMORY_DISK)

MEMORY_FLAGS = {
    MEMORY_VRAM: (),
    MEMORY_RAM: ("--offload-to-cpu",),
    MEMORY_DISK: ("--params-backend", "diffusion=disk,te=cpu,vae=cpu",
                  "--vae-tiling"),
}

BACKEND_AUTO = "auto"
BACKENDS = (BACKEND_AUTO, *binary.BACKENDS)

#: How the source is drawn in the comparison view.
FILTERS = ("nearest", "smooth")

MIN_STEPS, MAX_STEPS = 1, 100
MAX_SIDE = 2048


def _clamp_int(value, lo: int, hi: int, default: int) -> int:
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


def _clamp_float(value, lo: float, hi: float, default: float) -> float:
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return default


def mpx_bucket(width: int, height: int) -> str:
    """A coarse megapixel bucket, quantised to powers of two.

    512x512 and 512x576 land in the same bucket, which is the point: they cost
    almost the same and there is no reason to make the user measure both. 512
    and 1024 do not, because they do not.
    """
    megapixels = max(0.05, (width * height) / 1e6)
    return f"{2 ** round(math.log2(megapixels)):g}mp"


@dataclass
class Calibration:
    """Measured seconds per sampling step, keyed by recipe, size and grade."""

    rates: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def key(recipe_id: str, width: int, height: int,
            grade: str = hardware.FITS) -> str:
        return f"{recipe_id}@{mpx_bucket(width, height)}/{grade}"

    def get(self, recipe_id: str, width: int, height: int,
            grade: str = hardware.FITS) -> float | None:
        v = self.rates.get(self.key(recipe_id, width, height, grade))
        return v if isinstance(v, (int, float)) and v > 0 else None

    def record(self, recipe_id: str, width: int, height: int,
               sec_per_step: float, grade: str = hardware.FITS) -> None:
        """Blend a new measurement into the stored one.

        An exponential average rather than a replacement: one run that shared
        the GPU with a game should not throw the estimate off for every run
        after it, and one with a warm shader cache should not make it
        permanently optimistic.

        There is deliberately **no** outlier rejection. The obvious guard —
        discard anything far from the catalog's prior — cannot be given a
        defensible threshold when the same recipe legitimately varies threefold
        between resident and disk-streamed weights. Keying by grade removes the
        reason those figures collide, which is the honest fix rather than a
        filter that would reject a real measurement forever.
        """
        if not (sec_per_step and sec_per_step > 0):
            return
        k = self.key(recipe_id, width, height, grade)
        prev = self.rates.get(k)
        self.rates[k] = (0.6 * prev + 0.4 * sec_per_step
                         if isinstance(prev, (int, float)) and prev > 0
                         else float(sec_per_step))

    def to_dict(self) -> dict:
        return {k: round(v, 3) for k, v in self.rates.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        rates: dict[str, float] = {}
        if isinstance(d, dict):
            for k, v in d.items():
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    continue
                if f > 0:
                    rates[str(k)] = f
        return cls(rates=rates)


@dataclass
class Settings:
    recipe_id: str = catalog.DEFAULT_RECIPE_ID
    #: None means "match the source image".
    size: int | None = DEFAULT_SIZE
    #: None means "use the recipe's own value".
    steps: int | None = None
    cfg_scale: float | None = None
    strength: float | None = None
    ip_adapter_strength: float = 1.0
    negative_prompt: str = ""
    #: -1 means "draw a fresh seed each run".
    seed: int = -1
    randomize_seed: bool = True

    memory_mode: str = MEMORY_AUTO
    backend: str = BACKEND_AUTO
    threads: int = 0
    max_vram_gb: float = 0.0
    #: A directory holding sd-cli/sd-server, or a path to one of them. Empty
    #: means "search normally".
    engine_path: str = ""
    #: Empty means the XDG default. Settable because one recipe here is 34 GB.
    models_dir: str = ""
    #: Seconds an unused engine keeps its weights — and this machine's VRAM.
    idle_timeout: int = 600

    compare_filter: str = "nearest"
    last_prompt: str = ""
    last_open_dir: str = ""
    last_save_dir: str = ""
    calibration: Calibration = field(default_factory=Calibration)
    #: Anything this version does not understand, kept for the round trip.
    _extra: dict = field(default_factory=dict, repr=False)

    _KNOWN = ("recipe_id", "size", "steps", "cfg_scale", "strength",
              "ip_adapter_strength", "negative_prompt", "seed",
              "randomize_seed", "memory_mode", "backend", "threads",
              "max_vram_gb", "engine_path", "models_dir", "idle_timeout",
              "compare_filter", "last_prompt", "last_open_dir",
              "last_save_dir", "calibration")

    # -- derived ----------------------------------------------------------
    def recipe(self) -> catalog.Recipe:
        return catalog.get(self.recipe_id)

    def models_path(self):
        return paths.models_dir(self.models_dir)

    def engine_backend(self) -> str:
        """The backend name to tell `binary.memory_args` about.

        "auto" resolves to whatever the managed engine was fetched as, which is
        Vulkan here. It matters only for deciding whether flash attention is
        available, so guessing wrong costs a flag, not a run.
        """
        return (binary.DEFAULT_BACKEND if self.backend == BACKEND_AUTO
                else self.backend)

    def memory_flags(self, verdict: hardware.Verdict) -> tuple[str, ...]:
        if self.memory_mode == MEMORY_AUTO:
            return verdict.flags
        return MEMORY_FLAGS.get(self.memory_mode, ())

    # -- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        d = dict(self._extra)
        d.update({k: getattr(self, k) for k in self._KNOWN
                  if k != "calibration"})
        d["calibration"] = self.calibration.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        if not isinstance(d, dict):
            d = {}
        s = cls(_extra={k: v for k, v in d.items() if k not in cls._KNOWN})

        s.recipe_id = catalog.get(str(d.get("recipe_id",
                                            catalog.DEFAULT_RECIPE_ID))).id

        size = d.get("size", DEFAULT_SIZE)
        s.size = (_clamp_int(size, 64, MAX_SIDE, DEFAULT_SIZE)
                  if isinstance(size, int) else None)

        steps = d.get("steps")
        s.steps = (_clamp_int(steps, MIN_STEPS, MAX_STEPS, MIN_STEPS)
                   if isinstance(steps, int) else None)

        cfg = d.get("cfg_scale")
        s.cfg_scale = (_clamp_float(cfg, 0.0, 30.0, 1.0)
                       if isinstance(cfg, (int, float)) else None)

        strength = d.get("strength")
        s.strength = (_clamp_float(strength, 0.0, 1.0, 0.75)
                      if isinstance(strength, (int, float)) else None)

        s.ip_adapter_strength = _clamp_float(d.get("ip_adapter_strength", 1.0),
                                             0.0, 2.0, 1.0)
        s.negative_prompt = str(d.get("negative_prompt") or "")
        s.seed = _clamp_int(d.get("seed", -1), -1, 2 ** 31 - 1, -1)
        s.randomize_seed = bool(d.get("randomize_seed", True))

        s.memory_mode = (d.get("memory_mode") if d.get("memory_mode")
                         in MEMORY_MODES else MEMORY_AUTO)
        s.backend = d.get("backend") if d.get("backend") in BACKENDS else BACKEND_AUTO
        s.threads = _clamp_int(d.get("threads", 0), 0, 256, 0)
        s.max_vram_gb = _clamp_float(d.get("max_vram_gb", 0.0), 0.0, 512.0, 0.0)
        s.engine_path = str(d.get("engine_path") or "")
        s.models_dir = str(d.get("models_dir") or "")
        s.idle_timeout = _clamp_int(d.get("idle_timeout", 600), 0, 86400, 600)

        s.compare_filter = (d.get("compare_filter")
                            if d.get("compare_filter") in FILTERS else "nearest")
        s.last_prompt = str(d.get("last_prompt") or "")
        s.last_open_dir = str(d.get("last_open_dir") or "")
        s.last_save_dir = str(d.get("last_save_dir") or "")
        s.calibration = Calibration.from_dict(d.get("calibration") or {})
        return s


def load() -> Settings:
    return Settings.from_dict(paths.read_json(paths.settings_file(), {}))


def save(s: Settings) -> None:
    paths.write_json(paths.settings_file(), s.to_dict())
