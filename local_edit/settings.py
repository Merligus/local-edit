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
DEFAULT_SIZE = 512

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
            grade: str = hardware.FITS,
            backend: str = binary.DEFAULT_BACKEND) -> str:
        """Storage key. The backend is in it because it changes the answer most.

        On the development machine the same recipe at the same size samples at
        23.9 s on CUDA and 81.3 s on Vulkan — a factor of 3.4, larger than the
        spread between any two grades. Sharing a key between them would let one
        CUDA run tell every later Vulkan run it was three times faster than it
        is, and the exponential blend would take several runs to walk that back.

        Vulkan keeps the unsuffixed form so calibration collected before this
        distinction existed is still read rather than silently discarded.
        """
        base = f"{recipe_id}@{mpx_bucket(width, height)}/{grade}"
        return base if backend == binary.DEFAULT_BACKEND else f"{base}/{backend}"

    def get(self, recipe_id: str, width: int, height: int,
            grade: str = hardware.FITS,
            backend: str = binary.DEFAULT_BACKEND) -> float | None:
        v = self.rates.get(self.key(recipe_id, width, height, grade, backend))
        return v if isinstance(v, (int, float)) and v > 0 else None

    def record(self, recipe_id: str, width: int, height: int,
               sec_per_step: float, grade: str = hardware.FITS,
               backend: str = binary.DEFAULT_BACKEND) -> None:
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
        k = self.key(recipe_id, width, height, grade, backend)
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

    #: False until an output size has been settled on — either by the user
    #: touching the control, or by `SetupPage` picking one from the measured
    #: hardware on first run. See `pick_size`.
    size_chosen: bool = False

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
              "size_chosen", "compare_filter", "last_prompt", "last_open_dir",
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
        # A settings file that predates this flag but names a size was written
        # by a user who had one, so honour it rather than overriding on the next
        # launch.
        s.size_chosen = bool(d.get("size_chosen", "size" in d))

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


#: Where the output size starts before anything has been measured. Small on
#: purpose: see `SetupPage._auto_size` for why the app opens conservative and
#: earns the right to raise it rather than guessing from the hardware.
CONSERVATIVE_SIZE = 512

#: How long one edit should take, for `SetupPage._auto_size`. Five minutes is
#: about the limit of "let me try this and see".
AUTO_SIZE_TARGET_S = 300.0


def auto_size(recipe, hw, calibration: Calibration,
              target_seconds: float = AUTO_SIZE_TARGET_S,
              backend: str = binary.DEFAULT_BACKEND,
              references: int = 0) -> int:
    """The largest output size that stays under the target at measured speed.

    A fixed default cannot serve both a 4 GB Pascal card and a 24 GB one:
    768 px is ten minutes on the development machine and about a second on a
    modern GPU, while 512 px would be a needlessly small image on the latter.
    So the size adjusts itself — but **only from measurements**.

    That restriction is the whole design. The catalog's timing prior is anchored
    on one specific GPU, and there is no honest way to guess another card's
    speed from anything `hardware` can probe: VRAM tells you what fits, not how
    fast it runs. An earlier attempt scaled the prior by invented per-tier
    factors and produced the same answer on every machine, which is exactly the
    failure it was meant to avoid. So a size with no measurement behind it ends
    the search rather than being estimated into.

    The app therefore opens conservative and earns the right to raise it: the
    first edit is measured, and from the second onwards this moves up. The
    asymmetry justifies starting low — too large a default on a slow card is a
    fifteen-minute wait that reads as a hang, while too small a default on a
    fast card is an image that arrives in a second beside a visible estimate
    inviting a bigger one.

    Kept here, free of Qt, so it can be tested without a display.
    """
    from .engine import hardware as hardware_mod
    from .engine import runner as runner_mod

    best = CONSERVATIVE_SIZE
    for candidate in sorted(c for c in SIZE_CHOICES if c):
        if candidate < CONSERVATIVE_SIZE:
            continue
        width, height = runner_mod.plan_size(recipe, None, candidate)
        verdict = hardware_mod.verdict(recipe, hw, width, height,
                                       references=references)
        if verdict.grade in (hardware_mod.TOO_BIG, hardware_mod.NO_DISK):
            break
        rate = calibration.get(recipe.id, width, height, verdict.grade,
                               backend)
        if rate is None:
            break                     # no measurement here: do not speculate
        if runner_mod.estimate_seconds(recipe, width, height, recipe.steps,
                                       rate, verdict.grade,
                                       references=max(1, references),
                                       ) > target_seconds:
            break
        best = candidate
    return best


def fitting_size(recipe, hw, source_size: tuple[int, int] | None = None,
                 references: int = 0) -> int | None:
    """The largest offered size this machine can actually hold.

    Walks `SIZE_CHOICES` upward and stops at the first one the hardware cannot
    take. Returns `None` when *nothing* offered fits, which is a real answer:
    the caller then has no size to suggest and must say so rather than pretend.

    Unlike `auto_size` this asks only "does it fit", never "how long will it
    take", so it needs no measurement and gives an answer on a machine the app
    has never run on.
    """
    from .engine import hardware as hardware_mod
    from .engine import runner as runner_mod

    best: int | None = None
    for candidate in sorted(c for c in SIZE_CHOICES if c):
        width, height = runner_mod.plan_size(recipe, source_size, candidate)
        grade = hardware_mod.verdict(recipe, hw, width, height,
                                     references=references).grade
        if grade in (hardware_mod.TOO_BIG, hardware_mod.NO_DISK):
            break
        best = candidate
    return best


def plan_output_size(recipe, hw, source_size: tuple[int, int] | None,
                     longest: int | None,
                     references: int = 0) -> tuple[int, int, int | None]:
    """Output dimensions, with "match the source" capped at what fits.

    Returns `(width, height, capped_to)`; `capped_to` is the size the cap
    landed on, or `None` when no cap was applied.

    Only "match the source" is capped, and only when the uncapped size is
    actually `TOO_BIG` on this machine. Both restrictions matter:

    * An explicit choice from the combo box is the user's, and silently
      overriding it would make the control a lie. A too-large explicit size is
      handled by the confirmation in `MainWindow._start` instead, which asks.
    * Capping unconditionally would shrink full-resolution output on a card
      that can manage it, which is the whole point of owning such a card.

    "Match the source" is the one that needs this, because it is not a size at
    all — it is whatever the camera produced. A 3200x4032 phone photo asks for
    12.9 megapixels, roughly fifty times a 512px run, and the engine answers
    that it needs 10.4 GB for a single attention segment. That combination
    took a 4 GB card from "slow" to "fails after two and a half seconds".
    """
    from .engine import hardware as hardware_mod
    from .engine import runner as runner_mod

    width, height = runner_mod.plan_size(recipe, source_size, longest)
    if longest is not None or not source_size:
        return width, height, None
    if hardware_mod.verdict(recipe, hw, width, height,
                            references=references).grade != hardware_mod.TOO_BIG:
        return width, height, None

    cap = fitting_size(recipe, hw, source_size, references) or CONSERVATIVE_SIZE
    width, height = runner_mod.plan_size(recipe, source_size, cap)
    return width, height, cap


def load() -> Settings:
    return Settings.from_dict(paths.read_json(paths.settings_file(), {}))


def save(s: Settings) -> None:
    paths.write_json(paths.settings_file(), s.to_dict())
