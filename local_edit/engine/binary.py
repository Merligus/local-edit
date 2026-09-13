"""Finding and invoking stable-diffusion.cpp.

The engine is a pair of standalone binaries — `sd-cli` and `sd-server` — driven
over Vulkan, not a Python library. That is the same choice local-upscaler made,
though **not** for the reason first written here.

The original claim was that PyTorch could not run on this machine at all: that it
dropped Pascal (`sm_61`) after 2.6.0, and that Python 3.14 had no CUDA wheels.
Both are false, and measuring them said so:

    torch 2.14.0+cu126 on Python 3.14.7
    compiled archs : ['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']
    device         : NVIDIA GeForce GTX 1050 Ti  (sm_61)
    cuda available : True
    fp32 4096^2 matmul: 1.89 TFLOPS   (the card's spec sheet is ~2.1)

The cu126 index publishes `cp314` wheels; only the default cu128 index does not.
And `sm_61` being absent from the arch list does not matter, because CUDA cubins
are binary-compatible *within* a major compute capability — an `sm_60` binary
runs on any `sm_6x` part, with no PTX JIT involved.

So this is a trade-off, not a necessity, and the honest version is: ggml over
Vulkan is 46 MB against roughly 10 GB of torch and CUDA runtime, needs no
virtualenv on a project whose stated policy is system packages only, and has
per-segment weight streaming from *disk* (`--params-backend diffusion=disk`)
that diffusers has no equivalent for — which is what lets the 13-34 GB tiers be
attempted at all on 11 GB of RAM. Those are real advantages. "PyTorch will not
run here" was not one of them.

It also reports `fp16: 0`, because Pascal has no usable half-precision path
under Vulkan. Everything runs in fp32, which roughly doubles the working set —
`hardware.Hardware.fp16` carries that through to the memory grading, and it is
the single number that best explains why this card is slow.

**Flash attention is deliberately not enabled on Vulkan.** `--diffusion-fa` is
the obvious memory win, and upstream's own performance notes list its backends
as CPU, CUDA/ROCm and Metal — Vulkan is not among them. Passing it here would
either be ignored or produce wrong images, so `memory_args` adds it only for a
backend that supports it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .. import paths
from .catalog import Recipe

CLI_NAME = "sd-cli"
SERVER_NAME = "sd-server"

#: Which prebuilt release to fetch. Backends that have no Linux prebuilt are
#: absent on purpose: upstream publishes CUDA binaries for Windows only, so a
#: CUDA user on Linux builds from source and pins the result in settings. See
#: docs/COMPATIBILITY.md.
BACKENDS = ("vulkan", "cpu", "rocm")
DEFAULT_BACKEND = "vulkan"


@dataclass(frozen=True)
class Release:
    """A pinned upstream build. Updating one is a deliberate edit, never a drift."""

    backend: str
    tag: str
    asset: str
    size: int
    sha256: str

    def url(self) -> str:
        return ("https://github.com/leejet/stable-diffusion.cpp/releases/"
                f"download/{self.tag}/{self.asset}")


#: The pinned upstream build. The tag carries a build number that the asset
#: filenames do not — `master-859-7f410a3` against `sd-master-7f410a3-bin-…` —
#: so the names are written out rather than derived from it. Deriving them is
#: how the first version of this shipped three URLs that all 404'd;
#: `tests/test_catalog_remote.py` is what caught it.
_TAG = "master-859-7f410a3"
_STEM = "sd-master-7f410a3-bin-Linux-Ubuntu-24.04-x86_64"

RELEASES = {
    "vulkan": Release(
        "vulkan", _TAG, f"{_STEM}-vulkan.zip", 46_348_071,
        "3f10e3b00fc6574d014044e40c23db33d99d115e7c2ecf6ce8733fcc5287a743"),
    "cpu": Release(
        "cpu", _TAG, f"{_STEM}.zip", 33_224_779,
        "3f3e1a6b57a2e4d184aa9ea9ab916272ea92b4b79d5af667cad89d0a3edc2bc7"),
    "rocm": Release(
        "rocm", _TAG, f"{_STEM}-rocm-7.14.0.zip", 264_527_387,
        "4a2d142ae4c594a49016188807702d5bd8b5129e3274658f6797f18d029730b8"),
}

#: Backends whose ggml build has a flash-attention kernel. See the module
#: docstring — this is why the list is not simply "all of them".
FA_BACKENDS = frozenset({"cuda", "rocm", "cpu"})


# ------------------------------------------------------------------ locating
def managed_dir() -> Path:
    return paths.engine_dir()


def find(name: str, explicit: str | None = None) -> Path | None:
    """Locate `sd-cli` or `sd-server`.

    Three sources in order: a directory the user pinned in settings, the copy
    `--fetch-engine` manages, then `PATH`. The settings override wins outright
    because a user who built sd.cpp with CUDA for their own card meant it — that
    is the documented upgrade path for hardware this app cannot prebuild for.
    """
    if explicit:
        base = Path(explicit).expanduser()
        candidate = base / name if base.is_dir() else base
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    managed = managed_dir() / name
    if managed.is_file() and os.access(managed, os.X_OK):
        return managed
    found = shutil.which(name)
    return Path(found) if found else None


def find_cli(explicit: str | None = None) -> Path | None:
    return find(CLI_NAME, explicit)


def find_server(explicit: str | None = None) -> Path | None:
    return find(SERVER_NAME, explicit)


def child_env(exe: Path | None = None) -> dict[str, str]:
    """Environment for an engine subprocess.

    Two jobs. The first is the shared libraries: unlike local-upscaler's single
    self-contained executable, this release is `sd-cli` plus about twenty `.so`
    files sitting beside it, so the directory holding the binary has to be on
    `LD_LIBRARY_PATH` or it will not start.

    The second is the same one the upscaler had. Vulkan overlay layers inject
    themselves into every Vulkan process, and MangoHud prints its banner onto
    the very stream `progress.Parser` reads. Turned off explicitly rather than
    papered over in the parser.
    """
    env = dict(os.environ)
    directory = str((exe.parent if exe is not None else managed_dir()).resolve())
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{directory}:{existing}" if existing else directory
    env["MANGOHUD"] = "0"
    env["DISABLE_VKBASALT"] = "1"
    return env


def list_devices(explicit: str | None = None, timeout: float = 60.0) -> str:
    """Raw `--list-devices` output, or "".

    Both streams are returned joined: ggml logs the backend load to stderr and
    prints the device table to stdout, and `hardware._engine_fp16` needs the
    capability line from the former.
    """
    exe = find_cli(explicit)
    if exe is None:
        return ""
    try:
        r = subprocess.run([str(exe), "--list-devices"], capture_output=True,
                           text=True, timeout=timeout, env=child_env(exe))
    except (OSError, subprocess.SubprocessError):
        return ""
    return (r.stdout or "") + (r.stderr or "")


def probe(exe: Path, timeout: float = 30.0) -> str | None:
    """Return the usage text if `exe` runs and speaks the flags this app needs.

    Checked by capability rather than by version string, so a user's own build
    passes as long as it is new enough to have what is used here. `--ref-image`
    is the load-bearing one: without repeatable references there is no app.
    """
    try:
        r = subprocess.run([str(exe), "--help"], capture_output=True, text=True,
                           timeout=timeout, env=child_env(exe))
    except (OSError, subprocess.SubprocessError):
        return None
    text = (r.stdout or "") + (r.stderr or "")
    needed = ("--ref-image", "--diffusion-model", "--llm", "--params-backend",
              "--offload-to-cpu", "--vae-tiling")
    return text if all(flag in text for flag in needed) else None


# ------------------------------------------------------------- argv building
def model_args(recipe: Recipe, models_dir: Path) -> list[str]:
    """The `--diffusion-model … --llm … --vae …` flags for one recipe.

    Shared by both engines: `sd-server` parses the same generation options as
    `sd-cli` and uses them as the defaults for every request, which is how the
    weights get loaded once and stay loaded.
    """
    argv: list[str] = []
    for f in recipe.files:
        argv += [f.flag(), str(models_dir / f.subdir() / f.filename())]
    return argv


def memory_args(flags: tuple[str, ...], backend: str = DEFAULT_BACKEND,
                max_vram_gb: float = 0.0) -> list[str]:
    """Memory flags from a `hardware.Verdict`, plus backend-specific extras."""
    argv = list(flags)
    if backend in FA_BACKENDS:
        argv.append("--diffusion-fa")
    if max_vram_gb > 0:
        argv += ["--max-vram", f"{max_vram_gb:g}"]
    return argv


def sampling_args(recipe: Recipe, steps: int | None = None,
                  cfg_scale: float | None = None, seed: int = -1,
                  width: int = 0, height: int = 0,
                  negative_prompt: str = "") -> list[str]:
    """Generation parameters common to both engines."""
    argv = [
        "--steps", str(steps if steps is not None else recipe.steps),
        "--cfg-scale", f"{cfg_scale if cfg_scale is not None else recipe.cfg_scale:g}",
        "--sampling-method", recipe.sampling_method,
        "--seed", str(seed),
    ]
    if recipe.scheduler:
        argv += ["--scheduler", recipe.scheduler]
    if recipe.guidance is not None:
        argv += ["--guidance", f"{recipe.guidance:g}"]
    if recipe.flow_shift is not None:
        argv += ["--flow-shift", f"{recipe.flow_shift:g}"]
    if width and height:
        argv += ["-W", str(width), "-H", str(height)]
    if negative_prompt:
        argv += ["--negative-prompt", negative_prompt]
    return argv


def build_cli_argv(exe: Path, recipe: Recipe, models_dir: Path, *,
                   prompt: str, output: Path,
                   init_image: Path | None = None,
                   ref_images: tuple[Path, ...] = (),
                   ip_adapter_image: Path | None = None,
                   pm_id_dir: Path | None = None,
                   memory: tuple[str, ...] = (),
                   backend: str = DEFAULT_BACKEND,
                   steps: int | None = None, cfg_scale: float | None = None,
                   seed: int = -1, width: int = 0, height: int = 0,
                   strength: float | None = None, negative_prompt: str = "",
                   threads: int = 0, max_vram_gb: float = 0.0) -> list[str]:
    """One complete `sd-cli` command line.

    Used for recipes the server's JSON API cannot express — PhotoMaker's
    `--pm-id-images-dir` is a directory path fixed at server startup — and as a
    debuggable path for everything else: the argv this returns can be pasted
    into a shell verbatim.
    """
    argv = [str(exe), *model_args(recipe, models_dir),
            "-p", prompt, "-o", str(output)]
    argv += sampling_args(recipe, steps, cfg_scale, seed, width, height,
                          negative_prompt)
    argv += memory_args(memory, backend, max_vram_gb)

    if init_image is not None:
        argv += ["-i", str(init_image),
                 "--strength",
                 f"{strength if strength is not None else recipe.strength:g}"]
    for ref in ref_images:
        argv += ["-r", str(ref)]
    if len(ref_images) > 1:
        # Without this the model cannot tell the references apart, and a prompt
        # that says "the jacket in image 2" has nothing to refer to.
        argv.append("--increase-ref-index")
    if ip_adapter_image is not None:
        argv += ["--ip-adapter-image", str(ip_adapter_image)]
    if pm_id_dir is not None:
        argv += ["--pm-id-images-dir", str(pm_id_dir)]
    if threads > 0:
        argv += ["-t", str(threads)]
    argv.append("-v")
    return argv


def build_server_argv(exe: Path, recipe: Recipe, models_dir: Path, *,
                      port: int, host: str = "127.0.0.1",
                      memory: tuple[str, ...] = (),
                      backend: str = DEFAULT_BACKEND,
                      threads: int = 0, max_vram_gb: float = 0.0) -> list[str]:
    """The command line for a warm `sd-server` holding one recipe.

    Only the model and memory flags go here. Everything that varies per edit —
    prompt, references, seed, size — travels in the request body, which is the
    entire point of running a server: the weights are loaded once.
    """
    argv = [str(exe), *model_args(recipe, models_dir),
            "--listen-ip", host, "--listen-port", str(port)]
    argv += memory_args(memory, backend, max_vram_gb)
    if threads > 0:
        argv += ["-t", str(threads)]
    argv.append("-v")
    return argv
