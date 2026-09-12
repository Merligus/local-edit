"""What this machine can actually run, and what it would have to do to try.

local-upscaler needed nothing like this: its models were 1-64 MB and every one
of them fit on any GPU with a Vulkan driver, so the model list was just a list.
Here the catalog spans 3.5 GB to 34.5 GB against a card with 3.4 GiB free, and a
plain list would be a trap — half the entries would simply fail, minutes into a
run, with an allocator error.

So every recipe is graded against the live machine and the grade is shown in the
model list. Nothing is hidden: a recipe that needs a bigger GPU says so, with the
number, which is more useful than its absence. The grade is also what picks the
memory flags — `FITS` needs none, `OFFLOAD` adds `--offload-to-cpu`, `STREAM`
adds `--params-backend diffusion=disk`.

Three measurements, all cheap and all re-read per refresh, because "free VRAM"
changes when a browser opens a video:

* **VRAM** from `vulkaninfo`'s `DEVICE_LOCAL` heap `budget`, which is what the
  Vulkan allocator will actually hand out right now rather than the size printed
  on the box. `nvidia-smi` is the fallback for a machine without vulkan-tools.
* **RAM** from `MemAvailable`, never `MemFree` — the page cache is reclaimable
  and `MemFree` would call a healthy machine full.
* **Disk** from `statvfs` on the models directory, which may be a different
  volume from `$HOME`; that is the point of making it configurable.

**Swap is deliberately not counted as memory.** On the development machine all
12 GB of it is `zram` — compressed RAM, not a disk-backed extension of it. Model
weights are already quantised and do not compress, so a recipe that "fits in RAM
plus swap" would in practice thrash to death. `swap_is_zram` records this so the
UI can say why rather than silently ignoring 12 GB the user can see.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .catalog import Recipe

#: Weights and working set both fit in VRAM. Full speed.
FITS = "fits"
#: Weights live in RAM and stream to the GPU per layer. `--offload-to-cpu`.
OFFLOAD = "offload"
#: Weights live on disk and are read per segment. `--params-backend ...=disk`.
STREAM = "stream"
#: The working set alone will not fit on this card. No amount of offloading
#: helps, because offloading moves *weights*; activations have to be resident.
TOO_BIG = "too_big"
#: The card is big enough, but something else is holding the VRAM right now.
#: A different problem from TOO_BIG with a different fix — close the other
#: application, rather than buy a graphics card — so it gets its own word.
BUSY = "busy"
#: Not enough free space to download the weights in the first place.
NO_DISK = "no_disk"
#: The probe found nothing to grade against.
UNKNOWN = "unknown"

#: Rough wall-clock multipliers against `FITS`.
#:
#: Offloading is very nearly free, which is surprising until you read what it
#: does: upstream describes `--offload-to-cpu` as saving VRAM "without reducing
#: generation speed", because the per-layer copy from RAM is prefetched and
#: overlaps the previous layer's compute. The development machine's only
#: measurement so far was taken offloaded, so anything much above 1.0 here
#: would be double-counting a cost that did not appear.
#:
#: Disk streaming is a different matter — a segment has to be read before it
#: can be computed, and there is nothing to overlap it with on the first pass.
SPEED_PENALTY = {FITS: 1.0, OFFLOAD: 1.05, STREAM: 3.0,
                 TOO_BIG: 2.5, BUSY: 2.5, NO_DISK: 1.0, UNKNOWN: 1.0}

#: Leave this much VRAM for the compositor and everything else. The budget
#: figure already excludes other processes' current usage, but a desktop's
#: usage moves, and an allocator failure mid-run costs the whole run.
VRAM_MARGIN_GB = 0.25

#: Headroom on the working set, above sd.cpp's own reported peak compute buffer.
#:
#: That figure is the peak across segments and is not everything resident at
#: once: the staged parameters for the segment being computed sit beside it
#: (measured at 66-120 MB here), and a Vulkan allocator that has been handing
#: out and returning buffers for several minutes does not have its free space in
#: one piece.
#:
#: 25% is an engineering margin rather than a derivation, but it is not
#: arbitrary either — it is what separates the two observed outcomes on this
#: machine. FLUX.2 Klein 4B at 1024x1024 reports a 2.76 GB peak against 3.41 GB
#: free, which looks like a comfortable fit and is not: the real run died at
#: segment 15 of 27 with `ErrorOutOfDeviceMemory`. 2.76 x 1.25 = 3.45 exceeds
#: 3.41 and would have predicted it.
WORK_HEADROOM = 1.25

#: Above this share of the card held by something else, "your GPU is busy" is
#: the true explanation rather than "your GPU is small". Below it, the holder is
#: the desktop — which on this machine is about 0.6 GB of a 4.3 GB card, is
#: always there, and cannot be closed, so it is part of the card's real size.
BUSY_SHARE = 0.30
#: The same idea for RAM, where the cost of being wrong is the OOM killer.
RAM_MARGIN_GB = 1.0

_GIB = 1024 ** 3


@dataclass(frozen=True)
class Hardware:
    """A snapshot of what is available. All figures in GB (10^9), not GiB."""

    vram_gb: float = 0.0
    vram_total_gb: float = 0.0
    ram_gb: float = 0.0
    ram_total_gb: float = 0.0
    disk_gb: float = 0.0
    gpu_name: str = ""
    #: False on Pascal under Vulkan, which has no usable half-precision path.
    #: Activations are then fp32, so the working set is about twice the size.
    fp16: bool = True
    swap_is_zram: bool = False

    @property
    def known(self) -> bool:
        return self.vram_gb > 0 or self.ram_gb > 0

    @property
    def busy_gb(self) -> float:
        """VRAM something else is holding right now."""
        return max(0.0, self.vram_total_gb - self.vram_gb)

    def working_multiplier(self) -> float:
        return 1.0 if self.fp16 else 2.0


@dataclass(frozen=True)
class Verdict:
    """How `recipe` would run here, and what it would take."""

    recipe_id: str
    grade: str
    #: VRAM the working set needs at the requested size.
    working_gb: float
    #: Bytes still to download. 0 when the weights are already present.
    download_gb: float
    #: One short clause for the model list, e.g. "offloads to RAM".
    summary: str
    #: The sd.cpp flags this grade implies.
    flags: tuple[str, ...] = ()

    @property
    def runnable(self) -> bool:
        """Whether to let the user start this at all.

        Only a disk shortage is a hard stop, because it is the one failure the
        app can predict with certainty. `TOO_BIG` stays startable on purpose:
        sd.cpp cuts the graph further when a segment will not fit, so "needs
        about 9 GB of VRAM" is a statement about comfort, not possibility, and
        a user who wants to find out where their machine gives up is entitled
        to. The warning is in `summary`; the decision is theirs.
        """
        return self.grade != NO_DISK


# ------------------------------------------------------------------ probing
def _vulkan_vram() -> tuple[float, float, str, bool]:
    """(free_gb, total_gb, device_name, fp16) from vulkaninfo.

    Reads the first `DEVICE_LOCAL` heap. `budget` is Vulkan's own answer to
    "how much would you give me right now", which already accounts for the
    desktop's own allocations, so it needs no correction.

    `shaderFloat16` is read here as well as from the engine, because it doubles
    the working set and therefore changes every verdict — and on a fresh install
    there is no engine yet to ask. Vulkan and ggml agree on this machine:
    `shaderFloat16 = false` against the engine's `fp16: 0`.
    """
    try:
        r = subprocess.run(["vulkaninfo"], capture_output=True, text=True,
                           timeout=20, env=_probe_env())
    except (OSError, subprocess.SubprocessError):
        return 0.0, 0.0, "", True
    text = r.stdout or ""
    name = ""
    m = re.search(r"deviceName\s*=\s*(.+)", text)
    if m:
        name = m.group(1).strip()
    m = re.search(r"\bshaderFloat16\s*=\s*(true|false)", text)
    fp16 = m.group(1) == "true" if m else True

    # Walk the heaps and take the first flagged DEVICE_LOCAL. Heap 0 is not
    # reliably the device-local one on an integrated GPU.
    best = (0.0, 0.0)
    for block in re.split(r"memoryHeaps\[\d+\]:", text)[1:]:
        head = block[:600]
        if "MEMORY_HEAP_DEVICE_LOCAL_BIT" not in head:
            continue
        size = re.search(r"size\s*=\s*(\d+)", head)
        budget = re.search(r"budget\s*=\s*(\d+)", head)
        if size:
            total = int(size.group(1)) / 1e9
            free = int(budget.group(1)) / 1e9 if budget else total
            if total > best[1]:
                best = (free, total)
        if best[1]:
            break
    return best[0], best[1], name, fp16


def _nvidia_vram() -> tuple[float, float, str]:
    """(free_gb, total_gb, name) from nvidia-smi. Fallback for no vulkan-tools."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return 0.0, 0.0, ""
    line = (r.stdout or "").strip().splitlines()
    if not line:
        return 0.0, 0.0, ""
    parts = [p.strip() for p in line[0].split(",")]
    try:
        total = float(parts[1]) * _GIB / 1024 / 1e9 * 1024      # MiB -> GB
        used = float(parts[2]) * _GIB / 1024 / 1e9 * 1024
    except (IndexError, ValueError):
        return 0.0, 0.0, ""
    return max(0.0, total - used), total, parts[0]


def _engine_fp16(list_devices_output: str) -> bool | None:
    """Whether the GPU backend reports usable fp16.

    `sd-cli --list-devices` prints a capability line per device:

        ggml_vulkan: 0 = NVIDIA GeForce GTX 1050 Ti (NVIDIA) | uma: 0 |
        fp16: 0 | bf16: 0 | ...

    Pascal reports `fp16: 0`, and that single digit explains a great deal: every
    activation is then fp32, so the working set is about double what the same
    model needs on a card one generation newer.
    """
    m = re.search(r"\bfp16:\s*(\d)", list_devices_output or "")
    return None if m is None else m.group(1) == "1"


def _meminfo() -> tuple[float, float]:
    """(available_gb, total_gb) from /proc/meminfo."""
    available = total = 0.0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            if key == "MemAvailable":
                available = int(rest.split()[0]) * 1024 / 1e9
            elif key == "MemTotal":
                total = int(rest.split()[0]) * 1024 / 1e9
    except (OSError, ValueError, IndexError):
        pass
    return available, total


def _swap_is_zram() -> bool:
    """True when every active swap device is compressed RAM.

    See the module docstring: this is the difference between "you have 12 GB of
    headroom" and "you have none".
    """
    try:
        lines = Path("/proc/swaps").read_text().splitlines()[1:]
    except OSError:
        return False
    devices = [ln.split()[0] for ln in lines if ln.strip()]
    return bool(devices) and all("zram" in d for d in devices)


def _probe_env() -> dict[str, str]:
    env = dict(os.environ)
    env["MANGOHUD"] = "0"
    env["DISABLE_VKBASALT"] = "1"
    return env


def _free_disk(path: Path) -> float:
    """Free space on the volume `path` is (or would be) on, in GB.

    Walks up to the nearest parent that exists rather than creating the
    directory. Probing is a read, and it runs whenever the settings change — so
    an earlier version that called `mkdir` left a stray directory behind for
    every path the user typed while editing one.
    """
    candidate = path.expanduser()
    for _ in range(64):
        try:
            return shutil.disk_usage(candidate).free / 1e9
        except OSError:
            pass
        if candidate.parent == candidate:
            return 0.0
        candidate = candidate.parent
    return 0.0


def probe(models_dir: Path, list_devices_output: str = "") -> Hardware:
    """Measure the machine. Cheap enough to call on every settings change."""
    free, total, name, fp16 = _vulkan_vram()
    if total <= 0:
        free, total, name = _nvidia_vram()
    engine_fp16 = _engine_fp16(list_devices_output)
    if engine_fp16 is not None:
        fp16 = engine_fp16

    ram, ram_total = _meminfo()
    disk = _free_disk(models_dir)

    return Hardware(vram_gb=free, vram_total_gb=total, ram_gb=ram,
                    ram_total_gb=ram_total, disk_gb=disk, gpu_name=name,
                    fp16=fp16, swap_is_zram=_swap_is_zram())


# ------------------------------------------------------------------ grading
#: Exponent relating the working set to pixel count. **Measured**: sd.cpp
#: reports its own peak compute buffer, and for FLUX.2 Klein 4B on this machine
#: it is 491 MB at 512x512 and 2763 MB at 1024x1024 — four times the pixels for
#: 5.6 times the memory, an exponent of 1.25.
#:
#: This one *is* the quadratic attention term showing through, and it is the
#: reason a model that fits comfortably at 512 can fail to allocate at 1024 on
#: the same card. Treating it as linear, as the first draft did, understated the
#: 1024 working set by more than a gigabyte — on a card with 3.4 GiB free, that
#: is the difference between a correct verdict and a run that spills.
WORK_EXPONENT = 1.25


def working_gb(recipe: Recipe, width: int, height: int, hw: Hardware) -> float:
    """VRAM the run needs beyond the weights, at this output size."""
    megapixels = max(0.05, (width * height) / 1e6)
    return (recipe.working_gb * megapixels ** WORK_EXPONENT
            * hw.working_multiplier())


def verdict(recipe: Recipe, hw: Hardware, width: int, height: int,
            pending_bytes: int = 0) -> Verdict:
    """Grade one recipe against one machine at one output size."""
    need_work = working_gb(recipe, width, height, hw)
    need_weights = recipe.weights_gb()
    download = pending_bytes / 1e9

    def made(grade: str, summary: str, flags: tuple[str, ...] = ()) -> Verdict:
        return Verdict(recipe.id, grade, need_work, download, summary, flags)

    if not hw.known:
        return made(UNKNOWN, "hardware unknown")

    if download > 0 and hw.disk_gb > 0 and download > hw.disk_gb:
        return made(NO_DISK,
                    f"needs {download:.0f} GB free, {hw.disk_gb:.0f} GB left")

    vram = max(0.0, hw.vram_gb - VRAM_MARGIN_GB)
    ram = max(0.0, hw.ram_gb - RAM_MARGIN_GB)
    # Compared against the headroom-adjusted figure, because the question is
    # always "will the allocator actually manage this", not "do the reported
    # numbers sum to less than the card".
    need_live = need_work * WORK_HEADROOM

    if need_weights + need_live <= vram:
        return made(FITS, "fits your GPU")

    if need_live > vram:
        # The activations alone will not fit in the VRAM available. Offloading
        # cannot help — it moves *weights*, and activations must be resident —
        # so sd.cpp will cut the graph finer and try anyway. Still `runnable`:
        # a user who wants to find where their machine gives up is entitled to,
        # and the warning is in the summary.
        #
        # *Why* it does not fit decides the wording, because the two causes
        # have completely different fixes. A desktop compositor always holds a
        # few hundred megabytes and nobody can close it, so that is the card
        # being too small. Something holding most of the card is another
        # application, and closing it would genuinely solve this.
        crowded = (hw.vram_total_gb > 0
                   and hw.busy_gb / hw.vram_total_gb > BUSY_SHARE)
        return made(
            BUSY if crowded else TOO_BIG,
            (f"{hw.busy_gb:.1f} GB of your GPU is in use elsewhere" if crowded
             else f"needs about {need_live:.0f} GB of VRAM"),
            ("--params-backend", "diffusion=disk,te=cpu,vae=cpu",
             "--vae-tiling"))

    if need_weights <= ram:
        return made(OFFLOAD, "offloads to RAM", ("--offload-to-cpu",))
    return made(STREAM, "streams from disk",
                ("--params-backend", "diffusion=disk,te=cpu,vae=cpu",
                 "--vae-tiling"))


def speed_penalty(grade: str) -> float:
    return SPEED_PENALTY.get(grade, 1.0)
