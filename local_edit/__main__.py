"""Entry point: the GUI, and the handful of things worth doing without it.

Every heavy import is deferred into the subcommand that needs it, following
local-upscaler: `--help`, `--list-models` and `--fetch-engine` must not pay for
PySide6, and nothing here should import Qt on a machine with no display.

The Qt bootstrap order at the bottom is load-bearing, and each line earns its
place — see the comments there.
"""

from __future__ import annotations

import sys
from pathlib import Path

USAGE = """\
local-edit — edit images with a prompt and reference pictures, locally.

Usage:
  local-edit [IMAGE]                 open the editor, optionally with an image
  local-edit --list-models           what is available, and how it would run here
  local-edit --fetch-engine [BACKEND]
                                     download the engine (vulkan, cpu, rocm)
  local-edit --fetch-models ID...    download a model; 'all' for everything
  local-edit --bench [ID...]         time a real run and record the rate
  local-edit --devices               what the engine can compute on
  local-edit --refresh-sizes         re-read every weight file's size from
                                     HuggingFace and print catalog literals
  local-edit --install               add to the application menu
  local-edit --uninstall             remove it again
  local-edit --help                  this
"""


def say(*args: object) -> None:
    print(*args, flush=True)


def _models_dir() -> Path:
    from . import settings as st
    return st.load().models_path()


def _bar(done: int, total: int, label: str) -> None:
    """One rewriting line, so a 34 GB download does not scroll the terminal."""
    if total <= 0:
        return
    width = 32
    filled = int(width * done / total)
    sys.stderr.write(
        f"\r  |{'=' * filled}{' ' * (width - filled)}| "
        f"{done / 1e9:5.2f} / {total / 1e9:5.2f} GB  {label[:44]:<44}")
    sys.stderr.flush()
    if done >= total:
        sys.stderr.write("\n")


# ------------------------------------------------------------------ commands
def list_models() -> int:
    from .engine import binary, catalog, fetch, hardware, runner
    from .ui.text import human_bytes, human_time

    models = _models_dir()
    hw = hardware.probe(models, binary.list_devices())
    say(f"GPU  {hw.gpu_name or 'unknown'} — {hw.vram_gb:.1f} of "
        f"{hw.vram_total_gb:.1f} GB free"
        + ("" if hw.fp16 else ", no fp16 (activations run at full precision)"))
    say(f"RAM  {hw.ram_gb:.1f} of {hw.ram_total_gb:.1f} GB available"
        + ("  ·  swap is zram, which does not help with weights"
           if hw.swap_is_zram else ""))
    say(f"Disk {hw.disk_gb:.0f} GB free in {models}")
    say()
    # Every column is fixed width and every value is truncated to it. The
    # verdict summary is the one that varies most — "fits your GPU" against
    # "4.1 GB of your GPU is in use elsewhere" — and letting it run long pushes
    # the estimate out of its column on exactly the rows where the estimate
    # matters most.
    verdict_width = 30
    say(f"{'ID':<22} {'SIZE':>7} {'STATE':<14} "
        f"{'ON THIS MACHINE':<{verdict_width}} EST @768")
    for recipe in catalog.RECIPES:
        pending = fetch.download_size(recipe, models)
        v = hardware.verdict(recipe, hw, 768, 768, pending)
        state = "ready" if not pending else f"{human_bytes(pending)} to get"
        seconds = runner.estimate_seconds(recipe, 768, 768, recipe.steps,
                                          grade=v.grade, references=1)
        summary = v.summary if len(v.summary) <= verdict_width else \
            v.summary[:verdict_width - 1] + "\u2026"
        say(f"{recipe.id:<22} {recipe.weights_gb():>6.1f}G {state:<14} "
            f"{summary:<{verdict_width}} {human_time(seconds)}")
    say()
    say(f"default: {catalog.DEFAULT_RECIPE_ID}")
    return 0


def devices() -> int:
    from . import settings as st
    from .engine import binary
    # Must honour the settings override, or this reports on a different engine
    # than every other command uses — which it did, cheerfully printing the
    # Vulkan device list for a configuration pointed at a CUDA build.
    out = binary.list_devices(st.load().engine_path or None)
    if not out:
        say("No engine found. Fetch one with:  local-edit --fetch-engine")
        return 1
    say(out.strip())
    return 0


def fetch_engine(backend: str) -> int:
    from .engine import binary, fetch
    try:
        exe = fetch.fetch_engine(backend, progress=_bar)
    except fetch.FetchError as e:
        say(f"error: {e}")
        return 1
    say(f"Installed to {exe.parent}")
    if binary.probe(exe) is None:
        say("warning: the binary ran but does not report the flags this app "
            "needs. It may be too old.")
        return 1
    say("Verified. Devices it can use:")
    say(binary.list_devices().strip())
    return 0


def fetch_models(ids: list[str]) -> int:
    from .engine import catalog, fetch
    models = _models_dir()
    wanted = (catalog.RECIPES if "all" in ids
              else [catalog.get(i) for i in ids if catalog.exists(i)])
    unknown = [i for i in ids if i != "all" and not catalog.exists(i)]
    for i in unknown:
        say(f"warning: no model called {i!r}")
    if not wanted:
        say("nothing to do")
        return 1 if unknown else 0

    total = sum(fetch.download_size(r, models) for r in wanted)
    say(f"{len(wanted)} model(s), {total / 1e9:.1f} GB to download into {models}")
    for recipe in wanted:
        if fetch.have(recipe, models):
            say(f"  {recipe.id}: already complete")
            continue
        say(f"  {recipe.id}:")
        try:
            fetch.fetch_recipe(recipe, models, progress=_bar)
        except fetch.FetchError as e:
            say(f"error: {e}")
            return 1
    say("done")
    return 0


def bench(ids: list[str]) -> int:
    """Time a real run and store the rate, so estimates stop being guesses.

    Deliberately a *real* generation rather than a synthetic loop: the thing
    worth measuring is what the user will wait for, including the text-encoding
    pass and the VAE decode, on weights placed wherever this machine puts them.
    """
    import time

    from . import settings as st
    from .engine import binary, catalog, fetch, hardware, runner, server
    from .ui.text import human_time

    settings = st.load()
    models = settings.models_path()
    hw = hardware.probe(models, binary.list_devices())
    recipes = ([catalog.get(i) for i in ids if catalog.exists(i)] if ids
               else [catalog.get(settings.recipe_id)])
    if not recipes:
        say("no such model")
        return 1

    width = height = 512
    engine = server.EngineServer(idle_timeout=0)
    failed = 0
    try:
        for recipe in recipes:
            if not fetch.have(recipe, models):
                say(f"{recipe.id}: not downloaded — "
                    f"local-edit --fetch-models {recipe.id}")
                failed += 1
                continue
            v = hardware.verdict(recipe, hw, width, height)
            job = runner.Job(
                recipe=recipe, instruction="a photograph of a red bicycle "
                                           "against a white wall",
                width=width, height=height, seed=1, models_dir=models,
                memory=settings.memory_flags(v),
                backend=settings.engine_backend(),
                threads=settings.threads, max_vram_gb=settings.max_vram_gb,
                engine_path=settings.engine_path or None)
            say(f"{recipe.id}: {recipe.steps} steps at {width}x{height}, "
                f"{v.summary}…")
            started = time.monotonic()
            try:
                result = runner.Runner(
                    job, engine,
                    on_stage=lambda _k, text: say(f"  {text}")).run()
            except (runner.EditError, server.ServerError) as e:
                say(f"  failed: {e}")
                failed += 1
                continue
            elapsed = time.monotonic() - started
            say(f"  {result.sec_per_step:.2f} s/step  "
                f"({human_time(elapsed)} total, seed {result.seed})")
            settings.calibration.record(
                recipe_id=recipe.id, width=width, height=height,
                sec_per_step=result.sec_per_step, grade=v.grade,
                backend=settings.engine_backend())
            st.save(settings)
    finally:
        engine.stop()
    return 1 if failed else 0


def refresh_sizes() -> int:
    """Re-read every weight file's length from HuggingFace.

    A wrong byte count makes a model permanently un-downloadable — `fetch`
    refuses the file and no offline test can catch it — so the counts in
    `catalog` are generated here and pasted, never typed.
    """
    import json
    import urllib.error
    import urllib.request
    from functools import lru_cache

    from .engine import catalog

    @lru_cache(maxsize=None)
    def listing(repo: str) -> dict:
        url = f"https://huggingface.co/api/models/{repo}?blobs=true"
        with urllib.request.urlopen(url, timeout=60) as r:
            data = json.load(r)
        return {f["rfilename"]: f.get("size") or 0
                for f in data.get("siblings", [])}

    bad = 0
    for f in catalog.all_files():
        try:
            actual = listing(f.repo).get(f.path)
        except (urllib.error.URLError, OSError, ValueError) as e:
            say(f"  ?? {f.repo}::{f.path}  ({e})")
            bad += 1
            continue
        if actual is None:
            say(f"  !! MISSING  {f.repo}::{f.path}")
            bad += 1
        elif actual != f.size:
            say(f"  !! SIZE     {f.repo}::{f.path}")
            say(f"       catalog {f.size:>14,}   actual {actual:>14,}")
            bad += 1
        else:
            say(f"  ok {f.size:>14,}  {f.filename()}")
    say()
    say(f"{bad} problem(s) in {len(catalog.all_files())} files")
    return 1 if bad else 0


# ---------------------------------------------------------------------- gui
def gui(image: Path | None) -> int:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    # `PassThrough` before QApplication: rounding the device pixel ratio makes
    # `metrics.gu` disagree with what is actually painted on a fractional-scale
    # display, and the policy cannot be changed afterwards.
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    # The Wayland app_id, and how the compositor matches the window to its
    # .desktop entry for the icon and the task switcher. Must precede any window.
    app.setDesktopFileName("local-edit")
    app.setApplicationName("Local Edit")

    # Not optional on this machine: the system locale is pt_BR, and without
    # this Qt's standard dialog buttons come out in Portuguese inside an
    # otherwise English interface.
    from PySide6.QtCore import QLocale
    QLocale.setDefault(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))
    QIcon.setFallbackThemeName("breeze")

    from .ui import icons, metrics
    icons.install(app)
    metrics.install(app)

    # `setStyle` is deliberately never called — plasma-integration already picks
    # Breeze, the kdeglobals palette, the icon theme and the font.
    from .ui.main_window import MainWindow
    window = MainWindow()
    if image is not None:
        window.open_path(image)
    window.show()
    return app.exec()


# -------------------------------------------------------------------- router
def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return gui(None)

    head, rest = args[0], args[1:]
    if head in ("-h", "--help"):
        say(USAGE)
        return 0
    if head == "--list-models":
        return list_models()
    if head == "--devices":
        return devices()
    if head == "--fetch-engine":
        from .engine import binary
        return fetch_engine(rest[0] if rest else binary.DEFAULT_BACKEND)
    if head == "--fetch-models":
        if not rest:
            say("usage: local-edit --fetch-models ID... | all")
            return 1
        return fetch_models(rest)
    if head == "--bench":
        return bench(rest)
    if head == "--refresh-sizes":
        return refresh_sizes()
    if head in ("--install", "--uninstall"):
        from . import install
        return (install.install() if head == "--install" else install.uninstall())
    if head.startswith("-"):
        say(f"unknown option {head!r}\n")
        say(USAGE)
        return 2

    path = Path(head).expanduser()
    if not path.is_file():
        say(f"no such file: {path}")
        return 1
    return gui(path)


if __name__ == "__main__":
    sys.exit(main())
