"""Desktop integration: `python3 -m local_edit --install`.

Installs the icon and a complete `.desktop` entry so the app appears in the
application menu and can be pinned, instead of being started with
`python3 -m local_edit`.

Adapted from the soundboard project, minus its portal constraints, and keeping
the two details that were learned the hard way there:

* A scalable SVG alone is not enough — several launchers and panels only look in
  the fixed-size `hicolor/<size>x<size>/apps` directories, so the icon is also
  rasterised into the eight sizes they ask for.
* `icon-theme.cache` is removed rather than refreshed when it cannot be built
  correctly. A present-but-near-empty cache is worse than none at all: loaders
  trust it and stop scanning the directory, which hides every icon in the tree,
  including other applications'.

`MimeType` is declared so the app shows up under "Open With" for images, which
is the natural way to reach an image editor.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import paths

APP_ID = "local-edit"

#: Advertised under "Open With". A .desktop MimeType line cannot be wrapped, so
#: it is assembled here rather than written out inside the template.
#:
#: Extensions do not appear here — the system mime database maps them. That is
#: why `.jfif` files already offered this app from the file manager while the
#: in-app dialog was still hiding them: shared-mime-info resolves `.jfif` to
#: image/jpeg, which was declared all along.
MIME_TYPES = (
    "image/png", "image/jpeg", "image/webp", "image/avif", "image/bmp",
    "image/tiff", "image/gif", "image/jp2", "image/x-portable-pixmap",
    "image/x-portable-graymap", "image/x-pcx", "image/vnd.microsoft.icon",
)

DESKTOP_TEMPLATE = """[Desktop Entry]
Type=Application
Version=1.0
Name=Local Edit
GenericName=AI Image Editor
Comment=Edit images with a prompt and reference pictures, entirely on this machine
Exec={exec_line} %f
TryExec={try_exec}
Icon={icon}
Terminal=false
Categories=Graphics;Photography;2DGraphics;RasterGraphics;
Keywords=edit;image;ai;diffusion;inpaint;prompt;flux;generate;reference;
MimeType={mime_types}
StartupWMClass=local-edit
X-KDE-StartupNotify=true
"""

def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def icon_target() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local/share")
    return Path(base) / "icons" / "hicolor" / "scalable" / "apps" / f"{APP_ID}.svg"


def wrapper_path() -> Path:
    return project_root() / "bin" / APP_ID


def _exec_line() -> tuple[str, str]:
    """(Exec, TryExec). Prefers the wrapper; falls back to the interpreter."""
    wrapper = wrapper_path()
    if wrapper.is_file():
        return str(wrapper), str(wrapper)
    # No wrapper: run the module with the project root on PYTHONPATH via env.
    return (f'env PYTHONPATH={project_root()} {sys.executable} -m local_edit',
            sys.executable)


#: Sizes launchers and panels actually ask for.
PNG_SIZES = (16, 22, 24, 32, 48, 64, 128, 256)


def _render_png_sizes(src_svg: Path, icon_root: Path) -> list[int]:
    """Rasterise the icon into hicolor/<size>x<size>/apps/. Best-effort."""
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QGuiApplication, QImage, QPainter
        from PySide6.QtSvg import QSvgRenderer
    except ImportError:
        return []

    owns_app = QGuiApplication.instance() is None
    if owns_app:
        # Rendering needs a GUI application; force offscreen so `--install`
        # works over SSH or from a script with no display.
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QGuiApplication([])
    made: list[int] = []
    try:
        renderer = QSvgRenderer(str(src_svg))
        if not renderer.isValid():
            return []
        for size in PNG_SIZES:
            out = icon_root / f"{size}x{size}" / "apps" / f"{APP_ID}.png"
            try:
                out.parent.mkdir(parents=True, exist_ok=True)
                img = QImage(size, size, QImage.Format.Format_ARGB32)
                img.fill(Qt.GlobalColor.transparent)
                p = QPainter(img)
                renderer.render(p)
                p.end()
                if img.save(str(out), "PNG"):
                    made.append(size)
            except OSError:
                continue
    finally:
        if owns_app:
            del app
    return made


def _fix_icon_cache(icon_root: Path, say) -> None:
    """Keep `icon-theme.cache` honest, or get rid of it.

    `gtk-update-icon-cache` needs an `index.theme` in the directory it indexes.
    `~/.local/share/icons/hicolor` usually has none (the theme is *defined* by
    the system copy and merely extended here), and running the tool anyway
    produces a tiny cache that indexes almost nothing. That is worse than having
    no cache at all: loaders trust the cache and stop scanning the filesystem,
    so every icon in the directory — ours and other applications' — disappears.

    So: refresh it only when it can be built correctly, otherwise remove it.
    """
    cache = icon_root / "icon-theme.cache"
    has_index = (icon_root / "index.theme").is_file()
    exe = shutil.which("gtk-update-icon-cache")

    if has_index and exe:
        try:
            r = subprocess.run([exe, "-qtf", str(icon_root)],
                               capture_output=True, timeout=60)
            if r.returncode == 0:
                say("refreshed  gtk-update-icon-cache")
                return
        except (subprocess.SubprocessError, OSError):
            pass

    if cache.exists():
        try:
            cache.unlink()
            say(f"removed    stale {cache.name} (it would hide icons on disk)")
        except OSError:
            pass


def install(verbose: bool = True) -> int:
    """Install the icon, the wrapper bit and the .desktop entry."""
    def say(msg: str) -> None:
        if verbose:
            print(msg)

    src_icon = project_root() / "data" / f"{APP_ID}.svg"
    if not src_icon.is_file():
        say(f"error: {src_icon} is missing")
        return 1

    # -- icon ---------------------------------------------------------------
    dest_icon = icon_target()
    try:
        dest_icon.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_icon, dest_icon)
        say(f"icon      -> {dest_icon}")
    except OSError as e:
        say(f"error: could not install the icon: {e}")
        return 1

    # -- wrapper ------------------------------------------------------------
    wrapper = wrapper_path()
    if wrapper.is_file():
        try:
            wrapper.chmod(wrapper.stat().st_mode | 0o111)
            say(f"wrapper   -> {wrapper} (executable)")
        except OSError as e:
            say(f"warning: could not mark the wrapper executable: {e}")

    # -- desktop entry ------------------------------------------------------
    exec_line, try_exec = _exec_line()
    dest_desktop = paths.desktop_file()
    try:
        dest_desktop.parent.mkdir(parents=True, exist_ok=True)
        dest_desktop.write_text(DESKTOP_TEMPLATE.format(
            exec_line=exec_line, try_exec=try_exec, icon=APP_ID,
            mime_types=";".join(MIME_TYPES) + ";"))
        dest_desktop.chmod(0o644)
        say(f"launcher  -> {dest_desktop}")
        say(f"            Exec={exec_line}")
    except OSError as e:
        say(f"error: could not write the desktop entry: {e}")
        return 1

    # -- raster sizes -------------------------------------------------------
    # A scalable SVG alone is not enough in practice: some launchers and panels
    # only look in the fixed-size directories. Cheap insurance.
    made = _render_png_sizes(src_icon, dest_icon.parent.parent.parent)
    if made:
        say(f"raster    -> {len(made)} PNG sizes ({', '.join(str(s) for s in made)})")

    # -- refresh the caches -------------------------------------------------
    icon_root = dest_icon.parent.parent.parent
    _fix_icon_cache(icon_root, say)

    for cmd in (["update-desktop-database", str(dest_desktop.parent)],
                ["kbuildsycoca6", "--noincremental"]):
        exe = shutil.which(cmd[0])
        if not exe:
            continue
        try:
            subprocess.run([exe, *cmd[1:]], capture_output=True, timeout=60)
            say(f"refreshed  {cmd[0]}")
        except (subprocess.SubprocessError, OSError):
            pass      # cosmetic: the entry still works, it may just appear later

    say("")
    say("Installed. Local Edit is in your application menu — search for it,")
    say("then right-click → Pin to Favorites. Images also offer it under")
    say("\"Open With\".")
    return 0


def uninstall(verbose: bool = True) -> int:
    """Remove the icon and desktop entry.

    Downloaded models and the fetched engine are left alone — they are hundreds
    of megabytes the user may well want to keep, and removing a menu entry is
    not a request to re-download them.
    """
    def say(msg: str) -> None:
        if verbose:
            print(msg)

    targets = [icon_target(), paths.desktop_file()]
    root = icon_target().parent.parent.parent
    targets += [root / f"{s}x{s}" / "apps" / f"{APP_ID}.png" for s in PNG_SIZES]
    for p in targets:
        try:
            if p.exists():
                p.unlink()
                say(f"removed {p}")
        except OSError as e:
            say(f"warning: could not remove {p}: {e}")
    exe = shutil.which("kbuildsycoca6")
    if exe:
        try:
            subprocess.run([exe, "--noincremental"], capture_output=True, timeout=60)
        except (subprocess.SubprocessError, OSError):
            pass
    say("Uninstalled the launcher. Downloaded models and the engine are untouched.")
    return 0


def status() -> dict[str, bool]:
    return {
        "icon": icon_target().is_file(),
        "desktop": paths.desktop_file().is_file(),
        "wrapper_executable": (wrapper_path().is_file()
                               and os.access(wrapper_path(), os.X_OK)),
    }
