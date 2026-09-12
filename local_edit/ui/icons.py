"""Palette-aware icon loading for Qt Widgets.

The problem this solves is concrete, not hypothetical. On this machine the icon
theme is `Papirus-Light`, which carries almost none of the action icons this app
needs, so they fall through to **Breeze light**, whose SVGs contain:

    <style id="current-color-scheme">.ColorScheme-Text { color:#232629; }</style>
    <path class="ColorScheme-Text" fill="currentColor" .../>

`#232629` is near-black, and the window background here is `#1e2233`. Result:
invisible icons. Plasma's own apps avoid this via `KIconLoader`, which is KF6
C++ with no PySide6 bindings.

Qt's SVG module *does* implement a real CSS engine and *does* resolve
`currentColor` — it just resolves it against the document's own pinned `color`.
So the fix is one substitution inside the embedded stylesheet, mapping each
`ColorScheme-*` class to the matching `QPalette` role. No DOM walking.

Rules learned the hard way (carried over from the soundboard project, where
they were found):

* SVGs **without** a `current-color-scheme` block are deliberately colourful
  (Papirus folders, `media-record`'s semantic red). Recolouring them would
  flatten them to a single-colour blob, so they are used untouched.
* The cache key **must** include the palette, or a light/dark switch serves
  stale icons.
* Theme following uses an event filter on `ApplicationPaletteChange`. The
  `paletteChanged` signal is deprecated since Qt 6.0 and its shiboken exposure
  is build-dependent.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from PySide6.QtCore import QByteArray, QEvent, QObject, Qt
from PySide6.QtGui import QColor, QGuiApplication, QIcon, QImage, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer

# Matches `.ColorScheme-Foo { ... color: #abc ... }` inside the style block.
_CLASS_RE = re.compile(
    r"\.(ColorScheme-[A-Za-z]+)\s*\{[^}]*?color\s*:\s*(#[0-9a-fA-F]{3,8})",
    re.DOTALL)
_STYLE_RE = re.compile(
    r"<style[^>]*id\s*=\s*[\"']current-color-scheme[\"'][^>]*>(.*?)</style>",
    re.DOTALL | re.IGNORECASE)

# kdeglobals fallbacks for the semantic colours Breeze uses.
_POSITIVE = QColor("#27AE60")
_NEUTRAL = QColor("#F67400")
_NEGATIVE = QColor("#DA4453")

_SEARCH_DIRS = [
    Path.home() / ".local/share/icons",
    Path.home() / ".icons",
    Path("/usr/share/icons"),
    Path("/usr/local/share/icons"),
]


#: Classes that paint an icon's *foreground*, so their contrast against the
#: window matters. `ColorScheme-Background` is excluded deliberately — it is
#: meant to be low-contrast.
_FOREGROUND = frozenset({
    "ColorScheme-Text", "ColorScheme-ButtonText", "ColorScheme-Highlight",
    "ColorScheme-HighlightedText", "ColorScheme-Accent",
    "ColorScheme-PositiveText", "ColorScheme-NeutralText",
    "ColorScheme-NegativeText",
})

#: WCAG AA for graphical objects. Below this an icon reads as a smudge.
MIN_CONTRAST = 3.0


def _luminance(c: QColor) -> float:
    def ch(v: float) -> float:
        v /= 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    return 0.2126 * ch(c.red()) + 0.7152 * ch(c.green()) + 0.0722 * ch(c.blue())


def contrast_ratio(a: QColor, b: QColor) -> float:
    la, lb = _luminance(a), _luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _role_colour(cls: str, pal: QPalette, group: QPalette.ColorGroup,
                 selected: bool) -> QColor:
    R = QPalette.ColorRole
    if cls == "ColorScheme-Text":
        return pal.color(group, R.HighlightedText if selected else R.WindowText)
    if cls == "ColorScheme-ButtonText":
        return pal.color(group, R.HighlightedText if selected else R.ButtonText)
    if cls == "ColorScheme-Background":
        return pal.color(group, R.Highlight if selected else R.Window)
    if cls == "ColorScheme-Highlight":
        return pal.color(group, R.Highlight)
    if cls == "ColorScheme-HighlightedText":
        return pal.color(group, R.HighlightedText)
    if cls == "ColorScheme-Accent":
        # Accent is Qt 6.6+; it falls back to Highlight on its own.
        return pal.color(group, getattr(R, "Accent", R.Highlight))
    if cls == "ColorScheme-PositiveText":
        return _POSITIVE
    if cls == "ColorScheme-NeutralText":
        return _NEUTRAL
    if cls == "ColorScheme-NegativeText":
        return _NEGATIVE
    return pal.color(group, R.WindowText)


def _guarded_colour(cls: str, pal: QPalette, group: QPalette.ColorGroup,
                    selected: bool) -> QColor:
    """Honour Breeze's semantic class, but never at the cost of legibility.

    Breeze's icon classes assume Breeze's own colour scheme. On a custom scheme
    the faithful mapping can be *worse* than doing nothing: this user's
    `Accent`/`Highlight` is a dark grey `#4f5258`, so `dialog-information`
    (`ColorScheme-Accent`, Breeze blue) recoloured to a 2.4:1 smudge.

    So: take the semantic colour, and fall back to plain text colour when it
    would not be legible against the surface the icon sits on.
    """
    colour = _role_colour(cls, pal, group, selected)
    if cls not in _FOREGROUND:
        return colour
    R = QPalette.ColorRole
    surface = pal.color(group, R.Highlight if selected else R.Window)
    if contrast_ratio(colour, surface) >= MIN_CONTRAST:
        return colour
    fallback = pal.color(group, R.HighlightedText if selected else R.WindowText)
    if contrast_ratio(fallback, surface) > contrast_ratio(colour, surface):
        return fallback
    return colour


def recolour_svg(text: str, pal: QPalette, group: QPalette.ColorGroup,
                 selected: bool = False) -> str:
    """Repoint an icon's pinned colours at the live palette.

    Returns the text unchanged when there is no `current-color-scheme` block,
    which is how deliberately-colourful icons survive intact.
    """
    m = _STYLE_RE.search(text)
    if not m:
        return text
    block = m.group(0)

    def sub(mm: re.Match) -> str:
        colour = _guarded_colour(mm.group(1), pal, group, selected)
        return mm.group(0).replace(mm.group(2), colour.name())

    return text.replace(block, _CLASS_RE.sub(sub, block))


def is_themed(text: str) -> bool:
    return _STYLE_RE.search(text) is not None


# ----------------------------------------------------------- theme lookup
def _theme_dirs(theme: str) -> list[Path]:
    return [d / theme for d in _SEARCH_DIRS if (d / theme).is_dir()]


def _inherits(theme: str) -> list[str]:
    for d in _theme_dirs(theme):
        idx = d / "index.theme"
        if not idx.is_file():
            continue
        try:
            for line in idx.read_text(errors="replace").splitlines():
                if line.startswith("Inherits"):
                    _, _, v = line.partition("=")
                    return [t.strip() for t in v.split(",") if t.strip()]
        except OSError:
            pass
    return []


def theme_chain(start: str | None = None) -> list[str]:
    """BFS over `Inherits`, always ending with the Breeze/hicolor fallbacks."""
    start = start or QIcon.themeName() or "breeze"
    seen, order, queue = set(), [], [start]
    while queue:
        t = queue.pop(0)
        if not t or t in seen:
            continue
        seen.add(t)
        order.append(t)
        queue.extend(_inherits(t))
    for extra in ("breeze", "breeze-dark", "hicolor"):
        if extra not in seen:
            order.append(extra)
            seen.add(extra)
    return order


_index_cache: dict[str, dict[str, list[tuple[int, Path]]]] = {}
_SIZE_RE = re.compile(r"(\d+)")


def _index_theme(theme: str) -> dict[str, list[tuple[int, Path]]]:
    """name -> [(size, path)], built by walking the theme once.

    Themes disagree on layout (`breeze/actions/22/x.svg` vs
    `Papirus-Light/22x22/actions/x.svg`), so the size is taken from whichever
    path component looks like one rather than assuming a position. Scalable
    directories get a large sentinel so they win when nothing matches exactly.
    """
    if theme in _index_cache:
        return _index_cache[theme]
    out: dict[str, list[tuple[int, Path]]] = {}
    for root in _theme_dirs(theme):
        for dirpath, _dirnames, filenames in os.walk(root):
            rel = Path(dirpath).relative_to(root).parts
            size = 0
            for part in rel:
                if part == "scalable":
                    size = 4096
                    break
                m = _SIZE_RE.match(part)
                if m:
                    size = int(m.group(1))
                    break
            for fn in filenames:
                if not fn.endswith((".svg", ".png")):
                    continue
                out.setdefault(fn.rsplit(".", 1)[0], []).append((size, Path(dirpath) / fn))
    _index_cache[theme] = out
    return out


def find_icon_file(name: str, size: int = 22,
                   chain: list[str] | None = None) -> Path | None:
    """Resolve `name` through the theme chain, preferring SVG at a close size.

    Deliberately a name lookup across the chain, never a glob at a fixed path:
    `dialog-close` exists only at 24, the `dialog-*` status icons live under
    `status/` rather than `actions/`, and sizes differ per theme.
    """
    for theme in (chain if chain is not None else theme_chain()):
        entries = _index_theme(theme).get(name)
        if not entries:
            continue
        # Prefer SVG, then the closest size at or above the request.
        def rank(e: tuple[int, Path]) -> tuple[int, int, int]:
            sz, path = e
            svg = 0 if path.suffix == ".svg" else 1
            return (svg, 0 if sz >= size else 1, abs(sz - size))
        return min(entries, key=rank)[1]
    return None


# ------------------------------------------------------------------ loader
class _Icons(QObject):
    """Loads, recolours and caches icons; refreshes them on a theme change."""

    def __init__(self) -> None:
        super().__init__()
        self._cache: dict[tuple, QIcon] = {}
        self._tracked: list[tuple] = []   # (weak-ish target, setter, name)
        self._installed = False

    # -- lifecycle --------------------------------------------------------
    def install(self, app: QGuiApplication) -> None:
        if not self._installed:
            app.installEventFilter(self)
            self._installed = True

    def eventFilter(self, obj: QObject, ev: QEvent) -> bool:
        if ev.type() == QEvent.Type.ApplicationPaletteChange:
            self.invalidate()
        return super().eventFilter(obj, ev)

    def invalidate(self) -> None:
        """Drop cached pixmaps and re-apply icons to everything registered."""
        self._cache.clear()
        _index_cache.clear()
        live = []
        for target, setter, name in self._tracked:
            try:
                setter(self.icon(name))
                live.append((target, setter, name))
            except RuntimeError:
                pass          # the C++ object is gone
        self._tracked = live

    def themed(self, target: QObject, name: str, setter=None) -> QIcon:
        """Set an icon now and keep it correct across theme changes."""
        setter = setter or getattr(target, "setIcon", None)
        ic = self.icon(name)
        if setter is not None:
            setter(ic)
            self._tracked.append((target, setter, name))
        return ic

    # -- rendering --------------------------------------------------------
    def icon(self, name: str, size: int = 22) -> QIcon:
        app = QGuiApplication.instance()
        pal = app.palette() if app else QPalette()
        key = (name, size, pal.cacheKey())      # palette MUST be in the key
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        path = find_icon_file(name, size)
        if path is None:
            ic = QIcon.fromTheme(name)          # last resort: let Qt try
            self._cache[key] = ic
            return ic

        if path.suffix != ".svg":
            ic = QIcon(str(path))
            self._cache[key] = ic
            return ic

        try:
            text = path.read_text(errors="replace")
        except OSError:
            ic = QIcon()
            self._cache[key] = ic
            return ic

        ic = QIcon()
        if not is_themed(text):
            # Colourful by design — never recolour, but still cache.
            ic.addPixmap(self._render(text, size))
        else:
            G = QPalette.ColorGroup
            M = QIcon.Mode
            for group, mode, selected in ((G.Active, M.Normal, False),
                                          (G.Disabled, M.Disabled, False),
                                          (G.Active, M.Selected, True)):
                pix = self._render(recolour_svg(text, pal, group, selected), size)
                ic.addPixmap(pix, mode, QIcon.State.Off)
        self._cache[key] = ic
        return ic

    @staticmethod
    def _render(svg_text: str, size: int) -> QPixmap:
        app = QGuiApplication.instance()
        dpr = app.devicePixelRatio() if app else 1.0
        renderer = QSvgRenderer(QByteArray(svg_text.encode()))
        px = max(1, int(round(size * dpr)))
        img = QImage(px, px, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(Qt.GlobalColor.transparent)
        from PySide6.QtGui import QPainter
        p = QPainter(img)
        renderer.render(p)
        p.end()
        img.setDevicePixelRatio(dpr)
        return QPixmap.fromImage(img)


icons = _Icons()


def install(app: QGuiApplication) -> _Icons:
    icons.install(app)
    return icons


def icon(name: str, size: int = 22) -> QIcon:
    return icons.icon(name, size)


def themed(target, name: str, setter=None):
    """Module-level `icons.themed`, so call sites read as one dotted name.

    The singleton's method needs `icons.icons.themed(...)` from a
    `from . import icons` import, which reads like a typo.
    """
    return icons.themed(target, name, setter)
