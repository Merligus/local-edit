"""Spacing, typography and animation timing, derived the way Plasma derives them.

Using Kirigami's own formulas (Kirigami 6.28 `Units`) rather than hardcoded
pixels is what makes the app's density match the rest of the desktop instead of
merely looking Qt-ish. Everything scales from the font, so a HiDPI or
large-font setup gets proportionally larger spacing for free.

`AnimationDurationFactor` from kdeglobals is honoured, including the case that
breaks naive animation code: the user can set it to 0 ("Instant"), and then
every animation call site must jump straight to its final state rather than
starting a zero-length animation that never repaints.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QFileSystemWatcher, QObject, Signal
from PySide6.QtGui import QFontMetricsF, QGuiApplication

KDEGLOBALS = Path.home() / ".config" / "kdeglobals"

# Kirigami's fixed steps. Named as Kirigami names them so the mapping is obvious.
SMALL = 4
MEDIUM = 6
LARGE = 8
RADIUS = 5

#: Kirigami's four animation tiers, before the user's duration factor.
DUR_SHORT = 50
DUR_SNAP = 100
DUR_NORMAL = 200
DUR_LONG = 400

TOOLTIP_DELAY = 700


class _Metrics(QObject):
    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._grid = 18
        self._factor = 1.0
        self._watcher: QFileSystemWatcher | None = None

    # -- lifecycle --------------------------------------------------------
    def install(self, app: QGuiApplication) -> None:
        self.refresh()
        if self._watcher is None and KDEGLOBALS.exists():
            self._watcher = QFileSystemWatcher([str(KDEGLOBALS)], self)
            # Editors replace rather than modify, which drops the watch; re-add.
            self._watcher.fileChanged.connect(self._on_file_changed)

    def _on_file_changed(self, path: str) -> None:
        if self._watcher and path not in self._watcher.files() and Path(path).exists():
            self._watcher.addPath(path)
        before = (self._grid, self._factor)
        self.refresh()
        if (self._grid, self._factor) != before:
            self.changed.emit()

    def refresh(self) -> None:
        app = QGuiApplication.instance()
        if app is not None:
            # Kirigami: gridUnit is the font's line height, rounded.
            self._grid = max(8, round(QFontMetricsF(app.font()).height()))
        self._factor = self._read_factor()

    @staticmethod
    def _read_factor() -> float:
        """Read AnimationDurationFactor from kdeglobals ([KDE] section)."""
        try:
            for line in KDEGLOBALS.read_text(errors="replace").splitlines():
                if line.startswith("AnimationDurationFactor"):
                    _, _, v = line.partition("=")
                    return max(0.0, float(v.strip()))
        except (OSError, ValueError):
            pass
        return 1.0

    # -- values -----------------------------------------------------------
    @property
    def grid(self) -> int:
        """One grid unit — the base for every derived size."""
        return self._grid

    @property
    def animation_factor(self) -> float:
        return self._factor

    def gu(self, n: float) -> int:
        """`n` grid units in pixels."""
        return int(round(self._grid * n))

    def dur(self, base: int) -> int:
        """A duration scaled by the user's factor.

        Returns 0 when animations are disabled. Callers MUST check for 0 and
        apply the final state directly, or widgets stick mid-transition.
        """
        return int(round(base * self._factor))

    @property
    def animations_enabled(self) -> bool:
        return self.dur(DUR_NORMAL) > 0

    def __repr__(self) -> str:
        return (f"<metrics grid={self._grid} factor={self._factor:.4f} "
                f"dur(200)={self.dur(DUR_NORMAL)}>")


#: Process-wide singleton; `install(app)` after QApplication exists.
metrics = _Metrics()


def install(app: QGuiApplication) -> _Metrics:
    metrics.install(app)
    return metrics
