"""The second screen: what the run is doing, and how much longer.

The bar is genuinely determinate in every stage, which took some doing. The
engine's job API reports only `queued` / `generating` / `completed`, so the
numbers here come from `engine.progress` parsing the engine process's own
output — see that module for why that is the only way to get them.

Three different things are counted, and the bar says which:

* **bytes**, while a recipe downloads (3.5 to 34 GB, so this one matters);
* **tensors**, while weights load — a minute or two even on a warm server,
  because `sd-server` loads lazily on the first request;
* **steps**, while sampling, which is the part anyone actually waits for.

The remaining-time figure starts as the catalog's estimate and crosses over to
the measured step rate as steps complete, weighted so the first step — which on
this engine carries the text-encoding pass and would otherwise imply a wildly
pessimistic total — does not dominate. An estimate that visibly lurches is
worse than no estimate, so it is also clamped to rise only slowly.
"""

from __future__ import annotations

import time

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QProgressBar, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

from ..engine import runner
from . import metrics as me
from .text import human_bytes, human_time

#: Weight the measured rate gets once every step is done. Below that it is
#: blended with the prior in proportion to how much of the run has finished.
_MEASURED_WEIGHT = 0.9


class ProgressPage(QWidget):
    """A determinate progress bar, an ETA, and a way out."""

    cancel_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._started = 0.0
        self._stage_started = 0.0
        self._stage = ""
        self._done = 0
        self._total = 0
        self._prior_seconds = 0.0
        self._last_eta: float | None = None

        gu = me.metrics.gu
        outer = QVBoxLayout(self)
        outer.setContentsMargins(gu(2), gu(2), gu(2), gu(2))
        outer.setSpacing(gu(0.75))
        outer.addStretch(1)

        self._title = QLabel("Working…")
        font = self._title.font()
        font.setPointSizeF(font.pointSizeF() * 1.3)
        font.setBold(True)
        self._title.setFont(font)
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title.setWordWrap(True)
        outer.addWidget(self._title)

        self._subtitle = QLabel()
        self._subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._subtitle.setWordWrap(True)
        outer.addWidget(self._subtitle)

        self._bar = QProgressBar()
        self._bar.setSizePolicy(QSizePolicy.Policy.Expanding,
                                QSizePolicy.Policy.Fixed)
        self._bar.setMinimumWidth(gu(18))
        self._bar.setTextVisible(True)
        outer.addWidget(self._bar)

        self._detail = QLabel()
        self._detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._detail.setWordWrap(True)
        outer.addWidget(self._detail)

        row = QHBoxLayout()
        row.addStretch(1)
        self._cancel = QPushButton("Cancel")
        self._cancel.clicked.connect(self._on_cancel)
        row.addWidget(self._cancel)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(1)

        # 500 ms keeps the elapsed counter honest without repainting constantly.
        self._ticker = QTimer(self)
        self._ticker.setInterval(500)
        self._ticker.timeout.connect(self._retick)

    # -- lifecycle --------------------------------------------------------
    def begin(self, prior_seconds: float, subtitle: str) -> None:
        self._started = self._stage_started = time.monotonic()
        self._stage = ""
        self._done = self._total = 0
        self._prior_seconds = max(0.0, prior_seconds)
        self._last_eta = None
        self._cancel.setEnabled(True)
        self._cancel.setText("Cancel")
        self._title.setText("Starting…")
        self._subtitle.setText(subtitle)
        self._bar.setRange(0, 0)
        self._detail.setText("")
        self._ticker.start()

    def end(self) -> None:
        self._ticker.stop()

    # -- signals from the worker ------------------------------------------
    def on_stage(self, key: str, text: str) -> None:
        if key != self._stage:
            self._stage = key
            self._stage_started = time.monotonic()
            self._last_eta = None
            self._done = self._total = 0
            self._bar.setRange(0, 0)
        self._title.setText(text)
        self._retick()

    #: `QProgressBar` stores its range in a C `int`. Every recipe in this
    #: catalog is larger than that in bytes — the smallest is 3.5 billion — so
    #: a download bar fed raw byte counts raises OverflowError and takes the
    #: run down with it. The true figures stay in `_done`/`_total` for the
    #: detail line; only the widget sees the scaled ones.
    _BAR_MAX = 2_000_000_000

    def on_progress(self, done: int, total: int) -> None:
        self._done, self._total = done, total
        if total > 0:
            divisor = max(1, -(-total // self._BAR_MAX))    # ceil
            self._bar.setRange(0, total // divisor)
            self._bar.setValue(done // divisor)
        else:
            self._bar.setRange(0, 0)
        self._retick()

    # -- the numbers ------------------------------------------------------
    def _retick(self) -> None:
        elapsed = time.monotonic() - self._started
        parts = [f"Elapsed {human_time(elapsed)}"]

        if self._stage == runner.STAGE_DOWNLOAD and self._total:
            self._bar.setFormat("%p%")
            parts.append(f"{human_bytes(self._done)} of "
                         f"{human_bytes(self._total)}")
            rate = self._done / max(0.5, time.monotonic() - self._stage_started)
            if rate > 0:
                parts.append(f"{human_bytes(rate)}/s")
                parts.append("About " + human_time(
                    (self._total - self._done) / rate) + " left")
        elif self._stage == runner.STAGE_LOAD and self._total:
            self._bar.setFormat("%v of %m tensors")
            parts.append("Reading weights")
        elif self._stage == runner.STAGE_GENERATE and self._total:
            self._bar.setFormat("%v of %m steps")
            parts.append(f"Step {min(self._done + 1, self._total)} "
                         f"of {self._total}")
            eta = self._eta()
            if eta is not None:
                parts.append(f"About {human_time(eta)} left")
        else:
            self._bar.setFormat("%p%")

        self._detail.setText("   ·   ".join(parts))

    def _eta(self) -> float | None:
        """Seconds remaining, blending the prior with the observed step rate."""
        if not self._total:
            return None
        fraction = self._done / self._total
        prior_left = max(0.0, self._prior_seconds
                         - (time.monotonic() - self._started))
        if self._done <= 0:
            return prior_left or None

        stage_elapsed = time.monotonic() - self._stage_started
        measured_left = (stage_elapsed / self._done) * (self._total - self._done)
        # Trust the measurement more as the run progresses. Early on, the first
        # step's timing is mostly the text-encoding pass — 19 s of the 22 s a
        # bare step costs on this machine — and says little about the rest.
        weight = _MEASURED_WEIGHT * fraction
        eta = weight * measured_left + (1 - weight) * prior_left

        # Let it fall freely but rise only slowly, so one slow step does not
        # make the number jump backwards.
        if self._last_eta is not None and eta > self._last_eta:
            eta = min(eta, self._last_eta + 1.0)
        self._last_eta = eta
        return eta

    def _on_cancel(self) -> None:
        self._cancel.setEnabled(False)
        self._cancel.setText("Cancelling…")
        self._title.setText("Stopping the run…")
        self.cancel_requested.emit()
