"""Running one edit off the GUI thread.

`engine.runner` is deliberately Qt-free, so this is the whole of the bridge: a
`QObject` moved onto a `QThread` that calls `Runner.run()` and turns its plain
callbacks into signals.

The callbacks arrive on the worker thread. Emitting them as signals rather than
touching widgets directly is what makes that safe — a queued connection marshals
them back to the GUI thread, so the progress bar is updated by the thread that
owns it.

Cancellation goes the other way and is deliberately not a signal.
`Runner.cancel` sets a flag, POSTs to the engine's cancel endpoint and, on the
one-shot path, terminates the child — all safe to do from the GUI thread while
the worker is blocked in a poll or a read. Going through the event loop would
not work, because the worker thread is inside a blocking call and would not
process a queued slot until the run it was meant to interrupt had finished.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from ..engine import runner, server


class EditWorker(QObject):
    """Runs one `runner.Job` and reports on it."""

    #: (stage key, human text)
    stage = Signal(str, str)
    #: (done, total) within the current stage; total 0 means indeterminate.
    progress = Signal(int, int)
    #: `runner.Result`, as an object because Signal cannot type a dataclass.
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, job: runner.Job, engine: server.EngineServer) -> None:
        super().__init__()
        self._runner = runner.Runner(
            job, engine,
            on_stage=lambda key, text: self.stage.emit(key, text),
            on_progress=lambda done, total: self.progress.emit(done, total))

    @Slot()
    def run(self) -> None:
        try:
            result = self._runner.run()
        except runner.Cancelled:
            self.cancelled.emit()
        except (runner.EditError, server.ServerError) as e:
            self.failed.emit(str(e))
        except Exception as e:                          # noqa: BLE001
            # A worker thread that raises takes the whole app down with an
            # unhelpful abort, so anything unexpected is reported as a failure
            # the user can at least read.
            self.failed.emit(f"Unexpected error: {type(e).__name__}: {e}")
        else:
            self.finished.emit(result)

    def cancel(self) -> None:
        """Safe to call from the GUI thread while the run is in flight."""
        self._runner.cancel()

    @property
    def was_cancelled(self) -> bool:
        return self._runner.cancelled
