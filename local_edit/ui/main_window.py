"""The window, the three-screen flow, and the engine process that outlives it.

    Setup  --Generate-->  Progress  --done-->  Result
      ^                      |                   |
      +----------------------+-------------------+
          cancel / failure      Edit Another
                                Continue Editing This  -->  Setup, result as source

A `QStackedWidget` rather than dialogs, because the flow is linear and a modal
progress dialog over an empty window is just a worse version of this.

**This window owns the engine process.** `EngineServer` is deliberately not
owned by a run: the whole reason it exists is to outlive one, holding several
gigabytes of weights so that editing a prompt and pressing Generate again costs
seconds of setup rather than minutes. A `Runner` per edit that started its own
server would give all of that back.

The other side of holding it is that it holds the machine's VRAM — 3.4 GiB in
total here — so an idle engine is shut down on a timer. An image editor that
makes every other GPU application stutter for as long as it is open is not one
anybody keeps open.

Two conversions deserve a note, since both are places a large image could be
quietly duplicated:

* The result arrives from the engine as a PIL image and has to become a
  `QImage`. `QImage` requires its buffer to outlive it and does not take
  ownership, so the bytes are kept alive alongside it. Dropping them gives a
  window full of garbage or a crash, depending on the allocator.
* The *source* is loaded straight from its file with `QImage` rather than
  converted from a PIL copy, so only one full-size copy of each is ever alive.

Qt conventions follow local-upscaler's: no `setStyleSheet`, no `setStyle`, every
size derived from `ui.metrics`, icons through `ui.icons`.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from PySide6.QtCore import QThread, QTimer
from PySide6.QtGui import QGuiApplication, QImage
from PySide6.QtWidgets import QMainWindow, QMessageBox, QStackedWidget

from .. import settings as st
from ..engine import binary, hardware, runner, server
from . import metrics as me
from .progress_page import ProgressPage
from .result_page import ResultPage
from .setup_page import SetupPage
from .worker import EditWorker

PAGE_SETUP, PAGE_PROGRESS, PAGE_RESULT = 0, 1, 2

#: How often to re-probe free VRAM and retire an idle engine.
_HOUSEKEEPING_MS = 20_000


def pil_to_qimage(image: Image.Image) -> tuple[QImage, bytes]:
    """Convert without an intermediate copy, returning the buffer to keep alive.

    The caller **must** hold the returned bytes for as long as the QImage is
    used: `QImage` wraps the pointer it is given and does not copy it.
    """
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA" if "A" in image.mode else "RGB")
    fmt = (QImage.Format.Format_RGBA8888 if image.mode == "RGBA"
           else QImage.Format.Format_RGB888)
    data = image.tobytes()
    qimage = QImage(data, image.width, image.height,
                    image.width * len(image.mode), fmt)
    return qimage, data


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Local Edit")
        self._settings = st.load()
        self._engine = server.EngineServer(
            idle_timeout=float(self._settings.idle_timeout))
        self._thread: QThread | None = None
        self._worker: EditWorker | None = None
        #: The job currently running. The setup page keeps taking input while a
        #: run is in flight, so its widgets are NOT a record of what was
        #: started — rebuilding the job from them on completion would attribute
        #: the result to whatever model happens to be selected when it lands.
        self._job: runner.Job | None = None
        #: Keeps QImage buffers alive. See `pil_to_qimage`.
        self._buffers: list[bytes] = []

        self._stack = QStackedWidget()
        self._setup = SetupPage(self._settings)
        self._progress = ProgressPage()
        self._result = ResultPage(self._settings)
        for page in (self._setup, self._progress, self._result):
            self._stack.addWidget(page)
        self.setCentralWidget(self._stack)

        self._setup.generate_requested.connect(self._start)
        self._progress.cancel_requested.connect(self._cancel)
        self._result.back_requested.connect(self._back_to_setup)
        self._result.continue_requested.connect(self._continue_from)

        gu = me.metrics.gu
        self.resize(gu(42), gu(40))
        self.setMinimumSize(gu(26), gu(24))
        self._centre()

        self._housekeeping = QTimer(self)
        self._housekeeping.setInterval(_HOUSEKEEPING_MS)
        self._housekeeping.timeout.connect(self._housekeep)
        self._housekeeping.start()
        # The first probe runs `vulkaninfo` and `sd-cli --list-devices`, which
        # together take a couple of seconds. Deferred past the first paint so
        # the window appears immediately rather than after them.
        QTimer.singleShot(0, self._probe_hardware)

    def _centre(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        geometry = self.frameGeometry()
        geometry.moveCenter(screen.availableGeometry().center())
        self.move(geometry.topLeft())

    # -- hardware ---------------------------------------------------------
    def _probe_hardware(self, with_engine: bool = True) -> None:
        devices = (binary.list_devices(self._settings.engine_path or None)
                   if with_engine else "")
        self._setup.set_hardware(
            hardware.probe(self._settings.models_path(), devices))

    def _housekeep(self) -> None:
        """Retire an idle engine, and keep the free-VRAM figure current."""
        if self._thread is not None:
            return                              # a run is using both
        if self._engine.stop_if_idle():
            self._probe_hardware(with_engine=False)
        elif self._stack.currentIndex() == PAGE_SETUP:
            # Cheap: skips `--list-devices`, which would spawn a process every
            # 20 seconds to re-learn something that cannot change.
            self._probe_hardware(with_engine=False)

    # -- running ----------------------------------------------------------
    def _start(self) -> None:
        if self._thread is not None:
            return
        if binary.find_server(self._settings.engine_path or None) is None:
            QMessageBox.warning(
                self, "No engine",
                "The image engine is not installed.\n\n"
                "Fetch a copy with:\n"
                "    python3 -m local_edit --fetch-engine")
            return

        job = self._setup.build_job()
        self._job = job
        st.save(self._settings)

        verdict = self._setup.verdict()
        rate = self._settings.calibration.get(job.recipe.id, job.width,
                                              job.height, verdict.grade)
        prior = runner.estimate_seconds(
            job.recipe, job.width, job.height, job.effective_steps(), rate,
            verdict.grade, references=len(job.all_references()))
        refs = len(job.all_references())
        self._progress.begin(
            prior,
            f"{job.recipe.label}  ·  {job.width} x {job.height}"
            + (f"  ·  {refs} reference{'s' if refs != 1 else ''}" if refs else ""))
        self._stack.setCurrentIndex(PAGE_PROGRESS)

        self._thread = QThread(self)
        self._worker = EditWorker(job, self._engine)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.stage.connect(self._progress.on_stage)
        self._worker.progress.connect(self._progress.on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._thread.start()

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    def _teardown(self) -> None:
        """Stop the thread and drop both objects. Safe to call more than once."""
        self._progress.end()
        thread, self._thread = self._thread, None
        worker, self._worker = self._worker, None
        self._job = None
        if thread is not None:
            thread.quit()
            thread.wait(5000)
            thread.deleteLater()
        if worker is not None:
            worker.deleteLater()

    # -- outcomes ---------------------------------------------------------
    def _on_finished(self, result: runner.Result) -> None:
        job = self._job
        self._teardown()
        if job is None:                     # cancelled and torn down under us
            return

        verdict = hardware.verdict(job.recipe, self._setup.hardware(),
                                   job.width, job.height)
        # Keyword arguments on purpose. local-upscaler has a live bug at
        # main_window.py:169 where a float is passed positionally into a `device`
        # parameter, so every CPU run is recorded under the GPU key and its
        # calibration is never read back. Positional arguments into a function
        # whose tail is optional is exactly how that happens.
        self._settings.calibration.record(
            recipe_id=job.recipe.id, width=job.width, height=job.height,
            sec_per_step=result.sec_per_step, grade=verdict.grade)
        st.save(self._settings)

        edited, buffer = pil_to_qimage(result.image)
        source = QImage(str(job.source)) if job.source is not None else QImage()
        # Release the previous run's buffers only now that the new ones exist,
        # so switching never shows a blank canvas.
        self._buffers = [buffer]
        self._result.show_result(
            source=job.source, original=source, result=edited,
            recipe_label=job.recipe.label, recipe_id=job.recipe.id,
            seed=result.seed, steps=result.steps, elapsed=result.elapsed,
            prompt_text=result.prompt, notes=result.notes)
        self._stack.setCurrentIndex(PAGE_RESULT)
        self._setup.refresh()

    def _on_failed(self, message: str) -> None:
        self._teardown()
        self._stack.setCurrentIndex(PAGE_SETUP)
        self._probe_hardware(with_engine=False)
        QMessageBox.warning(self, "The edit failed", message)

    def _on_cancelled(self) -> None:
        self._teardown()
        self._stack.setCurrentIndex(PAGE_SETUP)

    def _back_to_setup(self) -> None:
        self._stack.setCurrentIndex(PAGE_SETUP)
        self._setup.refresh()

    def _continue_from(self, path: Path) -> None:
        """Feed a result back in as the next source, keeping everything else."""
        if self._setup.load_image(Path(path)):
            self._stack.setCurrentIndex(PAGE_SETUP)

    # -- window -----------------------------------------------------------
    def open_path(self, path: Path) -> bool:
        """Load an image given on the command line."""
        return self._setup.load_image(path)

    def closeEvent(self, event) -> None:
        if self._thread is not None:
            self._cancel()
            self._teardown()
        # Not optional: the engine is a child process holding gigabytes of RAM
        # and most of the GPU. Leaving it behind would outlive the window that
        # explains it.
        self._engine.stop()
        st.save(self._settings)
        super().closeEvent(event)
