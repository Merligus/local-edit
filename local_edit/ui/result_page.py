"""The third screen: source, result, the wipe between them, and what's next.

The three views are a segmented control over a single `ImageView`, not three
widgets, so zoom and position survive switching between them — the comparison is
only useful if "the same place" means the same place.

Two things here that local-upscaler's result page does not have, both because
editing is iterative in a way upscaling is not:

* a **filmstrip** of this session's results, because the second attempt at a
  prompt is usually worth comparing against the first, and losing the first to
  the second is how you end up regenerating it;
* **Continue editing this**, which feeds the result back in as the source and
  returns to the setup page with the prompt and references intact. Chaining
  "make it rainy" then "now at night" is the normal way to use this, and
  without it that means saving to disk and re-opening.

The filmstrip lives in memory and in `~/.cache`, and dies with the session.
Saving is still explicit, for the same reason it is in the upscaler: the result
lives in memory until asked for, and the default filename records the recipe and
seed, because after four attempts the files are otherwise indistinguishable.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QFileDialog, QHBoxLayout,
                               QLabel, QListWidget, QListWidgetItem, QMessageBox,
                               QPushButton, QToolButton, QVBoxLayout, QWidget)

from .. import settings as st
from . import metrics as me
from .image_view import MODE_COMPARE, MODE_ORIGINAL, MODE_RESULT, ImageView
from .text import human_time

SAVE_FILTER = ("PNG image (*.png);;JPEG image (*.jpg *.jpeg);;"
               "WebP image (*.webp);;All files (*)")

#: Filmstrip thumbnail edge, in grid units.
STRIP_GU = 3.4


class ResultPage(QWidget):
    """Shows the finished edit, keeps the session's earlier ones, offers both."""

    back_requested = Signal()
    #: Use this result as the source of the next edit.
    continue_requested = Signal(object)

    def __init__(self, settings: st.Settings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._source: Path | None = None
        self._recipe_id = ""
        self._seed = 0
        #: (source QImage, result QImage, summary, recipe id, seed, prompt).
        #: Holds the QImages for the whole session, which is affordable because
        #: an edit is about the size of its source — unlike a 4x upscale, where
        #: local-upscaler could not have kept even two.
        self._history: list[tuple] = []

        gu = me.metrics.gu
        outer = QVBoxLayout(self)
        outer.setContentsMargins(gu(0.75), gu(0.75), gu(0.75), gu(0.75))
        outer.setSpacing(gu(0.5))

        outer.addLayout(self._build_toolbar())
        self._view = ImageView()
        self._view.zoom_changed.connect(self._on_zoom_changed)
        outer.addWidget(self._view, 1)
        outer.addWidget(self._build_strip())
        outer.addLayout(self._build_footer())

        index = self._filter.findData(settings.compare_filter)
        if index >= 0:
            self._filter.setCurrentIndex(index)
        self._view.set_filter(settings.compare_filter)
        self._view.set_mode(MODE_COMPARE)

    # -- construction -----------------------------------------------------
    def _build_toolbar(self):
        gu = me.metrics.gu
        bar = QHBoxLayout()
        bar.setSpacing(gu(0.25))
        self._modes = QButtonGroup(self)
        self._modes.setExclusive(True)
        for index, (mode, label) in enumerate(((MODE_ORIGINAL, "Source"),
                                               (MODE_RESULT, "Result"),
                                               (MODE_COMPARE, "Compare"))):
            button = QToolButton()
            button.setText(label)
            button.setCheckable(True)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            button.setChecked(mode == MODE_COMPARE)
            self._modes.addButton(button, index)
            button.clicked.connect(lambda _c, m=mode: self._set_mode(m))
            bar.addWidget(button)
        bar.addSpacing(gu(1))

        self._filter = QComboBox()
        self._filter.addItem("Source: real pixels", "nearest")
        self._filter.addItem("Source: smoothed", "smooth")
        self._filter.setToolTip(
            "How the source is drawn when the result is a different size. Real "
            "pixels shows the source as it actually is.")
        self._filter.currentIndexChanged.connect(self._on_filter_changed)
        bar.addWidget(self._filter)

        bar.addStretch(1)
        self._zoom_label = QLabel()
        bar.addWidget(self._zoom_label)
        for text, tip, slot in (("Fit", "Fit the whole image (0)", self._fit),
                                ("100%", "One output pixel per screen pixel (1)",
                                 self._actual)):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            bar.addWidget(button)
        return bar

    def _build_strip(self) -> QWidget:
        gu = me.metrics.gu
        self._strip = QListWidget()
        self._strip.setViewMode(QListWidget.ViewMode.IconMode)
        self._strip.setFlow(QListWidget.Flow.LeftToRight)
        self._strip.setWrapping(False)
        self._strip.setMovement(QListWidget.Movement.Static)
        self._strip.setIconSize(QSize(gu(STRIP_GU), gu(STRIP_GU)))
        self._strip.setFixedHeight(gu(STRIP_GU) + gu(1.4))
        self._strip.setSpacing(gu(0.25))
        self._strip.setToolTip("This session's results. Click one to view it.")
        self._strip.currentRowChanged.connect(self._on_strip_selected)
        self._strip.setVisible(False)
        return self._strip

    def _build_footer(self):
        footer = QHBoxLayout()
        self._summary = QLabel()
        self._summary.setWordWrap(True)
        footer.addWidget(self._summary, 1)

        self._continue = QPushButton("Continue Editing This")
        self._continue.setToolTip(
            "Use this result as the image to edit, keeping the prompt and "
            "references.")
        self._continue.clicked.connect(self._on_continue)
        footer.addWidget(self._continue)

        self._back = QPushButton("Edit Another")
        self._back.setToolTip("Back to the first screen, keeping everything "
                              "as it was.")
        self._back.clicked.connect(self.back_requested.emit)
        footer.addWidget(self._back)

        self._save = QPushButton("Save Image…")
        self._save.setDefault(True)
        self._save.clicked.connect(self._save_as)
        footer.addWidget(self._save)
        return footer

    # -- content ----------------------------------------------------------
    def show_result(self, source: Path | None, original: QImage, result: QImage,
                    recipe_label: str, recipe_id: str, seed: int, steps: int,
                    elapsed: float, prompt_text: str,
                    notes: "tuple[str, ...]" = ()) -> None:
        # A pure text-to-image run has no source to compare against, so the
        # result stands in for it — the wipe then shows the same image on both
        # sides, which is honest, rather than a black panel that looks broken.
        original = original if not original.isNull() else result
        summary = (f"{result.width()} x {result.height()}   ·   {recipe_label}"
                   f"   ·   seed {seed}   ·   {steps} steps in "
                   f"{human_time(elapsed)}")
        if notes:
            summary += "\n" + "  ".join(notes)

        self._history.append((original, result, summary, recipe_id, seed,
                              prompt_text))
        self._add_to_strip(result, summary)
        self._show(len(self._history) - 1)
        self._source = source

    def _add_to_strip(self, result: QImage, tooltip: str) -> None:
        gu = me.metrics.gu(STRIP_GU)
        item = QListWidgetItem()
        item.setIcon(QIcon(QPixmap.fromImage(result).scaled(
            gu, gu, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)))
        item.setToolTip(tooltip)
        item.setText(str(self._strip.count() + 1))
        self._strip.addItem(item)
        self._strip.setVisible(self._strip.count() > 1)
        self._strip.blockSignals(True)
        self._strip.setCurrentRow(self._strip.count() - 1)
        self._strip.blockSignals(False)
        self._strip.scrollToBottom()

    def _show(self, index: int) -> None:
        if not (0 <= index < len(self._history)):
            return
        original, result, summary, recipe_id, seed, _prompt = self._history[index]
        self._recipe_id = recipe_id
        self._seed = seed
        self._view.set_images(original, result)
        self._summary.setText(summary)
        self._on_zoom_changed(self._view.zoom())

    def _on_strip_selected(self, row: int) -> None:
        self._show(row)

    def clear_history(self) -> None:
        self._history.clear()
        self._strip.clear()
        self._strip.setVisible(False)
        self._view.clear()

    # -- view controls ----------------------------------------------------
    def _set_mode(self, mode: str) -> None:
        self._view.set_mode(mode)
        self._filter.setEnabled(mode in (MODE_COMPARE, MODE_ORIGINAL))

    def _on_filter_changed(self) -> None:
        name = self._filter.currentData()
        self._settings.compare_filter = name
        self._view.set_filter(name)

    def _fit(self) -> None:
        self._view.fit()

    def _actual(self) -> None:
        self._view.zoom_to_actual()

    def _on_zoom_changed(self, zoom: float) -> None:
        self._zoom_label.setText(f"{zoom * 100:.0f}%")

    # -- chaining ---------------------------------------------------------
    def _on_continue(self) -> None:
        """Write the current result to the cache and hand back its path.

        A path rather than the QImage, because the setup page and the engine
        both want a file: the thumbnail is loaded from one, and the reference
        list and the server's base64 encoder both read from one. Writing it
        once here beats every consumer inventing its own temporary.
        """
        image = self._view.result_image()
        if image is None:
            return
        from .. import paths
        target = paths.unique_path(paths.work_dir() / "chain",
                                   f"edit-{self._recipe_id}-{self._seed}", ".png")
        if not image.save(str(target), "PNG"):
            QMessageBox.warning(self, "Could not continue",
                                f"Writing {target} failed, so this result "
                                f"cannot be used as the next input.")
            return
        self.continue_requested.emit(target)

    # -- saving -----------------------------------------------------------
    def _default_name(self) -> str:
        stem = self._source.stem if self._source else "edit"
        return f"{stem}_{self._recipe_id}_{self._seed}.png"

    def _save_as(self) -> None:
        image = self._view.result_image()
        if image is None:
            return
        start = (self._settings.last_save_dir
                 or (str(self._source.parent) if self._source else str(Path.home())))
        target, _ = QFileDialog.getSaveFileName(
            self, "Save Image", str(Path(start) / self._default_name()),
            SAVE_FILTER)
        if not target:
            return
        path = Path(target)
        if not path.suffix:
            path = path.with_suffix(".png")
        # JPEG has no alpha; saving an RGBA image to it silently loses the mask
        # or fails outright depending on the plugin, so flatten deliberately.
        to_save = image
        if path.suffix.lower() in (".jpg", ".jpeg") and image.hasAlphaChannel():
            to_save = image.convertToFormat(QImage.Format.Format_RGB32)
        if to_save.save(str(path), quality=95):
            self._settings.last_save_dir = str(path.parent)
            self._summary.setText(f"Saved to {path}")
        else:
            QMessageBox.warning(self, "Could not save",
                                f"Writing {path} failed. Check the folder is "
                                f"writable and has room for the file.")
