"""The first screen: an image, some references, a prompt, a model.

The layout answers the questions a user has here, in the order they have them:
what am I editing, what should it borrow from, what should change, and what will
that cost. The last one is why the model combo carries a hardware verdict rather
than only a name — with a catalog spanning 3.5 GB to 34 GB against a card with
3.4 GiB free, a bare list of names would be a trap.

Models that will not run comfortably here are shown anyway, annotated with what
they would need. Hiding them would mean a user has to already know a model
exists before they can find out it is too big, and on a machine that is about to
be replaced, "needs about 9 GB of VRAM" is exactly the information worth having.

Qt conventions follow local-upscaler's, which took them from soundboard: no
`setStyleSheet`, no `setStyle`, every size derived from `ui.metrics`, and icons
through `ui.icons` so a theme change repaints them.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from PIL import Image
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QImageReader, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QFrame, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QPlainTextEdit, QPushButton,
                               QScrollArea, QSizePolicy, QSpinBox, QToolButton,
                               QVBoxLayout, QWidget)

from .. import settings as st
from ..engine import binary, catalog, fetch, hardware, prompt as P, runner
from . import icons
from . import metrics as me
from .ref_list import ReferenceList
from .text import human_bytes, human_time, plural

#: Extensions listed first in the dialog, because they are what people have.
_COMMON_EXTENSIONS = ("png", "jpg", "jpeg", "jfif", "webp", "avif",
                      "tif", "tiff", "bmp", "gif")

THUMB_GU = 7


def readable_extensions() -> tuple[str, ...]:
    """Extensions both Pillow and Qt can open on this machine.

    Derived, not typed out, which is local-upscaler's hard-won lesson: its
    hand-written list omitted `.jfif`, which is not an obscure format but
    ordinary JPEG — Windows and Edge hand out `.jfif` when you save an image
    from the web, both libraries read it happily, and the only thing rejecting
    those files was a string in a module.

    Both libraries have to agree, because Qt draws the thumbnails and the
    comparison view while Pillow validates and re-encodes.
    """
    Image.init()
    qt_mimes = {bytes(m).decode().lower()
                for m in QImageReader.supportedMimeTypes()}
    found = set()
    for extension, plugin in Image.EXTENSION.items():
        if plugin not in Image.OPEN:
            continue                  # Pillow can write it but not read it
        mime, _ = mimetypes.guess_type("x" + extension)
        if mime and mime in qt_mimes:
            found.add(extension.lstrip(".").lower())
    found.update(("png", "jpg", "jpeg"))
    return tuple(sorted(found))


def image_filter() -> str:
    every = readable_extensions()
    ordered = ([e for e in _COMMON_EXTENSIONS if e in every]
               + [e for e in every if e not in _COMMON_EXTENSIONS])
    return f"Images ({' '.join(f'*.{e}' for e in ordered)});;All files (*)"


def readable(path: Path) -> bool:
    """Whether both libraries can open `path`. Reads headers only."""
    if QImage(str(path)).isNull():
        return False
    try:
        with Image.open(path) as probe:
            probe.verify()
    except Exception:                                   # noqa: BLE001
        return False                # Pillow raises many types for a bad file
    return True


class SetupPage(QWidget):
    """Choose an image, references, a prompt and a model."""

    generate_requested = Signal()

    def __init__(self, settings: st.Settings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._source: Path | None = None
        self._source_size: tuple[int, int] = (0, 0)
        #: Set by `output_size` when "match the source" had to be capped.
        self._capped_to: int | None = None
        self._hardware = hardware.Hardware()
        self._verdict: hardware.Verdict | None = None
        self._override_locked = False
        self._note = ""
        #: Whether `_auto_size` has moved the output size, so `_refresh` can
        #: explain a control that changed on its own.
        self._auto_sized = False
        #: What the model labels were last computed for; see
        #: `_refresh_recipe_labels`. `Hardware` is a frozen dataclass, so it
        #: compares by value and a re-probe returning the same numbers is
        #: correctly a no-op.
        self._labels_key: tuple | None = None

        gu = me.metrics.gu

        # The content scrolls; the estimate and the Generate button do not.
        #
        # There is more here than fits: with the Advanced group collapsed and no
        # references, the content alone wants about 1050 px against a 640 px
        # default window, and each reference row adds more. Letting the layout
        # squeeze instead would compress the prompt box and the thumbnail to
        # nothing on a small screen or a large font.
        #
        # Keeping the estimate and the button outside the scroll area is the
        # point of doing it this way rather than making the whole page scroll:
        # "About 6 min — 4 steps at 768 x 768 with 3 references" is the last
        # thing anyone reads before committing to a six-minute wait, and it must
        # not be the thing that scrolled off the bottom.
        content = QWidget()
        inner = QVBoxLayout(content)
        inner.setContentsMargins(gu(1.5), gu(1.5), gu(1.5), gu(0.5))
        inner.setSpacing(gu(0.75))
        inner.addLayout(self._build_source_row())
        inner.addWidget(self._build_references())
        inner.addWidget(self._build_prompt())
        inner.addWidget(self._build_model())
        inner.addWidget(self._build_advanced())
        inner.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(gu(0.5))
        outer.addWidget(scroll, 1)

        footer = QVBoxLayout()
        footer.setContentsMargins(gu(1.5), 0, gu(1.5), gu(1.5))
        footer.setSpacing(gu(0.5))
        self._estimate = QLabel()
        self._estimate.setWordWrap(True)
        footer.addWidget(self._estimate)

        go = QHBoxLayout()
        go.addStretch(1)
        self._go = QPushButton("Generate")
        self._go.setDefault(True)
        self._go.clicked.connect(self.generate_requested.emit)
        go.addWidget(self._go)
        footer.addLayout(go)
        outer.addLayout(footer)

        self.setAcceptDrops(True)
        self._load_settings()
        self._refresh()

    # -- construction -----------------------------------------------------
    def _build_source_row(self):
        gu = me.metrics.gu
        pick = QHBoxLayout()
        pick.setSpacing(gu(1))

        self._thumb = QLabel()
        self._thumb.setFixedSize(gu(THUMB_GU), gu(THUMB_GU))
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb.setFrameShape(QFrame.Shape.StyledPanel)
        pick.addWidget(self._thumb)

        details = QVBoxLayout()
        details.setSpacing(gu(0.25))
        self._name = QLabel("No image chosen")
        font = self._name.font()
        font.setBold(True)
        self._name.setFont(font)
        self._name.setWordWrap(True)
        self._dims = QLabel()
        self._dims.setWordWrap(True)
        details.addWidget(self._name)
        details.addWidget(self._dims)
        details.addStretch(1)

        row = QHBoxLayout()
        self._open = QPushButton("Open Image…")
        self._open.clicked.connect(self._choose_source)
        row.addWidget(self._open)
        self._clear = QPushButton("Clear")
        self._clear.setToolTip("Generate from the prompt alone, with no image "
                               "to edit.")
        self._clear.clicked.connect(self.clear_source)
        row.addWidget(self._clear)
        row.addStretch(1)
        details.addLayout(row)
        pick.addLayout(details, 1)
        return pick

    def _build_references(self) -> QWidget:
        box = QGroupBox("References")
        box.setToolTip("Extra images the model should borrow from — a face, an "
                       "outfit, a place.")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(*[me.metrics.gu(0.5)] * 4)
        self._refs = ReferenceList()
        self._refs.add_requested.connect(self._choose_references)
        self._refs.changed.connect(self._refresh)
        layout.addWidget(self._refs)
        return box

    def _build_prompt(self) -> QWidget:
        gu = me.metrics.gu
        box = QGroupBox("Prompt")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(gu(0.5), gu(0.5), gu(0.5), gu(0.5))
        layout.setSpacing(gu(0.25))

        self._prompt = QPlainTextEdit()
        self._prompt.setPlaceholderText(
            "What should change?  e.g. make the person ride a motorcycle")
        self._prompt.setFixedHeight(gu(3.4))
        self._prompt.textChanged.connect(self._on_prompt_changed)
        layout.addWidget(self._prompt)

        sent = QHBoxLayout()
        sent.setSpacing(gu(0.5))
        caption = QLabel("Sent to the model:")
        # De-emphasised with the font rather than `setEnabled(False)`, which
        # greys it to the disabled role and makes a perfectly live caption read
        # as a broken control.
        caption_font = caption.font()
        caption_font.setPointSizeF(max(6.0, caption_font.pointSizeF() * 0.92))
        caption_font.setItalic(True)
        caption.setFont(caption_font)
        sent.addWidget(caption)
        sent.addStretch(1)
        self._edit_composed = QToolButton()
        self._edit_composed.setText("Edit")
        self._edit_composed.setCheckable(True)
        self._edit_composed.setToolTip(
            "Take over the composed prompt and send it exactly as written.")
        icons.themed(self._edit_composed, "document-edit")
        self._edit_composed.toggled.connect(self._on_override_toggled)
        sent.addWidget(self._edit_composed)
        layout.addLayout(sent)

        self._composed = QPlainTextEdit()
        self._composed.setReadOnly(True)
        self._composed.setFixedHeight(gu(3.0))
        self._composed.textChanged.connect(self._on_composed_edited)
        layout.addWidget(self._composed)
        return box

    def _build_model(self) -> QWidget:
        gu = me.metrics.gu
        box = QGroupBox("Model")
        form = QFormLayout(box)
        form.setSpacing(gu(0.5))

        self._recipe = QComboBox()
        self._recipe.setSizePolicy(QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Fixed)
        for recipe in catalog.RECIPES:
            self._recipe.addItem(recipe.label, recipe.id)
        self._recipe.currentIndexChanged.connect(self._on_recipe_changed)
        # No row label: the group box already says "Model", and a form row
        # labelled the same thing reads as a mistake.
        form.addRow(self._recipe)

        self._blurb = QLabel()
        self._blurb.setWordWrap(True)
        form.addRow("", self._blurb)
        return box

    def _build_advanced(self) -> QWidget:
        gu = me.metrics.gu
        box = QGroupBox("Advanced")
        box.setCheckable(True)
        box.setChecked(False)
        form = QFormLayout(box)
        form.setSpacing(gu(0.5))

        self._size = QComboBox()
        for value in st.SIZE_CHOICES:
            self._size.addItem("Match the source image" if value is None
                               else f"{value} px on the long edge", value)
        self._size.setToolTip(
            "Cost grows faster than area — attention is quadratic in pixels, so "
            "doubling this is far more than four times the work.")
        self._size.currentIndexChanged.connect(self._on_advanced_changed)
        form.addRow("Output size", self._size)

        self._steps = QSpinBox()
        self._steps.setRange(0, st.MAX_STEPS)
        self._steps.setSpecialValueText("Model default")
        self._steps.setToolTip("More steps, more time, diminishing returns. "
                               "Distilled models are trained for very few.")
        self._steps.valueChanged.connect(self._on_advanced_changed)
        form.addRow("Steps", self._steps)

        self._cfg = QDoubleSpinBox()
        self._cfg.setRange(0.0, 30.0)
        self._cfg.setSingleStep(0.5)
        self._cfg.setDecimals(1)
        self._cfg.setSpecialValueText("Model default")
        self._cfg.setToolTip("How hard the model is pushed towards the prompt.")
        self._cfg.valueChanged.connect(self._on_advanced_changed)
        form.addRow("Guidance", self._cfg)

        seed_row = QHBoxLayout()
        self._seed = QSpinBox()
        self._seed.setRange(0, 2 ** 31 - 1)
        self._seed.setToolTip("The same seed with the same prompt gives the "
                              "same image.")
        self._seed.valueChanged.connect(self._on_advanced_changed)
        seed_row.addWidget(self._seed, 1)
        self._randomize = QCheckBox("Random each run")
        self._randomize.toggled.connect(self._on_randomize_toggled)
        seed_row.addWidget(self._randomize)
        form.addRow("Seed", seed_row)

        self._negative = QLineEdit()
        self._negative.setPlaceholderText("What to avoid (ignored by distilled "
                                          "models)")
        self._negative.textChanged.connect(self._on_advanced_changed)
        form.addRow("Negative", self._negative)

        self._memory = QComboBox()
        for value, label in ((st.MEMORY_AUTO, "Automatic"),
                             (st.MEMORY_VRAM, "Keep everything on the GPU"),
                             (st.MEMORY_RAM, "Offload weights to RAM"),
                             (st.MEMORY_DISK, "Stream weights from disk")):
            self._memory.addItem(label, value)
        self._memory.setToolTip(
            "Automatic picks from measured free VRAM and RAM. Override it if a "
            "run fails to allocate.")
        self._memory.currentIndexChanged.connect(self._on_advanced_changed)
        form.addRow("Memory", self._memory)

        self._backend = QComboBox()
        self._backend.addItem("Automatic", st.BACKEND_AUTO)
        for value in binary.BACKENDS:
            self._backend.addItem(value.capitalize(), value)
        self._backend.setToolTip("CPU works with no GPU driver, and is far "
                                 "slower.")
        self._backend.currentIndexChanged.connect(self._on_advanced_changed)
        form.addRow("Backend", self._backend)
        return box

    # -- settings <-> widgets ---------------------------------------------
    def _load_settings(self) -> None:
        s = self._settings
        for widget, value in ((self._recipe, s.recipe_id),
                              (self._size, s.size),
                              (self._memory, s.memory_mode),
                              (self._backend, s.backend)):
            index = widget.findData(value)
            if index >= 0:
                widget.blockSignals(True)
                widget.setCurrentIndex(index)
                widget.blockSignals(False)
        for widget, value in ((self._steps, s.steps or 0),
                              (self._cfg, s.cfg_scale or 0.0),
                              (self._seed, max(0, s.seed))):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        for widget, value in ((self._randomize, s.randomize_seed),):
            widget.blockSignals(True)
            widget.setChecked(value)
            widget.blockSignals(False)
        self._negative.blockSignals(True)
        self._negative.setText(s.negative_prompt)
        self._negative.blockSignals(False)
        self._prompt.blockSignals(True)
        self._prompt.setPlainText(s.last_prompt)
        self._prompt.blockSignals(False)
        self._seed.setEnabled(not s.randomize_seed)

    def _on_advanced_changed(self, *_: object) -> None:
        s = self._settings
        s.size = self._size.currentData()
        s.size_chosen = True          # the user has an opinion now
        s.steps = self._steps.value() or None
        s.cfg_scale = self._cfg.value() or None
        s.seed = -1 if self._randomize.isChecked() else self._seed.value()
        s.negative_prompt = self._negative.text()
        s.memory_mode = self._memory.currentData()
        s.backend = self._backend.currentData()
        self._refresh()

    def _on_randomize_toggled(self, checked: bool) -> None:
        self._settings.randomize_seed = checked
        self._seed.setEnabled(not checked)
        self._on_advanced_changed()

    def _on_recipe_changed(self, *_: object) -> None:
        self._settings.recipe_id = self._recipe.currentData()
        self._refresh()

    def _on_prompt_changed(self) -> None:
        self._settings.last_prompt = self._prompt.toPlainText()
        self._refresh()

    def _on_override_toggled(self, checked: bool) -> None:
        """Hand the composed prompt over to the user, or take it back."""
        self._composed.setReadOnly(not checked)
        if checked:
            self._composed.setFocus()
        else:
            self._override_locked = False
            self._refresh()

    def _on_composed_edited(self) -> None:
        # `_refresh` writes this box, so only a user edit — which can only
        # happen while the override toggle is on — counts as taking it over.
        if self._edit_composed.isChecked():
            self._override_locked = True

    # -- hardware ---------------------------------------------------------
    def set_hardware(self, hw: hardware.Hardware) -> None:
        self._hardware = hw
        self._refresh()

    def hardware(self) -> hardware.Hardware:
        """The last probe. `MainWindow` needs it to grade a finished run."""
        return self._hardware

    def verdict(self) -> hardware.Verdict:
        """How the selected recipe would run, at the selected size.

        `MainWindow._start` reads this to pick the memory flags and the initial
        ETA, so losing it breaks the Generate button and nothing else — which is
        exactly what happened once: a slice-based edit to the method above this
        one took this and `hardware()` with it, and because a Qt slot swallows
        the AttributeError, pressing Generate silently did nothing at all.
        """
        return self._grade()

    def _auto_size(self) -> bool:
        """Apply `settings.auto_size`, until the user has an opinion of their own.

        Returns True when it changed the size, so `_refresh` can say why: a
        control that moves on its own without explanation is worse than one that
        never moves.
        """
        s = self._settings
        if s.size_chosen:
            return False
        best = st.auto_size(s.recipe(), self._hardware, s.calibration,
                            backend=s.engine_backend())
        if best == s.size:
            return False
        s.size = best
        index = self._size.findData(best)
        if index >= 0:
            self._size.blockSignals(True)
            self._size.setCurrentIndex(index)
            self._size.blockSignals(False)
        return True

    def _grade(self) -> hardware.Verdict:
        s = self._settings
        recipe = s.recipe()
        width, height = self.output_size()
        pending = fetch.download_size(recipe, s.models_path())
        return hardware.verdict(recipe, self._hardware, width, height, pending)

    def _refresh_recipe_labels(self) -> None:
        """Re-label every entry with how it would run on this machine.

        Guarded by a key, because `_refresh` runs on every keystroke in the
        prompt box and this loop grades eight recipes and writes eight combo
        items. None of that depends on the prompt, and rewriting a combo box's
        text while the user types makes it flicker.
        """
        s = self._settings
        width, height = self.output_size()
        models = s.models_path()
        key = (width, height, str(models), self._hardware)
        if key == self._labels_key:
            return
        self._labels_key = key
        for i in range(self._recipe.count()):
            recipe = catalog.get(self._recipe.itemData(i))
            pending = fetch.download_size(recipe, models)
            v = hardware.verdict(recipe, self._hardware, width, height, pending)
            suffix = v.summary
            if pending:
                suffix += f" · {human_bytes(pending)} download"
            self._recipe.setItemText(i, f"{recipe.label} — {suffix}")

    # -- source -----------------------------------------------------------
    def _choose_source(self) -> None:
        start = self._settings.last_open_dir or str(Path.home())
        name, _ = QFileDialog.getOpenFileName(self, "Open Image", start,
                                              image_filter())
        if name:
            self.load_image(Path(name))

    def load_image(self, path: Path) -> bool:
        """Load `path` as the image to edit, or report why it cannot be used."""
        image = QImage(str(path))
        if image.isNull() or not readable(path):
            self._reject(path)
            return False
        self._source = path
        self._source_size = (image.width(), image.height())
        self._settings.last_open_dir = str(path.parent)
        side = me.metrics.gu(THUMB_GU)
        self._thumb.setPixmap(QPixmap.fromImage(image).scaled(
            side, side, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))
        self._name.setText(path.name)
        self._refresh()
        return True

    def clear_source(self) -> None:
        self._source = None
        self._source_size = (0, 0)
        self._thumb.clear()
        self._name.setText("No image chosen")
        self._refresh()

    def _reject(self, path: Path) -> None:
        """Clear the selection and say why.

        `_refresh` rewrites the detail line for the no-image case, so it runs
        *before* the reason is written — otherwise it overwrites the reason with
        the generic placeholder, which is how local-upscaler's rejected files
        once reported nothing more useful than "choose an image".
        """
        self.clear_source()
        self._name.setText("Could not read that file")
        self._dims.setText(f"{path.name} is not an image this app can open.")

    # -- references -------------------------------------------------------
    def _choose_references(self) -> None:
        start = self._settings.last_open_dir or str(Path.home())
        names, _ = QFileDialog.getOpenFileNames(self, "Add Reference Images",
                                                start, image_filter())
        self.add_references([Path(n) for n in names])

    def add_references(self, paths: "list[Path]") -> None:
        added = [p for p in paths if readable(p) and not self._refs.has(p)]
        for path in added:
            self._refs.add(path)
        if added:
            self._settings.last_open_dir = str(added[-1].parent)

    # -- drag and drop ----------------------------------------------------
    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    dragMoveEvent = dragEnterEvent

    def dropEvent(self, event) -> None:
        """Dropped files: the first becomes the source if there is none yet.

        The rule is stated in the placeholder text rather than inferred from
        where in the window the drop landed. A drop zone that behaves
        differently by a few pixels is worse than one rule that always holds.
        """
        paths = [Path(u.toLocalFile()) for u in event.mimeData().urls()
                 if u.isLocalFile()]
        paths = [p for p in paths if p.is_file()]
        if not paths:
            return
        if self._source is None and self.load_image(paths[0]):
            paths = paths[1:]
        self.add_references(paths)
        event.acceptProposedAction()

    # -- what the window needs --------------------------------------------
    @property
    def source(self) -> Path | None:
        return self._source

    @property
    def source_size(self) -> tuple[int, int]:
        return self._source_size

    def output_size(self) -> tuple[int, int]:
        width, height, capped = st.plan_output_size(
            self._settings.recipe(), self._hardware,
            self._source_size if self._source else None,
            self._settings.size)
        # Remembered so `_refresh` can say the cap happened. A size control
        # that quietly produces a different size than it names is worse than
        # one that refuses.
        self._capped_to = capped
        return width, height

    def fitting_size(self) -> int | None:
        """The largest offered size this machine can hold, or None if none can."""
        return st.fitting_size(self._settings.recipe(), self._hardware,
                               self._source_size if self._source else None)

    def use_size(self, longest: int) -> None:
        """Switch the output size, as if the user had picked it themselves.

        Marks `size_chosen`, because after this the app must not move the size
        again on its own — `_auto_size` would otherwise overwrite the choice
        the user just accepted in a dialog.
        """
        s = self._settings
        s.size = longest
        s.size_chosen = True
        index = self._size.findData(longest)
        if index >= 0:
            self._size.blockSignals(True)
            self._size.setCurrentIndex(index)
            self._size.blockSignals(False)
        self._refresh()

    def set_prompt(self, text: str) -> None:
        self._prompt.setPlainText(text)

    def build_job(self) -> runner.Job:
        s = self._settings
        recipe = s.recipe()
        width, height = self.output_size()
        verdict = self._grade()
        override = (self._composed.toPlainText().strip()
                    if self._override_locked else "")
        return runner.Job(
            recipe=recipe, source=self._source,
            references=tuple(self._refs.references()),
            instruction=self._prompt.toPlainText().strip(),
            prompt_override=override, negative_prompt=s.negative_prompt,
            width=width, height=height, seed=s.seed,
            steps=s.steps, cfg_scale=s.cfg_scale, strength=s.strength,
            ip_adapter_strength=s.ip_adapter_strength,
            models_dir=s.models_path(), memory=s.memory_flags(verdict),
            backend=s.engine_backend(), threads=s.threads,
            max_vram_gb=s.max_vram_gb,
            engine_path=s.engine_path or None,
            extra_args=recipe.extra_args)

    # -- the summary ------------------------------------------------------
    def _refresh(self) -> None:
        s = self._settings
        self._auto_sized = self._auto_size() or self._auto_sized
        recipe = s.recipe()
        width, height = self.output_size()
        verdict = self._grade()
        self._verdict = verdict

        self._refresh_recipe_labels()
        licence = f" · {recipe.licence}" if recipe.licence else ""
        self._blurb.setText(f"{recipe.blurb}\n{recipe.author}{licence}")

        # The source occupies reference 1 for an edit model, so the user's own
        # references start at 2 and the list must say so.
        offset = 1 if (self._source is not None and recipe.source_as_ref) else 0
        self._refs.set_numbering(offset, recipe.max_refs)

        if self._source is not None:
            self._dims.setText(f"{self._source_size[0]} x "
                               f"{self._source_size[1]}  ->  {width} x {height}")
        else:
            self._dims.setText(
                "Drop an image here, or generate from the prompt alone. "
                "Extra images dropped after the first become references.")

        self._refresh_composed(recipe)
        self._refresh_estimate(recipe, verdict, width, height)

    def _refresh_composed(self, recipe: catalog.Recipe) -> None:
        if self._override_locked:
            return
        refs, note = P.apply_limits(self._model_references(recipe),
                                    recipe.max_refs, recipe.ref_images,
                                    recipe.label)
        instruction = self._prompt.toPlainText().strip()
        if recipe.uses_photomaker():
            text, warning = P.photomaker_prompt(instruction)
            note = warning or note
        else:
            text = P.compose(refs, instruction, recipe.family)

        self._composed.blockSignals(True)
        self._composed.setPlainText(text)
        self._composed.blockSignals(False)
        self._composed.setToolTip(note or "")
        self._note = note

    def _model_references(self, recipe: catalog.Recipe) -> list[P.Reference]:
        refs = self._refs.references()
        if self._source is not None and recipe.source_as_ref:
            refs.insert(0, P.Reference(self._source, P.ROLE_PLAIN))
        return refs

    def _refresh_estimate(self, recipe: catalog.Recipe,
                          verdict: hardware.Verdict,
                          width: int, height: int) -> None:
        s = self._settings
        steps = s.steps or recipe.steps
        pending = fetch.download_size(recipe, s.models_path())
        rate = s.calibration.get(recipe.id, width, height, verdict.grade,
                                 s.engine_backend())
        # `references=` is not optional here. Each reference adds its tokens to
        # the sequence the transformer attends over — measured at 22.3 s/step
        # bare against 55.4 s/step with two — so leaving it out understated the
        # app's central use case by more than half, and disagreed with the ETA
        # the progress page went on to show.
        refs = len(P.apply_limits(self._model_references(recipe),
                                  recipe.max_refs, recipe.ref_images,
                                  recipe.label)[0])
        seconds = runner.estimate_seconds(recipe, width, height, steps, rate,
                                          verdict.grade, references=refs)

        bits: list[str] = []
        if not self._prompt.toPlainText().strip() and self._source is None:
            bits.append("Type a prompt, or open an image to edit.")
        basis = "measured on this machine" if rate else "estimated"
        bits.append(f"About {human_time(seconds)} ({basis}) — "
                    f"{plural(steps, 'step')} at {width} x {height}"
                    + (f" with {plural(refs, 'reference')}." if refs else "."))
        if pending:
            bits.append(f"Downloads {human_bytes(pending)} first.")
        if verdict.grade == hardware.NO_DISK:
            bits.append(f"There is not enough free space in "
                        f"{s.models_path()} for this model.")
        elif verdict.grade == hardware.TOO_BIG:
            bits.append("It will still try, by cutting the work into pieces, "
                        "but expect it to be slow.")
        if self._capped_to is not None:
            bits.append(f"The source is larger than this machine can generate, "
                        f"so the output is capped at {self._capped_to} px — "
                        f"pick a size in Advanced to override.")
        if self._hardware.swap_is_zram and verdict.grade in (hardware.STREAM,
                                                             hardware.TOO_BIG):
            bits.append("Swap on this machine is zram — compressed RAM — which "
                        "does not help with model weights.")
        if self._auto_sized and not s.size_chosen:
            bits.append(f"Set to {s.size} px from the speed measured on this "
                        f"machine — change it in Advanced.")
        if self._note:
            bits.append(self._note)

        self._estimate.setText(" ".join(bits))
        ready = (verdict.runnable
                 and (self._source is not None
                      or bool(self._prompt.toPlainText().strip())))
        self._go.setEnabled(ready)

    def refresh(self) -> None:
        """Re-read state that may have changed elsewhere (a finished download)."""
        self._refresh()
