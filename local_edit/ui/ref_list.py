"""The reference list: which images, what each is for, and in what order.

This widget is the app's reason to exist. The models take references as a
numbered list and expect the prompt to name them by number, so three things have
to be visible at once and none of them can be implicit:

* **what** each image is (a thumbnail — filenames are useless for telling two
  photographs apart);
* **what it is for** (the role, which is what `engine.prompt` turns into words);
* **which number it is**, because the prompt says "the jacket in image 2" and
  the user needs to be able to check that 2 is the jacket.

The number is not the row's position in this list. When an edit model is
selected the image being edited is reference **1** and the user's own references
start at 2 — see `runner.Job.all_references` — so the list is told its offset
rather than counting from one and quietly lying.

Rows are drag-reorderable because reordering **renumbers**, and renumbering
changes the prompt. That is not a nicety: `--increase-ref-index` assigns indices
in the order the references are passed, so the only way to say "no, the jacket
should be image 2" is to move it.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout, QLabel,
                               QListWidget, QListWidgetItem, QSizePolicy,
                               QToolButton, QVBoxLayout, QWidget)

from ..engine import prompt as P
from . import icons
from . import metrics as me

#: Thumbnail edge, in grid units.
THUMB_GU = 2.6


class ReferenceRow(QWidget):
    """One reference: index, thumbnail, name, role, remove."""

    role_changed = Signal()
    remove_requested = Signal()

    def __init__(self, path: Path, role: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.path = path

        gu = me.metrics.gu
        row = QHBoxLayout(self)
        row.setContentsMargins(gu(0.25), gu(0.2), gu(0.25), gu(0.2))
        row.setSpacing(gu(0.5))

        self._index = QLabel()
        self._index.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._index.setFixedWidth(gu(1.2))
        index_font = self._index.font()
        index_font.setBold(True)
        self._index.setFont(index_font)
        row.addWidget(self._index)

        side = gu(THUMB_GU)
        self._thumb = QLabel()
        self._thumb.setFixedSize(side, side)
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image = QImage(str(path))
        if not image.isNull():
            self._thumb.setPixmap(QPixmap.fromImage(image).scaled(
                side, side, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
        row.addWidget(self._thumb)

        self._name = QLabel(path.name)
        self._name.setToolTip(str(path))
        # Elide from the left: the tail of a filename is what distinguishes
        # IMG_20240712_portrait.jpg from IMG_20240712_jacket.jpg.
        self._name.setTextFormat(Qt.TextFormat.PlainText)
        self._name.setSizePolicy(QSizePolicy.Policy.Ignored,
                                 QSizePolicy.Policy.Preferred)
        row.addWidget(self._name, 1)

        self._role = QComboBox()
        for value in P.ROLES:
            self._role.addItem(P.ROLE_LABELS[value], value)
            self._role.setItemData(self._role.count() - 1, P.ROLE_HINTS[value],
                                   Qt.ItemDataRole.ToolTipRole)
        index = self._role.findData(role)
        self._role.setCurrentIndex(max(0, index))
        # Swallows the index `currentIndexChanged` emits; the signal here
        # carries no argument, and connecting it directly raises at runtime.
        self._role.currentIndexChanged.connect(
            lambda _index: self.role_changed.emit())
        row.addWidget(self._role)

        self._remove = QToolButton()
        self._remove.setAutoRaise(True)
        self._remove.setToolTip(f"Remove {path.name}")
        icons.themed(self._remove, "edit-delete-remove")
        self._remove.clicked.connect(self.remove_requested.emit)
        row.addWidget(self._remove)

    def role(self) -> str:
        return self._role.currentData()

    def set_number(self, n: int) -> None:
        self._index.setText(str(n))
        self._index.setToolTip(f"The prompt can refer to this as image {n}.")

    def set_dimmed(self, dimmed: bool) -> None:
        """Grey a row the current model will not use.

        Shown rather than removed: the user chose these images, and a model
        change should not silently discard them — switching back must bring
        them all into play again.
        """
        self.setEnabled(not dimmed)
        self._index.setText("—" if dimmed else self._index.text())

    def reference(self) -> P.Reference:
        return P.Reference(self.path, self.role())


class ReferenceList(QWidget):
    """The reference images, in order, each with a role."""

    changed = Signal()
    add_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._offset = 0
        self._limit = 99

        gu = me.metrics.gu
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(gu(0.25))

        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self._list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._list.setUniformItemSizes(True)
        self._list.setAlternatingRowColors(True)
        self._list.setToolTip(
            "Drag to reorder. The order is the numbering the prompt uses.")
        self._list.model().rowsMoved.connect(self._on_reordered)
        outer.addWidget(self._list, 1)

        row = QHBoxLayout()
        self._add = QToolButton()
        self._add.setText("Add reference…")
        self._add.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        icons.themed(self._add, "list-add")
        self._add.clicked.connect(self.add_requested.emit)
        row.addWidget(self._add)
        row.addStretch(1)
        self._hint = QLabel()
        self._hint.setWordWrap(True)
        row.addWidget(self._hint, 1)
        outer.addLayout(row)

        self._sync_height()

    # -- contents ---------------------------------------------------------
    def add(self, path: Path, role: str = P.ROLE_PLAIN) -> None:
        row = ReferenceRow(path, role)
        row.role_changed.connect(self.changed.emit)
        row.remove_requested.connect(lambda r=row: self._remove(r))

        item = QListWidgetItem()
        item.setSizeHint(row.sizeHint())
        # Rows carry their own widgets, so an item must not also be a drop
        # target for text — InternalMove plus ItemIsDropEnabled on the item
        # produces a drop *into* a row, which silently loses it.
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                      | Qt.ItemFlag.ItemIsDragEnabled)
        self._list.addItem(item)
        self._list.setItemWidget(item, row)
        self._renumber()
        self._sync_height()
        self.changed.emit()

    def add_many(self, paths: "list[Path] | tuple[Path, ...]") -> None:
        for path in paths:
            self.add(path)

    def _remove(self, row: ReferenceRow) -> None:
        for i in range(self._list.count()):
            item = self._list.item(i)
            if self._list.itemWidget(item) is row:
                self._list.takeItem(i)
                row.deleteLater()
                break
        self._renumber()
        self._sync_height()
        self.changed.emit()

    def clear(self) -> None:
        self._list.clear()
        self._sync_height()
        self.changed.emit()

    def rows(self) -> list[ReferenceRow]:
        out = []
        for i in range(self._list.count()):
            widget = self._list.itemWidget(self._list.item(i))
            if isinstance(widget, ReferenceRow):
                out.append(widget)
        return out

    def references(self) -> list[P.Reference]:
        return [row.reference() for row in self.rows()]

    def paths(self) -> list[Path]:
        return [row.path for row in self.rows()]

    def count(self) -> int:
        return self._list.count()

    def has(self, path: Path) -> bool:
        return any(row.path == path for row in self.rows())

    # -- numbering --------------------------------------------------------
    def set_numbering(self, offset: int, limit: int) -> None:
        """`offset` is how many references precede these; `limit` is the cap.

        The offset is normally 1 — the image being edited occupies reference 1
        for an edit model — and 0 for a model that takes the source as an init
        image instead.
        """
        self._offset = max(0, offset)
        self._limit = max(0, limit)
        self._renumber()

    def _renumber(self) -> None:
        rows = self.rows()
        for i, row in enumerate(rows):
            usable = (i + self._offset) < self._limit
            row.set_dimmed(not usable)
            if usable:
                row.set_number(i + self._offset + 1)
        over = max(0, len(rows) + self._offset - self._limit)
        self._hint.setText(
            f"{over} reference{'s' if over != 1 else ''} beyond what this "
            f"model uses." if over else "")

    def _on_reordered(self, *_: object) -> None:
        self._renumber()
        self.changed.emit()

    def _sync_height(self) -> None:
        """Grow with the contents, up to a few rows, then scroll.

        A fixed-height list would either waste a third of the setup page when
        empty or need scrolling at two references. The cap exists because the
        prompt and the model picker below matter more than a sixth thumbnail.
        """
        gu = me.metrics.gu
        rows = max(1, min(4, self._list.count()))
        row_height = gu(THUMB_GU) + gu(0.8)
        self._list.setFixedHeight(rows * row_height + gu(0.4))
        self._list.setVisible(self._list.count() > 0)

    def sizeHint(self) -> QSize:
        return QSize(me.metrics.gu(20), me.metrics.gu(8))
