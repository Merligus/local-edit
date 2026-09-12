"""The shared pan/zoom canvas, and the before/after wipe.

Taken from local-upscaler, where the same widget compares a source against its
upscale. The problem here is the same one: one widget serves all three views —
source, result, and the comparison — because they must share a transform.
Switching views is then just a repaint, so "look at this face before / after" is
one click rather than a hunt.

**The memory rule this file exists to enforce.** Nothing here ever materialises
a scaled copy of anything. Both images are kept at their native size, and every
paint computes the small source rectangle that is actually visible and hands
*that* to `drawImage`, which scales only the pixels being shown. Cost per frame
is bounded by the size of the widget, not by the size of the image. This matters
less for edits than it did for 4x upscales — the result is usually about the
size of the source — but a 4 MP source against a 1 MP result still costs nothing
to compare, and an edit chained five times never accumulates buffers.

The two images live in one coordinate space, "canvas space", which is the
result's pixel grid. The source is drawn stretched onto it, so both halves
occupy the same screen area at the same zoom and the slider reveals a difference
in *content* rather than a difference in size. An edit that changed the aspect
ratio therefore shows the source slightly distorted, which is the honest
rendering of "this is what that region became".
"""


from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPalette, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from . import metrics as me

MODE_ORIGINAL = "original"
MODE_RESULT = "result"
MODE_COMPARE = "compare"

#: Zoom limits, in widget pixels per canvas pixel.
MIN_ZOOM = 0.02
MAX_ZOOM = 32.0

#: Width of the grab area around the wipe handle, in device-independent pixels.
HANDLE_GRAB = 12


class ImageView(QWidget):
    """Displays the original, the edited result, or a wipe between them."""

    #: Emitted when the user drags the wipe handle, with the split in 0..1.
    split_changed = Signal(float)
    zoom_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._original: QImage | None = None
        self._result: QImage | None = None
        self._mode = MODE_RESULT
        self._split = 0.5
        self._filter = "nearest"

        self._zoom = 1.0
        self._center = QPointF(0.0, 0.0)     # canvas coords at widget centre
        self._panning = False
        self._grabbing_handle = False
        self._last_pos = QPoint()

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAutoFillBackground(False)

    # -- content ----------------------------------------------------------
    def set_images(self, original: QImage, result: QImage) -> None:
        self._original = original
        self._result = result
        self.fit()

    def clear(self) -> None:
        self._original = self._result = None
        self.update()

    @property
    def has_content(self) -> bool:
        return self._result is not None and self._original is not None

    def result_image(self) -> QImage | None:
        """The full-resolution result, for saving or for a follow-up edit."""
        return self._result

    def original_image(self) -> QImage | None:
        return self._original

    def canvas_size(self) -> tuple[int, int]:
        if self._result is None:
            return 0, 0
        return self._result.width(), self._result.height()

    # -- view state -------------------------------------------------------
    def set_mode(self, mode: str) -> None:
        if mode != self._mode:
            self._mode = mode
            self.update()

    def mode(self) -> str:
        return self._mode

    def set_filter(self, name: str) -> None:
        if name != self._filter:
            self._filter = name
            self.update()

    def set_split(self, value: float) -> None:
        value = max(0.0, min(1.0, float(value)))
        if abs(value - self._split) > 1e-6:
            self._split = value
            self.update()
            self.split_changed.emit(value)

    def split(self) -> float:
        return self._split

    def zoom(self) -> float:
        return self._zoom

    def set_zoom(self, zoom: float, anchor: QPointF | None = None) -> None:
        """Set the zoom, keeping `anchor` (widget coords) over the same pixel."""
        zoom = max(MIN_ZOOM, min(MAX_ZOOM, float(zoom)))
        if abs(zoom - self._zoom) < 1e-9:
            return
        if anchor is None:
            anchor = QPointF(self.width() / 2.0, self.height() / 2.0)
        before = self._widget_to_canvas(anchor)
        self._zoom = zoom
        after = self._widget_to_canvas(anchor)
        self._center += before - after
        self._clamp_center()
        self.update()
        self.zoom_changed.emit(self._zoom)

    def fit(self) -> None:
        """Zoom so the whole image is visible."""
        cw, ch = self.canvas_size()
        if not cw or not ch or self.width() <= 0:
            return
        self._zoom = max(MIN_ZOOM, min(MAX_ZOOM,
                                       min(self.width() / cw, self.height() / ch)))
        self._center = QPointF(cw / 2.0, ch / 2.0)
        self.update()
        self.zoom_changed.emit(self._zoom)

    def zoom_to_actual(self) -> None:
        """100%: one output pixel per screen pixel."""
        self.set_zoom(1.0)

    def zoom_by(self, factor: float, anchor: QPointF | None = None) -> None:
        self.set_zoom(self._zoom * factor, anchor)

    # -- coordinate helpers ----------------------------------------------
    def _widget_centre(self) -> QPointF:
        return QPointF(self.width() / 2.0, self.height() / 2.0)

    def _canvas_to_widget(self, p: QPointF) -> QPointF:
        c = self._widget_centre()
        return QPointF(c.x() + (p.x() - self._center.x()) * self._zoom,
                       c.y() + (p.y() - self._center.y()) * self._zoom)

    def _widget_to_canvas(self, p: QPointF) -> QPointF:
        c = self._widget_centre()
        return QPointF(self._center.x() + (p.x() - c.x()) / self._zoom,
                       self._center.y() + (p.y() - c.y()) / self._zoom)

    def _clamp_center(self) -> None:
        """Keep at least part of the image on screen."""
        cw, ch = self.canvas_size()
        if not cw:
            return
        self._center.setX(max(0.0, min(float(cw), self._center.x())))
        self._center.setY(max(0.0, min(float(ch), self._center.y())))

    def _canvas_rect_in_widget(self) -> QRectF:
        cw, ch = self.canvas_size()
        tl = self._canvas_to_widget(QPointF(0, 0))
        br = self._canvas_to_widget(QPointF(cw, ch))
        return QRectF(tl, br)

    def _handle_x(self) -> float:
        """Widget x of the wipe handle."""
        r = self._canvas_rect_in_widget()
        return r.left() + r.width() * self._split

    # -- painting ---------------------------------------------------------
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().color(QPalette.ColorRole.Base))
        if not self.has_content:
            return

        area = self._canvas_rect_in_widget()
        if self._mode == MODE_ORIGINAL:
            self._blit(painter, self._original, area, self.rect(), magnified=True)
        elif self._mode == MODE_RESULT:
            self._blit(painter, self._result, area, self.rect(), magnified=False)
        else:
            split_x = self._handle_x()
            left = QRect(self.rect().left(), self.rect().top(),
                         max(0, int(split_x) - self.rect().left()), self.rect().height())
            right = QRect(int(split_x), self.rect().top(),
                          max(0, self.rect().right() - int(split_x) + 1),
                          self.rect().height())
            self._blit(painter, self._original, area, left, magnified=True)
            self._blit(painter, self._result, area, right, magnified=False)
            self._draw_handle(painter, split_x)
        painter.end()

    def _blit(self, painter: QPainter, image: QImage, area: QRectF,
              clip: QRect, magnified: bool) -> None:
        """Draw the visible part of `image` into `clip`.

        `area` is where the whole canvas would land in widget coordinates. Only
        the intersection with `clip` is drawn, and only the source pixels behind
        that intersection are read — which is the whole point of this file.
        """
        target = area.intersected(QRectF(clip))
        if target.isEmpty() or image is None:
            return

        cw, ch = self.canvas_size()
        # Widget -> canvas -> this image's own pixel grid.
        img_per_canvas = image.width() / cw if cw else 1.0
        tl = self._widget_to_canvas(target.topLeft())
        br = self._widget_to_canvas(target.bottomRight())
        source = QRectF(tl.x() * img_per_canvas, tl.y() * img_per_canvas,
                        (br.x() - tl.x()) * img_per_canvas,
                        (br.y() - tl.y()) * img_per_canvas)
        source = source.intersected(QRectF(image.rect()))
        if source.isEmpty():
            return

        # Smooth only when minifying, where nearest-neighbour would alias badly.
        # When magnifying, honesty wins: show the pixels that are really there,
        # unless the user asked for a smoothed original to compare against.
        effective = self._zoom * (1.0 / img_per_canvas if img_per_canvas else 1.0)
        smooth = effective < 1.0 or (magnified and self._filter == "smooth")
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, smooth)
        painter.save()
        painter.setClipRect(clip)
        painter.drawImage(target, image, source)
        painter.restore()

    def _draw_handle(self, painter: QPainter, x: float) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pal = self.palette()
        # A dark hairline under a light one, so the handle stays visible over
        # both a blown-out sky and a black shadow.
        painter.setPen(QPen(QColor(0, 0, 0, 140), 3.0))
        painter.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        painter.setPen(QPen(pal.color(QPalette.ColorRole.HighlightedText), 1.0))
        painter.drawLine(QPointF(x, 0), QPointF(x, self.height()))

        r = me.metrics.gu(0.55)
        centre = QPointF(x, self.height() / 2.0)
        painter.setPen(QPen(QColor(0, 0, 0, 140), 3.0))
        painter.drawEllipse(centre, r, r)
        painter.setPen(QPen(pal.color(QPalette.ColorRole.HighlightedText), 1.5))
        painter.setBrush(pal.color(QPalette.ColorRole.Highlight))
        painter.drawEllipse(centre, r, r)

    # -- interaction ------------------------------------------------------
    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta:
            self.zoom_by(1.0015 ** delta, QPointF(event.position()))
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or not self.has_content:
            return
        near_handle = (self._mode == MODE_COMPARE
                       and abs(event.position().x() - self._handle_x()) <= HANDLE_GRAB)
        if near_handle:
            self._grabbing_handle = True
        else:
            self._panning = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        self._last_pos = event.position().toPoint()

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        if self._grabbing_handle:
            area = self._canvas_rect_in_widget()
            if area.width() > 0:
                self.set_split((pos.x() - area.left()) / area.width())
            return
        if self._panning:
            delta = pos.toPoint() - self._last_pos
            self._center -= QPointF(delta.x() / self._zoom, delta.y() / self._zoom)
            self._clamp_center()
            self._last_pos = pos.toPoint()
            self.update()
            return
        if self._mode == MODE_COMPARE and self.has_content:
            near = abs(pos.x() - self._handle_x()) <= HANDLE_GRAB
            self.setCursor(Qt.CursorShape.SplitHCursor if near
                           else Qt.CursorShape.OpenHandCursor)
        elif self.has_content:
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseReleaseEvent(self, event) -> None:
        self._panning = False
        self._grabbing_handle = False
        self.setCursor(Qt.CursorShape.OpenHandCursor if self.has_content
                       else Qt.CursorShape.ArrowCursor)

    def mouseDoubleClickEvent(self, event) -> None:
        # Toggle between fit and 100%, the two zoom levels anyone actually wants.
        self.fit() if self._zoom >= 1.0 else self.zoom_to_actual()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom_by(1.25)
        elif key == Qt.Key.Key_Minus:
            self.zoom_by(1 / 1.25)
        elif key == Qt.Key.Key_0:
            self.fit()
        elif key == Qt.Key.Key_1:
            self.zoom_to_actual()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event) -> None:
        # Keep the whole image visible while it is still zoomed out to fit;
        # once the user has zoomed in, leave their position alone.
        super().resizeEvent(event)
        cw, ch = self.canvas_size()
        if cw and event.oldSize().width() > 0:
            fit_zoom = min(event.oldSize().width() / cw, event.oldSize().height() / ch)
            if self._zoom <= fit_zoom * 1.001:
                self.fit()
