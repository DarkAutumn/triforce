"""Game view widget — renders scaled NES frames via QPainter."""

import numpy as np
from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QImage, QPainter, QColor
from PySide6.QtWidgets import QWidget


class GameView(QWidget):
    """Widget that displays an RGB numpy array (NES frame) scaled to fit."""

    # NES native resolution
    NES_WIDTH = 256
    NES_HEIGHT = 240

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("game_view")
        self.setMinimumSize(256, 240)
        self._frame_image: QImage | None = None

    def set_frame(self, rgb_array: np.ndarray):
        """Set the current frame from an RGB numpy array (H, W, 3) uint8."""
        if rgb_array is None:
            self._frame_image = None
            self.update()
            return

        h, w, channels = rgb_array.shape
        assert channels == 3, f"Expected 3 channels, got {channels}"

        # Ensure contiguous uint8 data for QImage
        data = np.ascontiguousarray(rgb_array, dtype=np.uint8)

        # QImage.Format_RGB888: 3 bytes per pixel, R-G-B order
        image = QImage(data.data, w, h, w * 3, QImage.Format.Format_RGB888)

        # Copy the data so the QImage owns it (numpy array may be transient)
        self._frame_image = image.copy()
        self.update()

    @property
    def frame_image(self) -> QImage | None:
        """The current QImage being displayed (for testing)."""
        return self._frame_image

    def paintEvent(self, _event):  # pylint: disable=invalid-name
        """Paint the frame scaled to fit the widget, preserving aspect ratio."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)

        if self._frame_image is None:
            painter.fillRect(self.rect(), QColor(0, 0, 0))
            painter.setPen(QColor(128, 128, 128))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No frame")
            painter.end()
            return

        # Scale to fit widget while preserving aspect ratio
        target_rect = self._scaled_rect()
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        painter.drawImage(target_rect, self._frame_image)
        painter.end()

    def _scaled_rect(self) -> QRect:
        """Compute the target rect that fits the frame in the widget with correct aspect ratio."""
        if self._frame_image is None:
            return self.rect()

        img_w = self._frame_image.width()
        img_h = self._frame_image.height()
        widget_w = self.width()
        widget_h = self.height()

        scale = min(widget_w / img_w, widget_h / img_h)
        scaled_w = int(img_w * scale)
        scaled_h = int(img_h * scale)

        x = (widget_w - scaled_w) // 2
        y = (widget_h - scaled_h) // 2
        return QRect(x, y, scaled_w, scaled_h)
