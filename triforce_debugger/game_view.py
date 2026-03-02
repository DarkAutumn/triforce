"""Game view widget — renders scaled NES frames with optional tile overlays."""

from enum import Flag, auto

import numpy as np
from PySide6.QtCore import Qt, QRect, QRectF
from PySide6.QtGui import QImage, QPainter, QColor, QFont, QPen
from PySide6.QtWidgets import QWidget


class OverlayFlags(Flag):
    """Bit-flags for which overlays are active. Multiple can be combined."""
    NONE = 0
    WAVEFRONT = auto()
    TILE_IDS = auto()
    WALKABILITY = auto()
    COORDINATES = auto()


# NES tile grid constants
_GRID_WIDTH = 32
_GRID_HEIGHT = 22
_NES_TILE_PX = 8          # one tile = 8 NES pixels
_NES_TOP_HUD_PX = 56      # HUD height above the tile area


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
        self._overlays: OverlayFlags = OverlayFlags.NONE
        self._game_state = None  # ZeldaGame, set each step for overlay data

    # ── Public API ─────────────────────────────────────────────

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

    def set_game_state(self, state):
        """Set the ZeldaGame state used by overlays (room tiles, wavefront, walkable)."""
        self._game_state = state
        if self._overlays != OverlayFlags.NONE:
            self.update()

    def set_overlay(self, flag: OverlayFlags, enabled: bool):
        """Enable or disable a specific overlay flag."""
        if enabled:
            self._overlays |= flag
        else:
            self._overlays &= ~flag
        self.update()

    @property
    def overlays(self) -> OverlayFlags:
        """Currently active overlay flags."""
        return self._overlays

    @property
    def frame_image(self) -> QImage | None:
        """The current QImage being displayed (for testing)."""
        return self._frame_image

    # ── Painting ───────────────────────────────────────────────

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

        # Draw overlays on top of the frame
        if self._overlays != OverlayFlags.NONE and self._game_state is not None:
            self._paint_overlays(painter, target_rect)

        painter.end()

    def _paint_overlays(self, painter: QPainter, target_rect: QRect):
        """Draw active overlays over the game frame."""
        state = self._game_state

        # Compute per-pixel scale from NES coords to widget coords
        scale_x = target_rect.width() / self.NES_WIDTH
        scale_y = target_rect.height() / self.NES_HEIGHT

        tile_w = _NES_TILE_PX * scale_x
        tile_h = _NES_TILE_PX * scale_y

        # The tile grid starts 1 tile left of the frame origin and _NES_TOP_HUD_PX below it
        origin_x = target_rect.x() - tile_w
        origin_y = target_rect.y() + _NES_TOP_HUD_PX * scale_y

        # Choose text color: white for dungeons/caves, black for overworld
        is_dark = state.level != 0 or getattr(state, 'in_cave', False)
        text_color = QColor(255, 255, 255) if is_dark else QColor(0, 0, 0)
        grid_color = QColor(0, 0, 0, 120) if not is_dark else QColor(255, 255, 255, 120)

        font_size = max(6, int(min(tile_w, tile_h) * 0.5))
        font = QFont("monospace", font_size)
        font.setStyleHint(QFont.StyleHint.Monospace)
        painter.setFont(font)

        grid_pen = QPen(grid_color, 1)

        for tile_x in range(_GRID_WIDTH):
            for tile_y in range(_GRID_HEIGHT):
                rx = origin_x + tile_x * tile_w
                ry = origin_y + tile_y * tile_h
                cell = QRectF(rx, ry, tile_w, tile_h)

                # Grid lines
                painter.setPen(grid_pen)
                painter.drawRect(cell)

                # Collect text from the highest-priority active overlay
                text = self._overlay_text(state, tile_x, tile_y)
                if text:
                    painter.setPen(text_color)
                    painter.drawText(cell, Qt.AlignmentFlag.AlignCenter, text)

    def _overlay_text(self, state, tile_x: int, tile_y: int) -> str:
        """Return overlay text for one tile cell based on active flags.

        When multiple overlays are active the first match in priority order wins.
        """
        if OverlayFlags.WAVEFRONT in self._overlays:
            wf = getattr(state, 'wavefront', None)
            if wf is not None:
                val = wf.get((tile_x, tile_y), None)
                return f"{val:02X}" if val is not None else ""

        if OverlayFlags.TILE_IDS in self._overlays:
            room = getattr(state, 'room', None)
            if room is not None:
                tiles = room.tiles
                if tile_x < tiles.shape[0] and tile_y < tiles.shape[1]:
                    return f"{int(tiles[tile_x, tile_y]):02X}"

        if OverlayFlags.WALKABILITY in self._overlays:
            room = getattr(state, 'room', None)
            if room is not None:
                walkable = room.walkable
                if tile_x < walkable.shape[0] and tile_y < walkable.shape[1]:
                    return "X" if walkable[tile_x, tile_y] else ""

        if OverlayFlags.COORDINATES in self._overlays:
            return f"{tile_x:02X}\n{tile_y:02X}"

        return ""

    # ── Geometry helpers ───────────────────────────────────────

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
