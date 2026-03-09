"""Game view widget — renders scaled NES frames with optional tile overlays."""

from enum import Flag, auto

import numpy as np
from PySide6.QtCore import Qt, QRect, QRectF, QPoint
from PySide6.QtGui import QImage, QPainter, QColor, QFont, QPen
from PySide6.QtWidgets import QWidget, QToolTip


class OverlayFlags(Flag):
    """Bit-flags for which overlays are active. Multiple can be combined."""
    NONE = 0
    WAVEFRONT = auto()
    TILE_IDS = auto()
    WALKABILITY = auto()


# NES tile grid constants (after emulator overscan clipping)
_VISIBLE_COLS = 30         # tile columns 1..30 (column 0 and 31 are clipped)
_VISIBLE_ROWS = 22
_NES_TILE_PX = 8           # one tile = 8 NES pixels
_NES_HUD_PX = 48 + 8       # HUD height + 1 tile offset in the 224px clipped frame


class GameView(QWidget):
    """Widget that displays an RGB numpy array (NES frame) scaled to fit."""

    # NES native resolution
    NES_WIDTH = 240
    NES_HEIGHT = 224

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("game_view")
        self.setMinimumSize(240, 224)
        self.setMouseTracking(True)
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
        """Draw active overlays into a transparent image and composite it over the frame."""
        state = self._game_state

        overlay = QImage(target_rect.size(), QImage.Format.Format_ARGB32_Premultiplied)
        overlay.fill(QColor(0, 0, 0, 0))

        op = QPainter(overlay)
        op.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        scale_x = target_rect.width() / self.NES_WIDTH
        scale_y = target_rect.height() / self.NES_HEIGHT

        tile_w = _NES_TILE_PX * scale_x
        tile_h = _NES_TILE_PX * scale_y

        origin_x = 0.0
        origin_y = _NES_HUD_PX * scale_y

        grid_color = QColor(255, 255, 255, 60)
        bg_color = QColor(0, 0, 0, 160)
        text_color = QColor(255, 255, 255)

        font_size = max(6, int(min(tile_w, tile_h) * 0.45))
        font = QFont("monospace", font_size)
        font.setStyleHint(QFont.StyleHint.Monospace)
        op.setFont(font)

        grid_pen = QPen(grid_color, 1)

        for col in range(_VISIBLE_COLS):
            tile_x = col + 1  # tile data index (skip clipped column 0)
            for tile_y in range(_VISIBLE_ROWS):
                rx = origin_x + col * tile_w
                ry = origin_y + tile_y * tile_h
                cell = QRectF(rx, ry, tile_w, tile_h)

                op.setPen(grid_pen)
                op.drawRect(cell)

                text = self._overlay_text(state, tile_x, tile_y)
                if text:
                    op.fillRect(cell, bg_color)
                    op.setPen(text_color)
                    op.drawText(cell, Qt.AlignmentFlag.AlignCenter, text)

        op.end()
        painter.drawImage(target_rect, overlay)

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

        return ""

    # ── Mouse hover tooltip ────────────────────────────────────

    def mouseMoveEvent(self, event):  # pylint: disable=invalid-name
        """Show tile info tooltip when hovering over the gameplay area."""
        tile = self._widget_pos_to_tile(event.pos())
        if tile is None:
            QToolTip.hideText()
            return

        tile_x, tile_y = tile
        state = self._game_state
        if state is None:
            QToolTip.hideText()
            return

        room = getattr(state, 'room', None)
        if room is None:
            QToolTip.hideText()
            return

        lines = [f"Tile: ({tile_x}, {tile_y})"]

        tiles = room.tiles
        if tile_x < tiles.shape[0] and tile_y < tiles.shape[1]:
            lines.append(f"Tile ID: {int(tiles[tile_x, tile_y]):02X}")

        wf = getattr(state, 'wavefront', None)
        if wf is not None:
            val = wf.get((tile_x, tile_y), None)
            lines.append(f"Wavefront: {val if val is not None else 'N/A'}")

        walkable = room.walkable
        if tile_x < walkable.shape[0] and tile_y < walkable.shape[1]:
            lines.append("Walkable" if walkable[tile_x, tile_y] else "Not Walkable")

        QToolTip.showText(event.globalPosition().toPoint(), "\n".join(lines), self)

    def _widget_pos_to_tile(self, pos: QPoint):
        """Convert a widget pixel position to (tile_x, tile_y), or None if outside gameplay."""
        if self._frame_image is None:
            return None

        target_rect = self._scaled_rect()
        if not target_rect.contains(pos):
            return None

        # Map widget pixel to NES pixel
        nes_x = (pos.x() - target_rect.x()) / target_rect.width() * self.NES_WIDTH
        nes_y = (pos.y() - target_rect.y()) / target_rect.height() * self.NES_HEIGHT

        # Must be below the HUD
        if nes_y < _NES_HUD_PX:
            return None

        col = int(nes_x / _NES_TILE_PX)
        row = int((nes_y - _NES_HUD_PX) / _NES_TILE_PX)

        # Column 0 is clipped off-screen; visible pixel 0 is tile column 1
        tile_x = col + 1
        tile_y = row

        if tile_x < 1 or tile_x > 30 or tile_y < 0 or tile_y >= _VISIBLE_ROWS:
            return None

        return tile_x, tile_y

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
