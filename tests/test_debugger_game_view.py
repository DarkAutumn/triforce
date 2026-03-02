"""Tests for QT-04: Game view widget — renders NES frames via QPainter."""

import sys

import numpy as np
from PySide6.QtCore import QRect
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication


def get_app():
    """Get or create a QApplication instance."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _make_game_view():
    """Create a GameView for testing."""
    from triforce_debugger.game_view import GameView  # pylint: disable=import-outside-toplevel
    _app = get_app()
    return GameView()


# ── Frame setting ─────────────────────────────────────────────


def test_initial_state_no_frame():
    """GameView starts with no frame image."""
    view = _make_game_view()
    assert view.frame_image is None
    view.close()


def test_set_frame_creates_qimage():
    """Setting a frame creates a QImage with correct dimensions."""
    view = _make_game_view()
    frame = np.zeros((240, 256, 3), dtype=np.uint8)
    view.set_frame(frame)
    assert view.frame_image is not None
    assert isinstance(view.frame_image, QImage)
    assert view.frame_image.width() == 256
    assert view.frame_image.height() == 240
    view.close()


def test_set_frame_preserves_pixel_data():
    """Pixel data survives the numpy-to-QImage conversion."""
    view = _make_game_view()
    frame = np.zeros((240, 256, 3), dtype=np.uint8)
    # Set a known pixel: red at (0, 0)
    frame[0, 0] = [255, 0, 0]
    # Set a known pixel: green at (100, 50)
    frame[100, 50] = [0, 255, 0]
    # Set a known pixel: blue at (239, 255)
    frame[239, 255] = [0, 0, 255]

    view.set_frame(frame)
    img = view.frame_image

    # QImage.pixel returns ARGB packed as uint32
    r, g, b, _ = img.pixelColor(0, 0).getRgb()
    assert (r, g, b) == (255, 0, 0), f"Expected red, got ({r}, {g}, {b})"

    r, g, b, _ = img.pixelColor(50, 100).getRgb()
    assert (r, g, b) == (0, 255, 0), f"Expected green, got ({r}, {g}, {b})"

    r, g, b, _ = img.pixelColor(255, 239).getRgb()
    assert (r, g, b) == (0, 0, 255), f"Expected blue, got ({r}, {g}, {b})"
    view.close()


def test_set_frame_non_nes_dimensions():
    """GameView accepts frames of arbitrary dimensions."""
    view = _make_game_view()
    frame = np.ones((100, 200, 3), dtype=np.uint8) * 128
    view.set_frame(frame)
    assert view.frame_image.width() == 200
    assert view.frame_image.height() == 100
    view.close()


def test_set_frame_none_clears():
    """Setting frame to None clears the current image."""
    view = _make_game_view()
    frame = np.zeros((240, 256, 3), dtype=np.uint8)
    view.set_frame(frame)
    assert view.frame_image is not None

    view.set_frame(None)
    assert view.frame_image is None
    view.close()


# ── Aspect ratio scaling ─────────────────────────────────────


def test_scaled_rect_fits_in_widget():
    """Scaled rect should fit within the widget bounds."""
    view = _make_game_view()
    view.resize(960, 896)
    frame = np.zeros((240, 256, 3), dtype=np.uint8)
    view.set_frame(frame)

    rect = view._scaled_rect()  # pylint: disable=protected-access
    assert rect.x() >= 0
    assert rect.y() >= 0
    assert rect.x() + rect.width() <= view.width()
    assert rect.y() + rect.height() <= view.height()
    view.close()


def test_scaled_rect_preserves_aspect_ratio():
    """Scaled rect should preserve the NES aspect ratio."""
    view = _make_game_view()
    view.resize(800, 600)
    frame = np.zeros((240, 256, 3), dtype=np.uint8)
    view.set_frame(frame)

    rect = view._scaled_rect()  # pylint: disable=protected-access
    original_ratio = 256 / 240
    scaled_ratio = rect.width() / rect.height()
    assert abs(original_ratio - scaled_ratio) < 0.02, \
        f"Aspect ratio mismatch: {original_ratio:.3f} vs {scaled_ratio:.3f}"
    view.close()


def test_scaled_rect_no_frame_returns_widget_rect():
    """With no frame, _scaled_rect returns the widget rect."""
    view = _make_game_view()
    view.resize(400, 300)

    rect = view._scaled_rect()  # pylint: disable=protected-access
    assert rect == view.rect()
    view.close()


# ── Paint event ───────────────────────────────────────────────


def test_paint_without_crash():
    """Painting with and without a frame should not crash."""
    view = _make_game_view()
    view.resize(400, 300)
    view.show()

    # Paint with no frame
    view.repaint()

    # Paint with a frame
    frame = np.zeros((240, 256, 3), dtype=np.uint8)
    view.set_frame(frame)
    view.repaint()

    view.close()


def test_minimum_size():
    """GameView has a minimum size of 256x240 (NES native)."""
    view = _make_game_view()
    assert view.minimumWidth() == 256
    assert view.minimumHeight() == 240
    view.close()


def test_object_name():
    """GameView has the expected object name for identification."""
    view = _make_game_view()
    assert view.objectName() == "game_view"
    view.close()
