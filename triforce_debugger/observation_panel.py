"""Observation panel — displays network input image, vector arrows, booleans, directional circles."""

import math

import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPainter, QImage, QColor, QPen, QPolygonF
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel

from triforce.zelda_enums import Direction


# ── Observation image widget ──────────────────────────────────

class ObsImageWidget(QWidget):
    """Displays the 84×84 grayscale observation image scaled up for visibility."""

    SCALE_FACTOR = 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("obs_image")
        self._qimage: QImage | None = None
        self.setMinimumSize(84 * self.SCALE_FACTOR, 84 * self.SCALE_FACTOR)

    def set_image(self, image_tensor):
        """Set image from observation tensor.  Accepts (C, H, W) or (frames, C, H, W)."""
        if image_tensor is None:
            self._qimage = None
            self.update()
            return

        img = image_tensor
        if hasattr(img, 'cpu'):
            img = img.cpu().numpy()

        # Take the last frame if stacked: (frames, C, H, W) → (C, H, W)
        if img.ndim == 4:
            img = img[-1]

        # (C, H, W) → (H, W) for single-channel grayscale
        if img.ndim == 3 and img.shape[0] == 1:
            img = img.squeeze(0)

        # Normalize to 0-255 uint8
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        # Grayscale → RGB for QImage
        if img.ndim == 2:
            rgb = np.stack([img, img, img], axis=-1)
        else:
            rgb = np.moveaxis(img, 0, -1)  # (C, H, W) → (H, W, C)

        rgb = np.ascontiguousarray(rgb)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self._qimage = qimg.copy()
        self.update()

    @property
    def current_image(self) -> QImage | None:
        """The current QImage (for testing)."""
        return self._qimage

    def paintEvent(self, _event):  # pylint: disable=invalid-name
        """Paint the observation image scaled up."""
        painter = QPainter(self)
        if self._qimage is None:
            painter.fillRect(self.rect(), QColor(0, 0, 0))
            painter.setPen(QColor(128, 128, 128))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No obs")
        else:
            scaled_w = self._qimage.width() * self.SCALE_FACTOR
            scaled_h = self._qimage.height() * self.SCALE_FACTOR
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
            painter.drawImage(QRectF(0, 0, scaled_w, scaled_h), self._qimage)
        painter.end()


# ── Vector circle widget ─────────────────────────────────────

class VectorCircleWidget(QWidget):
    """A labeled circle with a directional arrow showing an entity's direction and distance."""

    RADIUS = 30
    ARROW_COLOR = QColor(255, 0, 0)
    ARROWHEAD_SIZE = 7

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self._label = label
        self._vector = np.array([0.0, 0.0])
        self._scale = 0.0
        size = self.RADIUS * 2 + 4
        self.setFixedSize(size, size + 16)

    @property
    def label(self) -> str:
        """The widget label text."""
        return self._label

    def set_vector(self, vector: np.ndarray, scale: float):
        """Set the direction vector and scale (0=far/invisible, 1=close/full arrow)."""
        self._vector = np.asarray(vector, dtype=np.float64)
        self._scale = float(np.clip(scale, 0.0, 1.0))
        self.update()

    def paintEvent(self, _event):  # pylint: disable=invalid-name
        """Draw the labeled circle with directional arrow."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Label at top
        font = painter.font()
        font.setPointSize(7)
        painter.setFont(font)
        painter.setPen(QColor(0, 0, 0))
        painter.drawText(QRectF(0, 0, self.width(), 14),
                         Qt.AlignmentFlag.AlignCenter, self._label)

        # Circle
        cx = self.width() / 2
        cy = 14 + self.RADIUS + 2
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.drawEllipse(QPointF(cx, cy), self.RADIUS, self.RADIUS)

        # Arrow
        if self._scale > 0.01 and (self._vector[0] != 0 or self._vector[1] != 0):
            _draw_arrow(painter, QPointF(cx, cy), self._vector, self._scale,
                        self.RADIUS, self.ARROW_COLOR, self.ARROWHEAD_SIZE)

        painter.end()


# ── Directional circle widget ────────────────────────────────

class DirectionalCircleWidget(QWidget):
    """A labeled circle with N/S/E/W directional arrows for objective/source."""

    RADIUS = 30
    ARROW_COLOR = QColor(255, 0, 0)
    ARROWHEAD_SIZE = 7

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self._label = label
        self._directions: list[Direction] = []
        size = self.RADIUS * 2 + 4
        self.setFixedSize(size, size + 16)

    @property
    def label(self) -> str:
        """The widget label text."""
        return self._label

    @property
    def directions(self) -> list[Direction]:
        """Current active directions."""
        return self._directions

    def set_directions(self, directions: list[Direction]):
        """Set which cardinal directions to display arrows for."""
        self._directions = list(directions)
        self.update()

    def paintEvent(self, _event):  # pylint: disable=invalid-name
        """Draw the labeled circle with directional arrows."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Label at top
        font = painter.font()
        font.setPointSize(7)
        painter.setFont(font)
        painter.setPen(QColor(0, 0, 0))
        painter.drawText(QRectF(0, 0, self.width(), 14),
                         Qt.AlignmentFlag.AlignCenter, self._label)

        # Circle
        cx = self.width() / 2
        cy = 14 + self.RADIUS + 2
        center = QPointF(cx, cy)
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.drawEllipse(center, self.RADIUS, self.RADIUS)

        direction_vectors = {
            Direction.N: np.array([0.0, -1.0]),
            Direction.S: np.array([0.0, 1.0]),
            Direction.E: np.array([1.0, 0.0]),
            Direction.W: np.array([-1.0, 0.0]),
        }

        for d in self._directions:
            vec = direction_vectors.get(d)
            if vec is not None:
                _draw_arrow(painter, center, vec, 1.0,
                            self.RADIUS, self.ARROW_COLOR, self.ARROWHEAD_SIZE)

        # Center dot when active
        if self._directions:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0))
            painter.drawEllipse(center, 4, 4)

        painter.end()


# ── Boolean indicator ─────────────────────────────────────────

class BooleanIndicator(QLabel):
    """A text label colored dark blue (active) or white (inactive)."""

    ACTIVE_COLOR = "rgb(0, 0, 160)"
    INACTIVE_COLOR = "rgb(255, 255, 255)"

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self._active = False
        self._update_style()
        font = self.font()
        font.setPointSize(8)
        self.setFont(font)

    @property
    def active(self) -> bool:
        """Whether the indicator is currently active."""
        return self._active

    def set_active(self, active: bool):
        """Set active/inactive state and update color."""
        self._active = active
        self._update_style()

    def _update_style(self):
        color = self.ACTIVE_COLOR if self._active else self.INACTIVE_COLOR
        self.setStyleSheet(f"color: {color};")


# ── Shared arrow drawing ─────────────────────────────────────

def _draw_arrow(painter: QPainter, center: QPointF, vector: np.ndarray,
                scale: float, radius: float, color: QColor, head_size: float):
    """Draw an arrow from center in the direction of vector, scaled by scale * radius."""
    scale = float(np.clip(scale, 0.05, 1.0))
    end_x = center.x() + vector[0] * radius * scale
    end_y = center.y() + vector[1] * radius * scale
    end = QPointF(end_x, end_y)

    painter.setPen(QPen(color, 2))
    painter.drawLine(center, end)

    # Arrowhead
    angle = math.atan2(-vector[1], vector[0]) + math.pi
    left = QPointF(end_x + head_size * math.cos(angle - math.pi / 6),
                   end_y - head_size * math.sin(angle - math.pi / 6))
    right = QPointF(end_x + head_size * math.cos(angle + math.pi / 6),
                    end_y - head_size * math.sin(angle + math.pi / 6))

    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(color)
    painter.drawPolygon(QPolygonF([end, left, right]))


# ── Main observation panel ────────────────────────────────────

class ObservationPanel(QWidget):
    """Full observation panel: network input image, vector arrows, directional circles, booleans."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("observation_panel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Title
        title = QLabel("Observation")
        title.setObjectName("obs_title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)

        # Network input image
        self.obs_image = ObsImageWidget()
        layout.addWidget(self.obs_image)

        # Vector circles: 4 enemies, 2 projectiles, 2 items in a 4×2 grid
        vector_grid = QGridLayout()
        vector_grid.setSpacing(2)
        self.vector_widgets: dict[str, VectorCircleWidget] = {}
        labels = [
            "Enemy 1", "Enemy 2",
            "Enemy 3", "Enemy 4",
            "Proj 1", "Proj 2",
            "Item 1", "Item 2",
        ]
        for i, label in enumerate(labels):
            widget = VectorCircleWidget(label)
            self.vector_widgets[label] = widget
            vector_grid.addWidget(widget, i // 2, i % 2)
        layout.addLayout(vector_grid)

        # Directional circles: Objective + Source
        dir_layout = QHBoxLayout()
        dir_layout.setSpacing(4)
        self.objective_circle = DirectionalCircleWidget("Objective")
        self.source_circle = DirectionalCircleWidget("Source")
        dir_layout.addWidget(self.objective_circle)
        dir_layout.addWidget(self.source_circle)
        layout.addLayout(dir_layout)

        # Boolean indicators
        bool_layout = QHBoxLayout()
        bool_layout.setSpacing(6)
        self.bool_indicators: dict[str, BooleanIndicator] = {}
        for name in ("Enemies", "Beams", "Low HP", "Full HP"):
            indicator = BooleanIndicator(name)
            self.bool_indicators[name] = indicator
            bool_layout.addWidget(indicator)
        layout.addLayout(bool_layout)

        layout.addStretch()

    def update_observation(self, obs: dict):
        """Update all sub-widgets from an observation dict."""
        if obs is None:
            return

        # Network input image
        if "image" in obs:
            self.obs_image.set_image(obs["image"])

        # Vector widgets — enemy features
        self._update_vector("Enemy 1", obs, "enemy_features", 0)
        self._update_vector("Enemy 2", obs, "enemy_features", 1)
        self._update_vector("Enemy 3", obs, "enemy_features", 2)
        self._update_vector("Enemy 4", obs, "enemy_features", 3)

        # Vector widgets — projectile features
        self._update_vector("Proj 1", obs, "projectile_features", 0)
        self._update_vector("Proj 2", obs, "projectile_features", 1)

        # Vector widgets — item features
        self._update_vector("Item 1", obs, "item_features", 0)
        self._update_vector("Item 2", obs, "item_features", 1)

        # Directional circles
        if "information" in obs:
            info = obs["information"]
            self.objective_circle.set_directions(
                _directions_from_info(info, 0))
            self.source_circle.set_directions(
                _directions_from_info(info, 6))

            # Boolean indicators (indices 10-13)
            self.bool_indicators["Enemies"].set_active(_info_bool(info, 10))
            self.bool_indicators["Beams"].set_active(_info_bool(info, 11))
            self.bool_indicators["Low HP"].set_active(_info_bool(info, 12))
            self.bool_indicators["Full HP"].set_active(_info_bool(info, 13))

    def _update_vector(self, widget_name: str, obs: dict, feature_key: str, index: int):
        """Update a vector widget from observation features."""
        if feature_key not in obs:
            return
        features = obs[feature_key]
        if hasattr(features, 'cpu'):
            features = features.cpu().numpy()
        vector = features[index, 2:4]
        scale = float(1.0 - features[index, 1])
        self.vector_widgets[widget_name].set_vector(vector, scale)


def _directions_from_info(info, offset: int) -> list[Direction]:
    """Extract cardinal directions from the information vector at given offset."""
    directions = []
    vals = info if not hasattr(info, 'cpu') else info.cpu()
    if vals[offset + 0] > 0:
        directions.append(Direction.N)
    if vals[offset + 1] > 0:
        directions.append(Direction.S)
    if vals[offset + 2] > 0:
        directions.append(Direction.E)
    if vals[offset + 3] > 0:
        directions.append(Direction.W)
    return directions


def _info_bool(info, index: int) -> bool:
    """Extract a boolean from the information vector."""
    val = info[index]
    if hasattr(val, 'item'):
        val = val.item()
    return val > 0
