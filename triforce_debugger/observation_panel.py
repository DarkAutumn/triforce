"""Observation panel — displays network input image, entity list, booleans, directional circles."""

import math

import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPainter, QImage, QColor, QPen, QPolygonF, QFont
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea

from triforce.zelda_enums import Direction
from triforce.observation_wrapper import ENTITY_SLOTS, ENTITY_TYPE_NAMES


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


# ── Small arrow widget ────────────────────────────────────────

class SmallArrowWidget(QWidget):
    """A small circle with a directional arrow showing relative entity position."""

    RADIUS = 12
    ARROW_COLOR = QColor(255, 0, 0)
    ARROWHEAD_SIZE = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self._vector = np.array([0.0, 0.0])
        size = self.RADIUS * 2 + 4
        self.setFixedSize(size, size)

    def set_vector(self, rel_x: float, rel_y: float):
        """Set the direction vector from link to entity (already normalized to [-1,1])."""
        self._vector = np.array([float(rel_x), float(rel_y)], dtype=np.float64)
        self.update()

    def paintEvent(self, _event):  # pylint: disable=invalid-name
        """Draw the circle with directional arrow."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx = self.width() / 2
        cy = self.height() / 2
        center = QPointF(cx, cy)

        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.drawEllipse(center, self.RADIUS, self.RADIUS)

        mag = math.sqrt(self._vector[0] ** 2 + self._vector[1] ** 2)
        if mag > 0.01:
            norm = self._vector / mag
            scale = min(mag, 1.0)
            _draw_arrow(painter, center, norm, scale,
                        self.RADIUS, self.ARROW_COLOR, self.ARROWHEAD_SIZE)

        painter.end()


# ── Entity row widget ─────────────────────────────────────────

class EntityRowWidget(QWidget):
    """Single entity row: direction arrow + type name + properties."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(4)

        self.arrow = SmallArrowWidget()
        layout.addWidget(self.arrow)

        self.name_label = QLabel("")
        self.name_label.setMinimumWidth(90)
        font = QFont()
        font.setPointSize(8)
        font.setBold(True)
        self.name_label.setFont(font)
        layout.addWidget(self.name_label)

        self.props_label = QLabel("")
        props_font = QFont()
        props_font.setPointSize(7)
        self.props_label.setFont(props_font)
        layout.addWidget(self.props_label)
        layout.addStretch()

    def set_entity(self, type_name, rel_x, rel_y, health, stun, hurts, killable):
        """Update the row with entity data."""
        self.arrow.set_vector(rel_x, rel_y)
        self.name_label.setText(type_name)
        props = []
        if health > 0.01:
            props.append(f"HP:{health * 15:.0f}")
        if stun > 0.01:
            props.append(f"Stun:{stun:.2f}")
        if hurts > 0.5:
            props.append("\u26a0")  # ⚠
        if killable > 0.5:
            props.append("\u2694")  # ⚔
        self.props_label.setText(" ".join(props))

    def clear_entity(self):
        """Hide this row's data."""
        self.arrow.set_vector(0, 0)
        self.name_label.setText("")
        self.props_label.setText("")


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

        center = _paint_labeled_circle(painter, self.width(), self.RADIUS, self._label)

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


# ── Shared circle painting ────────────────────────────────────

def _paint_labeled_circle(painter: QPainter, width: int, radius: float,
                          label: str, bold: bool = False) -> QPointF:
    """Draw a labeled circle and return its center point."""
    font = painter.font()
    font.setPointSize(7)
    font.setBold(bold)
    painter.setFont(font)
    painter.setPen(QColor(0, 0, 0))
    painter.drawText(QRectF(0, 0, width, 14),
                     Qt.AlignmentFlag.AlignCenter, label)

    cx = width / 2
    cy = 14 + radius + 2
    center = QPointF(cx, cy)
    painter.setPen(QPen(QColor(0, 0, 0), 1))
    painter.drawEllipse(center, radius, radius)
    return center


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
    """Full observation panel: network input image, entity list, directional circles, booleans."""

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

        # Directional circles: Objective + Source (immediately under image)
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
        for name in ("Enemies", "Beams", "Low HP", "Full HP", "Clock"):
            indicator = BooleanIndicator(name)
            self.bool_indicators[name] = indicator
            bool_layout.addWidget(indicator)
        layout.addLayout(bool_layout)

        # Entity list header
        entity_header = QLabel("Entities")
        entity_header.setObjectName("entity_header")
        entity_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hfont = entity_header.font()
        hfont.setBold(True)
        hfont.setPointSize(8)
        entity_header.setFont(hfont)
        layout.addWidget(entity_header)

        # Entity rows in a fixed-size scroll area
        entity_container = QWidget()
        entity_layout = QVBoxLayout(entity_container)
        entity_layout.setContentsMargins(0, 0, 0, 0)
        entity_layout.setSpacing(0)

        self.entity_rows: list[EntityRowWidget] = []
        for _ in range(ENTITY_SLOTS):
            row = EntityRowWidget()
            row.setVisible(False)
            self.entity_rows.append(row)
            entity_layout.addWidget(row)
        entity_layout.addStretch()

        self._entity_scroll = QScrollArea()
        self._entity_scroll.setObjectName("entity_scroll")
        self._entity_scroll.setWidget(entity_container)
        self._entity_scroll.setWidgetResizable(True)
        self._entity_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._entity_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # Fixed height: show ~4 entity rows without expanding the panel
        self._entity_scroll.setFixedHeight(140)
        layout.addWidget(self._entity_scroll, stretch=1)

    def update_observation(self, obs: dict):
        """Update all sub-widgets from an observation dict."""
        if obs is None:
            return

        # Network input image
        if "image" in obs:
            self.obs_image.set_image(obs["image"])

        # Entity list
        if "entities" in obs:
            entities = obs["entities"]
            entity_types = obs.get("entity_types")
            if hasattr(entities, 'cpu'):
                entities = entities.cpu().numpy()
            if entity_types is not None and hasattr(entity_types, 'cpu'):
                entity_types = entity_types.cpu().numpy()

            for i, row in enumerate(self.entity_rows):
                if i >= entities.shape[0]:
                    row.setVisible(False)
                    continue

                presence = float(entities[i, 0])
                if presence < 0.5:
                    row.setVisible(False)
                    continue

                type_id = int(entity_types[i]) if entity_types is not None else 0
                type_name = ENTITY_TYPE_NAMES.get(type_id, f"Unknown({type_id})")
                row.set_entity(
                    type_name=type_name,
                    rel_x=float(entities[i, 1]),
                    rel_y=float(entities[i, 2]),
                    health=float(entities[i, 5]),
                    stun=float(entities[i, 6]),
                    hurts=float(entities[i, 7]),
                    killable=float(entities[i, 8]),
                )
                row.setVisible(True)

        # Directional circles
        if "information" in obs:
            info = obs["information"]
            self.objective_circle.set_directions(
                _directions_from_info(info, 0))
            self.source_circle.set_directions(
                _directions_from_info(info, 6))

            # Boolean indicators (indices 10-14)
            self.bool_indicators["Enemies"].set_active(_info_bool(info, 10))
            self.bool_indicators["Beams"].set_active(_info_bool(info, 11))
            self.bool_indicators["Low HP"].set_active(_info_bool(info, 12))
            self.bool_indicators["Full HP"].set_active(_info_bool(info, 13))
            self.bool_indicators["Clock"].set_active(_info_bool(info, 14))


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
