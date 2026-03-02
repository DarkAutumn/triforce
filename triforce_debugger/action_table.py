"""Action probability table widget for the Triforce Debugger.

Displays model action probabilities as directional arrow vectors (MOVE, SWORD, BEAMS)
plus a data grid with ACTION, DIRECTION, PROBABILITY columns.
"""

import math

import numpy as np
import torch
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)

from triforce.zelda_enums import ActionKind, Direction


MASKED_TEXT = "[masked]"
GREY = QColor(128, 128, 128)

# Arrow colors per action kind
ACTION_COLORS = {
    ActionKind.MOVE: QColor(0, 180, 0),
    ActionKind.SWORD: QColor(200, 0, 0),
    ActionKind.BEAMS: QColor(0, 100, 255),
}

DIRECTION_VECTORS = {
    Direction.N: np.array([0.0, -1.0]),
    Direction.S: np.array([0.0, 1.0]),
    Direction.E: np.array([1.0, 0.0]),
    Direction.W: np.array([-1.0, 0.0]),
}


class ProbabilityArrowWidget(QWidget):
    """Draws directional arrows for a single action kind (MOVE/SWORD/BEAMS).

    Each cardinal direction gets an arrow whose length = probability.
    Arrows with prob < 1% or masked are not drawn.
    """

    RADIUS = 28
    ARROWHEAD_SIZE = 6

    def __init__(self, action_kind: ActionKind, parent=None):
        super().__init__(parent)
        self._kind = action_kind
        self._color = ACTION_COLORS.get(action_kind, QColor(200, 200, 200))
        # {Direction: (prob, masked)}
        self._probs: dict[Direction, tuple[float, bool]] = {}
        size = self.RADIUS * 2 + 4
        self.setFixedSize(size, size + 16)

    def set_probabilities(self, probs: dict[Direction, tuple[float, bool]]):
        """Set per-direction probabilities. Each value is (prob, is_masked)."""
        self._probs = probs
        self.update()

    def paintEvent(self, _event):  # pylint: disable=invalid-name
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Label
        font = painter.font()
        font.setPointSize(7)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(0, 0, 0))
        painter.drawText(QRectF(0, 0, self.width(), 14),
                         Qt.AlignmentFlag.AlignCenter, self._kind.value)

        # Circle
        cx = self.width() / 2
        cy = 14 + self.RADIUS + 2
        center = QPointF(cx, cy)
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.drawEllipse(center, self.RADIUS, self.RADIUS)

        # Arrows
        for direction, (prob, masked) in self._probs.items():
            if masked or prob < 0.01:
                continue
            vec = DIRECTION_VECTORS.get(direction)
            if vec is None:
                continue
            scale = float(np.clip(prob, 0.05, 1.0))
            end_x = cx + vec[0] * self.RADIUS * scale
            end_y = cy + vec[1] * self.RADIUS * scale
            end = QPointF(end_x, end_y)

            painter.setPen(QPen(self._color, 2))
            painter.drawLine(center, end)

            # Arrowhead
            angle = math.atan2(-vec[1], vec[0]) + math.pi
            left = QPointF(end_x + self.ARROWHEAD_SIZE * math.cos(angle - math.pi / 6),
                           end_y - self.ARROWHEAD_SIZE * math.sin(angle - math.pi / 6))
            right = QPointF(end_x + self.ARROWHEAD_SIZE * math.cos(angle + math.pi / 6),
                            end_y - self.ARROWHEAD_SIZE * math.sin(angle + math.pi / 6))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self._color)
            painter.drawPolygon(QPolygonF([end, left, right]))

        painter.end()


class ActionTable(QWidget):
    """Probability arrow vectors + data grid + value estimate."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("action_table")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Arrow widgets for MOVE / SWORD / BEAMS
        arrow_layout = QHBoxLayout()
        arrow_layout.setSpacing(4)
        self._arrow_widgets: dict[ActionKind, ProbabilityArrowWidget] = {}
        for kind in (ActionKind.MOVE, ActionKind.SWORD, ActionKind.BEAMS):
            w = ProbabilityArrowWidget(kind)
            self._arrow_widgets[kind] = w
            arrow_layout.addWidget(w)
        layout.addLayout(arrow_layout)

        # Table
        self.table = QTableWidget(0, 3)
        self.table.setObjectName("action_prob_table")
        self.table.setHorizontalHeaderLabels(["ACTION", "DIRECTION", "PROBABILITY"])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.verticalHeader().setVisible(False)

        h_header = self.table.horizontalHeader()
        h_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        layout.addWidget(self.table, stretch=1)

        # Value label
        self.value_label = QLabel("Value: —")
        self.value_label.setObjectName("value_label")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.value_label)

    def update_probabilities(self, probabilities, action_mask_desc=None):
        """Update arrows and table with new probability data."""
        if probabilities is None:
            self.table.setRowCount(0)
            self.value_label.setText("Value: —")
            for w in self._arrow_widgets.values():
                w.set_probabilities({})
            return

        # Extract value
        value = probabilities.get('value', None)
        if value is not None:
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.value_label.setText(f"Value: {value:.4f}")
        else:
            self.value_label.setText("Value: —")

        # Build set of allowed (action, direction) pairs for mask lookup
        allowed = set()
        if action_mask_desc is not None:
            for action_kind, directions in action_mask_desc:
                for direction in directions:
                    allowed.add((action_kind, direction))

        # Collect rows and arrow data
        rows = []
        arrow_data: dict[ActionKind, dict[Direction, tuple[float, bool]]] = {
            k: {} for k in self._arrow_widgets
        }

        for key, value_list in probabilities.items():
            if key == 'value':
                continue
            action_kind = key
            for direction, prob in value_list:
                is_masked = action_mask_desc is not None and (action_kind, direction) not in allowed
                rows.append((action_kind, direction, prob, is_masked))
                if action_kind in arrow_data:
                    arrow_data[action_kind][direction] = (prob, is_masked)

        # Update arrow widgets
        for kind, widget in self._arrow_widgets.items():
            widget.set_probabilities(arrow_data.get(kind, {}))

        # Populate table
        self.table.setRowCount(len(rows))
        for i, (action_kind, direction, prob, is_masked) in enumerate(rows):
            action_item = QTableWidgetItem(action_kind.value)
            direction_item = QTableWidgetItem(direction.name)

            if is_masked:
                prob_item = QTableWidgetItem(MASKED_TEXT)
                prob_item.setForeground(GREY)
                action_item.setForeground(GREY)
                direction_item.setForeground(GREY)
            else:
                prob_item = QTableWidgetItem(f"{prob * 100:.1f}%")

            for item in (action_item, direction_item, prob_item):
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            self.table.setItem(i, 0, action_item)
            self.table.setItem(i, 1, direction_item)
            self.table.setItem(i, 2, prob_item)
