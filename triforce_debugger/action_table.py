"""Action probability table widget for the Triforce Debugger.

Displays model action probabilities in a table with ACTION, DIRECTION, PROBABILITY
columns, plus a Value estimate label. Masked actions shown as "[masked]" in grey.
"""

import torch
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)


MASKED_TEXT = "[masked]"
GREY = QColor(128, 128, 128)


class ActionTable(QWidget):
    """Table displaying per-action probabilities and the value estimate."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("action_table")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Header label
        header = QLabel("Action Probabilities")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

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
        """Update the table with new probability data.

        Args:
            probabilities: OrderedDict from ModelSelector.get_probabilities().
                Keys: 'value' -> tensor, ActionKind -> [(Direction, float), ...]
            action_mask_desc: list of (ActionKind, [Direction]) tuples describing
                which actions are currently allowed. If None, no masking shown.
        """
        if probabilities is None:
            self.table.setRowCount(0)
            self.value_label.setText("Value: —")
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

        # Collect rows
        rows = []
        for key, value_list in probabilities.items():
            if key == 'value':
                continue
            action_kind = key
            for direction, prob in value_list:
                is_masked = action_mask_desc is not None and (action_kind, direction) not in allowed
                rows.append((action_kind, direction, prob, is_masked))

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
