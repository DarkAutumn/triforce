"""Step history ring buffer and list widget for time-travel debugging.

Stores step snapshots in a fixed-capacity deque so the user can click any
past step and have all panels update to that step's data.  The list widget
uses a QAbstractItemModel for virtual scrolling (only visible rows rendered).
"""

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

from PySide6.QtCore import Qt, QAbstractItemModel, QModelIndex, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QTreeView, QWidget, QVBoxLayout, QAbstractItemView, QHeaderView


# ── Colors for reward display ─────────────────────────────────────────

COLOR_POSITIVE = QColor(0, 160, 0)
COLOR_NEGATIVE = QColor(200, 0, 0)
COLOR_NEUTRAL = QColor(128, 128, 128)

# Sentinel internal-id value for top-level (step) rows.  Child rows use
# their parent's display-row + 1, which is always >= 1.
_TOP_LEVEL_ID = 0


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class StepEntry:
    """A single snapshot stored in the step history."""
    step_number: int
    action: Any               # ActionTaken
    reward: Any               # StepRewards
    observation: Any          # obs dict (tensors)
    state: Any                # ZeldaGame
    action_mask: Any          # Tensor
    action_probabilities: Any # OrderedDict from model
    terminated: bool
    truncated: bool
    frame: Any                # RGB numpy array (last game frame)
    action_mask_desc: Any = None  # list of (ActionKind, [Direction]) for masking display
    value: float = 0.0        # V(s) from the model's value head
    advantage: Optional[float] = None  # GAE advantage (computed retroactively)


# ── Ring buffer ───────────────────────────────────────────────────────

class StepHistory:
    """Ring buffer storing step snapshots for time-travel debugging.

    Backed by ``collections.deque(maxlen=MAX_STEPS)``.  Oldest entries are
    silently evicted when the buffer is full.
    """

    MAX_STEPS = 50_000

    def __init__(self, maxlen: Optional[int] = None):
        capacity = maxlen if maxlen is not None else self.MAX_STEPS
        self._buffer: deque[StepEntry] = deque(maxlen=capacity)

    # -- mutators ----------------------------------------------------------

    def append(self, entry: StepEntry) -> None:
        """Append a step entry, evicting the oldest if at capacity."""
        self._buffer.append(entry)

    def clear(self) -> None:
        """Remove all entries (e.g. on episode reset)."""
        self._buffer.clear()

    # -- accessors ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self._buffer)

    def __getitem__(self, index: int) -> StepEntry:
        """Retrieve by position (0 = oldest, -1 = newest)."""
        return self._buffer[index]

    def get_by_index(self, index: int) -> StepEntry:
        """Retrieve by position (0 = oldest, -1 = newest).

        Raises IndexError if out of range.
        """
        return self._buffer[index]

    @property
    def is_full(self) -> bool:
        """True when the buffer is at its maximum capacity."""
        return len(self._buffer) == self._buffer.maxlen

    @property
    def newest(self) -> Optional[StepEntry]:
        """Return the most recent entry, or None if empty."""
        return self._buffer[-1] if self._buffer else None

    @property
    def oldest(self) -> Optional[StepEntry]:
        """Return the oldest entry, or None if empty."""
        return self._buffer[0] if self._buffer else None


# ── Helpers ───────────────────────────────────────────────────────────

def format_action(action) -> str:
    """Format an action for display.  Handles ActionTaken objects and strings."""
    if hasattr(action, 'kind') and hasattr(action, 'direction'):
        return f"{action.kind.value} {action.direction.name}"
    return str(action)


def format_reward_value(reward) -> str:
    """Format a reward value for display as ``+0.050`` / ``-0.120``."""
    if hasattr(reward, 'value'):
        return f"{reward.value:+.3f}"
    if isinstance(reward, (int, float)):
        return f"{reward:+.3f}"
    return str(reward)


def reward_color(reward) -> QColor:
    """Return green / red / grey based on the sign of the reward."""
    val = reward.value if hasattr(reward, 'value') else reward
    if not isinstance(val, (int, float)):
        return COLOR_NEUTRAL
    if val > 0:
        return COLOR_POSITIVE
    if val < 0:
        return COLOR_NEGATIVE
    return COLOR_NEUTRAL


def _outcome_list(reward) -> list:
    """Return a list of individual outcomes from a reward, or []."""
    if hasattr(reward, '__iter__'):
        return list(reward)
    return []


def _outcome_color(value: float) -> QColor:
    """Return green/red/grey for a single outcome value."""
    if value > 0:
        return COLOR_POSITIVE
    if value < 0:
        return COLOR_NEGATIVE
    return COLOR_NEUTRAL


# ── Qt item model (two-level tree) ────────────────────────────────────

class StepHistoryModel(QAbstractItemModel):
    """Two-level tree model wrapping :class:`StepHistory`.

    Top-level rows are steps (newest first, i.e. display row 0 = newest).
    Child rows show the individual reward/penalty breakdown for each step.
    Only visible rows are materialised thanks to QTreeView virtual scrolling.
    """

    COLUMNS = ("Step", "Action", "Advantage", "Reward")

    def __init__(self, history: StepHistory, parent=None):
        super().__init__(parent)
        self._history = history

    @property
    def history(self) -> StepHistory:
        """The underlying ring buffer."""
        return self._history

    # ── Index helpers ─────────────────────────────────────────────────

    def _buf_index(self, display_row: int) -> int:
        """Convert display row (0 = newest) to buffer index (0 = oldest)."""
        return len(self._history) - 1 - display_row

    # ── QAbstractItemModel interface ──────────────────────────────────

    def index(self, row, column, parent=QModelIndex()):
        if not self.hasIndex(row, column, parent):
            return QModelIndex()
        if not parent.isValid():
            return self.createIndex(row, column, _TOP_LEVEL_ID)
        # Child row — encode parent's display-row + 1 as internal id
        return self.createIndex(row, column, parent.row() + 1)

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()
        internal = index.internalId()
        if internal == _TOP_LEVEL_ID:
            return QModelIndex()
        return self.createIndex(internal - 1, 0, _TOP_LEVEL_ID)

    def rowCount(self, parent=QModelIndex()):
        if not parent.isValid():
            return len(self._history)
        if parent.internalId() == _TOP_LEVEL_ID:
            buf_idx = self._buf_index(parent.row())
            try:
                entry = self._history[buf_idx]
            except IndexError:
                return 0
            return len(_outcome_list(entry.reward))
        return 0  # no grandchildren

    def columnCount(self, parent=QModelIndex()):  # pylint: disable=unused-argument
        return len(self.COLUMNS)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if index.internalId() == _TOP_LEVEL_ID:
            return self._step_data(index, role)
        return self._child_data(index, role)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            if 0 <= section < len(self.COLUMNS):
                return self.COLUMNS[section]
        return None

    # ── Data providers ────────────────────────────────────────────────

    def _step_data(self, index, role):
        """Return data for a top-level step row."""
        buf_idx = self._buf_index(index.row())
        try:
            entry = self._history[buf_idx]
        except IndexError:
            return None
        col = index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            adv_str = f"{entry.advantage:+.3f}" if entry.advantage is not None else ""
            display = {0: f"#{entry.step_number}", 1: format_action(entry.action),
                       2: adv_str, 3: format_reward_value(entry.reward)}
            return display.get(col)

        if role == Qt.ItemDataRole.ForegroundRole:
            if col == 2 and entry.advantage is not None:
                return _outcome_color(entry.advantage)
            if col == 3:
                return reward_color(entry.reward)

        return None

    def _child_data(self, index, role):
        """Return data for a child reward-outcome row."""
        parent_display_row = index.internalId() - 1
        buf_idx = self._buf_index(parent_display_row)
        try:
            entry = self._history[buf_idx]
        except IndexError:
            return None

        outcomes = _outcome_list(entry.reward)
        if index.row() >= len(outcomes):
            return None
        outcome = outcomes[index.row()]
        col = index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            display = {0: outcome.name, 3: f"{outcome.value:+.3f}"}
            return display.get(col)

        if role == Qt.ItemDataRole.ForegroundRole and col == 3:
            return _outcome_color(outcome.value)

        return None


# ── Widget ────────────────────────────────────────────────────────────

class StepHistoryWidget(QWidget):
    """Step history tree view with expandable reward rows and sticky scroll.

    Emits :pyattr:`step_selected` with the *buffer index* of the clicked step
    so that other panels can time-travel to it.
    """

    step_selected = Signal(int)   # buffer index of the selected step

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("step_history")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._history = StepHistory()
        self._model = StepHistoryModel(self._history)

        self._tree = QTreeView()
        self._tree.setObjectName("step_history_tree")
        self._tree.setModel(self._model)
        self._tree.setRootIsDecorated(True)
        self._tree.setItemsExpandable(True)
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._tree.setUniformRowHeights(True)

        header = self._tree.header()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)

        layout.addWidget(self._tree)

        # Sticky-scroll: stay at the top (newest) unless the user scrolls away
        self._sticky_scroll = True
        self._tree.verticalScrollBar().valueChanged.connect(self._on_scroll)

        # Forward selection changes
        self._tree.selectionModel().currentChanged.connect(self._on_current_changed)

    # ── Public API ────────────────────────────────────────────────────

    @property
    def history(self) -> StepHistory:
        """The underlying ring buffer."""
        return self._history

    @property
    def model(self) -> StepHistoryModel:
        """The Qt item model."""
        return self._model

    @property
    def tree(self) -> QTreeView:
        """The tree view widget."""
        return self._tree

    def append_step(self, entry: StepEntry) -> None:
        """Append a step and auto-scroll to it if sticky."""
        was_full = self._history.is_full

        if was_full:
            # Insertion + eviction — safest to reset the model
            self._model.beginResetModel()
            self._history.append(entry)
            self._model.endResetModel()
        else:
            # Pure insertion at display row 0
            self._model.beginInsertRows(QModelIndex(), 0, 0)
            self._history.append(entry)
            self._model.endInsertRows()

        if self._sticky_scroll:
            self._tree.scrollToTop()

    def recompute_advantages(self, gamma: float, lam: float) -> None:
        """Recompute GAE advantages for all steps in the buffer.

        Uses the standard GAE formula:
          δ_t = r_t + γ·V(s_{t+1})·mask - V(s_t)
          A_t = δ_t + γ·λ·mask·A_{t+1}
        where mask=0 if step t was terminal/truncated (episode boundary).
        """
        n = len(self._history)
        if n == 0:
            return

        last_gae = 0.0
        for t in reversed(range(n)):
            entry = self._history[t]
            reward = entry.reward.value if hasattr(entry.reward, 'value') else float(entry.reward)

            if t + 1 < n:
                next_entry = self._history[t + 1]
                next_value = next_entry.value
                mask = 0.0 if entry.terminated or entry.truncated else 1.0
            else:
                next_value = 0.0
                mask = 0.0  # no future info for the latest step

            delta = reward + gamma * next_value * mask - entry.value
            last_gae = delta + gamma * lam * mask * last_gae
            entry.advantage = last_gae

        # Notify the view that the advantage column changed
        if n > 0:
            adv_col = 2  # Advantage column index
            top_left = self._model.index(0, adv_col)
            bottom_right = self._model.index(n - 1, adv_col)
            self._model.dataChanged.emit(top_left, bottom_right)  # pylint: disable=no-member

    def clear_history(self) -> None:
        """Clear all steps (e.g. on episode reset)."""
        self._model.beginResetModel()
        self._history.clear()
        self._model.endResetModel()
        self._sticky_scroll = True

    # ── Internals ─────────────────────────────────────────────────────

    def _on_scroll(self, value):
        """Track whether the user has scrolled away from the newest step."""
        self._sticky_scroll = value <= 0

    def _on_current_changed(self, current, _previous):
        """Emit step_selected when a top-level step row is clicked."""
        if not current.isValid():
            return
        if current.internalId() != _TOP_LEVEL_ID:
            return
        buf_idx = self._model._buf_index(current.row())  # pylint: disable=protected-access
        self.step_selected.emit(buf_idx)
