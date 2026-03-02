"""State tab for the detail panel.

Displays the full ZeldaGame state as an expandable QTreeWidget.  Nodes for
link, enemies[], items[], projectiles[], room, and objectives.  Each leaf
shows a field name and its value.

The ``extract_state_dict`` function converts a live ZeldaGame into a plain
nested dict so the tree can be populated without needing the emulator.
"""

from collections import OrderedDict
from enum import Enum
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QHeaderView,
    QAbstractItemView,
)

from triforce_debugger.state_differ import diff_state_dicts


# ── State extraction ─────────────────────────────────────────────────

def _enum_str(val: Any) -> str:
    """Return a human-friendly string for enum values, pass through others."""
    if isinstance(val, Enum):
        return f"{type(val).__name__}.{val.name}"
    return str(val)


def _safe_get(obj: Any, attr: str, default: Any = "N/A") -> Any:
    """Safely get an attribute, returning *default* on any error."""
    try:
        return getattr(obj, attr)
    except Exception:  # pylint: disable=broad-except
        return default


def _extract_position(pos) -> OrderedDict:
    """Extract (x, y) from a Position named-tuple or similar."""
    return OrderedDict([("x", int(pos[0])), ("y", int(pos[1]))])


def _extract_link(link) -> OrderedDict:
    """Extract Link state into a flat ordered dict."""
    d = OrderedDict()
    d["position"] = _extract_position(link.position)
    d["direction"] = _enum_str(link.direction)
    d["status"] = int(link.status)
    d["health"] = float(_safe_get(link, "health", 0))
    d["max_health"] = int(_safe_get(link, "max_health", 0))

    # Equipment
    d["sword"] = _enum_str(_safe_get(link, "sword", "N/A"))
    d["arrows"] = _enum_str(_safe_get(link, "arrows", "N/A"))
    d["boomerang"] = _enum_str(_safe_get(link, "boomerang", "N/A"))
    d["bow"] = bool(_safe_get(link, "bow", False))
    d["magic_rod"] = bool(_safe_get(link, "magic_rod", False))
    d["book"] = bool(_safe_get(link, "book", False))
    d["candle"] = _enum_str(_safe_get(link, "candle", "N/A"))
    d["potion"] = _enum_str(_safe_get(link, "potion", "N/A"))
    d["whistle"] = bool(_safe_get(link, "whistle", False))
    d["food"] = bool(_safe_get(link, "food", False))
    d["letter"] = bool(_safe_get(link, "letter", False))
    d["power_bracelet"] = bool(_safe_get(link, "power_bracelet", False))
    d["ring"] = _enum_str(_safe_get(link, "ring", "N/A"))
    d["raft"] = bool(_safe_get(link, "raft", False))
    d["ladder"] = bool(_safe_get(link, "ladder", False))
    d["magic_key"] = bool(_safe_get(link, "magic_key", False))
    d["selected_equipment"] = _enum_str(_safe_get(link, "selected_equipment", "N/A"))

    # Stats
    d["rupees"] = int(_safe_get(link, "rupees", 0))
    d["bombs"] = int(_safe_get(link, "bombs", 0))
    d["bomb_max"] = int(_safe_get(link, "bomb_max", 0))
    d["keys"] = int(_safe_get(link, "keys", 0))
    d["magic_shield"] = bool(_safe_get(link, "magic_shield", False))

    # Dungeon items
    d["compass"] = bool(_safe_get(link, "compass", False))
    d["map"] = bool(_safe_get(link, "map", False))
    d["triforce_pieces"] = int(_safe_get(link, "triforce_pieces", 0))

    # Status flags
    d["has_beams"] = bool(_safe_get(link, "has_beams", False))
    d["is_blocking"] = bool(_safe_get(link, "is_blocking", False))

    return d


def _extract_enemy(enemy) -> OrderedDict:
    """Extract a single Enemy into an ordered dict."""
    d = OrderedDict()
    d["id"] = _enum_str(enemy.id)
    d["index"] = int(enemy.index)
    d["position"] = _extract_position(enemy.position)
    d["direction"] = _enum_str(enemy.direction)
    d["health"] = int(enemy.health)
    d["stun_timer"] = int(enemy.stun_timer)
    d["spawn_state"] = int(enemy.spawn_state)
    d["status"] = int(enemy.status)
    d["is_dying"] = bool(enemy.is_dying)
    d["is_active"] = bool(enemy.is_active)
    d["is_stunned"] = bool(enemy.is_stunned)
    return d


def _extract_item(item) -> OrderedDict:
    """Extract a single Item into an ordered dict."""
    d = OrderedDict()
    d["id"] = _enum_str(item.id)
    d["index"] = int(item.index)
    d["position"] = _extract_position(item.position)
    d["timer"] = int(item.timer)
    return d


def _extract_projectile(proj) -> OrderedDict:
    """Extract a single Projectile into an ordered dict."""
    d = OrderedDict()
    d["id"] = _enum_str(proj.id)
    d["index"] = int(proj.index)
    d["position"] = _extract_position(proj.position)
    d["blockable"] = bool(proj.blockable)
    return d


def _extract_room(room) -> OrderedDict:
    """Extract Room state (metadata only — no tile arrays)."""
    d = OrderedDict()
    d["full_location"] = str(room.full_location)
    d["is_loaded"] = bool(room.is_loaded)
    d["cave_tile"] = str(room.cave_tile) if room.cave_tile else "None"
    return d


def _extract_objective(objective) -> OrderedDict:
    """Extract Objective state."""
    d = OrderedDict()
    d["kind"] = _enum_str(objective.kind)
    targets = objective.targets
    d["target_count"] = len(targets) if targets else 0
    next_rooms = objective.next_rooms
    d["next_room_count"] = len(next_rooms) if next_rooms else 0
    return d


def extract_state_dict(state) -> OrderedDict:
    """Convert a ZeldaGame state into a plain nested OrderedDict for display.

    Designed to be called while the state is still live.  All values are
    primitive types (int, float, str, bool) or nested OrderedDicts / lists
    so the tree widget can render them without touching the emulator.
    """
    d = OrderedDict()

    # Top-level game info
    game_info = OrderedDict()
    game_info["level"] = int(_safe_get(state, "level", 0))
    game_info["location"] = hex(int(_safe_get(state, "location", 0)))
    game_info["in_cave"] = bool(_safe_get(state, "in_cave", False))
    game_info["full_location"] = str(_safe_get(state, "full_location", "N/A"))
    game_info["game_over"] = bool(_safe_get(state, "game_over", False))
    game_info["frames"] = int(_safe_get(state, "frames", 0))
    d["game"] = game_info

    # Link
    link = _safe_get(state, "link", None)
    if link is not None and link != "N/A":
        d["link"] = _extract_link(link)
    else:
        d["link"] = OrderedDict([("error", "unavailable")])

    # Enemies
    enemies = _safe_get(state, "enemies", [])
    if enemies and enemies != "N/A":
        d["enemies"] = [_extract_enemy(e) for e in enemies]
    else:
        d["enemies"] = []

    # Items
    items = _safe_get(state, "items", [])
    if items and items != "N/A":
        d["items"] = [_extract_item(i) for i in items]
    else:
        d["items"] = []

    # Projectiles
    projectiles = _safe_get(state, "projectiles", [])
    if projectiles and projectiles != "N/A":
        d["projectiles"] = [_extract_projectile(p) for p in projectiles]
    else:
        d["projectiles"] = []

    # Room
    room = _safe_get(state, "room", None)
    if room is not None and room != "N/A":
        d["room"] = _extract_room(room)
    else:
        d["room"] = OrderedDict([("error", "unavailable")])

    return d


# ── Diff-highlighting colours ─────────────────────────────────────────

CHANGED_BLUE = QBrush(QColor(60, 120, 220))   # Persistent: value changed
FLASH_YELLOW = QBrush(QColor(200, 180, 50))    # Momentary: just changed
DEFAULT_FG = QBrush()                           # Reset to default


# ── Widget ────────────────────────────────────────────────────────────

class StateTab(QWidget):
    """State detail tab showing the full game state as an expandable tree."""

    FLASH_DURATION_MS = 300

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("state_tab")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._header = QLabel("Game State")
        self._header.setObjectName("state_header")
        layout.addWidget(self._header)

        self._tree = QTreeWidget()
        self._tree.setObjectName("state_tree")
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(["Field", "Value"])
        self._tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._tree.setAlternatingRowColors(True)

        header = self._tree.header()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

        layout.addWidget(self._tree, stretch=1)

        # Current and previous state dicts for diff computation
        self._current_state: OrderedDict | None = None
        self._previous_state: OrderedDict | None = None

        # Paths that are currently "changed" (colored blue)
        self._changed_paths: set = set()

        # Paths currently flashing (yellow) — cleared after timer
        self._flash_paths: set = set()
        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._clear_flash)  # pylint: disable=no-member

    # ── Public API ────────────────────────────────────────────

    def update_state(self, state) -> None:
        """Update the tree with a live ZeldaGame state.

        Extracts the state into a dict, then rebuilds the tree with diff
        highlighting against the previous state.
        """
        state_dict = extract_state_dict(state)
        self._set_state_dict(state_dict, diff_against=self._current_state)

    def update_state_dict(self, state_dict: OrderedDict) -> None:
        """Update the tree from a pre-extracted state dict."""
        self._set_state_dict(state_dict, diff_against=self._current_state)

    def show_step_state(self, state_dict: OrderedDict,
                        step_number: int = 0,
                        prev_state_dict: OrderedDict | None = None) -> None:
        """Switch to showing a historical step's state.

        *state_dict* should be the pre-extracted dict for that step.
        *prev_state_dict* is the state from the step before, for diff highlighting.
        """
        self._header.setText(f"Step #{step_number} State")
        self._set_state_dict(state_dict, diff_against=prev_state_dict)

    def show_live(self) -> None:
        """Switch header back to live mode label."""
        self._header.setText("Game State")

    def clear(self) -> None:
        """Clear the tree and reset state."""
        self._tree.clear()
        self._current_state = None
        self._previous_state = None
        self._changed_paths.clear()
        self._flash_paths.clear()
        self._header.setText("Game State")

    @property
    def current_state(self) -> OrderedDict | None:
        """The currently displayed state dict (for testing)."""
        return self._current_state

    @property
    def changed_paths(self) -> set:
        """Paths currently highlighted as changed (for testing)."""
        return set(self._changed_paths)

    @property
    def flash_paths(self) -> set:
        """Paths currently flashing yellow (for testing)."""
        return set(self._flash_paths)

    # ── Internals ─────────────────────────────────────────────

    def _set_state_dict(self, state_dict: OrderedDict,
                        diff_against: OrderedDict | None = None) -> None:
        """Rebuild the tree from a state dict, preserving expansion state."""
        expanded = self._save_expanded()
        self._previous_state = diff_against
        self._current_state = state_dict

        # Compute diff; accumulate changed paths (blue persists until next change)
        new_changes = diff_state_dicts(diff_against, state_dict)
        self._flash_paths = new_changes - self._changed_paths
        self._changed_paths = self._changed_paths | new_changes

        self._tree.clear()
        self._populate_tree(self._tree.invisibleRootItem(), state_dict)
        self._apply_diff_colors(self._tree.invisibleRootItem())
        self._restore_expanded(expanded)

        # Start flash timer if there are newly changed paths
        if self._flash_paths:
            self._flash_timer.start(self.FLASH_DURATION_MS)

    def _populate_tree(self, parent, data, prefix: str = "") -> None:
        """Recursively add items to the tree from a dict or list."""
        if isinstance(data, OrderedDict):
            for key, value in data.items():
                path = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (OrderedDict, list)):
                    node = QTreeWidgetItem(parent, [str(key), ""])
                    node.setData(0, Qt.ItemDataRole.UserRole, path)
                    if isinstance(value, list):
                        node.setText(1, f"[{len(value)}]")
                    self._populate_tree(node, value, path)
                else:
                    leaf = QTreeWidgetItem(parent, [str(key), str(value)])
                    leaf.setData(0, Qt.ItemDataRole.UserRole, path)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                path = f"{prefix}[{i}]"
                if isinstance(item, OrderedDict):
                    label = item.get("id", str(i))
                    node = QTreeWidgetItem(parent, [f"[{i}] {label}", ""])
                    node.setData(0, Qt.ItemDataRole.UserRole, path)
                    self._populate_tree(node, item, path)
                else:
                    leaf = QTreeWidgetItem(parent, [f"[{i}]", str(item)])
                    leaf.setData(0, Qt.ItemDataRole.UserRole, path)

    def _save_expanded(self) -> set:
        """Save the set of expanded node paths."""
        expanded = set()
        self._walk_items(self._tree.invisibleRootItem(), expanded)
        return expanded

    def _walk_items(self, parent, expanded: set) -> None:
        """Recursively collect paths of expanded items."""
        for i in range(parent.childCount()):
            child = parent.child(i)
            path = child.data(0, Qt.ItemDataRole.UserRole)
            if child.isExpanded() and path:
                expanded.add(path)
            self._walk_items(child, expanded)

    def _restore_expanded(self, expanded: set) -> None:
        """Restore expansion state from a set of paths."""
        self._expand_items(self._tree.invisibleRootItem(), expanded)

    def _expand_items(self, parent, expanded: set) -> None:
        """Recursively expand items whose paths are in the set."""
        for i in range(parent.childCount()):
            child = parent.child(i)
            path = child.data(0, Qt.ItemDataRole.UserRole)
            if path in expanded:
                child.setExpanded(True)
            self._expand_items(child, expanded)

    def _apply_diff_colors(self, parent) -> None:
        """Apply blue/yellow foreground colours to changed leaf items."""
        for i in range(parent.childCount()):
            child = parent.child(i)
            path = child.data(0, Qt.ItemDataRole.UserRole)
            if path in self._flash_paths:
                child.setForeground(1, FLASH_YELLOW)
            elif path in self._changed_paths:
                child.setForeground(1, CHANGED_BLUE)
            self._apply_diff_colors(child)

    def _clear_flash(self) -> None:
        """Timer callback: convert flash-yellow items to persistent blue."""
        self._recolor_flash(self._tree.invisibleRootItem())
        self._flash_paths.clear()

    def _recolor_flash(self, parent) -> None:
        """Walk the tree and recolor flash paths to blue."""
        for i in range(parent.childCount()):
            child = parent.child(i)
            path = child.data(0, Qt.ItemDataRole.UserRole)
            if path in self._flash_paths:
                child.setForeground(1, CHANGED_BLUE)
            self._recolor_flash(child)
