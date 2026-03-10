"""Tests for triforce_debugger.state_tab — headless (QT_QPA_PLATFORM=offscreen)."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from collections import OrderedDict
from PySide6.QtWidgets import QApplication
import pytest

from triforce_debugger.state_tab import StateTab, extract_state_dict


# ── Ensure a QApplication exists ──────────────────────────────────────
@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# ── Mock state objects ────────────────────────────────────────────────

class _MockPosition:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def __getitem__(self, idx):
        return (self._x, self._y)[idx]


class _MockDirection:
    """Mimics an enum member."""
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class _MockLink:
    def __init__(self):
        self.tile = _MockPosition(15, 10)
        self.position = _MockPosition(120, 80)
        self.direction = _MockDirection("N")
        self.status = 0
        self.health = 3.0
        self.max_health = 3
        self.sword = _MockDirection("WOOD")
        self.arrows = _MockDirection("NONE")
        self.boomerang = _MockDirection("NONE")
        self.bow = False
        self.magic_rod = False
        self.book = False
        self.candle = _MockDirection("NONE")
        self.potion = _MockDirection("NONE")
        self.whistle = False
        self.food = False
        self.letter = False
        self.power_bracelet = False
        self.ring = _MockDirection("NONE")
        self.raft = False
        self.ladder = False
        self.magic_key = False
        self.selected_equipment = _MockDirection("NONE")
        self.rupees = 42
        self.bombs = 3
        self.bomb_max = 8
        self.keys = 1
        self.magic_shield = False
        self.compass = True
        self.map = False
        self.triforce_pieces = 0
        self.has_beams = False
        self.is_blocking = False


class _MockEnemy:
    def __init__(self, idx, enemy_id, hp):
        self.index = idx
        self.id = _MockDirection(enemy_id)
        self.position = _MockPosition(100, 60)
        self.direction = _MockDirection("S")
        self.health = hp
        self.stun_timer = 0
        self.spawn_state = 0
        self.status = 0
        self.is_dying = False
        self.is_active = True
        self.is_stunned = False


class _MockItem:
    def __init__(self, idx, item_id):
        self.index = idx
        self.id = _MockDirection(item_id)
        self.position = _MockPosition(50, 50)
        self.timer = 255


class _MockProjectile:
    def __init__(self, idx, proj_id):
        self.index = idx
        self.id = _MockDirection(proj_id)
        self.position = _MockPosition(80, 100)
        self.blockable = True


class _MockRoom:
    def __init__(self):
        self.full_location = "(1, 0x72, False)"
        self.is_loaded = True
        self.cave_tile = None


class _MockObjective:
    def __init__(self):
        self.kind = _MockDirection("MOVE")
        self.targets = {1, 2, 3}
        self.next_rooms = {4, 5}


class _MockState:
    """Minimal ZeldaGame mock with the fields extract_state_dict needs."""
    def __init__(self):
        self.level = 1
        self.location = 0x72
        self.in_cave = False
        self.full_location = "(1, 0x72, False)"
        self.game_over = False
        self.frames = 1234
        self.link = _MockLink()
        self.enemies = [_MockEnemy(1, "Stalfos", 2), _MockEnemy(2, "Keese", 1)]
        self.items = [_MockItem(3, "Heart")]
        self.projectiles = [_MockProjectile(4, "Arrow")]
        self.room = _MockRoom()


# ── Helper ────────────────────────────────────────────────────────────

def _build_sample_dict() -> OrderedDict:
    """Build a state dict from mock objects."""
    return extract_state_dict(_MockState())


def _count_tree_items(tree_widget) -> int:
    """Count all items (recursively) in a QTreeWidget."""
    count = 0
    root = tree_widget.invisibleRootItem()
    stack = [root]
    while stack:
        node = stack.pop()
        for i in range(node.childCount()):
            count += 1
            stack.append(node.child(i))
    return count


# ── Tests ─────────────────────────────────────────────────────────────

class TestExtractStateDict:
    def test_returns_ordered_dict(self):
        d = _build_sample_dict()
        assert isinstance(d, OrderedDict)

    def test_top_level_keys(self):
        d = _build_sample_dict()
        assert list(d.keys()) == ["game", "link", "enemies", "items", "projectiles", "room"]

    def test_game_section(self):
        d = _build_sample_dict()
        game = d["game"]
        assert game["level"] == 1
        assert game["location"] == "0x72"
        assert game["in_cave"] is False
        assert game["game_over"] is False
        assert game["frames"] == 1234

    def test_link_section(self):
        d = _build_sample_dict()
        link = d["link"]
        assert link["tile"] == "(15, 10)"
        assert link["position"] == "(120, 80)"
        assert link["health"] == 3.0
        assert link["max_health"] == 3
        assert link["rupees"] == 42
        assert link["bombs"] == 3
        assert link["keys"] == 1

    def test_enemies_section(self):
        d = _build_sample_dict()
        enemies = d["enemies"]
        assert len(enemies) == 2
        assert enemies[0]["health"] == 2
        assert enemies[1]["health"] == 1

    def test_items_section(self):
        d = _build_sample_dict()
        items = d["items"]
        assert len(items) == 1
        assert items[0]["timer"] == 255

    def test_projectiles_section(self):
        d = _build_sample_dict()
        projs = d["projectiles"]
        assert len(projs) == 1
        assert projs[0]["blockable"] is True

    def test_room_section(self):
        d = _build_sample_dict()
        room = d["room"]
        assert room["is_loaded"] is True
        assert room["cave_tile"] == "None"

    def test_empty_enemies(self):
        state = _MockState()
        state.enemies = []
        d = extract_state_dict(state)
        assert d["enemies"] == []


class TestStateTabCreation:
    def test_widget_creates(self):
        tab = StateTab()
        assert tab.objectName() == "state_tab"
        assert tab._tree is not None
        assert tab._header.text() == "Game State"

    def test_initial_state_empty(self):
        tab = StateTab()
        assert tab.current_state is None
        assert tab._tree.topLevelItemCount() == 0


class TestStateTabUpdateDict:
    def test_populates_tree(self):
        tab = StateTab()
        d = _build_sample_dict()
        tab.update_state_dict(d)
        assert tab.current_state is not None
        # Top-level nodes: game, link, enemies, items, projectiles, room
        assert tab._tree.topLevelItemCount() == 6

    def test_tree_has_children(self):
        tab = StateTab()
        d = _build_sample_dict()
        tab.update_state_dict(d)
        total = _count_tree_items(tab._tree)
        # Should have many items (game fields, link fields, enemies, etc.)
        assert total > 20

    def test_game_node_children(self):
        tab = StateTab()
        d = _build_sample_dict()
        tab.update_state_dict(d)
        # "game" is the first top-level node
        game_node = tab._tree.topLevelItem(0)
        assert game_node.text(0) == "game"
        # game has: level, location, in_cave, full_location, game_over, frames
        assert game_node.childCount() == 6

    def test_enemies_list_node(self):
        tab = StateTab()
        d = _build_sample_dict()
        tab.update_state_dict(d)
        # "enemies" is the third top-level node (index 2)
        enemies_node = tab._tree.topLevelItem(2)
        assert enemies_node.text(0) == "enemies"
        assert enemies_node.text(1) == "[2]"  # count display
        assert enemies_node.childCount() == 2

    def test_enemy_child_has_id_label(self):
        tab = StateTab()
        d = _build_sample_dict()
        tab.update_state_dict(d)
        enemies_node = tab._tree.topLevelItem(2)
        first_enemy = enemies_node.child(0)
        # Label should contain the enemy id
        assert "Stalfos" in first_enemy.text(0)

    def test_leaf_value_display(self):
        tab = StateTab()
        d = _build_sample_dict()
        tab.update_state_dict(d)
        game_node = tab._tree.topLevelItem(0)
        # "level" is the first child of game
        level_leaf = game_node.child(0)
        assert level_leaf.text(0) == "level"
        assert level_leaf.text(1) == "1"

    def test_link_position_nested(self):
        tab = StateTab()
        d = _build_sample_dict()
        tab.update_state_dict(d)
        link_node = tab._tree.topLevelItem(1)
        # First child of link is "tile" (a string leaf)
        tile_node = link_node.child(0)
        assert tile_node.text(0) == "tile"
        assert tile_node.text(1) == "(15, 10)"
        # Second child is "position" (also a string leaf now)
        pos_node = link_node.child(1)
        assert pos_node.text(0) == "position"
        assert pos_node.text(1) == "(120, 80)"


class TestStateTabShowStep:
    def test_show_step_changes_header(self):
        tab = StateTab()
        d = _build_sample_dict()
        tab.show_step_state(d, step_number=42)
        assert tab._header.text() == "Step #42 State"

    def test_show_live_restores_header(self):
        tab = StateTab()
        d = _build_sample_dict()
        tab.show_step_state(d, step_number=10)
        tab.show_live()
        assert tab._header.text() == "Game State"


class TestStateTabClear:
    def test_clear_resets_everything(self):
        tab = StateTab()
        d = _build_sample_dict()
        tab.update_state_dict(d)
        tab.clear()
        assert tab.current_state is None
        assert tab._tree.topLevelItemCount() == 0
        assert tab._header.text() == "Game State"


class TestExpansionPreservation:
    def test_expansion_preserved_on_update(self):
        tab = StateTab()
        d = _build_sample_dict()
        tab.update_state_dict(d)

        # Expand the "game" node
        game_node = tab._tree.topLevelItem(0)
        game_node.setExpanded(True)
        assert game_node.isExpanded()

        # Update again with same data
        tab.update_state_dict(d)

        # "game" should still be expanded
        game_node = tab._tree.topLevelItem(0)
        assert game_node.isExpanded()
