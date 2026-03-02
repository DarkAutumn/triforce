"""Tests for triforce_debugger.state_differ — headless (QT_QPA_PLATFORM=offscreen).

Tests the pure diff engine and the StateTab diff-highlighting integration.
"""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from collections import OrderedDict

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
import pytest

from triforce_debugger.state_differ import diff_state_dicts
from triforce_debugger.state_tab import (
    StateTab, CHANGED_BLUE, FLASH_YELLOW, DEFAULT_FG,
)


# ── Ensure a QApplication exists ──────────────────────────────────────
@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# ── Helpers ───────────────────────────────────────────────────────────

def _make_state(**overrides) -> OrderedDict:
    """Build a minimal state dict with optional overrides."""
    game = OrderedDict([
        ("level", 1),
        ("location", "0x72"),
        ("frames", 100),
    ])
    link = OrderedDict([
        ("position", OrderedDict([("x", 120), ("y", 80)])),
        ("health", 3.0),
        ("max_health", 3),
        ("direction", "N"),
        ("rupees", 42),
    ])
    enemies = [
        OrderedDict([
            ("id", "Stalfos"),
            ("index", 1),
            ("position", OrderedDict([("x", 100), ("y", 60)])),
            ("health", 2),
        ]),
    ]
    d = OrderedDict([
        ("game", game),
        ("link", link),
        ("enemies", enemies),
    ])

    # Apply nested overrides like "link.health" or "game.frames"
    for dotpath, val in overrides.items():
        parts = dotpath.split(".")
        target = d
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = val

    return d


# ── Pure diff engine tests ────────────────────────────────────────────

class TestDiffStateDicts:
    def test_identical_returns_empty(self):
        s = _make_state()
        assert diff_state_dicts(s, s) == set()

    def test_none_old_returns_empty(self):
        s = _make_state()
        assert diff_state_dicts(None, s) == set()

    def test_none_new_returns_empty(self):
        s = _make_state()
        assert diff_state_dicts(s, None) == set()

    def test_both_none_returns_empty(self):
        assert diff_state_dicts(None, None) == set()

    def test_scalar_change_detected(self):
        old = _make_state()
        new = _make_state(**{"link.health": 2.0})
        result = diff_state_dicts(old, new)
        assert "link.health" in result

    def test_nested_dict_change(self):
        old = _make_state()
        new = _make_state(**{"link.position.x": 130})
        result = diff_state_dicts(old, new)
        assert "link.position.x" in result
        assert "link.position.y" not in result

    def test_multiple_changes(self):
        old = _make_state()
        new = _make_state(**{"link.health": 2.0, "game.frames": 200})
        result = diff_state_dicts(old, new)
        assert "link.health" in result
        assert "game.frames" in result
        # Unchanged fields should not be in result
        assert "link.rupees" not in result
        assert "game.level" not in result

    def test_list_item_change(self):
        old = _make_state()
        new = _make_state()
        new["enemies"][0]["health"] = 1
        result = diff_state_dicts(old, new)
        assert "enemies[0].health" in result

    def test_list_length_increase(self):
        old = _make_state()
        new = _make_state()
        new["enemies"].append(OrderedDict([
            ("id", "Keese"),
            ("index", 2),
            ("position", OrderedDict([("x", 50), ("y", 50)])),
            ("health", 1),
        ]))
        result = diff_state_dicts(old, new)
        # New item's leaves should all be marked
        assert "enemies[1].id" in result
        assert "enemies[1].health" in result

    def test_list_length_decrease(self):
        old = _make_state()
        old["enemies"].append(OrderedDict([
            ("id", "Keese"),
            ("index", 2),
            ("position", OrderedDict([("x", 50), ("y", 50)])),
            ("health", 1),
        ]))
        new = _make_state()
        result = diff_state_dicts(old, new)
        # Removed item's leaves should be marked
        assert "enemies[1].id" in result

    def test_no_false_positives(self):
        old = _make_state()
        new = _make_state()
        result = diff_state_dicts(old, new)
        assert len(result) == 0


class TestDiffEdgeCases:
    def test_empty_dicts(self):
        assert diff_state_dicts(OrderedDict(), OrderedDict()) == set()

    def test_key_added(self):
        old = OrderedDict([("a", 1)])
        new = OrderedDict([("a", 1), ("b", 2)])
        result = diff_state_dicts(old, new)
        assert "b" in result
        assert "a" not in result

    def test_key_removed(self):
        old = OrderedDict([("a", 1), ("b", 2)])
        new = OrderedDict([("a", 1)])
        result = diff_state_dicts(old, new)
        assert "b" in result

    def test_empty_list_to_populated(self):
        old = OrderedDict([("items", [])])
        new = OrderedDict([("items", [OrderedDict([("id", "Heart")])])])
        result = diff_state_dicts(old, new)
        assert "items[0].id" in result


# ── StateTab diff integration tests ──────────────────────────────────

class TestStateTabDiffHighlighting:
    def test_first_update_no_changes(self):
        tab = StateTab()
        s1 = _make_state()
        tab.update_state_dict(s1)
        # First update with no previous state — no changed paths
        assert tab.changed_paths == set()
        assert tab.flash_paths == set()

    def test_second_update_detects_changes(self):
        tab = StateTab()
        s1 = _make_state()
        tab.update_state_dict(s1)
        s2 = _make_state(**{"link.health": 2.0})
        tab.update_state_dict(s2)
        assert "link.health" in tab.changed_paths

    def test_flash_paths_are_subset_of_new_changes(self):
        tab = StateTab()
        s1 = _make_state()
        tab.update_state_dict(s1)
        s2 = _make_state(**{"link.health": 2.0})
        tab.update_state_dict(s2)
        # First change — flash includes everything new
        assert "link.health" in tab.flash_paths

    def test_persistent_change_not_reflashed(self):
        tab = StateTab()
        s1 = _make_state()
        tab.update_state_dict(s1)
        s2 = _make_state(**{"link.health": 2.0})
        tab.update_state_dict(s2)
        # Update again with health still changed, plus a new change
        s3 = _make_state(**{"link.health": 2.0, "link.rupees": 50})
        tab.update_state_dict(s3)
        # health is still changed (blue) but not flashing again
        assert "link.health" in tab.changed_paths
        assert "link.health" not in tab.flash_paths
        # rupees is newly changed → flashing
        assert "link.rupees" in tab.flash_paths

    def test_show_step_state_with_diff(self):
        tab = StateTab()
        s1 = _make_state()
        s2 = _make_state(**{"game.frames": 200})
        tab.show_step_state(s2, step_number=5, prev_state_dict=s1)
        assert "game.frames" in tab.changed_paths

    def test_clear_resets_diff_state(self):
        tab = StateTab()
        s1 = _make_state()
        tab.update_state_dict(s1)
        s2 = _make_state(**{"link.health": 2.0})
        tab.update_state_dict(s2)
        tab.clear()
        assert tab.changed_paths == set()
        assert tab.flash_paths == set()

    def test_flash_timer_clears_flash_paths(self):
        """Flash paths should be cleared after the timer fires."""
        tab = StateTab()
        s1 = _make_state()
        tab.update_state_dict(s1)
        s2 = _make_state(**{"link.health": 2.0})
        tab.update_state_dict(s2)
        assert len(tab.flash_paths) > 0
        # Manually fire the timer callback
        tab._clear_flash()
        assert tab.flash_paths == set()
        # Changed paths should still be there (blue)
        assert "link.health" in tab.changed_paths


class TestStateTabDiffColors:
    """Verify that tree items actually get the correct foreground brush."""

    def _find_item_by_path(self, tree, path):
        """Find a QTreeWidgetItem by its UserRole path."""
        root = tree.invisibleRootItem()
        return self._search(root, path)

    def _search(self, parent, path):
        from PySide6.QtCore import Qt
        for i in range(parent.childCount()):
            child = parent.child(i)
            if child.data(0, Qt.ItemDataRole.UserRole) == path:
                return child
            result = self._search(child, path)
            if result is not None:
                return result
        return None

    def test_changed_item_gets_blue(self):
        tab = StateTab()
        s1 = _make_state()
        tab.update_state_dict(s1)
        s2 = _make_state(**{"link.health": 2.0})
        tab.update_state_dict(s2)
        # Simulate flash timer expiry
        tab._clear_flash()
        item = self._find_item_by_path(tab._tree, "link.health")
        assert item is not None
        assert item.foreground(1) == CHANGED_BLUE

    def test_flash_item_gets_yellow(self):
        tab = StateTab()
        s1 = _make_state()
        tab.update_state_dict(s1)
        s2 = _make_state(**{"link.health": 2.0})
        tab.update_state_dict(s2)
        # Before flash timer fires, item should be yellow
        item = self._find_item_by_path(tab._tree, "link.health")
        assert item is not None
        assert item.foreground(1) == FLASH_YELLOW

    def test_unchanged_item_no_color(self):
        tab = StateTab()
        s1 = _make_state()
        tab.update_state_dict(s1)
        s2 = _make_state(**{"link.health": 2.0})
        tab.update_state_dict(s2)
        item = self._find_item_by_path(tab._tree, "link.rupees")
        assert item is not None
        assert item.foreground(1) == DEFAULT_FG

    def test_flash_to_blue_transition(self):
        tab = StateTab()
        s1 = _make_state()
        tab.update_state_dict(s1)
        s2 = _make_state(**{"link.health": 2.0})
        tab.update_state_dict(s2)
        item = self._find_item_by_path(tab._tree, "link.health")
        assert item.foreground(1) == FLASH_YELLOW
        tab._clear_flash()
        assert item.foreground(1) == CHANGED_BLUE
