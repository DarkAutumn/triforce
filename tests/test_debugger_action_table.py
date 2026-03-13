"""Tests for QT-11: Action probability table — formatting, masking, value display."""

import sys
from collections import OrderedDict
from enum import Enum

import torch
from PySide6.QtWidgets import QApplication


def get_app():
    """Get or create a QApplication instance."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


# Lightweight stand-ins matching the real enums' interface
class _ActionKind(Enum):
    MOVE = "MOVE"
    SWORD = "SWORD"
    BEAMS = "BEAMS"

class _Direction(Enum):
    N = 8
    S = 4
    W = 2
    E = 1


def _make_table():
    from triforce_debugger.action_table import ActionTable  # pylint: disable=import-outside-toplevel
    _app = get_app()
    return ActionTable()


def _make_probs(**kwargs):
    """Build a mock probability OrderedDict.

    Default: MOVE (N=0.45, S=0.30, W=0.15, E=0.10), value=0.342
    """
    probs = OrderedDict()
    probs['value'] = kwargs.get('value', torch.tensor(0.342))
    probs[_ActionKind.MOVE] = kwargs.get('move', [
        (_Direction.N, 0.45),
        (_Direction.S, 0.30),
        (_Direction.W, 0.15),
        (_Direction.E, 0.10),
    ])
    if 'sword' in kwargs:
        probs[_ActionKind.SWORD] = kwargs['sword']
    if 'beams' in kwargs:
        probs[_ActionKind.BEAMS] = kwargs['beams']
    return probs


# ── Widget creation ──────────────────────────────────────────

def test_table_creates_without_crash():
    """ActionTable instantiates headlessly."""
    table = _make_table()
    assert table.objectName() == "action_table"
    table.close()


def test_table_has_four_columns():
    """Table has ACTION, DIRECTION, PROBABILITY, UNMASKED columns."""
    table = _make_table()
    assert table.table.columnCount() == 4
    headers = [table.table.horizontalHeaderItem(i).text() for i in range(4)]
    assert headers == ["ACTION", "DIRECTION", "PROBABILITY", "UNMASKED"]
    table.close()


def test_table_starts_empty():
    """Table starts with no rows and default value text."""
    table = _make_table()
    assert table.table.rowCount() == 0
    assert table.value_label.text() == "Value: —"
    table.close()


def test_value_label_exists():
    """Value label is present with correct object name."""
    table = _make_table()
    assert table.value_label is not None
    assert table.value_label.objectName() == "value_label"
    table.close()


# ── Probability display ─────────────────────────────────────

def test_populate_move_only():
    """Populating with MOVE actions fills 4 rows."""
    table = _make_table()
    probs = _make_probs()
    table.update_probabilities(probs)

    assert table.table.rowCount() == 4
    # First row: MOVE, N, 45.0%
    assert table.table.item(0, 0).text() == "MOVE"
    assert table.table.item(0, 1).text() == "N"
    assert table.table.item(0, 2).text() == "45.0%"
    table.close()


def test_populate_multiple_action_kinds():
    """Multiple action kinds produce rows for all of them."""
    table = _make_table()
    probs = _make_probs(
        sword=[(_Direction.N, 0.08), (_Direction.S, 0.02),
               (_Direction.W, 0.05), (_Direction.E, 0.10)],
    )
    table.update_probabilities(probs)

    # 4 MOVE + 4 SWORD = 8
    assert table.table.rowCount() == 8
    # Verify SWORD rows — UNMASKED shows raw prob, PROBABILITY shows renormalized
    assert table.table.item(4, 0).text() == "SWORD"
    assert table.table.item(4, 1).text() == "N"
    assert table.table.item(4, 3).text() == "8.0%"   # raw unmasked
    table.close()


def test_probability_formatting():
    """UNMASKED column shows raw probabilities as 'XX.X%'."""
    table = _make_table()
    probs = _make_probs(move=[
        (_Direction.N, 0.001),
        (_Direction.S, 0.999),
        (_Direction.W, 0.0),
        (_Direction.E, 0.5),
    ])
    table.update_probabilities(probs)

    # UNMASKED column (3) shows raw values
    assert table.table.item(0, 3).text() == "0.1%"
    assert table.table.item(1, 3).text() == "99.9%"
    assert table.table.item(2, 3).text() == "0.0%"
    assert table.table.item(3, 3).text() == "50.0%"
    table.close()


def test_value_display_tensor():
    """Value from a tensor is formatted correctly."""
    table = _make_table()
    probs = _make_probs(value=torch.tensor(0.342))
    table.update_probabilities(probs)
    assert table.value_label.text() == "Value: 0.3420"
    table.close()


def test_value_display_negative():
    """Negative value displayed correctly."""
    table = _make_table()
    probs = _make_probs(value=torch.tensor(-1.5))
    table.update_probabilities(probs)
    assert table.value_label.text() == "Value: -1.5000"
    table.close()


# ── Masking ──────────────────────────────────────────────────

def test_masked_actions_show_masked_text():
    """Masked actions display '[masked]' in PROBABILITY; renormalized for unmasked."""
    from triforce_debugger.action_table import MASKED_TEXT  # pylint: disable=import-outside-toplevel
    table = _make_table()
    probs = _make_probs()
    # Only N and E are allowed
    mask_desc = [(_ActionKind.MOVE, [_Direction.N, _Direction.E])]
    table.update_probabilities(probs, action_mask_desc=mask_desc)

    # Row 0 (N): allowed, renormalized 0.45/(0.45+0.10) = 81.8%
    assert table.table.item(0, 2).text() == "81.8%"
    # Row 1 (S): masked
    assert table.table.item(1, 2).text() == MASKED_TEXT
    # Row 2 (W): masked
    assert table.table.item(2, 2).text() == MASKED_TEXT
    # Row 3 (E): allowed, renormalized 0.10/(0.45+0.10) = 18.2%
    assert table.table.item(3, 2).text() == "18.2%"
    # UNMASKED column shows raw values for all rows
    assert table.table.item(0, 3).text() == "45.0%"
    assert table.table.item(1, 3).text() == "30.0%"
    assert table.table.item(3, 3).text() == "10.0%"
    table.close()


def test_masked_action_items_are_grey():
    """Masked rows have grey foreground color."""
    from triforce_debugger.action_table import GREY  # pylint: disable=import-outside-toplevel
    table = _make_table()
    probs = _make_probs()
    mask_desc = [(_ActionKind.MOVE, [_Direction.N])]
    table.update_probabilities(probs, action_mask_desc=mask_desc)

    # Row 1 (S): masked — all cells should be grey
    for col in range(4):
        assert table.table.item(1, col).foreground().color() == GREY
    table.close()


def test_unmasked_action_items_not_grey():
    """Unmasked rows do not have grey foreground."""
    from triforce_debugger.action_table import GREY  # pylint: disable=import-outside-toplevel
    table = _make_table()
    probs = _make_probs()
    mask_desc = [(_ActionKind.MOVE, [_Direction.N, _Direction.S, _Direction.W, _Direction.E])]
    table.update_probabilities(probs, action_mask_desc=mask_desc)

    # Row 0 (N): allowed — should not be grey
    assert table.table.item(0, 2).foreground().color() != GREY
    table.close()


def test_no_mask_desc_shows_all_probs():
    """When action_mask_desc is None, all rows show probabilities (no masking)."""
    from triforce_debugger.action_table import MASKED_TEXT  # pylint: disable=import-outside-toplevel
    table = _make_table()
    probs = _make_probs()
    table.update_probabilities(probs, action_mask_desc=None)

    for row in range(4):
        assert table.table.item(row, 2).text() != MASKED_TEXT
    table.close()


# ── Clear / None ─────────────────────────────────────────────

def test_update_none_clears_table():
    """Passing None clears the table and resets value."""
    table = _make_table()
    probs = _make_probs()
    table.update_probabilities(probs)
    assert table.table.rowCount() > 0

    table.update_probabilities(None)
    assert table.table.rowCount() == 0
    assert table.value_label.text() == "Value: —"
    table.close()


def test_update_replaces_previous_data():
    """Calling update_probabilities again replaces old data."""
    table = _make_table()
    probs1 = _make_probs()
    table.update_probabilities(probs1)
    assert table.table.rowCount() == 4

    # Now add SWORD too
    probs2 = _make_probs(
        sword=[(_Direction.N, 0.1), (_Direction.S, 0.1),
               (_Direction.W, 0.1), (_Direction.E, 0.1)],
    )
    table.update_probabilities(probs2)
    assert table.table.rowCount() == 8
    table.close()


# ── Edge cases ───────────────────────────────────────────────

def test_empty_probabilities():
    """OrderedDict with only 'value' shows no rows."""
    table = _make_table()
    probs = OrderedDict()
    probs['value'] = torch.tensor(0.0)
    table.update_probabilities(probs)
    assert table.table.rowCount() == 0
    assert "0.0000" in table.value_label.text()
    table.close()


def test_value_missing_shows_dash():
    """If 'value' key missing, label shows dash."""
    table = _make_table()
    probs = OrderedDict()
    probs[_ActionKind.MOVE] = [(_Direction.N, 0.5), (_Direction.S, 0.5),
                                (_Direction.W, 0.0), (_Direction.E, 0.0)]
    table.update_probabilities(probs)
    assert table.value_label.text() == "Value: —"
    table.close()
