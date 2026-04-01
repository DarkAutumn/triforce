"""Tests for multihead action space adapter (MH-02).

Verifies:
- All (action_type, direction) combos produce correct NES buttons
- Multihead mask decomposition matches flat mask semantics
- ActionTaken fields are correct for multihead inputs
- get_action_taken handles numpy/torch multihead arrays
- Flat (non-multihead) mode is unchanged
- multihead_to_flat round-trips correctly
"""

import numpy as np
import pytest
import torch
import gymnasium as gym

from triforce.action_space import ZeldaActionSpace, ActionTaken
from triforce.zelda_enums import ActionKind, Direction


# NES button indices (MultiBinary(9) layout)
BTN_B = 0
BTN_UP = 4
BTN_DOWN = 5
BTN_LEFT = 6
BTN_RIGHT = 7
BTN_A = 8


class MockEnv(gym.Env):
    """Mock NES environment for testing ZeldaActionSpace without a ROM."""
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.MultiBinary(9)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(240, 256, 3))
        self.buttons = ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']

    @property
    def unwrapped(self):
        return self

    def reset(self, **kwargs):
        return np.zeros((240, 256, 3)), {}

    def step(self, action):
        return np.zeros((240, 256, 3)), 0.0, False, False, {}


def _make_action_space(actions=None, multihead=False):
    """Create a ZeldaActionSpace with mock env."""
    if actions is None:
        actions = ["MOVE", "SWORD", "BEAMS"]
    env = MockEnv()
    return ZeldaActionSpace(env, actions, multihead=multihead)


class TestMultiheadToFlat:
    """Verify multihead_to_flat maps (type_idx, dir_idx) → correct flat index."""

    def test_move_all_directions(self):
        space = _make_action_space()
        # MOVE is type 0, flat base index 0
        assert space.multihead_to_flat(0, 0) == 0   # MOVE N
        assert space.multihead_to_flat(0, 1) == 1   # MOVE S
        assert space.multihead_to_flat(0, 2) == 2   # MOVE W
        assert space.multihead_to_flat(0, 3) == 3   # MOVE E

    def test_sword_all_directions(self):
        space = _make_action_space()
        # SWORD is type 1, flat base index 4
        assert space.multihead_to_flat(1, 0) == 4   # SWORD N
        assert space.multihead_to_flat(1, 1) == 5   # SWORD S
        assert space.multihead_to_flat(1, 2) == 6   # SWORD W
        assert space.multihead_to_flat(1, 3) == 7   # SWORD E

    def test_beams_all_directions(self):
        space = _make_action_space()
        # BEAMS is type 2, flat base index 8
        assert space.multihead_to_flat(2, 0) == 8   # BEAMS N
        assert space.multihead_to_flat(2, 1) == 9   # BEAMS S
        assert space.multihead_to_flat(2, 2) == 10  # BEAMS W
        assert space.multihead_to_flat(2, 3) == 11  # BEAMS E


class TestMultiheadNESButtons:
    """Verify all (type, dir) combos produce correct NES button presses."""

    def test_move_buttons(self):
        """MOVE should only press direction buttons."""
        space = _make_action_space()
        combos = [
            (0, 0, [BTN_UP]),           # MOVE N
            (0, 1, [BTN_DOWN]),         # MOVE S
            (0, 2, [BTN_LEFT]),         # MOVE W
            (0, 3, [BTN_RIGHT]),        # MOVE E
        ]
        for type_idx, dir_idx, expected_buttons in combos:
            flat = space.multihead_to_flat(type_idx, dir_idx)
            action = ActionTaken(space, flat)
            for btn in expected_buttons:
                assert action.multi_binary[btn], f"Button {btn} should be True for ({type_idx}, {dir_idx})"
            assert not action.multi_binary[BTN_A], f"A should not be pressed for MOVE ({type_idx}, {dir_idx})"

    def test_sword_buttons(self):
        """SWORD should press direction + A."""
        space = _make_action_space()
        combos = [
            (1, 0, BTN_UP),     # SWORD N
            (1, 1, BTN_DOWN),   # SWORD S
            (1, 2, BTN_LEFT),   # SWORD W
            (1, 3, BTN_RIGHT),  # SWORD E
        ]
        for type_idx, dir_idx, dir_btn in combos:
            flat = space.multihead_to_flat(type_idx, dir_idx)
            action = ActionTaken(space, flat)
            assert action.multi_binary[dir_btn], f"Direction button should be True"
            assert action.multi_binary[BTN_A], f"A should be pressed for SWORD"

    def test_beams_buttons(self):
        """BEAMS should press direction + A (same as SWORD at NES level)."""
        space = _make_action_space()
        for dir_idx in range(4):
            flat = space.multihead_to_flat(2, dir_idx)
            action = ActionTaken(space, flat)
            assert action.multi_binary[BTN_A], f"A should be pressed for BEAMS dir={dir_idx}"

    def test_all_combos_match_flat(self):
        """Every multihead combo should match the equivalent flat action."""
        space = _make_action_space()
        for type_idx in range(3):
            for dir_idx in range(4):
                flat_idx = space.multihead_to_flat(type_idx, dir_idx)
                multihead_action = ActionTaken(space, flat_idx)
                flat_action = ActionTaken(space, flat_idx)
                assert np.array_equal(multihead_action.multi_binary, flat_action.multi_binary)
                assert multihead_action.kind == flat_action.kind
                assert multihead_action.direction == flat_action.direction


class TestMultiheadActionTakenFields:
    """Verify ActionTaken has correct kind and direction for multihead inputs."""

    EXPECTED = [
        (0, 0, ActionKind.MOVE, Direction.N),
        (0, 1, ActionKind.MOVE, Direction.S),
        (0, 2, ActionKind.MOVE, Direction.W),
        (0, 3, ActionKind.MOVE, Direction.E),
        (1, 0, ActionKind.SWORD, Direction.N),
        (1, 1, ActionKind.SWORD, Direction.S),
        (1, 2, ActionKind.SWORD, Direction.W),
        (1, 3, ActionKind.SWORD, Direction.E),
        (2, 0, ActionKind.BEAMS, Direction.N),
        (2, 1, ActionKind.BEAMS, Direction.S),
        (2, 2, ActionKind.BEAMS, Direction.W),
        (2, 3, ActionKind.BEAMS, Direction.E),
    ]

    @pytest.mark.parametrize("type_idx,dir_idx,expected_kind,expected_dir", EXPECTED)
    def test_action_taken_fields(self, type_idx, dir_idx, expected_kind, expected_dir):
        space = _make_action_space()
        flat = space.multihead_to_flat(type_idx, dir_idx)
        action = ActionTaken(space, flat)
        assert action.kind == expected_kind
        assert action.direction == expected_dir


class TestGetActionTakenMultihead:
    """Verify get_action_taken handles multihead numpy/torch arrays."""

    def test_numpy_array(self):
        space = _make_action_space()
        action = space.get_action_taken(np.array([1, 2]))
        assert isinstance(action, ActionTaken)
        assert action.kind == ActionKind.SWORD
        assert action.direction == Direction.W  # dir_idx 2 = W

    def test_torch_tensor(self):
        space = _make_action_space()
        action = space.get_action_taken(torch.tensor([2, 0]))
        assert isinstance(action, ActionTaken)
        assert action.kind == ActionKind.BEAMS
        assert action.direction == Direction.N  # dir_idx 0 = N

    def test_flat_int_still_works(self):
        """Flat integer actions should still work."""
        space = _make_action_space()
        action = space.get_action_taken(5)
        assert isinstance(action, ActionTaken)
        assert action.kind == ActionKind.SWORD
        assert action.direction == Direction.S  # flat index 5 = SWORD S

    def test_tuple_still_works(self):
        """(ActionKind, Direction) tuple should still work."""
        space = _make_action_space()
        action = space.get_action_taken((ActionKind.BEAMS, Direction.E))
        assert isinstance(action, ActionTaken)
        assert action.kind == ActionKind.BEAMS
        assert action.direction == Direction.E

    def test_numpy_scalar_still_works(self):
        """Scalar numpy array should still work."""
        space = _make_action_space()
        action = space.get_action_taken(np.int64(3))
        assert isinstance(action, ActionTaken)
        assert action.kind == ActionKind.MOVE
        assert action.direction == Direction.E  # flat index 3 = MOVE E


class TestFlatMaskToMultihead:
    """Verify flat mask decomposition into multihead [K*4] format."""

    def test_all_actions_available(self):
        """When all flat actions are valid, all types and all dirs should be True."""
        space = _make_action_space()
        flat_mask = torch.ones(12, dtype=torch.bool)
        mh_mask = space.flat_mask_to_multihead(flat_mask)
        assert mh_mask.shape == (12,)  # K=3, 3*4=12
        assert mh_mask.all(), "All should be True when flat mask is all True"

    def test_move_only(self):
        """When only MOVE is available, only MOVE directions should be True."""
        space = _make_action_space()
        flat_mask = torch.zeros(12, dtype=torch.bool)
        flat_mask[0:4] = True  # MOVE N/S/W/E
        mh_mask = space.flat_mask_to_multihead(flat_mask)
        assert mh_mask[0:4].all(), "MOVE all 4 directions should be True"
        assert not mh_mask[4:8].any(), "SWORD should be all False"
        assert not mh_mask[8:12].any(), "BEAMS should be all False"

    def test_partial_move_still_enables_type(self):
        """If any direction of a type is valid, that type+direction entry is True."""
        space = _make_action_space()
        flat_mask = torch.zeros(12, dtype=torch.bool)
        flat_mask[0] = True  # Only MOVE N
        mh_mask = space.flat_mask_to_multihead(flat_mask)
        assert mh_mask[0] == True, "MOVE N should be True"
        assert mh_mask[1] == False, "MOVE S should be False"
        assert not mh_mask[4:8].any(), "SWORD should be all False"
        assert not mh_mask[8:12].any(), "BEAMS should be all False"

    def test_sword_and_beams(self):
        """SWORD and BEAMS availability reflected correctly."""
        space = _make_action_space()
        flat_mask = torch.zeros(12, dtype=torch.bool)
        flat_mask[4:8] = True   # SWORD all dirs
        flat_mask[8:12] = True  # BEAMS all dirs
        mh_mask = space.flat_mask_to_multihead(flat_mask)
        assert not mh_mask[0:4].any(), "MOVE should be all False"
        assert mh_mask[4:8].all(), "SWORD all dirs should be True"
        assert mh_mask[8:12].all(), "BEAMS all dirs should be True"

    def test_per_type_direction_constraints_preserved(self):
        """Each type preserves its own direction constraints independently."""
        space = _make_action_space()
        flat_mask = torch.zeros(12, dtype=torch.bool)
        flat_mask[0] = True  # Only MOVE N
        mh_mask = space.flat_mask_to_multihead(flat_mask)
        mh_2d = mh_mask.view(3, 4)
        # MOVE: only N
        assert mh_2d[0, 0] == True, "MOVE N should be True"
        assert mh_2d[0, 1] == False, "MOVE S should be False"
        assert mh_2d[0, 2] == False, "MOVE W should be False"
        assert mh_2d[0, 3] == False, "MOVE E should be False"
        # SWORD/BEAMS: all False
        assert not mh_2d[1].any(), "SWORD should be all False"
        assert not mh_2d[2].any(), "BEAMS should be all False"

    def test_cross_type_direction_independence(self):
        """MOVE S masked but SWORD N valid: model cannot pick MOVE N."""
        space = _make_action_space()
        flat_mask = torch.zeros(12, dtype=torch.bool)
        flat_mask[1] = True  # MOVE S
        flat_mask[4] = True  # SWORD N
        mh_mask = space.flat_mask_to_multihead(flat_mask)
        mh_2d = mh_mask.view(3, 4)
        # MOVE: only S, NOT N
        assert mh_2d[0, 0] == False, "MOVE N should be False"
        assert mh_2d[0, 1] == True, "MOVE S should be True"
        # SWORD: only N
        assert mh_2d[1, 0] == True, "SWORD N should be True"
        assert mh_2d[1, 1] == False, "SWORD S should be False"

    def test_no_actions_gives_empty_mask(self):
        """Empty flat mask → all entries False."""
        space = _make_action_space()
        flat_mask = torch.zeros(12, dtype=torch.bool)
        mh_mask = space.flat_mask_to_multihead(flat_mask)
        assert not mh_mask.any(), "All entries should be False"


class TestMultiheadActionSpace:
    """Verify the gym action space is set correctly."""

    def test_multihead_true_gives_multidiscrete(self):
        space = _make_action_space(multihead=True)
        assert isinstance(space.action_space, gym.spaces.MultiDiscrete)
        assert list(space.action_space.nvec) == [3, 4]

    def test_multihead_false_gives_discrete(self):
        space = _make_action_space(multihead=False)
        assert isinstance(space.action_space, gym.spaces.Discrete)
        assert space.action_space.n == 12

    def test_num_action_types(self):
        space = _make_action_space()
        assert space.num_action_types == 3

    def test_total_actions_unchanged(self):
        """total_actions should always be the flat total, regardless of multihead."""
        space_flat = _make_action_space(multihead=False)
        space_mh = _make_action_space(multihead=True)
        assert space_flat.total_actions == space_mh.total_actions == 12


class TestFlatModeUnchanged:
    """Verify non-multihead mode is completely unchanged."""

    def test_flat_action_space_type(self):
        space = _make_action_space(multihead=False)
        assert isinstance(space.action_space, gym.spaces.Discrete)
        assert space.action_space.n == 12

    def test_flat_get_action_taken(self):
        space = _make_action_space(multihead=False)
        for i in range(12):
            action = space.get_action_taken(i)
            assert isinstance(action, ActionTaken)
            assert action.id == i

    def test_flat_action_kind_direction_mapping(self):
        """Verify the flat mapping is correct."""
        space = _make_action_space(multihead=False)
        expected = [
            (ActionKind.MOVE, Direction.N),
            (ActionKind.MOVE, Direction.S),
            (ActionKind.MOVE, Direction.W),
            (ActionKind.MOVE, Direction.E),
            (ActionKind.SWORD, Direction.N),
            (ActionKind.SWORD, Direction.S),
            (ActionKind.SWORD, Direction.W),
            (ActionKind.SWORD, Direction.E),
            (ActionKind.BEAMS, Direction.N),
            (ActionKind.BEAMS, Direction.S),
            (ActionKind.BEAMS, Direction.W),
            (ActionKind.BEAMS, Direction.E),
        ]
        for i, (exp_kind, exp_dir) in enumerate(expected):
            action = space.get_action_taken(i)
            assert action.kind == exp_kind, f"Flat {i}: expected {exp_kind}, got {action.kind}"
            assert action.direction == exp_dir, f"Flat {i}: expected {exp_dir}, got {action.direction}"


class TestMoveOnlyActionSpace:
    """Test with MOVE-only action space to ensure single-type works."""

    def test_single_type_multihead(self):
        space = _make_action_space(actions=["MOVE"], multihead=True)
        assert list(space.action_space.nvec) == [1, 4]
        assert space.num_action_types == 1

    def test_single_type_mask_decomposition(self):
        space = _make_action_space(actions=["MOVE"], multihead=True)
        flat_mask = torch.ones(4, dtype=torch.bool)
        mh_mask = space.flat_mask_to_multihead(flat_mask)
        assert mh_mask.shape == (4,)   # K=1, 1*4=4
        assert mh_mask.all()

    def test_single_type_multihead_to_flat(self):
        space = _make_action_space(actions=["MOVE"], multihead=True)
        for dir_idx in range(4):
            assert space.multihead_to_flat(0, dir_idx) == dir_idx
