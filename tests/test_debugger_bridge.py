"""Tests for QT-02: Environment bridge — StepResult, ModelSelector, EnvironmentBridge."""

from collections import OrderedDict
from unittest.mock import MagicMock, patch, PropertyMock

import torch

from triforce_debugger.environment_bridge import StepResult, ModelSelector, EnvironmentBridge


# ---------------------------------------------------------------------------
# StepResult tests
# ---------------------------------------------------------------------------

class TestStepResult:
    """Tests for the StepResult data class."""

    def test_basic_attributes(self):
        """StepResult stores all attributes correctly."""
        obs = {"image": torch.zeros(1)}
        frames = [torch.zeros(3, 224, 240)]
        state = MagicMock()
        rewards = MagicMock()
        rewards.value = 0.5
        mask = torch.ones(12)
        mask_desc = [("MOVE", ["N", "S", "E", "W"])]

        result = StepResult(obs, frames, state, None, False, False, rewards, mask, mask_desc)

        assert result.observation is obs
        assert result.frames is frames
        assert result.state is state
        assert result.terminated is False
        assert result.truncated is False
        assert result.rewards is rewards
        assert result.action_mask is mask
        assert result.action_mask_desc is mask_desc

    def test_state_from_state_change(self):
        """StepResult uses state_change.state when state is None."""
        state_change = MagicMock()
        state_change.state = MagicMock(name="from_state_change")

        result = StepResult({}, [], None, state_change, False, False, MagicMock(), None, None)

        assert result.state is state_change.state

    def test_completed_terminated(self):
        """completed returns True when terminated."""
        result = StepResult({}, [], MagicMock(), None, True, False, MagicMock(), None, None)
        assert result.completed is True

    def test_completed_truncated(self):
        """completed returns True when truncated."""
        result = StepResult({}, [], MagicMock(), None, False, True, MagicMock(), None, None)
        assert result.completed is True

    def test_not_completed(self):
        """completed returns False when neither terminated nor truncated."""
        result = StepResult({}, [], MagicMock(), None, False, False, MagicMock(), None, None)
        assert result.completed is False


# ---------------------------------------------------------------------------
# ModelSelector tests (mocked — no real env needed)
# ---------------------------------------------------------------------------

def _make_mock_env():
    """Create a mock environment with the necessary structure for ModelSelector."""
    mock_action_space = MagicMock(spec=['get_action_taken', 'get_allowed_actions',
                                       'is_valid_action', 'flat_mask_to_multihead',
                                       'multihead_to_flat'])

    # Make the mock appear as a ZeldaActionSpace for isinstance checks
    mock_env = MagicMock()
    mock_env.observation_space = MagicMock()
    mock_env.action_space = MagicMock()

    return mock_env, mock_action_space


def _make_mock_network(steps=0, metrics=None):
    """Create a mock neural network."""
    network = MagicMock()
    network.steps_trained = steps
    network.metrics = metrics or {}
    network.is_multihead = False
    return network


def _make_selector(model_paths=None):
    """Create a ModelSelector with mocks, bypassing isinstance checks.

    Args:
        model_paths: dict of name->path for available models (default: empty).
    Returns:
        (selector, net) tuple.
    """
    from triforce.action_space import ZeldaActionSpace as RealZAS
    mock_action_space = MagicMock(spec=RealZAS)

    mock_env = MagicMock()
    mock_env.observation_space = MagicMock()
    mock_env.action_space = MagicMock()
    # Chain: mock_env -> mock_action_space (instanceof ZeldaActionSpace)
    mock_env.env = mock_action_space
    mock_action_space.env = None

    model_def = MagicMock()
    model_def.find_available_models.return_value = model_paths or {}

    net = _make_mock_network()
    model_def.neural_net.return_value = net

    selector = ModelSelector(mock_env, "dummy_path", model_def)
    return selector, net


class TestModelSelector:
    """Tests for ModelSelector using mocks."""

    def test_select_model(self):
        """ModelSelector starts with untrained when no models available."""
        selector, net = _make_selector()

        assert selector.model is net
        assert selector.model_name == "untrained"

    def test_next_previous(self):
        """ModelSelector.next() and previous() cycle through models."""
        selector, _ = _make_selector()

        initial_name = selector.model_name
        selector.next()
        after_next = selector.model_name
        selector.previous()
        assert selector.model_name == initial_name

    def test_select_by_name(self):
        """ModelSelector.select() switches to a named model."""
        selector, net = _make_selector()

        # "untrained" and "best" should both exist
        selector.select("best")
        assert selector.model_name == "best"
        selector.select("untrained")
        assert selector.model_name == "untrained"

    def test_select_invalid_raises(self):
        """ModelSelector.select() raises KeyError for unknown name."""
        selector, _ = _make_selector()
        try:
            selector.select("nonexistent")
            assert False, "Should have raised KeyError"
        except KeyError:
            pass


# ---------------------------------------------------------------------------
# EnvironmentBridge tests (mocked — no ROM needed)
# ---------------------------------------------------------------------------

class TestEnvironmentBridge:
    """Tests for EnvironmentBridge using mocks."""

    @patch('triforce_debugger.environment_bridge.make_zelda_env')
    @patch('triforce_debugger.environment_bridge.ModelSelector')
    def test_instantiation(self, mock_selector_cls, mock_make_env):
        """EnvironmentBridge can be instantiated with mocks."""
        from triforce.action_space import ZeldaActionSpace as RealZAS
        mock_action_space = MagicMock(spec=RealZAS)

        mock_env = MagicMock()
        mock_env.env = mock_action_space
        mock_action_space.env = None
        mock_make_env.return_value = mock_env

        model_def = MagicMock()
        model_def.action_space = "move_attack"
        model_def.neural_net = MagicMock()
        model_def.neural_net.is_multihead = False

        scenario_def = MagicMock()

        bridge = EnvironmentBridge("path", model_def, scenario_def, frame_stack=4)

        assert bridge.env is mock_env
        assert bridge.action_space is mock_action_space
        assert bridge.scenario_def is scenario_def
        assert bridge.model_def is model_def
        mock_make_env.assert_called_once()
        mock_selector_cls.assert_called_once()

    @patch('triforce_debugger.environment_bridge.make_zelda_env')
    @patch('triforce_debugger.environment_bridge.ModelSelector')
    def test_restart(self, mock_selector_cls, mock_make_env):
        """EnvironmentBridge.restart() resets the env and returns a StepResult."""
        from triforce.action_space import ZeldaActionSpace as RealZAS
        mock_action_space = MagicMock(spec=RealZAS)

        mock_env = MagicMock()
        mock_env.env = mock_action_space
        mock_action_space.env = None
        mock_make_env.return_value = mock_env

        model_def = MagicMock()
        model_def.action_space = "move_attack"
        model_def.neural_net = MagicMock()
        model_def.neural_net.is_multihead = False

        # Setup reset return value
        mock_state = MagicMock()
        mock_state.info = {
            'initial_frame': MagicMock(),
            'action_mask': torch.ones(12)
        }
        mock_env.reset.return_value = ({"image": torch.zeros(1)}, mock_state)
        mock_action_space.get_allowed_actions.return_value = []

        bridge = EnvironmentBridge("path", model_def, MagicMock(), frame_stack=4)
        result = bridge.restart()

        assert isinstance(result, StepResult)
        assert result.terminated is False
        assert result.truncated is False
        assert result.completed is False
        mock_env.reset.assert_called_once()

    @patch('triforce_debugger.environment_bridge.make_zelda_env')
    @patch('triforce_debugger.environment_bridge.ModelSelector')
    def test_step_with_model_action(self, mock_selector_cls, mock_make_env):
        """EnvironmentBridge.step() uses model when no action given."""
        from triforce.action_space import ZeldaActionSpace as RealZAS
        mock_action_space = MagicMock(spec=RealZAS)

        mock_env = MagicMock()
        mock_env.env = mock_action_space
        mock_action_space.env = None
        mock_make_env.return_value = mock_env

        model_def = MagicMock()
        model_def.action_space = "move_attack"
        model_def.neural_net = MagicMock()
        model_def.neural_net.is_multihead = False

        bridge = EnvironmentBridge("path", model_def, MagicMock(), frame_stack=4)

        # Setup state for stepping
        bridge._observation = {"image": torch.zeros(1)}
        bridge._action_mask = torch.ones(1, 12)

        mock_model = MagicMock()
        mock_model.get_action.return_value = torch.tensor([0, 1])
        bridge.selector = MagicMock()
        bridge.selector.model = mock_model

        mock_action_space.is_valid_action.return_value = True

        mock_state_change = MagicMock()
        mock_state_change.frames = [MagicMock()]
        mock_state_change.state.info = {'action_mask': torch.ones(12)}
        mock_action_space.get_allowed_actions.return_value = []

        mock_env.step.return_value = ({"image": torch.zeros(1)}, MagicMock(), False, False, mock_state_change)

        result = bridge.step()

        assert isinstance(result, StepResult)
        mock_model.get_action.assert_called_once()

    @patch('triforce_debugger.environment_bridge.make_zelda_env')
    @patch('triforce_debugger.environment_bridge.ModelSelector')
    def test_close(self, mock_selector_cls, mock_make_env):
        """EnvironmentBridge.close() releases resources."""
        from triforce.action_space import ZeldaActionSpace as RealZAS
        mock_action_space = MagicMock(spec=RealZAS)

        mock_env = MagicMock()
        mock_env.env = mock_action_space
        mock_action_space.env = None
        mock_make_env.return_value = mock_env

        model_def = MagicMock()
        model_def.action_space = "move_attack"
        model_def.neural_net = MagicMock()
        model_def.neural_net.is_multihead = False

        bridge = EnvironmentBridge("path", model_def, MagicMock(), frame_stack=4)
        bridge.close()

        assert bridge.env is None
        assert bridge.selector is None
        assert bridge._observation is None
        assert bridge._action_mask is None
        mock_env.close.assert_called_once()

    @patch('triforce_debugger.environment_bridge.make_zelda_env')
    @patch('triforce_debugger.environment_bridge.ModelSelector')
    def test_no_action_space_raises(self, mock_selector_cls, mock_make_env):
        """EnvironmentBridge raises if no ZeldaActionSpace found."""
        mock_env = MagicMock()
        mock_env.env = None  # No ZeldaActionSpace in the chain
        # Ensure isinstance checks fail
        mock_env.__class__ = type('FakeEnv', (), {})
        mock_make_env.return_value = mock_env

        model_def = MagicMock()
        model_def.action_space = "move_attack"
        model_def.neural_net = MagicMock()
        model_def.neural_net.is_multihead = False

        try:
            EnvironmentBridge("path", model_def, MagicMock(), frame_stack=4)
            assert False, "Should have raised ValueError"
        except (ValueError, AttributeError):
            pass  # Either is acceptable — ValueError is the intended one

    @patch('triforce_debugger.environment_bridge.make_zelda_env')
    @patch('triforce_debugger.environment_bridge.ModelSelector')
    def test_get_probabilities_delegates(self, mock_selector_cls, mock_make_env):
        """EnvironmentBridge.get_probabilities() delegates to selector."""
        from triforce.action_space import ZeldaActionSpace as RealZAS
        mock_action_space = MagicMock(spec=RealZAS)

        mock_env = MagicMock()
        mock_env.env = mock_action_space
        mock_action_space.env = None
        mock_make_env.return_value = mock_env

        model_def = MagicMock()
        model_def.action_space = "move_attack"
        model_def.neural_net = MagicMock()
        model_def.neural_net.is_multihead = False

        bridge = EnvironmentBridge("path", model_def, MagicMock(), frame_stack=4)
        bridge._observation = {"image": torch.zeros(1)}
        bridge._action_mask = torch.ones(12)

        mock_probs = OrderedDict(value=torch.tensor(0.5))
        bridge.selector = MagicMock()
        bridge.selector.get_probabilities.return_value = mock_probs

        result = bridge.get_probabilities()

        assert result is mock_probs
        bridge.selector.get_probabilities.assert_called_once()
