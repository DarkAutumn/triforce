"""Tests for MultiHeadAgent (MH-01).

Verifies:
- Forward pass output shapes (action_type_logits, direction_logits, value)
- get_action_and_value shapes and return types
- Per-head masking (action type mask + direction mask)
- Joint log-prob = log π_type + log π_dir
- Save/load round-trip preserves outputs
- SharedNatureAgent is unaffected by MultiHeadAgent addition
"""

import os
import tempfile
import torch
import numpy as np
import pytest
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, MultiBinary

from triforce.models import MultiHeadAgent, SharedNatureAgent, Network, get_neural_network


def _make_obs_space():
    """Create a mock observation space matching ZeldaObservationWrapper output."""
    return Dict({
        "image": Box(low=0.0, high=1.0, shape=(1, 128, 128), dtype=np.float32),
        "entities": Box(low=-1.0, high=1.0, shape=(12, 9), dtype=np.float32),
        "entity_types": gym.spaces.MultiDiscrete([74] * 12),
        "information": MultiBinary(15),
    })


def _make_obs_batch(batch_size=2):
    """Create a batch of mock observations as tensors."""
    return {
        "image": torch.randn(batch_size, 1, 128, 128),
        "entities": torch.randn(batch_size, 12, 9),
        "entity_types": torch.zeros(batch_size, 12).long(),
        "information": torch.zeros(batch_size, 15),
    }


class TestMultiHeadForward:
    """Verify forward() output shapes."""

    def test_forward_returns_three_tensors(self):
        obs_space = _make_obs_space()
        action_space = MultiDiscrete([3, 4])
        agent = MultiHeadAgent(obs_space, action_space)

        obs = _make_obs_batch(2)
        result = agent.forward(obs)
        assert len(result) == 3, "forward() should return (type_logits, dir_logits, value)"

    def test_forward_shapes(self):
        obs_space = _make_obs_space()
        action_space = MultiDiscrete([3, 4])
        agent = MultiHeadAgent(obs_space, action_space)

        obs = _make_obs_batch(4)
        type_logits, dir_logits, value = agent.forward(obs)

        assert type_logits.shape == (4, 3), f"Expected (4, 3), got {type_logits.shape}"
        assert dir_logits.shape == (4, 4), f"Expected (4, 4), got {dir_logits.shape}"
        assert value.shape == (4, 1), f"Expected (4, 1), got {value.shape}"

    def test_forward_single_obs(self):
        """Single observation (no batch dim) should be auto-unsqueezed."""
        obs_space = _make_obs_space()
        action_space = MultiDiscrete([3, 4])
        agent = MultiHeadAgent(obs_space, action_space)

        obs = {
            "image": torch.randn(1, 128, 128),
            "entities": torch.randn(12, 9),
            "entity_types": torch.zeros(12).long(),
            "information": torch.zeros(15),
        }
        type_logits, dir_logits, value = agent.forward(obs)
        assert type_logits.shape == (1, 3)
        assert dir_logits.shape == (1, 4)


class TestMultiHeadGetActionAndValue:
    """Verify get_action_and_value() shapes and semantics."""

    def test_output_shapes_no_mask(self):
        obs_space = _make_obs_space()
        action_space = MultiDiscrete([3, 4])
        agent = MultiHeadAgent(obs_space, action_space)

        obs = _make_obs_batch(4)
        actions, log_prob, entropy, value = agent.get_action_and_value(obs, mask=None)

        assert actions.shape == (4, 2), f"Expected (4, 2), got {actions.shape}"
        assert log_prob.shape == (4,), f"Expected (4,), got {log_prob.shape}"
        assert entropy.shape == (4,), f"Expected (4,), got {entropy.shape}"
        assert value.shape == (4,), f"Expected (4,), got {value.shape}"

    def test_action_ranges(self):
        """Actions should be in valid ranges: type in [0, K), dir in [0, 4)."""
        obs_space = _make_obs_space()
        action_space = MultiDiscrete([3, 4])
        agent = MultiHeadAgent(obs_space, action_space)

        obs = _make_obs_batch(100)
        actions, _, _, _ = agent.get_action_and_value(obs, mask=None)

        assert (actions[:, 0] >= 0).all() and (actions[:, 0] < 3).all()
        assert (actions[:, 1] >= 0).all() and (actions[:, 1] < 4).all()

    def test_deterministic_mode(self):
        """Deterministic should return argmax actions consistently."""
        obs_space = _make_obs_space()
        action_space = MultiDiscrete([3, 4])
        agent = MultiHeadAgent(obs_space, action_space)

        obs = _make_obs_batch(4)
        a1, _, _, _ = agent.get_action_and_value(obs, mask=None, deterministic=True)
        a2, _, _, _ = agent.get_action_and_value(obs, mask=None, deterministic=True)
        assert torch.equal(a1, a2), "Deterministic mode should produce identical actions"

    def test_provided_actions(self):
        """When actions are provided, log_prob should be computed for those actions."""
        obs_space = _make_obs_space()
        action_space = MultiDiscrete([3, 4])
        agent = MultiHeadAgent(obs_space, action_space)

        obs = _make_obs_batch(4)
        given_actions = torch.tensor([[0, 0], [1, 1], [2, 2], [0, 3]])
        _, log_prob, entropy, value = agent.get_action_and_value(obs, mask=None, actions=given_actions)

        assert log_prob.shape == (4,)
        assert (log_prob <= 0).all(), "Log-probs should be <= 0"


class TestMultiHeadMasking:
    """Verify per-head masking works correctly."""

    def test_masked_action_type(self):
        """Masking out action types should prevent those from being selected."""
        obs_space = _make_obs_space()
        action_space = MultiDiscrete([3, 4])
        agent = MultiHeadAgent(obs_space, action_space)

        obs = _make_obs_batch(100)
        # Mask: only action type 1 allowed, all directions allowed
        mask = torch.zeros(100, 7, dtype=torch.bool)
        mask[:, 1] = True        # only type index 1
        mask[:, 3:7] = True      # all 4 directions

        actions, _, _, _ = agent.get_action_and_value(obs, mask=mask)
        assert (actions[:, 0] == 1).all(), "Only action type 1 should be selected"

    def test_masked_direction(self):
        """Masking out directions should prevent those from being selected."""
        obs_space = _make_obs_space()
        action_space = MultiDiscrete([3, 4])
        agent = MultiHeadAgent(obs_space, action_space)

        obs = _make_obs_batch(100)
        # Mask: all action types allowed, only direction 2 allowed
        mask = torch.zeros(100, 7, dtype=torch.bool)
        mask[:, 0:3] = True      # all 3 types
        mask[:, 5] = True         # only direction index 2 (offset by K=3)

        actions, _, _, _ = agent.get_action_and_value(obs, mask=mask)
        assert (actions[:, 1] == 2).all(), "Only direction 2 should be selected"

    def test_mask_assertion_no_valid_type(self):
        """Should assert if no valid action type in mask."""
        obs_space = _make_obs_space()
        action_space = MultiDiscrete([3, 4])
        agent = MultiHeadAgent(obs_space, action_space)

        obs = _make_obs_batch(1)
        mask = torch.zeros(1, 7, dtype=torch.bool)
        mask[:, 3:7] = True  # directions OK, but no action types
        with pytest.raises(ValueError, match="Empty action mask"):
            agent.get_action_and_value(obs, mask=mask)

    def test_mask_assertion_no_valid_direction(self):
        """Should assert if no valid direction in mask."""
        obs_space = _make_obs_space()
        action_space = MultiDiscrete([3, 4])
        agent = MultiHeadAgent(obs_space, action_space)

        obs = _make_obs_batch(1)
        mask = torch.zeros(1, 7, dtype=torch.bool)
        mask[:, 0:3] = True  # types OK, but no directions
        with pytest.raises(ValueError, match="Empty action mask"):
            agent.get_action_and_value(obs, mask=mask)


class TestJointLogProb:
    """Verify joint log-prob = log π_type + log π_dir."""

    def test_joint_log_prob_decomposition(self):
        """Joint log-prob should equal the sum of per-head log-probs."""
        obs_space = _make_obs_space()
        action_space = MultiDiscrete([3, 4])
        agent = MultiHeadAgent(obs_space, action_space)

        obs = _make_obs_batch(4)
        with torch.no_grad():
            type_logits, dir_logits, _ = agent.forward(obs)

            type_dist = torch.distributions.Categorical(logits=type_logits)
            dir_dist = torch.distributions.Categorical(logits=dir_logits)

            # Sample actions
            type_action = type_dist.sample()
            dir_action = dir_dist.sample()
            actions = torch.stack([type_action, dir_action], dim=-1)

            # Compute per-head log-probs manually
            expected_log_prob = type_dist.log_prob(type_action) + dir_dist.log_prob(dir_action)

            # Get joint log-prob from the agent
            _, actual_log_prob, _, _ = agent.get_action_and_value(obs, mask=None, actions=actions)

        assert torch.allclose(actual_log_prob, expected_log_prob, atol=1e-6), \
            f"Joint log-prob mismatch: {actual_log_prob} vs {expected_log_prob}"


class TestMultiHeadSaveLoad:
    """Verify save/load round-trip preserves model state."""

    def test_save_load_roundtrip(self):
        obs_space = _make_obs_space()
        action_space = MultiDiscrete([3, 4])
        agent = MultiHeadAgent(obs_space, action_space)
        agent.steps_trained = 42
        agent.episodes_evaluated = 10
        agent.metrics = {"test_metric": 1.5}

        obs = _make_obs_batch(2)
        with torch.no_grad():
            orig_actions, orig_lp, _, orig_val = agent.get_action_and_value(
                obs, mask=None, deterministic=True
            )

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            agent.save(path)

            loaded = MultiHeadAgent(obs_space, action_space)
            loaded.load(path)

            assert loaded.steps_trained == 42
            assert loaded.episodes_evaluated == 10
            assert loaded.metrics == {"test_metric": 1.5}

            with torch.no_grad():
                load_actions, load_lp, _, load_val = loaded.get_action_and_value(
                    obs, mask=None, deterministic=True
                )

            assert torch.equal(orig_actions, load_actions), "Actions should match after load"
            assert torch.allclose(orig_lp, load_lp, atol=1e-6), "Log-probs should match"
            assert torch.allclose(orig_val, load_val, atol=1e-6), "Values should match"
        finally:
            os.unlink(path)

    def test_load_mismatched_action_space_fails(self):
        """Loading with wrong action space should raise an error."""
        obs_space = _make_obs_space()
        agent = MultiHeadAgent(obs_space, MultiDiscrete([3, 4]))

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            agent.save(path)

            wrong_agent = MultiHeadAgent(obs_space, MultiDiscrete([5, 4]))
            # PyTorch load_state_dict raises RuntimeError on size mismatch
            # before our action_space check runs
            with pytest.raises((RuntimeError, ValueError)):
                wrong_agent.load(path)
        finally:
            os.unlink(path)


class TestSharedNatureAgentUnchanged:
    """Verify SharedNatureAgent still works identically after adding MultiHeadAgent."""

    def test_shared_nature_forward_shapes(self):
        obs_space = _make_obs_space()
        action_space = Discrete(12)
        agent = SharedNatureAgent(obs_space, action_space)

        obs = _make_obs_batch(2)
        logits, value = agent.forward(obs)

        assert logits.shape == (2, 12), f"Expected (2, 12), got {logits.shape}"
        assert value.shape == (2, 1), f"Expected (2, 1), got {value.shape}"

    def test_shared_nature_get_action_and_value(self):
        obs_space = _make_obs_space()
        action_space = Discrete(12)
        agent = SharedNatureAgent(obs_space, action_space)

        obs = _make_obs_batch(4)
        mask = torch.ones(4, 12, dtype=torch.bool)
        actions, log_prob, entropy, value = agent.get_action_and_value(obs, mask)

        assert actions.shape == (4,), f"Expected (4,), got {actions.shape}"
        assert log_prob.shape == (4,)
        assert entropy.shape == (4,)
        assert value.shape == (4,)

    def test_shared_nature_registered(self):
        """SharedNatureAgent should still be in NEURAL_NETWORK_DEFINITIONS."""
        cls = get_neural_network("SharedNatureAgent")
        assert cls is SharedNatureAgent


class TestMultiHeadRegistered:
    """Verify MultiHeadAgent is registered and discoverable."""

    def test_registered_in_definitions(self):
        cls = get_neural_network("MultiHeadAgent")
        assert cls is MultiHeadAgent

    def test_is_network_subclass(self):
        assert issubclass(MultiHeadAgent, Network)


class TestIsMultiheadFlag:
    """Verify is_multihead class attribute on Network and MultiHeadAgent."""

    def test_network_is_not_multihead(self):
        assert Network.is_multihead is False

    def test_shared_nature_agent_is_not_multihead(self):
        assert SharedNatureAgent.is_multihead is False

    def test_multihead_agent_is_multihead(self):
        assert MultiHeadAgent.is_multihead is True
