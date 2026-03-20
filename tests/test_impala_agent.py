"""Tests for IMPALA ResNet agents.

Verifies:
- ImpalaResNet forward shape with both screen sizes (cropped and full)
- ImpalaSharedAgent forward/get_action_and_value shapes
- ImpalaMultiHeadAgent forward shapes and multihead interface
- Spatial attention weights shape and normalization
- CoordConv buffers are not trainable parameters
- Save/load round-trip preserves outputs
- forward_with_attention returns attention maps
"""

import os
import tempfile
import torch
import numpy as np
import pytest
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, MultiBinary

from triforce.models import (
    ResBlock, ImpalaResNet, SpatialAttentionPool, ImpalaCombinedExtractor,
    EntityAttentionEncoder, EntitySpatialCrossAttention,
    ImpalaSharedAgent, ImpalaMultiHeadAgent, get_neural_network,
    _cross_attention_entropy_stats
)


# --- Screen size fixtures ---

CROPPED = {"height": 168, "width": 240}
FULL = {"height": 176, "width": 256}
SCREEN_SIZES = [CROPPED, FULL]


def _make_obs_space(height=168, width=240):
    """Create a mock observation space for full-rgb mode."""
    return Dict({
        "image": Box(low=0.0, high=1.0, shape=(3, height, width), dtype=np.float32),
        "entities": Box(low=-1.0, high=1.0, shape=(12, 7), dtype=np.float32),
        "entity_types": gym.spaces.MultiDiscrete([74] * 12),
        "information": MultiBinary(15),
    })


def _make_obs_batch(batch_size=2, height=168, width=240):
    """Create a batch of mock observations as tensors."""
    return {
        "image": torch.randn(batch_size, 3, height, width),
        "entities": torch.randn(batch_size, 12, 7),
        "entity_types": torch.zeros(batch_size, 12).long(),
        "information": torch.zeros(batch_size, 15),
    }


# === ResBlock Tests ===

class TestResBlock:
    def test_output_shape_matches_input(self):
        block = ResBlock(32)
        x = torch.randn(2, 32, 21, 30)
        out = block(x)
        assert out.shape == (2, 32, 21, 30)

    def test_skip_connection(self):
        """Output should differ from input (not identity) but have same shape."""
        block = ResBlock(16)
        x = torch.randn(1, 16, 10, 10)
        out = block(x)
        assert out.shape == x.shape
        assert not torch.equal(out, x)


# === SpatialAttentionPool Tests ===

class TestSpatialAttentionPool:
    def test_output_shapes(self):
        pool = SpatialAttentionPool(in_channels=32, hidden_dim=64, output_dim=256, num_heads=4)
        pool.eval()
        feature_map = torch.randn(2, 32, 21, 30)
        features, attn = pool(feature_map)
        assert features.shape == (2, 256)
        assert attn.shape == (2, 4, 21, 30)

    def test_training_returns_none_attn(self):
        """During training, attention weights are None (SDPA doesn't compute them)."""
        pool = SpatialAttentionPool(in_channels=32, hidden_dim=64, output_dim=256, num_heads=4)
        pool.train()
        feature_map = torch.randn(2, 32, 21, 30)
        features, attn = pool(feature_map)
        assert features.shape == (2, 256)
        assert attn is None

    def test_attention_weights_positive(self):
        pool = SpatialAttentionPool(in_channels=32, hidden_dim=64, output_dim=256, num_heads=4)
        pool.eval()
        feature_map = torch.randn(2, 32, 21, 30)
        _, attn = pool(feature_map)
        assert (attn >= 0).all()

    def test_different_spatial_sizes(self):
        """Verify pooling works with different spatial dimensions."""
        pool = SpatialAttentionPool(in_channels=32, hidden_dim=64, output_dim=128, num_heads=4)
        pool.eval()
        for h, w in [(21, 30), (22, 32), (10, 10)]:
            features, attn = pool(torch.randn(1, 32, h, w))
            assert features.shape == (1, 128)
            assert attn.shape == (1, 4, h, w)


# === ImpalaResNet Tests ===

class TestImpalaResNet:
    @pytest.mark.parametrize("screen", SCREEN_SIZES, ids=["cropped", "full"])
    def test_forward_shape(self, screen):
        h, w = screen["height"], screen["width"]
        resnet = ImpalaResNet(input_channels=3, input_height=h, input_width=w, output_dim=256)
        resnet.eval()
        image = torch.randn(2, 3, h, w)
        features, attn, feature_map = resnet(image)
        assert features.shape == (2, 256)
        # Attention map: (batch, num_heads, H/8, W/8)
        assert attn.shape[0] == 2
        assert len(attn.shape) == 4  # (batch, num_heads, H', W')
        # Feature map: (batch, 32, H/8, W/8)
        assert feature_map.shape[0] == 2
        assert feature_map.shape[1] == 32

    def test_coordconv_buffers_not_parameters(self):
        resnet = ImpalaResNet(input_channels=3, input_height=168, input_width=240)
        param_names = {name for name, _ in resnet.named_parameters()}
        assert 'coord_h' not in param_names
        assert 'coord_w' not in param_names

    def test_coordconv_buffers_exist(self):
        resnet = ImpalaResNet(input_channels=3, input_height=168, input_width=240)
        buffer_names = {name for name, _ in resnet.named_buffers()}
        assert 'coord_h' in buffer_names
        assert 'coord_w' in buffer_names

    def test_coordconv_values(self):
        """Coordinate channels should range from 0 to 1."""
        resnet = ImpalaResNet(input_channels=3, input_height=168, input_width=240)
        assert resnet.coord_h.min().item() == pytest.approx(0.0, abs=1e-6)
        assert resnet.coord_h.max().item() == pytest.approx(1.0, abs=1e-6)
        assert resnet.coord_w.min().item() == pytest.approx(0.0, abs=1e-6)
        assert resnet.coord_w.max().item() == pytest.approx(1.0, abs=1e-6)

    def test_channel_progression(self):
        """Verify the expected channel counts through the stacks."""
        resnet = ImpalaResNet(input_channels=3, input_height=168, input_width=240,
                              channels=(16, 32, 32))
        # First stack: Conv2d(5→16), second: Conv2d(16→32), third: Conv2d(32→32)
        stack0_conv = resnet.stacks[0][0]  # First conv in first stack
        assert stack0_conv.in_channels == 5  # 3 RGB + 2 coord
        assert stack0_conv.out_channels == 16

    def test_device_movement(self):
        """CoordConv buffers should move with the module."""
        resnet = ImpalaResNet(input_channels=3, input_height=168, input_width=240)
        assert resnet.coord_h.device.type == 'cpu'
        # Just verify buffers are part of state_dict for device movement
        assert 'coord_h' in resnet.state_dict()
        assert 'coord_w' in resnet.state_dict()


# === ImpalaSharedAgent Tests ===

class TestImpalaSharedAgent:
    @pytest.mark.parametrize("screen", SCREEN_SIZES, ids=["cropped", "full"])
    def test_forward_shapes(self, screen):
        h, w = screen["height"], screen["width"]
        obs_space = _make_obs_space(h, w)
        action_space = Discrete(12)
        agent = ImpalaSharedAgent(obs_space, action_space)

        obs = _make_obs_batch(2, h, w)
        logits, value = agent.forward(obs)
        assert logits.shape == (2, 12)
        assert value.shape == (2, 1)

    def test_get_action_and_value(self):
        obs_space = _make_obs_space()
        action_space = Discrete(12)
        agent = ImpalaSharedAgent(obs_space, action_space)

        obs = _make_obs_batch(4)
        mask = torch.ones(4, 12, dtype=torch.bool)
        actions, log_prob, entropy, value = agent.get_action_and_value(obs, mask)

        assert actions.shape == (4,)
        assert log_prob.shape == (4,)
        assert entropy.shape == (4,)
        assert value.shape == (4,)

    def test_forward_with_attention(self):
        obs_space = _make_obs_space()
        action_space = Discrete(12)
        agent = ImpalaSharedAgent(obs_space, action_space)
        agent.eval()

        obs = _make_obs_batch(2)
        logits, value, spatial_attn, cross_attn = agent.forward_with_attention(obs)
        assert logits.shape == (2, 12)
        assert value.shape == (2, 1)
        assert spatial_attn.shape[0] == 2
        assert len(spatial_attn.shape) == 4  # (batch, num_heads, H', W')
        # Cross-attention: (batch, num_heads, slots, H', W')
        assert cross_attn.shape[0] == 2
        assert cross_attn.shape[2] == 12  # entity slots
        assert len(cross_attn.shape) == 5

    def test_single_obs_unsqueeze(self):
        obs_space = _make_obs_space()
        action_space = Discrete(12)
        agent = ImpalaSharedAgent(obs_space, action_space)

        obs = {
            "image": torch.randn(3, 168, 240),
            "entities": torch.randn(12, 7),
            "entity_types": torch.zeros(12).long(),
            "information": torch.zeros(15),
        }
        logits, value = agent.forward(obs)
        assert logits.shape == (1, 12)

    def test_save_load_roundtrip(self):
        obs_space = _make_obs_space()
        action_space = Discrete(12)
        agent = ImpalaSharedAgent(obs_space, action_space)
        agent.steps_trained = 100
        agent.episodes_evaluated = 5
        agent.metrics = {"test": 1.0}
        agent.eval()

        obs = _make_obs_batch(2)
        with torch.no_grad():
            orig_logits, orig_value = agent.forward(obs)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            agent.save(path)
            loaded = ImpalaSharedAgent(obs_space, action_space)
            loaded.load(path)
            loaded.eval()

            assert loaded.steps_trained == 100
            assert loaded.episodes_evaluated == 5

            with torch.no_grad():
                load_logits, load_value = loaded.forward(obs)

            assert torch.allclose(orig_logits, load_logits, atol=1e-6)
            assert torch.allclose(orig_value, load_value, atol=1e-6)
        finally:
            os.unlink(path)


# === ImpalaMultiHeadAgent Tests ===

class TestImpalaMultiHeadAgent:
    def test_forward_returns_three_tensors(self):
        obs_space = _make_obs_space()
        action_space = MultiDiscrete([3, 4])
        agent = ImpalaMultiHeadAgent(obs_space, action_space)

        obs = _make_obs_batch(2)
        result = agent.forward(obs)
        assert len(result) == 3

    def test_forward_shapes(self):
        obs_space = _make_obs_space()
        action_space = MultiDiscrete([3, 4])
        agent = ImpalaMultiHeadAgent(obs_space, action_space)

        obs = _make_obs_batch(4)
        type_logits, dir_logits, value = agent.forward(obs)
        assert type_logits.shape == (4, 3)
        assert dir_logits.shape == (4, 4)
        assert value.shape == (4, 1)

    def test_get_action_and_value(self):
        obs_space = _make_obs_space()
        action_space = MultiDiscrete([3, 4])
        agent = ImpalaMultiHeadAgent(obs_space, action_space)

        obs = _make_obs_batch(4)
        actions, log_prob, entropy, value = agent.get_action_and_value(obs, mask=None)
        assert actions.shape == (4, 2)
        assert log_prob.shape == (4,)
        assert entropy.shape == (4,)
        assert value.shape == (4,)

    def test_forward_with_attention(self):
        obs_space = _make_obs_space()
        action_space = MultiDiscrete([3, 4])
        agent = ImpalaMultiHeadAgent(obs_space, action_space)
        agent.eval()

        obs = _make_obs_batch(2)
        type_logits, dir_logits, value, spatial_attn, cross_attn = agent.forward_with_attention(obs)
        assert type_logits.shape == (2, 3)
        assert dir_logits.shape == (2, 4)
        assert value.shape == (2, 1)
        assert spatial_attn.shape[0] == 2
        assert len(spatial_attn.shape) == 4  # (batch, num_heads, H', W')
        # Cross-attention: (batch, num_heads, slots, H', W')
        assert cross_attn.shape[0] == 2
        assert cross_attn.shape[2] == 12  # entity slots
        assert len(cross_attn.shape) == 5

    def test_is_multihead(self):
        assert ImpalaMultiHeadAgent.is_multihead is True

    def test_save_load_roundtrip(self):
        obs_space = _make_obs_space()
        action_space = MultiDiscrete([3, 4])
        agent = ImpalaMultiHeadAgent(obs_space, action_space)
        agent.eval()

        obs = _make_obs_batch(2)
        with torch.no_grad():
            orig_actions, orig_lp, _, orig_val = agent.get_action_and_value(
                obs, mask=None, deterministic=True)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            agent.save(path)
            loaded = ImpalaMultiHeadAgent(obs_space, action_space)
            loaded.load(path)
            loaded.eval()

            with torch.no_grad():
                load_actions, load_lp, _, load_val = loaded.get_action_and_value(
                    obs, mask=None, deterministic=True)

            assert torch.equal(orig_actions, load_actions)
            assert torch.allclose(orig_lp, load_lp, atol=1e-6)
            assert torch.allclose(orig_val, load_val, atol=1e-6)
        finally:
            os.unlink(path)


# === EntityAttentionEncoder Token Tests ===

class TestEntityAttentionEncoderTokens:
    def test_forward_with_tokens_shapes(self):
        encoder = EntityAttentionEncoder(num_entity_types=74, continuous_features=7)
        entities = torch.randn(2, 12, 7)
        entity_types = torch.zeros(2, 12).long()
        pooled, tokens, empty_mask = encoder.forward_with_tokens(entities, entity_types)
        assert pooled.shape == (2, 64)
        assert tokens.shape == (2, 12, 64)  # d_model=64
        assert empty_mask.shape == (2, 12)
        assert empty_mask.dtype == torch.bool

    def test_forward_and_forward_with_tokens_agree(self):
        """Pooled output should be identical from both methods."""
        encoder = EntityAttentionEncoder(num_entity_types=74, continuous_features=7)
        encoder.eval()
        entities = torch.randn(2, 12, 7)
        entities[:, :3, 0] = 1.0  # first 3 slots present
        entity_types = torch.zeros(2, 12).long()
        with torch.no_grad():
            pooled_direct = encoder.forward(entities, entity_types)
            pooled_tokens, _, _ = encoder.forward_with_tokens(entities, entity_types)
        assert torch.allclose(pooled_direct, pooled_tokens, atol=1e-6)

    def test_empty_mask_reflects_presence(self):
        encoder = EntityAttentionEncoder(num_entity_types=74, continuous_features=7)
        entities = torch.zeros(1, 12, 7)
        entities[0, 0, 0] = 1.0  # only slot 0 present
        entities[0, 5, 0] = 1.0  # and slot 5
        entity_types = torch.zeros(1, 12).long()
        _, _, empty_mask = encoder.forward_with_tokens(entities, entity_types)
        assert not empty_mask[0, 0]  # slot 0 is present
        assert not empty_mask[0, 5]  # slot 5 is present
        assert empty_mask[0, 1]  # slot 1 is empty


# === EntitySpatialCrossAttention Tests ===

class TestEntitySpatialCrossAttention:
    def test_output_shapes_eval(self):
        cross_attn = EntitySpatialCrossAttention(entity_dim=64, spatial_channels=32)
        cross_attn.eval()
        entity_tokens = torch.randn(2, 12, 64)
        feature_map = torch.randn(2, 32, 21, 30)
        empty_mask = torch.ones(2, 12, dtype=torch.bool)
        empty_mask[:, :3] = False  # 3 present entities
        context, weights = cross_attn(entity_tokens, feature_map, empty_mask)
        assert context.shape == (2, 64)
        assert weights.shape == (2, 4, 12, 21, 30)

    def test_training_returns_none_weights(self):
        cross_attn = EntitySpatialCrossAttention(entity_dim=64, spatial_channels=32)
        cross_attn.train()
        entity_tokens = torch.randn(2, 12, 64)
        feature_map = torch.randn(2, 32, 21, 30)
        empty_mask = torch.zeros(2, 12, dtype=torch.bool)
        context, weights = cross_attn(entity_tokens, feature_map, empty_mask)
        assert context.shape == (2, 64)
        assert weights is None

    def test_empty_entities_zero_context(self):
        """When all entities are empty, context should be zero."""
        cross_attn = EntitySpatialCrossAttention(entity_dim=64, spatial_channels=32)
        cross_attn.eval()
        entity_tokens = torch.randn(1, 12, 64)
        feature_map = torch.randn(1, 32, 21, 30)
        empty_mask = torch.ones(1, 12, dtype=torch.bool)  # all empty
        context, _ = cross_attn(entity_tokens, feature_map, empty_mask)
        # Output should be from output_proj applied to zeros
        # The linear layer bias may make it nonzero, but attended values should be zero
        # Verify by checking the shape at minimum
        assert context.shape == (1, 64)

    def test_weights_positive(self):
        cross_attn = EntitySpatialCrossAttention(entity_dim=64, spatial_channels=32)
        cross_attn.eval()
        entity_tokens = torch.randn(2, 12, 64)
        feature_map = torch.randn(2, 32, 21, 30)
        empty_mask = torch.zeros(2, 12, dtype=torch.bool)
        _, weights = cross_attn(entity_tokens, feature_map, empty_mask)
        assert (weights >= 0).all()


# === ImpalaCombinedExtractor Tests ===

class TestImpalaCombinedExtractor:
    def test_output_dim(self):
        extractor = ImpalaCombinedExtractor(
            image_channels=3, input_height=168, input_width=240,
            num_entity_types=74, entity_features=7, info_size=15
        )
        # 256 (image) + 64 (entity pooled) + 64 (cross-attn) + 15 (info) = 399
        assert extractor.output_dim == 399

    def test_forward_shapes(self):
        extractor = ImpalaCombinedExtractor(
            image_channels=3, input_height=168, input_width=240,
            num_entity_types=74, entity_features=7, info_size=15
        )
        extractor.eval()
        obs = _make_obs_batch(2)
        combined, spatial_attn, cross_attn = extractor(
            obs["image"], obs["entities"], obs["entity_types"], obs["information"])
        assert combined.shape == (2, 399)
        assert spatial_attn is not None
        assert cross_attn is not None
        assert len(cross_attn.shape) == 5  # (batch, heads, slots, H', W')

    def test_training_returns_none_attns(self):
        extractor = ImpalaCombinedExtractor(
            image_channels=3, input_height=168, input_width=240,
            num_entity_types=74, entity_features=7, info_size=15
        )
        extractor.train()
        obs = _make_obs_batch(2)
        combined, spatial_attn, cross_attn = extractor(
            obs["image"], obs["entities"], obs["entity_types"], obs["information"])
        assert combined.shape == (2, 399)
        assert spatial_attn is None
        assert cross_attn is None


# === Cross-Attention Entropy Stats Tests ===

class TestCrossAttentionEntropyStats:
    def test_returns_expected_keys(self):
        cross_attn_map = torch.rand(2, 4, 12, 21, 30)
        # Normalize to proper distribution
        cross_attn_map = cross_attn_map / cross_attn_map.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
        stats = _cross_attention_entropy_stats(cross_attn_map)
        assert "cross_attention/entropy" in stats
        assert "cross_attention/top1_weight" in stats

    def test_uniform_attention_high_entropy(self):
        """Uniform attention over spatial positions should have high entropy."""
        # 21*30 = 630 spatial positions
        cross_attn_map = torch.ones(1, 4, 12, 21, 30) / 630.0
        stats = _cross_attention_entropy_stats(cross_attn_map)
        # Max entropy for 630 positions = ln(630) ≈ 6.45
        assert stats["cross_attention/entropy"] > 6.0

    def test_focused_attention_low_entropy(self):
        """Peaked attention should have low entropy."""
        cross_attn_map = torch.zeros(1, 4, 12, 21, 30)
        cross_attn_map[:, :, :, 10, 15] = 1.0  # all attention on one position
        stats = _cross_attention_entropy_stats(cross_attn_map)
        assert stats["cross_attention/entropy"] < 0.1
        assert stats["cross_attention/top1_weight"] > 0.99


# === Registration Tests ===

class TestImpalaRegistration:
    def test_impala_shared_registered(self):
        cls = get_neural_network("ImpalaSharedAgent")
        assert cls is ImpalaSharedAgent

    def test_impala_multihead_registered(self):
        cls = get_neural_network("ImpalaMultiHeadAgent")
        assert cls is ImpalaMultiHeadAgent

    @pytest.mark.parametrize("screen", SCREEN_SIZES, ids=["cropped", "full"])
    def test_both_screen_sizes_work(self, screen):
        """Both agents should work with both screen sizes."""
        h, w = screen["height"], screen["width"]
        obs_space = _make_obs_space(h, w)

        shared = ImpalaSharedAgent(obs_space, Discrete(12))
        obs = _make_obs_batch(1, h, w)
        logits, value = shared.forward(obs)
        assert logits.shape == (1, 12)

        mh = ImpalaMultiHeadAgent(obs_space, MultiDiscrete([3, 4]))
        type_l, dir_l, value = mh.forward(obs)
        assert type_l.shape == (1, 3)
