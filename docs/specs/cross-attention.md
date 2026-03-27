# Cross-Attention Between Entities and Visual Features

## Status

**Implemented** — cross-attention module added in models.py, integrated into ImpalaCombinedExtractor.

## Problem

The current architecture processes visual features (via ImpalaResNet + SpatialAttentionPool) and entity features (via EntityAttentionEncoder) independently, then concatenates them into a single flat vector before the policy/value heads:

```
Image  → ImpalaResNet → SpatialAttentionPool → 256-d vector ─┐
Entities → EntityAttentionEncoder → 64-d vector ──────────────┼→ cat → 335-d → MLP → policy/value
Information → 15-d boolean vector ────────────────────────────┘
```

The model has no mechanism to relate *which* entity it's reasoning about to *where* that entity appears on screen. Entity vectors contain no position information (by design — spatial relationships should be learned from pixels). But the two branches never interact until after all spatial information has been collapsed to a single 256-d vector.

This creates two observable problems:

1. **The model can't ground entities spatially.** It knows an enemy exists (from entity features) and it sees something on screen (from the image), but it has no way to explicitly connect the two. It must learn this correspondence implicitly through the concatenated representation, which is difficult.

2. **We can't visualize entity-specific attention.** The current spatial attention heatmap shows what the model looks at overall, but we can't ask "where is the model looking when it thinks about enemy #3?"

## Proposed Solution

Replace the independent-then-concatenate architecture with **cross-attention** where entity tokens query the spatial feature map. Each entity asks "where am I on screen?" and the attention weights give us the answer.

### Architecture Overview

```
Image  → ImpalaResNet → feature_map (32, 21, 30) → spatial self-attention → 256-d
                              ↓                        ↑
                         K, V projections       cross-attention weights
                              ↓                        ↑
Entities → EntityAttentionEncoder → entity tokens → Q projection
                                                       ↓
                                              per-entity spatial attention
                                                       ↓
                                              entity-grounded features (64-d per entity)
                                                       ↓
                                              pool → entity context vector
```

### Detailed Design

#### Phase 1: Visual Feature Map (unchanged)

The ImpalaResNet produces a feature map of shape `(batch, 32, H', W')` where H'=21 and W'=30 (168/8 × 240/8). This is the same as today.

#### Phase 2: Entity Encoding (modified)

The EntityAttentionEncoder currently produces a single 64-d vector by mean-pooling over all entity tokens after transformer self-attention. Instead, we preserve the **per-entity token representations** before pooling.

Current flow:
```
(batch, 12, 15) → transformer → mean_pool → (batch, 64)
```

New flow:
```
(batch, 12, 15) → transformer → (batch, 12, 64)   [keep per-entity tokens]
```

The output is 12 entity tokens, each a 64-d vector that encodes entity type, health, direction, and relationships to other entities (from the transformer self-attention).

#### Phase 3: Cross-Attention (new)

A new `EntitySpatialCrossAttention` module. Entity tokens are **queries**, spatial positions are **keys and values**.

**Projections:**
- `Q_proj`: Linear(64 → hidden_dim) — projects entity tokens to queries
- `K_proj`: Conv2d(32 → hidden_dim, 1×1) — projects spatial feature map to keys
- `V_proj`: Conv2d(32 → hidden_dim, 1×1) — projects spatial feature map to values

Where `hidden_dim` = 64 (same as current self-attention, can be tuned).

**Computation:**

```
Q = Q_proj(entity_tokens)                     # (batch, 12, hidden_dim)
K = K_proj(feature_map).flatten(2).transpose   # (batch, hidden_dim, N) → (batch, N, hidden_dim)
V = V_proj(feature_map).flatten(2).transpose   # (batch, N, hidden_dim)

# Scaled dot-product attention
# Each entity token attends to all N=630 spatial positions
cross_attn_logits = Q @ K^T / sqrt(hidden_dim)  # (batch, 12, N)
cross_attn_weights = softmax(cross_attn_logits)  # (batch, 12, N)

# Weighted sum of spatial values for each entity
entity_spatial = cross_attn_weights @ V          # (batch, 12, hidden_dim)
```

**Empty-slot masking:** For entities with presence=0, zero out their cross-attention output. This prevents phantom entities from pulling information from the spatial map. Apply the same mask from EntityAttentionEncoder.

**Multi-head cross-attention:** Use the same 4-head structure as self-attention. Each head has `head_dim = hidden_dim / num_heads = 16`. This gives us 4 independent cross-attention patterns per entity — one head might learn to find the entity's screen position, another might attend to nearby geometry.

**Output per entity:** `(batch, 12, hidden_dim)` — each entity now has a spatially-grounded feature vector.

#### Phase 4: Aggregation

Pool the 12 entity-spatial vectors into a single context vector:

```
# Mask empty entities
present_mask = entity_features[:, :, 0].unsqueeze(-1)  # (batch, 12, 1)
masked = entity_spatial * present_mask

# Mean pool over present entities
entity_context = masked.sum(dim=1) / present_mask.sum(dim=1).clamp(min=1)  # (batch, hidden_dim)

# Project to output dim
entity_context = Linear(hidden_dim → 64) + ReLU
```

#### Phase 5: Combination (modified)

The final concatenation is structurally the same, but now the entity branch carries spatial grounding:

```
cat([image_features(256), entity_context(64), information(15)]) → 335-d
```

The output dimension is unchanged — policy and value heads need no modification.

### What About Spatial Self-Attention?

**Keep it.** The spatial self-attention pool (SpatialAttentionPool) serves a different purpose — it decides which parts of the screen matter *overall* for the value estimate and action selection. Cross-attention decides which parts of the screen matter *for each entity*.

The two are complementary:
- **Self-attention**: "There's something important in the top-right corner" (might be a door, a wall pattern, an item)
- **Cross-attention**: "Enemy #3 (Blue Goriya) is at *that* position on screen"

They share the same K and V projections from the spatial feature map (or can — this saves parameters).

### Sharing K/V Projections

The spatial self-attention already computes K and V over the feature map. Cross-attention needs K and V over the same feature map. We can share these projections:

```python
# Shared K, V from feature map (computed once)
K_spatial = K_proj(feature_map)  # (batch, hidden_dim, H', W')
V_spatial = V_proj(feature_map)  # (batch, hidden_dim, H', W')

# Self-attention: Q also from feature map
Q_self = Q_self_proj(feature_map)
self_attn_out, self_attn_weights = self_attention(Q_self, K_spatial, V_spatial)

# Cross-attention: Q from entity tokens
Q_cross = Q_cross_proj(entity_tokens)
cross_attn_out, cross_attn_weights = cross_attention(Q_cross, K_spatial, V_spatial)
```

This adds only one new projection (`Q_cross_proj`) compared to the current architecture. Whether to share or not is a tuning decision — sharing reduces parameters but couples the two attention mechanisms.

**Recommendation:** Start with separate projections for simplicity. Share later if parameter count is a concern.

## Visualization

### Per-Entity Spatial Heatmaps

Cross-attention produces weights of shape `(batch, num_heads, 12, H', W')` — for each entity and each attention head, a spatial distribution over the screen.

**In the debugger (game_view.py):**

The attention overlay gains a new mode. Currently we cycle through:
- Combined (max across heads)
- Head 1, Head 2, Head 3, Head 4

Add entity-specific views:
- **Entity overlay mode**: Select an entity slot (0-11). The heatmap shows that entity's cross-attention weights (max across heads), rendered as a JET overlay just like the current spatial attention.
- **Color-coded multi-entity**: Show all present entities simultaneously, each with a distinct color. Entity #0 in red, #1 in blue, #2 in green, etc. Overlay intensity = attention weight. This shows at a glance which entity is "looking at" which part of the screen.

**In the observation panel:**

Color each entity row's background by attention intensity. If entity #3 has strong cross-attention (high max weight), its row in the entity list is highlighted — indicating the model is actively grounding that entity spatially. Entities the model ignores remain dim.

### Keyboard Controls

Extend the existing `[` / `]` cycling:
- Current: Combined → Head 1 → Head 2 → Head 3 → Head 4
- New: Combined → Head 1-4 → Entity 0 → Entity 1 → ... → Entity 11 → back to Combined

Skip entities with presence=0 when cycling.

### TUI Metrics

Add to the training TUI and tensorboard:
- `losses/cross_attention/entropy`: Mean entropy of cross-attention weights across entities. High = diffuse attention (entity doesn't know where to look). Low = focused (entity found its target).
- `losses/cross_attention/top1_weight`: Mean top-1 weight. High = entity strongly fixates on one position.

Per-entity breakdown is too noisy for the TUI but useful in tensorboard.

## Impact on Existing Components

| Component | Change Required |
|---|---|
| `ImpalaResNet` | None — still produces `(batch, 32, H', W')` |
| `SpatialAttentionPool` | None — still pools the feature map for the image branch |
| `EntityAttentionEncoder` | Return per-entity tokens `(batch, 12, 64)` in addition to pooled vector |
| `ImpalaCombinedExtractor` | Add `EntitySpatialCrossAttention` between entity encoder and concatenation |
| `ImpalaSharedAgent` | Update `forward_with_attention` to return cross-attention weights |
| `ImpalaMultiHeadAgent` | Same as ImpalaSharedAgent |
| Policy/value heads | None — input dimension unchanged |
| `game_view.py` | New overlay modes for entity cross-attention |
| `observation_panel.py` | Optional entity row highlighting |
| `environment_bridge.py` | Return cross-attention weights alongside spatial attention |
| `ml_ppo.py` | Log cross-attention entropy/top1 metrics |
| `train.py` | Display cross-attention metrics in TUI |

## Model Compatibility

This is a **breaking change**. Models trained before cross-attention will not load into the new architecture because:
- `EntityAttentionEncoder` gains a new output path
- `ImpalaCombinedExtractor` adds the `EntitySpatialCrossAttention` module
- `state_dict` keys change

A new model must be trained from scratch, or a migration script could initialize the cross-attention weights randomly while loading existing weights for unchanged layers. The latter would still require retraining but might converge faster.

## Open Questions

1. **Should cross-attention replace or supplement the entity pooling?** Current proposal: cross-attention produces spatially-grounded entity features, then pools them. Alternative: feed both the pooled entity vector *and* the cross-attention context into the policy head (larger input dim).

Answer: Feed them both.

2. **How many cross-attention heads?** 4 matches the spatial self-attention. Could use fewer (2) since entity queries are lower-dimensional than spatial queries.

Answer: 4.

3. **Should the information vector (15 booleans) participate in cross-attention?** Some boolean features (objective direction, enemy presence) could benefit from spatial grounding. But 15 features is small — the added complexity may not be worth it.

Answer: No

4. **Training curriculum impact.** Cross-attention adds parameters and a new learning signal. Early training may be slower as the model learns entity-spatial correspondence. Consider whether the training circuit needs adjustment.

Answer: As part of this, double the length of main-circuit's legs (except for the last).

5. **K/V sharing.** Share spatial K/V between self-attention and cross-attention, or keep separate? Sharing saves ~4K parameters (small), but couples the representations.

Answer: Don't share.
