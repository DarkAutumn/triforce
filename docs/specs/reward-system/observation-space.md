# Observation Space for Movement Decisions

## Current Behavior

### Architecture Overview

The observation space is a multi-modal dictionary with six components, defined in `observation_wrapper.py` (line 79–86):

```python
observation_space = Dict({
    "image": Box(shape=(frame_stack, VIEWPORT_PIXELS, VIEWPORT_PIXELS)),  # (4, 128, 128)
    "enemy_features": Box(shape=(4, 6)),      # 4 enemies × 6 features
    "enemy_id": Discrete(4),                   # 4 enemy type IDs (integers)
    "item_features": Box(shape=(2, 4)),        # 2 items × 4 features
    "projectile_features": Box(shape=(2, 5)),  # 2 projectiles × 5 features
    "information": MultiBinary(14)             # 14 boolean features
})
```

These are processed by `SharedNatureAgent` (defined in `models.py`, lines 291–343), which uses a `CombinedExtractor` to fuse all modalities into a single 328-dimensional vector, then feeds that into separate policy and value MLP heads (each 3-layer, 64-dimensional).

### Image Component

The image path processes frames through:
1. **Frame stacking** (`_get_stacked_frames`, line 135): 4 frames selected at `frame_skip` intervals, padded with duplicates if insufficient.
2. **Grayscale conversion** (line 189): RGB → grayscale using standard perceptual weights `[0.2989, 0.5870, 0.1140]`.
3. **Viewport extraction** (`_get_image_observation`, line 170): 128×128 pixel window centered on Link's position with replicate padding at edges.
4. **NatureCNN** (`models.py`, lines 150–179): Standard 3-layer CNN (Conv2d 8×8 stride 4 → Conv2d 4×4 stride 2 → Conv2d 3×3 stride 1) → flatten → Linear → 256 features.

The CNN processes 65,536 input values (4×128×128) through 2.4M parameters into 256 features. This represents **78% of the CombinedExtractor's 328-dimensional output** and **97.6% of all trainable parameters** (2,437,536 of 2,497,733 total).

### Enemy Features (24 values)

Per enemy slot (4 slots), 6 features (`_get_enemy_features`, line 253):

| Index | Feature | Source |
|-------|---------|--------|
| 0 | presence | 0 or 1 |
| 1 | closeness | `_distance_to_proximity(distance, scale=100)` → [0.1, 1.0] |
| 2-3 | direction vector | Normalized (dx, dy) from Link to enemy |
| 4-5 | facing direction | Enemy's `direction.vector` (cardinal) |

Enemies are sorted by distance (`zelda_game.py`, line 84). The list uses `active_enemies` if any exist, otherwise falls back to all `enemies` (line 260–261).

### Enemy ID Embeddings (16 values after embedding)

Raw enemy type IDs (0–48 as `ZeldaEnemyKind` enum values) are passed as integers (`_get_enemy_ids`, line 272). The network embeds these through `nn.Embedding(150, 4)` (150 types, 4-dimensional embedding) in `CombinedExtractor` (line 239). This is well-designed — learned embeddings can capture enemy behavior similarities without manual feature engineering.

**Bug/Note**: `NUM_ENEMY_TYPES` is 49 in `observation_wrapper.py` (line 27), but `CombinedExtractor.__init__` uses `num_enemy_types=150` (default, line 219), and `SharedNatureAgent` passes 150 (line 308). The Discrete space says `Discrete(ENEMY_COUNT)` = `Discrete(4)`, which is the number of slots, not types. The mismatch between observation space declaration (4) and actual ID range (0–72) is a minor inconsistency but doesn't cause runtime errors because the embedding table is large enough.

### Item Features (8 values)

Per item slot (2 slots), 4 features (`_get_item_features`, line 289):

| Index | Feature |
|-------|---------|
| 0 | presence |
| 1 | closeness |
| 2-3 | direction vector |

**Critical gap**: No item type information. Hearts, keys, rupees, bombs, triforce pieces — all are represented identically. The treasure item (from `state.treasure`) is prepended to the items list (line 296–297), so it takes priority in slot ordering.

### Projectile Features (10 values)

Per projectile slot (2 slots), 5 features (`_get_projectile_features`, line 308):

| Index | Feature |
|-------|---------|
| 0 | presence |
| 1 | closeness |
| 2-3 | direction vector |
| 4 | blockable flag (-1 or +1) |

This is the most complete entity representation — it includes a behavioral property (blockable) that helps the model decide whether to face the projectile (shield block) or dodge.

### Information Vector (14 values)

Built in `_get_information` (line 334):

| Indices | Feature | Encoding |
|---------|---------|----------|
| 0-3 | Objective direction | Multi-hot N/S/E/W (supports diagonal) |
| 4 | Get item objective | Binary: ITEM or TREASURE |
| 5 | Fight objective | Binary: FIGHT |
| 6-9 | Source direction | One-hot: direction Link entered from |
| 10 | Has active enemies | Binary |
| 11 | Has beams available | Binary |
| 12 | Low health | Binary: `health <= 1` |
| 13 | Full health | Binary: `is_health_full` |

### Neural Network Data Flow

```
Image (4,128,128) ─→ NatureCNN ─→ 256 dims ─┐
Enemy features (4,6) ─→ flatten ─→ 24 dims ──┤
Enemy IDs (4,) ─→ Embedding(150,4) ─→ 16 dims┤
Item features (2,4) ─→ flatten ─→ 8 dims ────┤
Proj features (2,5) ─→ flatten ─→ 10 dims ───┤  concatenate
Information (14,) ─→ float ─→ 14 dims ────────┘     ↓
                                                  328 dims
                                                     ↓
                                              ┌──────┴──────┐
                                              ↓              ↓
                                        MLP policy     MLP value
                                        (3 layers)     (3 layers)
                                           64              64
                                            ↓               ↓
                                      action_net      value_net
                                         12               1
```

### Proximity Function

The `_distance_to_proximity` function (line 325) maps pixel distance to [0.1, 1.0]:
- distance ≤ 5px → 1.0 (maximum proximity)
- distance ≥ 100px → 0.1 (minimum proximity)
- Linear interpolation between

This means enemies at 100px or further all read as 0.1 — the model cannot distinguish "across the room" from "slightly out of range." Given the NES gameplay area is ~256×176 pixels, enemies beyond ~40% of the screen width are at minimum proximity resolution.

## Analysis

### Wavefront Information Is Computed But Not Shared

The wavefront pathfinding solution is computed every step in `state_change_wrapper.py` (line 372):
```python
state.wavefront = state.room.calculate_wavefront_for_link(objectives.targets)
```

The critic uses this wavefront to produce ±0.05/0.06 movement rewards (closer/further from optimal path). But the observation wrapper **never accesses the wavefront**. The model must discover the reward-maximizing path purely through trial-and-error on each unique room layout.

This is the single most significant observation-reward misalignment in the system. The wavefront already answers "which direction should I go?" but the model must re-derive this from the image and reward signal every time it encounters a room.

### Image Dominance Creates a Bottleneck

The image path (256 dims, 78% of combined features) processes 65,536 values through 2.4M parameters. The structured features (72 dims, 22%) pass through essentially no parameters before concatenation. This means:

1. The policy MLP's 328→64 linear layer must learn to weigh 256 CNN features against 72 structured features. The CNN features will dominate gradient flow due to magnitude.
2. Structured features contribute a minority of the gradient signal despite containing precise, noise-free game state information.

### Binary Health Creates a Dead Zone

Health encoding as two bits (low, full) creates three states:
- `low_health=1, full_health=0`: health ≤ 1 (dangerous)
- `low_health=0, full_health=0`: 1 < health < max (most common state)
- `low_health=0, full_health=1`: health = max (beams available)

The middle range (e.g., 2 hearts vs 5 hearts vs 10 hearts) is completely invisible. The model cannot learn health-proportional risk aversion. A single `health_fraction` float would fix this with zero architectural cost.

### Enemy Embeddings Are Good but Underpowered

The `nn.Embedding(150, 4)` approach for enemy types is well-designed. It allows the network to learn that RedDarknut (0x0C) behaves differently from GreenGel (0x14). However:

1. The 4-dimensional embedding is small relative to the behavioral variety among 49 enemy types.
2. No explicit behavioral features are provided (ranged attack, movement pattern, vulnerability).
3. The embedding must be learned entirely from the reward signal — there's no auxiliary loss encouraging meaningful grouping.

### Missing Item Type Information

The item feature vector has no type indicator. This means:
- A key (critical for progress) looks identical to a bomb drop (marginal)
- A heart (health recovery) looks identical to a rupee (economy)
- The model can only distinguish items by their position in the image viewport

Since the critic rewards REWARD_MAXIMUM (1.0) for key equipment pickups, the model gets huge rewards for collecting certain items but can't tell which item gives the big reward from the features alone.

### No Weapon Animation State

The model has no structured indication of whether:
- A sword swing is currently active
- Beams are in flight
- A bomb is ticking/exploding
- An arrow is flying

The `has_beams` feature indicates availability, not current state. This forces the model to rely entirely on the viewport image to determine weapon animation status, which affects timing decisions.

## Repro Scripts

### `observation_space_analysis.py`
Static analysis of the observation space structure. Enumerates all features, identifies information gaps, analyzes the proximity function, and identifies observation-reward misalignments. Key output:
- 65,536 image values compressed to 256 features
- 60 structured values (72 after embedding)
- 9 identified information gaps
- 5 observation-reward misalignments

### `observation_network_analysis.py`
Builds the actual `SharedNatureAgent` network and counts parameters. Key output:
- 2,497,733 total trainable parameters
- CNN: 2,437,536 params (97.6% of total)
- Enemy embedding: 600 params
- MLP extractor: 58,752 params
- Combined feature dimensions: 328 (78% image, 22% structured)
- Adding 40 proposed features would shift the ratio to 69.6%/30.4%

## Research

### Multi-Modal Observation Fusion in RL

The standard approach for combining image and structured observations is **separate encoding followed by concatenation** (Mnih et al., 2015; Baker et al., 2019). The Triforce architecture follows this pattern correctly with `CombinedExtractor`.

Recent work on **Hierarchical Adaptive Value Estimation (HAVE)** (OpenReview, 2024) proposes dynamically weighting modality contributions based on task context. This could help — during navigation, structured features (wavefront, tiles) matter more; during combat, image features (enemy positions, animation states) may dominate.

### Privileged Information / Asymmetric Actor-Critic

The wavefront is a form of **privileged information** — computed by the system but not observable by the agent. The RL literature offers two approaches:

1. **Asymmetric Actor-Critic** (Pinto et al., 2017): Give the wavefront to the value function (critic) only, not the policy. This improves value estimation without making the policy dependent on privileged data. However, in Triforce, the wavefront IS always available at inference time (it's computed from the room layout), so sharing it with the policy is safe.

2. **Teacher-Student Distillation** (Czarnecki et al., 2019): Train a teacher with full wavefront access, then distill to a student that only sees pixels. This is overkill here — the wavefront is always computable at runtime.

Since the wavefront is always available (computed from the tile map which is in RAM), it's not truly "privileged" — it's just not being passed through. The simplest fix is to add it to the observation.

### Entity-Centric RL

Clemens Winter's work on **Entity-Based RL** (2023) argues for representing each game entity as a feature vector with type embeddings, processed by attention or set-based architectures. The Triforce approach (fixed slots sorted by distance) is simpler but loses information about entity ordering and handles variable counts by truncation/padding.

The current 4-slot enemy limit could miss important enemies. In Zelda rooms with 6 enemies (e.g., Dungeon 1 rooms with 4 Stalfos + 2 Keese), the 2 farthest enemies are invisible to the structured features.

### Local Tile Maps in Navigation RL

Grid-world RL agents commonly receive a local observation window around the agent (e.g., 5×5 or 7×7 tiles). For NES Zelda, a 5×5 walkability grid centered on Link's tile would provide:
- Immediate wall information (prevents wall bumping)
- Doorway visibility (aids exit-finding)
- Only 25 additional binary features

This is far more sample-efficient than learning walkability from pixels. The tile layout is already computed (`zelda_game.py:current_tiles`, line 237) and walkability is cached per room (`room.py:walkable`, line 107).

## Findings

1. **The wavefront is computed every step but never included in observations** (`state_change_wrapper.py:372`). The critic rewards/penalizes based on wavefront distance changes (±0.05/0.06), but the model receives only objective direction (N/S/E/W) with no distance or path information. This is the largest observation-reward gap.

2. **Image features dominate the combined representation**: 256 of 328 dimensions (78%) come from the CNN, and the CNN accounts for 97.6% of trainable parameters. Structured features (precise, noise-free game state) are compressed into just 72 dimensions (22%).

3. **Health is encoded as two binary features** (low and full) in `_get_information` (line 342–343), creating a dead zone where 1 < health < max is invisible. The model cannot learn health-proportional risk strategies.

4. **Item type information is entirely absent** from item features. The `_get_item_features` method (line 289) only encodes presence, proximity, and direction. Hearts, keys, rupees, and equipment drops are indistinguishable.

5. **The proximity function saturates at 100px** (`DISTANCE_SCALE=100.0`, line 22). Enemies beyond 100px all read as 0.1 (minimum). On the 256×176px gameplay area, this means ~40% of the screen width beyond Link is at minimum resolution.

6. **Enemy type embeddings are architecturally sound** — `nn.Embedding(150, 4)` allows learned type distinctions. But the 4-dimensional embedding space is small for 49 behaviorally diverse enemy types, and there's no auxiliary objective encouraging meaningful clustering.

7. **The `enemy_id` observation space is declared as `Discrete(ENEMY_COUNT)` = `Discrete(4)`** (line 82), but actual enemy IDs range 0x01–0x48 (1–72). The space declaration is misleading (it refers to slot count, not type count), though it doesn't cause runtime errors because the embedding table (size 150) accommodates all IDs.

8. **No weapon animation state is observable in structured features**. The model cannot tell from features alone whether beams are in flight, a bomb is ticking, or a sword swing is active. Only `has_beams` (availability) is included.

9. **No local walkability information** is provided. The room's `walkable` tensor is computed and cached (`room.py:107`), and the current tile layout is available (`zelda_game.py:237`), but neither appears in the observation.

10. **Door lock/bar status is absent**. The game tracks door states (`zelda_game.py:224-234`), but the observation doesn't expose them. The model must learn locked door patterns from subtle visual tile differences in the grayscale viewport.

## Recommendations

1. **Add wavefront gradient direction and normalized distance to the information vector** (addresses Finding 1). Compute the best direction by checking which of Link's 4 neighboring tiles has the lowest wavefront value, and normalize the current wavefront distance by the room maximum. This adds 5 features (4 directional + 1 distance) and directly bridges the largest observation-reward gap. The wavefront is always available at runtime, so this is not privileged information.

2. **Add a 5×5 local walkability grid to observations** (addresses Findings 1, 9). Extract `room.walkable[link.tile.x-2 : link.tile.x+3, link.tile.y-2 : link.tile.y+3]` as 25 binary features. This provides immediate obstacle awareness without requiring the model to learn wall patterns from pixels, and significantly reduces wall collision frequency.

3. **Replace binary health with a continuous health fraction** (addresses Finding 3). Add `health_fraction = link.health / link.max_health` as a single float [0.0, 1.0] to the information vector. This enables health-proportional risk aversion. The two binary features (low_health, full_health) can be retained for backward compatibility or removed if retraining.

4. **Add item type IDs to item features** (addresses Finding 4). Extend item slots from 4 to 5 features by appending an item type indicator. Use a simple categorical encoding or learned embedding similar to enemy IDs. This allows the model to prioritize keys over hearts or rupees.

5. **Add door states to the information vector** (addresses Finding 10). Four additional features (N/S/E/W) with values for open/locked/barred. This prevents the model from repeatedly walking into locked doors and enables key-usage planning.

6. **Consider increasing the proximity distance scale from 100 to 150–200px** (addresses Finding 5). This gives the model better resolution for planning at medium range, where navigation decisions are most impactful. Test this carefully — it changes the meaning of existing proximity values.

7. **Add weapon animation state flags** (addresses Finding 8). Three binary features for sword/beam/bomb active states, sourced from `link.get_animation_state()`. This allows the model to time attacks and avoid redundant weapon activations without relying on the viewport image.

8. **Fix the `enemy_id` observation space declaration** (addresses Finding 7). Change `Discrete(ENEMY_COUNT)` to a more accurate space declaration (e.g., `Box` or `MultiDiscrete`) that reflects the actual ID range. This is a documentation/correctness fix that doesn't affect runtime behavior.

9. **Consider increasing enemy embedding dimension from 4 to 8** (addresses Finding 6). With 49 enemy types having diverse behaviors (melee vs ranged, burrowing vs flying, vulnerable vs invulnerable), 4 dimensions may be insufficient for learning meaningful distinctions. Alternatively, add explicit behavioral features (ranged flag, invulnerable flag) alongside the embedding.

10. **Evaluate the wavefront-in-observation tradeoff carefully** (addresses Finding 1). If the wavefront is fully exposed, the model may learn to "just follow the wavefront" without developing spatial reasoning. An asymmetric approach — wavefront in the value function only — could provide learning signal without shortcutting the policy. However, since the wavefront captures the optimal path by definition, sharing it with the policy is likely net positive for overall performance.
