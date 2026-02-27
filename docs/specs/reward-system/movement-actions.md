# Movement Action Abstraction

Investigation of how movement actions are abstracted in the FrameSkipWrapper, the
consequences of tile-by-tile movement for PPO training, directional asymmetry, and
alternative approaches from the RL literature.

## Current Behavior

### Movement Execution Pipeline

Movement flows through three layers:

1. **PPO selects an action** from the discrete action space (`ml_ppo_rollout_buffer.py:97–102`)
2. **ZeldaActionSpace** maps the integer to an `ActionTaken` with `kind=MOVE` and a direction (`action_space.py:193–201`)
3. **FrameSkipWrapper → ZeldaCooldownHandler._act_movement()** executes the NES button hold until a tile boundary is crossed (`frame_skip_wrapper.py:165–210`)

The key code in `_act_movement` (`frame_skip_wrapper.py:165–210`):

```python
def _act_movement(self, action, start_pos, frame_capture):
    old_tile_index = start_pos.tile_index
    stuck_max = 8
    stuck_count = 0
    prev_pos = start_pos
    for _ in range(MAX_MOVEMENT_FRAMES):   # MAX_MOVEMENT_FRAMES = 16
        # step one NES frame
        pos = Position(info['link_x'], info['link_y'])
        new_tile_index = pos.tile_index
        match action.direction:
            case Direction.N:
                if old_tile_index.y != new_tile_index.y:
                    break                  # stop immediately
            case Direction.S:
                if old_tile_index.y != new_tile_index.y:
                    self._act_for(action, WS_ADJUSTMENT_FRAMES, ...)  # +4 extra frames
                    break
            case Direction.E:
                if old_tile_index.x != new_tile_index.x:
                    break                  # stop immediately
            case Direction.W:
                if old_tile_index.x != new_tile_index.x:
                    self._act_for(action, WS_ADJUSTMENT_FRAMES, ...)  # +4 extra frames
                    break
        if prev_pos == pos:
            stuck_count += 1
        if stuck_count >= stuck_max:
            break
```

### Tile Size and Movement Speed

- **Tile size**: 8×8 pixels (`zelda_enums.py:381`: `TileIndex(x // 8, (y - GAMEPLAY_START_Y) // 8)`)
- **Link's speed**: 1.5 pixels/frame average (alternates 1px and 2px per frame in NES assembly)
- **Frames per tile crossing**: ~5.3 frames average (5 or 6 depending on sub-pixel alignment)

### Directional Asymmetry

The `WS_ADJUSTMENT_FRAMES = 4` constant (`frame_skip_wrapper.py:16`) creates a significant
asymmetry:

| Direction | Frames per tile | NES time (at 60fps) | Extra frames |
|-----------|----------------|---------------------|--------------|
| North     | ~5-6           | ~83-100ms           | 0            |
| East      | ~5-6           | ~83-100ms           | 0            |
| South     | ~9-10          | ~150-167ms          | +4           |
| West      | ~9-10          | ~150-167ms          | +4           |

**South/West takes 1.75× as long as North/East.** The extra 4 frames continue holding the
direction after the tile boundary is crossed, pushing Link ~6 pixels (0.75 tiles) past the
boundary. This means:
- After moving North, Link is at the TOP edge of a tile
- After moving South, Link is ~6px INTO the next tile
- Starting positions for subsequent actions are direction-dependent

### Wall Collision and Stuck Detection

When Link hits a wall (`frame_skip_wrapper.py:204–208`):
1. Position doesn't change for `stuck_max = 8` consecutive frames
2. Movement breaks after 8 frames (133ms at 60fps) of wasted time

**Action masking** (`action_space.py:227–234`): When `prevent_wall_bumping=True`, a failed
MOVE direction is masked out. But the mask resets when ANY direction succeeds. This prevents
consecutive same-direction bumps but allows alternating bump patterns (e.g., bump N, move E,
bump N, move E).

### PPO Buffer Impact

With `TARGET_STEPS = 2048` (`ml_ppo.py:28`):

| Metric | Tile-by-tile | Multi-tile (4 avg) |
|--------|-------------|-------------------|
| Movement actions/room | ~12 | ~3 |
| Combat actions/room | ~12 | ~12 |
| Total actions/room | ~24 | ~15 |
| Movement fraction | 50% | 20% |
| Rooms per buffer | ~85 | ~137 |
| Combat sample ratio | 50% | 80% |

The current system dedicates **half the PPO buffer to tiny movement decisions**, leaving
fewer samples for combat learning. Each movement decision is a single tile step carrying a
0.05 reward signal.

### Reward Per Real Time

The direction asymmetry creates an inherent reward-rate bias:

| Action | Reward | Frames | Reward/frame |
|--------|--------|--------|-------------|
| Move North (success) | +0.05 | ~6 | 0.0083 |
| Move South (success) | +0.05 | ~10 | 0.0050 |
| Wall collision | -0.25 | ~8 | -0.0312 |
| Sword hit | +0.25 | ~15 | 0.0167 |
| Sword miss | -0.06 | ~15 | -0.0040 |

North movement earns 1.7× the reward-per-frame of South movement, even though both
achieve the same strategic result (one tile of progress). Combat actions take ~2.5×
longer than North movement, meaning the model sees proportionally fewer combat observations.

## Analysis

### GAE Advantage Propagation

GAE with γ=0.99, λ=0.95 propagates future penalties backward through movement chains
(computed in `scripts/repros/movement_action_abstraction.py`):

**Optimal 6-tile path (all toward objective):**
- Advantages: [0.243, 0.209, 0.172, 0.133, 0.091, 0.047]
- First step gets 5.2× the advantage of last step
- All positive → correct signal

**Realistic path (4 toward, 1 lateral, 1 wall bump):**
- Advantages: [-0.025, -0.076, -0.131, -0.189, -0.251, -0.253]
- ALL steps get negative advantage, including the 4 correct moves
- The wall bump at the end poisons the entire trajectory
- First correct move gets advantage -0.025 (should be positive)

This demonstrates that GAE backward propagation over many small movement steps causes
**reward dilution and credit misassignment**. A multi-tile approach that combines 4 correct
steps into one action would isolate the wall-bump failure as a separate decision point.

### Movement Dominates Training Signal

With tile-by-tile movement, the model makes ~200 decisions per room traversal. In a 2048-step
buffer covering ~85 rooms, approximately 1024 entries are movement and only ~1024 are combat.
Since combat is where the model needs the most training signal (hitting enemies, dodging,
timing), this 50/50 split is suboptimal.

### The WS Adjustment Is a Band-Aid

The `WS_ADJUSTMENT_FRAMES = 4` exists because NES Link's sprite alignment differs by direction.
When moving South/West, the tile boundary is at the leading edge of the sprite, and the extra
frames ensure Link fully enters the new tile. But this creates cascading problems:
1. Direction-dependent timing
2. Direction-dependent reward rates
3. Asymmetric sub-pixel alignment for subsequent actions

A better fix would be to normalize ALL directions to the same behavior — either always add
adjustment frames or use a position-target rather than tile-boundary detection.

## Repro Scripts

### `scripts/repros/movement_frame_analysis.py`
Computes movement timing, tile boundary positions, PPO buffer composition, WS adjustment
impact, and wall-bump masking behavior. Key outputs:
- S/W takes 1.75× as many frames as N/E
- Wall bumps waste 133ms per occurrence
- Multi-tile movement would improve buffer efficiency by 1.6×

### `scripts/repros/movement_action_abstraction.py`
Analyzes reward accumulation under different movement strategies, GAE advantage computation
showing backward penalty propagation, action space size comparison, and reward-per-frame
directional bias. Key outputs:
- GAE propagates wall bump penalties through 4 correct preceding moves
- North earns 1.7× reward-per-frame vs South
- Multi-tile reduces action space impact to zero (same space, different semantics)

## Research

### Options Framework (Sutton, Precup & Singh, 1999)
The seminal **options framework** defines temporally extended actions ("options") with:
1. **Initiation set**: states where the option can start
2. **Internal policy**: sub-policy executed while option is active
3. **Termination condition**: when to return control to the high-level policy

Multi-tile movement maps naturally to this: the initiation set is any navigable tile, the
internal policy is "hold direction", and termination conditions are wall/enemy/door detection.

### Option-Critic Architecture (Bacon, Harb & Precup, 2017)
Learns both which options to select and their termination conditions end-to-end. For Triforce,
this could mean letting the network learn when to stop a multi-tile move rather than
hand-coding interrupt conditions.

### Atari Frame Skip (Standard Practice)
Standard Atari RL uses fixed action repeat (typically 4 frames per decision). This is analogous
to Triforce's tile-by-tile approach but at a much shorter timescale (~67ms per Atari decision
vs ~100-167ms per Triforce movement). The difference is that Atari's frame skip is uniform
across all actions, while Triforce's is direction-dependent and highly variable.

### PPO with Variable-Length Actions
Standard PPO assumes fixed-length steps. Variable-length actions (as in multi-tile movement)
require care with advantage estimation. Two established approaches:
1. **Sum rewards over the macro-action** and assign to the decision step (what Triforce
   already does implicitly — the frames within a tile-move are not separate PPO steps)
2. **Discount rewards within the macro-action** using the same γ, treating the macro-action
   as a single step with accumulated reward

Since Triforce already treats each tile-move as a single PPO step (not 5-6 individual frame
steps), extending this to multi-tile moves is architecturally consistent.

### Grid World Navigation in RL
Grid-world research shows that **coarser action granularity improves learning speed** when
the environment has clear spatial structure. Zelda's dungeon rooms are effectively grid worlds
with obstacles, making multi-tile movement well-suited.

## Findings

1. **Movement consumes 50% of the PPO buffer** for decisions that carry only 0.05 reward
   each, while combat (where learning is most needed) shares the other 50%.

2. **South/West movement takes 1.75× as many NES frames as North/East** due to
   `WS_ADJUSTMENT_FRAMES = 4` (`frame_skip_wrapper.py:16`), creating a directional bias
   in reward-per-frame rates (North earns 1.7× the reward/frame of South).

3. **GAE backward propagation over many small movement steps causes credit misassignment.**
   A wall bump at the end of a trajectory makes all preceding correct moves receive negative
   advantages, discouraging the model from making correct initial moves.

4. **Wall collision detection wastes 8 frames (133ms)** per bump through the stuck_max
   counter (`frame_skip_wrapper.py:177`), and the action masking system only prevents
   consecutive same-direction bumps, not alternating patterns.

5. **The action space size is NOT the bottleneck.** Multi-tile movement (Option B) requires
   zero changes to the action space — it changes only execution semantics within the
   frame skip wrapper. The 4-direction MOVE actions remain identical.

6. **Movement is already treated as a macro-action** in the PPO pipeline. Each tile-move is
   5-10 NES frames but occupies a single PPO buffer slot. Extending this to multi-tile
   moves is architecturally consistent with the existing design.

7. **The WS adjustment creates asymmetric sub-pixel positions** after S/W moves (~6px past
   boundary) vs N/E moves (exactly at boundary), which affects subsequent tile-crossing
   timings and wavefront position evaluations.

## Recommendations

1. **Implement multi-tile movement (Option B from todo) as the primary approach** (addresses
   findings 1, 3, 5, 6). Change `_act_movement` to continue holding direction after a tile
   boundary crossing, checking interrupt conditions at each tile:
   - Wall/obstacle ahead → stop
   - Enemy within threat radius (~3 tiles) → stop
   - Door threshold reached → stop
   - Max tiles reached (configurable, ~6) → stop
   This requires changes only in `frame_skip_wrapper.py:165–210`.

2. **Normalize directional timing** (addresses findings 2, 7). Replace the
   `WS_ADJUSTMENT_FRAMES` hack with position-target movement: for ALL directions, continue
   until Link's position is at a consistent alignment within the destination tile (e.g.,
   tile center or leading edge). This eliminates the N/E vs S/W asymmetry entirely.

3. **Scale movement reward proportionally to tiles moved** (addresses findings 1, 3). When
   a multi-tile action crosses N tiles toward the objective, reward should be
   `N × MOVE_CLOSER_REWARD` rather than a single fixed reward. This preserves the total
   reward for a path while compressing it into fewer buffer entries. Modify
   `critique_movement` (`critics.py:333–381`) to use wavefront delta rather than
   single-step comparison.

4. **Reduce stuck_max or replace with tile-map lookup** (addresses finding 4). Instead of
   waiting 8 frames to detect a wall, check the room's walkability map (`room.walkable`)
   BEFORE attempting movement. If the destination tile is unwalkable, skip the move
   immediately and apply the wall penalty. This saves ~8 frames per collision and makes
   wall detection instant.

5. **Keep the action space unchanged** (addresses finding 5). Do not pursue Option D
   (direction × duration) or Option C (destination-based) initially. Option B achieves the
   buffer composition improvement without increasing action space complexity. Option D's 75%
   action space increase and Option C's variable-target masking add complexity without clear
   benefit at this stage.

6. **Add interrupt conditions incrementally** (addresses finding 6). Start with
   wall-only interrupts (simplest case), then add enemy proximity interrupts once
   wall-interrupt movement is validated. Enemy threat radius should be configurable and
   tested empirically (suggested starting point: 3 tiles / 24 pixels).

7. **Measure buffer composition before and after** any movement changes. Track the ratio of
   MOVE vs non-MOVE actions in the PPO buffer, and compare training curves. The primary
   success metric is whether combat learning speed improves (more hit/kill rewards per
   buffer) without degrading navigation success rate.
