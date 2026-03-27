# Recording System Specification (`record.py`)

## Overview

A command-line tool to record gameplay videos of the trained RL agent playing The Legend of Zelda. Produces polished, 1920×1080 MP4/H.264 videos suitable for YouTube upload or further editing. No sound.

## Usage

```bash
python record.py <model_path> <scenario> [options]
```

### Positional Arguments

| Argument | Description |
|----------|-------------|
| `model_path` | Path to a `.pt` model file |
| `scenario` | Scenario name from `triforce.yaml` (e.g., `full-game`, `dungeon1`) |

### Optional Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--output-path DIR` | `./recording/` | Directory to save videos. Files are named `triforce_NN.mp4` with `NN` auto-incremented to the next available number. |
| `--count N` | `1` | Number of videos (full playthroughs) to record. Runs sequentially. |
| `--only-wins` | off | Only save videos where the scenario ended in success (determined by `SuccessMetric` — the end reason starts with `"success"`). Discard non-winning runs. |
| `--show-attention [N]` | off | Overlay attention heatmap on the gameplay. Optional `N` selects a specific head (1-indexed). Omitting `N` shows the combined view (`.max()` across all heads, same as the debugger). If the model has no attention, this flag is silently ignored. |
| `--no-observation` | off | Disable the observation panel. |
| `--no-model-actions` | off | Disable the action probabilities panel. |
| `--device DEVICE` | `cpu` | PyTorch device for model inference (`cpu` or `cuda`). Does not affect the emulator. |
| `--fps {30,60}` | `60` | Video frame rate. 60 matches NES native rate; 30 produces smaller files. |

## Video Layout

**Resolution**: 1920×1080 (16:9), MP4 container, H.264 codec.

The video is composed of three areas arranged to fill the 16:9 frame with a dark/black background:

```
┌─────────────────────────────────────────────────────────┐
│                    1920 × 1080                          │
│                                                         │
│  ┌──────────────────────────┐  ┌──────────────────────┐ │
│  │                          │  │   OBSERVATION PANEL   │ │
│  │                          │  │                       │ │
│  │     GAMEPLAY AREA        │  │  Enemies on screen    │ │
│  │     (NES frame,          │  │  Entity types, HP,    │ │
│  │      scaled ~4x,         │  │  stun, ⚠/⚔ icons    │ │
│  │      with optional       │  │                       │ │
│  │      attention overlay)  │  │  Objective direction  │ │
│  │                          │  │  Source direction     │ │
│  │                          │  │  Booleans: Enemies,   │ │
│  │                          │  │  Beams, Low/Full HP,  │ │
│  │                          │  │  Clock                │ │
│  │                          │  ├──────────────────────┤ │
│  │                          │  │  ACTION PROBABILITIES │ │
│  │                          │  │                       │ │
│  │                          │  │  ┌─────┐  ┌─────┐    │ │
│  │                          │  │  │walk │  │sword│    │ │
│  │                          │  │  │ gif │  │ gif │    │ │
│  │                          │  │  └─────┘  └─────┘    │ │
│  │                          │  │  ┌─────┐  ┌─────┐    │ │
│  │                          │  │  │bomb │  │boom │    │ │
│  │                          │  │  │ gif │  │ gif │    │ │
│  │                          │  │  └─────┘  └─────┘    │ │
│  └──────────────────────────┘  └──────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Layout Rules

1. **Gameplay area** occupies the left portion. The NES frame (256×240) is scaled up to fill as much vertical space as possible while maintaining its native aspect ratio (~4x scale ≈ 1024×960, centered vertically within 1080). This should be the dominant visual element — large enough to clearly see gameplay.

2. **Side panel** occupies the right portion (the remaining ~896px width). It is split vertically:
   - **Top**: Observation panel
   - **Bottom**: Action probabilities panel

3. When `--no-observation` is used, the action panel expands to fill the full right side.
4. When `--no-model-actions` is used, the observation panel expands to fill the full right side.
5. When both are disabled, the gameplay area centers in the full 1920×1080 frame.

## Observation Panel

Renders entity and game-state information as text/icons onto the video frame. Uses **FiraCode Nerd Font** for clean monospace rendering.

### Contents

Mirrors what `triforce_debugger/observation_panel.py` shows, adapted for video:

- **Entities**: Each entity row shows:
  - Type name (from `ENTITY_TYPE_NAMES`)
  - HP (if > 0), displayed as integer (e.g., `HP:3`)
  - Stun timer (if > 0)
  - ⚠ icon if `hurts > 0.5`
  - ⚔ icon if `killable > 0.5`

- **Objective & Source**: Rendered as directional indicators (N/S/E/W arrows or text labels).

- **Boolean indicators**: `Enemies`, `Beams`, `Low HP`, `Full HP`, `Clock` — shown as colored text/icons (active = bright, inactive = dim/grey).

### Rendering

- White text on dark background
- FiraCode Nerd Font at a size readable at 1080p
- Nerd Font icons where appropriate (e.g., sword icon for killable, warning for hurts)

## Action Probabilities Panel

Displays the model's post-masking action probabilities using animated GIF sprites surrounded by directional probability values.

### Layout

Four action types displayed in a 2×2 grid:

```
  ┌─────────────────┐  ┌─────────────────┐
  │     12.5%        │  │      0.0%        │
  │  8.1% 🚶 44.2%  │  │  0.0% ⚔️  0.0%  │
  │     35.2%        │  │      0.0%        │
  │   MOVEMENT       │  │  SWORD/BEAMS     │
  └─────────────────┘  └─────────────────┘
  ┌─────────────────┐  ┌─────────────────┐
  │      0.0%        │  │      0.0%        │
  │  0.0% 💣 0.0%   │  │  0.0% 🪃  0.0%  │
  │      0.0%        │  │      0.0%        │
  │     BOMB         │  │   BOOMERANG      │
  └─────────────────┘  └─────────────────┘
```

### Per Action Type

Each cell contains:
1. **Center**: The animated GIF sprite (from `img/`)
2. **Above**: N probability (e.g., `12.5%`)
3. **Below**: S probability
4. **Left**: W probability
5. **Right**: E probability

### Probability Rules

- Probabilities are **post-masking** (after the mask is applied and values are renormalized), matching how `action_table.py` computes the "PROBABILITY" column.
- Rounded to 1 decimal place (e.g., `12.5%`).
- If a probability is **< 0.1% before rounding** (i.e., raw value < 0.001), **do not display it** — leave that position empty (don't display 0%, display NOTHING). This avoids visual clutter from zero-probability actions.
- When all four directions for an action type are below threshold, show the sprite greyed/dimmed with no probabilities displayed.

### SWORD/BEAMS Combination

SWORD and BEAMS are mutually exclusive after masking (only one can be unmasked at a time). They share the `magic-sword.gif` sprite. The displayed probabilities combine both:
- Sum the renormalized probabilities of SWORD and BEAMS per direction (for safety, in case both are partially unmasked — in practice only one will have non-zero values).
- Display the combined result around the single `magic-sword.gif` sprite.

### GIF Animation

The GIF sprites (especially `walking.gif` with 2 frames at 250ms) should animate in the video:
- At 60 FPS: alternate sprite frames every ~15 video frames (250ms).
- At 30 FPS: alternate sprite frames every ~8 video frames.
- Single-frame GIFs (`magic-sword.gif`, `bomb.gif`) display as static images.
- Scale sprites up 2-3x from their native size for visibility at 1080p.

## Attention Overlay

When `--show-attention` is enabled:

1. After each model inference, call `model.forward_with_attention(obs)` to get spatial attention weights (shape: `(num_heads, H', W')`).
2. Apply the same visualization as `game_view.py`:
   - **Combined** (no head specified): `weights.max(axis=0)` across heads (NOT average).
   - **Specific head** (`--show-attention N`): Use head `N-1` (1-indexed for the user).
3. Normalize to [0, 1] using global min/max across all heads.
4. Upsample to gameplay area resolution via bilinear interpolation.
5. Apply JET colormap and composite over the gameplay frame at ~55% opacity (alpha ≈ 140/255).

## Success Detection

For `--only-wins`, a run is a "success" when `SuccessMetric.end_scenario()` would record `1` — i.e., the scenario's end `reason` starts with `"success"`. This is determined by the `ScenarioWrapper` end conditions.

Implementation approach: record all frames to a file. On scenario completion, check the end reason. If it's a success (or `--only-wins` is not set), finalize the video file. Otherwise, discard.

## File Naming

Videos are saved to `<output-path>/triforce_NN.mp4`:
- `NN` is zero-padded to 2 digits (e.g., `triforce_00.mp4`, `triforce_01.mp4`).
- The next available number is determined by scanning the output directory for existing `triforce_*.mp4` files.
- With `--count 5`, produces up to 5 videos. With `--only-wins`, may need to run more episodes than `count` to find enough wins.

## Architecture

### Dependencies

- **OpenCV** (`cv2`): Video encoding (H.264 via `cv2.VideoWriter` with `mp4v` or `avc1` fourcc).
- **Pillow** (`PIL`): GIF frame extraction and sprite loading.
- **NumPy**: Frame composition and attention overlay math.
- **FiraCode Nerd Font**: Render text via Pillow's `ImageFont.truetype()`. Must be installed on the system. Fall back to default monospace if unavailable.

### Module Structure

A single new file `record.py` at the project root (alongside `debug.py`, `evaluate.py`, `train.py`).

Internal organization:
- **`VideoRecorder`** class: Manages OpenCV VideoWriter, frame buffering for `--only-wins`, file naming.
- **`FrameComposer`** class: Composes the 1920×1080 frame from gameplay + panels. Handles layout, text rendering, sprite animation.
- **`ObservationRenderer`**: Renders the observation panel as a numpy array (PIL-based text rendering).
- **`ActionRenderer`**: Renders the action probabilities panel with animated GIF sprites and directional probabilities.
- **`AttentionOverlay`**: Applies the JET heatmap overlay to the gameplay frame (port of `game_view.py` logic, without Qt dependency).
- **Main loop**: Uses `EnvironmentBridge` pattern (from `environment_bridge.py`) to drive the environment, collect observations, run model inference, and compose frames.

### Data Flow (per frame)

```
1. env.step(action) → frames, state_change, observation, rewards
2. For each NES frame in state_change.frames:
   a. Scale gameplay frame to layout size
   b. If attention: overlay heatmap
   c. Render observation panel (entities, directions, booleans)
   d. Render action panel (sprites + probabilities)
   e. Compose into 1920×1080 canvas
   f. Write to VideoWriter (or buffer if --only-wins)
3. On episode end:
   - If --only-wins and not success: discard buffer
   - Else: finalize video, increment counter
```

### Reuse from Existing Code

- **`evaluate.py`**: Model loading pattern (`_load_network_from_path`, `_make_env_from_network`).
- **`environment_bridge.py`**: `ModelSelector.get_probabilities()` for post-mask probability extraction, `get_attention_weights()` for attention.
- **`game_view.py`**: JET colormap LUT (`_build_jet_lut`), attention upsampling/compositing logic.
- **`observation_panel.py`**: Entity formatting logic, direction extraction, boolean indicators.
- **`action_table.py`**: Probability renormalization logic (unmasked sum, per-direction display).

## Asset Files

Copy the following GIFs to `img/` in the repo:

| File | Source | Usage |
|------|--------|-------|
| `img/walking.gif` | `~/Downloads/walking.gif` | MOVE action (30×32, 2 frames, 250ms) |
| `img/magic-sword.gif` | `~/Downloads/magic-sword.gif` | SWORD/BEAMS combined (16×32, 1 frame) |
| `img/bomb.gif` | `~/Downloads/bomb.gif` | BOMB action (16×28, 1 frame) |
| `img/boomerang.gif` | `~/Downloads/boomerang.gif` | BOOMERANG action (28×54, 1 frame) |

## Edge Cases

- **Model without attention**: `--show-attention` is silently ignored (no error).
- **No scenarios matching success**: With `--only-wins` on a model that never wins, the tool runs indefinitely. Consider adding a `--max-attempts` limit (default: `count * 20`). Print progress: `"Completed 3/10 wins (15 attempts)"`.
- **Single emulator constraint**: Only one emulator per process. Videos are recorded sequentially.
- **`--only-wins` buffering**: For long scenarios, buffering all frames in memory may be expensive. Use a temporary file on disk that gets renamed to final name on success, or deleted on failure.
- **FiraCode not installed**: Fall back to Pillow's default font with a warning message.

## Linting

`record.py` must pass pylint with the project's `.pylintrc` configuration. Add `record.py` to the lint targets in the copilot instructions.

## Testing

Manual testing (no automated tests required for the recording tool):
1. `python record.py models/dungeon1.pt dungeon1 --count 1` — basic recording
2. `python record.py models/dungeon1.pt dungeon1 --only-wins --count 3` — win-only mode
3. `python record.py models/dungeon1.pt dungeon1 --show-attention` — attention overlay
4. `python record.py models/dungeon1.pt dungeon1 --no-observation --no-model-actions` — gameplay only
5. `python record.py models/dungeon1.pt dungeon1 --fps 30` — 30 FPS mode
6. Verify output plays correctly in VLC and uploads to YouTube without re-encoding.
