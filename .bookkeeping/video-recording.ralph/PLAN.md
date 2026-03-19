# Implementation Plan: video-recording

## Goal
Build `record.py` — a command-line tool that records 1920×1080 MP4 gameplay videos of the
trained Zelda RL agent, with observation panels, animated action probability sprites, and
optional attention heatmap overlay.

## Status
STATUS: IN_PROGRESS

## Tasks

- [x] Copy GIF assets from ~/Downloads to img/ directory (walking.gif, magic-sword.gif, bomb.gif, boomerang.gif)
- [x] Create record.py with CLI argument parsing (model_path, scenario, --output-path, --count, --only-wins, --show-attention, --no-observation, --no-model-actions, --device, --fps)
- [x] Implement environment setup: load model from .pt metadata, create env with render_mode='rgb_array' and translation=False, replicate EnvironmentBridge pattern for stepping/masks/probabilities
- [x] Implement VideoRecorder class: OpenCV H.264 MP4 writer, temp-file strategy for --only-wins (write to temp, rename on success, delete on failure), sequential file naming (triforce_NN.mp4)
- [x] Implement FrameComposer: 1920×1080 canvas layout, gameplay area scaled ~4x on left, side panel on right, handle --no-observation/--no-model-actions layout variants
- [x] Implement ObservationRenderer: Pillow-based rendering of entities (type, HP, stun, ⚠/⚔), objective/source directions, boolean indicators; FiraCode Nerd Font with fallback
- [x] Implement ActionRenderer: load and animate GIF sprites (walking.gif cycles at 250ms), render 2×2 grid with directional probabilities, SWORD/BEAMS combination, probability threshold (<0.1% hidden), greyed-out state for zero-probability action types
- [x] Implement AttentionOverlay: port JET colormap and attention compositing from game_view.py, support combined (.max) and per-head selection, bilinear upsample, alpha blend at ~55%
- [x] Implement main recording loop: sequential episode recording, --only-wins filtering with success detection, --count episodes, progress output, --max-attempts safety limit
- [x] Add record.py to pylint targets in .github/copilot-instructions.md, ensure record.py passes pylint
- [ ] End-to-end test: run record.py against a real model and verify the output video plays correctly and contains all three panels

## Notes

- Only one NES emulator per process — videos must be sequential.
- The environment must be created with `translation=False` to access raw state_change objects.
- Probabilities must be post-masking and renormalized (matching action_table.py's PROBABILITY column).
- GIF animation: walking.gif has 2 frames at 250ms; at 60fps switch every ~15 frames, at 30fps every ~8.
- For --only-wins: write to temp file, rename on success, delete on failure (avoids memory issues).
- FiraCode Nerd Font: try common system paths, fall back to PIL default with warning.
- The full spec is at docs/specs/recording.md — read it for complete details.
