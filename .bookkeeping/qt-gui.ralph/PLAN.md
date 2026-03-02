# Triforce Debugger — Implementation Plan

## Goal
Replace the pygame-based `zui/` GUI with a modern PySide6 Qt debugger application. The new GUI
provides time-travel step inspection, model browsing, evaluation management, and full observation
visibility. See `.bookkeeping/qt-gui.ralph/specs/SPEC.md` for the full specification.

## Status
STATUS: IN_PROGRESS

## Run Naming Convention
N/A — this is a GUI rewrite, no training runs.

---

## Phase 1: Foundation

- [x] **QT-01**: Project scaffolding. Create `triforce_debugger/` package with `__init__.py`. Create `debug.py` entry point that launches a QApplication + empty QMainWindow. Add `PySide6>=6.5` to requirements.txt. Install it. Verify: `python debug.py` opens a blank window and closes cleanly. Test: headless QApplication + QMainWindow instantiation with `QT_QPA_PLATFORM=offscreen`.

- [x] **QT-02**: Extract environment bridge. Copy `zui/environment_wrapper.py` logic into `triforce_debugger/environment_bridge.py`. Remove all pygame dependencies. Keep: `StepResult`, `EnvironmentWrapper` (step/restart/model loading). This is the Qt-independent core that the new GUI and tests both use. Test: instantiation and basic method signatures (mock the env if retro is unavailable headless).

- [x] **QT-03**: Main window layout. Implement `main_window.py` with the fixed layout from the spec: top section (obs panel placeholder, game view placeholder, right panel placeholder) and bottom section (step history placeholder, tabbed detail panel placeholder). Use QSplitter for the top/bottom split with bottom ≥50% height. Add the menu bar skeleton (File, View, Run menus with placeholder actions). Test: headless — window has correct widget hierarchy, splitter exists, menus exist.

- [x] **QT-04**: Game view widget. Implement `game_view.py` — a QWidget that receives an RGB numpy array (NES frame) and renders it scaled to fit the widget. Use QPainter + QImage for rendering. No overlays yet, just the base frame display. Test: feed a known numpy array, verify QImage is created with correct dimensions.

- [x] **QT-05**: Game timer and step loop. Implement `game_timer.py` — QTimer-based game loop. Capped mode (16ms/60fps) and uncapped mode (0ms). Wire it to `environment_bridge`: each tick calls `step()`, updates the game view. Add pause/resume/single-step methods. Wire to Run menu: Continue (F5), Pause (Shift+F5), Step (F10), Restart (Ctrl+Shift+F5). Test: timer fires, mode switching works.

- [ ] **QT-06**: Global keyboard shortcuts. Register application-level shortcuts (Qt.ApplicationShortcut) for F5, Shift+F5, F10, Ctrl+Shift+F5, arrow keys, A+arrow. Verify they fire regardless of which widget has focus. Test: headless shortcut signal emission.

## Phase 2: Model & Scenario Selection

- [ ] **QT-07**: Scenario selector. Implement `scenario_selector.py` — QComboBox populated from `triforce.json` scenarios. Changing scenario resets the environment (unless already on that scenario). Wire to the right panel. Test: combobox populated with correct scenario names from triforce.json.

- [ ] **QT-08**: Model browser tree view. Implement `model_browser.py` — QTreeView with QStandardItemModel. Recursively scans a directory for `.pt` files. Folders are expandable nodes, `.pt` files are leaves showing parsed step count (e.g., "1,501,764 steps"). Double-click loads the model. Currently loaded model is bolded. Test: build tree from a mock directory structure, verify node hierarchy and step count parsing.

- [ ] **QT-09**: File → Open Directory. Implement the File menu: Open Directory (Ctrl+O) opens QFileDialog, rescans and updates the model browser tree. File → Exit (Ctrl+Q) quits. Wire `debug.py` to accept `--path` argument for initial directory. Test: menu actions exist and are connected.

## Phase 3: Observation & Probabilities

- [ ] **QT-10**: Observation panel. Implement `observation_panel.py` — displays the network input image (84×84 grayscale, scaled up), vector observations (enemy/projectile/item arrows as compact directional indicators), boolean indicators (Enemies, Beams, Low HP, Full HP — green/grey), and directional circles (Objective, Source). Port the rendering logic from `zui/reward_debugger.py`'s `_show_observation()` to Qt (QPainter). Test: headless widget creation, correct sub-widget count.

- [ ] **QT-11**: Action probability table. Implement `action_table.py` — QTableWidget with columns ACTION, DIRECTION, PROBABILITY. Percentages as "45.2%", masked actions as "[masked]" in grey. Also display the Value estimate. Updated each step from `model_selector.get_probabilities()`. Port from `zui/model_selector.py`. Test: populate table with mock probability data, verify cell content formatting.

## Phase 4: Step History & Time-Travel

- [ ] **QT-12**: Step history ring buffer. Implement `step_history.py` — `StepHistory` class backed by `collections.deque(maxlen=50_000)`. Each entry: step_number, action, reward, observation, state, action_mask, action_probabilities, terminated, truncated, game frame. Methods: append, clear, get_by_index, length. Test: add/clear/retrieve, capacity limits, oldest items evicted.

- [ ] **QT-13**: Step history list widget. Add QAbstractListModel wrapping StepHistory. Virtual scrolling — only visible rows rendered. Each row shows: step number, action (e.g., "MOVE N"), total reward (colored green/red/grey). Expandable rows showing individual reward/penalty breakdown. Most recent at top. Auto-scrolls to latest unless user has scrolled up (sticky scroll). Test: model returns correct data for row indices, row count matches buffer.

- [ ] **QT-14**: Time-travel on step selection. Clicking a step in the history list updates ALL panels to show that step's data: game view (frame), observation panel, rewards tab, state tab, action probabilities. The game loop pauses when viewing a historical step. Press F5 to resume live (jumps back to current step). Test: selecting a step emits correct signals, panels receive the right data.

## Phase 5: Detail Tabs

- [ ] **QT-15**: Rewards tab. Implement `rewards_tab.py` — QTableWidget showing running rewards (NAME, COUNT, TOTAL VALUE columns), episode total, endings dict. When viewing a historical step, shows that step's individual reward breakdown. Cleared on reset. Test: populate with mock StepRewards, verify aggregation math.

- [ ] **QT-16**: State tab. Implement `state_tab.py` — QTreeWidget displaying the full ZeldaGame state as an expandable tree. Nodes for: link, enemies[], items[], projectiles[], room, objectives. Each leaf shows field name + value. Test: build tree from a mock state dict, verify node structure.

- [ ] **QT-17**: State diff engine. Implement `state_differ.py` — compares two ZeldaGame states, returns set of changed field paths. Wire to state_tab: changed values colored **blue** (persistent until next change), flash **yellow/green briefly (~300ms)** on the moment of change using QTimer. When viewing historical steps, diff against the previous step. Test: diff two mock states, verify changed paths detected.

- [ ] **QT-18**: Evaluation tab. Implement `evaluation_tab.py` — if `.eval.json` exists for the selected model, show a data grid with: episodes, success rate, percentiles (p25/p50/p75/p90/max), milestone histogram. If no eval exists, show "Run Evaluation" button with episode count spinner. Clicking runs `evaluate.py` in a QProcess (non-blocking), shows progress, auto-populates on completion. Test: parse a mock `.eval.json`, verify grid content.

## Phase 6: Overlays & View Menu

- [ ] **QT-19**: Game view overlays. Add overlay rendering to `game_view.py`: wavefront distance, tile IDs, walkability, tile coordinates. Each is a toggle checkbox in the View menu. Multiple overlays can be active simultaneously. Port overlay logic from `zui/reward_debugger.py`'s `_overlay_grid_and_text()`. Test: overlay toggle state management.

- [ ] **QT-20**: View menu — Uncap FPS. Add "Uncap FPS" toggle to View menu. When checked, game timer switches to 0ms interval. When unchecked, back to 16ms. Test: timer interval changes on toggle.

## Phase 7: Integration & Polish

- [ ] **QT-21**: End-to-end integration. Wire everything together: launch `debug.py`, scan for models, select a model + scenario, press F5, watch the game run with all panels updating live. Step through with F10, click history for time-travel, switch models mid-run. Fix any wiring issues. No automated test — manual verification.

- [ ] **QT-22**: Manual input (arrows + attack). Wire arrow key presses to override the model's action for one step (manual move). A+arrow for attack. Should work identically to the old pygame controls. Test: verify correct ActionTaken produced for each key combo.

- [ ] **QT-23**: Pylint cleanup. Run `pylint triforce_debugger/ debug.py` and fix all issues. Ensure the full test suite passes. Final commit.

- [ ] **QT-24**: Delete old GUI. Remove `run.py` and `zui/` directory. Update any imports or references. Verify nothing depends on the old code (grep for `from zui` and `import zui`). Test suite still passes.

---

## Parking Lot

_Ideas to revisit after the core GUI is done:_

- Video recording / replay saving
- Tensorboard embedding or inline training curves
- Multi-instance comparison (two models side by side)
- Network activation visualization (what neurons fire)
- Reward shaping playground (adjust reward weights live)

---

## Notes

_Updated by agents as they learn things._

- The `zui-obs-indicators` branch is the starting point (branched from main).
  - Actual branch name is `new-ui` (renamed from `zui-obs-indicators`).
- PySide6 must be installed in the .venv before anything else.
- PySide6 added to `.pylintrc` `extension-pkg-whitelist` to avoid false `no-name-in-module` errors.
- The env requires `retro` (gym-retro) which may need display — use mocks for headless tests.
- `environment_bridge.py` should be the ONLY file that imports from `triforce` env/scenario code.
  All other debugger modules interact through the bridge.
