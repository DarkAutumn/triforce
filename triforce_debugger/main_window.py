"""Main window layout for the Triforce Debugger."""

import logging
import os
import time
import traceback

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QSplitter,
    QTabWidget,
    QToolBar,
    QMenuBar,
    QFileDialog,
)

from triforce.zelda_enums import ActionKind, Direction
from triforce_debugger.action_table import ActionTable
from triforce_debugger.environment_bridge import EnvironmentBridge
from triforce_debugger.game_timer import GameTimer
from triforce_debugger.game_view import GameView, OverlayFlags
from triforce_debugger.model_browser import ModelBrowser
from triforce_debugger.observation_panel import ObservationPanel
from triforce_debugger.evaluation_tab import EvaluationTab
from triforce_debugger.rewards_tab import RewardsTab
from triforce_debugger.scenario_selector import ScenarioSelector
from triforce_debugger.state_tab import StateTab, extract_state_dict
from triforce_debugger.step_history import StepHistoryWidget, StepEntry

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main debugger window with the fixed panel layout."""

    # Manual input signals: direction is 'N', 'S', 'E', or 'W'
    manual_move_requested = Signal(str)
    manual_attack_requested = Signal(str)

    # Emitted when a step is selected for viewing (StepEntry), or None for live mode
    step_viewed = Signal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Triforce Debugger")
        self.resize(1280, 900)

        # Panels set in _build_layout / _build_right_panel
        self.obs_panel = None
        self.game_view = None
        self.right_panel = None
        self.right_tabs = None
        self.step_history = None
        self.detail_tabs = None
        self.rewards_tab = None
        self.state_tab = None
        self.evaluation_tab = None
        self.model_browser = None
        self.scenario_selector = None
        self.action_table = None
        self.main_splitter = None

        # Track 'A' key for attack modifier
        self._a_key_held = False

        # Time-travel state
        self._viewing_historical = False

        # Deferred splitter sizing (applied once in showEvent)
        self._initial_split_set = False

        # Game loop timer
        self.game_timer = GameTimer(self)

        # Environment bridge and step tracking
        self._bridge: EnvironmentBridge | None = None
        self._step_count = 0
        self._frame_stack = 3
        self._model_dir: str | None = None
        self._selected_model_path: str | None = None
        self._last_state_dict = None
        self._last_zelda_state = None

        # FPS counter
        self._fps_frame_count = 0
        self._fps_last_time = time.monotonic()
        self._fps_value = 0.0
        self._fps_timer = QTimer(self)
        self._fps_timer.timeout.connect(self._update_fps_display)  # pylint: disable=no-member
        self._fps_timer.start(500)  # update display twice per second

        self._build_menus()
        self._build_layout()
        self._wire_run_menu()
        self._wire_integration()

        # Install global event filter so arrow/A keys work regardless of focus
        from PySide6.QtWidgets import QApplication  # pylint: disable=import-outside-toplevel
        QApplication.instance().installEventFilter(self)

    # ── Menu Bar ──────────────────────────────────────────────

    def _build_menus(self):
        menu_bar: QMenuBar = self.menuBar()

        # File menu
        self.file_menu = menu_bar.addMenu("&File")
        self.action_open_dir = self.file_menu.addAction("Open Directory...")
        self.action_open_dir.setShortcut("Ctrl+O")
        self.file_menu.addSeparator()
        self.action_open_dir.triggered.connect(self.open_directory)
        self.file_menu.addSeparator()
        self.action_exit = self.file_menu.addAction("Exit")
        self.action_exit.setShortcut("Ctrl+Q")
        self.action_exit.triggered.connect(self.close)

        # View menu
        self.view_menu = menu_bar.addMenu("&View")
        self.action_overlay_wavefront = self.view_menu.addAction("Overlay: Wavefront")
        self.action_overlay_wavefront.setCheckable(True)
        self.action_overlay_tile_ids = self.view_menu.addAction("Overlay: Tile IDs")
        self.action_overlay_tile_ids.setCheckable(True)
        self.action_overlay_walkability = self.view_menu.addAction("Overlay: Walkability")
        self.action_overlay_walkability.setCheckable(True)
        self.view_menu.addSeparator()
        self.action_uncap_fps = self.view_menu.addAction("Uncap FPS")
        self.action_uncap_fps.setCheckable(True)
        self.action_uncap_fps.setChecked(True)  # default to uncapped

        # Run menu — all shortcuts are ApplicationShortcut so they fire
        # regardless of which widget has focus.
        app_ctx = Qt.ShortcutContext.ApplicationShortcut

        self.run_menu = menu_bar.addMenu("&Run")
        self.action_continue = self.run_menu.addAction("Continue")
        self.action_continue.setShortcut("F5")
        self.action_continue.setShortcutContext(app_ctx)
        self.action_pause = self.run_menu.addAction("Pause")
        self.action_pause.setShortcut("Shift+F5")
        self.action_pause.setShortcutContext(app_ctx)
        self.action_step = self.run_menu.addAction("Step")
        self.action_step.setShortcut("F10")
        self.action_step.setShortcutContext(app_ctx)
        self.action_restart = self.run_menu.addAction("Restart")
        self.action_restart.setShortcut("Ctrl+Shift+F5")
        self.action_restart.setShortcutContext(app_ctx)

    # ── Layout ────────────────────────────────────────────────

    def _build_layout(self):
        # Toolbar with scenario selector
        self.scenario_selector = ScenarioSelector()
        toolbar = QToolBar("Main")
        toolbar.setObjectName("main_toolbar")
        toolbar.setMovable(False)
        toolbar.addWidget(self.scenario_selector)
        self.addToolBar(toolbar)

        # Top section: obs panel | game view | right panel
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)

        self.obs_panel = ObservationPanel()
        self.game_view = GameView()
        self.right_panel = self._build_right_panel()

        top_layout.addWidget(self.obs_panel, stretch=1)
        top_layout.addWidget(self.game_view, stretch=3)
        top_layout.addWidget(self.right_panel, stretch=2)

        # Bottom section: step history | tabbed detail panel
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        self.step_history = StepHistoryWidget()
        self.detail_tabs = QTabWidget()
        self.detail_tabs.setObjectName("detail_tabs")
        self.rewards_tab = RewardsTab()
        self.state_tab = StateTab()
        self.evaluation_tab = EvaluationTab()
        self.detail_tabs.addTab(self.rewards_tab, "Rewards")
        self.detail_tabs.addTab(self.state_tab, "State")
        self.detail_tabs.addTab(self.evaluation_tab, "Evaluation")

        bottom_layout.addWidget(self.step_history, stretch=1)
        bottom_layout.addWidget(self.detail_tabs, stretch=2)

        # Wire time-travel: step selection → update panels
        self.step_history.step_selected.connect(self._on_step_selected)

        # Vertical splitter: top | bottom (bottom ≥50%)
        self.main_splitter = QSplitter(Qt.Orientation.Vertical)
        self.main_splitter.setObjectName("main_splitter")
        self.main_splitter.addWidget(top_widget)
        self.main_splitter.addWidget(bottom_widget)

        # Set initial sizes so bottom gets ≥50% of window height (deferred to showEvent)
        self.setCentralWidget(self.main_splitter)

    def _build_right_panel(self) -> QWidget:
        """Build the right panel: tabbed Models / Probabilities."""
        panel = QWidget()
        panel.setObjectName("right_panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self.right_tabs = QTabWidget()
        self.right_tabs.setObjectName("right_tabs")

        # ── Models tab ────────────────────────────────────────
        self.model_browser = ModelBrowser()

        # ── Probabilities tab ─────────────────────────────────
        self.action_table = ActionTable()

        self.right_tabs.addTab(self.model_browser, "Models")
        self.right_tabs.addTab(self.action_table, "Probabilities")

        layout.addWidget(self.right_tabs)

        return panel

    # ── Run menu wiring ──────────────────────────────────────

    def _wire_run_menu(self):
        """Connect Run menu actions, View > Uncap FPS, and overlay toggles."""
        self.action_continue.triggered.connect(self._on_continue)
        self.action_pause.triggered.connect(self.game_timer.pause)
        self.action_step.triggered.connect(self.game_timer.single_step)
        self.action_restart.triggered.connect(self._on_restart)
        self.action_uncap_fps.toggled.connect(self.game_timer.set_uncapped)

        # Overlay toggles
        self.action_overlay_wavefront.toggled.connect(
            lambda on: self.game_view.set_overlay(OverlayFlags.WAVEFRONT, on))
        self.action_overlay_tile_ids.toggled.connect(
            lambda on: self.game_view.set_overlay(OverlayFlags.TILE_IDS, on))
        self.action_overlay_walkability.toggled.connect(
            lambda on: self.game_view.set_overlay(OverlayFlags.WALKABILITY, on))

    def _wire_integration(self):
        """Connect all integration signals for end-to-end operation."""
        self.game_timer.step_requested.connect(self._do_env_step)
        self.game_timer.step_completed.connect(self._on_step_completed)
        self.game_timer.frame_ready.connect(self._on_frame_ready)
        self.model_browser.model_selected.connect(self._on_model_file_selected)
        self.scenario_selector.scenario_changed.connect(self._on_scenario_changed)
        self.manual_move_requested.connect(self._on_manual_move)
        self.manual_attack_requested.connect(self._on_manual_attack)

    # ── Time-travel ──────────────────────────────────────────

    @property
    def is_viewing_historical(self) -> bool:
        """True when showing a historical step rather than live data."""
        return self._viewing_historical

    def show_step(self, entry: StepEntry):
        """Update game view, observation panel, and action table from a step entry."""
        self._update_panels(entry)
        self.step_viewed.emit(entry)

    def _update_panels(self, entry: StepEntry):
        """Update visual panels without emitting step_viewed."""
        if entry.frame is not None:
            self.game_view.set_frame(entry.frame)
        if entry.observation is not None:
            self.obs_panel.update_observation(entry.observation)
        self.action_table.update_probabilities(
            entry.action_probabilities,
            entry.action_mask_desc
        )
        if self._viewing_historical and entry.reward is not None:
            self.rewards_tab.show_step_rewards(entry.reward, entry.step_number)
        # Update state tab during time-travel
        if self._viewing_historical and entry.state is not None:
            prev_dict = self._get_prev_state_dict(entry)
            self.state_tab.show_step_state(entry.state, entry.step_number, prev_dict)

    def _get_prev_state_dict(self, entry: StepEntry):
        """Find the state dict of the step before *entry* for diff highlighting."""
        history = self.step_history.history
        if len(history) < 2:
            return None
        for i, step in enumerate(history):
            if step.step_number == entry.step_number and i > 0:
                return history[i - 1].state
        return None

    def _on_step_selected(self, buf_index):
        """Handle step selection in history list (time-travel)."""
        try:
            entry = self.step_history.history[buf_index]
        except IndexError:
            return
        self._viewing_historical = True
        self.game_timer.pause()
        self.show_step(entry)

    def _on_continue(self):
        """Resume live play, exiting historical mode if active."""
        if self._viewing_historical:
            self._viewing_historical = False
            latest = self.step_history.history.newest
            if latest is not None:
                self._update_panels(latest)
            self.rewards_tab.show_running()
            self.state_tab.show_live()
            self.step_viewed.emit(None)
        self.game_timer.resume()

    # ── Environment integration ───────────────────────────────

    def _create_bridge(self):
        """Create the environment bridge from current selections."""
        scenario_def = self.scenario_selector.current_scenario
        if scenario_def is None or self._model_dir is None:
            return

        # Tear down existing bridge
        self._destroy_bridge()

        try:
            self._bridge = EnvironmentBridge(
                self._model_dir, scenario_def, self._frame_stack)

            # Select the specific .pt file if one was chosen
            if self._selected_model_path:
                self._bridge.selector.select_by_path(self._selected_model_path)

            self.model_browser.set_loaded_model(self._bridge.selector.model_path)
            self._do_restart()
            self.setWindowTitle(f"Triforce Debugger — {self._bridge.model_details}")
        except Exception:  # pylint: disable=broad-except
            log.error("Failed to create bridge:\n%s", traceback.format_exc())
            self._bridge = None

    def _destroy_bridge(self):
        """Tear down the current bridge if it exists."""
        self.game_timer.stop()
        if self._bridge:
            self._bridge.close()
            self._bridge = None

    def _do_restart(self):
        """Restart the episode and update all panels with initial state."""
        if not self._bridge:
            return

        self._step_count = 0
        self.step_history.clear_history()
        self.rewards_tab.clear()
        self.state_tab.clear()
        self._last_state_dict = None
        self._last_zelda_state = None

        step_result = self._bridge.restart()
        self._show_step_result(step_result, initial=True)

    def _on_restart(self):
        """Handle restart action (Ctrl+Shift+F5)."""
        if self._viewing_historical:
            self._viewing_historical = False
            self.rewards_tab.show_running()
            self.state_tab.show_live()
        self.game_timer.pause()
        self._do_restart()

    def _on_frame_ready(self, frame):
        """Render a single NES frame (called once per ~16ms when capped)."""
        self._fps_frame_count += 1
        if frame is not None:
            self.game_view.set_frame(frame)

    def _on_step_completed(self, step_result):
        """Handle a completed step result — update panels."""
        self._show_step_result(step_result)
        if step_result.completed:
            self._do_restart()

    def _do_env_step(self):
        """Perform one environment step and enqueue the frames."""
        if not self._bridge or self._viewing_historical:
            return

        try:
            step_result = self._bridge.step()
            frames = step_result.frames if step_result.frames else []
            self.game_timer.enqueue_step(frames, step_result)
        except Exception:  # pylint: disable=broad-except
            log.error("Step error:\n%s", traceback.format_exc())
            self.game_timer.pause()

    def _update_fps_display(self):
        """Update the FPS counter in the status bar."""
        now = time.monotonic()
        elapsed = now - self._fps_last_time
        if elapsed > 0:
            self._fps_value = self._fps_frame_count / elapsed
        self._fps_frame_count = 0
        self._fps_last_time = now

        mode = "uncapped" if self.game_timer.is_uncapped else "60fps cap"
        state = "running" if self.game_timer.is_running else "paused"
        buf = self.game_timer.buffer_depth
        self.statusBar().showMessage(
            f"FPS: {self._fps_value:.1f}  |  {mode}  |  {state}  |  buf: {buf}"
        )

    def _show_step_result(self, step_result, initial=False):
        """Update all panels from a StepResult (called at step boundaries)."""
        self._step_count += 1

        # Get last frame for step history snapshot
        frame = step_result.frames[-1] if step_result.frames else None

        # Extract state dict for the state tab
        state = step_result.state
        self._last_zelda_state = state
        state_dict = extract_state_dict(state) if state else None

        # Get probabilities from the model
        probs = None
        try:
            probs = self._bridge.get_probabilities()
        except Exception:  # pylint: disable=broad-except
            log.warning("Failed to get probabilities:\n%s", traceback.format_exc())

        # Update game view (frame + overlays)
        if frame is not None:
            self.game_view.set_frame(frame)
        if state:
            self.game_view.set_game_state(state)

        # Update observation panel
        if step_result.observation is not None:
            self.obs_panel.update_observation(step_result.observation)

        # Update action table
        self.action_table.update_probabilities(probs, step_result.action_mask_desc)

        # Update rewards tab (running totals)
        if not initial:
            self.rewards_tab.add_step_rewards(step_result.rewards)

        # Update state tab
        if state_dict:
            self.state_tab.update_state_dict(state_dict)
        self._last_state_dict = state_dict

        # Determine action for step history
        action = None
        if step_result.state_change and hasattr(step_result.state_change, 'action'):
            action = step_result.state_change.action

        # Append to step history
        entry = StepEntry(
            step_number=self._step_count,
            action=action,
            reward=step_result.rewards,
            observation=step_result.observation,
            state=state_dict,
            action_mask=step_result.action_mask,
            action_probabilities=probs,
            terminated=step_result.terminated,
            truncated=step_result.truncated,
            frame=frame,
            action_mask_desc=step_result.action_mask_desc,
        )
        self.step_history.append_step(entry)

        # Update window title with model info
        self.setWindowTitle(f"Triforce Debugger — {self._bridge.model_details}")

    def _on_model_file_selected(self, pt_path: str):
        """Handle model file selection from the browser."""
        self._selected_model_path = pt_path
        model_dir = os.path.dirname(pt_path)

        # If bridge exists with same config, just switch the model
        if self._bridge and model_dir == self._model_dir:
            if self._bridge.selector.select_by_path(pt_path):
                self.model_browser.set_loaded_model(pt_path)
                self.evaluation_tab.set_model(
                    pt_path, None,
                    self.scenario_selector.current_scenario_name)
                self.setWindowTitle(f"Triforce Debugger — {self._bridge.model_details}")
                self.right_tabs.setCurrentIndex(1)  # switch to Probabilities
                return

        # Otherwise, recreate the bridge
        self._model_dir = model_dir
        self._create_bridge()
        self.evaluation_tab.set_model(
            pt_path, None,
            self.scenario_selector.current_scenario_name)
        self.right_tabs.setCurrentIndex(1)  # switch to Probabilities

    def _on_scenario_changed(self, _name: str):
        """Handle scenario change — recreate the bridge if active."""
        if self._bridge:
            self._create_bridge()

    def showEvent(self, event):  # pylint: disable=invalid-name
        """Apply the initial splitter split once the window has its real geometry."""
        super().showEvent(event)
        if not self._initial_split_set:
            self._initial_split_set = True
            total = self.main_splitter.height()
            if total > 0:
                self.main_splitter.setSizes([total * 45 // 100, total * 55 // 100])

    def closeEvent(self, event):  # pylint: disable=invalid-name
        """Clean up the environment bridge on close."""
        self._destroy_bridge()
        super().closeEvent(event)

    # ── File menu actions ─────────────────────────────────────

    def set_model_path(self, path: str):
        """Set the initial model directory and scan for .pt files."""
        self.model_browser.scan_directory(path)

    def open_directory(self):
        """Open a file dialog to choose a model directory and rescan."""
        start_dir = self.model_browser.root_path or ""
        directory = QFileDialog.getExistingDirectory(
            self, "Open Model Directory", start_dir
        )
        if directory:
            self.model_browser.scan_directory(directory)

    # ── Global event filter for arrow / A+arrow keys ────────

    # Map Qt arrow keys to direction strings
    _ARROW_TO_DIR = {
        Qt.Key.Key_Up: 'N',
        Qt.Key.Key_Down: 'S',
        Qt.Key.Key_Left: 'W',
        Qt.Key.Key_Right: 'E',
    }

    def eventFilter(self, obj, event):  # pylint: disable=invalid-name
        """Intercept ALL key events application-wide for game controls."""
        from PySide6.QtCore import QEvent  # pylint: disable=import-outside-toplevel

        if event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key == Qt.Key.Key_A:
                self._a_key_held = True
                return True

            direction = self._ARROW_TO_DIR.get(key)
            if direction is not None and not event.isAutoRepeat():
                if self._a_key_held:
                    self.manual_attack_requested.emit(direction)
                else:
                    self.manual_move_requested.emit(direction)
                return True

        elif event.type() == QEvent.Type.KeyRelease:
            if event.key() == Qt.Key.Key_A and not event.isAutoRepeat():
                self._a_key_held = False
                return True

        return super().eventFilter(obj, event)

    # ── Manual input handlers ─────────────────────────────────

    _DIR_STR_TO_ENUM = {
        'N': Direction.N,
        'S': Direction.S,
        'E': Direction.E,
        'W': Direction.W,
    }

    def _on_manual_move(self, direction_str):
        """Handle manual move request (arrow key press)."""
        direction = self._DIR_STR_TO_ENUM.get(direction_str)
        if direction is not None:
            self._do_manual_step((ActionKind.MOVE, direction))

    def _on_manual_attack(self, direction_str):
        """Handle manual attack request (A+arrow key press)."""
        direction = self._DIR_STR_TO_ENUM.get(direction_str)
        if direction is None:
            return

        action_kind = ActionKind.SWORD
        if self._bridge and ActionKind.BEAMS in self._bridge.action_space.actions_allowed:  # pylint: disable=no-member
            if self._last_zelda_state and self._last_zelda_state.link.has_beams:
                action_kind = ActionKind.BEAMS

        self._do_manual_step((action_kind, direction))

    def _do_manual_step(self, action_tuple):
        """Execute a single environment step with a manual action."""
        if not self._bridge or self._viewing_historical:
            return

        if not self._bridge.is_valid_action(action_tuple):
            log.info("Invalid manual action: %s", action_tuple)
            return

        self.game_timer.pause()
        try:
            step_result = self._bridge.step(action=action_tuple)
            frames = step_result.frames if step_result.frames else []
            self.game_timer.play_frames(frames, step_result)
        except Exception:  # pylint: disable=broad-except
            log.error("Manual step error:\n%s", traceback.format_exc())
