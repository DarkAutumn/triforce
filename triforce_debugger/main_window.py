"""Main window layout for the Triforce Debugger."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QSplitter,
    QLabel,
    QTabWidget,
    QMenuBar,
    QFileDialog,
)

from triforce_debugger.action_table import ActionTable
from triforce_debugger.game_timer import GameTimer
from triforce_debugger.game_view import GameView
from triforce_debugger.model_browser import ModelBrowser
from triforce_debugger.observation_panel import ObservationPanel
from triforce_debugger.rewards_tab import RewardsTab
from triforce_debugger.scenario_selector import ScenarioSelector
from triforce_debugger.state_tab import StateTab
from triforce_debugger.step_history import StepHistoryWidget, StepEntry


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
        self.step_history = None
        self.detail_tabs = None
        self.rewards_tab = None
        self.state_tab = None
        self.evaluation_tab_placeholder = None
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

        self._build_menus()
        self._build_layout()
        self._wire_run_menu()

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
        self.action_overlay_coordinates = self.view_menu.addAction("Overlay: Coordinates")
        self.action_overlay_coordinates.setCheckable(True)
        self.view_menu.addSeparator()
        self.action_uncap_fps = self.view_menu.addAction("Uncap FPS")
        self.action_uncap_fps.setCheckable(True)

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
        self.evaluation_tab_placeholder = _placeholder("Evaluation")
        self.detail_tabs.addTab(self.rewards_tab, "Rewards")
        self.detail_tabs.addTab(self.state_tab, "State")
        self.detail_tabs.addTab(self.evaluation_tab_placeholder, "Evaluation")

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
        """Build the right panel: model browser, scenario selector, action probs."""
        panel = QWidget()
        panel.setObjectName("right_panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self.model_browser = ModelBrowser()
        self.scenario_selector = ScenarioSelector()
        self.action_table = ActionTable()

        layout.addWidget(self.model_browser, stretch=3)
        layout.addWidget(self.scenario_selector, stretch=0)
        layout.addWidget(self.action_table, stretch=2)

        return panel

    # ── Run menu wiring ──────────────────────────────────────

    def _wire_run_menu(self):
        """Connect Run menu actions and View > Uncap FPS to the game timer."""
        self.action_continue.triggered.connect(self._on_continue)
        self.action_pause.triggered.connect(self.game_timer.pause)
        self.action_step.triggered.connect(self.game_timer.single_step)
        self.action_uncap_fps.toggled.connect(self.game_timer.set_uncapped)

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
            self.step_viewed.emit(None)
        self.game_timer.resume()

    def showEvent(self, event):  # pylint: disable=invalid-name
        """Apply the initial splitter split once the window has its real geometry."""
        super().showEvent(event)
        if not self._initial_split_set:
            self._initial_split_set = True
            total = self.main_splitter.height()
            if total > 0:
                self.main_splitter.setSizes([total * 45 // 100, total * 55 // 100])

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

    # ── Arrow / A+arrow keyboard handling ─────────────────────

    # Map Qt arrow keys to direction strings
    _ARROW_TO_DIR = {
        Qt.Key.Key_Up: 'N',
        Qt.Key.Key_Down: 'S',
        Qt.Key.Key_Left: 'W',
        Qt.Key.Key_Right: 'E',
    }

    def keyPressEvent(self, event: QKeyEvent):  # pylint: disable=invalid-name
        """Handle arrow keys (move) and A+arrow (attack)."""
        key = event.key()

        if key == Qt.Key.Key_A:
            self._a_key_held = True
            return

        direction = self._ARROW_TO_DIR.get(key)
        if direction is not None and not event.isAutoRepeat():
            if self._a_key_held:
                self.manual_attack_requested.emit(direction)
            else:
                self.manual_move_requested.emit(direction)
            return

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):  # pylint: disable=invalid-name
        """Track release of 'A' key modifier."""
        if event.key() == Qt.Key.Key_A and not event.isAutoRepeat():
            self._a_key_held = False
            return

        super().keyReleaseEvent(event)


def _placeholder(label_text: str) -> QLabel:
    """Create a placeholder label widget for panels not yet implemented."""
    label = QLabel(label_text)
    label.setObjectName(label_text.lower().replace(" ", "_"))
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setStyleSheet("border: 1px dashed grey; color: grey;")
    label.setMinimumSize(50, 50)
    return label
