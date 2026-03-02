"""Model browser tree view for discovering and selecting .pt model files."""

import os
import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import QTreeView, QVBoxLayout, QWidget


# Role for storing the full file path on leaf items
PATH_ROLE = Qt.ItemDataRole.UserRole + 1

# Role for storing the step count (int or None) on leaf items
STEPS_ROLE = Qt.ItemDataRole.UserRole + 2


def parse_step_count(filename: str) -> int | None:
    """Extract the step count from a checkpoint filename.

    Convention: ``model-name-scenario_STEPCOUNT.pt`` — the integer after the
    last underscore before ``.pt``.  Returns *None* when no count can be parsed.
    """
    base = os.path.splitext(filename)[0]
    match = re.search(r'_(\d+)$', base)
    if match:
        return int(match.group(1))
    return None


def format_step_count(steps: int) -> str:
    """Format a step count with thousands separators (e.g. ``1,501,764 steps``)."""
    return f"{steps:,} steps"


class ModelBrowser(QWidget):
    """QTreeView that recursively displays ``.pt`` model files under a root
    directory.  Folders are expandable nodes; ``.pt`` files are leaves showing
    the parsed step count.  Double-clicking a leaf emits *model_selected* with
    the absolute path to the file.
    """

    # Emitted when the user double-clicks a .pt leaf
    model_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("model_browser")

        self._tree_view = QTreeView()
        self._tree_view.setObjectName("model_browser_tree")
        self._tree_view.setHeaderHidden(True)
        self._tree_view.setEditTriggers(QTreeView.EditTrigger.NoEditTriggers)

        self._model = QStandardItemModel()
        self._tree_view.setModel(self._model)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tree_view)

        self._root_path: str | None = None
        self._loaded_model_path: str | None = None

        self._tree_view.doubleClicked.connect(self._on_double_click)  # pylint: disable=no-member

    # ── Public API ────────────────────────────────────────────

    @property
    def tree_view(self) -> QTreeView:
        """The underlying QTreeView widget."""
        return self._tree_view

    @property
    def item_model(self) -> QStandardItemModel:
        """The underlying QStandardItemModel."""
        return self._model

    @property
    def root_path(self) -> str | None:
        """The currently scanned root directory."""
        return self._root_path

    def scan_directory(self, path: str):
        """Recursively scan *path* for ``.pt`` files and rebuild the tree."""
        self._root_path = path
        self._model.clear()

        if not path or not os.path.isdir(path):
            return

        root_item = self._model.invisibleRootItem()
        self._populate(root_item, path)

    def set_loaded_model(self, model_path: str | None):
        """Bold the item whose path matches *model_path*; un-bold all others."""
        prev = self._loaded_model_path
        self._loaded_model_path = model_path
        if prev != model_path:
            self._apply_bold(self._model.invisibleRootItem())

    @property
    def loaded_model_path(self) -> str | None:
        """Path of the currently loaded (bolded) model."""
        return self._loaded_model_path

    # ── Tree building ─────────────────────────────────────────

    def _populate(self, parent_item: QStandardItem, dir_path: str):
        """Recursively add directory and .pt file entries under *parent_item*."""
        try:
            entries = sorted(os.listdir(dir_path))
        except OSError:
            return

        # Separate directories and .pt files
        dirs = []
        pt_files = []
        for entry in entries:
            full = os.path.join(dir_path, entry)
            if os.path.isdir(full) and not entry.startswith('.') and entry != '__pycache__':
                dirs.append((entry, full))
            elif entry.endswith('.pt') and os.path.isfile(full):
                pt_files.append((entry, full))

        for name, full in dirs:
            # Only add directory node if it (recursively) contains .pt files
            folder_item = QStandardItem(name)
            folder_item.setEditable(False)
            self._populate(folder_item, full)
            if folder_item.rowCount() > 0:
                parent_item.appendRow(folder_item)

        for name, full in pt_files:
            steps = parse_step_count(name)
            if steps is not None:
                display = format_step_count(steps)
            else:
                display = name

            item = QStandardItem(display)
            item.setEditable(False)
            item.setData(full, PATH_ROLE)
            item.setData(steps, STEPS_ROLE)
            item.setToolTip(full)
            parent_item.appendRow(item)

    # ── Bold highlighting ─────────────────────────────────────

    def _apply_bold(self, parent: QStandardItem):
        """Walk the tree and bold/un-bold items based on *_loaded_model_path*."""
        for row in range(parent.rowCount()):
            item = parent.child(row)
            if item is None:
                continue

            path = item.data(PATH_ROLE)
            font = item.font()
            font.setBold(path is not None and path == self._loaded_model_path)
            item.setFont(font)

            # Recurse into folder nodes
            if item.rowCount() > 0:
                self._apply_bold(item)

    # ── Interaction ───────────────────────────────────────────

    def _on_double_click(self, index):
        """Handle double-click: emit model_selected for .pt leaves."""
        item = self._model.itemFromIndex(index)
        if item is None:
            return

        path = item.data(PATH_ROLE)
        if path is not None:
            self.model_selected.emit(path)
