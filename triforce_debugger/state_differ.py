"""State diff engine for comparing ZeldaGame state dicts.

Compares two state dicts (as produced by ``extract_state_dict``) and returns
the set of field paths that changed between them.  Used by ``StateTab`` to
highlight changed values in blue and flash newly changed values yellow/green.
"""

from collections import OrderedDict
from typing import Any, Set


def diff_state_dicts(old: OrderedDict | None, new: OrderedDict | None) -> Set[str]:
    """Compare two state dicts and return the set of changed leaf paths.

    Returns paths like ``"link.health"``, ``"enemies[0].position.x"``, etc.
    If *old* is ``None`` (first frame), returns an empty set (nothing changed).
    """
    if old is None or new is None:
        return set()

    changed: set = set()
    _diff_recursive(old, new, "", changed)
    return changed


def _diff_recursive(old: Any, new: Any, prefix: str, changed: set) -> None:
    """Recursively walk two nested structures and collect changed leaf paths."""
    if isinstance(new, OrderedDict) and isinstance(old, OrderedDict):
        all_keys = set(old.keys()) | set(new.keys())
        for key in all_keys:
            path = f"{prefix}.{key}" if prefix else key
            old_val = old.get(key)
            new_val = new.get(key)
            if old_val is None or new_val is None:
                # Key added or removed — mark as changed
                _mark_all_leaves(new_val, path, changed)
            else:
                _diff_recursive(old_val, new_val, path, changed)

    elif isinstance(new, list) and isinstance(old, list):
        max_len = max(len(old), len(new))
        for i in range(max_len):
            path = f"{prefix}[{i}]"
            if i >= len(old):
                # New item added
                _mark_all_leaves(new[i], path, changed)
            elif i >= len(new):
                # Item removed — mark old path as changed
                _mark_all_leaves(old[i], path, changed)
            else:
                _diff_recursive(old[i], new[i], path, changed)

    else:
        # Leaf comparison
        if old != new:
            changed.add(prefix)


def _mark_all_leaves(value: Any, prefix: str, changed: set) -> None:
    """Mark all leaf paths under *value* as changed."""
    if value is None:
        changed.add(prefix)
    elif isinstance(value, OrderedDict):
        for key, val in value.items():
            path = f"{prefix}.{key}" if prefix else key
            _mark_all_leaves(val, path, changed)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _mark_all_leaves(item, f"{prefix}[{i}]", changed)
    else:
        changed.add(prefix)
