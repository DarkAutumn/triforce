"""The declared inputs of a task and the checks that each is there (round 40, the owner: a run's input that resolves
to nothing is an error that names where it looked, never "none"; where none is a legitimate recipe, the task file
declares it, so absence and "none" can be told apart).

A task (``specialists/<name>.yaml`` or an experiment's yaml) declares:

- ``start_states``: globs of seed states; **every glob must match at least one file** (``check_start_states``: a
  glob that matched nothing was dropped and the family trained on the others).
- ``harvested_fraction`` > 0: harvested start states the resets draw; **the family's harvested dir must hold some
  of its namespaces** (``check_harvested``; the native pool refuses too, at build and when a file goes missing under
  a running pool).
- ``curriculum.enabled``: a track for **every seed state, except those ``curriculum.untracked`` declares** (globs of
  state names; ``check_curriculum``: a seed state without a track trained without one, seen only in ``run.json``).
- ``demos: {files, coeff}``: demonstration files the trainings clone; **every file must be in the demos data folder**
  (``task_demos``: ``paths.data_dir('demos')``, the checkout's ``target/demos`` while it has one, else the store's
  current snapshot).

The entry points call these before they wait for their turn (``triforce train``, ``triforce evaluate``, the lab's
trainer; ``task_demos`` the two trainers); a failure raises ``MissingInput`` and the tool exits 2 with its message.
Locations come from the config and ``triforce.paths`` only."""

import fnmatch
import json
import re
from pathlib import Path
from typing import Optional

from . import _native, paths


class MissingInput(Exception):
    """A declared input resolves to nothing (the message names where it looked)."""


def _stem(path: str) -> str:
    name = Path(path).name
    return name[:-len('.state')] if name.endswith('.state') else name


def _where(kind: str, folder) -> str:
    """``folder``'s missing-data note: when its parent (the kind's folder) is not there, the rule's two places
    (``paths.data_missing``: the caller's folder, the store's current snapshot)."""
    parent = Path(folder).parent
    return '' if parent.exists() else f"; {paths.data_missing(kind, parent)}"


def check_start_states(config, states_dir=None) -> dict:
    """Each ``start_states`` glob's matches (glob -> state names); raises ``MissingInput`` naming the globs that match
    no file under the states dir."""
    matched = {pattern: [_stem(p) for p in config.replace(start_states=[pattern]).resolve_start_states(states_dir)]
               for pattern in config.start_states}
    empty = [pattern for pattern, names in matched.items() if not names]
    if empty:
        where = states_dir or _native.paths()['states_dir']
        raise MissingInput(f"{config.name}: start_states {empty} match no state file under {where} ({_routes(config)});"
                           f" remove the glob from the task or build its states")
    return matched


def _routes(config) -> str:
    routes = config.to_dict().get('state_routes')
    return f"the states of routes {routes}" if routes else "every route's states"


def check_harvested(config, states_dir=None) -> list[str]:
    """The harvested start states the family draws (``harvested_fraction`` > 0; none drawn: []); raises
    ``MissingInput`` when it draws them and its harvested dir holds none of its namespaces."""
    if config.harvested_fraction <= 0:
        return []
    found = config.resolve_harvested_states(states_dir)
    if not found:
        folder = config.harvested_dir(states_dir)
        raise MissingInput(f"{config.name} draws {config.harvested_fraction:.0%} of its resets from harvested start "
                           f"states and {folder} holds none ({_routes(config)}){_where('harvested', folder)}")
    return found


def untracked(config) -> list[str]:
    """``curriculum.untracked``: the globs of the seed states the task declares to train without a track."""
    return list(config.to_dict()['curriculum'].get('untracked') or [])


def track_starts(trajectories) -> dict[str, str]:
    """Track name -> the seed state it was explored from (``<trajectories>/<track>/trajectory.json``
    ``start_state``), as the native curriculum loads them."""
    starts = {}
    for meta in sorted(Path(trajectories).glob('*/trajectory.json')):
        start = json.loads(meta.read_text(encoding='utf-8')).get('start_state')
        starts[meta.parent.name] = _stem(str(start)) if start else meta.parent.name
    return starts


def declaration(missing, seeds) -> list[str]:
    """``curriculum.untracked`` globs covering exactly ``missing`` among ``seeds`` (a room entry's chains ``x_0``,
    ``x_1``, ``x_2`` as one ``x_[0-9]`` when none of them has a track), for the refusal's message."""
    out = set()
    for name in missing:
        glob = re.sub(r'_\d+$', '_[0-9]', name)
        out.add(glob if set(fnmatch.filter(seeds, glob)) <= set(missing) else name)
    return sorted(out)


def check_curriculum(config, trajectories, states_dir=None) -> dict:
    """The curriculum's tracks (name -> seed state); raises ``MissingInput`` when ``trajectories`` holds no track,
    or a seed state of the task has no track and ``curriculum.untracked`` does not declare it."""
    starts = track_starts(trajectories)
    if not starts:
        raise MissingInput(f"{config.name}: the curriculum is on and {trajectories} holds no track "
                           f"(<track>/trajectory.json){_where('trajectories', trajectories)}")
    seeds = sorted(_stem(p) for p in config.resolve_start_states(states_dir))
    declared = untracked(config)
    have = set(starts.values())
    missing = [s for s in seeds if s not in have and not any(fnmatch.fnmatchcase(s, g) for g in declared)]
    if missing:
        raise MissingInput(f"{config.name}: {len(missing)} of {len(seeds)} seed states have no curriculum track in "
                           f"{trajectories} and the task does not declare them: {missing}. Explore them, or add to "
                           f"the task's curriculum.untracked: {declaration(missing, seeds)}")
    return starts


def check_task(config, states_dir=None, trajectories: Optional[str] = None) -> None:
    """Every input a training of ``config`` declares (``trajectories``: its curriculum's folder, None without one)."""
    check_start_states(config, states_dir)
    check_harvested(config, states_dir)
    if trajectories is not None:
        check_curriculum(config, trajectories, states_dir)


def task_demos(config) -> tuple[list[Path], Optional[float]]:
    """The task's ``demos:`` as (files, coeff) (none declared: ([], None)): each name under the demos data folder
    (``paths.data_dir('demos')``); raises ``MissingInput`` naming both places the rule looks (the checkout's folder,
    the store's current snapshot) when a file is not there."""
    demos = config.to_dict().get('demos')
    if not demos:
        return [], None
    folder = paths.data_dir('demos')
    files = [folder / name for name in demos['files']]
    missing = [name for name, path in zip(demos['files'], files) if not path.is_file()]
    if missing:
        store = paths.data_root('demos') / paths.CURRENT
        raise MissingInput(f"{config.name}: demos {missing} are not in {folder} (the demos data: this checkout's "
                           f"{paths.legacy_data('demos')} while it exists, else the store's current snapshot named by "
                           f"{store})")
    return files, float(demos['coeff'])
