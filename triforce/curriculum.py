"""Backward curriculum over explored trajectories (Salimans & Chen 2018, "Learning Montezuma's Revenge from a
Single Demonstration").

``triforce.explore`` writes one successful path per seed start state as ``<dir>/<state>/NNNN.state`` (the
savestate before step NNNN) plus ``trajectory.json``.  ``BackwardCurriculum`` starts some training episodes from
those savestates.  Each trajectory has a start *point*: episodes start uniformly in ``[point, point + stride)``,
where ``stride = ceil(length / stages)``.  The point begins ``stride`` steps before the end.  Once ``window``
episodes from the current stage have finished, the point moves ``stride`` steps earlier if at least
``threshold`` of them succeeded; otherwise the stage's counters restart.  A trajectory whose point reaches the
room entry (index 0) is complete and is no longer sampled: plain seed starts cover it from then on.

Progress is shared by every env worker through a small JSON file (``state_path``) under an ``flock``.  Workers
re-read it on every curriculum start, so a stage advanced by one worker applies to all of them.
"""

import fcntl
import glob
import json
import math
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

TRAJECTORY_FILE = 'trajectory.json'


@dataclass(frozen=True)
class CurriculumSpec:
    """Picklable curriculum settings (passed to env workers)."""
    trajectories: str               # directory written by triforce.explore for one specialist
    state_path: str                 # shared progress file (one per training run)
    fraction: float = 0.5           # share of resets that start from a trajectory while any is incomplete
    stages: int = 10
    window: int = 10
    threshold: float = 0.6


def load_trajectories(directory: str) -> Dict[str, List[str]]:
    """Seed state name -> savestate paths along its successful path (index 0 = room entry)."""
    tracks = {}
    for meta_path in sorted(glob.glob(os.path.join(directory, '*', TRAJECTORY_FILE))):
        with open(meta_path, encoding='utf-8') as file:
            meta = json.load(file)
        paths = sorted(glob.glob(os.path.join(os.path.dirname(meta_path), '[0-9]*.state')))
        if meta.get('success') and paths:
            tracks[meta['start_state']] = paths
    return tracks


def stride_for(length: int, stages: int) -> int:
    """Steps the start point moves per stage."""
    return max(1, math.ceil(length / stages))


@contextmanager
def _locked(path: str):
    with open(path + '.lock', 'a+', encoding='utf-8') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class BackwardCurriculum:
    """Per-env view of the shared curriculum (see module doc)."""

    def __init__(self, spec: CurriculumSpec, rng: Optional[random.Random] = None):
        self.spec = spec
        self.rng = rng or random.Random()
        self.tracks = load_trajectories(spec.trajectories)
        if not self.tracks:
            raise ValueError(f"No successful trajectories under {spec.trajectories}")
        os.makedirs(os.path.dirname(os.path.abspath(spec.state_path)), exist_ok=True)
        with _locked(spec.state_path):
            if not os.path.exists(spec.state_path):
                self._write({name: self._initial(paths) for name, paths in self.tracks.items()})

    def _initial(self, paths: List[str]) -> dict:
        length = len(paths)
        return {'length': length, 'point': max(0, length - stride_for(length, self.spec.stages)),
                'attempts': 0, 'successes': 0, 'stages_passed': 0}

    def _read(self) -> dict:
        with open(self.spec.state_path, encoding='utf-8') as file:
            return json.load(file)

    def _write(self, progress: dict):
        tmp = f"{self.spec.state_path}.{os.getpid()}.tmp"
        with open(tmp, 'w', encoding='utf-8') as file:
            json.dump(progress, file, indent=1)
        os.replace(tmp, self.spec.state_path)

    def progress(self) -> dict:
        """Name -> {length, point, attempts, successes, stages_passed}."""
        with _locked(self.spec.state_path):
            return self._read()

    def next(self) -> Optional[Tuple[str, Tuple[str, int]]]:
        """(savestate path, tag) for a curriculum start, or None (not this reset, or every track complete)."""
        if self.rng.random() >= self.spec.fraction:
            return None
        progress = self.progress()
        active = [name for name, track in progress.items() if track['point'] > 0 and name in self.tracks]
        if not active:
            return None
        name = self.rng.choice(active)
        track = progress[name]
        stride = stride_for(track['length'], self.spec.stages)
        index = self.rng.randrange(track['point'], min(track['length'], track['point'] + stride))
        return self.tracks[name][index], (name, index)

    def record(self, tag: Tuple[str, int], success: bool) -> dict:
        """Counts an episode started from ``tag``; advances the stage when the window is full.  Returns the
        episode's curriculum info: {'track', 'index', 'length', 'point'}."""
        name, index = tag
        with _locked(self.spec.state_path):
            progress = self._read()
            track = progress[name]
            stride = stride_for(track['length'], self.spec.stages)
            if track['point'] <= index < track['point'] + stride:      # episodes from an older stage don't count
                track['attempts'] += 1
                track['successes'] += int(success)
                if track['attempts'] >= self.spec.window:
                    if track['successes'] >= self.spec.threshold * track['attempts']:
                        track['point'] = max(0, track['point'] - stride)
                        track['stages_passed'] += 1
                    track['attempts'] = track['successes'] = 0
                self._write(progress)
        return {'track': name, 'index': index, 'length': track['length'], 'point': track['point']}
