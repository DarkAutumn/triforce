"""Training metrics (design §12, "Measure before tuning").

``EpisodeTracker`` watches every env step (one per env/worker) and accumulates into a ``RolloutMetrics``:
raw counts and sums that merge exactly across workers.  ``RolloutMetrics.summary()`` turns them into the
flat metric dict logged per iteration:

- ``episodes``, ``success_rate``, ``outcome/<success|hard-failure|soft-failure>`` (fractions),
  ``ending/<end-condition>`` (fractions) — from ``info['outcome']`` / ``info['ending']``.
- ``reward/total`` and ``reward/<term>``: mean per completed episode, from ``info['rewards']``.
- ``health_at_exit`` (hp / max_hp) and ``hearts_at_exit`` (``info['health']``) at episode end.
- ``episode_frames`` and ``episode_steps``: mean per completed episode (``info['frames']``).
- ``interrupt/<reason>``: fraction of steps that ended with each ``info['interrupt']`` reason.
- ``kind/<KIND>`` and ``duration/<KIND>/<frames>``: action-kind and per-kind duration histograms (fractions).
- ``rollout/steps``, ``rollout/frames``.

Entropy per head and steps/sec come from the trainer.  ``MetricsLogger`` writes every iteration's dict to
TensorBoard and to ``metrics.jsonl`` in the run directory.
"""

import json
import math
import os
import time
from collections import Counter
from typing import Optional

from .actions import DURATIONS, KINDS, TIMED_KINDS


class RolloutMetrics:
    """Mergeable raw counts/sums for a batch of rollouts."""
    # pylint: disable=too-many-instance-attributes

    def __init__(self):
        self.steps = 0
        self.frames = 0
        self.episodes = 0
        self.outcomes = Counter()
        self.endings = Counter()
        self.reward_terms = Counter()
        self.reward_total = 0.0
        self.health_fraction = 0.0
        self.hearts = 0.0
        self.episode_frames = 0
        self.episode_steps = 0
        self.interrupts = Counter()
        self.kinds = Counter()
        self.durations = Counter()

    def merge(self, other: 'RolloutMetrics') -> 'RolloutMetrics':
        """Adds other's counts into self."""
        for name, value in vars(other).items():
            mine = getattr(self, name)
            if isinstance(mine, Counter):
                mine.update(value)
            else:
                setattr(self, name, mine + value)
        return self

    def summary(self) -> dict[str, float]:
        """Flat metric dict (see module docstring)."""
        result = {'rollout/steps': float(self.steps), 'rollout/frames': float(self.frames),
                  'episodes': float(self.episodes)}
        if self.episodes:
            eps = float(self.episodes)
            result['success_rate'] = self.outcomes.get('success', 0) / eps
            for name, count in self.outcomes.items():
                result[f'outcome/{name}'] = count / eps
            for name, count in self.endings.items():
                result[f'ending/{name}'] = count / eps
            result['reward/total'] = self.reward_total / eps
            for name, total in self.reward_terms.items():
                result[f'reward/{name}'] = total / eps
            result['health_at_exit'] = self.health_fraction / eps
            result['hearts_at_exit'] = self.hearts / eps
            result['episode_frames'] = self.episode_frames / eps
            result['episode_steps'] = self.episode_steps / eps
        if self.steps:
            for name, count in self.interrupts.items():
                result[f'interrupt/{name}'] = count / self.steps
            for name, count in self.kinds.items():
                result[f'kind/{name}'] = count / self.steps
        for kind in TIMED_KINDS:
            total = sum(self.durations[(kind.name, d)] for d in DURATIONS)
            if total:
                for duration in DURATIONS:
                    result[f'duration/{kind.name}/{duration}'] = self.durations[(kind.name, duration)] / total
        return result


class EpisodeTracker:
    """Follows one env's steps and folds finished episodes into ``metrics``."""

    def __init__(self):
        self.metrics = RolloutMetrics()
        self._rewards = Counter()
        self._reward_total = 0.0
        self._frames = 0
        self._steps = 0

    def begin(self):
        """Starts a new episode (after env.reset)."""
        self._rewards = Counter()
        self._reward_total = 0.0
        self._frames = 0
        self._steps = 0

    def step(self, action, reward: float, info: dict, done: bool):
        """Records one env step; ``action`` is the (kind, direction, duration) index triple."""
        metrics = self.metrics
        kind = KINDS[int(action[0])]
        frames = int(info.get('frames', 0))
        metrics.steps += 1
        metrics.frames += frames
        metrics.kinds[kind.name] += 1
        if kind in TIMED_KINDS:
            metrics.durations[(kind.name, DURATIONS[int(action[2])])] += 1
        interrupt = info.get('interrupt')
        metrics.interrupts[str(interrupt) if interrupt else 'none'] += 1

        self._steps += 1
        self._frames += frames
        self._reward_total += reward
        for name, value in (info.get('rewards') or {}).items():
            self._rewards[name] += float(value)

        if done:
            metrics.episodes += 1
            metrics.outcomes[info.get('outcome', 'unknown')] += 1
            metrics.endings[info.get('ending', 'unknown')] += 1
            metrics.reward_terms.update(self._rewards)
            metrics.reward_total += self._reward_total
            health = float(info.get('health', 0.0))
            max_health = float(info.get('max_health', 0.0))
            metrics.hearts += health
            metrics.health_fraction += health / max_health if max_health > 0 else 0.0
            metrics.episode_frames += self._frames
            metrics.episode_steps += self._steps
            self.begin()

    def take(self) -> RolloutMetrics:
        """Returns the accumulated metrics and starts a fresh accumulator (in-progress episode is kept)."""
        result, self.metrics = self.metrics, RolloutMetrics()
        return result


class MetricsLogger:
    """TensorBoard + JSONL metric sink for one training run."""

    def __init__(self, run_dir: str, tensorboard: bool = True):
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self.jsonl_path = os.path.join(run_dir, 'metrics.jsonl')
        self._writer = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter  # pylint: disable=import-outside-toplevel
            self._writer = SummaryWriter(os.path.join(run_dir, 'tb'))

    def log(self, iteration: int, steps: int, metrics: dict[str, float]):
        """Writes one iteration's metrics."""
        clean = {k: float(v) for k, v in metrics.items() if v is not None and math.isfinite(float(v))}
        record = {'iteration': iteration, 'steps': steps, 'time': time.time(), **clean}
        with open(self.jsonl_path, 'a', encoding='utf-8') as file:
            file.write(json.dumps(record) + '\n')
        if self._writer is not None:
            for name, value in clean.items():
                self._writer.add_scalar(name, value, steps)
            self._writer.flush()

    def close(self):
        """Closes the TensorBoard writer."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def read_jsonl(path: str) -> list[dict]:
    """Reads a metrics.jsonl file."""
    with open(path, encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]


def format_summary(metrics: dict[str, float], keys: Optional[list[str]] = None) -> str:
    """One-line human summary for the console."""
    keys = keys or ['episodes', 'success_rate', 'reward/total', 'health_at_exit', 'episode_steps',
                    'episode_frames', 'entropy/kind', 'entropy/direction', 'entropy/duration',
                    'losses/approx_kl', 'charts/SPS']
    parts = []
    for key in keys:
        if key in metrics:
            value = metrics[key]
            parts.append(f"{key.split('/')[-1]}={value:.3g}" if isinstance(value, float) else f"{key}={value}")
    return ' '.join(parts)
