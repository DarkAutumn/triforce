"""Training metrics (design §12, "Measure before tuning").

``EpisodeTracker`` watches every env step (one per env/worker) and accumulates into a ``RolloutMetrics``:
raw counts and sums that merge exactly across workers.  ``RolloutMetrics.summary()`` turns them into the
flat metric dict logged per iteration:

- ``episodes``, ``success_rate``, ``outcome/<success|hard-failure|soft-failure>`` (fractions),
  ``ending/<end-condition>`` (fractions) — from ``info['outcome']`` / ``info['ending']``.  These and every other
  per-episode metric count only episodes that started at a room entry (seed/harvested start states).
  ``RolloutMetrics.start_results`` also counts them per (start state, success) for ``train.EarlyStop``; it is
  not part of the flat dict.
- ``curriculum/episodes``, ``curriculum/success_rate``, ``curriculum/start_fraction`` (mean start index /
  trajectory length; 0 = room entry) and ``curriculum/point_fraction`` (mean current stage point / length)
  for episodes started partway along an explored trajectory (``info['curriculum']``).
- ``reward/total`` and ``reward/<term>``: mean per completed episode, from ``info['rewards']``.
- ``health_at_exit`` (hp / max_hp) and ``hearts_at_exit`` (``info['health']``) at episode end, and
  ``health_change``: hearts at exit minus hearts at the start (after reset-time hp randomization), so +1.5 means
  Link left with 1.5 more hearts than he came in with and -1 means he lost a heart.  ``health_change_total`` adds
  the hearts wasted by picking up dropped hearts at (or near) full health (``info['hearts']``, ``hearts.py``): the
  heart change Link would have had without the max-health cap.
- ``hearts/dropped``, ``hearts/taken``, ``hearts/left`` (still on the floor at the episode's end) and
  ``hearts/wasted``: dropped hearts per completed episode.
- ``episode_frames`` and ``episode_steps``: mean per completed episode (``info['frames']``).
- ``linger/max_steps``, ``linger/max_frames``, ``linger/tail_steps``, ``linger/tail_frames``: means of the
  longest quiet streak and time since last progress; ``linger/worst_steps`` / ``linger/worst_frames`` are
  maxima across measured episodes. Curriculum starts use ``curriculum/linger/*``; absent data stays absent.
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
from .hearts import heart_change_total


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
        self.health_change = 0.0
        self.health_change_total = 0.0
        self.hearts_dropped = 0
        self.hearts_taken = 0
        self.hearts_left = 0
        self.hearts_wasted = 0.0
        self.episode_frames = 0
        self.episode_steps = 0
        self.interrupts = Counter()
        self.kinds = Counter()
        self.durations = Counter()
        self.curriculum_episodes = 0
        self.curriculum_successes = 0
        self.curriculum_start = 0.0
        self.curriculum_point = 0.0
        self.start_results = Counter()     # (start state label, success) -> episodes
        self.linger_episodes = 0
        self.linger_sums = Counter()
        self.linger_worst = Counter()
        self.curriculum_linger_episodes = 0
        self.curriculum_linger_sums = Counter()
        self.curriculum_linger_worst = Counter()
        self.linger_room_episodes = Counter()
        self.linger_room_sums = Counter()
        self.linger_room_worst = Counter()
        self.block_episodes = self.blocks = 0
        self.curriculum_block_episodes = self.curriculum_blocks = 0

    def merge(self, other: 'RolloutMetrics') -> 'RolloutMetrics':
        """Adds other's counts into self."""
        for name, value in vars(other).items():
            mine = getattr(self, name)
            if name in ('linger_worst', 'curriculum_linger_worst', 'linger_room_worst'):
                for key, maximum in value.items():
                    mine[key] = max(mine[key], maximum)
            elif isinstance(mine, Counter):
                mine.update(value)
            else:
                setattr(self, name, mine + value)
        return self

    def record_linger(self, linger: dict, curriculum: bool, room=None):
        """Folds one measured episode into its room-entry or curriculum accumulator."""
        prefix = 'curriculum_' if curriculum else ''
        count_name = f'{prefix}linger_episodes'
        setattr(self, count_name, getattr(self, count_name) + 1)
        sums = getattr(self, f'{prefix}linger_sums')
        worst = getattr(self, f'{prefix}linger_worst')
        for key in ('max_steps', 'max_frames', 'tail_steps', 'tail_frames'):
            sums[key] += linger[key]
        for unit in ('steps', 'frames'):
            worst[unit] = max(worst[unit], linger[f'max_{unit}'])
        if not curriculum and room is not None:
            level, location, in_cave = room
            room_key = f'{level}:0x{location:02x}' + (':cave' if in_cave else '')
            self.linger_room_episodes[room_key] += 1
            for key in ('max_steps', 'max_frames', 'tail_steps', 'tail_frames'):
                self.linger_room_sums[(room_key, key)] += linger[key]
            for unit in ('steps', 'frames'):
                key = (room_key, unit)
                self.linger_room_worst[key] = max(self.linger_room_worst[key], linger[f'max_{unit}'])

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
            result['health_change'] = self.health_change / eps
            result['health_change_total'] = self.health_change_total / eps
            result['hearts/dropped'] = self.hearts_dropped / eps
            result['hearts/taken'] = self.hearts_taken / eps
            result['hearts/left'] = self.hearts_left / eps
            result['hearts/wasted'] = self.hearts_wasted / eps
            result['episode_frames'] = self.episode_frames / eps
            result['episode_steps'] = self.episode_steps / eps
        if self.curriculum_episodes:
            eps = float(self.curriculum_episodes)
            result['curriculum/episodes'] = eps
            result['curriculum/success_rate'] = self.curriculum_successes / eps
            result['curriculum/start_fraction'] = self.curriculum_start / eps
            result['curriculum/point_fraction'] = self.curriculum_point / eps
        for prefix in ('', 'curriculum_'):
            count = getattr(self, f'{prefix}block_episodes')
            if count:
                result['curriculum/blocks' if prefix else 'blocks'] = getattr(self, f'{prefix}blocks') / count
        for prefix in ('', 'curriculum_'):
            count = getattr(self, f'{prefix}linger_episodes')
            if count:
                metric_prefix = 'curriculum/linger' if prefix else 'linger'
                for key, total in getattr(self, f'{prefix}linger_sums').items():
                    result[f'{metric_prefix}/{key}'] = total / count
                for unit, maximum in getattr(self, f'{prefix}linger_worst').items():
                    result[f'{metric_prefix}/worst_{unit}'] = float(maximum)
        for (room, key), total in self.linger_room_sums.items():
            result[f'linger/room/{room}/{key}'] = total / self.linger_room_episodes[room]
        for (room, unit), maximum in self.linger_room_worst.items():
            result[f'linger/room/{room}/worst_{unit}'] = float(maximum)
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

        if done and info.get('linger'):
            metrics.record_linger(info['linger'], bool(info.get('curriculum')), info.get('linger_room'))
        if done and 'blocks' in info:
            prefix = 'curriculum_' if info.get('curriculum') else ''
            setattr(metrics, f'{prefix}block_episodes', getattr(metrics, f'{prefix}block_episodes') + 1)
            setattr(metrics, f'{prefix}blocks', getattr(metrics, f'{prefix}blocks') + info['blocks'])

        if done and info.get('curriculum'):
            curriculum = info['curriculum']
            metrics.curriculum_episodes += 1
            metrics.curriculum_successes += int(info.get('outcome') == 'success')
            metrics.curriculum_start += curriculum['index'] / curriculum['length']
            metrics.curriculum_point += curriculum['point'] / curriculum['length']
            self.begin()
        elif done:
            metrics.episodes += 1
            metrics.outcomes[info.get('outcome', 'unknown')] += 1
            metrics.start_results[(info.get('start_state'), info.get('outcome') == 'success')] += 1
            metrics.endings[info.get('ending', 'unknown')] += 1
            metrics.reward_terms.update(self._rewards)
            metrics.reward_total += self._reward_total
            health = float(info.get('health', 0.0))
            max_health = float(info.get('max_health', 0.0))
            metrics.hearts += health
            health_change = health - float(info.get('start_health', health))
            hearts = info.get('hearts') or {}
            metrics.health_change += health_change
            metrics.health_change_total += heart_change_total(health_change, hearts)
            metrics.hearts_dropped += hearts.get('dropped', 0)
            metrics.hearts_taken += hearts.get('taken', 0)
            metrics.hearts_left += hearts.get('left', 0)
            metrics.hearts_wasted += hearts.get('wasted', 0.0)
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
    keys = keys or ['episodes', 'success_rate', 'reward/total', 'health_change', 'episode_steps',
                    'episode_frames', 'linger/max_steps', 'linger/tail_steps', 'linger/worst_steps',
                    'curriculum/success_rate', 'curriculum/point_fraction', 'entropy/kind',
                    'entropy/direction', 'entropy/duration', 'losses/approx_kl', 'charts/SPS']
    parts = []
    for key in keys:
        if key in metrics:
            value = metrics[key]
            label = key if key.startswith('curriculum/') else key.split('/')[-1]
            parts.append(f"{label}={value:.3g}" if isinstance(value, float) else f"{label}={value}")
    return ' '.join(parts)
