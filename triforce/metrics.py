"""Training metrics (design §12, "Measure before tuning").

``RolloutMetrics`` accumulates raw counts and sums: ``add_steps`` takes batches of env steps
(actions, frames, interrupt codes) and ``add_episode`` every terminal info (the pool's ``episode`` events).
``RolloutMetrics.summary()`` turns them into the flat metric dict logged per iteration:

- ``episodes``, ``success_rate``, ``outcome/<success|hard-failure|soft-failure|neutral>`` (fractions),
  ``ending/<end-condition>`` (fractions) — from ``info['outcome']`` / ``info['ending']``.  These and every other
  per-episode metric count only episodes that started at a room entry (seed/harvested start states).  They count
  the episodes that *ended* in the rollout, so slow failures arrive late (one to several rollouts after they start).
- ``in_flight/episodes``, ``in_flight/mean_steps``, ``in_flight/max_steps`` and ``in_flight/over_half_budget``
  (share of envs whose episode is older than half the step budget): the episode every env has running at the
  rollout's end (curriculum ones included; an env that just reset counts with age 0).
  ``success_rate_pessimistic``: this rollout's room-entry successes over its finished room-entry episodes plus the
  in-flight episodes older than the slowest success seen so far (counted as failures).
- ``started/success_rate``, ``started/episodes``, ``started/iteration``: the per-start cohort rate: the
  room-entry episodes that *started* in a rollout, reported once every episode started in it (and in every earlier
  rollout) has ended; a delayed but unbiased number.  ``RolloutMetrics.cohorts`` keeps the released cohorts' per
  (start state, success) counts for ``train.EarlyStop``; ``EpisodeTracker`` follows the episodes by job key.
- ``curriculum/episodes``, ``curriculum/success_rate``, ``curriculum/start_fraction`` (mean start index /
  trajectory length; 0 = room entry) and ``curriculum/point_fraction`` (mean current stage point / length)
  for episodes started partway along an explored trajectory (``info['curriculum']``).
- ``reward/total`` and ``reward/<term>``: mean per completed episode, from ``info['reward_total']`` and
  ``info['episode_rewards']`` (the env's reward, without the trainer's bootstrap value).
- ``health_at_exit`` (hp / max_hp) and ``hearts_at_exit`` (``info['health']``) at episode end, and
  ``health_change``: hearts at exit minus hearts at the start (after reset-time hp randomization), so +1.5 means
  Link left with 1.5 more hearts than he came in with and -1 means he lost a heart.  ``health_change_total`` adds
  the hearts wasted by picking up dropped hearts at (or near) full health (``info['hearts']['wasted']``): the
  heart change Link would have had without the max-health cap.
- ``hearts/dropped``, ``hearts/taken``, ``hearts/healed`` (health those pickups actually restored),
  ``hearts/left`` (still on the floor at the episode's end) and ``hearts/wasted``: dropped hearts per completed
  episode.
- ``bombs/used``: bombs Link placed per completed episode (``info['bombs']``), and ``cheated/bombs``,
  ``cheated/keys``, ``cheated/rupees``: cheated items used per completed episode (``info['cheated']``; a room that
  needs an item Link lacks grants it: a bomb in ``requires_bombs`` rooms, a key at a locked route door, rupees in
  ``requires_rupees`` shops; crates/zelda/src/cheats.rs).  ``bombs/<stat>`` for each of ``BOMB_STATS`` per
  completed episode (crates/zelda/src/bombs.rs): bombs placed, enemy hits credited to them, bombs that hit
  nothing, the damage they dealt (wood-sword hits: HP lost, a kill's remaining HP), 10th damage-free kills (any
  weapon), bombs placed while the next kill was the 10th, forced bomb drops armed, bomb pickups.
- ``boss/stuns`` and ``boss/eats``: Dodongo stuns (a bomb's smoke; the sword can kill him then) and bombs he
  swallowed per completed episode, from ``info['boss']`` (crates/env/src/critic.rs ``BossTransitions``).
- ``clips/<stat>`` for each of ``CLIP_STATS`` per completed episode (``info['clips']``; crates/zelda/src/clips.rs):
  grid-point skips completed (``clips/done``), started and not completed (``clips/aborted``), completed per
  direction (``clips/done_n`` ...).
- ``episode_frames`` and ``episode_steps``: mean per completed episode (``info['episode_frames']`` /
  ``info['episode_steps']``).
- ``linger/max_steps``, ``linger/max_frames``, ``linger/tail_steps``, ``linger/tail_frames``: means of the
  longest quiet streak and time since last progress; ``linger/worst_steps`` / ``linger/worst_frames`` are
  maxima across measured episodes. Curriculum starts use ``curriculum/linger/*``; absent data stays absent.
- ``interrupt/<reason>``: fraction of steps that ended with each interrupt (``INTERRUPTS[code]``).
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

import numpy as np

from ._native import BOMB_STATS, CHEAT_ITEMS, CLIP_STATS, INTERRUPTS
from .actions import DURATIONS, KINDS, TIMED_KINDS


def heart_change_total(health_change: float, hearts: Optional[dict]) -> float:
    """Real health change plus the hearts wasted by picking them up at (or near) full health."""
    return health_change + (hearts or {}).get('wasted', 0.0)


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
        self.hearts_healed = 0.0
        self.bombs_used = 0
        self.bomb_stats = Counter()
        self.cheated = Counter()
        self.boss_stuns = 0
        self.boss_eats = 0
        self.clip_stats = Counter()
        self.episode_frames = 0
        self.episode_steps = 0
        self.interrupts = Counter()
        self.kinds = Counter()
        self.durations = Counter()
        self.curriculum_episodes = 0
        self.curriculum_successes = 0
        self.curriculum_start = 0.0
        self.curriculum_point = 0.0
        self.cohorts: list[tuple[int, Counter]] = []    # released cohorts: (rollout, (start state, success) -> n)
        self.extra: dict[str, float] = {}               # per-rollout values (in_flight/*, started/*): not summed
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

    def add_steps(self, actions, frames, interrupts):
        """Counts one batch of env steps: ``actions`` ``[n, 3]`` (kind, direction, duration indices), ``frames``
        ``[n]`` and ``interrupts`` ``[n]`` (``INTERRUPTS`` codes or names; None/0 = none)."""
        actions = np.asarray(actions).reshape(-1, 3)
        frames = np.asarray(frames).reshape(-1)
        self.steps += len(actions)
        self.frames += int(frames.sum())
        for (kind_index, _, duration_index), interrupt in zip(actions.tolist(), list(interrupts)):
            kind = KINDS[int(kind_index)]
            self.kinds[kind.name] += 1
            if kind in TIMED_KINDS:
                self.durations[(kind.name, DURATIONS[int(duration_index)])] += 1
            if isinstance(interrupt, str):
                name = interrupt
            else:
                name = INTERRUPTS[int(interrupt)] if interrupt is not None else 'none'
            self.interrupts[name] += 1

    def add_episode(self, info: dict):
        """Folds one finished episode's terminal info into the counts."""
        if info.get('linger'):
            self.record_linger(info['linger'], bool(info.get('curriculum')), info.get('linger_room'))
        if 'blocks' in info:
            prefix = 'curriculum_' if info.get('curriculum') else ''
            setattr(self, f'{prefix}block_episodes', getattr(self, f'{prefix}block_episodes') + 1)
            setattr(self, f'{prefix}blocks', getattr(self, f'{prefix}blocks') + info['blocks'])
        if info.get('curriculum'):
            curriculum = info['curriculum']
            self.curriculum_episodes += 1
            self.curriculum_successes += int(info.get('outcome') == 'success')
            self.curriculum_start += curriculum['index'] / curriculum['length']
            self.curriculum_point += curriculum['point'] / curriculum['length']
            return
        self.episodes += 1
        self.outcomes[info.get('outcome', 'unknown')] += 1
        self.endings[info.get('ending', 'unknown')] += 1
        for name, value in (info.get('episode_rewards') or {}).items():
            self.reward_terms[name] += float(value)
        self.reward_total += float(info.get('reward_total', 0.0))
        health = float(info.get('health', 0.0))
        max_health = float(info.get('max_health', 0.0))
        self.hearts += health
        health_change = health - float(info.get('start_health', health))
        hearts = info.get('hearts') or {}
        self.health_change += health_change
        self.health_change_total += heart_change_total(health_change, hearts)
        self.hearts_dropped += hearts.get('dropped', 0)
        self.hearts_taken += hearts.get('taken', 0)
        self.hearts_left += hearts.get('left', 0)
        self.hearts_wasted += hearts.get('wasted', 0.0)
        self.hearts_healed += hearts.get('healed', 0.0)
        bombs = info.get('bombs') or {}
        self.bombs_used += bombs.get('used', 0)
        for stat in BOMB_STATS:
            self.bomb_stats[stat] += bombs.get(stat, 0)
        for item, n in (info.get('cheated') or {}).items():
            self.cheated[item] += n
        boss = info.get('boss') or {}
        self.boss_stuns += boss.get('stuns', 0)
        self.boss_eats += boss.get('eats', 0)
        clips = info.get('clips') or {}
        for stat in CLIP_STATS:
            self.clip_stats[stat] += clips.get(stat, 0)
        self.health_fraction += health / max_health if max_health > 0 else 0.0
        self.episode_frames += int(info.get('episode_frames', 0))
        self.episode_steps += int(info.get('episode_steps', 0))

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
            result['hearts/healed'] = self.hearts_healed / eps
            result['bombs/used'] = self.bombs_used / eps
            for stat in BOMB_STATS:
                result[f'bombs/{stat}'] = self.bomb_stats[stat] / eps
            for item in CHEAT_ITEMS:
                result[f'cheated/{item}'] = self.cheated[item] / eps
            result['boss/stuns'] = self.boss_stuns / eps
            result['boss/eats'] = self.boss_eats / eps
            for stat in CLIP_STATS:
                result[f'clips/{stat}'] = self.clip_stats[stat] / eps
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
        result.update(self.extra)
        return result


class EpisodeTracker:
    """Follows a training pool's episodes across rollouts by their job keys (``ppo.Collector``).

    ``update`` takes one rollout: each row's ``jobs`` / ``steps`` (decision index in the episode; 0 = its first row),
    whether each env's next observation starts a new episode (``fresh``) and the rollout's ``episode`` events.  An
    episode belongs to the cohort of the rollout holding its first row.  A cohort is released once every episode in
    it, and in every earlier cohort, has ended; it carries the per (start state, success) counts of its room-entry
    episodes (curriculum ones only close it).  ``budget``: the env's step limit (``max_steps``), for
    ``in_flight/over_half_budget``."""

    def __init__(self, budget: Optional[int] = None):
        self.budget = budget
        self.rollouts = 0
        self.released = 0
        self.slowest_success = 0
        self._started: dict[int, int] = {}                  # job -> rollout of its first row
        self._open: dict[int, list] = {}                    # rollout -> [episodes still running, Counter]

    def update(self, jobs, steps, fresh, events) -> tuple[list[tuple[int, Counter]], dict[str, float]]:
        """(released cohorts in rollout order, ``in_flight/*``, ``success_rate_pessimistic`` and ``started/*``)."""
        jobs, steps, fresh = np.asarray(jobs), np.asarray(steps), np.asarray(fresh, dtype=bool)
        self.rollouts += 1
        cohort = self._open[self.rollouts] = [0, Counter()]
        for job in jobs[steps == 0].tolist():
            self._started[job] = self.rollouts
            cohort[0] += 1
        finished = successes = 0
        for event in events:
            info = event['info']
            start = self._started.pop(event['job'], None)
            if start is not None:
                self._open[start][0] -= 1
            if info.get('curriculum'):
                continue
            success = info.get('outcome') == 'success'
            finished += 1
            successes += success
            if success:
                self.slowest_success = max(self.slowest_success, int(info.get('episode_steps', 0)))
            if start is not None:
                self._open[start][1][(info.get('start_state'), success)] += 1
        released = []
        while self.released < self.rollouts and self._open[self.released + 1][0] == 0:
            self.released += 1
            counts = self._open.pop(self.released)[1]
            if counts:      # a cohort without room-entry episodes carries nothing
                released.append((self.released, counts))

        ages = np.where(fresh, 0, steps[:, -1] + 1) if steps.shape[-1] else np.zeros(len(fresh), np.int64)
        values = {'in_flight/episodes': float(len(ages)), 'in_flight/mean_steps': float(ages.mean()),
                  'in_flight/max_steps': float(ages.max())}
        if self.budget:
            values['in_flight/over_half_budget'] = float((ages > self.budget / 2).mean())
        overdue = int((ages > self.slowest_success).sum())
        if finished + overdue:
            values['success_rate_pessimistic'] = successes / (finished + overdue)
        started = sum((counts for _, counts in released), Counter())
        total = sum(started.values())
        if total:
            values['started/success_rate'] = sum(n for (_, success), n in started.items() if success) / total
            values['started/episodes'] = float(total)
            values['started/iteration'] = float(released[-1][0])
        return released, values


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
    keys = keys or ['episodes', 'success_rate', 'success_rate_pessimistic', 'started/success_rate',
                    'harvested/success_rate', 'in_flight/mean_steps', 'in_flight/max_steps',
                    'in_flight/over_half_budget', 'reward/total',
                    'health_change', 'episode_steps', 'episode_frames', 'linger/max_steps', 'linger/tail_steps',
                    'linger/worst_steps', 'curriculum/success_rate', 'curriculum/point_fraction', 'entropy/kind',
                    'entropy/direction', 'entropy/duration', 'losses/approx_kl', 'charts/SPS']
    parts = []
    for key in keys:
        if key in metrics:
            value = metrics[key]
            label = (key if key.startswith(('curriculum/', 'in_flight/', 'started/', 'harvested/'))
                     else key.split('/')[-1])
            parts.append(f"{label}={value:.3g}" if isinstance(value, float) else f"{label}={value}")
    return ' '.join(parts)
