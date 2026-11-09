"""Prioritized level replay (Jiang et al., 2021; better-play §5): seed start states weighted by learning potential.

The cells are a train pool's seed start states (``EnvPool.start_states``); health and items stay randomized per reset,
harvested states and backward-curriculum starts keep their own draws.  After each rollout ``update`` scores every
finished room-entry episode by its **positive value loss**, the mean over the episode's steps of max(A_t, 0) with A_t
the GAE advantage from the learner's recomputed values (``RolloutBuffer.advantages``; an episode that spans rollouts
sums its rows from each).  A state's score S is an exponential moving average over its episodes (``alpha``, default
0.3: a new episode moves S 30% of the way, so S follows about the last three episodes, smoothing over the per-episode
health and action noise without lagging behind the policy for long).  ``weights`` mixes rank-based prioritization with
staleness and uniform sampling:

    P_S(s) ∝ rank(S_s)^(−1/β)          rank 1 = highest score; ties share the best rank
    P_C(s) ∝ c − C_s                   c = scored episodes so far, C_s = c when s was last scored (0 = never)
    P      = (1 − ρ)·P_S + ρ·P_C       P_C is uniform while every staleness is 0
    final  = (1 − u)·P + u / |states|

A state not scored yet takes the highest score (rank 1), so every state gets explored early.  Defaults β = 0.1,
ρ = 0.1 (the paper's), u = 0.1.
"""

import math
from typing import Optional

import numpy as np


class PrioritizedReplay:
    """Per-start-state PLR scores and the start weights they give (``EnvPool.set_start_weights``)."""

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(self, states, beta: float = 0.1, rho: float = 0.1, uniform: float = 0.1, alpha: float = 0.3):
        states = list(states)
        if not states:
            raise ValueError("prioritized level replay needs seed start states (EnvPool.start_states is empty)")
        if beta <= 0:
            raise ValueError(f"PLR beta must be > 0, got {beta}")
        for name, value in (('rho', rho), ('uniform', uniform)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"PLR {name} must be in [0, 1], got {value}")
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"PLR alpha must be in (0, 1], got {alpha}")
        self.states = states
        self.beta, self.rho, self.uniform, self.alpha = beta, rho, uniform, alpha
        self._index = {state: i for i, state in enumerate(states)}
        self.scores: list[Optional[float]] = [None] * len(states)
        self.last_seen = [0] * len(states)
        self.episodes = 0                       # scored episodes so far (the staleness clock)
        self._pending: dict[int, list] = {}     # job -> [sum of max(A, 0), rows] of episodes still running

    def update(self, jobs, advantages, events) -> int:
        """Folds one rollout in: ``jobs`` and ``advantages`` ``[n, T]`` (each row's episode job key and GAE
        advantage), ``events`` the rollout's ``episode`` events (``job``, ``info``).  Returns the episodes scored."""
        jobs = np.asarray(jobs, dtype=np.int64).reshape(-1)
        positive = np.maximum(np.asarray(advantages, dtype=np.float64).reshape(-1), 0.0)
        keys, inverse = np.unique(jobs, return_inverse=True)
        sums = np.bincount(inverse, weights=positive, minlength=len(keys))
        rows = np.bincount(inverse, minlength=len(keys))
        for job, total, count in zip(keys.tolist(), sums.tolist(), rows.tolist()):
            pending = self._pending.setdefault(job, [0.0, 0])
            pending[0] += total
            pending[1] += count
        scored = 0
        for event in events:
            pending = self._pending.pop(event['job'], None)
            info = event['info']
            index = self._index.get(info.get('start_state'))
            if pending is None or not pending[1] or info.get('curriculum') or index is None:
                continue
            self.score(index, pending[0] / pending[1])
            scored += 1
        return scored

    def score(self, index: int, value: float):
        """One finished episode of state ``index`` with positive value loss ``value``."""
        self.episodes += 1
        old = self.scores[index]
        self.scores[index] = value if old is None else (1.0 - self.alpha) * old + self.alpha * value
        self.last_seen[index] = self.episodes

    def weights(self) -> dict[str, float]:
        """{state: sampling probability} (sums to 1)."""
        known = [s for s in self.scores if s is not None]
        top = max(known) if known else 0.0
        scores = np.array([top if s is None else s for s in self.scores])
        ranks = 1 + (scores[None, :] > scores[:, None]).sum(axis=1)
        prioritized = ranks.astype(np.float64) ** (-1.0 / self.beta)
        prioritized /= prioritized.sum()
        staleness = self.episodes - np.array(self.last_seen, dtype=np.float64)
        n = len(self.states)
        stale = staleness / staleness.sum() if staleness.sum() > 0 else np.full(n, 1.0 / n)
        mixed = (1.0 - self.rho) * prioritized + self.rho * stale
        final = (1.0 - self.uniform) * mixed + self.uniform / n
        return dict(zip(self.states, final.tolist()))

    @staticmethod
    def stats(weights: dict[str, float]) -> dict:
        """``plr/entropy`` (nats), ``plr/top_state`` (the largest weight) and ``plr/weight/<state>``."""
        stats = {'plr/entropy': -sum(p * math.log(p) for p in weights.values() if p > 0),
                 'plr/top_state': max(weights.values())}
        stats.update({f'plr/weight/{state}': p for state, p in weights.items()})
        return stats
