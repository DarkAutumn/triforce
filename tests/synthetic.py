"""Synthetic observation spaces, tiny fake envs and a Python stand-in for the native ``EnvPool`` (model/PPO tests
without an emulator)."""

from collections import Counter

import numpy as np
from gymnasium import spaces

from triforce.actions import CARDINALS, DIRECTIONS, MASK_SIZE, NUM_DURATIONS, Action, ActionMask
from triforce.zelda_enums import ActionKind

NUM_SLOTS = 12
ENTITY_FEATURES = 5
GRID_SHAPE = (22, 32)
ROOM_VOCAB = 10
ENTITY_VOCAB = 8
TILE_CLASSES = 6


def observation_space() -> spaces.Dict:
    """A small dict space with the contract's keys (docs/architecture.md)."""
    return spaces.Dict({
        'room_id': spaces.Box(0, ROOM_VOCAB - 1, (1,), dtype=np.int64),
        'room_state': spaces.Box(0.0, 1.0, (5,), dtype=np.float32),
        'link': spaces.Box(-1.0, 1.0, (6,), dtype=np.float32),
        'entity_types': spaces.Box(0, ENTITY_VOCAB - 1, (NUM_SLOTS,), dtype=np.int64),
        'entity_features': spaces.Box(-1.0, 1.0, (NUM_SLOTS, ENTITY_FEATURES), dtype=np.float32),
        'grid': spaces.Box(0, TILE_CLASSES - 1, GRID_SHAPE, dtype=np.int64),
        'history': spaces.Box(-1.0, 1.0, (4,), dtype=np.float32),
        'objective': spaces.Box(0.0, 1.0, (3,), dtype=np.float32),
        'action_mask': spaces.Box(0, 1, (MASK_SIZE,), dtype=bool),
    })


def standard_mask() -> ActionMask:
    """MOVE (4 dirs, all durations), WAIT (all durations), SWORD (4 dirs). BOMB etc. illegal."""
    mask = ActionMask()
    for direction in CARDINALS:
        mask.allow(ActionKind.MOVE, direction)
        mask.allow(ActionKind.SWORD, direction)
    mask.allow(ActionKind.WAIT)
    return mask


def random_mask(rng: np.random.Generator) -> ActionMask:
    """A random legal mask: at least one kind; directional kinds get ≥1 direction; timed kinds get durations."""
    mask = ActionMask()
    kinds = [k for k in ActionKind if rng.random() < 0.5] or [ActionKind.MOVE]
    for kind in kinds:
        if kind == ActionKind.WAIT:
            mask.allow(kind)
            continue
        dirs = DIRECTIONS if kind == ActionKind.BOOMERANG else CARDINALS
        chosen = [d for d in dirs if rng.random() < 0.5] or [dirs[int(rng.integers(len(dirs)))]]
        for direction in chosen:
            mask.allow(kind, direction)
        if kind == ActionKind.MOVE:
            mask.duration[kind] = rng.random(NUM_DURATIONS) < 0.5
            mask.duration[kind, int(rng.integers(NUM_DURATIONS))] = True
    return mask


def random_observation(rng: np.random.Generator, mask: ActionMask | None = None, room: int | None = None,
                       present: int | None = None) -> dict:
    """A random observation matching observation_space()."""
    present = int(rng.integers(0, NUM_SLOTS + 1)) if present is None else present
    types = np.zeros(NUM_SLOTS, dtype=np.int64)
    features = np.zeros((NUM_SLOTS, ENTITY_FEATURES), dtype=np.float32)
    types[:present] = rng.integers(1, ENTITY_VOCAB, present)
    features[:present] = rng.uniform(-1, 1, (present, ENTITY_FEATURES))
    features[:present, 0] = 1.0
    return {
        'room_id': np.array([rng.integers(ROOM_VOCAB) if room is None else room], dtype=np.int64),
        'room_state': rng.uniform(0, 1, 5).astype(np.float32),
        'link': rng.uniform(-1, 1, 6).astype(np.float32),
        'entity_types': types,
        'entity_features': features,
        'grid': rng.integers(0, TILE_CLASSES, GRID_SHAPE).astype(np.int64),
        'history': rng.uniform(-1, 1, 4).astype(np.float32),
        'objective': np.zeros(3, dtype=np.float32),
        'action_mask': (mask or standard_mask()).to_flat(),
    }


# Room → the one rewarded action in the bandit.
BANDIT_TARGETS = {0: Action.parse('MOVE N 16'), 1: Action.parse('SWORD E'), 2: Action.parse('WAIT 32')}


class BanditEnv:
    """One-step episodes: the room id says which (kind, direction, duration) earns +1.

    Rewards 0.25 for the right kind with the wrong direction/duration so all three heads get signal.
    """
    observation_space = observation_space()

    def __init__(self, seed: int = 0, frames: int = 8):
        self.rng = np.random.default_rng(seed)
        self.frames = frames
        self.room = 0
        self.taken = []

    def reset(self, seed=None, options=None):
        """New random room."""
        _ = seed, options
        self.room = int(self.rng.integers(len(BANDIT_TARGETS)))
        return random_observation(self.rng, room=self.room, present=2), {}

    def step(self, action):
        """Scores the action against the room's target."""
        assert standard_mask().to_flat()[int(action[0])], f"illegal kind {action}"
        chosen = Action.from_indices(*action)
        self.taken.append(chosen)
        target = BANDIT_TARGETS[self.room]
        reward = 1.0 if chosen == target else 0.25 if chosen.kind == target.kind else 0.0
        info = {'frames': self.frames, 'interrupt': None, 'rewards': {'bandit': reward},
                'ending': 'done', 'outcome': 'success' if reward == 1.0 else 'soft-failure',
                'health': 3.0, 'max_health': 3.0}
        obs = random_observation(self.rng, room=self.room, present=2)
        return obs, reward, True, False, info

    def close(self):
        """No resources."""



class ListPool:
    """The native ``EnvPool`` protocol over Python gym-like envs (``reset`` / ``step``).

    The first ``recv`` returns every env's reset observation (zero reward and frames); a terminal step resets in
    the same row with ``final_obs`` / ``final_valid`` holding the terminal observation and ``bootstrap`` set for
    ``info['bootstrap_truncation']`` truncations.  Each finished episode emits an ``episode`` event whose info adds
    ``episode_steps``, ``episode_frames``, ``reward_total`` and ``episode_rewards`` to the terminal step's info.

    ``send`` steps the given envs at once; ``recv`` returns ready rows in ascending env order.  By default every
    ready env is returned; with ``schedule_seed`` each ``recv`` returns a random subset of at least ``min_ready`` of
    them (the rest stay ready), imitating asynchronous native workers.
    """

    def __init__(self, make_env, num_envs: int, schedule_seed=None):
        self.envs = [make_env() for _ in range(num_envs)]
        self.num_envs = num_envs
        self._ready = {}            # env -> its next row
        self._started = False
        self._events = []
        self._episodes = [None] * num_envs
        self._space = getattr(self.envs[0], 'observation_space', None) or observation_space()
        self._schedule = None if schedule_seed is None else np.random.default_rng(schedule_seed)

    def make_buffers(self):
        """Zeroed row buffers shaped like the native ones."""
        n = self.num_envs

        def group():
            out = {}
            for key, space in self._space.spaces.items():
                out[key] = np.zeros((n, *space.shape), dtype=space.dtype)
            return out
        rows = {'obs': group(), 'final_obs': group(), 'env_ids': np.zeros(n, np.int64),
                'reward': np.zeros(n, np.float32), 'frames': np.zeros(n, np.int64),
                'interrupt': np.zeros(n, np.int64), 'policy': np.zeros(n, np.int64)}
        for key in ('terminated', 'truncated', 'bootstrap', 'final_valid', 'finished'):
            rows[key] = np.zeros(n, bool)
        rows['mask'] = rows['obs']['action_mask']
        return rows

    def _begin(self, env_index):
        obs, _ = self.envs[env_index].reset()
        self._episodes[env_index] = {'steps': 0, 'frames': 0, 'reward': 0.0, 'rewards': Counter()}
        return obs

    def _step(self, i, action):
        row = {'terminated': False, 'truncated': False, 'bootstrap': False, 'final_valid': False, 'final_obs': None}
        obs, reward, terminated, truncated, info = self.envs[i].step(action)
        frames = int(info.get('frames', 0))
        episode = self._episodes[i]
        episode['steps'] += 1
        episode['frames'] += frames
        episode['reward'] += float(reward)
        episode['rewards'].update(info.get('rewards') or {})
        if terminated or truncated:
            row.update(final_obs=obs, terminated=terminated, truncated=truncated, final_valid=True,
                       bootstrap=bool(truncated and not terminated and info.get('bootstrap_truncation')))
            self._events.append({'env': i, 'type': 'episode', 'info': {
                **info, 'episode_steps': episode['steps'], 'episode_frames': episode['frames'],
                'reward_total': episode['reward'], 'episode_rewards': dict(episode['rewards'])}})
            obs = self._begin(i)
        row.update(obs=obs, reward=reward, frames=frames)
        return row

    def recv(self, out, min_ready=None, timeout=None):
        """Writes ready envs' rows (every env's reset on the first call) and returns how many (``timeout`` is
        ignored: sent envs are ready at once)."""
        if not self._started:
            self._started = True
            for i in range(self.num_envs):
                self._ready[i] = {'obs': self._begin(i), 'reward': 0.0, 'frames': 0, 'terminated': False,
                                  'truncated': False, 'bootstrap': False, 'final_valid': False, 'final_obs': None}
        ids = sorted(self._ready)
        if self._schedule is not None and len(ids) > 1:
            least = min(len(ids), max(1, min_ready or len(ids)))
            take = int(self._schedule.integers(least, len(ids) + 1))
            ids = sorted(int(i) for i in self._schedule.choice(ids, take, replace=False))
        for row, i in enumerate(ids):
            result = self._ready.pop(i)
            out['env_ids'][row] = i
            for key, value in result['obs'].items():
                out['obs'][key][row] = value
            for key, value in (result['final_obs'] or {}).items():
                out['final_obs'][key][row] = value
            for key in ('reward', 'frames', 'terminated', 'truncated', 'bootstrap', 'final_valid'):
                out[key][row] = result[key]
        return len(ids)

    def send(self, env_ids, actions):
        """Steps each given env with its action."""
        for i, action in zip(np.asarray(env_ids).tolist(), np.asarray(actions)):
            assert i not in self._ready, f"env {i} is not awaiting an action"
            self._ready[i] = self._step(i, tuple(int(x) for x in action))

    def poll_events(self):
        """Episode events since the last call."""
        events, self._events = self._events, []
        return events

    def close(self):
        """No resources."""
