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
    """The native ``EnvPool`` lockstep protocol over Python gym-like envs (``reset`` / ``step``).

    The first ``recv`` returns every env's reset observation (zero reward and frames); a terminal step resets in
    the same row with ``final_obs`` / ``final_valid`` holding the terminal observation and ``bootstrap`` set for
    ``info['bootstrap_truncation']`` truncations.  Each finished episode emits an ``episode`` event whose info adds
    ``episode_steps``, ``episode_frames``, ``reward_total`` and ``episode_rewards`` to the terminal step's info.
    """

    def __init__(self, make_env, num_envs: int):
        self.envs = [make_env() for _ in range(num_envs)]
        self.num_envs = num_envs
        self._pending = None
        self._events = []
        self._episodes = [None] * num_envs
        self._space = getattr(self.envs[0], 'observation_space', None) or observation_space()

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

    def recv(self, out, min_ready=None):
        """Writes every env's row (resetting on the first call or stepping the sent actions)."""
        _ = min_ready
        out['env_ids'][:] = np.arange(self.num_envs)
        for i in range(self.num_envs):
            for key in ('terminated', 'truncated', 'bootstrap', 'final_valid'):
                out[key][i] = False
            if self._pending is None:
                obs, reward, frames = self._begin(i), 0.0, 0
            else:
                obs, reward, terminated, truncated, info = self.envs[i].step(self._pending[i])
                frames = int(info.get('frames', 0))
                episode = self._episodes[i]
                episode['steps'] += 1
                episode['frames'] += frames
                episode['reward'] += float(reward)
                episode['rewards'].update(info.get('rewards') or {})
                if terminated or truncated:
                    for key, value in obs.items():
                        out['final_obs'][key][i] = value
                    out['terminated'][i], out['truncated'][i], out['final_valid'][i] = terminated, truncated, True
                    out['bootstrap'][i] = bool(truncated and not terminated and info.get('bootstrap_truncation'))
                    self._events.append({'env': i, 'type': 'episode', 'info': {
                        **info, 'episode_steps': episode['steps'], 'episode_frames': episode['frames'],
                        'reward_total': episode['reward'], 'episode_rewards': dict(episode['rewards'])}})
                    obs = self._begin(i)
            for key, value in obs.items():
                out['obs'][key][i] = value
            out['reward'][i] = reward
            out['frames'][i] = frames
        self._pending = None
        return self.num_envs

    def send(self, env_ids, actions):
        """Queues one action per env (all envs, in order)."""
        assert list(env_ids) == list(range(self.num_envs))
        self._pending = [tuple(int(x) for x in a) for a in np.asarray(actions)]

    def poll_events(self):
        """Episode events since the last call."""
        events, self._events = self._events, []
        return events

    def close(self):
        """No resources."""
