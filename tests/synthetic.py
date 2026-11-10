"""Synthetic observation spaces, tiny fake envs and a Python stand-in for the native ``EnvPool`` (model/PPO tests
without an emulator). The stand-in rolls out like the native pool: the Rust policy acts with keyed noise."""

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
    """The native ``EnvPool`` rollout protocol (``obs_spec``, ``rollout``, ``poll_events``) over Python gym-like
    envs (``reset`` / ``step``), acting with the given ``_native.Policy``.

    A terminal step resets the env at once; ``bootstrap_truncation`` truncations get the policy's value of the
    final observation in ``bootstrap_value``. Each finished episode emits an ``episode`` event (with its ``job``) whose
    info adds ``episode_steps``, ``episode_frames``, ``reward_total`` and ``episode_rewards`` to the terminal step's
    info. Each episode's job key is ``env * 1_000_003 + episode`` (``out['job']`` when given).
    """

    def __init__(self, make_env, num_envs: int):
        self.envs = [make_env() for _ in range(num_envs)]
        self.num_envs = num_envs
        self._obs = [None] * num_envs
        self._job = [0] * num_envs
        self._step = [0] * num_envs
        self._episode_count = [0] * num_envs
        self._events = []
        self._episodes = [None] * num_envs
        self._space = getattr(self.envs[0], 'observation_space', None) or observation_space()

    @property
    def obs_spec(self):
        """Key → ``{'shape', 'dtype'}``, like the native pool's."""
        return {key: {'shape': list(space.shape), 'dtype': np.dtype(space.dtype).name}
                for key, space in self._space.spaces.items()}

    def _begin(self, i):
        obs, _ = self.envs[i].reset()
        self._episodes[i] = {'steps': 0, 'frames': 0, 'reward': 0.0, 'rewards': Counter()}
        self._job[i] = i * 1_000_003 + self._episode_count[i]
        self._episode_count[i] += 1
        self._step[i] = 0
        self._obs[i] = obs

    def _batch(self, observations):
        return {key: np.stack([np.asarray(o[key]) for o in observations]) for key in self._space.spaces}

    def rollout(self, policies, steps, out, deterministic=False):
        """Steps every env ``steps`` decisions with ``policies[0]`` and writes the native rollout buffers."""
        from triforce import _native  # pylint: disable=import-outside-toplevel
        policy = policies[0]
        n = self.num_envs
        for i in range(n):
            if self._obs[i] is None:
                self._begin(i)

        def store_obs(t):
            for key, value in out['obs'].items():
                value[:, t] = np.stack([np.asarray(o[key]) for o in self._obs]).reshape(value[:, t].shape)
            out['starts'][:, t] = [float(s == 0) for s in self._step]

        for t in range(steps):
            store_obs(t)
            batch = self._batch(self._obs)
            if deterministic:
                result = policy.evaluate(batch, deterministic=True)
            else:
                result = policy.evaluate(batch, noise=_native.keyed_noise(self._job, self._step, policy.noise_size))
            out['actions'][:, t] = result['actions']
            out['logp_ent_val'][:, t] = np.stack([result['log_prob'], result['entropy'], result['value']], -1)
            if 'job' in out:
                out['job'][:, t] = self._job
            for i in range(n):
                action = tuple(int(a) for a in result['actions'][i])
                reward, frames, bootstrap_obs, interrupt = self._advance(i, action)
                out['rewards'][i, t] = reward
                out['frames'][i, t] = frames
                if 'interrupts' in out:
                    out['interrupts'][i, t] = interrupt
                out['bootstrap_value'][i, t] = 0.0 if bootstrap_obs is None else \
                    float(policy.value(self._batch([bootstrap_obs]))[0])
        store_obs(steps)
        return n

    def _advance(self, i, action):
        """Steps env ``i``; (reward, frames, final observation of a bootstrapped truncation or None, interrupt)."""
        obs, reward, terminated, truncated, info = self.envs[i].step(action)
        frames = int(info.get('frames', 0))
        episode = self._episodes[i]
        episode['steps'] += 1
        episode['frames'] += frames
        episode['reward'] += float(reward)
        episode['rewards'].update(info.get('rewards') or {})
        self._step[i] += 1
        bootstrap = None
        if terminated or truncated:
            if truncated and not terminated and info.get('bootstrap_truncation'):
                bootstrap = obs
            self._events.append({'env': i, 'type': 'episode', 'job': self._job[i], 'info': {
                **info, 'episode_steps': episode['steps'], 'episode_frames': episode['frames'],
                'reward_total': episode['reward'], 'episode_rewards': dict(episode['rewards'])}})
            self._begin(i)
        else:
            self._obs[i] = obs
        return reward, frames, bootstrap, int(info.get('interrupt') or 0)

    def poll_events(self):
        """Episode events since the last call."""
        events, self._events = self._events, []
        return events

    def close(self):
        """No resources."""
