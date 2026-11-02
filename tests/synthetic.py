"""Synthetic observation spaces and tiny fake envs for model/PPO tests (no emulator).

Lives in an importable module so spawn-based PPO workers can unpickle the fake env.
"""

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

