"""Gymnasium environment for one specialist (docs/architecture.md, design §3).

An episode starts from one of the specialist's start states (verified by RAM to be in its rooms), applies the
config's RAM randomization, and runs timed options through ``GameSession``.  Rewards come from the critic
(``critics.py``) plus a terminal reward; endings come from ``end_conditions.py`` and are classified
deny-by-default (an unlisted ending raises ``UnclassifiedEndingError``).

Actions are ``(kind, direction, duration)`` head indices.  Heads that don't apply to the kind are ignored.  A
masked (illegal) action raises ``InvalidActionError`` unless the env was built with ``strict_actions=False``,
in which case it is executed as given (the executor may then interrupt it, e.g. walking into a wall).
"""

import random
from typing import Optional, Union

import gymnasium as gym
import numpy as np

from .actions import NUM_DIRECTIONS, NUM_DURATIONS, NUM_KINDS, Action
from .critics import create_critic
from .emulator import GAME_DIR
from .end_conditions import MAX_STEPS, build_end_conditions, classify_ending
from .objectives import GameCompletion
from .observation import ObservationBuilder
from .rewards import SUCCESS, StepRewards, apply_terminal_rewards
from .session import GameSession
from .specialist import SpecialistConfig, load_specialist
from .start_states import SavestateCapture, StartStatePool, ram_randomization_values, verify_start_room
from .zelda_enums import Direction
from .zelda_game import ZeldaGame

_CARDINALS = (Direction.N, Direction.S, Direction.E, Direction.W)


class InvalidActionError(ValueError):
    """step() was given an action the current mask forbids."""


class ZeldaEnv(gym.Env):
    """One specialist's task as a gymnasium environment."""
    metadata = {'render_modes': []}
    observation_space = ObservationBuilder.observation_space()
    action_space = gym.spaces.MultiDiscrete([NUM_KINDS, NUM_DIRECTIONS, NUM_DURATIONS])

    def __init__(self, specialist: Union[str, SpecialistConfig], session: Optional[GameSession] = None,
                 start_state: Optional[str] = None, capture_dir: Optional[str] = GAME_DIR,
                 strict_actions: bool = True):
        """
        specialist: name of specialists/<name>.yaml or a loaded config.
        session: an existing GameSession to drive (not closed by close()); one is created otherwise.
        start_state: always start from this savestate (name or path) instead of the start-state pool.
        capture_dir: where first-entry savestates are written (None disables capture).
        strict_actions: raise InvalidActionError for masked actions.
        """
        super().__init__()
        self.config = specialist if isinstance(specialist, SpecialistConfig) else load_specialist(specialist)
        self._owns_session = session is None
        self.session = session or GameSession()
        self.start_state = start_state
        self.strict_actions = strict_actions
        self.rng = random.Random()

        self.pool = StartStatePool(self.config, self.rng) if start_state is None else None
        self.capture = SavestateCapture(capture_dir) if capture_dir else None
        self.critic = create_critic(self.config.reward_profile, self.config.reward.pbrs)
        self.end_conditions = build_end_conditions(self.config.rooms, self.config.success)
        self.builder = ObservationBuilder(objective=self.config.obs.objective)
        self.objectives: Optional[GameCompletion] = None

        self._episode_state: Optional[str] = None
        self._start_health = 0.0
        self._done = True
        self._verified = set()

    # --- gym API ---
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
            random.seed(seed)       # GameSession.load draws the game RNG bytes from `random`

        options = options or {}
        name = options.get('start_state') or self.start_state or self.pool.next()
        state = self.session.load(name)
        if name not in self._verified:
            verify_start_room(name, state.full_location, self.config.rooms, self.config.name)
            self._verified.add(name)

        values = ram_randomization_values(self.config.ram_randomization, state.link.max_health, self.rng)
        if values:
            state = self.session.apply_ram(values)

        self.objectives = GameCompletion()
        self._attach_objectives(None, state)
        self.critic.clear()
        for condition in self.end_conditions:
            condition.clear()

        self._episode_state = name
        self._start_health = state.link.health
        self._done = False
        obs = self.builder.reset(state, self.session.mask)
        return obs, self._info(state, {'start_state': name})

    def step(self, action):
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset()")

        kind, direction, duration = (int(x) for x in np.asarray(action).reshape(-1)[:3])
        act = Action.from_indices(kind, direction, duration)
        if not self.session.mask.is_allowed(act):
            if self.strict_actions:
                raise InvalidActionError(f"{act} is masked; legal: {self.session.mask}")

        change = self.session.act(act)
        self._attach_objectives(change.previous, change.state)

        rewards = StepRewards()
        self.critic.critique_gameplay(change, rewards)

        ending = None
        for condition in self.end_conditions:
            ending = condition.check(change)
            if ending is not None:
                break

        truncated = False
        if ending is None and self.session.steps >= self.config.max_steps:
            ending, truncated = MAX_STEPS, True
        terminated = ending is not None and not truncated

        state = change.state
        extra = {}
        if ending is not None:
            outcome = classify_ending(ending, self.config.success, self.config.endings)
            apply_terminal_rewards(rewards, ending, outcome, terminated, state.link.health, state.link.max_health)
            if self.pool is not None:
                self.pool.record(self._episode_state, outcome == SUCCESS, state.link.health < self._start_health)
            extra = {'ending': ending, 'outcome': outcome}
            self._done = True

        if self.capture is not None:
            saved = self.capture.on_step(change, self.session.emu)
            if saved:
                extra['captured_state'] = saved

        obs = self.builder.build(change, self.session.mask)
        info = self._info(state, extra)
        info.update(frames=change.frames, interrupt=change.interrupt, action=str(act), rewards=rewards.to_dict())
        return obs, float(rewards.value), terminated, truncated, info

    def close(self):
        if self._owns_session:
            self.session.close()

    # --- helpers ---
    def _attach_objectives(self, prev: Optional[ZeldaGame], state: ZeldaGame):
        """Sets state.objectives (always: end conditions use next_rooms) and, with PBRS on, state.pbrs_wavefront."""
        objectives = self.objectives.get_current_objectives(prev, state)
        state.objectives = objectives
        if self.config.reward.pbrs and not state.game_over:
            locked_doors = None
            if state.link.keys > 0 and state.level > 0:
                locked_doors = frozenset(d for d in _CARDINALS if state.is_door_locked(d)) or None
            state.pbrs_wavefront = state.room.calculate_wavefront_for_link(objectives.pbrs_targets,
                                                                           locked_doors=locked_doors)

    def _info(self, state: ZeldaGame, extra: dict) -> dict:
        info = {
            'frames': 0, 'interrupt': None, 'action': None, 'rewards': {},
            'room': (state.level, state.location, state.in_cave),
            'health': state.link.health, 'max_health': state.link.max_health,
            'steps': self.session.steps, 'total_frames': self.session.total_frames,
            'specialist': self.config.name,
        }
        info.update(extra)
        return info
