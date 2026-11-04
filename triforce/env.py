"""Gymnasium environment for one specialist (docs/architecture.md, design §3).

An episode starts from one of the specialist's start states (verified by RAM to be in its rooms), applies the
config's RAM randomization, and runs timed options through ``GameSession``.  With a ``curriculum``, a share of
episodes instead start partway along an explored trajectory (``curriculum.py``); their final ``info`` carries
``curriculum`` and they don't feed the start-state pool.  Rewards come from the critic
(``critics.py``) plus a terminal reward; endings come from ``end_conditions.py`` and are classified
deny-by-default (an unlisted ending raises ``UnclassifiedEndingError``).  The final ``info`` also carries
``hearts``, the episode's dropped-heart summary (``hearts.HeartTracker``).

Actions are ``(kind, direction, duration)`` head indices.  Heads that don't apply to the kind are ignored.  A
masked (illegal) action raises ``InvalidActionError`` unless the env was built with ``strict_actions=False``,
in which case it is executed as given (the executor may then interrupt it, e.g. walking into a wall).
"""

import random
from typing import Optional, Union

import gymnasium as gym
import numpy as np

from .actions import NUM_DIRECTIONS, NUM_DURATIONS, NUM_KINDS, Action
from .chain_hp import load_profile, room_key
from .critics import create_critic
from .curriculum import BackwardCurriculum, CurriculumSpec
from .emulator import GAME_DIR
from .end_conditions import MAX_STEPS, NO_ROOM_PROGRESS, build_end_conditions, classify_ending
from .game_map import DEFAULT_ROUTE
from .hearts import HeartTracker
from .linger import LingerTracker
from .objectives import GameCompletion, start_route
from .observation import ObservationBuilder
from .rewards import SUCCESS, StepRewards, apply_terminal_rewards
from .session import GameSession
from .specialist import SpecialistConfig, load_specialist
from .start_states import SavestateCapture, StartStatePool, ram_randomization_values, state_entry, state_label, \
    verify_start_room
from .zelda_enums import Direction, ZeldaItemKind
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
                 strict_actions: bool = True, curriculum: Optional[CurriculumSpec] = None):
        """
        specialist: name of specialists/<name>.yaml or a loaded config.
        session: an existing GameSession to drive (not closed by close()); one is created otherwise.
        start_state: always start from this savestate (name or path) instead of the start-state pool.
        capture_dir: where first-entry savestates are written (None disables capture).
        strict_actions: raise InvalidActionError for masked actions.
        curriculum: start a share of episodes from explored trajectories (``curriculum.BackwardCurriculum``).
        """
        super().__init__()
        self.config = specialist if isinstance(specialist, SpecialistConfig) else load_specialist(specialist)
        self._owns_session = session is None
        self.session = session or GameSession()
        self.start_state = start_state
        self.strict_actions = strict_actions
        self.rng = random.Random()
        self.linger = LingerTracker()

        self.pool = StartStatePool(self.config, self.rng) if start_state is None else None
        self.curriculum = BackwardCurriculum(curriculum, self.rng) if curriculum and start_state is None else None
        self.capture = SavestateCapture(capture_dir) if capture_dir else None
        if self.capture is not None:
            self.session.executor.capture_pre_entry = True
        self.critic = create_critic(self.config.reward_profile, self.config.reward.pbrs,
                                    block_reward=self.config.reward.block_reward,
                                    quiet_enemy_penalty=self.config.reward.quiet_enemy_penalty)
        self.end_conditions = build_end_conditions(self.config.rooms, self.config.success, self.config.one_room,
                                                   linger=self.linger, no_progress_steps=self.config.no_progress_steps)
        self.builder = ObservationBuilder(objective=self.config.obs.objective)
        self.objectives: Optional[GameCompletion] = None

        self._episode_state: Optional[str] = None
        self._curriculum_tag = None
        self._start_health = 0.0
        self._done = True
        self._verified = set()
        self._route, self._route_step = DEFAULT_ROUTE, None     # routes.yaml route of the episode (start_route)

    # --- gym API ---
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
            random.seed(seed)       # GameSession.load draws the game RNG bytes from `random`

        options = options or {}
        name = options.get('start_state') or self.start_state
        self._curriculum_tag = None
        if name is None and self.curriculum is not None:
            start = self.curriculum.next()
            if start is not None:
                name, self._curriculum_tag = start
        name = name or self.pool.next()
        self.session.set_disabled_kinds(self.config.items.disabled_kinds())
        state = self.session.load(name, fresh_room=self.config.fresh_rooms)
        if name not in self._verified:
            verify_start_room(name, state.full_location, self.config.rooms, self.config.name)
            self._verified.add(name)

        spec = self.config.ram_randomization
        chain = load_profile().get(room_key(state.full_location)) \
            if spec.chain_hp and self._curriculum_tag is None else None      # room entries only, not trajectory starts
        values = ram_randomization_values(spec, state.link.max_health, self.rng, chain)
        if values:
            state = self.session.apply_ram(values)
        self._route, self._route_step = start_route(state.full_location, state_entry(name), self.rng, self.config.route)
        return self._begin_episode(state, name)

    def restore(self, snapshot: dict, name: str):
        """Continues from a ``GameSession.snapshot()`` (exact emulator state, no RNG or RAM changes) as a new
        episode labelled ``name``; objectives, critic and end conditions restart as at reset.  For explore."""
        self._curriculum_tag = None
        self.session.set_disabled_kinds(self.config.items.disabled_kinds())
        return self._begin_episode(self.session.restore(snapshot), name)

    def _begin_episode(self, state: ZeldaGame, name: str):
        self.objectives = GameCompletion(self._route, self._route_step)
        self._attach_objectives(None, state)
        self.critic.clear()
        for condition in self.end_conditions:
            condition.clear()

        self._episode_state = name
        self._start_health = state.link.health
        self.hearts = HeartTracker()
        self.hearts.reset(state)
        self.linger.reset(state)
        self._blocks = 0
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
        taken = self.hearts.taken
        self.hearts.update(change)
        self._attach_objectives(change.previous, change.state)
        self.linger.update(change, self.hearts.taken - taken)
        self._blocks += change.blocks
        rewards = StepRewards()
        self.critic.critique_gameplay(change, rewards, quiet_steps=self.linger.tail_steps)

        ending = None
        for condition in self.end_conditions:
            ending = condition.check(change)
            if ending is not None:
                break

        truncated = ending == NO_ROOM_PROGRESS
        if ending is None and self.session.steps >= self.config.max_steps:
            ending, truncated = MAX_STEPS, True
        terminated = ending is not None and not truncated

        state = change.state
        extra = {}
        if ending is not None:
            outcome = classify_ending(ending, self.config.success, self.config.endings)
            apply_terminal_rewards(rewards, ending, outcome, terminated, state.link.health, state.link.max_health)
            self.hearts.finish(state)
            extra = {'ending': ending, 'outcome': outcome, 'start_state': state_label(self._episode_state),
                     'start_health': self._start_health, 'hearts': self.hearts.summary(),
                     'linger': self.linger.summary(), 'linger_room': self.linger.start_room, 'blocks': self._blocks}
            if self._curriculum_tag is not None:
                extra['curriculum'] = self.curriculum.record(self._curriculum_tag, outcome == SUCCESS)
            elif self.pool is not None:
                self.pool.record(self._episode_state, outcome == SUCCESS, state.link.health < self._start_health)
            self._done = True

        if self.capture is not None:
            saved = self.capture.on_step(change, self.session)
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
        if self.config.reward.nearby_items:
            for item in state.items:
                if item.id in (ZeldaItemKind.Heart, ZeldaItemKind.Clock):
                    objectives.pbrs_targets.update(item.link_overlap_tiles)
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
            'specialist': self.config.name, 'route': self._route,
        }
        info.update(extra)
        return info
