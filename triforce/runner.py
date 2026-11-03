"""Plays a route end to end with trained specialists (design §9).

On every room entry the runner looks up the room's ``specialist:`` tag in game.yaml, loads that specialist's
promoted checkpoint (cached), and lets it act until the next room transition.  Models switch only on room
transitions.  An untagged room, or a specialist without a promoted model, stops the run with an error.
"""

import json
import random
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from .actions import Action
from .game_map import GameMap
from .objectives import GameCompletion
from .observation import ObservationBuilder
from .registry import load_model, promoted_checkpoint
from .session import GameSession
from .specialist import load_specialist
from .zelda_enums import MapLocation

ROUTES_PATH = __file__.rsplit('/', 1)[0] + '/routes.yaml'
DEFAULT_MAX_STEPS = 50_000
# Per-room budget: a specialist's max_steps is a training episode cap (boss-aquamentus: 400, to keep curriculum
# episodes short); a chain room gets at least this many steps before it counts as stuck.
MIN_ROOM_STEPS = 2000

RoomKey = Tuple[int, int, bool]


class RunnerError(RuntimeError):
    """The runner can't continue: unknown room, missing model, bad route."""


@dataclass
class Route:
    """Where a route starts and which room ends it."""
    name: str
    start_state: str
    terminal: MapLocation

    @staticmethod
    def load(name: str, stop_at: Optional[MapLocation] = None, start_from: Optional[str] = None) -> 'Route':
        """Loads a route from routes.yaml.  ``stop_at`` replaces the terminal room and ``start_from`` (a savestate,
        e.g. a route entry state like ``0_37c_2``) the game start, for partial chain evals."""
        with open(ROUTES_PATH, 'r', encoding='utf-8') as f:
            routes = yaml.safe_load(f)
        steps = routes.get(name)
        if not isinstance(steps, list) or not steps:
            raise RunnerError(f"Unknown route '{name}' (routes.yaml has {[k for k, v in routes.items() if isinstance(v, list)]})")
        if steps[0].get('entry') != 'start':
            raise RunnerError(f"Route '{name}' must begin at the game start (entry: start)")
        terminals = [s for s in steps if s.get('exit') == 'terminal']
        if stop_at is None and not terminals:
            raise RunnerError(f"Route '{name}' has no terminal room")
        last = terminals[-1] if terminals else steps[-1]
        terminal = stop_at or MapLocation(int(last['level']), int(str(last['room']), 16), bool(last.get('in_cave')))
        return Route(name, start_from or 'start', terminal)


@dataclass
class RoomRecord:
    """One visit to one room."""
    index: int
    level: int
    room: str
    in_cave: bool
    specialist: str
    steps: int = 0
    frames: int = 0
    health_in: float = 0.0
    health_out: float = 0.0
    max_health: int = 0
    outcome: str = 'in-progress'    # exit-correct, exit-wrong, death, stuck, milestone, error


@dataclass
class RunResult:
    """Everything one run produced; written as JSON lines by ``write_jsonl``."""
    route: str
    seed: Optional[int]
    result: str = 'in-progress'     # success, failure, error
    reason: str = ''
    steps: int = 0
    frames: int = 0
    rooms: List[RoomRecord] = field(default_factory=list)

    @property
    def failed_room(self) -> Optional[RoomRecord]:
        """The room the run ended in, if it didn't succeed."""
        return self.rooms[-1] if self.rooms and self.result != 'success' else None

    def to_dict(self) -> dict:
        """JSON-friendly dict."""
        return asdict(self)


def write_jsonl(path: str, results: List[RunResult]):
    """Appends results (one per line) to path."""
    with open(path, 'a', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result.to_dict()) + '\n')


class _Specialist:
    """A loaded specialist: config, network and its own observation builder."""
    def __init__(self, name: str, checkpoint_path, device: str = 'cpu'):
        self.name = name
        self.config = load_specialist(name)
        with torch.random.fork_rng():   # building the network draws init weights; keep seeded runs reproducible
            self.network, self.checkpoint = load_model(checkpoint_path)
        self.network.eval()
        self.network.to(device)
        self.builder = ObservationBuilder(objective=self.config.obs.objective)

    def act(self, obs, deterministic: bool) -> Action:
        """Chooses an action for one observation."""
        with torch.no_grad():
            indices = self.network.get_action(obs, deterministic=deterministic)
        return Action.from_indices(*np.asarray(indices.cpu()).reshape(-1)[:3])


class Runner:
    """Drives one GameSession through a route, switching specialists on room transitions."""
    # pylint: disable=too-many-instance-attributes
    def __init__(self, session: Optional[GameSession] = None, models_dir=None,
                 checkpoints: Optional[Dict[str, str]] = None, deterministic: bool = False,
                 max_steps: int = DEFAULT_MAX_STEPS, renderer=None, game_map: Optional[GameMap] = None,
                 device: str = 'cpu'):
        """
        checkpoints: specialist name → checkpoint path, overriding the registry's promoted model.
        renderer: optional object with ``show(session, specialist, action, change)`` (watching only).
        device: torch device for the specialists' networks ('cpu' or 'cuda').
        """
        self._owns_session = session is None
        self.session = session or GameSession()
        self.models_dir = models_dir
        self.overrides = dict(checkpoints or {})
        self.deterministic = deterministic
        self.max_steps = max_steps
        self.renderer = renderer
        self.device = device
        self.game_map = game_map or GameMap.load()
        self._specialists: Dict[str, _Specialist] = {}

    def close(self):
        """Releases the emulator if the runner created it."""
        if self._owns_session:
            self.session.close()

    def specialist_for(self, location: MapLocation) -> str:
        """The specialist tagged on a room.  Raises RunnerError for unknown or untagged rooms."""
        room = self.game_map.get(location)
        if room is None:
            raise RunnerError(f"Room {_describe(location)} is not in game.yaml")
        if not room.specialist:
            raise RunnerError(f"Room {_describe(location)} has no specialist: tag in game.yaml")
        return room.specialist

    def _load(self, name: str) -> _Specialist:
        if name not in self._specialists:
            path = self.overrides.get(name) or promoted_checkpoint(name, self.models_dir)
            self._specialists[name] = _Specialist(name, path, self.device)
        return self._specialists[name]

    def run(self, route: Route, seed: Optional[int] = None) -> RunResult:
        """Plays route once.  Never raises for gameplay failures; RunnerError ends the run with result=error."""
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        result = RunResult(route.name, seed)
        try:
            self._run(route, result)
        except (RunnerError, FileNotFoundError) as err:
            result.result, result.reason = 'error', str(err)
            if result.rooms and result.rooms[-1].outcome == 'in-progress':
                result.rooms[-1].outcome = 'error'
        return result

    def _run(self, route: Route, result: RunResult):
        # pylint: disable=too-many-branches,too-many-statements
        session = self.session
        state = session.load(route.start_state, fresh_room=True)    # a pre-entry --from state spawns its full room
        objectives = GameCompletion()
        state.objectives = objectives.get_current_objectives(None, state)

        current: Optional[_Specialist] = None
        record: Optional[RoomRecord] = None
        obs = None
        change = None

        while True:
            location = state.full_location
            if record is None or (record.level, int(record.room, 16), record.in_cave) != _key(location):
                correct = None
                if record is not None:
                    record.health_out = state.link.health
                    correct = change is not None and location in change.previous.objectives.next_rooms
                    record.outcome = 'exit-correct' if correct else 'exit-wrong'

                if location == route.terminal:
                    result.result, result.reason = 'success', f"entered {_describe(location)}"
                    result.rooms.append(self._new_record(len(result.rooms), location, '-', state, 'milestone'))
                    return

                # A wrong exit into a room no specialist plays is a gameplay failure, not a game.yaml problem
                # (that is only an error when the objective wanted the room).
                room = self.game_map.get(location)
                if correct is False and (room is None or not room.specialist):
                    result.result = 'failure'
                    result.reason = f"{record.specialist} left the route into untagged {_describe(location)}"
                    return

                name = self.specialist_for(location)
                record = self._new_record(len(result.rooms), location, name, state)
                result.rooms.append(record)
                if current is None or current.name != name:
                    current = self._load(name)
                    session.set_disabled_kinds(current.config.items.disabled_kinds())
                    obs = current.builder.reset(state, session.mask)
                elif change is not None:
                    obs = current.builder.build(change, session.mask)
            elif change is not None:
                obs = current.builder.build(change, session.mask)

            room_steps = max(current.config.max_steps, MIN_ROOM_STEPS)
            if record.steps >= room_steps:
                record.outcome = 'stuck'
                result.result, result.reason = 'failure', f"{current.name} exceeded {room_steps} steps"
                return
            if result.steps >= self.max_steps:
                record.outcome = 'stuck'
                result.result, result.reason = 'failure', f"run exceeded {self.max_steps} steps"
                return

            action = current.act(obs, self.deterministic)
            if not session.mask.is_allowed(action):
                raise RunnerError(f"{current.name} chose masked action {action} (legal: {session.mask})")

            change = session.act(action)
            state = change.state
            state.objectives = objectives.get_current_objectives(change.previous, state)
            record.steps += 1
            record.frames += change.frames
            result.steps += 1
            result.frames += change.frames

            if self.renderer is not None:
                self.renderer.show(session, current.name, action, change)

            if state.game_over:
                record.outcome = 'death'
                record.health_out = 0.0
                result.result, result.reason = 'failure', f"died in {_describe(state.full_location)}"
                return

    @staticmethod
    def _new_record(index, location, name, state, outcome='in-progress') -> RoomRecord:
        return RoomRecord(index, location.level, f"0x{location.value:02x}", bool(location.in_cave), name,
                          health_in=state.link.health, health_out=state.link.health,
                          max_health=state.link.max_health, outcome=outcome)


def _key(location: MapLocation) -> RoomKey:
    return (location.level, location.value, bool(location.in_cave))


def _describe(location: MapLocation) -> str:
    return f"(level {location.level}, room 0x{location.value:02x}{', cave' if location.in_cave else ''})"


def parse_location(text: str) -> MapLocation:
    """'2:0x0e', '0:0x77:cave' → MapLocation."""
    parts = text.split(':')
    return MapLocation(int(parts[0]), int(parts[1], 16), len(parts) > 2 and parts[2] == 'cave')


def chain_summary(results: List[RunResult]) -> dict:
    """The milestone metric and a per-room failure breakdown."""
    runs = len(results)
    successes = sum(r.result == 'success' for r in results)
    failures: Dict[str, int] = {}
    for r in results:
        room = r.failed_room
        if room is not None:
            key = f"{room.level}:{room.room}{':cave' if room.in_cave else ''} {room.specialist} {room.outcome}"
            failures[key] = failures.get(key, 0) + 1
    return {'runs': runs, 'successes': successes, 'success_rate': successes / runs if runs else 0.0,
            'failures': dict(sorted(failures.items(), key=lambda kv: -kv[1])),
            'errors': sorted({r.reason for r in results if r.result == 'error'})}
