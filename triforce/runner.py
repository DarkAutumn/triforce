"""Plays a route end to end with trained specialists (design §9).

The chain itself runs natively (``ChainTask`` in an ``EnvPool``): on every room entry the room's game.yaml
``specialist:`` tag picks who acts, specialists switch only on room transitions, and a run ends with a
``RunResult``.  This module owns the models: each pool row names its acting specialist (the ``policy`` column, an
index into the loaded-model list), and the specialists act inside the pool (``EnvPool.play`` with their Rust
networks, docs/rust-actor.md).  Sampled actions use Gumbel noise keyed to (run seed, decision), so a seed plays the
same run whichever env or thread runs it.  A specialist without a promoted model ends the runs that need it as
``error`` with the registry's reason; an untagged room the route needs is an error too, a wrong exit into one a
gameplay failure.  With ``record`` set every run is recorded as a replay (``triforce.replay``).
"""

import difflib
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from . import __version__, _native, replay
from .paths import SPECIALISTS_DIR, STATES_DIR
from .provenance import (checkpoint_description, checkpoint_label, code_version, data_hash, has_hit_head,
                         location_text, models_header, network_description, replay_models)
from .registry import (CHECKPOINT_NAME, DEFAULT_ACT, act_mode, checkpoint_sha1, load_checkpoint, load_model,
                       own_models_until, parse_act, policy_checkpoint, route_part_key, route_policy, split_policy)
from .replay import ReplayDir
from .specialist import SpecialistConfig, list_specialists, load_specialist
from .zelda_enums import MapLocation

DEFAULT_MAX_STEPS = 50_000

RoomKey = Tuple[int, int, bool]


class RunnerError(RuntimeError):
    """The runner can't start: unknown or malformed route."""


@dataclass
class Route:
    """Where a route starts and what ends it.  The start: ``start_state`` (``'start'``: the game start state), or
    ``power_on`` (``{'name', 'delay', 'max_delay'}``: console power on, ``delay`` None = drawn per run seed from
    ``0..=max_delay``), or ``checkpoint`` (``{'state': bytes, 'step': int, 'granted': [bombs, keys, rupees],
    'route_granted': [bombs, keys, rupees]}``: the state another run ended in, e.g. ``Replay.states()[-1]``, and the
    items a room and a route step granted that Link still holds there).  The end: ``terminal`` (a room), or
    ``stop_step`` (a route step, reached by the route's own exit), or neither: the route's last triforce.
    ``start_step``: the route step the start is on."""
    name: str
    start_state: str
    terminal: Optional[MapLocation] = None
    stop_step: Optional[int] = None
    power_on: Optional[dict] = None
    checkpoint: Optional[dict] = None
    start_step: int = 0
    clips: bool = True                          # the run-time clips switch a run starts with (round 22)
    clips_off_from: Optional[object] = None     # turn it off on entering this route step (int) or room (MapLocation)

    @staticmethod
    def load(name: str, stop_at: Optional[MapLocation] = None, start_from: Optional[str] = None,
             states_dir=None, *, stop_step: Optional[int] = None, power_on: Optional[dict] = None,
             checkpoint: Optional[dict] = None, clips: bool = True, clips_off_from=None) -> 'Route':
        """Loads a route from routes.yaml.  ``stop_at`` ends it on entering that room, ``stop_step`` on reaching
        that route step, instead of on its last step's triforce.  One of ``start_from`` (a savestate, e.g. a
        route entry state like ``2_7ds_0``, in ``states_dir``, default ``paths.STATES_DIR``, or a path),
        ``power_on`` and ``checkpoint`` replaces the game start (``Route``).  The start state must exist, the
        stop must be after the start's route step, and a power-on name and delay must be ones the game takes
        (the native ``Route`` checks the same when the chain task is built)."""
        # pylint: disable=too-many-arguments, too-many-branches
        try:
            steps = _native.route_steps(name)
        except _native.ConfigError as err:
            raise RunnerError(str(err)) from err
        if not steps:
            raise RunnerError(f"Unknown route '{name}'")
        if steps[0]['entry'] != 'start':
            raise RunnerError(f"Route '{name}' must begin at the game start (entry: start)")
        if stop_at is None and stop_step is None and steps[-1]['exit'] not in ('triforce', 'end'):
            raise RunnerError(f"Route '{name}' ends on neither a triforce nor the game's end")
        if stop_at is not None and stop_step is not None:
            raise RunnerError("a stop room and a stop step are exclusive")
        if start_from == 'start':
            start_from = None
        starts = [what for what, given in (('a start state (--from)', start_from), ('a power-on start', power_on),
                                           ('a checkpoint', checkpoint)) if given is not None]
        if len(starts) > 1:
            raise RunnerError(f"{' and '.join(starts)} are exclusive: each replaces the game start")
        start = 0
        if start_from is not None:      # phase X: `--route d3first --from 3_7cs_0` plays d3first_3_7cs_0
            try:
                start_from = _native.route_state(name, start_from)
            except ValueError as err:
                raise RunnerError(str(err)) from err
            path = _start_state_path(start_from, states_dir)
            if not path.is_file():
                near = difflib.get_close_matches(path.stem, [p.stem for p in path.parent.glob('*.state')], n=5)
                raise RunnerError(f"Unknown start state '{start_from}': {path} not found (states in {path.parent}"
                                  + (f"; close: {near}" if near else '') + ')')
            start = _start_step(steps, path.stem)
        if power_on is not None:
            power_on = _power_on(power_on)
        if checkpoint is not None:
            start = checkpoint.get('step')
            if not isinstance(start, int) or not 0 <= start < len(steps):
                raise RunnerError(f"the checkpoint's step {start!r} is not a step of route '{name}' "
                                  f"(0..={len(steps) - 1})")
            checkpoint = {'state': bytes(checkpoint['state']), 'step': start,
                          'granted': list(checkpoint.get('granted') or (0, 0, 0)),
                          'route_granted': list(checkpoint.get('route_granted') or (0, 0, 0)),
                          'randomize_rng': bool(checkpoint.get('randomize_rng', False))}
        if stop_at is not None:
            later = [(s['level'], s['room'], s['in_cave']) for s in steps[start + 1:]]
            if (stop_at.level, stop_at.value, stop_at.in_cave) not in later:
                rooms = sorted(set(later), key=later.index)
                raise RunnerError(f"--stop-at {location_text(stop_at)} is not a room of route '{name}' after the start "
                                  f"step; rooms: {[location_text(MapLocation(*room)) for room in rooms]}")
            way = 'clips' if clips else 'plain'
            if clips_off_from is None and (stop_at.level, stop_at.value, stop_at.in_cave) not in [
                    (s['level'], s['room'], s['in_cave']) for s in steps[start + 1:] if s['way'] in (None, way)]:
                raise RunnerError(f"--stop-at {location_text(stop_at)} is not a room of the {way} way of route "
                                  f"'{name}' after the start step (the run stays on the {way} way)")
        if stop_step is not None and not start < stop_step < len(steps):
            raise RunnerError(f"--stop-at step:{stop_step} is not a step of route '{name}' after the start step "
                              f"{start}; steps: {start + 1}..={len(steps) - 1}")
        return Route(name, start_from or 'start', stop_at, stop_step, power_on, checkpoint, start, clips, clips_off_from)

    def steps_played(self) -> int:
        """How many route steps a run that gets to the end plays (an estimate's chain work, ``jobs.work``)."""
        return steps_played(self.name, self.start_step, self.terminal, self.stop_step)

    def steps_in_play(self) -> list[dict]:
        """The route steps (``_native.route_steps``) a run of this route can play: from the start step to its end
        (before ``stop_step`` or the first later step in ``terminal``, which end the run when reached; else to the
        route's last), on the way the clips switch takes (both ways when ``clips_off_from`` flips it on the way)."""
        steps = _native.route_steps(self.name)
        if self.stop_step is not None or self.terminal is not None:
            steps = steps[:self.start_step + steps_played(self.name, self.start_step, self.terminal, self.stop_step)]
        ways = (None, 'clips' if self.clips else 'plain') if self.clips_off_from is None else (None, 'clips', 'plain')
        return [step for step in steps[self.start_step:] if step['way'] in ways]


def steps_played(route_name: str, start_step: int = 0, stop_room: Optional[MapLocation] = None,
                 stop_step: Optional[int] = None) -> int:
    """The route steps from ``start_step`` to a run's end (``stop_step``, the first later step in ``stop_room``,
    else the route's last step), at least 1; both ways of a stretch count, so this is an upper bound."""
    steps = _native.route_steps(route_name)
    if stop_step is not None:
        return max(1, stop_step - start_step)
    if stop_room is not None:
        room = (stop_room.level, stop_room.value, stop_room.in_cave)
        found = next((step['index'] for step in steps[start_step + 1:]
                      if (step['level'], step['room'], step['in_cave']) == room), None)
        if found is not None:
            return max(1, found - start_step)
    return max(1, len(steps) - start_step)


def chain_steps(route_name: str, runs: int, start_from: Optional[str] = None, stop_at: Optional[str] = None) -> int:
    """A chain's work in route steps (``jobs.work`` ``chain_steps``): ``runs`` x the steps each plays from the start
    state ``start_from`` (a route entry state name or path; None: the game start) to ``stop_at`` (``--stop-at``);
    without checking either (a malformed one counts the whole route)."""
    try:
        steps = _native.route_steps(route_name)
        start = _start_step(steps, Path(start_from).stem) if start_from and start_from != 'start' else 0
        stop_room, stop_step = parse_stop(stop_at)
        return runs * steps_played(route_name, start, stop_room, stop_step)
    except (RunnerError, ValueError, _native.ConfigError):
        try:
            return runs * len(_native.route_steps(route_name))
        except _native.ConfigError:
            return 0


def chain_entry(route_name: str, runs: int, parallel: int, start_from: Optional[str] = None,
                stop_at: Optional[str] = None, power_on: bool = False, name: Optional[str] = None) -> dict:
    """One chain of a job's work for the queue's estimate (``jobs.work`` ``chains``, ``jobqueue.chain_model``): its
    ``shape`` (``<route>:<start step>-<end step>``, ``:power-on`` from console power on: the same segment whatever
    its start state is called), ``runs``, ``parallel``, ``steps`` (the route steps a run is asked to play) and the
    batch entry's ``name``."""
    steps = chain_steps(route_name, 1, None if power_on else start_from, stop_at)
    try:
        route_steps = _native.route_steps(route_name)
        start = (_start_step(route_steps, Path(start_from).stem)
                 if start_from and start_from != 'start' and not power_on else 0)
    except (ValueError, _native.ConfigError):     # a malformed name counts from the start, as ``chain_steps``
        start = 0
    shape = f"{route_name}:{start}-{start + steps}" + (':power-on' if power_on else '')
    return {'shape': shape, 'runs': runs, 'parallel': parallel, 'steps': steps,
            **({'name': name} if name is not None else {})}


def _power_on(options: dict) -> dict:
    """A power-on start's ``{'name', 'delay', 'max_delay'}`` checked and filled in (name LINK, a random delay over
    the whole range); ``RunnerError`` for a name the game can't register or a delay out of range."""
    unknown = sorted(set(options) - {'name', 'delay', 'max_delay'})
    if unknown:
        raise RunnerError(f"unknown power-on options {unknown} (name, delay, max_delay)")
    limit = _native.MAX_POWER_ON_DELAY
    try:
        name = _native.power_on_name(options.get('name') or 'LINK')
    except ValueError as err:
        raise RunnerError(str(err)) from err
    delay, max_delay = options.get('delay'), options.get('max_delay')
    max_delay = limit if max_delay is None else max_delay
    for what, value in (('delay', delay), ('max delay', max_delay)):
        if value is not None and (not isinstance(value, int) or not 0 <= value <= limit):
            raise RunnerError(f"the power-on {what} {value!r} must be a frame count in 0..={limit}")
    return {'name': name, 'delay': delay, 'max_delay': max_delay}


def _start_state_path(name: str, states_dir=None) -> Path:
    """Where the chain looks for start state ``name``: a ``.state`` path as is, else ``<states_dir>/<name>.state``."""
    path = Path(name)
    if path.suffix == '.state' and (path.is_absolute() or path.parent != Path('.')):
        return path
    return Path(states_dir or STATES_DIR) / (name if name.endswith('.state') else f'{name}.state')


def _start_step(steps: List[dict], label: str) -> int:
    """The first route step a start state named ``<level>_<room><side>[_v<visit>]_<chain>`` (or a harvested
    ``..._h<seed>_<n>``) begins: same room, entry and visit (``_native.state_name``, the env's
    ``start_states::StateName``); 0 when the name doesn't say."""
    named = _native.state_name(label)
    return next((s['index'] for s in steps
                 if (s['level'], s['room'], s['in_cave'], s['entry'], s['visit']) == named), 0)


@dataclass
class RoomRecord:
    """One visit to one room."""
    index: int
    level: int
    room: str
    in_cave: bool
    specialist: str
    # The policy that played the visit when it is not the family's own (round 32: ``family@route``, the route's own
    # run before its ``own_models_until`` step; the run is the chain's provenance ``models``).  Absent: the family's.
    policy: Optional[str] = None
    steps: int = 0
    frames: int = 0
    health_in: float = 0.0
    health_out: float = 0.0
    max_health: int = 0
    keys_in: Optional[int] = None   # keys carried on entering / leaving the room (None in pre-Nov-6 records)
    keys_out: Optional[int] = None
    bombs_in: Optional[int] = None  # natural bombs carried on entering / leaving the room (None before round 25)
    bombs_out: Optional[int] = None
    # Bombs of a route grant (routes.yaml `grant:`, crates/zelda/src/cheats.rs) held on entering the room (round 32:
    # d3first's bag from Dungeon 5's entrance; bombs_in + route_bombs_in is the bag).  0 in older records.
    route_bombs_in: int = 0
    outcome: str = 'in-progress'    # exit-correct, exit-wrong, retreat, retreat-limit, missed (heart waypoint), death,
    #                                 stuck (room or run budgets, wrong-exit or bounce limit), milestone, error
    # Dropped hearts during the visit: healed = real hearts gained, wasted = taken - healed, left = still on the
    # floor when the visit ended.
    hearts_dropped: int = 0
    hearts_taken: int = 0
    hearts_healed: float = 0.0
    hearts_wasted: float = 0.0
    hearts_left: int = 0
    hearts_expired: int = 0
    linger: dict = field(default_factory=dict)
    blocks: int = 0
    no_damage_max: Optional[int] = None     # longest stretch of actions without damage dealt or taken (None: older)
    # Bombs placed during the visit, and cheated items used (a room that needs an item Link lacks grants it: a bomb
    # in requires-bombs rooms, a key at a locked route door, rupees in requires-rupees shops;
    # crates/zelda/src/cheats.rs).  0 in older records.
    bombs_used: int = 0
    cheated_bombs: int = 0
    cheated_keys: int = 0
    cheated_rupees: int = 0
    # Dodongo stuns (a bomb's smoke) and bombs he swallowed during the visit (crates/env/src/critic.rs
    # BossTransitions).  0 in older records.
    boss_stuns: int = 0
    boss_eats: int = 0
    # Bombs in the visit (crates/zelda/src/bombs.rs): placed, enemy hits credited to them, placed without a hit,
    # damage dealt (wood-sword hits: HP lost, a kill's remaining HP), 10th damage-free kills (any weapon), bombs
    # placed while the next kill was the 10th, forced bomb drops armed (a bomb made the 10th kill), bomb pickups;
    # round 25: dropped bombs that appeared, were taken, were left on the floor on leaving, expired.
    # 0 in older records.
    bomb_placed: int = 0
    bomb_hits: int = 0
    bomb_misses: int = 0
    bomb_damage: int = 0
    bomb_tenth_kills: int = 0
    bomb_tenth_tries: int = 0
    bomb_armed: int = 0
    bomb_picked_up: int = 0
    bomb_dropped: int = 0
    bomb_drop_taken: int = 0
    bomb_drop_left: int = 0
    bomb_drop_expired: int = 0
    # Grid-point skips in the visit (crates/zelda/src/clips.rs): completed, aborted, completed per direction, and
    # aborted ones a MOVE finished (round 22). 0 in older records.
    clip_done: int = 0
    clip_aborted: int = 0
    clip_done_n: int = 0
    clip_done_s: int = 0
    clip_done_w: int = 0
    clip_done_e: int = 0
    clip_finished_by_move: int = 0
    # Round 22 (None/absent in older records): actions that lost health and hits on enemies in the visit, the route
    # step Link played on entering it (None off the route), the run-time clips switch then and every flip during the
    # visit ({'step': actions into the visit, 'clips': the new value}; step 0 = at the entry).
    hits: Optional[int] = None
    hits_dealt: Optional[int] = None
    step: Optional[int] = None
    clips_allowed: Optional[bool] = None
    clips_flips: List[dict] = field(default_factory=list)
    # Round 38 (None/absent in older records): steps a Bubble's touch landed in (the sword blocked, no health lost)
    # and the frames of the steps begun with the sword blocked.
    bubble_touches: Optional[int] = None
    sword_blocked_frames: Optional[int] = None
    # Round 38 (None/absent in older records; ``_native.AIMLESS_FIELDS``, docs/architecture.md "Aimless walking"):
    # the visit's calm stretches: frames beyond the walk their ground needed, calm frames, the walk's frames,
    # stretches, calm decisions, those that ended farther from the goal, MOVEs opposite the stretch's previous one.
    aimless_frames: Optional[int] = None
    calm_frames: Optional[int] = None
    calm_needed_frames: Optional[int] = None
    calm_stretches: Optional[int] = None
    calm_decisions: Optional[int] = None
    calm_away: Optional[int] = None
    calm_reversals: Optional[int] = None
    # Round 38, "Beams at range" (None/absent in older records; crates/zelda/src/beams.rs BeamStats): sword shots fired
    # and those the projection had meeting a monster; decisions lined up for a shot and those that moved or waited;
    # kills by a shot out of melee range, in melee range on a sword step, by anything else, and the summed distance
    # (pixels) of the first two; hearts lost; steps that cost the beams.
    beam_shots: Optional[int] = None
    beam_shots_lined: Optional[int] = None
    beam_lined: Optional[int] = None
    beam_lined_unfired: Optional[int] = None
    beam_kills_far: Optional[int] = None
    beam_kills_near: Optional[int] = None
    beam_kills_other: Optional[int] = None
    beam_kill_distance: Optional[int] = None
    beam_hearts_lost: Optional[float] = None
    beam_beams_lost: Optional[int] = None
    # Round 40 (None/absent in older records; BeamStats too): shots that damaged a monster (beams::shot_hit), the
    # frames of the visit's steps begun holding the beams (a sword and the hearts a shot needs), and the visit's frame
    # at the end of the first step that lost them (None: not lost in the visit).
    beam_shots_hit: Optional[int] = None
    beam_held_frames: Optional[int] = None
    beam_lost_frame: Optional[int] = None
    # Decisions taken in the visit (crates/zelda/src/decisions.rs, triforce/variety.py): n, low1/low5 (chosen with
    # probability under 1%/5% under the policy), low1_hit/low5_hit/low1_death/low5_death (followed by a hit or death
    # within 4 decisions), surprisal, act_entropy, actions (hex, two digits per decision), health ([decisions,
    # hits on their own step] by Link's half hearts at the decision; round 25) and, when the act mode's foresight ran
    # (``guard:K``, ``plan:K:S[:R]``), foresight (outcomes, trials, trial_frames, replaced).  {} in older records.
    decisions: dict = field(default_factory=dict)
    # Maze loops during the visit (round 24, crates/zelda/src/change.rs MazeLoop): correct ones (the sequence's next
    # edge, the visit goes on) and wrong ones (each one of the run's wrong exits).  0 in older records.
    maze_loops: int = 0
    maze_wrong_loops: int = 0
    # Round 25 (crates/zelda/src/supply.rs; 0/[] in older records): kills in the visit, the damage-free kill streak
    # ($627) on entering and its highest value, fairies dropped and taken, and every dropped heart or fairy
    # ({kind, step, deficit, to_drop, to_objective, fate: taken|expired|left|death, after, hits}).
    kills: int = 0
    streak_in: int = 0
    streak_max: int = 0
    fairies_dropped: int = 0
    fairies_taken: int = 0
    drops: List[dict] = field(default_factory=list)
    # Round 40 (None/absent in older records; crates/zelda/src/goals.rs, docs/architecture.md "Route goals"): what the
    # visit's route step expected, from the route's data (a screen wrap: the exit is a wrap exit or `wrap_in`; no hit:
    # the room has a planned path, `path_costs`; the carriers: the route's carrier targets apply), and what happened:
    # wraps completed, whether the objective was ever FIGHT (else the route crossed the room), WorldKillCycle on
    # entering and the deaths that moved it, a carrier aimed at on entering and at all, carriers killed, the bombs they
    # dropped and those taken (booleans as 0/1).
    goal_wrap_expected: Optional[int] = None
    goal_no_hit_expected: Optional[int] = None
    goal_carriers_expected: Optional[int] = None
    goal_wraps: Optional[int] = None
    goal_fought: Optional[int] = None
    goal_cycle_in: Optional[int] = None
    goal_cycle_kills: Optional[int] = None
    goal_carrier_in: Optional[int] = None
    goal_carrier_seen: Optional[int] = None
    goal_carrier_kills: Optional[int] = None
    goal_carrier_bombs: Optional[int] = None
    goal_carrier_bombs_taken: Optional[int] = None
    # Round 46 (None/absent in older records; goals.rs): the living monsters on entering, at most and on leaving; the
    # way out known, the shortest walk to it (8-pixel tiles) from where it was first known, the tiles walked, the
    # detour (tiles the shortest walk through the farthest-off tile Link stood on adds), the frames of actions that
    # ended farther from the way out and those of them with no bomb reason, drop-cycle deaths with a bomb reason.
    goal_alive_in: Optional[int] = None
    goal_alive_max: Optional[int] = None
    goal_alive_out: Optional[int] = None
    goal_way_known: Optional[int] = None
    goal_way_in: Optional[int] = None
    goal_walked: Optional[int] = None
    goal_detour: Optional[int] = None
    goal_away_frames: Optional[int] = None
    goal_away_free_frames: Optional[int] = None
    goal_reason_kills: Optional[int] = None


@dataclass
class RunResult:
    """Everything one run produced; written as JSON lines by ``write_jsonl``."""
    route: str
    seed: Optional[int]
    result: str = 'in-progress'     # success, failure, error
    reason: str = ''
    steps: int = 0
    frames: int = 0
    retreats: int = 0               # rooms left through a route step's `retreat` exit (4 end the run: retreat-limit)
    wrong_exits: int = 0            # wrong exits into tagged rooms (the 20th ends the run stuck: wrong-exit-limit)
    clips: Optional[bool] = None    # the run-time clips switch the run started with (round 22; None in older files)
    rooms: List[RoomRecord] = field(default_factory=list)

    @property
    def failed_room(self) -> Optional[RoomRecord]:
        """The room the run ended in, if it didn't succeed."""
        return self.rooms[-1] if self.rooms and self.result != 'success' else None

    @property
    def cheated(self) -> Dict[str, int]:
        """Cheated items used in the run, per item (``_native.CHEAT_ITEMS``): any means it wasn't beaten naturally."""
        return {item: sum(getattr(room, f'cheated_{item}') for room in self.rooms) for item in _native.CHEAT_ITEMS}

    def to_dict(self) -> dict:
        """JSON-friendly dict (a room's ``policy`` only where it is not the family's)."""
        data = asdict(self)
        for room in data['rooms']:
            if room['policy'] is None:
                del room['policy']
        return data

    @staticmethod
    def from_dict(data: dict) -> 'RunResult':
        """Inverse of ``to_dict`` (also reads the native pool's ``run`` events)."""
        rooms = [RoomRecord(**room) for room in data.get('rooms', ())]
        return RunResult(**{**data, 'rooms': rooms})


def write_jsonl(path, results: List[RunResult], append: bool = False):
    """Writes results (one per line) to path: a complete new file (temp file + ``os.replace``), or appended to it
    with ``append``."""
    path = Path(path)
    lines = ''.join(json.dumps(result.to_dict()) + '\n' for result in results)
    if append:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(lines)
        return
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write(lines)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def check_overrides(overrides: Dict[str, str], configs: Optional[Dict[str, SpecialistConfig]] = None,
                    networks: Optional[Dict[str, object]] = None, names: Optional[Iterable[str]] = None,
                    route: Optional[str] = None) -> set:
    """``Policies``' checks of its ``checkpoints`` overrides, loading no model (an entry point makes them before it
    waits its turn): known specialists, a route's part only on its route, a checkpoint file at each path; raises
    ``RunnerError``.  Returns the known specialists (``names``, else every one, with the given networks' and configs')."""
    known = (set(names if names is not None else list_specialists())
             | {split_policy(key)[0] for key in networks or {}} | set(configs or {}))
    unknown = sorted(key for key in overrides if split_policy(key)[0] not in known)
    if unknown:
        raise RunnerError(f"--checkpoint for unknown specialist(s) {unknown}; known: {sorted(known)}")
    parts = own_models_until(route) is not None
    for key in [*overrides, *(networks or {})]:
        part = split_policy(key)[1]
        if part is not None and (part != route or not parts):
            raise RunnerError(f"{key}: a route's part is played only on a route with own_models_until, "
                              f"the route it names (this chain plays {route})")
    for name, path in overrides.items():
        if not Path(path).is_file() and not (Path(path) / CHECKPOINT_NAME).is_file():
            raise RunnerError(f"--checkpoint {name}={path}: no checkpoint file there")
    return known


def chain_config(family: str, route: Optional[str], trained: bool = False) -> SpecialistConfig:
    """The config a chain of ``route`` plays ``family`` with in one slot (``SpecialistConfig.for_chain``, round 40):
    the family's own task (``specialists/<family>.yaml``: every rule of play) with the observation switches (``obs``)
    of the route's task for it (``specialists/<route>/<family>.yaml``) where the route has one; ``trained`` (the
    slot's network reads the generation's switched blocks, ``switched_reads``: it trained on the route's task) with
    every key the chain reads from that task (``env::config::TRAINED_FROM_ROUTE_TASK``: the masks, budgets and
    target switches too), so play matches training; no route: its own task."""
    return load_specialist(family) if route is None else SpecialistConfig.for_chain(family, route, trained)


def switched_reads(network) -> Dict[str, str]:
    """The blocks a task fills only with its ``obs`` switch on (zeros otherwise) that ``network`` reads, block ->
    switch (round 40: ``lab.recipe.SWITCH_KEYS`` and ``network_reads``, the lab's own reading: a ``hitnet``'s token
    groups, also as a ``hitshift``'s hit body; ``ZeldaPolicy`` reads none of them)."""
    from .lab.recipe import SWITCH_KEYS, network_reads  # pylint: disable=import-outside-toplevel
    reads = network_reads(network)
    return {key: switch for switch, keys in SWITCH_KEYS.items() for key in keys if key in reads}


def check_inputs(key: str, reads: Dict[str, str], config: SpecialistConfig, route: Optional[str],
                 given: bool = False):
    """Raises ``RunnerError`` when the network of policy ``key`` reads a block (``reads``: ``switched_reads``) whose
    ``obs`` switch is off in ``config``, the config it is chained with (``given``: a ``--specialist-config``): it
    would read zeros where it trained on the block."""
    off = sorted({switch for switch in reads.values() if not getattr(config.obs, switch)})
    if not off:
        return
    family = split_policy(key)[0]
    routed = route is not None and (SPECIALISTS_DIR / route / f'{family}.yaml').is_file()
    source = ('its --specialist-config' if given else f'specialists/{route}/{family}.yaml' if routed
              else f'specialists/{family}.yaml' + (f' (route {route} has no task of its own for it)' if route else ''))
    blocks = ', '.join(f"{block} (obs.{switch})" for block, switch in sorted(reads.items()) if switch in off)
    raise RunnerError(f"{key}: family {family}'s network reads {blocks}, but "
                      f"{' and '.join(f'obs.{s}' for s in off)} {'is' if len(off) == 1 else 'are'} off in the task "
                      f"it is chained with, {source}: it would read zeros. Chain it on a route whose task for "
                      f"{family} has the switch on, or give it a config that does (--specialist-config)")


class Policies:
    """The specialists' Rust networks for a chain pool, in policy-index order, plus why the others are missing.

    A policy is named by its key (round 32, ``registry.policy_run``): the family, and on a route with
    ``own_models_until`` also ``family@route`` for a family with a run promoted for the route, which plays the
    family's rooms before that step (the engine picks a step's key, ``zelda::GameMap::step_policy``); the family's
    key plays its default run there, or its ``late.<route>`` where it has one (round 38).  On a route without the key a
    family's ``promoted.<route>`` plays every step under the family's key (phase X).

    ``checkpoints`` (key -> checkpoint file or run dir) overrides the registry's run: a family's key both parts, a
    ``family@route`` key the route's part only; ``networks`` (key -> a ``ZeldaPolicy``) supplies a model directly
    (a family's for both parts); ``specialist_configs`` (family -> config) replaces a specialist's config at runtime,
    else each slot plays with ``chain_config``, by the network in it: its family's own task with the route task's
    observation switches, and the route task's rules of play too where the network reads the generation's blocks.  A
    network that reads an observation block its config's switch leaves zero is refused (``check_inputs``).
    ``checkpoints`` afterwards maps each loaded key to its file and ``acts`` holds, per policy, its model's
    act mode (the run's, ``registry.act_mode``; a given network acts by ``registry.DEFAULT_ACT``); ``described``
    holds per key the network as a replay's META describes it (``provenance.replay_models``)."""

    def __init__(self, models_dir=None, checkpoints: Optional[Dict[str, str]] = None,
                 specialist_configs: Optional[Dict[str, SpecialistConfig]] = None,
                 networks: Optional[Dict[str, object]] = None, names: Optional[Iterable[str]] = None,
                 route: Optional[str] = None, headless: Iterable[str] = ()):
        # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
        overrides = dict(checkpoints or {})
        configs = dict(specialist_configs or {})
        given = dict(networks or {})
        known = check_overrides(overrides, configs, given, names, route)
        parts = own_models_until(route) is not None
        self.names: List[str] = []
        self.configs: List[SpecialistConfig] = []
        self.policies: list = []
        self.checkpoints: Dict[str, str] = {}
        self.acts: List[str] = []
        self.missing: Dict[str, str] = {}
        # Per key the network as a replay's META describes it (``provenance.replay_models``: no path).
        self.described: Dict[str, dict] = {}
        # A run several keys name (a multi-family run promoted for several families, triforce.lab) loads once.
        loaded: Dict[str, object] = {}
        described: Dict[str, dict] = {}
        reads: Dict[str, Dict[str, str]] = {}
        for name in sorted(known):
            keys = [name]
            if parts:
                part = route_policy(name, route)
                # A family's override or network plays both parts; else the route's own run, where it has one.
                if part in overrides or part in given or (
                        name not in overrides and name not in given and route_part_key(name, models_dir, route)):
                    keys.append(part)
            for key in keys:
                network = given.get(key)
                act = DEFAULT_ACT
                if network is None:
                    if key in overrides:
                        path = overrides[key]
                    else:
                        try:
                            path = policy_checkpoint(key, models_dir, route)
                        except FileNotFoundError as err:
                            self.missing[key] = str(err)
                            continue
                    real = str(Path(path).resolve() / CHECKPOINT_NAME if Path(path).is_dir() else Path(path).resolve())
                    if real not in loaded:
                        with torch.random.fork_rng():   # building the network draws init weights; keep seeds intact
                            network, saved = load_model(path)
                            loaded[real] = network.to_native()
                            described[real] = checkpoint_description(network, saved, path)
                            reads[real] = switched_reads(network)
                    native = loaded[real]
                    read = reads[real]
                    self.described[key] = described[real]
                    self.checkpoints[key] = str(path)
                    act = act_mode(path)
                else:
                    native = loaded.setdefault(f'network:{id(network)}', network.to_native())
                    read = switched_reads(network)
                    self.described[key] = network_description(network)
                config = configs.get(name) or chain_config(name, route, trained=bool(read))
                check_inputs(key, read, config, route, given=name in configs)
                self.names.append(key)
                self.configs.append(config)
                self.policies.append(native)
                self.acts.append(act)
        self._check_heads(overrides, set(headless), models_dir, route)

    def _check_heads(self, overrides: Dict[str, str], headless: set, models_dir, route: Optional[str]):
        """Round 40 (a void comparison, Nov 23): a ``checkpoints`` override without a hit head that replaces a team
        network with one is refused unless ``headless`` declares its key; a declaration that declares nothing (no such
        override, or no head lost) is refused too.  Raises ``RunnerError``."""
        unknown = sorted(headless - set(overrides))
        if unknown:
            raise RunnerError(f"--headless {unknown}: no --checkpoint override of those keys")
        for key, path in sorted(overrides.items()):
            family, part = split_policy(key)
            replaced = [key] if part is not None else [family] + (
                [route_policy(family, route)] if own_models_until(route) is not None
                and route_part_key(family, models_dir, route) else [])
            with_head = []
            for team_key in replaced:
                try:
                    team = policy_checkpoint(team_key, models_dir, route)
                except FileNotFoundError:
                    continue
                if has_hit_head(load_checkpoint(team, check_schema=False)):
                    with_head.append(f"{team_key} ({checkpoint_label(team)})")
            lost = bool(with_head) and 'hit' not in self.described[key]['outputs']
            if lost and key not in headless:
                raise RunnerError(f"--checkpoint {key}={path} has no hit head and replaces {', '.join(with_head)}, "
                                  f"which has one: declare it with --headless {key}, or play a network with its head")
            if key in headless and not lost:
                raise RunnerError(f"--headless {key}: {path} replaces no network with a hit head that it lacks")

    def modes(self, act: Optional[str] = None) -> List[str]:
        """Per policy the act mode it plays by: its own (``acts``) for None, else ``act`` for every policy."""
        return list(self.acts) if act is None else [parse_act(act)] * len(self.policies)


def check_team(route: Route, policies: Policies):
    """Round 40 (missing is an error): raises ``RunnerError`` when a run of ``route`` would reach a policy key without
    a network (``Policies.missing``: until then only the runs that got there ended ``error``)."""
    until = own_models_until(route.name)
    needed = {}
    for step in route.steps_in_play():
        family = step['specialist']
        part = route_policy(family, route.name) if until is not None else None
        key = part if part is not None and step['index'] < until and part in policies.names else family
        if key in policies.missing:
            needed.setdefault(key, f"step {step['index']}: {policies.missing[key]}")
    if needed:
        raise RunnerError("no network plays policies these runs reach: "
                          + '; '.join(f"{key} ({why})" for key, why in sorted(needed.items())))


def check_configs(route: Route, configs: Dict[str, SpecialistConfig]):
    """Round 40: raises ``RunnerError`` for a specialist config override (``--specialist-config``) of a family whose
    rooms no step of a run of ``route`` plays (it was accepted and unused); loads no model."""
    unused = sorted(set(configs) - {step['specialist'] for step in route.steps_in_play()})
    if unused:
        raise RunnerError(f"--specialist-config for {unused}: no step from step {route.start_step} to this run's end "
                          f"plays those families")


class ChainPool:
    """An ``EnvPool`` of chain runs of one route (one run per seed, shared by ``num_envs`` envs) and the policies
    acting in it.  ``run()`` plays every seed; watchers call ``play()`` and follow ``poll_events()`` themselves.
    ``stall_steps`` (the watch player's simulations; None = off): room events carry ``stalled`` after that many
    actions in a room without damage dealt or taken, and the run ends ``stuck`` at twice as many.  ``record``
    (``replay.record_options``; None = off): every run is recorded
    as a TFREPLAY file and reported as a ``replay`` event; ``replays`` (a ``ReplayDir``) keeps those as files.
    ``act``: None = each policy's act mode (``Policies.acts``), else this act mode for every policy.  ``harvest`` (a
    directory; None = off): every correct room entry where the acting specialist changes (a handoff) saves its
    pre-entry state as
    ``harvest/<specialist>/<level>_<room><side>_h<seed>_<n>.state`` (``n``: the room's index in the run), a start
    state for that specialist's ``harvested_dir``; ``harvest_every_room``: every correct room entry does, also one
    from a room of the same family (the rooms a handoff harvest never sees).  ``hold_finished_frame``: each finished
    run holds the env's picture on its last screen until ``pool.set_hold(env, False)`` (a watched cell's result
    pause).  ``clips_live``:
    ``pool.set_clips_allowed`` also flips the runs in progress, from their next decision (the route re-plans);
    without it the switch reaches only the runs started afterwards."""

    def __init__(self, route: Route, seeds: Sequence[int], policies: Policies, num_envs: int = 1,
                 threads: Optional[int] = None, seed: int = 0, fps: float = 0.0,
                 act: Optional[str] = None,
                 max_steps: int = DEFAULT_MAX_STEPS, states_dir=None, record: Optional[dict] = None,
                 stall_steps: Optional[int] = None, replays: Optional[ReplayDir] = None, harvest=None,
                 hold_finished_frame: bool = False, clips_live: bool = False, harvest_every_room: bool = False):
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        if replays is not None and record is None:
            raise ValueError("a replay folder needs record options (replay.record_options)")
        self.route = route
        self.policies = policies
        self.acts = policies.modes(act)
        self.record = record
        self.replays = replays
        start_from = None if route.start_state == 'start' else route.start_state
        states_dir = None if states_dir is None else str(states_dir)
        task = _native.ChainTask(policies.configs, route.name, [int(s) for s in seeds], stop_at=route.terminal,
                                 start_from=start_from, max_steps=max_steps, missing=policies.missing,
                                 states_dir=states_dir, stall_steps=stall_steps,
                                 harvest=None if harvest is None else str(harvest),
                                 stop_step=route.stop_step, power_on=route.power_on, checkpoint=route.checkpoint,
                                 clips=route.clips, clips_off_from=route.clips_off_from,
                                 hold_finished_frame=hold_finished_frame, clips_live=clips_live,
                                 policies=policies.names, harvest_every_room=harvest_every_room)
        num_envs = max(1, min(num_envs, len(seeds)))
        self.pool = _native.EnvPool(task, num_envs=num_envs, threads=threads, seed=seed, fps=fps, record=record)

    def play(self):
        """Every env starts acting on its own (``EnvPool.play``)."""
        self.pool.play(self.policies.policies, act=self.acts)

    def poll_events(self) -> List[dict]:
        """The pool's events since the last call; with ``replays`` set, every finished run's recording (a
        ``replay`` event) is written to the folder first."""
        events = self.pool.poll_events()
        if self.replays is not None:
            for event in events:
                if event['type'] == 'replay':
                    self.replays.add(int(event['job']), event['data'])
        return events

    @staticmethod
    def results(events: Iterable[dict]) -> List[Tuple[int, RunResult]]:
        """(env, RunResult) of every ``run`` event in ``events``."""
        return [(e['env'], RunResult.from_dict(e['result'])) for e in events if e['type'] == 'run']

    def run(self, on_result: Optional[Callable[[RunResult], None]] = None) -> List[RunResult]:
        """Plays every seed; returns the results sorted by seed."""
        results: List[RunResult] = []
        self.play()
        while True:
            idle = self.pool.wait_idle(timeout=0.1)
            for _, result in self.results(self.poll_events()):
                results.append(result)
                if on_result is not None:
                    on_result(result)
            if idle:
                return sorted(results, key=lambda r: (r.seed is None, r.seed))

    def close(self):
        """Stops the pool's worker threads (idempotent)."""
        self.pool.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class Runner:
    """Plays a route with the promoted (or given) specialists: ``run(route, seed)`` for one run,
    ``run_many(route, seeds)`` for a batch on ``parallel`` envs."""
    # pylint: disable=too-many-instance-attributes

    def __init__(self, models_dir=None, checkpoints: Optional[Dict[str, str]] = None,
                 act: Optional[str] = None,
                 max_steps: int = DEFAULT_MAX_STEPS,
                 specialist_configs: Optional[Dict[str, SpecialistConfig]] = None,
                 networks: Optional[Dict[str, object]] = None, parallel: int = 1, threads: Optional[int] = None,
                 states_dir=None, harvest=None, route: Optional[str] = None, harvest_every_room: bool = False,
                 headless: Iterable[str] = ()):
        """
        checkpoints: specialist name → checkpoint path, overriding the registry's promoted model.
        specialist_configs: specialist name → loaded config, replacing its canonical YAML at runtime.
        networks: specialist name → a ``ZeldaPolicy`` to act with instead of a checkpoint.
        parallel: envs in the chain pool; threads: its worker threads (default min(parallel, cores)).
        act: None = each model's act mode (registry), else this act mode for all.
        harvest: save every handoff's pre-entry state under this directory (``ChainPool``); harvest_every_room: every
        correct room entry's.
        route: the route whose promoted runs play (phase X: ``promoted.<route>`` over ``promoted``; None: ``promoted``).
        headless: the ``checkpoints`` keys declared to play without the hit head of the network they replace
        (``Policies``).
        """
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        self.policies = Policies(models_dir, checkpoints, specialist_configs, networks, route=route, headless=headless)
        self.act = None if act is None else parse_act(act)
        self.max_steps = max_steps
        self.parallel = parallel
        self.threads = threads
        self.states_dir = states_dir
        self.harvest = harvest
        self.harvest_every_room = harvest_every_room

    def pool(self, route: Route, seeds: Sequence[int], **kwargs) -> ChainPool:
        """A chain pool over this runner's policies (``kwargs``: ``num_envs``, ``threads``, ``fps``, ``seed``,
        ``record``, ``replays``, ``stall_steps``)."""
        kwargs.setdefault('num_envs', self.parallel)
        kwargs.setdefault('threads', self.threads)
        return ChainPool(route, seeds, self.policies, act=self.act, max_steps=self.max_steps,
                         states_dir=self.states_dir, harvest=self.harvest,
                         harvest_every_room=self.harvest_every_room, **kwargs)

    def record_meta(self, route: Route, **extra) -> dict:
        """The META object of recordings of ``route`` played by this runner (``chain_meta``)."""
        return chain_meta(self.policies, self.act, route, max_steps=self.max_steps, **extra)

    def provenance(self, route: Route) -> dict:
        """What plays ``route``: per policy its name, checkpoint, the run's own family and run id (the checkpoint's;
        a run linked into another family's folder keeps its own), ``ckpt.pt`` sha1, Rust network hash, act mode and
        config hash; the specialists without a model; the route, start, end and step cap; the code version and the
        game data hash (``replay``).  Stored with every chain result (``chain_summary``, ``eval-chain``'s
        ``<output>.meta.json``)."""
        policies = self.policies
        models = models_header(policies.names, policies.policies, policies.checkpoints, policies.modes(self.act))
        for model, config in zip(models, policies.configs):
            checkpoint = policies.checkpoints.get(model['name'])
            described = policies.described.get(model['name'], {})
            model['family'], model['run'] = described.get('family'), described.get('run')
            model['checkpoint_sha1'] = None if checkpoint is None else checkpoint_sha1(checkpoint)
            model['config_sha256'] = hashlib.sha256(config.to_json().encode()).hexdigest()
        return {**route_header(route), 'max_steps': self.max_steps, 'models': models,
                'missing': dict(sorted(policies.missing.items())), 'version': __version__, 'code': code_version(),
                'data': data_hash()}

    def run_many(self, route: Route, seeds: Sequence[int],
                 on_result: Optional[Callable[[RunResult], None]] = None, record: Optional[dict] = None,
                 replays: Optional[ReplayDir] = None) -> List[RunResult]:
        """Plays route once per seed; results sorted by seed.  Never raises for gameplay failures.  ``record``
        and ``replays``: ``ChainPool``'s (every run recorded into a folder)."""
        seeds = list(seeds)
        if not seeds:
            return []
        with self.pool(route, seeds, record=record, replays=replays) as pool:
            return pool.run(on_result)

    def run(self, route: Route, seed: Optional[int] = None) -> RunResult:
        """Plays route once (seed None: 0)."""
        return self.run_many(route, [0 if seed is None else seed])[0]


def parse_location(text: str) -> MapLocation:
    """'2:0x0e', '0:0x77:cave' → MapLocation; anything else raises ``RunnerError``."""
    parts = text.split(':')
    try:
        if (not 2 <= len(parts) <= 3 or (len(parts) == 3 and parts[2] != 'cave')
                or not parts[1].lower().startswith('0x')):
            raise ValueError(text)
        return MapLocation(int(parts[0]), int(parts[1], 16), len(parts) == 3)
    except ValueError:
        raise RunnerError(f"'{text}' is not a room (LEVEL:0xROOM[:cave], e.g. 1:0x73)") from None


def parse_stop(text: Optional[str]) -> Tuple[Optional[MapLocation], Optional[int]]:
    """``--stop-at``: (room, None) for a room (``1:0x73``), (None, step) for a route step (``step:24``), (None,
    None) without one; anything else raises ``RunnerError``."""
    if not text:
        return None, None
    if text.startswith('step:'):
        if not text[5:].isdecimal():
            raise RunnerError(f"'{text}' is not a route step (step:N, e.g. step:24)")
        return None, int(text[5:])
    return parse_location(text), None


def stop_text(route: 'Route') -> Optional[str]:
    """What ends ``route`` as ``--stop-at`` takes it (``1:0x73``, ``step:24``); None: the route's own end."""
    return location_text(route.terminal) if route.stop_step is None else f'step:{route.stop_step}'


STOP_AT_HELP = ("end the route on entering this room (e.g. 1:0x73), or on reaching this route step by the route's "
                "own exit (step:N, e.g. step:24: back on the overworld with Dungeon 2's triforce)")


def add_power_on_arguments(parser) -> None:
    """``--power-on``, ``--name`` and ``--delay``: a start from console power on (``power_on_options``)."""
    limit = _native.MAX_POWER_ON_DELAY
    parser.add_argument('--power-on', action='store_true',
                        help="start every run from console power on, by controller inputs alone (title screen, "
                             "name entry, game start): no state file and no memory write; the run's randomness is "
                             "--delay")
    parser.add_argument('--name', default=None, metavar='NAME',
                        help="with --power-on: the name to register (1 to 8 of A-Z, 0-9 and space; default LINK; "
                             "ZELDA is refused, it starts the second quest)")
    parser.add_argument('--delay', default=None, metavar='N|random',
                        help=f"with --power-on: idle frames on the file-select screen before the name is entered, "
                             f"which set the game's random numbers: N in 0..={limit} (every value is a different "
                             f"game), or 'random' (default): drawn from that range by each run's seed")


def power_on_options(args) -> Optional[dict]:
    """The ``Route.load`` ``power_on`` of parsed ``add_power_on_arguments`` flags (None without ``--power-on``);
    ``RunnerError`` for ``--name`` or ``--delay`` without it, or a delay that is no frame count."""
    if not args.power_on:
        given = [flag for flag, value in (('--name', args.name), ('--delay', args.delay)) if value is not None]
        if given:
            raise RunnerError(f"{' and '.join(given)} need{'s' if len(given) == 1 else ''} --power-on")
        return None
    delay = None
    if args.delay not in (None, 'random'):
        if not args.delay.isdecimal():
            raise RunnerError(f"--delay {args.delay!r} must be a frame count in 0..={_native.MAX_POWER_ON_DELAY}, "
                              f"or 'random'")
        delay = int(args.delay)
    return _power_on({'name': args.name, 'delay': delay})


def parse_overrides(items: Iterable[str]) -> Dict[str, str]:
    """``SPECIALIST=PATH`` command-line items → {specialist: path}; a malformed item raises ``RunnerError``."""
    overrides = {}
    for item in items:
        name, sep, path = item.partition('=')
        if not sep or not name or not path:
            raise RunnerError(f"--checkpoint '{item}' is not SPECIALIST=PATH")
        overrides[name] = path
    return overrides


def chain_summary(results: List[RunResult]) -> dict:
    """The milestone metric, keys carried at the milestone and a per-room failure breakdown.

    ``arrival_keys`` counts successful runs by the keys Link carried into the milestone room (a ``stop_at`` room,
    or the room of the route's last triforce), so milestone arrivals and full key carriage stay separate
    counters.  ``cheated_<item>`` totals the cheated items used (bombs, keys, rupees) and ``runs_with_cheated_<item>``
    counts the runs that used any."""
    runs = len(results)
    successes = sum(r.result == 'success' for r in results)
    failures: Dict[str, int] = {}
    arrival_keys: Dict[str, int] = {}
    for r in results:
        room = r.failed_room
        if room is not None:
            key = f"{room.level}:{room.room}{':cave' if room.in_cave else ''} {room.specialist} {room.outcome}"
            failures[key] = failures.get(key, 0) + 1
        elif r.result == 'success' and r.rooms:
            keys = r.rooms[-1].keys_in
            label = 'unknown' if keys is None else str(keys)
            arrival_keys[label] = arrival_keys.get(label, 0) + 1
    cheated = [r.cheated for r in results]
    summary = {'runs': runs, 'successes': successes, 'success_rate': successes / runs if runs else 0.0,
               'arrival_keys': dict(sorted(arrival_keys.items())),
               'failures': dict(sorted(failures.items(), key=lambda kv: -kv[1])),
               'errors': sorted({r.reason for r in results if r.result == 'error'})}
    for item in _native.CHEAT_ITEMS:
        summary[f'cheated_{item}'] = sum(c[item] for c in cheated)
        summary[f'runs_with_cheated_{item}'] = sum(c[item] > 0 for c in cheated)
    return summary


class ChainTally:
    """A chain's runs so far, as a person watches them (each ``chain_progress`` event; the training GUI draws it):
    the wins, the deaths, the stuck runs (any other end) and the errors; how many runs took each dungeon's triforce,
    in the order the chain's stretch of ``route`` takes them (a triforce counts once its route step was left by its
    exit; a chain that plays to the route's end counts a win as the end step's level, Dungeon 9 on both routes);
    and the furthest route step a run reached, with the room it stood in there.  ``route`` None: only the endings
    and the furthest step (no route length, no triforces)."""

    def __init__(self, route: Optional[Route] = None):
        self.done = self.wins = self.deaths = self.stucks = self.errors = 0
        self.route_steps = None
        self.triforce_steps: Dict[int, int] = {}
        self.levels: List[int] = []
        self.end_level = None             # the level a win counts (a chain to the route's end step, exit 'end')
        self.furthest: Optional[Tuple[int, str]] = None
        self.taken: Dict[int, int] = {}
        self.span: Optional[Tuple[int, int]] = None      # the route steps the chain plays: from, up to (not incl.)
        if route is None:
            return
        steps = _native.route_steps(route.name)
        end = route.start_step + route.steps_played()
        self.route_steps = len(steps)
        self.span = (route.start_step, end)
        self.triforce_steps = {step['index']: step['level'] for step in steps
                               if step.get('exit') == 'triforce' and route.start_step <= step['index'] < end}
        self.levels = list(dict.fromkeys(level for _, level in sorted(self.triforce_steps.items())))
        if route.stop_step is None and route.terminal is None and steps[-1].get('exit') == 'end':
            self.end_level = steps[-1]['level']
            self.levels = list(dict.fromkeys([*self.levels, self.end_level]))

    def add(self, result: RunResult):
        """Counts one finished run."""
        self.done += 1
        won = result.result == 'success'
        if won:
            self.wins += 1
        elif result.result == 'error':
            self.errors += 1
        elif result.reason.startswith('died in'):
            self.deaths += 1
        else:
            self.stucks += 1
        taken = {self.triforce_steps[room.step] for room in result.rooms
                 if room.step in self.triforce_steps and room.outcome == 'exit-correct'}
        if won and self.end_level is not None:
            taken.add(self.end_level)
        for level in taken:
            self.taken[level] = self.taken.get(level, 0) + 1
        reached = max(((room.step, room) for room in result.rooms if room.step is not None),
                      key=lambda item: item[0], default=None)
        if won and self.end_level is not None:
            reached = (self.route_steps, None)
        if reached is not None and (self.furthest is None or reached[0] > self.furthest[0]):
            room = reached[1]
            where = 'the end' if room is None else f"{room.level}:{room.room}" + (':cave' if room.in_cave else '')
            self.furthest = (reached[0], where)

    def fields(self) -> dict:
        """The tally as ``chain_progress`` carries it beside ``successes`` (the wins): deaths, stucks, errors,
        triforces ([[level, runs], ...] in the route's order), furthest_step and furthest_room (None before a run
        reached a route step), route_steps, and from_step and to_step, the stretch of the route the chain plays (a
        success is a run that reached ``to_step``: the whole game only from step 0 to ``route_steps``)."""
        return {'deaths': self.deaths, 'stucks': self.stucks, 'errors': self.errors,
                'triforces': [[level, self.taken.get(level, 0)] for level in self.levels],
                'furthest_step': None if self.furthest is None else self.furthest[0],
                'furthest_room': None if self.furthest is None else self.furthest[1],
                'route_steps': self.route_steps,
                'from_step': None if self.span is None else self.span[0],
                'to_step': None if self.span is None else self.span[1]}


def route_header(route: Route) -> dict:
    """What a chain plays of ``route`` (``Runner.provenance``, a replay's META ``chain``): its name, start state, power
    on, checkpoint (its state by SHA-1), end, clips."""
    checkpoint = route.checkpoint and {'step': route.checkpoint['step'], 'granted': route.checkpoint['granted'],
                                       'route_granted': route.checkpoint['route_granted'],
                                       'randomize_rng': route.checkpoint['randomize_rng'],
                                       'state_sha1': hashlib.sha1(route.checkpoint['state']).hexdigest()}
    return {'route': route.name, 'start_from': None if route.start_state == 'start' else route.start_state,
            'power_on': route.power_on, 'checkpoint': checkpoint,
            'stop_at': location_text(route.terminal), 'stop_step': route.stop_step, 'clips': route.clips,
            'clips_off_from': (route.clips_off_from if isinstance(route.clips_off_from, (int, type(None)))
                               else location_text(route.clips_off_from))}


def chain_meta(policies: Policies, act: Optional[str], route: Route, max_steps: Optional[int] = None,
               **extra) -> dict:
    """The META object of recordings of ``route`` played by ``policies`` (``replay.meta``; ``act``:
    ``ChainPool``'s; ``extra``: more entries): what ``replay.record_options`` takes.  Its ``chain`` is the chain's
    provenance a reader needs without the repository (``replay.chain_records``): ``route_header`` (a start state by
    name, never a path), the step cap, the families without a network (names only: the reasons name paths), and the
    route's length and triforce steps (``route_steps``, ``triforces``: [[step, level], ...])."""
    models = replay_models(policies.names, policies.policies, policies.modes(act), policies.described)
    for model, config in zip(models, policies.configs):
        model['config_sha256'] = hashlib.sha256(config.to_json().encode()).hexdigest()
    header = route_header(route)
    if header['start_from'] is not None:
        header['start_from'] = Path(header['start_from']).name.removesuffix('.state')
    steps = _native.route_steps(route.name)
    chain = {**header, 'max_steps': max_steps, 'missing': sorted(policies.missing), 'route_steps': len(steps),
             'triforces': [[step['index'], step['level']] for step in steps if step.get('exit') == 'triforce']}
    return replay.meta('chain', models, route.name, stop_text(route), chain=chain, **extra)
