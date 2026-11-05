"""Record Python env parity fixtures on the FCEUmm oracle (plan step 10).  Recording tooling only.

    .venv/bin/python scripts/record_env_parity.py [--rooms 1_44,...] [--parallel 8]

For every room below, a worker process replays a chain movie on ``OracleEmulator`` to the room's entry position:
the last pre-entry frame before Link first becomes controllable in the room (the end of power_on_to_start for
the start room).  At that position it saves the oracle state, loads it with
``GameSession.load(bytes, randomize_rng=False, fresh_room=True)``, starts ``ZeldaEnv(<room's specialist>,
session=..., capture_dir=None, strict_actions=True)`` with ``ram_randomization`` cleared through
``env.restore(session.snapshot(), label)`` and plays up to 200 steps: the room's script in
``scripts/parity_actions.toml`` first (a masked scripted action is replaced by a random legal one), then
``random.Random(1000 + room_index)`` legal actions (tests/test_env.py:random_legal_action).  When an episode ends
before 200 steps and the game is not over, a new episode starts from the current position with ``env.restore``
(``episode_start``); a game over ends the fixture.

Each room is written to ``crates/env/tests/fixtures/parity/<room>.npz`` (``numpy.savez_compressed``).  Arrays:

- ``movie`` (corpus/movies, else target/staging), ``movie_base`` (the movie's ``base`` meta, '' for power-on movies),
  ``entry_frame`` (position = movie inputs run after the base; pokes tagged < entry_frame applied), ``oracle_state`` (the
  FCEUmm state body at the entry position, without OracleEmulator's 8-byte movie header), ``level``, ``room``,
  ``in_cave``, ``specialist``, ``route``, ``route_step`` (-1 = None), ``disabled_kinds``, ``label``.
- ``ram0`` (10,240 bytes: RAM + PRG-RAM after load), ``mask0``, ``obs0_<key>``: the episode's first state.
- per step ``i`` (arrays of length ``steps``): ``actions`` [i,3], ``episode_start``, ``frames``, ``interrupt``
  (index into ``INTERRUPTS``, -1 = none), ``control_frames``, ``blocks``, ``health_lost``, ``health_gained``,
  ``enemies_hit`` [i,12] (damage by slot), ``enemies_stunned`` [i,12] (count by slot), ``items_gained`` [i,8]
  (item ids in order, padded with ``ITEM_PAD``), ``changed_location``, ``gained_triforce``, ``mask`` [i,91]
  (mask after the step), ``ram`` [i,10240] (after the step), ``obs_<key>`` [i,...], ``reward``, ``terminated``,
  ``truncated``, ``ending``/``outcome`` (strings, '' when not terminal), ``info_json`` (terminal info, '' else),
  ``objective_kind``, ``next_rooms`` [i,4,3] ((level, room, cave), padded with -1), ``wavefront`` [i,32,22]
  (PBRS wavefront distance per tile, -1 unreachable / no wavefront), ``linger_tail_steps``, ``linger_max_steps``,
  ``hearts`` [i,5] (dropped, taken, left, expired, healed*2).
- ``block_events`` [n,4]: (step, frame - entry frame, slot, projectile id) for every confirmed shield block.
- reward terms, flattened: ``reward_term_step``, ``reward_term_name``, ``reward_term_value``.
"""

import argparse
import json
import multiprocessing
import os
import random
import sys
import tomllib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oracle_emulator import CORPUS, ROOT, OracleEmulator    # pylint: disable=wrong-import-position
import tfmovie    # pylint: disable=wrong-import-position

STAGING = os.path.join(ROOT, 'target', 'staging')
OUT = os.path.join(ROOT, 'crates', 'env', 'tests', 'fixtures', 'parity')
ACTIONS = os.path.join(ROOT, 'scripts', 'parity_actions.toml')
STEPS = 200
SCRIPTED = 40
ITEM_PAD = -999
INTERRUPTS = ('damage', 'spawn', 'enemy-near', 'transition', 'item', 'door', 'wall', 'game-over')

M1 = 'chain-milestone1-97'          # first milestone1 success in the sweep
FAIRY = 'chain-milestone1-fairy-0'  # first milestone1-fairy success

# (fixture name, level, room, in_cave, entry side, movie, route)
ROOMS = [
    ('0_77_start', 0, 0x77, False, 'start', M1, 'milestone1'),
    ('0_77_cave', 0, 0x77, True, 'S', M1, 'milestone1'),
    ('0_67', 0, 0x67, False, 'S', M1, 'milestone1'),
    ('0_38', 0, 0x38, False, 'S', M1, 'milestone1'),
    ('1_73', 1, 0x73, False, 'S', M1, 'milestone1'),
    ('1_72', 1, 0x72, False, 'E', M1, 'milestone1'),
    ('1_74', 1, 0x74, False, 'W', M1, 'milestone1'),
    ('1_44', 1, 0x44, False, 'W', M1, 'milestone1'),
    ('1_45', 1, 0x45, False, 'W', M1, 'milestone1'),
    ('1_35', 1, 0x35, False, 'S', M1, 'milestone1'),
    ('1_36', 1, 0x36, False, 'W', M1, 'milestone1'),
    ('2_7d', 2, 0x7d, False, 'S', M1, 'milestone1'),
    ('2_5e', 2, 0x5e, False, 'S', M1, 'milestone1'),
    ('0_39_fairy', 0, 0x39, False, 'S', FAIRY, 'milestone1-fairy'),
]


def movie_path(name: str) -> str:
    """Corpus movie, else the staging sweep output."""
    for directory in (os.path.join(CORPUS, 'movies'), STAGING):
        path = os.path.join(directory, f"{name}.tfmovie")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"movie {name} not found in corpus/movies or target/staging")


def random_legal_action(mask_flat, rng):
    """Uniform over legal kinds, then legal direction/duration for that kind (tests/test_env.py)."""
    from triforce.actions import DIRECTIONS, DURATIONS, KINDS, ActionMask    # pylint: disable=import-outside-toplevel
    mask = ActionMask.from_flat(mask_flat)
    kind = rng.choice([i for i in range(len(KINDS)) if mask.kind[i]])
    dirs = [i for i in range(len(DIRECTIONS)) if mask.direction[kind, i]] or [0]
    durs = [i for i in range(len(DURATIONS)) if mask.duration[kind, i]] or [0]
    return np.array([kind, rng.choice(dirs), rng.choice(durs)])


def _replay_to_entry(emu, movie: str, level: int, room: int, in_cave: bool, entry: str) -> bytes:
    """Replays `movie` and returns ``emu.save_state()`` at the room's entry position (emu is left there)."""
    from triforce.room_entry import is_pre_entry    # pylint: disable=import-outside-toplevel
    with open(movie_path(movie), 'rb') as f:
        _, inputs, pokes, meta = tfmovie.decode(f.read())
    if meta.get('base'):
        emu.start_from_base(meta['base'])
    stop = None
    if entry == 'start':
        if meta.get('base'):
            stop = 0
        else:
            with open(os.path.join(CORPUS, 'movies', 'power_on_to_start.tfmovie'), 'rb') as f:
                stop = len(tfmovie.decode(f.read())[1])

    def here():
        mode = emu.read(0x12)
        return emu.read(0x10) == level and emu.read(0xEB) == room and (mode == 0x0B) == in_cave

    last_pre_entry = None
    p = 0
    for k, value in enumerate(inputs):
        if stop is not None and k == stop:
            break
        if stop is None:
            if is_pre_entry(emu.read):
                last_pre_entry = emu.save_state()
            elif here() and emu.read(0x12) in (0x05, 0x0B) and last_pre_entry is not None:
                emu.load_state(last_pre_entry)
                return last_pre_entry
        while p < len(pokes) and pokes[p][0] == k:
            emu.write(pokes[p][1], pokes[p][2])
            p += 1
        emu.run_nes(value)
    if stop is None:
        raise ValueError(f"{movie}: never controllable in {level}:{room:#04x} cave={in_cave}")
    return emu.save_state()


def _change_fields(change):
    hit = np.zeros(12, dtype=np.int64)
    for slot, dmg in change.enemies_hit.items():
        hit[slot] += dmg
    stunned = np.zeros(12, dtype=np.int64)
    for slot in change.enemies_stunned:
        stunned[slot] += 1
    items = np.full(8, ITEM_PAD, dtype=np.int64)
    ids = [int(getattr(i, 'value', i)) for i in change.items_gained]
    assert len(ids) <= 8, ids
    items[:len(ids)] = ids
    return hit, stunned, items


def _objective_fields(state):
    from triforce.room import Room    # pylint: disable=import-outside-toplevel,unused-import
    objectives = state.objectives
    rooms = np.full((4, 3), -1, dtype=np.int64)
    ordered = sorted(objectives.next_rooms, key=lambda r: (r.level, r.value, r.in_cave))
    assert len(ordered) <= 4, ordered
    for i, r in enumerate(ordered):
        rooms[i] = (r.level, r.value, int(r.in_cave))
    grid = np.full((32, 22), -1, dtype=np.int16)
    wavefront = getattr(state, 'pbrs_wavefront', None) if not state.game_over else None
    if wavefront is not None:
        for x in range(32):
            for y in range(22):
                value = wavefront.get((x, y))
                if value is not None:
                    grid[x, y] = value
    return objectives.kind.value, rooms, grid


def record_room(index: int) -> str:
    """Worker: records ROOMS[index] and writes its npz.  Returns a one-line summary."""
    # pylint: disable=too-many-locals,too-many-statements
    import torch    # pylint: disable=import-outside-toplevel
    from triforce.actions import Action    # pylint: disable=import-outside-toplevel
    from triforce.env import ZeldaEnv    # pylint: disable=import-outside-toplevel
    from triforce.objectives import shared_game_map, start_route    # pylint: disable=import-outside-toplevel
    from triforce.session import GameSession    # pylint: disable=import-outside-toplevel
    from triforce.specialist import RamRandomization, load_specialist    # pylint: disable=import-outside-toplevel
    from triforce.zelda_enums import MapLocation    # pylint: disable=import-outside-toplevel

    torch.set_num_threads(1)
    name, level, room, in_cave, entry, movie, route = ROOMS[index]
    with open(ACTIONS, 'rb') as f:
        script = [Action.parse(a) for a in tomllib.load(f)['rooms'].get(name, [])][:SCRIPTED]

    emu = OracleEmulator()
    try:
        saved = _replay_to_entry(emu, movie, level, room, in_cave, entry)
        entry_frame = emu.frame
        session = GameSession(emulator=emu)
        session.load(saved, randomize_rng=False, fresh_room=True)
        location = MapLocation(level, room, in_cave)
        assert session.state.full_location == location, (name, session.state.full_location)
        specialist = shared_game_map().get(location).specialist
        config = load_specialist(specialist).model_copy(update={'ram_randomization': RamRandomization()})
        env = ZeldaEnv(config, session=session, capture_dir=None, strict_actions=True)
        env._route, env._route_step = start_route(location, entry, env.rng, route)    # pylint: disable=protected-access
        label = f"parity-{name}"

        captured = {}
        original_act = session.act

        def act(action):
            captured['change'] = original_act(action)
            return captured['change']

        session.act = act
        obs, _ = env.restore(session.snapshot(), label)
        out = {'movie': np.array(movie), 'movie_base': np.array(emu.base or ''), 'entry_frame': np.int64(entry_frame),
               'oracle_state': np.frombuffer(saved[8:], dtype=np.uint8),
               'level': np.int64(level), 'room': np.int64(room), 'in_cave': np.bool_(in_cave),
               'specialist': np.array(specialist), 'route': np.array(env._route),    # pylint: disable=protected-access
               'route_step': np.int64(-1 if env._route_step is None else env._route_step),    # pylint: disable=protected-access
               'disabled_kinds': np.array(sorted(int(k) for k in config.items.disabled_kinds()), dtype=np.int64),
               'label': np.array(label), 'ram0': emu.ram.copy(), 'mask0': session.mask.to_flat()}
        for key, value in obs.items():
            out[f'obs0_{key}'] = np.asarray(value)

        rng = random.Random(1000 + index)
        rows = {k: [] for k in ('actions', 'episode_start', 'frames', 'interrupt', 'control_frames', 'blocks',
                                'health_lost', 'health_gained', 'enemies_hit', 'enemies_stunned', 'items_gained',
                                'changed_location', 'gained_triforce', 'mask', 'ram', 'reward', 'terminated',
                                'truncated', 'ending', 'outcome', 'info_json', 'objective_kind', 'next_rooms',
                                'wavefront', 'linger_tail_steps', 'linger_max_steps', 'hearts')}
        obs_rows = {}
        block_events, term_step, term_name, term_value = [], [], [], []
        start_next = False
        for step in range(STEPS):
            if start_next:
                env.restore(session.snapshot(), label)
            rows['episode_start'].append(step == 0 or start_next)
            start_next = False
            mask = session.mask
            if step < len(script) and mask.is_allowed(script[step]):
                action = script[step].to_indices()
            else:
                action = random_legal_action(mask.to_flat(), rng)
            obs, reward, terminated, truncated, info = env.step(action)
            change = captured['change']
            hit, stunned, items = _change_fields(change)
            kind, next_rooms, grid = _objective_fields(change.state)
            rows['actions'].append(np.asarray(action, dtype=np.int64))
            rows['frames'].append(change.frames)
            rows['interrupt'].append(-1 if change.interrupt is None else INTERRUPTS.index(change.interrupt))
            rows['control_frames'].append(change.option.control_frames)
            rows['blocks'].append(change.blocks)
            rows['health_lost'].append(float(change.health_lost))
            rows['health_gained'].append(float(change.health_gained))
            rows['enemies_hit'].append(hit)
            rows['enemies_stunned'].append(stunned)
            rows['items_gained'].append(items)
            rows['changed_location'].append(bool(change.changed_location))
            rows['gained_triforce'].append(bool(change.gained_triforce))
            rows['mask'].append(session.mask.to_flat())
            rows['ram'].append(emu.ram.copy())
            rows['reward'].append(float(reward))
            rows['terminated'].append(bool(terminated))
            rows['truncated'].append(bool(truncated))
            rows['ending'].append(info.get('ending') or '')
            rows['outcome'].append(info.get('outcome') or '')
            done = terminated or truncated
            rows['info_json'].append(json.dumps(info, sort_keys=True, default=str) if done else '')
            rows['objective_kind'].append(kind)
            rows['next_rooms'].append(next_rooms)
            rows['wavefront'].append(grid)
            rows['linger_tail_steps'].append(env.linger.tail_steps)
            rows['linger_max_steps'].append(env.linger.max_steps)
            h = env.hearts.summary()
            rows['hearts'].append(np.array([h['dropped'], h['taken'], h['left'], h['expired'],
                                            round(h['healed'] * 2)], dtype=np.int64))
            for block in change.projectile_blocks:
                block_events.append((step, block.frame - entry_frame, block.slot, block.projectile_id))
            for term, value in info['rewards'].items():
                term_step.append(step)
                term_name.append(term)
                term_value.append(float(value))
            for key, value in obs.items():
                obs_rows.setdefault(key, []).append(np.asarray(value))
            if change.state.game_over:
                break
            start_next = done

        for key, values in rows.items():
            out[key] = np.array(values)
        for key, values in obs_rows.items():
            out[f'obs_{key}'] = np.stack(values)
        out['wavefront'] = out['wavefront'].astype(np.int16)
        out['block_events'] = np.array(block_events, dtype=np.int64).reshape(-1, 4)
        out['reward_term_step'] = np.array(term_step, dtype=np.int64)
        out['reward_term_name'] = np.array(term_name)
        out['reward_term_value'] = np.array(term_value, dtype=np.float64)
        out['interrupts'] = np.array(INTERRUPTS)
        os.makedirs(OUT, exist_ok=True)
        np.savez_compressed(os.path.join(OUT, f"{name}.npz"), **out)
        steps = len(rows['frames'])
        endings = [e for e in rows['ending'] if e]
        return f"{name}: entry {entry_frame}, {steps} steps, {sum(rows['frames'])} frames, " \
               f"{len(block_events)} blocks, endings {endings}"
    finally:
        emu.close()


def main(argv=None):
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--rooms', default=None, help='comma-separated fixture names (default: all)')
    parser.add_argument('--parallel', type=int, default=4)
    args = parser.parse_args(argv)
    names = [r[0] for r in ROOMS]
    wanted = args.rooms.split(',') if args.rooms else names
    indices = [names.index(n) for n in wanted]
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(args.parallel, maxtasksperchild=1) as pool:
        for line in pool.imap_unordered(record_room, indices):
            print(line, flush=True)


if __name__ == '__main__':
    main()
