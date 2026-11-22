"""Round 38, the aimless-walking score in the emulator (``zelda::aimless``; docs/architecture.md "Aimless walking"):
a scripted shortest walk (each MOVE the ``target`` observation's first step) scores about nothing, the same walk with
a detour scores what the detour wasted, and nothing is calm in a room whose monsters live. ``penalty-aimless``
(``reward.aimless_penalty``) charges each calm decision its rise of the stretch's waste, per frame and scaled like
the shaping terms, and nothing while a monster lives. (A vanishing heart that pulls Link is no calm goal:
``env::objective``'s ``a_heart_that_can_vanish_is_no_calm_goal``; the arithmetic: ``zelda::aimless``'s unit tests.)"""

import json

import pytest

from triforce import _native
from triforce.runner import RunResult
from triforce.specialist import load_specialist

from .conftest import REPO_STATES_DIR, require_state

DIRECTIONS = 'NSWE'
CAVE = 'd3first_0_77cave_0'         # the opening's sword cave, entered from 0:0x77: the old man, the fires, the sword
FRAMES_PER_TILE = 6                 # zelda::aimless::FRAMES_PER_TILE


def _env(state: str, family: str, **reward):
    """An env on ``state`` for ``family``, its ``reward`` settings updated with ``reward``."""
    require_state(state)
    config = load_specialist(family)
    if reward:
        fields = config.to_dict()
        fields['reward'].update(reward)
        config = _native.SpecialistConfig.from_json(json.dumps(fields))
    env = _native.Env(config, start_state=state, states_dir=str(REPO_STATES_DIR), track_visits=True)
    obs, _ = env.reset(seed=0)
    return env, obs


def _shortest(obs) -> str:
    """The ``target`` observation's first step down the objective's wavefront, as a MOVE short enough not to walk
    past a turn (16 frames far from the goal, 8 within 4 tiles, 4 within 2)."""
    target = obs['target']
    assert target[:4].max() > 0, 'the wavefront reaches Link'
    tiles = float(target[4]) * 32
    return f"MOVE {DIRECTIONS[int(target[:4].argmax())]} {4 if tiles <= 2 else 8 if tiles <= 4 else 16}"


def _play(env, obs, moves: list[str]) -> list[dict]:
    """Plays ``moves`` (a direction, a 16-frame MOVE; ``None``: the shortest walk's next step), then the shortest
    walk until Link leaves the room; the finished visits."""
    room = env.state.full_location
    for move in [*moves, *[None] * 40]:
        obs, _, terminated, truncated, _ = env.step(f'MOVE {move} 16' if move else _shortest(obs))
        if terminated or truncated or env.state.full_location != room:
            return env.take_visits()
    raise AssertionError('Link never left the room')


def test_a_shortest_walk_in_the_sword_cave_scores_nothing():
    env, obs = _env(CAVE, 'ow-start')
    assert env.aimless['calm'], 'the cave is calm: the old man, his fires, the sword'
    visit, = _play(env, obs, [])
    # Two stretches: to the sword (its pickup is the goal, the fanfare frozen), then out of the cave.
    assert visit['calm_stretches'] == 2 and visit['calm_decisions'] > 0
    assert (visit['aimless_frames'], visit['calm_reversals'], visit['calm_away']) == (0, 0, 0)
    assert abs(visit['calm_frames'] - visit['calm_needed_frames']) <= 16


def test_a_walk_away_and_back_scores_what_it_wasted():
    env, obs = _env(CAVE, 'ow-start')
    # Into the cave, then two MOVEs east and two back west before the walk to the sword.
    visit, = _play(env, obs, [None, None, 'E', 'E', 'W', 'W'])
    straight, = _play(*_env(CAVE, 'ow-start'), [])
    assert visit['calm_reversals'] >= 1 and visit['calm_away'] >= 1
    wasted = visit['calm_frames'] - straight['calm_frames']
    assert wasted >= 4 * 16
    # The detour's frames, less the stretch's allowance and the straight walk's slack (it needed a few frames more
    # than it took).
    assert wasted - 2 * 16 <= visit['aimless_frames'] <= wasted


def test_nothing_is_calm_while_a_monster_lives():
    env, obs = _env('d3first_8_4cc_0', 'pols-voice')      # Dungeon 8's 0x4c from its cellar: the Pols Voices
    for _ in range(12):
        assert env.state.enemies, 'the room is still fought'
        obs, _, terminated, truncated, _ = env.step('WAIT 8')
        assert not terminated and not truncated
        assert not env.aimless['calm']
    assert env.aimless['calm_decisions'] == 0 and env.aimless['aimless_frames'] == 0


# Into the cave, two MOVEs east and two back west, then on toward the sword: one stretch in the open.
DETOUR = ['N', 'N', 'E', 'E', 'W', 'W', 'N']


@pytest.mark.parametrize('penalty, scale', [(0.01, 1.0), (0.02, 0.5)])
def test_penalty_aimless_charges_each_calm_decision_its_rise_of_waste(penalty, scale):
    env, obs = _env(CAVE, 'ow-start', aimless_penalty=penalty, shaping_scale=scale)
    start, waste, charged = None, 0, []
    for move in DETOUR:
        assert env.aimless['calm'], 'the cave stays calm'
        before = round(float(obs['target'][4]) * 32)
        start = before if start is None else start
        obs, _, terminated, truncated, info = env.step(f'MOVE {move} 16')
        assert not terminated and not truncated
        # The stretch so far: lasted (its calm frames) less the walk its ground needed, never below 0.
        gained = start - round(float(obs['target'][4]) * 32)
        now = max(0, env.aimless['calm_frames'] - gained * FRAMES_PER_TILE)
        expected = penalty * scale * max(0, now - waste)
        waste = now
        paid = -info['rewards'].get('penalty-aimless', 0.0)
        assert paid == pytest.approx(expected, rel=1e-4, abs=1e-6), (move, info['rewards'])
        assert ('penalty-aimless' in info['rewards']) == (expected > 0), 'no term on a decision that wasted nothing'
        charged.append(paid)
    assert sum(charged) >= penalty * scale * 4 * 16, 'the detour paid for its four MOVEs at least'
    # Off by default: the same walk carries no term.
    env, _ = _env(CAVE, 'ow-start')
    for move in DETOUR:
        assert 'penalty-aimless' not in env.step(f'MOVE {move} 16')[4]['rewards']


def test_penalty_aimless_charges_nothing_while_a_monster_lives():
    env, _ = _env('d3first_8_4cc_0', 'pols-voice', aimless_penalty=1.0)
    for move in ['WAIT 8', 'MOVE W 8', 'MOVE E 8'] * 4:        # up the cellar's stairs: S is masked
        assert env.state.enemies, 'the room is still fought'
        _, _, terminated, truncated, info = env.step(move)
        assert not terminated and not truncated
        assert 'penalty-aimless' not in info['rewards']


def test_a_chain_room_record_reads_the_fields():
    """``RunResult.from_dict`` reads a chain's room records (``eval-chain`` failed on every run without them)."""
    room = {'index': 0, 'level': 0, 'room': '0x77', 'in_cave': False, 'specialist': 'ow-start',
            **{name: i + 1 for i, name in enumerate(_native.AIMLESS_FIELDS)}}
    result = RunResult.from_dict({'route': 'd3first', 'seed': 0, 'result': 'success', 'reason': '', 'steps': 1,
                                  'frames': 1, 'rooms': [room]})
    assert {name: getattr(result.rooms[0], name) for name in _native.AIMLESS_FIELDS} == {
        name: room[name] for name in _native.AIMLESS_FIELDS}
