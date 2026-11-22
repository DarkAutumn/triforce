"""Round 38, the aimless-walking score in the emulator (``zelda::aimless``; docs/architecture.md "Aimless walking"):
a scripted shortest walk (each MOVE the ``target`` observation's first step) scores about nothing, the same walk with
a detour scores what the detour wasted, and nothing is calm in a room whose monsters live. (A vanishing heart that
pulls Link is no calm goal: ``env::objective``'s ``a_heart_that_can_vanish_is_no_calm_goal``; the arithmetic:
``zelda::aimless``'s unit tests.)"""

from triforce import _native
from triforce.specialist import load_specialist

from .conftest import REPO_STATES_DIR, require_state

DIRECTIONS = 'NSWE'
CAVE = 'd3first_0_77cave_0'         # the opening's sword cave, entered from 0:0x77: the old man, the fires, the sword


def _env(state: str, family: str):
    require_state(state)
    env = _native.Env(load_specialist(family), start_state=state, states_dir=str(REPO_STATES_DIR),
                      track_visits=True)
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
