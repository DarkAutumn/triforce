"""A Like-Like's hold (Wizz50, Nov 27): while one holds Link (asm:IsLinkParalyzed, ``$512``) the game discards every
direction and no shot or touch reaches him. gen1-others/s1r4 learned to sit there: it pressed MOVE (offered, two
frames, nothing moved) for the chain's 2000 steps in 39 of 100 Dungeon 6 runs, while the 600-step no-damage
truncation was bootstrapped (free) in training. The mask offers no MOVE in a hold (the strikes in Link's facing stay
and free him), and wizzrobe@d3first ends a stall as a paid hard failure (trap-room's rule)."""

import numpy as np
import pytest

from triforce import _native
from triforce.actions import DIRECTIONS, KINDS
from triforce.lab.collect import task_yaml

from .conftest import REPO_STATES_DIR, require_state

STATE = 'd3first_6_38s_0'          # 6:0x38 from S: a Bubble, three Like-Likes, two Blue and two Red Wizzrobes
HELD = 0x512                        # asm:IsLinkParalyzed
STALLS = ('no-damage', 'max-steps')


def env_for(config=None, seed=1):
    require_state(STATE)
    config = config or _native.SpecialistConfig.load(task_yaml('wizzrobe@d3first'))
    env = _native.Env(config, start_state=STATE, states_dir=str(REPO_STATES_DIR), seed=0, strict_actions=False)
    env.reset(seed=seed)
    return env


def walk_into_a_like_like(env, limit=40):
    """Steps toward the nearest Like-Like until one holds Link (seeds 1-5 are held within 8-20 steps)."""
    for _ in range(limit):
        state = env.state
        if state.read(HELD):
            return
        like = min((e for e in state.enemies if e['name'] == 'LikeLike'), key=lambda e: e['distance'])
        dx, dy = like['x'] - state.link['x'], like['y'] - state.link['y']
        env.step(f"MOVE {('E' if dx > 0 else 'W') if abs(dx) > abs(dy) else ('S' if dy > 0 else 'N')} 4")
    pytest.fail(f'no Like-Like held Link within {limit} steps')


def offered(env, kind) -> list[str]:
    mask = np.asarray(env.mask)
    k = KINDS.index(kind)
    if not mask[k]:
        return []
    base = len(KINDS) + k * len(DIRECTIONS)
    return [d.name for i, d in enumerate(DIRECTIONS) if mask[base + i]]


@pytest.mark.parametrize('seed', [1, 2, 4])
def test_no_move_is_offered_in_a_hold_and_the_sword_frees_link(seed):
    env = env_for(seed=seed)
    walk_into_a_like_like(env)
    facing = env.state.link['direction']
    assert offered(env, KINDS[0]) == []                        # MOVE: the game discards every direction
    here, at = env.snapshot(), (env.state.link['x'], env.state.link['y'])
    for d in 'NSWE':                                           # what the mask no longer offers moves nothing
        env.restore(here, STATE)
        env.step(f'MOVE {d} 4')
        assert (env.state.link['x'], env.state.link['y']) == at and env.state.read(HELD)
    env.restore(here, STATE)
    assert offered(env, KINDS[2]) == [facing]                  # SWORD in the facing alone
    for _ in range(16):                                        # three magic-sword hits a Like-Like, one every 2 swings
        if not env.state.read(HELD):
            break
        _, _, terminated, truncated, _ = env.step(f'SWORD {facing}')
        assert not (terminated or truncated)
    assert not env.state.read(HELD)
    assert offered(env, KINDS[0])                              # free: MOVE is back


def test_the_task_names_both_stall_truncations_as_hard_failures():
    task = _native.SpecialistConfig.load(task_yaml('wizzrobe@d3first')).to_dict()
    assert {ending: task['endings'].get(ending) for ending in STALLS} == dict.fromkeys(STALLS, 'hard-failure')
    assert task['no_damage_steps'] == 200


def test_waiting_in_a_hold_ends_as_a_paid_failure():
    config = _native.SpecialistConfig.load(task_yaml('wizzrobe@d3first')).replace(no_damage_steps=30)
    env = env_for(config)
    walk_into_a_like_like(env)                                # 8-20 quiet steps, then the capture's hit
    for _ in range(30):
        _, reward, terminated, truncated, info = env.step('WAIT 4')
        if terminated or truncated:
            break
    assert env.state.read(HELD)
    assert (terminated, truncated) == (True, False)
    assert (info['ending'], info['outcome']) == ('no-damage', 'hard-failure')
    assert info['rewards']['penalty-terminal-failure'] < 0 and reward < 0
