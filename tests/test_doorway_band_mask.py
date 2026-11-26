"""The MOVE mask in a dungeon doorway's outer border band (Rooms49, Nov 26): asm:Link_FilterInput keeps only the input
on Link's facing axis there (asm:Link_MaskInputInBorder), so a press across it moves nothing, on the grid or off it.
Before the fix ``can_link_move`` offered every direction once Link was off the grid: one step into 1:0x22's E doorway
MOVE S was legal, moved nothing (interrupt ``wall``) and the wavefront target pointed S there, and the rooms teacher
gen1-rooms/s1r3 pressed it until the chain's 2000 steps ran out (14 of 100 D1 runs)."""

import numpy as np
import pytest

from triforce import _native
from triforce.actions import DIRECTIONS, KINDS
from triforce.lab.collect import task_yaml

from .conftest import REPO_STATES_DIR, require_state

MOVE_DIRS = len(KINDS)      # the MOVE row of the direction block starts after the kind block
DIRS = ('N', 'S', 'W', 'E')
# A start in each kind of doorway the route enters (its side; Link walks into the room the other way).
DOORWAYS = {'d3first_1_22e_0': 'W', 'd3first_4_02w_0': 'E', 'd3first_6_48s_0': 'N'}


def offered_moves(env) -> list[str]:
    mask = np.asarray(env.mask)
    return [DIRECTIONS[d].name for d in range(4) if mask[MOVE_DIRS + d]]


def position(env) -> tuple:
    link = env.state.link
    return link['x'], link['y']


@pytest.mark.parametrize('state', DOORWAYS)
def test_every_move_offered_in_a_doorway_band_moves_link(state):
    require_state(state)
    inward = DOORWAYS[state]
    config = _native.SpecialistConfig.load(task_yaml('trap-room@d3first'))
    env = _native.Env(config, start_state=state, states_dir=str(REPO_STATES_DIR), seed=0, strict_actions=False)
    env.reset(seed=1)
    across = [d for d in DIRS if d not in (inward, {'N': 'S', 'S': 'N', 'W': 'E', 'E': 'W'}[inward])]
    checked = 0
    for _ in range(6):                               # 1 then 4-6 pixels a step: through the band and out of it
        here, at = env.snapshot(), position(env)
        for d in offered_moves(env):
            env.restore(here, state)
            _, _, _, _, info = env.step(f'MOVE {d} 4')
            assert position(env) != at or info.get('interrupt') != 'wall', \
                f"{state} at {at}: MOVE {d} is offered and the game drops it"
            checked += d in across
        env.restore(here, state)
        env.step(f'MOVE {inward} 4')
    assert checked > 0                               # the walk left the band and offered a sideways move
