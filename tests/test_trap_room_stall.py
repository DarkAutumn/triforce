"""trap-room@d3first pays for a stall (Rooms49, Nov 26): 4:0x02's six traps kill a Link who dithers, and with the
600-step truncation bootstrapped the rooms teacher gen1-rooms/s1r3 learned to stand in the W doorway instead (0.9% in
training, 59 of 100 D4 runs stuck at the chain's 2000 steps). The task ends the no-damage and max-steps truncations as
hard failures (boss-gleeok's rule): terminated, with the terminal failure penalty, no bootstrap."""

import pytest

from triforce import _native
from triforce.lab.collect import task_yaml

from .conftest import REPO_STATES_DIR, require_state

STALLS = ('no-damage', 'max-steps')


def test_the_task_names_both_stall_truncations_as_hard_failures():
    task = _native.SpecialistConfig.load(task_yaml('trap-room@d3first')).to_dict()
    assert {ending: task['endings'].get(ending) for ending in STALLS} == dict.fromkeys(STALLS, 'hard-failure')
    assert task['no_damage_steps'] == 200       # a stall ends after 200 quiet actions, not the 600 of max_steps


@pytest.mark.parametrize('limit', ['no_damage_steps', 'max_steps'])
def test_standing_in_4x02s_doorway_ends_as_a_paid_failure(limit):
    require_state('d3first_4_02w_0')
    config = _native.SpecialistConfig.load(task_yaml('trap-room@d3first'))
    # The task's endings as they are, with the cap the test reaches short (the other one out of the way).
    config = config.replace(**{'max_steps': 600, 'no_damage_steps': 600, limit: 12})
    env = _native.Env(config, start_state='d3first_4_02w_0', states_dir=str(REPO_STATES_DIR), seed=0)
    env.reset(seed=3)
    for _ in range(12):
        _, reward, terminated, truncated, info = env.step('WAIT 4')
        if terminated or truncated:
            break
    assert (terminated, truncated) == (True, False)
    assert info['ending'] == ('no-damage' if limit == 'no_damage_steps' else 'max-steps')
    assert info['outcome'] == 'hard-failure'
    assert info['rewards']['penalty-terminal-failure'] < 0 and reward < 0
