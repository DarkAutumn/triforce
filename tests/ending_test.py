import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from triforce_lib import ZeldaActionReplay
from triforce_lib.scenario_dungeon_combat import ZeldaDungeonCombatEndCondition
from utilities import CriticWrapper

def test_dungeon_combat_end():
    replay = ZeldaActionReplay("1_44e.state")
    replay.env = CriticWrapper(replay.env, end_conditions=[ZeldaDungeonCombatEndCondition()])
    replay.reset()
    
    replay.run_steps('llluuuullllllllllllllld')

    _, _, terminated, truncated, info = replay.step('b')
    assert not terminated
    assert not truncated
    assert info['beam_hits'] == 0
    assert info['step_hits'] == 3
    assert info['action'] == 'item'

    found = None
    for i in range(100):
        _, _, terminated, truncated, info = replay.step('r')
        if terminated:
            found = i
            break
    
    assert found is not None