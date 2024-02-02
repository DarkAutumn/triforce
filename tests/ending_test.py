import os
import sys
from triforce_lib import ZeldaGameplayCritic
from triforce_lib.scenario_dungeon_combat import ZeldaDungeonCombatEndCondition
from utilities import CriticWrapper, RewardRecorder

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from triforce_lib import ZeldaActionReplay

def test_dungeon_combat_end():
    recorder = RewardRecorder()

    def wrapper(env):
        return CriticWrapper(env, end_conditions=[ZeldaDungeonCombatEndCondition(recorder)])

    replay = ZeldaActionReplay("1_44e.state", wrapper)
    
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