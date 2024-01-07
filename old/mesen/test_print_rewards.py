# a test script to print rewards to test scoring
import zeldaml
import importlib
import mesen

from zeldaml import zelda_memory_layout, ZeldaGameState, ZeldaScoreBasic
from mesen_zelda import MesenZeldaRecorder
 
class PrintRewards:
    def __init__(self):
        self.recorder = MesenZeldaRecorder()
        self.rewards = zeldaml.ZeldaScoreDungeon()
         
    def onFrame(self, cpuType):
        try:
            frame = self.recorder.capture()
            reward = self.rewards.score(frame.game_state)
            if reward != 0:
                print(f"reward: {reward}")
        except Exception as e:
            print(e)
            import traceback
            print(traceback.format_exc())
            
            mesen.removeEventCallback(onFrame, mesen.eventType.startFrame)


rewards = PrintRewards()
def onFrame(cpuType):
    rewards.onFrame(cpuType)

mesen.addEventCallback(onFrame, mesen.eventType.startFrame)