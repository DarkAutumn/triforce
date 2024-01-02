# a test script to print rewards to test scoring
import zelda
import importlib
import mesen

from zelda import zelda_memory_layout, ZeldaGameState, ZeldaScoreBasic
from mesen_zelda import MesenZeldaRecorder
 
# Reload so that if I make live changes to zelda.py they are reflected in Mesen
importlib.reload(zelda)

class PrintRewards:
    def __init__(self):
        self.recorder = MesenZeldaRecorder()
        self.rewards = zelda.ZeldaScoreBasic()
         
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