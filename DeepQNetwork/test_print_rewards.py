# a test script to print rewards to test scoring
import zelda
import importlib
import mesen

from zelda.zelda_constants import zelda_memory_layout
 
# Reload so that if I make live changes to zelda.py they are reflected in Mesen
importlib.reload(zelda)

class PrintRewards:
    def __init__(self):
        addr = mesen.registerFrameMemory(7, zelda.zelda_memory_layout.get_address_list())
        self.zelda_memory = zelda_memory_layout(addr)
        self.last_snapshot = self.zelda_memory.snapshot()
        self.rewards = zelda.LegendOfZeldaScorer()
         
    def onFrame(self, cpuType):
        try:
            curr_snapshot = self.zelda_memory.snapshot()
            reward = self.rewards.score(self.last_snapshot, curr_snapshot)
            if reward > 0.1:
                print(f"reward: {reward}")
        
            self.last_snapshot = curr_snapshot
        except Exception as e:
            print(e)
            import traceback
            print(traceback.format_exc())
            
            mesen.removeEventCallback(self.onFrame, mesen.eventType.startFrame)


mesen.addEventCallback(PrintRewards().onFrame, mesen.eventType.startFrame)
