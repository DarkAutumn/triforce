# a test script to print rewards to test scoring
import emu
import zelda
import importlib

# Reload so that if I make live changes to zelda.py they are reflected in Mesen
importlib.reload(zelda)

class PrintRewards:
    def __init__(self):
        addr = emu.registerFrameMemory(7, zelda.memoryLayout.get_memory_list())
        self.zelda_memory = zelda.MemoryWrapper(addr)
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
            
            emu.removeEventCallback(self.onFrame, self.eventType.startFrame)


emu.addEventCallback(PrintRewards().onFrame, eventType.startFrame)
