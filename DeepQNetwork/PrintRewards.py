# a test script to print rewards to test scoring
import emu
import zelda
import importlib
importlib.reload(zelda)

last_hearts = 0.0
def onFrame(cpuType):
    try:
        global last_snapshot
        global last_hearts
        curr_snapshot = zelda_memory.snapshot()
        
        if curr_snapshot.hearts != last_hearts:
            print(curr_snapshot.hearts)
            last_hearts = curr_snapshot.hearts

        if last_snapshot.hearts != curr_snapshot.hearts:
            print("found!")

        if last_snapshot is not None:
            reward = rewards.score(last_snapshot, curr_snapshot)
            if reward > 0.1:
                print(f"reward: {reward}")
         
        last_snapshot = curr_snapshot
    except Exception as e:
        print(e)
        import traceback
        print(traceback.format_exc())
        
        emu.removeEventCallback(onFrame, eventType.startFrame)

addr = emu.registerFrameMemory(7, zelda.memoryLayout.get_memory_list())
zelda_memory = zelda.MemoryWrapper(addr)
last_snapshot = zelda_memory.snapshot()
rewards = zelda.LegendOfZeldaScorer()
print("registered frame mem")

emu.addEventCallback(onFrame, eventType.startFrame)

class TestClass:
	def onFrame(self, count):
		print("here!")
		
emu.addEventCallback(TestClass().onFrame, eventType.startFrame)
		