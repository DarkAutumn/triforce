# a test script to print rewards to test scoring
from zelda import LegendOfZeldaAgent, ZeldaMemoryLayout, ZeldaMemoryWrapper, ZeldaGameStates
import importlib
import random
import emu
 
# Reload so that if I make live changes to zelda.py they are reflected in Mesen
importlib.reload(zelda)

iterations = 5
max_game_duration_sec = 1 * 60
frames_per_second = 60.1
save_every = 50

# after every action the agent takes, skip some frames so that we don't act on every frame
# it's important this be a range and not, say, every 2 frames because certain enemy animations
# are on a particular cycle, and would therefore be "invisible" to the agent without some
# variability
action_frame_skip_min = 5
action_frame_skip_max = 15

start_button_input = [False, False, False, False, False, True, False, False]
no_button_input = [False, False, False, False, False, False, False, False]

class LoadSaveStateAndSkipTitle:
    def __init__(self, old_on_frame, save_state, zeldaMemory : ZeldaMemoryWrapper):
        self.old_on_frame = old_on_frame
        self.zeldaMemory = zeldaMemory
        
        emu.removeEventCallback(self, old_on_frame)
        emu.addEventCallback(self.onFrame)

        self.state = 0  # loaded save
        emu.load_save_state(save_state)

        # skip a random number of frames to make sure RNG changes
        self.frame_skip = random.randint(0, 75)

    def onFrame(self, cpuType):
        if self.skip_frames:
            self.skip_frames -= 1
            return
        
        mem = self.zeldaMemory.snapshot()
        if mem.game_state == ZeldaGameStates.gameplay:
            # we are done!
            emu.removeEventCallback(self, self.onFrame)
            emu.addEventCallback(self.old_on_frame)
            self.old_on_frame(cpuType)

        if self.state == 0:
            # We are now at the title screen
            emu.setInput(start_button_input)
            self.state = 1
        
        else:
            emu.setInput(no_button_input)
            self.state = 0

class TrainAgent:
    def __init__(self):
        memory = emu.registerFrameMemory(7, ZeldaMemoryLayout.get_address_list())
        screen = emu.registerScreenMemory()
        self.agent = LegendOfZeldaAgent(memory, screen)
        self.total_iterations = iterations
        self.is_running = False
        self.current_iteration = 0

        self.current_frame = 0
        self.max_frame = max_game_duration_sec / frames_per_second

        self.skip_frames = 0
        self.current_input = None

    def onFrame(self, cpuType):
        if self.skip_frames:
            if self.current_input:
                emu.setInput(self.current_input)
            
            self.skip_frames -= 1
            if self.skip_frames == 0:
                self.current_input = None

            return

        try:
            if not self.is_running:
                # check if we are done
                if self.current_iteration >= self.total_iterations:
                    emu.removeEventCallback(self.onFrame, self.eventType.startFrame)
                    self.agent.save("completed.dat")
                    print("Complete!")
                    return
                
                LoadSaveStateAndSkipTitle(self.onFrame, "X:\\start.mss", self.agent.memory)
                self.is_running = True
                self.agent.begin_game()
                return

            gameplay_state = self.agent.capture_and_check_game_state()

            if gameplay_state == ZeldaGameStates.gameplay_animation_lock:
                # take no action if there's animation lock
                pass
            
            elif gameplay_state == ZeldaGameStates.game_over:
                # finish the iteration
                self.agent.end_game()
                self.is_running = False
            else:
                # our action is the controller input
                action = self.get_action_from_game_state()
                emu.setInput(action)

                # skip frames for the next action:
                if action_frame_skip_min < action_frame_skip_max:
                    self.current_input = action
                    self.skip_frames = random.randint(action_frame_skip_min, action_frame_skip_max)

        except Exception as e:
            print(e)
            import traceback
            print(traceback.format_exc())
            
            emu.removeEventCallback(self.onFrame, self.eventType.startFrame)

emu.addEventCallback(TrainAgent().onFrame, eventType.startFrame)
