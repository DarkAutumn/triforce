# a test script to print rewards to test scoring
import zelda
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

# input order: a b start select up down left right, use None to not set a value
start_button_input = [False, False, True, False, False, False, False, False]
no_button_input = [False, False, False, False, False, False, False, False]


class TrainAgent:
    def __init__(self):
        memory = emu.registerFrameMemory(7, zelda.ZeldaMemoryLayout.get_address_list())
        screen = emu.registerScreenMemory()
        self.agent = zelda.LegendOfZeldaAgent(memory, screen)
        self.total_iterations = iterations

        self.current_iteration = 0

        self.max_frame = max_game_duration_sec / frames_per_second

        self.skip_frames = 0
        self.current_input = None

    def start(self):
            emu.loadSaveState("x:\\start.mss")
            self.is_running = True
            self.agent.begin_game()
            self.enable()

    def onPollInput(self, _):
        if self.current_input:
            emu.setInput(0, 0, self.current_input)

    def onFrame(self, _):
        try:
            # alwawys check the gameplay state, this captures input/screen
            gameplay_state = self.agent.capture_and_check_game_state()

            if gameplay_state == zelda.ZeldaGameStates.gameplay_animation_lock:
                # if there's animation lock, it's fast, and we want the AI to be able to respond
                # when it lifts instead of standard frame waiting
                self.skip_frames = 0
                self.current_input = no_button_input

            elif self.skip_frames:
                # check if we should even process the event, hold current button input
                self.skip_frames -= 1
                if self.skip_frames == 0:
                    self.current_input = no_button_input
            
            elif gameplay_state == zelda.ZeldaGameStates.game_over:
                # finish the iteration
                self.agent.end_game()
                self.current_iteration += 1
                self.current_input = no_button_input
                print("game over")

                if self.current_iteration < self.total_iterations:
                    # start the next iteration
                    emu.loadSaveState("x:\\start.mss")
                    self.agent.begin_game()
                else:
                    # we are done
                    self.disable()
                    self.agent.save("completed.dat")
                    print("Complete!")
                    return

            else:
                # our action is the controller input
                action = self.agent.get_action_from_game_state()
                emu.setInput(0, 0, action)

                # skip frames for the next action:
                if action_frame_skip_min < action_frame_skip_max:
                    self.current_input = action
                    self.skip_frames = random.randint(action_frame_skip_min, action_frame_skip_max)

        except Exception as e:
            print(e)
            import traceback
            print(traceback.format_exc())

            self.disable()

    def enable(self):
        emu.addEventCallback(self.onFrame, eventType.startFrame)
        emu.addEventCallback(self.onPollInput, eventType.inputPolled)

    def disable(self):
        emu.removeEventCallback(self.onFrame, eventType.startFrame)
        emu.removeEventCallback(self.onPollInput, eventType.inputPolled)

trainer = TrainAgent()
trainer.enable()

