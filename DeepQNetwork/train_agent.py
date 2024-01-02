# a test script to print rewards to test scoring
import zelda
import importlib
import random
import mesen
from mesen_zelda import MesenZeldaRecorder, action_id_to_controller

# Reload so that if I make live changes to zelda.py they are reflected in Mesen
importlib.reload(zelda)

iterations = 5
max_game_duration_min = 1
frames_per_second = 60.1
max_game_duration_frames = max_game_duration_min * 60 * frames_per_second

save_every = 50

data_dir = "X:\\data"

# after every action the agent takes, skip some frames so that we don't act on every frame
# it's important this be a range and not, say, every 2 frames because certain enemy animations
# are on a particular cycle, and would therefore be "invisible" to the agent without some
# variability
action_frame_skip_min = 10
action_frame_skip_max = 20

# input order: a b start select up down left right, use None to not set a value
start_button_input = [False, False, True, False, False, False, False, False]
no_button_input = [False, False, False, False, False, False, False, False]

class TrainAgent:
    def __init__(self, save_state, model = "default", scorer = "default"):
        self.save_state = save_state
        memory = mesen.registerFrameMemory(7, zelda.zelda_memory_layout.get_address_list())
        self.agent = zelda.LegendOfZeldaAgent(model, scorer)
        self.total_iterations = iterations

        self.current_iteration = 0
        self.current_frame = 0

        self.action_cooldown = 0
        self.current_input = None

        self.zelda_game_state = zelda.ZeldaGameState(memory)
        self.complete = False

        self.started = False

    def capture_frame(self):
            # always capture the current frame
            frame = self.frames.capture()

            # update game state
            game_state = self.zelda_game_state
            game_state.set_memory(frame.memory)
            return (frame, game_state)

    def onPollInput(self):
        if self.current_input:
            mesen.setInput(0, 0, self.current_input)

    def onFrame(self):
        if self.complete:
            return
        
        if not self.started:
            self.begin_game()
            self.started = True
        
        try:
            self.current_frame += 1
            frame, game_state = self.capture_frame()

            # First check if link is animation locked since we special case that
            if game_state.is_link_animation_locked:
                # If there's animation lock, it's fast, and we want the AI to be able to respond
                # when it lifts instead of standard frame waiting.  In this case, clear the
                # action cooldown, stop sending input, and stop processing.
                self.action_cooldown = 0
                self.current_input = no_button_input
                return

            # Check if we are on action cooldown
            if self.action_cooldown:
                self.action_cooldown -= 1

                # if we still have a cooldown, stop processing
                if self.action_cooldown:
                    return
                    
                # If cooldown just ended, clear the input and continue processing. Note that input
                # is polled before the frame is processed, so we want to go ahead and run the agent
                # since it won't be polled again until the next frame.
                else:
                    self.current_input = no_button_input

            
            mode = game_state.mode

            # check for timeout
            if self.current_frame >= max_game_duration_frames:
                print("max game duration reached")
                mode = zelda.zelda_game_modes.game_over

            # check for game over
            if mode == zelda.zelda_game_modes.game_over:
                self.game_over()
                return

            # our action is the controller input
            action = self.agent.act(self.frames)
            if action is not None:
                buttons = action_id_to_controller(action)
                mesen.setInput(0, 0, buttons)
                self.current_input = buttons

            else:
                pass
                # No action can be taken at the current time


            # skip frames for the next action:
            if action_frame_skip_min < action_frame_skip_max:
                self.action_cooldown = random.randint(action_frame_skip_min, action_frame_skip_max)

        except Exception as e:
            print(e)
            import traceback
            print(traceback.format_exc())

            disable()

    def begin_game(self):
        mesen.loadSaveState(self.save_state)
        self.frames = MesenZeldaRecorder()
        self.agent.begin_game()
        self.current_frame = 0
        self.action_cooldown = random.randint(action_frame_skip_min, action_frame_skip_max)

    def end_game(self):
        self.current_input = no_button_input
        self.agent.end_game(self.frames)
        self.agent.learn()

        self.current_iteration += 1
        if self.current_iteration % save_every == 0:
            self.agent.save(data_dir + '\\' + f"zelda_model_{iterations}.dat")

    def game_over(self):
        # finish the iteration
        print("game over")
        self.end_game()

        if self.current_iteration < self.total_iterations:
            self.begin_game()
        else:
            # we are done
            self.complete = True
            self.agent.save(data_dir + '\\' + "completed.dat")
            print("Complete!")
            disable()




trainer = TrainAgent("x:\\dungeon1.mss", scorer="dungeon")

def onFrame(cpuType):
    trainer.onFrame()

def onPollInput(cpuType):
    trainer.onPollInput()

def enable():
    mesen.addEventCallback(onFrame, mesen.eventType.startFrame)
    mesen.addEventCallback(onPollInput, mesen.eventType.inputPolled)

def disable():
    mesen.removeEventCallback(onFrame, mesen.eventType.startFrame)
    mesen.removeEventCallback(onPollInput, mesen.eventType.inputPolled)

print("Started")
enable()
