# a test script to print rewards to test scoring
import zelda
import importlib
import random
import mesen
from mesen_zelda import GameReplay

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
    def __init__(self, save_state):
        self.save_state = save_state
        memory = mesen.registerFrameMemory(7, zelda.ZeldaMemoryLayout.get_address_list())
        screen = mesen.registerScreenMemory()
        self.agent = zelda.LegendOfZeldaAgent(memory, screen)
        self.total_iterations = iterations

        self.current_iteration = 0

        self.max_frame = max_game_duration_sec / frames_per_second

        self.action_cooldown = 0
        self.current_input = None
        self.frames = None

        self.state = zelda.ZeldaGameState()


    def capture_frame(self):
            # always capture the current frame
            frame = self.frames.capture()

            # update game state
            game_state = self.state
            game_state.set_memory(frame.memory)
            return (frame, game_state)

    def onPollInput(self, _):
        if self.current_input:
            mesen.setInput(0, 0, self.current_input)

    def onFrame(self, _):
        try:
            frame, game_state = self.capture_frame()

            # First check if link is animation locked since we special case that
            if game_state.is_link_animation_locked():
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

            
            mode = game_state.get_mode()

            # check for game over
            if mode == zelda.ZeldaGameStates.game_over:
                # finish the iteration
                self.agent.end_game(game_state)

                self.current_iteration += 1
                self.current_input = no_button_input
                print("game over")

                if self.current_iteration < self.total_iterations:
                    # begin a new game
                    self.frames = GameReplay()
                    mesen.loadSaveState("x:\\start.mss")
                    self.agent.begin_game()

                else:
                    # we are done
                    self.disable()
                    self.agent.save("completed.dat")
                    print("Complete!")

                return

            # our action is the controller input
            action = self.agent.act(self.frames)
            print(action)
            mesen.setInput(0, 0, action)

            # Assign the input we just sent to the current frame instead of
            # the next frame.  This isn't technically correct, but it makes
            # logic simpler to save.
            frame.input = frame.encode_input(action)
            frame.agent_action = True

            # skip frames for the next action:
            if action_frame_skip_min < action_frame_skip_max:
                self.current_input = action
                self.action_cooldown = random.randint(action_frame_skip_min, action_frame_skip_max)

        except Exception as e:
            print(e)
            import traceback
            print(traceback.format_exc())

            self.disable()

    def enable(self):
        mesen.addEventCallback(self.onFrame, mesen.eventType.startFrame)
        mesen.addEventCallback(self.onPollInput, mesen.eventType.inputPolled)

    def disable(self):
        mesen.removeEventCallback(self.onFrame, mesen.eventType.startFrame)
        mesen.removeEventCallback(self.onPollInput, mesen.eventType.inputPolled)

trainer = TrainAgent("x:\\start.mss")
trainer.enable()
