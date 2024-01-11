from typing import Dict
from .zelda_game import is_mode_death

class ZeldaEndCondition:
    def __init__(self, verbose=0):
        self.verbose = verbose

        # the number of timesteps the agent can be in the same position before we truncate
        self.position_timeout = 50
        
        # how long we've been in the same position on screen
        self._position_duration = 0
        self.end_causes = {}

    def report(self, source, message):
        if self.verbose >= 2:
            print(f"{source}: {message}")

        self.end_causes[source] = self.end_causes.get(source, 0) + 1

    def clear(self):
        self._position_duration = 0
        self.end_causes.clear()

    def is_scenario_ended(self, old : Dict[str, int], new : Dict[str, int]) -> (bool, bool):
        """Called to determine if the scenario has ended, returns (terminated, truncated)"""
        terminated = False

        if is_mode_death(new['mode']):
            self.report("terminated-game-over", "Game over")
            terminated = True
        
        elif new['triforce_of_power']:
            self.report("terminated-game-won", "Got the triforce of power")
            terminated = True

        # check truncation
        truncated = False
        last_position = (old['link_x'], old['link_y'])
        curr_position = (new['link_x'], new['link_y'])

        if last_position == curr_position:
            if self._position_duration >= self.position_timeout:
                self.report("truncated-position-timeout", f"Truncated - Stuck in same position for too long ({self._position_duration} steps)")
                truncated = True

            self._position_duration += 1
        else:
            self._position_duration = 0


        return terminated, truncated