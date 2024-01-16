from typing import Dict
from .zelda_game import is_mode_death

class ZeldaEndCondition:
    def __init__(self):
        # the number of timesteps the agent can be in the same position before we truncate
        self.position_timeout = 50
        
        # how long we've been in the same position on screen
        self._position_duration = 0
        self.end_causes = {}

    def clear(self):
        self._position_duration = 0
        self.end_causes.clear()

    def is_scenario_ended(self, old : Dict[str, int], new : Dict[str, int]) -> (bool, bool):
        """Called to determine if the scenario has ended, returns (terminated, truncated)"""
        reason = None
        terminated = False

        if is_mode_death(new['mode']):
            reason = "terminated-death"
            terminated = True
        
        elif new['triforce_of_power']:
            reason = "terminated-game-won"
            terminated = True

        # check truncation
        truncated = False
        if old['link_pos'] == new['link_pos'] and not new['step_kills'] and not new['step_injuries']:
            if self._position_duration >= self.position_timeout:
                reason = "truncated-position-timeout"
                truncated = True

            self._position_duration += 1
        else:
            self._position_duration = 0

        return terminated, truncated, reason