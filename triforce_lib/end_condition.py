from .zelda_game import is_mode_death

class ZeldaEndCondition:
    def __init__(self, verbose=0):
        self.verbose = verbose

    def print_verbose(self, message):
        if self.verbose >= 2:
            print(message)

    def is_terminated(self, info):
        raise NotImplementedError() 
    
    def is_truncated(self, info):
        raise NotImplementedError()
    
    def clear(self):
        pass

class ZeldaGameplayEndCondition(ZeldaEndCondition):
    """The basic end condition for all gameplay.
    We terminate if the agent dies or gets the triforce of power after beating Ganon.
    We truncate if the agent is stuck in the same x, y location for 50 timesteps (~12 sec).
    """
    def __init__(self, verbose=False):
        super().__init__(verbose)
        self._last_position = None
        self._position_duration = 0

    def is_terminated(self, info):
        if is_mode_death(info['mode']):
            self.print_verbose("Game over")
            return True
        
        if info['triforce_of_power']:
            self.print_verbose("Got the triforce of power")
            return True
    
    def is_truncated(self, info):
        position = (info['link_x'], info['link_y'])

        if position == self._last_position:
            self._position_duration += 1
            if self._position_duration > 50:
                self.print_verbose("Truncated - Stuck in same position for too long")
                return True
        else:
            self._position_duration = 0

        self._last_position = position
        return False
    
    def clear(self):
        self._last_position = None
        self._position_duration = 0
