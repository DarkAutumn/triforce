from . import zelda_constants as zelda

class ZeldaEndCondition:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def print_verbose(self, message):
        if self.verbose:
            print(message)

    def is_terminated(self, info):
        raise NotImplementedError() 
    
    def is_truncated(self, info):
        raise NotImplementedError()

class ZeldaGameplayEndCondition(ZeldaEndCondition):
    """The basic end condition for all gameplay.  The run is over if the player dies or gets the triforce of power
    after beating Ganon."""
    def is_terminated(self, info):
        if info['mode'] == zelda.mode_game_over or info['mode'] == zelda.mode_game_over_screen:
            self.print_verbose("Game over")
            return True
        
        if info['triforce_of_power']:
            self.print_verbose("Got the triforce of power")
            return True
    
    def is_truncated(self, _):
        return False
