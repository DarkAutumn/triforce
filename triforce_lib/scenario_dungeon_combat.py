import typing

from .end_condition import ZeldaEndCondition
from .scenario_dungeon import ZeldaDungeonCritic
from .zelda_game_data import zelda_game_data

class ZeldaDungeonCombatCritic(ZeldaDungeonCritic):
    def __init__(self, verbose=False):
        super().__init__(verbose)

        self.new_location_reward = 0
        self.leaving_penalty = self.reward_large
    
    def critique_location_discovery(self, old_state : typing.Dict[str, int], new_state : typing.Dict[str, int]):
        # Override.  Only allow rewards for new locations in dungeons
        reward = 0

        if old_state['location'] != new_state['location']:
            reward += self.leaving_penalty
            self.report(reward, f"Penalty for leaving the scenario! {reward}")

        return reward
    
class ZeldaDungeonCombatEndCondition(ZeldaEndCondition):
    def __init__(self, verbose=False):
        super().__init__(verbose)

    def clear(self):
        super().clear()
        self._room = None

    def is_scenario_ended(self, old_state : typing.Dict[str, int], new_state : typing.Dict[str, int]):
        terminated, truncated = super().is_scenario_ended(old_state, new_state)

        if self._room is None:
            self._room = zelda_game_data.get_room_by_location(new_state['level'], new_state['location'])

        if new_state['total_kills'] >= self._room.enemies:
            self.report("completed", "Completed the room!")
            terminated = True

        elif old_state['location'] != new_state['location']:
            self.report("left", "Left the room!")
            terminated = True

        return terminated, truncated