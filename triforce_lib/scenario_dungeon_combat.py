import typing

from .end_condition import ZeldaEndCondition
from .scenario_dungeon import ZeldaDungeonCritic
from .zelda_game_data import zelda_game_data
from .model_parameters import actions_per_second

class ZeldaDungeonCombatCritic(ZeldaDungeonCritic):
    def __init__(self, reporter=None):
        super().__init__(reporter)

        self.new_location_reward = 0
        self.leaving_penalty = -self.reward_large
    
    def critique_location_discovery(self, old_state : typing.Dict[str, int], new_state : typing.Dict[str, int]):
        # Override.  Only allow rewards for new locations in dungeons
        reward = 0

        if old_state['location'] != new_state['location']:
            reward += self.leaving_penalty
            self.report(reward, f"Penalty for leaving the scenario! {reward}", "leave-room-penalty")

        return reward
    
class ZeldaDungeonCombatEndCondition(ZeldaEndCondition):
    def __init__(self, reporter=None):
        super().__init__(reporter)

        # 3 minutes to beat each room
        total_minutes = 5
        total_seconds = total_minutes * 60
        self._step_max = total_seconds * actions_per_second

    def clear(self):
        super().clear()
        self._room = None
        self._frame_count = 0

    def is_scenario_ended(self, old_state : typing.Dict[str, int], new_state : typing.Dict[str, int]):
        terminated, truncated = super().is_scenario_ended(old_state, new_state)

        self._frame_count += 1

        if self._room is None:
            self._room = zelda_game_data.get_room_by_location(new_state['level'], new_state['location'])

        if new_state['total_kills'] >= self._room.enemies:
            self.report("completed", "Completed the room!")
            terminated = True

        elif old_state['location'] != new_state['location']:
            self.report("left", "Left the room!")
            terminated = True

        elif self._frame_count >= self._step_max:
            self.report("timed-out", "Timed out!")
            terminated = True

        return terminated, truncated