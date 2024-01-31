import typing

from .end_condition import ZeldaEndCondition
from .scenario_dungeon import ZeldaDungeonCritic
from .zelda_game_data import zelda_game_data
from .model_parameters import step_max

class ZeldaDungeonCombatCritic(ZeldaDungeonCritic):
    def __init__(self):
        super().__init__()

        self.new_location_reward = 0.0
        self.leaving_penalty = -self.reward_large
    
    def critique_location_discovery(self, old_state : typing.Dict[str, int], new_state : typing.Dict[str, int], rewards : typing.Dict[str, float]):
        # Override.  Only allow rewards for new locations in dungeons
        if old_state['location'] != new_state['location']:
            rewards['penalty-leave-room'] = self.leaving_penalty
    
class ZeldaDungeonCombatEndCondition(ZeldaEndCondition):
    def __init__(self):
        super().__init__()

        self._step_max = step_max

    def clear(self):
        super().clear()
        self._room = None
        self._frame_count = 0

    def is_scenario_ended(self, old_state : typing.Dict[str, int], new_state : typing.Dict[str, int]):
        terminated, truncated, reason = super().is_scenario_ended(old_state, new_state)

        self._frame_count += 1

        if not terminated and not truncated:
            if self._room is None:
                self._room = zelda_game_data.get_room_by_location(new_state['level'], new_state['location'])

            if old_state['location'] != new_state['location']:
                reason = "left-scenario"
                terminated = True

            elif new_state['objects'].enemy_count == 0:
                reason = "won-scenario"
                terminated = True

            elif self._frame_count >= self._step_max:
                reason = "timed-out"
                terminated = True

        return terminated, truncated, reason