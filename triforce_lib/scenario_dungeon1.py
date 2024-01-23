import typing

from .critic import ZeldaGameplayCritic
from .end_condition import ZeldaEndCondition
from .zelda_game import get_num_triforce_pieces

class Dungeon1Critic(ZeldaGameplayCritic):
    def __init__(self):
        super().__init__()

        self.health_change_reward = self.reward_large
        self.leave_dungeon_penalty = -self.reward_large

    def critique_location_discovery(self, old_state : typing.Dict[str, int], new_state : typing.Dict[str, int], rewards : typing.Dict[str, float]):
        if new_state['level'] != 1:
            rewards['penalty-left-dungeon'] = self.leave_dungeon_penalty
        
        elif old_state['location'] != new_state['location']:
            if old_state['location_objective'] == new_state['location']:
                rewards['reward-new-location'] = self.new_location_reward
            else:
                rewards['penalty-left-early'] = -self.new_location_reward

class Dungeon1EndCondition(ZeldaEndCondition):
    def clear(self):
        super().clear()
        self._new_rooms = set()
        self._frame_count = 0

    def is_scenario_ended(self, old_state : typing.Dict[str, int], new_state : typing.Dict[str, int]):
        terminated, truncated, reason = super().is_scenario_ended(old_state, new_state)

        self._frame_count += 1

        if not terminated and not truncated:
            if new_state['level'] != 1:
                reason = "left-scenario"
                terminated = True

            location = new_state['location']
            if location not in self._new_rooms:
                self._new_rooms.add(location)
                self._frame_count = 0
            else:
                self._frame_count += 1
                if self._frame_count > 800:
                    reason = "no-discovery-timeout"
                    terminated = True

            if get_num_triforce_pieces(old_state) < get_num_triforce_pieces(new_state):
                reason = "gained-triforce"
                terminated = True

        return terminated, truncated, reason
    