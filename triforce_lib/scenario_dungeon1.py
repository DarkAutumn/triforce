import typing
from typing import Any

from .critic import ZeldaGameplayCritic
from .end_condition import ZeldaEndCondition
from .zelda_game import get_heart_containers, get_num_triforce_pieces

class Dungeon1Critic(ZeldaGameplayCritic):
    def __init__(self):
        super().__init__()

        self.health_change_reward = self.reward_large
        self.leave_dungeon_penalty = -self.reward_maximum
        self.leave_early_penalty = -self.reward_maximum
        self.seen = set()

    def clear(self):
        super().clear()
        self.seen.clear()

    def critique_location_discovery(self, old_state : typing.Dict[str, int], new_state : typing.Dict[str, int], rewards : typing.Dict[str, float]):
        if new_state['level'] != 1:
            rewards['penalty-left-dungeon'] = self.leave_dungeon_penalty
        
        elif old_state['location'] != new_state['location']:
            if old_state['location_objective'] == new_state['location']:
                rewards['reward-new-location'] = self.new_location_reward
            else:
                rewards['penalty-left-early'] = self.leave_early_penalty

    def set_score(self, old : typing.Dict[str, int], new : typing.Dict[str, int]):
        new_location = new['location']
        self.seen.add(new_location)
        new['score'] = len(self.seen) - 1

class Dungeon1BossCritic(Dungeon1Critic):
    def clear(self):
        super().clear()
        self.score = 0
        self.too_close_threshold = 10

    def set_score(self, old : typing.Dict[str, int], new : typing.Dict[str, int]):
        if not self.score and new['step_kills']:
            self.score = 1

        new['score'] = self.score

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
            if location == 0x35:
                reason = "reached-boss"
                terminated = True

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
    
class Dungeon1BossEndCondition(ZeldaEndCondition):
    def is_scenario_ended(self, old_state : typing.Dict[str, int], new_state : typing.Dict[str, int]):
        terminated, truncated, reason = super().is_scenario_ended(old_state, new_state)

        if not terminated and not truncated:
            location = new_state['location']
            if location != 0x35:
                reason = "left-scenario"
                terminated = True

            if get_heart_containers(old_state) < get_heart_containers(new_state):
                reason = "won-scenario"
                terminated = True

        return terminated, truncated, reason