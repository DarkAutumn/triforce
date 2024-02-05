import typing
from typing import Dict

from .end_conditions import ZeldaEndCondition
from .critic import ZeldaGameplayCritic

class ZeldaDungeonCritic(ZeldaGameplayCritic):
    def __init__(self):
        super().__init__()

        # reward finding new locations and kills high in dungeons
        self.kill_reward = self.reward_large
        self.health_change_reward = self.reward_large
        self.leave_dungeon_penalty = -self.reward_large

    def critique_location_discovery(self, old_state : typing.Dict[str, int], new_state : typing.Dict[str, int], rewards : typing.Dict[str, float]):
        # Override.  Only allow rewards for new locations in dungeons
        if new_state['level'] != 0:
            super().critique_location_discovery(old_state, new_state, rewards)
        else:
            if old_state['location'] != new_state['location']:
                rewards['penalty-leave-room'] = self.leave_dungeon_penalty
