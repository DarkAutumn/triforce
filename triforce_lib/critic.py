import math
import numpy as np
import typing
import inspect
from .zelda_game import *

class ZeldaCritic:
    @property
    def reward_minimum(self):
        return 0.01
    
    @property
    def reward_tiny(self):
        return 0.05
    
    @property
    def reward_small(self):
        return 0.25
    
    @property
    def reward_medium(self):
        return 0.5
    
    @property
    def reward_large(self):
        return 0.75
    
    @property
    def reward_maximum(self):
        return 1.0

    def clear(self):
        """Called when the environment is reset to clear any saved state"""
        pass

    def critique_gameplay(self, old_state : typing.Dict[str, int], new_state : typing.Dict[str, int], rewards : typing.Dict[str, float]):
        """Called to get the reward for the transition from old_state to new_state"""
        raise NotImplementedError()

class ZeldaGameplayCritic(ZeldaCritic):
    def __init__(self):
        super().__init__()

        self._max_actions_on_same_screen = 1000
        self._position_change_cooldown = 5000000
        self._is_first_step = True

        # reward values
        self.rupee_reward = self.reward_small
        self.health_change_reward = self.reward_large
        self.kill_reward = self.reward_small
        self.new_location_reward = self.reward_medium
        
        # these are pivotal to the game, so they are rewarded highly
        self.bomb_reward = self.reward_large
        self.key_reward = self.reward_large
        self.heart_container_reward = self.reward_maximum
        self.triforce_reward = self.reward_maximum
        self.equipment_reward = self.reward_maximum

        # same room movement rewards
        self.wall_collision_penalty = -self.reward_tiny
        self.move_closer_reward = self.reward_tiny
        self.movement_reward_min = 2.5
        self.movement_reward_max = 8.0
        self.move_farther_penalty = -self.reward_tiny
        
        # state tracking
        self._visted_locations = [[False] * 256 ] * 2
        self._actions_on_same_screen = 0
        self._is_first_step = True

        # missed attack
        self.distance_threshold = 50
        self.attack_miss_penalty = -self.reward_minimum
        self.attack_no_enemies_penalty = -self.reward_tiny

    def clear(self):
        super().clear()
        self._visted_locations = [[False] * 256 ] * 2
        self._actions_on_same_screen = 0
        self._is_first_step = True

    def critique_gameplay(self, old : typing.Dict[str, int], new : typing.Dict[str, int], rewards : typing.Dict[str, float]):
        if self._is_first_step:
            self._is_first_step = False
            self.mark_visited(new['level'], new['location'])

        # health
        self.critique_health_change(old, new, rewards)
        self.critique_heart_containers(old, new, rewards)

        # triforce
        self.critique_triforce(old, new, rewards)

        # combat
        self.critique_attack(old, new, rewards)

        # items
        self.critique_item_pickup(old, new, rewards)
        self.critique_key_pickup_usage(old, new, rewards)
        self.critique_equipment_pickup(old, new, rewards)

        # movement
        self.critique_location_discovery(old, new, rewards)
        self.critique_movement(old, new, rewards)
    
    # reward helpers, may be overridden
    def critique_equipment_pickup(self, old, new, rewards):
        self.check_one_item(old, new, rewards, 'sword')
        self.check_one_item(old, new, rewards, 'arrows')
        self.check_one_item(old, new, rewards, 'bow')
        self.check_one_item(old, new, rewards, 'candle')
        self.check_one_item(old, new, rewards, 'whistle')
        self.check_one_item(old, new, rewards, 'food')
        self.check_one_item(old, new, rewards, 'potion')
        self.check_one_item(old, new, rewards, 'magic_rod')
        self.check_one_item(old, new, rewards, 'raft')
        self.check_one_item(old, new, rewards, 'magic_book')
        self.check_one_item(old, new, rewards, 'ring')
        self.check_one_item(old, new, rewards, 'step_ladder')
        self.check_one_item(old, new, rewards, 'magic_key')
        self.check_one_item(old, new, rewards, 'power_bracelet')
        self.check_one_item(old, new, rewards, 'letter')
        self.check_one_item(old, new, rewards, 'regular_boomerang')
        self.check_one_item(old, new, rewards, 'magic_boomerang')
        self.check_one_item(old, new, rewards, 'compass')
        self.check_one_item(old, new, rewards, 'map')
        self.check_one_item(old, new, rewards, 'compass9')
        self.check_one_item(old, new, rewards, 'map9')

    def check_one_item(self, old, new, rewards, item):
        if old[item] < new[item]:
            rewards['reward-equipment-gained'] = self.equipment_reward

    def critique_key_pickup_usage(self, old, new, rewards):
        # No matter if link picked up a key or used a key to open a door, it's a good outcome.
        # Make sure that link picked it up off the floor though, and didn't just bump into an
        # enemy with a key
        
        old_hearts = get_heart_halves(old)
        new_hearts = get_heart_halves(new)

        if old['keys'] > new['keys']:
            rewards['reward-used-key'] = self.key_reward
        elif old['keys'] < new['keys'] and old_hearts <= new_hearts:
            rewards['reward-gained-key'] = self.key_reward

    def critique_item_pickup(self, old, new, rewards):
        if old['rupees_to_add'] < new['rupees_to_add']:
            rewards['reward-gained-rupees'] = self.rupee_reward

        if old['bombs'] != 0 < new['bombs'] == 0:
            rewards['reward-gained-bombs'] = self.bomb_reward

    def critique_health_change(self, old, new, rewards):
        old_hearts = get_heart_halves(old)
        new_hearts = get_heart_halves(new)

        if new_hearts < old_hearts:
            rewards['penalty-losing-health'] = -self.health_change_reward
        elif new_hearts > old_hearts:
            rewards['reward-gaining-health'] = self.health_change_reward
    
    def critique_heart_containers(self, old, new, rewards):
        # we can only gain one heart container at a time
        if get_heart_containers(old) <get_heart_containers(new):
            rewards['reward-gained-heart-container'] = self.heart_container_reward
    
    def critique_triforce(self, old, new, rewards):
        if get_num_triforce_pieces(old) < get_num_triforce_pieces(new) or (old["triforce_of_power"] == 0 and new["triforce_of_power"] == 1):
            rewards['reward-gained-triforce'] = self.triforce_reward

    def critique_attack(self, _, new, rewards):
        if new['step_kills'] or new['step_injuries']:
            rewards['reward-injure-kill'] = self.kill_reward
        else:
            if new['action'] == 'attack':                
                if not new['enemies_on_screen']:
                    rewards['penalty-attack-no-enemies'] = self.attack_no_enemies_penalty

                elif new['enemy_vectors']:
                    enemy_vectors = [x[0] for x in new['enemy_vectors'] if abs(x[1]) > 0]
                    if enemy_vectors:
                        dotproducts = np.sum(new['link_vector'] * enemy_vectors, axis=1)
                        if not np.any(dotproducts > np.sqrt(2) / 2):
                            rewards['penalty-attack-miss'] = self.attack_miss_penalty
                        elif not new['has_beams']:
                            distance = new['enemy_vectors'][0][1]
                            if distance > self.distance_threshold:
                                rewards['penalty-attack-miss'] = self.attack_miss_penalty

    def critique_location_discovery(self, old, new, rewards):
        prev = (old['level'], old['location'])
        curr = (new['level'], new['location'])

        if self.new_location_reward and prev != curr and not self.has_visited(*curr):
            self.mark_visited(*curr)
            rewards['reward-new-location'] = self.new_location_reward
    
    def critique_movement(self, old, new, rewards):
        if new['action'] == 'movement':
            if old['link_pos'] == new['link_pos']:
                # link tried to move but ran into the wall instead
                rewards['penalty-wall-collision'] = self.wall_collision_penalty
            
            elif get_heart_halves(old) <= get_heart_halves(new):
                # Reward moving towards objectives, or moving to the closest item
                # Penalize directly moving away from the objective if not moving twoards an item
                # Don't reward if we took damage, or if we didn't move more than a threshold

                objective_vectors = [new['objective_vector'], new['closest_item_vector']]
                nonzero_vectors = [v for v in objective_vectors if v[0] or v[1]]

                if nonzero_vectors:
                    link_old_pos = np.array(old['link_pos'], dtype=np.float32)
                    link_new_pos = np.array(new['link_pos'], dtype=np.float32)

                    # we know the norm won't be zero since we know the position has changed
                    link_motion_vector = link_new_pos - link_old_pos
                    dist = np.linalg.norm(link_motion_vector)

                    if dist > self.movement_reward_min:
                        link_motion_vector = link_motion_vector / dist

                        # find points within a 90 degree cone of link's motion vector, COS(45) == sqrt(2)/2
                        dotproducts = np.sum(link_motion_vector * nonzero_vectors, axis=1)
                        if np.any(dotproducts >= np.sqrt(2) / 2):
                            percent = min(dist / self.movement_reward_max, 1)
                            rewards['reward-move-closer'] = self.move_closer_reward * percent

                        elif np.all(dotproducts <= -np.sqrt(2) / 2):
                            rewards['penalty-move-farther'] = self.move_farther_penalty

    # state helpers, some states are calculated
    def has_visited(self, level, location):
        return self._visted_locations[level][location]
    
    def mark_visited(self, level, location):
        self._visted_locations[level][location] = True
