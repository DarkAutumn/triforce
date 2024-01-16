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
        self.health_change_reward = self.reward_medium
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
        self.closing_distance_reward = self.reward_tiny
        self.moving_farther_penalty = -self.reward_tiny
        
        # state tracking
        self._visted_locations = [[False] * 256 ] * 2
        self._actions_on_same_screen = 0
        self._is_first_step = True

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
        self.critique_kills(old, new, rewards)

        # items
        self.critique_item_pickup(old, new, rewards)
        self.critique_key_pickup_usage(old, new, rewards)
        self.critique_equipment_pickup(old, new, rewards)

        # movement
        self.critique_location_discovery(old, new, rewards)
        self.critique_wall_collision(old, new, rewards)
        self.critique_closing_distance(old, new, rewards)
    
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

    def critique_kills(self, _, new, rewards):
        enemies_killed = new['step_kills']

        if new['step_kills'] or new['step_injuries']:
            rewards['reward-injure-kill'] = self.kill_reward

    def critique_location_discovery(self, old, new, rewards):
        prev = (old['level'], old['location'])
        curr = (new['level'], new['location'])

        if self.new_location_reward and prev != curr and not self.has_visited(*curr):
            self.mark_visited(*curr)
            rewards['reward-new-location'] = self.new_location_reward
    
    def critique_wall_collision(self, old, new, rewards):
        """"Reward the agent for moving out of the last position, but only if that did not happen as a result of taking a hit
        from an enemy."""

        if self.wall_collision_penalty is not None:
            if old['link_pos'] == new['link_pos'] and new['action'] == 'movement':
                # link bumped against the wall and didn't move despite choosing to move
                rewards['penalty-wall-collision'] = self.wall_collision_penalty
    
    def critique_closing_distance(self, old, new, rewards):
        if self.closing_distance_reward is not None and new['new_position'] and new['action'] == 'movement':
            objects = new['objects']

            if objects.enemy_count > 0:
                if old['link_pos'] != new['link_pos']:
                    # calculate Link's normalized motion vector
                    link_old_pos = np.array(old['link_pos'], dtype=np.float32)
                    link_new_pos = np.array(new['link_pos'], dtype=np.float32)

                    link_motion_vector = link_new_pos - link_old_pos
                    link_motion_vector = link_motion_vector / np.linalg.norm(link_motion_vector)

                    # calculate normalized vector from link to each enemy
                    enemy_ids = [id for id in objects.enumerate_enemy_ids() if id is not None]
                    enemy_pos = np.array([objects.get_position(id) for id in enemy_ids])

                    vector_to_enemies = enemy_pos - link_old_pos

                    # add a small epislon to avoid divide by zero
                    epsilon = 1e-6
                    norms = np.linalg.norm(vector_to_enemies, axis=1) + epsilon
                    vector_to_enemies = vector_to_enemies / norms[:, np.newaxis]

                    # find points within a 90 degree cone of link's motion vector, COS(45) == sqrt(2)/2
                    dotproducts = np.sum(link_motion_vector * vector_to_enemies, axis=1)
                    enemies_closer = np.sum(dotproducts >= np.sqrt(2) / 2)
                    enemies_farther = np.all(dotproducts <= -np.sqrt(2) / 2)

                    if enemies_closer:
                        percentage = enemies_closer / float(objects.enemy_count)
                        rewards['reward-close-distance'] = percentage * self.close_distance_reward
                        
                    elif enemies_farther:
                        percentage = enemies_farther / float(objects.enemy_count)
                        rewards['penalty-moving-farther'] = percentage * self.moving_farther_penalty

    # state helpers, some states are calculated
    def has_visited(self, level, location):
        return self._visted_locations[level][location]
    
    def mark_visited(self, level, location):
        self._visted_locations[level][location] = True
