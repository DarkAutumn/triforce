import numpy as np
from typing import Dict
from .zelda_game import *
from .zelda_game import get_heart_halves, is_in_cave, mode_gameplay, mode_cave

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

    def critique_gameplay(self, old_state : Dict[str, int], new_state : Dict[str, int], rewards : Dict[str, float]):
        """Called to get the reward for the transition from old_state to new_state"""
        raise NotImplementedError()
    
    def set_score(self, old_state : Dict[str, int], new_state : Dict[str, int]):
        """Override to set info['score']"""
        pass

class GameplayCritic(ZeldaCritic):
    def __init__(self):
        super().__init__()

        self._max_actions_on_same_screen = 1000
        self._position_change_cooldown = 5000000
        self._is_first_step = True

        # reward values
        self.rupee_reward = self.reward_small
        self.health_gained_reward = self.reward_large
        self.health_lost_penalty = -self.reward_large
        self.injure_kill_reward = self.reward_small
        self.new_location_reward = self.reward_medium
        
        # these are pivotal to the game, so they are rewarded highly
        self.bomb_pickup_reward = self.reward_large
        self.key_reward = self.reward_large
        self.heart_container_reward = self.reward_maximum
        self.triforce_reward = self.reward_maximum
        self.equipment_reward = self.reward_maximum

        # same room movement rewards
        self.wall_collision_penalty = -self.reward_tiny
        self.move_closer_reward = self.reward_tiny
        
        self.movement_scale_factor = 9.0
        self.move_away_penalty = -self.reward_tiny

        self.too_close_threshold = 20
        self.move_too_close_penalty = -self.reward_small
        
        # state tracking
        self._visted_locations = set()
        self._actions_on_same_screen = 0
        self._is_first_step = True

        # missed attack
        self.distance_threshold = 50
        self.attack_miss_penalty = -self.reward_tiny
        self.attack_no_enemies_penalty = -self.reward_minimum

        # items
        self.used_null_item_penalty = -self.reward_large
        self.bomb_miss_penalty = -self.reward_small
        self.bomb_hit_reward = self.reward_medium

    def clear(self):
        super().clear()
        self._visted_locations.clear()
        self._actions_on_same_screen = 0
        self._is_first_step = True

    def critique_gameplay(self, old : Dict[str, int], new : Dict[str, int], rewards : Dict[str, float]):
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
        self.critique_item_usage(old, new, rewards)

        # items
        self.critique_item_pickup(old, new, rewards)
        self.critique_key_pickup_usage(old, new, rewards)
        self.critique_equipment_pickup(old, new, rewards)

        # movement
        self.critique_location_discovery(old, new, rewards)
        self.critique_movement(old, new, rewards)
    
    # reward helpers, may be overridden
    def critique_equipment_pickup(self, old, new, rewards):
        if not self.equipment_reward:
            return

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
            rewards['reward-gained-bombs'] = self.bomb_pickup_reward

    def critique_health_change(self, old, new, rewards):
        old_hearts = get_heart_halves(old)
        new_hearts = get_heart_halves(new)

        if new_hearts < old_hearts:
            rewards['penalty-losing-health'] = self.health_lost_penalty
        elif new_hearts > old_hearts:
            rewards['reward-gaining-health'] = self.health_gained_reward
    
    def critique_heart_containers(self, old, new, rewards):
        # we can only gain one heart container at a time
        if get_heart_containers(old) <get_heart_containers(new):
            rewards['reward-gained-heart-container'] = self.heart_container_reward
    
    def critique_triforce(self, old, new, rewards):
        if get_num_triforce_pieces(old) < get_num_triforce_pieces(new) or (old["triforce_of_power"] == 0 and new["triforce_of_power"] == 1):
            rewards['reward-gained-triforce'] = self.triforce_reward

    def critique_attack(self, old, new, rewards):
        if new['step_hits']:
            if not is_in_cave(new):
                rewards['reward-hit'] = self.injure_kill_reward
            else:
                rewards['penalty-hit-cave'] = self.injure_kill_reward
        
        else:
            if new['action'] == 'attack':                
                if not new['enemies']:
                    rewards['penalty-attack-no-enemies'] = self.attack_no_enemies_penalty

                elif self.offscreen_sword_disabled(new):
                    rewards['penalty-attack-offscreen'] = self.attack_miss_penalty

                elif new['enemies']:
                    enemy_vectors = [enemy.vector for enemy in new['enemies'] if abs(enemy.distance) > 0]
                    if enemy_vectors:
                        dotproducts = np.sum(new['link_vector'] * enemy_vectors, axis=1)
                        if not np.any(dotproducts > np.sqrt(2) / 2):
                            rewards['penalty-attack-miss'] = self.attack_miss_penalty
                        elif not old['has_beams']:
                            distance = new['enemies'][0].distance
                            if distance > self.distance_threshold:
                                rewards['penalty-attack-miss'] = self.attack_miss_penalty

    def critique_item_usage(self, old, new, rewards):
        if new['action'] == 'item':
            selected = new['selected_item']
            if selected == 0 and not new['regular_boomerang'] and not new['magic_boomerang']:
                rewards['used-null-item'] = self.used_null_item_penalty
            elif selected == 1:  # bombs
                total_hits = new.get('bomb1_hits', 0) + new.get('bomb2_hits', 0)
                if total_hits == 0:
                    rewards['penalty-bomb-miss'] = self.bomb_miss_penalty
                else:
                    rewards['reward-bomb-hit'] = min(self.bomb_hit_reward * total_hits, 1.0)

    def offscreen_sword_disabled(self, new):
        x, y = new['link_pos']
        return x <= 16 or x >= 0xd9 or y <= 0x53 or y >= 0xc5


    def critique_location_discovery(self, old, new, rewards):
        prev = (old['level'], old['location'])
        curr = (new['level'], new['location'])

        if self.new_location_reward and prev != curr and not self.has_visited(*curr):
            self.mark_visited(*curr)
            rewards['reward-new-location'] = self.new_location_reward
    
    def critique_movement(self, old, new, rewards):
        if new['action'] != "movement" or old['location'] != new['location'] or is_in_cave(old) != is_in_cave(new):
            return
        
        # Did link run into a wall?
        if old['link_pos'] == new['link_pos']:
            # corner case: Entering a cave with text scroll, link is uncontrollable
            # but it's hard to detect this state, so we'll just ignore it
            if not is_in_cave(new) or new['link_y'] < 0x80:
                rewards['penalty-wall-collision'] = self.wall_collision_penalty
                return
            
        # did link move too close to an enemy?
        old_enemies_too_close = {x.id: x for x in old['enemies'] if x.distance < self.too_close_threshold}
        if old_enemies_too_close:
            new_enemies_too_close = [x for x in new['enemies'] if x.id in old_enemies_too_close]
            if new_enemies_too_close:
                for enemy in new_enemies_too_close:
                    if enemy.distance < old_enemies_too_close[enemy.id].distance:
                        rewards['penalty-move-too-close'] = self.move_too_close_penalty
                        return
                    
        # don't reward/penalize if we lost health, just taking the damage is penalty enough
        if get_heart_halves(old) <= get_heart_halves(new):

            # do we have an optimal path?
            old_path = old.get('optimal_path', [])
            new_path = new.get('optimal_path', [])
            if len(old_path) >= 2:
                optimal_direction = self.get_optimal_direction(old_path[0], old_path[1])
                direction = new['direction']
                
                # reward if we moved in the right direction
                if optimal_direction == direction:
                    rewards['reward-move-closer'] = self.move_closer_reward

                # penalize moving in the opposite direction
                elif self.is_opposite_direction(optimal_direction, direction):
                    rewards['penalty-move-farther'] = self.move_away_penalty

                # Only penalize here if the new tile we reached is farther from the target
                else:
                    # see if we reached a new tile
                    if new_path[0] != old_path[0]:
                        if len(new_path) >= len(old_path):
                            rewards['penalty-move-farther'] = self.move_away_penalty
                        elif len(new_path) < len(old_path):
                            rewards['reward-move-closer'] = self.move_closer_reward

    def is_opposite_direction(self, a, b):
        if a == 'N' and b == 'S':
            return True
        if a == 'S' and b == 'N':
            return True
        if a == 'E' and b == 'W':
            return True
        if a == 'W' and b == 'E':
            return True
        return False

    def get_optimal_direction(self, old_index, new_index):
        # given two (x, y) points as indexes, figure out if the movement was N S E or W
        if new_index[0] > old_index[0]:
            return 'S'
        elif new_index[0] < old_index[0]:
            return 'N'
        elif new_index[1] > old_index[1]:
            return 'E'
        elif new_index[1] < old_index[1]:
            return 'W'

    def manhattan_distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # state helpers, some states are calculated
    def has_visited(self, level, location):
        return (level, location) in self._visted_locations
    
    def mark_visited(self, level, location):
        self._visted_locations.add((level, location))


class Dungeon1Critic(GameplayCritic):
    def __init__(self):
        super().__init__()

        self.health_change_reward = self.reward_large
        self.leave_dungeon_penalty = -self.reward_maximum
        self.leave_early_penalty = -self.reward_maximum
        self.seen = set()

    def clear(self):
        super().clear()
        self.seen.clear()

    def critique_location_discovery(self, old_state : Dict[str, int], new_state : Dict[str, int], rewards : Dict[str, float]):
        if new_state['level'] != 1:
            rewards['penalty-left-dungeon'] = self.leave_dungeon_penalty
        
        elif old_state['location'] != new_state['location']:
            if old_state['location_objective'] == new_state['location']:
                rewards['reward-new-location'] = self.new_location_reward
            else:
                rewards['penalty-left-early'] = self.leave_early_penalty

    def set_score(self, old : Dict[str, int], new : Dict[str, int]):
        new_location = new['location']
        self.seen.add(new_location)
        new['score'] = len(self.seen) - 1 + get_heart_halves(new) * 0.5

class Dungeon1BeamCritic(Dungeon1Critic):
    def __init__(self):
        super().__init__()
        self.health_gained_reward = 0.0
        self.health_lost_penalty = -self.reward_maximum

class Dungeon1BombCritic(Dungeon1Critic):
    def clear(self):
        super().clear()
        self.score = 0
        self.bomb_miss_penalty = -self.reward_minimum

    def set_score(self, old : Dict[str, int], new : Dict[str, int]):
        if new['action'] == 'item':
            selected = new['selected_item']
            if selected == 1:  # bombs
                hits = new.get('bomb1_hits', 0) + new.get('bomb2_hits', 0)
                if hits:
                    self.score += hits
                else:
                    self.score -= 1


        new['score'] = self.score

class Dungeon1BossCritic(Dungeon1Critic):
    def clear(self):
        super().clear()
        self.total_damage = 0
        self.too_close_threshold = 10
        self.move_closer_reward = self.reward_small
        self.move_away_penalty = -self.reward_small
        self.injure_kill_reward = self.reward_large
        self.health_lost_penalty = -self.reward_small

    def set_score(self, old : Dict[str, int], new : Dict[str, int]):
        self.total_damage += new['step_hits']
        new['score'] = get_heart_halves(new) + self.total_damage

overworld_dungeon1_walk_rooms = set([0x78, 0x67, 0x68, 0x58, 0x48, 0x38, 0x37])

class Overworld1Critic(GameplayCritic):
    def clear(self):
        super().clear()
        self.seen = set()
        self.allowed_rooms = overworld_dungeon1_walk_rooms

        self.left_allowed_area_penalty = -self.reward_large
        self.left_without_sword_penalty = -self.reward_large
        self.leave_early_penalty = -self.reward_maximum
        self.entered_cave_penalty = -self.reward_large
        self.equipment_reward = None
        
    def critique_location_discovery(self, old, new, rewards):
        if old['location'] != new['location'] and old['location_objective']:
            if old['location_objective'] != new['location']:
                rewards['penalty-left-early'] = self.leave_early_penalty
                return

        level = new['level']
        location = new['location']

        if old['mode'] == mode_gameplay and location == 0x77 and new['mode'] == mode_cave:
            rewards['penalty-entered-cave'] = self.entered_cave_penalty

        elif level == 0:
            if location not in self.allowed_rooms:
                rewards['penalty-left-allowed-area'] = self.left_allowed_area_penalty

            elif old['location'] == 0x77 and location != 0x77 and not new['sword']:
                rewards['penalty-no-sword'] = self.left_without_sword_penalty
                
            else:
                super().critique_location_discovery(old, new, rewards)
            
        elif level == 1:
            # don't forget to reward for reaching level 1 dungeon
            super().critique_location_discovery(old, new, rewards)

    def set_score(self, old : Dict[str, int], new : Dict[str, int]):
        new_location = new['location']
        self.seen.add(new_location)
        new['score'] = new['sword'] + len(self.seen) - 1

class OverworldSwordCritic(GameplayCritic):
    def __init__(self):
        super().__init__()
        self.entered_cave = False

        self.entered_cave_reward = self.reward_large
        self.left_cave_penalty = -self.reward_large

    def clear(self):
        self.entered_cave = False

    def critique_location_discovery(self, old, new, rewards):
        if not self.entered_cave and is_in_cave(new):
            self.entered_cave = True
            rewards['reward-entered-cave'] = self.entered_cave_reward

        if is_in_cave(old) and not is_in_cave(new) and not new['sword']:
            rewards['penalty-left-cave'] = self.left_cave_penalty

    
    def set_score(self, old : Dict[str, int], new : Dict[str, int]):
        score = 0
        if self.entered_cave:
            score += 1

        if new['sword']:
            score += 1

        if new['sword'] and not is_in_cave(new):
            score += 1

        new['score'] = score