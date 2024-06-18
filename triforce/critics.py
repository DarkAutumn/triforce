"""Gameplay critics for Zelda."""

from typing import Dict, OrderedDict
import numpy as np
import gymnasium as gym
import torch

from triforce.wavefront import RoomWavefront

from .ml_torch import SelectedAction, direction_to_action, SelectedDirection
from .objective_selector import ObjectiveKind
from .zelda_wrapper import ActionType
from .zelda_game import Direction, TileState, ZeldaObject, ZeldaSoundsPulse1, get_heart_containers, get_heart_halves, \
    get_num_triforce_pieces, is_in_cave, is_sword_frozen, is_tile_walkable

REWARD_MINIMUM = 0.01
REWARD_TINY = 0.05
REWARD_SMALL = 0.25
REWARD_MEDIUM = 0.5
REWARD_LARGE = 0.75
REWARD_MAXIMUM = 1.0

def is_deflecting(old, new):
    """Returns True if the agent is deflecting an arrow, False otherwise."""
    arrow_deflected = ZeldaSoundsPulse1.ArrowDeflected.value
    return new['sound_pulse_1'] & arrow_deflected and (old['sound_pulse_1'] & arrow_deflected) != arrow_deflected

def _manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def _get_progress(direction, old_link_pos, new_link_pos, target):
    direction_vector = direction.to_vector()

    disp = np.array(new_link_pos) - np.array(old_link_pos)
    direction_movement = np.dot(disp, direction_vector) * direction_vector

    projected_new_pos = old_link_pos + direction_movement

    old_distance = _manhattan_distance(target, old_link_pos)
    new_distance = _manhattan_distance(target, projected_new_pos)

    return old_distance - new_distance

def _xy_from_coord(bottom_left):
    y, x = bottom_left
    old_link_tile = np.array([x, y], dtype=np.float32)
    return old_link_tile


def _get_direction(old, new):
    if new[0] > old[0]:
        return Direction.S
    if new[0] < old[0]:
        return Direction.N
    if new[1] > old[1]:
        return Direction.E
    if new[1] < old[1]:
        return Direction.W
    return None

def _find_second_turn(path):
    turn = 0
    direction = _get_direction(path[0], path[1])
    for i in range(2, len(path)):
        old_index = path[i - 1]
        new_index = path[i]

        new_direction = _get_direction(old_index, new_index)
        if new_direction != direction:
            turn += 1
            direction = new_direction
            if turn == 2:
                return old_index

    return path[-1]

class ZeldaCritic:
    """Base class for Zelda gameplay critics."""
    def __init__(self):
        self.health_lost = 0

    def clear(self):
        """Called when the environment is reset to clear any saved state."""
        self.health_lost = 0

    def critique_gameplay(self, old: Dict[str, int], new: Dict[str, int], rewards: Dict[str, float]):
        """
        Critiques the gameplay by comparing the old and new states and the rewards obtained.

        Args:
            old (Dict[str, int]): The old state of the game.
            new (Dict[str, int]): The new state of the game.
            rewards (Dict[str, float]): The rewards obtained during gameplay.
        """
        raise NotImplementedError()

    def set_score(self, old : Dict[str, int], new : Dict[str, int]):
        """Override to set info['score']"""

        diff = get_heart_halves(new) - get_heart_halves(old)
        if diff < 0:
            self.health_lost += diff

        new['score'] = self.health_lost

class GameplayCritic(ZeldaCritic):
    """Base class for Zelda gameplay critics."""
    def __init__(self):
        super().__init__()

        # reward values
        self.rupee_reward = REWARD_SMALL
        self.health_gained_reward = REWARD_LARGE
        self.new_location_reward = REWARD_MEDIUM

        # combat values
        self.wipeout_reward_on_hits = True
        self.health_lost_penalty = -REWARD_LARGE
        self.injure_kill_reward = REWARD_MEDIUM
        self.inure_kill_movement_room_reward = REWARD_SMALL
        self.block_projectile_reward = REWARD_LARGE

        self.didnt_fire_penalty = -REWARD_TINY
        self.fired_correctly_reward = REWARD_TINY

        # these are pivotal to the game, so they are rewarded highly
        self.bomb_pickup_reward = REWARD_LARGE
        self.key_reward = REWARD_LARGE
        self.heart_container_reward = REWARD_MAXIMUM
        self.triforce_reward = REWARD_MAXIMUM
        self.equipment_reward = REWARD_MAXIMUM

        # same room movement rewards
        self.wall_collision_penalty = -REWARD_TINY
        self.move_closer_reward = REWARD_TINY

        self.movement_scale_factor = 9.0
        self.move_away_penalty = -self.move_closer_reward - REWARD_MINIMUM

        self.warning_tile_penalty = -REWARD_TINY
        self.danger_tile_penalty = -REWARD_MEDIUM
        self.moved_to_safety_reward = REWARD_TINY

        # state tracking
        self._visted_locations = set()

        # missed attack
        self.distance_threshold = 28
        self.attack_miss_penalty = -self.move_closer_reward - REWARD_MINIMUM
        self.attack_no_enemies_penalty = -self.move_closer_reward * 2

        # items
        self.used_null_item_penalty = -REWARD_LARGE
        self.bomb_miss_penalty = -REWARD_SMALL
        self.bomb_hit_reward = REWARD_MEDIUM

    def clear(self):
        super().clear()
        self._visted_locations.clear()

    def critique_gameplay(self, old : Dict[str, int], new : Dict[str, int], rewards : Dict[str, float]):
        """
        Critiques the gameplay by comparing the old and new states and the rewards obtained.

        Args:
            old (Dict[str, int]): The old state of the game.
            new (Dict[str, int]): The new state of the game.
            rewards (Dict[str, float]): The rewards obtained during gameplay.
        """
        if not self._visted_locations:
            self.__mark_visited(new['level'], new['location'])

        # triforce
        self.critique_triforce(old, new, rewards)

        # combat
        self.critique_block(old, new, rewards)
        self.critique_attack(old, new, rewards)
        self.critique_item_usage(old, new, rewards)
        self.critique_aligned_enemy(old, new, rewards)

        # items
        self.critique_item_pickup(old, new, rewards)
        self.critique_key_pickup_usage(old, new, rewards)
        self.critique_equipment_pickup(old, new, rewards)

        # movement
        self.critique_location_discovery(old, new, rewards)
        self.critique_movement(old, new, rewards)

        # health - must be last
        self.critique_health_change(old, new, rewards)

    # reward helpers, may be overridden
    def critique_equipment_pickup(self, old, new, rewards):
        """
        Critiques the pickup of equipment items.

        Args:
            old (Dict[str, int]): The old state of the game.
            new (Dict[str, int]): The new state of the game.
            rewards (Dict[str, float]): The rewards obtained during gameplay.
        """
        if not self.equipment_reward:
            return

        self.__check_one_item(old, new, rewards, 'sword')
        self.__check_one_item(old, new, rewards, 'arrows')
        self.__check_one_item(old, new, rewards, 'bow')
        self.__check_one_item(old, new, rewards, 'candle')
        self.__check_one_item(old, new, rewards, 'whistle')
        self.__check_one_item(old, new, rewards, 'food')
        self.__check_one_item(old, new, rewards, 'potion')
        self.__check_one_item(old, new, rewards, 'magic_rod')
        self.__check_one_item(old, new, rewards, 'raft')
        self.__check_one_item(old, new, rewards, 'magic_book')
        self.__check_one_item(old, new, rewards, 'ring')
        self.__check_one_item(old, new, rewards, 'step_ladder')
        self.__check_one_item(old, new, rewards, 'magic_key')
        self.__check_one_item(old, new, rewards, 'power_bracelet')
        self.__check_one_item(old, new, rewards, 'letter')
        self.__check_one_item(old, new, rewards, 'regular_boomerang')
        self.__check_one_item(old, new, rewards, 'magic_boomerang')
        self.__check_one_item(old, new, rewards, 'compass')
        self.__check_one_item(old, new, rewards, 'map')
        self.__check_one_item(old, new, rewards, 'compass9')
        self.__check_one_item(old, new, rewards, 'map9')

    def __check_one_item(self, old, new, rewards, item):
        if old[item] < new[item]:
            rewards[f'reward-{item}-gained'] = self.equipment_reward

    def critique_key_pickup_usage(self, old, new, rewards):
        """
        Critiques the pickup and usage of keys.

        Args:
            old (Dict[str, int]): The old state of the game.
            new (Dict[str, int]): The new state of the game.
            rewards (Dict[str, float]): The rewards obtained during gameplay.
        """
        old_hearts = get_heart_halves(old)
        new_hearts = get_heart_halves(new)

        if old['keys'] > new['keys']:
            rewards['reward-used-key'] = self.key_reward
        elif old['keys'] < new['keys'] and old_hearts <= new_hearts:
            rewards['reward-gained-key'] = self.key_reward

    def critique_item_pickup(self, old, new, rewards):
        """
        Critiques the pickup of items.

        Args:
            old (Dict[str, int]): The old state of the game.
            new (Dict[str, int]): The new state of the game.
            rewards (Dict[str, float]): The rewards obtained during gameplay.
        """
        if old['rupees_to_add'] < new['rupees_to_add']:
            rewards['reward-gained-rupees'] = self.rupee_reward

        if old['bombs'] != 0 < new['bombs'] == 0:
            rewards['reward-gained-bombs'] = self.bomb_pickup_reward

    def critique_health_change(self, old, new, rewards):
        """
        Critiques the change in health.

        Args:
            old (Dict[str, int]): The old state of the game.
            new (Dict[str, int]): The new state of the game.
            rewards (Dict[str, float]): The rewards obtained during gameplay.
        """
        old_hearts = get_heart_halves(old)
        new_hearts = get_heart_halves(new)

        if get_heart_containers(old) < get_heart_containers(new):
            rewards['reward-gained-heart-container'] = self.heart_container_reward
        elif new_hearts > old_hearts:
            rewards['reward-gaining-health'] = self.health_gained_reward
        elif new_hearts < old_hearts:
            rewards['penalty-losing-health'] = self.health_lost_penalty

            if self.wipeout_reward_on_hits:
                for key, value in rewards.items():
                    if value > 0:
                        rewards[key] = 0

    def critique_triforce(self, old, new, rewards):
        """
        Critiques the acquisition of the triforce.

        Args:
            old (Dict[str, int]): The old state of the game.
            new (Dict[str, int]): The new state of the game.
            rewards (Dict[str, float]): The rewards obtained during gameplay.
        """
        if get_num_triforce_pieces(old) < get_num_triforce_pieces(new) or \
                (old["triforce_of_power"] == 0 and new["triforce_of_power"] == 1):
            rewards['reward-gained-triforce'] = self.triforce_reward

    def critique_block(self, old, new, rewards):
        """
        Critiques blocking of projectiles.

        Args:
            old (Dict[str, int]): The old state of the game.
            new (Dict[str, int]): The new state of tnhe game.
            rewards (Dict[str, float]): The rewards obtained during gameplay.
        """
        if is_deflecting(old, new):
            rewards['reward-block'] = self.block_projectile_reward

    def critique_aligned_enemy(self, old, new, rewards):
        """Critiques whether the agent fired sword beams towards an aligned enemy or not."""
        aligned_enemies = old['aligned_enemies']
        if aligned_enemies and old['beams_available']:
            match new['action']:
                case ActionType.MOVEMENT:
                    rewards['penalty-didnt-fire'] = self.didnt_fire_penalty

                case ActionType.ATTACK:
                    vector = aligned_enemies[0].vector
                    link_vector = new['link_direction'].to_vector()
                    dotproduct = np.dot(vector, link_vector)
                    if dotproduct > 0.8:
                        rewards['reward-fired-correctly'] = self.fired_correctly_reward

    def critique_attack(self, old, new, rewards):
        """Critiques attacks made by the player. """
        # pylint: disable=too-many-branches
        if 'beam_hits' in new and new['beam_hits']:
            rewards['reward-beam-hit'] = self.injure_kill_reward

        elif new['step_hits']:
            if not is_in_cave(new):
                if new['objective_kind'] == ObjectiveKind.FIGHT:
                    rewards['reward-hit'] = self.injure_kill_reward
                else:
                    rewards['reward-hit-move-room'] = self.inure_kill_movement_room_reward
            else:
                rewards['penalty-hit-cave'] = -self.injure_kill_reward

        elif new['action'] == ActionType.ATTACK:
            if not new['active_enemies']:
                rewards['penalty-attack-no-enemies'] = self.attack_no_enemies_penalty

            elif new['is_sword_frozen']:
                rewards['penalty-attack-offscreen'] = self.attack_miss_penalty

            elif new['active_enemies']:
                enemy_vectors = [enemy.vector for enemy in new['active_enemies'] if abs(enemy.distance) > 0]
                if enemy_vectors:
                    dotproducts = np.sum(new['link_direction'].to_vector() * enemy_vectors, axis=1)
                    if not np.any(dotproducts > np.sqrt(2) / 2):
                        rewards['penalty-attack-miss'] = self.attack_miss_penalty
                    elif not old['beams_available']:
                        distance = new['active_enemies'][0].distance
                        if distance > self.distance_threshold:
                            rewards['penalty-attack-miss'] = self.attack_miss_penalty

    def critique_item_usage(self, _, new, rewards):
        """
        Critiques the usage of items.

        Args:
            old (Dict[str, int]): The old state of the game.
            new (Dict[str, int]): The new state of the game.
            rewards (Dict[str, float]): The rewards obtained during gameplay.
        """
        if new['action'] == ActionType.ITEM:
            selected = new['selected_item']
            if selected == 0 and not new['regular_boomerang'] and not new['magic_boomerang']:
                rewards['used-null-item'] = self.used_null_item_penalty
            elif selected == 1:  # bombs
                total_hits = new.get('bomb1_hits', 0) + new.get('bomb2_hits', 0)
                if total_hits == 0:
                    rewards['penalty-bomb-miss'] = self.bomb_miss_penalty
                else:
                    rewards['reward-bomb-hit'] = min(self.bomb_hit_reward * total_hits, 1.0)

    def critique_location_discovery(self, old, new, rewards):
        """
        Critiques the discovery of new locations.

        Args:
            old (Dict[str, int]): The old state of the game.
            new (Dict[str, int]): The new state of the game.
            rewards (Dict[str, float]): The rewards obtained during gameplay.
        """
        prev = (old['level'], old['location'])
        curr = (new['level'], new['location'])

        if self.new_location_reward and prev != curr and not self.__has_visited(*curr):
            self.__mark_visited(*curr)
            rewards['reward-new-location'] = self.new_location_reward

    def critique_movement(self, old, new, rewards):
        """
        Critiques movement on the current screen.  This is the most difficult method to get right.  Movement in Zelda
        is complicated and unintended consequences are common.

        Args:
            old (dict): The old game state.
            new (dict): The new game state.
            rewards (dict): The rewards dictionary to update.

        Returns:
            None
        """
        # pylint: disable=too-many-branches, too-many-locals

        # Don't score movement if we moved to a new location or took damage.  The "movement" which occurs from
        # damage should never be rewarded, and it will be penalized by the health loss critic.
        if new['action'] != ActionType.MOVEMENT or new['took_damage'] or \
                old['location'] != new['location'] or is_in_cave(old) != is_in_cave(new):
            return

        # Did link run into a wall?
        if old['link_pos'] == new['link_pos']:
            rewards['penalty-wall-collision'] = self.wall_collision_penalty
            return

        self.critique_moving_into_danger(old, new, rewards)

        old_path = old.get("a*_path", [])
        new_path = new.get("a*_path", [])
        movement_direction = new['link_direction']

        # If an action put us into alignment for a sword beam shot, we should avoid penalizing the agent for this
        # move.  The agent shouldn't be able to get infinite rewards for moving into alignment since the enemy's motion
        # is fairly unpredictable
        moved_to_alignment = not old['aligned_enemies'] and new['aligned_enemies'] and old['health_full']

        # We check health_full here (not beams_available) because we still want to avoid penalizing the agent for
        # lining up the next shot, even if sword beams are already active.
        move_away_penalty = self.move_away_penalty if not moved_to_alignment else 0

        # There are places that can be clipped to that are impassible.  If this happens, we need to make sure not to
        # reward the agent for finding them.  This is because it's likely the agent will use this to break the
        # reward system for infinite rewards.
        if self.__any_impassible_tiles(new):
            rewards['penalty-bad-path'] = self.move_away_penalty  # don't take alignment into account for this

        # If we are headed to the same location as last time, simply check whether we made progress towards it.
        elif old_path and new_path and old_path[-1] == new_path[-1]:
            diff = len(old_path) - len(new_path)
            if diff > 0:
                rewards['reward-move-closer'] = self.move_closer_reward
            elif diff < 0:
                rewards['penalty-move-farther'] = move_away_penalty * abs(diff)

        # For most other cases, we calculate the progress towards the target using the manhattan distance towards
        # the second turn in the path.  That way we can reward any progress towards the target, even if the path
        # is the transpose of what A* picked.
        elif len(old_path) >= 2:
            target_y, target_x = _find_second_turn(old_path)
            target_tile = target_x, target_y

            old_link_tile = _xy_from_coord(old['link'].tile_coordinates[1])
            new_link_tile = _xy_from_coord(new['link'].tile_coordinates[1])

            progress = _get_progress(movement_direction, old_link_tile, new_link_tile, target_tile)

            if progress > 0:
                rewards['reward-move-closer'] = self.move_closer_reward
            elif progress < 0:
                rewards['penalty-move-farther'] = move_away_penalty

        # This should be relatively rare, but in cases where A* couldn't find a path to the target (usually because
        # a monster moved into a wall), we will reward simply moving closer.
        elif (target := new.get('objective_pos_or_dir', None)) is not None:
            old_link_pos = np.array(old.get('link_pos', (0, 0)), dtype=np.float32)
            new_link_pos = np.array(new.get('link_pos', (0, 0)), dtype=np.float32)

            if isinstance(target, Direction):
                if target == Direction.N:
                    progress = old_link_pos[1] - new_link_pos[1]
                elif target == Direction.S:
                    progress = new_link_pos[1] - old_link_pos[1]
                elif target == Direction.E:
                    progress = new_link_pos[0] - old_link_pos[0]
                elif target == Direction.W:
                    progress = old_link_pos[0] - new_link_pos[0]
            else:
                progress = _get_progress(movement_direction, old_link_pos, new_link_pos, target)

            if progress > 0:
                percent = min(abs(progress / self.movement_scale_factor), 1)
                rewards['reward-move-closer'] = self.move_closer_reward * percent
            else:
                rewards['penalty-move-farther'] = move_away_penalty

    def critique_moving_into_danger(self, old, new, rewards):
        """Critiques the agent for moving too close to an enemy or projectile.  These are added and subtracted
        independent of other movement rewards.  This ensures that even if the agent is moving in the right direction,
        it is still wary of moving too close to an enemy."""
        if not old['took_damage'] and not is_deflecting(old, new):
            warning_diff = new['link_warning_tiles'] - old['link_warning_tiles']
            danger_diff = new['link_danger_tiles'] - old['link_danger_tiles']

            if danger_diff > 0:
                rewards['penalty-dangerous-move'] = self.danger_tile_penalty
            elif danger_diff < 0:
                if len(old['active_enemies']) == len(new['active_enemies']):
                    rewards['reward-moved-to-safety'] = self.moved_to_safety_reward
            elif warning_diff > 0:
                rewards['penalty-risky-move'] = self.warning_tile_penalty
            elif warning_diff < 0:
                if len(old['active_enemies']) == len(new['active_enemies']):
                    rewards['reward-moved-to-safety'] = self.moved_to_safety_reward

    def __any_impassible_tiles(self, new):
        tile_states = new['tile_states']
        for tile in new['link'].tile_coordinates:
            if 0 <= tile[0] < tile_states.shape[0] and 0 <= tile[1] < tile_states.shape[1] \
                    and tile_states[tile] == TileState.IMPASSABLE.value:
                return True

        return False

    # state helpers, some states are calculated
    def __has_visited(self, level, location):
        return (level, location) in self._visted_locations

    def __mark_visited(self, level, location):
        self._visted_locations.add((level, location))


class Dungeon1Critic(GameplayCritic):
    """Critic specifically for dungeon 1."""
    def __init__(self):
        super().__init__()

        self.health_change_reward = REWARD_LARGE
        self.leave_dungeon_penalty = -REWARD_MAXIMUM
        self.leave_early_penalty = -REWARD_MAXIMUM
        self.seen = set()
        self.health_lost = 0

    def clear(self):
        super().clear()
        self.seen.clear()
        self.health_lost = 0

    def critique_location_discovery(self, old: Dict[str, int], new: Dict[str, int], rewards: Dict[str, float]):
        """
        Critiques the location discovery based on the old and new states and assigns rewards or penalties accordingly.

        Args:
            old (Dict[str, int]): The old state containing information about the previous location.
            new (Dict[str, int]): The new state containing information about the current location.
            rewards (Dict[str, float]): The rewards dictionary to update with rewards or penalties.

        Returns:
            None
        """
        if new['level'] != 1:
            rewards['penalty-left-dungeon'] = self.leave_dungeon_penalty
        elif old['location'] != new['location']:
            if old['location_objective'] == new['location']:
                rewards['reward-new-location'] = self.new_location_reward
            else:
                rewards['penalty-left-early'] = self.leave_early_penalty

    def critique_key_pickup_usage(self, old, new, rewards):
        if old['keys'] > new['keys'] and new['location'] == 0x73 and new['location_objective'] != 0x63:
            pass # do not give a reward for prematurely using a key
        else:
            super().critique_key_pickup_usage(old, new, rewards)


class Dungeon1BombCritic(Dungeon1Critic):
    """Critic specifically for dungeon 1 with bombs."""
    def __init__(self):
        super().__init__()
        self.bomb_miss_penalty = -REWARD_SMALL
        self.score = 0

    def clear(self):
        super().clear()
        self.score = 0
        self.bomb_miss_penalty = -REWARD_SMALL

    def set_score(self, old : Dict[str, int], new : Dict[str, int]):
        if new['action'] == ActionType.ITEM:
            selected = new['selected_item']
            # bombs
            if selected == 1:
                hits = new.get('bomb1_hits', 0) + new.get('bomb2_hits', 0)
                if hits:
                    self.score += hits
                else:
                    self.score -= 1

        new['score'] = self.score

class Dungeon1BossCritic(Dungeon1Critic):
    """Critic specifically for dungeon 1 with the boss."""
    def clear(self):
        super().clear()
        self.move_closer_reward = REWARD_SMALL
        self.move_away_penalty = -REWARD_SMALL
        self.injure_kill_reward = REWARD_LARGE
        self.health_lost_penalty = -REWARD_SMALL

OVERWORLD1_WALK = set([0x77, 0x78, 0x67, 0x68, 0x58, 0x48, 0x38, 0x37])
OVERWORLD2_WALK = set([0x37, 0x38, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x3d, 0x4d, 0x4c, 0x3c])
OVERWORLD2A_WALK = set([0x37, 0x38, 0x48, 0x49, 0x59, 0x5a, 0x5b, 0x5c, 0x5d, 0x4d, 0x4c, 0x3c])

class OverworldCritic(GameplayCritic):
    """Critic specifically for overworld 1."""
    def __init__(self):
        super().__init__()

        self.seen = set()

        self.left_allowed_area_penalty = -REWARD_LARGE
        self.left_without_sword_penalty = -REWARD_LARGE
        self.leave_early_penalty = -REWARD_MAXIMUM
        self.entered_cave_penalty = -REWARD_LARGE
        self.equipment_reward = None
        self.health_lost = 0

    def clear(self):
        super().clear()
        self.seen.clear()
        self.equipment_reward = None
        self.health_lost = 0

    def critique_location_discovery(self, old, new, rewards):
        if old['location'] != new['location']:
            if old['location_objective'] and old['location_objective'] != new['location']:
                rewards['penalty-left-early'] = self.leave_early_penalty
                return

            if old['objective_kind'] == ObjectiveKind.ENTER_CAVE:
                rewards['penalty-left-early'] = self.leave_early_penalty
                return

        level = new['level']
        location = new['location']
        triforce_pieces = get_num_triforce_pieces(new)

        if not is_in_cave(old) and location == 0x77 and is_in_cave(new):
            rewards['penalty-entered-cave'] = self.entered_cave_penalty

        elif level == 0:
            if location not in self._get_allowed_rooms(triforce_pieces):
                rewards['penalty-left-allowed-area'] = self.left_allowed_area_penalty

            elif old['location'] == 0x77 and location != 0x77 and not new['sword']:
                rewards['penalty-no-sword'] = self.left_without_sword_penalty

            else:
                super().critique_location_discovery(old, new, rewards)

        elif level == triforce_pieces + 1:
            # don't forget to reward for reaching the correct dungeon
            super().critique_location_discovery(old, new, rewards)

        else:
            rewards['penalty-left-allowed-area'] = self.left_allowed_area_penalty

    def _get_allowed_rooms(self, triforce_pieces):
        match triforce_pieces:
            case 0:
                return OVERWORLD1_WALK

            case 1:
                return OVERWORLD2_WALK

            case _:
                raise NotImplementedError("No support for more than 1 triforce piece")

class OverworldSwordCritic(GameplayCritic):
    """Critic specifically for the beginning of the game up through grabbing the first sword."""
    def __init__(self):
        super().__init__()

        self.cave_tranistion_reward = REWARD_LARGE
        self.cave_transition_penalty = -REWARD_MAXIMUM
        self.new_location_reward = REWARD_LARGE

    def critique_location_discovery(self, old, new, rewards):
        # entered cave
        if not is_in_cave(old) and is_in_cave(new):
            if new['sword']:
                rewards['penalty-reentered-cave'] = self.cave_transition_penalty
            else:
                rewards['reward-entered-cave'] = self.cave_tranistion_reward

        # left cave
        elif is_in_cave(old) and not is_in_cave(new):
            if new['sword']:
                rewards['reward-left-cave'] = self.cave_tranistion_reward
            else:
                rewards['penalty-left-cave-early'] = self.cave_transition_penalty

        elif new['location'] != 0x77:
            if new['sword']:
                rewards['reward-new-location'] = self.new_location_reward
            else:
                rewards['penalty-left-scenario'] = -self.new_location_reward

    def set_score(self, old : Dict[str, int], new : Dict[str, int]):
        score = 0
        if is_in_cave(new):
            score += 1

            if new['sword']:
                score += 1

        else:
            if new['sword']:
                score += 3

            if new['location'] != 0x77:
                score += 1

        new['score'] = score

CACHE_CAPACITY = 256
MOVEMENT_SCALE_FACTOR = 9.0

WRONG_ROOM_PENALTY = -REWARD_MAXIMUM
MOVE_CLOSER_REWARD = REWARD_TINY
MOVE_FURTHER_PENALTY = -REWARD_TINY - REWARD_MINIMUM
DAMAGE_PENALTY = -REWARD_LARGE
NO_MOVEMENT_PENALTY = -REWARD_LARGE
ENEMY_HIT_REWARD = REWARD_LARGE
DANGER_SENSE_REWARD = REWARD_SMALL
BLOCK_REWARD = REWARD_MAXIMUM

MAX_DANGER_DISTANCE = 240

class RoomInformation:
    """Information about a single Zelda room."""
    def __init__(self, level, location, tiles):
        self.level = level
        self.location = location
        self._pathing_cache = OrderedDict()
        self.wavefront = RoomWavefront((level, location), tiles)

def get_aligned_tile_dist(enemy : ZeldaObject, link : ZeldaObject, direction : SelectedDirection):
    """Get the distance to the nearest aligned enemy in the given direction."""
    match direction:
        case SelectedDirection.N:
            dir_vect = np.array([-1, 0], dtype=np.float32)
            link_tile = link.tile_coordinates[0] + np.array([0, 1])
        case SelectedDirection.S:
            dir_vect = np.array([1, 0], dtype=np.float32)
            link_tile = link.tile_coordinates[0]
        case SelectedDirection.W:
            dir_vect = np.array([0, -1], dtype=np.float32)
            link_tile = link.tile_coordinates[0] + np.array([1, 0])
        case SelectedDirection.E:
            dir_vect = np.array([0, 1], dtype=np.float32)
            link_tile = link.tile_coordinates[0]
        case _:
            raise ValueError(f"Invalid direction: {direction}")

    vector = np.array(enemy.tile_coordinates[0], dtype=np.float32) - np.array(link_tile, dtype=np.float32)
    tile_distance = np.dot(vector, dir_vect)

    if tile_distance < 0:
        return None

    if direction in (SelectedDirection.N, SelectedDirection.S):
        link_tiles = link.tile_coordinates[0][1]
        link_tiles = link_tiles, link_tiles + 1
        enemy_tiles = enemy.tile_coordinates[0][1]
        enemy_tiles = enemy_tiles, enemy_tiles + 1
        if tile_distance > 3:
            enemy_tiles = *enemy_tiles, enemy_tiles[0] - 1, enemy_tiles[1] + 1

    elif direction in (SelectedDirection.W, SelectedDirection.E):
        link_tiles = link.tile_coordinates[0][0]
        link_tiles = link_tiles, link_tiles + 1
        enemy_tiles = enemy.tile_coordinates[0][0]
        enemy_tiles = enemy_tiles, enemy_tiles + 1
        if tile_distance > 3:
            enemy_tiles = *enemy_tiles, enemy_tiles[0] - 1, enemy_tiles[1] + 1

    if any(tile in link_tiles for tile in enemy_tiles):
        # found overlap
        return tile_distance

    return None

def calculate_danger_sense_accuracy(link, enemies, direction):
    """Calculates how much reward should be given for the agent sensing danger in a particular direction."""
    closest_point = 3
    farthest_point = 14

    reward = 0

    if direction == SelectedDirection.NONE:
        any_danger = False
        for i in range(4):
            d = SelectedDirection(i)
            any_danger = any_danger or any(get_aligned_tile_dist(enemy, link, d) is not None for enemy in enemies)

        return -1.0 if any_danger else 1.0

    distances = [get_aligned_tile_dist(enemy, link, direction) for enemy in enemies]
    distances = [d for d in distances if d is not None]
    if distances:
        closest = min(distances)
        reward = 0.1 + (closest - farthest_point) * 0.9 / (closest_point - farthest_point)
        reward = np.clip(reward, 0.1, 1.0)
        return reward

    return -1.0

def is_walkable(tiles, x, y):
    """Check if a tile is walkable."""
    # check if the coordinates are within tiles
    if x < 0 or x >= tiles.shape[0] or y < 0 or y >= tiles.shape[1]:
        return True

    tile = tiles[x, y]
    return is_tile_walkable(tile)


class MultiHeadCritic(gym.Wrapper):
    """Critic for multiple objectives."""
    def __init__(self, env):
        super().__init__(env)
        self._last = None
        self._room_info = {}
        self.next_movement_reward = None
        self.pathfinding_mask = None

    def reset(self, *, seed = None, options = None):
        obs, info = super().reset(seed=seed, options=options)
        info['wavefront'] = self._get_wavefront_values(info)
        self._last = info
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        info['wavefront'] = self._get_wavefront_values(info)

        rewards = self.critique_gameplay(action, self._last, info)
        self._last = info
        return obs, rewards, terminated, truncated, info

    def critique_gameplay(self, action, last_info, info):
        """Critiques the gameplay by comparing the old and new states and the rewards obtained."""
        pathfinding, danger_sense, selection = action.squeeze(0).tolist()
        pathfinding = SelectedDirection(pathfinding)
        danger_sense = SelectedDirection(danger_sense)
        selection = SelectedAction(selection)

        movement_reward = self._critique_movement(selection, pathfinding, last_info, info)
        danger_sense_reward = self._critique_danger_sense(danger_sense, last_info)
        action_reward = self._critique_action()

        info['masks'] = self.pathfinding_mask, self._get_danger_sense_mask(info), \
                            self._get_action_mask(info)

        return movement_reward, danger_sense_reward, action_reward

    def _critique_action(self):
        return 0.0

    def _critique_movement(self, selection, direction, old, new):
        # Make sure we didn't select "no direction" and "movement".  This isn't an allowed state, so it shouldn't
        # happen, but we should check to make sure.
        if selection == SelectedAction.MOVEMENT and direction is None:
            return NO_MOVEMENT_PENALTY

        # if we moved locations, make sure it was to the right place
        if old['location'] != new['location']:
            if old['location_objective'] == new['location']:
                return MOVE_CLOSER_REWARD
            return WRONG_ROOM_PENALTY

        # if we blocked a projectile, reward the agent no matter the wavefront
        if selection == SelectedAction.MOVEMENT and is_deflecting(old, new):
            return BLOCK_REWARD

        # If link ran into a wall, penalize the agent, but also mask the direction so we get better actions to
        # learn from
        if selection == SelectedAction.MOVEMENT and old['link'].position == new['link'].position:
            if self.pathfinding_mask is None:
                self.pathfinding_mask = torch.ones(4, dtype=torch.float32)
            self.pathfinding_mask[direction_to_action(direction)] = 0
            return NO_MOVEMENT_PENALTY

        self.pathfinding_mask = None

        y, x = new['link'].tile_coordinates[0]
        ny, nx = self._apply_direction(y, x, direction)

        # Did we move within the map?
        wavefront = old['wavefront']
        if 0 <= ny <= wavefront.shape[0] and 0 <= nx <= wavefront.shape[1]:
            if wavefront[y, x] < wavefront[ny, nx]:
                return MOVE_FURTHER_PENALTY
            if wavefront[y, x] == wavefront[ny, nx]:
                return 0.0

            old_link_pos = np.array(old['link_pos'], dtype=np.float32)
            new_link_pos =  np.array(new['link_pos'], dtype=np.float32)
            match direction:
                case SelectedDirection.N:
                    dir_vec = np.array([0, -1], dtype=np.float32)
                case SelectedDirection.S:
                    dir_vec = np.array([0, 1], dtype=np.float32)
                case SelectedDirection.W:
                    dir_vec = np.array([-1, 0], dtype=np.float32)
                case SelectedDirection.E:
                    dir_vec = np.array([1, 0], dtype=np.float32)
                case _:
                    raise ValueError(f"Invalid direction: {direction}")

            movement_vect = new_link_pos - old_link_pos
            dist_in_dir = np.dot(movement_vect, dir_vec)
            return dist_in_dir / MOVEMENT_SCALE_FACTOR * MOVE_CLOSER_REWARD

        # We moved off the map.  See if we are moving in the right direction
        pos_dir = old['position_or_direction']
        if isinstance(pos_dir, Direction) and pos_dir.name == direction.name:
            return MOVE_CLOSER_REWARD

        return MOVE_FURTHER_PENALTY

    def _apply_direction(self, y, x, direction):
        match direction:
            case SelectedDirection.N:
                return y - 1, x
            case SelectedDirection.S:
                return y + 1, x
            case SelectedDirection.W:
                return y, x - 1
            case SelectedDirection.E:
                return y, x + 1
            case _:
                raise ValueError(f"Invalid direction: {direction}")

    def _critique_danger_sense(self, direction, old):
        factor = calculate_danger_sense_accuracy(old['link'], old['active_enemies'], direction)
        return DANGER_SENSE_REWARD * factor

    def _get_danger_sense_mask(self, info):
        if not info['active_enemies']:
            mask = torch.zeros(5, dtype=torch.float32)
            mask[4] = 1

        else:
            mask = torch.ones(5, dtype=torch.float32)

        return mask

    def _get_action_mask(self, info):
        if not info['active_enemies'] or is_sword_frozen(info):
            mask = torch.zeros(3, dtype=torch.float32)
            mask[SelectedAction.MOVEMENT.value] = 1
        else:
            mask = torch.ones(3, dtype=torch.float32)
            if not info['beams_available']:
                mask[SelectedAction.BEAMS.value] = 0

        return mask

    def _get_room_info(self, info) -> RoomInformation:
        level = info['level']
        location = info['location']
        key = level, location
        if key not in self._room_info:
            self._room_info[key] = RoomInformation(level, location, info['tiles'])

        return self._room_info[key]

    def _get_wavefront_values(self, info):
        room_info = self._get_room_info(info)
        wavefront = room_info.wavefront.get_wavefront(info)
        return wavefront.values
