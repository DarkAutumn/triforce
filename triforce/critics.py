"""Gameplay critics for Zelda."""

from typing import Dict, OrderedDict, Tuple
import numpy as np
import gymnasium as gym
import torch

from .astar import a_star
from .ml_torch import action_to_direction, SelectedAction, direction_to_action
from .objective_selector import ObjectiveKind
from .zelda_wrapper import ActionType
from .zelda_game import Direction, TileState, ZeldaSoundsPulse1, get_heart_containers, get_heart_halves, \
    get_num_triforce_pieces, is_in_cave, is_sword_frozen

REWARD_MINIMUM = 0.01
REWARD_TINY = 0.05
REWARD_SMALL = 0.25
REWARD_MEDIUM = 0.5
REWARD_LARGE = 0.75
REWARD_MAXIMUM = 1.0

def _manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def _get_progress(direction, old_link_pos, new_link_pos, target):
    direction_vector = direction.to_vector()

    disp = new_link_pos - old_link_pos
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
        if self._is_deflecting(old, new):
            rewards['reward-block'] = self.block_projectile_reward

    def _is_deflecting(self, old, new):
        arrow_deflected = ZeldaSoundsPulse1.ArrowDeflected.value
        return new['sound_pulse_1'] & arrow_deflected and (old['sound_pulse_1'] & arrow_deflected) != arrow_deflected

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
        if not old['took_damage'] and not self._is_deflecting(old, new):
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

MOVE_CLOSER_REWARD = REWARD_TINY
MOVE_FURTHER_PENALTY = -REWARD_TINY - REWARD_MINIMUM
DAMAGE_PENALTY = -REWARD_LARGE
NO_MOVEMENT_PENALTY = -REWARD_LARGE
ENEMY_HIT_REWARD = REWARD_LARGE
NO_DANGER_REWARD = REWARD_MEDIUM
DANGER_SENSE_REWARD = REWARD_LARGE

MAX_DANGER_DISTANCE = 240

class PathReward:
    def __init__(self, path):
        self.weight = len(path)
        self.rewards = [None] * 4
        self.second_turn = _find_second_turn(path)

class PathingData:
    """Pathing rewards for a single location."""
    def __init__(self, room_info, target, tile_map):
        self._tile_rewards = {}
        self.room_info = room_info
        self.target = target
        self.tile_map = tile_map

    def get_pathing_rewards(self, start_tile) -> PathReward:
        data = self._tile_rewards.get(start_tile, None)
        if data is None:
            path = a_star(start_tile, self.tile_map, self.target)

            pathing = self if path[-1] == self.target else self.room_info.get_pathing_data(path[-1], self.tile_map)
            pathing.update(path)

        return data

    def update(self, path):
        for i in range(len(path) - 1):
            tile = path[i]
            next_tile = path[i + 1]

            if tile not in self._tile_rewards:
                rewards = PathReward(path)
                self._tile_rewards[tile] = rewards
            else:
                rewards = self._tile_rewards[tile]

            direction = Direction.from_position_change(tile, next_tile)
            rewards[direction_to_action(direction)] = MOVE_CLOSER_REWARD

            opposite = direction.opposite()
            rewards[direction_to_action(opposite)] = MOVE_FURTHER_PENALTY

class RoomInformation:
    """Information about a single Zelda room."""
    def __init__(self, level, location):
        self.level = level
        self.location = location
        self._pathing_cache = OrderedDict()

    def get_pathing_data(self, target, tile_map) -> PathingData:
        if target not in self._pathing_cache:
            if len(self._pathing_cache) >= CACHE_CAPACITY:
                self._pathing_cache.popitem(last=False)

            self._pathing_cache[target] = PathingData(self, target, tile_map)

        return self._pathing_cache[target]


def calculate_alignment_factor(distance, alignment_distance):
    """Calculates alignment factor based on how close the enemy is to the beam path."""
    if distance <= 8:
        max_alignment_distance = 8
    else:
        max_alignment_distance = 8 + (8 * (distance - 8) / (175 - 8))

    alignment_factor = max(0, 1 - (alignment_distance / max_alignment_distance))
    return alignment_factor

def calculate_danger_sense_accuracy(link_pos, enemies, direction):
    """Calculates how much reward should be given for the agent sensing danger in a particular direction."""
    max_distance = 175
    min_distance = 8

    max_reward = 0

    for enemy_pos in enemies:
        dx = enemy_pos[0] - link_pos[0]
        dy = enemy_pos[1] - link_pos[1]

        distance = np.sqrt(dx**2 + dy**2)

        if distance > max_distance:
            continue

        # Distance reward
        if distance <= min_distance:
            distance_reward = 1.0
        else:
            distance_reward = np.clip(1.0 - (distance - min_distance) / (max_distance - min_distance) * 0.9, 0.1, 1.0)

        # Calculate alignment distance for the chosen direction
        if direction == 'up' and dy < 0:
            alignment_distance = abs(dx)
        elif direction == 'down' and dy > 0:
            alignment_distance = abs(dx)
        elif direction == 'left' and dx < 0:
            alignment_distance = abs(dy)
        elif direction == 'right' and dx > 0:
            alignment_distance = abs(dy)
        else:
            alignment_distance = float('inf')

        if alignment_distance != float('inf'):
            if distance <= 8:
                max_alignment_distance = 8
            else:
                max_alignment_distance = 8 + (8 * (distance - 8) / (175 - 8))

            alignment_factor = max(0, 1 - (alignment_distance / max_alignment_distance))
            total_reward = distance_reward * alignment_factor
            max_reward = max(max_reward, total_reward)

    if max_reward <= 0.1:
        max_reward = -1.0

    # Normalize reward to be between 0.1 and 1.0
    max_reward = np.clip(max_reward, 0.1, 1.0)

    return max_reward

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
        self._last = info
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        rewards = self._critique_gameplay(action, self._last, info)
        return obs, rewards, terminated, truncated, info

    def critique_gameplay(self, action, last_info, info):
        """Critiques the gameplay by comparing the old and new states and the rewards obtained."""
        movement_reward = self._critique_movement(action, last_info, info)
        danger_sense_reward = self._critique_danger_sense(action, last_info, info)
        action_reward = self._critique_action(action, last_info, info)

        info['movement_mask'] = self.pathfinding_mask
        info['danger_sense_mask'] = self._get_danger_sense_mask(info)
        info['action_mask'] = self._get_action_mask(info)

        return movement_reward, action_reward, danger_sense_reward

    def _critique_movement(self, action, old, new):
        movement_action = action[0]
        direction = action_to_direction(movement_action)
        selection = action[2]

        # Make sure we didn't select "no direction" and "movement".  This isn't an allowed state, so it shouldn't
        # happen, but we should check to make sure.
        if selection == SelectedAction.MOVEMENT and direction is None:
            return NO_MOVEMENT_PENALTY

        # If link ran into a wall, penalize the agent, but also mask the direction so we get better actions to
        # learn from
        if old['link'].position == new['link'].position:
            if self.pathfinding_mask is None:
                self.pathfinding_mask = torch.ones(4, dtype=torch.float32)
            self.pathfinding_mask[movement_action] = 0
            return NO_MOVEMENT_PENALTY

        self.pathfinding_mask = None

        # Ok, we have a movement action, let's see if went in the right direction
        last_path_reward, new_path_reward = self._step_movement_reward(new)
        directional_rewards = last_path_reward.rewards[movement_action]

        # if we don't have a directional reward, it means the actor did not follow the path,
        # we can see if this is a good or bad thing by comparing the last and new path weight
        if directional_rewards is None:
            if last_path_reward.weight < new_path_reward.weight:
                reward =  MOVE_FURTHER_PENALTY
            else:
                reward = MOVE_CLOSER_REWARD

            directional_rewards = last_path_reward.rewards[movement_action] = reward

        # Scale movement in the right direction to make sure we don't over-reward the agent
        if directional_rewards > 0:
            old_pos = old['link'].position
            new_pos = new['link'].position
            progress = _get_progress(direction, old_pos, new_pos, last_path_reward.second_turn)
            directional_rewards *= min(abs(progress / MOVEMENT_SCALE_FACTOR), 1)

        return directional_rewards

    def _step_movement_reward(self, new) -> Tuple[PathReward, PathReward]:
        last_movement_reward = self.next_movement_reward
        room_info = self._get_room_info(new['level'], new['location'])
        pathing_data = room_info.get_pathing_data(new['objective_pos_or_dir'], new['tile_states'])
        self.next_movement_reward = pathing_data.get_pathing_rewards(new['link'].tile_coordinates[1])
        return last_movement_reward, self.next_movement_reward

    def _critique_danger_sense(self, action, old, new):
        selected_is_attack = action[2] != SelectedAction.MOVEMENT
        if selected_is_attack:
            if new['step_hits']:
                return ENEMY_HIT_REWARD

            return -ENEMY_HIT_REWARD

        direction = action_to_direction(action[1])
        if direction is None:
            if not new['active_enemies']:
                return NO_DANGER_REWARD

            return -NO_DANGER_REWARD

        factor = calculate_danger_sense_accuracy(old['link'].position, old['active_enemies'], direction)
        if factor > 0:
            return DANGER_SENSE_REWARD * factor

        return -DANGER_SENSE_REWARD

    def _get_danger_sense_mask(self, info):
        if not info['active_enemies']:
            mask = torch.zeros(5, dtype=torch.float32)
            mask[4] = 1

        else:
            mask = torch.ones(5, dtype=torch.float32)

        return mask

    def _get_action_mask(self, info):
        if is_sword_frozen(info):
            mask = torch.zeros(3, dtype=torch.float32)
            mask[SelectedAction.MOVEMENT.value] = 1
        else:
            mask = torch.ones(3, dtype=torch.float32)
            if not info['beams_available']:
                mask[SelectedAction.BEAMS.value] = 0

        return mask

    def _get_room_info(self, level, location):
        key = (level, location)
        if key not in self._room_info:
            self._room_info[key] = RoomInformation(level, location)

        return self._room_info[key]
