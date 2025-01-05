"""Gameplay critics for Zelda."""

from enum import Enum
from typing import Dict
import numpy as np

from .objective_selector import ObjectiveKind
from .zelda_cooldown_handler import ActionType
from .zelda_game import TileState

from .zelda_enums import Direction, SelectedEquipment, SwordKind, ZeldaAnimationId, AnimationState
from .game_state_change import ZeldaStateChange

REWARD_MINIMUM = 0.01
REWARD_TINY = 0.05
REWARD_SMALL = 0.25
REWARD_MEDIUM = 0.5
REWARD_LARGE = 0.75
REWARD_MAXIMUM = 1.0

class ZeldaCritic:
    """Base class for Zelda gameplay critics."""
    def __init__(self):
        self.health_lost = 0

    def clear(self):
        """Called when the environment is reset to clear any saved state."""
        self.health_lost = 0

    def critique_gameplay(self, state_change : ZeldaStateChange, rewards: Dict[str, float]):
        """Critiques the gameplay by comparing the old and new states and the rewards obtained."""
        raise NotImplementedError()

    def get_score(self, state_change : ZeldaStateChange):
        """Override to set info['score']"""

        diff = state_change.current.link.health - state_change.previous.link.health
        if diff < 0:
            self.health_lost += diff

        return self.health_lost

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

    def critique_gameplay(self, state_change : ZeldaStateChange, rewards : Dict[str, float]):
        """Critiques the gameplay by comparing the old and new states and the rewards obtained."""
        curr = state_change.current
        if not self._visted_locations:
            self.__mark_visited(curr.level, curr.location)

        # triforce
        self.critique_triforce(state_change, rewards)

        # combat
        self.critique_block(state_change, rewards)
        self.critique_attack(state_change, rewards)
        self.critique_item_usage(state_change, rewards)
        self.critique_aligned_enemy(state_change, rewards)

        # items
        self.critique_item_pickup(state_change, rewards)
        self.critique_key_pickup_usage(state_change, rewards)
        self.critique_equipment_pickup(state_change, rewards)

        # movement
        self.critique_location_discovery(state_change, rewards)
        self.critique_movement(state_change, rewards)

        # health - must be last
        self.critique_health_change(state_change, rewards)

    # reward helpers, may be overridden
    def critique_equipment_pickup(self, state_change : ZeldaStateChange, rewards):
        """Critiques the pickup of equipment items."""
        if not self.equipment_reward:
            return

        self.__check_one_equipment(state_change, rewards, 'sword')
        self.__check_one_equipment(state_change, rewards, 'arrows')
        self.__check_one_equipment(state_change, rewards, 'bow')
        self.__check_one_equipment(state_change, rewards, 'candle')
        self.__check_one_equipment(state_change, rewards, 'whistle')
        self.__check_one_equipment(state_change, rewards, 'food')
        self.__check_one_equipment(state_change, rewards, 'potion')
        self.__check_one_equipment(state_change, rewards, 'magic_rod')
        self.__check_one_equipment(state_change, rewards, 'raft')
        self.__check_one_equipment(state_change, rewards, 'book')
        self.__check_one_equipment(state_change, rewards, 'ring')
        self.__check_one_equipment(state_change, rewards, 'ladder')
        self.__check_one_equipment(state_change, rewards, 'magic_key')
        self.__check_one_equipment(state_change, rewards, 'power_bracelet')
        self.__check_one_equipment(state_change, rewards, 'letter')
        self.__check_one_equipment(state_change, rewards, 'boomerang')
        self.__check_one_equipment(state_change, rewards, 'compass')
        self.__check_one_equipment(state_change, rewards, 'map')

    def __check_one_equipment(self, state_change : ZeldaStateChange, rewards, item):
        prev, curr = self.__get_equipment_change(state_change.previous.link, state_change.current.link, item)
        if prev < curr:
            rewards[f'reward-{item}-gained'] = self.equipment_reward

    def __get_equipment_change(self, prev_state, curr_state, item):
        prev = getattr(prev_state, item)
        curr = getattr(curr_state, item)

        if isinstance(prev, Enum):
            prev = prev.value
        elif isinstance(prev, bool):
            prev = int(prev)

        if isinstance(curr, Enum):
            curr = curr.value
        elif isinstance(curr, bool):
            curr = int(curr)
        return prev,curr

    def critique_key_pickup_usage(self, state_change : ZeldaStateChange, rewards):
        """Critiques the pickup and usage of keys."""
        prev_link = state_change.previous.link
        curr_link = state_change.current.link

        if prev_link.keys > curr_link.keys:
            rewards['reward-used-key'] = self.key_reward
        elif prev_link.keys < curr_link.keys:
            rewards['reward-gained-key'] = self.key_reward

    def critique_item_pickup(self, state_change : ZeldaStateChange, rewards):
        """Critiques the pickup of items."""
        prev, curr = state_change.previous, state_change.current
        if prev.rupees_to_add < curr.rupees_to_add:
            rewards['reward-gained-rupees'] = self.rupee_reward

        if prev.link.bombs != 0 < curr.link.bombs == 0:
            rewards['reward-gained-bombs'] = self.bomb_pickup_reward

    def critique_health_change(self, state_change : ZeldaStateChange, rewards):
        """Critiques the change in health."""
        prev_link, curr_link = state_change.previous.link, state_change.current.link
        if prev_link.max_health < curr_link.max_health:
            rewards['reward-gained-heart-container'] = self.heart_container_reward

        elif state_change.health_gained:
            rewards['reward-gaining-health'] = self.health_gained_reward

        elif state_change.health_lost:
            rewards['penalty-losing-health'] = self.health_lost_penalty

            if self.wipeout_reward_on_hits:
                for key, value in rewards.items():
                    if value > 0:
                        rewards[key] = 0

    def critique_triforce(self, state_change : ZeldaStateChange, rewards):
        """Critiques the acquisition of the triforce."""
        prev_link, curr_link = state_change.previous.link, state_change.current.link
        if prev_link.triforce_pieces < curr_link.triforce_pieces:
            rewards['reward-gained-triforce'] = self.triforce_reward

        if not prev_link.triforce_of_power and curr_link.triforce_of_power:
            rewards['reward-gained-triforce'] = self.triforce_reward

    def critique_block(self, state_change : ZeldaStateChange, rewards):
        """Critiques blocking of projectiles."""
        prev_link, curr_link = state_change.previous.link, state_change.current.link
        if not prev_link.is_blocking and curr_link.is_blocking:
            rewards['reward-block'] = self.block_projectile_reward

    def critique_aligned_enemy(self, state_change : ZeldaStateChange, rewards):
        """Critiques whether the agent fired sword beams towards an aligned enemy or not."""
        prev, curr = state_change.previous, state_change.current
        aligned_enemies = prev.aligned_enemies
        if aligned_enemies and prev.link.are_beams_available:
            match curr.action:
                case ActionType.MOVEMENT:
                    rewards['penalty-didnt-fire'] = self.didnt_fire_penalty

                case ActionType.ATTACK:
                    vector = aligned_enemies[0].vector
                    link_vector = curr.link.direction.to_vector()
                    dotproduct = np.dot(vector, link_vector)
                    if dotproduct > 0.8:
                        rewards['reward-fired-correctly'] = self.fired_correctly_reward

    def critique_attack(self, state_change : ZeldaStateChange, rewards):
        """Critiques attacks made by the player."""
        # pylint: disable=too-many-branches

        prev, curr = state_change.previous, state_change.current
        if state_change.hits and prev.link.are_beams_available \
                             and curr.link.get_animation_state(ZeldaAnimationId.BEAMS) != AnimationState.INACTIVE:
            rewards['reward-beam-hit'] = self.injure_kill_reward

        elif state_change.hits:
            if not curr.in_cave:
                if prev.objective_kind == ObjectiveKind.FIGHT:
                    rewards['reward-hit'] = self.injure_kill_reward
                else:
                    rewards['reward-hit-move-room'] = self.inure_kill_movement_room_reward
            else:
                rewards['penalty-hit-cave'] = -self.injure_kill_reward

        elif curr.action == ActionType.ATTACK:
            if not curr.enemies:
                rewards['penalty-attack-no-enemies'] = self.attack_no_enemies_penalty

            elif curr.link.is_sword_frozen:
                rewards['penalty-attack-offscreen'] = self.attack_miss_penalty

            elif (active_enemies := curr.active_enemies):
                enemy_vectors = [enemy.vector for enemy in active_enemies if abs(enemy.distance) > 0]
                if enemy_vectors:
                    link_vector = curr.link.direction.to_vector()
                    dotproducts = np.sum(link_vector * enemy_vectors, axis=1)
                    if not np.any(dotproducts > np.sqrt(2) / 2):
                        rewards['penalty-attack-miss'] = self.attack_miss_penalty
                    elif not prev.link.are_beams_available:
                        distance = active_enemies[0].distance
                        if distance > self.distance_threshold:
                            rewards['penalty-attack-miss'] = self.attack_miss_penalty

    def critique_item_usage(self, state_change : ZeldaStateChange, rewards):
        """Critiques the usage of items."""
        curr = state_change.current
        if curr.action == ActionType.ITEM:
            selected = curr.link.selected_equipment
            if selected == SelectedEquipment.NONE:
                rewards['used-null-item'] = self.used_null_item_penalty
            elif selected == SelectedEquipment.BOMBS:
                if state_change.hits == 0:
                    rewards['penalty-bomb-miss'] = self.bomb_miss_penalty
                else:
                    rewards['reward-bomb-hit'] = min(self.bomb_hit_reward * state_change.hits, 1.0)

    def critique_location_discovery(self, state_change : ZeldaStateChange, rewards):
        """Critiques the discovery of new locations."""
        prev = state_change.previous
        prev = (prev.location, prev.level)

        curr = state_change.current
        curr = (curr.location, curr.level)

        if self.new_location_reward and prev != curr and not self.__has_visited(*curr):
            self.__mark_visited(*curr)
            rewards['reward-new-location'] = self.new_location_reward

    def critique_movement(self, state_change : ZeldaStateChange, rewards):
        """
        Critiques movement on the current screen.  This is the most difficult method to get right.  Movement in Zelda
        is complicated and unintended consequences are common.
        """
        # pylint: disable=too-many-branches, too-many-locals

        prev = state_change.previous
        curr = state_change.current

        prev_link = prev.link
        curr_link = curr.link

        # Don't score movement if we moved to a new location or took damage.  The "movement" which occurs from
        # damage should never be rewarded, and it will be penalized by the health loss critic.
        if curr.action != ActionType.MOVEMENT or state_change.health_lost or prev.full_location != curr.full_location:
            return

        # Did link run into a wall?
        if prev_link.position == curr_link.position:
            rewards['penalty-wall-collision'] = self.wall_collision_penalty
            return

        self.critique_moving_into_danger(state_change, rewards)

        old_path = prev.get("a_star_path", [])
        new_path = curr.get("a_star_path", [])

        movement_direction = curr_link.direction

        # If an action put us into alignment for a sword beam shot, we should avoid penalizing the agent for this
        # move.  The agent shouldn't be able to get infinite rewards for moving into alignment since the enemy's motion
        # is fairly unpredictable
        moved_to_alignment = not prev.aligned_enemies and curr.aligned_enemies and prev_link.are_beams_available

        # We check health_full here (not beams_available) because we still want to avoid penalizing the agent for
        # lining up the next shot, even if sword beams are already active.
        move_away_penalty = self.move_away_penalty if not moved_to_alignment else 0

        # There are places that can be clipped to that are impassible.  If this happens, we need to make sure not to
        # reward the agent for finding them.  This is because it's likely the agent will use this to break the
        # reward system for infinite rewards.
        if self.__any_impassible_tiles(curr.tile_states, curr_link.tile_coordinates):
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
            target_y, target_x = self.__find_second_turn(old_path)
            target_tile = target_x, target_y

            old_link_tile = self.__xy_from_coord(prev_link.tile_coordinates[1])
            new_link_tile = self.__xy_from_coord(curr_link.tile_coordinates[1])

            progress = self.__get_progress(movement_direction, old_link_tile, new_link_tile, target_tile)

            if progress > 0:
                rewards['reward-move-closer'] = self.move_closer_reward
            elif progress < 0:
                rewards['penalty-move-farther'] = move_away_penalty

        # This should be relatively rare, but in cases where A* couldn't find a path to the target (usually because
        # a monster moved into a wall), we will reward simply moving closer.
        elif (target := curr.get('objective_pos_or_dir', None)) is not None:
            old_link_pos = np.array(prev_link.position, dtype=np.float32)
            new_link_pos = np.array(curr_link.position, dtype=np.float32)

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
                progress = self.__get_progress(movement_direction, old_link_pos, new_link_pos, target)

            if progress > 0:
                percent = min(abs(progress / self.movement_scale_factor), 1)
                rewards['reward-move-closer'] = self.move_closer_reward * percent
            else:
                rewards['penalty-move-farther'] = move_away_penalty

    def critique_moving_into_danger(self, state_change : ZeldaStateChange, rewards):
        """Critiques the agent for moving too close to an enemy or projectile.  These are added and subtracted
        independent of other movement rewards.  This ensures that even if the agent is moving in the right direction,
        it is still wary of moving too close to an enemy."""
        prev, curr = state_change.previous, state_change.current

        if not state_change.health_lost and not curr.link.is_blocking:
            warning_diff = curr.link_warning_tiles - prev.link_warning_tiles
            danger_diff = curr.link_danger_tiles - prev.link_danger_tiles

            if danger_diff > 0:
                rewards['penalty-dangerous-move'] = self.danger_tile_penalty
            elif danger_diff < 0:
                if len(prev.active_enemies) == len(curr.active_enemies):
                    rewards['reward-moved-to-safety'] = self.moved_to_safety_reward
            elif warning_diff > 0:
                rewards['penalty-risky-move'] = self.warning_tile_penalty
            elif warning_diff < 0:
                if len(prev.active_enemies) == len(curr.active_enemies):
                    rewards['reward-moved-to-safety'] = self.moved_to_safety_reward

    def __any_impassible_tiles(self, tile_states, tile_coordinates):
        for tile in tile_coordinates:
            if 0 <= tile[0] < tile_states.shape[0] and 0 <= tile[1] < tile_states.shape[1] \
                    and tile_states[tile] == TileState.IMPASSABLE.value:
                return True

        return False

    def __xy_from_coord(self, bottom_left):
        y, x = bottom_left
        old_link_tile = np.array([x, y], dtype=np.float32)
        return old_link_tile

    def __get_progress(self, direction, old_link_pos, new_link_pos, target):
        direction_vector = direction.to_vector()

        disp = new_link_pos - old_link_pos
        direction_movement = np.dot(disp, direction_vector) * direction_vector

        projected_new_pos = old_link_pos + direction_movement

        old_distance = self.__manhattan_distance(target, old_link_pos)
        new_distance = self.__manhattan_distance(target, projected_new_pos)

        return old_distance - new_distance

    def __manhattan_distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def __find_second_turn(self, path):
        turn = 0
        direction = self.__get_direction(path[0], path[1])
        for i in range(2, len(path)):
            old_index = path[i - 1]
            new_index = path[i]

            new_direction = self.__get_direction(old_index, new_index)
            if new_direction != direction:
                turn += 1
                direction = new_direction
                if turn == 2:
                    return old_index

        return path[-1]

    def __get_direction(self, old, new):
        if new[0] > old[0]:
            return Direction.S
        if new[0] < old[0]:
            return Direction.N
        if new[1] > old[1]:
            return Direction.E
        if new[1] < old[1]:
            return Direction.W
        return None

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

    def critique_location_discovery(self, state_change : ZeldaStateChange, rewards: Dict[str, float]):
        """Critiques the location discovery based on the old and new states and assigns rewards or penalties
        accordingly."""
        prev, curr = state_change.previous, state_change.current
        if curr.level != 1:
            rewards['penalty-left-dungeon'] = self.leave_dungeon_penalty
        elif prev.location != curr.location:
            if prev.location_objective == curr.location:
                rewards['reward-new-location'] = self.new_location_reward
            else:
                rewards['penalty-left-early'] = self.leave_early_penalty

    def critique_key_pickup_usage(self, state_change : ZeldaStateChange, rewards):
        prev, curr = state_change.previous, state_change.current

        if prev.link.keys > curr.link.keys and curr.location == 0x73 and curr.location_objective != 0x63:
            pass # do not give a reward for prematurely using a key
        else:
            super().critique_key_pickup_usage(state_change, rewards)


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

    def get_score(self, state_change: ZeldaStateChange):
        state = state_change.current
        if state.action == ActionType.ITEM and state.link.selected_equipment == SelectedEquipment.BOMBS:
            hits = state_change.damage_dealt
            if hits:
                self.score += hits
            else:
                self.score -= 1

        return self.score

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

    def critique_location_discovery(self, state_change : ZeldaStateChange, rewards):
        prev, curr = state_change.previous, state_change.current

        if prev.location != curr.location:
            if prev.location_objective != curr.location:
                rewards['penalty-left-early'] = self.leave_early_penalty
                return

            if prev.objective_kind == ObjectiveKind.ENTER_CAVE:
                rewards['penalty-left-early'] = self.leave_early_penalty
                return

        if prev.in_cave and curr.location == 0x77 and curr.in_cave:
            rewards['penalty-entered-cave'] = self.entered_cave_penalty

        elif curr.level == 0:
            if curr.location not in self._get_allowed_rooms(curr.link.triforce_pieces):
                rewards['penalty-left-allowed-area'] = self.left_allowed_area_penalty

            elif prev.location == 0x77 and curr.location != 0x77 and prev.link.sword == SwordKind.NONE:
                rewards['penalty-no-sword'] = self.left_without_sword_penalty

            else:
                super().critique_location_discovery(state_change, rewards)

        elif curr.level == curr.link.triforce_pieces + 1:
            # don't forget to reward for reaching the correct dungeon
            super().critique_location_discovery(state_change, rewards)

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

    def critique_location_discovery(self, state_change : ZeldaStateChange, rewards):
        # entered cave
        prev, curr = state_change.previous, state_change.current

        if not prev.in_cave and curr.in_cave:
            if curr.link.sword != SwordKind.NONE:
                rewards['penalty-reentered-cave'] = self.cave_transition_penalty
            else:
                rewards['reward-entered-cave'] = self.cave_tranistion_reward

        # left cave
        elif prev.in_cave and not curr.in_cave:
            if curr.link.sword != SwordKind.NONE:
                rewards['reward-left-cave'] = self.cave_tranistion_reward
            else:
                rewards['penalty-left-cave-early'] = self.cave_transition_penalty

        elif curr.location != 0x77:
            if curr.link.sword != SwordKind.NONE:
                rewards['reward-new-location'] = self.new_location_reward
            else:
                rewards['penalty-left-scenario'] = -self.new_location_reward

    def get_score(self, state_change : ZeldaStateChange):
        state = state_change.current

        score = 0
        if state.in_cave:
            score += 1

            if state.link.sword != SwordKind.NONE:
                score += 1

        else:
            if state.link.sword != SwordKind.NONE:
                score += 3

            if state.location != 0x77:
                score += 1

        return score
