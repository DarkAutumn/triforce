"""Gameplay critics for Zelda."""

from typing import Dict
import numpy as np

from .objective_selector import ObjectiveKind
from .zelda_wrapper import ActionType
from .zelda_game import Direction, TileState, ZeldaSoundsPulse1, get_heart_containers, get_heart_halves, \
    get_num_triforce_pieces, is_in_cave



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
        self.leave_early_penalty = -REWARD_MAXIMUM

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

        # missed attack
        self.distance_threshold = 28
        self.attack_miss_penalty = -self.move_closer_reward - REWARD_MINIMUM
        self.attack_no_enemies_penalty = -self.move_closer_reward * 2

        # items
        self.used_null_item_penalty = -REWARD_LARGE
        self.bomb_miss_penalty = -REWARD_SMALL
        self.bomb_hit_reward = REWARD_MEDIUM

    def critique_gameplay(self, old : Dict[str, int], new : Dict[str, int], rewards : Dict[str, float]):
        """
        Critiques the gameplay by comparing the old and new states and the rewards obtained.

        Args:
            old (Dict[str, int]): The old state of the game.
            new (Dict[str, int]): The new state of the game.
            rewards (Dict[str, float]): The rewards obtained during gameplay.
        """
        # triforce
        self.critique_triforce(old, new, rewards)

        # combat
        self.critique_block(old, new, rewards)
        self.critique_attack(old, new, rewards)
        self.critique_item_usage(old, new, rewards)
        self.critique_aligned_enemy(old, new, rewards)

        # items
        self.critique_item_pickup(old, new, rewards)
        self.critique_equipment_pickup(old, new, rewards)

        # movement
        self.critique_location_change(old, new, rewards)
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

        old_hearts = get_heart_halves(old)
        new_hearts = get_heart_halves(new)
        if old_hearts < new_hearts:
            return # don't reward if we lost health

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
        self.__check_one_item(old, new, rewards, 'keys')

    def __check_one_item(self, old, new, rewards, item):
        if old[item] < new[item]:
            rewards[f'reward-{item}-gained'] = self.equipment_reward

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
                if new['objective'].kind == ObjectiveKind.FIGHT:
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

    def critique_location_change(self, old, new, rewards):
        """
        Critiques the discovery of new locations.

        Args:
            old (Dict[str, int]): The old state of the game.
            new (Dict[str, int]): The new state of the game.
            rewards (Dict[str, float]): The rewards obtained during gameplay.
        """
        old_obj = old['objective']
        new_obj = new['objective']

        if new_obj.walk is None:
            rewards['penalty-leave-early'] = self.leave_early_penalty

        elif old_obj.walk == new_obj.walk:
            if old_obj.walk_index < new_obj.walk_index:
                rewards['reward-new-location'] = self.new_location_reward
            elif old_obj.walk_index > new_obj.walk_index:
                rewards['penalty-leave-early'] = self.leave_early_penalty

        elif old_obj.walk:
            full_location = new['level'], new['location']
            if old_obj.walk_index < len(old_obj.walk) - 1 and old_obj.walk[old_obj.walk_index + 1] == full_location:
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
        # pylint: disable=too-many-branches, too-many-locals, too-many-statements

        # Don't score movement if we moved to a new location or took damage.  The "movement" which occurs from
        # damage should never be rewarded, and it will be penalized by the health loss critic.

        if self.should_skip_movement_critique(old, new, rewards):
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
            target_y, target_x = self.__find_second_turn(old_path)
            target_tile = target_x, target_y

            old_link_tile = self.__xy_from_coord(old['link'].tile_coordinates[1])
            new_link_tile = self.__xy_from_coord(new['link'].tile_coordinates[1])

            progress = self.__get_progress(movement_direction, old_link_tile, new_link_tile, target_tile)

            if progress > 0:
                rewards['reward-move-closer'] = self.move_closer_reward
            elif progress < 0:
                rewards['penalty-move-farther'] = move_away_penalty

        # This should be relatively rare, but in cases where A* couldn't find a path to the target (usually because
        # a monster moved into a wall), we will reward simply moving closer.
        elif old['objective'].position_or_direction is not None:
            target = old['objective'].position_or_direction
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
                progress = self.__get_progress(movement_direction, old_link_pos, new_link_pos, target)

            if progress > 0:
                percent = min(abs(progress / self.movement_scale_factor), 1)
                rewards['reward-move-closer'] = self.move_closer_reward * percent
            else:
                rewards['penalty-move-farther'] = move_away_penalty

    def should_skip_movement_critique(self, old, new, rewards):
        """Whether or not we should skip the movement critique."""

        if new['action'] != ActionType.MOVEMENT or new['took_damage'] or old['location'] != new['location']:
            return True

        if old['objective'].kind == ObjectiveKind.ENTER_CAVE and not is_in_cave(old) and is_in_cave(new):
            rewards['reward-move-closer'] = self.move_closer_reward
            return True

        if old['objective'].kind == ObjectiveKind.EXIT_CAVE and is_in_cave(old) and not is_in_cave(new):
            rewards['reward-move-closer'] = self.move_closer_reward
            return True

        if is_in_cave(old) != is_in_cave(new):
            rewards['penalty-cave'] = self.leave_early_penalty
            return True

        # spending a key holds you in place for a bit, don't penalize this
        if old['keys'] > new['keys']:
            return True

        if old['objective'].kind == ObjectiveKind.PUSH_BLOCK and \
            old['secrets'][0].position != new['secrets'][0].position and \
            new['link_direction'] == new['objective'].position_or_direction:
            rewards['reward-move-closer'] = self.move_closer_reward
            return True

        if old['objective'].kind in (ObjectiveKind.STAIRS, ObjectiveKind.PUSH_BLOCK) and \
            old['secrets'][0].position != new['secrets'][0].position:
            return True

        if old['link_pos'] == new['link_pos']:
            rewards['penalty-wall-collision'] = self.wall_collision_penalty
            return True


        return False

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

class Dungeon1BombCritic(GameplayCritic):
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

class Dungeon1BossCritic(GameplayCritic):
    """Critic specifically for dungeon 1 with the boss."""
    def clear(self):
        super().clear()
        self.move_closer_reward = REWARD_SMALL
        self.move_away_penalty = -REWARD_SMALL
        self.injure_kill_reward = REWARD_LARGE
        self.health_lost_penalty = -REWARD_SMALL

class OverworldSwordCritic(GameplayCritic):
    """Critic specifically for the beginning of the game up through grabbing the first sword."""
    def __init__(self):
        super().__init__()

        self.cave_tranistion_reward = REWARD_LARGE
        self.cave_transition_penalty = -REWARD_MAXIMUM
        self.new_location_reward = REWARD_LARGE

    def critique_location_change(self, old, new, rewards):

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
