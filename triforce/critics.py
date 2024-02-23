"""Gameplay critics for Zelda."""

from typing import Dict
import numpy as np
from .zelda_wrapper import ActionType
from .zelda_game import ZeldaEnemy, ZeldaSoundsPulse1, get_heart_containers, get_heart_halves, \
    get_num_triforce_pieces, is_in_cave, tile_index_to_position

REWARD_MINIMUM = 0.01
REWARD_TINY = 0.05
REWARD_SMALL = 0.25
REWARD_MEDIUM = 0.5
REWARD_LARGE = 0.75
REWARD_MAXIMUM = 1.0

class ZeldaCritic:
    """Base class for Zelda gameplay critics."""

    def clear(self):
        """Called when the environment is reset to clear any saved state."""

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
        self.block_projectile_reward = REWARD_LARGE

        # these are pivotal to the game, so they are rewarded highly
        self.bomb_pickup_reward = REWARD_LARGE
        self.key_reward = REWARD_LARGE
        self.heart_container_reward = REWARD_MAXIMUM
        self.triforce_reward = REWARD_MAXIMUM
        self.equipment_reward = REWARD_MAXIMUM

        # same room movement rewards
        self.wall_collision_penalty = -REWARD_TINY
        self.move_closer_reward = REWARD_TINY
        self.optimal_path_reward = self.move_closer_reward

        self.minimum_movement_required = 1.5
        self.movement_scale_factor = 9.0
        self.move_away_penalty = -self.move_closer_reward - REWARD_MINIMUM

        self.too_close_threshold = 28
        self.enemy_too_close_penalty = -REWARD_SMALL

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
            new (Dict[str, int]): The new state of the game.
            rewards (Dict[str, float]): The rewards obtained during gameplay.
        """
        arrow_deflected = ZeldaSoundsPulse1.ArrowDeflected.value
        if new['sound_pulse_1'] & arrow_deflected and (old['sound_pulse_1'] & arrow_deflected) != arrow_deflected:
            rewards['reward-block'] = self.block_projectile_reward

    def critique_attack(self, old, new, rewards):
        """
        Critiques attacks made by the player.

        Args:
            old (Dict[str, int]): The old state of the game.
            new (Dict[str, int]): The new state of the game.
            rewards (Dict[str, float]): The rewards obtained during gameplay.
        """
        if new['step_hits']:
            if not is_in_cave(new):
                rewards['reward-hit'] = self.injure_kill_reward * new['step_hits']
            else:
                rewards['penalty-hit-cave'] = -self.injure_kill_reward

        elif new['action'] == ActionType.ATTACK:
            if not new['enemies']:
                rewards['penalty-attack-no-enemies'] = self.attack_no_enemies_penalty

            elif new['is_sword_frozen']:
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
        # The logic in this method is complicated, and tough to break up while still being readable.
        # pylint: disable=too-many-branches, too-many-statements, too-many-locals

        if new['action'] != ActionType.MOVEMENT:
            return

        if old['location'] != new['location'] or is_in_cave(old) != is_in_cave(new):
            return

        # Did link run into a wall?
        if old['link_pos'] == new['link_pos']:
            rewards['penalty-wall-collision'] = self.wall_collision_penalty
            return

        # If link took damage he gets knocked back.  Don't consider that 'movement'.
        if get_heart_halves(old) > get_heart_halves(new):
            return

        # In rooms where wallmasters or traps exist, we want to reward link for moving closer to the
        # center of the room to avoid that.  The a* path already takes into account this by weighting
        # the edges of the room heavy.  This means we need to reward link for following the path exactly.
        # Additionally, WallMaster code is weird.  They seem to jump around the map even when not visible
        # which throws off some of our other code such as checking for enemies that are too close.
        are_enemies_near = self._check_how_close(old, new, rewards)

        # If enemies or projectiles are nearby, no rewards for walking/following the path.  The agent
        # will discover rewards if they attack correctly or avoid damage.
        if not are_enemies_near:
            # do we have an optimal path?
            old_link_pos = np.array(old.get('link_pos', (0, 0)), dtype=np.float32)
            new_link_pos = np.array(new.get('link_pos', (0, 0)), dtype=np.float32)
            if len(old_path := old.get("a*_path", (None, None, []))[2]) >= 2:
                correct_direction, possible_direction = self.__get_optimal_directions(old_path)
                direction = new['direction']

                target_tile = self.__find_second_turn(old_path)
                target = np.array(tile_index_to_position(target_tile), dtype=np.float32)

                # target is the top left of the 8x8 tile, if we are left or above the target, add
                # 8 to the x or y to get to that edge of the tile.
                if new_link_pos[0] < target[0]:
                    target[0] += 8

                if new_link_pos[1] < target[1]:
                    target[1] += 8

                old_distance = np.linalg.norm(target - old_link_pos)
                new_distance = np.linalg.norm(target - new_link_pos)

                diff = abs(new_distance - old_distance)
                if diff >= self.minimum_movement_required:
                    percent = min(diff / self.movement_scale_factor, 1)
                else:
                    percent = None

                # reward if we moved in the right direction
                if direction == correct_direction:
                    if percent is not None:
                        rewards['reward-move-closer'] = self.move_closer_reward * percent

                elif direction == possible_direction:
                    if len(new.get("a*_path", (None, None, []))[2]) <= len(old_path):
                        if percent is not None:
                            rewards['reward-move-closer'] = self.move_closer_reward * percent
                    else:
                        rewards['penalty-move-farther'] = self.move_away_penalty
                else:
                    rewards['penalty-move-farther'] = self.move_away_penalty

            elif (target := new.get('objective_pos_or_dir', None)) is not None:
                # if A* couldn't find a path, we should still reward the agent for moving closer
                # to the objective.  This should be rare, and often happens when an enem moves
                # into a wall.  (Bosses or wallmasters.)
                if isinstance(target, str):
                    if target == 'N':
                        dist = new_link_pos[1] - old_link_pos[1]
                    elif target == 'S':
                        dist = old_link_pos[1] - new_link_pos[1]
                    elif target == 'E':
                        dist = old_link_pos[0] - new_link_pos[0]
                    elif target == 'W':
                        dist = new_link_pos[0] - old_link_pos[0]

                    percent = abs(dist / self.movement_scale_factor)
                else:
                    old_distance = np.linalg.norm(target - old_link_pos)
                    new_distance = np.linalg.norm(target - new_link_pos)
                    dist = new_distance - old_distance
                    percent = abs(dist / self.movement_scale_factor)

                if dist < 0:
                    rewards['reward-move-closer'] = self.move_closer_reward * percent
                else:
                    rewards['penalty-move-farther'] = self.move_away_penalty

    def _check_how_close(self, old, new, rewards):
        are_enemies_near = False

        # did link move too close to an enemy?
        new_enemies_or_projectiles = new['enemies'] + new['projectiles']
        if new_enemies_or_projectiles:
            if any(x.id == ZeldaEnemy.WallMaster for x in new['enemies']):
                pass

            else:
                # find enemies that were too close the last time, and punish for moving closer in that direction
                old_enemies_or_projectiles = old['enemies'] + old['projectiles']
                old_enemies_too_close = [x for x in old_enemies_or_projectiles \
                                         if x.distance < self.too_close_threshold]

                if old_enemies_too_close:
                    link_vector = new['link_vector']

                    # filter old_enemies_too_close to the ones we walked towards
                    old_enemies_walked_towards = [x for x in old_enemies_too_close \
                                                  if np.dot(link_vector, x.vector) > 0.7071]
                    if any(x for x in new_enemies_or_projectiles if x.id in old_enemies_walked_towards):
                        rewards['penalty-move-too-close'] = self.enemy_too_close_penalty

                    are_enemies_near = True

        return are_enemies_near

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
            return 'S'
        if new[0] < old[0]:
            return 'N'
        if new[1] > old[1]:
            return 'E'
        if new[1] < old[1]:
            return 'W'
        return None

    def __get_optimal_directions(self, path):
        first = None
        for i in range(1, len(path)):
            old_index = path[i - 1]
            new_index = path[i]

            direction = self.__get_direction(old_index, new_index)
            if first is None:
                first = direction
            elif first != direction:
                return first, direction

        return first, first

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

    def set_score(self, old : Dict[str, int], new : Dict[str, int]):
        if get_heart_halves(new) < get_heart_halves(old):
            self.health_lost -= 0.5

        new_location = new['location']
        self.seen.add(new_location)
        new['score'] = len(self.seen) - self.health_lost

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
    def __init__(self):
        super().__init__()
        self.total_damage = 0

    def clear(self):
        super().clear()
        self.total_damage = 0
        self.too_close_threshold = 10
        self.move_closer_reward = REWARD_SMALL
        self.move_away_penalty = -REWARD_SMALL
        self.injure_kill_reward = REWARD_LARGE
        self.health_lost_penalty = -REWARD_SMALL

    def set_score(self, old : Dict[str, int], new : Dict[str, int]):
        self.total_damage += new['step_hits']
        new['score'] = get_heart_halves(new) + self.total_damage

overworld_dungeon1_walk_rooms = set([0x77, 0x78, 0x67, 0x68, 0x58, 0x48, 0x38, 0x37])

class Overworld1Critic(GameplayCritic):
    """Critic specifically for overworld 1."""
    def __init__(self):
        super().__init__()

        self.seen = set()
        self.allowed_rooms = overworld_dungeon1_walk_rooms

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

            if old['objective_kind'] == 'cave':
                rewards['penalty-left-early'] = self.leave_early_penalty
                return

        level = new['level']
        location = new['location']

        if not is_in_cave(old) and location == 0x77 and is_in_cave(new):
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
        if get_heart_halves(new) < get_heart_halves(old):
            self.health_lost -= 0.5
        new_location = new['location']
        self.seen.add(new_location)
        new['score'] = len(self.seen) - self.health_lost

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