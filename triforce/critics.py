"""Gameplay critics for Zelda."""

from enum import Enum
from typing import Dict
import numpy as np

from triforce.action_space import ActionKind
from triforce.rewards import REWARD_LARGE, REWARD_MAXIMUM, REWARD_MEDIUM, REWARD_MINIMUM, REWARD_SMALL, REWARD_TINY, \
    Penalty, Reward, StepRewards

from .zelda_enums import Direction, SwordKind, ZeldaAnimationKind, AnimationState
from .game_state_change import ZeldaStateChange

HEALTH_LOST_PENALTY = Penalty("penalty-lost-health", -REWARD_LARGE)
HEALTH_GAINED_REWARD = Reward("reward-gained-health", REWARD_LARGE)
USED_KEY_REWARD = Reward("reward-used-key", REWARD_LARGE)
WALL_COLLISION_PENALTY = Penalty("penalty-wall-collision", -REWARD_TINY)
MOVE_CLOSER_REWARD = Reward("reward-move-closer", REWARD_TINY)
MOVE_AWAY_PENALTY = Penalty("penalty-move-away", -REWARD_TINY - REWARD_MINIMUM)
LATERAL_MOVE_PENALTY = Penalty("penalty-move-lateral", -REWARD_MINIMUM)
DANGER_TILE_PENALTY = Penalty("penalty-move-danger", -REWARD_MEDIUM)
MOVED_TO_SAFETY_REWARD = Reward("reward-move-safety", REWARD_TINY)
ATTACK_NO_ENEMIES_PENALTY = Penalty("penalty-attack-no-enemies", -MOVE_CLOSER_REWARD.value * 2)
ATTACK_MISS_PENALTY = Penalty("penalty-attack-miss", -REWARD_TINY - REWARD_MINIMUM)
ATTACK_OFFSCREEN_PENALTY = Penalty("penalty-attack-offscreen", ATTACK_MISS_PENALTY.value)

DIDNT_FIRE_PENALTY = Penalty("penalty-didnt-fire", -REWARD_TINY)
BLOCK_PROJECTILE_REWARD = Reward("reward-block-projectile", REWARD_LARGE)
FIRED_CORRECTLY_REWARD = Reward("reward-fired-correctly", REWARD_TINY)
INJURE_KILL_REWARD = Reward("reward-hit", REWARD_MEDIUM)
INJURE_KILL_MOVEMENT_ROOM_REWARD = Reward("reward-incidental-hit", REWARD_SMALL)
BEAM_ATTACK_REWARD = Reward("reward-beam-hit", REWARD_MEDIUM)
PENALTY_CAVE_ATTACK = Penalty("penalty-attack-cave", -REWARD_MAXIMUM)
USED_BOMB_PENALTY = Penalty("penalty-bomb-miss", -REWARD_MEDIUM)
BOMB_HIT_REWARD = Reward("reward-bomb-hit", REWARD_SMALL)
PENALTY_WRONG_LOCATION = Penalty("penalty-wrong-location", -REWARD_MEDIUM)

def _init_equipment_rewards():
    """Initializes the equipment rewards."""
    values = {
        'sword': REWARD_MAXIMUM,
        'arrows': REWARD_MAXIMUM,
        'bow': REWARD_MAXIMUM,
        'candle': REWARD_MAXIMUM,
        'whistle': REWARD_MAXIMUM,
        'food': REWARD_MAXIMUM,
        'potion': REWARD_MAXIMUM,
        'magic_rod': REWARD_MAXIMUM,
        'raft': REWARD_MAXIMUM,
        'book': REWARD_MAXIMUM,
        'ring': REWARD_MAXIMUM,
        'ladder': REWARD_MAXIMUM,
        'magic_key': REWARD_MAXIMUM,
        'power_bracelet': REWARD_MAXIMUM,
        'letter': REWARD_MAXIMUM,
        'boomerang': REWARD_MAXIMUM,
        'compass': REWARD_MAXIMUM,
        'map': REWARD_MAXIMUM,
        'rupees' : REWARD_SMALL,
        'heart-container' : REWARD_MAXIMUM,
        'triforce' : REWARD_MAXIMUM,
        'bombs' : REWARD_MAXIMUM,
        'keys' : REWARD_MAXIMUM,
    }

    return {k: Reward(f"reward-gained-{k}", v) for k, v in values.items()}

EQUIPMENT_REWARD_MAP = _init_equipment_rewards()

class ZeldaCritic:
    """Base class for Zelda gameplay critics."""
    def clear(self):
        """Called when the environment is reset to clear any saved state."""

    def critique_gameplay(self, state_change : ZeldaStateChange, rewards: Dict[str, float]):
        """Critiques the gameplay by comparing the old and new states and the rewards obtained."""
        raise NotImplementedError()


MOVEMENT_SCALE_FACTOR = 9.0
DISTANCE_THRESHOLD = 28

class GameplayCritic(ZeldaCritic):
    """Base class for Zelda gameplay critics."""
    def __init__(self):
        super().__init__()

        self._correct_locations = set()
        self._seen_locations = set()
        self._total_hits = 0
        self._room_enter_health = None
        self._equipment_rewards = {}

    def clear(self):
        super().clear()
        self._correct_locations.clear()
        self._seen_locations.clear()
        self._total_hits = 0
        self._room_enter_health = None

    def critique_gameplay(self, state_change : ZeldaStateChange, rewards : StepRewards):
        """Critiques the gameplay by comparing the old and new states and the rewards obtained."""
        # triforce
        self.critique_triforce(state_change, rewards)

        # combat
        self.critique_attack(state_change, rewards)
        self.critique_item_usage(state_change, rewards)

        # items
        self.critique_used_key(state_change, rewards)
        self.critique_equipment_pickup(state_change, rewards)

        # movement
        if state_change.action.kind == ActionKind.MOVE:
            self.critique_location_change(state_change, rewards)
            self.critique_movement(state_change, rewards)

            # Blocking projectiles only happens when not using an item
            self.critique_block(state_change, rewards)

        self.critique_health_change(state_change, rewards)

        # If we lost health, remove all rewards since we want that to be the focus
        if state_change.health_lost > 0:
            rewards.remove_rewards()

        self.set_score(state_change, rewards)

    # reward helpers, may be overridden
    def critique_equipment_pickup(self, state_change : ZeldaStateChange, rewards):
        """Critiques the pickup of equipment items."""
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
        self.__check_one_equipment(state_change, rewards, 'rupees')
        self.__check_one_equipment(state_change, rewards, 'keys')

    def __check_one_equipment(self, state_change : ZeldaStateChange, rewards, item):
        prev, curr = self.__get_equipment_change(state_change, item)
        if prev < curr:
            rewards.add(EQUIPMENT_REWARD_MAP[item])

    def __get_equipment_change(self, state_change, item):
        prev = getattr(state_change.previous.link, item)
        curr = getattr(state_change.state.link, item)

        if isinstance(prev, Enum):
            prev = prev.value
        elif isinstance(prev, bool):
            prev = int(prev)

        if isinstance(curr, Enum):
            curr = curr.value
        elif isinstance(curr, bool):
            curr = int(curr)

        return prev, curr

    def critique_used_key(self, state_change : ZeldaStateChange, rewards):
        """Critiques the pickup and usage of keys."""
        prev_link = state_change.previous.link
        curr_link = state_change.state.link

        if prev_link.keys > curr_link.keys:
            rewards.add(USED_KEY_REWARD)

    def critique_health_change(self, state_change : ZeldaStateChange, rewards):
        """Critiques the change in health."""
        prev_link, curr_link = state_change.previous.link, state_change.state.link
        if prev_link.max_health < curr_link.max_health:
            rewards.add(EQUIPMENT_REWARD_MAP['heart-container'])

        elif state_change.health_gained:
            rewards.add(HEALTH_GAINED_REWARD)

        elif state_change.health_lost:
            rewards.add(HEALTH_LOST_PENALTY)

    def critique_triforce(self, state_change : ZeldaStateChange, rewards):
        """Critiques the acquisition of the triforce."""
        prev_link, curr_link = state_change.previous.link, state_change.state.link
        if prev_link.triforce_pieces < curr_link.triforce_pieces:
            rewards.add(EQUIPMENT_REWARD_MAP['triforce'])

        if not prev_link.triforce_of_power and curr_link.triforce_of_power:
            rewards.add(EQUIPMENT_REWARD_MAP['triforce'])

    def critique_block(self, state_change : ZeldaStateChange, rewards):
        """Critiques blocking of projectiles."""
        prev_link, curr_link = state_change.previous.link, state_change.state.link
        if not prev_link.is_blocking and curr_link.is_blocking:
            rewards.add(BLOCK_PROJECTILE_REWARD)

    def critique_attack(self, state_change : ZeldaStateChange, rewards):
        """Critiques attacks made by the player."""
        # pylint: disable=too-many-branches

        prev, curr = state_change.previous, state_change.state
        if state_change.hits and prev.link.are_beams_available \
                             and curr.link.get_animation_state(ZeldaAnimationKind.BEAMS) != AnimationState.INACTIVE:
            rewards.add(BEAM_ATTACK_REWARD)

        elif state_change.hits:
            if not curr.in_cave:
                rewards.add(INJURE_KILL_REWARD)
            else:
                rewards.add(PENALTY_CAVE_ATTACK)

        elif state_change.action.kind in (ActionKind.SWORD, ActionKind.BEAMS):
            if not curr.enemies:
                rewards.add(ATTACK_NO_ENEMIES_PENALTY)

            elif curr.link.is_sword_frozen:
                rewards.add(ATTACK_OFFSCREEN_PENALTY)

            elif (active_enemies := curr.active_enemies):
                enemy_vectors = [enemy.vector for enemy in active_enemies if abs(enemy.distance) > 0]
                if enemy_vectors:
                    link_vector = curr.link.direction.to_vector()
                    dotproducts = np.sum(link_vector * enemy_vectors, axis=1)
                    if not np.any(dotproducts > np.sqrt(2) / 2):
                        rewards.add(ATTACK_MISS_PENALTY)
                    elif not prev.link.are_beams_available:
                        distance = active_enemies[0].distance
                        if distance > DISTANCE_THRESHOLD:
                            rewards.add(ATTACK_MISS_PENALTY)

    def critique_item_usage(self, state_change : ZeldaStateChange, rewards):
        """Critiques the usage of items."""
        # Always penalize using a bomb, but offset it by the reward for hitting something
        if state_change.previous.link.bombs > state_change.state.link.bombs:
            rewards.add(USED_BOMB_PENALTY)

        if state_change.action.kind == ActionKind.BOMBS:
            rewards.add(BOMB_HIT_REWARD, state_change.hits)

    def critique_location_change(self, state_change : ZeldaStateChange, rewards):
        """Critiques the discovery of new locations."""
        if self._room_enter_health is None:
            self._room_enter_health = state_change.previous.link.health

        prev = state_change.previous.full_location
        curr = state_change.state.full_location

        # Don't let the agent walk offscreen then right back on to get a quick reward
        if prev != curr and not self._correct_locations:
            self._correct_locations.add((prev, curr))

        if prev != curr:
            health_change = state_change.previous.link.health - self._room_enter_health
            reward = (np.clip(health_change, -3.0, 3.0) + 3) / 6
            reward = np.clip(reward, REWARD_MINIMUM, REWARD_MAXIMUM)

            if curr in state_change.previous.objectives.next_rooms:
                if (curr, prev) in self._correct_locations:
                    reward = REWARD_MINIMUM
                else:
                    self._correct_locations.add((curr, prev))

                rewards.add(Reward("reward-new-location", reward))
            else:
                rewards.add(PENALTY_WRONG_LOCATION)

            self._room_enter_health = state_change.state.link.health

    def critique_movement(self, state_change : ZeldaStateChange, rewards):
        """
        Critiques movement on the current screen.  This is the most difficult method to get right.  Movement in Zelda
        is complicated and unintended consequences are common.
        """
        # pylint: disable=too-many-branches, too-many-locals

        prev = state_change.previous
        curr = state_change.state

        prev_link = prev.link
        curr_link = curr.link

        if state_change.action.kind != ActionKind.MOVE:
            return

        # Don't score movement if we moved to a new location or took damage.  The "movement" which occurs from
        # damage should never be rewarded, and it will be penalized by the health loss critic.
        if state_change.action.kind != ActionKind.MOVE \
                or state_change.health_lost \
                or prev.full_location != curr.full_location:

            return

        # Did link run into a wall?
        if prev_link.position == curr_link.position:
            rewards.add(WALL_COLLISION_PENALTY)
            return

        # Did link get too close to an enemy?
        self.critique_moving_into_danger(state_change, rewards)

        # Did we move to a place we didn't think Link could get to?
        old_wavefront = prev.wavefront[prev_link.tile]
        new_wavefront = curr.wavefront[curr_link.tile]
        if old_wavefront < new_wavefront:
            rewards.add(MOVE_AWAY_PENALTY)

        elif old_wavefront == new_wavefront:
            rewards.add(LATERAL_MOVE_PENALTY)

        else:
            # We moved closer, but scale the reward by the pixels moved in case the agent finds a way to exploit
            # our reward system.
            match curr_link.direction:
                case Direction.N:
                    dir_vect = np.array([0, -1], dtype=np.float32)
                case Direction.S:
                    dir_vect = np.array([0, 1], dtype=np.float32)
                case Direction.E:
                    dir_vect = np.array([1, 0], dtype=np.float32)
                case Direction.W:
                    dir_vect = np.array([-1, 0], dtype=np.float32)
                case _:
                    raise ValueError("Invalid direction")

            movement = curr_link.position.numpy - prev_link.position.numpy
            progress = np.dot(movement, dir_vect)
            rewards.add(MOVE_CLOSER_REWARD, progress / MOVEMENT_SCALE_FACTOR)

    def critique_moving_into_danger(self, state_change : ZeldaStateChange, rewards):
        """Critiques the agent for moving too close to an enemy or projectile.  These are added and subtracted
        independent of other movement rewards.  This ensures that even if the agent is moving in the right direction,
        it is still wary of moving too close to an enemy."""
        prev, curr = state_change.previous, state_change.state

        if not state_change.health_lost and not curr.link.is_blocking:
            prev_active = [enemy.index for enemy in prev.active_enemies]

            prev_overlap = [tile
                            for enemy in prev.active_enemies
                            for tile in enemy.link_overlap_tiles
                            if tile in prev.link.self_tiles]

            curr_overlap = [tile
                            for enemy in curr.active_enemies
                            if enemy.index in prev_active
                            for tile in enemy.link_overlap_tiles
                            if tile in curr.link.self_tiles]

            danger_diff = len(curr_overlap) - len(prev_overlap)

            if danger_diff > 0:
                rewards.add(DANGER_TILE_PENALTY)
            elif danger_diff < 0:
                if len(prev.active_enemies) == len(curr.active_enemies):
                    rewards.add(MOVED_TO_SAFETY_REWARD)

    def set_score(self, state_change : ZeldaStateChange, rewards : StepRewards):
        """Sets the score based on how many rooms we have seen, enemies hit, and other factors."""
        self._seen_locations.add(state_change.state.full_location)
        seen_locations = len(self._seen_locations) - 1
        correct_locations = len(self._correct_locations) - 1
        self._total_hits += state_change.hits

        score = 0.5 * (seen_locations + correct_locations) + 0.1 * self._total_hits
        if state_change.previous.link.triforce_of_power < state_change.state.link.triforce_of_power:
            score += 100

        triforce_diff = state_change.state.link.triforce_pieces - state_change.previous.link.triforce_pieces
        if triforce_diff > 0:
            score += triforce_diff * 100

        rewards.score = score

REWARD_ENTERED_CAVE = Reward("reward-entered-cave", REWARD_LARGE)
REWARD_LEFT_CAVE = Reward("reward-left-cave", REWARD_LARGE)
REWARD_NEW_LOCATION = Reward("reward-new-location", REWARD_LARGE)
PENALTY_REENTERED_CAVE = Penalty("penalty-reentered-cave", -REWARD_MAXIMUM)
PENALTY_LEFT_CAVE_EARLY = Penalty("penalty-left-cave-early", -REWARD_MAXIMUM)
PENALTY_LEFT_SCENARIO = Penalty("penalty-left-scenario", -REWARD_LARGE)

class OverworldSwordCritic(GameplayCritic):
    """Critic specifically for the beginning of the game up through grabbing the first sword."""
    def __init__(self):
        super().__init__()

        self.cave_tranistion_reward = REWARD_LARGE
        self.cave_transition_penalty = -REWARD_MAXIMUM
        self.new_location_reward = REWARD_LARGE

    def critique_location_change(self, state_change : ZeldaStateChange, rewards):
        # entered cave
        prev, curr = state_change.previous, state_change.state

        if not prev.in_cave and curr.in_cave:
            if curr.link.sword != SwordKind.NONE:
                rewards.add(PENALTY_REENTERED_CAVE)
            else:
                rewards.add(REWARD_ENTERED_CAVE)

        # left cave
        elif prev.in_cave and not curr.in_cave:
            if curr.link.sword != SwordKind.NONE:
                rewards.add(REWARD_LEFT_CAVE)
            else:
                rewards.add(PENALTY_LEFT_CAVE_EARLY)

        elif curr.location != 0x77:
            if curr.link.sword != SwordKind.NONE:
                rewards.add(REWARD_NEW_LOCATION)
            else:
                rewards.add(PENALTY_LEFT_SCENARIO)

    def set_score(self, state_change : ZeldaStateChange, rewards : StepRewards):
        state = state_change.state

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

        rewards.score = score
