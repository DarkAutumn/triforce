"""Gameplay critics for Zelda."""

from enum import Enum
from typing import Dict
import torch

from triforce.action_space import ActionKind
from triforce.rewards import REWARD_LARGE, REWARD_MAXIMUM, REWARD_MEDIUM, REWARD_MINIMUM, REWARD_SMALL, REWARD_TINY, \
    Penalty, Reward, StepRewards
from triforce.zelda_game import ZeldaGame

from .zelda_enums import Direction, SwordKind, ZeldaAnimationKind, AnimationState, ZeldaEnemyKind
from .state_change_wrapper import StateChange

HEALTH_LOST_PENALTY = Penalty("penalty-lost-health", -REWARD_LARGE)
HEALTH_GAINED_REWARD = Reward("reward-gained-health", REWARD_LARGE)
USED_KEY_REWARD = Reward("reward-used-key", REWARD_SMALL)
WALL_COLLISION_PENALTY = Penalty("penalty-wall-collision", -REWARD_SMALL)
MOVE_CLOSER_REWARD = Reward("reward-move-closer", REWARD_TINY)
MOVE_AWAY_PENALTY = Penalty("penalty-move-away", -REWARD_TINY - REWARD_MINIMUM)
LATERAL_MOVE_PENALTY = Penalty("penalty-move-lateral", -REWARD_MINIMUM)
DANGER_TILE_PENALTY = Penalty("penalty-move-danger", -REWARD_MEDIUM)
MOVED_TO_SAFETY_REWARD = Reward("reward-move-safety", REWARD_TINY)
ATTACK_NO_ENEMIES_PENALTY = Penalty("penalty-attack-no-enemies", -MOVE_CLOSER_REWARD.value * 2)
ATTACK_MISS_PENALTY = Penalty("penalty-attack-miss", -REWARD_TINY - REWARD_MINIMUM)

DIDNT_FIRE_PENALTY = Penalty("penalty-didnt-fire", -REWARD_TINY)
BLOCK_PROJECTILE_REWARD = Reward("reward-block-projectile", REWARD_MEDIUM)
FIRED_CORRECTLY_REWARD = Reward("reward-fired-correctly", REWARD_TINY)
INJURE_KILL_REWARD = Reward("reward-hit", REWARD_SMALL)
INJURE_KILL_MOVEMENT_ROOM_REWARD = Reward("reward-incidental-hit", REWARD_SMALL)
BEAM_ATTACK_REWARD = Reward("reward-beam-hit", REWARD_SMALL)
PENALTY_CAVE_ATTACK = Penalty("penalty-attack-cave", -REWARD_MAXIMUM)
USED_BOMB_PENALTY = Penalty("penalty-bomb-miss", -REWARD_MEDIUM)
BOMB_HIT_REWARD = Reward("reward-bomb-hit", REWARD_SMALL)
PENALTY_WRONG_LOCATION = Penalty("penalty-wrong-location", -REWARD_MAXIMUM)
PENALTY_WALL_MASTER = Penalty("penalty-wall-master", -REWARD_MAXIMUM)
FIGHTING_WALLMASTER_PENALTY = Penalty("penalty-fighting-wallmaster", -REWARD_TINY)
MOVED_OFF_OF_WALLMASTER_REWARD = Reward("reward-moved-off-wallmaster", REWARD_TINY - REWARD_MINIMUM)
MOVED_ONTO_WALLMASTER_PENALTY = Penalty("penalty-moved-onto-wallmaster", -REWARD_TINY)
PENALTY_OFF_WAVEFRONT = Penalty("penalty-off-wavefront", -REWARD_TINY - REWARD_MINIMUM)

TILE_TIMEOUT = 8

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

    def critique_gameplay(self, state_change : StateChange, rewards: Dict[str, float]):
        """Critiques the gameplay by comparing the old and new states and the rewards obtained."""
        raise NotImplementedError()


MOVEMENT_SCALE_FACTOR = 9.0
DISTANCE_THRESHOLD = 28

class GameplayCritic(ZeldaCritic):
    """Base class for Zelda gameplay critics."""
    def __init__(self):
        super().__init__()

        self._correct_locations = set()
        self._total_hits = 0
        self._equipment_rewards = {}
        self._progress = 0.0
        self._tile_count = {}

    def clear(self):
        super().clear()
        self._correct_locations.clear()
        self._total_hits = 0
        self._progress = 0.0
        self._tile_count.clear()

    def critique_gameplay(self, state_change : StateChange, rewards : StepRewards):
        """Critiques the gameplay by comparing the old and new states and the rewards obtained."""
        # triforce
        self.critique_triforce(state_change, rewards)

        # combat
        self.critique_attack(state_change, rewards)
        self.critique_item_usage(state_change, rewards)

        # items
        #self.critique_used_key(state_change, rewards)
        self.critique_equipment_pickup(state_change, rewards)

        # movement
        if state_change.action.kind == ActionKind.MOVE:
            self.critique_location_change(state_change, rewards)
            self.critique_movement(state_change, rewards)

            # Blocking projectiles only happens when not using an item
            self.critique_block(state_change, rewards)
            self.critique_tile_position(state_change, rewards)

        self.critique_health_change(state_change, rewards)

        # Special cases
        self.critique_wallmaster(state_change, rewards)

        # If we lost health, remove all rewards since we want that to be the focus
        if state_change.health_lost > 0:
            rewards.remove_rewards()

    # Pre-computed list of equipment attribute names for batch checking
    _EQUIPMENT_ATTRS = ('sword', 'arrows', 'bow', 'candle', 'whistle', 'food', 'potion',
                        'magic_rod', 'raft', 'book', 'ring', 'ladder', 'magic_key',
                        'power_bracelet', 'letter', 'boomerang', 'compass', 'map', 'rupees', 'keys')

    # reward helpers, may be overridden
    def critique_equipment_pickup(self, state_change : StateChange, rewards):
        """Critiques the pickup of equipment items."""
        prev_link = state_change.previous.link
        curr_link = state_change.state.link
        for item in self._EQUIPMENT_ATTRS:
            prev = getattr(prev_link, item)
            curr = getattr(curr_link, item)

            if isinstance(prev, Enum):
                prev = prev.value
            elif isinstance(prev, bool):
                prev = int(prev)

            if isinstance(curr, Enum):
                curr = curr.value
            elif isinstance(curr, bool):
                curr = int(curr)

            if prev < curr:
                rewards.add(EQUIPMENT_REWARD_MAP[item])

    def critique_used_key(self, state_change : StateChange, rewards):
        """Critiques the pickup and usage of keys."""
        prev_link = state_change.previous.link
        curr_link = state_change.state.link

        if prev_link.keys > curr_link.keys:
            rewards.add(USED_KEY_REWARD)

    def critique_health_change(self, state_change : StateChange, rewards):
        """Critiques the change in health."""
        prev_link, curr_link = state_change.previous.link, state_change.state.link
        if prev_link.max_health < curr_link.max_health:
            rewards.add(EQUIPMENT_REWARD_MAP['heart-container'])

        elif state_change.health_gained:
            # Don't reward for refilling health after triforce pickup
            if not state_change.gained_triforce:
                rewards.add(HEALTH_GAINED_REWARD)

        elif state_change.health_lost:
            rewards.add(HEALTH_LOST_PENALTY)

    def critique_triforce(self, state_change : StateChange, rewards):
        """Critiques the acquisition of the triforce."""
        if state_change.gained_triforce:
            rewards.add(EQUIPMENT_REWARD_MAP['triforce'])

    def critique_wallmaster(self, state_change : StateChange, rewards):
        """Special handling for rooms with a wallmaster."""
        prev, curr = state_change.previous, state_change.state

        if ZeldaEnemyKind.Wallmaster not in curr.enemies:
            return

        # Did we get wallmastered?
        if prev.full_location != curr.full_location:
            if prev.full_location.manhattan_distance(curr.full_location) > 1:
                rewards.add(PENALTY_WALL_MASTER)

        # Are we on a tile which could be wallmastered?  If so, push away from it.
        elif self._is_wallmaster_tile(curr.link.tile):
            if state_change.action.kind != ActionKind.MOVE:
                rewards.add(FIGHTING_WALLMASTER_PENALTY)
            else:
                rewards.add(MOVED_ONTO_WALLMASTER_PENALTY)

        elif self._is_wallmaster_tile(prev.link.tile):
            # If we moved off the wallmaster tile, reward the agent
            if state_change.action.kind == ActionKind.MOVE:
                rewards.add(MOVED_OFF_OF_WALLMASTER_REWARD)

    def _is_wallmaster_tile(self, tile):
        return tile.x in (0x4, 0x1a) or tile.y in (0x4, 0x10)


    def critique_block(self, state_change : StateChange, rewards):
        """Critiques blocking of projectiles."""
        prev_link, curr_link = state_change.previous.link, state_change.state.link
        if not prev_link.is_blocking and curr_link.is_blocking:
            rewards.add(BLOCK_PROJECTILE_REWARD)

    def critique_attack(self, state_change : StateChange, rewards):
        """Critiques attacks made by the player."""
        # pylint: disable=too-many-branches

        for e_index in state_change.enemies_hit:
            enemy = state_change.state.get_enemy_by_index(e_index)
            if enemy is None:
                continue

            # no penalty or rewards for hitting wallmasters up close
            if enemy.id == ZeldaEnemyKind.Wallmaster and enemy.distance < 30:
                return

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

            elif (active_enemies := [x for x in curr.active_enemies if x.distance > 0]):
                enemy_vectors = torch.stack([x.vector for x in active_enemies])

                if enemy_vectors is not None:
                    dotproducts = torch.sum(curr.link.direction.vector * enemy_vectors, dim=1)
                    if not torch.any(dotproducts > torch.sqrt(torch.tensor(2)) / 2):
                        rewards.add(ATTACK_MISS_PENALTY)
                    elif not prev.link.are_beams_available:
                        distance = active_enemies[0].distance
                        if distance > DISTANCE_THRESHOLD:
                            rewards.add(ATTACK_MISS_PENALTY)
            else:
                rewards.add(ATTACK_MISS_PENALTY)

    def critique_item_usage(self, state_change : StateChange, rewards):
        """Critiques the usage of items."""
        # Always penalize using a bomb, but offset it by the reward for hitting something
        if state_change.previous.link.bombs > state_change.state.link.bombs:
            rewards.add(USED_BOMB_PENALTY)

        if state_change.action.kind == ActionKind.BOMBS:
            rewards.add(BOMB_HIT_REWARD, state_change.hits)

    def critique_location_change(self, state_change : StateChange, rewards):
        """Critiques the discovery of new locations."""
        prev = state_change.previous.full_location
        curr = state_change.state.full_location

        # Don't reward/penalize for changing location on triforce pickup
        if state_change.gained_triforce:
            return

        # Don't let the agent walk offscreen then right back on to get a quick reward
        if prev != curr and not self._correct_locations:
            self._correct_locations.add((prev, curr))

        if prev != curr:
            if curr in state_change.previous.objectives.next_rooms:
                if curr in self._correct_locations:
                    rewards.add(REWARD_REVIST_LOCATION)
                else:
                    rewards.add(REWARD_NEW_LOCATION)
                    self._correct_locations.add(curr)
            else:
                rewards.add(PENALTY_WRONG_LOCATION)


    def critique_tile_position(self, state_change : StateChange, rewards):
        """Critiques landing on the same tile over and over."""
        prev, curr = state_change.previous, state_change.state
        if prev.full_location != curr.full_location or state_change.hits or state_change.items_gained:
            self._tile_count.clear()
            return

        tile = curr.link.tile
        count = self._tile_count.get(tile, 0)
        count += 1
        self._tile_count[tile] = count

        if count >= TILE_TIMEOUT:
            rewards.add(Penalty("penalty-stuck-tile", -REWARD_MINIMUM * count))

    def critique_movement(self, state_change : StateChange, rewards):
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

        # Don't progress movement if we moved to a new location or took damage.  The "movement" which occurs from
        # damage should never be rewarded, and it will be penalized by the health loss critic.
        if state_change.action.kind != ActionKind.MOVE \
                or state_change.health_lost \
                or prev.full_location != curr.full_location:

            return

        # Did link run into a wall?
        if self._did_link_run_into_wall(prev, curr, rewards):
            return

        # Did link get too close to an enemy?
        self.critique_moving_into_danger(state_change, rewards)

        # Did we move to a place we didn't think Link could get to?
        old_wavefront = prev.wavefront.get(prev_link.tile)
        new_wavefront = curr.wavefront.get(curr_link.tile)
        if new_wavefront is None:
            rewards.add(PENALTY_OFF_WAVEFRONT)

        elif old_wavefront is None:
            pass # no reward or penalty for moving back to wavefront tile

        elif old_wavefront < new_wavefront:
            rewards.add(MOVE_AWAY_PENALTY)

        elif old_wavefront == new_wavefront:
            rewards.add(LATERAL_MOVE_PENALTY)

        else:
            rewards.add(MOVE_CLOSER_REWARD)


    def _did_link_run_into_wall(self, prev : ZeldaGame, curr : ZeldaGame, rewards):
        if prev.link.position != curr.link.position:
            return False

        door_entry = {(0xf, 0x4) : Direction.N,
                      (0xf, 0x10) : Direction.S,
                      (0x4, 0xa) : Direction.W,
                      (0x1a, 0xa) : Direction.E}

        if (direction := door_entry.get((curr.link.tile.x, curr.link.tile.y))) is not None:
            if prev.is_door_locked(direction):
                return True

        rewards.add(WALL_COLLISION_PENALTY)
        return True

    def critique_moving_into_danger(self, state_change : StateChange, rewards):
        """Critiques the agent for moving too close to an enemy or projectile.  These are added and subtracted
        independent of other movement rewards.  This ensures that even if the agent is moving in the right direction,
        it is still wary of moving too close to an enemy."""
        prev, curr = state_change.previous, state_change.state

        # Skip evaluation if health was lost or Link is blocking
        if state_change.health_lost or curr.link.is_blocking:
            return

        prev_active_indices = {enemy.index for enemy in prev.active_enemies}

        prev_overlap = {
            tile
            for enemy in prev.active_enemies
            for tile in enemy.link_overlap_tiles
            if tile in prev.link.self_tiles
        }

        curr_overlap = {
            tile
            for enemy in curr.active_enemies
            if enemy.index in prev_active_indices
            for tile in enemy.link_overlap_tiles
            if tile in curr.link.self_tiles
        }

        danger_diff = len(curr_overlap) - len(prev_overlap)
        if danger_diff > 0:
            rewards.add(DANGER_TILE_PENALTY)
        elif danger_diff < 0:
            if len(prev.active_enemies) == len(curr.active_enemies):
                rewards.add(MOVED_TO_SAFETY_REWARD)

REWARD_ENTERED_CAVE = Reward("reward-entered-cave", REWARD_LARGE)
REWARD_LEFT_CAVE = Reward("reward-left-cave", REWARD_LARGE)
REWARD_NEW_LOCATION = Reward("reward-new-location", REWARD_LARGE)
REWARD_REVIST_LOCATION = Reward("reward-revisit-location", REWARD_TINY)
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

    def critique_location_change(self, state_change : StateChange, rewards):
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
