"""Gameplay critics for Zelda."""

from enum import Enum
from typing import Dict

from triforce.action_space import ActionKind
from triforce.rewards import REWARD_LARGE, REWARD_MAXIMUM, REWARD_MEDIUM, REWARD_MINIMUM, REWARD_SMALL, REWARD_TINY, \
    Penalty, Reward, StepRewards

from .zelda_enums import SwordKind, ZeldaAnimationKind, AnimationState, ZeldaEnemyKind
from .state_change_wrapper import StateChange

PENALTY_LOST_BEAMS = Penalty("penalty-lost-beams", -REWARD_SMALL)
BEAM_DISTANCE_THRESHOLD = 48
USED_KEY_REWARD = Reward("reward-used-key", REWARD_SMALL)
PBRS_SCALE = 20.0
ROOM_STEP_GRACE = 150          # steps in a room before stalling penalty kicks in
ROOM_STEP_PENALTY_MIN = 0.01   # initial per-step penalty after grace
ROOM_STEP_PENALTY_MAX = 0.02   # maximum per-step penalty
ROOM_STEP_RAMP = 1850          # steps over grace to reach max penalty

# Combat rewards decay after this many events per room to prevent farming respawning enemies.
COMBAT_DECAY_THRESHOLD = 8
COMBAT_DECAY_RATE = 0.5

ATTACK_MISS_PENALTY = Penalty("penalty-attack-miss", -REWARD_MINIMUM)

PENALTY_CAVE_ATTACK = Penalty("penalty-attack-cave", -REWARD_MAXIMUM)
USED_BOMB_PENALTY = Penalty("penalty-bomb-used", -REWARD_SMALL)
BOMB_HIT_REWARD = Reward("reward-bomb-hit", REWARD_MEDIUM)
PENALTY_WRONG_LOCATION = Penalty("penalty-wrong-location", -REWARD_SMALL)
PENALTY_WALL_MASTER = Penalty("penalty-wall-master", -REWARD_MAXIMUM)
FIGHTING_WALLMASTER_PENALTY = Penalty("penalty-fighting-wallmaster", -REWARD_TINY)
MOVED_OFF_OF_WALLMASTER_REWARD = Reward("reward-moved-off-wallmaster", REWARD_TINY - REWARD_MINIMUM)
MOVED_ONTO_WALLMASTER_PENALTY = Penalty("penalty-moved-onto-wallmaster", -REWARD_TINY)

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
        'rupees' : REWARD_MEDIUM,
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



class GameplayCritic(ZeldaCritic):
    """Base class for Zelda gameplay critics."""
    def __init__(self):
        super().__init__()

        self._correct_locations = set()
        self._total_hits = 0
        self._equipment_rewards = {}
        self._progress = 0.0
        self._room_combat_counts = {}  # full_location -> combat event count
        self._room_steps = 0  # steps in current room (for stalling detection)
        self._stunned_enemies = set()  # (full_location, enemy_index) already stunned this room
        self._pbrs_tile = None  # baseline tile for PBRS, only updated on MOVE actions

    def clear(self):
        super().clear()
        self._correct_locations.clear()
        self._total_hits = 0
        self._progress = 0.0
        self._room_combat_counts.clear()
        self._room_steps = 0
        self._stunned_enemies.clear()
        self._pbrs_tile = None

    def _get_combat_decay(self, full_location):
        """Returns a decay multiplier for combat rewards in this room.

        After COMBAT_DECAY_THRESHOLD events, each additional event is scaled by
        COMBAT_DECAY_RATE^(excess). This prevents farming respawning enemies
        (e.g. Zoras) for infinite hit/block rewards.
        """
        count = self._room_combat_counts.get(full_location, 0)
        self._room_combat_counts[full_location] = count + 1
        excess = count - COMBAT_DECAY_THRESHOLD
        if excess <= 0:
            return 1.0
        return COMBAT_DECAY_RATE ** excess

    def critique_gameplay(self, state_change : StateChange, rewards : StepRewards):
        """Critiques the gameplay by comparing the old and new states and the rewards obtained."""
        # triforce
        self.critique_triforce(state_change, rewards)

        # combat
        self.critique_attack(state_change, rewards)
        self.critique_item_usage(state_change, rewards)

        # items
        self.critique_used_key(state_change, rewards)
        self.critique_equipment_pickup(state_change, rewards)

        # movement — applies to all actions since weapons press directional buttons
        # and can cause room changes or tile movement
        self.critique_location_change(state_change, rewards)
        self.critique_movement(state_change, rewards)

        # Blocking projectiles only happens when not using an item
        if state_change.action.kind == ActionKind.MOVE:
            self.critique_block(state_change, rewards)

        self.critique_health_change(state_change, rewards)

        # Special cases
        self.critique_wallmaster(state_change, rewards)

        # If we lost health, scale down positive rewards (equipment pickups exempt)
        if state_change.health_lost > 0:
            rewards.scale_rewards(0.5, exempt_prefixes=("reward-gained-",))

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
                # Scale reward by urgency: low health = higher reward for healing
                urgency = 1.0 - (prev_link.health / curr_link.max_health)
                reward_value = REWARD_SMALL + REWARD_MEDIUM * urgency
                rewards.add(Reward("reward-gained-health", reward_value))

        elif state_change.health_lost:
            # Scale penalty by damage amount (half-hearts)
            half_hearts = state_change.health_lost * 2
            penalty_value = min(REWARD_SMALL * half_hearts, REWARD_MAXIMUM)
            rewards.add(Penalty("penalty-lost-health", -penalty_value))

            # Extra penalty for losing beam capability
            if (prev_link.is_health_full and not curr_link.is_health_full
                    and curr_link.sword != SwordKind.NONE):
                rewards.add(PENALTY_LOST_BEAMS)

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
            decay = self._get_combat_decay(state_change.state.full_location)
            rewards.add(Reward("reward-block-projectile", REWARD_MEDIUM * decay))

    def critique_attack(self, state_change : StateChange, rewards):
        """Critiques attacks made by the player."""

        for e_index in state_change.enemies_hit:
            enemy = state_change.state.get_enemy_by_index(e_index)
            if enemy is None:
                continue

            # no penalty or rewards for hitting wallmasters up close
            if enemy.id == ZeldaEnemyKind.Wallmaster and enemy.distance < 30:
                continue

        prev, curr = state_change.previous, state_change.state
        if state_change.hits and prev.link.are_beams_available \
                             and curr.link.get_animation_state(ZeldaAnimationKind.BEAMS) != AnimationState.INACTIVE:
            decay = self._get_combat_decay(curr.full_location)
            rewards.add(Reward("reward-beam-hit", REWARD_MEDIUM * decay))

            # Distance bonus for beam hits from afar
            for e_index in state_change.enemies_hit:
                enemy = curr.get_enemy_by_index(e_index)
                if enemy and enemy.distance > BEAM_DISTANCE_THRESHOLD:
                    rewards.add(Reward("reward-beam-distance", REWARD_TINY * decay))
                    break

        elif state_change.hits:
            if not curr.in_cave:
                decay = self._get_combat_decay(curr.full_location)
                rewards.add(Reward("reward-hit", REWARD_SMALL * decay))
            else:
                rewards.add(PENALTY_CAVE_ATTACK)

        # Miss penalty for any weapon/item action that doesn't hit or stun.
        # Bombs get their own penalty-bomb-used on placement, so skip them here.
        elif (state_change.action.kind != ActionKind.MOVE
              and state_change.action.kind != ActionKind.BOMBS
              and not state_change.enemies_stunned):
            rewards.add(ATTACK_MISS_PENALTY)

        # Boomerang stun reward (only first stun per enemy per room to prevent farming)
        if state_change.enemies_stunned and state_change.action.kind == ActionKind.BOOMERANG:
            loc = curr.full_location
            new_stuns = [i for i in state_change.enemies_stunned if (loc, i) not in self._stunned_enemies]
            for i in new_stuns:
                self._stunned_enemies.add((loc, i))
            if new_stuns:
                decay = self._get_combat_decay(curr.full_location)
                rewards.add(Reward("reward-boomerang-stun", REWARD_SMALL * decay))

    def critique_item_usage(self, state_change : StateChange, rewards):
        """Critiques the usage of items."""
        # Detect bomb placement via animation state, not inventory delta (per_frame can mask it).
        if state_change.action.kind == ActionKind.BOMBS and self._bomb_was_placed(state_change):
            rewards.add(USED_BOMB_PENALTY)
            if state_change.hits:
                rewards.add(BOMB_HIT_REWARD, state_change.hits)

    @staticmethod
    def _bomb_was_placed(state_change : StateChange):
        """Returns True if a bomb was actually placed this step (animation INACTIVE → ACTIVE)."""
        prev = state_change.previous.link
        curr = state_change.state.link
        for kind in (ZeldaAnimationKind.BOMB_1, ZeldaAnimationKind.BOMB_2):
            if (curr.get_animation_state(kind) != AnimationState.INACTIVE
                    and prev.get_animation_state(kind) == AnimationState.INACTIVE):
                return True
        return False

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

    def critique_movement(self, state_change : StateChange, rewards):
        """Critiques movement using Potential-Based Reward Shaping (PBRS).

        F(s, s') = Φ(s') − Φ(s) where Φ(s) = −wavefront_distance(s) / PBRS_SCALE.
        Uses γ=1 so round trips cancel exactly — no oscillation exploits.

        For MOVE actions, PBRS always fires.  For weapon/item actions, PBRS only
        fires when the action caused a tile change (e.g. bomb at screen edge pushing
        Link into the next tile).  Sub-tile pixel shifts from weapons are ignored
        to prevent ratchet exploits.
        """
        prev = state_change.previous
        curr = state_change.state

        # Don't evaluate movement rewards if we took damage or changed rooms.
        if state_change.health_lost or prev.full_location != curr.full_location:
            self._room_steps = 0
            self._pbrs_tile = None
            return

        # Reset room step counter when enemies are killed or items gained
        if state_change.items_gained or len(curr.active_enemies) < len(prev.active_enemies):
            self._room_steps = 0

        # Room stalling penalty: flat -0.01 per step after grace, ramping to -0.02.
        # Resets on room change, enemy kills, or item pickups.
        # PBRS provides direction, this provides urgency to leave the room.
        self._room_steps += 1
        if self._room_steps > ROOM_STEP_GRACE:
            excess = self._room_steps - ROOM_STEP_GRACE
            t = min(excess / ROOM_STEP_RAMP, 1.0)
            penalty = -(ROOM_STEP_PENALTY_MIN + t * (ROOM_STEP_PENALTY_MAX - ROOM_STEP_PENALTY_MIN))
            rewards.add(Penalty("penalty-room-stalling", penalty))

        # For weapon/item actions, only compute PBRS when the tile actually changed.
        # Sub-tile pixel shifts from weapon directional buttons are ignored to prevent
        # the ratchet exploit (weapon shifts go unpenalized, corrective MOVE gets rewarded).
        is_move = state_change.action.kind == ActionKind.MOVE
        if not is_move and curr.link.tile == prev.link.tile:
            return

        # PBRS: F(s,s') = Φ(s') - Φ(s), Φ(s) = -distance / scale
        # After a room change (or first step), _pbrs_tile is None. Seed the
        # baseline from the current position and skip — prevents cross-room
        # wavefront comparisons.
        if self._pbrs_tile is None:
            self._pbrs_tile = curr.link.tile
            return

        wf = prev.pbrs_wavefront
        old_dist = wf.get(self._pbrs_tile)
        new_dist = wf.get(curr.link.tile)
        self._pbrs_tile = curr.link.tile

        if old_dist is None or new_dist is None:
            return

        shaped = (old_dist - new_dist) / PBRS_SCALE

        if shaped > 0:
            rewards.add(Reward("reward-pbrs-movement", shaped))
        elif shaped < 0:
            rewards.add(Penalty("penalty-pbrs-movement", shaped))


REWARD_ENTERED_CAVE = Reward("reward-entered-cave", REWARD_LARGE)
REWARD_LEFT_CAVE = Reward("reward-left-cave", REWARD_LARGE)
REWARD_NEW_LOCATION = Reward("reward-new-location", REWARD_MAXIMUM)
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
