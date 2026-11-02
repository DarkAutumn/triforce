"""Gameplay critics: per-step rewards for Zelda (ported from v1, design §6).

Magnitudes, combat decay, health halving, the stall penalty, Wallmaster shaping and PBRS come from v1.  v2
changes: the v2 action kinds (MOVE, WAIT, SWORD, BEAMS, BOMB, BOOMERANG, ARROW), PBRS is switchable per
specialist, and puzzle progress (locked door opened, bomb wall opened, push block that opens a path) earns a
one-time reward per room per episode.  Terminal rewards are added by ``rewards.apply_terminal_rewards``.
"""

from enum import Enum
from typing import Dict, Set, Tuple

from .rewards import REWARD_LARGE, REWARD_MAXIMUM, REWARD_MEDIUM, REWARD_MINIMUM, REWARD_SMALL, REWARD_TINY, \
    Penalty, Reward, StepRewards
from .room_state import RoomState, SecretTrigger, decode_room_state
from .state_change import StateChange
from .zelda_enums import ActionKind, AnimationState, SwordKind, ZeldaAnimationKind, ZeldaEnemyKind

PENALTY_LOST_BEAMS = Penalty("penalty-lost-beams", -REWARD_SMALL)
BEAM_DISTANCE_THRESHOLD = 48
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
FIGHTING_WALLMASTER_PENALTY = Penalty("penalty-fighting-wallmaster", -REWARD_TINY)
MOVED_OFF_OF_WALLMASTER_REWARD = Reward("reward-moved-off-wallmaster", REWARD_TINY - REWARD_MINIMUM)
MOVED_ONTO_WALLMASTER_PENALTY = Penalty("penalty-moved-onto-wallmaster", -REWARD_TINY)
REWARD_NEW_LOCATION = Reward("reward-new-location", REWARD_MAXIMUM)
REWARD_REVIST_LOCATION = Reward("reward-revisit-location", REWARD_TINY)

# Puzzle progress (design §6/§8): once per room per episode.
REWARD_OPENED_LOCKED_DOOR = Reward("reward-opened-locked-door", REWARD_MEDIUM)
REWARD_OPENED_BOMB_WALL = Reward("reward-opened-bomb-wall", REWARD_LARGE)
REWARD_PUSHED_BLOCK = Reward("reward-pushed-block", REWARD_LARGE)

# Moves that don't use a weapon (no miss penalty; projectile blocks count).
_NON_WEAPON_KINDS = (ActionKind.MOVE, ActionKind.WAIT)


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


class GameplayCritic:
    """The default reward profile.  ``pbrs`` switches potential-based movement shaping (reward.pbrs)."""
    def __init__(self, pbrs: bool = True):
        self.pbrs = pbrs
        self._correct_locations = set()
        self._room_combat_counts = {}  # full_location -> combat event count
        self._room_steps = 0  # steps in current room (for stalling detection)
        self._stunned_enemies = set()  # (full_location, enemy_index) already stunned this room
        self._pbrs_tile = None  # baseline tile for PBRS, only updated on MOVE actions
        self._puzzles_rewarded: Set[Tuple[object, str]] = set()   # (full_location, puzzle kind)

    def clear(self):
        """Called on env reset."""
        self._correct_locations.clear()
        self._room_combat_counts.clear()
        self._room_steps = 0
        self._stunned_enemies.clear()
        self._pbrs_tile = None
        self._puzzles_rewarded.clear()

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

        # items and puzzles
        self.critique_equipment_pickup(state_change, rewards)
        self.critique_puzzles(state_change, rewards)

        # movement — applies to all actions since weapons press directional buttons
        # and can cause room changes or tile movement
        self.critique_location_change(state_change, rewards)
        self.critique_movement(state_change, rewards)

        # Blocking projectiles only happens when not using an item
        if state_change.action.kind in _NON_WEAPON_KINDS:
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

    def critique_puzzles(self, state_change : StateChange, rewards):
        """One-time rewards (per room per episode) for opening a locked door, opening a bomb wall, or a push block
        that opens a path (secret trigger 4 shutters / 5 stairs).  Replaces v1's reward-used-key."""
        prev, curr = state_change.previous, state_change.state
        if prev.full_location != curr.full_location or curr.level == 0:
            return

        before: RoomState = decode_room_state(prev)
        after: RoomState = decode_room_state(curr)
        for direction, door in after.doors.items():
            old = before.doors.get(direction)
            if old is None or old.is_open or not door.is_open:
                continue
            if door.kind == 'key':
                self._reward_puzzle(curr.full_location, f'door-{direction.name}', REWARD_OPENED_LOCKED_DOOR, rewards)
            elif door.kind == 'bomb':
                self._reward_puzzle(curr.full_location, f'bomb-{direction.name}', REWARD_OPENED_BOMB_WALL, rewards)

        if after.has_push_block and before.block_push_complete == 0 and after.block_push_complete > 0 \
                and after.secret_trigger in (SecretTrigger.BLOCK_DOOR, SecretTrigger.BLOCK_STAIRS):
            self._reward_puzzle(curr.full_location, 'push-block', REWARD_PUSHED_BLOCK, rewards)

    def _reward_puzzle(self, location, kind, reward, rewards):
        key = (location, kind)
        if key in self._puzzles_rewarded or reward.name in rewards:
            return
        self._puzzles_rewarded.add(key)
        rewards.add(reward)

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
        """Special handling for rooms with a wallmaster: push Link off the tiles next to the walls.

        Being dragged out of the room is the 'wallmastered' hard-failure ending; its −20 is the terminal
        penalty (v1 added a separate penalty-wall-master of the same size)."""
        prev, curr = state_change.previous, state_change.state
        if prev.full_location != curr.full_location or ZeldaEnemyKind.Wallmaster not in curr.enemies:
            return

        objective_tile = self._is_objective_tile(curr, curr.link.tile)
        curr_on_wallmaster = self._is_wallmaster_tile(curr.link.tile) and not objective_tile
        prev_on_wallmaster = self._is_wallmaster_tile(prev.link.tile) and not objective_tile

        # Are we on a tile which could be wallmastered?  If so, push away from it.
        if curr_on_wallmaster:
            if state_change.action.kind != ActionKind.MOVE:
                rewards.add(FIGHTING_WALLMASTER_PENALTY)
            else:
                rewards.add(MOVED_ONTO_WALLMASTER_PENALTY)

        elif prev_on_wallmaster:
            # If we moved off the wallmaster tile, reward the agent
            if state_change.action.kind == ActionKind.MOVE:
                rewards.add(MOVED_OFF_OF_WALLMASTER_REWARD)

    @staticmethod
    def _is_wallmaster_tile(tile):
        return tile.x in (0x4, 0x1a) or tile.y in (0x4, 0x10)

    @staticmethod
    def _is_objective_tile(state, tile):
        objectives = getattr(state, 'objectives', None)
        if objectives is None:
            return False
        return tile in objectives.targets or tile in objectives.pbrs_targets

    def critique_block(self, state_change : StateChange, rewards):
        """Critiques blocking of projectiles."""
        prev_link, curr_link = state_change.previous.link, state_change.state.link
        if not prev_link.is_blocking and curr_link.is_blocking:
            decay = self._get_combat_decay(state_change.state.full_location)
            rewards.add(Reward("reward-block-projectile", REWARD_MEDIUM * decay))

    def critique_attack(self, state_change : StateChange, rewards):
        """Critiques attacks made by the player."""
        prev, curr = state_change.previous, state_change.state
        kind = state_change.action.kind
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

        # Miss penalty for any weapon action that doesn't hit or stun.  MOVE/WAIT aren't attacks; bombs get
        # their own penalty-bomb-used on placement.
        elif kind not in _NON_WEAPON_KINDS and kind != ActionKind.BOMB and not state_change.enemies_stunned:
            rewards.add(ATTACK_MISS_PENALTY)

        # Boomerang stun reward (only first stun per enemy per room to prevent farming)
        if state_change.enemies_stunned and kind == ActionKind.BOOMERANG:
            loc = curr.full_location
            new_stuns = [i for i in state_change.enemies_stunned if (loc, i) not in self._stunned_enemies]
            for i in new_stuns:
                self._stunned_enemies.add((loc, i))
            if new_stuns:
                decay = self._get_combat_decay(curr.full_location)
                rewards.add(Reward("reward-boomerang-stun", REWARD_SMALL * decay))

    def critique_item_usage(self, state_change : StateChange, rewards):
        """Critiques the usage of items."""
        # Detect bomb placement via animation state, not inventory delta.
        if state_change.action.kind == ActionKind.BOMB and self._bomb_was_placed(state_change):
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
        """Rewards moving to a room the objective wanted (correct exits that keep the episode going, e.g.
        between rooms of the same family); the end conditions decide whether the episode ends."""
        prev = state_change.previous.full_location
        curr = state_change.state.full_location
        if prev == curr or state_change.gained_triforce:
            return

        if curr in state_change.previous.objectives.next_rooms:
            if curr in self._correct_locations:
                rewards.add(REWARD_REVIST_LOCATION)
            else:
                rewards.add(REWARD_NEW_LOCATION)
                self._correct_locations.add(curr)
        else:
            rewards.add(PENALTY_WRONG_LOCATION)

    def critique_movement(self, state_change : StateChange, rewards):
        """Room stall penalty plus Potential-Based Reward Shaping (PBRS).

        F(s, s') = Φ(s') − Φ(s) where Φ(s) = −wavefront_distance(s) / PBRS_SCALE, γ=1 so round trips cancel.
        PBRS fires for MOVE; for every other kind (WAIT and weapons) only when Link's tile changed, so sub-tile
        shifts from weapon presses can't ratchet rewards and WAIT is neutral unless something moved Link.
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

        # Room stalling penalty: -0.01 per step after the grace period, ramping to -0.02.
        self._room_steps += 1
        if self._room_steps > ROOM_STEP_GRACE:
            excess = self._room_steps - ROOM_STEP_GRACE
            t = min(excess / ROOM_STEP_RAMP, 1.0)
            penalty = -(ROOM_STEP_PENALTY_MIN + t * (ROOM_STEP_PENALTY_MAX - ROOM_STEP_PENALTY_MIN))
            rewards.add(Penalty("penalty-room-stalling", penalty))

        if not self.pbrs:
            return

        if state_change.action.kind != ActionKind.MOVE and curr.link.tile == prev.link.tile:
            return

        # After a room change (or first step) seed the baseline and skip: no cross-room comparisons.
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


REWARD_PROFILES: Dict[str, type] = {
    'default': GameplayCritic,
}


def create_critic(profile: str, pbrs: bool) -> GameplayCritic:
    """Builds the critic for a specialist's reward_profile."""
    if profile not in REWARD_PROFILES:
        raise ValueError(f"Unknown reward_profile {profile!r}; known: {sorted(REWARD_PROFILES)}")
    return REWARD_PROFILES[profile](pbrs=pbrs)
