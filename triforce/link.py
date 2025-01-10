
from dataclasses import dataclass

import numpy as np
from .zelda_enums import ActionKind, AnimationState, ArrowKind, BoomerangKind, CandleKind, Direction, PotionKind, \
    RingKind, SelectedEquipmentKind, SwordKind, TileIndex, ZeldaAnimationKind, SoundKind
from .zelda_objects import ZeldaObject

ANIMATION_BEAMS_ACTIVE = 16
ANIMATION_BEAMS_HIT = 17

ANIMATION_BOMBS_ACTIVE = 18
ANIMATION_BOMBS_EXPLODED = (19, 20)

ANIMATION_ARROW_ACTIVE = 10
ANIMATION_ARROW_HIT = 20
ANIMATION_ARROW_END = 21

ANIMATION_BOOMERANG_MIN = 10
ANIMATION_BOOMERANG_MAX = 57

# pylint: disable=too-many-public-methods

@dataclass
class Link(ZeldaObject):
    """Structured data for Link's status."""

    direction : Direction
    status : int

    @property
    def link_overlap_tiles(self):
        """The tiles that the object overlaps with link's top-left tile."""
        x_dim, y_dim = self.dimensions
        for x in range(0, x_dim):
            for y in range(0, y_dim):
                yield TileIndex(self.tile[0] + y, self.tile[1] + x)

    # Health
    @property
    def max_health(self):
        """The maximum health link can have."""
        return self._get_heart_containers()

    @max_health.setter
    def max_health(self, value: int) -> None:
        """Set the maximum health for link."""
        self._heart_containers = value
        curr = self.game.hearts_and_containers
        self.game.hearts_and_containers = (curr & 0x0F) | (self._heart_containers << 4)

    @property
    def health(self):
        """The current health link has."""
        return self.heart_halves * 0.5

    @health.setter
    def health(self, value: float) -> None:
        """Set the current health for link."""
        self.heart_halves = int(value * 2)

    @property
    def heart_halves(self):
        """The number of half hearts link has."""
        full = self._get_full_hearts() * 2
        partial_hearts = self.game.partial_hearts
        if partial_hearts > 0xf0:
            return full

        partial_count = 1 if partial_hearts > 0 else 0
        return full - 2 + partial_count

    @heart_halves.setter
    def heart_halves(self, value: int) -> None:
        """Set the number of half hearts link has."""
        full_hearts = value // 2
        if full_hearts * 2 == value:
            partial_hearts = 0xfe
        else:
            full_hearts += 1
            partial_hearts = 1 if value % 2 else 0

        self.game.partial_hearts = partial_hearts
        self.game.hearts_and_containers = (full_hearts - 1) | (self.game.hearts_and_containers & 0xf0)

    def _get_full_hearts(self):
        """Returns the number of full hearts link has."""
        return (self.game.hearts_and_containers & 0x0F) + 1

    def _get_heart_containers(self):
        """Returns the number of heart containers link has."""
        if '_heart_containers' not in self.__dict__:
            self.__dict__['_heart_containers'] = (self.game.hearts_and_containers >> 4) + 1

        return self.__dict__['_heart_containers']

    # Calculated status
    def get_available_actions(self):
        """Returns the actions that are available to the agent."""
        # pylint: disable=too-many-branches
        available = []

        # Always able to move in at least one direction
        available.append(ActionKind.MOVE)

        if self.sword != SwordKind.NONE and not self.is_sword_frozen and not self.has_beams:
            available.append(ActionKind.SWORD)

        if self.has_beams:
            available.append(ActionKind.BEAMS)

        if self.bombs:
            available.append(ActionKind.BOMBS)

        if self.arrows != ArrowKind.NONE and self.rupees > 0 and self.bow:
            available.append(ActionKind.ARROW)

        if self.magic_rod:
            available.append(ActionKind.WAND)

        if self.candle != CandleKind.NONE:
            available.append(ActionKind.CANDLE)

        if self.boomerang != BoomerangKind.NONE:
            available.append(ActionKind.BOOMERANG)

        if self.whistle:
            available.append(ActionKind.WHISTLE)

        if self.potion != PotionKind.NONE:
            available.append(ActionKind.POTION)

        if self.food:
            available.append(ActionKind.FOOD)

        return available


    def has_item(self, item : SwordKind | BoomerangKind | ArrowKind):
        """Return whether Link has the given piece of equipment."""
        if isinstance(item, SwordKind):
            return self.sword.value >= item.value

        if isinstance(item, BoomerangKind):
            return self.boomerang.value >= item.value

        if isinstance(item, ArrowKind):
            return self.arrows.value >= item.value

        raise NotImplementedError(f"Item type {item} not yet implemented.")

    def has_triforce(self, level):
        """Returns whether link has the triforce for the given level."""
        return self.game.triforce & (1 << level)

    @property
    def is_health_full(self) -> bool:
        """Is link's health full."""
        return self.heart_halves == self.max_health * 2

    @property
    def has_beams(self) -> bool:
        """Returns True if link is able to fire sword beams in general."""
        return self.sword != SwordKind.NONE and self.is_health_full and not self.is_sword_frozen

    @property
    def are_beams_available(self) -> bool:
        """Returns True if link can immediately fire beams (e.g. has_beams and no sword is currently firing)."""
        return self.get_animation_state(ZeldaAnimationKind.BEAMS) == AnimationState.INACTIVE and self.has_beams

    @property
    def is_sword_frozen(self) -> bool:
        """Returns True when Link is at the edge of the screen.  During this time he cannot use his weapons or
        be harmed."""
        x, y = self.position
        if self.game.level == 0:
            return x < 0x8 or x > 0xe8 or y <= 0x44 or y >= 0xd8

        return x <= 0x10 or x >= 0xd9 or y <= 0x53 or y >= 0xc5

    @property
    def is_invincible(self) -> bool:
        """Returns True if link is invincible."""
        return self.is_sword_frozen or self.clock

    # Animation States
    def get_animation_state(self, animation_id: ZeldaAnimationKind) -> AnimationState:
        """Returns the state of the given animation."""
        state = AnimationState.INACTIVE

        match animation_id:
            case ZeldaAnimationKind.BEAMS:
                beams = self.game.beam_animation
                state = AnimationState.INACTIVE

                if beams == ANIMATION_BEAMS_ACTIVE:
                    state = AnimationState.ACTIVE

                if beams == ANIMATION_BEAMS_HIT:
                    state = AnimationState.HIT

            case ZeldaAnimationKind.BOMB_1:
                state = self._get_bomb_state(self.game.bomb_or_flame_animation)

            case ZeldaAnimationKind.BOMB_2:
                state = self._get_bomb_state(self.game.bomb_or_flame_animation2)

            case ZeldaAnimationKind.ARROW:
                arrows = self.game.arrow_magic_animation
                if ANIMATION_ARROW_ACTIVE <= arrows <= ANIMATION_ARROW_END:
                    state = AnimationState.ACTIVE

            case ZeldaAnimationKind.BOOMERANG:
                boomerang = self.game.bait_or_boomerang_animation

                if ANIMATION_BOOMERANG_MIN <= boomerang <= ANIMATION_BOOMERANG_MAX:
                    state = AnimationState.ACTIVE

            case _:
                raise ValueError(f"Not yet implemented: {animation_id}")

        return state

    def _get_bomb_state(self, bombs):
        """Returns the state of the bomb animation."""
        if ANIMATION_BOMBS_ACTIVE == bombs:
            return AnimationState.ACTIVE

        if bombs in ANIMATION_BOMBS_EXPLODED:
            return AnimationState.HIT

        return AnimationState.INACTIVE

    @property
    def is_blocking(self) -> bool:
        """Whether or not link is currently blocking a projectile (this returns true for as long as the block sound)
        is playing."""
        return self.game.is_sound_playing(SoundKind.ArrowDeflected)

    # Rupees, Bombs, Shield
    @property
    def rupees(self) -> int:
        """The number of rupees link has."""
        return self.game.rupees

    @rupees.setter
    def rupees(self, value: int) -> None:
        """Set the number of rupees link has."""
        self.game.rupees = value

    @property
    def bombs(self) -> int:
        """The number of bombs link has."""
        return self.game.bombs

    @bombs.setter
    def bombs(self, value: int) -> None:
        """Set the number of bombs link has."""
        self.game.bombs = value

    @property
    def bomb_max(self) -> int:
        """The maximum number of bombs link can carry."""
        return self.game.bomb_max

    @bomb_max.setter
    def bomb_max(self, value: int) -> None:
        """Set the maximum number of bombs link can carry."""
        self.game.bomb_max = value

    @property
    def magic_shield(self) -> bool:
        """Returns True if link has the magic shield."""
        return self.game.magic_shield

    @magic_shield.setter
    def magic_shield(self, value: bool) -> None:
        """Set the magic shield for link."""
        self.game.magic_shield = value

    # Dungeon items
    @property
    def triforce_pieces(self) -> int:
        """The number of triforce pieces link has (does not include Triforce of Power in level 9)."""
        return np.binary_repr(self.game.triforce).count('1')

    @property
    def triforce_of_power(self) -> bool:
        """Whether link has the triforce of power (Level 9)."""
        return bool(self.game.triforce_of_power)

    @triforce_of_power.setter
    def triforce_of_power(self, value) -> None:
        self.game.triforce_of_power = value

    @property
    def keys(self) -> int:
        """How many keys link currently posesses."""
        return self.game.keys

    @keys.setter
    def keys(self, value) -> int:
        self.game.keys = value

    @property
    def compass(self) -> bool:
        """Whether link has the compass for the current dungeon."""
        match self.game.level:
            case 0:
                return False

            case 9:
                return self.game.compass9

            case _:
                return self.game.compass

    @compass.setter
    def compass(self, value) -> None:
        value = 1 if value else 0
        match self.game.level:
            case 0:
                raise ValueError("Cannot set compass outside of a dungeon.")

            case 9:
                self.game.compass9 = value

            case _:
                self.game.compass = value

    @property
    def map(self) -> bool:
        """Whether link has the map for the current dungeon."""
        match self.game.level:
            case 0:
                return False

            case 9:
                return self.game.map9

            case _:
                return self.game.map

    @map.setter
    def map(self, value) -> None:
        value = 1 if value else 0
        match self.game.level:
            case 0:
                raise ValueError("Cannot set map outside of a dungeon.")

            case 9:
                self.game.map9 = value

            case _:
                self.game.map = value

    # Weapons and Equipment
    @property
    def selected_equipment(self) -> SelectedEquipmentKind:
        """The currently selected equipment."""
        return SelectedEquipmentKind(self.game.selected_item)

    @selected_equipment.setter
    def selected_equipment(self, value: SelectedEquipmentKind) -> None:
        """Set the currently selected equipment."""
        self.game.selected_item = value.value

    @property
    def clock(self) -> bool:
        """Returns True if link has the clock."""
        return self.game.clock

    @property
    def sword(self) -> SwordKind:
        """Which sword link currently has."""
        return SwordKind(self.game.sword)

    @sword.setter
    def sword(self, value: 'SwordKind') -> None:
        """Set the sword for Link in the game."""
        self.game.sword = value

    @property
    def arrows(self) -> ArrowKind:
        """Which arrow link currently has."""
        return ArrowKind(self.game.arrows)

    @arrows.setter
    def arrows(self, value: ArrowKind) -> None:
        """Set the arrow for Link in the game."""
        self.game.arrows = value

    @property
    def boomerang(self) -> BoomerangKind:
        """Which boomerang link currently has."""
        if self.game.magic_boomerang:
            return BoomerangKind.MAGIC

        if self.game.regular_boomerang:
            return BoomerangKind.WOOD

        return BoomerangKind.NONE

    @boomerang.setter
    def boomerang(self, value: BoomerangKind) -> None:
        """Set the boomerang for Link in the game."""
        if value == BoomerangKind.MAGIC:
            self.game.magic_boomerang = 2
        elif value == BoomerangKind.WOOD:
            self.game.regular_boomerang = 1
        else:
            self.game.magic_boomerang = 0
            self.game.regular_boomerang = 0

    @property
    def bow(self) -> bool:
        """Returns True if link has the bow."""
        return self.game.bow

    @bow.setter
    def bow(self, value: bool) -> None:
        """Set the bow for link."""
        self.game.bow = value

    @property
    def magic_rod(self) -> bool:
        """Returns True if link has the magic rod."""
        return self.game.magic_rod

    @magic_rod.setter
    def magic_rod(self, value: bool) -> None:
        """Set the magic rod for link."""
        self.game.magic_rod = value

    @property
    def book(self) -> bool:
        """Returns True if link has the book of magic."""
        return self.game.book

    @book.setter
    def book(self, value: bool) -> None:
        """Set the book of magic for link."""
        self.game.book = value

    @property
    def candle(self) -> CandleKind:
        """Returns True if link has the candle."""
        return CandleKind(self.game.candle)

    @candle.setter
    def candle(self, value: CandleKind) -> None:
        """Set the candle for link."""
        self.game.candle = value

    @property
    def potion(self) -> PotionKind:
        """Returns True if link has the potion."""
        return PotionKind(self.game.potion)

    @potion.setter
    def potion(self, value: PotionKind) -> None:
        """Set the potion for link."""
        self.game.potion = value

    @property
    def whistle(self) -> bool:
        """Returns True if link has the whistle."""
        return self.game.whistle

    @whistle.setter
    def whistle(self, value: bool) -> None:
        """Set the whistle for link."""
        self.game.whistle = value

    @property
    def food(self) -> bool:
        """Returns True if link has the food."""
        return self.game.food

    @food.setter
    def food(self, value: bool) -> None:
        """Set the food for link."""
        self.game.food = value

    @property
    def letter(self) -> bool:
        """Returns True if link has the letter."""
        return self.game.letter

    @letter.setter
    def letter(self, value: bool) -> None:
        """Set the letter for link."""
        self.game.letter = value

    @property
    def power_bracelet(self) -> bool:
        """Returns True if link has the power bracelet."""
        return self.game.power_bracelet

    @power_bracelet.setter
    def power_bracelet(self, value: bool) -> None:
        """Set the power bracelet for link."""
        self.game.power_bracelet = value

    @property
    def ring(self) -> RingKind:
        """Returns the kind of ring link is wearing."""
        return RingKind(self.game.ring)

    @ring.setter
    def ring(self, value : RingKind) -> None:
        """Sets which ring link is wearing."""
        self.game.ring = value

    @property
    def raft(self) -> bool:
        """Returns True if link has the raft."""
        return self.game.raft

    @raft.setter
    def raft(self, value: bool) -> None:
        """Set the raft for link."""
        self.game.raft = value

    @property
    def ladder(self) -> bool:
        """Returns True if link has the ladder."""
        return self.game.step_ladder

    @ladder.setter
    def ladder(self, value: bool) -> None:
        """Set the ladder for link."""
        self.game.step_ladder = value

    @property
    def magic_key(self) -> bool:
        """Returns True if link has the magic key."""
        return self.game.magic_key

    @magic_key.setter
    def magic_key(self, value: bool) -> None:
        """Set the magic key for link."""
        self.game.magic_key = value
