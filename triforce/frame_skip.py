"""Executes timed options frame by frame, stopping early on interrupts, and skips uncontrollable frames.

Ported from v1's ``FrameSkipWrapper`` (controllability detection, uncontrollable-state skipping, RAM item
selection, animation polling) and extended with v2's timed options and interrupts (design §5.1/§5.2).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .actions import Action
from .emulator import BUTTON_INDEX, NUM_BUTTONS, Emulator
from .room_entry import is_pre_entry
from .zelda_enums import ActionKind, Direction, SelectedEquipmentKind

# Game modes ($12).
MODE_DUNGEON_TRANSITION = 2
MODE_REVEAL = 3
MODE_SCROLL_COMPLETE = 4
MODE_GAMEPLAY = 5
MODE_SCROLL_START = 6
MODE_SCROLL = 7
MODE_GAME_OVER = 8
MODE_UNDERGROUND = 9
MODE_UNDERGROUND_TRANSITION = 10
MODE_CAVE = 11
MODE_CAVE_TRANSITION = 16
MODE_DYING = 17
MODE_HOLDING_TRIFORCE = 18

SCROLLING_MODES = frozenset({MODE_SCROLL_COMPLETE, MODE_SCROLL, MODE_SCROLL_START, MODE_UNDERGROUND_TRANSITION,
                             MODE_CAVE_TRANSITION, MODE_REVEAL, MODE_DUNGEON_TRANSITION, MODE_HOLDING_TRIFORCE})
GAME_OVER_MODES = frozenset({MODE_DYING, MODE_GAME_OVER})

STUN_FLAG = 0x40
CAVE_COOLDOWN = 60
MAX_ANIMATION_WAIT = 60
MAX_UNCONTROLLABLE_FRAMES = 2000

# Interrupt radius (design §5.2: start with R = 24 px).
INTERRUPT_RADIUS = 24

# RAM addresses (data.json / zelda_game_data.txt).
ADDR_LEVEL = 0x10
ADDR_MODE = 0x12
ADDR_LOCATION = 0xEB
ADDR_LINK_X = 0x70
ADDR_LINK_Y = 0x84
ADDR_LINK_STATUS = 0xAC
ADDR_SWORD_ANIMATION = 0xB9
ADDR_SHOVE_DIR = 0xC0
ADDR_TREASURE_FLAG = 0xBF
ADDR_TRIGGERED_DOOR_CMD = 0x54
ADDR_OPENED_DOORS = 0xEE
ADDR_HEARTS = 0x66F
ADDR_PARTIAL_HEARTS = 0x670
OBJ_X = 0x70
OBJ_Y = 0x84
OBJ_ID = 0x34F
OBJ_SPAWN_STATE = 0x405
INVENTORY = (0x657, 0x67E)  # sword .. rupees_to_add, excluding hearts (checked separately)
OBJ_ITEM_ID = 0x60
NON_THREAT_IDS = frozenset({0x40, 0x60, 0x63, 0x64, 0x68, 0x6A})

_ITEM_FOR_KIND = {
    ActionKind.BOMB: SelectedEquipmentKind.BOMBS.value,
    ActionKind.BOOMERANG: SelectedEquipmentKind.BOOMERANG.value,
    ActionKind.ARROW: SelectedEquipmentKind.ARROWS.value,
}

_DIR_BUTTONS = {Direction.N: ('UP',), Direction.S: ('DOWN',), Direction.W: ('LEFT',), Direction.E: ('RIGHT',),
                Direction.NW: ('UP', 'LEFT'), Direction.NE: ('UP', 'RIGHT'),
                Direction.SW: ('DOWN', 'LEFT'), Direction.SE: ('DOWN', 'RIGHT')}


def buttons_for(direction: Direction = Direction.NONE, *names) -> np.ndarray:
    """uint8[9] button mask for a direction plus extra buttons ('A', 'B')."""
    result = np.zeros(NUM_BUTTONS, dtype=np.uint8)
    for name in _DIR_BUTTONS.get(direction, ()) + names:
        result[BUTTON_INDEX[name]] = 1
    return result


class Interrupt:
    """Reasons a MOVE or WAIT ended early (``info['interrupt']``)."""
    DAMAGE = 'damage'
    SPAWN = 'spawn'
    ENEMY_NEAR = 'enemy-near'
    TRANSITION = 'transition'
    ITEM = 'item'
    DOOR = 'door'
    WALL = 'wall'
    GAME_OVER = 'game-over'

    ALL = (DAMAGE, SPAWN, ENEMY_NEAR, TRANSITION, ITEM, DOOR, WALL, GAME_OVER)


@dataclass
class OptionResult:
    """What happened while executing one option."""
    frames: int
    interrupt: Optional[str] = None
    control_frames: int = 0         # frames spent executing the option itself (before uncontrollable skipping)


def _is_threat(obj_id: int) -> bool:
    return obj_id != 0 and obj_id not in NON_THREAT_IDS


class _FrameWatch:
    """Snapshot of the RAM fields interrupts compare against."""
    def __init__(self, ram):
        self.mode = int(ram[ADDR_MODE])
        self.location = int(ram[ADDR_LOCATION])
        self.level = int(ram[ADDR_LEVEL])
        self.health = (int(ram[ADDR_HEARTS]) & 0x0F, int(ram[ADDR_PARTIAL_HEARTS]))
        self.containers = int(ram[ADDR_HEARTS]) >> 4
        self.ids = ram[OBJ_ID + 1:OBJ_ID + 12].copy()
        self.inventory = ram[INVENTORY[0]:INVENTORY[1]].copy()
        self.treasure_flag = int(ram[ADDR_TREASURE_FLAG])
        self.doors = (int(ram[ADDR_OPENED_DOORS]), int(ram[ADDR_TRIGGERED_DOOR_CMD]))
        self.link = (int(ram[ADDR_LINK_X]), int(ram[ADDR_LINK_Y]))
        self.near = self._near_slots(ram)

    def _near_slots(self, ram):
        lx, ly = self.link
        near = set()
        for i in range(11):
            obj_id = int(self.ids[i])
            if not _is_threat(obj_id):
                continue
            slot = i + 1
            if obj_id <= 0x48 and ram[OBJ_SPAWN_STATE + slot] != 0:
                continue
            dx = int(ram[OBJ_X + slot]) - lx
            dy = int(ram[OBJ_Y + slot]) - ly
            if dx * dx + dy * dy < INTERRUPT_RADIUS * INTERRUPT_RADIUS:
                near.add(slot)
        return near


def detect_interrupt(before: _FrameWatch, after: _FrameWatch, start_near: set) -> Optional[str]:
    """Compares two consecutive frames and returns the interrupt reason, if any."""
    # pylint: disable=too-many-return-statements
    if after.mode in GAME_OVER_MODES:
        return Interrupt.GAME_OVER

    if after.mode != before.mode or after.location != before.location or after.level != before.level:
        return Interrupt.TRANSITION

    if after.health < before.health or after.containers < before.containers:
        return Interrupt.DAMAGE

    for i in range(11):
        if _is_threat(int(after.ids[i])) and after.ids[i] != before.ids[i]:
            return Interrupt.SPAWN

    if (after.inventory != before.inventory).any() or after.treasure_flag != before.treasure_flag \
            or after.health > before.health or after.containers > before.containers:
        return Interrupt.ITEM

    if after.doors != before.doors:
        return Interrupt.DOOR

    if after.near - before.near - start_near:
        return Interrupt.ENEMY_NEAR

    return None


class OptionExecutor:
    """Runs actions on the emulator until Link is controllable again."""
    def __init__(self, emulator: Emulator):
        self.emu = emulator
        self.was_in_cave = False
        self.just_exited_cave = False
        self.capture_pre_entry = False          # keep the last frame before each room entry (room_entry)
        self.last_pre_entry: Optional[bytes] = None

    def reset(self):
        """Clears per-episode state."""
        self.was_in_cave = False
        self.just_exited_cave = False
        self.last_pre_entry = None

    def gain_control(self) -> int:
        """Skips frames until the agent can control Link.  Returns frames elapsed."""
        start = self.emu.frame
        self.emu.step(None, 1)
        self._skip_uncontrollable(None)
        return self.emu.frame - start

    def execute(self, action: Action, state) -> OptionResult:
        """Executes action from ZeldaGame ``state`` (the current, controllable state).  With ``capture_pre_entry``,
        ``last_pre_entry`` is the emulator state of the last frame before the room entry this action caused (None
        if it caused none)."""
        self.last_pre_entry = None
        start = self.emu.frame
        location = (state.level, state.location, state.in_cave)
        if action.kind in (ActionKind.MOVE, ActionKind.WAIT):
            interrupt = self._run_timed(action, state)
        else:
            interrupt = None
            self._run_attack(action)

        control_frames = self.emu.frame - start
        if interrupt != Interrupt.GAME_OVER and self.emu.read(ADDR_MODE) not in GAME_OVER_MODES:
            self._skip_uncontrollable(location)

        return OptionResult(self.emu.frame - start, interrupt, control_frames)

    # --- timed options ---
    def _run_timed(self, action: Action, state) -> Optional[str]:
        is_move = action.kind == ActionKind.MOVE
        pushing = is_move and state.is_push_block_direction(action.direction)
        if is_move and not pushing and not state.can_link_move(action.direction):
            # A move into a wall costs one frame, as in v1.
            self.emu.step(buttons_for(action.direction), 1)
            return Interrupt.WALL

        buttons = buttons_for(action.direction) if is_move else None
        before = _FrameWatch(self.emu.ram)
        start_near = before.near
        stalled = 0
        for _ in range(action.duration):
            self.emu.step(buttons, 1)
            after = _FrameWatch(self.emu.ram)
            interrupt = detect_interrupt(before, after, start_near)
            if interrupt:
                return interrupt

            if self._is_uncontrollable(after):
                return Interrupt.TRANSITION

            if is_move:
                if after.link != before.link:
                    self.just_exited_cave = False
                    stalled = 0
                elif not pushing:
                    # Link moves at least every other frame; two still frames means he hit something.
                    stalled += 1
                    if stalled >= 2:
                        return Interrupt.WALL

            before = after

        return None

    # --- weapons / items ---
    def _run_attack(self, action: Action):
        item = _ITEM_FOR_KIND.get(action.kind)
        if item is not None:
            self.emu.set_value('selected_item', item)

        facing = action.direction
        if facing in (Direction.NW, Direction.NE):
            facing = Direction.N
        elif facing in (Direction.SW, Direction.SE):
            facing = Direction.S
        self.emu.set_value('link_direction', facing.value)

        button = 'B' if item is not None else 'A'
        self.emu.step(buttons_for(action.direction, button), 1)

        # Poll until the animation finishes: the NES ignores input until link_status == 0 and the sword slot
        # is idle.  Items never touch the sword slot.
        for _ in range(MAX_ANIMATION_WAIT):
            self.emu.step(None, 1)
            if self.emu.read(ADDR_LINK_STATUS) == 0 and self.emu.read(ADDR_SWORD_ANIMATION) == 0:
                break

    # --- uncontrollable frames ---
    @staticmethod
    def _is_uncontrollable(watch: _FrameWatch) -> bool:
        return watch.mode in SCROLLING_MODES

    def _skip_uncontrollable(self, start_location):
        """Skips scrolling, transitions, stun/knockback and cave entry so the agent only sees controllable frames."""
        emu = self.emu
        prev_in_cave = self.was_in_cave
        for _ in range(MAX_UNCONTROLLABLE_FRAMES):
            mode = emu.read(ADDR_MODE)
            if mode in GAME_OVER_MODES:
                return

            in_cave = mode == MODE_CAVE
            if in_cave and not self.was_in_cave:
                emu.step(None, CAVE_COOLDOWN)
                self.was_in_cave = True
                continue

            if mode in SCROLLING_MODES or emu.read(ADDR_LINK_STATUS) & STUN_FLAG \
                    or self._is_level_transition(start_location):
                if self.capture_pre_entry and is_pre_entry(emu.read):
                    self.last_pre_entry = emu.save_state()
                emu.step(None, 1)
                continue

            break
        else:
            raise RuntimeError(f"Link never became controllable (mode={emu.read(ADDR_MODE)})")

        self.was_in_cave = emu.read(ADDR_MODE) == MODE_CAVE
        if prev_in_cave and not self.was_in_cave:
            self.just_exited_cave = True

        emu.step(None, 1)

    def _is_level_transition(self, start_location) -> bool:
        if start_location is None:
            return False
        level, location, in_cave = start_location
        now_in_cave = self.emu.read(ADDR_MODE) == MODE_CAVE
        if in_cave or now_in_cave:
            return False
        return level != self.emu.read(ADDR_LEVEL) and location == self.emu.read(ADDR_LOCATION)
