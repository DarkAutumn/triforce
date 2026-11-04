"""A game session: emulator + option executor + object model.  The single code path for driving the game.

Used by the gym environment, the runner and the interactive ``play`` tool, so all of them see exactly the same
frame skipping, interrupts, masks and look-ahead.
"""

import random
from typing import Optional

from .actions import Action, ActionMask, compute_action_mask
from .emulator import Emulator, load_state_bytes
from .frame_skip import ADDR_MODE, MODE_CAVE, OptionExecutor, OptionResult
from .room_entry import is_pre_entry, make_room_fresh, pending_room, randomize_spawn
from .state_change import FutureCreditLedger, StateChange
from .zelda_game import ZeldaGame


class GameSession:
    """Loads states and executes actions, producing ``ZeldaGame`` / ``StateChange`` objects."""
    def __init__(self, emulator: Optional[Emulator] = None, look_ahead: bool = True):
        """look_ahead: simulate fired weapons forward to credit their hits to the firing step (``StateChange``).
        Only rewards use it; watching games (``play_grid``) turns it off so no simulated frames are emulated."""
        self.emu = emulator or Emulator()
        self.look_ahead = look_ahead
        self.executor = OptionExecutor(self.emu)
        self.ledger = FutureCreditLedger()
        self.state: Optional[ZeldaGame] = None
        self.mask: Optional[ActionMask] = None
        self.total_frames = 0
        self.steps = 0
        self.disabled_kinds: frozenset = frozenset()    # action kinds the current specialist may not use

    def set_disabled_kinds(self, kinds) -> None:
        """Action kinds removed from every mask (a specialist's ``items`` config); recomputes the current mask."""
        self.disabled_kinds = frozenset(kinds)
        if self.state is not None:
            self.mask = self._compute_mask(self.state)

    def close(self):
        """Releases the emulator."""
        self.emu.close()

    def load(self, state, randomize_rng: bool = True, ram_overrides=None, fresh_room: bool = False,
             *, controllable: bool = False) -> ZeldaGame:
        """Loads a savestate (name, path, or raw bytes), applies RAM overrides, and gains control of Link.

        randomize_rng: new RNG bytes, and for a pre-entry savestate (``room_entry``) a random SpawnCycle/edge cell,
            all drawn from the ``random`` module (seed it for reproducible loads).
        fresh_room: for a pre-entry savestate, clear the pending room's kill history so it spawns every monster.
        controllable: the saved frame is already an action boundary (curriculum); don't advance it without
            input. Normal entry states still run gain_control through their transition.
        """
        data = state if isinstance(state, bytes) else load_state_bytes(state)
        self.emu.load_state(data)
        self.executor.reset()
        self.ledger.clear()

        if randomize_rng:
            for i in range(12):
                self.emu.set_value(f'rng_{i}', random.randint(1, 255))
        if is_pre_entry(self.emu.read):
            if randomize_rng:
                randomize_spawn(self.emu, random)
            pending = pending_room(self.emu.read) if fresh_room else None
            if pending is not None:
                make_room_fresh(self.emu, *pending)

        for name, value in (ram_overrides or {}).items():
            self.emu.set_value(name, value)

        if controllable:
            self.executor.was_in_cave = self.emu.read(ADDR_MODE) == MODE_CAVE
        else:
            self.executor.gain_control()
        self.total_frames = 0
        self.steps = 0
        self.state = self._build_state()
        return self.state

    def apply_ram(self, values: dict) -> ZeldaGame:
        """Writes named RAM variables at the current frame (no emulation) and rebuilds the state and mask.
        Used for reset-time RAM randomization that depends on the loaded state (e.g. hp relative to max)."""
        for name, value in values.items():
            self.emu.set_value(name, value)
        self.state = self._build_state()
        return self.state

    def snapshot(self) -> dict:
        """Everything needed to resume this session exactly (emulator state + executor flags)."""
        return {'emulator': self.emu.save_state(), 'was_in_cave': self.executor.was_in_cave,
                'just_exited_cave': self.executor.just_exited_cave, 'total_frames': self.total_frames,
                'steps': self.steps}

    def restore(self, snapshot: dict) -> ZeldaGame:
        """Resumes a ``snapshot()`` without stepping the emulator."""
        self.emu.load_state(snapshot['emulator'])
        self.executor.was_in_cave = snapshot['was_in_cave']
        self.executor.just_exited_cave = snapshot['just_exited_cave']
        self.total_frames = snapshot['total_frames']
        self.steps = snapshot['steps']
        self.ledger.clear()
        self.state = self._build_state()
        return self.state

    def act(self, action: Action) -> StateChange:
        """Executes one option and returns what changed."""
        assert self.state is not None, "Call load() first"
        prev = self.state
        _ = prev.enemies, prev.items, prev.link    # materialize before the emulator moves on

        result: OptionResult = self.executor.execute(action, prev)
        self.total_frames += result.frames
        self.steps += 1

        curr = self._build_state()
        change = StateChange(self.emu, prev, curr, action, result, self.ledger, look_ahead=self.look_ahead)
        change.action_mask = self.mask
        self.state = curr
        return change

    def _build_state(self) -> ZeldaGame:
        info = self.emu.info
        info['just_exited_cave'] = self.executor.just_exited_cave
        state = ZeldaGame(self.emu, info, self.emu.frame)
        self.mask = self._compute_mask(state)
        return state

    def _compute_mask(self, state: ZeldaGame) -> ActionMask:
        mask = compute_action_mask(state)
        for kind in self.disabled_kinds:
            mask.disallow_kind(kind)
        return mask
