"""Thin, fast wrapper over a stable-retro NES emulator.

The gym ``RetroEnv.step`` renders the screen and evaluates every variable in ``data.json`` on every
frame (~600 fps).  v2 never feeds pixels to a network, so this class drives the emulator core
directly (~3.7k fps) and decodes RAM only when someone asks for it.  Only one emulator may exist per
process (a stable-retro restriction).
"""

import gzip
import json
import os
from typing import Dict, Optional

import numpy as np
import stable_retro as retro

INTEGRATION_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'custom_integrations')
GAME_DIR = os.path.join(INTEGRATION_DIR, 'Zelda-NES')
retro.data.Integrations.add_custom_path(INTEGRATION_DIR)

# NES controller button order used by the Zelda-NES integration.
BUTTONS = ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
BUTTON_INDEX = {name: i for i, name in enumerate(BUTTONS) if name}
NUM_BUTTONS = len(BUTTONS)


def _load_variables():
    with open(os.path.join(GAME_DIR, 'data.json'), 'r', encoding='utf-8') as f:
        info = json.load(f)['info']

    unsigned, signed = [], []
    for name, spec in info.items():
        assert spec['type'] in ('=u1', '|u1', '=i1'), f"Unsupported RAM variable type {spec['type']} for {name}"
        (signed if spec['type'] == '=i1' else unsigned).append((name, spec['address']))

    return unsigned, signed


_UNSIGNED_VARS, _SIGNED_VARS = _load_variables()
VARIABLES = frozenset(name for name, _ in _UNSIGNED_VARS + _SIGNED_VARS)


def state_path(name: str) -> str:
    """Resolves a savestate name ('1_73n', '1_73n.state') or path to an absolute path."""
    if os.path.isabs(name) or os.path.exists(name):
        return os.path.abspath(name)

    if not name.endswith('.state'):
        name += '.state'

    return os.path.join(GAME_DIR, name)


def load_state_bytes(name: str) -> bytes:
    """Reads a (gzip-compressed) savestate into raw bytes accepted by ``Emulator.load_state``."""
    with gzip.open(state_path(name), 'rb') as f:
        return f.read()


class Emulator:
    """Owns the NES core.  Frame counting, button input, RAM reads/writes and save states."""
    def __init__(self, state: Optional[str] = None):
        initial = state_path(state) if state else retro.State.NONE
        self._env = retro.make(game='Zelda-NES', state=initial, inttype=retro.data.Integrations.CUSTOM_ONLY,
                               render_mode=None)
        self._env.reset()
        self.em = self._env.em
        self.data = self._env.data
        assert list(self._env.buttons) == BUTTONS, f"Unexpected button layout {self._env.buttons}"

        self.frame = 0
        self._ram_frame = -1
        self._ram = None
        self._info_frame = -1
        self._info = None
        self._no_buttons = np.zeros(NUM_BUTTONS, dtype=np.uint8)

    def close(self):
        """Releases the emulator."""
        if self._env is not None:
            self._env.close()
            self._env = None

    # --- time ---
    def step(self, buttons: Optional[np.ndarray] = None, frames: int = 1):
        """Holds ``buttons`` (uint8[9]) for ``frames`` frames."""
        self.em.set_button_mask(self._no_buttons if buttons is None else buttons, 0)
        for _ in range(frames):
            self.em.step()
            self.frame += 1

    # --- save states ---
    def save_state(self) -> bytes:
        """Snapshot of the full emulator state."""
        return self.em.get_state()

    def load_state(self, state: bytes):
        """Restores a snapshot from ``save_state`` or ``load_state_bytes``."""
        self.em.set_state(state)
        self._invalidate()

    # --- RAM ---
    @property
    def ram(self) -> np.ndarray:
        """All emulator memory blocks concatenated (2KB work RAM followed by cartridge RAM).  Read only."""
        if self._ram_frame != self.frame:
            blocks = self.data.memory.blocks
            self._ram = np.concatenate([np.frombuffer(blocks[k], dtype=np.uint8) for k in sorted(blocks)])
            self._ram_frame = self.frame
        return self._ram

    def read(self, address: int) -> int:
        """Cheap single-byte read of NES work RAM ($0000-$07FF) for per-frame polling."""
        block = address & ~0x3ff
        return self.data.memory.blocks[block][address - block]

    @property
    def info(self) -> Dict[str, int]:
        """All named variables from data.json, decoded from the current RAM."""
        if self._info_frame != self.frame:
            ram = self.ram
            info = {name: int(ram[addr]) for name, addr in _UNSIGNED_VARS}
            for name, addr in _SIGNED_VARS:
                value = int(ram[addr])
                info[name] = value - 256 if value > 127 else value
            self._info = info
            self._info_frame = self.frame
        return dict(self._info)

    def set_value(self, name: str, value: int):
        """Writes a named data.json variable."""
        self.data.set_value(name, int(value))
        self._invalidate()

    def write(self, address: int, value: int):
        """Writes one byte of NES work RAM ($0000-$07FF)."""
        self.data.memory.assign(address, '|u1', int(value))
        self._invalidate()

    def _invalidate(self):
        self._ram_frame = -1
        self._info_frame = -1

    # --- video ---
    def get_screen(self) -> np.ndarray:
        """The current RGB frame.  For rendering only; never fed to a model."""
        return self.em.get_screen()
