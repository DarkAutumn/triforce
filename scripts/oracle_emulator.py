"""FCEUmm oracle emulator for recording corpus movies with the Python game stack.

Implements the parts of ``triforce.emulator.Emulator`` the Python stack uses, on top of
``target/oracle/libtforacle.so`` (``oracle/fceumm/build.sh``).  Every frame's controller byte and every RAM
write is recorded so the run can be saved as a TFMOVIE.  Recordings normally start from the base state
(``start_from_base``: the end of ``corpus/movies/power_on_to_start.tfmovie``, the first gameplay frame after
registering LINK), which is cached on disk so the title/registration screens are emulated once, not per run;
the movie then names that base.  ``save_state`` embeds the movie position; ``load_state`` truncates the movie
back to it, so look-ahead and explore restores never leak into a movie.  One oracle per process (FCEUmm state
is global).
"""

import ctypes
import hashlib
import os
import struct
from typing import Callable, Dict, Optional

import numpy as np

import tfmovie
from triforce.emulator import _SIGNED_VARS, _UNSIGNED_VARS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBRARY = os.environ.get('TF_ORACLE', os.path.join(ROOT, 'target', 'oracle', 'libtforacle.so'))
ROM = os.environ.get('TRIFORCE_ROM', os.path.join(ROOT, 'roms', 'zelda.nes'))
CORPUS = os.path.join(ROOT, 'corpus')

# stable-retro button index -> NES controller bit (index 1 is unused)
_BUTTON_BITS = {0: 0x02, 2: 0x04, 3: 0x08, 4: 0x10, 5: 0x20, 6: 0x40, 7: 0x80, 8: 0x01}
_VARS = dict(_UNSIGNED_VARS + _SIGNED_VARS)
_STATE_HEADER = struct.Struct('<II')


def rom_payload_sha1(path: str = ROM) -> bytes:
    """SHA-1 of the ROM after its 16-byte iNES header."""
    with open(path, 'rb') as f:
        return hashlib.sha1(f.read()[16:]).digest()


def nes_byte(buttons) -> int:
    """stable-retro uint8[9] button array -> NES controller byte."""
    if buttons is None:
        return 0
    value = 0
    for i, bit in _BUTTON_BITS.items():
        if buttons[i]:
            value |= bit
    return value


class OracleEmulator:
    """FCEUmm-backed stand-in for ``triforce.emulator.Emulator`` that records a movie."""
    _open = False

    def __init__(self, rom: str = ROM):
        if OracleEmulator._open:
            raise RuntimeError("only one FCEUmm oracle per process")
        if not os.path.exists(LIBRARY):
            raise FileNotFoundError(f"{LIBRARY} not found: run oracle/fceumm/build.sh")
        self._lib = ctypes.CDLL(LIBRARY)
        rc = self._lib.tfo_open(rom.encode())
        if rc != 0:
            raise RuntimeError(f"tfo_open failed with {rc}")
        OracleEmulator._open = True
        self._lib.tfo_ram.restype = ctypes.POINTER(ctypes.c_uint8)
        self._lib.tfo_prg_ram.restype = ctypes.POINTER(ctypes.c_uint8)
        self._lib.tfo_state_size.restype = ctypes.c_size_t
        self._wram = np.ctypeslib.as_array(self._lib.tfo_ram(), shape=(0x800,))
        self._sram = np.ctypeslib.as_array(self._lib.tfo_prg_ram(), shape=(0x2000,))
        self.rom_sha1 = rom_payload_sha1(rom)

        self.frame = 0
        self.inputs = bytearray()
        self.pokes = []
        self.base: Optional[str] = None
        self.on_frame: Optional[Callable[[], None]] = None
        self._info_frame = -1
        self._info = None

    def close(self):
        """Releases the oracle."""
        if OracleEmulator._open and self._lib is not None:
            self._lib.tfo_close()
            OracleEmulator._open = False
            self._lib = None

    # --- time ---
    def step(self, buttons: Optional[np.ndarray] = None, frames: int = 1):
        """Holds ``buttons`` (uint8[9]) for ``frames`` frames."""
        self.run_nes(nes_byte(buttons), frames)

    def run_nes(self, value: int, frames: int = 1):
        """Holds a raw NES controller byte for ``frames`` frames."""
        for _ in range(frames):
            self._lib.tfo_run_frame(ctypes.c_uint8(value))
            self.inputs.append(value)
            self.frame += 1
            self._info_frame = -1
            if self.on_frame is not None:
                self.on_frame()

    # --- save states ---
    def save_state(self) -> bytes:
        """Movie position followed by the FCEUmm state."""
        size = self._lib.tfo_state_size()
        buf = (ctypes.c_uint8 * size)()
        if self._lib.tfo_save_state(buf, ctypes.c_size_t(size)) != 0:
            raise RuntimeError("tfo_save_state failed")
        return _STATE_HEADER.pack(len(self.inputs), len(self.pokes)) + bytes(buf)

    def load_state(self, state: bytes):
        """Restores a ``save_state`` snapshot and truncates the movie to its position."""
        frames, pokes = _STATE_HEADER.unpack_from(state, 0)
        if frames > len(self.inputs) or pokes > len(self.pokes):
            raise ValueError("state is not on this movie's history")
        body = state[_STATE_HEADER.size:]
        buf = (ctypes.c_uint8 * len(body)).from_buffer_copy(body)
        if self._lib.tfo_load_state(buf, ctypes.c_size_t(len(body))) != 0:
            raise RuntimeError("tfo_load_state failed")
        del self.inputs[frames:]
        del self.pokes[pokes:]
        self.frame = frames
        self._info_frame = -1

    # --- RAM ---
    @property
    def ram(self) -> np.ndarray:
        """2 KiB work RAM followed by 8 KiB PRG-RAM (index A - 0x5800 for $6000-$7FFF).  Read only."""
        return np.concatenate([self._wram, self._sram])

    def read(self, address: int) -> int:
        """Single byte of NES work RAM ($0000-$07FF)."""
        return int(self._wram[address])

    @property
    def info(self) -> Dict[str, int]:
        """All named data.json variables, decoded from the current RAM."""
        if self._info_frame != self.frame:
            info = {name: int(self._wram[addr]) for name, addr in _UNSIGNED_VARS}
            for name, addr in _SIGNED_VARS:
                value = int(self._wram[addr])
                info[name] = value - 256 if value > 127 else value
            self._info = info
            self._info_frame = self.frame
        return dict(self._info)

    def set_value(self, name: str, value: int):
        """Writes a named data.json variable."""
        self.write(_VARS[name], int(value) & 0xFF)

    def write(self, address: int, value: int):
        """Writes one byte of RAM ($0000-$07FF) or PRG-RAM ($6000-$7FFF), recorded as a poke."""
        value = int(value) & 0xFF
        if 0 <= address < 0x800:
            self._wram[address] = value
        elif 0x6000 <= address < 0x8000:
            self._sram[address - 0x6000] = value
        else:
            raise ValueError(f"write to ${address:04X} is outside RAM/PRG-RAM")
        self.pokes.append((len(self.inputs), address, value))
        self._info_frame = -1

    def get_screen(self) -> np.ndarray:
        """Not available: the oracle is for recording, not watching."""
        raise NotImplementedError("OracleEmulator does not render")

    # --- movies ---
    def _raw_state(self) -> bytes:
        size = self._lib.tfo_state_size()
        buf = (ctypes.c_uint8 * size)()
        if self._lib.tfo_save_state(buf, ctypes.c_size_t(size)) != 0:
            raise RuntimeError("tfo_save_state failed")
        return bytes(buf)

    def start_from_base(self, name: str = 'power_on_to_start'):
        """Moves a freshly powered-on oracle to the end of base movie ``name`` and starts recording from there.

        The base state is cached under target/oracle/ keyed by the base movie and oracle library hashes."""
        if self.inputs or self.pokes or self.base:
            raise RuntimeError("start_from_base needs a freshly powered-on oracle")
        movie = os.path.join(CORPUS, 'movies', f'{name}.tfmovie')
        key = hashlib.sha1()
        for path in (movie, LIBRARY):
            with open(path, 'rb') as f:
                key.update(f.read())
        cache = os.path.join(ROOT, 'target', 'oracle', f'base-{name}-{key.hexdigest()[:16]}.state')
        if os.path.exists(cache):
            with open(cache, 'rb') as f:
                body = f.read()
            buf = (ctypes.c_uint8 * len(body)).from_buffer_copy(body)
            if self._lib.tfo_load_state(buf, ctypes.c_size_t(len(body))) != 0:
                raise RuntimeError("tfo_load_state failed")
        else:
            self.play_movie(movie)
            tmp = f"{cache}.{os.getpid()}.tmp"
            with open(tmp, 'wb') as f:
                f.write(self._raw_state())
            os.replace(tmp, cache)
        self.inputs = bytearray()
        self.pokes = []
        self.frame = 0
        self._info_frame = -1
        self.base = name

    def play_movie(self, path: str):
        """Replays a TFMOVIE on a freshly powered-on oracle (starting from its base, if it names one), recording
        it as this emulator's movie."""
        with open(path, 'rb') as f:
            sha, inputs, pokes, meta = tfmovie.decode(f.read())
        if sha != self.rom_sha1:
            raise ValueError(f"{path} was recorded for a different ROM")
        if meta.get('base'):
            self.start_from_base(meta['base'])
        elif self.inputs or self.pokes or self.base:
            raise RuntimeError("play_movie needs a freshly powered-on oracle")
        p = 0
        for k, value in enumerate(inputs):
            while p < len(pokes) and pokes[p][0] == k:
                self.write(pokes[p][1], pokes[p][2])
                p += 1
            self.run_nes(value)
        while p < len(pokes):
            self.write(pokes[p][1], pokes[p][2])
            p += 1

    def save_movie(self, path: str, name: str, tags, source: dict):
        """Writes the recorded movie (relative to ``self.base`` when recording started from a base)."""
        meta = {'name': name, 'tags': list(tags), 'source': source}
        if self.base:
            meta['base'] = self.base
        tfmovie.write(path, self.rom_sha1, bytes(self.inputs), self.pokes, meta)
