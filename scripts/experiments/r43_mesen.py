"""Round 43, Seeds43: a power-on replay played on a second emulator (Mesen's libretro core) beside zcore, RAM compared.

    python scripts/experiments/r43_mesen.py REPLAY [--core PATH] [--to FRAME] [--every N] [--rom PATH]

The replay's controller bytes (A 1, B 2, SELECT 4, START 8, UP 16, DOWN 32, LEFT 64, RIGHT 128) go to both
machines from console power on, one byte a frame, and the replay's memory edits (a cheated bomb's `$658`) are made
on both before the frame they belong to.  zcore's work RAM ($0000-$07FF) after every frame is compared with
Mesen's: the first frame they differ is reported (with the bytes), and at the end the frames compared.  zcore is
TetaNES's core cut down (crates/zcore/UPSTREAM.md); Mesen is an independent emulator.

The core is a libretro shared object (`retro_*`), driven through ctypes: no frontend is needed.
"""

import argparse
import bz2
import ctypes as C
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from triforce import _native  # noqa: E402  pylint: disable=wrong-import-position

CORE = Path('/home/leculver/git/libretro-mesen/Libretro/mesen_libretro.so')
ROM = Path.home() / 'triforce-shared' / 'rom' / 'zelda.nes'
RETRO_MEMORY_SYSTEM_RAM = 2
RETRO_DEVICE_JOYPAD = 1
# libretro joypad ids for the replay's bits A, B, SELECT, START, UP, DOWN, LEFT, RIGHT.
JOYPAD_IDS = {0x01: 8, 0x02: 0, 0x04: 2, 0x08: 3, 0x10: 4, 0x20: 5, 0x40: 6, 0x80: 7}
ENV_GET_SYSTEM_DIRECTORY, ENV_GET_SAVE_DIRECTORY, ENV_GET_VARIABLE = 9, 31, 15
ENV_SET_PIXEL_FORMAT, ENV_GET_LOG_INTERFACE = 10, 27


class GameInfo(C.Structure):
    _fields_ = [('path', C.c_char_p), ('data', C.c_void_p), ('size', C.c_size_t), ('meta', C.c_char_p)]


class Variable(C.Structure):
    _fields_ = [('key', C.c_char_p), ('value', C.c_char_p)]


ENV_CB = C.CFUNCTYPE(C.c_bool, C.c_uint, C.c_void_p)
VIDEO_CB = C.CFUNCTYPE(None, C.c_void_p, C.c_uint, C.c_uint, C.c_size_t)
AUDIO_CB = C.CFUNCTYPE(None, C.c_int16, C.c_int16)
AUDIO_BATCH_CB = C.CFUNCTYPE(C.c_size_t, C.c_void_p, C.c_size_t)
POLL_CB = C.CFUNCTYPE(None)
STATE_CB = C.CFUNCTYPE(C.c_int16, C.c_uint, C.c_uint, C.c_uint, C.c_uint)


class Libretro:
    """One libretro core with one game loaded, stepped a frame at a time with a controller byte."""

    def __init__(self, core: Path, rom: Path, variables: dict):
        self.lib = C.CDLL(str(core))
        self.buttons = 0
        self._dir = C.c_char_p(str(Path('/tmp')).encode())
        self._vars = {k.encode(): C.c_char_p(v.encode()) for k, v in variables.items()}
        self._callbacks = [ENV_CB(self._env), VIDEO_CB(lambda *a: None), AUDIO_CB(lambda *a: None),
                           AUDIO_BATCH_CB(lambda data, frames: frames), POLL_CB(lambda: None),
                           STATE_CB(self._state)]
        env, video, audio, batch, poll, state = self._callbacks
        self.lib.retro_set_environment(env)
        self.lib.retro_init()
        self.lib.retro_set_video_refresh(video)
        self.lib.retro_set_audio_sample(audio)
        self.lib.retro_set_audio_sample_batch(batch)
        self.lib.retro_set_input_poll(poll)
        self.lib.retro_set_input_state(state)
        self._rom = rom.read_bytes()
        self._rom_buffer = C.create_string_buffer(self._rom, len(self._rom))
        info = GameInfo(str(rom).encode(), C.cast(self._rom_buffer, C.c_void_p), len(self._rom), None)
        self.lib.retro_load_game.restype = C.c_bool
        if not self.lib.retro_load_game(C.byref(info)):
            raise RuntimeError(f"{core} refused {rom}")
        self.lib.retro_set_controller_port_device(0, RETRO_DEVICE_JOYPAD)
        self.lib.retro_get_memory_data.restype = C.c_void_p
        self.lib.retro_get_memory_size.restype = C.c_size_t
        size = self.lib.retro_get_memory_size(RETRO_MEMORY_SYSTEM_RAM)
        pointer = self.lib.retro_get_memory_data(RETRO_MEMORY_SYSTEM_RAM)
        if not pointer or size < 0x800:
            raise RuntimeError(f"no system RAM from the core ({size} bytes)")
        self.ram = np.ctypeslib.as_array(C.cast(pointer, C.POINTER(C.c_uint8)), shape=(0x800,))

    def _env(self, cmd, data):
        if cmd in (ENV_GET_SYSTEM_DIRECTORY, ENV_GET_SAVE_DIRECTORY):
            C.cast(data, C.POINTER(C.c_char_p))[0] = self._dir.value
            return True
        if cmd == ENV_SET_PIXEL_FORMAT:
            return True
        if cmd == ENV_GET_VARIABLE:
            var = C.cast(data, C.POINTER(Variable))[0]
            value = self._vars.get(var.key)
            if value is None:
                return False
            C.cast(data, C.POINTER(Variable))[0].value = value.value
            return True
        return False

    def _state(self, port, device, index, button_id):
        if port != 0 or device != RETRO_DEVICE_JOYPAD:
            return 0
        return int(any(self.buttons & bit and button_id == rid for bit, rid in JOYPAD_IDS.items()))

    def run(self, buttons: int):
        self.buttons = buttons
        self.lib.retro_run()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('replay')
    parser.add_argument('--core', type=Path, default=CORE)
    parser.add_argument('--rom', type=Path, default=ROM)
    parser.add_argument('--to', type=int, help="the last frame compared (default: the replay's end)")
    parser.add_argument('--ram-init', default='All 0s', help="Mesen's power-on RAM (mesen_ramstate)")
    parser.add_argument('--show', type=int, default=20, help='frames with a difference printed (default 20)')
    args = parser.parse_args(argv)
    data = Path(args.replay).read_bytes()
    if args.replay.endswith('.bz2'):
        data = bz2.decompress(data)
    rep = _native.Replay(data)
    if (rep.source or {}).get('kind') != 'power-on':
        sys.exit(f"{args.replay}: not a power-on replay ({rep.source})")
    inputs = rep.inputs()
    by_frame = {}
    for frame, address, _, value in rep.edits():
        by_frame.setdefault(int(frame), []).append((int(address), int(value)))
    mesen = Libretro(args.core, args.rom, {'mesen_ramstate': args.ram_init})
    player = _native.ReplayPlayer(data)
    last = min(args.to or rep.frames, rep.frames)
    # What each emulator holds at the instant it ends a frame (the two stop at different points of it): the scratch
    # bytes $00-$0F, the stack page and the sprite buffer $0200-$02FF (refilled every frame).  Everything else is
    # the game's state; a difference there that lasts is a divergence.
    instant = np.zeros(0x800, dtype=bool)
    instant[0x00:0x10] = True
    instant[0x100:0x300] = True
    game_frames = instant_frames = 0
    addresses = {}
    for k in range(last):
        for address, value in by_frame.get(k, ()):
            if address < 0x800:
                mesen.ram[address] = value
        mesen.run(int(inputs[k]))
        player.step(1)
        ours = player.ram()[:0x800]
        differ = ours != mesen.ram
        if differ.any():
            game = np.nonzero(differ & ~instant)[0]
            instant_frames += not len(game)
            if len(game):
                game_frames += 1
                for a in game:
                    addresses[int(a)] = addresses.get(int(a), 0) + 1
                if game_frames <= args.show:
                    shown = ', '.join(f"${a:04X} zcore {ours[a]:02X} mesen {mesen.ram[a]:02X}" for a in game[:12])
                    print(f"frame {k + 1} (GameMode {ours[0x12]:02X}): {len(game)} bytes differ: {shown}", flush=True)
        if (k + 1) % 10000 == 0:
            print(f"frame {k + 1}: {game_frames} frames with a game-state difference so far", flush=True)
    print(f"{last} frames compared: game state identical on {last - game_frames}, different on {game_frames} "
          f"(bytes: {', '.join(f'${a:04X} x{n}' for a, n in sorted(addresses.items())) or 'none'}); "
          f"{instant_frames} more frames differ only in the scratch bytes, the stack or the sprite buffer")
    return 0 if game_frames == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
