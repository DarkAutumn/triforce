"""TFMOVIE v1 writer/reader (mirror of crates/tools/src/formats.rs).  Recording tooling only.

Layout (little-endian): magic ``TFMOVIE\\0``, u16 version=1, u16 0, 20-byte ROM payload SHA-1, u32 frames,
u32 run_count + (u8 buttons, u32 count) runs, u32 poke_count + (u32 frame, u16 addr, u8 value) pokes,
u32 meta_len + UTF-8 JSON metadata.  A poke tagged frame k is applied before input k runs.
"""

import json
import os
import struct

MAGIC = b'TFMOVIE\0'


def _allowed(addr: int) -> bool:
    return 0 <= addr < 0x800 or 0x6000 <= addr < 0x8000


def encode(rom_sha1: bytes, inputs: bytes, pokes, meta: dict) -> bytes:
    """inputs: one NES controller byte per frame; pokes: [(frame, addr, value)] in emission order."""
    assert len(rom_sha1) == 20
    out = bytearray(MAGIC)
    out += struct.pack('<HH', 1, 0)
    out += rom_sha1
    out += struct.pack('<I', len(inputs))
    runs = []
    for b in inputs:
        if runs and runs[-1][0] == b:
            runs[-1][1] += 1
        else:
            runs.append([b, 1])
    out += struct.pack('<I', len(runs))
    for b, n in runs:
        out += struct.pack('<BI', b, n)
    out += struct.pack('<I', len(pokes))
    last = 0
    for frame, addr, value in pokes:
        if not _allowed(addr):
            raise ValueError(f"poke address ${addr:04X} is outside $0000-$07FF/$6000-$7FFF")
        if frame < last or frame > len(inputs):
            raise ValueError(f"poke frame {frame} out of order or range")
        last = frame
        out += struct.pack('<IHB', frame, addr, value & 0xFF)
    meta_bytes = json.dumps(meta, sort_keys=True).encode('utf-8')
    out += struct.pack('<I', len(meta_bytes))
    out += meta_bytes
    return bytes(out)


def decode(data: bytes):
    """Returns (rom_sha1, inputs bytes, pokes list, meta dict)."""
    if data[:8] != MAGIC:
        raise ValueError("not a TFMOVIE file")
    version, _ = struct.unpack_from('<HH', data, 8)
    if version != 1:
        raise ValueError(f"unsupported TFMOVIE version {version}")
    sha = data[12:32]
    frames, run_count = struct.unpack_from('<II', data, 32)
    pos = 40
    inputs = bytearray()
    for _ in range(run_count):
        b, n = struct.unpack_from('<BI', data, pos)
        pos += 5
        inputs += bytes([b]) * n
    if len(inputs) != frames:
        raise ValueError("button runs do not cover the frame count")
    (poke_count,) = struct.unpack_from('<I', data, pos)
    pos += 4
    pokes = []
    for _ in range(poke_count):
        pokes.append(struct.unpack_from('<IHB', data, pos))
        pos += 7
    (meta_len,) = struct.unpack_from('<I', data, pos)
    pos += 4
    meta = json.loads(data[pos:pos + meta_len].decode('utf-8'))
    return sha, bytes(inputs), pokes, meta


def write(path: str, rom_sha1: bytes, inputs: bytes, pokes, meta: dict):
    """Writes a movie atomically."""
    data = encode(rom_sha1, inputs, pokes, meta)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, 'wb') as f:
        f.write(data)
    os.replace(tmp, path)
