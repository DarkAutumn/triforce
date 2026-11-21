"""A complete replay's facts: the triforces in the order taken ($671 at each room entry), the sword ($657), the
segments, Level 9's last rooms and their models, the end RAM (game mode $12 = $13: the ending), cheats and edits by
room.
    .venv/bin/python scripts/experiments/r32team_golden.py REPLAY"""
import sys
from triforce import _native

data = open(sys.argv[1], 'rb').read()
r = _native.Replay(data)
meta = r.meta
names = [m['name'] for m in meta['models']]
player = _native.ReplayPlayer(data)
rooms = [e for e in r.events() if e['type'] == 'room']


def room_at(frame):
    cur = None
    for e in rooms:
        if e['frame'] > frame:
            break
        cur = e
    return cur


def ram_at(frame):
    player.seek(frame)
    return player.ram()


prev_tri, prev_sword = 0, None
print('triforce / sword changes (sampled at room entries):')
for e in rooms:
    ram = ram_at(e['frame'])
    tri, sword = int(ram[0x671]), int(ram[0x657])
    if tri != prev_tri:
        new = [d + 1 for d in range(8) if (tri & ~prev_tri) >> d & 1]
        print(f"  frame {e['frame']:6d} step {e['step']:3d} L{e['level']} 0x{e['room']:02x}: triforce of level {new} "
              f"($671={tri:#04x})")
        prev_tri = tri
    if sword != prev_sword:
        print(f"  frame {e['frame']:6d} step {e['step']:3d} L{e['level']} 0x{e['room']:02x} cave={e['in_cave']}: "
              f"sword {prev_sword} -> {sword}")
        prev_sword = sword
print('segments:')
for i, s in enumerate(meta.get('segments', [])):
    print(f"  {i} seed {s['seed']} frames {s['frames']} {s['ending']['reason']}")
print('level 9, last rooms:')
for e in [e for e in rooms if e['level'] == 9][-6:]:
    pol = names[e['policy']] if e.get('policy') is not None else None
    print(f"  frame {e['frame']} step {e['step']} 0x{e['room']:02x} policy {pol}")
end = r.end_ram()
print(f"end_ram: game mode $12={end[0x12]:#04x} triforce $671={end[0x671]:#04x} sword $657={end[0x657]} "
      f"level $10={end[0x10]} room $EB={end[0xEB]:#04x}")
print('cheats by room:')
for frame, item, n in r.cheats():
    e = room_at(frame)
    print(f"  frame {frame:6d} {item} {n}: L{e['level']} 0x{e['room']:02x} step {e['step']} cave={e['in_cave']}")
print('edits (frame, addr, old, new):')
for f, a, o, n in r.edits():
    e = room_at(f)
    where = f"L{e['level']} 0x{e['room']:02x} step {e['step']}" if e else 'before control'
    print(f"  {f:6d} ${a:03x} {o} -> {n}  {where}")
