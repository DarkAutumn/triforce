"""Link's purse on entering each dungeon (and at each checkpoint) of a golden replay; its cheats.
    .venv/bin/python scripts/experiments/r32team_purse.py REPLAY"""
import sys
from triforce import _native

NAMES = {6: 'L3 in', 24: 'D3 out', 31: 'L4 in', 53: 'D4 out', 61: 'L1 in', 79: 'D1 out', 97: 'L5 in', 120: 'D5 out',
         145: 'L2 in', 155: 'D2 out', 174: 'L7 in', 194: 'D7 out', 211: 'L6 in', 227: 'D6 out', 233: 'L8 in',
         243: 'D8 out', 258: 'L9 in'}
data = open(sys.argv[1], 'rb').read()
r = _native.Replay(data)
player = _native.ReplayPlayer(data)
seen = set()
for e in r.events():
    if e.get('type') == 'room' and e.get('step') in NAMES and e['step'] not in seen:
        seen.add(e['step'])
        player.seek(e['frame'])
        ram = player.ram()
        print(f"step {e['step']:3d} {NAMES[e['step']]:7s} frame {e['frame']:6d}: hearts "
              f"{(ram[0x66F] & 0x0F) + ram[0x670] / 256:.2f} of {(ram[0x66F] >> 4) + 1}, bombs {ram[0x658]}, "
              f"keys {ram[0x66E]}, rupees {ram[0x66D]}")
print('segments:', [e for e in r.events() if e.get('type') == 'segment'])
cheats = r.cheats()
print('cheats:', len(cheats), cheats[:20])
