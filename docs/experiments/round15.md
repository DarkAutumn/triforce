# Round 15: the overworld, cheated items and Dungeon 3

Lee, Nov 9 (while away): after the bow, map the whole overworld (with the two maze rooms), map Dungeon 3, route on
to D3 after D1 and try to train it; formalize cheated items (bombs, keys, rupees); research kill streaks and forced
drops (for later); check Darknut shield clinks aren't counted as blocks.

## Summary

- **Overworld and D3 mapped** (`9b8376a`, merged `03e89c3`): all 128 overworld rooms in game.yaml (exits verified
  by walking every edge, enemies from the ROM, 71 cave entrances, dungeon entrances L1–L9), map
  `docs/routes/overworld-map.png`; mazes in code and yaml (0x1b: four Ns reach 0x0b, not five; 0x61: N W S W
  reaches 0x60); D3's 18 rooms + raft cellar, `docs/routes/d3.md`/`d3-map.png`. Lee's D3 directions match the real
  layout exactly.
- **Cheated items** (`999aa5a`, merged `10f8186`): one ledger (`zelda::cheats`) for bombs, keys and rupees.
- **Kill streaks / forced drops** researched in the ASM and written into backlog.md (implementation later).
- **Darknut clinks** are not blocks or hits (by construction; probe; test `75e3a20`).
- **Route through D3** (`d76906b`): ends on the D3 triforce; treasure rules (`skip`, `if_bombs_below: 4`); bomb-wall
  exit; families `ow-leever`, `zol`, `darknut`, `boss-manhandla`. 295 states, audit 0 problems.
- **Training (r54):** new families darknut 97%, zol 99%, boss-manhandla 91% (3.8 bombs per fight, 1.3 cheated),
  ow-leever 100%; changed families continued: trap-room 99%, equipment-room 100%, ow-octorok 99%, ow-zora 99%,
  ow-moblin 98%, ow-tektite 99%. All promoted.
- **Chains (120 runs):** D3 from its entrance at full health **18/120 (15%)**; from D1's exit (0x37) 7/120;
  full route 0/120. D3 losses: 0x59 (Darknuts; entered three times) 41, 0x69 (8 Darknuts on the way to the raft
  stairs) 22, Manhandla 15, 0x5b 9. Link loses ~1 heart per 0x59 visit and reaches 0x69 with ~2.3 of 6 hearts.
  Continuing darknut and boss-manhandla (r55).

## 1. Darknut clinks (Lee's question)

A block is counted only when a blockable enemy shot (types $53–$5C) is in flight before the frame and bounces with
its signature next to Link, with the parry sound requested, no health lost and Link facing it
(`crates/zelda/src/blocks.rs`). A Darknut has no shot, so his clink ($604 bit 0) can't count; hits count enemy HP
lost, so it can't pay a hit either. Probe in 3:0x5b: a face stab plays the clink, HP stays $40, the step counts 0
blocks / 0 hits; a back stab takes HP $40 → $20 and counts a hit. Darknut training (695 updates): 0 blocks.

## 2. Darknut per entry (r54 eval, 100 per state)

0x59 from E 98–99%, **from N 83–86% (−2 hearts)**, from S 100%; 0x5b from S 99–100%, from W 93–96%; 0x69 from N
96–100% (−0.9), from the cellar 100%. The weak entries are where Link meets the Darknuts head-on.

## 3. Stopped

The r55 continuation of darknut and boss-manhandla (8M more steps each) was cancelled by Lee (Nov 9): the 8M-step
r54 models are the baseline; Lee has new ideas for improving training to discuss first. The promoted team is r54's
(plus earlier rounds for unchanged families).
