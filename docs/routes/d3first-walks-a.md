# d3first walks, first half: the game start to Level 5's entrance with the wood sword (round 32)

Walk32a, phase X (`d3first`, the owner's list and answers of Nov 19). This document is the first half of the route:
the wood sword from 0:0x77, the forced screen scroll into Level 3, Dungeon 3 the owner's way, the raft to Level 4,
Dungeon 4, the way to Level 1, Dungeon 1 on phase IX's route, heart rock 0x2c, the 30 secret 0x2d, the blue candle
at 0x0c and the scroll W into Level 5's screen 0x0b. Every number is observed unless marked otherwise; trials are
20 seeds each. Dungeon 3 is walked on one reading only: the lead's, which the owner confirmed ("5a is skipped on
the way in, used on the way back. 7b -> 4b -> 49 -> 59 -> 69 -> 59 -> 5c -> 4c -> 4d -> 3d"); the other reading
(back by 0x4b's E key door) was dropped on the lead's instruction.

Scripts (`scripts/experiments/`): `r32_walks.py` (staging, the fighter, the actions), `r32_d3first.py` (the route as
steps leg by leg, `stage`, `rooms`, `chain`, `scan`, `table`; `VARIANTS` and `BEST`).

## How it was walked

- **Staging** (`r32_d3first.py stage`): one game from `start.state` (0:0x77, nothing held, 3 hearts) along the whole
  half, the monsters pinned inside the NW wall each frame (round 29's `Stager`; the rupee stash of 4:0x11 left where
  it lies). Rooms the route must empty (keys and bombs on all dead, shutters, push blocks, the three bosses) have
  their monsters put into the death state, so the game banks the kills and shows the item as a played run finds it.
  Every step's pre-entry frame is saved under the route's state names (`target/r32-walks/states`, not committed).
  Nothing is written but: one key (1:0x33's key is carried by a Stalfos and the pinned carrier drops it inside the
  wall) and the rupees missing at the candle shop (the owner's rule: missing rupees are cheated and counted).
- **Trials** (`rooms`): each step's state loaded for seeds 0-19 with the game's own randomness, the monsters live,
  the hearts real (as the staging left them: full, the route's containers), the step's actions played by a rule
  over buttons: round 29's `Bot` (it keeps off squares a harmful object will reach and stabs what will be in the
  sword's box) with the Darknut's shield (never stabbed head-on), the sword beam at full hearts, drops taken, a laid
  ladder backed off, a held Link stabbing the Like-Like; Manhandla by bombs (`ManhandlaFighter`: a bomb where the
  base will be during the blast), Gleeok by round 28's rule (`r28_gleeok.GleeokFighter`, level 4's head type);
  CLIP by the session's own macro (`Game.act('CLIP E')`), as the agent plays it. "Hearts" is the mean lost in the
  room (drops picked up do not offset it); "frames" is from the room's first frame to standing in the next one.
- **What the numbers are**: the price a simple rule pays with the wood sword, not what a trained model will pay.
  Where the rule failed for reasons of its own (a path, a timing) the action was mended and the room run again; the
  failures left are the rule's real losses or are named as the rule's.

## The answers

| # | Item | Answer |
|---|---|---|
| 1 | The start | 0x77's cave gives the wood sword (the middle ware at X `$78`), 0x77 W, 0x76 W, 0x75's wrap exit (`CLIP E`) into 0x74, the mouth: **20/20, 0.07 hearts (0x76's Tektites), about 1,750 frames** with 0x76 walked; with 0x76 **scrolled** (`CLIP E` at its E edge on Y `$85`/`$8D`, then W) 0x76 costs 0 hearts and 141 frames against 387. d2first's `d3-approach` clip way works from the game start unchanged (its 0x77 is entered from the cave instead of from N: a new entry). |
| 2 | Dungeon 3 | Keys 0 → 2, bombs 0 → 4 (0x4a) → 8 (0x5b) → 6 (two walls), hearts 3 → 4. The wood sword's dear rooms: **0x59 (5 Red Darknuts, kill all: 11/20 by sword, 1.95 hearts; 17/20 with 0x4a's bombs, 1.15 hearts, 3 bombs), 0x69 (8 Darknuts, crossed to the stairs: 15/20 fighting on the way, 1.60; 19/20 dashing, 0.60), Manhandla (sword only 0/20 at 3 hearts; by bombs 11/20, 1.95 hearts, 2 bombs)**, 0x6b (5 Zols split into 10 Gels: 19/20, 0.93). Everything else under 0.3 hearts. |
| 3 | Level 3 → 4, Dungeon 4 | The leg is d2first's (0x74 W, 0x73, 0x63, 0x64, 0x65, the raft from 0x55): 20/20 every room, 0.6 hearts in all; 0x64 can be scrolled (0 hearts, 142 frames). Dungeon 4 with the wood sword: Vires split into red Keese on any wood hit (as with the white sword, `$20` < `$40`: no change); **Gleeok's two heads take 16 wood hits (8 white): round 28's rule kills him in 6 of 20 at the route's 4 hearts (14 die) and spends 5.5-6 hearts a kill from 8-16 hearts**; 0x02's traps 1 heart a crossing, 0x21's gels 0.93 (0.08 when killed first), 0x32's Like-Likes 0.90 (9 wood hits each). |
| 4 | Level 4 → 1, Dungeon 1 | The way: raft S to 0x55, E 0x56, N 0x46, E 0x47, E 0x48, N 0x38, W 0x37 (7 rooms; 0x47 scrollable). 0x46 is the only room new to the route. Dungeon 1 as phase IX: out of 1:0x73 S and back in, the N door then open (the staging kept its 3 keys); 0x53's N wall to 0x43 from S. Costs: Aquamentus 1.08 hearts (6 wood hits), 0x22's traps out of the bow cellar 0 hearts going round the bottom (1 heart straight), 0x23's Goriyas 0.78 hearts and 2 deaths of 20 for a key the route does not need (skipped: 0 hearts), 0x45 0 hearts (the Wallmasters stay in the walls). |
| 5 | After Level 1 | 0x37 E, 0x38 N, 0x28 E, 0x29, 0x2a, 0x2b E (each scrollable: 142 frames, 0 hearts), 0x2c's rock wall (one bomb facing N from (`$90`,`$A5`)), the heart container (right ware, X `$98`: 7 containers), 0x2d's rock wall (one bomb from (`$50`,`$5D`)), its 30 rupees (X `$78`), back W to 0x2c, N 0x1c, N 0x0c (0x1d's S part leads only E), the candle (right ware), the wrap exit W into 0x0b, the mouth (`$70`,`$85`) up. **Rupees at the shop: 40 (the 10 of 4:0x11's stash and 0x2d's 30) of 60: 20 cheated** with no drops. |
| 6 | Step table | 93 steps (§9). New rooms: 3:0x4b, 3:0x4a, 0:0x46, 0:0x29, 0:0x2a, 0:0x2b, the caves of 0x2d and 0x0c. New entries: 0x77 out of the sword cave W, 3:0x5b S→N, 3:0x49 E→S, 1:0x43 from S, 1:0x73's second entry, 0x56 W→N, 0x47 W→E, 0x38 S→W and W→N, 0x28 S→E, 0x2c from W and cave→E, 0x2d W→cave, 0x1c S→N, 0x0c S→cave and cave→W. |

## 1. The start (steps 0-5)

From `start.state`: up to (`$40`,`$5D`) and into the cave; the old man's text halts Link a moment; the sword is the
one ware, taken walking up at X `$78` (Y `$9D`); out at (`$40`,`$4D`). 0x77 W along Y `$8D`. 0x76 (4 Red
Tektites): its rocks block Y `$7D`/`$8D` in the middle, so the walk goes (`$90`,`$8D`), (`$90`,`$7D`), W; 0x75 is
entered on Y `$7D` or `$8D`, both in the wrap exit's span (Y `$65`-`$AD`). 0x75: `CLIP E` at the E edge lands in
0x74 at X `$F0` (140 frames, its Leevers, Peahats and Zora never reached). 0x74: (`$D0`,`$7D`), (`$D0`,`$8D`),
(`$80`,`$8D`), up into the mouth.

| Step | Way | Done | Hearts lost | Frames |
|---|---|---|---|---|
| 0x77 → cave | walk | 20/20 | 0 | 427 |
| sword cave | walk, take | 20/20 | 0 | 323 |
| 0x77 cave → W | walk | 20/20 | 0 | 203 |
| 0x76 E → W | walk round its rocks | 20/20 | 0.07 (max 0.5) | 387 (max 455) |
| 0x76 E → W | **scroll**: `CLIP E` at X `$F0`, then W | 20/20 | 0 | 141 |
| 0x75 E → W | wrap exit `CLIP E` | 20/20 | 0 | 140 |
| 0x74 E → cave | walk | 20/20 | 0 | 318 (max 430) |

The owner's "76 (screen scroll to skip enemies, walking is fine)" is the plain screen wrap the engine has offered
since round 21 (`zelda::clips` wrap W/E, offered within 32 px of the bound to every overworld family; `scan`:
0x76 on Y `$85`-`$8D` both ways). Both ways work; the scroll is cheaper and faster.

## 2. Dungeon 3 (steps 6-23), the wood sword at 3 hearts

| Step | Room, in → out | Monsters | Way | Done | Hearts | Frames |
|---|---|---|---|---|---|---|
| 6 | 0x7c S → W | — | round the statues (Y `$95`) | 20/20 | 0 | 322 |
| 7 | 0x7b E → N | 6 Zols | the key (lying), N | 20/20 | 0.03 | 380 |
| 8 | 0x6b S → N | 5 Zols | kill all (the wood sword splits each into 2 Gels: 10), the key | 19/20 | 0.93 (max 3) | 1,073 |
| 9 | 0x5b S → N | 3 Red Darknuts | walk, stabbing | 20/20 | 0.25 | 367 |
| 9 | | | **dash** (no stab) | 20/20 | 0 | 289 |
| 10 | 0x4b S → W | 3 Zols | kill all (6 Gels), the key, the W key door | 19/20 | 0.05 | 940 |
| 11 | 0x4a E → W | 5 blue Keese | kill all, the 4 bombs | 19/20 | 0.07 | 789 |
| 12 | 0x49 E → S | 3 Bubbles, 2 Zols, 3 Keese | the key (lying), S | 20/20 | 0.07 | 393 |
| 13 | 0x59 N → S | 5 Red Darknuts | kill all by sword (20 hits; the S shutter) | **11/20** | **1.95** | 1,509 |
| 13 | | | **with 0x4a's bombs** (`BombFighter`) | **17/20** | **1.15** | 1,120; bombs 4 → 1.1 |
| 14 | 0x69 N → stairs | 8 Red Darknuts | walk, stabbing | 15/20 | 1.60 | 502 |
| 14 | | | **dash** to the stairs (`$C8`,`$8D`) | **19/20** | **0.60** | 315 |
| 15 | cellar 0x0f | 4 Keese | the raft | 20/20 | 0.07 | 800 |
| 16 | 0x69 stairs → N | 8 Darknuts | walk | 20/20 | 0 | 234 |
| 17 | 0x59 S → E | — (emptied) | the key door E | 20/20 | 0 | 315 |
| 18 | 0x5a W → E | 4 Traps, 4 Keese | walk | 20/20 | 0.15 | 467 |
| 19 | 0x5b W → E | 3 Red Darknuts | kill all by sword, its 4 bombs, the E wall | 20/20 | 1.20 | 1,246; bombs → 7 |
| 19 | | | with bombs | 19/20 | 0.70 | 861; bombs spent, 5 cheated wall bombs |
| 20 | 0x5c W → N | 3 Red Darknuts | walk | 20/20 | 0.05 | 307 |
| 21 | 0x4c S → E | 4 Traps, 2 Zols | the E wall from (`$C0`,`$8D`) | 20/20 | 0.12 | 460 |
| 22 | 0x4d W → N | Manhandla | **by bombs** (6 held) | **11/20** | **1.95** | 859; bombs → 3.9 |
| 22 | | | by the sword alone (16 hits) | **0/20** | 3 (all) | — |
| 23 | 0x3d S → piece | — | the piece | 20/20 | 0 | 643 |

Ledger (staging, no drops): keys 0 → 0x7b 1 → 0x6b 2 → 0x4b 3, its W door 2 → 0x49 3 → 0x59's E door 2: **2 keys
out**, none cheated. Bombs: 0x4a 4 → 0x5b 8 → its wall 7 → 0x4c 6 → Manhandla (the rule spends about 2) → **about
4 out** (6 if Manhandla were killed without bombs). Hearts: 3 containers in, Manhandla's makes 4.

**What the wood sword costs here.** Red Darknut: 4 wood hits (2 white), each from the side or behind; 0x59 must be
emptied (its S shutter, the raft's way) and holds 5: 20 hits among 5 one-heart touches, at 3 hearts. Zols split
into Gels (the white sword kills them whole): 0x6b's 5 Zols become 10 Gels for its key. Manhandla: 4 wood hits a
hand (2 white); the sword alone never won at 3 hearts; a bomb on the base kills every hand at the start spacing
(enemies.md "Manhandla"), so this route's Manhandla is a bomb fight and 0x4a's and 0x5b's bombs (the owner's "2x4
bomb pickups") are what pay for it. The rule's sum over the dungeon at its best plays (dash, bombs in 0x59, bombs
on Manhandla): about 5.3 hearts lost against 3 + 1 containers and drops; the chance of passing the five dear rooms
in a row by these per-room rates is 0.95 × 0.85 × 0.95 × 0.55 ≈ 0.42 ([INFERENCE]: independent rooms, full hearts
at each; a run arrives with fewer). The trained families start from far better than this rule in d2first's
Darknut rooms; this is a lower bound on what the room can be played for.

## 3. Level 3 to Level 4 and Dungeon 4 (steps 24-48), 4 hearts

The leg is d2first's steps 105-111 room for room (same entries and exits); with the wood sword:

| Room | Hearts | Frames |
|---|---|---|
| 0x74 cave → W | 0 | 241 |
| 0x73 E → N (4 Blue Moblins) | 0.20 (arrows) | 310 |
| 0x63 S → E | 0.15 | 435 |
| 0x64 W → E | 0.23 (rocks); **scrolled** `CLIP W` 0, 142 frames | 452 |
| 0x65 W → N | 0.03 | 287 |
| 0x55 dock → raft | 0 | 251 |
| 0x45 pier → mouth | 0.03 | 271 |

Dungeon 4 (d2first's clip way: 0x31 N onto 0x21's island, the ladder clip, 0x11's N wall, 0x01's key, 0x02, 0x12
from N, Gleeok):

| Step | Room | Way | Done | Hearts | Frames |
|---|---|---|---|---|---|
| 31 | 0x71 S → N | walk | 20/20 | 0 | 253 |
| 32 | 0x61 S → N (3 Vires) | walk | 20/20 | 0.38 | 336 |
| 33 | 0x51 S → W (8 Keese) | the key (`$90`,`$8D`) | 20/20 | 0.08 | 560 |
| 34 | 0x50 E → N (5 Vires) | walk | 20/20 | 0.33 | 543 |
| 35 | 0x40 S → N (5 Zols) | the key from (`$78`,`$7D`) | 20/20 | 0.28 | 572 |
| 36 | 0x30 S → E | the key door | 20/20 | 0.10 | 468 |
| 37 | 0x31 W → E (5 Vires) | kill all (shutter) | 20/20 | 0.33 | 1,036 |
| 38 | 0x32 W → stairs | kill all (2 Like-Likes 9 wood hits each, 2 Zols), push from (`$50`,`$8D`) E, stairs (`$C8`,`$5D`) | 20/20 | 0.90 | 1,608 |
| 39 | cellar 0x60 | the ladder | 20/20 | 0.10 | 800 |
| 40 | 0x32 stairs → W | walk | 20/20 | 0 | 373 |
| 41 | 0x31 E → N | the key door onto the island | 20/20 | 0 | 369 |
| 42 | 0x21 S → N (5 Gels) | `CLIP N` from (`$78`,`$7D`), one step down, bomb N | 13/20 | 0.93 | 440 |
| 42 | | **the Gels killed first** | 18/20 | 0.08 | 982 |
| 43 | 0x11 S → N | the ten rupees (the diamond walked), bomb N | 20/20 | 0 | 458 |
| 44 | 0x01 S → E (8 Keese, dark) | the key on the water (the ladder; taken walking up X `$78` to Y `$6D`), E over one water square | 19/20 | 1.38 | 405 |
| 45 | 0x02 W → S (6 Traps) | walk | 20/20 | **1.00 (every run)** | 335 |
| 46 | 0x12 N → E (5 Vires) | kill all, push from (`$50`,`$8D`) E | 20/20 | 0.42 | 1,052 |
| 47 | 0x13 W → N | Gleeok (2 heads, 16 wood hits) | **6/20 killed** (14 died; the 6 then stopped on a script fault taking the heart) | 3.55 | — |
| 48 | 0x03 S → piece | the piece | 20/20 | 0 | 645 |

Keys: 2 in → 0x51 3 → 0x40 4 → 0x30's E door 3 → 0x31's N door 2 → 0x01 3: **3 out**. Bombs: 6 (staging) → 0x21 5
→ 0x11 4: **4 out** (about 2 with Manhandla's real cost). Rupees: 10 (the stash). Hearts: 4, Gleeok's makes 5.

**What the wood sword changes in Dungeon 4.** Vires: none (any sword below the magic one splits a Vire into two red
Keese; bombs kill it outright, as before). Like-Likes: 9 wood hits against 5 white. Gleeok: 16 wood hits against 8
white (10 for the first head, 6 for the second; asm:Gleeok_CheckSegmentCollisions), one swing per 32 frames of the
heads' shared timer: the fight lasts twice as long under the same fireballs (1 heart each). Round 28's rule, built
for the magic sword, from full hearts of N containers (`rooms --steps 4_13w --containers N`; "killed" counts the
fights won, which then stopped on a script fault at the heart container, fixed afterwards and not re-run):

| Containers (full) | Gleeok killed | Died | Hearts lost (mean) |
|---|---|---|---|
| 4 (the route's) | 6/20 | 14 | 3.55 |
| 8 | 13/20 | 7 | 5.50 |
| 12 | 19/20 | 1 | 6.00 |
| 16 | 19/20 | 1 | 6.20 |

The wood-sword fight costs this rule about 5.5-6 hearts (fireballs, 1 heart each), against 4 containers on the
route.

## 4. Level 4 to Level 1 and Dungeon 1 (steps 49-74), 5 hearts

**The way.** From 0x45's mouth S by raft to 0x55, then E 0x56, N 0x46, E 0x47, E 0x48, N 0x38, W 0x37: seven rooms.
The lead's "0x56, 0x57, 0x47, 0x37" has no edges between 0x57 and 0x47 or 0x47 and 0x37 (game.yaml: 0x57 only E/W,
0x47 only E/W, 0x37 only E); the equal-length alternative E 0x57, E 0x58, N 0x48 has two more Octorok rooms. No
screen scroll shortens it (the wraps only cross a room W↔E in place, `scan`: 0x47 on Y `$85`-`$9D`).

| Room | Done | Hearts | Frames |
|---|---|---|---|
| 0x45 mouth → raft S | 20/20 | 0 | 253 |
| 0x55 dock → E (off the dock first, straight down) | 13/20 (7: knocked onto the dock by the Zora's fireball, sailed back to 0x45) | 0.20 | 288 |
| 0x56 W → N | 20/20 | 0.03 | 260 |
| 0x46 S → E (new room; Zora) | 20/20 | 0 | 256 |
| 0x47 W → E (Zora) | 20/20 | 0; scrolled 0, 142 frames | 288 |
| 0x48 W → N (4 Red Leevers) | 20/20 | 0.20 | 426 |
| 0x38 S → W | 20/20 | 0.38 | 317 |
| 0x37 E → mouth | 20/20 | 0 | 311 |

**Dungeon 1** as phase IX has it, entered at 5 hearts with Dungeon 3's and 4's bombs (about 2-4):

| Step | Room | Way | Done | Hearts | Frames |
|---|---|---|---|---|---|
| 57 | 0x73 S → out S | at once back down | 20/20 | 0 | 213 |
| 58 | 0:0x37 mouth → in | one step down off the mouth, up | 20/20 | 0 | 196 |
| 59 | 0x73 S → N | the N door: open (the staging kept 3 keys through it) | 20/20 | 0 | 253 |
| 60 | 0x63 S → N (3 Stalfos) | walk | 20/20 | 0.05 | 360 |
| 61 | 0x53 S → N (5 Stalfos) | kill all, the key, bomb N from (`$78`,`$5D`) | 20/20 | 0.33 | 1,012 |
| 61 | | no key: straight to the wall | 17/20 (3: the bomb missed, Link knocked off the spot) | 0.40 | 505 |
| 62 | 0x43 S → N (5 Gels) | walk | 20/20 | 0 | 372 |
| 63 | 0x33 S → N (3 Stalfos) | kill all, **the key a Stalfos carries** (it drops where the carrier dies), the N key door | 15/20 (the rule's water walking) | 0.18 | 985 |
| 64 | 0x23 S → W (3 Red Goriyas) | kill all, the key, the W key door | 16/20 (2 dead) | 0.78 | 1,004 |
| 64 | | **no key** | **20/20** | **0** | 367 |
| 65 | 0x22 E → stairs | `CLIP N` from (`$A0`,`$9D`), W | 20/20 | 0 | 220 |
| 66 | cellar 0x7f | the bow | 20/20 | 0.03 | 794 |
| 67 | 0x22 stairs → E | round the bottom, timed by the traps | 20/20 | 0 | 532 |
| 67 | | straight | 20/20 | 1.00 (a trap every run) | 356 |
| 68-70 | 0x23 W → S, 0x33 N → S, 0x43 N → E | walk | 20/20 each | 0-0.05 | 270-390 |
| 71 | 0x44 W → E (3 Red Goriyas) | kill all, the boomerang | 20/20 | 0.17 | 814 |
| 72 | 0x45 W → N (8 Wallmasters) | the key (`$A0`,`$BD`), the N key door | 20/20 | 0 | 513 |
| 73 | 0x35 S → E | Aquamentus (6 wood hits), the heart (`$C0`,`$8D`) | 19/20 | 1.08 | 626 |
| 74 | 0x36 W → piece | | 20/20 | 0 | 729 |

Keys: 3 in; the N door opened by the trick; 0x53 4 → 0x33 5, its N door 4 → 0x23 5, W door 4 → 0x43's E door 3 →
0x45 4, N door 3: **3 out** with every key room fought; **1 out** with 0x23's skipped (and 0x53's), so 0x23's Goriya
fight (the costliest Dungeon 1 room for this rule) is not needed. Bombs: 0x53's wall −1. Hearts: 5, Aquamentus's
makes 6. **The wood sword in Dungeon 1**: Stalfos 2 hits, Red Goriya 3, Aquamentus 6, Wallmasters 2 (level-1.md:
the starting sword is enough); the costs are small but for 0x23 (skippable) and Aquamentus.

## 5. After Level 1 to Level 5 (steps 75-92), 6 then 7 hearts

| Room | Way | Done | Hearts | Frames |
|---|---|---|---|---|
| 0x37 mouth → E | | 20/20 | 0 | 216 |
| 0x38 W → N | | 20/20 | 0.03 | 268 |
| 0x28 S → E (6 Peahats) | | 20/20 | 0 | 254 |
| 0x29 W → E (4 Red Leevers) | walk / **scroll** | 20/20 / 20/20 | 0.17 / 0 | 478 / 142 |
| 0x2a W → E (6 Blue Leevers) | walk / **scroll** | 18/20 / 20/20 | 1.00 / 0 | 569 / 142 |
| 0x2b W → E (Leevers, Peahats) | walk / **scroll** | 16/20 / 20/20 | 0.15 / 0 | 511 / 142 |
| 0x2c W → cave (4 Red Tektites) | the rock wall: one bomb facing N from (`$90`,`$A5`) | 20/20 | 0.28 | 492 |
| heart cave | the right ware (X `$98`): the 7th container | 20/20 | 0 | 373 |
| 0x2c cave → E | | 20/20 | 0.03 | 242 |
| 0x2d W → cave | the rock wall: one bomb facing N from (`$50`,`$5D`) | 20/20 | 0.23 | 474 |
| secret cave | 30 rupees, walking up at X `$78` | 20/20 | 0 | 201 |
| 0x2d cave → W | | 20/20 | 0 | 227 |
| 0x2c E → N | X `$30` | 19/20 | 0.05 | 410 |
| 0x1c S → N | X `$50` | 20/20 | 0 | 257 |
| 0x0c S → cave (6 Red Tektites) | the mouth from (`$80`,`$55`) | 20/20 | 0.15 | 555 |
| candle shop | 60 rupees (20 cheated), the right ware | 20/20 | 0 | 372 |
| 0x0c cave → W | the wrap exit `CLIP E` on Y `$8D` | 20/20 | 0.12 | 352 |
| 0x0b E → mouth | (`$70`,`$85`), up into Level 5 | 20/20 | 0 | 324 |

**The way and the scrolls.** The skip search (`r32_d3first.py scan`: `Game.skip_available` on every floor grid point,
monsters emptied) offers on this leg the plain screen wrap W/E in 0x28 (Y `$55`-`$8D`), 0x29, 0x2a, 0x2b (Y
`$55`-`$BD`), 0x2c, 0x2d (Y `$85`-`$BD`), 0x1c (Y `$85`-`$8D`), and the wrap exit only in 0x0c (Y `$65`-`$AD`, into
0x0b). Every W → E crossing of 0x29-0x2b is a scroll (`CLIP W` on arrival at X 0, then E): 0 hearts and 142 frames
each against up to 1 heart and 570 frames walking. **0x2d → 0x0c**: 0x1d (the lead's way) is two parts from S, and
its S part leads only E (to 0x1e and back in from E, five rooms); the walked way is back W into 0x2c, N to 0x1c
(empty), N to 0x0c's S edge: three rooms, 0x2c's Tektites twice. **0x0c → 0x0b** is d2first's `mountain` clip way's
last step (the wrap exit), now entered from 0x0c's shop; the plain way into 0x0b is through the Lost Hills (0x1b's
four Ns, from 0x1c W), which d2first's plain `mountain` way already plays.

**Rupees at the shop.** 4:0x11's stash (10, the diamond walked) + 0x2d's 30 = 40; none else on the way without
drops: **20 short of the candle's 60**, cheated and counted. Drops in a played run add some ([INFERENCE]: the
rule's runs were single rooms; chains were not run).

## 6. Ledgers through the half (staging: no drops, Manhandla's bombs not spent)

| After | Containers | Keys | Bombs | Rupees |
|---|---|---|---|---|
| the sword | 3 | 0 | 0 | 0 |
| Dungeon 3 | 4 | 2 | 6 (the rule: about 4) | 0 |
| Dungeon 4 | 5 | 3 | 4 (about 2) | 10 |
| Dungeon 1 | 6 | 3 (1 skipping 0x23 and 0x53) | 3 (about 1) | 10 |
| 0x2c's heart, 0x2d | 7 | 3 | 1 (about 0: cheated) | 40 |
| the candle, Level 5's mouth | 7 | 3 | 1 | 0 (20 cheated) |

The owner's "5-8 bombs leaving Level 1" is not met by the rooms (1-3 left before 0x2c's and 0x2d's two walls;
cheated as the owner allows). No key is cheated anywhere in the half.

## 7. The wood sword's cost per dungeon

| Dungeon | Containers in | Rooms the wood sword makes dear (the rule, 20 seeds) | Sum of mean hearts lost, best ways |
|---|---|---|---|
| 3 | 3 | 0x59 (5 Darknuts, 20 hits; 1.15 with bombs), 0x69 crossing (0.60 dashing), 0x6b (Gels, 0.93), Manhandla (bombs only: 1.95) | about 5.3 |
| 4 | 4 | Gleeok (16 hits; 6/20 killed at 4 hearts, 5.5-6 hearts a kill), 0x32's Like-Likes (0.90), 0x02's traps (1.00), 0x01's Keese (1.38) | about 6.3 without Gleeok's win |
| 1 | 5 | Aquamentus (1.08); the rest under 0.4 once 0x23 is skipped | about 2.0 |
| the legs | — | 0x2a walked (1.00: scrolled 0) | under 1 per leg with scrolls |

## 8. Not done (the budget ran out)

- Chains (`r32_d3first.py chain --legs d3 --best --seeds 20` and the others): every leg in one game with hearts
  carried and drops taken. Only per-room trials from full hearts were run.
- Gleeok's won fights stopped at the heart container on a script fault (the floor graph after the fight; fixed in
  `walk_graph` after the runs, not re-run): the kill counts above are the fights, not the room.
- 0x33's carried key and 0x23's water ring: the rule fails 5 and 4 of 20 on its own walking in water rooms with the
  ladder held, not on the fight.

## 9. The half as route steps

`r32_d3first.py table` (family: routes.yaml's `families`; "(new)" is this walk's proposal for an untagged room; "On
d2first": whether `d2first` has a step in the room with the same entry and exit). Each step's actions are in
`r32_d3first.py` `LEGS`; the best play where it differs is `BEST`.

| # | State | Room | In | Out | Taken / done | Family | On d2first |
|---|---|---|---|---|---|---|---|
| 0 | `start` | 0:0x77 | start | cave |  | ow-start | same entry and exit |
| 1 | `0_77cave` | 0:0x77 cave | S | S | sword | ow-start | same entry and exit |
| 2 | `0_77c` | 0:0x77 | cave | W |  | ow-start | same entry, new exit (N) |
| 3 | `0_76e` | 0:0x76 | E | W |  | ow-tektite | same entry and exit |
| 4 | `0_75e` | 0:0x75 | E | W | CLIP E (wrap exit) | ow-leever | same entry and exit |
| 5 | `0_74e` | 0:0x74 | E | cave |  | ow-tektite | same entry and exit |
| 6 | `3_7cs` | 3:0x7c | S | W |  | zol | same entry and exit |
| 7 | `3_7be` | 3:0x7b | E | N | key | zol | same entry and exit |
| 8 | `3_6bs` | 3:0x6b | S | N | clear, key | zol | same entry and exit |
| 9 | `3_5bs` | 3:0x5b | S | N |  | darknut | same entry, new exit (W) |
| 10 | `3_4bs` | 3:0x4b | S | W | clear, key | zol | new room |
| 11 | `3_4ae` | 3:0x4a | E | W | clear, bombs | stalfos-keese-gel (new) | new room |
| 12 | `3_49e` | 3:0x49 | E | S | key | zol | new entry (S->S) |
| 13 | `3_59n` | 3:0x59 | N | S | clear | darknut | same entry and exit |
| 14 | `3_69n` | 3:0x69 | N | cave |  | darknut | same entry and exit |
| 15 | `3_0fcave` | 3:0x0f cave | S | N | raft | equipment-room | same entry and exit |
| 16 | `3_69c` | 3:0x69 | cave | N |  | darknut | same entry and exit |
| 17 | `3_59s` | 3:0x59 | S | E |  | darknut | same entry and exit |
| 18 | `3_5aw` | 3:0x5a | W | E |  | trap-room | same entry and exit |
| 19 | `3_5bw` | 3:0x5b | W | E | clear, bombs, bomb RIGHT | darknut | same entry and exit |
| 20 | `3_5cw` | 3:0x5c | W | N |  | darknut | same entry and exit |
| 21 | `3_4cs` | 3:0x4c | S | E | bomb RIGHT | trap-room | same entry and exit |
| 22 | `3_4dw` | 3:0x4d | W | N | clear, heart | boss-manhandla | same entry and exit |
| 23 | `3_3ds` | 3:0x3d | S | triforce | triforce | zol | same entry and exit |
| 24 | `0_74c` | 0:0x74 | cave | W |  | ow-tektite | same entry and exit |
| 25 | `0_73e` | 0:0x73 | E | N |  | ow-moblin | same entry and exit |
| 26 | `0_63s` | 0:0x63 | S | E |  | ow-moblin | same entry and exit |
| 27 | `0_64w` | 0:0x64 | W | E |  | ow-octorok | same entry and exit |
| 28 | `0_65w` | 0:0x65 | W | N |  | ow-zora | same entry and exit |
| 29 | `0_55s` | 0:0x55 | S | N |  | ow-zora | same entry and exit |
| 30 | `0_45s` | 0:0x45 | S | cave |  | ow-peahat | same entry and exit |
| 31 | `4_71s` | 4:0x71 | S | N |  | vire | same entry and exit |
| 32 | `4_61s` | 4:0x61 | S | N |  | vire | same entry and exit |
| 33 | `4_51s` | 4:0x51 | S | W | key | stalfos-keese-gel | same entry and exit |
| 34 | `4_50e` | 4:0x50 | E | N |  | vire | same entry and exit |
| 35 | `4_40s` | 4:0x40 | S | N | key | zol | same entry and exit |
| 36 | `4_30s` | 4:0x30 | S | E |  | vire | same entry and exit |
| 37 | `4_31w` | 4:0x31 | W | E | clear | vire | same entry and exit |
| 38 | `4_32w` | 4:0x32 | W | cave | clear, push | like-like | same entry and exit |
| 39 | `4_60cave` | 4:0x60 cave | S | N | ladder | equipment-room | same entry and exit |
| 40 | `4_32c` | 4:0x32 | cave | W |  | like-like | same entry and exit |
| 41 | `4_31e` | 4:0x31 | E | N |  | vire | same entry and exit |
| 42 | `4_21s` | 4:0x21 | S | N | CLIP N, bomb UP | stalfos-keese-gel | same entry and exit |
| 43 | `4_11s` | 4:0x11 | S | N | bomb UP | trap-room | same entry and exit |
| 44 | `4_01s` | 4:0x01 | S | E |  | stalfos-keese-gel | same entry and exit |
| 45 | `4_02w` | 4:0x02 | W | S |  | trap-room | same entry and exit |
| 46 | `4_12n` | 4:0x12 | N | E | clear, push | vire | same entry and exit |
| 47 | `4_13w` | 4:0x13 | W | N | clear, heart | boss-gleeok | same entry and exit |
| 48 | `4_03s` | 4:0x03 | S | triforce | triforce | vire | same entry and exit |
| 49 | `0_45c` | 0:0x45 | cave | S |  | ow-peahat | same entry and exit |
| 50 | `0_55n` | 0:0x55 | N | E |  | ow-zora | same entry and exit |
| 51 | `0_56w` | 0:0x56 | W | N |  | ow-zora | same entry, new exit (E) |
| 52 | `0_46s` | 0:0x46 | S | E |  | ow-octorok (new) | new room |
| 53 | `0_47w` | 0:0x47 | W | E |  | ow-zora | new entry (E->cave, cave->E) |
| 54 | `0_48w` | 0:0x48 | W | N |  | ow-leever | same entry and exit |
| 55 | `0_38s` | 0:0x38 | S | W |  | ow-zora | same entry, new exit (N) |
| 56 | `0_37e` | 0:0x37 | E | cave |  | ow-octorok | same entry and exit |
| 57 | `1_73s` | 1:0x73 | S | S (out to the overworld) |  | stalfos-keese-gel | same entry, new exit (W) |
| 58 | `0_37c` | 0:0x37 | cave | cave |  | ow-octorok | same entry, new exit (E) |
| 59 | `1_73s_v2` | 1:0x73 | S | N |  | stalfos-keese-gel | same entry, new exit (W) |
| 60 | `1_63s` | 1:0x63 | S | N |  | stalfos-keese-gel | same entry and exit |
| 61 | `1_53s` | 1:0x53 | S | N | clear, key, bomb UP | stalfos-keese-gel | same entry, new exit (W) |
| 62 | `1_43s` | 1:0x43 | S | N |  | stalfos-keese-gel | new entry (N->E, W->N) |
| 63 | `1_33s` | 1:0x33 | S | N | clear, key | stalfos-keese-gel | same entry and exit |
| 64 | `1_23s` | 1:0x23 | S | W | clear, key | goriya | same entry and exit |
| 65 | `1_22e` | 1:0x22 | E | cave | CLIP N | trap-room | same entry and exit |
| 66 | `1_7fcave` | 1:0x7f cave | S | N | bow | equipment-room | same entry and exit |
| 67 | `1_22c` | 1:0x22 | cave | E |  | trap-room | same entry and exit |
| 68 | `1_23w` | 1:0x23 | W | S |  | goriya | same entry and exit |
| 69 | `1_33n` | 1:0x33 | N | S |  | stalfos-keese-gel | same entry and exit |
| 70 | `1_43n` | 1:0x43 | N | E |  | stalfos-keese-gel | same entry and exit |
| 71 | `1_44w` | 1:0x44 | W | E | clear, boomerang | goriya | same entry and exit |
| 72 | `1_45w` | 1:0x45 | W | N | key | wallmaster | same entry and exit |
| 73 | `1_35s` | 1:0x35 | S | E | clear, heart | boss-aquamentus | same entry and exit |
| 74 | `1_36w` | 1:0x36 | W | triforce | triforce | stalfos-keese-gel | same entry and exit |
| 75 | `0_37c_v2` | 0:0x37 | cave | E |  | ow-octorok | same entry and exit |
| 76 | `0_38w` | 0:0x38 | W | N |  | ow-zora | same entry, new exit (S) |
| 77 | `0_28s` | 0:0x28 | S | E |  | ow-zora | same entry, new exit (W) |
| 78 | `0_29w` | 0:0x29 | W | E |  | ow-leever (new) | new room |
| 79 | `0_2aw` | 0:0x2a | W | E |  | ow-leever (new) | new room |
| 80 | `0_2bw` | 0:0x2b | W | E |  | ow-leever (new) | new room |
| 81 | `0_2cw` | 0:0x2c | W | cave | bomb UP | ow-tektite | new entry (E->N, E->cave, cave->N) |
| 82 | `0_2ccave` | 0:0x2c cave | S | S | heart | ow-tektite | same entry and exit |
| 83 | `0_2cc` | 0:0x2c | cave | E |  | ow-tektite | same entry, new exit (N) |
| 84 | `0_2dw` | 0:0x2d | W | cave | bomb UP | ow-zora | new entry (E->N, N->W, S->N, S->W) |
| 85 | `0_2dcave` | 0:0x2d cave | S | S | rupees | ow-zora (new) | new room |
| 86 | `0_2dc` | 0:0x2d | cave | W |  | ow-zora | new entry (E->N, N->W, S->N, S->W) |
| 87 | `0_2ce` | 0:0x2c | E | N |  | ow-tektite | same entry and exit |
| 88 | `0_1cs` | 0:0x1c | S | N |  | ow-start | same entry, new exit (W) |
| 89 | `0_0cs` | 0:0x0c | S | cave |  | ow-tektite | new entry (E->W) |
| 90 | `0_0ccave` | 0:0x0c cave | S | S | 60 rupees, candle | ow-tektite (new) | new room |
| 91 | `0_0cc` | 0:0x0c | cave | W | CLIP E (wrap exit) | ow-tektite | new entry (E->W) |
| 92 | `0_0be` | 0:0x0b | E | cave |  | ow-leever | same entry and exit |

**What `d3first` needs that the engine and data do not have** (for the agent that writes routes.yaml; nothing of it
was built here):

- **A route and a state namespace.** `d3first` in routes.yaml with the steps above (a game-start step 0 as
  d2first's), and states of its own: every room name above exists in d2first's `triforce/states/` with other
  inventories (white sword, more hearts); round 22's audit ties a state's hearts to the route. Proposed:
  `triforce/states/d3first/<name>_<chain>.state`, and a per-route `promoted` pointer only where a family is trained
  for this route.
- **Stretches.** 0x76's scroll is the plain wrap (offered today); as a route step 0x76 E → W is unchanged, the policy
  chooses the wrap. The same for 0x64, 0x47, 0x29, 0x2a, 0x2b W → E: no new vocabulary.
- **Leaving a dungeon by its entrance** (1:0x73 S to 0:0x37 and back): no game.yaml exit exists (the start rooms'
  S exits are omitted); the phase IX agent who walks the door trick owns it. Here: step 57 `1_73s` exit S into
  `0_37c` (exit cave), then `1_73s_v2` (visit 2).
- **game.yaml**: 0:0x2d's rock wall as `requires_bombs` with `bomb_secret: [10, 2]` and the cave's 30 rupees as the
  route's `waypoint: rupees` (as 0x67's); 0:0x2c entered from W (its `bomb_secret` exists); the candle shop of 0x0c
  with `requires_rupees: 60` and `waypoint: candle`; the tags of 0:0x46 (ow-octorok), 0:0x29, 0x2a, 0x2b
  (ow-leever), 3:0x4a (stalfos-keese-gel), the caves of 0x2d (ow-zora) and 0x0c (ow-tektite).
- **3:0x4b, 0x4a, 0x49 on the route** (keys on all dead in 0x4b, bombs on all dead in 0x4a): `treasure` as default;
  0x5b's first pass `treasure: skip` and its last pass as d2first's (`bomb_reserve: 0`).
- **1:0x33's key is carried by a Stalfos** (it drops where its carrier dies): whatever d2first's objective does
  there holds; 1:0x23 and 1:0x53 may be `treasure: skip` (keys: §4).

## For the owner

1. **"Level 4 → make your way to Level 1"; the lead's 0x55, 0x56, 0x57, 0x47, 0x37.** The map has no N edge out of
   0x57 or 0x47 and 0x37 opens only E. Walked instead: 0x55 E, 0x56 N, 0x46 E, 0x47 E, 0x48 N, 0x38 W, 0x37 (7 rooms,
   0x46 new to the route; 0x47 scrollable). No scroll shortens it. Cost: about 0.8 hearts with the rule.
2. **"Go one screen east (0x2d) ... make your way to the screen east of Level 5."** 0x1d, N of 0x2d, is two parts and
   the part entered from S leads only E. Walked instead: back W into 0x2c, N 0x1c, N 0x0c (three rooms, 0x2c's
   Tektites twice, 0.05 hearts); the 0x1d way is 0x1d, 0x1e, 0x1d from E, 0x0d, 0x0c (five).
3. **"5-8 bombs leaving Level 1".** The rooms give 8 in Dungeon 3 (0x4a, 0x5b) and nothing after; the walls (0x5b,
   0x4c, 0x21, 0x11, 0x53) and Manhandla's bombs leave 1-3 at Level 1's exit and none after 0x2c's and 0x2d's walls.
   As you allow: cheated and counted.
4. **"About 30 rupees leaving Level 1".** The route's own rupees are 4:0x11's 10; with 0x2d's 30 Link has 40 at the
   candle: 20 cheated without drops.
5. **Dungeon 3 with the wood sword at 3 hearts** is the hard part of the half for a rule: 0x59's five Darknuts and
   Manhandla each lose about half the runs by the sword; 0x4a's bombs in 0x59 (17/20) and bombs on Manhandla are the
   way the rule survives. Nothing was changed in the route for it.
6. **Dungeon 1's key rooms.** With phase IX's door trick Link has keys to spare: 0x23's Goriya key (the costliest
   room for the rule: 2 deaths of 20) and 0x53's key are not needed (one key left at Level 1's exit). Walked both
   ways; the step table keeps them as today's route.
7. **0x76** ("screen scroll to skip enemies, walking is fine"): both walked; the scroll costs nothing and saves 246
   frames.
