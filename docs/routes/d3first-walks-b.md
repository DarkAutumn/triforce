# `d3first`, the second half: from Level 5 to the ending, walked (round 32, Nov 19)

Phase X of the owner's notes (the owner's list and answers of Nov 19, kept with the lead's reading in the round's
notes): a second route, `d3first`, beside `d2first`, to become the main route once a golden path on it reaches the
ending. This document walks its second half in the running game before anything of it is built: the recorder's four
warps, Dungeon 5 with the wood sword, the leg from the warp's landing at Level 3 to Level 2 (the dead woods' 100,
the world wrap's ladder heart, the raft heart, the Armos's 30), Dungeon 2 at ten hearts, the leg to Level 7 and
Dungeon 7 with the wood sword, and d2first's legs after Level 7 with this route's inventory. The first half (the start
to Level 5, Dungeon 3's changed way in, the purchases and secrets before Level 5) is another walk's.

Every number was observed (**walked**) unless a line says otherwise; where chance is involved it is from 20 trials,
each loaded with the game's own randomness for its seed (`Game.load(..., randomize_rng=True, seed=s)`), monsters
live. Scripts (`scripts/experiments/`): `r32b_walks.py` (staging, `d3first`'s inventory by milestone),
`r32b_recorder.py` (§1), `r32b_d5.py` (§2), `r32b_leg.py` (§3), `r32b_d7.py` (§5), `r32b_late.py` (§6). Run them
from `scripts/experiments` with the checkout's venv under `nice -n 10`. Nothing here changed engine code or route
data; what the route data and the engine must gain is written in §8 for the agent that builds it.

**Staging.** Out of a dungeon, `triforce.probe` puts Link into the level's triforce room with what `d3first` holds
there (`r32b_walks.D3`), he walks onto the piece and **the game itself** ends the level and sets him down at its
entrance (`r32b_walks.out_of`): his first playable frame, facing down, full hearts. On the overworld a staging walk
from such an exit (monster slots emptied every frame, the woken Armos kept) saves each room's pre-entry state; the
trials load those. Dungeon rooms are d2first's own pre-entry states (`5_05e_<c>`, `5_24e_<c>`) or the probe with
the route's ledger (`r26_walks.d7_state`), with the sword, hearts and bombs written. `d3first` by milestone (the
owner's list; the first half's ledger for keys, bombs and rupees, which the walks here write where they matter):

| Where | Containers | Sword | Pieces | Items besides the first half's (raft, ladder, bow, boomerang, blue candle) |
|---|---|---|---|---|
| Level 5's exit 0:0x0b | 8 | wood | 1, 3, 4, 5 | recorder |
| Level 2's entrance 0:0x3c | 10 (0x5f's, 0x2f's) | wood | 1, 3, 4, 5 | 100 + 30 rupees taken on the way |
| Level 2's exit | 11 | wood | 1-5 | |
| Level 7's entrance 0:0x42 | 11 | wood | 1-5 | arrows (0x44), the meat (0x34) |
| Level 7's exit | 12 | wood | 1-5, 7 | the red candle (7:0x4a), the meat given |
| Level 6's entrance | 12 | magic (0:0x21) | 1-5, 7 | |
| Level 6's exit | 13 | magic | 1-7 | |
| Level 8's exit | 14 | magic | 1-8 | |
| Level 9's entrance | 15 (0:0x47) | magic | 1-8 | |

## 1. The recorder's four warps

**The rule** (asm:Whirlwind_Summon, asm:Whirlwind_StepLevelIndex, asm:Whirlwind_PrevRoomIds,
asm:Whirlwind_DropOffYs; `zelda-asm/items.md` "Recorder"). Every play on the overworld in mode 5 (but in 0x42, the
pond) steps `TeleportingLevelIndex` (`$523`): +1 facing right or up, −1 facing left or down, then on the same way past
every level whose piece Link lacks (bit level − 1 of `InvTriforce`). The destination is the index's level AND 7 + 1;
the landing room is the one E of the table's (0x37, 0x3c, 0x74, 0x45, 0x0b, 0x22, 0x42, 0x6d for levels 1-8), Link
released at X `$80` on the level's Y (`$8D` but `$AD` for levels 2, 5, 7 and `$5D` for 8). Only the first play of a
room entry makes a whirlwind; a later play before the pickup steps the index again and the whirlwind carries Link to
where the index then points. The index is never cleared but at power on (`start.state` holds 0, walked) and a play
inside a dungeon does not step it, so on `d3first` it is 0 until the first warp, and each warp leaves it on the
level it went to. The route's warps, computed and then played:

| Out of | Pieces held | `$523` (AND 7) | The route's play | Lands | Frames B → set down | 20 seeds |
|---|---|---|---|---|---|---|
| Level 5, 0:0x0b | 1, 3, 4, 5 | 0 | **1 play facing right (or up)**: 0 → 1 (no L2) → 2 | 0x74 (L3), (`$80`,`$8D`) | 479 (with the 3-frame turn) | 20/20, 0 hearts |
| Level 2, 0:0x3c | 1-5 | 2 | **1 play facing right (or up)**: 2 → 3 | 0x45 (L4), (`$80`,`$8D`) | 479 | 20/20, 0 hearts |
| Level 6, 0:0x22 | 1-7 | 3 | **2 plays facing down** (as he comes out): 3 → 2 → 1 | 0x3c (L2), (`$80`,`$AD`) | 622 | 20/20, 0 hearts |
| Level 8, 0:0x6d | 1-8 | 1 | **1 play facing down**: 1 → 0 | 0x37 (L1), (`$80`,`$8D`) | 476 | 20/20, 0 hearts |

Every facing and count was played at each exit (`r32b_recorder.py`, the full table in its output); the ones that
matter:

- Out of Level 5 facing down (as Link comes out): 1 play lands in 0x0b itself (index `$FC`: 7, 6 and 5 skipped, L5),
  2 plays in 0x45, **3 plays in 0x74** (622 and 768 frames: three plays fit before the pickup). Facing left as down.
  Facing right or up: 1 play 0x74, 2 plays 0x45, 3 plays 0x0b.
- Out of Level 2 facing down: 1 play lands in 0x3c itself, 2 plays in 0x37 (L1), 3 in 0x0b; reaching 0x45 that way
  takes 4 (not played). Facing right or up: 1 play 0x45.
- Out of Level 6 facing right or up: 0x0b, 0x22, 0x42 for 1, 2, 3 plays; down: 0x74, **0x3c**, 0x37.
- Out of Level 8 facing down: **0x37**, 0x6d, 0x42; right or up: 0x74, 0x45, 0x0b.

The last two warps are d2first's own (its index there is `$FB` and `$F9`, the same AND 7: d2first's first warp went
down from 0, this route's up). The first two are new: **out of Levels 5 and 2 Link must face right (or up) before he
plays, and he comes out facing down.** Walked: 2 frames of Right (or of Up) turn him; a longer press up walks him
back toward the dungeon's mouth [inferred], so the turn is one MOVE E.

**The engine's vocabulary covers all four.** A route step `exit: recorder` with the next step's room a landing room
(`game_map::whirlwind_level`) gets its plays and facings from the game's own index and pieces at run time
(`crates/zelda/src/whirlwind.rs::plays`, its tests already hold 0 → `$FC` → `$FB` and the after-D6 and after-D8
counts); the session offers RECORDER only facing the way the next play must step (`objectives::recorder_facings`).
For 0x0b and 0x3c it computes 1 play facing up or right (the down way takes 3 and 4): RECORDER is masked while Link
faces down, so the family must turn first (MOVE E) and then play. Nothing new is needed in the engine; what is new is
that behaviour for ow-leever (0x0b; it plays out of 0x0b facing down on d2first) and for ow-octorok (0x3c), which
**has no `items: {recorder: true}`** today (d2first only lands in 0x3c): §8.

## 2. Dungeon 5 with the wood sword (7 containers on entry)

Today's Dungeon 5 route (`d5.md`, d2first steps 186-208) unchanged: 0x64 by the block clip (no fight), the Olive
Garden 5:0x05 emptied for the push (its six Blue Darknuts), the recorder in 0x04, the way back (0x64's Darknuts
crossed, 0x66 fought for its key and N shutter), Digdogger. `d3first` brings **7 containers** (3, Dungeons 3, 4 and
1, the heart rock; Digdogger's is the 8th; d2first brings 10) and the wood sword (d2first the white one).

**What a hit does** (asm:Sword_DamagePoints `$10`/`$20`/`$40`, asm:Obj_TypeHpPairs, the Darknut's shield in
asm:Damage_HandleWeaponHit; walked with one Blue Darknut of the Olive Garden alone, Link written beside it, one swing
at a time, `r32b_d5.py measure`, 20 seeds):

| Monster (Dungeon 5) | HP | Wood sword | White sword | Bomb (`$40`) | Walked |
|---|---|---|---|---|---|
| Blue Darknut (0x64, 0x05, 0x06, 0x37) | `$80` | **8** | 4 | **2** | wood from the side 8 (20/20), from behind 8 (20/20); white 4 and 4; from the front 0-2 swings land (it turns) and none kills; bombs laid on its way with Link facing its way: 2 blasts kill (18 of 20; 2 alive after one, the second bomb out of reach) |
| Gibdo (0x66's fight, the crossings) | `$70` | 7 | 4 | 2 | (asm only) |
| Pols Voice (0x27, 0x25; crossed) | `$A0` | 10 | 5 | – (bombs nothing; an arrow kills) | (asm only) |
| Digdogger's small one (0x24) | `$80` | **8** | 4 | **2** | below |

**The Olive Garden by the scripted fight that exists.** Round 25 left no scripted rule for the Garden (its gains were
the darknut model's own, `round25.md` §5); the scripted Blue Darknut fight is round 28's `r28_darknut.Fighter`
(Dungeon 8's 0x5e and 0x3e: stab a Darknut in the blade's reach that does not face the blade, the beam along a row or
column at full hearts, step aside from what is predicted to touch, else walk to the nearest flank). `r32b_d5.py
garden` plays it on the Garden from d2first's own `5_05e_<c>` states (the six live, natural spawns), the sword and
hearts and bombs written. With bombs it lays one instead of a swing whenever a Darknut is in the blade's reach and
Link holds any (`BombFighter`: the same side rule as the blade, the bomb's direction being Link's facing). 20 seeds a
line, to the last Darknut (the push and the stairs are not counted):

| Sword | Bombs in | Hearts in (of 7) | Cleared | Hearts lost (mean, max) | Frames (median) | Bombs used | HP by the sword / by blasts | Hearts on clearing (drops taken) |
|---|---|---|---|---|---|---|---|---|
| wood | 0 | 7 | **9/20** (11 dead) | 5.80, 8.0 | 4357 | 0 | 679 / 0 | 5.22 |
| wood | 0 | 5 | 9/20 | 4.45, 8.0 | 4195 | 0 | 659 / 0 | 3.44 |
| wood | 2 | 7 | 14/20 | 4.30, 8.0 | 4043 | 2.0 | 753 / 46 | 5.64 |
| wood | 2 | 5 | 13/20 | 3.45, 5.0 | 4019 | 2.0 | 734 / 19 | 4.69 |
| wood | 4 | 7 | 15/20 | 4.20, 10.0 | 4147 | 4.0 | 758 / 69 | 5.13 |
| wood | 4 | 5 | 11/20 | 3.55, 6.0 | 3629 | 4.0 | 686 / 60 | 5.27 |
| wood | 8 | 7 | **18/20** | 4.40, 7.0 | 3282 | 7.8 | 741 / 118 | 5.00 |
| wood | 8 | 5 | 17/20 | 2.95, 6.0 | 3296 | 8.0 | 731 / 106 | 5.00 |
| white | 0 | 7 | **17/20** | 3.85, 8.0 | 2421 | 0 | 886 / 0 | 5.71 |
| white | 8 | 7 | 17/20 | 3.15, 7.0 | 2335 | 7.7 | 720 / 124 | 5.53 |

(HP in sixteenths of a hit point, `$10` a wood hit; "hearts lost" counts every hit, "hearts on clearing" includes the
heart drops taken during the fight. What hits: Blue Darknut touches, 2 hearts each, and `$68` on a few frames.)

- **The wood sword doubles the fight and halves the clears without bombs: 9 of 20 against the white sword's 17**,
  4357 frames against 2421. Every Darknut takes eight landed hits from its side or back; the fight's length is what
  costs.
- **Bombs buy it back: eight give 18 of 20**, the white sword's rate, at 4.50 hearts and 3282 frames. Few blasts land
  with this rule (118 sixteenths in 20 runs of 8 bombs: about 1.5 of 8 bombs hit a Darknut, two blasts a kill), but
  every blast also knocks the Darknuts back and makes them invincible for 32 frames, which keeps them off Link. (The
  same lines at 8 containers, run first by mistake, gave the same clears: 9, 14, 15, 18 of 20; a death is a
  sequence of touches that more hearts only lengthens.) A
  better bomb rule (laid on a Darknut's path from its side, `measure`'s placement: 2 blasts a kill 18 of 20) would
  do more; none exists.
- **What the route brings to the Garden** (the owner: 5-8 bombs leaving Level 1, missing ones cheated): the heart
  rock's wall (0:0x2c) and the 30 secret (0:0x2d) take one each, Dungeon 5's 0x66 W and 0x65 W walls one each
  (0x65's four are refused westbound): **1-4 bombs at the Garden** with none dropped, 0 with 5 leaving Level 1 and a
  bomb lost elsewhere. That is the 2-4 bomb lines: 11-15 of 20. The cheat today grants a bomb only for a wall (a
  granted bomb opens the wall and nothing else); topping the bag up to 8 for the Garden is new (§8, For the owner).
- The scripted fighter is no model: these lines compare the swords and the bombs, not the rate a trained darknut
  family will reach (d2first enters Dungeon 5 with 7.74 of 10 hearts on the full route's chains, and the Garden is
  its dearest room there).

**Digdogger (5:0x24) with the wood sword** (`r32b_d5.py digdogger`: from d2first's `5_24e_<c>`, RECORDER facing W
after 6 frames in; the big one splits into one small one, `$80`; the fighter with the small one as its target; bombs
laid as in the Garden):

| Sword | Bombs | Hearts in (of 7) | Killed | Hearts lost (mean, max) | Frames from the play (median) | Swings | Bombs used |
|---|---|---|---|---|---|---|---|
| wood | 0 | 7 | 20/20 | **2.27**, 5.5 | **2281** | 23.9 | 0 |
| wood | 0 | 4 | 14/20 (6 dead) | 2.27, 4.0 | 2271 | 11.8 | 0 |
| wood | 4 | 7 | 20/20 | 2.23, 5.5 | 2558 | 24.7 | 3.8 |
| wood | 4 | 4 | 16/20 | 2.08, 4.0 | 1943 | 9.6 | 3.6 |
| white | 0 | 7 | 20/20 | 1.05, 3.5 | 979 | 13.7 | 0 |
| white | 0 | 4 | 17/20 | 1.50, 4.0 | 1448 | 6.3 | 0 |

Most of the hearts are the room's statue fireballs while the small one is chased (eight landed hits instead of four).
Bombs barely help here (the small one is fast and seldom stays in a blast).

**What Dungeon 5 costs more on this route**, from these walks: the Olive Garden (half the clears without bombs, +0.5 to +2 hearts and +900 to
+1900 frames against the white sword, bombs depending), Digdogger (+1.2 hearts, +1300 frames), and 3 containers fewer
(7 against 10) for the whole dungeon, whose other rooms are crossed and cost the same (the clip of 0x64 and the
crossings use no sword; 0x66's fight on the way back is Gibdos at 7 wood hits against 4).

## 3. From Level 3's landing to Level 2: the dead woods' 100, the world wrap, the raft heart, the Armos's 30

`r32b_leg.py stage` walks the whole leg once in one game from Level 5's exit (the warp of §1, then every room
below; monsters emptied): **5932 frames from the landing in 0x74 to Level 2's first room, scrolls, caves, the
candle's burn and the Armos's fade-in included**, 8 containers in and 10 out, rupees +130. `cross` plays each room
from its pre-entry state on 20 seeds with the monsters live and never fought (the Armos included):

| Room | In → out | Rule walked | 20 seeds | Hearts lost (mean, max) | Frames (median) | What hits |
|---|---|---|---|---|---|---|
| 0x74 | recorder → W | W along the landing row Y `$8D` | 20/20 | 0.00, 0 | 200 | – |
| 0x73 | E → N | W on `$8D` to X `$B0`, N between the trees | 20/20 | 0.10, 0.5 | 193 | Blue Moblin |
| 0x63 | S → W | N to Y `$BD`, W to X `$50`, N to `$7D`, out W | 20/20 | **0.55**, 1.0 | 341 | Red Moblins, arrows |
| 0x63 | S → W | (first tried) N to Y `$5D`, W to `$20`, S to `$7D` | 13/20 (7 stuck) | 2.30, 7.5 | 405 | Red Moblins, Octoroks |
| 0x62 | E → cave | to (`$A0`,`$5D`), the blue candle facing W, onto the stairs | 20/20 | 0.10, 1.0 | 202 | Moblin arrows |
| 0x62 cave | S → S | up at X `$78`: **+100 rupees** (20 → 120) | 20/20 | 0, 0 | 334 | – |
| 0x62 | cave → N | out of the stairs at (`$60`,`$7D`), N at X `$60` | 20/20 | 0.05, 0.5 | 133 | arrow |
| 0x52 | S → W | N to `$BD`, W to `$20`, N to `$8D`, out W | 20/20 | 0.05, 0.5 | 236 | Red Moblin |
| 0x51 | E → W | the wrap exit (`Game.act('CLIP E')` from the E entry) | 20/20 | 0, 0 | 140 | – |
| 0x50 | E (wrap) → S | from X `$F0` W to `$C0`, S | 20/20 | 0.05, 1.0 | 178 | Red Lynel |
| 0x60 | N → W | S to `$8D`, E to `$E0`, the wrap exit round the world | 20/20 | 0, 0 | 214 | – |
| 0x5f | E (wrap) → N | `CLIP W` (the ladder clip) onto the heart: **9 containers**; W by the ladder to land, N at X `$70` | 20/20 | **0.80**, 1.5 | 280 | Octoroks' rocks |
| 0x4f | S → N | N at X `$70` | 20/20 | 0.03, 0.5 | 205 | rock |
| 0x3f | S → N | onto the dock (X `$60`), the raft | 20/20 | 0.10, 0.5 | 232 | Octorok |
| 0x2f | S → cave | off the raft, up into the take-any cave | 20/20 | 0, 0 | 85 | – |
| 0x2f cave | S → S | the right ware: **10 containers** | 20/20 | 0, 0 | 372 | – |
| 0x2f | cave → S | the raft down | 20/20 | 0, 0 | 175 | – |
| 0x3f | N → W | off the dock, W on `$8D` | 19/20 (1 stuck) | 0.05, 0.5 | 254 | statue fireball |
| 0x3e | E → W | the wrap exit | 20/20 | 0, 0 | 140 | – |
| 0x3d | E (wrap) → cave | to (`$A0`,`$7D`), one touch W into the Armos, back off E 12 frames, the stairs, in | 20/20 | **0.20**, 1.0 | 205 | Moblin arrows |
| 0x3d cave | S → S | up at X `$78`: **+30 rupees** | 20/20 | 0, 0 | 195 | – |
| 0x3d | cave → S | out over the stairs at (`$90`,`$6D`): E to `$A0`, S | 20/20 | 0.33, 0.5 | 214 | Blue Moblins |
| 0x4d | N → W | S to `$5D`, W to `$10`, S to `$8D`, out W | 20/20 | 0.12, 1.0 | 282 | Blue Moblins |
| 0x4c | E → N | W on `$8D` to `$70`, N (d2first's step) | 20/20 | **0.62**, 2.5 | 299 | fast Octoroks |
| 0x3c | S → cave | N at X `$70` into Level 2 (d2first's step) | 20/20 | 0, 0 | 85 | – |

**The leg by these rules: 3.15 hearts and 5195 frames in the rooms (sums of means and medians).** Rooms on d2first
(0x73 E→N, 0x51 to 0x3e, 0x4c, 0x3c) cost what the coast stretch's scripted walks cost there (`d5-leg.md`): the sword
is not used.

**The dead woods' 100 (0:0x62).** The tree is the tile object `$64` at (`$80`,`$60`) (game.yaml `entrance: {cave:
'0x22', how: burn, tile: [16, 4], rupees: 100}`), the middle of the column of trees that walls the room's W half. It
stands on the top row Link can walk on its E side: the burn from the E on its row (Link at X `$A0`, Y `$5D`, facing
W: `GameView::burn_facings`' "W from X tree + `$18`-`$2F`") burned it 92 frames after B, 20 of 20; one fire (the blue
candle's one per room entry) is enough. Link walks W onto the stairs square (23 frames), and out of the cave he stands
at (`$60`,`$7D`), W of the burned square, from where N at X `$60` leaves the room. The 100 are taken walking up to
the old man at X `$78` (as 0x0f's): 20 → 120, room flag `$90`.

**Toward the graveyard, then the world wrap.** The owner: "Screen scroll toward the graveyard but make your way down
south to do the world wrap". Two ways from 0x62 to 0x5f's water were walked:

- **0x62 N, 0x52 W, 0x51's wrap exit into 0x50 (the graveyard's side), S into 0x60, its wrap exit round the world
  into 0x5f**: 904 frames from 0x62's cave exit to 0x5f (staged), 0.23 hearts (20 seeds). From 0x51 on this is
  exactly d2first's coast-heart clip way (steps 139-142): the same rooms entered from the same sides; only 0x52 S→W
  is new.
- **0x62 W into the Lost Woods (0x61), its sequence N, W, S, W into 0x60, its wrap exit**: 1308 frames, 0.25 hearts.
  0x61 cannot be left N or S out of order (asm:Maze_CheckStep: only E is free; a wrong step repeats the room and
  resets the count), so this is the only woods way to 0x60.

The first is 404 frames shorter and is the reading taken: the "screen scroll toward the graveyard" is 0x51's wrap exit
into 0x50, and "down south" is 0x50 S to 0x60, whose W edge wraps round the world into 0x5f (room id − 1).
**d2first's stretch works from here unchanged**: the wrap exits of 0x51 and 0x60 and the ladder clip on 0x5f's water
were played by the engine's own macros (`Game.act`), 20 of 20 each; the rooms need no new game.yaml entries; the
steps are d2first's (entered from the same sides), the new states are this route's (other inventory).

**The raft heart.** 0x4f N, 0x3f's dock at X `$60` and the raft N (161 frames) onto 0x2f, the take-any cave's right
ware: d2first's steps 156-161 exactly. Out of the cave the raft takes Link back S to 0x3f's dock; W along `$8D`.

**The Armos's 30 (0:0x3d).** The secret statue is the one at X `$90`, Y `$80` (asm:Armos_SecretRoomIds and
asm:Armos_SecretXs: room `$3D`, X `$90`; game.yaml `entrance: {cave: '0x21', how: armos, tile: [18, 8], rupees:
30}`), the middle statue of the room's centre row. A press into it from any side wakes it (walked from E, W, N and S:
the Armos appears on its square on the press's frame or 6 frames after, and the square shows stairs 63 frames after
the touch, `$3F`). Link arrives by the wrap exit at (`$F0`,`$8D`) inside the E wall's column, walks W to `$E0`, N to
`$5D`, W to `$A0`, S to `$7D` and presses W. Three plays of the wait (20 seeds each):

| Play | Hearts (mean, max) | Frames | Armos touches |
|---|---|---|---|
| **One touch, back off E 12 frames, wait, W onto the stairs** | **0.20, 1.0** | 205 | 0 |
| Hold on beside it (no back-off) | 0.65, 1.5 | 225 | 11 hearts' worth in 20 runs |
| Back off 16 frames, kill it with the sword or beam when it comes along the row | 0.42, 2.0 | 285 | 2 |

The Armos touches for a heart while it fades in (`zelda-asm/enemies.md` "Armos") and then walks at 0.5 or 1.5 px a
frame toward Link within `$51` px; stepping back a square is enough. The engine's vocabulary for it exists:
`Secret::Armos` (game.yaml `how: armos`, the room's secret Armos as the target, the stairs when it has faded in), as
0x34's (d2first's food shop). (`Secret::Push` is the gravestone's.) The stairs are not saved: they appear each time
that Armos is woken. Out of the cave Link stands above the stairs square at (`$90`,`$6D`); walking S there takes him
back down the stairs (the staging walk's first try did), so the way out is E a square to `$A0`, then S.

**Then Level 2.** 0x3d has no W edge (game.yaml exits N and S only; Level 2's room 0x3c has only S). "Scroll over to
Level 2" is 0x3d S, 0x4d W (new: d2first crosses 0x4d from S and W), 0x4c E→N and 0x3c S→cave, d2first's steps 12-13.

## 4. Dungeon 2 with the wood sword at ten hearts

d2first plays Dungeon 2 first, with the wood sword and 3 containers; `d3first` plays the same rooms (2:0x7d to the
triforce in 2:0x0d, d2first steps 14-23) with the same sword and **10 containers** (the leg of §3 ends at 10, its
scripted crossings losing about 3 of them: Link enters with roughly 7-10 hearts). Nothing in the dungeon is harder:
the monsters, their HP and the sword's damage are the same (Ropes `$10`: 1 wood hit; Red Goriyas `$30`: 3; Moldorm's
segments; Dodongo dies from one bomb and a sword hit of any grade, or two swallowed bombs: asm:Dodongo_CheckCollisions),
and every touch costs the same hearts out of three times as many. **What differs is what the families are shown and
what the bag holds** (`crates/env/src/observation.rs` `link` features):

| Input | d2first in Dungeon 2 | `d3first` in Dungeon 2 |
|---|---|---|
| `max_hp` (containers / 16), `hp`, `hp_fraction`, `hp_low`, `hp_full` | 3 containers | 10 (7-10 hearts) |
| `triforce` (pieces) | none | 1, 3, 4, 5 |
| `bow`, `boomerang`, `candle_blue`, `whistle`, `raft`, `ladder` | none of them | all |
| `bombs`, `bombs_spare` | what drops (0 at the entrance) | the bag from Dungeon 5: 0x65's four taken eastbound plus what the Olive Garden left |
| `keys`, `rupees` | 0 | the first half's keys; about 130-150 rupees (0x62's 100 and 0x3d's 30) |
| `sword` | wood | wood (the same) |

Which rooms' families see inputs they were never trained on:

- **d2-rope** (2:0x7d, 0x7e, 0x6e, 0x4e, 0x2e, 0x0d), **moldorm** (2:0x3e) and **boss-dodongo** (2:0x0e) play only
  Dungeon 2: their every start had 3 containers, no piece and none of those items. Ten containers, four pieces and six
  items are outside what they have seen (their `ram_randomization` draws hearts within the state's containers and
  bombs 0-8, nothing else).
- **goriya** (2:0x5e, 0x1e) also plays Dungeons 1, 7 and 9 with up to 15 containers and every item: its inputs here are
  within what it knows.
- **Bombs as a weapon:** goriya and moldorm are `bombs: free` and d2first's Dungeon 2 steps have no `bomb_reserve`
  (Link holds none there). With the bag `d3first` brings, goriya would throw them at the Red Goriyas of 0x5e and 0x1e
  and leave none for Dodongo, where a granted bomb is cheated. A `bomb_reserve: 1` on `d3first`'s Dungeon 2 steps
  before 0x0e (0 at Dodongo, as on d2first) keeps one for him (§8).

Not walked: Dungeon 2 has no room this route plays differently (no new step, no new fight), and a model's play of it
with this inventory needs `d3first`'s own route data (a step is matched by exactly what Link holds: d2first's steps
14-23 hold no piece, so a run holding four is off d2first's route in every room). The first chain of `d3first`'s
Dungeon 2 segment is that measurement.

## 5. After Level 2: the warp to Level 4, the arrows, the meat, Level 7; Dungeon 7 with the wood sword

**The leg.** Out of Level 2 at 0x3c, one play facing right (§1) sets Link down on 0x45's pier at (`$80`,`$8D`), 20
of 20, 479 frames: the very spot of d2first's step 210 (its whirlwind comes from 0x0b with two plays down). From
there the leg is d2first's steps 210-227 unchanged (the raft to 0x55, the dock stretch, 0x44's arrows for 80, 0x34's
Armos and its meat for 60, row 5 W to 0x52, N into 0x42, the pond by the recorder): same rooms, sides and exits.
What differs is what Link holds: the wood sword (the overworld families' `sword` input; the crossings are walked
without the sword on both routes), 11 containers (the same as d2first's: Digdogger's on d2first, Dodongo's here),
pieces 1-5 (the same) and **fewer rupees: about 130-150 into 0x44 against d2first's 168-180** (§7), so 0-10 after
the meat where d2first keeps 28-40. `r32b_late`-style confirmations of the shops with this route's purse: §6's table.

**Dungeon 7 with the wood sword** (d2first steps 228-247 unchanged; 11 containers in on both routes, d2first with
the white sword). Hits to kill (asm:Obj_TypeHpPairs, asm:Sword_DamagePoints; `zelda-asm/enemies.md`):

| Monster | HP | Wood | White | Bomb | Where on the route |
|---|---|---|---|---|---|
| Blue Goriya | `$50` | **5** | 3 | 1 | 0x69, 0x59, 0x49, 0x38, 0x18, 0x19, 0x1a, 0x1b (6), 0x29 |
| Red Goriya | `$30` | 3 | 2 | 1 | 0x59, 0x38, 0x1a, 0x29 |
| Digdogger's small ones | `$80` | **8** | 4 | 2 | 0x1c (three; 0x39's big one is crossed) |
| Wallmaster | `$20` | 2 | 1 | 1 | 0x0d (five) |
| Aquamentus | `$60` | **6** | 3 | 2 (a wooden arrow 3) | 0x2a |
| Dodongo (0x0c, crossed) | – | one bomb and a sword hit of any grade | | | 0x0c |

The dear rooms by the scripted fighter (`r32b_d7.py`: round 28's `Fighter` with each room's monsters as targets, no
bombs; the probe with the route's ledger of doors, keys and bombs, `r26_walks.d7_state`, the sword and hearts written;
20 seeds a line):

| Room | Sword | From 11 hearts: won, hearts lost (mean), frames | From 7 hearts: won, hearts lost, frames |
|---|---|---|---|
| 0x49 (3 Blue Goriyas, Keese, Bubbles; its N shutter) | wood | 13/20 (6 out of time, 1 dead), 4.08, 2169 | 3/20 (12 dead), 5.95, 4192 |
| | white | 16/20, 1.85, 1269 | 6/20 (5 dead), 4.15, 3787 |
| 0x1b (6 Blue Goriyas; the 4 bombs) | wood | 20/20, **1.70**, 2370 | 20/20, 2.10, 2627 |
| | white | 20/20, 0.55, 1326 | 20/20, 1.35, 1729 |
| 0x1c (the recorder; one small Digdogger, as d2first plays it) | wood | **13/20 (7 dead), 8.03**, 2921 | 6/20 (14 dead), 6.28, 3705 |
| | white | 20/20, 2.48, 1007 | 20/20, 2.80, 1491 |
| 0x1c (all three small ones) | wood | 9/20 (11 dead), 9.75, 5457 | 2/20, 6.83, 6230 |
| | white | 20/20, 4.67, 2672 | 16/20, 5.08, 3118 |
| 0x2a Aquamentus | wood | 18/20, **3.98**, 2737 | 16/20 (4 dead), 3.95, 2808 |
| | white | 19/20, 2.55, 1899 | 17/20 (3 dead), 3.23, 2631 |

What hits: in 0x49 and 0x1b the Goriyas' boomerangs (`$5C`) and touches (`$05`); in 0x1c the small ones (`$18`) and
the statue fireballs; in 0x2a the fireballs.

- **7:0x1c is the room the wood sword breaks.** Eight landed hits on a small Digdogger (four with the white) under the
  statues' fireballs: the fighter that kills one with the white sword in 20 of 20 for 2.5 hearts dies in 7 of 20 with
  the wood one from full hearts, and in 14 of 20 from seven. It was already d2first's second killer room of the
  dungeon (79 of 872 Dungeon 7 runs, round 26).
- **0x1b**, d2first's first killer (324 of 872), costs 1.2-0.8 hearts more but is cleared every time by the script.
- **0x49** and **Aquamentus**: about 2 and 1.4 hearts more from full hearts.
- The wood sword beam (`$10`) flies only at full hearts, as the white one's; Link enters with 11 of 11 only if the
  leg cost nothing.
- Bombs as a weapon (goriya and boss-digdogger are `free` above the reserve, 3 from the entrance and 2 from 0x1b on
  d2first) were not played by this script; the small Digdoggers take two blasts each (asm), which Dungeon 5's walk
  found hard to land on them (§2).

Taken together the dungeon costs roughly 5-9 hearts more by these scripts (0x1c 5.6 alone) than with the white sword,
from the same 11 containers. **This is the place the route's choice of the wood sword shows most** (For the owner).

## 6. Level 7 to the grave, Level 6, Level 8, Level 9: d2first's legs with this route's inventory

From Level 7's exit the inventory of both routes is the same but for the sword before the grave (wood here, white on
d2first) and the purse, keys and bombs. `r32b_late.py` plays rounds 27-29's leg walks again with `d3first`'s
inventory written at the staging (`r32b_walks.D3`: 12 containers, the wood sword, the red candle, arrows; out of Level
6 13 containers and the magic sword; out of Level 8 14), monsters live, 20 seeds:

- **Level 7 to the grave and Level 6** (`grave`, round 27's crossings with the wood sword before the grave): every
  crossing 20/20 (0x42 0.65 hearts, 0x51 0.45, the woods' four scrolls 0.72 together, 0x31 N→E 0.55, 0x32 0.23, the
  rest 0-0.15); the clip way 0x51's wrap exit and 0x50 20/20 at 0 hearts. **The magic sword at 12 containers: taken
  in 20 of 20 pushing the stone from below (0.05 hearts, 893 frames), 18 of 20 from above (2 failed walks to the
  stand square); at 11 containers refused** (InvSword unchanged, the stone back). The container count holds: 8 out of
  Level 5 (the first half's ledger), 9 at 0x5f and 10 at 0x2f (both counted by the game in §3's walk), Dungeon 2's
  Dodongo 11, Dungeon 7's Aquamentus 12.
- **Level 6's exit, the warp, Level 8** (`d8leg`): two plays facing down land in 0x3c at (`$80`,`$AD`) (§1); 0x3c 0
  hearts, 0x4c N→E 1.25, 0x4d 0.23, 0x5d 0.23, 0x6d's tree burned by the red candle facing W, one use, 20/20 at 0
  hearts; into level 8 20/20. Same rooms and costs as d2first's (`d8-walks.md`).
- **Level 8's exit, the warp, 0x47, Level 9** (`d9leg`): the warp walked in §1 (one play down, 0x37, 20/20). Round
  29's leg walk could not be replayed here: its staging walks on through level 9 and stopped at 9:0x63's W key door
  with the 2 keys written (it needs 3: 0x14 E, 0x16 N, 0x63 W). The leg itself (0x37 to 0x05's wall, 0x47's tree and
  heart, 15 containers) holds exactly d2first's inventory on this route (the same pieces, sword, candle, containers
  and index `$F9` AND 7 = 1); its walk in `d9-walks.md` §1 stands for it. Not replayed: `r32b_late.py d9leg` needs
  `keys: 3` in `D3[8]` (the route's level 9 key ledger after phase IX is the builder's).

What differs on these legs: before the grave the wood sword (no room of round 27's leg is fought); the purse (§7);
keys and bombs by phase IX's changes, which are d2first's too.

## 7. This half as route steps

For the agent that writes `routes.yaml`. "d2first N" = the same room, entry, exit and family as d2first's step N at
v1.9.0.0 (`_native.route_steps('d2first')`; phase IX's changes to those dungeons apply as on d2first); **new** = a
step d2first does not have. Every step needs `d3first`'s own start states (other sword, containers and pieces: §8).

| # | Room | In → out | Taken / done | Family | New or existing |
|---|---|---|---|---|---|
| D5 | 5:0x76 … 5:0x14 | as d2first | 0x64's block clip, the Olive Garden, the recorder (0x04), Digdogger's heart (**8**), the piece | gibdo, darknut, equipment-room, dodongo-crossing, zol, pols-voice, boss-digdogger | d2first 186-208 |
| 1 | 0:0x0b | cave → recorder | **1 play facing right** (MOVE E, RECORDER) → 0x74 | ow-leever | d2first 209 (other facing, computed) |
| 2 | 0:0x74 | recorder → W | | ow-tektite | **new** (`0_74r`) |
| 3 | 0:0x73 | E → N | | ow-moblin | d2first 106 |
| 4 | 0:0x63 | S → W | (do not burn 0x63's own tree: 20 rupees' door repair) | ow-moblin | **new** (`0_63s`) |
| 5 | 0:0x62 | E → cave | the tree burned (blue candle, facing W from (`$A0`,`$5D`)), the stairs | ow-moblin | **new** room (`0_62e`) |
| 6 | 0:0x62 cave | S → S | `waypoint: rupees`: 100 | ow-moblin | **new** (`0_62cave`) |
| 7 | 0:0x62 | cave → N | | ow-moblin | **new** (`0_62c`) |
| 8 | 0:0x52 | S → W | | ow-moblin | **new** (`0_52s`) |
| 9 | 0:0x51 | E → W | the wrap exit | ow-moblin | d2first 139 (clips way) |
| 10 | 0:0x50 | E → S | | ow-lynel | d2first 140 |
| 11 | 0:0x60 | N → W | the wrap exit round the world | ow-lynel | d2first 141 |
| 12 | 0:0x5f | E → N | the ladder clip, the heart container (**9**) | ow-zora | d2first 142 |
| 13 | 0:0x4f | S → N | | ow-zora | d2first 156 |
| 14 | 0:0x3f | S → N | the raft | ow-zora | d2first 157 |
| 15 | 0:0x2f | S → cave | | ow-start | d2first 158 |
| 16 | 0:0x2f cave | S → S | `waypoint: heart` (**10**) | ow-start | d2first 159 |
| 17 | 0:0x2f | cave → S | the raft | ow-start | d2first 160 |
| 18 | 0:0x3f | N → W | | ow-zora | d2first 161 |
| 19 | 0:0x3e | E → W | the wrap exit | ow-zora | d2first 162 (clips way; no plain way on this route) |
| 20 | 0:0x3d | E → cave | the Armos at X `$90` touched from E, back off, the stairs | ow-moblin | **new** (`0_3de`; d2first 163 leaves N) |
| 21 | 0:0x3d cave | S → S | `waypoint: rupees`: 30 | ow-moblin | **new** (`0_3dcave`) |
| 22 | 0:0x3d | cave → S | out E of the stairs, S | ow-moblin | **new** (`0_3dc`) |
| 23 | 0:0x4d | N → W | | ow-moblin | **new** (`0_4dn`) |
| 24 | 0:0x4c | E → N | | ow-octorok | d2first 12 |
| 25 | 0:0x3c | S → cave | | ow-octorok | d2first 13 |
| D2 | 2:0x7d … 2:0x0d | as d2first | Dodongo's heart (**11**), the piece; `bomb_reserve: 1` until 0x0e | d2-rope, goriya, moldorm, boss-dodongo | d2first 14-23 |
| 26 | 0:0x3c | cave → recorder | **1 play facing right** → 0x45 | ow-octorok (`recorder: true` needed) | **new** (d2first 24 leaves S) |
| 27-44 | 0:0x45 … 0:0x42 | as d2first | the raft, 0x44's arrows (80), 0x34's Armos and meat (60), 0x42's pond | ow-peahat, ow-zora, ow-octorok, ow-leever, ow-moblin | d2first 210-227 |
| D7 | 7:0x79 … 7:0x2b | as d2first | the red candle, the meat given, Aquamentus's heart (**12**), the piece | goriya, boss-digdogger, equipment-room, dodongo-crossing, wallmaster, boss-aquamentus | d2first 228-247 |
| 45-61 | 0:0x42 … 0:0x22 | as d2first | the grave's push, **the magic sword** (12 containers) | ow-moblin, ow-lynel, ow-ghini | d2first 248-264 |
| D6 | 6:0x79 … 6:0x0c | as d2first after phase IX | Gohma's heart (**13**), the piece | (d2first's) | d2first 265-280 |
| 62-67 | 0:0x22 … 0:0x6d | as d2first | 2 plays facing down → 0x3c; 0x6d's tree (the red candle) | ow-lynel, ow-octorok, ow-moblin | d2first 281-286 |
| D8 | 8:0x7e … 8:0x2c | as d2first after phase IX | Gleeok's heart (**14**), the piece | (d2first's) | d2first 287-296 |
| 68-82 | 0:0x6d … 0:0x05 | as d2first | 1 play facing down → 0x37; 0x47's tree and heart (**15**); 0x05's wall | ow-moblin, ow-octorok, ow-zora, ow-leever, ow-boulder, ow-lynel | d2first 297-311 |
| D9 | 9:0x76 … 9:0x32 | as d2first after phase IX | the red ring, the silver arrow, Ganon, Zelda | (d2first's) | d2first 312-357 |

Containers in bold are the count after the step. **11 new steps** (0x74, 0x63, 0x62 ×3, 0x52, 0x3d ×3, 0x4d, 0x3c
out by the recorder); every other step is d2first's room, entry and exit with this route's inventory.

### The rupee ledger of the whole route

Drops by stretch are read from d2first's committed start states (`r32b_late.py ledger`: the rupees its chains carry
at each segment boundary, chains 0-2): Dungeon 2 +8 to +16 (1 → 8-17), Dungeon 5 +0 to +1, Dungeon 7 +0 to +4,
Dungeon 6 −1 to −6 (Gohma's arrows), Dungeon 8 −3 to +6 (the Pols Voices' arrows, 0x4e's 5), the legs 0.

| Where | In | Out | `d3first` purse (estimate) | Note |
|---|---|---|---|---|
| Leaving Level 1 | | | 30 | the owner's figure (the first half's ledger) |
| 0:0x2d | +30 | | 60 | the 30 secret (bomb) |
| 0:0x0c | | −60 | 0 | the blue candle |
| Dungeon 5 | +0-1 | | 0-1 | drops |
| 0:0x62 | +100 | | 100-101 | walked: 20 → 120 |
| 0:0x3d | +30 | | 130-131 | walked: +30 |
| Dungeon 2 | +8-16 | | 138-147 | d2first's drops there |
| 0:0x44 | | −80 | 58-67 | the arrows |
| 0:0x34 | | −60 | **0-7** | the meat (d2first keeps 28-40 here) |
| Dungeon 7 | +0-4 | | 0-11 | |
| Dungeon 6, Gohma | | −1-3 | 0-8 | an arrow a shot; topped up and cheated there at 0 |
| Dungeon 8, 8:0x4c | | −1 (phase IX: the wall only) | 0-7 | no rupee cheat in this room |
| Dungeon 9, Ganon | | −1 | 0-6 | the silver arrow; none flies at 0, no cheat |

Income 190 + drops (about 10-20) against 200 spent on wares and 3-5 on arrows: **the route ends about even, and a run
with few drops reaches 0x34 short** (cheated there, `requires_rupees`), Gohma short (cheated there), and 8:0x4c and
Ganon with nothing for their arrows. Where rupees would be cheated: 0x34's meat first (the second purchase), Gohma's
arrows; Ganon's arrow is unguarded (For the owner, 7).

## 8. What the route data and the engine must gain (for the agent that builds `d3first`)

Nothing here was changed on main or in the worktree's engine or data; every walk ran on main's code at 1.9.0.0.

**game.yaml** (rooms are shared by both routes; d2first never enters 0x62, and its 0x3d step leaves N, so neither
change moves d2first):

```yaml
- level: 0
  location: '0x62'
  specialist: ow-moblin                    # new: 5 Blue Moblins, edge spawns (as 0x61, 0x63)
  exits:
    N: {level: 0, location: '0x52'}
    E: {level: 0, location: '0x63'}
    S: {level: 0, location: '0x72'}
    W: {level: 0, location: '0x61'}
    cave: {level: 0, location: '0x62', in_cave: true}     # new: the burned tree's stairs
  enemies: {BlueMoblin: 5}
  treasure: null
  edge_spawn: true
  entrance: {cave: '0x22', contents: secret-rupees, how: burn, tile: [16, 4], rupees: 100}
- level: 0
  location: '0x62'
  in_cave: true
  specialist: ow-moblin
  exits: {S: {level: 0, location: '0x62'}}
  enemies: null
  treasure: 100-rupees
# 0x3d: add `cave: {level: 0, location: '0x3d', in_cave: true}` to its exits (its entrance `how: armos, tile: [18, 8],
# rupees: 30` is there), and the cave:
- level: 0
  location: '0x3d'
  in_cave: true
  specialist: ow-moblin
  exits: {S: {level: 0, location: '0x3d'}}
  enemies: null
  treasure: 30-rupees
```

game.yaml's `how: burn` is read as a secret only in a room with a `cave` exit (`game_map::secret`), and the Armos's
MOVE into the statue is offered only in such a room (`GameView::is_secret_armos_direction`). With these entries
0x62's CANDLE is offered facing W from X `$98`-`$AF` on Y `$55`-`$65` (`burn_facings`; walked at (`$A0`,`$5D`)), and
0x3d's cave tiles before the stairs show are the squares beside the statue, (`$80`,`$7D`) and (`$A0`,`$7D`)
(`objectives::cave_tiles`), the second being where the walk stood, pressing W (the engine offers W and E only; from
below the waking Armos touches Link). One side effect for d2first: its 0x3d step (163, E → N on the lake's clip way)
crosses the room past the statue, and with the cave exit the mask will offer MOVE W or E into it there too (a woken
Armos costs a heart a touch); today the statue is a wall there. If that matters, the offer can be kept to a step
leaving by the cave.

**routes.yaml**: the route `d3first` (beside `d2first`); this half's new steps as they should read (the rest are
d2first's lines, §7):

```yaml
- {level: 0, room: '0x0b', specialist: ow-leever, entry: cave, exit: recorder}       # 1 play facing up/right: 0x74
- {level: 0, room: '0x74', specialist: ow-tektite, entry: recorder, exit: W}
- {level: 0, room: '0x73', specialist: ow-moblin, entry: E, exit: N}
- {level: 0, room: '0x63', specialist: ow-moblin, entry: S, exit: W}
- {level: 0, room: '0x62', specialist: ow-moblin, entry: E, exit: cave}
- {level: 0, room: '0x62', in_cave: true, specialist: ow-moblin, entry: S, exit: S, waypoint: rupees}
- {level: 0, room: '0x62', specialist: ow-moblin, entry: cave, exit: N}
- {level: 0, room: '0x52', specialist: ow-moblin, entry: S, exit: W}
- {level: 0, room: '0x51', specialist: ow-moblin, entry: E, exit: W}                 # the wrap exit
- {level: 0, room: '0x50', specialist: ow-lynel, entry: E, exit: S}
- {level: 0, room: '0x60', specialist: ow-lynel, entry: N, exit: W}                  # the wrap exit round the world
- {level: 0, room: '0x5f', specialist: ow-zora, entry: E, exit: N}                   # the ladder clip, the heart
- {level: 0, room: '0x4f', specialist: ow-zora, entry: S, exit: N}
- {level: 0, room: '0x3f', specialist: ow-zora, entry: S, exit: N}
- {level: 0, room: '0x2f', specialist: ow-start, entry: S, exit: cave}
- {level: 0, room: '0x2f', in_cave: true, specialist: ow-start, entry: S, exit: S, waypoint: heart}
- {level: 0, room: '0x2f', specialist: ow-start, entry: cave, exit: S}
- {level: 0, room: '0x3f', specialist: ow-zora, entry: N, exit: W}
- {level: 0, room: '0x3e', specialist: ow-zora, entry: E, exit: W}                   # the wrap exit
- {level: 0, room: '0x3d', specialist: ow-moblin, entry: E, exit: cave}
- {level: 0, room: '0x3d', in_cave: true, specialist: ow-moblin, entry: S, exit: S, waypoint: rupees}
- {level: 0, room: '0x3d', specialist: ow-moblin, entry: cave, exit: S}
- {level: 0, room: '0x4d', specialist: ow-moblin, entry: N, exit: W}
- {level: 0, room: '0x4c', specialist: ow-octorok, entry: E, exit: N}
- {level: 0, room: '0x3c', specialist: ow-octorok, entry: S, exit: cave, bomb_reserve: 1}   # Dodongo's bomb
# ... Dungeon 2 as d2first's steps 14-23, `bomb_reserve: 1` on 2:0x7d (0 on 2:0x0e as today) ...
- {level: 0, room: '0x3c', specialist: ow-octorok, entry: cave, exit: recorder}      # 1 play facing up/right: 0x45
- {level: 0, room: '0x45', specialist: ow-peahat, entry: recorder, exit: S}
# ... then d2first's steps 211-357 (with phase IX's changes) ...
```

No `stretch:` on this leg: clips only (the owner, Nov 18), and the woods way of §3 is longer. A run with the
run-time switch off has no way from 0x51 or 0x60 (as d2first's clip-only rooms of Dungeon 9).

**Specialists**: ow-octorok `items: {bombs: required, recorder: true}` (the recorder where the route plays it: 0x3c's
whirlwind out of Level 2; today it has none). ow-moblin has the candle (0x6d) and plays its first Armos (0x3d, as
ow-leever's 0x34); ow-tektite's 0x74 entered by the whirlwind; ow-leever's 0x0b now plays facing E.

**Start states**: a namespace for `d3first` (its states share d2first's names: `0_73e`, `0_51e`, `0_5fe`, ... hold
other swords, containers and pieces, and a state finds its step by name and progress). The staged pre-entry states
of this leg (`r32b_leg.py stage`, 26 of them from `0_74r` to `2_7ds`) are what the route's progress needs; the
monsters are not in them (pre-entry), the inventory is written at Level 5's piece.

**No engine change found necessary**: the whirlwind's plays and facings, the wrap exits of 0x51, 0x60 and 0x3e, the
ladder clip of 0x5f, the burned tree, the Armos secret, the secret-rupee caves, the raft and the take-any cave are all
vocabulary that exists; the walks played them by the engine's own macros where it has them.

## For the owner

What was said, what the game does, what was walked instead, and the cost. The lead decides from here.

1. **"After leaving level 5, use the Recorder to warp to Level 3."** Link comes out of a dungeon facing down, and
   down from the index the route has there lands in 0x0b itself (one play), 0x45 (two) or 0x74 (three: 768 frames).
   Facing right one play lands in 0x74 (479 frames). Walked: one MOVE E, one play. The same out of Level 2 for Level
   4 (down would take four plays). The two later warps are d2first's (down, two plays and one).
2. **"Get the 100 SECRET WNW of Level 3 ... Screen scroll toward the graveyard but make your way down south to do the
   world wrap."** Read and walked as: 0x74 W, 0x73 N, 0x63 W, 0x62's tree (the blue candle, from E on its row), the
   cave's 100; then 0x62 N into 0x52, W into 0x51, 0x51's wrap exit (the screen scroll) into 0x50 (the graveyard's
   side), S into 0x60, whose W edge wraps round the world onto 0x5f's water. The Lost Woods (0x62 W) lead to 0x60 only
   by their sequence N, W, S, W (they cannot be left N or S out of turn) and cost 404 frames more; walked beside,
   not taken.
3. **"Get the 30 SECRET at the armos east of Level 2. Scroll over to Level 2."** The Armos is the one at X `$90`
   (the centre statue). 0x3d has no W edge to 0x3c: the way is 0x3d S, 0x4d W, 0x4c N into 0x3c, three rooms
   (about 800 frames, 0.75 hearts by script).
4. **Hearts into Dungeon 5.** The assignment reads "8 hearts on entry"; the owner's list gives 3 + Dungeons 3, 4, 1 +
   the heart rock = **7 on entry**, 8 after Digdogger. Walked with 7. The container count at the grave is then
   exactly 12 (3, +3, the heart rock, Dungeon 5, 0x5f, 0x2f, Dungeon 2, Dungeon 7): the old man gave the magic sword
   at 12 and refused it at 11 (§6).
5. **The wood sword in Dungeon 5's Olive Garden** (six Blue Darknuts, 8 wood hits each, the push needs them all
   dead). By round 28's scripted fighter: without bombs it clears the Garden in 9-10 of 20 (the white sword 17 of
   20); with eight bombs 17-18 of 20. The route brings 1-4 bombs to the Garden (5-8 leaving Level 1, less the heart
   rock's wall, the 30 secret's wall and Dungeon 5's two walls). The cheat grants a bomb only for a wall today; a
   top-up of the bag for the Garden (or a bomb pickup the route takes on the way) is the owner's call. Without it the
   Garden is where this route's runs will end (d2first's Garden is already its dearest room with the white sword).
6. **The wood sword in Dungeon 7.** 0x1c's small Digdoggers take 8 hits each: the scripted fighter that kills one
   with the white sword 20 of 20 dies in 7 of 20 with the wood one from full hearts. 0x49, 0x1b and Aquamentus cost
   1-2 hearts more each. Nothing to walk instead (the route keeps the wood sword until the grave, as said); bombs on
   the small Digdoggers (two blasts each) and the recorder are the levers. Cost: Dungeon 7 is d2first's 46.6%
   segment with the white sword.
7. **Rupees** (§7): 30 leaving Level 1, +30, −60 (candle), +100, +30, −80 (arrows), −60 (meat) leaves about 0 with
   the drops of the dungeons (0-7 after the meat by d2first's drop rates), against the 3-5 rupees the arrows of Gohma, the Pols Voices and Ganon need (one rupee an
   arrow, none flies at 0). What exists (`zelda::cheats`): the shops' `requires_rupees` (0x44's 80, 0x34's 60) is
   topped up and counted when Link has fewer, and Gohma's room (6:0x1c) tops up a rupee an arrow while Gohma lives;
   8:0x4c's Pols Voices and Ganon have no rupee cheat. So on this route the shortfall is cheated at 0x34's meat (the
   second purchase) and at Gohma, and nothing is left for 8:0x4c's arrow and Ganon's silver arrow unless drops pay
   them. Cost: up to about 5 rupees cheated a run; Ganon's arrow needs one rupee the route does not guarantee (a
   `rupee_reserve` cannot keep what was never there: a top-up or a rupee room on the way is the owner's call).
8. **Dungeon 2's bombs.** d2first plays Dungeon 2 with an empty bag; this route comes with Dungeon 5's leftovers and
   0x65's four, and the goriya family throws bombs freely there (no reserve). A reserve of one keeps Dodongo's bomb.
   Not the owner's words; a route-data detail (§8).

