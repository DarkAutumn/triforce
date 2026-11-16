# Round 29 walks, the route part: from Dungeon 8's piece to Ganon's door in the running game (Nov 17)

`rest-of-game.md` "Gap list" (round 29, items 3, 6, 7 and the arrival spots of 5) and `d9.md` §11, answered by playing
the running game before anything of round 29 is built. This document is the **route**: the leg from Dungeon 8 to
Dungeon 9, the level's graph, doors, tunnels, walls, keys and every room that is only crossed. The big fights (the
Lanmolas of 0x55, the Wizzrobe rooms that must be emptied, the Patras, Ganon, Zelda's room) are another walk's;
where the route passes such a room it was staged empty here and that is said. Every number was observed
(**walked**) unless a line says otherwise; where chance is involved it is from at least 20 trials, each loaded with
the game's own randomness for its seed. Scripts (`scripts/experiments/`): `r29_route_walks.py` (staging, helpers),
`r29_route_leg.py` (item 1), `r29_route_d9.py` (items 2-6), `r29_route_scaffold.py` (a throwaway data root with a
scouting route `d9walk` so today's models can be played on the stretch).

Link on these walks is what a run with no drops brings out of Dungeon 8 (`d8.md` §7.3): 14 heart containers (full:
the D8 piece refills), the magic sword, the red candle, the bow and bought arrows, recorder, ladder, raft, boomerang,
3 keys, 0 bombs, 28 rupees, all eight pieces, no ring, the whirlwind index `$F9`.

## The answers

| # | Item | Answer (one line) | Corrects |
|---|---|---|---|
| 1 | The whirlwind | One play facing down (as Link comes out) or left: index `$F9` → `$F8`, 0x37 at (`$80`,`$8D`), 477 frames, **30 of 30 with the four Moblins live, 0 hearts**. Up or right: 0x74. From any other index one play facing down lands elsewhere | confirms `d9.md` §1.1 (inferred → walked) |
| 1 | 0x47 | The red candle facing S from X `$A8`-`$B8` on **either** row (Y `$8D` or `$9D`) burns the tree 94 frames after B; 12 frames of Down onto the stairs; the right ware at X `$98`: **15 containers, 15 hearts**; the left ware takes the potion and the container is gone for good | `d9.md` §1.2 (the stand row; "+1 heart" is a full refill only because Link was full) |
| 1 | 0x17, 0x05 | The ladder crosses the river in 61 frames at any row; 0x05's wall opens to **one bomb** facing N from X `$48`-`$58` (also facing E from `$38`, W from `$68`), 49 frames after B; 10 frames of Up onto the mouth | `d9.md` §1.2 ("about 70 frames") |
| 1 | The leg's cost | A scripted straight walk loses **8.6 hearts** (0x06 2.2, 0x05 2.9, the rest 3.4); the best play found per room 7.2. Models: ow-octorok, ow-zora and ow-lynel's 0x07 cross at 90-100% and under 0.7 hearts; **ow-moblin never plays the recorder (0 of 40), ow-leever 0x48 0 and 12 of 40, ow-boulder's ladder 0 of 40, ow-lynel 0x06 12 of 40 and 0x05 0 of 40** | `rest-of-game.md` survival table ("none given") |
| 2 | 0x66's man | He checks `InvTriforce` = `$FF` and nothing else: with any piece missing he stands at (`$78`,`$80`), both shutters stay shut and Link stops at Y `$8D`; with eight the shutters open 18 frames after Link's first frame. The fires cost ½ heart only above Y `$8D` | confirms level-9.md |
| 3 | The graph | The draft's way holds in one game from 0x6d to Ganon's door: 41 rooms and cellars after 0x76 with the ring, 31 without. **First movable spots are `$10` px inside an open or key door and `$20` px inside a bombed wall or shutter** | `d9.md` §2, §3.1 (every arrival spot was 8 px off) |
| 3 | Keys, bombs | Keys 3 → 2 (0x14 E) → 1 (0x16 N) → 0 (0x63 W) → 1 (0x61). Each of 0x63's three key doors takes the key on the first frame of the press. Bombs on a no-drop run: **5 written** (the rock wall, 0x65 N, 0x26 E, 0x27 N, 0x17 N), the Patra's four pay 0x06 W, 0x20 N, 0x31 W, 0x04 W: 0 left. Without the ring: 2 written. With 8 at 0x6d: none written, 3 left | confirms `d9.md` §3.5-§3.6 |
| 3 | Shutters, tunnels | 0x41 S and 0x31 S never open (full or emptied). Every tunnel: left ladder → the left-exit room, right ladder → the right-exit room, also for the ladder Link arrived on (**the wrong ladder is the one he came down**: the room behind him, its block back). The treasure cellars have one way out (the left ladder) | level-9.md ("either ladder" of 0x00 / 0x4F) |
| 4 | The red ring | Damage halves per grade, exactly (Red shot 4 → 2 → 1, Blue Wizzrobe touch 2 → 1 → ½, Patra child 2 → 1 → ½, Like-Like, Zol, Trap 1 → ½ → ¼, Keese ½ → ¼ → ⅛). **The detour's crossings cost 7.8 hearts (two bomb waits: 0x27 4.1, 0x17 2.8) and the ring saves 6.4 on the crossings after it**: on crossings alone it is a loss of 1.4; it pays only through the fights | `d9.md` §5 (costs were inferred) |
| 5 | Crossed rooms | Monsters match level-9.md in every room. Scripted crossings 20 of 20 everywhere; the expensive ones are the **bomb waits among Wizzrobes or a Patra: 0x65 4.5, 0x27 4.1, 0x17 2.8 hearts** before the ring, 0x31 1.0 and 0x04 0.8 with it. **0x14's spiral: 4.4 hearts walking, 0.05 with the sword (beams)**. Models: tunnels 118 of 120, Gel/Keese/Zol rooms 85-100%; **goriya dies in 0x65 (8 of 40; like-like 16 of 20 at −10 hearts), 0x15 (0 of 40), 0x27 (15 of 40); the Like-Like rooms end `stuck` (0-14 of 40)** | `d9.md` §4 (inferred), `rest-of-game.md` gap list |
| 6 | Ledgers | Rupees: nothing on the way takes or gives any (28 in, 28 at Ganon's door); **one arrow is one rupee, silver or wooden; at 0 no arrow flies**. 0x07's overworld cave takes 20 on entry | confirms `d9.md` §3.7 |
| 6 | Survival | **The route part alone costs about 23 hearts of 15** by the best scripted play, the big fights not counted: leg 7.2, level 9 before the ring 5.5, the detour 7.8, after the ring 2.6. Without the ring 20.9. Nothing refills on the way but drops | `rest-of-game.md` survival table |
| 7 | Engine | No missing model input found. Missing: the burned tree and the potion ware, 0x05's wall as a target, `ring` and `silver arrow` waypoints, kept kills for the four return visits, the Like-Like hold against the `stuck` rule, wrong-door masks at 0x63, families for every Wizzrobe, Patra and Like-Like crossing (§7) | — |

**The ten findings that most change round 29's plan**

1. **The hearts do not last to Ganon's door, before any fight is counted.** Best scripted play: 7.2 hearts on the
   leg, 15.9 on level 9's crossings with the ring detour, of 15 containers that nothing refills. Five places make 60%
   of it: overworld 0x06 (2.2) and 0x05 (2.9), and the bomb waits of 0x65 (4.5), 0x27 (4.1) and 0x17 (2.8).
2. **A bomb wait among Wizzrobes or beside a Patra is the costly thing, not the crossing.** 80 frames from B to
   the open wall, on the door square. Three other plays were tried and none is better: 0x65's bomb from the ladder
   in the water row (5.6 against 4.5), 0x27's and 0x17's bomb placed from the side with the fuse waited out in a
   corner (6.1 against 4.1; 3.6 against 2.8-3.5).
3. **The ring detour loses hearts on the route itself.** Its four walls and crossings cost 7.8 hearts at full
   damage (plus 0x07's fight), and it saves 6.4 on the fifteen crossings that follow. Whether it pays is decided by
   the fights after it (eight rooms and Ganon at a quarter), which the other walk measures.
4. **Every arrival spot in `d9.md` was 8 px off.** Open or key door: `$10` px in (S `$CD`, N `$4D`, W `$10`, E
   `$E0`); bombed wall or shutter: `$20` px in (`$BD`, `$5D`, `$20`, `$D0`). 0x30's arrival (`$D0`,`$8D`) is the
   right-hand Traps' own column.
5. **0x14's spiral must be fought, and the sword does it.** Walking it, a Like-Like holds Link in 20 of 20 runs
   (4.4 hearts); stabbing what stands in the lane, with beams while the hearts are full, 1 hold in 20 and 0.05
   hearts. The same guard takes 0x26 from 0.7 to 0.15 and 0x51 from 2.0 to 1.2 without the ring.
6. **Today's `stuck` rule ends the Like-Like rooms.** Link 50 actions in one place is `stuck`
   (`end_conditions.rs`), and a held Link cannot move: 0x14 0 of 20, 0x51 0 of 20, 0x41 2 of 20 (trap-room 14 of
   40), 0x17 0 and 11 of 40, 0x31 0 of 40, 0x03 0 of 20 (like-like too), nearly all ending `stuck` (like-like,
   goriya, zol). The hold itself was not read from those episodes: like-like also ended `stuck` in 0x16 (12 of 20),
   which has no Like-Like.
7. **Rooms the route has emptied are filled again by the evaluation.** The game keeps 0x07, 0x10 and 0x20 empty and
   0x61's Patra dead on the return visits (walked); the env's fresh-room write puts five Wizzrobes back into 0x20
   (goriya 0 of 40, 40 deaths) and the Patra back into 0x61.
8. **The whirlwind works as built, and no model plays it.** `zelda::whirlwind` offers RECORDER on Link's first frame
   in 0x6d with the index at `$F9`; ow-moblin (the family of 0x6d) rode it in 0 of 40 episodes and died in 33. One
   play, facing as he comes out, is all the step is.
9. **A route step is matched by exactly what Link holds.** The first scaffold left Dungeon 8 out, so its states
   held a piece the route did not: every room was silently off the route (RECORDER masked, every clips-off episode a
   wrong exit). One step for the D8 piece mended it. Round 29's states must carry the route's progress exactly, and
   nothing reports a mismatch.
10. **The zero-margin ledgers hold, but one wrong press breaks each.** 0x63 has three key doors and one key: N and
    S take it on the first frame. 0x07's overworld cave opens to a stray bomb and takes 20 rupees on entry (28 → 8;
    with under 20 it leaves none for Ganon's arrow). 0x47's left ware costs the fifteenth container.

## How the walks were staged

- **Out of Dungeon 8.** `triforce/probe.py` puts Link into 8:0x2c from S with the inventory above at 6.5 hearts and
  pieces 1-7, the index written `$F9`; he walks onto the piece and **the game itself** ends the level: out in
  0:0x6d at (`$60`,`$5D`) facing down, 14 hearts, `InvTriforce` `$FF` (`r29_route_walks.out_of_d8`).
- **The route in one game.** `r29_route_walks.route_states` then walks the leg and the draft's way through level 9
  to 0x52's N shutter once (13 seconds) and saves every room's pre-entry frame. On the way the monsters are **pinned**
  (their position rewritten inside the NW wall every frame, shots emptied): emptying the slots makes a room "all
  dead" to the game (asm:Secret_UpdateRoomAllDead), which shows its room item and opens its shutters, and a first
  pass picked up 0x26's rupees that way. In the rooms that must be emptied every counting monster is put into the
  death state (`ObjMetastate` `$10`, what a kill sets; a Patra's children first), so the game banks the kills and the
  return visits find what a played run finds. A wall reached with no bomb is opened with a written one (listed).
  0x66 and 0x06 are walked with nothing written (the men run their own setup).
- **Trials.** Every trial loads one of these states with `randomize_rng` and its seed: the monsters are made fresh
  and are live, the hearts real (15 full). "No ring" rows after the ring cellar are the same states with `InvRing`
  written 0. Hearts lost are read from the bytes (asm:HeartValues, asm:HeartPartial), so the ring's quarter and
  eighth hearts count.
- **A straight walk** goes from point to point and walks back onto its line after a knockback; while a Like-Like
  holds Link it stabs (one A every 12 frames). **The guard** (`--guard`) also stabs a monster the sword hurts when it
  stands within 40 px ahead in Link's lane, 112 px with full hearts (the beam).
- **Models.** `r29_route_scaffold.py` copies the data with the route `d9walk` (`d6` + the D8 piece's room + the
  draft's steps from 0x6d to 0x52), the rooms tagged, `requires_bombs` on the wall rooms and on 0:0x05 with
  `bomb_secret`, `ram_randomization: {}`, `bomb_reserve: 8` (bombs for walls only), start states from the staging
  above under the names the route gives its steps. The checkout's promoted runs (converted, 1.5.1.0), act mode
  `temperature:0.5+relax:16`, 20 episodes a state (families with clips: 20 with the switch on and 20 off), through
  `triforce eval-batch`. Guessed families: Wizzrobe and Patra crossings goriya, Like-Like rooms like-like, Zols
  zol, Gels and Keese stalfos-keese-gel, cellars equipment-room.

## 1. The leg (walked)

### 1.1 The whirlwind out of 0x6d

`r29_route_leg.py whirlwind`. Link comes out of Dungeon 8 at (`$60`,`$5D`) facing down, on the top row.

| Plays | Facing | Index | Lands |
|---|---|---|---|
| 1 | down (as he comes out) or left | `$F9` → `$F8` (level 1) | **0x37 at (`$80`,`$8D`)** |
| 2 | down or left | → `$F7` (level 8) | 0x6d itself |
| 1 | up or right | `$F9` → `$FA` (level 3) | 0x74 |
| 2 | up or right | → `$FB` (level 4) | 0x45 |

- One play facing down from every index, all eight pieces held: `$F8` 0x6d, **`$F9` 0x37**, `$FA` 0x3c, `$FB` 0x74,
  `$FC` 0x45, `$FD` 0x0b, `$FE` 0x22, `$FF` 0x42. Only the route's own count gives Dungeon 1.
- A press of B on Link's very first frame does nothing; from the second frame it plays.
- Timing: the whirlwind is made on the play, reaches Link 208 frames later (he is halted), leaves the screen at 284,
  0x37 is on screen at 388 and Link is released at (`$80`,`$8D`) at **475 frames**.
- **With the four Red Moblins live, Link waiting where he came out: 30 of 30 rides, 0 hearts.** No Moblin hurt him
  in the 208 frames before the pickup.
- A miss (Link kept invincible by writing his timer): the whirlwind is gone after 284 frames, the index reads `$F8`
  and the summoned flag stays 1; a second play in the room makes no whirlwind and steps the index to `$F7`. Out E
  and back in, one play facing up (`$F7` → `$F8`) rides to 0x37. The rule: after re-entering, play so that the last
  play before the pickup leaves the index on `$F8`.

### 1.2 Room by room

`r29_route_leg.py rooms --seeds 30` and `--guard`. Hearts are the mean over 30 seeds (maximum in brackets). Model
columns: success of 40 (20 with clips on, 20 off) and mean heart change.

| Room, entry → exit | Monsters made | Straight walk: path | Hearts lost | With the guard | Frames | Family model |
|---|---|---|---|---|---|---|
| 0x6d cave → whirlwind | 4 Red Moblins | B once | 0 | 0 | 477 | **ow-moblin 0/40** (33 deaths, 7 `max-steps`), −13.2 |
| 0x37 whirlwind → E | 1 fast Red Octorok | E along Y `$8D` | 0.15 (1) | 0 | 275 | ow-octorok 40/40, −0.05 |
| 0x38 W → S | 5 fast Red, 1 Blue Octorok, Zora | E to X `$78`, S | 0.30 (1) | 0.12 | 248 | ow-zora 40/40, −0.33 |
| 0x48 N → W | 4 Red Leevers, Zora | down to Y `$8D`, W | 0.63 (1) | 0.52 (856 frames) | 320 | **ow-leever 0/40** (40 wrong exits) |
| 0x47 E → tree, cave, container → out | Zora | W to X `$B0`, down to Y `$9D`, candle S, stairs, the right ware, out | 0.35 (1); 27 of 30 | same | 660 | no objective (§7) |
| 0x47 cave → E | Zora | up to Y `$8D`, E | 0 | 0 | 163 | (same) |
| 0x48 W → N | 4 Red Leevers, Zora | E to X `$78`, N | 0.90 (1) | 0.43 | 301 | **ow-leever 12/40** (19 wrong exits, 7 deaths) |
| 0x38 S → N | as above | straight up X `$78` | 0.58 (1.5) | 0.07 | 223 | ow-zora 40/40, −0.44 |
| 0x28 S → W | 6 Peahats, Zora | up to Y `$8D`, W | 0 | 0 | 254 | ow-zora 40/40, −0.18 |
| 0x27 E → N | 6 Peahats, Zora | W to X `$90`, N | 0 | 0 | 217 | ow-zora 36/40, −0.06 |
| 0x17 S → N | 3 Boulders, Zora | up to Y `$9D`, W over the river (the ladder) to X `$40`, N | 0.25 (1) | 0.25 | 308 | **ow-boulder 0/40** (40 wrong exits in 115 frames) |
| 0x07 S → W | 1 Red Lynel | up to Y `$8D`, W | 0.27 (2) | 0.27 | 254 | ow-lynel 40/40, −0.65 |
| 0x06 E → W | 2 Blue, 2 Red Lynels, 2 Peahats | W along Y `$8D` | **2.23 (5)** | 2.50, 26 of 30 (four runs ended in the room; the most lost was all 15) | 318 | **ow-lynel 12/40** (28 deaths), −11.7 |
| 0x05 E → the wall → level 9 | 2 Blue, 2 Red Lynels, Blue and Red Leever | W to X `$D0`, down to Y `$AD`, W to X `$50`, bomb N, the mouth | **2.92 (11)** | 2.77, 29 of 30 | 419 | **ow-lynel 0/40** (27 wrong exits, 13 deaths, no bomb placed) |

- **Sums.** Straight 8.58 hearts; with the guard 7.27; the better of the two per room **7.2**. The guard helps
  against Octoroks and Leevers and hurts against Lynels (Link stands to stab and is shot: 70 hearts to sword shots
  in 30 runs of 0x06, four of them not finished).
- **0x47's three failures** ended outside the cave at (`$D0`,`$9D`) after a hit; the script does not recover, and
  why was not read. All of 0x47's damage is the Zora's fireballs.
- **0x48**: the Leevers surface on Link's path; stopping to stab them costs time (856 frames against 320), not
  hearts.
- **0x06 and 0x05** were walked on one lane each (the door row; round the E rock by the S side). Other lanes were
  not tried.
- **Monsters** are the level data's in every room (`d9.md` §1.2's table).

### 1.3 0x47: the tree, the cave, the wares

`r29_route_leg.py tree`. The room is a strip two squares high (Y `$8D` and `$9D`); the tree is a tile object at
(`$B0`,`$B0`), under the strip.

- The red candle facing **S** from X `$A8`, `$B0` or `$B8`, on Y `$9D` or `$8D`: the tree is gone 94 frames after B
  and the room's flag is `$80`. From `$A0` or `$C0`, or facing E or W beside it: no.
- Link's own fire: standing still after B costs nothing; walking down into it costs ½ heart.
- The engine's `CANDLE S` is offered at (`$B0`,`$9D`) and burns the tree.
- The stairs take Link after 12 frames of Down from (`$B0`,`$9D`). In the cave he stands at (`$70`,`$D5`) and can
  move at once.
- **The right ware** (X `$98`): Up from Y `$BD` takes it after 23 frames at (`$98`,`$9D`): containers 14 → 15,
  hearts 14 → 15 (he was full), flag `$90`, Link halted 126 frames. **The left ware** (X `$58`): the red potion
  (`InvPotion` 2), containers stay 14, flag `$90`. After either the cave holds only its two fires, on every later
  visit: the other ware is gone for good.
- Out at (`$B0`,`$9D`) facing down, the stairs still open. (In the cave Link is halted while the old man's text
  types: his state read `$40` 160 frames after the stairs and 0 at 200.)

### 1.4 0x17's ladder, 0x05's wall, 0x07's cave

`r29_route_leg.py wall`.

- **0x17**: the river is one square wide (X `$60`-`$6F`) from Y `$8D` down; Link enters at (`$90`,`$DD`), E of it,
  and the N exit is the corridor at X `$40`, W of it. Holding Left on row Y `$9D` (the one walked) lays the ladder:
  61 frames from X `$90` to `$40`. Boulders fall in Link's half: 6.5 hearts to Boulders in 30 crossings.
- **0x05**: the rock wall is the tile object at (`$50`,`$A0`), in the S face of the W rock. From the square under
  it, (`$50`,`$AD`), Up moves Link 3 px (to Y `$AA`) and one bomb facing N opens it **49 frames after B**; so do X
  `$48` and `$58`, and a bomb facing E from (`$38`,`$AD`) or W from (`$68`,`$AD`) (it lands under the wall); `$40`
  and `$60` facing N do not. One bomb, flag `$80`. 10 frames of Up put Link on the mouth; level 9 is playable 182
  frames later at (`$78`,`$CD`).
- The wait: Link stands 49 frames with six monsters about; the straight walk's 2.92 hearts in 0x05 are Lynel touches
  (52 of 87 hearts) and sword shots.
- **0x07's cave**: a bomb facing N from (`$A0`,`$8D`), on the route's own row, opens the rock wall at
  (`$A0`,`$80`); 9 frames of Up enter it and the rupees go 28 → 8 with no question asked.

## 2. 0x66's man and the eight pieces (walked)

`r29_route_d9.py man`.

| `InvTriforce` | The man | Shutters (N and W) | How far Link walks up the middle |
|---|---|---|---|
| `$FF` | gone | open 18 frames after Link's first frame | through the N door (0x56) |
| `$7F` (no D8 piece) | stands at (`$78`,`$80`) | shut (300 frames watched) | Y `$8D` |
| `$FE`, `$00` | the same | shut | Y `$8D` |

- He checks the eight pieces and nothing else; nothing Link does in the room changes it. With fewer the room is a
  dead end: the only way out is S.
- Link's first frame is at (`$78`,`$CD`). The route walks up to Y `$8D` and W along it: **0 hearts in 20 of 20**.
  The fires at (`$48`,`$80`) and (`$A8`,`$80`) cost ½ heart to a Link above the row (Y `$89`, `$85`), nothing on it.
- **0x06's man**: Link's first frame is at (`$78`,`$CD`); from the next frame he is halted **134 frames** by the
  text; then up to Y `$8D` and no further. Struck four times with the sword he stays, and two fireballs appear and
  stay while Link is in the room.

## 3. The graph (walked)

`r29_route_d9.py graph | doors | tunnels | history`. One game from 0x6d; the rooms marked *emptied* were emptied by
staging (another walk's fights).

### 3.1 The way, room by room

Keys and bombs are what Link holds when he leaves the room, on a run with no drops; **w** marks a written bomb.

| # | Room | Enters by → first movable | What is done | Leaves by | Keys | Bombs |
|---|---|---|---|---|---|---|
| 1 | 0x76 | the stairs → (`$78`,`$CD`) | up | N open | 3 | 0 |
| 2 | 0x66 | S open → (`$78`,`$CD`) | up to Y `$8D`, W | W shutter (open) | 3 | 0 |
| 3 | 0x65 | E open → (`$E0`,`$8D`) | W to X `$78`, up over the water row on the ladder to Y `$5D` (37 frames), bomb N | N wall | 3 | 0 (**w**) |
| 4 | 0x55 | S wall → (`$78`,`$BD`) | *emptied*; (`$60`,`$9D`), Up 15 frames: the block; up to Y `$8D`, E onto the stairs | stairs | 3 | 0 |
| 5 | cellar 0x60 | right ladder, (`$C0`,`$5D`) | down, W, up the left ladder | left ladder | 3 | 0 |
| 6 | 0x14 | (`$60`,`$9D`), inside the spiral | E to `$B0`, up to `$5D`, W to `$20`, down to `$BD`, E to `$D0`, up to `$8D` (454 frames unhindered) | E key door | 2 | 0 |
| 7 | 0x15 | W key → (`$10`,`$8D`) | hold Right: the ladder twice | E open | 2 | 0 |
| 8 | 0x16 | W open → (`$10`,`$8D`) | round the blocks to (`$78`,`$AD`) or by the bottom lane | S open | 2 | 0 |
| 9 | 0x26 | N open → (`$78`,`$4D`) | to (`$D0`,`$8D`), bomb E | E wall | 2 | 0 (**w**) |
| 10 | 0x27 | W wall → (`$20`,`$8D`) | by X `$40` and the top row to (`$78`,`$5D`), bomb N | N wall | 2 | 0 (**w**) |
| 11 | 0x17 | S wall → (`$78`,`$BD`) | round the middle blocks by X `$60` to (`$78`,`$5D`), bomb N | N wall | 2 | 0 (**w**) |
| 12 | 0x07 | S wall → (`$78`,`$BD`) | *emptied*; the block from (`$60`,`$9D`); the stairs (`$80`,`$8D`) | stairs | 2 | 0 |
| 13 | cellar 0x00 | left ladder, (`$30`,`$5D`) | down, E to `$B0`, up to the shelf, W: **the red ring** at X `$88`; back | left ladder | 2 | 0 |
| 14 | 0x07 | (`$60`,`$9D`), the block back | S | S wall (open) | 2 | 0 |
| 15 | 0x17 | N wall → (`$78`,`$5D`) | round the blocks | S wall (open) | 2 | 0 |
| 16 | 0x27 | N wall → (`$78`,`$5D`) | W along the top row, down X `$40` | W wall (open) | 2 | 0 |
| 17 | 0x26 | E wall → (`$D0`,`$8D`) | to (`$78`,`$5D`) | N open | 2 | 0 |
| 18 | 0x16 | S open → (`$78`,`$CD`) | *emptied*: the bombs at (`$80`,`$90`) (+4, taken at Y `$95`); by X `$50` to (`$78`,`$5D`) | N key door | 1 | 4 |
| 19 | 0x06 | S key → (`$78`,`$CD`) | halted 134 frames; under the fire to (`$30`,`$8D`), bomb W | W wall | 1 | 3 |
| 20 | 0x05 | E wall → (`$D0`,`$8D`) | *emptied*; the block from (`$60`,`$9D`): stairs at (`$D0`,`$60`) | stairs | 1 | 3 |
| 21 | cellar 0x70 | right ladder | down, W, up the left ladder | left ladder | 1 | 3 |
| 22 | 0x63 | (`$A0`,`$9D`) | up to Y `$8D`, W | W key door | 0 | 3 |
| 23 | 0x62 | E key → (`$E0`,`$8D`) | W | W open | 0 | 3 |
| 24 | 0x61 | E open → (`$E0`,`$8D`) | the key at (`$D0`,`$60`), taken at Y `$65` walking up X `$D0`; *emptied*; the block; the stairs | stairs | 1 | 3 |
| 25 | cellar 0x75 | right ladder | W | left ladder | 1 | 3 |
| 26 | 0x20 | (`$60`,`$9D`) | *emptied*; to (`$78`,`$5D`), bomb N | N wall | 1 | 2 |
| 27 | 0x10 | S wall → (`$78`,`$BD`) | *emptied*; by the E lane to (`$D0`,`$8D`), Left 16 frames: stairs at (`$D0`,`$60`) 30 frames later | stairs | 1 | 2 |
| 28 | cellar 0x4F | left ladder | as 0x00: **the silver arrow** | left ladder | 1 | 2 |
| 29 | 0x10 | (`$60`,`$9D`), inside the C | W out of the C, by the W lane | S wall (open) | 1 | 2 |
| 30 | 0x20 | N wall → (`$78`,`$5D`) | still empty: the block, the stairs | stairs | 1 | 2 |
| 31 | cellar 0x75 | left ladder | E | right ladder | 1 | 2 |
| 32 | 0x61 | (`$60`,`$9D`) | the Patra stays dead; to (`$78`,`$5D`) | N open | 1 | 2 |
| 33 | 0x51 | S open → (`$78`,`$CD`) | Y `$BD`, W to X `$70`, up to Y `$7D`, E to X `$80`, up to Y `$5D`, W to X `$78` (107 frames unhindered) | N open | 1 | 2 |
| 34 | 0x41 | S (its shutter) → (`$78`,`$BD`) | straight up | N open | 1 | 2 |
| 35 | 0x31 | S (its shutter) → (`$78`,`$BD`) | to (`$30`,`$8D`), bomb W | W wall | 1 | 1 |
| 36 | 0x30 | E wall → (`$D0`,`$8D`) | *emptied*; the block; the Trap lure onto the stairs (§3.4) | stairs | 1 | 1 |
| 37 | cellar 0x67 | left ladder | E | right ladder | 1 | 1 |
| 38 | 0x04 | (`$C0`,`$6D`) | W to X `$90`, up to Y `$5D`, W to X `$70`, down to Y `$6D`, W to X `$30`, down to (`$30`,`$8D`), bomb W | W wall | 1 | 0 |
| 39 | 0x03 | E wall → (`$D0`,`$8D`) | *emptied* here (its fight is not walked, §5.3); the block; the stairs | stairs | 1 | 0 |
| 40 | cellar 0x77 | right ladder | W | left ladder | 1 | 0 |
| 41 | 0x52 | (`$60`,`$9D`) | *emptied*: the N shutter opens 9 frames later | N shutter | 1 | 0 |
| – | 0x42 | S (its shutter) → (`$78`,`$BD`) | Ganon's room: not walked here | | 1 | 0 |

Without the ring steps 9-17 fall out and 0x16 is entered once, from W, and left N: 31 rooms and cellars after 0x76
to Ganon's room, 2 written bombs (the rock wall, 0x65 N), the same keys.

### 3.2 Doors

- **Arrival spots.** Through an open doorway or a key door Link is first movable `$10` px inside the room edge's
  door square: from S Y `$CD`, from N Y `$4D`, from W X `$10`, from E X `$E0`. Through a bombed wall or a shutter
  `$20` px: Y `$BD`, Y `$5D`, X `$20`, X `$D0`. Every arrival on the way fits (§3.1). In a doorway Link cannot move
  sideways.
- **Key doors.** 0x14 E, 0x16 N and 0x63 W each take a key on the first frame Link presses into them from the door
  square. **0x63 N and 0x63 S do the same** with the last key: Link comes out in 0x53 at (`$78`,`$CD`) or 0x73 at
  (`$78`,`$4D`) with none, and 0x63 W is then shut to him. 0x63's W door is reached along Y `$8D`; N and S are 48 px
  off that row at X `$78`.
- **One-way shutters.** 0x41 S and 0x31 S: 200 frames of Down from the arrival spot, the room full and the room
  emptied: still inside, no door bit set. From 0x41 on there is no way back; what Ganon needs (the silver arrow, a
  rupee) and the two bombs for 0x31 W and 0x04 W must be held when Link leaves 0x61 N.
- **0x51's W shutter** opens 8 frames after its six Like-Likes are gone (the fall-back with two spare keys; not on
  the way).
- **Bomb walls.** One bomb each, the wall is passable **80 frames after B** in all eight. From the door square and
  from further back along the door's axis: 0x65's N wall from Y `$5D` to `$7D` (the last two on the ladder in the
  water row), 0x31's W wall from X `$30` to `$48`, not from `$50`. A bomb laid from beside the door square so that
  it lands on it opens the wall too (0x27 and 0x17 from (`$60`,`$5D`) facing E: 20 of 20).
- **Blocks.** They move after 14-15 frames of the one direction, only with no counting monster left (0x55: 30
  frames of Up with the Lanmolas alive, nothing). Out of a cellar the block is back.

### 3.3 Tunnels and cellars

Link arrives on top of a ladder at Y `$5D` (X `$C0` when he came from the cellar's right-exit room, X `$30` from
its left-exit room), walks down to the floor (Y `$BD`), along it and up a ladder.

| Cellar | Left ladder (X `$30`) → | Right ladder (X `$C0`) → | The route | The wrong ladder |
|---|---|---|---|---|
| 0x60 | 0x14 at (`$60`,`$9D`) | 0x55 at (`$60`,`$9D`) | in right, out left | right: back in 0x55 |
| 0x70 | 0x63 at (`$A0`,`$9D`) | 0x05 at (`$A0`,`$9D`) | in right, out left | right: back in 0x05 |
| 0x75 | 0x20 at (`$60`,`$9D`) | 0x61 at (`$60`,`$9D`) | first in right, out left; then in left, out right | the one he came down |
| 0x67 | 0x30 at (`$C0`,`$6D`) | 0x04 at (`$C0`,`$6D`) | in left, out right | left: back in 0x30 |
| 0x77 | 0x52 at (`$60`,`$9D`) | 0x03 at (`$60`,`$9D`) | in right, out left | right: back in 0x03 |
| 0x00, 0x4F | 0x07 / 0x10 at (`$60`,`$9D`) | none (Up from the floor at X `$C0` goes nowhere) | in left, the shelf by the short ladder at X `$B0`, out left | – |

- The wrong ladder puts Link back in the room he left, outside its ring, its block back: 0x05 one cellar after it
  was emptied was still empty (the history) with its block back in place. In 0x63 the stairs at (`$D0`,`$8D`),
  at the E end of the door row, lead back down into 0x70: the route walks W from (`$A0`,`$9D`), away from them.
- Every cellar holds 4 blue Keese. Scripted: 20 of 20 each, 0.10-0.30 hearts before the ring, 0.03-0.07 with it;
  equipment-room 118 of 120 (0x75's second pass 18 of 20: two wrong ladders).

### 3.4 Rooms that fill or lock again

`r29_route_d9.py history`: what Link finds on each return of the draft's way.

| Return | Entries since he left | Finds |
|---|---|---|
| 0x07 out of the ring cellar | 1 | 3 Bubbles, no Wizzrobe; the block back |
| 0x17, 0x27, 0x26 on the way back | 3, 5, 7 | full (never emptied) |
| 0x16 from S | 9 | the Patra (not killed on the first visit) |
| 0x10 out of the arrow cellar | 1 | 3 Bubbles; the block back, the stairs gone |
| 0x20 from N | 3 | empty; the block moves at once |
| 0x61 out of the tunnel | 7 | empty: the Patra stays dead (a Patra never returns) |

- Nothing on the draft's way needs a room to stay empty for more than three entries (0x61 rests on the Patra's
  own rule, not on the history); the six-room history is not strained.
- Shutters: 0x66's open on every entry with eight pieces; 0x52's N opens when its Patra is gone.
- **0x30's stairs** appear on the top-right Trap's home (`$D0`,`$5D`). The lure, with the room emptied: from
  (`$B0`,`$6D`) Up 3 frames to Y `$69` (both top Traps sense Link and charge along Y `$5D`, 12 px above him: no
  touch), back down, E along Y `$6D` to X `$D0`, Up: on the stairs 40 frames after the first Up, 0 hearts. Traps
  hold no chance: the same every time. Not walked with the Wizzrobes alive (the fight is the other walk's).

## 4. The red ring and the silver arrow (walked)

`r29_route_d9.py ring --seeds 20` (Link standing still at the arrival spot, 600 frames, `InvRing` written 0, 1, 2;
every hit's size read from the hearts' bytes) and `crossings`.

| Hit | No ring | Blue ring | Red ring | Hits seen (per grade) |
|---|---|---|---|---|
| Red Wizzrobe's shot | 4 | 2 | 1 | 28 |
| Blue Wizzrobe's touch (a two-heart touch) | 2 | 1 | ½ | 46 |
| Blue Wizzrobe's shot | 2 | 1 | ½ | 9 |
| Patra child | 2 | 1 | ½ | 29 |
| Like-Like, Zol, Trap | 1 | ½ | ¼ | 12, 8, 192 |
| Keese | ½ | ¼ | ⅛ | 20 |

Exactly halved per grade, over 340 hits. (Standing 600 frames at 0x41's arrival spot costs 9.6 hearts without the
ring: a Trap every 62 frames.)

**What the detour costs and saves on the route** (crossings only; means of 20, the better play of each):

| | Rooms and cellars entered | Bombs | Hearts |
|---|---|---|---|
| The detour: 0x26 (bomb), 0x27 (bomb), 0x17 (bomb), 0x07 (**fight**), cellar 0x00, 0x07, 0x17, 0x27, 0x26, and 0x16 a second time | 10 | 3 | 0.15 + 4.10 + 2.80 + (0x07's fight, at full damage) + 0 + 0.73 + 0 + 0.01 = **7.8 + the fight** |
| The fifteen crossings after it, with the ring | | | **2.6** |
| The same without | | | 8.9 |

- **On crossings the ring loses 1.4 hearts** (7.8 spent, 6.4 saved), before 0x07's fight. It also turns 0x16's
  Patra from optional into the only bomb source short of cheats. What it buys is three quarters of the damage of the
  eight fights after it and of Ganon (four hearts a touch without it).
- The detour's cost is two bomb waits: 0x27's door square is 19 px above the Patra's start (64 hearts to its
  children and 18 to the body in 20 crossings) and 0x17's has Like-Likes and four Wizzrobes about.

**The silver arrow.** 0x10 from S (`$78`,`$BD`), emptied by staging: by the bottom row and the E lane (outside the
C of blocks) to (`$D0`,`$8D`), Left 16 frames: the block at (`$C0`,`$90`) slides and the stairs
are at (`$D0`,`$60`) 30 frames later; Up from (`$D0`,`$8D`) onto them. Cellar 0x4F as 0x00: down the left ladder, E
to X `$B0`, up the short ladder to the shelf (Y `$8D`), W: the arrow is taken at X `$88`, Link halted 128 frames;
back and up the left ladder: 0x10 at (`$60`,`$9D`) inside the C, 3 Bubbles, no Wizzrobe; out by the W side.
`r29_route_d9.py arrow`: an arrow costs **one rupee, silver or wooden** (28 → 27, 1 → 0); at 0 rupees B shoots
nothing.

## 5. Every room that is only crossed (walked)

`r29_route_d9.py crossings --seeds 20` (straight), `--guard`, `--best`. "Seen" = the monsters made (120 frames into
the room); they match level-9.md's room table in every room. Hearts: mean of 20 (maximum). Models: scaffold
`d9walk`, success of 40 (20 for families without clips) and mean heart change; all model states hold 15 hearts.

### 5.1 Before the ring (full damage)

| Room | Seen | Scripted crossing | Straight | Guard | Model |
|---|---|---|---|---|---|
| 0x76 S → N | nothing; the statues do not shoot | up | 0 | 0 | stalfos-keese-gel 40/40, 0 |
| 0x66 S → W | 2 fires | §2 | 0 | 0 | stalfos-keese-gel 40/40, −0.08 |
| 0x65 E → N, bomb | 3 Bubbles, 3 Blue, 2 Red Wizzrobes (dark) | to (`$78`,`$5D`), bomb, wait on the square | **4.45 (8)** | 4.45 | **goriya 8/40** (32 deaths), −13.6 |
| cellar 0x60 | 4 Keese | §3.3 | 0.30 (1) | 0.30 | equipment-room 20/20, −0.07 |
| 0x14 tunnel → E | 5 Like-Likes | the spiral | **4.35 (5)**, held in 20 of 20, 1050 frames | **0.05 (1)**, held in 1, 1044 frames | **like-like 0/20** (`stuck`) |
| 0x15 W → E | 3 Blue Wizzrobes (dark) | hold Right | 0 | 0 | **goriya 0/40** (35 deaths), −14.2 |
| 0x16 W → S | Patra, 8 children | round to (`$78`,`$AD`); **by the bottom lane** | 1.30 (2); **0.70 (2)** | – | goriya 27/40 (13 deaths), −9.8 |
| 0x26 N → E, bomb | 8 Gels | to (`$D0`,`$8D`), bomb | 0.68 (2) | 0.15 (0.5) | stalfos-keese-gel 34/40, −2.4 |
| 0x27 W → N, bomb | Patra, 8 children | top row to (`$78`,`$5D`), bomb, wait | **4.10 (6)** | 4.10 | **goriya 15/40** (25 deaths), −13.2 |
| 0x17 S → N, bomb | 1 Bubble, 3 Like-Likes, 2 Blue, 2 Red Wizzrobes | by X `$60` to (`$78`,`$5D`), bomb, wait | 3.50 (7), held in 13 | **2.80 (10)**, held in 8 | **goriya 0/40** (`stuck` 39) |

- **0x65.** 115 frames to the door square, the bomb, 80 frames of waiting: 40 hearts to Blue shots along the top
  row, 27 to Red Wizzrobes' touches and 16 to Blue ones' in 20 runs. From the ladder at (`$78`,`$7D`), in the water
  row, the wall opens too but the wait costs more (5.55). The model walks into the water room's Wizzrobes and dies.
- **0x14.** The corridors are one square wide and the Like-Likes walk them. A held Link is freed only by killing
  the holder (three magic hits); walking on, he is held again by the next. With the guard he stops at 40 px (112 with
  full hearts) and kills them first: the spiral takes the same 1044 frames.
- **0x15.** Holding Right from the W door lays the ladder twice and nothing reached Link in 20 runs (381 frames).
  goriya dies to the three Blue Wizzrobes (35 of 40).
- **0x16.** No kill needed; the bottom lane (X `$20`, Y `$BD`) halves the cost. The children are out and harmful
  before Link is across (26 hearts in 20 runs on the short way).
- **0x27.** From the side ((`$60`,`$5D`) facing E, the fuse waited out in the NW corner) the wall opens 20 of 20
  but the Patra follows Link: 6.10.
- **0x17.** From the side with the corner wait: 3.60 (fewer holds, more Red shots). Two runs of 20 picked up a
  bomb drop (+4) from a Like-Like the guard killed.

### 5.2 After the ring (the same states with and without it)

The better of the straight and the guarded run; where they differ much both are given. The with-ring numbers are
the guarded run's, whose hearts are read from the bytes (the straight run's half-heart reading rounds the ring's
quarter hearts away).

| Room | Seen | Scripted crossing | With the ring | Without | Model (with the ring) |
|---|---|---|---|---|---|
| 0x07 cellar → S | 3 Bubbles | to (`$78`,`$BD`), S | 0 | 0 | goriya 40/40, −0.39 (in a room the env refilled: 3 Blue, 2 Red, §7) |
| 0x17 N → S | as 0x17 | round the blocks | 0.73 (2.75), held in 2 | 2.60 (10) | **goriya 11/40** (`stuck` 29) |
| 0x27 N → W | Patra | top row W, down X `$40` | 0 | 0 | goriya 40/40, −1.08 |
| 0x26 E → N | 8 Gels | to (`$78`,`$5D`), N | 0.01 | 0.05 | stalfos-keese-gel 40/40, −0.02 |
| 0x06 S → W, bomb | the man, 2 fires | §2 | 0 | 0 | stalfos-keese-gel 40/40, 0 (its own bomb) |
| cellar 0x70 | 4 Keese | | 0.07 | 0.28 | equipment-room 20/20 |
| 0x63 tunnel → W | 5 Zols | up to Y `$8D`, W | 0 (guard); 0.28 straight | 0 (guard); 1.50 straight | zol 20/20, −0.28 |
| 0x62 E → W | 8 Keese | W | 0.01 | 0.03; 0.60 straight | stalfos-keese-gel 40/40, 0 |
| cellar 0x75 | 4 Keese | | 0.07 | 0.28 | equipment-room 20/20 |
| 0x10 cellar → S | 3 Bubbles | out of the C by the W | 0 | 0 | goriya 40/40, −0.81 (refilled: 3 Blue, 2 Red) |
| 0x20 N → stairs | nothing | the block, the stairs: 300 frames | 0 | 0 | **goriya 0/40** (40 deaths: refilled) |
| cellar 0x75 back | 4 Keese | | 0.03 | 0.10 | equipment-room 18/20 |
| 0x61 tunnel → N | nothing | to (`$78`,`$5D`), N | 0 | 0 | goriya 40/40, −4.25 (the Patra put back) |
| 0x51 S → N | 6 Like-Likes | the zig-zag | 0.33 (1), held in 14-20 | 1.20 (4); 1.95 straight | **like-like 0/20** (`stuck`) |
| 0x41 S → N | 4 Traps, 4 Like-Likes (dark) | straight up | 0.16 (0.5), held in 10-13 | 0.70 (2); 1.45 straight | **like-like 2/20** (`stuck` 18) |
| 0x31 S → W, bomb | 1 Bubble, 3 Like-Likes, 2 Blue, 2 Red | to (`$30`,`$8D`), bomb | 1.03 (3.5) | 2.85 (8) | **goriya 0/40** (`stuck`) |
| cellar 0x67 | 4 Keese | | 0.04 | 0.18 | equipment-room 20/20 |
| 0x04 tunnel → W, bomb | 4 Traps, 2 Blue, 2 Red | §3.1 #38: the top row between X `$90` and `$70`, then (`$30`,`$8D`) | 0.76 (2.5) | 3.05 (10) | goriya 28/40 (6 deaths), −7.9 |
| cellar 0x77 | 4 Keese | | 0.06 | 0.25 | equipment-room 20/20 |

- **0x41.** Walking straight up, no Trap touched Link in 20 runs: all the damage is Like-Likes on the middle lane
  (held in 13 of 20; the guard brings it to 10). Standing still on the arrival spot is what the Traps punish (§4).
- **0x04.** No Trap touch on the crossing as walked (20 of 20); the damage is the Red Wizzrobes' shots during the
  bomb wait at (`$30`,`$8D`): 40 hearts without the ring, 10 with it.
- **0x51.** Like-Likes hold Link on the zig-zag in most runs with or without the guard; each hold is three stabs.
- **Bubbles.** 0x65, 0x17, 0x31, 0x07, 0x10 have them; a touch takes no heart and turns the sword off 160 frames
  (`$4C` = `$10`, d6-walks.md §5); the scripted crossings never needed the sword while it was off except the
  guard, which then walks on. Red and blue Bubbles (the sword off until a blue one is touched) are a second-quest
  monster: level 9's are the flashing kind (type `$2B`) in every room seen.
- **Traps** (0x41, 0x04, 0x30) and **statues** (0x76: none shoot) as above.

### 5.3 0x03 (Zols and Like-Likes, then the push): not walked

2 Bubbles, 2 Zols, 2 Like-Likes are made (matches the data); staged empty, the block moves and the stairs lead to
0x77 and 0x52. **No scripted fight was written for it**, and the one model tried (zol) ended `stuck` in 20 of 20
after 498 frames (held). It is a forced fight of the route that this walk has not played (the gap list has it
with the other fights, item 1).

## 6. Ledgers and the survival budget

**Bombs** (the bag holds 8; `r29_route_d9.py graph`):

| | At 0x6d | Rock wall | 65 N | 26 E | 27 N | 17 N | 0x16's Patra | 06 W | 20 N | 31 W | 04 W | Written |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| No drops, the ring | 0 | **w** | **w** | **w** | **w** | **w** | +4 → 4 | 3 | 2 | 1 | 0 | 5 |
| No drops, no ring | 0 | **w** | **w** | – | – | – | +4 → 4 | 3 | 2 | 1 | 0 | 2 |
| Full bag, the ring | 8 | 7 | 6 | 5 | 4 | 3 | +4 → 7 | 6 | 5 | 4 | 3 | 0 |

- All nine walls are forced on the draft's way (the ring's three are the detour's). The way's only source is
  0x16's Patra; a run short of bombs has cheated exactly the walls before it.
- Drops: a Like-Like killed in 0x17 left four bombs in 2 of 20 guarded crossings. Not a plan.
- **Another way with fewer:** none was walked. From the level data: without the ring the count is 5 walls inside;
  0x23's four bombs lie in the open two rooms off 0x14 (level-9.md's trace: 4 entries, 1 bomb, net +3), which the
  owner's rule (Nov 14: no extra rooms for bombs) excludes; the 0x51 W - 0x50 - 0x40 way round 0x31's wall needs two
  spare keys and Link has one.
- Two bombs must be held on leaving 0x61 N (0x31 W, 0x04 W; no way back from 0x41): on the no-drop run that is
  exactly what is left.

**Keys**: 3 in; 0x14 E (2), 0x16 N (1), 0x63 W (0), 0x61's key (1); 1 at Ganon's door, never needed. No margin from
0x16 to 0x61: one press into 0x63 N or S ends the run at 0x63.

**Rupees**: 28 in, 28 out; the way neither takes nor gives one (0x62's five appear only if its 8 Keese die; 0x26's
if its 8 Gels die). Ganon's arrow costs 1. The only loss on the route part is 0x07's overworld cave (−20).

**The survival budget of the route part** (hearts lost, the big fights skipped; best play seen of each room; models
where a model did better than the script are named):

| Part | Rooms | With the ring | Without the ring |
|---|---|---|---|
| Containers | | 14, 15 from 0x47 | the same |
| The leg, 0x6d to the mouth | 13 | 7.2 (0x06 2.2, 0x05 2.9) | 7.2 |
| 0x76 to 0x16 (first visit) | 7 | 5.5 (0x65 4.45) | 4.8 (0x16 is left N) |
| The detour and back | 9 | 7.8 + 0x07's fight | – |
| 0x06 to 0x52's tunnel | 15 | 2.6 | 8.9 |
| **Sum** | | **23.1** | **20.9** |
| Not counted | | 0x55, 0x07, 0x16, 0x05, 0x61, 0x20, 0x10, 0x30, 0x03, 0x52 | 0x55, 0x16 (for bombs), 0x05, 0x61, 0x20, 0x10, 0x30, 0x03, 0x52, at full damage |

- **Refills: none.** No fairy pond is on the way, the potion is excluded (and would cost the container), level 9
  has no piece. What gives hearts back is drops only; the crossings kill little and the scripted runs' numbers
  already include what they picked up.
- The models do better than the scripts nowhere it matters: ow-zora's 0x38 and 0x28 are within 0.3 hearts of the
  script, and every room that costs the script more than two hearts kills the model that was tried.
- So the route part needs 8 hearts more than Link has **before any fight**, by the best play seen. The five rooms to
  win back are named in finding 1; three of them are bomb waits.

## 7. What the engine lacks for round 29's route

In the terms of `d6.md` §14. Evidence is the scaffold `d9walk` (55 steps appended to `d6`; it loads: 336 steps) and
the walks above.

**Route steps and waypoints**

| Need | Evidence | Today |
|---|---|---|
| 0x47's burned tree as a cave entrance (`entrance: {how: burn}`, tile (22, 14)): an `exit: cave` target under the tree, CANDLE S there, then the stairs; CANDLE on for the family | §1.3: X `$A8`-`$B8`, either row, 94 frames; `CANDLE S` is offered and works | game.yaml data only; the scaffold enters and leaves 0x47 E (round 28 builds the burn for 0x6d) |
| 0x47's cave: `waypoint: heart` on the right ware and **the left ware never taken** | §1.3: the potion takes the container for good | `waypoint: heart` exists (0x2c, 0x2f); no ware mask |
| 0x05's rock wall as the `exit: cave` target | §1.4: one bomb from X `$48`-`$58`; ow-lynel placed no bomb in 40 episodes with `bomb_secret: [10, 12]` and `requires_bombs` added | the keys exist for 0x2c and 0x67; game.yaml 0x05 has only `entrance: {how: bomb}`; whether BOMB is offered under the wall was not read |
| The whirlwind step 0x6d → 0x37 | §1.1; RECORDER is offered on the first frame (index `$F9`, eight pieces) | built (round 26); the family must learn it: ow-moblin 0/40 |
| 0x66: the eight pieces as the step's condition | §2: fewer is a dead end with no ending | none; `waypoint: sword`'s refusal (`missed`) is the model |
| 0x06: the halt (134 frames), never SWORD at the man | §2: fireballs for the rest of the visit | `halted` is observed; no mask |
| `waypoint: ring` (cellar 0x00) and `waypoint: silver-arrow` (0x4F) | §3.1 #13, #28; the scaffold's cellar steps had no waypoint | `WAYPOINTS` holds heart, rupees, bow, raft, ladder, recorder, arrows, food, candle, sword |
| Tunnels with two exits, 0x75 both ways | §3.3; equipment-room 118/120; `9_75cave` and `9_75cave_v2` resolve | built (round 24) |
| Stairs that a push makes (0x05, 0x10, 0x30) and ring rooms (0x55, 0x07, 0x61, 0x20, 0x03): the stand square, the one direction, the stairs | §3.1; game.yaml has `push_block`, `stairs`, `trigger: push-block-stairs` | data present; 0x30's stairs on a Trap's home need the lure of §3.4 or one touch |
| One-way shutters 0x41 S, 0x31 S | §3.2 | game.yaml `shutter: true` on the exit; nothing says "never" |
| A room entered several times | 0x38, 0x48 from the same side twice (`0_38w_v2`, `0_48n_v2`); 0x16, 0x17, 0x27, 0x26, 0x07, 0x10, 0x20, 0x61 from two sides: no name clash | built |
| **Kept kills for return visits**: 0x07 and 0x10 out of their cellars, 0x20 from N, 0x61 out of the tunnel | §3.4 against the scaffold's episodes: 0x20 refilled (0/40, 40 deaths), 0x61's Patra back (−4.25 hearts) | the env's `fresh_rooms` write clears a start room's kills; `room_kills` stages kills for training only |
| **The route's progress in every state** | finding 9: a state with a piece the route does not have there is off the route, silently | `StepProgress::matches` (pieces, sword, bow, raft, ladder, recorder); the ring, the arrow grade and containers are not in it |
| The end of the route at Ganon's door, Ganon, Zelda | not this walk | – |

**Masks**

| Mask | Evidence | Covered by |
|---|---|---|
| MOVE into 0x63 N and S (key doors) | §3.2: the last key on the first frame | `keep_out: true` (round 27) fits: add it to game.yaml's two exits |
| BOMB outside the step's wall | 5 written bombs with none to spare; goriya placed 0.4-1.4 bombs a run in wall rooms | `required` + `bomb_reserve` (the scaffold held 8) |
| BOMB in overworld 0x07, MOVE into its opened cave | §1.4: −20 rupees | none for an overworld cave mouth; ow-lynel is `tenth` |
| The left ware of 0x47 | §1.3 | none |
| SWORD at 0x06's man | §2 | none |
| ARROW before Ganon | §4: a rupee a shot | `items.arrows` off, `rupee_reserve` |
| CANDLE only under 0x47's tree | Link's own fire ½ heart | `items.candle` per family |

**Families for the crossed rooms** (zero-shot results in §1.2 and §5)

| Rooms | Tried | Result |
|---|---|---|
| 0x37; 0x38 ×2, 0x28, 0x27; 0x07 | ow-octorok; ow-zora; ow-lynel | play today (36-40 of 40) |
| 0x6d | ow-moblin | 0/40: does not play the recorder. 0x6d is ow-moblin's and 0x28, 0x27 ow-zora's by `d2first`'s own steps (a room has one family: a scaffold that gives 0x6d to ow-leever, which plays 0x0b's whirlwind, or 0x28 and 0x27 to ow-peahat does not load) |
| 0x48 ×2; 0x17; 0x06, 0x05 | ow-leever; ow-boulder; ow-lynel | 0-12 of 40: new exits, the ladder, Lynels, the wall |
| 9:0x76, 0x66, 0x26 ×2, 0x06, 0x62; 0x63; the six tunnel passages | stalfos-keese-gel; zol; equipment-room | play today (0x26's wall 34/40) |
| 9:0x65, 0x15, 0x17 ×2, 0x31, 0x04 (Wizzrobe crossings, four with a bomb wait) | goriya; like-like (20 a room) | goriya 8, 0, 0, 11, 0, 28 of 40; like-like 16 (−10.0 hearts), 0, 4, 0, 5, 20 (−3.3, the ring) of 20: no family crosses them; one that crosses and waits out a fuse is new work |
| 9:0x16 W → S, 0x27 ×2 (Patra crossings) | goriya; like-like | goriya 27, 15, 40 of 40 (the last with the ring); like-like 2, 0, 20 of 20 |
| 9:0x14, 0x51, 0x41 (Like-Like rooms), 0x03 | like-like, trap-room (0x41); zol, like-like (0x03) | 0x14 0/20, 0x51 0/20, 0x41 2/20 and trap-room 14/40, 0x03 0/20 twice: `stuck`, see below |

**Does a route this long strain anything?**

- **The `stuck` rule against a Like-Like's hold** (`EndCondition::Timeout`, `position_timeout: 50`): a held Link
  cannot move, and 50 actions in one place end the episode. The Like-Like rooms ended so (finding 6). The script
  frees itself with stabs (d6-walks.md: 77 frames); the rule or the families must change before 0x14 can be played.
- **Step budgets.** The longest scripted crossings are 0x14 (1044-1050 frames) and 0x48 with the guard (856); a
  family's `max_steps` 2000 holds them.
- **The six-room history**: not strained (§3.4).
- **State names**: 55 steps, two `_v2` rooms and one `_v2` cellar; `entry: recorder` (`0_37r`) and cellar interiors
  (`9_60cave`) resolve.
- **Chain health.** Every model number here starts at 15 full hearts; a chain arrives in level 9 with what the leg
  left (the script: about 8 of 15).
- **No observation or action-space change was found necessary** for the route part: schema 6 holds `paralyzed`,
  `halted`, `ring_red`, `arrow_silver` and the doorway columns. Whether it holds the Bubble's sword block (`$4C`)
  was not checked here.

## The recommended route

The draft's way (§3.1), as steps. Entry and exit as routes.yaml would say them; bombs and keys after the step on a
no-drop run (**w**: the wall's bomb is a cheat unless a drop paid it).

| Step | Room | Entry → exit | What is done | Bombs | Keys |
|---|---|---|---|---|---|
| 1 | 0:0x6d | cave → recorder | one play, facing as he comes out | 0 | 3 |
| 2 | 0:0x37 | recorder → E | | 0 | 3 |
| 3 | 0:0x38 | W → S | | 0 | 3 |
| 4 | 0:0x48 | N → W | | 0 | 3 |
| 5 | 0:0x47 | E → cave | burn the tree from (`$B0`,`$9D`) facing S | 0 | 3 |
| 6 | 0:0x47 cave | S → S | `waypoint: heart`, the right ware only (15 containers) | 0 | 3 |
| 7 | 0:0x47 | cave → E | | 0 | 3 |
| 8 | 0:0x48 | W → N | | 0 | 3 |
| 9 | 0:0x38 | S → N | | 0 | 3 |
| 10 | 0:0x28 | S → W | | 0 | 3 |
| 11 | 0:0x27 | E → N | N at X `$70`-`$A0` | 0 | 3 |
| 12 | 0:0x17 | S → N | the ladder W over the river, N at X `$40` | 0 | 3 |
| 13 | 0:0x07 | S → W | no bomb, not into the cave | 0 | 3 |
| 14 | 0:0x06 | E → W | | 0 | 3 |
| 15 | 0:0x05 | E → cave | the rock wall from (`$50`,`$AD`) facing N | 0 (**w**) | 3 |
| 16-56 | level 9 | §3.1 #1-#41 | as walked | 0 (**w** ×4, +4 at 0x16) | 1 |

Changes to the draft: none in the rooms or their order. In the plays: 0x16's first crossing by the bottom lane;
0x14 fought with the sword, not walked; 0x04 by the top row between X `$90` and `$70`. Whether the ring detour
(steps #9-#17 of §3.1) stays is the owner's standing decision; this walk's numbers say it costs 1.4 hearts on the
crossings and three bombs, and is paid back only in the fights.

## Not walked, and why

- The big fights: 0x55, 0x07, 0x05, 0x20, 0x10, 0x30, the Patras of 0x16, 0x61, 0x52, Ganon, 0x32 (the other
  walk). Here they were staged empty.
- **0x03's fight** (§5.3): no script written; the one model ended `stuck`.
- 0x30's Trap lure with its Wizzrobes alive; the block clips of the ring rooms.
- Other lanes through overworld 0x06 and 0x05; a missed whirlwind with the Moblins live (it never missed).
- The whirlwind and 0x47 under a family model: the burn has no objective; ow-leever (which plays 0x0b's whirlwind)
  could not be given 0x6d (§7).
- Why ow-boulder leaves 0x17 by a wrong exit in 115 frames (whether the objective's path knows the ladder) and
  whether BOMB is offered under 0x05's wall: only the results were read.
- The fall-back ways (0x56's key, 0x23's bombs, 0x50/0x40) and the bomb recovery loop north of 0x31.
- Chain arrivals: every trial starts a room at full hearts.

## Questions only the owner can answer

1. **The red ring.** On the route it is a net loss of 1.4 hearts and three bombs; it pays through the fights only.
   Keep it whatever the fights' walk says, or let that walk's numbers decide?
2. **The five bombs.** A no-drop run cheats five (two without the ring). Is a full bag at Dungeon 8's exit a goal of
   the earlier rounds (8:0x3f's four, drops), or do counted cheats stand?
3. **0x14.** Killing its five Like-Likes with beams from full hearts costs 0.05 hearts and no time; without full
   hearts it is stabs at 40 px. May the route ask for it (a `clear`), or is it left to the family?
4. **The `stuck` rule** in Like-Like rooms: a longer limit while `paralyzed`, or none?
5. **0x47's potion.** Mask the left ware, or let taking it end the run as `missed`?
6. **The survival gap.** The route part costs 23 hearts of 15 by script. Is the answer better play in five rooms
   (training), a potion after all, or the game's CONTINUE (question 5 of `rest-of-game.md`)?
