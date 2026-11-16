# Round 27 walks: the leg from Dungeon 7, the magic sword and Dungeon 6 in the running game (Nov 16)

`rest-of-game.md` "Gap list" (round 27) and `d6.md` §12, answered by playing the running game before anything of
round 27 is built. Every number here was observed (**walked**); how each item was staged, what was written to RAM and
the trials are in its section. Where chance is involved a number is from at least 20 trials, each loaded with the
game's own randomness for its seed. Scripts (`scripts/experiments/`): `r27_walks.py` (helpers, staging),
`r27_leg_walks.py` (items 1-3), `r27_d6_walks.py` (item 4), `r27_wizzrobe.py` (item 5), `r27_gohma.py` (item 6),
`r27_walk_scaffold.py` (a throwaway data root with a scouting route `d6walk` so today's models can be played on the
stretch), and round 26's `r26_replay_damage.py` (who hurt Link, from replays).

Link on these walks is what a run brings out of Dungeon 7: 12 heart containers (full: the D7 piece refills), the white
sword until 0x21's cave and the magic sword after it, the red candle, the bow and bought arrows, recorder, ladder,
raft, boomerang, the food used, 2 keys, 2 bombs, 31 rupees (the median of 28-40, `d7-walks.md` §1), pieces 1-5 and 7.

## The answers

| # | Item | Answer (one line) | Corrects |
|---|---|---|---|
| 1 | The leg, clip way | A straight scripted walk loses **2.1 hearts** from 0x42 to 0x22's mouth without the sword stop (0x42 0.62, 0x52 0.20, 0x51's wrap 0, 0x50 0, graveyard 0, 0x31 back 0.50, 0x32 0.70, 0x22 0.10); the plain way +1.1 (0x51 0.40, Lost Woods 0.72). **ow-moblin crosses its rooms and the graveyard at 100%; ow-lynel dies in 0x50 (37 of 40) and 0x32 (29 of 40)** where the script loses 0 and 0.7 | `d6.md` §1.1 (costs were inferred), `rest-of-game.md` survival table "not measured" |
| 1 | Whirlwind | With the index as the route leaves it (`$FB`) and the D7 piece: one play facing down lands at 0x74, two at 0x3c, one facing up at 0x0b, two at 0x42; with the D6 piece too: two down 0x3c (the draft's next leg), two up 0x22 | confirms `late-mechanics.md` §2.9 (arithmetic → walked) |
| 2 | 0x21's stone | Push **from below** ((`$90`,`$9D`), Up) and stop on the stairs square: **40 of 40, 0.07 hearts, no flying Ghini**. The stone moves on the first frame, slides 32 frames, the stairs are at (`$90`,`$8D`). Walking on into the moved stone releases a flying Ghini (1 heart, 30 of 30). **The stairs are not kept: out of the cave the stone is back** | `d6.md` §1.4 ("flag `$20` ... the stairs are back on every later layout"), §8, §9 #8; `late-mechanics.md` §2.4 (inferred → walked) |
| 2 | Cave and sword | Sword at (`$78`,`$9D`): Up 5 frames from (`$78`,`$A5`); flag `$30`, held 128 frames. At 11 containers nothing is taken, the person stays (flag `$20`). Out at (`$50`,`$6D`) facing down; **the room's Ghini is made anew at (`$50`,`$BD`), under the exit, every time (29 of 29)**: leaving down column `$50` costs a heart, down column `$20` 0.07-0.68 in all | `d6.md` §1.5 ("column c5 is clear down to the S edge") |
| 3 | 0x22 | X `$70`, `$78`, `$80` all enter level 6 (11-12 frames of Up from Y `$8D`); `$88` does not. Slow tiles `$74`/`$75`: 0.77 px a frame (0x50's corridor and 0x32's steps 95 frames each) | confirms `d6.md` §1.1, §1.6 (code → walked) |
| 4 | D6 crossings | Scripted straight walks, 40 of 40 each: 0x79 0, **0x78 0**, 0x68 1.35, 0x48 0, **0x28 with the bomb 0.50**, 0x29 1.62, tunnel 0.56, 0x1d 0, 0x2c 2.55 hearts. Today's models: 0x78 and 0x2c 100% (goriya), 0x29 95-100% (goriya, zol), **0x28 0-25% with every family** | `d6.md` §2.3 (inferred; 0x2c "most dangerous" stands, 0x78 costs nothing) |
| 4 | D6 forced fights | 0x58 100% (stalfos-keese-gel, −1.8 hearts), 0x39 100% (vire, −0.85), 0x2d 97.5% (vire, −1.75); **0x38: best 20% (like-like, −11.2 hearts; planner 40%, −6.1); 0x3a: best 50% (like-like, −9.05; planner 30%)**; goriya, zol, trap-room 0% in both | `d6.md` §10 #1 (now with numbers) |
| 4 | Doors, blocks, tunnel | Keys 2 → 1 (0x79 W) → 2 (0x58) → 3 (0x2d) → 2 (0x2c N); one press into 0x29's N door spends a key at once. Blocks of 0x38, 0x3a, 0x09 move only with every counting monster dead (Bubble alone: moves), 15-16 frames held, from any side; 0x38's push opens **both** shutters (N 55 frames, S 64 frames after the push starts). The tunnel's left ladder leads back into 0x3a (block back), the right one to 0x1d (`$60`,`$9D`) | game.yaml 0x38 `push_block_opens: N` (names one); `d6.md` §2.2 (0x29 entry is (`$20`,`$8D`), 0x2d (`$78`,`$4D`)) |
| 5 | Red Wizzrobe | Cycle 208 frames (1 choosing, 63 fading in, 64 solid, 16 fading out, 64 hidden); shot 79 frames after the spot, 3 px a frame, **4 hearts** (160 of 160), small shield no help. **In a doorway no shot reaches Link (0 of 320); at X ≡ 8 (mod 16) vertical shots miss (0 of 42).** While hidden RAM holds the spot it left: the next spot does not exist until the frame it starts to fade in | `d6.md` §4.2 (code → walked; the doorway is new) |
| 5 | Blue Wizzrobe, kills | 0.5 px a frame, 32-frame diagonal teleports, shoots only on `FrameCounter` & `$1F` = 0 along Link's row; shot 2, touch 2 **also while it flashes (20 of 20)**. Red: 2 white / 1 magic / 1 beam / 1 bomb; Blue: 5 / 3 / 3 / bombs unreliable (4 of 20 alive after 12); boomerang and arrows nothing | `d6.md` §4.4 (walked) |
| 5 | Like-Like, Bubble | Catch 1 heart; held, nothing hurts Link (0 hearts in 600 frames with two Red and two Blue Wizzrobes alive, 20 of 20); only its death frees him (7 A presses, 77 frames). Bubble: `$4C` = `$10`, sword off 160 frames, 0 hearts | confirms `d6.md` §4.5, §5.1, §5.2, §12 #11 |
| 6 | Gohma | Eye fully open frames 19-114 of the room, next opening 386-510 frames later; an arrow kills **iff** the eye state is 3 when it arrives and Link's X − Gohma's X is in [−10, +5] (1746 of 1746 inside, 0 outside); fireball every 65 frames, 1 heart. **Script: 40 of 40 kills, heart container and piece, 1.45 hearts, 75 frames to the kill, 1-3 rupees.** Five boss families: 0 of 20 each, also with ARROW on (0-14 arrows a run) | `d6.md` §6 (code → walked) |
| 7 | Ledgers | Bombs 2 in, 1 spent (0x28), 1 out; keys 2 in, 2 out; rupees 31 in, 1-3 for Gohma. **Hearts: the best plays seen sum to 10.3 of 12 before 0x38 and 0x3a are counted, and those two cost 4.75 and 6.1 in their best wins** (table in §7) | `d6.md` §0 (rupees 2-14 → 28-40) |
| 8 | Engine | No missing model input found: schema 6 carries every column these monsters need. Missing: the gravestone push and the sword cave as route steps, Gohma's arrow rule, wrong-door masks, two new families and a retrained ow-lynel (§8) | — |

**The ten findings that most change round 27's plan**

1. **0x38 and 0x3a are not played by anything today.** The best zero-shot family (like-like) clears 0x38 in 4 of 20
   and loses 11.2 hearts on average; with the planner 4 of 10 and 6.1. 0x3a: 10 of 20 and 9.05. Blue Wizzrobes do
   most of the damage (touch and shot, 2 hearts each; 149 of 212 hearts in 0x38), not the Red ones' 4-heart shots. No
   scripted fight was written for these rooms: the only real plays are the models'.
2. **The survival budget does not close.** A scripted or best-model play of everything but 0x38 and 0x3a costs about
   10.3 hearts from Dungeon 7's piece to Dungeon 6's (leg 2.2, Dungeon 6's other rooms 8.2), of 12, with drops giving
   back under 0.3 hearts a room. The two block rooms then need to be won nearly without a hit.
3. **A doorway is out of every shot's reach.** Red Wizzrobe shots (0 of 320) and Gohma's fireballs (0 in 150 frames ×
   20) end at the room's bound; Link standing in the door he came in by is not hit. 0x78 is crossed with no damage
   by walking off the door row at once (40 of 40); waiting in the doorway first changes nothing there and costs more
   in 0x2c (traps).
4. **0x28 kills every model and costs a script half a heart.** Bottom row to X `$D0`, up to the door spot, one bomb
   facing E, through: 40 of 40, 0.50 hearts, 467 frames. goriya, like-like, vire, zol 0%, darknut 25%, trap-room
   22.5% from 12 hearts: they stay in the room (median death at frame 526-664) under three Blue Wizzrobes.
5. **Gohma is a script-sized problem.** Up to Y `$AD`, track its X, fire when the eye state is 3: 40 of 40 inside the
   first opening (75 frames), 1-3 rupees. What a model lacks is not skill but the rule: no objective aims an arrow,
   and a model that fires freely spent 0-14 rupees a run without a kill.
6. **The gravestone is cheap if pushed from below and Link stops on the stairs.** 40 of 40 at 0.07 hearts. The stairs
   are gone again after the cave, so a refusal (11 containers) or a second visit pushes again; and the cave's exit
   spot has the room's Ghini made directly under it.
7. **Today's ow-lynel cannot play its three new crossings.** 0x50 from the wrap (E → N): 0 of 40, 37 deaths to Red
   Lynels, where a straight walk up the corridor lost nothing in 30; 0x32: 7 of 40; 0x22: 35 of 40 with 41 hearts
   lost to Armos it woke. ow-moblin is the opposite: 100% on 0x42, 0x52, 0x51 (both ways), the Lost Woods and, tried
   zero-shot, all four graveyard crossings.
8. **Hidden positions: for a Red Wizzrobe `ram` and `last_seen` are the same picture.** RAM holds the spot it left
   through the 64 hidden frames; the next spot is drawn on the first frame of the next cycle, which is the frame it
   starts to fade in. RAM gives no advance knowledge of where it will appear; what it gives over the screen is the
   exact phase counter.
9. **The magic sword matters most in exactly the two hard rooms.** With the white sword like-like's 0x38 falls from
   20% to 5% and 0x3a from 50% to 0%; Vires split (0x39 −0.85 → −1.34 hearts, 0x2d −1.75 → −3.48).
10. **Bombs and arrows leak.** vire spent about one bomb in each of 0x39 and 0x2d (both of the dungeon's bombs are
    spoken for); darknut 1-2.5 in the Wizzrobe rooms. Dungeon 6 needs a bomb mode per family and the ARROW rule
    before any training.

## How the walks were staged

- **The leg.** `triforce/probe.py` puts Link into 7:0x2b from W with the inventory above at 5.5 hearts and pieces 1-5;
  he walks round the block ring onto the piece and **the game itself** ends the level: out in 0:0x42 at (`$60`,`$5D`)
  facing down 879 frames later, 12 hearts, `InvTriforce` `$5F`, the pond full again. From there the leg is walked once
  with the monster slots emptied every frame (tile objects kept) and each room's pre-entry frame saved
  (`r27_walks.leg_states`); 0x50's wrap arrival is the pre-entry frame of the agent's own `CLIP E` out of 0x51. Every
  trial loads one of these states with `randomize_rng` and its seed: the monsters are made fresh and are live. The
  states after 0x21 hold the magic sword (`InvSword` written 3).
- **Dungeon 6.** `triforce/probe.py` from `start.state` (`r27_walks.stage_d6` / `d6_state`): the inventory with the
  magic sword, 12 full hearts, the keys and bombs of the draft's ledger at that step, the `WorldFlags` of the doors
  opened before (0x79 W, 0x28 E, 0x2c N) and the key flags of 0x58 and 0x2d. Dark rooms stay dark. The tunnel and
  0x1d are reached by play from 0x3a (its seven put into the death state, the block pushed, the stairs).
- **Written to RAM in single measurements** (said again where used): Link's position and facing beside a monster for
  the hit counts and the touch tests; hearts rewritten each frame where a count needs him alive; monsters "put into
  the death state" (`ObjMetastate` `$10`, what a kill sets) where a room's secret is studied, not its fight.
- **Models.** `r27_walk_scaffold.py` copies the data with the route `d6walk` (extends `d7`: the draft's steps from
  0x42 to Dungeon 6's piece, the `woods` stretch with both ways), the rooms tagged, `requires_bombs` on 0x28,
  `ram_randomization: {}` (the states' own 12 hearts, keys and bombs), start states from the staging above. One
  scaffold per candidate family for the six Wizzrobe rooms (`--wizzrobe`). Promoted runs of round 25 (converted,
  1.5.1.0), each model's act mode `temperature:0.5+relax:16`, 20 episodes a state (families with clips: 20 with the
  switch on and 20 off); the planner `+plan:16:1`, 10 a state. Arrows are off in every family (round 26's default);
  for Gohma they were turned on in a second run. All through `triforce eval-batch`.

## 1. The leg from 0x42 to 0x22 (walked)

`r27_leg_walks.py rooms --seeds 30`. A "straight walk" goes from point to point and walks back onto its line after a
knockback; it never fights. Hearts are the mean over 30 seeds (maximum in brackets). Model columns: success of 40
(20 with clips on, 20 off) and mean heart change.

| Room, entry → exit | Monsters made | Straight walk: path | Hearts lost | Frames | Family model |
|---|---|---|---|---|---|
| 0x42 cave → S | 1 Red Moblin | from (`$60`,`$5D`): W to X `$40`, down to Y `$AD`, E to `$70`, S | 0.62 (1) | 187 | ow-moblin 40/40, −0.06 |
| 0x52 N → W | 2 Blue + 2 Red Moblins (from the edges) | down to Y `$5D`, W to X `$30`, down to `$8D`, W | 0.20 (1) | 142 | ow-moblin 40/40, −0.12 |
| 0x51 E → W, the wrap | 5 Blue Moblins | `CLIP E` offered on the entry frame at (`$F0`,`$8D`): one action, 140 frames | 0 | 140 | ow-moblin 40/40 over both ways (clips on and off), −0.07 |
| 0x50 E (wrap) → N | 5 Red Lynels | lands at (`$F0`,`$8D`) inside the rock, 11 frames W to X `$E0`, up the corridor | 0 | 198 | **ow-lynel 0/40 over both entries, 37 deaths** |
| 0x51 E → S (plain) | 5 Blue Moblins | W to X `$A0`, down | 0.40 (0.5) | 153 | (above) |
| 0x61 Lost Woods: N, W, S, W | 5 Blue Moblins each time | N at once; up the lane and W; to X `$A0` and S; W | 0, 0.22, 0.23, 0.27 | 1, 188, 119, 198 | ow-moblin 20/20, −0.57 (the first turn only) |
| 0x60 E → N (plain) | 2 Peahats, 2 Red, 2 Blue Lynels | W to X `$D0`, up | 0 | 84 | ow-lynel 5/20 (15 wrong exits) |
| 0x50 S → N (plain) | 5 Red Lynels | up to Y `$8D`, E to X `$E0`, up the corridor | 0 | 171 | (above) |
| 0x40 S → E | 1 Ghini | up the slow steps to Y `$BD`, E | 0 | 49 | ow-moblin 40/40, 0 |
| 0x41 W → N | 1 Ghini | E to X `$20`, N | 0 | 120 | ow-moblin 40/40, −0.28 |
| 0x31 S → N | 1 Ghini | straight up X `$20` | 0 | 120 | ow-moblin 80/80 over both entries, −0.44 |
| 0x21 | 1 Ghini | §2 | 0.07 | 893 | no objective (§8) |
| 0x31 N → E | 1 Ghini | from (`$50`,`$3D`) down to Y `$7D`, E (row `$8D` has stones) | 0.50 (1) | 174 | (above) |
| 0x32 W → N | 2 Peahats, 2 Red, 2 Blue Lynels | E to X `$10`, down to Y `$8D`, E to `$70`, up the steps | 0.70 (4) | 197 | **ow-lynel 7/40, 29 deaths** |
| 0x22 S → cave | 1 Red Lynel, Armos statues | straight up X `$70` | 0.10 (2) | 85 | ow-lynel 35/40, −1.95 |

- **Sums.** Clip way with the sword stop: 0.62 + 0.20 + 0 + 0 + 0 + 0 + 0 + 0.07 + 0.50 + 0.70 + 0.10 = **2.2 hearts**
  (the walk never dodges; the worst single crossing lost 4 in 0x32). Plain way: + 0.40 + 0.72 = 3.3.
- **0x42.** Link comes out of Dungeon 7 above the pond, which is full again; the Red Moblin is made on the room's W
  side, on the walk's way round the pond (all 18.5 hearts the 30 walks lost).
- **0x51's wrap.** `CLIP E` is in the mask on Link's first frame; 30 of 30 arrive in 0x50 with one action. He lands at
  X `$F0` inside the E rock and walks W out of it.
- **Slow tiles** (`r27_leg_walks.py entrance`): on tiles `$74`/`$75` Link moves 0.77 px a frame (73 px in 95 frames
  in 0x50's corridor and on 0x32's steps; 22 px in 29 frames on 0x22's and 0x40's two squares), 1.5 elsewhere.
  `mechanics.md`'s "tile identity unverified" is settled: the tiles are `$74`/`$75`.
- **The Lost Woods by buttons** work as the draft says (30 of 30 each turn: the room comes back until the fourth exit
  leads to 0x60). Under the engine's maze only the first turn was played by a model (the scaffold has one state).
- **ow-lynel.** In 0x50 the model was trained to leave S; sent N it wanders among five Red Lynels and dies (35 of 37
  deaths by a Lynel's touch; median death at frame 879). In 0x32 Blue and Red Lynels kill it in 29 of 40. In 0x22 it
  woke Armos statues (41 hearts to Armos over 40 runs, 3 deaths). With the planner: 0x50 0 of 20, 0x32 6 of 20,
  0x22 20 of 20 (−0.15).
- **The graveyard by other families** (zero-shot, both 0x31 entries together): ow-moblin 100% everywhere, ow-zora
  100%, ow-octorok 95-100%, ow-peahat 0x31 50%, ow-tektite 0x31 21%, ow-leever 0x40 0% (wrong exits, no hearts lost
  except ow-leever's 33 deaths in 0x41, whose cause was not read). The Ghini is slow and far from these paths.
- **Secrets the objective does not know on the leg:** 0x21's stone (§2); 0x51's tree (10 rupees, not on the route);
  the gravestones (§2.2); 0x22's Armos.

**The whirlwind's destination** (`r27_leg_walks.py whirlwind`; 0x52, monsters removed, `TeleportingLevelIndex` written
`$FB` = what two plays facing down at 0x0b leave, `d7-walks.md` §2; 0x42's play does not step it):

| Pieces held | Facing | One play | Two plays |
|---|---|---|---|
| 1-5 and 7 (after Dungeon 7) | down | `$FA` → 0x74 (L3) | `$F9` → 0x3c (L2) |
| | up | `$FC` → 0x0b (L5) | `$FE` → 0x42 (L7; L6 skipped) |
| 1-7 (after Dungeon 6) | down | 0x74 | **0x3c** (the draft's leg to Dungeon 8) |
| | up | 0x0b | `$FD` → 0x22 (L6) |

The leg to the sword uses no whirlwind; the count matters for the next leg, and it is as `late-mechanics.md` §2.9
computed.

## 2. The gravestone of 0x21, the cave, the sword (walked)

`r27_leg_walks.py sword --seeds 40`, `gravestone --seeds 20`. 0x21 from its S entry at (`$20`,`$DD`), Ghini live.

### 2.1 The push

- The stone is the tile object `$65` in slot `$0B` at (`$90`,`$90`). **From above**: Link at (`$90`,`$7D`), Down. **From
  below**: (`$90`,`$9D`), Up. The slide starts on the first frame of the press (no hold), 1 px every second frame for
  32 frames, Link following behind it at half speed. On frame 31-32 the object is gone, tiles (18,10)-(19,11) are the
  stairs `$70 $72 / $71 $73`, room flag `$20` is set, and Link stands on the stairs square (`$90`,`$8D`): the cave
  opens 2 frames later (34 frames from the first press to the mode change when nothing interferes).
- **The flying Ghini.** The stone's new square (Y `$A0` from above, `$80` from below) is a gravestone square. Pressing
  on into it releases a flying Ghini there (1-7 frames after the stairs appear). From above Link is already on the
  stairs when it appears and is never touched by it. From below, holding Up past the stairs walks Link into it: 1
  heart, 30 of 30, and the knockback is what puts him on the stairs. **Stopping on the stairs square from below: no
  flying Ghini at all (40 of 40).**
- A touch from the room's Ghini during the slide can throw Link past the stairs; the script then walks back onto
  (`$90`,`$8D`) round the moved stone (row `$8D` is free at X `$A0`), never into it.

| Push | Sword taken and out S | Hearts lost (mean, max) | Lost before the cave | Flying Ghinis at the stairs |
|---|---|---|---|---|
| from below, stop on the stairs, out by column `$20` | 40/40 | 0.07, 1 | 0 in 37, 1 in 3 | 0 in 40 |
| from above, out by column `$20` | 40/40 | 0.68, 4 | 0 in 26, 1-4 in 14 | 1 in 35, 0 in 5 (never touched) |
| from above, out by column `$50` | 40/40 | 1.65, 5 | as above | as above |
| from below, walking on Up (first version of the script) | 30/30 | 1.07, 2 | 1 in 28, 2 in 2 | 1 in 30 |

The walk to the stand square from below is 130 frames (up X `$20` to Y `$9D`, E along row `$9D`), from above 154
(row `$7D`, longer under the Ghini: all of "from above"'s extra hearts are its touches on the way).

### 2.2 Ghinis and the other stones

- One Ghini (`$21`) per graveyard room, made at one of six spots on entry from S ((`$50`,`$5D`) 9 of 30,
  (`$A0`,`$5D`) 5, (`$60`,`$7D`) 5, (`$A0`,`$9D`) 4, (`$90`,`$7D`) 4, (`$80`,`$8D`) 3), in a cloud for 32 frames, then
  drifting toward Link (code: 0.5 px a frame). It dies to **5 white or 3 magic** swings (20 of 20 each; Link written
  beside it, hearts not full so no beam). Touch 1 heart.
- **A scenery stone held into** (Link at (`$30`,`$7D`) or (`$60`,`$7D`), Down held 600 frames): a flying Ghini (`$22`)
  on the first frame and one more **every 72 frames**, nine in all (every free slot), at the stone's square, then
  flying off. They were not attacked here (code: immune; they die with the room's Ghini).
- The 13 scenery stones stand at X `$30`, `$60`, `$90`, `$C0` × Y `$6D`, `$8D`, `$AD`; rows `$5D`, `$7D`, `$9D`, `$BD`
  and the columns between are free in all four rooms.

### 2.3 The cave

- In: mode `$0B` (not 5), Link at (`$70`,`$D5`), the person `$6D` at (`$78`,`$80`). Link is free 33 frames after the
  warp; the walk Up to Y `$A5`, Right to X `$78`, Up is interrupted by the text (halted) and reaches the sword 287
  frames after he was first free.
- **Taken** at (`$78`,`$9D`), 5 frames of Up from (`$78`,`$A5`): `InvSword` 3, room flag `$20` → `$30`, Link holds it up
  128 frames, the person is gone. X `$70` and `$80` take nothing.
- **At 11 containers** (`HeartValues` written `$AA`): nothing at X `$70`, `$78` or `$80`, `InvSword` stays 2, flag stays
  `$20`, the person stays. Not irreversible; the engine's `SWORD_UPGRADES` asks 12 since round 26
  (`crates/zelda/src/objectives.rs`), which is the game's number.
- **Out**: 206 frames after the sword, at **(`$50`,`$6D`) facing down** (the room's exit spot, data). **The room is
  laid out anew: the stone `$65` is back at (`$90`,`$90`), the stairs are gone** (flag `$30` or `$20` set all the
  same), and the room's Ghini is made at (`$50`,`$BD`) on every exit (29 of 29), on the column under Link. Down to
  row `$7D`, W to X `$20`, down and out S avoids it.
- The whole stop, from 0x21's S edge back to 0x31: 893 frames (median).

## 3. To Dungeon 6's entrance and in (walked)

0x31 from N, 0x32, 0x22: in §1's table. 0x22's mouth (`r27_leg_walks.py entrance`): from (X,`$8D`) holding Up, X
`$70`, `$78` and `$80` enter after 12, 11 and 11 frames (at Y `$7D`), X `$88` walks on to Y `$85` and does not. Link
plays in 6:0x79 at (`$78`,`$CD`) 182-184 frames later with what he carried; no monster there.

## 4. Dungeon 6, every room of the draft's way (walked)

`r27_d6_walks.py`. Monsters made in every room match `level-6.md`'s rooms table (0x78 5 Red Wizzrobes; 0x68 5 Zols;
0x58 8 blue Keese; 0x48 6 Traps; 0x38 and 0x3a 1 Bubble, 3 Like-Likes, 2 Blue, 2 Red; 0x28 and 0x29 3 Blue, 2 Red;
0x39 5 Vires; 0x08 4 blue Keese; 0x1d 2 Bubbles, 2 Zols, 2 Like-Likes; 0x2d 2 Bubbles, 3 Vires; 0x2c 4 Traps, 2 Blue,
2 Red; 0x1c Gohma; off the way 0x18 Gleeok, 0x3c 3 Blue 2 Red, 0x7a 5 Red, 0x09 3 Blue 2 Red). Wizzrobes have no
spawn cloud: on the room's first frames their `ObjMetastate` is 1, the Reds choose their first spot on frame 2-3.

### 4.1 Room by room

Script: `crossings --seeds 40`, magic sword, 12 hearts, the path named, no fighting. Model: the best zero-shot family
of those tried (goriya, darknut, like-like, vire, zol, trap-room in the Wizzrobe rooms), success of 20 or 40, mean
heart change, how it fails.

| # | Room, entry → exit | Link first at | Script: path | Script result | Best model zero-shot | How the models fail |
|---|---|---|---|---|---|---|
| 1 | 0x79 S → W key door | (`$78`,`$CD`) | up to Y `$9D`, W to X `$20`, up to `$8D`, Left | 40/40, 0 hearts; 115 frames to the door, 214 of Left until he plays in 0x78; keys 2 → 1 | not run (no monsters) | — |
| 2 | 0x78 E → N | (`$E0`,`$8D`) in the doorway | W to `$D0`, up to `$7D`, W to `$78`, Up | **40/40, 0 hearts**, 280 frames | goriya 20/20, −2.5; darknut 40/40, −1.2; like-like 20/20, −0.5 | zol 0/20 (all die), trap-room 25/40 |
| 3 | 0x68 S → N | (`$78`,`$CD`) | straight up X `$78` | 40/40, 1.35 (max 4): Zols on the line | zol 20/20, −0.25 | — |
| 4 | 0x58 S → N, fight, key | (`$78`,`$CD`) | not scripted | — | stalfos-keese-gel 40/40, −1.82 (Keese 42, statue fire 30.5 hearts over 40); 823 frames; planner 20/20, −0.97 | — |
| 5 | 0x48 S → N | (`$78`,`$CD`) | hold Up at X `$78` | 40/40, 0 hearts, 260 frames | trap-room 40/40, −0.03 | — |
| 6 | **0x38 S → N, fight, push** | (`$78`,`$BD`), shutter shut behind within 40 frames | not scripted | — | like-like 4/20, −11.2 (wins lost 10.0, 2976 frames); planner 4/10, −6.1; darknut 4/40 | death 14 (nearest object at the last hit: Blue Wizzrobe 7, Bubble 3, Red Wizzrobe 2, other 2), stuck 2; goriya 0/20 (17 stuck), zol 0/20, trap-room 0/40, vire 3/40 |
| 7 | 0x28 S → E, bomb | (`$78`,`$BD`) | E along Y `$BD` to X `$D0`, up to (`$D0`,`$8D`), Right 2 frames, B (bomb), wait, Right | **40/40, 0.50 hearts** (Red touch), 467 frames, 1 bomb | darknut 10/40, −10.6; trap-room 9/40 | **goriya, like-like, vire, zol 0%**: they stay and die (Blue Wizzrobe 11-17 of 20 deaths, Red shot 2-5) |
| 8 | 0x29 W → S, dark | (`$20`,`$8D`) on the floor (through the hole) | down X `$20` to Y `$BD`, E to `$78`, Down | 40/40, 1.62 (max 5): Blue 40, Red 21, Blue shot 4 hearts over 40 | zol 20/20, −2.15; goriya 19/20, −2.0 | like-like 0/20 (all die) |
| 9 | 0x39 N → E, fight, dark | (`$78`,`$5D`), shut behind | not scripted | — | vire 40/40, −0.85, 1152 frames, **1.1 bombs a run**; planner 20/20, 0 hearts | — |
| 10 | **0x3a W → stairs, fight, push** | (`$10`,`$8D`) in the doorway | not scripted | — | like-like 10/20, −9.05 (wins lost 6.1, 2571 frames); planner 3/10; darknut 6/40 | death 10 (Blue Wizzrobe 7); goriya 0/20 (16 stuck), vire 0/40, zol 0/20 (12 wrong exits back W) |
| 11 | 0x08 tunnel | left ladder top (`$30`,`$5D`), cellar mode 9 | down to Y `$BD`, Right to X `$C0`, Up | 40/40, 0.56 (Keese), 407 frames | not run | — |
| 12 | 0x1d stairs → S | (`$60`,`$9D`) facing down | E to X `$78` on row `$9D`, Down | 40/40, 0 hearts, 257 frames | not run | — |
| 13 | 0x2d N → W, fight, key, dark | (`$78`,`$4D`) in the doorway | not scripted | — | vire 39/40, −1.75, 1176 frames, 0.9 bombs; planner 20/20, −0.25 | 1 wrong exit |
| 14 | 0x2c E → N key door | (`$E0`,`$8D`) in the doorway | W along the door row to X `$78`, up to Y `$5D`, Up | 40/40, 2.55 (max 6): Red 76, Blue 26 hearts over 40; 323 frames; keys 3 → 2 | goriya 20/20, −3.1 (Traps 50 of 62 hearts); darknut planner 20/20, −2.85 | zol 18/20 stuck; trap-room 19/40 deaths |
| 15 | 0x1c S → N, Gohma | (`$78`,`$CD`) in the doorway | §6 | 40/40, 1.45 | none: 0/20 with five boss families | §6 |
| 16 | 0x0c S → piece | (`$78`,`$CD`) | straight up | 40/40; out in 0:0x22 at (`$70`,`$7D`) 641 frames after the door, 13 of 13 hearts | not run | — |

- **0x78, 0x2c: waiting out the first volley in the doorway** (110 frames, then the same path): 0x78 still 0 hearts;
  0x2c 2.80 (max 9; the E traps reach a Link who lingers near X `$D0`: Trap 14 hearts). Going at once is better.
- **The door attributions** in the script rows are the object nearest to Link the frame before the loss; in the
  model rows (from replays) a shot that hits is gone on that frame, so part of "Blue Wizzrobe" is its shot.
- **goriya's 100% in 0x78 and 0x2c costs 2.5-3.1 hearts**, more than the blind script in 0x78 (0): the crossing
  models walk into fading Reds (touch 1) and, in 0x2c, into the traps.

### 4.2 Doors and keys (`doors`)

- **0x79 W** (key door): with 2 keys, 214 frames of Left from (`$20`,`$8D`) until Link plays in 0x78 at (`$E0`,`$8D`);
  keys 1; flags 0x79 `$22`, 0x78 `$21`. With 0 keys: 400 frames of Left, nothing. The run carries 2 keys out of
  Dungeon 7, so 0x7a's key (5 Red Wizzrobes, dark, E of 0x79) is not needed; 0x7a is not shut (Link walked back out
  in 17 frames).
- **0x29 N** (off the route): Up at (`$78`,`$5D`) spends a key **on the first frame** (2 → 1) and the scroll into
  0x19 starts.
- **0x2c N**: 175 frames of Up from (`$78`,`$5D`) to playing in 0x1c at (`$78`,`$CD`); keys 3 → 2.
- **Shut-in rooms behind open doorways:** 0x28 N → 0x18: Link first at (`$78`,`$BD`), Gleeok (`$44`), all doors shut
  (120 frames of Down: nothing). 0x2c S → 0x3c: first at (`$78`,`$5D`), 3 Blue + 2 Red, shut in.
- **Keys** (`keys`): 0x58's appears at (`$80`,`$90`) and the N shutter opens 18 frames after the last Keese is put
  into the death state; taken at (`$80`,`$95`) walking up; keys 1 → 2, flag `$30`. 0x2d's at (`$80`,`$90`) on the
  three Vires alone (two Bubbles alive); taken at (`$80`,`$8D`), keys 2 → 3. Whether the model runs took the keys
  was not read from their replays (the objective targets the room item; success there is the exit).
- Shutters: 0x38's S and 0x39's N shut behind Link within 40 frames of his first frame (0x28's was not re-read);
  0x39's N and E open 27 frames after its last Vire.

### 4.3 The rooms that must be empty before their block moves (`blocks`)

0x38 (block (`$60`,`$90`)), 0x3a ((`$70`,`$90`)), and off the way 0x09 ((`$60`,`$90`)). Link written beside the
block, monsters put into the death state by kind:

| Alive | 0x38 | 0x3a | 0x09 |
|---|---|---|---|
| everything | no move in 90 frames | no move | no move |
| only the 3 Like-Likes | no move | no move | — |
| only the 2 Blue Wizzrobes | no move | no move | no move (3) |
| only the 2 Red Wizzrobes | no move | no move | no move |
| **only the Bubble** | moves after 15 frames | moves after 15 frames | — |
| nothing | moves from W, E, N and S after 15-16 frames held | the same | 16 frames |

- The hold must be unbroken: 12 frames, 2 released, then 15 more.
- **0x38**: the block slides one square; the **N shutter opens 55 frames after the push starts, the S shutter 9
  frames later** (`CurOpenedDoors` `$00` → `$08` → `$0C`). game.yaml's `push_block_opens: N` names only the route's.
- **0x3a and 0x09**: stairs tiles at tile (26,4) about 31 frames after the block starts; Link takes them from
  (`$D0`,`$6D`) with 12 frames of Up (he stands at (`$D0`,`$5D`)).

### 4.4 The tunnel 0x08 (`tunnel`)

From 0x3a's stairs Link is in the cellar (mode 9) on the left ladder; 4 blue Keese. Down the left ladder to Y `$BD`,
Right to X `$C0`, Up the right ladder: 40 of 40, 0.56 hearts, 407 frames, **0x1d at (`$60`,`$9D`) facing down**. Up
the left ladder instead (22 frames from its top, 95 to play): **0x3a at (`$60`,`$9D`), the block back at (`$70`,`$90`),
the stairs gone**, only the Bubble there (the seven stay dead while the room is in the history). 0x1d → S: to X `$78`
along row `$9D`, Down: 40 of 40, 0 hearts, into 0x2d at (`$78`,`$4D`).

### 4.5 The trap room 0x48, 0x39's water, the dark rooms

- **0x48**: Up held at X `$78`: 40 of 40, no hit, 260 frames including the scroll; trap-room 40 of 40.
- **0x39** (`ladder`; Vires put into the death state): Right along the door row from X `$78`: the ladder is laid
  after 18 frames at X `$90`, Link plays in 0x3a's doorway 96 frames after starting. Without the ladder (`InvLadder`
  0) he stops at X `$90`: no way E.
- **Dark rooms** (`dark`; 0x29, 0x39, 0x2d). For the game: a candle fire (B, the red candle) starts the fade 30
  frames after B; `IsRoomBrightening` (`$51E`) is set **31 frames, during which nothing in the room moves** (monsters
  changed on 1 of the 31), then `CandleState` (`$51F`) is 2 and the room is lit until Link leaves. In a lit room
  (0x28) nothing happens. For our observation: `dark_room` is the level block's bit (attribute E `$80`), unchanged by
  the fire; entities and tiles are read from RAM whether lit or not. The candle buys the model nothing and costs 31
  frozen frames (code: its fire hurts a Link who walks into it).

### 4.6 The forced fights with a white sword, and from 8 hearts

| Room | like-like / vire / keese, magic, 12 hearts | white sword, 12 hearts | magic, 8 hearts |
|---|---|---|---|
| 0x58 | 40/40, −1.82 | 40/40, −2.14 | 40/40, −1.77 |
| 0x38 | 4/20, −11.2 | **1/20**, −11.65 | 2/20 |
| 0x39 | 40/40, −0.85 | 40/40, −1.34 (Vires split) | 40/40, −1.10 |
| 0x3a | 10/20, −9.05 | **0/20**, all die | 0/20 |
| 0x2d | 39/40, −1.75 | 39/40, −3.48 | 34/40, −2.40 |

darknut with the white sword: 0x38 4/40, 0x3a 7/40 (it bombs: 0.9-1.7 a run). A run that skipped the magic sword
would lose about 2.6 hearts more in 0x58, 0x39 and 0x2d together and has, with today's models, no measurable chance
in 0x38 and 0x3a.

## 5. Wizzrobes, the Like-Like's hold, the Bubble's timer (walked)

`r27_wizzrobe.py`. Rooms 0x78 (Reds) and 0x28 (Blues, the Reds put away).

### 5.1 Red Wizzrobe (`red`)

- **Cycle** from `ObjState` (`$AC` + slot), read every frame: `$FF` 1 frame (the spot is chosen), `$FE`-`$C0` 63 frames
  fading in, `$BF`-`$80` 64 solid, `$4F`-`$40` 16 fading out, `$3F`-`$01` hidden, then `$00` until a spot is accepted:
  208 frames, 210 when two tries were rejected. All Reds of a room start on frames 2-3 and stay within a few frames
  of each other.
- **The shot** is made on the frame `ObjState` is `$B0`: **79 frames after the spot is chosen** (object `$59` at the
  Red's position), 3 px a frame along the line it faces, 9-24 frames to a Link 32-80 px away.
- **Where it appears**, one Red, Link kept still at (`$60`,`$8D`), 180 appearances: on his row or column only, 32,
  48, 64 or 80 px away (E 63, W 67, N 21, S 29; the spots a wall or block forbids are redrawn the next frame). From
  a doorway they are all on the one side (E doorway: W 32-80 px, and 28 of 180 "wrapped" 176-192 px away at X
  `$20`/`$30`).
- **Damage and what blocks it** (one Red, so Link's own invincibility hides nothing; he faces up):

| Link stands at | Shots from E/W | from N/S | Hearts a hit |
|---|---|---|---|
| (`$60`,`$8D`), X on a square | 118 of 118 hurt | 42 of 42 hurt (16 from N, the way he faces: the small shield does nothing) | 4.0 (160 of 160) |
| (`$68`,`$8D`), X between squares | 118 of 118 hurt | **0 of 42** | 4.0 |
| the E doorway (`$E0`,`$8D`) | **0 of 160** (they end at X `$D2`) | — | — |
| the N doorway (`$78`,`$4D`) | — | **0 of 160** | — |

  A hit knocks Link back; 12 containers, no ring: **4 hearts, confirmed**. Only the magic shield would parry (not
  ours; not tested).
- **RAM while it is invisible** (the owner's question): through the 63 hidden frames `ObjX`/`ObjY` hold the spot it
  left; on the frames at `ObjState` 0 they hold each rejected try for a frame (seen: (`$E0`,`$BD`), (`$E0`,`$3D`),
  ...); the next spot is in RAM from the frame it is chosen, which is the first frame of the fade-in, when the
  screen starts to draw it. **The position is not chosen ahead: 0 frames.** It depends on Link's position on that
  frame and one random byte.

  | `obs.hidden_positions` | What the model reads for a hidden Red | What it can do with it |
  |---|---|---|
  | `ram` (today) | the spot it left (stale), `visible` 0, `phase`/`phase_time` counting down to the next shot | know when the next appearance comes (the counter) and that it will be on its own row or column then; nothing about where |
  | `last_seen` | the same spot (it was visible there last) | the same: for this monster the two encodings differ only on the retry frames |
  | `zero` | no position; presence and phase stay | the same timing; loses nothing useful, since the stale spot predicts nothing |

  What RAM gives that the screen does not is the exact phase (`ObjState`): a player sees the fade and can count. The
  decision that matters for Wizzrobes is therefore whether `phase`/`phase_time` stay (they are in schema 6), not the
  position encoding. (Ganon is a different case: his invisible position is live in RAM, `observation-next.md` §3.)

### 5.2 Blue Wizzrobe (`blue`, `touch`)

- Walks 1 px every second frame (0.5 px a frame), a median of 129 frames between teleports; teleports are 32 frames
  moving 1 px a frame on both axes (through blocks), 114-120 of ~330; single-frame both-axis steps are its turns
  onto the grid.
- **Shots** (object `$58`): all 319 seen were made on a frame with `FrameCounter` & `$1F` = 0. Link kept on the bottom
  row (Y `$BD`): 152-162 shots in 20 × 1500 frames, all along his row, at X `$78` as at `$70`. Kept on the door row
  (`$8D`, blocks between): 2-3 shots; the only two column shots came with Link at X `$70` (X on a square), none at
  `$78`. Shot 2 hearts.
- **Touch 2 hearts, also during its flash**: hit once with the magic sword, Link written onto it while its
  invincibility timer read 13: 2 hearts lost in 20 of 20. A still Link is found by them: 69-128 touches in 20 runs
  of 1500 frames.

### 5.3 What kills them (`kills`; Link written beside the monster while it is solid, 20 seeds each)

| | White sword | Magic sword | Magic beam (full hearts, 48 px away) | Bomb (placed 24 px away) | Boomerang, arrow |
|---|---|---|---|---|---|
| Red | 2 | **1** | 1 | 1 (20 of 20) | nothing (3 uses) |
| Blue | 5 | **3** | 3 (19 of 20; 5 once) | 3-7; 4 of 20 alive after 12 (it walks off during the fuse) | nothing |

The beam hits both. Bombs are a poor weapon against Blues and all of Dungeon 6's are spoken for.

### 5.4 The Like-Like's hold (`likelike`, 0x38)

- Caught (Link written onto it): 1 heart, `IsLinkParalyzed` (`$512`) set, 20 of 20.
- Held and doing nothing for 600 frames, alone: 0 hearts, never freed. **With the room's two Red and two Blue
  Wizzrobes and the Bubble alive: 0 hearts in 600 frames, 20 of 20** (about three Red volleys each): shots and
  touches pass a held Link.
- Freed only by its death: A every 12 frames, 7 presses, 77 frames (3 hits land; its invincibility eats the rest).
  Direction input did not move him. The magic shield it would eat is not ours.

### 5.5 The Bubble's sword-off timer (`bubble`)

Touch: 0 hearts, `SwordBlockedLongTimer` (`$4C`) = `$10`, back to 0 after exactly 160 frames (20 of 20). With the
timer running the mask offers BOMB, BOOMERANG, ARROW and no SWORD or BEAMS (read 60 frames after the touch, a
Like-Like in the room as a target).

## 6. Gohma, 6:0x1c (walked)

`r27_gohma.py`. Link in by the S key door at (`$78`,`$CD`), 31 rupees. (The assignment's "0x0b" is `level-6.md`'s boss
room `$1C`; 0x0b is an old man's room off the way.)

- **The eye** (`Gohma_ObjEyeState` `$46B` + slot): Gohma is at (`$80`,`$70`) from frame 3. The eye starts to open on
  frame 2 of the room (state 2), is **fully open (state 3) from frame 19 for 96 frames** (to 114) in 20 of 20 runs,
  state 2 until 136, then shut. Later openings start 386-510 frames after the one before (median 439, 44 seen),
  again 96 frames of state 3.
- **The arrow** (`arrow`: 20 seeds × 7 moments, Link written to every X from Gohma − 14 to + 10 on rows `$9D`, `$AD`,
  `$BD`, one arrow, the game put back after each): with eye state 3 at B and when the arrow arrives, **every arrow
  with Link's X − Gohma's X (at arrival) from −10 to +5 killed (1746 of 1746 from row `$AD`), none outside (0 of
  1254)**. At state 2
  or shut: 0 of 1500. From B to the kill: 1-5 frames from Y `$9D`, **6-11 from `$AD`**, 10-16 from `$BD`; Gohma moves
  half a pixel a frame meanwhile. One arrow kills.
- **Fireballs** (`$56`): one every 65 frames from frame 2 (380 intervals, all 65), about 1.7 px a frame once flying,
  **1 heart** (202 hits on a Link kept still at (`$78`,`$BD`)). In the S doorway none hit in 150 frames (20 seeds).
- **Touch** 2 hearts (Link written onto it).
- **Rupees** (`rupees`): each shot costs 1 (31 → 30 → 29 → 28 → 27 at the shut eye, nothing else happens). With 1
  rupee the arrow flies and kills, leaving 0. **With 0 rupees ARROW is not in the mask and B makes no arrow**; Gohma
  cannot be killed until a rupee is found (none in this room).
- **The script** (`kill --seeds 40`): up to Y `$AD`, keep Link's X 2-4 px left of Gohma's by walking on the row, a
  frame of Up to face it, B when the eye state is 3 with more than 22 frames left:

  | Result | Value |
  |---|---|
  | Gohma killed, heart container taken, out N, piece taken | **40 of 40** |
  | Frames from Link's first frame to the kill | median 75 (the first opening every time) |
  | Arrows = rupees spent | 1 in 30 runs, 2 in 1, 3 in 9 |
  | Hearts lost in the room | mean 1.45, max 4: all to fireballs, most of them already flying when Gohma died |
  | N shutter | opens 27 frames after the kill (40 of 40) |
  | Heart container | at (`$80`,`$90`), taken at (`$80`,`$95`) walking up: `HeartValues` + `$11` (13 containers, one more heart, no refill) |
  | The piece | 0x0c straight up; out in 0:0x22 at (`$70`,`$7D`), 13 of 13 hearts, `InvTriforce` `$7F` |

  The wasted arrows are fired while Link is still turning: walking, he turns up only on the 8-pixel grid
  (`level-6.md`), so a B one frame after a sideways step can fly sideways. Not separated further.
- **Models**: boss-aquamentus, boss-gleeok, boss-manhandla, boss-digdogger, boss-dodongo: 0 of 20 each, all dead
  (median 1634 frames; Gohma's touch 210 of 240 hearts: they walk into it). With ARROW turned on
  (`items: {arrows: true}`): boss-aquamentus, boss-gleeok, boss-digdogger again 0 of 20; boss-aquamentus fired 0-14
  arrows a run (mean 6) without a kill.
- The room's exits: S (the key door, open behind him), N shutter until the kill. Nothing else in the room.

## 7. Ledgers and the survival budget for Dungeon 6 (walked numbers, added up)

**Bombs.** In 2 (`d7.md` §4). 0x28's wall takes 1 (40 of 40 from the door spot). Out 1 (owed to Dungeon 8). Models
spent bombs where none is free: vire 1.1 in 0x39 and 0.9 in 0x2d, darknut 0.9-2.5 in the Wizzrobe rooms. Room bombs:
0x3c only (off the way, a shut-in fight). Red Wizzrobes and Vires can drop bombs (not counted here).

**Keys.** In 2. 0x79 W −1, 0x58 +1, 0x2d +1, 0x2c N −1: out 2, never below 1. With 0 keys carried the only way is
0x7a's key (5 Red Wizzrobes). 0x29's N door takes one on a single press.

**Rupees.** In 28-40 (median 31). Gohma: 1-3 by the script. Left for Ganon's arrow: 25 or more. On the route more
rupees come only from drops and 0x28's 5 (all five Wizzrobes dead: a fight the route does not make). A model with
ARROW on wasted up to 14 in one run: ARROW must be Gohma's alone and aimed by a rule (§8).

**Hearts.** No refill between Dungeon 7's piece and Dungeon 6's. What the way gives: drops only, and only for kills.
In the evaluations hearts dropped per room were 0.05-0.35 and hearts taken 0-0.15 (0x39 with the planner: 0.75
dropped, 0.30 taken); no fairy was seen. The straight walks kill nothing and get nothing.

| Stretch | Best play seen | Hearts lost (mean) | Left of 12 after it | Success of that play |
|---|---|---|---|---|
| Leg, clip way, 0x42 → 0x21's S edge | straight walk | 0.82 | 11.2 | 30/30 each room |
| 0x21: stone, cave, sword | script (from below) | 0.07 | 11.1 | 40/40 |
| 0x31, 0x32, 0x22 | straight walk | 1.30 | 9.8 | 30/30 each |
| 0x79, 0x78 | straight walk | 0 | 9.8 | 40/40 |
| 0x68 | zol model | 0.25 | 9.6 | 20/20 |
| 0x58 (fight, key) | stalfos-keese-gel, planner | 0.97 (1.82 without) | 8.6 | 20/20 |
| 0x48 | straight walk | 0 | 8.6 | 40/40 |
| **0x38 (fight, push)** | like-like, planner | **6.1 over all runs; 4.75 in the wins** | 3.8 in a win | **4/10** (2/20 from 8 hearts without the planner) |
| 0x28 (bomb) | script | 0.50 | 3.3 | 40/40 |
| 0x29 | straight walk | 1.62 | 1.7 | 40/40 |
| 0x39 (fight) | vire, planner | 0 (0.85 without) | 1.7 | 20/20, with a bomb it may not spend |
| **0x3a (fight, push)** | like-like | **9.05 over all runs; 6.1 in the wins** | dead | 10/20 from 12 hearts; 0/20 from 8 |
| 0x08, 0x1d | straight walk | 0.56 | — | 40/40 |
| 0x2d (fight, key) | vire, planner | 0.25 (1.75 without) | — | 20/20 |
| 0x2c | straight walk | 2.55 | — | 40/40 |
| 0x1c Gohma | script | 1.45 | — | 40/40 |
| **Sum without 0x38 and 0x3a** | | **10.3** (leg 2.2 + Dungeon 6 8.2) | 1.7 | |
| Sum with them at their best wins | | 21.2 | −9.2 | |

Read: every play above is either a blind walk or a model that never saw the room, so each number can fall with
training; but 12 hearts cover the stretch only if 0x38 and 0x3a together cost under two hearts and the crossings
lose no more than the blind walks do. The 0x38 → 0x3a pair multiplies to 4/10 × 10/20 = 20% even from full hearts
each (0x3a's states here were full, which a run's are not). Without the magic sword the pair is 1/20 × 0/20.

## 8. What the engine lacks for round 27 (evidence above; terms of `d7.md` §11 and `round26.md` §2)

**Route steps, waypoints and objective rules**

| Item | Evidence | Nearest code |
|---|---|---|
| The stretch `woods` after `d7` (0x51 wrap / 0x61 maze / 0x60 / 0x50) and steps 0x42 `entry: cave` … 0x22 `exit: cave`; Dungeon 6's 16 steps with the tunnel 0x08 (`in_cave`, `exit: E`) | the scaffold's `d6walk` loads and plays as written (`r27_walk_scaffold.py`) | routes.yaml only |
| **Gravestone push**: game.yaml `entrance: {how: push, tile: [18, 10]}` is data without a rule. Target: the stand square **below** the stone (`$90`,`$9D`), MOVE N from there, then stop: the stairs square (`$90`,`$8D`) is where Link stands; the stairs exist only until the room is left | §2.1: 40/40 at 0.07 hearts from below; the stone is back after the cave | `game_map::Secret` has `Armos`, `Recorder` only |
| **The sword cave**: an in-cave room `(0, 0x21, cave)` with `treasure: magic-sword` (middle ware, X `$78`); a waypoint done by `InvSword` 3; refused under 12 containers (the step must not be `missed` then) | §2.3 | `SWORD_UPGRADES` (12 since round 26), d2first's white-sword cave |
| Leaving 0x21 after the cave: path cost on column `$50` (the Ghini is made at (`$50`,`$BD`)) | §2.3: +1 heart | `path_costs` (round 26's shops) |
| Gravestone squares as never-press targets (a flying Ghini every 72 frames) | §2.2 | a mask or path cost on MOVE into a stone square |
| **Gohma's arrow**: a boss step whose kill needs ARROW facing N from the X window [Gohma − 10, + 5] at eye state 3; then the heart container waypoint (`waypoint: heart`), then N | §6: script 40/40; models 0/20 with or without ARROW | `GameCompletion`; `observation-next.md` §6 "arrow target masks" |
| Key waypoints in 0x58 and 0x2d (leave only with the key) | §4.2; d6.md §8 | the `waypoint` kinds have no `key` |
| 0x38: `push_block_opens` opens both shutters; 0x3a: stairs; both only after the kills (Bubble excepted) | §4.3 | built (4:0x32, 5:0x64) |
| The tunnel's left ladder is a wrong exit back into 0x3a with the block reset | §4.4 | `CELLAR_EAST_COLUMN` (round 24) |
| 0x22's entrance: X `$70`, `$78`, `$80` | §3 | game.yaml `tile: [14, 8]` only |

**Masks**

| Item | Evidence |
|---|---|
| ARROW on for boss-gohma only, offered only when the shot can kill (eye 3, X window, facing N), `rupee_reserve` kept for Ganon | §6: a free ARROW wasted up to 14 rupees a run |
| Bombs: `required` at 0x28's door spot only; `never` (or `tenth`) for the Wizzrobe, vire and Keese rooms of Dungeon 6; route `bomb_reserve` 1 after 0x28 | §7: vire and darknut spent 1-2.5 a room |
| MOVE into 0x29's N key door: off (one frame spends a key) | §4.2 |
| MOVE into 0x28's N and 0x2c's S doorways: off, or the two rooms made part of the run (shut-in fights with Gleeok and five Wizzrobes) | §4.2 |
| CANDLE off in Dungeon 6 (31 frozen frames, nothing gained) | §4.5 |

**Observation (schema 6): nothing missing was found.** The columns these walks call for exist already
(`docs/observation-next.md` §2): `touch_harms` (Blue Wizzrobe stays 1 while flashing: walked true, §5.2), `visible`,
`phase` / `phase_time` (Red Wizzrobe's `ObjState`, frames to its shot; Gohma's eye state and timers), `in_line`,
`ringleader` (0x78), `frame_phase_32` (the Blue's shot frames: walked, 319 of 319), `paralyzed`, `sword_blocked` and
`sword_blocked_time`, `arrow_usable`, `rupee_reserve`. **No second model-breaking input change is needed for this
stretch on the evidence here.** Two things a model cannot read and the walks show matter, both derivable from
positions it has: "Link stands in a doorway" (no shot reaches him) and "Link's X is between squares" (vertical Red
shots miss); neither needs a column. Not checked here: the live values of those columns in these rooms (round 26's
probes did that for the types).

**Families**

| Family | Rooms | Why |
|---|---|---|
| **wizzrobe** (new) | 6:0x78, 0x28, 0x29, 0x2c crossed; 0x38, 0x3a fought | no existing family does both; crossings are 95-100% with goriya but cost 2-3 hearts, the fights 0-50%; like-like's weights are the best start for the fights (it knows the block and the hold), goriya's for the crossings |
| **boss-gohma** (new) | 6:0x1c | arrows; nothing transfers (0/20 × 5) |
| **ow-ghini** (new) or ow-moblin widened | 0:0x40, 0x41, 0x31 (both entries), 0x21 with its cave | ow-moblin is 100% zero-shot on the crossings; 0x21 needs the two new steps |
| ow-lynel, retrained | 0x50 E → N and S → N, 0x60 E → N, 0x32, 0x22 | 0/40, 5/20, 7/40, 35/40 today |
| ow-moblin, widened | 0x42 cave → S, 0x52 N → W, 0x51 E → S, 0x61 | 100% zero-shot |
| vire, with the magic sword and bombs `never` | 6:0x39, 0x2d | 100% / 97.5% today but with a bomb each |
| stalfos-keese-gel | 6:0x58 (key waypoint), 0x79, 0x0c | 100%, −1.8 hearts: statue fire is new to it |
| zol | 6:0x68 (`treasure: skip`) | 100% |
| trap-room | 6:0x48 | 100% |
| like-like | 6:0x1d | not run (the script crosses it at 0 hearts) |
| equipment-room | the tunnel 6:0x08 | not run (the script: 0.56 hearts to its Keese) |

## 9. Not walked, and why

- **A scripted fight in 0x38 and 0x3a.** Not written: a bot that reads Red phases, Blue rows and the Bubble is the
  size of the problem itself; the models' runs are the only real plays of these rooms (§4.1, §4.6).
- **0x38 and 0x3a from chain arrivals**: no chain reaches them; states were staged at 12 and 8 full hearts.
- **0x7a's fight for the first key** (a run with no carried key), **0x18 (Gleeok) and 0x3c fought**: entered only to
  see that they shut (0x18, 0x3c) or not (0x7a).
- **Models in 0x1d, the tunnel, 0x79, 0x0c, 0x21**: no state (0x1d, tunnel: the scaffold stages by the probe, which
  does not go down stairs), no monsters (0x79, 0x0c), or no objective (0x21).
- **The Lost Woods' second to fourth turn and 0x60, 0x50 after them under a model**: one state per room; the
  buttons' walk is in §1.
- **The magic shield** against a Red shot (not ours), **flying Ghinis attacked** (code says immune).
- **Whether the fight models took 0x58's and 0x2d's keys**: not read from the replays.
- **Schema 6's column values in these rooms**: taken from `observation-next.md`, not re-probed.
- **A whole run of the stretch end to end**: every room here starts from a staged state at full hearts.

## 10. Questions only the owner can answer

1. **0x38 and 0x3a**: they are the dungeon. Train a `wizzrobe` family on them from like-like's weights with the
   planner as a teacher (the standing answer 1), and is a first team at "about 30% on its segment" still the bar
   when the two rooms multiply to 20% from full hearts before anything else is counted?
2. **Hidden positions for Red Wizzrobes**: the walk says `ram`, `last_seen` and `zero` give a model the same
   knowledge of where a Red will appear (none). May the wizzrobe family simply train on `ram`, the question being
   kept for Ganon?
3. **The phase counter**: a Red Wizzrobe's exact frames-to-shot is in schema 6 (`phase_time`). A player sees the fade
   and counts roughly. Keep it?
4. **Bombs in Dungeon 6**: `never` outside 0x28 for every family (both bombs are spoken for), or `free` with the
   extra bombs counted as cheats? A bomb kills a Red in one but is unreliable against Blues.
5. **The doorways into 0x18 and 0x3c and 0x29's key door**: masked as a rule, or left to wrong-exit endings (one
   step ends the run)?
6. **Rupees**: 28-40 arrive, Gohma took 1-3 by script. Is a miss allowed to cost Ganon's reserve, or is ARROW offered
   only when it kills (the aimed mask), which makes a miss impossible?
7. **The gravestone**: pushed from below as the walk found cheapest (the draft says from above), and the cave's
   refusal under 12 containers: end the run as `missed`, or walk on to Dungeon 6 with the white sword (measured:
   hopeless in 0x38 and 0x3a)?
8. **The graveyard's family**: a new ow-ghini (by enemy), or ow-moblin, which already crosses all four rooms?
9. **ow-lynel's 0x50**: retrain it for the corridor N, or take the plain way's 0x50 from S for both ways (the same
   corridor, the same Lynels)? Neither way is played today.
