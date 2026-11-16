# Round 29 walks: the fights of Dungeon 9, Ganon and the end of the game in the running game (Nov 17)

`rest-of-game.md` "Gap list" (round 29) and `d9.md` §11 #1, #5, #8, answered by playing the running game before
anything of round 29 is built. This document is about the **fights**; the leg, the level's graph, doors, tunnels,
bombs and keys are another walk's. Every number here was observed (**walked**); how each item was staged, what was
written to RAM and the trials are in its section. Where chance is involved a number is from at least 20 trials, each
loaded with the game's own randomness for its seed. Scripts (`scripts/experiments/`): `r29_fight_walks.py` (staging,
the rule-based fighter), `r29_fight_lanmola.py` (item 1), `r29_fight_patra.py` (item 2), `r29_fight_rooms.py` (item
3), `r29_fight_ganon.py` (items 4 and 5), `r29_fight_scaffold.py` (a throwaway data root with a scouting route
`d9fights` so today's models can be played in the fight rooms).

Link on these walks: 15 heart containers, the magic sword, bow, wooden arrow (silver in Ganon's room), red candle,
recorder, ladder, raft, boomerang, all eight pieces, 10 rupees, 4 bombs, 1 key; **every fight without the red ring
and with it, from 15 hearts and from 8**. Hearts are read to 1/256 (the ring's quarter hearts are below what the
status bar shows).

## The answers

| # | Item | Answer (one line) | Corrects |
|---|---|---|---|
| 1 | Lanmolas, 0x55 | 1 px a frame, a segment out every 8 frames, 2 hearts a touch (½ with the ring); a hit segment is harmless while it flashes; one swing takes up to 3 segments. **Rule: 20 of 20 in all four conditions, 1.00 hearts lost without the ring (0.25 with), 289 frames; to the cellar 615.** Every model family: 0 of 20, dead | `d9.md` §4.4 (code → walked) |
| 1 | Re-entry | **One worm dead, out by the S wall and back: the room is empty (20 of 20), also for good after a clear.** Three segments and no head: both worms whole again | `d9.md` §2, §4.4, `level-9.md`, `enemies.md`, `late-monsters.md` §3.15 ("back on every entry") |
| 2 | Patras | Children all out at frame 66 (`$47`) / 77 (`$48`); orbit 73 / 85 frames; first pattern a ring 38-53 / 18-35 px; second pattern `$47`: each child on its own flat ellipse (0-59 px, through the body), `$48`: the ring swells 21 → 85 px and back every orbit; flips at 64 then every 292-293 (`$47`), 75 then every 256 (`$48`). Body untouched while one child lives; 3 magic hits after. Arrows, bombs, boomerang: nothing (0 of 780 uses) | `d9.md` §4.5 ("ring half as tall", "circle twice as wide") |
| 2 | The pattern can be frozen | **The child in slot 3 dead: no flip ever again (40 of 40).** Slot 2 dead: the next flip comes at once | new |
| 2 | Kills by rule | 0x52: 20 of 20, 0.70 hearts (0.18 ring), 829 frames. 0x61: 20 of 20, 2.50 (0.63), 1431 frames; from 8 hearts 16 of 20. **0x16 (the swelling kind): 17 of 20 and 8.65 hearts without the ring, 20 of 20 and 2.33 with it** | — |
| 3 | Wizzrobe rooms, models | **Six families (like-like, moldorm, darknut, zol, goriya, vire), the red ring, 15 hearts: 0 clears in 0x05, 0x20, 0x10, 0x30 (and 0x55, 0x61); they die having lost 11-15 hearts with the ring.** The planner on like-like: 0 of 20 each. 0x03: `stuck` 20 of 20 | `d9.md` §9 #2 (now with numbers) |
| 3 | The same rooms by rule | A rule-based fighter empties them: **with the red ring 0x05 18 of 20 (1.45 hearts), 0x20 20 of 20 (2.79), 0x30 20 of 20 (1.59), 0x10 18 of 20 (5.99), 0x03 20 of 20 (0.05); without the ring 18, 14, 18, 7 and 20 of 20 at 5.8, 9.8, 6.5, 13.05 and 0.2 hearts. 0x07, the ring's own room, before the ring: 17 of 20 at 9.35 hearts, from 8 hearts 5 of 20.** After every clear but five (all in 0x10) the push and the stairs worked | — |
| 3 | Bubbles | Only the flashing kind (`$2B`) is in level 9: sword off 160 frames, 0 hearts. A red one (type written) takes the sword until a blue one is touched; neither exists here. Bubbles and Traps do not count for a block | `d9.md` §4.1 (confirmed) |
| 4 | Ganon | Scene 272 frames (the same with 7 pieces written); **he stands on the room's spawn spot, one of 7 seen in 20 entries**, and stays there, hidden, the first 63 frames of the fight; then 1 px a frame, a turn toward Link every 64 frames on alternating axes; fireball every 64 frames, born 4 px right of him, stands 18 frames, 1.7-1.9 px a frame, 1 heart (¼ ring); touch 4 (2 blue ring, 1 red). 4 magic hits, shown 64 frames each, the jump to X `$B0` after a hit on an even frame, `$30` on an odd one (60 of 60). Brown 508-509 frames, still, no shots; **one silver arrow from his row or column kills (20 of 20, 14 frames after B, 1 rupee)**; a wooden arrow flies through him and costs its rupee; with 0 rupees B fires nothing; left alone he is blue again at (`$B0`,`$A0`) with the count at four | `d9.md` §6.2 ("stands at (`$50`,`$5D`)"), §6.3 table (left-facing row) |
| 4 | Hidden positions | **RAM eyes: 20 of 20 in all four conditions, 1.25 hearts without the ring (0.31 with), 754 frames from the door, one arrow.** Screen only (fireball births, the last flash, the two jump spots): 14 of 20 without the ring (6 deaths), 20 of 20 with it at 1.79 hearts; sweeping without the births: 10 of 20, with the ring 20 of 20 at 3.68 | `d9.md` §12 #1 (now with numbers) |
| 5 | After Ganon | **A scripted run from his door to the first frame of game mode `$13`: 20 of 20 in all four conditions, 1478 frames (median), 714-750 from the arrow.** The Triforce of Power shows 159 frames after the arrow on his ashes; taking it holds Link 192 frames and refills nothing; the doors open 18 frames later. In 0x32 one stab up from (`$78`,`$AD`) kills both fires of the path (20 of 20, 0 hearts); walking through them costs ½ heart (⅛ with the ring). A silver arrow kills a fire too; the boomerang does not | `d9.md` §7 |
| 6 | Survival | **The fights alone cost the best play seen 41 hearts without the ring; with the ring 10.4 before it (0x55, 0x07) and 15.3 after: 25.7, of 15 with no refill.** Without the ring Link is dead in the fourth fight (0x20); with it in the sixth. Only Ganon, the Lanmolas, 0x03 and the `$47` Patras are cheap | `rest-of-game.md` survival table ("none: monsters frozen") |
| 7 | Engine | No missing model input found. Missing: every model, the Patra and Lanmola kill bookkeeping, Ganon's arrow rule and phases, the Triforce of Power and the end of the game as steps (§7) | — |

**The ten findings that most change round 29's plan**

1. **No model plays any fight of level 9.** Six families zero-shot, the red ring, 15 hearts: no clear in 0x55, 0x05,
   0x20, 0x10, 0x30, 0x03, 0x61 in 1,440 episodes; with the planner and without the ring 320 more, none. In the
   Wizzrobe rooms they die having lost all 15 hearts *with the ring*. like-like, Dungeon 6's best, is no better than
   the rest; moldorm does nothing for the Lanmolas.
2. **Every fight is won by a rule that reads RAM positions** (the same things the observation holds): Lanmolas 20 of
   20, the `$47` Patras 20 of 20, Ganon 20 of 20, and with the ring every Wizzrobe room 18-20 of 20. So the rooms are
   not unfair and no input is missing: what is missing is training, and these scripts are there to teach from.
3. **The hearts do not add up.** The fights cost the rule 41 hearts without the ring and 26 with it (10.4 before the
   ring), of 15 with no refill. The Wizzrobe rooms are the whole problem: 5.8-13 hearts each without the ring,
   1.5-6 with. A model must play them two to three times better than the rule, take the drops, or clip past two of
   them (§6).
4. **The red ring is not optional, and its own room is the worst fight.** Without the ring Link is dead in the
   fourth fight by rule. With it, 0x07 comes first at full damage: 9.35 hearts, 17 of 20 from 15 hearts, 5 of 20
   from 8. The block clip of 0x07 (not walked here) is worth more than any other single thing.
5. **Ganon is decided by whether the model sees him.** RAM eyes: 20 of 20 at 1.25 hearts, 482 frames of fight, one
   arrow. Screen only: 10-14 of 20 without the ring, because his unseen touch is 4 hearts; with the ring 20 of 20 at
   1.8-3.7. "Go to where he was last seen" can chase him for ever.
6. **Ganon is not where the documents put him.** He stands on the room's spawn spot (seven seen in 20 entries),
   stays there hidden for the first 63 frames of the fight, then walks by a rule with no chance in it. The side of
   his jump after a hit is the parity of the hit's frame.
7. **The silver arrow is a rule-sized thing.** One arrow from his row or column (a 21-px band) killed 20 of 20, 14
   frames after B; a wooden arrow or a shot at the blue Ganon only burns the rupee; with 0 rupees the room is a
   grave. ARROW belongs behind a rule like Gohma's, and the rupee reserve behind a top-up.
8. **A Patra's pattern can be frozen.** With the child in slot 3 dead it never changes again (40 of 40). Both
   fights begin in the second pattern; the `$48` kind's is a ring that swells to 85 px, and 0x16 is the one Patra a
   Link without the ring should not fight (8.65 hearts, 17 of 20; 2.33 with the ring).
9. **The Lanmolas do not come back.** One head's death empties the room for every later entry; the documents said the
   opposite. The fight itself is a hearts' worth by rule, and the beam helps.
10. **From Ganon's death to the end is a walk**: 20 of 20 by script, 714-750 frames. The Triforce of Power shows on his ashes 159 frames after the arrow, its fanfare holds Link 192 frames and refills nothing, the doors open 18 frames after it; in Zelda's room one stab up from (`$78`,`$AD`) puts out both fires of the path and the next 16 pixels up are the end of the game.

## How the walks were staged

- **Rooms.** `triforce/probe.py` from `start.state` (`r29_fight_walks.stage_d9` / `d9_state`): the inventory above
  is written in level 9's start room, the room id and the level block's door type are written, and Link walks into
  the room through the side the route enters by; the pre-entry frame is saved and every trial loads it with
  `randomize_rng` and its seed, so the monsters are made fresh and are live. Dark rooms stay dark. Entries: 0x55 S,
  0x61 E, 0x16 S (the second visit), 0x05, 0x30, 0x03 E, 0x10 and 0x07 S, 0x20 N (the second visit's side, the room
  full), 0x42 S, 0x32 S. **0x52 is reached only by its tunnel**: it is entered by its N shutter and Link is WRITTEN
  to the tunnel's arrival spot (`$60`,`$9D`) on his first frame. Link's first frame through a bombed wall or shutter
  is at Y `$BD` here (the probe opens the neighbour's side), 8 px below where the route's own entry leaves him.
- **Written to RAM in single measurements** (said again where used): Link's position and facing beside a monster for
  the hit boxes, the touch tests and the weapon tests; hearts rewritten each frame where a count needs him alive;
  monsters "put into the death state" (`ObjMetastate` `$10`, what a kill sets) for "who counts" and for the Patra
  pattern test; a Bubble's type for the two kinds the level does not hold; the B item (`SelectedItemSlot`, what the
  sub-screen would set).
- **The rule-based fighter** (`r29_fight_walks.Bot`). It reads positions from RAM as the observation does, knows
  the room's floor as a graph of half squares (found by trying every step once with the monsters put away), walks
  to a square from which a stab reaches a target, stabs when the target will be inside the sword's box
  (asm:Collision_CheckStabVsMonster) on the frames the sword is out, and keeps off squares a harmful object will
  reach.
  It is geometry, not search: no frame is tried and taken back, no random number is read. Each monster adds what
  is known of its movement (a worm's body follows its head; a Patra child's orbit; a Red Wizzrobe's line of fire).
  **It is a baseline a model should beat, not the best play there is**: its numbers are "the best play seen".
- **Models.** `r29_fight_scaffold.py` copies the data with the route `d9fights` (one step per fight room, extending
  `d6`), every fight room tagged with the one family under test, `ram_randomization: {}`, start states from the
  staging above. Promoted runs of round 25 (converted, 1.5.1.0), each model's act mode `temperature:0.5+relax:16`,
  20 episodes a room (clip families: 20 with the switch on and 20 off); the planner `+plan:16:1`, 20 a room. All
  through `triforce eval-batch`. The steps ask for the stairs (0x55, 0x05, 0x61, 0x20, 0x10, 0x30, 0x03) or the N
  door (0x16, its key door written open).

## 1. The red Lanmolas of 0x55 (walked)

`r29_fight_lanmola.py`. One list entry makes ten segments in slots `$01`-`$0A`: worm 1 `$01`-`$05`, worm 2
`$06`-`$0A`, heads `$05` and `$0A`.

- **Coming out and speed** (`facts`, 20 seeds, Link still): all ten at (`$40`,`$8D`); the heads move from frame 1,
  the next segment 8 frames later (1, 9, 17, 25, 33): the whole worm is out by frame 34. Every segment moves exactly
  1 px a frame (44,560 of 44,560 steps); Link walks 1.5.
- **Touch**: 2 hearts, 0.5 with the red ring (Link written onto a head, 20 of 20 each). **A segment that has just been
  hit does not hurt while it flashes** (Link written onto it, the others put away: 0 hearts in 15 of 15; 5 swings
  missed).
- **How segments die** (one clear logged death by death): the hit segment lives on and the worm's lowest slot turns
  into a dead dummy (`$5D`); **one swing took three segments** (slots 6, 7, 8 in one frame, 9 and 10 flashing): the
  sword is out for 8 frames and every segment in its box is hit once. `RoomKillCount` rises only with a head's
  death (0 → 1 → 2); `WorldKillCount` and `HelpDropCount` rise with every segment; `IsRoomAllDead` is set with the
  second head.
- **Other weapons**, one use from beside the worm, 20 seeds: wooden arrow 1 segment (17 of 20), bomb 3 in the median
  and up to 6 of 10 (16 of 20 hit), magic beam 1-3 (17 of 20), boomerang stuns 1-4 segments and kills none.
- **Re-entry** (`reentry`, 20 seeds each; out by the S wall into 0x65 and back at once):

| Before leaving | `RoomKillCount` | Back in the room | `LevelKillCounts[$55]`, room flags |
|---|---|---|---|
| 3 segments killed, no head | 0 | **10 segments** (20 of 20) | 0, `$24` |
| one worm dead, the other alive with 1-5 segments | 1 | **0 segments, `IsRoomAllDead` set** (18 of 18; in 2 trials the second worm fell too) | 1, `$64` |
| both dead | 2 | 0 segments (20 of 20) | 2, `$A4` |
| both dead, `RoomHistory` WRITTEN to hold no `$55` | 2 | 0 segments (20 of 20) | 2, `$A4` |

  The room's list count is 1 (ids `$32`-`$61`, asm:Room_EnterCommon) and each head's death is banked as one kill
  when Link leaves; a room not flagged all dead makes "count minus banked kills" on every entry
  (asm:RoomHistory_ModifyObjCountUW@CalcObjCount). The documents said the worms are back on every entry: wrong.
- **The clear by rule** (`kill`, 20 seeds): the fighter treats a body segment as following its head's path 8 frames
  a segment behind, and a head as unsure after its next turning point (every 16 px).

| Condition | Cleared | Hearts lost (mean, max) | Frames to the clear (median, max) | Swings |
|---|---|---|---|---|
| no ring, 15 hearts | 20 of 20 | 1.00, 2.00 | 289, 411 | 5-14 |
| no ring, 8 hearts | 20 of 20 | 1.30, 4.00 | 328, 668 | 5-15 |
| red ring, 15 hearts | 20 of 20 | 0.25, 0.50 | 289, 411 | 5-14 |
| red ring, 8 hearts | 20 of 20 | 0.33, 1.00 | 328, 668 | 5-15 |

  From full hearts the magic sword's beam helps (it kills segments); from 8 hearts there is none. After the clear:
  to (`$60`,`$9D`), Up 16 frames (the block), Up to Y `$8D`, Right to the stairs: in the cellar 0x60 in 20 of 20,
  615 frames (median) from the S wall, 223-272 frames after the last kill.
- **The one-worm way** (`trick`): kill one worm, out S and back (0 hearts lost in 0x65's doorway, 20 of 20), the
  block,
  the stairs: 20 of 20, but 1.30 hearts and 1026 frames against 1.10 and 615 for the whole clear. It is no cheaper
  for this fighter (the second worm hunts Link while he hunts the first), and in a chain it is a wrong exit.
- **Models** (zero-shot, the red ring, 15 hearts; §3's table): like-like, moldorm, darknut, zol, goriya, vire all die
  (178 of 180 episodes; moldorm 20 of 20 in 1906 frames, having lost 15 hearts with the ring). moldorm, the
  nearest relative, is no better than the rest.

## 2. The Patras (walked)

`r29_fight_patra.py`. 0x61 and 0x52 hold Patra `$47` (children `$25`), 0x16 (and 0x27, crossed) Patra `$48`
(children `$26`). Patra is slot 1, the children slots 2-9.

- **Timeline** (`facts`, 20 seeds a room, Link written into a corner): Patra at (`$80`,`$70`) from frame 3; the
  children appear one by one above it; **its state turns 1 (every child out: from now they hurt and can be hit) at
  frame 66 (`$47`) and 77 (`$48`)**, 20 of 20. The body moves about a quarter pixel a frame (0.35 counting both axes of its diagonals), turning as it likes.
- **The orbit** (fitted to the game to under 1 px): a child's angle (asm:ObjAngleWhole, 32 steps a circle) falls by
  `$70`/256 (`$47`) or `$60`/256 (`$48`) a frame: **73-74 and 85-86 frames an orbit**, counter-clockwise on screen.
  The pattern index (asm:Patra_ObjManeuverIndex) changes the size of a child's steps, not its place:

| | first pattern (index 0) | second pattern (index 1) | flips |
|---|---|---|---|
| `$47` (0x61, 0x52) | a ring of radius 46 px (38-53 from the body) | each child keeps its sideways swing (±46) and halves its up-and-down swing about the height it had at the flip: eight flat ellipses at eight heights, **0-59 px from the body, through its middle** | at frame 64 (before the children are live), then every 292-293 frames (4 orbits) |
| `$48` (0x16, 0x27) | a ring of radius 26 px (18-35) | each child circles twice as wide about a point on the far side of the body: **the whole ring swells from 21 to 85 px and back once an orbit** | at frame 75, then every 256 frames (3 orbits) |

  So both fights **begin in the second pattern** (the first flip comes before the children are live) and reach the
  plain ring only at frame 357 (`$47`) or 331 (`$48`). With the `$48` kind a room 176 px wide has no still place
  within
  85 px of the body for 256 frames in every 512.
- **The pattern can be frozen** (`freeze`; a child put into its death state by a WRITE at a chosen moment, flips of
  the
  next 1500 frames, 10 seeds for each cell): the flip needs slot 3's angle byte to read 0
  (asm:Patra_Update@TryChangeManeuver), and a dead child's angle stands still.

| Child killed | `$47` in pattern 0 | `$47` in pattern 1 | `$48` in pattern 0 | `$48` in pattern 1 |
|---|---|---|---|---|
| none, slot 4, slot 9 | 5 flips | 5 flips | 5 flips | 6 flips |
| **slot 3** | **0 flips** | **0** | **0** | **0** |
| slot 2 | 5-6, the first within 50 frames | 5-6, the first within 100 | 6, within 100 | 6, within 100 |

  Slot 2's death leaves the flip timer (asm:ObjTimer of slot 2) at 0, so the next flip comes as soon as slot 3's
  angle reads 0. The fighter's own kills show the same (slot 3 killed by the sword in pattern 0: the pattern stayed 0
  to the end of every such run). Slot 3 is the second child to appear, 45° behind the first.
- **Damage**: a child's or the body's touch 2 hearts, 0.5 with the red ring (Link written onto each, 20 of 20).
- **When the body can be hit**: Link written beside it, stabs 41 frames apart: with eight children alive HP stays
  `$B0`
  (20 of 20); with only the last child alive `$B0` (20 of 20); with none (slots emptied) dead after 3 magic sword
  hits (20 of 20). A child dies to 2.
- **Bombs, arrows, boomerang** (Link written beside the ring facing it, 400 frames, 10 seeds each kind): 170 silver
  arrows 0 hits, 50 bombs 0 hits, 170 boomerang throws 0 hits and no stun, in 0x61 and in 0x16. The sword: 83 hits in
  0x61, 104 in 0x16. **The beam hurts them** (full hearts, from 44 px outside the ring): 10 hits in 0x61, 97 hits and
  36 children dead in 0x16, whose small ring a beam crosses from side to side.
- **The kills by rule** (`kill`, 20 seeds): the fighter computes every child's place for the next orbit from its
  angle, stands on a square no child will pass within touch of and from which some child will cross the sword's
  box, stabs when one is about to, and kills slot 3 first when the first pattern runs. Frames are to the body's
  death.

| Room (kind) | no ring, 15 hearts | no ring, 8 hearts | red ring, 15 | red ring, 8 |
|---|---|---|---|---|
| 0x52 (`$47`, open floor) | 20 of 20, 0.70 hearts (max 4), 829 frames | 20 of 20, 1.20, 832 | 20 of 20, 0.18, 829 | 20 of 20, 0.30, 832 |
| 0x61 (`$47`, the block ring in the room) | 20 of 20, 2.50 (max 8), 1431 | **16 of 20**, 3.80, 1229 | 20 of 20, 0.63, 1431 | 20 of 20, 0.98, 1434 |
| 0x16 (`$48`) | **17 of 20, 8.65 (wins 7.53)**, 1068 | **7 of 20**, 6.40 | 20 of 20, 2.33 (max 5.5), 1133 | 20 of 20, 2.38, 1133 |
| 0x16 without the slot-3 rule | 17 of 20, 7.45 | 10 of 20, 6.10 | 20 of 20, 1.93 | 20 of 20, 1.98 |

  Every heart was lost to a child, nearly all while the fighter walked across the ring's reach to a new square; in
  0x61 the body likes to hover over the ring of blocks, out of the sword's reach from any floor square, and the
  fight waits for it (max 3549 frames). In 0x16 the slot-3 rule did not pay for this fighter (the kill has to
  happen inside the 256 frames of the small ring). After the kill: 0x52's N shutter open within 60 frames (20 of 20),
  0x61's `IsRoomAllDead` set (the block moves), 0x16's four bombs at (`$80`,`$90`).
- **Models** (§3's table): 0x61: no kill in 180 episodes (like-like dies, the others leave by a wrong door or stall).
  0x16 is *crossed* S → N (its key door written open) by moldorm in 19 of 20 for 6.25 hearts with the ring and
  three bombs thrown at the Patra for nothing, by darknut in 20 of 40; nobody killed it.

## 3. The rooms that must be emptied for their blocks (walked)

`r29_fight_rooms.py`, `r29_fight_scaffold.py`.

- **What is in them and who counts** (`look`): 0x05 and 0x20 three Blue + two Red Wizzrobes; 0x10 and 0x07 the same
  with three Bubbles; 0x30 two + two with four Traps; 0x03 two Zols, two Like-Likes, two Bubbles. With every
  Wizzrobe, Zol and Like-Like put into the death state (WRITTEN) `IsRoomAllDead` is set in each room with the
  Bubbles and Traps alive (5 of 5 each): **Bubbles and Traps do not count**. The magic sword kills a Zol outright
  (no Gels seen in any clear).
- **Bubbles** (`look`, `bubbles`): every Bubble of the route's rooms (0x65, 0x17, 0x31, 0x07, 0x10, 0x03) is the
  flashing kind `$2B`: touch 0 hearts, `SwordBlockedLongTimer` `$10`, A swings again 160-179 frames later (20 of 20).
  **The first quest's level 9 has no red and no blue Bubble.** With the type WRITTEN: a red one (`$2D`) sets
  `IsSwordBlocked` and A never comes back (450 frames, 20 of 20); a blue one (`$2C`) clears it (20 of 20). In a room
  that only the sword empties a red Bubble would end the run (bombs kill Wizzrobes badly, `d6-walks.md` §5.3); the
  question does not arise in this quest.
- **Today's models, zero-shot** (the red ring, 15 hearts, 20 episodes a room, 40 for clip families; endings; mean
  hearts lost in brackets where it tells):

| Room | like-like | moldorm | darknut | zol | goriya | vire | like-like + planner | like-like, no ring |
|---|---|---|---|---|---|---|---|---|
| 0x55 Lanmolas | death 20 | death 20 | death 40 | death 20 | death 40 | death 38 | death 11, wrong exit 9 | death 20 |
| 0x05 | death 20 (−15) | death 16, wrong exit 4 | wrong exit 40 (out again in 163 frames) | death 17 | death 40 | death 37 | death 20 | death 20 in 762 frames |
| 0x20 | death 20 | wrong exit 20 (at once) | death 40 | death 15 | death 31 | death 36 | death 20 | death 20 |
| 0x10 | death 19 | death 16 | death 34 | death 19 | death 40 | death 40 | death 11, wrong exit 9 | death 20 |
| 0x30 | death 20 | death 19 | death 40 | death 20 | death 40 | death 39 | death 20 | death 20 |
| 0x03 | stuck 20 (−0.4) | stuck 20 | stuck 31 | stuck 20 | stuck 40 | stuck 28 | stuck 19 | stuck 20 (−2.5) |
| 0x61 Patra | death 19 | wrong exit 16 | stuck 34 | wrong exit 19 | wrong exit 27, death 9 | wrong exit 28 | death 18 | death 20 |
| 0x16 Patra, crossed | wrong exit 19 | **N door 19** (−6.25, 3 bombs) | N door 20 of 40 (−5.0, 3 bombs) | wrong exit 19 | death 35 | wrong exit 40 | wrong exit 20 | wrong exit 20 |

  **Not one clear in 1,760 episodes.** In the Wizzrobe rooms a model with the red ring lasts 2,500-2,900 frames and
  loses all 15 hearts (a still Link in 0x05 with the ring lost 9.5 hearts in 1,500 frames: Blue touch and shot ½,
  Red shot 1); without the ring like-like is dead in 660-800 frames. The planner (`temperature:0.5+relax:16+
  plan:16:1`) makes like-like last about twice as long (4,400-5,300 frames) and clears nothing. like-like, the best
  family in Dungeon 6's block rooms (20% and 50% there), is not better than the others here: these rooms have three
  Blue Wizzrobes, not two, and no Like-Like to be held by. In 0x03, the room nearest to like-like's own, every
  family ends `stuck` (the no-progress rule) with almost no damage; whether they empty the room and then fail at
  the block and the stairs inside the ring was not looked into (the evaluation keeps no kill count).
- **The same rooms by rule** (`clear`; the fighter hits a Blue Wizzrobe while it walks, a Red one while it is solid,
  keeps off the line a Red faces before its shot and off the row or column a walking Blue faces, treats Bubbles
  and Traps as things that harm; held by a Like-Like it stabs until free). Frames are to the room empty; "cellar" is
  the push and the walk to the stairs after it.

| Room | no ring, 15 hearts | no ring, 8 hearts | red ring, 15 | red ring, 8 | Cellar after a clear |
|---|---|---|---|---|---|
| 0x07 (3 Blue, 2 Red, 3 Bubbles; **before the ring on the route**) | **17 of 20, 9.35 hearts** (wins 8.35), 1393 frames | **5 of 20**, 7.05 | 20 of 20, 2.51, 1478 | 20 of 20, 2.53 | 62 of 62 |
| 0x05 (3 Blue, 2 Red) | 18 of 20, 5.80 (wins 6.22), 1316 | 12 of 20, 5.45 | 18 of 20, 1.45 (max 3.5), 1316 | 18 of 20, 1.58 | 66 of 66 |
| 0x20 (3 Blue, 2 Red) | 14 of 20, 9.80 (wins 7.57), 1126 | 8 of 20, 6.80 | 20 of 20, 2.79 (max 5.75), 1291 | 20 of 20, 2.59 | 62 of 62 |
| 0x10 (3 Blue, 2 Red, 3 Bubbles) | **7 of 20, 13.05** (wins 9.43), 1577 | **0 of 20** | 18 of 20, 5.99 (wins 5.57), 2163 | 14 of 20, 5.23 | 34 of 39 |
| 0x30 (2 Blue, 2 Red, 4 Traps) | 18 of 20, 6.50 (wins 5.56), 915 | 13 of 20, 5.40 | 20 of 20, 1.59 (max 4.25), 956 | 20 of 20, 1.60 | 71 of 71 |
| 0x03 (2 Zols, 2 Like-Likes, 2 Bubbles) | 20 of 20, 0.20 (max 2), 464 | 20 of 20, 0.20 | 20 of 20, 0.05, 464 | 20 of 20, 0.05 | 80 of 80 |

  - **A rule that reads RAM and knows the Wizzrobes' two tells empties these rooms; no model does.** With the ring a
    Wizzrobe room costs it 1.5-2.8 hearts and a room with Bubbles as well 2.5-6; without the ring 6-13, which is a
    death in one room in four (0x05, 0x30), in three of ten (0x20) or two of three (0x10).
  - **Who hurts**: Red shots most (4 hearts each, 1 with the ring: 64 of 116 hearts in 0x05 without the ring, 118 of
    196 in 0x20), then Blue shots and Blue touches; Traps cost 0.35 hearts a run in 0x30. In 0x10 the three
    Bubbles take the sword away again and again (6 swings in the median run without the ring against 17 in 0x05) and
    the fighter is shot while it cannot fight.
  - **The push and the stairs** after the clear, from the fight's last square: 0x05, 0x30 Up from (`$60`,`$9D`), the
    stairs appear at (`$D0`,`$60`), Link stands at (`$D0`,`$5D`); 0x20, 0x03, 0x07 Up from (`$60`,`$9D`), up to Y `$8D`,
    right to (`$80`,`$8D`); 0x10 Left from (`$D0`,`$8D`), stairs at (`$D0`,`$60`). In the cellar in every clear of
    five rooms; in 0x10 5 of 39 walks failed on the way round its Bubbles. In 0x30 the walk onto the stairs square,
    the top-right Trap's home, cost nothing extra worth a line (Traps: 7 hearts in 20 runs for the whole fight).
  - 0x03 by rule: 20 of 20 at a fifth of a heart, and the block and stairs inside the ring work as in the other ring
    rooms. Whatever keeps the models `stuck` there is not the room.

## 4. Ganon, 9:0x42 (walked)

`r29_fight_ganon.py`. Link comes in by the S shutter with the silver arrow and 10 rupees.

### 4.1 The room on entry (`scene`)

- 20 of 20: Link halted (`ObjState` `$40`) from frame 1, holding up item `$1B` (a Triforce piece:
  asm:Ganon_ScenePhase0_LiftInDark writes it without looking at `InvTriforce`); both doors shut from frame 9; scene
  phase 1
  from frame 80; **the fight (phase 2, Link free) from frame 272**. With `InvTriforce` WRITTEN `$7F` (seven pieces)
  the
  scene is the same, frame for frame: fewer pieces stop Link at 0x66's old man, not here.
- **Where Ganon stands is not fixed.** His init sets no position: he is on the spawn spot the room gave its first
  object, which follows the game's spawn cycle (asm:SpawnCycle, carried from room to room). 20 entries by the S door:
  (`$50`,`$5D`) 5, (`$A0`,`$5D`) 4, (`$60`,`$7D`) 4, (`$A0`,`$9D`) 2, (`$90`,`$7D`) 2, (`$30`,`$6D`) 2, (`$80`,`$8D`)
  1. He is drawn there from frame 50 of the scene to its end, so a player sees the spot for 222 frames.

### 4.2 His walk while he is not drawn (`moves`; Link still at the door spot, hearts held, 20 seeds x 1500 frames)

- **He stands still on that spot for the first 63 frames of the fight** (his direction is 0 until his turn counter
  first reads a multiple of `$40`), hidden and hittable. Then: 1 px a frame (28,379 of 28,558 moving frames; the 179
  two-pixel frames are the step back at a wall); **a turn toward Link every 64 frames exactly (440 of 440), each on
  the other axis than the last**; the first turn is vertical (S toward the door, 20 of 20); reversal at a wall (182
  times); his X stays in `$30`-`$B0`, his Y in `$5D`-`$BD`. No random number: with Link still he walks one closed
  loop of 256 frames again and again.
- **What RAM holds**: `ObjX`/`ObjY` of slot 1, his true position, every frame, drawn or not; his direction
  (`ObjDir`), his turn counter (`$413`), `ObjTimer` (frames left shown), `ObjState` (brown, counting down), his HP as
  a
  hit counter, `FrameCounter` (the next fireball, the jump's side).
- **What a player can know**: where he stood in the scene (and that he stays 63 frames); each fireball's birth, 4 px
  right of him, every 64 frames; where he stands shown for 64 frames after a hit; the two spots he jumps to; that he
  turns toward Link. Nothing else: no sprite of his is drawn while he walks.
- **Fireballs** (480 seen): every one born on a frame with `FrameCounter` & `$3F` = 0, at (his X + 4, his Y); it
  **stands 18 frames** where it was born (and harms there), then flies straight at 1.7-1.9 px a frame toward where
  Link was; alive 22-98 frames. A still Link took 5.75 of them in 600 frames.
- **Damage** (the same 600 frames a seed, and Link written onto Ganon's middle, hidden and shown, 20 of 20 each):

| | no ring | blue ring | red ring |
|---|---|---|---|
| fireball | **1** | ½ | ¼ |
| his touch (hidden, shown, brown alike) | **4** | 2 | 1 |

  (The task's "fireballs 4 / 2 / 1" are his touch's numbers; a fireball is 1 / ½ / ¼.)

### 4.3 The sword (`sword`)

- **Where a stab finds him** (he stands hidden at (`$A0`,`$5D`); Link WRITTEN to every offset of a grid, A, 14 frames;
  a hit shows him). Link's position minus his:

| Link faces | Link X − his X (every pixel tried) | Link Y − his Y (every fourth) | `d9.md` §6.3 computed |
|---|---|---|---|
| right | −16 .. +15 | −4 .. +16 | X −18 .. +12 |
| left | +8 .. +39 | −4 .. +16 | X +4 .. +34 |
| up | 0 .. +23 | +4 .. +32 | X 0 .. +22 |
| down | 0 .. +23 | −8 .. +8 (he stood on the top row: squares above Y −8 are in the wall) | X −2 .. +20 |

  His touch reaches Link when Link's X − his X is 0 .. +16 and the same for Y (`d9.md` §6.3; the rule's fighters were
  touched only inside that square). So from the left a stab is safe at X −16 .. −1, from the right at +17 .. +39, from
  below at Y +17 .. +32. The computed X windows of `d9.md` sit 2-5 px too far left; the Y windows agree.

- Nothing else touches him while he is blue: the magic beam, a bomb beside him, the boomerang, a silver arrow: no
  hit, no change (one trial each at the standing, hidden Ganon). A stab while he is shown: HP unchanged.
- **Four magic hits**: HP `$F0` → `$B0` → `$70` → `$30` → brown, 20 of 20. After each of the first three he is shown,
  still, for exactly 64 frames (60 of 60), then jumps to Y `$A0` and **X `$B0` when the hit landed on an even
  `FrameCounter`, `$30` on an odd one (47 of 47 and 13 of 13)**: the side is decided by the frame of the hit, a thing
  the screen does not show. With RAM eyes the fighter stands by the right spot before he jumps: 69 frames between
  hits at best, 106 in the median.

### 4.4 The brown phase and the arrow (`brown`)

- **It begins on the fourth hit**: HP back to `$F0`, `ObjState` `$FF`; he stands where he was hit, is drawn, does not
  move and shoots nothing (20 of 20). A stab at him 20 frames in: nothing (HP `$F0`, the state counting on), 20 of 20.
- **Its length**: 508 frames (4 of 20) or 509 (16 of 20): the state falls by 1 on odd frames. He is drawn every frame
  for the first 415-416 and flickers for the last 93.
- **Left alone** he is blue again at (`$B0`,`$A0`) (20 of 20: the count runs out on an odd frame, so it is always the
  right-hand spot) with HP `$F0`: the four hits start over. Nothing else is lost.
- **The arrow** (Link WRITTEN on his row 64-80 px away, the arrow as the B item, a frame of the direction, B; 20 seeds
  each):

| | An arrow flew | What it did | Rupees |
|---|---|---|---|
| silver arrow, 10 rupees | 20 of 20 | **his death began 14 frames after B, 20 of 20** | −1 |
| wooden arrow (`InvArrow` 1) | 20 of 20 | through him; he stays brown | −1 |
| silver arrow, 0 rupees | 0 of 20 | nothing: he turns blue again when the time is up | 0 |

- **From where** (seed 0, Link WRITTEN beside him pixel by pixel): shooting sideways the arrow kills when Link's Y −
  his Y is 0 .. +20 (21 of 21 values, none outside); shooting down when Link's X − his X is −3 .. +17. A 21-px band
  about his middle: two or three of the floor's half-square rows or columns. (Shooting left and up were not measured:
  from where he stood those spots were outside the room.)
- The B item was written (`SelectedItemSlot`); the agent's own switch through the sub-screen was not played here. 508
  frames are enough for it: the rule fired its one arrow 30-60 frames into the phase.

### 4.5 No way out, and death (`scene`, `death`)

- Both doors are shut from frame 9 of the scene (`CurOpenedDoors` 0) and stay shut through the fight, the brown
  phase and his death, until the Triforce of Power's fanfare ends (§5).
- **Death** (Link written onto Ganon at 1 heart): game mode `$11` for 351 frames, then the continue menu (mode 8);
  START on its first choice: mode 3, and **Link stands in 9:0x76 at (`$78`,`$CD`) with 3 hearts of 15**, the silver
  arrow and his rupees kept, the room history cleared. For a chain a death is the end of the run.

### 4.6 The scripted kills, and what each choice of `obs.hidden_positions` lets a model do (`kill`)

One fighter, three pairs of eyes; the same geometry, the same dodging of fireballs (which every pair sees), the
arrow fired from his row or column once he is brown. 20 seeds each; frames are from the door (the scene's 272
included) to the frame the arrow starts his death.

| Eyes | What it knows of a hidden Ganon | no ring, 15 hearts | no ring, 8 hearts | red ring, 15 | red ring, 8 |
|---|---|---|---|---|---|
| **ram** | his position, direction and turn counter every frame; the jump's side from `FrameCounter` | **20 of 20**, 1.25 hearts (max 4), 754 frames | 20 of 20, 1.25 | 20 of 20, 0.31, 754 | 20 of 20, 0.31 |
| **tells** (screen) | his last drawn position; a fireball's birth for 12 frames; the two jump spots (the nearer first); it waits by the spot it believes in | **14 of 20** (6 deaths), 6.25 hearts (wins 2.50), 721 | 14 of 20, 4.15 | 20 of 20, 1.79 (max 4.5), 1008 | 20 of 20, 1.79 |
| **screen** (chaser) | the same, but walks to every birth | 13 of 20, 5.05 | 11 of 20 | 14 of 20, 1.34 | 14 of 20 |
| **sweep** | the same without the births: a stab at the believed spot every 20 frames | 10 of 20 (10 deaths), 10.1 | 8 of 20 | 20 of 20, 3.68 (max 7.5), 1731 | 20 of 20, 3.68 |

- With RAM eyes the fight is a script-sized problem: four stabs, one arrow, 482 frames of fight, and what is lost is
  lost to fireballs (0.75 hearts a run) and a rare touch (0.5).
- Without his position the cost is **his touch**: 4 hearts each time he walks into a Link who cannot see him. The
  screen-only fighters take 1.4 to 2.8 touches a run where RAM eyes take one in eight runs. Without the ring that
  kills 6 to 12 of 20; with the ring every waiting variant wins, because a touch is then 1 heart and he comes to
  Link by himself (he turns toward Link every 64 frames).
- The chaser shows the trap in "go where he was last seen": 6 of its 20 runs with the ring end without a kill in 12,000 frames; Link and Ganon walk
  the same rectangle half a turn apart, each going to where the other was (one run: no hit in 3,000 frames, no
  heart lost).
- What the three encodings would give a model:

| `obs.hidden_positions` | The model reads for a hidden Ganon | What that lets it do |
|---|---|---|
| `ram` | his true position, `phase` (hidden / shown / brown), `phase_time` | what the ram-eyed rule does: 20 of 20 at 1.25 hearts without the ring. More than a player sees |
| `last_seen` | the spot where he was last drawn (scene, flash, brown); the fireballs, as entities | what "sweep" does, and "tells" if it learns that a newborn fireball marks him: 10-14 of 20 without the ring, 20 of 20 with it at 1.8-3.7 hearts. The stale spot is right for 63 frames after the scene and wrong by a room's width after a jump to the other side |
| `zero` | no position; presence, phase, the fireballs | the same as `last_seen` less the remembered spot, which after the first jump is worth little: the jump spots are two constants of the game. It cannot be better than "tells" |

  A player with memory and the rule could do better than these screen fighters: his walk has no chance in it, so from
  the scene's spot, the turn times (63 frames in, then every 64) and Link's own position at each turn his place
  follows exactly, until a jump whose side the screen does not show. A script that keeps that book was not written;
  a model without memory cannot keep it.

## 5. After Ganon: the Triforce of Power, the shutters, Zelda (walked)

`r29_fight_ganon.py end`: the ram-eyed fight, then the walk, in one game, 20 seeds for each of four cases (no ring
and red ring; in 0x32 "stab" or "walk").

| From the arrow's hit | Frames (20 of 20 unless said) |
|---|---|
| his body bursts (dying count `$50`) | 79 |
| the Triforce of Power shows (dying count `$A0`): the room item, id `$0E`, on his ashes (his position + 8, 8) | 159 |
| Link reaches it and is halted (he was 2-4 squares away) | 24-52 more |
| the fanfare, until `InvPowerTriforce` is 1 | 192 more |
| both doors open (`CurOpenedDoors` `$00` → `$0C`) | 18 more |
| to (`$78`,`$5D`), N through the door, playing in 0x32 at (`$78`,`$BD`) | 247-271 more |
| 0x32 to the first frame of game mode `$13` | 68 ("stab"), 77 ("walk") |
| **his death to mode `$13`** | **714-750 ("stab"), 723-759 ("walk")** |

- **20 of 20 runs reached game mode `$13` in each of the four cases**; from his room's door 1478 frames in the
  median (1532 at most), the scene's 272 and the fight included.
- **The ash and the pickup**: nothing can be taken before frame 159; the item lies where he died (seven different
  places in 20 runs, all inside the floor); touching it starts the fanfare at once.
- **The fanfare refills nothing** (hearts before and after: +0.0 in 20 of 20; `d9.md` §3.8 has the code's reason).
- **The shutters**: `CurOpenedDoors` is `$00` from frame 9 of the scene until 18 frames after `InvPowerTriforce`
  turns 1, then `$0C`: N and S together. Ganon's death alone opens nothing.
- **Zelda's room** (0x32, in by the S door at (`$78`,`$BD`)): Zelda at (`$78`,`$88`) and four guard fires at
  (`$60`,`$B5`), (`$70`,`$9D`), (`$80`,`$9D`), (`$90`,`$B5`), made in the room's first frames. The way is straight up
  column X `$78`; the two fires at Y `$9D` stand on both sides of it and both overlap it.
  - **"stab"**: up to (`$78`,`$AD`), one stab facing up: **both fires of the path dead in 20 of 20** (the sword's box
    is 24 px across), then up: 0 hearts lost, mode `$13` 68 frames after the door.
  - **"walk"**: straight up without stopping: one fire touch, ½ heart (⅛ with the ring), a knockback, up again
    through them: mode `$13` 77 frames after the door, 20 of 20.
  - **What kills a fire**: one sword stroke from below (4 fires → 3), one silver arrow from below (4 → 3); the
    boomerang does nothing (4 → 4). **They hurt**: ½ heart a touch, ⅛ with the red ring (one touch each written).
- **The trigger**: every run ended with Link moved to (`$88`,`$88`) beside Zelda and game mode `$13` on the next
  frame; it came on the walk up at X `$78` when his Y reached `$95` (code: asm:Zelda_Update), whether the fires were
  alive or dead.
- Hearts over the whole of it: 1.25 without the ring by "stab" (all of it Ganon's fight), 1.75 by "walk"; 0.31 and
  0.44 with the ring.

## 6. The survival budget of the fights

Hearts lost in each fight by the best play seen (the rule-based fighter: no model cleared anything), 20 seeds, from
15 hearts; "–" is not on that way.

| Fight, in the route's order | no ring: hearts, clears of 20 | red ring: hearts, clears of 20 | Without the detour (no ring), running sum | With the ring detour, running sum |
|---|---|---|---|---|
| 0x55 Lanmolas | 1.00, 20 | 0.25, 20 | 1.0 | 1.0 (no ring yet) |
| 0x07 Wizzrobes, Bubbles (the ring's room) | 9.35, 17 | 2.51, 20 | – | **10.4** (no ring yet) |
| 0x16 Patra `$48` (for its bombs) | 8.65, 17 | 2.33, 20 | – (not needed without the detour) | 12.7 |
| 0x05 Wizzrobes | 5.80, 18 | 1.45, 18 | 6.8 | 14.1 |
| 0x61 Patra `$47` | 2.50, 20 | 0.63, 20 | 9.3 | 14.8 |
| 0x20 Wizzrobes | 9.80, 14 | 2.79, 20 | **19.1** | **17.6** |
| 0x10 Wizzrobes, Bubbles (the silver arrow) | 13.05, 7 | 5.99, 18 | 32.2 | 23.5 |
| 0x30 Wizzrobes, Traps | 6.50, 18 | 1.59, 20 | 38.7 | 25.1 |
| 0x03 Zols, Like-Likes | 0.20, 20 | 0.05, 20 | 38.9 | 25.2 |
| 0x52 Patra `$47` | 0.70, 20 | 0.18, 20 | 39.6 | 25.4 |
| 0x42 Ganon, RAM eyes | 1.25, 20 | 0.31, 20 | 40.8 | 25.7 |
| 0x42 Ganon, screen only ("tells") | 6.25, 14 | 1.79, 20 | (45.8) | (27.1) |
| 0x32 the fires on the way to Zelda (one stab; walking through costs 0.50 / 0.125) | 0 | 0 | **40.8** | **25.7** |

- **Neither way closes.** A Link of 14-15 containers with no refill (the D8 piece is the last) loses, by the best
  play seen, 41 hearts to the fights without the ring and 26 with it; the crossings between the fights (the other
  walk's) come on top. Without the ring the sum passes 15 in 0x20, the fourth fight; with the ring in 0x20 too, the
  sixth, and already at the ring Link has 4.6 hearts left of 15.
- **What a 15-heart Link cannot afford in sequence, without the ring**: any two Wizzrobe rooms (5.8 to 13 hearts
  each); 0x10 alone kills two runs in three. **With the ring**: 0x07 before it (9.35 hearts, a death in three runs of
  twenty from 15 hearts and in fifteen of twenty from 8) followed by anything; after the ring the four Wizzrobe
  rooms and 0x16 (14.2 hearts together).
- **What is cheap** either way: Ganon with RAM eyes (1.25), the Lanmolas (1.0), 0x03 (0.2), the two `$47` Patras
  (0.7 and 2.5), the fires (0).
- **What the numbers leave out**: drops. The fighter picks nothing up; 24 Wizzrobes drop from groups 0 and 2 (31%
  and 41%: hearts and fairies among them) and each Lanmola head from group 3. A play that takes them gets some of
  this back; how much was not measured. And these are one rule's numbers: its Wizzrobe play takes a Red shot a
  room, which a better player does not.
- **The block clips** (`d9.md` §3.4: 0x55, 0x07, 0x61, 0x20, 0x03 share the clippable layout) would remove 0x07 and
  0x20 from this table: with the ring that leaves 0x55 by clip or 1.0, then after the ring 0x16 2.33, 0x05 1.45,
  0x10 5.99, 0x30 1.59, 0x52 0.18, Ganon 0.31: 11.9 hearts, with nothing lost before the ring but the crossings.
  The clips were not walked here (the other walk's).

## 7. What the engine lacks for these fights (evidence above; terms of `d6.md` §14)

**Objectives and route steps**

- **Kill the children, then the body (Patra).** The body cannot be hit while a child lives (§2, 40 of 40) and a
  child's
  death is a dead dummy: no room kill, nothing for a stall rule keyed on room kills to see for sixteen sword hits.
  Schema 6 already says it (`weapon_target` 0 on the body, the children's own rows); the fight objective's target
  and the progress rule have to follow it (children first, the body after), and `PATRA2` (`$48`) is missing from the
  boss list (`crates/zelda/src/enums.rs`), so 0x16's body hits pay less than 0x61's.
- **Segment kills (Lanmola).** Eight of ten kills are dead dummies; `RoomKillCount` moves only with a head (§1). The
  progress rule must count them. The room needs no special exit rule: clear, the block, the stairs is the Olive
  Garden's rule, and the walk reached the cellar 20 of 20.
- **Ganon's phases.** FIGHT on a hidden Ganon as on a 16-px monster at his position + 8 (that is what the rule did:
  the engine's own target is his `ObjX`, which pulls Link into the touch box, `late-monsters.md` §3.17); nothing to
  hit while he is shown; **the silver arrow as a rule like Gohma's** (`GameCompletion::arrow_facings`,
  `objectives::gohma_stand`): ARROW only while his `ObjState` is nonzero (brown), the stand tile on his row or column
  (Link's Y within 0 .. +20 of his, or Link's X within −3 .. +17: §4.4), the facing toward him. The reward must take a
  hit from his `ObjTimer` turning `$40` (HP does
  drop for hits 1-3; the fourth resets it to `$F0`) and the kill from `Ganon_ObjDyingCount`.
- **The Triforce of Power** as the room's treasure: it is the room item (slot `$13`, id `$0E`) and shows 160 frames
  after the arrow; the fanfare holds Link 192 frames (as `menu_frames`); the doors open only then (§5). A step that
  leaves N before `InvPowerTriforce` is 1 cannot succeed.
- **The end of the game as the route's last step**: `exit: end` = the first frame of game mode `$13`
  (`triforce/draft_routes.yaml` already writes it; nothing reads it). The target is Link's Y exactly `$95` at X
  `$70`-`$80` in 0x32 (§5); the two fires on the way are a monster with `weapon_target` 1 in schema 6.
- **0x16's bombs** are a `treasure` nobody is made to fetch: moldorm and darknut left by the N door with the Patra
  alive (§3). If the route needs the four bombs the step needs the kill as its condition (a `waypoint: bombs` or the
  room's `treasure` rule made binding).

**Masks**

- **ARROW in Ganon's room**: off while he is blue (an arrow then is a rupee for nothing: no hit, §4.3), on while he
  is brown, facing him. One arrow killed in every scripted run (§4.6: 20 of 20 with one arrow, one rupee).
- **The rupee reserve**: `rupee_reserve: 1` through all of level 9 (as 6:0x1c keeps 1 for Ganon), and
  `requires_rupees` in 0x42 as in Gohma's room: with 0 rupees B fires nothing (§4.4) and the room is sealed.
- **BOMB off in the Patra rooms and Ganon's** (nothing of theirs is hurt by one: 0 of 50; the `free` families threw
  three at 0x16's Patra), and off in the Wizzrobe rooms as in Dungeon 6 (the route's bombs are all spoken for).
- **BEAMS** need no mask: the beam hurts Lanmola segments and Patra children and is free.

**Families**

- **lanmola** (9:0x55; one room, one monster the others never met: every family dies there with the ring).
- **patra** (9:0x61, 0x52, 0x16; 0x27 crossed on the ring detour): two kinds with different second patterns; 0x16 is
  the hard one (8.65 hearts without the ring by rule).
- **ganon** (9:0x42, with 9:0x32 as its last step or a family of its own for the walk to Zelda).
- **wizzrobe widened**: 9:0x05, 0x20, 0x10, 0x30 and 0x07 (and the crossings 0x65, 0x15, 0x17, 0x31, 0x04 of the
  other walk), with Dungeon 6's six rooms: one training on both dungeons, as the owner asked. Level 9's rooms add a
  third Blue Wizzrobe, three Bubbles (0x10, 0x07) and Traps (0x30).
- **To existing families**: 9:0x03 (two Zols, two Like-Likes, two Bubbles: the Breakfast Room again) to like-like;
  it is `stuck` there today in 20 of 20 and needs a look at the block and the stairs inside the ring before any
  training. The tunnels and the two item cellars to equipment-room.

**Does schema 6 have every column these monsters need?** (`docs/observation-next.md` §2.3, §3, §9)

- **Patra's children**: yes. `visible`, `touch_harms` and `weapon_target` are 0 until Patra's state is 1 (walked in
  round 26 and again here: frame 66 / 77); the body's `weapon_target` is 0 while a child lives. Not in the
  observation: the pattern index, the frames to the next flip, and which child is slot 3. The rule won without the
  first two in 0x61 and 0x52; they would help in 0x16. Slot 3 is the entity in NES slot 3; whether a model can tell
  slots apart depends on the entity ordering, which was not checked.
- **Worm segments**: yes. A segment that is not out yet (`ObjDir` 0) reads as harmless and not a target; the
  fighter needed nothing else but positions (it rebuilt "the body follows the head" from them).
- **Ganon while hidden**: the key `obs: {hidden_positions: ram | last_seen | zero}` is built and Ganon's type is one
  of its two; `phase` tells hidden, shown and brown, `phase_time` the frames left shown or brown. Not in the
  observation: the frame counter's phase (when the next fireball is born, which side the jump takes) and a standing
  fireball's 18 frames. The ram-eyed rule read the turn counter for its safety margin and the frame counter for the
  jump's side; a model sees him at the new spot one frame after the jump, which is enough.
- **Nothing found that a model cannot do without.** The one input that decides a fight is the owner's question
  (Ganon's position).

## 8. Not walked, and why

- **0x52 from its tunnel.** The probe cannot come up a cellar's stairs; Link was written to the arrival spot on his
  first frame. The Patra's own timeline does not depend on it.
- **Entries as the route makes them.** Through a bombed wall or a shutter the route's Link is first free at Y `$B5`
  (X `$C8`, `$28`); the probe's is 8 px nearer the wall. No fight began inside those 8 px.
- **Models in Ganon's room and in 0x52**: no rule aims the silver arrow and nothing turns ARROW on for a boss family
  but Gohma's; 0x52's start state cannot be made without the tunnel. Left for the build.
- **Why every family is `stuck` in 0x03**: the evaluation keeps no kill count per room; a replay would tell whether
  the room is emptied and the block and stairs are the problem.
- **The B-item switch through the sub-screen inside the brown phase** (written here), and the agent's ARROW action
  itself against Ganon.
- **A screen-only fighter that keeps the book of his walk** (§4.6): his walk is rule-bound, so it can be written; it
  was not.
- **After a death**: only the first frame after CONTINUE (0x76, 3 hearts). Whether a room cleared before the death
  is full again was not walked (for the Lanmolas the code says it stays empty).
- **The blue ring** (only the damage of Ganon's fireballs and touch), **the white sword**, **0x27's Patra with the
  bomb at the wall** (the other walk's), **shooting left and up at the brown Ganon**, **the sword's reach facing
  down from above Y −8**.
- **Which entity of the observation is "slot 3"** of a Patra (§7).

## 9. Questions only the owner can answer

1. **Ganon's position while he is not drawn** (`obs.hidden_positions` for the ganon family). `ram`: a rule wins 20
   of 20 at 1.25 hearts without the ring. Screen only: 10-14 of 20 without the ring, 20 of 20 with it at 1.8-3.7
   hearts; the difference is his 4-heart touch. So the question is tied to the next one: without the red ring a
   model that cannot see him has to be far better than these rules; with it "wait and stab where a fireball is born"
   is enough.
2. **The red ring.** It costs the fight of 0x07 before the ring and three bombs and ten room entries (the other
   walk's numbers); it saves three quarters of everything after. §6's table has the fights' side of that sum.
3. **0x16's Patra for its four bombs**: 17 of 20 and 8.65 hearts without the ring by rule, 2.33 with it. With the
   ring the draft's order (kill it on the second visit) stands; without the ring the route should not fight it at
   all.
4. **The rule-based fighters as teachers.** They win every fight no model wins. Round 25 cloned a scripted rule's
   clean runs during PPO for 5:0x37 (`train --demos`). May the new families (lanmola, patra, wizzrobe, ganon) start
   from these scripts' runs?
5. **The silver arrow**: a rule like Gohma's (ARROW offered only while he is brown, only facing him from his row or
   column) or a learned action? The walk says rule: one arrow, 20 of 20, and a wooden-arrow or blue-phase shot
   only burns the rupee the kill needs.
6. **Killing slot 3 first** (the Patra's frozen pattern) and **the one-worm re-entry** are quirks of the game's
   bookkeeping, like the clips. May a reward or a demo teach the first? (The second is a wrong exit and no cheaper;
   it need not be used.)
7. **The end**: the run ends on the first frame of game mode `$13` (`d9.md` §12 #4); the scripted walk reaches it
   714-750 frames after the arrow, 20 of 20. Is that the route's last step?
