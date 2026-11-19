# Phase IX walks: the owner's changes to `d2first` (Nov 19)

The owner watched the first complete game by the models (`corpus/replays/v1-golden-ending.tfreplay`) and asked for
six changes to `d2first` (the owner's notes of Nov 19, messages 1-6): Dungeon 1 (leave and re-enter at the entrance
so its locked N door is open; skip the key rooms 1:0x72 and 1:0x74; bomb 1:0x53's N wall into 0x43, skipping 0x52 and
0x42), Dungeon 6 (0x78 a ringleader room: opportunistic kills for a bomb pack; take 0x29's key; no bombs on Vires in
0x2d once Link has the magic sword), Dungeon 8 (do not take 0x4c's key), Dungeon 9 (0x66 N into 0x56, bomb its W wall
into 0x55, skipping 0x65). This document walks each change in the running game before anything is built: what the
game does, what it costs against today's route, the key ledger of the whole route after the changes, and the route
data and engine vocabulary the building agent needs. Nothing on `main` changed. Every number is **walked** (observed)
unless a line says otherwise; where chance is involved it is from 20 trials or more, each loaded with the game's own
randomness for its seed (`Game.load(..., randomize_rng=True)`), monsters live. Scripts: `scripts/experiments/r31_*.py`
(the last section lists the commands).

Today's numbers ("today") are the chains of record `r29land-1k` (1000 runs a segment, models alone, clips on), read
per route step by `r31_ledger.py`. Step numbers are `d2first`'s at v1.9.0.0.

## The answers

| § | Item | Answer (one line) |
|---|---|---|
| 1.1 | 1:0x73's N door | **Out S to 0:0x37 and straight back in: the N key door is open, 20 of 20, no key spent (0 keys or 3), and from that entry on it is open for good** (its door flag is saved: asm:Door_LayOut). 380 frames from the first Down to playing in 0x73 again, 0 hearts. Link must reach the scroll (the edge alone does nothing); an overworld scroll or a cave before coming back undoes it |
| 1.2 | 1:0x53's N wall | A bomb wall in the ROM (level-1.md `$53` N "bomb") and in game.yaml (`bomb_walls: [N]`), but **no `requires_bombs`**: the path search does not pass it and no bomb is granted. One bomb opens it 79 frames after B from X `$68`-`$88` on Y `$5D` facing N, from X `$78` Y `$5D`-`$85` facing N, facing E from X `$60`/`$68`, facing W from `$88`/`$90` |
| 1.3 | 1:0x43 from S | Scripted 20 of 20 at 0.50-0.85 hearts (Gels), 304-316 frames; today's models 0.44 hearts, 380 frames |
| 1.4 | The new Dungeon 1 | **Today's models play it** on a scratch route (no retraining): 99, 97 and 94 of 100 from 4.5, 3.5 and 2.5 hearts (today's Dungeon 1 from such hearts: 98.3, 94.2, 91.8%), every loss in a room the change leaves alone; 0 cheated keys; 0x53 (fight, key, bomb) is the slow room: 1240 frames against 608 to its W door today |
| 1.5 | Its keys | 3 in, 3 out, as today: the trick and the bomb save the two doors (0x73 N, 0x52 N) the two skipped keys (0x72, 0x74) paid for. Picked up: 0x53, 0x33, 0x23, 0x45; opened with a key: 0x33 N, 0x23 W, 0x43 E, 0x45 N |
| 2 | The key ledger | **No cheated key anywhere, today or after phase IX.** Dungeon 6 out with 3 (0x29's key), Dungeon 8 out with 3 (0x4c's not taken), level 9 entered with 3 as today. The one zero is 9:0x63 W → 0x62 → 0x61's key, before no key door, as today. Without 0x29's key, skipping 0x4c's would leave 9:0x63 W cheated |
| 3.1 | 6:0x78 | The leader is the Red Wizzrobe in object slot `$01` (asm:Secret_Trigger2Ringleader). At full hearts (the magic sword's beam) **up to two swings from the doorstep before frame 60: a kill in 100 of 100 runs, the leader (all five dead) in 17, a bomb drop in 23, for 13 frames (median) and 0.04 hearts**. Below full hearts (no beam) the Reds appear 2-5 squares away: a swing is possible in 32 of 100, the leader in 7 |
| 3.2 | 6:0x29's key | **On the floor at (`$80`,`$90`) from the first frame** (no trigger); taken 40 of 40 by walking E along the door row over the ring's W side (the ladder), back to X `$78` and down over its S side: 1.38 hearts, 312 frames, against today's path 1.65 hearts, 303 frames (scripted, same seeds) |
| 3.3 | 6:0x2d and 0x39 | The step-level `bombs: tenth` says it (BOMB only for the 10th damage-free kill, whose bomb arms a forced drop of 4). The only Vire rooms after the magic sword on the route are 6:0x39 (step 273: 705 bombs in 795 runs today) and 6:0x2d (step 277: 531 in 397); none in levels 8-9 |
| 4 | 8:0x4c without its key | Out of the ring, along row `$9D`, up column `$B0`, the wall: **20 of 20 at 0 hearts with one arrow (1 rupee), 273 frames**; straight without an arrow 2.40 hearts (max 4: from the chains' 4.5 hearts, 0.5 left); with the key 966 frames. One bomb. The arrows and a rupee reserve stay; 8 rupees for Ganon is still the reserve's reason |
| 5 | Dungeon 9 | 0x66 N through the shutter the eight pieces open: 20/20, 0 hearts, 250 frames. 0x56 S → W wall: 20/20, **0.25 hearts, 412 frames** (by the bottom row and X `$20`). 0x55 from E: **CLIP N at (`$A0`,`$9D`)**, offered by the engine, 20/20, 0 hearts, 197 frames. Bombs: the same count as today (0x56 W replaces 0x65 N) |
| 6 | Specification | §6: routes.yaml and game.yaml as they should read, families, states by name, the engine vocabulary (leaving a level by its entry door is **not** round 26's "room left by its entry door": game.yaml has no such exit and its convention forbids one), tests |

**The findings that most change the plan**

1. **The entrance door is open for good after one trip out.** It is not a stale screen: the door is drawn open by the
   entry's layout and the layout saves its flag (asm:Door_LayOut → asm:RoomFlags_SetDoorOpened); entering 0x63 saves
   the other side. Every later visit (and a Wallmaster's return to 0x73) finds it open.
2. **Dungeon 1's key count does not change.** The owner's reason for 6:0x29's key ("skipping an extra in d1") does
   not hold: D1 ends with 3 keys either way. The key 0x29 gives is the one 8:0x4c's skip takes away; together the two
   leave level 9 exactly as today.
3. **Today's models play the new Dungeon 1 without retraining** (§1.4), but the route data must say 1:0x53's wall is
   one Link may bomb (`requires_bombs`), or the objective leads round by 0x52 and a run with an empty bag (100 of 822
   full-route runs enter Dungeon 1 with 0 bombs) has no bomb.
4. **Leaving a dungeon by its entry door is new vocabulary.** game.yaml says "the entrance room's S door back to the
   overworld is no exit (leaving the dungeon is never a route step)"; the step 0x37 `cave → cave` has to make Link step
   off the mouth first (asm:UndergroundExitType); the new first visit of 0:0x37 from the cave renames the
   after-Dungeon-1 state `0_37c_*` to `0_37c_v2_*`, and the `d1` segment's target "0:0x37" would end on the trick.
5. **0x78's leader is a lottery the beam wins often enough.** A swing at a Red on the door row from the doorstep, at
   full hearts, kills one in every run and the leader in one run of six; a bomb pack falls in about one run of four,
   for a quarter second. Below full hearts it is a fight (the Reds stand out of the sword's reach), not a swing.
6. **Dungeon 9's new way is cheaper by script**: 0x66 N, 0x56, 0x55 from E cost 859 frames and 0.25 hearts against
   today's models' 1096 frames and 1.09 hearts in 0x66 W, 0x65, 0x55 from S (the scripted 0x65 cost 4.5 hearts,
   d9-walks.md), and the clip from E is shorter than the one from S.

## How the walks were staged

- **States.** Every trial starts from the route's own entry states (`triforce/states/`, chains 0-2, as the chain
  runs bring them: `1_73s` 5 of 5 hearts, 1 bomb, 3 keys; `6_78e`, `6_29w` 12 of 12, the magic sword; `8_4cc` 13 of 13,
  8 bombs, 37 rupees; `9_66s` 15 of 15, 6 bombs, 3 keys), loaded with `randomize_rng` and the trial's seed (chain
  `seed % 3`). Hearts the chains bring are written where they matter (`--hearts`); what the chains bring was read
  from `r29land-1k` (§1.4, §3, §4).
- **Rooms the route has not entered yet** (0x43 from S, 0x56 from S, 0x55 from E) were made once by a staging walk
  from the room before (`r29_route_walks.Stager`: the monsters pinned in the NW wall, so the room is not "all dead";
  the wall bombed; the scroll's last pre-entry frame saved), then played with the monsters made fresh and live.
- **Straight walks** go point to point, back onto the line after a knockback (`r27_walks.Tally`); a Link held by a
  Like-Like stabs once every 12 frames (`r29_clips.follow`). Hearts lost are booked to the object nearest Link (`$5F`
  is the ladder: the nearest object while he stood on it). The clips are the engine's own `CLIP` macro where
  `Game.skip_available` offers it.
- **The models** played the new Dungeon 1 through the job queue (`triforce eval-batch`) from a scratch checkout
  whose routes.yaml and game.yaml carried the new steps (§1.4); nothing of it is committed.

## 1. Dungeon 1

### 1.1 The entrance door (`r31_d1_door.py door`, `frames`)

**Why it opens (the disassembly).** Walking out of 0x73 S is a room scroll whose next room id is invalid. Mode 7's
first init step stores the old room's open doors and sets `CurOpenedDoors` (`$EE`) to the side Link enters the new
room by, the side opposite his facing (asm:Door_SetEnteringDoorwayOpened): facing down, `$EE` = 8 (N). Only then
does asm:Room_CalcNext find no room and load the overworld (asm:Room_CalcNextRoomForDoor → asm:Mode_LeaveLevelToOW).
Nothing in the overworld writes `$EE` until the next scroll (which sets it again) or a cave or cellar (asm:Mode_InitSubroom_Sub0
clears it). Coming back down the level's entrance, the room layout ORs the saved door flags into `$EE`
(asm:Door_AddRoomFlagsToOpenedDoors) and asm:Door_LayOut keeps the bits of door types 4-7: the N key door (type 5)
is drawn open, **and Door_LayOut saves the door flag of every key door or bomb wall it draws open** (asm:RoomFlags_SetDoorOpened).
asm:Door_TouchKey lets Link through an open key door without a key. Entering 0x63 from S, `$EE` = 4 by the same
scroll rule and 0x63's S key door is drawn open and saved too.

**The frame.** From (`$78`,`$CD`) holding Down: mode 6 at frame 12 (Y `$DD`, the edge), mode 7 at 14, **`$EE` = 8 at
frame 16** (mode 7, init step 1 → mode 2), the overworld laid out at 37, Link plays in 0:0x37 at (`$70`,`$7D`) facing
down at frame 184.

**Variants, 20 seeds each** (from `1_73s_<seed % 3>`; keys written 0 but in "with 3 keys"):

| Variant | N door passed | Keys spent | `$EE` on entry | Door flags 0x73 / 0x63 after | Hearts lost | Notes |
|---|---|---|---|---|---|---|
| no trick (control) | 0/20 | – | 0 | 0 / 0 | 0 | the door is a wall without a key |
| out S, back in at once | **20/20** | 0 | 8 | 8 / 4 (0x73's set on entry) | 0 | out 184 + in 196 = **380 frames**; N into 0x63 251 more |
| the same with the chain's 3 keys | **20/20** | **0** | 8 | 8 / 4 | 0 | no key is spent: the door is already open |
| 300 frames waited in 0:0x37 | 20/20 | 0 | 8 | 8 / 4 | 0.15 | the Red Octorok |
| out and in twice | 20/20 | 0 | 8 | 8 / 4 | 0 | |
| out, E into 0:0x38 and back W, in | **0/20** | – | 0 | 0 / 0 | 0.10 | the scroll back set `$EE` = 1 (E) |
| trick, then W into 0x72 and back E | 20/20 | 0 | 8 | 8 / 4 | 0 | `$EE` = 2 after 0x72, but the flag holds |
| down to the edge (Y `$D6`) and back up, no scroll | 0/20 | – | 0 | 0 / 0 | 0 | |
| trick, N into 0x63, back S into 0x73, N | 20/20 | 0 | 8 | 8 / 4 | 0 | |
| trick, N, back S, out S, in, N (a second use) | 20/20 | 0 | 8 | 8 / 4 | 0 | |

**The rule.** Link must reach the scroll (the first frame of mode 7's init, 16 frames of Down from the doorway);
standing at the edge does nothing. Any number of frames outside, as long as he neither scrolls to another overworld
room nor enters a cave before he comes back in. **Coming back in needs one grid step off the mouth first**: Link
comes out on the mouth's foot with asm:UndergroundExitType = 2, and the mouth takes nobody until he has completed a
grid step in mode 5 (mechanics.md; walked: Up alone for 400 frames does nothing; 8 frames of Down, then Up: in after
188 more). Once Link has played in 0x73 after the trip the door's flag is saved: it survives leaving again, every later
visit, a Wallmaster's return (the room flags are kept; asm:Wallmaster_State1Trip puts Link in 0x73 with the room
history cleared, not the flags) and works the same on every trip out (each one sets `$EE` = 8 again).

### 1.2 1:0x53's N wall (`r31_d1_walks.py bombspot`)

The ROM: level-1.md's rooms table `$53` N "bomb", `$43` S "bomb" (one wall, two faces). game.yaml: 0x53 `N: # bomb
wall`, `bomb_walls: [N]`; 0x43 `S: # bomb wall`, `bomb_walls: [S]`. **No `requires_bombs`** on 0x53: game.yaml's
convention passes an intact bomb wall in a path search "only in a `requires_bombs` room or once opened", and
`zelda::cheats::bomb_need` grants a bomb only there.

Staged from `1_53s_0` (the five Stalfos pinned), one bomb a trial (written if the bag was empty), Link facing N
unless said:

| Link stands at | Opens |
|---|---|
| Y `$5D`, X `$58`, `$60` | no |
| **Y `$5D`, X `$68`-`$88`** (every 8 px) | yes, 79 frames after B |
| Y `$5D`, X `$90`, `$98` | no |
| **X `$78`, Y `$5D`-`$85`** | yes, 79 frames |
| X `$78`, Y `$8D`-`$9D` | no |
| facing E from (`$60`,`$5D`), (`$68`,`$5D`); facing W from (`$88`,`$5D`), (`$90`,`$5D`) | yes, 79 frames |

The bomb must lie within the wall's hotspot (asm:Bomb_TryOpenWall): on the door square row or one square below it,
X `$68`-`$88`. The column X `$78` is clear of blocks (layout `$02`: blocks at columns 2 and 9 only).

### 1.3 1:0x43 from S (`r31_d1_walks.py cross43`)

Link comes in through the hole at (`$78`,`$BD`) with the five Gels. The middle block pair (layout `$1F`, X `$70`-`$8F`
on Y `$8D`) stands in the column: up to Y `$9D`, round one side, up to `$7D`, back to X `$78`, up.

| Way | Result | Hearts lost | Frames |
|---|---|---|---|
| W of the middle block | 20/20 | 0.85 (max 1.5), all Gel | 316 |
| E of the middle block | 20/20 | 0.50 (max 2.0) | 304 |
| today's models from S (scratch route, §1.4, chain 0) | 100/100 | 0.44 | 380 |
| today: 0x43 W → N (step 55, models) | 1000/1000 | 0.04 | 303 |

### 1.4 The new Dungeon 1 played by today's models (scratch route)

A scratch checkout of this branch (never committed) had routes.yaml steps 46-55 replaced by `0x73 S→N`, `0x63 S→N`,
`0x53 S→N`, `0x43 S→N` (all `stalfos-keese-gel`; steps 56-67 unchanged), game.yaml 0x53 `requires_bombs: true`, and
`heart_value.json` with the six dropped steps' rows removed (the rows of the steps kept, renumbered). The start states
`1_73s_<chain>` were made by `r31_d1_walks.py states`: the route's own `1_73s_<chain>` loaded, out S, back in, the
level entry's last pre-entry frame saved (`$EE` = 8, keys 3), with the hearts and bombs today's full-route chain
brings to Dungeon 1 (`r29land-1k-full`, 822 runs entering 1:0x73: keys 3 in 812, hearts 0.5-4.5 of 5, bombs 0 in
100, 3-4 in 126, 5-7 in 579): chain 0 4.5 hearts and 7 bombs, chain 1 3.5 and 4, chain 2 2.5 and 0. 100 runs each,
models alone (`--act model`), the promoted team of v1.9.0.0, to the D1 triforce (`--stop-at step:62`, 0:0x37).

| Start (hearts, bombs) | To the D1 triforce | Lost in | Frames a run (median) | Cheated keys / bombs | Today's Dungeon 1 from such hearts (`r29land-1k-full`) |
|---|---|---|---|---|---|
| chain 0 (4.5, 7) | **99/100** | 1 stuck in 0x33 (N → S) | 8336 | 0 / 0 | 98.3% (171 of 174 entering with 4.5) |
| chain 1 (3.5, 4) | **97/100** | 2 deaths in 0x44, 1 in 0x23 | 8574 | 0 / 0 | 94.2% (292 of 310 with 3.5-4) |
| chain 2 (2.5, 0) | **94/100** | 3 in 0x44, 2 in 0x23, 1 in the bow cellar | 8276 | 0 / **100** (0x53's wall: one each) | 91.8% (191 of 208 with 2.5-3) |

The new rooms, all 300 runs (each played in every run, no wrong exit):

| Room (new step) | Frames (median) | Hearts lost (mean) | Deaths | Keys in → out | Bombs |
|---|---|---|---|---|---|
| 0x73 S → N (the door open) | 254 | 0 | 0 | 3 → 3 | – |
| 0x63 S → N | 322 | −0.01 | 0 | 3 → 3 | – |
| 0x53 S → N: the five Stalfos, the key, the bomb | **1240** | −0.32 (drops) | 0 | **3 → 4** | one at the wall (7 → 6, 4 → 3; chain 2's granted) |
| 0x43 S → N | 378 | 0.42 | 0 | 4 → 4 | – |

The deaths are all in rooms the change leaves as they are (0x23, 0x44, the bow cellar), entered with the hearts the
chains bring; today's Dungeon 1 loses the same rooms (`r29land-1k-full`: of the runs that enter Dungeon 1 and do not
leave it, 0x44 25, Aquamentus 15, 0x23 15, the bow cellar 12, 0x33 12). Today's models had never seen 0x53's N exit,
0x43 from S or 0x73 with its door open; they play them. 0x53 is the room that costs: the model clears the room for
the key (as today) and then walks to the wall spot and bombs (1240 frames against 608 to the W door today).

### 1.5 The keys of the new Dungeon 1

Keys in: 3 (Dungeon 2's; the full-route chain: 3 in 812 of 822 runs, 2 in 10).

| Room | Today (steps 46-67) | New |
|---|---|---|
| 0x73 → 0x72, 0x74 | +1 +1 (5) | skipped |
| 0x73 N | −1 (4) | open by the trick (3) |
| 0x53 | +1 (5) | +1 (4) |
| 0x52 N | −1 (4) | skipped: the bomb wall |
| 0x33 | +1 −1 (its N door) | +1 −1 |
| 0x23 | +1 −1 (its W door) | +1 −1 |
| 0x43 E | −1 (3) | −1 (3) |
| 0x45 | +1 −1 (its N door) (3) | +1 −1 (3) |
| **Out** | **3** | **3** |

Found and used in the new Dungeon 1: 4 keys (0x53, 0x33, 0x23, 0x45), 4 key doors (0x33 N, 0x23 W, 0x43 E, 0x45 N);
0x73 N by the trick, 0x53 N by a bomb. The scratch chains: keys 3 → 4 at 0x53, 3 out, 0 cheated (§1.4).

### 1.6 Against today

| Stretch | Today (models, `r29land-1k-d1`, from full hearts) | New |
|---|---|---|
| 0x73 → 0x72 → 0x73 → 0x74 → 0x73 N (steps 46-50) | 2126 frames, 0.14 hearts | out S 184, 0:0x37 196, 0x73 S→N 254 (models): **634 frames**, 0 hearts |
| 0x53 → 0x52 → 0x42 → 0x43 → N (steps 52-55) | 608 + 311 + 324 + 303 = 1546 frames, −0.02 hearts | 0x53 S→N (fight, key, bomb) 1240 + 0x43 S→N 378 = **1618 frames**, 0.10 hearts (300 runs) |
| Dungeon 1 whole | 10040 frames a run (median, 1000 runs from full hearts, from step 46) | 8276-8574 from the trip's end (the 380-frame trip and 0x73 S → S not in it): about **8700-8950**, 11-13% less |

## 2. The key ledger of the whole route

Keys carry from dungeon to dungeon (asm:InvKeys). Read from the chains of record (`r31_ledger.py keys`: keys in and
out at every route step, 1000 runs a segment) and the level documents, then changed by phase IX's steps:

| Dungeon | Today: in → what happens → out | After phase IX |
|---|---|---|
| 2 | 0 → 0x7e +1, 0x4e +1, 0x3e +1 → **3** (no key door) | the same |
| 1 | 3 → 0x72 +1, 0x74 +1, 0x73 N −1, 0x53 +1, 0x52 N −1, 0x33 ±1, 0x23 ±1, 0x43 E −1, 0x45 ±1 → **3** | 3 → 0x53 +1, 0x33 ±1, 0x23 ±1, 0x43 E −1, 0x45 ±1 → **3** (§1.5) |
| 3 | 3 → 0x7b +1, 0x6b +1, 0x5a W −1, 0x49 +1 → **5** | the same |
| 4 | 5 → 0x51 +1, 0x40 +1, 0x30 E −1, 0x31 N −1 (the clip way), 0x01 +1 → **6** | the same |
| 5 | 6 → 0x06 W −1, 0x66 +1 (way back), 0x25 W −1 → **5** (4 or 6 in a few runs) | the same |
| 7 | 5 → 0x38 N −1, 0x18 E −1, 0x1b E −1 → **2** (of 822 runs leaving 0x1b, 248 still hold 3 and 14 hold 1; not read) | the same |
| 6 | 2 → 0x79 W −1, 0x58 +1, 0x2d +1, 0x2c N −1 → **2** | 2 → 0x79 W −1 (1), 0x58 +1, **0x29 +1**, 0x2d +1 (4), 0x2c N −1 → **3** |
| 8 | 2 → 0x5e +1, 0x4e N −1, 0x4c +1 → **3** | 3 → 0x5e +1 (4), 0x4e N −1, **0x4c not taken** → **3** |
| 9 | 3 → 0x14 E −1, 0x16 N −1, 0x63 W −1 (**0**), 0x61 +1 → **1** to the end | the same: 0x66 N, 0x56, 0x55 from E pass no key door and 0x56's key is not taken |

- **Cheated keys: 0 in every segment of `r29land-1k`** (`cheated_keys` of every room record). After phase IX no step
  reaches a key door with 0 keys: the fewest keys Link holds at a key door is 1, at 9:0x63 W (as today); in Dungeon 6
  he reaches 0x79 W with 2 and 0x2c N with 4, in Dungeon 8 0x4e N with 4. Link is at 0 keys only from 9:0x63 W to
  0x61's key (0x62 crossed W through an open door), as today.
- **The two Dungeon 6/8 changes go together.** Skipping 8:0x4c's key alone would bring level 9 in with 2: 0x14 E (1),
  0x16 N (0), and **9:0x63 W would need a cheated key**. Taking 6:0x29's key pays for it exactly.
- **Where it is tight, walked**: level 9 with 3 keys was walked in one game by d9-walks.md §3.1 (3 → 2 → 1 → 0 → 1;
  0x63's N and S key doors take the last key on the first frame of a press, so the step must keep out of them, as
  today). 6:0x29's key was taken in 40 of 40 crossings (§3.2); 8:0x4c's way out without it in 20 of 20 (§4); the new
  Dungeon 1 by today's models in 3 × 100 runs with 0 cheated keys (§1.4).
- The spare: Link ends the game with 1 key, as today.

## 3. Dungeon 6

### 3.1 0x78, the ringleader room (`r31_d6_walks.py ringleader`)

**The rule (code).** Level 6's room `$78` has secret trigger 2 (level-6.md: "ringleader: first monster dead kills
all"; game.yaml `trigger: ringleader`). asm:Secret_Trigger2Ringleader runs every frame: once object slot `$01` is
empty or holds a type ≥ `$53` (a shot, the dead dummy, a dropped item), every object of type `$01`-`$52` in normal
metastate starts its death sparkle (monsters still in a spawn cloud are caught on a later frame). **The leader is
whichever of the five Red Wizzrobes sits in slot `$01`**: the first of the room's list; nothing marks it on screen.
Deaths the trigger starts do not go through asm:Damage_KillMonster, so they add nothing to `WorldKillCount` or
`HelpDropCount` (mechanics.md), but each still ends in asm:Obj_EndMetastate → asm:Drop_SetUpItem: **every one of the
five rolls its own drop**. A Red Wizzrobe drops by group 2 (rate `$68`, 41%; enemies.md); the item is the kill
cycle's column (asm:Drop_ItemTable): a bomb in columns 1, 6 and 8, so **12.3% of deaths drop a bomb item, which is 4
bombs** (items.md, capped at the bag). The magic sword kills a Red in one hit (HP `$40`), the beam too.

**Where they appear.** A Red's first update picks a square 2-5 squares from Link in a cardinal direction, facing him,
on walkable floor with Y `$5D`-`$C3` (asm:Wizzrobe_Red_State3Appear); it fades in for 63 frames (hittable every other
frame), stands 64 (it shoots at the 16th solid frame, about frame 79), fades out, hides 64. From the E doorway the
leader's first square, 100 seeds: on the door row W of Link 45 (X `$B0` 17, `$90` 17, `$A0` 11), in column X `$D0`
above or below Link 39 (Y `$BD` 16, `$AD` 11, `$6D` 7, `$5D` 5), (`$20`,`$8D`) 16. (d6.md §2.3 inferred "all five on
the door row W of Link": Link has already stepped to X `$DC` when they choose, and column `$D0` is floor.)

**Crossings, 100 seeds each** (from `6_78e_<seed % 3>`, 12 of 12 hearts: the beam; then at 10 hearts written: no
beam). A swing is SWORD (A for one frame) at a Red that is fading in or solid in Link's lane ahead, within reach
(the beam's whole row at full hearts, else 24 px), not in the doorway (Link cannot swing there), 12 frames apart.
"Doorstep": Link stops at (`$D0`,`$8D`) facing W until frame 60 or his swings are spent, then the path. "Bombs
taken": a bomb drop within 64 px is walked to after its flashing and back.

| Play (12 hearts, beam) | Through | Hearts lost | Frames (median, max) | Runs with a kill | Leader dead (all five) | Bomb drops (runs) |
|---|---|---|---|---|---|---|
| **today: no swing** (W to `$D0`, up to `$7D`, W to `$78`, up) | 100 | 0 | 280, 280 | 0 | 0 | 0 |
| one swing in passing | 100 | 0 | 293, 293 | 55 | 9 | 7 (6) |
| two swings in passing | 100 | 0 | 293, 293 | 55 | 9 | 7 (6) |
| **doorstep, up to two swings** | 100 | **0.04** (max 4) | **293**, 327 | **100** | **17** | **24 (23)** |
| doorstep, at the leader only | 99 | 0.24 (max 12: one death) | 293, 502 | 73 | 26 | 26 (24) |
| doorstep, two swings, bombs taken | 100 | 0.07 | 293, 409 | 100 | 17 | 24 (23): 24 bombs gained in 16 runs (bag cap 8) |

| Play (10 hearts, no beam) | Through | Hearts lost | Frames | Runs with a kill | Leader dead | Bomb drops (runs) |
|---|---|---|---|---|---|---|
| today: no swing | 100 | 0 | 280 | 0 | 0 | 0 |
| one or two swings in passing | 100 | 0 | 280, 293 | 32 | 7 | 11 (10) |
| doorstep, up to two swings | 100 | 0.24 (max 4: a Red shot) | 328, 341 | 32 | 7 | 10 (8) |
| doorstep, at the leader only | 100 | 0.04 | 328 | 7 | 7 | 7 (5) |
| doorstep, two swings, bombs taken | 100 | 0.24 | 328, 447 | 32 | 7 | 10 (8): 8 bombs gained in 2 runs |

- The kills behave as the code says: 168 deaths in 100 doorstep runs, 24 bomb items (12.3% × 168 = 20.7 expected).
- The leader decides only how many roll a drop: a run that kills the leader gets five rolls (a bomb pack about one
  time in two), one that kills another Red one roll.
- Standing on the door row past frame ~79 is what costs: a Red's shot (4 hearts) flies E along it. The doorstep
  play leaves at frame 60 or after its swings.
- Bombs on the way in: the full-route chain enters 0x78 with 0 bombs in 37 of 90 runs and 1 in 43; the bomb reserve
  is 1 from 0x29 and 0x28's wall takes one. A pack here is worth a bomb wall later.
- Hearts on the way in: full (12 of 12) in 374 of 991 `post-d7` runs reaching 6:0x79 and 38 of 90 full-route runs at
  0x78; 10 or more in about 75%.

**A reward or route-step way to make it opportunistic (what exists, what would be new).**

- Exists, already on for `wizzrobe`: `reward.item_targets` (a dropped bomb is a PBRS target until taken or gone),
  `reward-hit` / `reward-beam-hit` / `reward-beam-distance` for a hit (critic.rs), `reward.heart_value` (hearts
  priced by their worth to the run, so a beam kept at full hearts is worth keeping).
- Exists but does not fit: `treasure: {if_bombs_below: N}` (7:0x1b, 8:0x3f) gates a room **treasure**; 0x78 has none.
  `treasure: clear` makes it a fight to the last monster (one leader kill clears it, but the leader is unmarked: a
  clear is up to five kills, the fight the owner does not want).
- **New, smallest**: a route step's own `drops: first` (the family switch `reward.drops_first` turned on for one step,
  as `bombs:` is per step since round 28): once Link has a wanted drop on the floor (a bomb below his maximum), it
  replaces the exit as the objective's target until taken or gone, so a pack that falls is taken before the N door.
  With it, the hit rewards already paid make the doorstep swing worth a model's while; nothing forces a fight.
- **New, if the models still speed through**: a per-step `bomb_value: V` (a bomb picked up priced as V hearts in the
  heart-value reward), to be set from a refit of `heart_value.json` with bombs as a second axis. Not proposed yet.

### 3.2 0x29's key (`r31_d6_walks.py key29`)

The room (level-6.md: `$29`, 3 Blue and 2 Red Wizzrobes, dark, layout `$14`: a water ring round an 8 × 3 floor) has
**no secret trigger**: its room item, the key at (`$80`,`$90`), is set up on entry (asm:Room_SetUpItemAndPushBlock)
and lies there from the first frame; walking over it takes it (40 of 40). Inside the ring it is reached with the
ladder (Link holds it from Dungeon 4). From the W hole Link stands at (`$20`,`$8D`) on the door row; the ring's W side
is one water square (X `$30`) on that row.

| Way, 40 seeds, 12 hearts | Through | Key | Hearts lost (mean, max) | Frames | Hearts lost to |
|---|---|---|---|---|---|
| today: down the W edge to Y `$BD`, E to X `$78`, S | 40 | 0 | 1.65, 7 | 303 | Blue shot 40, Red 18, Blue 8 |
| **E along the door row over the ring's W side, the key, back to X `$78`, down over the S side, S** | 40 | **40** | **1.38**, 3 | **312** | Red 25, Blue 20, (the ladder) 10 |
| down the W edge, E along the bottom row to X `$80`, up over the S side to the key and back | 40 | 40 | 3.60, 10 | 388 | Blue shot 64, Blue 36, ... |

Today's models cross 0x29 in 325 frames at 0.56 hearts (step 272, 27 deaths in 822 visits, entered with 6.2 hearts on
average). The key costs no fight and, by the door row, neither frames nor hearts against today's path; the room's N
key door stays kept out (game.yaml `keep_out: true`: one press spends a key).

### 3.3 No bombs on Vires after the magic sword

- The step-level `bombs:` mode (round 28; routes.yaml: "on this step the family's `items: {bombs:}` mode is this
  one") with **`tenth`**: BOMB only for the 10th damage-free kill, whose bomb kill arms a forced drop of 4 bombs (a
  gain, not a waste). `never` would refuse that too. The family stays `free` for Dungeon 4 (white sword, the owner:
  "fine beforehand").
- The Vire rooms of the route after the magic sword (step 260): **6:0x39 (step 273, fought for its shutters)** and
  **6:0x2d (step 277, fought for its key)**; levels 8 and 9 have Vire rooms (9:0x24, 0x37) off the route. Today the
  `vire` model bombs in both: 0x39 705 bombs in 795 runs (368 runs bombed), 0x2d 531 in 397 (306 runs; it picked up
  508, so it nearly paid its bombs back from drops). The magic sword kills a Vire in one hit and it does not split
  (enemies.md), so every one of those bombs could have stayed in the bag: the D6 triforce needs 0x28's wall's bomb and
  Dungeon 8 0x6e's and 0x4c's.

## 4. Dungeon 8, 0x4c without its key (`r31_d8_walks.py cross`)

Round 28's walk of the room (`r28_pols.cross_4c`) from the route's own `8_4cc_<chain>` (13 hearts, 8 bombs, 37
rupees, the bow and arrows). Link comes up the tunnel's left ladder at (`$70`,`$7D`) inside the block ring (layout
`$1C`); the eight Pols Voices are made inside the ring too, five or six on row `$9D`. **What stands in the way**: the
ring's blocks (out W one square, down to row `$9D`, E along it to X `$B0`, up that column to the top row, W to
(`$78`,`$5D`)) and the Pols Voices on row `$9D`. **Bombs: one**, at the N wall from (`$78`,`$5D`) facing N (0x4c is
`requires_bombs`; the chains bring 6 or 5).

| Play, 20 seeds | 13 hearts (the state) | 4.5 hearts (the chains' mean at step 294) |
|---|---|---|
| wall only, straight | 20/20, **2.40** (max 4), 269 frames | 20/20, 2.40 (max 4: 0.5 left), 269 |
| **wall only, one arrow when a Pols Voice is ahead on Link's line** | **20/20, 0 hearts, 273 frames, 1 rupee** | **20/20, 0, 273, 1 rupee** |
| by the corner key, straight (today's step without arrows) | 8/20, 12.0 | 0/20 (all dead) |
| by the corner key, with arrows (today's step played well) | 20/20, 0.50, 966 frames, 1.6 rupees (max 5) | 20/20, 0.50, 966 |

Today's models (step 294) take 1454 frames and gain 0.90 hearts there (drops), 0 deaths in 317 visits.

- **Must any be shot?** No (straight 20/20), but **one arrow is the difference between 2.4 hearts and none**: the
  first arrow, E along row `$9D` from the ring's SW corner, clears the row (round 28 §5.2). So pols-voice's
  `arrows: true` stays, and so does the rupee reserve: the room now spends 1 rupee, not 2.8.
- **`rupee_reserve: 8`** was "its arrows are paid from what Link holds above 8, and 8 shots stay for Ganon's silver
  arrow" (routes.yaml, decisions 2026-11-17, "the lead's number, not a measured need"). Without the key the room
  needs 1 arrow; the reserve's reason (Ganon) is unchanged and can stay as it is.
- Level 9's keys without 0x4c's: §2 (paid by 6:0x29's).

## 5. Dungeon 9 (`r31_d9_walks.py`)

- **0x66 out N.** The N door is a shutter (game.yaml `N: shutter: true`; level-9.md `$66` N "shutter"); the man checks
  the eight pieces only and both shutters open 18 frames after Link's first frame (d9-walks.md §2). From
  `9_66s_<chain>`: straight up X `$78` 20/20, 0 hearts, **250 frames** (today's way W: 372 frames scripted, 379 by the
  models). The fires at (`$48`,`$80`), (`$A8`,`$80`) did not touch a Link going straight up the middle.
- **0x56 crossed S to its W wall.** Level-9.md `$56`: 2 Bubbles, 2 Zols, 2 Like-Likes, the key on all dead (not
  wanted), W a bomb wall (`$55` E its other face; game.yaml 0x56 `bomb_walls: [W]`, no `specialist`, no
  `requires_bombs`). Layout `$19`: water across the whole room on rows Y `$6D` and `$AD`; from the S door Link crosses
  the lower one by the ladder.

| Way, 20 seeds, 15 hearts | Through | Hearts lost (mean, max) | Frames |
|---|---|---|---|
| up X `$78` over the water to Y `$8D`, W along it to (`$20`,`$8D`), bomb W | 20/20 | 0.80, 3 | 446 |
| **W along the bottom row to X `$20`, up over the water to (`$20`,`$8D`), bomb W** | 20/20 | **0.25**, 2 | **412** |

  The bomb: one, facing W at (`$20`,`$8D`) (the bag held 6; none written).
- **0x55 from E and the Lanmolas' clip.** Link comes in through the bombed E wall at (`$D0`,`$8D`). `Game.skip_available`
  (the engine's `zelda::clips::skip_target`, what the mask of a `clips: true` family reads), every grid point with
  the monsters away, offers CLIP on 27 points; nearest the E side: (`$A0`,`$9D`) N, (`$A0`,`$95`) N, (`$A0`,`$7D`) S,
  (`$A0`,`$75`) S, (`$A0`,`$6D`) S. Walked with the Lanmolas live, then MOVE onto the stairs (`r29_clips.clip_trial`):

| From the E arrival | Through (cellar) | Hearts lost | Frames |
|---|---|---|---|
| **down to Y `$9D`, W to X `$A0`, CLIP N** | **20/20** | **0** | **197** |
| up to Y `$7D`, W to X `$A0`, CLIP S | 20/20 | 0 | 226 |
| round to the S way's W lane (`$60`,`$9D`), CLIP N | 20/20 | 0.60 | 294 |
| round under the ring (`$70`,`$B5`), CLIP N | 20/20 | 0.20 | 273 |

- **Against today**: 0x66 W 379 + 0x65 543 (1.09 hearts, 0 deaths in 675) + 0x55 from S 174 frames by the models;
  the scripted 0x65 cost 4.5 hearts (the bomb wait among Wizzrobes, d9-walks.md). New, scripted: 250 + 412 (0.25
  hearts) + 197 = **859 frames, 0.25 hearts**.
- **The bomb budget on entering level 9**: the `post-d8` chain enters 9:0x76 with 6 (997 of 1000; its start state
  holds 7, 0:0x05's rock wall takes one); a full-route run leaves Dungeon 8 with 5 (224 of 248 `d8` runs) or 4, so
  4 or 3 after 0:0x05. Level 9's walls on the route: today 0x65 N, 0x26 E, 0x27 N, 0x17 N, 0x06 W, 0x20 N, 0x31 W,
  0x04 W (8, with 0x16's four bombs between); after phase IX **0x56 W instead of 0x65 N: the same count**, and the
  reserves (4 at 0x76, 3 from 0x55) mean the same.
- **0x56's family**: Like-Likes hold Link and `like-like` plays the level's other Like-Like rooms (0x14, 0x51, 0x41,
  0x03) with the hold in its training; its bomb mode is `tenth`, so the step needs `bombs: required` to lay the wall's
  bomb (as 8:0x6e for `boss-manhandla`).

## 6. The specification

What the route data, the engine and the tests must gain. Nothing here was changed on `main`; §1.4's scratch carried
the Dungeon 1 lines marked (scratch) and proved them with today's models.

### 6.1 routes.yaml (`d2first`)

Dungeon 1, replacing steps 46-55 (the comment above them too):

```yaml
# Dungeon 1 (the owner, Nov 19): in from 0:0x37, out S and straight back in (one grid step off the mouth first:
# asm:UndergroundExitType): the scroll out sets $EE to N (asm:Door_SetEnteringDoorwayOpened) and the entry's layout
# draws 0x73's N key door open and saves its flag (asm:Door_LayOut), so no key is spent, then or later; N through 0x63
# and 0x53 (its five Stalfos for the key), one bomb at 0x53's N wall into 0x43 from S.  0x72, 0x74, 0x52 and 0x42 are
# left out; the keys come out even (3 in, 3 out).
- {level: 1, room: '0x73', specialist: stalfos-keese-gel, entry: S, exit: S}
- {level: 0, room: '0x37', specialist: ow-octorok, entry: cave, exit: cave}
- {level: 1, room: '0x73', specialist: stalfos-keese-gel, entry: S, exit: N}          # (scratch: proved)
- {level: 1, room: '0x63', specialist: stalfos-keese-gel, entry: S, exit: N}          # (scratch)
- {level: 1, room: '0x53', specialist: stalfos-keese-gel, entry: S, exit: N}          # (scratch)
# The bow detour (Lee, Nov 9): 0x43 N through both north key rooms to the blade-trap room 0x22, ...
- {level: 1, room: '0x43', specialist: stalfos-keese-gel, entry: S, exit: N}          # (scratch)
```

Steps 56-67 stay as they are (the second visit of 0x43, `entry: N, exit: E`, keeps its name `1_43n`).

Dungeon 6:

```yaml
- {level: 6, room: '0x78', specialist: wizzrobe, entry: E, exit: N, drops: first}      # new key, §3.1 (optional)
- {level: 6, room: '0x29', specialist: wizzrobe, entry: W, exit: S, bomb_reserve: 1}   # its key taken (no `treasure: skip`)
- {level: 6, room: '0x39', specialist: vire, entry: N, exit: E, bombs: tenth}
- {level: 6, room: '0x2d', specialist: vire, entry: N, exit: W, bombs: tenth}
```

Dungeon 8:

```yaml
- {level: 8, room: '0x4c', specialist: pols-voice, entry: cave, exit: N, treasure: skip, rupee_reserve: 8}
```

Dungeon 9, replacing steps 313-315:

```yaml
- {level: 9, room: '0x66', specialist: stalfos-keese-gel, entry: S, exit: N}
- {level: 9, room: '0x56', specialist: like-like, entry: S, exit: W, treasure: skip, bombs: required}
- {level: 9, room: '0x55', specialist: lanmola, entry: E, exit: cave, bomb_reserve: 3}
```

The routes.yaml comments that describe the old ways (Dungeon 1's "fetches both key rooms' keys", the Dungeon 8 header's
"the corner key", the Dungeon 9 header's 0x65) are rewritten with them. `families`: `like-like` gains `(9, 0x56)`;
`stalfos-keese-gel` keeps its Dungeon 1 tags or loses `(1, 0x42)`, `(1, 0x52)`, `(1, 0x72)`, `(1, 0x74)` (off the route:
3:0x4b's precedent keeps a tag without states); `wizzrobe` the same for `(9, 0x65)`.

### 6.2 game.yaml

```yaml
- level: 1
  location: '0x73'
  exits:
    S:  # out of the level to the overworld: 0:0x37's cave mouth (asm:Mode_LeaveLevelToOW); one way, the way back
        # is 0x37's `cave` exit.  The route leaves by it once, for the door trick (phase IX).
      level: 0
      location: '0x37'
      leaves_level: true      # new key (or the convention "a dungeon exit to level 0 is one way"); keep_out unless the step leaves by it
    ...
- level: 1
  location: '0x53'
  ...
  bomb_walls: [N]
  requires_bombs: true    # phase IX: the N wall into 0x43 (zelda::cheats cheats one in)   (scratch: proved)
- level: 9
  location: '0x56'
  specialist: like-like
  requires_bombs: true    # phase IX: the W wall into 0x55
  ...
```

The convention comment of Dungeon 2 ("the entrance room's S door back to the overworld is no exit (leaving the dungeon
is never a route step)") gains "except 1:0x73's, the door trick's (phase IX)".

### 6.3 Start states (by name)

- New: `1_73s_v2_<c>` (0x73 S → N, after the trip out: `$EE` = 8; made from a played run or as `r31_d1_walks.py
  states` does), `0_37c_<c>` (0:0x37 out of 0x73 by S, cave → cave), `1_43s_<c>` (0x43 from S through the bombed
  wall), `9_56s_<c>`, `9_55e_<c>`.
- Renamed by the visit rule (docs/architecture.md "State names": a later visit from the same side adds `_v2`): today's
  `0_37c_<c>` (out of Dungeon 1 after its triforce) becomes **`0_37c_v2_<c>`**, because the new 0:0x37 `cave → cave`
  step comes first.
- Rebuilt (the keys and flags they carry change): `1_63s`, `1_53s` (3 keys, not 4; no 0x72/0x74 key flags) and the
  rest of Dungeon 1's (the same counts from 0x33 on); Dungeon 6 after 0x29 (`6_39n`, `6_3aw`, `6_08c`, `6_1dc`,
  `6_2dn`, `6_2ce`, `6_1cs`, `6_0cs`: one key more), `0_22c` and the leg to Dungeon 8, Dungeon 8's (`8_7es` … `8_2cs`:
  3 keys in) and `0_6dc`.
- Removed: `1_72e`, `1_73w`, `1_74w`, `1_73e`, `1_52e`, `1_42s`, `1_43w`, `9_65e`, `9_55s`.
- The chain segments (`eval_batch.SEGMENTS`, `scripts/experiments/r56_chains.sh`, AGENTS.md): `post-d1` starts at
  `0_37c_v2_0`; `d1`'s target "0:0x37" (`TARGETS`, `MILESTONES`) would now stop on the trick's first step out: it
  must be the step after the D1 triforce (`step:N`) or the second entry.
- `heart_value.json` is refitted (its rows are route steps: the scratch had to drop the six old rows to load).

### 6.4 The engine vocabulary

- **Leaving a level by its entry door is not round 26's "room left by its entry door".** That one (1:0x72 E → E,
  caves S → S, `d7.md` §9.3 "exists") is a room whose step leaves by the side it came in, inside one level. Leaving
  1:0x73 by S needs an exit game.yaml does not have and forbids by convention. What exists and needs nothing: the
  entry of the next step (`objectives::route_entry`: a level → the overworld is `cave`, the overworld → a level is
  `S`), and `exit_is_correct` (a step's next room comes from its game.yaml exit; across levels no side is checked).
  What is missing:
  1. game.yaml's S exit of 1:0x73 to 0:0x37 (above) and its load check (`GameMap::from_yaml` "exit S is not an exit
     of the room" until then), one way: F030 `exits_are_two_way_with_one_convention` skips it as it skips a wrap.
  2. The overworld step back in, `0:0x37 cave → cave`: Link stands on the mouth's foot with asm:UndergroundExitType
     = 2 and the mouth takes him only after a full grid step in mode 5 (walked: Up alone, 400 frames, nothing). The
     objective (or the executor's MOVE N there) must take him one grid step off first (8 frames of Down, then Up: 196
     frames in). [Not tried with the engine's objective: the scratch began on the far side of the trip.]
  3. A guard that the trick is not undone: from 0x73's S exit to 0:0x37's mouth no scroll and no cave (a MOVE E/W/N
     out of 0x37 on that step is a wrong exit, as any).
- `requires_bombs` on 1:0x53 and 9:0x56 (data only; the BOMB mask of `required` families and the path search follow).
- `bombs: tenth` on 6:0x39, 6:0x2d and `bombs: required` on 9:0x56 exist (round 28's step-level mode).
- `treasure: skip` on 8:0x4c and its removal on 6:0x29 exist; 0x29's key is a room item taken by walking (the
  objective's treasure target, as 0x58's and 0x2d's): the path to it crosses water (the ladder), as 0x39's door row.
- `drops: first` per step (§3.1) would be new: `RouteStep` gains the field, the env's reward config reads it as
  `reward.drops_first` for that step.

### 6.5 Tests that should pin it

- `crates/zelda/tests/` (beside `d6.rs` … `d9.rs`), a scripted walk `d1_door`: from `1_73s_0`, out S, one grid step,
  in; `$EE` = 8, 0x73's door flag bit 8 set on entry; N into 0x63 with 0 keys; 0x63's flag bit 4; and the control
  (no trip: the door holds).
- `game_map` tests: 1:0x73's S exit leads to 0:0x37 and is the only one-way level exit; `d2first`'s Dungeon 1 is the
  six steps above; the visits name `0_37c` (out by S) and `0_37c_v2` (after the triforce), `1_73s` and `1_73s_v2`.
- A key ledger test over routes.yaml and game.yaml: walking `d2first`'s steps, +1 for every key treasure a step takes
  (`treasure` not `skip`, the room's treasure `key`), −1 for every locked exit a step leaves by that no earlier step
  opened (from either side), 0x73 N opened by the trick; the count never below 0 at a locked exit, 1 at the end.
  With 6:0x29's `treasure: skip` put back it must fail at 9:0x63 W.
- `cheats`: `key_need` is 0 at 1:0x73 visit 2 (the door open) and no grant happens in the scratch's 300 runs (0
  cheated keys, §1.4).
- `bomb_mask`: BOMB offered to `stalfos-keese-gel` at 1:0x53's N wall spot and to `like-like` at 9:0x56's W wall spot
  on their steps, not elsewhere in those rooms; `vire` on 6:0x2d offers BOMB only for the 10th kill.
- `clips`: 9:0x55 entered from E offers `CLIP N` at (`$A0`,`$9D`) (§5).
- `objectives`: 6:0x29's objective leads to the key at (`$80`,`$90`), then S; 8:0x4c's straight to the N wall.

## For the owner

1. **Dungeon 1's door** (you: "if you leave the dungeon and reenter it, that door is unlocked, saving a key"). The game
   does it, and more: the door is unlocked for good from that entry on (its flag is saved), so a Wallmaster's return
   to 0x73 later finds it open too. What it needs: walk out until the screen scrolls, then straight back in (one step
   down off the entrance first); don't leave 0:0x37's screen or enter a cave in between. 380 frames, no hearts.
   Walked as said; no difference.
2. **The keys** (you: "my d1 route is correct ... they might not add up"; 0x29: "We need that key since we are skipping
   an extra in d1"). The new Dungeon 1 ends with 3 keys, the same as today: the two skipped key rooms paid for the two
   doors the trick and the bomb now open. The route stands as given. The key that is short is 8:0x4c's: skipping it
   alone would make level 9's 0x63 W door a cheated key; 0x29's key pays for it exactly, and Link ends the game with
   1 key as today. So your two Dungeon 6/8 changes belong together; the reason is Dungeon 8, not Dungeon 1.
3. **1:0x53's bomb.** It works (one bomb, 79 frames) but the route data does not yet say Link may bomb there; 12% of
   full-route runs reach Dungeon 1 with no bomb (the cheat ledger would then grant one, counted). Today's models
   played the new Dungeon 1 as it is (§1.4); 0x53 is slower than before (1240 frames against 608: the fight for the
   key, then the bomb).
4. **0x78** (you: "only costs a sword swing or two since they usually spawn next to the entrance"). True at full
   hearts, where the magic sword's beam reaches them: waiting a moment at the door and swinging at the first one on
   the row kills one in every run, the leader (all five) in one run of six, and drops a bomb pack in one run of four,
   for about a quarter second. Below full hearts they appear two to five squares away, out of the sword's reach:
   then it is a fight or nothing (a kill in 32 runs of 100). Walked as said at full hearts. A way for the models:
   §3.1 (a per-step "take the drops first"; no forced fight).
5. **0x2d's Vires.** As said, by `bombs: tenth` (the 10th-kill bomb stays, it is a pack of 4 coming back), and on 0x39
   too: it is the other Vire room after the magic sword, and today's model bombs there more (705 bombs in 795 runs).
   Before the magic sword (Dungeon 4) nothing changes.
6. **8:0x4c** (you: "You shouldn't need the key"). Right: out of the ring to the wall is 273 frames against 966 with the
   key. One arrow still matters there (0 hearts against 2.4 straight), so its family keeps its arrows.
7. **Dungeon 9** (you: "go north to 0x56 and bomb west. Skipping the more dangerous 0x65"). Walked as said: 0x66 N
   (the shutter the eight pieces open), 0x56 to its W wall, 0x55 from the E side with a block clip from there to the
   stairs (no Lanmola fight). 0.25 hearts and 859 frames by script against 1.09 hearts and 1096 frames by today's
   models; the same number of bombs.

## Scripts and commands

```sh
python scripts/experiments/r31_d1_door.py door [--seeds 20]     # §1.1 the variants
python scripts/experiments/r31_d1_door.py frames                # §1.1 the frame $EE takes bit 8
python scripts/experiments/r31_d1_walks.py bombspot             # §1.2
python scripts/experiments/r31_d1_walks.py cross43 [--seeds 20] # §1.3
python scripts/experiments/r31_d1_walks.py states OUT           # §1.4's post-trick 1_73s states
python scripts/experiments/r31_ledger.py steps --segments d1    # today's per-step frames, hearts, keys, bombs
python scripts/experiments/r31_ledger.py keys                   # §2: keys in and out at every step
python scripts/experiments/r31_ledger.py rooms FILE...          # §1.4: a chain output room by room
python scripts/experiments/r31_d6_walks.py ringleader [--seeds 100] [--hearts 10]   # §3.1
python scripts/experiments/r31_d6_walks.py key29 [--seeds 40]   # §3.2
python scripts/experiments/r31_d8_walks.py cross [--hearts 4.5] # §4
python scripts/experiments/r31_d9_walks.py n66|w56|scan55|clip55 # §5
```

§1.4's scratch: a checkout of this branch with the routes.yaml and game.yaml lines marked (scratch) in §6, its own
module (`maturin develop --release`), `heart_value.json` without the six dropped steps' rows, `TRIFORCE_STATES_DIR` a
copy of `triforce/states` with `1_73s_<c>` replaced by `r31_d1_walks.py states`' and `TRIFORCE_MODELS_DIR` this
checkout's promoted runs; then `triforce eval-batch NAME --chain "c<c>: --route d2first --from 1_73s_<c> --runs 100
--act model --stop-at step:62 --output ..."` for c = 0, 1, 2.
