# Round 28 walks: the leg from Dungeon 6 and Dungeon 8 in the running game (Nov 17)

`rest-of-game.md` "Gap list" (round 28) and `d8.md` §12, answered by playing the running game before anything of
round 28 is built. Every number here was observed (**walked**); how each item was staged, what was written to RAM and
the trials are in its section. Where chance is involved a number is from at least 20 trials, each loaded with the
game's own randomness for its seed. Scripts (`scripts/experiments/`): `r28_walks.py` (helpers, staging, the `Dodger`),
`r28_leg_walks.py` (item 1), `r28_d8_walks.py` (item 2), `r28_darknut.py` (item 3), `r28_manhandla.py` (item 4),
`r28_pols.py` (item 5), `r28_gleeok.py` (item 6), `r28_walk_scaffold.py` (a throwaway data root with a scouting
route `d8walk` so today's models can be played on the stretch), `r28_eval_summary.py` (one line per start state of
an evaluation), and round 26's `r26_replay_damage.py` (who hurt Link, from replays).

Link on these walks is what a run brings out of Dungeon 6 (`d6-walks.md` §7): 13 heart containers (full: the D6 piece
refills), the magic sword, the red candle, the bow and arrows, recorder, ladder, raft, boomerang, 2 keys, 1 bomb, 30
rupees (31 in, one arrow for Gohma), pieces 1-7, no ring, the small shield.

## The answers

| # | Item | Answer (one line) | Corrects |
|---|---|---|---|
| 1 | The whirlwind | Out of Dungeon 6 Link stands at (`$70`,`$7D`) facing down with the index at `$FB`. B on frames 0 and 153: `$FA`, `$F9`, picked up on frame 363, free in 0x3c at (`$80`,`$AD`) after 623 frames: **30 of 30, no heart lost**. The second play counts only on frames **152-216** after the first (65 frames). A miss is recovered by leaving S, coming back and playing once down, once up (20 of 20); two plays down after a miss land at 0x42 | `d8.md` §2.2 ("roughly 60-80 frames", inferred → walked) |
| 1 | The leg's rooms | Straight walks, 30 of 30 each: 0x3c 0, 0x4c 1.35, 0x4d 0.28, 0x5d 0.23 hearts. ow-octorok 40/40 in both its rooms (−0.06, −0.12), ow-moblin 0x4d 35/40 (−1.23), 0x5d 40/40 (−0.28). **The step out of 0x22 and the burn in 0x6d: 0 of 40 each** (the engine plays both when the buttons are pressed for it) | `rest-of-game.md` survival table ("none given") |
| 1 | The tree | Burns from X **`$B8`-`$CF`** facing W and **`$78`-`$8F`** facing E on row Y `$5D`; stairs 92 frames after B. Walking into the moving fire costs ½ heart; from frame 40 Link can hold Left (the tree stops him out of the fire's reach). Script: 30 of 30, no heart lost, 140 frames, one fire; **from 0x5d's left opening too** (30 of 30, 0.10) | `d8.md` §2.4 (`$B1`-`$CF`, `$71`-`$8F`); game.yaml 0x5d ("only the right one counts"); `level-8.md` ("not walked") |
| 2 | Dungeon 8's rooms | Monsters and counts as `level-8.md` in all ten rooms. Keys 2 → 3 (0x5e) → 2 (0x4e N) → 3 (0x4c's corner). Statues shoot on the room's **first frame** in 0x5e (one fireball per 39 frames) and 0x3e (one per 67). Bubbles: only the flashing `$2B` (sword off 157 frames, no heart). 0x3f's shutter shuts 10 frames after Link can move. The tunnel's right ladder leads back into 0x3f with all eight monsters | confirms `level-8.md` (trace → fought) |
| 2 | Blind crossings | Straight walks from 13 hearts: 0x4e 30/30 at **2.57** hearts, 0x3f 29/30 at **3.20** (13/30 from 3 hearts), the tunnel 0.50. Stepping aside (positions only): 0x4e 30/30 at 0.70, 0x3f 23/30. Fighting what comes near: 0x4e 19/20 at 0.25, 0x3f 19/20 at 0.90 | `d8.md` §3.2 ("1-2 touches") |
| 2 | The keyless pass | At 0x5e's E door while the shutters open: **with a key in hand a key is taken** (10 of 10); with none Link passes unpaid and the door is then open both ways (10 of 10). He is never shut into 0x5f | `d8.md` §5.4, `rest-of-game.md` softlocks ("stuck", inferred) |
| 3 | Blue Darknuts | 2 magic-sword hits from the side or behind (20/20), parried from the front; touch 2 hearts (20/20); bombs: 2, by round 19's facing rule (opposite 16/16 parried, else 40/40 hurt). Partial kills are banked and **outlast the six-room history**; only a level reload or a full clear past the history resets | `first-quest.md` "Banked kills"; `d8.md` §4.3 (code → walked) |
| 3 | The fights | **Scripted fight (positions only): 0x5e 30/30, 2.70 hearts, 1454 frames; 0x3e 30/30, 3.32 hearts, 2135 frames.** Fought in parts through the entry door: no better (19/20 each). **Today's darknut model: 0x5e 1/40, 0x3e 0/40** (all deaths, Darknut touches); with the planner 5/10 and 4/10 | `d8.md` §4.4, §10 #2 |
| 4 | Manhandla crossed | 0x6e straight up, one bomb at the wall: 30/30 at 1.33 hearts, 168 frames; stepping aside 29/30 at 0.47. trap-room 40/40 (−2.27) and gibdo 20/20 (−2.85) with exactly one bomb; **boss-manhandla 20/20 but 4.3 bombs a run, 3.15 of them cheated**. 0x2e straight up: 30/30 at 0.50 | `d8.md` §5.1, §11.3 (inferred → walked) |
| 5 | 0x4c | Pols Voice: 3 magic hits, touch 2, bombs nothing, **one arrow kills every one on its line for 1 rupee** (4 of 4, 20/20). Ring exit to the wall: straight 30/30 at 2.87; **with one arrow 30/30 at 0 hearts**. By the corner key: straight **9/30** (12.2 hearts); **with arrows 30/30 at 0.40 hearts and 2.8 rupees**. pols-voice model: 0/20 in every variant | `d8.md` §5.5 (code → walked) |
| 6 | Gleeok, four heads | 3 + 2 + 2 + 2 = 9 hits (20/20), **36 frames** apart at best (293-295 frames in all); attached heads never reach row Y `$9D` or lower; flying heads: up to 3, 0.96 px a frame, cannot be hurt, 1 heart, gone with the last head; one fireball at a time, a new one a median of 61 frames after the last, 1.7 px a frame, 1 heart. **Leaving S resets everything** (4 heads, HP `$A0`). **Scripted fight: 30/30 from 13 hearts at 1.93 hearts; 26/30 from 6; 19/30 from 4.** boss-gleeok: 0/20 | `d8.md` §6 (32 frames; rows `$AD`/`$BD`; the reset inferred) |
| 7 | Ledgers | Bombs 1 in, 0x6e's wall; 0x4c's wall is cheated unless 0x3f's 4 bombs are fought for (they also pay 0:0x05). Keys 2 in, 3 out. Rupees 30 in, 1-5 for arrows. **Hearts: the best plays seen lose 10.7 of 13 and take 2.5 back from the Darknuts' drops; Link reaches Gleeok with about 6.8** and leaves the room with about 4. Out with 14 of 14 | `rest-of-game.md` survival budget |
| 8 | Engine | No missing model input found. Missing: the burned tree as a secret (`how: burn` is read by nothing), ARROW for 0x4c, `keep_out` on two key doors, the bomb rule of 0x6e's family, and five families that cannot play their rooms today (§8) | — |

**The ten findings that most change round 28's plan**

1. **Both Darknut rooms are won by a small script and lost by every model.** A fighter that reads positions and
   facings only (stab what stands in reach and does not face the blade, step aside from what is predicted to touch
   Link, else walk to a Darknut's flank) clears 0x5e 30 of 30 at 2.70 hearts and 0x3e 30 of 30 at 3.32. The darknut
   model, trained on Dungeon 5's Blue Darknuts, wins 1 of 80 (white sword: 0 of 80; four bombs written: 12 of 80;
   goriya, like-like: 0); 517 of 530 hearts it lost were Darknut touches, not statue fire. The rooms are open floors;
   its home rooms have blocks.
2. **Arrows turn 0x4c from the dungeon's worst walk into its cheapest.** The eight Pols Voices are made inside the
   block ring, most of them on the row Link leaves by. One arrow along that row: 30 of 30 crossings without a heart
   lost. With arrows the corner key costs 0.40 hearts and 2.8 rupees (30 of 30); without them 9 of 30 arrive.
3. **Gleeok with four heads is a script-sized fight from full hearts and a coin toss from four.** Standing on row
   `$9D` (out of every attached head's reach, in the blade's) and stabbing up when the heads' timer is 0, stepping
   aside from fireballs and flying heads: 30 of 30 at 1.93 hearts from 13, 26 of 30 from 6, 19 of 30 from 4. The
   fireballs do the damage (46 of 58 hearts), one every 61 frames.
4. **The survival budget closes, by about four hearts, if every room is played as well as its best script.** The
   best plays seen lose 10.7 of 13 hearts and the two Darknut fights take 2.5 back from drops (hearts and fairies):
   Link reaches Gleeok with about 6.8 and no beam, where the script wins 26 of 30. With only plays that never failed
   (a straight walk through 0x6e, stepping aside in 0x4e, the darknut model's 0x3f) he reaches Gleeok with about 3.5.
5. **Today's models cannot play six steps of the stretch**: the two Darknut rooms (1 of 80; 9 of 20 with the
   planner), 0x4c (0 of 60), Gleeok (0 of 20, also with the planner, also from 5 hearts), and the two new overworld
   steps (the whirlwind out of 0x22, the tree of
   0x6d: 0 of 40 each). The rest transfers: ow-octorok 100%, ow-moblin's crossings 88-100%, the tunnel 100%, 0x6e
   100% by any `required` family, 0x3f 98-100% by goriya or darknut.
6. **0x6e wants a `required` family, not boss-manhandla.** trap-room and gibdo cross it with exactly one bomb (60 of
   60); boss-manhandla (`free`, in a `requires_bombs` room) fights the Manhandla with 4.3 bombs a run, 3.15 of them
   cheated in.
7. **Fighting a Darknut room in parts buys nothing.** The banked kills are real and outlast the room history (walked),
   but every re-entry makes the rest anew with the statues firing from frame 0; the same script leaving S after
   every three hearts lost wins 19 of 20 instead of 30 of 30 at the same cost.
8. **The whirlwind and the tree need no macro and no new input**, only a family that presses the buttons. Through
   the env as built: RECORDER, RECORDER, WAIT lands in 0x3c as `exit-correct` (20 of 20 seeds); MOVE to the window,
   CANDLE W, WAIT, MOVE W enters Dungeon 8 as `exit-correct` (19 of 20). The tree has no objective target yet.
9. **0x3f's four bombs come almost for free with a fighting crossing** (12 of 20 crossings that fight what comes near
   end with the room cleared and the bombs taken), and they are the only honest bombs for 0x4c's wall and for
   0:0x05's rock wall on the way to Dungeon 9.
10. **The keyless pass cannot strand Link.** With keys it takes a key (a wrong exit that costs one of Dungeon 9's);
    with none it opens the door for good.

## How the walks were staged

- **The leg.** A chain-made state of the D7 triforce step (`triforce/states/7_2bw_0.state`: its
  `TeleportingLevelIndex` `$FB` was left by the route's own two plays at 0x0b) is walked onto the D7 piece, along
  round 27's leg (the plain way, no sword stop) with the monster slots emptied every frame, and into level 6 by
  0x22's mouth. There the inventory above is written (pieces 1-5 and 7), `triforce/probe.py`'s room hop puts Link
  into 6:0x0c from S, he walks onto the piece and **the game itself** puts him out in 0:0x22: 13 of 13 hearts,
  `InvTriforce` `$7F`, index `$FB` (`r28_walks.out_of_d6`). From there the leg is walked once (the whirlwind ridden,
  monsters emptied, the tree burned) and each room's pre-entry frame saved (`r28_walks.leg_states`). Every trial loads
  one of these states with `randomize_rng` and its seed: the monsters are made fresh and are live.
- **Dungeon 8.** `triforce/probe.py` from `start.state` (`r28_walks.stage_d8` / `d8_state`): the inventory, 13 full
  hearts, the keys and bombs of the draft's ledger at that step, the `WorldFlags` of the doors opened before (0x6e N,
  0x4e N, 0x4c N) and the key flags of 0x5e and 0x4c. The tunnel and 0x4c are reached by play from 0x3f
  (`d8_cellar_states`; 0x4c's state holds the one bomb the no-drop run would have cheated in).
- **Written to RAM in single measurements** (said again where used): Link's position and facing beside a monster for
  hit counts and touch tests; hearts written back each frame where a count needs him alive; monsters "put into the
  death state" (`ObjMetastate` `$10`, what a kill sets) where a room's doors or memory are studied, not its fight;
  the probe's room hop (the room id written) to enter other rooms of the level for the history tests.
- **The scripts that play** read what a player sees: positions, facings, the heads' flash. `Dodger`
  (`r28_walks.py`) carries every object forward 14 frames along its last frames' motion (a fireball that has not
  moved yet toward where Link stood when it appeared) and takes the move with the fewest predicted overlaps, ties by
  the distance left. No look-ahead in the emulator.
- **Models.** `r28_walk_scaffold.py` copies the data with the route `d8walk` (extends `d6`: the draft's steps from
  0x22 to Dungeon 8's piece), the rooms tagged, `requires_bombs` on 0x6e and 0x4c, `ram_randomization: {}` (the
  states' own 13 hearts, keys and bombs), start states from the staging above. One scaffold per assignment of
  families to rooms. Promoted runs of round 25 (converted, 1.5.1.0), each model's act mode `temperature:0.5+relax:16`,
  20 episodes a state (families with clips: 20 with the switch on and 20 off); the planner `+plan:16:1`, 10 a state.
  All through `triforce eval-batch` (queued jobs).

## 1. The leg from 0x22 to Dungeon 8's stairs (walked)

### 1.1 The whirlwind out of 0x22

`r28_leg_walks.py whirlwind --seeds 30`. Link comes out of Dungeon 6 at (`$70`,`$7D`) **facing down**; the room's Red
Lynel is made at one of a dozen spots on rows `$AD`-`$BD` (20 seeds).

| Frame (first B = 0) | What happens |
|---|---|
| 0 | B: index `$FB` → `$FA`; the whirlwind is made at X 0 on Link's row and stands; everything is frozen for the tune |
| 152-216 | the window of the second play; B on 153: `$FA` → `$F9`, a second freeze of 152 frames |
| 305 | the whirlwind starts to move (2 px a frame) |
| 363 | it picks Link up |
| 623 | Link is set down and free in 0x3c at (`$80`,`$AD`), facing right |

- **30 of 30 seeds land in 0x3c with no heart lost** (plays on frames 0 and 153, Link not moved). The two tunes freeze
  the Lynel for 304 of the 363 frames. Standing on the same spot 700 frames without playing costs hearts in 13 of 30
  seeds (up to 9): the play on the first frame is what makes 0x22 safe.
- **The second play's window** (seed 0, every frame tried): on frames 100-151 after the first the button is not read
  (the index does not step; Link lands at 0x74, Dungeon 3); on **152-216** it steps and re-aims (0x3c); from 217 on
  Link is already carried (pickup at 216 without a second freeze) and the index does not step: 0x74.
- **Other plays from the same spot** (index `$FB`, pieces 1-7): one down 0x74; three down 0x37; one up or one right
  0x0b; down then up 0x45; one left 0x74 (left steps as down, right as up).
- **A miss.** Link stepping off the row before the whirlwind arrives (12 frames of Down from frame 310), or invincible
  as after a hit (the timer written while it passes): it passes, the index stays `$F9`, the room makes no second
  one; a third play in the same visit steps the index to `$F8` and brings nothing. **Recovery**: out S into 0x32, back
  in (Link at (`$70`,`$DD`)), one play facing down (`$F8`, the whirlwind), a frame of Up, one play facing up (`$F9`):
  0x3c, 20 of 20 (monsters emptied for this test). Two plays facing down instead: `$F8`, then `$F6` (level 8 is not
  held and is skipped): 0x42.
- **Through the engine as built** (the scaffold's `d8walk`, `_native.Env`): RECORDER is in the mask on Link's first
  frame, twice; after the second it is not; RECORDER, RECORDER, then WAIT 32 until the room changes ends
  `exit-correct` in 0x3c, 663 frames, 13 hearts, 20 of 20 seeds. **ow-lynel zero-shot** (`items: {recorder: true}`
  written in the scaffold): 0 of 40, 38 walk out of the room.

### 1.2 The rooms

`r28_leg_walks.py rooms --seeds 30`. A "straight walk" goes from point to point and walks back onto its line after a
knockback; it never fights. Hearts are the mean over 30 seeds (maximum in brackets). Model columns: success of 40 (20
with clips on, 20 off) and mean heart change.

| Room, entry → exit | Monsters seen | Straight walk: path | Hearts lost | Frames | Family model |
|---|---|---|---|---|---|
| 0x22 cave → whirlwind | 1 Red Lynel | two plays, no step | 0 | 623 | ow-lynel 0/40 |
| 0x3c whirlwind → S | 1 fast Blue Octorok | from (`$80`,`$AD`): W to X `$70`, down | 0 | 149 | ow-octorok 40/40, −0.06 |
| 0x4c N → E | 2 Blue + 2 fast Red Octoroks at once | down X `$70` to Y `$8D`, E | 1.35 (5.5) | 348 | ow-octorok 40/40, −0.12 |
| 0x4d W → S | 4 Blue Moblins | E to X `$50`, down to Y `$BD`, E to X `$C0`, down | 0.28 (1) | 289 | ow-moblin 35/40, −1.23 (5 wrong exits) |
| 0x4d W → S by the top row | | to X `$10`, row `$5D`, X `$E0`, row `$BD`, X `$C0` | 0.88 (2) | 431 | |
| 0x5d N → S (right opening) | 4 Blue Moblins | straight down X `$C0` | 0.23 (0.5) | 204 | ow-moblin 40/40, −0.28 |
| 0x5d N → S (in right, out left) | | down to Y `$8D`, W to X `$30`, down | 0.42 (1.5) | 310 | |
| 0x6d N → stairs | 4 Red Moblins | §1.3 | 0 | 140 | ow-moblin 0/40 (40 wrong exits) |

- **Sum of the straight walks: 1.86 hearts**; with ow-octorok in 0x4c 0.63.
- 0x4c: `d8.md` §2.3 gives 3 fast Red and 2 Blue Octoroks (list `$6A`, count 5); four were on screen at once in every
  one of 30 seeds watched for 240 frames (the fifth was not looked for further).
- 0x4d's shop tree at (`$D0`,`$A0`) is not on either walk's line; nothing here burned it.
- ow-moblin's 5 wrong exits in 0x4d are exits by another edge (W → S is new to it); its hearts there (1.23) are
  Moblin touches.

### 1.3 The candle on 0x6d's tree

`r28_leg_walks.py tree`. Link from N at X `$C0`, on row Y `$5D`; the tree is the tile object `$64` at (`$A0`,`$60`).

- **Where it burns** (Link written to every X, facing written, one use of the candle each, the game put back after
  each): facing W from X **`$B8` to `$CF`**, facing E from X **`$78` to `$8F`**, every X between (24 each); facing W
  in the column X `$C0` from Y `$50` to `$6F`. The fire comes to rest at Link's X ∓ `$20`: it must rest at X
  `$98`-`$AF`. `d8.md`'s window from the code (`$B1`-`$CF`) is 7 px too wide on the tree's far side; the draft's
  `$B8`-`$C0` is exactly the near edge.
- **Frames.** B on frame 0; the fire is seen from frame 1 at X − `$11`, moves half a pixel a frame, stands from frame
  30 at X − `$20`; **the stairs' four tiles (`$70`-`$73`) and room flag `$80` on frame 92**; the fire is gone the same
  frame.
- **What the flame does to Link.** Standing still: nothing. Walking W at once from X `$C0`: ½ heart on frame 15 (he
  catches the moving fire at X `$B5`) and a knockback. Walking W from frame 40: nothing: the tree stops him at X
  `$B0`, 16 px from the standing fire, and he walks onto the stairs when they appear. So: B, 30-40 frames still, then
  Left.
- **Onto the stairs**: 23 frames of Left from X `$C0` after they show; Link plays in 8:0x7e at (`$78`,`$CD`) 122
  frames later.
- **The red candle again**: a second fire 20 frames after the first is made (slots `$10` and `$11`); a third B does
  nothing while both burn. `HasUsedCandle` is set but the red candle does not read it.
- **With the Moblins live** (`rooms`): down to row `$5D`, a frame of Left (X `$BE`, facing W), B, still while a fire
  burns, then W: **30 of 30, no heart lost, 140 frames, one candle use each**. From 0x5d's **left opening** (in at X
  `$30`): E along row `$5D` to X `$78`, B facing E, the same: 30 of 30, 0.10 hearts, 202 frames. game.yaml's "only the
  right one counts" is not what the game does; the right one is shorter by one room's width of walking.
- **Through the engine as built**: `how: burn` is read by nothing, so the step has no target (the observation's
  `target` block says none). With ow-moblin's `items: {candle: true}` CANDLE is in the mask at the window; MOVE S to
  row `$5D`, MOVE W 4, CANDLE W, WAIT 32 four times, MOVE W: `exit-correct` into 8:0x7e in 19 of 20 seeds (one was
  knocked off the row and walked out W). **ow-moblin zero-shot: 0 of 40** (it leaves by an edge).

## 2. Dungeon 8, every room of the draft's way (walked)

`r28_d8_walks.py`. Monsters made in every room match `level-8.md`'s rooms table (20 seeds each): 0x7e none; 0x6e one
Manhandla (five objects: hands in slots 1-4 at HP `$40`, the base in slot 5); 0x5e 5 Blue Darknuts; 0x4e and 0x3f 2
Bubbles, 1 Blue and 2 Red Darknuts, 3 Gibdos; 0x3e 6 Blue Darknuts; 0x2f 4 blue Keese; 0x4c 8 Pols Voices; 0x3c the
Gleeok (type `$45` in slot 1; the heads live in the neck arrays); 0x2c none.

### 2.1 Room by room

Script: the play named, 13 hearts, the magic sword, 20-30 seeds. Model: the best zero-shot family of those tried,
success of 20 or 40, mean heart change.

| # | Room, entry → exit | Link first at | Script | Script result | Models zero-shot |
|---|---|---|---|---|---|
| 1 | 0x7e S → N | (`$78`,`$CD`) | straight up | 30/30, 0 hearts, 250 frames | stalfos-keese-gel 40/40 |
| 2 | 0x6e S → N, bomb wall | (`$78`,`$CD`) | §4 | straight 30/30, 1.33; stepping aside 29/30, 0.47 | trap-room 40/40, −2.27, 1 bomb; gibdo 20/20, −2.85, 1 bomb; boss-manhandla 20/20, −1.90, 4.3 bombs |
| 3 | **0x5e S → N, fight, key** | (`$78`,`$BD`), through the wall | §3 | **30/30, 2.70 (max 8.5), 1454 frames** | **darknut 1/40**; planner 5/10 (wins −4.4); goriya 0/40; like-like 0/20 |
| 4 | 0x4e S → N key door | (`$78`,`$CD`) in the doorway | straight up X `$78`; stepping aside; fighting what comes near | 30/30, 2.57 (max 5); 30/30, 0.70; 19/20, 0.25 | goriya 35/40, −5.88; gibdo 14/20, planner 10/10 at −1.5; darknut 15/40 |
| 5 | **0x3e S → E, fight** | (`$78`,`$CD`) in the doorway | §3 | **30/30, 3.32 (max 10.5), 2135 frames** | **darknut 0/40**; planner 4/10 (wins −3.25); goriya 0/40; like-like 0/20 |
| 6 | 0x3f W → stairs | (`$20`,`$8D`), shut in after 10 frames | straight E on the door row; stepping aside; fighting what comes near | 29/30, 3.20 (one death); 23/30, 2.03; 19/20, 0.90 | darknut 40/40, −2.92, 318 frames; goriya 39/40, −2.67; gibdo 7/20, planner 9/10 at −4.3 |
| 7 | 0x2f tunnel | right ladder top (`$C0`,`$5D`), cellar mode 9 | down to Y `$BD`, W to X `$30`, Up | 30/30, 0.50 (Keese), 294 frames in the cellar | equipment-room 20/20, −0.05 |
| 8 | 0x4c stairs → N, bomb wall (key) | (`$70`,`$7D`) inside the ring, facing down | §5 | wall only, one arrow: 30/30, 0; with the key and arrows: 30/30, 0.40 | **pols-voice 0/20** (key), 0/20 (wall only), 0/20 (ARROW on); gibdo 4/20; trap-room 0/40 |
| 9 | **0x3c S → N, Gleeok** | (`$78`,`$BD`), through the wall | §6 | **30/30, 1.93 (max 6), kill at frame 982** | **boss-gleeok 0/20**; planner 0/10; from 5 hearts 0/20 |
| 10 | 0x2c S → piece | (`$78`,`$CD`) | straight up | 30/30; the piece after 42 frames of Up; out in 0:0x6d at (`$60`,`$5D`) 641 frames after | stalfos-keese-gel 40/40 |

- **0x4e and 0x3f from fewer hearts** (the same plays): straight E through 0x3f from 6 hearts 29/30, from 3 hearts
  13/30; stepping aside in 0x4e from 3 hearts 27/30, in 0x3f from 6 hearts 22/30; fighting what comes near from 6
  hearts 0x4e 17/20, 0x3f 19/20.
- **Who hurts in the crossings** (straight walks, hearts over 30 runs): 0x4e Gibdos 42, Blue Darknut 16, Red 15;
  0x3f Gibdos 51, Red Darknuts 27, Blue 16. A Bubble's touch left the sword off on leaving 0x3f in 13 of 30 straight
  walks (it would arrive so in the tunnel, where the Keese then cost more; no room after it needs the sword at once).
- **0x3f's corridor** (`d8.md` §5.2: a Darknut facing W in it cannot be hit): not isolated. The fighting crossing
  timed out in 1 of 20 and the full clear in 4 of 20 (6000 frames); what held them was not read.
- **gibdo, goriya and darknut as crossers**: gibdo dies in 6 of 20 (0x4e) and 13 of 20 (0x3f) (Gibdos 175 of 345
  hearts, Red Darknuts 87, Blue 82); darknut is perfect in 0x3f and stuck or dead in 0x4e (9 stuck, 14 deaths of 40).

### 2.2 Doors, keys, shutters (`doors`)

Monsters put into the death state where a door's trigger is studied; frames are counted from that write.

- **0x5e**: the room is flagged empty on frame 19; the key shows at (`$80`,`$90`) with it; the **N shutter is open on
  frame 27, the W shutter on frame 36**. The key is taken at (`$78`,`$8D`) walking down column `$78` (keys 2 → 3,
  flag `$34`). N from (`$78`,`$5D`): Link plays in 0x4e at (`$78`,`$CD`) 166 frames later.
- **0x5e E** (key door, off the route): walking Right along the door row with 3 keys, a key is taken on the frame
  Link reaches the door spot (`$D0`,`$8D`); he plays in 0x5f at (`$10`,`$8D`) 266 frames after setting out.
- **The keyless pass** (Link written onto the E door spot, the five put into the death state, Right held from 0 to
  18 frames after the room is flagged empty, 10 starts each). The N shutter's command runs 8 frames, then 2 frames
  without a command, then the W shutter's 9.
  - With 2 keys: Link is let 0-12 px into the doorway unpaid, then **a key is taken in the 2-frame gap** and the door
    opens as usual: keys 2 → 1, 10 of 10. The pass never completes.
  - With 0 keys: he is not stopped, walks through the shut door during the W shutter's command and plays in 0x5f, the
    door's flag set on 0x5f's side only. Left from there (at once or 120 frames later): back into 0x5e, the door now
    open on both sides, no key spent, 10 of 10. **He is not shut in.**
- **0x4e N** (the route's key door): the key is spent on the first frame of Up at (`$78`,`$5D`) (3 → 2), the door is
  open on frame 9, Link plays in 0x3e at (`$78`,`$CD`) after 175 frames; 0x3e's S side is open (flags `$x8`/`$x4`).
  With 0 keys: 200 frames of Up, nothing. **0x4e W** (off the route): Left at (`$20`,`$8D`) takes a key on the first
  frame; Link plays in 0x4d (a Gohma) 213 frames later.
- **0x3e**: flagged empty on frame 19, the **E shutter open on frame 27**. The door row is blocked by the statue at X
  `$90` (Link stops at X `$80`): round it by row `$9D` (down at X `$80`, E to X `$A0`, up, E), 82 frames from
  (`$78`,`$8D`) to (`$D0`,`$8D`). E: Link plays in 0x3f at (`$20`,`$8D`) 216 frames later; **the shutter behind him is
  shut 10 frames after that** and holds against 200 frames of Left with the room's monsters alive.
- **0x3f**: its three Darknuts and three Gibdos dead (the two Bubbles alive): flagged empty on frame 19, the W
  shutter open on frame 27, **4 bombs** taken at (`$78`,`$8D`) on the door row (0 → 4).
- **0x7e S**: 109 frames of Down from the first spot start the change; Link plays in 0:0x6d at (`$60`,`$5D`) 175
  frames later. Back in by the stairs: 0x5e's banked 2 are 0, the room holds 5 Darknuts again; the bombed wall stays
  (flags 0x6e `$x8`, 0x5e `$x4`).
- **First spots**: an open doorway or a key door leaves Link in the doorway (Y `$CD`, X `$10` or `$E0`); a bombed
  wall or a shutter walks him in to the floor's first square ((`$78`,`$BD`), (`$20`,`$8D`)).

### 2.3 What a room remembers (`history`)

0x5e (from S, 5) and 0x3e (from S, 6); Link kept alive, kills by the death state, the room left S and entered again.

| Step | 0x5e | 0x3e |
|---|---|---|
| 2 killed, out S and back | 3 Darknuts; banked 2, flag bits 6-7 = 2 | 4; banked 2 |
| 2 more, out and back | 1; banked 4 | 2; banked 4 |
| then eight other rooms entered (the room is out of the six-room history) and back in | **1** (banked 4 kept) | **2** (banked 4 kept) |
| the rest killed, out S and back (in the history) | 0; the shutters open 11 and 20 frames after Link can move; banked `$0F`, bits 6-7 = 3 | 0; the E shutter after 11 frames |
| then other rooms entered until the room is out of the history, and back in | **5 again**; banked 0, bits cleared | **6 again** |
| out of the level by 0x7e S and back (2 banked before) | 5 again; banked 0 | not run |

- **What a re-entry gives**: the Darknuts left are made anew at spawn spots (0x5e after 2 kills: (`$83`,`$5D`),
  (`$95`,`$8D`), (`$B6`,`$5D`) 60 frames in), Link stands on the entry spot again, the shutters are shut, the statues
  fire from the first frame (3 fireballs in flight 60 frames in).
- The "other rooms" were entered by the probe's hop (the room id written while Link leaves); no hop's source room was
  the room under test (a hop that leaves "from" 0x5e banks the room on screen's kills under 0x5e's id: an artifact
  the first version of this test ran into).

### 2.4 The tunnel 0x2f (`tunnel`)

- From 0x3f's stairs Link is on the **right** ladder's top, (`$C0`,`$5D`), in mode 9; 4 blue Keese.
- Down to Y `$BD`, W to X `$30`, up the **left** ladder: 30 of 30, 0.50 hearts (max 1.5), 294 frames; **0x4c at
  (`$70`,`$7D`)**, inside the block ring, facing down.
- Up the right ladder at once: **0x3f again at (`$70`,`$7D`)** after 117 frames, the W shutter shut, all eight
  monsters there (the room is in the history with nothing banked).
- From 0x4c: Right from the arrival spot is the stairs after 26 frames (Link on the left ladder's top); Up at once:
  0x4c at (`$70`,`$7D`) again, its eight Pols Voices made anew.

### 2.5 Statues and Bubbles (`rooms`, `bubbles`)

- **Statues.** 0x5e (four corner sources) and 0x3e (two central ones): a fireball `$55` exists on the room's first
  frame in 20 of 20 seeds each. Over 600 frames: 15.4 fireballs in 0x5e (one per 39 frames), 9.0 in 0x3e (one per
  67). Standing on the entry spot for 150 frames costs 0.78 hearts in 0x5e (max 2.5), 0 in 0x3e. 0x7e's and 0x2c's
  statues never shot (20 × 600 frames).
- **Bubbles.** 0x4e and 0x3f hold two each, both the flashing type `$2B` in 20 of 20 seeds (no red `$2C`, no blue
  `$2D` on the way). Link written onto one: no heart lost, `SwordBlockedLongTimer` `$10`, back at 0 after 156-157
  frames (20 of 20). 30 frames after the touch the mask offers BOOMERANG, ARROW and MOVE, no SWORD or BEAMS, and A
  brings no sword out.

## 3. Blue Darknuts: 0x5e and 0x3e (walked)

`r28_darknut.py`.

### 3.1 What hurts them and what they do (`measure`; one Darknut kept, Link written beside it, 20 seeds)

| Test | Result |
|---|---|
| Magic sword from the side | 40 of 48 swings took HP; dead after 2 that landed, 20 of 20 |
| from behind | 40 of 51; dead after 2, 20 of 20 |
| from the front | 13 of 157 took HP (it had turned between the write and the blade); 18 of 20 alive after 8 swings |
| Magic beam, 13 of 13 hearts, 48 px away | side 6 of 20, behind 9 of 20, front 1 of 20 (it walks off the line while the beam flies) |
| A touch, 13 containers, no ring | 2 hearts, 20 of 20 |
| Bombs (8 written): its facing as the blast reaches it against Link's when he laid the bomb | opposite: parried (a flash, no HP) 16 of 16; perpendicular: HP 22 of 22; the same: HP 18 of 18; dead after 2 blasts that landed, 20 of 20 |

Round 19's facing rule for Red Darknuts holds for Blue ones unchanged. The route brings no bomb to either room.

### 3.2 The scripted fight (`clear --seeds 30`)

`Fighter` (in `r28_darknut.py`), a frame at a time, positions and facings only:

1. If a Darknut stands in the blade's reach (10-27 px ahead on Link's row or column, at most 9 px off it; at full
   hearts the beam's reach) and does not face the blade, and nothing is predicted to touch Link in the next frames:
   face it (a frame of the direction), then A; nine frames still.
2. Else, if a drop lies in the room and Link is not at full hearts: walk to it.
3. Else walk to the nearest square beside or behind a Darknut (18 px off it), by `Dodger` (which steps aside from
   Darknuts and fireballs first).
4. When none is left: the key (0x5e), the exit (0x3e: round the statue by row `$9D`).

| Room, hearts in | Cleared and out by the route's door | Hearts lost (mean, max) | Lost to | Frames (median) | To the last kill | Swings |
|---|---|---|---|---|---|---|
| 0x5e, 13 | **30/30** | 2.70, 8.5 | statue fire 45, Darknuts 34 (hearts over 30 runs) | 1454 | 1339 | 21.7 |
| 0x5e, 8 | 29/30 (one death) | 3.23, 8 | fire 55, Darknuts 42 | 1572 | 1456 | 12.1 |
| 0x3e, 13 | **30/30** | 3.32, 10.5 | Darknuts 68, fire 31 | 2135 | 1948 | 28.5 |
| 0x3e, 8 | 28/30 (two deaths) | 3.63, 8 | Darknuts 74.5, fire 34.5 | 1936 | 1775 | 13.3 |

- **Hearts down on leaving** (the losses less the drops the fight took; a Blue Darknut drops hearts and fairies and
  the script walks to a drop when Link is not full), mean over all 30 runs: 0x5e from 13 hearts **1.67**, from 8
  hearts 1.03 (the one death counted as 8); 0x3e from 13 **1.82**, from 8 1.07 (two deaths counted). The fights give
  back 1.0-1.5 hearts from 13 and more than 2 from 8.
- From 13 hearts the first swings are beams (twice the swings, the same kills); the beam is gone with the first hit
  taken, usually a statue fireball.
- In one of the 30 clears of 0x5e the script left without the key (it did not reach it in its 600 frames).

### 3.3 Fought in parts (`parts --seeds 20`)

The same fight, left by the entry door (S) whenever three hearts are lost since Link came in, and entered again at
once (0x6e's Manhandla and 0x4e's eight are what waits outside; the script turns round in the doorway).

| Room | Cleared and out | Hearts lost (mean, max) | Visits |
|---|---|---|---|
| 0x5e | 19/20 (one stuck in the doorway) | 2.73, 8.5 | one 16, two 4 |
| 0x3e | 19/20 (one stuck) | 2.70, 7.5 | one 14, two 5, three 1 |

No gain over the fight in one go (2.70 and 3.32, 30 of 30): a retreat throws away position, and the statues fire at
once on re-entry. As a rule to leave "when low" it is not needed either: the fight from 8 hearts wins 29 and 28 of 30.

### 3.4 Today's models (20 a state; darknut and goriya 40 with clips on and off)

| Model, variant | 0x5e | 0x3e |
|---|---|---|
| darknut (r75-d537-c5), magic sword, 0 bombs | **1/40** (38 deaths; median 839 frames) | **0/40** (median 504 frames) |
| darknut with the planner (`+plan:16:1`, 10 runs) | 5/10, wins −4.4 hearts, 1673 frames | 4/10, wins −3.25 |
| darknut, white sword | 0/40 | 0/40 |
| darknut, 4 bombs written (it used 3.3 and 3.0) | 12/40, wins −7.75 | 0/40 |
| goriya | 0/40 | 0/40 |
| like-like | 0/20 | 0/20 |

- From the replays (20 runs a room): of 530 hearts lost, **517 by Blue Darknut touches**, 9.5 by statue fireballs.
  The model does not die to what is new (the statues); it walks into Darknuts.
- Its home rooms for comparison (its own evaluation file, the family's randomisation): the Olive Garden 5:0x05
  82-87% at −2.6 to −2.9 hearts, 5:0x64 87-91%.
- What differs here was not separated further: open floors (layouts `$24`, `$23`) against the Garden's and 0x64's
  blocks, no bomb, no staged kills, 13 containers. The white-sword run rules the sword input out.

## 4. Manhandlas crossed, not fought (walked)

`r28_manhandla.py`. 0x6e: the Manhandla is made at the top of the room ((`$A0`,`$5D`) on the first frame, the base
near (`$59`,`$5D`) twenty frames later), on the row of the wall spot; it throws fireballs (`$56`, 1 heart) from the
start.

| Play (30 seeds) | Through the wall into 0x5e | Hearts lost (mean, max) | Lost to | Frames | Bomb laid on frame |
|---|---|---|---|---|---|
| 0x6e, straight up X `$78`, B facing up at (`$78`,`$5D`), standing through the fuse, Up | 30/30 | 1.33, 3 | fireballs 22, touches 18 | 168 | 85 |
| 0x6e, stepping aside (`Dodger`) on the way, during the fuse and back | 29/30 (one timed out) | 0.47, 2 | fireballs 14 | 306 | 163 |
| 0x2e (off the way) S → N key door, straight up | 30/30 | 0.50, 3 | touches 13, fireballs 2 | 107 | — |
| 0x2e, stepping aside | 30/30 | 0.20, 1 | fireballs 6 | 196 | — |

- **The wall**: B at (`$78`,`$5D`) facing up lays the bomb at (`$78`,`$4D`); the wall is open 80 frames after B
  (`CurOpenedDoors` bit 8), one frame of Up starts the change, Link plays in 0x5e at (`$78`,`$BD`) 173 frames later;
  flags 0x6e `$x8`, 0x5e `$x4`. Link's own blast does not hurt him.
- **The blast and the Manhandla** (60 crossings): with the base `$30` px or more from the bomb as the blast starts
  (44 runs) no hand died; at `$18`-`$2F` (15 runs) one hand died in 3. `HelpDropCount` on leaving: 0 in 56, 1 in 4;
  `IsHelpDropBomb` never set. (An earlier version of the dodging walk, 20 seeds, left one run at 9 with two hands
  dead: the blast on the base does credit kills, as the code says, but rarely as high as 10.)
- **The forced drop in 0x5e** (`drop`; counters written, the first Darknut killed by the sword with no hit taken, 20
  seeds each): `HelpDropCount` 9 or `$0A`: 5 rupees in 14 of 20 (taken by walking onto the drop); `$0A` with
  `IsHelpDropBomb` 1: **4 bombs in 14 of 20**; in the other 6 the drop was not reached in time or the kill took a hit
  first. With the counter at 0: nothing in 13 of 20, a heart-group drop in 7.
- So `d8.md`'s "a wall bomb on the base leaves a bomb drop pending for 0x5e's first kill" is possible and was not
  seen once in 60 crossings.

**Models** (0x6e tagged `requires_bombs`; the state holds the route's one bomb):

| Family (its bomb mode) | Crossed | Hearts | Bombs used | Cheated in | Frames |
|---|---|---|---|---|---|
| trap-room (`required`) | 40/40 | −2.27 | 1.00 | 0 | 610 |
| gibdo (`required`) | 20/20 | −2.85 | 1.00 | 0 | 610 |
| boss-manhandla (`free`) | 20/20 | −1.90 | 4.30 | 3.15 | 1131 |
| boss-manhandla with the planner | 10/10 | −1.80 | 4.90 | 3.80 | 1203 |

A `required` family gets BOMB only where it opens the wall (round 22's mask) and crosses with the one bomb. A `free`
family in a `requires_bombs` room bombs the Manhandla and the room's rule cheats a bomb in each time the bag is
empty: the crossing "works" and costs three bombs nobody has.

## 5. 0x4c: eight Pols Voices, the key, the wall (walked)

`r28_pols.py`. Link comes up at (`$70`,`$7D`) inside the block ring. The Pols Voices are made inside the ring too:
100 frames in, five or six of the eight are on row `$9D` (the row the way out runs along), the others beside it.

### 5.1 What hurts them (`measure`, 20 seeds)

| Test | Result |
|---|---|
| Magic sword, Link written beside one | dead after 3 swings that landed, 20 of 20 |
| A touch | 2 hearts, 20 of 20 |
| One arrow along a row holding 1, 2, 3, 4 Pols Voices 32 px apart (written there) | all dead for 1 rupee: 20 of 20 each |
| An arrow 12 frames after a sword hit | flashing: alive, 20 of 20 (the rupee is spent) |
| A bomb's blast with one held on it | no HP lost, 20 of 20 |

### 5.2 The way out (`cross --seeds 30`)

The ring's exit: W one square, down two to row `$9D`, E along it to X `$B0`, up that column to the top row, W to
(`$78`,`$5D`); B (bomb) facing up; Up. The key lies at (`$D0`,`$60`): from the top row W to X `$20`, down, E along the
bottom row, up column `$D0`, and the same way back to the wall.

| Play | To the wall and through | Hearts lost (mean, max) | Frames | Key | Rupees |
|---|---|---|---|---|---|
| wall only, straight | 30/30 | 2.87, 6 | 299 | — | 0 |
| wall only, stepping aside (`Dodger`) | 17/30 | 5.43, 13 | 1114 | — | 0 |
| **wall only, straight, an arrow when a Pols Voice is ahead on Link's line** | **30/30** | **0, 0** | 273 | — | 1 each |
| by the key, straight | 9/30 | 12.23, 13 | 1044 | 26 of 30 took it | 0 |
| by the key, stepping aside | 7/30 | 6.70, 13 | 3720 | 9 of 30 | 0 |
| **by the key, straight, with arrows** | **30/30** | **0.40, 2** | 966 | 30 of 30 | 2.8 (max 5) |

- The arrow rule: Link walking along a leg, a Pols Voice 8-96 px ahead within 8 px of his line, no arrow in flight:
  B (the bow selected), six frames, on. The first arrow, fired E along row `$9D` from the ring's SW corner, clears
  the row.
- Stepping aside fails here: Pols Voices hop over the ring's blocks and change course by jumping, which a straight
  prediction does not follow, and the ring leaves no room.
- The wall: the state's one bomb, laid from (`$78`,`$5D`) as in 0x6e. Link plays in 0x3c at (`$78`,`$BD`).

### 5.3 Models

| Family, variant | Result |
|---|---|
| pols-voice (r74-d5new), the key and the wall | 0/20, all dead (Pols Voice touches 260 of 260 hearts; median 1118 frames) |
| pols-voice with the planner | 0/10 (5 deaths, 5 out of steps) |
| pols-voice, `treasure: skip` (the wall only) | 0/20, all dead |
| pols-voice, wall only, `items: {arrows: true}` | 0/20, all dead |
| gibdo (`required`) | 4/20, −12.3 hearts |
| trap-room (`required`) | 0/40 (15 deaths, 13 stuck, 11 out of steps) |

## 6. Gleeok with four heads, 8:0x3c (walked)

`r28_gleeok.py`. Link in by the bombed S wall at (`$78`,`$BD`). The neck base stands at (`$7C`,`$6F`); blocks at
(`$40`,`$AD`) and (`$B0`,`$AD`); the floor narrows toward the top (row `$5D` is open at X `$70`-`$80` only).

### 6.1 The heads, the fireballs, the rows (`watch`, 20 seeds)

- **Attached heads** sweep X `$5C`-`$A0`, Y `$77`-`$8F` (20 × 900 frames; `d8.md` computed `$64`-`$94`).
- **Where they touch Link** (fireballs taken away, Link held at X `$78`, his timer cleared every frame so that every
  frame of overlap counts): row `$8D` 5563 frames of touch in 20 × 600, row `$95` 1644, **rows `$9D`, `$A5`, `$AD`,
  `$BD`: none**. A touch costs 3/8 heart.
- **Fireballs** (`$56`): one at a time; a new one is made 41-173 frames after the one before (median 61, 246 seen);
  it waits 18 frames, then flies 1.62-1.77 px a frame straight at where Link stood. **1 heart**, 148 hits on a Link
  standing still on the entry spot in 20 × 600 frames (7.4 hearts a run): the entry spot is no place to wait.

### 6.2 Hits (`hits`; Link written under the lowest head, hearts not full, A whenever the heads' timer is 0)

- **3, 2, 2, 2 hits, 9 in all, 20 of 20.** The shared HP byte goes `$A0` → `$60` → `$20` → (head falls) `$60` →
  `$20` → (falls) and so on.
- **36 frames between two hits that land** (min and median over 160 intervals); the whole fight 293-295 frames at
  that pace.
- **Flying heads**: one more after each of the first three cuts, **three at once**, all gone on the frame the last
  head falls (0 left in 20 of 20). One followed for 600 frames: 0.96 px a frame (1.4 at most). 20 magic-sword swings
  beside it: still there. Link written onto it: 1 heart.
- **After the kill**: the room is flagged empty 13 frames later and the heart container shows at (`$20`,`$C0`); the N
  shutter is open 21 frames after the kill. The container is taken at (`$28`,`$BD`): 14 containers, one heart more,
  no refill. The piece in 0x2c: out in 0:0x6d at (`$60`,`$5D`) with **14 of 14 hearts**, `InvTriforce` `$FF`.

### 6.3 Leaving by the S wall (`leave`)

| Left when | Before leaving | Back in |
|---|---|---|
| before any hit | 4 heads, HP `$A0` | 4 heads, HP `$A0` |
| after one head fell and one more hit | 3 heads, HP `$20`, 1 flying head, `RoomKillCount` 0 | **4 heads, HP `$A0`, no flying head, dead-neck mask 0**; banked 0 |

Down from (`$78`,`$BD`): Link plays in 0x4c's N doorway 159 frames later (its eight Pols Voices there again). A
retreat banks nothing and resets the fight; `d8.md`'s reading of the code is right.

### 6.4 The scripted fight (`kill --seeds 30`)

`GleeokFighter`, a frame at a time, positions and the heads' flash only: stand on row `$9D` under the head that hangs
lowest (at full hearts on row `$AD`, where the beam still reaches); when a head is within 8 px of Link's X and 10-27
px above him (any distance with the beam), the heads' timer is 0 and no fireball is predicted to hit: a frame of Up,
then A, nine frames still; else step aside from fireballs, flying heads and the heads (`Dodger`). After the kill:
the container, the N door, the piece.

| Hearts in | Killed, container and piece taken | Hearts lost in the room (mean, max) | Lost to | Frames to the kill | Swings |
|---|---|---|---|---|---|
| 13 | **30/30** | 1.93, 6 | fireballs 46, flying heads 12 (hearts over 30 runs) | 982 | 19.4 |
| 13, standing row `$A5` | 29/30 | 3.33, 13 | fireballs 75, flying heads 24 | 1731 | 19.5 |
| 6 | 26/30 (4 deaths) | 2.97, 6 | fireballs 76, flying heads 13 | 1686 | 10.3 |
| 4 | 19/30 (11 deaths) | 2.70, 4 | fireballs 71, flying heads 10 | 1595 | 9.7 |

- Row `$9D` against `$A5`: 8 px further down the blade reaches the heads only at the bottom of their swing; the fight
  takes twice as long and costs 1.4 hearts more.
- From 6 and 4 hearts there is no beam and the fight takes 700 frames longer than from 13; the wins lose 2.46 and
  1.89 hearts.

### 6.5 Models

boss-gleeok (r71-d, trained on Dungeon 4's two heads from the W door): 0/20 from 13 hearts (all dead; median 2668
frames; of 260 hearts, 210.5 with the Gleeok itself nearest (the base and heads), 40.5 fireballs, 9 flying heads);
with the planner 0/10 (9 deaths, median 3917 frames); from 5 hearts 0/20; with the white sword 0/20.

## 7. Ledgers and the survival budget (walked numbers, added up)

**Bombs.** In 1 (`d6-walks.md` §7). 0x6e's wall takes it (60 of 60 by a `required` family or the script). 0x4c's wall
needs one more: cheated on a no-drop run, as the draft says, **unless 0x3f is fought**: its 4 bombs were taken in 12
of 20 crossings that fight what comes near and in 15 of 20 deliberate clears (mean 1.10 hearts). Then 0x4c's wall
leaves 3, and 0:0x05's rock wall on the way to Dungeon 9 leaves 2 for level 9. Without 0x3f's bombs both walls are
cheats (2 bombs). Drops: Blue Darknuts drop no bombs (group 3); the Red ones and Gibdos of 0x4e and 0x3f do (two
crossings ended with 8 bombs: the room's four and a dropped four).

**Keys.** In 2. 0x5e +1 (taken in 59 of 60 clears), 0x4e N −1, 0x4c's corner +1 (30 of 30 with arrows): out 3, never
below 2. Without the corner key 2. 0x5e E and 0x4e W take a key on the first frame of pressing into them.

**Rupees.** In 30. 0x4c: 1 for the wall alone, 2.8 (max 5) with the key. 0x4e's 5 rupees come only with a clear
(taken in 13 of 17 clears). Out about 25-29.

**Hearts.** No refill between Dungeon 6's piece and Dungeon 8's; **Dungeon 8's piece is the last refill of the game.**

| Stretch | Best play seen | Hearts lost (mean) | Hearts left after it | Success of that play |
|---|---|---|---|---|
| 0x22, the whirlwind | script (B, B) | 0 | 13.0 | 30/30 |
| 0x3c | straight walk | 0 | 13.0 | 30/30 |
| 0x4c (overworld) | ow-octorok | 0.12 | 12.9 | 40/40 |
| 0x4d | straight walk | 0.28 | 12.6 | 30/30 |
| 0x5d | straight walk | 0.23 | 12.4 | 30/30 |
| 0x6d, the tree | script | 0 | 12.4 | 30/30 |
| 8:0x7e | straight walk | 0 | 12.4 | 30/30 |
| 0x6e (bomb wall) | script stepping aside | 0.47 | 11.9 | 29/30 (straight: 1.33, 30/30) |
| **0x5e (fight, key)** | scripted fight | **2.70**, 1.03 back from drops | 10.2 | 30/30 |
| 0x4e (key door) | script fighting what comes near | 0.25 | 10.0 | 19/20 (stepping aside: 0.70, 30/30) |
| **0x3e (fight)** | scripted fight | **3.32**, 1.50 back from drops | 8.2 | 30/30 |
| 0x3f (one way) | script fighting what comes near | 0.90 | 7.3 | 19/20 (darknut model: 2.92, 40/40) |
| 0x2f | equipment-room | 0.05 | 7.2 | 20/20 |
| 0x4c (key, wall) | script with arrows | 0.40 | 6.8 | 30/30 |
| **0x3c Gleeok** | scripted fight | **1.93** from 13; 2.97 from 6; 2.70 from 4 | about 4 | 30/30 from 13; 26/30 from 6; 19/30 from 4 |
| 0x2c, the piece | | 0 | **14 of 14** (the refill) | 30/30 |
| **Sum, best plays** | | **10.7 lost** (leg 0.6, Dungeon 8 10.0), 2.5 taken back | | |
| Sum, only plays that never failed | | 14.0 lost, 2.5 taken back | | |

The "back from drops" of the two fights is their measured hearts-down-on-leaving from 13 hearts (§3.2) against
their losses; from fewer hearts the fights took back more (from 8: 2.2 and 2.6), so the running column is on the
careful side. The crossings' and Gleeok's numbers are losses only.

Read: the means leave room, the tails do not. A single fight's worst run costs most of what is left (0x5e 8.5, 0x3e
10.5, Gleeok 6), and Gleeok is met with about 6.8 hearts and no beam, where the script that never lost from 13 wins
26 of 30 (19 of 30 from 4). Multiplying the scripts' successes at about the hearts they would have (0x6e 29/30,
0x4e 19/20, 0x3e 28-30/30, 0x3f 19/20, Gleeok 26/30, the rest 30/30) gives about 0.7 of the runs through. Every
number is a first script or a model that never saw the room, so training can beat them; but **what 0x5e and 0x3e
cost, and whether their drops are taken, decides Gleeok.** With the plays that never failed instead (0x6e straight,
0x4e stepping aside, the darknut model's 0x3f) Link reaches Gleeok with about 3.5 hearts.

**What a run enters Dungeon 9 with**: 14 containers, full; after 0:0x47's take-any heart 15 containers with 15 hearts
at most if nothing was lost on the way there (that container adds a heart, it does not refill); 3 keys (2 without
0x4c's corner); 0 bombs (2 if 0x3f was fought for its four); 25-29 rupees; the magic sword; no ring until the middle
of level 9.

## 8. What the engine lacks for round 28 (evidence above; terms of `d6.md` §14 and `architecture.md`)

**Route steps, waypoints and objective rules**

| Item | Evidence | Nearest code |
|---|---|---|
| The steps themselves: 0x22 `entry: cave, exit: recorder`; 0x3c `entry: recorder`; 0x4c, 0x4d, 0x5d; 0x6d `exit: cave`; Dungeon 8's ten steps with the tunnel 0x2f (`in_cave`, `exit: W`) | the scaffold's `d8walk` loads and plays as written (`r28_walk_scaffold.py`) | routes.yaml only |
| **The two recorder plays: built.** `exit: recorder` computes two plays facing down from `$FB` with seven pieces; 0x3c is a landing | §1.1: RECORDER, RECORDER, WAIT: `exit-correct`, 20 of 20 | `whirlwind::plan`, `GameCompletion::recorder_facings`, `WHIRLWIND_LANDINGS` (round 26) |
| A missed pickup: the room's edges become retreats (`Plan::Missed`); the recovery then needs one play down and one up, which the plan computes from the counter. Not walked through the engine | §1.1: by buttons 20 of 20 | `zelda::whirlwind` |
| **The candle on the tree as a secret kind**: game.yaml 0:0x6d has `entrance: {how: burn, tile: [20, 4]}`, read by nothing. Target: a stand tile in the window (from N: row Y `$5D`, X `$B8`-`$C0`, facing W; from the W part X `$78`-`$88`, facing E), CANDLE toward the tree there, the same tile while a fire burns, the stairs once they show (the flag is saved: later visits have them) | §1.3: 30 of 30 by script; through the env 19 of 20 with no target | `game_map::Secret` has `Armos`, `Recorder`, `Push` |
| 0x5d's S exit `span: [24, 25]` (the right opening only): both openings reach the tree; the right one stays the shorter | §1.3 | game.yaml data and its note |
| **The one-way room 0x3f**: nothing to build for the way forward (no step returns W; the shutter cannot be passed, so no wrong exit exists there). The tunnel's right ladder back into 0x3f is the wrong exit of the step `exit: W` | §2.2, §2.4; equipment-room 20/20 `exit-correct` | `CELLAR_EAST_COLUMN` (round 24) |
| **Leaving by the entry door on purpose, fighting in parts**: not needed. The script gains nothing by it | §3.3 | `retreat`; `MAX_RETREATS` 3, `MAX_BOUNCES` 5 stay |
| Keys of 0x5e and 0x4c: a room's `treasure: key` is taken before an exit counts (as Dungeon 1 and 6); 0x4c's lies in the open behind the long way | §5.2 | built |
| Gleeok's heart container, the N shutter, the piece | §6.2 | built (4:0x13) |
| `bomb_reserve`: 1 from 0x22 to 0x6e's wall, then 0 (or what 0x3f gave, less 0x4c's wall and 0:0x05's) | §7 | routes.yaml |
| `rupee_reserve` for 0x4c's arrows (Ganon's arrow and level 9's needs stay) | §5.2: at most 5 spent | `GameCompletion::rupee_reserve` |
| 0x3f's bombs: `treasure: skip` (cross) or fought for (the owner's question 1) | §2.1, §7 | route step `treasure:` |

**Masks**

| Item | Evidence |
|---|---|
| CANDLE only at 0x6d's window, facing the tree, while no fire of Link's burns; never in 0x4d (its shop tree) or elsewhere on the leg | §1.3; today a family with `candle: true` is offered it wherever a fire slot is free |
| RECORDER on the overworld by the plan only | built (§1.1) |
| BOMB in 0x6e and 0x4c only where it opens the wall: `requires_bombs` on both rooms and a **`required`** family | §4: trap-room and gibdo 1.00 bomb, boss-manhandla 4.3 |
| ARROW on for 0x4c's family; offered only with a Pols Voice ahead on Link's line would make a miss impossible | §5.2: 1-5 rupees by that rule; a free ARROW was not used at all by today's model |
| MOVE into 0x5e's E and 0x4e's W key doors: `keep_out: true` (as 6:0x29 N) | §2.2: a key on the first frame |
| SWORD and BEAMS off while a Bubble's timer runs | built (§2.5) |

**Observation (schema 6): nothing missing was found.** What these rooms call for is there (`observation-next.md`
§1-§2): a monster's `facing_*` and Link's (the Darknut's shield is the two together), `invincibility` (a Darknut's
and the heads' flash), fireballs and flying heads as entities with positions, Gleeok's attached heads synthesized per
neck with the shared HP and `touch_damage` 3/8, `weapon_target` 0 for Manhandla's and Gleeok's base, `sword_blocked`
and its time, `fire_active`, `candle_usable`, `recorder_usable`, `arrow_usable`, `bomb_usable`, rupees, keys, the
sword grade, 13-14 containers inside `max_hp`'s scale. The scripts above win with positions, facings and the flash
alone. Three things a model cannot read and the walks show matter, each derivable or a mask's business: whether a
Pols Voice is on Link's line (positions), the whirlwind's counter (the mask plays it), the tree's window (the
target). Not checked here: the live values of the columns in these rooms (round 26's probes did that per type), and
whether twelve entity slots hold Gleeok's base, four heads, three flying heads and a fireball at once (nine: they do
by count).

**Families**

| Family | Rooms | Today | What it needs |
|---|---|---|---|
| darknut | 8:0x5e, 0x3e | 1/40, 0/40 | retraining on open floors with statue fire; the scripted fight as demonstrations (`train --demos`) |
| the mixed rooms' family (the owner's question 4) | 8:0x4e, 0x3f | goriya 35/40 and 39/40; darknut 15/40 and 40/40; gibdo 14/20 and 7/20 | training on both; 2.6-5.9 hearts a crossing today against the scripts' 0.25-0.90 |
| a `required` family for the Manhandla crossing | 8:0x6e | trap-room 40/40, gibdo 20/20 (2.3-2.9 hearts) | the room in a `required` family, not boss-manhandla; training to dodge (script 0.47) |
| pols-voice | 8:0x4c | 0/20 | ARROW, `required` bombs for the wall, the ring and the long way to the key |
| boss-gleeok | 8:0x3c | 0/20 | four heads, from S, from 4-6 hearts; the scripted fight as demonstrations |
| equipment-room | 8:0x2f | 20/20 | nothing |
| stalfos-keese-gel (or any) | 8:0x7e, 0x2c | 40/40 | nothing |
| ow-octorok | 0x3c from the whirlwind, 0x4c | 40/40 each | the new state name (`0_3cr`) |
| ow-moblin | 0x4d W → S, 0x5d N → S, 0x6d N → stairs | 35/40, 40/40, 0/40 | `candle: true`, the burn's target, training on 0x6d and 0x4d |
| ow-lynel | 0x22 out of Dungeon 6 by the whirlwind | 0/40 | `recorder: true`, training (two presses and a wait) |

## 9. Not walked, and why

- **A whole run of the stretch end to end**: every room here starts from a staged state at full hearts or at the
  hearts named.
- **0x3f's corridor Darknut** (a W-facing Darknut in the one-wide corridor): not isolated; seen only as time-outs.
- **The recovery from a missed whirlwind through the engine**, and a miss caused by the Lynel's shot itself (none
  happened in 30 seeds; the miss was made by a step off the row and by a written timer).
- **0x4c's fifth Octorok** (overworld): four seen at once, the list's fifth not followed.
- **The forced bomb drop arising by itself** from 0x6e's blast: the counters were written.
- **A scripted Manhandla kill, 0x5f, 0x4d's Gohma, 0x7f, the book and magic-key routes**: off the draft's way.
- **Models from chain arrivals**: no chain reaches these rooms; **darknut's failure was not explained** beyond the
  variants in §3.4.
- **Schema 6's column values in these rooms**: taken from `observation-next.md`, not re-probed.
- **The planner beyond 10 runs a room.**

## 10. Questions only the owner can answer

1. **0x3f's four bombs.** Fight 0x3f (a clear costs the script 1.1 hearts and wins 15 of 20 as written) and have
   honest bombs for 0x4c's wall and 0:0x05's rock wall, or cross it and cheat two bombs?
2. **Arrows in 0x4c.** May pols-voice shoot (1 rupee for the wall, about 3 with the key), and should ARROW be offered
   only with a Pols Voice on Link's line, as Gohma's question 6 of round 27 asked for the aimed mask?
3. **0x4c's corner key.** Worth 0.4 hearts and 3 rupees with arrows (30 of 30); without arrows 9 of 30 survive it.
   Take it, or enter Dungeon 9 with two keys?
4. **The family of 0x4e and 0x3f** (Darknuts, Gibdos and Bubbles together): darknut, gibdo, or a new crossing family?
   goriya transfers best today.
5. **0x6e's family**: trap-room or gibdo (`required`: one bomb, the wall) instead of boss-manhandla (`free`: it kills
   the Manhandla with cheated bombs)?
6. **The scripted fights as teachers.** The Darknut fighter (30 of 30 in both rooms) and the Gleeok fighter (30 of 30
   from 13 hearts) read positions only. Use them as demonstrations for darknut and boss-gleeok, beside or instead of
   the planner?
7. **The bar for the stretch.** The best scripts leave about 4 hearts after Gleeok on average and about 0.7 of the
   runs alive. Is "about 30% on its segment" still the first team's bar here, and may 0x5e and 0x3e be trained with
   hearts-at-exit (drops taken included) priced above everything else?
8. **Fighting in parts**: the walk finds no gain; drop it from the plan (and leave `MAX_RETREATS` alone)?
9. **The tree**: a `burn` secret played by the model's own CANDLE (as the gravestone's MOVE), from the N entry at X
   `$C0`?
