# The late mechanics: the game's rules the route leans on, against what the route engine knows (rest-of-game research, Nov 15-16)

This document covers the mechanics the draft route `first-quest` (`triforce/draft_routes.yaml`,
`docs/routes/first-quest.md`) needs after the D5 triforce: overworld secrets, shops, the recorder and its whirlwind,
the raft, the ladder, wraps and mazes; dungeon doors, shutters, keys, bomb walls, push blocks, stairs, cellars, room
history, persons, drops, death and the end of the game. It was mined from the disassembly (`zelda-asm/`), the level
documents and the project's code before any of it is built, and holds each rule against the route engine.
`docs/routes/rest-of-game.md` is the synthesis over it and its five sibling documents (`d7.md`, `d6.md`, `d8.md`,
`d9.md`, `late-monsters.md`).

Sources: `zelda-asm/` (mechanics.md, items.md, enemies.md, data/, src/) and the route engine (`triforce/routes.yaml`,
`triforce/game.yaml`, `crates/zelda/src/{game_map,objectives,actions,room,game}.rs`, `crates/env/src/{chain,
end_conditions,config,observation}.rs`, `triforce/entry_states.py`, `triforce/explore.py`, `docs/architecture.md`).

Tags: **(trace)** observed in the running game per the cited document; **(code)** read from the disassembly or the
engine source; **(data)** ROM tables (`zelda-asm/data/`); **(draft)** claimed by `first-quest.md` /
`draft_routes.yaml`; **(inferred)** reasoning, not observed. `asm:Name` cites a label.

---

## 0. Findings that change the plan (read these first)

1. **The magic sword is never an objective today, and the threshold in the engine is wrong.**
   `crates/zelda/src/objectives.rs:60-64` `SWORD_UPGRADES` asks `max_health >= 13` for `magic-sword`; the game
   gives it at 12 containers (`HeartValues >= $B0`, asm:Cave_State2TalkOrShopOrDoorCharge) (code vs code). The
   draft arrives with exactly 12 (draft). Also no game.yaml room has `treasure: magic-sword` (0x21 has only an
   `entrance:` block, `triforce/game.yaml:646-663`) and there is no 0x21 cave room (`in_cave: true` overworld rooms
   exist only for 0x0a, 0x0f, 0x2c, 0x2f, 0x67, 0x77) (code). Both must be fixed before the leg.
2. **Overworld stairs are invisible to the objective.** `GameCompletion::cave_tiles` (objectives.rs:982-994) finds
   on the overworld only the cave-mouth tile `$F3` (`room.rs:19,238-245`) or a game.yaml `bomb_secret`. Every
   entrance the rest of the route opens is a stairs square (`$70`): the armos of 0x34, the gravestone of 0x21, the
   pond of 0x42, the trees of 0x6d and 0x47 (code, data). `Room::stairs_tiles` exists but is used only in dungeons.
3. **Arrows are a rupee drain no family masks.** `ItemsConfig.arrows` defaults to `true` (`crates/env/src/config.rs:
   363,377`) and no `specialists/*.yaml` sets it (grep, code). From 0x44's purchase on, every family may shoot an
   arrow wherever an enemy is in the mask, at 1 rupee a shot (asm:Arrow_Wield). The route leaves 0x34 with **2-14
   rupees** (draft ledger) and needs 1 for Gohma and 1 for Ganon. At 0 rupees Ganon's room is a **dead end** (its
   shutters open only on the Triforce of Power, which needs the silver arrow, which needs a rupee) (code:
   asm:Secret_Trigger3PowerTriforce, asm:Arrow_Wield; level-9.md "Dead end").
4. **The whirlwind's destination is a counter the whole run shares** (`TeleportingLevelIndex` `$523`, never cleared
   by room entry, not saved; reset only at power-on) (code: asm:Whirlwind_Summon). Every overworld recorder play
   in mode 5 outside 0x42 steps it, also when no whirlwind is made. The draft's "two plays facing down" are right
   only if no other overworld play happened since power-on (inferred, checked arithmetic in §2.9). The engine should
   compute the plays from `$523` and `InvTriforce` rather than hard-code them, and mask RECORDER on the overworld
   everywhere else.
5. **A route can't end without a triforce.** `Route::load` (`crates/env/src/chain.rs:161-165`) errors unless the
   last step exits `triforce`; level 9 has no piece (its pickup would not even start mode `$12`,
   asm:Item_TakeClass0PerLevel). The end is mode `$13` on the frame after Link stands at X `$70`-`$80`, Y = `$95`
   in 9:0x32 (trace, level-9.md L73). Needs a new terminal.
6. **The whirlwind arrival is not a room entry the engine can name.** `route_entry(prev, curr)` returns None for a
   jump that isn't one room (objectives.rs:244-260; it is how a Wallmaster drag is recognised), `STEP_ENTRIES` and
   `STEP_EXITS` (game_map.rs:263-265) have no `recorder` (code). The draft has `entry/exit: recorder` (draft_routes.py
   L11-12), routes.yaml does not.
7. **Shops have no objective, and the existing waypoint test would accept the wrong ware.** `waypoint_done` reads the
   overworld room's item-taken flag `$10` (objectives.rs:300-310), which the game sets for *any* ware taken
   (asm:Cave_State2TalkOrShopOrDoorCharge `@Take`). In 0x44 the magic shield (130) is affordable with the 142-154
   rupees Link brings (draft ledger) and would end the run's arrow plan (inferred §6).
8. **Start states for second visits are built wrong for this route.** `entry_states build` writes every state with
   "a fresh room" (`triforce/entry_states.py:327-341`, `make_snapshot_fresh`), and `make_room_fresh` clears the
   dungeon kill bits 6-7 and `LevelKillCounts` (`crates/zelda/src/room_entry.rs:91-110`) (code). The draft leans on
   banked kills (9:0x20 on the way back from the silver arrow, 9:0x61's dead Patra, 8:0x5e fought in parts, the
   cleared rooms one comes back up into: 7:0x1a, 9:0x07, 9:0x10, 9:0x55, 9:0x05) (draft). A fresh state resurrects
   even a non-recurring Patra (code: room_entry.rs clears bits 6-7, which is what keeps a non-recurring room dead,
   asm:RoomHistory_ModifyObjCountUW).
9. Already built and reusable (code; round 24): cave steps with `waypoint: heart` (the take-any cave's right ware,
   0x2f) and `waypoint: rupees`; tunnel cellars left by the step's own ladder (5:0x07); `exit: maze` (Lost Hills,
   and the Lost Woods' `MAZES` entry exists, game_map.rs:42-47); wrap exits and `wrap_in`; the overworld ladder and
   its clip in the game's ladder rooms; raft crossings and docks; push blocks (`push_block`, `push_block_opens`);
   bomb walls and rock walls (`bomb_secret`); RECORDER/FOOD/CANDLE actions (masked by family `items:`); a room left
   by the side it was entered by (1:0x72 `entry: E, exit: E` is a live d2first step, routes.yaml) — first-quest.md's
   "a room left by the door it was entered by" in "What is not built" is out of date on that point (draft vs code).

---

## 1. Ledgers on a run with no drops (draft numbers, checked)

| Point | Max hearts | Keys | Bombs (no drops; cheated) | Rupees | Sword |
|---|---|---|---|---|---|
| D5 triforce (0x0b) | 11 | 5 | 0-? (D5 ends at 0 in most chains; draft table) | 142-154 (42-54 + 0x0f's 100) | white |
| after 0x44 shop (arrows 80) | 11 | 5 | — | 62-74 | white |
| after 0x34 shop (food 60) | 11 | 5 | — | 2-14 | white |
| D7 out (0x42) | 12 | 2 | 2 (0 cheated; 7:0x1b +4) | 2-14 (+ drops) | white |
| 0x21 cave | 12 (exactly the sword's need) | 2 | 2 | 2-14 | **magic** |
| D6 out (0x22) | 13 | 2 (takes 0x58, 0x2d; spends 2) | 1 | 1-13 (Gohma: 1 arrow) | magic |
| D8 out (0x6d) | 14 | 3 | 0 (1 cheated: 8:0x4c) | 1-13 | magic |
| 0x47 cave | 15 | 3 | 0 | 1-13 | magic |
| 0x05 rock wall | 15 | 3 | 0 (1 cheated) | 1-13 | magic |
| D9, before 0x16's Patra | 15 | 3 → 0 by 3 doors | 0 (4 cheated before the Patra) | 1-13 | magic |
| D9 after 0x61 key | 15 | 1 | must hold **2** leaving 0x61 N (0x31 W, 0x04 W walls; one-way) | 1-13 | magic |
| Ganon (silver arrow) | 15 | 1 | — | 0-12 | magic + red ring |

- Rupee arithmetic (draft "Rupees", code): `InvRupees` must be >= price at the touch (asm:Cave_State2TalkOrShopOrDoorCharge);
  the debit is counted off 1 per 2 frames (asm:Inv_ChangeRupees): 160 frames for 80, 120 for 60; nothing in the shop
  can be touched meanwhile. Arrow: refused at `InvRupees` = 0, 1 rupee per shot (asm:Arrow_Wield). Margin at the low
  end: **one spare arrow in the whole run** (2 rupees, 2 needed arrows) (inferred).
- Rupee sources on the way (code: asm:Drop_SetUpItem): the tenth damage-free kill's forced drop is 5 rupees unless the
  killing blow was a bomb; table rupees (groups 0-3). No rupee room is on the route (Lee, Nov 14: no detours).
- Hearts: refills only at each triforce (mode `$12`, 43 frames per empty heart, asm:World_FillHearts) and from drops
  (heart +1, fairy +3, the 16th damage-free kill's forced fairy). The Triforce of Power refills nothing (code:
  asm:Item_UpdatePowerTriforceFanfare@FillHearts is unreachable). Red ring from 9:0x00 on: all damage /4
  (asm:Link_BeHarmed).

---

## 2. Overworld rules

### 2.1 Buying in a shop (0x44 arrows, 0x34 food)

**What the game does**
- Wares sit at X `$58`, `$78`, `$98`, Y `$98`; a ware is touched when Link's X **equals** the ware's X and
  |Y − `$98`| < 6; wares are tested right to left (code: asm:Cave_State2TalkOrShopOrDoorCharge, asm:Cave_WareXs).
  Link's grid Ys are 8k+5, so Y `$95` or `$9D` (inferred from asm:Obj_SnapToGrid).
- In a cave Link can never move up past Y `$8D` (asm:Person_BlockLinkMovingUp, called every frame in modes `$B`/`$C`)
  (code). He enters at (`$70`, `$DD`) facing up and walks to Y `$D5` in 5 frames (asm:ModeBCave_Init_Sub7_EnterCave)
  (code). He is halted (state `$40`) until the text is typed (6 frames per character, level-8.md L554) (code).
- Pay shop (CaveFlags `$02`): `InvRupees >= price` or **nothing happens** (no charge, no message); else the price is
  posted to `InvRupeesToSubtract`, the ware is taken (`Item_Take`), the overworld room's flag `$10` is set, and `$40`
  frames later the person and the other wares vanish: **one ware per visit** (asm:Cave_State4DelayThenHide) (code).
  Shops (types `$73`-`$7A`) reappear on every visit (asm:Cave_Init) (code).
- Wares (data, `data/levelblock-ow.md` "Caves"): 0x44 = cave `$1D`: magic shield 130 (X `$58`) | **bombs 20 (X `$78`)**
  | **wooden arrow 80 (X `$98`)**. 0x34 = cave `$20`: key 80 (X `$58`) | blue ring 250 (X `$78`) | **food 60 (X
  `$98`)**. Both wanted wares are the right one (draft agrees).
- Arrows are class 2: wooden stores 1; later the silver arrow stores 2 and wooden never downgrades it
  (asm:Item_TakeClass2Grade) (code). Food is class 0, never consumed except by the Grumble Goriya (asm:Food_Wield,
  asm:Grumble_State3TakeFood) (code).

**Draft dependence:** D5→D7 leg: `0x44 entry S exit N takes [wood-arrow]`, `0x34 entry S exit S takes [food]`
(draft). Order matters (inferred): 0x44 first leaves 62-74, below the key's 80 in 0x34, so 0x34 cannot sell the wrong
affordable ware except food; but 0x44 sells the magic shield (130) and bombs (20) to a Link with 142-154.

**Engine today:** no objective. `subroom_objective` (objectives.rs:1155-1177) takes a cave's treasure only when
`target()` is that cave (sword upgrade, active waypoint, or triforce). `waypoint: heart|rupees` exist (`WAYPOINTS`,
game_map.rs:266) but `waypoint_done` uses the room flag `$10`, which any ware sets (code). game.yaml has no in-cave
room for 0x44 or 0x34; `requires_rupees` exists for the cheat (`GameRoom::requires_rupees`, cheats "rupees" rule,
docs/architecture.md "Cheated items") but no room sets it (code).

**Build:** `waypoint: arrows` / `waypoint: food` (cave steps) whose done test is the inventory byte (`InvArrow` `$659`
>= 1, `InvFood` `$65D` = 1), the target the game.yaml `treasure_tile` of the wanted ware ([19, 11] = X `$98` as for
0x2c/0x2f), `missed` when the room is left without it or another ware was bought (`InvMagicShield` `$676`, bombs
count up, key count up); in-cave rooms `(0, 0x44, cave)`, `(0, 0x34, cave)` with `requires_rupees: 80` / `60` (Lee's
rule 5, Lee's decisions of Nov 16 (`docs/routes/rest-of-game.md` "What this is"): cheat and count rupees at the two
shops). Perception: Link X exact to the pixel; the observation has
`rupees` (observation.rs:370) but no ware entities (wares are drawn through slot `$13`, not objects) (code).

### 2.2 Take-any caves (0x47 heart container)

**Game:** cave `$11`, person `$6B`, once-only (flag `$10` → person gone, asm:Cave_Init): red potion X `$58` | — |
heart container X `$98` (data). No price; taking either ends the offer for good (code). Heart container = `$11` added
to `HeartValues` (one container, one heart), cap 16 (asm:Item_Take_CheckClass1@TakeHeartContainer) (code).

**Draft:** D8→D9 `0x47 entry E exit E takes [heart-container]` (15 hearts) (draft).

**Engine:** built: `waypoint: heart`, missed on the potion ("took the other ware", objectives.rs:820-853), game.yaml
`treasure_tile: [19, 11]` pattern (0x2c, 0x2f) (code). Missing: the cave room `(0, 0x47, cave)` and the burned-tree
entrance (§2.6).

### 2.3 The magic sword cave (0x21)

**Game:** cave `$13`, person `$6D`, middle ware X `$78`; free; taken only if `HeartValues >= $B0` (12 containers,
current health irrelevant); below that, touching the ware does nothing and the person stays (code:
asm:Cave_State2TalkOrShopOrDoorCharge `@TryToTake`). Once taken the person is gone for good (once-only) (code). The
sword is held up, Link halted `$7F` frames (asm:Item_Take) (code). Text "MASTER USING IT AND / YOU CAN HAVE THIS."
(data, levelblock-ow.md "Caves") types first while Link is halted (code).

**Draft:** 12 containers after D7 "exactly what Link has" (draft). Any heart container missed before (0x5f, 0x2f, the
four bosses) makes the sword unobtainable on this route (inferred; no other container source before 0x21 on it).

**Engine:** `SWORD_UPGRADES` wants 13 (objectives.rs:63) — **bug** (code vs code); no `treasure: magic-sword` room
(code). Fix: 12, plus an in-cave room `(0, 0x21, cave)` with `treasure: magic-sword` (middle ware, the default
`CAVE_TREASURE_TILE` (0x0F, 0x0B) = X `$78`, objectives.rs:25).

### 2.4 The gravestone of 0x21

**Game (code: asm:Rock_Update, asm:Spawn_CheckTouchedArmosOrGravestone):**
- Tile object `$65` in slot `$0B` at (`$90`, `$90`) (data: levelblock-ow.md row `$21`; game.yaml tile [18, 10]). No
  item needed. Push: Link's X **equals** `$90`, |Link Y + 3 − `$90`| < `$11`, vertical input exactly toward it (Up
  from below, Down from above); no hold timer: the first qualifying frame starts the slide (code).
- It slides `$10` px in 32 frames; then stairs appear at the room's **shortcut position** (attribute F bits 5-4 →
  `$90`,`$90` for 0x21, data), room flag `$20` is set (code). While it slides Link cannot move within `$10` px of it
  (asm:Link_CheckSlidingTileObjBlocking) (code).
- **Every gravestone square (tiles `$BC`-`$BF`) that blocks Link on a grid point with a direction held releases a
  flying Ghini** on that square, fading in `$3F` frames, touch 1 heart also while fading (enemies.md "Ghini") (code).
  Flying Ghinis are immune to every weapon; they die only when the room's Ghini dies, and once it is dead later ones
  can never be killed (enemies.md "Ghini" Notes) (code). 0x21 has one Ghini, HP `$90`: 5 white-sword hits, 3 magic
  (code). Whether pushing the secret gravestone itself releases one depends on its tiles being `$BC`-`$BF` on the push
  frame (inferred: likely, since Link is blocked by it before `Rock_Update` replaces the square; overworld.md: "Ghinis
  kept knocking Link off it" (trace)).
- **Not kept on re-layout (inferred from code):** asm:Secret_RestoreStairsOW does nothing when the tile at the stairs
  spot is a gravestone `$BC`; 0x21's stairs spot is the gravestone's own square (data). So coming back out of the
  cave, or a later visit, shows the gravestone again (game.yaml says "stairs where it stood", trace of the walk out
  not recorded). Matters after a refusal (11 containers) or a re-entry.

**Draft:** `0x21 entry S exit S`, "push the gravestone at tile (18, 10) down from above" (draft). 0x21 is entered
from 0x31 (S); 0x41, 0x31, 0x21 are graveyard rooms with a Ghini each (data).

**Engine:** nothing: no overworld tile-object push target (the push machinery is dungeon-only: `push_wanted` reads
`state.push_block_pending()` and game.yaml `push_block`, objectives.rs:1014-1063) (code). Tile objects are in no
GameView list ("people, dropped items and tile objects are in none of the lists", architecture.md "Core") (code).
Build: a `secret: push` rule (game.yaml `entrance.how: push` + `tile`) that targets the stand square above/below the
object, then the stairs; a mask or penalty for walking into other gravestone squares (each touch = an immortal
1-heart flyer while the Ghini lives).

### 2.5 Armos secrets (0x34; hazards in 0x22 and 0x3c)

**Game (code: asm:Spawn_CheckTouchedArmosOrGravestone, asm:Armos_InitOrFlyingGhiniFadeIn):** Link blocked on a grid
point by an armos square (`$C0`-`$C3`) with a direction held wakes an Armos on that square (no slot free: nothing).
After `$3F` frames of fade (touch 1 heart on drawn frames, no weapon hits it) the square becomes floor, or **stairs**
if this Armos is the room's secret one: rooms `$0B`, `$1C`, `$22`, `$34`, `$3D`, `$4E` with X `$B0`, `$B0`, `$30`,
`$40`, `$90`, `$A0` and **Y exactly `$80`**. The stairs are **not saved** in the room flags. Armos HP `$30`: 2 white,
1 magic; speed 0.5 or 1.5 px/frame chosen at random at wake (code).
- 0x34: the secret Armos at (`$40`, `$80`), tile (8, 8); "touched from the side it costs nothing, from below a heart"
  (trace, game.yaml 0x34 comment / draft). The computed Armos Y depends on Link's side (asm table
  `Spawn_TouchedSquareOffsetsY`): from the side Y = Link Y + 3 (+8 for the top tile) (code) = `$80` from Y `$75` or
  `$7D` (inferred).
- Coming back out of 0x34's cave, Link stands at the room's exit spot (`$40`, `$6D`) (attribute A high nibble,
  F bits 2-0: data row `$34`, asm:Room_EnterCommon) directly above the statue; the stairs are gone (not saved), so the
  statue is back (inferred from code). Pressing Down there wakes the Armos again (inferred). Walk sideways first.
- 0x22 (D6 entrance, 12 statues, the secret one at `$30`) and 0x3c (D2 entrance, eight squares beside the mouth,
  level-2.md L62) are crossed by the draft: any push into a statue wakes an Armos (code).

**Engine:** no Armos rule; the woken Armos is an enemy the objective sees, the statue squares are just walls (inferred).
Build: `secret: armos` (game.yaml `entrance.how: armos` exists as data) → target the side stand square of the right
statue, then the stairs; overworld stairs detection (§0.2).

### 2.6 Burning a tree (0x6d, 0x47)

**Game (code: asm:Candle_Wield, asm:Fire_Update, asm:Tree_Update):**
- CANDLE puts a fire `$10` px ahead of Link; it moves `$10` px more (31-32 frames), stands `$3F` frames, then frees
  its slot: ~95 frames. Tiles and room bounds don't stop it.
- **Blue candle: one fire per room entry** (`HasUsedCandle` `$513`, zeroed on every room entry). **Red: no limit but
  the two slots** `$10`/`$11`, shared with bombs; a bomb placed later overwrites a fire.
- The tree burns only from a **standing fire with timer < 2** (its last frames) whose midpoint is < `$10` px from the
  tree's on both axes; stairs appear and room flag `$80` is set (kept: later visits show the stairs,
  asm:RoomLayout_LayOutRoomOrCaveOW) (code). So the stairs come ~94 frames after the press, at the tree's square.
- The fire hurts Link (half a heart, ring-reduced) within `$0E` px when his invincibility timer is 0, moving or
  standing (asm:Fire_Update@DrawAndCheckCollisions) (code).
- Only in the secret's quest: 0x6d and 0x47 are Q1 trees (data `qs` column).
- Positions (draft, walked): 0x6d tree at (`$A0`, `$60`) = tile (20, 4): "facing W from X `$B8`-`$C0` on row Y
  `$5D`"; 0x47 tree (`$B0`, `$B0`) = tile (22, 14): "facing S from X `$A8`-`$B8`".
- The red candle is taken in 7:0x4a (game.yaml `treasure: red-candle`, line 5053); both trees come after D7 (draft).

**Engine:** CANDLE action exists (architecture.md "Item kinds"), offered with a candle, a free fire slot and (blue) no
fire yet this room; masked unless a family sets `items: {candle: true}` (config.rs:369, no family does) (code). No
burn target, no tree in GameView (code). Build: `secret: burn` → target the stand tile and facing; the fire
placement is deterministic (16 + 16 px along the facing), so the rule can be exact.

### 2.7 Bombing the rock wall of 0x05 (D9 entrance)

**Game:** opens while a bomb in `$10`/`$11` is in its **blast state `$13` (24 frames)** with midpoints < `$10` px on
both axes, Q1 only for 0x05 (asm:RockWall_Update) (code); unlike a dungeon wall (tested once at the blast's end,
`$18` px from a hotspot). Flag `$80` keeps it open (code). Draft: "facing N from X `$48`-`$58`", tile (10, 12) (draft).

**Engine:** built pattern: game.yaml `bomb_secret: [c, r]` + `requires_bombs` (0x2c, 0x67; objectives.rs:982-994,
`BombTarget` rock wall `$10` reach, architecture.md "Item kinds") (code). Only yaml to add (overworld.md L60 says
new rooms did not get the keys).

### 2.8 The recorder at 0x42's pond (D7 entrance)

**Game (code: asm:Flute_Wield, asm:FluteSecret_Update):** every play: tune + `FluteTimer` = `$98`, 152 frames in
which the play update returns at once (nothing moves, no input). In Q1 room `$42`, in mode 5, with `SecretColorCycle`
= 0 and a free slot in `$01`-`$09`: the flute-secret object; 81-88 frames after the freeze the stairs appear at X
`$60`, Y `$90` (tile (12, 10)), flag `$80` set (kept). **This play does not step the whirlwind index** (the room
branch returns before `Whirlwind_Summon`). Leaving by an edge before it finishes waits while the cycle runs backwards
(asm:FluteSecret_ReversePondCycle). B is cleared in the border bands (asm:Link_FilterInput): entering from S at Y
`$DD` facing up, the outer band behind him (Y >= `$D6`) clears B, hence "one step in from the edge" (draft; inferred
from code).

**Engine:** RECORDER exists but `recorder_wanted` returns false on the overworld (actions.rs:381-394: "a round that
does widens") (code). Build: widen `recorder_wanted` to 0x42 while flag `$80` is clear and `$51A` = 0; the stairs
need overworld stairs detection (§0.2).

### 2.9 The whirlwind (three legs)

**Game (code: asm:Whirlwind_Summon, asm:Whirlwind_StepLevelIndex, asm:Whirlwind_Update_Full,
asm:Whirlwind_InitDropOffAndBeginUpdate; items.md "Recorder"):**
- Overworld, mode 5, any room but 0x42 (Q1): the play first steps `TeleportingLevelIndex` (`$523`) by +1 if Link faces
  right or up, −1 if left or down, then keeps stepping the same way (mod 8) past levels whose piece he lacks; **with
  no piece nothing appears**. Then, only if none was summoned in this room entry (`IsWhirlwindSummoned`, cleared on
  entry), none is carrying him, and a slot `$01`-`$0B` is free, it makes one. **The index has stepped even when no
  whirlwind is made.**
- The whirlwind is made at X 0 on Link's Y, updates first after the freeze, may play a spawn cloud, then moves right
  2 px/frame; pickup is the ordinary touch test (< 9 px both axes), so **nothing picks Link up while his invincibility
  timer is nonzero, he holds the clock, he is halted or a Like-Like holds him**; a missed whirlwind leaves at X `$F0`
  and the room allows no second one until re-entered.
- Pickup runs the harm path with 0 damage: invincibility `$18`, and **zeroes `WorldKillCount`, `HelpDropCount`,
  `IsHelpDropBomb`** (the 16-kill fairy and 10-kill drop restart) (code: asm:Link_BeHarmed).
- Destination = the index's level at the moment of pickup (a second play before pickup re-steps it). Link is carried
  into the room west of the landing room, then scrolled right into it; released at X `$80`, Y = `$8D` (L1 0x37),
  `$AD` (L2 0x3c), `$8D` (L3 0x74), `$8D` (L4 0x45), `$AD` (L5 0x0b), `$8D` (L6 0x22), `$AD` (L7 0x42), `$5D` (L8
  0x6d), 86 frames after play starts there (code).
- Trace (items.md L419): from 0x0b with pieces 1-5 and the index 0, two plays facing down, the second on the first
  free frame after the first freeze: set down in 0x45, 625 frames after the first play; one play: 0x0b itself, 477.

**Check of the draft's counts (inferred, arithmetic on the code):** the index starts 0 at power-on and only overworld
plays move it; the route plays the recorder only in dungeons (no step) and at 0x42 (no step) besides the whirlwinds.
- 0x0b after D5 (pieces 1-5), facing down: play 1: 0→7(L8 no)→6→5→4 (L5) = 0x0b; play 2: 4→3 (L4) = **0x45** ✓.
- 0x22 after D6 (pieces 1-7), index 3, facing down: play 1: 2 (L3); play 2: 1 (L2) = **0x3c** ✓.
- 0x6d after D8 (pieces 1-8), index 1, facing down: play 1: 0 (L1) = **0x37** ✓.
- One stray overworld play, a play facing up/right, or a first whirlwind missed and re-summoned after re-entry shifts
  every later destination (code). Coming out of a level Link faces down (asm:Room_EnterCommon `@FaceDown`) (code); a
  model that turns before playing reverses the step.

**Engine:** none: no `recorder` entry/exit (game_map.rs:263-266), `route_entry` gives None (objectives.rs:244), the
chain would treat the arrival as an off-route jump (inferred), RECORDER masked on the overworld (actions.rs:387-394),
no observation of `$523` or `IsWhirlwindSummoned` (observation.rs link features, code). The executor already waits
out the freeze as `menu_frames` (architecture.md "Item kinds") (code). Whether the classifier files the whirlwind
(`$2E`) as an enemy that interrupts actions is unchecked (inferred risk).

### 2.10 Raft and docks (0x45 → 0x55 again)

**Game (code: asm:Dock_Update):** docks only in 0x3f (X `$60`) and 0x55 (X `$80`); a ride starts when Link's X equals
the dock X exactly and Y is exactly `$7D` (ride up) or `$3D` (arriving at 0x55's top edge from 0x45: ride down);
halted and untouchable the whole ride; 64 / 66 frames. **Built** (routes.yaml `raft_exits`, `RoomExit::raft`,
d2first steps 0x45 cave→S and 0x55) (code). The draft's D5→D7 leg reuses 0x45 S → 0x55 dock (draft). The whirlwind
releases Link in 0x45 at (`$80`, `$8D`) (code), already on the dock column.

### 2.11 The overworld ladder

**Game:** laid only in rooms `$17`, `$18`, `$19`, `$27`, `$4F`, `$5F`, in mode 5, on a grid point, with the held
direction = facing, a free monster slot, and water tile `$8D`-`$98` ahead (as the lookup returns them, so `$8D`/`$91`
never) (asm:Ladder_RoomIdsOW, asm:Link_EndMoveAndAnimate@CheckLadder) (code). One square per laying; no edge test
while it is out (code).
**Draft:** 0x17 "the ladder W over the river, then N at X `$40`" (Lee: no wrap there) (draft). **Built** generically
(round24.md §5: "the room model knows overworld water and its one-square crossings in the game's ladder rooms")
(code); 0x17 is a ladder room (code). Walk it before trusting the room model's crossings there.

### 2.12 Wrap exits (0x51 again; 0x60's water arrival)

**Game (trace, mechanics.md "Screen wrap"):** one perpendicular frame 5 px short of the bound in the 1-px cadence;
0x51 → 0x50 on rows Y `$75`-`$9D` (draft); 0x60 → 0x5f lands on water (only rows `$85`/`$8D` lead on, with the
ladder clip; without the ladder a softlock, mechanics.md L754) (trace). **Built**: game.yaml `wrap: true` + `span`,
`CLIP E` `SkipKind::WrapExit` (game_map.rs:127-132), and round 24's coast-heart (0x51 E→W, 0x60 N→W) (code). The
D7→sword leg reuses 0x51 E→W, then 0x50 **E→N** (new step: N at X `$E0` only) (draft).

### 2.13 Lost Woods (plain way of the D7→sword leg)

**Game (code: asm:Maze_CheckStep):** room `$61`; the direction tested is `ObjDir` at the room change; sequence N, W, S,
W; a correct non-final step brings 0x61 back with `MazeStep` + 1; a wrong one resets to 0 and is not itself step 0;
E always leaves to 0x62 keeping the counter; the 4th correct step leaves W into 0x60. `MazeStep` resets on any other
overworld room change (code). Entered from 0x51 (N side) the first correct step is N, back through the side Link came
in by (draft: "N (at once, on the arrival row)").
**Built:** `MAZES` has 0x61 (game_map.rs:42-47), `exit: maze`, `MazeLoop`, observation `maze_step` (round24.md §4)
(code). Only 0x1b's was played; 0x61 needs game.yaml's `maze:` block checked and states (code requires the block to
match, game_map.rs:944-983).

### 2.14 Overworld stairs, exit spots, `UndergroundExitType`

- Entering: mode 5, grid offset 0, `UndergroundExitType` = 0, X multiple of `$10` (room `$22`: of 8), Y low nibble
  `$D`, tile at (X, Y + `$0B`) is `$24`, `$88` or `$70`-`$73` (asm:Warp_CheckUnderLink) (code).
- Out of a cave/level Link stands at the room's exit spot facing down; `UndergroundExitType` stays nonzero until he
  completes one grid step in mode 5, so the entrance under him doesn't take him back (code).
- Exit spots on the route (data, `exit` column): 0x0b (`$70`,`$7D`); 0x44 (`$40`,`$4D`); 0x34 (`$40`,`$6D`); 0x42
  (`$60`,`$5D`); 0x21 (`$50`,`$6D`) per data (draft says stairs at the gravestone square `$90`,`$90`: the exit spot is
  the attribute's, not the stairs' — walk it; inferred risk of arriving away from the stairs); 0x22 (`$70`,`$7D`); 0x6d
  (`$60`,`$5D`) per data while the burned tree's stairs are at (`$A0`,`$60`) (inferred: same caveat); 0x47
  (`$B0`,`$9D`); 0x05 (`$50`,`$9D`).

### 2.15 Edge spawns, Zoras (context)

Rooms with `ed = E` bring monsters in from the edges one at a time (asm:Obj_Init@InitMonsterFromEdge); a cleared
overworld room stays empty only while among the last six rooms (asm:RoomHistory_ModifyObjCountOW) (code). Route rooms
with edge spawns: 0x6d, 0x52 (data). Zora rooms (`zo = Z`): 0x44, 0x34? (no: 0x34 `-`), 0x45? (`-`), 0x17, 0x27,
0x28? (Z), 0x47 (Z), 0x48 (Z) (data).

---

## 3. Dungeon rules

### 3.1 Door types and their two sides (code: asm:Door_Touch_JumpTable, asm:Door_FindTypeByDirBit)
- Each side of a room has its own 3-bit type: 0 open, 1 wall, 2/3 false wall, 4 bombable, 5/6 key, 7 shutter. A
  doorway can be a shutter on one side and open on the other (level 7: seven such, level-7.md L638) (code/data).
- Link is at a doorway only with X = `$78` (N/S) or Y = `$8D` (E/W); once `DoorwayDir` is set nothing blocks him
  (code). Entering a room through a side puts that side into `CurOpenedDoors` (asm:Door_SetEnteringDoorwayOpened)
  and then issues a close command: a shutter closes behind him; key doors and bombed walls stay open (code).
- Engine: door kinds from game.yaml exits (`locked`, `bomb_walls`, `shutter` display only, game_map.rs:112-140),
  barred-door state from tiles (`is_door_barred`) (code). One-way shutters (9:0x41/0x31/0x21 S, 0x01 S; 8:0x3f; level
  7's seven) are two-way exits in game.yaml (game_map test `exits_are_two_way_with_one_convention`) (code): a
  path search off the route could aim at a shutter that never opens (inferred).

### 3.2 Shutters and their triggers (code: asm:Secret_RunTrigger_JumpTable, asm:Shutter_OpenTriggered)
- Triggers: 1 all dead; 2 ringleader (slot 1 dead kills all, no shutter); 3 Triforce of Power held (Ganon); 4 block
  pushed; 5 block pushed → stairs at (`$D0`,`$60`); 6 life-or-money man gone; 7 all dead + room item appears. "All
  dead" ignores Bubbles `$2B`-`$2D` and types `$49`+ (traps, persons) (asm:Secret_UpdateRoomAllDead) (code).
- Shutters open one per 8 frames, never saved, close behind Link on the next entry (code).
- Engine: `shutter_requires_combat` turns the objective to Fight when the exit is barred and the trigger is
  all-dead / ringleader / last-boss / all-dead-item (objectives.rs:52-57, 1256-1272) (code). Trigger 3 (Ganon) is in
  the list as `LastBoss`, but after Ganon dies no enemy is left and the shutter is still shut until the Triforce of
  Power is taken: that needs the room's `treasure: triforce-of-power` (game.yaml line 6055) to be the objective
  (default `treasure` rule: Take) (inferred: works if the treasure kind is handled; walk it).

### 3.3 Key doors and the keyless pass (code: asm:Door_TouchKey)
- Test order: open → pass; a door command running → blocked while Link's `ObjTimer` ≠ 0, **pass free while it is 0**;
  magic key → open free; no keys → wall; else spend one, open (8 frames later), saved on both sides.
- The keyless pass happens during a shutter opening by trigger or a bombed wall opening (~8 frames; they leave
  Link's timer at 0); the door stays locked and unsaved on that side, the far side is drawn open and saved there
  (code). Not used by the draft (draft). Engine: keys counted from RAM; door open state from room flags
  (`seed_from_room_flags`, objectives.rs:898-922) (code); an accidental keyless pass leaves one more key than the
  ledger and the near side locked (inferred: harmless to the route, but the key ledger reports shift).
- Keys are one counter for all levels (code). Engine cheats a key only where the route step leaves through a still
  locked door (architecture.md "Cheated items") (code).

### 3.4 Bomb walls (code: asm:Bomb_TryOpenWall, items.md "Bombs")
- Tested **once per bomb, on the frame its blast ends** (`$48` frames after its first update), bomb X and Y each <
  `$18` from the hotspot (N `$78`,`$5D`; S `$78`,`$BD`; W `$20`,`$8D`; E `$D0`,`$8D`; first in range searched E, W,
  S, N), the side not already open, **no door command running**, type 4; never in a cellar. Open 8 frames later,
  saved both sides (code).
- One bomb in its fuse at a time; a second can be placed `$30` frames after the first (code).
- Engine: built (`bomb_walls`, `BombTarget::in_reach`, `awaits_bomb`, `bomb_spot`, `required` bomb mode) (code).
  Draft: 19 walls (draft).

### 3.5 Push blocks (code: asm:PushBlock_State0Idle, asm:PushBlock_State1Moving, mechanics.md "Push blocks")
- Block = first `$B0` tile in floor row Y `$90` (X `$20`-`$C0`), else X `$D0`. Moves only with no counting monster
  left (`IsRoomAllDead` ≠ 0; it reads 0 one frame in 256, restarting the count). Link exactly in line (same X, or Y +
  3 = block Y), ≤ `$10` px away, `ObjInputDir` exactly toward it, **16 consecutive frames**; moves `$10` px in 32
  frames, **once per room entry**; returns to its square on every entry (code).
- Ring rooms (5:0x64, 7:0x1a, 9:0x55, 9:0x07, 9:0x20, 9:0x61, 9:0x03; 1:0x22 type): the W block pushed N or S opens the
  ring; **pushed E it closes the stairs until the room is entered again** (draft/trace). Trigger-5 rooms (7:0x0d
  "push W", 9:0x05, 9:0x10 "push W", 9:0x30, 6:0x3a, 6:0x38 trigger 4): stairs at (`$D0`,`$60`) whatever the
  direction, but a block pushed into a lane can wall Link off (8:0x1f diamond, trace, level-8.md L115).
- Engine: built (`push_block: [N, S]`, `push_block_opens`, `push_requires_combat`, `push_block_targets`,
  objectives.rs:1014-1063, 1275-1281) (code). Not built: a recovery rule after a wrong push (leave and re-enter);
  the mask allows MOVE toward the block in every direction (`is_push_block_direction`, actions.rs:368) (code).

### 3.6 Stairs and the grid rule
Dungeon stairs `$70`-`$73`: mode 5, grid offset 0, X multiple of `$10`, Y low nibble `$D`; the cellar is the first in
the level's list whose attribute A or B is this room (asm:Warp_CheckUnderLink) (code). After coming up, one grid step
first (code). Engine: `stairs_tiles` targets (objectives.rs:982-985) (code). Built.

### 3.7 Cellars and tunnels (code: mechanics.md "Mode 9")
- Item cellar `$3F`, tunnel `$3E`. Link appears at Y `$41` facing down, walks to `$5D`; X `$30` if he came from the
  room in attribute A, else `$C0`. Four blue Keese whatever the data. Leave: Y < `$40` with Up held; X < `$80` → room A,
  else room B; he comes out at attribute C's fixed spot, the same for both rooms, facing down. Palette fades both
  ways; no doorways, no bomb test in a cellar (code).
- Engine: built (round 24: the step's own ladder, `CELLAR_EAST_COLUMN`, objectives.rs:46-48, 1179-1199) (code).
  Draft: tunnels 7:0x7b, 6:0x08, 8:0x2f, 9:0x60, 0x70, 0x75 (twice), 0x67, 0x77 (draft).
- A cellar takes a place in the six-room history (level-9.md L76, trace).

### 3.8 Where Link arrives, and rings of blocks
Out of a tunnel Link stands at the fixed spot of the room above (attribute C); outside its ring of blocks except
9:0x14 (spiral) and 9:0x10 (C of blocks), "inside blocks that lead out" (draft/trace). In 8:0x4c the stairs are two
squares E of the arrival (level-8.md L555, trace). Engine: arrival spots come from the game; the room model's
wavefront finds the way out (code). A block clip enters a ring with the monsters alive (5:0x64, 7:0x1a, 8:0x1f)
(trace).

### 3.9 Room history: banked kills, the six-room ring, what never returns (code: mechanics.md "Dungeon kill counts")
- Leaving a room (edge, stairs, cave) banks its kills: `LevelKillCounts[room]` (RAM, zeroed only at a level load) and
  flags bits 6-7 (min(kills, 2), 3 = all dead) (code).
- Entry: room in the six-room history, or flags not all dead → count − banked; flags all dead and out of the history
  → a **non-recurring** room (first type `$32`-`$48` except Lanmola: Dodongo `$32`, Gohma, Digdogger, Manhandla,
  Aquamentus, Ganon, Moldorm, Gleeok, Patra, ...) stays empty **forever**; any other room refills (code).
- The history: a ring of six room ids, a room added only if absent; zeroed at a level load, mode 3, and the continue
  menu (code). Cellars count (trace).
- Red Keese from Vires and Lanmola segments don't count toward `RoomKillCount`, so those rooms are never saved as
  fully killed by the count (enemies.md) (code).
- Draft dependence: 8:0x5e "kills are kept across a retreat S", 9:0x20 "still clear: push the ring's W block N",
  9:0x61 "the Patra stays dead", 5:0x66 refilled after the long detour (draft/trace).
- Engine: plays the game's own rule (it reads RAM); `ram_randomization.room_kills` stages banked kills in training
  (round24.md §7) (code). **Not handled:** start states are written fresh (§0.8) (code).

### 3.10 Dark rooms
Only the palette is dark; tiles are unchanged (game.yaml `dark: true` is display only, game_map.rs:178-179) (code).
A standing candle fire in a dungeon starts a 4-step fade (40 frames) during which the play update does only that
(asm:Candle_UpdateRoomLight) (code). The observation reads RAM tiles, so darkness costs the agent nothing (inferred);
lighting is never needed. Route dark rooms: 7:0x18, 6:0x29, 0x39, 0x2d, 9:0x65, 0x15, 0x41, 9:0x42 (Ganon's) (draft/data).

### 3.11 The men who halt Link and block the way up
- Dungeon persons (`$4B`-`$52`) and the Grumble Goriya (`$36`) stand at (`$78`, `$80`) with two fires (`$40`, half a
  heart touch) at X `$48`/`$A8`; Link is halted while the text types (6 frames/char) and **cannot move up past Y
  `$8D`** while the person is in slot 1 (asm:Person_BlockLinkMovingUp) (code). Hitting one turns on two fireball
  sources until the room is left (asm:Person_CheckCollisions) (code).
- On the route: 7:0x28 Grumble (needs the food; exit N) (draft); 9:0x66 old man (gone on every entry with all eight
  pieces, his fires stay) (trace); 9:0x06 old man (exit W, a bomb from (`$30`, `$8D`)) (trace).
- Grumble (trace, level-7.md L650): text 121-123 frames from the first play frame; START works meanwhile (choose the
  food then); B is masked in the S doorway; after 6 frames of Up (Y `$C5`) B puts the food at (`$78`, `$B5`); the
  Goriya takes it at once (Link halted), is gone `$40` frames later with `InvFood` = 0 and the flag set; a boomerang
  throw removes food from the ground (code).
- Engine: FOOD action exists, masked unless `items: {food: true}` (code). No objective rule "feed the Grumble": the
  objective would aim at the N door that the blocker forbids (inferred). Build: a Grumble rule (target: FOOD anywhere
  in the room once the text has typed; done = flag `$10` of 7:0x28 or `InvFood` 0).

### 3.12 Items that ride on monsters
The visible room item rides on the monster in slot 1 if its type is Like-Like `$17`, Stalfos `$2A` or Gibdo `$30`
(asm:Item_MoveAndDrawRoomItem); taken by touching the carrier (its touch damage too) or where it died; a slot-1
Stalfos/Gibdo never drops anything else (code). Engine: built (round 24, the Treasure target rides on the slot-1
Gibdo, d5.md L53) (code). On the 6-9 route no carried item is named by the draft (draft); check rooms whose slot 1
is a Like-Like with a visible item (inferred: none found in the draft notes).

### 3.13 Room items that appear when the room is cleared (trigger 7)
Hidden until no counting monster is left, then the item tune (asm:Secret_CheckUnderworld) (code). Route keys: 6:0x58
(8 Keese), 6:0x2d (3 Vires); boss heart containers (draft). Engine: built (Fight when `wanted && item`,
dungeon_room_objective, objectives.rs:1201-1254) (code).

### 3.14 Heart containers and the triforce
Container: `HeartValues` + `$11` (no refill) (code). Piece: flag `$10`, `InvTriforce` bit, mode `$12`: wait, flash,
**refill to full** (43 frames/heart), curtain, then the overworld at the level entrance's exit spot facing down
(asm:Mode12EndLevel_Update_Sub4LeaveLevel) (code). Engine: built (`exit: triforce`, `GainedTriforce`,
`BossDeadAndHeart`, chain warp handling, chain.rs:1226-1229) (code).

### 3.15 Forced drops: the tenth and the sixteenth (code: enemies.md "Drop tables")
- `HelpDropCount` +1 per killing hit (cap `$0A`); a death that ends while it is `$0A` drops **5 rupees, or a bomb**
  if the kill that brought it to `$0A` was a bomb blast. `WorldKillCount` +1 per killing hit (no cap); a death that
  ends while it reads exactly `$10` drops a **fairy** (3 hearts). Sequence without damage: 10, 16, 26, 36 (code).
- Slot-1 Stalfos/Gibdo and no-drop types skip; Dodongo sword kill forces a bomb (code).
- Every harm resets all three counters: monster touch, shot, Link's own fire, a Bubble touch (0 damage), the
  whirlwind pickup; untested touches (invincible, clock) don't (code).
- Engine: built (`tenth` bomb mode, observation `help_count`, `next_kill_arms`, `fairy_next`, observation.rs:400-404)
  (code). The rupee value of the tenth kill matters now (§1).

### 3.16 Bubbles
Q1 has only the flashing Bubble `$2B`: touch blocks the sword 151-160 frames (`SwordBlockedLongTimer`), B items still
work; no damage but the harm path (knockback, invincibility, counter reset) (code; enemies.md L841-869). Bubbles never
die and don't count for "all dead" (code). Route Bubble rooms: 9:0x07 (3), 9:0x10 (3), 9:0x65 (3), 8:0x3f (2), 6:0x1d,
7:0x49 (2) (data/draft). Engine: mask SWORD/BEAMS while `$4C` ≠ 0 (architecture.md) (code).

### 3.17 What a death does
- Game: mode `$11` (~320 frames, no input), then CONTINUE/SAVE/RETRY; every choice sets 3 hearts. CONTINUE keeps
  inventory, world flags and `LevelKillCounts`, clears the room history, restarts in the level's start room (code).
- Chain: the run ends `failure` with the room's outcome `death` (chain.rs:1075-1083) (code); training: `death` is a
  hard failure (end_conditions.rs) (code). No continue is modelled. A Wallmaster drag (7:0x0d → 7:0x79) is
  `wallmastered` in training (end_conditions.rs:217-224) and in a chain an entry into a tagged room by no side (route
  entry None) (code); 7:0x79 is tagged on the route, so the run goes on off-route (inferred: walk a chain through it).

### 3.18 The end of the game (code/trace: level-9.md L73, L617-622; items.md "Triforce of Power")
- 9:0x42: entry scene ~270 frames halted; Ganon 4 magic-sword hits only while invisible and at full extension, touch
  4 hearts (1 with the red ring), fireball `$56` 1 heart never parried; then ~510 frames to hit him with a **silver**
  arrow (`InvArrow` = 2; 1 rupee) or the count starts over; `$A0` frames later the Triforce of Power appears at his
  ashes; taking it halts Link `$C0` frames, then `InvPowerTriforce` = 1 and trigger 3 opens both shutters. No refill.
- 9:0x32: Zelda and four guard fires (`$3F`, half a heart, HP `$10`: any damaging hit kills one). Link at X `$70`-`$80`
  and Y **exactly `$95`** is halted and placed at (`$88`, `$88`); mode `$13` the next frame (trace).
- Engine: none (§0.5). Build: `exit: end` (draft has it) = success when `GameMode` = `$13`; the executor must not wait
  for control in mode `$13` (inferred).

---

## 4. Rooms with a mechanic on the route (overworld legs and dungeon specials)

| Room (step) | Entry, first movable | Before he can leave | How (positions, frames, order) | What hurts | Irreversible | Perception |
|---|---|---|---|---|---|---|
| 0x0b (D5 out) | warp: exit spot (`$70`,`$7D`) facing down (code) | whirlwind to 0x45 | RECORDER twice facing down, the 2nd after the 1st freeze (152 f) and before pickup; stand still on the row (draft/trace) | Red Leever (data); a hit before pickup = invincible = missed whirlwind (code) | a stray play shifts the index for every later leg (code) | `$523`, `InvTriforce`, facing, whirlwind position |
| 0x45 (whirlwind in) | released at (`$80`,`$8D`) ~86 f after play starts (code) | S at X `$80` exactly | walk S, raft down at 0x55 (built) | Peahat (data) | — | route entry "recorder" |
| 0x55, 0x65, 0x64, 0x54 | as d2first | wrap or plain way | built stretch shape (draft) | Zoras, Octoroks | 0x55 wrap rows (built) | — |
| 0x44 + cave `$1D` | cave: (`$70`,`$D5`) after 5 f; halted during text (code) | arrows bought | stand at (`$98`, `$95`/`$9D`); 80 rupees, 160 f count (code) | Octoroks, Zora outside (data) | **magic shield (`$58`) or bombs (`$78`) bought instead** (inferred §6) | rupees, ware X, `InvArrow` |
| 0x34 + cave `$20` | S edge | food bought | wake the `$40`,`$80` Armos from the side (tile (8, 8)); stairs after `$3F` f; kill/avoid the Armos (2 white hits, 1 heart touch); food at X `$98`, 60 rupees (trace/code) | 4 Blue Leevers (data), the Armos | stairs not saved; re-waking on the way out (inferred) | statue position, Armos fade timer |
| 0x42 (pond) | from S | stairs revealed | one step in (out of the border band), RECORDER; stairs at (`$60`,`$90`) 81-88 f after the freeze; enter at (`$60`,`$8D`) (code/draft) | Moblin (data) | — | flag `$80`, `$51A` |
| 7:0x28 (Grumble) | S doorway (`$78`,`$CD`) halted ~122 f (trace) | food given | Up 6 f, FOOD (B) at (`$78`,`$B5`) (trace) | none | food consumed (intended) | person in slot 1 |
| 7:0x1c, D7 Digdoggers | — | recorder in room | built (round 24) | — | — | — |
| 0x51/0x50 (sword leg) | E | wrap W (clip) / S to the woods (plain) | built shape; 0x50 N at X `$E0` (draft) | Moblins / Lynels | 0x51 wrap fail = ordinary room change (trace) | — |
| 0x61 Lost Woods (plain) | from N | sequence N W S W | built `exit: maze`; first step N back out the N side (draft) | Moblins (data) | — | `maze_step` |
| 0x40, 0x41, 0x31 (graveyard) | edges | cross | never press into a gravestone square: each press releases a flying Ghini (1 heart, immortal while the Ghini lives) (code) | Ghini, flying Ghinis | — | gravestone tiles `$BC`-`$BF` |
| 0x21 + cave `$13` | from S | magic sword | push the (`$90`,`$90`) gravestone up or down (X = `$90` exactly; draft: down from above); stairs at its square after 32 f; sword: middle ware X `$78`, needs 12 containers (code/draft) | Ghini (5 white hits), flying Ghinis | the push may need repeating on re-entry (inferred); <12 containers → nothing (code) | containers, gravestone, stairs |
| 0x22 (D6 in/out) | S / exit spot (`$70`,`$7D`) | D6 / whirlwind | mouth open; out: RECORDER twice facing down → 0x3c (code arithmetic) | Lynel, 12 Armos statues (code) | stray play | as 0x0b |
| 0x3c (whirlwind in) | released (`$80`,`$AD`) (code) | S through the one bottom square, X `$70` | avoid the eight armos squares (level-2.md L62) | Octorok, Armos | — | — |
| 0x6d (D8 in) | from N (0x5d S at X `$C0`, draft) | tree burned | CANDLE facing W from X `$B8`-`$C0`, Y `$5D`; stairs ~94 f later at (`$A0`,`$60`) (draft/code) | 4 Moblins, edge spawns (data); own fire ½ heart | — | candle slots, fire timer, tree |
| 0x6d (D8 out) | exit spot (code) | whirlwind to 0x37 | RECORDER once facing down (code arithmetic) | Moblins | stray play | as 0x0b |
| 0x47 + cave `$11` | from E | heart container | CANDLE facing S from X `$A8`-`$B8`; tree (`$B0`,`$B0`); right ware X `$98` (draft/data) | Zora (data) | **the potion taken: container lost forever** (code) | ware X |
| 0x17 | from S | N at X `$40` | ladder W over the river (draft) | Boulders (data) | — | water squares |
| 0x05 | from E | rock wall opened | bomb facing N from X `$48`-`$58`; blast within `$10` of (`$50`,`$A0`) (draft/code) | Lynels, Leevers (data) | needs a bomb (cheated if none, draft table) | bombs |
| 9:0x61 → N | from the tunnel | **hold 2 bombs** | the S shutters of 0x41, 0x31, 0x21 never open (draft/trace) | — | one-way: no bomb = stuck (code) | bombs, one-way flag |
| 9:0x42 Ganon | halted ~270 f (trace) | Triforce of Power taken | 4 magic hits, then a silver arrow (1 rupee) within ~510 f, take the piece (trace) | Ganon 4♥ (1♥ red ring), fireballs | **0 rupees or no silver arrow: dead end** (level-9.md L622) | rupees, Ganon phase |
| 9:0x32 Zelda | from S | — | walk up the middle to X `$70`-`$80`, Y `$95` (trace) | guard fires ½♥ (⅛ red ring) | — | — |

---

## 5. Route step types and objective rules to build, in dependency order

| # | What | Why (evidence) | Nearest thing that exists |
|---|---|---|---|
| 1 | **Fix the magic-sword target**: 12 containers; an in-cave room `(0, 0x21, cave)` with `treasure: magic-sword` | objectives.rs:63 (13) vs asm:Cave_State2TalkOrShopOrDoorCharge (`$B0`); no room has the treasure (game.yaml) | `SWORD_UPGRADES`, the white-sword cave room `(0, 0x0a, cave)` (game.yaml:194-202) |
| 2 | **Overworld stairs as cave tiles** (tile `$70` squares on level 0) | `cave_tiles` (objectives.rs:982-994) sees only `$F3` and `bomb_secret`; every new entrance is stairs (data) | `Room::stairs_tiles` (room.rs:250-256), dungeon branch of `cave_tiles` |
| 3 | **In-cave rooms and families** for 0x44, 0x34, 0x21, 0x47 (and their overworld `cave` exits) | route steps need game.yaml rooms (test `route_room_exit_destinations_exist`) (code) | `(0, 0x2f, cave)`, `(0, 0x0f, cave)` (game.yaml:948-959, 302-) |
| 4 | **Buy waypoints** `waypoint: arrows` / `food`, done by inventory, missed on a wrong ware; `requires_rupees` on the two caves | `waypoint_done` reads flag `$10` set by any ware (objectives.rs:300-310; asm:Cave_State2TalkOrShopOrDoorCharge `@Take`); Lee's rule 5 | `waypoint: heart|rupees`, `treasure_tile`, `cheats` rupees rule |
| 5 | **Secret steps on the overworld**: `push` (gravestone 0x21), `armos` (0x34), `burn` (0x6d, 0x47), `recorder` pond (0x42): the target is the stand tile + facing, done when the stairs tile appears or flag `$80`/`$20` | no tile-object targets (architecture.md "Core"); game.yaml `entrance.how` + `tile` exist as data only (game_map.rs:233-256 "read elsewhere or by nothing yet") | `bomb_secret` (rock walls: the closest complete pattern, BombTarget reach rules) |
| 6 | **Item actions on for the families that need them**: CANDLE (0x6d, 0x47 families), FOOD (7:0x28 family), RECORDER on the overworld (`recorder_wanted` widened to the whirlwind steps and 0x42) | actions.rs:381-394; config.rs:353-380 | round 24's kinds and `items:` switches |
| 7 | **Whirlwind step**: `exit: recorder` / `entry: recorder`; the objective plays RECORDER k times facing a computed way, stands on the whirlwind's row; the chain accepts the jump into the landing room as the route's entry; observation of `$523` (and `IsWhirlwindSummoned` `$508`) | game_map.rs:263-265; objectives.rs:244 (None for jumps); draft_routes.py L11 | `exit: maze` loop handling (`MazeLoop` as a transition); RECORDER's menu-frame wait |
| 8 | **Grumble rule**: FOOD is the objective in 7:0x28 until the flag is set | level-7.md L74, L650 | `recorder_wanted` / Digdogger split reward pattern |
| 9 | **Arrow use rule**: `arrows: false` for every family but Gohma's and Ganon's (or a `rupee_reserve` like `bomb_reserve`) | config.rs:363 default true, no family sets it; asm:Arrow_Wield | `bomb_reserve`, `BombUse` modes |
| 10 | **One-way doors** in game.yaml (exits passable one way) so path searches and re-plans never aim back through them; bombs held ≥ 2 when leaving 9:0x61 N (a `bomb_reserve` step) | draft "One-way rooms"; level-9.md L703 | `shutter: true` (display only), `bomb_reserve` |
| 11 | **Start states that keep history**: no `fresh` for return visits (9:0x20, 0x61, 0x07, 0x10, 0x55, 0x05 back from cellars; 7:0x1a; 8:0x5e parts) | entry_states.py:327-341; room_entry.rs:91-110 | `harvest` states from chains (architecture.md "Replays"/harvest), round 24 §12 "played states" |
| 12 | **The end**: `exit: end` = success on `GameMode` `$13`; `Route::load` accepts a route that ends so | chain.rs:161-165 | `terminal` exit (`STEP_EXITS`), `final_triforce` |
| 13 | **Triforce of Power as the objective in 9:0x42** after Ganon (shutters on trigger 3) | game.yaml:6055-6056; level-9.md L621 | default `treasure` Take, `COMBAT_SHUTTER_TRIGGERS` |
| 14 | Lost Woods states and `maze:` block check for 0x61 | game_map.rs:944-983 | 0x1b (round 24 §4) |
| 15 | Recovery after a wrong push (ring block pushed E): objective sends Link out and back in | trace (1:0x22, game.yaml comment line 2559) | `retreat:` steps, bounce limit |

Order: 1-3 are prerequisites of any leg step; 4-6 of the D5→D7 leg; 7 of three legs; 8 of D7; 9 must land before
the first arrow is bought (0x44), or every later family wastes rupees; 10-11 before D9's states are built; 12-13 last.

---

## 6. Softlocks and irreversible losses on the draft route

| Where | Wrong action | Consequence | Guard / detect |
|---|---|---|---|
| 0x44 shop | touch X `$58` (magic shield 130) or X `$78` (bombs 20) | one ware per visit; 130 leaves 12-24 rupees: no arrows (80) → Gohma (arrow only, enemies.md "Gohma") impossible → D6 impossible; bombs leave 122-134 and need a second visit; food later may fail at the low end (inferred arithmetic) | mask MOVE onto the other wares' stand squares (X `$58`/`$78`, Y `$95`/`$9D`) in that cave, or `missed`; detect `InvMagicShield`, bomb count up |
| 0x34 shop | key (80) unaffordable after 0x44 at 62-74 (inferred); blue ring unaffordable | none if the order is kept | keep 0x44 before 0x34 |
| 0x47 take-any | potion | heart container gone for good (once-only person) → 14 hearts | built: `missed` "took the other ware" |
| 0x21 | arrive with < 12 containers | no sword now; the gravestone must be pushed again later (inferred) | route asserts 12 at 0x21 (audit hearts) |
| any overworld room | RECORDER outside the planned steps, or facing up/right | whirlwind index shifted: later whirlwinds go elsewhere (not a softlock, a long detour) | mask RECORDER on the overworld; compute plays from `$523` |
| whirlwind step | hit just before pickup, clock held, or moved off the row | whirlwind missed; no second one until the room is re-entered; index already stepped | detect whirlwind gone with Link still in the room; re-plan from `$523` |
| everywhere after 0x44 | ARROW at ordinary monsters | 1 rupee each; at 0 rupees: Gohma refuses, **Ganon's room is a dead end** (only death) | `arrows: false` except Gohma/Ganon families; detect rupees < arrows still needed |
| 0x34 out of the cave | Down at the exit spot | wakes the Armos again (1 heart, inferred) | stand tile rule |
| graveyard rooms | press into gravestones | immortal flying Ghinis (1 heart each) while the Ghini lives; after it dies they can't be removed | penalty/mask on pushing gravestone squares except the secret one |
| ring push rooms | push the W block E | stairs sealed until re-entry | `push_block: [N, S]` (built); add the re-entry recovery |
| trigger-5 rooms | block pushed into a lane | stairs walled off (8:0x1f trace) | push directions in game.yaml per room |
| 9:0x61 N | leave with < 2 bombs | 0x31 W and 0x04 W unreachable behind one-way shutters: stuck (cheat would rescue) | `bomb_reserve: 2` from 0x16 on; mask BOMB as a weapon in 0x51-0x04 |
| 9:0x42 | let the brown phase run out | the sword count restarts (not fatal) | — |
| 9:0x42 | no silver arrow (skipped 9:0x4f) | dead end | `waypoint: silver-arrow` (cellar waypoint pattern) |
| 7:0x28 | boomerang thrown while the food lies down | food removed (not consumed), place again | none needed |
| 7:0x0d | Wallmaster grab | back to 7:0x79, 13-15 rooms again (trace) | chain test |
| any | death | run over (chain); the game would continue | Lee's decision (§10) |
| 0x60 → 0x5f (plain way, built) | wrap without the ladder | softlock (trace) | built (ladder held from D4) |

---

## 7. Hard problems, ranked

1. **The whirlwind steps.** A route step whose action is "play twice, the second inside an 80-frame window, facing a
   given way, standing still on the whirlwind's row, not hurt". Today's families learn to cross rooms by reward;
   nothing in a one-room episode rewards a counter that pays off three legs later. Needs a scripted or masked step
   (compute k from `$523`/`InvTriforce`; offer only RECORDER and WAIT), the arrival as an entry, and the counter in
   the observation; otherwise one stray play makes three legs wrong with no failure signal until the landing room.
2. **Rupee discipline over four dungeons.** Two arrows needed, 2-14 rupees held, ARROW unmasked for every family by
   default. A family per enemy cannot see that an arrow at a Wizzrobe in level 6 costs Ganon's arrow. Needs a
   route-level reserve (like `bomb_reserve`) or arrows masked outside two rooms; the cheat rule must count shop
   top-ups (Lee's rule 5).
3. **Secrets on the overworld that need exact positions and leave hazards.** The gravestone (exact X, flying Ghinis
   that can't be killed), the Armos (side touch, an awake Armos next to the stairs), two trees (stand tile and facing
   to 8 px, a 94-frame wait, own fire). Rewards can't be shaped by distance alone (the target is a tile object, not an
   exit); a model learns them only with a dedicated objective per secret.
4. **Shops.** Exact-X ware touches next to wrong wares Link can afford; one ware per visit; the state after a wrong
   buy is legal but ruins the plan. Masking, not reward, is the safe tool.
5. **History-dependent rooms and their start states.** Fresh states teach a different room than the chain meets
   (Patra alive, Wizzrobes back). Curricula from fresh states will over-fight; chains will meet empty rooms the
   family never trained in (inferred).
6. **The end sequence**: a halted entry scene, a phase-timed boss, a pickup that opens the doors, an exact-Y trigger.
   Each is simple; together they need a final family and a terminal the engine doesn't have.
7. **Graveyard crossings** (0x41, 0x31, 0x21, back 0x31, 0x32): any wall-hug into a gravestone spawns an immortal
   flyer; the overworld families never met this (inferred).

---

## 8. What must be built (beyond §5's list: actions, masks, observations)

- **Observation:** `$523` (whirlwind index), `$508`, `$522` (summoned, carrying), `InvPowerTriforce` `$672`, the
  overworld tile object (type, position, state: armos/gravestone/tree/rock wall) and revealed stairs, flying Ghinis as
  enemies (they are, type `$22`, if the classifier keeps them), shop wares (ids and X; drawn through slot `$13`, not
  objects), Zelda's trigger. The rest (rupees, keys, bombs, arrows, candle grade, food, recorder, help/fairy counters,
  maze step) exists (observation.rs:339-407) (code).
- **Masks:** RECORDER on the overworld only at whirlwind steps and 0x42; ARROW by rupee reserve; MOVE onto wrong-ware
  stand squares; pushing gravestone squares other than the secret one; CANDLE where the stand tile isn't reached
  (optional). (code: `compute_action_mask`, actions.rs:311-379.)
- **Executor:** RECORDER on the overworld waits out the freeze already (architecture.md "Item kinds"); the whirlwind
  ride (Link halted ~the scroll), the Triforce of Power fanfare (`$C0` halted), Ganon's entry (~270 f), the shop's
  rupee count (160/120 f), mode `$13` must not hang the step loop (inferred).
- **Chain:** accept a whirlwind jump as the route's entry; accept `exit: end`; count shop rupee cheats.
- **State building:** keep history for return visits (harvested or "played" states), not `make_room_fresh`.

---

## 9. What to walk in the game before trusting this

1. 0x21: does pushing the secret gravestone itself release a flying Ghini; does the stairs square come back after the
   cave (asm:Secret_RestoreStairsOW's `$BC` test); Link's exit spot vs the stairs. (overworld.md: first walk failed on
   Ghinis.)
2. 0x34: the walk out of the cave: is the statue back, does Down from the exit spot wake it.
3. 0x44/0x34 with the engine's real rupee counts, wrong-ware touches from the walk-in path (X `$70` start, X `$78`
   middle ware = bombs at 0x44, blue ring at 0x34).
4. The whirlwind window: frames from the first B to the latest second B that still re-steps before pickup, from each
   of 0x0b, 0x22, 0x6d's exit spots; whether the classifier/executor interrupts on the whirlwind; the chain's view of
   the arrival.
5. 0x6d/0x47 tree burns by the CANDLE action from the draft's stand tiles (the draft walks used buttons with the B
   item chosen by RAM, first-quest.md "What was walked").
6. 0x61 Lost Woods entered from 0x51 (the draft's plain way was walked, Nov 14; the engine's maze for 0x61 never ran).
7. 9:0x42 and 9:0x32 under the engine (Ganon was fought "with Link placed beside him"; the end mode `$13` seen once).
8. Return-visit rooms from built states vs chain arrivals (9:0x20, 0x61, 0x07, 0x10).
9. The fights were not played in the walks (first-quest.md); every "forced fight" and every crossing hazard above is
   unmeasured.
10. 7:0x0d Wallmaster grab inside a chain run (what the chain records and whether it recovers to 0x0d).

---

## 10. Open questions for Lee

1. **Death:** should a run that dies continue (the game's CONTINUE: 3 hearts, level restart, everything kept), at
   least for measuring "can the game be finished", or does a death stay the end of a run?
2. **Rupees for arrows:** may the route add a reserve rule (arrows masked except Gohma/Ganon), and if the low end of
   2 rupees is spent, is a cheated rupee before Ganon acceptable under rule 5 (cheats are "at the two shops" today)?
3. **The whirlwind:** may its steps be scripted (the engine computes and plays them, the model only waits), as a
   route mechanic rather than a learned one? Same question for the overworld secrets (push, armos, burn, pond) and the
   shop purchases.
4. **Shops' wrong wares:** mask them (cleanest) or let a wrong buy end the run as `missed`?
