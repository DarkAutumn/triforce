# d2first: Dungeon 2 first, never the fairy — the route

Lee's route (Nov 8), the only route since the reset the same day (`docs/decisions.md` 2026-11-08: milestone1 and
milestone1-fairy removed); the Dungeon 1 bow detour joined it on Nov 9, and the same day the route went on from the
D1 triforce to Dungeon 3 (the raft, Manhandla); it now ends on the D3 triforce. Data: `triforce/routes.yaml`
`d2first`, rooms in `triforce/game.yaml`. Dungeon 4 is scouted as the route `d2first-d4`, which continues this one
(`extends: d2first`; `docs/routes/d4.md`, Nov 11): `d2first` itself is unchanged. View it with

```bash
.venv/bin/python zelda-map.py --route d2first      # rooms outlined, arrows, step numbers; sidebar per step
```

The whole route is trained since round 16 (r56/r57): from the start, D3 reached 34/120 and its triforce 4/120. Everything
below was checked in the emulator (rooms loaded through `Game.load`, mode 5) unless it says otherwise.

## Route

Hearts are heart containers on entering the room; sword W = wood, Wh = white.

| Steps | Room (level:room) | Entry → exit | Family | Hearts / sword / items |
|---|---|---|---|---|
| 0–2 | 0:77, cave, 77 | start → cave → S → N | ow-start | 3 / none → W |
| 3–5 | 0:67, 68, 58 | S→E, W→N, S→**E** | ow-octorok | 3 / W |
| 6–7 | 0:59, 5a | W→E | ow-zora | 3 / W |
| 8–11 | 0:5b, 5c, 5d, 4d | W→E, W→E, W→N, S→W | ow-moblin | 3 / W |
| 12–13 | 0:4c, 3c | E→N, S→cave (D2 stairs) | ow-octorok | 3 / W |
| 14–21 | 2:7d 7e 6e 5e 4e 3e 2e 1e | as milestone1 (keys 0x7e/0x4e/0x3e, bombs at 0x1e) | d2-rope, goriya, moldorm | 3 / W, 3 keys, bombs |
| 22 | 2:0e | S → W (Dodongo, heart container) | boss-dodongo | 3 / W |
| 23 | 2:0d | E → triforce (warps to 0:3c) | d2-rope | 4 / W |
| 24–28 | 0:3c, 4c, 4d, 3d, 2d | cave→S, N→E, W→N, S→N, S→W | ow-octorok, ow-moblin, ow-zora | 4 / W, triforce 2 |
| 29–31 | 0:2c, 2c cave, 2c | E→cave (bomb the rock wall), S→S (`waypoint: heart`), cave→N | ow-tektite | 4 → 5 / W |
| 32–33 | 0:1c, 1b | S→W, E→W | ow-start | 5 / W |
| 34 | 0:1a | E→N | ow-tektite | 5 / W |
| 35–37 | 0:0a, 0a cave, 0a | S→cave (`retreat: S`), S→S (white sword), cave→S | ow-lynel | 5 / W → Wh |
| 38 | 0:1a | N→W | ow-tektite | 5 / Wh |
| 39–41 | 0:19, 18, 17 | E→W, E→W, E→S | ow-boulder | 5 / Wh |
| 42–44 | 0:27, 28, 38 | N→E, W→S, N→W | ow-zora | 5 / Wh |
| 45 | 0:37 | E → cave (D1 stairs) | ow-octorok | 5 / Wh |
| 46–50 | 1:73, 72, 73, 74, 73 | S→W, E→E (key), W→E, W→W (key), E→N | stalfos-keese-gel | 5 / Wh, 3 → 5 keys |
| 51–55 | 1:63 53 52 42 43 | S→N, S→W, E→N, S→E, W→**N** | stalfos-keese-gel | 5 / Wh, 5 → 4 keys |
| 56–63 | 1:33 23 22 7f(cellar) 22 23 33 43 | the bow detour (below): S→N (key), S→W (key), E→stairs, S→N (bow), stairs→E, W→S, N→S, N→E | stalfos-keese-gel, goriya, **trap-room**, **equipment-room** | 5 / Wh, bow |
| 64–66 | 1:44 45 35 | W→E (boomerang), W→N, S→E | goriya, wallmaster, boss-aquamentus | 5 / Wh, bow |
| 67 | 1:36 | W → triforce (warps to 0:37) | stalfos-keese-gel | 6 / Wh |
| 68–79 | 0:37 38 48 58 57 56 55 65 64 63 73 74 | to Dungeon 3 (below): cave→E, W→S, N→S, N→W, E→W, E→W, E→S, N→W, E→W, E→S, N→E, W→cave | ow-octorok, ow-zora, **ow-leever**, ow-moblin, ow-tektite | 6 / Wh, triforce 2+1 |
| 80–96 | 3:7c … 4d | Dungeon 3 (`docs/routes/d3.md`): the raft cellar 0x0f, the bomb wall into Manhandla 0x4d | **zol**, **darknut**, trap-room, equipment-room, **boss-manhandla** | 6 → 7 / Wh, raft |
| 97 | 3:3d | S → triforce: **route ends** | zol | 7 / Wh |

Choices and findings:
- **Row 5 to D2:** 0x5d N → 0x4d W → 0x4c N → 0x3c. 0x3c's only exits are S and the stairs
  (rows 0–2 are wall; the gap at row 3, cols 25–26 is an alcove), so 0x3d → 0x3c doesn't exist.
- **After D2:** the triforce puts Link on 0x3c, which has no N exit, so 0x3c → 0x2c goes round: 0x3c S, 0x4c E,
  0x4d N, 0x3d N, 0x2d W, 0x2c (Lee confirmed, Nov 8).
- **Dungeon 1 (Lee, Nov 8):** Link carries D2's three keys (0x7e, 0x4e, 0x3e) into D1, so the key-aware shortest
  path would go 0x73 N directly; the route still takes both key rooms like the old D1-first route: 0x73 W, 0x72
  back E, 0x73 E, 0x74 back W, 0x73 N. The objective follows the route's steps (below), so 0x73 sends Link W,
  then E, then N. The boomerang pickup at 0x44 stays.
- **0x1b is the mountain maze** (`asm:Maze_CheckStep`): W is its only normal exit; E and S bring Link back into
  0x1b and N reaches 0x0b only on the fourth N in a row (four Ns from inside 0x1b; walked in the emulator, Nov 9).
  game.yaml lists N/E/S as exits back into 0x1b plus a `maze:` block (`docs/routes/overworld.md`).

## The 0x2c secret

- OW attr B of 0x2c = $47: cave index $11, the take-any cave (person $6B, text selector $20).
- A rock-wall tile object ($63 in slot 11, `asm:RockWall_Update`) at x=144, y=160: tiles (18–19, 12–13) on the
  bottom face of the middle rock. A bomb at Link (144, 169) facing N opens it (tile $24, room flag $80); walking N
  into it enters the cave.
- Cave wares (`CaveItemIds` $422): $20 (potion, x=88) / none / **$1A heart container (x=152)**. Taking the right
  one raises max hearts 4 → 5; the left one leaves them at 4. Either sets 0x2c's item-taken flag ($10).
- game.yaml: 0x2c `requires_bombs: true`, `bomb_secret: [18, 12]`, `cave` exit; (0x2c, cave) `treasure:
  heart-container`, `treasure_tile: [19, 11]`. ow-tektite (the room's family) has bombs on (`items: {bombs: true}`).
- Objective: until the heart is taken (`waypoint: heart`, active after the D2 triforce and before D1's) the rock
  wall tiles are the cave entrance and the cave's target is the right ware. Leaving 0x2c other than into the cave
  before the heart, or taking the potion, is `missed`: no exit is correct (training: exit-wrong), and a chain run
  ends as a failure (room outcome `missed`).
- 0x0a's cave (attr B $4B, index $12, person $6C) holds the white sword ($02) in the middle; the old man hands it
  over only at 5+ containers (`objectives::SWORD_UPGRADES`).

## The bow detour (Lee, Nov 9)

From 0x43 (entered from W) the route goes N through the Stalfos key room 0x33 and the Goriya key room 0x23 into the
blade-trap room 0x22, pushes its west block, takes the stairs down to the item cellar 0x7f for the bow, climbs back
out and returns the same way (0x22 E, 0x23 S, 0x33 S) to 0x43, which it leaves E to 0x44 as before.

| Step | Room | Entry → exit | Family | Notes |
|---|---|---|---|---|
| 55 | 1:0x43 | W → N | stalfos-keese-gel | was W → E; N door open (attr A N type 0) |
| 56 | 1:0x33 | S → N | stalfos-keese-gel | Stalfos ×3, key (visible); N door locked (key door, shared with 0x23 S) |
| 57 | 1:0x23 | S → W | goriya | Red Goriya ×3, key when all are dead (trigger AllDeadItem); W door locked (shared with 0x22 E) |
| 58 | 1:0x22 | E → stairs (`cave`) | **trap-room** | push the west block N or S, then the stairs |
| 59 | 1:0x7f (cellar, `in_cave`) | S → N | **equipment-room** | `waypoint: bow`: take the bow, then up the left ladder |
| 60 | 1:0x22 | stairs (`cave`) → E | trap-room | Link comes out at (96, 173), below the (reset) block |
| 61 | 1:0x23 | W → S | goriya | doors stay open (room flags) |
| 62 | 1:0x33 | N → S | stalfos-keese-gel | |
| 63 | 1:0x43 | N → E | stalfos-keese-gel | locked E door to 0x44 (as before) |

Keys: +1 (0x33) −1 (0x33 N door) +1 (0x23) −1 (0x23 W door), so Link leaves the detour with the keys he brought.

Mechanics (emulator, rooms loaded through `Game.load` and read at mode 5, the cellar at mode 9; disassembly
`asm:Trap_Init_Full`, `asm:Trap_Update_Full`, `asm:PushBlock_Update`, `asm:Warp_CheckUnderLink`,
`asm:Warp_CheckSubroomExit`, `Mode9Cellar_Init_*`):
- **0x22** has no monsters: a trap generator ($4A) makes four blade traps ($49, object slots 1–4) in the corners
  (32, 93), (32, 189), (208, 93), (208, 189). A trap slides at Link when he lines up with it (|dy| < 14:
  horizontally, else |dx| < 14: vertically), only inward (`Trap_AllowedDirs`), fast (1.75 px/frame) to near the
  room's centre line ($78 / $90) and slowly (0.5 px/frame) back. They hurt on touch, can't be harmed and are not
  monsters (room state `enemies_remaining` 0); the observation shows them as entities `0x49` with `hurts_on_touch`
  1, `killable` 0. A scripted path along the bottom (door, W to x 176, S to y 173, W 144, S 181, W 112, N 173,
  W 96, then N to push) went through unhurt in 20 of 20 seeds with raw input (`target/traps/path.py`).
- **The stairs** (tiles $70 $72 / $71 $73) are there from the start at tiles (16, 10)–(17, 11), the centre of a
  diamond of eight blocks ($B0–$B3); the room's secret trigger is none (attr F & 7 = 0). The west block, tiles
  (12, 10)–(13, 11) at (96, 144), is the room's push block (attr D bit $40, object $68 in slot $B). `PushBlock_Update`
  moves it after 16 frames of pushing toward it while aligned, in **any** direction (the room needs no kills:
  `IsRoomAllDead`). Pushed **N** (from below, Link at (96, 157) walks half into it, y 149) or **S** (from above, Link at
  (96, 125)) it opens the way: (12, 10) → (14, 10) → the stairs. Pushed **E** it lands on (14, 10) and walls the
  stairs in for the rest of the visit (it moves once; re-entering the room resets it). W needs Link inside the
  diamond. A MOVE must last 16+ frames (`MOVE N 16`): a released button resets the push timer.
- **Down the stairs:** standing square-aligned on the top-left stairs tile (Link at (128, 141)) starts mode $10, then
  mode 9 with RoomId = the cellar ($7F, `LevelInfo_CellarRoomIdArray`; CellarSourceRoomId $527 = $22). Mode 9 lays
  the cellar out (submodes 0–8, IsUpdatingMode 0, the old room's tiles still in RAM) and walks Link down to (48, 93),
  the top of the left ladder; it is playable once IsUpdatingMode is 1 (about 100 frames).
- **The cellar 0x7f** is side view: walkable tiles are the walkways ($24) and ladders ($6F), the rest ($F3
  background, $FA brick) blocks, so the ordinary tile collision is the whole movement model (no gravity). Left
  ladder: columns 6–7 from the top row down to the floor (row 17, columns 4–27); right ladder: columns 22–23, rows
  12–17, up to the bow's walkway (row 11, columns 14–27). The bow lies at (128, 144) (Link takes it at (128, 141));
  four Blue Keese ($1B) spawn with it. Its item-taken flag is 0x7f's.
- **Out:** pressing up at the top of the left ladder (Y < $40, X < $80: room A = attr A $22) leaves through mode $A
  (RoomId already $22) to 0x22, Link at (96, 173) below the west block, which is pending again.

Code (`docs/architecture.md`): a cellar is a sub-room like a cave (`MapLocation.in_cave`, under its own room id),
entered `S` and left `cave` like a cave; `GameView::in_cellar` (mode 9); the executor skips mode 9's entry; the
cellar's exit is its top row (N); 0x22's objective points at the push spots (game.yaml `push_block: [N, S]`) while
the block is pending, then at the stairs; `reward-pushed-block` pays for a push in a listed direction;
`StepProgress` tracks the bow, so 0x22/0x23/0x33/0x43 without it are off the route and lead back to the cellar.

## New rooms and families

| Room | Enemies (emulator) | Family |
|---|---|---|
| 0:0a | BlueLynel ×1, Zora ×1; cave mouth at tiles (4, 2) | **ow-lynel** (new) |
| 0:0a cave | white sword | ow-lynel |
| 0:1a | RedTektite ×6, Zora ×1 | **ow-tektite** (new) |
| 0:1b, 0:1c | none | ow-start (empty rooms, like 0x39) |
| 0:17, 18, 19 | boulder generator ($31) ×1, boulders ($32) ×2–3, Zora ×1 | **ow-boulder** (new) |
| 0:27 | Peahat ×6, Zora ×1 | ow-zora (same as 0x28) |
| 0:28 (re-checked) | Peahat ×6, Zora ×1 | ow-zora |
| 0:38 (re-checked) | RedOctorok ×5, BlueSlowOctorok ×1, Zora ×1 | ow-zora |
| 0:2c, 2c cave | RedTektite ×4; potion / heart container | **ow-tektite** (Lee, Nov 8; was ow-leever-peahat) |
| 2:0d | none (triforce) | d2-rope (like 1:36 in stalfos-keese-gel) |
| 0:07, 16, 26 | RedLynel; boulders; Peahat ×6 + Zora | untagged neighbours (game.yaml only) |
| 1:22 | 4 blade traps ($49), no monsters; push block, stairs | **trap-room** (new, Nov 9) |
| 1:7f (cellar under 0x22) | Blue Keese ×4; the bow | **equipment-room** (new, Nov 9) |

New families (Lee, Nov 8): `ow-lynel`, `ow-tektite`, `ow-boulder`; (Lee, Nov 9) `trap-room` (blade-trap rooms) and
`equipment-room` (item cellars). `ow-leever-peahat` is gone with milestone1 (its other rooms 0x29–0x2b are not on
this route).

## Code

- **The objective follows the route step by step** (`zelda::objectives::GameCompletion`): on every room change it
  finds the step entering Link's room from the side he came in (`route_entry`: N/S/E/W, `S` into a cave, down
  stairs or down a cellar's, `cave` out of one or after the triforce's warp) with exactly the progress the route
  has there (`StepProgress`: triforce pieces, sword, bow and raft). A step leaving through a room edge makes that
  exit the correct one (plus the alternate 0x78 at 0x77). Off the route — no matching step, e.g. 0x1a from N
  without the white sword, or 0x22 out of the cellar without the bow — and on cave, stairs and triforce steps, the
  key-aware shortest path to the target (sword, heart then bow or raft waypoint, triforce in route order) decides.
  One-room training (`start_route`) picks the step the same way from the start state's entry; rooms the route
  crosses twice (0x77, 0x3c, 0x4c, 0x4d, 0x2c, 0x1a, 0x0a, 0x37, 0x38, 0x58, 1:0x73, 0x43, 0x33, 0x23, 0x22, 3:0x5b,
  0x5a, 0x69; 3:0x59 three times) are entered from a different side each time, so each step has its own start
  states and resets play every one.
- `retreat: S` on step 35 (see below); routes end on their last triforce (no `terminal` exits any more): since Nov 9
  the D3 triforce (`chain::Route::final_triforce` = 3; the D1 and D2 triforces warp Link on).
- `treasure:` on a step (Nov 9, `game_map::TreasureRule`): `skip`, or `{if_bombs_below: N}`; without it the room's
  treasure is taken as before. A step leaving through an intact game.yaml bomb wall targets the spot to bomb it
  from (`objectives::bomb_spot`, `docs/routes/d3.md`).
- An overworld cave never sends Link into another overworld cave's room. The heart waypoint reads the overworld
  room flags directly, so a Dungeon 1 state knows the heart was taken.
- `entry_states build` (default route d2first); bombs are on in `requires_bombs` rooms while building.

## States

Rebuilt after the reset (Nov 8) as chains 0, 1, 2 (`python -m triforce.entry_states build --chains 0,1,2`, route
d2first by default), named `<level>_<room><entry>_<chain>`; the D1-first states and the old d2first chains 10–12
are gone (a copy of the old directory is in the gitignored `target/states-milestone1/`). Bombs: natural counts
along the route (0 until D2, D2 pickups after, the cheated bomb only where `requires_bombs`); keys as carried (3
from D2, 5 after D1's key rooms).

190 files: `start` plus 63 per chain: every step after the game start (59) plus the 0x78 alternate, and Dungeon 1's
bow-side rooms (`entry_states path --source 1_43w_N --rooms 1:0x33,1:0x23,1:0x22 --chain N`: `1_33s_N`,
`1_23s_N`, `1_22e_N`). Chain 1 stopped at the Moldorm crossing (2:0x3e) and chain 2 at the Goriya shutter room
(2:0x1e) with the default 3000 iterations; both finished from there with `--start-step 19 --source 2_3es_1` /
`--start-step 21 --source 2_1es_2 --iterations 8000 --patience 400`. The bow room 0x22 needed one re-crossing in
chain 0. `entry_states audit` (every configured state, `--loads 3`): 187 states, 0 problems; the shipped-state
test checks each state's max hearts against the route: 3 through D2, 4 after Dodongo, 5 after the 0x2c cave
through Aquamentus, 6 in `1_36w_N`. Dodongo and the rock wall were both done by Go-Explore with natural bombs.

Curriculum trajectories (gitignored; the milestone1 ones moved to `target/trajectories-milestone1/`), re-explored
with `python -m triforce.explore --specialist NAME --parallel 16 --out <repo>/trajectories`: ow-zora 18/18,
ow-moblin 18/18, d2-rope 18/18, wallmaster 3/3, boss-aquamentus 3/3, goriya 10/12 then the two `2_1es_0/1`
with `--iterations 12000 --patience 400` (12/12), moldorm 0/3 then 3/3 with the same settings, boss-dodongo 3/3 plus
`--variant` seeds 1–9 (3/3 each): 30 tracks, of which 4 swallow kills (max BombHits > 0, `target/dodongo/how_killed.py`:
`2_0es_1`, `2_0es_1~s6`, `2_0es_0~s3`, `2_0es_2~s3`) were dropped, leaving 26 (every start state still has tracks).
The other families have no curriculum.

**The bow detour (Nov 9):** every state from step 56 on was rebuilt per chain with `python -m triforce.entry_states
build --chains N --start-step 55 --source 1_43w_N` (default 3000 iterations; Go-Explore crossed 0x22 in 4–8k env
steps per chain; 0x23 → 0x22 took 50–88k), 12 per chain: `1_33s`, `1_23s`, `1_22e`, `1_7fcave`, `1_22c`, `1_23w`,
`1_33n`, `1_43n`, `1_44w`, `1_45w`, `1_35s`, `1_36w` (the old path-built `1_33s/1_23s/1_22e` and the D1 states after
them are in the gitignored `target/states-pre-bow/`). Link carries 4 keys through the detour (3 into 0x44) and the
bow from `1_22c` on. Now 205 files (`start` plus 68 per chain); `entry_states audit --loads 3`: 205 states, 0
problems (the cellar spawns its 4 Keese). Trajectories (old D1 ones in `target/trajectories-pre-bow/`): trap-room
6/6, equipment-room 3/3, wallmaster 3/3, boss-aquamentus 3/3, goriya's D1 states 9/9 (`1_23s`, `1_23w`, `1_44w`).

## Lee's answers (Nov 8)

1. **Dungeon 1 grabs both keys**: route-step objectives (above), in one-room training and in chains.
2. **0x2c (and its cave) belongs to `ow-tektite`.** ow-leever-peahat is removed.
3. **The old routes are removed**, with their states, the fairy waypoint, spliced routes, terminal exits and the
   tests and defaults that named them. The only alternate left is 0x78 (it branches from 0x77, a d2first room). The
   Dungeon 1 bow-side rooms 0x33/0x23 (and the bow room 0x22) branch from 0x43, a d2first room, so they stayed
   (until Nov 9 off the route, states from `entry_states path`); since Nov 9 they are the route's bow detour.
4. **Lynel re-entry is allowed** — current design (Lee's follow-up, Nov 8):
   - Step 35 (0x0a from S, before the white sword) has `retreat: S`: leaving S is not a wrong exit but a `retreat`.
   - **One-room training** (ow-lynel): the episode ends with the ending `retreat` (re-entry can't happen in a
     one-room episode). Its outcome is configurable per specialist through `endings:`; the default is the new
     `neutral` outcome (no terminal reward or penalty, no wrong-location penalty), but **ow-lynel ships
     `endings: {retreat: soft-failure}`** for the first baseline, the same as a wrong exit, so a model can't learn
     to leave and reset for no reason.
   - **Chains**: the run continues (room outcome `retreat`); 0x1a without the white sword is off the route, so its
     objective sends Link back N. Runs count retreats (`RunResult.retreats`); the fourth (`MAX_RETREATS` = 3) ends
     the run as a failure, room outcome `retreat-limit`.
   - **Open:** whether a model can learn to use the retreat well. Lee will study Lynel spawn positions after the
     baseline and may switch ow-lynel's retreat to neutral, possibly only when the Lynel blocks the way to the cave.
5. **States**: every state rebuilt on d2first as chains 0, 1, 2 (below).
6. **Rooms crossed twice**: picking among the matching route steps at random is right; with the entry side and
   progress every step of such a room has its own states, so every step is reachable in training.
