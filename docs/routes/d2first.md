# d2first: Dungeon 2 first, never the fairy — the route

Lee's route (Nov 8), the only route since the reset the same day (`docs/decisions.md` 2026-11-08: milestone1 and
milestone1-fairy removed). Data: `triforce/routes.yaml` `d2first`, rooms in `triforce/game.yaml`. View it with

```bash
.venv/bin/python zelda-map.py --route d2first      # rooms outlined, arrows, step numbers; sidebar per step
```

Not trained yet: next is a baseline retrain of every family from scratch with the current training setup, then
Lee's training ideas. Everything below was checked in the emulator (rooms loaded through `Game.load`, mode 5)
unless it says otherwise.

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
| 51–58 | 1:63 53 52 42 43 44 45 35 | S→N, then as before from 0x63 | stalfos-keese-gel, goriya, wallmaster, boss-aquamentus | 5 / Wh, boomerang at 0x44 |
| 59 | 1:36 | W → triforce: **route ends** | stalfos-keese-gel | 6 / Wh |

Choices and findings:
- **Row 5 to D2:** 0x5d N → 0x4d W → 0x4c N → 0x3c. 0x3c's only exits are S and the stairs
  (rows 0–2 are wall; the gap at row 3, cols 25–26 is an alcove), so 0x3d → 0x3c doesn't exist.
- **After D2:** the triforce puts Link on 0x3c, which has no N exit, so 0x3c → 0x2c goes round: 0x3c S, 0x4c E,
  0x4d N, 0x3d N, 0x2d W, 0x2c (Lee confirmed, Nov 8).
- **Dungeon 1 (Lee, Nov 8):** Link carries D2's three keys (0x7e, 0x4e, 0x3e) into D1, so the key-aware shortest
  path would go 0x73 N directly; the route still takes both key rooms like the old D1-first route: 0x73 W, 0x72
  back E, 0x73 E, 0x74 back W, 0x73 N. The objective follows the route's steps (below), so 0x73 sends Link W,
  then E, then N. The boomerang pickup at 0x44 stays.
- **0x1b is the mountain maze** (`Z_01 CheckMazes`): W is its only normal exit; E and S bring Link back into
  0x1b and N reaches 0x0b only on the fourth N in a row. game.yaml now lists only W (it had N/E/S from v1).

## The 0x2c secret

- OW attr B of 0x2c = $47: cave index $11, the take-any cave (person $6B, text selector $20).
- A rock-wall tile object ($63 in slot 11, `Z_04 UpdateRockWall`) at x=144, y=160: tiles (18–19, 12–13) on the
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

New families (Lee, Nov 8): `ow-lynel`, `ow-tektite`, `ow-boulder`. `ow-leever-peahat` is gone with milestone1 (its
other rooms 0x29–0x2b are not on this route).

## Code

- **The objective follows the route step by step** (`zelda::objectives::GameCompletion`): on every room change it
  finds the step entering Link's room from the side he came in (`route_entry`: N/S/E/W, `S` into a cave or down
  stairs, `cave` out of one or after the triforce's warp) with exactly the progress the route has there
  (`StepProgress`: triforce pieces and sword). A step leaving through a room edge makes that exit the correct one
  (plus the alternate 0x78 at 0x77). Off the route — no matching step, e.g. 0x1a from N without the white sword —
  and on cave, stairs and triforce steps, the key-aware shortest path to the target (sword, heart waypoint,
  triforce in route order) decides, as before. One-room training (`start_route`) picks the step the same way from
  the start state's entry; rooms the route crosses twice (0x77, 0x3c, 0x4c, 0x4d, 0x2c, 0x1a, 0x0a, 1:0x73) are
  entered from a different side each time, so each step has its own start states and resets play every one.
- `retreat: S` on step 35 (see below); routes end on their last triforce (no `terminal` exits any more).
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

## Lee's answers (Nov 8)

1. **Dungeon 1 grabs both keys**: route-step objectives (above), in one-room training and in chains.
2. **0x2c (and its cave) belongs to `ow-tektite`.** ow-leever-peahat is removed.
3. **The old routes are removed**, with their states, the fairy waypoint, spliced routes, terminal exits and the
   tests and defaults that named them. The only alternate left is 0x78 (it branches from 0x77, a d2first room). The
   Dungeon 1 bow-side rooms 0x33/0x23 (and the bow room 0x22) branch from 0x43, a d2first room, so they stay: tagged
   in stalfos-keese-gel / goriya (0x22 untagged), states rebuilt with `entry_states path` from `1_43w_N`.
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
