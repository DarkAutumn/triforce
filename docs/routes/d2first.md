# d2first: Dungeon 2 first, never the fairy

Lee's route (Nov 8). Data: `triforce/routes.yaml` `d2first`, rooms in `triforce/game.yaml`. View it with

```bash
.venv/bin/python zelda-map.py --route d2first      # rooms outlined, arrows, step numbers; sidebar per step
```

Not trained. Everything below was checked in the emulator (rooms loaded through `Game.load`, mode 5) unless it
says otherwise.

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
| 29–31 | 0:2c, 2c cave, 2c | E→cave (bomb the rock wall), S→S (`waypoint: heart`), cave→N | ow-leever-peahat | 4 → 5 / W |
| 32–33 | 0:1c, 1b | S→W, E→W | ow-start | 5 / W |
| 34 | 0:1a | E→N | ow-tektite | 5 / W |
| 35–37 | 0:0a, 0a cave, 0a | S→cave, S→S (white sword), cave→S | ow-lynel | 5 / W → Wh |
| 38 | 0:1a | N→W | ow-tektite | 5 / Wh |
| 39–41 | 0:19, 18, 17 | E→W, E→W, E→S | ow-boulder | 5 / Wh |
| 42–44 | 0:27, 28, 38 | N→E, W→S, N→W | ow-zora | 5 / Wh |
| 45 | 0:37 | E → cave (D1 stairs) | ow-octorok | 5 / Wh |
| 46–54 | 1:73 63 53 52 42 43 44 45 35 | S→N, then as milestone1 from 0x63 | stalfos-keese-gel, goriya, wallmaster, boss-aquamentus | 5 / Wh, boomerang at 0x44 |
| 55 | 1:36 | W → triforce: **route ends** | stalfos-keese-gel | 6 / Wh |

Choices and findings:
- **Row 5 to D2:** 0x5d N → 0x4d W → 0x4c N → 0x3c, as milestone1-fairy. 0x3c's only exits are S and the stairs
  (rows 0–2 are wall; the gap at row 3, cols 25–26 is an alcove), so 0x3d → 0x3c doesn't exist.
- **After D2:** the triforce puts Link on 0x3c, which has no N exit, so 0x3c → 0x2c goes round: 0x3c S, 0x4c E,
  0x4d N, 0x3d N, 0x2d W, 0x2c (Lee confirmed, Nov 8).
- **Dungeon 1:** Link carries D2's three keys (0x7e, 0x4e, 0x3e) into D1, so the objective (key-aware shortest path)
  goes 0x73 N directly; milestone1's detours to the key rooms 0x72 and 0x74 are skipped. The boomerang pickup at
  0x44 stays. Keys: 3 → 2 (0x73 N) → 3 (0x53) → 2 (0x52 N) → 1 (0x43 E) → 2 (0x45) → 1 (0x45 N).
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
  heart-container`, `treasure_tile: [19, 11]`. ow-leever-peahat has bombs on (`items: {bombs: true}`).
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
| 0:2c cave | potion / heart container | ow-leever-peahat (a cave goes with its room, like 0x77's) |
| 2:0d | none (triforce) | d2-rope (like 1:36 in stalfos-keese-gel) |
| 0:07, 16, 26 | RedLynel; boulders; Peahat ×6 + Zora | untagged neighbours (game.yaml only) |

New families (Lee, Nov 8): `ow-lynel`, `ow-tektite`, `ow-boulder`, configs in `specialists/` from the ow-moblin
template, untrained; their start-state globs (`*_[0-9]*`) take the d2first states.

## Code

- Objectives take the route's triforce steps in route order (milestone1 unchanged: D1 is its only triforce step).
- An overworld cave never sends Link into another overworld cave's room (the white-sword target no longer matches
  0x2c's cave).
- `start_route` keeps only routes whose triforces before the start step match Link's pieces (a milestone1 D1 state
  never gets d2first's objective, and back).
- Chain runs of a route without a `terminal` step end on its last triforce (`Route.terminal` is optional,
  `Route.final_triforce`).
- `entry_states build --route NAME`; bombs are on in `requires_bombs` rooms while building.
- `zelda-map.py --route NAME`.

## States

Chains 10, 11, 12 (`python -m triforce.entry_states build --route d2first --chains 10,11,12`), named
`<level>_<room><entry>_<chain>` as usual. The two-digit chain keeps them out of the existing families'
`*_[0-9]` start-state globs, so no current training changes when they land; the new families' globs take them.
Bombs: natural counts along the route (0 until D2, 8 at Dodongo from D2 pickups, the cheated bomb only where
`requires_bombs`), keys as carried (3 from D2).

168 states, 56 per chain: every step after the game start (55) plus the 0x78 alternate. Chain 11's Moldorm crossing
(2:0x3e) needed a second build from `2_3es_11` with `--iterations 8000 --patience 400`; chains 10 and 12 rebuilt
Dungeon 1 from `1_73s_1x` after the route's D1 change. `entry_states audit --states '*_1[0-9]'`: 168 states, 0
problems (the audit now gives caves no monsters of their own and leaves the boulders out of a room's entry
count, like Zoras); the default audit of every configured state: 202, 0 problems. Max hearts in the states:
3 through D2 (`2_0es_1x`), 4 after Dodongo (`2_0de_1x`), 5 after the 0x2c cave (`0_2cc_1x`) through Aquamentus
(`1_35s_1x`), 6 in `1_36w_1x`. Dodongo and the rock wall were both done by Go-Explore with natural bombs.

## Open questions for Lee

1. **D1 without the key detours:** with D2's 3 keys the route skips 0x72/0x74 (see above). Keep, or force
   milestone1's exact D1 path (the objective would need route-order visits)?
2. **0x2c's family:** 0x2c is a Tektite room (RedTektite ×4). It stays in ow-leever-peahat because milestone1 uses
   it; move it (and its cave) to ow-tektite?
3. **Lynel pirouette:** the usual human play at 0x0a is to leave and re-enter until the Lynel spawns somewhere
   better (not between the water and the cave door). May the model do that?
4. **Bombs in milestone1's 0x2c:** `requires_bombs` also gives Link a cheated bomb when milestone1 crosses 0x2c
   (W→E), and ow-leever-peahat now has bombs; the cave there is a wrong exit. Fine?
5. **Shared rooms:** where d2first and milestone1 enter a room from the same side with the same triforce pieces
   but leave differently (0x58 from S: N vs E), one-room training picks a route at random, as with the fairy route.
6. The d2first states are outside the existing families' globs; include them when retraining on d2first?
