# d3first: Dungeon 3 first, the wood sword to the grave — the route as built

The owner's second route (Nov 19; `docs/decisions.md` 2026-11-19 "Two routes"), the main route since 2.0.0.0
(`triforce/registry.py` `MAIN_ROUTE`), **corrected by the owner on Nov 21** (decisions 2026-11-21 "The program";
round 36): Dungeons 3, 4, 1, 0x2c's heart, Dungeon 2, 0x0f's 100 rupees, Dungeon 5, the whirlwind
to Level 4's screen, the raft to 0x55, **a dock warp there** to the meat under the Armos (0x34), the arrows (0x44),
Dungeon 7 (wood sword, 9 containers), the world wrap at 0x60, the hearts at 0x5f and 0x2f, **a dock warp at 0x3f**
into the graveyard (0x31), the magic sword (12), Dungeons 6, 8, 9 (0x47's heart and the red ring kept; no blue
candle: the red one from Dungeon 7). The wood sword until the magic sword (no white sword). The first walks
(`docs/routes/d3first-walks-a.md`, `d3first-walks-b.md`, scripts `scripts/experiments/r32_*.py`, `r32b_*.py`) were of
the first order; the corrected middle is walked in `d3first-walks-c.md`. **Round 38 (2.0.0.7, the owner): 0x2d's
bomb-wall rupee cave left the route** (0x2d crossed W → S; 0x3d's Armos cave stays; round38.md "0x2d dropped").
Data: `triforce/routes.yaml` `d3first` (290 steps), rooms in `triforce/game.yaml`; its start states are
`d3first_<name>` (`states: d3first`, `docs/architecture.md` "Routes").

```bash
.venv/bin/python zelda-map.py --route d3first
.venv/bin/python -c "from triforce import _native as n; print(len(n.route_steps('d3first')))"
```

## How it is written: shared blocks

Every dungeon's inside but Dungeon 3's way in, and the legs both routes take, are written once in routes.yaml's
top-level `blocks:` and spliced into both routes by `{include: NAME}` (`zelda::game_map` `parse_element`; the lead's
decision, Nov 19: "dungeon insides are written once"). `d2first` parses to exactly the steps it had before the
blocks (`tests/fixtures/d2first_steps.json`, every field of every step: `tests/test_game_map.py::
test_d2first_is_its_golden_list`). A change to a block reaches both routes. What differs between the routes inside
a block (the sword, heart containers, items, pieces, the bag) is in their start states, not in route data; a route
sets its own reserves on its own steps before a block, and a block's own reserve steps hold on both.

| Block | Steps on d3first | Rooms |
|---|---|---|
| `dungeon-3` | 14-23 | 3:0x69 from 0x59 (whose fight from N, step 13, is each route's own: d3first's reserve is 0 there) to the triforce: the raft cellar, back E by 0x59 and 0x5a, 0x5b's last pass, 0x5c, 0x4c, Manhandla, 0x3d |
| `to-level-4` | 24-30 | 0:0x74 cave → W to 0x45's mouth (the raft from 0x55) |
| `dungeon-4` | 31-52 | 4:0x71 to 0x03, the stretch `d4-island` |
| `dungeon-1` | 61-78 | 1:0x73 out S to 0:0x37 and back in, N by 0x63, 0x53's wall, the bow, to 0x36 (phase IX's) |
| `dungeon-2`, `dungeon-2-north` | 95-97, 101-104 | 2:0x7d to 0x6e; 0x2e to 0x0d. 0x5e, 0x4e and 0x3e (98-100) are each route's own (rounds 37 and 39): d3first crosses 0x5e and the Moldorm, d2first fights both |
| `dungeon-5` | 123-144 | 5:0x66 to 0x14 (the entrance, step 122, is each route's own) |
| `pond-42` | 154-157 | 0:0x54 to 0x42's pond (round 36: split from `to-level-7`, whose head d2first alone plays) |
| `dungeon-7`, `dungeon-7-east` | 159-177 | 7:0x79 to 0x2b; the entrance (158) and 0x1b (170) are each route's own (round 34): d3first keeps the bag for 0x1b |
| `graveyard` | 191-196 | 0:0x21's gravestone, the magic sword (12 containers), 0x31, 0x32, 0x22 (round 36: split from `to-level-6`) |
| `dungeon-6` | 197-212 | 6:0x79 to 0x0c |
| `to-level-8` | 213-218 | 0:0x22 (two plays facing up: Link turns first) to 0x6d's burned tree |
| `dungeon-8` | 219-228 | 8:0x7e to 0x2c |
| `to-level-9` | 229-243 | 0:0x6d (one play facing down) to 0x05: 0x47's tree and heart (15), the ladder at 0x17 |
| `dungeon-9`, `dungeon-9-from-55` | 244-245, 247-289 | 9:0x76, 0x66; 0x55 to Zelda's room (`exit: end`): the red ring, the silver arrow, Ganon. 0x56 (246) is each route's own (round 39): d3first fights it for its key, d2first crosses it |

## The route

Containers are Link's heart containers on entering the step (as the audit counts them, `entry_states.RouteIndex`:
3, and one per heart-container room the route went through before it); W = the wood sword, Ma = the magic sword.
Its own steps in full; the blocks by their rooms.

| Step | State | Room | Entry → exit | Family | Taken, done | Containers / sword |
|---|---|---|---|---|---|---|
| 0 | `start` | 0:0x77 | start → cave | ow-start | | 3 / none |
| 1 | `0_77cave` | 0:0x77 cave | S → S | ow-start | the wood sword | 3 / none |
| 2 | `0_77c` | 0:0x77 | cave → W | ow-start | | 3 / W |
| 3 | `0_76e` | 0:0x76 | E → W | ow-tektite | walked, or scrolled (CLIP E, then W) | 3 / W |
| 4 | `0_75e` | 0:0x75 | E → W | ow-leever | the wrap exit (CLIP E) into 0x74 | 3 / W |
| 5 | `0_74e` | 0:0x74 | E → cave | ow-tektite | Level 3 | 3 / W |
| 6 | `3_7cs` | 3:0x7c | S → W | zol | `bomb_reserve: 6` | 3 / W |
| 7 | `3_7be` | 3:0x7b | E → N | zol | the key | 3 / W |
| 8 | `3_6bs` | 3:0x6b | S → N | zol | kill all (10 Gels), the key | 3 / W |
| 9 | `3_5bs` | 3:0x5b | S → N | darknut | crossed (`treasure: skip`) | 3 / W |
| 10 | `3_4bs` | 3:0x4b | S → W | zol | kill all, the key, the W key door | 3 / W |
| 11 | `3_4ae` | 3:0x4a | E → W | stalfos-keese-gel | kill all (5 Keese), 4 bombs | 3 / W |
| 12 | `3_49e` | 3:0x49 | E → S | zol | the key | 3 / W |
| 13 | `3_59n` | 3:0x59 | N → S | darknut | emptied (five Red Darknuts); `bomb_reserve: 0` (round 32: 0x4a's bombs are the fight's) | 3 / W |
| 14-23 | | `dungeon-3` | | darknut, equipment-room, trap-room, boss-manhandla, zol | the raft, 0x5b's 4 bombs and E wall, 0x4c's E wall, Manhandla's heart, the piece | 3 → 4 / W |
| 24-30 | | `to-level-4` | | ow-tektite, ow-moblin, ow-octorok, ow-zora, ow-peahat | 0x64 scrollable; the raft from 0x55 | 4 / W |
| 31-52 | | `dungeon-4` | | (d2first's) | the ladder, 0x01's key, Gleeok's heart, the piece | 4 → 5 / W |
| 53 | `0_45c` | 0:0x45 | cave → S | ow-peahat | `bomb_reserve: 8` (the Olive Garden) | 5 / W |
| 54 | `0_55n` | 0:0x55 | N → E | ow-zora | off the dock, E | 5 / W |
| 55 | `0_56w` | 0:0x56 | W → N | ow-zora | | 5 / W |
| 56 | `0_46s` | 0:0x46 | S → E | ow-zora | new room (a Zora) | 5 / W |
| 57 | `0_47w` | 0:0x47 | W → E | ow-zora | scrollable | 5 / W |
| 58 | `0_48w` | 0:0x48 | W → N | ow-leever | | 5 / W |
| 59 | `0_38s` | 0:0x38 | S → W | ow-zora | | 5 / W |
| 60 | `0_37e` | 0:0x37 | E → cave | ow-octorok | Level 1 | 5 / W |
| 61-78 | | `dungeon-1` | | (d2first's) | out S and back in (0:0x37 cave → cave), 0x53's key and N wall, the bow, Aquamentus's heart, the piece | 5 → 6 / W |
| 79 | `0_37c_v2` | 0:0x37 | cave → E | ow-octorok | `bomb_reserve: 8` | 6 / W |
| 80 | `0_38w` | 0:0x38 | W → N | ow-zora | | 6 / W |
| 81 | `0_28s` | 0:0x28 | S → E | ow-zora | | 6 / W |
| 82-84 | `0_29w`, `0_2aw`, `0_2bw` | 0:0x29, 0x2a, 0x2b | W → E | ow-leever | new rooms; each scrollable | 6 / W |
| 85 | `0_2cw` | 0:0x2c | W → cave | ow-tektite | the heart rock's wall (one bomb facing N from (`$90`,`$A5`)) | 6 / W |
| 86 | `0_2ccave` | 0:0x2c cave | S → S | ow-tektite | `waypoint: heart` (the right ware) | 6 / W |
| 87 | `0_2cc` | 0:0x2c | cave → E | ow-tektite | | 7 / W |
| 88 | `0_2dw` | 0:0x2d | W → S | ow-zora | crossed (round 38: its rock wall's rupee cave dropped) | 7 / W |
| 89 | `0_3dn` | 0:0x3d | N → cave | ow-moblin | the Armos woken (`how: armos`) | 7 / W |
| 90 | `0_3dcave` | 0:0x3d cave | S → S | ow-moblin | `waypoint: rupees`: 30 | 7 / W |
| 91 | `0_3dc` | 0:0x3d | cave → S | ow-moblin | | 7 / W |
| 92 | `0_4dn` | 0:0x4d | N → W | ow-moblin | | 7 / W |
| 93 | `0_4ce` | 0:0x4c | E → N | ow-octorok | | 7 / W |
| 94 | `0_3cs` | 0:0x3c | S → cave | ow-octorok | Level 2 (the reserve of 8 holds: D5's walls) | 7 / W |
| 95-104 | | `dungeon-2`, 0x5e, 0x4e, 0x3e, `dungeon-2-north` | | (d2first's; 0x5e and 0x3e own) | 0x5e crossed (round 37); 0x3e crossed, the Moldorm left and its key with it (round 39); 0x1e's 4 bombs, Dodongo's heart, the piece | 7 → 8 / W |
| 105 | `0_3cc` | 0:0x3c | cave → S | ow-octorok | | 8 / W |
| 106 | `0_4cn` | 0:0x4c | N → E | ow-octorok | | 8 / W |
| 107 | `0_4dw` | 0:0x4d | W → N | ow-moblin | | 8 / W |
| 108 | `0_3ds` | 0:0x3d | S → N | ow-moblin | | 8 / W |
| 109 | `0_2ds` | 0:0x2d | S → N | ow-zora | | 8 / W |
| 110-112 | `0_1ds`, `0_1ew`, `0_1fw` | 0:0x1d, 0x1e, 0x1f | S → E, W → E, W → N | ow-peahat, ow-tektite, ow-peahat | 0x1f's wall gap at X `$80` | 8 / W |
| 113-115 | `0_0fs`, `0_0fcave`, `0_0fc` | 0:0x0f | S → cave → S | ow-start | `waypoint: rupees`: 100 (open cave) | 8 / W |
| 116-118 | `0_1fn`, `0_1ee`, `0_1de` | 0:0x1f, 0x1e, 0x1d | N → W, E → W, E → N | ow-peahat, ow-tektite, ow-peahat | | 8 / W |
| 119 | `0_0ds` | 0:0x0d | S → W | ow-tektite | | 8 / W |
| 120 | `0_0ce` | 0:0x0c | E → W | ow-tektite | the wrap exit (CLIP E) onto the mountain top | 8 / W |
| 121 | `0_0be` | 0:0x0b | E → cave | ow-leever | Level 5 | 8 / W |
| 122 | `5_76s` | 5:0x76 | S → N | gibdo | `bomb_reserve: 8` | 8 / W |
| 123-144 | | `dungeon-5` | | (d2first's) | the Olive Garden, the recorder, Digdogger's heart, the piece | 8 → 9 / W |
| 145 | `0_0bc` | 0:0x0b | cave → recorder | ow-leever | two plays facing down, as he comes out | 9 / W |
| 146 | `0_45r` | 0:0x45 | recorder → S | ow-peahat | the raft S | 9 / W |
| 147 | `0_55n_v2` | 0:0x55 | N → recorder | ow-zora | **the dock warp** (the engine's): MOVE S, CLIP N into the water at X `$78`, RECORDER: N, a turn W, again | 9 / W |
| 148 | `0_34r` | 0:0x34 | recorder → cave | ow-leever | set down at (`$80`,`$8D`); the Armos | 9 / W |
| 149 | `0_34cave` | 0:0x34 cave | S → S | ow-leever | `waypoint: food` (60) | 9 / W |
| 150 | `0_34c` | 0:0x34 | cave → S | ow-leever | | 9 / W |
| 151 | `0_44n` | 0:0x44 | N → cave | ow-zora | | 9 / W |
| 152 | `0_44cave` | 0:0x44 cave | S → S | ow-zora | `waypoint: arrows` (80), `rupee_reserve: 2` | 9 / W |
| 153 | `0_44c` | 0:0x44 | cave → S | ow-zora | | 9 / W |
| 154-157 | | `pond-42` | | ow-zora, ow-moblin | the pond (the recorder, no index step) | 9 / W |
| 158-177 | | 7:0x79, `dungeon-7`, 7:0x1b, `dungeon-7-east` | | (d2first's, the entrance and 0x1b own) | the red candle, the meat given, Aquamentus's heart, the piece | 9 → 10 / W |
| 178 | `0_42c` | 0:0x42 | cave → S | ow-moblin | `bomb_reserve: 8` (every bomb for the walls) | 10 / W |
| 179 | `0_52n` | 0:0x52 | N → W | ow-moblin | | 10 / W |
| 180 | `0_51e` | 0:0x51 | E → W | ow-moblin | the wrap exit into 0x50 | 10 / W |
| 181 | `0_50e` | 0:0x50 | E → S | ow-lynel | | 10 / W |
| 182 | `0_60n` | 0:0x60 | N → W | ow-lynel | the wrap exit round the world into 0x5f | 10 / W |
| 183 | `0_5fe` | 0:0x5f | E → N | ow-zora | the ladder clip, the ladder heart | 10 / W |
| 184 | `0_4fs` | 0:0x4f | S → N | ow-zora | | 11 / W |
| 185 | `0_3fs` | 0:0x3f | S → N | ow-zora | the raft | 11 / W |
| 186-188 | `0_2fs`, `0_2fcave`, `0_2fc` | 0:0x2f | S → cave → S | ow-start | `waypoint: heart`; the raft back | 11 → 12 / W |
| 189 | `0_3fn` | 0:0x3f | N → recorder | ow-zora | **the dock warp** (the engine's): MOVE S, CLIP N into the water at X `$58`, RECORDER: N twice | 12 / W |
| 190 | `0_31r` | 0:0x31 | recorder → N | ow-ghini | set down at (`$80`,`$AD`) | 12 / W |
| 191-196 | | `graveyard` | | ow-ghini, ow-lynel | the magic sword (12) | 12 / W → Ma |
| 197-212 | | `dungeon-6` | | (d2first's) | Gohma's heart, the piece | 12 → 13 / Ma |
| 213-218 | | `to-level-8` | | (d2first's) | two plays facing up (index on Level 7: Link turns first): 0x3c; 0x6d's tree | 13 / Ma |
| 219-228 | | `dungeon-8` | | (d2first's) | Gleeok's heart, the piece | 13 → 14 / Ma |
| 229-243 | | `to-level-9` | | (d2first's) | one play facing down: 0x37; 0x47's heart (15); 0x05's wall | 14 → 15 / Ma |
| 244-289 | | `dungeon-9`, 0x56, `dungeon-9-from-55` | | (d2first's; 0x56 own) | 0x56's key (round 39: its 2 Zols and 2 Like-Likes killed, the ladder to the key), the red ring, the silver arrow, Ganon, Zelda (`exit: end`) | 15 / Ma |

Dungeon entrances (`crates/zelda/tests/d3first.rs`, `tests/test_game_map.py`): Level 3 with 3 containers and no
piece; Level 4 with 4, the raft, piece 3; Level 1 with 5, the raft and the ladder, pieces 3 and 4; Level 2 with 7,
the bow too, pieces 1, 3, 4; Level 5 with 8, pieces 1-4; Level 7 with 9, the recorder, the food and the arrows,
pieces 1-5; Level 6 with 12 and the magic sword, pieces 1-5 and 7; Level 8 with 13, pieces 1-7; Level 9 with 15,
all eight.

**The recorder chain** (round 36, `scripts/experiments/r36_recorder_search.py`: every sequence of one to three
plays, each facing, the one-frame turn before a later play, played in the emulator from the route's states with the
chain's index carried; the game's index `$523` is stepped by every overworld play, so the plays are one chain):

| Where | Index in (level) | The plays | Second play's window (staging) | Lands |
|---|---|---|---|---|
| 0:0x0b out of Level 5 | 0 (L1), pieces 1-5 | 2 facing down, as Link comes out | 63 frames | 0x45 (`$80`,`$8D`), index on L4 |
| 0:0x55, in the water | L4 | N, a one-frame turn W, again | 62 | 0x34 (dock warp) |
| 0:0x3f, in the water | L4, pieces 1-5, 7 | 2 facing N | 51 (X `$58`) | 0x31 (dock warp), index on L7 |
| 0:0x22 out of Level 6 | L7, pieces 1-7 | Link turns up (N or E), then 2 facing up | 64 | 0x3c, index on L2 |
| 0:0x6d out of Level 8 | L2, all | 1 facing down | - | 0x37 |

The search's tie-breaks: fewest turns between plays, fewest plays, fewest turns before the first, the widest window.
Out of Level 5 three plays up also land in 0x45 (57 frames, a turn first); every landing in 0x45 leaves the index on
Level 4, so no way to Level 4's screen changes 0x55's or 0x3f's plays. In the water only N and W turn Link in place
(S and E move him). The owner's memory ("twice north" at 0x55, "once" at 0x3f): by the arriving index, two N plays at
0x55 land in 0x34 only from Level 2's index and one N play at 0x3f lands in 0x31 only from Level 5's or Level 6's;
no history of this order gives either (the 0x34 landing needs Level 4 at the pickup). Misses: at 0x55 a late
second play lands in 0x77, N N in 0x26, W W in 0x2b; at 0x3f a late second play lands in 0x77, N W in 0x34.

**The dock warps** are played by the engine (`zelda::Session::set_dock_warp`, the step's level from
`GameCompletion::dock_warp`): on the dock below its trigger after the raft's ride down the mask is MOVE S only (any
other step boards the raft back up); where the block clip N into the water is legal (`zelda::clips::dock_clip_at`:
X on the grid W of the dock, T = the dock's row + 8 and the row stopped by water, G and G' land) CLIP N only, and a
MOVE along the land stops on the clip's column (`dock_spot`, the nearest the dock: 0x55 `$78`, 0x3f `$58`); in the
water on the row RECORDER only, which plays every play the index needs (`Executor::run_dock_warp`: N for a step up,
W for a step down, the turn one frame before the B on the first free frame) and waits for the pickup. The landing
is `Whirlwind_PrevRoomIds[level] - $10` (`zelda::game_map::dock_landing`; route entry `recorder`). A whirlwind out of
a dock room into any other room ends a chain run `stuck` (`dock-warp-missed`).

**Clips.** The screen scrolls the walks found cheaper (0x76, 0x64, 0x47, 0x29-0x2b: the plain wrap W/E) are CLIP
actions the families choose (`clips: true`), nothing in the steps. The wrap exits of 0x75, 0x0c, 0x51 and 0x60, and
both dock clips, are the only ways through those rooms on this route (no `stretch:`, clips only, the owner's Nov 18
rule): a run with the run-time switch off has no way past them. The shared blocks keep their stretches
(`d4-island`, `ring-07`, `arrow-20`, `arrow-20n`).

## New vocabulary and game data

- routes.yaml `blocks:` and `{include: NAME}` (above).
- `waypoint: blue-candle`: a purchase as `arrows`/`food` (`zelda::objectives` `WAYPOINT_BLUE_CANDLE`, done once Link
  holds a candle; game.yaml `PURCHASES`): 0:0x0c's shop cave (`treasure: blue-candle`, `treasure_tile: [19, 11]`,
  `requires_rupees: 60`, the other wares' squares kept off by `path_costs` as 0x44's). Dungeon 7's red candle stays
  `waypoint: candle` (a cellar's item, taken whatever candle Link holds).
- game.yaml room entries with `routes: [d3first]`: the room as d3first plays it (`zelda::game_map` `parse_rooms`,
  `GameMap::route_map`): over a shared entry its keys replace the shared ones and its `exits:` are added; without one
  it is a room only d3first has. The envs, the chain and the state builder play the map of the episode's route
  (`Session::set_map`; `_native.Game(route=)`, `game_rooms(route)`, `specialist_for(room, route)`). d3first's own:
  0x0c's, 0x2d's, 0x62's and 0x3d's caves (the `cave` exits and the in-cave rooms: 30, 100 and 30 rupees), 0x2d's
  rock wall (`requires_bombs`, `bomb_secret: [10, 2]`, the 30 secret; round 38 dropped 0x2d's wall and cave), the tags 0x29, 0x2a, 0x2b ow-leever, 0x46
  ow-zora (its one monster a Zora, as 0x47's), 0x62 ow-moblin, 3:0x4a stalfos-keese-gel. Walk A proposed ow-octorok
  for 0x46; its monsters are one Zora. d2first plays the shared rooms, the data its models were trained on: in a
  first form (one map for both) the new tags opened paths for d2first's off-route objective (ow-lynel's wrong exit
  from 0x60 into 0x50 then pointed S), 0x2d's wall offered BOMB and a granted bomb on d2first's crossings, and the
  rule written against that (no bomb on a step crossing a rock-wall room) took d2first's granted bomb at 0x67 S -> E
  (the schema 5 traces of ow-lynel, ow-zora, ow-tektite and ow-octorok, `tests/test_obs_legacy.py`); with the route maps
  they are what they were (`game_map::tests::a_routes_own_rooms_leave_the_shared_map_alone`, `crates/zelda/tests/
  d3first.rs` from d2first's `0_2ds_0` and `0_3ds_0`).
- Families gain rooms (routes.yaml `families`, `specialists/*.yaml` `rooms:`): ow-zora 0x46 and 0x2d's cave (gone in round 38);
  ow-leever 0x29, 0x2a, 0x2b; ow-moblin 0x62, its cave and 0x3d's cave; ow-tektite 0x0c's cave; stalfos-keese-gel
  3:0x4a. Their `start_states` globs (`0_46*`, `0_29*`, `0_2a*`, `0_2b*`, `0_62*`, `3_4a*`) come with the first
  states (a glob matching no state is refused: `tests/test_entry_states.py`).
- Segments and scopes: `eval_batch.ROUTE_SEGMENTS['d3first']` (`D3FIRST_SEGMENTS`, `_TARGETS`, `_MILESTONES`),
  `room_stats.ROUTE_SCOPES['d3first']`; `ROUTE=d3first TAG=... scripts/experiments/r56_chains.sh`.

## Bombs: the owner's three rules (round 37, 2.0.0.6)

The owner watched a golden replay (Nov 22). His words:

1. "In dungeons, link should only use bombs in rooms it somehow needs to clear. For example, it's fine to use bombs
   in 0x59 because we have to clear the room to open the south door, in 0x69 it needs to slide past the darknuts to
   get the raft, then out of the room (I use sword-hit an enemy to make them pass-through in this room, not kills
   with bombs)."
2. "Some rooms I want to explicitly say 'don't use bombs in this room' [...] 0x5b is a great example, the point of
   fighting the darknuts in that room is to get the bomb drop, wasting the bombs to kill them is not the right play.
   BUT link still needs to bomb the wall to get through there. So no bombs except the 1 to blow up the wall."
3. "I want to raise the 'get bombs from this room' floor, so in 0x5b he should fight the darknuts if he has 7 or
   less bombs, up from whatever it was. This is a blanket rule not for this room."

And over all three: "we must ALWAYS allow bombs and deeply encourage them if we are on the 10th kill force drop, no
matter the room or enemies, that remains true."

They are routes.yaml's top-level `bomb_rules:` under `d3first` (`zelda::game_map::BombRules`; `d2first` names none
and is unchanged: the owner puts it in scope "when we go back and work on that single-model", and the same entry
under its name switches them on there):

- `fights_only: true` (rule 1). In a dungeon room BOMB is a weapon only while the objective fights the room
  (`zelda::objectives::GameCompletion::weapon_bombs_off`, from the objective's own kind every step: a shutter the
  monsters hold on the step's exit, a treasure or a block that waits on the kills and the step takes, a boss). Where
  the route only crosses, or once the fight is over, a `free` family's mode is capped at `required`: BOMB only at
  the intact wall the step leaves through, and the 10th damage-free kill's bomb. `tenth`, `required` and `never`
  place no weapon bomb already; in a fight the family's mode and the reserve stand as before. The overworld is
  untouched. 3:0x59 (from N: five Red Darknuts, the S shutter) is a fight; 3:0x69 both ways is a crossing.
- `wall_only: [3:0x5b]` (rule 2). "Rooms where the owner forbids bombs but for the wall": in these BOMB is never a
  weapon, a fight or not (the cap at `required` whatever the family's mode or the step's). 3:0x5b's first pass (S
  to N, its bombs `skip`, no wall on the step) places no bomb at all; its last pass (W to E) fights its Darknuts by
  the sword for the room's four bombs and then bombs its E wall. The step's `bomb_reserve: 0` (the shared block's,
  for `d2first`) bites no more there. The list: **3:0x5b**; more will come.
- `treasure_below: 8` (rule 3). Every step that takes a room's game.yaml bombs (no `treasure:`, or a step's own
  `{if_bombs_below: M}`, which gives way) fights for them when Link comes in with 7 or fewer and crosses with 8 (the
  bag); judged on entering (`GameCompletion::step_treasure`), so the wall bomb Link spends in 3:0x5b with a full bag
  turns no crossing into a fight beside the opened wall (round 23's trouble), and the decision holds for the visit:
  the treasure is taken while Link holds fewer than the bag's maximum (the owner on 8:0x3f: in with 2, a drop to 6,
  the room's bombs left on the floor by the shared step's live `{if_bombs_below: 5}`); a treasure lying in the room
  is wanted below the maximum whatever he came in with. Under the floor, dropped bombs below the maximum are targets
  in every family and, once the fight is over, the only targets while Link can reach them (`Objective::bag_drops`;
  none at 8). On `d3first`: 3:0x4a, 3:0x5b's last pass, 2:0x1e, 5:0x65 eastbound (its four lie in the open: a
  pickup, no fight), 7:0x1b (round 34's floor of 2 gone), 8:0x3f (the shared step's 5 gives way), 9:0x16
  northbound. A room's own floor is `treasure_below_rooms: [{level, room, below: N}]`: 7:0x1b's crossing is
  `{level: 7, room: '0x1b', below: 2}` (the owner: "you have the ability to skip that fight for bombs if it kills
  too many runs"); measured both ways in round37.md (`d7` 12.6% with the floor, 12.7% crossing: the runs bring 0.21
  bombs, so nearly every visit fights either way) and left commented under `bomb_rules` for the lead.
- The 10th kill: every mode but `never` offers BOMB when the next kill is the 10th damage-free one (`$50` = 9; its
  bomb kill arms a forced drop of four), in a crossing, in 3:0x5b and on the overworld, below any reserve, as before
  (`never` is boss-gleeok's: only the sword hurts Gleeok; the `never` steps 0:0x07 and the state builders'). A
  granted (cheated) bomb still opens only its wall, the 10th kill too (round 26). The bomb kill is paid
  `reward-armed-bomb-drop` (0.75) and the drop's pickup `reward-gained-bombs` (0.5), in every family; not using
  the bomb costs nothing.

The dungeon steps of `d3first` by what the rules make of them (step numbers of 2.0.0.7; the engine's objective as
the scout read it every 30 frames of the golden replay, round37.md; cellars left out). **Clear**: BOMB by the
family's mode above the reserve while the fight lasts. **Clear < 8**: a fight only when Link comes in with 7 or
fewer. **Wall**: a crossing whose exit is a bomb wall (one bomb). **Owner's**: no weapon bomb, the wall's and the
10th kill's only. **Cross**: no bomb but the 10th kill's.

| Level | Clear | Clear < 8 | Clear + wall | Wall | Owner's | Cross |
|---|---|---|---|---|---|---|
| 3 | 0x6b S→N (8); 0x4b S→W (10); 0x59 N→S (13); 0x4d W→N (22) | 0x4a E→W (11) | — | 0x4c S→E (21) | 0x5b S→N (9); 0x5b W→E (19, < 8, wall) | 0x7c S→W (6); 0x7b E→N (7); 0x49 E→S (12); 0x69 N→cave (14); 0x69 cave→N (16); 0x59 S→E (17); 0x5a W→E (18); 0x5c W→N (20); 0x3d S→triforce (23) |
| 4 | 0x31 W→E (37); 0x32 W→cave (38); 0x12 N→E (50); 0x13 W→N (51) | — | — | 0x21 S→N (42); 0x21 W→N (46); 0x11 S→N (47) | — | 0x71 S→N (31); 0x61 S→N (32); 0x51 S→W (33); 0x50 E→N (34); 0x40 S→N (35); 0x30 S→E (36); 0x32 cave→W (40); 0x31 E→N (41); 0x31 E→W (43); 0x30 E→N (44); 0x20 S→E (45); 0x01 S→E (48); 0x02 W→S (49); 0x03 S→triforce (52) |
| 1 | 0x23 S→W (68); 0x44 W→E (75); 0x35 S→E (77) | — | 0x53 S→N (65) | — | — | 0x73 S→S (61); 0x73 S→N (63); 0x63 S→N (64); 0x43 S→N (66); 0x33 S→N (67); 0x22 E→cave (69); 0x22 cave→E (71); 0x23 W→S (72); 0x33 N→S (73); 0x43 N→E (74); 0x45 W→N (76); 0x36 W→triforce (78) |
| 2 | 0x7e W→N (96); 0x4e S→N (99); 0x2e S→N (101); 0x0e S→W (103) | 0x1e S→N (102) | — | — | — | 0x7d S→E (95); 0x6e S→N (97); 0x5e S→N (98: crossed since round 37, the owner); 0x3e S→N (100: crossed since round 39, the owner); 0x0d E→triforce (104) |
| 5 | 0x05 E→cave (128); 0x66 W→N (135); 0x24 E→N (143) | — | — | 0x66 S→W (123); 0x65 E→W (124) | — | 0x76 S→N (122); 0x64 E→cave (125); 0x06 cave→W (127); 0x05 cave→E (130); 0x06 W→cave (131); 0x64 cave→E (133); 0x65 W→E (134, its bombs a pickup < 8); 0x56 S→E (136); 0x57 W→N (137); 0x47 S→N (138); 0x37 S→N (139); 0x27 S→W (140); 0x26 E→W (141); 0x25 E→W (142); 0x14 S→triforce (144) |
| 7 | 0x49 S→N (161); 0x1c W→N (171); 0x0d W→cave (173); 0x2a W→E (176) | 0x1b W→E (170) | — | 0x69 S→N (159); 0x19 W→E (166); 0x1a cave→E (169); 0x0c S→E (172); 0x29 cave→E (175) | — | 0x79 S→N (158); 0x59 S→N (160); 0x39 S→W (162); 0x38 E→N (163); 0x28 S→N (164); 0x18 S→E (165); 0x1a W→cave (167); 0x2b W→triforce (177) |
| 6 | 0x58 S→N (200); 0x38 S→N (202); 0x39 N→E (205); 0x3a W→cave (206); 0x2d N→W (209); 0x1c S→N (211) | — | — | 0x28 S→E (203) | — | 0x79 S→W (197); 0x78 E→N (198); 0x68 S→N (199); 0x48 S→N (201); 0x29 W→S (204); 0x1d cave→S (208); 0x2c E→N (210); 0x0c S→triforce (212) |
| 8 | 0x5e S→N (221); 0x3e S→E (223); 0x3c S→N (227) | 0x3f W→cave (224) | — | 0x6e S→N (220); 0x4c cave→N (226) | — | 0x7e S→N (219); 0x4e S→N (222); 0x2c S→triforce (228) |
| 9 | 0x14 cave→E (249); 0x05 E→cave (264); 0x61 E→cave (268); 0x10 S→cave (272); 0x30 E→cave (282); 0x52 cave→N (287); 0x42 S→N (288) | 0x16 S→N (262) | 0x56 S→W (246: its key, round 39); 0x20 cave→N (271, plain way) | 0x26 N→E (252); 0x27 W→N (253); 0x17 S→N (254); 0x06 S→W (263); 0x20 cave→N (270); 0x31 S→W (281); 0x04 cave→W (284) | — | 0x76 S→N (244); 0x66 S→N (245); 0x55 E→cave (247); 0x15 W→E (250); 0x16 W→S (251); 0x07 S→cave (255); 0x07 cave→S (257, 258); 0x17 N→S (259); 0x27 N→W (260); 0x26 E→N (261); 0x63 cave→W (266); 0x62 E→W (267); 0x10 cave→S (274); 0x20 N→cave (275, 276); 0x61 cave→N (278); 0x51 S→N (279); 0x41 S→N (280); 0x03 E→cave (285); 0x32 S→end (289) |

A clear room becomes a crossing once its fight is over (3:0x59's Darknuts dead, a treasure taken, a shutter open):
the mask follows the objective every step. 8:0x6e's step `bombs: required` (the shared block's, for `d2first`) is
what `fights_only` gives it on `d3first` anyway; 9:0x56's `required` is like-like's only way to bomb its wall (a
fight for its key since round 39: like-like's `tenth` mode places no weapon bomb, the wall's and the 10th kill's only).

## Reserves

- **Bombs** (routes.yaml; a step's `bomb_reserve` holds until the next one names another). 6 from Dungeon 3's
  entrance to 0x49 as d2first's; **0 from 0x59's fight (step 13, d3first's own; the lead, Nov 20) through 0x5b's
  last pass**: 0x4a's four bombs are 0x59's (five Red Darknuts, all to die: the BombFighter rule wins 60 of 60 with
  them, the sword alone 11 of 20), and 0x5b's four, taken on its last pass, pay for the two walls and Manhandla; 6 at
  0x5c and 0x4c, 0 at Manhandla, 6 after him (the block's). 2 from Dungeon 4's entrance (the block's). **8 from Level
  4's exit (step 53) and again from Level 1's (79) through Dungeon 2 to Dungeon 5** (round 36: the first order's 1
  through Dungeon 2 is gone): every bomb Link has is kept for Dungeon 5's walls and the Olive Garden; the walls on
  the way (1:0x53, 0x2c; 0x2d's until round 38) are bombed as any wall, cheated where Link has none, and Dodongo's room offers BOMB
  whatever the reserve. **Dungeon 5's entrance grants nothing** (2.0.0.3; the owner, Nov 21: "don't blindly add 8
  bombs: the goal is a win without a cheated bomb"): Link opens 5:0x66's and 5:0x65's walls with the bombs he brings
  and fights the Garden with what is left (0 in 0x05: thrown freely, 8 after it, 0 at Digdogger). In the corrected
  order Dungeon 2 comes first and fills the bag: runs from Dungeon 2's entrance with 2 bombs reach Level 5 with 8 in
  231 of 296 arrivals (round 36's harvest of the `5_76s` states, round36.md "States"). **8 from Level 7's entrance to 0x1b**
  (step 158, round 34, the lead): the reserve keeps the bag for the walls (0x69, 0x19, 0x1a); **0x1b wants its bombs
  below 8** (round 37, the owner's floor; round 34's `{if_bombs_below: 2}` is gone, step 170); 2 from 0x1b as d2first's, 0 at Aquamentus. **8 from
  Level 7's exit (0x42, step 178) to Dungeon 6** (round 36): every bomb for 6:0x28's wall; then the blocks' (1 from
  6:0x29, 1 out of Level 6, 2 into Dungeon 8, 5 out of it, Dungeon 9's).
- **Rupees.** The route's own: 0x3d's 30, 0x0f's 100 (Dungeon 2 and its drops between; 0x2d's 30 dropped in round
  38, the owner: 10 + 30 + 100 against 140, the shortfall cheated and counted) against the meat (60, 0x34's cave)
  and the arrows (80, 0x44's cave, `rupee_reserve: 2` from there); 4:0x11's 10. The blocks'
  reserves after: 1 in Gohma's room, 8 from 8:0x4c, 1 through Level 9, 0 at Ganon. A reserve keeps rupees from
  ARROW; it cannot keep what was never there, so the arrow rooms' needs top the purse up (cheated, counted): Gohma's
  (round 27), Ganon's (round 29), 8:0x4c's while a Pols Voice lives (round 32).

## Ledgers (round 36: today's networks from the game start, `r36base-1k-full`, 1000 runs)

What the runs that get there hold on entering (means over the runs; the walks' staged ledgers of the first order are
in `d3first-walks-a.md` §7 and `-b.md`):

| Entering | Runs | Containers | Hearts | Bombs (none) | Keys |
|---|---:|---:|---:|---|---:|
| Level 3 | 1000 | 3 | 2.94 | 0.00 (1000) | 0 |
| Level 4 | 235 | 4 | 3.77 | 4.28 (22) | 2 |
| Level 1 | 45 | 5 | 4.58 | 0.56 (34) | 3 |
| Level 2 | 36 | 7 | 3.49 | 2.11 (14) | 3 |
| Level 5 | 30 | 8 | 6.97 | 7.07 (0) | 6 |
| Level 7 | 5 | 9 | 7.30 | 0.80 (3) | 5 |
| Level 6 | 1 | 12 | 11.50 | 0 | 2 |

Rupees in all (2.0.0.7): 30 (0x3d) + 100 (0x0f) + 10 (4:0x11) and the dungeons' drops, against 60 + 80 for the
wares and 3-5 for arrows: the corrected order buys both wares with the 100 already in hand (Dungeon 2 before them);
round 38 dropped 0x2d's 30, the purse's cushion (round38.md "0x2d dropped" for the rupees cheated at the shops).

## For the owner

From the corrected order (round 36, `d3first-walks-c.md`, round36.md "The corrected route"):

1. **The counts of the wrong warps.** "Twice north" at 0x55 and "once" at 0x3f land where the route needs only
   from indices this order never gives (above, "The recorder chain"): the route plays N, a one-frame turn W, and N
   again at 0x55, and N twice at 0x3f, both found by search in the emulator.
2. **The warps are an engine macro** after the block clip, as the recorder warps are: on the dock the mask leaves
   MOVE S, then CLIP N on the column, then RECORDER, which plays every play. Today's ow-zora crosses both from the
   route's states; with the monsters live the scripted rule lands 6 of 20 at 0x55 and 13 of 20 at 0x3f, every miss
   a Zora's hit (walks-c §2).
3. **Bombs.** Dungeon 2 before Dungeon 5 fills the bag: runs reach Level 5 with 8 bombs in most runs (7.07 on
   average from the game start), where the first order brought 2.03.
4. **0x3d's 30 rupees are taken** (the lead's choice: the route crosses the screen anyway), and 0x2d to Level 2 goes
   by 0x3d, 0x4d and 0x4c (0x3d has no W edge). Round 38 (the owner): 0x2d's bomb-wall cave is dropped.

From the first walks (`d3first-walks-a.md`, `-b.md`), still standing:

5. **Level 4 → Level 1.** 0x57 has no N edge and 0x47 none to 0x37; built as walked: 0x55 E, 0x56 N, 0x46 E, 0x47 E,
   0x48 N, 0x38 W, 0x37 (0x46 new; about 0.8 hearts by the rule).
6. **Dungeon 3 at 3 hearts with the wood sword** is the hardest stretch of the first half for a scripted rule (0x59's
   five Darknuts, Manhandla by bombs only); nothing was changed in the route for it.
7. **Dungeon 1's key rooms.** With phase IX's door trick 0x23's Goriya key and 0x53's key are not needed on this
   route (one key left at Level 1's exit). Built as the shared block.
8. **0x76**: both walked; the scroll costs nothing and saves 246 frames (a CLIP the family may choose).
9. **The warps out of Levels 5, 6 and 8** need the facings of the recorder chain; the engine computes them and masks
   RECORDER until Link faces a way that works; out of Level 6 Link turns up first.
10. **The wood sword in the Olive Garden** (9-10 of 20 by the scripted fighter without bombs, 17-18 with eight) and
    **in Dungeon 7** (0x1c's small Digdoggers kill the fighter 7 of 20 from full hearts).
11. **Rupees**: the arrow rooms (Gohma, 8:0x4c, Ganon) are topped up while their monsters live, cheated and counted.
12. **0x3d's secret lives in a room d2first crosses**; an engine rule keeps d2first's crossing what it was (no MOVE
    into the Armos in 0x3d). 0x2d's wall (round 38) is in neither route.
13. **Clips only** at 0x75, 0x0c, 0x51, 0x60, 0x3e and both docks: `CLIPS=off` chains cannot pass them.

## For the agents that build its states

Numbers of 2.0.0.7 (round 38: 290 steps). Steps 0-88 and 215-289 are the first order's states (round 32, below; the
to-level-8 visits of 0x4c from N and 0x4d from W renamed `_v2`, the corrected order visiting both first on the way to
Level 2; 0x2d's step 88 now leaves S). Steps 89-153 were rebuilt in round 38 by the explorer from
`d3first_0_2dw_<chain>`, steps 154-214 are round 36's (rebuilt then from `d3first_0_2dc_<chain>`), both with the
fights the explorer cannot clear skipped (`entry_states build --skip-fights`; their states are marked
`fight_skipped`), and Dungeon 5's entrance, Dungeon 5's exit and Dungeon 7's entrance (`5_76s`, `0_0bc`, `7_79s`)
were then cut from the promoted models' chain arrivals by their bombs (`scripts/experiments/r32c_models.py
--bombs-quantiles 0.5,0.1,0.9`: chain 0 the median, 1 the low tenth, 2 the high tenth), so the segments start with
what runs bring. The states after 0x44's cave not rebuilt in round 38 (steps 154-157 and 159-289) hold about 30
rupees more than runs bring there (0x2d's 30: 33 against the 7 of `7_79s_0`); rupees matter there only in the arrow
rooms, which top the purse up anyway. The fights the walks crossed only by a rule (a fight to the end with the wood
sword, a bomb fight; the first order's and round 36's step numbers):

| Stretch | Room (step) | The rule that crossed it | Its state after |
|---|---|---|---|
| Dungeon 3 | 3:0x6b (8): 5 Zols → 10 Gels for the key | the rule, 19/20 | `3_5bs` |
| Dungeon 3 | 3:0x4b (10): 3 Zols for the key, the W key door | 19/20 | `3_4ae` |
| Dungeon 3 | 3:0x4a (11): 5 Keese for the bombs | 19/20 | `3_49e` |
| Dungeon 3 | 3:0x59 (13): 5 Red Darknuts for the S shutter | `BombFighter` with 0x4a's bombs, 17/20 (sword 11/20) | `3_69n` |
| Dungeon 3 | 3:0x5b (19): 3 Red Darknuts for the bombs, the E wall | the sword, 20/20 | `3_5cw` |
| Dungeon 3 | 3:0x4d (22): Manhandla | `ManhandlaFighter` (bombs), 11/20; the sword 0/20 | `3_3ds` |
| Dungeon 4 | 4:0x13 (51): Gleeok, 16 wood hits | `r28_gleeok.GleeokFighter`, 6/20 at 4 containers | `4_03s` |
| Dungeon 4 | 4:0x32 (38): 2 Like-Likes (9 wood hits each), 2 Zols, the push | 20/20 | `4_60cave` |
| Dungeon 1 | 1:0x35 (77): Aquamentus, 6 wood hits | 19/20 | `1_36w` |
| Dungeon 5 | 5:0x05 (130): the Olive Garden, six Blue Darknuts | skipped (round 36) | `5_04cave` |
| Dungeon 5 | 5:0x24 (145): Digdogger, the small one 8 wood hits | skipped (round 36) | `5_14s` |
| Dungeon 7 | 7:0x49 (163): 3 Blue Goriyas for the N shutter | skipped (round 36) | `7_39s` |
| Dungeon 7 | 7:0x1b (172): 6 Blue Goriyas | skipped (round 36) | `7_1cw` |
| Dungeon 7 | 7:0x1c (173): the small Digdogger by the recorder | skipped (round 36) | `7_0cs` |
| Dungeon 7 | 7:0x2a (178): Aquamentus | skipped (round 36) | `7_2bw` |

Also skipped in round 36: 2:0x1e (104), Dodongo 2:0x0e (105), 5:0x66 from W (137), 6:0x38 (204), 6:0x3a (208) and
Gohma 6:0x1c (213). The explorer needed 8 attempts at 5:0x65 from E (chains 0, 2) and 2:0x3e (chain 1).

## The states as built

**Round 38** (2.0.0.7, 0x2d's cave dropped): steps 89-153 (`0_3dn` to `0_44c`) rebuilt by `python -m
triforce.entry_states build --route d3first --start-step 88 --source 'd3first_0_2dw_{chain}' --last-step 153
--chains 0,1,2 --skip-fights 2:0x1e:S,2:0x0e:S,5:0x05:E,5:0x66:W,5:0x24:E` (chain 1 resumed at 5:0x66 S→W with
`--attempts 8`); then `r32c_models.py 2_7ds --goal-step 122` (`5_76s`: 295 of 300 arrived, 225 with 8 bombs),
`5_76s --goal-step 145` (`0_0bc`: 83 arrived) and `5_76s --goal-step 158` (`7_79s`: 65 arrived), each `--bombs-quantiles
0.5,0.1,0.9 --chains 0 --runs 300 --seed0 10000 --parallel 10`; `0_2dcave` and `0_2dc` removed; 855 states, the
audit clean.

**Round 36** (the corrected order): steps 90-216 (`0_3dn` to `0_3cr`, 126 a chain) rebuilt by `python -m
triforce.entry_states build --route d3first --start-step 90 --source 'd3first_0_2dc_{chain}' --last-step 216
--skip-fights 2:0x1e:S,2:0x0e:S,5:0x05:E,5:0x66:W,5:0x24:E,7:0x49:S,7:0x1b:W,7:0x1c:W,7:0x2a:W,6:0x38:S,6:0x3a:W,6:0x1c:S`
(resumed with `--attempts 8` at 5:0x65 for chains 0 and 2 and at 2:0x3e for chain 1); `5_76s` and `7_79s` cut from
chain arrivals by `r32c_models.py` (above); the first order's 84 states off the corrected route removed (0x0c's
shop and wrap, 0x62's tree, 0x1c, 0x2c E, 0x74's warp landing, the leg to Level 7 by 0x73, 0x63, 0x52, 0x51, 0x50,
0x40, 0x41, 0x31 from S, 0x34 and 0x44 from S, 0x54, 0x64, 0x65, 0x3d and 0x3e from E). What follows is round 32's
build of the first order, whose states 0-89 and 217-291 are kept.

`d3first_*` (`triforce/states/`, chains 0-2, steps 0-303 of the first order, the game start to Ganon's room) were
built stretch by stretch by `scripts/experiments/r32c_states.py --first N --last M --rules ... --tag T` (one job a
stretch): each step of the clip way crossed per chain by the explorer (`entry_states build`, one step), a step named
in `--rules` by the walks' rule (`r32c_harvest.py`: the first won seed of a chain gives the next state, every seed
played for the rate) or by round 28's/29's rule (`STEM=ROOM`, `r28_demos.py`/`r29_demos.py ROOM --route d3first
--harvest`); where the explorer fails, the walks' actions or fight for the step (`r32_d3first.LEGS`,
`r32c_harvest.FIGHTS`/`tallies()`), round 28's/29's rule of the room, and with `--models` the promoted models
(`r32c_models.py`: `eval-chain --route d3first --from STEM_<chain> --stop-at step:<next> --replays`, the won replays'
pre-entry frame of the next room cut out). Every harvested state is written by `write_entry_state` (hearts
refilled, the room fresh); its manifest entry's `played_by` names the rule. No fight is skipped. Wins over runs,
per chain (0, 1, 2; the first order's step numbers):

| Stretch (steps) | Room (step) | How | Won |
|---|---|---|---|
| 1 (0-31) | 3:0x6b (8) -> `3_5bs` | the walks' rule | 56/60 |
| | 3:0x4b (10) -> `3_4ae` | the walks' rule | 60/60 |
| | 3:0x4a (11) -> `3_49e` | the walks' rule | 60/60 |
| | 3:0x59 (13) -> `3_69n` | `BombFighter` (BEST 'bombs') | 60/60 |
| | 3:0x5b (19) -> `3_5cw` | the sword | 56/60 |
| | 3:0x4d Manhandla (22) -> `3_3ds` | `ManhandlaFighter` by bombs | 37/60 |
| 2 (31-97) | 4:0x32 (38) -> `4_60cave` | the walks' rule | 59/60 |
| | 4:0x13 Gleeok (51) -> `4_03s` | `GleeokFighter` at 4 hearts, the heart and N door by `straight` moves | 10/90 (4, 3, 3) |
| | 1:0x35 Aquamentus (81) -> `1_36w` | the walks' rule | 90/90 |
| | 4:0x31 W->E (chains 0, 2), 4:0x12 N->E | the walks' actions (explorer failed) | 10/10 each |
| 3 (97-155) | 5:0x05 Olive Garden (107) -> `5_04cave` | `r32b_d5` BombFighter, then the explorer out | 9, 10, 12 /20 |
| | 5:0x66 W->N (110) -> `5_56s` | `r32b_d7.Fight` on the Gibdos, then the explorer | 17, 18, 17 /20 |
| | 5:0x24 Digdogger (122) -> `5_14s` | the recorder + `DigFighter` | 19, 20, 19 /20 |
| 4 (152-211) | 7:0x49 (181) -> `7_39s` | `r32b_d7` fight + explorer | 9, 12, 10 /20 |
| | 7:0x1b (190) -> `7_1cw` | `r32b_d7` fight + explorer | 20, 20, 20 /20 |
| | 7:0x1c (191) -> `7_0cs` | `r32b_d7` fight (the small Digdogger) | 15, 15, 12 /20 |
| | 7:0x2a Aquamentus (196) -> `7_2bw` | `r32b_d7` fight | 20, 19, 20 /20 |
| 5-7 (211-303) | 8:0x3c Gleeok (241) -> `8_2cs` | `r28_demos.py gleeok --route d3first` | 20, 19, 20 /20 |
| | 9:0x14 (263) -> `9_15w` | `r29_demos.py like-like-14` | 20, 20, 20 /20 |
| | 9:0x16 (276) -> `9_06s` | `r29_demos.py patra-16` | 20, 20, 20 /20 |
| | 9:0x05 (278) -> `9_70cave` | `r29_demos.py wizzrobe-05` (losses: exit-wrong) | 10, 9, 10 /20 |
| | 9:0x61 (282) -> `9_75cave` | `r29_demos.py patra-61` | 20, 20, 20 /20 |
| | 9:0x10 (286) -> `9_4fcave` | `r29_demos.py wizzrobe-10` (losses: deaths; 6.4-7.6 hearts lost in wins) | 17, 14, 19 /20 |
| | 9:0x30 (296) -> `9_67cave` | `r29_demos.py wizzrobe-30` (losses: exit-wrong) | 13, 12, 13 /20 |
| | 9:0x52 (301) -> `9_42s` | `r29_demos.py patra-52` | 20, 20, 20 /20 |
| | 9:0x42 Ganon (302) -> `9_32s` | `r29_demos.py ganon` | 20, 20, 20 /20 |

Rooms the explorer did not solve at its default settings (3 attempts of 3000 iterations, patience 200): 5:0x65 E->W
chain 2 (passed with 8 attempts), 2:0x1e S->N chain 0, and Dungeon 6's two fights: 6:0x38 S->N chains 0 and 2 and
6:0x3a W->stairs chain 1 (chain 1 crossed 0x38 at the defaults, 151k env steps; chains 0 and 2 met 0x3a only at the
new settings). At 8000 iterations, patience 400, 6 attempts the explorer crossed all of them: 0x38 in 380k and 674k
env steps (1421 s, 2415 s), 0x3a in 552k, 546k and 229k (chains 0-2). Everything else, Gohma and the triforce of
Dungeon 6, the recorder warps, 0x6d's tree by the blue candle, Dungeon 8's Darknut rooms (8:0x3e 927 s), 0x47's
heart, the ladder at 0x17 and the rock wall at 0x05, Dungeon 9's crossings (0x66 N, 0x56, the bomb, 0x55 from E and
its clip, the ring), the explorer crossed. The promoted models (copied into the worktree, stamped 2.0.0.0; wizzrobe
`r30w-wj-s89`) won 6:0x38 28/30 and 27/30 (chains 0, 2) and 6:0x3a 27/30 (chain 1) with `r32c_models.py`, run beside
the explorer as a fallback; the explorer's states were kept, so no shipped state is model-won. The stretch forks'
plain ways (4:0x30-0x21, 0:0x65-0x54, 0:0x61-0x50) were walked by `entry_states path --route d3first`, as d2first's.

## Phase IX

`d3first`'s Dungeons 1, 6, 8 and 9 are d2first's after phase IX (round 31, branch `r31-steps`): the blocks
`dungeon-1`, `dungeon-6`, `dungeon-8` and `dungeon-9` take its changes when it is merged, for both routes at once.
