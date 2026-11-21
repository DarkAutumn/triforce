# d3first: Dungeon 3 first, the wood sword to the grave — the route as built

The owner's second route (Nov 19; `docs/decisions.md` 2026-11-19 "Two routes"): the speedrunners' route, kept beside
`d2first`, and the main route since 2.0.0.0 (`triforce/registry.py` `MAIN_ROUTE`): a golden path on it from console
power on reached the ending (round32.md §11; decisions 2026-11-20 "2.0.0.0: `d3first` is the main route"). The
wood sword until the magic sword (no white sword); Dungeons 3, 4, 1, 5, 2, 7, the magic sword, 6, 8, 9.
Walked in the emulator before it was written: `docs/routes/d3first-walks-a.md` (the start to Level 5's entrance) and
`d3first-walks-b.md` (Level 5 to the ending), their scripts `scripts/experiments/r32_*.py` and `r32b_*.py`. Data:
`triforce/routes.yaml` `d3first` (304 steps since phase IX's merge: see "Phase IX" below), rooms in
`triforce/game.yaml`; its start states are `d3first_<name>` (`states: d3first`, `docs/architecture.md` "Routes").

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
| `dungeon-5` | 98-119 | 5:0x66 to 0x14 (the entrance, step 97, is each route's own: d3first's grants the bag) |
| `dungeon-2` | 145-154 | 2:0x7d to 0x0d |
| `to-level-7` | 156-173 | 0:0x45 out of the whirlwind to 0x42's pond: the stretch `dock-44`, the arrows, the meat |
| `dungeon-7` | 174-193 | 7:0x79 to 0x2b |
| `to-level-6` | 194-210 | 0:0x42 to 0x22: the stretch `woods`, the graveyard, the magic sword (12 containers) |
| `dungeon-6` | 211-226 | 6:0x79 to 0x0c |
| `to-level-8` | 227-232 | 0:0x22 (two plays facing down) to 0x6d's burned tree |
| `dungeon-8` | 233-242 | 8:0x7e to 0x2c |
| `to-level-9` | 243-257 | 0:0x6d (one play facing down) to 0x05: 0x47's tree and heart (15), the ladder at 0x17 |
| `dungeon-9` | 258-303 | 9:0x76 to Zelda's room (`exit: end`): the red ring, the silver arrow, Ganon |

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
| 88 | `0_2dw` | 0:0x2d | W → cave | ow-zora | the rock wall (one bomb facing N from (`$50`,`$5D`); `bombs: required`) | 7 / W |
| 89 | `0_2dcave` | 0:0x2d cave | S → S | ow-zora | `waypoint: rupees`: 30 | 7 / W |
| 90 | `0_2dc` | 0:0x2d | cave → W | ow-zora | | 7 / W |
| 91 | `0_2ce` | 0:0x2c | E → N | ow-tektite | | 7 / W |
| 92 | `0_1cs` | 0:0x1c | S → N | ow-start | | 7 / W |
| 93 | `0_0cs` | 0:0x0c | S → cave | ow-tektite | the shop's open mouth | 7 / W |
| 94 | `0_0ccave` | 0:0x0c cave | S → S | ow-tektite | `waypoint: blue-candle` (60, `requires_rupees`) | 7 / W |
| 95 | `0_0cc` | 0:0x0c | cave → W | ow-tektite | the wrap exit (CLIP E) onto the mountain top | 7 / W |
| 96 | `0_0be` | 0:0x0b | E → cave | ow-leever | Level 5 | 7 / W |
| 97 | `5_76s` | 5:0x76 | S → N | gibdo | `bomb_reserve: 8`, `grant: {bombs: 8}`: the bag topped up to 8 (round 32) | 7 / W |
| 98-119 | | `dungeon-5` | | (d2first's) | the Olive Garden, the recorder, Digdogger's heart, the piece | 7 → 8 / W |
| 120 | `0_0bc` | 0:0x0b | cave → recorder | ow-leever | one play facing right or up (MOVE E first) | 8 / W |
| 121 | `0_74r` | 0:0x74 | recorder → W | ow-tektite | | 8 / W |
| 122 | `0_73e_v2` | 0:0x73 | E → N | ow-moblin | | 8 / W |
| 123 | `0_63s_v2` | 0:0x63 | S → W | ow-moblin | | 8 / W |
| 124 | `0_62e` | 0:0x62 | E → cave | ow-moblin | the tree burned (the blue candle facing W from (`$A0`,`$5D`)) | 8 / W |
| 125 | `0_62cave` | 0:0x62 cave | S → S | ow-moblin | `waypoint: rupees`: 100 | 8 / W |
| 126 | `0_62c` | 0:0x62 | cave → N | ow-moblin | | 8 / W |
| 127 | `0_52s` | 0:0x52 | S → W | ow-moblin | | 8 / W |
| 128 | `0_51e` | 0:0x51 | E → W | ow-moblin | the wrap exit into 0x50 | 8 / W |
| 129 | `0_50e` | 0:0x50 | E → S | ow-lynel | | 8 / W |
| 130 | `0_60n` | 0:0x60 | N → W | ow-lynel | the wrap exit round the world into 0x5f | 8 / W |
| 131 | `0_5fe` | 0:0x5f | E → N | ow-zora | the ladder clip, the ladder heart | 8 / W |
| 132 | `0_4fs` | 0:0x4f | S → N | ow-zora | | 9 / W |
| 137 | `0_3fs` | 0:0x3f | S → N | ow-zora | the raft | 9 / W |
| 138 | `0_2fs` | 0:0x2f | S → cave | ow-start | | 9 / W |
| 139 | `0_2fcave` | 0:0x2f cave | S → S | ow-start | `waypoint: heart` (the raft heart) | 9 / W |
| 140 | `0_2fc` | 0:0x2f | cave → S | ow-start | the raft back | 10 / W |
| 141 | `0_3fn` | 0:0x3f | N → W | ow-zora | | 10 / W |
| 142 | `0_3ee` | 0:0x3e | E → W | ow-zora | the wrap exit into 0x3d | 10 / W |
| 143 | `0_3de` | 0:0x3d | E → cave | ow-moblin | the Armos at X `$90` woken from its E side | 10 / W |
| 144 | `0_3dcave` | 0:0x3d cave | S → S | ow-moblin | `waypoint: rupees`: 30 | 10 / W |
| 145 | `0_3dc` | 0:0x3d | cave → S | ow-moblin | out E of the stairs, then S | 10 / W |
| 146 | `0_4dn` | 0:0x4d | N → W | ow-moblin | | 10 / W |
| 147 | `0_4ce` | 0:0x4c | E → N | ow-octorok | | 10 / W |
| 148 | `0_3cs` | 0:0x3c | S → cave | ow-octorok | Level 2; `bomb_reserve: 1` (Dodongo's bomb) | 10 / W |
| 149-158 | | `dungeon-2` | | (d2first's) | Dodongo's heart, the piece | 10 → 11 / W |
| 159 | `0_3cc` | 0:0x3c | cave → recorder | ow-octorok | one play facing right or up (MOVE E first); `bomb_reserve: 0` | 11 / W |
| 160-177 | | `to-level-7` | | (d2first's) | the arrows (80), the meat (60), the pond | 11 / W |
| 178-197 | | `dungeon-7` | | (d2first's) | the red candle, the meat given, Aquamentus's heart, the piece | 11 → 12 / W |
| 198-214 | | `to-level-6` | | (d2first's) | the magic sword at the grave | 12 / W → Ma |
| 215-230 | | `dungeon-6` | | (d2first's) | Gohma's heart, the piece | 12 → 13 / Ma |
| 231-236 | | `to-level-8` | | (d2first's) | two plays facing down: 0x3c; 0x6d's tree | 13 / Ma |
| 237-246 | | `dungeon-8` | | (d2first's) | Gleeok's heart, the piece | 13 → 14 / Ma |
| 247-261 | | `to-level-9` | | (d2first's) | one play facing down: 0x37; 0x47's heart (15); 0x05's wall | 14 → 15 / Ma |
| 262-307 | | `dungeon-9` | | (d2first's) | the red ring, the silver arrow, Ganon, Zelda (`exit: end`) | 15 / Ma |

Dungeon entrances (`crates/zelda/tests/d3first.rs`, `tests/test_game_map.py`): Level 3 with 3 containers and no
piece; Level 4 with 4, the raft, piece 3; Level 1 with 5, the raft and the ladder, pieces 3 and 4; Level 5 with 7,
the bow too, pieces 1, 3, 4; Level 2 with 10, the recorder too, pieces 1, 3, 4, 5; Level 7 with 11, pieces 1-5;
Level 6 with 12 and the magic sword, pieces 1-5 and 7; Level 8 with 13, pieces 1-7; Level 9 with 15, all eight.

**The recorder** (walks-b §1; `zelda::whirlwind::plays` from the game's index `$523`, which only these four plays
step): out of Level 5 (index 0, pieces 1, 3, 4, 5) one play facing right or up lands in 0x74; out of Level 2 (index
on Level 3) one play facing right or up lands on 0x45's pier; out of Level 6 two plays facing down land in 0x3c; out
of Level 8 one play facing down lands in 0x37. A step `exit: recorder` with the landing room next is all the route
data says: the objective computes the plays and facing, the session offers RECORDER only facing the way the next
play must step (`GameCompletion::recorder_facings`), so on the first two the family turns E (a MOVE E) before it
plays; Link comes out of a level facing down. ow-leever (0x0b) and ow-octorok (0x3c, now `items: {recorder: true}`)
are the families that turn.

**Clips.** The screen scrolls the walks found cheaper (0x76, 0x64, 0x47, 0x29-0x2b: the plain wrap W/E) are CLIP
actions the families choose (`clips: true`), nothing in the steps. The wrap exits of 0x75, 0x0c, 0x51, 0x60 and 0x3e
are the only ways through those rooms on this route (no `stretch:`, clips only, the owner's Nov 18 rule): a run with
the run-time switch off has no way past them. The shared blocks keep their stretches (`d4-island`, `dock-44`,
`woods`, `ring-07`, `arrow-20`, `arrow-20n`).

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
  rock wall (`requires_bombs`, `bomb_secret: [10, 2]`, the 30 secret), the tags 0x29, 0x2a, 0x2b ow-leever, 0x46
  ow-zora (its one monster a Zora, as 0x47's), 0x62 ow-moblin, 3:0x4a stalfos-keese-gel. Walk A proposed ow-octorok
  for 0x46; its monsters are one Zora. d2first plays the shared rooms, the data its models were trained on: in a
  first form (one map for both) the new tags opened paths for d2first's off-route objective (ow-lynel's wrong exit
  from 0x60 into 0x50 then pointed S), 0x2d's wall offered BOMB and a granted bomb on d2first's crossings, and the
  rule written against that (no bomb on a step crossing a rock-wall room) took d2first's granted bomb at 0x67 S -> E
  (the schema 5 traces of ow-lynel, ow-zora, ow-tektite and ow-octorok, `tests/test_obs_legacy.py`); with the route maps
  they are what they were (`game_map::tests::a_routes_own_rooms_leave_the_shared_map_alone`, `crates/zelda/tests/
  d3first.rs` from d2first's `0_2ds_0` and `0_3ds_0`).
- Families gain rooms (routes.yaml `families`, `specialists/*.yaml` `rooms:`): ow-zora 0x46 and 0x2d's cave;
  ow-leever 0x29, 0x2a, 0x2b; ow-moblin 0x62, its cave and 0x3d's cave; ow-tektite 0x0c's cave; stalfos-keese-gel
  3:0x4a. Their `start_states` globs (`0_46*`, `0_29*`, `0_2a*`, `0_2b*`, `0_62*`, `3_4a*`) come with the first
  states (a glob matching no state is refused: `tests/test_entry_states.py`).
- Segments and scopes: `eval_batch.ROUTE_SEGMENTS['d3first']` (`D3FIRST_SEGMENTS`, `_TARGETS`, `_MILESTONES`),
  `room_stats.ROUTE_SCOPES['d3first']`; `ROUTE=d3first TAG=... scripts/experiments/r56_chains.sh`.

## Reserves

- **Bombs.** 6 from Dungeon 3's entrance to 0x49 as d2first's; **0 from 0x59's fight (step 13, d3first's own; the
  lead, Nov 20) through 0x5b's last pass**: 0x4a's four bombs are 0x59's (five Red Darknuts, all to die: the
  BombFighter rule wins 60 of 60 with them, the sword alone 11 of 20), and 0x5b's four, taken on its last pass, pay
  for the two walls and Manhandla; 6 at 0x5c and 0x4c, 0 at Manhandla, 6 after him (the block's). 2 from
  Dungeon 4's entrance (the block's). **8 from Level 4's exit to Dungeon 5** (d3first's own steps 53 and 83): every
  bomb Link has is kept for the Olive Garden (walks-b §2: the wood sword clears it 9 of 20 without bombs, 18 of 20
  with eight); the walls on the way (1:0x53 after phase IX, 0x2c, 0x2d) are bombed as any wall, cheated where Link
  has none. **Dungeon 5's entrance tops the bag up to 8** (round 32, `grant: {bombs: 8}`, the lead on the owner's
  "can be ignored since we have the ability to cheat them to the character in game"): a route grant, outstanding
  until used and cheated where used (`zelda::cheats`); 5:0x66's and 5:0x65's walls grant and take their own bomb
  while it is outstanding, so the reserve of 8 brings the whole bag to the Garden. The Garden itself is the block's
  (0 in 0x05: the bag thrown freely, 8 after it, 0 at Digdogger). **1 from
  Level 2's mouth through Dungeon 2** (step 148; walks-b §4: goriya and moldorm are `free` and would throw the bag
  this route brings; Dodongo's room offers BOMB whatever the reserve), **0 out of Level 2** (step 159), as d2first's
  from Dungeon 5 on; the blocks' after that.
- **Rupees.** No rupee reserve of d3first's own: its own steps hold no arrows (the bow from Dungeon 1, the arrows
  only at 0x44 in `to-level-7`), so a reserve there would keep nothing. Where the ledger says the route runs dry
  (0x34's meat, then Gohma, 8:0x4c and Ganon) the shared blocks' reserves already sit: 2 from the arrows (0x44's cave
  step), 1 in Gohma's room, 8 from 8:0x4c, 1 through Level 9, 0 at Ganon. A reserve keeps rupees from ARROW; it
  cannot keep what was never there, so the arrow rooms' needs top the purse up (cheated, counted): Gohma's (round
  27), Ganon's (round 29) and, round 32, 8:0x4c's while a Pols Voice lives (`requires_rupees: 9`, the reserve and a
  shot; on both routes: d2first arrives with 22-38 and is never topped up).
- **Measured (round 32, r32-base2's team, clips on).** Dungeon 5 (`--from 5_76s_0 --stop-at step:120`, 300 runs):
  6.0% (1.9% of 1000 before); every run enters the Olive Garden with the bag of 8 (before: 0 in 822 of 1000, 4 in
  the rest) and throws 7.7 there (5.9 cheated); the Garden loses 151 of 300 (845 of 1000 before); 8.1 cheated bombs
  a run (the walls' two, the Garden, Digdogger). Dungeon 8 (`--from 8_7es_0 --stop-at step:243`, 300 runs): 24.3%
  (6.8% before); 8:0x4c loses 5 of 107 entering (268 of 371 before), 8.0 Pols Voices killed a visit (0.88 before),
  0.93 cheated rupees a visit, 0.33 a run. `d2first`'s Dungeon 8 on seeds 0-99: 20 of 100 before and after, every
  run's outcome and rooms the same, no rupee cheated.

## Ledgers (from the walks; staging without drops unless said)

| After | Containers | Keys | Bombs | Rupees |
|---|---|---|---|---|
| the sword | 3 | 0 | 0 | 0 |
| Dungeon 3 | 4 | 2 | 6 (the rule's Manhandla: about 4) | 0 |
| Dungeon 4 | 5 | 3 | 4 (about 2) | 10 (4:0x11's stash) |
| Dungeon 1 | 6 | 3 (1 skipping 0x23's and 0x53's keys) | 3 (about 1) | 10 |
| 0x2c's heart, 0x2d's 30 | 7 | 3 | 1 (about 0: cheated) | 40 |
| the blue candle, Level 5's mouth | 7 | 3 | 1 | 0 (20 cheated of 60) |
| Dungeon 5 | 8 | (d2first's ledger: 3 spent, 2 taken) | what the Garden left, 0x65's four eastbound | 0-1 |
| 0x62's 100, 0x5f, 0x2f, 0x3d's 30 | 10 | | | 130-131 |
| Dungeon 2 | 11 | | | 138-147 (d2first's drops there) |
| 0x44's arrows (80), 0x34's meat (60) | 11 | | | 0-7 (cheated at 0x34 when short) |
| Dungeon 7 | 12 | | | 0-11 |
| the magic sword, Dungeon 6 | 13 | | | 0-8 (Gohma's arrows: topped up while he lives) |
| Dungeon 8 | 14 | | | 0-7 (8:0x4c's arrows: topped up to 9 while a Pols Voice lives, round 32) |
| 0x47's heart, Level 9 | 15 | | | 0-6 (Ganon's silver arrow: topped up to 1 while he lives, round 29) |

Rupees in all: 30 (0x2d) + 100 (0x62) + 30 (0x3d) + 10 (4:0x11) and the dungeons' drops, against 60 + 80 + 60 for the
wares and 3-5 for arrows.

## For the owner

From the first walk (`d3first-walks-a.md`):

1. **Level 4 → Level 1.** 0x57 has no N edge and 0x47 none to 0x37; built as walked: 0x55 E, 0x56 N, 0x46 E, 0x47 E,
   0x48 N, 0x38 W, 0x37 (0x46 new; about 0.8 hearts by the rule).
2. **0x2d → the candle shop.** 0x1d's S part leads only E; built as walked: back W into 0x2c, N 0x1c, N 0x0c.
3. **"5-8 bombs leaving Level 1"** is not met by the rooms (1-3 after Dungeon 1, none after 0x2c's and 0x2d's walls):
   walls are cheated and counted; the Garden's bombs are what the route brings (see 5).
4. **"About 30 rupees leaving Level 1".** The route's own are 4:0x11's 10; with 0x2d's 30 Link has 40 at the candle:
   20 cheated (the shop's `requires_rupees: 60`).
5. **Dungeon 3 at 3 hearts with the wood sword** is the hardest stretch of the first half for a scripted rule (0x59's
   five Darknuts, Manhandla by bombs only); nothing was changed in the route for it.
6. **Dungeon 1's key rooms.** With phase IX's door trick 0x23's Goriya key and 0x53's key are not needed on this
   route (one key left at Level 1's exit). Built as the shared block (the lead: "the rest is mostly the same"); a
   d3first-only Dungeon 1 that skips 0x23's fight would need its own block.
7. **0x76**: both walked; the scroll costs nothing and saves 246 frames (a CLIP the family may choose).

From the second walk (`d3first-walks-b.md`):

8. **The warps out of Levels 5 and 2** need Link to face right or up first (down takes three and four plays). The
   engine computes it and masks RECORDER while he faces down; ow-leever and ow-octorok must learn the turn.
9. **The dead woods and the world wrap** are 0x62's tree, then 0x51's wrap exit into 0x50 and 0x60's round the world
   (404 frames shorter than the Lost Woods' sequence).
10. **0x3d has no W edge**: "scroll over to Level 2" is 0x3d S, 0x4d W, 0x4c N.
11. **Hearts into Dungeon 5 are 7** (3, Dungeons 3, 4, 1, the heart rock); the grave sees exactly 12 and gives the
    magic sword (refused at 11).
12. **The wood sword in the Olive Garden**: 9-10 of 20 clears by the scripted fighter without bombs, 17-18 with
    eight; this route's bomb reserve 8 from Level 4's exit keeps what Link finds for it. Answered by the owner's list
    ("The 'you need x bombs and rupees' can be ignored since we have the ability to cheat them to the character in
    game") and the lead (Nov 20): Dungeon 5's entrance tops the bag up to 8, cheated and counted where thrown.
13. **The wood sword in Dungeon 7**: 0x1c's small Digdoggers (8 wood hits each) kill the scripted fighter 7 of 20
    from full hearts; 0x49, 0x1b and Aquamentus cost 1-2 hearts more each. Nothing to walk instead.
14. **Rupees**: about even over the route; a run with few drops reaches 0x34 short (cheated), Gohma short (cheated),
    and 8:0x4c and Ganon with little for their arrows: both topped up while their monsters live (Ganon's since round
    29, 8:0x4c's from round 32), cheated and counted.

From building it:

15. **Dungeon 2's bomb.** Walk B asked a reserve of one "until 0x0e, 0 at Dodongo": d2first's Dungeon 2 sets no
    reserve, and the shared block must stay d2first's step for step, so the reserve is 1 from Level 2's mouth through
    Dodongo's room too. It changes nothing there: Dodongo's room (`requires_bombs`, no wall) offers BOMB to his
    family whatever the reserve.
16. **0x2d's and 0x3d's secrets live in rooms d2first crosses.** Giving them their caves changed two engine rules so
    d2first's crossings stay what they were (no granted bomb in 0x2d, no MOVE into the Armos in 0x3d); both are
    tested in the game from d2first's own states.
17. **Clips only** at five rooms (0x75, 0x0c, 0x51, 0x60, 0x3e): d3first has no plain way; `CLIPS=off` chains cannot
    pass them.

## For the agents that build its states

The list of steps whose room the walks crossed only by a rule the explorer may not find (a fight to the end with the
wood sword, a bomb fight) is in the round's status notes and repeated here: the start state after such a room comes
from a rule-won run (the walks' scripts: `r32_d3first.py` `BEST`, `ManhandlaFighter`, `BombFighter`;
`r28_gleeok.GleeokFighter`; `r32b_d5.py garden`/`digdogger`; `r32b_d7.py`), not from the explorer.

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
| Dungeon 1 | 1:0x35 (81): Aquamentus, 6 wood hits | 19/20 | `1_36w` |
| Dungeon 5 | 5:0x05 (107): the Olive Garden, six Blue Darknuts | `r32b_d5.py garden` (`BombFighter`), 9-18/20 by bombs held | `5_04cave` |
| Dungeon 5 | 5:0x24 (122): Digdogger, the small one 8 wood hits | `r32b_d5.py digdogger`, 20/20 from 7 hearts | `5_14s` |
| Dungeon 7 | 7:0x49 (181): 3 Blue Goriyas for the N shutter | `r32b_d7.py`, 13/20 from 11 | `7_39s` |
| Dungeon 7 | 7:0x1b (190): 6 Blue Goriyas | 20/20 | `7_1cw` |
| Dungeon 7 | 7:0x1c (191): the small Digdogger by the recorder | 13/20 from 11 (7 dead) | `7_0cs` |
| Dungeon 7 | 7:0x2a (196): Aquamentus | 18/20 | `7_2bw` |

The rest (the legs, Dungeon 2 at ten containers, and Dungeons 6, 8, 9 with the magic sword: d2first's inventory
there but the purse) the walks crossed by walking or d2first's own rules.

## The states as built

`d3first_*` (`triforce/states/`, chains 0-2, steps 0-303, the game start to Ganon's room) were built stretch by
stretch by `scripts/experiments/r32c_states.py --first N --last M --rules ... --tag T` (one job a stretch): each step
of the clip way crossed per chain by the explorer (`entry_states build`, one step), a step named in `--rules` by the
walks' rule (`r32c_harvest.py`: the first won seed of a chain gives the next state, every seed played for the rate)
or by round 28's/29's rule (`STEM=ROOM`, `r28_demos.py`/`r29_demos.py ROOM --route d3first --harvest`); where the
explorer fails, the walks' actions or fight for the step (`r32_d3first.LEGS`, `r32c_harvest.FIGHTS`/`tallies()`),
round 28's/29's rule of the room, and with `--models` the promoted models (`r32c_models.py`: `eval-chain --route
d3first --from STEM_<chain> --stop-at step:<next> --replays`, the won replays' pre-entry frame of the next room cut
out). Every harvested state is written by `write_entry_state` (hearts refilled, the room fresh); its manifest entry's
`played_by` names the rule. No fight is skipped. Wins over runs, per chain (0, 1, 2):

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
