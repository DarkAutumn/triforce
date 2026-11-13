# Specialist families (the d2first route)

The families cover the rooms of the route, `d2first` (`docs/routes/d2first.md`; Lee's reset, Nov 8; the Dungeon 1
bow detour and the way on through Dungeon 3 since Nov 9, Dungeon 4 since Nov 12, round 22's ways since Nov 13), plus
the route's alternate 0x78 and three rooms round 22 took off the route (0:0x56, 0:0x57, 3:0x4b: still tagged, no
start states, for a Link who strays in). The enemy data comes from
ROM decode, v1 data and emulator checks. Room tags are in `triforce/game.yaml` (`specialist:`), and the same lists
are in `triforce/routes.yaml` under `families:`; the specialist configs must agree (checked at load).

Grouping rule (design §3.1, revised by Lee on Nov 3, 2026):
- Group rooms by their enemies, not by dungeon: a room full of Red Goriyas needs the same skills whichever
  dungeon it is in, only the geometry differs.
- A family normally covers at least 2 rooms; an enemy that needs its own skills (Wallmasters, Lynels, Tektites,
  falling boulders) gets its own family even with 1 room so far.
- Bosses get single-room families.
- A room that fits no family goes into the nearest one.

## Families

| Family | Main threat | Rooms (level, room) | Success (§3.3) | Rationale |
|---|---|---|---|---|
| `ow-start` | none | (0,0x77), (0,0x77 cave), (0,0x1c), (0,0x1b) | exit-correct. The correct exit depends on progress: no sword → cave; in the cave → S; with sword → N; round 22, after Dungeon 1 on the clip way, from N → W. 0x1c from S and the mountain maze 0x1b from E leave W. | Rooms without enemies: walking to a spot is this family's skill. v1 treats the cave interior as its own `MapLocation` (`in_cave`). |
| `ow-octorok` | Octorok (Red/Blue, slow/fast) | (0,0x67), (0,0x67 cave), (0,0x68), (0,0x58), (0,0x66), (0,0x4c), (0,0x3c), (0,0x37), (0,0x57), (0,0x64); alt (0,0x78) | exit-correct. At 0x3c (from S) and 0x37 the correct exit is the dungeon stairs: enter-room (2,0x7d) / (1,0x73). Round 22 after Dungeon 1: 0x58 N→S, 0x68 N→W, 0x67 E→cave (bomb the rock wall), the cave (`waypoint: rupees`, X `$78`), 0x67 cave→S (clips) or →W, 0x66 E→W and 0x64 E→W (plain). `items: {bombs: required}` since round 22. | Octoroks are the most numerous enemy in every one of these rooms. 0x64's potion shop and 0x66's shop mouths are open (wrong exits). 0x57 off the route since round 22. |
| `ow-zora` | Zora (fireball) | (0,0x59), (0,0x5a), (0,0x2d), (0,0x27), (0,0x28), (0,0x38), (0,0x56), (0,0x55), (0,0x65) | exit-correct. 0x38 twice (to D1 from N, toward D3 from W). 0x65 E→W (round 22's plain way, across its ford) and W→N (to the raft). | The single Zora and its fireball are the main threat on these screens, alongside Peahats (0x27, 0x28, 0x59) and Octoroks (0x38, 0x2d, 0x5a, 0x55, 0x56, 0x65). 0x56 off the route since round 22. |
| `ow-moblin` | Moblin | (0,0x5b), (0,0x5c), (0,0x5d), (0,0x4d), (0,0x3d), (0,0x63), (0,0x73) | exit-correct | Moblin rooms of row 5 on the way to Dungeon 2, 0x4d/0x3d after it, 0x63/0x73 on the plain way to Dungeon 3 and back from it. |
| `ow-leever` | Leever | (0,0x48), (0,0x75) | exit-correct. 0x48 N to S. 0x75 (round 22, the clip way) from E W by `CLIP E`, the wrap exit through its solid W edge into 0x74 (the only way out of its E part). | Nov 9: Leevers burrow and surface by Link, unlike any other route enemy; 0x48 (Red Leever ×4, Zora) is forced on the way from D1 to D3; 0x75 holds 3 Leevers, Peahats and a Zora. |
| `ow-tektite` | Tektite | (0,0x1a), (0,0x2c), (0,0x2c cave), (0,0x74), (0,0x76) | exit-correct. 0x1a: from E N to 0x0a, from N W (from N without the white sword: back N). 0x2c: bomb the rock wall, take the heart container in the take-any cave, out N. 0x74: into Dungeon 3 (enter-room (3,0x7c)) from W or (round 22, by the wrap exit) from E; out of Dungeon 3 W. 0x76 E→W (round 22's clip way). | Lee (Nov 8): Tektites' jump patterns are their own. 0x1a RedTektite×6 and a Zora; 0x2c RedTektite×4 (moved here from `ow-leever-peahat`, Lee Nov 8); 0x74 RedTektite×1; 0x76 RedTektite×4. 0x2c is `requires_bombs`, so the family has bombs. |
| `ow-lynel` | Lynel | (0,0x0a), (0,0x0a cave) | exit-correct: into the white-sword cave, take the sword (5 hearts), out S. Leaving S before the sword is a `retreat` (neutral by default; ow-lynel ships `endings: {retreat: soft-failure}` for the baseline). | Lee (Nov 8): Lynels are very hard and the main enemy of many later rooms. |
| `ow-boulder` | Falling boulders | (0,0x19), (0,0x18), (0,0x17) | exit-correct | Lee (Nov 8): boulders are hard to predict and can't be blocked. A boulder generator ($1F), its boulders ($20; both hazards, not enemies) and a Zora. |
| `stalfos-keese-gel` | Stalfos / Keese / Gel | (1,0x73), (1,0x72), (1,0x74), (1,0x63), (1,0x53), (1,0x52), (1,0x42), (1,0x43), (1,0x33), (1,0x36) | exit-correct. 0x73 three times: W (key room 0x72), then E (key room 0x74), then N. 0x43 twice: from W N (the bow detour), from N E. 0x33 twice: from S N, from N S. At 0x36 the "exit" is triforce-collected (the D1 triforce; the route goes on from 0x37). | Was `d1-melee`. Includes the key rooms and all D1 locked doors on the way except 0x45 N. 0x36 (triforce room) and 0x73 (entrance hall) have no enemies and go to the nearest family. 0x33 (Stalfos×3, key) is north of 0x43, on the bow detour. |
| `goriya` | Red Goriya | (2,0x5e), (2,0x1e), (1,0x44), (1,0x23) | exit-correct. 0x23 twice: from S W (to the trap room), from W S. | Every Red Goriya room, whatever the dungeon. 0x5e (×5) has the ringleader trigger; 0x1e (×5) needs kill-all to open the N shutter into Dodongo's room; 0x44 (×3) drops the boomerang when all are dead; 0x23 (×3) drops its key when all are dead, north of 0x33 on the bow detour. |
| `trap-room` | Blade traps | (1,0x22), (3,0x5a), (3,0x4c) | exit-correct. 1:0x22 from E: into the diamond, take the stairs down to the bow cellar (since round 21 by `CLIP N` under the east block; the push of the west block N or S still works); out of the cellar: E. 3:0x5a E→W and W→E. 3:0x4c S→E (W→E before round 22): bomb the E wall into Manhandla's room. | Lee (Nov 9): blade-trap rooms. Four traps ($49) that slide at Link when he lines up with them and can't be killed; 0x22 has no monsters, 0x5a 4 Blue Keese, 0x4c 2 Zols (entered from S, its two bottom traps charge along the door row). 3:0x4c is `requires_bombs`, so the family has bombs. |
| `equipment-room` | Blue Keese | (1,0x7f cellar), (3,0x0f cellar) | exit-correct: take the bow (raft), then up the left ladder to 0x22 (0x69) (leaving without it is a wrong exit). | Lee (Nov 9): every item cellar ("all equipment rooms are pretty much the same"): side view, walkways and ladders, four Blue Keese. |
| `wallmaster` | Wallmaster | (1,0x45) | exit-correct. 0x45 exits N through a locked door. | Wallmasters need their own skills (v1's Wallmaster shaping, design §6). |
| `boss-aquamentus` | Aquamentus | (1,0x35) | exit-correct after killing the boss and collecting the heart container; then E into 0x36. | Boss room per `asm:Level_InfoUW1` `asm:LevelInfo_BossRoomId`=$35. |
| `d2-rope` | Rope | (2,0x7d), (2,0x7e), (2,0x6e), (2,0x4e), (2,0x2e), (2,0x0d) | exit-correct; triforce-collected at 0x0d. 0x7d exits E into key room 0x7e, then 0x6e is entered from S. 0x2e requires kill-all to open the N shutter. | 0x7d is the empty entry room; 0x0d, the empty triforce room, joins like 1:0x36 joined stalfos-keese-gel. Off-route 0x6d left the family with the reset. |
| `moldorm` | Moldorm | (2,0x3e) | exit-correct (N into 0x2e). | Split out of d2-rope in round 11 (Lee, Nov 8): one Moldorm, not a boss room. |
| `boss-dodongo` | Dodongo (`Dodongo2` $32) | (2,0x0e) | exit-correct after killing him (bomb stun, then the sword) and taking the heart container; then W into 0x0d. | Round 12. Boss room per `asm:Level_InfoUW2` `asm:LevelInfo_BossRoomId`=$0E; `requires_bombs`. |
| `zol` | Zol (Gels when split), Bubble, Keese | (3,0x7c), (3,0x7b), (3,0x6b), (3,0x49), (3,0x4b), (3,0x3d) | exit-correct; triforce-collected at 0x3d. 0x7b, 0x49 visible keys; 0x6b's key when all are dead. 0x4b off the route since round 22 (tagged, no start states). | Nov 9: Dungeon 3's Zol rooms; the empty entrance hall 0x7c and triforce room 0x3d join like 1:0x73/0x36 and 2:0x7d/0x0d. Not `stalfos-keese-gel`: Zols are not D1's Gels, and its trained model stays as it is. |
| `darknut` | Darknut (`RedDarknut` $0B) | (3,0x5b), (3,0x59), (3,0x69), (3,0x5c) | exit-correct. 0x5b twice (S→W skipping its bombs; round 22: W→E through its bomb wall, taking the bombs only below 6); 0x59 three times (E→N; N→S after killing all 5 for the shutter; S→E); 0x69 twice (N→stairs, stairs→N, its bombs skipped); 0x5c W→N (round 22). | Nov 9: shielded in front, only hit from the side or back. Bombs drawn 0–8 at reset so 0x5b's bomb rule trains both ways; `free` bombs (a weapon above the reserve, and in 0x5b, `requires_bombs` since round 22). |
| `boss-manhandla` | Manhandla ($3C) | (3,0x4d) | exit-correct after killing him and taking the heart container; then N into 0x3d. | Boss room per Lee's D3 route (entered through 0x4c's bombed wall); `requires_bombs` (Lee: kill him with bombs dropped in his path). |

Rooms per family: `ow-start` 4, `ow-octorok` 11 (one an alternate), `ow-zora` 9, `ow-moblin` 7, `ow-leever` 2,
`ow-tektite` 5, `ow-lynel` 2, `ow-boulder` 3, `stalfos-keese-gel` 13, `goriya` 4, `trap-room` 5, `equipment-room` 3,
`wallmaster` 1, `boss-aquamentus` 1, `d2-rope` 6, `moldorm` 1, `boss-dodongo` 1, `zol` 7, `darknut` 4,
`boss-manhandla` 1, `vire` 8, `like-like` 1, `boss-gleeok` 1, `ow-peahat` 1.

**Bombs in `required` families** (round 22; decisions 2026-11-13 "Round 22 route"): ow-tektite, ow-octorok,
trap-room and stalfos-keese-gel are offered BOMB in their `requires_bombs` rooms only where the bomb would open the
wall the objective leaves through (a dungeon bomb wall within `$18` px of its hotspot, a rock wall within `$10` px),
else only on the 10th damage-free kill; `free` families (darknut, vire, the bosses) are untouched.

**Clips** (round 21; `docs/architecture.md` "Grid-point skip", `docs/experiments/round21.md`): a family's YAML
`clips: true` offers the `CLIP` action where a skip is available in its rooms; the default is false. On: `vire` (the
block clip in 4:0x50's diagonal and 4:0x03), `stalfos-keese-gel` (the ladder clip off 4:0x21's island), `trap-room`
(the block clip in 1:0x22) and the nine overworld families (the screen wrap; ow-lynel's 0x0a, ow-leever's 0x48 and
ow-peahat's 0x45 offer none, so for those three the switch changes nothing yet). Round 22: ow-leever's 0x75 offers
the wrap exit (`CLIP E`), and the run-time clips switch turns every CLIP off at any moment
(decisions 2026-11-13 "The run-time clips switch and route stretches").

### Dungeon 4 (scouted Nov 11, merged into `d2first` and trained Nov 12; `docs/routes/d4.md`)

`d2first` goes on from the Dungeon 3 triforce through Dungeon 4 (steps 98–122; 98–120 since round 21's shortcut;
105–133 since round 22).
Its rooms are tagged by enemy as everywhere; the existing families train on their Dungeon 4 rooms' start states too.

| Family | Main threat | Rooms (level, room) | Notes |
|---|---|---|---|
| `vire` (new) | Vire (splits into two Red Keese) | (4,0x71), (4,0x61), (4,0x50), (4,0x30), (4,0x31), (4,0x20), (4,0x12), (4,0x03) | The dungeon's main family, with its empty hall and triforce room. 0x30 S→E (a locked door) and, on the plain way of `d4-island` (round 22), E→N; 0x31 W→E after killing all five for the shutter, E→N through the locked door onto 0x21's island (clip way) or E→W (plain way); 0x20 S→E (plain way); 0x12 (from N since round 22): kill them, push the block, E through the shutter. `clips: true` (the block clip in 0x50). |
| `like-like` (new) | Like Like, Zol, Bubble | (4,0x32) | The Breakfast Room: kill them, push the block, down the stairs; back up, W. |
| `boss-gleeok` (new) | Gleeok, two heads ($43) | (4,0x13) | Kill him, the heart container, N. |
| `ow-peahat` (new) | Peahat | (0,0x45) | The island: from the raft's pier into the dungeon. |
| `zol` | | + (4,0x40) | Five Zols in the dark; take the key (in the open). |
| `stalfos-keese-gel` | | + (4,0x51), (4,0x21), (4,0x01) | Eight Blue Keese (take the key, in the open); five Blue Gels in the dark, round 21: in from S onto the island, `CLIP N` (the ladder clip, `clips: true`), bomb the N wall (`items: {bombs: required}`, which unmasks BOMB only in this `requires_bombs` room, and since round 22 only where the bomb opens the wall); from W (the plain way) bomb the N wall. 0x01 (round 22): eight Blue Keese in the dark, the key on the water (the ladder), E. |
| `trap-room` | | + (4,0x11), (4,0x02) | The rupee room: no monsters, no traps, bomb the N wall (round 22; the E wall before; Lee, Nov 12: it stays in this family). 0x02 (round 22): six blade traps in the dark, W→S. |
| `equipment-room` | | + (4,0x60 cellar) | The ladder cellar. |

Removed with the reset (Nov 8): `ow-leever-peahat` (0x29, 0x2a, 0x2b: milestone1 only; 0x2c went to `ow-tektite`),
the fairy pond 0x39 (ow-start), 0x48 (ow-zora; back on the route since Nov 9 as `ow-leever`), 0x49 (ow-octorok) and
2:0x6d (d2-rope).

## Changes from the design hypothesis (§3.1)

| Hypothesis | Decision | Evidence |
|---|---|---|
| `ow-zora` | Restored by Lee: the Zora fireball is the main threat on these screens | Zora appears in every water room on the route (OW attr A&$08), always ×1 next to 4–6 other enemies. |
| `d1-goriya` | Merged into `d1-wallmaster`; split out again as the cross-dungeon `goriya` family (Nov 3, 2026) | 0x44 was the only Goriya room in D1; the D2 Goriya rooms need the same skills |
| `d1-triforce` | Merged into `d1-melee`, now `stalfos-keese-gel` | Single non-boss room |
| `d2-*` | `d2-rope`; D2's Goriya rooms joined `goriya`; the Moldorm room its own `moldorm` | From the D2 ROM decode. D2's route has no locked doors or block pushes; it collects the 0x7e key. |
| `ow-leever-peahat` | Removed with milestone1 (Nov 8) | Its rooms 0x29–0x2b are not on d2first; 0x2c is a Tektite room |
| — | Added `ow-moblin`, `ow-lynel`, `ow-tektite`, `ow-boulder`; `trap-room`, `equipment-room` (Nov 9); `ow-leever`, `zol`, `darknut`, `boss-manhandla` (Nov 9, Dungeon 3) | Moblin-only rooms; Lee (Nov 8) for the next three; Lee (Nov 9) for the bow detour's trap room and cellar; by enemy for the D3 extension (decisions 2026-11-09) |

## Start states

Every route room has route-generated pre-entry states (`python -m triforce.entry_states build`, chains 0–2, staged
under `target/entry-states/<ts>` and copied in with `entry_states install <dir>`; `entry_states.json`); the bow
detour's (Nov 9) were built from `1_43w_N` with `--start-step 55`, the way on to and through Dungeon 3 (Nov 9)
from `1_36w_N` with `--start-step 67` (then `0_56e_N`, `--start-step 73`). The Dungeon 4 states (Nov 11:
`--route d2first-d4 --start-step 97` from `3_3ds_N`, then a route of its own) moved into the states dir with the
route merge (Nov 12; `docs/routes/d4.md`, "Start states").
