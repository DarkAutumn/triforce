# Specialist families (the d2first route)

The families cover the rooms of the route, `d2first` (`docs/routes/d2first.md`; Lee's reset, Nov 8; the Dungeon 1
bow detour and the way on through Dungeon 3 since Nov 9), plus the route's alternate 0x78. The enemy data comes from
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
| `ow-start` | none | (0,0x77), (0,0x77 cave), (0,0x1c), (0,0x1b) | exit-correct. The correct exit depends on progress: no sword → cave; in the cave → S; with sword → N. 0x1c from S and the mountain maze 0x1b from E leave W. | Rooms without enemies: walking to a spot is this family's skill. v1 treats the cave interior as its own `MapLocation` (`in_cave`). |
| `ow-octorok` | Octorok (Red/Blue, slow/fast) | (0,0x67), (0,0x68), (0,0x58), (0,0x4c), (0,0x3c), (0,0x37), (0,0x57), (0,0x64); alt (0,0x78) | exit-correct. At 0x3c (from S) and 0x37 the correct exit is the dungeon stairs: enter-room (2,0x7d) / (1,0x73). 0x3c, 0x4c, 0x37 and 0x58 are crossed twice (0x37 and 0x58 again after the D1 triforce, toward D3). | Octoroks are the most numerous enemy in every one of these rooms. 0x64's potion shop mouth is open (a wrong exit). |
| `ow-zora` | Zora (fireball) | (0,0x59), (0,0x5a), (0,0x2d), (0,0x27), (0,0x28), (0,0x38), (0,0x56), (0,0x55), (0,0x65) | exit-correct. 0x38 twice (to D1 from N, toward D3 from W). | The single Zora and its fireball are the main threat on these screens, alongside Peahats (0x27, 0x28, 0x59) and Octoroks (0x38, 0x2d, 0x5a, 0x55, 0x56, 0x65). |
| `ow-moblin` | Moblin | (0,0x5b), (0,0x5c), (0,0x5d), (0,0x4d), (0,0x3d), (0,0x63), (0,0x73) | exit-correct | Moblin rooms of row 5 on the way to Dungeon 2, 0x4d/0x3d after it, 0x63/0x73 on the way to Dungeon 3. |
| `ow-leever` | Leever | (0,0x48) | exit-correct (N to S) | Nov 9: Leevers burrow and surface by Link, unlike any other route enemy; 0x48 (Red Leever ×4, Zora) is forced on the way from D1 to D3. |
| `ow-tektite` | Tektite | (0,0x1a), (0,0x2c), (0,0x2c cave), (0,0x74) | exit-correct. 0x1a: from E N to 0x0a, from N W (from N without the white sword: back N). 0x2c: bomb the rock wall, take the heart container in the take-any cave, out N. 0x74: into Dungeon 3 (enter-room (3,0x7c)). | Lee (Nov 8): Tektites' jump patterns are their own. 0x1a RedTektite×6 and a Zora; 0x2c RedTektite×4 (moved here from `ow-leever-peahat`, Lee Nov 8); 0x74 RedTektite×1. 0x2c is `requires_bombs`, so the family has bombs. |
| `ow-lynel` | Lynel | (0,0x0a), (0,0x0a cave) | exit-correct: into the white-sword cave, take the sword (5 hearts), out S. Leaving S before the sword is a `retreat` (neutral by default; ow-lynel ships `endings: {retreat: soft-failure}` for the baseline). | Lee (Nov 8): Lynels are very hard and the main enemy of many later rooms. |
| `ow-boulder` | Falling boulders | (0,0x19), (0,0x18), (0,0x17) | exit-correct | Lee (Nov 8): boulders are hard to predict and can't be blocked. A boulder generator ($1F), its boulders ($20; both hazards, not enemies) and a Zora. |
| `stalfos-keese-gel` | Stalfos / Keese / Gel | (1,0x73), (1,0x72), (1,0x74), (1,0x63), (1,0x53), (1,0x52), (1,0x42), (1,0x43), (1,0x33), (1,0x36) | exit-correct. 0x73 three times: W (key room 0x72), then E (key room 0x74), then N. 0x43 twice: from W N (the bow detour), from N E. 0x33 twice: from S N, from N S. At 0x36 the "exit" is triforce-collected (the D1 triforce; the route goes on from 0x37). | Was `d1-melee`. Includes the key rooms and all D1 locked doors on the way except 0x45 N. 0x36 (triforce room) and 0x73 (entrance hall) have no enemies and go to the nearest family. 0x33 (Stalfos×3, key) is north of 0x43, on the bow detour. |
| `goriya` | Red Goriya | (2,0x5e), (2,0x1e), (1,0x44), (1,0x23) | exit-correct. 0x23 twice: from S W (to the trap room), from W S. | Every Red Goriya room, whatever the dungeon. 0x5e (×5) has the ringleader trigger; 0x1e (×5) needs kill-all to open the N shutter into Dodongo's room; 0x44 (×3) drops the boomerang when all are dead; 0x23 (×3) drops its key when all are dead, north of 0x33 on the bow detour. |
| `trap-room` | Blade traps | (1,0x22), (3,0x5a), (3,0x4c) | exit-correct. 1:0x22 from E: into the diamond, take the stairs down to the bow cellar (since round 21 by `CLIP N` under the east block; the push of the west block N or S still works); out of the cellar: E. 3:0x5a E→W and W→E. 3:0x4c W→E: bomb the E wall into Manhandla's room. | Lee (Nov 9): blade-trap rooms. Four traps ($49) that slide at Link when he lines up with them and can't be killed; 0x22 has no monsters, 0x5a 4 Blue Keese, 0x4c 2 Zols. 3:0x4c is `requires_bombs`, so the family has bombs. |
| `equipment-room` | Blue Keese | (1,0x7f cellar), (3,0x0f cellar) | exit-correct: take the bow (raft), then up the left ladder to 0x22 (0x69) (leaving without it is a wrong exit). | Lee (Nov 9): every item cellar ("all equipment rooms are pretty much the same"): side view, walkways and ladders, four Blue Keese. |
| `wallmaster` | Wallmaster | (1,0x45) | exit-correct. 0x45 exits N through a locked door. | Wallmasters need their own skills (v1's Wallmaster shaping, design §6). |
| `boss-aquamentus` | Aquamentus | (1,0x35) | exit-correct after killing the boss and collecting the heart container; then E into 0x36. | Boss room per `asm:Level_InfoUW1` `asm:LevelInfo_BossRoomId`=$35. |
| `d2-rope` | Rope | (2,0x7d), (2,0x7e), (2,0x6e), (2,0x4e), (2,0x2e), (2,0x0d) | exit-correct; triforce-collected at 0x0d. 0x7d exits E into key room 0x7e, then 0x6e is entered from S. 0x2e requires kill-all to open the N shutter. | 0x7d is the empty entry room; 0x0d, the empty triforce room, joins like 1:0x36 joined stalfos-keese-gel. Off-route 0x6d left the family with the reset. |
| `moldorm` | Moldorm | (2,0x3e) | exit-correct (N into 0x2e). | Split out of d2-rope in round 11 (Lee, Nov 8): one Moldorm, not a boss room. |
| `boss-dodongo` | Dodongo (`Dodongo2` $32) | (2,0x0e) | exit-correct after killing him (bomb stun, then the sword) and taking the heart container; then W into 0x0d. | Round 12. Boss room per `asm:Level_InfoUW2` `asm:LevelInfo_BossRoomId`=$0E; `requires_bombs`. |
| `zol` | Zol (Gels when split), Bubble, Keese | (3,0x7c), (3,0x7b), (3,0x6b), (3,0x49), (3,0x4b), (3,0x3d) | exit-correct; triforce-collected at 0x3d, the route's end. 0x7b, 0x49 visible keys; 0x6b, 0x4b keys when all are dead; 0x4b leaves E through a locked door. | Nov 9: Dungeon 3's Zol rooms; the empty entrance hall 0x7c and triforce room 0x3d join like 1:0x73/0x36 and 2:0x7d/0x0d. Not `stalfos-keese-gel`: Zols are not D1's Gels, and its trained model stays as it is. |
| `darknut` | Darknut (`RedDarknut` $0B) | (3,0x5b), (3,0x59), (3,0x69) | exit-correct. 0x5b twice (S→W skipping its bombs; W→N taking them only below 4 bombs); 0x59 three times (E→N; N→S after killing all 5 for the shutter; S→E); 0x69 twice (N→stairs, stairs→N, its bombs skipped). | Nov 9: shielded in front, only hit from the side or back. Bombs drawn 0–8 at reset (unusable here) so 0x5b's bomb rule trains both ways. |
| `boss-manhandla` | Manhandla ($3C) | (3,0x4d) | exit-correct after killing him and taking the heart container; then N into 0x3d. | Boss room per Lee's D3 route (entered through 0x4c's bombed wall); `requires_bombs` (Lee: kill him with bombs dropped in his path). |

Rooms per family: `ow-start` 4, `ow-octorok` 9 (one an alternate), `ow-zora` 9, `ow-moblin` 7, `ow-leever` 1,
`ow-tektite` 4, `ow-lynel` 2, `ow-boulder` 3, `stalfos-keese-gel` 12, `goriya` 4, `trap-room` 4, `equipment-room` 3,
`wallmaster` 1, `boss-aquamentus` 1, `d2-rope` 6, `moldorm` 1, `boss-dodongo` 1, `zol` 7, `darknut` 3,
`boss-manhandla` 1, `vire` 8, `like-like` 1, `boss-gleeok` 1, `ow-peahat` 1.

**Clips** (round 21; `docs/architecture.md` "Grid-point skip", `docs/experiments/round21.md`): a family's YAML
`clips: true` offers the `CLIP` action where a skip is available in its rooms; the default is false. On: `vire` (the
block clip in 4:0x50's diagonal and 4:0x03), `stalfos-keese-gel` (the ladder clip off 4:0x21's island), `trap-room`
(the block clip in 1:0x22) and the nine overworld families (the screen wrap; ow-lynel's 0x0a, ow-leever's 0x48 and
ow-peahat's 0x45 offer none, so for those three the switch changes nothing yet).

**Clips on and off** (round 22, Lee: "the models need to be trained with clips enabled, and also with the clip
action masked so it has to traverse as normal"; `docs/architecture.md` "The run-time clips switch"): a clip family
trains with both. `clips: true` now means `{enabled: true, off_share: 0.5}`: half the episodes start with the
run-time switch off (CLIP masked, the objective and the path search without skips, the route's plain way through a
stretch), half with it on; `off_share` sets another share. A start state the route enters on one way only fixes the
switch (4:0x30 from E, 4:0x20 from S, 4:0x21 from W off; 4:0x21 from S on); curriculum starts keep it on. Training
logs `clips/on|off/success_rate` (and episodes, health) per switch state, the early stop needs both, and the
evaluation plays both (`--clips both`, the default for a clip family; `eval.json`'s headline figure is over both,
`by_clips` per state). Families without clips play with the switch on and follow its way through a stretch.

### Dungeon 4 (scouted Nov 11, merged into `d2first` and trained Nov 12; `docs/routes/d4.md`)

`d2first` goes on from the Dungeon 3 triforce through Dungeon 4 (steps 98–122; 98–120 since round 21's shortcut;
round 22 holds both ways as the `d4-island` stretch, `docs/routes/d4.md`).
Its rooms are tagged by enemy as everywhere; the existing families train on their Dungeon 4 rooms' start states too.

| Family | Main threat | Rooms (level, room) | Notes |
|---|---|---|---|
| `vire` (new) | Vire (splits into two Red Keese) | (4,0x71), (4,0x61), (4,0x50), (4,0x30), (4,0x31), (4,0x20), (4,0x12), (4,0x03) | The dungeon's main family, with its empty hall and triforce room. 0x30 S→E (a locked door); 0x31 twice (W→E after killing all five for the shutter; then from E: with clips N through the locked door onto 0x21's island, without W); without clips 0x30 E→N (across the water on the ladder) and 0x20 S→E (round 22's plain way; round 21 had 0x20 off the route); 0x12: kill them, push the block, E through the shutter. `clips: true` (the block clip in 0x50), trained half with the switch off. |
| `like-like` (new) | Like Like, Zol, Bubble | (4,0x32) | The Breakfast Room: kill them, push the block, down the stairs; back up, W. |
| `boss-gleeok` (new) | Gleeok, two heads ($43) | (4,0x13) | Kill him, the heart container, N. |
| `ow-peahat` (new) | Peahat | (0,0x45) | The island: from the raft's pier into the dungeon. |
| `zol` | | + (4,0x40) | Five Zols in the dark; take the key (in the open). |
| `stalfos-keese-gel` | | + (4,0x51), (4,0x21) | Eight Blue Keese (take the key, in the open); five Blue Gels in the dark: with clips in from S onto the island, `CLIP N` (the ladder clip, `clips: true`); without clips in from W; then bomb the N wall (`items: {bombs: required}`, which unmasks BOMB only in this `requires_bombs` room). |
| `trap-room` | | + (4,0x11) | The rupee room: no monsters, no traps, bomb the E wall (Lee, Nov 12: it stays in this family). |
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
