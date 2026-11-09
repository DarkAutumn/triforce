# Specialist families (the d2first route)

The families cover the rooms of the route, `d2first` (`docs/routes/d2first.md`; Lee's reset, Nov 8), plus the
route's alternate 0x78 and Dungeon 1's bow-side rooms off 0x43. The enemy data comes from ROM decode, v1 data and
emulator checks. Room tags are in `triforce/game.yaml` (`specialist:`), and the same lists are in
`triforce/routes.yaml` under `families:`; the specialist configs must agree (checked at load).

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
| `ow-octorok` | Octorok (Red/Blue, slow/fast) | (0,0x67), (0,0x68), (0,0x58), (0,0x4c), (0,0x3c), (0,0x37); alt (0,0x78) | exit-correct. At 0x3c (from S) and 0x37 the correct exit is the dungeon stairs: enter-room (2,0x7d) / (1,0x73). 0x3c and 0x4c are crossed twice (before and after Dungeon 2). | Octoroks are the most numerous enemy in every one of these rooms. |
| `ow-zora` | Zora (fireball) | (0,0x59), (0,0x5a), (0,0x2d), (0,0x27), (0,0x28), (0,0x38) | exit-correct | The single Zora and its fireball are the main threat on these screens, alongside Peahats (0x27, 0x28, 0x59) and Octoroks (0x38, 0x2d, 0x5a). |
| `ow-moblin` | Moblin | (0,0x5b), (0,0x5c), (0,0x5d), (0,0x4d), (0,0x3d) | exit-correct | Moblin rooms of row 5 on the way to Dungeon 2, and 0x4d/0x3d after it. |
| `ow-tektite` | Tektite | (0,0x1a), (0,0x2c), (0,0x2c cave) | exit-correct. 0x1a: from E N to 0x0a, from N W (from N without the white sword: back N). 0x2c: bomb the rock wall, take the heart container in the take-any cave, out N. | Lee (Nov 8): Tektites' jump patterns are their own. 0x1a RedTektite×6 and a Zora; 0x2c RedTektite×4 (moved here from `ow-leever-peahat`, Lee Nov 8). 0x2c is `requires_bombs`, so the family has bombs. |
| `ow-lynel` | Lynel | (0,0x0a), (0,0x0a cave) | exit-correct: into the white-sword cave, take the sword (5 hearts), out S. Leaving S before the sword is a neutral `retreat`. | Lee (Nov 8): Lynels are very hard and the main enemy of many later rooms. |
| `ow-boulder` | Falling boulders | (0,0x19), (0,0x18), (0,0x17) | exit-correct | Lee (Nov 8): boulders are hard to predict and can't be blocked. A boulder generator ($31), its boulders ($32) and a Zora. |
| `stalfos-keese-gel` | Stalfos / Keese / Gel | (1,0x73), (1,0x72), (1,0x74), (1,0x63), (1,0x53), (1,0x52), (1,0x42), (1,0x43), (1,0x36); off-route (1,0x33) | exit-correct. 0x73 three times: W (key room 0x72), then E (key room 0x74), then N. At 0x36 the "exit" is triforce-collected, the route's end. | Was `d1-melee`. Includes the key rooms and all 4 D1 locked doors except 0x45 N. 0x36 (triforce room) and 0x73 (entrance hall) have no enemies and go to the nearest family. 0x33 (Stalfos×3, key) is north of 0x43, on the way to the bow; trained, used by a later route. |
| `goriya` | Red Goriya | (2,0x5e), (2,0x1e), (1,0x44); off-route (1,0x23) | exit-correct | Every Red Goriya room, whatever the dungeon. 0x5e (×5) has the ringleader trigger; 0x1e (×5) needs kill-all to open the N shutter into Dodongo's room; 0x44 (×3) drops the boomerang when all are dead; 0x23 (×3, key) is north of 0x33, before the bow room 0x22. |
| `wallmaster` | Wallmaster | (1,0x45) | exit-correct. 0x45 exits N through a locked door. | Wallmasters need their own skills (v1's Wallmaster shaping, design §6). |
| `boss-aquamentus` | Aquamentus | (1,0x35) | exit-correct after killing the boss and collecting the heart container; then E into 0x36. | Boss room per LevelInfoUW1 BossRoomId=$35. |
| `d2-rope` | Rope | (2,0x7d), (2,0x7e), (2,0x6e), (2,0x4e), (2,0x2e), (2,0x0d) | exit-correct; triforce-collected at 0x0d. 0x7d exits E into key room 0x7e, then 0x6e is entered from S. 0x2e requires kill-all to open the N shutter. | 0x7d is the empty entry room; 0x0d, the empty triforce room, joins like 1:0x36 joined stalfos-keese-gel. Off-route 0x6d left the family with the reset. |
| `moldorm` | Moldorm | (2,0x3e) | exit-correct (N into 0x2e). | Split out of d2-rope in round 11 (Lee, Nov 8): one Moldorm, not a boss room. |
| `boss-dodongo` | Dodongo (`Dodongo2` $32) | (2,0x0e) | boss-dead-and-heart: kill him (bomb stun, then the sword) and take the heart container. | Round 12. Boss room per LevelInfoUW2 BossRoomId=$0E; `requires_bombs`. |

Rooms per family: `ow-start` 4, `ow-octorok` 7 (one an alternate), `ow-zora` 6, `ow-moblin` 5, `ow-tektite` 3,
`ow-lynel` 2, `ow-boulder` 3, `stalfos-keese-gel` 9 (+1 off-route), `goriya` 3 (+1 off-route), `wallmaster` 1,
`boss-aquamentus` 1, `d2-rope` 6, `moldorm` 1, `boss-dodongo` 1.

Removed with the reset (Nov 8): `ow-leever-peahat` (0x29, 0x2a, 0x2b: milestone1 only; 0x2c went to `ow-tektite`),
the fairy pond 0x39 (ow-start), 0x48 (ow-zora), 0x49 (ow-octorok) and 2:0x6d (d2-rope).

## Changes from the design hypothesis (§3.1)

| Hypothesis | Decision | Evidence |
|---|---|---|
| `ow-zora` | Restored by Lee: the Zora fireball is the main threat on these screens | Zora appears in every water room on the route (OW attr A&$08), always ×1 next to 4–6 other enemies. |
| `d1-goriya` | Merged into `d1-wallmaster`; split out again as the cross-dungeon `goriya` family (Nov 3, 2026) | 0x44 was the only Goriya room in D1; the D2 Goriya rooms need the same skills |
| `d1-triforce` | Merged into `d1-melee`, now `stalfos-keese-gel` | Single non-boss room |
| `d2-*` | `d2-rope`; D2's Goriya rooms joined `goriya`; the Moldorm room its own `moldorm` | From the D2 ROM decode. D2's route has no locked doors or block pushes; it collects the 0x7e key. |
| `ow-leever-peahat` | Removed with milestone1 (Nov 8) | Its rooms 0x29–0x2b are not on d2first; 0x2c is a Tektite room |
| — | Added `ow-moblin`, `ow-lynel`, `ow-tektite`, `ow-boulder` | Moblin-only rooms; Lee (Nov 8) for the other three |

## Start states

Every route room has route-generated pre-entry states (`python -m triforce.entry_states build`, chains 0–2,
`entry_states.json`). Dungeon 1's off-route rooms toward the bow (0x33, 0x23 and the bow room 0x22) get theirs from
`entry_states path --source 1_43w_N --rooms 1:0x33,1:0x23,1:0x22 --chain N`.
