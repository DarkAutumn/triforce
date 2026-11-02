# Specialist families (milestone 1)

These families are proposed from the route in `docs/route.md`; the enemy data there comes from ROM decode plus v1 data plus RAM checks. Room tags are in `triforce/game.yaml` (`specialist:`), and the same lists are in `triforce/routes.yaml` under `families:`.

Grouping rule (design §3.1):
- Group rooms by main threat.
- Every non-boss family covers at least 2 rooms.
- Bosses get single-room families.
- A room that fits no family goes into the nearest one.

## Families

| Family | Main threat | Rooms (level, room) | Success (§3.3) | Rationale |
|---|---|---|---|---|
| `ow-start` | none | (0,0x77), (0,0x77 cave) | exit-correct. The correct exit depends on progress: no sword → cave; in the cave → S; with sword → N. | Matches the design. v1 treats the cave interior as its own `MapLocation` (`in_cave`), so this is 2 rooms. |
| `ow-octorok` | Octorok (Red/Blue, slow/fast) | (0,0x67), (0,0x68), (0,0x58), (0,0x38), (0,0x37), (0,0x2d), (0,0x4c), (0,0x3c); alt (0,0x78) | exit-correct. At 0x37 (first visit) and 0x3c, the correct exit is the dungeon stairs: enter-room (1,0x73) / (2,0x7d). | Octoroks are the most numerous enemy in every one of these rooms. 0x38 and 0x2d also have 1 Zora each, against 6 and 5 Octoroks. |
| `ow-leever-peahat` | Leever / Peahat / Tektite | (0,0x48), (0,0x28), (0,0x29), (0,0x2a), (0,0x2b), (0,0x2c) | exit-correct | Matches the design hypothesis. 0x2c has only RedTektite×4, which the design already puts in this family. 0x48 has RedLeever×4 plus a Zora. |
| `ow-moblin` | Moblin | (0,0x3d), (0,0x4d) | exit-correct | Not in the design table. These are the only 2 Moblin rooms on the canonical route, both BlueMoblin×4. |
| `d1-melee` | Stalfos / Keese / Gel | (1,0x73), (1,0x72), (1,0x74), (1,0x63), (1,0x53), (1,0x52), (1,0x42), (1,0x43), (1,0x36) | exit-correct. At 0x36 the "exit" is triforce-collected (pickup ends the level, `Z_01 TakeItem` → mode $12). | Matches the design. Includes the key rooms and all 4 D1 locked doors except 0x45 N. 0x36 is the triforce room: it has no enemies and is not a boss room, so it goes into the nearest non-boss family. Merging it into the boss family would break the single-room boss rule. |
| `d1-wallmaster` | Wallmaster (+ Red Goriya) | (1,0x44), (1,0x45) | exit-correct. 0x45 exits N through a locked door. | Only 1 Wallmaster room is on the route, so a pure family would violate ≥2. The design's `d1-goriya` also has only 1 route room (0x44; 0x23 is off-route). The two rooms are adjacent and v1 trained them together (`dungeon1-red-goriya-east`, `dungeon1-wallmaster-north-exit`, `dungeon1-late-chain`). This keeps v1's special Wallmaster shaping (design §6) in its own model. The boomerang from 0x44 is the main tool against Wallmasters. |
| `boss-aquamentus` | Aquamentus | (1,0x35) | boss-dead-and-heart. The model still owns the room until the E exit. | Boss room per LevelInfoUW1 BossRoomId=$35. |
| `d2-rope` | Rope (+ Moldorm) | (2,0x7d), (2,0x6d), (2,0x6e), (2,0x4e), (2,0x3e), (2,0x2e); alt (2,0x7e) | exit-correct. 0x2e requires kill-all to open the N shutter. | Rope is the main threat in 5 of 6 rooms. 0x7d is the empty entry room. 0x3e (Moldorm×1) is not a boss room (BossRoomId=$0E) and would be a one-room family, so it goes into the family of its neighbours 0x4e and 0x2e. |
| `d2-goriya` | Red Goriya | (2,0x5e), (2,0x1e) | exit-correct. At 0x1e the correct exit is enter-room:(2,0x0e), which is the milestone terminal. | Both rooms have RedGoriya×5. 0x5e has the ringleader trigger; 0x1e needs kill-all to open the N shutter. |
| `boss-dodongo` | Dodongo (`Dodongo2` $32) | (2,0x0e) | out of scope for milestone 1. The route ends on entry, and the predecessor's success is enter-room:(2,0x0e). | Tagged only so the runner never hits an untagged room (design §9). Not trained in milestone 1. Boss room per LevelInfoUW2 BossRoomId=$0E. |

Every non-boss family has ≥2 rooms: `ow-start` 2, `ow-octorok` 8(+1), `ow-leever-peahat` 6, `ow-moblin` 2, `d1-melee` 9, `d1-wallmaster` 2, `d2-rope` 6(+1), `d2-goriya` 2.

## Changes from the design hypothesis (§3.1)

| Hypothesis | Decision | Evidence |
|---|---|---|
| `ow-zora` | Dropped | Zora appears in 4 route rooms (0x48, 0x38, 0x28, 0x2d; OW attr A&$08), always ×1 next to 4–6 other enemies. No route room has Zora as its main threat, so Zora is absorbed into the family that owns each room (design §13 question). |
| `d1-goriya` | Merged into `d1-wallmaster` | 0x44 is the only Goriya room on the route |
| `d1-triforce` | Merged into `d1-melee` | Single non-boss room |
| `d2-*` | `d2-rope`, `d2-goriya` | From the D2 ROM decode. D2 needs no keys, block pushes or bombs on the canonical route. |
| — | Added `ow-moblin` | 0x3d and 0x4d are Moblin-only |
| — | Added `boss-dodongo` | Terminal tag only |

## Route rooms lacking savestates

| Room | Family | Note |
|---|---|---|
| (0, 0x77 cave) | ow-start | No mode $0B (cave) state exists |
| (2, 0x6d) | d2-rope | — |
| (2, 0x6e) | d2-rope | — |
| (2, 0x4e) | d2-rope | — |
| (2, 0x3e) | d2-rope | — |
| (2, 0x2e) | d2-rope | — |
| (2, 0x7e) alt | d2-rope | — |
| (2, 0x5e) | d2-goriya | — |
| (2, 0x1e) | d2-goriya | — |
| (2, 0x0e) | boss-dodongo | Terminal |

Entry sides without a matching state:
- (0, 0x77) at the cave mouth with a sword
- (0, 0x4d) from N

D2 has only `2_7dn`, plus `2_3cn` (mid-entry). The whole `d2-goriya` family and all but the entry room of `d2-rope` need seed states before training (design §7.2).
