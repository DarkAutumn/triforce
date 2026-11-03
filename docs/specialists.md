# Specialist families (milestone 1)

These families are proposed from the route in `docs/route.md`; the enemy data there comes from ROM decode plus v1 data plus RAM checks. Room tags are in `triforce/game.yaml` (`specialist:`), and the same lists are in `triforce/routes.yaml` under `families:`.

Grouping rule (design §3.1, revised by Lee on Nov 3, 2026):
- Group rooms by their enemies, not by dungeon: a room full of Red Goriyas needs the same skills whichever
  dungeon it is in, only the geometry differs.
- A family normally covers at least 2 rooms; an enemy that needs its own skills (Wallmasters) gets its own family
  even with 1 room so far.
- Bosses get single-room families.
- A room that fits no family goes into the nearest one.

## Families

| Family | Main threat | Rooms (level, room) | Success (§3.3) | Rationale |
|---|---|---|---|---|
| `ow-start` | none | (0,0x77), (0,0x77 cave); fairy route (0,0x39) | exit-correct. The correct exit depends on progress: no sword → cave; in the cave → S; with sword → N. 0x39: no exit is correct until the fairy pond has filled the hearts, then S. | Matches the design. v1 treats the cave interior as its own `MapLocation` (`in_cave`), so this is 2 rooms. The pond room has no enemies: walking to a spot is this family's skill. |
| `ow-octorok` | Octorok (Red/Blue, slow/fast) | (0,0x67), (0,0x68), (0,0x58), (0,0x37), (0,0x4c), (0,0x3c); alt (0,0x78); fairy route (0,0x49) | exit-correct. At 0x37 (first visit) and 0x3c, the correct exit is the dungeon stairs: enter-room (1,0x73) / (2,0x7d). | Octoroks are the most numerous enemy in every one of these rooms. |
| `ow-zora` | Zora (fireball) | (0,0x48), (0,0x38), (0,0x28), (0,0x2d); fairy route (0,0x59), (0,0x5a) | exit-correct | The single Zora and its fireball are the main threat on these screens, alongside Peahat×6 (0x28), Peahat×4 (0x59), RedLeever×4 (0x48) and Octoroks (0x38, 0x2d, 0x5a). 0x38 is visited twice (exit W toward D1, exit N toward D2; S on the fairy route). |
| `ow-leever-peahat` | Leever / Peahat / Tektite | (0,0x29), (0,0x2a), (0,0x2b), (0,0x2c) | exit-correct | Matches the design hypothesis. 0x2c has only RedTektite×4, which the design already puts in this family. |
| `ow-moblin` | Moblin | (0,0x3d), (0,0x4d); fairy route (0,0x5b), (0,0x5c), (0,0x5d) | exit-correct | Not in the design table. 0x3d and 0x4d are the only 2 Moblin rooms on milestone1, both BlueMoblin×4; the fairy route adds three more Moblin rooms. |
| `stalfos-keese-gel` | Stalfos / Keese / Gel | (1,0x73), (1,0x72), (1,0x74), (1,0x63), (1,0x53), (1,0x52), (1,0x42), (1,0x43), (1,0x36); off-route (1,0x33) | exit-correct. At 0x36 the "exit" is triforce-collected (pickup ends the level, `Z_01 TakeItem` → mode $12). | Was `d1-melee`. Includes the key rooms and all 4 D1 locked doors except 0x45 N. 0x36 (triforce room) and 0x73 (entrance hall) have no enemies and go to the nearest family. 0x33 (Stalfos×3, key) is north of the Gel map room, on the way to the bow (`docs/route.md`); trained now, used by a later route. |
| `goriya` | Red Goriya | (1,0x44), (2,0x5e), (2,0x1e); off-route (1,0x23) | exit-correct. At 0x1e the correct exit is enter-room:(2,0x0e), the milestone terminal. | Every Red Goriya room, whatever the dungeon (was split between `d1-wallmaster` and `d2-goriya`). 0x44 (×3) drops the boomerang when all are dead; 0x5e (×5) has the ringleader trigger; 0x1e (×5) needs kill-all to open the N shutter; 0x23 (×3, key) is north of 0x33, before the bow room 0x22. |
| `wallmaster` | Wallmaster | (1,0x45) | exit-correct. 0x45 exits N through a locked door. | Wallmasters need their own skills (v1's Wallmaster shaping, design §6); only 1 Wallmaster room is on the route so far. |
| `boss-aquamentus` | Aquamentus | (1,0x35) | boss-dead-and-heart. The model still owns the room until the E exit. | Boss room per LevelInfoUW1 BossRoomId=$35. |
| `d2-rope` | Rope (+ Moldorm) | (2,0x7d), (2,0x6d), (2,0x6e), (2,0x4e), (2,0x3e), (2,0x2e); alt (2,0x7e) | exit-correct. 0x2e requires kill-all to open the N shutter. | Rope is the main threat in 5 of 6 rooms. 0x7d is the empty entry room. 0x3e (Moldorm×1) is not a boss room (BossRoomId=$0E) and would be a one-room family, so it goes into the family of its neighbours 0x4e and 0x2e. |
| `boss-dodongo` | Dodongo (`Dodongo2` $32) | (2,0x0e) | out of scope for milestone 1. The route ends on entry, and the predecessor's success is enter-room:(2,0x0e). | Tagged only so the runner never hits an untagged room (design §9). Not trained in milestone 1. Boss room per LevelInfoUW2 BossRoomId=$0E. |

Rooms per family: `ow-start` 2(+1 fairy route), `ow-octorok` 6(+1, +1 fairy route), `ow-leever-peahat` 4,
`ow-zora` 4(+2 fairy route), `ow-moblin` 2(+3 fairy route), `stalfos-keese-gel` 9(+1 off-route),
`goriya` 3(+1 off-route), `wallmaster` 1, `d2-rope` 6(+1).

## Changes from the design hypothesis (§3.1)

| Hypothesis | Decision | Evidence |
|---|---|---|
| `ow-zora` | Restored by Lee: the Zora fireball is the main threat on these screens | Zora appears in 4 route rooms (0x48, 0x38, 0x28, 0x2d; OW attr A&$08), always ×1 next to 4–6 other enemies. |
| `d1-goriya` | Merged into `d1-wallmaster`; split out again as the cross-dungeon `goriya` family (Nov 3, 2026) | 0x44 was the only Goriya room on the route; the D2 Goriya rooms need the same skills |
| `d1-triforce` | Merged into `d1-melee`, now `stalfos-keese-gel` | Single non-boss room |
| `d2-*` | `d2-rope`; D2's Goriya rooms joined `goriya` | From the D2 ROM decode. D2 needs no keys, block pushes or bombs on the canonical route. |
| — | Added `ow-moblin` | 0x3d and 0x4d are Moblin-only |
| — | Added `boss-dodongo` | Terminal tag only |

## Start states

Every route room has route-generated pre-entry states (`triforce/entry_states.py build`, `entry_states.json`) except
the Dodongo room (2,0x0e), which milestone 1 only enters. Dungeon 1's off-route rooms toward the bow (0x33, 0x23 and
the bow room 0x22) get theirs from `entry_states path` (round 6).
