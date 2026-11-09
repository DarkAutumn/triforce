# Milestone 1 route (historical)

**Superseded (Nov 8):** the route is now `d2first` (`docs/routes/d2first.md`); `milestone1` and
`milestone1-fairy` were removed from `triforce/routes.yaml`. The room data below is still useful background.

Start of game → wooden sword → Dungeon 1 (Aquamentus + triforce) → overworld → Dungeon 2 → entering the D2 boss room (design §1).
The machine-readable version is `triforce/routes.yaml` (runtime data for the runner and objectives). Specialist tags are in `triforce/game.yaml`.

## Sources and how they were used

| Tag | Source | What it gave |
|---|---|---|
| ROM | `roms/zelda.nes` (or `$TRIFORCE_ROM`), decoded at the `bins.xml` offsets (+16-byte header, `build.ps1`): `LevelBlockOW` 99328, `LevelBlockUW1Q1` 100096, `LevelInfoOW/UW1/UW2` 103168/103420/103672, `ObjLists` 83574 | Per-room attrs A–F (6 × 128 bytes; layout `$687E..$6AFE` in `Variables.inc`) |
| disasm | `zelda1-disassembly/src` | Decoders: doors = `Z_05 FindDoorTypeByDoorBit` (A: N bits 7-5, S bits 4-2; B: W bits 7-5, E bits 4-2). Door type table is in `Z_05 @DoneClosedPF` (0 open, 1 wall, 2/3 false wall, 4 bombable, 5/6 key, 7 shutter). Enemies: C&$3F \| (D bit7→$40) = template/list, count = `LevelInfo_FoeCounts[C>>6]`, forced 1 for ids $32–$61 (`Z_05` ~1697-1730). Lists ≥$62 use `ObjListAddrs`. Room item: E&$1F, 3 = none (`Z_05 CreateRoomObjects`). Item IDs map through `Z_01 ItemIdToSlot` → `Items $657+slot` (`Variables.inc`): $19 key, $17 map, $16 compass, $1D boomerang, $1A heart container, $1B triforce, $00 bombs. Secret trigger: F&7 (`Z_05 CheckSecretTrigger_JumpTable`): 1 all-dead→shutters, 2 ringleader, 4 block→door, 5 block→stairs, 7 all-dead→item+shutters. Push block: D&$40. OW Zora: A&$08 (`Z_04` ~1760). OW cave/level: B&$FC; <$40 means level B>>2 (`Z_05 HandleWarpOW`). |
| v1 | `triforce/game.yaml` (v1), `triforce.yaml` scenarios, `end_conditions.py`, `objectives.py`, `docs/savestates.yml` | Room graph, overworld exits, D1 enemy/treasure data. The ROM decode reproduces every v1 D1 enemy and treasure entry, which validates the decoder. |
| RAM | Every `*.state` in the repo at the time (old stable-retro savestates, since regenerated as `TFZSTATE`): `$10` level, `$EB` room, `$12` mode, `$70/$84` Link x/y, `$34F+i` ObjType, `$66E` keys, `$658` bombs | Confirms which room each savestate is in, and the enemy types present |

Level facts from LevelInfo (ROM): D1 start room $73, boss room $35, triforce room $36, cellar $7F (bow, reached from $22). D2 start room $7D, boss room $0E, triforce room $0D, no cellars. OW level entrances (attr B): D1 = OW $37, D2 = OW $3C.

## Route overview

| Leg | Rooms |
|---|---|
| OW start → sword | 0x77 → cave 0x77 (wood sword) → 0x77 |
| OW → D1 | 0x77 N 0x67 E 0x68 N 0x58 N 0x48 N 0x38 W 0x37 → stairs |
| D1 | 0x73 W 0x72 E 0x73 E 0x74 W 0x73 N(key) 0x63 N 0x53 W 0x52 N(key) 0x42 E 0x43 E(key) 0x44 E 0x45 N(key) 0x35 E 0x36 → triforce |
| D1 → OW | Picking up the triforce sets GameMode $12. `EndGameMode12` then sets CurLevel=0 and Link reappears at OW 0x37 (`Z_01 TakeItem`, `Z_05 EndGameMode12`). There is no walk out of D1. `0_37c` (mode 4) is this state, and v1's `dungeon1-dungeon2` scenario starts from it. |
| OW → D2 | 0x37 E 0x38 N 0x28 E 0x29 E 0x2a E 0x2b E 0x2c E 0x2d S 0x3d S 0x4d W 0x4c N 0x3c → stairs |
| OW → D2, `milestone1-fairy` | 0x37 E 0x38 S 0x48 S 0x58 E 0x59 N 0x49 N 0x39 (fairy pond: hearts refilled) S 0x49 S 0x59 E 0x5a E 0x5b E 0x5c E 0x5d N 0x4d W 0x4c N 0x3c → stairs |
| D2 | 0x7d E 0x7e N 0x6e N 0x5e N 0x4e N 0x3e N 0x2e N 0x1e N 0x0e (milestone ends on entry) |

**How the route was chosen**
- **Start → D1.** BFS over v1's `game.yaml` gives two 6-room ties: via 0x67 or via 0x78. v1's `LeftOverworld1Area` allows both (`{77,78,67,68,58,48,38,37}`). The canonical route uses 0x67 and 0x78 is tagged as an alternate.
- **D1 → D2.** BFS gives two 11-room ties:
  - North: 0x28…0x2d, 0x3d, 0x4d. This is canonical: it matches the design's `ow-leever-peahat` family and stays inside v1's `LeftOverworld2Area` box.
  - South: 0x48, 0x58, 0x59, 0x5a, 0x5b, 0x5c, 0x5d, 0x4d. Tagged in round 6 as part of `milestone1-fairy`
    (Lee): the south road plus a 4-room detour 0x59 N 0x49 N 0x39 to the fairy pond, which refills every heart
    (`Z_04 UpdatePondFairy`: Link at the pond edge, Y=$AD, $70 ≤ X < $81, is halted until the hearts are full).
    `routes.yaml` defines it as milestone1 with the steps from 0x38 (from W) up to 0x4c (from E) replaced.
- **D1 path.** This is v1's "locked-door branch" (`dungeon1-late-chain`: 0x43→0x44→0x45→0x35, `Dungeon1DidntGetKey`). It needs no bombs, and Link starts with 0 bombs (RAM `start`: `$658`=0).
- **D2 path.** Two 8-room ties from 0x7d: via 0x6d or via 0x7e. The canonical route uses the eastern key room 0x7e, then enters 0x6e from S. 0x6d remains in the Rope enemy family, but is no longer a route alternate.

**Corrections to v1 `game.yaml`** (applied in v2 `game.yaml`):
- OW 0x3c cave target: 2:0x77 in_cave → **2:0x7d**. Source: LevelInfoUW2 StartRoomId; RAM `2_7dn` is in level 2, room $7D.
- OW 0x3c enemy: BlueSlowOctorok → **RedSlowOctorok**. Source: ROM template $0A and RAM ObjType $0A in `0_3cs`.
- OW 0x3c: added an S exit to 0x4c. v1 already had 0x4c N→0x3c, and `0_3cs` has Link entering at the S edge.

## Step table

Entry is the side of the room Link enters from. Doors list only non-wall sides (ROM). Savestates are RAM-verified, with Link's RAM side in parentheses: N/S/E/W edge, C = interior, `modeN` = mid-transition.

> **Savestate naming is unreliable.** The filename letter does not consistently match the RAM side, and `1_83e`/`1_83s` are actually room 0x53. Use RAM, not the name.

| # | (level, room) | Entry | Exit | Enemies [source] | Treasure | Doors | Puzzle / gate | Needs | Savestates |
|---|---|---|---|---|---|---|---|---|---|
| 1 | (0, 0x77) | start | cave | none [ROM, v1] | — | — | Enter sword cave. OW attr B = $42 → cave index $10 (`Z_05 HandleWarpOW`). | — | `start`(C), `0_77c`(C) (no sword); `0_77e`(E), `0_77n`(N), `0_77w`(W) (with sword) |
| 2 | (0, 0x77, cave) | S | S | old man NPC only (cave dweller $6A, `Z_05` @InCave) | wood-sword [ROM: cave 0 wares $3F,$01,$3F; v1] | — | Take the sword, then walk S out of the cave. | — | **none** |
| 3 | (0, 0x77) | cave | N | none [ROM, v1] | — | — | — | sword | `0_77n`(N) etc. (with sword; none at the cave mouth) |
| 4 | (0, 0x67) | S | E | BlueOctorok×4 [ROM, v1, RAM `test_ow_full_health`] | — | — | — | — | `0_67s`(mode7 from S), `0_67c`(N), `0_67e`(E), `0_67w`(W) |
| 5 | (0, 0x68) | W | N | BlueOctorok×4 [ROM, v1, RAM `0_68e`] | — | — | — | — | `0_68w`(W), `0_68e`, `0_68n`, `0_68s` |
| 6 | (0, 0x58) | S | N | RedOctorok×2, BlueOctorok×2 [ROM, v1, RAM `0_58s`] | — | — | — | — | `0_58s`(S), `0_58e`, `0_58n`, `0_58w` |
| 7 | (0, 0x48) | S | N | RedLeever×4, Zora×1 [ROM, v1, RAM `0_48n`] | — | — | — | — | `0_48s`(S), `0_48n`, `0_48w` |
| 8 | (0, 0x38) | S | W | RedOctorok×5, BlueSlowOctorok×1, Zora×1 [ROM, v1, RAM `0_38s`] | — | — | — | — | `0_38s`(S), `0_38n`, `0_38w` |
| 9 | (0, 0x37) | E | cave | RedOctorok×1 [ROM, v1, RAM `0_37e`] | — | — | Enter the D1 stairs (attr B level 1). The doorway is tile $F3 in the `0_37w` tile dump, so no item is needed. | — | `0_37e`(E), `0_37s`(C), `0_37w`(W); `1_37n` = mid-entry (level 1, mode $10) |
| 10 | (1, 0x73) | S | W | none [ROM, v1] | — | N:key S:open W:open E:open | N is locked; come back with a key. | — | `1_73n`(S), `1_73c`(S), `1_73s`(mode3) |
| 11 | (1, 0x72) | E | E | BlueKeese×3 [ROM, v1] | key (appears when all dead, F=7) | E:open | Kill all to get the key. | — | `1_72e`(mode7 from E) |
| 12 | (1, 0x73) | W | E | none | — | as #10 | — | — | `1_73w`(mode7 from W) |
| 13 | (1, 0x74) | W | W | Stalfos×5 [ROM, v1, RAM `1_74w`] | key (F=0, visible). A Stalfos in object slot 1 carries the room item (`Z_07 MoveAndDrawRoomItem`), so kill the carrier. | W:open | Kill the Stalfos that carries the key. | — | `1_74w`(W) |
| 14 | (1, 0x73) | E | N | none | — | as #10 | **Locked door N.** Uses key 1. | key | `1_73e`(mode4 from E) |
| 15 | (1, 0x63) | S | N | Stalfos×3 [ROM, v1, RAM `1_63n`] | — | N:open S:key | — | — | `1_63s`(S), `1_63n`(N) |
| 16 | (1, 0x53) | S | W | Stalfos×5 [ROM, v1, RAM `1_53s`] | key (appears when all dead) | N:bombable S:open W:open E:open | Kill all to get the key. N bomb wall → 0x43 is a shortcut that needs bombs (not used). | — | `1_53s`(S), `1_83s`(S), `1_53e`, `1_83e`, `1_53n`(C), `1_53w` |
| 17 | (1, 0x52) | E | N | BlueKeese×6 [ROM, v1, RAM `1_52e`] | — | N:key E:shutter | Enter through E shutter. Kill-all reopens it (F=1); not needed. **Locked door N.** Uses key 2. | key | `1_52e`(C), `1_52n`(N) |
| 18 | (1, 0x42) | S | E | BlueGel×3 [ROM, v1, RAM `1_42e`] | — | S:key W:shutter E:open | A push block opens the W shutter to old-man room 0x41 (F=4, D&$40). Off-route, not needed. | — | `1_42s`(S), `1_42e`, `1_42w`(C) |
| 19 | (1, 0x43) | W | E | BlueGel×5 [ROM, v1, RAM `1_43e`] | map (visible, optional) | N:open S:bombable W:open E:key | **Locked door E.** Uses key 3. | key | `1_43w`(W), `1_43e`, `1_43n`, `1_43s`(C) |
| 20 | (1, 0x44) | W | E | RedGoriya×3 [ROM, v1, RAM `1_44c`] | boomerang (appears when all dead; optional, only boomerang on route) | S:bombable W:key E:open | — | — | `1_44w`(W), `1_44c`(C), `1_44e`, `1_44s`(C) |
| 21 | (1, 0x45) | W | N | Wallmaster×8 [ROM, v1, RAM `1_45n`] | key (F=0, visible) | N:key W:open | Get the key. **Locked door N.** Uses key 4. v1 needed special Wallmaster shaping (design §6). | key | `1_45w`(W), `1_45n`(N) |
| 22 | (1, 0x35) | S | E | Aquamentus×1 [ROM, v1, RAM `1_35s`] | heart-container (appears when boss dead) | S:key E:shutter | Killing the boss opens the E shutter (F=7 → `TriggerShutters`). | — | `1_35s`(S), `1_35e`(C) |
| 23 | (1, 0x36) | W | triforce | none | triforce (visible) | W:open | Pickup ends the level and returns Link to OW 0x37. | — | `1_36w`(W) |
| 24 | (0, 0x37) | cave | E | RedOctorok×1 | — | — | Post-D1 spawn at the dungeon doorway. | — | `0_37c`(mode4, exact post-triforce state) |
| 25 | (0, 0x38) | W | N | as #8 | — | — | — | — | `0_38w`(W) |
| 26 | (0, 0x28) | S | E | Peahat×6, Zora×1 [ROM, v1, RAM `0_28e`] | — | — | — | — | `0_28s`(S), `0_28e`, `0_28w` |
| 27 | (0, 0x29) | W | E | RedLeever×4 [ROM, v1, RAM `0_29e`] | — | — | — | — | `0_29w`(W), `0_29e` |
| 28 | (0, 0x2a) | W | E | BlueLeever×6 [ROM, v1, RAM `0_2ae`] | — | — | — | — | `0_2aw`(W), `0_2ae`, `0_2as` |
| 29 | (0, 0x2b) | W | E | BlueLeever×2, Peahat×3, RedLeever×1 [ROM, v1, RAM `0_2be`] | — | — | — | — | `0_2bw`(W), `0_2be`, `0_2bs` |
| 30 | (0, 0x2c) | W | E | RedTektite×4 [ROM, v1, RAM `0_2ce`] | — | — | — | — | `0_2cw`(W), `0_2ce`, `0_2cn` |
| 31 | (0, 0x2d) | W | S | RedOctorok×3, BlueSlowOctorok×2, Zora×1 [ROM, v1, RAM `0_2de`] | — | — | — | — | `0_2dw`(W), `0_2de`, `0_2dn`, `0_2ds` |
| 32 | (0, 0x3d) | N | S | BlueMoblin×4 [ROM, v1, RAM `0_3dc`] | — | — | — | — | `0_3dn`(N), `0_3ds`, `0_3dc`(C) |
| 33 | (0, 0x4d) | N | W | BlueMoblin×4 [ROM, v1, RAM `0_4de`] | — | — | — | — | `0_4ds`(S), `0_4de`, `0_4dn`(E), `0_4dw`. **No N-side state.** |
| 34 | (0, 0x4c) | E | N | RedOctorok×3, BlueSlowOctorok×2 [ROM, v1, RAM `0_4ce`] | — | — | — | — | `0_4ce`(E), `0_4cn`(N) |
| 35 | (0, 0x3c) | S | cave | RedSlowOctorok×1 [ROM, RAM `0_3cs`; v1 had Blue] | — | — | Enter the D2 stairs (attr B level 2). The doorway is tile $F3 in the `0_3cs` tile dump, so no item is needed. | — | `0_3cs`(S); `2_3cn` = mid-entry (level 2, mode $10) |
| 36 | (2, 0x7d) | S | E | none [ROM, RAM `2_7dn`] | — | N:open S:open E:open | — | — | `2_7dn`(S) |
| 37 | (2, 0x7e) | W | N | Rope×5 [ROM] | key (appears when all dead) | N:open W:open | Native route-state generation collects the key before leaving. | — | `2_7ew_[0-2]` |
| 38 | (2, 0x6e) | S | N | Rope×3 [ROM] | — | N:open S:open W:open E:key | E key door → compass room 0x6f; not needed. | — | `2_6es_[0-2]` |
| 39 | (2, 0x5e) | S | N | RedGoriya×5 [ROM] | — | N:open S:open E:key | Ringleader (F=2): killing the slot-1 Goriya kills the rest (`Z_05 CheckSecretTriggerRingleader`). N is open. | — | **none** |
| 40 | (2, 0x4e) | S | N | Rope×5 [ROM] | key (appears when all dead; optional) | N:open S:shutter E:open | Enter from 0x5e, whose N door is open. The move uses the current room's door type (`Z_07 CalculateNextRoom`). The D1 analogue: entering 0x52 through its shutter is proven by `1_52e`. | — | **none** |
| 41 | (2, 0x3e) | S | N | Moldorm×1 [ROM] | key (appears when all dead; optional) | N:open S:open E:open | — | — | **none** |
| 42 | (2, 0x2e) | S | N | Rope×8 [ROM] | — | N:shutter S:open E:key | **Kill all 8 to open the N shutter** (F=1). | — | **none** |
| 43 | (2, 0x1e) | S | N | RedGoriya×5 [ROM] | bombs (appears when all dead) | N:shutter S:shutter E:bombable | **Kill all 5 to open the N shutter** (F=7). E bomb wall → old man 0x1f; not needed. | — | **none** |
| 44 | (2, 0x0e) | S | terminal | Dodongo2×1 [ROM] (boss, out of scope) | heart-container | S:open W:shutter | The milestone ends when Link enters this room. | — | **none** |

Alternate (equal length, tagged): (0, 0x78) BlueOctorok×4 [ROM, v1], states `0_78w`/`0_78n`/`0_78e`. D2 has no route alternate; the old western 0x6d branch remains a Rope-family training room.

## Puzzles on the route

| Kind | Where | Required? |
|---|---|---|
| Locked doors | D1: 0x73 N, 0x52 N, 0x43 E, 0x45 N | Yes, 4 keys |
| Kill-all shutters | D1 0x35 E (boss). D2 0x2e N, 0x1e N | Yes |
| Shutters entered from outside | D1 0x52 E; D2 0x4e S | Pass-through only |
| Push block | D1 0x42: opens W shutter to 0x41 | No (off-route) |
| Push block | D1 0x22: stairs to bow cellar 0x7F | No (off-route) |
| Bomb walls | D1 0x53 N/0x43 S, 0x54 N/0x44 S. D2 0x1e E, 0x2f S, 0x3f N/S, 0x4f N/S, 0x5f N/S, 0x6f N | No. The canonical route needs no bombs. |

## Key economy

**D1** (keys at entry: 0, RAM `1_73n` `$66E`=0)

| Step | Room | Event | Keys held |
|---|---|---|---|
| 11 | 0x72 | +key (kill 3 Keese) | 1 |
| 13 | 0x74 | +key (kill the carrier Stalfos) | 2 |
| 14 | 0x73 | −key (N door) | 1 |
| 16 | 0x53 | +key (kill 5 Stalfos) | 2 |
| 17 | 0x52 | −key (N door) | 1 |
| 19 | 0x43 | −key (E door) | 0 |
| 21 | 0x45 | +key (visible) | 1 |
| 21 | 0x45 | −key (N door) | 0 |

The route collects 4 keys and opens 4 locked doors, leaving **zero spare keys**.

- **Optional spare key:** 0x33 (Stalfos×3, key visible) is 1 room N of 0x43 through an open door.
- **Key traps:** 0x33 N and 0x23 W/S are locked doors off the route. Opening one wastes a key.
- **Key door pairs:** each locked door also appears as a key door in the adjacent room: 0x63 S, 0x42 S, 0x44 W, 0x35 S.

**D2** (keys at entry: 0, RAM `2_7dn` `$66E`=0)
- Locked doors on the canonical route: **none**. The key doors 0x2e E, 0x5e E and 0x6e E are all off-route.
- Keys collectible on route: 0x7e, 0x4e and 0x3e. No route door consumes them. Native route-state generation requires each pickup and preserves the carried-key balance; gameplay can cross these rooms without collecting their keys.
- Bombs are not required, but `2_7dn` holds 5.

## Historical savestate gaps (original route survey)

This list records the initial survey, not the current generated-state inventory. D2 now uses the eastern
0x7e branch and S-entry 0x6e; native generation rebuilds every downstream pre-entry state through 0x0e.

- (0, 0x77, cave)
- (2, 0x6e)
- (2, 0x5e)
- (2, 0x4e)
- (2, 0x3e)
- (2, 0x2e)
- (2, 0x1e)
- (2, 0x0e)

Missing entry sides:
- 0x77 at the cave mouth with a sword
- 0x4d from N
