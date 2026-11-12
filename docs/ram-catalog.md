# RAM feature catalog (Phase 0, design §4.1)

Every RAM/SRAM variable that v1 reads, plus the disassembly variables needed to decode **room state**.
One row per candidate observation feature.

**Citation keys**
- `asm:<Name>` → a label or RAM symbol in `zelda-asm/` (the annotated disassembly in this repository); resolve it with `python zelda-asm/tools/asm.py where <Name>`. RAM symbol names are defined in `zelda-asm/src/*.inc` and listed by address in `zelda-asm/index/ram.md`.
- `v1 file` → `triforce/<file>` on this repository's `v2-stable-retro` branch, the Python project this one replaced (`gd.txt` = `zelda_game_data.txt`, `data.json` = v1's retro integration variable table; this repo's equivalent is `crates/zelda/src/vars.rs`, `ram.md` = `docs/ram.md`).
- `ss <name>` → value observed by loading savestate `<name>.state` in this repo, stepping no-op until `GameMode=5`, and reading RAM. Historical: these observations were made in Phase 0 on the old stable-retro 1.0.1 savestates (since replaced by `TFZSTATE` files in `triforce/states/`, read with `triforce._native.Game`).

**Address space.** `Game.ram()` (`triforce._native`) returns 10240 bytes: index `0x000–0x7FF` = CPU RAM `$0000–$07FF`; index `0x800–0x27FF` = cartridge SRAM `$6000–$7FFF`. So **SRAM address `A` is at `ram()[A - 0x5800]`** (v1's `tile_layout` at index `0xD30` is `PlayAreaTiles $6530`). Verified: attribute bytes read this way decode to door layouts that match RAM door state in ss `1_35e`, `1_42e` (see §R.9).

**Object slot indices** (fiskbit notes in v1 `ram.md`, confirmed by asm and ss): `0` Link · `1–$B` monsters/projectiles/dropped items (`asm:Obj_FindEmptyMonsterSlot` searches `$B..1`) · `$B` is also the *tile-object* slot (push block, OW rock wall/tree; `asm:PushBlock_FindAndCreate`, `asm:Secret_Trigger5BlockStairs` "LDX #$0B ; Tile object slot") · `$C` see open questions · `$D` sword · `$E` beam/rod shot · `$F` boomerang/bait · `$10,$11` bomb/fire · `$12` arrow/rod · `$13` room item ("treasure"). Table address for slot `i` = base + `i`.
> The upstream disassembly's `; TRIFORCE:` slot comment (in what is now `zelda-asm/src/Variables.inc`, corrected there; see `asm:ObjState`) said sword = slot `$0B`, beam = `$0E`, and was wrong: `ObjState+$0D = $B9` is the sword slot (v1 `sword_animation`), and `$0B` is the tile-object slot.

Encoding vocabulary: `categorical(n)` · `ordinal/scale (÷max)` · `bitfield` · `timer (÷max)` · `position (÷max)`. Positions: x ∈ 0–255 (÷255); y play area is 64–239 (rows start at `$40`, `asm:Tile_GetAheadOfObj`), normalize `(y−64)/176`.

---

## 1. Game mode & location

| Address(es) | Meaning | Encoding | Valid when | Include? | Source |
|---|---|---|---|---|---|
| `$10` CurLevel | 0 = overworld, 1–9 = dungeon | categorical(10) | both | yes — room identity key | asm:CurLevel; v1 gd.txt `level` |
| `$EB` RoomId | room 0–$7F; high nibble = row, low = column (`asm:Unused_SplitRoomId`). Dungeons 1–6 share one 128-room grid (ss `1_*`, `2_*` both use flags block `$6FF`) | categorical(128) → with level, room embedding | both | yes — room identity | asm:Unused_SplitRoomId; v1 gd.txt `location` |
| `$12` GameMode | 0 demo, 1 menu, 2 load level, 3 unfurl, 4 enter room, **5 play**, 6 leave room, 7 scroll, 8 continue/save/retry menu (`asm:Mode8Continue_Init`), **9 cellar**, **$A leave cellar**, **$B cave**, **$C shortcut cave**, $D save, $E/$F register/elim, $10 stairs, $11 death, $12 end level (triforce), $13 win. Modes 5,9,A,B,C all run `Mode5Play_Update` | categorical(20) | both | no (feature) — use for frame skip / masks; add `in_cave`=(mode==$B), `in_cellar`=(mode∈{9,$A}) bits | asm:Mode_Update_JumpTable, asm:Mode_Init_JumpTable; v1 frame_skip_wrapper.py MODE_* |
| `$13` GameSubmode | sub-step within mode | categorical | both | no — transition plumbing | asm:GameSubmode |
| `$EC` NextRoomId | room being scrolled to | categorical(128) | both, only in mode 6/7 | no | asm:NextRoomId |
| `$526` LevelEntranceRoomId | OW room of the dungeon entrance last entered ($FF none); written only when a level entrance is entered, never by a cave or shortcut cave; mode 3 starts the OW in this room and resets it to $FF | categorical(128) | dungeon | no — runner bookkeeping | asm:LevelEntranceRoomId, asm:Warp_HandleOW@LoadLevel, asm:Mode3Unfurl_Init_Sub1 |
| `$527` CellarSourceRoomId | UW room the cellar was entered from | categorical(128) | cellar | no | asm:CellarSourceRoomId, asm:Mode9Cellar_Init_Sub8_EnterCellar |
| `$5A` UndergroundExitType | nonzero while stepping out of cave/dungeon/cellar; blocks re-entering (`Warp_CheckUnderLink`) | bitfield/flag | OW, transitions | maybe — v1 uses to mask cave re-entry | asm:Warp_CheckUnderLink; v1 data.json `underground_exit_type` |
| `$620` CurRoomHistoryIndex, `$621–$626` RoomHistory | ring of last 6 rooms visited (searched `LDY #$05`); used to restore partial kill counts | categorical(128)×6 | both | no — use explicit room flags instead | asm:RoomHistory_ModifyObjCountOW, asm:RoomHistory_ModifyObjCountUW; v1 ram.md |
| `$6BB1` LevelInfo_LevelNumber, `$6BAD` LevelInfo_StartRoomId, `$6BAE` LevelInfo_TriforceRoomId, `$6BBC` LevelInfo_BossRoomId, `$6BB2..` LevelInfo_CellarRoomIdArray (SRAM) | current level's static info (ss `1_*`: start $73, triforce $36, boss $35; ss `2_7dn`: start $7D, triforce $0D, boss $0E) | categorical | dungeon (OW block has start $77) | no — static; use for `is_boss_room`, `is_triforce_room` derived bits (maybe) | asm:LevelInfo_LevelNumber, asm:LevelInfo_StartRoomId, asm:LevelInfo_TriforceRoomId, asm:LevelInfo_BossRoomId, asm:LevelInfo_CellarRoomIdArray; ss |
| `$6BAF/$6BB0` LevelInfo_WorldFlagsAddr (SRAM) | pointer to this level's 128-byte room-flag block: OW `$067F`, L1–L6 `$06FF` (ss); L7–9 `$077F` (`asm:WorldFlags`; `zelda-asm/data/levelinfo.md`) | pointer | both | no — decoder input | asm:RoomFlags_Get, asm:WorldFlags; ss |

## 2. Link

| Address(es) | Meaning | Encoding | Valid when | Include? | Source |
|---|---|---|---|---|---|
| `$70` ObjX[0] | Link x (pixels) | position (÷255) | both | yes | v1 gd.txt `link_x` |
| `$84` ObjY[0] | Link y | position ((y−64)/176) | both | yes | v1 gd.txt `link_y` |
| `$98` ObjDir[0] | facing: 1 E, 2 W, 4 S, 8 N | categorical(4) one-hot | both | yes | v1 zelda_enums.Direction; ram.md |
| `$394` ObjGridOffset[0] | signed distance into current 8-px grid step (v1 `link_grid_offset`, i1); ≠0 bypasses tile collision | ordinal (÷8, signed) | both | yes — sub-tile offset | asm:Walker_CheckTileCollision@CheckGridOffset (via v1 room.py); data.json |
| `$3A8` ObjPosFrac[0] | sub-pixel | ordinal (÷256) | both | no | v1 ram.md |
| `$3F8` ObjInputDir[0] | current input direction bits | bitfield(4) | both | no — duplicates last action | asm:PushBlock_State0Idle |
| `$AC` ObjState[0] | Link state; `(v & $C0)==$40` = halted (can't act/take item); v1 treats bit `$40` as "stunned" and waits for `==0` after attacks | bitfield (bits 6,7) + low nibble categorical | both | maybe — skipped frames make it mostly 0 | asm:Item_TryTakeRoomItem; v1 frame_skip_wrapper.py STUN_FLAG |
| `$C0` ObjShoveDir[0] | knockback direction (bit7 = fresh) | bitfield | both | yes — knockback state | asm:ObjShoveDir; ram.md; data.json `link_shove_dir` |
| `$D3` ObjShoveDistance[0] | knockback remaining, in pixels; a hit sets Link's to `$20` (a monster's to `$40`), `asm:Shove_Begin` | timer (÷$20) | both | yes | asm:ObjShoveDistance, asm:Shove_Begin; ram.md |
| `$4F0` ObjInvincibilityTimer[0] | Link i-frames after hit (counts every 2 frames). A hit sets it to `$18` = 48 frames (`asm:Shove_Begin`); while InvClock is set `$10` is added every frame and the byte wraps (`asm:Mode5Play_UpdateWorld`), so only then does it exceed `$18` | timer (÷$30; the game's value after a hit is `$18`) | both | yes | asm:ObjInvincibilityTimer, asm:Shove_Begin, asm:Mode11Death_Init (comment), asm:Mode5Play_UpdateWorld |
| `$28` ObjTimer[0] | Link's object timer, e.g. $20 block after unlocking door | timer (÷$40) | both | maybe | asm:Door_TouchKey@TriggerDoor; data.json `link_obj_timer` |
| `$53` DoorwayDir | direction of doorway Link is standing in (UW); 0 = none; restricts movement to that axis | categorical(5) | dungeon | yes — cheap and affects legal moves | asm:Room_EnterCommon@Method2, asm:Link_ModifyDirInDoorway; data.json `doorway_dir` |
| `$512` IsLinkParalyzed | grabbed by Like-Like | bool | both | yes | asm:LikeLike_Update@Draw, asm:Link_Update |
| `$52E` IsSwordBlocked | sword disabled (red bubble) | bool | dungeon | yes — decoded (`sword_blocked`); masks SWORD and BEAMS (`GameView::is_sword_blocked`) | asm:Bubble_Update@BlockOrUnblock, asm:Link_HandleInput; ram.md |
| `$4C` SwordBlockedLongTimer | timer before sword re-enabled (non-zero: blocked) | timer | dungeon | yes — decoded (`sword_blocked_timer`); masks SWORD and BEAMS | asm:SwordBlockedLongTimer; ram.md `$4C` |
| `$66F` HeartValues | hi nibble = containers−1, lo nibble = full hearts | ordinal: max_hp=(hi+1)/16, hp | both | yes — hp/max_hp and max_hp | asm:HeartValues (definition comment); v1 link.py |
| `$670` HeartPartial | 0 empty, $01–$7F half, $80–$FF full partial | ordinal (0/0.5/1) | both | yes — folded into hp float | v1 link.py `health` |
| derived: beams ready | sword>0 ∧ (hi==lo nibble of `$66F`) ∧ `$670`≥$80 ∧ beam slot `$BA`==0 | bool | both | yes | v1 link.py `_is_health_full_for_beams` (cites asm:SwordShot_Make) |
| `$B9` ObjState[$D] | sword (melee) slot state; 0 idle | categorical(small) / bool active | both | maybe — "attacking" bit | v1 gd.txt `sword_animation`; frame_skip_wrapper.py |
| `$BA` ObjState[$E] | beam/rod shot: 0 none, $10 flying, $11 spreading, $80 rod shot | categorical | both | yes — beam in flight blocks beams | v1 link.py ANIMATION_*; ram.md |
| `$BB` ObjState[$F] | boomerang $10–$57 / bait $80–$82 | categorical (active bool) | both | maybe | v1 link.py; ram.md |
| `$BC,$BD` ObjState[$10,$11] | bomb $11 placed (one update), $12 lit, $13/$14 exploding; fire $21 moving, $22 standing | categorical | both | yes — bomb active/exploding (needed for bomb walls) | v1 link.py; ram.md; asm:Bomb_Wield, asm:Bomb_Update |
| `$BE` ObjState[$12] | arrow $10 flying, $20/$28 hit; rod $31–35 | categorical | both | maybe | v1 link.py; ram.md |
| `$7D–$82`, `$91–$96`, `$A5–$AA` | x, y, dir of slots $D–$12 (Link weapons) | position | both | maybe — weapon entities for later | asm:ObjX, asm:ObjY, asm:ObjDir + slot; ram.md |
| `$56,$57` Link_GoStraight* | input smoothing flags | bool | both | no | asm:Link_GoStraightWhenDiagInput, asm:Link_GoStraight |

## 3. Inventory & items

Items block `InvItems=$657`, inventory slot `s` is at `$657+s` (`asm:Item_IdToSlot`).

| Address(es) | Meaning | Encoding | Valid when | Include? | Source |
|---|---|---|---|---|---|
| `$657` sword | 0 none, 1 wood, 2 white, 3 magic | categorical(4) | both | yes | ram.md; v1 SwordKind |
| `$658` InvBombs | bombs | ordinal (÷`$67C`, default 8) | both | yes | ram.md |
| `$67C` InvMaxBombs | bomb capacity | ordinal (÷16) | both | no — constant on route | ram.md |
| `$659` InvArrow | 0/1 wood/2 silver | categorical(3) | both | yes (items-owned bits) | ram.md |
| `$65A` Bow | bool | bitfield | both | yes | ram.md |
| `$65B` InvCandle | 0/1 blue/2 red (ram.md; v1 enum names reversed) | categorical(3) | both | yes | ram.md, asm:Item_IdToSlot ids $06/$07 |
| `$65C` whistle, `$65D` food, `$65E` potion(0/1/2), `$65F` rod, `$660` raft, `$661` book, `$662` ring(0/1/2), `$663` ladder, `$664` magic key, `$665` bracelet, `$666` letter, `$66C` InvClock | owned flags | bitfield / small categorical | both | yes — items-owned bits (cheap) | ram.md; v1 gd.txt |
| `$66C` InvClock (also) | **clock active in this room**: freezes monsters, Link invulnerable; cleared on room load | bool | both | yes | asm:InvClock (all `LDA InvClock` checks in bank 4), asm:Link_ResetStateAndClock |
| `$674/$675` boomerang / magic boomerang | owned | bitfield | both | yes | ram.md |
| `$676` InvMagicShield | owned (blocks more projectiles) | bool | both | yes | ram.md |
| `$66D` InvRupees | rupees | ordinal (÷255) | both | yes (scaled) | ram.md |
| `$67D/$67E` InvRupeesToAdd/InvRupeesToSubtract | pending rupee change | ordinal | both | no | ram.md; v1 zelda_game.py |
| `$66E` InvKeys | keys held | ordinal (÷8, clip) | both | yes — key doors | ram.md |
| `$667` InvCompass, `$668` InvMap | **bit (level−1)** set = owned for levels 1–8 | bitfield → derive `has_compass/has_map` for current level | dungeon | yes (derived bit) | asm:Inv_HasCompass, asm:Inv_HasMap. **v1 bug**: link.py returns the raw byte (true if any level's compass) |
| `$669/$66A` InvCompass9/InvMap9 | level 9 | bool | L9 | no (out of milestone) | asm:Inv_HasCompass |
| `$671` InvTriforce | bit (level−1) per piece | bitfield(8) | both | yes — "D1 done" bit matters for the route | ram.md; v1 link.py `has_triforce` |
| `$672` InvPowerTriforce | v1 calls it triforce_of_power | bool | L9 | no | asm:InvPowerTriforce; v1 gd.txt |
| `$656` SelectedItemSlot | B-item: 0 boomerang, 1 bombs, 2 arrows, 4 candle, 5 whistle, 6 food, 7 potion, 8 rod | categorical(10) | both | yes | v1 SelectedEquipmentKind |
| `$513` HasUsedCandle | candle already used this room | bool | both | no (not on route) | asm:Candle_Wield@NotInUse |

## 4. Object slots (tables indexed by slot)

v1 reads slots 0–$B (12 bytes) of each table (gd.txt `[tables]`); add slot `$13` for the room item (see §R.4).

| Address(es) | Meaning | Encoding | Valid when | Include? | Source |
|---|---|---|---|---|---|
| `$34F+i` ObjType | object type: `$01–$48` monsters (v1 ZeldaEnemyKind), `$49/$4A` traps, `$4B–$52` UW persons, `$53–$5C` monster shots (`$55/$56` fireballs, `$5B` monster arrow, `$5C` boomerang), `$5D` dead dummy, `$60` **dropped item**, `$61` dock, `$62/$65/$66` rock/gravestone, `$63/$67` rock wall (OW bombable), `$64` tree (burnable), `$68` **push block**, `$6A+` cave persons. **`$34F` (slot 0) = RoomKillCount** | categorical(~$80) → embedding | both | yes — entity type | asm:Obj_Update_JumpTable, asm:RoomKillCount; v1 zelda_game.py `_is_id_enemy/_is_projectile` |
| `$70+i`, `$84+i` ObjX/ObjY | position | position | both | yes | v1 gd.txt obj_pos_x/y |
| `$98+i` ObjDir | facing (1/2/4/8, diagonals = sums) | categorical(9) | both | yes | ram.md; v1 enemy.py |
| `$AC+i` ObjState | per-type state; for `ObjType=$60` it is the **dropped item id** (Item_ObjItemId); Leever 3 = surfaced, Zora 2–4 = surfaced, Wallmaster 1 = active; block 0/1/2 idle/moving/done | categorical (per type) | both | yes — as "targetable/active" bit + item id for drops | v1 enemy.py `is_active`; asm:Item_ObjItemId; asm:PushBlock_Update |
| `$405+i` ObjMetastate | 0 = normal; 1–4 = spawn cloud (**not yet spawned**); `$10–$13` dying sparkle, `$14` → becomes item | categorical(3): none/spawning/dying | both | yes — spawning/dying bits | asm:Room_EnterCommon@ResetObjLoop, asm:Obj_UpdateMetastate; v1 enemy.py |
| `$492+i` ObjUninitialized | $FF = slot not yet initialized | bool | both | no — plumbing | asm:Room_EnterCommon@ResetObjLoop |
| `$485+i` ObjHP | HP in **high nibble** (v1 `>>4`); ss 1_74w stalfos `$20`, 1_44c goriya `$30` | ordinal (hi nibble ÷15) | both | yes | v1 zelda_game.py `_build_enemy`; ram.md; ss |
| `$3D+i` ObjStunTimer (slot 1 = `$3E`) | stun left, in 10-frame units: a boomerang hit sets `$10`, bomb smoke sets a Dodongo's to `$20`. The clock never writes it: InvClock is a separate byte that the monsters' update code tests beside it | timer (÷$10; ÷$20 for Dodongo) | both | yes | v1 gd.txt obj_stun_timer; asm:ObjStunTimer, asm:InvClock |
| `$28+i` ObjTimer | per-object action/shoot delay | timer (÷$FF) | both | maybe — predicts shots | asm:ObjTimer; ram.md `$29–$2E` |
| `$3A8+i` ObjPosFrac / Item_ObjItemLifetime | sub-pixel; **for items: lifetime**, ≥$F0 = flashing & not pickable | timer (÷$FF) | items only (`ObjType=$60` or slot $13) | maybe — item despawn | asm:Item_TryTake, asm:Item_DrawObject; v1 gd.txt `item_timer` |
| `$C0+i`, `$D3+i` ObjShoveDir/Distance | knockback of monsters | bitfield / timer | both | maybe | asm:ObjShoveDir, asm:ObjShoveDistance |
| `$4F0+i` ObjInvincibilityTimer | monster i-frames; a weapon hit sets `$10` = 32 frames (`asm:Shove_Begin`) | timer | both | maybe | asm:ObjInvincibilityTimer, asm:Shove_Begin |
| `$4B2+i` ObjInvincibilityMask | weapon immunities: 1 sword, 2 boomerang, 4 arrow, 8 bomb, $10 rod beam, $20 fire | bitfield(6) | both | yes — "killable by X" flags | ram.md (fiskbit `dynamic_immunities`) |
| `$4BF+i` ObjAttr | 1 custom collision, 2 thin frame, 4 custom sprite, 8 attr override, $20 no weapon collision, $40 thin (8×16), $80 no reverse on Link hit | bitfield(7) | both | maybe — "hurts-on-touch/killable" proxy | ram.md (fiskbit `dynamic_properties`); asm:Obj_InitTileObjOrItem |
| `$49E+i` ObjCollidedTile | tile number found by the object's last tile collision test (the tile ahead in its moving direction; in the OW walkable tiles are stored as `$26`) | categorical(256) | both | no | asm:ObjCollidedTile |
| `$394+i`, `$3BC+i`, `$3D0+i`, `$3E4+i`, `$3F8+i`, `$380+i` | grid offset, speed, anim counter/frame, input dir, moving limit | various | both | no — use Δposition instead | asm:ObjGridOffset, asm:ObjQSpeedFrac, asm:ObjAnimCounter, asm:ObjAnimFrame, asm:ObjInputDir, asm:ObjMovingLimit |
| `$412–$478` (+i) | per-type scratch (Wallmaster step, Gohma eye, Aquamentus fireball offset, push-block push timer `$41D` etc.) | per type | both | no (generic); maybe boss-specific later | `zelda-asm/src/ObjVars.inc` (e.g. asm:PushBlock_ObjPushTimer) |
| `$486–$490` | v1 `obj_health_1..b` | — | — | — | **v1 bug**: gd.txt names `0x491/0x492` as health_b/c; `$492` is ObjUninitialized. data.json uses `0x490/0x491`. Use `$485+i`, i≤$B |
| `$627` WorldKillCount | kills since Link last took damage (v1 `kill_streak`); 16th kill → fairy drop | ordinal (÷16) | both | maybe | asm:Link_BeHarmed@ResetHelp (reset on damage), asm:Drop_SetUpItem@LookUpItem |
| `$50` HelpDropCount, `$51` IsHelpDropBomb | kills toward guaranteed drop (10 → 5 rupees, or bomb if killed by bomb) | ordinal / bool | both | no | asm:Damage_KillMonster, asm:Drop_SetUpItem@LookUpItem |
| `$52A` WorldKillCycle | 0–9 column in Drop_ItemTable | categorical(10) | both | no | asm:Obj_EndMetastate, asm:Drop_SetUpItem@LookUpItem |
| `$34C` MonsterShotCount, `$514` IsZoraActive, `$510` ActiveRedLeeverCount, `$515` ActiveBoulderCount | counts | ordinal | per type | no — derivable from slots | asm:MonsterShotCount, asm:IsZoraActive, asm:ActiveRedLeeverCount, asm:ActiveBoulderCount |

## 5. Room state (dungeon = UW, overworld = OW)

| Address(es) | Meaning | Encoding | Valid when | Include? | Source |
|---|---|---|---|---|---|
| `flags[room]` = `$067F`/`$06FF`+RoomId (via `LevelInfo_WorldFlagsAddr`) | per-room history byte, **layout differs by mode** — §R.1 | bitfield(8) | meaning changes by mode | yes — core of Room-state block | asm:RoomFlags_Get, asm:RoomFlags_MarkVisited; asm:RoomHistory_SaveKillCountUW, asm:RoomHistory_SaveKillCountOW |
| `$687E+r` LevelBlockAttrsA … `$6AFE+r` …F (SRAM) | static per-room attributes A–F of the current level block — §R.2 | decoded to categoricals | meaning changes by mode | yes (decoded fields) | asm:LevelBlockAttrsA; asm:Door_FindTypeByDirBit, asm:Room_SetUpItemAndPushBlock |
| `$4CD` RoomAttrsF | copy of attr F for the current room | as F | both | no — use F | asm:Room_EnterCommon |
| derived door type ×4 (N,S,W,E) | from A/B: open/wall/false wall/false wall 2/bombable/key/key 2/shutter | categorical(8) ×4 | dungeon | yes | asm:Door_FindTypeByDirBit, asm:Door_LayOut table |
| derived door open ×4 | `flags & dirbit` (key/bomb persisted) OR `CurOpenedDoors & dirbit` (incl. shutters this visit) | bool ×4 | dungeon | yes | asm:Door_AddRoomFlagsToOpenedDoors, asm:Door_ShiftOpenDoorwayMask |
| `$EE` CurOpenedDoors | doors currently open in this room, bits E1 W2 S4 N8 (only true doors; open/walls cleared) | bitfield(4) | dungeon (garbage in OW: ss 0_37s=$08) | yes (via derived door-open) | asm:Door_LayOut, asm:Door_SetEnteringDoorwayOpened; data.json |
| `$54` TriggeredDoorCmd, `$55` TriggeredDoorDir | door animation in progress: 2→3→4 closing, 6→7→8 opening; Dir = bit | categorical(3) / bitfield | dungeon | maybe — "door animating" bool | asm:Room_EnterCommon@Method2, asm:Door_TriggerOpen, asm:Door_Update |
| `$4CE` ShutterTrigger | shutters have been triggered to open | bool | dungeon | maybe | asm:Shutter_OpenTriggered, asm:Shutter_SetTrigger |
| `$27` DoorTimer | door animation step timer | timer | dungeon | no | asm:Door_Update@LoopTransferRec |
| `$AB` RoomItemId (= ObjDir[$13]) | room item id (§R.4). **Stale when already taken** (not rewritten; ss 1_53w shows `$03` while attr E=`$19`) | categorical(36) | both (OW: only heart container in OW room $5F) | yes, but take from attr E masked by taken-flag | asm:Room_SetUpItemAndPushBlock, asm:Item_MoveAndDrawRoomItem |
| `$BF` ObjState[$13] | room item active: `$00` = visible/collectable, `$FF` = hidden (not yet dropped, none, or taken) | bool | both | yes — "treasure visible" | asm:Room_SetUpItemAndPushBlock, asm:Item_TryTake; v1 `treasure_flag` |
| `$83`/`$97` ObjX/Y[$13] | room item position (copied from slot 1 each frame when carried) | position | when present | yes | asm:Room_GetShortcutOrItemXYForRoomId, asm:Item_DrawRoomItemOnMonster; v1 `treasure_x/y` |
| `$3BB` Lifetime[$13] | room item flash/lifetime | timer | when present | no | asm:Item_DrawObject |
| derived `item_carried` | room item present ∧ ObjType[1] ∈ {$17 LikeLike, $2A Stalfos, $30 Gibdo} → drawn on monster 1 (D1 room $74 key) | bool | dungeon | yes | asm:Item_MoveAndDrawRoomItem |
| `$34E` RoomObjCount | monsters created on entry (= FoeCount − remembered kills; 1 for list ids $32–$61; bosses may set 8) | ordinal (÷8, clip) | both | yes (as part of enemies-left) | asm:Room_EnterCommon, asm:Moldorm_Init@LoopSegment, asm:Lanmola_Init |
| `$34F` RoomKillCount (= ObjType[0]) | kills in this room this visit | ordinal | both | yes (enemies-left = count − kills) | asm:RoomKillCount; v1 `room_kills` |
| `$34D` IsRoomAllDead | ≠0 when no live monsters (bubbles excluded); **incremented every check, so it wraps** (ss: $04, $18, $DA) — treat as `!=0` and prefer slot-derived count | bool | both | maybe (derive instead) | asm:Secret_UpdateRoomAllDead |
| `$35F` RoomObjTemplateType | object type of the first monster the room was populated with (slot 1 at room entry). Equals the room's monster list id only for ids below `$62` (an id below `$62` is itself an object type; `$62`+ selects a list of types). Read by the UW room history (non-recurring monsters) and by food | categorical(128) | both | maybe — redundant with room id | asm:RoomObjTemplateType, asm:Room_EnterCommon, asm:RoomHistory_SaveKillCountUW |
| `$560+RoomId` LevelKillCounts | exact partial kills per room in the current level ($F = all); cleared on level load | ordinal (÷8) | dungeon | yes (enemies-left for revisits) | asm:RoomHistory_SaveKillCountUW, asm:Mode2Load_Init |
| `$4CF` BlockPushComplete | 0 not pushed, 1 pushed (secret not yet handled), 2 pushed + handled | categorical(3) | dungeon | yes | asm:PushBlock_State1Moving, asm:Secret_Trigger5BlockStairs |
| `$AC+$B` = `$B7` ObjState[$B] (when ObjType[$B]=$68) | push block 0 idle, 1 moving, 2 done; position `$7B/$8F` | categorical(3) + position | dungeon | yes (as entity + flag) | asm:PushBlock_Update, asm:PushBlock_FindAndCreate |
| `$41D` PushBlock_ObjPushTimer[$B] | frames Link has pushed (block moves at $10) | timer (÷16) | dungeon | maybe | asm:PushBlock_State0Idle |
| `$52B/$52C/$52D` RoomTileObjType/X/Y | OW tile object (armos/rock) being activated | categorical / position | OW | no | asm:RoomTileObjType, asm:RoomTileObjX, asm:RoomTileObjY |
| `$51A` SecretColorCycle, `$51B` FluteUsedState | flute/pond secret progress | ordinal | OW | no (not on route) | asm:FluteSecret_Update |
| `$52F` MazeStep | lost woods/mountain maze progress | ordinal | OW | no | ram.md; asm:MazeStep |
| `$4CC` IsPersonFireballsEnabled | nonzero after the UW person in slot 1 was attacked: from then on two fireball sources left and right of the person shoot at Link (statue pattern 2); the person itself does not shoot | bool | dungeon | no | asm:IsPersonFireballsEnabled, asm:Statue_Update |

## 6. Tiles / layout

| Address(es) | Meaning | Encoding | Valid when | Include? | Source |
|---|---|---|---|---|---|
| `$6530–$67EF` PlayAreaTiles (`ram()` index `0xD30`, len `$2C0`) | 32 cols × 22 rows of 8-px tiles, **column-major** (`tiles[x*22+y]`); row r covers y = $40+8r | categorical(256) → tile class map (below) | both | yes — Room grid | asm:PlayAreaTiles, asm:RoomLayout_FillWalls, asm:PushBlock_FindAndCreate; v1 zelda_game.py `current_tiles` |
| `$34A` RoomFirstUnwalkableTile | walkability threshold: OW `$89`, UW `$78`; becomes `$99` after pond secret | ordinal — decoder input | both | no (input) | asm:Room_ObjBoundsOW, asm:Room_ObjBoundsUW, asm:FluteSecret_CuePondPaletteRow@CopyBytes; ss |
| `$346–$349` RoomBoundLeft/Right/Up/Down | OW `$11,$E0,$4E,$CD`; UW `$21,$D0,$5E,$BD` | position — decoder input | both | no (input; for masks) | asm:Room_ObjBoundsOW, asm:Room_ObjBoundsUW; ss |
| attr D & `$3F` (UW) / `$7F` (OW) | unique room layout id (column template) | categorical(64/128) | meaning changes by mode | maybe — "layout template ID" embedding | asm:Room_GetUniqueRoomId, asm:RoomLayout_LayOutRoomOW (OW keeps 7 bits) |
| `$530` PlayAreaAttrs | palette attrs | — | both | no | asm:PlayAreaAttrs |

**Tile → class rules** (apply in this order; values from asm and ss tile dumps of `1_42e`, `1_35e`, `0_77c`):

| Class | Rule | Source |
|---|---|---|
| walkable | `t < threshold` (`$34A`); OW also `t ∈ {$8D,$91,$9C,$AC,$AD,$CC,$D2,$D5,$DF}` | asm:Tile_WalkableTilesOW, asm:Tile_GetAheadOfObj |
| stairs | `$70–$73` (UW stair tile `$70`; OW revealed stairs `$70`) | asm:Secret_Trigger5BlockStairs, asm:Secret_RevealAndFlagStairsObj, asm:Mode10Stairs_Update_Full |
| cave/dungeon entrance | OW `$24` (black, bottom) under `$F3` (top) | asm:Room_EnterCommon ($24 = black entrance); ss 0_77c; v1 room.py |
| water | OW `$8D ≤ t < $99`; UW `$F4` (the Blue Wizzrobe teleports through "block or water" = tiles `$B0–$B3` and any tile `≥$F4`; every other unwalkable tile is a wall to it) | asm:Link_EndMoveAndAnimate, asm:Zora_TrySpawn@TryCandidate, asm:Wizzrobe_Blue_MoveAndCheckTile |
| block (solid) | `$B0–$B3` (any block/statue square). The *pushable* one is the first `$B0` in row `$A` from col 4 when attr D bit 6 set; it is ObjType `$68` in slot `$B` | asm:PushBlock_FindAndCreate, asm:Wizzrobe_Blue_MoveAndCheckTile |
| armos/gravestone | `$BC–$C3` (OW) | asm:Spawn_CheckTouchedArmosOrGravestone |
| UW floor | `$74–$77` (`$74` written when block leaves) | asm:PushBlock_State0Idle |
| door | **do not classify from tiles**; use derived door type/open state stamped onto the 4 door cells: N cols 14–17 rows 0–3; S cols 14–17 rows 18–21; W cols 0–3 rows 9–12; E cols 28–31 rows 9–12 (open doorway shows `$24` at cols 29/2, rows 10–11). These cells are padded out to the map edge; the door faces the game itself writes are N cols 14–17 rows 1–3, S cols 14–17 rows 18–20, W cols 1–3 rows 9–12, E cols 28–30 rows 9–12 (`asm:Door_PlayAreaAddrsLo`, `asm:Door_LayOut`) | ss 1_42e/1_35e; v1 room.py `*_DOOR_TILE`; asm:Door_PlayAreaAddrsLo |
| shutter (closed) tiles | `$AC–$AF` | v1 room.py BARRED_DOOR_TILES; ss 1_42e W door |
| locked door tiles | `$98–$A7` | v1 room.py DOOR_TILES |
| wall | everything else `≥ threshold` | asm:Tile_GetAheadOfObj |

Recommended grid: 16×11 of 16-px squares (row r covers y=$40+16r), class = max-priority class of its 2×2 tiles; Link's collision samples the *higher* of two tile values (v1 room.py `_check_vertical`, asm:Tile_GetAtObjOffset@FetchTile), so 8-px resolution (32×22) is the exact one. Choose per v1's movement port.

## 7. Timers / RNG

| Address(es) | Meaning | Encoding | Valid when | Include? | Source |
|---|---|---|---|---|---|
| `$15` FrameCounter | frame counter | — | both | no (leaks time; use History) | asm:FrameCounter; ram.md |
| `$18–$24` Random | LFSR RNG bytes, 13 (rng_0..12; `Nmi_Isr@ScrambleRandom` rotates $18–$24, monsters read `Random+1,X` up to $24) | — | both | no — load randomization redraws all 13 (`RNG_VARS`) | v1 gd.txt; asm:Random, asm:Nmi_Isr@ScrambleRandom |
| `$3C` FluteTimer | freeze while flute plays | timer | both | no | asm:Mode5Play_Update |
| `$26` LongTimerCycle | divider for every 10-frame ("long") timer: counts 9 down to 0 and wraps; paces the stun timers and the `$4A`–`$4D` long timers alike | — | both | no | asm:LongTimerCycle, asm:Nmi_Isr |
| `$4A`, `$4B`, `$4D` long timers | chase / edge-spawn / red-leever | timer | per type | no | asm:ChaseLongTimer, asm:MonstersFromEdgesLongTimer, asm:RedLeeverLongTimer |
| `$60` IsChasingOtherTarget, `$61`/`$62` ChaseTargetX/Y | **where monsters chase** ("shadow Link"). With `$60`=0, current Link X/Y are copied before weapon/bait updates. When `$4A` expires (0–7 ticks, one per 10 frames), `$60` flips; on 0→1, if target X still equals current Link X, both target bytes XOR `$FF`. Without bait this normally mirrors the current position even if Link moved: the upstream disassembly's "last frame" comment was misleading (`zelda-asm/` corrects it). On 1→0 the old target survives that update. Bait can override actual coordinates without changing the flag. Mirrored Y may be outside the play area; use full-byte coordinates, not an inferred identity | position + refresh bool | both | candidate (round 7 audit): actual monsters' goal; no RNG required | asm:Mode5Play_UpdateWorld (instruction order), asm:Boomerang_UpdateOrFood (bait), asm:Nmi_Isr (timer); bank 4 chase movement |
| `$524` SpawnCycle, `$525` CurEdgeSpawnCell | monster spawn pacing / edge cell | categorical | both | no | asm:SpawnCycle, asm:EdgeSpawn_FindNextCell@SetSpawnCell |

## 8. Misc

| Address(es) | Meaning | Encoding | Valid when | Include? | Source |
|---|---|---|---|---|---|
| `$605` CurTune0 | sfx: 1 parry (shield block, Darknut shield, boomerang on an immune monster), 2 monster hit (also a boomerang stun), 4 magic, 8 key pickup, $10 heart pickup, $20 set bomb, $40 low-heart warning | bitfield | both | no (obs) — reward/event signal | ram.md; v1 SoundKind; asm:Link_CheckMonsterTouch_NoReset@Parry, asm:Collision_CheckWeaponPointVsMonster, asm:Damage_HandleWeaponHit, asm:Damage_DealToMonster |
| `$607` CurTune1 | 1 rupee, 2 **item appears / key appears**, 4 **secret revealed**, 8 item taken (every item but $0E), $10 whistle, $20 enemy burst, $40 continue, $80 death | bitfield | both | no (obs) — reward/event signal | ram.md; asm:Secret_CheckUnderworld, asm:PushBlock_State1Moving, asm:Item_Take |
| `$E0` PauseState, `$E1` SubmenuState | pause / submenu | bool | both | no | asm:PauseState, asm:SubmenuState |
| `$F8/$FA` ButtonsPressed/Down | controller | — | both | no | asm:ButtonsPressed, asm:ButtonsDown |
| `$5E` SubmenuScrollProgress, `$FC/$FD` scroll | screen scroll | — | transitions | no | asm:SubmenuScrollProgress |
| `$E5` DeathModeCounter | death spin | — | death | no | asm:DeathModeCounter |
| `$62D+slot` QuestNumbers, `$630+slot` DeathCounts, `$16` CurSaveSlot | profile | — | both | no | asm:QuestNumbers, asm:DeathCounts, asm:CurSaveSlot |
| `$413/$422/$430` CaveFlags/CaveItemIds/CavePrices | cave wares (item id low 6 bits, flags hi 2) | categorical | cave (mode $B) | maybe — sword cave only needs "item taken" flag | asm:Cave_Init_SetUpTextAndWares |
| `$522` WhirlwindTeleportingState, `$523` TeleportingLevelIndex | whistle warp | — | OW | no | asm:WhirlwindTeleportingState, asm:TeleportingLevelIndex |
| `$64` LadderSlot | ladder placement | — | both | no | asm:LadderSlot |

---

# R. Room-state decoding (bit level)

Notation: `r` = RoomId (`$EB`), `S(A)` = SRAM byte `ram()[A-0x5800]`, `flags = ram[wfa + r]` with `wfa = S($6BAF) | S($6BB0)<<8`.

## R.1 Room flags byte (`RoomFlags_Get`)

| Bit | Dungeon (UW) | Overworld (OW) | Set by / read by |
|---|---|---|---|
| 0 | door **E** opened (key or bombed) | kill count bit 0 | UW: `asm:RoomFlags_SetDoorOpened` (mask `Util_BitMasks[dirIndex]`, 0=E,1=W,2=S,3=N); OW: `asm:RoomHistory_SaveKillCountOW` (`AND #$07`) |
| 1 | door **W** opened | kill count bit 1 | same |
| 2 | door **S** opened | kill count bit 2 | same |
| 3 | door **N** opened | — (no reader found) | `asm:Door_ShiftOpenDoorwayMask`, `asm:Door_AddRoomFlagsToOpenedDoors` |
| 4 | **room item taken** | **room item / cave item / secret reward taken** (e.g. sword cave room $77 → `$10`, ss 0_67w) | `asm:RoomFlags_SetItemTaken` (`ORA #$10`), `asm:RoomFlags_GetItemTaken` (used in OW caves too, `asm:Cave_Init@CheckTaken`, `asm:Cave_State2TalkOrShopOrDoorCharge@Take`) |
| 5 | **visited** (dungeon map) | **put the stairs back on entry** (`asm:Secret_RestoreStairsOW`, only while bit 7 is clear): set for the room where a rock or gravestone was pushed its full `$10` pixels and for the room shortcut stairs lead to. Not set by merely entering an OW room (that call runs only when `CurLevel≠0`) | `asm:RoomFlags_MarkVisited` (`ORA #$20`), `asm:Submenu_WriteScanningMapRoomMark`, `asm:Secret_RestoreStairsOW` |
| 6–7 | **kill count**: 0–2 partial kills (capped), **3 = all killed** | bit 7 = **secret found** (cave bombed/burned/stairs revealed) | UW `asm:RoomHistory_SaveKillCountUW` (`AND #$3F`, `ROR`×3, `#$C0`); OW `asm:Secret_RevealAndFlagTileObj` (`ORA #$80`), `asm:Secret_RestoreStairsOW`, `asm:RoomLayout_LayOutRoomOrCaveOW` (`AND #$80`) |

Observed (old savestate `1_35e`, since deleted, level-1 block): `$74:B0` (kills=2-of-cap, visited, key taken), `$53:F0` (all dead, visited, taken), `$35:F4` (boss dead, visited, heart taken, S door opened), `$52:68` (1 kill, visited, N opened), `$43:21` (visited, E opened). OW (old `0_67w`): `$67:07` (OW kill count 7 = all), `$77:10` (sword taken). The route-generated chain (`1_36w_2` walked back into 0x35, `tests/test_room_state.py`) reads `$53:F0`, `$35:F4`, `$74:70`, `$52:A8`, `$43:71`; the differences are kills and pickups taken in a different order.

**Opening a key/bomb door sets the flag on both sides immediately** (`asm:Door_Update_ResetCmdAndLayOut`: `RoomFlags_SetDoorOpened` then `ORA Util_BitMasks, X^1` into room `r + Door_NextRoomIdOffsets[X]`). Shutters are never persisted (`asm:Door_Update_ResetCmdAndLayOut` skips type 7). Bombable doors, once flagged, render as a hole (door face 9, `asm:Door_LayOut`).

**Kill persistence.** On leaving a room `RoomHistory_SaveKillCount{UW,OW}` stores kills; on entry `RoomHistory_ModifyObjCount{UW,OW}` subtracts:
- UW: if room ∈ RoomHistory (last 6) or `flags[6:7]≠3`: `count = FoeCount − LevelKillCounts[r]` (`$560+r`). If `flags[6:7]==3` and not in history: recurring monsters respawn (flag & `LevelKillCounts` reset) unless the template id is "non-recurring" (`$32 ≤ id < $49`, except `$3A,$3B`) → count 0.
- OW: kill count lives only in `flags[0:2]` (7 = all); if not in history and 7 → reset to 0 (monsters respawn).

## R.2 Level-block attribute bytes (`LevelBlockAttrsA..F`, SRAM `$687E/$68FE/$697E/$69FE/$6A7E/$6AFE` + r)

| Byte | Bits | Dungeon (UW) | Overworld (OW) | Source |
|---|---|---|---|---|
| A | 7–5 | **N door type** | 7–4: x where Link exits underground (`& $F0`) | asm:Door_FindTypeByDirBit; asm:Room_EnterCommon |
| A | 4–2 | **S door type** | bit 3 zora present, bit 2 sea sound | same; asm:Zora_TrySpawn, asm:Mode5Play_UpdateWorld@CheckOW |
| A | 1–0 | outer palette | outer palette | asm:Room_FillPlayAreaAttrs |
| B | 7–5 | **W door type** | 7–2: cave index (`(B&$FC)`: <$40 → level `B>>2`; $40+ → cave, $50 → shortcut, dweller ObjType `$6A+((B&$FC)−$40)/4`) | asm:Door_FindTypeByDirBit; asm:Warp_HandleOW, asm:Spawn_AssignPositions@InCave |
| B | 4–2 | **E door type** | (part of cave index) | same |
| B | 1–0 | inner palette | inner palette | asm:Room_FillPlayAreaAttrs |
| C | 7–6 | foe-count index → `LevelInfo_FoeCounts[$6BA2+idx]` (L1: 3,5,6,8; OW: 1,4,5,6 — ss) | same | asm:Room_EnterCommon@ResetObjLoop |
| C | 5–0 | monster list id low 6 bits | same | asm:Room_EnterCommon@ResetObjLoop |
| D | 7 | monster list id bit 6 | same | asm:Room_EnterCommon@ResetObjLoop |
| D | 6 | **push block present** | layout id bit 6 | asm:Room_SetUpItemAndPushBlock (`AND #$40`); asm:RoomLayout_LayOutRoomOW |
| D | 5–0 | unique room layout id (`$3E/$3F` = cellars) | layout id bits 0–5 | asm:Room_GetUniqueRoomId; asm:Mode11Death_Init_Sub1 |
| E | 7 | dark room | (cave block: E holds cave item ids + flags) | asm:Room_IsDark_Bank5; asm:Cave_Init_SetUpTextAndWares@CopyCaveBytes |
| E | 6–5 | boss ambient sound index | — | asm:Sample_SetBossRoarUW |
| E | 4–0 | **room item id**; `$03` means **none** | — | asm:Room_SetUpItemAndPushBlock |
| F | 7–6 | — | quest-secret selector | asm:Secret_IsForOtherQuest |
| F | 5–4 | **item position index** → `LevelInfo_ShortcutOrItemPosArray[$6BA7+idx]` | shortcut/stairs position index | asm:Room_GetShortcutOrItemXYForRoomId |
| F | 3 | — | monsters enter from screen edges | asm:Spawn_AssignPositions, asm:Obj_Init |
| F | 2–0 | **secret trigger** (R.3) | square row Link exits underground | asm:Secret_CheckUnderworld; asm:Room_EnterCommon |

Door types (`asm:Door_LayOut` comment table):

| Value | Type | Passable when | Persisted open in flags? |
|---|---|---|---|
| 0 | open | always | n/a (bit cleared from CurOpenedDoors) |
| 1 | wall | never | n/a |
| 2 | false wall | drawn as plain wall and treated as wall by map/doorway mask; Link walks through after pushing against it for a `$18`-frame timer (Link's ObjTimer: 0 → set `$18` and block; 1 → pass; `asm:Door_TouchFalseWall`) | n/a |
| 3 | false wall 2 | same handler as 2 (`asm:Door_Touch_JumpTable`) | n/a |
| 4 | **bombable** | after bomb (`asm:Bomb_TryOpenWall@LoopHotspot` triggers open cmd) | yes |
| 5 | **key** (locked) | after key (`asm:Door_TouchKey`: magic key or `DEC InvKeys`) | yes |
| 6 | key 2 | same handler (`asm:Door_TouchKey`) and same locked-door face as 5 (`asm:Door_Touch_JumpTable`, `asm:Door_LayOut`) | yes |
| 7 | **shutter** | when room trigger fires (R.3) — this visit only; entering through an opened door issues close cmd 2 (`asm:Room_EnterCommon@Method2`) | no |

Suggested per-direction door feature: `type ∈ {none(wall/false), open, key, bomb, shutter}` one-hot + `is_open` bit, where `is_open = type==open ∨ (flags & dirbit) ∨ (CurOpenedDoors & dirbit)`.

## R.3 Secret trigger (attr F bits 2–0, UW) — "what makes the room do something"

| Value | Name (`Secret_RunTrigger_JumpTable`) | Effect |
|---|---|---|
| 0 | none | — |
| 1 | all dead | shutters open when no monsters (bubbles ignored) |
| 2 | ringleader | killing monster in slot 1 kills all |
| 3 | last boss | shutters open when `InvPowerTriforce`; room item hidden until then |
| 4 | block → door | shutters open when `BlockPushComplete≠0` |
| 5 | block → stairs | when block pushed, stairs tile `$70` appear at ($D0,$60) via slot $B; `BlockPushComplete` 1→2 |
| 6 | money or life | shutters open when person (slot 1) gone |
| 7 | **all dead → item** ("clear-to-drop") | shutters open **and** room item activates (`ObjState[$13]=0`, tune1 `$02`) when all dead |

Sources: `asm:Secret_CheckUnderworld`, `Secret_Trigger*` (`asm:Secret_RunTrigger_JumpTable`), `asm:Room_SetUpItemAndPushBlock` (triggers 3 and 7 start the item hidden). Push block itself can only move when `IsRoomAllDead≠0` (`asm:PushBlock_State0Idle`), Link aligned within $10 px and pushing $10 frames.

## R.4 Room item ("treasure")

Decode (UW):
```
taken    = flags & 0x10
item_id  = attrE & 0x1F            # 0x03 = no item
has_item = item_id != 0x03 and not taken
hidden_until_clear = (attrF & 7) in (3, 7)
visible  = ram[0xBF] == 0x00        # ObjState[$13]; $FF = hidden/none/taken
pos      = ItemPos[(attrF >> 4) & 3]; x = pos & 0xF0; y = (pos & 0x0F) << 4
if item_id == 0x1B: x -= 8          # triforce shifted left
carried  = visible and ObjType[1] in (0x17, 0x2A, 0x30)   # item follows monster 1
```
`ItemPos` = `LevelInfo_ShortcutOrItemPosArray` (SRAM `$6BA7..$6BAA`; L1: `C9 AC 89 87`). OW: the only room item is the heart container in room `$5F` at ($C0,$90) (`asm:Room_SetUpItemAndPushBlock@MakeHeartContainerOW`); cave items use `flags & 0x10`.

Item ids (`asm:Item_IdToSlot` → inventory slot `$657+s`; names from slot table and v1 ZeldaItemKind):

| id | item | id | item | id | item | id | item |
|---|---|---|---|---|---|---|---|
| 00 | bombs (+4) | 09 | silver arrow | 12 | blue ring | 1B | triforce piece |
| 01 | wood sword | 0A | bow | 13 | red ring | 1C | magic shield |
| 02 | white sword | 0B | magic key | 14 | power bracelet | 1D | boomerang |
| 03 | magic sword / **"none" in room attr** | 0C | raft | 15 | letter | 1E | magic boomerang |
| 04 | food | 0D | ladder | 16 | **compass** | 1F | blue potion |
| 05 | whistle | 0E | Triforce of Power (`$672`, `asm:InvPowerTriforce`) | 17 | **map** | 20 | red potion |
| 06 | blue candle | 0F | 5 rupees | 18 | rupee | 21 | clock |
| 07 | red candle | 10 | magic rod | 19 | **key** | 22 | heart |
| 08 | arrow | 11 | book | 1A | **heart container** | 23 | fairy |

(`$0F` = 5 rupees per the help-drop comment in `asm:Drop_SetUpItem@LookUpItem`; `$3F` = "none" for `RoomItemId`, `asm:Item_MoveAndDrawRoomItem`.)

## R.5 Enemies remaining (incl. not yet spawned)

At room init every monster is written into `ObjType[1..count]` immediately (`asm:Room_EnterCommon@StoreObjCount`) with `ObjMetastate=1` (spawn cloud), so:
```
monsters    = [i for i in 1..0xB if 1 <= ObjType[i] < 0x49 and ObjType[i] not in (0x2B,0x2C,0x2D)]   # Secret_UpdateRoomAllDead
unspawned   = [i in monsters if 1 <= ObjMetastate[i] <= 4]
dying       = [i in monsters if 0x10 <= ObjMetastate[i] <= 0x13]
remaining   = len(monsters) - len(dying)
```
The game counts $01–$2A and $2E–$48 as living (traps $49/$4A and persons $4B–$52 don't). `room_state.rs` counts
`ObjectClass::Monster` minus the guard fires $3F, so it deliberately also leaves out the hazards the game counts
(boulders $1F/$20, standing fire $40).
Cross-check: `RoomObjCount($34E) − RoomKillCount($34F)`. Persistent across visits: UW `flags[6:7]`, `LevelKillCounts[$560+r]`; OW `flags[0:2]`. Exceptions: Digdogger children reset `RoomObjCount` (asm:Digdogger_HandleFlute@MakeChildren); some bosses set 8 (asm:Moldorm_Init@LoopSegment, asm:Lanmola_Init); OW edge-spawn rooms (F bit 3) place monsters at edges later (`asm:Obj_Init`).

## R.6 Push block

`has_block = attrD & 0x40` (UW). Object: `ObjType[$B]==$68`, position `($7B,$8F)`, state `ObjState[$B]` 0 idle / 1 moving / 2 done; `BlockPushComplete($4CF)` 0/1/2. **What it opens** = attr F trigger: 4 → shutters (all `type==7` doors), 5 → stairs at ($D0,$60). Trigger 0 with a block (D1 room `$22`) → the block only moves (see open questions).

## R.7 Dungeon item state

`has_map = InvMap & (1 << (level-1))`, `has_compass = InvCompass & (1 << (level-1))` (L9 uses `$669/$66A`), `keys = $66E`, `magic_key = $664`. Map/compass/key room items are attr E ids `$17/$16/$19` (§R.4).

## R.8 Overworld secrets

`secret_found = flags & 0x80`; tile objects in slot `$B` (`$62–$67`) are the OW bombable rock walls, trees, rocks (ss `0_67w`: ObjType[$B]=`$63` rock wall at ($70,$50)). Cave destination from attr B (§R.2).

## R.9 Verification: Dungeon 1 decoded from ss `1_73n` attributes

| r | N S W E (types) | trigger | item @ (x,y) | block | foes | v1 gd.txt `[rooms]` |
|---|---|---|---|---|---|---|
| 73 | key open open open | – | none | – | 0 list | NEW |
| 72 | wall wall wall open | dead→item | key @ (A0,C0) | – | 3 | E, key |
| 74 | wall wall open wall | – | key (carried by stalfos) | – | 5 | W, key |
| 63 | open key wall wall | – | none | – | 3 | NS |
| 53 | **bomb** open open open | dead→item | key @ (80,70) | – | 5 | SEW, key, secret N |
| 54 | bomb wall open wall | – | compass @ (C0,90) | – | 8 | W, compass |
| 52 | key wall wall shutter | all dead | none | – | 6 | NE |
| 42 | wall key shutter open | block→door | none | **yes** | 3 | EW |
| 43 | open **bomb** open key | – | map @ (C0,90) | – | 5 | NEW, map |
| 44 | wall bomb key open | dead→item | boomerang @ (80,70) | – | 3 | EW, boomerang |
| 45 | key wall open wall | – | key @ (A0,C0) | – | 8 (wallmasters) | WN, key |
| 35 | wall key wall shutter | dead→item | heart container | – | 1 (Aquamentus) | NE, heart-container |
| 36 | wall wall open wall | – | triforce @ (78,90) | – | 0 | W, triforce |
| 33 | key open wall wall | – | key @ (80,90) | – | 3 | NS, key |
| 23 | wall key key wall | dead→item | key @ (80,70) | – | 3 | SW, key |
| 22 | wall wall wall key | – | none | **yes** | 1 (list `$4A` ∈ [$32,$62) → count forced to 1, `asm:Room_EnterCommon@ResetObjLoop`) | — |

RAM agrees in ss `1_35e`: `CurOpenedDoors=$05` (S key door + E shutter), `flags[$35]=$F4`. (v1 lists `35` exits as "NE"; decode and RAM say S/E.)

---

# Recommended starting observation feature set (§4.2)

| Block | Rows |
|---|---|
| **Room identity** | `(CurLevel $10, RoomId $EB)` → embedding; optional layout id (attr D `&$3F` UW / `&$7F` OW) embedding; `in_cave`/`in_cellar` from `$12` |
| **Room state** | per direction ×4: door type one-hot {none, open, key, bomb, shutter} + is_open (R.2, `$EE`, flags bits 0–3); visited (bit 5); kill state (UW bits 6–7 one-hot, OW bits 0–2 ÷7); item: has_item, item id embedding, taken (bit 4), visible (`$BF==0`), hidden_until_clear (trigger 3/7), carried, item x/y; trigger one-hot(8); push block present, BlockPushComplete one-hot(3); enemies remaining ÷8, unspawned ÷8; IsRoomAllDead (derived); door animating (`$54≠0`); InvClock active; OW secret_found (bit 7); is_boss_room / is_triforce_room (LevelInfo) |
| **Link** | x, y, grid offset `$394`, facing one-hot `$98`, hp/max_hp (`$66F/$670`), max_hp, i-frames `$4F0`, knockback (`$C0` dir, `$D3`), paralyzed `$512`, doorway dir `$53`, sword level `$657`, beams-ready, beam/bomb slot active (`$BA`, `$BC/$BD`), bombs `$658`, keys `$66E`, rupees `$66D` ÷255, items-owned bits (§3), selected item `$656`, has_map/has_compass for this level, triforce bits `$671` |
| **Entities** | slots 1–$B + room item ($13): presence, ObjType embedding, x/y, dx/dy, Δpos, HP hi-nibble `$485`, stun `$3D`, dir `$98`, metastate {normal, spawning, dying}, ObjState-derived active (Leever/Zora/Wallmaster), immunities `$4B2`, item id for `$60` drops, item lifetime `$3A8` |
| **Room grid** | classes from §6 over PlayAreaTiles with door cells overwritten from R.2 (locked/shutter/bombable/open) and push block / stairs |
| **History** | not RAM — last action kind/dir/duration, frames since |

# Open questions / unverified

1. **Answered:** the level 7–9 room-flag block is at `$077F` (`asm:WorldFlags`: `$67F` overworld, `$6FF` levels 1–6, `$77F` levels 7–9; the level info blocks of levels 7–9 hold `$077F`, `zelda-asm/data/levelinfo.md`). Was: inferred from the `$67F–$7FF` copy loop, `asm:Mode1Menu_Update_Sub1Load@CopyWorldFlags`; not on milestone.
2. **Answered:** door types 2/3 ("false wall") let Link through after he pushes for a `$18`-frame timer (`asm:Door_TouchFalseWall`); type 6 ("key 2") uses the same handler as 5 (`asm:Door_TouchKey`, `asm:Door_Touch_JumpTable`). See R.2.
3. D1 room `$22` has push-block bit set but trigger 0; how the bow-cellar stairs are revealed (and v1's `1 13 stair` row; attrs of `$13` in the L1–6 grid look like another dungeon's room) — check with RouteSurvey/route data.
4. **Answered (the scan):** `asm:Secret_UpdateRoomAllDead` reads `ObjType+1+Y` for `Y = CurObjIndex ($B)` down to 0, so the all-dead scan covers slots `$0C`..`$01`. `Obj_FindEmptyMonsterSlot` only hands out `$B..$1`; whether anything else ever puts a type in slot `$C` was not checked.
5. ObjHP low nibble meaning (still open). **Answered:** start values of the timers: stun `$10` from a boomerang, `$20` for a bomb-smoked Dodongo, in 10-frame units (`asm:ObjStunTimer`); shove distance `$20` pixels for Link, `$40` for a monster; i-frame timer `$18` for Link, `$10` for a monster (`asm:Shove_Begin`; with the clock Link's keeps rising and wraps, `asm:Mode5Play_UpdateWorld`).
6. Link `ObjState[0]` bit meanings beyond `$40` (halted) — only v1 heuristics.
7. OW flag bits 3, 5, 6: no `RoomFlags_Get` reader found (grep) for bits 3 and 6; assume unused. **Answered for bit 5:** `asm:Secret_RestoreStairsOW` reads it (R.1).
8. **Answered:** `IsRoomAllDead` is incremented on every frame the room is empty (`asm:Secret_UpdateRoomAllDead`), so it reads 0 on one frame in 256 and the readers that test it for zero (push block, all-dead triggers) see "monsters present" on that frame. Derive from slots instead.
9. OW attr D bit 6 is used as a layout-id bit (`asm:RoomLayout_LayOutRoomOW` shifts 7 bits) — confirm against OW layout table size.
10. **Answered:** item `$0E` (inventory byte `$672`) is the Triforce of Power (`asm:InvPowerTriforce`); not on route.
