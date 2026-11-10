# RAM feature catalog (Phase 0, design §4.1)

Every RAM/SRAM variable that v1 reads, plus the disassembly variables needed to decode **room state**.
One row per candidate observation feature.

**Citation keys**
- `ASM Z_0n:Label` → `zelda1-disassembly/src/Z_0n.asm`, routine/table `Label` (local path `/home/leculver/work/git/zelda1-disassembly/src`). Variable names come from `Variables.inc` / `ObjVars.inc` / `CommonVars.inc` / `CaveVars.inc`.
- `v1 file` → `/home/leculver/work/git/triforce/triforce/<file>` (`gd.txt` = `zelda_game_data.txt`, `data.json` = v1's retro integration variable table; this repo's equivalent is `crates/zelda/src/vars.rs`, `ram.md` = `docs/ram.md`).
- `ss <name>` → value observed by loading savestate `<name>.state` in this repo, stepping no-op until `GameMode=5`, and reading RAM. Historical: these observations were made in Phase 0 on the old stable-retro 1.0.1 savestates (since replaced by `TFZSTATE` files in `triforce/states/`, read with `triforce._native.Game`).

**Address space.** `Game.ram()` (`triforce._native`) returns 10240 bytes: index `0x000–0x7FF` = CPU RAM `$0000–$07FF`; index `0x800–0x27FF` = cartridge SRAM `$6000–$7FFF`. So **SRAM address `A` is at `ram()[A - 0x5800]`** (v1's `tile_layout` at index `0xD30` is `PlayAreaTiles $6530`). Verified: attribute bytes read this way decode to door layouts that match RAM door state in ss `1_35e`, `1_42e` (see §R.9).

**Object slot indices** (fiskbit notes in v1 `ram.md`, confirmed by asm and ss): `0` Link · `1–$B` monsters/projectiles/dropped items (`ASM Z_07:FindEmptyMonsterSlot` searches `$B..1`) · `$B` is also the *tile-object* slot (push block, OW rock wall/tree; `ASM Z_05:FindAndCreatePushBlockObject`, `CheckSecretTriggerBlockStairs` "LDX #$0B ; Tile object slot") · `$C` see open questions · `$D` sword · `$E` beam/rod shot · `$F` boomerang/bait · `$10,$11` bomb/fire · `$12` arrow/rod · `$13` room item ("treasure"). Table address for slot `i` = base + `i`.
> The `; TRIFORCE:` comment in `Variables.inc` lines 49–59 (sword = slot `$0B`, beam = `$0E`) is wrong: `ObjState+$0D = $B9` is the sword slot (v1 `sword_animation`), and `$0B` is the tile-object slot.

Encoding vocabulary: `categorical(n)` · `ordinal/scale (÷max)` · `bitfield` · `timer (÷max)` · `position (÷max)`. Positions: x ∈ 0–255 (÷255); y play area is 64–239 (rows start at `$40`, `ASM Z_07:GetCollidingTileMoving`), normalize `(y−64)/176`.

---

## 1. Game mode & location

| Address(es) | Meaning | Encoding | Valid when | Include? | Source |
|---|---|---|---|---|---|
| `$10` CurLevel | 0 = overworld, 1–9 = dungeon | categorical(10) | both | yes — room identity key | ASM Variables.inc; v1 gd.txt `level` |
| `$EB` RoomId | room 0–$7F; high nibble = row, low = column (`ASM Z_05:SplitRoomId`). Dungeons 1–6 share one 128-room grid (ss `1_*`, `2_*` both use flags block `$6FF`) | categorical(128) → with level, room embedding | both | yes — room identity | ASM Z_05:SplitRoomId; v1 gd.txt `location` |
| `$12` GameMode | 0 demo, 1 menu, 2 load level, 3 unfurl, 4 enter room, **5 play**, 6 leave room, 7 scroll, 8 continue?, **9 cellar**, **$A leave cellar**, **$B cave**, **$C shortcut cave**, $D save, $E/$F register/elim, $10 stairs, $11 death, $12 end level (triforce), $13 win. Modes 5,9,A,B,C all run `UpdateMode5Play` | categorical(20) | both | no (feature) — use for frame skip / masks; add `in_cave`=(mode==$B), `in_cellar`=(mode∈{9,$A}) bits | ASM Z_07:UpdateMode_JumpTable, InitMode_JumpTable; v1 frame_skip_wrapper.py MODE_* |
| `$13` GameSubmode | sub-step within mode | categorical | both | no — transition plumbing | ASM Variables.inc |
| `$EC` NextRoomId | room being scrolled to | categorical(128) | both, only in mode 6/7 | no | ASM Variables.inc |
| `$526` CaveSourceRoomId | OW room the dungeon/cave was entered from ($FF none) | categorical(128) | dungeon/cave | no — runner bookkeeping | ASM Z_05 L7405, Z_07:InitMode3_Sub1 |
| `$527` CellarSourceRoomId | UW room the cellar was entered from | categorical(128) | cellar | no | ASM Variables.inc, Z_05 L6729 |
| `$5A` UndergroundExitType | nonzero while stepping out of cave/dungeon/cellar; blocks re-entering (`CheckWarps`) | bitfield/flag | OW, transitions | maybe — v1 uses to mask cave re-entry | ASM Z_05:CheckWarps; v1 data.json `underground_exit_type` |
| `$620` CurRoomHistoryIndex, `$621–$626` RoomHistory | ring of last 6 rooms visited (searched `LDY #$05`); used to restore partial kill counts | categorical(128)×6 | both | no — use explicit room flags instead | ASM Z_05:ModifyObjCountByHistoryOW/UW; v1 ram.md |
| `$6BB1` LevelInfo_LevelNumber, `$6BAD` StartRoomId, `$6BAE` TriforceRoomId, `$6BBC` BossRoomId, `$6BB2..` CellarRoomIdArray (SRAM) | current level's static info (ss `1_*`: start $73, triforce $36, boss $35; ss `2_7dn`: start $7D, triforce $0D, boss $0E) | categorical | dungeon (OW block has start $77) | no — static; use for `is_boss_room`, `is_triforce_room` derived bits (maybe) | ASM Variables.inc; ss |
| `$6BAF/$6BB0` LevelInfo_WorldFlagsAddr (SRAM) | pointer to this level's 128-byte room-flag block: OW `$067F`, L1–L6 `$06FF` (ss); L7–9 presumably `$077F` | pointer | both | no — decoder input | ASM Z_07:GetRoomFlags; ss |

## 2. Link

| Address(es) | Meaning | Encoding | Valid when | Include? | Source |
|---|---|---|---|---|---|
| `$70` ObjX[0] | Link x (pixels) | position (÷255) | both | yes | v1 gd.txt `link_x` |
| `$84` ObjY[0] | Link y | position ((y−64)/176) | both | yes | v1 gd.txt `link_y` |
| `$98` ObjDir[0] | facing: 1 E, 2 W, 4 S, 8 N | categorical(4) one-hot | both | yes | v1 zelda_enums.Direction; ram.md |
| `$394` ObjGridOffset[0] | signed distance into current 8-px grid step (v1 `link_grid_offset`, i1); ≠0 bypasses tile collision | ordinal (÷8, signed) | both | yes — sub-tile offset | ASM Z_07 L2874 (via v1 room.py); data.json |
| `$3A8` ObjPosFrac[0] | sub-pixel | ordinal (÷256) | both | no | v1 ram.md |
| `$3F8` ObjInputDir[0] | current input direction bits | bitfield(4) | both | no — duplicates last action | ASM Z_04:UpdateBlock0Idle |
| `$AC` ObjState[0] | Link state; `(v & $C0)==$40` = halted (can't act/take item); v1 treats bit `$40` as "stunned" and waits for `==0` after attacks | bitfield (bits 6,7) + low nibble categorical | both | maybe — skipped frames make it mostly 0 | ASM Z_01:TryTakeRoomItem; v1 frame_skip_wrapper.py STUN_FLAG |
| `$C0` ObjShoveDir[0] | knockback direction (bit7 = fresh) | bitfield | both | yes — knockback state | ASM Variables.inc; ram.md; data.json `link_shove_dir` |
| `$D3` ObjShoveDistance[0] | knockback remaining | timer (÷$20 [unverified max]) | both | yes | ASM Variables.inc; ram.md |
| `$4F0` ObjInvincibilityTimer[0] | Link i-frames after hit (counts every 2 frames) | timer (÷$30 [unverified]) | both | yes | ASM Variables.inc, Z_05 L2060 comment, Z_07 L1879 |
| `$28` ObjTimer[0] | Link's object timer, e.g. $20 block after unlocking door | timer (÷$40) | both | maybe | ASM Z_05 L4005; data.json `link_obj_timer` |
| `$53` DoorwayDir | direction of doorway Link is standing in (UW); 0 = none; restricts movement to that axis | categorical(5) | dungeon | yes — cheap and affects legal moves | ASM Z_05 L1622, Link_ModifyDirInDoorway; data.json `doorway_dir` |
| `$512` LinkParalyzed | grabbed by Like-Like | bool | both | yes | ASM Z_04 L6866, Z_07 L2094 |
| `$52E` SwordBlocked | sword disabled (red bubble) | bool | dungeon | yes — decoded (`sword_blocked`); masks SWORD and BEAMS (`GameView::is_sword_blocked`) | ASM Z_04 L1171, Z_05 L6937; ram.md |
| `$4C` SwordBlockedLongTimer | timer before sword re-enabled (non-zero: blocked) | timer | dungeon | yes — decoded (`sword_blocked_timer`); masks SWORD and BEAMS | ASM Variables.inc; ram.md `$4C` |
| `$66F` HeartValues | hi nibble = containers−1, lo nibble = full hearts | ordinal: max_hp=(hi+1)/16, hp | both | yes — hp/max_hp and max_hp | ASM Variables.inc comment; v1 link.py |
| `$670` HeartPartial | 0 empty, $01–$7F half, $80–$FF full partial | ordinal (0/0.5/1) | both | yes — folded into hp float | v1 link.py `health` |
| derived: beams ready | sword>0 ∧ (hi==lo nibble of `$66F`) ∧ `$670`≥$80 ∧ beam slot `$BA`==0 | bool | both | yes | v1 link.py `_is_health_full_for_beams` (cites Z_07 L4632) |
| `$B9` ObjState[$D] | sword (melee) slot state; 0 idle | categorical(small) / bool active | both | maybe — "attacking" bit | v1 gd.txt `sword_animation`; frame_skip_wrapper.py |
| `$BA` ObjState[$E] | beam/rod shot: 0 none, $10 flying, $11 spreading, $80 rod shot | categorical | both | yes — beam in flight blocks beams | v1 link.py ANIMATION_*; ram.md |
| `$BB` ObjState[$F] | boomerang $10–$57 / bait $80–$82 | categorical (active bool) | both | maybe | v1 link.py; ram.md |
| `$BC,$BD` ObjState[$10,$11] | bomb $11 placed (one update), $12 lit, $13/$14 exploding; fire $21 moving, $22 standing | categorical | both | yes — bomb active/exploding (needed for bomb walls) | v1 link.py; ram.md; ASM Z_01 WieldBomb, Z_07 UpdateBomb |
| `$BE` ObjState[$12] | arrow $10 flying, $20/$28 hit; rod $31–35 | categorical | both | maybe | v1 link.py; ram.md |
| `$7D–$82`, `$91–$96`, `$A5–$AA` | x, y, dir of slots $D–$12 (Link weapons) | position | both | maybe — weapon entities for later | ASM ObjX/ObjY/ObjDir + slot; ram.md |
| `$56,$57` Link_GoStraight* | input smoothing flags | bool | both | no | ASM Variables.inc |

## 3. Inventory & items

Items block `Items=$657`, inventory slot `s` is at `$657+s` (`ASM Z_01:ItemIdToSlot`).

| Address(es) | Meaning | Encoding | Valid when | Include? | Source |
|---|---|---|---|---|---|
| `$657` sword | 0 none, 1 wood, 2 white, 3 magic | categorical(4) | both | yes | ram.md; v1 SwordKind |
| `$658` InvBombs | bombs | ordinal (÷`$67C`, default 8) | both | yes | ram.md |
| `$67C` MaxBombs | bomb capacity | ordinal (÷16) | both | no — constant on route | ram.md |
| `$659` InvArrow | 0/1 wood/2 silver | categorical(3) | both | yes (items-owned bits) | ram.md |
| `$65A` Bow | bool | bitfield | both | yes | ram.md |
| `$65B` InvCandle | 0/1 blue/2 red (ram.md; v1 enum names reversed) | categorical(3) | both | yes | ram.md, ItemIdToSlot ids $06/$07 |
| `$65C` whistle, `$65D` food, `$65E` potion(0/1/2), `$65F` rod, `$660` raft, `$661` book, `$662` ring(0/1/2), `$663` ladder, `$664` magic key, `$665` bracelet, `$666` letter, `$66C` InvClock | owned flags | bitfield / small categorical | both | yes — items-owned bits (cheap) | ram.md; v1 gd.txt |
| `$66C` InvClock (also) | **clock active in this room**: freezes monsters, Link invulnerable; cleared on room load | bool | both | yes | ASM Z_04 (all `LDA InvClock` checks), Z_07 L1488 |
| `$674/$675` boomerang / magic boomerang | owned | bitfield | both | yes | ram.md |
| `$676` InvMagicShield | owned (blocks more projectiles) | bool | both | yes | ram.md |
| `$66D` InvRupees | rupees | ordinal (÷255) | both | yes (scaled) | ram.md |
| `$67D/$67E` RupeesToAdd/Subtract | pending rupee change | ordinal | both | no | ram.md; v1 zelda_game.py |
| `$66E` InvKeys | keys held | ordinal (÷8, clip) | both | yes — key doors | ram.md |
| `$667` InvCompass, `$668` InvMap | **bit (level−1)** set = owned for levels 1–8 | bitfield → derive `has_compass/has_map` for current level | dungeon | yes (derived bit) | ASM Z_05:HasCompass/HasMap. **v1 bug**: link.py returns the raw byte (true if any level's compass) |
| `$669/$66A` InvCompass9/InvMap9 | level 9 | bool | L9 | no (out of milestone) | ASM Z_05:HasCompass |
| `$671` InvTriforce | bit (level−1) per piece | bitfield(8) | both | yes — "D1 done" bit matters for the route | ram.md; v1 link.py `has_triforce` |
| `$672` LastBossDefeated | v1 calls it triforce_of_power | bool | L9 | no | Variables.inc; v1 gd.txt |
| `$656` SelectedItemSlot | B-item: 0 boomerang, 1 bombs, 2 arrows, 4 candle, 5 whistle, 6 food, 7 potion, 8 rod | categorical(10) | both | yes | v1 SelectedEquipmentKind |
| `$513` UsedCandle | candle already used this room | bool | both | no (not on route) | ASM Z_01 L4013 |

## 4. Object slots (tables indexed by slot)

v1 reads slots 0–$B (12 bytes) of each table (gd.txt `[tables]`); add slot `$13` for the room item (see §R.4).

| Address(es) | Meaning | Encoding | Valid when | Include? | Source |
|---|---|---|---|---|---|
| `$34F+i` ObjType | object type: `$01–$48` monsters (v1 ZeldaEnemyKind), `$49/$4A` traps, `$4B–$52` UW persons, `$53–$5C` monster shots (`$55/$56` fireballs, `$5B` monster arrow, `$5C` boomerang), `$5D` dead dummy, `$60` **dropped item**, `$61` dock, `$62/$65/$66` rock/gravestone, `$63/$67` rock wall (OW bombable), `$64` tree (burnable), `$68` **push block**, `$6A+` cave persons. **`$34F` (slot 0) = RoomKillCount** | categorical(~$80) → embedding | both | yes — entity type | ASM Z_07:UpdateObject_JumpTable, CommonVars.inc; v1 zelda_game.py `_is_id_enemy/_is_projectile` |
| `$70+i`, `$84+i` ObjX/ObjY | position | position | both | yes | v1 gd.txt obj_pos_x/y |
| `$98+i` ObjDir | facing (1/2/4/8, diagonals = sums) | categorical(9) | both | yes | ram.md; v1 enemy.py |
| `$AC+i` ObjState | per-type state; for `ObjType=$60` it is the **dropped item id** (Item_ObjItemId); Leever 3 = surfaced, Zora 2–4 = surfaced, Wallmaster 1 = active; block 0/1/2 idle/moving/done | categorical (per type) | both | yes — as "targetable/active" bit + item id for drops | v1 enemy.py `is_active`; ASM CommonVars.inc; Z_04:UpdateBlock |
| `$405+i` ObjMetastate | 0 = normal; 1–4 = spawn cloud (**not yet spawned**); `$10–$13` dying sparkle, `$14` → becomes item | categorical(3): none/spawning/dying | both | yes — spawning/dying bits | ASM Z_05 L1690, Z_07:UpdateMetaObject; v1 enemy.py |
| `$492+i` ObjUninitialized | $FF = slot not yet initialized | bool | both | no — plumbing | ASM Z_05 L1684 |
| `$485+i` ObjHP | HP in **high nibble** (v1 `>>4`); ss 1_74w stalfos `$20`, 1_44c goriya `$30` | ordinal (hi nibble ÷15) | both | yes | v1 zelda_game.py `_build_enemy`; ram.md; ss |
| `$3D+i` ObjStunTimer (slot 1 = `$3E`) | boomerang/clock stun | timer (÷ observed max) | both | yes | v1 gd.txt obj_stun_timer; ASM Z_04 InvClock/Stun checks |
| `$28+i` ObjTimer | per-object action/shoot delay | timer (÷$FF) | both | maybe — predicts shots | ASM Variables.inc; ram.md `$29–$2E` |
| `$3A8+i` ObjPosFrac / Item_ObjItemLifetime | sub-pixel; **for items: lifetime**, ≥$F0 = flashing & not pickable | timer (÷$FF) | items only (`ObjType=$60` or slot $13) | maybe — item despawn | ASM Z_01:TryTakeItem, Z_07:AnimateItemObject; v1 gd.txt `item_timer` |
| `$C0+i`, `$D3+i` ObjShoveDir/Distance | knockback of monsters | bitfield / timer | both | maybe | ASM Variables.inc |
| `$4F0+i` ObjInvincibilityTimer | monster i-frames | timer | both | maybe | ASM Variables.inc |
| `$4B2+i` ObjInvincibilityMask | weapon immunities: 1 sword, 2 boomerang, 4 arrow, 8 bomb, $10 rod beam, $20 fire | bitfield(6) | both | yes — "killable by X" flags | ram.md (fiskbit `dynamic_immunities`) |
| `$4BF+i` ObjAttr | 1 custom collision, 2 thin frame, 4 custom sprite, 8 attr override, $20 no weapon collision, $40 thin (8×16), $80 no reverse on Link hit | bitfield(7) | both | maybe — "hurts-on-touch/killable" proxy | ram.md (fiskbit `dynamic_properties`); ASM Z_07:InitTileObjOrItem |
| `$49E+i` ObjCollidedTile | last tile object collided with | categorical(256) | both | no | ASM Variables.inc |
| `$394+i`, `$3BC+i`, `$3D0+i`, `$3E4+i`, `$3F8+i`, `$380+i` | grid offset, speed, anim counter/frame, input dir, moving limit | various | both | no — use Δposition instead | ASM Variables.inc |
| `$412–$478` (+i) | per-type scratch (Wallmaster step, Gohma eye, Aquamentus fireball offset, push-block push timer `$41D` etc.) | per type | both | no (generic); maybe boss-specific later | ASM ObjVars.inc |
| `$486–$490` | v1 `obj_health_1..b` | — | — | — | **v1 bug**: gd.txt names `0x491/0x492` as health_b/c; `$492` is ObjUninitialized. data.json uses `0x490/0x491`. Use `$485+i`, i≤$B |
| `$627` WorldKillCount | kills since Link last took damage (v1 `kill_streak`); 16th kill → fairy drop | ordinal (÷16) | both | maybe | ASM Z_01 L5766 (reset on damage), Z_04 L11150 |
| `$50` HelpDropCount, `$51` HelpDropValue | kills toward guaranteed drop (10 → 5 rupees, or bomb if killed by bomb) | ordinal / bool | both | no | ASM Z_01 L6008–6021, Z_04 L11156–11167 |
| `$52A` WorldKillCycle | 0–9 column in DropItemTable | categorical(10) | both | no | ASM Z_07 L5493, Z_04 L11143 |
| `$34C` ActiveMonsterShots, `$514` ZoraActive, `$510` ActiveRedLeeverCount, `$515` ActiveBoulders | counts | ordinal | per type | no — derivable from slots | ASM Variables.inc |

## 5. Room state (dungeon = UW, overworld = OW)

| Address(es) | Meaning | Encoding | Valid when | Include? | Source |
|---|---|---|---|---|---|
| `flags[room]` = `$067F`/`$06FF`+RoomId (via `LevelInfo_WorldFlagsAddr`) | per-room history byte, **layout differs by mode** — §R.1 | bitfield(8) | meaning changes by mode | yes — core of Room-state block | ASM Z_07:GetRoomFlags, MarkRoomVisited; Z_05:SaveKillCountUW/OW |
| `$687E+r` LevelBlockAttrsA … `$6AFE+r` …F (SRAM) | static per-room attributes A–F of the current level block — §R.2 | decoded to categoricals | meaning changes by mode | yes (decoded fields) | ASM Variables.inc; Z_05:FindDoorTypeByDoorBit, CreateRoomObjects |
| `$4CD` LevelBlockAttrsByteF | copy of attr F for the current room | as F | both | no — use F | ASM Z_05 L1557 |
| derived door type ×4 (N,S,W,E) | from A/B: open/wall/false wall/false wall 2/bombable/key/key 2/shutter | categorical(8) ×4 | dungeon | yes | ASM Z_05:FindDoorTypeByDoorBit, LayOutDoors table |
| derived door open ×4 | `flags & dirbit` (key/bomb persisted) OR `CurOpenedDoors & dirbit` (incl. shutters this visit) | bool ×4 | dungeon | yes | ASM Z_05:AddDoorFlagsToCurOpenedDoors, CalcOpenDoorwayMask |
| `$EE` CurOpenedDoors | doors currently open in this room, bits E1 W2 S4 N8 (only true doors; open/walls cleared) | bitfield(4) | dungeon (garbage in OW: ss 0_37s=$08) | yes (via derived door-open) | ASM Z_05:LayOutDoors, SetEnteringDoorwayAsCurOpenedDoors; data.json |
| `$54` TriggeredDoorCmd, `$55` TriggeredDoorDir | door animation in progress: 2→3→4 closing, 6→7→8 opening; Dir = bit | categorical(3) / bitfield | dungeon | maybe — "door animating" bool | ASM Z_05 L1631, TriggerOpenDoor, L5139–5211 |
| `$4CE` ShutterTrigger | shutters have been triggered to open | bool | dungeon | maybe | ASM Z_05:CheckShutters, TriggerShutters |
| `$27` DoorTimer | door animation step timer | timer | dungeon | no | ASM Z_05 L5150 |
| `$AB` RoomItemId (= ObjDir[$13]) | room item id (§R.4). **Stale when already taken** (not rewritten; ss 1_53w shows `$03` while attr E=`$19`) | categorical(36) | both (OW: only heart container in OW room $5F) | yes, but take from attr E masked by taken-flag | ASM Z_05:CreateRoomObjects, Z_07:MoveAndDrawRoomItem |
| `$BF` ObjState[$13] | room item active: `$00` = visible/collectable, `$FF` = hidden (not yet dropped, none, or taken) | bool | both | yes — "treasure visible" | ASM Z_05:CreateRoomObjects, Z_01:TryTakeItem; v1 `treasure_flag` |
| `$83`/`$97` ObjX/Y[$13] | room item position (copied from slot 1 each frame when carried) | position | when present | yes | ASM Z_01:GetShortcutOrItemXYForRoom, Z_07:AnimateRoomItemOnMonster; v1 `treasure_x/y` |
| `$3BB` Lifetime[$13] | room item flash/lifetime | timer | when present | no | ASM Z_07:AnimateItemObject |
| derived `item_carried` | room item present ∧ ObjType[1] ∈ {$17 LikeLike, $2A Stalfos, $30 Gibdo} → drawn on monster 1 (D1 room $74 key) | bool | dungeon | yes | ASM Z_07:MoveAndDrawRoomItem |
| `$34E` RoomObjCount | monsters created on entry (= FoeCount − remembered kills; 1 for list ids $32–$61; bosses may set 8) | ordinal (÷8, clip) | both | yes (as part of enemies-left) | ASM Z_05 L1696–1753, Z_04 L4839/L9523 |
| `$34F` RoomKillCount (= ObjType[0]) | kills in this room this visit | ordinal | both | yes (enemies-left = count − kills) | ASM CommonVars.inc; v1 `room_kills` |
| `$34D` RoomAllDead | ≠0 when no live monsters (bubbles excluded); **incremented every check, so it wraps** (ss: $04, $18, $DA) — treat as `!=0` and prefer slot-derived count | bool | both | maybe (derive instead) | ASM Z_05:CheckHasLivingMonsters |
| `$35F` RoomObjTemplateType | monster list/template id for the room | categorical(128) | both | maybe — redundant with room id | ASM Z_05 L1778, SaveKillCountUW |
| `$560+RoomId` LevelKillCounts | exact partial kills per room in the current level ($F = all); cleared on level load | ordinal (÷8) | dungeon | yes (enemies-left for revisits) | ASM Z_05:SaveKillCountUW, Z_07:InitMode2 |
| `$4CF` BlockPushComplete | 0 not pushed, 1 pushed (secret not yet handled), 2 pushed + handled | categorical(3) | dungeon | yes | ASM Z_04:UpdateBlock1Moving, Z_05:CheckSecretTriggerBlockStairs |
| `$AC+$B` = `$B7` ObjState[$B] (when ObjType[$B]=$68) | push block 0 idle, 1 moving, 2 done; position `$7B/$8F` | categorical(3) + position | dungeon | yes (as entity + flag) | ASM Z_04:UpdateBlock, Z_05:FindAndCreatePushBlockObject |
| `$41D` ObjPushTimer[$B] | frames Link has pushed (block moves at $10) | timer (÷16) | dungeon | maybe | ASM Z_04:UpdateBlock0Idle |
| `$52B/$52C/$52D` RoomTileObjType/X/Y | OW tile object (armos/rock) being activated | categorical / position | OW | no | ASM Variables.inc |
| `$51A` SecretColorCycle, `$51B` UsedFlute | flute/pond secret progress | ordinal | OW | no (not on route) | ASM Z_07 L5918 |
| `$52F` MazeStep | lost woods/mountain maze progress | ordinal | OW | no | ram.md; ASM Variables.inc |
| `$4CC` PersonFireballsEnabled | UW person shoots | bool | dungeon | no | ASM Variables.inc |

## 6. Tiles / layout

| Address(es) | Meaning | Encoding | Valid when | Include? | Source |
|---|---|---|---|---|---|
| `$6530–$67EF` PlayAreaTiles (`ram()` index `0xD30`, len `$2C0`) | 32 cols × 22 rows of 8-px tiles, **column-major** (`tiles[x*22+y]`); row r covers y = $40+8r | categorical(256) → tile class map (below) | both | yes — Room grid | ASM Variables.inc, Z_05:FillWalls/FindAndCreatePushBlockObject; v1 zelda_game.py `current_tiles` |
| `$34A` ObjectFirstUnwalkableTile | walkability threshold: OW `$89`, UW `$78`; becomes `$99` after pond secret | ordinal — decoder input | both | no (input) | ASM Z_05:ObjectRoomBoundsOW/UW, Z_07 L5946; ss |
| `$346–$349` RoomBoundLeft/Right/Up/Down | OW `$11,$E0,$4E,$CD`; UW `$21,$D0,$5E,$BD` | position — decoder input | both | no (input; for masks) | ASM Z_05:ObjectRoomBoundsOW/UW; ss |
| attr D & `$3F` (UW) / `$7F` (OW) | unique room layout id (column template) | categorical(64/128) | meaning changes by mode | maybe — "layout template ID" embedding | ASM Z_07:GetUniqueRoomId, Z_05 L5785 (OW keeps 7 bits) |
| `$530` PlayAreaAttrs | palette attrs | — | both | no | ASM Variables.inc |

**Tile → class rules** (apply in this order; values from asm and ss tile dumps of `1_42e`, `1_35e`, `0_77c`):

| Class | Rule | Source |
|---|---|---|
| walkable | `t < threshold` (`$34A`); OW also `t ∈ {$8D,$91,$9C,$AC,$AD,$CC,$D2,$D5,$DF}` | ASM Z_07:WalkableTiles, GetCollidingTileMoving |
| stairs | `$70–$73` (UW stair tile `$70`; OW revealed stairs `$70`) | ASM Z_05:CheckSecretTriggerBlockStairs, Z_04:RevealAndFlagSecretStairsObj, Z_05:UpdateMode10Stairs_Full |
| cave/dungeon entrance | OW `$24` (black, bottom) under `$F3` (top) | ASM Z_05 L1595 ($24 = black entrance); ss 0_77c; v1 room.py |
| water | OW `$8D ≤ t < $99`; UW `$F4` (also `≥$F4` walls per Wizzrobe comment) | ASM Z_07 L3290–3302, Z_04 L1810, L7109–7115 |
| block (solid) | `$B0–$B3` (any block/statue square). The *pushable* one is the first `$B0` in row `$A` from col 4 when attr D bit 6 set; it is ObjType `$68` in slot `$B` | ASM Z_05:FindAndCreatePushBlockObject, Z_04 L7109 |
| armos/gravestone | `$BC–$C3` (OW) | ASM Z_01:CheckPassiveTileObjects |
| UW floor | `$74–$77` (`$74` written when block leaves) | ASM Z_04:UpdateBlock0Idle |
| door | **do not classify from tiles**; use derived door type/open state stamped onto the 4 door cells: N cols 14–17 rows 0–3; S cols 14–17 rows 18–21; W cols 0–3 rows 9–12; E cols 28–31 rows 9–12 (open doorway shows `$24` at cols 29/2, rows 10–11) | ss 1_42e/1_35e; v1 room.py `*_DOOR_TILE` |
| shutter (closed) tiles | `$AC–$AF` | v1 room.py BARRED_DOOR_TILES; ss 1_42e W door |
| locked door tiles | `$98–$A7` | v1 room.py DOOR_TILES |
| wall | everything else `≥ threshold` | ASM Z_07:GetCollidingTileMoving |

Recommended grid: 16×11 of 16-px squares (row r covers y=$40+16r), class = max-priority class of its 2×2 tiles; Link's collision samples the *higher* of two tile values (v1 room.py `_check_vertical`, ASM Z_07 L2264–2275), so 8-px resolution (32×22) is the exact one. Choose per v1's movement port.

## 7. Timers / RNG

| Address(es) | Meaning | Encoding | Valid when | Include? | Source |
|---|---|---|---|---|---|
| `$15` FrameCounter | frame counter | — | both | no (leaks time; use History) | ASM Variables.inc; ram.md |
| `$18–$24` Random | LFSR RNG bytes, 13 (rng_0..12; ScrambleRandom rotates $18–$24, monsters read `Random+1,X` up to $24) | — | both | no — load randomization redraws all 13 (`RNG_VARS`) | v1 gd.txt; ASM Variables.inc, Z_07 ScrambleRandom |
| `$3C` FluteTimer | freeze while flute plays | timer | both | no | ASM Z_07:UpdateMode5Play |
| `$26` StunCycle | stun pacing | — | both | no | ASM Variables.inc |
| `$4A`, `$4B`, `$4D` long timers | chase / edge-spawn / red-leever | timer | per type | no | ASM Variables.inc |
| `$60` ChaseOtherTarget, `$61`/`$62` ChaseTargetX/Y | **where monsters chase** ("shadow Link", Lee). With `$60`=0, current Link X/Y are copied before weapon/bait updates. When `$4A` expires (0–7 ticks, one per 10 frames), `$60` flips; on 0→1, if target X still equals current Link X, both target bytes XOR `$FF`. Without bait this normally mirrors the current position even if Link moved: the disassembly's "last frame" comment is misleading. On 1→0 the old target survives that update. Bait can override actual coordinates without changing the flag. Mirrored Y may be outside the play area; use full-byte coordinates, not an inferred identity | position + refresh bool | both | candidate (round 7 audit): actual monsters' goal; no RNG required | ASM Z_07 L1888–1951 (instruction order), L3788–3811 (bait), L481–507 (timer); Z_04 chase movement |
| `$524` SpawnCycle, `$525` CurEdgeSpawnCell | monster spawn pacing / edge cell | categorical | both | no | ASM Variables.inc, Z_05 L3509 |

## 8. Misc

| Address(es) | Meaning | Encoding | Valid when | Include? | Source |
|---|---|---|---|---|---|
| `$605` Tune0 | sfx: 1 parry (shield block, Darknut shield, boomerang on an immune monster), 2 monster hit (also a boomerang stun), 4 magic, 8 key pickup, $10 heart pickup, $20 set bomb, $40 low-heart warning | bitfield | both | no (obs) — reward/event signal | ram.md; v1 SoundKind; ASM Z_01 L5711-5715, L5900-5979 |
| `$607` Tune1 | 1 rupee, 2 **item appears / key appears**, 4 **secret revealed**, 8 item taken (every item but $0E), $10 whistle, $20 enemy burst, $40 continue, $80 death | bitfield | both | no (obs) — reward/event signal | ram.md; ASM Z_05 L2366, Z_04 L749, Z_01 TakeItem L4462-4464 |
| `$E0` Paused, `$E1` MenuState | pause / submenu | bool | both | no | ASM Variables.inc |
| `$F8/$FA` ButtonsPressed/Down | controller | — | both | no | ASM Variables.inc |
| `$5E` SubmenuScrollProgress, `$FC/$FD` scroll | screen scroll | — | transitions | no | ASM Variables.inc |
| `$E5` DeathModeCounter | death spin | — | death | no | ASM Variables.inc |
| `$62D+slot` QuestNumbers, `$630+slot` DeathCounts, `$16` CurSaveSlot | profile | — | both | no | ASM Variables.inc |
| `$413/$422/$430` CaveFlags/CaveItemIds/CavePrices | cave wares (item id low 6 bits, flags hi 2) | categorical | cave (mode $B) | maybe — sword cave only needs "item taken" flag | ASM Z_01:InitCaveContinue |
| `$522` WhirlwindTeleportingState, `$523` TeleportingLevelIndex | whistle warp | — | OW | no | ASM Variables.inc |
| `$64` LadderSlot | ladder placement | — | both | no | ASM Variables.inc |

---

# R. Room-state decoding (bit level)

Notation: `r` = RoomId (`$EB`), `S(A)` = SRAM byte `ram()[A-0x5800]`, `flags = ram[wfa + r]` with `wfa = S($6BAF) | S($6BB0)<<8`.

## R.1 Room flags byte (`GetRoomFlags`)

| Bit | Dungeon (UW) | Overworld (OW) | Set by / read by |
|---|---|---|---|
| 0 | door **E** opened (key or bombed) | kill count bit 0 | UW: `Z_05:SetDoorFlag` (mask `LevelMasks[dirIndex]`, 0=E,1=W,2=S,3=N); OW: `Z_05:SaveKillCountOW` (`AND #$07`) |
| 1 | door **W** opened | kill count bit 1 | same |
| 2 | door **S** opened | kill count bit 2 | same |
| 3 | door **N** opened | — (no reader found) | `Z_05:CalcOpenDoorwayMask`, `AddDoorFlagsToCurOpenedDoors` |
| 4 | **room item taken** | **room item / cave item / secret reward taken** (e.g. sword cave room $77 → `$10`, ss 0_67w) | `Z_01:SetRoomFlagUWItemState` (`ORA #$10`), `GetRoomFlagUWItemState` (used in OW caves too, Z_01 L98/L824) |
| 5 | **visited** (dungeon map) | — (only set when `CurLevel≠0`) | `Z_07:MarkRoomVisited` (`ORA #$20`), `Z_05:Submenu_WriteScanningMapRoomMark` |
| 6–7 | **kill count**: 0–2 partial kills (capped), **3 = all killed** | bit 7 = **secret found** (cave bombed/burned/stairs revealed) | UW `Z_05:SaveKillCountUW` (`AND #$3F`, `ROR`×3, `#$C0`); OW `Z_04:RevealAndFlagSecretTileObj` (`ORA #$80`), `Z_05:CheckShortcut`, `LayoutRoomOrCaveOW` (`AND #$80`) |

Observed (old savestate `1_35e`, since deleted, level-1 block): `$74:B0` (kills=2-of-cap, visited, key taken), `$53:F0` (all dead, visited, taken), `$35:F4` (boss dead, visited, heart taken, S door opened), `$52:68` (1 kill, visited, N opened), `$43:21` (visited, E opened). OW (old `0_67w`): `$67:07` (OW kill count 7 = all), `$77:10` (sword taken). The route-generated chain (`1_36w_2` walked back into 0x35, `tests/test_room_state.py`) reads `$53:F0`, `$35:F4`, `$74:70`, `$52:A8`, `$43:71`; the differences are kills and pickups taken in a different order.

**Opening a key/bomb door sets the flag on both sides immediately** (`Z_05` L5188–5205: `SetDoorFlag` then `ORA LevelMasks, X^1` into room `r + DoorNextRoomIdOffsets[X]`). Shutters are never persisted (L5184 skips type 7). Bombable doors, once flagged, render as a hole (door face 9, `Z_05:LayOutDoors`).

**Kill persistence.** On leaving a room `SaveKillCount{UW,OW}` stores kills; on entry `ModifyObjCountByHistory{UW,OW}` subtracts:
- UW: if room ∈ RoomHistory (last 6) or `flags[6:7]≠3`: `count = FoeCount − LevelKillCounts[r]` (`$560+r`). If `flags[6:7]==3` and not in history: recurring monsters respawn (flag & `LevelKillCounts` reset) unless the template id is "non-recurring" (`$32 ≤ id < $49`, except `$3A,$3B`) → count 0.
- OW: kill count lives only in `flags[0:2]` (7 = all); if not in history and 7 → reset to 0 (monsters respawn).

## R.2 Level-block attribute bytes (`LevelBlockAttrsA..F`, SRAM `$687E/$68FE/$697E/$69FE/$6A7E/$6AFE` + r)

| Byte | Bits | Dungeon (UW) | Overworld (OW) | Source |
|---|---|---|---|---|
| A | 7–5 | **N door type** | 7–4: x where Link exits underground (`& $F0`) | Z_05:FindDoorTypeByDoorBit; Z_05 L1577 |
| A | 4–2 | **S door type** | bit 3 zora present, bit 2 sea sound | same; Z_04 L1760, Z_07 L2034 |
| A | 1–0 | outer palette | outer palette | Z_05 L993 |
| B | 7–5 | **W door type** | 7–2: cave index (`(B&$FC)`: <$40 → level `B>>2`; $40+ → cave, $50 → shortcut, dweller ObjType `$6A+((B&$FC)−$40)/4`) | Z_05:FindDoorTypeByDoorBit; Z_05 L7376–7406, L1977–1988 |
| B | 4–2 | **E door type** | (part of cave index) | same |
| B | 1–0 | inner palette | inner palette | Z_05 L1002 |
| C | 7–6 | foe-count index → `LevelInfo_FoeCounts[$6BA2+idx]` (L1: 3,5,6,8; OW: 1,4,5,6 — ss) | same | Z_05 L1712–1721 |
| C | 5–0 | monster list id low 6 bits | same | Z_05 L1698–1708 |
| D | 7 | monster list id bit 6 | same | Z_05 L1702–1708 |
| D | 6 | **push block present** | layout id bit 6 | Z_05:CreateRoomObjects (`AND #$40`); Z_05 L5785 |
| D | 5–0 | unique room layout id (`$3E/$3F` = cellars) | layout id bits 0–5 | Z_07:GetUniqueRoomId; Z_05 L2070 |
| E | 7 | dark room | (cave block: E holds cave item ids + flags) | Z_05:IsDarkRoom_Bank5; Z_01 L146 |
| E | 6–5 | boss ambient sound index | — | Z_05:CheckBossSoundEffectUW |
| E | 4–0 | **room item id**; `$03` means **none** | — | Z_05:CreateRoomObjects |
| F | 7–6 | — | quest-secret selector | Z_04:IsQuestSecretMismatch |
| F | 5–4 | **item position index** → `LevelInfo_ShortcutOrItemPosArray[$6BA7+idx]` | shortcut/stairs position index | Z_01:GetShortcutOrItemXYForRoom |
| F | 3 | — | monsters enter from screen edges | Z_05 L1892, Z_07 L5532 |
| F | 2–0 | **secret trigger** (R.3) | square row Link exits underground | Z_05:CheckUnderworldSecrets; Z_05 L1580 |

Door types (`Z_05:LayOutDoors` comment table):

| Value | Type | Passable when | Persisted open in flags? |
|---|---|---|---|
| 0 | open | always | n/a (bit cleared from CurOpenedDoors) |
| 1 | wall | never | n/a |
| 2 | false wall | treated as wall by map/doorway mask [walk-through behavior unverified] | n/a |
| 3 | false wall 2 | as 2 [unverified] | n/a |
| 4 | **bombable** | after bomb (`Z_07` L4908–4925 triggers open cmd) | yes |
| 5 | **key** (locked) | after key (`Z_05` L3989–4006: magic key or `DEC InvKeys`) | yes |
| 6 | key 2 | as key [difference from 5 unverified] | yes |
| 7 | **shutter** | when room trigger fires (R.3) — this visit only; entering through an opened door issues close cmd 2 (`Z_05` L1625–1631) | no |

Suggested per-direction door feature: `type ∈ {none(wall/false), open, key, bomb, shutter}` one-hot + `is_open` bit, where `is_open = type==open ∨ (flags & dirbit) ∨ (CurOpenedDoors & dirbit)`.

## R.3 Secret trigger (attr F bits 2–0, UW) — "what makes the room do something"

| Value | Name (`CheckSecretTrigger_JumpTable`) | Effect |
|---|---|---|
| 0 | none | — |
| 1 | all dead | shutters open when no monsters (bubbles ignored) |
| 2 | ringleader | killing monster in slot 1 kills all |
| 3 | last boss | shutters open when `LastBossDefeated`; room item hidden until then |
| 4 | block → door | shutters open when `BlockPushComplete≠0` |
| 5 | block → stairs | when block pushed, stairs tile `$70` appear at ($D0,$60) via slot $B; `BlockPushComplete` 1→2 |
| 6 | money or life | shutters open when person (slot 1) gone |
| 7 | **all dead → item** ("clear-to-drop") | shutters open **and** room item activates (`ObjState[$13]=0`, tune1 `$02`) when all dead |

Sources: `Z_05:CheckUnderworldSecrets`, `CheckSecretTrigger*`, `CreateRoomObjects` (triggers 3 and 7 start the item hidden). Push block itself can only move when `RoomAllDead≠0` (`Z_04:UpdateBlock0Idle`), Link aligned within $10 px and pushing $10 frames.

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
`ItemPos` = `LevelInfo_ShortcutOrItemPosArray` (SRAM `$6BA7..$6BAA`; L1: `C9 AC 89 87`). OW: the only room item is the heart container in room `$5F` at ($C0,$90) (`CreateRoomObjects@MakeHeartContainerOW`); cave items use `flags & 0x10`.

Item ids (`Z_01:ItemIdToSlot` → inventory slot `$657+s`; names from slot table and v1 ZeldaItemKind):

| id | item | id | item | id | item | id | item |
|---|---|---|---|---|---|---|---|
| 00 | bombs (+4) | 09 | silver arrow | 12 | blue ring | 1B | triforce piece |
| 01 | wood sword | 0A | bow | 13 | red ring | 1C | magic shield |
| 02 | white sword | 0B | magic key | 14 | power bracelet | 1D | boomerang |
| 03 | magic sword / **"none" in room attr** | 0C | raft | 15 | letter | 1E | magic boomerang |
| 04 | food | 0D | ladder | 16 | **compass** | 1F | blue potion |
| 05 | whistle | 0E | slot `$672` | 17 | **map** | 20 | red potion |
| 06 | blue candle | 0F | 5 rupees | 18 | rupee | 21 | clock |
| 07 | red candle | 10 | magic rod | 19 | **key** | 22 | heart |
| 08 | arrow | 11 | book | 1A | **heart container** | 23 | fairy |

(`$0F` = 5 rupees per `Z_04` help-drop comment L11161; `$3F` = "none" for `RoomItemId`, `Z_07:MoveAndDrawRoomItem`.)

## R.5 Enemies remaining (incl. not yet spawned)

At room init every monster is written into `ObjType[1..count]` immediately (`Z_05` L1771–1777) with `ObjMetastate=1` (spawn cloud), so:
```
monsters    = [i for i in 1..0xB if 1 <= ObjType[i] < 0x49 and ObjType[i] not in (0x2B,0x2C,0x2D)]   # CheckHasLivingMonsters
unspawned   = [i in monsters if 1 <= ObjMetastate[i] <= 4]
dying       = [i in monsters if 0x10 <= ObjMetastate[i] <= 0x13]
remaining   = len(monsters) - len(dying)
```
The game counts $01–$2A and $2E–$48 as living (traps $49/$4A and persons $4B–$52 don't). `room_state.rs` counts
`ObjectClass::Monster` minus the guard fires $3F, so it deliberately also leaves out the hazards the game counts
(boulders $1F/$20, standing fire $40).
Cross-check: `RoomObjCount($34E) − RoomKillCount($34F)`. Persistent across visits: UW `flags[6:7]`, `LevelKillCounts[$560+r]`; OW `flags[0:2]`. Exceptions: Digdogger children reset `RoomObjCount` (Z_04 L5213); some bosses set 8 (Z_04 L4839, L9523); OW edge-spawn rooms (F bit 3) place monsters at edges later (`Z_07` L5532).

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
| 22 | wall wall wall key | – | none | **yes** | 1 (list `$4A` ∈ [$32,$62) → count forced to 1, `Z_05` L1723–1730) | — |

RAM agrees in ss `1_35e`: `CurOpenedDoors=$05` (S key door + E shutter), `flags[$35]=$F4`. (v1 lists `35` exits as "NE"; decode and RAM say S/E.)

---

# Recommended starting observation feature set (§4.2)

| Block | Rows |
|---|---|
| **Room identity** | `(CurLevel $10, RoomId $EB)` → embedding; optional layout id (attr D `&$3F` UW / `&$7F` OW) embedding; `in_cave`/`in_cellar` from `$12` |
| **Room state** | per direction ×4: door type one-hot {none, open, key, bomb, shutter} + is_open (R.2, `$EE`, flags bits 0–3); visited (bit 5); kill state (UW bits 6–7 one-hot, OW bits 0–2 ÷7); item: has_item, item id embedding, taken (bit 4), visible (`$BF==0`), hidden_until_clear (trigger 3/7), carried, item x/y; trigger one-hot(8); push block present, BlockPushComplete one-hot(3); enemies remaining ÷8, unspawned ÷8; RoomAllDead (derived); door animating (`$54≠0`); InvClock active; OW secret_found (bit 7); is_boss_room / is_triforce_room (LevelInfo) |
| **Link** | x, y, grid offset `$394`, facing one-hot `$98`, hp/max_hp (`$66F/$670`), max_hp, i-frames `$4F0`, knockback (`$C0` dir, `$D3`), paralyzed `$512`, doorway dir `$53`, sword level `$657`, beams-ready, beam/bomb slot active (`$BA`, `$BC/$BD`), bombs `$658`, keys `$66E`, rupees `$66D` ÷255, items-owned bits (§3), selected item `$656`, has_map/has_compass for this level, triforce bits `$671` |
| **Entities** | slots 1–$B + room item ($13): presence, ObjType embedding, x/y, dx/dy, Δpos, HP hi-nibble `$485`, stun `$3D`, dir `$98`, metastate {normal, spawning, dying}, ObjState-derived active (Leever/Zora/Wallmaster), immunities `$4B2`, item id for `$60` drops, item lifetime `$3A8` |
| **Room grid** | classes from §6 over PlayAreaTiles with door cells overwritten from R.2 (locked/shutter/bombable/open) and push block / stairs |
| **History** | not RAM — last action kind/dir/duration, frames since |

# Open questions / unverified

1. Level 7–9 room-flag block address (`$077F` inferred from the `$67F–$7FF` copy loop, `Z_02` L2830–2846); not on milestone.
2. Door types 2/3 ("false wall") and 6 ("key 2") — exact in-game behavior not traced.
3. D1 room `$22` has push-block bit set but trigger 0; how the bow-cellar stairs are revealed (and v1's `1 13 stair` row; attrs of `$13` in the L1–6 grid look like another dungeon's room) — check with RouteSurvey/route data.
4. Slot `$C`: `CheckHasLivingMonsters` reads `ObjType+1+Y` with `Y=CurObjIndex=$B` (slot `$C`) but `FindEmptyMonsterSlot` only uses `$1–$B`; confirm whether slot `$C` is ever occupied.
5. ObjHP low nibble meaning; max values for stun, shove distance, i-frame timers (set normalization from observed maxima).
6. Link `ObjState[0]` bit meanings beyond `$40` (halted) — only v1 heuristics.
7. OW flag bits 3, 5, 6: no `GetRoomFlags` reader found (grep); assume unused.
8. `RoomAllDead` wrap-around: incremented on each check; confirm it can hit 0 again (would briefly block pushes/triggers) — derive from slots instead.
9. OW attr D bit 6 is used as a layout-id bit (`Z_05` L5785–5794 shifts 7 bits) — confirm against OW layout table size.
10. `$0E` item (inventory slot `$672`) identity; not on route.
