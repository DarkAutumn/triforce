# Game data listings

How to read this: the files in this directory are decoded listings of the game's data tables. They are
written by `tools/romdata.py --rom <image>` from a PRG0 ROM image and from the tables in `src/`; do not
edit them by hand. This file describes every format and links each field to the routine that reads it.
In the listings every row starts with its id as `$xx`, so `grep '^\$32 '` finds a room. Hex is written
`$xx`; bit 7 is the high bit; a room id is `$00`-`$7F` (row in the high nibble, column in the low nibble
of a 16 x 8 map).

| File | Contents |
| --- | --- |
| `levelblock-ow.md` | overworld rooms: exits, caves, monsters, layout, secrets; cave wares; Q2 patches |
| `levelblock-uw-1to6-q1.md`, `levelblock-uw-7to9-q1.md` | underworld rooms of Q1: doors, monsters, item, secret |
| `levelblock-uw-1to6-q2.md`, `levelblock-uw-7to9-q2.md` | the same for Q2 |
| `levelinfo.md` | per level: start room, Triforce room, boss room, cellars, positions, map, palettes |
| `objlists.md` | monster lists and how a room selects one; object type names |
| `room-layouts.md` | room layouts (overworld, underworld, caves, cellars), column tables, squares |
| `person-text.md` | every text string of cave and dungeon persons |

# Blobs

The source includes 48 binary blobs with `.INCBIN`; `src/bins.xml` gives the offset of each in the PRG
image (the ROM file without its 16-byte header) and its length. Bank = offset / `$4000`.

| Blob | PRG offset | Length | Bank | Included at |
| --- | --- | --- | --- | --- |
| `dat/SongScriptItemTaken0.dat` | $00E5D | 19 | 0 | [`Song_ItemTakenScript`](../src/bank0/song-data.asm#L133) |
| `dat/SongScriptOverworld0.dat` | $00E70 | 621 | 0 | [`Song_OverworldScript`](../src/bank0/song-data.asm#L137) |
| `dat/SongScriptUnderworld0.dat` | $010DD | 199 | 0 | [`Song_UnderworldScript`](../src/bank0/song-data.asm#L141) |
| `dat/SongScriptEndLevel0.dat` | $011A4 | 89 | 0 | [`Song_EndLevelScript`](../src/bank0/song-data.asm#L145) |
| `dat/SongScriptLastLevel0.dat` | $011FD | 207 | 0 | [`Song_LastLevelScript`](../src/bank0/song-data.asm#L149) |
| `dat/SongScriptGanon0.dat` | $012CC | 43 | 0 | [`Song_GanonScript`](../src/bank0/song-data.asm#L153) |
| `dat/SongScriptEnding0.dat` | $012F7 | 404 | 0 | [`Song_EndingScript`](../src/bank0/song-data.asm#L157) |
| `dat/SongScriptDemo0.dat` | $0148B | 825 | 0 | [`Song_DemoScript`](../src/bank0/song-data.asm#L161) |
| `dat/SongScriptZelda0.dat` | $017C4 | 97 | 0 | [`Song_ZeldaScript`](../src/bank0/song-data.asm#L165) |
| `dat/PersonText.dat` | $0404C | 1366 | 1 | [`Person_Text`](../src/bank1/cave-init-person-text.asm#L54) |
| `dat/DemoSpritePatterns.dat` | $04DB4 | 2304 | 1 | [`Chr_DemoSpritePatterns`](../src/bank1/copy-code-demo-patterns.asm#L179) |
| `dat/DemoBackgroundPatterns.dat` | $056B4 | 2080 | 1 | [`Chr_DemoBackgroundPatterns`](../src/bank1/copy-code-demo-patterns.asm#L183) |
| `dat/CommonSpritePatterns.dat` | $0807F | 1792 | 2 | [`Chr_CommonSpritePatterns`](../src/bank2/common-patterns.asm#L134) |
| `dat/CommonBackgroundPatterns.dat` | $0877F | 1792 | 2 | [`Chr_CommonBackgroundPatterns`](../src/bank2/common-patterns.asm#L138) |
| `dat/CommonMiscPatterns.dat` | $08E7F | 224 | 2 | [`Chr_CommonMiscPatterns`](../src/bank2/common-patterns.asm#L142) |
| `dat/DemoTextFields.dat` | $0929A | 531 | 2 | [`Demo_TextFields`](../src/bank2/demo-tables-init.asm#L138) |
| `dat/CreditsTextLines.dat` | $0AC5C | 414 | 2 | [`Credits_TextLines`](../src/bank2/credits.asm#L138) |
| `dat/PatternBlockUWBG.dat` | $0C11B | 2080 | 3 | [`Chr_PatternsUWBackground`](../src/bank3/level-pattern-blocks.asm#L299) |
| `dat/PatternBlockOWBG.dat` | $0C93B | 2080 | 3 | [`Chr_PatternsOWBackground`](../src/bank3/level-pattern-blocks.asm#L303) |
| `dat/PatternBlockOWSP.dat` | $0D15B | 1824 | 3 | [`Chr_PatternsOWSprites`](../src/bank3/level-pattern-blocks.asm#L307) |
| `dat/PatternBlockUWSP358.dat` | $0D87B | 544 | 3 | [`Chr_PatternsUWMonsters358`](../src/bank3/level-pattern-blocks.asm#L311) |
| `dat/PatternBlockUWSP469.dat` | $0DA9B | 544 | 3 | [`Chr_PatternsUWMonsters469`](../src/bank3/level-pattern-blocks.asm#L315) |
| `dat/PatternBlockUWSP.dat` | $0DCBB | 256 | 3 | [`Chr_PatternsUWSpritesCommon`](../src/bank3/level-pattern-blocks.asm#L319) |
| `dat/PatternBlockUWSP127.dat` | $0DDBB | 544 | 3 | [`Chr_PatternsUWMonsters127`](../src/bank3/level-pattern-blocks.asm#L323) |
| `dat/PatternBlockUWSPBoss1257.dat` | $0DFDB | 1024 | 3 | [`Chr_PatternsUWBoss1257`](../src/bank3/level-pattern-blocks.asm#L327) |
| `dat/PatternBlockUWSPBoss3468.dat` | $0E3DB | 1024 | 3 | [`Chr_PatternsUWBoss3468`](../src/bank3/level-pattern-blocks.asm#L331) |
| `dat/PatternBlockUWSPBoss9.dat` | $0E7DB | 1024 | 3 | [`Chr_PatternsUWBoss9`](../src/bank3/level-pattern-blocks.asm#L335) |
| `dat/ObjLists.dat` | $14676 | 201 | 5 | [`Spawn_ObjLists`](../src/bank5/mode-inits-obj-lists.asm#L180) |
| `dat/RoomLayoutsOW.dat` | $15418 | 1936 | 5 | [`RoomLayout_RoomsOW`](../src/bank5/room-layouts-ow.asm#L58) |
| `dat/RoomLayoutsUW.dat` | $160DE | 504 | 5 | [`RoomLayout_RoomsUW`](../src/bank5/room-layouts-uw.asm#L174) |
| `dat/LevelBlockOW.dat` | $18400 | 768 | 6 | [`Level_BlockOW`](../src/bank6/level-blocks-common-data.asm#L66) |
| `dat/LevelBlockUW1Q1.dat` | $18700 | 768 | 6 | [`Level_BlockUW1to6Q1`](../src/bank6/level-blocks-common-data.asm#L70) |
| `dat/LevelBlockUW2Q1.dat` | $18A00 | 768 | 6 | [`Level_BlockUW7to9Q1`](../src/bank6/level-blocks-common-data.asm#L74) |
| `dat/LevelBlockUW1Q2.dat` | $18D00 | 768 | 6 | [`Level_BlockUW1to6Q2`](../src/bank6/level-blocks-common-data.asm#L78) |
| `dat/LevelBlockUW2Q2.dat` | $19000 | 768 | 6 | [`Level_BlockUW7to9Q2`](../src/bank6/level-blocks-common-data.asm#L82) |
| `dat/LevelInfoOW.dat` | $19300 | 252 | 6 | [`Level_InfoOW`](../src/bank6/level-blocks-common-data.asm#L86) |
| `dat/LevelInfoUW1.dat` | $193FC | 252 | 6 | [`Level_InfoUW1`](../src/bank6/level-blocks-common-data.asm#L90) |
| `dat/LevelInfoUW2.dat` | $194F8 | 252 | 6 | [`Level_InfoUW2`](../src/bank6/level-blocks-common-data.asm#L94) |
| `dat/LevelInfoUW3.dat` | $195F4 | 252 | 6 | [`Level_InfoUW3`](../src/bank6/level-blocks-common-data.asm#L98) |
| `dat/LevelInfoUW4.dat` | $196F0 | 252 | 6 | [`Level_InfoUW4`](../src/bank6/level-blocks-common-data.asm#L102) |
| `dat/LevelInfoUW5.dat` | $197EC | 252 | 6 | [`Level_InfoUW5`](../src/bank6/level-blocks-common-data.asm#L106) |
| `dat/LevelInfoUW6.dat` | $198E8 | 252 | 6 | [`Level_InfoUW6`](../src/bank6/level-blocks-common-data.asm#L110) |
| `dat/LevelInfoUW7.dat` | $199E4 | 252 | 6 | [`Level_InfoUW7`](../src/bank6/level-blocks-common-data.asm#L114) |
| `dat/LevelInfoUW8.dat` | $19AE0 | 252 | 6 | [`Level_InfoUW8`](../src/bank6/level-blocks-common-data.asm#L118) |
| `dat/LevelInfoUW9.dat` | $19BDC | 252 | 6 | [`Level_InfoUW9`](../src/bank6/level-blocks-common-data.asm#L122) |
| `dat/StoryTileAttrTransferBuf.dat` | $1A3FE | 1131 | 6 | [`Transfer_StoryTileAttrBuf`](../src/bank6/transfer-bufs-static.asm#L326) |
| `dat/GameTitleTransferBuf.dat` | $1A869 | 1121 | 6 | [`Transfer_GameTitleBuf`](../src/bank6/transfer-bufs-static.asm#L330) |
| `dat/PcmSamples.dat` | $1C000 | 9216 | 7 | [`Sample_PcmData`](../src/bank7/run-game-nmi.asm#L48) |

# Level blocks

A level block is 768 bytes: six arrays of 128 bytes, called A to F, each with one byte per room.

- Loading a level copies one block to RAM at [`LevelBlockAttrsA`](../src/Variables.inc#L694) (arrays B to F follow at 128-byte steps: [`LevelBlockAttrsB`](../src/Variables.inc#L697), [`LevelBlockAttrsC`](../src/Variables.inc#L700), [`LevelBlockAttrsD`](../src/Variables.inc#L702), [`LevelBlockAttrsE`](../src/Variables.inc#L705), [`LevelBlockAttrsF`](../src/Variables.inc#L708)) and one
  level info block after it; all code reads the RAM copy. [`Mode2Load_Init_Sub0`](../src/bank6/level-load-mode2.asm#L122), [`Level_CopyBlockAndIncSubmode`](../src/bank6/level-load-mode2.asm#L249), [`Level_SetBlockCopyDest`](../src/bank6/level-load-mode2.asm#L196)
- The block is chosen by the level number and the quest of the save slot: [`Level_BlockAddrsQ1`](../src/bank6/level-load-mode2.asm#L57) for Q1, [`Level_BlockAddrsQ2`](../src/bank6/level-load-mode2.asm#L89) for Q2. [`Mode2Load_Init_Sub0`](../src/bank6/level-load-mode2.asm#L122)

| Level | Q1 block | Q2 block | Level info |
| --- | --- | --- | --- |
| overworld | [`Level_BlockOW`](../src/bank6/level-blocks-common-data.asm#L66) (`dat/LevelBlockOW.dat`) | [`Level_BlockOW`](../src/bank6/level-blocks-common-data.asm#L66) (`dat/LevelBlockOW.dat`) | [`Level_InfoOW`](../src/bank6/level-blocks-common-data.asm#L86) (`dat/LevelInfoOW.dat`) |
| 1 | [`Level_BlockUW1to6Q1`](../src/bank6/level-blocks-common-data.asm#L70) (`dat/LevelBlockUW1Q1.dat`) | [`Level_BlockUW1to6Q2`](../src/bank6/level-blocks-common-data.asm#L78) (`dat/LevelBlockUW1Q2.dat`) | [`Level_InfoUW1`](../src/bank6/level-blocks-common-data.asm#L90) (`dat/LevelInfoUW1.dat`) |
| 2 | [`Level_BlockUW1to6Q1`](../src/bank6/level-blocks-common-data.asm#L70) (`dat/LevelBlockUW1Q1.dat`) | [`Level_BlockUW1to6Q2`](../src/bank6/level-blocks-common-data.asm#L78) (`dat/LevelBlockUW1Q2.dat`) | [`Level_InfoUW2`](../src/bank6/level-blocks-common-data.asm#L94) (`dat/LevelInfoUW2.dat`) |
| 3 | [`Level_BlockUW1to6Q1`](../src/bank6/level-blocks-common-data.asm#L70) (`dat/LevelBlockUW1Q1.dat`) | [`Level_BlockUW1to6Q2`](../src/bank6/level-blocks-common-data.asm#L78) (`dat/LevelBlockUW1Q2.dat`) | [`Level_InfoUW3`](../src/bank6/level-blocks-common-data.asm#L98) (`dat/LevelInfoUW3.dat`) |
| 4 | [`Level_BlockUW1to6Q1`](../src/bank6/level-blocks-common-data.asm#L70) (`dat/LevelBlockUW1Q1.dat`) | [`Level_BlockUW1to6Q2`](../src/bank6/level-blocks-common-data.asm#L78) (`dat/LevelBlockUW1Q2.dat`) | [`Level_InfoUW4`](../src/bank6/level-blocks-common-data.asm#L102) (`dat/LevelInfoUW4.dat`) |
| 5 | [`Level_BlockUW1to6Q1`](../src/bank6/level-blocks-common-data.asm#L70) (`dat/LevelBlockUW1Q1.dat`) | [`Level_BlockUW1to6Q2`](../src/bank6/level-blocks-common-data.asm#L78) (`dat/LevelBlockUW1Q2.dat`) | [`Level_InfoUW5`](../src/bank6/level-blocks-common-data.asm#L106) (`dat/LevelInfoUW5.dat`) |
| 6 | [`Level_BlockUW1to6Q1`](../src/bank6/level-blocks-common-data.asm#L70) (`dat/LevelBlockUW1Q1.dat`) | [`Level_BlockUW1to6Q2`](../src/bank6/level-blocks-common-data.asm#L78) (`dat/LevelBlockUW1Q2.dat`) | [`Level_InfoUW6`](../src/bank6/level-blocks-common-data.asm#L110) (`dat/LevelInfoUW6.dat`) |
| 7 | [`Level_BlockUW7to9Q1`](../src/bank6/level-blocks-common-data.asm#L74) (`dat/LevelBlockUW2Q1.dat`) | [`Level_BlockUW7to9Q2`](../src/bank6/level-blocks-common-data.asm#L82) (`dat/LevelBlockUW2Q2.dat`) | [`Level_InfoUW7`](../src/bank6/level-blocks-common-data.asm#L114) (`dat/LevelInfoUW7.dat`) |
| 8 | [`Level_BlockUW7to9Q1`](../src/bank6/level-blocks-common-data.asm#L74) (`dat/LevelBlockUW2Q1.dat`) | [`Level_BlockUW7to9Q2`](../src/bank6/level-blocks-common-data.asm#L82) (`dat/LevelBlockUW2Q2.dat`) | [`Level_InfoUW8`](../src/bank6/level-blocks-common-data.asm#L118) (`dat/LevelInfoUW8.dat`) |
| 9 | [`Level_BlockUW7to9Q1`](../src/bank6/level-blocks-common-data.asm#L74) (`dat/LevelBlockUW2Q1.dat`) | [`Level_BlockUW7to9Q2`](../src/bank6/level-blocks-common-data.asm#L82) (`dat/LevelBlockUW2Q2.dat`) | [`Level_InfoUW9`](../src/bank6/level-blocks-common-data.asm#L122) (`dat/LevelInfoUW9.dat`) |

## Overworld rooms

- A bits 7-4: X where Link appears when he comes out of a cave or a level into this room (`A & $F0`). [`Room_EnterCommon`](../src/bank5/enter-room.asm#L137)
- A bit 3: a Zora lives in this room. [`Zora_TrySpawn`](../src/bank4/zora-ow-inits.asm#L75)
- A bit 2: the sea sound plays in this room. [`Mode5Play_UpdateWorld`](../src/bank7/mode5-play-update.asm#L129)
- A bits 1-0: palette selector of the outer ring of the play area; B bits 1-0: palette selector of the
  inner part. [`Room_FillPlayAreaAttrs`](../src/bank5/mode7-scroll-init.asm#L379), [`Room_PaletteSelectorAttrs`](../src/bank5/mode7-scroll-init.asm#L368)
- B bit 0 also picks the palette row of a rock (`$62`) or armos (`$66`) tile object. [`Mode5Play_Init`](../src/bank7/init-mode5-play.asm#L60)
- B bits 7-2: cave index, read when Link stands on a stairs tile (`$70`-`$73`) or a cave entrance tile
  (`$24`, `$88`). `1`-`9`: enter that level. `$10` and above: a cave (`$14`: the shortcut cave, mode `$C`;
  any other: mode `$B`). The code treats every value below `$10` as a level number; the listings show
  `0` as no destination, assuming such rooms hold no stairs or cave tile (unverified). [`Warp_HandleOW`](../src/bank5/warps-stairs-caves.asm#L174)
- The person in a cave is object type `$6A + (cave index - $10)`. [`Spawn_AssignPositions`](../src/bank5/spawn-positions.asm#L64)
- C bits 5-0 and D bit 7 (as bit 6) form the monster list id; C bits 7-6 select one of the four
  monster counts of the level info ([`LevelInfo_FoeCounts`](../src/Variables.inc#L715)). See `objlists.md`. [`Room_EnterCommon`](../src/bank5/enter-room.asm#L137)
- D bits 6-0: room layout id (7 bits in the overworld). See `room-layouts.md`. [`RoomLayout_LayOutRoomOW`](../src/bank5/layout-room-ow.asm#L106)
- The layout id that Room_GetUniqueRoomId returns is only D bits 5-0, also in the overworld. [`Room_GetUniqueRoomId`](../src/bank7/tile-obj-tiles.asm#L40)
- E is not per room in the overworld: it holds the wares and prices of the caves (below). [`Cave_Init_SetUpTextAndWares`](../src/bank1/cave-init-person-text.asm#L145)
- F bits 7-6: the quest in which the room's tile-object secret (pushed rock or gravestone, bombed rock
  wall, burned tree) works: `0` both, `1` Q1 only, `2` Q2 only ([`Secret_QuestNumbers`](../src/bank4/rock-wall-tree-secrets.asm#L322)). [`Secret_IsForOtherQuest`](../src/bank4/rock-wall-tree-secrets.asm#L331), [`Rock_Update`](../src/bank4/rock-wall-tree-secrets.asm#L62), [`RockWall_Update`](../src/bank4/rock-wall-tree-secrets.asm#L214), [`Tree_Update`](../src/bank4/rock-wall-tree-secrets.asm#L249)
- F bits 5-4: index into the four positions of the level info ([`LevelInfo_ShortcutOrItemPosArray`](../src/Variables.inc#L717)) where the room's secret stairs
  appear; a position byte is X in the high nibble (`byte & $F0`) and Y in the low nibble
  (`(byte & $0F) * 16`). [`Room_GetShortcutOrItemXYForRoomId`](../src/bank1/bomb-candle-wield.asm#L280), [`Secret_RestoreStairsOW`](../src/bank5/ow-shortcut-map-square.asm#L54)
- F bit 3: monsters enter from the screen edges instead of starting inside the room. [`Spawn_AssignPositions`](../src/bank5/spawn-positions.asm#L64), [`Obj_Init`](../src/bank7/object-init-table.asm#L45)
- F bits 2-0: row where Link appears when he comes out of a cave or a level: `Y = row * 16 + $4D`; after
  a cave entrance tile (`$24`) he starts 16 pixels lower and walks up to that Y. [`Room_EnterCommon`](../src/bank5/enter-room.asm#L137)
- The tile object of a room is not in the level block: it comes from the layout (a square whose primary
  tile is `$E5`-`$EA`), except in the dock rooms $3F and $55, which get object type $61. [`RoomLayout_CheckTileObj`](../src/bank5/layout-room-ow.asm#L267), [`Room_SetUpTileObjOW`](../src/bank5/enter-room.asm#L423)

## Cave wares (overworld array E)

- Cave `n` = cave index - `$10` (20 caves). Bytes `E[3n]`, `E[3n+1]`, `E[3n+2]` are its three wares, left
  to right; `E[60+3n]` and the two bytes after it are their prices. [`Cave_Init_SetUpTextAndWares`](../src/bank1/cave-init-person-text.asm#L145)
- A ware byte holds the item id in bits 5-0 (`$3F` = nothing) and two cave flags in bits 7-6. [`Cave_Init_SetUpTextAndWares`](../src/bank1/cave-init-person-text.asm#L145), [`Cave_DrawWares`](../src/bank1/cave-person-prices.asm#L165)
- [`Cave_PersonTextSelectors`](../src/bank1/cave-init-person-text.asm#L71) gives each cave its text selector in bits 5-0 (see `person-text.md`) and two more cave flags in
  bits 7-6. [`Cave_Init_SetUpTextAndWares`](../src/bank1/cave-init-person-text.asm#L145)
- The eight flags are packed into [`CaveFlags`](../src/CaveVars.inc#L13): bits 7-6 from ware 0, bits 5-4 from ware 1, bits 3-2 from ware 2,
  bits 1-0 from the text selector byte. [`Cave_Init_SetUpTextAndWares`](../src/bank1/cave-init-person-text.asm#L145)
- Flag `$01` pick-up: Link can take a ware by touching it; without it the person only talks. [`Cave_State2TalkOrShopOrDoorCharge`](../src/bank1/cave-shop.asm#L55)
- Flag `$02` pay: taking a ware costs its price. [`Cave_State2TalkOrShopOrDoorCharge`](../src/bank1/cave-shop.asm#L55)
- Flag `$04` show-items: the wares are drawn. Flag `$08` show-prices: a rupee and the prices are drawn. [`Cave_DrawWares`](../src/bank1/cave-person-prices.asm#L165), [`Cave_State0TransferPrices`](../src/bank1/cave-person-prices.asm#L221)
- Flag `$10` hint: Link pays the price of the ware he touches and the person says a hint. [`Cave_State2TalkOrShopOrDoorCharge`](../src/bank1/cave-shop.asm#L55), [`Cave_State5HintOrMoneyGame`](../src/bank1/cave-hint-money-game.asm#L61)
- Flag `$20` money: the money game for person types below `$7B`; a person of type `$7B` or above
  gives Link the middle price as rupees. [`Cave_State5HintOrMoneyGame`](../src/bank1/cave-hint-money-game.asm#L61)
- Flag `$40` heart-requirement: the ware can be taken only when [`HeartValues`](../src/Variables.inc#L592) >= `$40` (person type `$6C`) or
  >= `$B0` (other types). [`Cave_State2TalkOrShopOrDoorCharge`](../src/bank1/cave-shop.asm#L55)
- Flag `$80` minus-sign: prices are shown with a minus sign. [`Cave_WritePriceListWithChar`](../src/bank1/cave-person-prices.asm#L245)
- The last 8 bytes of E (`E[$78]`-`E[$7F]`) are not read by any code found. (unverified) [`Cave_Init_SetUpTextAndWares`](../src/bank1/cave-init-person-text.asm#L145)

## Underworld rooms

- Doors: A bits 7-5 north, A bits 4-2 south, B bits 7-5 west, B bits 4-2 east. [`Door_FindTypeByDirBit`](../src/bank5/fill-walls.asm#L55)
- Door types: `0` open, `1` wall, `2` and `3` false wall (Link walks through after pushing), `4` bombable
  wall, `5` and `6` key door, `7` shutter. [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62), [`Door_LayOut`](../src/bank5/lay-out-doors.asm#L131), [`Room_CalcNext_JumpTable`](../src/bank7/next-room-room-flags.asm#L90)
- A bits 1-0 and B bits 1-0: palette selectors of the outer ring and the inner part. [`Room_FillPlayAreaAttrs`](../src/bank5/mode7-scroll-init.asm#L379)
- C bits 5-0 and D bit 7 (as bit 6): monster list id; C bits 7-6: index of the monster count. [`Room_EnterCommon`](../src/bank5/enter-room.asm#L137)
- D bit 6: the room has a push block; it is put on the block tile (`$B0`) found in the floor row at
  Y = `$90`. [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51), [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L228)
- D bits 5-0: room layout id. `$3E` and `$3F` are cellars (below). [`Room_GetUniqueRoomId`](../src/bank7/tile-obj-tiles.asm#L40), [`RoomLayout_LayOutUWFloor`](../src/bank5/uw-floor-push-block.asm#L67)
- E bit 7: the room is dark. [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124), [`Room_IsDark_Bank4`](../src/bank4/shoot-helpers-candle.asm#L182)
- E bits 6-5: index of the sample request ($00 $10 $20 $40) that repeats in the room until the flags of the level's
  boss room have both kill-count bits set. [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42), [`Sample_BossRoarRequests`](../src/bank5/boss-sound-door-face.asm#L30)
- E bits 4-0: room item id; `$03` stands for no item. [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)
- F bits 5-4: index into the four positions of the level info where the room item lies (same position
  bytes as above; a Triforce piece is drawn 8 pixels further left). [`Room_GetShortcutOrItemXYForRoomId`](../src/bank1/bomb-candle-wield.asm#L280), [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)
- F bits 2-0: secret trigger: `1` all monsters dead opens shutters; `2` killing the monster in the first
  slot kills all; `3` shutters open when the last boss is defeated; `4` pushing the block opens shutters;
  `5` pushing the block makes stairs appear at (`$D0`, `$60`); `6` shutters open when the life-or-money
  person is gone; `7` all monsters dead opens shutters and makes the room item appear. [`Secret_RunTrigger_JumpTable`](../src/bank5/underworld-secrets.asm#L105), [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61)
- With trigger `3` or `7` the room item starts hidden. [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)
- F bits 7-6 and bit 3 are read only by overworld code (the readers test the level number or belong to
  overworld tile objects); they are 0 in every underworld room. [`Obj_Init`](../src/bank7/object-init-table.asm#L45), [`Spawn_AssignPositions`](../src/bank5/spawn-positions.asm#L64), [`Secret_IsForOtherQuest`](../src/bank4/rock-wall-tree-secrets.asm#L331)

## Cellar rooms

- A cellar is a room of the level block with layout id `$3E` (tunnel) or `$3F` (treasure room); each
  level lists its cellars in the level info ([`LevelInfo_CellarRoomIdArray`](../src/Variables.inc#L725)). [`Mode9Cellar_Init_LayOutCellar`](../src/bank5/layout-room-ow.asm#L418), [`Mode11Death_Init_Sub1`](../src/bank5/mode11-death-init.asm#L73)
- Stairs in a room lead to the first listed cellar whose A or B byte equals the room id. [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L54)
- A: the room Link returns to when he leaves the cellar at X < `$80`; B: the room when X >= `$80`. Link
  enters at the left ladder when he comes from room A, else at the right ladder. [`Warp_CheckSubroomExit`](../src/bank5/cave-cellar-exits.asm#L43), [`Mode9Cellar_Init_Sub8_EnterCellar`](../src/bank5/cave-cellar-enter.asm#L276)
- C: where Link appears in the room he returns to: `X = C & $F0`, `Y = (C & $0F) * 16 + $0D`. [`Warp_CheckSubroomExit`](../src/bank5/cave-cellar-exits.asm#L43)
- E and F are read as in other underworld rooms (room item, its position, dark flag). [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)
- A cellar always holds four Keese blue; its own monster list is ignored. [`Spawn_AssignPositions`](../src/bank5/spawn-positions.asm#L64), [`Room_EnterCommon`](../src/bank5/enter-room.asm#L137)

# Level info

One block of 252 bytes per level, copied to RAM at [`LevelInfo_PalettesTransferBuf`](../src/Variables.inc#L714). Offsets are from the start of the block.

- [`Level_InfoAddrs`](../src/bank6/level-load-mode2.asm#L71) chooses the block by level number. The copy runs to `$6C7D`: 256 bytes, the block and the 4 bytes
  that follow it in the ROM. [`Mode2Load_Init_Sub1`](../src/bank6/level-load-mode2.asm#L156), [`Level_SetInfoCopyDest`](../src/bank6/level-load-mode2.asm#L214), [`Level_CopyBlockAndIncSubmode`](../src/bank6/level-load-mode2.asm#L249)
- `+$00` (36 bytes): transfer record with the level's 32 palette bytes; byte 20 is replaced with the
  save slot's colour for Link before the transfer. [`LevelInfo_PalettesTransferBuf`](../src/Variables.inc#L714), [`Palette_PatchLinkColorAndCueLevelPalettes`](../src/bank7/init-mode-dispatch.asm#L319)
- `+$24` (4 bytes): the four monster counts a room can select. [`LevelInfo_FoeCounts`](../src/Variables.inc#L715), [`Room_EnterCommon`](../src/bank5/enter-room.asm#L137)
- `+$28`: Link's Y when the level starts (his X is `$78`). [`LevelInfo_StartY`](../src/Variables.inc#L716), [`Mode3Unfurl_Init_Sub8_LayOutRoomPlaceLink`](../src/bank5/mode3-mode6-init.asm#L129)
- `+$29` (4 bytes): positions of secret stairs (overworld) or room items (underworld). [`LevelInfo_ShortcutOrItemPosArray`](../src/Variables.inc#L717), [`Room_GetShortcutOrItemXYForRoomId`](../src/bank1/bomb-candle-wield.asm#L280)
- `+$2D`: number of columns the submenu map is rotated right. [`LevelInfo_SubmenuMapRotation`](../src/Variables.inc#L719), [`Submenu_WriteSheetMapRowTransferRecord`](../src/bank5/submenu-map.asm#L49)
- `+$2E`: pixels added to the X of the status bar map markers. [`LevelInfo_StatusBarMapXOffset`](../src/Variables.inc#L720), [`StatusBar_UpdatePositionMarker`](../src/bank1/map-marker-curtain.asm#L72)
- `+$2F`: start room. [`LevelInfo_StartRoomId`](../src/Variables.inc#L721), [`Mode3Unfurl_Init_Sub1`](../src/bank7/init-mode-dispatch.asm#L294)
- `+$30`: Triforce room, shown on the status bar map with the compass (underworld only). [`LevelInfo_TriforceRoomId`](../src/Variables.inc#L722), [`StatusBar_UpdateTriforceMarker`](../src/bank7/next-room-room-flags.asm#L51)
- `+$31` (2 bytes, low byte first): RAM address of this level's room flags (one byte per room). [`LevelInfo_WorldFlagsAddr`](../src/Variables.inc#L723), [`RoomFlags_Get`](../src/bank7/next-room-room-flags.asm#L134)
- `+$33`: level number as written into the `LEVEL-n` text; 0 in the overworld. [`LevelInfo_LevelNumber`](../src/Variables.inc#L724), [`Mode3Unfurl_Init_Sub7_CueLevelNumber`](../src/bank5/mode3-mode6-init.asm#L114)
- `+$34` (10 bytes, `$FF` = unused): cellar rooms (underworld); in the overworld the first four are the
  rooms joined by the shortcut cave: stairs 1 to 3 lead to the next 1 to 3 entries after the current
  room, wrapping after the fourth. [`LevelInfo_CellarRoomIdArray`](../src/Variables.inc#L725), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L54), [`Warp_CheckSubroomExit`](../src/bank5/cave-cellar-exits.asm#L43)
- `+$3E`: boss room; the boss sound of a room stops once this room's flags say the boss is dead
  (underworld only). [`LevelInfo_BossRoomId`](../src/Variables.inc#L727), [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42)
- `+$3F` (16 bytes): submenu map mask, one byte per map column after the rotation; bit `$80 >> row`
  set = the room in that row is drawn. Room (row, column) uses byte `(column + rotation) & 15`. [`LevelInfo_SubmenuMapMask`](../src/Variables.inc#L728), [`Submenu_WriteSheetMapRowTransferRecord`](../src/bank5/submenu-map.asm#L49)
- `+$4F` (to `+$7B`): transfer records that draw the status bar map. [`LevelInfo_StatusBarMapTransferBuf`](../src/Variables.inc#L729), [`Transfer_BufAddrs`](../src/bank6/transfer-buf-dispatch.asm#L42)
- `+$7C` (96 bytes): three fade series of 4 steps x 8 palette bytes (written to PPU `$3F08`), used by
  fade cycles `$00`, `$20` and `$40` and backwards by `$80`, `$A0` and `$C0`. [`LevelInfo_PaletteCycles`](../src/Variables.inc#L730), [`Palette_AnimateFade`](../src/bank1/world-fading.asm#L44)
- `+$DC` (32 bytes): a fourth series, used by fade cycle `$60` when Link dies. [`LevelInfo_DeathPaletteSeries`](../src/Variables.inc#L731), [`Mode11Death_Init_Sub1`](../src/bank5/mode11-death-init.asm#L73), [`Palette_AnimateFade`](../src/bank1/world-fading.asm#L44)
- A transfer record is: PPU address (high byte, low byte), a control byte (bit 7: write downwards, bit 6:
  repeat one byte, bits 5-0: count, 0 = 64), then the data; a high byte >= `$80` ends the list. [`Transfer_WriteBuf`](../src/bank6/transfer-buf-dispatch.asm#L214), [`Transfer_WriteRecord`](../src/bank6/transfer-buf-dispatch.asm#L138)

# Second-quest patches

In Q2 the level data is patched in RAM after it is copied. [`Mode2Load_Update_Full`](../src/bank6/level-load-mode2.asm#L297)

- Levels 1-9 use their own Q2 level blocks (table above) and the Q1 level info blocks. [`Level_BlockAddrsQ2`](../src/bank6/level-load-mode2.asm#L89), [`Level_InfoAddrs`](../src/bank6/level-load-mode2.asm#L71)
- Level info, levels 1-9: bytes from offset `+$29` on are overwritten with the level's table ([`Quest_Q2LevelInfoPatchAddrs`](../src/bank6/quest2-replacements.asm#L197));
  [`Quest_Q2LevelInfoPatchSizes`](../src/bank6/quest2-replacements.asm#L211) holds each table's length. The loop counts from the length down to 0, so it copies one byte
  more than the table holds (the first byte of whatever follows it). [`Mode2Load_Update_Full`](../src/bank6/level-load-mode2.asm#L297)
- The tables replace the positions, map rotation and offset, start room, Triforce room, room flags
  address, level number, cellar rooms, boss room, map mask and status bar map; the palettes, the monster
  counts and the start Y stay. `levelinfo.md` lists the patched blocks. [`LevelInfo_ShortcutOrItemPosArray`](../src/Variables.inc#L717)
- The level number of a table is not always the number of the level it patches: in Q2 the number
  drawn as `LEVEL-n` differs from the internal level number ([`CurLevel`](../src/Variables.inc#L12)) for some levels (table below). All
  listings use the internal number. [`LevelInfo_LevelNumber`](../src/Variables.inc#L724), [`Mode3Unfurl_Init_Sub7_CueLevelNumber`](../src/bank5/mode3-mode6-init.asm#L114)

| Level | Table | Table bytes | Bytes copied | Last info offset written | Number shown |
| --- | --- | --- | --- | --- | --- |
| 1 | [`Quest_Q2LevelInfoPatch1`](../src/bank6/quest2-replacements.asm#L84) | 57 | 58 | `+$62` | 1 |
| 2 | [`Quest_Q2LevelInfoPatch2`](../src/bank6/quest2-replacements.asm#L96) | 55 | 56 | `+$60` | 3 |
| 3 | [`Quest_Q2LevelInfoPatch3`](../src/bank6/quest2-replacements.asm#L107) | 61 | 62 | `+$66` | 2 |
| 4 | [`Quest_Q2LevelInfoPatch4`](../src/bank6/quest2-replacements.asm#L119) | 57 | 58 | `+$62` | 5 |
| 5 | [`Quest_Q2LevelInfoPatch5`](../src/bank6/quest2-replacements.asm#L131) | 67 | 68 | `+$6C` | 4 |
| 6 | [`Quest_Q2LevelInfoPatch6`](../src/bank6/quest2-replacements.asm#L145) | 64 | 65 | `+$69` | 6 |
| 7 | [`Quest_Q2LevelInfoPatch7`](../src/bank6/quest2-replacements.asm#L158) | 63 | 64 | `+$68` | 8 |
| 8 | [`Quest_Q2LevelInfoPatch8`](../src/bank6/quest2-replacements.asm#L170) | 67 | 68 | `+$6C` | 7 |
| 9 | [`Quest_Q2LevelInfoPatch9`](../src/bank6/quest2-replacements.asm#L183) | 74 | 75 | `+$73` | 9 |

- Overworld: the level block is the Q1 block with 15 bytes replaced: 8 through [`Quest_Q2OverworldPatchOffsets`](../src/bank6/quest2-replacements.asm#L49) and [`Quest_Q2OverworldPatchValues`](../src/bank6/quest2-replacements.asm#L64) (offsets
  from array B; an offset above `$7F` lands in array C), the rest by fixed stores. [`Mode2Load_Update_Full`](../src/bank6/level-load-mode2.asm#L297)

| Array | Room | Q1 value | Q2 value | Written by |
| --- | --- | --- | --- | --- |
| A | $3C | $72 | $72 | fixed store |
| A | $74 | $82 | $72 | fixed store |
| B | $0E | $63 | $7B | replacement table |
| B | $0F | $8B | $83 | replacement table |
| B | $22 | $18 | $84 | replacement table |
| B | $34 | $83 | $0F | replacement table |
| B | $3C | $0A | $0B | replacement table |
| B | $45 | $12 | $12 | replacement table |
| B | $74 | $0F | $7A | replacement table |
| C | $0B | $10 | $2F | replacement table |
| D | $0B | $0A | $7B | fixed store |
| D | $3C | $0A | $7B | fixed store |
| D | $74 | $6F | $5A | fixed store |
| F | $3C | $03 | $01 | fixed store |
| F | $74 | $03 | $00 | fixed store |

`levelblock-ow.md` lists the patched rooms decoded.

# Monster lists (`dat/ObjLists.dat`)

- A room's monster list id (7 bits) is either an object type or a list: `0` no monsters; `$01`-`$61`
  that object type, repeated; `$62`-`$7F` list number id - `$62`. [`Room_EnterCommon`](../src/bank5/enter-room.asm#L137)
- The count comes from the level info; an id from `$32` to `$61` always makes one object. [`Room_EnterCommon`](../src/bank5/enter-room.asm#L137)
- [`Spawn_ObjListAddrs`](../src/bank5/mode-inits-obj-lists.asm#L185) holds the address of each list inside [`Spawn_ObjLists`](../src/bank5/mode-inits-obj-lists.asm#L180); a list is a run of object type bytes, of which the first
  `count` are used (lists have no end marker). [`Room_EnterCommon`](../src/bank5/enter-room.asm#L137)
- The count is lowered by the room's kill count before the objects are made. [`RoomHistory_ModifyObjCountOW`](../src/bank5/room-history-ow.asm#L93), [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50)

# Room layouts (`dat/RoomLayoutsOW.dat`, `dat/RoomLayoutsUW.dat`)

- Overworld: a layout is 16 column descriptors (the play area is 16 columns x 11 rows of 16-pixel
  squares); layout `n` starts at byte `16 * n` of [`RoomLayout_RoomsOW`](../src/bank5/room-layouts-ow.asm#L58). [`RoomLayout_LayOutRoomOW`](../src/bank5/layout-room-ow.asm#L106)
- A column descriptor is a column table number (high nibble, an entry of [`RoomLayout_ColumnDirectoryOW`](../src/bank6/level-blocks-common-data.asm#L171)) and a column number inside
  that table (low nibble). Column `k` starts at the `k+1`-th byte with bit 7 set. [`RoomLayout_LayOutRoomOrCaveOW`](../src/bank5/layout-room-ow.asm#L141)
- Overworld square descriptor: bits 5-0 square index, bit 6 = the square fills two rows, bit 7 = first
  byte of a column. Eleven rows are read. [`RoomLayout_LayOutRoomOrCaveOW`](../src/bank5/layout-room-ow.asm#L141)
- Square index < `$10`: the four tiles are entry `index` of [`RoomLayout_OWSecondarySquareTiles`](../src/bank5/layout-room-ow.asm#L88) (top left, bottom left, top right, bottom
  right). Other indexes: the tile is [`RoomLayout_OWSquareTiles`](../src/bank5/layout-room-ow.asm#L73)`[index]` and the next three tile numbers. [`RoomLayout_WriteSquareOW`](../src/bank5/layout-room-ow.asm#L311)
- A square whose primary tile is `$E5`-`$EA` makes the room's tile object (type from [`RoomLayout_TileObjTypes`](../src/bank5/layout-room-ow.asm#L59), drawn with the
  tile from [`RoomLayout_TileObjSquareTilesOW`](../src/bank5/layout-room-ow.asm#L64)); the last such square of a layout wins. [`RoomLayout_CheckTileObj`](../src/bank5/layout-room-ow.asm#L267)
- When the room's secret was already found, a `$E7` square is drawn as stairs, a `$E6` square as a cave
  entrance and a `$EA` square as stairs. [`RoomLayout_LayOutRoomOrCaveOW`](../src/bank5/layout-room-ow.asm#L141)
- Caves and cellars use the overworld routine with the layouts of [`RoomLayout_SubroomLayoutAddrs`](../src/bank5/layout-room-ow.asm#L387) (cave, shortcut cave, tunnel
  cellar, treasure cellar); in a level, column table 0 is [`RoomLayout_ColumnHeapUWCellar`](../src/bank5/room-layouts-uw.asm#L285) instead. [`ModeBCave_Init_LayOutCave`](../src/bank5/layout-room-ow.asm#L398), [`ModeCShortcut_Init_LayOutShortcut`](../src/bank5/layout-room-ow.asm#L409), [`Mode9Cellar_Init_LayOutCellar`](../src/bank5/layout-room-ow.asm#L418), [`RoomLayout_PatchColumnDirectoryForCellar`](../src/bank5/layout-room-ow.asm#L369)
- Underworld: a layout is 12 column descriptors (the floor is 12 columns x 7 rows inside the walls);
  layout `n` starts at byte `12 * n` of [`RoomLayout_RoomsUW`](../src/bank5/room-layouts-uw.asm#L174); descriptors work as above with [`RoomLayout_UWColumnHeapAddrs`](../src/bank5/uw-floor-push-block.asm#L40). [`RoomLayout_LayOutUWFloor`](../src/bank5/uw-floor-push-block.asm#L67)
- Underworld square descriptor: bits 2-0 square index into [`RoomLayout_UWSquareTiles`](../src/bank5/uw-floor-push-block.asm#L57), bits 6-4 number of extra rows the square
  repeats, bit 7 = first byte of a column. A primary tile from `$70` to `$F2` stands for four
  consecutive tiles, any other for one tile used four times. [`RoomLayout_LayOutUWFloor`](../src/bank5/uw-floor-push-block.asm#L67), [`RoomLayout_WriteSquareUW`](../src/bank5/uw-floor-push-block.asm#L177)
- Walls and doors of an underworld room are not in the layout: walls come from [`RoomLayout_UWWallTiles`](../src/bank5/room-layouts-uw.asm#L68), doors from the
  door types of the level block. [`RoomLayout_FillWalls`](../src/bank5/fill-walls.asm#L125), [`Door_LayOut`](../src/bank5/lay-out-doors.asm#L131), [`RoomLayout_LayOutRoom`](../src/bank5/mode12-lay-out-room.asm#L178)

# Person text (`dat/PersonText.dat`)

- [`Person_TextAddrs`](../src/bank1/cave-init-person-text.asm#L47) holds one address per string; a text selector is the byte offset into that table (string
  number x 2). [`Person_UpdateState_Textbox`](../src/bank1/person-textbox.asm#L51)
- Each byte is one character: bits 5-0 are the character code; bits 7-6 after the character: `$80` go
  to the second line, `$40` go to the third line, `$C0` end of the text. [`Person_UpdateState_Textbox`](../src/bank1/person-textbox.asm#L51)
- One character is shown every 6 frames. Code `$25` is a blank that takes no time. [`Person_UpdateState_Textbox`](../src/bank1/person-textbox.asm#L51)
- A character code is the number of a background tile; the blob [`Chr_CommonBackgroundPatterns`](../src/bank2/common-patterns.asm#L138) is loaded as background tiles
  `$00`-`$6F` (PPU `$1000`, the table the background uses). Read from its bitmaps: `$00`-`$09` digits,
  `$0A`-`$23` letters A-Z, `$24` blank, `$28` comma, `$29` `!`, `$2A` apostrophe, `$2B` `&`, `$2C` full
  stop, `$2D` double quote, `$2E` `?`, `$2F` dash. [`Chr_CommonPatternPpuAddrs`](../src/bank2/common-patterns.asm#L44), [`Chr_TransferCommonPatterns`](../src/bank2/common-patterns.asm#L54), [`Ppu_ResetRegisters`](../src/bank7/ppu-tablejump-input.asm#L42)
- Who says which string: [`Cave_PersonTextSelectors`](../src/bank1/cave-init-person-text.asm#L71) (caves), [`Cave_HintTextSelectorsType75`](../src/bank1/cave-hint-money-game.asm#L42) and [`Cave_HintTextSelectorsOther`](../src/bank1/cave-hint-money-game.asm#L48) (hint caves), [`Person_TextSelectorsLevels1257`](../src/bank1/underworld-person-init.asm#L67), [`Person_TextSelectorsLevels3468`](../src/bank1/underworld-person-init.asm#L134) and [`Person_TextSelectorsLevel9`](../src/bank1/underworld-person-init.asm#L162) (dungeon persons). [`Person_InitUnderworld_Full`](../src/bank1/underworld-person-init.asm#L45)
