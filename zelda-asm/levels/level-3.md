# Level 3: the raft and Manhandla

Level 3 lies under overworld room `$74`, in the south-west, and its entrance is an open cave mouth that needs no item or secret. It holds the raft (in cellar `$0F`, under room `$69`), a heart container and a Triforce piece; the boss is Manhandla in room `$4D`. The level is the Darknut level: nineteen red Darknuts in four rooms, which only a sword or a bomb can hurt and which block every hit from the front. Link needs a sword and one bomb (for the wall into the boss room); the level supplies its own keys and has bombs to take. The raft is what opens level 4: its entrance stands on an island reached only from the dock of overworld room `$55`.

Contents (generated; `L` = line in this file):

<!-- toc -->
- L21 At a glance
- L59 What Link must bring
- L74 What Link gets here
- L119 Map
- L146 Rooms
- L174 Floor plans
- L359 Monsters and boss
- L384 Route
- L418 Optional detours
- L430 Hazards and rules that matter here
- L448 Second quest
<!-- /toc -->

# At a glance

<!-- generated:facts -->
| Field | Value | Read by |
| --- | --- | --- |
| Internal level number | 3 | [`CurLevel`](../src/Variables.inc#L12) |
| Number shown on screen | LEVEL-3 | [`LevelInfo_LevelNumber`](../src/Variables.inc#L725) |
| Overworld entrance room (Q1) | $74 | [`Warp_HandleOW`](../src/bank5/warps-stairs-caves.asm#L178), [`Flute_SecretRoomIdsOW`](../src/bank7/flute.asm#L36) |
| Start room | $7C | [`LevelInfo_StartRoomId`](../src/Variables.inc#L722) |
| Triforce room | $3D (room item: Triforce piece ($1B)); the field places the compass marker | [`LevelInfo_TriforceRoomId`](../src/Variables.inc#L723), [`StatusBar_UpdateTriforceMarker`](../src/bank7/next-room-room-flags.asm#L51) |
| Boss room | $4D (1 x Manhandla ($3C)) | [`LevelInfo_BossRoomId`](../src/Variables.inc#L728), [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42) |
| Cellars | $0F (not in the cellar list; the search that has no end test finds it 72 bytes after the list's start) | [`LevelInfo_CellarRoomIdArray`](../src/Variables.inc#L726), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55) |
| Rooms | 18: 18 on the submenu map | [`LevelInfo_SubmenuMapMask`](../src/Variables.inc#L729) |
| Monster counts (index = level block C bits 7-6) | 3, 5, 6, 8 | [`LevelInfo_FoeCounts`](../src/Variables.inc#L716) |
| Key doors (room and side that has it) | 8 ($3B S, $4B N, $4A E, $4B W, $4B E, $4C W, $59 E, $5A W) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Bombable walls (room and side that has it) | 4 ($4C E, $4D W, $5B E, $5C W) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Shutters (room and side that has it) | 8 ($2A E, $3B N, $4D N, $4D S, $5D N, $59 S, $5C E, $5D W) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| False walls (room and side that has it) | 0 | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Doorways whose two sides differ | $2A E shutter / $2B open, $2B S open / $3B shutter, $3D S open / $4D shutter, $59 S shutter / $69 open | [`Door_FindTypeByDirBit`](../src/bank5/fill-walls.asm#L55) |
| Doors on sides with no room of the level | $7C S open | [`Room_CalcNext`](../src/bank7/next-room-room-flags.asm#L72) |
| Push blocks | 1 ($3B) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Dark rooms | 0 (none) | [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124) |
| Keys (room item) | 5 ($2A, $49, $4B, $6B, $7B) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Other room items | bomb ($00): $4A, $5B, $69; raft ($0C): $0F (cellar); 5 rupees ($0F): $5D; compass ($16): $5A; map ($17): $4C; heart container ($1A): $4D; Triforce piece ($1B): $3D | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |

Room by room data: [`levelblock-uw-1to6-q1.md`](../data/levelblock-uw-1to6-q1.md) (rows with `lvl` = `3` or `c3`).
<!-- /generated:facts -->

- **Getting in (Q1).** Overworld room `$74` has cave index 3, so its opening tile loads level 3; the room has no tile object and the opening is part of its layout (layout `$6F`, the black square at column 8, row 4): nothing has to be bombed, burnt, pushed or played. [levelblock-ow.md](../data/levelblock-ow.md), [`Warp_HandleOW`](../src/bank5/warps-stairs-caves.asm#L178)
- Room `$74` is closed by rock on its north, east and south sides; its only open edge is the west one (rows 4-6), so Link walks in from room `$73`. [room-layouts.md](../data/room-layouts.md) (`OW $6F`), [`RoomLayout_RoomsOW`](../src/bank5/room-layouts-ow.asm#L58)
- The mouth is entered by walking up onto X `$80`, Y `$7D` (trace: walking up onto it starts the stairs mode; Link sinks into the opening and the level loads). [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`Mode10Stairs_Update_Full`](../src/bank5/mode8-continue-mode10-stairs.asm#L176)
- **Where Link stands on entry.** Room `$7C`, X `$78`, Y `$DD` (`LevelInfo_StartY`), facing up; he is walked in to Y `$CD`, where control starts (trace). [`Mode3Unfurl_Init_Sub8_LayOutRoomPlaceLink`](../src/bank5/mode3-mode6-init.asm#L129), [`LevelInfo_StartY`](../src/Variables.inc#L717)
- **Walking out.** The south door of `$7C` leads off the map: the overworld is loaded and Link appears in room `$74` at X `$80`, Y `$7D`, facing down, on the mouth (trace). [`Room_CalcNextRoomForDoor`](../src/bank5/next-room-calc.asm#L76), [`Mode_LeaveLevelToOW`](../src/bank5/next-room-calc.asm#L107)
- **What ends the level.** Taking the Triforce piece in `$3D` starts mode `$12`: hearts refill to full, the curtain closes and Link is put on the overworld in room `$74` at X `$80`, Y `$7D`, facing down (trace: about 690 frames from the pickup to control on the overworld, with two hearts to refill). [`Item_TakeClass0PerLevel`](../src/bank1/take-item.asm#L421), [`Mode12EndLevel_Update_Sub4LeaveLevel`](../src/bank5/mode12-lay-out-room.asm#L156)
- **What a re-entry resets.** Loading the level zeroes the banked kill counts, so every room except the boss room is full again; the dead Manhandla stays dead (non-recurring type, room flags "all dead"). [`Mode2Load_Init`](../src/bank7/init-mode-dispatch.asm#L147), [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50)
- Kept across leaving, dying and saving (room flags): unlocked key doors and bombed walls, from both sides; taken room items (keys, bombs, map, compass, raft, heart container, Triforce piece). Not kept: shutters (closed on every entry) and the pushed block of `$3B`. [`Door_Update`](../src/bank5/update-doors.asm#L60), [`RoomFlags_SetItemTaken`](../src/bank1/take-item.asm#L140), [mechanics.md](../mechanics.md) "Room flags and saved state"
- Floor coordinates used below: floor column c is X = `$20` + c * `$10`, floor row r is Link's Y = `$5D` + r * `$10`; the doors are at X `$78` (N, S) and Y `$8D` (W, E), and a room item at `$80,$90` is taken by Link on the door row (Y `$8D`) at X `$78`-`$88`. [`Doorway_RequiredCoords`](../src/bank5/doorway.asm#L93), [`Item_TryTake`](../src/bank1/take-item.asm#L206)

# What Link must bring

Nothing from outside the level is strictly required except a sword; one bomb is needed on the route and can be collected inside.

- **Sword: required in practice, any grade.** The five red Darknuts of `$59` must all die to open the shutter to the raft, and a Darknut is hurt only by sword-class hits and bomb blasts (invincibility mask `$F6`): 4 wooden, 2 white or 1 magic sword hit each, never from the front. [`Darknut_Init`](../src/bank4/darknut-gibdo.asm#L44), [`Damage_HandleWeaponHit@CheckDarknut`](../src/bank1/weapon-damage-death.asm#L157), [enemies.md](../enemies.md) "Darknut (red, blue)"
- The white sword is available before this level if Link has 5 heart containers (`HeartValues` >= `$40`); it halves every sword count here and kills a Zol without splitting it. The old man of `$2B` asks about exactly that sword. [`Cave_State2TalkOrShopOrDoorCharge`](../src/bank1/cave-shop.asm#L55), [`Sword_DamagePoints`](../src/bank1/sword-bomb-arrow-collision.asm#L182), [items.md](../items.md) "Sword (Wooden, White, Magic) and sword beam"
- **Bombs: 1 on the route**, for the east wall of `$4C` (the way into the boss room); 2 more make the Manhandla fight short. None need be brought: `$5B`, which the route crosses twice, gives 4 bombs once its three Darknuts are dead. [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136), [`Item_Take_CheckClass1`](../src/bank1/take-item.asm#L337), [items.md](../items.md) "Bombs"
- With no bomb at all the boss room can still be reached through shutters: `$4C` S, `$5C` (kill 3 Darknuts), `$5D` (kill 2 Zols and 3 Keese; the Bubbles do not count), north into `$4D`: two more room entries and eight more kills (data and code only). [`Secret_Trigger1AllDead`](../src/bank5/underworld-secrets.asm#L152), [`Secret_UpdateRoomAllDead`](../src/bank5/underworld-secrets.asm#L124)
- **Keys: 0.** The route passes two key doors (`$5A` W, `$4B` E) and picks up two keys before them (`$7B`, lying in the open; `$4B`, after its 3 Zols). Five keys lie in the level against four key doors, so Link can leave with up to two more keys than he brought. [`Door_TouchKey`](../src/bank5/touch-door.asm#L160), [`InvKeys`](../src/Variables.inc#L584), [items.md](../items.md) "Keys and Magic Key"
- Each spare key brought saves a fight: with one key Link need not kill the Zols of `$4B`. [`Door_TouchKey`](../src/bank5/touch-door.asm#L160)
- **Candle: not needed.** No room of the level is dark (attribute E bit 7 is clear in all 18 rooms and the cellar). [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124)
- **Ladder: not needed.** No floor plan of the level has a water square. [`RoomLayout_UWSquareTiles`](../src/bank5/uw-floor-push-block.asm#L63)
- **Raft, recorder, food, bow and arrows, rupees: not needed.** No dock, no Digdogger, no hungry Goriya; the only person (`$2B`) just talks; Manhandla and every monster here die to the sword or bombs. [`Person_UpdateUnderworld_Full`](../src/bank1/underworld-person-update.asm#L44), [`Manhandla_Init`](../src/bank4/gleeok-manhandla-gohma-inits.asm#L155)
- **Hearts: no fixed number.** What hits on the route: red Darknut, Zol, Trap, Manhandla and its fireballs 1 heart each; Keese and Gel half a heart; rings halve or quarter it. Full hearts also give the sword beam, which a Darknut blocks from the front like the sword. [`Damage_ObjTypeDamageToLink`](../src/bank1/monster-collisions.asm#L200), [`Link_BeHarmed`](../src/bank1/link-collision-harm.asm#L193), [`SwordShot_Make`](../src/bank7/sword-and-rod.asm#L327)

# What Link gets here

## Raft (cellar `$0F`, under room `$69`)

- **Where.** Room `$69` has stairs in its layout (floor column 11, row 3: X `$D0`, Y `$8D`, in a notch of the east wall of blocks); they lead to treasure cellar `$0F`, whose room item is the raft at `$80,$90`. No secret has to be triggered: the stairs are there on entry and the eight Darknuts of `$69` need not die. [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)
- **Getting to `$69`.** Its only door is the north one, from `$59`, whose south side is a shutter: all five Darknuts of `$59` must be dead. [`Secret_Trigger1AllDead`](../src/bank5/underworld-secrets.asm#L152), [`Door_TouchShutter`](../src/bank5/touch-door.asm#L126)
- **Stairs.** Stepping onto X `$D0`, Y `$8D` with Link on the grid starts the descent (trace: from the north door, down to Y `$8D`, then right; 52 frames of walking right from X `$78`). [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)
- **In the cellar (trace).** Link appears at X `$30`, Y `$5D` on the left ladder. Down the ladder to Y `$BD` (72 frames), right along the lower walkway to X `$B0` (98), up the short ladder to Y `$8D` (35), left along the upper walkway: the raft is taken at X `$88` (30 frames). Pressing right on the left ladder at Y `$8D`, `$9D` or `$AD` does not move Link. [`Mode9Cellar_Init_Sub8_EnterCellar`](../src/bank5/cave-cellar-enter.asm#L276), [`Mode9Cellar_Init_LayOutCellar`](../src/bank5/layout-room-ow.asm#L418)
- Taking it holds Link for `$80` frames (item lift); four blue Keese fly in the cellar and can be killed with any weapon or ignored. [`Item_Take`](../src/bank1/take-item.asm#L259), [`ItemLiftTimer`](../src/Variables.inc#L416), [`Spawn_AssignPositions`](../src/bank5/spawn-positions.asm#L64)
- **Out.** Back the same way to X `$30`, up the left ladder; above Y `$40` with Up held Link leaves and stands in `$69` at X `$60`, Y `$8D`, facing down, with all Darknuts he did not kill back in the room (trace). [`Warp_CheckSubroomExit`](../src/bank5/cave-cellar-exits.asm#L43), [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50)

**What it changes.**
- `InvRaft` = 1. The raft is not a B-button item and has no water test: it works only on the two docks of the overworld, rooms `$3F` and `$55`, and nowhere in any dungeon. [`InvRaft`](../src/Variables.inc#L567), [`Room_SetUpTileObjOW`](../src/bank5/enter-room.asm#L429), [items.md](../items.md) "Raft"
- Rule: with the raft, standing exactly on the dock (X `$80` in room `$55`, X `$60` in room `$3F`; Y `$7D`) starts a ride up, 1 pixel per frame for 64 frames, ending in the room above; no button is pressed, and Link cannot be hurt during the ride. Without the raft the dock does nothing. [`Dock_Update`](../src/bank4/dock-raft.asm#L39), [mechanics.md](../mechanics.md) "Ladder and raft"
- **Dock of `$55` -> room `$45`: the entrance of level 4.** Room `$45` is an island: its layout is ringed by rock and water, and the only gap is the pier at column 8 of its bottom edge, above the dock. The level 4 mouth is the opening at column 8, row 4 (X `$80`, Y `$7D`). Nothing else in Q1 leads there on foot, so the raft is required for level 4. [room-layouts.md](../data/room-layouts.md) (`OW $41`, `OW $51`), [levelblock-ow.md](../data/levelblock-ow.md), [level-4.md](level-4.md)
- The dock of `$55` is in the east part of that room (floor column 8, row 4); the west part is cut off from it by water, so `$55` is entered from `$65` below through the lanes at columns 7-13, or from `$56`. [room-layouts.md](../data/room-layouts.md) (`OW $51`)
- **Dock of `$3F` -> room `$2F`:** a "take any one" cave (red potion or heart container), likewise reachable only by raft. [levelblock-ow.md](../data/levelblock-ow.md), [`Dock_Update`](../src/bank4/dock-raft.asm#L39)
- Coming back: walking down the pier off the bottom edge of `$45` (or `$2F`) puts Link at the top edge of the dock room at the dock's X, Y `$3D`, where the ride down starts by itself (66 frames). [`Dock_Update@State1`](../src/bank4/dock-raft.asm#L117)
- First needed: not in this level. On the overworld at once, for level 4, whose item is the next one the game requires. [level-4.md](level-4.md), [`Dock_Update`](../src/bank4/dock-raft.asm#L39)

## Heart container (room `$4D`)

- Hidden room item at `$80,$90`; it appears when Manhandla is dead (secret trigger 7), together with the opening of the north and south shutters (trace: shown 69 frames after the bomb that killed the last hand was placed). [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61), [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)
- **What it changes.** `HeartValues` + `$11`: one more container and one more full heart; the other hearts are not refilled. Containers gate the sword caves on the overworld (white sword 5, magic sword 12). [`Item_Take_CheckClass1`](../src/bank1/take-item.asm#L337), [`HeartValues`](../src/Variables.inc#L593), [items.md](../items.md) "Hearts, Heart Containers and Fairies"

## Triforce piece (room `$3D`)

- Room item at `$78,$90`, visible, no monsters in the room. From the south door walk straight up (X `$78`): the gap in the ring of blocks is at floor columns 5-6 of row 5; it is taken at Y `$95` (trace: 41 frames of Up). [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51), [`Item_TryTake`](../src/bank1/take-item.asm#L206)
- **What it changes.** Bit 2 of `InvTriforce` is set, hearts are refilled and Link is put outside (see "At a glance"). All eight pieces are needed to pass the old man at the start of level 9; the recorder's whirlwind carries Link only to levels whose piece he has; the compass marker of this level stops blinking. [`InvTriforce`](../src/Variables.inc#L597), [`Person_InitUnderworldLevel9`](../src/bank1/underworld-person-init.asm#L172), [`Whirlwind_Summon`](../src/bank1/whirlwind.asm#L243), [items.md](../items.md) "Triforce pieces and the Triforce of Power"

## Keys (five)

- `$7B`: at `$80,$90`, visible on entry, six Zols in the room; on the route, taken in passing along the door row. [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)
- `$4B`: hidden until its three Zols are dead; at `$80,$90`, inside the ring of blocks, which is closed on the north, east and south and reached from floor column 0 on the west. On the route. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61)
- `$6B`: hidden until its five Zols are dead. On the route, optional. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61)
- `$49`: visible on entry; `$2A`: visible on entry (its secret only opens the shutter). Both off the route, see "Optional detours". [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)
- **What a key changes.** `InvKeys` + 1, one counter for all levels; pushing into a key door spends one and the door stays open from both sides for good. Needed in this level at `$5A` W and `$4B` E; surplus keys carry to later levels. [`Door_TouchKey`](../src/bank5/touch-door.asm#L160), [`RoomFlags_SetDoorOpened`](../src/bank5/shutters-door-flags.asm#L41), [items.md](../items.md) "Keys and Magic Key"

## Bombs (three rooms), map, compass, rupees

- Bombs, 4 per pickup up to `InvMaxBombs`, all hidden until the room is cleared: `$5B` (3 Darknuts; on the route), `$4A` (5 Keese), `$69` (8 Darknuts). [`Item_Take_CheckClass1`](../src/bank1/take-item.asm#L337), [`InvMaxBombs`](../src/Variables.inc#L607)
- **What bombs change here.** A bomb opens the walls `$4C` E and `$5B` E, hurts a Darknut from the side or the back (`$40` = a red Darknut's whole HP) and kills a Manhandla hand outright. [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136), [`Bomb_CheckBombOrFireVsMonster`](../src/bank1/sword-bomb-arrow-collision.asm#L126)
- Compass (`$5A`) and map (`$4C`): both lie visible at `$80,$90`, on the straight line between the west and east doors, so the route takes them in passing (trace). They change only what is drawn: the status bar map, and the marker on the Triforce room `$3D`. [`Inv_HasMap`](../src/bank5/submenu-map.asm#L153), [`Inv_HasCompass`](../src/bank5/submenu-map.asm#L143), [items.md](../items.md) "Map and Compass"
- 5 rupees (`$5D`, at `$20,$60`): hidden until the room's Zols and Keese are dead. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61)

# Map

<!-- generated:map -->
```
         [2Ak ]S/-[2B  ]
                   |/S
                  [3B  ]            [3DT ]
                    K                |/S
[49k ]---[4Ai ] K [4Bk ] K [4Ci ] X [4DBi]
  |        |        |        |        S
[59  ] K [5Ai ]---[5Bi ] X [5C  ] S [5Di ]
 S/|                |
[69i ]            [6Bk ]
                    |
                  [7Bk ]---[7CS ]

[73S] = room $73 and its marks: S start room, B boss room, T Triforce room,
        k a key lies here, i another room item lies here, h not shown on the submenu map.
Between two rooms, the door: --- or | open, K key door, S shutter, X bombable wall,
        F false wall (walk-through), nothing = wall. A/B = the two sides differ: the left
        or upper room's side first, . = wall on that side.
Doors on sides with no room of the level: $7C S open.
```

- Cellar `$0F` (treasure cellar): both exits lead to `$69`, whose stairs lead here; not in the level's cellar list; item: raft ($0C). [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`levelblock-uw-1to6-q1.md`](../data/levelblock-uw-1to6-q1.md)
<!-- /generated:map -->

# Rooms

Generated from the level data by `tools/romdata.py`; do not edit between the markers.

<!-- generated:rooms -->
| room | N | S | W | E | monsters | item | secret | block | dark | layout | floor | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $2A | wall | wall | wall | shutter | 8 x Keese blue ($1B) | key ($19) at $80,$90 | all dead: shutters open | - | - | $1D | blocks | - |
| $2B | wall | open | open | wall | 1 x dungeon person ($4B) | - | - | - | - | $26 | blank | person $4B says: DID YOU GET THE SWORD / FROM THE OLD MAN ON / TOP OF THE WATERFALL? |
| $3B | shutter | key | wall | wall | 1 x Trap ($4A), 3 x (empty) ($00), 2 x Zol ($13) | - | push block: shutters open | B | - | $0A | blocks | - |
| $3D Triforce | wall | open | wall | wall | none | Triforce piece ($1B) at $80,$90 | - | - | - | $29 | blocks, statues | - |
| $49 | wall | open | wall | open | 3 x Bubble ($2B), 2 x Zol ($13), 3 x Keese blue ($1B) | key ($19) at $80,$90 | - | - | - | $00 | plain | - |
| $4A | wall | open | open | key | 5 x Keese blue ($1B) | bomb ($00) at $80,$90 | all dead: shutters open, room item appears | - | - | $0F | blocks | - |
| $4B | key | open | key | key | 3 x Zol ($13) | key ($19) at $80,$90 | all dead: shutters open, room item appears | - | - | $08 | blocks | - |
| $4C | wall | open | key | bomb | 1 x Trap ($4A), 3 x (empty) ($00), 2 x Zol ($13) | map ($17) at $80,$90 | - | - | - | $00 | plain | boss sound $40 |
| $4D boss | shutter | shutter | bomb | wall | 1 x Manhandla ($3C) | heart container ($1A) at $80,$90 | all dead: shutters open, room item appears | - | - | $25 | sand | - |
| $59 | open | shutter | wall | key | 5 x Darknut red ($0B) | - | all dead: shutters open | - | - | $1D | blocks | - |
| $5A | open | wall | key | open | 1 x Trap ($4A), 3 x (empty) ($00), 4 x Keese blue ($1B) | compass ($16) at $80,$90 | - | - | - | $00 | plain | - |
| $5B | open | open | open | bomb | 3 x Darknut red ($0B) | bomb ($00) at $80,$90 | all dead: shutters open, room item appears | - | - | $00 | plain | - |
| $5C | open | wall | bomb | shutter | 3 x Darknut red ($0B) | - | all dead: shutters open | - | - | $1F | blocks | - |
| $5D | shutter | wall | shutter | wall | 3 x Bubble ($2B), 2 x Zol ($13), 3 x Keese blue ($1B) | 5 rupees ($0F) at $20,$60 | all dead: shutters open, room item appears | - | - | $1E | blocks | boss sound $40 |
| $69 | open | wall | wall | wall | 8 x Darknut red ($0B) | bomb ($00) at $80,$90 | all dead: shutters open, room item appears | - | - | $1B | blocks, stairs | stairs lead to cellar $0F |
| $6B | open | open | wall | wall | 5 x Zol ($13) | key ($19) at $80,$90 | all dead: shutters open, room item appears | - | - | $11 | blocks | - |
| $7B | open | wall | wall | open | 6 x Zol ($13) | key ($19) at $80,$90 | - | - | - | $03 | blocks | - |
| $7C start | wall | open | open | wall | none | - | - | - | - | $21 | statues, sand | - |
| $0F cellar | - | - | - | - | 4 x Keese blue ($1B) | raft ($0C) at $80,$90 | - | - | - | $3F | treasure cellar | left exit to $69, right exit to $69; Link comes out at $60,$8D |
<!-- /generated:rooms -->

# Floor plans

<!-- generated:floors -->
One character per floor square, 12 columns x 7 rows inside the walls (left column = west, top row =
north): `.` floor (tile `$74`), `B` block (tile `$B0`; a push block object is placed on one in a room
with a push block), `W` water (tile `$F4`), `S` statue (tiles `$94` and `$B4`), `>` stairs (tile `$70`),
`:` sand (tile `$68`), `_` blank square (tile `$24`). [`RoomLayout_UWSquareTiles`](../src/bank5/uw-floor-push-block.asm#L63), [`RoomLayout_LayOutUWFloor`](../src/bank5/uw-floor-push-block.asm#L73), [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L234), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)
Stairs that a secret makes appear are not part of the layout.

```
layout $00  rooms: $49,$4C,$5A,$5B
............
............
............
............
............
............
............
```

```
layout $03  rooms: $7B
............
..B......B..
............
............
............
..B......B..
............
```

```
layout $08  rooms: $4B
............
.BBBBBBBBBB.
..........B.
..........B.
..........B.
.BBBBBBBBBB.
............
```

```
layout $0A  rooms: $3B
............
............
............
....B..B....
............
............
............
```

```
layout $0F  rooms: $4A
............
............
BBBBBBBBBBBB
............
BBBBBBBBBBBB
............
............
```

```
layout $11  rooms: $6B
..B........B
.B...B....B.
....B....B..
...B....B...
..B....B....
.B....B...B.
B........B..
```

```
layout $1B  rooms: $69
........BBBB
........BBBB
.........BBB
...........>
.........BBB
........BBBB
........BBBB
```

```
layout $1D  rooms: $2A,$59
............
............
..BB....BB..
..BB....BB..
..BB....BB..
............
............
```

```
layout $1E  rooms: $5D
............
............
.....BB.....
.....BB.....
.....BB.....
............
............
```

```
layout $1F  rooms: $5C
............
.BB......BB.
............
.....BB.....
............
.BB......BB.
............
```

```
layout $21  rooms: $7C
............
.S..S..S..S.
............
.S..S..S..S.
.....::.....
.S.:S::S:.S.
...::::::...
```

```
layout $25  rooms: $4D
::::::::::::
::::::::::::
::::::::::::
::::::::::::
::::::::::::
::::::::::::
::::::::::::
```

```
layout $26  rooms: $2B
____________
____________
____________
____________
____________
____________
____________
```

```
layout $29  rooms: $3D
............
.BBBBBBBBBB.
.B..S..S..B.
.B.S....S.B.
.B........B.
.BBBB..BBBB.
............
```

```
layout $3F (treasure cellar)  rooms: $0F
     c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 cA cB cC cD cE cF
r0   02 02 02 00 02 02 02 02 02 02 02 02 02 02 02 02
r1   03 03 03 01 03 03 03 03 03 03 03 03 03 03 03 03
r2   03 03 03 01 03 03 03 03 03 03 03 03 03 03 03 03
r3   03 03 03 01 03 03 03 02 02 02 02 02 02 02 03 03
r4   03 03 03 01 03 03 03 02 02 02 02 02 02 02 03 03
r5   03 03 03 01 03 03 03 0C 0C 0C 0C 0C 0C 0C 03 03
r6   03 03 03 01 03 03 03 03 03 03 03 01 03 03 03 03
r7   03 03 02 01 02 02 02 02 02 02 02 01 02 02 03 03
r8   03 03 0C 01 0C 0C 0C 0C 0C 0C 0C 01 0C 0C 03 03
r9   03 03 03 03 03 03 03 03 03 03 03 03 03 03 03 03
r10  03 03 03 03 03 03 03 03 03 03 03 03 03 03 03 03
squares (first tile): 00 = $24, 01 = $6F, 02 = $F3, 03 = $FA, 0C = $F3
```

A cellar is laid out by the overworld routine: 16 columns x 11 rows of overworld square indexes,
as in [`room-layouts.md`](../data/room-layouts.md) (sections "Cave and cellar layouts" and "Overworld squares").
[`Mode9Cellar_Init_LayOutCellar`](../src/bank5/layout-room-ow.asm#L418), [`RoomLayout_SubroomLayoutAddrs`](../src/bank5/layout-room-ow.asm#L387)
<!-- /generated:floors -->

# Monsters and boss

| Monster (type) | Rooms and count | HP | Hurt by | Does to Link | Section in enemies.md |
| --- | --- | --- | --- | --- | --- |
| Zol (`$13`) | `$7B` 6, `$6B` 5, `$4B` 3, `$3B` 2, `$4C` 2, `$49` 2, `$5D` 2 | `$20` | everything; damage `$10` (wooden sword, fire) splits it into 2 Gels, `$20` or more kills; boomerang stuns | touch 1 heart | "Zol" |
| Gel (`$14`, from a split Zol) | wherever a Zol is split | 0 | any hit, the boomerang included | touch half a heart | "Gel" |
| Keese, blue (`$1B`) | `$2A` 8, `$4A` 5, `$5A` 4, `$49` 3, `$5D` 3, cellar `$0F` 4 | 0 | any hit, the boomerang included | touch half a heart | "Keese (blue, red, black)" |
| Darknut, red (`$0B`) | `$69` 8, `$59` 5, `$5B` 3, `$5C` 3 | `$40` | sword, sword beam, rod swing, bomb, and only when not hitting its front; never stunned | touch 1 heart | "Darknut (red, blue)" |
| Bubble (`$2B`) | `$49` 3, `$5D` 3 | - | nothing | touch: no damage, knockback, sword unusable for 151-160 frames | "Bubble (flashing, red, blue)" |
| Trap (`$49`, made by generator `$4A`) | `$3B` 4, `$4C` 4, `$5A` 4 (the corners) | - | nothing | touch 1 heart | "Trap" |
| Manhandla (`$3C`) | `$4D` 1 (four hands and a base) | `$40` per hand | sword, beam, rod, arrow, bomb; not boomerang or fire | touch 1 heart; fireballs 1 heart, no shield blocks them | "Manhandla" |
| Old man (`$4B`) | `$2B` 1 | - | cannot die | nothing; fireballs from his fires once he is hit | "Old men and shopkeepers (cave and dungeon persons)" |

- The HP and damage columns come from the type tables; the behaviour of each kind is in [enemies.md](../enemies.md) under the heading named in the last column. [`Obj_TypeHpPairs`](../src/bank7/object-update-table.asm#L105), [`Damage_ObjTypeDamageToLink`](../src/bank1/monster-collisions.asm#L200)
- **Darknuts (`$59`, the one room where they must die).** A hit does nothing when the weapon points straight at the Darknut's front (weapon direction and Darknut direction exactly opposite); from the side or from behind it lands and knocks the Darknut back `$40` pixels. Boomerang, arrows and fire are refused, so nothing stuns one. [`Damage_HandleWeaponHit@CheckDarknut`](../src/bank1/weapon-damage-death.asm#L157), [`Darknut_Init`](../src/bank4/darknut-gibdo.asm#L44)
- A Darknut walks at 0.5 pixel per frame and turns at grid points, often toward Link when he is within 8 pixels of its row or column: stand beside its lane, not in it, and strike as it passes; a bomb counts by the direction Link faced when he placed it. [`Wanderer_MoveAndTurn`](../src/bank4/walker-wanderer-common.asm#L224), [`Bomb_CheckBombOrFireVsMonster`](../src/bank1/sword-bomb-arrow-collision.asm#L126)
- `$5B` and `$69` are crossed, not cleared: in `$69` the walk from the north door to the stairs takes about 100 frames and the walk back from the cellar exit to the door less; a straight walk through the eight Darknuts cost 1 to 1.5 hearts each way in the traces. [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`Damage_ObjTypeDamageToLink`](../src/bank1/monster-collisions.asm#L200)
- **Zols with the wooden sword** become two Gels each, which run off sideways and must also die before a room counts as cleared; a split Zol drops nothing. With the white sword a Zol dies at once. [`Zol_CheckCollisions`](../src/bank4/zol-gel.asm#L170), [`Zol_State2Split`](../src/bank4/zol-gel.asm#L113), [`Secret_UpdateRoomAllDead`](../src/bank5/underworld-secrets.asm#L124)
- **Bubbles and Traps never count as monsters**: rooms `$49`, `$5D`, `$3B`, `$4C` and `$5A` are "all dead" when their Zols and Keese are. [`Secret_UpdateRoomAllDead`](../src/bank5/underworld-secrets.asm#L124)
- **Boss: Manhandla, room `$4D`**, entered on the route through the bombed west wall (Link stands at X `$20`, Y `$8D`). It is four hands around a base; only the hands can die, and the death of the last one is the kill. Each dead hand makes the rest faster: 0.5, 1.0, 1.5, 2.0 pixels per frame. [`Manhandla_Init`](../src/bank4/gleeok-manhandla-gohma-inits.asm#L155), [`Manhandla_CheckCollisions`](../src/bank4/manhandla.asm#L206), [`Manhandla_Update@LoopSegment`](../src/bank4/manhandla.asm#L72)
- Hits per hand: wooden sword 4, white sword 2, bomb blast 1. A blast reaches every hand whose middle is less than `$18` pixels from the bomb's middle on both axes, so one bomb next to the base can take several hands at once, and hands that die on the same frame speed the rest up only one step. [`Sword_DamagePoints`](../src/bank1/sword-bomb-arrow-collision.asm#L182), [`Bomb_CheckBombOrFireVsMonster`](../src/bank1/sword-bomb-arrow-collision.asm#L126)
- (trace) With two bombs: Link walked right from the wall to X `$50` and placed a bomb (it lies at X `$60`, Y `$8D`; the base was at `$78,$6E`), stepped back left; the blast killed three hands. A second bomb placed from X `$22` facing right as the last hand came toward him ended the fight. Link was hit twice (2 hearts). [`Bomb_Wield`](../src/bank1/bomb-candle-wield.asm#L63), [`Manhandla_CheckCollisions`](../src/bank4/manhandla.asm#L206)
- Hands shoot fireballs aimed at Link (1 heart, not blocked by any shield), more often the faster they move; keep moving sideways to their line. [`Manhandla_Update`](../src/bank4/manhandla.asm#L60), [`Fireball_Shoot`](../src/bank4/monster-shots-fireballs.asm#L58)
- When the last hand dies: the heart container appears at `$80,$90`, the shutters N (to the Triforce room `$3D`) and S (to `$5D`) open, and the boss roar heard in `$4C` and `$5D` stops. Manhandla drops at most one item (drop group 3). [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61), [`Shutter_OpenTriggered`](../src/bank5/shutters-door-flags.asm#L68), [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42), [`Drop_GroupRates`](../src/bank4/drop-items.asm#L122)

# Route

How it was established: the room graph was searched from the generated tables under the door rules of [mechanics.md](../mechanics.md) "Doorways, doors, keys, shutters and bomb walls" (fewest room entries, then fewest monsters that must die), and every step was then walked in the running game from saved positions of this level; where monsters had to be out of the way for a geometry check they were removed, so fights other than Manhandla's are not part of the traces. Keys are counted for a Link who arrives with none.

| Step | Room | Entered from | What to do | Leave by | Keys after | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `$7C` | overworld (S) | Nothing to fight. The W door's row is blocked by statues: up to Y `$9D`, left to X `$20`, up to Y `$8D`, left | W | 0 | statues do not shoot |
| 2 | `$7B` | E | Walk left along Y `$8D`: the key is taken at X `$88`; on to X `$78`, then up. The 6 Zols need not die | N | 1 | - |
| 3 | `$6B` | S | Diagonal rows of blocks: left to X `$70`, up to Y `$8D`, right to X `$80`, up to Y `$5D`, left to X `$78`, up. Optional: kill the 5 Zols, the key appears at `$80,$90` on that path | N | 1 (2) | - |
| 4 | `$5B` | S | Up to Y `$8D`, left. The 3 Darknuts need not die now | W | 1 | bombs here on step 12 |
| 5 | `$5A` | E | Left along Y `$8D`: the compass is taken at X `$88`. Bait the two left Traps (see below), then push into the W key door: 1 key | W (key door) | 0 | 4 Traps, 4 Keese |
| 6 | `$59` | E | Kill all 5 Darknuts: the S shutter opens. Blocks stand on the door row: go round by Y `$AD` (X `$C0` down to `$AD`, left to X `$78`, down) | S (shutter) | 0 | the only forced fight before the boss |
| 7 | `$69` | N | Down to Y `$8D`, right to X `$D0`: the stairs. The 8 Darknuts need not die | stairs | 0 | - |
| 8 | `$0F` | stairs | Ladder down to Y `$BD`, right to X `$B0`, ladder up to Y `$8D`, left to X `$88`: the raft. Back the same way, up the left ladder | left ladder | 0 | 4 Keese; raft |
| 9 | `$69` | cellar, at `$60,$8D` | Right to X `$78`, up | N | 0 | Darknuts are back |
| 10 | `$59` | S | Empty if it was cleared on step 6. Up to Y `$AD`, right to X `$C0`, up to Y `$8D`, right | E (open now) | 0 | - |
| 11 | `$5A` | W | Straight right along Y `$8D`; the left Traps close behind Link | E | 0 | - |
| 12 | `$5B` | W | Right to X `$78`, up. With no bomb in hand: kill the 3 Darknuts and take the 4 bombs at `$80,$90` first | N | 0 | - |
| 13 | `$4B` | S | Kill the 3 Zols: the key appears. Up to Y `$BD`, left to X `$20`, up to Y `$8D`, right to the key (X `$78`); back left, down to Y `$BD`, right to X `$D0`, up to Y `$8D`, right into the key door | E (key door) | 1, then 0 | ring of blocks |
| 14 | `$4C` | W | Straight right along Y `$8D`: the map is taken at X `$78`. Stop at X `$C0`, facing right, place a bomb; when the wall opens (about 80 frames), walk right | E (bombed wall) | 0 | 4 Traps, 2 Zols; 1 bomb |
| 15 | `$4D` | W | Kill Manhandla. Take the heart container at `$80,$90` (Y `$8D`, X `$78`), then up at X `$78` | N (shutter) | 0 | boss |
| 16 | `$3D` | S | Straight up to the Triforce piece | (level ends) | 0 | - |

- Rooms in order: `$7C`, `$7B`, `$6B`, `$5B`, `$5A`, `$59`, `$69`, `$0F`, `$69`, `$59`, `$5A`, `$5B`, `$4B`, `$4C`, `$4D`, `$3D`: 15 room entries after the start. [levelblock-uw-1to6-q1.md](../data/levelblock-uw-1to6-q1.md)
- **Key balance.** +1 `$7B`, -1 `$5A` W, +1 `$4B`, -1 `$4B` E: the route pays for itself. Taking `$6B`'s key as well (5 Zols, same path) leaves Link one key up; arriving with a key makes the `$4B` fight unnecessary. [`Door_TouchKey`](../src/bank5/touch-door.asm#L160)
- **Order that matters.** The `$7B` key must be in hand before `$5A`; the Darknuts of `$59` must die before the raft; the raft is fetched before the boss because the Triforce piece ends the level; a bomb must be in hand on entering `$4C` (the last bombs on the way are in `$5B`). [`Item_TakeClass0PerLevel`](../src/bank1/take-item.asm#L421), [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136)
- **The `$5A` W door stands between two Traps.** Pushing into it at X `$20` holds Link for the 8 frames the door takes to open, in the column where the two left Traps meet: walking straight in cost a heart (trace). Bait instead: walk left to X `$2D` (both left Traps charge), step back right to X `$40`, wait until they have crept back to Y `$7A` and `$A9` (about 37 frames after they turn), then walk in; the door opened and Link passed untouched (trace). [`Trap_Update_Full`](../src/bank1/trap-rupee-stash.asm#L189), [`Door_TouchKey`](../src/bank5/touch-door.asm#L160)
- **Trap columns.** A corner Trap charges along its wall when Link's X comes within `$0E` of X `$20` or `$D0` (or his Y within `$0E` of Y `$5D` or `$BD`), at 1.75 pixels per frame to the middle, and returns at 0.5. Entering `$5A` or `$4C` from the W door (X `$10`) and walking on without stopping outruns the left pair (trace); the bomb spot X `$C0` is outside the right pair's column. [`Trap_Update_Full`](../src/bank1/trap-rupee-stash.asm#L189), [`Trap_StartXs`](../src/bank1/trap-rupee-stash.asm#L112), [`Trap_AllowedDirs`](../src/bank1/trap-rupee-stash.asm#L167)
- **The bomb for `$4C` E.** The bomb lies `$10` pixels ahead of Link and must end its blast with X less than `$18` from `$D0` and Y less than `$18` from `$8D`: Link facing right on Y `$8D` at X `$B0`-`$C0`. The wall is passable 8 frames after the blast ends (trace: 79 frames after the button). The B item must be the bombs. [`Bomb_WallHotspotsX`](../src/bank7/bomb.asm#L60), [`Link_PlaceWeaponAhead`](../src/bank1/bomb-candle-wield.asm#L154), [`SelectedItemSlot`](../src/Variables.inc#L553)
- **Returning through `$59`.** Entering from `$69` closes the S shutter behind Link; with the room already cleared the "all dead" trigger fires again at once, and the E door, unlocked from `$5A` on step 5, is open from this side too (trace). If `$59` was refilled (after leaving the level) its Darknuts need not die for the E door. [`Secret_Trigger1AllDead`](../src/bank5/underworld-secrets.asm#L152), [`Door_Update`](../src/bank5/update-doors.asm#L60)
- **Alternative of the same length, two bombs and no key:** from `$5B` bomb the E wall (same spot, X `$C0` facing right), cross `$5C` north without killing its Darknuts, enter `$4C` from the south, bomb its E wall. It skips `$4B` and its key door and saves the 3 Zols, at the price of a second bomb (data and code only, not walked). [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136), [`Door_TouchBombable`](../src/bank5/touch-door.asm#L108)
- **After a death.** CONTINUE restarts Link in `$7C` with 3 hearts. He keeps the inventory (keys, bombs, the raft); unlocked doors, bombed walls and taken items stay; rooms he had cleared refill (partly cleared ones stay reduced), except a dead Manhandla. The route simply restarts at step 1 with fewer things to do. [`Mode8Continue_Update_Full@HandleActivated`](../src/bank5/mode8-continue-mode10-stairs.asm#L143), [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50), [mechanics.md](../mechanics.md) "Hearts, rings and death"
- **Wallmasters:** none in this level. [objlists.md](../data/objlists.md)

# Optional detours

- **`$6B` key** (on the route, step 3): kill 5 Zols; +1 key, no extra room. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61)
- **`$5B` bombs** (on the route, steps 4 and 12): kill 3 Darknuts; +4 bombs, no extra room. [`Item_Take_CheckClass1`](../src/bank1/take-item.asm#L337)
- **`$49` key:** from `$59` go N (open door, no kills needed) and come back: +2 room entries, +1 key, lying visible at `$80,$90`. The room has 3 Bubbles (a touch takes the sword away for 151-160 frames), 2 Zols and 3 Keese; nothing has to die. [`Bubble_Update`](../src/bank4/bubble-keese.asm#L93), [`SwordBlockedLongTimer`](../src/Variables.inc#L62)
- **Reaching `$59` without a key:** `$5A` N, `$4A` W, `$49` S, `$59` (+2 entries on the way in, and it collects the `$49` key); the E door of `$59` then still costs a key on the way back, or Link returns the long way. [`Door_TouchKey`](../src/bank5/touch-door.asm#L160)
- **`$4A` bombs:** from `$5A` go N and come back: +2 entries; kill 5 Keese (one hit each, the boomerang kills), +4 bombs. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61), [`Keese_Update`](../src/bank4/bubble-keese.asm#L163)
- **`$69` bombs:** kill all 8 Darknuts of the stairs room; +4 bombs, no extra room. Worth it only with a strong sword. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61)
- **`$5D` 5 rupees:** after the boss go S from `$4D` (its S shutter is open), kill 2 Zols and 3 Keese for the rupees at `$20,$60` and for the N shutter back: +2 entries, +5 rupees. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61), [`Item_Take5Rupees`](../src/bank1/take-item.asm#L328)
- **`$2B` old man and `$2A` key:** `$4B` N key door (-1 key), `$3B` (kill 2 Zols, push the block, N shutter), `$2B` (text only), `$2A` (the shutter closes behind Link; kill 8 Keese to get out; key visible), and back: +6 entries, keys -1 +1. Worth nothing but the hint, which points at the white sword cave. [`Secret_Trigger4BlockShutters`](../src/bank5/underworld-secrets.asm#L215), [`Person_UpdateUnderworld_Full`](../src/bank1/underworld-person-update.asm#L44)
- **`$5C`:** 3 Darknuts and no item; its E shutter (all dead) leads to `$5D`. It matters only for the routes without a key or without a bomb described above. [`Secret_Trigger1AllDead`](../src/bank5/underworld-secrets.asm#L152)

# Hazards and rules that matter here

- **Shutters** (room and side that is shut: `$2A` E, `$3B` N, `$4D` N and S, `$59` S, `$5C` E, `$5D` N and W): each opens by its own room's secret, "all dead" everywhere except `$3B` (push block). A shutter is never saved: it is closed again on every entry, and entering through one closes it behind Link. [`Shutter_OpenTriggered`](../src/bank5/shutters-door-flags.asm#L68), [`Door_Update`](../src/bank5/update-doors.asm#L60), [mechanics.md](../mechanics.md) "Doorways, doors, keys, shutters and bomb walls"
- Four of them are shutters on one side only: `$2A` E, `$3B` N, `$4D` N and `$59` S are open doorways seen from `$2B`, `$2B`, `$3D` and `$69`, so Link can always walk into those rooms and is shut in until their secret is met. [`Door_FindTypeByDirBit`](../src/bank5/fill-walls.asm#L55)
- **"All dead"** means no object of type `$01`-`$2A` or `$2E`-`$48` in slots `$01`-`$0C`: Bubbles and Traps do not count, Gels from split Zols do. [`Secret_UpdateRoomAllDead`](../src/bank5/underworld-secrets.asm#L124)
- **Push block (`$3B` only, off the route):** the block is the left one of the two on the middle row (floor column 4: X `$60`, Y `$90`). It moves only after both Zols are dead, when Link pushes it from an exactly aligned spot for `$10` frames; its arrival opens the N shutter. It can be pushed once per entry. [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L234), [`PushBlock_State0Idle`](../src/bank4/push-block.asm#L74), [`Secret_Trigger4BlockShutters`](../src/bank5/underworld-secrets.asm#L215), [mechanics.md](../mechanics.md) "Push blocks and room secrets"
- **Bomb walls:** `$4C` E / `$4D` W and `$5B` E / `$5C` W; they look like plain wall until opened, open from either side, and stay open for good. Hotspots: bomb within `$18` of `$D0,$8D` (E wall) or `$20,$8D` (W wall). [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136), [`Bomb_WallHotspotsX`](../src/bank7/bomb.asm#L60), [`Bomb_WallHotspotsY`](../src/bank7/bomb.asm#L64)
- **Key doors:** `$59` E / `$5A` W, `$4A` E / `$4B` W, `$4B` E / `$4C` W, `$3B` S / `$4B` N. `$4B` has three of them: only the E one is on the route, and a key pushed into the W or N door is wasted. [`Door_TouchKey`](../src/bank5/touch-door.asm#L160)
- **False walls:** none. **Dark rooms:** none. **Water:** none. **Wallmasters, Like-Likes, hungry Goriya:** none. [levelblock-uw-1to6-q1.md](../data/levelblock-uw-1to6-q1.md)
- **Traps** (`$3B`, `$4C`, `$5A`): four per room, in the corners of the floor at X `$20`/`$D0`, Y `$5D`/`$BD`; nothing hurts them; 1 heart per touch. The W and E doors are each between two of them. [`Trap_Init_Full`](../src/bank1/trap-rupee-stash.asm#L134), [`Trap_StartXs`](../src/bank1/trap-rupee-stash.asm#L112), [enemies.md](../enemies.md) "Trap"
- **Bubbles** (`$49`, `$5D`; type `$2B` only): a touch does no damage but blocks the A button for 151-160 frames; the B item still works. Neither room is on the route. [`Bubble_Update`](../src/bank4/bubble-keese.asm#L93), [`Link_HandleInput`](../src/bank5/link-handle-input.asm#L68)
- **Statues** stand in `$7C` (they block its door row) and `$3D`; neither layout (`$21`, `$29`) is one of the two whose statues shoot. [`Statue_RoomLayoutIds`](../src/bank4/statues.asm#L45)
- **Old man, `$2B`:** "DID YOU GET THE SWORD FROM THE OLD MAN ON TOP OF THE WATERFALL?". In this level a person only talks; Link is held while the text is typed and cannot walk above Y `$8D` in the room. Hitting him makes fireballs fly at Link from beside his two fires until the room is left. [`Person_UpdateUnderworld_Full`](../src/bank1/underworld-person-update.asm#L44), [`Person_CheckCollisions`](../src/bank1/underworld-person-update.asm#L89), [`Person_BlockLinkMovingUp`](../src/bank1/modeb-cave-save-sprites.asm#L289), [person-text.md](../data/person-text.md)
- **Stairs and the cellar:** the stairs of `$69` work only with Link on the grid at X `$D0`, Y `$8D`. Cellar `$0F` is not in the level's cellar list (all `$FF`): the lookup has no end test and runs on through the level info until it meets a byte, 72 bytes after the list's start, that names a room whose attribute A or B is `$69`. It is an item cellar: both exits lead back to `$69`, at `$60,$8D`. [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`LevelInfo_CellarRoomIdArray`](../src/Variables.inc#L726), [`Warp_CheckSubroomExit`](../src/bank5/cave-cellar-exits.asm#L43)
- In the cellar no doorway or bomb-wall code runs, its own monster list is ignored and four blue Keese are made at Y `$9D`. [`Spawn_AssignPositions`](../src/bank5/spawn-positions.asm#L64), [`Mode9Cellar_Init_LayOutCellar`](../src/bank5/layout-room-ow.asm#L418)
- **Boss roar:** rooms `$4C` and `$5D` (the neighbours of `$4D` used as ways in) play the roar while Manhandla lives. [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42)
- **Blocks on door rows:** `$59` (blocks at floor columns 2-3 and 8-9, rows 2-4), `$4B` (ring), `$6B` (diagonals), `$7C` (statues) cannot be crossed straight from door to door; the route table gives the way round. [`RoomLayout_LayOutUWFloor`](../src/bank5/uw-floor-push-block.asm#L73)

# Second quest

- In Q2 the dungeon shown as LEVEL-3 is internal level 2 with a different level block; internal level 3 is shown there as LEVEL-2, starts in room `$79` and is entered from overworld room `$34`, while room `$74` holds a shop. [levelinfo.md](../data/levelinfo.md), [levelblock-uw-1to6-q2.md](../data/levelblock-uw-1to6-q2.md), [levelblock-ow.md](../data/levelblock-ow.md)
- This document covers the first quest only; nothing above (rooms, items, route) applies to either Q2 dungeon. [`Level_BlockUW1to6Q2`](../src/bank6/level-blocks-common-data.asm#L78), [`Quest_Q2LevelInfoPatch3`](../src/bank6/quest2-replacements.asm#L107)
