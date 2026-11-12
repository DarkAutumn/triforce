# Level 1: the bow, the wooden boomerang and Aquamentus

Level 1 is entered from overworld room `$37` and has 17 rooms and one cellar. Link collects the bow (cellar `$7F`
under room `$22`), the wooden boomerang (room `$44`), the heart container left by the boss Aquamentus (room `$35`)
and the Triforce piece (room `$36`); the map and the compass are here too. Nothing has to be brought but a sword:
the six keys lying in the level open its six key doors, no room is dark, and all water can be walked round. What
makes it different: with no key brought every one of the six is needed, two of them are carried about by a
Stalfos, the bow lies behind a room of blade traps and a push block, and the room before the boss holds eight
Wallmasters that carry Link back to the start room.

Contents (generated; `L` = line in this file):

<!-- toc -->
- L27 At a glance
- L64 What Link must bring
- L79 What Link gets here
- L135 Map
- L162 Rooms
- L189 Floor plans
- L385 Monsters and boss
- L417 Route
- L466 Optional detours
- L476 Hazards and rules that matter here
- L496 Second quest
<!-- /toc -->

# At a glance

<!-- generated:facts -->
| Field | Value | Read by |
| --- | --- | --- |
| Internal level number | 1 | [`CurLevel`](../src/Variables.inc#L12) |
| Number shown on screen | LEVEL-1 | [`LevelInfo_LevelNumber`](../src/Variables.inc#L725) |
| Overworld entrance room (Q1) | $37 | [`Warp_HandleOW`](../src/bank5/warps-stairs-caves.asm#L178), [`Flute_SecretRoomIdsOW`](../src/bank7/flute.asm#L36) |
| Start room | $73 | [`LevelInfo_StartRoomId`](../src/Variables.inc#L722) |
| Triforce room | $36 (room item: Triforce piece ($1B)); the field places the compass marker | [`LevelInfo_TriforceRoomId`](../src/Variables.inc#L723), [`StatusBar_UpdateTriforceMarker`](../src/bank7/next-room-room-flags.asm#L51) |
| Boss room | $35 (1 x Aquamentus ($3D)) | [`LevelInfo_BossRoomId`](../src/Variables.inc#L728), [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42) |
| Cellars | $7F | [`LevelInfo_CellarRoomIdArray`](../src/Variables.inc#L726), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55) |
| Rooms | 17: 17 on the submenu map | [`LevelInfo_SubmenuMapMask`](../src/Variables.inc#L729) |
| Monster counts (index = level block C bits 7-6) | 3, 5, 6, 8 | [`LevelInfo_FoeCounts`](../src/Variables.inc#L716) |
| Key doors (room and side that has it) | 12 ($22 E, $23 W, $23 S, $33 N, $35 S, $45 N, $42 S, $52 N, $43 E, $44 W, $63 S, $73 N) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Bombable walls (room and side that has it) | 4 ($43 S, $53 N, $44 S, $54 N) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Shutters (room and side that has it) | 3 ($35 E, $42 W, $52 E) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| False walls (room and side that has it) | 0 | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Doorways whose two sides differ | $35 E shutter / $36 open, $41 E open / $42 shutter, $52 E shutter / $53 open | [`Door_FindTypeByDirBit`](../src/bank5/fill-walls.asm#L55) |
| Doors on sides with no room of the level | $73 S open | [`Room_CalcNext`](../src/bank7/next-room-room-flags.asm#L72) |
| Push blocks | 2 ($22, $42) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Dark rooms | 0 (none) | [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124) |
| Keys (room item) | 6 ($23, $33, $45, $53, $72, $74) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Other room items | bow ($0A): $7F (cellar); compass ($16): $54; map ($17): $43; heart container ($1A): $35; Triforce piece ($1B): $36; wooden boomerang ($1D): $44 | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |

Room by room data: [`levelblock-uw-1to6-q1.md`](../data/levelblock-uw-1to6-q1.md) (rows with `lvl` = `1` or `c1`).
<!-- /generated:facts -->

- Entrance: overworld room `$37` leads to level 1 (cave index 1) through an opening that is part of the room's layout; the room has no tile object and no secret, so nothing has to be bombed, burned, pushed or played. [`Warp_HandleOW`](../src/bank5/warps-stairs-caves.asm#L178), [data/levelblock-ow.md](../data/levelblock-ow.md)
- Going in: Link steps on the opening from the square below it; the level loads when he stands at X `$70`, Y `$7D` (trace: in from the east edge of `$37` along Y `$8D` to X `$70`, then up one square; 184 frames to control in the level). [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`Warp_StartStairsMode`](../src/bank5/warps-stairs-caves.asm#L131)
- On entry Link is put in room `$73` at X `$78`, Y `$DD` facing up and is walked up to Y `$CD` (trace), in the S doorway. [`LevelInfo_StartY`](../src/Variables.inc#L717), [`Mode3Unfurl_Init_Sub8_LayOutRoomPlaceLink`](../src/bank5/mode3-mode6-init.asm#L129)
- Walking out: the S door of `$73` has no room behind it, so going through it loads the overworld; Link appears in `$37` at X `$70`, Y `$7D` facing down (trace). [`Room_CalcNextRoomForDoor`](../src/bank5/next-room-calc.asm#L76), [`Mode_LeaveLevelToOW`](../src/bank5/next-room-calc.asm#L107)
- End of the level: taking the Triforce piece in `$36` starts mode `$12`: hearts refilled, curtain, then the overworld at the same spot of `$37` (trace: 602 frames from the pickup to control, hearts full). [`Item_TakeClass0PerLevel`](../src/bank1/take-item.asm#L421), [`Mode12EndLevel_Update_Sub4LeaveLevel`](../src/bank5/mode12-lay-out-room.asm#L156)
- Coming out does not send Link back in: the opening under him is ignored until he has finished one grid step. [`UndergroundExitType`](../src/Variables.inc#L99)
- Kept when Link leaves and comes back (room flags): key doors unlocked and walls bombed (from both sides), room items taken, rooms visited, and Aquamentus dead. [`RoomFlags_SetDoorOpened`](../src/bank5/shutters-door-flags.asm#L41), [`RoomFlags_SetItemTaken`](../src/bank1/take-item.asm#L140), [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50)
- Reset by every level load: the per-room kill counts and the room history, so every room but the boss room is full of monsters again; shutters are closed again on every room entry and a push block is back in place on every room entry. [`Mode2Load_Init`](../src/bank7/init-mode-dispatch.asm#L147), [`LevelKillCounts`](../src/Variables.inc#L480), [`Door_Update`](../src/bank5/update-doors.asm#L60), [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L234)

# What Link must bring

Nothing but a sword. Each item, with the proof:

- **Sword: yes, any grade.** Stalfos, Goriyas and Aquamentus only die to a weapon that does damage, and before the bow and its arrows the sword is the one Link has; the wooden sword is enough (Stalfos 2 hits, red Goriya 3, Aquamentus 6; white sword 1, 2, 3). [`Sword_DamagePoints`](../src/bank1/sword-bomb-arrow-collision.asm#L182), [`Obj_TypeHpPairs`](../src/bank7/object-update-table.asm#L105), [items.md](../items.md) "Sword (Wooden, White, Magic) and sword beam"
- **Keys: none.** Six keys lie in the level (`$72`, `$74`, `$53`, `$33`, `$23`, `$45`) and it has six key doors, each with a face in two rooms (`$73` N = `$63` S, `$52` N = `$42` S, `$33` N = `$23` S, `$23` W = `$22` E, `$43` E = `$44` W, `$45` N = `$35` S); the route below takes all six keys and opens all six doors, ending with 0. [`Door_TouchKey`](../src/bank5/touch-door.asm#L160), [`InvKeys`](../src/Variables.inc#L584)
- Keys are one counter for the whole game, so each key brought from elsewhere lets Link skip one key pickup: first `$74` (two room entries), then `$72` (two more), then the Goriya fight in `$23`. [`InvKeys`](../src/Variables.inc#L584), [items.md](../items.md) "Keys and Magic Key"
- **Bombs: none needed.** The two bombable walls (`$53` N = `$43` S, `$54` N = `$44` S) are shortcuts between rooms that open and key doors also join; no monster here needs a bomb. One bomb on `$53` N saves two rooms and one key (see "Route"). [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136), [`Door_TouchBombable`](../src/bank5/touch-door.asm#L108)
- **Candle: no.** No room of the level has the dark flag (level block byte E bit 7 is 0 in all 18 rooms), so the darkness code never runs here. [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124), [data/levelblock-uw-1to6-q1.md](../data/levelblock-uw-1to6-q1.md)
- **Ladder: no.** Water squares are only in `$23` and `$33`, and in both every door and the key can be reached over floor squares (floor plans, layouts `$15` and `$17`). [`Link_EndMoveAndAnimate@CheckLadder`](../src/bank7/link-animate-ladder.asm#L209), [items.md](../items.md) "Ladder"
- **Raft, recorder, food: no.** Docks are overworld objects, no room holds a Digdogger and none holds the hungry Goriya (the rooms table lists every monster). [`Dock_Update`](../src/bank4/dock-raft.asm#L39), [`Grumble_Update`](../src/bank7/object-stubs-destroy-monster.asm#L106)
- **Bow, arrows, rupees: no.** Nothing in the level is immune to the sword, no person here sells or charges anything, and the bow itself is found here. [`Arrow_Wield`](../src/bank5/wield-items.asm#L187), [`Person_UpdateUnderworld_Full`](../src/bank1/underworld-person-update.asm#L44)
- **Hearts: the starting 3 are enough in principle.** Touches cost half a heart (Keese, Gel, Stalfos, red Goriya, Wallmaster, Aquamentus's fireballs) or one heart (blade trap, a Goriya's boomerang, Aquamentus's body) without a ring; nothing in the level refills hearts except monster drops and the Triforce piece at the end. [`Damage_ObjTypeDamageToLink`](../src/bank1/monster-collisions.asm#L200), [mechanics.md](../mechanics.md) "Hearts, rings and death"
- The small shield Link starts with stops a Goriya's boomerang when he faces it and is not attacking; it does not stop Aquamentus's fireballs (that takes the magic shield, which is not needed). [`Link_CheckMonsterTouch_NoReset@FacingOpposite`](../src/bank1/link-collision-harm.asm#L138), [items.md](../items.md) "Shield (small, magic)"

# What Link gets here

## Bow (item `$0A`), cellar `$7F` under room `$22`

- **Where:** the room item of the treasure cellar `$7F`, at X `$80`, Y `$90` on the upper walkway; the cellar is reached by the stairs in the middle of room `$22`. [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)
- **To reach it:** `$22` lies behind two key doors (`$33` N and `$23` W). Its stairs are there from the start (the room has no secret trigger) but stand inside a diamond of eight blocks; the west block of the diamond (X `$60`, Y `$90`) is the room's push block and must be pushed up or down to open a way in, unless Link clips through a block (see "Route", "Short way in `$22`"). No monster has to die first: the room's only objects are four blade traps, which do not count. [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L234), [`PushBlock_State0Idle`](../src/bank4/push-block.asm#L74), [`Secret_UpdateRoomAllDead`](../src/bank5/underworld-secrets.asm#L124)
- **In the cellar:** Link comes in at the top of the left ladder (X `$30`, Y `$5D`); go down to the floor (Y `$BD`), right to the second ladder (X `$B0`), up to Y `$8D`, left to X `$80`: the bow is taken there (trace). Four blue Keese fly about. Back the same way and up the left ladder past Y `$40`. [`Mode9Cellar_Init_Sub8_EnterCellar`](../src/bank5/cave-cellar-enter.asm#L276), [`Warp_CheckSubroomExit`](../src/bank5/cave-cellar-exits.asm#L43), [`Spawn_AssignPositions`](../src/bank5/spawn-positions.asm#L64)
- Taken outside mode 5, the bow is held up: Link is halted for `$80` frames while the item song plays. [`Item_Take`](../src/bank1/take-item.asm#L259), [`Link_CheckLiftItem`](../src/bank7/status-bar-lift-item.asm#L162)
- **What it changes:** `InvBow` becomes 1. The bow alone does nothing and cannot be selected; what it does is make the arrow slot (B item 2) selectable, and that slot is only occupied when Link also owns arrows. [`InvBow`](../src/Variables.inc#L560), [`Inv_FindAndSelectOccupiedSlot_CheckArrowNeedsBow`](../src/bank5/submenu-items.asm#L344), [`Link_WieldNothing`](../src/bank5/wield-items.asm#L237)
- **What shooting needs besides the bow:** arrows as an inventory grade (`InvArrow` 1 wooden, 2 silver; bought once, never used up) and at least 1 rupee: every shot costs 1 rupee and is refused at 0. Wooden arrows cost 80 rupees in the shop cave of overworld rooms `$25`, `$44`, `$4A`, `$6F`; the silver arrow is the item of a level 9 cellar. [`Arrow_Wield`](../src/bank5/wield-items.asm#L187), [`InvArrow`](../src/Variables.inc#L559), [`InvRupees`](../src/Variables.inc#L583), [data/levelblock-ow.md](../data/levelblock-ow.md)
- **What an arrow does:** it flies straight at 3 pixels per frame until a room bound or a monster stops it, one at a time, damage `$20` wooden (a white sword hit) or `$40` silver; tiles such as water and blocks do not stop it. [`Arrow_CheckArrowOrRodVsMonster`](../src/bank1/sword-bomb-arrow-collision.asm#L249), [`Obj_MoveShot`](../src/bank1/shot-aim-boomerang-speed.asm#L49), [items.md](../items.md) "Arrows and Bow (wooden, silver)"
- **Where it is first needed:** not in this level (Aquamentus takes arrows, 3 wooden, but the sword does the same). Later it is required for Gohma, which only an arrow flying up into its open eye hurts (level 6 boss room `$1C`; level 8 rooms `$1E`, `$4D`), and for Ganon, whom only the silver arrow kills (level 9). A Pols Voice (levels 5 and 8) dies to any arrow at once. [`Gohma_HandleWeaponCollision`](../src/bank4/gohma.asm#L367), [`Ganon_CheckCollisions@CheckArrowCollision`](../src/bank4/ganon-draw-collisions.asm#L427), [`Collision_CheckShotVsMonster`](../src/bank1/sword-bomb-arrow-collision.asm#L287), [enemies.md](../enemies.md) "Gohma (blue, red)"

## Wooden boomerang (item `$1D`), room `$44`

- **Where:** room `$44`, at X `$80`, Y `$70`. It is hidden until the three red Goriyas of the room are dead (secret trigger 7), then appears with the "item appears" tune (trace). The room is behind the key door `$43` E (or the bombable wall `$54` N). [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61), [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)
- **What it changes:** `InvBoomerang` becomes 1 and B item 0 throws it (no cost, one at a time, in the held direction, diagonals included); wooden range `$31` pixels, then it returns. [`Boomerang_Wield`](../src/bank5/wield-items.asm#L124), [`InvBoomerang`](../src/Variables.inc#L602), [`Boomerang_RangeLimits`](../src/bank5/wield-items.asm#L85)
- It does damage 0: a monster that is not immune is stunned for about 150-160 frames (it neither moves nor hurts by touch), and a monster with HP 0 (Keese, Gel) dies. Immune monsters send it back with the parry tune. [`Collision_CheckWeaponPointVsMonster`](../src/bank1/weapon-damage-death.asm#L82), [`Damage_DealToMonster`](../src/bank1/weapon-damage-death.asm#L181), [`ObjStunTimer`](../src/Variables.inc#L52)
- While it is out it picks up dropped items (hearts, rupees, bombs, fairies) it passes over; it does not pick up room items such as keys. [`Drop_UpdateItem`](../src/bank4/drop-items.asm#L297), [`Item_TryTakeRoomItem`](../src/bank1/take-item.asm#L173)
- **Where it is first needed:** nowhere is it required. In this level it comes just before the eight Wallmasters of `$45` (a stunned Wallmaster stops and cannot grab Link); Aquamentus is immune to it. Later it is the cheap way through rooms of Keese and Gels (it kills them) and past walkers such as Stalfos and Goriyas (it stops them). [`Wallmaster_State1Trip`](../src/bank4/wallmaster.asm#L231), [`Aquamentus_Init`](../src/bank4/aquamentus-digdogger-dodongo-inits.asm#L40), [items.md](../items.md) "Boomerang (wooden, magic)"

## Heart container (item `$1A`), room `$35`

- **Where:** the boss room, at X `$C0`, Y `$90`; hidden until Aquamentus is dead (secret trigger 7), then it appears on the same frame the room counts as empty (trace). [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61), [`Secret_UpdateRoomAllDead`](../src/bank5/underworld-secrets.asm#L124)
- **What it changes:** `HeartValues` + `$11`: one more container and one more full heart (the other hearts are not refilled). [`Item_Take_CheckClass1@TakeHeartContainer`](../src/bank1/take-item.asm#L392), [`HeartValues`](../src/Variables.inc#L593)
- Containers gate the sword caves on the overworld: the white sword needs 5 containers, the magic sword 12; current health does not count. [`Cave_State2TalkOrShopOrDoorCharge@TryToTake`](../src/bank1/cave-shop.asm#L161), [items.md](../items.md) "Hearts, Heart Containers and Fairies"
- It lies on the row of the E door (Link's Y `$8D`), so walking from the west along that row to the E door takes it on the way (trace). [`Item_TryTake`](../src/bank1/take-item.asm#L206), [`Item_TryTakeRoomItem`](../src/bank1/take-item.asm#L173)
- If Link leaves without it, it starts hidden again on the next entry and is shown as soon as the room, which makes no monster any more, is counted as empty. [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51), [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61)

## Triforce piece (item `$1B`), room `$36`

- **Where:** room `$36`, drawn at X `$78`, Y `$90` inside a ring of blocks that is open at the bottom middle; the room has no monsters and its statues do not shoot. Reached only through the boss room's E shutter. [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51), [`Statue_RoomLayoutIds`](../src/bank4/statues.asm#L45)
- **What it changes:** bit 0 of `InvTriforce` is set and the level ends at once (mode `$12`): full heal, then Link stands on overworld room `$37`. [`Item_TakeClass0PerLevel`](../src/bank1/take-item.asm#L421), [`InvTriforce`](../src/Variables.inc#L597), [`Mode12EndLevel_StartFillingHearts`](../src/bank5/mode12-lay-out-room.asm#L118)
- All eight pieces are what opens the way past the first old man of level 9; each piece held also adds its level to the places the recorder's whirlwind goes, and the level's compass marker stops blinking. [`Person_InitUnderworldLevel9`](../src/bank1/underworld-person-init.asm#L172), [`Whirlwind_Summon`](../src/bank1/whirlwind.asm#L243), [items.md](../items.md) "Triforce pieces and the Triforce of Power"

## Keys (item `$19`), six

| Room | Monsters | How the key is got | Key position |
| --- | --- | --- | --- |
| `$72` | 3 blue Keese | hidden until all are dead | X `$A0`, Y `$C0` |
| `$74` | 5 Stalfos | carried by the Stalfos in object slot 1: kill that one (or touch it) | where that Stalfos is or died |
| `$53` | 5 Stalfos | hidden until all are dead | X `$80`, Y `$70` |
| `$33` | 3 Stalfos | carried by the Stalfos in object slot 1 | where that Stalfos is or died |
| `$23` | 3 red Goriyas | hidden until all are dead | X `$80`, Y `$70` |
| `$45` | 8 Wallmasters | lies in the open from the start | X `$A0`, Y `$C0` |

- **What a key changes:** `InvKeys` + 1; touching a key door with a key spends it and opens the door for good, from both sides. [`Item_Take_CheckClass1`](../src/bank1/take-item.asm#L337), [`Door_TouchKey`](../src/bank5/touch-door.asm#L160), [`Door_Update`](../src/bank5/update-doors.asm#L60)
- A room item is taken when Link's X is within 8 pixels of the item's X and his Y + 3 within 8 of its Y: stand on it, or beside it on the same row. [`Item_TryTake`](../src/bank1/take-item.asm#L206), [`Item_TryTakeRoomItem`](../src/bank1/take-item.asm#L173)
- A visible room item in a room whose object slot 1 holds a Stalfos is moved onto that Stalfos every frame and is drawn on it; the pickup test uses the same position, so walking into the carrier takes the key (and usually the half heart of its touch), and killing it leaves the key where it died (trace, rooms `$74` and `$33`). [`Item_MoveAndDrawRoomItem`](../src/bank7/room-item-draw.asm#L65), [`Item_DrawRoomItemOnMonster`](../src/bank7/room-item-draw.asm#L46)
- In `$53` the key is hidden while the Stalfos live, so it is not carried: it appears at its own position. [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51), [`Item_MoveAndDrawRoomItem`](../src/bank7/room-item-draw.asm#L65)

## Map (item `$17`, room `$43`) and compass (item `$16`, room `$54`)

- The map lies in the open at X `$C0`, Y `$90` of `$43`, a room the route crosses three times; the compass lies in the open at X `$C0`, Y `$90` of `$54`, off the route. [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51), [data/levelblock-uw-1to6-q1.md](../data/levelblock-uw-1to6-q1.md)
- **What they change:** only pictures. The map draws the level's rooms on the status bar; the compass draws a marker at the Triforce room `$36`. No door, monster or item depends on either. [`Inv_HasMap`](../src/bank5/submenu-map.asm#L153), [`Inv_HasCompass`](../src/bank5/submenu-map.asm#L143), [items.md](../items.md) "Map and Compass"

# Map

<!-- generated:map -->
```
         [22  ] K [23k ]
                    K
                  [33k ]            [35Bi]S/-[36T ]
                    |                 K
[41  ]-/S[42  ]---[43i ] K [44i ]---[45k ]
           K        X        X
         [52  ]S/-[53k ]---[54i ]
                    |
                  [63  ]
                    K
         [72k ]---[73S ]---[74k ]

[73S] = room $73 and its marks: S start room, B boss room, T Triforce room,
        k a key lies here, i another room item lies here, h not shown on the submenu map.
Between two rooms, the door: --- or | open, K key door, S shutter, X bombable wall,
        F false wall (walk-through), nothing = wall. A/B = the two sides differ: the left
        or upper room's side first, . = wall on that side.
Doors on sides with no room of the level: $73 S open.
```

- Cellar `$7F` (treasure cellar): both exits lead to `$22`, whose stairs lead here; item: bow ($0A). [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`levelblock-uw-1to6-q1.md`](../data/levelblock-uw-1to6-q1.md)
<!-- /generated:map -->

# Rooms

Generated from the level data by `tools/romdata.py`; do not edit between the markers.

<!-- generated:rooms -->
| room | N | S | W | E | monsters | item | secret | block | dark | layout | floor | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $22 | wall | wall | wall | key | 1 x Trap ($4A) | - | - | B | - | $1A | blocks, stairs | stairs lead to cellar $7F |
| $23 | wall | key | key | wall | 3 x Goriya ($06) | key ($19) at $80,$70 | all dead: shutters open, room item appears | - | - | $15 | water | - |
| $33 | key | open | wall | wall | 3 x Stalfos ($2A) | key ($19) at $80,$90 | - | - | - | $17 | water | - |
| $35 boss | wall | key | wall | shutter | 1 x Aquamentus ($3D) | heart container ($1A) at $C0,$90 | all dead: shutters open, room item appears | - | - | $04 | blocks | - |
| $36 Triforce | wall | wall | open | wall | none | Triforce piece ($1B) at $80,$90 | - | - | - | $29 | blocks, statues | - |
| $41 | wall | wall | wall | open | 1 x dungeon person ($4C) | - | - | - | - | $26 | blank | person $4C says: EASTMOST PENNINSULA / IS THE SECRET. |
| $42 | wall | key | shutter | open | 3 x Gel ($15) | - | push block: shutters open | B | - | $22 | blocks | - |
| $43 | open | bomb | open | key | 5 x Gel ($15) | map ($17) at $C0,$90 | - | - | - | $1F | blocks | - |
| $44 | wall | bomb | key | open | 3 x Goriya ($06) | wooden boomerang ($1D) at $80,$70 | all dead: shutters open, room item appears | - | - | $02 | blocks | - |
| $45 | key | wall | open | wall | 8 x Wallmaster ($27) | key ($19) at $A0,$C0 | - | - | - | $0D | blocks | boss sound $10 |
| $52 | key | wall | wall | shutter | 6 x Keese blue ($1B) | - | all dead: shutters open | - | - | $1E | blocks | - |
| $53 | bomb | open | open | open | 5 x Stalfos ($2A) | key ($19) at $80,$70 | all dead: shutters open, room item appears | - | - | $02 | blocks | - |
| $54 | bomb | wall | open | wall | 8 x Keese blue ($1B) | compass ($16) at $C0,$90 | - | - | - | $03 | blocks | - |
| $63 | open | key | wall | wall | 3 x Stalfos ($2A) | - | - | - | - | $1E | blocks | - |
| $72 | wall | wall | wall | open | 3 x Keese blue ($1B) | key ($19) at $A0,$C0 | all dead: shutters open, room item appears | - | - | $00 | plain | - |
| $73 start | key | open | open | open | none | - | - | - | - | $21 | statues, sand | - |
| $74 | wall | wall | open | wall | 5 x Stalfos ($2A) | key ($19) at $C0,$90 | - | - | - | $1D | blocks | - |
| $7F cellar | - | - | - | - | 4 x Keese blue ($1B) | bow ($0A) at $80,$90 | - | - | - | $3F | treasure cellar | left exit to $22, right exit to $22; Link comes out at $60,$AD |
<!-- /generated:rooms -->

# Floor plans

<!-- generated:floors -->
One character per floor square, 12 columns x 7 rows inside the walls (left column = west, top row =
north): `.` floor (tile `$74`), `B` block (tile `$B0`; a push block object is placed on one in a room
with a push block), `W` water (tile `$F4`), `S` statue (tiles `$94` and `$B4`), `>` stairs (tile `$70`),
`:` sand (tile `$68`), `_` blank square (tile `$24`). [`RoomLayout_UWSquareTiles`](../src/bank5/uw-floor-push-block.asm#L63), [`RoomLayout_LayOutUWFloor`](../src/bank5/uw-floor-push-block.asm#L73), [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L234), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)
Stairs that a secret makes appear are not part of the layout.

```
layout $00  rooms: $72
............
............
............
............
............
............
............
```

```
layout $02  rooms: $44,$53
............
............
..B......B..
............
..B......B..
............
............
```

```
layout $03  rooms: $54
............
..B......B..
............
............
............
..B......B..
............
```

```
layout $04  rooms: $35
.......BBBBB
.........BBB
..........BB
............
..........BB
.........BBB
.......BBBBB
```

```
layout $0D  rooms: $45
............
.B.B.BB.B.B.
............
.B.B.BB.B.B.
............
.B.B.BB.B.B.
............
```

```
layout $15  rooms: $23
............
.WWW....WWW.
.W........W.
.W.WWWWWW.W.
.W........W.
.WWW....WWW.
............
```

```
layout $17  rooms: $33
WWWWW..WWWWW
W...W....W.W
W.W.W.WW.W.W
..W.W..W.W..
W.W.WW.W.W.W
W.W....W...W
WWWWW..WWWWW
```

```
layout $1A  rooms: $22
............
......B.....
.....B.B....
....B.>.B...
.....B.B....
......B.....
............
```

```
layout $1D  rooms: $74
............
............
..BB....BB..
..BB....BB..
..BB....BB..
............
............
```

```
layout $1E  rooms: $52,$63
............
............
.....BB.....
.....BB.....
.....BB.....
............
............
```

```
layout $1F  rooms: $43
............
.BB......BB.
............
.....BB.....
............
.BB......BB.
............
```

```
layout $21  rooms: $73
............
.S..S..S..S.
............
.S..S..S..S.
.....::.....
.S.:S::S:.S.
...::::::...
```

```
layout $22  rooms: $42
............
............
............
.....B......
............
............
............
```

```
layout $26  rooms: $41
____________
____________
____________
____________
____________
____________
____________
```

```
layout $29  rooms: $36
............
.BBBBBBBBBB.
.B..S..S..B.
.B.S....S.B.
.B........B.
.BBBB..BBBB.
............
```

```
layout $3F (treasure cellar)  rooms: $7F
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

| Monster (type) | Rooms (count) | HP | Killed or stunned by | Does to Link | Section in [enemies.md](../enemies.md) |
| --- | --- | --- | --- | --- | --- |
| Keese, blue (`$1B`) | `$72` (3), `$52` (6), `$54` (8), cellar `$7F` (4) | 0 | any weapon hit, the boomerang included | touch: half a heart | "Keese (blue, red, black)" |
| Gel (`$15`) | `$42` (3), `$43` (5) | 0 | any weapon hit, the boomerang included | touch: half a heart | "Gel" |
| Stalfos (`$2A`) | `$74` (5), `$63` (3), `$53` (5), `$33` (3) | `$20` | wooden sword 2 hits, white sword 1, bomb 1, arrow 1; boomerang stuns | touch: half a heart (it does not shoot in Q1) | "Stalfos" |
| Goriya, red (`$06`) | `$23` (3), `$44` (3) | `$30` | wooden sword 3 hits, white sword 2, bomb 1, wooden arrow 2; boomerang stuns | touch: half a heart; its boomerang: 1 heart | "Goriya (blue, red)" |
| Wallmaster (`$27`) | `$45` (8) | `$20` | wooden sword 2 hits, white sword 1, only while it is out of the wall; boomerang stuns | touch: half a heart, and it carries Link to room `$73` | "Wallmaster" |
| Blade trap (`$49`, made by `$4A`) | `$22` (4) | none | nothing | touch: 1 heart | "Trap" |
| Old man (`$4C`) | `$41` (1) | none | nothing; hitting him starts fireballs | fireballs after he is hit | "Old men and shopkeepers (cave and dungeon persons)" |
| Aquamentus (`$3D`) | `$35` (1) | `$60` | wooden sword 6 hits, white 3, magic 2 (beams count the same); wooden arrow 3; bomb blast 2; boomerang and fire do nothing | body: 1 heart; fireball: half a heart | "Aquamentus" |

- Counts are what a first entry makes; a room Link has partly cleared makes only the rest while he stays in the level, and a cleared room stays empty while it is among the last six rooms entered. [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50), [`RoomHistory`](../src/Variables.inc#L533)
- Keese and Gels never drop anything; a Stalfos in object slot 1 (the key carrier) never drops anything either; red Goriyas are in the drop group that holds bombs. [`Drop_NoDropTypes`](../src/bank4/drop-items.asm#L96), [`Drop_SetUpItem@Found`](../src/bank4/drop-items.asm#L206), [`Drop_Group2Types`](../src/bank4/drop-items.asm#L111)
- **Room `$33` (3 Stalfos, water maze):** the floor is a set of one-square corridors between water; from the S door the way to the N door is up at X `$80` to Y `$8D`, one square left (X `$70`), up to the top row, then X `$78` (trace). The key rides on a Stalfos that may be in a side corridor (all corridors join through the bottom row); in a corridor a Stalfos can only be met head on. [`RoomLayout_LayOutUWFloor`](../src/bank5/uw-floor-push-block.asm#L73), [`Wanderer_MoveAndTurn`](../src/bank4/walker-wanderer-common.asm#L224)
- **Rooms `$23` and `$44` (3 red Goriyas each, must all die):** a Goriya turns toward Link when he is `$50` pixels or less away on the longer axis and a red one throws on about 2 frames in 256 while it faces him; it stands still until its boomerang is back, which is the time to hit it. The boomerang flies over water and blocks. [`Goriya_Update`](../src/bank4/goriya.asm#L45), [`Walker_SetInputDirAndTryThrowBoomerang`](../src/bank4/goriya.asm#L156)
- In `$23` a bar of water (row Y `$90`, X `$50`-`$A0`) lies between the S door and the key at X `$80`, Y `$70`: walk round its end (column X `$40` or X `$B0`) into the inner ring, or come down from the top row through the gap at X `$60`-`$90`. [`RoomLayout_LayOutUWFloor`](../src/bank5/uw-floor-push-block.asm#L73), [`Item_TryTake`](../src/bank1/take-item.asm#L206)
- **Room `$22` (blade traps):** the four traps sit in the corners (X `$20` or `$D0`, Y `$5D` or `$BD`) and charge along a wall when Link is within `$0D` pixels of their row or column, at 1.75 pixels per frame to the middle line, then creep back at 0.5. Walking straight in from the E door is safe (trace: the two east traps charge and meet behind him); so are floor rows Y `$6D`-`$AD` and columns X `$30`-`$C0`. [`Trap_Update_Full`](../src/bank1/trap-rupee-stash.asm#L189), [`Trap_AllowedDirs`](../src/bank1/trap-rupee-stash.asm#L167)
- **Room `$45` (8 Wallmasters):** see "Hazards": one comes out each time Link stands on a wall line; a grabbed Link is back in `$73`. [`Wallmaster_Update@CheckLinkAtWall`](../src/bank4/wallmaster.asm#L111), [`Wallmaster_State1Trip`](../src/bank4/wallmaster.asm#L231)

**The boss: Aquamentus, room `$35`**

- It stands at the right of the room: Y fixed at `$80`, X pacing between `$88` and `$C7` at 1 pixel every 8 frames; it never comes to Link. [`Aquamentus_Move`](../src/bank4/aquamentus.asm#L86), [`Aquamentus_Init`](../src/bank4/aquamentus-digdogger-dodongo-inits.asm#L40)
- Every weapon is tested against one point, its head at (X + 8, Y + 8); the body below does not count. The sword blade reaches it from the row Y `$7D`, Link 16 to 24 pixels left of its X, facing right (trace: three white sword hits from there; from the row Y `$8D` the blade never hit). [`Obj_GetMiddle`](../src/bank1/monster-collisions.asm#L173), [`Sword_CheckMonsterCollision`](../src/bank1/sword-bomb-arrow-collision.asm#L192)
- At full health the sword beam does the sword's damage from across the room and hits from the rows Y `$7D` to `$8D` (trace: three white sword beams from X `$48`, Y `$8D` killed it, Link unhurt). [`SwordShot_Make`](../src/bank7/sword-and-rod.asm#L327), [`Sword_DamagePoints`](../src/bank1/sword-bomb-arrow-collision.asm#L182)
- After each hit it flashes for about 32 frames and nothing can hit it: with the wooden sword the fight is 6 hits at least 32 frames apart. [`Shove_Begin`](../src/bank1/shove-knockback.asm#L65), [`Obj_DecInvincibilityTimer`](../src/bank7/object-stubs-destroy-monster.asm#L119)
- It breathes three fireballs at Link every `$70`-`$7F` or `$F0`-`$FF` frames (the first volley 6 frames after it appears); each aims once at where Link is, waits about 16 frames, then flies straight, the outer two spreading up and down. Move one square off the line after it opens its mouth. [`Aquamentus_Shoot`](../src/bank4/aquamentus.asm#L166), [`Fireball_Update`](../src/bank4/monster-shots-fireballs.asm#L286)
- What it leaves: the heart container (the room item, X `$C0`, Y `$90`) and, by the ordinary drop rule, sometimes a heart, rupee or fairy. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61), [`Drop_SetUpItem`](../src/bank4/drop-items.asm#L157)
- What opens: the E shutter, the only way into the Triforce room, as soon as the room counts as empty (the open command takes two steps 8 frames apart; trace: open within 40 frames of the death). The S door is the key door Link came in by and stays open. [`Secret_RunTrigger_JumpTable`](../src/bank5/underworld-secrets.asm#L105), [`Shutter_OpenTriggered`](../src/bank5/shutters-door-flags.asm#L68), [`Door_Update`](../src/bank5/update-doors.asm#L60)
- Aquamentus is non-recurring: once killed it is never made again, in this visit or a later one, and the roar heard in `$45` stops. [`RoomHistory_SaveKillCountUW`](../src/bank5/room-history-uw.asm#L110), [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42)

# Route

How this was established: the room graph was built from the level data (rooms table above) and searched under the
door rules of [mechanics.md](../mechanics.md) "Doorways, doors, keys, shutters and bomb walls"; the search result is the table.
Each room of the table was then played in the running game from its entry door: the fights, pickups, pushes, door
and stairs steps marked (trace) were carried out there, room by room, not as one unbroken run. Link carried a white
sword and five heart containers in those runs.

The route assumes nothing is brought (0 keys, 0 bombs): 21 room entries after the start room. Positions are Link's
X, Y; door rows and columns: W and E doors at Y `$8D`, N and S doors at X `$78`.

| Step | Room | Entered from | What to do | Leave by | Keys after | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `$73` | overworld (S) | nothing here | W, open | 0 | no monsters; statues do not shoot |
| 1 | `$72` | E | kill the 3 blue Keese; the key appears at X `$A0`, Y `$C0`; take it | E, open | 1 | dead end |
| 2 | `$73` | W | cross | E, open | 1 | |
| 3 | `$74` | W | kill the Stalfos that carries the key (object slot 1; the key is drawn on it), take the key where it died | W, open | 2 | 5 Stalfos; the others need not die |
| 4 | `$73` | E | use a key on the N door | N, key | 1 | |
| 5 | `$63` | S | cross; go round the block of six squares in the middle | N, open | 1 | 3 Stalfos; no need to fight |
| 6 | `$53` | S | kill all 5 Stalfos; the key appears at X `$80`, Y `$70`; take it | W, open | 2 | with a bomb: N wall instead (see below) |
| 7 | `$52` | E | use a key on the N door (go round the middle block) | N, key | 1 | 6 blue Keese; no need to fight; the E side is a shutter that has closed behind Link |
| 8 | `$42` | S | cross | E, open | 1 | 3 Gels; W is a shutter (old man, optional) |
| 9 | `$43` | W | cross; the map lies at X `$C0`, Y `$90`, on the door row east of the middle block (optional) | N, open | 1 | 5 Gels |
| 10 | `$33` | S | kill the Stalfos that carries the key, take it; from the bottom row go up at X `$80` to Y `$8D`, left to X `$70`, up to Y `$5D`, right to X `$78`; use a key on the N door (trace) | N, key | 1 | water maze, no ladder needed |
| 11 | `$23` | S | kill all 3 red Goriyas; the key appears at X `$80`, Y `$70`: from row Y `$9D` go to X `$40`, up to Y `$7D`, right to X `$80`, up to Y `$6D` (key taken); on up to Y `$5D`, left to X `$20`, down to Y `$8D`; use a key on the W door (trace) | W, key | 1 | |
| 12 | `$22` | E | walk W to X `$B0`, S to Y `$AD`, W to X `$90`, S to Y `$B5`, W to X `$70`, N to Y `$AD`, W to X `$60`; hold Up: Link stops under the block at Y `$95`, the block starts to move 26 frames after Up was first held and has arrived at frame 57; go on up to Y `$8D`, then right to X `$80`: the stairs (trace) | stairs | 1 | 4 blade traps; do not push the block from its west side (pushed E it walls the stairs in) |
| 13 | `$7F` | stairs | down the left ladder to Y `$BD`, right to X `$B0`, up to Y `$8D`, left to X `$80`: the bow; back the same way and up the left ladder (trace) | left ladder | 1 | 4 blue Keese; item cellar |
| 14 | `$22` | cellar | Link stands at X `$60`, Y `$AD`, below the block (back in place); go E to X `$70`, S to Y `$B5`, E to X `$90`, N to Y `$AD`, E to X `$B0`, N to Y `$8D`, E (trace) | E, open now | 1 | |
| 15 | `$23` | W | cross to the S door | S, open now | 1 | empty if cleared |
| 16 | `$33` | N | down the corridor | S, open | 1 | Stalfos left alive are still here |
| 17 | `$43` | N | use a key on the E door | E, key | 0 | |
| 18 | `$44` | W | kill all 3 red Goriyas; the boomerang appears at X `$80`, Y `$70`; take it | E, open | 0 | S is a bombable wall |
| 19 | `$45` | W | E to X `$20`, S to Y `$9D`, E to X `$90`, S to Y `$B5`, E to X `$A0`: the key is taken; W to X `$90`, N to Y `$5D`, W to X `$78`, use the key on the N door (trace) | N, key | 0 | 8 Wallmasters; keep moving |
| 20 | `$35` | S | kill Aquamentus (stand on row Y `$7D` left of its head, or shoot beams); take the heart container at X `$C0`, Y `$90` | E, shutter (opens when it dies) | 0 | |
| 21 | `$36` | W | E to X `$20`, S to Y `$BD`, E to X `$78`, N: the Triforce piece is taken at Y `$95` (trace) | level ends | 0 | no monsters |

- **Key balance:** found 6 (`$72`, `$74`, `$53`, `$33`, `$23`, `$45`), used 6 (`$73` N, `$52` N, `$33` N, `$23` W, `$43` E, `$45` N), 0 left; the count never goes below 1 before a door that needs one because both start-room keys are fetched first. [`Door_TouchKey`](../src/bank5/touch-door.asm#L160), [`InvKeys`](../src/Variables.inc#L584)
- **With one bomb:** in `$53` stand at the N wall (X `$78`, Y `$5D`) facing up and place it; the wall opens 72 frames later and stays open (trace). Go N into `$43`, leaving out `$52` and `$42` and the `$52` N key door; then `$74` can be left out too: 17 room entries, 5 keys found and used. [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136), [`Bomb_WallHotspotsY`](../src/bank7/bomb.asm#L64)
- **Without the bow** (not recommended: later levels need it) steps 10-16 fall away: 13 room entries, keys of `$33` and `$23` not needed. [`Arrow_Wield`](../src/bank5/wield-items.asm#L187), [data/levelblock-uw-1to6-q1.md](../data/levelblock-uw-1to6-q1.md)
- **Order matters, keys:** take `$72` and `$74` before going N; from `$63` on Link cannot afford the doors ahead without them, and coming back costs four rooms each way. With keys in hand on entry they can be skipped (see "What Link must bring"). [`Door_TouchKey`](../src/bank5/touch-door.asm#L160), [`InvKeys`](../src/Variables.inc#L584)
- **Order matters, `$23`:** the key of `$23` is what pays for `$23` W; if Link holds a spare key the Goriya fight can be left out on the way in, but the key is then still hidden on the way back. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61), [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)
- **Order matters, `$22`:** the block moves once per room entry. Pushed up or down it opens the diamond; pushed right it lands on the only square between itself and the stairs. If that happens leave the room and come back: the block is reset on entry. [`PushBlock_State2Done`](../src/bank4/push-block.asm#L191), [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L234)
- **Short way in `$22`: the block clip** (no push, nothing needed but the d-pad). From the E door walk Left to X `$B0`, Down to Y `$AD`, Left to X `$A0`; then Up 14 frames, Right 1, Up 10 (Link stands half inside the east block at X `$A0`, Y `$8D`), Left 22: the stairs take him at X `$80` (trace: 117 frames from X `$E0`, Y `$8D` in the E doorway; the traps never move, no damage).
  Exact cadence, recovery after a miss, the other blocks and the reverse clip from above: [mechanics.md](../mechanics.md) `# Grid-point skip: block clip, ladder clip and their run-up`. The push-block way of step 12 stays valid. [`Walker_CheckTileCollision@CheckGridOffset`](../src/bank7/walker-collision-screen-edge.asm#L101), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)
- **Shutters on the route:** `$52` E (not needed: Link leaves N), `$35` E (Aquamentus). Both are closed again on every entry. [`Door_Update`](../src/bank5/update-doors.asm#L60), [`Shutter_OpenTriggered`](../src/bank5/shutters-door-flags.asm#L68)
- **Way back from the bow:** `$23` W and `$33` N stay unlocked; `$23`, `$33` and `$43` are still among the last six rooms entered (the cellar counts as one), so what was killed there stays dead (trace: `$23` empty on the way back). [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50), [`Room_RunEntryTasks`](../src/bank7/init-mode5-play.asm#L169)
- **After a death:** CONTINUE puts Link in `$73` with 3 hearts and everything he carried; unlocked doors, bombed walls and taken items stay; the room history is cleared, so cleared rooms are full again, except the boss room. Hidden items not yet taken (the keys of `$72`, `$53`, `$23`, the boomerang of `$44`) need their fights again. [`Mode8Continue_Init`](../src/bank5/mode-inits-obj-lists.asm#L49), [`RoomHistory_Clear`](../src/bank7/init-mode-dispatch.asm#L271), [mechanics.md](../mechanics.md) "Hearts, rings and death"
- **After a Wallmaster:** Link is in `$73` at the level's entry spot with the hearts he had left (trace: grabbed after standing 281 frames on the bottom wall line of `$45`, half a heart lost, the key he had just taken kept); the room history is cleared as after a death. The way back to `$45` is `$73` N, `$63`, `$53`, `$52` (N stays unlocked), `$42`, `$43` (E stays unlocked), `$44`: seven rooms, no keys. [`Wallmaster_State1Trip`](../src/bank4/wallmaster.asm#L231), [`Mode3Unfurl_Init_Sub1`](../src/bank7/init-mode-dispatch.asm#L294)

# Optional detours

- **Map, room `$43`:** no extra room; it lies on the door row east of the middle block (X `$C0`, Y `$90`), among 5 Gels. Worth the level's map picture only. [`Inv_HasMap`](../src/bank5/submenu-map.asm#L153), [items.md](../items.md) "Map and Compass"
- **Compass, room `$54`:** from `$53` through the open E door and back: 2 room entries, 8 blue Keese (no need to kill them), no keys. It lies at X `$C0`, Y `$90`. Worth the Triforce marker on the status bar map only. [`Inv_HasCompass`](../src/bank5/submenu-map.asm#L143), [`StatusBar_UpdateTriforceMarker`](../src/bank7/next-room-room-flags.asm#L51)
- **Old man, room `$41`:** from `$42`: kill the 3 Gels, push the block (X `$70`, Y `$90`; any direction, hold 16 frames; trace: from below, Link at X `$70`, Y `$9D` holding Up), the W shutter opens 7 frames after the block arrives; 2 room entries, no keys. He says "EASTMOST PENNINSULA IS THE SECRET." and nothing else happens; Link is halted about 216 frames while the text is typed (trace). Coming back E the shutter of `$42` has closed behind and the block is reset. [`Secret_Trigger4BlockShutters`](../src/bank5/underworld-secrets.asm#L215), [`Person_UpdateState_Textbox`](../src/bank1/person-textbox.asm#L51), [data/person-text.md](../data/person-text.md)
- **Second bomb wall, `$54` N = `$44` S:** with a bomb, `$53` E, `$54` N (stand at X `$78`, Y `$5D`, face up) reaches `$44` without the `$43` E key door: one room more than through `$43`, one key saved. [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136), [`Door_TouchBombable`](../src/bank5/touch-door.asm#L108)
- **Spare keys for later levels:** taking all six keys and using both bomb walls (2 bombs) needs only four doors (`$73` N, `$33` N, `$23` W, `$45` N): 21 room entries, Link leaves with 2 keys. A key is worth 80 or 100 rupees in the shops. [`InvKeys`](../src/Variables.inc#L584), [items.md](../items.md) "Keys and Magic Key"
- **A door without a key, `$52` N:** while a shutter is opening, a key door lets Link through unpaid. In `$52` stand at the N door (X `$78`, Y `$5D`) holding Up when the last Keese dies: as the E shutter opens Link passes N with the door still shut and no key spent (trace, with `InvKeys` set to 0 and the last Keese placed in front of Link's sword). With a key in hand, do not touch the door before the Keese is dead, or the key is spent the normal way. [`Door_TouchKey`](../src/bank5/touch-door.asm#L160), [`Door_Touch_BlockWhileTimer`](../src/bank5/touch-door.asm#L193), [`Shutter_OpenTriggered`](../src/bank5/shutters-door-flags.asm#L68)
- Monster drops are the only source of bombs, rupees and hearts in the level: red Goriyas (`$23`, `$44`) can drop bombs; Stalfos and Wallmasters drop from the group without bombs. [`Drop_Group2Types`](../src/bank4/drop-items.asm#L111), [`Drop_Group1Types`](../src/bank4/drop-items.asm#L105), [enemies.md](../enemies.md) "Drop tables"

# Hazards and rules that matter here

- **Shutters** (`$35` E, `$42` W, `$52` E): closed on every entry, also when Link has just come in through one; they open only by the room's secret: all monsters dead in `$52` and `$35`, the block pushed in `$42`. Nothing about a shutter is saved. [`Shutter_OpenTriggered`](../src/bank5/shutters-door-flags.asm#L68), [`Secret_RunTrigger_JumpTable`](../src/bank5/underworld-secrets.asm#L105), [`Door_Update`](../src/bank5/update-doors.asm#L60)
- The doorways `$41` E, `$53` W and `$36` W are plain openings on their own side: Link can always walk into `$42`, `$52` and `$35` through them, and is then behind a closed shutter. [`Door_FindTypeByDirBit`](../src/bank5/fill-walls.asm#L55), [`Room_EnterCommon@Method2`](../src/bank5/enter-room.asm#L211)
- **Key doors** take a key only when touched from the room, at X `$78` (N, S) or Y `$8D` (W, E) exactly, with the direction held; once open they are open from both sides for good. [`Doorway_CheckLink`](../src/bank5/doorway.asm#L137), [`Door_TouchKey`](../src/bank5/touch-door.asm#L160)
- **Bombable walls** (`$53` N, `$43` S, `$54` N, `$44` S) look like plain wall. The bomb must lie within `$17` pixels of the door spot on both axes when its blast ends: stand at the door spot against the wall, face it, place the bomb. One bomb opens the wall for both rooms. [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136), [`Bomb_WallHotspotsX`](../src/bank7/bomb.asm#L60)
- **Push blocks** (`$22`, `$42`): the block is the first block square from the left in the floor row Y `$90` (X `$60` in `$22`, X `$70` in `$42`). It moves only when the room has no monster left (Gels count, blade traps do not), Link is in line with it and at most `$10` pixels away, and exactly the one direction toward it is held for 16 frames in a row; it goes one square in that direction, once per room entry. [`PushBlock_State0Idle`](../src/bank4/push-block.asm#L74), [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L234), [`Secret_UpdateRoomAllDead`](../src/bank5/underworld-secrets.asm#L124)
- In `$42` the push (any direction) opens the W shutter; pushing with Gels alive does nothing (trace: 60 frames against the block, no move). In `$22` the push triggers nothing: it only clears the way to stairs that were there all along. [`Secret_Trigger4BlockShutters`](../src/bank5/underworld-secrets.asm#L215), [`BlockPushComplete`](../src/Variables.inc#L389)
- **Walking under a block:** Link's upper 11 pixels are never tested against tiles, so he can stand 8 pixels below a floor row with his head inside a block of that row and walk sideways (Y `$B5` under the row Y `$AD`). The route uses this in `$22` (round the lowest block of the diamond) and in `$45` (the key under a block). [`Tile_GetAheadOfObj`](../src/bank7/tile-collision.asm#L65), [mechanics.md](../mechanics.md) "Tile collision and walking half into blocks"
- **Blade traps, `$22`:** indestructible, 1 heart per touch, no shield helps. A trap senses Link only at rest in its corner; while it runs out or creeps back it senses nothing, so the time to cross a wall row or column is right after it has charged. At Y `$B5` Link is within the bottom traps' range and they do charge along the bottom wall (trace: at the walking pace of the route's steps 12 and 14 they do not reach him). [`Trap_Update_Full`](../src/bank1/trap-rupee-stash.asm#L189), [`Trap_Update_Full@FinishState2`](../src/bank1/trap-rupee-stash.asm#L314), [enemies.md](../enemies.md) "Trap"
- **Wallmasters, `$45`:** one leaves the wall each time Link stands exactly on a wall line (X `$20` or `$D0` with Y `$6D`-`$B4`; Y `$5D` or `$BD` with X `$29`-`$C7`) and the shared timer of `$60` frames has run out; it walks a fixed path to where Link stood. Entering by the W door puts Link on X `$20`, so the first one is on its way at once (trace); the N door spot (Y `$5D`) calls another. [`Wallmaster_Update@CheckLinkAtWall`](../src/bank4/wallmaster.asm#L111), [`Wallmaster_Update@SetUpToEmerge`](../src/bank4/wallmaster.asm#L192)
- A touch that hurts Link while a Wallmaster is out is a capture: he is carried into the wall and restarts in `$73`. Do not stop on a wall line; the key at X `$A0`, Y `$C0` can be taken from Y `$B5` without touching the bottom line (trace). [`Wallmaster_State1Trip@CheckCaptureAndDraw`](../src/bank4/wallmaster.asm#L302), [`Item_TryTake`](../src/bank1/take-item.asm#L206), [enemies.md](../enemies.md) "Wallmaster"
- **Water** (`$23`, `$33`): water squares block Link and walking monsters; Keese, boomerangs and arrows cross them. With the ladder (not found here) Link could cross one square at a time; the route never needs to. [`Link_EndMoveAndAnimate@CheckLadder`](../src/bank7/link-animate-ladder.asm#L209), [`Ladder_CheckAndDraw`](../src/bank5/ladder.asm#L44)
- **The old man, `$41`:** while he is in the room Link cannot go above Y `$8D`. Hitting him with any weapon does not hurt him; from then until Link leaves the room fireballs are shot at Link from beside the two fires. [`Person_BlockLinkMovingUp`](../src/bank1/modeb-cave-save-sprites.asm#L289), [`Person_CheckCollisions`](../src/bank1/underworld-person-update.asm#L89), [`Statue_Update`](../src/bank4/statues.asm#L86)
- **Statues** (`$73`, `$36`): decoration here. Only the layouts `$23` and `$24` have statues that shoot, and no room of this level uses them. [`Statue_RoomLayoutIds`](../src/bank4/statues.asm#L45), [enemies.md](../enemies.md) "Statue fireballs"
- **Stairs and the cellar:** the stairs of `$22` take Link when he stands on them aligned to the grid (X `$80`, Y `$8D`); the cellar is laid out side-on, Link walks only on the walkways and ladders, four blue Keese are always there, bombs open nothing in it. Leaving by either ladder top returns him to `$22` at X `$60`, Y `$AD`. [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`Warp_CheckSubroomExit`](../src/bank5/cave-cellar-exits.asm#L43), [mechanics.md](../mechanics.md) "Stairs, cellars and caves"
- **Carried keys:** in `$33` and `$74` the key is on a Stalfos, not at the position the level data gives. [`Item_MoveAndDrawRoomItem`](../src/bank7/room-item-draw.asm#L65), [enemies.md](../enemies.md) "Stalfos"
- **Boss roar:** room `$45` repeats the roar sample until the boss room's flags say Aquamentus is dead; it is sound only. [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42), [`Sample_BossRoarRequests`](../src/bank5/boss-sound-door-face.asm#L30)
- **Not in this level:** dark rooms, false walls, Bubbles, Like-Likes, statues that shoot, the hungry Goriya, persons that sell or charge. No code tests a room id of this level; the level number only selects the person text group (levels 1, 2, 5, 7). [`Person_InitUnderworld_JumpTable`](../src/bank1/underworld-person-init.asm#L53), [`Person_TextSelectorsLevels1257`](../src/bank1/underworld-person-init.asm#L67), [mechanics.md](../mechanics.md) "Special-cased rooms"

# Second quest

In the second quest the dungeon entered as level 1 is read from a different level block and its own level info:
[data/levelblock-uw-1to6-q2.md](../data/levelblock-uw-1to6-q2.md) and "Level 1, Q2" in [data/levelinfo.md](../data/levelinfo.md). Rooms, doors, items and
monsters differ; this document does not cover it. The quest differences in the code are gathered in
[mechanics.md](../mechanics.md) "Second quest differences".
