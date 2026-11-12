# Level 8: the magic key, the book and the four-headed Gleeok

Level 8 lies under a tree in overworld room `$6D` that fire burns away. It holds two major items, both in
treasure cellars behind a push block: the book (cellar `$6F` under room `$7C`) and the magic key (cellar `$0F`
under room `$1F`), plus a heart container and the eighth Triforce piece. The boss is a Gleeok with four heads
in room `$3C`, which only the sword hurts. Link must bring a candle to open the entrance, bombs for three
walls, and the bow with arrows and rupees for the Gohma that guards the magic key; no key has to be brought.
What sets it apart: almost every room is a fight against Darknuts, Pols Voices, Gibdos or a Manhandla, the
boss can only be reached through a tunnel cellar, and from the magic key on no door in the game costs a key.

> **Written from the ROM only.** Unlike levels 1-4, nothing here was walked in the running game; the route, the
> requirements and the room notes come from the level data and the code. When you work on this level, check each
> step against the game and the code, and update this document with what you find (mark it `(trace)`).

Contents (generated; `L` = line in this file):

<!-- toc -->
- L31 At a glance
- L69 What Link must bring
- L84 What Link gets here
- L136 Map
- L171 Rooms
- L208 Floor plans
- L410 Monsters and boss
- L450 Route
- L502 Optional detours
- L516 Hazards and rules that matter here
- L535 Second quest
<!-- /toc -->

# At a glance

<!-- generated:facts -->
| Field | Value | Read by |
| --- | --- | --- |
| Internal level number | 8 | [`CurLevel`](../src/Variables.inc#L12) |
| Number shown on screen | LEVEL-8 | [`LevelInfo_LevelNumber`](../src/Variables.inc#L725) |
| Overworld entrance room (Q1) | $6D (tile object: $64 tree at $A0,$60); not in Q1, the secret works in Q2 only: $6C (tile object: $64 tree at $50,$60) | [`Warp_HandleOW`](../src/bank5/warps-stairs-caves.asm#L178), [`Flute_SecretRoomIdsOW`](../src/bank7/flute.asm#L36) |
| Start room | $7E | [`LevelInfo_StartRoomId`](../src/Variables.inc#L722) |
| Triforce room | $2C (room item: Triforce piece ($1B)); the field places the compass marker | [`LevelInfo_TriforceRoomId`](../src/Variables.inc#L723), [`StatusBar_UpdateTriforceMarker`](../src/bank7/next-room-room-flags.asm#L51) |
| Boss room | $3C (1 x Gleeok ($45)) | [`LevelInfo_BossRoomId`](../src/Variables.inc#L728), [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42) |
| Cellars | $2F, $0F, $6F | [`LevelInfo_CellarRoomIdArray`](../src/Variables.inc#L726), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55) |
| Rooms | 25: 24 on the submenu map, 1 the map does not show ($2D) | [`LevelInfo_SubmenuMapMask`](../src/Variables.inc#L729) |
| Monster counts (index = level block C bits 7-6) | 3, 5, 6, 8 | [`LevelInfo_FoeCounts`](../src/Variables.inc#L716) |
| Key doors (room and side that has it) | 8 ($1E S, $2E N, $3E S, $4E N, $4D E, $4E W, $5E E, $5F W) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Bombable walls (room and side that has it) | 12 ($1D E, $1E W, $2D E, $2E W, $2E S, $3E N, $3B E, $3C W, $3C S, $4C N, $5E S, $6E N) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Shutters (room and side that has it) | 12 ($0E S, $1E N, $1E E, $3C N, $4D N, $3E E, $3F W, $4B E, $4C W, $5E N, $5E W, $7D W) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| False walls (room and side that has it) | 0 | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Doorways whose two sides differ | $1E E shutter / $1F open, $2C S open / $3C shutter, $3D S open / $4D shutter, $4E S open / $5E shutter, $5D E open / $5E shutter, $7C E open / $7D shutter | [`Door_FindTypeByDirBit`](../src/bank5/fill-walls.asm#L55) |
| Doors on sides with no room of the level | $7E S open | [`Room_CalcNext`](../src/bank7/next-room-room-flags.asm#L72) |
| Push blocks | 2 ($1F, $7C) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Dark rooms | 4 ($4B, $5D, $5F, $7F) | [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124) |
| Keys (room item) | 6 ($4B, $4C, $5C, $5D, $5E, $7F) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Other room items | bomb ($00): $0E, $1D, $3F; magic key ($0B): $0F (cellar); 5 rupees ($0F): $4E, $6E, $7D; book ($11): $6F (cellar); compass ($16): $5F; map ($17): $2E; heart container ($1A): $3C; Triforce piece ($1B): $2C | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |

Room by room data: [`levelblock-uw-7to9-q1.md`](../data/levelblock-uw-7to9-q1.md) (rows with `lvl` = `8` or `c8`).
<!-- /generated:facts -->

- Entrance, Q1: overworld room `$6D`. Its layout is a wood with a one-square path along the row at Y = `$5D`; the square at X = `$A0` of that path is the tree tile object (type `$64`, at `$A0`,`$60`), listed in [data/levelblock-ow.md](../data/levelblock-ow.md) (row `$6D`, `qs` = Q1) and drawn in [data/room-layouts.md](../data/room-layouts.md) (overworld layout `$68`, square `$28`). The path is open at the W edge and joins openings on the N and E edges. [`Tree_Update`](../src/bank4/rock-wall-tree-secrets.asm#L249)
- Opening it: a standing fire must burn out on the tree (its last frames, midpoints less than `$10` pixels apart on both axes). A candle fire comes to rest `$20` pixels ahead of Link, so Link stands on the path at X = `$80` facing right (or X = `$C0` facing left) and uses the candle; the stairs appear about 94 frames later. The blue candle gives one fire per room entry. [items.md](../items.md) "Candle (blue, red)". [`Candle_Wield`](../src/bank1/bomb-candle-wield.asm#L220), [`Fire_Update`](../src/bank7/fire.asm#L44), [`Tree_Update`](../src/bank4/rock-wall-tree-secrets.asm#L249)
- The burned tree is saved (overworld room flag `$80`): the stairs are there on every later visit, also after a death or a reset. The tree in room `$6C` is the Q2 entrance and does nothing in Q1. [`Secret_IsForOtherQuest`](../src/bank4/rock-wall-tree-secrets.asm#L331), [`Warp_HandleOW`](../src/bank5/warps-stairs-caves.asm#L178)
- Entry: Link starts in room `$7E` at X = `$78`, Y = `$DD` (just inside the S door) facing up; the room has no monsters and open doors on all four sides. [`LevelInfo_StartY`](../src/Variables.inc#L717), [`Mode3Unfurl_Init_Sub8_LayOutRoomPlaceLink`](../src/bank5/mode3-mode6-init.asm#L129)
- Walking out: the S door of `$7E` has no room behind it; going through it loads the overworld, and Link appears in room `$6D` at `$60`,`$5D`, on the path four squares W of the stairs. [`Room_CalcNextRoomForDoor`](../src/bank5/next-room-calc.asm#L76), [`Mode_LeaveLevelToOW`](../src/bank5/next-room-calc.asm#L107)
- End of the level: taking the Triforce piece in `$2C` starts mode `$12` (hearts refilled to full), then the overworld is loaded and Link stands at the same spot in `$6D`. [items.md](../items.md) "Triforce pieces and the Triforce of Power". [`Mode12EndLevel_Init`](../src/bank5/mode12-lay-out-room.asm#L56), [`Mode12EndLevel_Update_Sub4LeaveLevel`](../src/bank5/mode12-lay-out-room.asm#L156)
- What stays when Link leaves the level and comes back: unlocked key doors and bombed walls (room flag bits 0-3, set on both sides), taken room items (bit `$10`), and dead Manhandlas, Gohmas and the Gleeok (non-recurring types). [mechanics.md](../mechanics.md) "Room flags and saved state". [`RoomFlags_SetDoorOpened`](../src/bank5/shutters-door-flags.asm#L41), [`RoomFlags_SetItemTaken`](../src/bank1/take-item.asm#L140), [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50)
- What resets: every load of the level zeroes the per-room kill counts and the room history, so rooms of ordinary monsters are full again; shutters are closed on every room entry; a push block is back in its place on every room entry. [mechanics.md](../mechanics.md) "Monster spawning, edge spawns and respawn (room history)". [`Mode2Load_Init`](../src/bank7/init-mode-dispatch.asm#L147), [`Door_Update`](../src/bank5/update-doors.asm#L60), [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L234)
- Inside one visit a cleared room stays empty only while it is among the last six different rooms entered (cellars count); after that an ordinary room refills. [`RoomHistory`](../src/Variables.inc#L533), [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50)

# What Link must bring

Proved from the rooms table and the door, secret and monster code; the route is in `# Route`.

- **Candle** (blue is enough), on the overworld only: without a fire the tree in `$6D` stays a tree, unless it was burned on an earlier visit. Inside the level nothing needs it (see the dark rooms below). [`Tree_Update`](../src/bank4/rock-wall-tree-secrets.asm#L249), [`InvCandle`](../src/Variables.inc#L562)
- **Bombs: 3 at least.** The route crosses three bombable walls: `$6E` N (into `$5E`), `$3E` N (into `$2E`) and `$4C` N (into the boss room `$3C`). None can be avoided: `$6E` N is the only way out of the five southern rooms, the magic key's corner is reached only through `$3E` N or other bombable walls, and the boss room's other entry (`$3B` E) is a bombable wall too. A full bag (8, 12 or 16) is better: a blast is also the fastest weapon against Darknuts and Manhandla. [items.md](../items.md) "Bombs". [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136), [`InvBombs`](../src/Variables.inc#L558), [`InvMaxBombs`](../src/Variables.inc#L607)
- **Bow, any arrow, and rupees** (1 per shot; 3 hits with wooden arrows, 2 with silver): the Gohma in `$1E` takes damage from nothing else, and the shutter to `$1F` (the magic key) opens only when it is dead. Bring 20 rupees or more to allow for misses and for shooting Pols Voices. Without the bow the rest of the level, boss and Triforce included, can still be done, but not the magic key. [items.md](../items.md) "Arrows and Bow (wooden, silver)". [`Arrow_Wield`](../src/bank5/wield-items.asm#L187), [`Gohma_HandleWeaponCollision`](../src/bank4/gohma.asm#L367), [`InvBow`](../src/Variables.inc#L560), [`InvRupees`](../src/Variables.inc#L583)
- **Sword: any grade is accepted by the code**, but the Gleeok takes `$1C0` damage from the sword class only: 28 wooden, 14 white or 9 magic sword hits; a blue Darknut takes 8, 4 or 2 hits from the side or behind. The white sword is the practical minimum. [`Sword_DamagePoints`](../src/bank1/sword-bomb-arrow-collision.asm#L182), [`Gleeok_CheckSegmentCollisions`](../src/bank4/gleeok-head-body.asm#L128), [`Darknut_Init`](../src/bank4/darknut-gibdo.asm#L44)
- **Ladder: only for the key in `$7F`** (one water square at X = `$40` on the row Link walks, Y = `$8D`), which the main route takes as its second key. The alternatives under `# Route` avoid it. [items.md](../items.md) "Ladder". [`Link_EndMoveAndAnimate`](../src/bank7/link-animate-ladder.asm#L150), [`InvLadder`](../src/Variables.inc#L570)
- **Keys: none.** The route opens two key doors before it has the magic key (`$4E` N and `$2E` N) and picks up two keys first (`$7F`, `$5E`). Keys are one counter for all levels, so each key carried in saves one pickup: with one key in hand the `$7F` detour (two room entries, the ladder) is not needed. [`InvKeys`](../src/Variables.inc#L584), [`Door_TouchKey`](../src/bank5/touch-door.asm#L160)
- **Hearts and rings: no test in the code**, but the touch of a blue Darknut, Gibdo, Pols Voice or Gohma costs 2 hearts without a ring, 1 with the blue ring, half with the red ring; a red Darknut, Manhandla, the Gleeok's neck base or flying head and every fireball of type `$56` cost 1 heart. [mechanics.md](../mechanics.md) "Hearts, rings and death". [`Damage_ObjTypeDamageToLink`](../src/bank1/monster-collisions.asm#L200), [`Link_BeHarmed`](../src/bank1/link-collision-harm.asm#L193)
- **Not needed:** raft, recorder, food, power bracelet, letter or potion. No room of this level holds a Digdogger, a hungry Goriya, a false wall, a Wallmaster, a Like-Like or a trap (see the rooms table). [`Level_BlockUW7to9Q1`](../src/bank6/level-blocks-common-data.asm#L74)
- **Magic rod** (from level 6): not needed here; it is what makes the book useful. [`InvRod`](../src/Variables.inc#L566), [`InvBook`](../src/Variables.inc#L568)
- **Dark rooms** (`$4B`, `$5D`, `$5F`, `$7F`): the dark flag is read only by the palette fades of a room change and by the fire that lights the room. Monsters, collisions, items and doors work as in a lit room; only the picture is black. A standing fire (candle, or the book's fire) fades the room to light in 4 steps of 10 frames, during which nothing else updates. [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124), [`Room_IsDark_Bank4`](../src/bank4/shoot-helpers-candle.asm#L184), [`Mode4Enter_Init`](../src/bank5/enter-room.asm#L65), [`Candle_State0Begin`](../src/bank4/shoot-helpers-candle.asm#L147)

# What Link gets here

## Book (item `$11`), cellar `$6F` under room `$7C`

- Where: `$7C` is the dead end W of the start room: `$7E` W, then `$7D` W. The W door of `$7D` is a shutter that opens when its Manhandla is dead. [`Secret_Trigger1AllDead`](../src/bank5/underworld-secrets.asm#L152), [`Shutter_OpenTriggered`](../src/bank5/shutters-door-flags.asm#L68)
- In `$7C` the stairs (square X = `$80`, Y = `$90`) lie inside a diamond of eight blocks. The push block is the leftmost block of the middle row, at X = `$60`, Y = `$90`. It does not move until the room's 1 blue Darknut, 2 red Darknuts and 3 Gibdos are dead; the 2 Bubbles cannot die and do not count. [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L234), [`PushBlock_State0Idle`](../src/bank4/push-block.asm#L74), [`Secret_UpdateRoomAllDead`](../src/bank5/underworld-secrets.asm#L124)
- Push it up or down, never right: stand directly below it (X = `$60`, Y = `$9D`) and hold Up, or directly above it (X = `$60`, Y = `$7D`) and hold Down, for `$10` frames. The block slides one square in the direction pushed and becomes a block tile there. Pushed from the W side it lands on the square between the gap and the stairs and closes the diamond again until the room is re-entered. [`PushBlock_State0Idle`](../src/bank4/push-block.asm#L74), [`PushBlock_State1Moving`](../src/bank4/push-block.asm#L165)
- Then walk through the gap (X = `$60`, Y = `$8D`) two squares E onto the stairs (X = `$80`, Y = `$8D`). The room's secret trigger is 0: the push opens nothing else. [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61)
- Cellar `$6F` (layout in `# Floor plans`, treasure cellar): Link arrives on the left ladder at X = `$30`. Go down it to the bottom passage (Y = `$BD`), E to the second ladder at X = `$B0`, up it to the ledge (Y = `$8D`) and W to the book at X = `$80`. Four blue Keese fly here. Leave by the left ladder: Up at its top (Y < `$40`). Link comes out in `$7C` at `$60`,`$9D`, outside the diamond. [`Mode9Cellar_Init_Sub8_EnterCellar`](../src/bank5/cave-cellar-enter.asm#L276), [`Warp_CheckSubroomExit`](../src/bank5/cave-cellar-exits.asm#L43), [`Item_TryTakeRoomItem`](../src/bank1/take-item.asm#L173)
- **What it changes:** with `InvBook` set, every magic shot of the rod leaves a standing fire where it ends (on the first monster it overlaps or at the room bound), for 79 frames. The fire does `$10` damage to monsters that fire can hurt, lights a dark room, burns Link for half a heart if he walks into it, and burns overworld trees like a candle fire. It needs no candle and does not use up the blue candle's one fire per room. [items.md](../items.md) "Magic Rod and Book". [`SwordShot_HandleBlocked`](../src/bank7/sword-shot-magic-shot.asm#L172), [`InvBook`](../src/Variables.inc#L568)
- The book is never selected or used by itself, and without the rod it does nothing. Nothing in this level or later requires it: no door, secret or monster tests `InvBook`. In this level its fire can light `$7F`, `$5D`, `$5F` and `$4B`; it does not hurt Darknuts, Pols Voices, Manhandla, Gohma or the Gleeok (fire is masked for all of them). [`SwordShot_HandleBlocked`](../src/bank7/sword-shot-magic-shot.asm#L172), [`Damage_HandleWeaponHit`](../src/bank1/weapon-damage-death.asm#L127)

## Magic key (item `$0B`), cellar `$0F` under room `$1F`

- Where: `$1F` is the dead end in the NE corner, reached only from `$1E` through its E shutter, which opens when the Gohma of `$1E` is dead (arrows). [`Secret_Trigger1AllDead`](../src/bank5/underworld-secrets.asm#L152), [`Gohma_HandleWeaponCollision`](../src/bank4/gohma.asm#L367)
- `$1F` has the same floor as `$7C` (layout `$1A`): kill its 2 Pols Voices, 2 blue Darknuts and 2 red Darknuts, push the block at X = `$60`, Y = `$90` up or down, walk onto the stairs at X = `$80`, Y = `$8D`. [`PushBlock_State0Idle`](../src/bank4/push-block.asm#L74), [`Secret_UpdateRoomAllDead`](../src/bank5/underworld-secrets.asm#L124)
- Cellar `$0F` is the same treasure cellar as `$6F`: left ladder down, bottom passage E, second ladder up, W along the ledge to the key at X = `$80`. Link comes out in `$1F` at `$60`,`$9D`. The W door of `$1F` is an open doorway on this side, so the way back needs nothing. [`Mode9Cellar_Init_Sub8_EnterCellar`](../src/bank5/cave-cellar-enter.asm#L276), [`Warp_CheckSubroomExit`](../src/bank5/cave-cellar-exits.asm#L43)
- **What it changes:** a key door (door types 5 and 6) tests, in this order: already open; a door command running; `InvMagicKey` nonzero: the door opens and nothing is spent; only then the key count. So from the pickup on every locked door in the game opens on touch, `InvKeys` is never read or lowered again, and keys found later are worth nothing. [items.md](../items.md) "Keys and Magic Key". [`Door_TouchKey`](../src/bank5/touch-door.asm#L160), [`InvMagicKey`](../src/Variables.inc#L571), [`InvKeys`](../src/Variables.inc#L584)
- The status bar shows "XA" in place of the key count from then on; the count itself is kept but not drawn. [`StatusBar_FormatText`](../src/bank1/status-bar-rupees.asm#L148)
- A door opened with the magic key is still an ordinary opened door: it takes the 8-frame open command, Link's timer `$20`, and its flag is saved for both rooms. Shutters, bombable walls and walls are not key doors and are not affected. [`Door_TouchKey`](../src/bank5/touch-door.asm#L160), [`Door_Update`](../src/bank5/update-doors.asm#L60)
- First use in this level: none on the route (its two key doors are already open). Off the route it opens `$4E` W (to the second Gohma and the old man of `$3D`) and `$5E` E (to the compass in `$5F`). [`Door_TouchKey`](../src/bank5/touch-door.asm#L160)
- Later: level 9 has 16 key doorways and only 4 keys lying in it; with the magic key none of those keys is needed. See the facts table of [level-9.md](level-9.md). [`Door_TouchKey`](../src/bank5/touch-door.asm#L160)

## Heart container (item `$1A`), boss room `$3C`

- It is the room item of `$3C`, hidden until no monster is left (secret trigger 7), then shown at X = `$20`, Y = `$C0`: the bottom left floor square. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61), [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)
- **What it changes:** `HeartValues` + `$11`: one more container and one more full heart; the other hearts are not refilled (the Triforce piece does that one room later). [items.md](../items.md) "Hearts, Heart Containers and Fairies". [`Item_Take_CheckClass1`](../src/bank1/take-item.asm#L337), [`HeartValues`](../src/Variables.inc#L593)
- The container count is what the sword caves test (5 containers for the white sword, 12 for the magic sword); current health matters only for the sword beam, which needs full hearts. [`Cave_State2TalkOrShopOrDoorCharge`](../src/bank1/cave-shop.asm#L55), [`SwordShot_Make`](../src/bank7/sword-and-rod.asm#L327)

## Triforce piece (item `$1B`), room `$2C`

- `$2C` is N of the boss room; the boss room's N shutter opens when the Gleeok is dead. The room has no monsters and its statues do not shoot. The piece lies at X = `$78`, Y = `$90`; walk straight up from the S door (X = `$78`) through the gap in the block ring. [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51), [`Statue_RoomLayoutIds`](../src/bank4/statues.asm#L45)
- **What it changes:** bit `$80` of `InvTriforce` is set, the level ends (mode `$12`: full heart refill, then the overworld at the entrance in `$6D`). [items.md](../items.md) "Triforce pieces and the Triforce of Power". [`Item_TakeClass0PerLevel`](../src/bank1/take-item.asm#L421), [`Mode12EndLevel_StartFillingHearts`](../src/bank5/mode12-lay-out-room.asm#L118), [`InvTriforce`](../src/Variables.inc#L597)
- Later: the first old man of level 9 lets Link pass only with all eight pieces (`InvTriforce` = `$FF`); the recorder's whirlwind can now set Link down beside this level (it scrolls in from room `$6C` to `$6D`). [`Person_InitUnderworldLevel9`](../src/bank1/underworld-person-init.asm#L172), [`Whirlwind_PrevRoomIds`](../src/bank1/whirlwind.asm#L81)

## Keys (item `$19`): six rooms

- `$7F` (lies in the open; dark, water), `$5E` (appears when its 5 blue Darknuts are dead), `$5D` (lies in the open; dark), `$5C` (appears when its 3 red Darknuts are dead), `$4C` (lies in the open at `$D0`,`$60`, the top right corner), `$4B` (lies in the open; dark). All but the `$4C` key are at the room centre `$80`,`$90`. [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51), [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61)
- **What it changes:** `InvKeys` + 1; a key is spent on the first touch of a closed key door while Link has no magic key. The level has four key doorways (`$3E`/`$4E`, `$1E`/`$2E`, `$4D`/`$4E`, `$5E`/`$5F`); the route needs the first two. [`Item_Take_CheckClass1`](../src/bank1/take-item.asm#L337), [`Door_TouchKey`](../src/bank5/touch-door.asm#L160)
- Once the magic key is held the remaining keys are useless: do not walk for the `$4C`, `$4B`, `$5C` or `$5D` keys after it. [`Door_TouchKey`](../src/bank5/touch-door.asm#L160)

## Map (item `$17`) in `$2E`, compass (item `$16`) in `$5F`

- The map appears at the centre of `$2E` when its Manhandla is dead; the route passes through `$2E` twice, so it costs only that fight. The compass lies in the open at the centre of `$5F` (dark, 8 Pols Voices), behind the key door `$5E` E. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61), [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)
- **What it changes:** the map draws the level's rooms on the status bar; the compass puts a marker on room `$2C` there. Neither changes doors, monsters or room contents. Room `$2D` (the rupee stash) is on neither map. [items.md](../items.md) "Map and Compass". [`Inv_HasMap`](../src/bank5/submenu-map.asm#L153), [`Inv_HasCompass`](../src/bank5/submenu-map.asm#L143), [`StatusBar_UpdateTriforceMarker`](../src/bank7/next-room-room-flags.asm#L51)

## Bombs and rupees

- Bomb items (4 bombs each, up to the capacity) appear when all monsters are dead in `$0E` (5 red Darknuts), `$1D` (5 Pols Voices) and `$3F` (1 blue and 2 red Darknuts, 3 Gibdos; on the route). [`Item_Take_CheckClass1`](../src/bank1/take-item.asm#L337), [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61)
- 5-rupee items appear when all monsters are dead in `$7D` (on the route, the kill is required), `$6E` and `$4E` (both on the route, the kill is optional). Room `$2D` is a rupee stash: ten single rupees on the floor. [`Item_Take`](../src/bank1/take-item.asm#L259), [`RupeeStash_Update_Full`](../src/bank1/trap-rupee-stash.asm#L338)
- Dropped bombs come only from drop group 2: here red Darknuts and Gibdos. [enemies.md](../enemies.md) "Drop tables". [`Drop_SetUpItem`](../src/bank4/drop-items.asm#L157)

# Map

<!-- generated:map -->
```
                           [0Ei ]
                             S
                  [1Di ] X [1E  ]S/-[1F  ]
                    |        K
         [2CT ]   [2Dh ] X [2Ei ]
          |/S                X
[3B  ] X [3CBi]   [3D  ]   [3E  ] S [3Fi ]
  |        X       |/S       K
[4Bk ] S [4Ck ]   [4D  ] K [4Ei ]
                    |       |/S
         [5Ck ]---[5Dk ]-/S[5Ek ] K [5Fi ]
                             X
                           [6Ei ]
                             |
         [7C  ]-/S[7Di ]---[7ES ]---[7Fk ]

[73S] = room $73 and its marks: S start room, B boss room, T Triforce room,
        k a key lies here, i another room item lies here, h not shown on the submenu map.
Between two rooms, the door: --- or | open, K key door, S shutter, X bombable wall,
        F false wall (walk-through), nothing = wall. A/B = the two sides differ: the left
        or upper room's side first, . = wall on that side.
Doors on sides with no room of the level: $7E S open.
```

- Cellar `$2F` (tunnel cellar): joins `$4C` (left exit) and `$3F` (right exit), whose stairs lead here; item: none. [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`levelblock-uw-7to9-q1.md`](../data/levelblock-uw-7to9-q1.md)

- Cellar `$0F` (treasure cellar): both exits lead to `$1F`, whose stairs lead here; item: magic key ($0B). [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`levelblock-uw-7to9-q1.md`](../data/levelblock-uw-7to9-q1.md)

- Cellar `$6F` (treasure cellar): both exits lead to `$7C`, whose stairs lead here; item: book ($11). [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`levelblock-uw-7to9-q1.md`](../data/levelblock-uw-7to9-q1.md)
<!-- /generated:map -->

# Rooms

Generated from the level data by `tools/romdata.py`; do not edit between the markers.

<!-- generated:rooms -->
| room | N | S | W | E | monsters | item | secret | block | dark | layout | floor | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $0E | wall | shutter | wall | wall | 5 x Darknut red ($0B) | bomb ($00) at $80,$90 | all dead: shutters open, room item appears | - | - | $00 | plain | - |
| $1D | wall | open | wall | bomb | 5 x Pols Voice ($16) | bomb ($00) at $80,$90 | all dead: shutters open, room item appears | - | - | $11 | blocks | - |
| $1E | shutter | key | bomb | shutter | 1 x Gohma ($33) | - | all dead: shutters open | - | - | $24 | statues | - |
| $1F | wall | wall | open | wall | 2 x Pols Voice ($16), 2 x Darknut blue ($0C), 2 x Darknut red ($0B) | - | all dead: shutters open | B | - | $1A | blocks, stairs | stairs lead to cellar $0F |
| $2C Triforce | wall | open | wall | wall | none | Triforce piece ($1B) at $80,$90 | - | - | - | $29 | blocks, statues | - |
| $2D off-map | open | wall | wall | bomb | 1 x Rupee stash ($35) | - | - | - | - | $26 | blank | - |
| $2E | key | bomb | bomb | wall | 1 x Manhandla ($3C) | map ($17) at $80,$90 | all dead: shutters open, room item appears | - | - | $00 | plain | - |
| $3B | wall | open | wall | bomb | 1 x dungeon person ($50) | - | - | - | - | $26 | blank | person $50 says: 10TH ENEMY HAS THE BOMB.; boss sound $10 |
| $3C boss | shutter | bomb | bomb | wall | 1 x Gleeok ($45) | heart container ($1A) at $20,$C0 | all dead: shutters open, room item appears | - | - | $05 | blocks | - |
| $3D | wall | open | wall | wall | 1 x dungeon person ($4F) | - | - | - | - | $26 | blank | person $4F says: SPECTACLE ROCK IS / AN ENTRANCE TO DEATH.; boss sound $10 |
| $3E | bomb | key | wall | shutter | 6 x Darknut blue ($0C) | - | all dead: shutters open | - | - | $23 | statues | - |
| $3F | wall | wall | shutter | wall | 2 x Bubble ($2B), 1 x Darknut blue ($0C), 2 x Darknut red ($0B), 3 x Gibdo ($30) | bomb ($00) at $80,$90 | all dead: shutters open, room item appears | - | - | $1B | blocks, stairs | stairs lead to cellar $2F |
| $4B | open | wall | wall | shutter | 6 x Darknut red ($0B) | key ($19) at $80,$90 | all dead: shutters open | - | D | $23 | statues | - |
| $4C | bomb | wall | shutter | wall | 8 x Pols Voice ($16) | key ($19) at $D0,$60 | all dead: shutters open | - | - | $1C | blocks, stairs | stairs lead to cellar $2F; boss sound $10 |
| $4D | shutter | open | wall | key | 1 x Gohma ($33) | - | all dead: shutters open | - | - | $25 | sand | - |
| $4E | key | open | key | wall | 2 x Bubble ($2B), 1 x Darknut blue ($0C), 2 x Darknut red ($0B), 3 x Gibdo ($30) | 5 rupees ($0F) at $80,$90 | all dead: shutters open, room item appears | - | - | $00 | plain | - |
| $5C | wall | wall | wall | open | 3 x Darknut red ($0B) | key ($19) at $80,$90 | all dead: shutters open, room item appears | - | - | $24 | statues | - |
| $5D | open | wall | open | open | 2 x Pols Voice ($16), 2 x Gibdo ($30), 2 x Keese blue ($1B) | key ($19) at $80,$90 | - | - | D | $11 | blocks | - |
| $5E | shutter | bomb | shutter | key | 5 x Darknut blue ($0C) | key ($19) at $80,$90 | all dead: shutters open, room item appears | - | - | $24 | statues | - |
| $5F | wall | wall | key | wall | 8 x Pols Voice ($16) | compass ($16) at $80,$90 | - | - | D | $18 | water | - |
| $6E | bomb | open | wall | wall | 1 x Manhandla ($3C) | 5 rupees ($0F) at $80,$90 | all dead: shutters open, room item appears | - | - | $25 | sand | - |
| $7C | wall | wall | wall | open | 2 x Bubble ($2B), 1 x Darknut blue ($0C), 2 x Darknut red ($0B), 3 x Gibdo ($30) | - | - | B | - | $1A | blocks, stairs | stairs lead to cellar $6F |
| $7D | wall | wall | shutter | open | 1 x Manhandla ($3C) | 5 rupees ($0F) at $80,$90 | all dead: shutters open, room item appears | - | - | $23 | statues | - |
| $7E start | open | open | open | open | none | - | - | - | - | $21 | statues, sand | - |
| $7F | wall | wall | open | wall | 3 x Bubble ($2B), 2 x Zol ($13), 3 x Keese blue ($1B) | key ($19) at $80,$90 | - | - | D | $16 | water | - |
| $2F cellar | - | - | - | - | 4 x Keese blue ($1B) | - | - | - | - | $3E | tunnel cellar | left exit to $4C, right exit to $3F; Link comes out at $70,$7D |
| $0F cellar | - | - | - | - | 4 x Keese blue ($1B) | magic key ($0B) at $80,$90 | - | - | - | $3F | treasure cellar | left exit to $1F, right exit to $1F; Link comes out at $60,$9D |
| $6F cellar | - | - | - | - | 4 x Keese blue ($1B) | book ($11) at $80,$90 | - | - | - | $3F | treasure cellar | left exit to $7C, right exit to $7C; Link comes out at $60,$9D |
<!-- /generated:rooms -->

# Floor plans

<!-- generated:floors -->
One character per floor square, 12 columns x 7 rows inside the walls (left column = west, top row =
north): `.` floor (tile `$74`), `B` block (tile `$B0`; a push block object is placed on one in a room
with a push block), `W` water (tile `$F4`), `S` statue (tiles `$94` and `$B4`), `>` stairs (tile `$70`),
`:` sand (tile `$68`), `_` blank square (tile `$24`). [`RoomLayout_UWSquareTiles`](../src/bank5/uw-floor-push-block.asm#L63), [`RoomLayout_LayOutUWFloor`](../src/bank5/uw-floor-push-block.asm#L73), [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L234), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)
Stairs that a secret makes appear are not part of the layout.

```
layout $00  rooms: $0E,$2E,$4E
............
............
............
............
............
............
............
```

```
layout $05  rooms: $3C
BBBBB..BBBBB
BBBB....BBBB
BB........BB
............
............
..B......B..
............
```

```
layout $11  rooms: $1D,$5D
..B........B
.B...B....B.
....B....B..
...B....B...
..B....B....
.B....B...B.
B........B..
```

```
layout $16  rooms: $7F
WWWWW..WWWWW
W...WWWW...W
W.WWW..WWW.W
..W......W..
W.WWW..WWW.W
W...WWWW...W
WWWWW..WWWWW
```

```
layout $18  rooms: $5F
............
............
WWWWWWWWWWWW
............
............
............
............
```

```
layout $1A  rooms: $1F,$7C
............
......B.....
.....B.B....
....B.>.B...
.....B.B....
......B.....
............
```

```
layout $1B  rooms: $3F
........BBBB
........BBBB
.........BBB
...........>
.........BBB
........BBBB
........BBBB
```

```
layout $1C  rooms: $4C
..........B.
.BBBBBBBB.B.
.B.....>B.B.
.B...BBBB.B.
.B........B.
.BBBBBBBBBB.
............
```

```
layout $21  rooms: $7E
............
.S..S..S..S.
............
.S..S..S..S.
.....::.....
.S.:S::S:.S.
...::::::...
```

```
layout $23  rooms: $3E,$4B,$7D
............
............
............
....S..S....
............
............
............
```

```
layout $24  rooms: $1E,$5C,$5E
S..........S
............
............
............
............
............
S..........S
```

```
layout $25  rooms: $4D,$6E
::::::::::::
::::::::::::
::::::::::::
::::::::::::
::::::::::::
::::::::::::
::::::::::::
```

```
layout $26  rooms: $2D,$3B,$3D
____________
____________
____________
____________
____________
____________
____________
```

```
layout $29  rooms: $2C
............
.BBBBBBBBBB.
.B..S..S..B.
.B.S....S.B.
.B........B.
.BBBB..BBBB.
............
```

```
layout $3E (tunnel cellar)  rooms: $2F
     c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 cA cB cC cD cE cF
r0   02 02 02 00 02 02 02 02 02 02 02 02 00 02 02 02
r1   03 03 03 01 03 03 03 03 03 03 03 03 01 03 03 03
r2   03 03 03 01 03 03 03 03 03 03 03 03 01 03 03 03
r3   03 03 03 01 03 03 03 03 03 03 03 03 01 03 03 03
r4   03 03 03 01 03 03 03 03 03 03 03 03 01 03 03 03
r5   03 03 03 01 03 03 03 03 03 03 03 03 01 03 03 03
r6   03 03 03 01 03 03 03 03 03 03 03 03 01 03 03 03
r7   03 03 02 01 02 02 02 02 02 02 02 02 01 02 03 03
r8   03 03 0C 01 0C 0C 0C 0C 0C 0C 0C 0C 01 0C 03 03
r9   03 03 03 03 03 03 03 03 03 03 03 03 03 03 03 03
r10  03 03 03 03 03 03 03 03 03 03 03 03 03 03 03 03
squares (first tile): 00 = $24, 01 = $6F, 02 = $F3, 03 = $FA, 0C = $F3
```

```
layout $3F (treasure cellar)  rooms: $0F,$6F
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

HP and damage are raw values (`$10` = one wooden sword hit; sword `$10`/`$20`/`$40` by grade, bomb `$40`, arrow `$20`/`$40`);
damage to Link is without a ring. "Route" rooms are in bold.

| Monster (type) | Rooms and count | HP | What hurts it | Touch | Section of [enemies.md](../enemies.md) |
| --- | --- | --- | --- | --- | --- |
| Manhandla (`$3C`) | **`$7D`** 1, **`$6E`** 1, **`$2E`** 1 | `$40` per hand, 4 hands | sword, beam, rod, magic shot, arrow, bomb; not boomerang, not fire | 1 heart; fireballs 1 heart, no shield stops them | Manhandla |
| Darknut blue (`$0C`) | **`$5E`** 5, **`$3E`** 6, **`$7C`** 1, **`$1F`** 2, **`$3F`** 1, **`$4E`** 1 | `$80` | sword class and bomb, never from the front; no stun | 2 hearts | Darknut (red, blue) |
| Darknut red (`$0B`) | **`$7C`** 2, **`$1F`** 2, **`$3F`** 2, **`$4E`** 2, `$0E` 5, `$4B` 6, `$5C` 3 | `$40` | as blue | 1 heart | Darknut (red, blue) |
| Gibdo (`$30`) | **`$7C`** 3, **`$3F`** 3, **`$4E`** 3, `$5D` 2 | `$70` | every weapon; boomerang stuns | 2 hearts | Gibdo |
| Pols Voice (`$16`) | **`$1F`** 2, **`$4C`** 8, `$1D` 5, `$5D` 2, `$5F` 8 | `$A0` | sword class; any arrow kills outright and flies on | 2 hearts | Pols Voice |
| Bubble, flashing (`$2B`) | **`$7F`** 3, **`$7C`** 2, **`$3F`** 2, **`$4E`** 2 | cannot die | nothing | no damage; sword disabled 151-160 frames | Bubble (flashing, red, blue) |
| Zol (`$13`) | **`$7F`** 2 | `$20` | every weapon; a `$10` hit splits it into two Gels | 1 heart | Zol |
| Keese blue (`$1B`) | **`$7F`** 3, `$5D` 2, every cellar 4 | 0 | every weapon, boomerang included | half a heart | Keese (blue, red, black) |
| Gohma (`$33`) | **`$1E`** 1, `$4D` 1 | `$60` | an arrow flying up into the fully open eye, nothing else | 2 hearts; fireballs 1 heart | Gohma (blue, red) |
| Gleeok, 4 heads (`$45`) | **`$3C`** 1 (boss) | `$A0`, then `$60` per further head | sword, beam, rod swing only | neck base and flying head 1 heart, attached head 3/8 heart, fireball 1 heart | Gleeok |
| Rupee stash (`$35`) | `$2D` | - | - | - | Rupee stash |
| Old men (`$4F`, `$50`) | `$3D`, `$3B` | - | cannot die; a weapon hit turns on fireballs | no damage | Old men and shopkeepers |

**Getting through the hard rooms**

- Darknuts (38 of the level's monsters, 16 of them blue): a hit counts only when the weapon's direction is not exactly opposite to the Darknut's facing. Strike from the side or from behind, or place a bomb while Link faces any way but head-on to it; a blast does `$40` (a red Darknut dies, a blue one needs two). The boomerang does not stun them and arrows, fire and the magic shot do nothing. [`Damage_HandleWeaponHit`](../src/bank1/weapon-damage-death.asm#L127), [`Darknut_Init`](../src/bank4/darknut-gibdo.asm#L44), [`Darknut_Update`](../src/bank4/darknut-gibdo.asm#L89)
- Blue Darknuts walk 0.625 pixels a frame, red ones 0.5; at each grid point they turn toward Link about half the time when he is within 8 pixels of their row or column. They never shoot, so distance is safe. [`Darknut_Update`](../src/bank4/darknut-gibdo.asm#L89)
- Bubbles (`$7C`, `$3F`, `$4E`, `$7F`): a touch does no damage but knocks Link back and disables the A button for 151-160 frames; the B item still works, so bombs and arrows remain. In `$7C` all six other monsters must die with two Bubbles in the room. [`Bubble_Update`](../src/bank4/bubble-keese.asm#L93), [`SwordBlockedLongTimer`](../src/Variables.inc#L62), [`Link_HandleInput`](../src/bank5/link-handle-input.asm#L68)
- Pols Voice: sword hits needed are 10 / 5 / 3 by grade, or one arrow each (1 rupee); an arrow goes on through a Pols Voice and can kill a line of them. An arrow does not count during the about 32 frames after a sword hit. Bombs do nothing. [`PolsVoice_DrawAndCheckCollisions`](../src/bank4/pols-voice.asm#L197), [`Collision_CheckShotVsMonster`](../src/bank1/sword-bomb-arrow-collision.asm#L287)
- Gibdo: `$70` HP, 2 bombs or 4 white / 2 magic sword hits; the boomerang stuns it for about 160 frames, and a stunned Gibdo's touch does not hurt. [`Gibdo_Update`](../src/bank4/darknut-gibdo.asm#L69)
- Manhandla (`$7D` must be killed; `$6E` and `$2E` only crossed): the fight ends when the last of the four hands dies; the middle cannot die. Each dead hand speeds it up by 0.5 pixel a frame. A bomb whose blast catches all four hands at once (bomb within 8 pixels of the middle's position on both axes) ends it in one frame. It moves in 8 directions, ignores blocks and bounces off the room bounds. [`Manhandla_CheckCollisions`](../src/bank4/manhandla.asm#L206), [`Manhandla_Init`](../src/bank4/gleeok-manhandla-gohma-inits.asm#L155)
- Gohma in `$1E` (entered from the S door, Link below it): it stands at X = `$80`, Y = `$70` and paces `$20` pixels right, left or down and back. Only an arrow flying up that meets the eye while it is fully open (96 frames of each opening; openings start 384-510 frames apart, the first on entry) does damage: `$20` wooden, `$40` silver, against `$60` HP. Link's X must be from Gohma's X - `$0A` to Gohma's X + 5 when he shoots; an arrow that touches any other part is lost. It shoots a fireball every 65 frames, and the four corner statues of the room shoot too. [`Gohma_HandleWeaponCollision`](../src/bank4/gohma.asm#L367), [`Gohma_Init`](../src/bank4/gleeok-manhandla-gohma-inits.asm#L221), [`Statue_Update`](../src/bank4/statues.asm#L86)
- Statues shoot in the rooms with layout `$24` (`$1E`, `$5C`, `$5E`: four sources in the corners) and `$23` (`$3E`, `$4B`, `$7D`: two sources at the centre statues): fireballs of type `$55`, half a heart, which the magic shield stops when Link stands still facing them. The statues of `$7E` and `$2C` do not shoot. [`Statue_RoomLayoutIds`](../src/bank4/statues.asm#L45), [`Statue_Update`](../src/bank4/statues.asm#L86)

**The boss: Gleeok with four heads, room `$3C`**

- Entered from the S side (the bombed wall from `$4C`). The body is drawn at the top centre and is not an object; four necks hang from X = `$7C`, Y = `$6F`, each head swinging at most about `$0B` pixels per neck segment from the one above. [`Gleeok_Init`](../src/bank4/gleeok-manhandla-gohma-inits.asm#L59), [`Gleeok_Update`](../src/bank4/gleeok.asm#L175)
- Only damage type 1 hurts: sword, sword beam, rod swing. Bombs, arrows, fire, the boomerang and the magic shot do nothing. [`Gleeok_Init`](../src/bank4/gleeok-manhandla-gohma-inits.asm#L59), [`Damage_HandleWeaponHit`](../src/bank1/weapon-damage-death.asm#L127)
- All attached heads share one HP byte: the first head falls after `$A0` damage, each later one after `$60`, whichever head is struck. In sword hits: 10 + 6 + 6 + 6 wooden, 5 + 3 + 3 + 3 white, 3 + 2 + 2 + 2 magic. After each hit no head can be hit (or hurts) for about 32 frames. [`Gleeok_CheckSegmentCollisions`](../src/bank4/gleeok-head-body.asm#L128), [`Sword_DamagePoints`](../src/bank1/sword-bomb-arrow-collision.asm#L182)
- A cut-off head becomes a flying head (type `$46`) that chases Link, shoots and cannot be destroyed; up to three of them fly during the fight. They vanish when the last neck loses its head. One Gleeok fireball exists at a time (1 heart, no shield stops it). [`Gleeok_UpdateFlyingHead`](../src/bank4/gleeok.asm#L78), [`Fireball_Update`](../src/bank4/monster-shots-fireballs.asm#L286)
- When it dies: the heart container appears at `$20`,`$C0` and the N shutter opens (secret trigger 7); the kill may also drop an item of drop group 3. The Gleeok is non-recurring: it stays dead. Its death also ends the boss roar heard in `$3B`, `$3D` and `$4C`. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61), [`RoomHistory_SaveKillCountUW`](../src/bank5/room-history-uw.asm#L110), [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42)
- The room's S and W sides are bombable walls, not shutters: Link can leave the fight the way he came. [`Door_TouchBombable`](../src/bank5/touch-door.asm#L108)

# Route

Established from the level data and the code; not walked in the running game. The search (room graph with
door types, keys, secrets and cellar links; cost = room entries, then fights) gives 25 room entries after the
start room.

Positions: W and E doors are on the row Y = `$8D`, N and S doors on the column X = `$78`. "Kill all" never
includes Bubbles.

| Step | Room | Entered from | What to do | Leave by | Keys after | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `$7E` | level entrance (S) | nothing here | E (open) | 0 | no monsters |
| 1 | `$7F` | W | walk E along the row: floor, one water square (ladder), floor; take the key at `$80`,`$90`; walk back the same way | W (open) | 1 | dark; 3 Bubbles, 2 Zols, 3 Keese: nothing has to die; needs the ladder |
| 2 | `$7E` | E | cross the room | W (open) | 1 | |
| 3 | `$7D` | E | kill the Manhandla: the W shutter opens (5 rupees appear at the centre) | W (shutter) | 1 | two centre statues shoot |
| 4 | `$7C` | E | kill 1 blue and 2 red Darknuts and 3 Gibdos; push the block at `$60`,`$90` up (from `$60`,`$9D`) or down (from `$60`,`$7D`); walk onto the stairs at `$80`,`$8D` | stairs | 1 | 2 Bubbles; never push the block from the W |
| 5 | `$6F` | stairs (left ladder) | ladder down, bottom passage E, second ladder up, ledge W: take the book | left ladder, Up at the top | 1 | 4 blue Keese |
| 6 | `$7C` | cellar (at `$60`,`$9D`) | nothing (the room is empty if it was cleared) | E (open on this side) | 1 | |
| 7 | `$7D` | W | cross | E (open) | 1 | Manhandla stays dead |
| 8 | `$7E` | W | cross to the N door | N (open) | 1 | |
| 9 | `$6E` | S | bomb the N wall: stand at the top of the floor at X = `$78` facing up | N (bombed wall) | 1 | Manhandla: need not die |
| 10 | `$5E` | S | kill 5 blue Darknuts: the key appears at `$80`,`$90` and the N shutter opens; take the key | N (shutter) | 2 | four corner statues shoot |
| 11 | `$4E` | S | walk straight up; unlock the N door | N (key door, 1 key) | 1 | 2 Bubbles, 3 Darknuts, 3 Gibdos: need not die |
| 12 | `$3E` | S | kill the 6 blue Darknuts now (needed at step 20); bomb the N wall | N (bombed wall) | 1 | two centre statues shoot |
| 13 | `$2E` | S | walk straight up; unlock the N door (killing the Manhandla gives the map) | N (key door, 1 key) | 0 | Manhandla: need not die |
| 14 | `$1E` | S | kill the Gohma with arrows shot up into the open eye: the E shutter opens | E (shutter) | 0 | four corner statues shoot; needs bow, arrows, rupees |
| 15 | `$1F` | W | kill 2 Pols Voices, 2 blue and 2 red Darknuts; push the block at `$60`,`$90` up or down; stairs at `$80`,`$8D` | stairs | 0 | |
| 16 | `$0F` | stairs (left ladder) | as step 5: take the magic key | left ladder | 0 (magic key) | 4 blue Keese |
| 17 | `$1F` | cellar (at `$60`,`$9D`) | nothing | W (open on this side) | - | |
| 18 | `$1E` | E | cross to the S door | S (key door, already open) | - | Gohma stays dead; statues still shoot |
| 19 | `$2E` | N | walk straight down | S (bombed wall, open) | - | |
| 20 | `$3E` | N | if the Darknuts are dead the E shutter opens at once; else kill them | E (shutter) | - | |
| 21 | `$3F` | W | walk straight E along the row to the stairs at `$D0`,`$8D` (killing all gives 4 bombs) | stairs | - | 2 Bubbles, 3 Darknuts, 3 Gibdos: need not die |
| 22 | `$2F` | stairs (right ladder, X = `$C0`) | ladder down, bottom passage W, left ladder (X = `$30`) up | left ladder, Up at the top | - | 4 blue Keese; tunnel |
| 23 | `$4C` | cellar (at `$70`,`$7D`, inside the block ring) | go W one square, down two squares to the row Y = `$9D`, E along it to X = `$B0`, N up that column to the top row (Y = `$5D`), W to X = `$78`; bomb the N wall facing up | N (bombed wall) | - | 8 Pols Voices: need not die; do not step onto the stairs two squares E of the arrival spot |
| 24 | `$3C` | S | kill the Gleeok (sword only); take the heart container at `$20`,`$C0` | N (shutter) | - | boss |
| 25 | `$2C` | S | walk up the column X = `$78` to the Triforce piece | level ends | - | no monsters |

Room list: `$7E` `$7F` `$7E` `$7D` `$7C` `$6F` `$7C` `$7D` `$7E` `$6E` `$5E` `$4E` `$3E` `$2E` `$1E` `$1F` `$0F`
`$1F` `$1E` `$2E` `$3E` `$3F` `$2F` `$4C` `$3C` `$2C`.

- Key balance: found 2 (`$7F`, `$5E`), used 2 (`$4E` N at step 11, `$2E` N at step 13), 0 left when the magic key is taken; no key door is touched after that. Bombs used on walls: 3 (`$6E` N, `$3E` N, `$4C` N). [`Door_TouchKey`](../src/bank5/touch-door.asm#L160), [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136)
- Second key, alternative 1 (same number of room entries, no ladder): skip step 1; in `$5E`, once the Darknuts are dead, also the W shutter is open: go W into `$5D` (dark; 2 Pols Voices, 2 Gibdos, 2 Keese, nothing has to die), take the key lying at the centre and come back E (open on that side). The centre is reached from the E door by going round the diagonal row of blocks along the bottom row. [`Shutter_OpenTriggered`](../src/bank5/shutters-door-flags.asm#L68), [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)
- Second key, alternative 2 (same number of room entries, no second key, 2 more bombs): from `$2E` bomb the W wall into `$2D` (rupee stash, 10 rupees), go N into `$1D` (5 Pols Voices, need not die) and bomb its E wall into `$1E`. On the way back the `$1E` S key door opens with the magic key. [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136), [`RupeeStash_Update_Full`](../src/bank1/trap-rupee-stash.asm#L338)
- With one key carried in from an earlier level, step 1 and the walk back (two room entries) drop out: 23 entries. [`InvKeys`](../src/Variables.inc#L584)
- Bomb walls: the test is made once, when the blast ends, on the bomb's own position: within `$18` pixels on both axes of (`$78`,`$5D`) for a N wall. A bomb is placed `$10` pixels ahead of Link, so Link at X = `$78` on the top floor row facing up is right. The wall can be walked through 8 frames later and stays open for good, from both sides. [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136), [`Bomb_WallHotspotsX`](../src/bank7/bomb.asm#L60), [`Door_Update`](../src/bank5/update-doors.asm#L60)
- Order that matters: the `$5E` key must be taken before `$4E` N; the Gohma must die before `$1F` can be entered; in `$7C` and `$1F` the block is pushed only after the room is cleared, and only up or down; in `$3E` the Darknuts must be dead in the visit in which Link leaves E (or the room still be remembered as cleared: it is, steps 13-19 enter only four other rooms). [`PushBlock_State0Idle`](../src/bank4/push-block.asm#L74), [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50)
- Shutters are never saved: one that opened closes again when Link next enters the room, and reopens only when the room's trigger is met again (at once in a room that comes up empty). Entering through a shutter closes it behind Link. [`Door_Update`](../src/bank5/update-doors.asm#L60), [`Shutter_OpenTriggered`](../src/bank5/shutters-door-flags.asm#L68)
- The book first, then the north: steps 1-8 can also be done after step 20 instead, but then the Triforce cannot be taken before walking back; the order given keeps the level's end at the Triforce. Without the book the route is steps 0-2 and 9-25: 19 room entries. [`Item_TakeClass0PerLevel`](../src/bank1/take-item.asm#L421)
- After a death: the continue menu puts Link back in `$7E` with 3 hearts (containers kept), without reloading the level. Inventory (book, magic key, keys, bombs left), unlocked doors, bombed walls and taken items are kept; Manhandlas, Gohmas and the Gleeok that died stay dead. The room history is cleared, so cleared rooms of ordinary monsters are full again. [mechanics.md](../mechanics.md) "Hearts, rings and death". [`Mode8Continue_Update_Full`](../src/bank5/mode8-continue-mode10-stairs.asm#L68), [`Mode8Continue_Init`](../src/bank5/mode-inits-obj-lists.asm#L49), [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50)
- No Wallmaster exists in this level: nothing but death or the S door of `$7E` takes Link back to the start. [`Level_BlockUW7to9Q1`](../src/bank6/level-blocks-common-data.asm#L74)

# Optional detours

- Map (`$2E`): no extra room; kill the Manhandla there at step 13 or 19 and take the map at the centre. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61)
- Compass (`$5F`): from `$5E` through its E key door (1 key, or free with the magic key) and back: 2 room entries. It lies at `$80`,`$90`, on the row of the W door; the water of that room is one row further N and is not in the way. Dark, 8 Pols Voices, nothing has to die. [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51), [`Door_TouchKey`](../src/bank5/touch-door.asm#L160)
- Keys `$5D` and `$5C`: W of `$5E` (its W shutter opens with the same kill as the N one). `$5D`: key in the open, +2 entries. `$5C` (one room further W through open doors): kill 3 red Darknuts for the key, +2 more entries. Worth it only before the magic key. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61)
- Key `$4C`: in the room of step 23, at the top right corner (`$D0`,`$60`), outside the block ring; reached by the left column, the bottom row and the right column. No extra room, a long walk, useless with the magic key. [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)
- Key `$4B` and the hint of `$3B`: from `$4C` kill the 8 Pols Voices to open its W shutter, enter `$4B` (dark, 6 red Darknuts, two shooting statues, key in the open), go N through the open door to `$3B` (old man: "10TH ENEMY HAS THE BOMB."), then bomb the E wall of `$3B` into the boss room. 2 more room entries than the direct way, 1 bomb either way. [`Person_UpdateUnderworld_Full`](../src/bank1/underworld-person-update.asm#L44), [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136)
- The hint means the help drop: the tenth kill in a row without Link being hurt, made with a bomb, drops a bomb. [enemies.md](../enemies.md) "Drop tables". [`Damage_KillMonster`](../src/bank1/weapon-damage-death.asm#L216), [`Drop_SetUpItem`](../src/bank4/drop-items.asm#L157)
- Bombs `$0E`: from `$1E` through its N shutter (open once the Gohma is dead) into `$0E`; kill 5 red Darknuts: 4 bombs appear and the S shutter lets Link back. +2 entries. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61)
- Bombs `$1D` and rupee stash `$2D`: on alternative 2 of the route; or from `$1E` bomb the W wall (1 bomb), kill 5 Pols Voices in `$1D` for 4 bombs, S into `$2D` for 10 rupees, and bomb its E wall into `$2E`. Replaces step 19 by 3 entries; 2 bombs spent, 4 gained. [`RupeeStash_Update_Full`](../src/bank1/trap-rupee-stash.asm#L338), [`Item_Take_CheckClass1`](../src/bank1/take-item.asm#L337)
- Bombs `$3F`: no extra room; kill the 3 Darknuts and 3 Gibdos at step 21 and 4 bombs appear at the centre. Useful before the boss wall if the bag is empty. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61)
- Rupees: 5 each in `$7D` (always, step 3), `$6E` (kill the Manhandla) and `$4E` (kill 3 Darknuts and 3 Gibdos). [`Item_Take`](../src/bank1/take-item.asm#L259)
- Second Gohma (`$4D`) and the hint of `$3D`: from `$4E` through its W key door (free with the magic key), or from `$5D` N; kill the Gohma to open the N shutter of `$4D`; the old man of `$3D` says "SPECTACLE ROCK IS AN ENTRANCE TO DEATH.". Nothing else is there: 4 entries from `$4E` and back. [`Person_TextSelectorsLevels3468`](../src/bank1/underworld-person-init.asm#L134), [`Gohma_HandleWeaponCollision`](../src/bank4/gohma.asm#L367)

# Hazards and rules that matter here

- Shutters (9 doorways): each opens only while its own room's trigger is met, here always "no monsters left" (triggers 1 and 7); Bubbles and fireballs do not count as monsters. They open one at a time, 8 frames each, in the order N, S, W, E. [mechanics.md](../mechanics.md) "Doorways, doors, keys, shutters and bomb walls". [`Secret_Trigger1AllDead`](../src/bank5/underworld-secrets.asm#L152), [`Shutter_OpenTriggered`](../src/bank5/shutters-door-flags.asm#L68), [`Door_TouchShutter`](../src/bank5/touch-door.asm#L126)
- Six doorways have a shutter on one side and an open doorway on the other (`$1E`/`$1F`, `$2C`/`$3C`, `$3D`/`$4D`, `$4E`/`$5E`, `$5D`/`$5E`, `$7C`/`$7D`): the type is read from the room Link is leaving, so the dead ends `$7C`, `$1F`, `$2C` and `$3D` can always be left. [`Door_FindTypeByDirBit`](../src/bank5/fill-walls.asm#L55)
- Push blocks (`$7C`, `$1F`): no move while a monster is left; Link exactly in line with the block, at most `$10` pixels away, holding only the direction toward it for `$10` frames in a row; one move per room entry, in the direction pushed. Both rooms have secret trigger 0 or 1: the push itself opens nothing, it only clears the way to stairs that are part of the floor. [mechanics.md](../mechanics.md) "Push blocks and room secrets". [`PushBlock_State0Idle`](../src/bank4/push-block.asm#L74), [`PushBlock_State1Moving`](../src/bank4/push-block.asm#L165)
- A block pushed E (from the W side) seals the stairs; leave the room and come back to reset it. In `$1F` that means back through `$1E`, whose E shutter then needs the room empty (it is, the Gohma is non-recurring). [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L234), [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50)
- Bombable walls (6 doorways): drawn as plain wall until opened; the positions are the middle of each side. No false walls in this level. [`Door_TouchBombable`](../src/bank5/touch-door.asm#L108), [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136)
- Key doors: unlocking sets Link's timer to `$20` and the door opens 8 frames later. Only one door command runs at a time: while a shutter or bombed wall of the same room is opening the key test is skipped, and a touch on a locked door then passes Link through it unopened and unpaid (the keyless pass of [mechanics.md](../mechanics.md) "Doorways, doors, keys, shutters and bomb walls"; it can happen in `$5E` at the E door while the shutters open). No route step depends on it. [`Door_TouchKey`](../src/bank5/touch-door.asm#L160), [`Door_Update`](../src/bank5/update-doors.asm#L60)
- Dark rooms (`$7F`, `$5D`, `$5F`, `$4B`): picture only. A standing candle or book fire lights the room until Link leaves it. [`Room_IsDark_Bank4`](../src/bank4/shoot-helpers-candle.asm#L184), [`Candle_State0Begin`](../src/bank4/shoot-helpers-candle.asm#L147)
- Water: `$7F` (water on both sides of the centre floor where the key lies; one square of it on the row from the W door, crossed with the ladder, which is laid when Link walks into the water tile `$F4` holding the direction he faces) and `$5F` (one full row of water N of the door row; the compass is on the near side). [items.md](../items.md) "Ladder". [`Link_EndMoveAndAnimate`](../src/bank7/link-animate-ladder.asm#L150), [`Ladder_CheckAndDraw`](../src/bank5/ladder.asm#L44)
- Bubbles: sword off for 151-160 frames per touch, and the touch zeroes the kill counters that drive the forced drops. [`Bubble_Update`](../src/bank4/bubble-keese.asm#L93), [`Link_BeHarmed`](../src/bank1/link-collision-harm.asm#L193)
- Statues: see `# Monsters and boss`; they shoot from the first frame in `$1E`, `$5C`, `$5E`, `$3E`, `$4B`, `$7D` and cannot be stopped by killing anything. [`Statue_Update`](../src/bank4/statues.asm#L86)
- Old men (`$3B`: "10TH ENEMY HAS THE BOMB."; `$3D`: "SPECTACLE ROCK IS AN ENTRANCE TO DEATH."): in this level they only talk. Link is halted while the text is typed (one character per 6 frames) and cannot walk above Y = `$8D` in their rooms. Hitting one with a weapon does not kill him; it turns on fireballs from beside the two fires until Link leaves the room. [`Person_UpdateUnderworld_Full`](../src/bank1/underworld-person-update.asm#L44), [`Person_CheckCollisions`](../src/bank1/underworld-person-update.asm#L89), [`Person_TextSelectorsLevels3468`](../src/bank1/underworld-person-init.asm#L134)
- Stairs: a staircase takes Link down when he stands exactly on it (X a multiple of `$10`, Y ending in `$D`) at the end of a step; after coming up he must finish one grid step before a staircase works again. In `$4C` the stairs are two squares E of where he comes out. [mechanics.md](../mechanics.md) "Stairs, cellars and caves". [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)
- Cellars: four blue Keese each, whatever the data says; no bomb wall test and no doorways in a cellar; the exit is chosen by Link's X at the top of a ladder (X < `$80`: the left exit's room). The tunnel `$2F` is the only link between the boss side (`$4C`, `$3C`, `$2C`, `$3B`, `$4B`) and the rest of the level. [`Warp_CheckSubroomExit`](../src/bank5/cave-cellar-exits.asm#L43), [`Spawn_AssignPositions`](../src/bank5/spawn-positions.asm#L64), [`Mode9Cellar_LinkStartXs`](../src/bank5/cave-cellar-enter.asm#L262)
- Boss roar: rooms `$3B`, `$3D` and `$4C` repeat sample request `$10` until the Gleeok is dead; it is sound only. [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42)
- Room refill: an ordinary room cleared more than six different rooms ago is full again; Manhandla, Gohma, Gleeok and the rupee stash never come back. [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50), [`RoomHistory_SaveKillCountUW`](../src/bank5/room-history-uw.asm#L110)
- No room id of this level is special-cased by the code; the only tests by value that apply here are the cellar layouts `$3E`/`$3F` and the statue layouts `$23`/`$24`. [mechanics.md](../mechanics.md) "Special-cased rooms". [`Mode9Cellar_Init_LayOutCellar`](../src/bank5/layout-room-ow.asm#L418), [`Statue_RoomLayoutIds`](../src/bank4/statues.asm#L45)

# Second quest

- In Q2 the dungeons are loaded from a different level block, so the level entered with `CurLevel` = 8 is another dungeon: its rooms are in [data/levelblock-uw-7to9-q2.md](../data/levelblock-uw-7to9-q2.md) and its patched level info (shown as LEVEL-7, start room `$79`, Triforce room `$1B`, boss room `$1C`) in [data/levelinfo.md](../data/levelinfo.md). [`Level_BlockUW7to9Q2`](../src/bank6/level-blocks-common-data.asm#L82), [`Quest_Q2LevelInfoPatch8`](../src/bank6/quest2-replacements.asm#L170)
- Its overworld entrance is the tree of room `$6C`, not `$6D`. This document covers the first quest only. [`Secret_IsForOtherQuest`](../src/bank4/rock-wall-tree-secrets.asm#L331)
