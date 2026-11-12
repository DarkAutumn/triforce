# Game mechanics

How the game works underneath: modes and the frame loop, object slots, random numbers, timers, Link's
movement, tile collision, screen edges, the overworld screen wrap and the grid-point skip (block and ladder clips), doors and secrets, stairs and caves, the
mazes, rooms the code treats specially, spawning and room history, combat, hearts and death, saved state, level
data and the second quest. Every bullet ends with the label that proves it. `(trace)` marks a result observed
in the running game; `(unverified)` a statement not confirmed in the code. Monsters are in
[enemies.md](enemies.md), items in [items.md](items.md), decoded level data in [data/](data/README.md).

Contents (generated; `L` = line in this file):

<!-- toc -->
- L36 Game modes and the frame loop
- L105 Object slots and update order
- L155 Random numbers
- L184 Timers
- L225 Link's movement: grid, speed and sub-pixels
- L289 Tile collision and walking half into blocks
- L346 Screen edges and room transitions
- L400 Screen wrap (overworld)
- L485 Grid-point skip: block clip, ladder clip and their run-up
- L775 Doorways, doors, keys, shutters and bomb walls
- L832 Push blocks and room secrets
- L867 Stairs, cellars and caves
- L910 Lost Woods and Lost Hills (mazes)
- L925 Special-cased rooms
- L965 Ladder and raft
- L989 Monster spawning, edge spawns and respawn (room history)
- L1051 Combat: collision boxes, damage, invincibility, knockback, stun
- L1123 Hearts, rings and death
- L1160 Room flags and saved state
- L1189 Level data formats
- L1202 Second quest differences
<!-- /toc -->

# Game modes and the frame loop

The whole game runs inside the NMI handler, one frame per vertical blank. `GameMode` (`$12`) picks a row of two
jump tables; `IsUpdatingMode` (`$11`) picks the init or the update column; `GameSubmode` (`$13`) is the step inside it.

| Mode | Meaning | Init routine | Update routine |
|---|---|---|---|
| `$00` | demo, title | [`Mode0Demo_InitPatternsOrDemo`](src/bank7/init-mode-dispatch.asm#L114) | [`Mode0Demo_Update`](src/bank2/mode0-demo.asm#L79) |
| `$01` | file menu | [`Mode1Menu_Init`](src/bank7/init-mode-dispatch.asm#L136) | [`Mode1Menu_Update`](src/bank2/mode1-menu-update.asm#L43) |
| `$02` | load level | [`Mode2Load_Init`](src/bank7/init-mode-dispatch.asm#L147) | [`Mode2Load_Update`](src/bank7/update-mode-dispatch.asm#L145) |
| `$03` | unfurl (curtain) | [`Mode3Unfurl_Init`](src/bank7/init-mode-dispatch.asm#L234) | [`Mode3Unfurl_Update`](src/bank7/update-mode-dispatch.asm#L177) |
| `$04` | enter room | [`Mode4Enter_Init`](src/bank5/enter-room.asm#L65) | [`Mode4Enter_UpdateEnterOrLeave`](src/bank7/update-mode-dispatch.asm#L197) |
| `$05` | play | [`Mode5Play_Init`](src/bank7/init-mode5-play.asm#L60) | [`Mode5Play_Update`](src/bank7/mode5-play-update.asm#L59) |
| `$06` | leave room | [`Mode6Leave_Init`](src/bank5/mode3-mode6-init.asm#L214) | [`Mode4Enter_UpdateEnterOrLeave`](src/bank7/update-mode-dispatch.asm#L197) |
| `$07` | scroll to next room | [`Mode7Scroll_Init`](src/bank7/init-mode-dispatch.asm#L181) | [`Mode7Scroll_Update`](src/bank7/update-mode-dispatch.asm#L96) |
| `$08` | continue/save/retry menu | [`Mode8Continue_Init`](src/bank5/mode-inits-obj-lists.asm#L49) | [`Mode8Continue_Update`](src/bank7/update-mode-dispatch.asm#L112) |
| `$09` | cellar | [`Mode9Cellar_Init`](src/bank5/room-history-ow.asm#L173) | [`Mode5Play_Update`](src/bank7/mode5-play-update.asm#L59) |
| `$0A` | leave cave or cellar | [`ModeALeaveCellar_Init`](src/bank5/cave-cellar-enter.asm#L74) | [`Mode5Play_Update`](src/bank7/mode5-play-update.asm#L59) |
| `$0B` | cave | [`ModeBCave_Init`](src/bank5/room-history-ow.asm#L43) | [`Mode5Play_Update`](src/bank7/mode5-play-update.asm#L59) |
| `$0C` | shortcut stairs cave | [`ModeCShortcut_Init`](src/bank5/room-history-ow.asm#L63) | [`Mode5Play_Update`](src/bank7/mode5-play-update.asm#L59) |
| `$0D` | save | [`ModeDSave_Init`](src/bank5/mode-inits-obj-lists.asm#L67) | [`ModeDSave_Update`](src/bank2/moded-save.asm#L66) |
| `$0E` | register name | [`ModeERegister_Init`](src/bank7/init-mode-dispatch.asm#L214) | [`ModeERegister_Update`](src/bank2/register-eliminate-update.asm#L42) |
| `$0F` | eliminate file | [`ModeERegister_Init`](src/bank7/init-mode-dispatch.asm#L214) | [`ModeFEliminate_Update`](src/bank2/register-eliminate-update.asm#L178) |
| `$10` | stairs (sink into an entrance) | [`Mode10Stairs_Init`](src/bank5/mode-inits-obj-lists.asm#L82) | [`Mode10Stairs_Update`](src/bank7/update-mode-dispatch.asm#L120) |
| `$11` | death | [`Mode11Death_Init`](src/bank5/mode11-death-init.asm#L41) | [`Mode11Death_Update`](src/bank7/update-mode-dispatch.asm#L128) |
| `$12` | end level (Triforce piece) | [`Mode12EndLevel_Init`](src/bank5/mode12-lay-out-room.asm#L56) | [`Mode12EndLevel_Update`](src/bank7/update-mode-dispatch.asm#L136) |
| `$13` | win game | [`Mode13WinGame_Init`](src/bank7/init-mode-dispatch.asm#L222) | [`Mode13WinGame_Update`](src/bank2/mode13-win-game.asm#L51) |

**Init, update and submodes**
- The main thread is an empty loop; every frame is one run of the NMI handler, which keeps the NMI disabled until it returns. [`Reset_LoopForever`](src/bank7/run-game-nmi.asm#L84) [`Nmi_Isr`](src/bank7/run-game-nmi.asm#L140)
- Each frame runs exactly one of the two: the mode's init routine while `IsUpdatingMode` = 0, its update routine otherwise. [`Nmi_Isr@Update`](src/bank7/run-game-nmi.asm#L311) [`IsUpdatingMode`](src/Variables.inc#L13)
- Init dispatch: bank 5 is switched in, then a table jump on `GameMode`; modes `$0E` and `$0F` share one init. [`Mode_Init`](src/bank7/init-mode-dispatch.asm#L80) [`Mode_Init_JumpTable`](src/bank7/init-mode-dispatch.asm#L87)
- Update dispatch: bank 2 is switched in, then a table jump on `GameMode`. [`Mode_Update`](src/bank7/update-mode-dispatch.asm#L62) [`Mode_Update_JumpTable`](src/bank7/update-mode-dispatch.asm#L69)
- An init may last many frames, stepping `GameSubmode`; it ends by calling the routine that sets `GameSubmode` = 0 and `IsUpdatingMode` = 1, so the first update runs on the next frame. [`Mode_BeginUpdate`](src/bank1/status-bar-rupees.asm#L53)
- Ending a mode clears `IsUpdatingMode` and `GameSubmode`; the next frame runs the init of whatever `GameMode` then holds. [`Mode_End`](src/bank7/update-mode-dispatch.asm#L164)
- The "next mode" routine adds 1 to `GameMode` first: 2 -> 3, 3 -> 4, 4 -> 5, 6 -> 7. [`Mode_GoToNext`](src/bank7/update-mode-dispatch.asm#L155)
- Inits with submodes: mode 1; mode 2; mode 3 (0-8); mode 4 (0-3, 0 = the room entry itself); mode 7 (0-6); mode 9 (0-9); mode `$A` (0-`$0A`); modes `$B` and `$C` (0-8); modes `$E`/`$F` (0-4); mode `$13` (0-4). [`Mode3Unfurl_Init_JumpTable`](src/bank7/init-mode-dispatch.asm#L242) [`Mode4Enter_Init`](src/bank5/enter-room.asm#L65) [`Mode7Scroll_InitSubmodes_JumpTable`](src/bank5/mode7-scroll-init.asm#L60) [`Mode9Cellar_Init_JumpTable`](src/bank5/room-history-ow.asm#L178) [`ModeALeaveCellar_Init_JumpTable`](src/bank5/cave-cellar-enter.asm#L79) [`ModeBCave_Init_JumpTable`](src/bank5/room-history-ow.asm#L48) [`Mode13WinGame_Init_JumpTable`](src/bank2/mode13-ending-init.asm#L54)
- Updates with submodes: mode 0; mode 1 (0 choose, 1 act); mode 7; mode `$D` (0-2); mode `$11` (`$00`-`$0C`); mode `$12`; mode `$13` (0-4). [`Mode1Menu_Update_JumpTable`](src/bank2/mode1-menu-update.asm#L48) [`Mode7Scroll_UpdateSubmode_JumpTable`](src/bank5/mode7-scroll-update.asm#L70) [`ModeDSave_Update_JumpTable`](src/bank2/moded-save.asm#L71) [`Mode11Death_Update_Full_JumpTable`](src/bank5/mode11-death.asm#L66) [`Mode13WinGame_Update_JumpTable`](src/bank2/mode13-win-game.asm#L56)
- Modes 5, 9, `$A`, `$B` and `$C` all use the play update; modes 4 and 6 share one update that walks Link without input. [`Mode_Update_JumpTable`](src/bank7/update-mode-dispatch.asm#L69) [`Mode4Enter_UpdateEnterOrLeave`](src/bank7/update-mode-dispatch.asm#L197)
- Mode 2's update lasts one frame (Q2 patches), then mode 3 follows. [`Mode2Load_Update`](src/bank7/update-mode-dispatch.asm#L145)
- Mode 5's init lasts one frame. [`Mode5Play_Init`](src/bank7/init-mode5-play.asm#L60)
- The first frame after reset runs no mode: it copies the common code and data to RAM and writes the save RAM markers. [`Mode_InitGameOrMode`](src/bank7/init-mode-dispatch.asm#L60)
- After reset `GameMode` = 0 and `IsUpdatingMode` = 0 (RAM is cleared), so the game starts in the init of mode 0. [`Reset_RunGame`](src/bank7/run-game-nmi.asm#L70) [`Reset_ClearRam`](src/bank5/save-ram-clear-ram.asm#L81)

**What the NMI does each frame, in order**
- 1: writes PPU control with the NMI disabled (toggling the base name table if a switch was requested), then the PPU mask. [`Nmi_Isr`](src/bank7/run-game-nmi.asm#L140)
- 2: sprite DMA from page `$02`. [`Nmi_Isr@SetPpuMask`](src/bank7/run-game-nmi.asm#L169)
- 3: sends the transfer buffer selected by `TileBufSelector` to the PPU (bank 6). [`Transfer_SendCurBuf`](src/bank6/transfer-buf-dispatch.asm#L116)
- 4: with `IsSprite0CheckActive` set, waits for the sprite-0 hit and splits the scroll below the status bar (bank 5). [`Scroll_WaitSprite0AndSplit`](src/bank5/sprite0-split-scroll.asm#L34)
- 5: writes the scroll registers only while a mode is updating and the mode is 0, 5, 9, `$B`, `$C` or `$13`. [`Nmi_Isr@CheckScroll`](src/bank7/run-game-nmi.asm#L212)
- 6: ticks the timers, unless `SubmenuState` or `PauseState` is nonzero. [`Nmi_Isr@UpdateTimers`](src/bank7/run-game-nmi.asm#L239)
- 7: reads the controllers, unless `IsSprite0CheckActive` is set (room scroll): then `ButtonsPressed` and `ButtonsDown` keep their last values. [`Nmi_Isr@CheckInput`](src/bank7/run-game-nmi.asm#L272) [`Input_ReadControllers`](src/bank7/ppu-tablejump-input.asm#L200)
- 8: steps the random shift register. [`Nmi_Isr@ScrambleRandom`](src/bank7/run-game-nmi.asm#L279)
- 9: runs the sound driver (bank 0), then adds 1 to `FrameCounter`. [`Nmi_Isr@LoopRandom`](src/bank7/run-game-nmi.asm#L295) [`FrameCounter`](src/Variables.inc#L22)
- 10: runs the mode's init or update, then re-enables the NMI. [`Nmi_Isr@EnableNMI`](src/bank7/run-game-nmi.asm#L314)
- Timers, input and the random step all come before the mode code of the same frame. [`Nmi_Isr`](src/bank7/run-game-nmi.asm#L140)
- A frame that runs past the next vertical blank loses that NMI: no timer tick, random step or `FrameCounter` increment happens for the lost frame. [`Nmi_Isr`](src/bank7/run-game-nmi.asm#L140)

**Play update, pause and submenu**
- While `FluteTimer` is nonzero (`$98` frames after the flute is used) the play update returns at once: no input, no pause, nothing moves. [`Mode5Play_Update`](src/bank7/mode5-play-update.asm#L59) [`Flute_Wield`](src/bank7/flute.asm#L49)
- While a candle is brightening a dark room only that fade runs. [`Mode5Play_Update`](src/bank7/mode5-play-update.asm#L59) [`Candle_UpdateRoomLight`](src/bank4/shoot-helpers-candle.asm#L132)
- Select (newly pressed) toggles `PauseState` between 0 and 1; it is ignored while `PauseState` = 2. [`Mode5Play_Update@CheckMenuAndPause`](src/bank7/mode5-play-update.asm#L71) [`PauseState`](src/Variables.inc#L218)
- `PauseState` = 2 is set by drinking a potion and cleared when the heart refill ends. [`Potion_Wield`](src/bank5/wield-items.asm#L269) [`World_FillHearts`](src/bank5/cave-cellar-enter.asm#L343)
- While paused only the heart refill and the rupee counter run. [`Mode5Play_Update@CheckPaused`](src/bank7/mode5-play-update.asm#L89) [`World_UpdateHeartsAndRupees`](src/bank7/mode5-play-update.asm#L311)
- Start (newly pressed, not paused, submenu closed) sets `SubmenuState` = 1; nothing else runs that frame. [`Mode5Play_Update@NotInMenu`](src/bank7/mode5-play-update.asm#L113)
- While `SubmenuState` is nonzero the play update only runs the submenu, which ends with hearts and rupees. [`Mode5Play_Update@CheckMenu`](src/bank7/mode5-play-update.asm#L98) [`Submenu_UpdateAndDrawMeters`](src/bank5/submenu-states.asm#L60)
- Opening the submenu takes 64 frames in a dungeon and 65 in the overworld from the Start press to the first active frame; closing scrolls for 59 frames. [`Submenu_ScrollDownUW`](src/bank5/submenu-states.asm#L207) [`Submenu_ScrollUp`](src/bank5/submenu-states.asm#L345)
- Pause and submenu stop all NMI timers; the random register, `FrameCounter` and sound keep running. [`Nmi_Isr@UpdateTimers`](src/bank7/run-game-nmi.asm#L239)

# Object slots and update order

Per-object arrays (`ObjX`, `ObjY`, `ObjDir`, `ObjState`, `ObjTimer`, ...) are indexed by object slot `$00`-`$13`.
Some arrays are shorter: `ObjType`, `ObjMetastate`, `ObjStunTimer`, `ObjHP`, `ObjAttr` and `ObjInputDir` cover `$00`-`$0C` only.

**What lives in each slot**
- `$00`: Link. [`Link_Update`](src/bank7/mode5-play-update.asm#L330)
- `$01`-`$0B`: monsters and every other room object (persons, monster shots, dropped items, the ladder); a slot is free when its `ObjType` is 0. [`Mode5Play_UpdateWorld@LoopObject`](src/bank7/mode5-play-update.asm#L198) [`ObjType`](src/Variables.inc#L303)
- `$01`: the first monster the room makes; its type is kept as `RoomObjTemplateType`; cave and dungeon persons are in slot 1. [`Room_EnterCommon@StoreObjTemplate`](src/bank5/enter-room.asm#L378) [`Spawn_AssignPositions@InCave`](src/bank5/spawn-positions.asm#L155)
- `$0B`: also the room's tile object (push block, rock, gravestone, armos statue, rock wall, tree, dock). [`Room_SetUpTileObjOW`](src/bank5/enter-room.asm#L429) [`Room_SetUpItemAndPushBlock`](src/bank5/create-room-objects.asm#L51)
- `$0C`: no object is updated here; its `ObjX` is a curtain column and its `ObjTimer` paces the palette fade. [`Mode_UpdateCurtainEffect`](src/bank1/map-marker-curtain.asm#L150) [`Palette_AnimateFade`](src/bank1/world-fading.asm#L44)
- `$0D`: sword or rod. [`Sword_UpdateSwordOrRod`](src/bank7/sword-and-rod.asm#L100)
- `$0E`: sword shot or magic shot. [`SwordShot_UpdateOrMagicShot`](src/bank7/sword-shot-magic-shot.asm#L56)
- `$0F`: boomerang or food. [`Boomerang_UpdateOrFood`](src/bank7/boomerang-arrow-food.asm#L70)
- `$10` and `$11`: bomb or fire, two slots. [`Bomb_UpdateBombOrFire`](src/bank7/bomb.asm#L71)
- `$12`: rod or arrow. [`Rod_UpdateRodOrArrow`](src/bank7/sword-and-rod.asm#L57)
- `$13`: the room item: item id in `RoomItemId` (= `ObjDir[$13]`), `ObjState[$13]` 0 = present, negative = hidden or none; also used to draw the item Link lifts and cave wares. [`RoomItemId`](src/Variables.inc#L195) [`Item_MoveAndDrawRoomItem`](src/bank7/room-item-draw.asm#L65)

**Order within one frame of play**
- With the clock, `$10` is added to Link's invincibility timer first. [`Mode5Play_UpdateWorld`](src/bank7/mode5-play-update.asm#L129)
- Link is updated first: invincibility timer, input, movement, animation and warp check. [`Link_Update`](src/bank7/mode5-play-update.asm#L330)
- If Link's update ended the mode (`IsUpdatingMode` = 0: room change, stairs), weapons and objects are not updated that frame. [`Mode5Play_UpdateWorld@NotInvincible`](src/bank7/mode5-play-update.asm#L139)
- Then the monsters' chase target is copied from Link's position (while `IsChasingOtherTarget` = 0). [`Mode5Play_UpdateWorld@CheckChaseTarget`](src/bank7/mode5-play-update.asm#L145)
- Then the weapon slots in the order `$0D`, `$0E`, `$0F`, `$10`, `$11`, `$12`. [`Mode5Play_UpdateWorld@UpdateWeapons`](src/bank7/mode5-play-update.asm#L154)
- Then object slots `$0B` down to `$01`, one after the other (`CurObjIndex`). [`Mode5Play_UpdateWorld@LoopObject`](src/bank7/mode5-play-update.asm#L198) [`CurObjIndex`](src/Variables.inc#L284)
- Per slot: the invincibility timer is counted (also for an empty slot), then the object's update, then, for objects with attribute bit `$01` clear and metastate 0, the generic walking draw (unless bit `$04`) and the collision test against Link and his weapons. [`Mode5Play_UpdateWorld@LoopObject`](src/bank7/mode5-play-update.asm#L198) [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101)
- So a weapon moved this frame is tested against each monster after that monster has moved, higher slots first. [`Mode5Play_UpdateWorld`](src/bank7/mode5-play-update.asm#L129)
- After the loop: dungeon checks (statues, secrets, shutters, doors) or the overworld Zora spawn, then the item lift, the room item and its pickup, then hearts and rupees. [`Mode5Play_UpdateWorld@CheckUW`](src/bank7/mode5-play-update.asm#L244) [`Mode5Play_UpdateWorld@FinishUpdatePlay`](src/bank7/mode5-play-update.asm#L296)

**Uninitialized slots and the spawn cloud**
- A slot with `ObjUninitialized` nonzero only runs the init on its first update; its own update starts on a later frame. [`Obj_Update@Init`](src/bank7/object-update-table.asm#L144) [`Obj_Init`](src/bank7/object-init-table.asm#L45)
- The init loads `ObjAttr` from the type's attribute byte and `ObjHP` = HP nibble x `$10`, then runs the type's init routine. [`Obj_Init@FetchAttrs`](src/bank7/object-init-table.asm#L141) [`Obj_Init_JumpTable`](src/bank7/object-init-table.asm#L177)
- While `ObjMetastate` is nonzero (1-3 spawn cloud, `$10`-`$13` death sparkle) the type's update does not run: the object neither moves nor is tested for collisions. [`Obj_Update@Update`](src/bank7/object-update-table.asm#L150) [`ObjMetastate`](src/Variables.inc#L327)
- First cloud step lasts as many frames as the slot number (`ObjTimer` = slot), the next two 6 frames each, so higher slots start acting later. [`Obj_Init@NormalSpawn`](src/bank7/object-init-table.asm#L122) [`Obj_AnimateMetastate@IncMetastate`](src/bank7/object-animation.asm#L71)
- Armos (`$1E`), flying Ghini (`$22`) and types `$53` and up get no slot-number timer. [`Obj_Init@NormalSpawn`](src/bank7/object-init-table.asm#L122)
- An overworld monster that walks in from the screen edge has no cloud (metastate 0) and stays uninitialized until `MonstersFromEdgesLongTimer` is 0 and its edge cell is far enough from Link. [`Obj_Init@InitMonsterFromEdge`](src/bank7/object-init-table.asm#L79) [`Obj_Init@UninitMonsterFromEdge`](src/bank7/object-init-table.asm#L70)
- A death sets metastate `$10`; when the sparkle ends the slot becomes type `$60` (dropped item), uninitialized, attribute `$81`. [`DeadDummy_StartDeathSparkle`](src/bank7/object-stubs-destroy-monster.asm#L139) [`Obj_EndMetastate@DropItem`](src/bank7/object-update-table.asm#L345)

**Finding and freeing slots**
- A new object (monster shot, split monster, ladder, fairy) takes the highest free slot from `$0B` down to `$01`; slot 0 and slots above `$0B` are never returned. [`Obj_FindEmptyMonsterSlot`](src/bank7/object-stubs-destroy-monster.asm#L169)
- An object made in a slot higher than the maker's is not updated until the next frame; one made in a lower slot gets its init later the same frame (the loop counts down). [`Mode5Play_UpdateWorld@NextObject`](src/bank7/mode5-play-update.asm#L227)
- Freeing a slot sets type 0, clears its shove, timer, state and invincibility timer, and leaves it uninitialized with metastate 1 for the next user. [`Obj_DestroyMonster`](src/bank7/object-stubs-destroy-monster.asm#L153) [`Obj_ResetSlotFields`](src/bank1/bomb-candle-wield.asm#L308)

**Room-entry clearing**
- Every room entry (modes 4, 9, `$B`, `$C`) zeroes RAM `$0300`-`$051F`: object types, grid offsets, position fractions, speeds, metastates, HP, the room kill count and the other per-room variables. [`Room_EnterCommon`](src/bank5/enter-room.asm#L143) [`Util_ClearRam0300UpTo`](src/bank7/ppu-tablejump-input.asm#L162)
- Slots `$0B`-`$01` then get: uninitialized `$FF`, state 0, direction 0, stun 0, no shove, animation counter 1, metastate 1, q-speed `$20`. [`Room_EnterCommon@ResetObjLoop`](src/bank5/enter-room.asm#L265)
- Leaving and entering a room both clear `LadderSlot` and the state of weapon slots `$0D`-`$12`: nothing Link has out survives a room change. [`Obj_ResetWeaponSlotsAndLadder`](src/bank5/mode3-mode6-init.asm#L263)
- The same moments set Link's state to 0 and turn the clock off. [`Link_ResetStateAndClock`](src/bank7/init-mode-dispatch.asm#L354)
- Object types go to slots 1..count from the room's monster list; slots `$01`-`$09` all get a spawn position whatever the count. [`Room_EnterCommon@StoreObjTemplate`](src/bank5/enter-room.asm#L378) [`Spawn_AssignPositions`](src/bank5/spawn-positions.asm#L64)

# Random numbers

One 13-byte shift register, `Random` at `$18`-`$24`, is the only random source; nothing reseeds it during play.

**The register**
- Each step: new bit = (bit 1 of `$18`) XOR (bit 1 of `$19`); then the 13 bytes rotate right as one 104-bit value, the new bit entering bit 7 of `$18` and each byte's bit 0 entering bit 7 of the next byte. [`Nmi_Isr@ScrambleRandom`](src/bank7/run-game-nmi.asm#L279) [`Random`](src/Variables.inc#L29)
- It steps once per NMI frame, before the mode code, in every mode: also while paused, in the submenu, during room scrolls, inits and menus. [`Nmi_Isr@ScrambleRandom`](src/bank7/run-game-nmi.asm#L279)
- It does not step on a lost (lag) frame, because that frame has no NMI. [`Nmi_Isr`](src/bank7/run-game-nmi.asm#L140)
- Seed: reset stores `$40` in `$18` and 0 in the other 12 bytes, so the sequence is the same after every power-on and reset. [`Reset_ClearRam`](src/bank5/save-ram-clear-ram.asm#L81)
- The step reads no controller and no game state: the register's value depends only on the number of frames run since reset. [`Nmi_Isr@LoopRandom`](src/bank7/run-game-nmi.asm#L295)
- Because of the rotation, byte n+1 holds what byte n held 8 frames earlier; consecutive frames of one byte differ by a one-bit shift. [`Nmi_Isr@LoopRandom`](src/bank7/run-game-nmi.asm#L295)

**Who reads which byte**
- Monsters read `Random,X` or `Random+1,X` with X = their object slot (`$01`-`$0B`), so each slot sees a different byte of the register. [`Random`](src/Variables.inc#L29) [`Wanderer_MoveAndTurn`](src/bank4/walker-wanderer-common.asm#L224)
- Walking monsters: `Random+1,X` against the turn rate at each 16-pixel grid point (turn toward the chase target), `Random,X` as the new turn timer. [`Wanderer_MoveAndTurn`](src/bank4/walker-wanderer-common.asm#L224)
- A walking monster blocked by a tile takes a perpendicular direction from bit 7 of `Random,X`. [`Walker_AltDir_GetRandomObjPerpendicularDir`](src/bank7/walker-collision-screen-edge.asm#L335)
- Flyers' random turn: `Random+1,X` >= `$A0` no turn, `$50`-`$9F` one 8-way step clockwise, below `$50` counter-clockwise. [`Dir_TurnObjRandomly8`](src/bank4/fairy-flyer-common.asm#L527)
- Item drops: the dead monster's `Random,X` is compared with its drop group's rate. [`Drop_SetUpItem`](src/bank4/drop-items.asm#L157)
- Chase target: `Random+1` AND 7 (no slot index) reloads `ChaseLongTimer` each time it expires. [`Mode5Play_UpdateWorld@UpdateWeapons`](src/bank7/mode5-play-update.asm#L154)
- Edge spawns: 2 + (`Random+1,X` AND 3) reloads `MonstersFromEdgesLongTimer` at each entry attempt. [`Obj_Init@InitMonsterFromEdge`](src/bank7/object-init-table.asm#L79)
- Some timing uses `FrameCounter` instead of the register: the invincibility countdown (even frames) and the step out of a cave (every 4th frame). [`Obj_DecInvincibilityTimer`](src/bank7/object-stubs-destroy-monster.asm#L119) [`Mode4Enter_StepOutside`](src/bank7/update-mode-dispatch.asm#L247)

**What a run can and cannot influence**
- Cannot: the sequence itself; button presses never enter the register. [`Nmi_Isr@ScrambleRandom`](src/bank7/run-game-nmi.asm#L279)
- Can: the frame on which a value is read; any delay (standing still, a later room entry, a longer menu) shifts every later read. [`Nmi_Isr@ScrambleRandom`](src/bank7/run-game-nmi.asm#L279)
- Can: pausing or opening the submenu, which freezes the timers and the world but not the register, so the world resumes against different random bytes. [`Nmi_Isr@UpdateTimers`](src/bank7/run-game-nmi.asm#L239) [`Mode5Play_Update@CheckPaused`](src/bank7/mode5-play-update.asm#L89)
- Can: which slot a monster is in (kill order, room history lowering the count), since the slot is the byte index. [`Room_EnterCommon@StoreObjCount`](src/bank5/enter-room.asm#L331) [`Obj_FindEmptyMonsterSlot`](src/bank7/object-stubs-destroy-monster.asm#L169)
- Spawn positions are not random: they walk a 9-entry list from `SpawnCycle`, which carries over between rooms. [`Spawn_AssignPositions`](src/bank5/spawn-positions.asm#L64) [`SpawnCycle`](src/Variables.inc#L460)

# Timers

Zero-page `$26`-`$4E` is one block that the NMI counts down; other timers are counted by the code that owns them.

**Frame timers (`$27`-`$3C`)**
- Every frame each nonzero byte of `$27`-`$3C` loses 1 and stops at 0: `DoorTimer` (`$27`), `ObjTimer` for slots `$00`-`$13` (`$28`-`$3B`), `FluteTimer` (`$3C`). [`Nmi_Isr@LoopTimer`](src/bank7/run-game-nmi.asm#L262) [`ObjTimer`](src/Variables.inc#L43)
- The tick comes before the mode code, so a timer set to N by game code is found at 0 by the same code N frames later. [`Nmi_Isr@UpdateTimers`](src/bank7/run-game-nmi.asm#L239)
- No timer in the block ticks while `SubmenuState` or `PauseState` is nonzero; they do tick in every mode, including room scrolls and inits. [`Nmi_Isr@UpdateTimers`](src/bank7/run-game-nmi.asm#L239)
- They also do not tick on a lost (lag) frame. [`Nmi_Isr`](src/bank7/run-game-nmi.asm#L140)
- `FluteTimer` = `$98` freezes the whole play update while it runs. [`Flute_Wield`](src/bank7/flute.asm#L49) [`FluteTimer`](src/Variables.inc#L49)

**Long timers and the divider (`$26`, `$3D`-`$4E`)**
- `LongTimerCycle` (`$26`) loses 1 every frame; on the frame it drops below 0 it is reset to 9 and the tick loop starts at `$4E` instead of `$3C`. [`Nmi_Isr@UpdateTimers`](src/bank7/run-game-nmi.asm#L239) [`LongTimerCycle`](src/Variables.inc#L41)
- So `$3D`-`$4E` lose 1 every 10th frame: a long timer set to N expires 10(N-1)+1 to 10N frames later, depending on the divider's phase. [`Nmi_Isr@DecTimers`](src/bank7/run-game-nmi.asm#L259)
- The divider is one shared counter, never reset by the code that loads a long timer. [`LongTimerCycle`](src/Variables.inc#L41)
- `ObjStunTimer` (`$3D`-`$49`, slots `$00`-`$0C`): stun left. [`ObjStunTimer`](src/Variables.inc#L52)
- `ChaseLongTimer` (`$4A`): until the monsters' chase target flips between Link and the mirrored point (255 - X, 255 - Y); reload 0-7. [`Mode5Play_UpdateWorld@UpdateWeapons`](src/bank7/mode5-play-update.asm#L154) [`ChaseLongTimer`](src/Variables.inc#L58)
- `MonstersFromEdgesLongTimer` (`$4B`): until the next overworld monster may enter from a screen edge; reload 2-5. [`Obj_Init@InitMonsterFromEdge`](src/bank7/object-init-table.asm#L79)
- `SwordBlockedLongTimer` (`$4C`): while nonzero the A button does not swing the sword. [`Link_HandleInput`](src/bank5/link-handle-input.asm#L68) [`SwordBlockedLongTimer`](src/Variables.inc#L62)
- `RedLeeverLongTimer` (`$4D`): until a Red Leever may surface. [`RedLeeverLongTimer`](src/Variables.inc#L65)
- `$4E` is ticked but no code uses it. [`UnusedLongTimer`](src/Variables.inc#L68)

**Object timers and their double uses**
- `ObjTimer,X` is each object's own delay counter; what it times depends on the type and state. [`ObjTimer`](src/Variables.inc#L43)
- Every spawn-cloud and death-sparkle step is timed by the slot's `ObjTimer` (slot number, then 6 frames per step). [`Obj_AnimateMetastate@CheckTimer`](src/bank7/object-animation.asm#L66) [`Obj_Init@NormalSpawn`](src/bank7/object-init-table.asm#L122)
- Slot entries without an object are used as mode timers: `ObjTimer[$0B]` the death sequence steps, `ObjTimer[$01]` the continue menu's flash, `ObjTimer[$0C]` the palette fade (10 frames per step), `ObjTimer[$13]` the boomerang sound interval. [`Mode11Death_Update_Full`](src/bank5/mode11-death.asm#L61) [`Mode8Continue_Update_Full`](src/bank5/mode8-continue-mode10-stairs.asm#L68) [`Palette_AnimateFade`](src/bank1/world-fading.asm#L44) [`Boomerang_PlaySfx`](src/bank1/modeb-cave-save-sprites.asm#L223)
- Freeing or reusing a slot zeroes its `ObjTimer`. [`Obj_ResetSlotFields`](src/bank1/bomb-candle-wield.asm#L308)

**Invincibility timers**
- `ObjInvincibilityTimer,X` (slots `$00`-`$13`, not in the NMI block) loses 1 only on frames with an even `FrameCounter`: a value N lasts about 2N frames. [`Obj_DecInvincibilityTimer`](src/bank7/object-stubs-destroy-monster.asm#L119) [`ObjInvincibilityTimer`](src/Variables.inc#L408)
- It is counted in the play update (Link at the start of his update, slots `$0B`-`$01` in the object loop), so it stands still while paused and in the submenu. [`Link_Update`](src/bank7/mode5-play-update.asm#L330) [`Mode5Play_UpdateWorld@LoopObject`](src/bank7/mode5-play-update.asm#L198)
- A hit gives Link `$18` (48 frames) and a monster `$10` (32 frames). [`Shove_Begin`](src/bank1/shove-knockback.asm#L65)
- With the clock, `$10` is added to Link's timer every frame; it never runs out and the byte wraps. [`Mode5Play_UpdateWorld`](src/bank7/mode5-play-update.asm#L129)
- A room change clears Link's invincibility timer, knockback and state. [`Mode5Play_GoToNextMode`](src/bank7/walker-collision-screen-edge.asm#L281)

**Stun timer**
- A boomerang hit on a monster not immune to it sets `ObjStunTimer,X` = `$10`: 151-160 frames. [`Collision_CheckWeaponPointVsMonster`](src/bank1/weapon-damage-death.asm#L82)
- While its stun timer or `InvClock` is nonzero a walking monster does not move. [`Walker_Move@CheckStunned`](src/bank7/walker-move.asm#L87)
- A stunned monster's touch does not hurt Link; the same test also reads Link's own entry `ObjStunTimer[0]`. [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48)
- Stun timers are zeroed on room entry and when the monster dies. [`Room_EnterCommon@ResetObjLoop`](src/bank5/enter-room.asm#L265) [`Damage_KillMonster`](src/bank1/weapon-damage-death.asm#L216)

# Link's movement: grid, speed and sub-pixels

Each frame of play Link's input is turned into a facing and a moving direction, every blocker is applied, and
then he is moved along one axis. All tests use the position the previous frame left.

**Per-frame order**
- Order: border input mask and A/B (state 0 only), direction rules, move, then the end-of-move step (grid offset reset, ladder, warps, drawing) and the grid snap. [`Link_Update`](src/bank7/mode5-play-update.asm#L330) [`Link_HandleInput`](src/bank5/link-handle-input.asm#L68) [`Walker_Move`](src/bank7/walker-move.asm#L64) [`Link_EndMoveAndAnimate`](src/bank7/link-animate-ladder.asm#L150)
- `ObjInputDir` for slot 0 starts each frame as the four direction bits held on controller 1 (1 right, 2 left, 4 down, 8 up). [`Mode5Play_Update@CheckMenu`](src/bank7/mode5-play-update.asm#L98) [`ObjInputDir`](src/Variables.inc#L324)

**Grid and grid offset**
- Link's grid is 8 pixels; every other slot's is `$10` pixels. [`Obj_MoveByQSpeed`](src/bank7/walker-move.asm#L244) [`PositiveGridCellSize`](src/Variables.inc#L261)
- `ObjGridOffset` (`$394,X`, signed) counts pixels moved since the last grid point: +1 per pixel right or down, -1 per pixel left or up; 0 = on a grid point. [`Obj_AddQSpeedToPosFrac`](src/bank1/qspeed-direction-math.asm#L52) [`Obj_SubQSpeedFromPosFrac`](src/bank1/qspeed-direction-math.asm#L94) [`ObjGridOffset`](src/Variables.inc#L311)
- Limits: at offset +8 or -8 (`$F8`) for Link, +`$10` or -`$10` for others, a pixel is withheld in both directions (the fraction still changes); the object stops exactly one cell from the grid point it left. [`Obj_AddQSpeedToPosFrac@ClearCarry`](src/bank1/qspeed-direction-math.asm#L66) [`Obj_SubQSpeedFromPosFrac@SetCarry`](src/bank1/qspeed-direction-math.asm#L108)
- 0 is not a limit: an offset whose sign is opposite to the heading counts to 0 and on through it (+1, 0, -1). [`Obj_AddQSpeedToPosFrac`](src/bank1/qspeed-direction-math.asm#L52)
- After Link's move, in modes 5 and 7 up, an offset that is a nonzero multiple of 8 is set to 0; no other value is corrected. [`Link_EndMoveAndAnimate@TruncGridOffset`](src/bank7/link-animate-ladder.asm#L177)
- At offset 0 the position is snapped to X = multiple of 8, Y = (multiple of 8) + 5, e.g. (`$78`, `$8D`). [`Obj_SnapToGrid`](src/bank7/mode5-play-update.asm#L375)
- Walking monsters reset their offset to 0 when it is a multiple of `$10` and decide their direction there. [`Wanderer_MoveAndTurn`](src/bank4/walker-wanderer-common.asm#L224)
- Tiles ahead and the screen edge are tested only on a frame that starts with offset 0. [`Walker_CheckTileCollision@CheckGridOffset`](src/bank7/walker-collision-screen-edge.asm#L101)

**Speed and sub-pixels**
- The q-speed (`ObjQSpeedFrac`, `$3BC,X`) is added to (right, down) or subtracted from (left, up) the 8-bit `ObjPosFrac` (`$3A8,X`) 4 times a frame; each carry or borrow is 1 pixel, so pixels per frame = q-speed / 64, at most 4. [`Obj_MoveByQSpeed@ApplyQSpeedToPosition`](src/bank7/walker-move.asm#L262) [`ObjQSpeedFrac`](src/Variables.inc#L317)
- Link's q-speed is `$60` = 1.5 pixels per frame: `ObjPosFrac` alternates `$00`/`$80` and he moves 1, 2, 1, 2 ... pixels; an 8-pixel step takes 5 or 6 frames. [`Link_InitSpeed`](src/bank5/link-handle-input.asm#L277) [`Obj_MoveByQSpeed`](src/bank7/walker-move.asm#L244)
- When the limit withholds the second pixel of a 2-pixel frame the fraction has still advanced: that pixel is lost, not postponed. [`Obj_AddQSpeedToPosFrac@ClearCarry`](src/bank1/qspeed-direction-math.asm#L66)
- Slow tiles: in the overworld, while Link's `ObjCollidedTile` is `$74` or `$75` (the mountain stairs; the tile identity is unverified) his q-speed is `$30` = 0.75 pixels per frame; `ObjPosFrac` is zeroed when the speed first drops. [`Link_InitSpeed`](src/bank5/link-handle-input.asm#L277) [`ObjCollidedTile`](src/Variables.inc#L341)
- The speed is set at room entry, by the grid-point rules and by "keep going" between grid points (cases 2 and 4 below), from the tile found by Link's last tile lookup; a turn back (case 3) does not set it. [`Link_InitSpeed`](src/bank5/link-handle-input.asm#L277) [`Link_ModifyDirOnGridLine`](src/bank5/link-handle-input.asm#L338)
- Monsters start a room with q-speed `$20` (0.5 pixels per frame) until their own code changes it. [`Room_EnterCommon@ResetObjLoop`](src/bank5/enter-room.asm#L265)
- One axis per frame: with several bits in the moving direction the order is right, left, down, up. [`Obj_MoveByQSpeed@ApplyQSpeedToPosition`](src/bank7/walker-move.asm#L262)
- Coordinates are 8 bits and wrap (`$00` - 1 = `$FF`). [`Obj_MoveByQSpeed`](src/bank7/walker-move.asm#L244)

**Turning at a grid point (offset 0)**
- No direction held: nothing changes and Link does not move. [`Link_ModifyDirAtGridPoint`](src/bank5/link-handle-input.asm#L117)
- One direction held: it becomes `ObjDir` and `ObjInputDir` whether or not the tile ahead is walkable (Link turns to face a wall). [`Link_ModifyDirAtGridPoint_HaveInput`](src/bank5/link-handle-input.asm#L157)
- Two held, neither walkable: `ObjInputDir` = 0, facing unchanged. [`Link_ModifyDirAtGridPoint_HaveInput`](src/bank5/link-handle-input.asm#L157)
- Two held, one walkable: he goes that way; `Link_GoStraight` is set for this step. [`Link_ModifyDirAtGridPoint_HaveInput`](src/bank5/link-handle-input.asm#L157) [`Link_GoStraight`](src/Variables.inc#L93)
- Two held, both walkable, overworld: the vertical one wins (directions are tested right, left, down, up and the last walkable one is taken). [`Link_ModifyDirAtGridPoint`](src/bank5/link-handle-input.asm#L117)
- Two held, both walkable, dungeon: the first such grid point turns Link perpendicular to his facing and sets `Link_GoStraightWhenDiagInput`; while that flag is set he keeps his facing at later grid points. [`Link_ModifyDirAtGridPoint_HaveInput@CheckGoStraightFlag`](src/bank5/link-handle-input.asm#L220) [`Link_GoStraightWhenDiagInput`](src/Variables.inc#L89)
- "Walkable" here is only the tile ahead being below `RoomFirstUnwalkableTile`; doors, room bounds and tile objects are not considered. [`Link_ModifyDirAtGridPoint`](src/bank5/link-handle-input.asm#L117)

**Between grid points (offset not 0): the six cases, in order**
- The held bits are first reduced to one direction: right, else left, else down, else up. [`Dir_GetOpposite`](src/bank1/qspeed-direction-math.asm#L137)
- Case 1, nothing held: Link stops where he is, between grid points. [`Link_ModifyDirOnGridLine`](src/bank5/link-handle-input.asm#L338)
- Case 2, same as `ObjDir`: keep going. [`Link_ModifyDirOnGridLine`](src/bank5/link-handle-input.asm#L338)
- Case 3, opposite of `ObjDir`: he reverses at once; `ObjGridOffset` is not changed, so it now counts toward 0. [`Link_ModifyDirOnGridLine`](src/bank5/link-handle-input.asm#L338) [`Link_SetDirAndInputDir`](src/bank5/link-handle-input.asm#L365)
- Case 4, perpendicular with `Link_GoStraight` set (diagonal input at the last grid point): keep going. [`Link_ModifyDirOnGridLine`](src/bank5/link-handle-input.asm#L338)
- Case 5, perpendicular, 4 or more pixels from the grid point: keep going in `ObjDir`. [`Link_ModifyDirOnGridLine`](src/bank5/link-handle-input.asm#L338)
- Case 6, perpendicular, less than 4 pixels out and still heading away from the grid point he left (offset sign agrees with the heading): he reverses and the offset is rewritten, negative n -> 8 + n, positive n -> n - 8; if the sign disagrees nothing changes. [`Link_SetInputDir@ReverseDir`](src/bank5/link-handle-input.asm#L404)
- So a perpendicular turn never happens between grid points: Link walks on (or back) to a grid point and turns there. [`Walker_Move@ChooseObjDirOrInputDir`](src/bank7/walker-move.asm#L70)
- Case 6 followed by case 3 leaves an offset counting toward 0 at the far grid point; a 2-pixel frame can step it from +1 to -1 (or -1 to +1), so no frame starts at offset 0 there and the tile, warp and screen-edge tests are skipped at that point. [`Link_ModifyDirOnGridLine`](src/bank5/link-handle-input.asm#L338) [`Obj_MoveByQSpeed`](src/bank7/walker-move.asm#L244)

**Dungeon door special cases**
- In a doorway (`DoorwayDir` nonzero) the input is forced onto the axis Link faces: forward if held, else backward if held, else (only sideways held) forward. [`Link_ModifyDirInDoorway`](src/bank5/doorway.asm#L52)
- At X = `$20` or `$D0` with two walkable directions held, Link turns perpendicular whatever the flag says, except at Y = `$85` facing down, where he keeps going down (to the side-door row Y = `$8D`). [`Link_ModifyDirAtGridPoint_HaveInput`](src/bank5/link-handle-input.asm#L157) [`Doorway_RequiredCoords`](src/bank5/doorway.asm#L93)
- At (`$78`, `$5D`) facing left or right with the flag set, he turns perpendicular (toward the top door); the bottom door has no such case. [`Link_ModifyDirAtGridPoint_HaveInput@CheckGoStraightFlag`](src/bank5/link-handle-input.asm#L220)
- In a dungeon's outer border band a held direction perpendicular to Link's facing is dropped (state 0 only). [`Link_MaskInputInBorder@MaskButtons`](src/bank5/border-input-mask.asm#L133)

**What stops movement**
- Halted: while (`ObjState` AND `$C0`) = `$40` Link's update does nothing but count the invincibility timer. [`Link_Update`](src/bank7/mode5-play-update.asm#L330)
- Sword or item use: while (`ObjState` AND `$F0`) is `$10` or `$20` the moving direction is zeroed; state `$31` does not block; a passable doorway or the laid ladder can restore it, which is why the sword gives the grid-point skip an extra position everywhere but on a laid ladder (`# Grid-point skip: block clip, ladder clip and their run-up`). [`Walker_Move@SetMovingDir`](src/bank7/walker-move.asm#L104)
- Knockback: while `ObjShoveDir` is nonzero directions are not processed and the shove routine moves Link instead (4 pixels per frame). [`Link_HandleInput`](src/bank5/link-handle-input.asm#L68) [`Shove_Update`](src/bank7/shove.asm#L55)
- Paralysis (`IsLinkParalyzed`, a Like-Like's hold): direction input is discarded, A and B still work. [`Link_Update`](src/bank7/mode5-play-update.asm#L330)
- An unwalkable tile ahead on a grid point zeroes the moving direction and `ButtonsPressed`. [`Link_StopAtUnwalkableTile`](src/bank7/walker-collision-screen-edge.asm#L167)
- A sliding tile object (state 1) within `$10` pixels on both axes stops Link; a person in slot 1 stops him moving up while Y < `$8E`. [`Link_CheckSlidingTileObjBlocking`](src/bank1/tile-block-power-triforce.asm#L45) [`Person_BlockLinkMovingUp`](src/bank1/modeb-cave-save-sprites.asm#L289)
- Room bounds stop Link only in a dungeon room outside doorways, and a laid ladder can override them (its direction rule runs after the bound test); in the overworld only tiles and the screen-edge test limit him. [`Walker_Move@CheckBoundary`](src/bank7/walker-move.asm#L185) [`Room_BoundDir`](src/bank1/room-bounds.asm#L254) [`Ladder_CheckAndDraw@HandleInput`](src/bank5/ladder.asm#L125)

# Tile collision and walking half into blocks

Background collision is a lookup in the play area tile map for one point of the object (its hotspot), made only
on a frame that starts on a grid point. Link's upper body is never tested, which is why he overlaps blocks from
below. The grid itself is under `# Link's movement: grid, speed and sub-pixels`.

**The tile map**
- The play area is 32 columns by 22 rows of 8-pixel tiles, stored by column (22 bytes each) at `$6530`; it starts at screen Y = `$40`, below the status bar. [`PlayAreaTiles`](src/Variables.inc#L691) [`Tile_PlayAreaColumnAddrs`](src/bank7/run-game-nmi.asm#L54)
- Column = X / 8; row = (Y - `$40`) / 8. [`Tile_GetAtObjOffset@FetchTile`](src/bank7/tile-collision.asm#L144)

**Which tiles are walkable**
- A tile is walkable when its number is below `RoomFirstUnwalkableTile`; the lookup only returns the tile, each caller makes the compare. [`Walker_TestTileAhead`](src/bank7/walker-collision-screen-edge.asm#L138) [`RoomFirstUnwalkableTile`](src/Variables.inc#L292)
- The limit is `$89` in the overworld and `$78` in dungeons, copied on every room entry and when play starts in modes 5, `$B` and `$C`. [`Room_ObjBoundsOW`](src/bank5/mode3-mode6-init.asm#L159) [`Room_ObjBoundsUW`](src/bank5/mode3-mode6-init.asm#L164) [`Room_SetUpObjBounds`](src/bank5/mode3-mode6-init.asm#L175)
- Overworld exception list: tiles `$8D`, `$91`, `$9C`, `$AC`, `$AD`, `$CC`, `$D2`, `$D5`, `$DF` are at or above `$89` but are reported as tile `$26`, so they pass; this holds for every object slot and is not done in dungeons. [`Tile_WalkableTilesOW`](src/bank7/tile-collision.asm#L43) [`Tile_GetAtObjOffset@CheckWalkable`](src/bank7/tile-collision.asm#L182)
- The flute's pond secret raises the limit to `$99` at its step `$0A`, which makes the pond's tiles (those below `$99`) walkable; the next room entry puts `$89` back. [`FluteSecret_CuePondPaletteRow`](src/bank7/flute-secret-pond.asm#L127) [`Room_SetUpObjBounds`](src/bank5/mode3-mode6-init.asm#L175)
- Overworld room `$1F`: for Link tested up or down at X = `$80` with Y < `$56` the result is forced to `$26`: a wall he walks through. [`Tile_GetAtObjOffset@SetTile`](src/bank7/tile-collision.asm#L199)
- The tile found stays in `ObjCollidedTile,X`; Link's value selects his slow speed (tiles `$74`, `$75`) and tells the armos and gravestone touch what he walked into. [`ObjCollidedTile`](src/Variables.inc#L341) [`Link_InitSpeed`](src/bank5/link-handle-input.asm#L277) [`Spawn_CheckTouchedArmosOrGravestone`](src/bank1/armos-ghini-spawn.asm#L54)

**Where the tile is sampled**
- Every object's hotspot is (`ObjX`, `ObjY` + `$0B`): its left edge, 11 pixels below the top of the 16-pixel sprite. [`Tile_GetAtObjOffset`](src/bank7/tile-collision.asm#L97)
- Tile ahead = hotspot moved along the direction's axis: Link up -8, left -8, down +8, right +`$10`; every other slot up -`$10`, left -`$10`, down +8, right +`$10`. [`Tile_GetAheadOfObj`](src/bank7/tile-collision.asm#L65)
- Up and down read two tiles, columns X / 8 and X / 8 + 1 of that row (the full 16-pixel width), and keep the higher number: a blocking tile in either column wins. [`Tile_GetAtObjOffset@FetchTile`](src/bank7/tile-collision.asm#L144)
- Left and right read one tile, in the hotspot's own row. [`Tile_GetAtObjOffset@FetchTile`](src/bank7/tile-collision.asm#L144)
- No look-ahead at three borders: going down with hotspot Y >= `$DD`, right with X >= `$F0`, left with X < `$10`, the tile under the hotspot is read instead. [`Tile_GetAtObjOffset@AdjustX`](src/bank7/tile-collision.asm#L123) [`Tile_GetAtObjOffset@CheckLeftBoundary`](src/bank7/tile-collision.asm#L134)
- The tile under the hotspot, with no offset, is what spawn spot tests and a surfacing Red Leever use. [`Tile_GetUnderObj`](src/bank7/tile-collision.asm#L52) [`Spawn_IsSafeSpot`](src/bank5/spawn-positions.asm#L189)

**When it is tested**
- Only on a frame that starts with `ObjGridOffset` = 0, once, before the move; between grid points nothing is tested and the object keeps moving. [`Walker_CheckTileCollision@CheckGridOffset`](src/bank7/walker-collision-screen-edge.asm#L101) [`Walker_Move@CheckTiles`](src/bank7/walker-move.asm#L206)
- Grid points are 8 pixels apart for Link and `$10` for every other slot; an aligned position is X = multiple of 8, Y = multiple of 8 plus 5. [`Obj_MoveByQSpeed`](src/bank7/walker-move.asm#L244) [`Obj_SnapToGrid`](src/bank7/mode5-play-update.asm#L375)
- Link's turning rules make the same lookup for each held direction on a grid point. [`Link_ModifyDirAtGridPoint`](src/bank5/link-handle-input.asm#L117)
- Knockback also tests the tile ahead only at grid offset 0, and ends the shove there if it is unwalkable. [`Shove_MovePixels@LoopShovePixel`](src/bank7/shove.asm#L134)

**Why Link overlaps blocks**
- For an aligned Link the hotspot row covers `ObjY` + `$0B`..`$12`. Going up he is stopped by the row covering `ObjY` + 3..`$0A`; his top 11 pixels, `ObjY`..`ObjY` + `$0A`, lie above the hotspot row and are never tested. [`Tile_GetAheadOfObj`](src/bank7/tile-collision.asm#L65)
- Below a 16-pixel block he therefore stops at `ObjY` = block top + 5: over the block's whole lower tile row and 3 pixels of its upper one (trace: level 2 room `$7D`, a statue square, solid like a block, at Y `$B0`-`$BF`, Link stops at Y = `$B5`). [`Walker_TestTileAhead`](src/bank7/walker-collision-screen-edge.asm#L138)
- Going down he is stopped by the row covering `ObjY` + `$13`..`$1A`: he stops at `ObjY` = block top - `$13`, his sprite ending 3 pixels above the block (trace: block top `$90`, Link stops at Y = `$7D`). [`Tile_GetAheadOfObj`](src/bank7/tile-collision.asm#L65)
- Left and right test only the hotspot row, so he walks along under a block with his upper 11 pixels inside it (trace: Y = `$95` under a block at Y `$90`-`$9F`, X `$50` to `$28` unhindered). [`Tile_GetAtObjOffset@FetchTile`](src/bank7/tile-collision.asm#L144)
- Sideways there is no overlap in the tested row: left reads the tile touching his left edge (X - 8), right the tile touching his right edge (X + `$10`), and he stops flush (trace: X = `$40` right of a block at X `$30`-`$3F`, X = `$20` left of it). [`Tile_GetAheadOfObj`](src/bank7/tile-collision.asm#L65)
- His grid is 8 pixels, half a 16-pixel block, so he can stand half a block off the block columns; going up or down there both of his columns are still tested (trace: X = `$28` and X = `$38` under a block at X `$30`-`$3F` both stop at Y = `$B5`). [`Tile_GetAtObjOffset@FetchTile`](src/bank7/tile-collision.asm#L144)
- Link is drawn 2 pixels below `ObjY` where `CurLevel` = 0 and in cellars, at `ObjY` in dungeon rooms: the visible overlap is 9 pixels in the overworld, 11 in a dungeon. [`Link_EndMoveAndAnimate@Animate`](src/bank7/link-animate-ladder.asm#L298)

**Outside the play area**
- The row is not range checked. A hotspot Y below `$40` gives rows 24-31, one from `$F0` up gives rows 22-23. A column has 22 bytes, so these read rows 2-9 and rows 0-1 of the next column to the right (a vertical test also reads the column after that). [`Tile_GetAtObjOffset@FetchTile`](src/bank7/tile-collision.asm#L144)
- From column `$1F` such reads fall past the tile map, at `$67F0` and up. [`Tile_PlayAreaColumnAddrs`](src/bank7/run-game-nmi.asm#L54)
- The sums are 8 bits: the hotspot Y wraps for `ObjY` >= `$F5`, and so do the look-ahead offsets. [`Tile_GetAtObjOffset`](src/bank7/tile-collision.asm#L97)
- What this does to a Link who has passed the top or bottom screen edge is under `# Screen wrap (overworld)`. [`Link_CheckScreenEdge`](src/bank7/walker-collision-screen-edge.asm#L243)

**Link versus monsters**
- Link blocked on a grid point: his moving direction and `ButtonsPressed` are zeroed for the frame and he keeps facing the tile; in the overworld the armos and gravestone touch runs and the screen edge test is still made. [`Link_StopAtUnwalkableTile`](src/bank7/walker-collision-screen-edge.asm#L167)
- A monster blocked on a grid point tries, in order, a random direction perpendicular to its facing, the opposite of that, the reverse of its facing, and then does not move this frame. [`Walker_GetNextAltDir_JumpTable`](src/bank7/walker-collision-screen-edge.asm#L322) [`Walker_TryNextAltDir`](src/bank7/walker-collision-screen-edge.asm#L158)
- A monster with a walkable tile ahead must also pass the room bounds: left refused at X < `$11` (dungeon `$21`), right at X >= `$E0` (`$D0`), up at Y < `$4E` (`$5E`), down at Y >= `$CD` (`$BD`). [`Walker_CheckRoomBound`](src/bank7/walker-collision-screen-edge.asm#L298) [`Room_BoundDirHorizontally`](src/bank1/room-bounds.asm#L116) [`Room_BoundDirVertically`](src/bank1/room-bounds.asm#L185)
- Link is held by those bounds only in a dungeon room outside doorways, and there on every frame, not just on grid points, unless a laid ladder restores the move after the bound test (`# Grid-point skip: block clip, ladder clip and their run-up`, "Ladder clip"); in the overworld, caves and cellars only tiles and the screen edge test limit him. [`Walker_Move@CheckBoundary`](src/bank7/walker-move.asm#L185) [`Ladder_CheckAndDraw@HandleInput`](src/bank5/ladder.asm#L125)
- In a dungeon doorway (`DoorwayDir` nonzero) Link's tile test is skipped. [`Walker_CheckTileCollision`](src/bank7/walker-collision-screen-edge.asm#L89)
- Monsters move on a `$10`-pixel grid and test one tile (two when vertical) of the cell they are about to enter: up and left look `$10` back from the hotspot where Link looks 8. [`Tile_GetAheadOfObj`](src/bank7/tile-collision.asm#L65)
- A monster neither moves nor tests while the clock is held or its stun timer runs. [`Walker_Move@CheckStunned`](src/bank7/walker-move.asm#L87)
- Two blockers of Link are not tiles: a sliding push block, rock or gravestone within `$10` pixels, and the line Y < `$8E` in front of a dungeon person. [`Link_CheckSlidingTileObjBlocking`](src/bank1/tile-block-power-triforce.asm#L45) [`Person_BlockLinkMovingUp`](src/bank1/modeb-cave-save-sprites.asm#L289)

# Screen edges and room transitions

A room change by walking starts in one place: the screen edge test that follows Link's tile test. It runs once a
frame, before the move, on a grid point, and compares for equality.

**The bounds**
- Link's screen edge coordinates: `ObjY` = `$3D` (up), `$DD` (down); `ObjX` = `$00` (left), `$F0` (right). All four are grid points. [`Link_ScreenEdgeBounds`](src/bank7/walker-collision-screen-edge.asm#L70)
- The coordinate must EQUAL the bound (`CMP` / `BNE`). A Link beyond a bound, or one who passes over it between two frames, does not change rooms. [`Link_CheckScreenEdge`](src/bank7/walker-collision-screen-edge.asm#L243)

**Conditions of the test (all required)**
- `GameMode` = 5 (play). [`Walker_AcceptMovingDir`](src/bank7/walker-collision-screen-edge.asm#L200)
- No ladder laid down (`LadderSlot` = 0). [`Walker_AcceptMovingDir`](src/bank7/walker-collision-screen-edge.asm#L200) [`LadderSlot`](src/Variables.inc#L104)
- `ObjGridOffset` = 0 at the start of the frame: outside a doorway the tile test returns before the edge test when the offset is nonzero, and the edge test checks it again. [`Walker_CheckTileCollision@CheckGridOffset`](src/bank7/walker-collision-screen-edge.asm#L101) [`Walker_AcceptMovingDir`](src/bank7/walker-collision-screen-edge.asm#L200)
- Outside a doorway, a moving direction on arrival: standing still, using the sword or an item (states `$10`, `$20`), or stopped by a sliding tile object or a person, Link is not tested. [`Walker_CheckTileCollision`](src/bank7/walker-collision-screen-edge.asm#L89) [`Walker_Move@SetMovingDir`](src/bank7/walker-move.asm#L104)
- A held direction in `ObjInputDir`; the direction tested is its lowest set bit (right, left, down, up), not `ObjDir` and not the moving direction. [`Link_CheckScreenEdge`](src/bank7/walker-collision-screen-edge.asm#L243) [`Dir_GetOpposite`](src/bank1/qspeed-direction-math.asm#L137)
- `ObjY` is compared for up and down, `ObjX` for left and right; the other coordinate is not looked at. [`Link_CheckScreenEdge`](src/bank7/walker-collision-screen-edge.asm#L243)
- Overworld: an unwalkable tile ahead does not prevent the test; a Link standing on a bound leaves even when the lookup says blocked. Dungeon: it does, but in a doorway there is no tile test. [`Link_StopAtUnwalkableTile`](src/bank7/walker-collision-screen-edge.asm#L167) [`Walker_CheckTileCollision`](src/bank7/walker-collision-screen-edge.asm#L89)
- The test precedes the move: the room change starts on the frame after the one that put Link on the bound, and he does not move on it. [`Walker_Move@CheckTiles`](src/bank7/walker-move.asm#L206) [`Mode5Play_GoToNextMode`](src/bank7/walker-collision-screen-edge.asm#L281)

**What starts**
- `ObjDir` = the single input direction; `GameMode` + 1 = 6 (leave room) with `GameSubmode` and `IsUpdatingMode` = 0; Link's `ObjState`, knockback and invincibility timer are cleared. [`Link_CheckScreenEdge`](src/bank7/walker-collision-screen-edge.asm#L243) [`Mode5Play_GoToNextMode`](src/bank7/walker-collision-screen-edge.asm#L281)
- Mode 6 init: Link's state and the clock (`InvClock`) are zeroed, the room's kill count is banked, the six weapon slots and the ladder are cleared. [`Mode6Leave_Init`](src/bank5/mode3-mode6-init.asm#L214) [`Link_ResetStateAndClock`](src/bank7/init-mode-dispatch.asm#L354) [`Obj_ResetWeaponSlotsAndLadder`](src/bank5/mode3-mode6-init.asm#L263)
- Mode 6 update: in the overworld the walk distance is 0 and mode 7 (scroll) follows at once; through a dungeon false wall or bombed wall Link first walks `$20` pixels. [`Mode4Enter_UpdateEnterOrLeave`](src/bank7/update-mode-dispatch.asm#L197) [`Mode6Leave_LinkGridOffsets`](src/bank5/mode3-mode6-init.asm#L197)
- Next room: computed once per scroll, in mode 7 init step 1, from Link's facing `ObjDir` (not his position). [`Mode7Scroll_Init_Sub1`](src/bank5/mode7-scroll-init.asm#L77) [`Room_CalcNext`](src/bank7/next-room-room-flags.asm#L72)
- `NextRoomId` = `RoomId` + offset: up -`$10`, down +`$10`, left -1, right +1, in 8 bits with no column test: left from column 0 gives the last room of the row above, right from column `$F` the first of the row below. [`Room_NextRoomIdOffsets`](src/bank5/next-room-calc.asm#L36) [`Room_CalcNextRoomForDoor`](src/bank5/next-room-calc.asm#L76)
- Overworld: every side counts as open, and the mazes may replace the result (rooms `$61`, `$1B`). [`Room_CalcNext_OW`](src/bank7/next-room-room-flags.asm#L102) [`Maze_CheckStep`](src/bank1/mazes-lost-woods-hills.asm#L56)
- Dungeon: the door type on the side Link faces decides; types 1 (wall) and 8 (no side matched) compute nothing and clear `NextRoomDir`. [`Room_CalcNext_JumpTable`](src/bank7/next-room-room-flags.asm#L90) [`Room_CalcNoNextRoom`](src/bank5/next-room-calc.asm#L53)
- Invalid room: a result with bit 7 set (up from row 0, down from row 7, left of `$00`, right of `$7F`) abandons the scroll and loads the overworld: mode 2, `CurLevel` = 0, `UndergroundExitType` = 2. Walking out of a dungeon's entrance works this way. [`Room_CalcNextRoomForDoor`](src/bank5/next-room-calc.asm#L76) [`Mode_LeaveLevelToOW`](src/bank5/next-room-calc.asm#L107)
- `RoomId` becomes `NextRoomId` when the mode 7 init ends. [`Mode7Scroll_Init_Finish`](src/bank5/mode7-scroll-init.asm#L333)
- The scroll carries Link to the opposite edge: he arrives at X = `$F0` (going left) or `$00` (right), Y = `$DD` (up) or `$3D` (down), the other coordinate unchanged (trace). [`Mode7Scroll_ScrollWorld`](src/bank5/scroll-world.asm#L46) [`Mode7Scroll_ScrollWorldH`](src/bank5/scroll-world.asm#L154)
- Mode 4 then enters the room: RAM `$0300`-`$051F` is zeroed, so `ObjPosFrac` and `ObjGridOffset` start at 0. In the overworld Link stays where the scroll left him; in a dungeon he is put at X = `$00` or `$F0` and walks in `$18` or `$28` pixels. [`Room_EnterCommon`](src/bank5/enter-room.asm#L143) [`Room_EnterCommon@Method2`](src/bank5/enter-room.asm#L211) [`Mode4Enter_WalkInGridOffsets`](src/bank5/mode-inits-obj-lists.asm#L149)
- Overworld timing (trace): counted from the frame the test fires, `GameMode` reads 6 for 2 frames, 7 for 98 (left, right), 79 (up) or 77 (down), 4 for 4, and is 5 again after 104, 85 or 83 frames. [`Mode7Scroll_ScrollWorld`](src/bank5/scroll-world.asm#L46)

**Border bands that mask buttons**
- The mask runs once a frame before the A and B tests, only while Link's state is 0, and looks only at the coordinate on his facing axis. [`Link_FilterInput`](src/bank5/border-input-mask.asm#L54) [`Link_HandleInput`](src/bank5/link-handle-input.asm#L68)
- Inner band, both worlds, Link facing the edge: facing down Y >= `$BE`, up Y < `$54`, right X >= `$D1`, left X < `$1F`. `ButtonsPressed` keeps only A (`$80`): the sword starts, the B item does not. [`Link_BorderBounds`](src/bank5/border-input-mask.asm#L38) [`Link_FilterInput`](src/bank5/border-input-mask.asm#L54)
- Outer band: overworld down Y >= `$D6`, up Y < `$45`, right X >= `$E9`, left X < `$07`; dungeon down Y >= `$C6`, up Y < `$55`, right X >= `$D9`, left X < `$17`. `ButtonsPressed` is cleared: no sword either. [`Link_BorderBounds`](src/bank5/border-input-mask.asm#L38) [`Link_MaskInputInBorder@MaskButtons`](src/bank5/border-input-mask.asm#L133)
- The outer band is tested on the side Link faces when the inner test hit, otherwise on the side directly behind him (a Link who has just come in through that edge); facing along an edge masks nothing. [`Link_FilterInput`](src/bank5/border-input-mask.asm#L54)
- Dungeon only, outer band: `ObjInputDir` is also ANDed with the facing axis, so a held perpendicular direction is dropped. The overworld never changes `ObjInputDir` here. [`Link_MaskInputInBorder@MaskButtons`](src/bank5/border-input-mask.asm#L133)
- Only the buttons newly pressed this frame on controller 1 are affected; a sword or item already in use goes on. [`Link_MaskInputInBorder`](src/bank5/border-input-mask.asm#L119)
- Facing up in a dungeon at Y = `$54` nothing on the facing side is masked: there the outer bound (`$55`) lies outside the inner one (`$54`). [`Link_FilterInput`](src/bank5/border-input-mask.asm#L54)

**Dungeon edges**
- In a dungeon room the room bounds keep Link within X `$20`..`$D0`, Y `$5D`..`$BD`, except where a laid ladder carries him past them (next bullet); he reaches a screen edge only through a doorway (X = `$78` for up and down, Y = `$8D` for left and right), where the bounds and the tile test are skipped. [`Walker_Move@CheckBoundary`](src/bank7/walker-move.asm#L185) [`Doorway_CheckLink`](src/bank5/doorway.asm#L137) [`Doorway_RequiredCoords`](src/bank5/doorway.asm#L93)
- A laid ladder's case A rewrites the moving direction after the bound test: after the ladder clip north in level 4 room `$21` Link stands at Y = `$55`, 8 pixels past the top bound, with no ladder out, and walks along Y = `$55` to X = `$20` (trace). [`Walker_Move@CheckBoundary`](src/bank7/walker-move.asm#L185) [`Ladder_CheckAndDraw@HandleInput`](src/bank5/ladder.asm#L125)
- A false wall or a bombed wall (door types 2, 3, 4) starts the room change the moment Link passes it, without the edge test. [`Doorway_CheckLink`](src/bank5/doorway.asm#L137)

**Cave edge exit**
- In a cave (mode `$B`) the edge test is called directly on every frame of Link's move routine: no mode 5, ladder or grid offset condition. In the shortcut cave (mode `$C`) it is called only on a grid point with Y <> `$9D`. [`Warp_CheckCaveEdgeExit`](src/bank5/cave-cellar-exits.asm#L209) [`Warp_CheckSubroomExit`](src/bank5/cave-cellar-exits.asm#L43)
- A hit there becomes mode `$A` (leave cave) with `RoomId` unchanged. [`ModeALeaveCellar_GoFromCave`](src/bank5/cave-cellar-exits.asm#L178)
- Inside a cave Link cannot move up while Y < `$8E`, person or no person. [`Warp_CheckSubroomExit@InCave`](src/bank5/cave-cellar-exits.asm#L104) [`Person_BlockLinkMovingUp`](src/bank1/modeb-cave-save-sprites.asm#L289)
- Cave of room `$77` (trace): Link starts at (`$70`, `$D5`); holding Down he is on Y = `$DD` after 5 frames and mode `$A` starts on the next; after 34 frames of mode `$A` and 62 of mode 4 (the rise out of the entrance) he plays at (`$40`, `$4D`) in room `$77`. [`Warp_CheckCaveEdgeExit`](src/bank5/cave-cellar-exits.asm#L209) [`Mode4Enter_StepOutside`](src/bank7/update-mode-dispatch.asm#L247)
- A cellar (mode 9) is left differently: Y < `$40` with Up held. [`Warp_CheckSubroomExit`](src/bank5/cave-cellar-exits.asm#L43)

# Screen wrap (overworld)

In the overworld Link can walk past a screen edge without a room change; his 8-bit coordinate wraps and he
comes in at the opposite side of the same room. It is one case of a general fault: one frame of perpendicular
input lets Link cross a grid point without the tests that are made there. Everything marked (trace) was
reproduced frame by frame.

**Code path**
- Per frame: direction rules, then the tests and the move, then the offset reset, then the grid snap. [`Link_Update`](src/bank7/mode5-play-update.asm#L330) [`Link_HandleInput`](src/bank5/link-handle-input.asm#L68) [`Walker_Move`](src/bank7/walker-move.asm#L64)
- On a grid point one held direction becomes `ObjDir` and `ObjInputDir`. [`Link_ModifyDirAtGridPoint_HaveInput`](src/bank5/link-handle-input.asm#L157)
- Between grid points Link moves in `ObjDir` whatever is held. [`Walker_Move@ChooseObjDirOrInputDir`](src/bank7/walker-move.asm#L70)
- Between grid points one held direction perpendicular to `ObjDir`, with `Link_GoStraight` = 0, |`ObjGridOffset`| < 4 and the offset's sign agreeing with the heading, reverses `ObjDir` and rewrites the offset for the new heading: -3 becomes +5, +3 becomes -5 (negative n -> 8 + n, positive n -> n - 8). Link steps back that frame. [`Link_ModifyDirOnGridLine`](src/bank5/link-handle-input.asm#L338) [`Link_SetInputDir@ReverseDir`](src/bank5/link-handle-input.asm#L404)
- Next frame the original direction is the opposite of `ObjDir`: `ObjDir` and `ObjInputDir` are set to it and the offset is NOT rewritten. Its sign is now opposite to the heading, so it counts toward 0 at the grid point ahead instead of toward +-8. [`Link_ModifyDirOnGridLine`](src/bank5/link-handle-input.asm#L338) [`Link_SetDirAndInputDir`](src/bank5/link-handle-input.asm#L365)
- The move applies q-speed `$60` to the fraction 4 times a frame: 2 pixels when the fraction is below `$80` going left or up, or at least `$80` going right or down; otherwise 1. The fraction changes by `$80` every moving frame. [`Obj_MoveByQSpeed@ApplyQSpeedToPosition`](src/bank7/walker-move.asm#L262) [`Obj_SubQSpeedFromPosFrac`](src/bank1/qspeed-direction-math.asm#L94) [`Obj_AddQSpeedToPosFrac`](src/bank1/qspeed-direction-math.asm#L52)
- A pixel is withheld only at offset +8 or -8; nothing stops or snaps Link at 0, and the offset is reset to 0 only when it is a nonzero multiple of 8 after the move. [`Obj_SubQSpeedFromPosFrac@SetCarry`](src/bank1/qspeed-direction-math.asm#L108) [`Link_EndMoveAndAnimate@TruncGridOffset`](src/bank7/link-animate-ladder.asm#L177)
- The tile test and the screen edge test are made only on a frame that starts with offset 0, and the edge test needs equality with the bound. [`Walker_CheckTileCollision@CheckGridOffset`](src/bank7/walker-collision-screen-edge.asm#L101) [`Link_CheckScreenEdge`](src/bank7/walker-collision-screen-edge.asm#L243)
- The overworld's border mask clears buttons only and leaves `ObjInputDir` alone, so the perpendicular press reaches the direction rules; and no room bound is applied to Link there. [`Link_MaskInputInBorder@MaskButtons`](src/bank5/border-input-mask.asm#L133) [`Walker_Move@CheckBoundary`](src/bank7/walker-move.asm#L185)
- Positions are 8 bits: `$00` - 1 = `$FF`, `$FF` + 1 = `$00`. [`Obj_MoveByQSpeed`](src/bank7/walker-move.asm#L244)

**Why the edge test is missed**
- Normal walk: the offset counts away from 0 and the limit stops Link at +-8, exactly on the bound; the offset becomes 0 and the next frame's test finds him there. [`Obj_SubQSpeedFromPosFrac@SetCarry`](src/bank1/qspeed-direction-math.asm#L108) [`Link_EndMoveAndAnimate@TruncGridOffset`](src/bank7/link-animate-ladder.asm#L177)
- After reverse-then-forward the offset is +-1 one pixel short of the bound. If that frame moves 2 pixels it ends at -+1, one pixel past the bound: no frame starts on the bound with offset 0 and the equality never holds. [`Link_CheckScreenEdge`](src/bank7/walker-collision-screen-edge.asm#L243)
- Link walks on; 8 pixels past the bound the offset reaches -+8 and is reset, and from there he is an ordinary walker at a position no bound matches. [`Link_EndMoveAndAnimate@TruncGridOffset`](src/bank7/link-animate-ladder.asm#L177)

**The rule (all four edges, trace)**
- Where: overworld, play mode, no ladder laid; Link walking straight at an open screen edge with only that direction held (a diagonal at the last grid point sets `Link_GoStraight` and the press is ignored). The run-up, the recovery from a wrong cadence and the sword's extra position are the same for every grid point and are under `# Grid-point skip: block clip, ladder clip and their run-up`. [`Link_ModifyDirAtGridPoint_HaveInput`](src/bank5/link-handle-input.asm#L157) [`Link_GoStraight`](src/Variables.inc#L93)
- Run-up: Link must leave the last grid point before the bound (X = `$08` or `$E8`, Y = `$45` or `$D5`) with a 1-pixel frame, i.e. with the fraction `$80` there going left or up, `$00` going right or down; a full cell walked toward the edge into that grid point always leaves it so. [`Obj_SubQSpeedFromPosFrac`](src/bank1/qspeed-direction-math.asm#L94) [`ObjPosFrac`](src/Variables.inc#L316)
- Position: two frames after that grid point Link is 3 pixels past it, 5 pixels short of the bound: left X = `$05` (offset `$FD`, fraction `$80`); right X = `$EB` (`$03`, `$00`); up Y = `$42` (`$FD`, `$80`); down Y = `$D8` (`$03`, `$00`). [`Link_ScreenEdgeBounds`](src/bank7/walker-collision-screen-edge.asm#L70) [`ObjGridOffset`](src/Variables.inc#L311)
- Input: for exactly the next frame hold a perpendicular direction, either one. At a left or right edge release the edge direction for that frame (held together, the horizontal bit wins); at the top or bottom Up or Down may stay held (trace: Up+Left wraps at the top; Up+Left does not at the left). [`Dir_GetOpposite`](src/bank1/qspeed-direction-math.asm#L137) [`Link_ModifyDirOnGridLine`](src/bank5/link-handle-input.asm#L338)
- Then hold the edge direction; idle frames before it change nothing (trace: 1 and 5). On the fifth such frame the offset steps from +-1 to -+1 across the bound. [`Link_SetDirAndInputDir`](src/bank5/link-handle-input.asm#L365)
- Without the sword no other position works: from offset +-1 the step back ends on the grid point behind (offset reset); in the other cadence (2 pixels first) the offsets are +-2 and +-3 and both runs land exactly on 0 at the bound (trace: every frame of the last 19 or more before the room change, both perpendicular directions, on two rooms per edge: left `$77`, `$38`; right `$77`, `$37`; top `$77`, `$3D`; bottom `$67`, `$3D`: only the position above passes the bound). With the sword the left edge also wraps from X = `$07` (offset `$FF`, fraction `$00`) with Up+A; at the other three edges that position (X = `$E9`, Y = `$44`, Y = `$D6`) lies in the outer border band, where A is cleared (trace). [`Link_SetInputDir@ReverseDir`](src/bank5/link-handle-input.asm#L404) [`Link_MaskInputInBorder@MaskButtons`](src/bank5/border-input-mask.asm#L133)
- Reproduced (trace): all four edges, either perpendicular direction: left and right in rooms `$77`, `$38`, `$37`, `$78`, right also in `$60`; top in `$77`, `$3D`, `$5D`; bottom in `$67`, `$3D`. Not reproduced: a wrap from any other position or cadence, a second skip in a row (`# Grid-point skip: block clip, ladder clip and their run-up`, "No second skip in a row"), any wrap from a dungeon doorway (details below). [`Link_CheckScreenEdge`](src/bank7/walker-collision-screen-edge.asm#L243)

**Frame table** (trace): left edge of room `$77` on row Y = `$8D`, Left held except where shown; values after each frame. `$84` stays `$8D`; `$EB` stays `$77` in the wrap run. From rest at (`$10`, `$8D`) with `$3A8` = `$80` the same rows follow 6 frames of Left (X `$0F`, `$0D`, `$0C`, `$0A`, `$09`, `$08`; offsets `$FF`, `$FD`, `$FC`, `$FA`, `$F9`, `$00`), so Up comes on the 9th frame (trace).

| frame | normal: input | `$70` X | `$98` dir | `$394` offset | `$3A8` fraction | `$3F8` input | `$12` mode | wrap: input | `$70` X | `$98` dir | `$394` offset | `$3A8` fraction | `$3F8` input | `$12` mode |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Left | `$08` | 2 | `$00` | `$80` | 2 | 5 | Left | `$08` | 2 | `$00` | `$80` | 2 | 5 |
| 1 | Left | `$07` | 2 | `$FF` | `$00` | 2 | 5 | Left | `$07` | 2 | `$FF` | `$00` | 2 | 5 |
| 2 | Left | `$05` | 2 | `$FD` | `$80` | 2 | 5 | Left | `$05` | 2 | `$FD` | `$80` | 2 | 5 |
| 3 | Left | `$04` | 2 | `$FC` | `$00` | 2 | 5 | Up | `$07` | 1 | `$07` | `$00` | 8 | 5 |
| 4 | Left | `$02` | 2 | `$FA` | `$80` | 2 | 5 | Left | `$05` | 2 | `$05` | `$80` | 2 | 5 |
| 5 | Left | `$01` | 2 | `$F9` | `$00` | 2 | 5 | Left | `$04` | 2 | `$04` | `$00` | 2 | 5 |
| 6 | Left | `$00` | 2 | `$00` | `$80` | 2 | 5 | Left | `$02` | 2 | `$02` | `$80` | 2 | 5 |
| 7 | Left | `$00` | 2 | `$00` | `$80` | 2 | 6 | Left | `$01` | 2 | `$01` | `$00` | 2 | 5 |
| 8 | Left | `$00` | 2 | `$00` | `$80` | 2 | 6 | Left | `$FF` | 2 | `$FF` | `$80` | 2 | 5 |
| 9-12 | Left | `$00` | 2 | `$00` | `$80` | 2 | 7 | Left | `$FE`, `$FC`, `$FB`, `$F9` | 2 | `$FE`, `$FC`, `$FB`, `$F9` | `$00`, `$80`, `$00`, `$80` | 2 | 5 |
| 13 | Left | `$00` | 2 | `$00` | `$80` | 2 | 7 | Left | `$F8` | 2 | `$00` | `$00` | 2 | 5 |
| 18 | Left | `$00` | 2 | `$00` | `$80` | 2 | 7 | Left | `$F0` | 2 | `$00` | `$80` | 2 | 5 |
| 111 | (ignored) | `$F0` | 2 | `$00` | `$00` | 0 | 5, room `$76` | | | | | | | |

**Frame table, right edge** (trace): room `$77` on row Y = `$8D`, from rest at (`$E0`, `$8D`) with `$3A8` = `$00`, Right held except where shown; values after each frame; `$12` stays 5 and `$EB` stays `$77`.

| frame | input | `$70` X | `$98` dir | `$394` offset | `$3A8` fraction | `$3F8` input |
|---|---|---|---|---|---|---|
| 0 | - | `$E0` | 1 | `$00` | `$00` | 1 |
| 1-5 | Right | `$E1`, `$E3`, `$E4`, `$E6`, `$E7` | 1 | `$01`, `$03`, `$04`, `$06`, `$07` | `$80`, `$00`, `$80`, `$00`, `$80` | 1 |
| 6 | Right | `$E8` | 1 | `$00` | `$00` | 1 |
| 7 | Right | `$E9` | 1 | `$01` | `$80` | 1 |
| 8 | Right | `$EB` | 1 | `$03` | `$00` | 1 |
| 9 | Up | `$E9` | 2 | `$F9` | `$80` | 8 |
| 10 | Right | `$EB` | 1 | `$FB` | `$00` | 1 |
| 11-13 | Right | `$EC`, `$EE`, `$EF` | 1 | `$FC`, `$FE`, `$FF` | `$80`, `$00`, `$80` | 1 |
| 14 | Right | `$F1` | 1 | `$01` | `$00` | 1 |
| 19 | Right | `$F8` | 1 | `$00` | `$80` | 1 |
| 24 | Right | `$00` | 1 | `$00` | `$00` | 1 |

**After the wrap, edge by edge** (c = Link's column, X / 8)
- Left: Link comes in at X = `$FF`; the first grid point is X = `$F8`, where the tile ahead is column `$1E`, then `$F0`, `$E8` and so on as from any position. On an open row he walks left across the same room, and at X = `$F0` Right starts the ordinary change to the room on the right (trace: room `$77`, Y = `$8D`). [`Tile_GetAheadOfObj`](src/bank7/tile-collision.asm#L65) [`Link_CheckScreenEdge`](src/bank7/walker-collision-screen-edge.asm#L243)
- Left onto a row whose right edge is solid: he stops at X = `$F8` and is stuck: the tile ahead (column `$1E`) and the tile under him (column `$1F`; no look-ahead at X >= `$F0`) are solid, and `$F8` is not the right bound (trace: room `$38`, Y = `$8D`; no direction moved him). [`Tile_GetAtObjOffset@AdjustX`](src/bank7/tile-collision.asm#L123) [`Link_ScreenEdgeBounds`](src/bank7/walker-collision-screen-edge.asm#L70)
- Right: the grid point X = `$F8` reads the tile under him, then X = `$00` reads column 2 (X + `$10`). Columns 0 and 1, the 16-pixel square at the left edge, are never read (trace: room `$78`, rows Y = `$75` with columns 0-1 solid and Y = `$7D` with column 0 solid: he stands at X = `$00` inside them and walks on to X = `$10`). [`Tile_GetAtObjOffset@AdjustX`](src/bank7/tile-collision.asm#L123) [`Tile_GetAheadOfObj`](src/bank7/tile-collision.asm#L65)
- Right onto a row with two solid squares at the left (trace: room `$37`, Y = `$8D`): he stops at X = `$00` inside the wall; Right, Up and Down do nothing; Left fires the edge test, because X equals the bound and the overworld makes the test even with an unwalkable tile ahead. The room changes to `$36`, where he arrives at X = `$F0` inside solid tiles and cannot walk on: the arrival spot is never tested against the new room. Second trace: room `$60` (row 6, column 0), rows Y = `$85` and `$8D`: Left enters room `$5F`, the last room of the row above, at X = `$F0` standing on water (the ladder clip in `# Grid-point skip: block clip, ladder clip and their run-up`). [`Link_StopAtUnwalkableTile`](src/bank7/walker-collision-screen-edge.asm#L167) [`Room_EnterCommon@Method2`](src/bank5/enter-room.asm#L211)
- Up (the vertical variant): Link passes `$3D` (`$3E` to `$3C`) and reaches grid points `$35`, `$2D` ... `$05` over the status bar, then `$FD`, `$F5`, `$ED` and `$E5` at the bottom of the screen. [`Obj_MoveByQSpeed`](src/bank7/walker-move.asm#L244)
- He keeps moving there because nothing bounds him and the lookup has no row check: at the ten grid points `$35`, `$2D` ... `$05`, `$FD`, `$F5`, `$ED` it reads rows 9, 8 ... 0 of columns c + 1 and c + 2, tiles of the same room one column to the right. He stops where one of them is unwalkable (trace: room `$5D`, X = `$30`, a passage 2 tiles wide: stops at Y = `$25`, column 8 row 7 solid). [`Tile_GetAtObjOffset@FetchTile`](src/bank7/tile-collision.asm#L144)
- At Y = `$E5` the lookup is back inside the map: row 21, the bottom row, of his own columns. Solid: he stops at `$E5`, 8 pixels below the bottom bound; Down takes him back through `$ED`, `$F5` (trace: room `$77`, X = `$78`). Open: he walks up into the room from its bottom edge, a complete vertical wrap (trace: room `$3D`, X = `$70`, openings at columns `$0E`-`$11` top and bottom). [`Tile_GetAheadOfObj`](src/bank7/tile-collision.asm#L65)
- Down: Link passes `$DD` (`$DC` to `$DE`) to `$E5`. At `$E5` and `$ED` the hotspot Y is `$DD` or more, so there is no look-ahead: they read rows 0 and 1 of columns c + 1 and c + 2. From `$F5` the hotspot Y has wrapped and the look-ahead is back: `$F5` to `$25` read rows 3 to 9 of those columns (row 2 is never read), `$2D` and `$35` rows 0 and 1 of his own columns. [`Tile_GetAtObjOffset`](src/bank7/tile-collision.asm#L97)
- Down, traces: room `$67`, X = `$70`: stops at `$E5` (row 0 solid there); Up brings him back to `$DD`, where Down then changes rooms normally. Room `$3D`, X = `$70`: he goes round and comes down from the top edge. [`Tile_GetAtObjOffset@FetchTile`](src/bank7/tile-collision.asm#L144)

**Why it cannot happen in dungeons**
- A dungeon screen edge is reached only inside a doorway; everywhere else the room bounds hold Link, a laid ladder aside (`# Screen edges and room transitions`, "Dungeon edges"). [`Walker_Move@CheckBoundary`](src/bank7/walker-move.asm#L185) [`Doorway_CheckLink`](src/bank5/doorway.asm#L137)
- In a doorway an input that is only perpendicular is replaced by the facing direction, on every frame: Link keeps walking forward. [`Link_ModifyDirInDoorway`](src/bank5/doorway.asm#L52)
- In the dungeon's outer band (Y < `$55`, Y >= `$C6`, X < `$17`, X >= `$D9`, which covers the last cell before each bound), with state 0, the perpendicular bits are removed first and no input is left: Link stands still for that frame. [`Link_MaskInputInBorder@MaskButtons`](src/bank5/border-input-mask.asm#L133)
- Either way the reversal never runs (trace: level 2 room `$7D`, top doorway, Left or Right for one frame at every position from Y = `$5C` to `$3D`: at `$5C`-`$54` he keeps walking up with `ObjInputDir` = 8, at `$52`-`$3D` he pauses with `ObjInputDir` = 0; every run ends in the normal room change). [`Link_ModifyDirOnGridLine`](src/bank5/link-handle-input.asm#L338)
- The skip itself does work inside a dungeon room away from the doors and walls: the block clip and the ladder clip are under `# Grid-point skip: block clip, ladder clip and their run-up`. [`Link_ModifyDirOnGridLine`](src/bank5/link-handle-input.asm#L338)

# Grid-point skip: block clip, ladder clip and their run-up

One frame of perpendicular input, 5 pixels short of a grid point, lets Link cross that grid point with no frame
starting on it, so nothing that is tested only there is tested: the tile ahead, the screen edge, the warps, the stop
of a laid ladder. `# Screen wrap (overworld)` is this skip at a screen bound; this section is the skip itself, the
run-up into it, the recovery from a failed or spoiled try, and its uses on solid squares: the block clip and the
ladder clip. Names: s = Link's heading; T = the grid point whose tests are to be skipped; G = the last grid point
before T (8 pixels short of it); G' = the one before G (`$10` short); "good fraction" = `ObjPosFrac` `$80` heading up
or left, `$00` heading down or right (the value that makes the next frame a 1-pixel frame). Table columns are RAM
after the frame: `$70` `ObjX`, `$84` `ObjY`, `$98` `ObjDir`, `$394` `ObjGridOffset`, `$3A8` `ObjPosFrac`, `$3F8`
`ObjInputDir`, `$AC` `ObjState`, `$12` `GameMode`, `$EB` `RoomId`, `$64` `LadderSlot`. (trace) marks results
reproduced frame by frame with controller input only.

**The skip in one paragraph**

Link walks toward T holding only s. On the frame that starts 5 pixels short of T, 3 pixels past G, in the working
cadence, a held direction perpendicular to s is case 6 of the rules between grid points: `ObjDir` is reversed, the
offset is rewritten as seen from T (-3 becomes +5, +3 becomes -5) and Link steps back that frame (offset +-7 after
it). Holding s again is case 3: `ObjDir` turns back and the offset is kept, so with its sign now opposite to the
heading it counts toward 0 at T instead of toward +-8. On the 5th frame of s a frame starts at offset +-1 and moves
2 pixels: the offset steps to -+1 and no frame ever starts at offset 0 on T, so T's tile test, screen-edge test,
warp check and turning rules are not made. The offset runs on to -+8 at the grid point past T, is reset there, and
that grid point is tested as usual. One skip removes exactly one grid point's tests and carries Link one tile (8
pixels) past where he would have stopped, whatever made T a stop.
[`Link_SetInputDir@ReverseDir`](src/bank5/link-handle-input.asm#L404) [`Link_SetDirAndInputDir`](src/bank5/link-handle-input.asm#L365)
[`Walker_CheckTileCollision@CheckGridOffset`](src/bank7/walker-collision-screen-edge.asm#L101) [`Link_EndMoveAndAnimate@TruncGridOffset`](src/bank7/link-animate-ladder.asm#L177)

**Preconditions and the working position**
- Mode 5; Link not halted and not being knocked back; for the plain skip his state is 0 (no sword or item in use). Nothing else is needed: no item, key or button besides the d-pad, except a laid ladder for the ladder clip (trace). [`Link_HandleInput`](src/bank5/link-handle-input.asm#L68)
- Only s held when Link left G: a diagonal held on G sets `Link_GoStraight`, and the press is then case 4 and ignored. [`Link_ModifyDirAtGridPoint_HaveInput`](src/bank5/link-handle-input.asm#L157) [`Link_GoStraight`](src/Variables.inc#L93)
- Dungeon: not in a doorway, where the input is forced onto the facing axis, and not in the outer border band (`# Screen edges and room transitions`, "Border bands that mask buttons"), where a perpendicular bit is dropped at state 0. The overworld band clears buttons only and lets the press through. [`Link_ModifyDirInDoorway`](src/bank5/doorway.asm#L52) [`Link_MaskInputInBorder@MaskButtons`](src/bank5/border-input-mask.asm#L133)
- Trick state, read at the start of the press frame: heading up Y = T + 5, `ObjGridOffset` = `$FD`, `ObjPosFrac` = `$80`; left X = T + 5, `$FD`, `$80`; down Y = T - 5, `$03`, `$00`; right X = T - 5, `$03`, `$00` (trace, all four headings). It is the 3rd frame after the frame that ended on G, when Link left G with the good fraction. [`Link_SetInputDir@ReverseDir`](src/bank5/link-handle-input.asm#L404)
- T by obstacle: the screen bounds X `$00`, `$F0`, Y `$3D`, `$DD` (trick X `$05`, `$EB`, Y `$42`, `$D8`); below a 16-pixel solid square whose top is `by`, T = `by` + 5 (trick `by` + `$0A`); above it T = `by` - `$13`; right of a square at X `bx`..`bx` + `$0F` T = `bx` + `$10`, left of it T = `bx` - `$10`; on a laid ladder the distance-0 point: ladder X for a ladder laid left or right, ladder Y - 3 for one laid up or down. [`Tile_GetAheadOfObj`](src/bank7/tile-collision.asm#L65) [`Link_ScreenEdgeBounds`](src/bank7/walker-collision-screen-edge.asm#L70) [`Ladder_CheckAndDraw@CheckDistanceToLadder`](src/bank5/ladder.asm#L82)
- The press: exactly one frame whose held bits, reduced in the order right, left, down, up, give a direction perpendicular to s; either perpendicular direction works. Heading up or down: Left or Right, alone or with Up or Down (Up+Left reduces to Left). Heading left or right: Up or Down alone, s released for that frame, since Left+Up reduces to Left (trace: Left, Right and all four diagonals work heading up in level 4 room `$21`; no diagonal works heading left or right; Up+Left wraps at the top edge, not at the left). [`Dir_GetOpposite`](src/bank1/qspeed-direction-math.asm#L137)
- Then hold s. Idle frames between the press and s change nothing (trace: 1, 5 and 30). Link crosses T on the 5th frame of s. [`Link_SetDirAndInputDir`](src/bank5/link-handle-input.asm#L365)

**Cadence: why a run-up is needed**
- `ObjPosFrac` is one byte shared by both axes. With h = 2 x coordinate + 1 when `ObjPosFrac` >= `$80` (a position in half pixels), every moving frame changes h by exactly 3 (1.5 pixels): heading up or left 2 pixels when the fraction is `$00`, 1 when `$80`; heading down or right 1 when `$00`, 2 when `$80`; the fraction flips each moving frame. [`Obj_SubQSpeedFromPosFrac`](src/bank1/qspeed-direction-math.asm#L94) [`Obj_AddQSpeedToPosFrac`](src/bank1/qspeed-direction-math.asm#L52) [`Obj_MoveByQSpeed@ApplyQSpeedToPosition`](src/bank7/walker-move.asm#L262)
- The trick state is h = 2(T + 5) + 1 heading up or left, 2(T - 5) heading down or right. A straight walk that reaches no grid point on the way gets there iff its distance in h is a multiple of 3; the walk out of G gets there iff Link leaves G with the good fraction. [`Obj_SubQSpeedFromPosFrac`](src/bank1/qspeed-direction-math.asm#L94)
- What changes h by anything but 3 a frame: the withheld pixel at offset +-8 (a grid point reached by a full cell; the fraction still advances); room entry (`$0300`-`$051F` zeroed: fraction `$00`); the overworld slow tiles (fraction zeroed when the speed first drops); knockback (4 whole pixels a frame, fraction untouched). [`Obj_SubQSpeedFromPosFrac@SetCarry`](src/bank1/qspeed-direction-math.asm#L108) [`Room_EnterCommon`](src/bank5/enter-room.asm#L143) [`Link_InitSpeed`](src/bank5/link-handle-input.asm#L277) [`Shove_MovePixels`](src/bank7/shove.asm#L129)
- What does not: standing still; a turn at a grid point (the fraction is kept, but which value is good flips with the heading); a sword swing or B item (Link does not move); case 6 itself (it rewrites the offset only; its step back is an ordinary moving frame). [`Walker_Move@SetMovingDir`](src/bank7/walker-move.asm#L104) [`Link_SetInputDir@ReverseDir`](src/bank5/link-handle-input.asm#L404)
- So the cadence cannot be picked by timing (trace: reaching X = `$08` walking left, pausing, then Left wraps at `$05`; reaching it walking down, `$3A8` = `$00`, no frame wraps; a pause at X = `$07` or `$05` is harmless). [`ObjPosFrac`](src/Variables.inc#L316)
- Moves are exactly reversible: walking back retraces the same (position, fraction) pairs, and a plain turn-back lands exactly on the grid point left, with no withheld pixel. Walking back and forth inside a cell therefore never changes the cadence (trace: from rest on G with the wrong fraction, every Up/Down sequence of 1-8 frames that stays between G and T: the trick state is never reached). [`Link_ModifyDirOnGridLine`](src/bank5/link-handle-input.asm#L338)
- A full cell walked straight into a grid point always ends there with the good fraction for its own heading, whatever the fraction at its start: from the good fraction the cell is 1, 2, 1, 2, 1, 1 pixels in 6 frames (the last 2 cut by the limit), from the other 2, 1, 2, 1, 2 in 5 frames; both end on `$80` after a cell walked up or left, `$00` after one walked down or right (trace: from (`$8D`, `$00`) and (`$8D`, `$80`) Up arrives at (`$85`, `$80`)). [`Obj_AddQSpeedToPosFrac@ClearCarry`](src/bank1/qspeed-direction-math.asm#L66)
- So the fraction on G is fixed by how Link got there: a full cell walked into it up or left gives `$80`, down or right `$00`, on either axis (a cell walked left sets up's value, right down's); room entry `$00`; a partial walk or a turn-back whatever it was before (trace: room `$77`, every rest position X `$05`-`$18`, both fractions and both offset signs; level 1 room `$22`, arriving on (`$A0`, `$9D`) by a cell walked Left, then Up 2 and Right: clip; arriving on it walking Down from `$95`: no clip). [`ObjPosFrac`](src/Variables.inc#L316)
- Rule: the last full cell walked before the skip must have been up or left for a skip heading up or left, down or right for one heading down or right; a full cell walked away from T sets the wrong fraction. [`Obj_SubQSpeedFromPosFrac@SetCarry`](src/bank1/qspeed-direction-math.asm#L108)
- Minimum run-up: none when Link stands on G with the good fraction (s 2 frames, press on the 3rd); otherwise one full cell, G' to G, walked toward T. From rest on G' or farther back on the line, either fraction, holding s works (trace: room `$22`, every rest position on a grid point from Y `$A5` down to `$B5`; room `$77`, all such). [`Obj_SubQSpeedFromPosFrac@SetCarry`](src/bank1/qspeed-direction-math.asm#L108)
- From rest between G' and G: it works when Link left G' toward T (offset sign = heading); when he got there walking away from G he retraces onto G with the fraction he had there, so it works only if that was good (trace: room `$22` heading up, rest at Y = G + 1, + 3, + 4, + 6, + 7 with fraction `$80`, `$00`, `$80`, `$00`, `$80`, the states of a walk away from G left with `$00`, all fail; room `$21`, from 5 such states each the phase on G is kept). [`Obj_SubQSpeedFromPosFrac`](src/bank1/qspeed-direction-math.asm#L94)
- Between G and T only two states lead to the skip: the trick state and the frame before it (heading up Y = G - 1, `$FF`, `$00`), from which one more frame of s gives the trick state (trace: room `$22`, rest at `$9C` with `$00` reached walking up from G: 1 Up, then Right clips; rest at `$9C` with `$80`, `$9B` either, `$9A` with `$00`, `$96`-`$99` either: no clip). [`Link_SetInputDir@ReverseDir`](src/bank5/link-handle-input.asm#L404)
- The other positions fail by landing on a grid point: from G with the wrong fraction the frames out of G reach 2 and 3 pixels past G at the other fraction; a press 2 past G steps back 2 pixels exactly onto G (offset reset), a press 3 past G counts 5, 3, 2, 0 and lands exactly on T; a plain press 1 past G turns the rewritten +-7 into +-8 with its 1-pixel step back, reset on G (trace). [`Link_EndMoveAndAnimate@TruncGridOffset`](src/bank7/link-animate-ladder.asm#L177)
- Backing off: from G with the wrong fraction one cell back, to G', is enough; from T (stopped after a failed press) or from between G and T one cell back is not, because it ends on G walking away, with the fraction of the opposite heading (trace: room `$22` from T; room `$5F`, backing off only to X `$E8` gave no clip). The rule for both: walk away until Link stands on G' with offset 0, then walk toward T. Situations differ only in where G' lies: below a block `by` + `$15`; at the left screen edge X `$10`; on a ladder laid left or right, and at the arrival in room `$5F`, G' is the lay point itself (X `$50`/`$A0` in level 4 room `$21`, X `$F0` in `$5F`), so a Link who starts from the lay point always has the full cell; on a ladder laid up the lay point is G (Y `$85` in room `$21`) and G' is one cell below it (Y `$8D`). [`Obj_AddQSpeedToPosFrac`](src/bank1/qspeed-direction-math.asm#L52) [`Ladder_LinkOffsetsX`](src/bank7/link-animate-ladder.asm#L58)
- With a ladder out, the walk back toward the water after backing off costs one more frame: the first frame back only puts the old ladder away (Link does not move), and the ladder is laid again on arrival at the lay point (trace). [`Ladder_CheckAndDraw@StashLadder`](src/bank5/ladder.asm#L110)
- Frames from rest on a grid point n cells before G (n >= 1) to the press, holding s: 6(n - 1) + 5 + 2 when the first frame moves 2 pixels (fraction `$00` heading up or left, `$80` down or right), 6(n - 1) + 6 + 2 otherwise; the press is the frame after (trace, heading up: from Y `$A5` 7 or 8 frames, `$AD` 13 or 14, `$B5` 19 or 20). [`Obj_MoveByQSpeed`](src/bank7/walker-move.asm#L244)

**Recovery when too close or in the wrong cadence**
- On G with the wrong fraction: walk away 6 frames to G' (the offset resets to 0 there), then s 7 frames, press, s (trace: room `$22` from (`$A0`, `$9D`) with `$00`: Down 6, Up 7, Right 1; room `$77` from (`$08`, `$8D`) with `$00`: Right 6, Left 7, Up 1, Left 5: wrap). [`Link_EndMoveAndAnimate@TruncGridOffset`](src/bank7/link-animate-ladder.asm#L177)
- Stopped on T after a failed press: walk away two cells to G' (11 frames when Link reached T walking toward it, else 12), then s 7 frames, press, s (trace: room `$22` from (`$A0`, `$95`): Down 11, Up 7, Right 1, Up 10). [`Obj_AddQSpeedToPosFrac`](src/bank1/qspeed-direction-math.asm#L52)
- Anywhere between G and T but the trick state and the frame before it: release, walk away until on G' (passing back over G), then s 7 frames, press. [`Link_ModifyDirOnGridLine`](src/bank5/link-handle-input.asm#L338)
- Between G' and G after stepping back from G (offset sign opposite to s): s retraces onto G with the fraction Link had there; then apply the rule for G. [`Link_ModifyDirOnGridLine`](src/bank5/link-handle-input.asm#L338)
- With a ladder out add the re-laying frame (trace: room `$21` north from (`$85`, `$00`, `$00`): Down 6, Up 8, press: 15 frames instead of 3; room `$5F` after a miss: Right 11 frames to X `$F0`, Left 8 to `$E5`, Up 1, Left: clip). [`Ladder_CheckAndDraw@StashLadder`](src/bank5/ladder.asm#L110)
- Cheaper where the line beside is open: a full cell walked up or left on the other axis sets the good fraction for a skip heading up or left, down or right for one heading down or right (trace: room `$77` from G = (`$08`, `$8D`) with `$00`: Up 3, Left 4 (Link finishes the cell up to Y `$85` and turns left there), Up 1, Left 5: wrap on row `$85`, 13 frames instead of 19). [`Link_ModifyDirOnGridLine`](src/bank5/link-handle-input.asm#L338)
- Nothing short of the back-off rescues the wrong cadence: not the sword (trace: no sequence of 5 frames of Left, Up, nothing and A from G = (`$08`, `$8D`) with `$00` wrapped), not other presses (trace: room `$21` from the wrong phase at Y `$85`, 16,134 input scripts, every 3-frame window of Up, A, Up+A, Left, Left+A, nothing and every 4-frame window adding Down, after 0-7 Up frames: none passed). [`Link_SetInputDir@ReverseDir`](src/bank5/link-handle-input.asm#L404)
- Knockback from a monster's touch spoils the cadence; the same rules repair it (room `$21` with its Gels, under "Ladder clip"). [`Shove_Update`](src/bank7/shove.asm#L55)

**Decision table** (applied every frame, first matching row; c = Link's coordinate on the heading axis, o = `ObjGridOffset`, f = `ObjPosFrac`, read at the start of the frame; "press" = one frame of perpendicular, then s held)

| Link's state | input |
|---|---|
| on G' or farther back on the line (o = 0), any f | hold s; press on the frame that starts in the trick state |
| between G' and G, having left G' toward T (o negative heading up or left, positive heading down or right) | hold s; press in the trick state |
| between G' and G after stepping back from G (o of the other sign) | hold s: Link retraces onto G with the f he had there; then the rows for G |
| on G, good f | s 2 frames, press on the 3rd |
| between G and the trick state with h - h(trick) a positive multiple of 3 (G - 1 or G + 1 toward T, o = `$FF` with f = `$00` heading up or left, o = `$01` with f = `$80` heading down or right) | s 1 frame, then press |
| trick state (table below) | press now, then hold s |
| sword state (table below), sword owned, A not masked, no ladder out | A now (alone or with the perpendicular), the perpendicular on one of the next 12 frames, then s |
| on G with the other f | walk away until on G' (6 frames), then s 7 frames, press |
| on T, stopped | walk away to G' (11 or 12 frames), then s 7 frames, press |
| between G and T in any other state (6 pixels short of T, 5 short with the other f, nearer T) | lost: release, walk away until on G' (passing back over G), then s 7 frames, press |
| on G with the other f, the line beside open (alternative) | a full cell on the other axis, up or left for a skip heading up or left, down or right otherwise; then the rows for G on the new line |
| ladder out, walking back toward the water after a back-off | one more frame of s: the first only puts the old ladder away |

| heading | trick state (c, o, f) | sword state (c, o, f) | good f on G |
|---|---|---|---|
| up | T + 5, `$FD`, `$80` | T + 7, `$FF`, `$00` | `$80` |
| left | T + 5, `$FD`, `$80` | T + 7, `$FF`, `$00` | `$80` |
| down | T - 5, `$03`, `$00` | T - 7, `$01`, `$80` | `$00` |
| right | T - 5, `$03`, `$00` | T - 7, `$01`, `$80` | `$00` |

The table holds in all four directions (trace: room `$22` east block from all four sides, both perpendicular directions, plain and sword; level 2 room `$7D` statues at X `$30` and `$60` from all four sides; overworld room `$77` bottom wall heading down, top heading up, left rock row heading left; all four screen edges). Worked example, level 4 room `$21` heading up from the island, ladder laid at Y `$80` (T = `$7D`, G = `$85`, G' = `$8D`; trace: all 16 start states reached Y `$55`):

| Link (Y, `$394`, `$3A8`) | input |
|---|---|
| `$8D` > Y > `$85` with `$394` negative (walking up from `$8D`), or Y >= `$8D` | Up until (`$82`, `$FD`, `$80`), press |
| `$8D` > Y > `$85` with `$394` positive (left `$85` walking down) | Up: Link arrives on `$85` with the fraction he left it with; if `$80`, Up 2 more and press; else Down to `$8D` first |
| (`$8D`, `$00`, `$00`) / (`$8D`, `$00`, `$80`), no ladder out | Up 7 / Up 8, press |
| (`$85`, `$00`, `$80`) | Up 2, press |
| (`$84`, `$FF`, `$00`) | Up 1, press |
| (`$82`, `$FD`, `$80`) | press |
| (`$85`, `$00`, `$00`) | Down 6 to `$8D`, Up 8, press |
| (`$83`, `$FE`, `$80`) / (`$82`, `$FD`, `$00`) / (`$81`, `$FC`, `$00`) | Down 7 / 8 / 8, Up 8, press |
| (`$80`, `$FB`, `$80`) / (`$7F`, `$FA`, either) / (`$7E`, `$F9`, `$00`) | Down 9 / 9-10 / 10, Up 8, press |
| (`$7D`, `$00`, `$80`), stopped by the ladder | Down 11, Up 8, press |

**The sword's extra position**
- A newly pressed is handled in Link's update before the direction rules (state 0 only, after the border mask), and the direction rules still run during the swing; only the move is zeroed while (`ObjState` AND `$F0`) is `$10` or `$20`. [`Link_HandleInput`](src/bank5/link-handle-input.asm#L68) [`Walker_Move@SetMovingDir`](src/bank7/walker-move.asm#L104)
- A press during the swing is case 6 without the step back and without the fraction flip. From 1 pixel past G in the working cadence (the sword state: heading up or left c = T + 7, `$FF`, `$00`; down or right T - 7, `$01`, `$80`) Link stays at offset +-7 and the following s crosses T in the working cadence (trace: room `$22`, Up 13, A, Right, Up: (`$A0`, `$8D`); room `$77`, X = `$07` with Up+A: left wrap). [`Link_SetInputDir@ReverseDir`](src/bank5/link-handle-input.asm#L404)
- So with the sword the skip works from two consecutive frames of the approach (1 and 3 pixels past G), plain from one (3 past G); no other position works with or without it (trace: every frame of the approach from Y `$AD`, both cadences). [`Link_EndMoveAndAnimate@TruncGridOffset`](src/bank7/link-animate-ladder.asm#L177)
- Timing: A on the frame that starts 1 or 3 pixels past G (Link does not move on it; A with the perpendicular on the same frame works too), the perpendicular on any of the next 12 frames and held at most 12 frames, then s (trace: held 13 frames from 1 past G it carries Link back onto G). Link's move is zeroed on the A frame (state `$10`) and on the 12 frames after it (state `$11`, timed by the sword's states 1 and 2); then state `$31` lasts 1 frame and lets him move. [`Sword_Wield`](src/bank5/wield-items.asm#L59) [`Sword_UpdateSwordOrRod`](src/bank7/sword-and-rod.asm#L100) [`Link_AdvanceItemUseState`](src/bank7/object-animation.asm#L223)
- B items freeze Link the same way (state `$10`): bomb, arrow (bow, arrows and a rupee), boomerang and rod all skip from 1 pixel past G (trace). The candle freezes him too, but its fire lies on his path and burns him: knocked back, no skip (trace). B with nothing usable (0 bombs) does not freeze him. [`Link_WieldItem`](src/bank5/wield-items.asm#L95) [`Link_UseItem_SetState10`](src/bank1/bomb-candle-wield.asm#L143) [`Fire_Update`](src/bank7/fire.asm#L44)
- A must not be masked: facing the edge in the overworld's outer border band (X < `$07`, X >= `$E9`, Y < `$45`, Y >= `$D6`) A is cleared. The sword states at the screen bounds are X `$07` (left, allowed), X `$E9` (right), Y `$44` (top) and Y `$D6` (bottom), the last three masked: only the left edge has a sword position (trace: right, top and bottom edges did not wrap with the sword; at X `$E9` A did nothing and the room changed; the plain X `$EB` wraps). [`Link_FilterInput`](src/bank5/border-input-mask.asm#L54) [`Link_BorderBounds`](src/bank5/border-input-mask.asm#L38)
- On a laid ladder there is no sword position: the ladder code runs after the zeroing and puts a moving direction back (case C for the step back, case A forward), so Link moves during the swing and the press acts as without the sword (trace: room `$5F`, Up+A at X `$E7`: step back to `$E8`, stopped at `$E0`; room `$21`, every frame from Y `$86` to `$7D` with A on any of 3 frames around the press: only the plain trick state works). A alone on the trick frame is an idle frame there, and the press on the next frame still works (trace). [`Ladder_CheckAndDraw@HandleInput`](src/bank5/ladder.asm#L125)
- The sword is never needed: every recipe below is plain. [`Link_SetInputDir@ReverseDir`](src/bank5/link-handle-input.asm#L404)

**Block clip**
- Rule, heading up into a 16-pixel solid square whose top is `by`: T = `by` + 5, where the square stops Link; G = `by` + `$0D`; G' = `by` + `$15`. One skip takes him to Y = `by` - 3 = T - 8, half inside the square, where the next test reads its upper tile row and stops him (trace: east block of level 1 room `$22`, (`$A0`, `$95`) to (`$A0`, `$8D`): the press frame plus 10 frames of Up). [`Walker_CheckTileCollision@CheckGridOffset`](src/bank7/walker-collision-screen-edge.asm#L101)
- The way on is sideways: at Y = `by` - 3 his hotspot (Y + `$0B`) is in the square's lower tile row, and a left or right test reads only the one tile ahead in that row (X - 8 or X + `$10`), never the tile he stands in; where that tile is open he walks out sideways with his upper body overlapping the square. [`Tile_GetAtObjOffset@FetchTile`](src/bank7/tile-collision.asm#L144)
- From the side the clip leads nowhere: vertical tests read two columns (X / 8 and X / 8 + 1), so a Link stopped 8 pixels into a square from the side (X = `bx` + 8 or `bx` - 8) still reads a block column going up or down, and the square's second column going on (trace: from the right of the east block of room `$22` he stops at (`$A8`, `$8D`): Left and Up blocked; only Right and Down move). [`Tile_GetAtObjOffset@FetchTile`](src/bank7/tile-collision.asm#L144)
- Room `$22` (level 1, layout `$1A`): blocks at (`$80`,`$70`), (`$70`,`$80`), (`$90`,`$80`), (`$60`,`$90`) (the push block), (`$A0`,`$90`), (`$70`,`$A0`), (`$90`,`$A0`), (`$80`,`$B0`); stairs at (`$80`,`$90`), taken at (`$80`, `$8D`); inside squares (`$80`,`$80`), (`$70`,`$90`), (`$90`,`$90`), (`$80`,`$A0`). [`Warp_CheckUnderLink`](src/bank5/warps-stairs-caves.asm#L55)
- Room `$22` recipe (Link's state 0, no knockback, only Up held when he leaves G, Link on the block's column; nothing in the inventory): from (`$E0`, `$8D`) in the E doorway Left to X `$B0` (36 frames), Down to Y `$AD` (23), Left to X `$A0` (11), Up 14, Right 1, Up 10, Left 22: Link is on the stairs (mode `$10`, then 9, cellar `$7F`), 117 frames, no damage (trace). [`Link_HandleInput`](src/bank5/link-handle-input.asm#L68)
- Into the diamond: from (`$A0`, `$8D`) the left test reads (`$98`, `$98`), the inside square (`$90`, `$90`), so Left walks him in; on X = `$80` the stairs take him (trace: 22 frames of Left from the clip). [`Tile_GetAtObjOffset@FetchTile`](src/bank7/tile-collision.asm#L144) [`Warp_CheckUnderLink`](src/bank5/warps-stairs-caves.asm#L55)

**Frame table, room `$22` east block** (trace): Link at rest on (`$A0`, `$AD`) after walking Left to X `$A0`; values after each frame; X = `$A0` until frame 26; `$AC` = 0 throughout.

| frame | input | `$84` Y | `$98` | `$394` | `$3A8` | `$3F8` |
|---|---|---|---|---|---|---|
| 0 | - | `$AD` | 2 | `$00` | `$80` | 2 |
| 1-5 | Up | `$AC`, `$AA`, `$A9`, `$A7`, `$A6` | 8 | `$FF`, `$FD`, `$FC`, `$FA`, `$F9` | `$00`, `$80`, `$00`, `$80`, `$00` | 8 |
| 6 | Up | `$A5` | 8 | `$00` | `$80` | 8 |
| 7-11 | Up | `$A4`, `$A2`, `$A1`, `$9F`, `$9E` | 8 | `$FF`, `$FD`, `$FC`, `$FA`, `$F9` | `$00`, `$80`, `$00`, `$80`, `$00` | 8 |
| 12 | Up | `$9D` (G) | 8 | `$00` | `$80` | 8 |
| 13 | Up | `$9C` | 8 | `$FF` | `$00` | 8 |
| 14 | Up | `$9A` | 8 | `$FD` | `$80` | 8 |
| 15 | Right | `$9C` | 4 | `$07` | `$00` | 1 |
| 16 | Up | `$9A` | 8 | `$05` | `$80` | 8 |
| 17 | Up | `$99` | 8 | `$04` | `$00` | 8 |
| 18 | Up | `$97` | 8 | `$02` | `$80` | 8 |
| 19 | Up | `$96` | 8 | `$01` | `$00` | 8 |
| 20 | Up | `$94` (T = `$95` passed) | 8 | `$FF` | `$80` | 8 |
| 21-24 | Up | `$93`, `$91`, `$90`, `$8E` | 8 | `$FE`, `$FC`, `$FB`, `$F9` | `$00`, `$80`, `$00`, `$80` | 8 |
| 25 | Up | `$8D` | 8 | `$00` | `$00` | 8 |
| 26 | Left | `$8D` (X `$9E`) | 2 | `$FE` | `$80` | 2 |

- Frame by frame: frame 15 Right is case 6 (`$FD` rewritten to `$05`, then 2 pixels back: `$07`); frame 16 Up is case 3, the offset kept, now counting toward 0 at T; frame 20 starts at `$01` with fraction `$00`, a 2-pixel frame, `$01` to `$FF`: T's test is never made; frame 25 the offset reaches -8 at Y `$8D` and is reset, and the test there reads the block's upper tile row. [`Link_SetInputDir@ReverseDir`](src/bank5/link-handle-input.asm#L404) [`Link_SetDirAndInputDir`](src/bank5/link-handle-input.asm#L365) [`Obj_SubQSpeedFromPosFrac`](src/bank1/qspeed-direction-math.asm#L94)
- Which blocks of room `$22` lead somewhere from below (trace): the east block, (`$A0`,`$95`) to (`$A0`,`$8D`), Left in; the west block, (`$60`,`$95`) to (`$60`,`$8D`), Right in; the lower-left block, (`$70`,`$A5`) to (`$70`,`$9D`), Right to X `$80`, Up onto the stairs; the lower-right block, (`$90`,`$A5`) to (`$90`,`$9D`), Left in; the upper blocks, (`$70`/`$90`, `$85`) to `$7D`. Beside the east, west and four diagonal blocks the tile ahead sideways belongs to an inside square. [`Tile_GetAtObjOffset@FetchTile`](src/bank7/tile-collision.asm#L144)
- The south block (`$80`,`$B0`): from rest on (`$80`,`$BD`) after walking Left along the bottom row (fraction `$80`) it clips to (`$80`,`$AD`), but the squares beside that row are outside the diamond: nothing gained (trace). The north block (`$80`,`$70`) cannot be clipped from above: its stop is Y `$5D`, the top room bound, so there is no G. [`Walker_Move@CheckBoundary`](src/bank7/walker-move.asm#L185)
- From above (the reverse block clip, below) every block with a full cell above its stop works; from the left or right Link gets 8 pixels in and no further. [`Tile_GetAheadOfObj`](src/bank7/tile-collision.asm#L65)
- Push block caveat: the west block is the room's push block; holding Up against it for 16 frames from (`$60`, `$8D`) or below pushes it, so after its clip go Right at once. [`PushBlock_State0Idle`](src/bank4/push-block.asm#L74)
- Traps: the four blade traps of room `$22` charge only when Link is within `$0D` of a trap's row (Y `$5D`, `$BD`) or column (X `$20`, `$D0`). The east- and west-block clips, their run-up and the walk to the stairs stay inside X `$30`-`$C0`, Y `$6D`-`$AD`: the traps never move (trace: the whole recipe with the traps in the room, hearts unchanged). The lower-left block's run-up passes Y `$B5`, within the bottom traps' reach, and also ended without damage at walking pace (trace). [`Trap_Update_Full`](src/bank1/trap-rupee-stash.asm#L189) [`Trap_Update_Full@TryCharge`](src/bank1/trap-rupee-stash.asm#L216)
- No softlock in room `$22`: half inside a block Link can always walk back the way he came (the test behind him reads the square he left), and the stairs lead to cellar `$7F`, whose exits put him at (`$60`, `$AD`) outside the diamond. [`Warp_CheckUnderLink`](src/bank5/warps-stairs-caves.asm#L55)
- Other solid squares clip by the same table, 8 pixels deep, from every side tried (trace): level 2 room `$7D` statues (below one at Y `$B0`-`$BF` Link ends at Y `$AD`, not `$B5`; above it at `$A5`, not `$9D`; right of one at X `$30`-`$3F` at X `$38`, not `$40`; left of one at X `$60`-`$6F` at `$58`, not `$50`); overworld room `$77` (8 pixels into the bottom wall heading down, the top wall heading up, the rock at the left of row Y `$95` heading left). The overworld needs no sword: the plain recipe is enough. [`Walker_TestTileAhead`](src/bank7/walker-collision-screen-edge.asm#L138)
- Dungeon walls are not passed by the plain skip: the room bounds are tested every frame by inequality, and Link stops 1 pixel past the grid point (trace: room `$7D`, X = `$1F` at the left wall, Y = `$5C` at the top wall). Only a laid ladder carries him past a bound ("Ladder clip"). [`Room_BoundDir`](src/bank1/room-bounds.asm#L254) [`Walker_Move@CheckBoundary`](src/bank7/walker-move.asm#L185)
- At a screen edge (overworld): a right wrap passes a solid 16-pixel square at the left edge; a left wrap is stopped by one at the right edge; an up wrap is stopped by a solid bottom row; a down wrap needs row 0 open one column to the right (`# Screen wrap (overworld)`, "After the wrap"). [`Tile_GetAtObjOffset@AdjustX`](src/bank7/tile-collision.asm#L123) [`Tile_GetAtObjOffset@FetchTile`](src/bank7/tile-collision.asm#L144)

**Reverse block clip**
- The block clip heading down. From above the east block of room `$22`: run-up cell (`$A0`, `$6D`) to (`$A0`, `$75`) (fraction `$00`), Down 2 frames (Y `$78`, offset `$03`, fraction `$00`), Left or Right 1 frame, hold Down: Link stops at (`$A0`, `$85`), 8 pixels into the block (trace, traps in the room). [`Link_SetInputDir@ReverseDir`](src/bank5/link-handle-input.asm#L404)
- His hotspot row is then the block's upper tile row, so Left takes him into the inside square (`$90`, `$90`); at X `$80` Down puts him on the stairs (trace; the walk from (`$E0`, `$8D`): Left to X `$B0`, Up to Y `$6D`, Left to X `$A0`, Down to `$75`, then the clip). [`Tile_GetAtObjOffset@FetchTile`](src/bank7/tile-collision.asm#L144)
- From above the west block the same gives (`$60`, `$85`), and Right walks into the diamond (trace). [`Tile_GetAheadOfObj`](src/bank7/tile-collision.asm#L65)
- A second, horizontal clip does not pass a block's second tile column: from above the west block at X `$58` (half a column off) Link clips to (`$58`, `$85`); Right is then blocked by the block's right tile (X + `$10` = `$68`), and a horizontal clip from X `$48` stops him at `$58` again (trace). Room `$22` needs no second clip. [`Tile_GetAheadOfObj`](src/bank7/tile-collision.asm#L65)

**Ladder clip**
- What a laid ladder is: laid at Link's Y - 5 (up), Y + `$13` (down), X -+ `$10` with Y + 3 (left, right), one square ahead, when he stands on a grid point facing water with that direction held and no ladder out (all conditions in `# Ladder and raft`). [`Ladder_LinkOffsetsY`](src/bank7/link-animate-ladder.asm#L62) [`Ladder_LinkOffsetsX`](src/bank7/link-animate-ladder.asm#L58) [`Link_EndMoveAndAnimate@SetUpLadder`](src/bank7/link-animate-ladder.asm#L246)
- Distance = |Link Y + 3 - ladder Y| for a ladder laid up or down, |Link X - ladder X| for one laid left or right: a ladder laid up starts at distance 8, one laid down, left or right at `$10`. [`Ladder_CheckAndDraw@CheckDistanceToLadder`](src/bank5/ladder.asm#L82)
- The ladder code runs every frame after the room bound, the doorway check and Link's tile test, and may rewrite the moving direction they left. With a direction held: A, distance not 0 and Link facing the ladder's way: he moves in his facing direction whatever the tile, the bound or the door said; B, distance 0 and the ladder's direction is his moving direction (the tile test let him on): go on; C, facing opposite to the ladder: go back; D, anything else stops him, except E: a ladder laid up with exactly Up held repeats the tile test 8 pixels higher and that decides. [`Walker_Move@CheckTiles`](src/bank7/walker-move.asm#L206) [`Ladder_CheckAndDraw@HandleInput`](src/bank5/ladder.asm#L125) [`Ladder_CheckAndDraw`](src/bank5/ladder.asm#L44)
- So with a ladder out the only grid point whose tile can stop Link is T, the one at distance 0; without the skip he stops there when the square beyond is water too (trace: room `$21` going up at Y `$7D`, where case E finds water at `$78`, going left at X `$40`, going right at X `$B0`; room `$5F` at X `$E0`). [`Link_StopAtUnwalkableTile`](src/bank7/walker-collision-screen-edge.asm#L167)
- The skip removes that one test; the trick state is 5 pixels short of T as everywhere, and the press's step back is allowed by case C. [`Link_SetInputDir@ReverseDir`](src/bank5/link-handle-input.asm#L404) [`Ladder_CheckAndDraw@HandleInput`](src/bank5/ladder.asm#L125)
- Past T: at T -+ 8 (distance 8, water ahead) case A carries him on; at T -+ `$10` (distance `$10`, ladder state 2) the next frame puts the ladder away and Link does not move; on that frame, standing on a grid point with no ladder out, the direction held and water ahead, a new ladder is laid one square farther (the relay). [`Ladder_CheckAndDraw@StashLadder`](src/bank5/ladder.asm#L110) [`Link_EndMoveAndAnimate@SetUpLadder`](src/bank7/link-animate-ladder.asm#L246)
- Reach of one press, ladders laid left, right or down: lay point L (Link's position when it is laid, G' of the skip), distance 0 at L -+ `$10` (skipped), relay at L -+ `$20`, the new ladder's distance 0 at L -+ `$30`, where the tile ahead must be walkable: 3 water squares; 2 when the tile ahead at the relay point is already land and no new ladder is laid (trace: room `$21` west and east, room `$20` row 1, room `$5F`). [`Link_EndMoveAndAnimate@SetUpLadder`](src/bank7/link-animate-ladder.asm#L246)
- Ladder laid up: lay point L = water top + 5 (G of the skip), distance 0 at L - 8 (skipped), relay at L - `$18`, the new ladder's distance 0 at L - `$20`, where Y + 3 must be land, or by case E Y - 5: room `$21`'s two squares need one press (trace). [`Ladder_CheckAndDraw`](src/bank5/ladder.asm#L44)
- A second press is needed only for wider water: one at each further distance-0 point, in the same trick state. Left, right or down the relay point is G' of the next distance-0 point, so any fraction serves; up it is G, and Link reaches it with the good fraction (frame 24 of the room `$21` table: (`$6D`, `$80`)). No water wider than the reach of one press was tried. [`Walker_CheckTileCollision@CheckGridOffset`](src/bank7/walker-collision-screen-edge.asm#L101)
- The ladder works in every dungeon room and in overworld rooms `$17`, `$18`, `$19`, `$27`, `$4F`, `$5F` only. [`Ladder_RoomIdsOW`](src/bank7/link-animate-ladder.asm#L76)

*Overworld: room `$5F` and its heart container, entered from room `$60`*
- Room `$5F` (row 5, column 15) can be entered standing on water from room `$60` (row 6, column 0) by a right wrap: `$60` has its right edge open on tile rows 8-`$0D` (aligned Y `$75`-`$A5`) and columns 0-3 solid on every row. [`Tile_GetAtObjOffset@AdjustX`](src/bank7/tile-collision.asm#L123)
- Wrap in `$60`, row Y `$8D` (or `$85`): coming in from room `$61` at X `$F0`, walk Left 2 cells to X `$E0` (from X `$E8` the first step right would be a 2-pixel step, the wrong cadence), then Right 7, Up 1, Right: X `$00` on frame 23, inside the solid left wall, where Right, Up and Down do nothing (trace). At X `$F8` the tile under Link is read (no look-ahead at X >= `$F0`), at X `$00` column 2 (X + `$10`), solid. [`Tile_GetAtObjOffset@AdjustX`](src/bank7/tile-collision.asm#L123)
- Then Left: X `$00` equals the left bound, the overworld makes the edge test with a solid tile ahead, and `RoomId` - 1 in 8 bits is `$5F`, the last room of the row above. 104 frames later Link plays in room `$5F` at (`$F0`, `$8D`) facing left, `$394` = 0, `$3A8` = `$00`, standing on water (tiles `$95` in columns `$1C`-`$1F` of his row): the arrival spot is never tested (trace). [`Link_StopAtUnwalkableTile`](src/bank7/walker-collision-screen-edge.asm#L167) [`Room_CalcNextRoomForDoor`](src/bank5/next-room-calc.asm#L76) [`Room_EnterCommon@Method2`](src/bank5/enter-room.asm#L211)
- Room `$5F`, rows Y `$85`/`$8D` (tile rows `$0A`-`$0B`): land up to X `$8F`; water `$90`-`$9F`; platform `$A0`-`$AF` (tiles `$76`/`$77`); water `$B0`-`$BF`; platform `$C0`-`$CF` with the heart container at (`$C0`, `$90`), the overworld's only room-item heart container; water `$D0`-`$FF`. Room `$4F` has the same layout, without the heart container. [`Room_SetUpItemAndPushBlock@MakeHeartContainerOW`](src/bank5/create-room-objects.asm#L116)
- Recipe in `$5F` (`InvLadder` = 1, a free monster slot): hold Left; the first play frame does not move Link, at the end of the 2nd the ladder is laid at X `$E0`; Link walks onto it, `$F0` to `$E8` in frames 3-7, then `$E7`, `$E5`; frame 10: Up (or Down) for exactly 1 frame, the frame that starts at X `$E5` (`$394` `$FD`, `$3A8` `$80`), 5 pixels short of `$E0` and the 3rd frame after the frame that ended on `$E8`; then Left: `$E0` passed on frame 15, `$D8` on frame 20, `$D0` on frame 25, where the ladder is put away and Link loses 1 frame; the heart container is taken at X `$C8` on frame 32. Input: Left 9, Up 1, Left 22 = 32 frames (trace). [`Link_ModifyDirOnGridLine`](src/bank5/link-handle-input.asm#L338) [`Item_TryTake`](src/bank1/take-item.asm#L206)
- Water crossed with that one press: the square Link arrived on (`$F0`-`$FF`), the ladder's square (`$E0`-`$EF`) and the square `$D0`-`$DF` that the laid ladder does not cover, X from `$F0` to `$D0`, 32 pixels. At X `$D0` the tile ahead is the platform, so no new ladder is laid. West of the heart platform the ladder works as usual: at X `$C0` a new ladder over `$B0`-`$BF`, platform `$A0`, ladder over `$90`-`$9F`, land at `$88`. [`Ladder_CheckAndDraw`](src/bank5/ladder.asm#L44) [`Link_EndMoveAndAnimate@CheckWaterOW`](src/bank7/link-animate-ladder.asm#L240)
- Variants that work (trace): Up or Down as the press; rows Y `$85` and `$8D` (both take the heart container: range |X - `$C0`| < 9, Link's Y `$85`-`$95`); row Y `$95` (all water): at `$D0` a new ladder is laid toward `$C0` and Link takes the heart container from X `$C8` while still over water; room `$4F`: the same clip crosses to the platform. [`Item_TryTake`](src/bank1/take-item.asm#L206) [`Link_EndMoveAndAnimate@SetUpLadder`](src/bank7/link-animate-ladder.asm#L246)

**Frame table, room `$60` row Y `$8D`: right wrap, then Left into `$5F`** (trace; frame 0 = at rest on X `$E0` after walking 2 cells left from the `$F0` arrival; the monsters of `$60` were present)

| frame | input | `$70` X | `$98` | `$394` | `$3A8` | `$3F8` | `$12` | `$EB` |
|---|---|---|---|---|---|---|---|---|
| 0 | - | `$E0` | 2 | `$00` | `$80` | 2 | 5 | `$60` |
| 1-4 | Right | `$E2`, `$E3`, `$E5`, `$E6` | 1 | `$02`, `$03`, `$05`, `$06` | `$00`, `$80`, `$00`, `$80` | 1 | 5 | `$60` |
| 5 | Right | `$E8` | 1 | `$00` | `$00` | 1 | 5 | `$60` |
| 6 | Right | `$E9` | 1 | `$01` | `$80` | 1 | 5 | `$60` |
| 7 | Right | `$EB` | 1 | `$03` | `$00` | 1 | 5 | `$60` |
| 8 | Up | `$E9` | 2 | `$F9` | `$80` | 8 | 5 | `$60` |
| 9 | Right | `$EB` | 1 | `$FB` | `$00` | 1 | 5 | `$60` |
| 13 | Right | `$F1` | 1 | `$01` | `$00` | 1 | 5 | `$60` |
| 18 | Right | `$F8` | 1 | `$00` | `$80` | 1 | 5 | `$60` |
| 23 | Right | `$00` | 1 | `$00` | `$00` | 1 | 5 | `$60` |
| 24 | Left | `$00` | 2 | `$00` | `$00` | 2 | 6 | `$60` |
| 128 | - | `$F0` | 2 | `$00` | `$00` | | 5 | `$5F` |

**Frame table, room `$5F` ladder clip** (trace; frame 0 = the first play frame after the arrival; `$AC` = 0 and Y = `$8D` throughout; ladder X = `ObjX` of the ladder's slot)

| frame | input | `$70` X | `$98` | `$394` | `$3A8` | `$3F8` | `$64` | ladder X |
|---|---|---|---|---|---|---|---|---|
| 0 | - | `$F0` | 2 | `$00` | `$00` | 0 | 0 | - |
| 1 | Left | `$F0` | 2 | `$00` | `$00` | 0 | 0 | - |
| 2 | Left | `$F0` | 2 | `$00` | `$00` | 2 | `$0B` | `$E0` |
| 3 | Left | `$EE` | 2 | `$FE` | `$80` | 2 | `$0B` | `$E0` |
| 4 | Left | `$ED` | 2 | `$FD` | `$00` | 2 | `$0B` | `$E0` |
| 5 | Left | `$EB` | 2 | `$FB` | `$80` | 2 | `$0B` | `$E0` |
| 6 | Left | `$EA` | 2 | `$FA` | `$00` | 2 | `$0B` | `$E0` |
| 7 | Left | `$E8` | 2 | `$00` | `$80` | 2 | `$0B` | `$E0` |
| 8 | Left | `$E7` | 2 | `$FF` | `$00` | 2 | `$0B` | `$E0` |
| 9 | Left | `$E5` | 2 | `$FD` | `$80` | 2 | `$0B` | `$E0` |
| 10 | Up | `$E7` | 1 | `$07` | `$00` | 8 | `$0B` | `$E0` |
| 11 | Left | `$E5` | 2 | `$05` | `$80` | 2 | `$0B` | `$E0` |
| 12 | Left | `$E4` | 2 | `$04` | `$00` | 2 | `$0B` | `$E0` |
| 13 | Left | `$E2` | 2 | `$02` | `$80` | 2 | `$0B` | `$E0` |
| 14 | Left | `$E1` | 2 | `$01` | `$00` | 2 | `$0B` | `$E0` |
| 15 | Left | `$DF` | 2 | `$FF` | `$80` | 2 | `$0B` | `$E0` |
| 19 | Left | `$D9` | 2 | `$F9` | `$80` | 2 | `$0B` | `$E0` |
| 20 | Left | `$D8` | 2 | `$00` | `$00` | 2 | `$0B` | `$E0` |
| 25 | Left | `$D0` | 2 | `$00` | `$80` | 2 | `$0B` | `$E0` |
| 26 | Left | `$D0` | 2 | `$00` | `$80` | 2 | 0 | - |
| 27 | Left | `$CF` | 2 | `$FF` | `$00` | 2 | 0 | - |
| 32 | Left | `$C8` | 2 | `$00` | `$80` | 2 | 0 | - (heart container taken, `HeartValues` `$22` to `$33`) |
| 38 | Left | `$C0` | 2 | `$00` | `$80` | 2 | `$0B` | `$B0` (the plain ladder to the next platform) |

*Dungeon: level 4 room `$21` from the south*
- Room `$21` from its S door (the key door from room `$31`; in `$31` from the E door Left along Y `$8D`, where the ladder crosses its four single squares, to X `$78`, then Up): Link walks in to (`$78`, `$CD`) on the island (rows 3-6 in the middle, water two or more squares wide on every side). Going up he stops at Y `$85`, where the tile ahead (Y + 3 = `$88`) is water `$F4`, and the ladder is laid at Y `$80` on that frame (trace). Traces in `$21` were made without its 5 Gels unless stated. [`Link_EndMoveAndAnimate@SetUpLadder`](src/bank7/link-animate-ladder.asm#L246)
- Preconditions: mode 5; `InvLadder`; a free monster slot (the 5 Gels leave 6); `LadderSlot` = 0 when Link reaches Y `$85`; Up held so `ObjInputDir` = `ObjDir` = 8; any X from `$50` to `$A0` (trace: all 11 grid columns); Link must reach `$85` walking up from Y `$8D` or below (trace: standing on `$85` with `$00`, after walking right along Y `$85` under Up+Right or walking down from the water, fails). [`Link_EndMoveAndAnimate@CheckLadder`](src/bank7/link-animate-ladder.asm#L209)
- Trick state: Y `$82`, `$FD`, `$80`, facing up, ladder out (state 2): Y + 3 - ladder Y = 5. From the frame Link arrives on `$85` (offset 0, `$80`, ladder laid): Up, Up, press on the 3rd frame, then Up until Y `$55`. From rest at (`$78`, `$8D`) with `$80`: Up 8, press 1, Up 34 = 43 frames to (`$78`, `$55`) (trace). One press crosses both water squares. [`Ladder_CheckAndDraw@CheckDistanceToLadder`](src/bank5/ladder.asm#L82) [`Ladder_CheckAndDraw@HandleInput`](src/bank5/ladder.asm#L125)
- Why: frame 9 is the reversal (-3 rewritten to +5, then 2 pixels down); frame 14 steps the offset from +1 to -1 across Y `$7D` with no frame at offset 0 there; at `$75` (distance 8, water ahead) case A carries him on; frame 25, at `$6D` (distance `$10`, state 2), puts the first ladder away (Link does not move) and lays the second at Y `$68` (Y + 3 = `$70` is water), distance 8; its distance-0 point `$65` has a walkable tile ahead (`$68`, the top floor row); at `$5D` (distance 8) the room bound zeroes the move and case A restores it; at `$55` (distance `$10`) the ladder is put away and Link stands 8 pixels past the top bound. [`Walker_Move@CheckBoundary`](src/bank7/walker-move.asm#L185) [`Ladder_CheckAndDraw@StashLadder`](src/bank5/ladder.asm#L110)
- Where he comes out: (`$78`, `$55`), 8 pixels into the top wall, no ladder out; Down 5 frames puts him on the N door spot (`$78`, `$5D`); Left walks him along Y `$55` to X `$20` (trace). [`Ladder_CheckAndDraw@StashLadder`](src/bank5/ladder.asm#L110)
- On to room `$11`: from (`$78`, `$5D`) Up for 1 frame (it faces him up and puts away the ladder he laid going down), B with a bomb: the N bomb wall opens and Up takes him into room `$11` (trace). On to room `$20`: from Y `$55` Left to X `$20`, Down to Y `$8D`, Left: the W door (trace). [`Bomb_TryOpenWall`](src/bank7/bomb.asm#L136) [`Doorway_CheckLink`](src/bank5/doorway.asm#L137)

**Frame table, room `$21` north** (trace): frame 0 = Link at (`$78`, `$8D`), offset 0, fraction `$80`, no ladder; values after each frame; X = `$78` and Link's `$AC` = 0 throughout; the ladder in slot `$0B` (ladder Y = `$8F`, ladder state = `$B7`).

| frame | input | `$84` Y | `$98` | `$394` | `$3A8` | `$3F8` | `$64` | ladder Y | ladder state |
|---|---|---|---|---|---|---|---|---|---|
| 1-5 | Up | `$8C`, `$8A`, `$89`, `$87`, `$86` | 8 | `$FF`, `$FD`, `$FC`, `$FA`, `$F9` | `$00`, `$80`, `$00`, `$80`, `$00` | 8 | 0 | - | - |
| 6 | Up | `$85` | 8 | `$00` | `$80` | 8 | `$0B` | `$80` | 1 |
| 7 | Up | `$84` | 8 | `$FF` | `$00` | 8 | `$0B` | `$80` | 2 |
| 8 | Up | `$82` | 8 | `$FD` | `$80` | 8 | `$0B` | `$80` | 2 |
| 9 | Left | `$84` | 4 | `$07` | `$00` | 2 | `$0B` | `$80` | 2 |
| 10 | Up | `$82` | 8 | `$05` | `$80` | 8 | `$0B` | `$80` | 2 |
| 11-13 | Up | `$81`, `$7F`, `$7E` | 8 | `$04`, `$02`, `$01` | `$00`, `$80`, `$00` | 8 | `$0B` | `$80` | 2 |
| 14 | Up | `$7C` | 8 | `$FF` | `$80` | 8 | `$0B` | `$80` | 2 |
| 15-18 | Up | `$7B`, `$79`, `$78`, `$76` | 8 | `$FE`, `$FC`, `$FB`, `$F9` | `$00`, `$80`, `$00`, `$80` | 8 | `$0B` | `$80` | 2 |
| 19 | Up | `$75` | 8 | `$00` | `$00` | 8 | `$0B` | `$80` | 2 |
| 20-23 | Up | `$73`, `$72`, `$70`, `$6F` | 8 | `$FE`, `$FD`, `$FB`, `$FA` | `$80`, `$00`, `$80`, `$00` | 8 | `$0B` | `$80` | 2 |
| 24 | Up | `$6D` | 8 | `$00` | `$80` | 8 | `$0B` | `$80` | 2 |
| 25 | Up | `$6D` | 8 | `$00` | `$80` | 8 | `$0B` | `$68` | 1 |
| 26-30 | Up | `$6C` ... `$66` | 8 | `$FF` ... `$F9` | alternating | 8 | `$0B` | `$68` | 2 |
| 31 | Up | `$65` | 8 | `$00` | `$80` | 8 | `$0B` | `$68` | 2 |
| 37 | Up | `$5D` | 8 | `$00` | `$80` | 8 | `$0B` | `$68` | 2 |
| 43 | Up | `$55` | 8 | `$00` | `$80` | 8 | `$0B` | `$68` | 2 |
| 44 | Up | `$55` | 8 | `$00` | `$80` | 8 | 0 | - | - |

- Other directions from the island (trace): west, rows Y `$8D`, `$95`, `$9D`: ladder laid at X `$40` from X `$50`; trick (`$45`, `$FD`, `$80`), Up or Down; Link ends at X `$20`, and on Y `$8D` walks straight on through the W door into room `$20`. East, same rows: ladder at X `$B0` from X `$A0`; trick (`$AB`, `$03`, `$00`), Up or Down; Link ends at X `$D0` in the E column (the map lies at (`$D0`, `$C0`); the E side is a wall). Left and right the grid point before the trick is X `$48`/`$A8`, one cell past the lay point, so from the lay point: Left 8 and Up, or Right 8 and Up. [`Link_EndMoveAndAnimate@SetUpLadder`](src/bank7/link-animate-ladder.asm#L246) [`Ladder_LinkOffsetsX`](src/bank7/link-animate-ladder.asm#L58)
- South from the top row back to the island: ladder at Y `$70` from Y `$5D`, trick (`$68`, `$03`, `$00`), Left; Link ends on the island at Y `$9D` (trace). [`Ladder_LinkOffsetsY`](src/bank7/link-animate-ladder.asm#L62)
- Room `$20`, row 1 (Y `$6D`), three water squares X `$30`-`$5F`: from X `$60` Left 9, Up: Link reaches X `$20` with one press (ladders laid at `$50` and `$30`); the ring of `$20` is open, so this saves nothing there (trace). Room `$30` has no use for it: its water is one row, which the plain ladder crosses. [`Link_EndMoveAndAnimate@SetUpLadder`](src/bank7/link-animate-ladder.asm#L246) [`Ladder_CheckAndDraw`](src/bank5/ladder.asm#L44)
- With the 5 Gels of room `$21` alive the decision table succeeded after each of 100 idle delays (0-297 frames) after entry; 73 runs took at least half a heart: a touch's knockback spoils the cadence and the back-off repairs it (trace). [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73) [`Shove_Update`](src/bank7/shove.asm#L55)

**Failed attempts and softlocks**
- Plain skip, any place: a press on any other frame, two perpendicular frames (the second is case 5 and carries Link on to the grid point behind him, where the offset is reset), Left then Right, or the opposite of s instead of a perpendicular (a plain turn-back retraces): Link stops normally on T, no damage, nothing changed but his position (trace: room `$22`, all four screen edges, room `$5F` at X `$EE`, `$ED`, `$EB`, `$EA`, `$E8`, `$E7`, `$E4`, `$E2`, `$E1`, room `$21` at Y `$7D`). Recovery: back off to G' ("Recovery"). [`Link_ModifyDirOnGridLine`](src/bank5/link-handle-input.asm#L338)
- A perpendicular held during the crossing while Link is 1-3 pixels past T is case 6 again and undoes the skip: he ends on T as after a failed press (trace: room `$21`, Left or Right held for many frames before Y `$79`). From 4 pixels past T on it is case 5 and harmless. [`Link_SetInputDir@ReverseDir`](src/bank5/link-handle-input.asm#L404)
- Screen wrap: a missed press is the ordinary room change (at the right edge of `$60` into `$61`, the Lost Woods). After a wrap Link can be stuck: left onto a row whose right edge is solid he stops at X `$F8` and no direction moves him; right onto a row with two solid squares at the left he stands inside them at X `$00` and only Left moves him, into the room on the left, where he may again stand inside solid tiles (`# Screen wrap (overworld)`, "After the wrap"). [`Maze_CheckStep`](src/bank1/mazes-lost-woods-hills.asm#L56) [`Tile_GetAtObjOffset@AdjustX`](src/bank7/tile-collision.asm#L123)
- Room `$60` to `$5F` without the ladder is a softlock: at (`$F0`, `$8D`) of `$5F` no direction moves Link (water ahead, above and below); Right changes to `$60` at (`$00`, `$8D`) inside the wall, where only Left moves him, back to `$5F`. Neither spot can start a new skip (trace). [`Link_StopAtUnwalkableTile`](src/bank7/walker-collision-screen-edge.asm#L167)
- Room `$5F`, missed ladder clip: Link stands on the ladder at X `$E0` facing water; Left does nothing. Recovery: Right 11 frames to `$F0` (case C), then Left (the old ladder is put away and a new one laid at `$E0` on the first frame), Left 8 frames to `$E5`, Up, Left: clip (trace). Backing off only to `$E8` is too short: arriving there walking right gives `$3A8` = `$00`, and no frame of the next Left walk clips (trace). [`Ladder_CheckAndDraw@StashLadder`](src/bank5/ladder.asm#L110) [`Obj_AddQSpeedToPosFrac`](src/bank1/qspeed-direction-math.asm#L52)
- Room `$5F`, Right held on at `$F0` after a miss: the ladder is put away and on the same frame a new one is laid at X `$F0` + `$10` = `$00`; with a ladder out there is no edge test, so Link walks onto it to X `$00` of `$5F`, inside the solid left column, with no room change; Left brings him back (trace). To leave east instead: at `$F0` release for 1 frame (the ladder is put away and none is laid without a held direction), then Right: room change to `$60` (trace). [`Walker_CheckTileCollision`](src/bank7/walker-collision-screen-edge.asm#L89) [`Walker_AcceptMovingDir`](src/bank7/walker-collision-screen-edge.asm#L200)
- Room `$5F`, Up or Down on the grid point `$F0` with the ladder out: the left ladder is put away and an upward one laid; Left is then refused (case D). Recovery: Down 3, Left 11 (Link finishes the cell to Y `$95`, where the vertical ladder is put away and a left one laid), Up at `$E5`, Left 22: the heart container is taken from row `$95` (trace). [`Ladder_CheckAndDraw@HandleInput`](src/bank5/ladder.asm#L125)
- Room `$21`, failed press: Link stops on the ladder at Y `$7D` (distance 0) over the water; Down walks him back to the island (case C), where the decision table starts again (Down 11, Up 8, press) (trace). [`Ladder_CheckAndDraw@HandleInput`](src/bank5/ladder.asm#L125)
- Room `$21`, after a successful skip Link may stop anywhere over the water: Up or Down from there always reaches land (trace: every frame of the crossing, stopped 3 frames, then Up or Down). [`Ladder_CheckAndDraw`](src/bank5/ladder.asm#L44)
- Room `$21`, Left or Right over the water lays a sideways ladder along the water row (Y `$6D`, row 1 is water): Link stops at its distance-0 point (X `$68` or `$88`) and Up or Down do nothing while it is out. Not a softlock: walk back to X `$78` (distance `$10`), then Up: the turn puts the sideways ladder away and lays one upward (trace). [`Ladder_CheckAndDraw@StashLadder`](src/bank5/ladder.asm#L110)
- With the ladder there is always a way back: a laid ladder lets Link back the way he came, and on a grid point with no ladder out every water tile ahead lays one, given a free monster slot (trace: no position in rooms `$21` and `$5F` was found where no direction moves him). [`Obj_FindEmptyMonsterSlot`](src/bank7/object-stubs-destroy-monster.asm#L169) [`Link_EndMoveAndAnimate@CheckLadder`](src/bank7/link-animate-ladder.asm#L209)

**No second skip in a row**
- A skip needs the frame that starts 1 pixel before T to move 2 pixels, i.e. h at a fixed remainder (mod 3) from T. After the crossing Link reaches the next grid point, T -+ 8, without a withheld pixel and with the fraction that is wrong for skipping it: no frame starts 3 pixels before T -+ 8, and the frame 1 pixel before it comes with the other fraction, so neither the plain nor the sword state for T -+ 8 occurs (frames 20-25 of the room `$22` table). Only a withheld pixel changes the remainder, and that happens only on a grid point reached by a full cell, whose test is then made. [`Obj_SubQSpeedFromPosFrac@SetCarry`](src/bank1/qspeed-direction-math.asm#L108) [`Link_EndMoveAndAnimate@TruncGridOffset`](src/bank7/link-animate-ladder.asm#L177)
- The next grid point that can be skipped is T -+ `$18`: Link reaches T -+ `$10` by a full cell from T -+ 8, so with the good fraction. Through a 16-pixel block that never helps, since T -+ 8 is inside it and stops him; on a ladder T -+ 8 and T -+ `$10` are passed by case A and the relay, which is why wider water takes a second press ("Ladder clip"). [`Ladder_CheckAndDraw@HandleInput`](src/bank5/ladder.asm#L125)
- Searched (trace): east block of room `$22`, from each of the five frames around the crossing of `$95`, every sequence of 1-5 frames of Up, Down, Left, Right, A or nothing, then Up: 46,650 runs, none past `$8D`; level 2 room `$7D`, after the first press, at each of the next 14 frames a second press (Left, Right or Down) of 1 or 2 frames with 0 or 1 idle frame after it, then Up: 168 runs, none past Y `$AD`. [`Obj_SubQSpeedFromPosFrac@SetCarry`](src/bank1/qspeed-direction-math.asm#L108)

**Lengths** (frames from the standstill positions given; trace unless said otherwise)
- Left wrap, room `$77`: from rest on (`$08`, `$8D`) with `$80`: Left 2, Up 1, Left 5, past the bound on the 8th frame, X `$F0` of the same room on frame 18; from (`$10`, `$8D`) with `$80`: Left 8, Up 1, Left 5, X `$FF` on frame 14. [`Link_CheckScreenEdge`](src/bank7/walker-collision-screen-edge.asm#L243)
- Right wrap, room `$77`: from rest on (`$E0`, `$8D`) with `$00`: Right 8, Up 1, Right 5 (X `$F1` on frame 14); X `$00` on frame 24. [`Link_CheckScreenEdge`](src/bank7/walker-collision-screen-edge.asm#L243)
- Room `$60` to `$5F`: from rest on (`$E0`, `$8D`) with `$80`: Right 7, Up 1, Right 15 (X `$00` on frame 23), Left 1 (frame 24, the room change starts); Link plays in `$5F` on frame 128. [`Room_EnterCommon@Method2`](src/bank5/enter-room.asm#L211)
- Ladder clip, room `$5F`: from the arrival Left 9, Up 1, Left 22 = 32 frames to the heart container; 160 frames from rest on X `$E0` of room `$60`. [`Item_TryTake`](src/bank1/take-item.asm#L206)
- Block clip, room `$22`: from rest on (`$A0`, `$AD`) with `$80`: Up 14, Right 1, Up 10 = 25 frames to (`$A0`, `$8D`), then Left 22 to the stairs; 117 frames from (`$E0`, `$8D`) in the E doorway. [`Warp_CheckUnderLink`](src/bank5/warps-stairs-caves.asm#L55)
- Reverse block clip, room `$22`: from rest on (`$A0`, `$75`) with `$00`: Down 2, press 1, Down 10 to (`$A0`, `$85`) (counted from the movement rule; the trace recorded the positions, not the frames). [`Obj_AddQSpeedToPosFrac`](src/bank1/qspeed-direction-math.asm#L52)
- Ladder clip, room `$21`: from rest on (`$78`, `$8D`) with `$80`: Up 8, press 1, Up 34 = 43 frames to (`$78`, `$55`); Down 5 to the N door spot; Up 1, B: the wall is passable 81 frames after the button. From the island entry (`$78`, `$CD`), 9 cells before G: 55 or 56 frames of Up, then the press (the run-up formula above). [`Ladder_CheckAndDraw@StashLadder`](src/bank5/ladder.asm#L110) [`Bomb_TryOpenWall`](src/bank7/bomb.asm#L136)
- Back-offs: see "Recovery" (6 + 7 + 1 from G with the wrong fraction; 11 or 12 + 7 + 1 from T; one more frame with a ladder out). [`Link_EndMoveAndAnimate@TruncGridOffset`](src/bank7/link-animate-ladder.asm#L177)
- What must not be interrupted: before the press, stops and turn-backs inside the cell are harmless; a full cell walked away from T, knockback or a room change spoil the cadence. The press is exactly 1 frame. Between the press and the crossing of T only s or nothing may be held (idle frames are harmless); until Link is 4 pixels past T no perpendicular. After that nothing matters until the next grid point; on a ladder Link may stop anywhere after the crossing. The 104 frames of the room change from `$60` to `$5F` take no input. [`Link_ModifyDirOnGridLine`](src/bank5/link-handle-input.asm#L338)

# Doorways, doors, keys, shutters and bomb walls

Dungeon rooms only. Each side of a room has one door type from the level block; whether a door is open right now is a bit in `CurOpenedDoors`, and whether it was opened earlier is a bit in the room's flag byte. Direction bits: `1` right, `2` left, `4` down, `8` up.

**Door types and where they come from**
- A side's door type is 3 bits of the room's level block bytes: attribute A bits 7-5 = up (N), bits 4-2 = down (S); attribute B bits 7-5 = left (W), bits 4-2 = right (E); the lookup reads the level data only, never the open state. [`Door_FindTypeByDirBit`](src/bank5/fill-walls.asm#L55)
- Types: `0` open doorway, `1` wall, `2` and `3` false wall, `4` bombable wall, `5` and `6` key door (key, key 2: same handler), `7` shutter; the per-room values are listed in `data/levelblock-uw-*.md`. [`Door_Touch_JumpTable`](src/bank5/touch-door.asm#L62)
- False walls and unopened bombable walls are drawn with the tiles of a plain wall; a key door or shutter shows its closed face until its bit is in `CurOpenedDoors`, then an open doorway (a bombed wall shows a hole). [`Door_LayOut`](src/bank5/lay-out-doors.asm#L131)
- Laying out a room first ORs the room's saved door flags (flag bits 0-3, same order as the direction bits) into `CurOpenedDoors`, so a door opened on an earlier visit is open again. [`Door_AddRoomFlagsToOpenedDoors`](src/bank5/door-flags-dark-room.asm#L84)

**Doorway movement**
- Link is at a doorway only with his cross coordinate exact: X = `$78` for the up and down doors, Y = `$8D` for the left and right doors. [`Doorway_RequiredCoords`](src/bank5/doorway.asm#L93)
- From the room a door is touched when Link is in its wide span (Y `$3D`-`$5D` up, `$BD`-`$DD` down, X `$00`-`$20` left, `$CF`-`$F0` right) and the input direction equals the door's direction; any other input changes nothing. [`Doorway_BoundsMinOver`](src/bank5/doorway.asm#L100) [`Doorway_BoundsMaxOver`](src/bank5/doorway.asm#L104) [`Doorway_CheckLink@TestDoorwayDoor`](src/bank5/doorway.asm#L208)
- A handler blocks Link by setting `[0E]` = `$FF`; a pass sets Link's facing, his moving direction and `DoorwayDir` to the door's direction. [`Door_TouchWall`](src/bank5/touch-door.asm#L75) [`Doorway_CheckLink`](src/bank5/doorway.asm#L137)
- The block only works while `DoorwayDir` = 0: once it is nonzero the tile test, the `[0E]` test and the room bounds are all skipped, so nothing stops Link inside a doorway. [`Walker_CheckTileCollision`](src/bank7/walker-collision-screen-edge.asm#L89) [`Doorway_CheckLink`](src/bank5/doorway.asm#L137)
- In a doorway the input is forced onto the axis Link faces: forward if held, else backward if held, else (only a perpendicular direction held) forward. [`Link_ModifyDirInDoorway`](src/bank5/doorway.asm#L52)
- Stepping back out clears `DoorwayDir` when Link leaves the narrow span (Y <= `$5B` up, >= `$BF` down, X <= `$1E` left, >= `$D2` right), 2-3 pixels before the floor spot. [`Doorway_BoundsMinUnder`](src/bank5/doorway.asm#L111) [`Doorway_BoundsMaxUnder`](src/bank5/doorway.asm#L115)
- Entering a room from a neighbour sets `DoorwayDir` to Link's facing; he walks in with grid offset +-`$18` through an open doorway or key door and +-`$28` through any other type (X = `$00` facing right, `$F0` facing left). [`Room_EnterCommon@Method2`](src/bank5/enter-room.asm#L211) [`Room_EnterCommon@ChooseWalkDistance`](src/bank5/enter-room.asm#L240)
- Doorways are not checked in cellars (mode 9). [`Walker_Move`](src/bank7/walker-move.asm#L64)

**Touching each type**
- `0` open: Link passes and walks through the doorway to the screen edge. [`Door_TouchOpen`](src/bank5/touch-door.asm#L80)
- `1` wall: always blocks; the room behind a wall is never computed as a next room. [`Door_TouchWall`](src/bank5/touch-door.asm#L75) [`Room_CalcNoNextRoom`](src/bank5/next-room-calc.asm#L53)
- `2`, `3` false wall: the first touch sets Link's `ObjTimer` to `$18` and blocks; he passes on the frame he is pushing with the timer at exactly 1; if the timer reaches 0 unseen the wait starts over. [`Door_TouchFalseWall`](src/bank5/touch-door.asm#L90)
- `4` bombable wall: passes only when its bit is in `CurOpenedDoors`. [`Door_TouchBombable`](src/bank5/touch-door.asm#L108)
- `7` shutter: passes only when no door command is running and its bit is in `CurOpenedDoors`; after the first touch of an open shutter, later touches pass only while Link's `ObjTimer` is 0. [`Door_TouchShutter`](src/bank5/touch-door.asm#L126) [`ShutterTouchedDirs`](src/Variables.inc#L432)
- Passing a false wall or an opened bombable wall starts the room change on that frame; Link then walks `$20` pixels (grid offset +-`$28` down to +-8) before the scroll. [`Doorway_CheckLink`](src/bank5/doorway.asm#L137) [`Mode6Leave_Init`](src/bank5/mode3-mode6-init.asm#L214)
- Leaving through a type `2` false wall requests the "secret found" tune; type `3` does not. [`Mode6Leave_Init`](src/bank5/mode3-mode6-init.asm#L214)

**Keys and the magic key**
- Key door, tests in this order: (1) bit in `CurOpenedDoors`: pass; (2) a door command is running (`TriggeredDoorCmd` <> 0): blocked while Link's `ObjTimer` <> 0, pass when it is 0; (3) `InvMagicKey` <> 0: open, no key spent; (4) `InvKeys` = 0: wall; (5) `InvKeys` - 1, open. [`Door_TouchKey`](src/bank5/touch-door.asm#L160)
- Unlocking sets the open command (`TriggeredDoorCmd` = 6) and Link's `ObjTimer` = `$20`; Link is blocked until the door's bit enters `CurOpenedDoors` 8 frames later. [`Door_TouchKey`](src/bank5/touch-door.asm#L160) [`Door_TriggerOpen`](src/bank5/shutters-door-flags.asm#L117)
- Keyless pass, exactly as coded: while any door command is running and Link's `ObjTimer` is 0, touching a closed key door passes with no key spent and the door still closed; the pass sets `DoorwayDir`, after which Link walks through the closed door to the next room. [`Door_TouchKey`](src/bank5/touch-door.asm#L160) [`Door_Touch_BlockWhileTimer`](src/bank5/touch-door.asm#L193) [`Doorway_CheckLink`](src/bank5/doorway.asm#L137)
- Commands that leave Link's timer at 0 and so allow the keyless pass: a shutter opening by a secret trigger and a bombed wall opening (about 8 frames each; several shutters open one after another). A key unlock sets the timer to `$20` and a close command to `$30`. [`Shutter_OpenTriggered`](src/bank5/shutters-door-flags.asm#L68) [`Bomb_TryOpenWall`](src/bank7/bomb.asm#L136) [`Door_Update`](src/bank5/update-doors.asm#L60)
- After such a pass the old room's door stays locked and unsaved; in the new room the side Link enters by is put into `CurOpenedDoors`, so a key door or bombable wall on that side is drawn open and its flag is saved there. [`Door_SetEnteringDoorwayOpened`](src/bank5/mode7-scroll-init.asm#L368) [`Door_LayOut`](src/bank5/lay-out-doors.asm#L131)
- Only one door command exists, so a key cannot unlock a door while another door is opening or closing. [`TriggeredDoorCmd`](src/Variables.inc#L85) [`Door_TouchKey`](src/bank5/touch-door.asm#L160)

**Door commands, timers and saving**
- A command is `TriggeredDoorCmd` (6 open, 2 close) plus `TriggeredDoorDir`; it runs in two steps 8 frames apart (`DoorTimer`): 6 -> 7 -> done, 2 -> 3 -> done. [`Door_Update`](src/bank5/update-doors.asm#L60) [`DoorTimer`](src/Variables.inc#L42)
- The door update runs each frame of dungeon play after Link, his weapons and the monsters, so a command set during a frame gets its first step that frame; it does nothing in mode `$12`. [`Door_Update`](src/bank5/update-doors.asm#L60)
- Opening: the side enters `CurOpenedDoors` only at the second step, 8 frames after the first; until then it counts as closed. [`Door_Update`](src/bank5/update-doors.asm#L60)
- An opened key door or bombable wall sets its flag bit in this room and the opposite side's bit in the room behind it (room id +1 E, -1 W, +`$10` S, -`$10` N; not range-checked, the neighbour's door type not looked at), so it stays open from both sides. [`Door_Update`](src/bank5/update-doors.asm#L60) [`Door_NextRoomIdOffsets`](src/bank5/lay-out-doors.asm#L365) [`RoomFlags_SetDoorOpened`](src/bank5/shutters-door-flags.asm#L41)
- A shutter that opens only enters `CurOpenedDoors`: nothing is saved, and it is closed again on the next visit. [`Door_Update`](src/bank5/update-doors.asm#L60)
- Entering a room through a side that is in `CurOpenedDoors` issues a close command for it: Link's `ObjTimer` becomes `$30` whatever the type, then only a shutter closes (flag bit cleared, bit removed 8 frames later); a key door or bombable wall stays open. [`Room_EnterCommon@Method2`](src/bank5/enter-room.asm#L211) [`Door_Update`](src/bank5/update-doors.asm#L60) [`RoomFlags_ClearDoorOpened`](src/bank5/shutters-door-flags.asm#L52)
- Key doors and shutters request the door sound on both steps; a bombable wall requests none. [`Door_PrepareTransferRecords`](src/bank5/update-doors.asm#L217)

**Shutters**
- Shutters open only while `ShutterTrigger` is nonzero; each frame the first closed shutter in the order up, down, left, right gets an open command, one at a time. [`Shutter_OpenTriggered`](src/bank5/shutters-door-flags.asm#L68)
- `ShutterTrigger` is cleared on the first frame no closed shutter is left, so a shutter that closes behind Link needs a new trigger. [`Shutter_OpenTriggered`](src/bank5/shutters-door-flags.asm#L68)
- `ShutterTrigger` is set by secret triggers 1, 3, 4, 6 and 7 (next section), by paying the life-or-money man, and by the level 9 man's init when `InvTriforce` = `$FF`. [`Shutter_SetTrigger`](src/bank5/underworld-secrets.asm#L158) [`Person_LifeOrMoneyState2Choose`](src/bank1/life-or-money.asm#L121) [`Person_InitUnderworldLevel9`](src/bank1/underworld-person-init.asm#L172)

**Bomb walls**
- A bomb tests for a wall once, on the frame its blast ends (state `$13` -> `$14`, `$48` frames after its first update), not during the blast. [`Bomb_TryOpenWall`](src/bank7/bomb.asm#L136) [`Bomb_StateTimes`](src/bank7/bomb.asm#L55)
- The bomb's own `ObjX` and `ObjY` must each be less than `$18` pixels from a hotspot: up wall (`$78`,`$5D`), down (`$78`,`$BD`), left (`$20`,`$8D`), right (`$D0`,`$8D`); only the first hotspot in range (searched right, left, down, up) is used. [`Bomb_WallHotspotsX`](src/bank7/bomb.asm#L60) [`Bomb_WallHotspotsY`](src/bank7/bomb.asm#L64)
- It opens the wall only if that side is not already open, no door command is running and the side's type is 4; never in the overworld or in a cellar (mode 9). [`Bomb_TryOpenWall`](src/bank7/bomb.asm#L136)
- A passing test sets the open command; the wall is passable 8 frames later. [`Bomb_TryOpenWall`](src/bank7/bomb.asm#L136) [`Door_Update`](src/bank5/update-doors.asm#L60)

# Push blocks and room secrets

**Dungeon push block**
- A dungeon room has a push block when attribute D bit 6 is set; the object is type `$68` in object slot `$0B`. [`Room_SetUpItemAndPushBlock`](src/bank5/create-room-objects.asm#L51) [`PushBlock_FindAndCreate`](src/bank5/uw-floor-push-block.asm#L234)
- Q1 rooms with the bit (level: room, from `data/levelblock-uw-*.md`; rooms the submenu map does not show included): 1: `$22`, `$42`; 3: `$3B`; 4: `$12`, `$32`; 5: `$05`, `$06`, `$64`; 6: `$09`, `$38`, `$3A`; 7: `$0D`, `$1A`, `$29`; 8: `$1F`, `$7C`; 9: `$03`, `$04`, `$05`, `$07`, `$10`, `$20`, `$30`, `$37`, `$55`, `$61`, `$74`. [`Room_SetUpItemAndPushBlock`](src/bank5/create-room-objects.asm#L51)
- The block is the first block tile (`$B0`) found left to right in the floor row at Y = `$90` (X `$20`-`$C0`); with none found it is put at X = `$D0`. [`PushBlock_FindAndCreate`](src/bank5/uw-floor-push-block.asm#L234)
- It cannot move until the room has no monsters left (`IsRoomAllDead` nonzero); Bubbles (`$2B`-`$2D`) and types `$49` and up (traps, persons, shots) do not count as monsters. [`PushBlock_State0Idle`](src/bank4/push-block.asm#L74) [`Secret_UpdateRoomAllDead`](src/bank5/underworld-secrets.asm#L124)
- Link must be exactly in line (same X, or his Y + 3 = block Y), at most `$10` pixels away on the other axis, with `ObjInputDir` exactly the one direction toward the block. [`PushBlock_State0Idle`](src/bank4/push-block.asm#L74) [`PushBlock_PushDirs`](src/bank4/push-block.asm#L44)
- The push must hold for `$10` consecutive frames; any frame failing a condition restarts the count. [`PushBlock_State0Idle`](src/bank4/push-block.asm#L74) [`PushBlock_ObjPushTimer`](src/ObjVars.inc#L30)
- The block moves in whichever of the four directions Link pushes, `$10` pixels at q-speed `$20` (0.5 pixel per frame, `$20` frames); a block moves once per room entry. [`PushBlock_State1Moving`](src/bank4/push-block.asm#L165) [`PushBlock_State2Done`](src/bank4/push-block.asm#L191)
- On arrival the "secret found" tune plays and `BlockPushComplete` becomes 1, which secret triggers 4 and 5 read; with any other trigger the push has no effect. [`PushBlock_State1Moving`](src/bank4/push-block.asm#L165) [`BlockPushComplete`](src/Variables.inc#L389)
- While a block, rock or gravestone slides, Link cannot move if he is less than `$10` pixels from it on both axes. [`Link_CheckSlidingTileObjBlocking`](src/bank1/tile-block-power-triforce.asm#L45)
- `IsRoomAllDead` is incremented every frame the room is empty, so it reads 0 one frame in 256 (the push count restarts then). [`Secret_UpdateRoomAllDead`](src/bank5/underworld-secrets.asm#L124)

**Dungeon secret triggers** (attribute F bits 2-0, run every frame of dungeon play)
- `0`: none. [`Secret_RunTrigger_JumpTable`](src/bank5/underworld-secrets.asm#L105)
- `1`: no monsters left -> shutters open. [`Secret_Trigger1AllDead`](src/bank5/underworld-secrets.asm#L152)
- `2`: ringleader: once slot `$01` is empty or holds a type >= `$53`, every object of type `$01`-`$52` in slots `$01`-`$0C` in the normal metastate starts its death sparkle; no shutter opens. [`Secret_Trigger2Ringleader`](src/bank5/underworld-secrets.asm#L176)
- `3`: Triforce of Power taken (`InvPowerTriforce` nonzero) -> shutters open; the room item starts hidden. [`Secret_Trigger3PowerTriforce`](src/bank5/underworld-secrets.asm#L255) [`Room_SetUpItemAndPushBlock`](src/bank5/create-room-objects.asm#L51)
- `4`: block pushed -> shutters open. [`Secret_Trigger4BlockShutters`](src/bank5/underworld-secrets.asm#L215)
- `5`: block pushed -> on that frame the tile object moves to X `$D0`, Y `$60` and stairs tiles (`$70`-`$73`) appear there, always at that spot. [`Secret_Trigger5BlockStairs`](src/bank5/underworld-secrets.asm#L226)
- `6`: slot `$01` empty (the life-or-money man gone) -> shutters open. [`Secret_Trigger6LifeOrMoney`](src/bank5/underworld-secrets.asm#L265)
- `7`: no monsters left -> shutters open and the hidden room item appears (if not already taken), with the "item appears" tune. [`Secret_CheckUnderworld`](src/bank5/underworld-secrets.asm#L61)

**Overworld secrets (tile objects in slot `$0B`)**
- A room has at most one tile object, made from a placeholder square of its layout: rock `$62`, rock wall `$63`, tree `$64`, gravestone `$65`. [`RoomLayout_CheckTileObj`](src/bank5/layout-room-ow.asm#L267) [`RoomLayout_TileObjTypes`](src/bank5/layout-room-ow.asm#L59)
- Attribute F bits 7-6 say whose secret it is (0 both quests, 1 Q1 only, 2 Q2 only); in the other quest the tile object does nothing. [`Secret_IsForOtherQuest`](src/bank4/rock-wall-tree-secrets.asm#L331)
- Rock (needs `InvBracelet`) and gravestone (needs nothing): Link's X = object X, |Link Y + 3 - object Y| < `$11`, vertical input exactly toward it; no push timer; they move only up or down, `$10` pixels in 32 frames. [`Rock_Update`](src/bank4/rock-wall-tree-secrets.asm#L62) [`Rock_PushDirs`](src/bank4/rock-wall-tree-secrets.asm#L51)
- After the push, stairs appear at the room's shortcut position (level info position picked by attribute F bits 5-4), not where the object stood, and room flag `$20` is set. [`Rock_Update`](src/bank4/rock-wall-tree-secrets.asm#L62) [`Room_GetShortcutOrItemXYForRoomId`](src/bank1/bomb-candle-wield.asm#L280)
- Rock wall: opens to a cave mouth when a bomb in slot `$10` or `$11` is in its blast state `$13` with midpoints less than `$10` pixels apart on both axes. [`RockWall_Update`](src/bank4/rock-wall-tree-secrets.asm#L214) [`Collision_CheckTileObjWeapon`](src/bank4/rock-wall-tree-secrets.asm#L304)
- Tree: becomes stairs when a standing fire (state `$22`) in slot `$10` or `$11` has its timer below 2 (about to go out) within the same distance. [`Tree_Update`](src/bank4/rock-wall-tree-secrets.asm#L249)
- Rock wall, tree and recorder pond secrets set room flag `$80`; such a room is laid out with the stairs or cave mouth already open and no tile object. [`Secret_RevealAndFlagTileObj`](src/bank4/rock-wall-tree-secrets.asm#L287) [`RoomLayout_LayOutRoomOrCaveOW`](src/bank5/layout-room-ow.asm#L141)
- Room flag `$20` puts the stairs back at the shortcut position when the room is laid out, except where the tile there is a gravestone (`$BC`). [`Secret_RestoreStairsOW`](src/bank5/ow-shortcut-map-square.asm#L54)
- Stairs under an Armos are not saved in the room flags. [`Armos_InitOrFlyingGhiniFadeIn`](src/bank4/ghini-armos.asm#L154)

# Stairs, cellars and caves

**Starting a transition from a tile**
- The check runs once a frame after Link's move, only in mode 5, with grid offset 0, `UndergroundExitType` = 0, X a multiple of `$10` and Y low nibble `$D`; the tile read is the one at (X, Y + `$0B`). [`Warp_CheckUnderLink`](src/bank5/warps-stairs-caves.asm#L55) [`Link_EndMoveAndAnimate`](src/bank7/link-animate-ladder.asm#L150)
- `UndergroundExitType` stays nonzero after Link comes out of a cave, cellar or level until he completes one grid step in mode 5, so the entrance he stands on does not take him back in. [`UndergroundExitType`](src/Variables.inc#L99) [`Link_EndMoveAndAnimate`](src/bank7/link-animate-ladder.asm#L150)
- Dungeon: tiles `$70`-`$73` are stairs; the destination is the first cellar in the level's cellar list whose attribute A or B equals the current room; mode `$10` then mode 9. [`Warp_CheckUnderLink`](src/bank5/warps-stairs-caves.asm#L55) [`LevelInfo_CellarRoomIdArray`](src/Variables.inc#L726)
- Overworld: tile `$24` (black opening), `$88` or stairs `$70`-`$73`; the room's cave index (attribute B bits 7-2) picks the target: 1-9 load that level (mode 2), `$14` the shortcut cave (mode `$C`), any other value a cave (mode `$B`). [`Warp_HandleOW`](src/bank5/warps-stairs-caves.asm#L178)
- Every transition first banks the room's kill count and passes through mode `$10`; all sound is silenced and `FluteTimer` cleared except when the target is a cellar. [`Warp_StartStairsMode`](src/bank5/warps-stairs-caves.asm#L131) [`RoomHistory_SaveKillCount`](src/bank5/warps-stairs-caves.asm#L151)
- Starting the transition clears Link's state, knockback and invincibility timer. [`Mode_EndPrepareMode`](src/bank5/cave-cellar-exits.asm#L188)

**Mode `$10` (stairs)**
- On tile `$24` Link sinks `$10` pixels, 1 pixel every 4 frames (61-64 frames), drawn behind the background; on any other tile (stairs) the mode ends on its first update frame. [`Mode10Stairs_Init`](src/bank5/mode-inits-obj-lists.asm#L82) [`Mode10Stairs_Update_Full`](src/bank5/mode8-continue-mode10-stairs.asm#L176)
- It ends by setting `GameMode` = `ModeAfterStairs` (9, `$B`, `$C` or 2). [`Mode10Stairs_Update_Full`](src/bank5/mode8-continue-mode10-stairs.asm#L176) [`ModeAfterStairs`](src/Variables.inc#L103)

**Mode 9 (cellar)**
- A cellar is a room of the level block with layout id `$3E` (tunnel with two ladders) or `$3F` (item cellar). [`Mode9Cellar_Init_LayOutCellar`](src/bank5/layout-room-ow.asm#L418)
- Link appears at Y = `$41` facing down and walks `$1C` pixels to Y = `$5D`; X = `$30` if he came from the room named by the cellar's attribute A, else `$C0`. [`Mode9Cellar_Init_Sub8_EnterCellar`](src/bank5/cave-cellar-enter.asm#L276) [`Mode9Cellar_LinkStartXs`](src/bank5/cave-cellar-enter.asm#L262) [`Mode9Cellar_Init_Sub9_WalkIn`](src/bank5/cave-cellar-enter.asm#L317)
- The cellar's own monster list is discarded; four blue Keese are made instead. [`Room_EnterCommon`](src/bank5/enter-room.asm#L143) [`Spawn_AssignPositions`](src/bank5/spawn-positions.asm#L64)
- Leaving: Link's Y < `$40` with Up held; destination room = the cellar's attribute A if his X < `$80`, else attribute B. [`Warp_CheckSubroomExit`](src/bank5/cave-cellar-exits.asm#L43)
- He comes out at X = attribute C high nibble * `$10`, Y = low nibble * `$10` + `$0D`, the same spot whichever room he goes to, facing down; the destination is marked visited. [`Warp_CheckSubroomExit`](src/bank5/cave-cellar-exits.asm#L43) [`ModeALeaveCellar_GoFromCellar`](src/bank5/cave-cellar-exits.asm#L173) [`Room_EnterCommon@FaceDown`](src/bank5/enter-room.asm#L203)
- Cellar entry and exit fade the palettes (4 steps 10 frames apart each way); caves do not fade. [`Mode9Cellar_Init_Sub1_FadeOutLevel`](src/bank5/cave-cellar-enter.asm#L135) [`ModeALeaveCellar_Init_Sub2_FadeOutCellar`](src/bank5/cave-cellar-enter.asm#L162)

**Mode `$B` (cave)**
- Every cave uses one fixed layout; Link starts at X = `$70`, Y = `$DD` facing up and walks 8 pixels to Y = `$D5`. [`ModeBCave_Init_LayOutCave`](src/bank5/layout-room-ow.asm#L398) [`ModeBCave_Init_Sub7_EnterCave`](src/bank1/modeb-cave-save-sprites.asm#L104) [`ModeBCave_Init_Sub8_WalkIn`](src/bank5/cave-cellar-enter.asm#L246)
- In a cave Link cannot move up while his Y < `$8E`, whether or not a person is there. [`Person_BlockLinkMovingUp`](src/bank1/modeb-cave-save-sprites.asm#L289) [`Warp_CheckSubroomExit`](src/bank5/cave-cellar-exits.asm#L43)
- Cave edge exit: reaching a screen edge bound while moving toward it leaves the cave (mode `$A`, `RoomId` unchanged); the edge test is called directly, without the mode 5, ladder and grid offset 0 conditions of normal play. [`Warp_CheckCaveEdgeExit`](src/bank5/cave-cellar-exits.asm#L209) [`Link_CheckScreenEdge`](src/bank7/walker-collision-screen-edge.asm#L243)

**Mode `$C` (shortcut stairs cave)**
- Same init as a cave except the layout (three stairs). [`ModeCShortcut_Init_JumpTable`](src/bank5/room-history-ow.asm#L69) [`ModeCShortcut_Init_LayOutShortcut`](src/bank5/layout-room-ow.asm#L409)
- With grid offset 0 and Y = `$9D`, X = `$50`, `$80` or `$B0` selects stairs 1, 2 or 3; destination = entry (i + n) AND 3 of the overworld's shortcut room list, i = index of the current room, n = stairs number. [`Warp_CheckSubroomExit`](src/bank5/cave-cellar-exits.asm#L43) [`LevelInfo_CellarRoomIdArray`](src/Variables.inc#L726)
- The destination room gets room flag `$20`, so its stairs are drawn when it is laid out. [`ModeALeaveCellar_GoFromCellar`](src/bank5/cave-cellar-exits.asm#L173) [`RoomFlags_MarkVisited`](src/bank7/next-room-room-flags.asm#L120) [`Secret_RestoreStairsOW`](src/bank5/ow-shortcut-map-square.asm#L54)

**Mode `$A` (leaving) and arriving back**
- Mode `$A` init has 11 submodes (sprites, palettes, fade out, lay out the room, 22 tile rows, attributes, fade in) and ends by switching to mode 4. [`ModeALeaveCellar_Init_JumpTable`](src/bank5/cave-cellar-enter.asm#L79) [`ModeALeaveCellar_Init_SubA_GoToMode4`](src/bank5/cave-cellar-enter.asm#L195)
- Back on the overworld Link is put at the room's exit spot: X = attribute A high nibble * `$10`, Y = (attribute F bits 2-0) * `$10` + `$4D`, facing down. [`Room_EnterCommon`](src/bank5/enter-room.asm#L143)
- If he went in through an opening tile (`$24`) he starts `$10` pixels lower and rises to that Y at 1 pixel every 4 frames; if he took stairs he just appears there. [`Room_EnterCommon`](src/bank5/enter-room.asm#L143) [`UndergroundEntranceTile`](src/Variables.inc#L106) [`Mode4Enter_StepOutside`](src/bank7/update-mode-dispatch.asm#L247)

**Levels**
- A level entrance sets `CurLevel` to the cave index and `LevelEntranceRoomId` to the overworld room; caves and cellars do not touch it. [`Warp_HandleOW`](src/bank5/warps-stairs-caves.asm#L178) [`LevelEntranceRoomId`](src/Variables.inc#L463)
- A level starts with Link at X = `$78`, Y = `LevelInfo_StartY`, facing up, in `LevelInfo_StartRoomId`. [`Mode3Unfurl_Init_Sub8_LayOutRoomPlaceLink`](src/bank5/mode3-mode6-init.asm#L129) [`Mode3Unfurl_Init_Sub1`](src/bank7/init-mode-dispatch.asm#L294)
- Walking out of a level is a room change whose next room id has bit 7 set (down from row 7, up from row 0): the overworld is loaded through mode 2 with `UndergroundExitType` = 2. [`Room_CalcNextRoomForDoor`](src/bank5/next-room-calc.asm#L76) [`Mode_LeaveLevelToOW`](src/bank5/next-room-calc.asm#L107)
- The overworld then starts in `LevelEntranceRoomId` if it is not `$FF` (it is consumed: set back to `$FF`), else in the overworld's start room. [`Mode3Unfurl_Init_Sub1`](src/bank7/init-mode-dispatch.asm#L294)

# Lost Woods and Lost Hills (mazes)

The mazes are rules applied to the next-room calculation of an overworld room change; nothing in the rooms themselves is special.

- Lost Woods = overworld room `$61`; the sequence is up, left, down, left (`$08`, `$02`, `$04`, `$02`). [`Maze_LostWoodsDirs`](src/bank1/mazes-lost-woods-hills.asm#L36)
- Lost Hills = overworld room `$1B`; the sequence is up four times. [`Maze_LostHillsDirs`](src/bank1/mazes-lost-woods-hills.asm#L41)
- The direction tested is Link's facing direction `ObjDir` at the room change, once per overworld room scroll. [`Maze_CheckStep`](src/bank1/mazes-lost-woods-hills.asm#L56) [`Room_CalcNextRoomForDoor`](src/bank5/next-room-calc.asm#L76)
- A correct step other than the fourth adds 1 to the step counter and makes the next room the same room: the scroll brings the maze room up again. [`Maze_CheckStep@AdvanceMaze`](src/bank1/mazes-lost-woods-hills.asm#L69)
- The fourth correct step requests the "secret found" tune and keeps the computed neighbour: room `$60` (left of the Lost Woods) or `$0B` (above the Lost Hills); the counter stays 3. [`Maze_CheckStep@PlaySecretTune`](src/bank1/mazes-lost-woods-hills.asm#L113)
- A wrong step sets the counter to 0 and repeats the room; the wrong exit itself is not counted as step 0 even if its direction is the first of the sequence. [`Maze_CheckStep@Reset`](src/bank1/mazes-lost-woods-hills.asm#L92)
- Free exits: right out of the Lost Woods (to `$62`) and left out of the Lost Hills (to `$1A`) work at any step and leave the counter unchanged. [`Maze_CheckStep@Mismatch`](src/bank1/mazes-lost-woods-hills.asm#L86) [`Maze_CheckStep@CheckMountainMaze`](src/bank1/mazes-lost-woods-hills.asm#L97)
- The counter is `MazeStep` (`$52F`), values 0-3. [`MazeStep`](src/Variables.inc#L476)
- It is reset to 0 by a room change out of any other overworld room and by a wrong step; dungeon room changes do not run the maze code and no other code writes it, so it survives caves, cellars and levels. [`Maze_CheckStep@ResetMazeStep`](src/bank1/mazes-lost-woods-hills.asm#L81)
- Entering the maze room from outside always starts at step 0, because the room change that brought Link in left a non-maze room. [`Maze_CheckStep@ResetMazeStep`](src/bank1/mazes-lost-woods-hills.asm#L81)

# Special-cased rooms

Every room id, layout id or level number the code compares by value. Overworld room ids unless a level is named; per-room data (door types, items, secrets) is in `data/`.

**Overworld rooms**
- `$3F` and `$55`: the docks; the tile object in slot `$0B` is always the dock (type `$61`); dock X is `$80` in `$55` and `$60` otherwise. [`Room_SetUpTileObjOW`](src/bank5/enter-room.asm#L429) [`Dock_Update`](src/bank4/dock-raft.asm#L39)
- `$1F`: for Link only, a vertical tile test at X = `$80` with Y < `$56` reports a walkable tile, so he walks through the wall in that column. [`Tile_GetAtObjOffset@SetTile`](src/bank7/tile-collision.asm#L199)
- `$5F`: the only overworld room item, a heart container (`$1A`) at X `$C0`, Y `$90`, present only in mode 5. [`Room_SetUpItemAndPushBlock`](src/bank5/create-room-objects.asm#L51)
- `$24`: the Armos at X `$E0`, Y `$80` uncovers the power bracelet (item `$14`) as the room item. [`Armos_SecretRoomIds`](src/bank4/ghini-armos.asm#L138) [`Armos_InitOrFlyingGhiniFadeIn`](src/bank4/ghini-armos.asm#L154)
- `$0B`, `$1C`, `$22`, `$34`, `$3D`, `$4E`: the Armos at Y `$80` and X `$B0`, `$B0`, `$30`, `$40`, `$90`, `$A0` leaves stairs when it wakes; in both quests. [`Armos_SecretRoomIds`](src/bank4/ghini-armos.asm#L138) [`Armos_SecretXs`](src/bank4/ghini-armos.asm#L143)
- `$42` (Q1 only) and `$06`, `$29`, `$2B`, `$30`, `$3A`, `$3C`, `$58`, `$60`, `$6E`, `$72` (Q2 only): recorder secret rooms; the recorder drains the pond and reveals stairs at X `$60`, Y `$90` instead of calling the whirlwind. [`Flute_SecretRoomIdsOW`](src/bank7/flute.asm#L36) [`FluteSecret_RevealPondStairs`](src/bank7/flute-secret-pond.asm#L157)
- `$17`, `$18`, `$19`, `$27`, `$4F`, `$5F`: the only overworld rooms where the ladder can be laid. [`Ladder_RoomIdsOW`](src/bank7/link-animate-ladder.asm#L76)
- `$22`: the stairs and entrance check accepts X a multiple of 8 instead of `$10`. [`Warp_CheckUnderLink`](src/bank5/warps-stairs-caves.asm#L55)
- `$0F`: entering it with `UndergroundExitType` = 0 (walking in, not coming out of underground) requests the "secret found" tune on every entry. [`Mode5Play_Init`](src/bank7/init-mode5-play.asm#L60)
- `$61` and `$1B`: the Lost Woods and Lost Hills. [`Maze_CheckStep`](src/bank1/mazes-lost-woods-hills.asm#L56)
- `$44`: every cave takes its palette selectors from this room's attributes. [`ModeBCave_Init_Sub5_FillAttrsTransferTopHalf`](src/bank5/cave-cellar-enter.asm#L215)
- `$36`, `$3B`, `$73`, `$44`, `$0A`, `$21`, `$41`, `$6C`: the rooms the whirlwind scrolls from for levels 1-8; Link arrives one room east (`$37`, `$3C`, `$74`, `$45`, `$0B`, `$22`, `$42`, `$6D`), same table in both quests. [`Whirlwind_PrevRoomIds`](src/bank1/whirlwind.asm#L81)
- `$0B`, `$3C`, `$74` (immediate stores) and `$0E`, `$0F`, `$22`, `$34`, `$3C`, `$45`, `$74`, `$0B` (table): rooms whose level block bytes the Q2 load patches. [`Mode2Load_Update_Full`](src/bank6/level-load-mode2.asm#L297) [`Quest_Q2OverworldPatchOffsets`](src/bank6/quest2-replacements.asm#L49)
- Level entrances are not tested by room id: a room is an entrance when its cave index (attribute B bits 7-2) is 1-9; the shortcut cave is cave index `$14`. [`Warp_HandleOW`](src/bank5/warps-stairs-caves.asm#L178)
- `LevelEntranceRoomId` = `$FF` means "no entrance remembered": the overworld then starts in the level info's start room. [`Mode3Unfurl_Init_Sub1`](src/bank7/init-mode-dispatch.asm#L294)

**Room ids in general**
- A next room id with bit 7 set (`$80` and up) is "off the map": the overworld is loaded instead. [`Room_CalcNextRoomForDoor`](src/bank5/next-room-calc.asm#L76)
- A cleared room history holds six zeroes, which is also the id of room `$00`: that room counts as already in the history until the entries are overwritten. [`RoomHistory_Clear`](src/bank7/init-mode-dispatch.asm#L271) [`Room_RunEntryTasks`](src/bank7/init-mode5-play.asm#L169)

**Layout ids (attribute D bits 5-0)**
- `$3E`, `$3F`: cellars (even = tunnel, odd = item cellar); the death sequence skips the door redraw for them. [`Mode9Cellar_Init_LayOutCellar`](src/bank5/layout-room-ow.asm#L418) [`Mode11Death_Init_Sub1`](src/bank5/mode11-death-init.asm#L73)
- `$21`: a push block would be put at (`$40`, `$80`) without searching; no room in the level data combines this layout with the push block flag. [`PushBlock_FindAndCreate`](src/bank5/uw-floor-push-block.asm#L234)
- `$24` and `$23`: the dungeon layouts whose statues shoot fireballs (4 and 2 statues). [`Statue_RoomLayoutIds`](src/bank4/statues.asm#L45)

**Level numbers (`CurLevel`)**
- 0 is the overworld; most dungeon-only and overworld-only code branches on it. [`Mode5Play_UpdateWorld`](src/bank7/mode5-play-update.asm#L129)
- Levels 1, 2, 5, 7 have persons that do more than talk; levels 3, 4, 6, 8, 9 have talk-only persons (compares against 3, 5, 7). [`Person_UpdateUnderworld_Full`](src/bank1/underworld-person-update.asm#L44)
- Person init and text tables are chosen per level: 1, 2, 5, 7 / 3, 4, 6, 8 / 9. [`Person_InitUnderworld_JumpTable`](src/bank1/underworld-person-init.asm#L53)
- Level 9: the man of type `$4B` opens the shutters and vanishes when `InvTriforce` = `$FF`. [`Person_InitUnderworldLevel9`](src/bank1/underworld-person-init.asm#L172)
- Level 9: the Triforce marker on the status bar map always blinks. [`StatusBar_UpdatePositionMarker`](src/bank1/map-marker-curtain.asm#L72)
- Level 9: compass and map use their own inventory bytes (bit 0 of the slot two further on) instead of bit (level - 1). [`Item_TakeClass0PerLevel`](src/bank1/take-item.asm#L421) [`Inv_HasCompass`](src/bank5/submenu-map.asm#L143) [`Inv_HasMap`](src/bank5/submenu-map.asm#L153)
- Level song request by level: overworld `$01`, levels 1-8 `$40`, level 9 `$20`. [`Song_LevelRequests`](src/bank7/update-mode-dispatch.asm#L218)
- Mode `$12` (end of level) is the one mode in which the door update does nothing. [`Door_Update`](src/bank5/update-doors.asm#L60)

# Ladder and raft

The items themselves (ids, pickup, inventory bytes) are in `items.md` under `# Ladder` and `# Raft`; this section gives the room and tile rules.

**Ladder** (`InvLadder`, object type `$5F`)
- Rooms: every dungeon room; in the overworld only rooms `$17`, `$18`, `$19`, `$27`, `$4F`, `$5F`. [`Ladder_RoomIdsOW`](src/bank7/link-animate-ladder.asm#L76) [`Link_EndMoveAndAnimate@CheckLadderRoom`](src/bank7/link-animate-ladder.asm#L190)
- It is laid only in mode 5 with Link on a grid point, outside a doorway, not halted, no ladder already out, a free monster slot, and `ObjInputDir` nonzero and equal to `ObjDir`. [`Link_EndMoveAndAnimate@CheckLadder`](src/bank7/link-animate-ladder.asm#L209) [`Link_EndMoveAndAnimate@SetUpLadder`](src/bank7/link-animate-ladder.asm#L246)
- Tile ahead of Link: dungeon `$F4` (water); overworld `$8D`-`$98` as the tile lookup returns them, which has already turned the walkable `$8D` and `$91` into `$26`. [`Link_EndMoveAndAnimate@CheckWaterOW`](src/bank7/link-animate-ladder.asm#L240) [`Tile_WalkableTilesOW`](src/bank7/tile-collision.asm#L43)
- The ladder takes the highest free slot of `$0B`..`$01`, one cell ahead of Link, facing as he does; `LadderSlot` holds the slot (0 = none). [`Link_EndMoveAndAnimate@SetUpLadder`](src/bank7/link-animate-ladder.asm#L246) [`LadderSlot`](src/Variables.inc#L104)
- It has no timer: it is put away when Link leaves its line, is more than `$10` pixels from it, or is exactly `$10` away after crossing or facing another way. On that frame a new ladder is laid at once, one square farther, when Link stands on a grid point with water ahead and the direction still held (trace: level 4 room `$21`, overworld room `$5F` row Y = `$95`). [`Ladder_CheckAndDraw@StashLadder`](src/bank5/ladder.asm#L110) [`Link_EndMoveAndAnimate@SetUpLadder`](src/bank7/link-animate-ladder.asm#L246)
- On the ladder Link may only go on in its direction or back the way he came; other input stops him. A perpendicular press between grid points is first turned into a reversal by the rules between grid points, and the ladder lets that through as the way back (case C): that is how the ladder clip starts. [`Ladder_CheckAndDraw@HandleInput`](src/bank5/ladder.asm#L125) [`Link_SetInputDir@ReverseDir`](src/bank5/link-handle-input.asm#L404)
- Only the grid point at distance 0 is decided by the tile ahead; at every other distance, facing the ladder's way with a direction held, Link moves on whatever the tile test or the room bound said (case A). One skip of the distance-0 test therefore crosses two or three squares (`# Grid-point skip: block clip, ladder clip and their run-up`, "Ladder clip"). [`Ladder_CheckAndDraw@HandleInput`](src/bank5/ladder.asm#L125)
- A ladder laid upward with exactly Up held repeats the tile test 8 pixels higher and lets Link on only if that tile is walkable. [`Ladder_CheckAndDraw`](src/bank5/ladder.asm#L44)
- On the frame the ladder is put away Link does not move. [`Ladder_CheckAndDraw@StashLadder`](src/bank5/ladder.asm#L110) [`Walker_Move`](src/bank7/walker-move.asm#L64)
- While a ladder is out the screen-edge room change is not tested: at X = `$F0` facing right over water the next ladder lands at X = `$00` and Link walks onto it to the left edge of the same room (trace: overworld room `$5F`). [`Walker_CheckTileCollision`](src/bank7/walker-collision-screen-edge.asm#L89) [`Ladder_LinkOffsetsX`](src/bank7/link-animate-ladder.asm#L58)
- Every room entry and exit clears `LadderSlot`. [`Obj_ResetWeaponSlotsAndLadder`](src/bank5/mode3-mode6-init.asm#L263)

**Raft** (`InvRaft`, dock tile object type `$61`)
- Docks exist only in overworld rooms `$3F` (dock X `$60`) and `$55` (dock X `$80`); without the raft the dock does nothing. [`Room_SetUpTileObjOW`](src/bank5/enter-room.asm#L429) [`Dock_Update`](src/bank4/dock-raft.asm#L39)
- A ride starts only when Link's X equals the dock X exactly and his Y is exactly `$7D` (standing at the dock: ride up) or `$3D` (at the top screen edge: ride down). [`Dock_Update`](src/bank4/dock-raft.asm#L39)
- Link is halted (`ObjState` `$40`) for the whole ride and faces up or down; the "secret found" tune plays at the start. [`Dock_Update`](src/bank4/dock-raft.asm#L39) [`Dock_RaftLinkDirs`](src/bank4/dock-raft.asm#L28)
- Raft and Link move 1 pixel per frame: 64 frames up (`$7D` to `$3D`), 66 frames down (`$3D` to `$7F`). [`Dock_Update@HandleOtherStates`](src/bank4/dock-raft.asm#L94) [`Dock_Update@State1`](src/bank4/dock-raft.asm#L117)
- Riding up ends at Y `$3D` with a room change into the room above; riding down ends at Y `$7F` with grid offset 2, so the ride up does not restart at once. [`Dock_Update`](src/bank4/dock-raft.asm#L39) [`Mode5Play_GoToNextMode`](src/bank7/walker-collision-screen-edge.asm#L281)

# Monster spawning, edge spawns and respawn (room history)

Monsters are made when a room is entered (modes 4, 9, `$B`, `$C`) from two attribute bytes of the room. Nothing refills a room while Link stays in it, apart from the special spawners listed below.

**Room entry: list and count**
- Every room entry zeroes `$0300`-`$051F`, then resets object slots `$0B`..`$01`: uninitialized, state 0, no stun or knockback, `ObjMetastate` 1 (spawn cloud), q-speed `$20`. [`Room_EnterCommon@ResetObjLoop`](src/bank5/enter-room.asm#L265)
- Monster list id = attribute byte C bits 5-0, plus `$40` when attribute byte D bit 7 is set (`$00`-`$7F`). [`Room_EnterCommon`](src/bank5/enter-room.asm#L143)
- Monster count = one of the level's four counts, chosen by attribute byte C bits 7-6; ids `$32`-`$61` (bosses and other single objects) always get count 1. [`Room_EnterCommon`](src/bank5/enter-room.asm#L143) [`LevelInfo_FoeCounts`](src/Variables.inc#L716)
- Id `$01`-`$61` is an object type: slots `$01`..count all get that type. [`Room_EnterCommon@StoreObjCount`](src/bank5/enter-room.asm#L331)
- Id `$62`-`$7F` selects one of 30 lists (index = id - `$62`): the first count bytes of the list go to slots `$01`..count; a 0 byte leaves a slot empty; lists have no end marker, so a count above the list's length reads on into the next list. [`Room_EnterCommon@PlaceList`](src/bank5/enter-room.asm#L354) [`Spawn_ObjListAddrs`](src/bank5/mode-inits-obj-lists.asm#L190) [`Spawn_ObjLists`](src/bank5/mode-inits-obj-lists.asm#L185)
- Id 0 or count 0 makes no monsters; the final count is kept in `RoomObjCount` and slot `$01`'s type in `RoomObjTemplateType`. [`Room_EnterCommon@StoreObjTemplate`](src/bank5/enter-room.asm#L378) [`RoomObjCount`](src/Variables.inc#L298) [`RoomObjTemplateType`](src/Variables.inc#L305)
- A cellar (mode 9) ignores its list: four blue Keese (`$1B`) go to slots `$01`-`$04` at X `$20`, `$60`, `$90`, `$D0`, all at Y `$9D` (the Y table is read with the wrong index register, so only its first entry is used). [`Spawn_AssignPositions@AssignSpecialPositions`](src/bank5/spawn-positions.asm#L128) [`Mode9Cellar_KeeseXs`](src/bank5/spawn-positions.asm#L44) [`Mode9Cellar_KeeseYs`](src/bank5/spawn-positions.asm#L49)
- A cave (modes `$B`, `$C`) empties slots `$01`-`$08` and puts the cave person, type `$6A` + (cave index - `$10`), in slot `$01`. [`Spawn_AssignPositions@InCave`](src/bank5/spawn-positions.asm#L155)
- The overworld tile object goes to slot `$0B`: the dock (`$61`) in rooms `$3F` and `$55`, else the rock, rock wall, tree, gravestone or armos the layout found. [`Room_SetUpTileObjOW`](src/bank5/enter-room.asm#L429)

**Spawn spots**
- Positions come from one of four 9-entry lists chosen by the lowest set bit of Link's direction on entry (right, left, down, up); every list keeps clear of the side Link enters by. [`Spawn_PosListAddrsLo`](src/bank5/mode-inits-obj-lists.asm#L108) [`Spawn_PosListFacingRight`](src/bank5/mode-inits-obj-lists.asm#L124) [`Spawn_PosListFacingLeft`](src/bank5/mode-inits-obj-lists.asm#L130) [`Spawn_PosListFacingDown`](src/bank5/mode-inits-obj-lists.asm#L136) [`Spawn_PosListFacingUp`](src/bank5/mode-inits-obj-lists.asm#L142)
- A spot byte is row in the high nibble, column in the low nibble: X = column * `$10`, Y = row * `$10` + `$0D`. [`Spawn_AssignPositions@LoopSpawnSpot`](src/bank5/spawn-positions.asm#L103)
- The walk through the list starts at `SpawnCycle` (0-8) and stores it back, so it carries over from room to room: the same room can hand out different spots on another visit. No random number is used. [`Spawn_AssignPositions`](src/bank5/spawn-positions.asm#L64) [`SpawnCycle`](src/Variables.inc#L460)
- Slots `$01`-`$09` all get a position whatever the count; an unsafe spot is skipped without advancing the slot. [`Spawn_AssignPositions@LoopSpawnSpot`](src/bank5/spawn-positions.asm#L103)
- A spot is unsafe when its tile is unwalkable (tile >= `RoomFirstUnwalkableTile`) or when it is closer than `$22` pixels to Link on both axes. [`Spawn_IsSafeSpot`](src/bank5/spawn-positions.asm#L189) [`Spawn_IsFarFromLink`](src/bank5/spawn-positions.asm#L208)
- The distance test uses 8-bit differences, so a real distance of `$DF` or more reads as `$21` or less (Link at X `$00`, spot at X `$F0` counts as `$10` apart). [`Spawn_IsFarFromLink`](src/bank5/spawn-positions.asm#L208)
- No spots are assigned when the list id is 0 or `$37` (Zelda), the count is 0, or the overworld room brings its monsters in from the screen edges. [`Spawn_AssignPositions`](src/bank5/spawn-positions.asm#L64)

**Spawn cloud**
- A slot flagged uninitialized only runs its init on its first update; while `ObjMetastate` is 1-3 the object's own update does not run and its collisions are not tested (it cannot be hit and cannot hurt Link). [`Obj_Update`](src/bank7/object-update-table.asm#L120) [`Mode5Play_UpdateWorld@LoopObject`](src/bank7/mode5-play-update.asm#L198)
- The first cloud step lasts as many frames as the slot number (`ObjTimer` = slot), the next two steps 6 frames each; then the monster acts, so higher slots appear later. [`Obj_Init@NormalSpawn`](src/bank7/object-init-table.asm#L122) [`Obj_AnimateMetastate`](src/bank7/object-animation.asm#L54) [`Obj_EndMetastate`](src/bank7/object-update-table.asm#L308)
- Type inits that replace the first step: Octoroks and Ghinis (slot + 1) * `$10` frames, Tektites 4, 8, `$14` or `$28` frames (random). [`Octorok_InitFast`](src/bank4/zora-ow-inits.asm#L219) [`Tektite_Init`](src/bank4/zora-ow-inits.asm#L174)
- No cloud at all: Boulders, Leevers, Peahats, a Zora, an Armos, a flying Ghini, every type `$53` and up, and monsters that enter from a screen edge. [`Boulder_Init`](src/bank4/zora-ow-inits.asm#L164) [`Leever_Init`](src/bank4/zora-ow-inits.asm#L194) [`Peahat_Init`](src/bank4/zora-ow-inits.asm#L243) [`Obj_Init@NormalSpawn`](src/bank7/object-init-table.asm#L122)

**Edge spawns**
- Only on the overworld, in rooms with attribute byte F bit 3; never for Zora `$11`, Armos `$1E`, Whirlwind `$2E`, Standing fire `$40` or types `$53` and up. The flying Ghini (`$22`) is not exempt. [`Obj_Init`](src/bank7/object-init-table.asm#L45) [`RoomAttrsF`](src/Variables.inc#L385)
- One monster enters at a time: while `MonstersFromEdgesLongTimer` is nonzero the monster stays uninitialized and is tried again on a later frame. [`Obj_Init@UninitMonsterFromEdge`](src/bank7/object-init-table.asm#L70) [`MonstersFromEdgesLongTimer`](src/Variables.inc#L60)
- Each attempt reloads the timer with 2 + (the slot's random byte AND 3) = 2-5 units of 10 frames, also when the attempt is refused. [`Obj_Init@InitMonsterFromEdge`](src/bank7/object-init-table.asm#L79)
- An attempt is refused when the edge cell is closer than `$22` pixels to Link on both axes; the cell's tile is not tested again. [`Spawn_IsFarFromLink`](src/bank5/spawn-positions.asm#L208)
- Position: X = cell column * `$10`, Y = cell row * `$10` - 3 (rows `$4`-`$E`, columns `$0`-`$F`). [`Obj_Init@InitMonsterFromEdge`](src/bank7/object-init-table.asm#L79) [`CurEdgeSpawnCell`](src/Variables.inc#L461)
- The cell search steps counterclockwise round the outer ring (down the left edge, right along the bottom, up the right edge, left along the top) from the stored cell and takes the first cell whose top left tile is below `$84`. [`EdgeSpawn_FindNextCell@LoopEdgeCell`](src/bank5/edge-spawn-cell.asm#L40)
- The cell persists between rooms and starts at `$40` (top left) after reset; no random number picks it. [`EdgeSpawn_FindNextCell`](src/bank5/edge-spawn-cell.asm#L35) [`Reset_ClearRam`](src/bank5/save-ram-clear-ram.asm#L81)

**Special spawns**
- Zora: overworld rooms with attribute byte A bit 3; one at a time, in the highest free slot, on a random water square (12 random candidates per frame; see enemies.md "# Zora"); each dive destroys it and a new one with full HP is made. [`Zora_TrySpawn`](src/bank4/zora-ow-inits.asm#L77) [`Zora_Update`](src/bank4/zora-ow-inits.asm#L301)
- A killed Zora does not come back until the room is entered again, and its death is not counted in the room's kill count. [`Zora_Update`](src/bank4/zora-ow-inits.asm#L301) [`Obj_EndMetastate`](src/bank7/object-update-table.asm#L308)
- Boulders: the spawner (type `$1F`) keeps at most 3 Boulders (`$20`) alive; a new one starts at Y `$40` at a random X in the screen half that holds the chase target, then the spawner waits 0-`$1F` frames. [`Boulder_UpdateSpawner`](src/bank4/tektite-boulder.asm#L62) [`ActiveBoulderCount`](src/Variables.inc#L431)
- Armos and flying Ghini: Link pushing into an armos square (tiles `$C0`-`$C3`) or gravestone square (`$BC`-`$BF`) at grid offset 0 makes type `$1E` or `$22` on that square in the highest free slot, fading in for `$3F` frames; none if an object already sits at that position. [`Spawn_CheckTouchedArmosOrGravestone`](src/bank1/armos-ghini-spawn.asm#L54)
- Whirlwind: the flute makes one per room visit in a free slot, at X 0 on Link's Y; in the destination room it is put in slot `$09`, whatever is there. [`Whirlwind_Summon`](src/bank1/whirlwind.asm#L243) [`Whirlwind_InitDropOffAndBeginUpdate`](src/bank1/whirlwind.asm#L360)

**Overworld room history**
- `RoomHistory` is a ring of six room ids; a room is added on entry only if it is not already there. It is zeroed when a level loads, at the start of mode 3 and in the continue menu. [`Room_RunEntryTasks`](src/bank7/init-mode5-play.asm#L169) [`RoomHistory_Clear`](src/bank7/init-mode-dispatch.asm#L271) [`RoomHistory`](src/Variables.inc#L533)
- Leaving a room (screen edge, stairs, cave or level entrance) banks its kills. [`RoomHistory_SaveKillCount`](src/bank5/warps-stairs-caves.asm#L151) [`Mode6Leave_Init`](src/bank5/mode3-mode6-init.asm#L214)
- Overworld room flags bits 0-2 = saved kill count: old count + `RoomKillCount`, capped at 7; 7 ("all dead") at once when the kills reach `RoomObjCount`, which includes a room that made no monsters. [`RoomHistory_SaveKillCountOW`](src/bank5/room-history-ow.asm#L146)
- Entering with count 7 while the room is among the six in the history: no monsters. [`RoomHistory_ModifyObjCountOW`](src/bank5/room-history-ow.asm#L93)
- Entering with count 7 after the room has dropped out of the history: the count is cleared and the full set returns (the only way an overworld room refills). [`RoomHistory_ModifyObjCountOW`](src/bank5/room-history-ow.asm#L93)
- Entering with count 1-6: the room makes its count minus the saved kills, in or out of the history; a partly cleared room stays reduced until it is cleared completely. [`RoomHistory_ModifyObjCountOW@CheckKillCount`](src/bank5/room-history-ow.asm#L117)

**Dungeon kill counts**
- Two stores: `LevelKillCounts[RoomId]` (RAM only, `$0F` = all dead) reduces the count on entry; room flags bits 6-7 (saved) hold min(kills, 2), or 3 for all dead. [`RoomHistory_SaveKillCountUW`](src/bank5/room-history-uw.asm#L110) [`LevelKillCounts`](src/Variables.inc#L480)
- A room is banked as all dead when it made no monsters, when the kills reach `RoomObjCount`, or when at least one kill was made and the first monster's type is non-recurring. [`RoomHistory_SaveKillCountUW`](src/bank5/room-history-uw.asm#L110)
- Non-recurring = list id or type `$32`-`$48` except `$3A`/`$3B` (Lanmola): Dodongo `$32`, Gohma, rupee stash, Grumble, Zelda, Digdogger, Manhandla, Aquamentus, Ganon, guard and standing fire, Moldorm, Gleeok, Patra. Dodongo type `$31` is outside the range. [`RoomHistory_ModifyObjCountUW`](src/bank5/room-history-uw.asm#L50)
- Entry, room in the history or flags not "all dead": count minus `LevelKillCounts[RoomId]` (none if negative). [`RoomHistory_ModifyObjCountUW@CalcObjCount`](src/bank5/room-history-uw.asm#L86)
- Entry, flags all dead and room out of the history: a non-recurring room makes nothing, permanently; any other room has its flags bits 6-7 and banked count cleared and refills. [`RoomHistory_ModifyObjCountUW`](src/bank5/room-history-uw.asm#L50)
- `LevelKillCounts` is zeroed only when a level is loaded (mode 2), so after leaving and re-entering a dungeon a partly cleared room is full again while a dead boss stays dead. [`Mode2Load_Init`](src/bank7/init-mode-dispatch.asm#L147)

# Combat: collision boxes, damage, invincibility, knockback, stun

Every collision is a test between two points: a hit needs the distance on each axis to be less than a threshold (exclusive), computed as the absolute value of an 8-bit difference. [`Collision_TestPointsWithThresholds`](src/bank1/object-overlap-test.asm#L98)

**Order and gates**
- Per monster and frame the weapon slots are tested in this order: `$0F` boomerang, `$0E` sword shot or magic shot, `$10` and `$11` bomb or fire, `$0D` sword, `$12` arrow or rod; then the monster's touch on Link. [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101)
- `ObjAttr` bit `$20` (Digdogger `$38`/`$39`, Trap, monster shots `$53`-`$5C`): no weapon is tested, only the touch on Link. [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101) [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70)
- While the monster's `ObjInvincibilityTimer` is nonzero nothing is tested: it cannot be hit and its touch does not hurt Link. [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101)
- The timer is tested once, before the first weapon: two weapons overlapping a monster in one frame both deal damage, and both can run the death code (kill counters + 1 twice). [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101)
- A monster in its spawn cloud or death sparkle (`ObjMetastate` not 0) gets no collision pass from the object loop. [`Mode5Play_UpdateWorld@LoopObject`](src/bank7/mode5-play-update.asm#L198)
- The monster's point is its middle (X + 8, Y + 8), or (X + 4, Y + 8) with `ObjAttr` bit `$40` (Gels, Patra children, Gohma, Lanmola, Moldorm, Gleeok, most shots). [`Obj_GetMiddle`](src/bank1/monster-collisions.asm#L173) [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70)

**Weapon boxes and damage** (damage in `ObjHP` units; `$10` = one wooden sword hit)
- Sword (slot `$0D`): only in sword state 2 (extended); damage type 1; `$10` wooden, `$20` white, `$40` magic. [`Sword_CheckMonsterCollision`](src/bank1/sword-bomb-arrow-collision.asm#L192) [`Sword_DamagePoints`](src/bank1/sword-bomb-arrow-collision.asm#L182)
- Sword and rod box: under `$0C` pixels across Link's facing axis and under `$10` along it, from the weapon point (X + 6, Y + 8) when Link faces up or down, (X + 8, Y + 6) when he faces left or right. [`Collision_CheckStabVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L215) [`Collision_CheckThinWeaponVsMonster_Thresholds`](src/bank1/object-overlap-test.asm#L47)
- Rod swing (slot `$12`, state >= `$30`): damage type 1, damage `$20`, tested in every state of the swing. [`Arrow_CheckArrowOrRodVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L249)
- Sword shot (slot `$0E`): type 1, damage of the sword held, box under `$0C`; a spreading shot (state bit 0) hits nothing. [`SwordShot_CheckMonsterCollision`](src/bank1/sword-bomb-arrow-collision.asm#L64)
- Magic shot (slot `$0E`, state bit 7): type `$10`, damage `$20`, box under `$0C`. [`SwordShot_CheckMonsterCollision`](src/bank1/sword-bomb-arrow-collision.asm#L64)
- Arrow (slot `$12`, state < `$20`): type 4, damage `$20` wooden, `$40` silver, box under `$0B`. [`Arrow_CheckArrowOrRodVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L249)
- The thin weapons' point is chosen by Link's current facing, not the weapon's: a flying arrow or shot shifts its point by 2 pixels per axis when Link turns to the other axis. [`Collision_CheckThinWeaponVsMonster_Thresholds`](src/bank1/object-overlap-test.asm#L47)
- Bomb (slots `$10`/`$11`): only in state `$13` (the blast); type 8, damage `$40`, box under `$18` from the bomb's middle (X + 8, Y + 8). [`Bomb_CheckBombOrFireVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L126)
- Fire (slots `$10`/`$11`, state >= `$20`): type `$20`, damage `$10`, box under `$0E` from its middle. [`Bomb_CheckBombOrFireVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L126)
- Boomerang (slot `$0F`): type 2, damage 0, point (X + 4, Y + 8), box under `$0A`; food in the same slot never collides. [`Boomerang_CheckMonsterCollision`](src/bank1/weapon-damage-death.asm#L48)
- A sword shot or magic shot that overlaps a monster ends there even when the monster is immune; an arrow turns into a spark (state `$20`), except on a Pols Voice. [`SwordShot_CheckMonsterCollision`](src/bank1/sword-bomb-arrow-collision.asm#L64) [`Collision_CheckShotVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L287)

**HP, immunity and death**
- `ObjHP` = the type's nibble of the HP table * `$10` (even type high nibble, odd type low nibble), loaded at init. [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105) [`Obj_Init@FetchAttrs`](src/bank7/object-init-table.asm#L141) [`ObjHP`](src/Variables.inc#L336)
- A hit subtracts the damage; the monster dies when HP < damage or the result is 0. Nibble 0 (Gels, Keese) dies to any hit, the boomerang's damage 0 included. [`Damage_DealToMonster`](src/bank1/weapon-damage-death.asm#L181)
- `ObjInvincibilityMask` bits name damage types that do nothing: 1 sword, rod swing and sword shot; 2 boomerang; 4 arrow; 8 bomb; `$10` magic shot; `$20` fire. It is zeroed with the rest of the room's RAM on entry. [`Damage_HandleWeaponHit`](src/bank1/weapon-damage-death.asm#L127) [`ObjInvincibilityMask`](src/Variables.inc#L343)
- Masks set by code: Darknut and Wizzrobes `$F6` (sword class and bomb hurt); Pols Voice, Patra and its children, Gleeok `$FE` (sword class only); Ganon `$FA` (sword class and arrow); Gohma `$FB` (arrow); Aquamentus and Manhandla `$E2` (no boomerang, no fire); Dodongo `$FF`, `$FE` while stunned. [`Darknut_Init`](src/bank4/darknut-gibdo.asm#L44) [`Wizzrobe_DrawAndCheckCollisions`](src/bank4/wizzrobe.asm#L683) [`PolsVoice_DrawAndCheckCollisions`](src/bank4/pols-voice.asm#L197) [`Patra_Init`](src/bank4/patra-ganon-inits.asm#L42) [`Gleeok_Init`](src/bank4/gleeok-manhandla-gohma-inits.asm#L59) [`Manhandla_Init`](src/bank4/gleeok-manhandla-gohma-inits.asm#L155) [`Ganon_Init`](src/bank4/patra-ganon-inits.asm#L96) [`Gohma_Init`](src/bank4/gleeok-manhandla-gohma-inits.asm#L221) [`Aquamentus_Init`](src/bank4/aquamentus-digdogger-dodongo-inits.asm#L40) [`Dodongo_CheckCollisionsStandardSize`](src/bank4/dodongo.asm#L347)
- An immune hit plays the parry tune, except for bomb and fire. [`Damage_PlayParryTuneUnlessBombOrFire`](src/bank1/weapon-damage-death.asm#L199)
- Special cases before damage: Gohma has its own handler; a Zol or Vire takes the weapon's direction (not from the boomerang); an arrow overlapping a Pols Voice sets its HP to 0 and kills it past the mask. [`Damage_HandleWeaponHit@CheckZolVire`](src/bank1/weapon-damage-death.asm#L143) [`Gohma_HandleWeaponCollision`](src/bank4/gohma.asm#L367) [`Collision_CheckShotVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L287)
- Death start: `ObjMetastate` = `$10`, stun timer, knockback and invincibility timer cleared; the sparkle runs 6 frames for each of three images, then the slot becomes a dropped item (type `$60`). [`Damage_KillMonster`](src/bank1/weapon-damage-death.asm#L216) [`Obj_AnimateMetastate`](src/bank7/object-animation.asm#L54) [`Obj_EndMetastate`](src/bank7/object-update-table.asm#L308)

**Parry (Darknut) and Link's shield**
- A Darknut (`$0B`, `$0C`) takes no damage and no knockback when the weapon's direction and its own are exact opposites (together `$0C` or `$03`); hits from the side or behind go through. [`Damage_HandleWeaponHit@CheckDarknut`](src/bank1/weapon-damage-death.asm#L157) [`Damage_ParryOrShove`](src/bank1/sword-bomb-arrow-collision.asm#L318)
- Link parries a shot only in state `$00`-`$0F` (not swinging or using an item) and facing against it on one axis. [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)
- Any shield parries types `$53`, `$54`, `$5B` (arrow), `$5C` (boomerang); the magic shield is needed for `$55` (fireball), `$57` (sword shot), `$58`, `$59` (magic shots); `$56` and `$5A` are never parried. [`Link_CheckMonsterTouch_NoReset@FacingOpposite`](src/bank1/link-collision-harm.asm#L138) [`InvMagicShield`](src/Variables.inc#L604)

**Touching Link**
- Touch box: Link's middle (X + 8, Y + 8) against the object's middle, both axis distances under 9 pixels. [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)
- No touch is tested while Link's invincibility timer, the clock, Link's stun timer or the monster's stun timer is nonzero, while Link's state is exactly `$40` (halted) or a Like-Like holds him. [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48) [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)
- Shots (types `$53` and up) touch Link only in states `$10`-`$1F`. [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)
- A Wallmaster or Like-Like whose touch harmed Link starts its capture in the same pass. [`Collision_CheckMonsterVsWeaponsAndLink@CheckCaptors`](src/bank1/monster-collisions.asm#L151)

**Invincibility timers**
- `ObjInvincibilityTimer` drops by 1 on even frames only, so a value N lasts about 2N frames. [`Obj_DecInvincibilityTimer`](src/bank7/object-stubs-destroy-monster.asm#L119)
- Link harmed: timer `$18` (about 48 frames). Monster hit by a weapon that knocks it back: `$10` (about 32 frames). [`Shove_Begin`](src/bank1/shove-knockback.asm#L65)
- The timer is set only by the knockback code: a boomerang hit and the Pols Voice arrow kill set none, and a monster immune to the damage type gets none. [`Shove_Begin`](src/bank1/shove-knockback.asm#L65) [`Collision_CheckWeaponPointVsMonster`](src/bank1/weapon-damage-death.asm#L82)
- With the clock, `$10` is added to Link's timer every frame, so he stays invincible. [`Mode5Play_UpdateWorld`](src/bank7/mode5-play-update.asm#L129)

**Knockback**
- Link: `$20` pixels; monster: `$40` pixels; both move up to 4 pixels per frame (8 and 16 frames), one pixel at a time. [`Shove_Begin`](src/bank1/shove-knockback.asm#L65) [`Shove_MovePixels`](src/bank7/shove.asm#L129)
- Nothing new starts while the object hit still has a nonzero invincibility timer. [`Shove_Begin@CheckInvincible`](src/bank1/shove-knockback.asm#L187)
- A monster is pushed in the direction the weapon faces, not away from Link. [`Shove_Begin@MonsterDefender`](src/bank1/shove-knockback.asm#L171)
- Link is pushed away from the attacker along one axis: between grid points the axis he faces; on a grid point sideways when the attacker's middle is within 3 pixels of his height, else up or down. [`Shove_Begin@UseDistance`](src/bank1/shove-knockback.asm#L100)
- After hitting Link a monster reverses its direction, unless its `ObjAttr` has bit `$80` or it is a Vire. [`Shove_Begin`](src/bank1/shove-knockback.asm#L65)
- The first frame of a knockback moves nothing; between grid points a knockback across a monster's facing axis is cancelled (its invincibility timer stays). [`Shove_Update@CheckPerpendicularShove`](src/bank7/shove.asm#L74)
- A knockback ends early at an unwalkable tile (tested on grid points only), at the room bounds, and for an upward push while Link's Y < `$8E` in a room whose slot `$01` holds a dungeon person or the Grumble Goriya. [`Shove_MovePixels@CheckBoundary`](src/bank7/shove.asm#L146) [`Shove_MovePixels@CheckBlocked`](src/bank7/shove.asm#L165)
- Sword, rod, sword shot, magic shot, arrow, bomb and fire knock back; the boomerang does not. Gohma is knocked back only when hit on its two middle parts with the eye fully open. [`Damage_ParryOrShove`](src/bank1/sword-bomb-arrow-collision.asm#L318) [`Bomb_CheckBombOrFireVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L126) [`Shove_Begin@Gohma`](src/bank1/shove-knockback.asm#L199)

**Stun**
- A boomerang that overlaps a monster always turns back (state `$50`); unless the monster's mask has bit 2 it sets `ObjStunTimer` = `$10` and deals damage 0. [`Collision_CheckWeaponPointVsMonster`](src/bank1/weapon-damage-death.asm#L82)
- Stun timers count once every 10 frames, so `$10` is 151-160 frames; a new hit reloads it. [`ObjStunTimer`](src/Variables.inc#L52) [`Nmi_Isr`](src/bank7/run-game-nmi.asm#L140)
- A stunned monster does not move and its touch does not hurt Link; it can still be hit. [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48) [`ObjStunTimer`](src/Variables.inc#L52)

**Kill counters**
- At the killing hit: `WorldKillCount` + 1 (no cap) and `HelpDropCount` + 1 up to `$0A`; `IsHelpDropBomb` is set only when the kill that brings the count to `$0A` is a bomb kill. Deaths started without `Damage_KillMonster` (ringleader deaths, Dodongo) add nothing here. [`Damage_KillMonster`](src/bank1/weapon-damage-death.asm#L216) [`DeadDummy_StartDeathSparkle`](src/bank7/object-stubs-destroy-monster.asm#L139)
- Any harm to Link zeroes those three counters, damage 0 included (Bubble touch, whirlwind pickup). [`Link_BeHarmed@ResetHelp`](src/bank1/link-collision-harm.asm#L210)
- When the death sparkle ends: `WorldKillCycle` steps 0..9 except for dead dummy `$5D`, child Gel `$14` and red Keese `$1C`; `RoomKillCount` + 1 except for those and Zora `$11`. [`Obj_EndMetastate`](src/bank7/object-update-table.asm#L308) [`WorldKillCycle`](src/Variables.inc#L468) [`RoomKillCount`](src/CommonVars.inc#L32)
- A Dodongo killed by the sword sets `HelpDropCount` and `IsHelpDropBomb` to `$0A` itself (the next drop is a bomb). [`Dodongo_CheckCollisions`](src/bank4/dodongo.asm#L296) [`HelpDropCount`](src/Variables.inc#L72)

# Hearts, rings and death

**Heart bytes**
- `HeartValues`: high nibble = heart containers - 1, low nibble = full hearts; `HeartPartial` fills the next heart (0 empty, `$01`-`$7F` drawn half, `$80`-`$FF` drawn full). Full health is low nibble = high nibble with `HeartPartial` = `$FF`. [`HeartValues`](src/Variables.inc#L593) [`HeartPartial`](src/Variables.inc#L595) [`Text_FormatHeartRows`](src/bank1/format-decimal-hearts.asm#L81)
- A new file and a file switched to Q2 start with `$22` and `$FF` (3 hearts). [`ModeERegister_Update`](src/bank2/register-eliminate-update.asm#L42) [`Profile_SwitchToSecondQuest`](src/bank2/second-quest-bank-stub.asm#L45)

**Damage to Link**
- Damage = the touching object's entry in the damage table: low nibble = whole hearts, high nibble = 1/256 hearts in place (`$80` = half a heart), read as the 16-bit amount [0D]:[0E]. [`Link_HarmByObjType`](src/bank1/link-collision-harm.asm#L164) [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)
- 4 hearts: Ganon `$3E`, shots `$59` and `$5A`. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)
- 2 hearts: blue Lynel `$01`, blue Darknut `$0C`, Pols Voice, small and large Digdogger, Blue Wizzrobe `$23`, Patra and its children, Gibdo, Gohma, Lanmola, monster sword shot `$57`, magic shot `$58`. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)
- 1 heart: red Lynel `$02`, blue Goriya `$05`, red Darknut `$0B`, Blue Leever `$0F`, Vire, Zol, Like-Like, Armos, Ghinis, Red Wizzrobe `$24`, Dodongo, Manhandla, Aquamentus, Gleeok and its head, Traps, fireball `$56`, monster boomerang `$5C`. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)
- Half a heart: Moblins, red Goriya `$06`, Octoroks, Tektites, Red Leever `$10`, Zora, Gels, Peahat, Keese, Boulders, Wallmaster, Rope, Stalfos, guard and standing fire, Moldorm, flying rock `$53`, shot `$54`, fireball `$55`, monster arrow `$5B`. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)
- 0: Bubbles, whirlwind, pond fairy, rupee stash, Grumble, Zelda, dungeon persons; the touch still knocks Link back and starts his invincibility timer. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200) [`Link_HarmByObjType`](src/bank1/link-collision-harm.asm#L164)
- Link's own fire costs half a heart (`$0080`) within `$0E` pixels when his invincibility timer is 0. [`Fire_Update`](src/bank7/fire.asm#L44)
- Rings: the 16-bit damage is shifted right once per ring grade: blue ring 1/2, red ring 1/4 (half a heart becomes `$0040` or `$0020`). [`Link_BeHarmed`](src/bank1/link-collision-harm.asm#L193) [`InvRing`](src/Variables.inc#L569)
- Subtraction: the fraction comes off `HeartPartial`; if it is larger, one full heart is turned into a partial of `$FF` and the remainder taken from that (a borrow costs 1/256 heart extra); then the whole hearts come off the low nibble. [`Link_BeHarmed@BorrowHeart`](src/bank1/link-collision-harm.asm#L238)
- Link dies when the whole-heart damage exceeds his full hearts, or a borrow is needed with no full heart left. [`Link_BeHarmed@HandleDied`](src/bank1/link-collision-harm.asm#L254)
- Every harm zeroes `WorldKillCount`, `HelpDropCount` and `IsHelpDropBomb`, and (except from the whirlwind) requests the hurt sample. [`Link_BeHarmed`](src/bank1/link-collision-harm.asm#L193)
- The sword shot needs low nibble = high nibble of `HeartValues` and `HeartPartial` >= `$80`: from full health up to `$7F`/256 of a heart may be missing. [`SwordShot_Make`](src/bank7/sword-and-rod.asm#L327)
- Low-hearts beep: requested on every frame the object loop runs while the low nibble of `HeartValues` is 0 (at most the partial heart left). [`Mode5Play_UpdateWorld`](src/bank7/mode5-play-update.asm#L129)

**Gaining hearts**
- A heart adds 1 full heart, a fairy 3; once full hearts = containers - 1 the partial is set to `$FF` and the rest discarded. [`Item_TakeHeartsNoSound`](src/bank1/take-item.asm#L475)
- A heart container adds `$11` (one container and one full heart) up to 16 containers; it does not refill the rest. [`Item_Take_CheckClass1`](src/bank1/take-item.asm#L337)
- The refill (potion, pond fairy, end of level) adds 6 to `HeartPartial` per frame, 43 frames per empty heart, and ends at full health; its end also clears `IsSwordBlocked`. [`World_FillHearts`](src/bank5/cave-cellar-enter.asm#L343)

**Death**
- Death sets `GameMode` = `$11`, full hearts and `HeartPartial` to 0 (containers kept), Link idle and facing down. [`Link_BeHarmed@HandleDied`](src/bank1/link-collision-harm.asm#L254)
- Init: Link flashes for 33 frames in the frozen room (monsters are not updated), then sound is silenced. [`Mode11Death_Init`](src/bank5/mode11-death-init.asm#L41) [`Mode11Death_Init_Sub1`](src/bank5/mode11-death-init.asm#L73)
- Update, 13 steps, about 320 frames with no controller input read: redraw (28 frames), 4 spins (a quarter turn every 5 frames), palette fade (4 steps of 10 frames), grey Link `$18` frames, spark `$0F` frames, wait `$2E`, GAME OVER `$60` frames. [`Mode11Death_Update_Full_JumpTable`](src/bank5/mode11-death.asm#L66) [`Mode11Death_Update_Sub7Spin`](src/bank5/mode11-death.asm#L176) [`Mode11Death_Update_SubASpark`](src/bank5/mode11-death.asm#L236) [`Mode11Death_Update_SubBGameOver`](src/bank5/mode11-death.asm#L280)
- The last step adds 1 to the save slot's `DeathCounts` (stops at `$FF`) and enters mode 8, the CONTINUE / SAVE / RETRY menu. [`Mode11Death_Update_SubCEnd`](src/bank5/mode11-death.asm#L293) [`DeathCounts`](src/Variables.inc#L543)
- Mode 8 is also reached without dying: Up + A held on controller 2 in the item submenu. [`Submenu_UpdateActive`](src/bank5/submenu-states.asm#L301)
- In mode 8 Select moves the cursor, Start confirms; the line flashes `$40` frames, then every choice sets 3 hearts (containers kept), Link's state 0 and the clock off. [`Mode8Continue_Update_Full@HandleActivated`](src/bank5/mode8-continue-mode10-stairs.asm#L143)
- CONTINUE goes to mode 3 without reloading the level: inventory, world flags and `LevelKillCounts` in RAM are kept, the room history is cleared, and Link restarts in the level's start room (in the overworld in `LevelEntranceRoomId` when it is set). [`Mode8Continue_ChoiceModes`](src/bank5/mode8-continue-mode10-stairs.asm#L49) [`Mode8Continue_Init`](src/bank5/mode-inits-obj-lists.asm#L49) [`Mode3Unfurl_Init_Sub1`](src/bank7/init-mode-dispatch.asm#L294)
- SAVE goes to mode `$D`: the profile is written to the save file (see the next section), then the title's file check runs. [`ModeDSave_Update_Sub0BuildFileB`](src/bank2/moded-save.asm#L80) [`ModeDSave_Update_Sub2Leave`](src/bank2/moded-save.asm#L287)
- RETRY goes to mode 0 submode 1 without saving: the slot info, death count included, is read from the save file again and the file menu reloads the profile, so everything since the last save is lost. [`Mode8Continue_Update_Full@HandleActivated`](src/bank5/mode8-continue-mode10-stairs.asm#L143) [`Mode0Demo_Update_Sub2`](src/bank2/mode0-demo.asm#L127) [`Mode1Menu_Update_Sub1Load`](src/bank2/mode1-menu-update.asm#L104)

# Room flags and saved state

**The flag byte**
- One flag byte per room in `WorldFlags`: `$067F` overworld, `$06FF` levels 1-6, `$077F` levels 7-9 (`$80` bytes each); the current level's block address is in the level info, the same in both quests. [`WorldFlags`](src/Variables.inc#L614) [`RoomFlags_Get`](src/bank7/next-room-room-flags.asm#L134) [`LevelInfo_WorldFlagsAddr`](src/Variables.inc#L724)
- Overworld bits 0-2: saved kill count, 7 = all dead. [`RoomHistory_SaveKillCountOW`](src/bank5/room-history-ow.asm#L146)
- Overworld bit 4 (`$10`): the cave's gift, offer or ware was taken, or the door repair was charged (per room, not per cave type). [`RoomFlags_SetItemTaken`](src/bank1/take-item.asm#L140) [`Cave_Init`](src/bank1/cave-init-person-text.asm#L101)
- Overworld bit 5 (`$20`): put the stairs back when the room is laid out; set when a pushed rock or gravestone finishes its `$10`-pixel move and for the room shortcut stairs lead to. [`RoomFlags_MarkVisited`](src/bank7/next-room-room-flags.asm#L120) [`Secret_RestoreStairsOW`](src/bank5/ow-shortcut-map-square.asm#L54)
- Overworld bit 7 (`$80`): secret found; set when a bombed rock wall opens, a tree burns or the flute pond's stairs appear. [`Secret_RevealAndFlagTileObj`](src/bank4/rock-wall-tree-secrets.asm#L287) [`Secret_RevealAndFlagStairsObj`](src/bank4/rock-wall-tree-secrets.asm#L283) [`RockWall_Update`](src/bank4/rock-wall-tree-secrets.asm#L214)
- Overworld bits 3 and 6: never read and never set. The flag byte is reached only through `RoomFlags_Get` and `LevelInfo_WorldFlagsAddr` (plus whole-block copies between the profile and the save file); in the overworld that code tests and writes bits 0-2 (the kill count, capped at 7 before it is stored), 4, 5 and 7 only; the door, visited and roar code that uses the other bits runs in dungeons. [`RoomFlags_Get`](src/bank7/next-room-room-flags.asm#L134) [`RoomHistory_SaveKillCountOW`](src/bank5/room-history-ow.asm#L146) [`RoomFlags_SetItemTaken`](src/bank1/take-item.asm#L140) [`Secret_RestoreStairsOW`](src/bank5/ow-shortcut-map-square.asm#L54) [`WorldFlags`](src/Variables.inc#L614)
- Dungeon bits 0-3: door opened, unlocked or bombed on the side with that direction index (bit 0 right/E, 1 left/W, 2 down/S, 3 up/N); set for this room and the room behind the door, cleared when a shutter closes. [`RoomFlags_SetDoorOpened`](src/bank5/shutters-door-flags.asm#L41) [`RoomFlags_ClearDoorOpened`](src/bank5/shutters-door-flags.asm#L52) [`Door_Update`](src/bank5/update-doors.asm#L60)
- Dungeon bit 4 (`$10`): room item taken. [`RoomFlags_SetItemTaken`](src/bank1/take-item.asm#L140) [`RoomFlags_GetItemTaken`](src/bank1/take-item.asm#L151)
- Dungeon bit 5 (`$20`): room visited (drawn on the map); set on entering a room, not on leaving. [`RoomFlags_MarkVisited`](src/bank7/next-room-room-flags.asm#L120) [`Room_RunEntryTasks_MarkVisited`](src/bank7/init-mode5-play.asm#L156)
- Dungeon bits 6-7: kills banked, min(kills, 2), or 3 = all dead; the boss room's "all dead" also stops the boss roar in the level. [`RoomHistory_SaveKillCountUW`](src/bank5/room-history-uw.asm#L110) [`Sample_SetBossRoarUW`](src/bank5/boss-sound-door-face.asm#L42)

**What a save holds**
- A save file is: name (8 bytes), the items block `InvItems` (`$28` bytes: every inventory byte, rupees, keys, `HeartValues`, `HeartPartial`, Triforce pieces, bomb capacity), the world flags (`$180` bytes), an active flag, one unused byte, the death count and the quest number. [`ModeDSave_Update_Sub0BuildFileB`](src/bank2/moded-save.asm#L80) [`Save_CalcFileAChecksum`](src/bank2/mode1-init-save-files.asm#L153)
- Saving is two-phase: mode `$D` builds file B with a 16-bit checksum, then copies it over file A; a pending valid file B is committed at the next file check, a file A with bad markers or checksum is erased. [`ModeDSave_Update_Sub1Commit`](src/bank2/moded-save.asm#L155) [`Save_CopyFileBToFileA`](src/bank2/moded-save.asm#L190) [`Save_CommitOrRepairFiles`](src/bank2/mode1-init-save-files.asm#L82)
- The items block is copied before mode `$D` refills the hearts in RAM, so the file holds the hearts as they were; coming from the continue menu that is 3 hearts. [`ModeDSave_Update_Sub0BuildFileB`](src/bank2/moded-save.asm#L80)
- The file menu always draws full hearts for a slot, whatever the file holds. [`Mode0Demo_Update_Sub2`](src/bank2/mode0-demo.asm#L127)
- Loading a slot copies the items block and world flags into RAM, zeroes `IsSwordBlocked`, Link's state and the clock, and sets `CurLevel` = 0; the menu init sets `LevelEntranceRoomId` = `$FF`, so play always resumes in the overworld's start room. [`Mode1Menu_Update_Sub1Load`](src/bank2/mode1-menu-update.asm#L104) [`Mode3Unfurl_Init_Sub1`](src/bank7/init-mode-dispatch.asm#L294)
- The clock byte is inside the saved items block but is zeroed on load and on every room entry. [`Link_ResetStateAndClock`](src/bank7/init-mode-dispatch.asm#L354) [`InvClock`](src/Variables.inc#L580)

**What is not saved, and when it resets**
- Every room entry zeroes `$0300`-`$051F`: all object slots, the room's kill count, door and secret state, candle use, the Zora and Boulder counters. [`Room_EnterCommon`](src/bank5/enter-room.asm#L143)
- `LevelKillCounts` (128 bytes) and the room history are zeroed at each level load; the history also at mode 3 and in the continue menu. [`Mode2Load_Init`](src/bank7/init-mode-dispatch.asm#L147) [`RoomHistory_Clear`](src/bank7/init-mode-dispatch.asm#L271)
- `SpawnCycle`, `CurEdgeSpawnCell`, `WorldKillCycle`, `WorldKillCount` and `HelpDropCount` persist across rooms and levels and are cleared only by reset (and the counters by harm to Link). [`Reset_ClearRam`](src/bank5/save-ram-clear-ram.asm#L81) [`Link_BeHarmed@ResetHelp`](src/bank1/link-collision-harm.asm#L210)
- Save RAM is valid when its two markers are `$5A` and `$A5`; otherwise the working part is zeroed at reset or before a save. [`Save_InitSaveRam`](src/bank5/save-ram-clear-ram.asm#L35)
- Eliminating a slot formats its file A at once. [`ModeFEliminate_DeleteSlot`](src/bank2/register-eliminate-update.asm#L205) [`Save_FormatFileA`](src/bank2/mode1-init-save-files.asm#L212)

# Level data formats

A level is two blocks copied from ROM to RAM in mode 2; all code reads the RAM copy. The formats, field by field with the routine that reads each field, are in `data/README.md`; the decoded listings are beside it.

- Level block: 768 bytes = six arrays A-F of 128 bytes, one byte per room (`RoomId` `$00`-`$7F`, row in the high nibble). One block for the overworld, one for levels 1-6, one for levels 7-9; the dungeon blocks exist in a Q1 and a Q2 version. [`Mode2Load_Init_Sub0`](src/bank6/level-load-mode2.asm#L122) [`Level_BlockAddrsQ1`](src/bank6/level-load-mode2.asm#L57) [`LevelBlockAttrsA`](src/Variables.inc#L695)
- Overworld rooms decoded (cave exits, cave index, monsters, layout, secret quest, edge spawns): `data/levelblock-ow.md`, format in `data/README.md` "Overworld rooms". [`Level_BlockOW`](src/bank6/level-blocks-common-data.asm#L66)
- Dungeon rooms decoded (door types, monsters, push block, layout, dark flag, boss roar, room item, secret trigger): `data/levelblock-uw-1to6-q1.md`, `-7to9-q1.md`, `-1to6-q2.md`, `-7to9-q2.md`; format in "Underworld rooms" and "Cellar rooms". [`Level_BlockUW1to6Q1`](src/bank6/level-blocks-common-data.asm#L70) [`Level_BlockUW7to9Q1`](src/bank6/level-blocks-common-data.asm#L74)
- Level info: 252 bytes per level (palettes, four monster counts, start Y, item or stairs positions, start, Triforce and boss rooms, room flags address, level number, cellar rooms, maps, fade series): `data/levelinfo.md`, format in "Level info". [`Mode2Load_Init_Sub1`](src/bank6/level-load-mode2.asm#L156) [`Level_InfoAddrs`](src/bank6/level-load-mode2.asm#L71) [`LevelInfo_FoeCounts`](src/Variables.inc#L716)
- Monster lists (ids `$62`-`$7F`) and object type names: `data/objlists.md`. [`Spawn_ObjLists`](src/bank5/mode-inits-obj-lists.asm#L185)
- Room layouts, column tables and squares (overworld, dungeon, caves, cellars): `data/room-layouts.md`. [`RoomLayout_LayOutRoom`](src/bank5/mode12-lay-out-room.asm#L178)
- Person text strings: `data/person-text.md`. [`Person_TextAddrs`](src/bank1/cave-init-person-text.asm#L47)
- The Q2 patches applied in RAM after the copy are listed in `data/README.md` "Second-quest patches". [`Mode2Load_Update_Full`](src/bank6/level-load-mode2.asm#L297)

# Second quest differences

The quest is `QuestNumbers[CurSaveSlot]`: 0 = Q1, nonzero = Q2. The code paths below are all that read it.

**Entering Q2**
- The quest number is part of the save file and is copied to `QuestNumbers` with the other slot info before the file menu. [`Mode0Demo_Update_Sub2`](src/bank2/mode0-demo.asm#L127) [`QuestNumbers`](src/Variables.inc#L542)
- A new file whose name starts with the five characters ZELDA is registered in Q2 (characters 6-8 are not compared). [`ModeERegister_Update@CompareToZelda`](src/bank2/register-eliminate-update.asm#L112) [`ModeERegister_ZeldaNameChars`](src/bank2/register-eliminate-update.asm#L33)
- Start on the last screen of the ending resets the profile for Q2: all `$180` world flags and the `$28`-byte items block zeroed, 3 hearts, bomb capacity 8, quest = 1; name and death count stay. [`Profile_SwitchToSecondQuest`](src/bank2/second-quest-bank-stub.asm#L45) [`Mode13WinGame_Update_Sub4WaitStart`](src/bank2/mode13-win-game.asm#L308)
- The quest is set to 1, not incremented: finishing Q2 starts Q2 again. [`Profile_SwitchToSecondQuest`](src/bank2/second-quest-bank-stub.asm#L45)
- The file menu draws a sword beside the Link of a slot that is in Q2. [`Mode1Menu_WriteLinkSprites`](src/bank2/mode1-menu-update.asm#L162)

**Level data**
- Dungeons load separate Q2 level blocks (one for levels 1-6, one for 7-9); the overworld block and all level info blocks are shared with Q1. [`Mode2Load_Init_Sub0`](src/bank6/level-load-mode2.asm#L122) [`Level_BlockAddrsQ2`](src/bank6/level-load-mode2.asm#L89)
- Dungeon level info is patched in RAM from offset `$29` on: item positions, map rotation and offset, start room, Triforce room, level number, cellar rooms, boss room, map mask, status bar map. Palettes, monster counts and start Y stay. [`Mode2Load_Update_Full@ReplaceInfoBytes`](src/bank6/level-load-mode2.asm#L318) [`Quest_Q2LevelInfoPatchAddrs`](src/bank6/quest2-replacements.asm#L197)
- Q2 start / Triforce / boss rooms by internal level (`CurLevel`): 1 `$77`/`$08`/`$07`; 2 `$75`/`$20`/`$30`; 3 `$79`/`$1B`/`$2B`; 4 `$72`/`$00`/`$10`; 5 `$7D`/`$4F`/`$5F`; 6 `$74`/`$16`/`$26`; 7 `$7F`/`$2D`/`$3D`; 8 `$79`/`$1B`/`$1C`; 9 `$74`/`$07`/`$17`. [`Quest_Q2LevelInfoPatch1`](src/bank6/quest2-replacements.asm#L84) [`Quest_Q2LevelInfoPatch9`](src/bank6/quest2-replacements.asm#L183)
- Level numbering: the digit drawn in "LEVEL-n" comes from the patched level info, not from `CurLevel`; the Q2 tables for `CurLevel` 1-9 hold 1, 3, 2, 5, 4, 6, 8, 7, 9. [`Quest_Q2LevelInfoPatch2`](src/bank6/quest2-replacements.asm#L96) [`LevelInfo_LevelNumber`](src/Variables.inc#L725)
- The room flags addresses in the Q2 tables equal the Q1 values (`$06FF`, `$077F`). [`Quest_Q2LevelInfoPatch1`](src/bank6/quest2-replacements.asm#L84) [`Quest_Q2LevelInfoPatch7`](src/bank6/quest2-replacements.asm#L158)
- Overworld patch: 15 bytes of the level block are written, 8 from a table and 7 by fixed stores, in rooms `$0B`, `$0E`, `$0F`, `$22`, `$34`, `$3C`, `$45`, `$74`. [`Mode2Load_Update_Full@PatchQ2Rooms`](src/bank6/level-load-mode2.asm#L329) [`Quest_Q2OverworldPatchOffsets`](src/bank6/quest2-replacements.asm#L49) [`Quest_Q2OverworldPatchValues`](src/bank6/quest2-replacements.asm#L64)
- Patched entrances: room `$22` leads to cave `$21` instead of level 6; room `$34` becomes the level 3 entrance (cave `$20` in Q1); room `$74` leads to cave `$1E` instead of level 3; rooms `$0E` and `$0F` lead to caves `$1E` and `$20`. [`Quest_Q2OverworldPatchValues`](src/bank6/quest2-replacements.asm#L64)
- Room `$0B` gets monster list id `$2F` (pond fairy) instead of Red Leevers; rooms `$0B` and `$3C` get layout `$7B`, room `$74` layout `$5A`, and rooms `$3C` and `$74` new cave exit positions. [`Mode2Load_Update_Full@PatchQ2Rooms`](src/bank6/level-load-mode2.asm#L329) [`RoomLayout_RoomOW7B`](src/bank5/room-layouts-ow.asm#L77)
- Pattern tables are chosen by `CurLevel` alone; no quest test. [`Chr_TransferLevelPatterns`](src/bank3/level-pattern-blocks.asm#L132)

**Secrets and the recorder**
- Overworld tile-object secrets (pushed rock or gravestone, bombed rock wall, burned tree) carry a quest in attribute byte F bits 7-6: 0 both, 1 Q1 only, 2 Q2 only; in the other quest the object ignores Link. [`Secret_IsForOtherQuest`](src/bank4/rock-wall-tree-secrets.asm#L331) [`Secret_QuestNumbers`](src/bank4/rock-wall-tree-secrets.asm#L322)
- Flute secrets: Q1 only room `$42`; Q2 rooms `$06`, `$29`, `$2B`, `$30`, `$3A`, `$3C`, `$58`, `$60`, `$6E`, `$72` and not `$42`. In a listed room of the other quest the flute summons the whirlwind. [`Flute_Wield@Found`](src/bank7/flute.asm#L82) [`Flute_SecretRoomIdsOW`](src/bank7/flute.asm#L36)
- The whirlwind's destination rooms are one table for both quests (it scrolls in beside rooms `$37`, `$3C`, `$74`, `$45`, `$0B`, `$22`, `$42`, `$6D` for levels 1-8). [`Whirlwind_PrevRoomIds`](src/bank1/whirlwind.asm#L81)

**Monsters**
- Rope: HP `$10` in Q1, `$40` in Q2; in Q2 its palette cycles every frame. [`Rope_Init`](src/bank4/rope.asm#L32) [`Rope_Update@Animate`](src/bank4/rope.asm#L125)
- Stalfos: in Q2 only it stops and fires a sword shot (type `$57`, 2 hearts): a sequence starts on 8 of 256 random values when it wants to shoot, stands `$30` frames and fires `$20` frames in; a hit cancels it. [`Stalfos_Update`](src/bank4/stalfos.asm#L43) [`Stalfos_Update@PrepareToShoot`](src/bank4/stalfos.asm#L65)
- Bubble: no code tests the quest. The sword-disabling Bubble `$2D` and the curing Bubble `$2C` come only from monster lists `$7D`-`$7F`, which only rooms of the Q2 dungeon blocks use (17 rooms); the flashing Bubble `$2B` is in both quests. [`Bubble_Update@BlockOrUnblock`](src/bank4/bubble-keese.asm#L137) [`Spawn_ObjLists`](src/bank5/mode-inits-obj-lists.asm#L185)

**Ending**
- The credits scroll ends at screen 2, scroll `$78` in Q1 and at screen 3, scroll 0 in Q2, and the lines differ: Q1 lines `$00`-`$0F`; Q2 lines `$00`-`$0B` and `$10`-`$16`, with the slot's name and death count on line `$11`. [`Credits_DrawRow`](src/bank2/credits.asm#L164) [`Credits_LastScreens`](src/bank2/credits.asm#L44) [`Credits_LastVScrolls`](src/bank2/credits.asm#L48)
