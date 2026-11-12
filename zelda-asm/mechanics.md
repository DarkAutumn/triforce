# Game mechanics

How the game works underneath: modes and the frame loop, object slots, random numbers, timers, Link's
movement, tile collision, screen edges and the overworld screen wrap, doors and secrets, stairs and caves, the
mazes, rooms the code treats specially, spawning and room history, combat, hearts and death, saved state, level
data and the second quest. Every bullet ends with the label that proves it. `(trace)` marks a result observed
in the running game; `(unverified)` a statement not confirmed in the code. Monsters are in
[enemies.md](enemies.md), items in [items.md](items.md), decoded level data in [data/](data/README.md).

Contents (generated; `L` = line in this file):

<!-- toc -->
- L35 Game modes and the frame loop
- L104 Object slots and update order
- L154 Random numbers
- L183 Timers
- L224 Link's movement: grid, speed and sub-pixels
- L288 Tile collision and walking half into blocks
- L345 Screen edges and room transitions
- L398 Screen wrap (overworld)
- L476 Doorways, doors, keys, shutters and bomb walls
- L533 Push blocks and room secrets
- L568 Stairs, cellars and caves
- L611 Lost Woods and Lost Hills (mazes)
- L626 Special-cased rooms
- L666 Ladder and raft
- L689 Monster spawning, edge spawns and respawn (room history)
- L751 Combat: collision boxes, damage, invincibility, knockback, stun
- L823 Hearts, rings and death
- L860 Room flags and saved state
- L889 Level data formats
- L902 Second quest differences
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
| `$04` | enter room | [`Mode4Enter_Init`](src/bank5/enter-room.asm#L59) | [`Mode4Enter_UpdateEnterOrLeave`](src/bank7/update-mode-dispatch.asm#L197) |
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
| `$13` | win game | [`Mode13WinGame_Init`](src/bank7/init-mode-dispatch.asm#L222) | [`Mode13WinGame_Update`](src/bank2/mode13-win-game.asm#L49) |

**Init, update and submodes**
- The main thread is an empty loop; every frame is one run of the NMI handler, which keeps the NMI disabled until it returns. [`Reset_LoopForever`](src/bank7/run-game-nmi.asm#L84) [`Nmi_Isr`](src/bank7/run-game-nmi.asm#L140)
- Each frame runs exactly one of the two: the mode's init routine while `IsUpdatingMode` = 0, its update routine otherwise. [`Nmi_Isr@Update`](src/bank7/run-game-nmi.asm#L311) [`IsUpdatingMode`](src/Variables.inc#L13)
- Init dispatch: bank 5 is switched in, then a table jump on `GameMode`; modes `$0E` and `$0F` share one init. [`Mode_Init`](src/bank7/init-mode-dispatch.asm#L80) [`Mode_Init_JumpTable`](src/bank7/init-mode-dispatch.asm#L87)
- Update dispatch: bank 2 is switched in, then a table jump on `GameMode`. [`Mode_Update`](src/bank7/update-mode-dispatch.asm#L62) [`Mode_Update_JumpTable`](src/bank7/update-mode-dispatch.asm#L69)
- An init may last many frames, stepping `GameSubmode`; it ends by calling the routine that sets `GameSubmode` = 0 and `IsUpdatingMode` = 1, so the first update runs on the next frame. [`Mode_BeginUpdate`](src/bank1/status-bar-rupees.asm#L53)
- Ending a mode clears `IsUpdatingMode` and `GameSubmode`; the next frame runs the init of whatever `GameMode` then holds. [`Mode_End`](src/bank7/update-mode-dispatch.asm#L164)
- The "next mode" routine adds 1 to `GameMode` first: 2 -> 3, 3 -> 4, 4 -> 5, 6 -> 7. [`Mode_GoToNext`](src/bank7/update-mode-dispatch.asm#L155)
- Inits with submodes: mode 1; mode 2; mode 3 (0-8); mode 4 (0-3, 0 = the room entry itself); mode 7 (0-6); mode 9 (0-9); mode `$A` (0-`$0A`); modes `$B` and `$C` (0-8); modes `$E`/`$F` (0-4); mode `$13` (0-4). [`Mode3Unfurl_Init_JumpTable`](src/bank7/init-mode-dispatch.asm#L242) [`Mode4Enter_Init`](src/bank5/enter-room.asm#L59) [`Mode7Scroll_InitSubmodes_JumpTable`](src/bank5/mode7-scroll-init.asm#L60) [`Mode9Cellar_Init_JumpTable`](src/bank5/room-history-ow.asm#L178) [`ModeALeaveCellar_Init_JumpTable`](src/bank5/cave-cellar-enter.asm#L79) [`ModeBCave_Init_JumpTable`](src/bank5/room-history-ow.asm#L48) [`Mode13WinGame_Init_JumpTable`](src/bank2/mode13-ending-init.asm#L54)
- Updates with submodes: mode 0; mode 1 (0 choose, 1 act); mode 7; mode `$D` (0-2); mode `$11` (`$00`-`$0C`); mode `$12`; mode `$13` (0-4). [`Mode1Menu_Update_JumpTable`](src/bank2/mode1-menu-update.asm#L48) [`Mode7Scroll_UpdateSubmode_JumpTable`](src/bank5/mode7-scroll-update.asm#L70) [`ModeDSave_Update_JumpTable`](src/bank2/moded-save.asm#L71) [`Mode11Death_Update_Full_JumpTable`](src/bank5/mode11-death.asm#L66) [`Mode13WinGame_Update_JumpTable`](src/bank2/mode13-win-game.asm#L54)
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
- `$01`: the first monster the room makes; its type is kept as `RoomObjTemplateType`; cave and dungeon persons are in slot 1. [`Room_EnterCommon@StoreObjTemplate`](src/bank5/enter-room.asm#L372) [`Spawn_AssignPositions@InCave`](src/bank5/spawn-positions.asm#L155)
- `$0B`: also the room's tile object (push block, rock, gravestone, armos statue, rock wall, tree, dock). [`Room_SetUpTileObjOW`](src/bank5/enter-room.asm#L423) [`Room_SetUpItemAndPushBlock`](src/bank5/create-room-objects.asm#L51)
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
- Per slot: the invincibility timer is counted (also for an empty slot), then the object's update, then, for objects with attribute bit `$01` clear and metastate 0, the generic walking draw (unless bit `$04`) and the collision test against Link and his weapons. [`Mode5Play_UpdateWorld@LoopObject`](src/bank7/mode5-play-update.asm#L198) [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L97)
- So a weapon moved this frame is tested against each monster after that monster has moved, higher slots first. [`Mode5Play_UpdateWorld`](src/bank7/mode5-play-update.asm#L129)
- After the loop: dungeon checks (statues, secrets, shutters, doors) or the overworld Zora spawn, then the item lift, the room item and its pickup, then hearts and rupees. [`Mode5Play_UpdateWorld@CheckUW`](src/bank7/mode5-play-update.asm#L244) [`Mode5Play_UpdateWorld@FinishUpdatePlay`](src/bank7/mode5-play-update.asm#L296)

**Uninitialized slots and the spawn cloud**
- A slot with `ObjUninitialized` nonzero only runs the init on its first update; its own update starts on a later frame. [`Obj_Update@Init`](src/bank7/object-update-table.asm#L137) [`Obj_Init`](src/bank7/object-init-table.asm#L45)
- The init loads `ObjAttr` from the type's attribute byte and `ObjHP` = HP nibble x `$10`, then runs the type's init routine. [`Obj_Init@FetchAttrs`](src/bank7/object-init-table.asm#L141) [`Obj_Init_JumpTable`](src/bank7/object-init-table.asm#L177)
- While `ObjMetastate` is nonzero (1-3 spawn cloud, `$10`-`$13` death sparkle) the type's update does not run: the object neither moves nor is tested for collisions. [`Obj_Update@Update`](src/bank7/object-update-table.asm#L143) [`ObjMetastate`](src/Variables.inc#L327)
- First cloud step lasts as many frames as the slot number (`ObjTimer` = slot), the next two 6 frames each, so higher slots start acting later. [`Obj_Init@NormalSpawn`](src/bank7/object-init-table.asm#L122) [`Obj_AnimateMetastate@IncMetastate`](src/bank7/object-animation.asm#L71)
- Armos (`$1E`), flying Ghini (`$22`) and types `$53` and up get no slot-number timer. [`Obj_Init@NormalSpawn`](src/bank7/object-init-table.asm#L122)
- An overworld monster that walks in from the screen edge has no cloud (metastate 0) and stays uninitialized until `MonstersFromEdgesLongTimer` is 0 and its edge cell is far enough from Link. [`Obj_Init@InitMonsterFromEdge`](src/bank7/object-init-table.asm#L79) [`Obj_Init@UninitMonsterFromEdge`](src/bank7/object-init-table.asm#L70)
- A death sets metastate `$10`; when the sparkle ends the slot becomes type `$60` (dropped item), uninitialized, attribute `$81`. [`DeadDummy_StartDeathSparkle`](src/bank7/object-stubs-destroy-monster.asm#L139) [`Obj_EndMetastate@DropItem`](src/bank7/object-update-table.asm#L338)

**Finding and freeing slots**
- A new object (monster shot, split monster, ladder, fairy) takes the highest free slot from `$0B` down to `$01`; slot 0 and slots above `$0B` are never returned. [`Obj_FindEmptyMonsterSlot`](src/bank7/object-stubs-destroy-monster.asm#L169)
- An object made in a slot higher than the maker's is not updated until the next frame; one made in a lower slot gets its init later the same frame (the loop counts down). [`Mode5Play_UpdateWorld@NextObject`](src/bank7/mode5-play-update.asm#L227)
- Freeing a slot sets type 0, clears its shove, timer, state and invincibility timer, and leaves it uninitialized with metastate 1 for the next user. [`Obj_DestroyMonster`](src/bank7/object-stubs-destroy-monster.asm#L153) [`Obj_ResetSlotFields`](src/bank1/bomb-candle-wield.asm#L308)

**Room-entry clearing**
- Every room entry (modes 4, 9, `$B`, `$C`) zeroes RAM `$0300`-`$051F`: object types, grid offsets, position fractions, speeds, metastates, HP, the room kill count and the other per-room variables. [`Room_EnterCommon`](src/bank5/enter-room.asm#L137) [`Util_ClearRam0300UpTo`](src/bank7/ppu-tablejump-input.asm#L162)
- Slots `$0B`-`$01` then get: uninitialized `$FF`, state 0, direction 0, stun 0, no shove, animation counter 1, metastate 1, q-speed `$20`. [`Room_EnterCommon@ResetObjLoop`](src/bank5/enter-room.asm#L259)
- Leaving and entering a room both clear `LadderSlot` and the state of weapon slots `$0D`-`$12`: nothing Link has out survives a room change. [`Obj_ResetWeaponSlotsAndLadder`](src/bank5/mode3-mode6-init.asm#L263)
- The same moments set Link's state to 0 and turn the clock off. [`Link_ResetStateAndClock`](src/bank7/init-mode-dispatch.asm#L354)
- Object types go to slots 1..count from the room's monster list; slots `$01`-`$09` all get a spawn position whatever the count. [`Room_EnterCommon@StoreObjTemplate`](src/bank5/enter-room.asm#L372) [`Spawn_AssignPositions`](src/bank5/spawn-positions.asm#L64)

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
- Monsters read `Random,X` or `Random+1,X` with X = their object slot (`$01`-`$0B`), so each slot sees a different byte of the register. [`Random`](src/Variables.inc#L29) [`Wanderer_MoveAndTurn`](src/bank4/walker-wanderer-common.asm#L223)
- Walking monsters: `Random+1,X` against the turn rate at each 16-pixel grid point (turn toward the chase target), `Random,X` as the new turn timer. [`Wanderer_MoveAndTurn`](src/bank4/walker-wanderer-common.asm#L223)
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
- Can: which slot a monster is in (kill order, room history lowering the count), since the slot is the byte index. [`Room_EnterCommon@StoreObjCount`](src/bank5/enter-room.asm#L325) [`Obj_FindEmptyMonsterSlot`](src/bank7/object-stubs-destroy-monster.asm#L169)
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
- Stun timers are zeroed on room entry and when the monster dies. [`Room_EnterCommon@ResetObjLoop`](src/bank5/enter-room.asm#L259) [`Damage_KillMonster`](src/bank1/weapon-damage-death.asm#L216)

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
- Walking monsters reset their offset to 0 when it is a multiple of `$10` and decide their direction there. [`Wanderer_MoveAndTurn`](src/bank4/walker-wanderer-common.asm#L223)
- Tiles ahead and the screen edge are tested only on a frame that starts with offset 0. [`Walker_CheckTileCollision@CheckGridOffset`](src/bank7/walker-collision-screen-edge.asm#L101)

**Speed and sub-pixels**
- The q-speed (`ObjQSpeedFrac`, `$3BC,X`) is added to (right, down) or subtracted from (left, up) the 8-bit `ObjPosFrac` (`$3A8,X`) 4 times a frame; each carry or borrow is 1 pixel, so pixels per frame = q-speed / 64, at most 4. [`Obj_MoveByQSpeed@ApplyQSpeedToPosition`](src/bank7/walker-move.asm#L262) [`ObjQSpeedFrac`](src/Variables.inc#L317)
- Link's q-speed is `$60` = 1.5 pixels per frame: `ObjPosFrac` alternates `$00`/`$80` and he moves 1, 2, 1, 2 ... pixels; an 8-pixel step takes 5 or 6 frames. [`Link_InitSpeed`](src/bank5/link-handle-input.asm#L272) [`Obj_MoveByQSpeed`](src/bank7/walker-move.asm#L244)
- When the limit withholds the second pixel of a 2-pixel frame the fraction has still advanced: that pixel is lost, not postponed. [`Obj_AddQSpeedToPosFrac@ClearCarry`](src/bank1/qspeed-direction-math.asm#L66)
- Slow tiles: in the overworld, while Link's `ObjCollidedTile` is `$74` or `$75` (the mountain stairs; the tile identity is unverified) his q-speed is `$30` = 0.75 pixels per frame; `ObjPosFrac` is zeroed when the speed first drops. [`Link_InitSpeed`](src/bank5/link-handle-input.asm#L272) [`ObjCollidedTile`](src/Variables.inc#L341)
- The speed is set at room entry, by the grid-point rules and by "keep going" between grid points (cases 2 and 4 below), from the tile found by Link's last tile lookup; a turn back (case 3) does not set it. [`Link_InitSpeed`](src/bank5/link-handle-input.asm#L272) [`Link_ModifyDirOnGridLine`](src/bank5/link-handle-input.asm#L333)
- Monsters start a room with q-speed `$20` (0.5 pixels per frame) until their own code changes it. [`Room_EnterCommon@ResetObjLoop`](src/bank5/enter-room.asm#L259)
- One axis per frame: with several bits in the moving direction the order is right, left, down, up. [`Obj_MoveByQSpeed@ApplyQSpeedToPosition`](src/bank7/walker-move.asm#L262)
- Coordinates are 8 bits and wrap (`$00` - 1 = `$FF`). [`Obj_MoveByQSpeed`](src/bank7/walker-move.asm#L244)

**Turning at a grid point (offset 0)**
- No direction held: nothing changes and Link does not move. [`Link_ModifyDirAtGridPoint`](src/bank5/link-handle-input.asm#L117)
- One direction held: it becomes `ObjDir` and `ObjInputDir` whether or not the tile ahead is walkable (Link turns to face a wall). [`Link_ModifyDirAtGridPoint_HaveInput`](src/bank5/link-handle-input.asm#L157)
- Two held, neither walkable: `ObjInputDir` = 0, facing unchanged. [`Link_ModifyDirAtGridPoint_HaveInput`](src/bank5/link-handle-input.asm#L157)
- Two held, one walkable: he goes that way; `Link_GoStraight` is set for this step. [`Link_ModifyDirAtGridPoint_HaveInput`](src/bank5/link-handle-input.asm#L157) [`Link_GoStraight`](src/Variables.inc#L93)
- Two held, both walkable, overworld: the vertical one wins (directions are tested right, left, down, up and the last walkable one is taken). [`Link_ModifyDirAtGridPoint`](src/bank5/link-handle-input.asm#L117)
- Two held, both walkable, dungeon: the first such grid point turns Link perpendicular to his facing and sets `Link_GoStraightWhenDiagInput`; while that flag is set he keeps his facing at later grid points. [`Link_ModifyDirAtGridPoint_HaveInput@CheckGoStraightFlag`](src/bank5/link-handle-input.asm#L217) [`Link_GoStraightWhenDiagInput`](src/Variables.inc#L89)
- "Walkable" here is only the tile ahead being below `RoomFirstUnwalkableTile`; doors, room bounds and tile objects are not considered. [`Link_ModifyDirAtGridPoint`](src/bank5/link-handle-input.asm#L117)

**Between grid points (offset not 0): the six cases, in order**
- The held bits are first reduced to one direction: right, else left, else down, else up. [`Dir_GetOpposite`](src/bank1/qspeed-direction-math.asm#L137)
- Case 1, nothing held: Link stops where he is, between grid points. [`Link_ModifyDirOnGridLine`](src/bank5/link-handle-input.asm#L333)
- Case 2, same as `ObjDir`: keep going. [`Link_ModifyDirOnGridLine`](src/bank5/link-handle-input.asm#L333)
- Case 3, opposite of `ObjDir`: he reverses at once; `ObjGridOffset` is not changed, so it now counts toward 0. [`Link_ModifyDirOnGridLine`](src/bank5/link-handle-input.asm#L333) [`Link_SetDirAndInputDir`](src/bank5/link-handle-input.asm#L360)
- Case 4, perpendicular with `Link_GoStraight` set (diagonal input at the last grid point): keep going. [`Link_ModifyDirOnGridLine`](src/bank5/link-handle-input.asm#L333)
- Case 5, perpendicular, 4 or more pixels from the grid point: keep going in `ObjDir`. [`Link_ModifyDirOnGridLine`](src/bank5/link-handle-input.asm#L333)
- Case 6, perpendicular, less than 4 pixels out and still heading away from the grid point he left (offset sign agrees with the heading): he reverses and the offset is rewritten, negative n -> 8 + n, positive n -> n - 8; if the sign disagrees nothing changes. [`Link_SetInputDir@ReverseDir`](src/bank5/link-handle-input.asm#L399)
- So a perpendicular turn never happens between grid points: Link walks on (or back) to a grid point and turns there. [`Walker_Move@ChooseObjDirOrInputDir`](src/bank7/walker-move.asm#L70)
- Case 6 followed by case 3 leaves an offset counting toward 0 at the far grid point; a 2-pixel frame can step it from +1 to -1 (or -1 to +1), so no frame starts at offset 0 there and the tile, warp and screen-edge tests are skipped at that point. [`Link_ModifyDirOnGridLine`](src/bank5/link-handle-input.asm#L333) [`Obj_MoveByQSpeed`](src/bank7/walker-move.asm#L244)

**Dungeon door special cases**
- In a doorway (`DoorwayDir` nonzero) the input is forced onto the axis Link faces: forward if held, else backward if held, else (only sideways held) forward. [`Link_ModifyDirInDoorway`](src/bank5/doorway.asm#L52)
- At X = `$20` or `$D0` with two walkable directions held, Link turns perpendicular whatever the flag says, except at Y = `$85` facing down, where he keeps going down (to the side-door row Y = `$8D`). [`Link_ModifyDirAtGridPoint_HaveInput`](src/bank5/link-handle-input.asm#L157) [`Doorway_RequiredCoords`](src/bank5/doorway.asm#L93)
- At (`$78`, `$5D`) facing left or right with the flag set, he turns perpendicular (toward the top door); the bottom door has no such case. [`Link_ModifyDirAtGridPoint_HaveInput@CheckGoStraightFlag`](src/bank5/link-handle-input.asm#L217)
- In a dungeon's outer border band a held direction perpendicular to Link's facing is dropped (state 0 only). [`Link_MaskInputInBorder@MaskButtons`](src/bank5/border-input-mask.asm#L133)

**What stops movement**
- Halted: while (`ObjState` AND `$C0`) = `$40` Link's update does nothing but count the invincibility timer. [`Link_Update`](src/bank7/mode5-play-update.asm#L330)
- Sword or item use: while (`ObjState` AND `$F0`) is `$10` or `$20` the moving direction is zeroed; state `$31` does not block; a passable doorway or the laid ladder can restore it. [`Walker_Move@SetMovingDir`](src/bank7/walker-move.asm#L104)
- Knockback: while `ObjShoveDir` is nonzero directions are not processed and the shove routine moves Link instead (4 pixels per frame). [`Link_HandleInput`](src/bank5/link-handle-input.asm#L68) [`Shove_Update`](src/bank7/shove.asm#L50)
- Paralysis (`IsLinkParalyzed`, a Like-Like's hold): direction input is discarded, A and B still work. [`Link_Update`](src/bank7/mode5-play-update.asm#L330)
- An unwalkable tile ahead on a grid point zeroes the moving direction and `ButtonsPressed`. [`Link_StopAtUnwalkableTile`](src/bank7/walker-collision-screen-edge.asm#L167)
- A sliding tile object (state 1) within `$10` pixels on both axes stops Link; a person in slot 1 stops him moving up while Y < `$8E`. [`Link_CheckSlidingTileObjBlocking`](src/bank1/tile-block-power-triforce.asm#L45) [`Person_BlockLinkMovingUp`](src/bank1/modeb-cave-save-sprites.asm#L289)
- Room bounds stop Link only in a dungeon room outside doorways; in the overworld only tiles and the screen-edge test limit him. [`Walker_Move@CheckBoundary`](src/bank7/walker-move.asm#L185) [`Room_BoundDir`](src/bank1/room-bounds.asm#L254)

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
- The tile found stays in `ObjCollidedTile,X`; Link's value selects his slow speed (tiles `$74`, `$75`) and tells the armos and gravestone touch what he walked into. [`ObjCollidedTile`](src/Variables.inc#L341) [`Link_InitSpeed`](src/bank5/link-handle-input.asm#L272) [`Spawn_CheckTouchedArmosOrGravestone`](src/bank1/armos-ghini-spawn.asm#L53)

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
- Knockback also tests the tile ahead only at grid offset 0, and ends the shove there if it is unwalkable. [`Shove_MovePixels@LoopShovePixel`](src/bank7/shove.asm#L129)

**Why Link overlaps blocks**
- For an aligned Link the hotspot row covers `ObjY` + `$0B`..`$12`. Going up he is stopped by the row covering `ObjY` + 3..`$0A`; his top 11 pixels, `ObjY`..`ObjY` + `$0A`, lie above the hotspot row and are never tested. [`Tile_GetAheadOfObj`](src/bank7/tile-collision.asm#L65)
- Below a 16-pixel block he therefore stops at `ObjY` = block top + 5: over the block's whole lower tile row and 3 pixels of its upper one (trace: level 2 room `$7D`, block at Y `$B0`-`$BF`, Link stops at Y = `$B5`). [`Walker_TestTileAhead`](src/bank7/walker-collision-screen-edge.asm#L138)
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
- Link is held by those bounds only in a dungeon room outside doorways, and there on every frame, not just on grid points; in the overworld, caves and cellars only tiles and the screen edge test limit him. [`Walker_Move@CheckBoundary`](src/bank7/walker-move.asm#L185)
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
- `NextRoomId` = `RoomId` + offset: up -`$10`, down +`$10`, left -1, right +1, in 8 bits with no column test: left from column 0 gives the last room of the row above, right from column `$F` the first of the row below. [`Room_NextRoomIdOffsets`](src/bank5/next-room-calc.asm#L36) [`Room_CalcNextRoomForDoor`](src/bank5/next-room-calc.asm#L69)
- Overworld: every side counts as open, and the mazes may replace the result (rooms `$61`, `$1B`). [`Room_CalcNext_OW`](src/bank7/next-room-room-flags.asm#L102) [`Maze_CheckStep`](src/bank1/mazes-lost-woods-hills.asm#L56)
- Dungeon: the door type on the side Link faces decides; types 1 (wall) and 8 (no side matched) compute nothing and clear `NextRoomDir`. [`Room_CalcNext_JumpTable`](src/bank7/next-room-room-flags.asm#L90) [`Room_CalcNoNextRoom`](src/bank5/next-room-calc.asm#L46)
- Invalid room: a result with bit 7 set (up from row 0, down from row 7, left of `$00`, right of `$7F`) abandons the scroll and loads the overworld: mode 2, `CurLevel` = 0, `UndergroundExitType` = 2. Walking out of a dungeon's entrance works this way. [`Room_CalcNextRoomForDoor`](src/bank5/next-room-calc.asm#L69) [`Mode_LeaveLevelToOW`](src/bank5/next-room-calc.asm#L100)
- `RoomId` becomes `NextRoomId` when the mode 7 init ends. [`Mode7Scroll_Init_Finish`](src/bank5/mode7-scroll-init.asm#L319)
- The scroll carries Link to the opposite edge: he arrives at X = `$F0` (going left) or `$00` (right), Y = `$DD` (up) or `$3D` (down), the other coordinate unchanged (trace). [`Mode7Scroll_ScrollWorld`](src/bank5/scroll-world.asm#L46) [`Mode7Scroll_ScrollWorldH`](src/bank5/scroll-world.asm#L154)
- Mode 4 then enters the room: RAM `$0300`-`$051F` is zeroed, so `ObjPosFrac` and `ObjGridOffset` start at 0. In the overworld Link stays where the scroll left him; in a dungeon he is put at X = `$00` or `$F0` and walks in `$18` or `$28` pixels. [`Room_EnterCommon`](src/bank5/enter-room.asm#L137) [`Room_EnterCommon@Method2`](src/bank5/enter-room.asm#L205) [`Mode4Enter_WalkInGridOffsets`](src/bank5/mode-inits-obj-lists.asm#L144)
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
- In a dungeon room the room bounds keep Link within X `$20`..`$D0`, Y `$5D`..`$BD`; he reaches a screen edge only through a doorway (X = `$78` for up and down, Y = `$8D` for left and right), where the bounds and the tile test are skipped. [`Walker_Move@CheckBoundary`](src/bank7/walker-move.asm#L185) [`Doorway_CheckLink`](src/bank5/doorway.asm#L137) [`Doorway_RequiredCoords`](src/bank5/doorway.asm#L93)
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
- Between grid points one held direction perpendicular to `ObjDir`, with `Link_GoStraight` = 0, |`ObjGridOffset`| < 4 and the offset's sign agreeing with the heading, reverses `ObjDir` and rewrites the offset for the new heading: -3 becomes +5, +3 becomes -5 (negative n -> 8 + n, positive n -> n - 8). Link steps back that frame. [`Link_ModifyDirOnGridLine`](src/bank5/link-handle-input.asm#L333) [`Link_SetInputDir@ReverseDir`](src/bank5/link-handle-input.asm#L399)
- Next frame the original direction is the opposite of `ObjDir`: `ObjDir` and `ObjInputDir` are set to it and the offset is NOT rewritten. Its sign is now opposite to the heading, so it counts toward 0 at the grid point ahead instead of toward +-8. [`Link_ModifyDirOnGridLine`](src/bank5/link-handle-input.asm#L333) [`Link_SetDirAndInputDir`](src/bank5/link-handle-input.asm#L360)
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
- Where: overworld, play mode, no ladder laid; Link walking straight at an open screen edge with only that direction held (a diagonal at the last grid point sets `Link_GoStraight` and the press is ignored). [`Link_ModifyDirAtGridPoint_HaveInput`](src/bank5/link-handle-input.asm#L157) [`Link_GoStraight`](src/Variables.inc#L93)
- Cadence: Link must leave the last grid point before the bound (X = `$08` or `$E8`, Y = `$45` or `$D5`) with a 1-pixel frame, i.e. with the fraction at least `$80` there going left or up, below `$80` going right or down. Walking one full cell toward the edge first guarantees it: from then on every cell is 1, 2, 1, 2, 1, 1 pixels in 6 frames. [`Obj_SubQSpeedFromPosFrac`](src/bank1/qspeed-direction-math.asm#L94) [`ObjPosFrac`](src/Variables.inc#L316)
- Position: two frames after that grid point Link is 3 pixels past it, 5 pixels short of the bound: left X = `$05` (offset `$FD`, fraction `$80`); right X = `$EB` (`$03`, `$00`); up Y = `$42` (`$FD`, `$80`); down Y = `$D8` (`$03`, `$00`). [`Link_ScreenEdgeBounds`](src/bank7/walker-collision-screen-edge.asm#L70) [`ObjGridOffset`](src/Variables.inc#L311)
- Input: for exactly the next frame hold a perpendicular direction, either one. At a left or right edge release the edge direction for that frame (held together, the horizontal bit wins); at the top or bottom Up or Down may stay held (trace: Up+Left wraps at the top; Up+Left does not at the left). [`Dir_GetOpposite`](src/bank1/qspeed-direction-math.asm#L137) [`Link_ModifyDirOnGridLine`](src/bank5/link-handle-input.asm#L333)
- Then hold the edge direction; idle frames before it change nothing (trace: 1 and 5). On the fifth such frame the offset steps from +-1 to -+1 across the bound. [`Link_SetDirAndInputDir`](src/bank5/link-handle-input.asm#L360)
- Two perpendicular frames fail: the second carries Link on to the grid point behind him, where the offset is reset (trace). [`Link_ModifyDirOnGridLine`](src/bank5/link-handle-input.asm#L333)
- No other position works. The reversal needs an offset of 1 to 3. From +-1 the step back ends on the grid point behind (offset reset); in the other cadence (2 pixels first) the offsets are +-2 and +-3 and both runs land exactly on 0 at the bound (trace: every frame of the last 19 or more before the room change, both perpendicular directions, on two rooms per edge: left `$77`, `$38`; right `$77`, `$37`; top `$77`, `$3D`; bottom `$67`, `$3D`: only the position above passes the bound). [`Link_SetInputDir@ReverseDir`](src/bank5/link-handle-input.asm#L399)
- The cadence cannot be picked by timing: only moves change `ObjPosFrac` (room entry and the slow tiles zero it), and standing still keeps it. From rest on the last grid point it depends on the fraction left over (trace: reaching X = `$08` walking left, pausing, then Left wraps at `$05`; reaching it walking down, fraction `$00`, no frame wraps). A pause at X = `$07` or `$05` is harmless (trace). [`ObjPosFrac`](src/Variables.inc#L316) [`Room_EnterCommon`](src/bank5/enter-room.asm#L137) [`Link_InitSpeed`](src/bank5/link-handle-input.asm#L272)
- Reproduced (trace): all four edges, either perpendicular direction: left and right in rooms `$77`, `$38`, `$37`, `$78`; top in `$77`, `$3D`, `$5D`; bottom in `$67`, `$3D`. Not reproduced: a wrap from any other position or cadence, a second skip in a row, any wrap from a dungeon doorway (details below). [`Link_CheckScreenEdge`](src/bank7/walker-collision-screen-edge.asm#L243)

**Frame table** (trace): left edge of room `$77` on row Y = `$8D`, Left held except where shown; values after each frame. `$84` stays `$8D`; `$EB` stays `$77` in the wrap run.

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

**After the wrap, edge by edge** (c = Link's column, X / 8)
- Left: Link comes in at X = `$FF`; the first grid point is X = `$F8`, where the tile ahead is column `$1E`, then `$F0`, `$E8` and so on as from any position. On an open row he walks left across the same room, and at X = `$F0` Right starts the ordinary change to the room on the right (trace: room `$77`, Y = `$8D`). [`Tile_GetAheadOfObj`](src/bank7/tile-collision.asm#L65) [`Link_CheckScreenEdge`](src/bank7/walker-collision-screen-edge.asm#L243)
- Left onto a row whose right edge is solid: he stops at X = `$F8` and is stuck: the tile ahead (column `$1E`) and the tile under him (column `$1F`; no look-ahead at X >= `$F0`) are solid, and `$F8` is not the right bound (trace: room `$38`, Y = `$8D`; no direction moved him). [`Tile_GetAtObjOffset@AdjustX`](src/bank7/tile-collision.asm#L123) [`Link_ScreenEdgeBounds`](src/bank7/walker-collision-screen-edge.asm#L70)
- Right: the grid point X = `$F8` reads the tile under him, then X = `$00` reads column 2 (X + `$10`). Columns 0 and 1, the 16-pixel square at the left edge, are never read (trace: room `$78`, rows Y = `$75` with columns 0-1 solid and Y = `$7D` with column 0 solid: he stands at X = `$00` inside them and walks on to X = `$10`). [`Tile_GetAtObjOffset@AdjustX`](src/bank7/tile-collision.asm#L123) [`Tile_GetAheadOfObj`](src/bank7/tile-collision.asm#L65)
- Right onto a row with two solid squares at the left (trace: room `$37`, Y = `$8D`): he stops at X = `$00` inside the wall; Right, Up and Down do nothing; Left fires the edge test, because X equals the bound and the overworld makes the test even with an unwalkable tile ahead. The room changes to `$36`, where he arrives at X = `$F0` inside solid tiles and cannot walk on: the arrival spot is never tested against the new room. [`Link_StopAtUnwalkableTile`](src/bank7/walker-collision-screen-edge.asm#L167) [`Room_EnterCommon@Method2`](src/bank5/enter-room.asm#L205)
- Up (the vertical variant): Link passes `$3D` (`$3E` to `$3C`) and reaches grid points `$35`, `$2D` ... `$05` over the status bar, then `$FD`, `$F5`, `$ED` and `$E5` at the bottom of the screen. [`Obj_MoveByQSpeed`](src/bank7/walker-move.asm#L244)
- He keeps moving there because nothing bounds him and the lookup has no row check: at the ten grid points `$35`, `$2D` ... `$05`, `$FD`, `$F5`, `$ED` it reads rows 9, 8 ... 0 of columns c + 1 and c + 2, tiles of the same room one column to the right. He stops where one of them is unwalkable (trace: room `$5D`, X = `$30`, a passage 2 tiles wide: stops at Y = `$25`, column 8 row 7 solid). [`Tile_GetAtObjOffset@FetchTile`](src/bank7/tile-collision.asm#L144)
- At Y = `$E5` the lookup is back inside the map: row 21, the bottom row, of his own columns. Solid: he stops at `$E5`, 8 pixels below the bottom bound; Down takes him back through `$ED`, `$F5` (trace: room `$77`, X = `$78`). Open: he walks up into the room from its bottom edge, a complete vertical wrap (trace: room `$3D`, X = `$70`, openings at columns `$0E`-`$11` top and bottom). [`Tile_GetAheadOfObj`](src/bank7/tile-collision.asm#L65)
- Down: Link passes `$DD` (`$DC` to `$DE`) to `$E5`. At `$E5` and `$ED` the hotspot Y is `$DD` or more, so there is no look-ahead: they read rows 0 and 1 of columns c + 1 and c + 2. From `$F5` the hotspot Y has wrapped and the look-ahead is back: `$F5` to `$25` read rows 3 to 9 of those columns (row 2 is never read), `$2D` and `$35` rows 0 and 1 of his own columns. [`Tile_GetAtObjOffset`](src/bank7/tile-collision.asm#L97)
- Down, traces: room `$67`, X = `$70`: stops at `$E5` (row 0 solid there); Up brings him back to `$DD`, where Down then changes rooms normally. Room `$3D`, X = `$70`: he goes round and comes down from the top edge. [`Tile_GetAtObjOffset@FetchTile`](src/bank7/tile-collision.asm#L144)

**Passing barriers**
- The skip is not tied to the edge. 5 pixels short of any grid point, in the same cadence, the same input skips that point's tile test, and Link walks one tile (8 pixels) further before the next test. [`Walker_CheckTileCollision@CheckGridOffset`](src/bank7/walker-collision-screen-edge.asm#L101)
- So he passes an obstacle one tile deep along his heading. Into a 16-pixel block he gets 8 pixels, then the block's second tile stops him (trace, level 2 room `$7D`: below a block at Y `$B0`-`$BF` he ends at Y = `$AD`, not `$B5`; above it at `$A5`, not `$9D`; right of a block at X `$30`-`$3F` at `$38`, not `$40`; left of one at X `$60`-`$6F` at `$58`, not `$50`. Overworld room `$77`: 8 pixels into the bottom wall and into the rock at the left of row Y = `$95`). [`Walker_TestTileAhead`](src/bank7/walker-collision-screen-edge.asm#L138)
- A second skip in a row was not reproduced: after the crossing the offset has its normal sign and the limit stops Link on the next grid point. Tried on the block above: after the first press, at each of the next 14 frames, a second press (Left, Right or Down) of 1 or 2 frames, with 0 or 1 idle frame after it, then Up: 168 runs, none past Y = `$AD`. [`Obj_SubQSpeedFromPosFrac@SetCarry`](src/bank1/qspeed-direction-math.asm#L108)
- With the edge: a right wrap passes a solid 16-pixel square at the left edge; a left wrap is stopped by one at the right edge; an up wrap is stopped by a solid bottom row; a down wrap needs row 0 open one column to the right. [`Tile_GetAtObjOffset@AdjustX`](src/bank7/tile-collision.asm#L123) [`Tile_GetAtObjOffset@FetchTile`](src/bank7/tile-collision.asm#L144)
- Dungeon walls are not passed: the room bounds are tested every frame by inequality, and Link stops 1 pixel past the grid point (trace: room `$7D`, X = `$1F` at the left wall, Y = `$5C` at the top wall). [`Room_BoundDir`](src/bank1/room-bounds.asm#L254) [`Walker_Move@CheckBoundary`](src/bank7/walker-move.asm#L185)

**Why it cannot happen in dungeons**
- A dungeon screen edge is reached only inside a doorway; everywhere else the room bounds hold Link. [`Walker_Move@CheckBoundary`](src/bank7/walker-move.asm#L185) [`Doorway_CheckLink`](src/bank5/doorway.asm#L137)
- In a doorway an input that is only perpendicular is replaced by the facing direction, on every frame: Link keeps walking forward. [`Link_ModifyDirInDoorway`](src/bank5/doorway.asm#L52)
- In the dungeon's outer band (Y < `$55`, Y >= `$C6`, X < `$17`, X >= `$D9`, which covers the last cell before each bound), with state 0, the perpendicular bits are removed first and no input is left: Link stands still for that frame. [`Link_MaskInputInBorder@MaskButtons`](src/bank5/border-input-mask.asm#L133)
- Either way the reversal never runs (trace: level 2 room `$7D`, top doorway, Left or Right for one frame at every position from Y = `$5C` to `$3D`: at `$5C`-`$54` he keeps walking up with `ObjInputDir` = 8, at `$52`-`$3D` he pauses with `ObjInputDir` = 0; every run ends in the normal room change). [`Link_ModifyDirOnGridLine`](src/bank5/link-handle-input.asm#L333)
- The skip itself does work inside a dungeon room away from the doors and walls (the block traces above). [`Link_ModifyDirOnGridLine`](src/bank5/link-handle-input.asm#L333)

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
- Entering a room from a neighbour sets `DoorwayDir` to Link's facing; he walks in with grid offset +-`$18` through an open doorway or key door and +-`$28` through any other type (X = `$00` facing right, `$F0` facing left). [`Room_EnterCommon@Method2`](src/bank5/enter-room.asm#L205) [`Room_EnterCommon@ChooseWalkDistance`](src/bank5/enter-room.asm#L234)
- Doorways are not checked in cellars (mode 9). [`Walker_Move`](src/bank7/walker-move.asm#L64)

**Touching each type**
- `0` open: Link passes and walks through the doorway to the screen edge. [`Door_TouchOpen`](src/bank5/touch-door.asm#L80)
- `1` wall: always blocks; the room behind a wall is never computed as a next room. [`Door_TouchWall`](src/bank5/touch-door.asm#L75) [`Room_CalcNoNextRoom`](src/bank5/next-room-calc.asm#L46)
- `2`, `3` false wall: the first touch sets Link's `ObjTimer` to `$18` and blocks; he passes on the frame he is pushing with the timer at exactly 1; if the timer reaches 0 unseen the wait starts over. [`Door_TouchFalseWall`](src/bank5/touch-door.asm#L90)
- `4` bombable wall: passes only when its bit is in `CurOpenedDoors`. [`Door_TouchBombable`](src/bank5/touch-door.asm#L108)
- `7` shutter: passes only when no door command is running and its bit is in `CurOpenedDoors`; after the first touch of an open shutter, later touches pass only while Link's `ObjTimer` is 0. [`Door_TouchShutter`](src/bank5/touch-door.asm#L123) [`ShutterTouchedDirs`](src/Variables.inc#L432)
- Passing a false wall or an opened bombable wall starts the room change on that frame; Link then walks `$20` pixels (grid offset +-`$28` down to +-8) before the scroll. [`Doorway_CheckLink`](src/bank5/doorway.asm#L137) [`Mode6Leave_Init`](src/bank5/mode3-mode6-init.asm#L214)
- Leaving through a type `2` false wall requests the "secret found" tune; type `3` does not. [`Mode6Leave_Init`](src/bank5/mode3-mode6-init.asm#L214)

**Keys and the magic key**
- Key door, tests in this order: (1) bit in `CurOpenedDoors`: pass; (2) a door command is running (`TriggeredDoorCmd` <> 0): blocked while Link's `ObjTimer` <> 0, pass when it is 0; (3) `InvMagicKey` <> 0: open, no key spent; (4) `InvKeys` = 0: wall; (5) `InvKeys` - 1, open. [`Door_TouchKey`](src/bank5/touch-door.asm#L157)
- Unlocking sets the open command (`TriggeredDoorCmd` = 6) and Link's `ObjTimer` = `$20`; Link is blocked until the door's bit enters `CurOpenedDoors` 8 frames later. [`Door_TouchKey`](src/bank5/touch-door.asm#L157) [`Door_TriggerOpen`](src/bank5/shutters-door-flags.asm#L117)
- Keyless pass, exactly as coded: while any door command is running and Link's `ObjTimer` is 0, touching a closed key door passes with no key spent and the door still closed; the pass sets `DoorwayDir`, after which Link walks through the closed door to the next room. [`Door_TouchKey`](src/bank5/touch-door.asm#L157) [`Door_Touch_BlockWhileTimer`](src/bank5/touch-door.asm#L190) [`Doorway_CheckLink`](src/bank5/doorway.asm#L137)
- Commands that leave Link's timer at 0 and so allow the keyless pass: a shutter opening by a secret trigger and a bombed wall opening (about 8 frames each; several shutters open one after another). A key unlock sets the timer to `$20` and a close command to `$30`. [`Shutter_OpenTriggered`](src/bank5/shutters-door-flags.asm#L68) [`Bomb_TryOpenWall`](src/bank7/bomb.asm#L136) [`Door_Update`](src/bank5/update-doors.asm#L60)
- After such a pass the old room's door stays locked and unsaved; in the new room the side Link enters by is put into `CurOpenedDoors`, so a key door or bombable wall on that side is drawn open and its flag is saved there. [`Door_SetEnteringDoorwayOpened`](src/bank5/mode7-scroll-init.asm#L354) [`Door_LayOut`](src/bank5/lay-out-doors.asm#L131)
- Only one door command exists, so a key cannot unlock a door while another door is opening or closing. [`TriggeredDoorCmd`](src/Variables.inc#L85) [`Door_TouchKey`](src/bank5/touch-door.asm#L157)

**Door commands, timers and saving**
- A command is `TriggeredDoorCmd` (6 open, 2 close) plus `TriggeredDoorDir`; it runs in two steps 8 frames apart (`DoorTimer`): 6 -> 7 -> done, 2 -> 3 -> done. [`Door_Update`](src/bank5/update-doors.asm#L60) [`DoorTimer`](src/Variables.inc#L42)
- The door update runs each frame of dungeon play after Link, his weapons and the monsters, so a command set during a frame gets its first step that frame; it does nothing in mode `$12`. [`Door_Update`](src/bank5/update-doors.asm#L60)
- Opening: the side enters `CurOpenedDoors` only at the second step, 8 frames after the first; until then it counts as closed. [`Door_Update`](src/bank5/update-doors.asm#L60)
- An opened key door or bombable wall sets its flag bit in this room and the opposite side's bit in the room behind it (room id +1 E, -1 W, +`$10` S, -`$10` N; not range-checked, the neighbour's door type not looked at), so it stays open from both sides. [`Door_Update`](src/bank5/update-doors.asm#L60) [`Door_NextRoomIdOffsets`](src/bank5/lay-out-doors.asm#L365) [`RoomFlags_SetDoorOpened`](src/bank5/shutters-door-flags.asm#L41)
- A shutter that opens only enters `CurOpenedDoors`: nothing is saved, and it is closed again on the next visit. [`Door_Update`](src/bank5/update-doors.asm#L60)
- Entering a room through a side that is in `CurOpenedDoors` issues a close command for it: Link's `ObjTimer` becomes `$30` whatever the type, then only a shutter closes (flag bit cleared, bit removed 8 frames later); a key door or bombable wall stays open. [`Room_EnterCommon@Method2`](src/bank5/enter-room.asm#L205) [`Door_Update`](src/bank5/update-doors.asm#L60) [`RoomFlags_ClearDoorOpened`](src/bank5/shutters-door-flags.asm#L52)
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
- A dungeon room has a push block when attribute D bit 6 is set; the object is type `$68` in object slot `$0B`. [`Room_SetUpItemAndPushBlock`](src/bank5/create-room-objects.asm#L51) [`PushBlock_FindAndCreate`](src/bank5/uw-floor-push-block.asm#L228)
- Q1 rooms with the bit (level: room, from `data/levelblock-uw-*.md`): 1: `$22`, `$42`; 3: `$3B`; 4: `$12`, `$32`; 5: `$05`, `$06`, `$64`; 6: `$09`, `$38`, `$3A`; 7: `$0D`, `$29`; 8: `$1F`, `$7C`; 9: `$03`, `$04`, `$05`, `$10`, `$20`, `$30`, `$37`, `$55`, `$61`, `$74`. [`Room_SetUpItemAndPushBlock`](src/bank5/create-room-objects.asm#L51)
- The block is the first block tile (`$B0`) found left to right in the floor row at Y = `$90` (X `$20`-`$C0`); with none found it is put at X = `$D0`. [`PushBlock_FindAndCreate`](src/bank5/uw-floor-push-block.asm#L228)
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
- Overworld: tile `$24` (black opening), `$88` or stairs `$70`-`$73`; the room's cave index (attribute B bits 7-2) picks the target: 1-9 load that level (mode 2), `$14` the shortcut cave (mode `$C`), any other value a cave (mode `$B`). [`Warp_HandleOW`](src/bank5/warps-stairs-caves.asm#L175)
- Every transition first banks the room's kill count and passes through mode `$10`; all sound is silenced and `FluteTimer` cleared except when the target is a cellar. [`Warp_StartStairsMode`](src/bank5/warps-stairs-caves.asm#L131) [`RoomHistory_SaveKillCount`](src/bank5/warps-stairs-caves.asm#L151)
- Starting the transition clears Link's state, knockback and invincibility timer. [`Mode_EndPrepareMode`](src/bank5/cave-cellar-exits.asm#L188)

**Mode `$10` (stairs)**
- On tile `$24` Link sinks `$10` pixels, 1 pixel every 4 frames (61-64 frames), drawn behind the background; on any other tile (stairs) the mode ends on its first update frame. [`Mode10Stairs_Init`](src/bank5/mode-inits-obj-lists.asm#L82) [`Mode10Stairs_Update_Full`](src/bank5/mode8-continue-mode10-stairs.asm#L176)
- It ends by setting `GameMode` = `ModeAfterStairs` (9, `$B`, `$C` or 2). [`Mode10Stairs_Update_Full`](src/bank5/mode8-continue-mode10-stairs.asm#L176) [`ModeAfterStairs`](src/Variables.inc#L103)

**Mode 9 (cellar)**
- A cellar is a room of the level block with layout id `$3E` (tunnel with two ladders) or `$3F` (item cellar). [`Mode9Cellar_Init_LayOutCellar`](src/bank5/layout-room-ow.asm#L418)
- Link appears at Y = `$41` facing down and walks `$1C` pixels to Y = `$5D`; X = `$30` if he came from the room named by the cellar's attribute A, else `$C0`. [`Mode9Cellar_Init_Sub8_EnterCellar`](src/bank5/cave-cellar-enter.asm#L276) [`Mode9Cellar_LinkStartXs`](src/bank5/cave-cellar-enter.asm#L262) [`Mode9Cellar_Init_Sub9_WalkIn`](src/bank5/cave-cellar-enter.asm#L317)
- The cellar's own monster list is discarded; four blue Keese are made instead. [`Room_EnterCommon`](src/bank5/enter-room.asm#L137) [`Spawn_AssignPositions`](src/bank5/spawn-positions.asm#L64)
- Leaving: Link's Y < `$40` with Up held; destination room = the cellar's attribute A if his X < `$80`, else attribute B. [`Warp_CheckSubroomExit`](src/bank5/cave-cellar-exits.asm#L43)
- He comes out at X = attribute C high nibble * `$10`, Y = low nibble * `$10` + `$0D`, the same spot whichever room he goes to, facing down; the destination is marked visited. [`Warp_CheckSubroomExit`](src/bank5/cave-cellar-exits.asm#L43) [`ModeALeaveCellar_GoFromCellar`](src/bank5/cave-cellar-exits.asm#L173) [`Room_EnterCommon@FaceDown`](src/bank5/enter-room.asm#L197)
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
- Back on the overworld Link is put at the room's exit spot: X = attribute A high nibble * `$10`, Y = (attribute F bits 2-0) * `$10` + `$4D`, facing down. [`Room_EnterCommon`](src/bank5/enter-room.asm#L137)
- If he went in through an opening tile (`$24`) he starts `$10` pixels lower and rises to that Y at 1 pixel every 4 frames; if he took stairs he just appears there. [`Room_EnterCommon`](src/bank5/enter-room.asm#L137) [`UndergroundEntranceTile`](src/Variables.inc#L106) [`Mode4Enter_StepOutside`](src/bank7/update-mode-dispatch.asm#L247)

**Levels**
- A level entrance sets `CurLevel` to the cave index and `LevelEntranceRoomId` to the overworld room; caves and cellars do not touch it. [`Warp_HandleOW`](src/bank5/warps-stairs-caves.asm#L175) [`LevelEntranceRoomId`](src/Variables.inc#L463)
- A level starts with Link at X = `$78`, Y = `LevelInfo_StartY`, facing up, in `LevelInfo_StartRoomId`. [`Mode3Unfurl_Init_Sub8_LayOutRoomPlaceLink`](src/bank5/mode3-mode6-init.asm#L129) [`Mode3Unfurl_Init_Sub1`](src/bank7/init-mode-dispatch.asm#L294)
- Walking out of a level is a room change whose next room id has bit 7 set (down from row 7, up from row 0): the overworld is loaded through mode 2 with `UndergroundExitType` = 2. [`Room_CalcNextRoomForDoor`](src/bank5/next-room-calc.asm#L69) [`Mode_LeaveLevelToOW`](src/bank5/next-room-calc.asm#L100)
- The overworld then starts in `LevelEntranceRoomId` if it is not `$FF` (it is consumed: set back to `$FF`), else in the overworld's start room. [`Mode3Unfurl_Init_Sub1`](src/bank7/init-mode-dispatch.asm#L294)

# Lost Woods and Lost Hills (mazes)

The mazes are rules applied to the next-room calculation of an overworld room change; nothing in the rooms themselves is special.

- Lost Woods = overworld room `$61`; the sequence is up, left, down, left (`$08`, `$02`, `$04`, `$02`). [`Maze_LostWoodsDirs`](src/bank1/mazes-lost-woods-hills.asm#L36)
- Lost Hills = overworld room `$1B`; the sequence is up four times. [`Maze_LostHillsDirs`](src/bank1/mazes-lost-woods-hills.asm#L41)
- The direction tested is Link's facing direction `ObjDir` at the room change, once per overworld room scroll. [`Maze_CheckStep`](src/bank1/mazes-lost-woods-hills.asm#L56) [`Room_CalcNextRoomForDoor`](src/bank5/next-room-calc.asm#L69)
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
- `$3F` and `$55`: the docks; the tile object in slot `$0B` is always the dock (type `$61`); dock X is `$80` in `$55` and `$60` otherwise. [`Room_SetUpTileObjOW`](src/bank5/enter-room.asm#L423) [`Dock_Update`](src/bank4/dock-raft.asm#L39)
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
- `$36`, `$3B`, `$73`, `$44`, `$0A`, `$21`, `$41`, `$6C`: the rooms the whirlwind scrolls from for levels 1-8; Link arrives one room east (`$37`, `$3C`, `$74`, `$45`, `$0B`, `$22`, `$42`, `$6D`), same table in both quests. [`Whirlwind_PrevRoomIds`](src/bank1/whirlwind.asm#L80)
- `$0B`, `$3C`, `$74` (immediate stores) and `$0E`, `$0F`, `$22`, `$34`, `$3C`, `$45`, `$74`, `$0B` (table): rooms whose level block bytes the Q2 load patches. [`Mode2Load_Update_Full`](src/bank6/level-load-mode2.asm#L297) [`Quest_Q2OverworldPatchOffsets`](src/bank6/quest2-replacements.asm#L49)
- Level entrances are not tested by room id: a room is an entrance when its cave index (attribute B bits 7-2) is 1-9; the shortcut cave is cave index `$14`. [`Warp_HandleOW`](src/bank5/warps-stairs-caves.asm#L175)
- `LevelEntranceRoomId` = `$FF` means "no entrance remembered": the overworld then starts in the level info's start room. [`Mode3Unfurl_Init_Sub1`](src/bank7/init-mode-dispatch.asm#L294)

**Room ids in general**
- A next room id with bit 7 set (`$80` and up) is "off the map": the overworld is loaded instead. [`Room_CalcNextRoomForDoor`](src/bank5/next-room-calc.asm#L69)
- A cleared room history holds six zeroes, which is also the id of room `$00`: that room counts as already in the history until the entries are overwritten. [`RoomHistory_Clear`](src/bank7/init-mode-dispatch.asm#L271) [`Room_RunEntryTasks`](src/bank7/init-mode5-play.asm#L169)

**Layout ids (attribute D bits 5-0)**
- `$3E`, `$3F`: cellars (even = tunnel, odd = item cellar); the death sequence skips the door redraw for them. [`Mode9Cellar_Init_LayOutCellar`](src/bank5/layout-room-ow.asm#L418) [`Mode11Death_Init_Sub1`](src/bank5/mode11-death-init.asm#L73)
- `$21`: a push block would be put at (`$40`, `$80`) without searching; no room in the level data combines this layout with the push block flag. [`PushBlock_FindAndCreate`](src/bank5/uw-floor-push-block.asm#L228)
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
- It has no timer: it is put away when Link leaves its line, is more than `$10` pixels from it, or is exactly `$10` away after crossing or facing another way. [`Ladder_CheckAndDraw@StashLadder`](src/bank5/ladder.asm#L110)
- On the ladder Link may only go on in its direction or back the way he came; other input stops him. [`Ladder_CheckAndDraw@HandleInput`](src/bank5/ladder.asm#L125)
- A ladder laid upward with exactly Up held repeats the tile test 8 pixels higher and lets Link on only if that tile is walkable. [`Ladder_CheckAndDraw`](src/bank5/ladder.asm#L44)
- On the frame the ladder is put away Link does not move. [`Ladder_CheckAndDraw@StashLadder`](src/bank5/ladder.asm#L110) [`Walker_Move`](src/bank7/walker-move.asm#L64)
- While a ladder is out the screen-edge room change is not tested. [`Walker_CheckTileCollision`](src/bank7/walker-collision-screen-edge.asm#L89)
- Every room entry and exit clears `LadderSlot`. [`Obj_ResetWeaponSlotsAndLadder`](src/bank5/mode3-mode6-init.asm#L263)

**Raft** (`InvRaft`, dock tile object type `$61`)
- Docks exist only in overworld rooms `$3F` (dock X `$60`) and `$55` (dock X `$80`); without the raft the dock does nothing. [`Room_SetUpTileObjOW`](src/bank5/enter-room.asm#L423) [`Dock_Update`](src/bank4/dock-raft.asm#L39)
- A ride starts only when Link's X equals the dock X exactly and his Y is exactly `$7D` (standing at the dock: ride up) or `$3D` (at the top screen edge: ride down). [`Dock_Update`](src/bank4/dock-raft.asm#L39)
- Link is halted (`ObjState` `$40`) for the whole ride and faces up or down; the "secret found" tune plays at the start. [`Dock_Update`](src/bank4/dock-raft.asm#L39) [`Dock_RaftLinkDirs`](src/bank4/dock-raft.asm#L28)
- Raft and Link move 1 pixel per frame: 64 frames up (`$7D` to `$3D`), 66 frames down (`$3D` to `$7F`). [`Dock_Update@HandleOtherStates`](src/bank4/dock-raft.asm#L94) [`Dock_Update@State1`](src/bank4/dock-raft.asm#L117)
- Riding up ends at Y `$3D` with a room change into the room above; riding down ends at Y `$7F` with grid offset 2, so the ride up does not restart at once. [`Dock_Update`](src/bank4/dock-raft.asm#L39) [`Mode5Play_GoToNextMode`](src/bank7/walker-collision-screen-edge.asm#L281)

# Monster spawning, edge spawns and respawn (room history)

Monsters are made when a room is entered (modes 4, 9, `$B`, `$C`) from two attribute bytes of the room. Nothing refills a room while Link stays in it, apart from the special spawners listed below.

**Room entry: list and count**
- Every room entry zeroes `$0300`-`$051F`, then resets object slots `$0B`..`$01`: uninitialized, state 0, no stun or knockback, `ObjMetastate` 1 (spawn cloud), q-speed `$20`. [`Room_EnterCommon@ResetObjLoop`](src/bank5/enter-room.asm#L259)
- Monster list id = attribute byte C bits 5-0, plus `$40` when attribute byte D bit 7 is set (`$00`-`$7F`). [`Room_EnterCommon`](src/bank5/enter-room.asm#L137)
- Monster count = one of the level's four counts, chosen by attribute byte C bits 7-6; ids `$32`-`$61` (bosses and other single objects) always get count 1. [`Room_EnterCommon`](src/bank5/enter-room.asm#L137) [`LevelInfo_FoeCounts`](src/Variables.inc#L716)
- Id `$01`-`$61` is an object type: slots `$01`..count all get that type. [`Room_EnterCommon@StoreObjCount`](src/bank5/enter-room.asm#L325)
- Id `$62`-`$7F` selects one of 30 lists (index = id - `$62`): the first count bytes of the list go to slots `$01`..count; a 0 byte leaves a slot empty; lists have no end marker, so a count above the list's length reads on into the next list. [`Room_EnterCommon@PlaceList`](src/bank5/enter-room.asm#L348) [`Spawn_ObjListAddrs`](src/bank5/mode-inits-obj-lists.asm#L185) [`Spawn_ObjLists`](src/bank5/mode-inits-obj-lists.asm#L180)
- Id 0 or count 0 makes no monsters; the final count is kept in `RoomObjCount` and slot `$01`'s type in `RoomObjTemplateType`. [`Room_EnterCommon@StoreObjTemplate`](src/bank5/enter-room.asm#L372) [`RoomObjCount`](src/Variables.inc#L298) [`RoomObjTemplateType`](src/Variables.inc#L305)
- A cellar (mode 9) ignores its list: four blue Keese (`$1B`) go to slots `$01`-`$04` at X `$20`, `$60`, `$90`, `$D0`, all at Y `$9D` (the Y table is read with the wrong index register, so only its first entry is used). [`Spawn_AssignPositions@AssignSpecialPositions`](src/bank5/spawn-positions.asm#L128) [`Mode9Cellar_KeeseXs`](src/bank5/spawn-positions.asm#L44) [`Mode9Cellar_KeeseYs`](src/bank5/spawn-positions.asm#L49)
- A cave (modes `$B`, `$C`) empties slots `$01`-`$08` and puts the cave person, type `$6A` + (cave index - `$10`), in slot `$01`. [`Spawn_AssignPositions@InCave`](src/bank5/spawn-positions.asm#L155)
- The overworld tile object goes to slot `$0B`: the dock (`$61`) in rooms `$3F` and `$55`, else the rock, rock wall, tree, gravestone or armos the layout found. [`Room_SetUpTileObjOW`](src/bank5/enter-room.asm#L423)

**Spawn spots**
- Positions come from one of four 9-entry lists chosen by the lowest set bit of Link's direction on entry (right, left, down, up); every list keeps clear of the side Link enters by. [`Spawn_PosListAddrsLo`](src/bank5/mode-inits-obj-lists.asm#L103) [`Spawn_PosListFacingRight`](src/bank5/mode-inits-obj-lists.asm#L119) [`Spawn_PosListFacingLeft`](src/bank5/mode-inits-obj-lists.asm#L125) [`Spawn_PosListFacingDown`](src/bank5/mode-inits-obj-lists.asm#L131) [`Spawn_PosListFacingUp`](src/bank5/mode-inits-obj-lists.asm#L137)
- A spot byte is row in the high nibble, column in the low nibble: X = column * `$10`, Y = row * `$10` + `$0D`. [`Spawn_AssignPositions@LoopSpawnSpot`](src/bank5/spawn-positions.asm#L103)
- The walk through the list starts at `SpawnCycle` (0-8) and stores it back, so it carries over from room to room: the same room can hand out different spots on another visit. No random number is used. [`Spawn_AssignPositions`](src/bank5/spawn-positions.asm#L64) [`SpawnCycle`](src/Variables.inc#L460)
- Slots `$01`-`$09` all get a position whatever the count; an unsafe spot is skipped without advancing the slot. [`Spawn_AssignPositions@LoopSpawnSpot`](src/bank5/spawn-positions.asm#L103)
- A spot is unsafe when its tile is unwalkable (tile >= `RoomFirstUnwalkableTile`) or when it is closer than `$22` pixels to Link on both axes. [`Spawn_IsSafeSpot`](src/bank5/spawn-positions.asm#L189) [`Spawn_IsFarFromLink`](src/bank5/spawn-positions.asm#L208)
- The distance test uses 8-bit differences, so a real distance of `$DF` or more reads as `$21` or less (Link at X `$00`, spot at X `$F0` counts as `$10` apart). [`Spawn_IsFarFromLink`](src/bank5/spawn-positions.asm#L208)
- No spots are assigned when the list id is 0 or `$37` (Zelda), the count is 0, or the overworld room brings its monsters in from the screen edges. [`Spawn_AssignPositions`](src/bank5/spawn-positions.asm#L64)

**Spawn cloud**
- A slot flagged uninitialized only runs its init on its first update; while `ObjMetastate` is 1-3 the object's own update does not run and its collisions are not tested (it cannot be hit and cannot hurt Link). [`Obj_Update`](src/bank7/object-update-table.asm#L113) [`Mode5Play_UpdateWorld@LoopObject`](src/bank7/mode5-play-update.asm#L198)
- The first cloud step lasts as many frames as the slot number (`ObjTimer` = slot), the next two steps 6 frames each; then the monster acts, so higher slots appear later. [`Obj_Init@NormalSpawn`](src/bank7/object-init-table.asm#L122) [`Obj_AnimateMetastate`](src/bank7/object-animation.asm#L54) [`Obj_EndMetastate`](src/bank7/object-update-table.asm#L301)
- Type inits that replace the first step: Octoroks and Ghinis (slot + 1) * `$10` frames, Tektites 4, 8, `$14` or `$28` frames (random). [`Octorok_InitFast`](src/bank4/zora-ow-inits.asm#L217) [`Tektite_Init`](src/bank4/zora-ow-inits.asm#L172)
- No cloud at all: Boulders, Leevers, Peahats, a Zora, an Armos, a flying Ghini, every type `$53` and up, and monsters that enter from a screen edge. [`Boulder_Init`](src/bank4/zora-ow-inits.asm#L162) [`Leever_Init`](src/bank4/zora-ow-inits.asm#L192) [`Peahat_Init`](src/bank4/zora-ow-inits.asm#L241) [`Obj_Init@NormalSpawn`](src/bank7/object-init-table.asm#L122)

**Edge spawns**
- Only on the overworld, in rooms with attribute byte F bit 3; never for Zora `$11`, Armos `$1E`, Whirlwind `$2E`, Standing fire `$40` or types `$53` and up. The flying Ghini (`$22`) is not exempt. [`Obj_Init`](src/bank7/object-init-table.asm#L45) [`RoomAttrsF`](src/Variables.inc#L385)
- One monster enters at a time: while `MonstersFromEdgesLongTimer` is nonzero the monster stays uninitialized and is tried again on a later frame. [`Obj_Init@UninitMonsterFromEdge`](src/bank7/object-init-table.asm#L70) [`MonstersFromEdgesLongTimer`](src/Variables.inc#L60)
- Each attempt reloads the timer with 2 + (the slot's random byte AND 3) = 2-5 units of 10 frames, also when the attempt is refused. [`Obj_Init@InitMonsterFromEdge`](src/bank7/object-init-table.asm#L79)
- An attempt is refused when the edge cell is closer than `$22` pixels to Link on both axes; the cell's tile is not tested again. [`Spawn_IsFarFromLink`](src/bank5/spawn-positions.asm#L208)
- Position: X = cell column * `$10`, Y = cell row * `$10` - 3 (rows `$4`-`$E`, columns `$0`-`$F`). [`Obj_Init@InitMonsterFromEdge`](src/bank7/object-init-table.asm#L79) [`CurEdgeSpawnCell`](src/Variables.inc#L461)
- The cell search steps counterclockwise round the outer ring (down the left edge, right along the bottom, up the right edge, left along the top) from the stored cell and takes the first cell whose top left tile is below `$84`. [`EdgeSpawn_FindNextCell@LoopEdgeCell`](src/bank5/edge-spawn-cell.asm#L40)
- The cell persists between rooms and starts at `$40` (top left) after reset; no random number picks it. [`EdgeSpawn_FindNextCell`](src/bank5/edge-spawn-cell.asm#L35) [`Reset_ClearRam`](src/bank5/save-ram-clear-ram.asm#L81)

**Special spawns**
- Zora: overworld rooms with attribute byte A bit 3; one at a time, in the highest free slot, on a random water square (up to 13 random candidates per frame); each dive destroys it and a new one with full HP is made. [`Zora_TrySpawn`](src/bank4/zora-ow-inits.asm#L75) [`Zora_Update`](src/bank4/zora-ow-inits.asm#L299)
- A killed Zora does not come back until the room is entered again, and its death is not counted in the room's kill count. [`Zora_Update`](src/bank4/zora-ow-inits.asm#L299) [`Obj_EndMetastate`](src/bank7/object-update-table.asm#L301)
- Boulders: the spawner (type `$1F`) keeps at most 3 Boulders (`$20`) alive; a new one starts at Y `$40` at a random X in the screen half that holds the chase target, then the spawner waits 0-`$1F` frames. [`Boulder_UpdateSpawner`](src/bank4/tektite-boulder.asm#L62) [`ActiveBoulderCount`](src/Variables.inc#L431)
- Armos and flying Ghini: Link pushing into an armos square (tiles `$C0`-`$C3`) or gravestone square (`$BC`-`$BF`) at grid offset 0 makes type `$1E` or `$22` on that square in the highest free slot, fading in for `$3F` frames; none if an object already sits at that position. [`Spawn_CheckTouchedArmosOrGravestone`](src/bank1/armos-ghini-spawn.asm#L53)
- Whirlwind: the flute makes one per room visit in a free slot, at X 0 on Link's Y; in the destination room it is put in slot `$09`, whatever is there. [`Whirlwind_Summon`](src/bank1/whirlwind.asm#L242) [`Whirlwind_InitDropOffAndBeginUpdate`](src/bank1/whirlwind.asm#L359)

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
- Per monster and frame the weapon slots are tested in this order: `$0F` boomerang, `$0E` sword shot or magic shot, `$10` and `$11` bomb or fire, `$0D` sword, `$12` arrow or rod; then the monster's touch on Link. [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L97)
- `ObjAttr` bit `$20` (Digdogger `$38`/`$39`, Trap, monster shots `$53`-`$5C`): no weapon is tested, only the touch on Link. [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L97) [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70)
- While the monster's `ObjInvincibilityTimer` is nonzero nothing is tested: it cannot be hit and its touch does not hurt Link. [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L97)
- The timer is tested once, before the first weapon: two weapons overlapping a monster in one frame both deal damage, and both can run the death code (kill counters + 1 twice). [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L97)
- A monster in its spawn cloud or death sparkle (`ObjMetastate` not 0) gets no collision pass from the object loop. [`Mode5Play_UpdateWorld@LoopObject`](src/bank7/mode5-play-update.asm#L198)
- The monster's point is its middle (X + 8, Y + 8), or (X + 4, Y + 8) with `ObjAttr` bit `$40` (Gels, Patra children, Gohma, Lanmola, Moldorm, Gleeok, most shots). [`Obj_GetMiddle`](src/bank1/monster-collisions.asm#L169) [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70)

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
- `ObjHP` = the type's nibble of the HP table * `$10` (even type high nibble, odd type low nibble), loaded at init. [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L98) [`Obj_Init@FetchAttrs`](src/bank7/object-init-table.asm#L141) [`ObjHP`](src/Variables.inc#L336)
- A hit subtracts the damage; the monster dies when HP < damage or the result is 0. Nibble 0 (Gels, Keese) dies to any hit, the boomerang's damage 0 included. [`Damage_DealToMonster`](src/bank1/weapon-damage-death.asm#L181)
- `ObjInvincibilityMask` bits name damage types that do nothing: 1 sword, rod swing and sword shot; 2 boomerang; 4 arrow; 8 bomb; `$10` magic shot; `$20` fire. It is zeroed with the rest of the room's RAM on entry. [`Damage_HandleWeaponHit`](src/bank1/weapon-damage-death.asm#L127) [`ObjInvincibilityMask`](src/Variables.inc#L343)
- Masks set by code: Darknut and Wizzrobes `$F6` (sword class and bomb hurt); Pols Voice, Patra and its children, Gleeok `$FE` (sword class only); Ganon `$FA` (sword class and arrow); Gohma `$FB` (arrow); Aquamentus and Manhandla `$E2` (no boomerang, no fire); Dodongo `$FF`, `$FE` while stunned. [`Darknut_Init`](src/bank4/darknut-gibdo.asm#L44) [`Wizzrobe_DrawAndCheckCollisions`](src/bank4/wizzrobe.asm#L683) [`PolsVoice_DrawAndCheckCollisions`](src/bank4/pols-voice.asm#L197) [`Patra_Init`](src/bank4/patra-ganon-inits.asm#L41) [`Gleeok_Init`](src/bank4/gleeok-manhandla-gohma-inits.asm#L57) [`Manhandla_Init`](src/bank4/gleeok-manhandla-gohma-inits.asm#L153) [`Ganon_Init`](src/bank4/patra-ganon-inits.asm#L95) [`Gohma_Init`](src/bank4/gleeok-manhandla-gohma-inits.asm#L219) [`Aquamentus_Init`](src/bank4/aquamentus-digdogger-dodongo-inits.asm#L40) [`Dodongo_CheckCollisionsStandardSize`](src/bank4/dodongo.asm#L345)
- An immune hit plays the parry tune, except for bomb and fire. [`Damage_PlayParryTuneUnlessBombOrFire`](src/bank1/weapon-damage-death.asm#L199)
- Special cases before damage: Gohma has its own handler; a Zol or Vire takes the weapon's direction (not from the boomerang); an arrow overlapping a Pols Voice sets its HP to 0 and kills it past the mask. [`Damage_HandleWeaponHit@CheckZolVire`](src/bank1/weapon-damage-death.asm#L143) [`Gohma_HandleWeaponCollision`](src/bank4/gohma.asm#L367) [`Collision_CheckShotVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L287)
- Death start: `ObjMetastate` = `$10`, stun timer, knockback and invincibility timer cleared; the sparkle runs 6 frames for each of three images, then the slot becomes a dropped item (type `$60`). [`Damage_KillMonster`](src/bank1/weapon-damage-death.asm#L216) [`Obj_AnimateMetastate`](src/bank7/object-animation.asm#L54) [`Obj_EndMetastate`](src/bank7/object-update-table.asm#L301)

**Parry (Darknut) and Link's shield**
- A Darknut (`$0B`, `$0C`) takes no damage and no knockback when the weapon's direction and its own are exact opposites (together `$0C` or `$03`); hits from the side or behind go through. [`Damage_HandleWeaponHit@CheckDarknut`](src/bank1/weapon-damage-death.asm#L157) [`Damage_ParryOrShove`](src/bank1/sword-bomb-arrow-collision.asm#L318)
- Link parries a shot only in state `$00`-`$0F` (not swinging or using an item) and facing against it on one axis. [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)
- Any shield parries types `$53`, `$54`, `$5B` (arrow), `$5C` (boomerang); the magic shield is needed for `$55` (fireball), `$57` (sword shot), `$58`, `$59` (magic shots); `$56` and `$5A` are never parried. [`Link_CheckMonsterTouch_NoReset@FacingOpposite`](src/bank1/link-collision-harm.asm#L138) [`InvMagicShield`](src/Variables.inc#L604)

**Touching Link**
- Touch box: Link's middle (X + 8, Y + 8) against the object's middle, both axis distances under 9 pixels. [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)
- No touch is tested while Link's invincibility timer, the clock, Link's stun timer or the monster's stun timer is nonzero, while Link's state is exactly `$40` (halted) or a Like-Like holds him. [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48) [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)
- Shots (types `$53` and up) touch Link only in states `$10`-`$1F`. [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)
- A Wallmaster or Like-Like whose touch harmed Link starts its capture in the same pass. [`Collision_CheckMonsterVsWeaponsAndLink@CheckCaptors`](src/bank1/monster-collisions.asm#L147)

**Invincibility timers**
- `ObjInvincibilityTimer` drops by 1 on even frames only, so a value N lasts about 2N frames. [`Obj_DecInvincibilityTimer`](src/bank7/object-stubs-destroy-monster.asm#L119)
- Link harmed: timer `$18` (about 48 frames). Monster hit by a weapon that knocks it back: `$10` (about 32 frames). [`Shove_Begin`](src/bank1/shove-knockback.asm#L65)
- The timer is set only by the knockback code: a boomerang hit and the Pols Voice arrow kill set none, and a monster immune to the damage type gets none. [`Shove_Begin`](src/bank1/shove-knockback.asm#L65) [`Collision_CheckWeaponPointVsMonster`](src/bank1/weapon-damage-death.asm#L82)
- With the clock, `$10` is added to Link's timer every frame, so he stays invincible. [`Mode5Play_UpdateWorld`](src/bank7/mode5-play-update.asm#L129)

**Knockback**
- Link: `$20` pixels; monster: `$40` pixels; both move up to 4 pixels per frame (8 and 16 frames), one pixel at a time. [`Shove_Begin`](src/bank1/shove-knockback.asm#L65) [`Shove_MovePixels`](src/bank7/shove.asm#L124)
- Nothing new starts while the object hit still has a nonzero invincibility timer. [`Shove_Begin@CheckInvincible`](src/bank1/shove-knockback.asm#L187)
- A monster is pushed in the direction the weapon faces, not away from Link. [`Shove_Begin@MonsterDefender`](src/bank1/shove-knockback.asm#L171)
- Link is pushed away from the attacker along one axis: between grid points the axis he faces; on a grid point sideways when the attacker's middle is within 3 pixels of his height, else up or down. [`Shove_Begin@UseDistance`](src/bank1/shove-knockback.asm#L100)
- After hitting Link a monster reverses its direction, unless its `ObjAttr` has bit `$80` or it is a Vire. [`Shove_Begin`](src/bank1/shove-knockback.asm#L65)
- The first frame of a knockback moves nothing; between grid points a knockback across a monster's facing axis is cancelled (its invincibility timer stays). [`Shove_Update@CheckPerpendicularShove`](src/bank7/shove.asm#L69)
- A knockback ends early at an unwalkable tile (tested on grid points only), at the room bounds, and for an upward push while Link's Y < `$8E` in a room whose slot `$01` holds a dungeon person or the Grumble Goriya. [`Shove_MovePixels@CheckBoundary`](src/bank7/shove.asm#L141) [`Shove_MovePixels@CheckBlocked`](src/bank7/shove.asm#L160)
- Sword, rod, sword shot, magic shot, arrow, bomb and fire knock back; the boomerang does not. Gohma is knocked back only when hit on its two middle parts with the eye fully open. [`Damage_ParryOrShove`](src/bank1/sword-bomb-arrow-collision.asm#L318) [`Bomb_CheckBombOrFireVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L126) [`Shove_Begin@Gohma`](src/bank1/shove-knockback.asm#L199)

**Stun**
- A boomerang that overlaps a monster always turns back (state `$50`); unless the monster's mask has bit 2 it sets `ObjStunTimer` = `$10` and deals damage 0. [`Collision_CheckWeaponPointVsMonster`](src/bank1/weapon-damage-death.asm#L82)
- Stun timers count once every 10 frames, so `$10` is 151-160 frames; a new hit reloads it. [`ObjStunTimer`](src/Variables.inc#L52) [`Nmi_Isr`](src/bank7/run-game-nmi.asm#L140)
- A stunned monster does not move and its touch does not hurt Link; it can still be hit. [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48) [`ObjStunTimer`](src/Variables.inc#L52)

**Kill counters**
- At the killing hit: `WorldKillCount` + 1 (no cap) and `HelpDropCount` + 1 up to `$0A`; `IsHelpDropBomb` is set only when the kill that brings the count to `$0A` is a bomb kill. Deaths started without `Damage_KillMonster` (ringleader deaths, Dodongo) add nothing here. [`Damage_KillMonster`](src/bank1/weapon-damage-death.asm#L216) [`DeadDummy_StartDeathSparkle`](src/bank7/object-stubs-destroy-monster.asm#L139)
- Any harm to Link zeroes those three counters, damage 0 included (Bubble touch, whirlwind pickup). [`Link_BeHarmed@ResetHelp`](src/bank1/link-collision-harm.asm#L210)
- When the death sparkle ends: `WorldKillCycle` steps 0..9 except for dead dummy `$5D`, child Gel `$14` and red Keese `$1C`; `RoomKillCount` + 1 except for those and Zora `$11`. [`Obj_EndMetastate`](src/bank7/object-update-table.asm#L301) [`WorldKillCycle`](src/Variables.inc#L468) [`RoomKillCount`](src/CommonVars.inc#L32)
- A Dodongo killed by the sword sets `HelpDropCount` and `IsHelpDropBomb` to `$0A` itself (the next drop is a bomb). [`Dodongo_CheckCollisions`](src/bank4/dodongo.asm#L294) [`HelpDropCount`](src/Variables.inc#L72)

# Hearts, rings and death

**Heart bytes**
- `HeartValues`: high nibble = heart containers - 1, low nibble = full hearts; `HeartPartial` fills the next heart (0 empty, `$01`-`$7F` drawn half, `$80`-`$FF` drawn full). Full health is low nibble = high nibble with `HeartPartial` = `$FF`. [`HeartValues`](src/Variables.inc#L593) [`HeartPartial`](src/Variables.inc#L595) [`Text_FormatHeartRows`](src/bank1/format-decimal-hearts.asm#L81)
- A new file and a file switched to Q2 start with `$22` and `$FF` (3 hearts). [`ModeERegister_Update`](src/bank2/register-eliminate-update.asm#L42) [`Profile_SwitchToSecondQuest`](src/bank2/second-quest-bank-stub.asm#L43)

**Damage to Link**
- Damage = the touching object's entry in the damage table: low nibble = whole hearts, high nibble = 1/256 hearts in place (`$80` = half a heart), read as the 16-bit amount [0D]:[0E]. [`Link_HarmByObjType`](src/bank1/link-collision-harm.asm#L164) [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L196)
- 4 hearts: Ganon `$3E`, shots `$59` and `$5A`. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L196)
- 2 hearts: blue Lynel `$01`, blue Darknut `$0C`, Pols Voice, small and large Digdogger, Blue Wizzrobe `$23`, Patra and its children, Gibdo, Gohma, Lanmola, monster sword shot `$57`, magic shot `$58`. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L196)
- 1 heart: red Lynel `$02`, blue Goriya `$05`, red Darknut `$0B`, Blue Leever `$0F`, Vire, Zol, Like-Like, Armos, Ghinis, Red Wizzrobe `$24`, Dodongo, Manhandla, Aquamentus, Gleeok and its head, Traps, fireball `$56`, monster boomerang `$5C`. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L196)
- Half a heart: Moblins, red Goriya `$06`, Octoroks, Tektites, Red Leever `$10`, Zora, Gels, Peahat, Keese, Boulders, Wallmaster, Rope, Stalfos, guard and standing fire, Moldorm, flying rock `$53`, shot `$54`, fireball `$55`, monster arrow `$5B`. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L196)
- 0: Bubbles, whirlwind, pond fairy, rupee stash, Grumble, Zelda, dungeon persons; the touch still knocks Link back and starts his invincibility timer. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L196) [`Link_HarmByObjType`](src/bank1/link-collision-harm.asm#L164)
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
- Overworld bits 3 and 6: no reader or writer found (unverified). [`WorldFlags`](src/Variables.inc#L614)
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
- Every room entry zeroes `$0300`-`$051F`: all object slots, the room's kill count, door and secret state, candle use, the Zora and Boulder counters. [`Room_EnterCommon`](src/bank5/enter-room.asm#L137)
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
- Monster lists (ids `$62`-`$7F`) and object type names: `data/objlists.md`. [`Spawn_ObjLists`](src/bank5/mode-inits-obj-lists.asm#L180)
- Room layouts, column tables and squares (overworld, dungeon, caves, cellars): `data/room-layouts.md`. [`RoomLayout_LayOutRoom`](src/bank5/mode12-lay-out-room.asm#L178)
- Person text strings: `data/person-text.md`. [`Person_TextAddrs`](src/bank1/cave-init-person-text.asm#L47)
- The Q2 patches applied in RAM after the copy are listed in `data/README.md` "Second-quest patches". [`Mode2Load_Update_Full`](src/bank6/level-load-mode2.asm#L297)

# Second quest differences

The quest is `QuestNumbers[CurSaveSlot]`: 0 = Q1, nonzero = Q2. The code paths below are all that read it.

**Entering Q2**
- The quest number is part of the save file and is copied to `QuestNumbers` with the other slot info before the file menu. [`Mode0Demo_Update_Sub2`](src/bank2/mode0-demo.asm#L127) [`QuestNumbers`](src/Variables.inc#L542)
- A new file whose name starts with the five characters ZELDA is registered in Q2 (characters 6-8 are not compared). [`ModeERegister_Update@CompareToZelda`](src/bank2/register-eliminate-update.asm#L112) [`ModeERegister_ZeldaNameChars`](src/bank2/register-eliminate-update.asm#L33)
- Start on the last screen of the ending resets the profile for Q2: all `$180` world flags and the `$28`-byte items block zeroed, 3 hearts, bomb capacity 8, quest = 1; name and death count stay. [`Profile_SwitchToSecondQuest`](src/bank2/second-quest-bank-stub.asm#L43) [`Mode13WinGame_Update_Sub4WaitStart`](src/bank2/mode13-win-game.asm#L305)
- The quest is set to 1, not incremented: finishing Q2 starts Q2 again. [`Profile_SwitchToSecondQuest`](src/bank2/second-quest-bank-stub.asm#L43)
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
- The whirlwind's destination rooms are one table for both quests (it scrolls in beside rooms `$37`, `$3C`, `$74`, `$45`, `$0B`, `$22`, `$42`, `$6D` for levels 1-8). [`Whirlwind_PrevRoomIds`](src/bank1/whirlwind.asm#L80)

**Monsters**
- Rope: HP `$10` in Q1, `$40` in Q2; in Q2 its palette cycles every frame. [`Rope_Init`](src/bank4/rope.asm#L32) [`Rope_Update@Animate`](src/bank4/rope.asm#L125)
- Stalfos: in Q2 only it stops and fires a sword shot (type `$57`, 2 hearts): a sequence starts on 8 of 256 random values when it wants to shoot, stands `$30` frames and fires `$20` frames in; a hit cancels it. [`Stalfos_Update`](src/bank4/stalfos.asm#L43) [`Stalfos_Update@PrepareToShoot`](src/bank4/stalfos.asm#L65)
- Bubble: no code tests the quest. The sword-disabling Bubble `$2D` and the curing Bubble `$2C` come only from monster lists `$7D`-`$7F`, which only rooms of the Q2 dungeon blocks use (17 rooms); the flashing Bubble `$2B` is in both quests. [`Bubble_Update@BlockOrUnblock`](src/bank4/bubble-keese.asm#L137) [`Spawn_ObjLists`](src/bank5/mode-inits-obj-lists.asm#L180)

**Ending**
- The credits scroll ends at screen 2, scroll `$78` in Q1 and at screen 3, scroll 0 in Q2, and the lines differ: Q1 lines `$00`-`$0F`; Q2 lines `$00`-`$0B` and `$10`-`$16`, with the slot's name and death count on line `$11`. [`Credits_DrawRow`](src/bank2/credits.asm#L158) [`Credits_LastScreens`](src/bank2/credits.asm#L42) [`Credits_LastVScrolls`](src/bank2/credits.asm#L46)
