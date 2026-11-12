; RAM symbols, part A: zero page and $0100-$04BF.
;
; Reading guide:
;   - "[N]" after a name's meaning is the array length in bytes; "per object slot" arrays are indexed by
;     the object slot number (see "Object slots" below).
;   - [00]..[0F] (zero page $00-$0F) are scratch temporaries and parameters; they have no symbols.
;   - Reset_ClearRam (power-on and reset) zeroes $00-$EF, $F3, $F5-$F7 and $0300-$07FE.
;   - Room_EnterCommon (modes 4, 9, $B, $C: every room entry) zeroes $0300-$051F, so every array and
;     scalar here at $0300 or above starts a room at 0 unless the room setup writes it afterwards.

; ---- Level, game mode, frame ----
CurLevel := $10                     ; Current level: 0 = overworld (OW), 1-9 = dungeon level (UW).
IsUpdatingMode := $11               ; 0 = run the mode's init routine each frame; nonzero = run its update.
;   Set by Mode_BeginUpdate, cleared by Mode_End; tested in Nmi_Isr and Mode5Play_Update.
GameMode := $12                     ; Game mode, $00-$13; index into the Mode_Init and Mode_Update jump tables.
;   0 demo, 1 file menu, 2 load level, 3 unfurl, 4 enter room, 5 play, 6 leave room, 7 scroll, 8 continue
;   menu, 9 cellar, $A leave cellar, $B cave, $C shortcut stairs, $D save, $E register, $F eliminate, $10
;   stairs, $11 death, $12 end level, $13 win game.
GameSubmode := $13                  ; Step within the mode's init or update routine; reset to 0 between them.
TileBufSelector := $14              ; Which PPU transfer buffer Nmi_Isr sends this frame; 0 = DynTileBuf.
;   Byte offset (always even) into Transfer_BufAddrs. Reset to 0 after every transfer (Transfer_SendCurBuf).
FrameCounter := $15                 ; Incremented once per frame in Nmi_Isr (also while paused); wraps at $FF.
CurSaveSlot := $16                  ; Save slot in use, 0-2; in the file menu also 3 register, 4 eliminate.
IsVideoHeldOff := $17               ; Nonzero = Nmi_Isr leaves rendering off in frames that send DynTileBuf.
;   Mode3Unfurl_Init_Sub0 sets 1 and Mode3Unfurl_Init_Sub8_LayOutRoomPlaceLink sets 0 (the only writers; Reset_ClearRam zeroes it): the screen stays
;   off through the init steps of mode 3. Not consulted while IsSprite0CheckActive is set.

; ---- Random numbers ----
Random := $18                       ; [13] $18-$24. Random shift register, stepped once per frame in Nmi_Isr.
;   Stepped also while paused. Each step: new bit = (bit 1 of $18) XOR (bit 1 of $19); then all 13 bytes
;   rotate right as one 104-bit value, the new bit entering bit 7 of $18. Reset_ClearRam seeds $18 with $40, the
;   rest 0. Monsters read Random,X or Random+1,X with X = their object slot; Random+1 is the most used byte.

; ---- Timers decremented in Nmi_Isr ----
; Nmi_Isr runs this once per frame unless PauseState or SubmenuState is nonzero:
;   - every frame: each nonzero byte of $27-$3C (DoorTimer, ObjTimer, FluteTimer) is decremented;
;   - LongTimerCycle is decremented every frame; on the frame it passes below 0 it is reset to 9 and each
;     nonzero byte of $27-$4E is decremented. So $3D-$4E (ObjStunTimer and the "long timers") tick
;     once every 10 frames: a long timer set to N expires 10*(N-1)+1 to 10*N frames later.
; A timer stays at 0 once it gets there.
LongTimerCycle := $26               ; Divider for the 10-frame timers: counts 9 down to 0 and wraps (Nmi_Isr).
DoorTimer := $27                    ; Frames until the next step of a door opening or closing (8 per step).
ObjTimer := $28                     ; [20] per object slot, $28-$3B. Countdown in frames, 1 per frame to 0.
;   Each object type uses its entry for its own delays. Slot 0 is Link's (a door close command sets it to
;   $30 on its first step, whatever the door type). Entries of slots without an object double as mode
;   timers: ObjTimer[$0B] times the steps of the death mode ([$0A] is written once there and never read),
;   ObjTimer[1] the $40-frame flash of the chosen line in the continue menu (mode 8), [$0C] the palette
;   fade (Palette_AnimateFade), [$13] the boomerang sound interval (Boomerang_PlaySfx).
FluteTimer := $3C                   ; Frames left of the flute tune ($98 at start); the room is frozen then.
;   Flute_Wield sets $98; while nonzero Mode5Play_Update returns at once, so the whole room is frozen.
;   Warp_StartStairsMode clears it unless the target is mode 9.
ObjStunTimer := $3D                 ; [13] per object slot $00-$0C, $3D-$49. Stun left, in 10-frame units.
;   A boomerang hit sets $10 (up to 160 frames); a Dodongo stunned by bomb smoke loads $20. While nonzero a
;   monster does not move (Walker_Move, and the tests in the monsters' own update routines) and its touch
;   does not hurt Link (Link_CheckMonsterTouch). The state timers of the Blue Leever and the Zora keep running:
;   their update code tests neither the stun nor the clock. Zeroed on room entry and, when the monster
;   dies, by Damage_KillMonster.
ChaseLongTimer := $4A               ; 10-frame units until the monsters' chase target flips (0-7 at reload).
;   The target flips between Link and the decoy position; reloaded with Random+1 AND 7 in Mode5Play_Update.
MonstersFromEdgesLongTimer := $4B   ; 10-frame units until the next monster may walk in from a screen edge
;   (OW rooms with the "monsters from edges" attribute); reloaded with a random 2-5 when one does.
SwordBlockedLongTimer := $4C        ; 10-frame units during which Link's sword is blocked (flashing Bubble).
;   The flashing Bubble (type $2B) sets $10 on touch (up to 160 frames). Tested with IsSwordBlocked in
;   Link_HandleInput.
RedLeeverLongTimer := $4D           ; 10-frame units until a Red Leever may surface.
;   Set to 5 at room init (Leever_Init), 2 after each one surfaces. Another symbol for $4D is used by the
;   ending sequence.
UnusedLongTimer := $4E              ; Last byte of the range Nmi_Isr decrements every 10 frames. No code
;   reads or writes it by this address.

; ---- Help drops ----
HelpDropCount := $50                ; Kills since Link last took damage, capped at $0A; $0A forces a drop.
;   At $0A the next item drop is forced (see IsHelpDropBomb). The forced drop and the 16-kill fairy (see
;   WorldKillCount) both zero this byte and IsHelpDropBomb; so does Link_BeHarmed.
;   Dodongo_CheckCollisions sets both bytes to $0A when a weapon hit kills a Dodongo (forces a bomb drop).
IsHelpDropBomb := $51               ; Which forced drop: 0 = 5 rupees, nonzero = bomb.
;   Incremented only when the kill that brings HelpDropCount to exactly $0A is a bomb kill (damage type 8).

; ---- Link's movement and doors ----
IsDrawnItemNarrow := $52            ; Set to 1 by Anim_WriteItemSprites when the item drawn is half width (one
;   sprite); 0 otherwise. Read by Link_DrawLiftingItem to lift with one hand.
DoorwayDir := $53                   ; Direction bit of the dungeon doorway Link stands in; 0 = not in one.
;   While nonzero Link can move only along that axis (Link_ModifyDirInDoorway) and the room boundary check is
;   skipped (Walker_Move). Set on entering a room by walking; 0 in the OW.
TriggeredDoorCmd := $54             ; Door animation command: 0 none, 2 close, 6 open (Door_Update).
;   The command is incremented every 8 frames (DoorTimer): 2 -> 3 -> done, 6 -> 7 -> done, then reset to 0.
;   Only one door can be triggered at a time (Shutter_OpenTriggered and the bomb test for nonzero).
TriggeredDoorDir := $55             ; Direction bit (1 R, 2 L, 4 D, 8 U) of the door being opened or closed.
Link_GoStraightWhenDiagInput := $56 ; Flag: 1 = keep Link's direction at grid points under diagonal input.
;   Used at grid points when two directions are held and both are walkable (UW): 0 = turn to the direction
;   perpendicular to the facing direction and set the flag; 1 = keep the facing direction. Cleared when one
;   direction is held (Link_ModifyDirAtGridPoint).
Link_GoStraight := $57              ; Flag: 1 = Link keeps his direction until the next grid point.
;   Set at a grid point when two directions are held and at least one is walkable. While set, Link keeps his
;   direction between grid points when the input is perpendicular to it. Cleared at every grid point (grid
;   offset 0).
EmptyMonsterSlot := $59             ; Object slot returned by Obj_FindEmptyMonsterSlot: the highest free slot in
;   $0B..$01 (ObjType = 0). Not changed when no slot is free.
UndergroundExitType := $5A          ; Nonzero while Link leaves a cave or cellar (1) or a dungeon (2).
;   1 = cave or cellar, 2 = dungeon level (Mode_LeaveLevelToOW). Selects where Room_EnterCommon puts Link and
;   makes Warp_CheckUnderLink ignore entrances. Cleared in mode 5 once Link completes a step onto a grid point (grid
;   offset back to 0), and when a whirlwind picks Link up.
ModeAfterStairs := $5B              ; Game mode to enter when the stairs mode ($10) ends (Warp_StartStairsMode).
LadderSlot := $64                   ; Object slot holding the laid-down ladder; 0 = ladder not in use.
;   Reset by Obj_ResetWeaponSlotsAndLadder (room entry and exit).
UndergroundEntranceTile := $65      ; OW tile Link stood on when he went underground ($24 or $70-$73).
;   The tile under Link at the OW warp check (Warp_HandleOW). $24 = cave or dungeon opening: Link sinks into
;   it and later rises out of it; any other value (stairs $70-$73) skips that walk.

; ---- Scrolling and name tables ----
VScrollAddrHi := $58                ; High byte of the mid-frame PPU address for vertical room scrolling.
;   Written to the PPU address register at the sprite-0 split while a room scrolls vertically (with
;   VScrollAddrLo). Also used by the credits scroll.
SwitchNameTablesReq := $5C          ; Nonzero = Nmi_Isr toggles the base name table ($2000/$2800) this frame.
;   Reset to 0 after each transfer (Transfer_SendCurBuf).
SubmenuScanRoomId := $5D            ; Room id ($7F down to 0) being scanned to draw the submenu dungeon map.
SubmenuScrollProgress := $5E        ; Countdown from $2B while the submenu scrolls in (row * 2 + flag).
;   Decremented every frame; bits 1-7 = submenu row being transferred, bit 0 = 1 for a row of black tiles.
;   Negative when everything is transferred.
OddBaseNameTableOverride := $5F     ; 0/1 ORed into PPU control bit 0 at the split during horizontal scroll.
;   During a horizontal room scroll 1 = the scroll position is based on name table 1.
VScrollAddrLo := $E2                ; Low byte of the mid-frame PPU address for vertical room scrolling.
IsSprite0CheckActive := $E3         ; Nonzero = Nmi_Isr waits for the sprite-0 hit and splits the scroll.
;   The split is below the status bar. Controller input is not read while set. Set while a room scrolls
;   (mode 7) and in some cave, cellar and death steps.
VScrollStartFrame := $E6            ; FrameCounter phase on which vertical room scrolling moves one row.
;   A row moves every 2nd frame in the OW, every 4th in the UW (Mode7Scroll_ScrollWorld).
CurColumn := $E8                    ; Tile column being copied to the transfer buffer; valid 1..$20.
;   Used by the room scroll and the curtain effect; Transfer_CopyColumnToTileBuf copies only for 1..$20.
CurRow := $E9                       ; Tile row being copied to the transfer buffer, 0-$15; $16 or more = none.
;   Mode 7 leaves $FF, or 0 when the new room is dark, which makes mode 4 redraw the play area.
PrevRow := $ED                      ; Last row copied by Transfer_CopyColumnOrRowToTileBuf (a row is not copied twice).
CurVScroll := $FC                   ; Vertical scroll value Nmi_Isr writes to the PPU scroll register.
;   Written in modes 0, 5, 9, $B, $C, $13 once the mode is updating: submenu, story and credits scrolling.
CurHScroll := $FD                   ; Horizontal scroll value Nmi_Isr writes to the PPU scroll register.
;   Changed by Mode7Scroll_ScrollWorld while a room scrolls left or right; also written at the sprite-0 split.
CurPpuMask := $FE                   ; Copy of the last value written to PPU mask register $2001.
CurPpuControl := $FF                ; Copy of the last value written to PPU control register $2000.

; ---- Monsters' chase target ----
IsChasingOtherTarget := $60         ; 0 = monsters chase Link; 1 = they chase the stored decoy position.
;   While 0, ChaseTargetX/Y are copied from Link every frame; while 1 they are left alone. Toggled each time
;   ChaseLongTimer expires.
ChaseTargetX := $61                 ; X that chasing monsters steer toward (Link, a decoy point, or food).
;   Link's X while IsChasingOtherTarget = 0. On the switch to 1, if it equals Link's X (it was copied from it
;   earlier in the same frame), X and Y are both XORed with $FF: a point mirrored across the room, which
;   monsters then walk toward instead of Link. Food on the ground overwrites the target with the food's
;   position for the monster types it attracts (Boomerang_UpdateOrFood).
ChaseTargetY := $62                 ; Y the chasing monsters steer toward (see ChaseTargetX).

; ---- Hearts refill ----
World_IsFillingHearts := $63        ; Nonzero while hearts refill (potion, pond fairy, end of level).
;   Set by: potion (1), pond fairy ($40), end of level (2). Cleared by World_FillHearts when hearts equal
;   heart containers.

; ---- Audio (bank 0) ----
SongScriptPtrLo := $66              ; Pointer (2 bytes lo/hi, $66-$67) to the current song phrase script.
SongScriptPtrHi := $67              ; High byte of the song script pointer.
EffectScratch := $68                ; Noise effect scratch: stairs effect, frames left of one step ($0C to 1);
;   sea effect, the noise volume byte ($10 up and back down to $10).
EffectCounter := $69                ; Steps left in the current noise-channel sound effect; counts down to 0.
CurNoteLowPeriodSq0 := $6A          ; Low period byte of the note playing on square channel 0 (for vibrato).
CurNoteLowPeriodSq1 := $6B          ; Low period byte of the note playing on square channel 1 (for vibrato).
SongPhraseIndex := $6C              ; Index into Song_PhraseHeaderOffsets of the current phrase of a multi-phrase song.
CustomEnvelopeOffsetTune1 := $6D    ; Offset into Audio_Tune1EnvelopeDuties; counts down from $1F during a note.
NoteLengthTune1 := $6E              ; Length in frames of notes on tune channel 1 (set by the tune script).
NoteCounterTune1 := $6F             ; Frames left of the current note on tune channel 1.

; ---- Object slots ----
; The arrays below named Obj... hold one byte per object slot. Slots, as Mode5Play_Update uses them:
;   $00       Link.
;   $01-$0B   monsters and other room objects, updated each frame from $0B down to $01. $0B is also
;             where the room's tile object goes (push block, rock, dock, ...). Obj_FindEmptyMonsterSlot
;             hands out $0B..$01 only.
;   $0C       never handed out; its entries are scratch (ObjX[$0C] and ObjX[$0D] are the curtain
;             columns in mode 3, ObjTimer[$0C] is the palette fade timer).
;   $0D       sword                         (Sword_UpdateSwordOrRod)
;   $0E       sword shot or magic shot      (SwordShot_UpdateOrMagicShot)
;   $0F       boomerang or food             (Boomerang_UpdateOrFood)
;   $10, $11  bomb or candle fire           (Bomb_UpdateBombOrFire, one call per slot)
;   $12       arrow or rod                  (Rod_UpdateRodOrArrow; rod states go on to Sword_UpdateSwordOrRod)
;   $13       the room item: ObjX/ObjY[$13] its position, ObjDir[$13] = RoomItemId its item id,
;             ObjState[$13] = $FF while it is hidden or absent.
; Arrays of 20 bytes cover slots $00-$13; arrays of 13 cover $00-$0C (Link, monsters, slot $0C) and
; are followed by other variables, so weapon slots must not index them.

; ---- Object slot arrays (zero page) ----
ObjX := $70                         ; [20] per object slot, $70-$83. X of the object's left edge in pixels.
ObjY := $84                         ; [20] per object slot, $84-$97. Y of the object's top edge in pixels.
;   The play area starts at Y = $40 below the status bar. On a grid point Y = (multiple of 8) OR 5, e.g. $7D
;   (Obj_SnapToGrid).
ObjDir := $98                       ; [20] per object slot, $98-$AB. Direction: 1 right, 2 left, 4 down, 8 up.
;   A thrown boomerang may hold two bits (diagonal). A spreading sword shot reuses its entry as a pixel
;   offset.
RoomItemId := $AB                   ; = ObjDir[$13]. Item id of the room item.
;   Room_SetUpItemAndPushBlock stores the raw 5-bit item attribute of a dungeon room whose item is not yet taken (a
;   room without an item holds 3, with ObjState[$13] negative), and $1A for the OW heart container; the
;   Armos secret stores $14. Item_MoveAndDrawRoomItem skips the item when the value is $3F (nothing stores $3F
;   through this symbol) or when ObjState[$13] is negative.
ObjState := $AC                     ; [20] per object slot, $AC-$BF. State; meaning depends on object type.
;   Link (slot 0): 0 = idle (only then are A and B handled). $10 = sword or item just used; each time Link's
;   ObjAnimCounter runs out Link_AdvanceItemUseState steps $10 -> $11 -> $31 -> 0 (and $20 -> $21 -> $31 -> 0).
;   (state AND $C0) = $40: Link is halted, Link_Update skips input and movement (item lift, persons, pond
;   fairy, whirlwind, raft ride, Wallmaster capture, ...). Weapons (slots $0D-$12): 0 = inactive; high
;   nibble = major state, low nibble = step. Sword 1-5; rod $31-$35; sword shot $10 flying, $11 spreading;
;   magic shot $80 (bit 7 set);
;   boomerang $10 out, $2x-$3x slowing at the far end, $40 returning, $50 returning fast; food $80-$82; bomb
;   $11-$14 (blast damages at $13); fire $21 moving, $22 standing; arrow $10 flying, $2x spark. Room item
;   (slot $13): 0 = present, $FF = hidden or none.
ObjShoveDir := $C0                  ; [19] per object slot, $C0-$D2. Knockback direction bits; 0 = none.
;   Bit 7 = knockback just started (Shove_Update clears it on its first call). Shove_Begin also sets bit 6 for
;   monsters with attribute $80; no code reads that bit (Shove_Update masks the direction bits it tests and
;   every other reader only tests the byte for zero).
ObjShoveDistance := $D3             ; [13] per object slot $00-$0C, $D3-$DF. Pixels of knockback left.
;   $20 for Link, $40 for a monster (Shove_Begin); Shove_Update moves up to 4 pixels per frame.

; ---- Pause, submenu, death ----
PauseState := $E0                   ; 0 running; 1 paused with Select; 2 paused by the game (potion refill).
;   Select is ignored in state 2. While nonzero the world is not updated and the Nmi_Isr timers do not tick.
SubmenuState := $E1                 ; Item submenu state: 0 closed, 1 opening ... 7/8 active, 8/9 closing.
;   1 is set by Start in mode 5; then scroll-in steps; 7 (UW) or 8 (OW) active; 8 (UW) or 9 (OW) scrolling
;   out. While nonzero the world and timers are frozen.
SubmenuSearchDir := $EF             ; Direction bits of the last item-slot search in the submenu.
;   Stored by Inv_FindAndSelectOccupiedSlot (bit 0 = step forward, else bit 1 = backward; Inv_StepItemSlotInSearchDir);
;   the submenu's selection update acts only when the input direction differs from it.
DeathModeCounter := $E5             ; Death mode ($11) countdown: spins left, later frames of the spark.
;   Link's remaining spins (starts at 4), then the frames left of the spark (starts at $0F).

; ---- Rooms ----
WhirlwindPrevRoomId := $EA          ; Room the screen scrolls in from when the whirlwind delivers Link.
;   Taken from Whirlwind_PrevRoomIds for the level; used as the room id during the scroll.
RoomId := $EB                       ; Current room, $00-$7F: high nibble = map row, low nibble = column.
NextRoomId := $EC                   ; Room being entered (Room_CalcNext); bit 7 set = no such room.
NextRoomDir := $E7                  ; Direction bit of the side Link leaves by, for the next-room calculation.
;   Stored by Room_CalcNext, read by Room_CalcNextRoomForDoor; zeroed for door types without a room
;   behind them, when the level is left, and at the end of the room scroll (mode 7).
CurOpenedDoors := $EE               ; Direction bits of this room's true doors that are open now.
;   Bits: 1 right, 2 left, 4 down, 8 up. Bits for plain openings and walls are cleared when the doors are
;   laid out.

; ---- Boot and bank flags (not cleared with the rest of the zero page) ----
IsScrollMirroringSet := $F3         ; Nonzero = name table mirroring is set for the room scroll (mode 7).
;   Mode7Scroll_Init sets it (and the mirroring) once the sprite-0 check is on, Mode7Scroll_Update clears it at the
;   end.
IsGameInitialized := $F4            ; Nonzero once the common code and data were copied to RAM.
;   Tested by Mode_InitGameOrMode, which does the copy once.
HasCommonPatterns := $F5            ; $5A once the common pattern tables were sent to the PPU (Mode0Demo_InitPatternsOrDemo).
HasDemoPatterns := $F6              ; $A5 while the demo pattern tables are loaded (tested by Mode0Demo_InitPatternsOrDemo).
ReturnToBank4 := $F7                ; Nonzero = switch back to bank 4 after changing a tile-object tile.
;   The bank 4 callers increment it before the call; the routine clears it on return.

; ---- Controllers ----
ButtonsPressed := $F8               ; [2] controller 1, 2. Buttons newly down this frame.
;   Bits: $80 A, $40 B, $20 Select, $10 Start, 8 Up, 4 Down, 2 Left, 1 Right. Not read while
;   IsSprite0CheckActive is set.
ButtonsDown := $FA                  ; [2] controller 1, 2. Buttons held this frame; bits as ButtonsPressed.

; ---- Movement limits ($0100 page) ----
UnusedScrollEndByte := $10C         ; Zeroed by the last step of the room scroll (Mode7Scroll_Update_Sub7).
;   No code reads it.
PositiveGridCellSize := $10E        ; Positive limit of the grid offset: 8 for Link, $10 for other objects.
NegativeGridCellSize := $10F        ; Negative limit of the grid offset: $F8 (-8) for Link, $F0 for others.

; ---- Sprites and PPU transfer buffer ----
Sprites := $200                     ; [256] $0200-$02FF. Sprite (OAM) buffer copied to the PPU every frame.
;   64 records of Y, tile, attributes, X. Y = $F8 hides a sprite. Link uses records $12 and $13 (offsets $48,
;   $4C); room objects use the cycled records from offset $60 up.
DynTileBufLen := $301               ; Offset in DynTileBuf of its end marker, where the next record goes.
DynTileBuf := $302                  ; [61] $0302-$033E. Dynamic PPU transfer buffer (records, $FF-terminated).
;   A record is: address hi, address lo, control (bit 7 = step 32, bit 6 = repeat one byte, bits 0-5 = count,
;   0 = 64), data; a byte with bit 7 set where an address is expected ends it. Emptied ($FF) after each
;   transfer.
RollingSpriteIndex := $341          ; 0-$27. Index into Sprite_RollingOffsets for the next object's sprite pair.
;   Advanced by Sprite_CycleCurIndex as objects are drawn, so they take the sprite records in turn.
FirstSpriteIndex := $342            ; RollingSpriteIndex the frame starts with; advances by 1 every frame.
;   Sprite_HideObjectSprites does it, which rotates the sprite records between objects from frame to frame.
LeftSpriteOffset := $343            ; Offset into Sprites of the record for the drawn object's left half.
RightSpriteOffset := $344           ; Offset into Sprites of the record for the drawn object's right half.

; ---- Room state ----
OpenDoorwayMask := $33F             ; Submenu map scratch: direction bits of a room's passable sides.
;   Bit set = that side can be walked through (Door_ShiftOpenDoorwayMask); $13 marks an unvisited room.
;   Door_LayOut calls Door_ShiftOpenDoorwayMask too, which rewrites the byte; only the submenu map reads it.
CurObjIndex := $340                 ; Object slot being updated; $0B down to 1 each frame in mode 5.
;   Mode5Play_Update leaves it at $0B after the loop.
RoomBoundLeft := $346               ; Left X bound for objects: $11 in the OW, $21 in the UW.
;   Tested by Room_BoundDirHorizontally. This and the next four bytes are copied together by
;   Room_SetUpObjBounds.
RoomBoundRight := $347              ; X bound on the right: $E0 in the OW, $D0 in the UW.
RoomBoundUp := $348                 ; Y bound at the top: $4E in the OW, $5E in the UW.
RoomBoundDown := $349               ; Y bound at the bottom: $CD in the OW, $BD in the UW.
RoomFirstUnwalkableTile := $34A     ; Tiles at or above this number block movement: $89 OW, $78 UW.
IsShotCollidingWithLink := $34B     ; Set by Link_CheckMonsterTouch when a monster's shot touches Link.
;   A shot is object type $53 or above; MonsterShot_CheckLinkCollision reads it to start the shot's bounce.
MonsterShotCount := $34C            ; Counted monster shots in flight; Shooter_ShootLimited allows at most 4.
IsRoomAllDead := $34D               ; Nonzero once no monsters are left in the dungeon room.
;   Bubbles ($2B-$2D) and types $49 and above do not count. Opens shutters and triggers "all dead" secrets.
RoomObjCount := $34E                ; Number of monsters the room makes on entry.
;   Taken from the room attributes after the room history is applied; some monsters adjust it (Zol and Vire
;   splitting, Digdogger, Moldorm, Lanmola).

; ---- Object slot arrays ($034F-$04BF) ----
ObjType := $34F                     ; [13] per object slot $00-$0C, $34F-$35B. Object type id; 0 = empty slot.
;   Entry 0 is not Link's type: that byte has its own symbol as the room's kill count.
RoomObjTemplateType := $35F         ; Object type of the first monster the room was populated with.
;   It is slot 1 at room entry. Read by the dungeon room history (non-recurring monsters) and by food (which
;   types it attracts).
ObjMovingLimit := $380              ; [20] per object slot. Per-type limit: boomerang range, Trap home spot.
;   A boomerang's maximum distance in pixels ($31 wooden, $FF magic, $51 a Goriya's), then its turn and return
;   timers; a Trap's home coordinate. Other symbols for $380 belong to other object types.
ObjGridOffset := $394               ; [20] per object slot. Signed pixels moved since the last grid point.
;   0 = on a grid point. Limits +-8 for Link, +-$10 for monsters (see PositiveGridCellSize); Link's is set
;   back to 0 when it reaches a multiple of 8. Link_HandleInput applies different turning rules at 0 and off
;   0, and knockback tests tiles only at 0 (Shove_Update). Also the pixels left to walk in modes 4 and 6, and a
;   boomerang's distance flown.
ObjPosFrac := $3A8                  ; [20] per object slot. Fraction (1/256 pixel) of the moving position.
ObjQSpeedFrac := $3BC               ; [20] per object slot. Q-speed; pixels per frame = value / 64.
;   Obj_MoveByQSpeed adds it to ObjPosFrac 4 times per frame and moves 1 pixel per carry, so pixels per frame =
;   value / 64 ($20 = 0.5, $60 = 1.5 Link's normal speed, $30 = 0.75 Link on OW stairs, $C0 = 3). Room entry
;   sets $20 for slots 1-$0B.
ObjAnimCounter := $3D0              ; [20] per object slot. Frames until the next animation frame.
;   For Link its reaching 0 also advances his item-use state (Link_AdvanceItemUseState).
ObjAnimFrame := $3E4                ; [20] per object slot. Animation frame; walking objects alternate 0, 1.
ObjInputDir := $3F8                 ; [13] per object slot $00-$0C. Direction bits the object wants to go.
;   Link: the direction bits held on controller 1 (low nibble of ButtonsDown), then reduced by
;   Link_HandleInput. Monsters: the direction chosen by their movement code.
ObjMetastate := $405                ; [13] per object slot $00-$0C. 0 normal; 1-4 spawn cloud; $10-$14 dying.
;   1-4 are the spawn cloud, $10-$14 the death sparkle. A step ends when the slot's ObjTimer reaches 0 and the
;   next lasts 6 frames; the first cloud step lasts as many frames as the slot number (Obj_Init). 4 returns
;   to 0; $14 turns the slot into a dropped item (type $60). Room entry sets 1 for slots 1-$0B. Only at 0 does
;   the object's own update routine run.
ObjShootTimer := $451               ; [13] per object slot $00-$0C. Countdown to a walking monster's shot.
;   Used by Shooter_RunShootTimer (Moblin, Lynel, Octorok) and, in Q2 only, by Stalfos: set to $30 when the monster
;   decides to shoot, decremented once per update; the monster stands still while it is nonzero and fires
;   when it reaches $10. Other object types use $451 under other names.
ObjHP := $485                       ; [13] per object slot $00-$0C. Hit points in the high nibble (HP x $10).
;   Loaded from Obj_TypeHpPairs at init. Damage_DealToMonster subtracts the weapon's damage; the monster dies when
;   the damage is at least the HP left.
ObjUninitialized := $492            ; [12] per object slot $00-$0B. Nonzero = init routine not yet run.
;   Room entry sets $FF for slots 1-$0B; Obj_Update then runs Obj_Init and clears it.
ObjCollidedTile := $49E             ; [20] per object slot. Tile found by the last tile collision test.
;   The test looks ahead of the object in its moving direction. In the OW walkable tiles are stored as $26.
ObjInvincibilityMask := $4B2        ; [13] per object slot $00-$0C. Damage types that cannot hurt the monster.
;   Bits: 1 sword (and rod, sword shot), 2 boomerang, 4 arrow, 8 bomb, $10 magic shot, $20 fire. Some masks
;   also have bits $40 and $80 (Aquamentus: $E2); no damage type uses those bits.
ObjAttr := $4BF                     ; [13] per object slot $00-$0C. Behaviour bits (Obj_TypeAttributes).
;   Copied when the object is initialized. Bits: $01 the object checks collisions and draws itself; $02 drawn
;   half width; $04 draws itself (collisions still checked for it); $08 does not use the sprite attribute
;   table; $10 turns around when blocked (no object type has it: the code that tests it never branches);
;   $20 Link's weapons cannot touch it; $40 half width for collisions; $80 does not turn around after
;   hitting Link (Shove_Begin also marks its knockback direction with bit 6, which nothing reads).

; ---- Title, story demo, name registration (alias addresses of object variables) ----
TriforceGlowCycle := $413           ; Title screen: step 0-7 of the Triforce glow palette cycle.
DemoScrolledScreenCount := $415     ; Story demo: number of whole screens scrolled so far.
DemoLineAttrVramAddrLo := $417      ; Story demo: PPU address (lo; hi at $418) of the next line's attributes.
DemoLineAttrVramAddrHi := $418      ; Story demo: high byte of the attribute PPU address.
DemoLineIndex := $419               ; Story demo: index of the next line of the scroll (into Demo_LineAttrs).
DemoTimer := $41A                   ; Demo: frame counter for the waits between demo steps (counts up).
DemoScrolledLineCount := $41B       ; Story demo: pixel lines scrolled, wraps every 256 (one screen).
DemoLineTileVramAddrLo := $41C      ; Story demo: PPU address (lo, hi at $41D) of the next line's tiles.
DemoLineTileVramAddrHi := $41D      ; Story demo: high byte of the tile PPU address.
IsHoldingButton := $426             ; Name registration: nonzero while a direction button is being held.
;   Same address as IsFileBReadyToSave, which the Register update code uses once the name is entered.
HeldButton := $427                  ; Name registration: the direction bits held when the current hold began.
IsButtonRepeating := $428           ; Name registration: 0 = next timer value $10, 1 = 8 (later repeats).
ButtonRepeatTimer := $429           ; Name registration: frames until a held direction moves the cursor again.
;   The move happens on the frame after the timer reaches 0: $11 frames to the first repeat, then every 9.
ModeE_HasWrappedBoardY := $42A      ; Name registration: 1 = the last cursor move wrapped vertically.
;   Set by ModeERegister_StepCharBoardCursorY when the cursor passes the top or bottom row of the character board.
DemoSubphase := $42D                ; Demo: step within the current demo phase.
DemoLineTextIndex := $42E           ; Story demo: index of the next text line (into Demo_LineTextAddrs).
DemoItemRow := $42F                 ; Story demo: next item list row (Demo_LeftItemIds, Demo_RightItemIds).
DemoPhase0Subphase1Timer := $438    ; Title screen fade: frames until the next palette step.

; Second part of the RAM map: $4CC-$67F game state, sound driver state and inventory; hardware registers;
; battery-backed save RAM ($6000-$7FFF) with the save files, the play area tile map, the level block
; and the level info.
; NOTE: Room_EnterCommon (modes 4, 9, $B, $C) zeroes $300-$51F on every room entry, so every symbol
;   below $520 starts each room at 0. $520-$7FE is zeroed only by Reset_ClearRam at reset.

; ---- Room state, cleared on room entry ----
IsPersonFireballsEnabled := $4CC    ; Nonzero after the person in slot 1 was hit: the room's statues shoot
;   Set from ObjMetastate+1 by the person update; Statue_Update then fires with pattern 2.
RoomAttrsF := $4CD                  ; Copy of LevelBlockAttrsF[RoomId], taken on room entry (bits: see there).
ShutterTrigger := $4CE              ; 1 = open the room's shutter doors; cleared when none is left closed.
;   Set by Shutter_SetTrigger (room secret met), the life-or-money man when paid, and the level 9
;   old man (type $4B) when InvTriforce = $FF.
BlockPushComplete := $4CF           ; Push block: 0 not pushed, 1 fully pushed, 2 pushed and stairs made.
UnusedUniqueRoomId := $4E4          ; The current room's unique room id, stored by Room_CalcNextRoomForDoor.
;   No code reads it.
StatusBarMapTrigger := $4E5         ; 1 = transfer the level's status bar map (set when the map is picked up).

; ---- Flyers and Gleeok, cleared on room entry ----
FlyerMaxSpeedFrac := $4D1           ; Flyer top speed: speeding up ends when (Flyer_ObjSpeed & $E0) >= this.
;   One byte for the whole room, written by the inits: $40 Patra, $80 Moldorm, $A0 Flyer_FinishInit and
;   the item fairy, $C0 Keese_InitBlue, $E0 Gleeok_InitFlyingHead.
GleeokCurNeck := $4D7               ; Gleeok: neck being processed, counted down from (ObjType - $42) to 0.
GleeokSignedRefSegmentDistance := $4D8 ; Gleeok: signed distance between reference neck segments.
GleeokPrimarySegmentLimits := $4D9  ; [2] X, Y. Gleeok neck segment spacing limit, tier 1: at most 4 pixels.
GleeokSecondarySegmentLimits := $4DB ; [2] X, Y. Tier 2 spacing limit: tier 1 + 4, at most 8 pixels.
GleeokTertiarySegmentLimits := $4DD ; [2] X, Y. Tier 3 spacing limit: tier 2 + 4, at most $B pixels.
GleeokAnimationTimer := $4E6        ; Gleeok body: frames until the next animation frame (6 while writhing).
GleeokBodyAnimationFrame := $4E7    ; Gleeok body animation frame, cycles 0-3.
GleeokDeadNeckMask := $511          ; Gleeok: bit n set = the head of neck n has been cut off.

; ---- Object slot arrays ----
ObjInvincibilityTimer := $4F0       ; [20] per object slot $00-$13. Nonzero = cannot be harmed or shoved.
;   Counts down 1 every second frame (Obj_DecInvincibilityTimer); sprites flash meanwhile.
;   Link gets $18 (48 frames) when harmed, a monster $10 (32 frames) when hit.
;   While InvClock is set, $10 is added to Link's timer every frame.

; ---- Item lift, drawing and counters, cleared on room entry ----
LeftAlignHalfWidthObj := $504       ; Nonzero while the lifted item is drawn: a narrow item is not centered.
ItemTypeToLift := $505              ; Item id Link is holding up, 0 = none ($1B Triforce piece in scenes).
ItemLiftTimer := $506               ; Frames left of the lift; $80 when an item is taken outside mode 5.
;   Link_CheckLiftItem decrements it first and keeps Link halted while it is nonzero ($7F halted frames); Keese
;   do not move meanwhile. Item id 0 (bomb) is never lifted: ItemTypeToLift = 0 means none. Mode $13 reuses
;   the byte as a frame counter.
ChildDigdoggerCount := $507         ; Small Digdoggers the big one splits into: 3 (Digdogger_InitWith3Children) or 1 (..2).
IsWhirlwindSummoned := $508         ; 1 = the flute already summoned a whirlwind in this room: no second one.
IsTriforceFanfareActive := $509     ; 1 while Link holds up the Triforce of Power (Link halted, ObjTimer $C0).
LanmolaViableDirMask := $50F        ; Lanmola turn scratch: direction bits allowed (all but the opposite one).
ActiveRedLeeverCount := $510        ; Red Leevers (type $10) above ground; a third may not surface.
IsLinkParalyzed := $512             ; Nonzero while a Like-Like holds Link: direction input is discarded.
;   Cleared when the Like-Like dies or the room is found empty of monsters.
HasUsedCandle := $513               ; 1 = a candle fire was made in this room; the blue candle then refuses.
;   Only tested when InvCandle = 1, so the red candle is unlimited.
IsZoraActive := $514                ; 1 while the room's Zora exists (one at a time).
ActiveBoulderCount := $515          ; Boulders (type $20) alive; the boulder spawner stops at 3.
ShutterTouchedDirs := $519          ; Direction bits of the open shutters Link has touched in this room.
;   Door_TouchShutter: the first touch of an open shutter passes and sets its bit; later touches pass only
;   while Link's ObjTimer is 0.
SecretColorCycle := $51A            ; Flute pond secret: 0 none, 1-$B draining, $C done.
;   Draining steps once every 8 frames (FrameCounter AND 7 = 4): at $A the water becomes walkable, at $B
;   the stairs appear. While Link leaves the room it counts back down to 0, one step on every frame with
;   FrameCounter bit 2 set (4 frames of every 8); the room scroll waits for it.
FluteUsedState := $51B              ; 0; 1 once the flute was played in this room; 2 while Digdogger shrinks.
FadeCycle := $51C                   ; Palette fade: high nibble = series, low nibble = step 0-3 (4 = done).
;   Bit 7 set = run the series backwards. $00 into a cellar, $40/$C0 dark room and back, $60 death.
;   One step per 10 frames (Palette_AnimateFade); data in LevelInfo_PaletteCycles.
PatternBlockIndex := $51D           ; Index of the CHR pattern block being copied to the PPU (transfer loops).
IsRoomBrightening := $51E           ; 1 while a candle fire is fading a dark room to light.
CandleState := $51F                 ; Dark room lighting: 0 not lit, 1 brightening, 2 lit (Candle_UpdateRoomLight).

; ---- Credits (mode $13), cleared on room entry ----
CreditsRow := $50A                  ; Credits: tile row being drawn.
CreditsTileOffset := $50B           ; Credits: pixels scrolled since the last row was drawn; a row every 8.
CreditsVramPage := $50C             ; Credits: nametable page index 0-$B of the row being drawn.
CreditsVramLine := $50D             ; Credits: line within the page (8 lines, 6 in pages with page mod 4 = 3).
CreditsLineIndex := $50E            ; Credits: index of the next text line.

; ---- Cross-room state ($520-$52F; survives room changes) ----
PrevOpenedDoors := $521             ; CurOpenedDoors of the room being left, kept while scrolling to the next.
WhirlwindTeleportingState := $522   ; 0 none, 1 whirlwind carrying Link away, 2 dropping him off.
TeleportingLevelIndex := $523       ; Whirlwind destination: (value & 7) = level number - 1.
;   Each flute use steps it +1 when Link faces right or up, -1 when left or down, then on past
;   levels whose Triforce piece is missing (Whirlwind_Summon).
SpawnCycle := $524                  ; Next index 0-8 into the 9-entry spawn position list; kept between rooms.
CurEdgeSpawnCell := $525            ; Last screen-edge spawn cell: high nibble = row, low nibble = column.
;   Starts at $40 (Reset_ClearRam); EdgeSpawn_FindNextCell continues from it.
LevelEntranceRoomId := $526         ; Overworld room of the dungeon entrance last entered; $FF = none.
;   Mode 3 starts the overworld in this room, or at LevelInfo_StartRoomId when $FF. Not set by caves.
CellarSourceRoomId := $527          ; Dungeon room from which the current cellar was entered.
HasSkippedDemo := $528              ; $5A once Start was pressed in the title demo; stops the demo animation.
IsSwordShotForced := $529           ; Nonzero would allow the sword shot at any health. Only ever cleared.
WorldKillCycle := $52A              ; Kill cycle 0-9: column of the drop table used for the next drop.
;   +1 (wrapping 9 to 0) for each monster that finishes dying, except types $5D, $14 (child Gel)
;   and $1C (red Keese). Not reset by room changes, saving or death; only by Reset_ClearRam at reset.
RoomTileObjType := $52B             ; Object type of the room's tile object found by the layout, 0 = none.
RoomTileObjX := $52C                ; X of that tile object (column * $10).
RoomTileObjY := $52D                ; Y of that tile object (row * $10 + $40).
IsSwordBlocked := $52E              ; 1 = sword disabled until cured (set by Bubble type $2D, cleared by $2C).
;   Also cleared when a heart refill completes and when a save slot is loaded.
MazeStep := $52F                    ; Correct steps so far (0-3) in the Lost Woods ($61) or Lost Hills ($1B).

; ---- Play area attributes and level kill counts ----
PlayAreaAttrs := $530               ; [48] Nametable attribute bytes of the play area (from attrs A and B).
LevelKillCounts := $560             ; [128] per room of the current dungeon: kills banked when leaving it.
;   Subtracted from the room's monster count on entry; $F = all dead (nothing spawns).
;   Zeroed when a level is loaded (Mode2Load_Init) and when a cleared room with recurring monsters resets.

; ---- Sound driver state (bank 0) ----
CurNoteLowPeriodTrg := $5F0         ; Low period byte of the triangle note playing (base for vibrato).
SongTrgSustainFlag := $5F1          ; Song phrase: header byte 7. Bit 7 set = triangle linear counter $1F,
;   clear = $FF (Song_Drive).
SampleCounter := $5F2               ; Frames left of the DMC sample ($A0 at start).
SeaSfxCounter := $5F3               ; Sea noise effect phase counter, counts down from $D0.
NoteLengthTableBase := $5F4         ; Song phrase: base offset into the note length table.
FirstNoteIndexSongNoise := $5F5     ; Song phrase: script offset where the noise track starts (it loops).
BackgroundSample := $5F6            ; Looping DMC sample (request bits $F0), restarted when no other plays.
SongRequest := $600                 ; Song to start (one bit): $01 overworld, $02 Ganon, $04 end of level,
;   $08 item, $10 ending, $20 level 9, $40 dungeon, $80 title; the value $06 is Zelda's song. All five
;   request bytes are zeroed by Audio_Drive every frame.
SampleRequest := $601               ; DMC sample to start (bit mask; bits $10-$40 loop).
;   Sample_Play ORs into it. Bit 7 set: only bits $F0 are kept and the DMC level starts at $7F instead of 0
;   (Sample_Drive).
Tune1Request := $602                ; Square-1 tune to start (bit mask). Bit 7 set = silence the song first,
;   then play tune $80 (Link dying).
EffectRequest := $603               ; Noise effect to start (bit mask). Sfx_Play ORs into it.
Tune0Request := $604                ; Square-0 tune to start (bit mask; $40 low-hearts beep, $80 stops song).
CurTune0 := $605                    ; Square-0 tune playing (the whole Tune0Request byte), 0 = none.
CurEffect := $606                   ; Noise effect playing, 0 = none.
CurTune1 := $607                    ; Square-1 tune playing, 0 = none; the song skips square 1 meanwhile.
CurSample := $608                   ; DMC sample playing, 0 = none.
CurSong := $609                     ; Song playing (its request bit), 0 = none.
NoteOffsetSongSq1 := $60A           ; Song script offset of the next square-1 note.
NoteOffsetSongSq0 := $60B           ; Song script offset of the next square-0 note (0 = track absent).
NoteOffsetSongTrg := $60C           ; Song script offset of the next triangle note (0 = track absent).
NoteOffsetSongNoise := $60D         ; Song script offset of the next noise note.
Tune0ScriptOffset := $60E           ; Offset of the next byte of the square-0 tune in Audio_Tune0Scripts.
NoteLengthSongSq0 := $60F           ; Length in frames of the current square-0 song note.
NoteLengthSongSq1 := $610           ; Length in frames of the current square-1 song note.
NoteCounterSongSq1 := $611          ; Frames left of the current square-1 song note.
CustomEnvelopeOffsetSongSq1 := $612 ; Square-1 volume envelope position, counts down to 0.
NoteCounterSongSq0 := $613          ; Frames left of the current square-0 song note.
CustomEnvelopeOffsetSongSq0 := $614 ; Square-0 volume envelope position, counts down to 0.
NoteLengthSongTrg := $615           ; Length in frames of the current triangle song note.
NoteCounterSongTrg := $616          ; Frames left of the current triangle song note.
NoteCounterSongNoise := $617        ; Frames left of the current noise song note.
Tune1ScriptOffset := $618           ; Offset of the next byte of the square-1 tune in Audio_Tune1Scripts.
SongEnvelopeSelector := $619        ; Song phrase: volume envelope selector (byte 6 of the phrase header).
UnusedSound61A := $61A              ; Zeroed by Sample_Silence; no other code refers to it.
SongVibrationCounterSq1 := $61B     ; Frames since the square-1 song note began (vibrato phase).
SongVibrationCounterSq0 := $61C     ; Frames since the square-0 song note began (vibrato phase).
SongVibrationCounterTrg := $61D     ; Frames since the triangle song note began (vibrato phase).
SongRepetitionsTrg := $61E          ; Triangle track: repetitions left of the passage being repeated.
SongRepeatStartOffset := $61F       ; Triangle track: script offset where the repeated passage starts.

; ---- Room history and kill streak ----
CurRoomHistoryIndex := $620         ; Slot 0-5 of RoomHistory that the next new room overwrites.
RoomHistory := $621                 ; [6] Ids of the last six different rooms entered (ring buffer).
;   A room found here keeps its banked kills. A room not found here whose flags say "all dead"
;   has its kills reset (it respawns), unless its dungeon monster list is a non-recurring one.
;   Zeroed by RoomHistory_Clear (mode 2 level load, mode 3 start, and the init of the continue menu, mode 8).
WorldKillCount := $627              ; Monsters killed since Link was last harmed (zeroed by Link_BeHarmed).
;   Drop_SetUpItem runs when a death sparkle ends: every death that ends while this reads exactly $10
;   drops a fairy. The drop does not reset it. Not capped; wraps.

; ---- Save slot info (copied from the save files for the file menu) ----
QuestNumbers := $62D                ; [3] per save slot: 0 = Q1, 1 = Q2. Read in play through CurSaveSlot.
DeathCounts := $630                 ; [3] per save slot: deaths, stops at $FF.
IsSaveSlotActive := $633            ; [3] per save slot: nonzero = registered. The menu reads 5 entries.
IsRegisterSaveFileOptionEnabled := $636 ; IsSaveSlotActive[3], the menu's Register option: always 1.
IsEliminateSaveFileOptionEnabled := $637 ; IsSaveSlotActive[4], the menu's Eliminate option: always 1.
SaveSlotNames := $638               ; [24] 3 slots x 8 characters: the slot names (tile codes, $24 = space).
SaveSlotHearts := $650              ; [6] 3 slots x (HeartValues, HeartPartial) shown on the file menu.

; ---- Inventory: the items block ($657-$67E, $28 bytes, copied to and from the save file) ----
; Item slot n is byte InvItems+n. Item_IdToSlot maps an item id to its slot; Item_IdToDescriptor gives the
; class: $0x set the byte to x, $1x add x, $2x grade x (kept only if not lower than the current one).
SelectedItemSlot := $656            ; Item slot on the B button: 0 boomerangs, 1-8 = InvItems+n, $F letter.
;   Not part of the items block. Link_WieldItem_JumpTable is indexed by it; slot 3 (bow) does nothing.
InvItems := $657                    ; [40] Base of the items block, indexed by item slot. InvItems+0 = sword.
InvSword := $657                    ; Sword: 0 none, 1 wooden, 2 white, 3 magic. Same byte as InvItems.
;   Sword damage $10/$20/$40 by grade (Sword_DamagePoints).
InvBombs := $658                    ; Bombs held, 0 to InvMaxBombs. A bomb pickup adds 4.
InvArrow := $659                    ; Arrows: 0 none, 1 wooden (damage $20), 2 silver (damage $40).
InvBow := $65A                      ; Bow: 0 or 1. Without it the submenu skips the arrow slot and Arrow_Wield
;   refuses to shoot.
InvCandle := $65B                   ; Candle: 0 none, 1 blue (one fire per room), 2 red.
InvFlute := $65C                    ; Flute: 0 or 1. The code reaches this byte only by index (InvItems, Y).
InvFood := $65D                     ; Food: 0 or 1. Zeroed when the Grumble Goriya takes it.
InvPotion := $65E                   ; Potion doses 0-2. Blue potion adds 1 (max 2), red sets 2; a use takes 1.
InvRod := $65F                      ; Magic rod: 0 or 1. The code reaches this byte only by index.
InvRaft := $660                     ; Raft: 0 or 1.
InvBook := $661                     ; Magic book: 0 or 1. With it a magic shot that ends makes a fire.
InvRing := $662                     ; Ring: 0 none, 1 blue, 2 red. Damage to Link is halved once per grade.
InvLadder := $663                   ; Ladder: 0 or 1.
InvMagicKey := $664                 ; Magic key: 0 or 1. Opens locked doors without spending InvKeys.
InvBracelet := $665                 ; Power bracelet: 0 or 1. Needed to push a rock ($62), not a gravestone.
InvLetter := $666                   ; Letter: 0 none, 1 held, 2 shown to the cave person of type $74.
InvCompass := $667                  ; Compasses of levels 1-8: bit (level - 1). Reached only by index ($10).
InvMap := $668                      ; Maps of levels 1-8: bit (level - 1). Reached only by index (slot $11).
InvCompass9 := $669                 ; Compass of level 9: bit 0. Reached only by index (slot $12).
InvMap9 := $66A                     ; Map of level 9: bit 0. Reached only by index (slot $13).
InvFairy := $66B                    ; Slot $14 of the fairy item (id $23). Never written: a fairy only heals.
;   The code reaches this byte only by index (InvItems, Y).
InvClock := $66C                    ; Clock: 1 = monsters frozen, Link invincible. Zeroed by Link_ResetStateAndClock
;   (every room entry, the mode 3 start, leaving the continue menu) and when a save slot is loaded; it is
;   saved but never survives a load.
InvRupees := $66D                   ; Rupees, 0-$FF. Changes by 1 via InvRupeesToAdd / InvRupeesToSubtract.
InvKeys := $66E                     ; Keys, 0-$FF.
; Health. HeartValues and HeartPartial are item slots $18 and $19 of the items block.
;   health = (HeartValues & $0F) full hearts + HeartPartial / $100 of one more heart.
;   Heart containers = (HeartValues >> 4) + 1. At full health the low nibble equals the high nibble
;   and HeartPartial = $FF: the last heart is always the partial one. New file: $22 and $FF (3 hearts).
;   Damage is a 16-bit amount, high byte = whole hearts, low byte = 1/256 hearts (Link_BeHarmed).
;   The sword shot needs full health, which SwordShot_Make tests as: low nibble = high nibble of
;   HeartValues, and HeartPartial >= $80. From full health, damage of $7F or less (under half a
;   heart) keeps the sword shot; exactly half a heart ($80) leaves $7F and loses it.
HeartValues := $66F                 ; High nibble = heart containers - 1; low nibble = full hearts.
;   A heart container adds $11, up to $FF (16 containers). Death zeroes the low nibble.
HeartPartial := $670                ; Partial heart fill: 0 empty, $01-$7F drawn half, $80-$FF drawn full.
;   A heart pickup or a finished refill sets $FF; World_FillHearts adds 6 per step; death sets 0.
InvTriforce := $671                 ; Triforce pieces: bit (level - 1) for levels 1-8; $FF = all eight.
InvPowerTriforce := $672            ; Slot $1B, Triforce of Power: 1 once its fanfare ends. Opens the shutters
;   of a room whose secret trigger is 3 (Secret_Trigger3PowerTriforce).
InvFiveRupees := $673               ; Slot $1C of the 5-rupee item (id $0F). Never written: 5 x Item_TakeOneRupee.
;   The code reaches this byte only by index (InvItems, Y).
InvBoomerang := $674                ; Wooden boomerang: 0 or 1.
InvMagicBoomerang := $675           ; Magic boomerang: 0 or 1. Sets the throw's range limit to $FF, else $31.
InvMagicShield := $676              ; Magic shield: 0 or 1. Zeroed when a Like-Like eats it.
;   $677-$67B (slots $20-$24): no symbol and no direct reference; no item id maps to these slots
;   (Item_IdToSlot ends at $1F). They are saved with the rest of the items block.
InvMaxBombs := $67C                 ; Bomb capacity: 8 in a new file, +4 per upgrade bought (100 rupees).
InvRupeesToAdd := $67D              ; Rupees still to be credited to InvRupees, 1 every second frame.
;   Zeroed when InvRupees reaches $FF.
InvRupeesToSubtract := $67E         ; Rupees still to be taken from InvRupees, 1 every second frame.
;   Zeroed when InvRupees reaches 0. Shops refuse to sell while it is nonzero.

; ---- World flags (saved) ----
WorldFlags := $67F                  ; [384] One flag byte per room: $67F overworld, $6FF levels 1-6, $77F 7-9.
;   LevelInfo_WorldFlagsAddr points at the current level's 128 bytes (RoomFlags_Get).
;   Overworld: bits 0-2 kills banked (7 = all dead), bit 4 cave item or offer taken, bit 7 secret found,
;   bit 5 = put the stairs back on entry (Secret_RestoreStairsOW): set by RoomFlags_MarkVisited when a pushed rock or
;   gravestone has moved its full $10 pixels, and for the room that shortcut stairs lead to.
;   Dungeon: bits 0-3 one per door direction (door opened, unlocked or bombed), bit 4 room item
;   taken, bit 5 room visited (map), bits 6-7 kills banked (3 = all dead).

; ---- PPU and controller registers ----
PpuControl := $2000                 ; PPU control register (write).
PpuMask := $2001                    ; PPU mask register (write).
PpuStatus := $2002                  ; PPU status register (read; reading resets the address latch).
OamAddr := $2003                    ; OAM address (write).
OamData := $2004                    ; OAM data. Not used by the code.
PpuScroll := $2005                  ; PPU scroll (two writes: X, Y).
PpuAddr := $2006                    ; PPU VRAM address (two writes: high, low).
PpuData := $2007                    ; PPU VRAM data.
SpriteDma := $4014                  ; OAM DMA: page number of the sprite buffer to copy.
Ctrl1 := $4016                      ; Controller strobe (write) and controller 1 data (read).
Ctrl2FrameCtr := $4017              ; Controller 2 data (read); APU frame counter (write).

; ---- APU registers ----
Sq0Duty := $4000                    ; Square 0: duty, envelope, volume.
Sq0Sweep := $4001                   ; Square 0: sweep.
Sq0Timer := $4002                   ; Square 0: period low byte.
Sq0Length := $4003                  ; Square 0: length counter, period high bits.
Sq1Duty := $4004                    ; Square 1: duty, envelope, volume.
Sq1Sweep := $4005                   ; Square 1: sweep.
Sq1Timer := $4006                   ; Square 1: period low byte.
Sq1Length := $4007                  ; Square 1: length counter, period high bits.
TrgLinear := $4008                  ; Triangle: linear counter.
TrgTimer := $400A                   ; Triangle: period low byte.
TrgLength := $400B                  ; Triangle: length counter, period high bits.
NoiseVolume := $400C                ; Noise: envelope, volume.
NoisePeriod := $400E                ; Noise: mode and period.
NoiseLength := $400F                ; Noise: length counter.
DmcFreq := $4010                    ; DMC: flags and rate.
DmcCounter := $4011                 ; DMC: output level.
DmcAddress := $4012                 ; DMC: sample address.
DmcLength := $4013                  ; DMC: sample length.
ApuStatus := $4015                  ; APU channel enables.

; ---- Save RAM: markers and save files ($6000-$652F) ----
; File A is the committed copy of each save slot. Saving (mode D) first builds file B in the RAM that
; otherwise holds the level block (name $6880, items $6898, flags $6910 for slot 0: Save_FileBAddressSets),
; then copies it over file A; IsSaveFileBCommitted tells an interrupted copy from a finished one.
SaveRamBegin := $6001               ; $5A once save RAM is initialized (with SaveRamEnd = $A5).
;   Without both markers Save_InitSaveRam zeroes $6530-$7FFF and marks the three file B copies committed.
SaveFileAName0 := $6002             ; [8] File A, slot 0: name. (The SaveFileA* blocks are addressed through
;   the Save_FileAAddressSets pointer table, never by these names.)
SaveFileAName1 := $600A             ; [8] File A, slot 1: name.
SaveFileAName2 := $6012             ; [8] File A, slot 2: name.
SaveFileAItems0 := $601A            ; [40] File A, slot 0: copy of the items block InvItems ($657-$67E).
SaveFileAItems1 := $6042            ; [40] File A, slot 1: items block.
SaveFileAItems2 := $606A            ; [40] File A, slot 2: items block.
SaveFileAWorldFlags0 := $6092       ; [384] File A, slot 0: copy of WorldFlags ($67F-$7FE).
SaveFileAWorldFlags1 := $6212       ; [384] File A, slot 1: world flags.
SaveFileAWorldFlags2 := $6392       ; [384] File A, slot 2: world flags.
SaveFileASlotActive0 := $6512       ; File A, slot 0: nonzero = slot registered (source of IsSaveSlotActive).
SaveFileASlotActive1 := $6513       ; File A, slot 1: slot registered.
SaveFileASlotActive2 := $6514       ; File A, slot 2: slot registered.
SaveFileAUnused0 := $6515           ; File A, slot 0: a byte with no use (only zeroed, summed and copied).
;   Misnamed: the pointer sets (Save_FileAAddressSets) put the death count three bytes further on.
SaveFileAUnused1 := $6516           ; File A, slot 1: byte with no use.
SaveFileAUnused2 := $6517           ; File A, slot 2: byte with no use.
SaveFileADeaths0 := $6518           ; File A, slot 0: death count (source of DeathCounts).
SaveFileADeaths1 := $6519           ; File A, slot 1: death count.
SaveFileADeaths2 := $651A           ; File A, slot 2: death count.
SaveFileAQuestNumber0 := $651B      ; File A, slot 0: quest, 0 = Q1, 1 = Q2 (source of QuestNumbers).
SaveFileAQuestNumber1 := $651C      ; File A, slot 1: quest.
SaveFileAQuestNumber2 := $651D      ; File A, slot 2: quest.
SaveFileOpenMarkers := $651E        ; [3] per slot: $5A while file A is whole; 0 during the B-to-A copy.
SaveFileCloseMarkers := $6521       ; [3] per slot: $A5 while file A is whole; 0 during the B-to-A copy.
SaveFileAChecksums := $6524         ; [3] x 2 bytes: file A checksum per slot; a mismatch reformats the slot.
IsSaveFileBCommitted := $652A       ; [3] per slot: $FF = file B fully copied to file A; 0 = copy pending.

; ---- Save RAM: play area, level block and level info (working copies, not saved game data) ----
PlayAreaTiles := $6530              ; [704] Play area tile map: 32 columns x 22 rows, stored by column.
;   Addressed through Tile_PlayAreaColumnAddrs; ends at $67EF, where the BANK_06_DATA copy begins.
; Level block: six 128-byte arrays indexed by RoomId, copied from ROM when a level is loaded (mode 2).
; The overworld, levels 1-6 and levels 7-9 each have their own block; Q2 patches some bytes.
LevelBlockAttrsA := $687E           ; [128] Dungeon: bits 7-5 N door type, 4-2 S door type, 1-0 outer palette.
;   Overworld: bits 7-4 X where Link comes out of a cave, bit 3 Zora, bit 2 sea sound, 1-0 outer
;   palette. Cellar rooms: whole byte = room the left stairs lead to.
LevelBlockAttrsB := $68FE           ; [128] Dungeon: bits 7-5 W door type, 4-2 E door type, 1-0 inner palette.
;   Overworld: bits 7-2 cave index (1-9 = level entrance; $10 up = cave person type $6A + n - $10;
;   $14 = shortcut cave), 1-0 inner palette. Cellar rooms: room the right stairs lead to.
LevelBlockAttrsC := $697E           ; [128] Bits 7-6 index into LevelInfo_FoeCounts, 5-0 monster list id.
;   Cellar rooms: bits 7-4 X and bits 3-0 row (Y = row * $10 + $D) where Link comes out upstairs.
LevelBlockAttrsD := $69FE           ; [128] Bit 7 = bit 6 of the monster list id. Dungeon: bit 6 push block,
;   bits 5-0 unique room id (room layout; Room_GetUniqueRoomId masks $3F). Overworld: bits 6-0 layout id (the
;   overworld has layout ids above $3F).
LevelBlockAttrsE := $6A7E           ; [128] Dungeon: bit 7 dark room, 6-5 boss roar index, 4-0 room item id
;   (3 = none). Overworld: not per room: cave wares, 3 item ids per cave at +0 (bits 7-6 = cave
;   flags) and their 3 prices at +60.
LevelBlockAttrsF := $6AFE           ; [128] Bits 5-4 index into LevelInfo_ShortcutOrItemPosArray.
;   Dungeon: bits 2-0 secret trigger (3 = shutters open once the Triforce of Power is taken, 7 = kill all
;   for the item; see Secret_CheckUnderworld). Overworld: bits 7-6 quest of the secret (1 = Q1 only,
;   2 = Q2 only),
;   bit 3 monsters enter from the screen edges, bits 2-0 row where Link comes out of a cave.
; Level info: 252 bytes copied from ROM with the level block.
LevelInfo_PalettesTransferBuf := $6B7E ; [36] PPU transfer record with the level's palettes.
LevelInfo_FoeCounts := $6BA2        ; [4] Monster counts selectable by attr C bits 7-6 (dungeons: 3, 5, 6, 8).
LevelInfo_StartY := $6BA6           ; Link's Y when the level is entered ($DD in dungeons, $8D overworld).
LevelInfo_ShortcutOrItemPosArray := $6BA7 ; [4] Positions: high nibble X, low nibble Y (each * $10).
;   Room item spots in dungeons, shortcut stairs spots in the overworld; chosen by attr F bits 5-4.
LevelInfo_SubmenuMapRotation := $6BAB ; Columns the submenu map is rotated to match the level's shape.
LevelInfo_StatusBarMapXOffset := $6BAC ; Pixels added to the status bar position marker's X.
LevelInfo_StartRoomId := $6BAD      ; Room where Link starts the level.
LevelInfo_TriforceRoomId := $6BAE   ; Room marked by the compass (Triforce marker on the status bar map).
LevelInfo_WorldFlagsAddr := $6BAF   ; 2 bytes lo/hi: address of this level's 128 bytes in WorldFlags.
LevelInfo_LevelNumber := $6BB1      ; Level number 1-9 for the "LEVEL-n" text; 0 = overworld.
LevelInfo_CellarRoomIdArray := $6BB2 ; [10] Dungeon: room ids of the cellars, $FF-padded.
;   Overworld: the first 4 are the rooms linked by the shortcut stairs.
LevelInfo_BossRoomId := $6BBC       ; Boss room; while its kills flag is not "all dead" rooms play the roar.
LevelInfo_SubmenuMapMask := $6BBD   ; [16] per map column: bit set = a room exists in that row.
LevelInfo_StatusBarMapTransferBuf := $6BCD ; [45] PPU transfer record of the status bar map.
LevelInfo_PaletteCycles := $6BFA    ; [96] Fade series: 8-byte palette halves, indexed through FadeCycle.
LevelInfo_DeathPaletteSeries := $6C5A ; [32] Fade series $60 (death): 4 steps of 8 bytes.
SaveRamEnd := $7FFF                 ; $A5 once save RAM is initialized; see SaveRamBegin.
