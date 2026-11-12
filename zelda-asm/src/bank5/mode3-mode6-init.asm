; ============================================================================
; FILE:     bank5/mode3-mode6-init.asm   BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 6373-6586
; SUBJECTS: Door, Mode3Unfurl, Mode6Leave, Obj, Room, Transfer
; TOPIC:    Mode 3 (unfurl) init submodes 2-8: attributes, status bar, "LEVEL-n" text, first room
;           layout, Link's start position. The room bounds for objects (overworld and dungeon).
;           Mode 6 (leave room) init: how far Link walks through a door. The weapon slot reset.
; SEE ALSO: bank7/init-mode-dispatch.asm (Mode3Unfurl_Init jump table and its submodes 0 and 1; the mode
;           table that reaches Mode6Leave_Init; Link_ResetStateAndClock); bank7/update-mode-dispatch.asm
;           (Mode3Unfurl_Update; Mode4Enter_UpdateEnterOrLeave, the walk that Mode6Leave_Init sets up);
;           bank1/room-bounds.asm (the tests that read the room bounds); bank5/enter-room.asm
;           (Room_EnterCommon, the entering counterpart; calls Door_GetPassedType and
;           Room_SetUpObjBounds); bank5/fill-walls.asm (Door_FindTypeByDirBit); bank5/touch-door.asm
;           (door types); bank5/mode11-death.asm and bank5/mode7-scroll-update.asm (the routines
;           around Transfer_CopyPlayAreaAttrsHalf); bank1/map-marker-curtain.asm (curtain).
; KEY FACTS:
;   - Room bounds for objects, overworld: left $11, right $E0, up $4E, down $CD, first unwalkable
;     tile $89. Dungeon: $21, $D0, $5E, $BD, $78. Room_ObjBoundsOW, Room_ObjBoundsUW
;   - Room_SetUpObjBounds runs on every room entry, so it is what puts RoomFirstUnwalkableTile back
;     after something raised it during a room visit.
;   - Every room change clears the state of weapon slots $0D-$12 and the ladder slot: a bomb, fire,
;     boomerang, food, arrow, shot or sword that is out is gone. Obj_ResetWeaponSlotsAndLadder
;   - Leaving a room also zeroes Link's state and InvClock (Link_ResetStateAndClock) and banks the room's
;     kill count (RoomHistory_SaveKillCount). Mode6Leave_Init
;   - Dungeon doors of type 2, 3 (false walls) and 4 (bombable wall): on leaving, Link's grid
;     offset is set to +$28 or -$28 and he walks until it is +8 or -8, $20 pixels, before the
;     scroll. Other door types and the overworld: offset 0, no walk. Mode6Leave_Init,
;     Mode6Leave_LinkGridOffsets
;   - Leaving through a door of type 2 requests the "secret found" tune (Tune1Request $04); type 3,
;     the other false wall, does not. Mode6Leave_Init
;   - A level starts with Link at X = $78, Y = LevelInfo_StartY, facing up. Mode3Unfurl_Init_Sub8_LayOutRoomPlaceLink
;   - Door_GetPassedType looks at the door ahead of Link in mode 6 and at the door behind him in
;     any other mode (mode 4, entering).
; CONTENTS: (generated; L = line in this file)
;   L0058 Mode3Unfurl_Init_Sub2_FillAttrsCuePalettes     code  Mode 3 init, submode 2: builds the first room's attribute bytes and cues the level palettes.
;   L0069 Mode3Unfurl_Init_Sub3_TransferTopHalfAttrs     code  Submode step: cues the top half of PlayAreaAttrs (24 bytes) for PPU $23D0; next submode.
;   L0079 Mode3Unfurl_Init_Sub4_TransferBottomHalfAttrs  code  Submode step: cues the bottom half of PlayAreaAttrs (24 bytes) for PPU $23E8; next submode.
;   L0087 Mode3Unfurl_Init_Sub5_CueStatusBar             code  Mode 3 init, submode 5: cues the fixed tiles of the status bar; next submode.
;   L0091 Mode3Unfurl_Init_SelectTransferBuf             code  Stores static transfer buffer selector A in TileBufSelector and goes to the next submode.
;   L0095 Mode3Unfurl_Init_AdvanceSubmode                code  Shared exit of the mode 3 init submodes in this file: next submode.
;   L0102 Mode3Unfurl_Init_Sub6_CueStatusBarMap          code  Mode 3 init, submode 6: cues the status bar map (in a dungeon only if Link has its map).
;   L0114 Mode3Unfurl_Init_Sub7_CueLevelNumber           code  Mode 3 init, submode 7: in a dungeon, cues the "LEVEL-n" text of the status bar; next submode.
;   L0129 Mode3Unfurl_Init_Sub8_LayOutRoomPlaceLink      code  Mode 3 init, submode 8: lays out the first room, arms the curtain, places Link, starts the update.
;   L0159 Room_ObjBoundsOW                               data  TABLE: overworld room bounds for objects; index = left, right, up, down, first unwalkable tile.
;   L0164 Room_ObjBoundsUW                               data  TABLE: dungeon room bounds for objects; same five entries as Room_ObjBoundsOW.
;   L0175 Room_SetUpObjBounds                            code  Copies the five room bound values of the overworld or of dungeons to RoomBoundLeft onward.
;   L0197 Mode6Leave_LinkGridOffsets                     data  TABLE: Link's grid offset at the start of leaving a dungeon room; index = 0 facing left or up,
;   L0214 Mode6Leave_Init                                code  Mode 6 (leave room) init: resets Link, banks the kill count, sets how far he walks to the edge.
;   L0263 Obj_ResetWeaponSlotsAndLadder                  code  Clears the ladder slot and the state of Link's six weapon slots ($0D-$12).
;   L0281 Door_GetPassedType                             code  Gets the type of the dungeon door Link passes: ahead of him in mode 6, behind him otherwise.
;   L0307 Transfer_CopyPlayAreaAttrsHalf                 code  Writes a transfer record of 24 play area attribute bytes at the start of DynTileBuf.
; ============================================================================

; ----------------------------------------------------------------------------
; Mode 3 init, submode 2: builds the first room's attribute bytes and cues the level palettes.
; OUT:  PlayAreaAttrs filled for RoomId; TileBufSelector = $18; GameSubmode incremented.
; ----------------------------------------------------------------------------
Mode3Unfurl_Init_Sub2_FillAttrsCuePalettes:
    LDA RoomId
    JSR Room_FillPlayAreaAttrs
    LDA #$18                    ; Selector $18 = LevelInfo_PalettesTransferBuf (the level's palettes).
    BNE Mode3Unfurl_Init_SelectTransferBuf       ; Always taken: cue it and go to the next submode.

; ----------------------------------------------------------------------------
; Submode step: cues the top half of PlayAreaAttrs (24 bytes) for PPU $23D0; next submode.
; NOTE: Mode 3 init submode 3. The mode $A and $B/$C attribute steps jump here too
;       (bank5/cave-cellar-enter.asm).
; ----------------------------------------------------------------------------
Mode3Unfurl_Init_Sub3_TransferTopHalfAttrs:
    LDA #$D0                    ; Low byte of PPU address $23D0 (name table 0 attributes, top half).
    LDY #$17                    ; Last PlayAreaAttrs offset of the top half ($00-$17).
:
    JMP Transfer_CuePlayAreaAttrsHalfNT0AndAdvanceSubmode

; ----------------------------------------------------------------------------
; Submode step: cues the bottom half of PlayAreaAttrs (24 bytes) for PPU $23E8; next submode.
; NOTE: Mode 3 init submode 4. Also called by Mode_InitSubroom_TransferBottomHalfAttrs.
; ----------------------------------------------------------------------------
Mode3Unfurl_Init_Sub4_TransferBottomHalfAttrs:
    LDA #$E8                    ; Low byte of PPU address $23E8 (bottom half).
    LDY #$2F                    ; Last PlayAreaAttrs offset of the bottom half ($18-$2F).
    BNE :-                      ; Always taken: to the JMP in Mode3Unfurl_Init_Sub3_TransferTopHalfAttrs.

; ----------------------------------------------------------------------------
; Mode 3 init, submode 5: cues the fixed tiles of the status bar; next submode.
; ----------------------------------------------------------------------------
Mode3Unfurl_Init_Sub5_CueStatusBar:
    LDA #$0E                    ; Selector $0E = Transfer_StatusBarStaticsBuf.

; Stores static transfer buffer selector A in TileBufSelector and goes to the next submode.
Mode3Unfurl_Init_SelectTransferBuf:
    STA TileBufSelector

; Shared exit of the mode 3 init submodes in this file: next submode.
Mode3Unfurl_Init_AdvanceSubmode:
    INC GameSubmode
    RTS

; ----------------------------------------------------------------------------
; Mode 3 init, submode 6: cues the status bar map (in a dungeon only if Link has its map).
; ----------------------------------------------------------------------------
Mode3Unfurl_Init_Sub6_CueStatusBarMap:
    LDA CurLevel
    BEQ :+                      ; Overworld: always draw the map.
    JSR Inv_HasMap
    BEQ Mode3Unfurl_Init_AdvanceSubmode             ; Dungeon whose map Link does not have: just go to the next submode.
:
    LDA #$44                    ; Selector $44 = LevelInfo_StatusBarMapTransferBuf.
    BNE Mode3Unfurl_Init_SelectTransferBuf       ; Always taken.

; ----------------------------------------------------------------------------
; Mode 3 init, submode 7: in a dungeon, cues the "LEVEL-n" text of the status bar; next submode.
; ----------------------------------------------------------------------------
Mode3Unfurl_Init_Sub7_CueLevelNumber:
    LDA LevelInfo_LevelNumber
    BEQ Mode3Unfurl_Init_AdvanceSubmode             ; 0 = overworld: no text. Next submode.
    STA Transfer_LevelNumberBuf+9    ; The last character of the record's "LEVEL-n" is the number.
    LDA #$0C                    ; Selector $0C = Transfer_LevelNumberBuf.
    BNE Mode3Unfurl_Init_SelectTransferBuf       ; Always taken.

; ----------------------------------------------------------------------------
; Mode 3 init, submode 8: lays out the first room, arms the curtain, places Link, starts the update.
; OUT:  ObjX+12 = $10 and ObjX+13 = $11 (curtain columns); IsVideoHeldOff = 0; Link at X = $78,
;       Y = LevelInfo_StartY, facing up; GameSubmode = 0 and IsUpdatingMode set (Mode_BeginUpdate).
; NOTE: Mode3Unfurl_Update then opens the curtain from the middle outward: each step of
;       Mode_UpdateCurtainEffect copies the two columns, decrements ObjX+12 and increments ObjX+13.
;       Mode 3 ends when ObjX+12 reaches 0.
; ----------------------------------------------------------------------------
Mode3Unfurl_Init_Sub8_LayOutRoomPlaceLink:
    JSR RoomLayout_LayOutRoom

    ; Curtain columns for Mode_UpdateCurtainEffect, kept in the X bytes of object slots $0C and
    ; $0D. Each is a tile column number + 1: ObjX+12 starts at $10 and goes down, ObjX+13 starts
    ; at $11 and goes up.
    LDY #$10
    STY ObjX+12
    INY
    STY ObjX+13

    ; IsVideoHeldOff = 0. Mode3Unfurl_Init_Sub0 set it to 1; while it is nonzero (and the sprite-0 check is off)
    ; Nmi_Isr leaves the PPU mask alone, so the video stays off. From now on Nmi_Isr turns the video
    ; on in any frame without a static transfer (TileBufSelector = 0).
    LDA #$00
    STA IsVideoHeldOff
    LDA #$08                    ; Link faces up.
    STA ObjDir
    LDA #$78                    ; Link's X: the middle of the room.
    STA ObjX
    LDA LevelInfo_StartY        ; Link's Y: the level's start Y from the level info.
    STA ObjY
    JMP Mode_BeginUpdate

; TABLE: overworld room bounds for objects; index = left, right, up, down, first unwalkable tile.
; Values are pixels (X for left and right, Y for up and down) and, last, a tile number.
; Room_SetUpObjBounds copies the five bytes to RoomBoundLeft, RoomBoundRight, RoomBoundUp,
; RoomBoundDown and RoomFirstUnwalkableTile. The bound tests (bank1/room-bounds.asm) refuse a move
; left when X < left, right when X >= right, up when Y < up, down when Y >= down; a tile numbered
; at or above the last value blocks movement.
Room_ObjBoundsOW:
    .BYTE $11, $E0, $4E, $CD, $89

; TABLE: dungeon room bounds for objects; same five entries as Room_ObjBoundsOW.
; Read by Room_SetUpObjBounds as Room_ObjBoundsOW + 5.
Room_ObjBoundsUW:
    .BYTE $21, $D0, $5E, $BD, $78

; ----------------------------------------------------------------------------
; Copies the five room bound values of the overworld or of dungeons to RoomBoundLeft onward.
; OUT:  RoomBoundLeft, RoomBoundRight, RoomBoundUp, RoomBoundDown, RoomFirstUnwalkableTile set;
;       in the overworld also DoorwayDir = 0. X = 5; Y = 5 or $0A.
; NOTE: Called on every room entry (Room_EnterCommon) and when play starts in modes 5, $B and
;       $C (bank7/init-mode5-play.asm). That resets RoomFirstUnwalkableTile after a room in which
;       it was raised (the flute pond sets $99, bank7/flute-secret-pond.asm).
; ----------------------------------------------------------------------------
Room_SetUpObjBounds:
    LDY #$05                    ; Offset of the dungeon values from Room_ObjBoundsOW.
    LDA CurLevel
    BNE :+                      ; In a dungeon: copy from offset 5.

    ; Overworld: offset 0. There are no doorways in the overworld, so clear DoorwayDir too.
    LDY #$00
    STY DoorwayDir
:
    LDX #$00
:
    LDA Room_ObjBoundsOW, Y
    STA RoomBoundLeft, X
    INY
    INX
    CPX #$05
    BNE :-
    RTS

; TABLE: Link's grid offset at the start of leaving a dungeon room; index = 0 facing left or up,
; 1 facing right or down (both when the door is a false wall or a bombable wall), 2 otherwise;
; value = signed pixels. Mode4Enter_UpdateEnterOrLeave walks Link until the offset is 0, +8 or -8.
Mode6Leave_LinkGridOffsets:
    .BYTE $28, $D8, $00         ; +$28, -$28, 0

; ----------------------------------------------------------------------------
; Mode 6 (leave room) init: resets Link, banks the kill count, sets how far he walks to the edge.
; OUT:  ObjGridOffset (slot 0) = 0, $28 or $D8; IsUpdatingMode set (Mode_BeginUpdate); then falls
;       into Obj_ResetWeaponSlotsAndLadder.
; NOTE: Overworld: offset 0, so Mode4Enter_UpdateEnterOrLeave goes on to mode 7 (scroll) at once.
; NOTE: Dungeon: the type of the door in Link's direction decides. Types 2, 3 (false walls) and 4
;       (bombable wall): offset +$28 when Link faces left or up, -$28 when right or down; he then
;       walks until the offset is +8 or -8, that is $20 pixels at his q-speed (normally $60 = 1.5
;       pixels per frame: about 22 frames), and only then the scroll starts. Types 0 (open), 1
;       (wall), 5, 6 (key doors) and 7 (shutter): offset 0, no walk.
; NOTE: Door type 2 requests the "secret found" tune (Tune1Request $04); type 3 does not.
; NOTE: Link_ResetStateAndClock zeroes InvClock as well as Link's state: the clock stops working when
;       Link leaves the room where he took it.
; ----------------------------------------------------------------------------
Mode6Leave_Init:
    JSR Link_ResetStateAndClock        ; Link's ObjState = 0; InvClock = 0.
    JSR Sprite_DrawBetweenRooms
    LDA CurLevel
    BEQ :+
    JSR Sprite_WriteBlankPrioritySprites    ; Dungeon only: blank the first $10 sprite records.
:
    JSR Link_EndMoveAndAnimateBetweenRooms    ; Draws Link in the overworld; returns at once in a dungeon.
    JSR RoomHistory_SaveKillCount           ; Bank the room's kills (RoomHistory_SaveKillCountOW or RoomHistory_SaveKillCountUW).
    LDA CurLevel
    BEQ @SetWalkDistance0       ; Overworld: no walk.
    JSR Door_GetPassedType       ; A = type of the door ahead; Y = 1 if Link faces right or down, else 0.

    ; Door type 2 (false wall): request the "secret found" tune.
    PHA
    AND #$07
    CMP #$02
    BNE :+
    LDA #$04
    STA Tune1Request
:
    PLA

    ; Types 2, 3 and 4 (false walls, bombable wall): Link has to cross the wall. Keep Y (0 or 1)
    ; to pick +$28 (moving left or up, the offset counts down) or -$28 (right or down, it counts
    ; up). Types 0, 1 and 5-7: index 2, offset 0, no walk.
    AND #$07
    CMP #$02
    BCC @SetWalkDistance0
    CMP #$05
    BCC :+

@SetWalkDistance0:
    LDY #$02
:
    LDA Mode6Leave_LinkGridOffsets, Y
    STA ObjGridOffset

    ; In mode 6 this call runs Room_SetUpItemAndPushBlock, adds the room to RoomHistory if it is not there,
    ; and ends in Mode_BeginUpdate. It returns here and execution falls into Obj_ResetWeaponSlotsAndLadder.
    JSR Room_RunEntryTasks

; ----------------------------------------------------------------------------
; Clears the ladder slot and the state of Link's six weapon slots ($0D-$12).
; OUT:  LadderSlot = 0; ObjState of object slots $0D-$12 = 0; A = 0; Y = $FF.
; NOTE: Runs when Link leaves a room (falls in from Mode6Leave_Init), enters a room, cave or cellar
;       (Room_EnterCommon and its callers) and at the end of mode $A. A sword, shot, boomerang,
;       food, bomb, fire, rod or arrow that is out does not survive a room change.
; ----------------------------------------------------------------------------
Obj_ResetWeaponSlotsAndLadder:
    LDA #$00
    STA LadderSlot              ; No ladder is laid.
    LDY #$05                    ; Slots $0D + 5 down to $0D + 0.
:
    STA a:ObjState+13, Y
    DEY
    BPL :-
    RTS

; ----------------------------------------------------------------------------
; Gets the type of the dungeon door Link passes: ahead of him in mode 6, behind him otherwise.
; IN:   ObjDir (slot 0); GameMode; RoomId
; OUT:  A = door type (0 open, 1 wall, 2 and 3 false wall, 4 bombable, 5 and 6 key, 7 shutter);
;       Y = [0F] = 1 if Link faces right or down, else 0. Uses [00]-[03].
; NOTE: In mode 4 (entering) Link already faces into the new room, so the door he came through
;       is on the opposite side: the direction is reversed before the lookup.
; ----------------------------------------------------------------------------
Door_GetPassedType:
    LDY #$00                    ; [0F] = 1 if ObjDir has the right (1) or down (4) bit, else 0.
    LDA ObjDir
    AND #$05
    BEQ :+
    INY
:
    STY $0F
    LDA ObjDir
    LDY GameMode
    CPY #$06
    BEQ :+                      ; Mode 6: the door in Link's direction.
    JSR Dir_GetOpposite          ; Any other mode: the door in the opposite direction.
:
    STA $02                     ; [02] = direction bit of the door to look up.
    JSR Door_FindTypeByDirBit
    LDY $0F
    RTS

; ----------------------------------------------------------------------------
; Writes a transfer record of 24 play area attribute bytes at the start of DynTileBuf.
; IN:   X = PPU address high byte; A = PPU address low byte; Y = offset in PlayAreaAttrs of the
;       last byte to copy ($17 for the top half, $2F for the bottom half)
; OUT:  DynTileBuf = address, count $18, PlayAreaAttrs[Y-$17 .. Y], end marker $FF. X = 0.
; NOTE: Always writes at offset 0 of DynTileBuf and does not update DynTileBufLen.
; ----------------------------------------------------------------------------
Transfer_CopyPlayAreaAttrsHalf:
    STX DynTileBuf
    STA DynTileBuf+1
    LDX #$18
    STX DynTileBuf+2            ; Count: $18 bytes, written left to right.
    LDA #$FF
    STA DynTileBuf+3, X         ; End marker after the 24 data bytes.
:
    LDA PlayAreaAttrs, Y        ; Copy from the last byte back to the first.
    STA DynTileBuf+2, X
    DEY
    DEX
    BNE :-
    RTS
