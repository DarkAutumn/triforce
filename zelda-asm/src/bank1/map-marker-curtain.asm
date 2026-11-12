; ============================================================================
; FILE:     bank1/map-marker-curtain.asm BANK: 1 (RAM code: stored at $A500, run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 4134-4320
; SUBJECTS: Math, Mode, StatusBar
; TOPIC:    The position marker sprites on the status bar map (Link's room and the compass's Triforce
;           room), the curtain effect that redraws the play area two tile columns at a time, and small
;           16-bit add/subtract helpers on zero-page pairs [00:01], [02:03], [04:05].
; SEE ALSO: bank7/next-room-room-flags.asm (StatusBar_UpdateTriforceMarker calls StatusBar_UpdatePositionMarker with
;           X = 4; Util_BitMasks); bank5/mode12-lay-out-room.asm (mode $12 sets the curtain columns and calls
;           Mode_UpdateCurtainEffect; Transfer_CopyColumnToTileBuf); bank2/mode13-ending-init.asm (calls the
;           _Bank2 variant); bank1/whirlwind.asm and bank7/init-mode-dispatch.asm (callers of
;           StatusBar_UpdateLinkPositionMarker).
; KEY FACTS:
;   - Link's map marker is not updated in mode 9 (cellar) or while a whirlwind carries Link
;     (WhirlwindTeleportingState <> 0). (StatusBar_UpdateLinkPositionMarker)
;   - Marker position: Y = $17 + 4 * room row. X = column * 4 + $11 in the overworld, column * 8 + $12
;     in a dungeon, plus LevelInfo_StatusBarMapXOffset. Room row = bits 6-4 of the room id, column =
;     bits 3-0. (StatusBar_UpdatePositionMarker)
;   - Link's marker is sprite record $15 (Sprites+84) with attribute 0; the Triforce marker is record
;     $16 (Sprites+88); both use tile $3E. (StatusBar_UpdatePositionMarker)
;   - The Triforce marker alternates between attributes 2 and 3 every 16 frames while the level's
;     Triforce piece is missing from InvTriforce, and always in level 9; with the piece it stays at 3.
;     (StatusBar_UpdatePositionMarker)
;   - The curtain effect copies two tile columns per step, ObjX[$0D] counting up and ObjX[$0C] counting
;     down, then stores 5 in ObjTimer[0] and waits for it to reach 0. (Mode_UpdateCurtainEffect)
;   - Mode 3 opens the curtain from the middle (columns $10 and $11, until ObjX[$0C] = 0); modes $12
;     and $13 close it from the edges ($20 and 1, until ObjX[$0C] < $11). 16 steps either way.
;     (Mode3Unfurl_Init_Sub8_LayOutRoomPlaceLink, Mode3Unfurl_Update, Mode12EndLevel_Init, Mode12EndLevel_Update_Sub3CloseCurtain, Zelda_State1WaitForEnding,
;     Mode13WinGame_UpdateCurtain)
;   - Mode_UpdateCurtainEffect leaves bank 5 switched in; the _Bank2 variant switches bank 2 back in.
;     (Mode_UpdateCurtainEffect, Mode_UpdateCurtainEffect_ReturnToBank2)
; CONTENTS: (generated; L = line in this file)
;   L0053 StatusBar_UpdateLinkPositionMarker      code  Places Link's position marker on the status bar map at the current room.
;   L0072 StatusBar_UpdatePositionMarker          code  Writes one status bar map marker sprite for a room: Link's (X = 0) or the Triforce room's (X = 4).
;   L0126 StatusBar_UpdatePositionMarker_Return   code  Shared RTS of the two routines above.
;   L0132 Mode_UpdateCurtainEffect_ReturnToBank2  code  Runs one curtain-effect step for a caller in bank 2, then switches bank 2 back in.
;   L0150 Mode_UpdateCurtainEffect                code  Curtain effect step: copies two tile columns of the play area to the transfer buffer, then waits.
;   L0181 Math_Add1ToInt16At0                     code  Adds 1 to the 16-bit value at [00:01].
;   L0189 Math_AddToInt16At0                      code  Adds A (unsigned) to the 16-bit value at [00:01].
;   L0202 Math_Add1ToInt16At2                     code  Adds 1 to the 16-bit value at [02:03].
;   L0210 Math_AddToInt16At2                      code  Adds A (unsigned) to the 16-bit value at [02:03].
;   L0223 Math_Add1ToInt16At4                     code  Adds 1 to the 16-bit value at [04:05].
;   L0231 Math_AddToInt16At4                      code  Adds A (unsigned) to the 16-bit value at [04:05].
;   L0244 Math_Sub1FromInt16At4                   code  Subtracts 1 from the 16-bit value at [04:05].
; ============================================================================

; ----------------------------------------------------------------------------
; Places Link's position marker on the status bar map at the current room.
; OUT:  sprite record at Sprites+84 written (falls into StatusBar_UpdatePositionMarker with A = RoomId, X = 0).
; NOTE: Does nothing in mode 9 (cellar) or while WhirlwindTeleportingState <> 0, so the marker keeps
;       its last position then.
; ----------------------------------------------------------------------------
StatusBar_UpdateLinkPositionMarker:
    LDA GameMode                ; Return if mode 9 (cellar) or whirlwind teleporting.
    CMP #$09
    BEQ StatusBar_UpdatePositionMarker_Return
    LDX WhirlwindTeleportingState
    BNE StatusBar_UpdatePositionMarker_Return
    LDA RoomId
    LDX #$00

; ----------------------------------------------------------------------------
; Writes one status bar map marker sprite for a room: Link's (X = 0) or the Triforce room's (X = 4).
; IN:   A = room id (bits 6-4 row, bits 3-0 column); X = 0 for Link's marker (sprite record at
;       Sprites+84), 4 for the Triforce marker (record at Sprites+88).
; OUT:  the record's Y, tile ($3E), attributes and X written; [00] = map left edge ($11 or $12).
; NOTE: Y = $17 + row * 4. X = column * 4 + $11 (overworld, CurLevel = 0) or column * 8 + $12
;       (dungeon), plus LevelInfo_StatusBarMapXOffset.
; NOTE: Attributes: Link's marker 0. Triforce marker 3; but while the level's bit is clear in
;       InvTriforce, or always in level 9, it is 2 during the first 16 of every 32 frames (blinks).
; ----------------------------------------------------------------------------
StatusBar_UpdatePositionMarker:
    PHA                         ; Push the room ID.
    AND #$70                    ; Isolate the row of the room ID (row * $10).
    LSR                         ; Row * 4: the rows in the status bar map are 4 pixels tall.
    LSR
    ADC #$17                    ; The map's top row is at Y = $17. (Carry is clear after the shifts.)
    STA Sprites+84, X           ; Place the position marker vertically.
    LDA #$11                    ; Map X at $11 in OW.
    LDY CurLevel
    BEQ :+
    LDA #$12                    ; Map X at $12 in UW.
:
    STA $00                     ; Store the appropriate status bar map left coordinate in [00].
    PLA                         ; Pop the room ID.
    AND #$0F                    ; Isolate the column of the room ID.
    CPY #$00
    BEQ :+
    ASL                         ; In UW, columns in status bar map are 8 pixels wide.
:
    ASL                         ; In OW, they are 4 pixels wide.
    ASL
    ADC $00                     ; Add the status bar map left coordinate.
    ADC LevelInfo_StatusBarMapXOffset    ; Add a horizontal offset from level info to center the map.
    STA Sprites+87, X           ; Place the position marker horizontally.
    LDA #$3E
    STA Sprites+85, X           ; Use the position marker tile.
    LDA #$00
    CPX #$00
    BEQ @SetAttr                ; Link's marker (X = 0) uses attribute 0.
    LDA #$03                    ; Triforce marker: attribute 3 by default.
    PHA
    LDY CurLevel
    CPY #$09
    BEQ :+                      ; In level 9, always blink.
    LDA InvTriforce
    AND Util_BitMasks-1, Y         ; Bit (level - 1) of InvTriforce.
    BNE @PopSetAttr             ; Link has this level's piece: keep attribute 3.
:
    ; The piece is missing (or level 9): blink.
    LDA FrameCounter
    AND #$1F
    CMP #$10
    BCS @PopSetAttr             ; Second 16 frames of every 32: attribute 3.
    PLA
    LDA #$02                    ; First 16 frames of every 32: attribute 2.
    PHA

@PopSetAttr:
    PLA

@SetAttr:
    STA Sprites+86, X

; Shared RTS of the two routines above.
StatusBar_UpdatePositionMarker_Return:
    RTS

; ----------------------------------------------------------------------------
; Runs one curtain-effect step for a caller in bank 2, then switches bank 2 back in.
; ----------------------------------------------------------------------------
Mode_UpdateCurtainEffect_ReturnToBank2:
    JSR Mode_UpdateCurtainEffect
    LDA #$02
    JMP Mmc1_SwitchBank_Bank2

; ----------------------------------------------------------------------------
; Curtain effect step: copies two tile columns of the play area to the transfer buffer, then waits.
; IN:   ObjX[$0C] ($7C) = column that moves down by 1 per step; ObjX[$0D] ($7D) = column that moves
;       up by 1 per step (each is a play area tile column + 1, $01-$20); ObjTimer[0] = delay left.
; OUT:  when ObjTimer[0] = 0: both columns queued by Transfer_CopyColumnToTileBuf (first ObjX[$0D]'s, then
;       ObjX[$0C]'s), ObjX[$0C] decremented, ObjX[$0D] incremented, CurColumn = $FF, ObjTimer[0] = 5.
;       Otherwise nothing.
; NOTE: Leaves bank 5 switched in and MMC1 control set to $0E (vertical mirroring). Uses [0A].
; NOTE: No range test on the columns: the caller stops calling. Mode 3 starts with ObjX[$0C] = $10
;       and ObjX[$0D] = $11 and stops at ObjX[$0C] = 0 (the room appears from the middle outward).
;       Modes $12 and $13 start with $20 and 1 over a blank tile map and stop at ObjX[$0C] < $11
;       (the room is wiped from both screen edges to the middle).
; ----------------------------------------------------------------------------
Mode_UpdateCurtainEffect:
    LDA ObjTimer
    BNE @Exit                   ; Delay until ObjTimer[0] expires.
    LDA #$01                    ; Start with the column index in ObjX[$0D].
    STA $0A

@LoopColumn:
    LDX $0A
    LDA ObjX+12, X              ; Get current column index.
    STA CurColumn               ; Set CurColumn for column copying routine to use.
    LDA #$05
    JSR Mmc1_SwitchBank_Bank5
    LDA #$0E                    ; Set vertical nametable mirroring.
    JSR Mmc1_SetControl_Bank5
    JSR Transfer_CopyColumnToTileBuf
    DEC $0A                     ; Now reference the column with index in ObjX[$0C].
    BPL @LoopColumn             ; Loop until both columns were copied.
    LDA #$FF                    ; Invalidate CurColumn
    STA CurColumn
    LDA #$05                    ; Wait for ObjTimer[0] to count down from 5 before the next pair.
    STA ObjTimer
    DEC ObjX+12                 ; Change the column indexes.
    INC ObjX+13

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Adds 1 to the 16-bit value at [00:01].
; OUT:  [00:01] = sum; A = low byte of the sum.
; ----------------------------------------------------------------------------
Math_Add1ToInt16At0:
    LDA #$01

; ----------------------------------------------------------------------------
; Adds A (unsigned) to the 16-bit value at [00:01].
; IN:   A = amount.
; OUT:  [00:01] = sum; A = low byte of the sum.
; ----------------------------------------------------------------------------
Math_AddToInt16At0:
    CLC
    ADC $00
    STA $00
    BCC :+
    INC $01
:
    RTS

; ----------------------------------------------------------------------------
; Adds 1 to the 16-bit value at [02:03].
; OUT:  [02:03] = sum; A = low byte of the sum.
; ----------------------------------------------------------------------------
Math_Add1ToInt16At2:
    LDA #$01

; ----------------------------------------------------------------------------
; Adds A (unsigned) to the 16-bit value at [02:03].
; IN:   A = amount.
; OUT:  [02:03] = sum; A = low byte of the sum.
; ----------------------------------------------------------------------------
Math_AddToInt16At2:
    CLC
    ADC $02
    STA $02
    BCC :+
    INC $03
:
    RTS

; ----------------------------------------------------------------------------
; Adds 1 to the 16-bit value at [04:05].
; OUT:  [04:05] = sum; A = low byte of the sum.
; ----------------------------------------------------------------------------
Math_Add1ToInt16At4:
    LDA #$01

; ----------------------------------------------------------------------------
; Adds A (unsigned) to the 16-bit value at [04:05].
; IN:   A = amount.
; OUT:  [04:05] = sum; A = low byte of the sum.
; ----------------------------------------------------------------------------
Math_AddToInt16At4:
    CLC
    ADC $04
    STA $04
    BCC :+
    INC $05
:
    RTS

; ----------------------------------------------------------------------------
; Subtracts 1 from the 16-bit value at [04:05].
; OUT:  [04:05] = difference; A = low byte of the difference.
; ----------------------------------------------------------------------------
Math_Sub1FromInt16At4:
    LDA $04
    SEC
    SBC #$01
    STA $04
    BCS :+
    DEC $05
:
    RTS
