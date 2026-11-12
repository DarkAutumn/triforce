; ============================================================================
; FILE:     bank5/mode7-scroll-update.asm BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 1039-1261
; SUBJECTS: Mode7Scroll, Transfer
; TOPIC:    Mode 7 (scroll to the next room), update half: the eight update submodes that set up the
;           scroll start, scroll one step per frame while copying the new room's tiles into nametable 0,
;           transfer its attributes, and hand over to mode 4 (enter room).
; SEE ALSO: bank5/mode7-scroll-init.asm (mode 7 init submodes: lays out the next room, sets RoomId, turns
;           the sprite-0 split on); bank5/scroll-world.asm (Mode7Scroll_ScrollWorld: the per-frame scroll step and
;           Link's slide); bank5/sprite0-split-scroll.asm (applies the scroll below the status bar);
;           bank5/mode12-lay-out-room.asm (Transfer_CopyColumnToTileBuf, Transfer_CopyRowToTileBuf); bank5/enter-room.asm
;           (mode 4, entered from submode 7); bank7/update-mode-dispatch.asm (Mode7Scroll_Update caller).
; KEY FACTS:
;   - Submode order: scrolling up 0 -> 1 -> 2 -> 3 -> 4 -> 6 (-> 7); other directions 0 -> 2 -> 3 -> 4 ->
;     5 -> 6 (-> 7). Mode7Scroll_UpdateSubmode_JumpTable, Mode7Scroll_Update_Sub0, Mode7Scroll_Update_Sub4.
;   - Submode 3 is the scroll itself: Mode7Scroll_ScrollWorld once per frame until it advances the submode.
;     Horizontal: 4 pixels per frame in the overworld (64 frames), 2 in a dungeon (128 frames). Vertical:
;     one 8-pixel row every 2 frames in the overworld, every 4 frames in a dungeon.
;     Mode7Scroll_Update_Sub3, Mode7Scroll_Update_Sub2 (Mode7Scroll_ScrollWorld in bank5/scroll-world.asm).
;   - The start values of CurColumn are chosen so that columns $20..1 (left) or 1..$20 (right) are copied
;     during the last 32 frames of a horizontal scroll. Mode7Scroll_Update_Sub0_Horizontal, Transfer_CopyColumnOrRowToTileBuf.
;   - Submode 7 ends the mode: GameMode = 4 with GameSubmode = 1 and IsUpdatingMode = 0, sprite-0 split
;     off. Controller input is not read while the split is on, so the whole scroll ignores input.
;     Mode7Scroll_Update_Sub7.
;   - Scrolling into a dark dungeon room adds one frame (submode 6 -> 7) and sets CurRow = 0, which makes
;     mode 4 submode 1 redraw all $16 play-area rows, one per frame. Mode7Scroll_Update_Sub6.
;   - RoomId already holds the new room during all of these submodes (set at the end of mode 7 init).
;     Mode7Scroll_Update_Sub0_Up, Mode7Scroll_Update_Sub6.
;   - Frames spent in these submodes when the new room is not dark, counted from the code (not timed):
;     left or right 69 in the overworld (0, 2, 64 x 3, 4, 5, 6+7), 133 in a dungeon; down 48 / 90
;     (22 row steps); up 50 / 94 (23 row steps, submodes 0, 1, 2, 3, 4, 6+7).
;     Mode7Scroll_UpdateSubmode_JumpTable, Mode7Scroll_Update_Sub2, Mode7Scroll_Update_Sub3.
; CONTENTS: (generated; L = line in this file)
;   L0058 Mode7Scroll_UpdateSubmodeAndDrawLink               code  Mode 7 update entry: runs the current scroll submode, then draws Link (overworld only).
;   L0065 Mode7Scroll_UpdateSubmode                          code  Dispatches to the mode 7 update routine for GameSubmode (0-7).
;   L0070 Mode7Scroll_UpdateSubmode_JumpTable                data  TABLE: mode 7 update submode handlers; index = GameSubmode; value = routine address.
;   L0086 Mode7Scroll_Update_Sub0                            code  Submode 0: sets the scroll start position and first row or column for Link's direction.
;   L0104 Mode7Scroll_Inc2Submodes                           code  Advances GameSubmode by 2 (0 -> 2 here; 4 -> 6 from Mode7Scroll_Update_Sub4).
;   L0110 Mode7Scroll_Update_Sub0_Horizontal                 code  Submode 0, left or right: chooses the first CurColumn value, then goes to submode 2.
;   L0137 Mode7Scroll_Update_Sub0_Up                         code  Submode 0, up: sets the scroll start, fills the new room's attributes, falls into submode 1's code.
;   L0150 Mode7Scroll_Update_Sub1                            code  Submode 1 (up only): cues half of the new room's attributes for NT 0 and advances the submode.
;   L0161 Transfer_ChoosePlayAreaAttrsHalfBySubmode          code  Picks the top half of the play-area attributes in submode 0, the bottom half in any other submode.
;   L0172 Transfer_GetPlayAreaAttrsBottomHalf                code  Turns a top-half attribute request into the bottom-half one.
;   L0185 Mode7Scroll_Update_Sub2                            code  Submode 2: sets VScrollStartFrame so that a vertical scroll moves its first row on the next frame.
;   L0207 Mode7Scroll_Update_Sub3                            code  Submode 3: scrolls one step and copies the row or column that came into view to the transfer buffer.
;   L0227 Mode7Scroll_Update_Sub6                            code  Submode 6: if the room just entered is a dark dungeon room, sets CurRow = 0 and goes to submode 7.
;   L0242 Mode7Scroll_Update_Sub7                            code  Submode 7: ends the scroll and switches to mode 4 (enter room), init submode 1.
;   L0258 Mode7Scroll_Update_Sub4                            code  Submode 4: cues the top half of the new room's attributes for NT 0, unless scrolling up.
;   L0269 Mode7Scroll_Update_Sub4And5_TransferAttrs          code  Submodes 4 and 5: cue the top half (submode 4) or bottom half (5) of the new room's attributes.
;   L0291 Transfer_CuePlayAreaAttrsHalfNT0AndAdvanceSubmode  code  Cues a half of PlayAreaAttrs for transfer to NT 0's attribute table ($23xx); advances GameSubmode.
;   L0302 Transfer_CopyColumnOrRowToTileBuf                  code  Copies the current tile row, or else the current tile column, of the new room to the transfer buffer.
; ============================================================================

; ----------------------------------------------------------------------------
; Mode 7 update entry: runs the current scroll submode, then draws Link (overworld only).
; NOTE: Link_EndMoveAndAnimateBetweenRooms returns at once in a dungeon (CurLevel <> 0).
; ----------------------------------------------------------------------------
Mode7Scroll_UpdateSubmodeAndDrawLink:
    JSR Mode7Scroll_UpdateSubmode
    JMP Link_EndMoveAndAnimateBetweenRooms

; ----------------------------------------------------------------------------
; Dispatches to the mode 7 update routine for GameSubmode (0-7).
; ----------------------------------------------------------------------------
Mode7Scroll_UpdateSubmode:
    LDA GameSubmode
    JSR Util_TableJump

; TABLE: mode 7 update submode handlers; index = GameSubmode; value = routine address.
Mode7Scroll_UpdateSubmode_JumpTable:
    .ADDR Mode7Scroll_Update_Sub0                        ; 0: set scroll start values by direction
    .ADDR Mode7Scroll_Update_Sub1                        ; 1: (up only) new room attributes, bottom half
    .ADDR Mode7Scroll_Update_Sub2                        ; 2: pick the vertical scroll frame phase
    .ADDR Mode7Scroll_Update_Sub3                        ; 3: scroll, copying rows or columns
    .ADDR Mode7Scroll_Update_Sub4                        ; 4: attributes top half (skipped when going up)
    .ADDR Mode7Scroll_Update_Sub4And5_TransferAttrs    ; 5: attributes bottom half
    .ADDR Mode7Scroll_Update_Sub6                        ; 6: dark room test, usually falls into 7
    .ADDR Mode7Scroll_Update_Sub7                        ; 7: go to mode 4 submode 1

; ----------------------------------------------------------------------------
; Submode 0: sets the scroll start position and first row or column for Link's direction.
; IN:   ObjDir (slot 0) = scroll direction bit (1 right, 2 left, 4 down, 8 up).
; OUT:  Down, left, right: GameSubmode = 2. Up: falls through Mode7Scroll_Update_Sub0_Up into submode 1's code, which
;       cues the top half of the new room's attributes and leaves GameSubmode = 1.
; ----------------------------------------------------------------------------
Mode7Scroll_Update_Sub0:
    LDA #$00
    STA VScrollAddrLo           ; Low byte of the split-scroll PPU address (vertical scrolling).
    STA CurHScroll              ; Horizontal scroll offset (horizontal scrolling).
    LDA #$08
    BIT ObjDir
    BNE Mode7Scroll_Update_Sub0_Up                ; Facing up.
    LSR
    BIT ObjDir                  ; A = 4: facing down?
    BEQ Mode7Scroll_Update_Sub0_Horizontal

    ; Scrolling down.
    LDA #$21                    ; Start at $2100, the top of the play area in NT 0 (the room being left).
    STA VScrollAddrHi
    LDA #$FF                    ; Mode7Scroll_ScrollWorld increments CurRow before each row is copied: first row is 0.
    STA CurRow

; Advances GameSubmode by 2 (0 -> 2 here; 4 -> 6 from Mode7Scroll_Update_Sub4).
Mode7Scroll_Inc2Submodes:
    INC GameSubmode
    INC GameSubmode
    RTS

; Submode 0, left or right: chooses the first CurColumn value, then goes to submode 2.
Mode7Scroll_Update_Sub0_Horizontal:
    ; Transfer_CopyColumnOrRowToTileBuf copies a column only while 1 <= CurColumn <= $20. Mode7Scroll_ScrollWorld changes
    ; CurColumn by one each frame, and the scroll takes 64 frames in the overworld (4 pixels a frame)
    ; and 128 in a dungeon (2 pixels a frame). The start values below make CurColumn pass through
    ; the valid range during the last 32 frames:
    ;   right: $E0 (overworld) or $A0 (dungeon), counting up through 0 to $20;
    ;   left:  $41 (overworld) or $81 (dungeon), counting down to 1.
    LDY #$A0                    ; Right, dungeon.
    LDX CurLevel
    BNE :+
    LDY #$E0                    ; Right, overworld.
:
    LSR                         ; A = 2.
    BIT ObjDir
    BEQ @SetColumn              ; Not facing left: keep the value for right.

    ; Scrolling left.
    LDY #$81                    ; Left, dungeon.
    LDX CurLevel
    BNE @SetColumn
    LDY #$41                    ; Left, overworld.

@SetColumn:
    STY CurColumn
    JMP Mode7Scroll_Inc2Submodes            ; Go to submode 2.

; Submode 0, up: sets the scroll start, fills the new room's attributes, falls into submode 1's code.
Mode7Scroll_Update_Sub0_Up:
    LDA #$28                    ; Start at $2800, the top of NT 2, which holds the room being left.
    STA VScrollAddrHi
    LDA #$16                    ; Mode7Scroll_ScrollWorld decrements CurRow before each row is copied: first row is $15.
    STA CurRow
    LDA RoomId                  ; RoomId is already the new room.
    JSR Room_FillPlayAreaAttrs

; ----------------------------------------------------------------------------
; Submode 1 (up only): cues half of the new room's attributes for NT 0 and advances the submode.
; NOTE: Reached twice when scrolling up: by fall-through in submode 0 (top half, $23D0) and on the
;       next frame as submode 1 (bottom half, $23E8). Ends with GameSubmode = 2.
; ----------------------------------------------------------------------------
Mode7Scroll_Update_Sub1:
    JSR Transfer_ChoosePlayAreaAttrsHalfBySubmode
    JMP Transfer_CuePlayAreaAttrsHalfNT0AndAdvanceSubmode

; ----------------------------------------------------------------------------
; Picks the top half of the play-area attributes in submode 0, the bottom half in any other submode.
; IN:   GameSubmode.
; OUT:  A = low byte of the PPU attribute address ($D0 top, $E8 bottom);
;       Y = offset of the last byte of that half in PlayAreaAttrs ($17 top, $2F bottom).
; NOTE: Also called by the death mode (Mode11Death_Update_Sub0 / _Sub1).
; ----------------------------------------------------------------------------
Transfer_ChoosePlayAreaAttrsHalfBySubmode:
    LDA #$D0                    ; PPU address $xxD0: attributes of the top of the play area.
    LDY #$17                    ; End offset of the first $18 attribute bytes.
    LDX GameSubmode
    BEQ :+                      ; Submode 0: return the top half.

; ----------------------------------------------------------------------------
; Turns a top-half attribute request into the bottom-half one.
; IN:   A = low byte of the PPU address of the top half.
; OUT:  A = A + $18; Y = $2F (offset of the last byte of the second half of PlayAreaAttrs).
; ----------------------------------------------------------------------------
Transfer_GetPlayAreaAttrsBottomHalf:
    LDY #$2F
    CLC
    ADC #$18
:
    RTS

; ----------------------------------------------------------------------------
; Submode 2: sets VScrollStartFrame so that a vertical scroll moves its first row on the next frame.
; NOTE: VScrollStartFrame = (FrameCounter + 1) AND 3 in a dungeon, AND 1 in the overworld; Mode7Scroll_ScrollWorld
;       moves a row only on frames whose counter has that phase: every 4th frame in a dungeon, every
;       2nd in the overworld. Horizontal scrolling ignores the phase.
; ----------------------------------------------------------------------------
Mode7Scroll_Update_Sub2:
    INC GameSubmode
    LDA FrameCounter

    ; Add 1: the phase must match the frame counter as Mode7Scroll_ScrollWorld will see it next frame.
    CLC
    ADC #$01

    ; Same masking as in Mode7Scroll_ScrollWorld.
    AND #$03
    LDY CurLevel
    BNE :+
    AND #$01                    ; Overworld.
:
    STA VScrollStartFrame
    RTS

; ----------------------------------------------------------------------------
; Submode 3: scrolls one step and copies the row or column that came into view to the transfer buffer.
; NOTE: Mode7Scroll_ScrollWorld advances GameSubmode when the scroll is complete; on that frame the row and
;       column trackers are invalidated (CurRow = PrevRow = $FF, CurColumn = 0).
; ----------------------------------------------------------------------------
Mode7Scroll_Update_Sub3:
    JSR Mode7Scroll_ScrollWorld
    JSR Transfer_CopyColumnOrRowToTileBuf
    LDA GameSubmode
    CMP #$03
    BEQ :+                      ; Still scrolling.
    LDY #$FF                    ; Scroll finished: no current or previous row, no current column.
    STY CurRow
    STY PrevRow
    INY
    STY CurColumn
:
    RTS

; ----------------------------------------------------------------------------
; Submode 6: if the room just entered is a dark dungeon room, sets CurRow = 0 and goes to submode 7.
; NOTE: Otherwise (overworld, or a room that is not dark) falls straight into submode 7 this frame.
;       CurRow = 0 (instead of $FF) makes mode 4 submode 1 transfer the whole play area again, one
;       row per frame for $16 frames (Mode4Enter_Init). RoomId is the new room here.
; ----------------------------------------------------------------------------
Mode7Scroll_Update_Sub6:
    LDA CurLevel
    BEQ Mode7Scroll_Update_Sub7  ; Overworld: no dark rooms.
    LDY RoomId
    JSR Room_IsDark_Bank5
    BEQ Mode7Scroll_Update_Sub7  ; Not a dark room.
    LDA #$00
    STA CurRow
    INC GameSubmode
    RTS

; ----------------------------------------------------------------------------
; Submode 7: ends the scroll and switches to mode 4 (enter room), init submode 1.
; NOTE: Turns the sprite-0 split off; Mode7Scroll_Update (bank 7) then restores horizontal mirroring.
; ----------------------------------------------------------------------------
Mode7Scroll_Update_Sub7:
    LDA #$01
    STA GameSubmode
    LSR                         ; A = 0.
    STA IsUpdatingMode          ; Run the next mode's init routine.
    STA UnusedScrollEndByte     ; No other reference to this address in the source.
    STA NextRoomDir             ; The direction bit of the next-room calculation; cleared.
    STA IsSprite0CheckActive    ; The scroll is finished: no more sprite-0 split.
    LDA #$04                    ; Mode 4 (enter room), submode 1.
    STA GameMode
    RTS

; ----------------------------------------------------------------------------
; Submode 4: cues the top half of the new room's attributes for NT 0, unless scrolling up.
; NOTE: Scrolling up already sent both halves in submodes 0 and 1, so it skips to submode 6.
; ----------------------------------------------------------------------------
Mode7Scroll_Update_Sub4:
    LDA #$08
    BIT ObjDir
    BEQ Mode7Scroll_Update_Sub4And5_TransferAttrs    ; Not facing up.
    JMP Mode7Scroll_Inc2Submodes            ; Facing up: go to submode 6.

; ----------------------------------------------------------------------------
; Submodes 4 and 5: cue the top half (submode 4) or bottom half (5) of the new room's attributes.
; NOTE: In submode 5 of a horizontal scroll OddBaseNameTableOverride is cleared, so the split goes
;       back to basing the picture on NT 0. Advances GameSubmode.
; ----------------------------------------------------------------------------
Mode7Scroll_Update_Sub4And5_TransferAttrs:
    LDA #$D0                    ; PPU address $23D0: attributes of the top half of the play area in NT 0.
    LDY #$17                    ; End offset of the first half of PlayAreaAttrs.
    LDX GameSubmode
    CPX #$04
    BEQ Transfer_CuePlayAreaAttrsHalfNT0AndAdvanceSubmode    ; Submode 4: top half.

    ; Submode 5: bottom half.
    PHA                         ; Save the low address byte.
    LDA ObjDir
    CMP #$04
    BCS :+                      ; Direction 4 or 8: vertical scroll, leave the override alone.
    LDA #$00
    STA OddBaseNameTableOverride    ; Horizontal scroll: stop using NT 1 as the base nametable.
:
    PLA
    JSR Transfer_GetPlayAreaAttrsBottomHalf

; ----------------------------------------------------------------------------
; Cues a half of PlayAreaAttrs for transfer to NT 0's attribute table ($23xx); advances GameSubmode.
; IN:   A = low byte of the PPU address; Y = offset of the last byte to copy from PlayAreaAttrs.
; ----------------------------------------------------------------------------
Transfer_CuePlayAreaAttrsHalfNT0AndAdvanceSubmode:
    LDX #$23                    ; High byte of the PPU address.
    JMP Transfer_CuePlayAreaAttrsHalfAndAdvanceSubmode

; ----------------------------------------------------------------------------
; Copies the current tile row, or else the current tile column, of the new room to the transfer buffer.
; IN:   CurRow, PrevRow, CurColumn.
; OUT:  PrevRow = CurRow when a row was copied.
; NOTE: A row is copied when CurRow < $16 and differs from PrevRow (a vertical scroll stays on one
;       row for 2 or 4 frames). Otherwise a column is copied when 1 <= CurColumn <= $20.
; ----------------------------------------------------------------------------
Transfer_CopyColumnOrRowToTileBuf:
    LDA CurRow
    CMP #$16
    BCS @CheckColumn            ; No valid row.
    CMP PrevRow
    BEQ @Exit                   ; This row was already copied.
    STA PrevRow
    JMP Transfer_CopyRowToTileBuf

@CheckColumn:
    LDA CurColumn
    BEQ @Exit
    CMP #$21
    BCS @Exit
    JMP Transfer_CopyColumnToTileBuf

@Exit:
    RTS
