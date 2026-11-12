; ============================================================================
; FILE:     bank5/mode7-scroll-update.asm BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 1039-1261
; SUBJECTS: (none)
; TOPIC:    Mode 7 update submodes
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0029 UpdateMode7SubmodeAndDrawLink                     code
;   L0033 UpdateMode7ScrollSubmode                          code
;   L0037 UpdateMode7ScrollSubmode_JumpTable                data
;   L0047 UpdateMode7Scroll_Sub0                            code
;   L0064 Inc2Submodes                                      code
;   L0069 ScrollHorizontal                                  code
;   L0097 ScrollUp                                          code
;   L0106 UpdateMode7Scroll_Sub1                            code
;   L0110 ChooseAttrSourceAndDestForSubmode                 code
;   L0126 GetPlayAreaAttrsBottomHalfInfo                    code  Params:
;   L0133 UpdateMode7Scroll_Sub2                            code
;   L0154 UpdateMode7Scroll_Sub3                            code
;   L0168 UpdateMode7Scroll_Sub6                            code
;   L0185 UpdateMode7Scroll_Sub7                            code
;   L0197 UpdateMode7Scroll_Sub4                            code
;   L0204 UpdateMode7Scroll_Sub4And5_TransferNTAttrs        code
;   L0226 CueTransferPlayAreaAttrsHalfAndAdvanceSubmodeNT0  code  Params:
;   L0230 CopyColumnOrRowToTileBuf                          code
; ============================================================================

UpdateMode7SubmodeAndDrawLink:
    JSR UpdateMode7ScrollSubmode
    JMP Link_EndMoveAndAnimateBetweenRooms

UpdateMode7ScrollSubmode:
    LDA GameSubmode
    JSR TableJump

UpdateMode7ScrollSubmode_JumpTable:
    .ADDR UpdateMode7Scroll_Sub0
    .ADDR UpdateMode7Scroll_Sub1
    .ADDR UpdateMode7Scroll_Sub2
    .ADDR UpdateMode7Scroll_Sub3
    .ADDR UpdateMode7Scroll_Sub4
    .ADDR UpdateMode7Scroll_Sub4And5_TransferNTAttrs
    .ADDR UpdateMode7Scroll_Sub6
    .ADDR UpdateMode7Scroll_Sub7

UpdateMode7Scroll_Sub0:
    LDA #$00
    STA VScrollAddrLo           ; Reset low byte of VScroll address for vertical scrolling.
    STA CurHScroll              ; Reset horizontal scroll offset for horizontal scrolling.
    LDA #$08
    BIT ObjDir
    BNE ScrollUp                ; If scrolling up, go handle it.
    LSR
    BIT ObjDir
    BEQ ScrollHorizontal

    ; Scrolling down.
    LDA #$21                    ; Start scrolling from $2100, the top of play area in NT 0 (current room).
    STA VScrollAddrHi
    LDA #$FF                    ; Start past the first row, because the scrolling process first increments it.
    STA CurRow

Inc2Submodes:
    INC GameSubmode             ; From submode 0 go to submode 2.
    INC GameSubmode
    RTS

ScrollHorizontal:
    ; Scrolling left or right.
    ;
    ; Keep in mind that columns will not be copied if CurColumn
    ; is not between 1 and $20.
    ;
    ; Set up a starting column number well past the beginning or
    ; end of the play area; so that we don't write to a part of
    ; the scroll region.
    LDY #$A0                    ; $A0 in UW.
    LDX CurLevel
    BNE :+
    LDY #$E0                    ; $E0 in OW.
:
    LSR
    BIT ObjDir
    BEQ @SetColumn

    ; Scrolling left.
    LDY #$81                    ; $81 in UW.
    LDX CurLevel
    BNE @SetColumn
    LDY #$41                    ; $41 in OW.

@SetColumn:
    STY CurColumn               ; Set CurColumn to the value we determined.
    JMP Inc2Submodes            ; Go to submode 2.

ScrollUp:
    ; Scrolling up.
    LDA #$28                    ; Start scrolling from $2800, the top of play area in NT 2 (current room).
    STA VScrollAddrHi
    LDA #$16                    ; Start past the last row, because the scrolling process first decrements it.
    STA CurRow
    LDA RoomId
    JSR FillPlayAreaAttrs

UpdateMode7Scroll_Sub1:
    JSR ChooseAttrSourceAndDestForSubmode
    JMP CueTransferPlayAreaAttrsHalfAndAdvanceSubmodeNT0

ChooseAttrSourceAndDestForSubmode:
    ; Results:
    ; A: low PPU address
    ; Y: end offset in PlayAreaAttrs to copy from
    LDA #$D0                    ; Low byte $D0 mean destination PPU address of $23D0, $27D0, $2BD0, or $2FD0.
    LDY #$17                    ; This is the offset of the end of the first half of play area NT attributes.
    LDX GameSubmode
    BEQ :+                      ; Return.

; Params:
; A: low PPU address
;
; Returns:
; A: low PPU address + $18
; Y: end offset of second half of play area NT attributes
;
GetPlayAreaAttrsBottomHalfInfo:
    LDY #$2F                    ; This is the offset of the end of the second half of play area NT attributes.
    CLC
    ADC #$18                    ; Add $18 to point to the bottom half of NT attributes in PPU memory.
:
    RTS

UpdateMode7Scroll_Sub2:
    ; Calculates a VScrollingStartFrame value that enables
    ; immediate vertical scrolling in the next frame and submode.
    INC GameSubmode
    LDA FrameCounter

    ; Add 1, so that the frame value calculated in the next frame
    ; will match the VScrollingStartFrame value we calculate here.
    CLC
    ADC #$01

    ; Calculate VScrollingStartFrame the same way as the frame
    ; value in the next submode.
    AND #$03
    LDY CurLevel
    BNE :+
    AND #$01
:
    STA VScrollStartFrame
    RTS

UpdateMode7Scroll_Sub3:
    JSR ScrollWorld
    JSR CopyColumnOrRowToTileBuf
    LDA GameSubmode
    CMP #$03
    BEQ :+                      ; If the submode has advanced,
    LDY #$FF                    ; then invalidate current and previous row, and reset the current column.
    STY CurRow
    STY PrevRow
    INY
    STY CurColumn
:
    RTS

UpdateMode7Scroll_Sub6:
    LDA CurLevel
    BEQ UpdateMode7Scroll_Sub7  ; If in OW, then don't need to handle dark rooms. Go finish up.
    LDY RoomId
    JSR IsDarkRoom_Bank5
    BEQ UpdateMode7Scroll_Sub7  ; If the room isn't dark, then go finish up.

    ; Set CurRow = 0 (was $FF) to signal to mode 4 that
    ; it might need to brighten the new room, because we
    ; scrolled from a dark one.
    ;
    ; This isn't a concern, if you enter a room by any other mode.
    LDA #$00
    STA CurRow
    INC GameSubmode
    RTS

UpdateMode7Scroll_Sub7:
    LDA #$01
    STA GameSubmode
    LSR
    STA IsUpdatingMode          ; Reset to initialize the next mode.
    STA $010C                   ; UNKNOWN: [$010C] seems to be a variable that's no longer used.
    STA $E7                     ; UNKNOWN: [E7] seems to be used only for calculating NextRoomId.
    STA IsSprite0CheckActive    ; Reset sprite-0 check, because we finished scrolling.
    LDA #$04                    ; Go to mode 4 submode 1.
    STA GameMode
    RTS

UpdateMode7Scroll_Sub4:
    ; Transfer new room's attributes to NT 0.
    LDA #$08
    BIT ObjDir
    BEQ UpdateMode7Scroll_Sub4And5_TransferNTAttrs    ; If not facing up, then go transfer new room attributes to NT 0.
    JMP Inc2Submodes            ; Go to submode 6.

UpdateMode7Scroll_Sub4And5_TransferNTAttrs:
    LDA #$D0                    ; This will be PPU address $23D0, attributes for top half of play area in NT 0.
    LDY #$17                    ; Ending offset of first half of attributes.
    LDX GameSubmode
    CPX #$04                    ; If in submode 4,
    BEQ CueTransferPlayAreaAttrsHalfAndAdvanceSubmodeNT0    ; then transfer top half.

    ; Transfer bottom half.
    PHA                         ; Save low byte of PPU address of top of play area attributes.
    LDA ObjDir
    CMP #$04
    BCS :+                      ; If scrolling horizontally,
    LDA #$00
    STA OddBaseNameTableOverride    ; then don't use NT 1 anymore.
:
    PLA                         ; Restore low byte of PPU address of top of play area attributes.
    JSR GetPlayAreaAttrsBottomHalfInfo

; Params:
; A: low PPU address
; Y: end offset in PlayAreaAttrs to copy from
;
CueTransferPlayAreaAttrsHalfAndAdvanceSubmodeNT0:
    LDX #$23                    ; High byte of destination PPU address for play area attributes.
    JMP CueTransferPlayAreaAttrsHalfAndAdvanceSubmode

CopyColumnOrRowToTileBuf:
    ; Copies a row if CurRow is valid. Otherwise, tries to copy a column.
    ;
    ; If CurRow = PrevRow, does nothing. After a row is copied,
    ; PrevRow is assigned CurRow.
    LDA CurRow                  ; If CurRow is valid, then copy row.
    CMP #$16
    BCS @CheckColumn
    CMP PrevRow                 ; unless CurRow = LastRow, then don't repeat
    BEQ @Exit
    STA PrevRow
    JMP CopyRowToTileBuf

@CheckColumn:
    LDA CurColumn               ; If 0 < CurColumn < $21, then copy column.
    BEQ @Exit
    CMP #$21
    BCS @Exit
    JMP CopyColumnToTileBuf

@Exit:
    RTS
