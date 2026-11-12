; ============================================================================
; FILE:     bank5/submenu-states.asm     BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 155-550
; SUBJECTS: (none)
; TOPIC:    Submenu state machine, triforce display
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0033 UpdateMenuAndMeters              code
;   L0037 UpdateMenu                       code
;   L0045 UpdateMenuUW_JumpTable           data
;   L0059 UpdateMenuOW_JumpTable           data
;   L0071 UpdateMenuCommon1                code
;   L0112 UpdateMenu_Return                code
;   L0115 UpdateMenuCommon2                code
;   L0120 SelectTransferBufAndIncState     code
;   L0126 UpdateMenuCommon3                code
;   L0132 UpdateMenuCommon4                code
;   L0138 UpdateMenu5UW                    code
;   L0142 UpdateMenu5OW                    code
;   L0147 UpdateMenuScrollDownOW           code
;   L0151 UpdateMenuScrollDownUW           code
;   L0240 UpdateMenuActive                 code
;   L0277 UpdateMenuScrollUp               code
;   L0313 MovePositionMarkers              code  Params:
;   L0334 Submenu_Return                   code
;   L0337 TriforceTransferBufOffsets       data
;   L0342 TriforceTriforceBufReplacements  data
;   L0347 TriforceTransferBufTiles         data
;   L0352 UpdateMenuStartOW                code
; ============================================================================

UpdateMenuAndMeters:
    JSR UpdateMenu
    JMP UpdateHeartsAndRupees

UpdateMenu:
    LDA MenuState
    LDY CurLevel
    BEQ :+

    ; Update menu in UW.
    JSR TableJump

UpdateMenuUW_JumpTable:
    .ADDR UpdateMenu_Return
    .ADDR UpdateMenuCommon1
    .ADDR UpdateMenuCommon2
    .ADDR UpdateMenuCommon3
    .ADDR UpdateMenuCommon4
    .ADDR UpdateMenu5UW
    .ADDR UpdateMenuScrollDownUW
    .ADDR UpdateMenuActive
    .ADDR UpdateMenuScrollUp
:
    ; Update menu in OW.
    JSR TableJump

UpdateMenuOW_JumpTable:
    .ADDR UpdateMenu_Return
    .ADDR UpdateMenuStartOW
    .ADDR UpdateMenuCommon1
    .ADDR UpdateMenuCommon2
    .ADDR UpdateMenuCommon3
    .ADDR UpdateMenuCommon4
    .ADDR UpdateMenu5OW
    .ADDR UpdateMenuScrollDownOW
    .ADDR UpdateMenuActive
    .ADDR UpdateMenuScrollUp

UpdateMenuCommon1:
    JSR HideAllSprites
    JSR UpdatePlayerPositionMarker
    JSR UpdateTriforcePositionMarker

    ; Move position markers and hardware vertical scroll position
    ; down 1 pixel.
    ;
    ; Because we'll be above the top of NT 0, switch to NT 2 to
    ; be at the bottom of NT 2.
    LDA #$EF
    STA CurVScroll
    STA SwitchNameTablesReq
    LDA #$01                    ; 1 pixel
    JSR MovePositionMarkers
    INC MenuState               ; Set menu state 2.

    ; SubmenuScrollProgress begins at $2B. Each frame it will be
    ; decremented. It encodes a submenu row index in bits 1 to 7,
    ; and a flag in bit 0.
    ;
    ; When the flag is 1, a full row of black tiles will be transferred
    ; at the current row. Otherwise, one of various static visual
    ; elements will be transferred.
    ;
    ; For example:
    ; 1. In the first frame of scrolling, $2B indicates that a full row
    ;    of black tiles must be transferred to row $15.
    ; 2. In the second frame ($2A), submenu row will again be
    ;    $15, but something else will be transferred.
    ; 3. In the third frame ($29), a full row of black tiles will be
    ;    transferred to row $14.
    LDA #$2B
    STA SubmenuScrollProgress

    ; In UW, this variable will be used to scan every room in order
    ; to build the big sheet map in the submenu. It will range
    ; from $7F to 0.
    LDA #$7F
    STA CurScanRoomId

UpdateMenu_Return:
    RTS

UpdateMenuCommon2:
    ; Cue the transfer of first set of submenu nametable attributes to NT 2.
    ; Advance state.
    LDA #$48

SelectTransferBufAndIncState:
    STA TileBufSelector
:
    INC MenuState
    RTS

UpdateMenuCommon3:
    ; Cue the transfer of second set of submenu nametable attributes to NT 2.
    ; Advance state.
    LDA #$4A
    BNE SelectTransferBufAndIncState

UpdateMenuCommon4:
    ; Cue the transfer of a blank row of tiles to the bottom of NT 2.
    ; Advance state.
    LDA #$4C
    BNE SelectTransferBufAndIncState

UpdateMenu5UW:
    JSR Submenu_CueTransferRowUW
    JMP :-

UpdateMenu5OW:
    ; Cue the transfer of "TRIFORCE" text.
    LDA #$5C
    BNE SelectTransferBufAndIncState

UpdateMenuScrollDownOW:
    JSR Submenu_CueTransferRowOW
    JMP :+

UpdateMenuScrollDownUW:
    JSR Submenu_CueTransferRowUW
:
    ; Move position markers and advance nametable scrolling;
    ; so that we scroll down 3 pixels.
    LDA #$03
    JSR MovePositionMarkers
    LDA CurVScroll
    SEC
    SBC #$03
    STA CurVScroll

    ; There's nothing else to do until we reach hardware VScroll=$41. Return.
    CMP #$41
    BNE @Exit

    ; VScroll reached $41.
    ; Advance the submenu state.
    ; If in OW or in a cellar, we're done. Return.
    INC MenuState
    LDA CurLevel
    BEQ @Exit
    LDA GameMode
    CMP #$09
    BEQ @Exit

    ; Calculate the X coordinate of the submenu position marker.
    ;
    ; First, mask off the high nibble of the room ID and multiply by
    ; the width of a tile, 8. Store the result in [00].
    LDA RoomId
    AND #$0F
    ASL
    ASL
    ASL
    STA $00

    ; If the submenu map's rotation >= 8, it's the same as a
    ; negative or left rotation by ($10 - rotation value).
    ;
    ; Subtract the two values as shown. Multiply the result by 8,
    ; the width of a tile. Then negate it. The final result is the
    ; negative offset.
    LDA LevelInfo_SubmenuMapRotation
    CMP #$08
    BCC @ShortRotation
    LDA #$10
    SBC LevelInfo_SubmenuMapRotation
    ASL
    ASL
    ASL
    JSR Negate
    JMP @SumMarkerX

@ShortRotation:
    ; The submenu map's rotation < 8.
    ; It represents the number of tiles to move right.
    ; So, multiply it by 8.
    ASL
    ASL
    ASL

@SumMarkerX:
    ; Add the offset we calculated, and $62 to [00] to get
    ; the position marker's X coordinate.
    CLC
    ADC $00
    CLC
    ADC #$62
    STA Sprites+83

    ; Mask off the low nibble of room ID to get a multiple of $10.
    ; Divide by 2 to get a multiple of the tile height.
    ; Then add $69 to get the Y coordinate of the sprite.
    LDA RoomId
    AND #$F0
    LSR
    ADC #$69
    STA Sprites+80

    ; Write tile $3E (dot) and attributes 0 (Link palette row 4).
    LDA #$3E
    STA Sprites+81
    LDA #$00
    STA Sprites+82

@Exit:
    RTS

UpdateMenuActive:
    JSR DrawSubmenuItems
    JSR UpdateSubmenuSelection

    ; If buttons Up and A are down on the second controller, then
    ; 1. reset submenu state
    ; 2. go to mode 8
    ; 3. silence sound
    LDA ButtonsDown+1
    AND #$88
    CMP #$88
    BNE :+
    JSR EndGameMode
    STA MenuState
    LDA #$08
    STA GameMode
    LDA #$00
    STA SongEnvelopeSelector
    JMP SilenceSound
:
    ; If Start was pressed, then hide all sprites except
    ; Link and triforce position markers, and go to the next state (scroll up).
    LDA ButtonsPressed
    AND #$10
    BEQ Submenu_Return
    LDA Sprites+84              ; Save Link's position marker's Y.
    PHA
    LDA Sprites+88              ; Save triforce's position marker's Y.
    PHA
    JSR HideAllSprites
    PLA                         ; Restore triforce's position marker's Y.
    STA Sprites+88
    PLA                         ; Restore Link's position marker's Y.
    STA Sprites+84
    INC MenuState
    RTS

UpdateMenuScrollUp:
    ; Move position marker's and vertical scroll up 3 pixels.
    LDA #$FD
    JSR MovePositionMarkers
    LDA CurVScroll
    CLC
    ADC #$03
    STA CurVScroll

    ; If hardware vertical scroll still < $F0, then return.
    CMP #$F0
    BCC Submenu_Return

    ; Once vertical scroll >= $F0, switch to NT0.
    STA SwitchNameTablesReq

    ; In UW, clear the first $10 sprites. This clears the submenu cursor.
    ;
    ; UNKNOWN: But why not do this in OW, too?
    LDA CurLevel
    BEQ :+
    JSR WriteBlankPrioritySprites
:
    ; We reached the end. So, reset hardware vertical scroll
    ; and submenu state.
    LDA #$00
    STA CurVScroll
    STA MenuState

    ; Move the position markers 2 pixels down, because we
    ; overshot and reached vertical scroll $F2.
    LDA #$02

; Params:
; A: vertical velocity of menu scrolling
;
MovePositionMarkers:
    STA $00                     ; [00] holds the vertical velocity of menu scrolling.

    ; If Link's position marker is visible, then move it by the velocity in [00].
    LDA Sprites+84
    CMP #$F8
    BEQ :+
    CLC
    ADC $00
    STA Sprites+84
:
    ; If in UW and have the compass, then move the triforce position marker.
    LDA CurLevel
    BEQ Submenu_Return
    JSR HasCompass
    BEQ Submenu_Return
    LDA Sprites+88
    CLC
    ADC $00
    STA Sprites+88

Submenu_Return:
    RTS

TriforceTransferBufOffsets:
    .BYTE $00, $08, $09, $01, $0A, $0B, $12, $1E
    .BYTE $1F, $17, $24, $25, $13, $14, $21, $13
    .BYTE $20, $21, $15, $16, $22, $16, $22, $23

TriforceTriforceBufReplacements:
    .BYTE $E7, $E7, $F5, $E8, $F5, $E8, $E7, $E7
    .BYTE $F5, $E8, $F5, $E8, $E5, $F5, $E5, $E8
    .BYTE $F5, $E8, $F5, $E6, $E6, $E7, $E7, $F5

TriforceTransferBufTiles:
    .BYTE $E9, $E9, $24, $EA, $24, $EA, $E9, $E9
    .BYTE $24, $EA, $24, $EA, $24, $24, $24, $24
    .BYTE $24, $24, $24, $24, $24, $24, $24, $24

UpdateMenuStartOW:
    ; Put the tiles into the submenu triforce transfer buffers that
    ; make it look completely empty, and no piece has been won.
    LDY #$17
:
    LDX TriforceTransferBufOffsets, Y
    LDA TriforceTransferBufTiles, Y
    STA TriforceRow0TransferBuf+4, X
    DEY
    BPL :-

    ; Reset triforce tile slot. These range from 0 to $17.
    INY

    ; Outer loop.
    ; A level test bit in [06] will be used to see whether
    ; we have a triforce piece.
    LDA #$01
    STA $06                     ; [06] level test bit

@LoopPiece:
    ; Inner loop.
    ; Each level bit will make 3 triforce tile slots be
    ; checked and replaced.
    LDA #$03
    STA $07                     ; [07] inner loop counter

@LoopTile:
    ; Get the offset into the submenu triforce transfer buffers of
    ; the current tile slot.
    LDX TriforceTransferBufOffsets, Y

    ; If the level test bit is not in the triforce mask, then we don't
    ; have this piece. So, go loop again, and leave the tile in the
    ; empty state.
    LDA $06                     ; [06] level test mask
    BIT InvTriforce
    BEQ @NextLoopTile

    ; We have the triforce piece.
    ;
    ; If the base/empty tile already in the transfer buf = $E5 or $E6,
    ; then make it $F5. These are tiles that were half full, and now
    ; need to be full.
    ;
    ; Else replace it with the one at the current triforce tile slot
    ; in the replacement list.
    LDA TriforceRow0TransferBuf+4, X
    CMP #$E5
    BEQ @ReplaceWithF5
    CMP #$E6
    BEQ @ReplaceWithF5
    LDA TriforceTriforceBufReplacements, Y
    JMP :+

@ReplaceWithF5:
    LDA #$F5
:
    STA TriforceRow0TransferBuf+4, X

@NextLoopTile:
    ; Bottom of the inner loop.
    ;
    ; Increment the triforce tile slot.
    ; Decrement [07] and loop again, if it's not 0.
    INY
    DEC $07                     ; [07] inner loop counter
    BNE @LoopTile

    ; Bottom of the outer loop.
    ;
    ; Shift the level test bit [06] left to test the next level.
    ; When it becomes 0, we're done.
    ASL $06
    BNE @LoopPiece
    INC MenuState
    RTS
