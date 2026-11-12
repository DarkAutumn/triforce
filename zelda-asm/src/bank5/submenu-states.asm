; ============================================================================
; FILE:     bank5/submenu-states.asm     BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 155-550
; SUBJECTS: Submenu
; TOPIC:    The item submenu's state machine (SubmenuState): scrolling the submenu in over the play area,
;           the active state, scrolling it out; the overworld Triforce display built from InvTriforce.
; SEE ALSO: bank7/mode5-play-update.asm (Mode5Play_Update: Start sets SubmenuState 1 and calls this file
;           every frame while it is nonzero); bank5/submenu-row-transfer.asm (the row transferred each
;           scroll frame); bank5/submenu-items.asm (item drawing and cursor in the active state);
;           bank5/submenu-map.asm (dungeon sheet map, Inv_HasCompass); bank1/map-marker-curtain.asm (status
;           bar position markers); bank6/level-blocks-common-data.asm (Transfer_TriforceRow0Buf).
; KEY FACTS:
;   - SubmenuState values: UW 1-5 set-up (one frame each), 6 scroll down, 7 active, 8 scroll up;
;     OW has one more set-up state, so 1-6 set-up, 7 scroll down, 8 active, 9 scroll up
;     (Submenu_UpdateUW_JumpTable, Submenu_UpdateOW_JumpTable).
;   - Opening scrolls 3 pixels per frame for 58 frames (CurVScroll $EF down to $41). Start is pressed
;     on frame 0; the active state first runs on frame 64 in UW (5 set-up frames), 65 in OW (6)
;     (Submenu_ScrollDownUW).
;   - Closing scrolls 3 pixels per frame for 59 frames (CurVScroll $41 up to $F2), then SubmenuState = 0
;     (Submenu_ScrollUp).
;   - In the active state Up+A held on controller 2 ends the mode and goes to mode 8 (the
;     continue/save/retry menu) with sound silenced (Submenu_UpdateActive).
;   - Start (controller 1, newly pressed) in the active state begins closing (Submenu_UpdateActive).
;   - The OW Triforce picture is rebuilt from InvTriforce each time the submenu opens: bit n = piece
;     of level n+1, three tiles per piece (Submenu_BuildTriforceOW, Submenu_TriforceTileOffsets).
;   - The UW submenu shows a position dot for Link's room unless GameMode is 9 (cellar); its X uses
;     LevelInfo_SubmenuMapRotation (Submenu_ScrollDownUW).
;   - The status bar Triforce marker follows the scroll only in UW with the level's compass
;     (Submenu_MovePositionMarkers).
; CONTENTS: (generated; L = line in this file)
;   L0060 Submenu_UpdateAndDrawMeters           code  Runs one frame of the submenu state machine, then updates the heart and rupee meters.
;   L0067 Submenu_Update                        code  Dispatches on SubmenuState through the UW table (CurLevel nonzero) or the OW table.
;   L0076 Submenu_UpdateUW_JumpTable            data  TABLE: UW submenu state handlers; index = SubmenuState (0-8); value = handler address.
;   L0091 Submenu_UpdateOW_JumpTable            data  TABLE: OW submenu state handlers; index = SubmenuState (0-9); value = handler address.
;   L0108 Submenu_BeginScrollDown               code  Submenu start state: hides sprites, redraws the map markers and sets up the scroll-in.
;   L0150 Submenu_Update_Return                 code  State 0 handler (submenu closed) and exit of the routine above: returns.
;   L0156 Submenu_CueAttrs1                     code  Submenu set-up state: cues the first set of submenu name table attributes for NT 2.
;   L0160 Submenu_SelectTransferBufAndIncState  code  Sets TileBufSelector = A (a static transfer buffer for this frame) and advances SubmenuState.
;   L0169 Submenu_CueAttrs2                     code  Submenu set-up state: cues the second set of submenu name table attributes for NT 2.
;   L0176 Submenu_CueBlankBottomRow             code  Submenu set-up state: cues a blank row of tiles for the bottom row of NT 2 ($2BA0).
;   L0183 Submenu_CueFirstRowUW                 code  UW state 5: cues one row transfer (first step of SubmenuScrollProgress, $2B) without scrolling.
;   L0190 Submenu_CueTriforceText               code  OW state 6: cues the transfer of the "TRIFORCE" text.
;   L0197 Submenu_ScrollDownOW                  code  OW scroll-down state: transfers one submenu row, then scrolls like the UW state below.
;   L0207 Submenu_ScrollDownUW                  code  Scroll-down state: transfers one submenu row and scrolls the submenu in by 3 pixels.
;   L0301 Submenu_UpdateActive                  code  Active state: draws the items, moves the cursor, and checks for Start or the controller 2 exit.
;   L0345 Submenu_ScrollUp                      code  Scroll-up state: scrolls the submenu out by 3 pixels a frame, then closes it (SubmenuState = 0).
;   L0386 Submenu_MovePositionMarkers           code  Moves the status bar position markers (Link's, and the Triforce's in UW) vertically by A pixels.
;   L0408 Submenu_Return                        code  Shared exit of the submenu routines above.
;   L0414 Submenu_TriforceTileOffsets           data  TABLE: where each Triforce piece's tiles go; index = (level - 1) * 3 + 0..2; value = byte offset
;   L0422 Submenu_TriforceHeldTiles             data  TABLE: tiles drawn for a Triforce piece that is held; index as Submenu_TriforceTileOffsets;
;   L0429 Submenu_TriforceEmptyTiles            data  TABLE: tiles of the empty Triforce outline; index as Submenu_TriforceTileOffsets; value = tile
;   L0440 Submenu_BuildTriforceOW               code  OW state 1: rebuilds the Triforce tiles in the Triforce row transfer buffers from InvTriforce.
; ============================================================================

; ----------------------------------------------------------------------------
; Runs one frame of the submenu state machine, then updates the heart and rupee meters.
; NOTE: Called by Mode5Play_Update every frame while SubmenuState is nonzero, instead of updating the
;       world.
; ----------------------------------------------------------------------------
Submenu_UpdateAndDrawMeters:
    JSR Submenu_Update
    JMP World_UpdateHeartsAndRupees

; ----------------------------------------------------------------------------
; Dispatches on SubmenuState through the UW table (CurLevel nonzero) or the OW table.
; ----------------------------------------------------------------------------
Submenu_Update:
    LDA SubmenuState
    LDY CurLevel
    BEQ :+                      ; In OW, go use the OW table.

    ; In UW.
    JSR Util_TableJump

; TABLE: UW submenu state handlers; index = SubmenuState (0-8); value = handler address.
Submenu_UpdateUW_JumpTable:
    .ADDR Submenu_Update_Return         ; 0 closed
    .ADDR Submenu_BeginScrollDown         ; 1 start: markers, scroll position, counters
    .ADDR Submenu_CueAttrs1         ; 2 cue attributes, part 1
    .ADDR Submenu_CueAttrs2         ; 3 cue attributes, part 2
    .ADDR Submenu_CueBlankBottomRow         ; 4 cue blank bottom row
    .ADDR Submenu_CueFirstRowUW             ; 5 cue the first submenu row
    .ADDR Submenu_ScrollDownUW    ; 6 scroll down (opening)
    .ADDR Submenu_UpdateActive          ; 7 active
    .ADDR Submenu_ScrollUp        ; 8 scroll up (closing)
:
    ; In OW.
    JSR Util_TableJump

; TABLE: OW submenu state handlers; index = SubmenuState (0-9); value = handler address.
Submenu_UpdateOW_JumpTable:
    .ADDR Submenu_Update_Return         ; 0 closed
    .ADDR Submenu_BuildTriforceOW         ; 1 build the Triforce tiles from InvTriforce
    .ADDR Submenu_BeginScrollDown         ; 2 start: markers, scroll position, counters
    .ADDR Submenu_CueAttrs1         ; 3 cue attributes, part 1
    .ADDR Submenu_CueAttrs2         ; 4 cue attributes, part 2
    .ADDR Submenu_CueBlankBottomRow         ; 5 cue blank bottom row
    .ADDR Submenu_CueTriforceText             ; 6 cue the "TRIFORCE" text
    .ADDR Submenu_ScrollDownOW    ; 7 scroll down (opening)
    .ADDR Submenu_UpdateActive          ; 8 active
    .ADDR Submenu_ScrollUp        ; 9 scroll up (closing)

; ----------------------------------------------------------------------------
; Submenu start state: hides sprites, redraws the map markers and sets up the scroll-in.
; OUT:  CurVScroll = $EF with a name table switch requested; SubmenuScrollProgress = $2B;
;       SubmenuScanRoomId = $7F; SubmenuState + 1.
; ----------------------------------------------------------------------------
Submenu_BeginScrollDown:
    JSR Sprite_HideAll
    JSR StatusBar_UpdateLinkPositionMarker
    JSR StatusBar_UpdateTriforceMarker

    ; Move the position markers and the vertical scroll position
    ; down 1 pixel.
    ;
    ; Scrolling up from the top of NT 0 shows the bottom of NT 2, so
    ; request a switch of the base name table to NT 2 and start at
    ; vertical scroll $EF (its last pixel row).
    LDA #$EF
    STA CurVScroll
    STA SwitchNameTablesReq     ; Any nonzero value requests the switch.
    LDA #$01                    ; 1 pixel
    JSR Submenu_MovePositionMarkers
    INC SubmenuState

    ; SubmenuScrollProgress begins at $2B. Each frame of the scroll-down
    ; states decrements it. It encodes a submenu row index in bits 1 to 7,
    ; and a flag in bit 0.
    ;
    ; When the flag is 1, a full row of black tiles is transferred
    ; at the current row. Otherwise, one of various static visual
    ; elements is transferred.
    ;
    ; For example:
    ; 1. In the first frame, $2B means a full row of black tiles
    ;    goes to row $15.
    ; 2. In the second frame ($2A), the row is again $15, but that
    ;    row's own content is transferred.
    ; 3. In the third frame ($29), a full row of black tiles goes
    ;    to row $14.
    LDA #$2B
    STA SubmenuScrollProgress

    ; In UW, this variable is used to scan every room in order
    ; to build the sheet map in the submenu. It runs from $7F to 0.
    LDA #$7F
    STA SubmenuScanRoomId

; State 0 handler (submenu closed) and exit of the routine above: returns.
Submenu_Update_Return:
    RTS

; ----------------------------------------------------------------------------
; Submenu set-up state: cues the first set of submenu name table attributes for NT 2.
; ----------------------------------------------------------------------------
Submenu_CueAttrs1:
    LDA #$48                    ; Selector of Transfer_SubmenuAttrs1Buf.

; Sets TileBufSelector = A (a static transfer buffer for this frame) and advances SubmenuState.
Submenu_SelectTransferBufAndIncState:
    STA TileBufSelector
:
    INC SubmenuState
    RTS

; ----------------------------------------------------------------------------
; Submenu set-up state: cues the second set of submenu name table attributes for NT 2.
; ----------------------------------------------------------------------------
Submenu_CueAttrs2:
    LDA #$4A                    ; Selector of Transfer_SubmenuAttrs2Buf.
    BNE Submenu_SelectTransferBufAndIncState    ; Always taken.

; ----------------------------------------------------------------------------
; Submenu set-up state: cues a blank row of tiles for the bottom row of NT 2 ($2BA0).
; ----------------------------------------------------------------------------
Submenu_CueBlankBottomRow:
    LDA #$4C                    ; Selector of Transfer_BlankBottomRowNT2Buf.
    BNE Submenu_SelectTransferBufAndIncState    ; Always taken.

; ----------------------------------------------------------------------------
; UW state 5: cues one row transfer (first step of SubmenuScrollProgress, $2B) without scrolling.
; ----------------------------------------------------------------------------
Submenu_CueFirstRowUW:
    JSR Submenu_CueTransferRowUW
    JMP :-                      ; Go advance SubmenuState and return.

; ----------------------------------------------------------------------------
; OW state 6: cues the transfer of the "TRIFORCE" text.
; ----------------------------------------------------------------------------
Submenu_CueTriforceText:
    LDA #$5C                    ; Selector of Transfer_TriforceTextBuf.
    BNE Submenu_SelectTransferBufAndIncState    ; Always taken.

; ----------------------------------------------------------------------------
; OW scroll-down state: transfers one submenu row, then scrolls like the UW state below.
; ----------------------------------------------------------------------------
Submenu_ScrollDownOW:
    JSR Submenu_CueTransferRowOW
    JMP :+

; ----------------------------------------------------------------------------
; Scroll-down state: transfers one submenu row and scrolls the submenu in by 3 pixels.
; OUT:  CurVScroll - 3; at $41 SubmenuState + 1 (active) and, in UW outside mode 9, sprite record
;       $14 (Sprites+80) becomes the dot marking Link's room on the submenu map.
; NOTE: Lasts 58 frames ($EF down to $41 in steps of 3); the active state runs from the next frame.
; ----------------------------------------------------------------------------
Submenu_ScrollDownUW:
    JSR Submenu_CueTransferRowUW
:
    ; Move the position markers down 3 pixels and move the
    ; vertical scroll position up by the same amount.
    LDA #$03
    JSR Submenu_MovePositionMarkers
    LDA CurVScroll
    SEC
    SBC #$03
    STA CurVScroll

    ; Nothing else to do until the vertical scroll reaches $41.
    CMP #$41
    BNE @Exit

    ; Vertical scroll reached $41. The submenu is fully shown.
    ; Advance the submenu state to active.
    ; If in OW or in mode 9 (cellar), we're done. Return.
    INC SubmenuState
    LDA CurLevel
    BEQ @Exit
    LDA GameMode
    CMP #$09
    BEQ @Exit

    ; Calculate the X coordinate of the submenu position marker.
    ;
    ; First, take the column of the room ID (low nibble) and multiply by
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
    SBC LevelInfo_SubmenuMapRotation    ; Carry is set by the CMP above.
    ASL
    ASL
    ASL
    JSR Math_Negate
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

    ; Take the row of the room ID (high nibble, a multiple of $10).
    ; Divide by 2 to get a multiple of the tile height.
    ; Then add $69 to get the Y coordinate of the sprite.
    LDA RoomId
    AND #$F0
    LSR                         ; Leaves carry clear.
    ADC #$69
    STA Sprites+80

    ; Write tile $3E (dot) and attributes 0 (sprite palette 0, in front of the background).
    LDA #$3E
    STA Sprites+81
    LDA #$00
    STA Sprites+82

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Active state: draws the items, moves the cursor, and checks for Start or the controller 2 exit.
; OUT:  Start: SubmenuState + 1 (scroll up). Up+A on controller 2: SubmenuState = 0, GameMode = 8.
; NOTE: Up+A held on controller 2 leaves play for mode 8 (continue/save/retry menu) without dying.
; ----------------------------------------------------------------------------
Submenu_UpdateActive:
    JSR Submenu_DrawItems
    JSR Submenu_UpdateSelection

    ; If buttons Up ($08) and A ($80) are both held on the second controller, then
    ; 1. end the mode (IsUpdatingMode and GameSubmode = 0) and reset the submenu state
    ; 2. go to mode 8
    ; 3. silence sound
    LDA ButtonsDown+1
    AND #$88
    CMP #$88
    BNE @CheckStart
    JSR Mode_End             ; Returns A = 0.
    STA SubmenuState
    LDA #$08
    STA GameMode
    LDA #$00
    STA SongEnvelopeSelector
    JMP Audio_SilenceSongTune1AndSample

@CheckStart:
    ; If Start was pressed on the first controller, then hide all sprites except
    ; Link's and the Triforce's position markers, and go to the next state (scroll up).
    LDA ButtonsPressed
    AND #$10
    BEQ Submenu_Return
    LDA Sprites+84              ; Save Link's position marker's Y.
    PHA
    LDA Sprites+88              ; Save the Triforce position marker's Y.
    PHA
    JSR Sprite_HideAll
    PLA                         ; Restore the Triforce position marker's Y.
    STA Sprites+88
    PLA                         ; Restore Link's position marker's Y.
    STA Sprites+84
    INC SubmenuState
    RTS

; ----------------------------------------------------------------------------
; Scroll-up state: scrolls the submenu out by 3 pixels a frame, then closes it (SubmenuState = 0).
; OUT:  CurVScroll + 3; once it is >= $F0: name table switch requested, CurVScroll = 0,
;       SubmenuState = 0.
; NOTE: Lasts 59 frames ($41 up to $F2 in steps of 3); play resumes on the next frame.
; ----------------------------------------------------------------------------
Submenu_ScrollUp:
    ; Move the position markers up 3 pixels ($FD = -3) and the vertical scroll down 3 pixels.
    LDA #$FD
    JSR Submenu_MovePositionMarkers
    LDA CurVScroll
    CLC
    ADC #$03
    STA CurVScroll

    ; If the vertical scroll is still < $F0, then return.
    CMP #$F0
    BCC Submenu_Return

    ; Vertical scroll >= $F0 ($F2 in practice): request the switch back to NT 0.
    STA SwitchNameTablesReq

    ; In UW, fill the first $10 sprite records with the blank priority sprites
    ; (Sprite_WriteBlankPrioritySprites): the dungeon doorway masks, which Sprite_HideAll
    ; removed when the submenu opened. The overworld path never writes these masks.
    LDA CurLevel
    BEQ :+
    JSR Sprite_WriteBlankPrioritySprites
:
    ; We reached the end. So, reset the vertical scroll
    ; and submenu state.
    LDA #$00
    STA CurVScroll
    STA SubmenuState

    ; Move the position markers 2 pixels down, because the scroll
    ; overshot to $F2: the markers moved down 1 + 58 * 3 = 175 pixels
    ; while opening and up 59 * 3 = 177 while closing.
    LDA #$02

; ----------------------------------------------------------------------------
; Moves the status bar position markers (Link's, and the Triforce's in UW) vertically by A pixels.
; IN:   A = signed pixels to add to the markers' Y (positive = down).
; OUT:  [00] = A; Sprites+84 (Link's marker Y) and Sprites+88 (Triforce marker Y) changed.
; NOTE: Link's marker is left alone when hidden (Y = $F8). The Triforce marker moves only in UW
;       when the level's compass is held.
; ----------------------------------------------------------------------------
Submenu_MovePositionMarkers:
    STA $00                     ; [00] holds the signed vertical step.

    ; If Link's position marker is visible, then move it by the step in [00].
    LDA Sprites+84
    CMP #$F8
    BEQ :+
    CLC
    ADC $00
    STA Sprites+84
:
    ; If in UW and have the compass, then move the Triforce position marker.
    LDA CurLevel
    BEQ Submenu_Return
    JSR Inv_HasCompass
    BEQ Submenu_Return
    LDA Sprites+88
    CLC
    ADC $00
    STA Sprites+88

; Shared exit of the submenu routines above.
Submenu_Return:
    RTS

; TABLE: where each Triforce piece's tiles go; index = (level - 1) * 3 + 0..2; value = byte offset
; from Transfer_TriforceRow0Buf+4 into the four consecutive Triforce row transfer buffers.
; Offsets $13, $21 (levels 5 and 6) and $16, $22 (levels 7 and 8) are shared by two pieces.
Submenu_TriforceTileOffsets:
    .BYTE $00, $08, $09, $01, $0A, $0B, $12, $1E    ; level 1, level 2, level 3 (first two)
    .BYTE $1F, $17, $24, $25, $13, $14, $21, $13    ; level 3 (last), level 4, level 5, level 6 (first)
    .BYTE $20, $21, $15, $16, $22, $16, $22, $23    ; level 6 (last two), level 7, level 8

; TABLE: tiles drawn for a Triforce piece that is held; index as Submenu_TriforceTileOffsets;
; value = tile. Where the slot already holds half tile $E5 or $E6 (the neighbouring piece),
; $F5 (full) is written instead of this value.
Submenu_TriforceHeldTiles:
    .BYTE $E7, $E7, $F5, $E8, $F5, $E8, $E7, $E7    ; level 1, level 2, level 3 (first two)
    .BYTE $F5, $E8, $F5, $E8, $E5, $F5, $E5, $E8    ; level 3 (last), level 4, level 5, level 6 (first)
    .BYTE $F5, $E8, $F5, $E6, $E6, $E7, $E7, $F5    ; level 6 (last two), level 7, level 8

; TABLE: tiles of the empty Triforce outline; index as Submenu_TriforceTileOffsets; value = tile
; ($E9/$EA outline edges, $24 blank).
Submenu_TriforceEmptyTiles:
    .BYTE $E9, $E9, $24, $EA, $24, $EA, $E9, $E9    ; level 1, level 2, level 3 (first two)
    .BYTE $24, $EA, $24, $EA, $24, $24, $24, $24    ; level 3 (last), level 4, level 5, level 6 (first)
    .BYTE $24, $24, $24, $24, $24, $24, $24, $24    ; level 6 (last two), level 7, level 8

; ----------------------------------------------------------------------------
; OW state 1: rebuilds the Triforce tiles in the Triforce row transfer buffers from InvTriforce.
; OUT:  the buffers at Transfer_TriforceRow0Buf+4.. hold the outline plus every held piece;
;       SubmenuState + 1. Uses [06] (level bit) and [07] (tile counter).
; NOTE: Bit n of InvTriforce is the piece of level n+1 (levels 1-8); each piece is three tiles.
; ----------------------------------------------------------------------------
Submenu_BuildTriforceOW:
    ; Put the tiles into the submenu Triforce transfer buffers that
    ; make it look completely empty, as if no piece has been won.
    LDY #$17
:
    LDX Submenu_TriforceTileOffsets, Y
    LDA Submenu_TriforceEmptyTiles, Y
    STA Transfer_TriforceRow0Buf+4, X
    DEY
    BPL :-

    ; Y becomes 0: the Triforce tile slot, which ranges from 0 to $17.
    INY

    ; Outer loop.
    ; A level test bit in [06] is used to see whether
    ; we have a Triforce piece.
    LDA #$01
    STA $06                     ; [06] level test bit

@LoopPiece:
    ; Inner loop.
    ; Each level bit makes 3 Triforce tile slots be
    ; checked and replaced.
    LDA #$03
    STA $07                     ; [07] inner loop counter

@LoopTile:
    ; Get the offset into the submenu Triforce transfer buffers of
    ; the current tile slot.
    LDX Submenu_TriforceTileOffsets, Y

    ; If the level test bit is not in InvTriforce, then we don't
    ; have this piece. So, go loop again, and leave the tile in the
    ; empty state.
    LDA $06                     ; [06] level test bit
    BIT InvTriforce
    BEQ @NextLoopTile

    ; We have the Triforce piece.
    ;
    ; If the tile already in the transfer buf = $E5 or $E6,
    ; then make it $F5. These are tiles that an earlier piece left
    ; half full, and that now need to be full.
    ;
    ; Else replace it with the one at the current Triforce tile slot
    ; in the replacement list.
    LDA Transfer_TriforceRow0Buf+4, X
    CMP #$E5
    BEQ @ReplaceWithF5
    CMP #$E6
    BEQ @ReplaceWithF5
    LDA Submenu_TriforceHeldTiles, Y
    JMP :+

@ReplaceWithF5:
    LDA #$F5
:
    STA Transfer_TriforceRow0Buf+4, X

@NextLoopTile:
    ; Bottom of the inner loop.
    ;
    ; Increment the Triforce tile slot.
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
    INC SubmenuState
    RTS
