; ============================================================================
; FILE:     bank5/submenu-map.asm        BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 7567-7750
; SUBJECTS: Submenu
; TOPIC:    Submenu dungeon map, compass and map tests
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0016 MapRowMasks                             data
;   L0019 Submenu_WriteSheetMapRowTransferRecord  code
;   L0114 HasCompass                              code  Returns:
;   L0121 HasMap                                  code  Returns:
;   L0145 Submenu_WriteScanningMapRoomMark        code  Params:
; ============================================================================

MapRowMasks:
    .BYTE $80, $40, $20, $10, $08, $04, $02, $01

Submenu_WriteSheetMapRowTransferRecord:
    LDY #$10                    ; The row is $10 tiles and bytes long.

    ; Get the submenu row by dividing (current menu scrolling value) by 2.
    LDA SubmenuScrollProgress
    LSR
    TAX

    ; Write an end marker at the end of the row data.
    LDA #$FF
    STA DynTileBuf+3, Y
    LDA #$10                    ; We'll transfer $10 bytes.
    STA DynTileBuf+2

    ; Calculate PPU address ($290C + X*$20).
    LDA #$28
    STA DynTileBuf
    LDA #$EC

@Add20H:
    CLC
    ADC #$20
    BCC :+
    INC DynTileBuf
:
    DEX
    BPL @Add20H
    STA DynTileBuf+1

    ; Write a map mark in the dynamic transfer buf for each room
    ; in the range currently being scanned.
    LDA SubmenuScanRoomId
    PHA                         ; Save the current scanned room index before writing any marks.
:
    JSR Submenu_WriteScanningMapRoomMark    ; Y register is $10.
    DEC SubmenuScanRoomId
    DEY
    BNE :-

    ; Restore the current scanned room index and subtract $10.
    PLA
    SEC
    SBC #$10
    STA SubmenuScanRoomId

    ; If the level info indicates that the submenu map should be
    ; rotated horizontally, then loop to rotate right the number of
    ; bytes indicated.
    LDX LevelInfo_SubmenuMapRotation

@Rotate:
    BEQ @DoneRotate
    LDA DynTileBuf+18
    PHA
    LDY #$0E
:
    LDA DynTileBuf+3, Y
    STA DynTileBuf+4, Y
    DEY
    BPL :-
    PLA
    STA DynTileBuf+3
    DEX
    JMP @Rotate

@DoneRotate:
    ; Exclude rooms that are not part of this level, and certain rooms
    ; that should never be marked visited in the submenu map.
    ;
    ; First calculate the map row:
    ; ((current menu scrolling value) / 2) - $D
    LDA SubmenuScrollProgress
    SEC
    SBC #$1A
    LSR
    TAX
    LDY #$0F

@MaskRooms:
    ; For each element ($10) in the row, starting from $F:
    ; If the current row is set in the submenu map mask,
    ; then replace the tile with $F5, a blank map tile.
    LDA LevelInfo_SubmenuMapMask, Y
    AND MapRowMasks, X
    BNE :+
    LDA #$F5
    STA DynTileBuf+3, Y
:
    DEY
    BPL @MaskRooms
    RTS

; Returns:
; A: 0 if compass of current level is missing.
;
HasCompass:
    LDX #$10                    ; Check compasses.
    BNE :+

; Returns:
; A: 0 if map of current level is missing.
;
HasMap:
    LDX #$11                    ; Check maps.
:
    LDA CurLevel
    BEQ @Exit                   ; If in OW, then return.
    SEC
    SBC #$01                    ; Base the level number on zero.
    CMP #$08
    BCC :+                      ; If in level 9,
    INX                         ; then check the level 9 variables.
    INX
:
    AND #$07                    ; Sanitize the zero-based level number.
    TAY
    LDA InvItems, X
    AND LevelMasks, Y           ; Return the item value for the current level.

@Exit:
    RTS

; Params:
; Y: offset from the third element in dynamic transfer buf
;    to write at (between 1 and $10)
;
Submenu_WriteScanningMapRoomMark:
    TYA                         ; Save dynamic transfer buf offset.
    PHA
    JSR GetRoomFlags            ; Call this to load the address of level block world flags.
    LDA RoomId                  ; Save current room ID.
    PHA

    ; Temporarily set current room ID to the room ID we're scanning,
    ; in order to look up its information.
    LDA SubmenuScanRoomId
    STA RoomId

    ; Set OpenDoorMask to $13, in case the room has not been
    ; visited. $E2 will be added, yielding $F5, which is the blank
    ; map mark tile.
    LDA #$13
    STA OpenDoorwayMask

    ; Get the currently scanned room's visit state.
    LDY RoomId
    LDA ($00), Y
    AND #$20                    ; Visit state

    ; If it's been visited, then check each of the 4 doors,
    ; and build an open door mask.
    ;
    ; Start with direction up and direction index 3.
    BEQ @WriteMapTile
    LDA #$08
    STA $02                     ; [02] holds door direction
    LDX #$03
:
    JSR FindDoorTypeByDoorBit
    JSR CalcOpenDoorwayMask
    DEX
    LSR $02
    BNE :-

@WriteMapTile:
    ; Restore current room ID and dynamic buffer offset.
    PLA
    STA RoomId
    PLA
    TAY

    ; The map marks are arranged in the same order as all the
    ; possible values of OpenDoorMask.
    ;
    ; Add $E2 for the first map mark tile, and store it in the
    ; dynamic transfer buf at the current offset.
    LDA OpenDoorwayMask
    CLC
    ADC #$E2
    STA DynTileBuf+2, Y
    RTS
