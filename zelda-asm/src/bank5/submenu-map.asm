; ============================================================================
; FILE:     bank5/submenu-map.asm        BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 7567-7750
; SUBJECTS: Inv, Submenu
; TOPIC:    The dungeon map sheet of the item submenu: builds one row of 16 room marks per call from
;           the rooms' visited flags and doors; plus the tests "does Link have this level's compass /
;           map".
; SEE ALSO: bank5/submenu-row-transfer.asm (Submenu_CueTransferRowUW calls the row writer for submenu
;           rows $0D-$14); bank5/door-flags-dark-room.asm (Door_ShiftOpenDoorwayMask); bank5/fill-walls.asm
;           (Door_FindTypeByDirBit); bank7/next-room-room-flags.asm (RoomFlags_Get, Util_BitMasks,
;           RoomFlags_MarkVisited, the compass marker); bank5/submenu-items.asm and
;           bank5/mode3-mode6-init.asm (other callers of Inv_HasCompass and Inv_HasMap).
; KEY FACTS:
;   - The map sheet shows only rooms whose room flag bit $20 (visited) is set; it does not depend on
;     owning the map (Submenu_WriteScanningMapRoomMark).
;   - A visited room's mark is tile $E2 + a 4-bit mask of its passable sides (8 up, 4 down, 2 left,
;     1 right); an unvisited room is tile $F5 (Submenu_WriteScanningMapRoomMark).
;   - A side counts as passable when its door type is 0 (open), or is 4 or more and the room flag
;     bit for that direction is set (door opened); types 1-3 never do
;     (Submenu_WriteScanningMapRoomMark through Door_ShiftOpenDoorwayMask).
;   - Each row is rotated right by LevelInfo_SubmenuMapRotation columns and then blanked wherever
;     LevelInfo_SubmenuMapMask says no room exists (Submenu_WriteSheetMapRowTransferRecord).
;   - Compasses and maps of levels 1-8 are bit (level - 1) of item slots $10 and $11; level 9 uses
;     bit 0 of slots $12 and $13. Both tests return 0 in the OW (Inv_HasCompass, Inv_HasMap).
; CONTENTS: (generated; L = line in this file)
;   L0034 Submenu_MapRowMasks                     data  TABLE: bit of LevelInfo_SubmenuMapMask for a map row; index = map row 0 (top) to 7; value = $80 >> row.
;   L0049 Submenu_WriteSheetMapRowTransferRecord  code  Builds the PPU transfer record for one row (16 room marks) of the submenu's dungeon map sheet.
;   L0143 Inv_HasCompass                          code  Tests whether Link has the compass of the current level.
;   L0153 Inv_HasMap                              code  Tests whether Link has the map of the current level.
;   L0183 Submenu_WriteScanningMapRoomMark        code  Writes the map sheet tile for room SubmenuScanRoomId into the row being built.
; ============================================================================

; TABLE: bit of LevelInfo_SubmenuMapMask for a map row; index = map row 0 (top) to 7; value = $80 >> row.
Submenu_MapRowMasks:
    .BYTE $80, $40, $20, $10, $08, $04, $02, $01

; ----------------------------------------------------------------------------
; Builds the PPU transfer record for one row (16 room marks) of the submenu's dungeon map sheet.
; IN:   SubmenuScrollProgress = submenu row * 2 (rows $0D-$14 are map rows 0-7); SubmenuScanRoomId =
;       the room of this row's rightmost column
; OUT:  DynTileBuf = record: address $290C + row * $20, 16 tiles, $FF end marker;
;       SubmenuScanRoomId reduced by $10 (the row above)
; NOTE: Rows are produced bottom to top as SubmenuScrollProgress counts down, one row per call.
; NOTE: Order of work: marks for the 16 rooms by room column; rotate the row right by
;       LevelInfo_SubmenuMapRotation; then blank (tile $F5) every column whose
;       LevelInfo_SubmenuMapMask byte lacks this row's bit. The mask is indexed by the column
;       after rotation.
; ----------------------------------------------------------------------------
Submenu_WriteSheetMapRowTransferRecord:
    LDY #$10                    ; 16 tiles in the row.

    ; X = submenu row.
    LDA SubmenuScrollProgress
    LSR
    TAX

    ; End marker after the 16 tiles, and the tile count.
    LDA #$FF
    STA DynTileBuf+3, Y
    LDA #$10
    STA DynTileBuf+2

    ; PPU address = $28EC + (row + 1) * $20 = $290C + row * $20.
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

    ; Write the 16 marks from the right end (Y = $10, room SubmenuScanRoomId)
    ; to the left end (Y = 1, 15 rooms lower).
    LDA SubmenuScanRoomId
    PHA
:
    JSR Submenu_WriteScanningMapRoomMark
    DEC SubmenuScanRoomId
    DEY
    BNE :-

    ; Next call: the row above (room id - $10).
    PLA
    SEC
    SBC #$10
    STA SubmenuScanRoomId

    ; Rotate the 16 tiles right by one column, LevelInfo_SubmenuMapRotation times:
    ; the last tile wraps around to the first column.
    LDX LevelInfo_SubmenuMapRotation

@Rotate:
    BEQ @DoneRotate
    LDA DynTileBuf+18           ; The last tile.
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
    ; Blank the columns that have no room in this row.
    ; X = map row = submenu row - $0D = (SubmenuScrollProgress - $1A) / 2.
    LDA SubmenuScrollProgress
    SEC
    SBC #$1A
    LSR
    TAX
    LDY #$0F

@MaskRooms:
    ; For columns $0F down to 0: if the column's mask byte lacks this row's bit,
    ; write the blank map tile $F5.
    LDA LevelInfo_SubmenuMapMask, Y
    AND Submenu_MapRowMasks, X
    BNE :+
    LDA #$F5
    STA DynTileBuf+3, Y
:
    DEY
    BPL @MaskRooms
    RTS

; ----------------------------------------------------------------------------
; Tests whether Link has the compass of the current level.
; OUT:  A = 0 (Z set) when in the OW or the compass is missing; nonzero when he has it. In a
;       dungeon X = the item slot read and Y = the bit number.
; NOTE: Levels 1-8: bit (level - 1) of item slot $10 (InvCompass). Level 9: bit 0 of slot $12
;       (InvCompass9). Shares its code with Inv_HasMap.
; ----------------------------------------------------------------------------
Inv_HasCompass:
    LDX #$10                    ; Item slot of the compasses.
    BNE :+

; ----------------------------------------------------------------------------
; Tests whether Link has the map of the current level.
; OUT:  A = 0 (Z set) when in the OW or the map is missing; nonzero when he has it. In a dungeon
;       X = the item slot read and Y = the bit number.
; NOTE: Levels 1-8: bit (level - 1) of item slot $11 (InvMap). Level 9: bit 0 of slot $13 (InvMap9).
; ----------------------------------------------------------------------------
Inv_HasMap:
    LDX #$11                    ; Item slot of the maps.
:
    LDA CurLevel
    BEQ @Exit                   ; OW: return 0.
    SEC
    SBC #$01                    ; Level number from 0.
    CMP #$08
    BCC :+
    INX                         ; Level 9: use the level 9 slot, two slots further on.
    INX
:
    AND #$07                    ; Bit number: 0-7 for levels 1-8, 0 for level 9.
    TAY
    LDA InvItems, X
    AND Util_BitMasks, Y

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Writes the map sheet tile for room SubmenuScanRoomId into the row being built.
; IN:   Y = column position 1-$10 (the tile goes to DynTileBuf+2+Y); SubmenuScanRoomId = room
; OUT:  Y and RoomId preserved; OpenDoorwayMask = the room's passable sides, or $13 if unvisited
; NOTE: Unvisited (room flag bit $20 clear): tile $F5, blank. Visited: tile $E2 + mask, where the
;       mask has bit 8 up, 4 down, 2 left, 1 right set for each side that is door type 0 (open) or
;       a door of type 4 or more whose "opened" room flag bit is set (Door_ShiftOpenDoorwayMask).
; NOTE: RoomId is set to the scanned room for the duration, because the door lookups work on
;       RoomId.
; ----------------------------------------------------------------------------
Submenu_WriteScanningMapRoomMark:
    TYA                         ; Save the column position.
    PHA
    JSR RoomFlags_Get            ; For [00:01] = address of the level's room flags.
    LDA RoomId                  ; Save the real room.
    PHA

    ; Look at the scanned room instead.
    LDA SubmenuScanRoomId
    STA RoomId

    ; Default for an unvisited room: $13 + $E2 = $F5, the blank map tile.
    LDA #$13
    STA OpenDoorwayMask

    ; Visited?
    LDY RoomId
    LDA ($00), Y
    AND #$20

    ; If visited, shift one bit per side into the mask: direction bits 8, 4, 2, 1
    ; with direction index X = 3 to 0.
    BEQ @WriteMapTile
    LDA #$08
    STA $02                     ; [02] = direction bit of the side.
    LDX #$03
:
    JSR Door_FindTypeByDirBit
    JSR Door_ShiftOpenDoorwayMask
    DEX
    LSR $02
    BNE :-

@WriteMapTile:
    PLA                         ; Restore the real room and the column position.
    STA RoomId
    PLA
    TAY

    ; The 16 map mark tiles $E2-$F1 are in mask order.
    LDA OpenDoorwayMask
    CLC
    ADC #$E2
    STA DynTileBuf+2, Y
    RTS
