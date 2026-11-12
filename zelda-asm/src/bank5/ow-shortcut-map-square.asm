; ============================================================================
; FILE:     bank5/ow-shortcut-map-square.asm BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 6079-6371
; SUBJECTS: (none)
; TOPIC:    Overworld stairs restored when a room is laid out (rooms whose rock or gravestone was
;           pushed, rooms reached by shortcut stairs); rewriting one 16x16 square of the play area
;           tile map under a tile object; the row-by-row copy of the tile map to the screen; filler.
; SEE ALSO: bank5/layout-room-ow.asm (LayoutRoomOW, WriteSquareOW, PrimarySquaresOW,
;           SecondarySquaresOW, CheckTileObject); bank5/mode12-lay-out-room.asm (LayOutRoom, which
;           calls CheckShortcut, and CopyRowToTileBuf); bank7/tile-obj-tiles.asm (ChangeTileObjTiles,
;           the caller of ChangePlayMapSquareOW); bank4/rock-wall-tree-secrets.asm and
;           bank5/cave-cellar-exits.asm (the two places that set overworld room flag bit 5);
;           bank7/next-room-room-flags.asm (GetRoomFlags, MarkRoomVisited).
; KEY FACTS:
;   - Overworld room flag bit 5 ($20) means "put the stairs back when this room is laid out". It is
;     set by MarkRoomVisited for the room where a rock or gravestone finished its push, and for the
;     room Link reaches by shortcut stairs. CheckShortcut is its only reader in the overworld.
;   - CheckShortcut does nothing when flag bit 7 (secret found) is set, and nothing when the tile at
;     the stairs spot is a gravestone ($BC). Any other tile there except rock wall $D8 is replaced
;     by the stairs square (tiles $70-$73). CheckShortcut
;   - The stairs spot is the LevelInfo_ShortcutOrItemPosArray entry picked by the room's attribute
;     byte F bits 5-4 (GetShortcutOrItemXY). CheckShortcut
;   - ChangePlayMapSquareOW changes only the tile map in RAM (what collision tests read); its
;     caller cues the PPU transfer. It serves dungeon tile objects too (push block, stairs).
;   - Square rule: a tile $27-$F2 stands for the four consecutive tiles T..T+3; other tiles are
;     looked up in PrimarySquaresOW and drawn from SecondarySquaresOW. ChangePlayMapSquareOW
;   - The play area tile map is PlayAreaTiles at $6530: 32 columns of 22 tiles, column by column.
;     FetchTileMapAddr
;   - Redrawing the play area takes 22 frames: one 32-tile row per frame.
;     CopyNextRowToTransferBuf, CopyNextRowToTransferBufAndAdvanceSubmodeWhenDone
;   - The file ends with 982 bytes of $FF filler ($AC2A-$AFFF).
; CONTENTS: (generated; L = line in this file)
;   L0054 CheckShortcut                                      code  Overworld: after a room is laid out, puts its stairs square back if room flag bit 5 says so.
;   L0126 ChangePlayMapSquareOW                              code  Rewrites the 16x16 square of the play area tile map under an object, from the tile in [05].
;   L0193 FetchTileMapAddr                                   code  Points [00:01] at the start of the play area tile map, PlayAreaTiles ($6530).
;   L0207 CopyNextRowToTransferBufAndAdvanceSubmodeWhenDone  code  Submode step: copies one tile row to the transfer buffer; after the last row, next submode.
;   L0217 CopyNextRowToTransferBuf                           code  Copies tile row CurRow of the play area to the dynamic transfer buffer, then increments CurRow.
;   L0229 LayoutRoom_SubmodeTask                             code  Submode step: lays out the room's tile map, sets CurRow to 0 and goes to the next submode.
; ============================================================================

; ----------------------------------------------------------------------------
; Overworld: after a room is laid out, puts its stairs square back if room flag bit 5 says so.
; IN:   RoomId; PlayAreaTiles already laid out; RoomTileObjType as found by the layout
; OUT:  one square of PlayAreaTiles rewritten, or nothing. Uses [00:01], [0D], A, X, Y.
; NOTE: Acts only when room flag bit 7 (secret found) is clear and bit 5 ($20) is set. Bit 5 is set
;       by MarkRoomVisited: for the room where a rock or gravestone was pushed its full distance
;       (bank4/rock-wall-tree-secrets.asm) and for the room Link arrives in by shortcut stairs
;       (GoToModeAFromCellar, bank5/cave-cellar-exits.asm).
; NOTE: The spot comes from GetShortcutOrItemXY. What is written depends on the tile already there:
;       gravestone $BC: nothing. Rock wall $D8: stairs if the room's tile object is a rock ($62),
;       else square $0C and the tile object is removed. Any other tile (tree $C4, rock $C8, ...):
;       stairs, tiles $70-$73.
; ----------------------------------------------------------------------------
CheckShortcut:
    JSR GetRoomFlags            ; A = flags of this room, Y = RoomId, [00:01] = flags address.
    ASL
    BCS @Exit                   ; Bit 7 set: the room's secret was already found. Return.
    LDA ($00), Y
    AND #$20
    BEQ @Exit                   ; Bit 5 clear: no stairs to put back. Return.
    JSR FetchTileMapAddr        ; [00:01] = $6530. Not used: [00:01] is reloaded below.
    JSR GetShortcutOrItemXY     ; A = X of the stairs spot, Y = its Y (multiples of $10).

    ; X / 4 = (X / 8) * 2: the tile column times the size of an address, an offset into
    ; PlayAreaColumnAddrs.
    LSR
    LSR
    TAX
    LDA PlayAreaColumnAddrs, X  ; [00:01] = address of the tile column at the spot's X.
    STA $00
    LDA PlayAreaColumnAddrs+1, X
    STA $01
    TYA                         ; Y coordinate - $40 (the status bar height) ...
    SEC
    SBC #$40
    LSR                         ; ... / 8 = tile row within the column.
    LSR
    LSR
    TAY                         ; Y = tile row, the offset used by WriteSquareOW.
    LDA ($00), Y                ; The top left tile of the square now at the stairs spot.
    CMP #$C4
    BEQ @WriteStairs            ; Tree: write the stairs.
    CMP #$BC
    BEQ @Exit                   ; Gravestone: leave it. Return.
    CMP #$D8
    BNE @WriteStairs            ; Anything but a rock wall: write the stairs.
    LDA RoomTileObjType
    CMP #$62
    BEQ @WriteStairs            ; Rock wall here, but the room's tile object is a rock: stairs.

    ; Rock wall at the spot and the tile object is not a rock: remove the tile object and write
    ; square $0C (SecondarySquaresOW: two $F3 tiles above two $24 tiles). No room that reaches
    ; this branch was identified (unverified).
    LDA #$00
    STA RoomTileObjType
    LDA #$0C
    STA $0D                     ; [0D] = square index $0C (< $10: tiles come from SecondarySquaresOW).

@Write:
    JSR WriteSquareOW

@Exit:
    RTS

@WriteStairs:
    LDA #$10                    ; [0D] = $10: any index >= $10 makes WriteSquareOW write A..A+3.
    STA $0D
    LDA #$70                    ; First stairs tile; the square is $70, $71, $72, $73.
    BNE @Write                  ; Always taken.

; ----------------------------------------------------------------------------
; Rewrites the 16x16 square of the play area tile map under an object, from the tile in [05].
; IN:   X = object slot (ObjX and ObjY are rounded down to multiples of $10); [05] = first tile
; OUT:  four bytes of PlayAreaTiles written; X preserved. Uses [00:01], [0D], A, Y.
; NOTE: Only the tile map in RAM changes (the map that tile collision reads). The caller,
;       ChangeTileObjTiles, cues the matching PPU transfer. Despite "OW" in the name it also serves
;       dungeon rooms (push block tiles $74 and $B0, revealed stairs $70).
; NOTE: Tile $27-$F2: the square is T top left, T+1 bottom left, T+2 top right, T+3 bottom right.
;       Any other tile: PrimarySquaresOW entries $0E down to $01 are searched for it and the index
;       found selects four tiles of SecondarySquaresOW ($24 -> four $24, $26 -> four $26). No match
;       leaves index 0, four $24 tiles.
; NOTE: ChangeTileObjTiles builds its PPU records with other limits: four consecutive tiles only
;       for $46-$F2, four copies of the tile otherwise. For a tile $27-$45 the screen would show
;       four copies while this map holds T..T+3; the callers found pass $24, $26 and tiles >= $70.
; ----------------------------------------------------------------------------
ChangePlayMapSquareOW:
    TXA                         ; Save the object slot.
    PHA

    ; Round X down to the square, then X / 4 = (X / 8) * 2: offset of the tile column's address
    ; in PlayAreaColumnAddrs.
    LDA ObjX, X
    AND #$F0
    LSR
    LSR
    TAX

    ; [00:01] = address of that tile column.
    LDA PlayAreaColumnAddrs, X
    STA $00
    LDA PlayAreaColumnAddrs+1, X
    STA $01
    PLA                         ; X = object slot again (kept on the stack).
    PHA
    TAX

    ; Round Y down to the square, subtract $40 (status bar), divide by 8: the tile row.
    LDA ObjY, X
    AND #$F0
    SEC
    SBC #$40
    LSR
    LSR
    LSR

    ; [00:01] += row: now the address of the square's top left tile.
    JSR AddToInt16At0

    ; Assume a square of four consecutive tiles starting at [05]: any square index >= $10 makes
    ; WriteSquareOW write A, A+1, A+2, A+3.
    LDY #$00
    LDX #$10
    LDA $05

    ; A tile < $27 or >= $F3 is not the first of four consecutive tiles. Find it in
    ; PrimarySquaresOW, index $0E down to 1; X then is the square index for SecondarySquaresOW.
    ; If it is not found, X ends at 0.
    CMP #$27
    BCC :+
    CMP #$F3
    BCC @Write
:
    LDX #$0E
:
    CMP PrimarySquaresOW, X
    BEQ @Write
    DEX
    BNE :-

@Write:
    STX $0D                     ; [0D] = square index; A = first tile; Y = 0.
    JSR WriteSquareOW
    PLA                         ; Restore the object slot.
    TAX
    RTS

; ----------------------------------------------------------------------------
; Points [00:01] at the start of the play area tile map, PlayAreaTiles ($6530).
; OUT:  [00] = $30, [01] = $65; A = $65.
; NOTE: The map holds 32 columns of 22 ($16) tiles, column after column, so a tile is at
;       $6530 + column * $16 + row.
; ----------------------------------------------------------------------------
FetchTileMapAddr:
    LDA #$30
    STA $00
    LDA #$65
    STA $01
    RTS

; ----------------------------------------------------------------------------
; Submode step: copies one tile row to the transfer buffer; after the last row, next submode.
; IN:   CurRow = row to copy (0-$15)
; OUT:  C = 1 on the call that copied the last row (GameSubmode was incremented), else C = 0.
; NOTE: One row per frame: redrawing the whole play area takes $16 (22) frames. It is a submode of
;       the mode 9, $A, $B and $C init tables and is called by the death mode.
; ----------------------------------------------------------------------------
CopyNextRowToTransferBufAndAdvanceSubmodeWhenDone:
    JSR CopyNextRowToTransferBuf
    BCS :+                      ; All rows copied: go to INC GameSubmode in LayoutRoom_SubmodeTask.
    RTS

; ----------------------------------------------------------------------------
; Copies tile row CurRow of the play area to the dynamic transfer buffer, then increments CurRow.
; IN:   CurRow = row to copy (0-$15)
; OUT:  CurRow + 1; C = 1 when CurRow has reached $16 (all 22 rows copied), else C = 0.
; ----------------------------------------------------------------------------
CopyNextRowToTransferBuf:
    JSR CopyRowToTileBuf
    INC CurRow
    LDA CurRow
    CMP #$16                    ; $16 = number of tile rows in the play area.
    RTS

; ----------------------------------------------------------------------------
; Submode step: lays out the room's tile map, sets CurRow to 0 and goes to the next submode.
; NOTE: Submode 4 of mode $A (leaving a cave or cellar): rebuilds the room Link returns to. In the
;       overworld LayOutRoom ends in CheckShortcut.
; ----------------------------------------------------------------------------
LayoutRoom_SubmodeTask:
    JSR LayOutRoom
    LDA #$00
    STA CurRow                  ; The row copy of the next submode starts at row 0.
:
    INC GameSubmode
    RTS

; Filler: 982 bytes of $FF from the end of the code above ($AC2A) to $AFFF; the next routine
; (InitMode3_Sub2 in bank5/mode3-mode6-init.asm) starts at $B000. No label; nothing refers to it.
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF
