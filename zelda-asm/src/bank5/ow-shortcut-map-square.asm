; ============================================================================
; FILE:     bank5/ow-shortcut-map-square.asm BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 6079-6371
; SUBJECTS: Mode, ModeALeaveCellar, Secret, Tile, Transfer
; TOPIC:    Overworld stairs restored when a room is laid out (rooms whose rock or gravestone was
;           pushed, rooms reached by shortcut stairs); rewriting one 16x16 square of the play area
;           tile map under a tile object; the row-by-row copy of the tile map to the screen; filler.
; SEE ALSO: bank5/layout-room-ow.asm (RoomLayout_LayOutRoomOW, RoomLayout_WriteSquareOW, RoomLayout_OWSquareTiles,
;           RoomLayout_OWSecondarySquareTiles, RoomLayout_CheckTileObj); bank5/mode12-lay-out-room.asm (RoomLayout_LayOutRoom, which
;           calls Secret_RestoreStairsOW, and Transfer_CopyRowToTileBuf); bank7/tile-obj-tiles.asm (Tile_ChangeTileObjTiles,
;           the caller of Tile_ChangePlayAreaSquare); bank4/rock-wall-tree-secrets.asm and
;           bank5/cave-cellar-exits.asm (the two places that set overworld room flag bit 5);
;           bank7/next-room-room-flags.asm (RoomFlags_Get, RoomFlags_MarkVisited).
; KEY FACTS:
;   - Overworld room flag bit 5 ($20) means "put the stairs back when this room is laid out". It is
;     set by RoomFlags_MarkVisited for the room where a rock or gravestone finished its push, and for the
;     room Link reaches by shortcut stairs. Secret_RestoreStairsOW is its only reader in the overworld.
;   - Secret_RestoreStairsOW does nothing when flag bit 7 (secret found) is set, and nothing when the tile at
;     the stairs spot is a gravestone ($BC). Any other tile there except rock wall $D8 is replaced
;     by the stairs square (tiles $70-$73). Secret_RestoreStairsOW
;   - The stairs spot is the LevelInfo_ShortcutOrItemPosArray entry picked by the room's attribute
;     byte F bits 5-4 (Room_GetShortcutOrItemXY). Secret_RestoreStairsOW
;   - Tile_ChangePlayAreaSquare changes only the tile map in RAM (what collision tests read); its
;     caller cues the PPU transfer. It serves dungeon tile objects too (push block, stairs).
;   - Square rule: a tile $27-$F2 stands for the four consecutive tiles T..T+3; other tiles are
;     looked up in RoomLayout_OWSquareTiles and drawn from RoomLayout_OWSecondarySquareTiles. Tile_ChangePlayAreaSquare
;   - The play area tile map is PlayAreaTiles at $6530: 32 columns of 22 tiles, column by column.
;     Tile_FetchPlayAreaTilesAddr
;   - Redrawing the play area takes 22 frames: one 32-tile row per frame.
;     Transfer_CopyNextPlayAreaRow, Mode_CopyNextRowAndAdvanceSubmodeWhenDone
;   - The file ends with 982 bytes of $FF filler ($AC2A-$AFFF).
; CONTENTS: (generated; L = line in this file)
;   L0054 Secret_RestoreStairsOW                     code  Overworld: after a room is laid out, puts its stairs square back if room flag bit 5 says so.
;   L0129 Tile_ChangePlayAreaSquare                  code  Rewrites the 16x16 square of the play area tile map under an object, from the tile in [05].
;   L0196 Tile_FetchPlayAreaTilesAddr                code  Points [00:01] at the start of the play area tile map, PlayAreaTiles ($6530).
;   L0210 Mode_CopyNextRowAndAdvanceSubmodeWhenDone  code  Submode step: copies one tile row to the transfer buffer; after the last row, next submode.
;   L0220 Transfer_CopyNextPlayAreaRow               code  Copies tile row CurRow of the play area to the dynamic transfer buffer, then increments CurRow.
;   L0232 ModeALeaveCellar_Init_Sub4_LayOutRoom      code  Submode step: lays out the room's tile map, sets CurRow to 0 and goes to the next submode.
; ============================================================================

; ----------------------------------------------------------------------------
; Overworld: after a room is laid out, puts its stairs square back if room flag bit 5 says so.
; IN:   RoomId; PlayAreaTiles already laid out; RoomTileObjType as found by the layout
; OUT:  one square of PlayAreaTiles rewritten, or nothing. Uses [00:01], [0D], A, X, Y.
; NOTE: Acts only when room flag bit 7 (secret found) is clear and bit 5 ($20) is set. Bit 5 is set
;       by RoomFlags_MarkVisited: for the room where a rock or gravestone was pushed its full distance
;       (bank4/rock-wall-tree-secrets.asm) and for the room Link arrives in by shortcut stairs
;       (ModeALeaveCellar_GoFromCellar, bank5/cave-cellar-exits.asm).
; NOTE: The spot comes from Room_GetShortcutOrItemXY. What is written depends on the tile already there:
;       gravestone $BC: nothing. Rock wall $D8: stairs if the room's tile object is a rock ($62),
;       else square $0C and the tile object is removed. Any other tile (tree $C4, rock $C8, ...):
;       stairs, tiles $70-$73.
; ----------------------------------------------------------------------------
Secret_RestoreStairsOW:
    JSR RoomFlags_Get            ; A = flags of this room, Y = RoomId, [00:01] = flags address.
    ASL
    BCS @Exit                   ; Bit 7 set: the room's secret was already found. Return.
    LDA ($00), Y
    AND #$20
    BEQ @Exit                   ; Bit 5 clear: no stairs to put back. Return.
    JSR Tile_FetchPlayAreaTilesAddr        ; [00:01] = $6530. Not used: [00:01] is reloaded below.
    JSR Room_GetShortcutOrItemXY     ; A = X of the stairs spot, Y = its Y (multiples of $10).

    ; X / 4 = (X / 8) * 2: the tile column times the size of an address, an offset into
    ; Tile_PlayAreaColumnAddrs.
    LSR
    LSR
    TAX
    LDA Tile_PlayAreaColumnAddrs, X  ; [00:01] = address of the tile column at the spot's X.
    STA $00
    LDA Tile_PlayAreaColumnAddrs+1, X
    STA $01
    TYA                         ; Y coordinate - $40 (the status bar height) ...
    SEC
    SBC #$40
    LSR                         ; ... / 8 = tile row within the column.
    LSR
    LSR
    TAY                         ; Y = tile row, the offset used by RoomLayout_WriteSquareOW.
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
    ; square $0C (RoomLayout_OWSecondarySquareTiles: two $F3 tiles above two $24 tiles). The level
    ; data never reaches this branch, in either quest: flag bit 5 can be set only in the rooms with a
    ; rock or gravestone object ($09, $11, $1B, $1D, $20, $21, $23, $49, $79) and in the four
    ; shortcut rooms ($1D, $23, $49, $79, LevelInfo_CellarRoomIdArray); of these only $1D has a rock
    ; wall tile ($D8) at its stairs spot ($50,$70), and its tile object is a rock.
    LDA #$00
    STA RoomTileObjType
    LDA #$0C
    STA $0D                     ; [0D] = square index $0C (< $10: tiles come from RoomLayout_OWSecondarySquareTiles).

@Write:
    JSR RoomLayout_WriteSquareOW

@Exit:
    RTS

@WriteStairs:
    LDA #$10                    ; [0D] = $10: any index >= $10 makes RoomLayout_WriteSquareOW write A..A+3.
    STA $0D
    LDA #$70                    ; First stairs tile; the square is $70, $71, $72, $73.
    BNE @Write                  ; Always taken.

; ----------------------------------------------------------------------------
; Rewrites the 16x16 square of the play area tile map under an object, from the tile in [05].
; IN:   X = object slot (ObjX and ObjY are rounded down to multiples of $10); [05] = first tile
; OUT:  four bytes of PlayAreaTiles written; X preserved. Uses [00:01], [0D], A, Y.
; NOTE: Only the tile map in RAM changes (the map that tile collision reads). The caller,
;       Tile_ChangeTileObjTiles, cues the matching PPU transfer. Despite "OW" in the name it also serves
;       dungeon rooms (push block tiles $74 and $B0, revealed stairs $70).
; NOTE: Tile $27-$F2: the square is T top left, T+1 bottom left, T+2 top right, T+3 bottom right.
;       Any other tile: RoomLayout_OWSquareTiles entries $0E down to $01 are searched for it and the index
;       found selects four tiles of RoomLayout_OWSecondarySquareTiles ($24 -> four $24, $26 -> four $26). No match
;       leaves index 0, four $24 tiles.
; NOTE: Tile_ChangeTileObjTiles builds its PPU records with other limits: four consecutive tiles only
;       for $46-$F2, four copies of the tile otherwise. For a tile $27-$45 the screen would show
;       four copies while this map holds T..T+3; the callers found pass $24, $26 and tiles >= $70.
; ----------------------------------------------------------------------------
Tile_ChangePlayAreaSquare:
    TXA                         ; Save the object slot.
    PHA

    ; Round X down to the square, then X / 4 = (X / 8) * 2: offset of the tile column's address
    ; in Tile_PlayAreaColumnAddrs.
    LDA ObjX, X
    AND #$F0
    LSR
    LSR
    TAX

    ; [00:01] = address of that tile column.
    LDA Tile_PlayAreaColumnAddrs, X
    STA $00
    LDA Tile_PlayAreaColumnAddrs+1, X
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
    JSR Math_AddToInt16At0

    ; Assume a square of four consecutive tiles starting at [05]: any square index >= $10 makes
    ; RoomLayout_WriteSquareOW write A, A+1, A+2, A+3.
    LDY #$00
    LDX #$10
    LDA $05

    ; A tile < $27 or >= $F3 is not the first of four consecutive tiles. Find it in
    ; RoomLayout_OWSquareTiles, index $0E down to 1; X then is the square index for RoomLayout_OWSecondarySquareTiles.
    ; If it is not found, X ends at 0.
    CMP #$27
    BCC :+
    CMP #$F3
    BCC @Write
:
    LDX #$0E
:
    CMP RoomLayout_OWSquareTiles, X
    BEQ @Write
    DEX
    BNE :-

@Write:
    STX $0D                     ; [0D] = square index; A = first tile; Y = 0.
    JSR RoomLayout_WriteSquareOW
    PLA                         ; Restore the object slot.
    TAX
    RTS

; ----------------------------------------------------------------------------
; Points [00:01] at the start of the play area tile map, PlayAreaTiles ($6530).
; OUT:  [00] = $30, [01] = $65; A = $65.
; NOTE: The map holds 32 columns of 22 ($16) tiles, column after column, so a tile is at
;       $6530 + column * $16 + row.
; ----------------------------------------------------------------------------
Tile_FetchPlayAreaTilesAddr:
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
Mode_CopyNextRowAndAdvanceSubmodeWhenDone:
    JSR Transfer_CopyNextPlayAreaRow
    BCS :+                      ; All rows copied: go to INC GameSubmode in ModeALeaveCellar_Init_Sub4_LayOutRoom.
    RTS

; ----------------------------------------------------------------------------
; Copies tile row CurRow of the play area to the dynamic transfer buffer, then increments CurRow.
; IN:   CurRow = row to copy (0-$15)
; OUT:  CurRow + 1; C = 1 when CurRow has reached $16 (all 22 rows copied), else C = 0.
; ----------------------------------------------------------------------------
Transfer_CopyNextPlayAreaRow:
    JSR Transfer_CopyRowToTileBuf
    INC CurRow
    LDA CurRow
    CMP #$16                    ; $16 = number of tile rows in the play area.
    RTS

; ----------------------------------------------------------------------------
; Submode step: lays out the room's tile map, sets CurRow to 0 and goes to the next submode.
; NOTE: Submode 4 of mode $A (leaving a cave or cellar): rebuilds the room Link returns to. In the
;       overworld RoomLayout_LayOutRoom ends in Secret_RestoreStairsOW.
; ----------------------------------------------------------------------------
ModeALeaveCellar_Init_Sub4_LayOutRoom:
    JSR RoomLayout_LayOutRoom
    LDA #$00
    STA CurRow                  ; The row copy of the next submode starts at row 0.
:
    INC GameSubmode
    RTS

; Filler: 982 bytes of $FF from the end of the code above ($AC2A) to $AFFF; the next routine
; (Mode3Unfurl_Init_Sub2_FillAttrsCuePalettes in bank5/mode3-mode6-init.asm) starts at $B000. No label; nothing refers to it.
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
