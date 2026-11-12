; ============================================================================
; FILE:     bank5/layout-room-ow.asm     BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 5751-6077
; SUBJECTS: Mode9Cellar, ModeBCave, ModeCShortcut, RoomLayout
; TOPIC:    Lays out an overworld room, a cave or a dungeon cellar (16 x 11 squares of 2 x 2 tiles) into
;           the play area tile map; turns a found secret into stairs or a cave mouth; finds the room's
;           tile object (rock, rock wall, tree, gravestone, armos) while laying out.
; SEE ALSO: bank5/room-layouts-ow.asm (RoomLayout_RoomsOW, cave layouts, RoomLayout_ColumnHeapOW0-F);
;           bank5/room-layouts-uw.asm (RoomLayout_CellarUW0/1, RoomLayout_ColumnHeapUWCellar);
;           bank5/mode12-lay-out-room.asm (RoomLayout_LayOutRoom, the caller of RoomLayout_LayOutRoomOW);
;           bank5/uw-floor-push-block.asm (the dungeon floor equivalent);
;           bank5/ow-shortcut-map-square.asm (Secret_RestoreStairsOW, Tile_ChangePlayAreaSquare use RoomLayout_WriteSquareOW);
;           bank5/enter-room.asm (turns RoomTileObjType/X/Y into the object in slot $0B);
;           bank5/room-history-ow.asm (mode 9/B/C init tables that call the cave and cellar entries);
;           bank4/rock-wall-tree-secrets.asm (the tile objects' behaviour).
; KEY FACTS:
;   - An overworld room layout is 16 bytes, one per square column: high nibble = column heap, low
;     nibble = column index in the heap. Layout address = RoomLayout_RoomsOW + id * 16, id =
;     LevelBlockAttrsD bits 6-0 (7 bits here, not the 6 that Room_GetUniqueRoomId returns) (RoomLayout_LayOutRoomOW).
;   - Square descriptor: bit 7 = first square of a column, bit 6 = write the square twice, bits 5-0 =
;     square index into RoomLayout_OWSquareTiles (RoomLayout_LayOutRoomOrCaveOW).
;   - Square indexes below $10 take their four tiles from RoomLayout_OWSecondarySquareTiles; $10 and up use four
;     consecutive tiles starting at the RoomLayout_OWSquareTiles tile (RoomLayout_WriteSquareOW).
;   - If the room's secret flag (room flags bit 7) is set, a tree square ($E7) or square $EA is laid out
;     as stairs (tile $70) and a rock wall square ($E6) as a cave mouth (secondary square $0C), and none
;     of them becomes a tile object (RoomLayout_LayOutRoomOrCaveOW).
;   - Squares $E5-$EA are placeholders: each records the room's tile object (types $62 rock, $63 rock
;     wall, $64 tree, $65 gravestone, $66 armos, $67 rock wall) at column * $10, row * $10 + $40 and is
;     drawn as the tile in RoomLayout_TileObjSquareTilesOW. One tile object per room: the last placeholder
;     laid out (rightmost column, lowest row) wins. No column heap holds square index $2A or $2B
;     (tiles $E9, $EA), so the layout data never produces types $66 and $67 (RoomLayout_CheckTileObj).
;   - Caves use two fixed layouts: RoomLayout_CaveOW for an ordinary cave, RoomLayout_ShortcutCaveOW for the
;     shortcut stairs room (ModeBCave_Init_LayOutCave, ModeCShortcut_Init_LayOutShortcut).
;   - Cellars are laid out by the overworld routine: layout id even ($3E) = RoomLayout_CellarUW0, the
;     tunnel with two ladders; odd ($3F) = RoomLayout_CellarUW1, the item cellar
;     (Mode9Cellar_Init_LayOutCellar). In a dungeon, entry 0 of RoomLayout_ColumnDirectoryOW (a RAM copy) points
;     at RoomLayout_ColumnHeapUWCellar (RoomLayout_PatchColumnDirectoryForCellar).
; CONTENTS: (generated; L = line in this file)
;   L0059 RoomLayout_TileObjTypes                   data  TABLE: object type of the tile object for placeholder squares $E5-$EA; index = square tile - $E5.
;   L0064 RoomLayout_TileObjSquareTilesOW           data  TABLE: first tile of the square drawn in place of placeholder squares $E5-$EA; index = square tile -
;   L0073 RoomLayout_OWSquareTiles                  data  TABLE: overworld squares; index = bits 5-0 of a square descriptor ($00-$37); value = the square's
;   L0088 RoomLayout_OWSecondarySquareTiles         data  TABLE: the four tiles of overworld squares $00-$0F; index = square index * 4 + cell; cells in the
;   L0106 RoomLayout_LayOutRoomOW                   code  Lays out the overworld room RoomId into the play area tile map.
;   L0141 RoomLayout_LayOutRoomOrCaveOW             code  Lays out 16 x 11 squares from a 16-byte layout: overworld room, cave or cellar.
;   L0267 RoomLayout_CheckTileObj                   code  Turns a tile object placeholder square ($E5-$EA) into the room's tile object and its drawn tile.
;   L0301 RoomLayout_CheckTileObj_Return            code  Shared return of RoomLayout_LayOutRoomOrCaveOW and RoomLayout_CheckTileObj.
;   L0311 RoomLayout_WriteSquareOW                  code  Writes one 2 x 2 tile overworld square into the play area tile map.
;   L0369 RoomLayout_PatchColumnDirectoryForCellar  code  Points entry 0 of RoomLayout_ColumnDirectoryOW at the overworld heap 0 or, in a dungeon, at the cellar heap.
;   L0387 RoomLayout_SubroomLayoutAddrs             data  TABLE: layout addresses of caves and cellars; index = 0 ordinary cave, 2 shortcut stairs room,
;   L0398 ModeBCave_Init_LayOutCave                 code  Mode $B init step: lays out the ordinary cave room and advances GameSubmode.
;   L0409 ModeCShortcut_Init_LayOutShortcut         code  Mode $C init step: lays out the shortcut stairs room and advances GameSubmode.
;   L0418 Mode9Cellar_Init_LayOutCellar             code  Mode 9 init step: lays out the cellar (tunnel or item cellar) and advances GameSubmode.
; ============================================================================

; TABLE: object type of the tile object for placeholder squares $E5-$EA; index = square tile - $E5.
;   $62 rock, $63 rock wall, $64 tree, $65 gravestone, $66 armos, $67 rock wall.
;   No overworld column heap holds square index $2A or $2B (tiles $E9, $EA), so types $66 and $67 are
;   never produced by the layout data (the types themselves exist in the object tables).
RoomLayout_TileObjTypes:
    .BYTE $62, $63, $64, $65, $66, $67

; TABLE: first tile of the square drawn in place of placeholder squares $E5-$EA; index = square tile -
;   $E5. $C8 rock, $D8 rock wall, $C4 tree, $BC gravestone, $C0 armos, $C0.
RoomLayout_TileObjSquareTilesOW:
    .BYTE $C8, $D8, $C4, $BC, $C0, $C0

; TABLE: overworld squares; index = bits 5-0 of a square descriptor ($00-$37); value = the square's
;   first tile. Indexes $00-$0F are drawn from RoomLayout_OWSecondarySquareTiles instead (the value here is only used
;   by Tile_ChangePlayAreaSquare to find the index of a tile). Values $E5-$EA (indexes $26-$2B) are tile
;   object placeholders, see RoomLayout_CheckTileObj. Known tiles: $24 blank (black), $6F ladder (cellars),
;   $70 stairs, $C8 rock, $BC gravestone, $C4 tree, $D8 rock wall, $C0 armos statue; the rest are
;   terrain tiles not identified here.
RoomLayout_OWSquareTiles:
    .BYTE $24, $6F, $F3, $FA, $98, $90, $8F, $95    ; $00-$07 (secondary squares)
    .BYTE $8E, $90, $74, $76, $F3, $24, $26, $89    ; $08-$0F (secondary squares; $0C = cave mouth)
    .BYTE $03, $04, $70, $C8, $BC, $8D, $8F, $93    ; $10-$17: $12 stairs, $13 rock, $14 gravestone
    .BYTE $95, $C4, $CE, $D8, $B0, $B4, $AA, $AC    ; $18-$1F: $19 tree, $1B rock wall
    .BYTE $B8, $9C, $A6, $9A, $A2, $A0, $E5, $E6    ; $20-$27: $26 rock object, $27 rock wall object
    .BYTE $E7, $E8, $E9, $EA, $C0, $E0, $78, $7A    ; $28-$2F: $28 tree object, $29 gravestone object,
                                                    ;   $2A, $2B placeholders in no column heap,
                                                    ;   $2C armos statue
    .BYTE $7E, $80, $CC, $D0, $D4, $DC, $89, $84    ; $30-$37

; TABLE: the four tiles of overworld squares $00-$0F; index = square index * 4 + cell; cells in the
;   order top-left, bottom-left, top-right, bottom-right; two squares per row below.
;   $00 blank (black), $01 ladder ($6F), $0C = $F3,$24,$F3,$24: what a rock wall becomes once its secret
;   is found (cave mouth), $0D blank.
RoomLayout_OWSecondarySquareTiles:
    .BYTE $24, $24, $24, $24, $6F, $6F, $6F, $6F
    .BYTE $F3, $F3, $F3, $F3, $FA, $FA, $FA, $FA
    .BYTE $98, $95, $26, $26, $90, $95, $90, $95
    .BYTE $8F, $90, $8F, $90, $95, $96, $95, $96
    .BYTE $8E, $93, $90, $95, $90, $95, $92, $97
    .BYTE $74, $74, $75, $75, $76, $77, $76, $77
    .BYTE $F3, $24, $F3, $24, $24, $24, $24, $24
    .BYTE $26, $26, $26, $26, $89, $88, $8B, $88

; ----------------------------------------------------------------------------
; Lays out the overworld room RoomId into the play area tile map.
; IN:   RoomId; LevelBlockAttrsD (layout id in bits 6-0).
; OUT:  falls into RoomLayout_LayOutRoomOrCaveOW with [02:03] = the room's 16 layout bytes.
; NOTE: the layout id is 7 bits here: bit 7 is shifted out, bit 6 is kept. RoomLayout_RoomsOW holds ids
;       $00-$78; ids $79-$7B fall on the three 16-byte tables after it (RoomLayout_CaveOW-2). The Q1
;       overworld uses ids up to $78.
; ----------------------------------------------------------------------------
RoomLayout_LayOutRoomOW:
    ; [02:03] = start of the overworld room layouts.
    LDA RoomLayout_RoomsOWAddr
    STA $02
    LDA RoomLayout_RoomsOWAddr+1
    STA $03
    LDA #$00                    ; [06] collects the high byte of the multiplication below.
    STA $06
    LDX RoomId
    LDA LevelBlockAttrsD, X     ; Bits 6-0: layout id.

    ; [02:03] += layout id * $10 (16 column bytes per layout). The first ASL drops bit 7; the next
    ; three shift bits 6-4 into [06].
    ASL
    ASL
    ROL $06
    ASL
    ROL $06
    ASL
    ROL $06
    ADC $02
    STA $02
    LDA $06
    ADC $03
    STA $03

; ----------------------------------------------------------------------------
; Lays out 16 x 11 squares from a 16-byte layout: overworld room, cave or cellar.
; IN:   [02:03] = address of the 16 layout bytes; RoomId; RoomLayout_ColumnDirectoryOW.
; OUT:  PlayAreaTiles fully written; RoomTileObjType/X/Y set if a placeholder square was met (not
;       cleared here otherwise). Uses [00]-[0D].
; NOTE: [08:09] = this level's room flags, [04:05] = current square descriptor, [00:01] = play area
;       cursor, [06] = column 0-15, [07] = row 0-10, [0C] = $40 while a doubled square is pending,
;       [0D] = square index. A found secret (room flags bit 7) replaces tree and rock wall squares.
; ----------------------------------------------------------------------------
RoomLayout_LayOutRoomOrCaveOW:
    LDA LevelInfo_WorldFlagsAddr
    STA $08
    LDA LevelInfo_WorldFlagsAddr+1
    STA $09
    JSR Tile_FetchPlayAreaTilesAddr

    ; For each of the 16 square columns, indexed by [06]:
    LDA #$00
    STA $0C                     ; No doubled square pending.
    STA $06

@LoopColumnOW:
    LDY $06
    LDA ($02), Y                ; Layout byte of this column.
    AND #$F0                    ; X = heap number * 2 (high nibble).
    LSR
    LSR
    LSR
    TAX

    ; [04:05] = address of that column heap.
    LDA RoomLayout_ColumnDirectoryOW, X
    STA $04
    LDA RoomLayout_ColumnDirectoryOW+1, X
    STA $05
    LDA ($02), Y                ; Layout byte again.
    AND #$0F                    ; X = index of the column inside the heap.
    TAX
    LDY #$FF
:
    ; Walk the heap, counting descriptors with bit 7 set (column starts), until column X is reached.
    INY
    LDA ($04), Y
    BPL :-                      ; Not a column start.
    DEX
    BPL :-                      ; Not yet the column wanted.

    TYA
    JSR Math_AddToInt16At4           ; [04:05] now points at the column's first square descriptor.
    LDA #$00                    ; Row = 0.
    STA $07

@LoopSquareOW:
    LDY #$00
    LDA ($04), Y                ; Square descriptor.
    AND #$3F                    ; Bits 5-0: square index, kept in [0D] and X.
    STA $0D
    TAX
    LDA RoomLayout_OWSquareTiles, X
    PHA                         ; Save the square's tile.
    LDY RoomId
    LDA ($08), Y                ; Room flags of this room.
    AND #$80                    ; Bit 7: the room's secret was found.
    BEQ @SkipSecret
    PLA                         ; The square's tile.

    ; The secret was found in this room.
    CMP #$E7
    BEQ @MakeStairs             ; Tree placeholder: stairs.
    CMP #$E6
    BEQ @MakeCave               ; Rock wall placeholder: cave mouth.
    CMP #$EA
    BNE @RestoreSquare          ; Anything but placeholder $EA stays as it is.

@MakeStairs:
    ; Tree ($E7) or placeholder $EA: stairs, tile $70, with a square index of $10 so that
    ; RoomLayout_WriteSquareOW writes four consecutive tiles.
    LDA #$10
    STA $0D
    LDA #$70
    BNE @RestoreSquare          ; Always taken.

@MakeCave:
    ; Rock wall ($E6): secondary square $0C, the cave mouth. A = $0C matches no placeholder, and
    ; RoomLayout_WriteSquareOW ignores A for a secondary square.
    LDA #$0C
    STA $0D

@RestoreSquare:
    PHA

@SkipSecret:
    PLA                         ; The square's tile (possibly replaced above).
    JSR RoomLayout_CheckTileObj         ; A placeholder $E5-$EA sets the room's tile object and its real tile.
    LDY #$00
    JSR RoomLayout_WriteSquareOW
    LDA #$02                    ; Move the cursor 2 tiles down: the next square of this column.
    JSR Math_AddToInt16At0
    LDY #$00
    LDA ($04), Y                ; Square descriptor.
    AND #$40                    ; Bit 6: write this square twice.
    BEQ @NextSquare
    EOR $0C                     ; Toggle the pending flag: $40 after the first copy (stay on this
    STA $0C                     ; descriptor), 0 after the second (go on).
    BNE :+

@NextSquare:
    JSR Math_Add1ToInt16At4          ; Point to the next square descriptor.
:
    INC $07                     ; Next row.
    LDA $07
    CMP #$0B                    ; A column is $B squares high.
    BNE @LoopSquareOW

    ; The cursor is now at the top of the next tile column (a column is $16 tiles). Skip that
    ; column too: it is the right half of the squares just written.
    LDA #$16
    JSR Math_AddToInt16At0
    INC $06                     ; Next column.
    LDA $06
    CMP #$10                    ; A room is $10 squares wide.
    BCS RoomLayout_CheckTileObj_Return             ; All columns done: return.
    JMP @LoopColumnOW

; ----------------------------------------------------------------------------
; Turns a tile object placeholder square ($E5-$EA) into the room's tile object and its drawn tile.
; IN:   A = square tile; [06] = square column 0-15; [07] = square row 0-10.
; OUT:  not a placeholder: A unchanged. Placeholder: A = tile from RoomLayout_TileObjSquareTilesOW;
;       RoomTileObjType = RoomLayout_TileObjTypes entry; RoomTileObjX = column * $10; RoomTileObjY = row * $10
;       + $40. Uses [0A], X.
; NOTE: only one tile object per room is kept: each placeholder overwrites the previous one.
;       Types $66 (armos) and $67 are never produced by the layout data: no overworld column heap
;       holds square index $2A or $2B (tiles $E9, $EA). Armos statues are laid out as plain squares
;       of tile $C0 (square index $2C).
; ----------------------------------------------------------------------------
RoomLayout_CheckTileObj:
    ; Compare A with $EA down to $E5; X = matching index 5 down to 0.
    LDX #$EA
    STX $0A
    LDX #$05
:
    CMP $0A
    BEQ :+                      ; Found the placeholder.
    DEC $0A
    DEX
    BPL :-
    BMI RoomLayout_CheckTileObj_Return             ; Not $E5-$EA: return with A unchanged.
:
    LDA RoomLayout_TileObjSquareTilesOW, X
    PHA                         ; Save the tile to draw.
    LDA RoomLayout_TileObjTypes, X
    STA RoomTileObjType
    LDA $06                     ; Square column.
    ASL                         ; X coordinate = column * $10.
    ASL
    ASL
    ASL
    STA RoomTileObjX
    LDA $07                     ; Square row.
    ASL                         ; Y coordinate = row * $10 + $40 (the play area starts at Y = $40).
    ASL
    ASL
    ASL
    CLC
    ADC #$40
    STA RoomTileObjY
    PLA                         ; The tile to draw.

; Shared return of RoomLayout_LayOutRoomOrCaveOW and RoomLayout_CheckTileObj.
RoomLayout_CheckTileObj_Return:
    RTS

; ----------------------------------------------------------------------------
; Writes one 2 x 2 tile overworld square into the play area tile map.
; IN:   A = the square's first tile; [0D] = square index; [00:01] + Y = address of its top-left cell.
; OUT:  four cells written (the map is stored by column, $16 tiles per column). Changes A, X, Y.
; NOTE: square index $10 and up: tiles A, A+1, A+2, A+3 in the order top-left, bottom-left, top-right,
;       bottom-right. Below $10: the four tiles of RoomLayout_OWSecondarySquareTiles entry [0D]; A is ignored.
; ----------------------------------------------------------------------------
RoomLayout_WriteSquareOW:
    LDX $0D                     ; Square index.
    CPX #$10
    BCC @WriteType3             ; Below $10: secondary square.

    ; Square index $10 and up: four consecutive tiles.
    TAX
    STA ($00), Y                ; Tile+0 to the top-left cell.
    INY
    INX
    TXA
    STA ($00), Y                ; Tile+1 to the bottom-left cell.
    TYA
    CLC
    ADC #$15                    ; +$15 (one column, $16, minus the row already stepped).
    TAY
    INX
    TXA
    STA ($00), Y                ; Tile+2 to the top-right cell.
    INX
    TXA

@WriteLastTile:
    INY
    STA ($00), Y                ; Last tile to the bottom-right cell.
    RTS

@WriteType3:
    ; Square index below $10: four tiles from RoomLayout_OWSecondarySquareTiles.
    TXA                         ; X = square index * 4.
    ASL
    ASL
    TAX
    LDA RoomLayout_OWSecondarySquareTiles, X
    STA ($00), Y                ; Top-left cell.
    INY
    INX
    LDA RoomLayout_OWSecondarySquareTiles, X
    STA ($00), Y                ; Bottom-left cell.
    TYA
    CLC
    ADC #$15
    TAY
    INX
    LDA RoomLayout_OWSecondarySquareTiles, X
    STA ($00), Y                ; Top-right cell.
    INX
    LDA RoomLayout_OWSecondarySquareTiles, X
    JMP @WriteLastTile          ; Go write the bottom-right cell.

; ----------------------------------------------------------------------------
; Points entry 0 of RoomLayout_ColumnDirectoryOW at the overworld heap 0 or, in a dungeon, at the cellar heap.
; IN:   CurLevel (0 = overworld).
; OUT:  RoomLayout_ColumnDirectoryOW[0] = RoomLayout_ColumnHeapOW0 (overworld) or RoomLayout_ColumnHeapUWCellar (dungeon). Changes A,
;       X, Y.
; NOTE: called by RoomLayout_LayOutRoom for every room, so the directory (a RAM copy) is right for the cellars
;       of the current dungeon, whose layouts use only heap 0.
; ----------------------------------------------------------------------------
RoomLayout_PatchColumnDirectoryForCellar:
    ; Overworld: the address of RoomLayout_ColumnHeapOW0.
    LDA RoomLayout_ColumnHeapOWAddr
    LDX RoomLayout_ColumnHeapOWAddr+1
    LDY CurLevel
    BEQ :+

    ; Dungeon: the address of the cellar column heap.
    LDA #<RoomLayout_ColumnHeapUWCellar
    LDX #>RoomLayout_ColumnHeapUWCellar
:
    STA RoomLayout_ColumnDirectoryOW
    STX RoomLayout_ColumnDirectoryOW+1
    RTS

; TABLE: layout addresses of caves and cellars; index = 0 ordinary cave, 2 shortcut stairs room,
;   4 cellar with layout id $3E (tunnel), 6 cellar with layout id $3F (item cellar); value = address
;   of 16 layout bytes for RoomLayout_LayOutRoomOrCaveOW.
RoomLayout_SubroomLayoutAddrs:
    .ADDR RoomLayout_CaveOW
    .ADDR RoomLayout_ShortcutCaveOW
    .ADDR RoomLayout_CellarUW0
    .ADDR RoomLayout_CellarUW1

; ----------------------------------------------------------------------------
; Mode $B init step: lays out the ordinary cave room and advances GameSubmode.
; NOTE: the unnamed label below is the common tail of this routine and the two that follow:
;       X = offset into RoomLayout_SubroomLayoutAddrs.
; ----------------------------------------------------------------------------
ModeBCave_Init_LayOutCave:
    LDX #$00                    ; Ordinary cave layout.
:
    LDA RoomLayout_SubroomLayoutAddrs, X
    STA $02
    LDA RoomLayout_SubroomLayoutAddrs+1, X
    STA $03
    INC GameSubmode
    JMP RoomLayout_LayOutRoomOrCaveOW

; Mode $C init step: lays out the shortcut stairs room and advances GameSubmode.
ModeCShortcut_Init_LayOutShortcut:
    LDX #$02                    ; Shortcut stairs room layout.
    BNE :-                      ; Always taken.

; ----------------------------------------------------------------------------
; Mode 9 init step: lays out the cellar (tunnel or item cellar) and advances GameSubmode.
; IN:   RoomId = the cellar's room; its layout id is $3E or $3F.
; OUT:  CurRow = 0 (the rows are copied to the screen from the top afterwards).
; ----------------------------------------------------------------------------
Mode9Cellar_Init_LayOutCellar:
    LDA #$00
    STA CurRow

    ; Bit 0 of the layout id selects the layout: even ($3E) = offset 4, the tunnel with a ladder at
    ; each end; odd ($3F) = offset 6, the item cellar.
    LDX #$04
    JSR Room_GetUniqueRoomId
    AND #$01
    BEQ :-
    LDX #$06
    BNE :-                      ; Always taken.
