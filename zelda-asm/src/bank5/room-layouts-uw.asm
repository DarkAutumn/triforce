; ============================================================================
; FILE:     bank5/room-layouts-uw.asm    BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 4410-4535
; SUBJECTS: Door, RoomLayout
; TOPIC:    Dungeon (UW) room graphics data: the wall outline tile list, the door face tile sets of the
;           four walls, the 42 floor layouts with their ten column heaps, and the two cellar layouts
;           with their column heap. No code.
; SEE ALSO: bank5/fill-walls.asm (RoomLayout_FillWalls reads RoomLayout_UWWallTiles); bank5/lay-out-doors.asm (Door_LayOut
;           and the tables that index the door faces); bank5/update-doors.asm (door animation rewrites
;           4 tiles of a face); bank5/uw-floor-push-block.asm (RoomLayout_LayOutUWFloor decodes RoomLayout_RoomsUW and
;           the column heaps; RoomLayout_UWColumnHeapAddrs, RoomLayout_UWSquareTiles); bank5/layout-room-ow.asm
;           (Mode9Cellar_Init_LayOutCellar and RoomLayout_LayOutRoomOrCaveOW decode the cellar layouts;
;           RoomLayout_SubroomLayoutAddrs, RoomLayout_PatchColumnDirectoryForCellar, RoomLayout_OWSecondarySquareTiles);
;           bank5/room-layouts-ow.asm (the overworld counterpart).
; KEY FACTS:
;   - The play area tile map is 32 columns x 22 rows of 8-pixel tiles. In a dungeon room the outer ring
;     (columns 0 and 31, rows 0 and 21) stays brick tile $F6, the walls take columns 1-3 and 28-30 and
;     rows 1-3 and 18-20, and the floor takes columns 4-27, rows 4-17: 12 x 7 squares of 16 pixels.
;     RoomLayout_UWWallTiles, RoomLayout_RoomsUW (code: RoomLayout_FillWalls, RoomLayout_LayOutUWFloor).
;   - Each wall has 5 door faces of 12 tiles: 1 open doorway, 2 locked door, 3 shutter, 4 bombable wall
;     (looks like plain wall), 5 bombed hole. Door_FaceTilesE, Door_FaceTilesW, Door_FaceTilesS,
;     Door_FaceTilesN.
;   - Faces 1, 2 and 3 of a wall differ from each other only in a 2x2 block of tiles (the passage),
;     and so do faces 4 and 5; a door only changes inside its group. In face 1 the passage tiles are
;     $24 and $74-$77, all below $78, the first unwalkable tile number in a dungeon
;     (RoomFirstUnwalkableTile); in faces 2, 3 and 4 every tile is $78 or above; in face 5 two
;     passage tiles are $24 and two are $8C-$93. Door_FaceTilesE (the same holds for W, S, N).
;   - A floor layout is 12 bytes, one per column of squares: high nibble = column heap number (0-9),
;     low nibble = column number in that heap. 42 layouts, indexed by the unique room id ($00-$29).
;     RoomLayout_RoomsUW.
;   - A column heap byte is a square descriptor: bit 7 = first square of a column, bits 6-4 = extra
;     repeats, bits 2-0 = square index (RoomLayout_UWSquareTiles). A column is 7 squares; columns share
;     descriptors, and the last column of a heap may run into the next heap. RoomLayout_ColumnHeapUW0.
;   - Layout $00 is twelve $00 bytes: every column is seven floor squares (an empty room).
;     RoomLayout_RoomsUW, RoomLayout_ColumnHeapUW0.
;   - Cellars (unique room ids $3E and $3F) use the overworld format instead: 16 column bytes, 11
;     squares per column, squares from the overworld tables. RoomLayout_CellarUW0, RoomLayout_CellarUW1,
;     RoomLayout_ColumnHeapUWCellar.
; CONTENTS: (generated; L = line in this file)
;   L0068 RoomLayout_UWWallTiles         data  TABLE: dungeon wall outline tiles for play area columns 1-15, stored by column; read by RoomLayout_FillWalls.
;   L0102 Door_FaceTilesE                data  TABLE: 5 door faces x 12 tiles for the door in the E (right) wall: play area columns 28-30, rows 9-12
;   L0115 Door_FaceTilesW                data  TABLE: 5 door faces x 12 tiles for the door in the W (left) wall: play area columns 1-3, rows 9-12.
;   L0128 Door_FaceTilesS                data  TABLE: 5 door faces x 12 tiles for the door in the S (bottom) wall: tile columns 14-17, rows 18-20.
;   L0141 Door_FaceTilesN                data  TABLE: 5 door faces x 12 tiles for the door in the N (top) wall: play area columns 14-17, rows 1-3.
;   L0174 RoomLayout_RoomsUW             data  DATA: 42 dungeon floor layouts x 12 column descriptors (504 bytes); index = unique room id * 12.
;   L0180 RoomLayout_ColumnHeapUW0       data  TABLE: dungeon column heap 0: 8 columns of square descriptors (format above).
;   L0187 RoomLayout_ColumnHeapUW1       data  TABLE: dungeon column heap 1: 8 columns of square descriptors.
;   L0196 RoomLayout_ColumnHeapUW2       data  TABLE: dungeon column heap 2: 7 columns of square descriptors.
;   L0204 RoomLayout_ColumnHeapUW3       data  TABLE: dungeon column heap 3: 7 columns of square descriptors.
;   L0212 RoomLayout_ColumnHeapUW4       data  TABLE: dungeon column heap 4: 7 columns of square descriptors.
;   L0220 RoomLayout_ColumnHeapUW5       data  TABLE: dungeon column heap 5: 8 columns of square descriptors.
;   L0228 RoomLayout_ColumnHeapUW6       data  TABLE: dungeon column heap 6: 8 columns of square descriptors (block column, then water columns).
;   L0236 RoomLayout_ColumnHeapUW7       data  TABLE: dungeon column heap 7: 8 columns of square descriptors.
;   L0245 RoomLayout_ColumnHeapUW8       data  TABLE: dungeon column heap 8: 9 columns of square descriptors.
;   L0255 RoomLayout_ColumnHeapUW9       data  TABLE: dungeon column heap 9: 9 columns of square descriptors.
;   L0269 RoomLayout_CellarUW0           data  TABLE: cellar layout for unique room id $3E (even id): 16 column numbers into RoomLayout_ColumnHeapUWCellar.
;   L0274 RoomLayout_CellarUW1           data  TABLE: cellar layout for unique room id $3F (odd id): 16 column numbers into RoomLayout_ColumnHeapUWCellar.
;   L0285 RoomLayout_ColumnHeapUWCellar  data  TABLE: cellar column heap: 5 columns of overworld-format square descriptors, 11 squares per column.
; ============================================================================

; TABLE: dungeon wall outline tiles for play area columns 1-15, stored by column; read by RoomLayout_FillWalls.
;   Columns 1-3 (the left wall) have 10 tiles each, for rows 1-10. Columns 4-15 (the left half of the
;   top wall) have 3 tiles, for rows 1-3, then a $00 terminator. RoomLayout_FillWalls writes each tile at its row
;   and, flipped, at the mirrored row of the bottom half (rows 20 up to 11), then rotates the left half
;   of the room into the right half. 78 ($4E) bytes. Tiles: $DE vertical edge line, $DC horizontal
;   edge line, $F5 wall interior, $E0 outer corner; tiles below $DC come in groups of four
;   orientations (see RoomLayout_FillWalls).
RoomLayout_UWWallTiles:
    .BYTE $E0, $F5, $F5, $F5, $F5, $B8, $F5, $D4    ; column 1, rows 1-8
    .BYTE $F5, $F5, $F5, $C4, $DE, $DE, $BC, $C8    ; column 1, rows 9-10; column 2, rows 1-6
    .BYTE $DE, $BC, $DE, $DE, $F5, $DC, $C4, $DE    ; column 2, rows 7-10; column 3, rows 1-4
    .BYTE $C8, $DE, $BC, $C8, $DE, $DE, $F5, $DC    ; column 3, rows 5-10; column 4, rows 1-2
    .BYTE $DC, $00, $C0, $D0, $DC, $00, $F5, $DC    ; column 4, row 3, end; column 5; column 6, rows 1-2
    .BYTE $CC, $00, $F5, $DC, $DC, $00, $F5, $CC    ; column 6, row 3, end; column 7; column 8, rows 1-2
    .BYTE $D0, $00, $F5, $DC, $DC, $00, $C0, $D0    ; column 8, row 3, end; column 9; column 10, rows 1-2
    .BYTE $DC, $00, $F5, $DC, $CC, $00, $F5, $DC    ; column 10, row 3, end; column 11; column 12, rows 1-2
    .BYTE $DC, $00, $D8, $CC, $D0, $00, $F5, $DC    ; column 12, row 3, end; column 13; column 14, rows 1-2
    .BYTE $DC, $00, $F5, $DC, $DC, $00              ; column 14, row 3, end; column 15

; Door faces. Each of the four tables below holds 5 faces of 12 tiles for the door of one wall:
;   face 1 = open doorway           (door type 0, and any opened key door or shutter)
;   face 2 = locked door            (door types 5 and 6)
;   face 3 = shutter                (door type 7)
;   face 4 = bombable wall, intact  (door type 4; the same tiles as plain wall)
;   face 5 = bombed hole            (door type 4, opened)
; Door types 1-3 (wall, false walls) have no face: the wall written by RoomLayout_FillWalls stays.
; A face is two halves of 6 tiles; each half is stored by column, top to bottom:
;   E and W doors: 3 columns x 4 rows. Half 0 = rows 0-1, half 1 = rows 2-3.
;                  Tile index = half * 6 + column * 2 + (row - 2 * half).
;   S and N doors: 4 columns x 3 rows. Half 0 = columns 0-1, half 1 = columns 2-3.
;                  Tile index = half * 6 + (column - 2 * half) * 3 + row.
; Door_LayOut copies a whole face into the play area tile map; Door_Update sends only the four
; passage tiles to the screen (Door_PassageTileIndexes). Faces 1-3 of a wall differ from each other
; only in those four tiles, and so do faces 4 and 5; faces 4 and 5 differ from faces 1-3 in every
; tile. A door changes only between faces 1, 2, 3 or between faces 4 and 5.
; Rows of 8 bytes below do not line up with the 12-byte faces; the trailing comments give the face
; and tile indexes.

; TABLE: 5 door faces x 12 tiles for the door in the E (right) wall: play area columns 28-30, rows 9-12
;   Face 1 as laid out (rows top to bottom):  $88 $8A $87 / $74 $24 $87 / $75 $24 $87 / $89 $8B $87.
;   The passage is tile indexes 1, 3, 6, 8 (columns 0-1, rows 1-2): $74 $24 / $75 $24 when open.
Door_FaceTilesE:
    .BYTE $88, $74, $8A, $24, $87, $87, $75, $89    ; face 1 (open), tiles 0-7
    .BYTE $24, $8B, $87, $87, $88, $A4, $8A, $A6    ; face 1, tiles 8-11; face 2 (locked), tiles 0-3
    .BYTE $87, $87, $A5, $89, $A7, $8B, $87, $87    ; face 2, tiles 4-11
    .BYTE $88, $AC, $8A, $AE, $87, $87, $AD, $89    ; face 3 (shutter), tiles 0-7
    .BYTE $AF, $8B, $87, $87, $DF, $DF, $DF, $DF    ; face 3, tiles 8-11; face 4 (bombable wall), tiles 0-3
    .BYTE $F5, $F5, $DF, $DF, $DF, $DF, $F5, $F5    ; face 4, tiles 4-11
    .BYTE $DF, $24, $DF, $92, $F5, $F5, $24, $DF    ; face 5 (bombed hole), tiles 0-7
    .BYTE $93, $DF, $F5, $F5                        ; face 5, tiles 8-11

; TABLE: 5 door faces x 12 tiles for the door in the W (left) wall: play area columns 1-3, rows 9-12.
;   Face 1 as laid out (rows top to bottom):  $82 $83 $85 / $82 $24 $76 / $82 $24 $77 / $82 $84 $86.
;   The passage is tile indexes 3, 5, 8, 10 (columns 1-2, rows 1-2): $24 $76 / $24 $77 when open.
Door_FaceTilesW:
    .BYTE $82, $82, $83, $24, $85, $76, $82, $82    ; face 1 (open), tiles 0-7
    .BYTE $24, $84, $77, $86, $82, $82, $83, $A0    ; face 1, tiles 8-11; face 2 (locked), tiles 0-3
    .BYTE $85, $A2, $82, $82, $A1, $84, $A3, $86    ; face 2, tiles 4-11
    .BYTE $82, $82, $83, $AC, $85, $AE, $82, $82    ; face 3 (shutter), tiles 0-7
    .BYTE $AD, $84, $AF, $86, $F5, $F5, $DE, $DE    ; face 3, tiles 8-11; face 4 (bombable wall), tiles 0-3
    .BYTE $DE, $DE, $F5, $F5, $DE, $DE, $DE, $DE    ; face 4, tiles 4-11
    .BYTE $F5, $F5, $DE, $90, $DE, $24, $F5, $F5    ; face 5 (bombed hole), tiles 0-7
    .BYTE $91, $DE, $24, $DE                        ; face 5, tiles 8-11

; TABLE: 5 door faces x 12 tiles for the door in the S (bottom) wall: tile columns 14-17, rows 18-20.
;   Face 1 as laid out (rows top to bottom):  $7E $76 $74 $80 / $7F $24 $24 $81 / $7D $7D $7D $7D.
;   The passage is tile indexes 3, 6, 4, 7 (columns 1-2, rows 0-1): $76 $74 / $24 $24 when open.
Door_FaceTilesS:
    .BYTE $7E, $7F, $7D, $76, $24, $7D, $74, $24    ; face 1 (open), tiles 0-7
    .BYTE $7D, $80, $81, $7D, $7E, $7F, $7D, $9C    ; face 1, tiles 8-11; face 2 (locked), tiles 0-3
    .BYTE $9D, $7D, $9E, $9F, $7D, $80, $81, $7D    ; face 2, tiles 4-11
    .BYTE $7E, $7F, $7D, $A8, $A9, $7D, $AA, $AB    ; face 3 (shutter), tiles 0-7
    .BYTE $7D, $80, $81, $7D, $DD, $DD, $F5, $DD    ; face 3, tiles 8-11; face 4 (bombable wall), tiles 0-3
    .BYTE $DD, $F5, $DD, $DD, $F5, $DD, $DD, $F5    ; face 4, tiles 4-11
    .BYTE $DD, $DD, $F5, $24, $8E, $F5, $24, $8F    ; face 5 (bombed hole), tiles 0-7
    .BYTE $F5, $DD, $DD, $F5                        ; face 5, tiles 8-11

; TABLE: 5 door faces x 12 tiles for the door in the N (top) wall: play area columns 14-17, rows 1-3.
;   Face 1 as laid out (rows top to bottom):  $78 $78 $78 $78 / $79 $24 $24 $7B / $7A $77 $75 $7C.
;   The passage is tile indexes 4, 7, 5, 8 (columns 1-2, rows 1-2): $24 $24 / $77 $75 when open.
Door_FaceTilesN:
    .BYTE $78, $79, $7A, $78, $24, $77, $78, $24    ; face 1 (open), tiles 0-7
    .BYTE $75, $78, $7B, $7C, $78, $79, $7A, $78    ; face 1, tiles 8-11; face 2 (locked), tiles 0-3
    .BYTE $98, $99, $78, $9A, $9B, $78, $7B, $7C    ; face 2, tiles 4-11
    .BYTE $78, $79, $7A, $78, $A8, $A9, $78, $AA    ; face 3 (shutter), tiles 0-7
    .BYTE $AB, $78, $7B, $7C, $F5, $DC, $DC, $F5    ; face 3, tiles 8-11; face 4 (bombable wall), tiles 0-3
    .BYTE $DC, $DC, $F5, $DC, $DC, $F5, $DC, $DC    ; face 4, tiles 4-11
    .BYTE $F5, $DC, $DC, $F5, $8C, $24, $F5, $8D    ; face 5 (bombed hole), tiles 0-7
    .BYTE $24, $F5, $DC, $DC                        ; face 5, tiles 8-11

; Dungeon floor format (decoded by RoomLayout_LayOutUWFloor). The floor is 12 columns x 7 rows of 16-pixel
; squares (play area tile columns 4-27, rows 4-17).
;   Layout: 12 column descriptors, left to right. High nibble = column heap number (index into
;     RoomLayout_UWColumnHeapAddrs, 0-9); low nibble = column number inside that heap.
;   Column heap: a string of square descriptors. Bit 7 set = a column starts here (column N of a heap
;     starts at the heap's N-th descriptor with bit 7 set, counting from 0). Bits 6-4 = number of
;     extra repeats (the square is written that many times plus one). Bits 2-0 = square index into
;     RoomLayout_UWSquareTiles. Bit 3 is not read and is 0 everywhere.
;   Laying out a column reads descriptors from its start until 7 squares are written and ignores bit 7
;   while doing so: columns overlap (a column may continue through the start of the next one), a
;   repeat count may be cut off by the seventh row, and the last column of heaps 0, 1, 2, 3, 4 and 6
;   runs on into the first bytes of the heap that follows it.
;   Square indexes (RoomLayout_UWSquareTiles gives the square's first tile; RoomLayout_WriteSquareUW writes tiles
;   T, T+1, T+2, T+3 for a first tile in $70-$F2 and one tile four times otherwise):
;   0 = $B0-$B3 block (the square PushBlock_FindAndCreate looks for), 1 = $74-$77 floor,
;   2 = $94-$97 and 3 = $B4-$B7 (solid: $78 and above; appearance unverified), 4 = $70-$73 stairs,
;   5 = $68 x 4 (walkable: below $78; appearance unverified), 6 = $F4 x 4 water (the tile the ladder
;   test looks for in a dungeon), 7 = $24 x 4 blank (black, walkable).
; The column lists below give each column's 7 square indexes, top to bottom.

; DATA: 42 dungeon floor layouts x 12 column descriptors (504 bytes); index = unique room id * 12.
;   The unique room id is the low 6 bits of LevelBlockAttrsD for the room (Room_GetUniqueRoomId); ids $00-$29
;   have a layout here. Layout $00 is twelve $00 bytes (an empty floor).
RoomLayout_RoomsUW:
    .INCBIN "dat/RoomLayoutsUW.dat"

; TABLE: dungeon column heap 0: 8 columns of square descriptors (format above).
;   col 0: 1 1 1 1 1 1 1     col 1: 0 1 1 1 1 1 0     col 2: 1 1 1 1 1 0 1     col 3: 0 0 0 4 0 0 0
;   col 4: 0 0 0 1 0 0 0     col 5: 0 0 0 1 1 1 1     col 6: 0 0 1 1 1 0 0     col 7: 0 0 1 1 1 0 1
RoomLayout_ColumnHeapUW0:
    .BYTE $E1, $80, $C1, $00, $01, $A0, $04, $A0
    .BYTE $01, $A0, $31, $90, $21, $90, $21, $00

; TABLE: dungeon column heap 1: 8 columns of square descriptors.
;   col 0: 1 0 1 4 1 0 1     col 1: 1 0 1 0 1 0 1     col 2: 1 0 1 1 1 0 1     col 3: 1 0 0 0 0 0 1
;   col 4: 1 0 0 0 1 0 1     col 5: 1 0 4 0 1 0 1     col 6: 0 0 1 1 1 1 1     col 7: 0 1 1 1 1 1 1
RoomLayout_ColumnHeapUW1:
    .BYTE $81, $00, $01, $04, $81, $00, $01, $00
    .BYTE $01, $00, $81, $00, $21, $00, $81, $40
    .BYTE $81, $20, $01, $00, $81, $00, $04, $00
    .BYTE $01, $00, $01, $90, $41, $80

; TABLE: dungeon column heap 2: 7 columns of square descriptors.
;   col 0: 1 1 1 1 1 1 0     col 1: 0 1 1 1 0 1 1     col 2: 1 1 0 0 0 1 1     col 3: 1 1 0 1 0 1 1
;   col 4: 0 0 0 0 0 0 1     col 5: 1 0 1 1 1 1 1     col 6: 1 1 1 1 0 1 1
RoomLayout_ColumnHeapUW2:
    .BYTE $D1, $80, $21, $00, $91, $20, $91, $00
    .BYTE $01, $00, $11, $D0, $81, $00, $41, $B1
    .BYTE $00

; TABLE: dungeon column heap 3: 7 columns of square descriptors.
;   col 0: 1 1 0 1 1 1 0     col 1: 1 1 1 0 1 1 1     col 2: 1 1 1 1 6 1 1     col 3: 1 1 0 1 1 1 1
;   col 4: 1 1 1 1 0 1 0     col 5: 1 0 1 1 1 0 0     col 6: 0 0 7 7 2 1 1
RoomLayout_ColumnHeapUW3:
    .BYTE $91, $00, $A1, $00, $21, $B1, $06, $91
    .BYTE $00, $B1, $00, $81, $00, $21, $90, $17
    .BYTE $02

; TABLE: dungeon column heap 4: 7 columns of square descriptors.
;   col 0: 1 1 7 7 1 7 7     col 1: 1 6 6 1 1 1 1     col 2: 1 1 1 1 7 1 1     col 3: 1 1 6 1 1 1 1
;   col 4: 7 1 1 1 1 7 7     col 5: 1 0 1 3 1 0 1     col 6: 1 0 1 2 1 0 1
RoomLayout_ColumnHeapUW4:
    .BYTE $91, $17, $01, $17, $81, $16, $B1, $07
    .BYTE $91, $06, $31, $87, $31, $17, $81, $00
    .BYTE $01, $03, $81, $00, $01, $02, $01

; TABLE: dungeon column heap 5: 8 columns of square descriptors.
;   col 0: 0 1 0 0 1 1 3     col 1: 3 7 7 7 7 7 3     col 2: 3 1 1 1 1 1 3     col 3: 1 1 1 1 1 1 3
;   col 4: 0 0 1 0 0 1 1     col 5: 1 1 0 1 1 0 0     col 6: 0 0 7 7 3 1 1     col 7: 0 0 0 0 0 1 1
RoomLayout_ColumnHeapUW5:
    .BYTE $80, $01, $10, $11, $83, $47, $83, $41
    .BYTE $03, $D1, $03, $90, $01, $10, $91, $00
    .BYTE $11, $90, $17, $03, $11, $C0, $11

; TABLE: dungeon column heap 6: 8 columns of square descriptors (block column, then water columns).
;   col 0: 0 0 0 0 0 0 0     col 1: 6 6 6 6 6 6 6     col 2: 1 6 6 6 6 6 1     col 3: 6 6 6 6 6 1 6
;   col 4: 1 6 1 6 1 6 1     col 5: 6 1 6 1 6 1 6     col 6: 1 6 1 1 1 6 1     col 7: 6 1 6 6 6 1 6
RoomLayout_ColumnHeapUW6:
    .BYTE $E0, $E6, $81, $C6, $81, $86, $01, $06
    .BYTE $01, $06, $81, $06, $21, $86, $01, $26
    .BYTE $01

; TABLE: dungeon column heap 7: 8 columns of square descriptors.
;   col 0: 6 1 1 1 1 1 6     col 1: 6 1 6 6 6 6 6     col 2: 1 1 1 1 1 1 2     col 3: 6 6 6 1 6 6 6
;   col 4: 1 6 6 6 6 6 6     col 5: 1 6 6 1 1 6 6     col 6: 7 7 7 7 7 7 7     col 7: 5 5 5 5 5 5 5
RoomLayout_ColumnHeapUW7:
    .BYTE $86, $41, $86, $01, $46, $D1, $02, $A6
    .BYTE $01, $26, $81, $56, $81, $16, $11, $16
    .BYTE $E7, $E5

; TABLE: dungeon column heap 8: 9 columns of square descriptors.
;   col 0: 1 0 0 1 0 0 1     col 1: 1 7 7 7 1 1 1     col 2: 1 1 1 6 1 1 1     col 3: 0 1 0 0 1 1 2
;   col 4: 2 7 7 7 7 7 2     col 5: 2 1 1 1 1 1 2     col 6: 0 1 0 0 1 1 1     col 7: 1 1 1 2 1 1 1
;   col 8: 1 1 1 3 1 1 1
RoomLayout_ColumnHeapUW8:
    .BYTE $81, $10, $01, $10, $81, $27, $A1, $06
    .BYTE $21, $80, $01, $10, $11, $82, $47, $82
    .BYTE $41, $02, $80, $01, $10, $A1, $02, $A1
    .BYTE $03, $21

; TABLE: dungeon column heap 9: 9 columns of square descriptors.
;   col 0: 0 7 7 7 1 1 1     col 1: 1 0 2 1 1 0 1     col 2: 1 0 3 1 1 0 1     col 3: 1 2 1 2 1 2 1
;   col 4: 1 2 1 2 1 2 5     col 5: 1 1 1 1 1 5 5     col 6: 1 1 1 1 5 5 5     col 7: 1 3 1 3 1 3 1
;   col 8: 1 3 1 3 1 3 5
RoomLayout_ColumnHeapUW9:
    .BYTE $80, $27, $21, $81, $00, $02, $11, $00
    .BYTE $81, $00, $03, $11, $00, $81, $02, $81
    .BYTE $02, $01, $02, $01, $02, $05, $C1, $15
    .BYTE $B1, $25, $81, $03, $81, $03, $01, $03
    .BYTE $01, $03, $05

; Cellar format. A cellar is laid out by the overworld decoder (RoomLayout_LayOutRoomOrCaveOW): 16 column
; descriptors, each column 11 squares high. While CurLevel is not 0, RoomLayout_PatchColumnDirectoryForCellar
; points entry 0 of RoomLayout_ColumnDirectoryOW at RoomLayout_ColumnHeapUWCellar, so the high nibble of every descriptor
; below is 0 and the low nibble is the column number in that heap. Mode9Cellar_Init_LayOutCellar
; picks the layout by bit 0 of the unique room id: $3E = RoomLayout_CellarUW0, $3F = RoomLayout_CellarUW1.

; TABLE: cellar layout for unique room id $3E (even id): 16 column numbers into RoomLayout_ColumnHeapUWCellar.
RoomLayout_CellarUW0:
    .BYTE $04, $04, $00, $01, $00, $00, $00, $00
    .BYTE $00, $00, $00, $00, $01, $00, $04, $04

; TABLE: cellar layout for unique room id $3F (odd id): 16 column numbers into RoomLayout_ColumnHeapUWCellar.
RoomLayout_CellarUW1:
    .BYTE $04, $04, $00, $01, $00, $00, $00, $03
    .BYTE $03, $03, $03, $02, $03, $03, $04, $04

; TABLE: cellar column heap: 5 columns of overworld-format square descriptors, 11 squares per column.
;   Descriptor: bit 7 = a column starts here, bit 6 = write the square twice, bits 5-0 = square index.
;   All indexes here are below $10, so each square is four tiles from RoomLayout_OWSecondarySquareTiles:
;   0 = $24 x 4 (blank), 1 = $6F x 4, 2 = $F3 x 4, 3 = $FA x 4, $0C = $F3 on the top row and $24 on the
;   bottom row. (What tiles $6F, $F3 and $FA look like in the cellar tile set is unverified.)
;   col 0: 2 3 3 3 3 3 3 2 C 3 3     col 1: 0 1 1 1 1 1 1 1 1 3 3     col 2: 2 3 3 2 2 C 1 1 1 3 3
;   col 3: 2 3 3 2 2 C 3 2 C 3 3     col 4: 2 3 3 3 3 3 3 3 3 3 3
RoomLayout_ColumnHeapUWCellar:
    .BYTE $82, $43, $43, $43, $02, $0C, $43, $80
    .BYTE $41, $41, $41, $41, $43, $82, $43, $42
    .BYTE $0C, $01, $41, $43, $82, $43, $42, $0C
    .BYTE $03, $02, $0C, $43, $82, $43, $43, $43
    .BYTE $43, $43
