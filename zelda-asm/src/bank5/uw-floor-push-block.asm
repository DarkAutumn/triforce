; ============================================================================
; FILE:     bank5/uw-floor-push-block.asm BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 5311-5535
; SUBJECTS: (none)
; TOPIC:    Lays out the floor of a dungeon room (12 x 7 squares of 2 x 2 tiles) from its room layout
;           and the dungeon column heaps, and creates the push block object for rooms flagged with one.
; SEE ALSO: bank5/room-layouts-uw.asm (RoomLayoutsUW, ColumnHeapUW0-9: the data decoded here);
;           bank5/mode12-lay-out-room.asm (LayOutRoom: walls and doors first, then LayoutUWFloor);
;           bank5/layout-room-ow.asm (overworld, cave and cellar layout: same scheme, other bit fields);
;           bank5/create-room-objects.asm (CreateRoomObjects calls FindAndCreatePushBlockObject);
;           bank4/push-block.asm (the push block object's states); bank7/tile-obj-tiles.asm
;           (GetUniqueRoomId).
; KEY FACTS:
;   - A dungeon room layout is 12 bytes, one per floor column: high nibble = column heap 0-9, low nibble =
;     index of the column inside that heap. Layout address = RoomLayoutsUW + layout id * 12; layout id =
;     LevelBlockAttrsD bits 5-0 (LayoutUWFloor).
;   - A square descriptor in a column heap: bit 7 = first square of a column, bits 6-4 = extra repeats
;     (the square is written 1-8 times), bits 2-0 = index into PrimarySquaresUW; bit 3 is not read
;     (LayoutUWFloor).
;   - The floor starts at tile column 4, tile row 4 of the play area ($658C; pixel $20,$60) and is 12
;     squares wide, 7 high (LayoutUWFloor).
;   - Square tiles $70-$F2 expand to four consecutive tiles (top-left, bottom-left, top-right,
;     bottom-right); any other tile fills all four cells (WriteSquareUW).
;   - The push block object always goes in object slot $0B, type $68, at the first block tile ($B0)
;     found left to right in the middle floor row (Y = $90, X = $20-$C0 in steps of $10); if none is
;     found it is put at X = $D0 (FindAndCreatePushBlockObject).
;   - Layout id $21 (the level entrance room) has a special push block position ($40,$80); no room of
;     the game's level data combines layout $21 with the push block flag, so it is never used
;     (FindAndCreatePushBlockObject).
; CONTENTS: (generated; L = line in this file)
;   L0040 ColumnDirectoryUW             data  TABLE: addresses of the ten dungeon column heaps; index = high nibble of a room layout byte; value =
;   L0056 PrimarySquaresUW              data  TABLE: first tile of each dungeon floor square; index = bits 2-0 of a square descriptor; value = tile.
;   L0066 LayoutUWFloor                 code  Writes the 12 x 7 squares of the current dungeon room's floor into the play area tile map.
;   L0176 WriteSquareUW                 code  Writes one 2 x 2 tile dungeon square into the play area tile map.
;   L0227 FindAndCreatePushBlockObject  code  Creates the push block object (type $68) in object slot $0B at the room's block tile.
; ============================================================================

; TABLE: addresses of the ten dungeon column heaps; index = high nibble of a room layout byte; value =
;   address of a run of square descriptors (see LayoutUWFloor).
ColumnDirectoryUW:
    .ADDR ColumnHeapUW0
    .ADDR ColumnHeapUW1
    .ADDR ColumnHeapUW2
    .ADDR ColumnHeapUW3
    .ADDR ColumnHeapUW4
    .ADDR ColumnHeapUW5
    .ADDR ColumnHeapUW6
    .ADDR ColumnHeapUW7
    .ADDR ColumnHeapUW8
    .ADDR ColumnHeapUW9

; TABLE: first tile of each dungeon floor square; index = bits 2-0 of a square descriptor; value = tile.
;   0 $B0 block (the tile FindAndCreatePushBlockObject searches for), 1 $74 floor tile (unverified),
;   2 $94 and 3 $B4 statues (unverified), 4 $70 stairs, 5 $68 (one tile repeated; sand, unverified),
;   6 $F4 (one tile repeated; water, unverified), 7 $24 blank (black).
PrimarySquaresUW:
    .BYTE $B0, $74, $94, $B4, $70, $68, $F4, $24

; ----------------------------------------------------------------------------
; Writes the 12 x 7 squares of the current dungeon room's floor into the play area tile map.
; IN:   RoomId; LevelBlockAttrsD (layout id). The walls and doors are already laid out by the caller.
; OUT:  PlayAreaTiles columns 4-27, rows 4-17 written. Uses [00]-[08].
; NOTE: [02:03] = the room's 12 layout bytes, [04:05] = current square descriptor, [00:01] = play area
;       cursor, [06] = column 0-11, [07] = row 0-6, [08] = repeats done of the current square.
; ----------------------------------------------------------------------------
LayoutUWFloor:
    JSR GetUniqueRoomId
    PHA                         ; Save the layout id (unique room id).
    LDA #<RoomLayoutsUW         ; [02:03] = start of the dungeon room layouts.
    STA $02
    LDA #>RoomLayoutsUW
    STA $03
    PLA
    ASL                         ; [02:03] += layout id * 12 (id * 4, added three times):
                                ; each layout has 12 column bytes.
    ASL
    STA $00
    JSR AddToInt16At2
    LDA $00
    JSR AddToInt16At2
    LDA $00
    JSR AddToInt16At2
    LDA #$8C                    ; [00:01] = $658C = PlayAreaTiles + 4 * $16 + 4: tile column 4, row 4,
                                ; the top-left tile of the floor.
    STA $00
    LDA #$65
    STA $01

    ; For each of the 12 floor columns, indexed by [06]:
    LDY #$00
    STY $06

@LoopColumnUW:
    LDY $06
    LDA ($02), Y                ; Layout byte of this column.
    AND #$F0                    ; X = heap number * 2 (high nibble).
    LSR
    LSR
    LSR
    TAX
    LDA ColumnDirectoryUW, X    ; [04:05] = address of that column heap.
    STA $04
    LDA ColumnDirectoryUW+1, X
    STA $05
    LDA ($02), Y                ; Layout byte again.
    AND #$0F                    ; X = index of the column inside the heap.
    TAX
    LDY #$00

@FindSquare:
    ; Walk the heap, counting descriptors with bit 7 set (column starts), until column X is reached.
    LDA ($04), Y
    BPL :+                      ; Not a column start.
    DEX
    BMI @FoundColumn            ; This is the column wanted.
:
    INY
    JMP @FindSquare

@FoundColumn:
    TYA
    JSR AddToInt16At4           ; [04:05] now points at the column's first square descriptor.
    LDA #$00
    STA $07                     ; Row = 0.
    STA $08                     ; Repeats done = 0.

@LoopSquareRow:
    LDY #$00
    LDA ($04), Y                ; Square descriptor.
    AND #$07                    ; Bits 2-0: index into PrimarySquaresUW.
    TAX
    LDA PrimarySquaresUW, X
    LDY #$00
    JSR WriteSquareUW
    LDA #$02                    ; Move the cursor 2 tiles down: the next square of this column.
    JSR AddToInt16At0
    LDY #$00
    LDA ($04), Y                ; Square descriptor.
    AND #$70                    ; Bits 6-4: number of extra repeats.
    LSR
    LSR
    LSR
    LSR
    CMP $08
    BEQ :+                      ; All repeats done: go on to the next descriptor.
    INC $08                     ; Else count a repeat and write the same square again.
    JMP @NextLoopSquareRow
:
    LDA #$00                    ; Reset the repeat count.
    STA $08
    JSR Add1ToInt16At4          ; Point to the next square descriptor.

@NextLoopSquareRow:
    INC $07                     ; Next row.
    LDA $07
    CMP #$07                    ; The floor is 7 squares high.
    BCC @LoopSquareRow
    LDA #$1E                    ; The cursor is 14 tiles below the column top: +$1E (2 * $16 - 14)
                                ; puts it at the top of the floor two tile columns to the right.
    JSR AddToInt16At0
    INC $06                     ; Next column.
    LDA $06
    CMP #$0C                    ; The floor is 12 squares wide.
    BCS :+
    JMP @LoopColumnUW
:
    RTS

; ----------------------------------------------------------------------------
; Writes one 2 x 2 tile dungeon square into the play area tile map.
; IN:   A = the square's tile (PrimarySquaresUW value); [00:01] + Y = address of its top-left cell.
; OUT:  four cells written (the map is stored by column, $16 tiles per column). Changes A, X, Y.
; NOTE: tiles $70-$F2 are the first of four consecutive tiles (top-left, bottom-left, top-right,
;       bottom-right); tiles below $70 or from $F3 up fill all four cells.
; ----------------------------------------------------------------------------
WriteSquareUW:
    CMP #$70
    BCC @WriteType2
    CMP #$F3
    BCS @WriteType2

    ; Tile $70-$F2: four consecutive tiles.
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

@WriteType2:
    ; Tile below $70 or $F3 and up: the same tile in all four cells.
    STA ($00), Y                ; Top-left cell.
    INY
    STA ($00), Y                ; Bottom-left cell.
    PHA
    TYA
    CLC
    ADC #$15
    TAY
    PLA
    STA ($00), Y                ; Top-right cell.
    JMP @WriteLastTile          ; Go write the bottom-right cell.

; ----------------------------------------------------------------------------
; Creates the push block object (type $68) in object slot $0B at the room's block tile.
; IN:   the room is laid out in PlayAreaTiles. Called by CreateRoomObjects when the room's
;       LevelBlockAttrsD bit 6 (push block) is set.
; OUT:  ObjType[$0B] = $68; ObjState[$0B] = 0; ObjDir[$0B] = 0; ObjX[$0B], ObjY[$0B] set. Uses [00:01].
; NOTE: the block is the leftmost tile $B0 in tile row $A of the play area (the middle floor row,
;       Y = $90), testing tile columns 4, 6, ... 24 (X = $20-$C0). If none matches, the object is still
;       created, at X = $D0.
; ----------------------------------------------------------------------------
FindAndCreatePushBlockObject:
    ; Reset the block's state and direction.
    LDA #$00
    STA ObjState+11
    STA ObjDir+11

    ; Layout id $21 (the level entrance room): put the block at ($40, $80) without searching.
    ; No room in the game's level data has both layout $21 and the push block flag, so this
    ; branch never runs.
    JSR GetUniqueRoomId
    CMP #$21
    BNE @FindBlockTile
    LDA #$40
    STA ObjX+11
    ASL                         ; $80
    STA ObjY+11
    JMP @SetType

@FindBlockTile:
    ; Look for a block tile ($B0) in tile row $A, starting in tile column 4. X is the offset into
    ; PlayAreaColumnAddrs (2 bytes per tile column), stepped by one square (2 tile columns).
    LDX #$08
    LDY #$0A

@LoopColumn:
    LDA PlayAreaColumnAddrs, X
    STA $00
    LDA PlayAreaColumnAddrs+1, X
    STA $01
    LDA ($00), Y
    CMP #$B0
    BEQ @Found
    INX
    INX
    INX
    INX
    CPX #$34                    ; Stop before tile column 26.
    BNE @LoopColumn

@Found:
    ; X = tile column * 2, and a tile column is 8 pixels wide: ObjX = X * 4.
    ; (When nothing was found, X = $34 and ObjX = $D0.)
    TXA
    ASL
    ASL
    STA ObjX+11
    LDA #$90                    ; Tile row $A is at Y = $40 + $A * 8 = $90.
    STA ObjY+11

@SetType:
    LDA #$68                    ; Push block object type.
    STA ObjType+11
    RTS
