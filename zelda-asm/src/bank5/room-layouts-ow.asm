; ============================================================================
; FILE:     bank5/room-layouts-ow.asm    BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 4228-4408
; SUBJECTS: RoomLayout
; TOPIC:    Data only: the overworld room layouts (16 column descriptors per layout), the layouts of
;           the cave and the shortcut-stairs cave, and the 16 column tables ("heaps") of square
;           descriptors that the layouts refer to.
; SEE ALSO: bank5/layout-room-ow.asm (RoomLayout_LayOutRoomOW and RoomLayout_LayOutRoomOrCaveOW decode all of this;
;           RoomLayout_OWSquareTiles and RoomLayout_OWSecondarySquareTiles turn a square index into tiles; RoomLayout_SubroomLayoutAddrs);
;           bank6/level-blocks-common-data.asm (RoomLayout_ColumnDirectoryOW: the addresses of the 16 heaps);
;           bank5/room-layouts-uw.asm (the dungeon and cellar counterparts);
;           bank6/level-load-mode2.asm (the Q2 patch that gives rooms $0B and $3C layout $7B).
; KEY FACTS:
;   - A layout is 16 bytes, one per column of 16-pixel squares, left to right. Layout n is at
;     RoomLayout_RoomsOW + 16 * n; n = bits 6-0 of the room's LevelBlockAttrsD byte. RoomLayout_RoomsOW
;   - Column descriptor: high nibble = heap number (entry of RoomLayout_ColumnDirectoryOW), low nibble = column
;     number in that heap. Column k starts at the (k+1)-th byte of the heap that has bit 7 set.
;     RoomLayout_ColumnHeapOW0
;   - Square descriptor: bits 5-0 = square index, bit 6 = the square fills two rows, bit 7 = a
;     column may start here. A column is read for 11 rows and runs on through later bytes whatever
;     their bit 7, so columns share bytes. RoomLayout_ColumnHeapOW0
;   - Square index $10 and up selects RoomLayout_OWSquareTiles[index] (first of 4 consecutive tiles); below
;     $10 it selects 4 tiles of RoomLayout_OWSecondarySquareTiles. RoomLayout_ColumnHeapOW0
;   - RoomLayout_RoomsOW is 1936 bytes = layouts $00-$78. Ids $79, $7A and $7B fall on the three 16-byte
;     tables that follow it. RoomLayout_CaveOW, RoomLayout_ShortcutCaveOW, RoomLayout_RoomOW7B
;   - Caves use RoomLayout_CaveOW; the shortcut-stairs cave uses RoomLayout_ShortcutCaveOW.
;     RoomLayout_CaveOW, RoomLayout_ShortcutCaveOW
;   - RoomLayout_RoomOW7B is not a cave: it is overworld layout $7B, used only in Q2 (rooms $0B and
;     $3C). RoomLayout_RoomOW7B
; CONTENTS: (generated; L = line in this file)
;   L0058 RoomLayout_RoomsOW           data  DATA: overworld room layouts, 1936 bytes = 121 layouts ($00-$78) of 16 column descriptors each
;   L0063 RoomLayout_CaveOW            data  TABLE: layout of an overworld cave (mode $B); index = column 0-15; value = column descriptor
;   L0069 RoomLayout_ShortcutCaveOW    data  TABLE: layout of the shortcut-stairs cave (mode $C); index = column 0-15; value = column
;   L0077 RoomLayout_RoomOW7B          data  TABLE: overworld room layout $7B; index = column 0-15; value = column descriptor.
;   L0089 RoomLayout_ColumnHeapOW0     data  TABLE: column heap 0 (53 bytes, 9 column starts); value = square descriptor.
;   L0099 RoomLayout_ColumnHeapOW1     data  TABLE: column heap 1 (49 bytes, 10 column starts); value = square descriptor.
;   L0109 RoomLayout_ColumnHeapOW2     data  TABLE: column heap 2 (66 bytes, 10 column starts); value = square descriptor.
;   L0121 RoomLayout_ColumnHeapOW3     data  TABLE: column heap 3 (68 bytes, 10 column starts); value = square descriptor.
;   L0133 RoomLayout_ColumnHeapOW4     data  TABLE: column heap 4 (50 bytes, 10 column starts); value = square descriptor.
;   L0143 RoomLayout_ColumnHeapOW5     data  TABLE: column heap 5 (60 bytes, 9 column starts); value = square descriptor.
;   L0154 RoomLayout_ColumnHeapOW6     data  TABLE: column heap 6 (59 bytes, 9 column starts); value = square descriptor.
;   L0165 RoomLayout_ColumnHeapOW7     data  TABLE: column heap 7 (59 bytes, 9 column starts); value = square descriptor.
;   L0176 RoomLayout_ColumnHeapOW8     data  TABLE: column heap 8 (62 bytes, 9 column starts); value = square descriptor.
;   L0187 RoomLayout_ColumnHeapOW9     data  TABLE: column heap 9 (65 bytes, 9 column starts); value = square descriptor.
;   L0199 RoomLayout_ColumnHeapOWA     data  TABLE: column heap $A (69 bytes, 10 column starts); value = square descriptor.
;   L0211 RoomLayout_ColumnHeapOWB     data  TABLE: column heap $B (61 bytes, 9 column starts); value = square descriptor.
;   L0222 RoomLayout_ColumnHeapOWC     data  TABLE: column heap $C (54 bytes, 9 column starts); value = square descriptor.
;   L0232 RoomLayout_ColumnHeapOWD     data  TABLE: column heap $D (66 bytes, 9 column starts); value = square descriptor.
;   L0244 RoomLayout_ColumnHeapOWE     data  TABLE: column heap $E (52 bytes, 9 column starts); value = square descriptor.
;   L0254 RoomLayout_ColumnHeapOWF     data  TABLE: column heap $F (71 bytes, 10 column starts); value = square descriptor.
;   L0266 RoomLayout_RoomsOWAddr       data  DATA: address of RoomLayout_RoomsOW, read by RoomLayout_LayOutRoomOW as the base of the layout table.
;   L0271 RoomLayout_ColumnHeapOWAddr  data  DATA: address of column heap 0, which RoomLayout_PatchColumnDirectoryForCellar writes to entry 0 of
; ============================================================================

; DATA: overworld room layouts, 1936 bytes = 121 layouts ($00-$78) of 16 column descriptors each
; (high nibble = heap number, low nibble = column number in the heap). Layout id = LevelBlockAttrsD
; bits 6-0. Read through RoomLayout_RoomsOWAddr by RoomLayout_LayOutRoomOW.
RoomLayout_RoomsOW:
    .INCBIN "dat/RoomLayoutsOW.dat"

; TABLE: layout of an overworld cave (mode $B); index = column 0-15; value = column descriptor
; (high nibble heap, low nibble column). Entry 0 of RoomLayout_SubroomLayoutAddrs. Also layout id $79.
RoomLayout_CaveOW:
    .BYTE $00, $00, $95, $95, $95, $95, $95, $C2
    .BYTE $C2, $95, $95, $95, $95, $95, $00, $00

; TABLE: layout of the shortcut-stairs cave (mode $C); index = column 0-15; value = column
; descriptor. Entry 1 of RoomLayout_SubroomLayoutAddrs. Also layout id $7A.
RoomLayout_ShortcutCaveOW:
    .BYTE $00, $00, $95, $95, $95, $F8, $95, $C2
    .BYTE $F8, $95, $95, $F8, $95, $95, $00, $00

; TABLE: overworld room layout $7B; index = column 0-15; value = column descriptor.
; NOTE: Not a cave layout despite the name and no code names this label: RoomLayout_LayOutRoomOW reaches it as
;       RoomLayout_RoomsOW + 16 * $7B. Only the Q2 patch of the overworld block assigns id $7B (rooms $0B
;       and $3C; Mode2Load_Update_Full@PatchQ2Rooms).
RoomLayout_RoomOW7B:
    .BYTE $00, $A9, $64, $66, $02, $53, $54, $D1
    .BYTE $54, $54, $56, $02, $64, $66, $A8, $00

; The 16 tables below are the column heaps. RoomLayout_ColumnDirectoryOW holds their addresses in this order.
; Each byte is a square descriptor: bits 5-0 square index (RoomLayout_OWSquareTiles; below $10
; RoomLayout_OWSecondarySquareTiles), bit 6 = the square fills two rows, bit 7 = start of a column. RoomLayout_LayOutRoomOrCaveOW
; skips to the wanted bit-7 byte and then reads 11 rows' worth of descriptors, ignoring bit 7.

; TABLE: column heap 0 (53 bytes, 9 column starts); value = square descriptor.
; NOTE: In dungeons RoomLayout_PatchColumnDirectoryForCellar points directory entry 0 at the cellar heap
;       instead; RoomLayout_ColumnHeapOWAddr is what restores it in the overworld.
RoomLayout_ColumnHeapOW0:
    .BYTE $DB, $5B, $5B, $DB, $5B, $1B, $0E, $1A
    .BYTE $5B, $DB, $4E, $4E, $4E, $0E, $1A, $1B
    .BYTE $DB, $4E, $32, $1B, $34, $4E, $1A, $1B
    .BYTE $DB, $5B, $1B, $4E, $4E, $4E, $DB, $0E
    .BYTE $32, $5B, $1B, $4E, $1A, $1B, $DB, $5B
    .BYTE $1B, $4E, $4E, $1A, $1B, $DB, $CE, $4E
    .BYTE $4E, $4E, $0E, $1A, $1B

; TABLE: column heap 1 (49 bytes, 10 column starts); value = square descriptor.
RoomLayout_ColumnHeapOW1:
    .BYTE $9B, $0C, $4E, $4E, $4E, $0E, $1A, $1B
    .BYTE $CA, $4E, $0E, $4E, $4E, $4A, $C5, $45
    .BYTE $05, $0B, $C5, $45, $05, $45, $45, $45
    .BYTE $D9, $28, $59, $D9, $59, $59, $19, $D9
    .BYTE $59, $4E, $0E, $59, $D9, $4E, $4E, $4E
    .BYTE $0E, $59, $8E, $4E, $4E, $CE, $4E, $4E
    .BYTE $0E

; TABLE: column heap 2 (66 bytes, 10 column starts); value = square descriptor.
RoomLayout_ColumnHeapOW2:
    .BYTE $D9, $59, $0E, $4E, $4E, $4E, $DB, $0E
    .BYTE $5B, $5B, $4E, $1A, $1B, $DB, $0E, $33
    .BYTE $5B, $1B, $4E, $1A, $1B, $CE, $4E, $4E
    .BYTE $4E, $0E, $59, $DB, $4E, $33, $1B, $35
    .BYTE $4E, $1A, $1B, $D9, $4E, $19, $0E, $19
    .BYTE $4E, $D9, $0E, $19, $0E, $19, $0E, $19
    .BYTE $0E, $D9, $4E, $0E, $19, $0E, $4E, $D9
    .BYTE $19, $4E, $4E, $0E, $19, $D9, $0E, $59
    .BYTE $59, $59

; TABLE: column heap 3 (68 bytes, 10 column starts); value = square descriptor.
RoomLayout_ColumnHeapOW3:
    .BYTE $D9, $0E, $59, $59, $19, $0E, $D9, $4E
    .BYTE $19, $0E, $28, $4E, $59, $DB, $4E, $13
    .BYTE $0E, $13, $4E, $1A, $1B, $DB, $0E, $13
    .BYTE $0E, $13, $0E, $13, $0E, $1A, $1B, $DB
    .BYTE $4E, $0E, $13, $4E, $0E, $1A, $1B, $C8
    .BYTE $48, $17, $4E, $4E, $1A, $1B, $C5, $45
    .BYTE $07, $4E, $4E, $1A, $1B, $C5, $45, $07
    .BYTE $0E, $1A, $DB, $5B, $1B, $4E, $4E, $4A
    .BYTE $DB, $4E, $4E, $15

; TABLE: column heap 4 (50 bytes, 10 column starts); value = square descriptor.
RoomLayout_ColumnHeapOW4:
    .BYTE $C8, $48, $17, $4E, $4E, $59, $C9, $49
    .BYTE $18, $4E, $4E, $D9, $59, $4E, $06, $C5
    .BYTE $45, $45, $C9, $49, $09, $0B, $49, $49
    .BYTE $09, $C8, $48, $08, $48, $C8, $48, $17
    .BYTE $4E, $4E, $4E, $DB, $4E, $4E, $06, $C5
    .BYTE $45, $07, $4E, $4E, $D9, $4E, $2C, $4E
    .BYTE $4E, $59

; TABLE: column heap 5 (60 bytes, 9 column starts); value = square descriptor.
RoomLayout_ColumnHeapOW5:
    .BYTE $DB, $5B, $1B, $0E, $4A, $4A, $0A, $DB
    .BYTE $5B, $4E, $06, $C5, $45, $07, $4E, $4E
    .BYTE $4E, $DB, $0E, $15, $48, $17, $4E, $1A
    .BYTE $1B, $DB, $0E, $06, $45, $07, $4E, $1A
    .BYTE $1B, $DB, $0E, $06, $05, $45, $45, $45
    .BYTE $DB, $0E, $16, $49, $18, $4E, $1A, $1B
    .BYTE $D9, $0E, $59, $59, $19, $0E, $CE, $4E
    .BYTE $4E, $06, $45, $45

; TABLE: column heap 6 (59 bytes, 9 column starts); value = square descriptor.
RoomLayout_ColumnHeapOW6:
    .BYTE $D9, $4E, $4E, $16, $49, $49, $D9, $4E
    .BYTE $4E, $06, $C5, $45, $07, $4E, $D9, $59
    .BYTE $4E, $15, $48, $48, $DB, $0E, $1C, $1F
    .BYTE $4E, $4E, $1A, $1B, $DB, $0E, $1D, $0C
    .BYTE $4E, $4E, $1A, $1B, $DB, $0E, $1E, $20
    .BYTE $4E, $4E, $1A, $1B, $DB, $0E, $21, $24
    .BYTE $4E, $4E, $1A, $1B, $DB, $0E, $22, $0C
    .BYTE $4E, $4E, $4A

; TABLE: column heap 7 (59 bytes, 9 column starts); value = square descriptor.
RoomLayout_ColumnHeapOW7:
    .BYTE $DB, $0E, $23, $25, $4E, $4E, $1A, $1B
    .BYTE $DB, $4E, $0E, $26, $4E, $0E, $1A, $1B
    .BYTE $DB, $5B, $07, $4E, $1A, $1B, $DB, $5B
    .BYTE $07, $4E, $4E, $1A, $1B, $CE, $0E, $59
    .BYTE $59, $19, $0E, $D9, $4E, $0E, $26, $4E
    .BYTE $0E, $D9, $4E, $4E, $4E, $4E, $0E, $DB
    .BYTE $4E, $19, $0E, $19, $4E, $1A, $1B, $DB
    .BYTE $5B, $5B, $45

; TABLE: column heap 8 (62 bytes, 9 column starts); value = square descriptor.
RoomLayout_ColumnHeapOW8:
    .BYTE $C5, $05, $5B, $5B, $DB, $5B, $27, $0E
    .BYTE $1A, $5B, $DB, $0E, $14, $0E, $14, $0E
    .BYTE $14, $4E, $0E, $CA, $4A, $0A, $4E, $4E
    .BYTE $1A, $1B, $DB, $1B, $35, $4E, $4E, $0E
    .BYTE $1A, $1B, $9B, $27, $4E, $4E, $06, $C5
    .BYTE $45, $05, $2F, $4E, $4E, $0E, $DB, $0E
    .BYTE $22, $0C, $4E, $4E, $4A, $DB, $0E, $14
    .BYTE $0E, $29, $0E, $14, $4E, $0E

; TABLE: column heap 9 (65 bytes, 9 column starts); value = square descriptor.
RoomLayout_ColumnHeapOW9:
    .BYTE $DB, $35, $4E, $4E, $4E, $1A, $1B, $DB
    .BYTE $1B, $34, $4E, $06, $C5, $45, $05, $2E
    .BYTE $4E, $4E, $0E, $CE, $0E, $14, $0E, $14
    .BYTE $0E, $14, $0E, $1A, $1B, $CE, $0E, $14
    .BYTE $0E, $14, $0E, $14, $4E, $0E, $DB, $4D
    .BYTE $4D, $4D, $0D, $1A, $1B, $C5, $05, $1B
    .BYTE $0C, $4E, $0E, $4A, $0A, $DB, $1B, $35
    .BYTE $4E, $06, $45, $C5, $4E, $4E, $4E, $0E
    .BYTE $45

; TABLE: column heap $A (69 bytes, 10 column starts); value = square descriptor.
RoomLayout_ColumnHeapOWA:
    .BYTE $DB, $5B, $34, $4E, $4E, $1A, $1B, $DB
    .BYTE $5B, $35, $4E, $4E, $1A, $1B, $9B, $35
    .BYTE $4E, $4E, $4E, $0E, $33, $1B, $9B, $34
    .BYTE $4E, $4E, $4E, $0E, $32, $1B, $9B, $34
    .BYTE $4E, $4E, $4E, $0E, $1A, $1B, $9B, $35
    .BYTE $4E, $4E, $4E, $0E, $1A, $1B, $DB, $5B
    .BYTE $34, $0E, $32, $DB, $5B, $35, $0E, $33
    .BYTE $5B, $DB, $34, $4E, $4E, $0E, $32, $DB
    .BYTE $35, $4E, $4E, $0E, $33

; TABLE: column heap $B (61 bytes, 9 column starts); value = square descriptor.
RoomLayout_ColumnHeapOWB:
    .BYTE $DB, $5B, $07, $4E, $4E, $4E, $DB, $0E
    .BYTE $06, $05, $45, $2F, $0E, $1A, $1B, $DB
    .BYTE $0E, $06, $05, $45, $2E, $0E, $1A, $1B
    .BYTE $CA, $4A, $0A, $0E, $1A, $DB, $5B, $15
    .BYTE $45, $45, $45, $DB, $4E, $4E, $4E, $0E
    .BYTE $CA, $4E, $4E, $4E, $0E, $1A, $1B, $DB
    .BYTE $4E, $2C, $0E, $2C, $4E, $1A, $1B, $CA
    .BYTE $4A, $4E, $06, $45, $45

; TABLE: column heap $C (54 bytes, 9 column starts); value = square descriptor.
RoomLayout_ColumnHeapOWC:
    .BYTE $9B, $35, $4E, $0E, $13, $4E, $0E, $1A
    .BYTE $1B, $F6, $76, $0F, $4E, $1A, $1B, $5B
    .BYTE $DB, $4D, $4D, $4D, $4D, $0D, $CE, $4E
    .BYTE $0E, $30, $45, $45, $05, $DB, $F7, $F7
    .BYTE $77, $77, $37, $77, $1A, $1B, $DB, $1B
    .BYTE $34, $4E, $4E, $0E, $1A, $1B, $DB, $34
    .BYTE $0E, $4E, $4E, $0E, $1A, $1B

; TABLE: column heap $D (66 bytes, 9 column starts); value = square descriptor.
RoomLayout_ColumnHeapOWD:
    .BYTE $C5, $45, $0B, $4E, $4E, $4E, $DB, $0E
    .BYTE $06, $45, $07, $4E, $4E, $9B, $27, $4E
    .BYTE $4E, $4E, $0E, $1A, $1B, $9B, $27, $4E
    .BYTE $4E, $4E, $4E, $0E, $DB, $1B, $27, $07
    .BYTE $4E, $1A, $5B, $1B, $CE, $4E, $4E, $4E
    .BYTE $33, $5B, $DB, $5B, $1B, $0E, $15, $48
    .BYTE $48, $DB, $0E, $2D, $0C, $4E, $4E, $4A
    .BYTE $DB, $0E, $1C, $1F, $0E, $1E, $20, $0E
    .BYTE $1A, $1B

; TABLE: column heap $E (52 bytes, 9 column starts); value = square descriptor.
RoomLayout_ColumnHeapOWE:
    .BYTE $C8, $48, $08, $C5, $45, $45, $08, $48
    .BYTE $48, $DB, $35, $CE, $4E, $4E, $4E, $0E
    .BYTE $33, $1B, $DB, $0E, $33, $5B, $27, $4E
    .BYTE $1A, $1B, $DB, $4E, $4E, $1C, $1F, $0E
    .BYTE $1A, $1B, $DB, $5B, $1B, $35, $4E, $33
    .BYTE $DB, $5B, $1B, $34, $4E, $32, $DB, $4E
    .BYTE $4E, $4E, $0E, $33

; TABLE: column heap $F (71 bytes, 10 column starts); value = square descriptor.
RoomLayout_ColumnHeapOWF:
    .BYTE $9B, $5B, $34, $4E, $0E, $32, $DB, $1B
    .BYTE $35, $4E, $0E, $33, $1B, $DB, $0E, $1E
    .BYTE $20, $0E, $1C, $1F, $0E, $1A, $1B, $DB
    .BYTE $4E, $4E, $1E, $20, $0E, $1A, $1B, $D9
    .BYTE $59, $19, $4E, $4E, $59, $DB, $0E, $2D
    .BYTE $0C, $4E, $4E, $1A, $1B, $CE, $4E, $0E
    .BYTE $31, $05, $C5, $45, $07, $4E, $28, $59
    .BYTE $19, $DB, $4D, $4D, $12, $4D, $1A, $1B
    .BYTE $DB, $77, $77, $77, $37, $1A, $1B

; DATA: address of RoomLayout_RoomsOW, read by RoomLayout_LayOutRoomOW as the base of the layout table.
RoomLayout_RoomsOWAddr:
    .ADDR RoomLayout_RoomsOW

; DATA: address of column heap 0, which RoomLayout_PatchColumnDirectoryForCellar writes to entry 0 of
; RoomLayout_ColumnDirectoryOW in the overworld.
RoomLayout_ColumnHeapOWAddr:
    .ADDR RoomLayout_ColumnHeapOW0
