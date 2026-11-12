; ============================================================================
; FILE:     bank6/level-blocks-common-data.asm BANK: 6 (switchable; BANK_06_DATA is copied to RAM $67F0)
; UPSTREAM: Z_06.asm 390-483
; SUBJECTS: Bank, Level, RoomLayout, Transfer
; TOPIC:    The level data in ROM: five 768-byte level blocks (room attributes of the overworld and of
;           the two dungeon maps, Q1 and Q2) and ten 252-byte level info blocks; then the common data
;           block, $8E bytes of patchable PPU transfer buffers and the overworld column directory that
;           live in RAM at $67F0-$687D.
; SEE ALSO: bank6/level-load-mode2.asm (the tables and code that copy these to RAM, Q2 patches);
;           Variables.inc (LevelBlockAttrsA-F bit layouts, LevelInfo_* fields); data/README.md and the
;           data/levelblock-*.md, data/levelinfo.md listings (every room and field decoded);
;           bank6/transfer-buf-dispatch.asm (transfer record format, selector table);
;           bank5/layout-room-ow.asm (RoomLayout_ColumnDirectoryOW users); bank5/submenu-states.asm (Triforce rows).
; KEY FACTS:
;   - A level block is six 128-byte arrays A-F, one byte per room id $00-$7F; the game only ever reads
;     the RAM copy at LevelBlockAttrsA-F. Level_BlockOW.
;   - Levels 1-6 share one 128-room block and levels 7-9 another; Q1 and Q2 have separate dungeon
;     blocks, the overworld has one block for both quests. Level_BlockUW1to6Q1, Level_BlockUW7to9Q1,
;     Level_BlockUW1to6Q2, Level_BlockUW7to9Q2.
;   - A level info block is 252 bytes; there is one per level and none per quest (Q2 patches the RAM
;     copy). Level_InfoOW, Level_InfoUW1.
;   - Overworld rooms whose array B cave index is 1-9 (the CurLevel loaded from there): $37 = 1,
;     $3C = 2, $74 = 3, $45 = 4, $0B and $1B = 5, $22 and $30 = 6, $42 and $19 = 7, $6D and $6C = 8,
;     $05 and $00 = 9. Of each pair the second has array F bits 7-6 = 2 (secret works in Q2 only);
;     $6D and $05 have 1 (Q1 only). Level_BlockOW.
;   - Labels from Transfer_MenuPalettesBuf on are linked at their RAM addresses ($67F0 up, segment
;     BANK_06_DATA); the bytes sit in ROM at Bank_CommonDataBlock and are copied once at power-on or
;     reset.
;   - The common data is in RAM because code patches it: Link's colour per save slot and the mode E/F
;     cursor colour in Transfer_MenuPalettesBuf, the level's sprite colours in
;     Transfer_LevelPaletteRow7Buf, the digit in Transfer_LevelNumberBuf, entry 0 of RoomLayout_ColumnDirectoryOW
;     (cellars), the piece tiles in Transfer_TriforceRow0Buf..Transfer_TriforceRow3Buf.
;   - Transfer_TriforceTextBuf has no end marker inside the block: in RAM the next byte is
;     LevelBlockAttrsA[0]. It is selected only by the overworld submenu, where that byte is $A3 (bit 7
;     set), which ends the transfer.
; CONTENTS: (generated; L = line in this file)
;   L0066 Level_BlockOW                 data  DATA: overworld level block, both quests (768 bytes: arrays A-F x 128 rooms).
;   L0070 Level_BlockUW1to6Q1           data  DATA: level block of levels 1-6, Q1 (768 bytes). Listing: data/levelblock-uw-1to6-q1.md.
;   L0074 Level_BlockUW7to9Q1           data  DATA: level block of levels 7-9, Q1 (768 bytes). Listing: data/levelblock-uw-7to9-q1.md.
;   L0078 Level_BlockUW1to6Q2           data  DATA: level block of levels 1-6, Q2 (768 bytes). Listing: data/levelblock-uw-1to6-q2.md.
;   L0082 Level_BlockUW7to9Q2           data  DATA: level block of levels 7-9, Q2 (768 bytes). Listing: data/levelblock-uw-7to9-q2.md.
;   L0086 Level_InfoOW                  data  DATA: level info of the overworld (252 bytes, LevelInfo_* layout). Listing: data/levelinfo.md.
;   L0090 Level_InfoUW1                 data  DATA: level info of level 1 (252 bytes); in Q2 the RAM copy is patched from offset $29 on.
;   L0094 Level_InfoUW2                 data  DATA: level info of level 2 (252 bytes); patched in Q2 like level 1.
;   L0098 Level_InfoUW3                 data  DATA: level info of level 3 (252 bytes); patched in Q2 like level 1.
;   L0102 Level_InfoUW4                 data  DATA: level info of level 4 (252 bytes); patched in Q2 like level 1.
;   L0106 Level_InfoUW5                 data  DATA: level info of level 5 (252 bytes); patched in Q2 like level 1.
;   L0110 Level_InfoUW6                 data  DATA: level info of level 6 (252 bytes); patched in Q2 like level 1.
;   L0114 Level_InfoUW7                 data  DATA: level info of level 7 (252 bytes); patched in Q2 like level 1.
;   L0118 Level_InfoUW8                 data  DATA: level info of level 8 (252 bytes); patched in Q2 like level 1.
;   L0122 Level_InfoUW9                 data  DATA: level info of level 9 (252 bytes); patched in Q2 like level 1.
;   L0127 Bank_CommonDataBlock          data  DATA: start of the common data block in ROM ($8E bytes at $9CD8, right after the level 9 info).
;   L0146 Transfer_MenuPalettesBuf      data  DATA: transfer buffer (selector $12): all 32 palette bytes to $3F00, as used by the file menus.
;   L0157 Transfer_LevelPaletteRow7Buf  data  DATA: transfer buffer (selector $06): 4 bytes to $3F1C, sprite palette row 7.
;   L0162 Transfer_LevelNumberBuf       data  DATA: transfer buffer (selectors $0C and $2E): the text "LEVEL-n" at $2042 in the status bar.
;   L0171 RoomLayout_ColumnDirectoryOW  data  TABLE: overworld column directory; index = heap number 0-15 (* 2); value = heap address (low, high).
;   L0180 Transfer_TriforceRow0Buf      data  DATA: transfer buffer (selector $52): row 0 of the submenu Triforce outline, 4 tiles at $2AEE.
;   L0184 Transfer_TriforceRow1Buf      data  DATA: transfer buffer (selector $54): row 1 of the Triforce outline, 6 tiles at $2B0D.
;   L0189 Transfer_TriforceRow2Buf      data  DATA: transfer buffer (selector $56): row 2 of the Triforce outline, 8 tiles at $2B2C.
;   L0194 Transfer_TriforceRow3Buf      data  DATA: transfer buffer (selector $58): row 3 of the Triforce outline, 10 tiles at $2B4B.
;   L0202 Transfer_TriforceTextBuf      data  DATA: transfer buffer (selector $5C): the text "TRIFORCE" at $2BAC (overworld submenu).
; ============================================================================

; DATA: overworld level block, both quests (768 bytes: arrays A-F x 128 rooms).
; In Q2 Mode2Load_Update_Full rewrites 15 bytes of the RAM copy. Listing: data/levelblock-ow.md.
Level_BlockOW:
    .INCBIN "dat/LevelBlockOW.dat"

; DATA: level block of levels 1-6, Q1 (768 bytes). Listing: data/levelblock-uw-1to6-q1.md.
Level_BlockUW1to6Q1:
    .INCBIN "dat/LevelBlockUW1Q1.dat"

; DATA: level block of levels 7-9, Q1 (768 bytes). Listing: data/levelblock-uw-7to9-q1.md.
Level_BlockUW7to9Q1:
    .INCBIN "dat/LevelBlockUW2Q1.dat"

; DATA: level block of levels 1-6, Q2 (768 bytes). Listing: data/levelblock-uw-1to6-q2.md.
Level_BlockUW1to6Q2:
    .INCBIN "dat/LevelBlockUW1Q2.dat"

; DATA: level block of levels 7-9, Q2 (768 bytes). Listing: data/levelblock-uw-7to9-q2.md.
Level_BlockUW7to9Q2:
    .INCBIN "dat/LevelBlockUW2Q2.dat"

; DATA: level info of the overworld (252 bytes, LevelInfo_* layout). Listing: data/levelinfo.md.
Level_InfoOW:
    .INCBIN "dat/LevelInfoOW.dat"

; DATA: level info of level 1 (252 bytes); in Q2 the RAM copy is patched from offset $29 on.
Level_InfoUW1:
    .INCBIN "dat/LevelInfoUW1.dat"

; DATA: level info of level 2 (252 bytes); patched in Q2 like level 1.
Level_InfoUW2:
    .INCBIN "dat/LevelInfoUW2.dat"

; DATA: level info of level 3 (252 bytes); patched in Q2 like level 1.
Level_InfoUW3:
    .INCBIN "dat/LevelInfoUW3.dat"

; DATA: level info of level 4 (252 bytes); patched in Q2 like level 1.
Level_InfoUW4:
    .INCBIN "dat/LevelInfoUW4.dat"

; DATA: level info of level 5 (252 bytes); patched in Q2 like level 1.
Level_InfoUW5:
    .INCBIN "dat/LevelInfoUW5.dat"

; DATA: level info of level 6 (252 bytes); patched in Q2 like level 1.
Level_InfoUW6:
    .INCBIN "dat/LevelInfoUW6.dat"

; DATA: level info of level 7 (252 bytes); patched in Q2 like level 1.
Level_InfoUW7:
    .INCBIN "dat/LevelInfoUW7.dat"

; DATA: level info of level 8 (252 bytes); patched in Q2 like level 1.
Level_InfoUW8:
    .INCBIN "dat/LevelInfoUW8.dat"

; DATA: level info of level 9 (252 bytes); patched in Q2 like level 1.
Level_InfoUW9:
    .INCBIN "dat/LevelInfoUW9.dat"

; DATA: start of the common data block in ROM ($8E bytes at $9CD8, right after the level 9 info).
; Bank_CopyCommonDataToRam copies the bytes below to $67F0-$687D at power-on or reset.
Bank_CommonDataBlock:

.SEGMENT "BANK_06_DATA"


.EXPORT RoomLayout_ColumnDirectoryOW
.EXPORT Transfer_LevelNumberBuf
.EXPORT Transfer_LevelPaletteRow7Buf
.EXPORT Transfer_MenuPalettesBuf
.EXPORT Transfer_TriforceRow0Buf


; Everything below is linked to run from RAM at $67F0-$687D. Transfer record format: PPU address high,
; low, control (bit 7 = step 32, bit 6 = repeat one byte, bits 5-0 = count), data; $FF ends a buffer.

; DATA: transfer buffer (selector $12): all 32 palette bytes to $3F00, as used by the file menus.
; Patched in RAM: +20 + 4 * save slot = colour 1 of sprite palette rows 4-6 (Link's colour for save
; slots 0-2, by ring; Mode1Menu_Init_Sub1Palettes, Item_TakeClass2Grade); +32 = colour 1 of sprite row 7, the cursor colour
; set by the mode E/F init ($15 in mode $E, $30 in mode $F; ModeERegister_Init_Full).
Transfer_MenuPalettesBuf:
    .BYTE $3F, $00, $20, $0F, $30, $00, $12, $0F    ; $3F00, 32 bytes; background rows 0-3 ...
    .BYTE $16, $27, $36, $0F, $0C, $1C, $2C, $0F
    .BYTE $12, $1C, $2C, $0F, $29, $27, $07, $0F    ; ... sprite rows 4-7 start at +19
    .BYTE $22, $27, $07, $0F, $26, $27, $07, $0F
    .BYTE $15, $27, $30, $FF

; DATA: transfer buffer (selector $06): 4 bytes to $3F1C, sprite palette row 7.
; The 4 colours at +3 are overwritten with row 7 of the level's palettes
; (LevelInfo_PalettesTransferBuf+31) when a dungeon room is entered whose object in slot 1 is not one
; of the bosses with a palette row of their own (Mode5Play_Init). Overworld rooms never select it.
Transfer_LevelPaletteRow7Buf:
    .BYTE $3F, $1C, $04, $0F, $0F, $0F, $0F, $FF

; DATA: transfer buffer (selectors $0C and $2E): the text "LEVEL-n" at $2042 in the status bar.
; The digit at +9 is overwritten with LevelInfo_LevelNumber when a dungeon is unfurled (Mode3Unfurl_Init_Sub7_CueLevelNumber).
Transfer_LevelNumberBuf:
    .BYTE $20, $42, $07, $15, $0E, $1F, $0E, $15    ; $2042, 7 tiles: L E V E L
    .BYTE $62, $00, $FF                             ; dash ($62), digit (patched), end

; TABLE: overworld column directory; index = heap number 0-15 (* 2); value = heap address (low, high).
; The heap number is the high nibble of a column descriptor in a room layout. The addresses are those
; of RoomLayout_ColumnHeapOW0 ($9BD8) .. RoomLayout_ColumnHeapOWF ($9F55) in bank 5. Entry 0 is rewritten in RAM each time a
; room is laid out: to RoomLayout_ColumnHeapUWCellar when CurLevel is not 0, to RoomLayout_ColumnHeapOW0 in the overworld
; (RoomLayout_PatchColumnDirectoryForCellar).
RoomLayout_ColumnDirectoryOW:
    .BYTE $D8, $9B, $0D, $9C, $3E, $9C, $80, $9C
    .BYTE $C4, $9C, $F6, $9C, $32, $9D, $6D, $9D
    .BYTE $A8, $9D, $E6, $9D, $27, $9E, $6C, $9E
    .BYTE $A9, $9E, $DF, $9E, $21, $9F, $55, $9F

; DATA: transfer buffer (selector $52): row 0 of the submenu Triforce outline, 4 tiles at $2AEE.
; Rows 0-3 must stay contiguous: the submenu writes the tiles of held Triforce pieces at offsets
; $00-$25 from Transfer_TriforceRow0Buf+4, which reach into all four buffers (Submenu_TriforceTileOffsets).
Transfer_TriforceRow0Buf:
    .BYTE $2A, $EE, $04, $ED, $E9, $EA, $EE, $FF

; DATA: transfer buffer (selector $54): row 1 of the Triforce outline, 6 tiles at $2B0D.
Transfer_TriforceRow1Buf:
    .BYTE $2B, $0D, $06, $ED, $E9, $24, $24, $EA
    .BYTE $EE, $FF

; DATA: transfer buffer (selector $56): row 2 of the Triforce outline, 8 tiles at $2B2C.
Transfer_TriforceRow2Buf:
    .BYTE $2B, $2C, $08, $ED, $E9, $24, $24, $24
    .BYTE $24, $EA, $EE, $FF

; DATA: transfer buffer (selector $58): row 3 of the Triforce outline, 10 tiles at $2B4B.
Transfer_TriforceRow3Buf:
    .BYTE $2B, $4B, $0A, $ED, $E9, $24, $24, $24
    .BYTE $24, $24, $24, $EA, $EE, $FF

; DATA: transfer buffer (selector $5C): the text "TRIFORCE" at $2BAC (overworld submenu).
; NOTE: no $FF end marker: the RAM copy ends here at $687D, so the transfer loop next reads $687E =
; LevelBlockAttrsA[0] as an address byte. The selector is set only by Submenu_CueTriforceText; the overworld
; block's first byte is $A3, which has bit 7 set and so ends the transfer.
Transfer_TriforceTextBuf:
    .BYTE $2B, $AC, $08, $1D, $1B, $12, $0F, $18
    .BYTE $1B, $0C, $0E
