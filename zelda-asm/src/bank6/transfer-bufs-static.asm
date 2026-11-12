; ============================================================================
; FILE:     bank6/transfer-bufs-static.asm BANK: 6 (switchable, $8000-$BFFF)
; UPSTREAM: Z_06.asm 655-874
; SUBJECTS: (none)
; TOPIC:    The fixed PPU transfer buffers in ROM: screen text, box frames, attribute fills and palette
;           rows for the file menus, status bar, submenu, caves, cellars, death and ending; then the
;           bank's copy of the reset code and its interrupt vectors.
; SEE ALSO: bank6/transfer-buf-dispatch.asm (TransferBufAddrs gives each buffer its selector; record
;           format and the transfer loop); bank6/level-blocks-common-data.asm (the buffers that live in
;           RAM because they are patched); bank7/init-mode5-play.asm (which room gets which sprite
;           palette row 7); bank7/reset-mmc1-vectors.asm (the reset code this bank's tail copies).
; KEY FACTS:
;   - Every buffer is a list of records: PPU address high, low; control (bit 7 = step 32, bit 6 = repeat
;     one byte, bits 5-0 = count, 0 = 64); data; a $FF where an address is expected ends the buffer.
;     Mode1TileTransferBuf.
;   - Text bytes are tile numbers: $00-$09 digits, $0A-$23 letters A-Z, $24 blank, $62 dash.
;     Mode8TextTileBuffer.
;   - Records to $3F1C replace sprite palette row 7 (the last of the four sprite palettes); a room's
;     boss or tile object picks one: GanonPaletteRow7TransferBuf, AquamentusPaletteRow7TransferBuf,
;     OrangeBossPaletteRow7TransferBuf (Digdogger, Dodongo), GleeokPaletteRow7TransferBuf,
;     GhostPaletteRow7TransferBuf, GreenBgPaletteRow7TransferBuf, BrownBgPaletteRow7TransferBuf,
;     RedArmosPaletteRow7TransferBuf.
;   - Records to $3F08 replace background palette rows 2 and 3: CaveBgPaletteRowsTransferBuf,
;     WhitePaletteBottomHalfTransferBuf, Mode11BackgroundPaletteBottomHalfTransferBuf.
;   - The price texts are fixed: "-100" (BombCapacityPriceTextTransferBuf) and "-1" / "-50"
;     (LifeOrMoneyCostTextTransferBuf).
;   - The unlabeled bytes in segments BANK_06_ISR ($BF50) and BANK_06_VEC ($BFFA) are a copy of the
;     fixed bank's reset code (IsrReset, SetMMC1Control, SwitchBank; only the two JSR targets differ,
;     $BFxx for $FFxx) and a vector table: NMI $E484, reset $BF50, IRQ $BFF0. Whether the copy can
;     ever run is (unverified).
; CONTENTS: (generated; L = line in this file)
;   L0079 Mode1TileTransferBuf                          data  DATA: transfer buffer (selector $14): the file select screen.
;   L0103 ModeFCharsTransferBuf                         data  DATA: transfer buffer (selector $16): the register/eliminate character board (modes $E and $F).
;   L0122 GanonPaletteRow7TransferBuf                   data  DATA: transfer buffer (selector $36): sprite palette row 7 ($3F1C) for Ganon: $0F $16 $2C $3C.
;   L0127 EndingPaletteTransferBuf                      data  DATA: transfer buffer (selector $6A): the ending palettes.
;   L0133 BlankTextBoxLines                             data  DATA: transfer buffer (selector $1E): 24 blank tiles at $21A4 and at $21C4 (person text lines 1, 2).
;   L0138 BlankPersonWares                              data  DATA: transfer buffer (selector $2A): 24 blank tiles at $21E4 (text line 3), 13 at $22C8 (prices).
;   L0143 SubmenuTriforceApexTransferBuf                data  DATA: transfer buffer (selector $50): apex of the submenu Triforce outline, 2 tiles at $2ACF.
;   L0147 SubmenuTriforceBottomTransferBuf              data  DATA: transfer buffer (selector $5A): bottom of the submenu Triforce outline, 12 tiles at $2B6A.
;   L0154 GhostPaletteRow7TransferBuf                   data  DATA: transfer buffer (selector $20): sprite palette row 7 = $0F $30 $00 $12.
;   L0159 GreenBgPaletteRow7TransferBuf                 data  DATA: transfer buffer (selector $22): sprite palette row 7 = $0F $1A $37 $12.
;   L0164 BrownBgPaletteRow7TransferBuf                 data  DATA: transfer buffer (selector $24): sprite palette row 7 = $0F $17 $37 $12.
;   L0168 CaveBgPaletteRowsTransferBuf                  data  DATA: transfer buffer (selector $3E): background palette rows 2-3 ($3F08, 8 bytes) for caves.
;   L0173 CellarAttrsTransferBuf                        data  DATA: transfer buffer (selector $26): cellar attributes: $AA x 32 at $23D0 and x 16 at $23F0.
;   L0178 WhitePaletteBottomHalfTransferBuf             data  DATA: transfer buffer (selector $78): background palette rows 2-3 ($3F08) all white ($30).
;   L0185 RedArmosPaletteRow7TransferBuf                data  DATA: transfer buffer (selector $7A): sprite palette row 7 = $0F $0F $1C $16.
;   L0189 GleeokPaletteRow7TransferBuf                  data  DATA: transfer buffer (selector $7C): sprite palette row 7 = $0F $2A $1A $0C, for Gleeok ($43-$45).
;   L0193 AquamentusPaletteRow7TransferBuf              data  DATA: transfer buffer (selector $08): sprite palette row 7 = $0F $0A $29 $30, for Aquamentus ($3D).
;   L0198 OrangeBossPaletteRow7TransferBuf              data  DATA: transfer buffer (selector $0A): sprite palette row 7 = $0F $17 $27 $30.
;   L0202 BombCapacityPriceTextTransferBuf              data  DATA: transfer buffer (selector $6C): the text "-100" at $22CD (price of the more-bombs offer).
;   L0206 LifeOrMoneyCostTextTransferBuf                data  DATA: transfer buffer (selector $76): "-1", five blanks, "-50" at $22CB (life-or-money man).
;   L0212 Mode8TextTileBuffer                           data  DATA: transfer buffer (selector $04): the continue menu.
;   L0222 StatusBarStaticsTransferBuf                   data  DATA: transfer buffer (selector $0E): the fixed parts of the status bar.
;   L0234 InventoryTextTransferBuf                      data  DATA: transfer buffer (selector $30): the text "INVENTORY" at $2984.
;   L0240 SubmenuBoxesTopsTransferBuf                   data  DATA: transfer buffer (selector $32): tops of the two submenu boxes.
;   L0247 SubmenuBoxesSidesTransferBuf                  data  DATA: transfer buffer (selector $34): sides of the two submenu boxes.
;   L0253 SelectedItemBoxBottomTransferBuf              data  DATA: transfer buffer (selector $38): bottom of the selected-item box, 4 tiles at $2A27.
;   L0257 UseBButtonTextTransferBuf                     data  DATA: transfer buffer (selector $3A): the text "USE B BUTTON" at $2A42.
;   L0263 InventoryBoxBottomTransferBuf                 data  DATA: transfer buffer (selector $3C): the text "FOR THIS" at $2A64 and the inventory box bottom.
;   L0270 SubmenuMapRemainderTransferBuf                data  DATA: transfer buffer (selector $40): dungeon submenu texts and the top edge of the map sheet.
;   L0278 SheetMapBottomEdgeTransferBuf                 data  DATA: transfer buffer (selector $42): bottom edge of the dungeon map sheet, 16 tiles at $2BAC.
;   L0284 SubmenuAttrs1TransferBuf                      data  DATA: transfer buffer (selector $48): submenu attributes 1: $05 x 3 at $2BD9, $00 x 11 at $2BDC.
;   L0289 SubmenuAttrs2TransferBuf                      data  DATA: transfer buffer (selector $4A): submenu attributes, part 2: $55 x 22 at $2BE9.
;   L0293 BlankBottomRowNT2TransferBuf                  data  DATA: transfer buffer (selector $4C): 32 blank tiles at $2BA0 (a row of the second name table).
;   L0297 BlankRowTransferBuf                           data  DATA: transfer buffer (selector $4E): 32 blank tiles at $28E0.
;   L0301 Mode11DeadLinkPalette                         data  DATA: transfer buffer (selector $2C): dead Link's palette (sprite row 4, $3F10) = $0F $10 $30 $00.
;   L0306 GameOverTransferBuf                           data  DATA: transfer buffer (selector $46): the death screen's "GAME OVER " text at $224C.
;   L0313 Mode11BackgroundPaletteBottomHalfTransferBuf  data  DATA: transfer buffer (selector $5E): death: background palette rows 2-3 ($3F08).
;   L0318 Mode11PlayAreaAttrsTopHalfTransferBuf         data  DATA: transfer buffer (selector $60): death: attributes $FF x 24 at $23D0 (upper play area).
;   L0322 Mode11PlayAreaAttrsBottomHalfTransferBuf      data  DATA: transfer buffer (selector $62): death: attributes $FF x 24 at $23E8 (lower play area).
;   L0326 StoryTileAttrTransferBuf                      data  DATA: transfer buffer (selector $02): story screen tiles and attributes (1131-byte blob).
;   L0330 GameTitleTransferBuf                          data  DATA: transfer buffer (selector $10): title screen tiles and attributes (1121-byte blob).
; ============================================================================

; DATA: transfer buffer (selector $14): the file select screen.
; Zeroes 63 attribute bytes from $23C0, sets 3 attribute bytes each at $23D4/$23DC/$23E4, writes
; "- S E L E C T -" ($20A8), the box frame ($2103-$233C), " NAME " ($210A), " LIFE " ($2113),
; "REGISTER YOUR NAME" ($22A6), "ELIMINATION MODE" ($22E6).
Mode1TileTransferBuf:
    .BYTE $23, $C0, $7F, $00, $23, $D4, $03, $40
    .BYTE $50, $50, $23, $DC, $03, $44, $55, $55
    .BYTE $23, $E4, $03, $44, $55, $55, $20, $A8
    .BYTE $0F, $62, $24, $1C, $24, $0E, $24, $15
    .BYTE $24, $0E, $24, $0C, $24, $1D, $24, $62
    .BYTE $21, $03, $01, $69, $21, $04, $58, $6A
    .BYTE $21, $1C, $01, $6B, $21, $23, $D0, $6C
    .BYTE $21, $3C, $D0, $6C, $23, $23, $01, $6E
    .BYTE $23, $24, $58, $6A, $23, $3C, $01, $6D
    .BYTE $21, $0A, $06, $24, $17, $0A, $16, $0E
    .BYTE $24, $21, $13, $06, $24, $15, $12, $0F
    .BYTE $0E, $24, $22, $A6, $12, $1B, $0E, $10
    .BYTE $12, $1C, $1D, $0E, $1B, $24, $22, $18
    .BYTE $1E, $1B, $24, $17, $0A, $16, $0E, $22
    .BYTE $E6, $10, $0E, $15, $12, $16, $12, $17
    .BYTE $0A, $1D, $12, $18, $17, $24, $16, $18
    .BYTE $0D, $0E, $FF

; DATA: transfer buffer (selector $16): the register/eliminate character board (modes $E and $F).
; Box frame ($2205-$231B) and four rows of characters with blanks between: A-K ($2226), L-V ($2266),
; W-Z then tiles $62 (dash), $63, $28-$2C ($22A6), 0-9 ($22E6). In the person text character map
; (data/README.md) $28-$2C are , ! ' & and full stop; the glyph of $63 is pattern data, not decoded
; here (unverified).
ModeFCharsTransferBuf:
    .BYTE $22, $05, $01, $69, $22, $06, $55, $6A
    .BYTE $22, $1B, $01, $6B, $22, $25, $C7, $6C
    .BYTE $22, $3B, $C7, $6C, $23, $05, $01, $6E
    .BYTE $23, $06, $55, $6A, $23, $1B, $01, $6D
    .BYTE $22, $26, $15, $0A, $24, $0B, $24, $0C
    .BYTE $24, $0D, $24, $0E, $24, $0F, $24, $10
    .BYTE $24, $11, $24, $12, $24, $13, $24, $14
    .BYTE $22, $66, $15, $15, $24, $16, $24, $17
    .BYTE $24, $18, $24, $19, $24, $1A, $24, $1B
    .BYTE $24, $1C, $24, $1D, $24, $1E, $24, $1F
    .BYTE $22, $A6, $15, $20, $24, $21, $24, $22
    .BYTE $24, $23, $24, $62, $24, $63, $24, $28
    .BYTE $24, $29, $24, $2A, $24, $2B, $24, $2C
    .BYTE $22, $E6, $13, $00, $24, $01, $24, $02
    .BYTE $24, $03, $24, $04, $24, $05, $24, $06
    .BYTE $24, $07, $24, $08, $24, $09, $FF

; DATA: transfer buffer (selector $36): sprite palette row 7 ($3F1C) for Ganon: $0F $16 $2C $3C.
GanonPaletteRow7TransferBuf:
    .BYTE $3F, $1C, $04, $0F, $16, $2C, $3C, $FF

; DATA: transfer buffer (selector $6A): the ending palettes.
; Background rows 2-3 ($3F08, 8 bytes) and sprite row 7 ($3F1C, 4 bytes).
EndingPaletteTransferBuf:
    .BYTE $3F, $08, $08, $0F, $22, $10, $00, $0F
    .BYTE $2A, $10, $00, $3F, $1C, $04, $0F, $27
    .BYTE $06, $16, $FF

; DATA: transfer buffer (selector $1E): 24 blank tiles at $21A4 and at $21C4 (person text lines 1, 2).
BlankTextBoxLines:
    .BYTE $21, $A4, $58, $24, $21, $C4, $58, $24
    .BYTE $FF

; DATA: transfer buffer (selector $2A): 24 blank tiles at $21E4 (text line 3), 13 at $22C8 (prices).
BlankPersonWares:
    .BYTE $21, $E4, $58, $24, $22, $C8, $4D, $24
    .BYTE $FF

; DATA: transfer buffer (selector $50): apex of the submenu Triforce outline, 2 tiles at $2ACF.
SubmenuTriforceApexTransferBuf:
    .BYTE $2A, $CF, $02, $ED, $EE, $FF

; DATA: transfer buffer (selector $5A): bottom of the submenu Triforce outline, 12 tiles at $2B6A.
SubmenuTriforceBottomTransferBuf:
    .BYTE $2B, $6A, $0C, $EB, $EF, $F1, $F1, $F1
    .BYTE $F1, $F1, $F1, $F1, $F1, $F0, $EC, $FF

; DATA: transfer buffer (selector $20): sprite palette row 7 = $0F $30 $00 $12.
; Used for a gravestone tile object ($65) and for armos tile object $66 when the room's inner palette
; is odd (InitMode5Play).
GhostPaletteRow7TransferBuf:
    .BYTE $3F, $1C, $04, $0F, $30, $00, $12, $FF

; DATA: transfer buffer (selector $22): sprite palette row 7 = $0F $1A $37 $12.
; Used for a rock ($62) or armos $66 tile object when the room's inner palette is even (InitMode5Play).
GreenBgPaletteRow7TransferBuf:
    .BYTE $3F, $1C, $04, $0F, $1A, $37, $12, $FF

; DATA: transfer buffer (selector $24): sprite palette row 7 = $0F $17 $37 $12.
; Used for a rock tile object ($62) when the room's inner palette is odd (InitMode5Play).
BrownBgPaletteRow7TransferBuf:
    .BYTE $3F, $1C, $04, $0F, $17, $37, $12, $FF

; DATA: transfer buffer (selector $3E): background palette rows 2-3 ($3F08, 8 bytes) for caves.
CaveBgPaletteRowsTransferBuf:
    .BYTE $3F, $08, $08, $0F, $30, $00, $12, $0F
    .BYTE $07, $0F, $17, $FF

; DATA: transfer buffer (selector $26): cellar attributes: $AA x 32 at $23D0 and x 16 at $23F0.
CellarAttrsTransferBuf:
    .BYTE $23, $D0, $60, $AA, $23, $F0, $50, $AA
    .BYTE $FF

; DATA: transfer buffer (selector $78): background palette rows 2-3 ($3F08) all white ($30).
WhitePaletteBottomHalfTransferBuf:
    .BYTE $3F, $08, $08, $0F, $30, $30, $30, $0F
    .BYTE $30, $30, $30, $FF

; DATA: transfer buffer (selector $7A): sprite palette row 7 = $0F $0F $1C $16.
; Used in every overworld room whose tile object is not a rock ($62), gravestone ($65) or armos $66,
; including rooms with no tile object (InitMode5Play).
RedArmosPaletteRow7TransferBuf:
    .BYTE $3F, $1C, $04, $0F, $0F, $1C, $16, $FF

; DATA: transfer buffer (selector $7C): sprite palette row 7 = $0F $2A $1A $0C, for Gleeok ($43-$45).
GleeokPaletteRow7TransferBuf:
    .BYTE $3F, $1C, $04, $0F, $2A, $1A, $0C, $FF

; DATA: transfer buffer (selector $08): sprite palette row 7 = $0F $0A $29 $30, for Aquamentus ($3D).
AquamentusPaletteRow7TransferBuf:
    .BYTE $3F, $1C, $04, $0F, $0A, $29, $30, $FF

; DATA: transfer buffer (selector $0A): sprite palette row 7 = $0F $17 $27 $30.
; Used for Digdogger ($38, $39) and Dodongo ($31, $32) (SpecialBossPaletteObjTypes).
OrangeBossPaletteRow7TransferBuf:
    .BYTE $3F, $1C, $04, $0F, $17, $27, $30, $FF

; DATA: transfer buffer (selector $6C): the text "-100" at $22CD (price of the more-bombs offer).
BombCapacityPriceTextTransferBuf:
    .BYTE $22, $CD, $04, $62, $01, $00, $00, $FF

; DATA: transfer buffer (selector $76): "-1", five blanks, "-50" at $22CB (life-or-money man).
LifeOrMoneyCostTextTransferBuf:
    .BYTE $22, $CB, $0A, $62, $01, $24, $24, $24
    .BYTE $24, $24, $62, $05, $00, $FF

; DATA: transfer buffer (selector $04): the continue menu.
; Zeroes 63 attribute bytes from $23C0 and writes "CONTINUE" ($214A), "SAVE" ($21AA), "RETRY" ($220A).
Mode8TextTileBuffer:
    .BYTE $23, $C0, $7F, $00, $21, $4A, $08, $0C
    .BYTE $18, $17, $1D, $12, $17, $1E, $0E, $21
    .BYTE $AA, $04, $1C, $0A, $1F, $0E, $22, $0A
    .BYTE $05, $1B, $0E, $1D, $1B, $22, $FF

; DATA: transfer buffer (selector $0E): the fixed parts of the status bar.
; Attributes ($23C2, 14 bytes), the B and A box tops and "-LIFE-" ($206F), box bottoms ($20CF), box
; sides ($208F, $2091, $2092, $2094, 2 tiles down each), and 4 tiles down from $206B: $F7 left of the
; rupee count ($206C), blank, $F9 left of the key count ($20AC), $61 left of the bomb count ($20CC).
StatusBarStaticsTransferBuf:
    .BYTE $23, $C2, $0E, $40, $00, $00, $44, $55
    .BYTE $55, $00, $00, $04, $00, $00, $44, $55
    .BYTE $55, $20, $6F, $0E, $69, $0B, $6B, $69
    .BYTE $0A, $6B, $24, $24, $62, $15, $12, $0F
    .BYTE $0E, $62, $20, $CF, $06, $6E, $6A, $6D
    .BYTE $6E, $6A, $6D, $20, $8F, $C2, $6C, $20
    .BYTE $91, $C2, $6C, $20, $92, $C2, $6C, $20
    .BYTE $94, $C2, $6C, $20, $6B, $84, $F7, $24
    .BYTE $F9, $61, $FF

; DATA: transfer buffer (selector $30): the text "INVENTORY" at $2984.
InventoryTextTransferBuf:
    .BYTE $29, $84, $09, $12, $17, $1F, $0E, $17
    .BYTE $1D, $18, $1B, $22, $FF

; DATA: transfer buffer (selector $32): tops of the two submenu boxes.
; Selected-item box: 4 tiles at $29C7. Inventory box: $29CF-$29DB.
SubmenuBoxesTopsTransferBuf:
    .BYTE $29, $C7, $04, $69, $6A, $6A, $6B, $29
    .BYTE $CF, $01, $69, $29, $D0, $4B, $6A, $29
    .BYTE $DB, $01, $6B, $FF

; DATA: transfer buffer (selector $34): sides of the two submenu boxes.
; Written downwards from $29E7, $29EA (2 tiles each) and $29EF, $29FB (4 tiles each).
SubmenuBoxesSidesTransferBuf:
    .BYTE $29, $E7, $C2, $6C, $29, $EA, $C2, $6C
    .BYTE $29, $EF, $C4, $6C, $29, $FB, $C4, $6C
    .BYTE $FF

; DATA: transfer buffer (selector $38): bottom of the selected-item box, 4 tiles at $2A27.
SelectedItemBoxBottomTransferBuf:
    .BYTE $2A, $27, $04, $6E, $6A, $6A, $6D, $FF

; DATA: transfer buffer (selector $3A): the text "USE B BUTTON" at $2A42.
UseBButtonTextTransferBuf:
    .BYTE $2A, $42, $0C, $1E, $1C, $0E, $24, $0B
    .BYTE $24, $0B, $1E, $1D, $1D, $18, $17, $FF

; DATA: transfer buffer (selector $3C): the text "FOR THIS" at $2A64 and the inventory box bottom.
; The box bottom is at $2A6F-$2A7B.
InventoryBoxBottomTransferBuf:
    .BYTE $2A, $64, $08, $0F, $18, $1B, $24, $1D
    .BYTE $11, $12, $1C, $2A, $6F, $01, $6E, $2A
    .BYTE $70, $4B, $6A, $2A, $7B, $01, $6D, $FF

; DATA: transfer buffer (selector $40): dungeon submenu texts and the top edge of the map sheet.
; "COMPASS" ($2B43), "MAP" ($2AA5), 16 edge tiles at $2A8C.
SubmenuMapRemainderTransferBuf:
    .BYTE $2B, $43, $07, $0C, $18, $16, $19, $0A
    .BYTE $1C, $1C, $2A, $A5, $03, $16, $0A, $19
    .BYTE $2A, $8C, $10, $F5, $F5, $FD, $F5, $F5
    .BYTE $FD, $F5, $F5, $FD, $F5, $F5, $F5, $FD
    .BYTE $F5, $F5, $F5, $FF

; DATA: transfer buffer (selector $42): bottom edge of the dungeon map sheet, 16 tiles at $2BAC.
SheetMapBottomEdgeTransferBuf:
    .BYTE $2B, $AC, $10, $F5, $FE, $F5, $F5, $F5
    .BYTE $FE, $F5, $F5, $F5, $F5, $FE, $F5, $F5
    .BYTE $F5, $FE, $F5, $FF

; DATA: transfer buffer (selector $48): submenu attributes 1: $05 x 3 at $2BD9, $00 x 11 at $2BDC.
SubmenuAttrs1TransferBuf:
    .BYTE $2B, $D9, $43, $05, $2B, $DC, $4B, $00
    .BYTE $FF

; DATA: transfer buffer (selector $4A): submenu attributes, part 2: $55 x 22 at $2BE9.
SubmenuAttrs2TransferBuf:
    .BYTE $2B, $E9, $56, $55, $FF

; DATA: transfer buffer (selector $4C): 32 blank tiles at $2BA0 (a row of the second name table).
BlankBottomRowNT2TransferBuf:
    .BYTE $2B, $A0, $60, $24, $FF

; DATA: transfer buffer (selector $4E): 32 blank tiles at $28E0.
BlankRowTransferBuf:
    .BYTE $28, $E0, $60, $24, $FF

; DATA: transfer buffer (selector $2C): dead Link's palette (sprite row 4, $3F10) = $0F $10 $30 $00.
Mode11DeadLinkPalette:
    .BYTE $3F, $10, $04, $0F, $10, $30, $00, $FF

; DATA: transfer buffer (selector $46): the death screen's "GAME OVER " text at $224C.
; Also 3 attribute bytes at $23E3 and 10 blank tiles at $226C.
GameOverTransferBuf:
    .BYTE $23, $E3, $03, $0F, $0F, $CF, $22, $4C
    .BYTE $0A, $10, $0A, $16, $0E, $24, $18, $1F
    .BYTE $0E, $1B, $24, $22, $6C, $4A, $24, $FF

; DATA: transfer buffer (selector $5E): death: background palette rows 2-3 ($3F08).
; Both rows become $0F $17 $16 $26.
Mode11BackgroundPaletteBottomHalfTransferBuf:
    .BYTE $3F, $08, $08, $0F, $17, $16, $26, $0F
    .BYTE $17, $16, $26, $FF

; DATA: transfer buffer (selector $60): death: attributes $FF x 24 at $23D0 (upper play area).
Mode11PlayAreaAttrsTopHalfTransferBuf:
    .BYTE $23, $D0, $58, $FF, $FF

; DATA: transfer buffer (selector $62): death: attributes $FF x 24 at $23E8 (lower play area).
Mode11PlayAreaAttrsBottomHalfTransferBuf:
    .BYTE $23, $E8, $58, $FF, $FF

; DATA: transfer buffer (selector $02): story screen tiles and attributes (1131-byte blob).
StoryTileAttrTransferBuf:
    .INCBIN "dat/StoryTileAttrTransferBuf.dat"

; DATA: transfer buffer (selector $10): title screen tiles and attributes (1121-byte blob).
GameTitleTransferBuf:
    .INCBIN "dat/GameTitleTransferBuf.dat"

.SEGMENT "BANK_06_ISR"




; DATA: this bank's copy of the reset handler, at $BF50-$BFBF (unlabeled; 112 bytes of code).
; The same instructions as IsrReset, SetMMC1Control and SwitchBank in bank 7, with the two JSR
; targets local ($BF98, $BFAC instead of $FF98, $FFAC):
;   $BF50  SEI, CLD, PPU control = 0, stack = $FF, wait for two vertical blanks,
;          write $FF to $8000/$A000/$C000/$E000 (reset the MMC1 shift register),
;          A = $0F, JSR $BF98 (MMC1 control), write 0 to $A000 five times (CHR bank 0),
;          A = 7, JSR $BFAC (switch to bank 7), JMP $E440 (RunGame, as in bank 7's IsrReset)
;   $BF98  five serial writes of A to $8000 (copy of SetMMC1Control), RTS
;   $BFAC  five serial writes of A to $E000 (copy of SwitchBank), RTS
; The vector and the JSRs name $BFxx addresses (this bank mapped at $8000-$BFFF), while the CPU
; reads its vectors at $FFFA-$FFFF; whether the copy can ever run is (unverified).
    .BYTE $78, $D8, $A9, $00, $8D, $00, $20, $A2
    .BYTE $FF, $9A, $AD, $02, $20, $29, $80, $F0
    .BYTE $F9, $AD, $02, $20, $29, $80, $F0, $F9
    .BYTE $09, $FF, $8D, $00, $80, $8D, $00, $A0
    .BYTE $8D, $00, $C0, $8D, $00, $E0, $A9, $0F
    .BYTE $20, $98, $BF, $A9, $00, $8D, $00, $A0
    .BYTE $4A, $8D, $00, $A0, $4A, $8D, $00, $A0
    .BYTE $4A, $8D, $00, $A0, $4A, $8D, $00, $A0
    .BYTE $A9, $07, $20, $AC, $BF, $4C, $40, $E4
    .BYTE $8D, $00, $80, $4A, $8D, $00, $80, $4A
    .BYTE $8D, $00, $80, $4A, $8D, $00, $80, $4A
    .BYTE $8D, $00, $80, $60, $8D, $00, $E0, $4A
    .BYTE $8D, $00, $E0, $4A, $8D, $00, $E0, $4A
    .BYTE $8D, $00, $E0, $4A, $8D, $00, $E0, $60

.SEGMENT "BANK_06_VEC"




; DATA: interrupt vectors at $BFFA: NMI $E484, reset $BF50 (the copy above), IRQ $BFF0.
; $E484 is IsrNmi: bank 7's own vector table (IsrVector) holds the same NMI address. $BFF0 lies in
; the $FF fill between the reset copy and these vectors.
    .BYTE $84, $E4, $50, $BF, $F0, $BF
