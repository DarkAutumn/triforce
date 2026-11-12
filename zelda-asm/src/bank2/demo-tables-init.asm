; ============================================================================
; FILE:     bank2/demo-tables-init.asm   BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 361-538
; SUBJECTS: Demo, Mode0Demo, Title
; TOPIC:    Data for the title screen and the story scroll (title sprites, per-line flags, item parade
;           lists, story text, palettes) and the mode 0 init subphases that set up each demo phase.
; SEE ALSO: bank2/mode0-demo.asm (Mode0Demo_Init dispatches the init subphases here);
;           bank2/demo-story-scroll.asm (reads Demo_LineAttrs, the item lists and the text);
;           bank2/title-animation.asm (copies Title_InitialSprites every frame);
;           bank6/transfer-buf-dispatch.asm (Transfer_BufAddrs: selectors $02 and $10 used here).
; KEY FACTS:
;   - Demo_LineAttrs has one byte per 8-pixel line of the story scroll (175 lines): bit $80 = the line
;     has text (next entry of Demo_LineTextAddrs), bit $40 = also send a row of 8 zero attribute bytes,
;     bit $20 = spawn the next row of the item parade.
;   - The item parade has 22 rows: Demo_LeftItemIds at X = $44 (Demo_LeftItemColumnX) and Demo_RightItemIds
;     at X = $AC (Demo_RightItemColumnX). Ids are item ids; $30-$33 are the four sprite rows of the closing
;     picture and have no right-hand item.
;   - A text field in Demo_TextFields is: column offset in the line, tile codes, $FF.
;     Demo_LineTextAddrs has 29 of them.
;   - Title init (phase 0) takes 3 frames: clear screen; palette record and variable reset (object
;     slots $01-$0A set to state $FF = free); title song ($80) and the title name tables (selector
;     $10). Mode0Demo_Init_TransferTitlePalette, Mode0Demo_Init_PlayTitleSong.
;   - Story init (phase 1) takes 3 frames: clear screen; story palette and variable reset; story name
;     tables (selector $02) with CurVScroll = $10 and the base name table switched.
;     Mode0Demo_Init_TransferStoryPalette, Mode0Demo_Init_TransferStoryTiles.
;   - The last init subphase of a phase sets IsUpdatingMode and DemoSubphase = 0. Mode0Demo_Init_End.
; CONTENTS: (generated; L = line in this file)
;   L0051 Title_InitialSprites                 data  TABLE: the 28 title screen sprites as OAM records (Y, tile, attributes, X), copied to Sprites every
;   L0069 Demo_LineAttrs                       data  TABLE: flags of each 8-pixel line of the story scroll; index = DemoLineIndex (0-174);
;   L0104 Demo_LeftItemIds                     data  TABLE: item shown in the left column of each item parade row; index = DemoItemRow (0-21);
;   L0111 Demo_RightItemIds                    data  TABLE: item shown in the right column of each item parade row; index = DemoItemRow (0-17);
;   L0117 Demo_LeftItemColumnX                 data  DATA: X coordinate ($44) of the left item parade column.
;   L0121 Demo_RightItemColumnX                data  DATA: X coordinate ($AC) of the right item parade column.
;   L0126 Demo_StoryFinalSpriteTiles           data  TABLE: sprite tiles of the closing picture; index = (item id - $30) * 6 + sprite 0-5 from the left;
;   L0132 Demo_StoryFinalSpriteAttrs           data  TABLE: sprite attributes of the closing picture; same index as Demo_StoryFinalSpriteTiles.
;   L0138 Demo_TextFields                      data  DATA: the story text, a list of fields: column offset in the line, tile codes, $FF.
;   L0142 Demo_LineTextAddrs                   data  TABLE: address of each story text field in Demo_TextFields; index = DemoLineTextIndex (0-28).
;   L0148 Mode0Demo_Init_ClearArtifacts        code  Init subphase 0 of both demo phases: hides sprites, clears the screen, then advances the subphase.
;   L0152 Mode0Demo_IncSubphase                code  Increments DemoSubphase and returns; shared tail of the demo init and animation subphases.
;   L0157 Title_PaletteTransferRecord          data  DATA: PPU transfer record with the 32 title screen palette bytes ($3F00, length $20, $FF end).
;   L0169 Mode0Demo_Init_TransferTitlePalette  code  Title init subphase 1: queues the title palette, zeroes the demo variables, frees slots $01-$0A.
;   L0202 Mode0Demo_Init_PlayTitleSong         code  Title init subphase 2: requests the title song ($80) and the title name tables (selector $10).
;   L0212 Demo_StoryPaletteTransferRecord      data  DATA: PPU transfer record with the 32 story screen palette bytes ($3F00, length $20, $FF end).
;   L0223 Mode0Demo_Init_TransferStoryPalette  code  Story init subphase 1: queues the story palette and zeroes the demo variables.
;   L0248 Mode0Demo_Init_TransferStoryTiles    code  Story init subphase 2: switches the base name table, sets CurVScroll = $10, cues the story tiles.
;   L0258 Mode0Demo_Init_End                   code  Shared end of both demo inits: A = TileBufSelector to cue; resets DemoSubphase, sets IsUpdatingMode.
; ============================================================================

; TABLE: the 28 title screen sprites as OAM records (Y, tile, attributes, X), copied to Sprites every
; frame of the title wait; $70 bytes.
Title_InitialSprites:
    .BYTE $77, $CA, $C2, $D0, $77, $CC, $C2, $C8
    .BYTE $77, $CA, $82, $28, $77, $CC, $82, $30
    .BYTE $27, $CA, $42, $D0, $27, $CC, $42, $C8
    .BYTE $27, $CA, $02, $28, $27, $CC, $02, $30
    .BYTE $57, $CE, $02, $74, $57, $D0, $02, $7C
    .BYTE $31, $D2, $02, $57, $4F, $D2, $02, $CC
    .BYTE $67, $D2, $02, $7B, $83, $D2, $02, $50
    .BYTE $31, $D4, $02, $5F, $3F, $D4, $02, $24
    .BYTE $41, $D4, $02, $64, $7B, $D4, $02, $90
    .BYTE $27, $D6, $02, $50, $2B, $D6, $02, $A0
    .BYTE $4F, $D6, $02, $2C, $7B, $D6, $02, $BC
    .BYTE $67, $A0, $03, $60, $67, $A0, $03, $68
    .BYTE $67, $A0, $03, $70, $67, $A0, $03, $78
    .BYTE $67, $A0, $03, $80, $67, $A0, $03, $88

; TABLE: flags of each 8-pixel line of the story scroll; index = DemoLineIndex (0-174);
; bit $80 = text line, $40 = clear a row of name table attributes, $20 = spawn an item parade row.
Demo_LineAttrs:
    .BYTE $80, $00, $00, $00, $00, $00, $00, $00
    .BYTE $40, $80, $80, $00, $60, $00, $00, $00
    .BYTE $40, $80, $00, $00, $60, $00, $00, $00
    .BYTE $00, $C0, $00, $00, $60, $00, $40, $00
    .BYTE $00, $C0, $00, $00, $20, $00, $40, $00
    .BYTE $00, $C0, $00, $00, $20, $00, $40, $00
    .BYTE $40, $80, $80, $00, $20, $00, $40, $00
    .BYTE $40, $80, $80, $00, $60, $00, $00, $00
    .BYTE $40, $80, $80, $00, $60, $00, $00, $00
    .BYTE $40, $80, $00, $00, $60, $00, $00, $00
    .BYTE $40, $80, $80, $00, $60, $00, $00, $00
    .BYTE $40, $C0, $80, $00, $20, $00, $40, $00
    .BYTE $40, $80, $80, $00, $20, $00, $40, $00
    .BYTE $40, $80, $80, $00, $20, $00, $40, $00
    .BYTE $40, $80, $00, $00, $20, $00, $40, $00
    .BYTE $40, $80, $80, $00, $60, $00, $00, $00
    .BYTE $40, $80, $80, $00, $60, $00, $00, $00
    .BYTE $00, $C0, $00, $00, $60, $00, $00, $00
    .BYTE $40, $00, $C0, $00, $00, $20, $40, $00
    .BYTE $00, $00, $40, $00, $00, $00, $00, $20
    .BYTE $00, $20, $00, $20, $00, $20, $00, $00
    .BYTE $00, $00, $00, $00, $00, $00, $00

; Padding: 51 zero bytes after Demo_LineAttrs; the scroll ends before DemoLineIndex reaches them.
    .BYTE $00, $00, $00, $00, $00, $00, $00, $00
    .BYTE $00, $00, $00, $00, $00, $00, $00, $00
    .BYTE $00, $00, $00, $00, $00, $00, $00, $00
    .BYTE $00, $00, $00, $00, $00, $00, $00, $00
    .BYTE $00, $00, $00, $00, $00, $00, $00, $00
    .BYTE $00, $00, $00, $00, $00, $00, $00, $00
    .BYTE $00, $00, $00

; TABLE: item shown in the left column of each item parade row; index = DemoItemRow (0-21);
; value = item id, or $30-$33 = a row of the closing picture (centred, no right-hand item).
Demo_LeftItemIds:
    .BYTE $22, $23, $18, $1F, $15, $01, $03, $1D
    .BYTE $00, $08, $06, $12, $14, $0C, $10, $19
    .BYTE $17, $1B, $30, $31, $32, $33

; TABLE: item shown in the right column of each item parade row; index = DemoItemRow (0-17);
; value = item id. In row 17 both columns hold $1B (Triforce piece) and are drawn at X = $78.
Demo_RightItemIds:
    .BYTE $1A, $21, $0F, $20, $04, $02, $1C, $1E
    .BYTE $0A, $09, $07, $13, $05, $0D, $11, $0B
    .BYTE $16, $1B

; DATA: X coordinate ($44) of the left item parade column.
Demo_LeftItemColumnX:
    .BYTE $44

; DATA: X coordinate ($AC) of the right item parade column.
Demo_RightItemColumnX:
    .BYTE $AC

; TABLE: sprite tiles of the closing picture; index = (item id - $30) * 6 + sprite 0-5 from the left;
; value = tile, 0 = no sprite at that position.
Demo_StoryFinalSpriteTiles:
    .BYTE $E0, $E2, $EC, $EE, $F8, $FA, $E4, $E6
    .BYTE $F0, $F2, $FC, $FE, $E8, $EA, $F4, $F6
    .BYTE $DC, $DE, $00, $00, $78, $78, $00, $00

; TABLE: sprite attributes of the closing picture; same index as Demo_StoryFinalSpriteTiles.
Demo_StoryFinalSpriteAttrs:
    .BYTE $00, $00, $00, $00, $00, $00, $00, $00
    .BYTE $00, $00, $00, $00, $00, $00, $00, $00
    .BYTE $00, $00, $00, $00, $00, $40, $00, $00

; DATA: the story text, a list of fields: column offset in the line, tile codes, $FF.
Demo_TextFields:
    .INCBIN "dat/DemoTextFields.dat"

; TABLE: address of each story text field in Demo_TextFields; index = DemoLineTextIndex (0-28).
Demo_LineTextAddrs:
    .INCLUDE "dat/DemoLineTextAddrs.inc"

; ----------------------------------------------------------------------------
; Init subphase 0 of both demo phases: hides sprites, clears the screen, then advances the subphase.
; ----------------------------------------------------------------------------
Mode0Demo_Init_ClearArtifacts:
    JSR Ppu_TurnOffVideoAndClearNameTables

; Increments DemoSubphase and returns; shared tail of the demo init and animation subphases.
Mode0Demo_IncSubphase:
    INC DemoSubphase
    RTS

; DATA: PPU transfer record with the 32 title screen palette bytes ($3F00, length $20, $FF end).
Title_PaletteTransferRecord:
    .BYTE $3F, $00, $20, $36, $0F, $00, $10, $36
    .BYTE $17, $27, $0F, $36, $08, $1A, $28, $36
    .BYTE $30, $3B, $22, $36, $30, $3B, $16, $36
    .BYTE $17, $27, $0F, $36, $08, $1A, $28, $36
    .BYTE $30, $3B, $22, $FF

; ----------------------------------------------------------------------------
; Title init subphase 1: queues the title palette, zeroes the demo variables, frees slots $01-$0A.
; OUT:  DynTileBuf holds the palette record; $0412-$041C, $041F-$0429, $0437-$0441, DemoLineTextIndex
;       and DemoItemRow = 0; ObjState of slots $01-$0A = $FF.
; ----------------------------------------------------------------------------
Mode0Demo_Init_TransferTitlePalette:
    LDX #$23                    ; Write a tile buf record for the title palette.
    STX $0300
    STX DynTileBufLen

@CopyTitlePalette:
    LDA Title_PaletteTransferRecord, X
    STA DynTileBuf, X
    DEX
    BPL @CopyTitlePalette
    LDX #$0A                    ; Reset variables used in this phase.
    LDA #$00
    STA DemoLineTextIndex
    STA DemoItemRow

@ClearVars:
    STA TriforceGlowTimer, X
    STA IsWaterfallInitialized, X
    STA DemoPhase0Subphase1Cycle, X
    DEX
    BPL @ClearVars
    LDX #$0A                    ; Mark objects 1 to 10 disabled.

@DisableObjects:
    LDA #$FF
    STA ObjState, X
    DEX
    BNE @DisableObjects
    JMP Mode0Demo_IncSubphase             ; Go advance the DemoSubphase and return.

; ----------------------------------------------------------------------------
; Title init subphase 2: requests the title song ($80) and the title name tables (selector $10).
; ----------------------------------------------------------------------------
Mode0Demo_Init_PlayTitleSong:
    LDA #$80                    ; Request the title song.
    STA SongRequest

    ; Select transfer buffer 8 (offset $10 in table):
    ; title nametables and attributes.
    LDA #$10
    JMP Mode0Demo_Init_End

; DATA: PPU transfer record with the 32 story screen palette bytes ($3F00, length $20, $FF end).
Demo_StoryPaletteTransferRecord:
    .BYTE $3F, $00, $20, $0F, $30, $30, $30, $0F
    .BYTE $21, $30, $30, $0F, $16, $30, $30, $0F
    .BYTE $29, $1A, $09, $0F, $29, $37, $17, $0F
    .BYTE $02, $22, $30, $0F, $16, $27, $30, $0F
    .BYTE $0B, $1B, $2B, $FF

; ----------------------------------------------------------------------------
; Story init subphase 1: queues the story palette and zeroes the demo variables.
; OUT:  DynTileBuf holds the palette record; $0412-$041C, $041F-$0429, $0437-$0441, $0444-$044E = 0.
; ----------------------------------------------------------------------------
Mode0Demo_Init_TransferStoryPalette:
    LDX #$23                    ; Copy the transfer record for the story palette.
    STX $0300
    STX DynTileBufLen

@CopyStoryPalette:
    LDA Demo_StoryPaletteTransferRecord, X
    STA DynTileBuf, X
    DEX
    BPL @CopyStoryPalette
    LDX #$0A                    ; Reset variables used in this phase.
    LDA #$00

@ClearVars:
    STA $0412, X
    STA $041F, X
    STA $0437, X
    STA $0444, X
    DEX
    BPL @ClearVars
    JMP Mode0Demo_IncSubphase             ; Go to the next subphase and return.

; ----------------------------------------------------------------------------
; Story init subphase 2: switches the base name table, sets CurVScroll = $10, cues the story tiles.
; ----------------------------------------------------------------------------
Mode0Demo_Init_TransferStoryTiles:
    INC SwitchNameTablesReq
    LDA #$10
    STA CurVScroll

    ; Select transfer buffer 1 (offset 2 in table):
    ; Story nametables and attributes.
    LDA #$02

; Shared end of both demo inits: A = TileBufSelector to cue; resets DemoSubphase, sets IsUpdatingMode.
Mode0Demo_Init_End:
    STA TileBufSelector
    LDA #$00
    STA DemoSubphase
    INC IsUpdatingMode
    RTS
