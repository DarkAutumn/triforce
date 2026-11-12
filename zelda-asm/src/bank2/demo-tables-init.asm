; ============================================================================
; FILE:     bank2/demo-tables-init.asm   BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 361-538
; SUBJECTS: (none)
; TOPIC:    Demo tables; init subphases, title/story palettes
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0030 InitialTitleSprites                   data
;   L0046 DemoLineAttrs                         data
;   L0079 DemoLeftItemIds                       data
;   L0084 DemoRightItemIds                      data
;   L0089 DemoItemColumnX1                      data
;   L0092 DemoItemColumnX2                      data
;   L0095 DemoStoryFinalSpriteTiles             data
;   L0100 DemoStoryFinalSpriteAttrs             data
;   L0105 DemoTextFields                        data
;   L0108 DemoLineTextAddrs                     data
;   L0111 InitDemoSubphaseClearArtifacts        code
;   L0114 IncSubphase                           code
;   L0118 TitlePaletteTransferRecord            data
;   L0125 InitDemoSubphaseTransferTitlePalette  code
;   L0155 InitDemoSubphasePlayTitleSong         code
;   L0164 StoryPaletteTransferRecord            data
;   L0171 InitDemoSubphaseTransferStoryPalette  code
;   L0193 InitDemoSubphaseTransferStoryTiles    code
;   L0202 EndInitDemo                           code
; ============================================================================

InitialTitleSprites:
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

DemoLineAttrs:
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

; Unknown block
    .BYTE $00, $00, $00, $00, $00, $00, $00, $00
    .BYTE $00, $00, $00, $00, $00, $00, $00, $00
    .BYTE $00, $00, $00, $00, $00, $00, $00, $00
    .BYTE $00, $00, $00, $00, $00, $00, $00, $00
    .BYTE $00, $00, $00, $00, $00, $00, $00, $00
    .BYTE $00, $00, $00, $00, $00, $00, $00, $00
    .BYTE $00, $00, $00

DemoLeftItemIds:
    .BYTE $22, $23, $18, $1F, $15, $01, $03, $1D
    .BYTE $00, $08, $06, $12, $14, $0C, $10, $19
    .BYTE $17, $1B, $30, $31, $32, $33

DemoRightItemIds:
    .BYTE $1A, $21, $0F, $20, $04, $02, $1C, $1E
    .BYTE $0A, $09, $07, $13, $05, $0D, $11, $0B
    .BYTE $16, $1B

DemoItemColumnX1:
    .BYTE $44

DemoItemColumnX2:
    .BYTE $AC

DemoStoryFinalSpriteTiles:
    .BYTE $E0, $E2, $EC, $EE, $F8, $FA, $E4, $E6
    .BYTE $F0, $F2, $FC, $FE, $E8, $EA, $F4, $F6
    .BYTE $DC, $DE, $00, $00, $78, $78, $00, $00

DemoStoryFinalSpriteAttrs:
    .BYTE $00, $00, $00, $00, $00, $00, $00, $00
    .BYTE $00, $00, $00, $00, $00, $00, $00, $00
    .BYTE $00, $00, $00, $00, $00, $40, $00, $00

DemoTextFields:
    .INCBIN "dat/DemoTextFields.dat"

DemoLineTextAddrs:
    .INCLUDE "dat/DemoLineTextAddrs.inc"

InitDemoSubphaseClearArtifacts:
    JSR TurnOffVideoAndClearArtifacts

IncSubphase:
    INC DemoSubphase
    RTS

TitlePaletteTransferRecord:
    .BYTE $3F, $00, $20, $36, $0F, $00, $10, $36
    .BYTE $17, $27, $0F, $36, $08, $1A, $28, $36
    .BYTE $30, $3B, $22, $36, $30, $3B, $16, $36
    .BYTE $17, $27, $0F, $36, $08, $1A, $28, $36
    .BYTE $30, $3B, $22, $FF

InitDemoSubphaseTransferTitlePalette:
    LDX #$23                    ; Write a tile buf record for the title palette.
    STX $0300
    STX DynTileBufLen

@CopyTitlePalette:
    LDA TitlePaletteTransferRecord, X
    STA DynTileBuf, X
    DEX
    BPL @CopyTitlePalette
    LDX #$0A                    ; Reset variables used in this phase.
    LDA #$00
    STA DemoLineTextIndex
    STA DemoItemRow

@ClearVars:
    STA TriforceGlowTimer, X
    STA InitializedWaterfallAnimation, X
    STA DemoPhase0Subphase1Cycle, X
    DEX
    BPL @ClearVars
    LDX #$0A                    ; Mark objects 1 to 10 disabled.

@DisableObjects:
    LDA #$FF
    STA ObjState, X
    DEX
    BNE @DisableObjects
    JMP IncSubphase             ; Go advance the DemoSubphase and return.

InitDemoSubphasePlayTitleSong:
    LDA #$80                    ; Request the title song.
    STA SongRequest

    ; Select transfer buffer 8 (offset $10 in table):
    ; title nametables and attributes.
    LDA #$10
    JMP EndInitDemo

StoryPaletteTransferRecord:
    .BYTE $3F, $00, $20, $0F, $30, $30, $30, $0F
    .BYTE $21, $30, $30, $0F, $16, $30, $30, $0F
    .BYTE $29, $1A, $09, $0F, $29, $37, $17, $0F
    .BYTE $02, $22, $30, $0F, $16, $27, $30, $0F
    .BYTE $0B, $1B, $2B, $FF

InitDemoSubphaseTransferStoryPalette:
    LDX #$23                    ; Copy the transfer record for the story palette.
    STX $0300
    STX DynTileBufLen

@CopyStoryPalette:
    LDA StoryPaletteTransferRecord, X
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
    JMP IncSubphase             ; Go to the next subphase and return.

InitDemoSubphaseTransferStoryTiles:
    INC SwitchNameTablesReq
    LDA #$10
    STA CurVScroll

    ; Select transfer buffer 1 (offset 2 in table):
    ; Story nametables and attributes.
    LDA #$02

EndInitDemo:
    STA TileBufSelector
    LDA #$00
    STA DemoSubphase
    INC IsUpdatingMode
    RTS
