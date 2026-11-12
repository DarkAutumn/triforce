; ============================================================================
; FILE:     bank2/title-animation.asm    BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 1003-1348
; SUBJECTS: (none)
; TOPIC:    Title triforce glow, waterfall, palette fade
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0024 TriforcePaletteTransferRecord        data
;   L0027 TriforceGlowingColors                data
;   L0030 AnimateDemoPhase0Subphase0Artifacts  code
;   L0074 WaterfallWaveTiles                   data
;   L0077 WaterfallCrestTiles                  data
;   L0080 WaterfallSpriteXs                    data
;   L0083 WaterfallWaveSpriteOffsets           data
;   L0089 UpdateWaterfallAnimation             code
;   L0119 UpdateSpritesForWaterfallWave        code
;   L0173 UpdateSpritesForWaterfallCrest       code
;   L0198 DemoPhase0Subphase1Palettes          data
;   L0266 DemoPhase0Subphase1Delays            data
;   L0273 AnimateDemoPhase0Subphase1           code
; ============================================================================

TriforcePaletteTransferRecord:
    .BYTE $3F, $04, $04, $36, $17, $27, $0F, $FF

TriforceGlowingColors:
    .BYTE $27, $37, $37, $27, $17, $07, $07, $17

AnimateDemoPhase0Subphase0Artifacts:
    ; Set up sprites for the title.
    ; Copy initial sprite data to Sprites area.
    LDY #$70

@CopySprites:
    LDA InitialTitleSprites-1, Y
    STA Sprites-1, Y
    DEY
    BNE @CopySprites
    JSR UpdateWaterfallAnimation

    ; When you reach the end of the glow cycle,
    ; Append a transfer record for the triforce palette.
    LDA TriforceGlowTimer
    BNE @DecGlowTimer
    LDY #$07

@CopyTriforcePalette:
    LDA TriforcePaletteTransferRecord, Y
    STA DynTileBuf, Y
    DEY
    BPL @CopyTriforcePalette

    ; Patch the palette record with the color
    ; for the current point in the cycle.
    LDY TriforceGlowCycle
    LDA TriforceGlowingColors, Y
    STA DynTileBuf+5
    LDA #$06                    ; Restart the glow timer.
    STA TriforceGlowTimer
    INC TriforceGlowCycle       ; Advance the glow cycle.
    LDA TriforceGlowCycle
    CMP #$08                    ; When the glow cycle finishes,
    BNE @DecGlowTimer
    LDA #$10                    ; delay twice as long for one step of the cycle.
    STA TriforceGlowTimer
    LDA #$00                    ; Restart the glow cycle.
    STA TriforceGlowCycle

@DecGlowTimer:
    DEC TriforceGlowTimer
    RTS

WaterfallWaveTiles:
    .BYTE $B2, $B4, $B6, $B8

WaterfallCrestTiles:
    .BYTE $A2, $A4, $A6, $A8

WaterfallSpriteXs:
    .BYTE $50, $58, $60, $68

WaterfallWaveSpriteOffsets:
    .BYTE $70, $80, $90

; Unknown block
    .BYTE $A0, $B0, $C0, $D0, $E0

UpdateWaterfallAnimation:
    LDA InitializedWaterfallAnimation
    BNE @UpdateSprites
    LDA #$B6                    ; Initialize animation values.
    STA TitleWaveYs+0
    LDA #$C8
    STA TitleWaveYs+1
    LDA #$D8
    STA TitleWaveYs+2

    ; UNKNOWN:
    ; These don't seem to be used in the demo.
    ; Maybe the waterfall used to be bigger?
    LDA #$C0
    STA TitleWaveYs+3
    LDA #$D0
    STA TitleWaveYs+4
    LDA #$DD
    STA TitleWaveYs+5
    INC InitializedWaterfallAnimation

@UpdateSprites:
    LDX #$02
:
    JSR UpdateSpritesForWaterfallWave
    DEX
    BPL :-
    JSR UpdateSpritesForWaterfallCrest
    RTS

UpdateSpritesForWaterfallWave:
    ; Add 2 to current waterfall wave Y.
    ; But keep it in the range $B2..$E3.
    INC TitleWaveYs, X
    INC TitleWaveYs, X
    LDA TitleWaveYs, X
    CMP #$E3
    BCC :+
    LDA #$B2
    STA TitleWaveYs, X
:
    STA $05                     ; Keep a copy of the new value in [$05].

    ; Depending on the Y coordinate of the wave,
    ; modify the animation state.
    ;
    ; < $B9, use tile offset 0
    ; < $C2, use tile offset 8
    ; else, use tile offset $10
    TAY
    LDA #$10
    CPY #$C2
    BCS @SetOffset
    LSR
    CPY #$B9
    BCS @SetOffset
    LDA #$00

@SetOffset:
    STA $00
    STX $02                     ; Keep a copy of the wave index in [$02].
    LDY WaterfallWaveSpriteOffsets, X    ; Y gets offset of first sprite in current wave.
    LDX #$03                    ; For each sprite (4) in current wave, indexed by X:

@LoopSprite:
    LDA WaterfallWaveTiles, X   ; Get base tile for current sprite.
    CLC
    ADC $00                     ; Modify the tile according to the current state of the wave.
    STA Sprites+1, Y
    LDA WaterfallSpriteXs, X    ; Set sprite X to each part of wave in a row.
    STA Sprites+3, Y
    LDA $05                     ; Set sprite Y to rolling value.
    STA Sprites, Y
    LDA #$00                    ; Set sprite attributes: normal with palette 4.
    STA Sprites+2, Y
    INY                         ; Advance to the next sprite.
    INY
    INY
    INY
    DEX
    BPL @LoopSprite
    LDX $02                     ; Put the wave index in X again.
    RTS

UpdateSpritesForWaterfallCrest:
    LDX #$03                    ; For each sprite (4) in crest, indexed by X:
    LDY #$F0

@LoopSprite:
    ; Every 16 frames, switch between
    ; the two frames of animation.
    LDA FrameCounter
    AND #$08
    ADC WaterfallCrestTiles, X
    STA Sprites+1, Y
    LDA #$A8                    ; The Y coordinate is fixed in place.
    STA Sprites, Y
    LDA WaterfallSpriteXs, X    ; Set sprite X to each part of crest in a row.
    STA Sprites+3, Y
    LDA #$00                    ; Set sprite attributes: normal with palette 4.
    STA Sprites+2, Y
    INY                         ; Advance to the next sprite.
    INY
    INY
    INY
    DEX
    BPL @LoopSprite
    RTS

DemoPhase0Subphase1Palettes:
    .BYTE $36, $0F, $00, $10, $36, $17, $27, $0F
    .BYTE $36, $08, $1A, $28, $36, $30, $3B, $22
    .BYTE $36, $30, $3B, $16, $36, $17, $27, $0F
    .BYTE $36, $08, $1A, $28, $36, $30, $3B, $22
    .BYTE $39, $0F, $00, $10, $39, $17, $27, $0F
    .BYTE $39, $08, $1A, $28, $39, $30, $3B, $22
    .BYTE $39, $30, $3B, $16, $39, $17, $27, $0F
    .BYTE $39, $08, $1A, $28, $39, $30, $3B, $22
    .BYTE $31, $0F, $00, $10, $31, $17, $27, $0F
    .BYTE $31, $08, $1A, $28, $31, $30, $3B, $22
    .BYTE $31, $30, $3B, $16, $31, $17, $27, $0F
    .BYTE $31, $08, $1A, $28, $31, $30, $3B, $22
    .BYTE $3C, $0F, $00, $10, $3C, $17, $27, $0F
    .BYTE $3C, $08, $1A, $28, $3C, $30, $3B, $22
    .BYTE $3C, $30, $3B, $16, $3C, $17, $27, $0F
    .BYTE $3C, $08, $1A, $28, $3C, $30, $3B, $22
    .BYTE $3B, $0F, $00, $10, $3B, $17, $27, $0F
    .BYTE $3B, $08, $1A, $28, $3B, $10, $3B, $22
    .BYTE $3B, $10, $3B, $16, $3B, $17, $27, $0F
    .BYTE $3B, $08, $1A, $28, $3B, $10, $3B, $22
    .BYTE $2C, $0F, $00, $10, $2C, $17, $27, $0F
    .BYTE $2C, $08, $1A, $28, $2C, $10, $3B, $22
    .BYTE $2C, $10, $3B, $16, $2C, $17, $27, $0F
    .BYTE $2C, $08, $1A, $28, $2C, $10, $3B, $22
    .BYTE $1C, $0F, $00, $10, $1C, $17, $27, $0F
    .BYTE $1C, $08, $1A, $28, $1C, $10, $3B, $22
    .BYTE $1C, $10, $3B, $16, $1C, $17, $27, $0F
    .BYTE $1C, $08, $1A, $28, $1C, $10, $3B, $22
    .BYTE $02, $0F, $00, $10, $02, $06, $27, $0F
    .BYTE $02, $0A, $1A, $18, $02, $10, $2B, $12
    .BYTE $02, $10, $2B, $06, $02, $06, $27, $0F
    .BYTE $02, $0A, $1A, $18, $02, $10, $2B, $12
    .BYTE $0C, $0F, $00, $10, $0C, $03, $16, $0F
    .BYTE $0C, $01, $0A, $08, $0C, $00, $1B, $02
    .BYTE $0C, $00, $1B, $02, $0C, $03, $16, $0F
    .BYTE $0C, $01, $0A, $08, $0C, $00, $1B, $02
    .BYTE $0F, $0F, $0F, $00, $0F, $01, $11, $0F
    .BYTE $0F, $0C, $01, $02, $0F, $00, $01, $0C
    .BYTE $0F, $00, $01, $0C, $0F, $01, $11, $0F
    .BYTE $0F, $0C, $01, $02, $0F, $00, $01, $0C
    .BYTE $0F, $0F, $0F, $00, $0F, $01, $11, $0F
    .BYTE $0F, $0F, $0C, $01, $0F, $01, $0C, $0F
    .BYTE $0F, $01, $0C, $0F, $0F, $01, $11, $0F
    .BYTE $0F, $0F, $0C, $01, $0F, $01, $0C, $0F
    .BYTE $0F, $0F, $0F, $0F, $0F, $0F, $01, $0F
    .BYTE $0F, $0F, $0F, $0C, $0F, $0C, $0F, $0F
    .BYTE $0F, $0C, $0F, $0F, $0F, $0F, $01, $0F
    .BYTE $0F, $0F, $0F, $0C, $0F, $0C, $0F, $0F
    .BYTE $0F, $0F, $0F, $0F, $0F, $0F, $0F, $0F
    .BYTE $0F, $0F, $0F, $0F, $0F, $0F, $0F, $0F
    .BYTE $0F, $0F, $0F, $0F, $0F, $0F, $0F, $0F
    .BYTE $0F, $0F, $0F, $0F, $0F, $0F, $0F, $0F
    .BYTE $0F, $0F, $0F, $0F, $0F, $0F, $0F, $0F
    .BYTE $0F, $0F, $0F, $0F, $0F, $0F, $0F, $0F
    .BYTE $0F, $0F, $0F, $0F, $0F, $0F, $0F, $0F
    .BYTE $0F, $0F, $0F, $0F, $0F, $0F, $0F, $0F

; Unknown block
    .BYTE $0F, $0F, $0F, $0F, $0F, $0F, $0F, $0F
    .BYTE $0F, $0F, $0F, $0F, $0F, $0F, $0F, $0F
    .BYTE $0F, $0F, $0F, $0F, $0F, $0F, $0F, $0F
    .BYTE $0F, $0F, $0F, $0F, $0F, $0F, $0F, $0F
    .BYTE $0F, $0F, $0F, $0F, $0F, $0F, $0F, $0F
    .BYTE $0F, $0F, $0F, $0F, $0F, $0F, $0F, $0F
    .BYTE $0F, $0F, $0F, $0F, $0F, $0F, $0F, $0F
    .BYTE $0F, $0F, $0F, $0F, $0F, $0F, $0F, $0F

DemoPhase0Subphase1Delays:
    .BYTE $08, $08, $06, $05, $04, $03, $02, $02
    .BYTE $02, $C0, $06, $04, $C0, $03

; Unknown block
    .BYTE $04, $04

AnimateDemoPhase0Subphase1:
    LDA DemoPhase0Subphase1Timer    ; When subphase timer expires,
    BNE @UpdateAnimation        ; go update animation only.

    ; Calculate the address to the palette to transfer.
    ; Addr = DemoPhase0Subphase1Palettes + (DemoPhase0Subphase1Palettes * $20)
    LDA #$00
    STA $01
    LDA DemoPhase0Subphase1Cycle
    ASL
    ASL
    ASL
    ASL
    ROL $01
    ASL
    ROL $01
    ADC #<DemoPhase0Subphase1Palettes
    STA $00
    LDA $01
    ADC #>DemoPhase0Subphase1Palettes
    STA $01
    LDA #$3F                    ; Set up the header of the transfer record.
    STA DynTileBuf
    LDA #$00
    STA DynTileBuf+1
    LDA #$20
    STA DynTileBuf+2
    LDY #$1F                    ; Put an end marker at the end.
    LDA #$FF
    STA DynTileBuf+4, Y

@CopyPalette:
    LDA ($00), Y                ; Copy the chosen palette into the transfer record.
    STA DynTileBuf+3, Y
    DEY
    BPL @CopyPalette

    ; Advance the subphase cycle.
    INC DemoPhase0Subphase1Cycle

    ; Set the timer to the delay for the current point in the cycle.
    LDY DemoPhase0Subphase1Cycle
    LDA DemoPhase0Subphase1Delays-1, Y
    STA DemoPhase0Subphase1Timer
    CPY #$0E
    BCC @UpdateAnimation        ; If we reached the end of the cycle,

    ; Advance to the next demo phase.
    INC DemoPhase
    LDA #$00                    ; Reset the demo subphase.
    STA DemoSubphase
    STA IsUpdatingMode          ; We have to initialize the new demo phase.

@UpdateAnimation:
    DEC DemoPhase0Subphase1Timer
    JSR UpdateWaterfallAnimation
    RTS

; Unknown block
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
    .BYTE $FF
