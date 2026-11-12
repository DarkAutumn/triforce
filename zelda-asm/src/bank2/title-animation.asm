; ============================================================================
; FILE:     bank2/title-animation.asm    BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 1003-1348
; SUBJECTS: Mode0Demo, Title
; TOPIC:    Title screen animation: the title sprites, the glowing Triforce palette, the waterfall
;           sprites, and the palette fade to black (phase 0 subphase 1) that leads into the story.
; SEE ALSO: bank2/demo-story-scroll.asm (Mode0Demo_Animate_Phase0_Sub0 calls the sprite and glow routine
;           here); bank2/demo-tables-init.asm (Title_InitialSprites, the title palette and the variable
;           reset); bank2/mode0-demo.asm (Mode0Demo_Animate_Phase0_JumpTable).
; KEY FACTS:
;   - The Triforce colour (palette byte $3F06) steps through 8 values, each held 6 frames, the last
;     $10 frames: one glow cycle is 58 frames. Title_TriforceGlowColors,
;     Title_AnimateSpritesAndGlow.
;   - The waterfall is 3 waves of 4 sprites (sprite records at offsets $70, $80, $90) that move down 2
;     pixels per frame and wrap from Y >= $E3 to $B2, plus a 4-sprite crest fixed at Y = $A8 (offset
;     $F0) that changes tile every 8 frames. Title_DrawWaterfallWave,
;     Title_DrawWaterfallCrest.
;   - A wave uses tile set +0 while Y < $B9, +8 while Y < $C2, else +$10. Title_DrawWaterfallWave.
;   - The fade sends 14 full 32-byte palettes; palette n is held Title_FadeDelays[n] frames
;     (8, 8, 6, 5, 4, 3, 2, 2, 2, $C0, 6, 4, $C0), 434 frames in all; with the 14th palette DemoPhase
;     becomes 1 and IsUpdatingMode 0 at once. Mode0Demo_Animate_Phase0_Sub1.
;   - TitleWaveYs+3..+5, the 5 bytes after Title_WaterfallWaveSpriteOffsets, the 64 bytes after
;     Title_FadePalettes and the 2 bytes after Title_FadeDelays are never read.
;   - The $FF bytes at the end of the file are padding (297 bytes).
; CONTENTS: (generated; L = line in this file)
;   L0042 Title_TriforcePaletteTransferRecord  data  DATA: PPU transfer record for palette bytes $3F04-$3F07; its byte 5 (for $3F06) is patched per step.
;   L0046 Title_TriforceGlowColors             data  TABLE: Triforce glow colour; index = TriforceGlowCycle (0-7); value = NES colour for $3F06.
;   L0052 Title_AnimateSpritesAndGlow          code  Title subphase 0 per frame: copies the title sprites, animates the waterfall, steps the glow.
;   L0097 Title_WaterfallWaveTiles             data  TABLE: first-frame tile of each waterfall wave sprite; index = sprite 0-3 (read from 3 down).
;   L0101 Title_WaterfallCrestTiles            data  TABLE: first-frame tile of each waterfall crest sprite; index = sprite 0-3.
;   L0105 Title_WaterfallSpriteXs              data  TABLE: X coordinate of the 4 sprites of a waterfall wave or crest; index = sprite 0-3.
;   L0109 Title_WaterfallWaveSpriteOffsets     data  TABLE: offset in Sprites of the first sprite of each waterfall wave; index = wave 0-2.
;   L0118 Title_UpdateWaterfall                code  Sets the waterfall wave Ys on first use, then moves and draws the 3 waves and the crest.
;   L0151 Title_DrawWaterfallWave              code  Moves one waterfall wave down 2 pixels (wrapping to $B2) and writes its 4 sprites.
;   L0208 Title_DrawWaterfallCrest             code  Writes the 4 sprites of the waterfall crest at Y = $A8, alternating two tile sets every 8 frames.
;   L0234 Title_FadePalettes                   data  TABLE: the 14 palettes of the title fade, 32 bytes each; index = DemoPhase0Subphase1Cycle * $20.
;   L0304 Title_FadeDelays                     data  TABLE: frames each fade palette is held; index = DemoPhase0Subphase1Cycle - 1 after the increment.
;   L0314 Mode0Demo_Animate_Phase0_Sub1        code  Title subphase 1: fades the title palettes to black in 14 timed steps, then starts the story phase.
; ============================================================================

; DATA: PPU transfer record for palette bytes $3F04-$3F07; its byte 5 (for $3F06) is patched per step.
Title_TriforcePaletteTransferRecord:
    .BYTE $3F, $04, $04, $36, $17, $27, $0F, $FF

; TABLE: Triforce glow colour; index = TriforceGlowCycle (0-7); value = NES colour for $3F06.
Title_TriforceGlowColors:
    .BYTE $27, $37, $37, $27, $17, $07, $07, $17

; ----------------------------------------------------------------------------
; Title subphase 0 per frame: copies the title sprites, animates the waterfall, steps the glow.
; ----------------------------------------------------------------------------
Title_AnimateSpritesAndGlow:
    ; Set up sprites for the title.
    ; Copy initial sprite data to Sprites area.
    LDY #$70

@CopySprites:
    LDA Title_InitialSprites-1, Y
    STA Sprites-1, Y
    DEY
    BNE @CopySprites
    JSR Title_UpdateWaterfall

    ; When you reach the end of the glow cycle,
    ; Append a transfer record for the Triforce palette.
    LDA TriforceGlowTimer
    BNE @DecGlowTimer
    LDY #$07

@CopyTriforcePalette:
    LDA Title_TriforcePaletteTransferRecord, Y
    STA DynTileBuf, Y
    DEY
    BPL @CopyTriforcePalette

    ; Patch the palette record with the color
    ; for the current point in the cycle.
    LDY TriforceGlowCycle
    LDA Title_TriforceGlowColors, Y
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

; TABLE: first-frame tile of each waterfall wave sprite; index = sprite 0-3 (read from 3 down).
Title_WaterfallWaveTiles:
    .BYTE $B2, $B4, $B6, $B8

; TABLE: first-frame tile of each waterfall crest sprite; index = sprite 0-3.
Title_WaterfallCrestTiles:
    .BYTE $A2, $A4, $A6, $A8

; TABLE: X coordinate of the 4 sprites of a waterfall wave or crest; index = sprite 0-3.
Title_WaterfallSpriteXs:
    .BYTE $50, $58, $60, $68

; TABLE: offset in Sprites of the first sprite of each waterfall wave; index = wave 0-2.
Title_WaterfallWaveSpriteOffsets:
    .BYTE $70, $80, $90

; Five more sprite offsets that continue the table above; no wave index reaches them.
    .BYTE $A0, $B0, $C0, $D0, $E0

; ----------------------------------------------------------------------------
; Sets the waterfall wave Ys on first use, then moves and draws the 3 waves and the crest.
; ----------------------------------------------------------------------------
Title_UpdateWaterfall:
    LDA IsWaterfallInitialized
    BNE @UpdateSprites
    LDA #$B6                    ; Initialize animation values.
    STA TitleWaveYs+0
    LDA #$C8
    STA TitleWaveYs+1
    LDA #$D8
    STA TitleWaveYs+2

    ; TitleWaveYs+3..+5 are initialized here but never read: only waves 0-2 are drawn.
    LDA #$C0
    STA TitleWaveYs+3
    LDA #$D0
    STA TitleWaveYs+4
    LDA #$DD
    STA TitleWaveYs+5
    INC IsWaterfallInitialized

@UpdateSprites:
    LDX #$02
:
    JSR Title_DrawWaterfallWave
    DEX
    BPL :-
    JSR Title_DrawWaterfallCrest
    RTS

; ----------------------------------------------------------------------------
; Moves one waterfall wave down 2 pixels (wrapping to $B2) and writes its 4 sprites.
; IN:   X = wave 0-2.
; OUT:  X preserved; [00] = tile offset, [02] = wave, [05] = new Y.
; ----------------------------------------------------------------------------
Title_DrawWaterfallWave:
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
    LDY Title_WaterfallWaveSpriteOffsets, X    ; Y gets offset of first sprite in current wave.
    LDX #$03                    ; For each sprite (4) in current wave, indexed by X:

@LoopSprite:
    LDA Title_WaterfallWaveTiles, X   ; Get base tile for current sprite.
    CLC
    ADC $00                     ; Modify the tile according to the current state of the wave.
    STA Sprites+1, Y
    LDA Title_WaterfallSpriteXs, X    ; Set sprite X to each part of wave in a row.
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

; ----------------------------------------------------------------------------
; Writes the 4 sprites of the waterfall crest at Y = $A8, alternating two tile sets every 8 frames.
; ----------------------------------------------------------------------------
Title_DrawWaterfallCrest:
    LDX #$03                    ; For each sprite (4) in crest, indexed by X:
    LDY #$F0

@LoopSprite:
    ; Every 8 frames, switch between
    ; the two frames of animation (16-frame period).
    LDA FrameCounter
    AND #$08
    ADC Title_WaterfallCrestTiles, X
    STA Sprites+1, Y
    LDA #$A8                    ; The Y coordinate is fixed in place.
    STA Sprites, Y
    LDA Title_WaterfallSpriteXs, X    ; Set sprite X to each part of crest in a row.
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

; TABLE: the 14 palettes of the title fade, 32 bytes each; index = DemoPhase0Subphase1Cycle * $20.
Title_FadePalettes:
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

; Two more all-black ($0F) palettes; the fade ends before the cycle reaches them.
    .BYTE $0F, $0F, $0F, $0F, $0F, $0F, $0F, $0F
    .BYTE $0F, $0F, $0F, $0F, $0F, $0F, $0F, $0F
    .BYTE $0F, $0F, $0F, $0F, $0F, $0F, $0F, $0F
    .BYTE $0F, $0F, $0F, $0F, $0F, $0F, $0F, $0F
    .BYTE $0F, $0F, $0F, $0F, $0F, $0F, $0F, $0F
    .BYTE $0F, $0F, $0F, $0F, $0F, $0F, $0F, $0F
    .BYTE $0F, $0F, $0F, $0F, $0F, $0F, $0F, $0F
    .BYTE $0F, $0F, $0F, $0F, $0F, $0F, $0F, $0F

; TABLE: frames each fade palette is held; index = DemoPhase0Subphase1Cycle - 1 after the increment.
; The 14th value ($03) is loaded but the phase ends on that frame.
Title_FadeDelays:
    .BYTE $08, $08, $06, $05, $04, $03, $02, $02
    .BYTE $02, $C0, $06, $04, $C0, $03

; Two more delay values; no cycle value reaches them.
    .BYTE $04, $04

; ----------------------------------------------------------------------------
; Title subphase 1: fades the title palettes to black in 14 timed steps, then starts the story phase.
; ----------------------------------------------------------------------------
Mode0Demo_Animate_Phase0_Sub1:
    LDA DemoPhase0Subphase1Timer    ; When subphase timer expires,
    BNE @UpdateAnimation        ; go update animation only.

    ; Calculate the address to the palette to transfer.
    ; Addr = Title_FadePalettes + (DemoPhase0Subphase1Cycle * $20)
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
    ADC #<Title_FadePalettes
    STA $00
    LDA $01
    ADC #>Title_FadePalettes
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
    LDA Title_FadeDelays-1, Y
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
    JSR Title_UpdateWaterfall
    RTS

; Padding: 297 bytes of $FF, not referenced by any code.
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
