; ============================================================================
; FILE:     bank0/sound-effects.asm      BANK: 0 (switchable, $8000-$BFFF)
; UPSTREAM: Z_00.asm 223-403
; SUBJECTS: (none)
; TOPIC:    Noise effects: arrow, stairs, sword, flame, bomb, sea
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0023 SilenceSample       code
;   L0033 PlayArrowSfx        code
;   L0044 ContinueArrowSfx    code
;   L0049 PlayStairsSfx       code
;   L0057 ContinueStairsSfx   code
;   L0068 PlaySfxNote         code
;   L0079 SetSfxVolumeLength  code
;   L0085 SilenceSfxIfEnded   code
;   L0094 PlaySwordSfx        code
;   L0099 ContinueSwordSfx    code
;   L0104 DriveEffect         code
;   L0200 BombSfxNotes        data
; ============================================================================

SilenceSample:
    LDA #$0F
    STA ApuStatus
    LDA #$00
    STA CurSample
    STA CurTune1
    STA $061A
    STA BackgroundSample
    RTS

PlayArrowSfx:
    STY CurEffect
    LDA #$05
    STA EffectCounter

    ; If "heart taken" is requested in tune channel 0, then cancel it.
    LDA Tune0Request
    AND #$EF
    BNE ContinueArrowSfx
    STA Tune0Request

ContinueArrowSfx:
    LDY EffectCounter
    LDA ArrowSfxNotes-1, Y
    BNE PlaySfxNote

PlayStairsSfx:
    STY CurEffect
    LDA #$38
    STA EffectCounter
:
    LDA #$0D
    STA $68                     ; The sound of one step lasts $C frames.

ContinueStairsSfx:
    DEC $68
    LDY $68
    BEQ :-
    CPY #$07
    BCC :+
    LDA #$10
    BNE SetSfxVolumeLength
:
    LDA StairsSfxNotes-1, Y

PlaySfxNote:
    TAX
    AND #$0F
    STA NoisePeriod
    TXA
    LSR
    LSR
    LSR
    LSR
    ORA #$10

SetSfxVolumeLength:
    STA NoiseVolume
    LDA #$08
    STA NoiseLength
    DEC EffectCounter

SilenceSfxIfEnded:
    BNE :+
    LDA #$F0
    STA NoiseVolume
    LDA #$00
    STA CurEffect
:
    RTS

PlaySwordSfx:
    STY CurEffect
    LDA #$0A
    STA EffectCounter

ContinueSwordSfx:
    LDY EffectCounter
    LDA SwordSfxNotes-1, Y
    BNE PlaySfxNote

DriveEffect:
    LDY EffectRequest

    ; $80 is a signal to silence samples and tune 1.
    BMI SilenceSample
    LDA CurEffect
    LSR EffectRequest
    BCS PlaySwordSfx
    LSR
    BCS ContinueSwordSfx
    LSR EffectRequest
    BCS PlayArrowSfx
    LSR
    BCS ContinueArrowSfx
    LSR EffectRequest
    BCS @PlayFlameSfx
    LSR
    BCS @ContinueFlameSfx
    LSR EffectRequest
    BCS PlayStairsSfx
    LSR
    BCS ContinueStairsSfx
    LSR EffectRequest
    BCS @PlayBombSfx
    LSR
    BCS @ContinueBombSfx
    LSR
    BCS @ContinueSeaSfx
    LSR EffectRequest
    BCS @PlaySeaSfx
    RTS

@PlayBombSfx:
    STY CurEffect
    LDA #$18
    STA EffectCounter

@ContinueBombSfx:
    LDY EffectCounter
    LDA BombSfxNotes-1, Y
    BNE PlaySfxNote

@PlayFlameSfx:
    STY CurEffect
    LDA #$20
    STA EffectCounter

@ContinueFlameSfx:
    LDA EffectCounter
    LSR
    TAY
    LDX #$0E
    STX NoisePeriod
    LDA FlameSfxNotes-1, Y
    JMP SetSfxVolumeLength

@PlaySeaSfx:
    STY CurEffect
    LDA #$D0
    STA SeaSfxCounter
    LDA #$10
    STA $68                     ; The volume begins and ends at $10.

@ContinueSeaSfx:
    LDA SeaSfxCounter
    CMP #$BF
    BCC :+

    ; SFX counter >= $BF. Increase volume fast.
    INC $68
    BNE @SetSeaSfxParams
:
    ; SFX counter < $BF. Decrease volume slowly down to $10 --
    ; when (counter MOD 8) = 7.
    LDA SeaSfxCounter
    LSR
    BCC @SetSeaSfxParams
    LSR
    BCC @SetSeaSfxParams
    LSR
    BCC @SetSeaSfxParams
    LDA $68
    CMP #$10
    BEQ @SetSeaSfxParams
    DEC $68

@SetSeaSfxParams:
    LDA $68
    STA NoiseVolume
    LDX #$03
    STX NoisePeriod
    LDA #$08
    STA NoiseLength
    DEC SeaSfxCounter
    JMP SilenceSfxIfEnded

BombSfxNotes:
    .BYTE $1F, $2F, $2E, $3F, $3F, $4C, $4E, $5F
    .BYTE $6F, $6F, $7E, $8F, $9E, $AF, $BE, $CF
    .BYTE $DE, $EF, $FE, $FD, $FE, $FF, $FF, $FE
