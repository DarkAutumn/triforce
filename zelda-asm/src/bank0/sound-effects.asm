; ============================================================================
; FILE:     bank0/sound-effects.asm      BANK: 0 (switchable, $8000-$BFFF)
; UPSTREAM: Z_00.asm 223-403
; SUBJECTS: (none)
; TOPIC:    Noise-channel sound effects (sword, arrow, flame, stairs, bomb, sea) and their driver
;           DriveEffect; SilenceSample, which stops the sample and tune channel 1.
; SEE ALSO: bank0/song-note-tables.asm (SwordSfxNotes, ArrowSfxNotes, FlameSfxNotes, StairsSfxNotes);
;           bank0/drive-audio-tune0.asm (DriveAudio calls DriveEffect);
;           bank1/modeb-cave-save-sprites.asm (PlayEffect ORs a bit into EffectRequest).
; KEY FACTS:
;   - EffectRequest bits: $01 sword, $02 arrow, $04 flame, $08 stairs, $10 bomb, $20 sea; bit 7 calls
;     SilenceSample instead. DriveEffect.
;   - Lengths in frames: sword $0A, arrow 5, flame $20, stairs $38, bomb $18, sea $D0. PlaySwordSfx,
;     PlayArrowSfx, PlayStairsSfx and the cheap labels of DriveEffect.
;   - Each frame only the first match runs, tested in this order: sword request, sword playing, arrow
;     request, arrow playing, flame, stairs, bomb (request then playing each), sea playing, sea request.
;     A new effect replaces CurEffect with the whole request byte. DriveEffect.
;   - A sound effect note byte is noise volume in the high nibble and noise period in the low nibble.
;     PlaySfxNote.
;   - The stairs effect repeats a $0C-frame step: 6 silent frames, then 6 notes. ContinueStairsSfx.
;   - Starting the arrow effect cancels a Tune0Request that is exactly $10 in the same frame.
;     PlayArrowSfx.
; CONTENTS: (generated; L = line in this file)
;   L0042 SilenceSample       code  Stops the DMC sample (current and background) and tune channel 1.
;   L0056 PlayArrowSfx        code  Starts the arrow effect (5 frames). The effect entry points below share code by falling through.
;   L0068 ContinueArrowSfx    code  Plays the next note of the arrow effect.
;   L0074 PlayStairsSfx       code  Starts the stairs effect ($38 frames). IN: Y = original EffectRequest.
;   L0083 ContinueStairsSfx   code  Plays the next frame of the stairs effect; [68] counts $0C..1 within one step.
;   L0095 PlaySfxNote         code  Sets the noise channel from a sound effect note. IN: A = volume (high nibble), period (low nibble).
;   L0107 SetSfxVolumeLength  code  Writes A to NoiseVolume, reloads the noise length and counts EffectCounter down.
;   L0114 SilenceSfxIfEnded   code  Ends the current effect (mutes noise, clears CurEffect) if the Z flag is set (counter reached 0).
;   L0124 PlaySwordSfx        code  Starts the sword effect ($0A frames). IN: Y = original EffectRequest.
;   L0130 ContinueSwordSfx    code  Plays the next note of the sword effect.
;   L0140 DriveEffect         code  Starts a requested noise effect or continues the one playing, for one frame.
;   L0237 BombSfxNotes        data  TABLE: bomb effect notes; index = EffectCounter - 1 (read from the end); volume/period nibbles.
; ============================================================================

; ----------------------------------------------------------------------------
; Stops the DMC sample (current and background) and tune channel 1.
; NOTE: Reached from DriveEffect when EffectRequest has bit 7 set.
; ----------------------------------------------------------------------------
SilenceSample:
    LDA #$0F
    STA ApuStatus
    LDA #$00
    STA CurSample
    STA CurTune1
    STA $061A
    STA BackgroundSample
    RTS

; ----------------------------------------------------------------------------
; Starts the arrow effect (5 frames). The effect entry points below share code by falling through.
; IN:   Y = EffectRequest as it was at the start of DriveEffect (becomes CurEffect).
; ----------------------------------------------------------------------------
PlayArrowSfx:
    STY CurEffect
    LDA #$05
    STA EffectCounter

    ; If Tune0Request has no bit other than $10 (the tune requested for a heart pickup), clear it.
    LDA Tune0Request
    AND #$EF
    BNE ContinueArrowSfx
    STA Tune0Request

; Plays the next note of the arrow effect.
ContinueArrowSfx:
    LDY EffectCounter
    LDA ArrowSfxNotes-1, Y
    BNE PlaySfxNote

; Starts the stairs effect ($38 frames). IN: Y = original EffectRequest.
PlayStairsSfx:
    STY CurEffect
    LDA #$38
    STA EffectCounter
:
    LDA #$0D
    STA $68                     ; The sound of one step lasts $C frames.

; Plays the next frame of the stairs effect; [68] counts $0C..1 within one step.
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

; Sets the noise channel from a sound effect note. IN: A = volume (high nibble), period (low nibble).
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

; Writes A to NoiseVolume, reloads the noise length and counts EffectCounter down.
SetSfxVolumeLength:
    STA NoiseVolume
    LDA #$08
    STA NoiseLength
    DEC EffectCounter

; Ends the current effect (mutes noise, clears CurEffect) if the Z flag is set (counter reached 0).
SilenceSfxIfEnded:
    BNE :+
    LDA #$F0
    STA NoiseVolume
    LDA #$00
    STA CurEffect
:
    RTS

; Starts the sword effect ($0A frames). IN: Y = original EffectRequest.
PlaySwordSfx:
    STY CurEffect
    LDA #$0A
    STA EffectCounter

; Plays the next note of the sword effect.
ContinueSwordSfx:
    LDY EffectCounter
    LDA SwordSfxNotes-1, Y
    BNE PlaySfxNote

; ----------------------------------------------------------------------------
; Starts a requested noise effect or continues the one playing, for one frame.
; NOTE: Bit order of EffectRequest and CurEffect: $01 sword, $02 arrow, $04 flame, $08 stairs, $10 bomb,
;       $20 sea. The first match wins; a playing sea effect is checked before a sea request.
; ----------------------------------------------------------------------------
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

; TABLE: bomb effect notes; index = EffectCounter - 1 (read from the end); volume/period nibbles.
BombSfxNotes:
    .BYTE $1F, $2F, $2E, $3F, $3F, $4C, $4E, $5F
    .BYTE $6F, $6F, $7E, $8F, $9E, $AF, $BE, $CF
    .BYTE $DE, $EF, $FE, $FD, $FE, $FF, $FF, $FE
