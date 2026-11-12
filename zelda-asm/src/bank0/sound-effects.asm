; ============================================================================
; FILE:     bank0/sound-effects.asm      BANK: 0 (switchable, $8000-$BFFF)
; UPSTREAM: Z_00.asm 223-403
; SUBJECTS: Sample, Sfx
; TOPIC:    Noise-channel sound effects (sword, arrow, flame, stairs, bomb, sea) and their driver
;           Sfx_Drive; Sample_Silence, which stops the sample and tune channel 1.
; SEE ALSO: bank0/song-note-tables.asm (Sfx_SwordNotes, Sfx_ArrowNotes, Sfx_FlameVolumes, Sfx_StairsNotes);
;           bank0/drive-audio-tune0.asm (Audio_Drive calls Sfx_Drive);
;           bank1/modeb-cave-save-sprites.asm (Sfx_Play ORs a bit into EffectRequest).
; KEY FACTS:
;   - EffectRequest bits: $01 sword, $02 arrow, $04 flame, $08 stairs, $10 bomb, $20 sea; bit 7 calls
;     Sample_Silence instead. Sfx_Drive.
;   - Lengths in frames: sword $0A, arrow 5, flame $20, stairs $38, bomb $18, sea $D0. Sfx_StartSword,
;     Sfx_StartArrow, Sfx_StartStairs and the cheap labels of Sfx_Drive.
;   - Each frame only the first match runs, tested in this order: sword request, sword playing, arrow
;     request, arrow playing, flame, stairs, bomb (request then playing each), sea playing, sea request.
;     A new effect replaces CurEffect with the whole request byte. Sfx_Drive.
;   - A sound effect note byte is noise volume in the high nibble and noise period in the low nibble.
;     Sfx_PlayNote.
;   - The stairs effect repeats a $0C-frame step: 6 silent frames, then 6 notes. Sfx_ContinueStairs.
;   - Starting the arrow effect cancels a Tune0Request that is exactly $10 in the same frame.
;     Sfx_StartArrow.
; CONTENTS: (generated; L = line in this file)
;   L0042 Sample_Silence          code  Stops the DMC sample (current and background) and tune channel 1.
;   L0056 Sfx_StartArrow          code  Starts the arrow effect (5 frames). The effect entry points below share code by falling through.
;   L0068 Sfx_ContinueArrow       code  Plays the next note of the arrow effect.
;   L0074 Sfx_StartStairs         code  Starts the stairs effect ($38 frames). IN: Y = original EffectRequest.
;   L0083 Sfx_ContinueStairs      code  Plays the next frame of the stairs effect; EffectScratch counts $0C..1 within one step.
;   L0095 Sfx_PlayNote            code  Sets the noise channel from a sound effect note. IN: A = volume (high nibble), period (low nibble).
;   L0107 Sfx_SetVolumeAndLength  code  Writes A to NoiseVolume, reloads the noise length and counts EffectCounter down.
;   L0114 Sfx_SilenceIfEnded      code  Ends the current effect (mutes noise, clears CurEffect) if the Z flag is set (counter reached 0).
;   L0124 Sfx_StartSword          code  Starts the sword effect ($0A frames). IN: Y = original EffectRequest.
;   L0130 Sfx_ContinueSword       code  Plays the next note of the sword effect.
;   L0140 Sfx_Drive               code  Starts a requested noise effect or continues the one playing, for one frame.
;   L0237 Sfx_BombNotes           data  TABLE: bomb effect notes; index = EffectCounter - 1 (read from the end); volume/period nibbles.
; ============================================================================

; ----------------------------------------------------------------------------
; Stops the DMC sample (current and background) and tune channel 1.
; NOTE: Reached from Sfx_Drive when EffectRequest has bit 7 set.
; ----------------------------------------------------------------------------
Sample_Silence:
    LDA #$0F
    STA ApuStatus
    LDA #$00
    STA CurSample
    STA CurTune1
    STA UnusedSound61A          ; Written here only; nothing reads it.
    STA BackgroundSample
    RTS

; ----------------------------------------------------------------------------
; Starts the arrow effect (5 frames). The effect entry points below share code by falling through.
; IN:   Y = EffectRequest as it was at the start of Sfx_Drive (becomes CurEffect).
; ----------------------------------------------------------------------------
Sfx_StartArrow:
    STY CurEffect
    LDA #$05
    STA EffectCounter

    ; If Tune0Request has no bit other than $10 (the tune requested for a heart pickup), clear it.
    LDA Tune0Request
    AND #$EF
    BNE Sfx_ContinueArrow
    STA Tune0Request

; Plays the next note of the arrow effect.
Sfx_ContinueArrow:
    LDY EffectCounter
    LDA Sfx_ArrowNotes-1, Y
    BNE Sfx_PlayNote

; Starts the stairs effect ($38 frames). IN: Y = original EffectRequest.
Sfx_StartStairs:
    STY CurEffect
    LDA #$38
    STA EffectCounter
:
    LDA #$0D
    STA EffectScratch           ; The sound of one step lasts $C frames.

; Plays the next frame of the stairs effect; EffectScratch counts $0C..1 within one step.
Sfx_ContinueStairs:
    DEC EffectScratch
    LDY EffectScratch
    BEQ :-
    CPY #$07
    BCC :+
    LDA #$10
    BNE Sfx_SetVolumeAndLength
:
    LDA Sfx_StairsNotes-1, Y

; Sets the noise channel from a sound effect note. IN: A = volume (high nibble), period (low nibble).
Sfx_PlayNote:
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
Sfx_SetVolumeAndLength:
    STA NoiseVolume
    LDA #$08
    STA NoiseLength
    DEC EffectCounter

; Ends the current effect (mutes noise, clears CurEffect) if the Z flag is set (counter reached 0).
Sfx_SilenceIfEnded:
    BNE :+
    LDA #$F0
    STA NoiseVolume
    LDA #$00
    STA CurEffect
:
    RTS

; Starts the sword effect ($0A frames). IN: Y = original EffectRequest.
Sfx_StartSword:
    STY CurEffect
    LDA #$0A
    STA EffectCounter

; Plays the next note of the sword effect.
Sfx_ContinueSword:
    LDY EffectCounter
    LDA Sfx_SwordNotes-1, Y
    BNE Sfx_PlayNote

; ----------------------------------------------------------------------------
; Starts a requested noise effect or continues the one playing, for one frame.
; NOTE: Bit order of EffectRequest and CurEffect: $01 sword, $02 arrow, $04 flame, $08 stairs, $10 bomb,
;       $20 sea. The first match wins; a playing sea effect is checked before a sea request.
; ----------------------------------------------------------------------------
Sfx_Drive:
    LDY EffectRequest

    ; $80 is a signal to silence samples and tune 1.
    BMI Sample_Silence
    LDA CurEffect
    LSR EffectRequest
    BCS Sfx_StartSword
    LSR
    BCS Sfx_ContinueSword
    LSR EffectRequest
    BCS Sfx_StartArrow
    LSR
    BCS Sfx_ContinueArrow
    LSR EffectRequest
    BCS @PlayFlameSfx
    LSR
    BCS @ContinueFlameSfx
    LSR EffectRequest
    BCS Sfx_StartStairs
    LSR
    BCS Sfx_ContinueStairs
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
    LDA Sfx_BombNotes-1, Y
    BNE Sfx_PlayNote

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
    LDA Sfx_FlameVolumes-1, Y
    JMP Sfx_SetVolumeAndLength

@PlaySeaSfx:
    STY CurEffect
    LDA #$D0
    STA SeaSfxCounter
    LDA #$10
    STA EffectScratch           ; The volume begins and ends at $10.

@ContinueSeaSfx:
    LDA SeaSfxCounter
    CMP #$BF
    BCC :+

    ; SFX counter >= $BF. Increase volume fast.
    INC EffectScratch
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
    LDA EffectScratch
    CMP #$10
    BEQ @SetSeaSfxParams
    DEC EffectScratch

@SetSeaSfxParams:
    LDA EffectScratch
    STA NoiseVolume
    LDX #$03
    STX NoisePeriod
    LDA #$08
    STA NoiseLength
    DEC SeaSfxCounter
    JMP Sfx_SilenceIfEnded

; TABLE: bomb effect notes; index = EffectCounter - 1 (read from the end); volume/period nibbles.
Sfx_BombNotes:
    .BYTE $1F, $2F, $2E, $3F, $3F, $4C, $4E, $5F
    .BYTE $6F, $6F, $7E, $8F, $9E, $AF, $BE, $CF
    .BYTE $DE, $EF, $FE, $FD, $FE, $FF, $FF, $FE
