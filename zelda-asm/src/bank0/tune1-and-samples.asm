; ============================================================================
; FILE:     bank0/tune1-and-samples.asm  BANK: 0 (switchable, $8000-$BFFF)
; UPSTREAM: Z_00.asm 405-581
; SUBJECTS: Audio, Sample
; TOPIC:    Tune channel 1 (jingles on square channel 1, with note lengths; they mute the song's
;           square 1 track) and the DMC sample driver with its address, length and rate tables.
; SEE ALSO: bank0/drive-audio-tune0.asm (Audio_Drive calls Audio_DriveTune1 and Sample_Drive);
;           bank0/note-emit-vibrato.asm (Audio_EmitSquareNote1, Audio_VibratePitch); bank0/drive-song.asm
;           (Song_Silence; the song skips square 1 while CurTune1 is nonzero);
;           bank0/sound-effects.asm (Sample_Silence); bank7/run-game-nmi.asm (Sample_PcmData, the sample data);
;           bank1/modeb-cave-save-sprites.asm (Sample_Play ORs a bit into SampleRequest).
; KEY FACTS:
;   - Tune1Request is a bit mask; the lowest set bit picks one of 8 tunes. A request with bit 7 set
;     first silences the song (Song_Silence), then plays tune $80. Audio_DriveTune1.
;   - Tune 1 script bytes: $00 ends the tune; a byte with bit 7 set sets the note length in frames
;     (low 7 bits) and is followed by a note; any other byte is a note id. Audio_DriveTune1.
;   - Tune $40 ("Game Over") restarts itself when its script ends; every other tune stops. Audio_DriveTune1.
;   - Tunes $10 and $80 (flute, Link dying) get the Audio_Tune1EnvelopeDuties volume curve and pitch vibrato;
;     the others play at the fixed duty/volume byte $86. Audio_DriveTune1@CheckVibrate.
;   - SampleRequest: the lowest set bit (bits 0-6) picks the sample. Every sample is timed for $A0 frames
;     by SampleCounter. Bits $10-$40 loop: the sample restarts when the counter runs out and is also
;     remembered in BackgroundSample, which resumes after a one-shot sample (bits $01-$08). Sample_Drive.
;   - A request with bit 7 set keeps only its bits $F0, starts the DMC output level at $7F instead of 0
;     and restarts the same way each time its counter runs out. Sample_Drive@ChangeSampleMid.
;   - DMC register values: sample address = $C000 + value * $40, length = value * $10 + 1 bytes (APU
;     hardware rule, not shown by this code). Sample_DmcAddrs, Sample_DmcLengths.
; CONTENTS: (generated; L = line in this file)
;   L0039 Audio_Tune1Scripts         data  TABLE: tune channel 1 scripts. Bytes 0-7 = script offset (from Audio_Tune1Scripts) of tune bit 0-7; the
;   L0061 Audio_DriveTune1           code  Starts a requested tune on square channel 1 or advances the current one by one frame.
;   L0149 Audio_Tune1EnvelopeDuties  data  TABLE: Sq1Duty values (duty $80, constant volume 5-$F) for tunes $10 and $80; index =
;   L0158 Sample_Drive               code  Starts a requested DMC sample, times the one playing, and restarts looping or background samples.
;   L0219 Sample_DmcAddrs            data  TABLE: DMC sample address register values; index = sample bit number 0-6.
;   L0223 Sample_DmcLengths          data  TABLE: DMC sample length register values; index = sample bit number 0-6; bytes = value * $10 + 1.
;   L0227 Sample_DmcRates            data  TABLE: DMC rate register values (rate index, no loop, no IRQ); index = sample bit number 0-6.
; ============================================================================

; TABLE: tune channel 1 scripts. Bytes 0-7 = script offset (from Audio_Tune1Scripts) of tune bit 0-7; the
; scripts follow.
Audio_Tune1Scripts:
    .BYTE $0C, $08, $11, $1C, $28, $33, $40, $62
    .BYTE $8A, $4E, $58, $60, $8A, $5E, $94, $60
    .BYTE $00, $8A, $42, $06, $3C, $30, $2E, $3E
    .BYTE $44, $CC, $02, $00, $83, $40, $42, $48
    .BYTE $4A, $02, $50, $4C, $54, $94, $56, $00
    .BYTE $94, $3A, $3E, $A8, $50, $8A, $4E, $02
    .BYTE $CC, $4A, $00, $81, $28, $3E, $24, $82
    .BYTE $3A, $81, $16, $30, $1A, $82, $34, $00
    .BYTE $94, $56, $42, $02, $4C, $52, $42, $5C
    .BYTE $4A, $5A, $02, $4C, $5A, $56, $02, $50
    .BYTE $4C, $5A, $02, $54, $5A, $58, $02, $50
    .BYTE $54, $4C, $42, $02, $4C, $50, $48, $4A
    .BYTE $50, $00, $8A, $08, $08, $08, $85, $3C
    .BYTE $3A, $38, $36, $3A, $38, $36, $34, $38
    .BYTE $36, $34, $32, $36, $34, $32, $30, $34
    .BYTE $32, $30, $2E, $2A, $28, $A8, $26, $00

; ----------------------------------------------------------------------------
; Starts a requested tune on square channel 1 or advances the current one by one frame.
; OUT:  CurTune1 = the tune's request byte ($80 for a request with bit 7), 0 when the script ends.
; ----------------------------------------------------------------------------
Audio_DriveTune1:
    LDA Tune1Request
    BMI @SilenceThenPlay
    BNE @ChangeTune
    LDA CurTune1
    BNE @KeepPlaying
    RTS

@SilenceThenPlay:
    JSR Song_Silence
    LDA #$80

@ChangeTune:
    STA CurTune1
    LDY #$00
:
    ; Get the index of the lowest set tune bit: 1 to 8.
    INY
    LSR
    BCC :-
    LDA Audio_Tune1Scripts-1, Y
    STA Tune1ScriptOffset
    LDA #$01
    STA NoteCounterTune1

@KeepPlaying:
    DEC NoteCounterTune1
    BNE @CheckVibrate
    LDY Tune1ScriptOffset
    INC Tune1ScriptOffset
    LDA Audio_Tune1Scripts, Y
    BMI @PrepNote
    BNE @PlayNote

    ; We've reached the end of the tune.
    ;
    ; If tune is "Game Over", then go play it again.
    LDA CurTune1
    CMP #$40
    BEQ @ChangeTune
    LDX #$90
    STX Sq1Duty
    LDX #$18
    STX Sq1Length
    LDX #$00
    STX CurTune1
    STX Sq1Timer
    RTS

@PrepNote:
    AND #$7F
    STA NoteLengthTune1
    LDY Tune1ScriptOffset
    INC Tune1ScriptOffset
    LDA Audio_Tune1Scripts, Y

@PlayNote:
    JSR Audio_EmitSquareNote1
    LDA #$7F
    STA Sq1Sweep
    LDA #$86
    STA Sq1Duty
    LDA NoteLengthTune1
    STA NoteCounterTune1
    LDA #$1F
    STA CustomEnvelopeOffsetTune1

@CheckVibrate:
    ; If the tune is not one of "flute" or "Link dying", then return.
    LDA CurTune1
    AND #$90
    BEQ @Exit
    LDY CustomEnvelopeOffsetTune1
    BEQ :+
    DEC CustomEnvelopeOffsetTune1
:
    LDA Audio_Tune1EnvelopeDuties, Y
    STA Sq1Duty
    LDA NoteCounterTune1
    LDX CurNoteLowPeriodSq1
    JSR Audio_VibratePitch
    STX Sq1Timer

@Exit:
    RTS

; TABLE: Sq1Duty values (duty $80, constant volume 5-$F) for tunes $10 and $80; index =
; CustomEnvelopeOffsetTune1, counted down from $1F to 0 during a note, so read from the end.
Audio_Tune1EnvelopeDuties:
    .BYTE $95, $96, $97, $98, $99, $9A, $9B, $9C
    .BYTE $9D, $9E, $9F, $9F, $9F, $9F, $9F, $9F
    .BYTE $9F, $9F, $9F, $9F, $9F, $9F, $9F, $9E
    .BYTE $9D, $9C, $9B, $9A, $99, $98, $97, $96

; ----------------------------------------------------------------------------
; Starts a requested DMC sample, times the one playing, and restarts looping or background samples.
; ----------------------------------------------------------------------------
Sample_Drive:
    LDA SampleRequest
    BMI @ChangeSampleMid
    BNE @ChangeSampleLow
    LDA CurSample
    BEQ @CheckBgSample
    DEC SampleCounter
    BNE @Exit
    LDA CurSample
    BMI @ChangeSampleMid
    AND #$70
    BNE @ChangeSampleLow
    LDA #$00
    STA CurSample
    LDA #$0F
    STA ApuStatus

@CheckBgSample:
    LDA BackgroundSample
    BNE :+

@Exit:
    RTS

@ChangeSampleLow:
    LDX #$00
    BEQ :+

@ChangeSampleMid:
    LDX #$7F
    AND #$F0
:
    STX DmcCounter
    STA CurSample
    TAX
    AND #$F0
    BEQ :+
    STA BackgroundSample
:
    TXA
    LDY #$00
:
    INY
    LSR
    BCC :-
    LDA Sample_DmcRates-1, Y
    STA DmcFreq
    LDA Sample_DmcAddrs-1, Y
    STA DmcAddress
    LDA Sample_DmcLengths-1, Y
    STA DmcLength
    LDA #$A0
    STA SampleCounter
    LDA #$0F
    STA ApuStatus
    LDA #$1F
    STA ApuStatus
    RTS

; TABLE: DMC sample address register values; index = sample bit number 0-6.
; Address = $C000 + value * $40.
Sample_DmcAddrs:
    .BYTE $00, $4C, $80, $1D, $20, $28, $4C

; TABLE: DMC sample length register values; index = sample bit number 0-6; bytes = value * $10 + 1.
Sample_DmcLengths:
    .BYTE $75, $C0, $40, $0A, $B0, $90, $D0

; TABLE: DMC rate register values (rate index, no loop, no IRQ); index = sample bit number 0-6.
Sample_DmcRates:
    .BYTE $0F, $0F, $0D, $0F, $0E, $0F, $0E
