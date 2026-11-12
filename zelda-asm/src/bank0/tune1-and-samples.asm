; ============================================================================
; FILE:     bank0/tune1-and-samples.asm  BANK: 0 (switchable, $8000-$BFFF)
; UPSTREAM: Z_00.asm 405-581
; SUBJECTS: (none)
; TOPIC:    Tune channel 1 scripts and driver; DMC samples
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0018 TuneScripts1         data
;   L0036 DriveTune1           code
;   L0122 CustomEnvelopeTune1  data
;   L0128 DriveSample          code
;   L0187 SampleAddrs          data
;   L0190 SampleLengths        data
;   L0193 SampleRates          data
; ============================================================================

TuneScripts1:
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

DriveTune1:
    LDA Tune1Request
    BMI @SilenceThenPlay
    BNE @ChangeTune
    LDA CurTune1
    BNE @KeepPlaying
    RTS

@SilenceThenPlay:
    JSR SilenceSong
    LDA #$80

@ChangeTune:
    STA CurTune1
    LDY #$00
:
    ; Get the index for the song bit: 1 to 8.
    INY
    LSR
    BCC :-
    LDA TuneScripts1-1, Y
    STA Tune1ScriptOffset
    LDA #$01
    STA NoteCounterTune1

@KeepPlaying:
    DEC NoteCounterTune1
    BNE @CheckVibrate
    LDY Tune1ScriptOffset
    INC Tune1ScriptOffset
    LDA TuneScripts1, Y
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
    LDA TuneScripts1, Y

@PlayNote:
    JSR EmitSquareNote1
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
    LDA CustomEnvelopeTune1, Y
    STA Sq1Duty
    LDA NoteCounterTune1
    LDX CurNoteLowPeriodSq1
    JSR VibratePitch
    STX Sq1Timer

@Exit:
    RTS

CustomEnvelopeTune1:
    .BYTE $95, $96, $97, $98, $99, $9A, $9B, $9C
    .BYTE $9D, $9E, $9F, $9F, $9F, $9F, $9F, $9F
    .BYTE $9F, $9F, $9F, $9F, $9F, $9F, $9F, $9E
    .BYTE $9D, $9C, $9B, $9A, $99, $98, $97, $96

DriveSample:
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
    LDA SampleRates-1, Y
    STA DmcFreq
    LDA SampleAddrs-1, Y
    STA DmcAddress
    LDA SampleLengths-1, Y
    STA DmcLength
    LDA #$A0
    STA SampleCounter
    LDA #$0F
    STA ApuStatus
    LDA #$1F
    STA ApuStatus
    RTS

SampleAddrs:
    .BYTE $00, $4C, $80, $1D, $20, $28, $4C

SampleLengths:
    .BYTE $75, $C0, $40, $0A, $B0, $90, $D0

SampleRates:
    .BYTE $0F, $0F, $0D, $0F, $0E, $0F, $0E
