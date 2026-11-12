; ============================================================================
; FILE:     bank0/note-emit-vibrato.asm  BANK: 0 (switchable, $8000-$BFFF)
; UPSTREAM: Z_00.asm 583-692
; SUBJECTS: (none)
; TOPIC:    Square and triangle note emit, duty/sweep, vibrato
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0024 SetSq0DutyAndSweep               code  Params:
;   L0029 EmitSquareNoteWithDutyAndSweep0  code
;   L0037 EmitSquareNote0                  code  Params:
;   L0047 Audio_EmitNote_Return            code
;   L0054 SetSq1DutyAndSweep               code  Params:
;   L0059 EmitSquareNoteWithDutyAndSweep1  code
;   L0071 EmitSquareNote1                  code  Params:
;   L0085 EmitTriangleNote                 code  Params:
;   L0105 VibratePitch                     code  Params:
; ============================================================================

; Params:
; X: duty byte
; Y: sweep byte
;
SetSq0DutyAndSweep:
    STY Sq0Sweep_4001
    STX Sq0Duty_4000
    RTS

EmitSquareNoteWithDutyAndSweep0:
    ; Begin unverified code 1C0A
    JSR SetSq0DutyAndSweep
    ; End unverified code

; Params:
; A: offset of a note in period table
;
EmitSquareNote0:
    TAY
    LDA NotePeriodTable+1, Y
    BEQ Audio_EmitNote_Return
    STA CurNoteLowPeriodSq0
    STA Sq0Timer_4002
    LDA NotePeriodTable, Y
    ORA #$08
    STA Sq0Length_4003

Audio_EmitNote_Return:
    RTS

; Params:
; X: duty byte
; Y: sweep byte
;
SetSq1DutyAndSweep:
    STX Sq1Duty_4004
    STY Sq1Sweep_4005
    RTS

EmitSquareNoteWithDutyAndSweep1:
    ; Begin unverified code 1C28
    JSR SetSq1DutyAndSweep
    ; End unverified code

; Params:
; A: note ID (offset of note in period table)
;
; Returns:
; Y: note ID
; A: 0 if the note is a a rest
;
EmitSquareNote1:
    TAY
    LDA NotePeriodTable+1, Y
    BEQ Audio_EmitNote_Return
    STA CurNoteLowPeriodSq1
    STA Sq1Timer_4006
    LDA NotePeriodTable, Y
    ORA #$08
    STA Sq1Length_4007
    RTS

; Params:
; A: offset of a note in period table
;
EmitTriangleNote:
    TAY
    LDA NotePeriodTable+1, Y
    BEQ Audio_EmitNote_Return
    STA CurNoteLowPeriodTrg
    STA TrgTimer_400A
    LDA NotePeriodTable, Y
    ORA #$08
    STA TrgLength_400B
    RTS

; Params:
; A: note counter
; X: period value
;
; Returns:
; X: (period - 1) to (period + 1)
;
; If note counter < $10, then return.
;
VibratePitch:
    CMP #$10
    BCC @Exit

    ; Is bit 2 set?
    LSR
    LSR
    LSR
    BCS @GoDown

    ; If not set, then add 1 to X.
    TXA
    ADC #$01
    BNE :+

@GoDown:
    ; If set, then subtract 1 from X.
    TXA
    CLC
    ADC #$FF
:
    TAX

@Exit:
    RTS
