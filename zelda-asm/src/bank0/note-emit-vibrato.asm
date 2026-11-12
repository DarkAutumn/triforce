; ============================================================================
; FILE:     bank0/note-emit-vibrato.asm  BANK: 0 (switchable, $8000-$BFFF)
; UPSTREAM: Z_00.asm 583-692
; SUBJECTS: Audio
; TOPIC:    Helpers that start a note on square 0, square 1 or the triangle channel from a note id,
;           set a square channel's duty and sweep registers, and wobble a pitch for vibrato.
; SEE ALSO: bank0/song-note-tables.asm (NotePeriodTable); bank0/drive-song.asm, bank0/drive-audio-tune0.asm
;           and bank0/tune1-and-samples.asm (the callers).
; KEY FACTS:
;   - A note id is an even byte offset into NotePeriodTable (2 bytes per note, high byte first). A note
;     whose period low byte is 0 is a rest: nothing is written and A returns 0. EmitSquareNote0,
;     EmitSquareNote1, EmitTriangleNote.
;   - Emitting a note writes the period low byte to the channel's timer register and the period high
;     byte OR $08 to its length register, and saves the low byte for vibrato. EmitSquareNote1.
;   - Vibrato applies while the counter passed in A is >= $10 and alternates period+1 / period-1 every
;     4 frames (bit 2 of the counter). The song passes frames since the note began (vibrato starts $10
;     frames in); tune channel 1 passes frames left (vibrato stops $0F frames before the note ends).
;     VibratePitch.
;   - EmitSquareNoteWithDutyAndSweep0 and EmitSquareNoteWithDutyAndSweep1 are not referenced anywhere.
; CONTENTS: (generated; L = line in this file)
;   L0036 SetSq0DutyAndSweep               code  Writes square channel 0's duty/volume and sweep registers.
;   L0042 EmitSquareNoteWithDutyAndSweep0  code  Unreferenced entry: sets square 0 duty (X) and sweep (Y), then falls into EmitSquareNote0.
;   L0050 EmitSquareNote0                  code  Starts a note on square channel 0 from its period table entry; does nothing for a rest.
;   L0061 Audio_EmitNote_Return            code  Shared return of the three note emit routines (taken for a rest).
;   L0068 SetSq1DutyAndSweep               code  Writes square channel 1's duty/volume and sweep registers.
;   L0074 EmitSquareNoteWithDutyAndSweep1  code  Unreferenced entry: sets square 1 duty (X) and sweep (Y), then falls into EmitSquareNote1.
;   L0082 EmitSquareNote1                  code  Starts a note on square channel 1 from its period table entry; does nothing for a rest.
;   L0098 EmitTriangleNote                 code  Starts a note on the triangle channel from its period table entry; does nothing for a rest.
;   L0115 VibratePitch                     code  Moves a period low byte up or down by 1 for vibrato, by the note's frame counter.
; ============================================================================

; ----------------------------------------------------------------------------
; Writes square channel 0's duty/volume and sweep registers.
; IN:   X = duty byte; Y = sweep byte
; ----------------------------------------------------------------------------
SetSq0DutyAndSweep:
    STY Sq0Sweep
    STX Sq0Duty
    RTS

; Unreferenced entry: sets square 0 duty (X) and sweep (Y), then falls into EmitSquareNote0.
EmitSquareNoteWithDutyAndSweep0:
    JSR SetSq0DutyAndSweep

; ----------------------------------------------------------------------------
; Starts a note on square channel 0 from its period table entry; does nothing for a rest.
; IN:   A = note id (offset of the note in NotePeriodTable)
; OUT:  Y = note id; A = 0 (Z set) if the note is a rest
; ----------------------------------------------------------------------------
EmitSquareNote0:
    TAY
    LDA NotePeriodTable+1, Y
    BEQ Audio_EmitNote_Return
    STA CurNoteLowPeriodSq0
    STA Sq0Timer
    LDA NotePeriodTable, Y
    ORA #$08
    STA Sq0Length

; Shared return of the three note emit routines (taken for a rest).
Audio_EmitNote_Return:
    RTS

; ----------------------------------------------------------------------------
; Writes square channel 1's duty/volume and sweep registers.
; IN:   X = duty byte; Y = sweep byte
; ----------------------------------------------------------------------------
SetSq1DutyAndSweep:
    STX Sq1Duty
    STY Sq1Sweep
    RTS

; Unreferenced entry: sets square 1 duty (X) and sweep (Y), then falls into EmitSquareNote1.
EmitSquareNoteWithDutyAndSweep1:
    JSR SetSq1DutyAndSweep

; ----------------------------------------------------------------------------
; Starts a note on square channel 1 from its period table entry; does nothing for a rest.
; IN:   A = note id (offset of the note in NotePeriodTable)
; OUT:  Y = note id; A = 0 (Z set) if the note is a rest
; ----------------------------------------------------------------------------
EmitSquareNote1:
    TAY
    LDA NotePeriodTable+1, Y
    BEQ Audio_EmitNote_Return
    STA CurNoteLowPeriodSq1
    STA Sq1Timer
    LDA NotePeriodTable, Y
    ORA #$08
    STA Sq1Length
    RTS

; ----------------------------------------------------------------------------
; Starts a note on the triangle channel from its period table entry; does nothing for a rest.
; IN:   A = note id (offset of the note in NotePeriodTable)
; OUT:  Y = note id; A = 0 (Z set) if the note is a rest
; ----------------------------------------------------------------------------
EmitTriangleNote:
    TAY
    LDA NotePeriodTable+1, Y
    BEQ Audio_EmitNote_Return
    STA CurNoteLowPeriodTrg
    STA TrgTimer
    LDA NotePeriodTable, Y
    ORA #$08
    STA TrgLength
    RTS

; ----------------------------------------------------------------------------
; Moves a period low byte up or down by 1 for vibrato, by the note's frame counter.
; IN:   A = counter (song: frames since the note began; tune 1: frames left); X = period low byte
; OUT:  X = period + 1 if bit 2 of A is clear, period - 1 if set; unchanged if A < $10
; NOTE: Only the low byte changes. If period + 1 wraps to 0 the code falls into the -1 path.
; ----------------------------------------------------------------------------
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
