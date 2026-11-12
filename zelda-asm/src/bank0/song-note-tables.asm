; ============================================================================
; FILE:     bank0/song-note-tables.asm   BANK: 0 (switchable, $8000-$BFFF)
; UPSTREAM: Z_00.asm 1055-1225
; SUBJECTS: (none)
; TOPIC:    Noise/note-length/period tables, envelopes; ISR stub, vectors
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0030 NoiseVolumes                   data
;   L0033 NoisePeriods                   data
;   L0036 NoiseLengths                   data
;   L0048 GetSongNoiseNoteLength         code  Params:
;   L0062 GetSongNoteLength              code  Params:
;   L0070 GetSongNoteLengthWithAbsIndex  code
;   L0081 StairsSfxNotes                 data
;   L0089 NotePeriodTable                data  Big-endian 16-bit period values.
;   L0111 PrepareCustomSongEnvelope      code  Returns:
;   L0124 ShapeSongVolume                code  Params:
;   L0141 CustomEnvelopeSong             data
;   L0147 SwordSfxNotes                  data
;   L0151 ArrowSfxNotes                  data
;   L0154 FlameSfxNotes                  data
;   L0158 NoteLengthTable0               data
;   L0161 NoteLengthTable1               data
;   L0164 NoteLengthTable2               data
;   L0167 NoteLengthTable3               data
;   L0170 NoteLengthTable4               data
; ============================================================================

NoiseVolumes:
    .BYTE $10, $1C, $1C, $1C

NoisePeriods:
    .BYTE $00, $03, $0A, $03

NoiseLengths:
    .BYTE $00, $18, $18, $58

; Params:
; A: control note
;
; Returns:
; A: note length
; X: original control note
;
; Rotate so that bits 0, 7, 6 move to bits 2, 1, 0.
;
GetSongNoiseNoteLength:
    TAX
    ROR
    TXA
    ROL
    ROL
    ROL

; Params:
; A: control note
;
; Returns:
; A: note length
;
GetSongNoteLength:
    AND #$07
    CLC
    ADC NoteLengthTableBase
    TAY
    LDA NoteLengthTable0, Y
    RTS

GetSongNoteLengthWithAbsIndex:
    ; Begin unverified code 1EF1
    AND #$07
    TAY
    LDA NoteLengthTable0, Y
    RTS
    ; End unverified code

; Unknown block
    .BYTE $CB

StairsSfxNotes:
    .BYTE $0E, $0E, $4C, $6D, $8C, $CD

; Unknown block
    .BYTE $FF

; Big-endian 16-bit period values.
;
NotePeriodTable:
    .BYTE $00, $23, $00, $6A, $03, $27, $00, $97
    .BYTE $00, $00, $02, $F9, $02, $CF, $02, $A6
    .BYTE $02, $80, $02, $5C, $02, $3A, $02, $1A
    .BYTE $01, $FC, $01, $DF, $01, $C4, $01, $AB
    .BYTE $01, $93, $01, $7C, $01, $67, $01, $53
    .BYTE $01, $40, $01, $2E, $01, $1D, $01, $0D
    .BYTE $00, $FE, $00, $EF, $00, $E2, $00, $D5
    .BYTE $00, $C9, $00, $BE, $00, $B3, $00, $A9
    .BYTE $00, $A0, $00, $8E, $00, $86, $00, $77
    .BYTE $00, $7E, $00, $71, $00, $54, $00, $64
    .BYTE $00, $5F, $00, $59, $00, $50, $00, $47
    .BYTE $00, $43, $00, $3F, $00, $38, $00, $32
    .BYTE $00, $21, $05, $4D, $05, $01, $04, $B9
    .BYTE $04, $35, $03, $F8, $03, $BF, $03, $89
    .BYTE $03, $57

; Returns:
; A: starting custom envelope offset
; X: duty byte $82
; Y: sweep byte $7F
;
PrepareCustomSongEnvelope:
    LDA SongEnvelopeSelector
    LDA #$20
    LDX #$82
    LDY #$7F
    RTS

; Params:
; Y: envelope offset
;
; Returns:
; A: duty/volume value
;
ShapeSongVolume:
    LDA SongEnvelopeSelector
    BPL :+
    LDA CustomEnvelopeSong, Y
    AND #$0F
    BNE @ReturnValue
:
    LDA CustomEnvelopeSong, Y
    LSR
    LSR
    LSR
    LSR

@ReturnValue:
    ORA #$90
    RTS

CustomEnvelopeSong:
    .BYTE $04, $24, $24, $34, $34, $35, $35, $35
    .BYTE $45, $45, $46, $46, $46, $46, $46, $46
    .BYTE $46, $46, $57, $57, $57, $57, $68, $68
    .BYTE $68, $68, $79, $79, $79, $68, $68, $57

SwordSfxNotes:
    .BYTE $47, $67, $87, $A8, $B9, $9A, $8A, $5A
    .BYTE $9B, $8B

ArrowSfxNotes:
    .BYTE $FB, $F9, $9D, $6E, $3F

FlameSfxNotes:
    .BYTE $1A, $1A, $1C, $1D, $1D, $1E, $1E, $1F
    .BYTE $1F, $1E, $1A, $19, $16, $13, $11, $11

NoteLengthTable0:
    .BYTE $03, $0A, $01, $14, $05, $28, $3C, $70

NoteLengthTable1:
    .BYTE $07, $1B, $35, $14, $0D, $28, $3C, $50

NoteLengthTable2:
    .BYTE $06, $0C, $08, $18, $24, $30, $48, $10

NoteLengthTable3:
    .BYTE $07, $0D, $09, $1B, $24, $36, $48, $10

NoteLengthTable4:
    .BYTE $3C, $50, $0A, $05, $14, $0D, $28, $0E

.SEGMENT "BANK_00_ISR"




; Unknown block
    .BYTE $78, $D8, $A9, $00, $8D, $00, $20, $A2
    .BYTE $FF, $9A, $AD, $02, $20, $29, $80, $F0
    .BYTE $F9, $AD, $02, $20, $29, $80, $F0, $F9
    .BYTE $09, $FF, $8D, $00, $80, $8D, $00, $A0
    .BYTE $8D, $00, $C0, $8D, $00, $E0, $A9, $0F
    .BYTE $20, $98, $BF, $A9, $00, $8D, $00, $A0
    .BYTE $4A, $8D, $00, $A0, $4A, $8D, $00, $A0
    .BYTE $4A, $8D, $00, $A0, $4A, $8D, $00, $A0
    .BYTE $A9, $07, $20, $AC, $BF, $4C, $40, $E4
    .BYTE $8D, $00, $80, $4A, $8D, $00, $80, $4A
    .BYTE $8D, $00, $80, $4A, $8D, $00, $80, $4A
    .BYTE $8D, $00, $80, $60, $8D, $00, $E0, $4A
    .BYTE $8D, $00, $E0, $4A, $8D, $00, $E0, $4A
    .BYTE $8D, $00, $E0, $4A, $8D, $00, $E0, $60

.SEGMENT "BANK_00_VEC"




; Unknown block
    .BYTE $84, $E4, $50, $BF, $F0, $BF
