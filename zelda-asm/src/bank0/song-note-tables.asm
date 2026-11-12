; ============================================================================
; FILE:     bank0/song-note-tables.asm   BANK: 0 (switchable, $8000-$BFFF)
; UPSTREAM: Z_00.asm 1055-1225
; SUBJECTS: (none)
; TOPIC:    Tables and small lookups of the sound driver: drum (noise) parameters, note length tables,
;           the note period table, the song volume envelope, the sword/arrow/flame/stairs effect notes;
;           then the bank's reset stub and vectors.
; SEE ALSO: bank0/drive-song.asm (the song driver that uses the lookups); bank0/sound-effects.asm
;           (DriveEffect reads the effect note tables); bank0/note-emit-vibrato.asm (reads
;           NotePeriodTable); bank0/song-data.asm (phrase headers choose the length table and envelope);
;           bank7/reset-mmc1-vectors.asm (IsrReset, the same reset code in the fixed bank).
; KEY FACTS:
;   - Note lengths are frames, 8 per table; a phrase header's base selects the table: $00
;     NoteLengthTable0 (underworld), $08 NoteLengthTable1 (ending), $10 NoteLengthTable2 (overworld and
;     the single-phrase songs), $20 NoteLengthTable4 (demo). No header uses $18 (NoteLengthTable3).
;     GetSongNoteLength.
;   - A noise track byte holds a length index in bits 0, 7, 6 and a drum sound 0-3 in bits 5-4; sound 0
;     is silent (volume $10). GetSongNoiseNoteLength, NoiseVolumes.
;   - NotePeriodTable has 57 two-byte entries (note ids $00-$70, even), period high byte first; id $08
;     is the rest (period 0).
;   - Every song note that is not a rest starts its envelope at offset $20 with duty byte $82 and sweep
;     $7F; the envelope selector loaded first is overwritten. Offset $20 is one past CustomEnvelopeSong:
;     a note's first frame reads the first byte of SwordSfxNotes ($47). PrepareCustomSongEnvelope.
;   - The envelope offset counts down, so CustomEnvelopeSong is read from its end; header envelope bit 7
;     picks the low nibbles, otherwise the high nibbles; the result is ORed with $90. ShapeSongVolume.
;   - GetSongNoteLengthWithAbsIndex is not referenced anywhere.
;   - The bytes in BANK_00_ISR are this bank's copy of the reset handler (ends JMP $E440 after switching
;     to bank 7); BANK_00_VEC holds the vectors NMI $E484, reset $BF50, IRQ $BFF0 (see the comments
;     there).
; CONTENTS: (generated; L = line in this file)
;   L0053 NoiseVolumes                   data  TABLE: noise volume register value of a song drum sound; index = sound 0-3 (0 = silent).
;   L0057 NoisePeriods                   data  TABLE: noise period register value of a song drum sound; index = sound 0-3.
;   L0061 NoiseLengths                   data  TABLE: noise length register value of a song drum sound; index = sound 0-3.
;   L0070 GetSongNoiseNoteLength         code  Gets the length in frames of a noise track byte, then falls into GetSongNoteLength.
;   L0083 GetSongNoteLength              code  Gets the note length in frames for a length control byte, from the phrase's note length table.
;   L0092 GetSongNoteLengthWithAbsIndex  code  Unreferenced routine: note length for the low 3 bits of A from NoteLengthTable0, ignoring the base.
;   L0102 StairsSfxNotes                 data  TABLE: stairs effect notes; index = step counter [68] - 1 (1-6); volume/period nibbles.
;   L0109 NotePeriodTable                data  TABLE: note periods (APU timer values); index = note id (even); high byte first; 0 = rest.
;   L0131 PrepareCustomSongEnvelope      code  Loads the starting envelope offset and the duty and sweep bytes for a new song note.
;   L0143 ShapeSongVolume                code  Computes the duty/volume byte of a song's square channel for the current envelope offset.
;   L0163 CustomEnvelopeSong             data  TABLE: song volume envelopes; index = envelope offset (counts down from $20 to 0, then stays at 0).
;   L0170 SwordSfxNotes                  data  TABLE: sword effect notes; index = EffectCounter - 1 (read from the end); volume/period nibbles.
;   L0175 ArrowSfxNotes                  data  TABLE: arrow effect notes; index = EffectCounter - 1 (read from the end); volume/period nibbles.
;   L0180 FlameSfxNotes                  data  TABLE: flame effect noise volume bytes; index = EffectCounter / 2 - 1 (read from the end).
;   L0185 NoteLengthTable0               data  TABLE: note lengths for base $00 (underworld song); index = low 3 bits of a control byte; frames.
;   L0189 NoteLengthTable1               data  TABLE: note lengths for base $08 (ending song); index = low 3 bits of a control byte; frames.
;   L0193 NoteLengthTable2               data  TABLE: note lengths for base $10 (overworld and single-phrase songs); index = low 3 bits; frames.
;   L0197 NoteLengthTable3               data  TABLE: note lengths for base $18 (no phrase header uses it); index = low 3 bits; frames.
;   L0201 NoteLengthTable4               data  TABLE: note lengths for base $20 (demo song); index = low 3 bits of a control byte; frames.
; ============================================================================

; TABLE: noise volume register value of a song drum sound; index = sound 0-3 (0 = silent).
NoiseVolumes:
    .BYTE $10, $1C, $1C, $1C

; TABLE: noise period register value of a song drum sound; index = sound 0-3.
NoisePeriods:
    .BYTE $00, $03, $0A, $03

; TABLE: noise length register value of a song drum sound; index = sound 0-3.
NoiseLengths:
    .BYTE $00, $18, $18, $58

; ----------------------------------------------------------------------------
; Gets the length in frames of a noise track byte, then falls into GetSongNoteLength.
; IN:   A = noise track byte
; OUT:  A = note length in frames; X = the original byte
; NOTE: Rotates so that bits 0, 7, 6 move to bits 2, 1, 0 (the length index).
; ----------------------------------------------------------------------------
GetSongNoiseNoteLength:
    TAX
    ROR
    TXA
    ROL
    ROL
    ROL

; ----------------------------------------------------------------------------
; Gets the note length in frames for a length control byte, from the phrase's note length table.
; IN:   A = control byte (low 3 bits = index); NoteLengthTableBase = the phrase's table offset
; OUT:  A = note length in frames; Y = table offset
; ----------------------------------------------------------------------------
GetSongNoteLength:
    AND #$07
    CLC
    ADC NoteLengthTableBase
    TAY
    LDA NoteLengthTable0, Y
    RTS

; Unreferenced routine: note length for the low 3 bits of A from NoteLengthTable0, ignoring the base.
GetSongNoteLengthWithAbsIndex:
    AND #$07
    TAY
    LDA NoteLengthTable0, Y
    RTS

; One byte that nothing references.
    .BYTE $CB

; TABLE: stairs effect notes; index = step counter [68] - 1 (1-6); volume/period nibbles.
StairsSfxNotes:
    .BYTE $0E, $0E, $4C, $6D, $8C, $CD

; One byte that nothing references.
    .BYTE $FF

; TABLE: note periods (APU timer values); index = note id (even); high byte first; 0 = rest.
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

; ----------------------------------------------------------------------------
; Loads the starting envelope offset and the duty and sweep bytes for a new song note.
; OUT:  A = $20 (envelope offset); X = $82 (duty byte); Y = $7F (sweep byte)
; NOTE: The first load (SongEnvelopeSelector) is overwritten at once; all songs get the same values.
; ----------------------------------------------------------------------------
PrepareCustomSongEnvelope:
    LDA SongEnvelopeSelector
    LDA #$20
    LDX #$82
    LDY #$7F
    RTS

; ----------------------------------------------------------------------------
; Computes the duty/volume byte of a song's square channel for the current envelope offset.
; IN:   Y = envelope offset; SongEnvelopeSelector bit 7 set = use low nibbles, clear = high nibbles
; OUT:  A = $90 OR the nibble of CustomEnvelopeSong,Y
; ----------------------------------------------------------------------------
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

; TABLE: song volume envelopes; index = envelope offset (counts down from $20 to 0, then stays at 0).
; High nibble = volume for selector bit 7 clear, low nibble = volume for selector bit 7 set.
; Offset $20 (a note's first frame) is past the 32 bytes: it reads SwordSfxNotes' first byte, $47.
CustomEnvelopeSong:
    .BYTE $04, $24, $24, $34, $34, $35, $35, $35
    .BYTE $45, $45, $46, $46, $46, $46, $46, $46
    .BYTE $46, $46, $57, $57, $57, $57, $68, $68
    .BYTE $68, $68, $79, $79, $79, $68, $68, $57

; TABLE: sword effect notes; index = EffectCounter - 1 (read from the end); volume/period nibbles.
SwordSfxNotes:
    .BYTE $47, $67, $87, $A8, $B9, $9A, $8A, $5A
    .BYTE $9B, $8B

; TABLE: arrow effect notes; index = EffectCounter - 1 (read from the end); volume/period nibbles.
ArrowSfxNotes:
    .BYTE $FB, $F9, $9D, $6E, $3F

; TABLE: flame effect noise volume bytes; index = EffectCounter / 2 - 1 (read from the end).
; On the last frame (counter 1) the index is -1: the byte read is the last of ArrowSfxNotes ($3F).
FlameSfxNotes:
    .BYTE $1A, $1A, $1C, $1D, $1D, $1E, $1E, $1F
    .BYTE $1F, $1E, $1A, $19, $16, $13, $11, $11

; TABLE: note lengths for base $00 (underworld song); index = low 3 bits of a control byte; frames.
NoteLengthTable0:
    .BYTE $03, $0A, $01, $14, $05, $28, $3C, $70

; TABLE: note lengths for base $08 (ending song); index = low 3 bits of a control byte; frames.
NoteLengthTable1:
    .BYTE $07, $1B, $35, $14, $0D, $28, $3C, $50

; TABLE: note lengths for base $10 (overworld and single-phrase songs); index = low 3 bits; frames.
NoteLengthTable2:
    .BYTE $06, $0C, $08, $18, $24, $30, $48, $10

; TABLE: note lengths for base $18 (no phrase header uses it); index = low 3 bits; frames.
NoteLengthTable3:
    .BYTE $07, $0D, $09, $1B, $24, $36, $48, $10

; TABLE: note lengths for base $20 (demo song); index = low 3 bits of a control byte; frames.
NoteLengthTable4:
    .BYTE $3C, $50, $0A, $05, $14, $0D, $28, $0E

.SEGMENT "BANK_00_ISR"




; This bank's copy of the reset handler, at $BF50 (machine code as data). It disables NMI, sets the
; stack, waits for two vertical blanks, resets the MMC1 shift register, JSRs $BF98 with $0F (MMC1
; control), writes 0 to $A000 serially, JSRs $BFAC with 7 (switch to bank 7) and jumps to $E440.
; $BF98 (row 10): five serial writes of A to $8000, RTS. $BFAC (row 12, byte 5): the same to $E000.
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




; Vectors: NMI $E484, reset $BF50 (the stub above), IRQ $BFF0.
    .BYTE $84, $E4, $50, $BF, $F0, $BF
