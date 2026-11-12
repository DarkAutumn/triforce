; ============================================================================
; FILE:     bank0/song-data.asm          BANK: 0 (switchable, $8000-$BFFF)
; UPSTREAM: Z_00.asm 9-104
; SUBJECTS: (none)
; TOPIC:    Music data: the table that maps songs and phrase numbers to phrase headers, the 8-byte phrase
;           headers of every song, and the song scripts (note streams) they point into.
; SEE ALSO: bank0/drive-song.asm (DriveSong reads all of this: phrase order, header fields, script bytes);
;           bank0/song-note-tables.asm (note lengths, periods, envelopes the scripts index);
;           bank7/update-mode-dispatch.asm (LevelSongIds: which song a level requests).
; KEY FACTS:
;   - A song is one bit of SongRequest/CurSong: $01 overworld, $02 Ganon, $04 end level, $08 item taken,
;     $10 ending, $20 last level, $40 underworld, $80 demo; $06 is Zelda's song. Header labels below and
;     the dispatch in DriveSong.
;   - SongTable holds byte offsets (from SongTable itself) of phrase headers; DriveSong indexes it from 1.
;     Entries 2, 3, 4, 6 are the single-phrase songs $02, $04, $08, $20; entry 36 is Zelda's song.
;   - Phrase order: overworld plays entry 9 once, then loops entries 10-15; underworld loops 16-17;
;     ending plays 18-20 once, then loops 21-25; demo loops 26-35. SongTable with DriveSong.
;   - A phrase header is 8 bytes: note length table base, script address (2), triangle offset, square 0
;     offset, noise offset, envelope selector, triangle sustain flag. SongHeaderDemo0 comment.
;   - The item-taken and end-level headers are 7 bytes long; their 8th byte is the first byte of the
;     header that follows. SongHeaderItemTaken0, SongHeaderEndLevel0.
;   - Each multi-phrase song keeps all its phrases in one script blob; a phrase header points at its own
;     start address inside the blob (two ending phrases share one start). SongHeaderOverworld0,
;     SongHeaderEnding0.
; CONTENTS: (generated; L = line in this file)
;   L0052 SongTable              data  TABLE: offset of a phrase header from SongTable; index = song number or phrase index, minus 1.
;   L0074 SongHeaderDemo0        data  TABLE: phrase headers of the demo (title and story) song $80; index = phrase 0-7 * 8; 8 bytes each.
;   L0089 SongHeaderItemTaken0   data  DATA: phrase header of the item-taken song $08 (7 bytes; byte 7 is the next header's first byte).
;   L0093 SongHeaderEndLevel0    data  DATA: phrase header of the end-level song $04 (7 bytes; byte 7 is the next header's first byte).
;   L0098 SongHeaderOverworld0   data  TABLE: phrase headers of the overworld song $01; index = phrase 0-4 * 8; 8 bytes each.
;   L0106 SongHeaderUnderworld0  data  TABLE: phrase headers of the underworld song $40; index = phrase 0-1 * 8; 8 bytes each.
;   L0111 SongHeaderLastLevel0   data  DATA: phrase header of the last-level song $20 (one phrase, repeated).
;   L0115 SongHeaderGanon0       data  DATA: phrase header of the Ganon song $02 (one phrase, played once).
;   L0119 SongHeaderEnding0      data  TABLE: phrase headers of the ending song $10; index = phrase 0-6 * 8; 8 bytes each.
;   L0129 SongHeaderZelda        data  DATA: phrase header of Zelda's song $06 (one phrase, played once).
;   L0133 SongScriptItemTaken0   data  DATA: song script of the item-taken song (19 bytes).
;   L0137 SongScriptOverworld0   data  DATA: song script shared by the overworld phrases (621 bytes).
;   L0141 SongScriptUnderworld0  data  DATA: song script shared by the underworld phrases (199 bytes).
;   L0145 SongScriptEndLevel0    data  DATA: song script of the end-level song (89 bytes).
;   L0149 SongScriptLastLevel0   data  DATA: song script of the last-level song (207 bytes).
;   L0153 SongScriptGanon0       data  DATA: song script of the Ganon song (43 bytes).
;   L0157 SongScriptEnding0      data  DATA: song script shared by the ending phrases (404 bytes).
;   L0161 SongScriptDemo0        data  DATA: song script shared by the demo phrases (825 bytes).
;   L0165 SongScriptZelda0       data  DATA: song script of Zelda's song (97 bytes).
; ============================================================================

; TABLE: offset of a phrase header from SongTable; index = song number or phrase index, minus 1.
; Entries 1-8 belong to the songs with bit 0-7 (lowest set bit + 1); only 2, 3, 4 and 6 are read that way.
; Entries 9-15 overworld phrases, 16-17 underworld, 18-25 ending, 26-35 demo, 36 Zelda's song.
; Entries 1, 5, 7 and 8 are not read for any song value in use: songs $01, $10, $40 and $80 take their
; phrases from the ranges above.
SongTable:
    .BYTE $7D, $B5, $6E, $67, $7D, $AD, $64, $64
    .BYTE $75, $7D, $85, $95, $7D, $8D, $95, $9D
    .BYTE $A5, $BD, $C5, $CD, $D5, $DD, $D5, $E5
    .BYTE $ED, $24, $2C, $34, $3C, $44, $34, $4C
    .BYTE $54, $5C, $44, $F5

; TABLE: phrase headers of the demo (title and story) song $80; index = phrase 0-7 * 8; 8 bytes each.
; Every phrase header in this file has this format (read by PrepPhrase):
;
; 0: Note length table base (offset from NoteLengthTable0)
; 1: Song script address low
; 2: Song script address high
; 3: Triangle track offset in script (0 = no triangle track)
; 4: Square 0 track offset in script (0 = no square 0 track)
; 5: Noise track offset in script (only songs $01 overworld, $10 ending and $80 demo play it)
; 6: Envelope selector: bit 7 set = low nibbles of CustomEnvelopeSong, clear = high nibbles
; 7: Triangle sustain: bit 7 set = linear counter $1F (short notes), clear = $FF (held notes)
;
; The square 1 track always starts at script offset 0.
; Note that the last byte of the "Item taken" header overlaps
; "End level", and "End level" overlaps "Overworld".
SongHeaderDemo0:
    .BYTE $20, $8B, $94, $3B, $1D, $4F, $80, $01
    .BYTE $20, $DC, $94, $27, $57, $23, $01, $80
    .BYTE $20, $A1, $95, $38, $17, $B8, $80, $80
    .BYTE $20, $F1, $95, $6C, $26, $68, $80, $80
    .BYTE $20, $8D, $96, $3E, $25, $21, $80, $80
    .BYTE $20, $EB, $96, $19, $0D, $31, $80, $80
    .BYTE $20, $20, $97, $3F, $27, $7C, $80, $80
    .BYTE $20, $8F, $97, $1D, $11, $0D, $80, $80

; First 3 bytes of an unreferenced phrase header that overlaps the "Item taken" header. Only SongTable
; entries 7 and 8 point here, and DriveSong never reads those.
    .BYTE $10, $A1, $8E

; DATA: phrase header of the item-taken song $08 (7 bytes; byte 7 is the next header's first byte).
SongHeaderItemTaken0:
    .BYTE $10, $5D, $8E, $0D, $07, $00, $80

; DATA: phrase header of the end-level song $04 (7 bytes; byte 7 is the next header's first byte).
SongHeaderEndLevel0:
    .BYTE $10, $A4, $91, $46, $22, $00, $80

; TABLE: phrase headers of the overworld song $01; index = phrase 0-4 * 8; 8 bytes each.
; Phrase 0 is the intro, played once.
SongHeaderOverworld0:
    .BYTE $10, $70, $8E, $32, $5D, $8E, $01, $80
    .BYTE $10, $0F, $8F, $35, $16, $CE, $01, $80
    .BYTE $10, $55, $8F, $60, $26, $88, $01, $80
    .BYTE $10, $32, $90, $59, $2D, $A4, $01, $80
    .BYTE $10, $E4, $8F, $3B, $1A, $F2, $01, $80

; TABLE: phrase headers of the underworld song $40; index = phrase 0-1 * 8; 8 bytes each.
SongHeaderUnderworld0:
    .BYTE $00, $DD, $90, $45, $22, $00, $01, $01
    .BYTE $00, $3A, $91, $39, $1C, $00, $01, $01

; DATA: phrase header of the last-level song $20 (one phrase, repeated).
SongHeaderLastLevel0:
    .BYTE $10, $FD, $91, $A5, $53, $CD, $80, $80

; DATA: phrase header of the Ganon song $02 (one phrase, played once).
SongHeaderGanon0:
    .BYTE $10, $CC, $92, $22, $10, $00, $80, $01

; TABLE: phrase headers of the ending song $10; index = phrase 0-6 * 8; 8 bytes each.
SongHeaderEnding0:
    .BYTE $08, $F7, $92, $22, $50, $59, $01, $80
    .BYTE $08, $F7, $92, $2F, $50, $59, $01, $80
    .BYTE $08, $52, $93, $7A, $1B, $C2, $80, $80
    .BYTE $08, $86, $93, $46, $24, $8E, $01, $80
    .BYTE $08, $9D, $93, $44, $23, $77, $01, $80
    .BYTE $08, $ED, $93, $1B, $0E, $27, $01, $80
    .BYTE $08, $1A, $94, $40, $1A, $6B, $80, $80

; DATA: phrase header of Zelda's song $06 (one phrase, played once).
SongHeaderZelda:
    .BYTE $10, $C4, $97, $3F, $20, $00, $80, $80

; DATA: song script of the item-taken song (19 bytes).
SongScriptItemTaken0:
    .INCBIN "dat/SongScriptItemTaken0.dat"

; DATA: song script shared by the overworld phrases (621 bytes).
SongScriptOverworld0:
    .INCBIN "dat/SongScriptOverworld0.dat"

; DATA: song script shared by the underworld phrases (199 bytes).
SongScriptUnderworld0:
    .INCBIN "dat/SongScriptUnderworld0.dat"

; DATA: song script of the end-level song (89 bytes).
SongScriptEndLevel0:
    .INCBIN "dat/SongScriptEndLevel0.dat"

; DATA: song script of the last-level song (207 bytes).
SongScriptLastLevel0:
    .INCBIN "dat/SongScriptLastLevel0.dat"

; DATA: song script of the Ganon song (43 bytes).
SongScriptGanon0:
    .INCBIN "dat/SongScriptGanon0.dat"

; DATA: song script shared by the ending phrases (404 bytes).
SongScriptEnding0:
    .INCBIN "dat/SongScriptEnding0.dat"

; DATA: song script shared by the demo phrases (825 bytes).
SongScriptDemo0:
    .INCBIN "dat/SongScriptDemo0.dat"

; DATA: song script of Zelda's song (97 bytes).
SongScriptZelda0:
    .INCBIN "dat/SongScriptZelda0.dat"
