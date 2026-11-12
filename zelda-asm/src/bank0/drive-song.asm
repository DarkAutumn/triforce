; ============================================================================
; FILE:     bank0/drive-song.asm         BANK: 0 (switchable, $8000-$BFFF)
; UPSTREAM: Z_00.asm 694-1053
; SUBJECTS: (none)
; TOPIC:    The music driver: starts a requested song, steps through its phrases, and plays one frame
;           of the square 1, square 0, triangle and noise tracks of the current phrase.
; SEE ALSO: bank0/song-data.asm (SongTable, phrase headers and their format, scripts);
;           bank0/song-note-tables.asm (note lengths, envelopes, noise parameters);
;           bank0/note-emit-vibrato.asm (note emit, VibratePitch); bank0/drive-audio-tune0.asm (DriveAudio).
; KEY FACTS:
;   - SongRequest/CurSong values: $01 overworld, $40 underworld, $10 ending and $80 demo are multi-phrase
;     songs; $06 is Zelda's song (SongTable entry $24); any other value is a single phrase picked by its
;     lowest set bit. DriveSong, PlayNextPhrase.
;   - First phrase / loop range (SongTable entries): overworld 9 then 10-15; underworld 16-17; ending
;     18-20 then 21-25; demo 26-35. DriveSong@PlayFirst..., PlayNextPhrase.
;   - The square 1 track starts at script offset 0 and its $00 byte ends the phrase for every track.
;     Songs with a bit of $F1 set ($01, $10, $20, $40, $80) then play their next phrase or repeat;
;     all others stop through SilenceSong. KeepPlayingSong@SongEnded.
;   - Script bytes on the square and triangle tracks: bit 7 set = note length control (low 3 bits index
;     the phrase's note length table) followed by a note; otherwise a note id. PrepNote, PlayNote.
;   - Triangle track only: $F1-$FF start a passage repeated (value - $F0) times, $F0 ends the passage,
;     $00 writes 0 to TrgLinear. ApplySq1Effects@PrepNoteOrPassage.
;   - The noise track plays only for songs $01, $10 and $80 and loops on its own $00 byte. One byte
;     gives both the length and one of 4 drum sounds. ApplySq1Effects@HandleNoise.
;   - While CurTune1 (CurTune0) is nonzero the song keeps time but does not write square 1 (square 0).
;     PlayNote, ApplySq1Effects@PlaySq0.
;   - Only the demo song ($80) gets vibrato on the square channels; the triangle always does.
;     ApplySq1Effects.
;   - SilenceSong clears CurSong and cuts every channel (ApuStatus 0 then $0F), which also stops a
;     playing DMC sample. SilenceSong.
; CONTENTS: (generated; L = line in this file)
;   L0046 KeepPlaying         code  Jump to KeepPlayingSong (within branch range of DriveSong).
;   L0053 DriveSong           code  Starts the song in SongRequest, or plays one frame of the current song.
;   L0089 SetPrevPhraseIndex  code  Stores Y as the phrase index before the first phrase of a multi-phrase song (or its loop point).
;   L0093 PlayNextPhrase      code  Picks the next phrase of the song in A: advances SongPhraseIndex, wrapping to the song's loop point.
;   L0145 PrepPhrase          code  Loads the phrase header for SongTable entry Y (1-based) and resets all track positions and counters.
;   L0174 KeepPlayingSong     code  Plays one frame of the current phrase, starting with the square 1 track.
;   L0194 SilenceSong         code  Stops the song: clears CurSong and disables, then re-enables, the four tone and noise channels.
;   L0203 PlayAgain           code  Jump to PlayNextPhrase for a song that repeats or continues.
;   L0207 PrepNote            code  Handles a square 1 length control byte: sets the note length, then reads the note that follows.
;   L0215 PlayNote            code  Starts the square 1 note in A (unless tune channel 1 is playing) and reloads its note counter.
;   L0234 ApplySq1Effects     code  Per-frame volume envelope and vibrato for square 1; then the square 0, triangle and noise tracks.
; ============================================================================

; Jump to KeepPlayingSong (within branch range of DriveSong).
KeepPlaying:
    JMP KeepPlayingSong

; ----------------------------------------------------------------------------
; Starts the song in SongRequest, or plays one frame of the current song.
; NOTE: The routine runs on through the labels below to the end of this file.
; ----------------------------------------------------------------------------
DriveSong:
    LDA SongRequest
    BNE @ChangeSong
    LDA CurSong
    BNE KeepPlaying
    RTS

@ChangeSong:
    STA CurSong
    BMI @PlayFirstDemoPhrase
    CMP #$06
    BNE :+
    LDY #$24
    BNE PrepPhrase
:
    CMP #$01
    BEQ @PlayFirstOverworldPhrase
    CMP #$40
    BEQ @PlayFirstUnderworldPhrase
    CMP #$10
    BNE PlayNextPhrase
    LDY #$11
    BNE SetPrevPhraseIndex

@PlayFirstDemoPhrase:
    LDY #$19
    BNE SetPrevPhraseIndex

@PlayFirstUnderworldPhrase:
    LDY #$0F
    BNE SetPrevPhraseIndex

@PlayFirstOverworldPhrase:
    LDY #$08

; Stores Y as the phrase index before the first phrase of a multi-phrase song (or its loop point).
SetPrevPhraseIndex:
    STY SongPhraseIndex

; Picks the next phrase of the song in A: advances SongPhraseIndex, wrapping to the song's loop point.
PlayNextPhrase:
    TAX
    BMI @PlayNextDemoPhrase
    CMP #$01
    BEQ @PlayNextOverworldPhrase
    CMP #$40
    BEQ @PlayNextUnderworldPhrase
    CMP #$10
    BNE @PlaySinglePhraseSong

    ; Play the next phrase of the ending song.
    INC SongPhraseIndex
    LDY SongPhraseIndex
    CPY #$1A
    BNE PrepPhrase
    LDY #$14
    BNE SetPrevPhraseIndex

@PlayNextUnderworldPhrase:
    INC SongPhraseIndex
    LDY SongPhraseIndex
    CPY #$12
    BNE PrepPhrase
    LDY #$0F
    BNE SetPrevPhraseIndex

@PlayNextOverworldPhrase:
    INC SongPhraseIndex
    LDY SongPhraseIndex
    CPY #$10
    BNE PrepPhrase
    LDY #$09
    BNE SetPrevPhraseIndex

@PlayNextDemoPhrase:
    INC SongPhraseIndex
    LDY SongPhraseIndex
    CPY #$24
    BNE PrepPhrase
    LDY #$19
    BNE SetPrevPhraseIndex

@PlaySinglePhraseSong:
    ; Get the index for the song bit: 2 to 6.
    TXA
    LDY #$00
:
    INY
    LSR
    BCC :-

; Loads the phrase header for SongTable entry Y (1-based) and resets all track positions and counters.
PrepPhrase:
    LDA SongTable-1, Y
    TAY
    LDA SongTable, Y
    STA NoteLengthTableBase
    LDA SongTable+1, Y
    STA SongScriptPtrLo
    LDA SongTable+2, Y
    STA SongScriptPtrHi
    LDA SongTable+3, Y
    STA NoteOffsetSongTrg
    LDA SongTable+4, Y
    STA NoteOffsetSongSq0
    LDA SongTable+5, Y
    STA NoteOffsetSongNoise
    STA FirstNoteIndexSongNoise
    LDA SongTable+6, Y
    STA SongEnvelopeSelector
    LDA SongTable+7, Y
    STA $05F1                   ; Header byte 7: triangle sustain flag (see @ApplyTrgEffects).
    LDA #$01
    STA NoteCounterSongSq1
    STA NoteCounterSongSq0
    STA NoteCounterSongTrg
    STA NoteCounterSongNoise
    LSR
    STA NoteOffsetSongSq1

; Plays one frame of the current phrase, starting with the square 1 track.
KeepPlayingSong:
    DEC NoteCounterSongSq1
    BNE ApplySq1Effects
    LDY NoteOffsetSongSq1
    INC NoteOffsetSongSq1
    LDA (SongScriptPtrLo), Y
    BEQ @SongEnded
    BPL PlayNote
    BNE PrepNote

@SongEnded:
    ; If this is a song that repeats, then go play again.
    LDA CurSong
    AND #$F1
    BNE PlayAgain

; ----------------------------------------------------------------------------
; Stops the song: clears CurSong and disables, then re-enables, the four tone and noise channels.
; NOTE: Also called for Tune0Request $80 and Tune1Request bit 7. The write of 0 stops the DMC sample.
; ----------------------------------------------------------------------------
SilenceSong:
    LDA #$00
    STA CurSong
    STA ApuStatus
    LDA #$0F
    STA ApuStatus
    RTS

; Jump to PlayNextPhrase for a song that repeats or continues.
PlayAgain:
    JMP PlayNextPhrase

; Handles a square 1 length control byte: sets the note length, then reads the note that follows.
PrepNote:
    JSR GetSongNoteLength
    STA NoteLengthSongSq1
    LDY NoteOffsetSongSq1
    INC NoteOffsetSongSq1
    LDA (SongScriptPtrLo), Y

; Starts the square 1 note in A (unless tune channel 1 is playing) and reloads its note counter.
PlayNote:
    ; If something is playing in tune channel 1, then
    ; don't play a square note here.
    LDX CurTune1
    BNE @SkipSq1
    JSR EmitSquareNote1
    BEQ :+
    JSR PrepareCustomSongEnvelope
:
    STA CustomEnvelopeOffsetSongSq1
    JSR SetSq1DutyAndSweep
    LDA #$00
    STA SongVibrationCounterSq1

@SkipSq1:
    LDA NoteLengthSongSq1
    STA NoteCounterSongSq1

; Per-frame volume envelope and vibrato for square 1; then the square 0, triangle and noise tracks.
ApplySq1Effects:
    ; If something is playing in tune channel 1, then
    ; skip effects for square channel 1.
    LDY CurTune1
    BNE @HandleSq0
    INC SongVibrationCounterSq1
    LDY CustomEnvelopeOffsetSongSq1
    BEQ :+
    DEC CustomEnvelopeOffsetSongSq1
:
    JSR ShapeSongVolume
    STA Sq1Duty
    LDX #$7F
    STX Sq1Sweep
    LDA CurSong
    BPL @HandleSq0

    ; The demo/title song ($80) vibrates the pitch.
    LDA SongVibrationCounterSq1
    LDX CurNoteLowPeriodSq1
    JSR VibratePitch
    STX Sq1Timer

@HandleSq0:
    LDY NoteOffsetSongSq0
    BEQ @HandleTrg
    DEC NoteCounterSongSq0
    BNE @ApplySq0Effects
    LDY NoteOffsetSongSq0
    INC NoteOffsetSongSq0
    LDA (SongScriptPtrLo), Y
    BPL @PlaySq0
    JSR GetSongNoteLength
    STA NoteLengthSongSq0
    LDY NoteOffsetSongSq0
    INC NoteOffsetSongSq0
    LDA (SongScriptPtrLo), Y

@PlaySq0:
    LDX CurTune0
    BNE @SkipSq0
    JSR EmitSquareNote0
    BEQ :+
    JSR PrepareCustomSongEnvelope
:
    STA CustomEnvelopeOffsetSongSq0
    JSR SetSq0DutyAndSweep
    LDA #$00
    STA SongVibrationCounterSq0

@SkipSq0:
    LDA NoteLengthSongSq0
    STA NoteCounterSongSq0

@ApplySq0Effects:
    LDX CurTune0
    BNE @HandleTrg
    INC SongVibrationCounterSq0
    LDY CustomEnvelopeOffsetSongSq0
    BEQ :+
    DEC CustomEnvelopeOffsetSongSq0
:
    JSR ShapeSongVolume
    STA Sq0Duty
    LDA CurSong
    BPL :+

    ; The demo/title song ($80) vibrates the pitch.
    LDA SongVibrationCounterSq0
    LDX CurNoteLowPeriodSq0
    JSR VibratePitch
    STX Sq0Timer
:
    LDA #$7F
    STA Sq0Sweep

@HandleTrg:
    LDA NoteOffsetSongTrg
    BNE :+
    ; No phrase header has triangle offset 0, so this jump is never taken.
    JMP @HandleNoise
:
    DEC NoteCounterSongTrg
    BNE @ApplyTrgEffects

@PrepNoteOrPassage:
    LDY NoteOffsetSongTrg
    INC NoteOffsetSongTrg
    LDA (SongScriptPtrLo), Y
    BEQ @SetTrgLinear
    BPL @PlayNoteTrg
    CMP #$F0
    BEQ @EndOfPassage
    BCC @PrepNoteTrg

    ; Control note >= $F1. The low nibble defines the number of
    ; repetititions of the passage starting at the next offset.
    SEC
    SBC #$F0
    STA SongRepetitionsTrg
    LDA NoteOffsetSongTrg
    STA SongRepeatStartOffset
    JMP @PrepNoteOrPassage

@EndOfPassage:
    DEC SongRepetitionsTrg
    BEQ :+
    LDA SongRepeatStartOffset
    STA NoteOffsetSongTrg
:
    JMP @PrepNoteOrPassage

@PrepNoteTrg:
    JSR GetSongNoteLength
    STA NoteLengthSongTrg
    LDA #$1F
    STA TrgLinear
    LDY NoteOffsetSongTrg
    INC NoteOffsetSongTrg
    LDA (SongScriptPtrLo), Y
    BEQ @SetTrgLinear

@PlayNoteTrg:
    JSR EmitTriangleNote
    LDA #$00
    STA SongVibrationCounterTrg
    LDX NoteLengthSongTrg
    STX NoteCounterSongTrg

@ApplyTrgEffects:
    INC SongVibrationCounterTrg
    LDA SongVibrationCounterTrg
    LDX CurNoteLowPeriodTrg
    JSR VibratePitch
    STX TrgTimer

    ; Header byte 7 [05F1]: bit 7 set = linear counter $1F, clear = $FF.
    LDA $05F1
    BPL :+
    LDA #$1F
    BNE @SetTrgLinear
:
    LDA #$FF

@SetTrgLinear:
    STA TrgLinear

@HandleNoise:
    ; Only the overworld ($01), ending ($10) and demo ($80) songs play the noise track.
    LDA CurSong
    AND #$91
    BEQ @Exit
    DEC NoteCounterSongNoise
    BNE @Exit
:
    LDY NoteOffsetSongNoise
    INC NoteOffsetSongNoise
    LDA (SongScriptPtrLo), Y
    BNE :+

    ; We've reached the end of the track. Noise always repeats.
    LDA FirstNoteIndexSongNoise
    STA NoteOffsetSongNoise
    BNE :-
:
    JSR GetSongNoiseNoteLength
    STA NoteCounterSongNoise

    ; From the original control note, extract an index 0-3
    ; to look up noise parameters.
    TXA
    AND #$3E
    LSR
    LSR
    LSR
    LSR
    TAY
    LDA NoiseVolumes, Y
    STA NoiseVolume
    LDA NoisePeriods, Y
    STA NoisePeriod
    LDA NoiseLengths, Y
    STA NoiseLength

@Exit:
    RTS
