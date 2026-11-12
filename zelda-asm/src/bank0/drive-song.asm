; ============================================================================
; FILE:     bank0/drive-song.asm         BANK: 0 (switchable, $8000-$BFFF)
; UPSTREAM: Z_00.asm 694-1053
; SUBJECTS: (none)
; TOPIC:    DriveSong: phrase sequencing, per-channel notes
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0022 KeepPlaying         code
;   L0025 DriveSong           code
;   L0060 SetPrevPhraseIndex  code
;   L0063 PlayNextPhrase      code
;   L0114 PrepPhrase          code
;   L0142 KeepPlayingSong     code
;   L0158 SilenceSong         code
;   L0166 PlayAgain           code
;   L0169 PrepNote            code
;   L0176 PlayNote            code
;   L0194 ApplySq1Effects     code
; ============================================================================

KeepPlaying:
    JMP KeepPlayingSong

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

SetPrevPhraseIndex:
    STY SongPhraseIndex

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
    STA $05F1                   ; TODO: [05F1]
    LDA #$01
    STA NoteCounterSongSq1
    STA NoteCounterSongSq0
    STA NoteCounterSongTrg
    STA NoteCounterSongNoise
    LSR
    STA NoteOffsetSongSq1

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

SilenceSong:
    LDA #$00
    STA CurSong
    STA ApuStatus
    LDA #$0F
    STA ApuStatus
    RTS

PlayAgain:
    JMP PlayNextPhrase

PrepNote:
    JSR GetSongNoteLength
    STA NoteLengthSongSq1
    LDY NoteOffsetSongSq1
    INC NoteOffsetSongSq1
    LDA (SongScriptPtrLo), Y

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
    ; Begin unverified code 1E1E
    JMP @HandleNoise
    ; End unverified code
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

    ; TODO: [05F1] ?
    LDA $05F1
    BPL :+
    LDA #$1F
    BNE @SetTrgLinear
:
    LDA #$FF

@SetTrgLinear:
    STA TrgLinear

@HandleNoise:
    ; If the song is not demo nor ending, then return.
    ; They don't use noise.
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
