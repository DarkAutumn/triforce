; ============================================================================
; FILE:     bank0/drive-audio-tune0.asm  BANK: 0 (switchable, $8000-$BFFF)
; UPSTREAM: Z_00.asm 106-221
; SUBJECTS: Audio
; TOPIC:    The sound driver's per-frame entry point (Audio_Drive) and tune channel 0: short jingles on
;           square channel 0, one script byte per frame (among them the low-hearts warning beep).
; SEE ALSO: bank7/run-game-nmi.asm (the NMI handler calls Audio_Drive once per frame);
;           bank0/sound-effects.asm (Sfx_Drive), bank0/tune1-and-samples.asm (Audio_DriveTune1, Sample_Drive),
;           bank0/drive-song.asm (Song_Drive, Song_Silence); bank0/note-emit-vibrato.asm (Audio_EmitSquareNote0).
; KEY FACTS:
;   - Audio_Drive runs the five drivers in the order tune 1, noise effect, sample, song, tune 0, then
;     zeroes all five request bytes: a request lasts one frame. Audio_Drive.
;   - While PauseState is nonzero all channels are cut every frame (ApuStatus 0 then $0F) and only tune
;     channel 0 is driven; song, tune 1, effect and sample positions freeze, and requests for them made
;     in such a frame are dropped (the request bytes are still zeroed). Audio_Drive.
;   - Tune0Request is a bit mask; the lowest set bit picks the tune. $80 is not a tune: it silences the
;     song. $40 (heart warning) starts only when no tune 0 is playing; any other request preempts.
;     Audio_DriveTune0.
;   - A tune 0 script is read one byte per frame: $00 ends the tune, a byte with bit 7 set is written to
;     Sq0Duty and the next byte is its note, any other byte is a note id for Audio_EmitSquareNote0. Audio_DriveTune0.
;   - While CurTune0 is nonzero the song's square 0 track is muted (see Song_Drive).
; CONTENTS: (generated; L = line in this file)
;   L0033 Audio_Drive                   code  Runs the sound driver for one frame: all channel drivers, then clears the sound requests.
;   L0073 Audio_Tune0Scripts            data  TABLE: tune channel 0 scripts. Bytes 0-6 = script offset (from Audio_Tune0Scripts) of tune bit 0-6; the
;   L0090 Audio_DriveTune0_SilenceSong  code  Jump to Song_Silence for tune 0 request $80 (within branch range of Audio_DriveTune0).
;   L0097 Audio_DriveTune0              code  Starts a requested tune on square channel 0 or plays the next script byte of the current one.
; ============================================================================

; ----------------------------------------------------------------------------
; Runs the sound driver for one frame: all channel drivers, then clears the sound requests.
; NOTE: Called once per frame from the NMI handler. While paused only tune channel 0 plays.
; ----------------------------------------------------------------------------
Audio_Drive:
    ; If the game is paused, then silence all channels
    ; by first disabling them, then enabling them.
    ;
    ; Then go drive tune channel 0 only.
    LDA PauseState
    BEQ @Play
    LDA #$00
    STA ApuStatus
    LDA #$0F
    STA ApuStatus
    BNE @Paused

@Play:
    ; Else the game is not paused.
    ;
    ; Synchronize the APU frame counter once a video frame.
    LDA #$FF
    STA Ctrl2FrameCtr

    ; Drive each game sound channel.
    JSR Audio_DriveTune1
    JSR Sfx_Drive
    JSR Sample_Drive
    JSR Song_Drive

@Paused:
    JSR Audio_DriveTune0

    ; Reset all requests for sound.
    LDA #$00
    STA Tune0Request
    STA EffectRequest
    STA Tune1Request
    STA SampleRequest
    STA SongRequest
    RTS

; TABLE: tune channel 0 scripts. Bytes 0-6 = script offset (from Audio_Tune0Scripts) of tune bit 0-6; the
; scripts follow. The offset for bit 6 ($40, heart warning) is 7, the byte right after the offsets.
Audio_Tune0Scripts:
    .BYTE $1C, $4C, $27, $5C, $46, $67, $07, $95
    .BYTE $50, $08, $08, $08, $08, $08, $90, $08
    .BYTE $08, $08, $08, $08, $08, $08, $08, $08
    .BYTE $08, $08, $08, $00, $82, $4A, $48, $4A
    .BYTE $08, $08, $08, $08, $08, $08, $00, $9F
    .BYTE $1E, $22, $24, $26, $9F, $28, $2A, $2C
    .BYTE $2E, $9A, $28, $2A, $2C, $2E, $9C, $28
    .BYTE $2A, $2C, $2E, $96, $28, $2A, $2C, $2E
    .BYTE $98, $28, $2A, $2C, $2E, $00, $99, $42
    .BYTE $4A, $50, $54, $00, $99, $70, $0A, $70
    .BYTE $0E, $70, $10, $9F, $70, $2A, $12, $1E
    .BYTE $2A, $70, $1E, $00, $9A, $42, $08, $08
    .BYTE $56, $08, $08, $00, $08, $08, $00, $9F
    .BYTE $40, $30, $40, $3A, $28, $00

; Jump to Song_Silence for tune 0 request $80 (within branch range of Audio_DriveTune0).
Audio_DriveTune0_SilenceSong:
    JMP Song_Silence

; ----------------------------------------------------------------------------
; Starts a requested tune on square channel 0 or plays the next script byte of the current one.
; OUT:  CurTune0 = the Tune0Request byte of the tune playing, 0 when the script ends.
; ----------------------------------------------------------------------------
Audio_DriveTune0:
    LDA Tune0Request

    ; Tune $80 is only a signal to silence the song.
    BMI Audio_DriveTune0_SilenceSong
    BEQ @CheckCurrentTune

    ; If the requested tune is not "heart warning", then go play it.
    CMP #$40
    BNE @ChangeTune

    ; Else only play "heart warning" if nothing else is playing.
    LDX CurTune0
    BEQ @ChangeTune

@CheckCurrentTune:
    LDA CurTune0
    BNE @KeepPlaying
    RTS

@ChangeTune:
    STA CurTune0
    LDY #$00
:
    ; Get the index of the lowest set tune bit: 1 to 7.
    INY
    LSR
    BCC :-
    LDA Audio_Tune0Scripts-1, Y
    STA Tune0ScriptOffset

@KeepPlaying:
    LDY Tune0ScriptOffset
    INC Tune0ScriptOffset
    LDA Audio_Tune0Scripts, Y
    BMI @PrepNote
    BNE @PlayNote

    ; We've reached the end of the tune.
    LDX #$90
    STX Sq0Duty
    LDX #$18
    STX Sq0Length
    LDX #$00
    STX Sq0Timer
    STX CurTune0
    RTS

@PrepNote:
    STA Sq0Duty
    LDY Tune0ScriptOffset
    INC Tune0ScriptOffset
    LDA Audio_Tune0Scripts, Y

@PlayNote:
    JSR Audio_EmitSquareNote0
    LDA #$7F
    STA Sq0Sweep
    RTS
