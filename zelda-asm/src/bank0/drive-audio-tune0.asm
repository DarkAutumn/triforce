; ============================================================================
; FILE:     bank0/drive-audio-tune0.asm  BANK: 0 (switchable, $8000-$BFFF)
; UPSTREAM: Z_00.asm 106-221
; SUBJECTS: (none)
; TOPIC:    DriveAudio entry point; tune channel 0 scripts and driver
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0015 DriveAudio         code
;   L0053 TuneScripts0       data
;   L0069 L18C9_SilenceSong  code
;   L0072 DriveTune0         code
; ============================================================================

DriveAudio:
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
    JSR DriveTune1
    JSR DriveEffect
    JSR DriveSample
    JSR DriveSong

@Paused:
    JSR DriveTune0

    ; Reset all requests for sound.
    LDA #$00
    STA Tune0Request
    STA EffectRequest
    STA Tune1Request
    STA SampleRequest
    STA SongRequest
    RTS

TuneScripts0:
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

L18C9_SilenceSong:
    JMP SilenceSong

DriveTune0:
    LDA Tune0Request

    ; Tune $80 is only a signal to silence the song.
    BMI L18C9_SilenceSong
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
    ; Get the index for the song bit: 1 to 8.
    INY
    LSR
    BCC :-
    LDA TuneScripts0-1, Y
    STA Tune0ScriptOffset

@KeepPlaying:
    LDY Tune0ScriptOffset
    INC Tune0ScriptOffset
    LDA TuneScripts0, Y
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
    LDA TuneScripts0, Y

@PlayNote:
    JSR EmitSquareNote0
    LDA #$7F
    STA Sq0Sweep
    RTS
