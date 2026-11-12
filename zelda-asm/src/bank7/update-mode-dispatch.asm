; ============================================================================
; FILE:     bank7/update-mode-dispatch.asm BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 1649-1809
; SUBJECTS: (none)
; TOPIC:    UpdateMode jump table, modes 2/3/4/6, GoToNextMode
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0028 UpdateMode                   code
;   L0034 UpdateMode_JumpTable         data
;   L0056 UpdateMode7Scroll            code
;   L0069 UpdateMode8ContinueQuestion  code
;   L0074 UpdateMode10Stairs           code
;   L0079 UpdateMode11Death            code
;   L0084 UpdateMode12EndLevel         code
;   L0089 UpdateMode2Load              code
;   L0098 GoToNextMode                 code  Returns:
;   L0107 EndGameMode                  code  Sets IsUpdatingMode to 0.
;   L0113 UpdateMode3Unfurl            code
;   L0125 UpdateMode4and6EnterLeave    code
;   L0145 LevelSongIds                 data
;   L0149 GoToNextModePlayLevelSong    code
;   L0154 GoToNextModeResetGridOffset  code
;   L0158 L1EBF8_Exit                  code
;   L0161 StepOutside                  code
; ============================================================================

UpdateMode:
    LDA #$02
    JSR SwitchBank
    LDA GameMode
    JSR TableJump

UpdateMode_JumpTable:
    .ADDR UpdateMode0Demo
    .ADDR UpdateMode1Menu
    .ADDR UpdateMode2Load
    .ADDR UpdateMode3Unfurl
    .ADDR UpdateMode4and6EnterLeave
    .ADDR UpdateMode5Play
    .ADDR UpdateMode4and6EnterLeave
    .ADDR UpdateMode7Scroll
    .ADDR UpdateMode8ContinueQuestion
    .ADDR UpdateMode5Play
    .ADDR UpdateMode5Play
    .ADDR UpdateMode5Play
    .ADDR UpdateMode5Play
    .ADDR UpdateModeDSave
    .ADDR UpdateModeERegister
    .ADDR UpdateModeFElimination
    .ADDR UpdateMode10Stairs
    .ADDR UpdateMode11Death
    .ADDR UpdateMode12EndLevel
    .ADDR UpdateMode13WinGame

UpdateMode7Scroll:
    LDA #$05
    JSR SwitchBank
    JSR UpdateMode7SubmodeAndDrawLink
    LDA IsSprite0CheckActive
    BNE @Exit                   ; If still checking sprite 0, return.
    STA IsScrollMirroringSet             ; TODO: Reset [$F3].
    LDA #$0F                    ; Set horizontal mirroring and our normal PRG ROM bank mode.
    JSR SetMMC1Control

@Exit:
    RTS

UpdateMode8ContinueQuestion:
    LDA #$05
    JSR SwitchBank
    JMP UpdateMode8ContinueQuestion_Full

UpdateMode10Stairs:
    LDA #$05
    JSR SwitchBank
    JMP UpdateMode10Stairs_Full

UpdateMode11Death:
    LDA #$05
    JSR SwitchBank
    JMP UpdateMode11Death_Full

UpdateMode12EndLevel:
    LDA #$05
    JSR SwitchBank
    JMP UpdateMode12EndLevel_Full

UpdateMode2Load:
    JSR TurnOffAllVideo
    LDA #$06
    JSR SwitchBank
    JSR UpdateMode2Load_Full

; Returns:
; A: zero
;
GoToNextMode:
    INC GameMode

; Sets IsUpdatingMode to 0.
; Sets submode to 0.
;
; Returns:
; A: 0
;
EndGameMode:
    LDA #$00
    STA IsUpdatingMode
    STA GameSubmode
    RTS

UpdateMode3Unfurl:
    JSR UpdateWorldCurtainEffect
    LDA ObjX+12
    BNE L1EBF8_Exit             ; If the left column hasn't reached the left edge, then return.
    LDA #$0F                    ; Set horizontal mirroring.
    JSR SetMMC1Control
    LDA UndergroundExitType
    BEQ :+                      ; If underground exit type <> 0, then in OW and ...
    JMP GoToNextModeResetGridOffset    ; go to next mode, and reset Link's relative position.
:
    JMP GoToNextModePlayLevelSong    ; Else go play level song, next mode, and reset Link's relative position.

UpdateMode4and6EnterLeave:
    LDA UndergroundExitType
    BNE StepOutside             ; If UndergroundExitType is set, go step out of underground (or cellar).

    ; We're walking from room to room:
    ; * entering in mode 4 (method 2)
    ; * leaving in mode 6
    LDA ObjGridOffset           ; If relative position reaches 0, 8, or -8; then go to the next mode.
    BEQ GoToNextModeResetGridOffset
    CMP #$08
    BEQ GoToNextModeResetGridOffset
    CMP #$F8
    BEQ GoToNextModeResetGridOffset
    LDA ObjDir
    STA ObjInputDir
    STA $0F                     ; Pass direction in [0F] to MoveObject.
    LDX #$00                    ; Link object index.
    JSR MoveObject
    JMP Link_EndMoveAndAnimateInRoom

LevelSongIds:
    .BYTE $01, $40, $40, $40, $40, $40, $40, $40
    .BYTE $40, $20

GoToNextModePlayLevelSong:
    LDY CurLevel
    LDA LevelSongIds, Y
    STA SongRequest             ; Play the song for the level.

GoToNextModeResetGridOffset:
    JSR GoToNextMode
    STA ObjGridOffset

L1EBF8_Exit:
    RTS

StepOutside:
    ; We're stepping out of underground or cellar.
    ; If in UW, then go finish the mode.
    LDA CurLevel
    BNE GoToNextModePlayLevelSong

    ; If the player stepped on stairs instead of an opening, then
    ; go finish the mode.
    LDA UndergroundEntranceTile
    CMP #$24
    BNE GoToNextModePlayLevelSong

    ; We're entering the OW room from a cave or dungeon.
    LDA #$05
    JSR SwitchBank
    JSR AnimateAndDrawLinkBehindBackground

    ; Every 4 frames, move Link up 1 pixel.
    LDA FrameCounter
    AND #$03
    BNE @Exit
    DEC ObjY
    LDA ObjY
    CMP StairsTargetY
    BEQ GoToNextModePlayLevelSong    ; If Link has reached the target position, go finish the mode.

@Exit:
    RTS
