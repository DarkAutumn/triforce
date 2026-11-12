; ============================================================================
; FILE:     bank7/update-mode-dispatch.asm BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 1649-1809
; SUBJECTS: (none)
; TOPIC:    The update half of the mode machine: the per-mode update dispatch, trampolines to the far
;           updates, the fixed-bank updates of modes 2, 3, 4/6 and 7, and the routines that end a mode
;           (GoToNextMode, EndGameMode), including Link's rise out of a cave or dungeon entrance.
; SEE ALSO: bank7/run-game-nmi.asm (IsrNmi calls UpdateMode once per frame while IsUpdatingMode <> 0);
;           bank7/init-mode-dispatch.asm (InitMode and the init side); bank7/mode5-play-update.asm
;           (UpdateMode5Play, Link_EndMoveAndAnimateInRoom); bank7/walker-move.asm (MoveObject);
;           bank5/enter-room.asm (InitMode_EnterRoom sets the walk-in distance and StairsTargetY);
;           bank5/mode3-mode6-init.asm (InitMode6 sets the walk-out distance); bank5/mode7-scroll-update.asm;
;           bank1/map-marker-curtain.asm (UpdateWorldCurtainEffect); bank6/level-load-mode2.asm
;           (UpdateMode2Load_Full).
; KEY FACTS:
;   - Modes 5, 9, $A, $B and $C share the play update; modes 4 and 6 share one update.
;     UpdateMode_JumpTable
;   - Ending a mode clears IsUpdatingMode and GameSubmode, so the next frame runs the init of the mode
;     then in GameMode; GoToNextMode also adds 1 to GameMode first (2->3, 3->4, 4->5, 6->7).
;     GoToNextMode, EndGameMode
;   - Mode 2's update runs for one frame: it applies the Q2 patches, then goes to mode 3.
;     UpdateMode2Load
;   - Modes 4 and 6 walk Link in his facing direction with MoveObject until ObjGridOffset is 0, 8 or
;     $F8, then go to the next mode with ObjGridOffset = 0; no input is used.
;     UpdateMode4and6EnterLeave
;   - Coming out of an overworld opening (entrance tile $24) Link rises 1 pixel every 4 frames (when
;     FrameCounter AND 3 = 0) until ObjY = StairsTargetY ($10 pixels: 61 to 64 frames); mode 5 starts
;     only then. From stairs (any other tile) or from a cellar inside a dungeon mode 4 ends on its
;     first update frame. StepOutside
;   - The level song restarts when mode 3 ends with UndergroundExitType = 0 and whenever Link finishes
;     stepping out of underground: $01 overworld, $40 levels 1-8, $20 level 9. LevelSongIds,
;     GoToNextModePlayLevelSong
;   - The room scroll ends (mode 7) by restoring horizontal mirroring once the sprite-0 check is off.
;     UpdateMode7Scroll
; CONTENTS: (generated; L = line in this file)
;   L0061 UpdateMode                   code  Switches to bank 2 and jumps to the update routine of the current game mode.
;   L0068 UpdateMode_JumpTable         data  TABLE: mode update routines; index = GameMode ($00-$13); value = routine address.
;   L0095 UpdateMode7Scroll            code  Mode 7 update: runs the scroll step; when the sprite-0 check ends, restores horizontal mirroring.
;   L0111 UpdateMode8ContinueQuestion  code  Mode 8 update: trampoline to the continue/save/retry menu update in bank 5.
;   L0119 UpdateMode10Stairs           code  Mode $10 update: trampoline to the stairs walk update in bank 5.
;   L0127 UpdateMode11Death            code  Mode $11 update: trampoline to the death sequence update in bank 5.
;   L0135 UpdateMode12EndLevel         code  Mode $12 update: trampoline to the end-of-level sequence update in bank 5.
;   L0144 UpdateMode2Load              code  Mode 2 update: applies the Q2 patches to the loaded level data, then goes to mode 3.
;   L0154 GoToNextMode                 code  Adds 1 to GameMode and ends the current mode, so the new mode's init runs next frame.
;   L0163 EndGameMode                  code  Ends the current mode: clears IsUpdatingMode and GameSubmode.
;   L0176 UpdateMode3Unfurl            code  Mode 3 update: opens the curtain; when it is fully open, goes to mode 4.
;   L0196 UpdateMode4and6EnterLeave    code  Mode 4 and 6 update: walks Link into or out of a room without input, then goes to the next mode.
;   L0217 LevelSongIds                 data  TABLE: song request for a level; index = CurLevel (0 = overworld, 1-9); value = SongRequest bit.
;   L0224 GoToNextModePlayLevelSong    code  Requests the current level's song, then goes to the next mode with Link's grid offset reset.
;   L0230 GoToNextModeResetGridOffset  code  Second entry: goes to the next mode and sets Link's ObjGridOffset to 0, without a song request.
;   L0235 L1EBF8_Exit                  code  Shared RTS, also used by UpdateMode3Unfurl and UpdateMode5Play.
;   L0246 StepOutside                  code  Mode 4 update while leaving underground: raises Link out of an entrance, then goes to mode 5.
; ============================================================================

; ----------------------------------------------------------------------------
; Switches to bank 2 and jumps to the update routine of the current game mode.
; NOTE: Called by IsrNmi once per frame while IsUpdatingMode is nonzero. Bank 2 holds the updates of
;       modes 0, 1, $D, $E, $F and $13; the fixed-bank routines of the other modes switch banks as
;       they need.
; ----------------------------------------------------------------------------
UpdateMode:
    LDA #$02
    JSR SwitchBank
    LDA GameMode
    JSR TableJump

; TABLE: mode update routines; index = GameMode ($00-$13); value = routine address.
UpdateMode_JumpTable:
    .ADDR UpdateMode0Demo               ; $00 demo
    .ADDR UpdateMode1Menu               ; $01 file menu
    .ADDR UpdateMode2Load               ; $02 load level
    .ADDR UpdateMode3Unfurl             ; $03 unfurl
    .ADDR UpdateMode4and6EnterLeave     ; $04 enter room
    .ADDR UpdateMode5Play               ; $05 play
    .ADDR UpdateMode4and6EnterLeave     ; $06 leave room (same routine as $04)
    .ADDR UpdateMode7Scroll             ; $07 scroll
    .ADDR UpdateMode8ContinueQuestion   ; $08 continue menu
    .ADDR UpdateMode5Play               ; $09 cellar (play update)
    .ADDR UpdateMode5Play               ; $0A leave cellar (play update)
    .ADDR UpdateMode5Play               ; $0B cave (play update)
    .ADDR UpdateMode5Play               ; $0C shortcut stairs (play update)
    .ADDR UpdateModeDSave               ; $0D save
    .ADDR UpdateModeERegister           ; $0E register
    .ADDR UpdateModeFElimination        ; $0F eliminate
    .ADDR UpdateMode10Stairs            ; $10 stairs
    .ADDR UpdateMode11Death             ; $11 death
    .ADDR UpdateMode12EndLevel          ; $12 end level
    .ADDR UpdateMode13WinGame           ; $13 win game

; ----------------------------------------------------------------------------
; Mode 7 update: runs the scroll step; when the sprite-0 check ends, restores horizontal mirroring.
; NOTE: Clears IsScrollMirroringSet and writes MMC1 control $0F on every frame on which
;       IsSprite0CheckActive is 0.
; ----------------------------------------------------------------------------
UpdateMode7Scroll:
    LDA #$05
    JSR SwitchBank
    JSR UpdateMode7SubmodeAndDrawLink
    LDA IsSprite0CheckActive
    BNE @Exit                   ; If still checking sprite 0, return.
    STA IsScrollMirroringSet    ; A = 0: the scroll mirroring is no longer set.
    LDA #$0F                    ; Set horizontal mirroring and the normal PRG ROM bank mode.
    JSR SetMMC1Control

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Mode 8 update: trampoline to the continue/save/retry menu update in bank 5.
; ----------------------------------------------------------------------------
UpdateMode8ContinueQuestion:
    LDA #$05
    JSR SwitchBank
    JMP UpdateMode8ContinueQuestion_Full

; ----------------------------------------------------------------------------
; Mode $10 update: trampoline to the stairs walk update in bank 5.
; ----------------------------------------------------------------------------
UpdateMode10Stairs:
    LDA #$05
    JSR SwitchBank
    JMP UpdateMode10Stairs_Full

; ----------------------------------------------------------------------------
; Mode $11 update: trampoline to the death sequence update in bank 5.
; ----------------------------------------------------------------------------
UpdateMode11Death:
    LDA #$05
    JSR SwitchBank
    JMP UpdateMode11Death_Full

; ----------------------------------------------------------------------------
; Mode $12 update: trampoline to the end-of-level sequence update in bank 5.
; ----------------------------------------------------------------------------
UpdateMode12EndLevel:
    LDA #$05
    JSR SwitchBank
    JMP UpdateMode12EndLevel_Full

; ----------------------------------------------------------------------------
; Mode 2 update: applies the Q2 patches to the loaded level data, then goes to mode 3.
; NOTE: Runs once: it falls through into GoToNextMode. Video stays off.
; ----------------------------------------------------------------------------
UpdateMode2Load:
    JSR TurnOffAllVideo
    LDA #$06
    JSR SwitchBank
    JSR UpdateMode2Load_Full

; ----------------------------------------------------------------------------
; Adds 1 to GameMode and ends the current mode, so the new mode's init runs next frame.
; OUT:  A = 0; IsUpdatingMode = 0; GameSubmode = 0.
; ----------------------------------------------------------------------------
GoToNextMode:
    INC GameMode

; ----------------------------------------------------------------------------
; Ends the current mode: clears IsUpdatingMode and GameSubmode.
; OUT:  A = 0.
; NOTE: GameMode is not changed here. Callers that jump to another mode write GameMode themselves
;       after the call; from the next frame IsrNmi runs that mode's init routine.
; ----------------------------------------------------------------------------
EndGameMode:
    LDA #$00
    STA IsUpdatingMode
    STA GameSubmode
    RTS

; ----------------------------------------------------------------------------
; Mode 3 update: opens the curtain; when it is fully open, goes to mode 4.
; NOTE: Done when ObjX[$0C] (the curtain column that moves toward the left edge) is 0. Then sets
;       horizontal mirroring (the curtain step leaves MMC1 control at $0E). With UndergroundExitType
;       = 0 the level song is requested here; otherwise (Link is about to step out of a cave or
;       dungeon entrance) the song waits for StepOutside.
; ----------------------------------------------------------------------------
UpdateMode3Unfurl:
    JSR UpdateWorldCurtainEffect
    LDA ObjX+12
    BNE L1EBF8_Exit             ; If the left column hasn't reached the left edge, then return.
    LDA #$0F                    ; Set horizontal mirroring.
    JSR SetMMC1Control
    LDA UndergroundExitType
    BEQ :+                      ; If leaving underground (exit type <> 0),
    JMP GoToNextModeResetGridOffset    ; go to the next mode without the song; reset Link's grid offset.
:
    JMP GoToNextModePlayLevelSong    ; Else request the level song, go to the next mode, reset grid offset.

; ----------------------------------------------------------------------------
; Mode 4 and 6 update: walks Link into or out of a room without input, then goes to the next mode.
; NOTE: With UndergroundExitType <> 0 the work is done by StepOutside instead. Otherwise Link moves in
;       ObjDir with MoveObject (no tile collision) each frame until ObjGridOffset is 0, 8 or $F8; the
;       starting ObjGridOffset (set by the mode's init) is the distance to walk. Mode 4 then goes to
;       mode 5, mode 6 to mode 7, with ObjGridOffset = 0. 8 and $F8 are Link's grid offset limits, at
;       which MoveObject no longer moves him.
; ----------------------------------------------------------------------------
UpdateMode4and6EnterLeave:
    LDA UndergroundExitType
    BNE StepOutside             ; If UndergroundExitType is set, go step out of underground (or cellar).

    ; Link is walking from room to room:
    ; * entering in mode 4
    ; * leaving in mode 6
    LDA ObjGridOffset           ; If the grid offset has reached 0, 8, or -8; then go to the next mode.
    BEQ GoToNextModeResetGridOffset
    CMP #$08
    BEQ GoToNextModeResetGridOffset
    CMP #$F8
    BEQ GoToNextModeResetGridOffset
    LDA ObjDir
    STA ObjInputDir
    STA $0F                     ; Pass direction in [0F] to MoveObject.
    LDX #$00                    ; Link's object slot.
    JSR MoveObject
    JMP Link_EndMoveAndAnimateInRoom

; TABLE: song request for a level; index = CurLevel (0 = overworld, 1-9); value = SongRequest bit.
LevelSongIds:
    .BYTE $01, $40, $40, $40, $40, $40, $40, $40    ; $01 overworld song; $40 dungeon song (levels 1-7)
    .BYTE $40, $20                                  ; level 8: $40; level 9: $20 (level 9 song)

; ----------------------------------------------------------------------------
; Requests the current level's song, then goes to the next mode with Link's grid offset reset.
; ----------------------------------------------------------------------------
GoToNextModePlayLevelSong:
    LDY CurLevel
    LDA LevelSongIds, Y
    STA SongRequest             ; Play the song for the level.

; Second entry: goes to the next mode and sets Link's ObjGridOffset to 0, without a song request.
GoToNextModeResetGridOffset:
    JSR GoToNextMode            ; Returns A = 0.
    STA ObjGridOffset

; Shared RTS, also used by UpdateMode3Unfurl and UpdateMode5Play.
L1EBF8_Exit:
    RTS

; ----------------------------------------------------------------------------
; Mode 4 update while leaving underground: raises Link out of an entrance, then goes to mode 5.
; NOTE: Ends at once (with the level song) in a dungeon (leaving a cellar) and when Link went down by
;       stairs (UndergroundEntranceTile <> $24). From an opening ($24) Link is drawn behind the
;       background and ObjY drops by 1 on each frame with FrameCounter AND 3 = 0, until it equals
;       StairsTargetY; InitMode_EnterRoom starts him $10 pixels below that, so the rise takes 61 to
;       64 frames (depending on the phase of FrameCounter); this routine uses no input.
; ----------------------------------------------------------------------------
StepOutside:
    ; If in UW (stepping out of a cellar), then go finish the mode.
    LDA CurLevel
    BNE GoToNextModePlayLevelSong

    ; If Link went underground by stairs instead of an opening (tile $24), then
    ; go finish the mode.
    LDA UndergroundEntranceTile
    CMP #$24
    BNE GoToNextModePlayLevelSong

    ; Link is coming up into the OW room from a cave or dungeon opening.
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
