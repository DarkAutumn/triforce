; ============================================================================
; FILE:     bank7/update-mode-dispatch.asm BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 1649-1809
; SUBJECTS: Mode, Mode10Stairs, Mode11Death, Mode12EndLevel, Mode2Load, Mode3Unfurl, Mode4Enter,
;          Mode7Scroll, Mode8Continue, Song
; TOPIC:    The update half of the mode machine: the per-mode update dispatch, trampolines to the far
;           updates, the fixed-bank updates of modes 2, 3, 4/6 and 7, and the routines that end a mode
;           (Mode_GoToNext, Mode_End), including Link's rise out of a cave or dungeon entrance.
; SEE ALSO: bank7/run-game-nmi.asm (Nmi_Isr calls Mode_Update once per frame while IsUpdatingMode <> 0);
;           bank7/init-mode-dispatch.asm (Mode_Init and the init side); bank7/mode5-play-update.asm
;           (Mode5Play_Update, Link_EndMoveAndAnimateInRoom); bank7/walker-move.asm (Obj_MoveByQSpeed);
;           bank5/enter-room.asm (Room_EnterCommon sets the walk-in distance and StairsTargetY);
;           bank5/mode3-mode6-init.asm (Mode6Leave_Init sets the walk-out distance); bank5/mode7-scroll-update.asm;
;           bank1/map-marker-curtain.asm (Mode_UpdateCurtainEffect); bank6/level-load-mode2.asm
;           (Mode2Load_Update_Full).
; KEY FACTS:
;   - Modes 5, 9, $A, $B and $C share the play update; modes 4 and 6 share one update.
;     Mode_Update_JumpTable
;   - Ending a mode clears IsUpdatingMode and GameSubmode, so the next frame runs the init of the mode
;     then in GameMode; Mode_GoToNext also adds 1 to GameMode first (2->3, 3->4, 4->5, 6->7).
;     Mode_GoToNext, Mode_End
;   - Mode 2's update runs for one frame: it applies the Q2 patches, then goes to mode 3.
;     Mode2Load_Update
;   - Modes 4 and 6 walk Link in his facing direction with Obj_MoveByQSpeed until ObjGridOffset is 0, 8 or
;     $F8, then go to the next mode with ObjGridOffset = 0; no input is used.
;     Mode4Enter_UpdateEnterOrLeave
;   - Coming out of an overworld opening (entrance tile $24) Link rises 1 pixel every 4 frames (when
;     FrameCounter AND 3 = 0) until ObjY = StairsTargetY ($10 pixels: 61 to 64 frames); mode 5 starts
;     only then. From stairs (any other tile) or from a cellar inside a dungeon mode 4 ends on its
;     first update frame. Mode4Enter_StepOutside
;   - The level song restarts when mode 3 ends with UndergroundExitType = 0 and whenever Link finishes
;     stepping out of underground: $01 overworld, $40 levels 1-8, $20 level 9. Song_LevelRequests,
;     Mode_GoToNext_PlayLevelSong
;   - The room scroll ends (mode 7) by restoring horizontal mirroring once the sprite-0 check is off.
;     Mode7Scroll_Update
; CONTENTS: (generated; L = line in this file)
;   L0062 Mode_Update                    code  Switches to bank 2 and jumps to the update routine of the current game mode.
;   L0069 Mode_Update_JumpTable          data  TABLE: mode update routines; index = GameMode ($00-$13); value = routine address.
;   L0096 Mode7Scroll_Update             code  Mode 7 update: runs the scroll step; when the sprite-0 check ends, restores horizontal mirroring.
;   L0112 Mode8Continue_Update           code  Mode 8 update: trampoline to the continue/save/retry menu update in bank 5.
;   L0120 Mode10Stairs_Update            code  Mode $10 update: trampoline to the stairs walk update in bank 5.
;   L0128 Mode11Death_Update             code  Mode $11 update: trampoline to the death sequence update in bank 5.
;   L0136 Mode12EndLevel_Update          code  Mode $12 update: trampoline to the end-of-level sequence update in bank 5.
;   L0145 Mode2Load_Update               code  Mode 2 update: applies the Q2 patches to the loaded level data, then goes to mode 3.
;   L0155 Mode_GoToNext                  code  Adds 1 to GameMode and ends the current mode, so the new mode's init runs next frame.
;   L0164 Mode_End                       code  Ends the current mode: clears IsUpdatingMode and GameSubmode.
;   L0177 Mode3Unfurl_Update             code  Mode 3 update: opens the curtain; when it is fully open, goes to mode 4.
;   L0197 Mode4Enter_UpdateEnterOrLeave  code  Mode 4 and 6 update: walks Link into or out of a room without input, then goes to the next mode.
;   L0218 Song_LevelRequests             data  TABLE: song request for a level; index = CurLevel (0 = overworld, 1-9); value = SongRequest bit.
;   L0225 Mode_GoToNext_PlayLevelSong    code  Requests the current level's song, then goes to the next mode with Link's grid offset reset.
;   L0231 Mode_GoToNext_ResetGridOffset  code  Second entry: goes to the next mode and sets Link's ObjGridOffset to 0, without a song request.
;   L0236 Mode_GoToNext_Return           code  Shared RTS, also used by Mode3Unfurl_Update and Mode5Play_Update.
;   L0247 Mode4Enter_StepOutside         code  Mode 4 update while leaving underground: raises Link out of an entrance, then goes to mode 5.
; ============================================================================

; ----------------------------------------------------------------------------
; Switches to bank 2 and jumps to the update routine of the current game mode.
; NOTE: Called by Nmi_Isr once per frame while IsUpdatingMode is nonzero. Bank 2 holds the updates of
;       modes 0, 1, $D, $E, $F and $13; the fixed-bank routines of the other modes switch banks as
;       they need.
; ----------------------------------------------------------------------------
Mode_Update:
    LDA #$02
    JSR Mmc1_SwitchBank
    LDA GameMode
    JSR Util_TableJump

; TABLE: mode update routines; index = GameMode ($00-$13); value = routine address.
Mode_Update_JumpTable:
    .ADDR Mode0Demo_Update               ; $00 demo
    .ADDR Mode1Menu_Update               ; $01 file menu
    .ADDR Mode2Load_Update               ; $02 load level
    .ADDR Mode3Unfurl_Update             ; $03 unfurl
    .ADDR Mode4Enter_UpdateEnterOrLeave     ; $04 enter room
    .ADDR Mode5Play_Update               ; $05 play
    .ADDR Mode4Enter_UpdateEnterOrLeave     ; $06 leave room (same routine as $04)
    .ADDR Mode7Scroll_Update             ; $07 scroll
    .ADDR Mode8Continue_Update   ; $08 continue menu
    .ADDR Mode5Play_Update               ; $09 cellar (play update)
    .ADDR Mode5Play_Update               ; $0A leave cellar (play update)
    .ADDR Mode5Play_Update               ; $0B cave (play update)
    .ADDR Mode5Play_Update               ; $0C shortcut stairs (play update)
    .ADDR ModeDSave_Update               ; $0D save
    .ADDR ModeERegister_Update           ; $0E register
    .ADDR ModeFEliminate_Update        ; $0F eliminate
    .ADDR Mode10Stairs_Update            ; $10 stairs
    .ADDR Mode11Death_Update             ; $11 death
    .ADDR Mode12EndLevel_Update          ; $12 end level
    .ADDR Mode13WinGame_Update           ; $13 win game

; ----------------------------------------------------------------------------
; Mode 7 update: runs the scroll step; when the sprite-0 check ends, restores horizontal mirroring.
; NOTE: Clears IsScrollMirroringSet and writes MMC1 control $0F on every frame on which
;       IsSprite0CheckActive is 0.
; ----------------------------------------------------------------------------
Mode7Scroll_Update:
    LDA #$05
    JSR Mmc1_SwitchBank
    JSR Mode7Scroll_UpdateSubmodeAndDrawLink
    LDA IsSprite0CheckActive
    BNE @Exit                   ; If still checking sprite 0, return.
    STA IsScrollMirroringSet    ; A = 0: the scroll mirroring is no longer set.
    LDA #$0F                    ; Set horizontal mirroring and the normal PRG ROM bank mode.
    JSR Mmc1_SetControl

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Mode 8 update: trampoline to the continue/save/retry menu update in bank 5.
; ----------------------------------------------------------------------------
Mode8Continue_Update:
    LDA #$05
    JSR Mmc1_SwitchBank
    JMP Mode8Continue_Update_Full

; ----------------------------------------------------------------------------
; Mode $10 update: trampoline to the stairs walk update in bank 5.
; ----------------------------------------------------------------------------
Mode10Stairs_Update:
    LDA #$05
    JSR Mmc1_SwitchBank
    JMP Mode10Stairs_Update_Full

; ----------------------------------------------------------------------------
; Mode $11 update: trampoline to the death sequence update in bank 5.
; ----------------------------------------------------------------------------
Mode11Death_Update:
    LDA #$05
    JSR Mmc1_SwitchBank
    JMP Mode11Death_Update_Full

; ----------------------------------------------------------------------------
; Mode $12 update: trampoline to the end-of-level sequence update in bank 5.
; ----------------------------------------------------------------------------
Mode12EndLevel_Update:
    LDA #$05
    JSR Mmc1_SwitchBank
    JMP Mode12EndLevel_Update_Full

; ----------------------------------------------------------------------------
; Mode 2 update: applies the Q2 patches to the loaded level data, then goes to mode 3.
; NOTE: Runs once: it falls through into Mode_GoToNext. Video stays off.
; ----------------------------------------------------------------------------
Mode2Load_Update:
    JSR Ppu_TurnOffAllVideo
    LDA #$06
    JSR Mmc1_SwitchBank
    JSR Mode2Load_Update_Full

; ----------------------------------------------------------------------------
; Adds 1 to GameMode and ends the current mode, so the new mode's init runs next frame.
; OUT:  A = 0; IsUpdatingMode = 0; GameSubmode = 0.
; ----------------------------------------------------------------------------
Mode_GoToNext:
    INC GameMode

; ----------------------------------------------------------------------------
; Ends the current mode: clears IsUpdatingMode and GameSubmode.
; OUT:  A = 0.
; NOTE: GameMode is not changed here. Callers that jump to another mode write GameMode themselves
;       after the call; from the next frame Nmi_Isr runs that mode's init routine.
; ----------------------------------------------------------------------------
Mode_End:
    LDA #$00
    STA IsUpdatingMode
    STA GameSubmode
    RTS

; ----------------------------------------------------------------------------
; Mode 3 update: opens the curtain; when it is fully open, goes to mode 4.
; NOTE: Done when ObjX[$0C] (the curtain column that moves toward the left edge) is 0. Then sets
;       horizontal mirroring (the curtain step leaves MMC1 control at $0E). With UndergroundExitType
;       = 0 the level song is requested here; otherwise (Link is about to step out of a cave or
;       dungeon entrance) the song waits for Mode4Enter_StepOutside.
; ----------------------------------------------------------------------------
Mode3Unfurl_Update:
    JSR Mode_UpdateCurtainEffect
    LDA ObjX+12
    BNE Mode_GoToNext_Return             ; If the left column hasn't reached the left edge, then return.
    LDA #$0F                    ; Set horizontal mirroring.
    JSR Mmc1_SetControl
    LDA UndergroundExitType
    BEQ :+                      ; If leaving underground (exit type <> 0),
    JMP Mode_GoToNext_ResetGridOffset    ; go to the next mode without the song; reset Link's grid offset.
:
    JMP Mode_GoToNext_PlayLevelSong    ; Else request the level song, go to the next mode, reset grid offset.

; ----------------------------------------------------------------------------
; Mode 4 and 6 update: walks Link into or out of a room without input, then goes to the next mode.
; NOTE: With UndergroundExitType <> 0 the work is done by Mode4Enter_StepOutside instead. Otherwise Link moves in
;       ObjDir with Obj_MoveByQSpeed (no tile collision) each frame until ObjGridOffset is 0, 8 or $F8; the
;       starting ObjGridOffset (set by the mode's init) is the distance to walk. Mode 4 then goes to
;       mode 5, mode 6 to mode 7, with ObjGridOffset = 0. 8 and $F8 are Link's grid offset limits, at
;       which Obj_MoveByQSpeed no longer moves him.
; ----------------------------------------------------------------------------
Mode4Enter_UpdateEnterOrLeave:
    LDA UndergroundExitType
    BNE Mode4Enter_StepOutside             ; If UndergroundExitType is set, go step out of underground (or cellar).

    ; Link is walking from room to room:
    ; * entering in mode 4
    ; * leaving in mode 6
    LDA ObjGridOffset           ; If the grid offset has reached 0, 8, or -8; then go to the next mode.
    BEQ Mode_GoToNext_ResetGridOffset
    CMP #$08
    BEQ Mode_GoToNext_ResetGridOffset
    CMP #$F8
    BEQ Mode_GoToNext_ResetGridOffset
    LDA ObjDir
    STA ObjInputDir
    STA $0F                     ; Pass direction in [0F] to Obj_MoveByQSpeed.
    LDX #$00                    ; Link's object slot.
    JSR Obj_MoveByQSpeed
    JMP Link_EndMoveAndAnimateInRoom

; TABLE: song request for a level; index = CurLevel (0 = overworld, 1-9); value = SongRequest bit.
Song_LevelRequests:
    .BYTE $01, $40, $40, $40, $40, $40, $40, $40    ; $01 overworld song; $40 dungeon song (levels 1-7)
    .BYTE $40, $20                                  ; level 8: $40; level 9: $20 (level 9 song)

; ----------------------------------------------------------------------------
; Requests the current level's song, then goes to the next mode with Link's grid offset reset.
; ----------------------------------------------------------------------------
Mode_GoToNext_PlayLevelSong:
    LDY CurLevel
    LDA Song_LevelRequests, Y
    STA SongRequest             ; Play the song for the level.

; Second entry: goes to the next mode and sets Link's ObjGridOffset to 0, without a song request.
Mode_GoToNext_ResetGridOffset:
    JSR Mode_GoToNext            ; Returns A = 0.
    STA ObjGridOffset

; Shared RTS, also used by Mode3Unfurl_Update and Mode5Play_Update.
Mode_GoToNext_Return:
    RTS

; ----------------------------------------------------------------------------
; Mode 4 update while leaving underground: raises Link out of an entrance, then goes to mode 5.
; NOTE: Ends at once (with the level song) in a dungeon (leaving a cellar) and when Link went down by
;       stairs (UndergroundEntranceTile <> $24). From an opening ($24) Link is drawn behind the
;       background and ObjY drops by 1 on each frame with FrameCounter AND 3 = 0, until it equals
;       StairsTargetY; Room_EnterCommon starts him $10 pixels below that, so the rise takes 61 to
;       64 frames (depending on the phase of FrameCounter); this routine uses no input.
; ----------------------------------------------------------------------------
Mode4Enter_StepOutside:
    ; If in UW (stepping out of a cellar), then go finish the mode.
    LDA CurLevel
    BNE Mode_GoToNext_PlayLevelSong

    ; If Link went underground by stairs instead of an opening (tile $24), then
    ; go finish the mode.
    LDA UndergroundEntranceTile
    CMP #$24
    BNE Mode_GoToNext_PlayLevelSong

    ; Link is coming up into the OW room from a cave or dungeon opening.
    LDA #$05
    JSR Mmc1_SwitchBank
    JSR Link_AnimateAndDrawBehindBackground

    ; Every 4 frames, move Link up 1 pixel.
    LDA FrameCounter
    AND #$03
    BNE @Exit
    DEC ObjY
    LDA ObjY
    CMP StairsTargetY
    BEQ Mode_GoToNext_PlayLevelSong    ; If Link has reached the target position, go finish the mode.

@Exit:
    RTS
