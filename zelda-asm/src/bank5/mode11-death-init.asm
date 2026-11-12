; ============================================================================
; FILE:     bank5/mode11-death-init.asm  BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 2039-2097
; SUBJECTS: Audio, Mode11Death
; TOPIC:    Init half of the death mode ($11): Link flashes in the frozen room for 33 frames, then the
;           death sequence is set up (red fade series, four spins) and all sound is silenced.
; SEE ALSO: bank1/link-collision-harm.asm (where Link's hearts reach 0 and mode $11 is entered);
;           bank5/mode11-death.asm (the update half: spin, fade, spark, GAME OVER);
;           bank7/init-mode-dispatch.asm (Mode_Init_JumpTable calls Mode11Death_Init every frame until
;           IsUpdatingMode is set); bank5/submenu-states.asm (jumps to Audio_SilenceSongTune1AndSample).
; KEY FACTS:
;   - The init runs once per frame in two steps selected by GameSubmode: step 0 lasts one frame,
;     step 1 waits for Link's ObjTimer. Mode11Death_Init, Mode11Death_Init_Sub1.
;   - Step 0 empties HeartPartial, clears PauseState, redraws the status bar, and starts the flash:
;     Link's ObjInvincibilityTimer = $10 (counts every second frame = 32 frames) and ObjTimer = $21
;     (33 frames). Mode11Death_Init.
;   - Monsters and other objects are not updated during the death mode; only Link is redrawn.
;     Mode11Death_Init.
;   - Step 1, once the 33 frames are over: FadeCycle = $60 (the death palette series),
;     DeathModeCounter = 4 spins, Link faces down (ObjDir = 4), ObjState = 0, GameSubmode = 0, then
;     the update half takes over. Mode11Death_Init_Sub1.
;   - Outside a cellar (unique room ids $3E, $3F) the room's doors are laid out again from
;     CurOpenedDoors before the sequence. Mode11Death_Init_Sub1.
;   - Audio_SilenceSongTune1AndSample requests $80 on tune channel 0 (stops the song) and $80 as the effect request
;     (stops the DMC sample and tune channel 1). Audio_SilenceSongTune1AndSample.
; CONTENTS: (generated; L = line in this file)
;   L0041 Mode11Death_Init                 code  Death mode init, run each frame: step 0 freezes the room and starts Link's 33-frame flash.
;   L0073 Mode11Death_Init_Sub1            code  Death mode init step 1: waits out the flash, then sets up the death sequence and starts updating.
;   L0101 Audio_SilenceSongTune1AndSample  code  Requests silence: $80 to tune channel 0 (stops the song) and $80 as the effect request.
;   L0107 Mode11Death_Init_Return          code  Shared return of the routines above.
; ============================================================================

; ----------------------------------------------------------------------------
; Death mode init, run each frame: step 0 freezes the room and starts Link's 33-frame flash.
; IN:   GameSubmode = 0 (first frame) or 1 (waiting).
; OUT:  Step 0: PauseState = 0, HeartPartial = 0, ObjInvincibilityTimer = $10, ObjTimer = $21,
;       GameSubmode = 1.
; NOTE: Every frame of both steps Link's invincibility timer is counted down and Link is redrawn, so
;       he flashes while the timer runs.
; ----------------------------------------------------------------------------
Mode11Death_Init:
    LDX #$00                    ; Link's slot: count down his invincibility timer (the flash in step 1).
    JSR Obj_DecInvincibilityTimer
    JSR Link_EndMoveAndAnimate
    LDA GameSubmode
    BNE Mode11Death_Init_Sub1

    ; Step 0 (one frame).
    JSR Sprite_HideAll
    JSR StatusBar_UpdateTriforceMarker
    JSR Link_DrawBetweenRooms
    JSR StatusBar_DrawItemsAndEnsureSelection
    JSR Ppu_ClearGrayscaleInMask ; Clears bit 0 (grayscale) of CurPpuMask.
    LDA #$00
    STA PauseState
    STA HeartPartial
    JSR StatusBar_FormatText
    INC GameSubmode
    LDA #$10                    ; Link flashes while this is nonzero: $10 * 2 = 32 frames.
    STA ObjInvincibilityTimer

    ; The invincibility timer counts down every second frame and the object timer every frame,
    ; so $21 frames of ObjTimer cover the $10 steps of the flash.
    LDA #$21
    STA ObjTimer
    RTS

; ----------------------------------------------------------------------------
; Death mode init step 1: waits out the flash, then sets up the death sequence and starts updating.
; OUT:  FadeCycle = $60; DeathModeCounter = 4; ObjDir = 4 (down); ObjState = 0; GameSubmode = 0;
;       CurRow = 0; IsUpdatingMode set; falls into Audio_SilenceSongTune1AndSample.
; ----------------------------------------------------------------------------
Mode11Death_Init_Sub1:
    LDA ObjTimer
    BNE Mode11Death_Init_Return             ; Still flashing: wait.
    JSR Room_GetUniqueRoomId
    AND #$3E
    CMP #$3E
    BEQ :+                      ; Unique room ids $3E and $3F are cellars: they have no doors.
    LDA CurOpenedDoors          ; Elsewhere, lay out the doors again with the current open-door mask
    STA PrevOpenedDoors         ; (Door_LayOutWithPrevOpened lays them out from PrevOpenedDoors; it does
    JSR Door_LayOutWithPrevOpened         ; nothing in the OW).
:
    LDA #$60                    ; Fade series 6, step 0: the death palette series.
    STA FadeCycle
    LDA #$02                    ; ObjTimer[$0A] = 2. No death-mode routine in this bank reads it.
    STA ObjTimer+10
    LDA #$00                    ; Reset the submode, the row to copy, and Link's state.
    STA GameSubmode
    STA CurRow
    STA ObjState
    LDA #$04                    ; 4 spins, starting with Link facing down (direction bit 4).
    STA DeathModeCounter
    STA ObjDir
    INC IsUpdatingMode          ; From the next frame the update half runs.

; ----------------------------------------------------------------------------
; Requests silence: $80 to tune channel 0 (stops the song) and $80 as the effect request.
; NOTE: An effect request with bit 7 set stops the DMC sample and tune channel 1 (Sfx_Drive).
; ----------------------------------------------------------------------------
Audio_SilenceSongTune1AndSample:
    LDA #$80
    STA Tune0Request
    STA EffectRequest

; Shared return of the routines above.
Mode11Death_Init_Return:
    RTS
