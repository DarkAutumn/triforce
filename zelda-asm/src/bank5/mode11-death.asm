; ============================================================================
; FILE:     bank5/mode11-death.asm       BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 2523-2718
; SUBJECTS: Mode11Death, Transfer
; TOPIC:    Update half of the death mode ($11), thirteen steps: redraw the room for the death
;           palette, spin Link four times, fade the room, turn Link grey, show the spark, print
;           GAME OVER, count the death and go to the continue menu (mode 8).
; SEE ALSO: bank5/mode11-death-init.asm (the init half: flash, DeathModeCounter = 4, FadeCycle = $60);
;           bank5/mode8-continue-mode10-stairs.asm (mode 8, which follows);
;           bank1/world-fading.asm (Palette_AnimateFade); bank5/mode7-scroll-update.asm
;           (Transfer_ChoosePlayAreaAttrsHalfBySubmode; it also jumps to the two attribute-transfer entries
;           here); bank5/cave-cellar-enter.asm (Mode_InitSubroom_SelectTransferBuf);
;           bank6/transfer-buf-dispatch.asm (Transfer_BufAddrs: the buffers named by selector).
; KEY FACTS:
;   - GameSubmode $00-$0C selects the step; the mode ends in step $0C by entering mode 8.
;     Mode11Death_Update_Full_JumpTable.
;   - Step 1 requests tune 1 bit $80, which silences the song and plays the death tune.
;     Mode11Death_Update_Sub1.
;   - Steps 0-6 take 28 frames: one frame each except step 2, which copies the 22 play-area rows
;     to nametable 2, one row per frame. Mode11Death_Update_Sub2.
;   - Spin (step 7): Link turns every 5 frames in the order down, right, up, left; each return to
;     down counts one of the 4 spins of DeathModeCounter: 16 turns. Mode11Death_Update_Sub7Spin.
;   - Fade (step 8): the 4 palette steps of fade series 6, one per 10 frames.
;     Mode11Death_Update_Sub8Fade.
;   - Step 9 turns Link grey and waits $18 frames; step $0A shows the spark for $0F frames (tile $62
;     for the first 10, tile $64 for the last 5), requests tune 0 bit $10 and waits $2E frames.
;     Mode11Death_Update_Sub9GreyLink, Mode11Death_Update_SubASpark.
;   - Step $0B prints GAME OVER and waits $60 frames. Mode11Death_Update_SubBGameOver.
;   - Step $0C: GameMode = 8, the Game Over music (tune 1 bit $40, which repeats) starts, and
;     DeathCounts of the save slot goes up by 1, stopping at $FF. Mode11Death_Update_SubCEnd.
;   - None of the thirteen steps reads the controller: every wait runs its full length, about 320
;     frames in all (28 + 80 spin + 30 fade + $18 + $0F + $2E + $60).
;     Mode11Death_Update_Full_JumpTable.
; CONTENTS: (generated; L = line in this file)
;   L0061 Mode11Death_Update_Full                            code  Death mode update: runs the step selected by GameSubmode ($00-$0C).
;   L0066 Mode11Death_Update_Full_JumpTable                  data  TABLE: death mode steps; index = GameSubmode; value = routine address.
;   L0086 Mode11Death_Update_Sub1                            code  Death step 1: starts the death tune, then cues the bottom half of the play-area attributes.
;   L0093 Mode11Death_Update_Sub0                            code  Death step 0: cues the top half of the play-area attributes for nametable 2.
;   L0101 Transfer_CuePlayAreaAttrsHalfNT2AndAdvanceSubmode  code  Cues a transfer of half the play-area attributes to nametable 2 ($2Bxx), then advances the submode.
;   L0110 Transfer_CuePlayAreaAttrsHalfAndAdvanceSubmode     code  Cues a transfer of half the play-area attributes ($18 bytes), then advances the submode.
;   L0120 Mode11Death_Update_Sub2                            code  Death step 2: copies the play-area tiles to nametable 2, one row per frame (rows 0-$15).
;   L0134 Mode11Death_Update_Sub3                            code  Death step 3: cues the static transfer of the death attributes, top half (selector $60).
;   L0138 Mode11Death_SelectTransferBufAndAdvanceSubmode     code  Sets TileBufSelector = A and advances the submode (jump to the shared helper).
;   L0144 Mode11Death_Update_Sub4                            code  Death step 4: cues the static transfer of the death attributes, bottom half (selector $62).
;   L0151 Mode11Death_Update_Sub5                            code  Death step 5: turns the sprite-0 split off and cues the death background palette (selector $5E).
;   L0160 Mode11Death_Update_Sub6                            code  Death step 6: clears bit 0 of CurPpuControl (selects an even-numbered nametable).
;   L0166 Mode11Death_AdvanceSubmode                         code  Shared tail: advances the submode and returns.
;   L0176 Mode11Death_Update_Sub7Spin                        code  Death step 7: spins Link, one quarter turn every 5 frames, until 4 full spins are done.
;   L0211 Mode11Death_Update_Sub8Fade                        code  Death step 8: runs the palette fade (FadeCycle, series 6) until its 4 steps are done.
;   L0220 Mode11Death_Update_Sub9GreyLink                    code  Death step 9: cues the grey palette row for dead Link and starts an $18-frame wait.
;   L0236 Mode11Death_Update_SubASpark                       code  Death step $0A: after the wait, replaces Link by a spark for $0F frames, then hides it.
;   L0269 Mode11Death_SetTimerAndAdvanceSubmode              code  Shared tail: ObjTimer[$0B] = A, advances the submode.
;   L0274 Mode11Death_Update_Return                          code  Shared return of the death steps.
;   L0280 Mode11Death_Update_SubBGameOver                    code  Death step $0B: after the wait, cues the GAME OVER text and starts a $60-frame wait.
;   L0293 Mode11Death_Update_SubCEnd                         code  Death step $0C: after the wait, counts the death and goes to the continue menu (mode 8).
; ============================================================================

; ----------------------------------------------------------------------------
; Death mode update: runs the step selected by GameSubmode ($00-$0C).
; ----------------------------------------------------------------------------
Mode11Death_Update_Full:
    LDA GameSubmode
    JSR Util_TableJump

; TABLE: death mode steps; index = GameSubmode; value = routine address.
Mode11Death_Update_Full_JumpTable:
    .ADDR Mode11Death_Update_Sub0              ; 0: play-area attributes, top half, to nametable 2
    .ADDR Mode11Death_Update_Sub1              ; 1: death tune; attributes, bottom half
    .ADDR Mode11Death_Update_Sub2              ; 2: play-area tiles to nametable 2, a row per frame
    .ADDR Mode11Death_Update_Sub3              ; 3: static death attributes, top half
    .ADDR Mode11Death_Update_Sub4              ; 4: static death attributes, bottom half
    .ADDR Mode11Death_Update_Sub5              ; 5: death background palette; sprite-0 split off
    .ADDR Mode11Death_Update_Sub6              ; 6: select an even nametable
    .ADDR Mode11Death_Update_Sub7Spin              ; 7: Link spins 4 times
    .ADDR Mode11Death_Update_Sub8Fade  ; 8: palette fade
    .ADDR Mode11Death_Update_Sub9GreyLink              ; 9: Link turns grey; $18-frame wait
    .ADDR Mode11Death_Update_SubASpark              ; A: spark; $2E-frame wait
    .ADDR Mode11Death_Update_SubBGameOver              ; B: GAME OVER text; $60-frame wait
    .ADDR Mode11Death_Update_SubCEnd              ; C: death count, go to mode 8

; ----------------------------------------------------------------------------
; Death step 1: starts the death tune, then cues the bottom half of the play-area attributes.
; NOTE: Tune 1 request $80 silences the song before the tune starts. Falls into step 0's code,
;       which picks the bottom half because GameSubmode is not 0.
; ----------------------------------------------------------------------------
Mode11Death_Update_Sub1:
    LDA #$80                    ; Death tune (and silence the song).
    STA Tune1Request

; ----------------------------------------------------------------------------
; Death step 0: cues the top half of the play-area attributes for nametable 2.
; ----------------------------------------------------------------------------
Mode11Death_Update_Sub0:
    ; A = low PPU address ($D0 top half, $E8 bottom half), Y = end offset in PlayAreaAttrs.
    JSR Transfer_ChoosePlayAreaAttrsHalfBySubmode

; ----------------------------------------------------------------------------
; Cues a transfer of half the play-area attributes to nametable 2 ($2Bxx), then advances the submode.
; IN:   A = low byte of the PPU address; Y = offset in PlayAreaAttrs of the last byte to copy.
; ----------------------------------------------------------------------------
Transfer_CuePlayAreaAttrsHalfNT2AndAdvanceSubmode:
    LDX #$2B                    ; High byte of the PPU address: the attribute table of nametable 2.

; ----------------------------------------------------------------------------
; Cues a transfer of half the play-area attributes ($18 bytes), then advances the submode.
; IN:   X = high byte and A = low byte of the PPU address; Y = offset in PlayAreaAttrs of the last
;       byte to copy.
; OUT:  DynTileBuf holds the record; GameSubmode incremented.
; ----------------------------------------------------------------------------
Transfer_CuePlayAreaAttrsHalfAndAdvanceSubmode:
    JSR Transfer_CopyPlayAreaAttrsHalf
    INC GameSubmode
    RTS

; ----------------------------------------------------------------------------
; Death step 2: copies the play-area tiles to nametable 2, one row per frame (rows 0-$15).
; NOTE: 22 frames. After the last row the submode advances and sprite 0 is written and its
;       check enabled (Scroll_WriteAndEnableSprite0).
; ----------------------------------------------------------------------------
Mode11Death_Update_Sub2:
    JSR Mode_CopyNextRowAndAdvanceSubmodeWhenDone
    BCC :+                      ; C set = that was the last row.
    JSR Scroll_WriteAndEnableSprite0
:
    LDA DynTileBuf              ; Add 8 to the high byte of the record's PPU address: the same
    CLC                         ; row in nametable 2 instead of nametable 0.
    ADC #$08
    STA DynTileBuf
    RTS

; ----------------------------------------------------------------------------
; Death step 3: cues the static transfer of the death attributes, top half (selector $60).
; ----------------------------------------------------------------------------
Mode11Death_Update_Sub3:
    LDA #$60                    ; Transfer_Mode11PlayAreaAttrsTopHalfBuf

; Sets TileBufSelector = A and advances the submode (jump to the shared helper).
Mode11Death_SelectTransferBufAndAdvanceSubmode:
    JMP Mode_InitSubroom_SelectTransferBuf

; ----------------------------------------------------------------------------
; Death step 4: cues the static transfer of the death attributes, bottom half (selector $62).
; ----------------------------------------------------------------------------
Mode11Death_Update_Sub4:
    LDA #$62                    ; Transfer_Mode11PlayAreaAttrsBottomHalfBuf
    JMP Mode11Death_SelectTransferBufAndAdvanceSubmode

; ----------------------------------------------------------------------------
; Death step 5: turns the sprite-0 split off and cues the death background palette (selector $5E).
; ----------------------------------------------------------------------------
Mode11Death_Update_Sub5:
    LDA #$00
    STA IsSprite0CheckActive
    LDA #$5E                    ; Transfer_Mode11BackgroundPaletteBottomHalfBuf
    JMP Mode11Death_SelectTransferBufAndAdvanceSubmode

; ----------------------------------------------------------------------------
; Death step 6: clears bit 0 of CurPpuControl (selects an even-numbered nametable).
; ----------------------------------------------------------------------------
Mode11Death_Update_Sub6:
    LDA CurPpuControl
    AND #$FE
    STA CurPpuControl

; Shared tail: advances the submode and returns.
Mode11Death_AdvanceSubmode:
    INC GameSubmode
    RTS

; ----------------------------------------------------------------------------
; Death step 7: spins Link, one quarter turn every 5 frames, until 4 full spins are done.
; IN:   DeathModeCounter = spins left (4 from the init); ObjDir = 4 (down) at the start.
; NOTE: Turn order: down (4) -> right (1) -> up (8) -> left (2) -> down. The counter drops on each
;       left -> down turn, so the step makes 16 turns; ObjTimer[$0B] is set to 5 at each turn.
; ----------------------------------------------------------------------------
Mode11Death_Update_Sub7Spin:
    LDA DeathModeCounter
    BEQ Mode11Death_AdvanceSubmode       ; All spins done: next step.
    LDA ObjTimer+11
    BNE @DrawLink               ; Between turns: only redraw Link.
    LDA #$05                    ; Time to turn: arm the timer for the next turn.
    STA ObjTimer+11
    LDA ObjDir
    LSR
    LSR
    BCC @CheckOtherDirs         ; C = bit 1 (left). Not facing left: go handle the others.

    ; Facing left: one spin is complete. Face down next.
    DEC DeathModeCounter
    LDA #$04

@SetLinkDirAndDraw:
    STA ObjDir

@DrawLink:
    JMP Link_EndMoveAndAnimate

@CheckOtherDirs:
    ; A = direction >> 2. Facing down (4) gives 1 = right; facing up (8) gives 2 = left:
    ; nonzero, so use it as the next direction.
    BNE @SetLinkDirAndDraw

    ; A = 0: facing right. Face up next.
    LDA #$08
    BNE @SetLinkDirAndDraw      ; Always taken.

; ----------------------------------------------------------------------------
; Death step 8: runs the palette fade (FadeCycle, series 6) until its 4 steps are done.
; NOTE: One palette step per 10 frames (Palette_AnimateFade).
; ----------------------------------------------------------------------------
Mode11Death_Update_Sub8Fade:
    JSR Palette_AnimateFade
    BEQ Mode11Death_AdvanceSubmode       ; Z set = the fade is done: next step.
    RTS

; ----------------------------------------------------------------------------
; Death step 9: cues the grey palette row for dead Link and starts an $18-frame wait.
; OUT:  DeathModeCounter = $0F (frames of spark for the next step); ObjTimer[$0B] = $18.
; ----------------------------------------------------------------------------
Mode11Death_Update_Sub9GreyLink:
    LDA #$2C                    ; Transfer_Mode11DeadLinkPaletteBuf
    STA TileBufSelector
    LDA #$0F

    ; From here on DeathModeCounter counts the frames the spark is shown.
    STA DeathModeCounter
    LDA #$18
    BNE Mode11Death_SetTimerAndAdvanceSubmode    ; Always taken: set the timer, advance the submode.

; ----------------------------------------------------------------------------
; Death step $0A: after the wait, replaces Link by a spark for $0F frames, then hides it.
; NOTE: The spark uses Link's two sprite slots ($12, $13) at Link's position: tile $62 while
;       DeathModeCounter >= 6 (10 frames), tile $64 for the last 5. When the counter reaches 0 the
;       sprites are hidden, tune 0 bit $10 is requested and a $2E-frame wait starts.
; ----------------------------------------------------------------------------
Mode11Death_Update_SubASpark:
    LDA ObjTimer+11
    BNE Mode11Death_Update_Return             ; Wait until the timer expires.
    LDX #$62                    ; The small spark tile.
    LDA DeathModeCounter        ; Frames of spark left.
    CMP #$06
    BCS :+                      ; 6 or more left: small spark.
    LDX #$64                    ; Else the big spark tile.
:
    LDA ObjY
    STA Sprites+72              ; Link's Y for both halves.
    STA Sprites+76
    STX Sprites+73              ; The chosen spark tile for both halves.
    STX Sprites+77
    LDA #$01                    ; Left half: sprite palette 1 (palette row 5).
    STA Sprites+74
    LDA #$41                    ; Right half: the same palette, flipped horizontally.
    STA Sprites+78
    LDA ObjX                    ; Link's X for the left half,
    STA Sprites+75
    CLC                         ; and 8 pixels further right for the right half.
    ADC #$08
    STA Sprites+79
    DEC DeathModeCounter
    BNE Mode11Death_Update_Return             ; Spark frames left: return.
    LDA #$10                    ; Request tune 0 bit $10 (the tune also used for a heart pickup).
    STA Tune0Request
    LDA #$F8                    ; Hide the two spark sprites.
    STA Sprites+72
    STA Sprites+76
    LDA #$2E                    ; Wait $2E frames before the next step.

; Shared tail: ObjTimer[$0B] = A, advances the submode.
Mode11Death_SetTimerAndAdvanceSubmode:
    STA ObjTimer+11
    INC GameSubmode

; Shared return of the death steps.
Mode11Death_Update_Return:
    RTS

; ----------------------------------------------------------------------------
; Death step $0B: after the wait, cues the GAME OVER text and starts a $60-frame wait.
; ----------------------------------------------------------------------------
Mode11Death_Update_SubBGameOver:
    LDA ObjTimer+11
    BNE Mode11Death_Update_Return             ; Wait until the timer expires.
    LDA #$60                    ; Wait $60 frames before the next step.
    STA ObjTimer+11
    LDA #$46                    ; Transfer_GameOverBuf
    JMP Mode11Death_SelectTransferBufAndAdvanceSubmode

; ----------------------------------------------------------------------------
; Death step $0C: after the wait, counts the death and goes to the continue menu (mode 8).
; OUT:  GameMode = 8 with IsUpdatingMode and GameSubmode reset; Tune1Request = $40 (Game Over
;       music); DeathCounts[CurSaveSlot] incremented unless it is $FF.
; ----------------------------------------------------------------------------
Mode11Death_Update_SubCEnd:
    LDA ObjTimer+11
    BNE @Exit                   ; Wait until the timer expires.
    JSR Mode_End
    LDA #$08                    ; Mode 8: continue menu.
    STA GameMode
    LDA #$40                    ; Game Over music.
    STA Tune1Request
    LDX CurSaveSlot
    LDA DeathCounts, X          ; Count the death for this save slot,
    CMP #$FF                    ; up to the maximum $FF.
    BEQ @Exit
    INC DeathCounts, X
@Exit:
    RTS
