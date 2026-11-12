; ============================================================================
; FILE:     bank5/mode11-death-init.asm  BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 2039-2097
; SUBJECTS: (none)
; TOPIC:    Death init
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0015 InitMode11       code
;   L0043 InitMode11_Sub1  code
;   L0067 SilenceSound     code
;   L0072 L14A96_Exit      code
; ============================================================================

InitMode11:
    LDX #$00                    ; Decrease Link's invincibility timer. This is needed in submode 1.
    JSR DecrementInvincibilityTimer
    JSR Link_EndMoveAndAnimate
    LDA GameSubmode
    BNE InitMode11_Sub1         ; Go handle submode 1.

    ; Submode 0.
    JSR HideAllSprites
    JSR UpdateTriforcePositionMarker
    JSR DrawLinkBetweenRooms
    JSR DrawStatusBarItemsAndEnsureItemSelected
    JSR MaskCurPpuMaskGrayscale
    LDA #$00
    STA PauseState
    STA HeartPartial
    JSR FormatStatusBarText
    INC GameSubmode
    LDA #$10                    ; Set the invincibility timer, so that Link will flash for a fixed amount of time.
    STA ObjInvincibilityTimer

    ; Invincibility timers count down every 2 frames, while
    ; object timers count every frame. So, these 2 timers measure
    ; about the same amount of time.
    LDA #$21
    STA ObjTimer
    RTS

InitMode11_Sub1:
    LDA ObjTimer
    BNE L14A96_Exit             ; If ObjTimer hasn't expired, then return.
    JSR GetUniqueRoomId
    AND #$3E
    CMP #$3E
    BEQ :+                      ; If not in a cellar (unique room ID's $3E and $3F),
    LDA CurOpenedDoors          ; then lay out the doors again. But this seems redundant.
    STA PrevOpenedDoors
    JSR LayOutDoorsPrev
:
    LDA #$60                    ; DeathPaletteCycle
    STA FadeCycle
    LDA #$02                    ; Set a timer. But no one waits for it?
    STA ObjTimer+10
    LDA #$00                    ; Reset submode, CurRow, and player's state.
    STA GameSubmode
    STA CurRow
    STA ObjState
    LDA #$04                    ; There will be 4 turns starting from down direction.
    STA DeathModeCounter
    STA ObjDir
    INC IsUpdatingMode          ; Start updating.

SilenceSound:
    LDA #$80
    STA Tune0Request
    STA EffectRequest

L14A96_Exit:
    RTS
