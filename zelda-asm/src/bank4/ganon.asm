; ============================================================================
; FILE:     bank4/ganon.asm              BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 10284-10545
; SUBJECTS: Ganon
; TOPIC:    Ganon phases, teleport, shooting, dying
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0023 UpdateGanon              code
;   L0027 UpdateGanon_JumpTable    data
;   L0037 Ganon_ScenePhase0        code  Description:
;   L0097 Ganon_ScenePhase1        code  Description:
;   L0116 Ganon_DrawBodyFrame0     code
;   L0128 Ganon_ScenePhase2        code  Description:
;   L0155 Ganon_RandomizeLocation  code  Description:
;   L0165 Ganon_MoveAndShoot       code
;   L0193 GanonStartXs             data
;   L0196 Ganon_UpdateBrownState   code
;   L0220 L_Ganon_DrawBody         code
;   L0225 Ganon_Dying              code
; ============================================================================

UpdateGanon:
    LDA Ganon_ScenePhase
    JSR TableJump

UpdateGanon_JumpTable:
    .ADDR Ganon_ScenePhase0
    .ADDR Ganon_ScenePhase1
    .ADDR Ganon_ScenePhase2

; Description:
; Lift up the Triforce of Courage while the room is dark.
;
; Lift the Triforce of Courage -- represented with a triforce piece.
;
Ganon_ScenePhase0:
    LDA #$1B
    STA ItemTypeToLift

    ; If Link's timer has not expired, then go do nothing until the last frame.
    LDA ObjTimer
    BNE @CheckTimeToShout

    ; Once the timer expires, it's time to brighten the room.
    ;
    ; Save the current object slot.
    TXA
    PHA
    JSR UpdateCandle
    PLA                         ; Restore the current object slot.
    TAX

    ; Turn off the room brightening flag, because we're controlling it
    ; here instead of in the Mode 5 object update loop.
    LDA #$00
    STA IsRoomBrightening

    ; On the first frame of the fade-to-light cycle, play the
    ; Triforce/Ganon song.
    LDA FadeCycle
    CMP #$C0
    BNE @CheckFadeCycle
    LDA #$02
    STA SongRequest

@CheckFadeCycle:
    ; If the fade cycle has not ended, then go draw Ganon.
    LDA FadeCycle
    AND #$0F
    CMP #$04
    BNE Ganon_DrawBodyFrame0

    ; Set Link's timer to $C0 for the next scene phase.
    LDA #$C0
    STA ObjTimer

    ; Set scene phase 1, and go draw Ganon.
    INC Ganon_ScenePhase
    BNE Ganon_DrawBodyFrame0

@CheckTimeToShout:
    ; If timer = 1, then play Boss hit/hurt sound effect.
    ; In this case, I think it means Ganon is shouting.
    CMP #$01
    BNE :+
    LDA #$02
    STA SampleRequest
:
    RTS

; Description:
; Lift up the Triforce of Courage while the room is light.
;
; Lift the Triforce of Courage -- represented with a triforce piece.
;
Ganon_ScenePhase1:
    LDA #$1B
    STA ItemTypeToLift

    ; If Link's timer hasn't expired, then go draw Ganon.
    LDA ObjTimer
    BNE Ganon_DrawBodyFrame0

    ; Once the timer expires:
    ; 1. unhalt Link
    ; 2. clear the item to lift
    ; 3. play level 9 song
    ; 4. go to scene phase 2: fighting / not holding triforce
    STA ObjState
    STA ItemTypeToLift
    LDA #$20
    STA SongRequest
    INC Ganon_ScenePhase

Ganon_DrawBodyFrame0:
    ; When holding the Triforce of Courage, only draw Ganon
    ; with animation frame 0.
    LDA #$00
    STA Ganon_ObjAnimationFrame, X
    JMP Ganon_DrawBody

; Description:
; Fighting Ganon.
;
; If Ganon is dying, go handle it.
;
Ganon_ScenePhase2:
    LDA Ganon_ObjDyingCount, X
    BNE Ganon_Dying
    JSR Ganon_CheckCollisions
    JSR PlayBossHitCryIfNeeded

    ; Go handle the brown state specially.
    LDA ObjState, X
    BNE Ganon_UpdateBrownState

    ; State = 0: Blue
    ;
    ; If timer = 0, then go move around and shoot.
    LDA ObjTimer, X
    BEQ Ganon_MoveAndShoot

    ; If timer > 1, then Ganon is blue and visible. Only draw.
    ; Collisions were checked already.
    ;
    ; Else timer = 1. Fall thru and randomize Ganon's location,
    ; in anticipation of moving around when timer becomes 0 next frame.
    CMP #$01
    BNE L_Ganon_DrawBody

; Description:
; Put Ganon at Y=$A0, and a random X of $30 or $B0.
;
Ganon_RandomizeLocation:
    LDA #$A0
    STA ObjY, X
    LDA FrameCounter
    AND #$01
    TAY
    LDA GanonStartXs, Y
    STA ObjX, X
    RTS

Ganon_MoveAndShoot:
    ; Change the animation frame every screen frame; so that
    ; when Ganon is hit, he appears in a random pose.
    INC Ganon_ObjAnimationFrame, X

    ; The animation cycle has 6 frames.
    LDA Ganon_ObjAnimationFrame, X
    CMP #$06
    BNE @Move
    LDA #$00
    STA Ganon_ObjAnimationFrame, X

@Move:
    ; Move like a blue wizzrobe teleporting.
    LDA #$01
    STA ObjRemDistance, X
    JSR BlueWizzrobe_TurnSometimesAndMoveAndCheckTile

    ; Shoot every $40 frames.
    LDA FrameCounter
    AND #$3F
    BNE @Exit
    LDA #$56                    ; Fireball
    JSR ShootFireball

@Exit:
    RTS

GanonStartXs:
    .BYTE $30, $B0

Ganon_UpdateBrownState:
    ; Every other frame decrement the state.
    ; But every frame draw Ganon either translucent or opaque.
    LDA FrameCounter
    LSR
    BCC @Draw
    DEC ObjState, X

    ; If the state becomes 0, then Ganon is blue again.
    ; Go switch his palette row to reflect it.
    BNE @Draw
    JSR Ganon_RandomizeLocation
    JMP Ganon_AppendPaletteRowTransferRecord_Blue

@Draw:
    ; When state >= $30, draw Ganon opaque.
    ; When state goes below, draw him translucent.
    LDA ObjState, X
    CMP #$30
    BCS L_Ganon_DrawBody
    LDA FrameCounter
    LSR
    BCC :+

L_Ganon_DrawBody:
    JSR Ganon_DrawBody
:
    RTS

Ganon_Dying:
    ; Ganon is dying.
    ; Increment Ganon phase every frame.
    INC Ganon_ObjDyingCount, X

    ; If it just went from $FF to 0, then keep it at $FF.
    LDA Ganon_ObjDyingCount, X
    BNE :+
    LDA #$FF
    STA Ganon_ObjDyingCount, X
:
    ; If Ganon phase < $50, then go draw only.
    CMP #$50
    BCC L_Ganon_DrawBody

    ; If > $50, then go handle ashes only.
    BNE @HandleAshes

    ; Otherwise Ganon phase = $50.
    ; Set up the burst and ashes.
    JSR Ganon_AppendPaletteRowTransferRecord_Triforce

    ; Ganon is now a pile of ashes at offset (7, 8) from where he was.
    LDA ObjX, X
    ADC #$07
    STA ObjX, X
    LDA ObjY, X
    ADC #$08
    STA ObjY, X
    JSR Ganon_SetUpBurstRays
    JSR PlayBossDeathCry
    LDA #$02                    ; Ganon/Triforce song
    STA SongRequest

@HandleAshes:
    JSR Ganon_DrawAshes

    ; Keep drawing the burst artifacts while Ganon phase < $A0.
    LDA Ganon_ObjDyingCount, X
    CMP #$A0
    BCC Ganon_DrawBurst

    ; If > $A0, then there's nothing left to do, except return.
    BNE @Exit

    ; Otherwise Ganon phase = $A0.
    ; Activate the room item (triforce of power).
    JSR Ganon_ActivateRoomItem

    ; Put it at Ganon's ashes' location.
    LDA ObjX, X
    STA ObjX+19
    LDA ObjY, X
    STA ObjY+19

    ; Increase the room kill count.
    INC ObjType

@Exit:
    RTS
