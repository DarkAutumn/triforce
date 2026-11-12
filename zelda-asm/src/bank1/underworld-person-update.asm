; ============================================================================
; FILE:     bank1/underworld-person-update.asm BANK: 1 (switchable, $8000-$BFFF)
; UPSTREAM: Z_01.asm 1170-1326
; SUBJECTS: (none)
; TOPIC:    Dungeon old man update, fireballs when attacked, more-bombs offer
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0022 UpdateUnderworldPerson_Full                      code
;   L0041 UpdateUnderworldPerson_Full_JumpTable            data
;   L0046 UpdatePersonState_ResetCharOffset                code
;   L0051 UpdatePersonState_DoNothing                      code
;   L0054 Person_CheckCollisions                           code
;   L0067 UpdateUnderworldPersonComplex                    code
;   L0095 UpdateUnderworldPersonComplex_JumpTable          data
;   L0102 UpdateUnderworldPersonComplexState_Begin         code
;   L0117 UpdateUnderworldPersonComplexState_SenseLink     code
;   L0167 UpdateUnderworldPersonComplexState_DelayAndQuit  code
;   L0175 Person_DrawAndCheckCollisions                    code
; ============================================================================

UpdateUnderworldPerson_Full:
    ; If level is not one of {3, 4, 6, 8, 9}, then
    ; go handle people that do more than talk.
    LDA CurLevel
    CMP #$03
    BCC :+
    CMP #$05
    BEQ :+
    CMP #$07
    BNE @DrawAndCheckCollisions
:
    JMP UpdateUnderworldPersonComplex

@DrawAndCheckCollisions:
    ; Else simply draw, check collisions, and show the text.
    JSR Person_DrawAndCheckCollisions
    LDA ObjState+1
    JSR TableJump

UpdateUnderworldPerson_Full_JumpTable:
    .ADDR UpdatePersonState_ResetCharOffset
    .ADDR UpdatePersonState_Textbox
    .ADDR UpdatePersonState_DoNothing

UpdatePersonState_ResetCharOffset:
    LDA #$00
    STA PersonTextIndex
    INC ObjState+1

UpdatePersonState_DoNothing:
    RTS

Person_CheckCollisions:
    JSR CheckMonsterCollisions

    ; If the person died, then Link hit him. So, start shooting
    ; fireballs, then reset the metastate, so that he's not dead.
    LDA ObjMetastate+1
    BEQ :+
    STA PersonFireballsEnabled
    LDA #$00
    STA ObjMetastate+1
:
    RTS

UpdateUnderworldPersonComplex:
    ; In state 4, the person is shown every other frame.
    LDA ObjState+1
    CMP #$04
    BNE :+
    LDA FrameCounter
    AND #$01
    BNE @HandleState
:
    JSR Person_DrawAndCheckCollisions

    ; If this is the "more bombs" man, then show a rupee
    ; at ($78, $98) using room item object slot $13.
    LDA ObjType+1
    CMP #$4F
    BNE @HandleState
    LDA #$78
    STA ObjX+19
    LDA #$98
    STA ObjY+19
    LDA #$18
    LDX #$13
    JSR AnimateItemObject

@HandleState:
    LDA ObjState+1
    JSR TableJump

UpdateUnderworldPersonComplex_JumpTable:
    .ADDR UpdateUnderworldPersonComplexState_Begin
    .ADDR UpdatePersonState_Textbox
    .ADDR UpdateUnderworldPersonComplexState_SenseLink
    .ADDR UpdatePersonState_CueTransferBlankPersonWares
    .ADDR UpdateUnderworldPersonComplexState_DelayAndQuit

UpdateUnderworldPersonComplexState_Begin:
    ; If this is the "more bombs" man, then show the price of the offer.
    LDA ObjType+1
    CMP #$4F
    BNE :+
    LDA #$6C                    ; "-100"
    STA TileBufSelector
:
    ; Set a timer of $A frames before showing characters,
    ; and go to state 1.
    LDA #$0A
    STA ObjTimer+1
    INC ObjState+1
    RTS

UpdateUnderworldPersonComplexState_SenseLink:
    ; If this is not the "more bombs" man, then return.
    LDA ObjType+1
    CMP #$4F
    BNE @Exit

    ; If Link's X <> $78, then return.
    LDA ObjX
    CMP #$78
    BNE @Exit

    ; If the distance between Link's Y and the rupee's ($98) < 6,
    ; then go see if Link can pay for it.
    ; Else return.
    LDA ObjY
    SEC
    SBC #$98
    JSR Abs
    CMP #$06
    BCC @CheckCanPay

@Exit:
    RTS

@CheckCanPay:
    ; If rupee count < 100, then return.
    LDA #$64
    CMP InvRupees
    BEQ :+
    BCS @Exit
:
    ; Post 100 rupees more to subtract.
    CLC
    ADC RupeesToSubtract
    STA RupeesToSubtract

    ; Play the "key taken" sound effect.
    LDA #$08
    STA Tune0Request

    ; Increase max bombs by 4, set the amount on hand to the max.
    LDA MaxBombs
    CLC
    ADC #$04
    STA MaxBombs
    STA InvBombs

    ; Go mark this offer taken.
    JMP L_Person_FlagItemTakenAndAdvanceState

UpdateUnderworldPersonComplexState_DelayAndQuit:
    ; If the object timer has expired, then destroy this object.
    LDA ObjTimer+1
    BNE :+
    STA ObjType+1
:
    RTS

Person_DrawAndCheckCollisions:
    JSR Person_CheckCollisions
    JSR Anim_FetchObjPosForSpriteDescriptor
    JMP DrawObjectMirrored
