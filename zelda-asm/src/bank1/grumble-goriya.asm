; ============================================================================
; FILE:     bank1/grumble-goriya.asm     BANK: 1 (switchable, $8000-$BFFF)
; UPSTREAM: Z_01.asm 1482-1567
; SUBJECTS: (none)
; TOPIC:    Hungry Goriya takes the food; filler
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0017 UpdateGrumble_Full                     code
;   L0037 UpdateGrumble_Full_JumpTable           data
;   L0043 UpdateGrumble1                         code
;   L0056 L_Person_FlagItemTakenAndAdvanceState  code
;   L0070 UpdateGrumble3                         code
;   L0088 InitUnderworldPerson_DoNothing         code
; ============================================================================

UpdateGrumble_Full:
    ; If state <> 3, then go check collisions and draw every frame.
    LDA ObjState+1
    CMP #$03
    BNE @DrawAndCheckCollisions

    ; In state 3, the person is translucent by drawing
    ; (and checking object collisions) every other frame.
    LDA FrameCounter
    AND #$01
    BNE :+

@DrawAndCheckCollisions:
    JSR Person_CheckCollisions
    JSR Anim_FetchObjPosForSpriteDescriptor
    JSR DrawObjectNotMirrored
:
    LDA ObjState+1
    JSR TableJump

UpdateGrumble_Full_JumpTable:
    .ADDR UpdatePersonState_Textbox
    .ADDR UpdateGrumble1
    .ADDR UpdatePersonState_CueTransferBlankPersonWares
    .ADDR UpdateGrumble3

UpdateGrumble1:
    ; If there's no food in object slot $F, then return.
    LDY #$0F
    LDA a:ObjState, Y
    ASL
    BCC InitUnderworldPerson_DoNothing

    ; Else halt Link and play the "secret found" tune.
    LDA #$40
    STA ObjState
    LDA #$04
    STA Tune1Request

L_Person_FlagItemTakenAndAdvanceState:
    ; Flag the secret/item of this room taken.
    JSR SetRoomFlagUWItemState

    ; Set object timer $40 for when we get to state 3.
    LDA #$40
    STA ObjTimer+1

    ; Cue transfer record of the blank textbox lines,
    ; and go to the next state, where a transfer record is cued
    ; to clear cave/person wares.
    LDA #$1E
    JMP CueTransferBufAndAdvanceState

UpdateGrumble3:
    JSR Link_EndMoveAndAnimate_Bank1

    ; If the timer has not expired, then return.
    LDA ObjTimer+1
    BNE InitUnderworldPerson_DoNothing

    ; Deactivate the food object in slot $F.
    LDY #$0F
    STA a:ObjState, Y

    ; Get rid of the food from the inventory.
    STA InvFood

    ; Make Link idle again, and reset this object's type.
    STA ObjState
    STA ObjType+1

InitUnderworldPerson_DoNothing:
    RTS

; Unknown block
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF
