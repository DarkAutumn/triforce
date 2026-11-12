; ============================================================================
; FILE:     bank1/grumble-goriya.asm     BANK: 1 (switchable, $8000-$BFFF)
; UPSTREAM: Z_01.asm 1482-1567
; SUBJECTS: (none)
; TOPIC:    Update of the Grumble Goriya (object type $36), the hungry Goriya that blocks the way until
;           Link puts down the food; the "offer taken" step shared with other persons; filler.
; SEE ALSO: bank1/underworld-person-init.asm (InitGrumble_Full); bank5/wield-items.asm (WieldFood puts
;           the food in slot $0F, state $80); bank7/boomerang-arrow-food.asm (the food object);
;           bank7/walker-move.asm and bank1/modeb-cave-save-sprites.asm (CheckPersonBlocking: while
;           slot 1 holds type $36 or $4B-$52, Link cannot move up when his Y < $8E);
;           bank1/underworld-person-update.asm (Person_CheckCollisions).
; KEY FACTS:
;   - The Goriya reacts to food anywhere in the room: the only test is bit 7 of ObjState+$0F (the
;     food is active), made every frame once its text has been typed. UpdateGrumble1
;   - On seeing the food: Link is halted (ObjState = $40), the "secret found" tune plays, the room's
;     item-taken flag ($10) is set at once, and a $40-frame timer starts. UpdateGrumble1,
;     L_Person_FlagItemTakenAndAdvanceState
;   - When the timer ends: the food object is removed, InvFood = 0 (the food is used up for good),
;     Link is released and slot 1 is freed (ObjType+1 = 0), so nothing blocks the way up any more.
;     UpdateGrumble3
;   - Because the room flag is set, the Goriya is not created again on later visits (see
;     InitGrumble_Full). L_Person_FlagItemTakenAndAdvanceState
;   - This code changes nothing else in the room: it sets no shutter trigger and gives no item.
;     UpdateGrumble3
;   - A weapon hit does not kill the Goriya; like any person it turns the fireballs on.
;     UpdateGrumble_Full
; CONTENTS: (generated; L = line in this file)
;   L0041 UpdateGrumble_Full                     code  Updates the Grumble Goriya (type $36): checks collisions, draws it, then runs its state.
;   L0061 UpdateGrumble_Full_JumpTable           data  TABLE: states of the Grumble Goriya; index = ObjState+1; value = routine address.
;   L0073 UpdateGrumble1                         code  Grumble state 1: waits until the food is out (slot $0F active with bit 7), then halts Link.
;   L0092 L_Person_FlagItemTakenAndAdvanceState  code  Marks the room's offer taken: sets the item-taken room flag, starts a $40-frame timer, blanks text.
;   L0109 UpdateGrumble3                         code  Grumble state 3: when the timer ends, removes the food and the Goriya and releases Link.
;   L0128 InitUnderworldPerson_DoNothing         code  Shared return; also the level 0 entry of the dungeon person init table (does nothing).
; ============================================================================

; ----------------------------------------------------------------------------
; Updates the Grumble Goriya (type $36): checks collisions, draws it, then runs its state.
; IN:   X = object slot (1). ObjState+1 = state (0-3).
; NOTE: In state 3 it is drawn, and checked for collisions, only on even frames (it flickers).
; ----------------------------------------------------------------------------
UpdateGrumble_Full:
    ; If state <> 3, then go check collisions and draw every frame.
    LDA ObjState+1
    CMP #$03
    BNE @DrawAndCheckCollisions

    ; In state 3, skip odd frames.
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

; TABLE: states of the Grumble Goriya; index = ObjState+1; value = routine address.
UpdateGrumble_Full_JumpTable:
    .ADDR UpdatePersonState_Textbox                         ; 0: type the text, then unhalt Link
    .ADDR UpdateGrumble1                                    ; 1: wait for the food
    .ADDR UpdatePersonState_CueTransferBlankPersonWares     ; 2: blank text line 3, price row (1 frame)
    .ADDR UpdateGrumble3                                    ; 3: flicker, then take the food and vanish

; ----------------------------------------------------------------------------
; Grumble state 1: waits until the food is out (slot $0F active with bit 7), then halts Link.
; OUT:  when food is seen: ObjState = $40 (Link halted), Tune1Request = $04, then falls into
;       L_Person_FlagItemTakenAndAdvanceState (room flag, $40-frame timer, state 2).
; NOTE: The position of the food and of Link do not matter.
; ----------------------------------------------------------------------------
UpdateGrumble1:
    ; Slot $0F holds the boomerang or the food; bit 7 of its state means food.
    ; If it is clear, then return.
    LDY #$0F
    LDA a:ObjState, Y
    ASL
    BCC InitUnderworldPerson_DoNothing

    ; Else halt Link and play the "secret found" tune.
    LDA #$40
    STA ObjState
    LDA #$04
    STA Tune1Request

; ----------------------------------------------------------------------------
; Marks the room's offer taken: sets the item-taken room flag, starts a $40-frame timer, blanks text.
; OUT:  room flag $10 set; ObjTimer+1 = $40; TileBufSelector = $1E; ObjState+1 incremented.
; NOTE: Also the tail of the bomb capacity purchase and of the life-or-money payment.
; ----------------------------------------------------------------------------
L_Person_FlagItemTakenAndAdvanceState:
    JSR SetRoomFlagUWItemState

    ; Timer for the flicker state that follows the next one (it counts down meanwhile).
    LDA #$40
    STA ObjTimer+1

    ; Cue the transfer that blanks textbox lines 1 and 2, and go to the next state,
    ; where the transfer that blanks line 3 and the price row is cued.
    LDA #$1E
    JMP CueTransferBufAndAdvanceState

; ----------------------------------------------------------------------------
; Grumble state 3: when the timer ends, removes the food and the Goriya and releases Link.
; OUT:  at timer 0: ObjState+$0F = 0, InvFood = 0, ObjState = 0, ObjType+1 = 0.
; NOTE: Link stays halted (and is animated) until then, about $40 frames after the food was seen.
; ----------------------------------------------------------------------------
UpdateGrumble3:
    JSR Link_EndMoveAndAnimate_Bank1

    ; If the timer has not expired, then return. (A = 0 below.)
    LDA ObjTimer+1
    BNE InitUnderworldPerson_DoNothing

    ; Deactivate the food object in slot $F.
    LDY #$0F
    STA a:ObjState, Y

    ; Remove the food from the inventory.
    STA InvFood

    ; Make Link idle again, and free the Goriya's slot (type 0).
    STA ObjState
    STA ObjType+1

; Shared return; also the level 0 entry of the dungeon person init table (does nothing).
InitUnderworldPerson_DoNothing:
    RTS

; DATA: filler ($FF), 82 bytes, not referenced.
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
