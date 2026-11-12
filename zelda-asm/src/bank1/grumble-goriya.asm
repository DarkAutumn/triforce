; ============================================================================
; FILE:     bank1/grumble-goriya.asm     BANK: 1 (switchable, $8000-$BFFF)
; UPSTREAM: Z_01.asm 1482-1567
; SUBJECTS: Grumble, Person
; TOPIC:    Update of the Grumble Goriya (object type $36), the hungry Goriya that blocks the way until
;           Link puts down the food; the "offer taken" step shared with other persons; filler.
; SEE ALSO: bank1/underworld-person-init.asm (Grumble_Init_Full); bank5/wield-items.asm (Food_Wield puts
;           the food in slot $0F, state $80); bank7/boomerang-arrow-food.asm (the food object);
;           bank7/walker-move.asm and bank1/modeb-cave-save-sprites.asm (Person_BlockLinkMovingUp: while
;           slot 1 holds type $36 or $4B-$52, Link cannot move up when his Y < $8E);
;           bank1/underworld-person-update.asm (Person_CheckCollisions).
; KEY FACTS:
;   - The Goriya reacts to food anywhere in the room: the only test is bit 7 of ObjState+$0F (the
;     food is active), made every frame once its text has been typed. Grumble_State1WaitForFood
;   - On seeing the food: Link is halted (ObjState = $40), the "secret found" tune plays, the room's
;     item-taken flag ($10) is set at once, and a $40-frame timer starts. Grumble_State1WaitForFood,
;     Person_FlagTakenAndBlankText
;   - When the timer ends: the food object is removed, InvFood = 0 (the food is used up for good),
;     Link is released and slot 1 is freed (ObjType+1 = 0), so nothing blocks the way up any more.
;     Grumble_State3TakeFood
;   - Because the room flag is set, the Goriya is not created again on later visits (see
;     Grumble_Init_Full). Person_FlagTakenAndBlankText
;   - This code changes nothing else in the room: it sets no shutter trigger and gives no item.
;     Grumble_State3TakeFood
;   - A weapon hit does not kill the Goriya; like any person it turns the fireballs on.
;     Grumble_Update_Full
; CONTENTS: (generated; L = line in this file)
;   L0041 Grumble_Update_Full           code  Updates the Grumble Goriya (type $36): checks collisions, draws it, then runs its state.
;   L0061 Grumble_Update_JumpTable      data  TABLE: states of the Grumble Goriya; index = ObjState+1; value = routine address.
;   L0073 Grumble_State1WaitForFood     code  Grumble state 1: waits until the food is out (slot $0F active with bit 7), then halts Link.
;   L0092 Person_FlagTakenAndBlankText  code  Marks the room's offer taken: sets the item-taken room flag, starts a $40-frame timer, blanks text.
;   L0109 Grumble_State3TakeFood        code  Grumble state 3: when the timer ends, removes the food and the Goriya and releases Link.
;   L0128 Person_InitUnderworld_Return  code  Shared return; also the level 0 entry of the dungeon person init table (does nothing).
; ============================================================================

; ----------------------------------------------------------------------------
; Updates the Grumble Goriya (type $36): checks collisions, draws it, then runs its state.
; IN:   X = object slot (1). ObjState+1 = state (0-3).
; NOTE: In state 3 it is drawn, and checked for collisions, only on even frames (it flickers).
; ----------------------------------------------------------------------------
Grumble_Update_Full:
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
    JSR Anim_DrawObjectNotMirrored
:
    LDA ObjState+1
    JSR Util_TableJump

; TABLE: states of the Grumble Goriya; index = ObjState+1; value = routine address.
Grumble_Update_JumpTable:
    .ADDR Person_UpdateState_Textbox                         ; 0: type the text, then unhalt Link
    .ADDR Grumble_State1WaitForFood                                    ; 1: wait for the food
    .ADDR Person_UpdateState_BlankPriceRow     ; 2: blank text line 3, price row (1 frame)
    .ADDR Grumble_State3TakeFood                                    ; 3: flicker, then take the food and vanish

; ----------------------------------------------------------------------------
; Grumble state 1: waits until the food is out (slot $0F active with bit 7), then halts Link.
; OUT:  when food is seen: ObjState = $40 (Link halted), Tune1Request = $04, then falls into
;       Person_FlagTakenAndBlankText (room flag, $40-frame timer, state 2).
; NOTE: The position of the food and of Link do not matter.
; ----------------------------------------------------------------------------
Grumble_State1WaitForFood:
    ; Slot $0F holds the boomerang or the food; bit 7 of its state means food.
    ; If it is clear, then return.
    LDY #$0F
    LDA a:ObjState, Y
    ASL
    BCC Person_InitUnderworld_Return

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
Person_FlagTakenAndBlankText:
    JSR RoomFlags_SetItemTaken

    ; Timer for the flicker state that follows the next one (it counts down meanwhile).
    LDA #$40
    STA ObjTimer+1

    ; Cue the transfer that blanks textbox lines 1 and 2, and go to the next state,
    ; where the transfer that blanks line 3 and the price row is cued.
    LDA #$1E
    JMP Person_CueTransferBufAndAdvanceState

; ----------------------------------------------------------------------------
; Grumble state 3: when the timer ends, removes the food and the Goriya and releases Link.
; OUT:  at timer 0: ObjState+$0F = 0, InvFood = 0, ObjState = 0, ObjType+1 = 0.
; NOTE: Link stays halted (and is animated) until then, about $40 frames after the food was seen.
; ----------------------------------------------------------------------------
Grumble_State3TakeFood:
    JSR Link_EndMoveAndAnimate_ReturnToBank1

    ; If the timer has not expired, then return. (A = 0 below.)
    LDA ObjTimer+1
    BNE Person_InitUnderworld_Return

    ; Deactivate the food object in slot $F.
    LDY #$0F
    STA a:ObjState, Y

    ; Remove the food from the inventory.
    STA InvFood

    ; Make Link idle again, and free the Goriya's slot (type 0).
    STA ObjState
    STA ObjType+1

; Shared return; also the level 0 entry of the dungeon person init table (does nothing).
Person_InitUnderworld_Return:
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
