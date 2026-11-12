; ============================================================================
; FILE:     bank1/life-or-money.asm      BANK: 1 (switchable, $8000-$BFFF)
; UPSTREAM: Z_01.asm 1328-1480
; SUBJECTS: Person
; TOPIC:    Update of the life-or-money man (dungeon person type $51): Link must give up a heart
;           container or 50 rupees; paying opens the room's shutters.
; SEE ALSO: bank1/underworld-person-init.asm (Person_InitLifeOrMoney_Full: text, removal once
;           paid); bank1/underworld-person-update.asm (Person_DrawAndCheckCollisions, state 4);
;           bank1/grumble-goriya.asm (Person_FlagTakenAndBlankText); bank1/person-textbox.asm.
; KEY FACTS:
;   - Two items are shown at Y = $98: a heart container at X = $58 and a rupee at X = $98. Link
;     chooses by standing at exactly that X with Y within 5 pixels of $98, once the text has ended.
;     Person_LifeOrMoneyItemXs, Person_LifeOrMoneyItemIds, Person_LifeOrMoneyState2Choose
;   - The rupee costs 50 and is ignored while InvRupees < 50; the heart container has no
;     requirement. Person_LifeOrMoneyState2Choose
;   - Heart container with 4 or more containers: one container and one full heart are removed (full
;     hearts stop at 0; HeartPartial is kept). Person_LifeOrMoneyState2Choose@Reduce
;   - Heart container with 3 or fewer containers: no container is removed; instead full hearts and
;     HeartPartial are both set to 0. Person_LifeOrMoneyState2Choose@PayWithHeartContainer
;   - Either payment sets ShutterTrigger = 1 and the room's item-taken flag and starts a $40-frame
;     timer; the man flickers until it ends and is gone for good.
;     Person_LifeOrMoneyState2Choose@EndLifeOrMoney
; CONTENTS: (generated; L = line in this file)
;   L0038 Person_UpdateLifeOrMoney_Full       code  Updates the life-or-money man (type $51): draws him and the two items, then runs his state.
;   L0055 Person_UpdateLifeOrMoney_JumpTable  data  TABLE: states of the life-or-money man; index = ObjState+1; value = routine address.
;   L0063 Person_LifeOrMoneyItemXs            data  TABLE: X of the two items offered (Y is $98); index = 0 heart container, 1 rupee; value = pixels.
;   L0067 Person_LifeOrMoneyItemIds           data  TABLE: item ids drawn for the two choices; index = 0, 1; value = item id.
;   L0074 Person_DrawLifeOrMoneyItems         code  Draws the heart container at ($58, $98) and the rupee at ($98, $98) through object slot $13.
;   L0104 Person_LifeOrMoneyState0ShowCosts   code  Life-or-money state 0: shows the costs "-1" and "-50" and waits $0A frames before the text starts.
;   L0121 Person_LifeOrMoneyState2Choose      code  Life-or-money state 2: takes a heart container or 50 rupees when Link stands on one of the items.
; ============================================================================

; ----------------------------------------------------------------------------
; Updates the life-or-money man (type $51): draws him and the two items, then runs his state.
; IN:   X = object slot (1). ObjState+1 = person state (0-4).
; NOTE: In state 4 he and the items are drawn, and he is checked for collisions, only on even frames.
; ----------------------------------------------------------------------------
Person_UpdateLifeOrMoney_Full:
    ; In state 4, skip drawing on odd frames (the person and the items flicker).
    LDA ObjState+1
    CMP #$04
    BNE @Draw
    LDA FrameCounter
    AND #$01
    BNE :+

@Draw:
    JSR Person_DrawAndCheckCollisions
    JSR Person_DrawLifeOrMoneyItems
:
    LDA ObjState+1
    JSR Util_TableJump

; TABLE: states of the life-or-money man; index = ObjState+1; value = routine address.
Person_UpdateLifeOrMoney_JumpTable:
    .ADDR Person_LifeOrMoneyState0ShowCosts          ; 0: show the costs, set the text delay
    .ADDR Person_UpdateState_Textbox                         ; 1: type the text, then unhalt Link
    .ADDR Person_LifeOrMoneyState2Choose          ; 2: wait for Link to choose and pay
    .ADDR Person_UpdateState_BlankPriceRow     ; 3: blank text line 3 and the costs row
    .ADDR Person_State4DelayAndQuit   ; 4: flicker, then remove the person

; TABLE: X of the two items offered (Y is $98); index = 0 heart container, 1 rupee; value = pixels.
Person_LifeOrMoneyItemXs:
    .BYTE $58, $98

; TABLE: item ids drawn for the two choices; index = 0, 1; value = item id.
Person_LifeOrMoneyItemIds:
    .BYTE $1A, $18              ; $1A heart container, $18 rupee

; ----------------------------------------------------------------------------
; Draws the heart container at ($58, $98) and the rupee at ($98, $98) through object slot $13.
; OUT:  ObjX+$13, ObjY+$13 changed; X = $FF.
; ----------------------------------------------------------------------------
Person_DrawLifeOrMoneyItems:
    ; Loop over each item to draw, from 1 to 0.
    LDX #$01

@LoopItem:
    TXA                         ; Save loop index.
    PHA

    ; Look up and set the X coordinate of the item.
    LDA Person_LifeOrMoneyItemXs, X
    STA ObjX+19

    ; Set Y=$98, and look up the item id.
    LDA #$98
    STA ObjY+19
    LDA Person_LifeOrMoneyItemIds, X

    ; Switch to the room item object slot.
    LDX #$13
    JSR Item_DrawObject
    PLA                         ; Restore loop index.
    TAX
    DEX
    BPL @LoopItem
    RTS

; ----------------------------------------------------------------------------
; Life-or-money state 0: shows the costs "-1" and "-50" and waits $0A frames before the text starts.
; OUT:  ObjTimer+1 = $0A; TileBufSelector = $76; ObjState+1 = 1.
; ----------------------------------------------------------------------------
Person_LifeOrMoneyState0ShowCosts:
    ; Delay in the next state $A frames before showing the first character.
    LDA #$0A
    STA ObjTimer+1
    LDA #$76                    ; Transfer selector: the "-1   -50" cost text.
    JMP Person_CueTransferBufAndAdvanceState

; ----------------------------------------------------------------------------
; Life-or-money state 2: takes a heart container or 50 rupees when Link stands on one of the items.
; OUT:  on payment: HeartValues (and HeartPartial) or InvRupeesToSubtract changed, ShutterTrigger = 1,
;       the room's item-taken flag set, ObjTimer+1 = $40, textbox blanked, ObjState+1 = 3.
; NOTE: Link's X must equal the item's X ($58 or $98) and his Y must be within 5 pixels of $98.
; NOTE: With 3 or fewer heart containers the "life" choice keeps the containers and empties all
;       hearts (full hearts = 0, HeartPartial = 0). This routine does not start Link's death.
; NOTE: The rupee choice needs InvRupees >= 50; below that, standing on the rupee does nothing.
;       InvRupeesToSubtract is not tested, so rupees still being counted off count as owned.
; ----------------------------------------------------------------------------
Person_LifeOrMoneyState2Choose:
    ; Loop over each item to check, from 1 (rupee) to 0 (heart container).
    LDX #$01

@LoopItem:
    ; If Link's X doesn't match the item's, then go loop again.
    LDA ObjX
    CMP Person_LifeOrMoneyItemXs, X
    BNE @NextItem

    ; If the vertical distance between the item and Link < 6, then
    ; go handle paying appropriately.
    LDA ObjY
    SEC
    SBC #$98
    JSR Math_Abs
    CMP #$06
    BCC @Pay

@NextItem:
    DEX
    BPL @LoopItem

@Exit:
    RTS

@Pay:
    ; Index 0: Link chose the heart container. Go handle it.
    CPX #$00
    BEQ @PayWithHeartContainer

    ; Link chose the money.
    ; But if he has fewer than 50 ($32) rupees, then return.
    LDA #$32
    CMP InvRupees
    BEQ @PayWithMoney
    BCS @Exit

@PayWithMoney:
    ; Post 50 rupees more to subtract (A is still $32), and go flag this offer taken.
    CLC
    ADC InvRupeesToSubtract
    STA InvRupeesToSubtract
    JMP @Paid

@PayWithHeartContainer:
    ; Link chose the heart container.
    ;
    ; HeartValues: high nibble = heart containers - 1, low nibble = full hearts.
    ; If the high nibble < 3 (fewer than 4 heart containers), then
    ; - leave the number of heart containers alone
    ; - set the full hearts and HeartPartial to 0 (no health left)
    ;
    ; Then go flag this offer taken.
    LDA HeartValues
    AND #$F0
    CMP #$30
    BCS @Reduce
    STA HeartValues
    LDA #$00
    STA HeartPartial
    JMP @Paid

@Reduce:
    ; Store the high nibble less one heart container in [00].
    LDA HeartValues
    PHA
    AND #$F0
    SEC
    SBC #$10
    STA $00

    ; Take one from the full hearts (low nibble).
    PLA
    AND #$0F
    SEC
    SBC #$01

    ; If the result is negative, then make the full hearts 0.
    BPL :+
    LDA #$00
:
    ; Combine the heart containers and full hearts, and store it.
    ORA $00
    STA HeartValues

@Paid:
    LDA #$08                    ; "key taken" sound effect
    STA Tune0Request
    LDA #$01                    ; Open the room's shutter doors.
    STA ShutterTrigger
    JMP Person_FlagTakenAndBlankText
