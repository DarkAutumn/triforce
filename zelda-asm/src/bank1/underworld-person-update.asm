; ============================================================================
; FILE:     bank1/underworld-person-update.asm BANK: 1 (switchable, $8000-$BFFF)
; UPSTREAM: Z_01.asm 1170-1326
; SUBJECTS: Person
; TOPIC:    Update of the dungeon persons (types $4B-$50, $52): talk-only persons, the persons of levels
;           1, 2, 5, 7 including the bomb capacity offer (type $4F), and what happens when Link hits one.
; SEE ALSO: bank1/underworld-person-init.asm (init, text selectors); bank1/person-textbox.asm (state 1);
;           bank1/life-or-money.asm and bank1/grumble-goriya.asm (the other persons; the latter holds
;           Person_FlagTakenAndBlankText); bank4/statues.asm (fireballs once a person was hit);
;           bank1/monster-collisions.asm (Collision_CheckMonsterVsWeaponsAndLink); bank7/walker-move.asm (a person
;           blocks Link from walking up).
; KEY FACTS:
;   - Levels 3, 4, 6, 8, 9: persons only talk (3 states). Levels 1, 2, 5, 7: the 5-state machine of
;     Person_UpdateUnderworldComplex. Person_UpdateUnderworld_Full
;   - Hitting a person with a weapon does not kill him: the dying metastate is cleared again and
;     IsPersonFireballsEnabled is set, which makes fireballs fly until the room is left.
;     Person_CheckCollisions
;   - Type $4F (levels 1, 2, 5, 7) sells bomb capacity: a rupee is shown at ($78, $98); Link buys by
;     standing at X = $78 with Y within 5 pixels of $98 while InvRupees >= 100.
;     Person_ComplexState2SellBombCapacity
;   - The purchase posts 100 rupees to subtract, adds 4 to InvMaxBombs and refills InvBombs to the
;     new maximum, and sets the room's item-taken flag (the man never returns).
;     Person_ComplexState2SellBombCapacity
;   - The purchase starts a $40-frame timer; the man flickers until it ends, then his slot is freed.
;     Person_State4DelayAndQuit
; CONTENTS: (generated; L = line in this file)
;   L0044 Person_UpdateUnderworld_Full              code  Updates a dungeon person (types $4B-$50, $52): talk-only in levels 3, 4, 6, 8, 9, else complex.
;   L0064 Person_UpdateUnderworldTalker_JumpTable   data  TABLE: states of a talk-only dungeon person; index = ObjState+1; value = routine address.
;   L0072 Person_TalkerState0ResetText              code  Talk-only person state 0: restarts the text at its first character and goes to state 1.
;   L0078 Person_TalkerState2DoNothing              code  Talk-only person state 2: does nothing (the person stays); also the return of the routine above.
;   L0089 Person_CheckCollisions                    code  Runs weapon and Link collisions for the person; a hit enables fireballs instead of killing him.
;   L0107 Person_UpdateUnderworldComplex            code  Updates a dungeon person of levels 1, 2, 5, 7: draws him (and type $4F's rupee), runs his state.
;   L0136 Person_UpdateUnderworldComplex_JumpTable  data  TABLE: states of a dungeon person in levels 1, 2, 5, 7; index = ObjState+1; value = routine address.
;   L0147 Person_ComplexState0Begin                 code  Complex person state 0: type $4F shows its "-100" price; waits $0A frames before the text starts.
;   L0171 Person_ComplexState2SellBombCapacity      code  Complex person state 2: type $4F sells 4 more bomb capacity for 100 rupees at its rupee.
;   L0225 Person_State4DelayAndQuit                 code  Person state 4 (complex persons, life-or-money): frees the person's slot once ObjTimer+1 reaches 0.
;   L0237 Person_DrawAndCheckCollisions             code  Checks the person's collisions (see Person_CheckCollisions), then draws him mirrored.
; ============================================================================

; ----------------------------------------------------------------------------
; Updates a dungeon person (types $4B-$50, $52): talk-only in levels 3, 4, 6, 8, 9, else complex.
; IN:   X = object slot (1). ObjState+1 = person state.
; ----------------------------------------------------------------------------
Person_UpdateUnderworld_Full:
    ; Levels below 3, level 5 and level 7 (in a dungeon: 1, 2, 5, 7) have
    ; persons that do more than talk.
    LDA CurLevel
    CMP #$03
    BCC :+
    CMP #$05
    BEQ :+
    CMP #$07
    BNE @TalkOnly
:
    JMP Person_UpdateUnderworldComplex

@TalkOnly:
    ; Levels 3, 4, 6, 8, 9: draw, check collisions, and show the text.
    JSR Person_DrawAndCheckCollisions
    LDA ObjState+1
    JSR Util_TableJump

; TABLE: states of a talk-only dungeon person; index = ObjState+1; value = routine address.
Person_UpdateUnderworldTalker_JumpTable:
    .ADDR Person_TalkerState0ResetText     ; 0: restart the text
    .ADDR Person_UpdateState_Textbox             ; 1: type the text, then unhalt Link
    .ADDR Person_TalkerState2DoNothing           ; 2: stay

; ----------------------------------------------------------------------------
; Talk-only person state 0: restarts the text at its first character and goes to state 1.
; ----------------------------------------------------------------------------
Person_TalkerState0ResetText:
    LDA #$00
    STA PersonTextIndex
    INC ObjState+1

; Talk-only person state 2: does nothing (the person stays); also the return of the routine above.
Person_TalkerState2DoNothing:
    RTS

; ----------------------------------------------------------------------------
; Runs weapon and Link collisions for the person; a hit enables fireballs instead of killing him.
; IN:   X = object slot (1).
; OUT:  IsPersonFireballsEnabled nonzero if the person's metastate became nonzero (he was "killed");
;       ObjMetastate+1 = 0 again.
; NOTE: IsPersonFireballsEnabled is never cleared here, and no code writes 0 to it by name: $04CC lies
;       in the $0300-$051F range that Room_EnterCommon zeroes (Util_ClearRam0300UpTo) on room entry.
; ----------------------------------------------------------------------------
Person_CheckCollisions:
    JSR Collision_CheckMonsterVsWeaponsAndLink

    ; A nonzero metastate means a weapon of Link's killed the person (his HP is 0).
    ; Turn on the fireballs and clear the metastate, so that he does not die.
    LDA ObjMetastate+1
    BEQ :+
    STA IsPersonFireballsEnabled
    LDA #$00
    STA ObjMetastate+1
:
    RTS

; ----------------------------------------------------------------------------
; Updates a dungeon person of levels 1, 2, 5, 7: draws him (and type $4F's rupee), runs his state.
; IN:   X = object slot (1). ObjState+1 = person state (0-4).
; NOTE: In state 4 the person is drawn, and checked for collisions, only on even frames.
; ----------------------------------------------------------------------------
Person_UpdateUnderworldComplex:
    ; In state 4, skip drawing on odd frames (the person flickers).
    LDA ObjState+1
    CMP #$04
    BNE :+
    LDA FrameCounter
    AND #$01
    BNE @HandleState
:
    JSR Person_DrawAndCheckCollisions

    ; If this is type $4F (bomb capacity offer), then show a rupee (item $18)
    ; at ($78, $98) using the room item object slot $13.
    LDA ObjType+1
    CMP #$4F
    BNE @HandleState
    LDA #$78
    STA ObjX+19
    LDA #$98
    STA ObjY+19
    LDA #$18
    LDX #$13
    JSR Item_DrawObject

@HandleState:
    LDA ObjState+1
    JSR Util_TableJump

; TABLE: states of a dungeon person in levels 1, 2, 5, 7; index = ObjState+1; value = routine address.
Person_UpdateUnderworldComplex_JumpTable:
    .ADDR Person_ComplexState0Begin          ; 0: show the price, set the text delay
    .ADDR Person_UpdateState_Textbox                         ; 1: type the text, then unhalt Link
    .ADDR Person_ComplexState2SellBombCapacity      ; 2: wait for Link to buy (type $4F only)
    .ADDR Person_UpdateState_BlankPriceRow     ; 3: blank text line 3 and the price row
    .ADDR Person_State4DelayAndQuit   ; 4: flicker, then remove the person

; ----------------------------------------------------------------------------
; Complex person state 0: type $4F shows its "-100" price; waits $0A frames before the text starts.
; OUT:  ObjTimer+1 = $0A; ObjState+1 = 1; for type $4F TileBufSelector = $6C.
; ----------------------------------------------------------------------------
Person_ComplexState0Begin:
    ; If this is type $4F (bomb capacity offer), then show the price of the offer.
    LDA ObjType+1
    CMP #$4F
    BNE :+
    LDA #$6C                    ; Transfer selector: the "-100" price text.
    STA TileBufSelector
:
    ; Set a timer of $A frames before showing characters,
    ; and go to state 1.
    LDA #$0A
    STA ObjTimer+1
    INC ObjState+1
    RTS

; ----------------------------------------------------------------------------
; Complex person state 2: type $4F sells 4 more bomb capacity for 100 rupees at its rupee.
; OUT:  on purchase: InvRupeesToSubtract + 100, InvMaxBombs + 4, InvBombs = InvMaxBombs, the room's
;       item-taken flag set, ObjTimer+1 = $40, textbox blanked, ObjState+1 = 3.
; NOTE: Link must be at X = $78 exactly and within 5 pixels of Y = $98, with InvRupees >= 100.
; NOTE: Any other person type stays in this state for good (it returns at once).
; NOTE: InvRupeesToSubtract is not tested: only InvRupees, which pending debits have not yet lowered.
; NOTE: InvMaxBombs is not capped here: the 8-bit add wraps (the room flag allows one purchase per room).
; ----------------------------------------------------------------------------
Person_ComplexState2SellBombCapacity:
    ; If this is not type $4F, then return.
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
    JSR Math_Abs
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
    ; Post 100 rupees more to subtract (A is still $64).
    CLC
    ADC InvRupeesToSubtract
    STA InvRupeesToSubtract

    ; Play the "key taken" sound effect.
    LDA #$08
    STA Tune0Request

    ; Increase max bombs by 4, set the amount on hand to the max.
    LDA InvMaxBombs
    CLC
    ADC #$04
    STA InvMaxBombs
    STA InvBombs

    ; Go set the room's item-taken flag, blank the text and go to state 3.
    JMP Person_FlagTakenAndBlankText

; ----------------------------------------------------------------------------
; Person state 4 (complex persons, life-or-money): frees the person's slot once ObjTimer+1 reaches 0.
; NOTE: The timer was set to $40 frames when the offer was taken; the fires in slots 2 and 3 stay.
; ----------------------------------------------------------------------------
Person_State4DelayAndQuit:
    ; If the object timer has expired, then set the type to 0 (A = 0): the slot is empty.
    LDA ObjTimer+1
    BNE :+
    STA ObjType+1
:
    RTS

; ----------------------------------------------------------------------------
; Checks the person's collisions (see Person_CheckCollisions), then draws him mirrored.
; IN:   X = object slot (1).
; ----------------------------------------------------------------------------
Person_DrawAndCheckCollisions:
    JSR Person_CheckCollisions
    JSR Anim_FetchObjPosForSpriteDescriptor
    JMP Anim_DrawObjectMirrored
