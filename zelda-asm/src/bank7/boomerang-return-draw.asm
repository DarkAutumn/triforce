; ============================================================================
; FILE:     bank7/boomerang-return-draw.asm BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 4092-4360
; SUBJECTS: Arrow, Boomerang
; TOPIC:    Second half of the boomerang update (Link's in slot $0F and a monster's): states $3x slow down,
;           $4x/$5x return to the thrower and the catch; the spin animation and drawing shared with arrows.
; SEE ALSO: bank7/boomerang-arrow-food.asm (Boomerang_UpdateOrArrow: states $1x fly out and $2x spark, and
;           the frame, attribute and speed tables used here); bank5/wield-items.asm (Boomerang_Wield: range
;           $31 or $FF, q-speed $C0); bank4/goriya.asm (a Goriya's throw: range $51, q-speed $A0);
;           bank1/shot-aim-boomerang-speed.asm (Dir_GetDirsAndDistancesToTarget, Dir_CalcAimSpeedIndex,
;           Boomerang_SetReturnSpeed); bank1/link-collision-harm.asm (Link_CheckMonsterTouch);
;           bank7/object-animation.asm (Link_AdvanceItemUseState: how Link's catch state $20 runs out).
; KEY FACTS:
;   - State $3x (range reached): q-speed $40 (1 pixel per frame) for ObjMovingLimit = $10 frames, then
;     state $40. A boomerang moving left with X < 2 goes to $40 at once. Boomerang_UpdateSlowAndReturn
;   - State $4x: flies at the thrower at half the table speed for $10 frames (ObjMovingLimit $20 is
;     decremented twice a frame), then $5x at full speed (2 pixels per frame along an axis, 1.5 + 1.5
;     on a diagonal). Boomerang_UpdateSlowAndReturn@MoveTowardThrower, Boomerang_SetReturnSpeed
;   - A boomerang blocked at a room bound, or a monster's that touched Link, skips $3x and $4x: spark
;     $2x for 3 frames, then $50. Link's boomerang that overlaps a monster goes straight to $50, no
;     spark, from any state. Boomerang_HandleBlockedOrArrow (bank7/boomerang-arrow-food.asm),
;     Boomerang_DrawAndCheckCollision, Collision_CheckWeaponPointVsMonster (bank1/weapon-damage-death.asm)
;   - Caught when both X and Y are within 8 pixels of the thrower's. Boomerang_UpdateSlowAndReturn@CheckOtherStates
;   - Link's catch: his state is ORed with $20 and his animation counter set to 1, slot $0F is freed.
;     While his state is nonzero A and B are ignored (Link_HandleInput). Boomerang_UpdateSlowAndReturn@CheckOtherStates
;   - A monster's catch: thrower's state = 0 and its ObjTimer = $30, $50 or $70 frames chosen by
;     Random,X (< $30, < $70, else); the boomerang is destroyed. Boomerang_UpdateSlowAndReturn@CatchBoomerang
;   - A monster's boomerang tests Link only on the frames its 2-frame animation counter does not expire
;     (every other frame), and can only collide in state $1x. Boomerang_AnimateAndCheckCollision,
;     Boomerang_DrawAndCheckCollision
;   - Every boomerang, a monster's too, is drawn with palette row 4 + InvMagicBoomerang; sparks ($2x) of
;     boomerangs and arrows use row 5. Boomerang_DrawSpinFrame, Arrow_DrawArrowOrBoomerang
;   - Link's boomerang plays effect 2 when its animation advances, at most once per $0A frames.
;     Boomerang_AnimateAndCheckCollision
; CONTENTS: (generated; L = line in this file)
;   L0052 Boomerang_UpdateSlowAndReturn       code  Boomerang states $3x (slow down) and $4x/$5x (return to the thrower and be caught).
;   L0199 Boomerang_AnimateAndCheckCollision  code  Advances a boomerang's spin every 2 frames, then draws it (a monster's also tests Link first).
;   L0217 Boomerang_DrawAndCheckCollision     code  Draws a boomerang; a monster's boomerang that touches Link first turns into a spark (state $20).
;   L0232 Boomerang_DrawSpinFrame             code  Picks the boomerang's frame and sprite attributes for its spin step and draws it.
;   L0274 Arrow_OffsetAndDraw                 code  Draws an arrow, shifted so that the narrow sprite lines up with its flight path.
;   L0283 Arrow_DrawArrowOrBoomerang          code  Draws an arrow or boomerang (item slot in Y) at its position plus [00], [01]; a spark in row 5.
; ============================================================================

; ----------------------------------------------------------------------------
; Boomerang states $3x (slow down) and $4x/$5x (return to the thrower and be caught).
; IN:   A = major state (ObjState,X AND $F0), not $1x or $2x; X = boomerang's object slot ($0F Link's,
;       below $0D a monster's); [00] = 0 (set by Boomerang_UpdateOrArrow).
; OUT:  Falls into Boomerang_AnimateAndCheckCollision unless the boomerang was caught.
; NOTE: $3x lasts $10 frames at 1 pixel per frame, $4x $10 frames at half speed, $5x until caught.
;       Any major state other than $30 is handled as a return state.
; ----------------------------------------------------------------------------
Boomerang_UpdateSlowAndReturn:
    CMP #$30
    BNE @ReturnToThrower

    ; State $3x: the boomerang reached its range and slows down.
    ;
    ; Grid offset is the distance flown in state $1x; it is not needed any more.
    LDA #$00
    STA ObjGridOffset, X

    ; Q-speed $40 = 1 pixel per frame.
    LDA #$40
    STA ObjQSpeedFrac, X

    ; If the direction has the left bit and X < 2, go to state $40 without moving: one more
    ; pixel left could wrap X around to the right side of the screen.
    ; [0F] = direction for Obj_MoveByQSpeed, which moves along one axis only (it tests the bits in the
    ; order right, left, down, up), so a diagonal boomerang drifts horizontally in this state.
    LDA ObjDir, X
    STA $0F
    AND #$02
    BEQ @MoveBoomerang
    LDA ObjX, X
    CMP #$02
    BCC @SetState40

@MoveBoomerang:
    JSR Obj_MoveByQSpeed

    ; In state $3x ObjMovingLimit counts frames ($10, set when the range was reached).
    DEC ObjMovingLimit, X
    BNE @AnimateBoomerang

@SetState40:
    ; State $40 (slow return). ObjMovingLimit becomes the countdown to state $50: it starts at $20
    ; and Boomerang_SetReturnSpeed decrements it twice a frame, so $40 lasts $10 frames.
    LDA #$20
    STA ObjMovingLimit, X
    LDA #$40
    STA ObjState, X

@AnimateBoomerang:
    JMP Boomerang_AnimateAndCheckCollision

@ReturnToThrower:
    ; State $4x or $5x: return to the thrower, $4x at half speed, $5x at full speed.
    LDA #$00
    STA ObjGridOffset, X

    ; A = target slot: 0 (Link; A is still 0) for Link's boomerang, else the monster that threw
    ; it (ObjRefId). Get the direction and distance to the target on each axis:
    ; [0B]/[03] horizontal, [0A]/[04] vertical; [00] counts the axes with distance <= 8.
    CPX #$0D
    BCS :+
    LDA ObjRefId, X             ; Thrower's object slot.
:
    JSR Dir_GetDirsAndDistancesToTarget

    ; [00] = 2 means within 8 pixels of the thrower in X and in Y (top-left corners compared):
    ; caught. Otherwise go move toward the thrower.
    LDA $00
    CMP #$02
    BNE @MoveTowardThrower

    ; Caught. A monster's boomerang is handled separately.
    LDA #$00
    STA ObjMovingLimit, X
    CPX #$0D
    BCC @MonsterCatches

    ; Link caught it: OR his state with $20.
    ; - Idle (0) becomes $20, the catch pose; Link_AdvanceItemUseState then runs it $20 -> $21 -> $31
    ;   -> 0, and A and B are ignored until it is 0 again (Link_HandleInput).
    ; - Caught while still in the throw (state $10 or $11) it becomes $30 or $31, the last state
    ;   of an item use, which Link_AdvanceItemUseState clears next.
    LDA ObjState
    ORA #$20
    STA ObjState

    ; Link's animation counter = 1, so that his state advances on his next update.
    LDA #$01
    STA ObjAnimCounter

    ; Free the boomerang slot ($0F): another boomerang or the food can be used again.
    LDY #$0F
    LDA #$00
    STA a:ObjState, Y
    RTS

@MonsterCatches:
    ; A monster caught the boomerang it threw.
    ;
    ; Set the thrower's ObjTimer from the boomerang slot's random byte:
    ;   < $30: $30 frames;  < $70: $50 frames;  else: $70 frames
    LDY #$30
    LDA Random, X
    CMP #$30
    BCC @SetThrowerTimer
    LDY #$50
    CMP #$70
    BCC @SetThrowerTimer
    LDY #$70

@SetThrowerTimer:
    TYA
    LDY ObjRefId, X
    STA a:ObjTimer, Y

    ; Thrower's state = 0 (it was $80, waiting for the boomerang), then destroy the boomerang
    ; and take it off the monster shot count.
    LDA #$00
    STA a:ObjState, Y
    JMP MonsterShot_DestroyCounted

@MoveTowardThrower:
    ; Choose speed index 0-8 from the two distances, starting at the middle index 4 (equal
    ; speed on both axes). Index 0 is all horizontal, 8 all vertical.
    LDY #$04
    JSR Dir_CalcAimSpeedIndex

    ; Move vertically toward the thrower. Boomerang_SetReturnSpeed halves the q-speed in state $4x and
    ; counts ObjMovingLimit down to the switch to state $50.
    LDA Boomerang_ReturnQSpeedsY, Y
    JSR Boomerang_SetReturnSpeed
    LDA $0A                     ; [0A] vertical direction to the thrower (8 up, 4 down)
    STA $0F
    STA ObjDir, X
    TYA                         ; Save speed index.
    PHA
    JSR Obj_MoveByQSpeed

    ; Move horizontally toward the thrower. ObjDir is left holding the horizontal direction.
    PLA
    TAY                         ; Restore speed index.
    LDA Boomerang_ReturnQSpeedsX, Y
    JSR Boomerang_SetReturnSpeed
    LDA $0B                     ; [0B] horizontal direction to the thrower (2 left, 1 right)
    STA $0F
    STA ObjDir, X
    JSR Obj_MoveByQSpeed

; ----------------------------------------------------------------------------
; Advances a boomerang's spin every 2 frames, then draws it (a monster's also tests Link first).
; IN:   X = boomerang's object slot.
; NOTE: On the frame the spin advances a monster's boomerang skips the collision test with Link,
;       so it can only touch him every other frame.
; ----------------------------------------------------------------------------
Boomerang_AnimateAndCheckCollision:
    ; When the animation counter reaches 0: set it to 2 frames again, advance the minor state
    ; (low nibble, the spin step) within 0-7 (AND $77 also keeps bit 7 clear), and for Link's
    ; boomerang request the boomerang sound (Boomerang_PlaySfx allows it once per $0A frames).
    DEC ObjAnimCounter, X
    BNE Boomerang_DrawAndCheckCollision
    LDA #$02
    STA ObjAnimCounter, X
    INC ObjState, X
    LDA ObjState, X
    AND #$77
    STA ObjState, X
    CPX #$0D
    BCC Boomerang_DrawSpinFrame      ; A monster's boomerang: draw without testing Link this frame.
    LDY #$02                    ; Effect 2: boomerang
    JSR Boomerang_PlaySfx

; Draws a boomerang; a monster's boomerang that touches Link first turns into a spark (state $20).
Boomerang_DrawAndCheckCollision:
    ; Only a monster's boomerang (slot below $0D) tests Link. Link_CheckMonsterTouch harms Link or
    ; lets him parry, and reports a touch only while the boomerang is in state $1x.
    ; On a touch: state $20 (spark) for 3 frames, after which it returns in state $50.
    CPX #$0D
    BCS Boomerang_DrawSpinFrame
    JSR Link_CheckMonsterTouch
    LDA IsShotCollidingWithLink
    BEQ Boomerang_DrawSpinFrame
    LDA #$03
    STA ObjAnimCounter, X
    LDA #$20
    STA ObjState, X

; Picks the boomerang's frame and sprite attributes for its spin step and draws it.
Boomerang_DrawSpinFrame:
    ; [00], [01] = 0: no offset from the object's position.
    ; Y = low nibble of state: spin step 0-7, or 8 for the spark (state $28).
    ; [0C] = frame image for that step.
    LDA #$00
    STA $00
    LDA ObjState, X
    AND #$0F
    TAY
    LDA #$00
    STA $01
    LDA Boomerang_FrameCycle, Y
    STA $0C

    ; [04] = base sprite attribute for that step (flip bits; 1 for the spark).
    TYA
    LDA Boomerang_SpriteAttrCycle, Y
    STA $04

    ; Add InvMagicBoomerang (0 or 1) to the attribute: palette row 4 for the wooden boomerang,
    ; row 5 once Link owns the magic one. A monster's boomerang goes through here as well and
    ; takes the same colour.
    ;
    ; The branch that would skip the addition compares the attribute with 8; no table value
    ; is 8, so it is never taken.
    LDY #$00
    CMP #$08
    BEQ :+
    LDY InvMagicBoomerang
:
    TYA
    CLC
    ADC $04
    STA $04
    LDY #$1D                    ; Item slot $1D: boomerang
    JMP Arrow_DrawArrowOrBoomerang

; ----------------------------------------------------------------------------
; Draws an arrow, shifted so that the narrow sprite lines up with its flight path.
; IN:   X = arrow's object slot; Y = reverse direction index of the arrow (0 up, 1 down, 2 left,
;       3 right); [04], [05] = sprite attributes; [0C] = frame; [0F] = horizontal flip.
; ----------------------------------------------------------------------------
Arrow_OffsetAndDraw:
    ; [00], [01] = offsets: X - 4 when vertical, Y + 3 when horizontal.
    LDA Arrow_SpriteOffsetsXByRDir, Y
    STA $00
    LDA Arrow_SpriteOffsetsYByRDir, Y
    STA $01
    LDY #$02                    ; Item slot 2: arrow

; Draws an arrow or boomerang (item slot in Y) at its position plus [00], [01]; a spark in row 5.
Arrow_DrawArrowOrBoomerang:
    ; [00], [01] = object position + offsets = sprite position.
    LDA ObjX, X
    CLC
    ADC $00
    STA $00
    LDA ObjY, X
    CLC
    ADC $01
    STA $01

    ; In state $2x (spark) both sprite attributes become 1: palette row 5, no flipping.
    ; Otherwise keep the attributes already in [04] and [05].
    LDA ObjState, X
    AND #$F0
    CMP #$20
    BNE @Draw
    LDA #$01
    JSR Anim_SetSpriteDescriptorAttributes

@Draw:
    JMP Anim_WriteItemSprites
