; ============================================================================
; FILE:     bank7/boomerang-return-draw.asm BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 4092-4360
; SUBJECTS: (none)
; TOPIC:    Second half of the boomerang update (Link's in slot $0F and a monster's): states $3x slow down,
;           $4x/$5x return to the thrower and the catch; the spin animation and drawing shared with arrows.
; SEE ALSO: bank7/boomerang-arrow-food.asm (UpdateArrowOrBoomerang: states $1x fly out and $2x spark, and
;           the frame, attribute and speed tables used here); bank5/wield-items.asm (WieldBoomerang: range
;           $31 or $FF, q-speed $C0); bank4/goriya.asm (a Goriya's throw: range $51, q-speed $A0);
;           bank1/shot-aim-boomerang-speed.asm (GetDirectionsAndDistancesToTarget, _CalcDiagonalSpeedIndex,
;           SetBoomerangSpeed); bank1/link-collision-harm.asm (CheckLinkCollision);
;           bank7/object-animation.asm (AnimateLinkObjState: how Link's catch state $20 runs out).
; KEY FACTS:
;   - State $3x (range reached): q-speed $40 (1 pixel per frame) for ObjMovingLimit = $10 frames, then
;     state $40. A boomerang moving left with X < 2 goes to $40 at once. CheckState30
;   - State $4x: flies at the thrower at half the table speed for $10 frames (ObjMovingLimit $20 is
;     decremented twice a frame), then $5x at full speed (2 pixels per frame along an axis, 1.5 + 1.5
;     on a diagonal). CheckState30@MoveTowardThrower, SetBoomerangSpeed
;   - A boomerang blocked at a room bound, or a monster's that touched Link, skips $3x and $4x: spark
;     $2x for 3 frames, then $50. Link's boomerang that overlaps a monster goes straight to $50, no
;     spark, from any state. HandleArrowOrBoomerangBlocked (bank7/boomerang-arrow-food.asm),
;     DrawBoomerangAndCheckCollision, CheckMonsterWeaponCollision (bank1/weapon-damage-death.asm)
;   - Caught when both X and Y are within 8 pixels of the thrower's. CheckState30@CheckOtherStates
;   - Link's catch: his state is ORed with $20 and his animation counter set to 1, slot $0F is freed.
;     While his state is nonzero A and B are ignored (Link_HandleInput). CheckState30@CheckOtherStates
;   - A monster's catch: thrower's state = 0 and its ObjTimer = $30, $50 or $70 frames chosen by
;     Random,X (< $30, < $70, else); the boomerang is destroyed. CheckState30@CatchBoomerang
;   - A monster's boomerang tests Link only on the frames its 2-frame animation counter does not expire
;     (every other frame), and can only collide in state $1x. AnimateBoomerangAndCheckCollision,
;     DrawBoomerangAndCheckCollision
;   - Every boomerang, a monster's too, is drawn with palette row 4 + InvMagicBoomerang; sparks ($2x) of
;     boomerangs and arrows use row 5. CalcBoomerangFrame, L_DrawArrowOrBoomerang
;   - Link's boomerang plays effect 2 when its animation advances, at most once per $0A frames.
;     AnimateBoomerangAndCheckCollision
; CONTENTS: (generated; L = line in this file)
;   L0052 CheckState30                       code  Boomerang states $3x (slow down) and $4x/$5x (return to the thrower and be caught).
;   L0199 AnimateBoomerangAndCheckCollision  code  Advances a boomerang's spin every 2 frames, then draws it (a monster's also tests Link first).
;   L0217 DrawBoomerangAndCheckCollision     code  Draws a boomerang; a monster's boomerang that touches Link first turns into a spark (state $20).
;   L0232 CalcBoomerangFrame                 code  Picks the boomerang's frame and sprite attributes for its spin step and draws it.
;   L0274 OffsetAndDrawArrow                 code  Draws an arrow, shifted so that the narrow sprite lines up with its flight path.
;   L0283 L_DrawArrowOrBoomerang             code  Draws an arrow or boomerang (item slot in Y) at its position plus [00], [01]; a spark in row 5.
; ============================================================================

; ----------------------------------------------------------------------------
; Boomerang states $3x (slow down) and $4x/$5x (return to the thrower and be caught).
; IN:   A = major state (ObjState,X AND $F0), not $1x or $2x; X = boomerang's object slot ($0F Link's,
;       below $0D a monster's); [00] = 0 (set by UpdateArrowOrBoomerang).
; OUT:  Falls into AnimateBoomerangAndCheckCollision unless the boomerang was caught.
; NOTE: $3x lasts $10 frames at 1 pixel per frame, $4x $10 frames at half speed, $5x until caught.
;       Any major state other than $30 is handled as a return state.
; ----------------------------------------------------------------------------
CheckState30:
    CMP #$30
    BNE @CheckOtherStates

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
    ; [0F] = direction for MoveObject, which moves along one axis only (it tests the bits in the
    ; order right, left, down, up), so a diagonal boomerang drifts horizontally in this state.
    LDA ObjDir, X
    STA $0F
    AND #$02
    BEQ @MoveBoomerang
    LDA ObjX, X
    CMP #$02
    BCC @SetState40

@MoveBoomerang:
    JSR MoveObject

    ; In state $3x ObjMovingLimit counts frames ($10, set when the range was reached).
    DEC ObjMovingLimit, X
    BNE @AnimateBoomerang

@SetState40:
    ; State $40 (slow return). ObjMovingLimit becomes the countdown to state $50: it starts at $20
    ; and SetBoomerangSpeed decrements it twice a frame, so $40 lasts $10 frames.
    LDA #$20
    STA ObjMovingLimit, X
    LDA #$40
    STA ObjState, X

@AnimateBoomerang:
    JMP AnimateBoomerangAndCheckCollision

@CheckOtherStates:
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
    JSR GetDirectionsAndDistancesToTarget

    ; [00] = 2 means within 8 pixels of the thrower in X and in Y (top-left corners compared):
    ; caught. Otherwise go move toward the thrower.
    LDA $00
    CMP #$02
    BNE @MoveTowardThrower

    ; Caught. A monster's boomerang is handled separately.
    LDA #$00
    STA ObjMovingLimit, X
    CPX #$0D
    BCC @CatchBoomerang

    ; Link caught it: OR his state with $20.
    ; - Idle (0) becomes $20, the catch pose; AnimateLinkObjState then runs it $20 -> $21 -> $31
    ;   -> 0, and A and B are ignored until it is 0 again (Link_HandleInput).
    ; - Caught while still in the throw (state $10 or $11) it becomes $30 or $31, the last state
    ;   of an item use, which AnimateLinkObjState clears next.
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

@CatchBoomerang:
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
    JMP DestroyCountedMonsterShot

@MoveTowardThrower:
    ; Choose speed index 0-8 from the two distances, starting at the middle index 4 (equal
    ; speed on both axes). Index 0 is all horizontal, 8 all vertical.
    LDY #$04
    JSR _CalcDiagonalSpeedIndex

    ; Move vertically toward the thrower. SetBoomerangSpeed halves the q-speed in state $4x and
    ; counts ObjMovingLimit down to the switch to state $50.
    LDA BoomerangQSpeedFracsY, Y
    JSR SetBoomerangSpeed
    LDA $0A                     ; [0A] vertical direction to the thrower (8 up, 4 down)
    STA $0F
    STA ObjDir, X
    TYA                         ; Save speed index.
    PHA
    JSR MoveObject

    ; Move horizontally toward the thrower. ObjDir is left holding the horizontal direction.
    PLA
    TAY                         ; Restore speed index.
    LDA BoomerangQSpeedFracsX, Y
    JSR SetBoomerangSpeed
    LDA $0B                     ; [0B] horizontal direction to the thrower (2 left, 1 right)
    STA $0F
    STA ObjDir, X
    JSR MoveObject

; ----------------------------------------------------------------------------
; Advances a boomerang's spin every 2 frames, then draws it (a monster's also tests Link first).
; IN:   X = boomerang's object slot.
; NOTE: On the frame the spin advances a monster's boomerang skips the collision test with Link,
;       so it can only touch him every other frame.
; ----------------------------------------------------------------------------
AnimateBoomerangAndCheckCollision:
    ; When the animation counter reaches 0: set it to 2 frames again, advance the minor state
    ; (low nibble, the spin step) within 0-7 (AND $77 also keeps bit 7 clear), and for Link's
    ; boomerang request the boomerang sound (PlayBoomerangSfx allows it once per $0A frames).
    DEC ObjAnimCounter, X
    BNE DrawBoomerangAndCheckCollision
    LDA #$02
    STA ObjAnimCounter, X
    INC ObjState, X
    LDA ObjState, X
    AND #$77
    STA ObjState, X
    CPX #$0D
    BCC CalcBoomerangFrame      ; A monster's boomerang: draw without testing Link this frame.
    LDY #$02                    ; Effect 2: boomerang
    JSR PlayBoomerangSfx

; Draws a boomerang; a monster's boomerang that touches Link first turns into a spark (state $20).
DrawBoomerangAndCheckCollision:
    ; Only a monster's boomerang (slot below $0D) tests Link. CheckLinkCollision harms Link or
    ; lets him parry, and reports a touch only while the boomerang is in state $1x.
    ; On a touch: state $20 (spark) for 3 frames, after which it returns in state $50.
    CPX #$0D
    BCS CalcBoomerangFrame
    JSR CheckLinkCollision
    LDA IsShotCollidingWithLink
    BEQ CalcBoomerangFrame
    LDA #$03
    STA ObjAnimCounter, X
    LDA #$20
    STA ObjState, X

; Picks the boomerang's frame and sprite attributes for its spin step and draws it.
CalcBoomerangFrame:
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
    LDA BoomerangFrameCycle, Y
    STA $0C

    ; [04] = base sprite attribute for that step (flip bits; 1 for the spark).
    TYA
    LDA BoomerangBaseSpriteAttrCycle, Y
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
    JMP L_DrawArrowOrBoomerang

; ----------------------------------------------------------------------------
; Draws an arrow, shifted so that the narrow sprite lines up with its flight path.
; IN:   X = arrow's object slot; Y = reverse direction index of the arrow (0 up, 1 down, 2 left,
;       3 right); [04], [05] = sprite attributes; [0C] = frame; [0F] = horizontal flip.
; ----------------------------------------------------------------------------
OffsetAndDrawArrow:
    ; [00], [01] = offsets: X - 4 when vertical, Y + 3 when horizontal.
    LDA RDirectionToOffsetsX, Y
    STA $00
    LDA RDirectionToOffsetsY, Y
    STA $01
    LDY #$02                    ; Item slot 2: arrow

; Draws an arrow or boomerang (item slot in Y) at its position plus [00], [01]; a spark in row 5.
L_DrawArrowOrBoomerang:
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
