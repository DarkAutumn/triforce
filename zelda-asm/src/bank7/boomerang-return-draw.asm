; ============================================================================
; FILE:     bank7/boomerang-return-draw.asm BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 4092-4360
; SUBJECTS: (none)
; TOPIC:    Boomerang slow/return states, drawing
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0017 CheckState30                       code
;   L0173 AnimateBoomerangAndCheckCollision  code
;   L0191 DrawBoomerangAndCheckCollision     code
;   L0206 CalcBoomerangFrame                 code
;   L0255 OffsetAndDrawArrow                 code  Params:
;   L0264 L_DrawArrowOrBoomerang             code
; ============================================================================

CheckState30:
    ; If major state <> $30, go check other states.
    CMP #$30
    BNE @CheckOtherStates

    ; State $3x. Slow down.
    ;
    ; Reset grid offset for movement.
    LDA #$00
    STA ObjGridOffset, X

    ; Go slow at q-speed fraction $40 (1 pixel a frame).
    LDA #$40
    STA ObjQSpeedFrac, X

    ; If facing left and X < 2, then go change state to $40. The
    ; boomerang is too close to the left edge of the screen.
    ; Another move might make it wrap around.
    LDA ObjDir, X
    STA $0F
    AND #$02
    BEQ @MoveBoomerang
    LDA ObjX, X
    CMP #$02
    BCC @SetState40

@MoveBoomerang:
    ; Move the boomerang.
    JSR MoveObject

    ; Decrement timer ObjMovingLimit. Once it reaches 0, change
    ; state to $40.
    DEC ObjMovingLimit, X
    BNE @AnimateBoomerang

@SetState40:
    ; Set state $40, and a time to move of $20 frames.
    ;
    ; In major state $40, ObjMovingLimit is a timer to change the state to $50.
    ;
    ; Also, animate, draw, and handle any collision (with Link, if a monster's boomerang).
    LDA #$20
    STA ObjMovingLimit, X
    LDA #$40
    STA ObjState, X

@AnimateBoomerang:
    JMP AnimateBoomerangAndCheckCollision

@CheckOtherStates:
    ; State $40 or $50. Return to thrower: $40 slow, $50 fast.
    ;
    ; The boomerang will now return. Reset the grid offset.
    LDA #$00
    STA ObjGridOffset, X

    ; Get the horizontal and vertical directions and distances to the thrower.
    CPX #$0D
    BCS :+
    LDA ObjRefId, X             ; Monster thrower object index
:
    JSR GetDirectionsAndDistancesToTarget

    ; If the boomerang hasn't reached the thrower ([00] < 2),
    ; then go move towards the thrower, draw, and check collisions.
    LDA $00
    CMP #$02
    BNE @MoveTowardThrower

    ; The boomerang has reached the thrower.
    ;
    ; Reset the distance to move.
    ; If this is a monster's boomerang, go handle the monster catching it.
    LDA #$00
    STA ObjMovingLimit, X
    CPX #$0D
    BCC @CatchBoomerang

    ; Link has caught the boomerang.
    ;
    ; Combine Link's state with $20 for having caught the boomerang.
    ;
    ; If Link catches it some time after throwing it,
    ; his state will become $20, and he will show the catch animation.
    ;
    ; But if he catches it right after throwing it, the state becomes
    ; $30 ($10 OR $20), which is the end state of using an item.
    LDA ObjState
    ORA #$20
    STA ObjState

    ; Make to set Link's animation counter to 1, so that he picks
    ; up the new state.
    LDA #$01
    STA ObjAnimCounter

    ; Deactivate the boomerang.
    LDY #$0F
    LDA #$00
    STA a:ObjState, Y
    RTS

@CatchBoomerang:
    ; A monster caught the boomerang that it threw.
    ;
    ; Set a timer for the thrower based on a random value:
    ; < $30: $30
    ; < $70: $50
    ; Else:  $70
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

    ; Reset the thrower's state to make it idle or restart its state machine.
    ; Destroy the monster's boomerang.
    LDA #$00
    STA a:ObjState, Y
    JMP DestroyCountedMonsterShot

@MoveTowardThrower:
    ; Move towards the thrower.
    ;
    ; Middle speed index 4 goes equally fast in X and Y axes.
    LDY #$04
    JSR _CalcDiagonalSpeedIndex

    ; Move vertically towards the thrower.
    LDA BoomerangQSpeedFracsY, Y
    JSR SetBoomerangSpeed
    LDA $0A                     ; [0A] vertical direction
    STA $0F
    STA ObjDir, X
    TYA                         ; Save speed index.
    PHA
    JSR MoveObject

    ; Move horizontally towards the thrower.
    PLA
    TAY                         ; Restore speed index.
    LDA BoomerangQSpeedFracsX, Y
    JSR SetBoomerangSpeed
    LDA $0B                     ; [0B] horizontal direction
    STA $0F
    STA ObjDir, X
    JSR MoveObject

AnimateBoomerangAndCheckCollision:
    ; Decrement animation counter; and when it reaches 0:
    ; 1. arm it again for 2 frames
    ; 2. advance the minor state within a cycle of 8
    ; 3. see if it's time to play the sound effect, if the boomerang belongs to Link
    DEC ObjAnimCounter, X
    BNE DrawBoomerangAndCheckCollision
    LDA #$02
    STA ObjAnimCounter, X
    INC ObjState, X
    LDA ObjState, X
    AND #$77
    STA ObjState, X
    CPX #$0D
    BCC CalcBoomerangFrame
    LDY #$02                    ; Boomerang effect
    JSR PlayBoomerangSfx

DrawBoomerangAndCheckCollision:
    ; If the boomerang belongs to a monster, then check for
    ; collision with Link.
    ;
    ; If they collide, then set state $20 and animation counter 3.
    CPX #$0D
    BCS CalcBoomerangFrame
    JSR CheckLinkCollision
    LDA ShotCollidesWithLink
    BEQ CalcBoomerangFrame
    LDA #$03
    STA ObjAnimCounter, X
    LDA #$20
    STA ObjState, X

CalcBoomerangFrame:
    ; The minor state (low 3 bits of state) is used to
    ; set the right frame in [0C] for this point in the spinning cycle.
    ;
    ; Reset [00] and [01]. These are the offsets for boomerang's position.
    ; In other words, don't offset the coordinate below.
    LDA #$00
    STA $00
    LDA ObjState, X
    AND #$0F
    TAY
    LDA #$00
    STA $01
    LDA BoomerangFrameCycle, Y
    STA $0C

    ; Store the base sprite attribute for this point in the cycle in [04].
    TYA
    LDA BoomerangBaseSpriteAttrCycle, Y
    STA $04

    ; If base sprite attribute in [04] = 8, then leave [04] as is.
    ; Else add the item value of the magic boomerang.
    ;
    ; If not sparking, then the magic boomerang's item value (0 or 1)
    ; in the inventory will be added to the base sprite attribute.
    ; So, palette row 4 or 5 will be chosen.
    ;
    ; If sparking, then base sprite attribute 1 from element 8 is used.
    ; Adding the inventory value results in palette row 4 or 5.
    ;
    ; None of the values in the base sprite attributes array is 8.
    ; So, this branch will never be taken. Maybe this is left over from
    ; development.
    LDY #$00
    CMP #$08
    BEQ :+
    LDY InvMagicBoomerang
:
    TYA
    CLC
    ADC $04
    STA $04
    LDY #$1D                    ; Boomerang item slot
    JMP L_DrawArrowOrBoomerang

; Params:
; Y: reverse direction index that the weapon is facing
;
OffsetAndDrawArrow:
    ; [00] and [01] will hold sprite X and Y.
    ; Begin with the horizontal and vertical offsets.
    LDA RDirectionToOffsetsX, Y
    STA $00
    LDA RDirectionToOffsetsY, Y
    STA $01
    LDY #$02                    ; Arrow item slot

L_DrawArrowOrBoomerang:
    ; Add the weapon's true coordinates to offsets in [00] and [01].
    LDA ObjX, X
    CLC
    ADC $00
    STA $00
    LDA ObjY, X
    CLC
    ADC $01
    STA $01

    ; If state = $2x (spark), use palette row 5 to write sprites.
    ; Otherwise, use the sprite attributes we already calculated.
    LDA ObjState, X
    AND #$F0
    CMP #$20
    BNE @Draw
    LDA #$01
    JSR Anim_SetSpriteDescriptorAttributes

@Draw:
    JMP Anim_WriteItemSprites
