; ============================================================================
; FILE:     bank4/gohma.asm              BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 8193-8503
; SUBJECTS: (none)
; TOPIC:    Gohma: eye states, arrow in the eye
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0018 UpdateGohma                  code
;   L0189 GohmaLegOffsetsX             data
;   L0197 Gohma_AnimateAndDraw         code  Params:
;   L0219 Gohma_DrawLegsOneSide        code  Params:
;   L0253 Gohma_SetSpriteAttributes    code  Returns:
;   L0259 Gohma_CheckCollisions        code
;   L0291 Gohma_HandleWeaponCollision  code
; ============================================================================

UpdateGohma:
    ; If flagged not to continue straight, then choose a random facing direction.
    LDA Gohma_ObjGoStraight, X
    BNE @Move

    ; If Random:
    ;   >= $B0: right
    ;   >= $60: left
    ;   Else:   down
    LDA #$01
    LDY Random, X
    CPY #$B0
    BCS @ChangeDir
    ASL
    CPY #$60
    BCS @ChangeDir
    ASL

@ChangeDir:
    STA ObjDir, X

    ; Now we can go straight without changing direction for some time.
    INC Gohma_ObjGoStraight, X

    ; Go animate the eye.
    JMP @AnimateEye

@Move:
    ; Add $80 to the movement accumulator.
    LDA Gohma_ObjMoveAccum, X
    CLC
    ADC #$80
    STA Gohma_ObjMoveAccum, X

    ; If movement accumulator didn't overflow, then go animate the eye.
    BCC @AnimateEye

    ; Increase the distance traveled, and move 1.
    INC Gohma_ObjDistTraveled, X

    ; Set a direction mask in [02], starting with right (1).
    LDA #$01
    STA $02

    ; If Gohma's facing direction has a right component, then add 1 to X.
    LDA ObjDir, X
    BIT $02
    BEQ :+
    INC ObjX, X
:
    ; If Gohma's facing direction has a left component, then subtract 1 from X.
    ASL $02
    BIT $02
    BEQ :+
    DEC ObjX, X
:
    ; If Gohma's facing direction has a down component, then add 1 to Y.
    ASL $02
    BIT $02
    BEQ :+
    INC ObjY, X
:
    ; If Gohma's facing direction has an up component, then subtract 1 from Y.
    ASL $02
    BIT $02
    BEQ :+
    DEC ObjY, X
:
    ; If Gohma has not traveled $20 pixels, then go animate the eye.
    LDA Gohma_ObjDistTraveled, X
    CMP #$20
    BNE @AnimateEye

    ; Gohma has traveled $20 pixels. Reset the distance traveled.
    ; Reverse the facing direction, and increase the number of sprints.
    LDA #$00
    STA Gohma_ObjDistTraveled, X
    JSR ReverseObjDir8
    LDA Gohma_ObjSprints, X
    INC Gohma_ObjSprints, X

    ; If the previous number of sprints was odd, then flag that Gohma
    ; should randomly change direction.
    LSR
    BCC @AnimateEye
    LDA #$00
    STA Gohma_ObjGoStraight, X

@AnimateEye:
    ; Animate the eye.
    ;
    ; First, if the counter for when to open the eye next = 0, then
    ; set it to a random value at least $C0; and set the timer for
    ; keeping the eye open to $80.
    ;
    ; Note that $C0 is not a number of frames, because it's changed
    ; every other frame.
    LDA Gohma_ObjNextOpenEyeCounter, X
    BNE @DecNextOpenEyeCounter
    LDA #$80
    STA Gohma_ObjEyeOpenTimer, X
    LDA #$C0
    ORA Random, X
    STA Gohma_ObjNextOpenEyeCounter, X

@DecNextOpenEyeCounter:
    ; Decrement the next-open-eye counter.
    LDA FrameCounter
    LSR
    BCC :+
    DEC Gohma_ObjNextOpenEyeCounter, X
:
    ; If open eye timer = 0, then the eye is closed.
    ; Go animate it differently.
    LDA Gohma_ObjEyeOpenTimer, X
    BEQ @AnimateClosedEye

    ; Else decrement the open eye timer.
    DEC Gohma_ObjEyeOpenTimer, X

    ; If open eye timer >= $70 and < $10, then the eye is fully open.
    ; So, set its frame image to 2. Else use 3 for a half open eye.
    LDY #$02
    CMP #$70
    BCS :+
    CMP #$10
    BCC :+
    INY
:
    TYA

@SetEyeStateAndShoot:
    STA Gohma_ObjEyeState, X

@ShootAnimateCollide:
    ; Decrement the shoot timer.
    DEC Gohma_ObjShootTimer, X

    ; If it became 0, then set it to $41, and shoot fireball $56.
    BNE @AnimateAndCheckCollisions
    LDA #$41
    STA Gohma_ObjShootTimer, X
    LDA #$56
    JSR ShootFireball

@AnimateAndCheckCollisions:
    ; Pass the frame image for the eye.
    LDA Gohma_ObjEyeState, X
    JSR Gohma_AnimateAndDraw
    JMP Gohma_CheckCollisions

@AnimateClosedEye:
    ; Here the eye is closed. So animate it this way.
    ; Increment the animation counter, and if <> 8, then
    ; go shoot, draw, and check collisions.
    INC Gohma_ObjEyeAnimCounter, X
    LDA Gohma_ObjEyeAnimCounter, X
    CMP #$08
    BNE @ShootAnimateCollide

    ; Roll over the eye animation counter to 0.
    ; Clear bit 1 and invert bit 0 to switch between frame images 0 and 1.
    LDA #$00
    STA Gohma_ObjEyeAnimCounter, X
    LDA Gohma_ObjEyeState, X
    AND #$01
    EOR #$01

    ; Go set eye state/frame image, shoot, draw, and check collisions
    JMP @SetEyeStateAndShoot

GohmaLegOffsetsX:
    .BYTE $F0, $10

; Params:
; A: eye frame image
;
; Draw the eye with the frame image passed in.
;
Gohma_AnimateAndDraw:
    PHA
    JSR Anim_FetchObjPosForSpriteDescriptor
    JSR Gohma_SetSpriteAttributes
    PLA
    JSR DrawObjectMirrored
    LDA #$10
    JSR Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor

    ; Draw the right legs.
    LDY #$01
    JSR Gohma_DrawLegsOneSide

    ; Draw the left legs.
    LDY #$00

; Params:
; Y: side: 0 for left, 1 for right
;
; Add the horizontal offset for the current side to Gohma's X to
; store the sprite's X in [00].
;
Gohma_DrawLegsOneSide:
    LDA ObjX, X
    CLC
    ADC GohmaLegOffsetsX, Y
    STA $00

    ; The object's Y is the sprite's.
    LDA ObjY, X
    STA $01
    JSR Gohma_SetSpriteAttributes

    ; There are only two frame images for legs: left and right.
    ; There are two animation frames. They are animated as follows:
    ;
    ; Frame 0:
    ; ((leg index + frame number) AND 1) = 0 or 1
    ; horizontal flipping = frame number = 0
    ; 
    ; Frame 1:
    ; ((leg index + frame number) AND 1) = 1 or 0
    ; horizontal flipping = frame number = 1
    JSR Anim_SetObjHFlipForSpriteDescriptor
    TYA
    CLC
    ADC ObjAnimFrame, X
    AND #$01                    ; There are only two images.
    CLC
    ADC #$04                    ; Leg frame images start at 4.
    JMP DrawObjectNotMirrored

; Returns:
; [04]: left side sprite attributes
; [05]: right side sprite attributes
;
Gohma_SetSpriteAttributes:
    LDA ObjType, X
    SEC
    SBC #$32
    JMP Anim_SetSpriteDescriptorAttributes

Gohma_CheckCollisions:
    ; Save the boss's X coordinate.
    LDA ObjX, X
    PHA

    ; Subtract $10 from the X coordinate, so we start checking collisions
    ; on the left side.
    SEC
    SBC #$10
    STA ObjX, X

    ; Check object collisions in 5 parts, 8 pixels apart,
    ; from left to right, indexed by [0F].
    LDA #$05
    STA $0F                     ; Loop counter

@LoopSprite:
    JSR CheckMonsterCollisions

    ; Add 8 to the X coordinate for the next part.
    LDA ObjX, X
    CLC
    ADC #$08
    STA ObjX, X
    DEC $0F                     ; Loop counter
    BNE @LoopSprite

    ; Restore the boss's X coordinate.
    PLA
    STA ObjX, X
    RTS

Gohma_HandleWeaponCollision:
    ; If hit by an arrow, then set the arrow to spark.
    CPY #$12
    BNE @CheckHitEye
    LDA #$28
    STA a:ObjState, Y
    LDA #$04
    STA ObjAnimCounter, Y

@CheckHitEye:
    ; If it did not hit part 3 nor 4, then go play the parry sound effect.
    LDA $0F
    CMP #$03
    BEQ @CheckEyeOpen
    CMP #$04
    BNE @PlayParryTune

@CheckEyeOpen:
    ; If the state of the eye is not 3, then go parry.
    LDA Gohma_ObjEyeState, X
    CMP #$03
    BNE @PlayParryTune

    ; If the direction of the arrow is not up, then go parry.
    LDA a:ObjDir, Y
    CMP #$08
    BNE @PlayParryTune
    LDA #$02                    ; Boss hit sound effect
    STA SampleRequest

    ; Deal damage and cry out.
    JSR DealDamage
    JSR PlayBossDeathCryIfNeeded

@PlayParryTune:
    LDA #$01
    STA Tune0Request
    RTS
