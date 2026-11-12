; ============================================================================
; FILE:     bank4/gohma.asm              BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 8193-8503
; SUBJECTS: Gohma
; TOPIC:    Gohma (object types $33, $34) per-frame update: out-and-back movement, the eye's open/closed
;           cycle, fireballs, drawing of eye and legs, five-part collision, and the rule that only an
;           arrow flying up into the fully open eye deals damage.
; SEE ALSO: bank4/gleeok-manhandla-gohma-inits.asm (Gohma_Init: start position, invincibility mask $FB,
;           shoot timer); bank1/weapon-damage-death.asm (Damage_HandleWeaponHit jumps to
;           Gohma_HandleWeaponCollision; Damage_DealToMonster); bank1/sword-bomb-arrow-collision.asm
;           (Arrow_CheckArrowOrRodVsMonster: arrow damage $20 wooden, $40 silver);
;           bank1/shove-knockback.asm (Shove_Begin: Gohma gets invincibility frames only from an eye hit);
;           bank4/monster-shots-fireballs.asm (Fireball_Shoot, Fireball_Update);
;           bank7/object-update-table.asm (Obj_TypeHpPairs: HP $60 for $33, $20 for $34).
; KEY FACTS:
;   - Damage needs all of: an arrow (the only damage type the mask $FB of Gohma_Init lets through),
;     collision part 3 or 4 (the parts at ObjX and ObjX - 8), eye state 3, and arrow direction up (8).
;     Anything else only plays the parry sound. Gohma_HandleWeaponCollision, Gohma_CheckCollisions
;   - Eye state 3 lasts $60 (96) frames inside each $80-frame opening: $11 frames at state 2, $60 at
;     state 3, $0F at state 2. Gohma_Update@AnimateEye
;   - Openings start every 2 * ($C0 | Random,X) frames = 384 to 510 frames, counted from the start
;     of the previous opening. Gohma_Update@AnimateEye, Gohma_Update@DecNextOpenEyeCounter
;   - Movement is 1 pixel every 2 frames. Gohma picks right (Random,X >= $B0), left (>= $60) or down
;     (else), goes $20 pixels, comes back $20 pixels, then picks again: it always returns to where it
;     started (Gohma_Init puts it at X $80, Y $70) and never goes above that. Gohma_Update
;   - Fireball type $56 (the shield cannot block it) every $41 (65) frames, whatever the eye does; the
;     first one on the first update frame if the timer was 0 before Gohma_Init incremented it. Only a
;     full object table (slots $01-$0B) stops it. Gohma_Update@ShootAnimateCollide
;   - Collision is tested in 5 parts 8 pixels apart from ObjX - $10 to ObjX + $10, left to right,
;     against Link and weapons alike. An arrow sparks on the first part it overlaps, so the eye is
;     reached only when the arrow's hit point X is ObjX - 1 to ObjX + $0E. Gohma_CheckCollisions
;   - An arrow on parts 3 or 4 with the eye at state 3 starts Gohma's invincibility timer ($10, about
;     32 frames; no weapon or Link test meanwhile) whatever the arrow's direction. Shove_Begin@Gohma
;   - A successful hit ends with the parry sound request too (the code falls through), replacing the
;     "harmed" sound request of Damage_DealToMonster. Gohma_HandleWeaponCollision
;   - Sprite palette row = object type - $32: 1 for type $33, 2 for type $34. Gohma_SetSpriteAttributes
; CONTENTS: (generated; L = line in this file)
;   L0063 Gohma_Update                 code  Updates Gohma: moves it out and back, runs the eye's open/closed cycle, shoots, draws, collides.
;   L0236 Gohma_LegOffsetsX            data  TABLE: X offset of Gohma's leg sprites from ObjX; index = side (0 left, 1 right); value = pixels.
;   L0245 Gohma_AnimateAndDraw         code  Draws Gohma: the eye at its position with the given frame image, then the legs on both sides.
;   L0265 Gohma_DrawLegsOneSide        code  Draws the legs on one side of Gohma, $10 pixels left or right of the eye.
;   L0301 Gohma_SetSpriteAttributes    code  Sets both sprite attribute bytes to Gohma's palette row: object type - $32 (1 for $33, 2 for $34).
;   L0322 Gohma_CheckCollisions        code  Checks Gohma's collisions with Link and his weapons in five parts, 8 pixels apart.
;   L0367 Gohma_HandleWeaponCollision  code  Decides what a weapon that got past Gohma's invincibility mask does: harm through the eye, or parry.
; ============================================================================

; ----------------------------------------------------------------------------
; Updates Gohma: moves it out and back, runs the eye's open/closed cycle, shoots, draws, collides.
; IN:   X = object slot
; NOTE: Movement: a new direction is picked when Gohma_ObjGoStraight is 0 (no move that frame):
;       Random,X >= $B0 right (80 in 256), >= $60 left (80 in 256), else down (96 in 256). Then 1
;       pixel every 2 frames for $20 pixels, reverse, $20 pixels back, and pick again. No room
;       boundary or tile test is made; the path stays within $20 pixels of the start position.
; NOTE: Eye: when Gohma_ObjNextOpenEyeCounter is 0 the eye opens for $80 frames and the counter is
;       reloaded with $C0 | Random,X ($C0-$FF). The counter drops by 1 on frames with an odd
;       FrameCounter, also while the eye is open. Eye state by open timer value before the frame's
;       decrement: $80-$70 -> 2, $6F-$10 -> 3 (the only vulnerable state), $0F-$01 -> 2. Closed, the
;       state alternates 1, 0 every 8 frames; it can stay 2 for up to 7 frames after closing.
; NOTE: Fireball: Gohma_ObjShootTimer is decremented every frame; at 0 it is reloaded with $41 and a
;       type $56 fireball is created (aimed at Link's position by Fireball_Update). Gohma_Init increments
;       the timer, so the first fireball comes on the first update frame if the slot's byte was 0.
; ----------------------------------------------------------------------------
Gohma_Update:
    ; If flagged not to continue straight, then choose a random facing direction.
    LDA Gohma_ObjGoStraight, X
    BNE @Move

    ; If Random,X:
    ;   >= $B0: right (1)
    ;   >= $60: left (2)
    ;   Else:   down (4)
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

    ; Now go straight (out and back) without choosing again.
    INC Gohma_ObjGoStraight, X

    ; No movement on this frame.
    JMP @AnimateEye

@Move:
    ; Add $80 to the position fraction: one pixel every second frame.
    LDA Gohma_ObjPosFrac, X
    CLC
    ADC #$80
    STA Gohma_ObjPosFrac, X

    ; If the fraction didn't overflow, then go animate the eye.
    BCC @AnimateEye

    ; Increase the distance traveled, and move 1 pixel.
    INC Gohma_ObjDistTraveled, X

    ; [02] is a direction mask, shifted left once per test, starting with right (1).
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
    ; (Up only occurs as the way back from a down sprint.)
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
    JSR Dir_ReverseObjDir8
    LDA Gohma_ObjSprints, X
    INC Gohma_ObjSprints, X

    ; If the number of sprints before this one was odd (this was the
    ; way back), then choose a new direction on the next frame.
    LSR
    BCC @AnimateEye
    LDA #$00
    STA Gohma_ObjGoStraight, X

@AnimateEye:
    ; If the counter for when to open the eye next = 0, then open it:
    ; set the open timer to $80 frames, and reload the counter with a
    ; random value of at least $C0.
    ;
    ; The counter is not in frames: it drops by 1 every other frame, so
    ; $C0-$FF means 384 to 510 frames until the next opening.
    LDA Gohma_ObjNextOpenEyeCounter, X
    BNE @DecNextOpenEyeCounter
    LDA #$80
    STA Gohma_ObjEyeOpenTimer, X
    LDA #$C0
    ORA Random, X
    STA Gohma_ObjNextOpenEyeCounter, X

@DecNextOpenEyeCounter:
    ; Decrement the next-open-eye counter on frames with an odd FrameCounter.
    LDA FrameCounter
    LSR
    BCC :+
    DEC Gohma_ObjNextOpenEyeCounter, X
:
    ; If open eye timer = 0, then the eye is closed.
    ; Go animate it differently.
    LDA Gohma_ObjEyeOpenTimer, X
    BEQ @AnimateClosedEye

    ; Else decrement the open eye timer. A keeps the value before the decrement.
    DEC Gohma_ObjEyeOpenTimer, X

    ; If that value >= $70 or < $10, the eye is opening or closing: eye
    ; state 2. In between ($6F down to $10, $60 frames) it is fully
    ; open: eye state 3, the only state in which an arrow harms Gohma.
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
    ; Decrement the shoot timer, every frame.
    DEC Gohma_ObjShootTimer, X

    ; If it became 0, then set it to $41 frames, and shoot fireball $56
    ; (the shield cannot block it).
    BNE @AnimateAndCheckCollisions
    LDA #$41
    STA Gohma_ObjShootTimer, X
    LDA #$56
    JSR Fireball_Shoot

@AnimateAndCheckCollisions:
    ; Pass the frame image for the eye.
    LDA Gohma_ObjEyeState, X
    JSR Gohma_AnimateAndDraw
    JMP Gohma_CheckCollisions

@AnimateClosedEye:
    ; Here the eye is closed.
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

; TABLE: X offset of Gohma's leg sprites from ObjX; index = side (0 left, 1 right); value = pixels.
Gohma_LegOffsetsX:
    .BYTE $F0, $10              ; -$10 left legs, +$10 right legs

; ----------------------------------------------------------------------------
; Draws Gohma: the eye at its position with the given frame image, then the legs on both sides.
; IN:   A = eye frame image (Gohma_ObjEyeState: 0/1 closed, 2 half open, 3 fully open); X = object slot
; NOTE: Advances the leg animation: ObjAnimCounter,X is reloaded with $10 frames per leg frame.
;       Falls through into Gohma_DrawLegsOneSide for the left legs.
; ----------------------------------------------------------------------------
Gohma_AnimateAndDraw:
    PHA
    JSR Anim_FetchObjPosForSpriteDescriptor
    JSR Gohma_SetSpriteAttributes
    PLA
    JSR Anim_DrawObjectMirrored
    LDA #$10                    ; Animation counter reload: $10 frames per animation frame.
    JSR Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor

    ; Draw the right legs.
    LDY #$01
    JSR Gohma_DrawLegsOneSide

    ; Draw the left legs.
    LDY #$00

; ----------------------------------------------------------------------------
; Draws the legs on one side of Gohma, $10 pixels left or right of the eye.
; IN:   Y = side: 0 left, 1 right; X = object slot
; ----------------------------------------------------------------------------
Gohma_DrawLegsOneSide:
    ; [00] := sprite X = Gohma's X + the offset for this side.
    LDA ObjX, X
    CLC
    ADC Gohma_LegOffsetsX, Y
    STA $00

    ; The object's Y is the sprite's.
    LDA ObjY, X
    STA $01
    JSR Gohma_SetSpriteAttributes

    ; There are only two frame images for legs (4 and 5), and two
    ; animation frames. They are animated as follows:
    ;
    ; Animation frame 0:
    ;   image = 4 + ((side + 0) AND 1): 4 left, 5 right
    ;   not flipped horizontally
    ;
    ; Animation frame 1:
    ;   image = 4 + ((side + 1) AND 1): 5 left, 4 right
    ;   flipped horizontally
    JSR Anim_SetObjHFlipForSpriteDescriptor
    TYA
    CLC
    ADC ObjAnimFrame, X
    AND #$01                    ; There are only two images.
    CLC
    ADC #$04                    ; Leg frame images start at 4.
    JMP Anim_DrawObjectNotMirrored

; ----------------------------------------------------------------------------
; Sets both sprite attribute bytes to Gohma's palette row: object type - $32 (1 for $33, 2 for $34).
; IN:   X = object slot
; OUT:  [04] = left side sprite attributes; [05] = right side sprite attributes
; ----------------------------------------------------------------------------
Gohma_SetSpriteAttributes:
    LDA ObjType, X
    SEC
    SBC #$32
    JMP Anim_SetSpriteDescriptorAttributes

; ----------------------------------------------------------------------------
; Checks Gohma's collisions with Link and his weapons in five parts, 8 pixels apart.
; IN:   X = object slot
; OUT:  ObjX,X unchanged; [0F] = 0
; NOTE: [0F] numbers the part being tested: 5 at ObjX - $10, 4 at ObjX - 8, 3 at ObjX, 2 at ObjX + 8,
;       1 at ObjX + $10. Gohma_HandleWeaponCollision and Shove_Begin read [0F]; parts 3 and 4 are the
;       eye.
; NOTE: Each part runs the whole of Collision_CheckMonsterVsWeaponsAndLink, so Link is harmed by touching any part
;       and every part is skipped while ObjInvincibilityTimer,X is nonzero.
; NOTE: Parts are tested left to right and an arrow turns into a spark on the first part it overlaps
;       (box: under $0B pixels on both axes between the arrow's hit point and the part's middle, which
;       is the part's X + 4, ObjY + 8). So part 5 takes arrows with hit point X up to ObjX - 2, part 4
;       ObjX - 1 to ObjX + 6, part 3 ObjX + 7 to ObjX + $0E. The hit point X is the arrow's ObjX + 6
;       while Link faces up or down (Collision_CheckThinWeaponVsMonster_Thresholds).
; ----------------------------------------------------------------------------
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
    ; from left to right, counted down by [0F] from 5 to 1.
    LDA #$05
    STA $0F                     ; Part number

@LoopPart:
    JSR Collision_CheckMonsterVsWeaponsAndLink

    ; Add 8 to the X coordinate for the next part.
    LDA ObjX, X
    CLC
    ADC #$08
    STA ObjX, X
    DEC $0F                     ; Part number
    BNE @LoopPart

    ; Restore the boss's X coordinate.
    PLA
    STA ObjX, X
    RTS

; ----------------------------------------------------------------------------
; Decides what a weapon that got past Gohma's invincibility mask does: harm through the eye, or parry.
; IN:   X = Gohma's object slot; Y = weapon object slot; [0F] = collision part (5 to 1, from
;       Gohma_CheckCollisions); [07] = damage points
; OUT:  ObjHP,X reduced or Gohma dying (Damage_DealToMonster) when the hit counts; Tune0Request = 1 (parry sound)
; NOTE: Reached from Damage_HandleWeaponHit only for damage types the mask allows. Gohma_Init sets
;       mask $FB, so that is the arrow alone (object slot $12).
; NOTE: The hit counts only if the part is 3 or 4, Gohma_ObjEyeState,X = 3 and the arrow's direction
;       is up (8). Damage is $20 (wooden arrow) or $40 (silver arrow) against HP $60 (type $33) or
;       $20 (type $34).
; NOTE: A hit that counts also ends in @PlayParryTune, so the "harmed" sound request is replaced by
;       the parry sound request.
; ----------------------------------------------------------------------------
Gohma_HandleWeaponCollision:
    ; If hit by the weapon in slot $12 (an arrow), then set it to spark:
    ; state $28, animation counter 4. The arrow collision routine
    ; (Collision_CheckShotVsMonster) stores state $20 and counter 3 itself
    ; after this returns.
    CPY #$12
    BNE @CheckHitEye
    LDA #$28
    STA a:ObjState, Y
    LDA #$04
    STA ObjAnimCounter, Y

@CheckHitEye:
    ; If it did not hit part 3 nor 4 (the eye), then go play the parry sound effect.
    LDA $0F
    CMP #$03
    BEQ @CheckEyeOpen
    CMP #$04
    BNE @PlayParryTune

@CheckEyeOpen:
    ; If the state of the eye is not 3 (fully open), then go parry.
    LDA Gohma_ObjEyeState, X
    CMP #$03
    BNE @PlayParryTune

    ; If the direction of the arrow is not up, then go parry.
    LDA a:ObjDir, Y
    CMP #$08
    BNE @PlayParryTune
    LDA #$02                    ; Boss hit cry sample
    STA SampleRequest

    ; Deal damage, and play the boss death cry if Gohma is now dying.
    JSR Damage_DealToMonster
    JSR Sample_PlayBossDeathCryIfDying

@PlayParryTune:
    LDA #$01                    ; Parry sound
    STA Tune0Request
    RTS
