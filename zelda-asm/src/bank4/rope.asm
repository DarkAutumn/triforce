; ============================================================================
; FILE:     bank4/rope.asm               BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 4537-4668
; SUBJECTS: Rope
; TOPIC:    Rope (type $28): init with HP by quest, and the update that walks slowly, turns at random and
;           rushes when Link is in line with it.
; SEE ALSO: bank7/walker-move.asm (Walker_Move); bank7/walker-collision-screen-edge.asm
;           (Walker_FaceUnblockedDir, and Walker_CheckTileCollision which turns a blocked Rope and so ends a
;           rush); bank4/walker-wanderer-common.asm (Walker_Init); bank1/monster-collisions.asm.
; KEY FACTS:
;   - HP is $10 in Q1 and $40 in Q2 (HP nibbles 1 and 4; ObjHP is the nibble * $10). Rope_Init.
;   - Q-speed $20 (0.5 px/frame) when walking, $60 (1.5 px/frame) when rushing. Rope_Update@Rush.
;   - A rush starts only at q-speed $20 and grid offset 0, when Link's X is within 7 pixels of the
;     Rope's X (rush up or down) or else his Y within 7 of its Y (rush left or right). It uses Link's
;     real position (ObjX, ObjY), not the chase target. Rope_Update, Rope_Update@CheckVertical.
;   - A rush ends when the facing direction changes during the move (a blocked Rope is turned by
;     Walker_CheckTileCollision): q-speed goes back to $20. Rope_Update@CheckSpeed.
;   - While walking, each time ObjTimer runs out it is reloaded with Random AND $3F (0-63 frames) and
;     the Rope turns to an unblocked direction if it is on a grid point. Rope_Update.
;   - The clock or a stun stops movement and turning, but the rush test still runs. Rope_Update.
;   - In Q2 a Rope's sprite palette changes every frame (FrameCounter AND 3). Rope_Update@Animate.
; CONTENTS: (generated; L = line in this file)
;   L0032 Rope_Init    code  Initializes a Rope (type $28): HP $10 in Q1 or $40 in Q2, then the common walker init.
;   L0051 Rope_Update  code  Updates a Rope: walks at 0.5 px/frame, turns at random, rushes at 1.5 px/frame when Link lines up.
; ============================================================================

; ----------------------------------------------------------------------------
; Initializes a Rope (type $28): HP $10 in Q1 or $40 in Q2, then the common walker init.
; IN:   X = object slot
; OUT:  ObjHP,X = $10 (Q1) or $40 (Q2)
; ----------------------------------------------------------------------------
Rope_Init:
    LDA #$10
    STA ObjHP, X
    LDY CurSaveSlot
    LDA QuestNumbers, Y         ; 0 = Q1, nonzero = Q2
    BEQ :+
    LDA #$40
    STA ObjHP, X
:
    JMP Walker_Init

; ----------------------------------------------------------------------------
; Updates a Rope: walks at 0.5 px/frame, turns at random, rushes at 1.5 px/frame when Link lines up.
; IN:   X = object slot
; OUT:  ObjDir,X and ObjQSpeedFrac,X ($20 walk, $60 rush) updated; drawn; collisions checked.
; NOTE: The rush test needs q-speed $20 and grid offset 0, and |Link X - Rope X| < 8 (vertical rush)
;       or else |Link Y - Rope Y| < 8 (horizontal rush). The rush lasts until the Rope's facing
;       direction changes during a move.
; ----------------------------------------------------------------------------
Rope_Update:
    ; Move in the facing direction.
    LDA ObjDir, X
    STA ObjInputDir, X
    PHA                         ; Save the facing direction to compare after the move.

    ; With the clock, or while stunned, don't move or turn.
    LDA InvClock
    ORA ObjStunTimer, X
    BNE @CheckSpeed
    JSR Walker_Move

    ; When the grid offset reaches a multiple of $10 ($10 or -$10: a whole square), reset it to 0.
    LDA ObjGridOffset, X
    AND #$0F
    BNE :+
    STA ObjGridOffset, X
:
    ; If not rushing (q-speed <> $60) and the timer has run out, then set the timer to a
    ; random value from 0 to $3F frames, and turn to face an unblocked direction
    ; (Walker_FaceUnblockedDir does nothing unless the grid offset is 0).
    LDA ObjQSpeedFrac, X
    CMP #$60
    BEQ @CheckSpeed
    LDA ObjTimer, X
    BNE @CheckSpeed
    LDA Random, X
    AND #$3F
    STA ObjTimer, X
    JSR Walker_FaceUnblockedDir

@CheckSpeed:
    PLA                         ; The facing direction from before the move.

    ; If the facing direction has changed, then set q-speed to $20 (0.5 px/frame).
    ; This is what ends a rush.
    CMP ObjDir, X
    BEQ :+
    LDA #$20
    STA ObjQSpeedFrac, X
:
    ; Only test for a rush at q-speed $20 and grid offset 0. Otherwise go draw.
    LDA ObjQSpeedFrac, X
    CMP #$20
    BNE @Animate
    LDA ObjGridOffset, X
    BNE @Animate

    ; If the absolute horizontal distance between Link and the Rope >= 8,
    ; then go test the vertical distance.
    LDA ObjX
    SEC
    SBC ObjX, X
    JSR Math_Abs
    CMP #$08
    BCS @CheckVertical

    ; Link is in the Rope's column. Face up.
    LDA #$08
    STA ObjDir, X

    ; But if Link's Y >= the Rope's Y, then face down (8 shifted right is 4).
    LDA ObjY
    CMP ObjY, X
    BCC @Rush

@FlipDir:
    LSR ObjDir, X               ; Up (8) -> down (4), or left (2) -> right (1).

@Rush:
    ; Rush: q-speed $60 (1.5 px/frame).
    LDA #$60
    STA ObjQSpeedFrac, X

@Animate:
    LDA #$0A                    ; $0A frames per animation frame
    JSR Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor

    ; If facing left, then flip horizontally.
    LDA ObjDir, X
    AND #$02
    LSR
    STA $0F                     ; [0F] horizontal flipping
    LDA #$02                    ; Sprite attributes 2: palette row 6 (red)
    JSR Anim_SetSpriteDescriptorAttributes

    ; In Q2, Ropes flash: the sprite attributes are the low 2 bits of the frame counter,
    ; so the palette row changes every frame.
    LDY CurSaveSlot
    LDA QuestNumbers, Y
    BEQ @Draw
    LDA FrameCounter
    AND #$03
    JSR Anim_SetSpriteDescriptorAttributes

@Draw:
    ; The frame image is the animation frame (0 or 1).
    LDA ObjAnimFrame, X
    JSR Anim_DrawObjectNotMirrored
    JMP Collision_CheckMonsterVsWeaponsAndLink

@CheckVertical:
    ; If the absolute vertical distance between Link and the Rope >= 8,
    ; then Link is not in line. Go draw.
    LDA ObjY
    SEC
    SBC ObjY, X
    JSR Math_Abs
    CMP #$08
    BCS @Animate

    ; Link is in the Rope's row. Face left.
    LDA #$02
    STA ObjDir, X

    ; But if Link's X >= the Rope's X, then face right.
    ; Either way, go rush and draw.
    LDA ObjX
    CMP ObjX, X
    BCS @FlipDir
    BCC @Rush
