; ============================================================================
; FILE:     bank4/rope.asm               BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 4537-4668
; SUBJECTS: (none)
; TOPIC:    Rope init (HP by quest) and rush at Link
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0013 InitRope    code
;   L0025 UpdateRope  code
; ============================================================================

InitRope:
    ; Rope has $10 HP in quest 1, $40 HP in quest 2.
    LDA #$10
    STA ObjHP, X
    LDY CurSaveSlot
    LDA QuestNumbers, Y
    BEQ :+
    LDA #$40
    STA ObjHP, X
:
    JMP InitWalker

UpdateRope:
    ; Set the input direction to the facing direction.
    LDA ObjDir, X
    STA ObjInputDir, X
    PHA                         ; Save the original facing direction

    ; If we have the clock, or the monster is stunned, then don't move.
    LDA InvClock
    ORA ObjStunTimer, X
    BNE @CheckSpeed
    JSR Walker_Move

    ; Truncate the grid offset at $10.
    LDA ObjGridOffset, X
    AND #$0F
    BNE :+
    STA ObjGridOffset, X
:
    ; If not rushing (qspeed <> $60) and timer = 0, then
    ; set timer to a random value up to $3F, and turn to face
    ; an unblocked direction.
    LDA ObjQSpeedFrac, X
    CMP #$60
    BEQ @CheckSpeed
    LDA ObjTimer, X
    BNE @CheckSpeed
    LDA Random, X
    AND #$3F
    STA ObjTimer, X
    JSR _FaceUnblockedDir

@CheckSpeed:
    PLA                         ; Pop the original facing direction

    ; If the facing direction has changed, then set qspeed to $20 (slow).
    CMP ObjDir, X
    BEQ :+
    LDA #$20
    STA ObjQSpeedFrac, X
:
    ; If qspeed <> $20 or grid offset <> 0, then go draw.
    LDA ObjQSpeedFrac, X
    CMP #$20
    BNE @Animate
    LDA ObjGridOffset, X
    BNE @Animate

    ; If the absolute horizontal distance between Link and the monster >= 8,
    ; then go see if the vertical distance is smaller.
    LDA ObjX
    SEC
    SBC ObjX, X
    JSR Abs
    CMP #$08
    BCS @CheckVertical

    ; Face up.
    LDA #$08
    STA ObjDir, X

    ; But, if Link is down from the monster, then face down.
    LDA ObjY
    CMP ObjY, X
    BCC @Rush

@FlipDir:
    LSR ObjDir, X

@Rush:
    ; Make the rope rush (qspeed $60).
    LDA #$60
    STA ObjQSpeedFrac, X

@Animate:
    LDA #$0A                    ; Animation counter $A
    JSR Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor

    ; If facing left, then flip horizontally.
    LDA ObjDir, X
    AND #$02
    LSR
    STA $0F                     ; [0F] horizontal flipping
    LDA #$02                    ; Red sprite palette
    JSR Anim_SetSpriteDescriptorAttributes

    ; In the second quest, ropes always flash.
    ; So, set sprite attributes to all the cycled palette rows.
    LDY CurSaveSlot
    LDA QuestNumbers, Y
    BEQ @Draw
    LDA FrameCounter
    AND #$03
    JSR Anim_SetSpriteDescriptorAttributes

@Draw:
    ; The frame image is based on the movement frame.
    LDA ObjAnimFrame, X
    JSR DrawObjectNotMirrored
    JMP CheckMonsterCollisions

@CheckVertical:
    ; If the absolute vertical distance between Link and the monster >= 8,
    ; then go draw.
    LDA ObjY
    SEC
    SBC ObjY, X
    JSR Abs
    CMP #$08
    BCS @Animate

    ; Face left.
    LDA #$02
    STA ObjDir, X

    ; But, if Link is right of the monster, then face right.
    ; Either way, go increase speed and draw.
    LDA ObjX
    CMP ObjX, X
    BCS @FlipDir
    BCC @Rush
