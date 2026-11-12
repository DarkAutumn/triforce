; ============================================================================
; FILE:     bank4/darknut-gibdo.asm      BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 6449-6514
; SUBJECTS: (none)
; TOPIC:    InitDarknut, UpdateGibdo, UpdateDarknut
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0014 InitDarknut    code
;   L0030 UpdateGibdo    code
;   L0040 UpdateDarknut  code
; ============================================================================

InitDarknut:
    LDA #$F6                    ; Invincible to everything but sword and bomb.
    STA ObjInvincibilityMask, X

    ; If the darknut is red, then move at q-speed $20 (0.5 pixels a frame).
    LDA #$20
    LDY ObjType, X
    CPY #$0B
    BEQ :+

    ; Else move a blue darknut at q-speed $28 (0.625 pixels a frame).
    LDA #$28
:
    STA ObjQSpeedFrac, X
    JMP InitWalker

UpdateGibdo:
    LDA #$80                    ; Turn rate $80
    JSR UpdateCommonWanderer
    JSR CheckMonsterCollisions
    LDA #$08                    ; 8 screen frames an animation frame
    JSR Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor
    JSR Anim_SetObjHFlipForSpriteDescriptor
    LDA #$00                    ; Frame image 0
    JMP DrawObjectNotMirrored

UpdateDarknut:
    LDA #$80                    ; Turn rate $80
    JSR UpdateCommonWanderer
    JSR CheckMonsterCollisions

    ; Darknuts are never stunned.
    LDA #$00
    STA ObjStunTimer, X
    LDA #$08                    ; 8 screen frames an animation frame
    JSR Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor

    ; If facing left, then enable horizontal flipping.
    LDA ObjDir, X
    CMP #$02
    BNE :+
    INC $0F
:
    ; Shift right 2, yielding: 2 up, 1 down, 0 horizontal
    ; This is the base frame image.
    LSR
    LSR

    ; If movement frame = 1, add 3 to frame image to access
    ; the second animation frame's images.
    LDY ObjAnimFrame, X
    BEQ :+
    CLC
    ADC #$03

    ; Also if movement frame = 1, and facing up, then flip horizontally.
    ;
    ; Down has two frame images, but up only has one. By flipping
    ; horizontally, you get two frame images.
    LDY ObjDir, X
    CPY #$08
    BNE :+
    INC $0F
:
    JSR DrawObjectNotMirrored
    RTS
