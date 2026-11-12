; ============================================================================
; FILE:     bank4/octorok.asm            BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 2967-3068
; SUBJECTS: (none)
; TOPIC:    Octorok update, rock shooting
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0015 UpdateOctorock                        code
;   L0112 DrawObjectMirroredAndCheckCollisions  code  Params:
; ============================================================================

    ; End unverified code

UpdateOctorock:
    ; If the monster is a fast octorock, use turn rate $A0.
    ; Else use turn rate $70.
    LDA #$A0
    LDY ObjType, X
    CPY #$09                    ; Blue octorock (fast and slow)
    BCS :+
    LDA #$70
:
    STA ObjTurnRate, X
    JSR Wanderer_TargetPlayer

    ; If the octorock is slow, then use q-speed $20.
    ; Else double it.
    LDA #$20
    LDY ObjType, X
    CPY #$07
    BEQ @Shoot
    CPY #$09
    BEQ @Shoot
    ASL

@Shoot:
    ; Shoot a flying rock (object type $53).
    LDY #$53
    JSR _TryShooting
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; If facing down, then use frame image offset 2.
    LDA ObjDir, X
    TAY
    AND #$0C
    BEQ @CheckHorizontal
    LDA #$02
    CPY #$08
    BNE @SaveOffset

    ; Else facing up. Use frame image offset 1.
    LDA #$01
    BNE @SaveOffset

@CheckHorizontal:
    ; If facing right, then set horizontal flipping, and use frame image offset 0.
    CPY #$01
    BNE :+
    INC $0F
:
    ; Else facing left. Use frame image offset 0, but no horizontal flipping.
    LDA #$00

@SaveOffset:
    PHA                         ; Save frame image offset for the direction.

    ; Decrement the animation counter, and roll over if needed.
    DEC ObjAnimCounter, X
    BNE @CalcFinalOffset
    LDA #$06
    STA ObjAnimCounter, X

    ; There are actually 6 frame images divided into 2 sets.
    ; Each set depicts one animation frame. In each set there are
    ; 3 frame images: 1 for each direction (left, up, down).
    ;
    ; The movement frame will keep track of the base offset of
    ; the current set for the current animation frame: 0 or 3.
    ;
    ; So, XOR the movement frame with 3.
    LSR
    EOR ObjAnimFrame, X
    STA ObjAnimFrame, X

@CalcFinalOffset:
    ; Add the base offset and the directional offset, and save the result.
    PLA
    CLC
    ADC ObjAnimFrame, X
    PHA

    ; If facing vertically, go draw mirrored.
    LDA ObjDir, X
    AND #$0C
    BNE @DrawMirrored

    ; Else draw not mirrored.
    PLA
    JSR DrawObjectNotMirrored
    JMP CheckMonsterCollisions

@DrawMirrored:
    PLA                         ; Restore the frame image.

; Params:
; A: frame image
; X: object index
; [00]: object X
; [01]: object Y
;
DrawObjectMirroredAndCheckCollisions:
    JSR DrawObjectMirrored
    JMP CheckMonsterCollisions
