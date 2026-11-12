; ============================================================================
; FILE:     bank4/vire.asm               BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 6915-7025
; SUBJECTS: (none)
; TOPIC:    Vire: hop, split into red Keese
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0018 UpdateVire                 code
;   L0054 UpdateVireState            code
;   L0058 UpdateVireState_JumpTable  data
;   L0062 VireJumpOffsets            data
;   L0066 UpdateVireState0           code
;   L0096 CheckVireCollisions        code
;   L0116 DrawVire                   code
; ============================================================================

UpdateVire:
    JSR UpdateVireState

    ; If state >= 2, then go split up into 2 keeses.
    LDA ObjState, X
    CMP #$02
    BCS @SplitUp
    JSR CheckVireCollisions
    JMP DrawVire

@SplitUp:
    ; As with Zol, we'll destroy one monster, and make two.
    ; So, increase object count by 1.
    INC RoomObjCount
    JSR DestroyMonster

    ; Two times, look for an empty monster slot.
    ; If one is found, then use the "shoot" operation to create a red keese.
    LDY #$01

@LoopMakeKeese:
    TYA                         ; Save loop counter.
    PHA
    JSR FindEmptyMonsterSlot
    BEQ @NextLoopMakeKeese
    LDA #$1C                    ; Red Keese object type
    STA $00
    JSR Shoot

@NextLoopMakeKeese:
    PLA                         ; Restore loop counter.
    TAY
    DEY
    BPL @LoopMakeKeese
    RTS

UpdateVireState:
    LDA ObjState, X
    JSR TableJump

UpdateVireState_JumpTable:
    .ADDR UpdateVireState0
    .ADDR UpdateZolState1_Shove

VireJumpOffsets:
    .BYTE $00, $FD, $FE, $FF, $FF, $00, $FF, $00
    .BYTE $00, $01, $00, $01, $01, $02, $03, $00

UpdateVireState0:
    LDA #$80                    ; Turn rate $80
    JSR UpdateCommonWanderer

    ; If we have the clock, or monster is stunned; then return.
    LDA InvClock
    ORA ObjStunTimer, X
    BNE @Exit

    ; If going vertically, then return.
    LDA ObjDir, X
    AND #$03
    BEQ @Exit

    ; Get the absolute value of the grid offset (0 to $F);
    ; so it becomes an index into the jump offset array.
    LDA ObjGridOffset, X
    JSR Abs
    TAY

    ; Add the vertical offset; so that as the vire moves left or right,
    ; it jumps.
    LDA ObjY, X
    CLC
    ADC VireJumpOffsets, Y
    STA ObjY, X

@Exit:
    RTS

CheckVireCollisions:
    ; If state <> 0, then the vire's going to split up.
    ; So, no need to check collisions. Return.
    LDA ObjState, X
    BNE @Exit
    JSR CheckMonsterCollisions

    ; If it was killed, then return.
    LDA ObjMetastate, X
    BNE @Exit

    ; If temporarily invincible, then it was harmed.
    ; Advance the state to split up.
    LDA ObjInvincibilityTimer, X
    BEQ @Exit
    INC ObjState, X

@Exit:
    RTS

DrawVire:
    LDA #$0A                    ; Animation counter $A
    JSR Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor

    ; The movement frame is the fram image number, unless facing up.
    ; In this case, add 2 to the frame image.
    LDA ObjDir, X
    AND #$08
    LSR
    LSR
    CLC
    ADC ObjAnimFrame, X
    JMP DrawObjectMirrored
