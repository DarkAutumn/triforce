; ============================================================================
; FILE:     bank7/object-stubs-destroy-monster.asm BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 5751-5862
; SUBJECTS: (none)
; TOPIC:    Bank-1 trampolines, invincibility timer, dead dummy, DestroyMonster
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0027 UpdateWhirlwind                    code
;   L0032 InitRupeeStash                     code
;   L0037 UpdateRupeeStash                   code
;   L0042 InitTrap                           code
;   L0047 UpdateTrap                         code
;   L0052 InitUnderworldPerson               code
;   L0057 InitUnderworldPersonLifeOrMoney    code
;   L0062 InitGrumble                        code
;   L0067 UpdateUnderworldPerson             code
;   L0072 UpdateUnderworldPersonLifeOrMoney  code
;   L0077 UpdateGrumble                      code
;   L0085 DecrementInvincibilityTimer        code  Params:
;   L0096 UpdateDeadDummy                    code
;   L0106 DestroyMonster                     code  Params:
;   L0113 SetTypeAndClearObject              code  Params:
;   L0126 FindEmptyMonsterSlot               code  Look for an empty object slot for a monster from $B to 1.
; ============================================================================

UpdateWhirlwind:
    LDA #$01
    JSR SwitchBank
    JMP UpdateWhirlwind_Full

InitRupeeStash:
    LDA #$01
    JSR SwitchBank
    JMP InitRupeeStash_Full

UpdateRupeeStash:
    LDA #$01
    JSR SwitchBank
    JMP UpdateRupeeStash_Full

InitTrap:
    LDA #$01
    JSR SwitchBank
    JMP InitTrap_Full

UpdateTrap:
    LDA #$01
    JSR SwitchBank
    JMP UpdateTrap_Full

InitUnderworldPerson:
    LDA #$01
    JSR SwitchBank
    JMP InitUnderworldPerson_Full

InitUnderworldPersonLifeOrMoney:
    LDA #$01
    JSR SwitchBank
    JMP InitUnderworldPersonLifeOrMoney_Full

InitGrumble:
    LDA #$01
    JSR SwitchBank
    JMP InitGrumble_Full

UpdateUnderworldPerson:
    LDA #$01
    JSR SwitchBank
    JMP UpdateUnderworldPerson_Full

UpdateUnderworldPersonLifeOrMoney:
    LDA #$01
    JSR SwitchBank
    JMP UpdateUnderworldPersonLifeOrMoney_Full

UpdateGrumble:
    LDA #$01
    JSR SwitchBank
    JMP UpdateGrumble_Full

; Params:
; X: object index
;
DecrementInvincibilityTimer:
    LDA ObjInvincibilityTimer, X
    BEQ @Exit                   ; If the timer is already zero, then return.
    LDA FrameCounter            ; Every two frames, decrease the timer.
    LSR
    BCS @Exit
    DEC ObjInvincibilityTimer, X

@Exit:
    RTS

UpdateDeadDummy:
    LDA #$20                    ; Monster died sound effect
    STA Tune1Request
    LDA #$10                    ; First metastate of death spark
    STA ObjMetastate, X
    RTS

; Params:
; X: object index
;
DestroyMonster:
    LDA #$00

; Params:
; A: object type
; X: object index
;
SetTypeAndClearObject:
    STA ObjType, X
    LDA #$00
    JMP DestroyObject_WRAM

; Look for an empty object slot for a monster from $B to 1.
;
; Returns:
; A: 0 if found an empty slot
; Y: an empty slot, if found
; Z: 0 if found an empty slot
; [59]: an empty slot, if found
;
FindEmptyMonsterSlot:
    LDY #$0C

@Loop:
    DEY
    BEQ @End
    LDA ObjType, Y
    BNE @Loop
    STY EmptyMonsterSlot

@End:
    CPY #$00
    RTS
