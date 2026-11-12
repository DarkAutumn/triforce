; ============================================================================
; FILE:     bank7/object-stubs-destroy-monster.asm BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 5751-5862
; SUBJECTS: (none)
; TOPIC:    Fixed-bank trampolines to the bank-1 init/update routines of Whirlwind, Rupee stash, Trap,
;           dungeon persons and Grumble; invincibility countdown; start of the death sparkle; freeing
;           an object slot; search for a free monster slot.
; SEE ALSO: bank7/object-init-table.asm and bank7/object-update-table.asm (the tables that name these
;           trampolines; what happens when the death sparkle ends); bank1/bomb-candle-wield.asm
;           (DestroyObject_WRAM); bank1/weapon-damage-death.asm (monster death calls UpdateDeadDummy);
;           bank7/object-animation.asm (spawn cloud and death sparkle stages).
; KEY FACTS:
;   - ObjInvincibilityTimer counts down by 1 only on frames with an even FrameCounter, so a value N
;     lasts about 2N frames (DecrementInvincibilityTimer).
;   - A monster's death starts with ObjMetastate = $10 and the request $20 in Tune1Request
;     (UpdateDeadDummy); the dead dummy type $5D does exactly this as both its init and its update.
;   - DestroyMonster clears the slot's type (0 = free) and, through DestroyObject_WRAM, its shove,
;     timer, state and invincibility timer; it leaves ObjUninitialized = $FF and ObjMetastate = 1,
;     ready for the next object put in the slot.
;   - FindEmptyMonsterSlot searches slots $0B down to $01 and takes the highest free one; slot 0 and
;     slots above $0B are never returned.
; CONTENTS: (generated; L = line in this file)
;   L0046 UpdateWhirlwind                    code  Whirlwind ($2E) update: switches to bank 1 and jumps to UpdateWhirlwind_Full.
;   L0052 InitRupeeStash                     code  Rupee stash ($35) init: switches to bank 1 and jumps to InitRupeeStash_Full.
;   L0058 UpdateRupeeStash                   code  Rupee stash ($35) update: switches to bank 1 and jumps to UpdateRupeeStash_Full.
;   L0064 InitTrap                           code  Trap ($49, $4A) init: switches to bank 1 and jumps to InitTrap_Full.
;   L0070 UpdateTrap                         code  Trap ($49, $4A) update: switches to bank 1 and jumps to UpdateTrap_Full.
;   L0076 InitUnderworldPerson               code  Dungeon person ($4B-$50, $52) init: switches to bank 1 and jumps to InitUnderworldPerson_Full.
;   L0082 InitUnderworldPersonLifeOrMoney    code  Life-or-money man ($51) init: switches to bank 1, jumps to InitUnderworldPersonLifeOrMoney_Full.
;   L0088 InitGrumble                        code  Grumble Goriya ($36) init: switches to bank 1 and jumps to InitGrumble_Full.
;   L0094 UpdateUnderworldPerson             code  Dungeon person update: switches to bank 1 and jumps to UpdateUnderworldPerson_Full.
;   L0100 UpdateUnderworldPersonLifeOrMoney  code  Life-or-money man update: switches to bank 1, jumps to UpdateUnderworldPersonLifeOrMoney_Full.
;   L0106 UpdateGrumble                      code  Grumble Goriya update: switches to bank 1 and jumps to UpdateGrumble_Full.
;   L0119 DecrementInvincibilityTimer        code  Counts an object's invincibility timer down by 1 on every second frame.
;   L0139 UpdateDeadDummy                    code  Starts an object's death: requests the death sound and sets the first dying metastate ($10).
;   L0153 DestroyMonster                     code  Frees an object slot: type 0, then the common per-slot reset (DestroyObject_WRAM).
;   L0158 SetTypeAndClearObject              code  Second entry: puts object type A in slot X and resets the slot, so a new object starts there.
;   L0169 FindEmptyMonsterSlot               code  Finds the highest free object slot among $0B down to $01.
; ============================================================================

; ----------------------------------------------------------------------------
; Whirlwind ($2E) update: switches to bank 1 and jumps to UpdateWhirlwind_Full.
; IN:   X = object slot (as for every trampoline below).
; NOTE: None of these trampolines switches the bank back; bank 1 stays selected.
; ----------------------------------------------------------------------------
UpdateWhirlwind:
    LDA #$01
    JSR SwitchBank
    JMP UpdateWhirlwind_Full

; Rupee stash ($35) init: switches to bank 1 and jumps to InitRupeeStash_Full.
InitRupeeStash:
    LDA #$01
    JSR SwitchBank
    JMP InitRupeeStash_Full

; Rupee stash ($35) update: switches to bank 1 and jumps to UpdateRupeeStash_Full.
UpdateRupeeStash:
    LDA #$01
    JSR SwitchBank
    JMP UpdateRupeeStash_Full

; Trap ($49, $4A) init: switches to bank 1 and jumps to InitTrap_Full.
InitTrap:
    LDA #$01
    JSR SwitchBank
    JMP InitTrap_Full

; Trap ($49, $4A) update: switches to bank 1 and jumps to UpdateTrap_Full.
UpdateTrap:
    LDA #$01
    JSR SwitchBank
    JMP UpdateTrap_Full

; Dungeon person ($4B-$50, $52) init: switches to bank 1 and jumps to InitUnderworldPerson_Full.
InitUnderworldPerson:
    LDA #$01
    JSR SwitchBank
    JMP InitUnderworldPerson_Full

; Life-or-money man ($51) init: switches to bank 1, jumps to InitUnderworldPersonLifeOrMoney_Full.
InitUnderworldPersonLifeOrMoney:
    LDA #$01
    JSR SwitchBank
    JMP InitUnderworldPersonLifeOrMoney_Full

; Grumble Goriya ($36) init: switches to bank 1 and jumps to InitGrumble_Full.
InitGrumble:
    LDA #$01
    JSR SwitchBank
    JMP InitGrumble_Full

; Dungeon person update: switches to bank 1 and jumps to UpdateUnderworldPerson_Full.
UpdateUnderworldPerson:
    LDA #$01
    JSR SwitchBank
    JMP UpdateUnderworldPerson_Full

; Life-or-money man update: switches to bank 1, jumps to UpdateUnderworldPersonLifeOrMoney_Full.
UpdateUnderworldPersonLifeOrMoney:
    LDA #$01
    JSR SwitchBank
    JMP UpdateUnderworldPersonLifeOrMoney_Full

; Grumble Goriya update: switches to bank 1 and jumps to UpdateGrumble_Full.
UpdateGrumble:
    LDA #$01
    JSR SwitchBank
    JMP UpdateGrumble_Full

; ----------------------------------------------------------------------------
; Counts an object's invincibility timer down by 1 on every second frame.
; IN:   X = object slot (0 = Link).
; OUT:  ObjInvincibilityTimer,X one lower if it was nonzero and FrameCounter is even.
; NOTE: A value N therefore lasts about 2N frames. Called for Link at the start of his update and
;       in the death mode's init, and for every slot $0B-$01 in the play-mode object loop, before
;       the slot's type is tested (so also for empty slots).
; ----------------------------------------------------------------------------
DecrementInvincibilityTimer:
    LDA ObjInvincibilityTimer, X
    BEQ @Exit                   ; Already 0.
    LDA FrameCounter
    LSR
    BCS @Exit                   ; Odd frame: no change.
    DEC ObjInvincibilityTimer, X

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Starts an object's death: requests the death sound and sets the first dying metastate ($10).
; IN:   X = object slot.
; OUT:  Tune1Request = $20; ObjMetastate,X = $10; A = $10.
; NOTE: This is the init and the update routine of the dead dummy (type $5D), and it is called
;       directly when a monster's HP runs out and when Dodongo dies. From metastate $10 the
;       object plays the death sparkle; at its end the slot becomes a dropped item object, type
;       $60 (UpdateMetaObjectEnd), and SetUpDroppedItem decides whether an item appears.
; ----------------------------------------------------------------------------
UpdateDeadDummy:
    LDA #$20                    ; Tune 1 request bit $20: the monster death sound.
    STA Tune1Request
    LDA #$10                    ; First metastate of the death sparkle.
    STA ObjMetastate, X
    RTS

; ----------------------------------------------------------------------------
; Frees an object slot: type 0, then the common per-slot reset (DestroyObject_WRAM).
; IN:   X = object slot.
; OUT:  ObjType,X = 0; shove direction and distance, ObjTimer, ObjState and ObjInvincibilityTimer
;       cleared; ObjUninitialized,X = $FF; ObjMetastate,X = 1; A = 1.
; NOTE: No death sparkle, no drop, no kill counted.
; ----------------------------------------------------------------------------
DestroyMonster:
    LDA #$00

; Second entry: puts object type A in slot X and resets the slot, so a new object starts there.
; The new object is uninitialized (its init runs on its next update) with metastate 1 (spawn cloud).
SetTypeAndClearObject:
    STA ObjType, X
    LDA #$00
    JMP DestroyObject_WRAM

; ----------------------------------------------------------------------------
; Finds the highest free object slot among $0B down to $01.
; OUT:  Found: Y = EmptyMonsterSlot = the slot, A = 0, Z clear (BNE taken).
;       None free: Y = 0, Z set (BEQ taken), A = slot 1's type (nonzero), EmptyMonsterSlot unchanged.
; NOTE: X is preserved. Callers test the result with BEQ (none free) or with A (CMP #$01).
; ----------------------------------------------------------------------------
FindEmptyMonsterSlot:
    LDY #$0C

@Loop:
    DEY
    BEQ @End                    ; Reached slot 0 (Link): none free.
    LDA ObjType, Y
    BNE @Loop                   ; Slot in use.
    STY EmptyMonsterSlot

@End:
    CPY #$00
    RTS
