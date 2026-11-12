; ============================================================================
; FILE:     bank7/object-stubs-destroy-monster.asm BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 5751-5862
; SUBJECTS: DeadDummy, Grumble, Obj, Person, RupeeStash, Trap, Whirlwind
; TOPIC:    Fixed-bank trampolines to the bank-1 init/update routines of Whirlwind, Rupee stash, Trap,
;           dungeon persons and Grumble; invincibility countdown; start of the death sparkle; freeing
;           an object slot; search for a free monster slot.
; SEE ALSO: bank7/object-init-table.asm and bank7/object-update-table.asm (the tables that name these
;           trampolines; what happens when the death sparkle ends); bank1/bomb-candle-wield.asm
;           (Obj_ResetSlotFields); bank1/weapon-damage-death.asm (monster death calls DeadDummy_StartDeathSparkle);
;           bank7/object-animation.asm (spawn cloud and death sparkle stages).
; KEY FACTS:
;   - ObjInvincibilityTimer counts down by 1 only on frames with an even FrameCounter, so a value N
;     lasts about 2N frames (Obj_DecInvincibilityTimer).
;   - A monster's death starts with ObjMetastate = $10 and the request $20 in Tune1Request
;     (DeadDummy_StartDeathSparkle); the dead dummy type $5D does exactly this as both its init and its update.
;   - Obj_DestroyMonster clears the slot's type (0 = free) and, through Obj_ResetSlotFields, its shove,
;     timer, state and invincibility timer; it leaves ObjUninitialized = $FF and ObjMetastate = 1,
;     ready for the next object put in the slot.
;   - Obj_FindEmptyMonsterSlot searches slots $0B down to $01 and takes the highest free one; slot 0 and
;     slots above $0B are never returned.
; CONTENTS: (generated; L = line in this file)
;   L0046 Whirlwind_Update             code  Whirlwind ($2E) update: switches to bank 1 and jumps to Whirlwind_Update_Full.
;   L0052 RupeeStash_Init              code  Rupee stash ($35) init: switches to bank 1 and jumps to RupeeStash_Init_Full.
;   L0058 RupeeStash_Update            code  Rupee stash ($35) update: switches to bank 1 and jumps to RupeeStash_Update_Full.
;   L0064 Trap_Init                    code  Trap ($49, $4A) init: switches to bank 1 and jumps to Trap_Init_Full.
;   L0070 Trap_Update                  code  Trap ($49, $4A) update: switches to bank 1 and jumps to Trap_Update_Full.
;   L0076 Person_InitUnderworld        code  Dungeon person ($4B-$50, $52) init: switches to bank 1 and jumps to Person_InitUnderworld_Full.
;   L0082 Person_InitLifeOrMoney       code  Life-or-money man ($51) init: switches to bank 1, jumps to Person_InitLifeOrMoney_Full.
;   L0088 Grumble_Init                 code  Grumble Goriya ($36) init: switches to bank 1 and jumps to Grumble_Init_Full.
;   L0094 Person_UpdateUnderworld      code  Dungeon person update: switches to bank 1 and jumps to Person_UpdateUnderworld_Full.
;   L0100 Person_UpdateLifeOrMoney     code  Life-or-money man update: switches to bank 1, jumps to Person_UpdateLifeOrMoney_Full.
;   L0106 Grumble_Update               code  Grumble Goriya update: switches to bank 1 and jumps to Grumble_Update_Full.
;   L0119 Obj_DecInvincibilityTimer    code  Counts an object's invincibility timer down by 1 on every second frame.
;   L0139 DeadDummy_StartDeathSparkle  code  Starts an object's death: requests the death sound and sets the first dying metastate ($10).
;   L0153 Obj_DestroyMonster           code  Frees an object slot: type 0, then the common per-slot reset (Obj_ResetSlotFields).
;   L0158 Obj_SetTypeAndClear          code  Second entry: puts object type A in slot X and resets the slot, so a new object starts there.
;   L0169 Obj_FindEmptyMonsterSlot     code  Finds the highest free object slot among $0B down to $01.
; ============================================================================

; ----------------------------------------------------------------------------
; Whirlwind ($2E) update: switches to bank 1 and jumps to Whirlwind_Update_Full.
; IN:   X = object slot (as for every trampoline below).
; NOTE: None of these trampolines switches the bank back; bank 1 stays selected.
; ----------------------------------------------------------------------------
Whirlwind_Update:
    LDA #$01
    JSR Mmc1_SwitchBank
    JMP Whirlwind_Update_Full

; Rupee stash ($35) init: switches to bank 1 and jumps to RupeeStash_Init_Full.
RupeeStash_Init:
    LDA #$01
    JSR Mmc1_SwitchBank
    JMP RupeeStash_Init_Full

; Rupee stash ($35) update: switches to bank 1 and jumps to RupeeStash_Update_Full.
RupeeStash_Update:
    LDA #$01
    JSR Mmc1_SwitchBank
    JMP RupeeStash_Update_Full

; Trap ($49, $4A) init: switches to bank 1 and jumps to Trap_Init_Full.
Trap_Init:
    LDA #$01
    JSR Mmc1_SwitchBank
    JMP Trap_Init_Full

; Trap ($49, $4A) update: switches to bank 1 and jumps to Trap_Update_Full.
Trap_Update:
    LDA #$01
    JSR Mmc1_SwitchBank
    JMP Trap_Update_Full

; Dungeon person ($4B-$50, $52) init: switches to bank 1 and jumps to Person_InitUnderworld_Full.
Person_InitUnderworld:
    LDA #$01
    JSR Mmc1_SwitchBank
    JMP Person_InitUnderworld_Full

; Life-or-money man ($51) init: switches to bank 1, jumps to Person_InitLifeOrMoney_Full.
Person_InitLifeOrMoney:
    LDA #$01
    JSR Mmc1_SwitchBank
    JMP Person_InitLifeOrMoney_Full

; Grumble Goriya ($36) init: switches to bank 1 and jumps to Grumble_Init_Full.
Grumble_Init:
    LDA #$01
    JSR Mmc1_SwitchBank
    JMP Grumble_Init_Full

; Dungeon person update: switches to bank 1 and jumps to Person_UpdateUnderworld_Full.
Person_UpdateUnderworld:
    LDA #$01
    JSR Mmc1_SwitchBank
    JMP Person_UpdateUnderworld_Full

; Life-or-money man update: switches to bank 1, jumps to Person_UpdateLifeOrMoney_Full.
Person_UpdateLifeOrMoney:
    LDA #$01
    JSR Mmc1_SwitchBank
    JMP Person_UpdateLifeOrMoney_Full

; Grumble Goriya update: switches to bank 1 and jumps to Grumble_Update_Full.
Grumble_Update:
    LDA #$01
    JSR Mmc1_SwitchBank
    JMP Grumble_Update_Full

; ----------------------------------------------------------------------------
; Counts an object's invincibility timer down by 1 on every second frame.
; IN:   X = object slot (0 = Link).
; OUT:  ObjInvincibilityTimer,X one lower if it was nonzero and FrameCounter is even.
; NOTE: A value N therefore lasts about 2N frames. Called for Link at the start of his update and
;       in the death mode's init, and for every slot $0B-$01 in the play-mode object loop, before
;       the slot's type is tested (so also for empty slots).
; ----------------------------------------------------------------------------
Obj_DecInvincibilityTimer:
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
;       $60 (Obj_EndMetastate), and Drop_SetUpItem decides whether an item appears.
; ----------------------------------------------------------------------------
DeadDummy_StartDeathSparkle:
    LDA #$20                    ; Tune 1 request bit $20: the monster death sound.
    STA Tune1Request
    LDA #$10                    ; First metastate of the death sparkle.
    STA ObjMetastate, X
    RTS

; ----------------------------------------------------------------------------
; Frees an object slot: type 0, then the common per-slot reset (Obj_ResetSlotFields).
; IN:   X = object slot.
; OUT:  ObjType,X = 0; shove direction and distance, ObjTimer, ObjState and ObjInvincibilityTimer
;       cleared; ObjUninitialized,X = $FF; ObjMetastate,X = 1; A = 1.
; NOTE: No death sparkle, no drop, no kill counted.
; ----------------------------------------------------------------------------
Obj_DestroyMonster:
    LDA #$00

; Second entry: puts object type A in slot X and resets the slot, so a new object starts there.
; The new object is uninitialized (its init runs on its next update) with metastate 1 (spawn cloud).
Obj_SetTypeAndClear:
    STA ObjType, X
    LDA #$00
    JMP Obj_ResetSlotFields

; ----------------------------------------------------------------------------
; Finds the highest free object slot among $0B down to $01.
; OUT:  Found: Y = EmptyMonsterSlot = the slot, A = 0, Z clear (BNE taken).
;       None free: Y = 0, Z set (BEQ taken), A = slot 1's type (nonzero), EmptyMonsterSlot unchanged.
; NOTE: X is preserved. Callers test the result with BEQ (none free) or with A (CMP #$01).
; ----------------------------------------------------------------------------
Obj_FindEmptyMonsterSlot:
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
