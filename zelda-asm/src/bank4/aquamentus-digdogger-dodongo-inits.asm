; ============================================================================
; FILE:     bank4/aquamentus-digdogger-dodongo-inits.asm BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 4842-4911
; SUBJECTS: Aquamentus, Digdogger, Dodongo
; TOPIC:    Init routines of Aquamentus (type $3D), the big Digdogger (types $38, $39; also run for
;           each small Digdogger, type $18) and Dodongo (types $31, $32): roar, start position or
;           direction, immunities, Digdogger's speed and child count.
; SEE ALSO: bank4/aquamentus.asm (Aquamentus_Update); bank4/digdogger.asm (update, the flute split
;           that calls Digdogger_InitWith3Children for each child); bank4/dodongo.asm (Dodongo_Update);
;           bank7/object-init-table.asm (Obj_Init: attribute byte, HP, then these routines);
;           bank4/fairy-flyer-common.asm (Dir_EightWayDirs).
; KEY FACTS:
;   - Aquamentus's invincibility mask is $E2: the boomerang ($02) and fire ($20) cannot hurt it;
;     sword, arrow, bomb and magic shot can. Aquamentus_Init
;   - Aquamentus always starts at ($B0, $80), whatever spot the room gave it. Aquamentus_Init
;   - Digdogger type $38 splits into 3 small Digdoggers, type $39 into 1 (ChildDigdoggerCount).
;     Type $39 is rewritten to $38 at init, so the two differ only in that count.
;     Digdogger_InitWith3Children, Digdogger_InitWith1Child
;   - Digdogger starts in a random 8-way direction (Random,X AND 7) with speed low byte $3F and
;     target speed low byte $80, in 1/256 pixel per frame. Digdogger_InitWith3Children
;   - Dodongo starts facing right (1) when Random,X < $80, else left (2); both types use the same
;     init. Dodongo_Init
;   - Each init requests its roar through SampleRequest: $10 Aquamentus, $40 Digdogger, $20
;     Dodongo. Aquamentus_Init, Digdogger_InitWith3Children, Dodongo_Init
; CONTENTS: (generated; L = line in this file)
;   L0040 Aquamentus_Init              code  Initializes Aquamentus: immunity to boomerang and fire, roar, fixed start position ($B0, $80).
;   L0065 Digdogger_InitWith3Children  code  Initializes a Digdogger that splits into 3 (type $38); also the common init of every small one.
;   L0097 Digdogger_InitWith1Child     code  Initializes a Digdogger that splits into 1 (type $39), then turns it into type $38.
;   L0116 Dodongo_Init                 code  Initializes a Dodongo (type $31 or $32): roar, and a random facing direction, right or left.
; ============================================================================

; ----------------------------------------------------------------------------
; Initializes Aquamentus: immunity to boomerang and fire, roar, fixed start position ($B0, $80).
; IN:   X = object slot.
; OUT:  writes ObjInvincibilityMask,X = $E2, ObjX,X = $B0, ObjY,X = $80, SampleRequest = $10.
; NOTE: HP is $60 and the attribute byte $89 (type $3D in Obj_TypeHpPairs and
;       Obj_TypeAttributes); touching it costs Link $0100 = 1 heart before rings
;       (Damage_ObjTypeDamageToLink). The metastate is left alone, so the spawn cloud plays first.
; ----------------------------------------------------------------------------
Aquamentus_Init:
    ; Invincibility mask $E2: bit $02 boomerang and bit $20 fire cannot hurt it.
    ; (Bits $40 and $80 are set too; no damage type is known for them.)
    LDA #$E2
    STA ObjInvincibilityMask, X
    LDA #$10                    ; Request the roar sample shared by Aquamentus, Gleeok and Ganon.
    STA SampleRequest

    ; Start at ($B0, $80).
    LDA #$B0
    STA ObjX, X
    LDA #$80
    STA ObjY, X
    RTS

; ----------------------------------------------------------------------------
; Initializes a Digdogger that splits into 3 (type $38); also the common init of every small one.
; IN:   X = object slot.
; OUT:  writes ObjDir,X (random 8-way), Digdogger_ObjSpeedFrac,X = $3F,
;       Digdogger_ObjTargetSpeedFrac,X = $80, ChildDigdoggerCount = 3, SampleRequest = $40.
; NOTE: The speed and target speed high bytes are not written here. The room reset zeroes them;
;       the split code, which calls this for each child, then adds 1 to the child's target high
;       byte. So each child requests the roar too and the child count is set to 3 again.
;       HP is $F0 for types $38 and $39 (Obj_TypeHpPairs).
; ----------------------------------------------------------------------------
Digdogger_InitWith3Children:
    LDA #$40                    ; Request the roar sample shared by Digdogger, Manhandla and Patra.
    STA SampleRequest

    ; Random 8-way direction.
    LDA Random, X
    AND #$07
    TAY
    LDA Dir_EightWayDirs, Y
    STA ObjDir, X

    ; Speed low byte $3F (16-bit speed in 1/256 pixel per frame; high byte not written here).
    LDA #$3F
    STA Digdogger_ObjSpeedFrac, X

    ; Target speed low byte $80 (high byte not written here).
    LDA #$80
    STA Digdogger_ObjTargetSpeedFrac, X

    ; It will split into 3 small Digdoggers.
    LDA #$03
    STA ChildDigdoggerCount
    RTS

; ----------------------------------------------------------------------------
; Initializes a Digdogger that splits into 1 (type $39), then turns it into type $38.
; IN:   X = object slot.
; OUT:  as Digdogger_InitWith3Children, but ObjType,X = $38 and ChildDigdoggerCount = 1.
; NOTE: Types $38 and $39 have the same update routine (Obj_Update_JumpTable) and the same
;       attribute, HP and contact damage table values; no effect of the type change was found
;       (unverified).
; ----------------------------------------------------------------------------
Digdogger_InitWith1Child:
    JSR Digdogger_InitWith3Children

    ; Change the object type from $39 to $38.
    LDA #$38
    STA ObjType, X

    ; It will split into 1 small Digdogger.
    LDA #$01
    STA ChildDigdoggerCount
    RTS

; ----------------------------------------------------------------------------
; Initializes a Dodongo (type $31 or $32): roar, and a random facing direction, right or left.
; IN:   X = object slot.
; OUT:  writes ObjDir,X = 1 (right) or 2 (left), SampleRequest = $20.
; NOTE: Position and metastate are left alone: Dodongo appears from a spawn cloud at the spot the
;       room gave it. Touching it costs Link $0100 = 1 heart before rings (Damage_ObjTypeDamageToLink).
; ----------------------------------------------------------------------------
Dodongo_Init:
    LDA #$20                    ; Request the roar sample shared by Dodongo and Gohma.
    STA SampleRequest

    ; Face right (1) if Random,X < $80, else left (2).
    LDY #$01
    LDA Random, X
    CMP #$80
    BCC :+
    INY
:
    STY ObjDir, X
    RTS
