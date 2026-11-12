; ============================================================================
; FILE:     bank4/aquamentus-digdogger-dodongo-inits.asm BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 4842-4911
; SUBJECTS: (none)
; TOPIC:    Inits for Aquamentus, Digdogger, Dodongo
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0015 InitAquamentus  code
;   L0029 InitDigdogger1  code
;   L0057 InitDigdogger2  code
;   L0072 InitDodongo     code
; ============================================================================

InitAquamentus:
    ; Aquamentus is invincible to boomerang and fire.
    LDA #$E2
    STA ObjInvincibilityMask, X
    LDA #$10                    ; Play Aquamentus/Gleeok/Ganon roar.
    STA SampleRequest

    ; The monster goes at ($B0, $80).
    LDA #$B0
    STA ObjX, X
    LDA #$80
    STA ObjY, X
    RTS

InitDigdogger1:
    LDA #$40                    ; Play Digdogger/Manhandla/Patra roar.
    STA SampleRequest

    ; Set a random 8-way direction.
    LDA Random, X
    AND #$07
    TAY
    LDA Directions8, Y
    STA ObjDir, X

    ; Set low speed byte to $3F.
    ; This assumes that the high byte is 0 from when the room
    ; was reset; or will bet set during digdogger split-up.
    LDA #$3F
    STA Digdogger_ObjSpeedFrac, X

    ; Set low target speed byte to $80.
    ; This assumes that the high byte is 0 from when the room
    ; was reset; or will bet set during digdogger split-up.
    LDA #$80
    STA Digdogger_ObjTargetSpeedFrac, X

    ; Set child count of 3.
    LDA #$03
    STA ChildDigdoggerCount
    RTS

InitDigdogger2:
    JSR InitDigdogger1

    ; Change object type to $38 (Digdogger1).
    ;
    ; UNKNOWN:
    ; Why? Everything seems to work as usual, if you leave it as $39.
    LDA #$38
    STA ObjType, X

    ; Set child count of 1.
    LDA #$01
    STA ChildDigdoggerCount
    RTS

InitDodongo:
    LDA #$20                    ; Play Dodongo/Gohma roar.
    STA SampleRequest

    ; Randomly face left or right.
    LDY #$01
    LDA Random, X
    CMP #$80
    BCC :+
    INY
:
    STY ObjDir, X
    RTS
