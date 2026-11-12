; ============================================================================
; FILE:     bank4/patra-ganon-inits.asm  BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 9526-9585
; SUBJECTS: (none)
; TOPIC:    InitPatra, InitGanon
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0013 InitPatra  code
;   L0058 InitGanon  code
; ============================================================================

InitPatra:
    ; Invincible to everything but sword.
    LDA #$FE
    STA ObjInvincibilityMask, X

    ; Start at location ($80, $70) and facing up.
    LDA #$80
    STA ObjX, X
    LDA #$70
    STA ObjY, X
    LDA #$08
    STA ObjDir, X

    ; Set flying speed $1F, and maximum $40.
    LDA #$1F
    STA Flyer_ObjSpeed, X
    LDA #$40
    STA FlyerMaxSpeedFrac
    STA SampleRequest           ; Play Digdogger/Manhandla/Patra roar.

    ; Set object timer to $FF. When it runs out, the maneuver will be switched.
    LDA #$FF
    STA ObjTimer+1, X

    ; Patra type $47 goes with Patra Child type $25.
    ; Patra type $48 goes with Patra Child type $26.
    LDA #$25
    LDY ObjType+1
    CPY #$47
    BEQ :+
    LDA #$26
:
    ; Loop over object slots 2 to 9, making patra children.
    LDY #$02
    STA $00                     ; [00] holds the patra child type.
:
    LDA $00
    STA ObjType, Y
    LDA #$FE                    ; Patra children are also only weak to the sword.
    STA ObjInvincibilityMask, Y
    INY
    CPY #$0A
    BNE :-
    RTS

InitGanon:
    LDA #$FA                    ; Invincible to everything but sword and arrow.
    STA ObjInvincibilityMask, X

    ; Halt Link.
    LDA #$40
    STA ObjState

    ; Set Link's timer to $40 for scene phase 0.
    STA ObjTimer
    LDA #$02                    ; Boss hit sound effect. But here it's more of a shout.
    STA SampleRequest
    LDA #$10                    ; Play Aquamentus/Gleeok/Ganon roar.
    JSR PlaySample
    JMP ResetObjMetastateAndTimer
