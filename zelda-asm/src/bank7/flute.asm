; ============================================================================
; FILE:     bank7/flute.asm              BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 2496-2598
; SUBJECTS: (none)
; TOPIC:    Flute secrets table, WieldFlute
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0014 FluteRoomSecretsOW  data
;   L0018 WieldFlute          code
;   L0115 L1EFCF_Exit         code
; ============================================================================

FluteRoomSecretsOW:
    .BYTE $42, $06, $29, $2B, $30, $3A, $3C, $58
    .BYTE $60, $6E, $72

WieldFlute:
    ; Play the flute's tune.
    LDA #$10
    STA Tune1Request

    ; Set the flute timer for $98 frames.
    LDA #$98
    STA FluteTimer

    ; In UW, go flag that we used the flute, and return.
    LDA CurLevel
    BNE @FlagFluteUsed

    ; If not in mode 5, return.
    LDA GameMode
    CMP #$05
    BNE @Exit

    ; Get and save the quest number.
    LDY CurSaveSlot
    LDA QuestNumbers, Y
    PHA

    ; Look for the current room in this list of $A that have a flute secret.
    LDA RoomId
    LDY #$0A

@FindSecretRoom:
    CMP FluteRoomSecretsOW, Y
    BEQ @Found
    DEY
    BPL @FindSecretRoom
    BMI @FluteSecretNotFound

@Found:
    ; Found.
    ;
    ; If it was the first room in the list, handle it specially.
    ; This room is the only one in Q1 with a flute secret. It's also
    ; the only one in the list without a flute secret in Q2.
    ;
    ; In Q2, go summon the whirlwind. In Q1, continue to reveal
    ; the secret.
    CPY #$00
    BNE @Normal
    PLA
    BNE @SummonWhirlwind
    BEQ @RevealSecret

@Normal:
    ; For other rooms in the list:
    ; In Q1, go summon the whirlwind.
    ; In Q2, continue to reveal the secret.
    PLA
    BEQ @SummonWhirlwind

@RevealSecret:
    ; If the cycle of colors is complete or in progress, then return. The secret is already revealed.
    LDA SecretColorCycle
    BNE @Exit

    ; Look for an empty slot from 9 to 1.
    LDY #$09

@FindEmptySlot:
    DEY
    BMI @Exit
    LDA ObjType+1, Y
    BNE @FindEmptySlot

    ; If we find one, set up a flute secret object there.
    ; It will handle the rest.
    LDA #$5E
    STA ObjType+1, Y

@Exit:
    RTS

@FluteSecretNotFound:
    ; Not found.
    ; Pop and throw away the quest number.
    PLA

@SummonWhirlwind:
    ; Summon the whirlwind.
    LDA #$01
    JSR SwitchBank
    JSR SummonWhirlwind
    LDA #$05                    ; Restore the bank at the beginning of the routine.
    JMP SwitchBank

@FlagFluteUsed:
    ; Flag that we used the flute.
    LDA UsedFlute
    BNE L1EFCF_Exit
    INC UsedFlute

L1EFCF_Exit:
    RTS
