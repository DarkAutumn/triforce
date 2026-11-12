; ============================================================================
; FILE:     bank2/second-quest-bank-stub.asm BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 4034-4112
; SUBJECTS: (none)
; TOPIC:    Switch profile to second quest; ISR stub, vectors
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0014 WorldFlagBlockAddrs         data
;   L0017 SwitchProfileToSecondQuest  code
;   L0074 SwitchBank_Local2           code
; ============================================================================

WorldFlagBlockAddrs:
    .BYTE $7F, $06, $FF, $06, $7F, $07

SwitchProfileToSecondQuest:
    ; Clear $80 bytes of each block of world flags.
    LDX #$04

@ClearBlock:
    LDA WorldFlagBlockAddrs, X
    STA $00
    LDA WorldFlagBlockAddrs+1, X
    STA $01
    LDY #$7F
    LDA #$00
:
    STA ($00), Y
    DEY
    BPL :-
    DEX
    DEX
    BPL @ClearBlock

    ; Clear $28 byte block of items.
    LDY #$27
:
    STA Items, Y
    DEY
    BPL :-

    ; Set 3 full hearts and heart containers.
    LDA #$22
    STA HeartValues
    DEC HeartPartial
    LDA #$08
    STA MaxBombs
    LDY CurSaveSlot
    LDA #$01
    STA QuestNumbers, Y
    RTS

.SEGMENT "BANK_02_ISR"


.EXPORT SwitchBank_Local2


; Unknown block
    .BYTE $78, $D8, $A9, $00, $8D, $00, $20, $A2
    .BYTE $FF, $9A, $AD, $02, $20, $29, $80, $F0
    .BYTE $F9, $AD, $02, $20, $29, $80, $F0, $F9
    .BYTE $09, $FF, $8D, $00, $80, $8D, $00, $A0
    .BYTE $8D, $00, $C0, $8D, $00, $E0, $A9, $0F
    .BYTE $20, $98, $BF, $A9, $00, $8D, $00, $A0
    .BYTE $4A, $8D, $00, $A0, $4A, $8D, $00, $A0
    .BYTE $4A, $8D, $00, $A0, $4A, $8D, $00, $A0
    .BYTE $A9, $07, $20, $AC, $BF, $4C, $40, $E4
    .BYTE $8D, $00, $80, $4A, $8D, $00, $80, $4A
    .BYTE $8D, $00, $80, $4A, $8D, $00, $80, $4A
    .BYTE $8D, $00, $80, $60

SwitchBank_Local2:
    STA $E000
    LSR
    STA $E000
    LSR
    STA $E000
    LSR
    STA $E000
    LSR
    STA $E000
    RTS

.SEGMENT "BANK_02_VEC"




; Unknown block
    .BYTE $84, $E4, $50, $BF, $F0, $BF
