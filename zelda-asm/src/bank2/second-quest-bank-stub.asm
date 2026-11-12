; ============================================================================
; FILE:     bank2/second-quest-bank-stub.asm BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 4034-4112
; SUBJECTS: (none)
; TOPIC:    Resets the active profile for Q2 after the ending (world flags, inventory, hearts), then
;           bank 2's tail: its copy of the reset handler, its bank-switch routine and its vectors.
; SEE ALSO: bank2/mode13-win-game.asm (UpdateMode13WinGame_Sub4, the only caller of
;           SwitchProfileToSecondQuest); bank2/moded-save.asm (mode $D writes the profile to the save
;           file afterwards); bank7/reset-mmc1-vectors.asm (IsrReset, SetMMC1Control, SwitchBank: the
;           originals of the code copied here); bank1/map-marker-curtain.asm (caller of
;           SwitchBank_Local2).
; KEY FACTS:
;   - Winning the game zeroes all $180 world flag bytes ($067F-$07FE: overworld, levels 1-6, levels
;     7-9; WorldFlagBlockAddrs) and all $28 bytes of the items block (InvItems, $657-$67E).
;     SwitchProfileToSecondQuest
;   - Then HeartValues = $22 (3 heart containers, 2 full hearts) and HeartPartial = $FF (the third
;     heart full), InvMaxBombs = 8, and QuestNumbers for the current save slot = 1 (Q2).
;     SwitchProfileToSecondQuest
;   - Nothing else of the profile is touched here: the save slot's name and death count stay.
;     SwitchProfileToSecondQuest
;   - The quest number is set to 1, not incremented: finishing Q2 starts Q2 again.
;     SwitchProfileToSecondQuest
;   - SwitchBank_Local2 writes A to the MMC1 PRG bank register ($E000), 5 serial writes, bit 0
;     first; same code as the fixed bank's SwitchBank.
;   - The unlabeled bytes in BANK_02_ISR ($BF50) are a copy of the reset handler, and BANK_02_VEC
;     ($BFFA) a copy of the vectors: NMI $E484, reset $BF50, IRQ $BFF0. Purpose (a reset while this
;     bank is mapped at $C000-$FFFF) is (unverified).
; CONTENTS: (generated; L = line in this file)
;   L0035 WorldFlagBlockAddrs         data  TABLE: start addresses (low, high) of the three $80-byte world flag blocks: $067F, $06FF, $077F.
;   L0043 SwitchProfileToSecondQuest  code  Resets the active profile for a new game in Q2: clears world flags and items, 3 hearts, 8 bombs max.
;   L0110 SwitchBank_Local2           code  Switches the PRG bank at $8000-$BFFF to bank A (5 serial writes to MMC1 register $E000).
; ============================================================================

; TABLE: start addresses (low, high) of the three $80-byte world flag blocks: $067F, $06FF, $077F.
WorldFlagBlockAddrs:
    .BYTE $7F, $06, $FF, $06, $7F, $07

; ----------------------------------------------------------------------------
; Resets the active profile for a new game in Q2: clears world flags and items, 3 hearts, 8 bombs max.
; OUT:  QuestNumbers for CurSaveSlot = 1; HeartValues = $22; HeartPartial = $FF; InvMaxBombs = 8.
; NOTE: called once, when Start is pressed on the final screen of the ending.
; ----------------------------------------------------------------------------
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
    STA InvItems, Y
    DEY
    BPL :-

    ; Set 3 full hearts and heart containers.
    LDA #$22
    STA HeartValues
    DEC HeartPartial
    LDA #$08
    STA InvMaxBombs
    LDY CurSaveSlot
    LDA #$01
    STA QuestNumbers, Y
    RTS

.SEGMENT "BANK_02_ISR"


.EXPORT SwitchBank_Local2


; DATA: machine code without labels, bank 2's copy of the fixed bank's reset handler, at $BF50-$BFAB.
; Decoded: SEI; CLD; LDA #$00; STA $2000; LDX #$FF; TXS; wait twice for bit 7 of $2002 (vblank);
; ORA #$FF; STA $8000, $A000, $C000, $E000 (reset the MMC1 shift register); LDA #$0F; JSR $BF98
; (MMC1 control); LDA #$00 written serially to $A000 (5 writes); LDA #$07; JSR $BFAC
; (SwitchBank_Local2: fixed bank 7); JMP $E440 (in the fixed bank).
; The last 20 bytes ($BF98) are the MMC1 control routine: A written serially to $8000, then RTS.
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

; ----------------------------------------------------------------------------
; Switches the PRG bank at $8000-$BFFF to bank A (5 serial writes to MMC1 register $E000).
; IN:   A = bank number.
; NOTE: bank 2's copy; after it returns, execution continues in the newly mapped bank.
; ----------------------------------------------------------------------------
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




; DATA: bank 2's copy of the interrupt vectors at $BFFA: NMI $E484, reset $BF50, IRQ $BFF0.
    .BYTE $84, $E4, $50, $BF, $F0, $BF
