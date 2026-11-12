; ============================================================================
; FILE:     bank7/reset-mmc1-vectors.asm BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 5972-6050
; SUBJECTS: Isr, Mmc1, Reset
; TOPIC:    The reset handler, the two MMC1 serial-write routines (control register, PRG bank), the
;           cartridge footer bytes and the NMI/reset/IRQ vectors at the end of the fixed bank.
; SEE ALSO: bank7/run-game-nmi.asm (Reset_RunGame, where reset ends up; Nmi_Isr); bank1/shove-knockback.asm
;           (worked decode of a switchable bank's copy of the reset stub and vectors);
;           bank5/mode-inits-obj-lists.asm (ModeDSave_Init jumps to Reset_Handler).
; KEY FACTS:
;   - Mmc1_SwitchBank writes the low 5 bits of A, lowest bit first, to $E000: the PRG bank mapped at
;     $8000-$BFFF. Every far call in the fixed bank is "LDA #bank / JSR SwitchBank / JMP or JSR"
;     and nothing restores the previous bank automatically.
;   - Mmc1_SetControl does the same to $8000 (the mapper's control register). The game uses two
;     values: $0F (horizontal mirroring) and $0E (vertical mirroring); both keep bank 7 fixed at
;     $C000 (Reset_Handler and the mode code that calls Mmc1_SetControl).
;   - Reset: interrupts off, NMI off, stack $01FF, two waits for vertical blank, mapper registers
;     reset, control $0F, CHR register 0, bank 7 selected, then Reset_RunGame (Reset_Handler).
;   - Besides the reset vector, Reset_Handler is entered from ModeDSave_Init when Save_InitSaveRam reports that it
;     had to initialize save RAM.
;   - Segment BANK_07_ISR starts at $FF50 and BANK_07_VEC at $FFEB (Z.cfg); the vectors are NMI =
;     Nmi_Isr, reset = Reset_Handler, IRQ = $FFF0, an address inside the footer bytes (Isr_Vectors).
; CONTENTS: (generated; L = line in this file)
;   L0042 Reset_Handler    code  Reset handler: puts the CPU, PPU and mapper in a known state and jumps to Reset_RunGame.
;   L0085 Mmc1_SetControl  code  Writes the low 5 bits of A to the MMC1 control register ($8000), lowest bit first.
;   L0104 Mmc1_SwitchBank  code  Selects the PRG bank at $8000-$BFFF: writes the low 5 bits of A to $E000, lowest bit first.
;   L0131 Isr_Vectors      data  TABLE: the CPU vectors at $FFFA; index = NMI, reset, IRQ/BRK; value = handler address. The IRQ
; ============================================================================

.SEGMENT "BANK_07_ISR"


.EXPORT Reset_Handler


; ----------------------------------------------------------------------------
; Reset handler: puts the CPU, PPU and mapper in a known state and jumps to Reset_RunGame.
; NOTE: Also the target of ModeDSave_Init when save RAM had to be initialized. Each switchable bank
;       ends with its own copy of this stub and of the vectors (segments BANK_0N_ISR at $BF50 and
;       BANK_0N_VEC at $BFFA; decoded in bank1/shove-knockback.asm).
; ----------------------------------------------------------------------------
Reset_Handler:
    SEI                         ; Disable interrupts.
    CLD                         ; Clear decimal mode.
    LDA #$00
    STA PpuControl              ; PPU control 0: NMI off.
    LDX #$FF
    TXS                         ; Stack pointer = $FF (stack top $01FF).
:
    LDA PpuStatus               ; Wait for the vertical blank flag (bit 7).
    AND #$80
    BEQ :-
:
    LDA PpuStatus               ; Wait for it a second time,
    AND #$80                    ;   in case the first one was left over from before the reset.
    BEQ :-
    ORA #$FF                    ; A = $FF: a write with bit 7 set resets an MMC1 shift register.
    STA $8000
    STA $A000
    STA $C000
    STA $E000
    LDA #$0F                    ; Control = $0F: horizontal mirroring, $8000 switchable, $C000 fixed.
    JSR Mmc1_SetControl
    LDA #$00                    ; CHR bank register 0 = 0 (five serial writes of 0).
    STA $A000
    LSR
    STA $A000
    LSR
    STA $A000
    LSR
    STA $A000
    LSR
    STA $A000
    LDA #$07                    ; Map bank 7 at $8000 as well.
    JSR Mmc1_SwitchBank
    JMP Reset_RunGame

; ----------------------------------------------------------------------------
; Writes the low 5 bits of A to the MMC1 control register ($8000), lowest bit first.
; IN:   A = control value: $0F horizontal mirroring, $0E vertical mirroring (the only two used).
; OUT:  A shifted right 4 times.
; NOTE: Bits 1-0 are the mirroring; bits 3-2 = 3 in both values (PRG bank switched at $8000, last
;       bank fixed at $C000); bit 4 = 0. The bit meanings are the mapper's, not visible in the code.
; ----------------------------------------------------------------------------
Mmc1_SetControl:
    STA $8000
    LSR
    STA $8000
    LSR
    STA $8000
    LSR
    STA $8000
    LSR
    STA $8000
    RTS

; ----------------------------------------------------------------------------
; Selects the PRG bank at $8000-$BFFF: writes the low 5 bits of A to $E000, lowest bit first.
; IN:   A = bank number 0-7.
; OUT:  A shifted right 4 times (0 for banks 0-7); X and Y preserved.
; NOTE: Callers in the fixed bank do not restore the old bank; code that returns into a
;       switchable bank must select that bank again itself.
; ----------------------------------------------------------------------------
Mmc1_SwitchBank:
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

.SEGMENT "BANK_07_VEC"




; DATA: cartridge footer at $FFEB-$FFF9 (15 bytes, no label, read by no code). The first five
; bytes are the ASCII text "ZELDA". $FFF0-$FFF1 = $D7C8 is the 16-bit sum of the other 16,382
; bytes of this bank (computed from the ROM image). The eight bytes $FFF2-$FFF9 add up to $00 mod
; $100. (unverified) What $FFF2-$FFF8 ($00, $00, $38, $04, $01, $04, $01) stand for: no code
; reads them, so the tree cannot decide it; $04 at $FFF7 equals the title length minus 1.
    .BYTE $5A, $45, $4C, $44, $41, $D7, $C8, $00    ; "ZELDA" at $FFEB; $D7 is at $FFF0
    .BYTE $00, $38, $04, $01, $04, $01, $BE

; TABLE: the CPU vectors at $FFFA; index = NMI, reset, IRQ/BRK; value = handler address. The IRQ
; entry $FFF0 points into the footer bytes above; Reset_Handler sets the interrupt-disable flag (SEI).
Isr_Vectors:
    .ADDR Nmi_Isr
    .ADDR Reset_Handler
    .ADDR $FFF0
