; ============================================================================
; FILE:     bank7/reset-mmc1-vectors.asm BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 5972-6050
; SUBJECTS: (none)
; TOPIC:    Reset, MMC1 control, bank switch, vectors
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0021 IsrReset        code
;   L0057 SetMMC1Control  code
;   L0069 SwitchBank      code
;   L0090 IsrVector       data
; ============================================================================

.SEGMENT "BANK_07_ISR"


.EXPORT IsrReset


IsrReset:
    SEI                         ; Disable interrupts.
    CLD                         ; Clear decimal mode.
    LDA #$00
    STA PpuControl         ; Set the PPU to a base state with no NMI.
    LDX #$FF
    TXS                         ; Set the stack to $01FF.
:
    LDA PpuStatus          ; Wait for one VBLANK.
    AND #$80
    BEQ :-
:
    LDA PpuStatus          ; Wait for another VBLANK.
    AND #$80                    ;   in case the first was left over from a previous run.
    BEQ :-
    ORA #$FF
    STA $8000                   ; Reset all MMC1 shift registers.
    STA $A000
    STA $C000
    STA $E000
    LDA #$0F                    ; Set our normal mirroring and PRG ROM bank mode.
    JSR SetMMC1Control
    LDA #$00                    ; Use CHR RAM bank 0 for PPU address $00000.
    STA $A000
    LSR
    STA $A000
    LSR
    STA $A000
    LSR
    STA $A000
    LSR
    STA $A000
    LDA #$07
    JSR SwitchBank
    JMP RunGame

SetMMC1Control:
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

SwitchBank:
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




; Unknown block
    .BYTE $5A, $45, $4C, $44, $41, $D7, $C8, $00
    .BYTE $00, $38, $04, $01, $04, $01, $BE

IsrVector:
    .ADDR IsrNmi
    .ADDR IsrReset
    .ADDR $FFF0
