; ============================================================================
; FILE:     bank5/sprite0-split-scroll.asm BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 1263-1367
; SUBJECTS: (none)
; TOPIC:    Sprite-0 split; padding
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0012 WaitAndScrollToSplitBottom  code
; ============================================================================

WaitAndScrollToSplitBottom:
    LDA PpuStatus          ; Wait for Sprite 0 Hit.
    AND #$40
    BEQ WaitAndScrollToSplitBottom
    LDA PpuStatus

    ; Wait cycles.
    ; TODO: Why do these differ?
    ; (cycle at 8535 - cycle at 852b) = 1005  (see CPU status in debugger)
    ; (multiply instruction timing appropriately) = 1010
    LDY #$03
:
    LDX #$30
:
    DEX
    BPL :-
    DEY
    BPL :--
    NOP                         ; Wait 18 cycles.
    NOP
    NOP
    NOP
    NOP
    NOP
    NOP
    NOP
    NOP
    LDA GameMode
    CMP #$08
    BCS :++++
    LDA GameSubmode
    BEQ @Exit                   ; If GameSubmode is 0, then return.
    LDA ObjDir
    CMP #$04
    BCC :++                     ; TODO: If facing horizontally, then go update scroll registers.

    ; Scrolling vertically, will set PPUADDR instead of PPUSCROLL.
    ; Wait about 666 cycles.
    LDY #$5E
:
    NOP
    DEY
    BPL :-
    NOP                         ; Wait 10 cycles.
    NOP
    NOP
    NOP
    NOP
    LDA PpuStatus          ; Prepare for writing to PPUADDR.

    ; Set PPUADDR mid-frame to 
    ; achieve split-frame scrolling.
    LDA VScrollAddrHi
    LDY VScrollAddrLo
    STA PpuAddr
    STY PpuAddr
    LDA PpuData
    LDA PpuData
    RTS
:
    LDY #$5E
:
    NOP
    DEY
    BPL :-
    NOP
    NOP
    NOP
    LDA CurPpuControl
    AND #$FE
    ORA OddBaseNameTableOverride
    STA CurPpuControl
    STA PpuControl
    LDA CurHScroll
    STA PpuScroll
    LDA #$00
    STA PpuScroll

@Exit:
    RTS
:
    CMP #$11
    BCS :+                      ; If GameMode < $11,
    JMP TurnOffAllVideo         ; then turn off video and return.
:
    LDA CurPpuControl      ; else change base nametable (0 -> 1 or 2 -> 3).
    ORA #$01
    STA CurPpuControl
    STA PpuControl
    RTS

; Unknown block
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF
