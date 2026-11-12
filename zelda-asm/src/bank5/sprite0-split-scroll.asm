; ============================================================================
; FILE:     bank5/sprite0-split-scroll.asm BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 1263-1367
; SUBJECTS: Scroll
; TOPIC:    The sprite-0 split: waits inside the frame handler for the sprite-0 hit at the bottom of the
;           status bar, then sets the scroll (or blanks the screen) for the play area below it. Followed
;           by 99 bytes of $FF filler.
; SEE ALSO: bank7/run-game-nmi.asm (the frame handler calls this when IsSprite0CheckActive is set);
;           bank5/mode7-scroll-init.asm (Scroll_WriteAndEnableSprite0 places sprite 0 and sets the flag);
;           bank5/scroll-world.asm (updates CurHScroll, VScrollAddrHi/Lo, OddBaseNameTableOverride);
;           bank5/mode7-scroll-update.asm (the scroll submodes; submode 7 clears the flag).
; KEY FACTS:
;   - Runs once per frame while IsSprite0CheckActive is nonzero, busy-waiting on PPU status bit 6, so
;     the status bar stays fixed while the play area scrolls. Scroll_WaitSprite0AndSplit.
;   - GameMode < 8 (room scroll, mode 7): nothing is changed while GameSubmode = 0; a horizontal scroll
;     (Link's direction 1 or 2) writes CurHScroll and nametable bit OddBaseNameTableOverride; a vertical
;     scroll (4 or 8) writes VScrollAddrHi/Lo to the PPU address register.
;     Scroll_WaitSprite0AndSplit@HorizontalSplit.
;   - GameMode 8-$10: all video is turned off below the split. GameMode >= $11: base nametable bit 0 is
;     set below the split. Scroll_WaitSprite0AndSplit@ModeAtLeast8.
;   - The delays are cycle-counted loops: about 1005 CPU cycles plus 9 NOPs after the hit, then about
;     666 more before the scroll registers are written. Scroll_WaitSprite0AndSplit.
; CONTENTS: (generated; L = line in this file)
;   L0034 Scroll_WaitSprite0AndSplit  code  Waits for the sprite-0 hit, then sets the play area's scroll for this frame (split-screen scroll).
; ============================================================================

; ----------------------------------------------------------------------------
; Waits for the sprite-0 hit, then sets the play area's scroll for this frame (split-screen scroll).
; IN:   GameMode, GameSubmode, ObjDir (slot 0), CurHScroll, OddBaseNameTableOverride, VScrollAddrHi/Lo.
; OUT:  Writes the PPU control, scroll or address registers mid-frame; CurPpuControl updated.
; NOTE: Loops until PPU status bit 6 is set: sprite 0 must be on screen over opaque background.
;       The unnamed padding after the routine fills the gap up to Mode8Continue_Init.
; ----------------------------------------------------------------------------
Scroll_WaitSprite0AndSplit:
    LDA PpuStatus               ; Wait for the sprite-0 hit (status bit 6).
    AND #$40
    BEQ Scroll_WaitSprite0AndSplit
    LDA PpuStatus

    ; Delay loop: 4 outer passes of 49 inner passes; about 1005 CPU cycles by instruction count
    ; (LDY 2, per outer pass LDX 2 + 49 * (DEX 2 + BPL 3) - 1 + DEY 2 + BPL 3, last BPL 1 less).
    LDY #$03
@DelayOuter:
    LDX #$30
:
    DEX
    BPL :-
    DEY
    BPL @DelayOuter
    NOP                         ; 9 NOPs: 18 more cycles.
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
    BCS @ModeAtLeast8           ; Modes 8 and up do not scroll.
    LDA GameSubmode
    BEQ @Exit                   ; Submode 0: leave the scroll as the frame handler set it.
    LDA ObjDir
    CMP #$04
    BCC @HorizontalSplit        ; Link faces right (1) or left (2): horizontal scroll.

    ; Vertical scroll: the PPU address register is set instead of the scroll register.
    ; Delay about 666 cycles ($5F passes of NOP, DEY, BPL).
    LDY #$5E
:
    NOP
    DEY
    BPL :-
    NOP                         ; 5 NOPs: 10 more cycles.
    NOP
    NOP
    NOP
    NOP
    LDA PpuStatus               ; Reset the PPU address latch.

    ; Point the PPU at the nametable row where the play area picture starts this frame.
    LDA VScrollAddrHi
    LDY VScrollAddrLo
    STA PpuAddr
    STY PpuAddr
    LDA PpuData                 ; Two dummy reads.
    LDA PpuData
    RTS

@HorizontalSplit:
    ; Horizontal scroll: the same delay loop, then PPU control and scroll.
    LDY #$5E
:
    NOP
    DEY
    BPL :-
    NOP
    NOP
    NOP
    LDA CurPpuControl           ; Base nametable bit 0 = OddBaseNameTableOverride (NT 0 or NT 1).
    AND #$FE
    ORA OddBaseNameTableOverride
    STA CurPpuControl
    STA PpuControl
    LDA CurHScroll              ; X scroll = CurHScroll, Y scroll = 0.
    STA PpuScroll
    LDA #$00
    STA PpuScroll

@Exit:
    RTS

@ModeAtLeast8:
    CMP #$11
    BCS :+                      ; Modes $11 and up.
    JMP Ppu_TurnOffAllVideo         ; Modes 8-$10: blank everything below the split.
:
    LDA CurPpuControl           ; Modes $11 and up: set base nametable bit 0 (NT 0 -> 1, NT 2 -> 3).
    ORA #$01
    STA CurPpuControl
    STA PpuControl
    RTS

; DATA: filler ($FF), 99 bytes, up to Mode8Continue_Init.
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
