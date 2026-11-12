; ============================================================================
; FILE:     bank5/scroll-world.asm       BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 552-721
; SUBJECTS: (none)
; TOPIC:    Per-frame world scrolling
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0019 ScrollWorld         code
;   L0077 ResetVScrollLo      code
;   L0081 IncSubmode          code
;   L0085 ScrollWorldDownOrH  code
;   L0117 ScrollWorldH        code
;   L0147 SwitchToNT1         code
;   L0162 L14287_Exit         code
;   L0165 ScrollWorldRight    code
; ============================================================================

ScrollWorld:
    LDA FrameCounter

    ; The purpose of this is to delay vertical scrolling.
    ; OW: once every two frames
    ; UW: once every four frames
    ; Horizontal scrolling happens every frame.
    AND #$03
    LDY CurLevel
    BNE :+
    AND #$01
:
    CMP VScrollStartFrame
    BNE ScrollWorldH
    LDA #$08                    ; If the player is facing up,
    BIT ObjDir
    BEQ ScrollWorldDownOrH

    ; then scroll up.
    ; Scrolling up starts from the top of NT 2 at $2800.
    DEC CurRow
    LDA ObjY                    ; Move the player down 1 tile length if not at edge.
    CMP #$DD
    BCS :+
    ADC #$08
    STA ObjY
:
    LDA VScrollAddrLo           ; Subtract $20 from VScroll address for a row.
    SEC
    SBC #$20
    STA VScrollAddrLo
    BCS :+
    DEC VScrollAddrHi
:
    CMP #$E0
    BNE @Exit                   ; If the result doesn't end in $E0, then return.
    LDA VScrollAddrHi
    CMP #$20
    BEQ @LimitLow               ; If the result is $20E0, then the scroll position reached the status bar. Go sanitize and keep it there.
    CMP #$27
    BNE @Exit                   ; If the result is not $27E0, then return.

    ; VScroll address is $27E0. So, we rolled from the
    ; top of NT 2 to the last row of NT 0. Change the address
    ; to $23A0, the true last row of NT 0.
    LDA #$23
    STA VScrollAddrHi
    LDA #$A0
    STA VScrollAddrLo

@Exit:
    RTS

@LimitLow:
    ; Don't let the scroll position go above $2100, the bottom of
    ; the status bar; nor below $2800, the top of NT 2.
    INC VScrollAddrHi

ResetVScrollLo:
    LDA #$00
    STA VScrollAddrLo

IncSubmode:
    INC GameSubmode             ; And we're done.
    RTS

ScrollWorldDownOrH:
    LSR                         ; If the player is facing down,
    BIT ObjDir
    BEQ ScrollWorldH

    ; then scroll down.
    INC CurRow
    LDA ObjY                    ; Move the player up 1 tile length, if not at edge.
    CMP #$3E
    BCC :+
    SBC #$08
    STA ObjY
:
    LDA VScrollAddrLo           ; Add $20 to VScroll address for a row.
    CLC
    ADC #$20
    STA VScrollAddrLo
    BCC :+
    INC VScrollAddrHi
:
    CMP #$C0
    BNE L14287_Exit             ; If the result doesn't end in $C0, then return.
    LDA VScrollAddrHi
    CMP #$23
    BNE L14287_Exit             ; If the result is not $23C0, then return.

    ; VScroll address is $23C0, the bottom of NT 0. So, we need
    ; to roll to $2800, the top of NT 2.
    LDA #$28
    STA VScrollAddrHi
    JMP ResetVScrollLo

ScrollWorldH:
    LDA #$02                    ; Use the default scroll speed: 2 pixels a frame.
    LDX #$FE
    LDY CurLevel
    BNE :+
    ASL                         ; Scroll twice as fast if in the overworld.
    LDX #$FC
:
    STA $00                     ; [00] holds the speed.
    STX $01                     ; [01] holds the X position where we must start using NT 1.
    LDA #$02                    ; If player does not face left,
    BIT ObjDir
    BEQ ScrollWorldRight        ; then go scroll right.

    ; Scroll left.
    DEC CurColumn
    LDA ObjX                    ; Move player right by [00] if not at edge.
    CMP #$F0
    BCS :+
    ADC $00
    STA ObjX
:
    LDA CurHScroll              ; Update nametable X scroll by speed in [00].
    SEC
    SBC $00
    STA CurHScroll
    BEQ IncSubmode              ; Stop when you've scrolled to the end (CurHScroll=0).
    CMP $01                     ; Does scroll position match reference scroll position [01]?
    BNE L14287_Exit             ; If not, then return.

SwitchToNT1:
    ; When we begin scrolling left, CurHScroll = 0 and base
    ; nametable is 0. This shows the current room completely.
    ;
    ; The first time in ScrollWorld, the speed [00] will be
    ; subtracted from CurHScroll to show most of NT 0, and
    ; a little of NT 1 to its left. In other words, the
    ; scroll position now refers to a position in NT 1.
    ;
    ; [01] is this first position shown in NT 1. So, once
    ; CurHScroll matches it (in the first call to this routine),
    ; turn on this flag that makes NT 1 the base nametable.
    LDA #$01
    STA OddBaseNameTableOverride    ; Start basing the horizontal scroll position on NT 1.

L14287_Exit:
    RTS

ScrollWorldRight:
    LSR
    BIT ObjDir                  ; If the player isn't facing right either,
    BEQ L14287_Exit             ; then quit.
    INC CurColumn
    LDA ObjX
    CMP #$01
    BCC :+                      ; Move player left by [00] if not at edge.
    SBC $00
    STA ObjX
:
    LDA CurHScroll              ; Update nametable X scroll by speed in [00].
    CLC
    ADC $00
    STA CurHScroll
    BNE L14287_Exit             ; If scroll position hasn't reached 0, then return.

    ; Because CurHScroll is now 0 and base nametable is 0;
    ; we would be showing the new room at the end of the scroll
    ; but with old attributes.
    ;
    ; So, now use NT 1, until after we copy the attributes.
    JSR SwitchToNT1
    JMP IncSubmode              ; Go to the next submode.
