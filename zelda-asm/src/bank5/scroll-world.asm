; ============================================================================
; FILE:     bank5/scroll-world.asm       BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 552-721
; SUBJECTS: Mode7Scroll
; TOPIC:    The per-frame step of the room-to-room scroll (mode 7): moves the scroll position by one
;           tile row (vertical) or 2/4 pixels (horizontal), carries Link across the screen, and ends
;           the scroll submode when the next room is fully shown.
; SEE ALSO: bank5/mode7-scroll-update.asm (Mode7Scroll_Update_Sub0 sets the start values, Sub2 sets
;           VScrollStartFrame, Sub3 calls Mode7Scroll_ScrollWorld every frame and copies the row or column);
;           bank5/mode7-scroll-init.asm (draws the next room before the scroll);
;           bank5/sprite0-split-scroll.asm (applies VScrollAddrHi/Lo, CurHScroll and
;           OddBaseNameTableOverride below the status bar).
; KEY FACTS:
;   - The scroll direction is Link's facing, ObjDir of slot 0 (Mode7Scroll_ScrollWorld, Mode7Scroll_ScrollWorldH).
;   - Horizontal scroll: every frame, 4 pixels in OW (64 frames per room), 2 pixels in UW
;     (128 frames) (Mode7Scroll_ScrollWorldH).
;   - Vertical scroll: one tile row (8 pixels) every 2nd frame in OW, every 4th frame in UW, on
;     the frames where the masked FrameCounter equals VScrollStartFrame (Mode7Scroll_ScrollWorld).
;   - Scrolling down takes 22 steps ($2100 to $23C0), scrolling up 23 (22 rows plus the step that
;     detects $20E0): 85 / 89 frames in UW, 43 / 45 in OW, the first step on the first frame
;     (Mode7Scroll_ScrollWorld, Mode7Scroll_ScrollWorldDownOrH; Mode7Scroll_Update_Sub2 sets the phase).
;   - Link is carried with the picture: Y +8 per step while Y < $DD (up), Y -8 while Y >= $3E
;     (down), X + speed while X < $F0 (left), X - speed while X >= 1 (right) (Mode7Scroll_ScrollWorld,
;     Mode7Scroll_ScrollWorldDownOrH, Mode7Scroll_ScrollWorldH, Mode7Scroll_ScrollWorldRight).
;   - CurRow (vertical) or CurColumn (horizontal) is stepped here; the caller copies that row or
;     column of the new room to the name table (Mode7Scroll_ScrollWorld, Mode7Scroll_ScrollWorldH).
; CONTENTS: (generated; L = line in this file)
;   L0046 Mode7Scroll_ScrollWorld             code  Advances the room scroll by one step in the direction Link faces (mode 7 update, submode 3).
;   L0106 Mode7Scroll_ResetVScrollLo          code  Clears VScrollAddrLo, then ends the scroll submode (shared by the up and down scrolls).
;   L0111 Mode7Scroll_ScrollWorld_IncSubmode  code  Advances GameSubmode: the scroll is complete (vertical or horizontal).
;   L0116 Mode7Scroll_ScrollWorldDownOrH      code  Vertical step frame, Link not facing up: scrolls down one row if he faces down, else horizontally.
;   L0154 Mode7Scroll_ScrollWorldH            code  Horizontal part of the scroll step: scrolls left if Link faces left, else tries right.
;   L0185 Mode7Scroll_SwitchToNT1             code  Sets OddBaseNameTableOverride = 1 so that the horizontal scroll position is based on NT 1.
;   L0201 Mode7Scroll_ScrollWorld_Return      code  Shared exit of the scroll routines in this file.
;   L0205 Mode7Scroll_ScrollWorldRight        code  Scrolls right by [00] pixels if Link faces right; otherwise does nothing.
; ============================================================================

; ----------------------------------------------------------------------------
; Advances the room scroll by one step in the direction Link faces (mode 7 update, submode 3).
; IN:   ObjDir (slot 0) = scroll direction; VScrollStartFrame = FrameCounter phase of vertical steps.
; OUT:  VScrollAddrHi/Lo or CurHScroll moved; CurRow or CurColumn stepped; ObjX or ObjY of Link
;       moved; GameSubmode + 1 when the scroll is complete. Uses [00], [01] (horizontal).
; NOTE: Vertical: 8 pixels per step, a step every 2nd frame in OW and every 4th in UW.
;       Horizontal: 4 pixels per frame in OW, 2 in UW.
; ----------------------------------------------------------------------------
Mode7Scroll_ScrollWorld:
    LDA FrameCounter

    ; Vertical scrolling moves only on the frames where the low bits
    ; of FrameCounter equal VScrollStartFrame:
    ; OW: once every two frames (mask 1)
    ; UW: once every four frames (mask 3)
    ; Horizontal scrolling happens every frame.
    AND #$03
    LDY CurLevel
    BNE :+
    AND #$01
:
    CMP VScrollStartFrame
    BNE Mode7Scroll_ScrollWorldH            ; Not a vertical step frame. Only a horizontal scroll moves.
    LDA #$08                    ; If Link is facing up,
    BIT ObjDir
    BEQ Mode7Scroll_ScrollWorldDownOrH

    ; then scroll up.
    ; Scrolling up starts from the top of NT 2 at $2800.
    DEC CurRow
    LDA ObjY                    ; Move Link down 1 tile (8 pixels) while his Y < $DD.
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
    BNE @Exit                   ; If the low byte isn't $E0, then return.
    LDA VScrollAddrHi
    CMP #$20
    BEQ @LimitLow               ; $20E0: passed the top of the play area in NT 0. Go finish.
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
    ; The address went one row above $2100, the top of the play area
    ; (the bottom of the status bar). Put it back at $2100 and end the scroll.
    INC VScrollAddrHi

; Clears VScrollAddrLo, then ends the scroll submode (shared by the up and down scrolls).
Mode7Scroll_ResetVScrollLo:
    LDA #$00
    STA VScrollAddrLo

; Advances GameSubmode: the scroll is complete (vertical or horizontal).
Mode7Scroll_ScrollWorld_IncSubmode:
    INC GameSubmode
    RTS

; Vertical step frame, Link not facing up: scrolls down one row if he faces down, else horizontally.
Mode7Scroll_ScrollWorldDownOrH:
    LSR                         ; A = 4. If Link is facing down,
    BIT ObjDir
    BEQ Mode7Scroll_ScrollWorldH

    ; then scroll down.
    INC CurRow
    LDA ObjY                    ; Move Link up 1 tile (8 pixels) while his Y >= $3E.
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
    BNE Mode7Scroll_ScrollWorld_Return             ; If the low byte isn't $C0, then return.
    LDA VScrollAddrHi
    CMP #$23
    BNE Mode7Scroll_ScrollWorld_Return             ; If the result is not $23C0, then return.

    ; VScroll address is $23C0, just past the bottom row of NT 0. So, roll
    ; to $2800, the top of NT 2, where the next room was drawn, and end the scroll.
    LDA #$28
    STA VScrollAddrHi
    JMP Mode7Scroll_ResetVScrollLo

; ----------------------------------------------------------------------------
; Horizontal part of the scroll step: scrolls left if Link faces left, else tries right.
; OUT:  [00] = pixels per frame (4 in OW, 2 in UW); [01] = CurHScroll value after the first step
;       left ($FC in OW, $FE in UW).
; NOTE: A full room is 256 pixels: 64 frames in OW, 128 frames in UW.
; ----------------------------------------------------------------------------
Mode7Scroll_ScrollWorldH:
    LDA #$02                    ; Use the default scroll speed: 2 pixels a frame.
    LDX #$FE
    LDY CurLevel
    BNE :+
    ASL                         ; Scroll twice as fast (4 pixels a frame) in OW.
    LDX #$FC
:
    STA $00                     ; [00] holds the speed.
    STX $01                     ; [01] holds the scroll position where we must start using NT 1.
    LDA #$02                    ; If Link does not face left,
    BIT ObjDir
    BEQ Mode7Scroll_ScrollWorldRight        ; then go scroll right.

    ; Scroll left.
    DEC CurColumn
    LDA ObjX                    ; Move Link right by [00] while his X < $F0.
    CMP #$F0
    BCS :+
    ADC $00
    STA ObjX
:
    LDA CurHScroll              ; Move the horizontal scroll position left by the speed in [00].
    SEC
    SBC $00
    STA CurHScroll
    BEQ Mode7Scroll_ScrollWorld_IncSubmode              ; The scroll is complete when CurHScroll returns to 0.
    CMP $01                     ; Is this the first step (scroll position = [01])?
    BNE Mode7Scroll_ScrollWorld_Return             ; If not, then return.

; Sets OddBaseNameTableOverride = 1 so that the horizontal scroll position is based on NT 1.
Mode7Scroll_SwitchToNT1:
    ; When we begin scrolling left, CurHScroll = 0 and base
    ; nametable is 0. This shows the current room completely.
    ;
    ; The first time in Mode7Scroll_ScrollWorld, the speed [00] is
    ; subtracted from CurHScroll to show most of NT 0, and
    ; a little of NT 1 to its left. In other words, the
    ; scroll position now refers to a position in NT 1.
    ;
    ; [01] is this first position shown in NT 1. So, once
    ; CurHScroll matches it (in the first call to this routine),
    ; turn on this flag that makes NT 1 the base nametable.
    LDA #$01
    STA OddBaseNameTableOverride

; Shared exit of the scroll routines in this file.
Mode7Scroll_ScrollWorld_Return:
    RTS

; Scrolls right by [00] pixels if Link faces right; otherwise does nothing.
Mode7Scroll_ScrollWorldRight:
    LSR                         ; A = 1.
    BIT ObjDir                  ; If Link isn't facing right either,
    BEQ Mode7Scroll_ScrollWorld_Return             ; then return.
    INC CurColumn
    LDA ObjX
    CMP #$01
    BCC :+                      ; Move Link left by [00] while his X >= 1.
    SBC $00
    STA ObjX
:
    LDA CurHScroll              ; Move the horizontal scroll position right by the speed in [00].
    CLC
    ADC $00
    STA CurHScroll
    BNE Mode7Scroll_ScrollWorld_Return             ; If the scroll position hasn't wrapped to 0, then return.

    ; Because CurHScroll is now 0 and base nametable is 0;
    ; we would be showing the new room at the end of the scroll
    ; but with old attributes.
    ;
    ; So, now use NT 1, until after the attributes are copied
    ; (Mode7Scroll_Update_Sub4And5_TransferAttrs clears the flag).
    JSR Mode7Scroll_SwitchToNT1
    JMP Mode7Scroll_ScrollWorld_IncSubmode              ; Go to the next submode.
