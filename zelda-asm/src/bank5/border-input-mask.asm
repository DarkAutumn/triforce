; ============================================================================
; FILE:     bank5/border-input-mask.asm  BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 2720-2864
; SUBJECTS: Link
; TOPIC:    Border bounds masking the A button and directions
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0031 BorderBounds       data  TRIFORCE: Screen lock boundaries — Python: Link.is_sword_screen_locked, get_sword_directions_allowed
;   L0035 Link_FilterInput   code
;   L0101 MaskInputInBorder  code  Params:
;   L0158 L14DFF_Exit        code
; ============================================================================

; TRIFORCE: Screen lock boundaries — Python: Link.is_sword_screen_locked, get_sword_directions_allowed
; TRIFORCE: OW bounds (first 4): down=$D6, up=$45, right=$E9, left=$07
; TRIFORCE: UW bounds (next 4):  down=$C6, up=$55, right=$D9, left=$17
; TRIFORCE: Inner bounds (last 4): down=$BE, up=$54, right=$D1, left=$1F
; TRIFORCE: Inner bounds mask A button (sword). Outer bounds mask directions.
; TRIFORCE: Direction-dependent: checks facing dir vs boundary. Only blocks movement TOWARD edge.
; TRIFORCE: Python approximation: checks position against bounds regardless of direction (known approx).
;
; Three sets of border coordinates:
; - outer OW
; - outer UW
; - inner
;
; Within each set, the coordinates are arranged:
; down, up, right, left
;
BorderBounds:
    .BYTE $D6, $45, $E9, $07, $C6, $55, $D9, $17
    .BYTE $BE, $54, $D1, $1F

Link_FilterInput:
    ; [00] holds the opposite of Link's facing direction.
    LDA ObjDir
    JSR GetOppositeDir
    STA $00

    ; Call GetOppositeDir again to use this mapping:
    ; In Dir: 1 2 4 8
    ;         -------
    ; Index:  2 3 0 1
    JSR GetOppositeDir

    ; Get the appropriate coordinate for the direction:
    ; X for horizontal
    ; Y for vertical
    LDA ObjX
    CPY #$02
    BCS :+
    LDA ObjY
:
    STA $02                     ; [02] holds the coordinate (X or Y).

    ; Mask A button and directions if inner border is crossed.
    TYA
    PHA                         ; Save reverse direction index.
    CLC
    ADC #$08                    ; Add 8 for the third set of border bounds. Each is 4 bytes.
    TAY
    LDA #$80                    ; Button A will be masked off, if boundary is crossed.
    JSR MaskInputInBorder
    PLA                         ; Restore reverse direction index.
    TAY

    ; If inner boundary was not crossed, then we'll check the outer
    ; boundary using player's direction and reverse direction index.
    LDA $01
    CMP #$FF
    BNE :+
    LDA ObjDir
    STA $00                     ; [00] now holds facing direction instead of the opposite.
    JSR GetOppositeDir          ; Get the reverse direction index for the actual facing.
:
    ; If in UW, use the second set of bounds by adding 4
    ; to reverse direction index.
    ;
    ; Else in OW, and use first set that begins at offset 0.
    LDA CurLevel
    BEQ :+
    TYA
    CLC
    ADC #$04
    TAY
:
    ; We won't mask buttons (A) this time. Only directions now.
    LDA #$00

; Params:
; A: mask of buttons to block
; Y: offset into BorderBounds
; [00]: direction
; [02]: coordinate
;
; Returns:
; [01]: button mask that was used
;       (set to $FF if boundary not crossed)
;
MaskInputInBorder:
    STA $01

    ; If direction is positive (right or down), then go handle it separately.
    LDA $00
    AND #$0A
    BEQ @PositiveDir

    ; The direction is negative (left or up).
    ;
    ; If the coordinate does not cross (<) the boundary, then
    ; go set mask $FF, so that no button is excluded.
    LDA $02
    CMP BorderBounds, Y
    BCC @ExcludeNone

@MaskButtons:
    ; The boundary was crossed. Mask buttons.
    LDA ButtonsPressed
    AND $01                     ; [01] button mask
    STA ButtonsPressed

    ; If in OW or button mask <> 0, return.
    LDA CurLevel
    BEQ @Exit
    LDA $01
    BNE @Exit

    ; Mask off input directions perpendicular to player's facing.
    LDY #$0C
    LDA ObjDir
    AND #$0C
    BNE :+
    LDY #$03
:
    TYA
    AND ObjInputDir
    STA ObjInputDir

@Exit:
    RTS

@PositiveDir:
    ; The direction is positive.
    ;
    ; If the coordinate crosses (<) the boundary, then
    ; go mask buttons.
    LDA $02                     ; Get coordinate in [02].
    CMP BorderBounds, Y
    BCC @MaskButtons

@ExcludeNone:
    ; The coordinate is not crossed; then set mask $FF,
    ; so that no button is excluded.
    LDA #$FF
    STA $01                     ; [01] holds button mask.

L14DFF_Exit:
    RTS
