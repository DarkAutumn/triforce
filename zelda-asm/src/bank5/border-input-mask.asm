; ============================================================================
; FILE:     bank5/border-input-mask.asm  BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 2720-2864
; SUBJECTS: Link
; TOPIC:    Border bands near the screen edges in which Link's newly pressed A and B buttons are discarded
;           (no sword, no item) and, in dungeons, sideways input is removed.
; SEE ALSO: bank5/link-handle-input.asm (Link_HandleInput calls Link_FilterInput, then tests A and B);
;           bank5/wield-items.asm (Sword_Wield, Link_WieldItem: what the surviving buttons start);
;           bank7/walker-collision-screen-edge.asm (Link_CheckScreenEdge: the room change at the edge itself);
;           bank5/doorway.asm (Link_ModifyDirInDoorway: the other dungeon restriction near a door).
; KEY FACTS:
;   - Link_FilterInput runs once a frame, only while Link's state (ObjState) is 0, before the A and B
;     tests of Link_HandleInput. Only the coordinate on Link's facing axis is tested (X when he faces left
;     or right, Y when he faces up or down).
;   - Inner band, both worlds, Link facing the edge (Link_BorderBounds+8): facing down Y >= $BE, up Y < $54,
;     right X >= $D1, left X < $1F. ButtonsPressed is ANDed with $80: A survives, B and the rest are
;     cleared, so the sword works and the B item does not.
;   - Outer band, Link facing the edge: overworld down Y >= $D6, up Y < $45, right X >= $E9, left X < $07;
;     dungeon down Y >= $C6, up Y < $55, right X >= $D9, left X < $17. ButtonsPressed is cleared: no
;     sword either.
;   - Outer band behind Link (he faces away from the edge he stands at, e.g. just after entering a room):
;     same bounds, same result: ButtonsPressed cleared. Facing parallel to the nearby edge masks nothing.
;   - Dungeons only (CurLevel <> 0), outer band only: ObjInputDir is ANDed with Link's facing axis ($0C or
;     $03), so a held perpendicular direction is dropped. In the overworld ObjInputDir is never changed
;     here (Link_MaskInputInBorder@MaskButtons).
;   - Only ButtonsPressed of controller 1 (newly pressed this frame) is changed; ButtonsDown is not.
; CONTENTS: (generated; L = line in this file)
;   L0038 Link_BorderBounds              data  TABLE: border coordinates; index = set * 4 + side; value = X or Y coordinate of Link (pixels).
;   L0054 Link_FilterInput               code  Discards Link's newly pressed buttons (and, in dungeons, sideways input) near the screen edges.
;   L0119 Link_MaskInputInBorder         code  Tests one coordinate of Link against one border bound and masks ButtonsPressed when it is reached.
;   L0172 Link_MaskInputInBorder_Return  code  Shared exit: the RTS of Link_MaskInputInBorder, also branched to by Sword_Wield.
; ============================================================================

; TABLE: border coordinates; index = set * 4 + side; value = X or Y coordinate of Link (pixels).
;   Sets: 0 outer overworld, 1 outer dungeon, 2 inner (both worlds). Sides in each set: down, up, right,
;   left. Down and right bounds are reached at coordinate >= bound; up and left at coordinate < bound.
;   The inner set discards every newly pressed button but A; the outer sets discard A as well.
Link_BorderBounds:
    .BYTE $D6, $45, $E9, $07, $C6, $55, $D9, $17    ; Outer OW: D, U, R, L. Outer UW: D, U, R, L.
    .BYTE $BE, $54, $D1, $1F                        ; Inner: D, U, R, L.

; ----------------------------------------------------------------------------
; Discards Link's newly pressed buttons (and, in dungeons, sideways input) near the screen edges.
; IN:   ObjDir = Link's facing direction; ObjX, ObjY; ButtonsPressed; CurLevel.
; OUT:  ButtonsPressed ANDed with $80 (inner band, facing the edge) or cleared (outer band, facing the
;       edge or with the edge behind); in a dungeon's outer band ObjInputDir keeps only the facing axis.
;       Uses [00], [01], [02].
; NOTE: Two tests. 1) Inner band on the side Link faces. 2) Outer band: on the side Link faces when test
;       1 hit, else on the side behind him. Only the coordinate on the facing axis is looked at.
;       Called by Link_HandleInput only while Link's state is 0.
; NOTE: Every outer bound lies inside its inner band except the dungeon's up bound ($55 against $54):
;       facing up in a dungeon at Y = $54 test 1 misses, so nothing on the facing side is masked there.
; ----------------------------------------------------------------------------
Link_FilterInput:
    ; [00] holds the opposite of Link's facing direction. Link_MaskInputInBorder takes "left or up" in [00]
    ; as "test coordinate >= bound", so the opposite direction selects the test for the side he faces.
    LDA ObjDir
    JSR Dir_GetOpposite
    STA $00

    ; Call Dir_GetOpposite again, on the opposite direction, for its index in Y. The result is the
    ; side index of the direction Link faces, in Link_BorderBounds order:
    ; Facing: 1 (right) 2 (left) 4 (down) 8 (up)
    ; Y:      2         3        0        1
    JSR Dir_GetOpposite

    ; Get the coordinate on the facing axis:
    ; X for horizontal (Y >= 2), Y for vertical.
    LDA ObjX
    CPY #$02
    BCS :+
    LDA ObjY
:
    STA $02                     ; [02] holds the coordinate (X or Y).

    ; Test 1: inner band on the side Link faces.
    TYA
    PHA                         ; Save the side index.
    CLC
    ADC #$08                    ; Offset 8 = the third (inner) set of bounds.
    TAY
    LDA #$80                    ; AND mask for ButtonsPressed: only A ($80) is kept.
    JSR Link_MaskInputInBorder
    PLA                         ; Restore the side index.
    TAY

    ; [01] = $FF means Link is not in the inner band on the side he faces. Then test the outer band
    ; on the side behind him instead: [00] becomes the facing direction, and Y the index
    ; Dir_GetOpposite gives for it (up 0, down 1, left 2, right 3), which in Link_BorderBounds order
    ; (down, up, right, left) is the opposite side.
    ; Otherwise [00] and Y stay, and the outer band is tested on the side he faces.
    LDA $01
    CMP #$FF
    BNE :+
    LDA ObjDir
    STA $00
    JSR Dir_GetOpposite
:
    ; In a dungeon use the second set of bounds (offset 4); in the overworld the first (offset 0).
    LDA CurLevel
    BEQ :+
    TYA
    CLC
    ADC #$04
    TAY
:
    ; Test 2: outer band. AND mask 0 discards every newly pressed button, A included.
    LDA #$00

; ----------------------------------------------------------------------------
; Tests one coordinate of Link against one border bound and masks ButtonsPressed when it is reached.
; IN:   A = AND mask for ButtonsPressed ($80 keeps only A; 0 clears all); Y = offset into Link_BorderBounds;
;       [00] = direction: left or up (bits $0A) tests coordinate >= bound, right or down tests
;       coordinate < bound; [02] = the coordinate.
; OUT:  [01] = the mask if the bound was reached, $FF if not. When reached: ButtonsPressed ANDed with
;       the mask; and if in a dungeon with mask 0, ObjInputDir ANDed with Link's facing axis.
; NOTE: In the overworld (CurLevel = 0) ObjInputDir is never touched.
; ----------------------------------------------------------------------------
Link_MaskInputInBorder:
    STA $01

    ; If [00] is right or down, go test "coordinate < bound".
    LDA $00
    AND #$0A
    BEQ @PositiveDir

    ; [00] is left or up: the bound is reached when coordinate >= bound.
    ; If coordinate < bound, go report $FF and change nothing.
    LDA $02
    CMP Link_BorderBounds, Y
    BCC @ExcludeNone

@MaskButtons:
    ; The bound was reached. Keep only the newly pressed buttons in mask [01].
    LDA ButtonsPressed
    AND $01
    STA ButtonsPressed

    ; In the overworld, or if the mask <> 0 (inner band), that is all.
    LDA CurLevel
    BEQ @Exit
    LDA $01
    BNE @Exit

    ; Dungeon, outer band: drop the input directions perpendicular to Link's facing.
    ; Keep up/down ($0C) if he faces up or down, else left/right ($03).
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
    ; [00] is right or down: the bound is reached when coordinate < bound.
    LDA $02
    CMP Link_BorderBounds, Y
    BCC @MaskButtons

@ExcludeNone:
    ; The bound was not reached. Report $FF in [01]; nothing is masked.
    LDA #$FF
    STA $01

; Shared exit: the RTS of Link_MaskInputInBorder, also branched to by Sword_Wield.
Link_MaskInputInBorder_Return:
    RTS
