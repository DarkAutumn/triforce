; ============================================================================
; FILE:     bank5/edge-spawn-cell.asm    BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 3418-3513
; SUBJECTS: (none)
; TOPIC:    Picks the next cell on the screen's outer ring where a monster may walk in from the edge
;           (overworld rooms with the "monsters from edges" attribute).
; SEE ALSO: bank7/object-init-table.asm (InitObject: the only caller; turns the cell into X and Y, sets
;           MonstersFromEdgesLongTimer, rejects a spot too close to Link);
;           bank5/save-ram-clear-ram.asm (the starting cell $40); bank5/spawn-positions.asm
;           (IsDistanceSafeToSpawn).
; KEY FACTS:
;   - CurEdgeSpawnCell = row in the high nibble ($4 top .. $E bottom), column in the low nibble ($0
;     left .. $F right). InitObject places the monster at X = column * $10, Y = row * $10 - 3.
;   - The search steps around the ring counterclockwise, one cell a step: down the left edge, right
;     along the bottom, up the right edge, left along the top. It starts from the cell after the stored
;     one, so successive monsters enter at successive open cells (FindNextEdgeSpawnCell@LoopEdgeCell).
;   - The corner cells $E0 (bottom left) and $4F (top right) are never tested: the step into them also
;     moves along the next edge.
;   - A cell is accepted when the tile at its top left is below $84. That is stricter than walking in
;     the overworld (RoomFirstUnwalkableTile = $89): tiles $84-$88 are refused.
;   - If no cell on the ring qualifies, the search ends back at the stored cell and keeps it, although
;     that cell has just failed the tile test too.
;   - No random numbers are used; the cell persists between rooms (CurEdgeSpawnCell).
; CONTENTS: (generated; L = line in this file)
;   L0035 FindNextEdgeSpawnCell  code  Advances CurEdgeSpawnCell counterclockwise around the screen's outer ring to the next open cell.
; ============================================================================

; ----------------------------------------------------------------------------
; Advances CurEdgeSpawnCell counterclockwise around the screen's outer ring to the next open cell.
; OUT:  CurEdgeSpawnCell = the cell found (high nibble row $4-$E, low nibble column $0-$F).
;       Uses [00]-[01] (tile map address) and [0A] (cell being tested).
; NOTE: Open = tile number below $84 at the cell's top left tile. The stored cell is the last one
;       tested; if it fails as well, the lap ends there and it is kept anyway. Deterministic: no RNG.
; ----------------------------------------------------------------------------
FindNextEdgeSpawnCell:
    ; [0A] = the cell being tested, starting from the stored cell.
    LDA CurEdgeSpawnCell
    STA $0A

@LoopEdgeCell:
    ; Step to the next cell on the edge of the screen, counterclockwise.
    ;
    ; First, if low nibble = 0, then we're at the left edge.
    ; Move down one row (add $10).
    LDY #$10
    LDA $0A
    AND #$0F
    BEQ @HandleLeftOrRight

    ; Else if low nibble <> $F, then we're at the top or bottom.
    ; Don't move vertically.
    LDY #$F0
    CMP #$0F
    BNE @CheckTopOrBottom

@HandleLeftOrRight:
    ; Left edge: Y = $10. Right edge (low nibble = $F): Y = $F0, move up one row.
    TYA
    CLC
    ADC $0A
    STA $0A

@CheckTopOrBottom:
    ; Next, if high nibble = $E, then we're at the bottom edge (possibly having just arrived).
    ; Move right one column.
    LDA $0A
    AND #$F0
    CMP #$E0
    BNE :+
    INC $0A
    JMP @PointToColumn
:
    ; Else if high nibble <> 4, then we're at the left or right.
    ; Don't move horizontally.
    CMP #$40
    BNE @PointToColumn

    ; Else high nibble = 4. We're at the top edge.
    ; Move left one column.
    DEC $0A

@PointToColumn:
    ; Get the address of the room's tile map: the top of the leftmost tile column.
    JSR FetchTileMapAddr

    ; Add $2C to the address as many times as the low nibble of [0A],
    ; in order to point to the cell's left tile column.
    LDA $0A
    AND #$0F
    TAY
    BEQ @GetRowOffset
:
    LDA #$2C                    ; A tile column is $16 tiles; a cell is two tile columns wide.
    JSR AddToInt16At0
    DEY
    BNE :-

@GetRowOffset:
    ; Turn the row part of [0A] into a tile row within the column:
    ; tile row = (([0A] AND $F0) - $40) / 8, two tile rows per cell row, row 4 at the top.
    LDA $0A
    AND #$F0
    SEC
    SBC #$40
    LSR
    LSR
    LSR
    TAY

    ; If the tile < $84, the cell is open. Go use it.
    ; The limit is lower than the overworld's first unwalkable tile ($89), so tiles $84-$88, which
    ; Link can walk on, are refused for monsters coming in from the edges.
    LDA ($00), Y
    CMP #$84
    BCC @SetSpawnCell

    ; Bottom of the loop.
    ; If we're back at the stored cell (it has just failed the test too), stop and keep it.
    LDA $0A
    CMP CurEdgeSpawnCell
    BNE @LoopEdgeCell

@SetSpawnCell:
    ; Set the spawn cell to the one we found.
    LDA $0A
    STA CurEdgeSpawnCell
    RTS
