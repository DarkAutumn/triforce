; ============================================================================
; FILE:     bank5/edge-spawn-cell.asm    BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 3418-3513
; SUBJECTS: (none)
; TOPIC:    Edge cell search for monster spawns
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0012 FindNextEdgeSpawnCell  code
; ============================================================================

FindNextEdgeSpawnCell:
    ; Load [0A] with the value before the call.
    LDA CurEdgeSpawnCell
    STA $0A

@LoopEdgeCell:
    ; Loop to look for a place to spawn a monster from the edge
    ; of the screen. Move counterclockwise, one square at a time.
    ;
    ; First, if low nibble = 0, then we're at the left edge.
    ; We'll move down $10 pixels.
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
    ; Else low nibble = $F. We're at the right edge,
    ; and we'll move up $10 pixels.
    TYA
    CLC
    ADC $0A                     ; Add $10 or -$10 at the left or right edge.
    STA $0A

@CheckTopOrBottom:
    ; Next, if high nibble = $E, then we're at the bottom edge.
    ; Move right one pixel.
    LDA $0A
    AND #$F0
    CMP #$E0
    BNE :+
    INC $0A
    JMP @PointToColumn
:
    ; Else if high nibble <> 4, then we're at the left or right.
    ; Dont' move horizontally.
    CMP #$40
    BNE @PointToColumn

    ; Else high nibble = 4. We're at the top edge.
    ; Move left 1 pixel.
    DEC $0A

@PointToColumn:
    ; Time to get the address of the column.
    ; Starting at the top of the leftmost column.
    JSR FetchTileMapAddr

    ; Add $2C to the address as many times as the low nibble of [0A],
    ; in order to point to the column we want.
    LDA $0A
    AND #$0F
    TAY
    BEQ @GetRowOffset
:
    LDA #$2C                    ; Each column is $16 tiles. Each square column has two columns.
    JSR AddToInt16At0
    DEY
    BNE :-

@GetRowOffset:
    ; Turn the row part of [0A] into a tile index.
    ; row := (([0A] AND $F0) - $40) / 8
    LDA $0A
    AND #$F0
    SEC
    SBC #$40
    LSR
    LSR
    LSR
    TAY

    ; If the tile < $84, then it's walkable. So, go use it.
    ; $84 is a sand tile, which is OK for many monsters; but not
    ; when coming in from the edges.
    LDA ($00), Y
    CMP #$84
    BCC @SetSpawnCell

    ; Bottom of the loop.
    ; If you reach the original cell, then stop.
    LDA $0A
    CMP CurEdgeSpawnCell
    BNE @LoopEdgeCell

@SetSpawnCell:
    ; Set the spawn cell to the one we found.
    LDA $0A
    STA CurEdgeSpawnCell
    RTS
