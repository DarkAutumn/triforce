; ============================================================================
; FILE:     bank7/tile-collision.asm     BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 2137-2321
; SUBJECTS: (none)
; TOPIC:    Walkable tiles, colliding tile lookup
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0015 WalkableTiles           data
;   L0026 GetCollidableTileStill  code  Params:
;   L0039 GetCollidingTileMoving  code  Params:
;   L0069 GetCollidableTile       code  Params:
; ============================================================================

WalkableTiles:
    .BYTE $8D, $91, $9C, $AC, $AD, $CC, $D2, $D5
    .BYTE $DF

; Params:
; X: object index
;
; Returns:
; A: tile
; [049E][X]: tile
;
GetCollidableTileStill:
    LDY #$00
    STY $0F
    BEQ GetCollidableTile

; Params:
; X: object index
; [0F]: direction
;
; Returns:
; A: tile
; [049E][X]: tile
;
GetCollidingTileMoving:
    LDY #$F8                    ; Use -8 ($F8) for the hotspot offset, if it's Link.
    CPX #$00
    BEQ :+                      ; If it's not Link,
    LDY #$F0                    ; then use -$10 ($F0) for the hotspot offset.
:
    ; If direction is up, left, or 0,
    ; then use the offset we determined above.
    LDA $0F
    AND #$05
    BEQ GetCollidableTile

    ; Else if it's down,
    ; then use offset 8.
    LDY #$08
    AND #$04
    BNE GetCollidableTile

    ; For right, use offset $10.
    LDY #$10

; Params:
; X: object index
; Y: offset from hotspot
; [0F]: direction or 0
;
; Returns:
; A: tile
; [049E][X]: tile
;
GetCollidableTile:
    STY $04
    LDA ObjY, X                 ; Start with a Y coordinate $B pixels down from top of object.
    CLC
    ADC #$0B
    TAY
    PHA                         ; Save this first adjusted Y coordinate.
    LDA $0F
    AND #$0C
    BEQ @AdjustX                ; If direction is horizontal or 0, go adjust X coordinate.
    AND #$04                    ; If direction is down and Y coordinate < $DD,
    BEQ @AdjustY
    CPY #$DD
    BCS @AsIsX

@AdjustY:
    PLA                         ; then pop Y coordinate,
    CLC                         ; and add [04] offset from hotspot.
    ADC $04
    PHA                         ; Save adjusted Y coordinate.

@AsIsX:
    ; Take the X coordinate as is.
    LDY ObjX, X
    JMP @FetchTile

@AdjustX:
    ; Adjust the X coordinate.
    LDY ObjX, X

    ; If direction is left and X coordinate >= $10
    ; or direction is right and X coordinate < $F0,
    LDA $0F
    AND #$01
    BEQ @CheckLeftBoundary
    CPY #$F0
    BCS @FetchTile
    BCC @AddHotspotOffset

@CheckLeftBoundary:
    CPY #$10
    BCC @FetchTile

@AddHotspotOffset:
    TYA                         ; then add [04] offset from hotspot.
    CLC
    ADC $04
    TAY

@FetchTile:
    TYA
    AND #$F8                    ; Mask to make X coordinate a multiple of 8, the column size.

    ; Divide this multiple by 4 to get the offset of the 2-byte address.
    LSR
    LSR
    TAY
    LDA PlayAreaColumnAddrs, Y  ; Put the address of the column in [00:01].
    STA $00
    LDA PlayAreaColumnAddrs+1, Y
    STA $01
    PLA                         ; Restore adjusted Y coordinate.
    SEC                         ; Subtract status bar height $40 from Y coordinate.
    SBC #$40
    LSR                         ; Divide Y coordinate by 8 to get the row it sits inside.
    LSR
    LSR
    TAY
    LDA ($00), Y                ; Get the tile at this column and row.
    STA ObjCollidedTile, X      ; Store the tile in the object's tile slot.
    LDA $0F
    AND #$0C
    BEQ @CheckWalkable          ; If direction in [0F] is horizontal or 0, skip checking another column.

    ; The direction in [0F] is vertical. Check the tile in the next column.
    ; It might be a blocking tile. Blocking tiles are arranged after
    ; walkable tiles.
    TYA
    CLC
    ADC #$16
    TAY
    LDA ($00), Y
    CMP ObjCollidedTile, X
    BCC @CheckWalkable          ; If second tile number >= first tile number,
    STA ObjCollidedTile, X      ; Use the second one, because it might be blocking.

@CheckWalkable:
    LDA ObjCollidedTile, X      ; Get the tile to return, if needed.
    LDY CurLevel
    BNE @Exit                   ; If in UW, then return.

    ; Look for the tile in a list of walkable tiles. If it's found,
    ; then turn it into $26 to simplify walkability tests.
    LDA ObjCollidedTile, X
    LDY #$09

@MatchWalkableTile:
    DEY
    BMI @SetTile                ; If the tile wasn't found, quit the loop.
    CMP WalkableTiles, Y
    BNE @MatchWalkableTile      ; If the tile doesn't match this element, go check the next one.

    ; The tile was found in the list. Substitute $26 for it.
    LDA #$26

@SetTile:
    STA ObjCollidedTile, X
    CPX #$00
    BNE @ReturnTile             ; If the object is not Link, then return without changing anything else.
    LDA RoomId
    CMP #$1F
    BNE @ReturnTile             ; If not in special OW room $1F, return. There's a false wall.
    LDA #$0C
    AND $0F
    BEQ @ReturnTile             ; If direction is horizontal or 0, then return.

    ; If Link is at X=$80, Y<$56; then set the tile to walkable tile $26.
    LDA ObjX
    CMP #$80
    BNE @ReturnTile
    LDA ObjY
    CMP #$56
    BCS @ReturnTile
    LDA #$26
    STA ObjCollidedTile

@ReturnTile:
    LDA ObjCollidedTile, X

@Exit:
    RTS
