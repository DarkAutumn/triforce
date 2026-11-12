; ============================================================================
; FILE:     bank7/tile-collision.asm     BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 2137-2321
; SUBJECTS: Tile
; TOPIC:    Tile lookup for collision: which play-area tile lies under an object or one step ahead of
;           it in a direction, the overworld's list of extra walkable tiles, and the walk-through wall
;           of overworld room $1F.
; SEE ALSO: bank7/walker-collision-screen-edge.asm (Walker_CheckTileCollision: the main caller, compares
;           the result with RoomFirstUnwalkableTile); bank7/shove.asm (knockback uses the same test);
;           bank5/link-handle-input.asm (Link's turning tests); bank7/run-game-nmi.asm
;           (Tile_PlayAreaColumnAddrs); bank7/flute-secret-pond.asm (raises RoomFirstUnwalkableTile to $99).
; KEY FACTS:
;   - A tile is walkable when its number is below RoomFirstUnwalkableTile ($89 OW, $78 UW); the compare
;     is done by the callers. In the OW the nine tiles of Tile_WalkableTilesOW are reported as $26, so they
;     pass too (Tile_GetAtObjOffset@CheckWalkable).
;   - The reference point (hotspot) of every object is (ObjX, ObjY + $0B). The play area starts at
;     Y = $40; column = X / 8, row = (hotspot Y - $40) / 8 (Tile_GetAtObjOffset@FetchTile).
;   - Offsets added to the hotspot per direction (Tile_GetAheadOfObj): up -8 for Link, -$10 for
;     other objects; left -8 for Link, -$10 for others; down +8; right +$10. Direction 0 is treated
;     like left.
;   - Moving up or down, two tiles are read: columns X / 8 and X / 8 + 1 of the same row; the higher
;     tile number is the result. Moving left or right only one tile is read, in the hotspot's row
;     (Tile_GetAtObjOffset@FetchTile). So for an object aligned to the grid (Y = 8k + 5) only the
;     8-pixel-high band ObjY + $0B..$12 is tested horizontally, and the upper half of the sprite may
;     overlap solid tiles.
;   - No offset is added going down when ObjY + $0B >= $DD, going right when ObjX >= $F0, or going
;     left (or direction 0) when ObjX < $10: the tile under the hotspot is used (Tile_GetAtObjOffset).
;   - There is no range check on the row: a hotspot Y below $40 or at $F0 and above indexes past the
;     22 rows of its column into the following column's bytes (Tile_GetAtObjOffset@FetchTile).
;   - OW room $1F (CurLevel = 0 and RoomId = $1F): for Link tested up or down at X = $80 with
;     Y < $56, the tile is forced to $26 (walkable): a wall that can be walked through
;     (Tile_GetAtObjOffset@SetTile).
;   - The result is also left in ObjCollidedTile,X (Tile_GetAtObjOffset).
; CONTENTS: (generated; L = line in this file)
;   L0043 Tile_WalkableTilesOW  data  TABLE: overworld tiles at or above $89 that are walkable all the same; index = 0-8 (search order 8
;   L0052 Tile_GetUnderObj      code  Gets the tile under an object's hotspot (ObjX, ObjY + $0B), with no look-ahead.
;   L0065 Tile_GetAheadOfObj    code  Gets the tile one step ahead of an object in direction [0F].
;   L0097 Tile_GetAtObjOffset   code  Reads the play-area tile at an object's hotspot moved by Y pixels along the axis of [0F].
; ============================================================================

; TABLE: overworld tiles at or above $89 that are walkable all the same; index = 0-8 (search order 8
; down to 0); value = tile number. Tile_GetAtObjOffset reports a match as tile $26. Not used in the UW.
Tile_WalkableTilesOW:
    .BYTE $8D, $91, $9C, $AC, $AD, $CC, $D2, $D5
    .BYTE $DF

; ----------------------------------------------------------------------------
; Gets the tile under an object's hotspot (ObjX, ObjY + $0B), with no look-ahead.
; IN:   X = object slot.
; OUT:  A = ObjCollidedTile,X = tile (an OW walkable tile is returned as $26); [0F] = 0; [04] = 0.
; ----------------------------------------------------------------------------
Tile_GetUnderObj:
    LDY #$00
    STY $0F
    BEQ Tile_GetAtObjOffset       ; Always taken.

; ----------------------------------------------------------------------------
; Gets the tile one step ahead of an object in direction [0F].
; IN:   X = object slot; [0F] = direction (one of 1 right, 2 left, 4 down, 8 up; or 0).
; OUT:  A = ObjCollidedTile,X = tile (an OW walkable tile is returned as $26).
; NOTE: Offset from the hotspot (ObjX, ObjY + $0B): up and left -8 for Link, -$10 for any other slot;
;       down +8; right +$10. With [0F] = 0 the up/left offset is applied to X, as for left.
; NOTE: Callers compare A with RoomFirstUnwalkableTile: carry set (A >= it) means blocked.
; ----------------------------------------------------------------------------
Tile_GetAheadOfObj:
    LDY #$F8                    ; Link: -8 for up and left.
    CPX #$00
    BEQ :+
    LDY #$F0                    ; Other objects: -$10 for up and left.
:
    ; Up, left or 0 (no right or down bit): keep that offset.
    LDA $0F
    AND #$05
    BEQ Tile_GetAtObjOffset

    ; Down: +8.
    LDY #$08
    AND #$04
    BNE Tile_GetAtObjOffset

    ; Right: +$10.
    LDY #$10

; ----------------------------------------------------------------------------
; Reads the play-area tile at an object's hotspot moved by Y pixels along the axis of [0F].
; IN:   X = object slot; Y = signed pixel offset; [0F] = direction, or 0 (then Y is added to X).
; OUT:  A = ObjCollidedTile,X = tile; in the OW a tile listed in Tile_WalkableTilesOW is returned as $26.
;       Uses [00:01] (column address) and [04] (the offset).
; NOTE: Vertical directions read the tiles of two adjacent columns (X / 8 and the next) in the same row
;       and keep the higher number; horizontal directions and 0 read one tile.
; NOTE: The offset is dropped (tile under the hotspot is read) for down when ObjY + $0B >= $DD, for
;       right when ObjX >= $F0, and for left or 0 when ObjX < $10.
; NOTE: Row = (Y - $40) / 8 with no range check: Y below $40 gives rows 24-31 and Y from $F0 up gives
;       rows 22-23, which read bytes of the following column (a column is 22 bytes).
; NOTE: OW room $1F, Link only, [0F] vertical, ObjX = $80, ObjY < $56: the result is forced to $26.
; ----------------------------------------------------------------------------
Tile_GetAtObjOffset:
    STY $04
    LDA ObjY, X                 ; Hotspot Y = ObjY + $0B.
    CLC
    ADC #$0B
    TAY
    PHA                         ; Save the hotspot Y.
    LDA $0F
    AND #$0C
    BEQ @AdjustX                ; Horizontal or 0: the offset applies to X.
    AND #$04
    BEQ @AdjustY                ; Up: always add the offset.
    CPY #$DD
    BCS @AsIsX                  ; Down with hotspot Y >= $DD: no offset.

@AdjustY:
    PLA                         ; Add the offset in [04] to the hotspot Y and save it again.
    CLC
    ADC $04
    PHA

@AsIsX:
    ; Vertical: take the X coordinate as is.
    LDY ObjX, X
    JMP @FetchTile

@AdjustX:
    LDY ObjX, X

    ; Right with X >= $F0, or left (or 0) with X < $10: no offset. Otherwise add [04] to X.
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
    TYA
    CLC
    ADC $04
    TAY

@FetchTile:
    ; Y register = X coordinate to test; the Y coordinate to test is on the stack.
    TYA
    AND #$F8                    ; Column = X / 8; columns are 8 pixels wide.

    ; (X AND $F8) / 4 = column * 2 = offset of the column's 2-byte address.
    LSR
    LSR
    TAY
    LDA Tile_PlayAreaColumnAddrs, Y  ; [00:01] = address of the column (22 tiles, top to bottom).
    STA $00
    LDA Tile_PlayAreaColumnAddrs+1, Y
    STA $01
    PLA                         ; Y coordinate to test.
    SEC                         ; The play area starts below the status bar, at Y = $40.
    SBC #$40
    LSR                         ; Row = (Y - $40) / 8. Not range checked.
    LSR
    LSR
    TAY
    LDA ($00), Y
    STA ObjCollidedTile, X
    LDA $0F
    AND #$0C
    BEQ @CheckWalkable          ; Horizontal or 0: one tile is enough.

    ; Vertical: also read the tile in the same row of the next column ($16 = 22 bytes further, 8 pixels
    ; to the right) and keep the higher tile number. Unwalkable tiles have the higher numbers, so a
    ; blocking tile in either column wins.
    TYA
    CLC
    ADC #$16
    TAY
    LDA ($00), Y
    CMP ObjCollidedTile, X
    BCC @CheckWalkable          ; The second tile is lower: keep the first.
    STA ObjCollidedTile, X

@CheckWalkable:
    LDA ObjCollidedTile, X
    LDY CurLevel
    BNE @Exit                   ; In the UW, return the tile as read.

    ; In the OW, look the tile up in Tile_WalkableTilesOW (entries 8 down to 0). A listed tile is replaced
    ; with $26, which is below RoomFirstUnwalkableTile, so the callers' compare sees it as walkable.
    LDA ObjCollidedTile, X
    LDY #$09

@MatchWalkableTile:
    DEY
    BMI @SetTile                ; Not in the list: keep the tile.
    CMP Tile_WalkableTilesOW, Y
    BNE @MatchWalkableTile
    LDA #$26

@SetTile:
    STA ObjCollidedTile, X

    ; Special case: Link (slot 0) in OW room $1F, tested in a vertical direction, at X = $80 and
    ; Y < $56. The tile is reported as $26, so Link walks through the wall there.
    CPX #$00
    BNE @ReturnTile
    LDA RoomId
    CMP #$1F
    BNE @ReturnTile
    LDA #$0C
    AND $0F
    BEQ @ReturnTile             ; Horizontal or 0: no change.
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
