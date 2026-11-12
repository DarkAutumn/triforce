; ============================================================================
; FILE:     bank1/armos-ghini-spawn.asm  BANK: 1 (switchable, $8000-$BFFF)
; UPSTREAM: Z_01.asm 2212-2364
; SUBJECTS: (none)
; TOPIC:    Overworld: Link walking into an armos statue square wakes an Armos ($1E); walking into a
;           gravestone square releases a flying Ghini ($22).
; SEE ALSO: bank7/walker-collision-screen-edge.asm (PlayerUnwalkable calls CheckPassiveTileObjects);
;           bank4/ghini-armos.asm (InitArmosOrFlyingGhini: the fade-in, armos secrets; the updates);
;           bank7/tile-collision.asm (GetCollidingTileMoving sets ObjCollidedTile);
;           bank7/object-stubs-destroy-monster.asm (FindEmptyMonsterSlot).
; KEY FACTS:
;   - Runs only on the overworld, when Link's move is blocked by an unwalkable tile, his grid
;     offset is 0 and a direction is held: CheckPassiveTileObjects (caller PlayerUnwalkable).
;   - Blocking tiles $BC-$BF (gravestone square) make a flying Ghini, type $22; tiles $C0-$C3
;     (armos square) make an Armos, type $1E: CheckPassiveTileObjects@ChooseObjType.
;   - The monster is placed exactly on the touched square and gets ObjTimer = $3F, the fade-in
;     time in frames: CheckPassiveTileObjects@ChooseObjType.
;   - It takes the highest free object slot in $0B-$01; with no free slot nothing happens. Nothing
;     is made either when that slot's ObjUninitialized is 0 (a slot freed by zeroing only its
;     ObjType): CheckPassiveTileObjects@FindSlot.
;   - Nothing is made if another slot already holds an object at the same X and Y (the monster
;     still fading in): CheckPassiveTileObjects@FindObjAtLocation.
;   - No RNG and no count limit here: a gravestone can be touched again once its Ghini has moved.
;   - InitObject exempts the Armos ($1E) from screen-edge entry but not the flying Ghini ($22): in
;     a room whose RoomAttrsF has bit 3 the Ghini's init goes through the edge-entry code, which
;     can move it to an edge cell (seen in InitObject; not traced).
; CONTENTS: (generated; L = line in this file)
;   L0035 LinkToSquareOffsetsX     data  TABLE: X offset from the X computed for Link to the touched square; index = direction index of
;   L0039 LinkToSquareOffsetsY     data  TABLE: Y offset from the Y computed for Link to the touched square; same index; signed pixels.
;   L0053 CheckPassiveTileObjects  code  Wakes an Armos or releases a flying Ghini when Link walks into an armos or gravestone square.
; ============================================================================

; TABLE: X offset from the X computed for Link to the touched square; index = direction index of
;        Link's facing (Y from GetOppositeDir: 0 up, 1 down, 2 left, 3 right); value = signed pixels.
LinkToSquareOffsetsX:
    .BYTE $00, $00, $F0, $10    ; up 0, down 0, left -$10, right +$10

; TABLE: Y offset from the Y computed for Link to the touched square; same index; signed pixels.
LinkToSquareOffsetsY:
    .BYTE $FB, $13, $03, $03    ; up -5, down +$13, left +3, right +3

; ----------------------------------------------------------------------------
; Wakes an Armos or releases a flying Ghini when Link walks into an armos or gravestone square.
; IN:   ObjCollidedTile (Link's) = the unwalkable tile that blocked him; ObjDir, ObjX, ObjY,
;       ObjGridOffset, ObjInputDir of Link
; OUT:  X = 0; maybe a new object: type $1E or $22 at the square, ObjMetastate 0, ObjTimer $3F.
;       Uses [00]-[03].
; NOTE: Tiles $BC-$BF give a flying Ghini, $C0-$C3 an Armos. A square is 2x2 tiles: t and t+1
;       are its left column (top, bottom), t+2 and t+3 its right column. The X and Y of the free
;       slot are overwritten even when no monster is made. The new object is uninitialized, so
;       its init (InitArmosOrFlyingGhini) runs each frame until the timer is 0.
; ----------------------------------------------------------------------------
CheckPassiveTileObjects:
    ; If grid offset <> 0 or input direction = 0, return.
    LDA ObjGridOffset
    BNE @ReturnX0
    LDA ObjInputDir
    BEQ @ReturnX0

    ; Compare the collided tile with tiles $BC to $C3 (8 tiles).
    ; These are tiles for gravestone and armos squares.
    LDA #$BB
    STA $02                     ; [02] counts up through the candidate tiles; ends as the matched tile.
    LDX #$08
    LDA ObjCollidedTile

@NextTile:
    INC $02
    DEX
    BPL @TestTile

@ReturnX0:
    ; No match (or nothing to do): return with X = 0, Link's object slot.
    LDX #$00
    RTS

@TestTile:
    CMP $02
    BNE @NextTile

    ; Compute the location of the touched square.
    ; Start with Link's location.
    LDA ObjX
    STA $00                     ; [00] holds X coordinate
    LDA ObjY
    STA $01                     ; [01] holds Y coordinate
    LDA ObjDir
    AND #$0C
    BEQ @CheckHorizontal

    ; Link is facing vertically. Make [00] the square's X.
    ;
    ; If the tile is in the square's left column ((tile AND 3) < 2), the square
    ; begins 8 pixels right of Link's X: add 8.
    ;
    ; Either way, AND with $F0 aligns X with the square.
    LDA $02
    AND #$03
    TAY
    LDA $00
    CPY #$02
    BCS :+
    CLC
    ADC #$08
:
    AND #$F0
    STA $00
    JMP @FindSlot

@CheckHorizontal:
    ; Link is facing horizontally.
    ;
    ; If the tile is a top tile of the square (bit 0 clear), the square begins
    ; 8 pixels lower than when a bottom tile is touched: add 8 to Y.
    LDA $02
    LSR
    BCS @FindSlot
    LDA $01
    CLC
    ADC #$08
    STA $01

@FindSlot:
    ; Look for an empty slot in $0B-$01 for the monster.
    ; If none was found, then return.
    JSR FindEmptyMonsterSlot
    BEQ @ExitX0

    LDX EmptyMonsterSlot        ; X = the empty slot found.

    ; Offset the coordinates we calculated by the table entries for the direction
    ; Link is facing (Y = reverse direction index from GetOppositeDir).
    ;
    ; Set the new object's location to them.
    LDA ObjDir
    JSR GetOppositeDir
    LDA $00
    CLC
    ADC LinkToSquareOffsetsX, Y
    STA ObjX, X
    LDA $01
    CLC
    ADC LinkToSquareOffsetsY, Y
    STA ObjY, X

    ; Return, if the empty slot is flagged initialized (ObjUninitialized = 0).
    ; A slot freed by zeroing only its ObjType stays in that condition.
    LDA ObjUninitialized, X
    BEQ @ExitX0

    ; Because armos and flying ghini fade into existence over a little time,
    ; see if one is already beginning to appear at that location.
    ; Don't instantiate another one, if there is.
    LDY #$0B
    STX $03                     ; [03] holds the empty slot, to skip it in the search.

@FindObjAtLocation:
    CPY $03
    BEQ @NextObjAtLocation      ; Skip the empty slot we found.
    LDA a:ObjX, Y
    CMP ObjX, X
    BNE @NextObjAtLocation      ; Slot Y is not at this X: try the next slot.
    LDA a:ObjY, Y
    CMP ObjY, X
    BNE @NextObjAtLocation      ; Slot Y is not at this Y: try the next slot.
    LDA ObjType, Y
    BNE @ExitX0                 ; An object is at this location: return without making a monster.
    LDA ObjUninitialized, Y
    BEQ @ExitX0                 ; An empty but initialized slot has these coordinates: also return.
    BNE @ChooseObjType          ; An empty, uninitialized slot matches: stop searching, make the monster.

@NextObjAtLocation:
    DEY
    BNE @FindObjAtLocation

@ChooseObjType:
    ; Choose the right object type for the tile.
    LDA #$1E                    ; Armos
    LDY $02                     ; [02] collided tile
    CPY #$C0
    BCS :+                      ; If tile >= $C0 (armos square), then make an armos.
    LDA #$22                    ; Flying Ghini (gravestone square, tiles $BC-$BF)
:
    STA ObjType, X
    JSR ResetObjMetastate       ; Metastate 0: no spawn cloud.
    LDA #$3F                    ; Fade in for $3F frames (the object stays uninitialized until then).
    STA ObjTimer, X

@ExitX0:
    LDX #$00                    ; Restore Link's object slot in X.
    RTS
