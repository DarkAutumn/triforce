; ============================================================================
; FILE:     bank5/fill-walls.asm         BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 4537-4719
; SUBJECTS: (none)
; TOPIC:    Door type lookup; FillWalls
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0032 FindDoorTypeByDoorBit  code  Params:
;   L0076 ReachedTopWallBottom   code
;   L0085 DecBottomOffset        code
;   L0090 FillWalls              code
;   L0114 LoopWallTile           code
;   L0142 MoveWallPtrs           code
;   L0152 NextLoopWallTile       code
; ============================================================================

; Params:
; [02]: bitmask for target door.
;
; Returns:
; A: door type for desired direction in current room.
; [01]: same as [02] if found; $10 otherwise
; [03]: reverse direction index
;
; Door bits:
; 1: E
; 2: W
; 4: S
; 8: N
;
FindDoorTypeByDoorBit:
    LDA #$01                    ; Start with single-bit mask 1.
    STA $01
    LDA #$03                    ; Test 4 directions.
    STA $03

@LoopDoorBit:
    LDY RoomId
    LDA LevelBlockAttrsA, Y     ; Get attr byte A for S/N doors.
    LDY $03
    CPY #$02
    BCC :+                      ; If counter >= 2, then get attr byte B instead for E/W doors.
    LDY RoomId
    LDA LevelBlockAttrsB, Y
:
    STA $00                     ; Store either attribute byte in [$00].
    LDA $03
    AND #$01
    BNE :+                      ; If counter is even, then ...
    LSR $00                     ; Isolate N/W doors.
    LSR $00
    LSR $00
:
    LSR $00                     ; Isolate S/E doors, if we jump here.
    LSR $00

    ; Now [$00] holds isolated door type attribute.
    ;
    ; Test mask passed in [$02] with current single-bit mask in [$01].
    LDA $02
    BIT $01
    BNE :+                      ; If they match, then go return door type for current direction.
    ASL $01                     ; Shift the single-bit mask left.
    DEC $03
    BPL @LoopDoorBit            ; Go test the next direction.
    ; Begin unverified code 1642C
    LDA #$08                    ; Else return an invalid door type.
    RTS
    ; End unverified code
:
    LDA $00
    AND #$07
    RTS

ReachedTopWallBottom:
    ; Read a zero tile. Reached the bottom of a top wall.
    ;
    ; Set up A and X to move top and bottom offsets to the next column.
    LDA #$13
    STA $06
    LDX #$19
    BNE MoveWallPtrs            ; Go move top and bottom pointers.

DecBottomOffset:
    ; Subtract A=1 from bottom offset.
    JSR Sub1FromInt16At4
    JMP NextLoopWallTile        ; Go increment the wall tile list address, and continue.

FillWalls:
    ; Load the address of WallTileList.
    LDA #<WallTileList
    STA $00
    LDA #>WallTileList
    STA $01

    ; Load the address of second tile in second row of PlayAreaTiles.
    ; This is where we'll start loading tiles for the room.
    LDA #$47
    STA $02
    LDA #$65
    STA $03

    ; Load the address of second last tile in second row of PlayAreaTiles.
    ; This is where we'll stop loading tiles for the room.
    LDA #$5A
    STA $04
    LDA #$65
    STA $05
    LDA #$0A                    ; Load $A pairs of tiles, top and bottom ($14 tiles total).
    STA $06
    LDY #$00

LoopWallTile:
    LDA ($00), Y                ; Get a tile from wall tile list.
    BEQ ReachedTopWallBottom
    STA ($02), Y                ; Set the tile at the top and bottom locations.
    STA ($04), Y

    ; If the tile isn't the vertical line $DE nor anything >= $E2,
    ; then set the bottom tile to the next one, which is flipped
    ; vertically. For example, $E0 => $E1.
    CMP #$DE
    BEQ :+
    CMP #$E2
    BCS :+
    ADC #$01
    STA ($04), Y
:
    LDA #$01                    ; Set A to advance to next tile in column.
    LDX #$01                    ; 1 means subtract 1 from bottom offset.
    DEC $06                     ; Decrement count.
    BNE MoveWallPtrs            ; If reached the end of the column,
    LDA #$0A                    ; then count $A tile pairs again.
    STA $06
    LDA #$0D                    ; Set A to advance to next column (at second tile).

    ; $1F means add this amount to bottom offset to move
    ; it to bottom of next column.
    LDX #$1F

MoveWallPtrs:
    JSR AddToInt16At2           ; Increase top offset by 1 tile, or by another amount to get to the next column (at second tile).
    TXA                         ; We need the amount to add or subtract in A.
    DEX
    BEQ DecBottomOffset         ; If X was 1, then go subtract 1 from bottom offset.

    ; Else add the original X value to bottom offset,
    ; intending to move it to the bottom of the next column.
    JSR AddToInt16At4

NextLoopWallTile:
    JSR Add1ToInt16At0          ; Increment the wall tile list address.
    CMP #<(WallTileList + $4E)
    BNE LoopWallTile            ; If wall tile list pointer hasn't reached the end ($94EE), go process tiles again. At this point, we'll have written the walls on the left half of the play area.

    ; Copy rotated 180 degrees, accounting for appropriate
    ; horizontal or vertical flipping of tiles.
    ;
    ; The source is in the top left.
    LDA #$30
    STA $02
    LDA #$65
    STA $03
    LDA #$EF                    ; The destination is the bottom right.
    STA $04
    LDA #$67
    STA $05

@LoopRotate:
    LDA ($02), Y                ; Copy 1 tile.
    STA ($04), Y
    CMP #$DD
    BEQ @SwapWithVertical       ; If this is a horizontal line, then go flip it vertically.
    CMP #$E0                    ; Tiles >= $E0 don't need to be flipped.
    BCS @NextLoopRotate         ; Skip this if tile doesn't need to be flipped.
    CMP #$DC
    BCS :+                      ; If tile < $DC,
    ADC #$01                    ; then tile needs 2 added to rotate it 180 degrees.
    STA ($04), Y
:
    CLC                         ; Tiles >= $DC only need 1 added to flip them.
    ADC #$01

@SetRotatedTile:
    STA ($04), Y

@NextLoopRotate:
    JSR Sub1FromInt16At4
    JSR Add1ToInt16At2
    CMP #$90                    ; Once source pointer reaches the middle ($6690), we're done.
    BNE @LoopRotate
    LDA $03
    CMP #$66
    BNE @LoopRotate
    RTS

@SwapWithVertical:
    LDA #$DC
    BNE @SetRotatedTile
