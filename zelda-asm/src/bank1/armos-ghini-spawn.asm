; ============================================================================
; FILE:     bank1/armos-ghini-spawn.asm  BANK: 1 (switchable, $8000-$BFFF)
; UPSTREAM: Z_01.asm 2212-2364
; SUBJECTS: (none)
; TOPIC:    Touching armos statues or gravestones spawns Armos or Ghini
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0016 LinkToSquareOffsetsX     data
;   L0019 LinkToSquareOffsetsY     data
;   L0025 CheckPassiveTileObjects  code  Returns:
; ============================================================================

    ; End unverified code

LinkToSquareOffsetsX:
    .BYTE $00, $00, $F0, $10

LinkToSquareOffsetsY:
    .BYTE $FB, $13, $03, $03

; Returns:
; X: 0
;
CheckPassiveTileObjects:
    ; If grid offset <> 0 or input direction = 0, return.
    LDA ObjGridOffset
    BNE @ReturnX0
    LDA ObjInputDir
    BEQ @ReturnX0

    ; Compare the collided tile with tiles $BC to $C3.
    ; These are tiles for gravestone and armos squares.
    LDA #$BB
    STA $02                     ; [02] holds the collided tile.
    LDX #$08
    LDA ObjCollidedTile

@NextTile:
    INC $02
    DEX
    BPL @TestTile

@ReturnX0:
    LDX #$00
    RTS

@TestTile:
    CMP $02
    BNE @NextTile

    ; We'll be computing the location of the square.
    ; Start with Link's location.
    LDA ObjX
    STA $00                     ; [00] holds X coordinate
    LDA ObjY
    STA $01                     ; [01] holds Y coordinate
    LDA ObjDir
    AND #$0C
    BEQ @CheckHorizontal

    ; Link is facing vertically.
    ;
    ; If Link touched the left side of the square with the right side
    ; of his body, then add 8 to X.
    ;
    ; If Link touched the right side of the square with the left side
    ; of his body, then bitwise AND'ing with $F0 will align X with the square.
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
    ; If Link touched the top of the square with the bottom
    ; of his body, then add 8 to Y.
    LDA $02
    LSR
    BCS @FindSlot
    LDA $01
    CLC
    ADC #$08
    STA $01

@FindSlot:
    ; Look for an empty slot to instantiate a monster in.
    ; If none was found, then return.
    JSR FindEmptyMonsterSlot
    BEQ @ExitX0

    ; When dealing with objects, X usually indicates the slot number
    ; of the object. Set it to the empty slot found.
    LDX EmptyMonsterSlot

    ; Offset the coordinates we calculated by a square length
    ; in the direction Link is facing.
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

    ; Return, if the empty slot we found indicates it was initialized.
    ; I don't know how it would get in this state, because when
    ; a monster is destroyed, type is assigned 0, and it's uninitialized.
    LDA ObjUninitialized, X
    BEQ @ExitX0

    ; Because armos and flying ghini fade into existence over a little time,
    ; see if one is already beginning to appear at that location.
    ; Don't instantiate another one, if there is.
    LDY #$0B
    STX $03                     ; [03] holds the index of the slot to compare to the empty one.

@FindObjAtLocation:
    CPY $03
    BEQ @NextObjAtLocation      ; Skip the empty slot we found. So go decrement index [03].
    LDA a:ObjX, Y
    CMP ObjX, X
    BNE @NextObjAtLocation      ; If there's no other object at this X, go try another.
    LDA a:ObjY, Y
    CMP ObjY, X
    BNE @NextObjAtLocation      ; If there's no other object at this Y, go try another.
    LDA ObjType, Y
    BNE @ExitX0                 ; If there's truly an object there, return without instantiating an object.
    LDA ObjUninitialized, Y
    BEQ @ExitX0                 ; If there was an initialized object at that location, return.
    BNE @ChooseObjType          ; If there was never an object there, go instantiate one.

@NextObjAtLocation:
    DEY
    BNE @FindObjAtLocation

@ChooseObjType:
    ; Choose the right object type for the tile.
    LDA #$1E                    ; Armos
    LDY $02                     ; [02] collided tile
    CPY #$C0
    BCS :+                      ; If tile >= $C0, then instantiate an armos.
    LDA #$22                    ; Flying Ghini
:
    STA ObjType, X
    JSR ResetObjMetastate
    LDA #$3F                    ; Fade in for $3F frames.
    STA ObjTimer, X

@ExitX0:
    LDX #$00                    ; Restore player index in X.
    RTS
