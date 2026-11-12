; ============================================================================
; FILE:     bank5/uw-floor-push-block.asm BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 5311-5535
; SUBJECTS: (none)
; TOPIC:    Dungeon floor layout, squares, push block creation
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0016 ColumnDirectoryUW             data
;   L0028 PrimarySquaresUW              data
;   L0031 LayoutUWFloor                 code
;   L0137 WriteSquareUW                 code  Params:
;   L0183 FindAndCreatePushBlockObject  code
; ============================================================================

ColumnDirectoryUW:
    .ADDR ColumnHeapUW0
    .ADDR ColumnHeapUW1
    .ADDR ColumnHeapUW2
    .ADDR ColumnHeapUW3
    .ADDR ColumnHeapUW4
    .ADDR ColumnHeapUW5
    .ADDR ColumnHeapUW6
    .ADDR ColumnHeapUW7
    .ADDR ColumnHeapUW8
    .ADDR ColumnHeapUW9

PrimarySquaresUW:
    .BYTE $B0, $74, $94, $B4, $70, $68, $F4, $24

LayoutUWFloor:
    JSR GetUniqueRoomId
    PHA                         ; Save unique room ID.
    LDA #<RoomLayoutsUW         ; Load the address of room column directory in [$02:03].
    STA $02
    LDA #>RoomLayoutsUW
    STA $03
    PLA                         ; Restore unique room ID.
    ASL                         ; Add ((unique room ID) * $C) to address in [$02:03]. Each unique room has $C columns.
    ASL
    STA $00
    JSR AddToInt16At2
    LDA $00
    JSR AddToInt16At2
    LDA $00
    JSR AddToInt16At2
    LDA #$8C                    ; Load the address of top-left tile in floor of play area in [$00:01].
    STA $00
    LDA #$65
    STA $01

    ; For each column in room, indexed by [06]:
    LDY #$00
    STY $06

@LoopColumnUW:
    LDY $06
    LDA ($02), Y                ; Get a column descriptor.
    AND #$F0                    ; Put column table number * 2 in X.
    LSR
    LSR
    LSR
    TAX
    LDA ColumnDirectoryUW, X    ; Load the column table address for this descriptor in [$04:05].
    STA $04
    LDA ColumnDirectoryUW+1, X
    STA $05
    LDA ($02), Y                ; Get the column descriptor again.
    AND #$0F                    ; Put column index in X.
    TAX
    LDY #$00

@FindSquare:
    LDA ($04), Y                ; Get a square descriptor.
    BPL :+                      ; If high bit is set,
    DEX                         ; then we've found the beginning of a column;
    BMI @FoundColumn            ; If this is the column we want, then go handle it.
:
    INY                         ; Increment the square descriptor offset.
    JMP @FindSquare             ; Go get the next square descriptor.

@FoundColumn:
    ; We found the column.
    TYA
    JSR AddToInt16At4           ; Advance the square descriptor pointer by the offset, so we don't have to keep the offset in Y.
    LDA #$00
    STA $07                     ; Reset processed row count.
    STA $08                     ; Reset repeat count.

@LoopSquareRow:
    ; Write and repeat squares from the column.
    LDY #$00
    LDA ($04), Y                ; Get the square descriptor.
    AND #$07                    ; Get the square index from the descriptor.
    TAX
    LDA PrimarySquaresUW, X
    LDY #$00
    JSR WriteSquareUW
    LDA #$02                    ; Point to next square in column in play area.
    JSR AddToInt16At0
    LDY #$00
    LDA ($04), Y                ; Get the square descriptor.
    AND #$70                    ; Isolate the count.
    LSR
    LSR
    LSR
    LSR
    CMP $08
    BEQ :+                      ; If we haven't repeated this square as specified,
    INC $08                     ; then increment the processed repeat count;
    JMP @NextLoopSquareRow      ; and go increment the row, and check if we're done in this column.
:
    LDA #$00                    ; Reset repeat count.
    STA $08
    JSR Add1ToInt16At4          ; Point to the next square descriptor.

@NextLoopSquareRow:
    INC $07                     ; Increment the processed row count.
    LDA $07
    CMP #$07                    ; There are 7 square rows in UW floor.
    BCC @LoopSquareRow          ; If we haven't written 7 rows, then go process a square.
    LDA #$1E                    ; Move 2 columns right and to the top of the floor area.
    JSR AddToInt16At0
    INC $06                     ; Increment column index.
    LDA $06
    CMP #$0C
    BCS :+                      ; If we haven't processed all columns,
    JMP @LoopColumnUW           ; then go process the next one.
:
    RTS

; Params:
; A: primary square
; Y: offset from [$00:01]
; [$00:01]: pointer to play area
;
WriteSquareUW:
    CMP #$70
    BCC @WriteType2
    CMP #$F3
    BCS @WriteType2

    ; Type 1 square.
    ; Primary is the first tile. Next 3 tiles in CHR form the rest of the square.
    ; Primary >= $70 and < $F3.
    TAX
    STA ($00), Y                ; Write tile+0 to (col, row).
    INY
    INX
    TXA
    STA ($00), Y                ; Write tile+1 to (col, row+1).
    TYA
    CLC
    ADC #$15
    TAY
    INX
    TXA
    STA ($00), Y                ; Write tile+2 to (col+1, row).
    INX
    TXA

@WriteLastTile:
    INY
    STA ($00), Y                ; Write last tile to (col+1, row+1).
    RTS

@WriteType2:
    ; Type 2 square.
    ; Primary is a tile used for the whole square.
    ; Primary < $70 or >= $F3.
    STA ($00), Y                ; Write tile to (col, row).
    INY
    STA ($00), Y                ; Write tile to (col, row+1).
    PHA
    TYA
    CLC
    ADC #$15
    TAY
    PLA
    STA ($00), Y                ; Write tile to (col+1, row).
    JMP @WriteLastTile          ; Go write tile to (col+1, row+1).

FindAndCreatePushBlockObject:
    ; Reset block state and direction.
    LDA #$00
    STA ObjState+11
    STA ObjDir+11

    ; If the unique room ID is $21, then
    ; put the push block object at ($40, $80), and go set the type.
    ;
    ; UNKNOWN:
    ; UW room layout $21 is the entrance room. But none of them
    ; have the push block flag set in LevelBlock attributes.
    JSR GetUniqueRoomId
    CMP #$21
    BNE :+
    ; Begin unverified code 167FE
    LDA #$40
    STA ObjX+11
    ASL
    STA ObjY+11
    JMP @SetType                ; Go set the object type and return.
    ; End unverified code
:
    ; Look for a block tile in row $A of play area, starting in column 4.
    LDX #$08
    LDY #$0A

@LoopColumn:
    LDA PlayAreaColumnAddrs, X
    STA $00
    LDA PlayAreaColumnAddrs+1, X
    STA $01
    LDA ($00), Y
    CMP #$B0
    BEQ @Found
    INX
    INX
    INX
    INX
    CPX #$34
    BNE @LoopColumn

@Found:
    ; The block was found in a column with an address at
    ; offset %X in column table. So the column number would
    ; be (%X/2), and X coordinate ((%X/2)*8). That means
    ; multiplying %X by 4.
    TXA
    ASL
    ASL
    STA ObjX+11
    LDA #$90                    ; Row $A is at $90 from the top of the screen.
    STA ObjY+11

@SetType:
    LDA #$68                    ; Block object type
    STA ObjType+11
    RTS
