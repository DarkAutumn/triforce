; ============================================================================
; FILE:     bank5/layout-room-ow.asm     BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 5751-6077
; SUBJECTS: (none)
; TOPIC:    Overworld squares, layout, tile objects, cave/cellar layout
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0025 TileObjectTypes                  data
;   L0028 TileObjectPrimarySquaresOW       data
;   L0031 PrimarySquaresOW                 data
;   L0040 SecondarySquaresOW               data
;   L0050 LayoutRoomOW                     code
;   L0080 LayoutRoomOrCaveOW               code  Params:
;   L0206 CheckTileObject                  code  Params:
;   L0239 L16AF0_Exit                      code
;   L0248 WriteSquareOW                    code  Params:
;   L0302 PatchColumnDirectoryForCellar    code
;   L0317 SubroomLayoutAddrs               data
;   L0323 LayoutCaveAndAvanceSubmode       code
;   L0333 LayoutShortcutAndAdvanceSubmode  code
;   L0337 LayoutCellarAndAdvanceSubmode    code
; ============================================================================

TileObjectTypes:
    .BYTE $62, $63, $64, $65, $66, $67

TileObjectPrimarySquaresOW:
    .BYTE $C8, $D8, $C4, $BC, $C0, $C0

PrimarySquaresOW:
    .BYTE $24, $6F, $F3, $FA, $98, $90, $8F, $95
    .BYTE $8E, $90, $74, $76, $F3, $24, $26, $89
    .BYTE $03, $04, $70, $C8, $BC, $8D, $8F, $93
    .BYTE $95, $C4, $CE, $D8, $B0, $B4, $AA, $AC
    .BYTE $B8, $9C, $A6, $9A, $A2, $A0, $E5, $E6
    .BYTE $E7, $E8, $E9, $EA, $C0, $E0, $78, $7A
    .BYTE $7E, $80, $CC, $D0, $D4, $DC, $89, $84

SecondarySquaresOW:
    .BYTE $24, $24, $24, $24, $6F, $6F, $6F, $6F
    .BYTE $F3, $F3, $F3, $F3, $FA, $FA, $FA, $FA
    .BYTE $98, $95, $26, $26, $90, $95, $90, $95
    .BYTE $8F, $90, $8F, $90, $95, $96, $95, $96
    .BYTE $8E, $93, $90, $95, $90, $95, $92, $97
    .BYTE $74, $74, $75, $75, $76, $77, $76, $77
    .BYTE $F3, $24, $F3, $24, $24, $24, $24, $24
    .BYTE $26, $26, $26, $26, $89, $88, $8B, $88

LayoutRoomOW:
    ; Load the address of room column directory in [$02:03].
    LDA RoomLayoutsOWAddr
    STA $02
    LDA RoomLayoutsOWAddr+1
    STA $03
    LDA #$00                    ; Reset [06] for use in multiplication below.
    STA $06
    LDX RoomId                  ; Get unique room ID (OW).
    LDA LevelBlockAttrsD, X     ; The low 6 bits have the unique room ID.

    ; Add ((unique room ID) * $10) to address in [$02:03]. Each unique room has $10 columns.
    ASL
    ASL
    ROL $06
    ASL
    ROL $06
    ASL
    ROL $06
    ADC $02
    STA $02
    LDA $06
    ADC $03
    STA $03

; Params:
; [02:03]: address of room column directory
;
; Load the address of world flags in [$08:09].
;
LayoutRoomOrCaveOW:
    LDA LevelInfo_WorldFlagsAddr
    STA $08
    LDA LevelInfo_WorldFlagsAddr+1
    STA $09
    JSR FetchTileMapAddr

    ; For each column in room, indexed by [06]:
    LDA #$00
    STA $0C                     ; Reset [$0C] used for tracking repeat state.
    STA $06

@LoopColumnOW:
    LDY $06
    LDA ($02), Y                ; Get a column descriptor.
    AND #$F0                    ; Put column table number * 2 in X.
    LSR
    LSR
    LSR
    TAX

    ; Load the column table address for this descriptor in [$04:05].
    LDA ColumnDirectoryOW, X
    STA $04
    LDA ColumnDirectoryOW+1, X
    STA $05
    LDA ($02), Y                ; Get the column descriptor.
    AND #$0F                    ; Put column index in X.
    TAX
    LDY #$FF
:
    ; Look for the beginning of a column.
    INY
    LDA ($04), Y                ; Get a square descriptor.
    BPL :-                      ; If high bit is clear, then go read the next square descriptor.
    DEX
    BPL :-                      ; If this isn't the column we want, then go keep looking.

    ; We found the column.
    TYA
    JSR AddToInt16At4           ; Advance the square descriptor pointer by the offset, so we don't have to keep the offset in Y.
    LDA #$00                    ; Reset row number in [07].
    STA $07

@LoopSquareOW:
    LDY #$00
    LDA ($04), Y                ; Get the square descriptor.
    AND #$3F                    ; Get square index and put it in [$0D] and X.
    STA $0D
    TAX
    LDA PrimarySquaresOW, X
    PHA                         ; Save primary square.
    LDY RoomId                  ; Get room flags.
    LDA ($08), Y
    AND #$80
    BEQ @SkipSecret             ; If the secret wasn't found in this room, then skip all this.
    PLA                         ; Restore primary square.

    ; The secret was found in this room.
    CMP #$E7
    BEQ @MakeStairs             ; If this is a tree, go turn it into stairs.
    CMP #$E6
    BEQ @MakeCave               ; If this is a rock wall, go turn it into a cave entrance.
    CMP #$EA
    BNE @RestoreSquare          ; If this not a special armos, go leave the primary as is.

@MakeStairs:
    ; This is a tree or a special armos ($EA).
    ; Set the primary to stairs ($70), and square index to the
    ; first value ($10) for a type 1 square.
    LDA #$10
    STA $0D
    LDA #$70
    BNE @RestoreSquare

@MakeCave:
    ; This is a rock wall. Turn it into a cave entrance.
    LDA #$0C
    STA $0D

@RestoreSquare:
    PHA

@SkipSecret:
    PLA                         ; Restore primary square, if it wasn't modified above.
    JSR CheckTileObject
    LDY #$00
    JSR WriteSquareOW
    LDA #$02                    ; Point to next square in column in play area.
    JSR AddToInt16At0
    LDY #$00
    LDA ($04), Y                ; Get square descriptor.
    AND #$40
    BEQ @NextSquare             ; If we need to repeat this tile,
    EOR $0C                     ; then flip [$0C].
    STA $0C
    BNE :+                      ; After the second time flipping it, [$0C] = 0, and we've repeated it once. So,

@NextSquare:
    JSR Add1ToInt16At4          ; Point to the next square descriptor.
:
    INC $07                     ; Increment the processed row count.
    LDA $07
    CMP #$0B                    ; There are $B square rows in the play area.
    BNE @LoopSquareOW           ; If we haven't written $B rows, then go process a square.

    ; At the end of a column, we've reached the top of the next one.
    ; Move one more column over to get to the next square column.
    LDA #$16
    JSR AddToInt16At0
    INC $06                     ; Increment column index.
    LDA $06
    CMP #$10
    BCS L16AF0_Exit             ; If we have processed all columns, then return.
    JMP @LoopColumnOW           ; Go process the next one.

; Params:
; A: primary square
;
; Returns:
; A: primary square corresponding to tile object, else argument
;
; Tile object primary squares $E9 and $EA are unused; which
; means that tile object types $66 and $67 are unused.
; Instead, armos is represented by primary square $C0.
;
CheckTileObject:
    ; Find the index X corresponding to the primary square: $E5  => 0; $EA => 5.
    LDX #$EA
    STX $0A
    LDX #$05
:
    CMP $0A
    BEQ :+                      ; If we found the primary, go handle it.
    DEC $0A
    DEX
    BPL :-
    BMI L16AF0_Exit             ; If the primary isn't between $E5 to $EA, then return.
:
    LDA TileObjectPrimarySquaresOW, X
    PHA                         ; Save primary square.
    LDA TileObjectTypes, X
    STA RoomTileObjType
    LDA $06                     ; Get current column in play area where we'll put a square.
    ASL                         ; Store the X coordinate of tile object (column * $10).
    ASL
    ASL
    ASL
    STA RoomTileObjX
    LDA $07                     ; Get current row in play area where we'll put a square.
    ASL                         ; Store the Y coordinate of tile object ((row * $10) + $40).
    ASL
    ASL
    ASL
    CLC
    ADC #$40
    STA RoomTileObjY
    PLA                         ; Restore primary square.

L16AF0_Exit:
    RTS

; Params:
; A: primary square
; Y: offset from [$00:01]
; [$0D]: square index
; [$00:01]: pointer to play area
;
WriteSquareOW:
    LDX $0D                     ; Get square index.
    CPX #$10
    BCC @WriteType3             ; If square index < $10, go handle a secondary square.

    ; Type 1 square.
    ; Primary is the first tile. Next 3 tiles in CHR form the rest of the square.
    ; Square index >= $10.
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

@WriteType3:
    ; Type 3 square.
    ; Square index refers to a set of 4 tile indexes in secondary squares table.
    ; Square index < $10.
    TXA                         ; X := (square index * 4)
    ASL
    ASL
    TAX
    LDA SecondarySquaresOW, X
    STA ($00), Y                ; Write tile+0 to (col, row).
    INY
    INX
    LDA SecondarySquaresOW, X
    STA ($00), Y                ; Write tile+1 to (col, row+1).
    TYA
    CLC
    ADC #$15
    TAY
    INX
    LDA SecondarySquaresOW, X
    STA ($00), Y                ; Write tile+2 to (col+1, row).
    INX
    LDA SecondarySquaresOW, X
    JMP @WriteLastTile          ; Go write tile+3 to (col+1, row+1).

PatchColumnDirectoryForCellar:
    ; In OW, set first address of directory to start of OW column heap, as expected.
    LDA ColumnHeapOWAddr
    LDX ColumnHeapOWAddr+1
    LDY CurLevel
    BEQ :+

    ; In UW, set the first address of column directory to start of UW cellar column heap.
    LDA #<ColumnHeapUWCellar
    LDX #>ColumnHeapUWCellar
:
    STA ColumnDirectoryOW
    STX ColumnDirectoryOW+1
    RTS

SubroomLayoutAddrs:
    .ADDR RoomLayoutOWCave0
    .ADDR RoomLayoutOWCave1
    .ADDR RoomLayoutUWCellar0
    .ADDR RoomLayoutUWCellar1

LayoutCaveAndAvanceSubmode:
    LDX #$00                    ; Usual cave
:
    LDA SubroomLayoutAddrs, X
    STA $02
    LDA SubroomLayoutAddrs+1, X
    STA $03
    INC GameSubmode
    JMP LayoutRoomOrCaveOW

LayoutShortcutAndAdvanceSubmode:
    LDX #$02                    ; Offset of address of column directory of shortcut cave
    BNE :-

LayoutCellarAndAdvanceSubmode:
    ; Reset CurRow for when we start transferring rows
    ; after laying out the room.
    LDA #$00
    STA CurRow

    ; Get the offset of column directory address for the kind of cellar:
    ; - tunnel $3E:   offset 4
    ; - treasure $3F: offset 6
    LDX #$04
    JSR GetUniqueRoomId
    AND #$01
    BEQ :-
    LDX #$06
    BNE :-
