; ============================================================================
; FILE:     bank5/lay-out-doors.asm      BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 4721-5036
; SUBJECTS: (none)
; TOPIC:    Door tables; LayOutDoors
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0030 DoorBits                                  data
;   L0033 DoorFaceTilesAddrsLo                      data
;   L0039 DoorFaceTilesAddrsHi                      data
;   L0045 PlayAreaDoorFaceAddrsLo                   data
;   L0051 PlayAreaDoorFaceAddrsHi                   data
;   L0057 NextDoorTileOffsets                       data
;   L0060 DirIndexToDoorSecondHalfOffsets           data
;   L0063 DirIndexToDoorColumnCount                 data
;   L0066 DirIndexToDoorRowCountMinusOne            data
;   L0069 LayOutDoors                               code
;   L0077 L_LayOutDoors_LoopDoors                   code
;   L0087 L_LayOutDoors_LoopHalves                  code
;   L0311 FetchDoorAddrsFaceTilesSrcAndPlayAreaDst  code  Params:
;   L0328 L165D4_Exit                               code
;   L0331 HorizontalDoorFaceIndexes                 data
;   L0335 HorizontalDoorFaceIndexesBaseOffsets      data
;   L0338 DoorVramAddrsHi                           data
;   L0341 DoorVramAddrsLo                           data
;   L0344 DoorNextRoomIdOffsets                     data
; ============================================================================

DoorBits:
    .BYTE $01, $02, $04, $08

DoorFaceTilesAddrsLo:
    .LOBYTES DoorFaceTilesE
    .LOBYTES DoorFaceTilesW
    .LOBYTES DoorFaceTilesS
    .LOBYTES DoorFaceTilesN

DoorFaceTilesAddrsHi:
    .HIBYTES DoorFaceTilesE
    .HIBYTES DoorFaceTilesW
    .HIBYTES DoorFaceTilesS
    .HIBYTES DoorFaceTilesN

PlayAreaDoorFaceAddrsLo:
    .LOBYTES $67A1
    .LOBYTES $654F
    .LOBYTES $6676
    .LOBYTES $6665

PlayAreaDoorFaceAddrsHi:
    .HIBYTES $67A1
    .HIBYTES $654F
    .HIBYTES $6676
    .HIBYTES $6665

NextDoorTileOffsets:
    .BYTE $14, $01, $01

DirIndexToDoorSecondHalfOffsets:
    .BYTE $02, $02, $2C, $2C

DirIndexToDoorColumnCount:
    .BYTE $03, $03, $02, $02

DirIndexToDoorRowCountMinusOne:
    .BYTE $01, $01, $02, $02

LayOutDoors:
    ; Copies the right door face for the direction, type, and state
    ; of the door to the play area tile map.
    ;
    ; Also, clears door bits from CurOpenedDoors of doorways
    ; that are not true doors.
    LDX #$03

L_LayOutDoors_LoopDoors:
    ; For each door, indexed by X, from 3 (N) to 0:
    LDA #$01

    ; For each half of a door, indexed by [06], from 1 to 0:
    ;
    ; [06] indicates whether we're handling the first half of
    ; the door: Left half for N/S, Top half for E/W.
    STA $06

L_LayOutDoors_LoopHalves:
    TXA
    PHA                         ; Save the current direction index (door index).
    STA $0B                     ; [0B] holds direction index.
    LDA DoorBits, X
    STA $02                     ; [02] holds the direction (bit) for the current door.
    JSR FindDoorTypeByDoorBit

    ; Begin looking for the door face for the door type.
    ; In general, use the door type as the provisional face;
    ; except turn 4 into 8.
    CMP #$05
    BCS @DoneClosedPF           ; If door type > 4, go use it as the provisional face.
    CMP #$04
    BNE @ClearFromOpenedMask    ; Door type < 4, go process it.
    LDA #$08
    BNE @DoneClosedPF           ; Door type = 4, go use 8 as the provisional face.

@SetPF9:
    ; We jump here if the door type is bombable and was opened.
    ; Change the provisional door face to value 9, which
    ; will become the "hole in the wall".
    LDA #$09
    BNE @DoneOpenPF

@ClearFromOpenedMask:
    ; Door type < 4 (open or any wall), clear door bit from
    ; the mask of opened doors, because this is not a true door.
    PHA
    LDA $02                     ; [02] current direction
    EOR #$FF
    AND CurOpenedDoors
    STA CurOpenedDoors
    PLA

    ; Furthermore, if door type = 0, then use 4 as the provisional face.
    CMP #$01
    BCS @DoneClosedPF
    LDA #$04

@DoneClosedPF:
    ; The door type (DT) has been mapped to a
    ; provisional door face value (PF) in %A as follows:
    ;
    ; DT PF Meaning
    ; -------------
    ; 0  4  open
    ; 1  1  wall
    ; 2  2  false wall
    ; 3  3  false wall 2
    ; 4  8  bombable
    ; 5  5  key
    ; 6  6  key 2
    ; 7  7  shutter
    ;
    ; Also at this point, the door bit in CurOpenedDoors
    ; has been cleared, if the type was "open" or any wall.
    PHA                         ; Save the provisional face.

    ; If the current direction points to a door that has been opened,
    ; then see if we need to set door flags.
    LDA CurOpenedDoors
    AND $02                     ; [02] current door direction
    TAX
    PLA                         ; Restore the provisional face.
    CPX $02
    BNE @DoneOpenPF

    ; The current direction points to a door that has been opened.
    ; It can only be "bombable", "key", "key 2", or "shutter".
    TAY                         ; Save provisional door face in Y register.
    PLA                         ; Get the direction index.
    PHA
    TAX                         ; Put direction index in X.
    TYA                         ; Restore provisional door face value.

    ; If provisional door face is not for a shutter, then
    ; it's for a "key" or "bombable". So, set the door's flag.
    ;
    ; Else it is for a shutter. Go set provisional door face to 4.
    CMP #$07
    BEQ :+                      ; If provisional door face is for "key" or "bombable",
    PHA
    JSR SetDoorFlag             ; then set the door's flag.
    PLA

    ; If the provisional door face = 8 (closed "bombable"),
    ; then go make it 9 (open "bombable").
    ; Else make it 4 (open "door").
    CMP #$08
    BEQ @SetPF9
:
    LDA #$04

@DoneOpenPF:
    ; For opened doors, the provisional door face has become:
    ;
    ; DT PF Meaning
    ; -------------
    ; 0  4  open
    ; 1  1  wall
    ; 2  2  false wall
    ; 3  3  false wall 2
    ; 4  9  bombable
    ; 5  4  key
    ; 6  4  key 2
    ; 7  4  shutter
    ; If handling the first half [06], then calculate OpenDoorwayMask.
    LDX $06
    BEQ :+
    LDX $0B                     ; [0B] direction index
    PHA
    JSR FindDoorTypeByDoorBit
    JSR CalcOpenDoorwayMask
    PLA
:
    ; If provisional door face < 4 (any wall), then go loop another half,
    ; because the visible door face is already a wall.
    CMP #$04
    BCC @NextLoopDoorHalf

    ; For all other provisional face values, calculate:
    ; door face := provisional face - 3
    SEC
    SBC #$03
    TAY

    ; Furthermore, if the latest provisional door face >= 3, then
    ; subtract one.
    CPY #$03
    BCC :+
    DEY
:
    ; Now Y holds the door face index (DFC closed, DFO opened) for the door type (DT).
    ;
    ; DT DFC DFO Meaning
    ; ------------------
    ; 0  1   1   open
    ; 4  4   5   bombable
    ; 5  2   1   key
    ; 6  2   1   key 2
    ; 7  3   1   shutter
    PLA                         ; Get the direction index.
    PHA
    JSR FetchDoorAddrsFaceTilesSrcAndPlayAreaDst

    ; If handling the second half, then offset to the second half
    ; of the tiles.
    LDA $06
    BNE :+
    LDA DirIndexToDoorSecondHalfOffsets, X    ; Advance destination tile address to the second half of door.
    JSR AddToInt16At0
    LDA #$06                    ; Advance source tile address to the second half of door.
    JSR AddToInt16At2
:
    ; Fix Y at 0 for copying source tiles to destination.
    ; Pointers will be incremented instead of Y.
    LDY #$00
    LDA DirIndexToDoorColumnCount, X
    STA $05                     ; [05] holds the column count.

@LoopColumn:
    ; For each column (3 or 2), indexed by [05] down to 1:
    PLA
    PHA                         ; Put the direction index in X.
    TAX

    ; For each row (2 or 3) in the door, indexed by X,
    ; starting from highest index down to 0:
    LDA DirIndexToDoorRowCountMinusOne, X
    TAX

@LoopRowTile:
    LDA ($02), Y                ; Copy 1 door tile.
    STA ($00), Y
    JSR Add1ToInt16At2          ; Increment source tile address.
    LDA NextDoorTileOffsets, X  ; Get the offset needed for the next play area tile.
    JSR AddToInt16At0

    ; If we're at the last row, and the direction is horizontal (< 2), then
    ; go 1 more tile down, to start the next column at the right place.
    ; We have to compensate for the fact that E/W doors are 
    ; shorter vertically than N/S doors.
    CPX #$00
    BNE :+
    PLA                         ; Put direction index in A.
    PHA
    CMP #$02
    BCS :+                      ; If direction index is horizontal (< 2),
    JSR Add1ToInt16At0
:
    ; Bottom of the tile row copying loop.
    ; Decrement the row index.
    DEX
    BPL @LoopRowTile

    ; Bottom of the column copying loop.
    ; Decrease the column counter.
    DEC $05
    BNE @LoopColumn

@NextLoopDoorHalf:
    ; Bottom of the door halves loop.
    PLA
    TAX                         ; Restore X to the door index.
    DEC $06                     ; Decrement [06] door half counter.
    BMI :+
    JMP L_LayOutDoors_LoopHalves    ; then go handle it.
:
    ; Bottom of the door loop.
    ; Decrement door index.
    DEX
    BMI L165D4_Exit             ; If finished the last door (X < 0), then return.
    JMP L_LayOutDoors_LoopDoors ; Go handle the next door.

; Params:
; A: direction index
; Y: face index
;
; Returns:
; X: direction index
; [$00:01]: address of door inside play area tile map
; [$02:03]: address of door tiles for direction and face
;
FetchDoorAddrsFaceTilesSrcAndPlayAreaDst:
    TAX
    LDA DoorFaceTilesAddrsLo, X
    STA $02
    LDA DoorFaceTilesAddrsHi, X
    STA $03
    LDA PlayAreaDoorFaceAddrsLo, X
    STA $00
    LDA PlayAreaDoorFaceAddrsHi, X
    STA $01
:
    DEY                         ; Add ($C * (face - 1)) to [$02:03].
    BEQ L165D4_Exit
    LDA #$0C
    JSR AddToInt16At2
    JMP :-

L165D4_Exit:
    RTS

HorizontalDoorFaceIndexes:
    .BYTE $01, $03, $06, $08, $03, $05, $08, $0A
    .BYTE $03, $06, $04, $07, $05, $08

HorizontalDoorFaceIndexesBaseOffsets:
    .BYTE $00, $04, $08, $0A

DoorVramAddrsHi:
    .BYTE $22, $22, $23, $21

DoorVramAddrsLo:
    .BYTE $5C, $42, $4F, $4F

DoorNextRoomIdOffsets:
    .BYTE $01, $FF, $10, $F0
