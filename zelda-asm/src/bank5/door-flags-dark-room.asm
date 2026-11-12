; ============================================================================
; FILE:     bank5/door-flags-dark-room.asm BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 7752-7842
; SUBJECTS: (none)
; TOPIC:    Open doorway mask, door flags, dark room test
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0026 CalcOpenDoorwayMask           code  Call for each direction index (3 to 0) to build an open doorway mask
;   L0063 AddDoorFlagsToCurOpenedDoors  code
;   L0078 SplitRoomId                   code
;   L0099 IsDarkRoom_Bank5              code  Params:
; ============================================================================

; Call for each direction index (3 to 0) to build an open doorway mask
; based on doorway type and room flags. Each call will shift the next
; doorway state bit into OpenDoorwayMask.
;
; Params:
; A: door type
; X: direction index
;
; Returns:
; A: untouched
;
CalcOpenDoorwayMask:
    LDY #$00
    PHA                         ; Save door type.
    CMP #$04
    BCC @ByDoorType             ; If door type < 4 (open or any wall), go shift the appropriate bit.

    ; Else we have to find out the walkability from the room flags.
    TXA
    PHA                         ; Save direction index.
    TYA
    PHA                         ; Save Y=0
    JSR GetRoomFlags
    CLC                         ; Clear carry to anticipate a zero AND result, meaning not walkable.
    AND LevelMasks, X           ; AND room flags with single-bit mask for direction.
    BEQ :+                      ; If result is not zero, then door hasn been opened,
    SEC                         ; and set carry, so that a 1 will be shifted into the mask.
:
    PLA
    TAY                         ; Restore Y=0.
    PLA
    TAX                         ; Restore direction index in X.

@ShiftIntoMask:
    LDA OpenDoorwayMask, Y
    ROL
    AND #$0F
    STA OpenDoorwayMask, Y
    PLA                         ; Restore door type.
    RTS

@ByDoorType:
    ; The door type indicates the walkability.
    CMP #$00
    BEQ @ShiftIntoMask          ; If door type is "open", carry is set, and go shift a 1 into the mask.
    CLC
    BCC @ShiftIntoMask          ; Door type is a wall, carry is clear, and go shift a 0 into the mask.

AddDoorFlagsToCurOpenedDoors:
    JSR GetRoomFlags
    LDX #$03

@LoopDoorBit:
    LDA ($00), Y
    AND LevelMasks, X
    BEQ :+
    ORA CurOpenedDoors
    STA CurOpenedDoors
:
    DEX
    BPL @LoopDoorBit
    RTS

SplitRoomId:
    ; Begin unverified code 17680
    LDA RoomId
    PHA
    AND #$0F
    TAY
    PLA
    LSR
    LSR
    LSR
    LSR
    TAX
    RTS
    ; End unverified code

; Params:
; Y: room ID
;
; Returns:
; A: $80 if dark, else 0
;
IsDarkRoom_Bank5:
    LDA CurLevel
    BEQ :+
    LDA LevelBlockAttrsE, Y
    AND #$80
:
    RTS
