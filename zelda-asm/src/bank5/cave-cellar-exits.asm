; ============================================================================
; FILE:     bank5/cave-cellar-exits.asm  BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 3059-3217
; SUBJECTS: (none)
; TOPIC:    Leaving cellars, caves, shortcut stairs
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0023 CheckSubroom         code  Params:
;   L0147 GoToModeAFromCellar  code  Params:
;   L0151 GoToModeAFromCave    code
;   L0155 EndPrepareMode       code
;   L0165 L14F72_Exit          code
;   L0168 CheckCaveEdge        code
; ============================================================================

; Params:
; [0F]: movement direction
;
; Returns:
; [0F]: untouched, or 0
;
CheckSubroom:
    LDA GameMode
    CMP #$09
    BNE @InCave

    ; In a cellar.
    ;
    ; If Link's Y >= $40 or input direction <> up, return.
    LDA ObjY
    CMP #$40
    BCS L14EC6_Exit
    LDA ObjInputDir
    AND #$08
    BEQ L14EC6_Exit

    ; Look for this room's ID in the 6-element cellar room array.
    LDY #$06
    LDA RoomId
    PHA
:
    DEY
    CMP LevelInfo_CellarRoomIdArray, Y
    BNE :-

    ; Determine the destination room ID, and go there.
    ;
    ; If Link's X < $80 look in level block attributes A, else B.
    ;
    ; Note that in cellars, level block attributes A and B indicate
    ; the destination room ID only.
    TAY
    LDA ObjX
    CMP #$80
    BCS @GetRightSideRoom
    LDA LevelBlockAttrsA, Y
    JMP :+

@GetRightSideRoom:
    LDA LevelBlockAttrsB, Y
:
    JSR GoToModeAFromCellar

    ; Set Link's position in the destination room.
    PLA
    TAY
    LDA LevelBlockAttrsC, Y
    PHA
    AND #$F0
    STA ObjX
    PLA
    ASL
    ASL
    ASL
    ASL
    ORA #$0D
    STA ObjY
    RTS

@InCave:
    ; In a cave.
    ;
    ; Check whether a person is blocking the upper half of the room.
    PHA
    JSR CheckPersonBlocking
    PLA

    ; If mode is not $C (shortcuts), go check the screen edge.
    CMP #$0C
    BNE CheckCaveEdge

    ; In a shortcut cave (mode $C).
    ;
    ; If grid offset <> 0, return.
    LDA ObjGridOffset
    BNE L14F72_Exit

    ; If Link's Y <> $9D, go check the screen edge.
    LDA ObjY
    CMP #$9D
    BNE CheckCaveEdge

    ; See if Link is on one of the 3 shortcut stairs.
    ; X = $50: 1
    ; X = $80: 2
    ; X = $B0: 3
    ;
    ; If he's not on any, then return.
    LDY #$01
    LDA ObjX
    CMP #$50
    BEQ :+
    INY
    CMP #$80
    BEQ :+
    INY
    CMP #$B0
    BNE L14F72_Exit
:
    ; Look for the current room in the cellar/shortcut room array.
    STY $00
    LDY #$FF
    LDA RoomId
:
    INY
    CMP LevelInfo_CellarRoomIdArray, Y
    BNE :-

    ; Add the value of the shortcut chosen to whatever index
    ; the current room has in the array.
    ;
    ; This yields the index of one of the three shortcut rooms
    ; after the current one.
    TYA
    CLC
    ADC $00

    ; Wrap around if needed. Go to that room by way of mode $A.
    AND #$03
    TAY
    LDA LevelInfo_CellarRoomIdArray, Y

; Params:
; A: destination room ID
;
GoToModeAFromCellar:
    STA RoomId
    JSR MarkRoomVisited

GoToModeAFromCave:
    LDA #$0A
    STA GameMode

EndPrepareMode:
    LDA #$00
    STA GameSubmode
    STA IsUpdatingMode
    STA $0F
    STA ObjState
    STA ObjShoveDir
    STA ObjShoveDistance
    STA ObjInvincibilityTimer

L14F72_Exit:
    RTS

CheckCaveEdge:
    JSR CheckScreenEdge

    ; If the player touched the edge of the screen and triggered
    ; a transition to another mode, then go set up the right mode.
    LDA IsUpdatingMode
    BEQ GoToModeAFromCave
    RTS
