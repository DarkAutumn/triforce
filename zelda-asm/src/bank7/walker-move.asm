; ============================================================================
; FILE:     bank7/walker-move.asm        BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 2600-2827
; SUBJECTS: (none)
; TOPIC:    Walker_Move (Link and monsters), MoveObject
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0013 Walker_Move  code
;   L0181 MoveObject   code  Params:
; ============================================================================

Walker_Move:
    ; If the object is being shoved, then go shove it.
    LDA ObjShoveDir, X
    BEQ @ChooseObjDirOrInputDir
    JMP Obj_Shove

@ChooseObjDirOrInputDir:
    ; Summary: Choose object direction or input direction for movement as appropriate.
    ;
    ; Player: If there's input, then use input direction when grid offset = 0,
    ;         or object direction when <> 0.
    ;
    ; Other objects: Move in input direction, unless stunned or magic clock is active.
    ;
    ; if slot = 0 and grid offset <> 0 then
    ;   if input dir = 0 then
    ;     use 0
    ;   else
    ;     use object direction
    ; else
    ;   if slot <> 0 and (has clock or stunned) then
    ;     return
    ;   else
    ;     if input dir = 0
    ;       use 0
    ;     else
    ;       use input direction
    CPX #$00
    BNE @CheckStunned
    LDA ObjGridOffset
    BEQ @CheckStunned
    LDA ObjInputDir
    BEQ @ZeroMovingDir
    LDA ObjDir
    BNE @SetMovingDir

@CheckStunned:
    CPX #$00
    BEQ @FilterInput
    LDA InvClock
    ORA ObjStunTimer, X
    BNE L1EFCF_Exit

@FilterInput:
    LDA ObjInputDir, X
    BEQ @ZeroMovingDir
    JSR GetOppositeDir          ; Use this to take only one input direction, in case there are 2 (diagonal).
    LDA ReverseDirections, Y
    BNE @SetMovingDir

@ZeroMovingDir:
    LDA #$00

@SetMovingDir:
    ; Mask off everything but directions.
    ; (I don't think there's anything else)
    AND #$0F
    STA $0F                     ; [0F] holds the movement direction.

    ; [0E]: $FF, if blocked by a door
    ;       else will hold the reverse index of the direction of a doorway if found
    ; 
    LDA #$00
    STA $0E

    ; If object is Link and using or catching an item then
    ; reset movement direction.
    CPX #$00
    BNE @OnlyLink
    LDA ObjState, X
    AND #$F0
    CMP #$10
    BEQ :+
    CMP #$20
    BNE @OnlyLink
:
    STX $0F                     ; [0F] movement direction

@OnlyLink:
    ; If object is not Link, then skip the code below.
    CPX #$00
    BNE @CheckBoundary

    ; Check if tile objects block the player.
    LDA #$01
    JSR SwitchBank
    JSR CheckTileObjectsBlocking

    ; Check whether a person blocks the player.
    LDA ObjType+1
    CMP #$36                    ; Grumble moblin
    BEQ @CheckBlocked
    CMP #$4B                    ; Person1
    BCC @CheckSubroomOrDoorways
    CMP #$53                    ; Person8
    BCS @CheckSubroomOrDoorways

@CheckBlocked:
    JSR CheckPersonBlocking

@CheckSubroomOrDoorways:
    ; If in a doorway, then go check doorways instead of subrooms.
    LDA DoorwayDir
    BNE @CheckDoorways

    ; Check subrooms (caves and cellars) in modes 9, $B, $C.
    LDA GameMode
    CMP #$09
    BEQ @InSubroom
    CMP #$0B
    BEQ @InSubroom
    CMP #$0C
    BNE @SkipSubroom

@InSubroom:
    LDA #$05
    JSR SwitchBank
    JSR CheckSubroom

    ; If game mode = 9, or in OW, or in a doorway, then skip
    ; checking room boundaries.
    LDA GameMode
    CMP #$09
    BEQ @CheckDoorways

@SkipSubroom:
    LDA CurLevel
    BEQ @CheckDoorways
    LDA DoorwayDir
    BNE @CheckDoorways

@CheckBoundary:
    ; This code applies to all objects.
    ;
    ; Prevents movement outside the walls or border of a room,
    ; even Link in front of a doorway.
    JSR BoundByRoom

@CheckDoorways:
    ; If object is Link, and level is in UW, and mode <> 9,
    ; then check doorways.
    CPX #$00
    BNE @CheckTiles
    LDA CurLevel
    BEQ @CheckTiles
    LDA GameMode
    CMP #$09
    BEQ @CheckTiles
    LDA #$05
    JSR SwitchBank
    JSR CheckDoorway
    LDX #$00                    ; Restore Link's object index.

@CheckTiles:
    ; This code applies to all objects.
    JSR Walker_CheckTileCollision

    ; The player will check for the ladder.
    CPX #$00
    BNE MoveObject
    LDA #$05
    JSR SwitchBank
    JSR CheckLadder             ; CheckLadder

; Params:
; X: object index
; [0F]: direction
;
MoveObject:
    LDA #$08                    ; Positive limit is 8 for Link.
    LDY #$F8                    ; Negative limit is -8 for Link.
    CPX #$00
    BEQ :+                      ; If the object isn't Link,
    LDA #$10                    ; then use $10 and -$10.
    LDY #$F0
:
    STA PositiveGridCellSize    ; Store the limits that we determined.
    STY NegativeGridCellSize
    LDA $0F
    BEQ @Exit                   ; If direction is none, then return.

    ; To allow a wide speed range, we keep the quarter speed.
    ; Here, apply it 4 times to get the full speed.
    JSR @ApplyQSpeedToPosition
    JSR @ApplyQSpeedToPosition
    JSR @ApplyQSpeedToPosition

@ApplyQSpeedToPosition:
    LDA $0F                     ; Check the direction variable for each direction bit.
    LSR
    BCS @Right
    LSR
    BCS @Left
    LSR
    BCS @Down

    ; Up
    JSR SubQSpeedFromPositionFraction
    LDA ObjY, X
    SBC #$00
    STA ObjY, X

@Exit:
    RTS

@Down:
    ; Down
    JSR AddQSpeedToPositionFraction
    LDA ObjY, X
    ADC #$00
    STA ObjY, X
    RTS

@Right:
    ; Right
    JSR AddQSpeedToPositionFraction
    LDA ObjX, X
    ADC #$00
    STA ObjX, X
    RTS

@Left:
    ; Left
    JSR SubQSpeedFromPositionFraction
    LDA ObjX, X
    SBC #$00
    STA ObjX, X
    RTS
