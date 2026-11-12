; ============================================================================
; FILE:     bank5/link-handle-input.asm  BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 6919-7253
; SUBJECTS: Link
; TOPIC:    Link_HandleInput: buttons, grid turning rules, speed
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0027 AxisMasks                  data  Indexed by reverse direction index.
;   L0030 Link_HandleInput           code
;   L0068 Link_ModifyDirAtGridPoint  code
;   L0109 L172FC_Exit                code
;   L0112 HaveInput                  code
;   L0232 InitLinkSpeed              code
;   L0259 Link_ModifyDirOnGridLine   code
;   L0290 SetObjDirAndInputDir       code  Params:
;   L0293 SetLinkInputDir            code
; ============================================================================

; Indexed by reverse direction index.
;
; Is used in mapping a direction to a mask for its axis.
; For example:
; - right masks off up and down
; - up masks off left and right
;
AxisMasks:
    .BYTE $0C, $0C, $03, $03

Link_HandleInput:
    ; If state = 0, handle A and B buttons.
    LDA ObjState
    BNE @CheckMovement
    JSR Link_FilterInput

    ; If the sword is not blocked, then handle the sword if A is pressed.
    LDA SwordBlockedLongTimer
    ORA IsSwordBlocked
    BNE :+
    LDA ButtonsPressed
    AND #$80
    BEQ :+
    JSR WieldSword
:
    ; If B is pressed, handle the item.
    LDA ButtonsPressed
    AND #$40
    BEQ @CheckMovement
    JSR WieldItem

@CheckMovement:
    ; If player was shoved, return.
    LDX #$00
    LDA ObjShoveDir
    BNE L172FC_Exit

    ; If in UW, then move correctly inside doorways.
    LDA CurLevel
    BEQ :+
    JSR Link_ModifyDirInDoorway
:
    ; Change directions according to whether the player is at an intersection point
    ; (grid offset = 0) or between points along a line (grid offset <> 0).
    LDA ObjGridOffset
    BEQ Link_ModifyDirAtGridPoint
    JMP Link_ModifyDirOnGridLine

Link_ModifyDirAtGridPoint:
    ; Grid offset = 0, so A = 0 here.
    ; Reset some variables.
    ; [0B] holds input direction count.
    STA $0B
    STA $0C                     ; [0C] holds walkable direction count.
    STA Link_GoStraight         ; Reset this. We'll figure out if needs to be set again.

    ; Look for walkable directions that are components of the
    ; input directions.
    ;
    ; Keep track of how many and which directions were input
    ; directions and walkable.
    LDY #$03

@LoopDir:
    LDA ObjInputDir
    AND ReverseDirections, Y
    BEQ @NextLoopDir            ; If this direction doesn't match, skip it.
    STA $0F                     ; [0F] holds the last input direction found.
    TYA                         ; Save the reverse direction index.
    PHA
    INC $0B                     ; Increment the input dir count in [0B].
    JSR GetCollidingTileMoving
    CMP RoomFirstUnwalkableTile
    BCS :+                      ; If this is unwalkable, skip it.
    LDA $0F
    STA $0D                     ; [0D] holds the last walkable direction found.
    INC $0C                     ; Increase the walkable direction count in [0C].
:
    PLA                         ; Restore the reverse direction index.
    TAY

@NextLoopDir:
    DEY
    BPL @LoopDir

    ; If there were no input directions, then return.
    LDY $0B
    BNE HaveInput

L172FC_Exit:
    RTS

HaveInput:
    ; There were input directions.
    ;
    ; If there was only one input direction, go set object direction
    ; to input direction, and set Link's speed.
    ;
    ; Also, *RESET* Link_GoStraightWhenDiagInput
    ; (X is still 0, because it's Link's object index).
    LDA $0F
    CPY #$01
    BEQ @SetLinkDirAndSpeed

    ; There were more than one input directions.
    ;
    ; If none were walkable, go set input direction to 0, and return.
    LDA $0C
    BNE :+
    JMP SetLinkInputDir
:
    ; Two input directions, and at least one of them is walkable.
    ;
    ; Make Link go straight in one direction on the next grid line.
    TAY
    INC Link_GoStraight

    ; If in OW or only one direction of the two is walkable, then go set
    ; object direction and input direction to last walkable direction
    ; found, and set Link's speed.
    ;
    ; Also, *RESET* Link_GoStraightWhenDiagInput.
    LDX #$00
    LDA $0D                     ; [0D] holds last walkable direction found.
    CPY #$01
    BEQ @SetLinkDirAndSpeed
    LDY CurLevel
    BEQ @SetLinkDirAndSpeed

    ; In UW. There are two input directions, and they're both walkable.
    ;
    ; Normally, turn to the direction that's perpendicular to object
    ; direction. But keep going straight after that, while diagonal
    ; input is held.
    ;
    ; Handle special cases for doors.

    ; 1. Link at horizontal doors.
    ;
    ; If Link's X = $20 or $D0 then
    ;   If Link's Y = $85 and facing down then
    ;     Go set object direction and input direction to object
    ;     direction, and set Link's speed. Also, *RESET*
    ;     Link_GoStraightWhenDiagInput.
    ;   Else
    ;     Go turn to the perpendicular direction
    LDY ObjX
    CPY #$20
    BEQ :+
    CPY #$D0
    BNE :++
:
    LDY ObjY
    CPY #$85
    BNE :++
    LDA ObjDir
    AND #$04
    BEQ :++

@SetLinkDirToObjDir:
    LDA ObjDir
    BNE @SetLinkDirAndSpeed
:
    ; If allowed to turn when input is diagonal (2 directions), then
    ; go take the direction perpendicular to object direction.
    LDA ObjDir
    LDX Link_GoStraightWhenDiagInput
    BEQ :+

    ; At this point, Link_GoStraightWhenDiagInput is true.

    ; 2. Link at top door.
    ;
    ; If Link's X <> $78 or Link's Y <> $5D then
    ;   Go set object direction and input direction to object
    ;   direction, and set Link's speed. Also, *SET*
    ;   Link_GoStraightWhenDiagInput.
    LDY CurLevel
    BEQ @SetLinkDirAndSpeed     ; Also do it if in OW. But if in OW, we returned already.
    LDY ObjX
    CPY #$78
    BNE @SetLinkDirAndSpeed
    LDY ObjY
    CPY #$5D
    BNE @SetLinkDirAndSpeed

    ; If object direction is vertical then
    ;   Go set object direction and input direction to object
    ;   direction, and set Link's speed. Also, *SET*
    ;   Link_GoStraightWhenDiagInput.
    AND #$03
    BEQ @SetLinkDirToObjDir
:
    ; Find the input direction that's perpendicular to the
    ; object's direction.
    LDA ObjDir
    INX                         ; Make sure to set Link_GoStraightWhenDiagInput.
    JSR GetOppositeDir          ; Do this to get reverse index of object direction.
    LDA ObjInputDir
    PHA                         ; Save input directions.
    AND AxisMasks, Y            ; Mask input directions with the mask for the object's direction's axis.
    STA $0C
    PLA                         ; Restore input directions.
    EOR $0C

@SetLinkDirAndSpeed:
    ; Set object and input direction to value in A.
    ; Set Link_GoStraightWhenDiagInput to value X.
    STX Link_GoStraightWhenDiagInput
    JSR SetObjDirAndInputDir
    LDX #$00

InitLinkSpeed:
    LDA #$60                    ; Link gets a quarter speed (QSpeed) of $60 by default (1.5 pixels a frame).
    STA $00
    LDA CurLevel
    BNE @SetSpeed               ; If in UW, go use this speed.

    ; In OW. If standing on mountain stairs, then
    ; use a lower quarter speed of $30.
    LDA ObjCollidedTile
    CMP #$74
    BEQ :+
    CMP #$75
    BNE @SetSpeed
:
    LDA #$30
    STA $00
    CMP ObjQSpeedFrac
    BEQ @SetSpeed               ; If Link's speed is not this lower speed,
    LDA #$00                    ; then reset the position fraction.
    STA ObjPosFrac

@SetSpeed:
    LDA $00
    STA ObjQSpeedFrac
:
    RTS

Link_ModifyDirOnGridLine:
    ;
    ; If not moving, then return.
    LDA ObjInputDir
    BEQ :-

    ; If there's more than one component in the input direction,
    ; then take only one of them.
    ;
    ; After this point, Y holds the reverse index of this single direction.
    JSR GetOppositeDir
    LDA ReverseDirections, Y

    ; If the single input direction matches object direction,
    ; then keep going in this direction.
    CMP ObjDir
    BEQ InitLinkSpeed

    ; If the single input direction is the opposite of object direction, then
    ; change to the single input direction.
    ORA ObjDir
    CMP #$03                    ; Combined opposite horizontals (1 OR 2).
    BEQ :+
    CMP #$0C                    ; Combined opposite verticals (4 OR 8).
    BNE :++                     ; If the directions are perpendicular, go handle this case.
:
    LDA ReverseDirections, Y

; Params:
; A: direction
;
SetObjDirAndInputDir:
    STA ObjDir

SetLinkInputDir:
    STA ObjInputDir
    RTS
:
    ; The directions are perpendicular.
    ;
    ; Keep going in facing direction, if that's what Link's been told to do.
    LDA Link_GoStraight
    BNE InitLinkSpeed

    ; Link's movement grid cell size is 8. If he's moved half that
    ; length or more, then return.
    LDA ObjGridOffset
    JSR Abs
    PHA
    LDA ObjDir
    JSR GetOppositeDir
    STA $01                     ; [01] holds the opposite of facing direction.
    PLA
    CMP #$04
    BCS @Exit

    ; If Link had turned back and is facing a grid point that he
    ; had started walking from, then return.
    LDA ObjDir
    AND #$0A
    BEQ :+
    LDA ObjGridOffset
    BPL @Exit
    BMI @ReverseDir
:
    LDA ObjGridOffset
    BMI @Exit

@ReverseDir:
    ; Reverse Link's direction.
    LDA $01
    STA ObjDir

    ; Reverse the grid offset. Yield an offset for the same position
    ; in the line, but in the opposite direction. For example,
    ; -1 => 7
    ;  3 => -5
    ;
    ; Positive offset: -8 - -offset
    ; Negative offset:  8 - -offset
    LDA #$08
    LDY ObjGridOffset
    BMI :+
    LDA #$F8
:
    PHA
    TYA
    JSR Negate
    STA $01
    PLA
    SEC
    SBC $01
    STA ObjGridOffset

@Exit:
    RTS
