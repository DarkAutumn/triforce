; ============================================================================
; FILE:     bank1/shot-aim-boomerang-speed.asm BANK: 1 (stored at $A500-$B76F; copied to RAM and run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 3638-3851
; SUBJECTS: (none)
; TOPIC:    MoveShot, direction/distance to target, diagonal speed, boomerang speed
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0026 MoveShot                            code  Params:
;   L0066 GetDirectionsAndDistancesToTarget   code  Params:
;   L0100 _CalcDiagonalSpeedIndex             code  Params:
;   L0174 SetBoomerangSpeed                   code  Params:
;   L0207 GetOneDirectionAndDistanceToTarget  code  Params:
;   L0229 Dir_ToTarget_Return                 code
; ============================================================================

; Params:
; A: direction
; X: object index
; [0E]: 0 to change grid offset, else leave it as is
;
; Returns:
; [0E]: $80 if blocked
; [0F]: moving direction, or 0 if boundary was crossed
;
MoveShot:
    ; If a boundary was crossed, go set [0E] to $80 instead of moving.
    JSR BoundByRoomWithA
    BEQ @ReturnBlocked

    ; Move the object as if grid offset were 0.
    LDA ObjGridOffset, X
    PHA
    LDA #$00
    STA ObjGridOffset, X
    JSR MoveObject
    PLA

    ; If [0E] is set, keep the original value in grid offset,
    ; or else add the new amount to it.
    LDY $0E
    BNE :+
    CLC
    ADC ObjGridOffset, X
:
    STA ObjGridOffset, X
    RTS

@ReturnBlocked:
    LDA #$80
    STA $0E
    RTS

; Params:
; A: object index of target
; X: object index of origin
; [00]: 0
;
; Returns:
; [00]: the number of axes where the distance <= 8
; [03]: distance X
; [04]: distance Y
; [0A]: vertical direction from origin to target
; [0B]: horizontal direction from origin to target
;
GetDirectionsAndDistancesToTarget:
    ; Handle the horizontal.
    PHA
    TAY
    LDA #$02
    STA $0A
    LDA a:ObjX, Y
    LDY ObjX, X
    JSR GetOneDirectionAndDistanceToTarget
    STA $03
    LDA $0A
    STA $0B

    ; Handle the vertical.
    PLA
    TAY
    LDA #$08
    STA $0A
    LDA a:ObjY, Y
    LDY ObjY, X
    JSR GetOneDirectionAndDistanceToTarget
    STA $04
    RTS

; Params:
; Y: middle speed index (4)
; [03]: horizontal distance
; [04]: vertical distance
; [0A]: vertical direction
; [0B]: horizontal direction
;
; Returns:
; Y: speed index (0 to 8) for the angle
;
_CalcDiagonalSpeedIndex:
    ; TODO: call it speed index or angle?
    ; Store the middle speed index in [00]. It will be changed to the
    ; right speed index to use incrementally.
    ; Lower speed indexes yield faster X speeds and lower Y speeds.
    ; Higher speed indexes yield faster Y speeds and lower X speeds.
    STY $00

    ; Assuming horizontal distance >= vertical distance, the
    ; index offset in [01] is negative (-1) to go towards the X axis.
    LDA #$FF
    STA $01

    ; TODO:
    ; If horizontal distance [03] < vertical distance [04], swap [03] and [04],
    ; and store positive speed index 1 in [01] to go towards the Y axis.
    ;
    ; The point is to put the greater value in [03] and the lesser one in [04],
    ; and use a speed index offset (1 or -1) that will point us in the direction
    ; of the farther distance faster.
    LDA $03
    CMP $04
    BCS @Swap
    PHA
    LDA $04
    STA $03
    PLA
    STA $04
    LDA #$01
    STA $01

@Swap:
    ; TODO: call it speed index or angle?
    ; If the difference in distances is within 8 pixels,
    ; return the speed index (angle) we have.
    LDA $03
    SEC
    SBC $04
    CMP #$08
    BCC @Return

@Turn:
    ; TODO: call it speed index or angle?
    ; Otherwise, go to the next speed index (angle).
    LDA $00
    CLC
    ADC $01
    STA $00

    ; TODO: call it speed index or angle?
    ; If a speed index (angle) limit (0 or 8), we've gone all the way to an axis
    ; (0, 90, 180, 270 degrees). So return this speed index (angle).
    BEQ @Return
    CMP #$08
    BEQ @Return

    ; Subtract the lesser distance from the greater distance.
    ; If they still haven't crossed, then we can turn more.
    LDA $03
    SEC
    SBC $04
    STA $03
    CMP $04
    BCS @Turn

@Return:
    ; TODO: call it speed index or angle?
    ; Return the speed index (angle) we found.
    LDY $00
    RTS

; Params:
; A: q-speed fraction
;
SetBoomerangSpeed:
    STA ObjQSpeedFrac, X

    ; If in major state $50, use this speed as is.
    LDA ObjState, X
    AND #$F0
    CMP #$40
    BNE @Exit

    ; Else, in major state $40, go at half the speed.
    LSR ObjQSpeedFrac, X

    ; Once we've traveled the target amount of time, set state $50 to go faster.
    ; In major state $40, ObjMovingLimit is a timer to change the state to $50.
    DEC ObjMovingLimit, X
    BNE @Exit
    LDA #$50
    STA ObjState, X

@Exit:
    RTS

; Params:
; A: target coordinate
; Y: origin coordinate
; [00]: a value
; [0A]: the decreasing direction on the coordinates' axis (2 or 8)
;
; Returns:
; A: the distance from origin to target
; [00]: the original value + 1, if < 8 pixels between origin and target
; [0A]: the direction from origin to target
;
GetOneDirectionAndDistanceToTarget:
    ; If the origin coordinate already >= target coordinate, then
    ; the direction in [0A] is already correct. Otherwise, shift it right
    ; to flip it.
    ;
    ; After this, [02] will >= [01].
    STA $01
    STY $02
    CPY $01
    BCS @Subtract
    STA $02
    STY $01
    LSR $0A

@Subtract:
    ; If the difference between the coordinates < 9, increment [00].
    LDA $02
    SEC
    SBC $01
    CMP #$09
    BCS Dir_ToTarget_Return
    INC $00
Dir_ToTarget_Return:
    RTS
