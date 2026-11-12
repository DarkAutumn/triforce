; ============================================================================
; FILE:     bank4/fairy-flyer-common.asm BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 11447-11868
; SUBJECTS: (none)
; TOPIC:    Item fairy; shared flyer state machine
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0044 SetUpFairyObject              code
;   L0063 ResetFlyerState               code
;   L0075 UpdateFairyObject             code  Params:
;   L0079 DrawFairy                     code
;   L0092 ControlFairyFlight            code
;   L0096 ControlFairyFlight_JumpTable  data
;   L0102 Flyer_FairyDecideState        code
;   L0109 Flyer_DoNothing               code
;   L0112 Directions8                   data
;   L0115 Flyer_Delay                   code
;   L0124 Flyer_SlowDown                code
;   L0130 Flyer_SpeedUp                 code
;   L0149 Flyer_SetFlyingState          code
;   L0154 Flyer_CompareMaxSpeed         code
;   L0160 MoveFlyer                     code
;   L0219 BoundFlyer                    code  Returns:
;   L0234 ReverseObjDir8                code
;   L0253 L132F8_Exit                   code
;   L0256 DeferBounce                   code
;   L0267 L13307_Exit                   code
;   L0274 Flyer_Chase                   code  Description:
;   L0284 SetFlyingState1               code
;   L0288 SetDelayAndTurn               code
;   L0293 TurnTowardsPlayer8            code
;   L0371 LoopRight                     code
;   L0379 NextLoopRight                 code
;   L0388 SetDir8ForIndex               code
;   L0393 L1336F_Exit                   code
;   L0396 TestDir                       code
;   L0413 Flyer_Wander                  code  Description:
;   L0428 TurnRandomlyDir8              code
;   L0452 GetObjDir8Index               code  Returns:
;   L0464 L133AC_Exit                   code
; ============================================================================

SetUpFairyObject:
    ; Play "fairy appears" tune. It's also "object taken".
    LDA #$08
    STA Tune1Request
    JSR ResetFlyerState

    ; Face up at first.
    LDA #$08
    STA ObjDir, X

    ; Set flying speed fraction to $7F.
    LDA #$7F
    STA Flyer_ObjSpeed, X

    ; Set maximum speed fraction to $A0.
    LDA #$A0
    STA FlyingMaxSpeedFrac
    RTS

ResetFlyerState:
    LDA #$00
    STA Flyer_ObjSpeedFrac, X
    STA Flyer_ObjTurns, X
    STA Flyer_ObjDistTraveled, X
    STA Flyer_ObjFlyingState, X
    STA ObjInvincibilityTimer, X
    RTS

; Params:
; X: object index
;
UpdateFairyObject:
    JSR ControlFairyFlight
    JSR MoveFlyer

DrawFairy:
    JSR Anim_FetchObjPosForSpriteDescriptor
    JSR Anim_SetSpriteDescriptorRedPaletteRow

    ; Every 4 frames, switch between the two sprite frames.
    ASL
    AND FrameCounter
    LSR
    LSR
    STA $0C
    LDY #$14
    JMP Anim_WriteItemSprites

ControlFairyFlight:
    LDA Flyer_ObjFlyingState, X
    JSR TableJump

ControlFairyFlight_JumpTable:
    .ADDR Flyer_SpeedUp
    .ADDR Flyer_FairyDecideState
    .ADDR Flyer_DoNothing
    .ADDR Flyer_Wander

Flyer_FairyDecideState:
    ; Set up 6 turns, and go to state 3.
    LDA #$03
    STA Flyer_ObjFlyingState, X
    LDA #$06
    STA Flyer_ObjTurns, X

Flyer_DoNothing:
    RTS

Directions8:
    .BYTE $08, $09, $01, $05, $04, $06, $02, $0A

Flyer_Delay:
    ; Delay until ObjTimer[X] expires. Then go to flying state 0.
    LDA ObjTimer, X
    BNE :+
    LDA #$00
    STA Flyer_ObjFlyingState, X
:
    RTS

Flyer_SlowDown:
    ; Decrease speed each frame.
    ; When it goes below a threshold ($20), go to flying state 5.
    DEC Flyer_ObjSpeed, X
    JMP :+

Flyer_SpeedUp:
    ; Increase speed each frame.
    ; When it reaches a threshold, go to flying state 1.
    INC Flyer_ObjSpeed, X
:
    LDA Flyer_ObjSpeed, X
    AND #$E0
    BNE Flyer_CompareMaxSpeed

    ; (whole_speed & $E0) = 0
    ; Set a random timer between $40 and $7F.
    LDA Random, X
    AND #$3F
    ORA #$40
    STA ObjTimer, X

    ; Go to flying state 5.
    LDA #$05

Flyer_SetFlyingState:
    STA Flyer_ObjFlyingState, X
:
    RTS

Flyer_CompareMaxSpeed:
    CMP FlyingMaxSpeedFrac
    BCC :-                      ; If (whole_speed & $E0) < threshold, return.
    LDA #$01                    ; Else go to flying state 1, and return.
    JMP Flyer_SetFlyingState

MoveFlyer:
    ; Add (flying speed AND $E0) to flying position fraction.
    LDA Flyer_ObjSpeed, X
    AND #$E0
    CLC
    ADC Flyer_ObjSpeedFrac, X
    STA Flyer_ObjSpeedFrac, X
    BCC @Exit                   ; If the fraction isn't whole yet, then return.

    ; Because this object can move in 8 directions,
    ; test the object's direction with each cardinal direction,
    ; to modify the coordinates accordingly.
    ;
    ; Test right.
    LDA #$01
    STA $02
    LDA ObjDir, X
    BIT $02
    BEQ @Left                   ; Right is not a component? Go check next direction.
    INC ObjX, X
    INC Flyer_ObjOffsetX, X

@Left:
    ; Test left.
    ASL $02
    BIT $02
    BEQ @Down                   ; Left is not a component? Go check next direction.
    DEC ObjX, X
    DEC Flyer_ObjOffsetX, X

@Down:
    ; Test down.
    ASL $02
    BIT $02
    BEQ @Up                     ; Down is not a component? Go check next direction.
    INC ObjY, X
    INC Flyer_ObjOffsetY, X

@Up:
    ; Test up.
    ASL $02
    BIT $02
    BEQ @End                    ; Up is not a component? Go finish up.
    DEC ObjY, X
    DEC Flyer_ObjOffsetY, X

@End:
    ; Increase the distance traveled; and keep the object within
    ; the bounds of the room.
    INC Flyer_ObjDistTraveled, X
    JSR BoundFlyer

@Exit:
    RTS

; Returns:
; A: original facing or the opposite
; [0F]: 0 if blocked
;
BoundFlyer:
    LDA ObjDir, X
    STA $0F
    JSR BoundDirectionHorizontally

    ; Check vertical boundaries, if object is not a boulder.
    LDA ObjType, X
    CMP #$20
    BEQ :+
    JSR BoundDirectionVertically
:
    ; If movement wasn't restricted, then return.
    LDA $0F
    BNE L132F8_Exit

ReverseObjDir8:
    ; Get the opposite direction of the one the object is facing.
    JSR GetObjDir8Index
    TYA
    CLC
    ADC #$04
    AND #$07
    TAY

    ; If the object type = $41 (moldorm), then go see if we need
    ; to set the deferred bounce direction.
    LDA ObjType, X
    CMP #$41
    BEQ DeferBounce

    ; Apply the new direction.
    LDA Directions8, Y
    STA ObjDir, X

L132F8_Exit:
    RTS

DeferBounce:
    ; If the object is segment 5 or $A of moldorm (a head), then
    ; assign the new direction to the deferred bounce direction.
    CPX #$05
    BEQ :+
    CPX #$0A
    BNE L13307_Exit
:
    LDA Directions8, Y
    STA Moldorm_ObjBounceDir, X

L13307_Exit:
    RTS

; Description:
; Turn towards the player a number of times.
; Then go to state 1. After each turn delay $10 frames.
;
Flyer_Chase:
    ; Delay until timer = 0.
    LDA ObjTimer, X
    BNE L13307_Exit             ; If timer is not expired, then return.

    ; Decrease the turn counter.
    ; Once there are no more turns, go to flying state 1.
    DEC Flyer_ObjTurns, X
    BNE SetDelayAndTurn

SetFlyingState1:
    LDA #$01
    JMP Flyer_SetFlyingState

SetDelayAndTurn:
    ; Set a delay of $10 frames.
    LDA #$10
    STA ObjTimer, X

TurnTowardsPlayer8:
    ; Build a direction in [00] towards the player.
    ;
    ; First, calculate a horizontal component:
    ;
    ; If   observed player X = object X, [00] := 0 (none)
    ; elif observed player X > object X, [00] := 1 (right)
    ; else                               [00] := 2 (left)
    LDY #$00
    STY $00
    INY
    LDA ChaseTargetX
    CMP ObjX, X
    BEQ @CalcVertical
    BCS :+
    INY
:
    STY $00

@CalcVertical:
    ; Second, calculate a vertical component.
    ; Combine it with the horizontal component:
    ;
    ; Y := 4 (down)
    ; if observed player Y <> object Y then
    ;   if observed player Y < object Y then
    ;     Y := Y << 1   -- makes it 8 (up)
    ;   [00] := [00] | Y
    LDY #$01
    LDA ChaseTargetY
    CMP ObjY, X
    BEQ @SkipVertical
    BCS :+
    INY
:
    TYA
    ASL
    ASL
    ORA $00
    STA $00

@SkipVertical:
    ; Get the index of the object's direction.
    JSR GetObjDir8Index

    ; Check 3 directions turning left:
    ; 1. right from object direction
    ; 2. object direction
    ; 3. left from object direction
    ;
    ; If the target direction matches any of these, then
    ; leave the object's direction alone.
    LDA #$03
    STA $01                     ; [01] holds the count of directions to check
    INY                         ; Turn right.

@LoopLeft:
    TYA
    AND #$07                    ; Modulo 8 to roll over the index as needed.
    TAY
    LDA Directions8, Y
    CMP $00
    BEQ L1336F_Exit             ; Return if it matches.
    DEY                         ; Turn left.
    DEC $01
    BNE @LoopLeft

    ; Check 3 directions turning right:
    ; 1. left of object direction
    ; 2. object direction
    ; 3. right of object direction
    ;
    ; If any direction component of the target direction matches
    ; these, then see if we should turn to this direction.
    LDA #$03
    STA $01
    INY                         ; This brings the index to one turn left of the object's direction.

LoopRight:
    TYA
    AND #$07                    ; Modulo 8 to roll over the index as needed.
    TAY
    LDA Directions8, Y
    BIT $00
    BNE TestDir

NextLoopRight:
    INY                         ; Turn right.
    DEC $01
    BNE LoopRight

    ; We didn't find a direction to switch to.
    ; So turn left once; to one turn right of object direction.
    DEY

SetDir8ForIndex:
    ; Apply the new direction.
    LDA Directions8, Y
    STA ObjDir, X

L1336F_Exit:
    RTS

TestDir:
    ; If (test direction OR target direction) < 7, then allow this turn.
    ; Else go try the next test direction.
    ;
    ; For 3/5 of the arrangements of angles between Link and
    ; objects; this test yields a turn towards Link. The rest of
    ; the time, the object will keep going in its original direction
    ; or turn away from Link.
    ORA $00
    CMP #$07
    BCS NextLoopRight
    BCC SetDir8ForIndex

; Description:
; Delay and turn randomly a number of times.
; The go to state 1. After each turn, delay $10 frames.
;
Flyer_Wander:
    ; Delay until timer = 0.
    LDA ObjTimer, X
    BNE L133AC_Exit

    ; Decrease the turn counter.
    ; Once there are no more turns, go to flying state 1.
    DEC Flyer_ObjTurns, X
    BNE :+
    JMP SetFlyingState1
:
    ; Set a delay of $10 frames.
    LDA #$10
    STA ObjTimer, X

TurnRandomlyDir8:
    ; Turn according to a random value:
    ; >= $A0: don't turn
    ; >= $50: turn right
    ; Else:   turn left
    JSR GetObjDir8Index
    LDA Random+1, X
    CMP #$A0
    BCS @SetDir
    INY
    CMP #$50
    BCS @SetDir
    DEY
    DEY

@SetDir:
    TYA
    AND #$07
    TAY
    JMP SetDir8ForIndex         ; Go set object direction for 8-way index.

; Returns:
; Y: index of 8-way direction of flyer object
;
GetObjDir8Index:
    LDY #$07
:
    LDA ObjDir, X
    CMP Directions8, Y
    BEQ L133AC_Exit
    DEY
    BPL :-
    ; Begin unverified code 133AB
    INY                         ; If not found, then use index 0.
    ; End unverified code

L133AC_Exit:
    RTS
