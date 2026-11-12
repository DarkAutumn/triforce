; ============================================================================
; FILE:     bank7/walker-collision-screen-edge.asm BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 2829-3161
; SUBJECTS: (none)
; TOPIC:    Walker tile collision, CheckScreenEdge, GoToNextModeFromPlay, alternate directions
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0035 PlayerScreenEdgeBounds                      data  Up, down, left, right.
;   L0061 Walker_CheckTileCollision                   code  Params:
;   L0110 Reverse                                     code
;   L0119 CheckTiles                                  code
;   L0139 CheckObjAttr10AndAltDir                     code
;   L0147 TryNextDir                                  code
;   L0156 PlayerUnwalkable                            code
;   L0177 L1F148_Exit                                 code
;   L0184 ResetMovingDir                              code  Returns:
;   L0189 GoWalkableDir                               code
;   L0209 CheckScreenEdge                             code
;   L0240 GoToNextModeFromPlay                        code  Returns:
;   L0251 ExitX0                                      code
;   L0255 CheckBoundary                               code
;   L0276 Walker_GetNextAltDir                        code  Description:
;   L0281 Walker_GetNextAltDir_JumpTable              data
;   L0287 Walker_AltDir_GetRandomObjPerpendicularDir  code
;   L0303 Walker_AltDir_GetMovingOppositeDir          code
;   L0317 ReverseObjDir                               code
;   L0326 Walker_AltDir_EndLoop                       code
;   L0337 _FaceUnblockedDir                           code  Description:
;   L0364 L1F1FC_Exit                                 code
; ============================================================================

; Up, down, left, right.
;
PlayerScreenEdgeBounds:
    .BYTE $3D, $DD, $00, $F0

; Params:
; [0E]: $FF, if blocked by a door
; [0F]: movement direction
;
; Returns:
; [0F]: untouched, or changed if blocked or need to change
;
; Summary:
;
; For the player: if at a doorway or walkable tile, check the screen edge.
; Then allow movement or go to the next room. Otherwise,
; check passive tile objects and stop moving.
;
; For other objects: First, if moving, test the colliding tile for
; walkability. Otherwise search for a walkable direction starting
; from input direction.
;
; When you find a walkable direction, see if it makes the object
; cross a boundary. If it doesn't, then make this direction
; the objection direction. Otherwise, keep looping.
;
; If no walkable direction is found, then stop moving.
;
Walker_CheckTileCollision:
    ; if is player
    ;   if at doorway
    ;     go handle a walkable direction
    ;   if blocked by door
    ;     return
    CPX #$00
    BNE @CheckGridOffset
    LDA DoorwayDir
    BEQ :+
    JMP GoWalkableDir
:
    LDA $0E
    BMI L1F148_Exit

@CheckGridOffset:
    ; If grid offset <> 0, return.
    LDA ObjGridOffset, X
    BNE L1F148_Exit

    ; Grid offset = 0.
    ; The object is at a point between cells in the grid.
    ; So, we can check tiles for collision.
    ;
    ; Reset [0E] alternate direction search step.
    STA $0E

    ; If there's a movement direction, go check tile collision.
    LDA $0F
    BNE CheckTiles

    ; Moving direction = 0.
    ; If object is the player, return.
    CPX #$00
    BEQ L1F148_Exit

    ; Moving direction = 0.
    ; The object is not the player.
    ;
    ; If object does not have attribute $10, then
    ; set movement direction to input direction,
    ; and go try to turn to an unblocked direction in order to move.
    LDA ObjAttr, X
    AND #$10                    ; "Reverse when blocked" object attribute
    BNE Reverse
    LDA ObjInputDir, X
    STA $0F
    JMP TryNextDir

Reverse:
    ; UNKNOWN:
    ; This seems to be unused code triggered by object attribute $10.
    ; But $10 is not used in the object attribute array at 07:FAEF.
    ; Begin unverified code 1F110
    JSR ReverseObjDir
    JMP CheckBoundary
    ; End unverified code

CheckTiles:
    ; The code below applies to Link and other objects.
    ;
    ; Test the moving direction's walkability.
    ;
    ; For the player, this is enough. For other objects, this 
    ; becomes a loop looking for an unblocked direction to move in.
    ;
    ; If the colliding tile is walkable, go handle a walkable direction.
    JSR GetCollidingTileMoving
    CMP ObjectFirstUnwalkableTile
    BCC GoWalkableDir

    ; The colliding tile is unwalkable.
    ;
    ; If object is the player, then go check passive tile objects and
    ; the screen edge, or stop moving.
    CPX #$00
    BEQ PlayerUnwalkable

CheckObjAttr10AndAltDir:
    ; Not the player.
    ; If object attribute has $10, go to the unused code above
    ; to reverse direction.
    LDA ObjAttr, X
    AND #$10
    BNE Reverse

TryNextDir:
    ; Get the next direction to check, and set moving direction to it.
    ; If not at the end of the loop, go test the new direction's walkability.
    JSR Walker_GetNextAltDir
    STA $0F                     ; Set new moving direction, or 0 at the end of the loop.
    LDA $0E
    BNE CheckTiles
    RTS

PlayerUnwalkable:
    ; Tile is not walkable. Object is player.
    ;
    ; Check passive tile objects and the screen edge, and stop moving.
    ;
    ; First, if in OW, check passive tile objects (armos and regular gravestone).
    LDA CurLevel
    BNE @StopMoving
    LDA #$01
    JSR SwitchBank
    JSR CheckPassiveTileObjects

@StopMoving:
    ; Reset moving direction and buttons pressed.
    ; If in UW, we're done.
    ; In OW, go check the screen edge.
    JSR ResetMovingDir
    STA ButtonsPressed
    LDA CurLevel
    BEQ GoWalkableDir

L1F148_Exit:
    RTS

; Returns:
; A: 0
; [0F]: 0
;
ResetMovingDir:
    LDA #$00
    STA $0F
    RTS

GoWalkableDir:
    ; If object is not Link, then the tile is walkable. Go see if this direction
    ; makes the object cross a boundary. Then go check another direction if it does,
    ; or else make this the object direction.
    CPX #$00
    BNE CheckBoundary

    ; The object is Link. We end up here regardless of walkability.
    ;
    ; If not in mode 5 or ladder is in use, return.
    LDA GameMode
    CMP #$05
    BNE ExitX0
    LDA LadderSlot
    BNE L1F148_Exit

    ; If grid offset <> 0, return.
    LDA ObjGridOffset
    BNE ExitX0

CheckScreenEdge:
    LDX ObjY

    ; If not moving, then return.
    LDA ObjInputDir
    BEQ ExitX0

    ; If the single moving direction is vertical, use Y.
    ; Else use X coordinate.
    JSR GetOppositeDir          ; Call this to get a single moving direction.
    LDA ReverseDirections, Y
    AND #$0C
    BNE :+
    LDX ObjX
:
    STX $00

    ; If Link's coordinate does not match the screen edge coordinate
    ; in the single moving direction, then return.
    LDA $00
    CMP PlayerScreenEdgeBounds, Y
    BNE ExitX0

    ; We're moving to the next screen.
    ; Make sure Link faces the single moving direction.
    LDA ReverseDirections, Y
    STA ObjDir

; Returns:
; X: 0 (Link's object slot)
;
GoToNextModeFromPlay:
    INC GameMode
    LDA #$00
    STA GameSubmode
    STA IsUpdatingMode
    STA $0F
    STA ObjState
    STA ObjShoveDir
    STA ObjShoveDistance
    STA ObjInvincibilityTimer

ExitX0:
    LDX #$00
    RTS

CheckBoundary:
    ; If object crossed a boundary, then go check object
    ; attribute $10 and another direction.
    ; Else set object direction to moving direction [0F] and return.
    JSR BoundByRoom
    BEQ CheckObjAttr10AndAltDir
    STA ObjDir, X
    RTS

; Description:
; Calculates the next alternate direction to move in while
; searching for an unblocked direction.
;
; Params:
; [0E]: the current step
;
; Returns:
; A: a direction for the current step, or 0 at the end of the loop
; [0E]: the next step, or 0 to end the loop
; ObjDir: might be modified
;
Walker_GetNextAltDir:
    LDA $0E
    INC $0E
    JSR TableJump

Walker_GetNextAltDir_JumpTable:
    .ADDR Walker_AltDir_GetRandomObjPerpendicularDir
    .ADDR Walker_AltDir_GetMovingOppositeDir
    .ADDR ReverseObjDir
    .ADDR Walker_AltDir_EndLoop

Walker_AltDir_GetRandomObjPerpendicularDir:
    LDY #$00
    LDA Random, X
    ASL
    BCS :+
    INY
:
    LDA ObjDir, X
    AND #$0C
    BEQ :+
    INY
    INY
:
    LDA ReverseDirections, Y
    RTS

Walker_AltDir_GetMovingOppositeDir:
    LDA $0F
    PHA
    AND #$0A
    BEQ @IncreasingDir
    PLA
    LSR
    RTS

@IncreasingDir:
    PLA
    ASL
    RTS

ReverseObjDir:
    ; Note:
    ; Also reverses movement direction in [0F].
    LDA ObjDir, X
    JSR GetOppositeDir
    STA ObjDir, X
    STA $0F
    RTS

Walker_AltDir_EndLoop:
    LDA #$00
    STA $0E
    RTS

; Description:
; Search for an unblocked direction. Set facing direction to it,
; if one is found.
;
; If not at a square boundary, then return.
;
_FaceUnblockedDir:
    LDA ObjGridOffset, X
    BNE L1F1FC_Exit

    ; Else reset [0E] to start the search for an unblocked direction.
    STA $0E

@LoopSearchStep:
    JSR Walker_GetNextAltDir

    ; Set the moving direction [0F] to the direction returned.
    STA $0F

    ; But quit if none was found.
    BEQ L1F1FC_Exit

    ; If blocked in this direction by a tile or the room boundary,
    ; then loop again.
    JSR GetCollidingTileMoving
    CMP ObjectFirstUnwalkableTile
    BCS @LoopSearchStep
    JSR BoundByRoom
    BEQ @LoopSearchStep

    ; Set the facing direction to the one found.
    STA ObjDir, X

L1F1FC_Exit:
    RTS
