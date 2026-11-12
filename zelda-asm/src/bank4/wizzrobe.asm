; ============================================================================
; FILE:     bank4/wizzrobe.asm           BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 7027-7636
; SUBJECTS: Wizzrobe
; TOPIC:    Blue and Red Wizzrobe: walk/teleport/fade, magic shots
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0061 UpdateBlueWizzrobe                                    code  Description:
;   L0067 Wizzrobe_DrawAndCheckCollisionsIntermittently         code
;   L0080 L_Wizzrobe_DrawAndCheckCollisions                     code
;   L0083 BlueWizzrobe_WalkOrTeleport                           code
;   L0116 L11E17_Exit                                           code
;   L0119 GoEveryOtherFrame                                     code
;   L0126 BlueWizzrobe_TurnSometimesAndMoveAndCheckTile         code
;   L0129 BlueWizzrobe_MoveAndCheckTile                         code
;   L0186 BlueWizzrobe_AdvanceCounterAndTurnTowardLinkIfNeeded  code  Description:
;   L0193 L_BlueWizzrobe_TurnTowardLinkIfNeeded                 code  Description:
;   L0198 BlueWizzrobe_TurnTowardLink                           code
;   L0231 BlueWizzrobeTeleportOffsetsX                          data
;   L0235 BlueWizzrobeTeleportOffsetsY                          data
;   L0243 BlueWizzrobe_Move                                     code  Description:
;   L0255 L11EAF_Exit                                           code
;   L0258 BlueWizzrobeTeleportMaxOffsetsX                       data
;   L0261 BlueWizzrobeTeleportMaxOffsetsY                       data
;   L0264 BlueWizzrobeTeleportDirs                              data
;   L0267 BlueWizzrobe_ChooseTeleportTarget                     code
;   L0311 BeginTeleporting                                      code
;   L0326 BlueWizzrobe_AlignWithNearestSquareAndRandomizeTimer  code
;   L0333 BlueWizzrobe_AlignWithNearestSquare                   code
;   L0351 RedWizzrobe_AlignAndSetY                              code  Params:
;   L0358 WizzrobeCollisionOffsetsX                             data
;   L0362 WizzrobeCollisionOffsetsY                             data
;   L0372 Wizzrobe_GetCollidableTile                            code  Params:
;   L0384 Wizzrobe_GetCollidableTileForDir                      code  Params:
;   L0413 Wizzrobe_GetBaseCollidableTile                        code  Params:
;   L0424 BlueWizzrobe_TryShooting                              code
;   L0456 L11F7E_Exit                                           code
;   L0459 BlueWizzrobe_CheckSquareColumn                        code
;   L0482 ShootMagicShot58                                      code
;   L0488 ShootMagicShot                                        code  Params:
;   L0500 UpdateRedWizzrobe                                     code
;   L0520 UpdateRedWizzrobe_JumpTable                           data
;   L0526 RedWizzrobeOffsetsX                                   data
;   L0530 RedWizzrobeOffsetsY                                   data
;   L0534 RedWizzrobeDirections                                 data
;   L0537 UpdateRedWizzrobe_1                                   code
;   L0554 UpdateRedWizzrobe_3                                   code
;   L0604 UpdateRedWizzrobe_0                                   code
;   L0607 UpdateRedWizzrobe_2                                   code
;   L0617 Wizzrobe_DrawAndCheckCollisions                       code
; ============================================================================

; Description:
; Blue Wizzrobe has two states: walking and teleporting.
; When walking, the object timer is set. When teleporting,
; the teleporting distance remaining is set.
;
; If we have the magic clock, then only go draw and check collisions.
;
UpdateBlueWizzrobe:
    LDA InvClock
    BNE L_Wizzrobe_DrawAndCheckCollisions
    JSR BlueWizzrobe_WalkOrTeleport
    JSR BlueWizzrobe_TryShooting

Wizzrobe_DrawAndCheckCollisionsIntermittently:
    ; If the teleporting distance remaining is even, then draw
    ; and check collisions, else return.
    ;
    ; This means that when 0, the monster is walking.
    ; So, draw and check collisions every frame.
    ;
    ; When not 0, the monster is teleporting, and must be drawn
    ; translucently -- every other frame.
    LDA ObjRemDistance, X
    LSR
    BCS L11E17_Exit

L_Wizzrobe_DrawAndCheckCollisions:
    JMP Wizzrobe_DrawAndCheckCollisions

BlueWizzrobe_WalkOrTeleport:
    ; If timer has not expired, then go walk.
    LDA ObjTimer, X
    BNE @Walk

    ; Timer = 0. Only teleporting now.
    ;
    ; If there is still distance to go while translucent, then
    ; decrement the distance remaining, and move.
    LDA ObjRemDistance, X
    BEQ @WalkAgain
    DEC ObjRemDistance, X
    JMP BlueWizzrobe_MoveAndCheckTile

@WalkAgain:
    ; Distance remaining = 0. Still teleporting.
    JSR BlueWizzrobe_AlignWithNearestSquareAndRandomizeTimer
    JMP BlueWizzrobe_TurnTowardLink

@Walk:
    ; Timer:
    ;   >= $10: Turn if needed and move every other frame
    ;   > 1:    Do nothing
    ;   = 1:    Choose a place to teleport to.
    ;
    ; Notice that walking is slower than teleporting; and there is
    ; a pause of $10 frames at the end.
    CMP #$10
    BCS GoEveryOtherFrame
    CMP #$01
    BNE L11E17_Exit
    JSR BlueWizzrobe_ChooseTeleportTarget

L11E17_Exit:
    RTS

GoEveryOtherFrame:
    ; Every other frame, advance the turn counter, turn if needed,
    ; and move.
    LDA FrameCounter
    LSR
    BCS L_BlueWizzrobe_TurnTowardLinkIfNeeded

BlueWizzrobe_TurnSometimesAndMoveAndCheckTile:
    JSR BlueWizzrobe_AdvanceCounterAndTurnTowardLinkIfNeeded

BlueWizzrobe_MoveAndCheckTile:
    JSR BlueWizzrobe_Move
    JSR Wizzrobe_GetCollidableTile

    ; If walkable, then return.
    BCC @Exit

    ; Go handle a wall, if the tile is not a block nor water.
    LDA Wizzrobe_ObjLastTile, X
    AND #$FC
    CMP #$B0
    BEQ @HitBlockOrWater
    CMP #$F4
    BCC @HitWall

@HitBlockOrWater:
    ; If the monster was already teleporting, then return,
    ; so that it moves again next frame.
    LDA ObjRemDistance, X
    BNE L11E17_Exit

    ; Else go start teleporting thru this obstacle.
    JMP BeginTeleporting

@HitWall:
    ; The monster hit a wall.
    ; If facing direction has a vertical component, then flip it.
    LDY ObjDir, X
    TYA
    AND #$0C
    BEQ @CheckHorizontal
    TYA
    EOR #$0C
    STA ObjDir, X
    TAY

@CheckHorizontal:
    ; If facing direction has a horizontal component, then flip it.
    TYA
    AND #$03
    BEQ @Move
    TYA
    EOR #$03
    STA ObjDir, X

@Move:
    JMP BlueWizzrobe_Move

@Exit:
    RTS

; Description:
; Increment the turn counter. Then turn toward Link
; when the turn counter is a multiple of $40.
;
; Increment the turn counter.
;
BlueWizzrobe_AdvanceCounterAndTurnTowardLinkIfNeeded:
    INC Wizzrobe_ObjBlueTurnCounter, X

; Description:
; When the counter is a multiple of $40, switch between facing
; toward Link horizontally or vertically. All other times, return.
;
L_BlueWizzrobe_TurnTowardLinkIfNeeded:
    LDA Wizzrobe_ObjBlueTurnCounter, X
    AND #$3F
    BNE L11EAF_Exit

BlueWizzrobe_TurnTowardLink:
    ; If the multiple of $40 is even, then turn horizontally,
    ; else vertically.
    LDA Wizzrobe_ObjBlueTurnCounter, X
    AND #$40
    BNE @TurnVertically

    ; If the monster's X >= Link's X, then choose left, else right.
    LDA #$02
    LDY ObjX, X
    CPY ObjX
    BCS :+
    LSR
:
    JMP @SetDir

@TurnVertically:
    ; If the monster's Y >= Link's Y, then choose up, else down.
    LDA #$08
    LDY ObjY, X
    CPY ObjY
    BCS @SetDir
    LSR

@SetDir:
    ; If the direction chosen = facing direction, then return.
    CMP ObjDir, X
    BEQ L11EAF_Exit

    ; Else face toward Link.
    STA ObjDir, X
    JMP BlueWizzrobe_AlignWithNearestSquare

BlueWizzrobeTeleportOffsetsX:
    .BYTE $00, $01, $FF, $00, $00, $01, $FF, $00
    .BYTE $00, $01, $FF

BlueWizzrobeTeleportOffsetsY:
    .BYTE $00, $00, $00, $00, $01, $01, $01, $00
    .BYTE $FF, $FF, $FF

; Description:
; This procedure can be called MoveSimple8. But it's only used
; by blue wizzrobe and Ganon, which move alike.
;
BlueWizzrobe_Move:
    ; Index the offsets using the direction value directly.
    LDY ObjDir, X
    LDA ObjX, X
    CLC
    ADC BlueWizzrobeTeleportOffsetsX, Y
    STA ObjX, X
    LDA ObjY, X
    CLC
    ADC BlueWizzrobeTeleportOffsetsY, Y
    STA ObjY, X

L11EAF_Exit:
    RTS

BlueWizzrobeTeleportMaxOffsetsX:
    .BYTE $E0, $20, $E0, $20

BlueWizzrobeTeleportMaxOffsetsY:
    .BYTE $E0, $E0, $20, $20

BlueWizzrobeTeleportDirs:
    .BYTE $0A, $09, $06, $05

BlueWizzrobe_ChooseTeleportTarget:
    ; Choose a random value between 0 and 3 that will be used
    ; to test and set a direction to move in.
    LDA Random, X
    AND #$03
    TAY

    ; The random value implies a random direction.
    ; Save the original X coordinate; and add to it the offset for
    ; the random direction.
    LDA ObjX, X
    PHA
    CLC
    ADC BlueWizzrobeTeleportMaxOffsetsX, Y
    STA ObjX, X

    ; Save the original Y coordinate; and add to it the offset for
    ; the random direction.
    LDA ObjY, X
    PHA
    CLC
    ADC BlueWizzrobeTeleportMaxOffsetsY, Y
    STA ObjY, X
    TYA                         ; Save the random index.
    PHA

    ; Test the walkability in the random direction chosen.
    LDA BlueWizzrobeTeleportDirs, Y
    TAY
    JSR Wizzrobe_GetCollidableTileForDir
    PLA                         ; Restore the random index.
    TAY
    PLA                         ; Restore the original Y.
    STA ObjY, X
    PLA                         ; Restore the original X.
    STA ObjX, X

    ; If not walkable, then all we can do is align with the nearest square.
    BCS BlueWizzrobe_AlignWithNearestSquareAndRandomizeTimer

    ; Set the random diagonal direction.
    LDA BlueWizzrobeTeleportDirs, Y
    STA ObjDir, X

BeginTeleporting:
    ; Will move $20 pixels horizontally and vertically.
    LDA #$20
    STA ObjRemDistance, X

    ; Flip bit 6 of turn counter; so that we change the axis to turn toward.
    LDA Wizzrobe_ObjBlueTurnCounter, X
    EOR #$40
    STA Wizzrobe_ObjBlueTurnCounter, X

    ; Reset the timer to note that the monster is teleporting.
    ; Align with the nearest square.
    LDA #$00
    BEQ :+

BlueWizzrobe_AlignWithNearestSquareAndRandomizeTimer:
    ; Set a random timer >= $70.
    LDA Random, X
    ORA #$70
:
    STA ObjTimer, X

BlueWizzrobe_AlignWithNearestSquare:
    ; If in the left half of a square, jump to the left edge.
    ; Else jump to the next square.
    LDA ObjX, X
    CLC
    ADC #$08
    AND #$F0
    STA ObjX, X

    ; If in the top half of a square, jump to the top edge.
    ; Else jump to the next square.
    LDA ObjY, X
    CLC
    ADC #$08

; Params:
; A: Y coordinate, square ($10) aligned
;
RedWizzrobe_AlignAndSetY:
    AND #$F0
    SEC
    SBC #$03
    STA ObjY, X
    RTS

WizzrobeCollisionOffsetsX:
    .BYTE $0F, $00, $00, $04, $08, $00, $00, $04
    .BYTE $08, $00

WizzrobeCollisionOffsetsY:
    .BYTE $04, $04, $00, $08, $08, $08, $00, $F8
    .BYTE $00, $00

; Params:
; X: object index
;
; Returns:
; C: 0 if walkable
;
Wizzrobe_GetCollidableTile:
    LDY ObjDir, X

; Params:
; X: object index
; Y: 8-way direction
;
; Returns:
; C: 0 if walkable
;
; Subtract 1 from direction to make an 8-way direction index.
;
Wizzrobe_GetCollidableTileForDir:
    DEY

    ; Look up and add the collision offset to the monster's X coordinate.
    LDA ObjX, X
    PHA                         ; Save the original X.
    CLC
    ADC WizzrobeCollisionOffsetsX, Y
    STA ObjX, X

    ; Look up and add the collision offset to the monster's Y coordinate.
    LDA ObjY, X
    PHA                         ; Save the original Y.
    CLC
    ADC WizzrobeCollisionOffsetsY, Y
    STA ObjY, X
    JSR Wizzrobe_GetBaseCollidableTile
    PLA                         ; Restore the original Y.
    STA ObjY, X
    PLA                         ; Restore the original X.
    STA ObjX, X
    RTS

; Params:
; X: object index
;
; Returns:
; C: 0 if walkable
;
Wizzrobe_GetBaseCollidableTile:
    JSR GetCollidableTileStill
    LDA ObjCollidedTile, X

    ; Return C=0 if walkable.
    CMP RoomFirstUnwalkableTile

    ; Store the last collided tile.
    STA Wizzrobe_ObjLastTile, X
    RTS

BlueWizzrobe_TryShooting:
    ; Once every $20 frames, and when not fading, we'll try to shoot.
    ; Otherwise, return.
    LDA ObjRemDistance, X
    BNE L11F7E_Exit
    LDA FrameCounter
    AND #$1F
    BNE L11F7E_Exit

    ; If Link and the monster are not within the same square row,
    ; then go see about the same square column.
    LDA ObjY, X
    AND #$F0
    STA $00
    LDA ObjY
    AND #$F0
    CMP $00
    BNE BlueWizzrobe_CheckSquareColumn

    ; If the monster is to the right of Link, then choose left.
    ; Else choose right.
    LDA #$02
    LDY ObjX, X
    CPY ObjX
    BCS :+
    LSR
:
    ; If the direction chosen = the monster's facing direction,
    ; then go shoot.
    CMP ObjDir, X
    BEQ ShootMagicShot58

L11F7E_Exit:
    RTS

BlueWizzrobe_CheckSquareColumn:
    ; If Link and the monster are not within the same square column,
    ; then return.
    LDA ObjX, X
    AND #$F0
    STA $00
    LDA ObjX
    CMP $00
    BNE L11F7E_Exit

    ; If the monster is down from Link, then choose up.
    ; Else choose down.
    LDA #$08
    LDY ObjY, X
    CPY ObjY
    BCS :+
    LSR
:
    ; If the direction chosen <> the monster's facing direction,
    ; then return.
    CMP ObjDir, X
    BNE L11F7E_Exit

ShootMagicShot58:
    LDA #$58

; Params:
; A: shot object type
;
ShootMagicShot:
    STA $00

    ; If we have the magic clock, then return.
    LDA InvClock
    BNE L11F7E_Exit

    ; Else play the magic sound, shoot, and return.
    LDA #$04
    STA Tune0Request
    JMP ShootLimited

UpdateRedWizzrobe:
    ; If we have the magic clock, then go check collisions and draw.
    LDA InvClock
    BEQ :+
    JMP Wizzrobe_DrawAndCheckCollisions
:
    ; Else increase animation counter and decrease state every frame.
    INC Wizzrobe_ObjRedAnimCounter, X
    DEC ObjState, X

    ; Shift the state right 6 times to get the index (0 to 3) of a state routine.
    LDA ObjState, X
    LSR
    LSR
    LSR
    LSR
    LSR
    LSR
    JSR TableJump

UpdateRedWizzrobe_JumpTable:
    .ADDR UpdateRedWizzrobe_0
    .ADDR UpdateRedWizzrobe_1
    .ADDR UpdateRedWizzrobe_2
    .ADDR UpdateRedWizzrobe_3

RedWizzrobeOffsetsX:
    .BYTE $00, $00, $E0, $20, $00, $00, $C0, $40
    .BYTE $00, $00, $D0, $30, $00, $00, $B0, $50

RedWizzrobeOffsetsY:
    .BYTE $E0, $20, $00, $00, $C0, $40, $00, $00
    .BYTE $D0, $30, $00, $00, $B0, $50, $00, $00

RedWizzrobeDirections:
    .BYTE $04, $08, $01, $02

UpdateRedWizzrobe_1:
    ; Handle state group 1 ($40 to $7F).
    ;
    ; If state = $7F, make it $4F.
    ; $7F is the first frame of state group ($40 to $7F).
    ; By doing this, we skip $30 frames.
    LDA ObjState, X
    CMP #$7F
    BNE :+
    LDA #$4F
    STA ObjState, X
:
    ; Increment the fade counter. Depending on the low bit,
    ; the monster will be drawn every other frame.
    INC Wizzrobe_ObjRedFadeCounter, X
    JMP Wizzrobe_DrawAndCheckCollisionsIntermittently

UpdateRedWizzrobe_3:
    ; Handle state group 3 ($C0 to $FF).
    ;
    ; If state <> $FF, go fade as in state group 1.
    LDY ObjState, X
    INY
    BNE UpdateRedWizzrobe_1

    ; State $FF.
    ;
    ; Face in a random direction.
    LDA Random, X
    PHA                         ; Save the random value.
    AND #$03
    TAY
    LDA RedWizzrobeDirections, Y
    STA ObjDir, X
    PLA                         ; Restore the random value.

    ; Use a random value from 0 to $F to index into an array of offsets.
    AND #$0F
    TAY
    LDA ObjX

    ; Add the X offset and Link's X to set the monster's position.
    ADC RedWizzrobeOffsetsX, Y
    AND #$F0                    ; Align with a square.
    STA ObjX, X

    ; Add the Y offset and Link's Y to set the monster's position.
    LDA ObjY
    CLC
    ADC #$03                    ; Add 3 to help with the alignment calculation below.
    ADC RedWizzrobeOffsetsY, Y
    JSR RedWizzrobe_AlignAndSetY

    ; If Y coordinate >= $5D and < $C4, then the monster
    ; is not in a wall. So, check tile collision.
    CMP #$5D
    BCC :+
    CMP #$C4
    BCS :+
    JSR Wizzrobe_GetCollidableTile

    ; If walkable, then return.
    ; Else increase state to 0, in order to check again next frame.
    BCC UpdateRedWizzrobe_0
:
    INC ObjState, X

UpdateRedWizzrobe_0:
    RTS

UpdateRedWizzrobe_2:
    ; Handle state group 2 ($80 to $BF).
    ;
    ; If state = $B0, then shoot a magic shot $59.
    LDA ObjState, X
    CMP #$B0
    BNE Wizzrobe_DrawAndCheckCollisions
    LDA #$59
    JSR ShootMagicShot

Wizzrobe_DrawAndCheckCollisions:
    ; Invincible to everything but sword and bomb.
    LDA #$F6
    STA ObjInvincibilityMask, X
    JSR GetObjectMiddle

    ; If invincibility timer = 0, then check collisions with weapons.
    LDA ObjInvincibilityTimer, X
    BNE :+
    LDY #$0E
    JSR CheckMonsterSwordShotOrMagicShotCollision
    LDY #$10
    JSR CheckMonsterBombOrFireCollision
    LDY #$11
    JSR CheckMonsterBombOrFireCollision
    LDY #$0D
    JSR CheckMonsterSwordCollision
:
    ; In this state, the monster can always harm Link,
    ; even if it can't be harmed.
    JSR CheckLinkCollision

    ; Every 4 frames switch between two frame images.
    LDA Wizzrobe_ObjRedAnimCounter, X
    LSR
    LSR
    AND #$01
    PHA                         ; Save the frame image number.
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; If facing up, then go add 2 to frame image number, and draw mirrored.
    LDA ObjDir, X
    AND #$08
    BNE :+

    ; If facing left, then flip horizontally.
    LDA ObjDir, X
    LSR
    AND #$01
    STA $0F
    PLA                         ; Restore the frame image number.
    JMP DrawObjectNotMirrored
:
    PLA
    CLC
    ADC #$02                    ; Up facing frame image numbers are 2 and 3.
    JMP DrawObjectMirrored
