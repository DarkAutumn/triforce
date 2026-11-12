; ============================================================================
; FILE:     bank4/monster-shots-fireballs.asm BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 790-1093
; SUBJECTS: Fireball
; TOPIC:    ShootFireball, UpdateMonsterShot, shot bounce/destroy, UpdateFireball
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0028 ShootFireball              code  Params:
;   L0056 ShotBounceWidths           data
;   L0059 ShotBounceHeights          data
;   L0062 UpdateMonsterShot          code
;   L0103 L_DrawShot                 code
;   L0144 DestroyMonsterShot         code
;   L0152 DestroyCountedMonsterShot  code
;   L0157 BounceShot                 code
;   L0198 CheckShotLinkCollision     code  Params:
;   L0215 FireballQSpeedsX           data
;   L0218 FireballQSpeedsY           data
;   L0222 UpdateFireball             code
;   L0319 Fireball_MoveOneAxis       code  Params:
; ============================================================================

; Params:
; A: object type
; X: shooter's object index
;
ShootFireball:
    STA $00                     ; [00] holds the object type.

    ; Return, if there's no empty slot.
    JSR FindEmptyMonsterSlot
    BEQ :+
    TXA                         ; Save the shooter's object index.
    PHA
    TYA                         ; Switch to and save the new object's slot.
    TAX
    LDA $00
    JSR SetTypeAndClearObject
    TXA                         ; Restore the new object's slot.
    TAY
    PLA                         ; Restore the shooter's object index.
    TAX

    ; Set the new object's location to the shooter's.
    ; Offset the new object's X right 4 pixels.
    LDA ObjX, X
    CLC
    ADC #$04
    STA a:ObjX, Y
    LDA ObjY, X
    STA a:ObjY, Y
:
    RTS

ShotBounceWidths:
    .BYTE $01, $FF

ShotBounceHeights:
    .BYTE $FE, $02, $FF, $FF

UpdateMonsterShot:
    ; Set moving direction to facing direction.
    LDA ObjDir, X
    STA $0F

    ; If major state <> 1, go bounce.
    LDA ObjState, X
    AND #$F0
    CMP #$10
    BNE BounceShot

    ; If object type < $55 (like flying rock $53), then
    ; check for tile collision in addition to other checks.
    LDA ObjType, X
    CMP #$55
    BCS @CheckBoundary

    ; Additionally, if the flying rock is held back by a timer, then
    ; only check for collision with Link.
    LDA ObjTimer, X
    BNE @CheckLinkCollision

    ; If the flying rock hit a tile or room boundary, then go destroy it.
    JSR GetCollidingTileMoving
    CMP RoomFirstUnwalkableTile
    BCS DestroyMonsterShot

@CheckBoundary:
    ; No shots can cross the room boundary.
    JSR BoundByRoom
    BEQ DestroyMonsterShot

    ; Move the object, and check for a collision with Link.
    ; Go destroy the shot object, if there was a harmful collision.
    JSR MoveObject

@CheckLinkCollision:
    JSR CheckShotLinkCollision
    LDA $06                     ; Not zero, if collided.
    BNE DestroyMonsterShot

L_DrawShot:
    ; If object type = arrow ($5B), then draw an arrow, and return.
    LDA ObjType, X
    CMP #$5B
    BNE :+
    JMP DrawArrow
:
    ; If object type = sword shot or magic shot, then go draw, and return.
    CMP #$57
    BCC @DrawOthers
    CMP #$5A
    BCS @DrawOthers
    JMP DrawSwordShotOrMagicShot

@DrawOthers:
    ; Prepare to draw other shot types.
    JSR Anim_FetchObjPosForSpriteDescriptor
    STA $0D                     ; [0D] holds frame image 0

    ; Assume object type >= $55. Prepare sprite attributes that
    ; make the shot flash.
    LDA FrameCounter
    AND #$03

    ; If object type < $55 (flying rock is $53), then
    ; add 4 to X coordinate, and use sprite attributes 0.
    LDY ObjType, X
    CPY #$55
    BCS @Draw
    LDY #$03
:
    INC $00
    DEY
    BPL :-
    LDA #$00

@Draw:
    JSR Anim_SetSpriteDescriptorAttributes
    LDA $0D                     ; [0D] frame image
    JMP DrawObjectNotMirrored

DestroyMonsterShot:
    ; Destroy fireballs directly. These are not kept track of.
    LDA ObjType, X
    CMP #$55
    BEQ :+
    CMP #$56
    BEQ :+

DestroyCountedMonsterShot:
    DEC MonsterShotCount
:
    JMP DestroyMonster

BounceShot:
    ; Get the reverse direction index for the bounce direction.
    LDA MonsterShot_ObjBounceDir, X
    JSR GetOppositeDir

    ; Modify the coordinates according to the displacements for
    ; the bounce direction.
    LDA ObjY, X
    CLC
    ADC ShotBounceHeights, Y
    STA ObjY, X
    LDA ObjX, X
    CLC
    ADC ShotBounceWidths, Y
    STA ObjX, X

    ; Add 2 to the bounce distance.
    ; The absolute value of one of the displacements is 2.
    LDA MonsterShot_ObjBounceDist, X
    CLC
    ADC #$02
    STA MonsterShot_ObjBounceDist, X

    ; Go destroy the shot, if the bounce counter has reached the limit.
    ; Otherwise, go draw.
    CMP #$20
    BCS DestroyMonsterShot
    BCC L_DrawShot

; Params:
; X: monster object index
;
; Returns:
; [00]: 0 for Link slot
; [06]: 1 if objects collide
; [09]: 0 for Link damage type (none)
; [0C]: 1 if objects collide
; [034B]: IsShotCollidingWithLink
;
; Reset bounce distance.
;
CheckShotLinkCollision:
    LDA #$00
    STA MonsterShot_ObjBounceDist, X
    JSR CheckLinkCollision
    LDA IsShotCollidingWithLink
    BEQ :+

    ; If the shot hits Link, then start to bounce off of Link's shield.
    ; Set the bounce direction to Link's facing direction.
    ; Set state $30 to bounce.
    LDA ObjDir
    STA MonsterShot_ObjBounceDir, X
    LDA #$30
    STA ObjState, X
:
    RTS

FireballQSpeedsX:
    .BYTE $70, $68, $60, $58, $50, $3C, $26, $10

FireballQSpeedsY:
    .BYTE $00, $10, $26, $3C, $50, $58, $60, $68
    .BYTE $70

UpdateFireball:
    LDA ObjState, X
    BNE @State1

    ; State 0.
    ;
    ; Reset horizontal and vertical position fractions.
    STA Fireball_ObjPosFracX, X
    STA Fireball_ObjPosFracY, X

    ; Link is the target. So, register A has 0 -- his object slot.
    JSR GetDirectionsAndDistancesToTarget

    ; Remember the horizontal and vertical directions toward target.
    LDA $0B
    STA Fireball_ObjDirToTargetX, X
    LDA $0A
    STA Fireball_ObjDirToTargetY, X

    ; Set the facing direction to the combination of the two.
    ORA $0B
    STA ObjDir, X
    LDY #$04
    JSR _CalcDiagonalSpeedIndex

    ; Look up and set the horizontal and vertical q-speeds.
    LDA FireballQSpeedsX, Y
    STA Fireball_ObjQSpeedX, X
    LDA FireballQSpeedsY, Y
    STA Fireball_ObjQSpeedY, X

    ; Set state to $10: monster shot active.
    ; Set object timer to $10, so it's seen but delays a little before moving.
    LDA #$10
    STA ObjState, X
    STA ObjTimer, X
    RTS

@State1:
    ; State 1.
    ;
    ; Fireballs are delayed and visible for a few frames before moving.
    ; If timer <> 0, go draw.
    LDA ObjTimer, X
    BNE @CheckCollisionAndDraw

    ; If the room boundary blocks the fireball, then destroy it.
    LDA ObjDir, X
    JSR BoundByRoomWithA
    BNE @Move

@DestroyMonster:
    JMP DestroyMonster

@Move:
    ; A movement routine is used that works along one axis.
    ; So, load its parameters, and call it for each axis individually.
    ;
    ; First, load the horizontal direction into [0F].
    LDA Fireball_ObjDirToTargetX, X
    STA $0F

    ; Load the horizontal q-speed and position fraction,
    ; move, then save the position fraction.
    LDA Fireball_ObjQSpeedX, X
    LDY Fireball_ObjPosFracX, X
    JSR Fireball_MoveOneAxis
    STA Fireball_ObjPosFracX, X

    ; Second, load the vertical direction into [0F].
    LDA Fireball_ObjDirToTargetY, X
    STA $0F

    ; Load the vertical q-speed and position fraction,
    ; move, then save the position fraction.
    LDA Fireball_ObjQSpeedY, X
    LDY Fireball_ObjPosFracY, X
    JSR Fireball_MoveOneAxis
    STA Fireball_ObjPosFracY, X

@CheckCollisionAndDraw:
    ; Check for collision with Link. If hit, then go destroy the fireball.
    JSR CheckShotLinkCollision
    LDA IsShotCollidingWithLink
    BNE @DestroyMonster
    JMP L_DrawShot

; Params:
; A: q-speed
; Y: position fraction
; [0F]: movement direction
;
; Returns:
; A: position fraction
;
; Load the q-speed and position fraction from the parameters.
;
Fireball_MoveOneAxis:
    STA ObjQSpeedFrac, X
    TYA
    STA ObjPosFrac, X
    JSR MoveObject

    ; Return the updated position fraction.
    LDA ObjPosFrac, X
    RTS
