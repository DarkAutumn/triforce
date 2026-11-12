; ============================================================================
; FILE:     bank4/ghini-armos.asm        BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 3070-3372
; SUBJECTS: (none)
; TOPIC:    Ghini; Armos tables, init, update
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0020 UpdateGhini                                 code
;   L0048 DrawGhiniAndCheckCollisions                 code
;   L0098 DrawObjectNotMirroredAndCheckLinkCollision  code  Params:
;   L0102 SecretArmosRoomIds                          data
;   L0105 SecretArmosXs                               data
;   L0108 InitArmosOrFlyingGhini                      code
;   L0250 L_EndInitFlyingGhini                        code
;   L0260 UpdateArmos                                 code
;   L0291 DrawArmosAndCheckCollisions                 code
; ============================================================================

UpdateGhini:
    LDA #$FF                    ; Pass in turn rate $FF to turn as often as possible.
    JSR UpdateCommonWanderer
    JSR DrawGhiniAndCheckCollisions
    JSR CheckMonsterCollisions

    ; If the monster is still alive, then return.
    LDA ObjMetastate, X
    BEQ @Exit

    ; For each object slot from $B to 1:
    ; If the object is a flying ghini ($22),
    ; then kill it by setting metastate to $11.
    LDY #$0B

@LoopObject:
    LDA ObjType, Y
    CMP #$22
    BNE :+
    LDA #$11
    STA ObjMetastate, Y
:
    DEY
    BNE @LoopObject

@Exit:
    RTS

DrawGhiniAndCheckCollisions:
    JSR Anim_FetchObjPosForSpriteDescriptor
    STA $0D                     ; Reset [0D]: frame image number. "Back" frame for up.

    ; If the direction has an "up" component, then keep the
    ; frame image 0, and go figure out horizontal flipping.
    LDA ObjDir, X
    AND #$08
    BNE @CheckHorizontalFlip

    ; Else make the frame image 1 for the "down" frame.
    ; Note that this is the frame image to use, even if there's no
    ; down component in the direction.
    INC $0D

    ; If right is a component of the direction, then go set horizontal
    ; flipping. Else leave horizontal flipping = 0.
    LDA ObjDir, X
    AND #$01
    BEQ @DrawAndCheckCollisions
    BNE @SetHorizontalFlip

@CheckHorizontalFlip:
    ; Set horizontal flipping if right is a direction component.
    ; This essentially the same test as above, but for when we jump
    ; from the "up" test.
    LDA ObjDir, X
    AND #$02
    BEQ @DrawAndCheckCollisions

@SetHorizontalFlip:
    INC $0F

@DrawAndCheckCollisions:
    LDA $0D

; Params:
; A: frame image
; X: object index
; [00]: object X
; [01]: object Y
; [0F]: flip horizontally
;
; Returns:
; [00]: 0 for Link slot
; [06]: 1 if objects collide
; [09]: 0 for Link damage type (none)
; [0C]: 1 if objects collide
; [034B]: IsShotCollidingWithLink
;
DrawObjectNotMirroredAndCheckLinkCollision:
    JSR DrawObjectNotMirrored
    JMP CheckLinkCollision

SecretArmosRoomIds:
    .BYTE $24, $0B, $1C, $22, $34, $3D, $4E

SecretArmosXs:
    .BYTE $E0, $B0, $B0, $30, $40, $90, $A0

InitArmosOrFlyingGhini:
    ; In UpdateObject, this object was flagged "initialized".
    ; But if it's still fading in (timer > 0), then flag it "uninitialized".
    LDA ObjTimer, X
    STA ObjUninitialized, X

    ; If it's still fading in, then go finish some initialization needed
    ; for fading in.
    BNE @FinishInit

    ; Once the object timer reaches 0; if the object is flying ghini,
    ; go finish initializing it.
    LDA ObjType, X
    CMP #$22                    ; Flying Ghini
    BEQ @FinishInit

    ; Else we have an armos.
    ;
    ; Store stairs tile $70 in [00] for use later.
    LDA #$70
    STA $00

    ; Look for the current room ID in an array of room ID's that
    ; have armos secrets.
    LDY #$06

@LoopArmosRoom:
    LDA RoomId

    ; If this element does not match the current room ID, go loop again.
    CMP SecretArmosRoomIds, Y
    BNE @NextLoopArmosRoom

    ; The secret (usually stairs) is under one of the armoses.
    ;
    ; If the armos's X coordinate does not match the one needed
    ; for the current room in the list, then go loop again.
    LDA ObjX, X
    CMP SecretArmosXs, Y
    BNE @NextLoopArmosRoom

    ; If the armos's Y coordinate <> $80, go loop again.
    LDA ObjY, X
    CMP #$80
    BNE @NextLoopArmosRoom

    ; If the loop index <> 0, go use the stairs tile to patch the floor.
    CPY #$00
    BNE @UseChosenTile

    ; The room ID at index 0 is $24. Instead of stairs, this room
    ; has the power bracelet.
    ;
    ; Set up the room item with the armos's coordinates.
    STA ObjY+19
    LDA ObjX, X
    STA ObjX+19

    ; This resets the room item's state, which activates it.
    STY ObjState+19

    ; Set the room item ID.
    LDA #$14
    STA RoomItemId

    ; If the bracelet has not been taken, then play the "secret found" tune.
    ;
    ; Either way, quit the loop, and replace the tile with a floor tile.
    ;
    ; Note that, if the item was already taken, then the room item
    ; object will stay activated. But in E6EA (MoveAndDrawRoomItem),
    ; it will be ignored, because that routine checks whether
    ; the item was taken.
    JSR GetRoomFlagUWItemState
    BNE @UseFloorTile
    JSR PlaySecretFoundTune
    BNE @UseFloorTile

@NextLoopArmosRoom:
    ; Bottom of the loop.
    DEY
    BPL @LoopArmosRoom

@UseFloorTile:
    ; Not found, or found the power bracelet.
    ; Use a floor tile $26 to replace the armos.
    LDA #$26
    STA $00

@UseChosenTile:
    LDA $00

    ; If the replacement tile is a stairs tile, then play the "secret found" tune.
    PHA
    CMP #$70
    BNE :+
    JSR PlaySecretFoundTune
:
    ; Replace the tile where the armos was.
    INC ReturnToBank4
    PLA
    JSR ChangeTileObjTiles

    ; Because the armos object is aligned to the square grid
    ; (because it was made from a square),
    ; set its grid offset to 3 to compensate.
    ;
    ; Most objects are aligned to the grid vertically when the
    ; low nibble is $D.
    LDA #$03
    STA ObjGridOffset, X

    ; Set a speed (q-speed fraction) randomly:
    ; $20 if random value < $80
    ; $60 otherwise
    LDA #$20
    LDY Random+1, X
    CPY #$80
    BCC :+
    LDA #$60
:
    STA ObjQSpeedFrac, X

@FinishInit:
    ; Facing and input directions down.
    LDA #$04
    STA ObjInputDir, X
    STA ObjDir, X

    ; Every other frame, return without drawing.
    LDA ObjTimer, X
    LSR
    BCS :+

    ; Finish initializing based on type.
    LDA ObjType, X
    CMP #$22                    ; Flying Ghini
    BEQ L_EndInitFlyingGhini
    JSR DrawArmosAndCheckCollisions
:
    RTS

L_EndInitFlyingGhini:
    JSR DrawGhiniAndCheckCollisions

    ; If we're finally leaving the object flagged "initialized",
    ; then reset metastate and initialize this as a flyer.
    LDA ObjUninitialized, X
    BNE :-
    JSR ResetObjMetastateAndTimer
    JMP EndInitFlyer

UpdateArmos:
    JSR UpdateGoriya

    ; If it's still shoved, then don't animate. Only go draw.
    LDA ObjShoveDir, X
    BNE DrawArmosAndCheckCollisions

    ; Animate.
    ;
    ; Decrement the animation counter. But if it hasn't reached 0,
    ; then only go draw.
    DEC ObjAnimCounter, X
    BNE DrawArmosAndCheckCollisions

    ; Armos animation frames last 6 screen frames.
    LDA #$06
    STA ObjAnimCounter, X

    ; Armos doesn't simply flip horizontally to animate. That would
    ; change the position of the shield. To keep fixed it in place,
    ; there are separate frame images for each animation frame.
    ;
    ; Frame images 0 and 1 depict the first animation frame, facing front and back.
    ; Frame images 2 and 3 depict the second animation frame, facing front and back.
    ;
    ; Here flip between the two sets by XOR'ing with 2.
    ; Then choose the direction at the point of drawing below.
    LDA ObjAnimFrame, X
    EOR #$02
    STA ObjAnimFrame, X

DrawArmosAndCheckCollisions:
    JSR Anim_FetchObjPosForSpriteDescriptor
    LDY ObjDir, X

    ; If facing up, then use the second frame image in the set ("back").
    ; Else use the second.
    CPY #$08
    BNE :+
    LDA #$01
:
    CLC
    ADC ObjAnimFrame, X
    JSR DrawObjectNotMirrored

    ; If the armos is still fading in, then go for check collision with Link,
    ; but not with weapons.
    LDA ObjTimer, X
    BNE @CheckLinkCollision

    ; Else check for all collisions.
    JSR CheckMonsterCollisions

    ; If the armos has died, then set object type to $5D (dead dummy).
    LDA ObjMetastate, X
    BEQ :+
    LDA #$5D
    STA ObjType, X
:
    RTS

@CheckLinkCollision:
    JMP CheckLinkCollision
