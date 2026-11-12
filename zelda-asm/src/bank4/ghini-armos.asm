; ============================================================================
; FILE:     bank4/ghini-armos.asm        BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 3070-3372
; SUBJECTS: Armos, Ghini, Obj
; TOPIC:    Ghini ($21) update and the Ghini sprite; the fade-in "init" shared by Armos ($1E) and flying
;           Ghini ($22), with the secrets hidden under seven Armos statues; Armos update and sprite.
; SEE ALSO: bank1/armos-ghini-spawn.asm (Spawn_CheckTouchedArmosOrGravestone: touching a statue or gravestone creates
;           the object with timer $3F); bank4/flying-ghini-peahat.asm (Ghini_UpdateFlying);
;           bank4/goriya.asm (Goriya_Update: the Armos walks with it); bank4/walker-wanderer-common.asm
;           (Wanderer_Update: Ghini movement); bank4/zora-ow-inits.asm (Octorok_InitSlowOrGhini,
;           Flyer_FinishInit); bank7/tile-obj-tiles.asm (Tile_ChangeTileObjTiles); bank1/monster-collisions.asm
;           (Damage_ObjTypeDamageToLink).
; KEY FACTS:
;   - When a Ghini ($21) dies, every flying Ghini ($22) in slots $01-$0B is set dying (metastate $11)
;     in the same frame (Ghini_Update).
;   - A flying Ghini is never tested against Link's weapons: its draw routine only calls
;     Link_CheckMonsterTouch. It dies only when a Ghini dies (Ghini_DrawAndCheckLinkCollision, Ghini_Update).
;   - Ghini: 9 HP, q-speed $20 (0.5 px/frame), turn rate $FF, drop group 1 (Ghini_Update,
;     Octorok_InitSlowOrGhini, Obj_TypeHpPairs, Drop_Group1Types).
;   - Armos and flying Ghini fade in for $3F frames: the init routine reruns every frame while ObjTimer
;     is nonzero, drawing on even timer values only (Armos_InitOrFlyingGhiniFadeIn).
;   - A fading-in Armos or flying Ghini already hurts Link on the frames it is drawn, and a fading-in
;     Armos cannot be hit by weapons (Armos_DrawAndCheckCollisions, Ghini_DrawAndCheckLinkCollision).
;   - Armos: 3 HP; its q-speed is picked once when it wakes: $20 (0.5 px/frame) if Random+1,X < $80,
;     else $60 (1.5 px/frame) (Armos_InitOrFlyingGhiniFadeIn).
;   - Secrets under an Armos at Y = $80: stairs in rooms $0B (X $B0), $1C ($B0), $22 ($30), $34 ($40),
;     $3D ($90), $4E ($A0); the power bracelet (item $14) in room $24 (X $E0)
;     (Armos_SecretRoomIds, Armos_SecretXs).
;   - Armos stairs are not saved in the room flags and have no quest test: they appear each time that
;     Armos is woken (Armos_InitOrFlyingGhiniFadeIn).
;   - Touching a Ghini, flying Ghini or Armos costs $0100 damage (one heart before rings)
;     (Damage_ObjTypeDamageToLink).
;   - A dead Armos becomes object type $5D (dead dummy), which never drops an item and does not advance
;     WorldKillCycle or RoomKillCount (Armos_DrawAndCheckCollisions, Obj_EndMetastate,
;     Drop_NoDropTypes).
; CONTENTS: (generated; L = line in this file)
;   L0056 Ghini_Update                              code  Updates a Ghini ($21): wanders toward Link, draws, checks hits; its death kills all flying Ghinis.
;   L0091 Ghini_DrawAndCheckLinkCollision           code  Draws a Ghini or flying Ghini facing its direction, then checks its collision with Link only.
;   L0132 Obj_DrawNotMirroredAndCheckLinkCollision  code  Draws the object not mirrored, then checks its collision with Link only (no weapon tests).
;   L0138 Armos_SecretRoomIds                       data  TABLE: overworld rooms with a secret under an Armos; index = secret 0-6; value = RoomId.
;   L0143 Armos_SecretXs                            data  TABLE: X coordinate of the Armos that hides the secret; index = secret 0-6 (as Armos_SecretRoomIds);
;   L0154 Armos_InitOrFlyingGhiniFadeIn             code  Fades in an Armos ($1E) or flying Ghini ($22) over ObjTimer frames, then finishes waking it.
;   L0286 Ghini_FinishFlyingFadeIn                  code  Flying Ghini tail of the fade-in above: draws it and, once the timer is 0, starts it as a flyer.
;   L0304 Armos_Update                              code  Updates a woken Armos ($1E): walks like a Goriya (never shoots), animates, draws, checks hits.
;   L0332 Armos_DrawAndCheckCollisions              code  Draws an Armos and checks collisions: with Link only while it fades in, with weapons too afterwards.
; ============================================================================

; ----------------------------------------------------------------------------
; Updates a Ghini ($21): wanders toward Link, draws, checks hits; its death kills all flying Ghinis.
; IN:   X = object slot.
; NOTE: 9 HP. Turn rate $FF: at every grid point it tests whether the chase target is in line (less
;       than 9 pixels off on one axis) and turns toward it. Does not move with the clock or while
;       stunned (Wanderer_Update).
; NOTE: Flying Ghinis ($22) cannot be hit by weapons; they die only here, when a Ghini dies.
; ----------------------------------------------------------------------------
Ghini_Update:
    LDA #$FF                    ; Turn rate $FF: never below Random+1,X, so always test for Link in line.
    JSR Wanderer_Update
    JSR Ghini_DrawAndCheckLinkCollision
    JSR Collision_CheckMonsterVsWeaponsAndLink

    ; Ghini_DrawAndCheckLinkCollision already tested Link; this adds the weapon tests (and tests Link again).
    ; If the Ghini is not dying (metastate 0), then return.
    LDA ObjMetastate, X
    BEQ @Exit

    ; The Ghini is dying. For each object slot from $0B down to $01:
    ; if the object is a flying Ghini ($22), set its metastate to $11 (dying).
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

; ----------------------------------------------------------------------------
; Draws a Ghini or flying Ghini facing its direction, then checks its collision with Link only.
; IN:   X = object slot.
; OUT:  as Link_CheckMonsterTouch.
; NOTE: Frame image 0 (back) when the direction has an up bit, else 1 (front). The image is flipped
;       horizontally when moving right in the front view, or left in the back view.
; ----------------------------------------------------------------------------
Ghini_DrawAndCheckLinkCollision:
    JSR Anim_FetchObjPosForSpriteDescriptor
    STA $0D                     ; A = 0. [0D] = frame image number; 0 is the back view (moving up).

    ; If the direction has an "up" component, then keep the
    ; frame image 0, and go figure out horizontal flipping.
    LDA ObjDir, X
    AND #$08
    BNE @CheckHorizontalFlip

    ; Else make the frame image 1, the front view. It is used for down and for
    ; purely horizontal directions.
    INC $0D

    ; Front view: if right ($01) is a component of the direction, go set
    ; horizontal flipping. Else leave it 0.
    LDA ObjDir, X
    AND #$01
    BEQ @DrawAndCheckCollisions
    BNE @SetHorizontalFlip

@CheckHorizontalFlip:
    ; Back view: set horizontal flipping if left ($02) is a direction component.
    ; (The opposite bit from the front view.)
    LDA ObjDir, X
    AND #$02
    BEQ @DrawAndCheckCollisions

@SetHorizontalFlip:
    INC $0F

@DrawAndCheckCollisions:
    LDA $0D

; ----------------------------------------------------------------------------
; Draws the object not mirrored, then checks its collision with Link only (no weapon tests).
; IN:   A = frame image; X = object slot; [00] = object X; [01] = object Y; [0F] = 1 to flip horizontally.
; OUT:  [06] and [0C] = 1 if the object touches Link; [00] = 0 (Link's slot); [09] = 0;
;       IsShotCollidingWithLink set by Link_CheckMonsterTouch.
; NOTE: Also the draw tail of the Boulder.
; ----------------------------------------------------------------------------
Obj_DrawNotMirroredAndCheckLinkCollision:
    JSR Anim_DrawObjectNotMirrored
    JMP Link_CheckMonsterTouch

; TABLE: overworld rooms with a secret under an Armos; index = secret 0-6; value = RoomId.
;        Index 0 (room $24) holds the power bracelet; indexes 1-6 hold stairs.
Armos_SecretRoomIds:
    .BYTE $24, $0B, $1C, $22, $34, $3D, $4E

; TABLE: X coordinate of the Armos that hides the secret; index = secret 0-6 (as Armos_SecretRoomIds);
;        value = ObjX in pixels. The Armos's Y must be $80 in every case.
Armos_SecretXs:
    .BYTE $E0, $B0, $B0, $30, $40, $90, $A0

; ----------------------------------------------------------------------------
; Fades in an Armos ($1E) or flying Ghini ($22) over ObjTimer frames, then finishes waking it.
; IN:   X = object slot; ObjTimer,X = frames of fade-in left (set to $3F when the statue was touched).
; NOTE: Reruns every frame while the timer is nonzero (it puts the object back to "uninitialized").
;       On the frame the timer is 0 an Armos reveals its secret (stairs or the power bracelet),
;       turns its statue square into floor tile $26 or stairs $70, and picks q-speed $20 or $60 at random.
; NOTE: The stairs under an Armos are not recorded in the room flags and are not quest-dependent.
; ----------------------------------------------------------------------------
Armos_InitOrFlyingGhiniFadeIn:
    ; Obj_Update flagged this object "initialized" before calling here.
    ; While it is still fading in (timer > 0), flag it "uninitialized" again,
    ; so that this routine runs again next frame.
    LDA ObjTimer, X
    STA ObjUninitialized, X

    ; If it's still fading in, then only go face down and draw.
    BNE @FinishInit

    ; The timer has reached 0. A flying Ghini has nothing more to set up here.
    LDA ObjType, X
    CMP #$22                    ; Flying Ghini
    BEQ @FinishInit

    ; Else this is an Armos waking up.
    ;
    ; [00] = the tile that will replace the statue; start with stairs tile $70.
    LDA #$70
    STA $00

    ; Look for the current room in the list of rooms that have a secret under an Armos.
    LDY #$06

@LoopArmosRoom:
    LDA RoomId

    ; If this element does not match the current room ID, go loop again.
    CMP Armos_SecretRoomIds, Y
    BNE @NextLoopArmosRoom

    ; The secret is under one particular Armos of the room.
    ;
    ; If this Armos's X coordinate is not the one listed for the room, go loop again.
    LDA ObjX, X
    CMP Armos_SecretXs, Y
    BNE @NextLoopArmosRoom

    ; If this Armos's Y coordinate <> $80, go loop again.
    LDA ObjY, X
    CMP #$80
    BNE @NextLoopArmosRoom

    ; This Armos hides the secret. If the index <> 0, go put stairs in its place.
    CPY #$00
    BNE @UseChosenTile

    ; Index 0 is room $24. Instead of stairs, it has the power bracelet.
    ;
    ; Put the room item (object slot $13) at the Armos's coordinates. A = ObjY,X = $80 here.
    STA ObjY+19
    LDA ObjX, X
    STA ObjX+19

    ; Y = 0: reset the room item's state, which activates it.
    STY ObjState+19

    ; Set the room item id: $14, the power bracelet.
    LDA #$14
    STA RoomItemId

    ; If the room item has not been taken yet (room flag $10 clear), play the "secret found" tune.
    ; Either way, go replace the statue with a floor tile: both BNEs below are always taken
    ; when reached (Sfx_PlaySecretFoundTune returns A = 4).
    ;
    ; If the item was already taken, the room item object stays activated, but the code that
    ; draws and takes the room item tests the same room flag and ignores it.
    JSR RoomFlags_GetItemTaken
    BNE @UseFloorTile
    JSR Sfx_PlaySecretFoundTune
    BNE @UseFloorTile

@NextLoopArmosRoom:
    ; Try the next room in the list, down to index 0.
    DEY
    BPL @LoopArmosRoom

@UseFloorTile:
    ; No secret under this Armos, or it was the power bracelet.
    ; Replace the statue with floor tile $26.
    LDA #$26
    STA $00

@UseChosenTile:
    LDA $00

    ; If the replacement tile is the stairs tile, then play the "secret found" tune.
    PHA
    CMP #$70
    BNE :+
    JSR Sfx_PlaySecretFoundTune
:
    ; Replace the 2x2 tiles of the square where the statue stood (returning to bank 4 afterwards).
    INC ReturnToBank4
    PLA
    JSR Tile_ChangeTileObjTiles

    ; The Armos was made from a square, so its Y is a multiple of $10, while walking objects
    ; reach a grid point when the low nibble of Y is $D. Set its grid offset to 3 to compensate.
    LDA #$03
    STA ObjGridOffset, X

    ; Pick the walking q-speed at random, once:
    ; $20 (0.5 px/frame) if Random+1,X < $80, else $60 (1.5 px/frame).
    LDA #$20
    LDY Random+1, X
    CPY #$80
    BCC :+
    LDA #$60
:
    STA ObjQSpeedFrac, X

@FinishInit:
    ; Face down, and make the input direction down.
    LDA #$04
    STA ObjInputDir, X
    STA ObjDir, X

    ; When the timer is odd, return without drawing (the object flickers while fading in).
    LDA ObjTimer, X
    LSR
    BCS :+

    ; Draw (and check collision with Link) according to type.
    LDA ObjType, X
    CMP #$22                    ; Flying Ghini
    BEQ Ghini_FinishFlyingFadeIn
    JSR Armos_DrawAndCheckCollisions
:
    RTS

; Flying Ghini tail of the fade-in above: draws it and, once the timer is 0, starts it as a flyer.
Ghini_FinishFlyingFadeIn:
    JSR Ghini_DrawAndCheckLinkCollision

    ; While still fading in ("uninitialized" flag = timer <> 0), return.
    ; Else reset metastate and timer, and start the flyer state machine
    ; (flying state 0, maximum speed $A0, speed $1F).
    LDA ObjUninitialized, X
    BNE :-
    JSR Obj_ResetMetastateAndTimer
    JMP Flyer_FinishInit

; ----------------------------------------------------------------------------
; Updates a woken Armos ($1E): walks like a Goriya (never shoots), animates, draws, checks hits.
; IN:   X = object slot.
; NOTE: 3 HP. Speed was fixed at wake-up: q-speed $20 or $60 (0.5 or 1.5 px/frame).
; NOTE: At a grid point it faces the chase target along the longer axis if that distance is < $51
;       pixels; otherwise it keeps its direction (Goriya_Update).
; ----------------------------------------------------------------------------
Armos_Update:
    JSR Goriya_Update

    ; If it is being shoved, then don't animate. Only go draw.
    LDA ObjShoveDir, X
    BNE Armos_DrawAndCheckCollisions

    ; Decrement the animation counter. If it hasn't reached 0, then only go draw.
    DEC ObjAnimCounter, X
    BNE Armos_DrawAndCheckCollisions

    ; Each Armos animation frame lasts 6 frames.
    LDA #$06
    STA ObjAnimCounter, X

    ; Frame images 0 and 1 are the first animation frame, front and back;
    ; 2 and 3 are the second. Flip between the two sets by XOR'ing with 2.
    ; The front or back image is chosen when drawing, below.
    LDA ObjAnimFrame, X
    EOR #$02
    STA ObjAnimFrame, X

; ----------------------------------------------------------------------------
; Draws an Armos and checks collisions: with Link only while it fades in, with weapons too afterwards.
; IN:   X = object slot.
; NOTE: An Armos that dies here has its type changed to $5D (dead dummy): no item drop, and the kill
;       does not advance WorldKillCycle or RoomKillCount (Obj_EndMetastate, Drop_NoDropTypes).
; ----------------------------------------------------------------------------
Armos_DrawAndCheckCollisions:
    JSR Anim_FetchObjPosForSpriteDescriptor
    LDY ObjDir, X

    ; A = 0 here. If facing up, use the second frame image of the set (back).
    ; Else use the first (front).
    CPY #$08
    BNE :+
    LDA #$01
:
    CLC
    ADC ObjAnimFrame, X
    JSR Anim_DrawObjectNotMirrored

    ; If the Armos is still fading in (timer <> 0), then go check for a collision
    ; with Link only: weapons cannot touch it yet.
    LDA ObjTimer, X
    BNE @CheckLinkCollision

    ; Else check for collisions with Link's weapons and with Link.
    JSR Collision_CheckMonsterVsWeaponsAndLink

    ; If the Armos is now dying (metastate <> 0), then change its object type to $5D (dead dummy).
    LDA ObjMetastate, X
    BEQ :+
    LDA #$5D
    STA ObjType, X
:
    RTS

@CheckLinkCollision:
    JMP Link_CheckMonsterTouch
