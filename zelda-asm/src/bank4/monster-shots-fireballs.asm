; ============================================================================
; FILE:     bank4/monster-shots-fireballs.asm BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 790-1093
; SUBJECTS: Fireball, MonsterShot
; TOPIC:    Monster projectiles: creating a fireball (types $55/$56), the update of straight-line shots
;           (flying rock $53, $54, sword shot $57, magic shots $58/$59, $5A), the bounce off Link's
;           shield, destroying shots, and the aimed fireball's flight.
; SEE ALSO: bank1/link-collision-harm.asm (Link_CheckMonsterTouch: which shots the shields block, damage);
;           bank4/shoot-helpers-candle.asm (Shooter_ShootLimited/Shooter_Shoot: counted shots, limit of 4);
;           bank4/walker-wanderer-common.asm (MonsterShot_Init: shot q-speed);
;           bank4/moblin-lynel-shooting.asm (monster arrow $5B reuses the bounce and destroy code);
;           bank1/shot-aim-boomerang-speed.asm (aiming helpers used by Fireball_Update).
; KEY FACTS:
;   - Fireballs ($55, $56) are not counted in MonsterShotCount and have no limit except a free object
;     slot in $01-$0B (Fireball_Shoot, MonsterShot_Destroy). Other shots are counted.
;   - A fireball appears 4 pixels right of its shooter and aims once at Link's position. For the next
;     $0F frames it stays in place (it can already hit Link); from the $10th it flies in that fixed
;     direction (Fireball_Update).
;   - Fireball speed is one of 9 (X, Y) q-speed pairs from $70/$00 (1.75 px/frame along one axis) through
;     $50/$50 (1.25 px/frame on both) (Fireball_QSpeedsX, Fireball_QSpeedsY).
;   - A fireball is destroyed at the room boundary or on any touch of Link, blocked or not; it never
;     tests tiles and never bounces (Fireball_Update).
;   - Straight shots are destroyed at the room boundary or when they harm Link; only types $53 and $54
;     are also destroyed by unwalkable tiles (MonsterShot_Update).
;   - A shot that Link's shield blocks goes to state $30 and drifts 2 pixels per frame the way Link
;     faces for $10 frames, harmless, then is destroyed (MonsterShot_CheckLinkCollision, MonsterShot_Bounce).
;   - A shot can hurt Link only while its state is $1x (Link_CheckMonsterTouch tests this; MonsterShot_Bounce never
;     calls it).
;   - While Link's invincibility timer, the clock or Link's stun timer is nonzero, Link_CheckMonsterTouch
;     reports no touch: shots and fireballs fly through Link (MonsterShot_CheckLinkCollision).
;   - A fireball's ObjDir always holds one horizontal and one vertical bit, even when it flies along
;     an axis; the shield test accepts either one opposing Link's facing (Fireball_Update).
; CONTENTS: (generated; L = line in this file)
;   L0058 Fireball_Shoot                  code  Creates a fireball object of the given type 4 pixels right of the shooter.
;   L0088 MonsterShot_BounceStepsX        data  TABLE: X step per frame of a bouncing shot; index = reverse direction index of Link's facing
;   L0092 MonsterShot_BounceStepsY        data  TABLE: Y step per frame of a bouncing shot; same index; value = signed pixels.
;   L0102 MonsterShot_Update              code  Updates a straight-flying monster shot: types $53, $54, $57, $58, $59 and $5A.
;   L0144 MonsterShot_Draw                code  Draws a monster shot according to its type; also the draw step of fireballs and bouncing shots.
;   L0189 MonsterShot_Destroy             code  Destroys a monster shot; MonsterShotCount is decremented unless it is a fireball ($55, $56).
;   L0198 MonsterShot_DestroyCounted      code  Destroys a counted shot (also monster arrows and boomerangs): decrements MonsterShotCount first.
;   L0209 MonsterShot_Bounce              code  Moves a shot that Link's shield blocked: 2 pixels per frame the way Link faced, for $10 frames.
;   L0249 MonsterShot_CheckLinkCollision  code  Tests a shot against Link and starts its bounce (state $30) when it touched him.
;   L0268 Fireball_QSpeedsX               data  TABLE: fireball horizontal q-speed; index = aim index 0-8 from Dir_CalcAimSpeedIndex (0 = along
;   L0272 Fireball_QSpeedsY               data  TABLE: fireball vertical q-speed; same index; value = q-speed ($70 = 1.75, $50 = 1.25 px/frame).
;   L0286 Fireball_Update                 code  Updates a fireball (types $55, $56): aims at Link once, holds still, then flies straight.
;   L0382 Fireball_MoveOneAxis            code  Moves a fireball along one axis with that axis's own q-speed and position fraction.
; ============================================================================

; ----------------------------------------------------------------------------
; Creates a fireball object of the given type 4 pixels right of the shooter.
; IN:   A = object type ($55 fireball the magic shield blocks, $56 fireball no shield blocks);
;       X = shooter's object slot.
; OUT:  X = shooter's slot; Y = the new slot, or 0 when none was free; [00] = object type.
; NOTE: No shot limit and no MonsterShotCount: the only condition is a free slot in $01-$0B. The new
;       object is uninitialized: its next update runs its init (MonsterShot_Init, no spawn cloud), and
;       the update after that is Fireball_Update in state 0, which aims it.
; ----------------------------------------------------------------------------
Fireball_Shoot:
    STA $00                     ; [00] holds the object type.

    ; Return, if there's no empty slot (Z set by Obj_FindEmptyMonsterSlot).
    JSR Obj_FindEmptyMonsterSlot
    BEQ @Exit
    TXA                         ; Save the shooter's object slot.
    PHA
    TYA                         ; Switch to the new object's slot.
    TAX
    LDA $00
    JSR Obj_SetTypeAndClear
    TXA                         ; Y = the new object's slot.
    TAY
    PLA                         ; Restore the shooter's object slot.
    TAX

    ; Set the new object's location to the shooter's, 4 pixels to the right.
    LDA ObjX, X
    CLC
    ADC #$04
    STA a:ObjX, Y
    LDA ObjY, X
    STA a:ObjY, Y
@Exit:
    RTS

; TABLE: X step per frame of a bouncing shot; index = reverse direction index of Link's facing
; (0 up, 1 down, 2 left, 3 right); value = signed pixels. Only two bytes belong to this label: indexes
; 2 and 3 read the first two bytes of MonsterShot_BounceStepsY ($FE, $02).
MonsterShot_BounceStepsX:
    .BYTE $01, $FF              ; Link facing up: +1; down: -1 (left: -2; right: +2 from the next table)

; TABLE: Y step per frame of a bouncing shot; same index; value = signed pixels.
MonsterShot_BounceStepsY:
    .BYTE $FE, $02, $FF, $FF    ; Link facing up: -2; down: +2; left: -1; right: -1

; ----------------------------------------------------------------------------
; Updates a straight-flying monster shot: types $53, $54, $57, $58, $59 and $5A.
; IN:   X = object slot.
; NOTE: State $1x = flying; any other state runs MonsterShot_Bounce. The q-speed comes from the type's init
;       routine (MonsterShot_Init). Destroyed when it reaches the room boundary or harms Link; types
;       $53 and $54 also when the tile ahead is unwalkable. A blocked shot starts to bounce instead.
; ----------------------------------------------------------------------------
MonsterShot_Update:
    ; Set moving direction [0F] to facing direction.
    LDA ObjDir, X
    STA $0F

    ; If major state <> $10, go bounce.
    LDA ObjState, X
    AND #$F0
    CMP #$10
    BNE MonsterShot_Bounce

    ; If object type < $55 (flying rock $53, or $54), then
    ; check for tile collision in addition to other checks.
    LDA ObjType, X
    CMP #$55
    BCS @CheckBoundary

    ; While this shot's object timer runs, it does not move and
    ; only checks for a collision with Link.
    LDA ObjTimer, X
    BNE @CheckLinkCollision

    ; If the tile ahead is unwalkable, then go destroy the shot.
    JSR Tile_GetAheadOfObj
    CMP RoomFirstUnwalkableTile
    BCS MonsterShot_Destroy

@CheckBoundary:
    ; No shots can cross the room boundary (Room_BoundDir returns 0 when crossed).
    JSR Room_BoundDir
    BEQ MonsterShot_Destroy

    ; Move the object, and check for a collision with Link.
    ; Go destroy the shot object, if there was a harmful collision.
    JSR Obj_MoveByQSpeed

@CheckLinkCollision:
    JSR MonsterShot_CheckLinkCollision
    LDA $06                     ; 1 if Link was harmed; 0 if no touch or the shield blocked it.
    BNE MonsterShot_Destroy

; Draws a monster shot according to its type; also the draw step of fireballs and bouncing shots.
MonsterShot_Draw:
    ; If object type = arrow ($5B), then draw an arrow, and return.
    LDA ObjType, X
    CMP #$5B
    BNE :+
    JMP Arrow_Draw
:
    ; If object type = sword shot ($57) or magic shot ($58, $59), then go draw, and return.
    CMP #$57
    BCC @DrawOthers
    CMP #$5A
    BCS @DrawOthers
    JMP SwordShot_DrawOrMagicShot

@DrawOthers:
    ; Prepare to draw other shot types ($53-$56, $5A).
    JSR Anim_FetchObjPosForSpriteDescriptor
    STA $0D                     ; [0D] holds frame image 0

    ; Assume object type >= $55. Prepare sprite attributes that
    ; make the shot flash: the palette changes every frame.
    LDA FrameCounter
    AND #$03

    ; If object type < $55 (flying rock is $53), then
    ; add 4 to the sprite's X coordinate, and use sprite attributes 0.
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
    JMP Anim_DrawObjectNotMirrored

; ----------------------------------------------------------------------------
; Destroys a monster shot; MonsterShotCount is decremented unless it is a fireball ($55, $56).
; IN:   X = object slot.
; ----------------------------------------------------------------------------
MonsterShot_Destroy:
    ; Fireballs are not counted, so skip the decrement for them.
    LDA ObjType, X
    CMP #$55
    BEQ :+
    CMP #$56
    BEQ :+

; Destroys a counted shot (also monster arrows and boomerangs): decrements MonsterShotCount first.
MonsterShot_DestroyCounted:
    DEC MonsterShotCount
:
    JMP Obj_DestroyMonster

; ----------------------------------------------------------------------------
; Moves a shot that Link's shield blocked: 2 pixels per frame the way Link faced, for $10 frames.
; IN:   X = object slot.
; NOTE: The sideways step is 1 pixel per frame (MonsterShot_BounceStepsX/Heights). No collision test of any
;       kind runs in this state. At bounce distance $20 the shot is destroyed.
; ----------------------------------------------------------------------------
MonsterShot_Bounce:
    ; Y = reverse direction index of the bounce direction (8 up -> 0, 4 down -> 1, 2 left -> 2,
    ; 1 right -> 3). The opposite direction returned in A is not used.
    LDA MonsterShot_ObjBounceDir, X
    JSR Dir_GetOpposite

    ; Modify the coordinates according to the displacements for
    ; the bounce direction.
    LDA ObjY, X
    CLC
    ADC MonsterShot_BounceStepsY, Y
    STA ObjY, X
    LDA ObjX, X
    CLC
    ADC MonsterShot_BounceStepsX, Y
    STA ObjX, X

    ; Add 2 to the bounce distance.
    ; The absolute value of one of the displacements is 2.
    LDA MonsterShot_ObjBounceDist, X
    CLC
    ADC #$02
    STA MonsterShot_ObjBounceDist, X

    ; Go destroy the shot, if the bounce distance has reached the limit ($20).
    ; Otherwise, go draw.
    CMP #$20
    BCS MonsterShot_Destroy
    BCC MonsterShot_Draw

; ----------------------------------------------------------------------------
; Tests a shot against Link and starts its bounce (state $30) when it touched him.
; IN:   X = shot's object slot.
; OUT:  [06] = 1 if Link was harmed, 0 if there was no touch or the shield blocked the shot;
;       [0C] = 1 if Link was harmed; [00] = 0 (Link's slot); [09] = 0 (no damage type);
;       IsShotCollidingWithLink = 1 on any touch, blocked or not.
; NOTE: Clears MonsterShot_ObjBounceDist every call. That byte is also the shot's ObjGridOffset, so
;       Obj_MoveByQSpeed's grid-offset limit never stops a flying shot. State $30 is set on a harmful touch
;       too; the callers destroy the shot in that case.
; ----------------------------------------------------------------------------
MonsterShot_CheckLinkCollision:
    LDA #$00
    STA MonsterShot_ObjBounceDist, X
    JSR Link_CheckMonsterTouch
    LDA IsShotCollidingWithLink
    BEQ @Exit

    ; The shot touched Link. Remember Link's facing direction as the
    ; bounce direction, and set state $30 to bounce.
    LDA ObjDir
    STA MonsterShot_ObjBounceDir, X
    LDA #$30
    STA ObjState, X
@Exit:
    RTS

; TABLE: fireball horizontal q-speed; index = aim index 0-8 from Dir_CalcAimSpeedIndex (0 = along
; the X axis, 4 = diagonal, 8 = along the Y axis); value = q-speed (pixels per frame = value / 64).
; Index 8 reads the first byte of Fireball_QSpeedsY ($00).
Fireball_QSpeedsX:
    .BYTE $70, $68, $60, $58, $50, $3C, $26, $10

; TABLE: fireball vertical q-speed; same index; value = q-speed ($70 = 1.75, $50 = 1.25 px/frame).
Fireball_QSpeedsY:
    .BYTE $00, $10, $26, $3C, $50, $58, $60, $68
    .BYTE $70

; ----------------------------------------------------------------------------
; Updates a fireball (types $55, $56): aims at Link once, holds still, then flies straight.
; IN:   X = object slot.
; NOTE: State 0 (first update): takes the directions and distances to Link's ObjX/ObjY (slot 0, not
;       the chase target), picks one of 9 speed pairs, sets state $10 and ObjTimer $10; nothing is
;       drawn or tested on that frame. State $10: for the next $0F frames (timer nonzero) it stays
;       put but is drawn and can hit Link; from the $10th frame it moves on both axes each
;       frame. Destroyed at the room boundary and on any touch of Link (harmful or blocked). No tile
;       test. Not counted in MonsterShotCount.
; ----------------------------------------------------------------------------
Fireball_Update:
    LDA ObjState, X
    BNE @Flying

    ; State 0.
    ;
    ; Reset horizontal and vertical position fractions (A = 0).
    STA Fireball_ObjPosFracX, X
    STA Fireball_ObjPosFracY, X

    ; Link is the target. So, register A has 0 -- his object slot.
    JSR Dir_GetDirsAndDistancesToTarget

    ; Remember the horizontal ([0B]: 1 or 2) and vertical ([0A]: 4 or 8) directions toward Link.
    ; On equal coordinates the direction is left or up.
    LDA $0B
    STA Fireball_ObjDirToTargetX, X
    LDA $0A
    STA Fireball_ObjDirToTargetY, X

    ; Set the facing direction to the combination of the two.
    ORA $0B
    STA ObjDir, X

    ; Y = aim index 0-8, starting from 4 (diagonal) and stepping toward the axis
    ; with the longer distance.
    LDY #$04
    JSR Dir_CalcAimSpeedIndex

    ; Look up and set the horizontal and vertical q-speeds.
    LDA Fireball_QSpeedsX, Y
    STA Fireball_ObjQSpeedX, X
    LDA Fireball_QSpeedsY, Y
    STA Fireball_ObjQSpeedY, X

    ; Set state to $10: monster shot active.
    ; Set object timer to $10, so it's seen but waits $10 frames before moving.
    LDA #$10
    STA ObjState, X
    STA ObjTimer, X
    RTS

@Flying:
    ; Any state other than 0.
    ;
    ; Fireballs are visible and can hit Link while they wait to move.
    ; If timer <> 0, go check for collision and draw.
    LDA ObjTimer, X
    BNE @CheckCollisionAndDraw

    ; If the fireball has reached the room boundary in its (diagonal) direction, then destroy it.
    LDA ObjDir, X
    JSR Room_BoundDirA
    BNE @Move

@Destroy:
    JMP Obj_DestroyMonster

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
    ; Check for collision with Link. If touched (harmed or blocked), then go destroy the fireball.
    JSR MonsterShot_CheckLinkCollision
    LDA IsShotCollidingWithLink
    BNE @Destroy
    JMP MonsterShot_Draw

; ----------------------------------------------------------------------------
; Moves a fireball along one axis with that axis's own q-speed and position fraction.
; IN:   A = q-speed; Y = position fraction; [0F] = direction on this axis; X = object slot.
; OUT:  A = updated position fraction; ObjQSpeedFrac,X and ObjPosFrac,X hold this axis's values.
; ----------------------------------------------------------------------------
Fireball_MoveOneAxis:
    STA ObjQSpeedFrac, X
    TYA
    STA ObjPosFrac, X
    JSR Obj_MoveByQSpeed

    ; Return the updated position fraction.
    LDA ObjPosFrac, X
    RTS
