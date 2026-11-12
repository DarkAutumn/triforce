; ============================================================================
; FILE:     bank7/boomerang-arrow-food.asm BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 3763-4090
; SUBJECTS: Anim, Arrow, Boomerang
; TOPIC:    Food on the ground (object slot $0F): its lifetime and which monsters it attracts. The outward
;           flight and the spark of arrows and boomerangs, Link's and monsters' alike; arrow drawing;
;           sprite tables shared by the sword, rod, shots, arrow and boomerang.
; SEE ALSO: bank7/boomerang-return-draw.asm (boomerang states $3x-$5x, the catch, boomerang drawing,
;           Arrow_OffsetAndDraw); bank5/wield-items.asm (Boomerang_Wield, Arrow_Wield, Food_Wield: slot rules,
;           the rupee an arrow costs, range limits); bank1/shot-aim-boomerang-speed.asm (Obj_MoveShot,
;           Boomerang_SetReturnSpeed); bank1/room-bounds.asm (where a flight is blocked);
;           bank1/weapon-damage-death.asm (boomerang stun); bank1/sword-bomb-arrow-collision.asm (arrow
;           damage); bank4/drop-items.asm (Drop_UpdateItem: weapons that pick up items);
;           bank4/moblin-lynel-shooting.asm (MonsterArrow_Update); bank4/goriya.asm (monster boomerang).
; KEY FACTS:
;   - Slot $0F holds the boomerang (state bit 7 clear) or food (bit 7 set), never both.
;       Boomerang_UpdateOrFood
;   - Food lies for three timer periods of $FF frames (states $80, $81, $82): $2FD = 765 frames, then
;       the slot is freed. Boomerang_UpdateOrFood
;   - Food attracts by the room's first monster type (RoomObjTemplateType), not per monster: $03-$0A
;       (Moblin, Goriya, Octorok), $12 (Vire), $1B and $1C (blue and red Keese). Every frame it
;       overwrites ChaseTargetX/Y with its own position. Boomerang_UpdateOrFood
;   - Arrow and boomerang states: $1x flying out, $2x spark, $3x boomerang slowing, $4x slow return,
;       $5x fast return; the low 3 bits are the boomerang's spin step. Boomerang_UpdateOrArrow
;   - A flight in state $1x is blocked only by the room bounds (Obj_MoveShot), not by tiles.
;       Boomerang_UpdateOrArrow
;   - A boomerang turns back when the pixels flown (absolute ObjGridOffset) reach ObjMovingLimit: $31
;       wooden, $51 a Goriya's. The magic boomerang's $FF is never reached (the absolute value of a
;       signed byte is at most $80), so it flies until blocked or until it hits. Boomerang_UpdateOrArrow
;   - On a diagonal throw only the horizontal movement is counted toward the limit.
;       Boomerang_UpdateOrArrow
;   - The spark state lasts 3 frames. Then an arrow frees its slot (4th frame), and a boomerang
;       returns fast (state $50). Boomerang_UpdateOrArrow_State2Spark, Boomerang_HandleBlockedOrArrow
;   - An arrow has no range limit: it flies at its q-speed until blocked or until the collision code
;       stops it. Boomerang_UpdateOrArrow
; CONTENTS: (generated; L = line in this file)
;   L0070 Boomerang_UpdateOrFood               code  Updates object slot $0F each frame: the boomerang, or food lying on the ground.
;   L0125 Boomerang_FrameCycle                 data  TABLE: boomerang sprite frame; index = low nibble of the boomerang's state (0-7 = spin step, 8 = spark);
;   L0134 Boomerang_SpriteAttrCycle            data  TABLE: boomerang base sprite attributes; index = low nibble of the boomerang's state (as above);
;   L0141 Boomerang_ReturnQSpeedsY             data  TABLE: vertical q-speed of a returning boomerang; index = speed index 0-8 from Dir_CalcAimSpeedIndex
;   L0147 Boomerang_ReturnQSpeedsX             data  TABLE: horizontal q-speed of a returning boomerang; index = the same speed index; value = q-speed.
;   L0154 Anim_WeaponFramesByRDir              data  TABLE: weapon sprite frame by direction; index = reverse direction index (0 up, 1 down, 2 left,
;   L0160 Anim_WeaponBaseAttrsByRDir           data  TABLE: weapon base sprite attributes by direction; index = reverse direction index (0 up, 1 down,
;   L0165 Arrow_SpriteOffsetsXByRDir           data  TABLE: arrow sprite X offset by direction; index = reverse direction index (0 up, 1 down, 2 left,
;   L0170 Arrow_SpriteOffsetsYByRDir           data  TABLE: arrow sprite Y offset by direction; index = reverse direction index; value = pixels added
;   L0192 Boomerang_UpdateOrArrow              code  Updates a flying arrow or boomerang, Link's or a monster's: flight out, spark, boomerang return.
;   L0289 Arrow_Draw                           code  Draws a flying arrow (Link's or a monster's) pointing in its direction.
;   L0308 Arrow_SetAttrAndDraw                 code  Second entry of Arrow_Draw: A = base sprite attribute, Y = reverse direction index, [0C], [0F] set.
;   L0352 Boomerang_UpdateOrArrow_State2Spark  code  Handles the spark state ($2x) of an arrow or boomerang; passes other states to Boomerang_UpdateSlowAndReturn.
;   L0404 Boomerang_HandleBlockedOrArrow       code  Moves a blocked arrow or boomerang to its next major state (adds $10) and draws it.
;   L0419 Boomerang_DrawOrArrowAndCollide      code  Draws an arrow's spark, or draws a boomerang and (for a monster's) tests it against Link.
; ============================================================================

; ----------------------------------------------------------------------------
; Updates object slot $0F each frame: the boomerang, or food lying on the ground.
; IN:   X = object slot ($0F; called once a frame by Mode5Play_Update)
; NOTE: State 0 = empty. Bit 7 clear = boomerang (handled by Boomerang_UpdateOrArrow). Bit 7 set = food.
;       Food: Food_Wield starts it in state $80 with ObjTimer = $FF. Each time the timer reaches 0 the
;       state goes up by 1 and the timer restarts at $FF; on reaching $83 the food is removed. The
;       three states differ in nothing else: 3 x $FF = $2FD (765) frames on the ground.
;       While it lies there, if RoomObjTemplateType (the type of the first monster the room was
;       populated with) is $03-$0A (Moblin, Goriya, Octorok), $12 (Vire), $1B (blue Keese) or $1C
;       (red Keese), the monsters' chase target is set to the food's position every frame. The test
;       is on that one type, not on each monster: every monster that steers by ChaseTargetX/Y then
;       goes for the food, and none does when the first monster is of another type.
;       This runs after Mode5Play_Update has copied Link's position to the chase target, so the food
;       wins on every frame, whatever IsChasingOtherTarget says.
;       Food does not touch monsters: Boomerang_CheckMonsterCollision returns when bit 7 is set.
; ----------------------------------------------------------------------------
Boomerang_UpdateOrFood:
    ; If state = 0, the slot is empty. Return.
    LDA ObjState, X
    BEQ SwordShot_Return

    ; If bit 7 of the state is clear, go update the boomerang.
    ASL
    BCC Boomerang_UpdateOrArrow

    ; The object is food.
    ;
    ; Once the timer has expired, increment the state and set the timer to $FF again.
    ; If the state reaches $83, go deactivate the object.
    LDA ObjTimer, X
    BNE @CheckMonsterType
    INC ObjState, X
    LDA ObjState, X
    AND #$0F
    CMP #$03
    BEQ @Deactivate
    LDA #$FF
    STA ObjTimer, X

@CheckMonsterType:
    ; If the room's first monster type is $03-$0A, $12, $1B or $1C, make the food the chase target.
    LDA RoomObjTemplateType
    CMP #$03                    ; Below $03 (Lynel $01/$02, or none): not attracted.
    BCC @Draw
    CMP #$0B                    ; $03-$0A: Moblin, Goriya, Octorok (all variants).
    BCC @AttractMonsters
    CMP #$12                    ; Vire
    BEQ @AttractMonsters
    CMP #$1B                    ; Blue Keese
    BEQ @AttractMonsters
    CMP #$1C                    ; Red Keese
    BNE @Draw

@AttractMonsters:
    LDA ObjX, X
    STA ChaseTargetX
    LDA ObjY, X
    STA ChaseTargetY

@Draw:
    ; Draw the food.
    JSR Anim_FetchObjPosForSpriteDescriptor
    LDA #$02                    ; Sprite attributes: palette row 6.
    LDY #$06                    ; Item sprite slot 6: food.
    JMP Anim_WriteStaticItemSpritesWithAttributes

@Deactivate:
    JMP Obj_ResetState

; TABLE: boomerang sprite frame; index = low nibble of the boomerang's state (0-7 = spin step, 8 = spark);
;   value = frame of item sprite slot $1D (0-2 = three views of the boomerang, 3 = spark).
Boomerang_FrameCycle:
    .BYTE $00, $01, $02, $01, $00, $01, $02, $01
    .BYTE $03

; TABLE: boomerang base sprite attributes; index = low nibble of the boomerang's state (as above);
;   value = OAM flip bits ($40 horizontal, $80 vertical) for the spin steps; entry 8 (spark) is 1.
;   Boomerang_DrawSpinFrame adds InvMagicBoomerang (0 or 1) to it, which selects the palette. It skips the
;   addition only for a table value of 8, which does not occur, so the addition always happens, for a
;   monster's boomerang too.
Boomerang_SpriteAttrCycle:
    .BYTE $00, $00, $00, $40, $40, $C0, $80, $80
    .BYTE $01

; TABLE: vertical q-speed of a returning boomerang; index = speed index 0-8 from Dir_CalcAimSpeedIndex
;   (0 = thrower is level with it, 4 = on the diagonal, 8 = straight above or below); value = q-speed
;   ($80 = 2 pixels a frame, $60 = 1.5). State $4x uses half of it (Boomerang_SetReturnSpeed).
Boomerang_ReturnQSpeedsY:
    .BYTE $00, $20, $36, $4C, $60, $68, $70, $78
    .BYTE $80

; TABLE: horizontal q-speed of a returning boomerang; index = the same speed index; value = q-speed.
;   It is the table above read backward.
Boomerang_ReturnQSpeedsX:
    .BYTE $80, $78, $70, $68, $60, $4C, $36, $20
    .BYTE $00

; TABLE: weapon sprite frame by direction; index = reverse direction index (0 up, 1 down, 2 left,
;   3 right, as returned in Y by Dir_GetOpposite); value = frame: 0 = vertical image, 1 = horizontal.
;   Used for the sword, rod, sword shot, magic shot and arrow.
Anim_WeaponFramesByRDir:
    .BYTE $00, $00, $01, $01

; TABLE: weapon base sprite attributes by direction; index = reverse direction index (0 up, 1 down,
;   2 left, 3 right); value = OAM attributes. Only "down" is nonzero: $80 flips the vertical image
;   so that it points down. (Left is flipped separately, through [0F].)
Anim_WeaponBaseAttrsByRDir:
    .BYTE $00, $80, $00, $00

; TABLE: arrow sprite X offset by direction; index = reverse direction index (0 up, 1 down, 2 left,
;   3 right); value = signed pixels added to ObjX when drawing: -4 for a vertical arrow.
Arrow_SpriteOffsetsXByRDir:
    .BYTE $FC, $FC, $00, $00

; TABLE: arrow sprite Y offset by direction; index = reverse direction index; value = pixels added
;   to ObjY when drawing: +3 for a horizontal arrow.
Arrow_SpriteOffsetsYByRDir:
    .BYTE $00, $00, $03, $03

; ----------------------------------------------------------------------------
; Updates a flying arrow or boomerang, Link's or a monster's: flight out, spark, boomerang return.
; IN:   X = object slot: $0F Link's boomerang, $12 Link's arrow (from Rod_UpdateRodOrArrow), or a monster
;       slot holding a monster arrow (type $5B, from MonsterArrow_Update) or boomerang (type $5C)
; NOTE: Major state (high nibble): $1x flying out; $2x spark; $3x boomerang slowing down; $4x boomerang
;       returning slowly; $5x returning fast (the last three are in Boomerang_UpdateSlowAndReturn). The low 3 bits of a
;       boomerang's state are its spin step; bit 3 is set only in the spark state $28.
;       State $1x moves at ObjQSpeedFrac (Link's arrow and boomerang $C0 = 3 pixels a frame, a monster
;       arrow $80 = 2, a Goriya's boomerang $A0 = 2.5) along each axis of ObjDir; a boomerang thrown
;       diagonally moves that far on both axes. Obj_MoveShot blocks the flight only at the room bounds;
;       tiles are not tested. Blocked: spark (state $20).
;       An arrow has no range limit. A boomerang turns back when the absolute value of its
;       ObjGridOffset (pixels flown; on a diagonal only the horizontal part is counted) is at least
;       ObjMovingLimit: $31 for the wooden boomerang (reached after 17 frames, 51 pixels), $51 for a
;       Goriya's. The magic boomerang's limit $FF cannot be reached, because the absolute value of
;       the signed offset is at most $80: it flies until blocked or until it hits something.
;       On reaching the limit the boomerang goes to state $30 with ObjMovingLimit = $10 (now a timer).
;       Monster collisions of Link's weapons are not tested here but in the monsters' collision code.
; ----------------------------------------------------------------------------
Boomerang_UpdateOrArrow:
    ; If not active (state = 0), then return.
    LDA ObjState, X
    BEQ SwordShot_Return

    ; Reset [00], which will count the axes on which the boomerang is within 8 pixels of its
    ; thrower. See states $40 and $50 in Boomerang_UpdateSlowAndReturn, and Dir_GetDirsAndDistancesToTarget.
    LDA #$00
    STA $00

    ; If not in major state $10, go check other states.
    LDA ObjState, X
    AND #$F0
    CMP #$10
    BEQ @State10
    JMP Boomerang_UpdateOrArrow_State2Spark

@State10:
    ; State $1x. Fly away from the thrower.
    ;
    ; Reset [0E], which is used:
    ; - into Obj_MoveShot:   0 = add the movement to the grid offset; else leave the grid offset alone
    ; - out of Obj_MoveShot: $80 if blocked at a room bound
    LDA #$00
    STA $0E

    ; If the object's direction has a horizontal component, then
    ; move it horizontally. (A = only the horizontal bits.)
    LDA ObjDir, X
    AND #$03
    BEQ :+
    JSR Obj_MoveShot
    INC $0E                     ; A second Obj_MoveShot call (vertical) will not change the grid offset.
:
    ; If the object was blocked (bit 7 of [0E]), go handle it.
    LDA $0E
    ASL
    BCS @HandleBlocked

    ; If the object's direction has a vertical component, then
    ; move it vertically. (A = only the vertical bits.)
    LDA ObjDir, X
    AND #$0C
    BEQ :+
    JSR Obj_MoveShot
:
    ; If the object was blocked, go handle it.
    LDA $0E
    ASL
    BCS @HandleBlocked

    ; If the object is an arrow (a monster's: slot < $0D and type $5B; Link's: slot $12),
    ; go draw it. Nothing else happens to a flying arrow here.
    CPX #$0D
    BCS @CheckLinkArrow
    LDA ObjType, X
    CMP #$5B
    BEQ Arrow_Draw

@CheckLinkArrow:
    CPX #$12
    BEQ Arrow_Draw

    ; For boomerangs, get the absolute value of the weapon's grid offset:
    ; the distance traveled.
    LDA ObjGridOffset, X
    BPL @CompareLimit
    EOR #$FF
    CLC
    ADC #$01

@CompareLimit:
    ; Compare the distance to the boomerang's limit. Below it: only animate and check collision.
    ;
    ; At or past the limit: set state $20 and ObjMovingLimit = $10, and go on as if blocked.
    ; Boomerang_HandleBlockedOrArrow adds $10, so the state becomes $30 (slowing down), not a spark.
    ; In major state $30, ObjMovingLimit is a timer: $10 frames until state $40.
    CMP ObjMovingLimit, X
    BCC @AnimateBoomerang
    LDA #$10
    STA ObjMovingLimit, X
    LDA #$20
    STA ObjState, X

@HandleBlocked:
    JMP Boomerang_HandleBlockedOrArrow

@AnimateBoomerang:
    JMP Boomerang_AnimateAndCheckCollision

; ----------------------------------------------------------------------------
; Draws a flying arrow (Link's or a monster's) pointing in its direction.
; IN:   X = object slot ($12, or a monster slot holding type $5B); ObjDir,X = direction
; NOTE: Also the tail of Boomerang_UpdateOrArrow for arrows in state $1x, called by MonsterArrow_Update
;       while the monster's arrow is held back by its timer, and by the monster shot draw code for
;       type $5B.
; ----------------------------------------------------------------------------
Arrow_Draw:
    ; If facing left, set horizontal flipping in [0F].
    LDA #$00
    STA $0F
    LDA ObjDir, X
    CMP #$02
    BNE :+
    INC $0F
:
    JSR Dir_GetOpposite          ; Y = reverse direction index of the arrow's direction.

    ; Store the weapon's frame (0 vertical, 1 horizontal) in [0C].
    LDA Anim_WeaponFramesByRDir, Y
    STA $0C

    ; A = base sprite attribute ($80, vertical flip, when pointing down).
    LDA Anim_WeaponBaseAttrsByRDir, Y

; Second entry of Arrow_Draw: A = base sprite attribute, Y = reverse direction index, [0C], [0F] set.
Arrow_SetAttrAndDraw:
    STA $04

    ; If this is a monster's arrow (slot < $0D and type $5B), add 2 to the sprite attributes.
    ; This chooses palette row 6. (A monster slot of any other type would reach @LinkArrow with
    ; A = its object type instead of the attribute; no caller sends one.)
    ;
    ; Otherwise, add the arrow item value (InvArrow) less 1. This chooses
    ; palette row 4 for the wooden arrow (1), row 5 for the silver arrow (2).
    CPX #$0D
    BCS @LinkArrow
    LDA ObjType, X
    CMP #$5B
    BNE @LinkArrow
    LDA $04
    CLC
    ADC #$02
    BNE @SetAttr                ; Always taken.

@LinkArrow:
    CLC
    ADC InvArrow
    SEC
    SBC #$01

@SetAttr:
    STA $04

    ; Copy the attribute to [05] (right sprite), and go draw.
    LDA $04
    STA $05
    JMP Arrow_OffsetAndDraw

; ----------------------------------------------------------------------------
; Handles the spark state ($2x) of an arrow or boomerang; passes other states to Boomerang_UpdateSlowAndReturn.
; IN:   X = object slot; A = major state (state AND $F0), not $10
; NOTE: Continuation of Boomerang_UpdateOrArrow. The spark state lasts 3 frames: the frame on which
;       the weapon was blocked (ObjAnimCounter set to 3), and two more here (counter 2, then 1).
;       On the frame the counter reaches 0, an arrow is removed (Link's slot $12 is freed; a
;       monster's arrow is destroyed and MonsterShotCount decremented), and a boomerang goes to
;       state $50 (fast return).
;       The state is forced to $28 so that minor state 8 selects the boomerang's spark image; on
;       the first of the 3 frames a boomerang still shows its spin image (in the spark's palette).
; ----------------------------------------------------------------------------
Boomerang_UpdateOrArrow_State2Spark:
    ; If major state <> $20, go check other states.
    CMP #$20
    BNE Boomerang_UpdateSlowAndReturn

    ; State $2x. Spark.
    ;
    ; Change the state to $28, and decrement the animation counter.
    ; If it hasn't reached 0, then go draw and check for collision
    ; (with Link, if this is a monster's boomerang).
    LDA #$28
    STA ObjState, X
    DEC ObjAnimCounter, X
    BNE Boomerang_DrawOrArrowAndCollide

    ; Animation counter = 0.
    ;
    ; Set state to $40. For a boomerang it becomes $50 below, in
    ; Boomerang_HandleBlockedOrArrow.
    LDA #$40
    STA ObjState, X

    ; If this is an arrow (a monster's: slot < $0D and type $5B; Link's: slot $12), then
    ; deactivate it. Else go send the boomerang back.
    CPX #$0D
    BCS @CheckLinkArrow
    LDA ObjType, X
    CMP #$5B
    BEQ @Deactivate

@CheckLinkArrow:
    CPX #$12
    BNE Boomerang_HandleBlockedOrArrow

@Deactivate:
    JSR Obj_ResetState

    ; If it's a monster's arrow, then destroy the object in the monster slot (clearing its
    ; object type) and decrement the count of monster shots.
    CPX #$0D
    BCS :+
    JSR MonsterShot_DestroyCounted
:
    RTS

; ----------------------------------------------------------------------------
; Moves a blocked arrow or boomerang to its next major state (adds $10) and draws it.
; IN:   X = object slot; ObjState,X = $1x (blocked at a room bound: becomes $2x, spark), $20 (boomerang
;       at its range limit: becomes $30, slowing down) or $40 (spark over: becomes $50, fast return)
; NOTE: Sets ObjAnimCounter = 3: the length of the spark, or of the boomerang's next spin step.
;       The low nibble of a boomerang's state (its spin step) is 0 after the $20 and $40 cases.
; ----------------------------------------------------------------------------
Boomerang_HandleBlockedOrArrow:
    LDA #$03
    STA ObjAnimCounter, X

    ; Add $10 to the state.
    LDA ObjState, X
    CLC
    ADC #$10
    STA ObjState, X

; ----------------------------------------------------------------------------
; Draws an arrow's spark, or draws a boomerang and (for a monster's) tests it against Link.
; IN:   X = object slot
; NOTE: Arrows reach this only in the spark state, so the arrow path always draws the spark image.
; ----------------------------------------------------------------------------
Boomerang_DrawOrArrowAndCollide:
    ; If the weapon is a boomerang (neither a monster's arrow nor Link's arrow),
    ; then go draw it and check for a collision.
    CPX #$0D
    BCS @CheckLinkWeapon
    LDA ObjType, X
    CMP #$5B
    BEQ @PrepareArrow

@CheckLinkWeapon:
    CPX #$12
    BEQ @PrepareArrow
    JMP Boomerang_DrawAndCheckCollision

@PrepareArrow:
    ; Draw the arrow's spark.
    ;
    ; Set frame 2 of the arrow's item sprite slot (spark) in [0C],
    ; and no horizontal flipping in [0F].
    LDA #$02
    STA $0C
    LDA #$00
    STA $0F

    ; Get the reverse index of the weapon's direction in Y (for the sprite offsets), and base
    ; sprite attribute 0 in A, and go draw.
    LDA ObjDir, X
    JSR Dir_GetOpposite
    LDA #$00
    JMP Arrow_SetAttrAndDraw
