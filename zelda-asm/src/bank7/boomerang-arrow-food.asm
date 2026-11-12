; ============================================================================
; FILE:     bank7/boomerang-arrow-food.asm BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 3763-4090
; SUBJECTS: (none)
; TOPIC:    Food, boomerang tables, arrow/boomerang flight
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0026 UpdateBoomerangOrFood                   code
;   L0086 BoomerangFrameCycle                     data
;   L0090 BoomerangBaseSpriteAttrCycle            data
;   L0094 BoomerangQSpeedFracsY                   data
;   L0098 BoomerangQSpeedFracsX                   data
;   L0102 RDirectionToWeaponFrame                 data
;   L0111 RDirectionToWeaponBaseAttribute         data  The base sprite attribute for weapon sprites, 1 for each
;   L0114 RDirectionToOffsetsX                    data
;   L0117 RDirectionToOffsetsY                    data
;   L0120 UpdateArrowOrBoomerang                  code
;   L0213 DrawArrow                               code
;   L0233 SetAttrAndDrawArrow                     code
;   L0265 CheckState20                            code
;   L0313 HandleArrowOrBoomerangBlocked           code
;   L0324 DrawArrowOrBoomerangAndCheckCollisions  code
; ============================================================================

UpdateBoomerangOrFood:
    ; If state = 0, then not active. So, return.
    LDA ObjState, X
    BEQ L1F49F_Exit

    ; If the high bit of state is clear, then go update the boomerang.
    ASL
    BCC UpdateArrowOrBoomerang

    ; The object is food.
    ;
    ; Once the timer has expired, increment the state and set timer to $FF.
    ; If we reach state $83, then go deactivate the object.
    ;
    ; Nothing changes between the states. So, they are used as
    ; a way to extend the timer. $2FD screen frames in total.
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
    ; If the template object type in the room is one of:
    ; moblin, goriya, octorock, vire, keese
    ; then attract these monsters to the food.
    ;
    ; They will chase the food instead of Link when given a chance.
    LDA RoomObjTemplateType
    CMP #$03                    ; Blue moblin
    BCC @Draw
    CMP #$0B                    ; Red darknut
    BCC @AttractMonsters
    CMP #$12                    ; Vire
    BEQ @AttractMonsters
    CMP #$1B                    ; Blue keese
    BEQ @AttractMonsters
    CMP #$1C                    ; Red keese
    BNE @Draw

@AttractMonsters:
    LDA ObjX, X
    STA ChaseTargetX
    LDA ObjY, X
    STA ChaseTargetY

@Draw:
    ; Draw the food.
    JSR Anim_FetchObjPosForSpriteDescriptor
    LDA #$02                    ; Red sprite palette
    LDY #$06                    ; Food item slot
    JMP Anim_WriteStaticItemSpritesWithAttributes

@Deactivate:
    JMP ResetObjState

BoomerangFrameCycle:
    .BYTE $00, $01, $02, $01, $00, $01, $02, $01
    .BYTE $03

BoomerangBaseSpriteAttrCycle:
    .BYTE $00, $00, $00, $40, $40, $C0, $80, $80
    .BYTE $01

BoomerangQSpeedFracsY:
    .BYTE $00, $20, $36, $4C, $60, $68, $70, $78
    .BYTE $80

BoomerangQSpeedFracsX:
    .BYTE $80, $78, $70, $68, $60, $4C, $36, $20
    .BYTE $00

RDirectionToWeaponFrame:
    .BYTE $00, $00, $01, $01

; The base sprite attribute for weapon sprites, 1 for each
; direction in reverse direction order: up, down, left, right
;
; All of them are 0, except for the element for "down".
; That one flips the vertical sword or rod image so that it points down.
;
RDirectionToWeaponBaseAttribute:
    .BYTE $00, $80, $00, $00

RDirectionToOffsetsX:
    .BYTE $FC, $FC, $00, $00

RDirectionToOffsetsY:
    .BYTE $00, $00, $03, $03

UpdateArrowOrBoomerang:
    ; If not active (state = 0), then return.
    LDA ObjState, X
    BEQ L1F49F_Exit

    ; Reset [00], which will hold the number of axes where the distance <= 8
    ; when calculating the angle to return to thrower.
    ; See states $40 and $50 and GetDirectionsAndDistancesToTarget.
    LDA #$00
    STA $00

    ; If not in major state $10, go check other states.
    LDA ObjState, X
    AND #$F0
    CMP #$10
    BEQ @State10
    JMP CheckState20

@State10:
    ; State $1x. Fly away from the thrower.
    ;
    ; Reset [0E], which will indicate:
    ; - into MoveShot:   0 update grid offset, else don't
    ; - out of MoveShot: $80 if blocked
    LDA #$00
    STA $0E

    ; If the object's direction has a horizontal component, then
    ; move it horizontally.
    LDA ObjDir, X
    AND #$03
    BEQ :+
    JSR MoveShot
    INC $0E                     ; If MoveShot were called again, then don't update grid offset.
:
    ; If the object was blocked, go handle it.
    LDA $0E
    ASL
    BCS @HandleBlocked

    ; If the object's direction has a vertical component, then
    ; move it vertically.
    LDA ObjDir, X
    AND #$0C
    BEQ :+
    JSR MoveShot
:
    ; If the object was blocked, go handle it.
    LDA $0E
    ASL
    BCS @HandleBlocked

    ; If the object is an arrow (Link's or a monster's), go set up sprite parameters.
    CPX #$0D
    BCS @CheckPlayerArrow
    LDA ObjType, X
    CMP #$5B
    BEQ DrawArrow

@CheckPlayerArrow:
    CPX #$12
    BEQ DrawArrow

    ; For boomerangs, get the absolute value of the weapon's grid offset:
    ; the distance traveled.
    LDA ObjGridOffset, X
    BPL @CompareLimit
    EOR #$FF
    CLC
    ADC #$01

@CompareLimit:
    ; For boomerangs, compare it to the movement limit.
    ; If it reached the limit, set state $20 and a new movement
    ; limit of $10, and go handle the rest as if blocked.
    ; State $20 will become $30 below in HandleArrowOrBoomerangBlocked.
    ;
    ; Otherwise, only animate and check collision as usual.
    ;
    ; In major state $30, ObjMovingLimit is a timer to change the state to $40.
    CMP ObjMovingLimit, X
    BCC @AnimateBoomerang
    LDA #$10
    STA ObjMovingLimit, X
    LDA #$20
    STA ObjState, X

@HandleBlocked:
    JMP HandleArrowOrBoomerangBlocked

@AnimateBoomerang:
    JMP AnimateBoomerangAndCheckCollision

DrawArrow:
    ; This is an arrow. Set up sprite parameters.
    ;
    ; If facing left, set horizontal flipping in [0F].
    LDA #$00
    STA $0F
    LDA ObjDir, X
    CMP #$02
    BNE :+
    INC $0F
:
    JSR GetOppositeDir          ; Get the reverse direction index that the weapon is facing.

    ; Store the weapon's frame (up or right) in [0C].
    LDA RDirectionToWeaponFrame, Y
    STA $0C

    ; Store the base sprite attribute in [04].
    LDA RDirectionToWeaponBaseAttribute, Y

SetAttrAndDrawArrow:
    STA $04

    ; If this is a monster's arrow, add 2 to sprite attributes in [04].
    ; This chooses palette row 6.
    ;
    ; Otherwise, add the arrow item value less 1. This chooses
    ; palette row 4 if the arrow is wooden, otherwise 5 for silver.
    CPX #$0D
    BCS @PlayerArrow
    LDA ObjType, X
    CMP #$5B
    BNE @PlayerArrow
    LDA $04
    CLC
    ADC #$02
    BNE @SetAttr

@PlayerArrow:
    CLC
    ADC InvArrow
    SEC
    SBC #$01

@SetAttr:
    STA $04

    ; Copy the attribute to [05], and go draw.
    LDA $04
    STA $05
    JMP OffsetAndDrawArrow

CheckState20:
    ; If major state <> $20, go check other states.
    CMP #$20
    BNE CheckState30

    ; State $2x. Spark.
    ;
    ; Change the state to $28, and decrement animation counter.
    ; If it hasn't reached 0, then go draw and check for collision
    ; (with Link, if this is a monster's boomerang).
    ;
    ; Minor state 8 selects the frame image and base sprite attribute
    ; for the spark.
    LDA #$28
    STA ObjState, X
    DEC ObjAnimCounter, X
    BNE DrawArrowOrBoomerangAndCheckCollisions

    ; Animation counter = 0.
    ;
    ; Set state to $40. It will become $50 below in
    ; HandleArrowOrBoomerangBlocked.
    LDA #$40
    STA ObjState, X

    ; If this is an arrow (monster's or player's), then deactivate it.
    ; Else go handle the boomerang being blocked.
    CPX #$0D
    BCS @CheckBoomerangBlocked
    LDA ObjType, X
    CMP #$5B
    BEQ @Deactivate

@CheckBoomerangBlocked:
    CPX #$12
    BNE HandleArrowOrBoomerangBlocked

@Deactivate:
    JSR ResetObjState

    ; If it's a monster's arrow, then destroy it in the monster slot,
    ; including clearing object type.
    CPX #$0D
    BCS :+
    JSR DestroyCountedMonsterShot
:
    RTS

HandleArrowOrBoomerangBlocked:
    ; Use animation counter to count down 3 screen frames in case the state is $20 (spark).
    LDA #$03
    STA ObjAnimCounter, X

    ; Add $10 to the state to make it $2x.
    LDA ObjState, X
    CLC
    ADC #$10
    STA ObjState, X

DrawArrowOrBoomerangAndCheckCollisions:
    ; If the weapon is a boomerang (not a monster arrow nor player arrow),
    ; then go draw it and check for a collision.
    CPX #$0D
    BCS @DrawPlayerWeapon
    LDA ObjType, X
    CMP #$5B
    BEQ @PrepareArrow

@DrawPlayerWeapon:
    CPX #$12
    BEQ @PrepareArrow
    JMP DrawBoomerangAndCheckCollision

@PrepareArrow:
    ; Start preparing to draw an arrow.
    ;
    ; Set arrow slot frame 2 (spark) in [0C],
    ; and no horizontal flipping in [0F].
    LDA #$02
    STA $0C
    LDA #$00
    STA $0F

    ; Get reverse index of weapon's direction, and base sprite attribute 0,
    ; and go draw.
    LDA ObjDir, X
    JSR GetOppositeDir
    LDA #$00
    JMP SetAttrAndDrawArrow
