; ============================================================================
; FILE:     bank5/ladder.asm             BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 3219-3416
; SUBJECTS: (none)
; TOPIC:    Ladder rules, movement override, drawing
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0019 CheckLadder                        code  Params:
;   L0207 SetMovingDirAndSwitchToPlayerSlot  code  Params:
; ============================================================================

; Params:
; [0F]: movement direction
;
; Returns:
; [0F]: movement direction or 0
;
CheckLadder:
    ; If the ladder's not in use, return.
    LDX LadderSlot
    BEQ @Exit

    ; If the ladder is done (state 0), go put away the ladder.
    LDA ObjState, X
    BEQ @StashLadder

    ; If the ladder is facing vertically, calculate the vertical distance
    ; between the ladder and Link.
    ; If Link's X no longer matches the ladder's, go put the ladder away.
    LDA ObjDir, X
    AND #$0C
    BEQ :+
    LDA ObjX
    CMP ObjX, X
    BNE @StashLadder
    LDA ObjY
    CLC
    ADC #$03
    SEC
    SBC ObjY, X
    JMP @CheckDistanceToLadder
:
    ; If instead it's facing horizontally, calculate the horizontal distance
    ; between the ladder and Link.
    ; If Link's Y no longer matches the ladder's, go put the ladder away.
    LDA ObjY
    CLC
    ADC #$03
    CMP ObjY, X
    BNE @StashLadder
    LDA ObjX
    SEC
    SBC ObjX, X

@CheckDistanceToLadder:
    ; If absolute distance < $10, go handle movement on the ladder and set state 2.
    ; If > $10, go put the ladder away.
    JSR Abs
    STA $00                     ; [00] holds the absolute distance between Link and the ladder in whichever axis is relevant.
    CMP #$10
    BCC @SetState2
    CMP #$10
    BNE @StashLadder

    ; Distance = $10. If player's not facing in the same direction as
    ; the ladder, go put away the ladder.
    LDA ObjDir
    CMP ObjDir, X
    BNE @StashLadder

    ; Distance = $10, and player's facing in the same direction.
    ;
    ; If the ladder's still in initial state 1, then Link's facing the
    ; ladder and had not stepped onto it yet.
    ; Go handle movement on it, but don't change state yet.
    ;
    ; But if ladder state is 2, then it means Link completely stepped
    ; off of the ladder. So the ladder should be put away. In this
    ; case, fall thru.
    LDA ObjState, X
    CMP #$01
    BEQ @HandleInput

@StashLadder:
    ; Put away the ladder. Reset ladder slot and destroy the object.
    LDA #$00
    STA LadderSlot
    JSR DestroyMonster

@Exit:
    RTS

@SetState2:
    ;  Set state 2, because we're on the ladder.
    LDA #$02
    STA ObjState, X

@HandleInput:
    ; If input direction = 0, go draw and reset moving direction.
    LDA ObjInputDir
    BEQ @DrawLadder

    ; Input direction <> 0. The player intends to move.
    ; So, see if we need to override the moving direction.
    ;
    ; A. If the distance <> 0, and they are facing the same way,
    ; go draw and set moving direction to Link's direction.
    ;
    ; This lets Link step onto the ladder. Otherwise he would have
    ; been blocked by the water.
    ;
    ; Or, Link already passed over the point right over the ladder
    ; and was allowed to move onto the tiles after the ladder.
    LDA ObjDir
    LDY $00                     ; [00] distance between Link and ladder
    BEQ :+
    CMP ObjDir, X
    BEQ @DrawLadder
:
    ; Distance = 0, or facing different directions.
    ;
    ; B. If ladder's direction = Link's *moving* direction,
    ; then go draw and keep moving.
    ;
    ; This only applies when distance = 0. Otherwise, (A) would have caught it.
    ;
    ; Before checking the ladder, a tile collision check allowed Link
    ; to move onto the tiles after the ladder.
    ;
    ; Note that if Link is not moving, then this test will fail,
    ; and case (D) will catch it.
    LDA ObjDir, X
    CMP $0F                     ; [0F] Link's moving direction
    BEQ @DrawLadder

    ; C. If opposite of ladder's direction = Link's facing direction,
    ; then go draw and move in ladder's direction.
    ;
    ; You can always step off the ladder where you came from.
    JSR GetOppositeDir
    CMP ObjDir
    BEQ @DrawLadder

    ; D. If opposite of ladder's direction <> down,
    ; or input direction <> up,
    ; then go reset moving direction, and draw.
    ;
    ; This will catch all cases of Link facing perpendicular to
    ; ladder direction. Also, it will catch Link not moving, unless
    ; ladder direction is up and input direction is up (E).
    CMP #$04
    BNE :+
    LDA ObjInputDir
    CMP #$08
    BNE :+

    ; E. The ladder's direction is up and input direction is up.
    ;
    ; An earlier call to check tile collision would have blocked
    ; movement, because the tile under Link's top half is a water tile.
    ; But because Link is squarely on the ladder, we really have
    ; to check the tile above that one.
    ;
    ; Set moving direction to input direction (up) and switch X to
    ; the player's slot for the purpose of checking tile collision below.
    ; Based on that, we'll set moving direction according to walkability.
    JSR SetMovingDirAndSwitchToPlayerSlot

    ; Check the colliding tile as if Link was 8 pixels up.
    LDA ObjY
    PHA
    SEC
    SBC #$08
    STA ObjY
    JSR GetCollidingTileMoving
    PLA
    STA ObjY

    ; If the tile is walkable, go draw the ladder and leave the moving
    ; direction as the input direction (in a roundabout way).
    ; Else fall thru to reset moving direction and draw the ladder.
    LDA $0F
    LDY ObjCollidedTile
    CPY RoomFirstUnwalkableTile
    BCC @HandleInput
:
    ; Set A to reset moving direction in [0F].
    LDA #$00

@DrawLadder:
    ; Draw the ladder.
    PHA
    LDX LadderSlot
    JSR Anim_FetchObjPosForSpriteDescriptor
    LDY #$0C
    LDA #$00
    JSR Anim_WriteStaticItemSpritesWithAttributes
    PLA

; Params:
; A: direction
;
; Returns:
; [0F]: direction
;
SetMovingDirAndSwitchToPlayerSlot:
    STA $0F
    LDX #$00
    RTS
