; ============================================================================
; FILE:     bank5/ladder.asm             BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 3219-3416
; SUBJECTS: Ladder
; TOPIC:    The laid-down ladder (object type $5F): each frame decides whether it stays, lets Link walk
;           across it by overriding his moving direction, and draws it.
; SEE ALSO: bank7/link-animate-ladder.asm (Link_EndMoveAndAnimate lays the ladder down: Ladder_RoomIdsOW,
;           Ladder_LinkOffsetsX/Y, the water tile test); bank7/walker-move.asm (Walker_Move calls
;           Ladder_CheckAndDraw after the tile collision test, then falls into Obj_MoveByQSpeed);
;           bank7/walker-collision-screen-edge.asm (no screen-edge room change while LadderSlot <> 0);
;           bank5/mode3-mode6-init.asm (Obj_ResetWeaponSlotsAndLadder clears LadderSlot).
; KEY FACTS:
;   - LadderSlot = the ladder's object slot, 0 = no ladder out. The ladder's ObjState: 1 = laid down,
;     Link not yet on it; 2 = Link is or has been on it (Ladder_CheckAndDraw@SetState2).
;   - The ladder has no timer. It is put away (Ladder_CheckAndDraw@StashLadder) when: its state is 0; Link is
;     off its line (X differs for a vertical ladder, Y + 3 differs for a horizontal one); Link is more
;     than $10 pixels from it along its axis; or he is exactly $10 away and either faces another way
;     than the ladder or the state is already 2 (he has crossed and stepped off).
;   - On the ladder (distance < $10, or $10 in state 1) with a direction held, Link may move: in his
;     facing direction when it equals the ladder's and distance <> 0; in the ladder's direction when
;     that is his moving direction; back the way he came; all else is stopped ([0F] = 0).
;   - A ladder laid upward, exactly Up held, and none of the three cases above: the tile test is
;     repeated with Link's Y 8 less; Link may go on only if that tile is walkable (below
;     RoomFirstUnwalkableTile) (Ladder_CheckAndDraw, case E).
;   - In the frame the ladder is put away Ladder_CheckAndDraw returns X = the ladder's slot, not 0, and leaves
;     [0F] alone, so the Obj_MoveByQSpeed that follows in Walker_Move acts on the freed slot and Link does
;     not move that frame.
;   - The ladder is drawn with item slot $0C's sprites, palette attribute 0 (Ladder_CheckAndDraw@DrawLadder).
; CONTENTS: (generated; L = line in this file)
;   L0044 Ladder_CheckAndDraw               code  Maintains the laid-down ladder: puts it away, or draws it and decides Link's moving direction on it.
;   L0235 Ladder_SetMovingDirAndSelectLink  code  Stores A as the moving direction and selects Link's object slot.
; ============================================================================

; ----------------------------------------------------------------------------
; Maintains the laid-down ladder: puts it away, or draws it and decides Link's moving direction on it.
; IN:   [0F] = Link's moving direction after the tile collision test; LadderSlot.
; OUT:  [0F] = moving direction, or 0 when Link may not move; X = 0. In the frame the ladder is put
;       away instead: [0F] unchanged and X = the ladder's slot. Ladder's ObjState set to 2 whenever
;       Link is less than $10 pixels from it. Uses [00].
; NOTE: Distance = |Link Y + 3 - ladder Y| for a ladder laid up or down, |Link X - ladder X| for one
;       laid left or right. The ladder stays as long as Link keeps to its line within $10 pixels; it
;       has no timer.
; ----------------------------------------------------------------------------
Ladder_CheckAndDraw:
    ; If the ladder's not in use, return.
    LDX LadderSlot
    BEQ @Exit

    ; If the ladder object's state is 0, go put away the ladder.
    LDA ObjState, X
    BEQ @StashLadder

    ; If the ladder is facing vertically, calculate the vertical distance
    ; between the ladder and Link.
    ; If Link's X no longer matches the ladder's, go put the ladder away.
    LDA ObjDir, X
    AND #$0C
    BEQ @Horizontal
    LDA ObjX
    CMP ObjX, X
    BNE @StashLadder
    LDA ObjY
    CLC
    ADC #$03
    SEC
    SBC ObjY, X
    JMP @CheckDistanceToLadder

@Horizontal:
    ; If instead it's facing horizontally, calculate the horizontal distance
    ; between the ladder and Link.
    ; If Link's Y + 3 no longer matches the ladder's Y, go put the ladder away.
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
    JSR Math_Abs
    STA $00                     ; [00] = absolute distance between Link and the ladder on its axis.
    CMP #$10
    BCC @SetState2
    CMP #$10
    BNE @StashLadder

    ; Distance = $10. If Link's not facing in the same direction as
    ; the ladder, go put away the ladder.
    LDA ObjDir
    CMP ObjDir, X
    BNE @StashLadder

    ; Distance = $10, and Link's facing in the same direction.
    ;
    ; If the ladder's still in initial state 1, then Link's facing the
    ; ladder and has not stepped onto it yet.
    ; Go handle movement on it, but don't change state yet.
    ;
    ; But if ladder state is 2, then Link has been on the ladder and has stepped
    ; completely off its far end. So the ladder is put away: fall thru.
    LDA ObjState, X
    CMP #$01
    BEQ @HandleInput

@StashLadder:
    ; Put away the ladder. Reset ladder slot and destroy the object.
    ; Obj_DestroyMonster keeps X, so X stays the ladder's slot on return.
    LDA #$00
    STA LadderSlot
    JSR Obj_DestroyMonster

@Exit:
    RTS

@SetState2:
    ; Set state 2, because Link is on the ladder.
    LDA #$02
    STA ObjState, X

@HandleInput:
    ; If input direction = 0, go draw and reset moving direction.
    LDA ObjInputDir
    BEQ @DrawLadder

    ; Input direction <> 0. A direction is held.
    ; So, see if we need to override the moving direction.
    ;
    ; A. If the distance <> 0, and Link and the ladder face the same way,
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
    ; then go draw and move in that direction (A = opposite of the ladder's direction).
    ;
    ; Link can always step off the ladder where he came from.
    JSR Dir_GetOpposite
    CMP ObjDir
    BEQ @DrawLadder

    ; D. If opposite of ladder's direction <> down (the ladder is not laid upward),
    ; or input direction <> up alone,
    ; then go reset moving direction, and draw.
    ;
    ; This will catch all cases of Link facing perpendicular to
    ; ladder direction. Also, it will catch Link not moving, unless
    ; ladder direction is up and input direction is up (E).
    CMP #$04
    BNE @StopMoving
    LDA ObjInputDir
    CMP #$08
    BNE @StopMoving

    ; E. The ladder's direction is up and input direction is up.
    ;
    ; An earlier call to check tile collision would have blocked
    ; movement, because the tile under Link's top half is a water tile.
    ; But because Link is squarely on the ladder, we really have
    ; to check the tile above that one.
    ;
    ; Set moving direction to input direction (up) and switch X to
    ; Link's slot for the purpose of checking tile collision below.
    ; Based on that, we'll set moving direction according to walkability.
    JSR Ladder_SetMovingDirAndSelectLink

    ; Check the colliding tile as if Link was 8 pixels up.
    LDA ObjY
    PHA
    SEC
    SBC #$08
    STA ObjY
    JSR Tile_GetAheadOfObj
    PLA
    STA ObjY

    ; If the tile is walkable, loop back to @HandleInput with X = 0. There the slot-indexed
    ; compares read Link's own direction: with distance <> 0 case A passes his facing direction on
    ; as the moving direction; with distance 0 case B passes up on only if he faces up, and
    ; otherwise the moving direction ends up 0.
    ; Else fall thru to reset moving direction and draw the ladder.
    LDA $0F
    LDY ObjCollidedTile
    CPY RoomFirstUnwalkableTile
    BCC @HandleInput

@StopMoving:
    ; Set A to reset moving direction in [0F].
    LDA #$00

@DrawLadder:
    ; Draw the ladder. A (the moving direction to set) is saved across the drawing.
    PHA
    LDX LadderSlot
    JSR Anim_FetchObjPosForSpriteDescriptor
    LDY #$0C                    ; Ladder item slot
    LDA #$00                    ; Sprite palette attribute 0
    JSR Anim_WriteStaticItemSpritesWithAttributes
    PLA

; ----------------------------------------------------------------------------
; Stores A as the moving direction and selects Link's object slot.
; IN:   A = direction.
; OUT:  [0F] = A; X = 0.
; ----------------------------------------------------------------------------
Ladder_SetMovingDirAndSelectLink:
    STA $0F
    LDX #$00
    RTS
