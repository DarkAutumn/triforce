; ============================================================================
; FILE:     bank4/rock-wall-tree-secrets.asm BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 3565-3858
; SUBJECTS: Collision, Rock, RockWall, Secret, Tree
; TOPIC:    Overworld tile objects that hide a secret: the pushable rock ($62) and gravestone ($65), the
;           bombable rock wall ($63), the burnable tree ($64); revealing and flagging the secret; the
;           test of which quest a room's secret belongs to.
; SEE ALSO: bank5/layout-room-ow.asm (RoomLayout_CheckTileObj: which square makes which tile object);
;           bank5/ow-shortcut-map-square.asm (Secret_RestoreStairsOW: redraws stairs from room flag $20);
;           bank7/tile-obj-tiles.asm (Tile_ChangeTileObjTiles); bank1/bomb-candle-wield.asm
;           (Room_GetShortcutOrItemXYForRoomId); bank7/bomb.asm and bank7/fire.asm (weapon states $13, $22);
;           bank7/flute-secret-pond.asm (FluteSecret_RevealPondStairs uses Secret_RevealAndFlagStairsObj);
;           bank4/ghini-armos.asm (secrets under Armos statues).
; KEY FACTS:
;   - A rock ($62) can be pushed only with the power bracelet (InvBracelet); a gravestone ($65) needs
;     nothing (Rock_Update).
;   - Push condition: Link's X equals the object's X, |Link's Y + 3 - object's Y| < $11, and the vertical
;     bits of Link's input direction are exactly up (Link below) or down (Link above). Rocks and
;     gravestones never move sideways (Rock_Update, Rock_PushDirs).
;   - The pushed object moves $10 pixels with Obj_MoveByQSpeed at its slot's q-speed ($20 = 0.5 px/frame from
;     the room-entry reset, so 32 frames), then stairs (tile $70) appear at the room's shortcut
;     position, not where the object stood (Rock_Update, Room_EnterCommon).
;   - A finished push sets room flag $20 (RoomFlags_MarkVisited), not $80 (Rock_Update).
;   - A rock wall ($63) opens (black tile $24) when a bomb in slot $10 or $11 is in state $13 and the
;     two midpoints are less than $10 pixels apart on both axes (RockWall_Update).
;   - A tree ($64) burns to stairs when a fire in slot $10 or $11 is standing (state $22) with its
;     timer below 2, i.e. as the fire is about to go out, within the same $10 pixels. Candle fire and
;     the book's fire are not told apart (Tree_Update).
;   - Rock wall, tree and flute stairs set room flag $80 "secret found" (Secret_RevealAndFlagTileObj).
;   - Bits 7-6 of RoomAttrsF say whose secret it is: 0 both quests, 1 Q1 only, 2 Q2 only. In the other
;     quest the tile object does nothing (Secret_IsForOtherQuest, Secret_QuestNumbers).
;   - Object types $66 and $67 are routed to the rock and rock wall updates by the update table; a
;     type $66 leaves Armos tile $C0 after a push. The room layout code notes that no overworld layout
;     produces either type (Rock_Update, RoomLayout_CheckTileObj).
; CONTENTS: (generated; L = line in this file)
;   L0051 Rock_PushDirs                  data  TABLE: input direction that pushes a rock or gravestone; index = 0 Link below it, 1 Link above it;
;   L0062 Rock_Update                    code  Updates a pushable rock ($62) or gravestone ($65): waits for Link's push, slides, reveals stairs.
;   L0202 Secret_RevealTileObj           code  Puts a secret's tiles at the tile object's position, removes the object and plays the secret tune.
;   L0214 RockWall_Update                code  Updates a bombable rock wall ($63): opens a cave mouth when a bomb blast touches it.
;   L0249 Tree_Update                    code  Updates a burnable tree ($64): turns into stairs when one of Link's fires burns out on it.
;   L0283 Secret_RevealAndFlagStairsObj  code  Reveals stairs (tile $70) at the tile object's position and flags the room's secret found.
;   L0287 Secret_RevealAndFlagTileObj    code  Second entry: reveals the square starting with tile A, then sets room flag $80 (secret found).
;   L0296 Secret_RevealAndFlag_Return    code  Shared exit of Tree_Update and the reveal routines above.
;   L0304 Collision_CheckTileObjWeapon   code  Tests whether a weapon object (bomb or fire) touches a tile object.
;   L0322 Secret_QuestNumbers            data  TABLE: quest that a room's secret belongs to; index = bits 7-6 of RoomAttrsF (1 or 2; 0 is never
;   L0331 Secret_IsForOtherQuest         code  Tests whether the room's secret belongs to the quest that is not being played.
; ============================================================================

; TABLE: input direction that pushes a rock or gravestone; index = 0 Link below it, 1 Link above it;
;        value = direction bit ($08 up, $04 down).
Rock_PushDirs:
    .BYTE $08, $04

; ----------------------------------------------------------------------------
; Updates a pushable rock ($62) or gravestone ($65): waits for Link's push, slides, reveals stairs.
; IN:   X = object slot.
; NOTE: State 0 waits; state 1 slides $10 pixels up or down, then puts stairs at the room's shortcut
;       position, sets room flag $20 and removes the object. The rock needs the power bracelet.
; NOTE: The slide takes 32 frames at the q-speed $20 that room entry gives every slot.
; NOTE: Also the update of object type $66, which needs the bracelet too and leaves Armos tile $C0.
; ----------------------------------------------------------------------------
Rock_Update:
    LDA ObjState, X
    BNE @State1

    ; State 0: waiting to be pushed.
    ;
    ; A gravestone ($65) needs no item. Any other type (the rock) needs the
    ; power bracelet: without it, return.
    LDA ObjType, X
    CMP #$65
    BEQ @CheckQuest
    LDA InvBracelet
    BEQ @Exit

@CheckQuest:
    ; If this room's secret belongs to the other quest, then return.
    JSR Secret_IsForOtherQuest
    BCS @Exit

    ; Rocks and gravestones are pushed only vertically.
    ; If Link's X is not exactly the object's X, then return.
    LDA ObjX
    CMP ObjX, X
    BNE @Exit

    ; Take (Link's Y + 3) - (object's Y). If it is not negative, Link is below the
    ; object: keep it as the distance, with index Y = 0 (push direction up).
    LDY #$00
    LDA ObjY
    CLC
    ADC #$03
    SEC
    SBC ObjY, X
    BPL @CheckDistance

    ; Else negate it; Link is above the object: index Y = 1 (push direction down).
    INY
    EOR #$FF
    CLC
    ADC #$01

@CheckDistance:
    ; If the distance >= $11, then return. (Link standing against the object gives $10.)
    CMP #$11
    BCS @Exit

    ; If Link's input direction has no vertical component, then return.
    LDA ObjInputDir
    AND #$0C
    BEQ @Exit

    ; If the vertical bits are not exactly the direction that pushes from Link's side
    ; (up from below, down from above), then return.
    CMP Rock_PushDirs, Y
    BNE @Exit

    ; Make it the object's direction, and go to state 1.
    STA ObjDir, X
    INC ObjState, X

    ; Remember the room where the push began.
    LDA RoomId
    STA Rock_ObjRoomId, X

    ; Replace the object's square with floor tile $26. The moving rock or
    ; gravestone is drawn as sprites from now on.
    INC ReturnToBank4
    LDA #$26                    ; Gray floor tile
    JSR Tile_ChangeTileObjTiles

@Exit:
    RTS

@State1:
    ; State 1: sliding.
    ;
    ; Move in the object's direction at the slot's q-speed. Obj_MoveByQSpeed stops
    ; an object other than Link after $10 pixels.
    LDA ObjDir, X
    STA $0F
    JSR Obj_MoveByQSpeed

    ; Draw the sprites one pixel above the object's location.
    JSR Anim_FetchObjPosForSpriteDescriptor
    DEC $01
    JSR Anim_DrawObjectNotMirrored

    ; If the grid offset (signed distance moved) is neither $10 nor -$10, then return.
    LDA ObjGridOffset, X
    CMP #$10
    BEQ :+
    CMP #$F0
    BNE @Exit
:
    ; The push is complete. Set room flag $20 for the room where the push began
    ; (RoomId is swapped in for the call).
    LDA RoomId
    PHA
    LDA Rock_ObjRoomId, X
    STA RoomId
    JSR RoomFlags_MarkVisited
    PLA
    STA RoomId

    ; Choose the tile for the square where the object came to rest: rock ($62) = $C8.
    LDA #$C8
    LDY ObjType, X
    CPY #$62
    BEQ @ReplaceTiles

    ; Gravestone ($65) = $BC.
    LDA #$BC
    CPY #$65
    BEQ @ReplaceTiles

    ; Any other type (only $66 can get here) = Armos tile $C0.
    LDA #$C0

@ReplaceTiles:
    ; Change the tiles at the resting place.
    INC ReturnToBank4
    JSR Tile_ChangeTileObjTiles

    ; [00] = stairs tile $70, the secret to reveal.
    LDA #$70
    STA $00

    ; Move the object to the room's shortcut position (from bits 5-4 of the room's
    ; attribute byte F), where the stairs will appear.
    LDY Rock_ObjRoomId, X
    JSR Room_GetShortcutOrItemXYForRoomId
    STA ObjX, X
    STY ObjY, X
    LDA $00

; ----------------------------------------------------------------------------
; Puts a secret's tiles at the tile object's position, removes the object and plays the secret tune.
; IN:   A = first tile of the square to draw ($70 stairs, $24 black cave mouth); X = object slot.
; NOTE: Does not touch the room flags; Secret_RevealAndFlagTileObj does.
; ----------------------------------------------------------------------------
Secret_RevealTileObj:
    INC ReturnToBank4
    JSR Tile_ChangeTileObjTiles
    JSR Obj_DestroyMonster_Bank4
    JMP Sfx_PlaySecretFoundTune

; ----------------------------------------------------------------------------
; Updates a bombable rock wall ($63): opens a cave mouth when a bomb blast touches it.
; IN:   X = object slot.
; NOTE: Tests bombs in slots $10 and $11 for state $13 (the blast, $18 frames) and a midpoint distance
;       below $10 pixels on both axes. Does nothing in the quest the secret does not belong to.
; ----------------------------------------------------------------------------
RockWall_Update:
    ; If this room's secret belongs to the other quest, then return.
    JSR Secret_IsForOtherQuest
    BCS @Exit

    ; Look for a bomb in its blast state ($13) in slot $10, then $11.
    ; If there is none, then return.
    LDY #$10
    LDA a:ObjState, Y
    CMP #$13
    BEQ @FoundBomb
    INY                         ; Point to slot $11.
    LDA a:ObjState, Y
    CMP #$13
    BNE @Exit

@FoundBomb:
    ; [00] = the bomb's object slot.
    STY $00

    ; If the bomb touches the rock wall, reveal the secret and flag it found.
    JSR Collision_CheckTileObjWeapon
    BEQ @Exit
    LDA #$24                    ; Black tile: the cave mouth.
    JSR Secret_RevealAndFlagTileObj

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Updates a burnable tree ($64): turns into stairs when one of Link's fires burns out on it.
; IN:   X = object slot.
; NOTE: Tests fires in slots $10 and $11 for state $22 (standing) with ObjTimer < 2, and a midpoint
;       distance below $10 pixels on both axes. Does nothing in the quest the secret does not belong to.
; ----------------------------------------------------------------------------
Tree_Update:
    ; If this room's secret belongs to the other quest, then return.
    JSR Secret_IsForOtherQuest
    BCS Secret_RevealAndFlag_Return

    ; Look for a standing fire (state $22) in slot $10, then $11.
    ; If there is none, then return.
    LDY #$10
    LDA a:ObjState, Y
    CMP #$22
    BEQ @FoundFire
    INY
    LDA a:ObjState, Y
    CMP #$22
    BNE Secret_RevealAndFlag_Return

@FoundFire:
    ; [00] = the fire's object slot.
    STY $00

    ; If the fire's timer >= 2, then return: the tree burns only as the fire is about to go out.
    LDA a:ObjTimer, Y
    CMP #$02
    BCS Secret_RevealAndFlag_Return

    ; If the fire does not touch the tree, then return. Else fall through to reveal stairs.
    JSR Collision_CheckTileObjWeapon
    BEQ Secret_RevealAndFlag_Return

; ----------------------------------------------------------------------------
; Reveals stairs (tile $70) at the tile object's position and flags the room's secret found.
; IN:   X = object slot of the tile object.
; NOTE: Also used for the stairs that the flute reveals in a pond.
; ----------------------------------------------------------------------------
Secret_RevealAndFlagStairsObj:
    LDA #$70                    ; First tile of stairs

; Second entry: reveals the square starting with tile A, then sets room flag $80 (secret found).
Secret_RevealAndFlagTileObj:
    JSR Secret_RevealTileObj

    ; Set flag $80 in the current room's flags: the secret has been found.
    JSR RoomFlags_Get
    ORA #$80
    STA ($00), Y

; Shared exit of Tree_Update and the reveal routines above.
Secret_RevealAndFlag_Return:
    RTS

; ----------------------------------------------------------------------------
; Tests whether a weapon object (bomb or fire) touches a tile object.
; IN:   X = tile object's slot; Y = weapon's object slot; [00] = weapon's object slot.
; OUT:  A = [06] = 1 (Z clear) if the midpoints are less than $10 pixels apart in both X and Y.
; ----------------------------------------------------------------------------
Collision_CheckTileObjWeapon:
    ; [04], [05] = the weapon's midpoint: its position + 8 on each axis.
    LDA a:ObjX, Y
    CLC
    ADC #$08
    STA $04
    LDA a:ObjY, Y
    CLC
    ADC #$08
    STA $05

    ; [02], [03] = the tile object's midpoint.
    JSR Obj_GetMiddle
    LDA #$10                    ; Collision threshold: $10 pixels.
    JMP Collision_TestPoints

; TABLE: quest that a room's secret belongs to; index = bits 7-6 of RoomAttrsF (1 or 2; 0 is never
;        looked up); value = quest number as in QuestNumbers (0 = Q1, 1 = Q2).
Secret_QuestNumbers:
    .BYTE $00, $00, $01         ; (unused), 1 = Q1 only, 2 = Q2 only

; ----------------------------------------------------------------------------
; Tests whether the room's secret belongs to the quest that is not being played.
; OUT:  C = 1 if the secret is for the other quest (the caller then does nothing); C = 0 if it applies.
; NOTE: Bits 7-6 of RoomAttrsF: 0 = both quests, 1 = Q1 only, 2 = Q2 only. Value 3 would index past
;       the table.
; ----------------------------------------------------------------------------
Secret_IsForOtherQuest:
    ; Shift bits 7-6 of RoomAttrsF down to bits 1-0.
    LDA RoomAttrsF
    LSR
    LSR
    LSR
    LSR
    LSR
    LSR

    ; If it's 0, the secret is in both quests: return C=0.
    BEQ @ReturnFalse

    ; Look up the quest number (0 or 1) that the value (1 or 2) stands for.
    TAY
    LDA Secret_QuestNumbers, Y

    ; If it equals the quest number of the save slot in use, then return C=0.
    LDY CurSaveSlot
    CMP QuestNumbers, Y
    BEQ @ReturnFalse

    ; Else return C=1: the secret belongs to the other quest.
    SEC
    RTS

@ReturnFalse:
    CLC
    RTS
