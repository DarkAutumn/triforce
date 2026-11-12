; ============================================================================
; FILE:     bank4/rock-wall-tree-secrets.asm BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 3565-3858
; SUBJECTS: (none)
; TOPIC:    Push rock/gravestone, bombable rock wall, burnable tree, secret reveal
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0023 RockPushDirections            data  Also used for the gravestone.
;   L0026 UpdateRockOrGravestone        code
;   L0167 RevealSecretTileObj           code  Params:
;   L0173 UpdateRockWall                code
;   L0203 UpdateTree                    code
;   L0236 RevealAndFlagSecretStairsObj  code  Params:
;   L0243 RevealAndFlagSecretTileObj    code  Params:
;   L0251 L10F3E_Exit                   code
;   L0264 CheckTileObjWeaponCollision   code  Params:
;   L0279 SecretQuestNumbers            data
;   L0286 IsQuestSecretMismatch         code  Returns:
; ============================================================================

; Also used for the gravestone.
RockPushDirections:
    .BYTE $08, $04

UpdateRockOrGravestone:
    LDA ObjState, X
    BNE @State1

    ; State 0.
    ;
    ; If this is not a gravestone (but it's a rock), and there is no
    ; bracelet, then return.
    LDA ObjType, X
    CMP #$65
    BEQ @CheckQuest
    LDA InvBracelet
    BEQ @Exit

@CheckQuest:
    ; If there is a mismatch between the current quest and the quest
    ; needed for the secret in this room, then return.
    JSR IsQuestSecretMismatch
    BCS @Exit

    ; The gravestone and rocks can only be pushed vertically.
    ; Return, if X's don't match.
    LDA ObjX
    CMP ObjX, X
    BNE @Exit

    ; If the difference between Link's (Y + 3) and the block's Y
    ; is positive; then go use it along with a flag that indicates that
    ; Link is "down" from the object.
    LDY #$00
    LDA ObjY
    CLC
    ADC #$03
    SEC
    SBC ObjY, X
    BPL @CheckDistance

    ; Else negate it, and indicate Link is "up" from the object.
    INY
    EOR #$FF
    CLC
    ADC #$01

@CheckDistance:
    ; If the distance >= $11, return.
    CMP #$11
    BCS @Exit

    ; If the input direction has no vertical component, then return.
    LDA ObjInputDir
    AND #$0C
    BEQ @Exit

    ; If this vertical input direction does not match the one needed
    ; to push the rock or gravestone, then return.
    CMP RockPushDirections, Y
    BNE @Exit

    ; Set it to the object's direction. Go to the next state.
    STA ObjDir, X
    INC ObjState, X

    ; Store current room ID.
    LDA RoomId
    STA TileObjRoomId, X

    ; Change from a gravestone tile to the gray floor tile.
    ; The gravestone object will be drawn above it.
    INC ReturnToBank4
    LDA #$26                    ; Gray floor tile
    JSR ChangeTileObjTiles

@Exit:
    RTS

@State1:
    ; State 1.
    ;
    ; Move in the object's direction.
    LDA ObjDir, X
    STA $0F
    JSR MoveObject

    ; Draw the sprites one pixel above the object's location.
    JSR Anim_FetchObjPosForSpriteDescriptor
    DEC $01
    JSR DrawObjectNotMirrored

    ; If the object's grid offset (distance moved) <> $10 nor -$10,
    ; then return.
    LDA ObjGridOffset, X
    CMP #$10
    BEQ :+
    CMP #$F0
    BNE @Exit
:
    ; Flag the room visited.
    LDA RoomId
    PHA
    LDA TileObjRoomId, X
    STA RoomId
    JSR MarkRoomVisited
    PLA
    STA RoomId

    ; If object is a rock ($62), go use a rock tile ($C8).
    LDA #$C8
    LDY ObjType, X
    CPY #$62
    BEQ @ReplaceTiles

    ; If the object is a gravestone ($65), go use a gravestone tile ($BC).
    LDA #$BC
    CPY #$65
    BEQ @ReplaceTiles

    ; Else use an armos tile ($C0). Dead code?
    ; Begin unverified code 10EC9
    LDA #$C0
    ; End unverified code

@ReplaceTiles:
    ; Change the tiles at the end location.
    INC ReturnToBank4
    JSR ChangeTileObjTiles

    ; Store the stairs tile in [00] for use in revealing the secret.
    LDA #$70
    STA $00

    ; Get the position of the stairs in this room, and set it for the object.
    LDY TileObjRoomId, X
    JSR GetShortcutOrItemXYForRoom
    STA ObjX, X
    STY ObjY, X
    LDA $00

; Params:
; A: the first tile of tile object
; X: object index
;
RevealSecretTileObj:
    INC ReturnToBank4
    JSR ChangeTileObjTiles
    JSR DestroyMonster_Bank4
    JMP PlaySecretFoundTune

UpdateRockWall:
    ; If the secret in this room does not apply to the current quest,
    ; then return.
    JSR IsQuestSecretMismatch
    BCS @Exit

    ; If there is no detonating bomb (state $13) in slots $10 and $11,
    ; then return.
    LDY #$10
    LDA a:ObjState, Y
    CMP #$13
    BEQ @FoundBomb
    INY                         ; Point to slot $11.
    LDA a:ObjState, Y
    CMP #$13
    BNE @Exit

@FoundBomb:
    ; Store the bomb's object slot in [00].
    STY $00

    ; If the tile object and weapon collide, then reveal the secret.
    JSR CheckTileObjWeaponCollision
    BEQ @Exit
    LDA #$24                    ; Black tile
    JSR RevealAndFlagSecretTileObj

@Exit:
    RTS

UpdateTree:
    ; If the secret in this room does not apply to the current quest,
    ; then return.
    JSR IsQuestSecretMismatch
    BCS L10F3E_Exit

    ; If there is no standing fire (state $22) in slots $10 and $11,
    ; then return.
    LDY #$10
    LDA a:ObjState, Y
    CMP #$22
    BEQ @FoundFire
    INY
    LDA a:ObjState, Y
    CMP #$22
    BNE L10F3E_Exit

@FoundFire:
    ; Store the fire's object slot in [00].
    STY $00

    ; If the standing fire's timer >= 2, return.
    LDA a:ObjTimer, Y
    CMP #$02
    BCS L10F3E_Exit

    ; If the tile object and weapon collide, then reveal the secret.
    JSR CheckTileObjWeaponCollision
    BEQ L10F3E_Exit

; Params:
; X: object index of the tile object
;
RevealAndFlagSecretStairsObj:
    LDA #$70                    ; First tile of stairs

; Params:
; A: the first tile of tile object
; X: object index
;
RevealAndFlagSecretTileObj:
    JSR RevealSecretTileObj

    ; Flag the secret found in this room.
    JSR GetRoomFlags
    ORA #$80                    ; Secret found/taken
    STA ($00), Y

L10F3E_Exit:
    RTS

; Params:
; X: object index
; Y: weapon object index
; [00]: weapon object index
;
; Returns:
; A: 1 if objects collide
;
; Put the midpoint coordinate of the weapon in [04] and [05].
;
CheckTileObjWeaponCollision:
    LDA a:ObjX, Y
    CLC
    ADC #$08
    STA $04
    LDA a:ObjY, Y
    CLC
    ADC #$08
    STA $05

    ; Put the tile object's midpoint coordinates in [02] and [03].
    JSR GetObjectMiddle
    LDA #$10                    ; 10 pixel collision threshold
    JMP DoObjectsCollide

SecretQuestNumbers:
    .BYTE $00, $00, $01

; Returns:
; C: 1 if there is a mismatch between current quest
;    and quest number for the secret in the room
;
IsQuestSecretMismatch:
    ; The quest secret is in the high two bits.
    LDA LevelBlockAttrsByteF
    LSR
    LSR
    LSR
    LSR
    LSR
    LSR

    ; If it's 0, then return C=0.
    BEQ @ReturnFalse

    ; Look up a quest number (0 or 1) indexed by quest secret (0 or 1).
    TAY
    LDA SecretQuestNumbers, Y

    ; If the save slot's quest number matches the room's
    ; secret quest number, then return C=0.
    LDY CurSaveSlot
    CMP QuestNumbers, Y
    BEQ @ReturnFalse

    ; Else return C=1. There is a mismatch in quest numbers.
    SEC
    RTS

@ReturnFalse:
    CLC
    RTS
