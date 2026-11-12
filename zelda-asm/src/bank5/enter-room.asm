; ============================================================================
; FILE:     bank5/enter-room.asm         BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 1450-1868
; SUBJECTS: (none)
; TOPIC:    Entering a room: Link placement, monster list decode, tile object
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0018 InitMode4                      code
;   L0083 SetFadeCycleAndAdvanceSubmode  code  Params:
;   L0089 InitMode4_Sub3                 code
;   L0094 InitMode4_GoToSub0             code
;   L0108 InitMode_EnterRoom             code  Description:
;   L0395 DrawLinkBetweenRooms           code
;   L0410 SetupTileObjectOW              code
; ============================================================================

InitMode4:
    LDX GameSubmode
    BEQ InitMode_EnterRoom      ; If in submode 0, go perform common tasks to enter a room.
    DEX
    BNE @CheckSub2

    ; Submode 1.
    ; Copies the whole play area to NT 0, one row each frame.
    ;
    ; TODO:
    ; If mode 7 scrolls to a dark room, then it sets
    ; CurRow to zero. Otherwise it would be $FF.
    ; A positive value triggers this submode 1 to transfer
    ; the whole play area.
    ;
    ; But why? As far as I can tell, the NT 0 is already in
    ; a good state reflecting the new room. The only thing
    ; needed is possibly to brighten the room by way of
    ; changing the palette.
    ;
    ; So why does mode 7 trigger this?
    ; Is there another mode that truly needs this behavior?
    LDA CurRow
    BMI :+                      ; If CurRow is negative, then go to the next submode.
    JSR CopyNextRowToTransferBuf
    BCC @Exit                   ; If the last row has been copied,
:
    INC GameSubmode             ; then go to the next submode.

@Exit:
    RTS

@CheckSub2:
    DEX
    BNE InitMode4_Sub3

    ; Submode 2.
    LDY RoomId
    JSR IsDarkRoom_Bank5
    BNE InitMode4_GoToSub0      ; If this room is dark, then nothing else to do. Go to submode 0.

    ; This is a light room.
    ; See if we have to brighten it after leaving a dark room.
    LDA ObjDir
    JSR CalcNextRoomByDir

    ; Subtract RoomId from next room ID that was calculated
    ; to get room ID offset in that direction.
    SEC
    SBC RoomId
    JSR Negate                  ; We really want the opposite offset/direction.
    CLC                         ; Add the current room ID to that to get the previous room ID.
    ADC RoomId
    TAY
    JSR IsDarkRoom_Bank5
    BEQ InitMode4_GoToSub0      ; If previous room was light, then go to submode 0.

    ; The previous room was dark.
    LDA CandleState
    BNE InitMode4_GoToSub0      ; But if it was lit up, then nothing else to do. Go to submode 0.
    LDA #$C0                    ; Start a fade-to-light cycle (reverse of cycle $40).

; Params:
; A: start index of cycle
;
SetFadeCycleAndAdvanceSubmode:
    STA FadeCycle
    INC GameSubmode
:
    RTS

InitMode4_Sub3:
    ; Submode 3.
    JSR AnimateWorldFading
    BNE :-                      ; If not done animating, then return.

InitMode4_GoToSub0:
    LDA #$00
    STA GameSubmode
    STA CandleState
    RTS

; Description:
; Clears intraroom data.
; Set up Link to walk into a room.
; Decodes and lays out objects.
; Switches from initializing the game mode to updating it.
;
; This is called for modes 4, 9, $B, $C.
;
InitMode_EnterRoom:
    JSR DrawSpritesBetweenRooms
    JSR ResetPlayerState

    ; Reset [0300] to [051F].
    LDA #$05
    LDY #$1F
    JSR ClearRam0300UpTo

    ; Reset door trigger info.
    LDA #$00
    STA TriggeredDoorCmd
    STA TriggeredDoorDir

    ; Store level block attribute byte F for convenience.
    LDY RoomId
    LDA LevelBlockAttrsF, Y
    STA LevelBlockAttrsByteF
    JSR ResetInvObjState

    ; There's more than one way to enter a room.
    ;
    ; 1: Out of a cave, dungeon, or cellar
    ;     a. Walking exit
    ;     b. Next to stairs
    ; 2: Room to room
    ;
    ; Determine which one we need.
    LDA UndergroundExitType
    BEQ @Method2                ; If not leaving underground, go handle method 2.
    LDA CurLevel
    BNE :+                      ; If in UW and stepping out of cellar, Link is already where he needs to be. But go handle the rest (method 1-b).

    ; Enter method 1.
    ;
    ; Set up Link's exit from underground.
    LDY RoomId
    LDA LevelBlockAttrsA, Y
    AND #$F0                    ; X coordinate where link comes out of underground
    STA ObjX
    LDA LevelBlockAttrsF, Y
    AND #$07                    ; Square row where Link comes out of underground, starting from square row 1. See the addition below.
    ASL                         ; Multiply by $10 to get a height in pixels.
    ASL
    ASL
    ASL

    ; Add the first Y where Link can come out of underground.
    ; $4D is $50 (the second square row) - 3 (Link's offset above the row).
    ADC #$4D
    STA ObjY

    ; If the player didn't step on a cave entrance to go underground,
    ; then he used the stairs. So, skip starting the exit effect (method 1-b).
    LDY UndergroundEntranceTile
    CPY #$24                    ; $24 is black entrance, and contrasts with $70 to $73 (stairs).
    BNE :+

    ; Enter method 1-a.
    ;
    ; Store the Y coordinate that we determined as the Target Y.
    STA StairsTargetY
    CLC
    ADC #$10
    STA ObjY                    ; But have Link start walking out from $10 pixels below.
    LDA #$08                    ; Play walking sound effect.
    STA EffectRequest
:
    ; This part is the same no matter if the player went
    ; underground by stairs or a cave entrance.
    LDA #$04
    STA ObjDir                  ; Link faces down when coming out of a cave or dungeon.
    LDA #$00
    STA DoorwayDir              ; Reset DoorwayDir, assuming there is no doorway.
    JMP @PlaceObjects           ; Go reset Link's relative position, and work on objects.

@Method2:
    ; Enter method 2.
    ; Room to room.
    ;
    ; Get the direction that the player is entering from.
    LDA ObjDir
    STA DoorwayDir              ; The player entered the doorway in a room. So, set DoorwayDir.
    JSR GetOppositeDir

    ; If the player is entering from a door that was opened, then
    ; set the "close" door command.
    AND CurOpenedDoors
    STA TriggeredDoorDir
    BEQ :+
    LDA #$02
    STA TriggeredDoorCmd
:
    ; If in OW, go reset Link's relative position, and work on objects.
    LDA CurLevel
    BEQ @PlaceObjects

    ; If facing right, then set Link's X to 0 at the left edge.
    ; If direction is left, then set Link's X to $F0 at the right edge.
    LDY #$00
    LDA ObjDir
    AND #$03
    BEQ @ChooseWalkDistance
    AND #$01
    BNE :+
    LDY #$F0
:
    STY ObjX

@ChooseWalkDistance:
    ; Choose a distance to walk depending on the door you
    ; come out of.
    ;
    ; We'll get a 0 or 1 depending on the direction of the door.
    ;
    ; If the attribute of the door is "open", "key", or "key 2";
    ; then you have shorter distance to cover. So use the
    ; previous value as is to use indexes 0 and 1.
    ;
    ; Otherwise, add 2 to use indexes 2 and 3 to walk farther
    ; to get out of the way of the wall or door.
    JSR GetPassedDoorType
    AND #$07
    BEQ :+
    CMP #$05
    BEQ :+
    CMP #$06
    BEQ :+
    INY
    INY
:
    LDA EnteringRoomRelativePositions, Y

@PlaceObjects:
    STA ObjGridOffset
    JSR SetupObjRoomBounds

    ; Set current object slot variable to $B, to be ready for the
    ; first frame of mode 5. So, that it begins updating objects
    ; at slot $B.
    LDX #$0B
    STX CurObjIndex
:
    ; Reset common state of objects $B to 1.
    DEC ObjUninitialized, X
    JSR ResetShoveInfo
    STA ObjState, X
    STA ObjDir, X
    STA ObjStunTimer, X
    INC ObjAnimCounter, X
    INC ObjMetastate, X         ; By default, objects start in metastate 1 (first cloud state).
    LDA #$20                    ; Set the default speed.
    STA ObjQSpeedFrac, X
    DEX
    BNE :-

    ; Put the monster list ID in [02].
    LDY RoomId
    LDA LevelBlockAttrsC, Y
    PHA
    AND #$3F                    ; Low 6 bits of monster list ID
    STA $02
    LDA LevelBlockAttrsD, Y
    ASL                         ; High bit of monster list ID
    BCC :+                      ; If high bit is set in attribute byte D,
    LDA $02                     ; Then set bit 6 of monster list ID.
    CLC
    ADC #$40
    STA $02
:
    PLA

    ; Get index of monster count from high 2 bits of attribute byte C.
    AND #$C0
    CLC
    ROL
    ROL
    ROL
    TAY

    ; Get the object count from level info.
    LDA LevelInfo_FoeCounts, Y

    ; But make the count 1, if  the object list ID >= $32 and < $62.
    ; This includes bosses and other non-recurring objects.
    LDY $02
    CPY #$62
    BCS :+
    CPY #$32
    BCC :+
    LDA #$01
:
    STA $03                     ; Put the count in [03].

    ; Use the room history to modify the number of objects to make.
    LDA CurLevel
    BNE :+
    JSR ModifyObjCountByHistoryOW
    JMP @StoreObjCount
:
    JSR ModifyObjCountByHistoryUW

    ; If in mode 9 (most caves), then
    ; reset object count and object list ID.
    LDA GameMode
    CMP #$09
    BNE @StoreObjCount
    LDA #$00
    STA $02
    STA $03

@StoreObjCount:
    LDA $03                     ; Store the object count.
    STA RoomObjCount

    ; If object count = 0 or object list ID = 0, then
    ; skip instantiating objects.
    BEQ @SkipObjects
    LDA $02
    BEQ @SkipObjects

    ; If object list ID / object template ID >= $62,
    ; then it refers to a list. Go handle it.
    CMP #$62
    BCS @PlaceList

    ; Object list ID / object template ID refers to a
    ; repeated object.
    ;
    ; Set the object type at each element from 1 to Object Count
    ; to object template ID.
    LDX #$00
:
    LDA $02
    STA ObjType+1, X
    INX
    DEC $03
    BNE :-
    JMP @StoreObjTemplate

@PlaceList:
    ; Object list ID / object template ID truly refers to a list of object types.
    ;
    ; Get the index of the list itself by subtracting $62.
    LDA $02
    SEC
    SBC #$62

    ; Put the address of the list in [04:05].
    ASL
    TAY
    LDA ObjListAddrs, Y
    STA $04
    INY
    LDA ObjListAddrs, Y
    STA $05

    ; Copy elements from the list to ObjType up to object count [03].
    LDY #$00
:
    LDA ($04), Y
    STA ObjType+1, Y
    INY
    CPY $03
    BNE :-

@StoreObjTemplate:
    ; Remember the object template type.
    LDA ObjType+1
    STA RoomObjTemplateType

@SkipObjects:
    JSR AssignObjSpawnPositions
    LDA CurLevel
    BNE :+
    JSR SetupTileObjectOW
:
    JSR UpdatePlayerPositionMarker
    LDA #$00
    STA ObjStunTimer
    STA ObjShoveDir
    STA ObjShoveDistance
    LDA #$04
    STA ObjAnimCounter
    JSR InitLinkSpeed
    JSR RunCrossRoomTasksAndBeginUpdateMode_EnterPlayModes

DrawLinkBetweenRooms:
    JSR ResetCurSpriteIndex
    LDA GameMode                ; If mode is not $B nor $C (caves),
    CMP #$0B
    BEQ :+
    CMP #$0C
    BEQ :+
    JSR Link_EndMoveAndAnimateBetweenRooms
:
    LDA UndergroundExitType
    BEQ :+                      ; If coming out of a cave or dungeon, then ...
    JSR PutLinkBehindBackground
:
    RTS

SetupTileObjectOW:
    ; If this is a room with a dock ($3F and $55), then
    ; set the dock object type ($61) in tile object slot ($B).
    LDA RoomId
    CMP #$3F
    BEQ :+
    CMP #$55
    BNE @PlaceTileObj
:
    LDA #$61
    JMP @SetType

@PlaceTileObj:
    ; There's no dock here, but there might be a tile object that
    ; we found while laying out the room. Set the type, X, and Y
    ; to the details we determined.
    LDA RoomTileObjX
    STA ObjX+11
    LDA RoomTileObjY
    STA ObjY+11
    LDA RoomTileObjType

@SetType:
    STA ObjType+11
    JSR ResetRoomTileObjInfo
    STA ObjState+11             ; Activate room tile object.
    RTS
