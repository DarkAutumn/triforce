; ============================================================================
; FILE:     bank5/mode3-mode6-init.asm   BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 6373-6586
; SUBJECTS: (none)
; TOPIC:    Unfurl submodes, object room bounds, leave room
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0028 InitMode3_Sub2                          code
;   L0034 InitMode3_Sub3_TransferTopHalfAttrs     code
;   L0040 InitMode3_Sub4_TransferBottomHalfAttrs  code
;   L0045 InitMode3_Sub5                          code
;   L0048 SelectTransferBuf                       code
;   L0051 L1701A_Exit                             code
;   L0055 InitMode3_Sub6                          code
;   L0064 InitMode3_Sub7                          code
;   L0071 InitMode3_Sub8                          code
;   L0101 ObjectRoomBoundsOW                      data  Description:
;   L0104 ObjectRoomBoundsUW                      data
;   L0107 SetupObjRoomBounds                      code
;   L0126 LeavingRoomRelativePositions            data
;   L0129 InitMode6                               code
;   L0179 ResetInvObjState                        code
;   L0199 GetPassedDoorType                       code  Returns door type in direction Link's facing (mode 6),
;   L0228 CopyPlayAreaAttrsHalfToDynTransferBuf   code  Params:
; ============================================================================

InitMode3_Sub2:
    LDA RoomId
    JSR FillPlayAreaAttrs
    LDA #$18
    BNE SelectTransferBuf       ; Go cue a transfer of palettes, and go to next submode.

InitMode3_Sub3_TransferTopHalfAttrs:
    LDA #$D0                    ; Low byte of destination PPU address for NT attributes.
    LDY #$17                    ; Offset of the end of first half of play area NT attributes.
:
    JMP CueTransferPlayAreaAttrsHalfAndAdvanceSubmodeNT0

InitMode3_Sub4_TransferBottomHalfAttrs:
    LDA #$E8                    ; Low byte of destination PPU address for NT attributes.
    LDY #$2F                    ; Offset of the end of second half of play area NT attributes.
    BNE :-

InitMode3_Sub5:
    LDA #$0E                    ; Cue transfer of static elements of status bar.

SelectTransferBuf:
    STA TileBufSelector

L1701A_Exit:
    INC GameSubmode
    RTS

InitMode3_Sub6:
    LDA CurLevel
    BEQ :+                      ; If in OW, skip checking the level's map.
    JSR HasMap
    BEQ L1701A_Exit             ; If we don't have the map, go to the next submode.
:
    LDA #$44
    BNE SelectTransferBuf       ; Cue transfer of map in status bar, and go to next submode.

InitMode3_Sub7:
    LDA LevelInfo_LevelNumber
    BEQ L1701A_Exit             ; If level is OW, then go to next submode.
    STA LevelNumberTransferBuf+9    ; Patch the level number character in "LEVEL-X" transfer buf.
    LDA #$0C                    ; Cue transfer of "LEVEL-X" text and go to next submode.
    BNE SelectTransferBuf

InitMode3_Sub8:
    JSR LayOutRoom

    ; Set up columns numbers for curtain effect.
    ;
    ; Decrease from column $F ($10-1).
    LDY #$10
    STY ObjX+12
    INY                         ; Increase from column $10 ($11-1).
    STY ObjX+13
    LDA #$00                    ; TODO: ?
    STA $17
    LDA #$08                    ; Make Link face up by default.
    STA ObjDir
    LDA #$78                    ; Put Link in the middle horizontally by default.
    STA ObjX
    LDA LevelInfo_StartY        ; Put Link at StartY from level info by default.
    STA ObjY
    JMP BeginUpdateMode

; Description:
; Two sets of 5 elements:
; * left bound for objects
; * right bound for objects
; * up bound for objects
; * down bound for objects
; * first unwalkable tile
;
; The first set is for OW. The second is for UW.
;
ObjectRoomBoundsOW:
    .BYTE $11, $E0, $4E, $CD, $89

ObjectRoomBoundsUW:
    .BYTE $21, $D0, $5E, $BD, $78

SetupObjRoomBounds:
    LDY #$05                    ; Offset of second set of bounds.
    LDA CurLevel
    BNE :+                      ; If in UW, use second of bounds, and go copy them.

    ; Reset DoorwayDir and use first set of bounds.
    LDY #$00                    ; Offset of first set of bounds.
    STY DoorwayDir
:
    LDX #$00
:
    LDA ObjectRoomBoundsOW, Y
    STA RoomBoundLeft, X
    INY
    INX
    CPX #$05
    BNE :-
    RTS

LeavingRoomRelativePositions:
    .BYTE $28, $D8, $00

InitMode6:
    JSR ResetPlayerState
    JSR DrawSpritesBetweenRooms
    LDA CurLevel
    BEQ :+                      ; If in UW,
    JSR WriteBlankPrioritySprites    ; then clear the sprites that go above all the rest.
:
    JSR Link_EndMoveAndAnimateBetweenRooms
    JSR SaveKillCount
    LDA CurLevel
    BEQ @SetWalkDistance0
    JSR GetPassedDoorType       ; Get door type in direction we're facing

    ; If the door type is "false wall",
    ; then play the "found secret" tune.
    PHA
    AND #$07
    CMP #$02
    BNE :+
    LDA #$04                    ; Play "found secret" tune.
    STA Tune1Request
:
    PLA

    ; If the player is at an "open", "key", or "shutter" door,
    ; then the player walked all the way to the edge of the play
    ; area. So, use index 2 to set Link's relative position to 0.
    ;
    ; Otherwise, the player is at wall level. Link has to walk to the
    ; edge of the play area. So, set a relative position that
    ; reflects that distance: $28 or -$28.
    ;
    ; If walking in a decreasing direction (left or up), then
    ; Link's relative position is $28 and has to decrease to 0.
    ;
    ; If walking in an increasing direction (right or down), then
    ; Link's relative position is -$28 and has to increase to 0.
    AND #$07
    CMP #$02
    BCC @SetWalkDistance0
    CMP #$05
    BCC :+

@SetWalkDistance0:
    LDY #$02
:
    LDA LeavingRoomRelativePositions, Y
    STA ObjGridOffset
    JSR RunCrossRoomTasksAndBeginUpdateMode

ResetInvObjState:
    ; Reset LadderSlot. No ladder is active.
    LDA #$00
    STA LadderSlot

    ; Reset ObjState of all weapons.
    LDY #$05
:
    STA a:ObjState+13, Y
    DEY
    BPL :-
    RTS

; Returns door type in direction Link's facing (mode 6),
; or opposite direction (other modes).
;
; Returns:
; A: door type
; Y: whether Link is facing an increasing direction (right or down)
;
GetPassedDoorType:
    ; If Link's direction is right or down, then 1 will be returned.
    ; Else 0.
    LDY #$00
    LDA ObjDir
    AND #$05
    BEQ :+
    INY
:
    STY $0F

    ; If in mode 6, use Link's direction to get a door type.
    ; Else use the opposite direction.
    LDA ObjDir
    LDY GameMode
    CPY #$06
    BEQ :+
    JSR GetOppositeDir
:
    STA $02
    JSR FindDoorTypeByDoorBit
    LDY $0F
    RTS

; Params:
; X: high PPU address
; A: low PPU address
; Y: end offset in PlayAreaAttrs to copy from
;
CopyPlayAreaAttrsHalfToDynTransferBuf:
    STX DynTileBuf
    STA DynTileBuf+1
    LDX #$18
    STX DynTileBuf+2
    LDA #$FF
    STA DynTileBuf+3, X
:
    LDA PlayAreaAttrs, Y
    STA DynTileBuf+2, X
    DEY
    DEX
    BNE :-
    RTS
