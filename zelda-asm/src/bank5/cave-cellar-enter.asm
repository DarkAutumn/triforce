; ============================================================================
; FILE:     bank5/cave-cellar-enter.asm  BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 6588-6803
; SUBJECTS: (none)
; TOPIC:    Mode A; cave/cellar walk-in; heart fill
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0034 InitModeA                                       code
;   L0038 InitModeA_JumpTable                             data
;   L0051 InitModeSubroom_Sub0                            code
;   L0060 DrawSpritesBetweenRoomsAndAdvanceSubmode        code
;   L0064 InitMode9_TransferAttrs                         code
;   L0067 L1712E_SelectTransferBufAndAdvanceSubmode       code
;   L0070 InitModeSubroom_AdvanceSubmode                  code
;   L0074 InitMode9_FadeToDark                            code
;   L0081 InitModeSubroom_AnimateFade                     code
;   L0086 InitMode9_FadeToLight                           code
;   L0090 InitModeA_Sub2                                  code
;   L0094 InitModeA_Sub8                                  code
;   L0098 InitModeB_Sub1                                  code
;   L0103 InitModeA_Sub1                                  code
;   L0111 InitModeA_SubA_GoToMode4                        code
;   L0119 InitModeA_Sub6_FillTileAttrsAndTransferTopHalf  code
;   L0123 InitModeB_Sub5_FillTileAttrsAndTransferTopHalf  code
;   L0129 InitModeAOrB_TransferBottomHalfAttrs            code
;   L0141 InitMode_WalkCave                               code
;   L0156 CellarLadderXs                                  data
;   L0159 InitMode9_EnterCellar                           code
;   L0199 InitMode9_WalkCellar                            code
;   L0217 World_FillHearts                                code
; ============================================================================

InitModeA:
    LDA GameSubmode
    JSR TableJump

InitModeA_JumpTable:
    .ADDR InitModeSubroom_Sub0
    .ADDR InitModeA_Sub1
    .ADDR InitModeA_Sub2
    .ADDR InitModeSubroom_AnimateFade
    .ADDR LayoutRoom_SubmodeTask
    .ADDR CopyNextRowToTransferBufAndAdvanceSubmodeWhenDone
    .ADDR InitModeA_Sub6_FillTileAttrsAndTransferTopHalf
    .ADDR InitModeAOrB_TransferBottomHalfAttrs
    .ADDR InitModeA_Sub8
    .ADDR InitModeSubroom_AnimateFade
    .ADDR InitModeA_SubA_GoToMode4

InitModeSubroom_Sub0:
    LDA #$00
    STA CurRow
    STA CurOpenedDoors
    LDA CurLevel
    BNE DrawSpritesBetweenRoomsAndAdvanceSubmode
    JSR DrawSpritesBetweenRoomsAndAdvanceSubmode
    JMP WriteAndEnableSprite0

DrawSpritesBetweenRoomsAndAdvanceSubmode:
    INC GameSubmode
    JMP DrawSpritesBetweenRooms

InitMode9_TransferAttrs:
    LDA #$26                    ; Cellar NT attributes

L1712E_SelectTransferBufAndAdvanceSubmode:
    STA TileBufSelector

InitModeSubroom_AdvanceSubmode:
    INC GameSubmode
    RTS

InitMode9_FadeToDark:
    LDA #$00                    ; Light level -> dark cellar cycle
:
    LDY CurLevel
    BEQ InitModeSubroom_AdvanceSubmode
    JSR SetFadeCycleAndAdvanceSubmode

InitModeSubroom_AnimateFade:
    LDY CurLevel
    BEQ InitModeSubroom_AdvanceSubmode
    JMP UpdateMode11Death_Sub8_AnimateFade

InitMode9_FadeToLight:
    LDA #$A0                    ; Dark cellar -> light cellar cycle
    BNE :-

InitModeA_Sub2:
    LDA #$20                    ; Light cellar -> dark cellar cycle
    BNE :-

InitModeA_Sub8:
    LDA #$80                    ; Dark cellar -> light level cycle
    BNE :-

InitModeB_Sub1:
    ; Transfer the cave BG palette rows.
    LDA #$3E
    BNE L1712E_SelectTransferBufAndAdvanceSubmode

InitModeA_Sub1:
    LDA CurLevel
    BNE InitModeSubroom_AdvanceSubmode

    ; Transfer the OW palette again, because it was changed
    ; for a cave.
    JMP PatchAndCueLevelPalettesTransferAndAdvanceSubmode

InitModeA_SubA_GoToMode4:
    JSR ResetInvObjState
    LDA #$00
    STA GameSubmode
    LDA #$04
    STA GameMode
    RTS

InitModeA_Sub6_FillTileAttrsAndTransferTopHalf:
    LDA RoomId
    JMP :+

InitModeB_Sub5_FillTileAttrsAndTransferTopHalf:
    LDA #$44                    ; An OW room that has the same NT attributes as a cave.
:
    JSR FillPlayAreaAttrs
    JMP InitMode3_Sub3_TransferTopHalfAttrs

InitModeAOrB_TransferBottomHalfAttrs:
    JSR InitMode3_Sub4_TransferBottomHalfAttrs
    JMP @DisableSprite0Check
    ; Begin unverified code 17179
    INC GameSubmode
    ; End unverified code

@DisableSprite0Check:
    LDA #$00
    STA IsSprite0CheckActive
    RTS

InitMode_WalkCave:
    ; If reached the end of the walk, then go start updating.
    LDA ObjGridOffset
    BEQ :+

    ; Move and draw facing up.
    LDA ObjDir
    STA ObjInputDir
    STA $0F
    LDX #$00
    JSR MoveObject
    JMP Link_EndMoveAndAnimateInRoom
:
    JMP RunCrossRoomTasksAndBeginUpdateMode

CellarLadderXs:
    .BYTE $30, $C0

InitMode9_EnterCellar:
    LDA GameSubmode
    PHA                         ; Save the submode.
    JSR InitMode_EnterRoom
    JSR ResetInvObjState
    PLA                         ; Restore the submode.
    STA GameSubmode

    ; Each cellar can have two destination rooms: A and B.
    ; Tunnels use both. Treasure rooms only use room A.
    ;
    ; If the room that Link came from is the room A of this cellar,
    ; then look up the X coordinate of the ladder on the left at index 0.
    ; Else use index 1 to get the X coordinate on the right.
    LDY RoomId
    LDX #$00
    LDA CellarSourceRoomId
    CMP LevelBlockAttrsA, Y
    BEQ :+
    INX
:
    LDA CellarLadderXs, X
    STA ObjX

    ; Link goes at Y=$41, and facing down.
    LDA #$41
    STA ObjY
    LDA #$04
    STA ObjDir

    ; Set a grid offset appropriate for the distance to travel:
    ;   ($5D - $41) = $1C = ($100 - $E4)
    LDA #$E4
    STA ObjGridOffset
    LDA #$00
    STA IsUpdatingMode
    STA DoorwayDir
    INC GameSubmode
    RTS

InitMode9_WalkCellar:
    ; Set Link's input direction to the facing direction, and
    ; update the player object; so that it walks down the stairs
    ; of the cellar. Stop when it reaches Y coordinate $5D.
    LDA ObjDir
    STA ObjInputDir
    JSR UpdatePlayer
    LDA ObjY
    CMP #$5D
    BNE :+
    LDA #$00                    ; Reset player state.
    STA ObjState
    LDA #$01                    ; Remember that this is a cellar.
    STA UndergroundExitType
    STA IsUpdatingMode          ; Start updating the mode.
:
    RTS

World_FillHearts:
    LDA World_IsFillingHearts
    BEQ @Exit                   ; If not filling hearts, then return.
    LDA #$10                    ; Play the "heart taken" tune.
    STA Tune0Request
    LDA HeartPartial
    CMP #$F8
    BCS @CompleteHeart          ; If HeartPartial >= $F8, go complete a heart.
    CLC                         ; else add 6.
    ADC #$06
    STA HeartPartial
    RTS

@CompleteHeart:
    LDA #$00                    ; Set HeartPartial to zero for the next heart.
    STA HeartPartial
    JSR CompareHeartsToContainers
    BNE @IncHearts              ; If hearts <> heart containers, go increase hearts.

    ; They're equal, so make HeartPartial full by
    ; decreasing from 0 to $FF.
    DEC HeartPartial
    LDA #$00                    ; We reached the end. So stop filling hearts.
    STA SwordBlocked
    STA World_IsFillingHearts
    STA Paused

@Exit:
    RTS

@IncHearts:
    INC HeartValues
    RTS
