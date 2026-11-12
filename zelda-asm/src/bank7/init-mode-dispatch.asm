; ============================================================================
; FILE:     bank7/init-mode-dispatch.asm BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 1257-1489
; SUBJECTS: (none)
; TOPIC:    InitMode jump table, several mode inits, room history
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0029 InitializeGameOrMode                               code
;   L0045 InitMode                                           code
;   L0051 InitMode_JumpTable                                 data
;   L0073 InitMode0                                          code
;   L0092 InitMode1                                          code
;   L0097 InitMode2                                          code
;   L0125 InitMode7                                          code
;   L0155 InitModeEandF                                      code
;   L0160 InitMode13                                         code
;   L0168 InitMode3                                          code
;   L0175 InitMode3_JumpTable                                data
;   L0186 InitMode3_Sub0                                     code
;   L0197 ClearRoomHistory                                   code  Returns:
;   L0211 SaveSlotToPaletteRowOffset                         data  Each element is the offset of a palette row starting from
;   L0214 InitMode3_Sub1                                     code
;   L0231 PatchAndCueLevelPalettesTransferAndAdvanceSubmode  code
;   L0248 DrawSpritesBetweenRooms                            code
;   L0257 ResetPlayerState                                   code
; ============================================================================

InitializeGameOrMode:
    LDA IsGameInitialized
    BNE InitMode
    LDA #$01
    JSR SwitchBank
    JSR CopyCommonCodeToRam
    LDA #$06
    JSR SwitchBank
    JSR CopyCommonDataToRam
    LDA #$5A                    ; Mark Save RAM initialized, so we can check after reset.
    STA SaveRamBegin
    LDA #$A5
    STA SaveRamEnd
    INC IsGameInitialized
    RTS

InitMode:
    LDA #$05
    JSR SwitchBank
    LDA GameMode
    JSR TableJump

InitMode_JumpTable:
    .ADDR InitMode0
    .ADDR InitMode1
    .ADDR InitMode2
    .ADDR InitMode3
    .ADDR InitMode4
    .ADDR InitMode5Play
    .ADDR InitMode6
    .ADDR InitMode7
    .ADDR InitMode8
    .ADDR InitMode9
    .ADDR InitModeA
    .ADDR InitModeB
    .ADDR InitModeC
    .ADDR InitModeD
    .ADDR InitModeEandF
    .ADDR InitModeEandF
    .ADDR InitMode10
    .ADDR InitMode11
    .ADDR InitMode12
    .ADDR InitMode13

InitMode0:
    LDA HasCommonPatterns
    CMP #$5A
    BEQ :+
    LDA #$02
    JSR SwitchBank
    JMP TransferCommonPatterns
:
    LDA HasDemoPatterns
    CMP #$A5
    BEQ :+
    LDA #$01
    JSR SwitchBank
    JMP TransferDemoPatterns
:
    LDA #$02
    JSR SwitchBank
    JMP InitDemo_RunTasks

InitMode1:
    LDA #$02
    JSR SwitchBank
    JMP InitMode1_Full

InitMode2:
    JSR TurnOffAllVideo
    LDA GameSubmode
    BNE @InitSubmodes

    ; Submode 0.
    ; First step, transfer level pattern blocks and copy common code.
    JSR ClearRoomHistory

    ; Clear level kill counts.
    LDY #$7F

@ClearCounts:
    STA LevelKillCounts, Y
    DEY
    BPL @ClearCounts
    LDA #$03
    JSR SwitchBank
    JSR TransferLevelPatternBlocks
    LDA #$01
    JSR SwitchBank
    JSR CopyCommonCodeToRam

@InitSubmodes:
    LDA #$06
    JSR SwitchBank
    JMP InitMode2_Submodes

InitMode7:
    LDA #$05
    JSR SwitchBank
    JSR InitMode7Submodes
    LDA IsSprite0CheckActive
    BEQ @Exit                   ; If not checking sprite 0, return.

    ; After the last submode (5 or 6), sprite-0 check was enabled.
    ; So, set the appropriate mirroring for scrolling during mode update.
    LDA IsScrollMirroringSet             ; TODO: [$F3] ?
    BNE @Exit
    INC IsScrollMirroringSet

    ; If player is facing horizontally, then enable vertical mirroring;
    ; else horizontal mirroring.
    LDA ObjDir
    CMP #$04
    BCC @SetVertical
    LDA #$0F                    ; horizontal mirroring
    BNE @SetMirroring

@SetVertical:
    LDA #$0E                    ; vertical mirroring

@SetMirroring:
    JSR SetMMC1Control

@Exit:
    RTS

InitModeEandF:
    LDA #$02
    JSR SwitchBank
    JMP InitModeEandF_Full

InitMode13:
    ; Make sure horizontal mirroring is on.
    LDA #$0F
    JSR SetMMC1Control
    LDA #$02
    JSR SwitchBank
    JMP InitMode13_Full

InitMode3:
    LDA #$05
    JSR SwitchBank
    JSR TurnOffAllVideo
    LDA GameSubmode
    JSR TableJump

InitMode3_JumpTable:
    .ADDR InitMode3_Sub0
    .ADDR InitMode3_Sub1
    .ADDR InitMode3_Sub2
    .ADDR InitMode3_Sub3_TransferTopHalfAttrs
    .ADDR InitMode3_Sub4_TransferBottomHalfAttrs
    .ADDR InitMode3_Sub5
    .ADDR InitMode3_Sub6
    .ADDR InitMode3_Sub7
    .ADDR InitMode3_Sub8

InitMode3_Sub0:
    LDA #$01
    STA $17                     ; TODO: [17] ?
    INC GameSubmode
    JSR TurnOffVideoAndClearArtifacts

; Returns:
; A: 0
;
; Also resets [0529].
;
ClearRoomHistory:
    LDY #$05
    LDA #$00
    STA IsSwordShotForced

@Loop:
    ; Clear the room history.
    STA RoomHistory, Y
    DEY
    BPL @Loop
    RTS

; Each element is the offset of a palette row starting from
; row 4. Index by save slot number.
SaveSlotToPaletteRowOffset:
    .BYTE $00, $04, $08

InitMode3_Sub1:
    LDA CurLevel
    BNE @UseStartRoomId         ; If in UW, then go set room ID to StartRoomId.
    LDA LevelEntranceRoomId
    CMP #$FF
    BNE @SetRoomId              ; If it's set to a valid room ID, then start there.

@UseStartRoomId:
    LDA LevelInfo_StartRoomId

@SetRoomId:
    STA RoomId
    CMP LevelEntranceRoomId
    BNE PatchAndCueLevelPalettesTransferAndAdvanceSubmode    ; If CaveEnteredRoomId was valid,
    LDA #$FF                    ; then keep CaveEnteredRoomId invalid by default.
    STA LevelEntranceRoomId

PatchAndCueLevelPalettesTransferAndAdvanceSubmode:
    LDX CurSaveSlot
    LDY SaveSlotToPaletteRowOffset, X

    ; Get the color at byte 1 of row 4, 5, or 6 of menu palettes,
    ; according to save slot. This holds Link's color in that
    ; save slot.
    LDA MenuPalettesTransferBuf+20, Y

    ; Put the value in byte 1 of row 4 of level palettes that will
    ; be transferred.
    STA LevelInfo_PalettesTransferBuf+20
    LDA #$18                    ; Cue transfer of level palettes.
    STA TileBufSelector
    INC GameSubmode
    RTS

DrawSpritesBetweenRooms:
    JSR HideAllSprites
    JSR UpdatePlayerPositionMarker
    JSR UpdateTriforcePositionMarker
    LDA #$05
    JSR SwitchBank
    JSR DrawLinkBetweenRooms
    JMP DrawStatusBarItemsAndEnsureItemSelected

ResetPlayerState:
    LDA #$00
    STA ObjState
    STA InvClock
    RTS
