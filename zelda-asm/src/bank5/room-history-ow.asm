; ============================================================================
; FILE:     bank5/room-history-ow.asm    BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 3515-3655
; SUBJECTS: (none)
; TOPIC:    Mode B/C/9 init tables; overworld room history and kill counts
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0021 InitModeB                                   code
;   L0025 InitModeB_JumpTable                         data
;   L0036 InitModeC                                   code
;   L0040 InitModeC_JumpTable                         data
;   L0051 ModifyObjCountByHistoryOW                   code
;   L0098 SaveKillCountOW                             code
;   L0121 InitMode9                                   code
;   L0125 InitMode9_JumpTable                         data
;   L0141 Unused_DrawObjectNotMirroredOverLink_Bank5  code  Description:
;   L0151 Unused_DrawObjectMirroredOverLink_Bank5     code  Description:
; ============================================================================

InitModeB:
    LDA GameSubmode
    JSR TableJump

InitModeB_JumpTable:
    .ADDR InitModeSubroom_Sub0
    .ADDR InitModeB_Sub1
    .ADDR InitModeSubroom_AdvanceSubmode
    .ADDR LayoutCaveAndAvanceSubmode
    .ADDR CopyNextRowToTransferBufAndAdvanceSubmodeWhenDone
    .ADDR InitModeB_Sub5_FillTileAttrsAndTransferTopHalf
    .ADDR InitModeAOrB_TransferBottomHalfAttrs
    .ADDR InitModeB_EnterCave_Bank5
    .ADDR InitMode_WalkCave

InitModeC:
    LDA GameSubmode
    JSR TableJump

InitModeC_JumpTable:
    .ADDR InitModeSubroom_Sub0
    .ADDR InitModeB_Sub1
    .ADDR InitModeSubroom_AdvanceSubmode
    .ADDR LayoutShortcutAndAdvanceSubmode
    .ADDR CopyNextRowToTransferBufAndAdvanceSubmodeWhenDone
    .ADDR InitModeB_Sub5_FillTileAttrsAndTransferTopHalf
    .ADDR InitModeAOrB_TransferBottomHalfAttrs
    .ADDR InitModeB_EnterCave_Bank5
    .ADDR InitMode_WalkCave

ModifyObjCountByHistoryOW:
    ; Look for the room in the history.
    LDY #$05
    LDA RoomId
:
    CMP RoomHistory, Y
    BEQ :+                      ; If found, go check kill count in depth.
    DEY
    BPL :-

    ; It wasn't found. Check the kill count in the room flags.
    JSR GetRoomFlags
    AND #$07
    CMP #$07
    BNE :+                      ; If kill count < max, go check kill count in depth.

    ; The room is not in history, and all foes were defeated.
    ; So, reset the kill count in room flags.
    LDA ($00), Y
    AND #$F8
    STA ($00), Y

@Exit:
    RTS
:
    ; If kill count in room flags = 0, leave object count alone, and return.
    JSR GetRoomFlags
    AND #$07
    BEQ @Exit

    ; If kill count = 7, go reset object count and object list ID.
    ; Else subtract kill count from object count.
    CMP #$07
    BEQ @ResetObjList
    STA $04
    LDA $03
    SEC
    SBC $04
    BPL :+                      ; If the result >= 0, go set the object count to this.

@ResetObjList:
    LDA #$00                    ; Else, reset object count and object list ID.
    STA $02
:
    STA $03
    RTS

SaveKillCountOW:
    JSR GetRoomFlags
    AND #$07                    ; Put kill count from the room's flags in [02].
    STA $02
    LDA ($00), Y                ; Reset kill count part of the room's flags.
    AND #$F8
    STA ($00), Y
    LDA RoomKillCount
    CMP RoomObjCount
    BCS @LimitCount             ; If RoomKillCount >= RoomFoeCount, go store the max kill count.
    AND #$07                    ; Limit RoomKillCount to 7.
    CLC                         ; Add it to kill count from flags.
    ADC $02
    CMP #$07
    BCC :+                      ; If the total > 7,

@LimitCount:
    LDA #$07                    ; then limit it to 7.
:
    ORA ($00), Y                ; Combine this new total and the room's flags.
    STA ($00), Y
    RTS

InitMode9:
    LDA GameSubmode
    JSR TableJump

InitMode9_JumpTable:
    .ADDR InitModeSubroom_Sub0
    .ADDR InitMode9_FadeToDark
    .ADDR InitModeSubroom_AnimateFade
    .ADDR LayoutCellarAndAdvanceSubmode
    .ADDR CopyNextRowToTransferBufAndAdvanceSubmodeWhenDone
    .ADDR InitMode9_TransferAttrs
    .ADDR InitMode9_FadeToLight
    .ADDR InitModeSubroom_AnimateFade
    .ADDR InitMode9_EnterCellar
    .ADDR InitMode9_WalkCellar

; Description:
; This procedure is the same as DrawObjectNotMirroredOverLink
; in bank 4.
;
Unused_DrawObjectNotMirroredOverLink_Bank5:
    ; Begin unverified code 15123
    LDY #$00
    BEQ Unused_DrawObjectMirroredOverLink_Bank5
    LDY #$01

; Description:
; This procedure is the same as DrawObjectMirroredOverLink
; in bank 4.
;
Unused_DrawObjectMirroredOverLink_Bank5:
    STY $0C
    LDY ObjType, X
    INY
    STA $0D
    STY $0E
    STX $08
    LDA #$40
    STA LeftSpriteOffset
    LDA #$44
    JMP DrawObjectWithAnimAndSpecificSprites
