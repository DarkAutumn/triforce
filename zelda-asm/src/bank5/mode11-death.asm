; ============================================================================
; FILE:     bank5/mode11-death.asm       BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 2523-2718
; SUBJECTS: (none)
; TOPIC:    Death update
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0032 UpdateMode11Death_Full                            code
;   L0036 UpdateMode11Death_Full_JumpTable                  data
;   L0051 UpdateMode11Death_Sub1                            code
;   L0058 UpdateMode11Death_Sub0                            code
;   L0066 CueTransferPlayAreaAttrsHalfAndAdvanceSubmodeNT2  code  Params:
;   L0074 CueTransferPlayAreaAttrsHalfAndAdvanceSubmode     code  Params:
;   L0079 UpdateMode11Death_Sub2                            code
;   L0091 UpdateMode11Death_Sub3                            code
;   L0094 SelectTransferBufAndAdvanceSubmode                code
;   L0097 UpdateMode11Death_Sub4                            code
;   L0101 UpdateMode11Death_Sub5                            code
;   L0107 UpdateMode11Death_Sub6                            code
;   L0112 L14CD7_IncSubmode                                 code
;   L0116 UpdateMode11Death_Sub7                            code
;   L0150 UpdateMode11Death_Sub8_AnimateFade                code
;   L0155 UpdateMode11Death_Sub9                            code
;   L0166 UpdateMode11Death_SubA                            code
;   L0198 UpdateMode11Death_SetTimerIncSubmode              code
;   L0202 L14D55_Exit                                       code
;   L0205 UpdateMode11Death_SubB                            code
;   L0213 UpdateMode11Death_SubC                            code
; ============================================================================

UpdateMode11Death_Full:
    LDA GameSubmode
    JSR TableJump

UpdateMode11Death_Full_JumpTable:
    .ADDR UpdateMode11Death_Sub0
    .ADDR UpdateMode11Death_Sub1
    .ADDR UpdateMode11Death_Sub2
    .ADDR UpdateMode11Death_Sub3
    .ADDR UpdateMode11Death_Sub4
    .ADDR UpdateMode11Death_Sub5
    .ADDR UpdateMode11Death_Sub6
    .ADDR UpdateMode11Death_Sub7
    .ADDR UpdateMode11Death_Sub8_AnimateFade
    .ADDR UpdateMode11Death_Sub9
    .ADDR UpdateMode11Death_SubA
    .ADDR UpdateMode11Death_SubB
    .ADDR UpdateMode11Death_SubC

UpdateMode11Death_Sub1:
    ; Submode 1 prepares bottom half of play area attributes.
    ;
    ; Play death tune.
    LDA #$80
    STA Tune1Request

UpdateMode11Death_Sub0:
    ; Submode 0 prepares top half of play area attributes.
    JSR ChooseAttrSourceAndDestForSubmode

; Params:
; A: low PPU address
; Y: end offset in PlayAreaAttrs to copy from
;
CueTransferPlayAreaAttrsHalfAndAdvanceSubmodeNT2:
    LDX #$2B                    ; Use nametable 2.

; Params:
; X: high PPU address
; A: low PPU address
; Y: end offset in PlayAreaAttrs to copy from
;
CueTransferPlayAreaAttrsHalfAndAdvanceSubmode:
    JSR CopyPlayAreaAttrsHalfToDynTransferBuf
    INC GameSubmode
    RTS

UpdateMode11Death_Sub2:
    ; Updates all play area tiles.
    JSR CopyNextRowToTransferBufAndAdvanceSubmodeWhenDone
    BCC :+
    JSR WriteAndEnableSprite0
:
    LDA DynTileBuf              ; Modify the transfer record, so that it changes nametable 2.
    CLC
    ADC #$08
    STA DynTileBuf
    RTS

UpdateMode11Death_Sub3:
    LDA #$60                    ; Cue transfer of top half of play area attributes.

SelectTransferBufAndAdvanceSubmode:
    JMP L1712E_SelectTransferBufAndAdvanceSubmode

UpdateMode11Death_Sub4:
    LDA #$62                    ; Cue transfer of bottom half of play area attributes.
    JMP SelectTransferBufAndAdvanceSubmode    ; Go set this TileBufSelector value, and advance submode.

UpdateMode11Death_Sub5:
    LDA #$00
    STA IsSprite0CheckActive
    LDA #$5E                    ; Cue transfer of bottom half of background palette for this mode.
    JMP SelectTransferBufAndAdvanceSubmode    ; Go set this TileBufSelector value, and advance submode.

UpdateMode11Death_Sub6:
    LDA CurPpuControl_2000
    AND #$FE                    ; Make sure we're using an even nametable number.
    STA CurPpuControl_2000

L14CD7_IncSubmode:
    INC GameSubmode
    RTS

UpdateMode11Death_Sub7:
    LDA DeathModeCounter
    BEQ L14CD7_IncSubmode       ; Once Link has turned enough times, go to the next submode.
    LDA ObjTimer+11
    BNE @DrawLink               ; If timer hasn't expired, only redraw sprites.
    LDA #$05                    ; Once ObjTimer[11] expires, arm it again, and change direction.
    STA ObjTimer+11
    LDA ObjDir
    LSR
    LSR
    BCC @CheckOtherDirs         ; If not left, then go check other directions.

    ; Facing left.
    DEC DeathModeCounter
    LDA #$04                    ; Face the next direction (down).

@SetLinkDirAndDraw:
    STA ObjDir

@DrawLink:
    JMP Link_EndMoveAndAnimate  ; Go redraw Link.

@CheckOtherDirs:
    ; If not right, then vertical. %A was shifted to become
    ; horizontal direction counter-clockwise from original
    ; vertical direction. Got set it and redraw.
    BNE @SetLinkDirAndDraw

    ; Facing right.
    ;
    ; Face the next direction (up).
    LDA #$08
    BNE @SetLinkDirAndDraw      ; Go set the direction and redraw.

UpdateMode11Death_Sub8_AnimateFade:
    JSR AnimateWorldFading
    BEQ L14CD7_IncSubmode       ; If done, then go to the next submode.
    RTS

UpdateMode11Death_Sub9:
    LDA #$2C                    ; Cue the transfer of the dead Link (grey) palette row.
    STA TileBufSelector
    LDA #$0F

    ; In these two submodes, [E5] DeathModeCounter counts down
    ; how long the spark lasts.
    STA DeathModeCounter
    LDA #$18
    BNE UpdateMode11Death_SetTimerIncSubmode    ; Go set a delay of $17 ($18-1) frames, and advance the submode.

UpdateMode11Death_SubA:
    LDA ObjTimer+11
    BNE L14D55_Exit             ; Delay until the timer expires, and we can show the spark.
    LDX #$62                    ; The little spark tile.
    LDA DeathModeCounter        ; How much time is left for the spark?
    CMP #$06
    BCS :+                      ; If counter/spark timer >= 6, use little spark tile $62.
    LDX #$64                    ; Else use the big one.
:
    LDA ObjY
    STA Sprites+72              ; Use Link's Y for both sides.
    STA Sprites+76
    STX Sprites+73              ; Use the spark tile we chose.
    STX Sprites+77
    LDA #$01                    ; Use palette 5 for the left.
    STA Sprites+74
    LDA #$41                    ; Use the same palette, but flip horizontally on the right.
    STA Sprites+78
    LDA ObjX                    ; Use Link's X.
    STA Sprites+75
    CLC                         ; The second sprite is 8 pixels to the right.
    ADC #$08
    STA Sprites+79
    DEC DeathModeCounter        ; Count down how long you see the spark.
    BNE L14D55_Exit             ; If not zero yet, then return.
    LDA #$10                    ; Play "heart taken" tune.
    STA Tune0Request
    LDA #$F8                    ; Hide the Link/spark sprites.
    STA Sprites+72
    STA Sprites+76
    LDA #$2E                    ; Set a delay of $2D ($2E-1) frames for the next submode.

UpdateMode11Death_SetTimerIncSubmode:
    STA ObjTimer+11
    INC GameSubmode

L14D55_Exit:
    RTS

UpdateMode11Death_SubB:
    LDA ObjTimer+11
    BNE L14D55_Exit             ; If the timer hasn't expired, then return.
    LDA #$60                    ; Set a delay of $5F ($60-1) frames for the next submode.
    STA ObjTimer+11
    LDA #$46                    ; Cue transfer of "GAME OVER" text, and advance submode.
    JMP SelectTransferBufAndAdvanceSubmode

UpdateMode11Death_SubC:
    LDA ObjTimer+11
    BNE :+                      ; If the timer hasn't expired, then return.
    JSR EndGameMode
    LDA #$08                    ; Go to the Continue Question mode.
    STA GameMode
    LDA #$40                    ; Request Game Over music.
    STA Tune1Request
    LDX CurSaveSlot
    LDA DeathCounts, X          ; Increase the death count for current profile.
    CMP #$FF                    ; Up to the maximum $FF.
    BEQ :+
    INC DeathCounts, X
:
    RTS
