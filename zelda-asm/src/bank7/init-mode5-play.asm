; ============================================================================
; FILE:     bank7/init-mode5-play.asm    BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 1491-1647
; SUBJECTS: (none)
; TOPIC:    Boss/tile-object palettes, InitMode5Play
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0017 SpecialBossPaletteTransferBufSelectors                 data
;   L0021 SpecialBossPaletteObjTypes                             data
;   L0025 InitMode5Play                                          code
;   L0107 RunCrossRoomTasksAndBeginUpdateMode_PlayModesNoCellar  code
;   L0113 RunCrossRoomTasksAndBeginUpdateMode_EnterPlayModes     code
;   L0120 RunCrossRoomTasksAndBeginUpdateMode                    code
; ============================================================================

SpecialBossPaletteTransferBufSelectors:
    .BYTE $08, $36, $0A, $0A, $0A, $0A, $7C, $7C
    .BYTE $7C

SpecialBossPaletteObjTypes:
    .BYTE $3D, $3E, $38, $39, $32, $31, $43, $44
    .BYTE $45

InitMode5Play:
    JSR DrawSpritesBetweenRooms
    JSR Link_EndMoveAndAnimate
    LDA CurLevel
    BEQ @InOW

    ; In UW.
    ;
    ; Certain bosses and boss-like monsters need
    ; their own palettes. If we're dealing with one, then get
    ; the matching transfer buffer selector.
    LDY #$08
    LDA ObjType+1

@FindSpecialBoss:
    CMP SpecialBossPaletteObjTypes, Y
    BNE @NextSpecialBoss        ; If the object type doesn't match, go check the next one.
    LDX SpecialBossPaletteTransferBufSelectors, Y
    BNE @SelectTransferBufAndFinishInitPlay    ; Go cue the transfer of this palette row. The selector is in X.

@NextSpecialBoss:
    DEY
    BPL @FindSpecialBoss
    BMI @UseLevelPalette        ; If it doesn't match any boss type, go copy and transfer level palette row 7.

@InOW:
    ; In OW.
    ;
    ; If the room is 0F and we just walked in instead of
    ; coming out of underground; then play "secret found" tune.
    LDA RoomId
    CMP #$0F
    BNE @ChooseTileObjPalette
    LDA UndergroundExitType
    BNE @ChooseTileObjPalette
    LDA #$04
    STA Tune1Request

@ChooseTileObjPalette:
    ; Choose a palette transfer buf for a tile object that has sprites.
    ; Rock, gravestone, Armos1, Armos2
    LDX #$20                    ; Ghost palette row
    LDA ObjType+11
    CMP #$65
    BEQ @SelectTransferBufAndFinishInitPlay    ; If tile object type is gravestone ($65), go cue transfer buf $20.
    CMP #$66
    BEQ @UseXOrGreenPalette     ; If tile object type is Armos1 ($66), go see if the room's attributes should override transfer buf $20.
    CMP #$62
    BNE @UseRedArmosPalette     ; If type is not rock ($62) (so Armos2), go set the right selector.

    ; It's a rock tile object.
    ; Choose palette based on inner palette attribute to match it.
    LDX #$24                    ; Brown palette row

@UseXOrGreenPalette:
    LDY RoomId
    LDA LevelBlockAttrsB, Y
    AND #$01
    BNE @SelectTransferBufAndFinishInitPlay    ; If the room's inner palette attribute is odd, go cue transfer buf chosen above ($20 if Armos1, else $24).
    LDX #$22                    ; Green palette row
    BNE @SelectTransferBufAndFinishInitPlay    ; Else it's even, so go cue transfer buf $22.

@UseRedArmosPalette:
    ; Tile object is Armos2. Use red Armos palette row.
    LDX #$7A
    BNE @SelectTransferBufAndFinishInitPlay    ; Go cue this transfer buf.

@UseLevelPalette:
    ; Patch and cue palette row 7 transfer buf (6) with row 7
    ; of level palette.
    LDY #$03

@PatchColors:
    LDA LevelInfo_PalettesTransferBuf+31, Y
    STA LevelPaletteRow7TransferBuf+3, Y
    DEY
    BPL @PatchColors
    LDX #$06

@SelectTransferBufAndFinishInitPlay:
    STX TileBufSelector

RunCrossRoomTasksAndBeginUpdateMode_PlayModesNoCellar:
    ; Called in modes 5, $B, $C.
    LDA #$05
    JSR SwitchBank
    JSR SetupObjRoomBounds

RunCrossRoomTasksAndBeginUpdateMode_EnterPlayModes:
    ; Called in modes 4, 5, 9, $B, $C.
    LDA CurLevel
    BEQ RunCrossRoomTasksAndBeginUpdateMode
    JSR MarkRoomVisited
    JSR WriteBlankPrioritySprites

RunCrossRoomTasksAndBeginUpdateMode:
    ; Called in modes 4, 5, 6, 9, $B, $C.
    LDA #$05
    JSR SwitchBank
    JSR CreateRoomObjects

    ; Look for the current room in room history.
    LDY #$00
    LDX #$05
    LDA RoomId

@LoopHistory:
    CMP RoomHistory, X
    BNE :+
    INY
:
    DEX
    BPL @LoopHistory

    ; If it's not found, then store at the cycling history index.
    CPY #$00
    BNE @RunTasksMode5
    LDX CurRoomHistoryIndex
    STA RoomHistory, X
    INC CurRoomHistoryIndex
    LDA CurRoomHistoryIndex
    CMP #$06
    BCC @RunTasksMode5
    LDA #$00
    STA CurRoomHistoryIndex

@RunTasksMode5:
    ; If not in mode 5, go start updating the mode.
    LDA GameMode
    CMP #$05
    BNE @BeginUpdate            ; If not in mode 5, go start updating.

    ; Run cross-room tasks that apply only to mode 5.
    LDA CurLevel
    BEQ @CheckWhirlwind
    LDA #$05
    JSR SwitchBank
    JSR CheckBossSoundEffectUW

@BeginUpdate:
    JMP BeginUpdateMode

@CheckWhirlwind:
    LDA #$01
    JSR SwitchBank
    JMP CheckInitWhirlwindAndBeginUpdate

; Unknown block
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF
