; ============================================================================
; FILE:     bank5/mode12-lay-out-room.asm BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 5537-5749
; SUBJECTS: (none)
; TOPIC:    End-level sequence; LayOutRoom dispatcher
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0025 InitMode12                           code
;   L0043 UpdateMode12EndLevel_Full            code
;   L0049 UpdateMode12EndLevel_Full_JumpTable  data
;   L0056 UpdateMode12EndLevel_Sub0            code
;   L0063 UpdateMode12EndLevel_Sub1            code
;   L0077 L16887_Exit                          code
;   L0080 StartFillingHearts                   code
;   L0087 L1688C_IncSubmode                    code
;   L0091 UpdateMode12EndLevel_Sub2            code
;   L0097 UpdateMode12EndLevel_Sub3            code
;   L0111 UpdateMode12EndLevel_Sub4            code
;   L0125 LayOutRoom                           code
;   L0147 CopyColumnToTileBuf                  code  Params:
;   L0188 CopyRowToTileBuf                     code
; ============================================================================

InitMode12:
    LDA #$04                    ; Play "End Level" song.
    STA SongRequest
    LDA #$20                    ; Set decreasing column for UpdateWorldCurtainEffect.
    STA ObjX+12
    LDA #$01                    ; Set increasing column for UpdateWorldCurtainEffect.
    STA ObjX+13
    LDA #$30                    ; Set to delay $2F ($30-1) frames when updating mode.
    STA ObjTimer
    LDA #$24                    ; Fill tile map with blanks.
    STA $0A
    JSR FillTileMap
    INC IsUpdatingMode
    JSR HideObjectSprites
    LDA #$1B                    ; Triforce item type.
    STA ItemTypeToLift
    JMP SetUpAndDrawLinkLiftingItem

UpdateMode12EndLevel_Full:
    JSR HideObjectSprites
    JSR DrawLinkLiftingItem
    LDA GameSubmode
    JSR TableJump

UpdateMode12EndLevel_Full_JumpTable:
    .ADDR UpdateMode12EndLevel_Sub0
    .ADDR UpdateMode12EndLevel_Sub1
    .ADDR UpdateMode12EndLevel_Sub2
    .ADDR UpdateMode12EndLevel_Sub3
    .ADDR UpdateMode12EndLevel_Sub4

UpdateMode12EndLevel_Sub0:
    LDA ObjTimer
    BNE L16887_Exit             ; Delay (return) until timer expires.
    LDA #$30                    ; Set to run next submode for $2F ($30-1) frames.
    STA ObjTimer
    BNE L1688C_IncSubmode

UpdateMode12EndLevel_Sub1:
    ; Flash the screen.
    ;
    ; $18 is LevelPaletteTransferBuf.
    LDY #$18
    LDA ObjTimer
    BEQ StartFillingHearts
    AND #$07                    ; Every 4 frames, switch palettes.
    CMP #$04
    BCC :+
    LDY #$78                    ; WhitePaletteBottomHalfTransferBuf
:
    STY TileBufSelector

L16887_Exit:
    RTS

StartFillingHearts:
    ; Start filling hearts, and go to next submode.
    ;
    ; UNKNOWN: Why 2? Either way, non-zero works.
    LDA #$02
    STA World_IsFillingHearts

L1688C_IncSubmode:
    INC GameSubmode
    RTS

UpdateMode12EndLevel_Sub2:
    JSR UpdateHeartsAndRupees
    LDA World_IsFillingHearts
    BEQ :+                      ; If finished filling hearts, then go set the timer for the next submode.
    RTS

UpdateMode12EndLevel_Sub3:
    LDA ObjTimer
    BNE :++
    JSR UpdateWorldCurtainEffect
    LDA ObjX+12
    CMP #$11
    BCS :++                     ; If decreasing column hasn't reached the middle (still >= $11), then return.
:
    LDA #$80                    ; Set up a delay of $7F ($80-1) frames for next submode.
    STA ObjTimer
    INC GameSubmode
:
    RTS

UpdateMode12EndLevel_Sub4:
    LDA ObjTimer
    BNE :-
    JSR HideAllSprites
    LDA CurPpuControl_2000
    AND #$FB                    ; Make sure VRAM address increment is 1.
    STA CurPpuControl_2000
    STA PpuControl_2000
    JMP EndGameMode12
:
    ; Is in OW.
    JSR LayoutRoomOW
    JMP CheckShortcut

LayOutRoom:
    JSR PatchColumnDirectoryForCellar
    LDA CurLevel
    BEQ :-

    ; Is in UW.
    ;
    ; Fill PlayArea with brick tiles that are seen at the margins.
    LDA #$F6
    STA $0A
    JSR FillTileMap
    JSR AddDoorFlagsToCurOpenedDoors
    JSR FillWalls
    JSR LayOutDoors
    JMP LayoutUWFloor

; Params:
; CurColumn: target column + 1
;
; Put $651A in [$00:01]; $16 before $6530 which is the tile map address.
;
; The first iteration of the loop below will add $16 to it before using it.
CopyColumnToTileBuf:
    LDA #$1A
    STA $00
    LDA #$65
    STA $01
    LDX CurColumn
    DEX                         ; Start with X = CurColumn - 1
    TXA
    LDY DynTileBufLen           ; Fill dynamic transfer buf from last position written.
    STA DynTileBuf+1, Y         ; Use PPU address $21xx: a tile along the first row in play area.
    LDA #$21
    STA DynTileBuf, Y
:
    ; Keep adding $16 until you point to the target column.
    LDA #$16
    JSR AddToInt16At0
    DEX
    BPL :-
    LDA #$96                    ; $96: $16 tiles, vertically in nametable.
    STA DynTileBuf+2, Y
    TXA                         ; X is $FF. Use it as the end marker.
    STA DynTileBuf+25, Y
    TYA                         ; Move the dynamic transfer buf offset to X.
    TAX
    LDY #$00                    ; Reset the tile counter in [$06] and Y.
    STY $06
:
    LDA ($00), Y                ; Load source tile in TileMap.
    STA DynTileBuf+3, X         ; Store it in dynamic transfer buf.
    JSR Add1ToInt16At0          ; Increment the source address.
    INX                         ; Increment the destination offset.
    INC $06                     ; Increment the counter.
    LDA $06
    CMP #$16
    BCC :-                      ; If we haven't copied $16 tiles, then loop again.
    INX
    INX
    INX
    STX DynTileBufLen
    RTS

CopyRowToTileBuf:
    ; Put in 00:01 the address of the
    ; first tile of current row in play area.
    LDA #$65
    STA $01
    LDA CurRow
    TAX
    CLC
    ADC #$30
    STA $00
    BCC :+
    ; Begin unverified code 16932
    INC $01
    ; End unverified code
:
    ; Indicate the target VRAM address:
    ; $2100 + (CurRow * $20)
    LDA #$20
    STA DynTileBuf
    LDA #$E0
    STA DynTileBuf+1

@Add20H:
    LDA DynTileBuf+1
    CLC
    ADC #$20
    STA DynTileBuf+1
    BCC :+
    INC DynTileBuf
:
    DEX
    BPL @Add20H
    LDA #$20                    ; Indicate 32 bytes to copy.
    STA DynTileBuf+2
    STX DynTileBuf+35           ; Put an end marker.

    ; Copy a row from column map in RAM to tile buf.
    LDX #$00
    LDY #$00
:
    LDA ($00), Y                ; Copy a tile.
    STA DynTileBuf+3, X
    LDA #$16
    JSR AddToInt16At0           ; Advance to the tile in the same row, but next column of play area.
    INX                         ; Advance to the next tile in the row in transfer buf.
    CPX #$20
    BCC :-                      ; If we haven't written $20 tiles, then go write another.
    LDA #$23                    ; The transfer buf is $23 bytes (3 for header, $20 for payload).
    STA DynTileBufLen
    RTS
