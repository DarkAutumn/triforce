; ============================================================================
; FILE:     bank1/tile-block-power-triforce.asm BANK: 1 (switchable, $8000-$BFFF)
; UPSTREAM: Z_01.asm 2012-2211
; SUBJECTS: (none)
; TOPIC:    Block/rock/gravestone blocking Link; Triforce of Power fanfare
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0027 CheckTileObjectsBlocking       code  Params:
;   L0075 CheckPowerTriforceFanfare      code
;   L0121 TakePowerTriforce              code
;   L0131 PaletteRow7TransferRecord      data
;   L0138 GanonColorTriples              data  First: Ganon brown state
;   L0142 ReplaceGanonBrownPaletteRow    code
;   L0147 ReplaceGanonBluePaletteRow     code
;   L0152 ReplaceAshesPaletteRow         code
;   L0201 Unused_DrawAshes_Bank1         code  Description:
;   L0210 Unused_ActivateRoomItem_Bank1  code  Description:
; ============================================================================

; Params:
; [0F]: movement direction
;
; Returns:
; [0F]: untouched, or 0
;
CheckTileObjectsBlocking:
    ; Look for a tile object among objects 1 to 12 that can block
    ; the player's movement.
    LDX #$0C

@LoopTileObj:
    LDA ObjType, X
    CMP #$68                    ; Block
    BEQ @Found
    CMP #$62                    ; Rock
    BEQ @Found
    CMP #$65                    ; Gravestone
    BEQ @Found
    CMP #$66                    ; Armos1. But it seems to be unused, along with Armos2 ($67).
    BNE @NextTileObj            ; If it doesn't match any, go check the next object.

@Found:
    LDA ObjState, X
    CMP #$01
    BNE @NextTileObj            ; If not active, go check the next object.

    ; If X distance between Link and the object >= $10, skip it.
    LDA ObjX
    SEC
    SBC ObjX, X
    JSR Abs
    CMP #$10
    BCS @NextTileObj

    ; If Y distance between Link and the object >= $10, skip it.
    LDA ObjY
    CLC
    ADC #$03                    ; First, adjust Link's position.
    SEC
    SBC ObjY, X
    JSR Abs
    CMP #$10
    BCS @NextTileObj

    ; Link is near the object. Reset movement direction.
    LDA #$00
    STA $0F

@NextTileObj:
    DEX
    BNE @LoopTileObj
    RTS

CheckPowerTriforceFanfare:
    ; If the fanfare is not active, then return.
    LDA TriforceFanfareActive
    BEQ @Exit

    ; When Link's timer expires, go finish up the fanfare.
    LDA ObjTimer
    BEQ @EndFanfare

    ; While Link's timer is counting down, every 4 frames, switch
    ; between the level palette and the white palette.
    LDY #$18

    ; Redundant check of Link's timer that won't branch.
    LDA ObjTimer
    BEQ @FillHearts
    AND #$07
    CMP #$04
    BCC :+
    LDY #$78
:
    STY TileBufSelector
    RTS

@FillHearts:
    ; Begin unverified code 6196
    LDA #$02
    STA World_IsFillingHearts
    INC GameSubmode
    RTS
    ; End unverified code

@EndFanfare:
    ; The fanfare is done.
    JSR ReplaceAshesPaletteRow
    LDA #$20                    ; Play the song for level 9 again.
    STA SongRequest
    LDA #$01
    STA LastBossDefeated
    LSR
    STA ObjState                ; Make Link idle and not halted.
    STA TriforceFanfareActive   ; Flag the fanfare not active.

@Exit:
    RTS

TakePowerTriforce:
    INC TriforceFanfareActive

    ; Halt Link for $C0 frames.
    LDA #$C0
    STA ObjTimer
    LDA #$40
    STA ObjState
    RTS

PaletteRow7TransferRecord:
    .BYTE $3F, $1C, $04, $0F, $07, $17, $27, $FF

; First: Ganon brown state
; Second: Ganon blue state
; Third: Ashes and Triforce
;
GanonColorTriples:
    .BYTE $07, $17, $30, $16, $2C, $3C, $27, $06
    .BYTE $16

ReplaceGanonBrownPaletteRow:
    ; Begin unverified code 61CE
    LDY #$02
    BNE :+

ReplaceGanonBluePaletteRow:
    LDY #$05
    BNE :+
    ; End unverified code

ReplaceAshesPaletteRow:
    LDY #$08
:
    TYA
    PHA

    ; Get the dynamic transfer buf length, so we write from were we last wrote.
    LDX DynTileBufLen

    ; Copy the palette row 7 transfer record to dynamic transfer buf (8 bytes).
    LDY #$00

@CopyPaletteRecord:
    LDA PaletteRow7TransferRecord, Y
    STA DynTileBuf, X
    INX
    INY
    CPY #$08
    BNE @CopyPaletteRecord

    ; Update the dynamic transfer buf length.
    STX DynTileBufLen
    PLA
    TAY

    ; Replace the last 3 color bytes (brown shades) that we just
    ; copied to the dynamic transfer buf with the colors of the
    ; pile of ashes.
    ;
    ; Note that here the dynamic transfer buf is being written with
    ; absolute offsets, instead of relative ones used in copying
    ; the base/template palette row transfer record.
    LDX #$02

@ReplaceColors:
    LDA GanonColorTriples, Y
    STA DynTileBuf+4, X
    DEY
    DEX
    BPL @ReplaceColors

    ; Switch back to the current object slot.
    LDX CurObjIndex
    RTS

; Description:
; This procedure is the same as DrawAshPile in bank 2 and
; Ganon_DrawAshes in bank 4.
;
Unused_DrawAshes_Bank1:
    ; Begin unverified code 6200
    JSR Anim_FetchObjPosForSpriteDescriptor
    LDA #$0B
    JMP DrawObjectNotMirrored

; Description:
; This procedure is the same as Ganon_ActivateRoomItem in bank 4.
;
Unused_ActivateRoomItem_Bank1:
    LDA ObjState+19
    BEQ :+
    JSR GetRoomFlagUWItemState
    BNE :+
    LDA #$00
    STA ObjState+19
    LDA #$02
    STA Tune1Request
:
    RTS
