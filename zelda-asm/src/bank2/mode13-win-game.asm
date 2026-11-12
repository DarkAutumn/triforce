; ============================================================================
; FILE:     bank2/mode13-win-game.asm    BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 3405-3670
; SUBJECTS: (none)
; TOPIC:    Win game update
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0024 UpdateMode13WinGame                  code
;   L0028 UpdateMode13WinGame_JumpTable        data
;   L0035 EndingFlashColors                    data
;   L0038 UpdateMode13WinGame_Sub0_Flash       code
;   L0091 DrawLinkZeldaTriforces               code
;   L0139 UpdateMode13WinGame_Sub1             code
;   L0172 PeaceTextboxCharTransferRecTemplate  data
;   L0175 PeaceTextboxCharAddrsLo              data
;   L0184 PeaceText                            data
;   L0193 UpdatePeaceTextbox                   code
;   L0247 LAB7E_Exit                           code
;   L0250 UpdateMode13WinGame_Sub4             code
;   L0286 DrawAshPile                          code
; ============================================================================

UpdateMode13WinGame:
    LDA GameSubmode
    JSR TableJump

UpdateMode13WinGame_JumpTable:
    .ADDR UpdateMode13WinGame_Sub0_Flash
    .ADDR UpdateMode13WinGame_Sub1
    .ADDR UpdateMode13WinGame_Sub1
    .ADDR UpdateMode13WinGame_Sub3
    .ADDR UpdateMode13WinGame_Sub4

EndingFlashColors:
    .BYTE $0F, $12, $16, $2A

UpdateMode13WinGame_Sub0_Flash:
    JSR HideAllSprites
    INC ItemLiftTimer
    LDA ItemLiftTimer
    CMP #$C0
    BEQ @AdvanceSubmode
    JSR DrawLinkZeldaTriforces

@ChangePalette:
    LDX ItemLiftTimer

    ; Don't flash until $40 frames have passed.
    CPX #$40
    BCC @Exit

    ; Copy the level palette transfer buf.
    LDY #$23
:
    LDA LevelInfo_PalettesTransferBuf, Y
    STA DynTileBuf, Y
    DEY
    BPL :-

    ; Use (timer MOD 4) as an index to look up a color.
    TXA
    AND #$03
    TAX

    ; Change element 0 of palette 4 in buffer to change
    ; the background color.
    LDA EndingFlashColors, X
    STA DynTileBuf+19

@Exit:
    RTS

@AdvanceSubmode:
    ; Play the ending song.
    LDA #$10
    STA SongRequest

    ; Wait $40 frames at the beginning of the next mode
    ; before showing text.
    LDA #$40
    STA ObjTimer

    ; Set a long timer of $40 ($280 frames) for the whole duration
    ; of the next submode.
    LDA #$40
    STA EndingPeaceLongTimer
    INC GameSubmode
    JMP @ChangePalette

DrawLinkZeldaTriforces:
    ; The triforce over Link goes in the room object slot $13.
    ; Set its location $10 pixels above Link.
    LDA ObjX
    STA ObjX+19
    LDA ObjY
    SEC
    SBC #$10
    STA ObjY+19

    ; Draw Link.
    LDX #$00
    JSR Anim_FetchObjPosForSpriteDescriptor
    JSR Anim_SetSpriteDescriptorAttributes
    STA $0C
    LDA #$48
    STA LeftSpriteOffset
    LDA #$4C
    STA RightSpriteOffset
    LDY #$21
    JSR Anim_WriteSpecificItemSprites

    ; Draw triforce over Link.
    LDA #$1B
    LDX #$13
    JSR AnimateItemObject
    LDX #$01
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; Draw Zelda.
    TXA
    JSR DrawObjectMirrored

    ; The triforce over Zelda goes in slot 2.
    ; Set its location $10 pixels above her.
    LDA ObjX+1
    STA ObjX+2
    LDA ObjY+1
    SEC
    SBC #$10
    STA ObjY+2

    ; Draw triforce over Zelda.
    LDX #$02
    LDA #$1B
    JSR AnimateItemObject
    RTS

UpdateMode13WinGame_Sub1:
    LDA EndingPeaceLongTimer
    BEQ @AdvanceSubmode
    JSR HideAllSprites

    ; Characters stop emitting when the long timer is $10.
    ; So, keep showing Link and Zelda until it reaches 4.
    ; Then hide them and wait until it reaches 0.
    LDA EndingPeaceLongTimer
    CMP #$04
    BCC @Exit
    JSR DrawLinkZeldaTriforces

    ; Once in submode 2, only delay, instead of emitting characters.
    LDA GameSubmode
    CMP #$01
    BNE @Exit

    ; There is a delay before showing text.
    LDA ObjTimer
    BNE @Exit
    JSR UpdatePeaceTextbox

@Exit:
    RTS

@AdvanceSubmode:
    ; Transfer the ending palette, and go to the next submode.
    LDA #$6A
    STA TileBufSelector
    INC GameSubmode
    RTS

PeaceTextboxCharTransferRecTemplate:
    .BYTE $22, $A4, $01, $24, $FF

PeaceTextboxCharAddrsLo:
    .BYTE $AC, $AD, $AE, $AF, $B0, $B1, $B2, $B3
    .BYTE $E4, $E5, $E6, $E7, $E8, $E9, $EA, $EB
    .BYTE $EC, $ED, $EE, $EF, $F0, $F1, $F2, $F3
    .BYTE $F4, $F5, $F6, $F7, $F8, $F9, $FA, $FB
    .BYTE $46, $47, $48, $49, $4A, $4B, $4C, $4D
    .BYTE $4E, $4F, $50, $51, $52, $53, $54, $55
    .BYTE $56, $57, $58, $59

PeaceText:
    .BYTE $0F, $12, $17, $0A, $15, $15, $22, $28
    .BYTE $19, $0E, $0A, $0C, $0E, $24, $1B, $0E
    .BYTE $1D, $1E, $1B, $17, $1C, $24, $1D, $18
    .BYTE $24, $11, $22, $1B, $1E, $15, $0E, $2C
    .BYTE $1D, $11, $12, $1C, $24, $0E, $17, $0D
    .BYTE $1C, $24, $1D, $11, $0E, $24, $1C, $1D
    .BYTE $18, $1B, $22, $2C, $FF

UpdatePeaceTextbox:
    ; Only emit a character once every 8 frames --
    ; when (counter [0412] MOD 8) = 4.
    INC PeaceCharDelayCounter
    LDA PeaceCharDelayCounter
    AND #$07
    CMP #$04
    BNE @Exit

    ; Copy the 5 bytes of the textbox character transfer record template
    ; to the dynamic transfer buf.
    LDY #$04
:
    LDA PeaceTextboxCharTransferRecTemplate, Y
    STA DynTileBuf, Y
    DEY
    BPL :-

    ; Load a character from the string and copy it to
    ; the transfer record until we read character $FF.
    LDY PeaceCharIndex
    LDA PeaceText, Y
    CMP #$FF
    BEQ @AdvanceSubmode
    STA DynTileBuf+3

    ; A regular space takes as much time to emit as any character.
    ; But it makes no sound.
    CMP #$24
    BEQ :+
    LDA #$10                    ; "Heart taken/character" tune
    STA Tune0Request
:
    ; Point to the next character in the string.
    INC PeaceCharIndex

    ; Replace the low byte of the VRAM address with the one
    ; where the next character should be written.
    LDA PeaceTextboxCharAddrsLo, Y
    STA DynTileBuf+1

    ; Once the low VRAM address rolls over,
    ; increment the high address.
    CMP #$A0
    BCS @Exit
    LDA #$23
    STA DynTileBuf

@Exit:
    RTS

@AdvanceSubmode:
    INC GameSubmode

LAB7E_Exit:
    RTS

UpdateMode13WinGame_Sub4:
    JSR HideAllSprites

    ; Put the triforce in object slot 2 at location ($78, $88).
    LDX #$02
    LDA #$78
    STA ObjX, X
    LDA #$88
    STA ObjY, X
    LDA #$0E                    ; Triforce item ID
    JSR AnimateItemObject

    ; Reuse object slot 2 for the ash pile.
    LDX #$02
    LDA #$3E                    ; Ganon object type
    STA ObjType, X
    JSR DrawAshPile

    ; Don't let the player skip ahead for a little while.
    LDA ObjTimer
    BNE LAB7E_Exit

    ; If Start hasn't been pressed, then return.
    LDA ButtonsPressed
    AND #$10
    BEQ LAB7E_Exit

    ; Start was pressed. We'll transition to mode $D to save.
    JSR EndGameMode
    LDA #$0D
    STA GameMode
    JSR TurnOffAllVideo
    JSR TurnOffVideoAndClearArtifacts
    JSR SilenceAllSound
    JMP SwitchProfileToSecondQuest

DrawAshPile:
    JSR Anim_FetchObjPosForSpriteDescriptor
    LDA #$0B                    ; Ganon-ashes frame image
    JMP DrawObjectNotMirrored
