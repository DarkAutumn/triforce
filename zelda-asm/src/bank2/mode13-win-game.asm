; ============================================================================
; FILE:     bank2/mode13-win-game.asm    BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 3405-3670
; SUBJECTS: Ending, Mode13WinGame
; TOPIC:    Mode $13 (win game) update: background flash with Link and Zelda holding Triforce pieces,
;           the "FINALLY,PEACE RETURNS TO HYRULE.THIS ENDS THE STORY." text, and the final screen
;           that waits for Start, then switches the profile to Q2 and goes to mode $D (save).
; SEE ALSO: bank2/credits.asm (Mode13WinGame_Update_Sub3Credits, the credits scroll between submodes 2 and 4);
;           bank2/mode13-ending-init.asm (the init submodes that run first);
;           bank2/second-quest-bank-stub.asm (Profile_SwitchToSecondQuest); bank2/moded-save.asm (mode
;           $D); bank7/update-mode-dispatch.asm (the jump table entry that reaches this file).
; KEY FACTS:
;   - Mode13WinGame_Update dispatches on GameSubmode: 0 flash, 1 text, 2 wait, 3 credits, 4 wait for
;     Start. Submodes 1 and 2 share Mode13WinGame_Update_Sub1PeaceText (Mode13WinGame_Update_JumpTable).
;   - Submode 0 ends when ItemLiftTimer, +1 per frame, reaches $C0 ($C0 frames if it enters at 0;
;     its start value is not set in this file, (unverified)); from count $40 on the background colour
;     changes every frame through Ending_FlashColors. Its last frame requests song $10 (ending),
;     sets ObjTimer = $40 and EndingPeaceLongTimer = $40. Mode13WinGame_Update_Sub0Flash
;   - Text: after the $40 frame delay, one character every 8 frames, 52 characters (Ending_PeaceText),
;     at the nametable addresses in Ending_PeaceCharAddrsLo. Ending_UpdatePeaceTextbox
;   - Submodes 1 and 2 together end when EndingPeaceLongTimer reaches 0; Link and Zelda stop being
;     drawn when it is below 4. Mode13WinGame_Update_Sub1PeaceText
;   - Submode 4 is the only place in mode $13 that reads the controller: once ObjTimer ($40 frames,
;     set at the end of the credits) has expired, Start ($10 in ButtonsPressed) ends the mode.
;     Mode13WinGame_Update_Sub4WaitStart
;   - Leaving through Start always sets GameMode = $D and calls Profile_SwitchToSecondQuest, also
;     when the finished quest was already Q2. Mode13WinGame_Update_Sub4WaitStart
;   - Triforce pieces (item $1B) are drawn $10 pixels above Link (object slot $13) and above Zelda
;     (slot 1, piece in slot 2). Ending_DrawLinkZeldaTriforces
; CONTENTS: (generated; L = line in this file)
;   L0049 Mode13WinGame_Update                 code  Mode $13 (win game) update: runs the handler of the current GameSubmode (0-4).
;   L0054 Mode13WinGame_Update_JumpTable       data  TABLE: mode $13 update handlers; index = GameSubmode; value = routine address (1 and 2 share one).
;   L0062 Ending_FlashColors                   data  TABLE: background colours of the ending flash; index = ItemLiftTimer AND 3; value = NES colour.
;   L0070 Mode13WinGame_Update_Sub0Flash       code  Update submode 0: Link and Zelda hold Triforce pieces until ItemLiftTimer = $C0; flashes from $40.
;   L0127 Ending_DrawLinkZeldaTriforces        code  Draws Link and Zelda, each with a Triforce piece (item $1B) $10 pixels above.
;   L0179 Mode13WinGame_Update_Sub1PeaceText   code  Update submodes 1 and 2: draws the pair, types the peace text (submode 1 only), ends on the timer.
;   L0213 Ending_PeaceCharTransferRecTemplate  data  DATA: transfer record template for one text character: address $22A4, count 1, tile $24, end $FF.
;   L0218 Ending_PeaceCharAddrsLo              data  TABLE: low nametable address byte of each character; index = position in Ending_PeaceText (0-51); the
;   L0229 Ending_PeaceText                     data  DATA: the peace text, 52 tile codes then $FF: "FINALLY," / "PEACE RETURNS TO HYRULE." /
;   L0242 Ending_UpdatePeaceTextbox            code  Every 8th frame writes the next Ending_PeaceText character to the screen; at $FF goes to submode 2.
;   L0297 Ending_UpdatePeaceTextbox_Return     code  Shared exit of Ending_UpdatePeaceTextbox and Mode13WinGame_Update_Sub4WaitStart.
;   L0305 Mode13WinGame_Update_Sub4WaitStart   code  Update submode 4: draws the Triforce of Power and Ganon's ashes; Start saves and begins Q2.
;   L0345 Ending_DrawAshPile                   code  Draws the ash pile (frame $0B, not mirrored) for the object in slot X.
; ============================================================================

; ----------------------------------------------------------------------------
; Mode $13 (win game) update: runs the handler of the current GameSubmode (0-4).
; ----------------------------------------------------------------------------
Mode13WinGame_Update:
    LDA GameSubmode
    JSR Util_TableJump

; TABLE: mode $13 update handlers; index = GameSubmode; value = routine address (1 and 2 share one).
Mode13WinGame_Update_JumpTable:
    .ADDR Mode13WinGame_Update_Sub0Flash
    .ADDR Mode13WinGame_Update_Sub1PeaceText
    .ADDR Mode13WinGame_Update_Sub1PeaceText
    .ADDR Mode13WinGame_Update_Sub3Credits
    .ADDR Mode13WinGame_Update_Sub4WaitStart

; TABLE: background colours of the ending flash; index = ItemLiftTimer AND 3; value = NES colour.
Ending_FlashColors:
    .BYTE $0F, $12, $16, $2A

; ----------------------------------------------------------------------------
; Update submode 0: Link and Zelda hold Triforce pieces until ItemLiftTimer = $C0; flashes from $40.
; OUT:  at count $C0: SongRequest = $10 (ending song), ObjTimer = $40, EndingPeaceLongTimer = $40,
;       GameSubmode = 1.
; ----------------------------------------------------------------------------
Mode13WinGame_Update_Sub0Flash:
    JSR Sprite_HideAll
    INC ItemLiftTimer
    LDA ItemLiftTimer
    CMP #$C0
    BEQ @AdvanceSubmode
    JSR Ending_DrawLinkZeldaTriforces

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
    LDA Ending_FlashColors, X
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

; ----------------------------------------------------------------------------
; Draws Link and Zelda, each with a Triforce piece (item $1B) $10 pixels above.
; NOTE: uses object slot $13 for Link's piece and slot 2 for Zelda's (Zelda is slot 1).
; ----------------------------------------------------------------------------
Ending_DrawLinkZeldaTriforces:
    ; The Triforce piece over Link goes in the room object slot $13.
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

    ; Draw the Triforce piece over Link.
    LDA #$1B
    LDX #$13
    JSR Item_DrawObject
    LDX #$01
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; Draw Zelda.
    TXA
    JSR Anim_DrawObjectMirrored

    ; The Triforce piece over Zelda goes in slot 2.
    ; Set its location $10 pixels above her.
    LDA ObjX+1
    STA ObjX+2
    LDA ObjY+1
    SEC
    SBC #$10
    STA ObjY+2

    ; Draw the Triforce piece over Zelda.
    LDX #$02
    LDA #$1B
    JSR Item_DrawObject
    RTS

; ----------------------------------------------------------------------------
; Update submodes 1 and 2: draws the pair, types the peace text (submode 1 only), ends on the timer.
; OUT:  when EndingPeaceLongTimer = 0: TileBufSelector = $6A (ending palette), GameSubmode + 1.
; ----------------------------------------------------------------------------
Mode13WinGame_Update_Sub1PeaceText:
    LDA EndingPeaceLongTimer
    BEQ @AdvanceSubmode
    JSR Sprite_HideAll

    ; Characters stop emitting when the long timer is $10.
    ; So, keep showing Link and Zelda until it reaches 4.
    ; Then hide them and wait until it reaches 0.
    LDA EndingPeaceLongTimer
    CMP #$04
    BCC @Exit
    JSR Ending_DrawLinkZeldaTriforces

    ; Once in submode 2, only delay, instead of emitting characters.
    LDA GameSubmode
    CMP #$01
    BNE @Exit

    ; There is a delay before showing text.
    LDA ObjTimer
    BNE @Exit
    JSR Ending_UpdatePeaceTextbox

@Exit:
    RTS

@AdvanceSubmode:
    ; Transfer the ending palette, and go to the next submode.
    LDA #$6A
    STA TileBufSelector
    INC GameSubmode
    RTS

; DATA: transfer record template for one text character: address $22A4, count 1, tile $24, end $FF.
Ending_PeaceCharTransferRecTemplate:
    .BYTE $22, $A4, $01, $24, $FF

; TABLE: low nametable address byte of each character; index = position in Ending_PeaceText (0-51); the
; high byte is $22, or $23 where the value is below $A0. Three lines of 8, 24 and 20 characters.
Ending_PeaceCharAddrsLo:
    .BYTE $AC, $AD, $AE, $AF, $B0, $B1, $B2, $B3
    .BYTE $E4, $E5, $E6, $E7, $E8, $E9, $EA, $EB
    .BYTE $EC, $ED, $EE, $EF, $F0, $F1, $F2, $F3
    .BYTE $F4, $F5, $F6, $F7, $F8, $F9, $FA, $FB
    .BYTE $46, $47, $48, $49, $4A, $4B, $4C, $4D
    .BYTE $4E, $4F, $50, $51, $52, $53, $54, $55
    .BYTE $56, $57, $58, $59

; DATA: the peace text, 52 tile codes then $FF: "FINALLY," / "PEACE RETURNS TO HYRULE." /
; "THIS ENDS THE STORY." ($24 space; glyphs of $28 and $2C (unverified): comma, period).
Ending_PeaceText:
    .BYTE $0F, $12, $17, $0A, $15, $15, $22, $28
    .BYTE $19, $0E, $0A, $0C, $0E, $24, $1B, $0E
    .BYTE $1D, $1E, $1B, $17, $1C, $24, $1D, $18
    .BYTE $24, $11, $22, $1B, $1E, $15, $0E, $2C
    .BYTE $1D, $11, $12, $1C, $24, $0E, $17, $0D
    .BYTE $1C, $24, $1D, $11, $0E, $24, $1C, $1D
    .BYTE $18, $1B, $22, $2C, $FF

; ----------------------------------------------------------------------------
; Every 8th frame writes the next Ending_PeaceText character to the screen; at $FF goes to submode 2.
; NOTE: each character except a space requests tune $10 on square 0.
; ----------------------------------------------------------------------------
Ending_UpdatePeaceTextbox:
    ; Only emit a character once every 8 frames --
    ; when (PeaceCharDelayCounter MOD 8) = 4.
    INC PeaceCharDelayCounter
    LDA PeaceCharDelayCounter
    AND #$07
    CMP #$04
    BNE @Exit

    ; Copy the 5 bytes of the textbox character transfer record template
    ; to the dynamic transfer buf.
    LDY #$04
:
    LDA Ending_PeaceCharTransferRecTemplate, Y
    STA DynTileBuf, Y
    DEY
    BPL :-

    ; Load a character from the string and copy it to
    ; the transfer record until we read character $FF.
    LDY PeaceCharIndex
    LDA Ending_PeaceText, Y
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
    LDA Ending_PeaceCharAddrsLo, Y
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

; Shared exit of Ending_UpdatePeaceTextbox and Mode13WinGame_Update_Sub4WaitStart.
Ending_UpdatePeaceTextbox_Return:
    RTS

; ----------------------------------------------------------------------------
; Update submode 4: draws the Triforce of Power and Ganon's ashes; Start saves and begins Q2.
; NOTE: Start is ignored until ObjTimer expires. Then GameMode = $D and the profile is reset by
;       Profile_SwitchToSecondQuest.
; ----------------------------------------------------------------------------
Mode13WinGame_Update_Sub4WaitStart:
    JSR Sprite_HideAll

    ; Put the Triforce of Power in object slot 2 at location ($78, $88).
    LDX #$02
    LDA #$78
    STA ObjX, X
    LDA #$88
    STA ObjY, X
    LDA #$0E                    ; Triforce item ID
    JSR Item_DrawObject

    ; Reuse object slot 2 for the ash pile.
    LDX #$02
    LDA #$3E                    ; Ganon object type
    STA ObjType, X
    JSR Ending_DrawAshPile

    ; Start is ignored until ObjTimer ($40 frames, set when the credits end) expires.
    LDA ObjTimer
    BNE Ending_UpdatePeaceTextbox_Return

    ; If Start hasn't been pressed, then return.
    LDA ButtonsPressed
    AND #$10
    BEQ Ending_UpdatePeaceTextbox_Return

    ; Start was pressed. We'll transition to mode $D to save.
    JSR Mode_End
    LDA #$0D
    STA GameMode
    JSR Ppu_TurnOffAllVideo
    JSR Ppu_TurnOffVideoAndClearNameTables
    JSR Audio_SilenceAll
    JMP Profile_SwitchToSecondQuest

; ----------------------------------------------------------------------------
; Draws the ash pile (frame $0B, not mirrored) for the object in slot X.
; IN:   X = object slot.
; ----------------------------------------------------------------------------
Ending_DrawAshPile:
    JSR Anim_FetchObjPosForSpriteDescriptor
    LDA #$0B                    ; Ganon-ashes frame image
    JMP Anim_DrawObjectNotMirrored
