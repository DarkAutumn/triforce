; ============================================================================
; FILE:     bank2/mode1-menu-update.asm  BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 2735-2913
; SUBJECTS: Mode1Menu
; TOPIC:    Mode 1 (file select) update: Select moves the cursor, Start either loads the chosen save
;           slot into the profile and goes on to the next mode, or enters Register / Eliminate.
; SEE ALSO: bank2/mode1-slot-tiles.asm and bank2/mode1-init-save-files.asm (menu init);
;           bank1/modeb-cave-save-sprites.asm (Save_FetchFileAAddressSet); bank2/register-eliminate-init.asm
;           (modes $E and $F); bank7/update-mode-dispatch.asm (Mode_GoToNext, Mode_End).
; KEY FACTS:
;   - CurSaveSlot is the menu choice: 0-2 save slots, 3 Register, 4 Eliminate. Select steps to the next
;     choice whose IsSaveSlotActive entry is nonzero, so unregistered slots are skipped
;     (Mode1Menu_Update_Sub0Choose).
;   - Start on Register or Eliminate sets GameMode to CurSaveSlot + $B = $E or $F
;     (Mode1Menu_Update_Sub1Load).
;   - Start on a slot copies file A's items block ($28 bytes) to InvItems and its world flags ($180
;     bytes) to WorldFlags, then goes to the next mode (Mode1Menu_Update_Sub1Load).
;   - Loading a slot zeroes IsSwordBlocked, Link's ObjState and InvClock; Start in the menu also sets
;     CurLevel = 0 (overworld) and SelectedItemSlot = 0 (Mode1Menu_Update_Sub1Load).
;   - A slot in Q2 (QuestNumbers nonzero) shows a sword sprite beside its Link
;     (Mode1Menu_WriteLinkSprites).
; CONTENTS: (generated; L = line in this file)
;   L0033 Mode1Menu_CursorSpriteTriplet  data  DATA: tile, attributes and X of the menu cursor sprite (heart tile $F3 at X $28).
;   L0037 Mode1Menu_CursorSpriteYs       data  TABLE: Y of the menu cursor sprite; index = CurSaveSlot (0-2 save slots, 3 Register, 4 Eliminate).
;   L0043 Mode1Menu_Update               code  Mode 1 (file select) update: runs the handler selected by GameSubmode.
;   L0048 Mode1Menu_Update_JumpTable     data  TABLE: mode 1 update handlers; index = GameSubmode (0 choose, 1 act on the choice).
;   L0055 Mode1Menu_Update_Sub0Choose    code  Mode 1 submode 0: Select moves the cursor to the next enabled choice; Start goes to submode 1.
;   L0104 Mode1Menu_Update_Sub1Load      code  Mode 1 submode 1: loads the chosen save slot into the profile, or enters mode $E or $F.
;   L0162 Mode1Menu_WriteLinkSprites     code  Draws one Link per save slot down the menu, and a sword beside each slot that is in Q2.
; ============================================================================

; DATA: tile, attributes and X of the menu cursor sprite (heart tile $F3 at X $28).
Mode1Menu_CursorSpriteTriplet:
    .BYTE $F3, $03, $28

; TABLE: Y of the menu cursor sprite; index = CurSaveSlot (0-2 save slots, 3 Register, 4 Eliminate).
Mode1Menu_CursorSpriteYs:
    .BYTE $5C, $74, $8C, $A8, $B8

; ----------------------------------------------------------------------------
; Mode 1 (file select) update: runs the handler selected by GameSubmode.
; ----------------------------------------------------------------------------
Mode1Menu_Update:
    LDA GameSubmode
    JSR Util_TableJump

; TABLE: mode 1 update handlers; index = GameSubmode (0 choose, 1 act on the choice).
Mode1Menu_Update_JumpTable:
    .ADDR Mode1Menu_Update_Sub0Choose
    .ADDR Mode1Menu_Update_Sub1Load

; ----------------------------------------------------------------------------
; Mode 1 submode 0: Select moves the cursor to the next enabled choice; Start goes to submode 1.
; ----------------------------------------------------------------------------
Mode1Menu_Update_Sub0Choose:
    LDA ButtonsPressed
    AND #$10
    BNE @Exit                   ; If Start was pressed, the go to the next submode.
    LDA ButtonsPressed
    AND #$20
    BEQ :+                      ; If Select was pressed,

@ChangeSelection:
    LDA #$01                    ; Request to play the selection change SFX (same as rupee taken).
    STA Tune1Request
    INC CurSaveSlot             ; Select the next slot or option.
    LDA CurSaveSlot
    CMP #$05
    BNE :+                      ; If the index is out of bounds,
    LDA #$00                    ; then wrap around to zero.
    STA CurSaveSlot
:
    LDY CurSaveSlot

    ; Since CurSaveSlot is used to index menu choices, which
    ; includes register and eliminate in addition to save slots;
    ; the IsSaveSlotActive array includes elements at the end
    ; for these options.
    LDA IsSaveSlotActive, Y
    BEQ @ChangeSelection        ; If option or save slot isn't active, then go advance the index.
    LDY #$02                    ; Write the tile, attributes, and X for the sprite record.

@WriteCursorSprite:
    LDA Mode1Menu_CursorSpriteTriplet, Y
    STA Sprites+1, Y
    DEY
    BPL @WriteCursorSprite
    LDY CurSaveSlot
    LDA Mode1Menu_CursorSpriteYs, Y
    STA Sprites                 ; Set the sprite Y for current option.
    LDA #$58                    ; The base Y of Link sprites is $58.
    STA $01
    LDA #$30                    ; The X coordinate of Link sprites is $30.
    STA $00
    JMP Mode1Menu_WriteLinkSprites

@Exit:
    INC GameSubmode
    RTS

; ----------------------------------------------------------------------------
; Mode 1 submode 1: loads the chosen save slot into the profile, or enters mode $E or $F.
; ----------------------------------------------------------------------------
Mode1Menu_Update_Sub1Load:
    LDA #$00
    STA CurTune1
    LDA #$00
    STA CurLevel                ; Begin in OW.
    STA SelectedItemSlot        ; Reset item index.
    JSR Ppu_TurnOffAllVideo
    LDA CurSaveSlot
    CMP #$03
    BCC @ChoseSlot              ; If not a save slot,
    LDA CurSaveSlot             ; Go to the mode for each option (register, eliminate).
    CLC
    ADC #$0B
    STA GameMode
    JMP Mode_End

@ChoseSlot:
    ; A save slot was chosen.
    JSR Ppu_TurnOffAllVideo
    JSR Save_FetchFileAAddressSet
    LDY #$27                    ; Copy Items block from file A to profile.

@CopyItems:
    LDA ($00), Y
    STA InvItems, Y
    DEY
    BPL @CopyItems
    LDA #$00
    STA IsSwordBlocked
    STA ObjState                ; Reset Link's state.
    STA InvClock                ; Reset clock item.

    ; Copy WorldFlags block from file A to profile.
    TAY

@CopyWorldFlags:
    LDA ($02), Y
    STA ($0E), Y
    INC $02
    BNE :+
    INC $03
:
    INC $0E
    BNE :+
    INC $0F
:
    LDA $0E
    CMP #$FF
    BNE @CopyWorldFlags
    LDA $0F
    CMP #$07
    BNE @CopyWorldFlags         ; If not done (destination address < $07FF), then copy more.
    JMP Mode_GoToNext

; ----------------------------------------------------------------------------
; Draws one Link per save slot down the menu, and a sword beside each slot that is in Q2.
; IN:   [00] = X, [01] = Y of the first Link.
; ----------------------------------------------------------------------------
Mode1Menu_WriteLinkSprites:
    LDA #$08                    ; Put the left tile of Link in [$02].
    STA $02
    LDA #$0A                    ; Put the right tile of Link in [$03].
    STA $03

    ; [$04] is used as an index and sprite attributes.
    ; It takes on values 0 to 2.
    ; As an index, it represents a save slot.
    ; As attributes, these values represent palettes 4 to 6.
    LDA #$00
    JSR Anim_SetSpriteDescriptorAttributes

    ; We want to start with sprite 4 (offset $10).
    ; Begin with 8, so that the loop will add 8 and
    ; put us at the offset we want.
    LDA #$08
    STA LeftSpriteOffset

@LoopSlot:
    LDA LeftSpriteOffset
    CLC
    ADC #$08                    ; Each Link is two sprites (8 bytes).
    STA LeftSpriteOffset
    CLC
    ADC #$04                    ; The right side is the next sprite.
    STA RightSpriteOffset
    LDA #$01                    ; This object has two sides (sprites).
    STA $07
    LDA #$08                    ; The two sides are 8 pixels apart.
    STA $0A
    LDA $00                     ; Save the X coordinate.
    PHA
    JSR Anim_WriteSpritePairNotFlashing    ; [08] is not set; cycling CurSpriteIndex does not matter here.
    TAX
    PLA                         ; Restore the X coordinate.
    STA $00
    LDY $04                     ; Use [$04] as a save slot number.
    LDA QuestNumbers, Y
    BEQ @NextSlot               ; If in first quest, then skip the second quest marker.
    LDY LeftSpriteOffset
    LDA $01                     ; Put the sword 3 pixels above the Y in [01].
    SEC
    SBC #$03
    STA Sprites+128, Y          ; Sprite offsets $90, $98, $A0 (sprites $24, $26, $28) hold the swords.
    LDA #$20                    ; Use sword tiles.
    STA Sprites+129, Y
    LDA #$03                    ; Use palette 7.
    STA Sprites+130, Y
    LDA $00                     ; Put the sword $C pixels to the right of Link.
    CLC
    ADC #$0C
    STA Sprites+131, Y

@NextSlot:
    LDA $01                     ; Move Y down for the next slot.
    CLC
    ADC #$18
    STA $01
    INC $04                     ; Look at the next slot, and use the next palette.
    INC $05
    LDA $04
    CMP #$03
    BNE @LoopSlot               ; If we're not done, then look at the next slot.
    RTS
