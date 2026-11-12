; ============================================================================
; FILE:     bank2/mode1-menu-update.asm  BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 2735-2913
; SUBJECTS: (none)
; TOPIC:    File-select menu update, Start loads profile
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0018 Mode1CursorSpriteTriplet   data
;   L0021 Mode1CursorSpriteYs        data
;   L0024 UpdateMode1Menu            code
;   L0028 UpdateMode1Menu_JumpTable  data
;   L0032 UpdateMode1Menu_Sub0       code
;   L0078 UpdateMode1Menu_Sub1       code
;   L0132 Mode1_WriteLinkSprites     code
; ============================================================================

Mode1CursorSpriteTriplet:
    .BYTE $F3, $03, $28

Mode1CursorSpriteYs:
    .BYTE $5C, $74, $8C, $A8, $B8

UpdateMode1Menu:
    LDA GameSubmode
    JSR TableJump

UpdateMode1Menu_JumpTable:
    .ADDR UpdateMode1Menu_Sub0
    .ADDR UpdateMode1Menu_Sub1

UpdateMode1Menu_Sub0:
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
    LDA Mode1CursorSpriteTriplet, Y
    STA Sprites+1, Y
    DEY
    BPL @WriteCursorSprite
    LDY CurSaveSlot
    LDA Mode1CursorSpriteYs, Y
    STA Sprites                 ; Set the sprite Y for current option.
    LDA #$58                    ; The base Y of Link sprites is $58.
    STA $01
    LDA #$30                    ; The X coordinate of Link sprites is $30.
    STA $00
    JMP Mode1_WriteLinkSprites

@Exit:
    INC GameSubmode
    RTS

UpdateMode1Menu_Sub1:
    LDA #$00
    STA Tune1
    LDA #$00
    STA CurLevel                ; Begin in OW.
    STA SelectedItemSlot        ; Reset item index.
    JSR TurnOffAllVideo
    LDA CurSaveSlot
    CMP #$03
    BCC @ChoseSlot              ; If not a save slot,
    LDA CurSaveSlot             ; Go to the mode for each option (register, eliminate).
    CLC
    ADC #$0B
    STA GameMode
    JMP EndGameMode

@ChoseSlot:
    ; The player chose a save slot.
    JSR TurnOffAllVideo
    JSR FetchFileAAddressSet
    LDY #$27                    ; Copy Items block from file A to profile.

@CopyItems:
    LDA ($00), Y
    STA Items, Y
    DEY
    BPL @CopyItems
    LDA #$00
    STA SwordBlocked
    STA ObjState                ; Reset player state.
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
    JMP GoToNextMode

Mode1_WriteLinkSprites:
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
    JSR Anim_WriteSpritePairNotFlashing    ; We didn't set [$08]. But we don't care if CurSpriteIndex is cycled.
    TAX
    PLA                         ; Restore the X coordinate.
    STA $00
    LDY $04                     ; Use [$04] as a save slot number.
    LDA QuestNumbers, Y
    BEQ @NextSlot               ; If in first quest, then skip the second quest marker.
    LDY LeftSpriteOffset
    LDA $01                     ; Put the sword 3 pixels below Link.
    SEC
    SBC #$03
    STA Sprites+128, Y          ; Use sprites $20 to $23 for the swords.
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
