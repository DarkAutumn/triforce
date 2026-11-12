; ============================================================================
; FILE:     bank2/mode13-ending-init.asm BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 3203-3403
; SUBJECTS: (none)
; TOPIC:    Mode $13 init: curtain, Zelda's textbox
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0026 InitMode13_Full                       code
;   L0030 InitMode13_Full_JumpTable             data
;   L0037 InitMode13_Sub0                       code
;   L0046 UpdateEndGameCurtainEffect            code
;   L0068 PlayAreaAttr0TransferBuf              data
;   L0071 InitMode13_Sub1                       code
;   L0088 LA958_Exit                            code
;   L0091 ThanksText                            data
;   L0098 InitMode13_Sub2                       code
;   L0110 ThanksTextboxCharTransferRecTemplate  data
;   L0113 ThanksTextboxLineAddrsLo              data
;   L0119 UpdateZeldaTextbox                    code  Returns:
;   L0210 LA9F4_Exit                            code
;   L0213 InitMode13_Sub3                       code
;   L0220 InitMode13_Sub4                       code
; ============================================================================

InitMode13_Full:
    LDA GameSubmode
    JSR TableJump

InitMode13_Full_JumpTable:
    .ADDR InitMode13_Sub0
    .ADDR InitMode13_Sub1
    .ADDR InitMode13_Sub2
    .ADDR InitMode13_Sub3
    .ADDR InitMode13_Sub4

InitMode13_Sub0:
    JSR UpdateEndGameCurtainEffect
    LDA GameSubmode
    BEQ LA958_Exit
    JSR HideAllSprites
    JSR Link_EndMoveAndDraw
    LDX #$01                    ; Zelda is in object slot 1.
    JMP Person_Draw

UpdateEndGameCurtainEffect:
    LDA ObjTimer
    BNE @Exit
    LDA CurSong
    BNE @Exit
    JSR UpdateWorldCurtainEffect_Bank2

    ; When the right curtain edge reaches the middle (< $11),
    ; go to the next submode.
    LDA ObjX+12
    CMP #$11
    BCS @Exit

    ; UNKNOWN: This assignment looks unneeded.
    ; Nothing looks at this timer until it's set again in update/submode 0.
    LDA #$80
    STA ObjTimer
    INC GameSubmode

@Exit:
    RTS

PlayAreaAttr0TransferBuf:
    .BYTE $23, $D8, $68, $00, $FF

InitMode13_Sub1:
    LDY #$04
:
    LDA PlayAreaAttr0TransferBuf, Y
    STA DynTileBuf, Y
    DEY
    BPL :-

    ; Point to the front of the first textbox line,
    ; and reset the current character index.
    LDA #$A4
    STA TextboxCharVramLo
    LDA #$00
    STA TextboxCharIndex
    STA ObjState+1
    INC GameSubmode

LA958_Exit:
    RTS

ThanksText:
    .BYTE $1D, $11, $0A, $17, $14, $1C, $24, $15
    .BYTE $12, $17, $14, $28, $22, $18, $1E, $2A
    .BYTE $1B, $8E, $64, $1D, $11, $0E, $24, $11
    .BYTE $0E, $1B, $18, $24, $18, $0F, $24, $11
    .BYTE $22, $1B, $1E, $15, $0E, $EC

InitMode13_Sub2:
    JSR UpdateZeldaTextbox
    LDA ObjState+1
    BEQ :+

    ; Set a delay of $50 frames in the next submode.
    LDA #$50
    STA ObjTimer+1
    INC GameSubmode
:
    RTS

ThanksTextboxCharTransferRecTemplate:
    .BYTE $21, $A4, $01, $24, $FF

ThanksTextboxLineAddrsLo:
    .BYTE $C4, $E4, $A4

; Returns:
; ObjState[1]: 0 if still updating
;
UpdateZeldaTextbox:
    JSR Link_EndMoveAndDraw

    ; If Zelda's timer has not expired, then return.
    LDA ObjTimer+1
    BNE LA9F4_Exit

    ; Set the timer to wait 6 frames after the next character about
    ; to be shown.
    LDA #$06
    STA ObjTimer+1

    ; Copy the 5 bytes of the textbox character transfer record template
    ; to the dynamic transfer buf.
    LDY #$04
:
    LDA ThanksTextboxCharTransferRecTemplate, Y
    STA DynTileBuf, Y
    DEY
    BPL :-
:
    ; Replace the low byte of the VRAM address with the one
    ; where the next character should be written.
    LDA TextboxCharVramLo
    STA DynTileBuf+1

    ; Increment the low VRAM address for the next character.
    INC TextboxCharVramLo
    LDA #<ThanksText
    STA $00
    LDA #>ThanksText
    STA $01

    ; Load the person text current character index.
    LDY TextboxCharIndex

    ; Increment the index variable to point to the next character
    ; for the next time.
    INC TextboxCharIndex

    ; Get the current character.
    LDA ($00), Y

    ; If the character is $25, then it's a special space. It will still
    ; take up space, but will not take time to show -- meaning that
    ; we'll go look up the next character to transfer.
    AND #$3F
    CMP #$25
    BEQ :-

    ; We have a non-space character. Put it in the transfer record.
    STA DynTileBuf+3

    ; Play the "heart taken/character" tune.
    LDA #$10
    STA Tune0Request

    ; If the high 2 bits of character element = 0, then return.
    LDA ($00), Y
    AND #$C0
    BEQ LA9F4_Exit

    ; Determine an index based on the high 2 bits of the character element:
    ;   $80: 0
    ;   $40: 1
    ;   $C0: 2
    LDY #$02
    CMP #$C0
    BEQ :+
    DEY
    CMP #$40
    BEQ :+
    DEY
:
    ; The index chooses the low VRAM address of the start
    ; of another line:
    ;   0: $C4: front of the second line
    ;   1: $E4: front of the third line
    ;   2: $A4: front of the first line
    LDA ThanksTextboxLineAddrsLo, Y
    STA TextboxCharVramLo

    ; If index = 2, then we've reached the end of the text,
    ; and low VRAM address is moved to the front of the first line.
    ; So, advance the state of the person object, and unhalt Link.
    CPY #$02
    BNE LA9F4_Exit
    INC ObjState+1
    LDA #$00
    STA ObjState

LA9F4_Exit:
    RTS

InitMode13_Sub3:
    LDA ObjTimer+1
    BNE LA9F4_Exit
    JSR SilenceAllSound
    INC GameSubmode
    RTS

InitMode13_Sub4:
    LDA #$08
    STA CreditsTileOffset
    JSR BeginUpdateMode
    STA PeaceCharDelayCounter
    STA PeaceCharIndex
    JMP HideAllSprites
