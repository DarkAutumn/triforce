; ============================================================================
; FILE:     bank2/mode13-ending-init.asm BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 3203-3403
; SUBJECTS: (none)
; TOPIC:    Mode $13 (win game) init, five submodes: the curtain effect runs over Zelda's room, the
;           play area attributes are reset, Zelda's "THANKS LINK,YOU'RE THE HERO OF HYRULE." is
;           typed out, then a pause, silence, and the hand-over to the mode's update.
; SEE ALSO: bank2/mode13-win-game.asm (the update submodes that follow); bank2/credits.asm (the credits
;           scroll, update submode 3); bank1/map-marker-curtain.asm (UpdateWorldCurtainEffect_Bank2);
;           bank7/init-mode-dispatch.asm (the trampoline that switches to bank 2 and jumps here);
;           bank1/person-textbox.asm (the cave textbox, same record format and 6-frame pacing).
; KEY FACTS:
;   - InitMode13_Full dispatches on GameSubmode through InitMode13_Full_JumpTable (submodes 0-4).
;   - Submode 0 waits for ObjTimer (slot 0) to expire and the current song to end (CurSong = 0)
;     before the curtain moves; the curtain is done when column index ObjX+12 drops below $11.
;     UpdateEndGameCurtainEffect
;   - Zelda is object slot 1 throughout: Person_Draw with X = 1 (InitMode13_Sub0); her ObjTimer+1
;     paces the text and her ObjState+1 becomes nonzero when the text is complete.
;   - Text speed: one character per 6 frames (UpdateZeldaTextbox); 38 bytes in ThanksText, one of
;     them a space code that carries the jump to the third line.
;   - ThanksText bytes: low 6 bits = tile code, high 2 bits = what follows the character
;     ($80 go to line 2, $40 go to line 3, $C0 end of text). Lines start at nametable $21A4, $21C4,
;     $21E4 (ThanksTextboxLineAddrsLo).
;   - After the text: $50 frames of delay (InitMode13_Sub2, InitMode13_Sub3), all sound silenced,
;     then InitMode13_Sub4 sets CreditsTileOffset to 8 and calls BeginUpdateMode.
;   - No controller input is read anywhere in this file (InitMode13_Full and its submodes).
; CONTENTS: (generated; L = line in this file)
;   L0049 InitMode13_Full                       code  Mode $13 (win game) init: runs the handler of the current GameSubmode (0-4).
;   L0054 InitMode13_Full_JumpTable             data  TABLE: mode $13 init handlers; index = GameSubmode; value = routine address.
;   L0064 InitMode13_Sub0                       code  Init submode 0: advances the closing curtain; on the frame it finishes, draws Link and Zelda.
;   L0078 UpdateEndGameCurtainEffect            code  Steps the curtain once ObjTimer and the current song have run out; when done, goes to submode 1.
;   L0102 PlayAreaAttr0TransferBuf              data  DATA: transfer record: $28 bytes of $00 (count byte $68 = repeat flag $40 + $28) to the nametable
;   L0108 InitMode13_Sub1                       code  Init submode 1: queues PlayAreaAttr0TransferBuf and resets the textbox position and Zelda's state.
;   L0126 LA958_Exit                            code  Shared exit of InitMode13_Sub0 and InitMode13_Sub1.
;   L0133 ThanksText                            data  DATA: Zelda's text, 38 bytes: low 6 bits = tile code ($00-$09 digits, $0A-$23 letters A-Z, $24
;   L0143 InitMode13_Sub2                       code  Init submode 2: types Zelda's text; when it is complete, starts a $50 frame delay and advances.
;   L0156 ThanksTextboxCharTransferRecTemplate  data  DATA: transfer record template for one textbox character: address $21A4, count 1, tile $24, end $FF.
;   L0161 ThanksTextboxLineAddrsLo              data  TABLE: low nametable address byte of a text line's first character; index = 0 line 2, 1 line 3,
;   L0169 UpdateZeldaTextbox                    code  Draws Link and, every 6 frames, writes the next character of ThanksText to the screen.
;   L0261 LA9F4_Exit                            code  Shared exit of UpdateZeldaTextbox and InitMode13_Sub3.
;   L0267 InitMode13_Sub3                       code  Init submode 3: when Zelda's timer (the $50 frame delay) expires, silences all sound and advances.
;   L0278 InitMode13_Sub4                       code  Init submode 4: presets the credits scroll and text counters, ends init, and hides all sprites.
; ============================================================================

; ----------------------------------------------------------------------------
; Mode $13 (win game) init: runs the handler of the current GameSubmode (0-4).
; NOTE: reached from the fixed bank's mode-init trampoline with bank 2 switched in.
; ----------------------------------------------------------------------------
InitMode13_Full:
    LDA GameSubmode
    JSR TableJump

; TABLE: mode $13 init handlers; index = GameSubmode; value = routine address.
InitMode13_Full_JumpTable:
    .ADDR InitMode13_Sub0
    .ADDR InitMode13_Sub1
    .ADDR InitMode13_Sub2
    .ADDR InitMode13_Sub3
    .ADDR InitMode13_Sub4

; ----------------------------------------------------------------------------
; Init submode 0: advances the closing curtain; on the frame it finishes, draws Link and Zelda.
; ----------------------------------------------------------------------------
InitMode13_Sub0:
    JSR UpdateEndGameCurtainEffect
    LDA GameSubmode
    BEQ LA958_Exit
    JSR HideAllSprites
    JSR Link_EndMoveAndDraw
    LDX #$01                    ; Zelda is in object slot 1.
    JMP Person_Draw

; ----------------------------------------------------------------------------
; Steps the curtain once ObjTimer and the current song have run out; when done, goes to submode 1.
; OUT:  GameSubmode incremented and ObjTimer = $80 once ObjX+12 (curtain column index) < $11.
; NOTE: the curtain routine reloads ObjTimer with 5 after each step, which spaces the steps.
; ----------------------------------------------------------------------------
UpdateEndGameCurtainEffect:
    LDA ObjTimer
    BNE @Exit
    LDA CurSong
    BNE @Exit
    JSR UpdateWorldCurtainEffect_Bank2

    ; When the decreasing curtain column index is below $11,
    ; go to the next submode.
    LDA ObjX+12
    CMP #$11
    BCS @Exit

    ; No code in this file reads this ObjTimer (slot 0) value: init submodes 1-4 use ObjTimer+1,
    ; and update submode 0 sets ObjTimer again before update submode 1 reads it.
    LDA #$80
    STA ObjTimer
    INC GameSubmode

@Exit:
    RTS

; DATA: transfer record: $28 bytes of $00 (count byte $68 = repeat flag $40 + $28) to the nametable
; attributes $23D8-$23FF, then the end marker $FF.
PlayAreaAttr0TransferBuf:
    .BYTE $23, $D8, $68, $00, $FF

; ----------------------------------------------------------------------------
; Init submode 1: queues PlayAreaAttr0TransferBuf and resets the textbox position and Zelda's state.
; ----------------------------------------------------------------------------
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

; Shared exit of InitMode13_Sub0 and InitMode13_Sub1.
LA958_Exit:
    RTS

; DATA: Zelda's text, 38 bytes: low 6 bits = tile code ($00-$09 digits, $0A-$23 letters A-Z, $24
; space; $28, $2A, $2C punctuation, glyphs (unverified): comma, apostrophe, period); high 2 bits =
; $80 next goes to line 2, $40 to line 3, $C0 end of text.
; Reads "THANKS LINK,YOU'RE" / (one space) / "THE HERO OF HYRULE."
ThanksText:
    .BYTE $1D, $11, $0A, $17, $14, $1C, $24, $15
    .BYTE $12, $17, $14, $28, $22, $18, $1E, $2A
    .BYTE $1B, $8E, $64, $1D, $11, $0E, $24, $11
    .BYTE $0E, $1B, $18, $24, $18, $0F, $24, $11
    .BYTE $22, $1B, $1E, $15, $0E, $EC

; ----------------------------------------------------------------------------
; Init submode 2: types Zelda's text; when it is complete, starts a $50 frame delay and advances.
; ----------------------------------------------------------------------------
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

; DATA: transfer record template for one textbox character: address $21A4, count 1, tile $24, end $FF.
ThanksTextboxCharTransferRecTemplate:
    .BYTE $21, $A4, $01, $24, $FF

; TABLE: low nametable address byte of a text line's first character; index = 0 line 2, 1 line 3,
; 2 line 1 (selected by the high 2 bits of a ThanksText byte); high byte is $21.
ThanksTextboxLineAddrsLo:
    .BYTE $C4, $E4, $A4

; ----------------------------------------------------------------------------
; Draws Link and, every 6 frames, writes the next character of ThanksText to the screen.
; OUT:  ObjState+1 incremented (nonzero) and ObjState (Link) = 0 once the last character is written.
; NOTE: paced by ObjTimer+1 (Zelda's timer); each character requests tune $10 on square 0.
; ----------------------------------------------------------------------------
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

; Shared exit of UpdateZeldaTextbox and InitMode13_Sub3.
LA9F4_Exit:
    RTS

; ----------------------------------------------------------------------------
; Init submode 3: when Zelda's timer (the $50 frame delay) expires, silences all sound and advances.
; ----------------------------------------------------------------------------
InitMode13_Sub3:
    LDA ObjTimer+1
    BNE LA9F4_Exit
    JSR SilenceAllSound
    INC GameSubmode
    RTS

; ----------------------------------------------------------------------------
; Init submode 4: presets the credits scroll and text counters, ends init, and hides all sprites.
; NOTE: uses the A = 0 that BeginUpdateMode returns to clear the two "peace" text counters.
; ----------------------------------------------------------------------------
InitMode13_Sub4:
    LDA #$08
    STA CreditsTileOffset
    JSR BeginUpdateMode
    STA PeaceCharDelayCounter
    STA PeaceCharIndex
    JMP HideAllSprites
