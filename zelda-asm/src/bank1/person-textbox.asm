; ============================================================================
; FILE:     bank1/person-textbox.asm     BANK: 1 (switchable, $8000-$BFFF)
; UPSTREAM: Z_01.asm 570-676
; SUBJECTS: Person
; TOPIC:    The textbox state shared by cave persons, dungeon persons and the hungry Goriya: types the
;           person's text one character at a time, then advances the person's state and unhalts Link.
; SEE ALSO: bank1/cave-init-person-text.asm (Person_Text, Person_TextAddrs, Link_Unhalt; the text
;           encoding); bank1/cave-person-prices.asm (cave person states 1 and 7);
;           bank1/underworld-person-update.asm, bank1/life-or-money.asm, bank1/grumble-goriya.asm (the
;           other state tables that use this handler); bank6/transfer-bufs-static.asm
;           (Transfer_BlankTextLines12Buf).
; KEY FACTS:
;   - One character is typed every 6 frames (ObjTimer+1 reloaded with 6); characters with tile code
;     $25 are skipped in the same frame and cost no time, ordinary blanks ($24) cost 6 frames.
;     Person_UpdateState_Textbox
;   - A person's setup halts Link; he is unhalted, and the person's state advanced, in the frame the
;     last character (bits 7-6 = $C0) is typed. The handler itself neither halts Link nor tests
;     whether he is halted. Person_UpdateState_Textbox
;   - Textbox lines start at nametable $21A4, $21C4, $21E4. A character with bits 7-6 = $80 is
;     followed by line 2, $40 by line 3. Person_TextboxLineAddrsLo, Person_UpdateState_Textbox
;   - Every typed character requests tune 0 bit $10. Person_UpdateState_Textbox
;   - The handler always uses object slot 1's timer and state (ObjTimer+1, ObjState+1).
;     Person_UpdateState_Textbox
; CONTENTS: (generated; L = line in this file)
;   L0032 Person_TextboxCharTransferRecTemplate  data  DATA: transfer record template for one textbox character: nametable address $21A4 (the low byte is
;   L0037 Person_TextboxLineAddrsLo              data  TABLE: low byte of the nametable address ($21xx) where a textbox line starts; index 0 = line 2
;   L0051 Person_UpdateState_Textbox             code  Person state: types the next character of the person's text; at the end, next state and unhalt Link.
; ============================================================================

; DATA: transfer record template for one textbox character: nametable address $21A4 (the low byte is
; replaced), 1 tile, $24 (replaced by the character), end marker $FF.
Person_TextboxCharTransferRecTemplate:
    .BYTE $21, $A4, $01, $24, $FF

; TABLE: low byte of the nametable address ($21xx) where a textbox line starts; index 0 = line 2
; ($C4), 1 = line 3 ($E4), 2 = line 1 ($A4).
Person_TextboxLineAddrsLo:
    .BYTE $C4, $E4, $A4

; ----------------------------------------------------------------------------
; Person state: types the next character of the person's text; at the end, next state and unhalt Link.
; IN:   PersonTextSelector = offset of the text's address in Person_TextAddrs; PersonTextIndex = index
;       of the next character; PersonTextVramLo = low nametable address of the next cell;
;       ObjTimer+1 = frames until the next character.
; OUT:  DynTileBuf = one-tile transfer record; PersonTextIndex and PersonTextVramLo advanced;
;       ObjTimer+1 = 6; at the end of the text ObjState+1 is incremented, ObjState (Link) = 0 and
;       PersonTextVramLo = $A4. Uses [00:01] (text address).
; NOTE: 6 frames per character; tile code $25 is skipped at no cost in time. Link is drawn every
;       frame with his animation counter held (Link_EndMoveAndDraw).
; ----------------------------------------------------------------------------
Person_UpdateState_Textbox:
    JSR Link_EndMoveAndDraw_ReturnToBank1

    ; Wait until the person's timer has run out.
    LDA ObjTimer+1
    BNE @Exit

    ; 6 frames until the character after this one.
    LDA #$06
    STA ObjTimer+1

    ; Copy the 5-byte one-character transfer record template to the dynamic transfer buffer.
    LDY #$04
:
    LDA Person_TextboxCharTransferRecTemplate, Y
    STA DynTileBuf, Y
    DEY
    BPL :-

@WriteChar:
    ; Set the record's low nametable address to the current cell, and step to the next cell.
    LDA PersonTextVramLo
    STA DynTileBuf+1
    INC PersonTextVramLo

    ; [00:01] = address of the person's text.
    LDY PersonTextSelector
    LDA Person_TextAddrs, Y
    STA $00
    INY
    LDA Person_TextAddrs, Y
    STA $01

    ; Y = index of the current character; the stored index moves on to the next one.
    LDY PersonTextIndex
    INC PersonTextIndex

    ; Low 6 bits = character tile.
    LDA ($00), Y

    ; Tile $25 is a blank that takes a cell but no time: go on to the next character in this
    ; same frame. (Its high 2 bits are not examined.)
    AND #$3F
    CMP #$25
    BEQ @WriteChar

    ; Put the character in the transfer record.
    STA DynTileBuf+3

    ; Request tune 0 bit $10 (the sound of a text character).
    LDA #$10
    STA Tune0Request

    ; High 2 bits of the character = 0: the line goes on; return.
    LDA ($00), Y
    AND #$C0
    BEQ @Exit

    ; Otherwise turn them into an index into Person_TextboxLineAddrsLo:
    ;   $80: 0 (go to line 2)   $40: 1 (go to line 3)   $C0: 2 (end of text, back to line 1)
    LDY #$02
    CMP #$C0
    BEQ @ChangeLine
    DEY
    CMP #$40
    BEQ @ChangeLine
    DEY

@ChangeLine:
    ; The next character goes to the start of that line.
    LDA Person_TextboxLineAddrsLo, Y
    STA PersonTextVramLo

    ; Index 2 = end of the text: advance the person's state and unhalt Link.
    CPY #$02
    BNE @Exit
    INC ObjState+1
    JSR Link_Unhalt

@Exit:
    RTS
