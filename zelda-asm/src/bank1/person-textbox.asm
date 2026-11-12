; ============================================================================
; FILE:     bank1/person-textbox.asm     BANK: 1 (switchable, $8000-$BFFF)
; UPSTREAM: Z_01.asm 570-676
; SUBJECTS: (none)
; TOPIC:    Textbox typing, one character per 6 frames, unhalts Link
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0014 TextboxCharTransferRecTemplate  data
;   L0017 TextboxLineAddrsLo              data
;   L0020 UpdatePersonState_Textbox       code
; ============================================================================

TextboxCharTransferRecTemplate:
    .BYTE $21, $A4, $01, $24, $FF

TextboxLineAddrsLo:
    .BYTE $C4, $E4, $A4

UpdatePersonState_Textbox:
    JSR Link_EndMoveAndDraw_Bank1

    ; If the person's timer has not expired, then return.
    LDA ObjTimer+1
    BNE @Exit

    ; Set the timer to wait 6 frames after the next character about
    ; to be shown.
    LDA #$06
    STA ObjTimer+1

    ; Copy the 5 bytes of the textbox character transfer record template
    ; to the dynamic transfer buf.
    LDY #$04
:
    LDA TextboxCharTransferRecTemplate, Y
    STA DynTileBuf, Y
    DEY
    BPL :-

@WriteChar:
    ; Replace the low byte of the VRAM address with the one
    ; where the next character should be written.
    LDA PersonTextPtr
    STA DynTileBuf+1

    ; Increment the low VRAM address for the next character.
    INC PersonTextPtr

    ; Use the person text selector to look up the address of the
    ; text for the textbox. Store the address in [00:01].
    LDY PersonTextSelector
    LDA PersonTextAddrs, Y
    STA $00
    INY
    LDA PersonTextAddrs, Y
    STA $01

    ; Load the person text current character index.
    LDY PersonTextIndex

    ; Increment the index variable to point to the next character
    ; for next time.
    INC PersonTextIndex

    ; Get the current character.
    LDA ($00), Y

    ; If the character is $25, then it's a special space. It will still
    ; take up space, but will not take time to show -- meaning that
    ; we'll go look up the next character to transfer.
    AND #$3F
    CMP #$25
    BEQ @WriteChar

    ; We have a non-space character. Put it in the transfer record.
    STA DynTileBuf+3

    ; Play the "heart taken/character" sound effect.
    LDA #$10
    STA Tune0Request

    ; If the high 2 bits of character element = 0, then return.
    LDA ($00), Y
    AND #$C0
    BEQ @Exit

    ; Determine an index based on the high 2 bits of the character element:
    ;   $80: 0
    ;   $40: 1
    ;   $C0: 2
    LDY #$02
    CMP #$C0
    BEQ @ChangeLine
    DEY
    CMP #$40
    BEQ @ChangeLine
    DEY

@ChangeLine:
    ; The index chooses the low VRAM address of the start
    ; of another line:
    ;   0: $C4: front of the second line
    ;   1: $E4: front of the third line
    ;   2: $A4: front of the first line
    ;
    ; Look up the low VRAM address and store it.
    LDA TextboxLineAddrsLo, Y
    STA PersonTextPtr

    ; If index = 2, then we've reached the end of the text,
    ; and low VRAM address is moved to the front of the first line.
    ; So, advance the state of the person object, and unhalt Link.
    CPY #$02
    BNE @Exit
    INC ObjState+1
    JSR UnhaltLink

@Exit:
    RTS
