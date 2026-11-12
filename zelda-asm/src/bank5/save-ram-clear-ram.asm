; ============================================================================
; FILE:     bank5/save-ram-clear-ram.asm BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 7409-7480
; SUBJECTS: (none)
; TOPIC:    InitSaveRam, ClearRam, first edge spawn cell, RNG seed
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0016 InitSaveRam  code  Returns:
;   L0052 ClearRam     code
; ============================================================================

; Returns:
; C: 1 if cleared; 0 if already cleared
;
InitSaveRam:
    LDA SaveRamBegin            ; If already cleared, then return false.
    CMP #$5A
    BNE :+
    LDA SaveRamEnd
    CMP #$A5
    BEQ @ReturnFalse
:
    LDA #$FF                    ; Since we're clearing save RAM, treat file B as committed.
    STA IsSaveFileBCommitted
    STA IsSaveFileBCommitted+1
    STA IsSaveFileBCommitted+2
    LDA #$65                    ; Clear from $6530 to the end of Save RAM.
    STA $01
    LDA #$30
    STA $00
    LDY #$00
:
    LDA #$00
    STA ($00), Y
    LDA $00
    CLC
    ADC #$01
    STA $00
    LDA $01
    ADC #$00
    STA $01
    CMP #$80
    BNE :-
    SEC                         ; Return C=1.
    RTS

@ReturnFalse:
    CLC                         ; Return C=0.
    RTS

ClearRam:
    LDA #$07
    LDY #$FE                    ; It doesn't touch [$07FF].
    JSR ClearRam0300UpTo
    LDA #$00                    ; Clear a few individual variables that are left.
    STA ReturnToBank4
    STA HasCommonPatterns
    STA HasDemoPatterns
    STA IsScrollMirroringSet
    LDY #$EF
:
    STA $0000, Y                ; Clear RAM from 0 to $EF.
    DEY
    CPY #$FF
    BNE :-

    ; The first time looking for an edge cell to spawn a monster
    ; from, look here first.
    LDA #$40
    STA CurEdgeSpawnCell
    STA Random

    ; These are part of the IsSaveFileActive array.
    ;
    ; Mode 1 Menu checks 5 elements of the array to see
    ; if the option can be chosen. The first 3 are the save
    ; slots. The last two are the register and eliminate options.
    ;
    ; So, theses two should always be set.
    LDA #$01
    STA IsRegisterSaveFileOptionEnabled
    STA IsEliminateSaveFileOptionEnabled
    RTS
