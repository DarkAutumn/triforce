; ============================================================================
; FILE:     bank5/save-ram-clear-ram.asm BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 7409-7480
; SUBJECTS: Reset, Save
; TOPIC:    Power-on memory setup: zeroing the working part of save RAM when it carries no valid
;           markers, and clearing the console RAM with the few starting values that are not 0 (the
;           random number seed, the first screen-edge spawn cell, the two menu options).
; SEE ALSO: bank7/run-game-nmi.asm (Reset_RunGame calls both at reset); bank5/mode-inits-obj-lists.asm
;           (ModeDSave_Init calls Save_InitSaveRam before saving and resets the game if it cleared anything);
;           bank7/ppu-tablejump-input.asm (Util_ClearRam0300UpTo); bank5/edge-spawn-cell.asm
;           (EdgeSpawn_FindNextCell continues from CurEdgeSpawnCell).
; KEY FACTS:
;   - Save RAM counts as initialized when SaveRamBegin ($6001) = $5A and SaveRamEnd ($7FFF) = $A5;
;     otherwise $6530-$7FFF is zeroed and the three IsSaveFileBCommitted bytes are set to $FF
;     (Save_InitSaveRam). The markers are not written here.
;   - The random number shift register starts every power-on and reset as Random = $40 with the other
;     12 bytes 0 (Reset_ClearRam): the sequence after reset is the same every time.
;   - CurEdgeSpawnCell starts at $40 (Reset_ClearRam).
;   - Reset_ClearRam zeroes $0000-$00EF, $F3, $F5, $F6, $F7 and $0300-$07FE. It leaves $F0-$F2, $F4
;     (IsGameInitialized), $F8-$FF, pages 1 and 2 and $07FF alone (Reset_ClearRam).
; CONTENTS: (generated; L = line in this file)
;   L0035 Save_InitSaveRam  code  Zeroes save RAM from $6530 up when its two validity markers are missing.
;   L0081 Reset_ClearRam    code  Clears the console RAM at reset and sets the few variables that do not start at 0.
; ============================================================================

; ----------------------------------------------------------------------------
; Zeroes save RAM from $6530 up when its two validity markers are missing.
; OUT:  C = 1 when it cleared; C = 0 when save RAM was already initialized (nothing written)
; NOTE: Valid = SaveRamBegin ($6001) is $5A and SaveRamEnd ($7FFF) is $A5. The cleared range
;       $6530-$7FFF holds the play area tiles, the level block and level info copies and the RAM
;       copy of the common code; the saved files below $6530 are not zeroed, only their three
;       IsSaveFileBCommitted bytes are set to $FF.
; NOTE: Callers: Reset_RunGame at reset, and ModeDSave_Init, which jumps to the reset handler when C = 1.
; ----------------------------------------------------------------------------
Save_InitSaveRam:
    LDA SaveRamBegin            ; Both markers present: return C = 0.
    CMP #$5A
    BNE :+
    LDA SaveRamEnd
    CMP #$A5
    BEQ @ReturnFalse
:
    LDA #$FF                    ; Mark file B of each slot as committed (no copy to file A pending).
    STA IsSaveFileBCommitted
    STA IsSaveFileBCommitted+1
    STA IsSaveFileBCommitted+2
    LDA #$65                    ; [00:01] = $6530, the first address to clear.
    STA $01
    LDA #$30
    STA $00
    LDY #$00

@ClearByte:
    LDA #$00
    STA ($00), Y
    LDA $00                     ; Increment the 16-bit address.
    CLC
    ADC #$01
    STA $00
    LDA $01
    ADC #$00
    STA $01
    CMP #$80                    ; Stop when the address reaches $8000.
    BNE @ClearByte
    SEC                         ; Return C = 1.
    RTS

@ReturnFalse:
    CLC                         ; Return C = 0.
    RTS

; ----------------------------------------------------------------------------
; Clears the console RAM at reset and sets the few variables that do not start at 0.
; OUT:  $0300-$07FE = 0 (DynTileBuf = $FF, its end marker); $00-$EF, $F3, $F5, $F6, $F7 = 0;
;       CurEdgeSpawnCell = Random = $40; the Register and Eliminate menu options = 1
; NOTE: Random ($18) is the first byte of the 13-byte random shift register; the other 12 bytes
;       are 0 after this, so the random sequence starts from the same state at every reset.
; NOTE: Not cleared: $F0-$F2, $F4 (IsGameInitialized), $F8-$FF (controller state, scroll values and
;       PPU register copies), pages 1 and 2 (stack, sprites) and $07FF.
; ----------------------------------------------------------------------------
Reset_ClearRam:
    LDA #$07                    ; Clear $0300-$07FE ($07FF is left alone).
    LDY #$FE
    JSR Util_ClearRam0300UpTo
    LDA #$00                    ; Clear four variables above $EF one by one.
    STA ReturnToBank4
    STA HasCommonPatterns
    STA HasDemoPatterns
    STA IsScrollMirroringSet
    LDY #$EF
:
    STA $0000, Y                ; Clear $00-$EF.
    DEY
    CPY #$FF
    BNE :-

    ; The first search for a screen-edge cell to spawn a monster from starts at cell $40.
    ; The same value seeds the random number register.
    LDA #$40
    STA CurEdgeSpawnCell
    STA Random

    ; The file menu tests five "active" flags to decide whether an entry can be chosen:
    ; three save slots, then the Register and Eliminate options. The last two are always on.
    LDA #$01
    STA IsRegisterSaveFileOptionEnabled
    STA IsEliminateSaveFileOptionEnabled
    RTS
