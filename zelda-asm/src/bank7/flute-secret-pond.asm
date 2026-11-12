; ============================================================================
; FILE:     bank7/flute-secret-pond.asm  BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 5864-5970
; SUBJECTS: (none)
; TOPIC:    Tile object/item init, flute secret pond
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0022 InitTileObjOrItem                code
;   L0032 InitFluteSecret                  code
;   L0042 ResetObjMetastateAndTimer        code
;   L0046 ResetObjMetastate                code
;   L0055 WaterPaletteTransferBufTemplate  data
;   L0058 PondCycleColors                  data
;   L0062 UpdateFluteSecret                code
;   L0086 CueTransferPondPaletteRow        code  Params:
;   L0107 L1FF28_Exit                      code
;   L0110 RevealPondStairs                 code
;   L0119 AnimatePond                      code
; ============================================================================

InitTileObjOrItem:
    ; Set object attribute to $81, and go reset metastate and timer;
    ; so that the object is ready to start updating itself.
    ;
    ; Obj attr $81 = "Custom check collision and drawing"
    ;              + "Reverse after hitting Link"
    LDA #$81
    STA ObjAttr, X
    BNE ResetObjMetastateAndTimer

InitFluteSecret:
    ; Summary:
    ; The flute secret object manages the secret color cycle.
    ;
    ; The flute secret object is initialized in the same frame that
    ; the flute was wielded. But this object won't update until
    ; the flute timer expires.
    LDA #$01
    STA SecretColorCycle

ResetObjMetastateAndTimer:
    LDA #$00
    STA ObjTimer, X

ResetObjMetastate:
    ; Reset the object metastate so that it's ready to start
    ; behaving autonomously (updating on its own).
    ; This is the normal state outside of the spawning cloud and
    ; the death sparkle.
    LDA #$00
    STA ObjMetastate, X
    RTS

WaterPaletteTransferBufTemplate:
    .BYTE $3F, $0C, $04, $0F, $17, $37, $12, $FF

PondCycleColors:
    .BYTE $12, $11, $22, $21, $31, $32, $33, $35
    .BYTE $34, $36, $37, $37

UpdateFluteSecret:
    ; If secret color cycle >= $C, return.
    LDY SecretColorCycle
    CPY #$0C
    BCS L1FF28_Exit

    ; 7 of every 8 frames, return.
    LDA FrameCounter
    AND #$07
    CMP #$04
    BNE L1FF28_Exit

    ; So, every 8 frames:
    ; 1. Increment the secret color cycle count.
    ; 2. Change the water palette.
    ;
    ; But when the count = $B, go reveal the stairs.
    INC SecretColorCycle
    CPY #$0B
    BEQ RevealPondStairs

; Params:
; Y: a point in the cycle (0 to $B)
;
CueTransferPondPaletteRow:
    ; Copy water palette (palette row 3) transfer buf to dynamic
    ; transfer buf.
    TYA
    PHA
    LDY #$07

@CopyBytes:
    LDA WaterPaletteTransferBufTemplate, Y
    STA DynTileBuf, Y
    DEY
    BPL @CopyBytes
    PLA
    TAY
    LDA PondCycleColors, Y      ; Patch byte 3 of palette row with the right color in the cycle.
    STA DynTileBuf+6
    CPY #$0A
    BNE L1FF28_Exit             ; If the index is $A,
    LDA #$99                    ; then make most water tiles (< $99) walkable.
    STA ObjectFirstUnwalkableTile

L1FF28_Exit:
    RTS

RevealPondStairs:
    ; Set X and Y in this slot for the stairs in the pond.
    ; Go reveal the stairs as a secret.
    LDA #$60
    STA ObjX, X
    LDA #$90
    STA ObjY, X
    JMP RevealAndFlagSecretStairsObj

AnimatePond:
    ; Take turns between:
    ; * 4 frames stepping the cycle
    ; * 4 frames delaying
    LDA FrameCounter
    AND #$04
    BEQ L1FF28_Exit
    DEC SecretColorCycle
    LDY SecretColorCycle
    JMP CueTransferPondPaletteRow
