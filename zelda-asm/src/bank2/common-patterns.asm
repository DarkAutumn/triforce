; ============================================================================
; FILE:     bank2/common-patterns.asm    BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 55-178
; SUBJECTS: (none)
; TOPIC:    The three CHR pattern blocks every part of the game shares (sprites, background, misc) and
;           the routine that copies them to the PPU pattern tables once after power-on.
; SEE ALSO: bank7/init-mode-dispatch.asm (InitMode0 calls TransferCommonPatterns while HasCommonPatterns
;           is not $5A); bank1/copy-code-demo-patterns.asm (the demo pattern transfer that follows);
;           bank3/level-pattern-blocks.asm (per-level patterns and the bank 3 copy of the block transfer).
; KEY FACTS:
;   - Three blocks are sent in one call: $0700 bytes to PPU $0000 (sprites), $0700 bytes to $1000
;     (background), $00E0 bytes to $1F20 (misc). CommonPatternBlockAddrs, CommonPatternBlockSizes,
;     CommonPatternVramAddrs, TransferCommonPatterns.
;   - TransferCommonPatterns sets HasCommonPatterns = $5A when done, so mode 0 init does it only once
;     per power-on, and leaves PatternBlockIndex = 0.
;   - Sizes and PPU addresses are stored high byte first (.DBYT); block addresses low byte first.
;   - TransferPatternBlock_Bank2 copies one byte per loop pass with rendering off and increments
;     PatternBlockIndex at the end.
;   - The $FF bytes after CommonMiscPatterns are padding (161 bytes), never read.
; CONTENTS: (generated; L = line in this file)
;   L0032 CommonPatternBlockAddrs     data  TABLE: CPU address of each common pattern block; index = PatternBlockIndex (0-2).
;   L0038 CommonPatternBlockSizes     data  TABLE: size in bytes of each common pattern block, high byte first; index = PatternBlockIndex.
;   L0044 CommonPatternVramAddrs      data  TABLE: PPU pattern-table address each common block is written to, high byte first; index as above.
;   L0054 TransferCommonPatterns      code  Copies the three common pattern blocks to the PPU with video off and sets HasCommonPatterns = $5A.
;   L0096 TransferPatternBlock_Bank2  code  Writes one pattern block to PPU memory and increments PatternBlockIndex.
;   L0134 CommonSpritePatterns        data  DATA: common sprite patterns, $0700 bytes of CHR tiles for PPU $0000 (112 tiles of 16 bytes).
;   L0138 CommonBackgroundPatterns    data  DATA: common background patterns, $0700 bytes of CHR tiles for PPU $1000.
;   L0142 CommonMiscPatterns          data  DATA: common misc patterns, $00E0 bytes of CHR tiles for PPU $1F20 (background tiles $F2-$FF).
; ============================================================================

; TABLE: CPU address of each common pattern block; index = PatternBlockIndex (0-2).
CommonPatternBlockAddrs:
    .ADDR CommonSpritePatterns
    .ADDR CommonBackgroundPatterns
    .ADDR CommonMiscPatterns

; TABLE: size in bytes of each common pattern block, high byte first; index = PatternBlockIndex.
CommonPatternBlockSizes:
    .DBYT $0700
    .DBYT $0700
    .DBYT $00E0

; TABLE: PPU pattern-table address each common block is written to, high byte first; index as above.
CommonPatternVramAddrs:
    .DBYT $0000
    .DBYT $1000
    .DBYT $1F20

; ----------------------------------------------------------------------------
; Copies the three common pattern blocks to the PPU with video off and sets HasCommonPatterns = $5A.
; IN:   PatternBlockIndex = first block to send (0 on entry from InitMode0).
; OUT:  PatternBlockIndex = 0; HasCommonPatterns = $5A.
; ----------------------------------------------------------------------------
TransferCommonPatterns:
    JSR TurnOffAllVideo
    LDA PpuStatus          ; Clear address latch and scroll.

@LoopBlock:
    LDA PatternBlockIndex
    ASL
    TAX

    ; Put block address in [00:01] and size in [03:02].
    ; Load destination VRAM address and set it.
    ; The size and VRAM address have the high byte first.
    LDA CommonPatternBlockAddrs, X
    STA $00
    LDA CommonPatternBlockSizes, X
    STA $02
    LDA CommonPatternVramAddrs, X
    STA PpuAddr
    INX
    LDA CommonPatternBlockAddrs, X
    STA $01
    LDA CommonPatternBlockSizes, X
    STA $03
    LDA CommonPatternVramAddrs, X
    JSR TransferPatternBlock_Bank2

    ; Loop until pattern block index = 3.
    LDA PatternBlockIndex
    CMP #$03
    BNE @LoopBlock
    LDA #$5A                    ; Mark this block copied.
    STA HasCommonPatterns
    LDA #$00                    ; Reset pattern block index.
    STA PatternBlockIndex
    RTS

; ----------------------------------------------------------------------------
; Writes one pattern block to PPU memory and increments PatternBlockIndex.
; IN:   [00:01] = source address; [03:02] = size ([02] is the high byte, [03] the low byte);
;       A = low byte of the destination PPU address (the high byte was already written to PpuAddr).
; OUT:  [00:01] = address after the block; [02], [03] = 0; PatternBlockIndex incremented.
; ----------------------------------------------------------------------------
TransferPatternBlock_Bank2:
    STA PpuAddr
    LDY #$00

@Loop:
    ; Load and transfer 1 byte to VRAM.
    LDA ($00), Y
    STA PpuData

    ; Increment the 16-bit source address at [00:01].
    LDA $00
    CLC
    ADC #$01
    STA $00
    LDA $01
    ADC #$00
    STA $01

    ; Decrement the 16-bit amount remaining in [03:02].
    LDA $03
    SEC
    SBC #$01
    STA $03
    LDA $02
    SBC #$00
    STA $02

    ; Loop until the amount remaining = 0.
    LDA $02
    BNE @Loop
    LDA $03
    BNE @Loop

    ; The block is done. Increment the block index.
    INC PatternBlockIndex
    RTS

; DATA: common sprite patterns, $0700 bytes of CHR tiles for PPU $0000 (112 tiles of 16 bytes).
CommonSpritePatterns:
    .INCBIN "dat/CommonSpritePatterns.dat"

; DATA: common background patterns, $0700 bytes of CHR tiles for PPU $1000.
CommonBackgroundPatterns:
    .INCBIN "dat/CommonBackgroundPatterns.dat"

; DATA: common misc patterns, $00E0 bytes of CHR tiles for PPU $1F20 (background tiles $F2-$FF).
CommonMiscPatterns:
    .INCBIN "dat/CommonMiscPatterns.dat"

; Padding: 161 bytes of $FF, not referenced by any code.
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF
