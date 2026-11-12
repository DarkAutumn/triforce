; ============================================================================
; FILE:     bank1/copy-code-demo-patterns.asm BANK: 1 (switchable, $8000-$BFFF)
; UPSTREAM: Z_01.asm 1569-1743
; SUBJECTS: (none)
; TOPIC:    Copies the common code segment from bank 1 to RAM; sends the title/story demo pattern
;           (CHR) blocks to the PPU; the two pattern blobs; filler.
; SEE ALSO: bank1/common-code-header.asm (the BANK_01_CODE segment that is copied);
;           bank7/init-mode-dispatch.asm (the callers: InitializeGameOrMode, InitMode2, InitMode0);
;           bank2/common-patterns.asm (the same transfer for the common patterns);
;           bank3/level-pattern-blocks.asm (the per-level pattern transfer).
; KEY FACTS:
;   - The common code runs from RAM at $6C90-$7EFF; it is copied there once after reset and again at the
;     start of every level load (mode 2, submode 0). CopyCommonCodeToRam
;   - The demo patterns are two blocks: $0900 bytes to PPU address $0700 (sprite table) and $0820
;     bytes to $1700 (background table). DemoPatternBlockSizes, DemoPatternVramAddrs
;   - HasDemoPatterns = $A5 marks the demo patterns as loaded; mode 0 init calls the transfer only
;     when it holds another value. TransferDemoPatterns
;   - Both blocks go out in one call with video turned off. TransferDemoPatterns
; CONTENTS: (generated; L = line in this file)
;   L0037 CopyCommonCodeToRam         code  Copies the BANK_01_CODE segment (common code) from this bank to RAM, where it runs.
;   L0078 DemoPatternBlockAddrs       data  TABLE: addresses of the demo pattern blocks; index = PatternBlockIndex (0 sprites, 1 background);
;   L0083 DemoPatternBlockSizes       data  TABLE: sizes of the demo pattern blocks; index = PatternBlockIndex; value = bytes, high byte first.
;   L0089 DemoPatternVramAddrs        data  TABLE: PPU destination of the demo pattern blocks; index = PatternBlockIndex; value = PPU address,
;   L0098 TransferDemoPatterns        code  Sends both demo pattern blocks to the PPU with video off and marks them loaded.
;   L0141 TransferPatternBlock_Bank1  code  Writes one pattern block to the PPU and increments PatternBlockIndex.
;   L0179 DemoSpritePatterns          data  DATA: demo sprite patterns (CHR), $0900 bytes = 144 tiles of 16 bytes, sent to PPU $0700.
;   L0183 DemoBackgroundPatterns      data  DATA: demo background patterns (CHR), $0820 bytes = 130 tiles of 16 bytes, sent to PPU $1700.
; ============================================================================

; ----------------------------------------------------------------------------
; Copies the BANK_01_CODE segment (common code) from this bank to RAM, where it runs.
; OUT:  RAM from __BANK_01_CODE_RUN__ ($6C90) up to __BANK_01_CODE_RUN_END__ ($7F00) filled;
;       [00]-[03] and A, Y changed.
; NOTE: Bank 1 must be switched in. Called once after reset (InitializeGameOrMode) and at the start of
;       every level load (InitMode2, submode 0).
; ----------------------------------------------------------------------------
CopyCommonCodeToRam:
    LDA #<__BANK_01_CODE_LOAD__ ; Source address $A500 -> [00:01].
    STA $00
    LDA #>__BANK_01_CODE_LOAD__
    STA $01
    LDA #<__BANK_01_CODE_RUN__  ; Destination address $6C90 -> [02:03].
    STA $02
    LDA #>__BANK_01_CODE_RUN__
    STA $03
    LDY #$00

@Loop:
    LDA ($00), Y                ; Copy 1 byte.
    STA ($02), Y

    ; Increment source address.
    LDA $00
    CLC
    ADC #$01
    STA $00
    LDA $01
    ADC #$00
    STA $01

    ; Increment destination address.
    LDA $02
    CLC
    ADC #$01
    STA $02
    LDA $03
    ADC #$00
    STA $03
    CMP #>__BANK_01_CODE_RUN_END__    ; Done once the destination reaches $7F00.
    BNE @Loop
    LDA $02
    CMP #<__BANK_01_CODE_RUN_END__
    BNE @Loop
    RTS

; TABLE: addresses of the demo pattern blocks; index = PatternBlockIndex (0 sprites, 1 background);
;        value = address in this bank.
DemoPatternBlockAddrs:
    .ADDR DemoSpritePatterns
    .ADDR DemoBackgroundPatterns

; TABLE: sizes of the demo pattern blocks; index = PatternBlockIndex; value = bytes, high byte first.
DemoPatternBlockSizes:
    .DBYT $0900
    .DBYT $0820

; TABLE: PPU destination of the demo pattern blocks; index = PatternBlockIndex; value = PPU address,
;        high byte first.
DemoPatternVramAddrs:
    .DBYT $0700                 ; In pattern table 0 (sprites), from tile $70.
    .DBYT $1700                 ; In pattern table 1 (background), from tile $70.

; ----------------------------------------------------------------------------
; Sends both demo pattern blocks to the PPU with video off and marks them loaded.
; OUT:  HasDemoPatterns = $A5; PatternBlockIndex = 0; [00]-[03] changed.
; NOTE: Called from mode 0 init while HasDemoPatterns <> $A5. Assumes PatternBlockIndex = 0 on entry.
; ----------------------------------------------------------------------------
TransferDemoPatterns:
    JSR TurnOffAllVideo
    LDA PpuStatus          ; Clear address latch.

@LoopBlock:
    ; Loop over each block.
    LDA PatternBlockIndex
    ASL
    TAX

    ; Put block address in [00:01] and size in [03:02].
    ; Load destination VRAM address and set it.
    ; The size and VRAM address have the high byte first.
    LDA DemoPatternBlockAddrs, X
    STA $00
    LDA DemoPatternBlockSizes, X
    STA $02
    LDA DemoPatternVramAddrs, X
    STA PpuAddr
    INX
    LDA DemoPatternBlockAddrs, X
    STA $01
    LDA DemoPatternBlockSizes, X
    STA $03
    LDA DemoPatternVramAddrs, X
    JSR TransferPatternBlock_Bank1

    ; Loop until pattern block index = 2.
    LDA PatternBlockIndex
    CMP #$02
    BNE @LoopBlock
    LDA #$A5                    ; Mark the demo patterns loaded.
    STA HasDemoPatterns
    LDA #$00
    STA PatternBlockIndex
    RTS

; ----------------------------------------------------------------------------
; Writes one pattern block to the PPU and increments PatternBlockIndex.
; IN:   [00:01] = source address; [02] = size high byte, [03] = size low byte; A = low byte of the
;       PPU address (the high byte was already written to PpuAddr).
; OUT:  PatternBlockIndex incremented; [00:01] advanced past the block; [02], [03] = 0.
; ----------------------------------------------------------------------------
TransferPatternBlock_Bank1:
    STA PpuAddr
    LDY #$00

@Loop:
    ; Load and transfer one byte to VRAM.
    LDA ($00), Y
    STA PpuData

    ; Increment the 16-bit address at [00:01].
    LDA $00
    CLC
    ADC #$01
    STA $00
    LDA $01
    ADC #$00
    STA $01

    ; Decrement the 16-bit amount at [03:02].
    LDA $03
    SEC
    SBC #$01
    STA $03
    LDA $02
    SBC #$00
    STA $02

    ; Loop again if the amount left <> 0.
    LDA $02
    BNE @Loop
    LDA $03
    BNE @Loop

    ; This block is done. Increment the block index.
    INC PatternBlockIndex
    RTS

; DATA: demo sprite patterns (CHR), $0900 bytes = 144 tiles of 16 bytes, sent to PPU $0700.
DemoSpritePatterns:
    .INCBIN "dat/DemoSpritePatterns.dat"

; DATA: demo background patterns (CHR), $0820 bytes = 130 tiles of 16 bytes, sent to PPU $1700.
DemoBackgroundPatterns:
    .INCBIN "dat/DemoBackgroundPatterns.dat"

; DATA: filler ($FF), 300 bytes, not referenced.
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
    .BYTE $FF, $FF, $FF, $FF
