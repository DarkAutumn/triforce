; ============================================================================
; FILE:     bank1/copy-code-demo-patterns.asm BANK: 1 (switchable, $8000-$BFFF)
; UPSTREAM: Z_01.asm 1569-1743
; SUBJECTS: (none)
; TOPIC:    Copy common code to RAM; demo pattern transfer and data
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0019 CopyCommonCodeToRam         code
;   L0058 DemoPatternBlockAddrs       data
;   L0062 DemoPatternBlockSizes       data
;   L0066 DemoPatternVramAddrs        data
;   L0070 TransferDemoPatterns        code
;   L0112 TransferPatternBlock_Bank1  code  Params:
;   L0149 DemoSpritePatterns          data
;   L0152 DemoBackgroundPatterns      data
; ============================================================================

CopyCommonCodeToRam:
    LDA #<__BANK_01_CODE_LOAD__ ; Source address $A500.
    STA $00
    LDA #>__BANK_01_CODE_LOAD__
    STA $01
    LDA #<__BANK_01_CODE_RUN__  ; Destination address $6C90.
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
    CMP #>__BANK_01_CODE_RUN_END__    ; Once you reach $7F00, you're done.
    BNE @Loop
    LDA $02
    CMP #<__BANK_01_CODE_RUN_END__
    BNE @Loop
    RTS

DemoPatternBlockAddrs:
    .ADDR DemoSpritePatterns
    .ADDR DemoBackgroundPatterns

DemoPatternBlockSizes:
    .DBYT $0900
    .DBYT $0820

DemoPatternVramAddrs:
    .DBYT $0700
    .DBYT $1700

TransferDemoPatterns:
    JSR TurnOffAllVideo
    LDA PpuStatus_2002          ; Clear address latch.

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
    STA PpuAddr_2006
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
    LDA #$A5
    STA TransferredDemoPatterns
    LDA #$00
    STA PatternBlockIndex
    RTS

; Params:
; [00:01]: source address
; [03:02]: size
; A: low byte of desination VRAM address
;
TransferPatternBlock_Bank1:
    STA PpuAddr_2006
    LDY #$00

@Loop:
    ; Load and transfer one byte to VRAM.
    LDA ($00), Y
    STA PpuData_2007

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

DemoSpritePatterns:
    .INCBIN "dat/DemoSpritePatterns.dat"

DemoBackgroundPatterns:
    .INCBIN "dat/DemoBackgroundPatterns.dat"

; Unknown block
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
