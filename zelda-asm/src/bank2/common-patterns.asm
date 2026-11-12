; ============================================================================
; FILE:     bank2/common-patterns.asm    BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 55-178
; SUBJECTS: (none)
; TOPIC:    Common CHR pattern blocks and transfer
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0019 CommonPatternBlockAddrs     data
;   L0024 CommonPatternBlockSizes     data
;   L0029 CommonPatternVramAddrs      data
;   L0034 TransferCommonPatterns      code
;   L0075 TransferPatternBlock_Bank2  code  Params:
;   L0112 CommonSpritePatterns        data
;   L0115 CommonBackgroundPatterns    data
;   L0118 CommonMiscPatterns          data
; ============================================================================

CommonPatternBlockAddrs:
    .ADDR CommonSpritePatterns
    .ADDR CommonBackgroundPatterns
    .ADDR CommonMiscPatterns

CommonPatternBlockSizes:
    .DBYT $0700
    .DBYT $0700
    .DBYT $00E0

CommonPatternVramAddrs:
    .DBYT $0000
    .DBYT $1000
    .DBYT $1F20

TransferCommonPatterns:
    JSR TurnOffAllVideo
    LDA PpuStatus_2002          ; Clear address latch and scroll.

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
    STA PpuAddr_2006
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
    STA TransferredCommonPatterns
    LDA #$00                    ; Reset pattern block index.
    STA PatternBlockIndex
    RTS

; Params:
; [00:01]: source address
; [03:02]: size
; A: low byte of desination VRAM address
;
TransferPatternBlock_Bank2:
    STA PpuAddr_2006
    LDY #$00

@Loop:
    ; Load and transfer 1 byte to VRAM.
    LDA ($00), Y
    STA PpuData_2007

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

CommonSpritePatterns:
    .INCBIN "dat/CommonSpritePatterns.dat"

CommonBackgroundPatterns:
    .INCBIN "dat/CommonBackgroundPatterns.dat"

CommonMiscPatterns:
    .INCBIN "dat/CommonMiscPatterns.dat"

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
    .BYTE $FF
