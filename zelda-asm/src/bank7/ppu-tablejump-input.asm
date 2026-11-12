; ============================================================================
; FILE:     bank7/ppu-tablejump-input.asm BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 552-735
; SUBJECTS: (none)
; TOPIC:    PPU reset, TableJump, sprite hiding, controller reading
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0019 ResetPpuRegisters  code
;   L0035 ClearNameTable     code  Params:
;   L0089 TableJump          code
;   L0104 HideAllSprites     code
;   L0125 ClearRam0300UpTo   code  Params:
;   L0148 TurnOffAllVideo    code
;   L0154 ReadInputs         code
;   L0165 ReadOneController  code
; ============================================================================

ResetPpuRegisters:
    LDA #$00
    STA PpuScroll_2005
    STA CurHScroll
    STA PpuScroll_2005
    STA CurVScroll
    LDA #$30
    STA PpuControl_2000
    STA CurPpuControl_2000
    RTS

; Params:
; A: Hi byte of starting VRAM address
; X: Tile number
; Y: Tile attribute byte
;
ClearNameTable:
    STA $00
    STX $01
    STY $02
    LDA PpuStatus_2002
    LDA CurPpuControl_2000      ; Make sure to auto-increment VRAM by 1.
    AND #$FB
    STA PpuControl_2000
    STA CurPpuControl_2000
    LDA $00
    STA PpuAddr_2006
    LDY #$00
    STY PpuAddr_2006

    ; Fill one nametable with the tile.
    LDX #$04
    CMP #$20
    BCS :+
    ; Begin unverified code 1E5B6
    LDX $02                     ; UNKNOWN: Unused. Even so, it wouldn't make sense.
    ; End unverified code
:
    LDY #$00
    LDA $01

@LoopTile:
    STA PpuData_2007
    DEY
    BNE @LoopTile
    DEX
    BNE @LoopTile

    ; Fill the related attributes with the attribute byte.
    LDY $02
    LDA $00
    CMP #$20
    BCC @RestoreX
    ADC #$02
    STA PpuAddr_2006
    LDA #$C0
    STA PpuAddr_2006
    LDX #$40

@LoopAttr:
    STY PpuData_2007
    DEX
    BNE @LoopAttr

@RestoreX:
    ; Set X to the passed in value.
    ; Y was already its passed in value.
    LDX $01
    RTS

TableJump:
    ASL
    TAY
    PLA
    STA $00
    PLA
    STA $01
    INY
    LDA ($00), Y
    STA $02
    INY
    LDA ($00), Y
    STA $03
    JMP ($0002)

HideAllSprites:
    LDY #$00
    LDX #$40

@Loop:
    LDA #$F8
    STA Sprites, Y
    INY
    INY
    INY
    INY
    DEX
    BNE @Loop
    RTS

; Params:
; A: high byte of high address to reset
; Y: low byte of high address to reset
;
; Resets all bytes from $300 to byte Y of page A.
;
ClearRam0300UpTo:
    STA $01
    LDA #$00
    STA $00

@Loop:
    LDA #$00
    STA ($00), Y
    DEY
    CPY #$FF
    BNE @Loop
    DEC $01
    LDA $01
    CMP #$03
    BCS @Loop

    ; We overwrote the dynamic transfer buf with zeroes.
    ; But the cleared state of the tile buf has the end marker at
    ; the beginning. Write the end marker.
    LDA #$FF
    STA DynTileBuf
    RTS

TurnOffAllVideo:
    LDA #$00
    STA PpuMask_2001
    STA CurPpuMask_2001
    RTS

ReadInputs:
    LDA #$01                    ; Signal the controllers to poll.
    STA Ctrl1_4016
    LDA #$00                    ; Finish polling.
    STA Ctrl1_4016
    STA $03
    STA $04
    TAX
    JSR ReadOneController
    INX

ReadOneController:
    ; Read over and over until you get 
    ; two of the same readings in a row
    ; (three from the very beginning).
    ; [02] : previous reading in this loop
    ; [03]/2 : successive matching readings (for each controller)
    STA $02
    LDA #$01                    ; TODO: Why poll again?
    STA Ctrl1_4016
    LDA #$00
    STA Ctrl1_4016
    LDY #$08

@Read:
    LDA Ctrl1_4016, X
    LSR
    ROL ButtonsPressed, X       ; Roll the button bits in. Here ButtonsPressed means "down now".
    LSR
    ROL $00                     ; The expansion port reading will be in [$00].
    DEY
    BNE @Read
    LDA ButtonsPressed, X
    CMP $02
    BNE ReadOneController       ; If we didn't get the same reading, then try again.
    INC $03, X
    LDY $03, X
    CPY #$02
    BCC ReadOneController       ; If we didn't get at least two of the same readings, then try again.
    LDA $00                     ; Combine the controller and expansion inputs.
    ORA ButtonsPressed, X
    STA ButtonsPressed, X
    PHA
    EOR ButtonsDown, X
    AND ButtonsPressed, X
    STA ButtonsPressed, X       ; Now ButtonsPressed means "down now instead of before".
    PLA
    STA ButtonsDown, X
    RTS
