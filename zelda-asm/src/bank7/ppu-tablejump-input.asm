; ============================================================================
; FILE:     bank7/ppu-tablejump-input.asm BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 552-735
; SUBJECTS: (none)
; TOPIC:    Engine helpers of the fixed bank: PPU register reset, name table fill, the inline jump-table
;           dispatcher TableJump, hiding all sprites, clearing RAM pages, and reading the controllers.
; SEE ALSO: bank7/run-game-nmi.asm (IsrNmi calls ReadInputs once per frame; RunGame and the mode inits
;           call the PPU helpers); bank5/save-ram-clear-ram.asm (ClearRam) and bank5/enter-room.asm
;           (InitMode_EnterRoom), the two callers of ClearRam0300UpTo.
; KEY FACTS:
;   - Button bits in ButtonsDown and ButtonsPressed: $80 A, $40 B, $20 Select, $10 Start, 8 Up, 4 Down,
;     2 Left, 1 Right; the first bit read ends up in bit 7 (ReadOneController@Read).
;   - ButtonsDown = buttons held this frame; ButtonsPressed = buttons held now that were not held at the
;     previous read (ReadOneController).
;   - Bit 1 of each controller port byte (expansion port controller) is ORed into the same button bits, so
;     either device can drive Link (ReadOneController).
;   - A reading is accepted only after two passes that each equal the pass before them; a pass that
;     differs just causes another read (ReadOneController).
;   - TableJump never returns to its caller: the address table must follow the JSR, A is the entry index,
;     and the target's RTS returns to the caller's caller. It overwrites [00]-[03] and Y (TableJump).
;   - ClearRam0300UpTo zeroes $0300 through byte Y of page A and then writes the empty marker $FF to
;     DynTileBuf; entering a room clears $0300-$051F with it: the object slot arrays from ObjType
;     ($034F) to ObjInvincibilityTimer ($04F0-$0503) and the item lift variables. The zero-page slot
;     arrays (position, direction, state, timers) are not touched (ClearRam0300UpTo).
;   - Sprites are hidden by setting their Y to $F8 (HideAllSprites).
; CONTENTS: (generated; L = line in this file)
;   L0042 ResetPpuRegisters  code  Zeroes the PPU scroll and its RAM copies and sets PPU control to $30.
;   L0062 ClearNameTable     code  Fills one name table with a tile and its attribute table with one attribute byte.
;   L0121 TableJump          code  Jumps to entry A of the address table that directly follows the JSR to this routine.
;   L0140 HideAllSprites     code  Hides all 64 sprites by setting the Y of every record in the sprite buffer to $F8.
;   L0162 ClearRam0300UpTo   code  Zeroes RAM from $0300 up to and including byte Y of page A, then marks DynTileBuf empty.
;   L0187 TurnOffAllVideo    code  Turns all rendering off: writes 0 to the PPU mask register and to CurPpuMask.
;   L0200 ReadInputs         code  Reads both controllers into ButtonsDown and ButtonsPressed (index 0 = controller 1, 1 = controller 2).
;   L0223 ReadOneController  code  Reads one controller until a reading repeats, then updates its ButtonsDown and ButtonsPressed.
; ============================================================================

; ----------------------------------------------------------------------------
; Zeroes the PPU scroll and its RAM copies and sets PPU control to $30.
; OUT:  CurHScroll = CurVScroll = 0; CurPpuControl = $30 (8x16 sprites, background patterns at $1000,
;       name table $2000, NMI disabled).
; ----------------------------------------------------------------------------
ResetPpuRegisters:
    LDA #$00
    STA PpuScroll
    STA CurHScroll
    STA PpuScroll
    STA CurVScroll
    LDA #$30
    STA PpuControl
    STA CurPpuControl
    RTS

; ----------------------------------------------------------------------------
; Fills one name table with a tile and its attribute table with one attribute byte.
; IN:   A = high byte of the name table address ($20, $24, $28 or $2C); X = tile; Y = attribute byte.
; OUT:  X and Y unchanged; [00] = A, [01] = X, [02] = Y; PPU address increment set to 1.
; NOTE: Writes $400 tile bytes (the 64 attribute bytes included), then rewrites the attribute table at
;       ((A + 3) * $100) + $C0. The only caller (ClearNameTableWithHiAddr) passes $20 or $28, tile $24,
;       attribute 0. With A < $20 the page count would come from [02] and no attributes would be
;       written; no caller does that.
; ----------------------------------------------------------------------------
ClearNameTable:
    STA $00
    STX $01
    STY $02
    LDA PpuStatus               ; Reset the PPU address write latch.
    LDA CurPpuControl           ; Clear bit 2: PPU address increment of 1.
    AND #$FB
    STA PpuControl
    STA CurPpuControl
    LDA $00
    STA PpuAddr
    LDY #$00
    STY PpuAddr

    ; Fill 4 pages of 256 bytes with the tile.
    LDX #$04
    CMP #$20
    BCS :+
    ; Not reached: every call has A >= $20. It would use the attribute byte as the page count.
    LDX $02
:
    LDY #$00
    LDA $01

@LoopTile:
    STA PpuData
    DEY
    BNE @LoopTile
    DEX
    BNE @LoopTile

    ; Fill the 64 attribute bytes at ((A + 3) * $100) + $C0 with the attribute byte.
    LDY $02
    LDA $00
    CMP #$20
    BCC @RestoreX               ; Not reached (A >= $20 in every call).
    ADC #$02                    ; Carry is set by the CMP: adds 3 ($20 -> $23, $28 -> $2B).
    STA PpuAddr
    LDA #$C0
    STA PpuAddr
    LDX #$40

@LoopAttr:
    STY PpuData
    DEX
    BNE @LoopAttr

@RestoreX:
    ; Return X as passed in. Y already holds its passed value.
    LDX $01
    RTS

; ----------------------------------------------------------------------------
; Jumps to entry A of the address table that directly follows the JSR to this routine.
; IN:   A = entry index (0-$7F); the table of 2-byte addresses starts right after the JSR instruction.
; OUT:  does not return to the caller: the target routine's RTS returns to the caller's caller.
;       [00:01] = return address popped from the stack (last byte of the JSR), [02:03] = target, Y = A*2+2.
; NOTE: X is not changed, so a caller's X (for example an object slot) reaches the target.
; ----------------------------------------------------------------------------
TableJump:
    ASL
    TAY
    PLA
    STA $00
    PLA
    STA $01
    INY                         ; The popped address is 1 below the table, so entry bytes are at +1, +2.
    LDA ($00), Y
    STA $02
    INY
    LDA ($00), Y
    STA $03
    JMP ($0002)

; ----------------------------------------------------------------------------
; Hides all 64 sprites by setting the Y of every record in the sprite buffer to $F8.
; OUT:  X = 0, Y = 0, A = $F8.
; ----------------------------------------------------------------------------
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

; ----------------------------------------------------------------------------
; Zeroes RAM from $0300 up to and including byte Y of page A, then marks DynTileBuf empty.
; IN:   A = high byte of the last address to clear (page, >= 3); Y = low byte of the last address.
; OUT:  [00:01] = $0200; Y = $FF; A = $FF; X unchanged.
; NOTE: ClearRam passes $07/$FE (clears $0300-$07FE); InitMode_EnterRoom passes $05/$1F on every
;       room entry (clears $0300-$051F).
; ----------------------------------------------------------------------------
ClearRam0300UpTo:
    STA $01
    LDA #$00
    STA $00

@Loop:
    ; Clear from Y down to 0 in the current page, then the whole of each lower page down to page 3.
    LDA #$00
    STA ($00), Y
    DEY
    CPY #$FF
    BNE @Loop
    DEC $01
    LDA $01
    CMP #$03
    BCS @Loop

    ; The loop zeroed the dynamic transfer buffer; an empty buffer must start with the end marker $FF.
    LDA #$FF
    STA DynTileBuf
    RTS

; ----------------------------------------------------------------------------
; Turns all rendering off: writes 0 to the PPU mask register and to CurPpuMask.
; ----------------------------------------------------------------------------
TurnOffAllVideo:
    LDA #$00
    STA PpuMask
    STA CurPpuMask
    RTS

; ----------------------------------------------------------------------------
; Reads both controllers into ButtonsDown and ButtonsPressed (index 0 = controller 1, 1 = controller 2).
; OUT:  ButtonsDown,X = buttons held; ButtonsPressed,X = buttons newly held since the last read.
;       Uses [00], [02], [03], [04].
; NOTE: Called once per frame by IsrNmi, except while IsSprite0CheckActive is set. Reads controller 1
;       by JSR and then falls into ReadOneController for controller 2.
; ----------------------------------------------------------------------------
ReadInputs:
    LDA #$01                    ; Strobe: latch the buttons of both controllers.
    STA Ctrl1
    LDA #$00                    ; End the strobe.
    STA Ctrl1
    STA $03                     ; [03], [04] = number of matching readings for controller 1, 2.
    STA $04
    TAX
    JSR ReadOneController
    INX

; ----------------------------------------------------------------------------
; Reads one controller until a reading repeats, then updates its ButtonsDown and ButtonsPressed.
; IN:   X = 0 for controller 1, 1 for controller 2; A = reading to compare the first read with (0 for
;       controller 1; controller 1's result for controller 2); [03+X] = 0.
; OUT:  ButtonsDown,X = accepted reading; ButtonsPressed,X = its bits that were not in the old
;       ButtonsDown,X; A = ButtonsDown,X.
; NOTE: Bits: $80 A, $40 B, $20 Select, $10 Start, 8 Up, 4 Down, 2 Left, 1 Right.
; NOTE: Each pass strobes again and reads 8 bits. A pass counts when it equals the pass before it
;       ([02]); the reading is accepted on the second counted pass (the count in [03+X] is not reset by
;       a mismatch). The first pass is compared with the A passed in: 0 for controller 1 (2 passes if
;       nothing is held, at least 3 otherwise), controller 1's result for controller 2.
; ----------------------------------------------------------------------------
ReadOneController:
    ; [02] = previous reading in this loop; [03+X] = passes that matched their predecessor.
    STA $02
    LDA #$01                    ; Strobe again, so that every pass starts from the first button (A).
    STA Ctrl1
    LDA #$00
    STA Ctrl1
    LDY #$08

@Read:
    LDA Ctrl1, X
    LSR
    ROL ButtonsPressed, X       ; Bit 0 (standard controller) into ButtonsPressed, used here as scratch.
    LSR
    ROL $00                     ; Bit 1 (expansion port controller) into [00].
    DEY
    BNE @Read
    LDA ButtonsPressed, X
    CMP $02
    BNE ReadOneController       ; Differs from the previous pass: read again.
    INC $03, X
    LDY $03, X
    CPY #$02
    BCC ReadOneController       ; Fewer than two matching passes: read again.
    LDA $00                     ; Merge the expansion port buttons of the last pass.
    ORA ButtonsPressed, X
    STA ButtonsPressed, X
    PHA
    EOR ButtonsDown, X          ; Bits that changed since the previous frame's read,
    AND ButtonsPressed, X       ; and are held now:
    STA ButtonsPressed, X       ; ButtonsPressed = newly pressed buttons.
    PLA
    STA ButtonsDown, X
    RTS
