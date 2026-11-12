; ============================================================================
; FILE:     bank1/map-marker-curtain.asm BANK: 1 (stored at $A500-$B76F; copied to RAM and run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 4134-4320
; SUBJECTS: (none)
; TOPIC:    Status bar map marker, curtain effect, 16-bit add helpers
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0023 UpdatePlayerPositionMarker      code
;   L0035 UpdatePositionMarker            code  Params:
;   L0087 L6AAF_Exit                      code
;   L0090 UpdateWorldCurtainEffect_Bank2  code
;   L0101 UpdateWorldCurtainEffect        code  Params:
;   L0135 Add1ToInt16At0                  code  Params:
;   L0146 AddToInt16At0                   code  Params:
;   L0158 Add1ToInt16At2                  code  Params:
;   L0168 AddToInt16At2                   code  Params:
;   L0179 Add1ToInt16At4                  code  Params:
;   L0189 AddToInt16At4                   code  Params:
;   L0201 Sub1FromInt16At4                code  Params:
; ============================================================================

UpdatePlayerPositionMarker:
    LDA GameMode                ; Return if mode 9 or whirlwind teleporting.
    CMP #$09
    BEQ L6AAF_Exit
    LDX WhirlwindTeleportingState
    BNE L6AAF_Exit
    LDA RoomId
    LDX #$00

; Params:
; A: room ID
;
UpdatePositionMarker:
    PHA                         ; Push the room ID.
    AND #$70                    ; Sanitize the row of the room ID.
    LSR                         ; The rows in status bar map are 4 pixels tall.
    LSR
    ADC #$17                    ; Add $17 pixels to Y coordinate.
    STA Sprites+84, X           ; Place the position marker vertically.
    LDA #$11                    ; Map X at $11 in OW.
    LDY CurLevel
    BEQ :+
    LDA #$12                    ; Map X at $12 in UW.
:
    STA $00                     ; Store the appropriate status bar map left coordinate in [$00].
    PLA                         ; Pop the room ID.
    AND #$0F                    ; Isolate the column of the room ID.
    CPY #$00
    BEQ :+
    ASL                         ; In UW, columns in status bar map are 8 pixels wide.
:
    ASL                         ; In OW, they are 4 pixels wide.
    ASL
    ADC $00                     ; Add the status bar map left coordinate.
    ADC LevelInfo_StatusBarMapXOffset    ; Add a horizontal offset from level info to center the map.
    STA Sprites+87, X           ; Place the position marker horizontally.
    LDA #$3E
    STA Sprites+85, X           ; Use the position marker tile.
    LDA #$00
    CPX #$00
    BEQ @SetAttr                ; If we're updating the triforce marker,
    LDA #$03                    ; Set up a default "inactive" palette (attribute), which means "not gotten".
    PHA
    LDY CurLevel
    CPY #$09
    BEQ :+                      ; If you're in level 9, use the "active" palette only.
    LDA InvTriforce
    AND LevelMasks-1, Y
    BNE @PopSetAttr             ; If player hasn't gotten the piece,
:
    LDA FrameCounter
    AND #$1F
    CMP #$10
    BCS @PopSetAttr             ; then every 16 frames,
    PLA
    LDA #$02                    ; switch between two palettes.
    PHA

@PopSetAttr:
    PLA

@SetAttr:
    STA Sprites+86, X

L6AAF_Exit:
    RTS

UpdateWorldCurtainEffect_Bank2:
    JSR UpdateWorldCurtainEffect
    LDA #$02
    JMP SwitchBank_Local2

; Params:
; [$7C]/ObjX[$C]: column index that will be decreased + 1
; [$7D]/ObjX[$D]: column index that will be increased + 1
;
; Switches to bank 5.
;
UpdateWorldCurtainEffect:
    LDA ObjTimer
    BNE @Exit                   ; Delay until ObjTimer[0] expires.
    LDA #$01                    ; Start with column index in [$7D] ObjX[$D].
    STA $0A

@LoopColumn:
    LDX $0A
    LDA ObjX+12, X              ; Get current column index.
    STA CurColumn               ; Set CurColumn for column copying routine to use.
    LDA #$05
    JSR SwitchBank_Local5
    LDA #$0E                    ; Set vertical nametable mirroring.
    JSR SetMMC1Control_Local5
    JSR CopyColumnToTileBuf
    DEC $0A                     ; Now reference the column with index in [$7C] ObjX[$C].
    BPL @LoopColumn             ; If we copied both columns, then quit.
    LDA #$FF                    ; Invalidate CurColumn
    STA CurColumn
    LDA #$05                    ; Set a delay of 4 frames (5-1) in ObjTimer[0].
    STA ObjTimer
    DEC ObjX+12                 ; Change the column indexes.
    INC ObjX+13

@Exit:
    RTS

; Params:
; [$00:01]: a 16-bit value to increment
;
; Returns:
; [$00:01]: sum
; A: low byte of sum
;
Add1ToInt16At0:
    LDA #$01

; Params:
; A: value 1
; [$00:01]: value 2
;
; Returns:
; [$00:01]: sum
; A: low byte of sum
;
AddToInt16At0:
    CLC
    ADC $00
    STA $00
    BCC :+
    INC $01
:
    RTS

; Params:
; [$02:03]: a 16-bit value to increment
;
Add1ToInt16At2:
    LDA #$01

; Params:
; A: value 1
; [$02:03]: value 2
;
; Returns:
; [$02:03]: sum
;
AddToInt16At2:
    CLC
    ADC $02
    STA $02
    BCC :+
    INC $03
:
    RTS

; Params:
; [$04:05]: a 16-bit value to increment
Add1ToInt16At4:
    LDA #$01

; Params:
; A: value 1
; [$04:05]: value 2
;
; Returns:
; [$04:05]: sum
;
AddToInt16At4:
    CLC
    ADC $04
    STA $04
    BCC :+
    INC $05
:
    RTS

; Params:
; [$04:05]: a 16-bit value to decrease
;
Sub1FromInt16At4:
    LDA $04
    SEC
    SBC #$01
    STA $04
    BCS :+
    DEC $05
:
    RTS
