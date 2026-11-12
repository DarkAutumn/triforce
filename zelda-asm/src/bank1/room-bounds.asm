; ============================================================================
; FILE:     bank1/room-bounds.asm        BANK: 1 (stored at $A500-$B76F; copied to RAM and run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 3306-3509
; SUBJECTS: (none)
; TOPIC:    Silence sound, Link sprites behind doors, room boundary checks
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0020 SilenceAllSound                       code
;   L0029 SpriteRelativeExtents                 data
;   L0032 ShowLinkSpritesBehindHorizontalDoors  code
;   L0074 BoundDirectionHorizontally            code  Params:
;   L0125 BoundDirectionReturn                  code
;   L0142 BoundDirectionVertically              code  Params:
;   L0193 L6825_Exit                            code
;   L0206 BoundByRoomWithA                      code  Params:
;   L0219 BoundByRoom                           code  Params:
; ============================================================================

SilenceAllSound:
    LDA #$80
    STA Tune0Request
    STA EffectRequest
    ASL
    STA CurTune0
    STA CurTune1
    RTS

SpriteRelativeExtents:
    .BYTE $08, $00

ShowLinkSpritesBehindHorizontalDoors:
    LDY #$0A
    LDX #$00
    LDA ObjX, X
    STA $00
    LDX #$01

@LoopSprite:
    LDA $00
    CLC
    ADC SpriteRelativeExtents, X
    CMP #$E9
    BCS @AddPriority
    CMP #$10
    BCS @NextSprite

@AddPriority:
    LDA Sprites+64, Y
    ORA #$20
    STA Sprites+64, Y

@NextSprite:
    INY
    INY
    INY
    INY
    CPY #$00
    BNE :+
    LDY #$20                    ; UNKNOWN: This looks unreachable.
:
    DEX
    BPL @LoopSprite
    RTS

; Params:
; X: object index
; [0F]: direction
;
; Returns:
; Y: direction of boundary that was crossed
; [0F]: original direction, or 0 if boundary was crossed
;
BoundDirectionHorizontally:
    LDY #$02                    ; Checking left direction.
    LDA ObjX, X
    STA $00                     ; [00] holds X coordinate
    CPX #$00
    BEQ @Left

    ; If object slot >= $D (items and weapons) or object type = $5C (boomerang),
    ; then add $B to X coordinate.
    CPX #$0D
    BCS :+
    LDA ObjType, X
    CMP #$5C
    BNE @Left
:
    LDA $00
    CLC
    ADC #$0B
    STA $00

@Left:
    ; Check left boundary.
    ; If X coordinate crosses (<) left bound, go reset the direction.
    LDA $00
    CMP RoomBoundLeft
    BCC BoundDirectionReturn

    ; If object is not the player and (slot >= $D or type = $5C),
    ; then subtract $17 from X coordinate in order to check the
    ; right bound.
    CPX #$00
    BEQ @Right                  ; If object is the player, skip this.
    CPX #$0D
    BCS :+
    LDA ObjType, X
    CMP #$5C
    BNE @Right                  ; If not boomerang type, skip this.
:
    LDA $00
    SEC
    SBC #$17
    STA $00

@Right:
    ; Check right boundary.
    ; If X coordinate is within (<) right bound, return and leave direction alone.
    LDY #$01
    LDA $00
    CMP RoomBoundRight
    BCC L6825_Exit

BoundDirectionReturn:
    ; If the reference direction in Y register is 0, then
    ; return and leave direction alone.
    ; Else reset the target direction.
    TYA
    AND $0F
    BEQ L6825_Exit
    JMP ResetMovingDir

; Params:
; X: object index
; [0F]: direction
;
; Returns:
; Y: direction of boundary that was crossed
; [0F]: original direction, or 0 if boundary was crossed
;
BoundDirectionVertically:
    LDY #$08                    ; Checking up direction.
    LDA ObjY, X
    STA $00                     ; [00] holds Y coordinate
    CPX #$00
    BEQ @Up

    ; If object slot >= $D (items and weapons) or object type = $5C (boomerang),
    ; then add $F to X coordinate.
    CPX #$0D
    BCS :+
    LDA ObjType, X
    CMP #$5C
    BNE @Up
:
    LDA $00
    CLC
    ADC #$0F
    STA $00

@Up:
    ; Check up boundary.
    ; If Y coordinate crosses (<) up bound, go reset the direction.
    LDA $00
    CMP RoomBoundUp
    BCC BoundDirectionReturn

    ; If object is not the player and (slot >= $D or type = $5C),
    ; then subtract $21 from Y coordinate in order to check the
    ; down bound.
    CPX #$00
    BEQ @Down                   ; If object is the player, skip this.
    CPX #$0D
    BCS :+
    LDA ObjType, X
    CMP #$5C
    BNE @Down                   ; If not boomerang type, skip this.
:
    LDA $00
    SEC
    SBC #$21
    STA $00

@Down:
    ; Check down boundary.
    ; If Y coordinate is within (<) down bound, return and leave direction alone.
    LDY #$04
    LDA $00
    CMP RoomBoundDown
    BCS BoundDirectionReturn    ; If Y coordinate crosses (>=) right boundary, go reset target direction.

L6825_Exit:
    RTS

; Params:
; A: direction
; X: object index
;
; Returns:
; A: original direction, or 0 if boundary was crossed
; Y: direction of boundary that was crossed
; Z: 1 if boundary was crossed
; [0F]: original direction, or 0 if boundary was crossed
;
BoundByRoomWithA:
    STA $0F

; Params:
; X: object index
; [0F]: direction
;
; Returns:
; A: original direction, or 0 if boundary was crossed
; Y: direction of boundary that was crossed
; Z: 1 if boundary was crossed
; [0F]: original direction, or 0 if boundary was crossed
;
BoundByRoom:
    JSR BoundDirectionHorizontally
    JSR BoundDirectionVertically
    LDA $0F
    RTS
