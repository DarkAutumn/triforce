; ============================================================================
; FILE:     bank4/shoot-helpers-candle.asm BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 11311-11445
; SUBJECTS: (none)
; TOPIC:    Shoot helpers; candle room brightening, dark room test
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0034 _ShootIfWanted            code  Params:
;   L0053 ShootLimited              code  Params:
;   L0080 Shoot                     code  Params:
;   L0105 ReturnDidNotShoot         code
;   L0110 UpdateCandle              code
;   L0114 UpdateCandle_JumpTable    data
;   L0119 UpdateCandle_Begin        code
;   L0127 L_Candle_IncState         code
;   L0130 UpdateCandle_Done         code
;   L0133 L_Candle_StopBrightening  code
;   L0139 UpdateCandle_Brightening  code
;   L0151 IsDarkRoom_Bank4          code  Params:
; ============================================================================

; Params:
; A: shot object type
; X: shooter object index
;
; Returns:
; C: 1 if succeeded
; Y: shot slot
;
; Description:
; The shot starts in state $10.
;
_ShootIfWanted:
    STA $00

    ; If the monster does not want to shoot, then return C=0.
    LDA ObjWantsToShoot, X
    BEQ ReturnDidNotShoot

; Params:
; [00]: shot object type
;
; Returns:
; C: 1 if succeeded
; Y: shot slot
;
; Description:
; The shot starts in state $10.
;
; If no slot is found, return C=0.
;
ShootLimited:
    JSR FindEmptyMonsterSlot
    BEQ ReturnDidNotShoot

    ; If the object type to shoot is a true shot (projectile),
    ; and we're at the limit of active shots (4), then return C=0.
    LDA $00
    CMP #$53
    BCC Shoot
    LDA ActiveMonsterShots
    CMP #$04
    BCS ReturnDidNotShoot

    ; Else increase the number of active shots.
    INC ActiveMonsterShots

; Params:
; [00]: shot object type
; [59]: EmptyMonsterSlot
;
; Returns:
; C: 1 if succeeded
; Y: shot slot
;
; Description:
; The shot starts in state $10.
;
Shoot:
    LDX EmptyMonsterSlot
    LDA $00
    JSR SetTypeAndClearObject

    ; Set state to $10 and reset timer, so that the shot moves right away.
    LDY EmptyMonsterSlot
    LDX CurObjIndex
    LDA #$10
    STA a:ObjState, Y
    LDA #$00
    STA a:ObjTimer, Y

    ; Set the shot's direction and coordinates the same as the thrower's.
    LDA ObjDir, X
    STA a:ObjDir, Y
    LDA ObjX, X
    STA a:ObjX, Y
    LDA ObjY, X
    STA a:ObjY, Y

    ; Return C=1.
    SEC
    RTS

ReturnDidNotShoot:
    ; Return C=0.
    CLC
    RTS

UpdateCandle:
    LDA CandleState
    JSR TableJump

UpdateCandle_JumpTable:
    .ADDR UpdateCandle_Begin
    .ADDR UpdateCandle_Brightening
    .ADDR UpdateCandle_Done

UpdateCandle_Begin:
    LDY RoomId
    JSR IsDarkRoom_Bank4
    BEQ UpdateCandle_Done       ; If it's not a dark room, return.
    LDA #$C0                    ; Start a fade-to-light cycle (reverse of $40).
    STA FadeCycle
    INC BrighteningRoom

L_Candle_IncState:
    INC CandleState

UpdateCandle_Done:
    RTS

L_Candle_StopBrightening:
    ; Stop brightening the room, and go set the candle state to done.
    LDA #$00
    STA BrighteningRoom
    BEQ L_Candle_IncState

UpdateCandle_Brightening:
    ; Animating a brightening cycle.
    JSR AnimateWorldFading
    BEQ L_Candle_StopBrightening    ; If done fading, go stop brightening the room.
    RTS

; Params:
; Y: room ID
;
; Returns:
; A: $80 if dark, else 0
;
IsDarkRoom_Bank4:
    LDA CurLevel
    BEQ :+
    LDA LevelBlockAttrsE, Y
    AND #$80
:
    RTS
