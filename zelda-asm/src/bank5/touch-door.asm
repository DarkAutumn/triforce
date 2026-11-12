; ============================================================================
; FILE:     bank5/touch-door.asm         BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 3909-4017
; SUBJECTS: (none)
; TOPIC:    Door type handlers: wall, false wall, bombable, key, shutter
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0029 TouchDoor            code  Params:
;   L0033 TouchDoor_JumpTable  data
;   L0043 TouchDoorWall        code
;   L0047 TouchDoorOpen        code
;   L0050 TouchDoorFalse       code
;   L0066 TouchDoorBombable    code
;   L0073 TouchDoorShutter     code
;   L0092 TouchDoorKey         code
;   L0121 BlockAtWall          code
;   L0124 BlockUntilTime       code
;   L0129 L15292_Exit          code
; ============================================================================

; Params:
; [0C]: door direction
; [0E]: reverse index of direction
;
; Returns:
; [0E]: untouched, or $FF if blocked
;
TouchDoor:
    AND #$07
    JSR TableJump

TouchDoor_JumpTable:
    .ADDR TouchDoorOpen
    .ADDR TouchDoorWall
    .ADDR TouchDoorFalse
    .ADDR TouchDoorFalse
    .ADDR TouchDoorBombable
    .ADDR TouchDoorKey
    .ADDR TouchDoorKey
    .ADDR TouchDoorShutter

TouchDoorWall:
    LDY #$FF
    STY $0E

TouchDoorOpen:
    RTS

TouchDoorFalse:
    ; At first, Link's timer = 0. So set it to $18 frames, and block movement.
    ; Subsequently, block movement until timer = 1.
    LDA ObjTimer
    BEQ @SetTimer
    CMP #$01
    BNE @BlockMovement
    RTS

@SetTimer:
    LDA #$18
    STA ObjTimer

@BlockMovement:
    JMP TouchDoorWall

TouchDoorBombable:
    ; Block movement, if this door's direction is not in the open door mask.
    LDA $0C
    AND CurOpenedDoors
    BEQ TouchDoorWall
    RTS

TouchDoorShutter:
    ; If a door is triggered or this door wasn't already opened,
    ; then block movement.
    LDA TriggeredDoorCmd
    BNE TouchDoorWall
    LDA $0C
    AND CurOpenedDoors
    BEQ TouchDoorWall

    ; TODO: ?
    AND $0519
    BEQ :+
    BNE BlockUntilTime          ; Go block movement while timer <> 0.
:
    LDA $0519
    ORA $0C
    STA $0519
    RTS

TouchDoorKey:
    ; If this door was already opened, return.
    LDA $0C
    AND CurOpenedDoors
    BNE L15292_Exit

    ; If a door is triggered, go block movement while Link's timer <> 0.
    LDA TriggeredDoorCmd
    BNE BlockUntilTime

    ; If we don't have the magic key nor any normal keys, 
    ; go block movement.
    LDA InvMagicKey
    BNE @TriggerDoor
    LDA InvKeys
    BEQ BlockAtWall

    ; If we don't have the magic key, decrease the key count.
    DEC InvKeys

@TriggerDoor:
    ; Trigger this door to open.
    LDA $0C
    JSR TriggerOpenDoor

    ; Set player's timer to block for $20 frames.
    LDA #$20
    STA ObjTimer

BlockAtWall:
    JMP TouchDoorWall

BlockUntilTime:
    ; Block movement while Link's timer <> 0.
    LDA ObjTimer
    BNE BlockAtWall

L15292_Exit:
    RTS
