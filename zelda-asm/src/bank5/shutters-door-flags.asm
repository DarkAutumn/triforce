; ============================================================================
; FILE:     bank5/shutters-door-flags.asm BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 2099-2182
; SUBJECTS: (none)
; TOPIC:    Door flags, CheckShutters, TriggerOpenDoor
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0022 SetDoorFlag      code  Params:
;   L0035 ResetDoorFlag    code  Params:
;   L0043 CheckShutters    code
;   L0093 TriggerOpenDoor  code  Params:
; ============================================================================

; Params:
; X: door direction index
;
; Returns:
; Y: room ID
; [00:01]: address of level block world flags
;
SetDoorFlag:
    JSR GetRoomFlags
    ORA LevelMasks, X
    STA ($00), Y
    RTS

; Params:
; X: door direction index
;
; Returns:
; Y: room ID
; [00:01]: address of level block world flags
;
ResetDoorFlag:
    JSR GetRoomFlags
    LDA LevelMasks, X
    EOR #$FF
    AND ($00), Y
    STA ($00), Y
    RTS

CheckShutters:
    ;
    ;
    ; If the shutters have not been triggered to open, return.
    LDA ShutterTrigger
    BEQ :+

    ; Loop to look for an unopened shutter.
    ; The loop variable is a direction bit in [0E], starting with up (8).
    LDA #$08
    STA $0E

@LoopOpenedDoorBit:
    ; If the door for the direction bit is not opened, go see if it's a shutter.
    LDA $0E
    AND CurOpenedDoors
    BEQ @TriggerIfShutter

@NextLoopOpenedDoorBit:
    ; This door was opened. Shift the direction bit right.
    ; Loop until direction bit = 0.
    LSR $0E
    LDA $0E
    BNE @LoopOpenedDoorBit
:
    ; If there were no unopened shutters, then turn off the shutter trigger.
    LDA #$00
    STA ShutterTrigger
    RTS

@TriggerIfShutter:
    ; Pass the direction bit as an argument to get the door type.
    LDA $0E
    STA $02
    JSR FindDoorTypeByDoorBit

    ; If it's not a shutter, go loop again.
    CMP #$07
    BNE @NextLoopOpenedDoorBit

    ; If there's already a triggered door command, quit.
    LDA a:TriggeredDoorCmd
    BNE :+

    ; Else set a command to open the door in this direction.
    LDA $02

; Params:
; A: direction
;
TriggerOpenDoor:
    STA a:TriggeredDoorDir
    LDA #$06                    ; Open door command
    STA a:TriggeredDoorCmd
:
    RTS
