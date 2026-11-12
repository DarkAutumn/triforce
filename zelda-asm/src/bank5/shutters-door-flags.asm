; ============================================================================
; FILE:     bank5/shutters-door-flags.asm BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 2099-2182
; SUBJECTS: (none)
; TOPIC:    Saved "door opened" room flags (set and clear), the per-frame check that opens a dungeon
;           room's shutter doors once they have been triggered, and the entry that starts a door's
;           opening animation.
; SEE ALSO: bank5/underworld-secrets.asm (TriggerShutters sets ShutterTrigger);
;           bank5/update-doors.asm (UpdateDoors runs the open/close command; calls SetDoorFlag and
;           ResetDoorFlag); bank5/touch-door.asm (a key door calls TriggerOpenDoor);
;           bank5/lay-out-doors.asm (SetDoorFlag for opened key doors and bombed walls);
;           bank5/fill-walls.asm (FindDoorTypeByDoorBit); bank7/next-room-room-flags.asm
;           (GetRoomFlags, LevelMasks); bank7/mode5-play-update.asm (calls CheckShutters each frame
;           in a dungeon, after CheckUnderworldSecrets and before UpdateDoors).
; KEY FACTS:
;   - A dungeon room's flag byte keeps one "door opened" bit per side: bit 0 right (E), bit 1 left
;     (W), bit 2 down (S), bit 3 up (N), the door direction index being 0-3 in that order.
;     SetDoorFlag, ResetDoorFlag.
;   - Shutters (door type 7) open only while ShutterTrigger is nonzero. CheckShutters looks at the
;     sides up, down, left, right in that order and starts the first shutter not yet in
;     CurOpenedDoors. CheckShutters.
;   - Only one door command runs at a time (TriggeredDoorCmd): a room with several shutters opens
;     them one after another, each waiting for the previous command to finish. CheckShutters.
;   - ShutterTrigger is cleared on the first frame on which every side is either open or not a
;     shutter, so a shutter that closes later needs a new trigger. CheckShutters.
;   - Opening any door = TriggeredDoorDir = its direction bit, TriggeredDoorCmd = 6. TriggerOpenDoor.
;   - TriggerOpenDoor does not test whether a command is already running; its callers do.
;     TriggerOpenDoor.
; CONTENTS: (generated; L = line in this file)
;   L0041 SetDoorFlag      code  Sets the saved "door opened" flag of one side of the current room.
;   L0052 ResetDoorFlag    code  Clears the saved "door opened" flag of one side of the current room.
;   L0068 CheckShutters    code  While ShutterTrigger is set, starts opening the room's next closed shutter door.
;   L0117 TriggerOpenDoor  code  Starts a door's opening animation.
; ============================================================================

; ----------------------------------------------------------------------------
; Sets the saved "door opened" flag of one side of the current room.
; IN:   X = door direction index (0 right, 1 left, 2 down, 3 up).
; OUT:  Y = RoomId; [00:01] = address of the level's room flags; flag bit LevelMasks[X] set.
; ----------------------------------------------------------------------------
SetDoorFlag:
    JSR GetRoomFlags
    ORA LevelMasks, X           ; Bit 1 << X.
    STA ($00), Y
    RTS

; ----------------------------------------------------------------------------
; Clears the saved "door opened" flag of one side of the current room.
; IN:   X = door direction index (0 right, 1 left, 2 down, 3 up).
; OUT:  Y = RoomId; [00:01] = address of the level's room flags; flag bit LevelMasks[X] cleared.
; ----------------------------------------------------------------------------
ResetDoorFlag:
    JSR GetRoomFlags
    LDA LevelMasks, X
    EOR #$FF
    AND ($00), Y
    STA ($00), Y
    RTS

; ----------------------------------------------------------------------------
; While ShutterTrigger is set, starts opening the room's next closed shutter door.
; IN:   ShutterTrigger; CurOpenedDoors; TriggeredDoorCmd.
; OUT:  TriggeredDoorDir/TriggeredDoorCmd = 6 for one closed shutter, or ShutterTrigger = 0 when no
;       closed shutter is left. Uses [0E], and [00]-[03] through FindDoorTypeByDoorBit.
; NOTE: One shutter per door command; while a command is running the routine returns and tries
;       again on a later frame.
; ----------------------------------------------------------------------------
CheckShutters:
    ; If the shutters have not been triggered to open, go leave the trigger at 0 and return.
    LDA ShutterTrigger
    BEQ @ClearTrigger

    ; Look for a closed shutter. [0E] is the direction bit of the side, starting with up (8),
    ; then down (4), left (2), right (1).
    LDA #$08
    STA $0E

@LoopOpenedDoorBit:
    ; If this side is not in the open-door mask, go see if it's a shutter.
    LDA $0E
    AND CurOpenedDoors
    BEQ @TriggerIfShutter

@NextLoopOpenedDoorBit:
    ; This side is open or is not a shutter. Go on to the next direction bit, until none is left.
    LSR $0E
    LDA $0E
    BNE @LoopOpenedDoorBit
@ClearTrigger:
    ; No closed shutter was found: turn off the shutter trigger.
    LDA #$00
    STA ShutterTrigger
    RTS

@TriggerIfShutter:
    ; Get the door type of this side ([02] = direction bit).
    LDA $0E
    STA $02
    JSR FindDoorTypeByDoorBit

    ; If it's not a shutter (door type 7), go check the next side.
    CMP #$07
    BNE @NextLoopOpenedDoorBit

    ; A closed shutter. If a door command is already running, return and try again later.
    LDA a:TriggeredDoorCmd
    BNE :+

    ; Else start opening the door in this direction.
    LDA $02

; ----------------------------------------------------------------------------
; Starts a door's opening animation.
; IN:   A = direction bit of the door (1 right, 2 left, 4 down, 8 up).
; OUT:  TriggeredDoorDir = A; TriggeredDoorCmd = 6 (open), which UpdateDoors then carries out.
; ----------------------------------------------------------------------------
TriggerOpenDoor:
    STA a:TriggeredDoorDir
    LDA #$06                    ; Open door command
    STA a:TriggeredDoorCmd
:
    RTS
