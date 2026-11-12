; ============================================================================
; FILE:     bank5/touch-door.asm         BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 3909-4017
; SUBJECTS: (none)
; TOPIC:    What happens when Link pushes into a dungeon doorway: one handler per door type (open,
;           wall, false wall, bombable wall, locked door, shutter) that lets him pass or blocks him.
; SEE ALSO: bank5/doorway.asm (CheckDoorway calls TouchDoor every frame Link pushes into a doorway);
;           bank5/shutters-door-flags.asm (TriggerOpenDoor, CheckShutters, SetDoorFlag);
;           bank5/update-doors.asm (UpdateDoors runs the open and close commands, sets CurOpenedDoors
;           and saves the open flag in this room and the next); bank7/bomb.asm (a bomb near a
;           bombable wall triggers its open command); bank5/fill-walls.asm (FindDoorTypeByDoorBit).
; KEY FACTS:
;   - Door types: 0 open, 1 wall, 2 and 3 false wall, 4 bombable wall, 5 and 6 locked door,
;     7 shutter. TouchDoor_JumpTable
;   - A handler blocks Link by setting [0E] = $FF; returning without doing so lets him pass. The
;     block has an effect only while DoorwayDir = 0 (see CheckDoorway in bank5/doorway.asm).
;     TouchDoorWall
;   - False wall: the first touch sets Link's ObjTimer to $18 and blocks; he passes on the frame he
;     touches it with the timer at 1. If the timer reaches 0 first, the wait starts over.
;     TouchDoorFalse
;   - Bombable wall: passable only when its bit is set in CurOpenedDoors. TouchDoorBombable
;   - Locked door: with the magic key (InvMagicKey) no key is spent; otherwise one key (InvKeys) is
;     spent, and with none the door is a wall. Unlocking sets the open command and Link's ObjTimer to
;     $20; Link is blocked until the door's bit is in CurOpenedDoors (the open command takes about
;     9 frames in UpdateDoors) or that timer runs out. TouchDoorKey
;   - Locked door while a door command is running (TriggeredDoorCmd <> 0) and Link's ObjTimer is 0:
;     the handler lets him pass without a key and without opening the door; CheckDoorway then sets
;     DoorwayDir and Link walks through the closed door to the next room. The close command sets
;     his timer to $30 and a key unlock to $20; a shutter opened by CheckShutters and a wall opened
;     by a bomb do not set it. TouchDoorKey, BlockUntilTime
;   - Shutter: a wall unless no door command is running and its bit is set in CurOpenedDoors.
;     TouchDoorShutter
;   - Only one door command exists (TriggeredDoorCmd), so a key cannot unlock a door while another
;     door is opening or closing. TouchDoorKey
; CONTENTS: (generated; L = line in this file)
;   L0057 TouchDoor            code  Runs the handler of the door type Link is pushing against in a dungeon doorway.
;   L0062 TouchDoor_JumpTable  data  TABLE: door type handlers; index = door type (0-7, from room attrs A and B); value = address.
;   L0075 TouchDoorWall        code  Door type 1 (wall), and the shared "blocked" exit of the other handlers: sets [0E] = $FF.
;   L0080 TouchDoorOpen        code  Door type 0 (open doorway): lets Link pass. Also the RTS of TouchDoorWall.
;   L0090 TouchDoorFalse       code  Door types 2 and 3 (false wall): makes Link push for the length of a $18-frame timer, then pass.
;   L0108 TouchDoorBombable    code  Door type 4 (bombable wall): lets Link pass only when the wall has been opened.
;   L0123 TouchDoorShutter     code  Door type 7 (shutter): lets Link pass only when it is open and no door is opening or closing.
;   L0157 TouchDoorKey         code  Door types 5 and 6 (locked door): passes if open; else spends a key (or uses the magic key) to open.
;   L0186 BlockAtWall          code  Blocks Link: jumps to TouchDoorWall. Shared by TouchDoorKey and BlockUntilTime.
;   L0190 BlockUntilTime       code  Blocks Link while his ObjTimer (slot 0) is not 0; lets him pass when it is 0.
;   L0195 L15292_Exit          code  Shared RTS of TouchDoorKey and BlockUntilTime: Link passes.
; ============================================================================

; ----------------------------------------------------------------------------
; Runs the handler of the door type Link is pushing against in a dungeon doorway.
; IN:   A = door type (low 3 bits used); [0C] = the door's direction bit (1 right, 2 left, 4 down,
;       8 up); [0E] = reverse direction index of that direction
; OUT:  [0E] = $FF when the door blocks Link, else unchanged
; NOTE: Called by CheckDoorway on every frame Link is in the doorway span with the input direction
;       equal to the door's direction, also while he walks through it.
; ----------------------------------------------------------------------------
TouchDoor:
    AND #$07
    JSR TableJump

; TABLE: door type handlers; index = door type (0-7, from room attrs A and B); value = address.
TouchDoor_JumpTable:
    .ADDR TouchDoorOpen         ; 0: open doorway
    .ADDR TouchDoorWall         ; 1: wall
    .ADDR TouchDoorFalse        ; 2: false wall
    .ADDR TouchDoorFalse        ; 3: false wall
    .ADDR TouchDoorBombable     ; 4: bombable wall
    .ADDR TouchDoorKey          ; 5: locked door
    .ADDR TouchDoorKey          ; 6: locked door
    .ADDR TouchDoorShutter      ; 7: shutter

; ----------------------------------------------------------------------------
; Door type 1 (wall), and the shared "blocked" exit of the other handlers: sets [0E] = $FF.
; ----------------------------------------------------------------------------
TouchDoorWall:
    LDY #$FF
    STY $0E

; Door type 0 (open doorway): lets Link pass. Also the RTS of TouchDoorWall.
TouchDoorOpen:
    RTS

; ----------------------------------------------------------------------------
; Door types 2 and 3 (false wall): makes Link push for the length of a $18-frame timer, then pass.
; NOTE: Uses Link's ObjTimer (slot 0). Timer 0: set it to $18 and block. Timer 1: pass. Any other
;       value: block. Link must be pushing on the frame the timer reads 1; if it runs out to 0
;       unseen, the next touch starts another $18 frames. A timer already running for another
;       reason is used as it is.
; ----------------------------------------------------------------------------
TouchDoorFalse:
    LDA ObjTimer
    BEQ @SetTimer
    CMP #$01
    BNE @BlockMovement
    RTS                         ; Timer = 1: pass.

@SetTimer:
    LDA #$18
    STA ObjTimer

@BlockMovement:
    JMP TouchDoorWall

; ----------------------------------------------------------------------------
; Door type 4 (bombable wall): lets Link pass only when the wall has been opened.
; IN:   [0C] = the door's direction bit
; ----------------------------------------------------------------------------
TouchDoorBombable:
    LDA $0C
    AND CurOpenedDoors
    BEQ TouchDoorWall           ; Not open: wall.
    RTS

; ----------------------------------------------------------------------------
; Door type 7 (shutter): lets Link pass only when it is open and no door is opening or closing.
; IN:   [0C] = the door's direction bit
; NOTE: RAM $0519 (no symbol) collects the direction bits of open shutters Link has touched since
;       the room was entered; it is zeroed with $0300-$051F by InitMode_EnterRoom. The first touch
;       of an open shutter passes and sets its bit. Every later touch (CheckDoorway touches it again
;       on each frame of the walk through) passes only while Link's ObjTimer is 0. Why the timer is
;       consulted here is (unverified).
; ----------------------------------------------------------------------------
TouchDoorShutter:
    ; A door command is running, or this shutter is not open: wall.
    LDA TriggeredDoorCmd
    BNE TouchDoorWall
    LDA $0C
    AND CurOpenedDoors
    BEQ TouchDoorWall

    ; A = the shutter's bit. Touched before in this room: block while Link's timer <> 0.
    AND $0519
    BEQ :+
    BNE BlockUntilTime
:
    ; First touch: record the bit and pass.
    LDA $0519
    ORA $0C
    STA $0519
    RTS

; ----------------------------------------------------------------------------
; Door types 5 and 6 (locked door): passes if open; else spends a key (or uses the magic key) to open.
; IN:   [0C] = the door's direction bit
; NOTE: Order of tests: (1) bit set in CurOpenedDoors: pass. (2) a door command is running
;       (TriggeredDoorCmd <> 0, this door's or any other's): blocked while Link's ObjTimer <> 0,
;       pass when it is 0, with no key spent and the door still closed. (3) InvMagicKey <> 0: open,
;       no key spent. (4) InvKeys = 0: wall. (5) InvKeys - 1, open.
;       Opening = TriggerOpenDoor (command 6) and Link's ObjTimer = $20; Link is blocked on that
;       frame, and after it by case (2) until case (1) applies or the timer reaches 0.
; NOTE: Case (2) with the timer at 0 is a way through a locked door without a key: the pass makes
;       CheckDoorway set DoorwayDir, after which nothing stops Link in the doorway. It needs a door
;       command that did not set Link's timer: a shutter opening (CheckShutters) or a bombed wall
;       opening (bank7/bomb.asm), each about 9 frames long. UpdateDoors sets the timer to $30 for
;       a close command.
; ----------------------------------------------------------------------------
TouchDoorKey:
    ; Already open: pass.
    LDA $0C
    AND CurOpenedDoors
    BNE L15292_Exit

    ; A door command is running: block only while Link's timer <> 0.
    LDA TriggeredDoorCmd
    BNE BlockUntilTime

    ; No magic key and no keys: wall.
    LDA InvMagicKey
    BNE @TriggerDoor
    LDA InvKeys
    BEQ BlockAtWall

    ; No magic key: spend a key.
    DEC InvKeys

@TriggerDoor:
    ; Set the command that opens this door.
    LDA $0C
    JSR TriggerOpenDoor

    ; Link's timer = $20 frames (keeps case (2) above blocking while the door opens).
    LDA #$20
    STA ObjTimer

; Blocks Link: jumps to TouchDoorWall. Shared by TouchDoorKey and BlockUntilTime.
BlockAtWall:
    JMP TouchDoorWall

; Blocks Link while his ObjTimer (slot 0) is not 0; lets him pass when it is 0.
BlockUntilTime:
    LDA ObjTimer
    BNE BlockAtWall

; Shared RTS of TouchDoorKey and BlockUntilTime: Link passes.
L15292_Exit:
    RTS
