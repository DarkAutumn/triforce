; ============================================================================
; FILE:     bank5/touch-door.asm         BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 3909-4017
; SUBJECTS: Door
; TOPIC:    What happens when Link pushes into a dungeon doorway: one handler per door type (open,
;           wall, false wall, bombable wall, locked door, shutter) that lets him pass or blocks him.
; SEE ALSO: bank5/doorway.asm (Doorway_CheckLink calls Door_Touch every frame Link pushes into a doorway);
;           bank5/shutters-door-flags.asm (Door_TriggerOpen, Shutter_OpenTriggered, RoomFlags_SetDoorOpened);
;           bank5/update-doors.asm (Door_Update runs the open and close commands, sets CurOpenedDoors
;           and saves the open flag in this room and the next); bank7/bomb.asm (a bomb near a
;           bombable wall triggers its open command); bank5/fill-walls.asm (Door_FindTypeByDirBit).
; KEY FACTS:
;   - Door types: 0 open, 1 wall, 2 and 3 false wall, 4 bombable wall, 5 and 6 locked door,
;     7 shutter. Door_Touch_JumpTable
;   - A handler blocks Link by setting [0E] = $FF; returning without doing so lets him pass. The
;     block has an effect only while DoorwayDir = 0 (see Doorway_CheckLink in bank5/doorway.asm).
;     Door_TouchWall
;   - False wall: the first touch sets Link's ObjTimer to $18 and blocks; he passes on the frame he
;     touches it with the timer at 1. If the timer reaches 0 first, the wait starts over.
;     Door_TouchFalseWall
;   - Bombable wall: passable only when its bit is set in CurOpenedDoors. Door_TouchBombable
;   - Locked door: with the magic key (InvMagicKey) no key is spent; otherwise one key (InvKeys) is
;     spent, and with none the door is a wall. Unlocking sets the open command and Link's ObjTimer to
;     $20; Link is blocked until the door's bit is in CurOpenedDoors (the open command takes about
;     9 frames in Door_Update) or that timer runs out. Door_TouchKey
;   - Locked door while a door command is running (TriggeredDoorCmd <> 0) and Link's ObjTimer is 0:
;     the handler lets him pass without a key and without opening the door; Doorway_CheckLink then sets
;     DoorwayDir and Link walks through the closed door to the next room. The close command sets
;     his timer to $30 and a key unlock to $20; a shutter opened by Shutter_OpenTriggered and a wall opened
;     by a bomb do not set it. Door_TouchKey, Door_Touch_BlockWhileTimer
;   - Shutter: a wall unless no door command is running and its bit is set in CurOpenedDoors.
;     Door_TouchShutter
;   - Only one door command exists (TriggeredDoorCmd), so a key cannot unlock a door while another
;     door is opening or closing. Door_TouchKey
; CONTENTS: (generated; L = line in this file)
;   L0057 Door_Touch                  code  Runs the handler of the door type Link is pushing against in a dungeon doorway.
;   L0062 Door_Touch_JumpTable        data  TABLE: door type handlers; index = door type (0-7, from room attrs A and B); value = address.
;   L0075 Door_TouchWall              code  Door type 1 (wall), and the shared "blocked" exit of the other handlers: sets [0E] = $FF.
;   L0080 Door_TouchOpen              code  Door type 0 (open doorway): lets Link pass. Also the RTS of Door_TouchWall.
;   L0090 Door_TouchFalseWall         code  Door types 2 and 3 (false wall): makes Link push for the length of a $18-frame timer, then pass.
;   L0108 Door_TouchBombable          code  Door type 4 (bombable wall): lets Link pass only when the wall has been opened.
;   L0126 Door_TouchShutter           code  Door type 7 (shutter): lets Link pass only when it is open and no door is opening or closing.
;   L0160 Door_TouchKey               code  Door types 5 and 6 (locked door): passes if open; else spends a key (or uses the magic key) to open.
;   L0189 Door_Touch_BlockAtWall      code  Blocks Link: jumps to Door_TouchWall. Shared by Door_TouchKey and Door_Touch_BlockWhileTimer.
;   L0193 Door_Touch_BlockWhileTimer  code  Blocks Link while his ObjTimer (slot 0) is not 0; lets him pass when it is 0.
;   L0198 Door_Touch_Return           code  Shared RTS of Door_TouchKey and Door_Touch_BlockWhileTimer: Link passes.
; ============================================================================

; ----------------------------------------------------------------------------
; Runs the handler of the door type Link is pushing against in a dungeon doorway.
; IN:   A = door type (low 3 bits used); [0C] = the door's direction bit (1 right, 2 left, 4 down,
;       8 up); [0E] = reverse direction index of that direction
; OUT:  [0E] = $FF when the door blocks Link, else unchanged
; NOTE: Called by Doorway_CheckLink on every frame Link is in the doorway span with the input direction
;       equal to the door's direction, also while he walks through it.
; ----------------------------------------------------------------------------
Door_Touch:
    AND #$07
    JSR Util_TableJump

; TABLE: door type handlers; index = door type (0-7, from room attrs A and B); value = address.
Door_Touch_JumpTable:
    .ADDR Door_TouchOpen         ; 0: open doorway
    .ADDR Door_TouchWall         ; 1: wall
    .ADDR Door_TouchFalseWall        ; 2: false wall
    .ADDR Door_TouchFalseWall        ; 3: false wall
    .ADDR Door_TouchBombable     ; 4: bombable wall
    .ADDR Door_TouchKey          ; 5: locked door
    .ADDR Door_TouchKey          ; 6: locked door
    .ADDR Door_TouchShutter      ; 7: shutter

; ----------------------------------------------------------------------------
; Door type 1 (wall), and the shared "blocked" exit of the other handlers: sets [0E] = $FF.
; ----------------------------------------------------------------------------
Door_TouchWall:
    LDY #$FF
    STY $0E

; Door type 0 (open doorway): lets Link pass. Also the RTS of Door_TouchWall.
Door_TouchOpen:
    RTS

; ----------------------------------------------------------------------------
; Door types 2 and 3 (false wall): makes Link push for the length of a $18-frame timer, then pass.
; NOTE: Uses Link's ObjTimer (slot 0). Timer 0: set it to $18 and block. Timer 1: pass. Any other
;       value: block. Link must be pushing on the frame the timer reads 1; if it runs out to 0
;       unseen, the next touch starts another $18 frames. A timer already running for another
;       reason is used as it is.
; ----------------------------------------------------------------------------
Door_TouchFalseWall:
    LDA ObjTimer
    BEQ @SetTimer
    CMP #$01
    BNE @BlockMovement
    RTS                         ; Timer = 1: pass.

@SetTimer:
    LDA #$18
    STA ObjTimer

@BlockMovement:
    JMP Door_TouchWall

; ----------------------------------------------------------------------------
; Door type 4 (bombable wall): lets Link pass only when the wall has been opened.
; IN:   [0C] = the door's direction bit
; ----------------------------------------------------------------------------
Door_TouchBombable:
    LDA $0C
    AND CurOpenedDoors
    BEQ Door_TouchWall           ; Not open: wall.
    RTS

; ----------------------------------------------------------------------------
; Door type 7 (shutter): lets Link pass only when it is open and no door is opening or closing.
; IN:   [0C] = the door's direction bit
; NOTE: ShutterTouchedDirs collects the direction bits of open shutters Link has touched since
;       the room was entered; it is zeroed with $0300-$051F by Room_EnterCommon. The first touch
;       of an open shutter passes and sets its bit. Every later touch (Doorway_CheckLink touches it again
;       on each frame of the walk through) passes only while Link's ObjTimer is 0. A block has an
;       effect only while DoorwayDir = 0, so the timer cannot stop Link during the walk through
;       that the first touch began; it holds him only when he has left the doorway span and
;       touches the same open shutter again while his timer runs (set to $18 by a false wall, $20
;       by a key unlock, $30 by a door closing: Door_TouchFalseWall, Door_TouchKey, Door_Update).
; ----------------------------------------------------------------------------
Door_TouchShutter:
    ; A door command is running, or this shutter is not open: wall.
    LDA TriggeredDoorCmd
    BNE Door_TouchWall
    LDA $0C
    AND CurOpenedDoors
    BEQ Door_TouchWall

    ; A = the shutter's bit. Touched before in this room: block while Link's timer <> 0.
    AND ShutterTouchedDirs
    BEQ :+
    BNE Door_Touch_BlockWhileTimer
:
    ; First touch: record the bit and pass.
    LDA ShutterTouchedDirs
    ORA $0C
    STA ShutterTouchedDirs
    RTS

; ----------------------------------------------------------------------------
; Door types 5 and 6 (locked door): passes if open; else spends a key (or uses the magic key) to open.
; IN:   [0C] = the door's direction bit
; NOTE: Order of tests: (1) bit set in CurOpenedDoors: pass. (2) a door command is running
;       (TriggeredDoorCmd <> 0, this door's or any other's): blocked while Link's ObjTimer <> 0,
;       pass when it is 0, with no key spent and the door still closed. (3) InvMagicKey <> 0: open,
;       no key spent. (4) InvKeys = 0: wall. (5) InvKeys - 1, open.
;       Opening = Door_TriggerOpen (command 6) and Link's ObjTimer = $20; Link is blocked on that
;       frame, and after it by case (2) until case (1) applies or the timer reaches 0.
; NOTE: Case (2) with the timer at 0 is a way through a locked door without a key: the pass makes
;       Doorway_CheckLink set DoorwayDir, after which nothing stops Link in the doorway. It needs a door
;       command that did not set Link's timer: a shutter opening (Shutter_OpenTriggered) or a bombed wall
;       opening (bank7/bomb.asm), each about 9 frames long. Door_Update sets the timer to $30 for
;       a close command.
; ----------------------------------------------------------------------------
Door_TouchKey:
    ; Already open: pass.
    LDA $0C
    AND CurOpenedDoors
    BNE Door_Touch_Return

    ; A door command is running: block only while Link's timer <> 0.
    LDA TriggeredDoorCmd
    BNE Door_Touch_BlockWhileTimer

    ; No magic key and no keys: wall.
    LDA InvMagicKey
    BNE @TriggerDoor
    LDA InvKeys
    BEQ Door_Touch_BlockAtWall

    ; No magic key: spend a key.
    DEC InvKeys

@TriggerDoor:
    ; Set the command that opens this door.
    LDA $0C
    JSR Door_TriggerOpen

    ; Link's timer = $20 frames (keeps case (2) above blocking while the door opens).
    LDA #$20
    STA ObjTimer

; Blocks Link: jumps to Door_TouchWall. Shared by Door_TouchKey and Door_Touch_BlockWhileTimer.
Door_Touch_BlockAtWall:
    JMP Door_TouchWall

; Blocks Link while his ObjTimer (slot 0) is not 0; lets him pass when it is 0.
Door_Touch_BlockWhileTimer:
    LDA ObjTimer
    BNE Door_Touch_BlockAtWall

; Shared RTS of Door_TouchKey and Door_Touch_BlockWhileTimer: Link passes.
Door_Touch_Return:
    RTS
