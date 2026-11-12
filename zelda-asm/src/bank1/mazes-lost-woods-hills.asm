; ============================================================================
; FILE:     bank1/mazes-lost-woods-hills.asm BANK: 1 (RAM code: stored at $A500, run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 4832-4935
; SUBJECTS: Filler
; TOPIC:    The Lost Woods (overworld room $61) and Lost Hills (overworld room $1B) direction puzzles,
;           applied on every overworld room change; a pixel-to-nametable-address helper; 10 filler bytes.
; SEE ALSO: bank5/next-room-calc.asm (CalculateNextRoomForDoor sets NextRoomId, then calls CheckMazes when
;           CurLevel is 0); bank7/next-room-room-flags.asm (CalculateNextRoom, CalculateNextRoomOW);
;           bank5/mode7-scroll-init.asm (InitMode7_Sub1 asks for the next room once per room scroll);
;           bank7/tile-obj-tiles.asm (ChangeTileObjTiles calls MapScreenPosToPpuAddr).
; KEY FACTS:
;   - Lost Woods = room $61: leave up, left, down, left (ForestMazeDirs). Lost Hills = room $1B: leave
;     up four times (MountainMazeDirs). CheckMazes
;   - A correct step other than the fourth adds 1 to MazeStep and sets NextRoomId = RoomId: the scroll
;     brings up the same room again. CheckMazes@AdvanceMaze
;   - The fourth correct step plays the "secret found" tune (Tune1Request = 4) and keeps the NextRoomId the
;     caller computed: $60 (left of the Lost Woods) or $0B (above the Lost Hills). MazeStep stays 3.
;     CheckMazes@PlaySecretTune
;   - A wrong step sets MazeStep = 0 and repeats the room. The wrong exit itself is not counted as step 0,
;     even if its direction is the first of the sequence. CheckMazes@Reset
;   - Free exits: right out of the Lost Woods (to $62) and left out of the Lost Hills (to $1A), at any
;     step. They leave MazeStep unchanged. CheckMazes@Mismatch, CheckMazes@CheckMountainMaze
;   - The direction tested is Link's facing direction ObjDir at the room change. CheckMazes
;   - A room change out of any other overworld room sets MazeStep = 0. Dungeon room changes do not call
;     CheckMazes, and no other code writes MazeStep. CheckMazes@ResetMazeStep
; CONTENTS: (generated; L = line in this file)
;   L0036 ForestMazeDirs         data  TABLE: Lost Woods (room $61) sequence; index = MazeStep 0-3; value = direction bit Link must leave by
;   L0041 MountainMazeDirs       data  TABLE: Lost Hills (room $1B) sequence; index = MazeStep 0-3; value = direction bit Link must leave by
;   L0056 CheckMazes             code  Applies the Lost Woods ($61) and Lost Hills ($1B) rules to an overworld room change.
;   L0125 MapScreenPosToPpuAddr  code  Converts a screen pixel position to the nametable 0 address of the tile that covers it.
;   L0149 Filler_6DFA            data  DATA: filler ($FF), 10 bytes after the maze and address code.
; ============================================================================

; TABLE: Lost Woods (room $61) sequence; index = MazeStep 0-3; value = direction bit Link must leave by
; (8 up, 2 left, 4 down, 2 left).
ForestMazeDirs:
    .BYTE $08, $02, $04, $02

; TABLE: Lost Hills (room $1B) sequence; index = MazeStep 0-3; value = direction bit Link must leave by
; (8 up, four times).
MountainMazeDirs:
    .BYTE $08, $08, $08, $08

; ----------------------------------------------------------------------------
; Applies the Lost Woods ($61) and Lost Hills ($1B) rules to an overworld room change.
; IN:   RoomId = room being left; ObjDir = Link's facing direction; NextRoomId = the neighbour in that
;       direction (set by the caller); MazeStep = correct steps so far, 0-3.
; OUT:  NextRoomId = RoomId when the room repeats, else unchanged; MazeStep + 1, kept or 0;
;       Tune1Request = 4 on the fourth correct step. Uses A, X, Y.
; NOTE: Called only by CalculateNextRoomForDoor, and only with CurLevel = 0: once per overworld room
;       scroll. In every room but $61 and $1B it only zeroes MazeStep.
;       Lost Woods: up, left, down, left leaves to room $60; right always leaves to $62.
;       Lost Hills: up four times leaves to room $0B; left always leaves to $1A.
;       Any other exit repeats the room and starts the sequence over.
; ----------------------------------------------------------------------------
CheckMazes:
    LDX MazeStep
    LDA ObjDir                  ; A = Link's facing direction, compared against the tables below.
    LDY RoomId
    CPY #$61                    ; Lost Woods?
    BNE @CheckMountainMaze
    CMP ForestMazeDirs, X
    BNE @Mismatch

    ; Right direction for this step. The fourth one (step 3) lets Link out.
    CPX #$03
    BEQ @PlaySecretTune

@AdvanceMaze:
    ; A correct step before the last: count it, and make the next room this same room.
    INC MazeStep

@SetNextRoom:
    STY NextRoomId

@Exit:
    RTS

; Zeroes MazeStep. Reached by branch from any non-maze room (then NextRoomId is left alone) and by
; JSR from @Reset. Returns A = 0, Z set.
@ResetMazeStep:
    LDA #$00
    STA MazeStep
    RTS

@Mismatch:
    ; Wrong direction in the Lost Woods. Right (1) is the free exit: return with NextRoomId ($62) and
    ; MazeStep untouched.
    CMP #$01
    BEQ @Exit

@Reset:
    ; Any other wrong direction: MazeStep := 0 and repeat the room.
    JSR @ResetMazeStep
    BEQ @SetNextRoom            ; Always taken (A = 0).

@CheckMountainMaze:
    CPY #$1B                    ; Lost Hills?
    BNE @ResetMazeStep          ; Any other room: MazeStep := 0; NextRoomId stays as computed.
    CMP MountainMazeDirs, X
    BEQ @Match

    ; Wrong direction in the Lost Hills. Left (2) is the free exit: return with NextRoomId ($1A) and
    ; MazeStep untouched. Anything else starts over and repeats the room.
    CMP #$02
    BNE @Reset
    RTS

@Match:
    CPX #$03
    BNE @AdvanceMaze            ; Steps 0-2: count the step and repeat the room.

@PlaySecretTune:
    ; Fourth correct step: request the "secret found" tune and return with NextRoomId as the caller
    ; computed it ($60 from the Lost Woods, $0B from the Lost Hills). MazeStep is left at 3.
    LDA #$04
    STA Tune1Request
    RTS

; ----------------------------------------------------------------------------
; Converts a screen pixel position to the nametable 0 address of the tile that covers it.
; IN:   [02] = Y in pixels (screen coordinate, status bar rows included); [03] = X in pixels.
; OUT:  [00] = high byte, [01] = low byte of $2000 + (Y / 8) * 32 + (X / 8); A = [01].
; ----------------------------------------------------------------------------
MapScreenPosToPpuAddr:
    ; A row of tiles is 32 bytes and 8 pixels tall, so the row offset is (Y AND $F8) * 4.
    ; Shift Y left twice through [00], which starts at $08: the two bits shifted out of Y land in the
    ; low bits of [00] and the 8 becomes $20, giving the high byte $20-$23.
    LDA #$08
    STA $00
    LDA $02
    ASL
    ROL $00
    ASL
    ROL $00
    AND #$E0                    ; Drop the bits that came from Y's low three bits (pixels within a tile).
    STA $01

    ; Column = X / 8 (0-31), in the low five bits of the low byte.
    LDA $03
    LSR
    LSR
    LSR
    ORA $01
    STA $01
    RTS

; DATA: filler ($FF), 10 bytes after the maze and address code.
Filler_6DFA:
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF
