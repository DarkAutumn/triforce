; ============================================================================
; FILE:     bank5/door-flags-dark-room.asm BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 7752-7842
; SUBJECTS: (none)
; TOPIC:    Dungeon door helpers: the mask of sides of a room that can be walked through (for the submenu
;           map), copying a room's saved door flags into CurOpenedDoors, and the dark room test.
; SEE ALSO: bank5/submenu-map.asm (Submenu_WriteScanningMapRoomMark, the only reader of OpenDoorwayMask);
;           bank5/lay-out-doors.asm (LayOutDoors: door types to door faces, also calls
;           CalcOpenDoorwayMask); bank5/fill-walls.asm (FindDoorTypeByDoorBit); bank5/mode12-lay-out-room.asm
;           (dungeon room layout calls AddDoorFlagsToCurOpenedDoors); bank5/shutters-door-flags.asm
;           (SetDoorFlag); bank7/next-room-room-flags.asm (GetRoomFlags, LevelMasks);
;           bank4/shoot-helpers-candle.asm (IsDarkRoom_Bank4, the same test in bank 4).
; KEY FACTS:
;   - A side counts as passable when its door type is 0 (open). Types 1-3 (wall, false wall, false wall
;     2) never do. Types 4-7 (bombable, key, key 2, shutter) do only when the room's saved flag bit for
;     that direction is set. CalcOpenDoorwayMask
;   - Saved room flags of a dungeon room: bits 0-3 are the opened-door bits, in direction-bit order
;     (1 right, 2 left, 4 down, 8 up). CalcOpenDoorwayMask, AddDoorFlagsToCurOpenedDoors
;   - Laying out a dungeon room first ORs the saved door bits into CurOpenedDoors, so doors opened on an
;     earlier visit are open again (LayOutRoom is the only caller). AddDoorFlagsToCurOpenedDoors
;   - A room is dark when CurLevel is not 0 and bit 7 of LevelBlockAttrsE for the room is set. The
;     overworld (CurLevel 0) is never dark. IsDarkRoom_Bank5
;   - SplitRoomId has no caller. SplitRoomId
; CONTENTS: (generated; L = line in this file)
;   L0042 CalcOpenDoorwayMask           code  Shifts one bit into OpenDoorwayMask: 1 if the room's side with this door type can be walked through.
;   L0084 AddDoorFlagsToCurOpenedDoors  code  ORs the current room's saved opened-door flags (room flags bits 0-3) into CurOpenedDoors.
;   L0104 SplitRoomId                   code  Unused: splits RoomId into its map column (Y, low nibble) and map row (X, high nibble).
;   L0124 IsDarkRoom_Bank5              code  Tests whether a room is dark (bank 5 copy; IsDarkRoom_Bank4 is identical).
; ============================================================================

; ----------------------------------------------------------------------------
; Shifts one bit into OpenDoorwayMask: 1 if the room's side with this door type can be walked through.
; IN:   A = door type of the side (0 open, 1 wall, 2 false wall, 3 false wall 2, 4 bombable, 5 key,
;       6 key 2, 7 shutter); X = direction index (0 right, 1 left, 2 down, 3 up); RoomId = the room.
; OUT:  OpenDoorwayMask = ((OpenDoorwayMask << 1) | bit) & $0F; A and X unchanged; Y = 0. For door
;       types 4-7 also [00:01] = address of the level's room flags (GetRoomFlags).
; NOTE: Called for X = 3 down to 0, the four bits end up as direction bits (8 up, 4 down, 2 left,
;       1 right). Type 0 gives 1; types 1-3 give 0, so a false wall never counts as passable; types
;       4-7 give the room's saved door flag for that direction (GetRoomFlags bit LevelMasks,X).
;       The caller clears or presets OpenDoorwayMask; this routine only shifts.
; ----------------------------------------------------------------------------
CalcOpenDoorwayMask:
    LDY #$00
    PHA                         ; Save door type.
    CMP #$04
    BCC @ByDoorType             ; Door type < 4 (open or any wall): the type alone decides.

    ; Door type 4-7: passable only if the room's saved door flag for this direction is set.
    TXA
    PHA                         ; Save direction index.
    TYA
    PHA                         ; Save Y=0
    JSR GetRoomFlags
    CLC                         ; Carry clear = not passable, unless the flag is set.
    AND LevelMasks, X           ; Single-bit mask of the direction: $01, $02, $04, $08.
    BEQ :+                      ; Flag clear: the door has not been opened.
    SEC                         ; Flag set: shift a 1 into the mask.
:
    PLA
    TAY                         ; Restore Y=0.
    PLA
    TAX                         ; Restore direction index in X.

@ShiftIntoMask:
    LDA OpenDoorwayMask, Y      ; Y = 0
    ROL                         ; Carry becomes bit 0.
    AND #$0F
    STA OpenDoorwayMask, Y
    PLA                         ; Restore door type.
    RTS

@ByDoorType:
    CMP #$00                    ; CMP #$00 always sets carry.
    BEQ @ShiftIntoMask          ; Door type 0 (open): shift a 1 into the mask.
    CLC
    BCC @ShiftIntoMask          ; Door type 1-3 (a wall): shift a 0 into the mask. Always taken.

; ----------------------------------------------------------------------------
; ORs the current room's saved opened-door flags (room flags bits 0-3) into CurOpenedDoors.
; OUT:  CurOpenedDoors; [00:01] = address of the level's room flags; Y = RoomId; X = $FF.
; NOTE: Called only by LayOutRoom's dungeon path, before LayOutDoors, which then clears the bits of
;       sides that are not true doors (door types 0-3).
; ----------------------------------------------------------------------------
AddDoorFlagsToCurOpenedDoors:
    JSR GetRoomFlags            ; [00:01] = room flags base, Y = RoomId
    LDX #$03                    ; Direction index 3 (up) down to 0 (right).

@LoopDoorBit:
    LDA ($00), Y
    AND LevelMasks, X           ; $08, $04, $02, $01
    BEQ :+                      ; This door was never opened.
    ORA CurOpenedDoors
    STA CurOpenedDoors
:
    DEX
    BPL @LoopDoorBit
    RTS

; ----------------------------------------------------------------------------
; Unused: splits RoomId into its map column (Y, low nibble) and map row (X, high nibble).
; OUT:  Y = RoomId & $0F; X = RoomId >> 4
; NOTE: No caller anywhere in the program.
; ----------------------------------------------------------------------------
SplitRoomId:
    LDA RoomId
    PHA
    AND #$0F
    TAY
    PLA
    LSR
    LSR
    LSR
    LSR
    TAX
    RTS

; ----------------------------------------------------------------------------
; Tests whether a room is dark (bank 5 copy; IsDarkRoom_Bank4 is identical).
; IN:   Y = room id
; OUT:  A = $80 and Z clear if dark; A = 0 and Z set if not (always in the overworld, CurLevel = 0).
; NOTE: The dark flag is bit 7 of LevelBlockAttrsE. This tests the room's attribute only, not whether
;       the room has since been lit.
; ----------------------------------------------------------------------------
IsDarkRoom_Bank5:
    LDA CurLevel
    BEQ :+                      ; Overworld: never dark.
    LDA LevelBlockAttrsE, Y
    AND #$80
:
    RTS
