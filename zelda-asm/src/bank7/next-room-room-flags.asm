; ============================================================================
; FILE:     bank7/next-room-room-flags.asm BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 737-803
; SUBJECTS: Room
; TOPIC:    The compass (Triforce) marker on the status bar map; picking the room Link scrolls into from
;           his direction and the door type on that side; the per-room flag byte (read, mark visited); the
;           single-bit mask table.
; SEE ALSO: bank5/next-room-calc.asm (CalculateNextRoomForDoor, CalculateNoNextRoom: the table targets);
;           bank5/fill-walls.asm (FindDoorTypeByDoorBit); bank5/mode7-scroll-init.asm (the caller of
;           CalculateNextRoom); bank1/map-marker-curtain.asm (UpdatePositionMarker);
;           bank1/mazes-lost-woods-hills.asm (CheckMazes changes the OW result);
;           bank5/submenu-map.asm (HasCompass); bank4/rock-wall-tree-secrets.asm and
;           bank5/cave-cellar-exits.asm (OW uses of MarkRoomVisited); bank5/ow-shortcut-map-square.asm
;           (CheckShortcut reads the bit).
; KEY FACTS:
;   - The next room is chosen from Link's facing direction ObjDir, not from his position or input
;     (CalculateNextRoom, CalculateNextRoomOW).
;   - In the OW every side counts as an open door (type 0); in a dungeon the door type of the side Link
;     faces decides (CalculateNextRoom, CalculateNextRoom_JumpTable).
;   - Door types 1 (wall) and 8 (no side matched) go to CalculateNoNextRoom, which leaves NextRoomId
;     unchanged and clears [E7]; types 0 and 2-7 (open, false walls, bombable, key doors, shutter)
;     compute the neighbour (CalculateNextRoom_JumpTable).
;   - Room flags are one byte per room at (LevelInfo_WorldFlagsAddr) + RoomId (GetRoomFlags).
;   - MarkRoomVisited sets flag bit 5 ($20). Dungeon: set on every room entry and for the room a cellar
;     leads to; it is the "visited" bit of the map. OW: set for the room where a pushed rock or
;     gravestone finishes its move and for the room Link arrives in by shortcut stairs; it is the bit
;     CheckShortcut tests before putting stairs into the room (MarkRoomVisited).
;   - The Triforce marker is drawn only in a dungeon whose compass Link holds (level 9's compass is a
;     separate inventory byte), at LevelInfo_TriforceRoomId (UpdateTriforcePositionMarker).
;   - LevelMasks is the general bit table 1 << n: level number - 1 for compass, map and Triforce
;     pieces, direction index for door flags, neck index for Gleeok's dead necks (LevelMasks).
; CONTENTS: (generated; L = line in this file)
;   L0051 UpdateTriforcePositionMarker  code  In a dungeon with its compass, draws the Triforce room marker on the status bar map.
;   L0072 CalculateNextRoom             code  Sets NextRoomId to the room on the side Link faces, if the door type there lets him through.
;   L0085 CalculateNextRoom_TableJump   code  Second entry: A = door type (table index), Y = direction bit.
;   L0090 CalculateNextRoom_JumpTable   data  TABLE: handler per door type for CalculateNextRoom; index = door type; value = routine address.
;   L0102 CalculateNextRoomOW           code  OW path of CalculateNextRoom: Link's direction with door type 0 (open).
;   L0110 LevelMasks                    data  TABLE: single-bit masks; index = bit number 0-7; value = 1 << index.
;   L0120 MarkRoomVisited               code  Sets bit 5 ($20) of the current room's flag byte.
;   L0126 Room_MarkVisited_Return       code  Shared RTS of MarkRoomVisited and UpdateTriforcePositionMarker.
;   L0134 GetRoomFlags                  code  Reads the current room's flag byte.
; ============================================================================

; ----------------------------------------------------------------------------
; In a dungeon with its compass, draws the Triforce room marker on the status bar map.
; OUT:  sprite record at Sprites+$58 written (UpdatePositionMarker with X = 4). Bank 5 is left switched in
;       when CurLevel is not 0.
; NOTE: Does nothing in the OW or without the level's compass. Called every frame of the play update in
;       a dungeon, and by the submenu scroll, the room-entry init and the death init.
; ----------------------------------------------------------------------------
UpdateTriforcePositionMarker:
    LDA CurLevel
    BEQ Room_MarkVisited_Return ; In the OW: return.
    LDA #$05
    JSR SwitchBank
    JSR HasCompass              ; A = 0 when Link does not have this level's compass.
    BEQ Room_MarkVisited_Return
    LDA LevelInfo_TriforceRoomId
    LDX #$04                    ; Sprite offset 4 = the Triforce marker (0 is Link's position marker).
    JMP UpdatePositionMarker

; ----------------------------------------------------------------------------
; Sets NextRoomId to the room on the side Link faces, if the door type there lets him through.
; IN:   ObjDir (slot 0) = Link's direction bit; CurLevel; RoomId.
; OUT:  NextRoomId; [E7] = the direction bit used (0 after CalculateNoNextRoom or when the OW is
;       reloaded). Bank 5 is switched in (dungeon path).
; NOTE: Dungeon: looks up the door type of that side (FindDoorTypeByDoorBit) and dispatches on it. OW:
;       door type 0 (open) for every side. If the result has bit 7 set (off the top or bottom of the
;       map, or left of room $00 / right of room $7F), CalculateNextRoomForDoor reloads the OW instead.
; NOTE: Called once per room scroll, by the mode 7 init (InitMode7_Sub1).
; ----------------------------------------------------------------------------
CalculateNextRoom:
    LDY CurLevel
    BEQ CalculateNextRoomOW     ; In the OW all sides are open.

    ; Dungeon: look up the door type on the side Link faces.
    LDA ObjDir
    STA $02
    LDA #$05
    JSR SwitchBank
    JSR FindDoorTypeByDoorBit   ; A = door type 0-7, or 8 if no side matched; [01] = the matching bit.
    LDY $01                     ; Equals [02] (Link's direction) whenever a side matched.

; Second entry: A = door type (table index), Y = direction bit.
CalculateNextRoom_TableJump:
    STY $E7                     ; [E7] = direction bit; read by CalculateNextRoomForDoor.
    JSR TableJump

; TABLE: handler per door type for CalculateNextRoom; index = door type; value = routine address.
CalculateNextRoom_JumpTable:
    .ADDR CalculateNextRoomForDoor                      ; 0 open
    .ADDR CalculateNoNextRoom                           ; 1 wall
    .ADDR CalculateNextRoomForDoor                      ; 2 false wall
    .ADDR CalculateNextRoomForDoor                      ; 3 false wall
    .ADDR CalculateNextRoomForDoor                      ; 4 bombable wall
    .ADDR CalculateNextRoomForDoor                      ; 5 key door
    .ADDR CalculateNextRoomForDoor                      ; 6 key door
    .ADDR CalculateNextRoomForDoor                      ; 7 shutter
    .ADDR CalculateNoNextRoom                           ; 8 no side matched the direction

; OW path of CalculateNextRoom: Link's direction with door type 0 (open).
CalculateNextRoomOW:
    LDY ObjDir
    LDA #$00
    BEQ CalculateNextRoom_TableJump    ; Always taken.

; TABLE: single-bit masks; index = bit number 0-7; value = 1 << index.
; Used with (level number - 1) for compasses, maps and Triforce pieces, with a direction index
; for the door bits of the room flags, and with the neck index for Gleeok's dead neck mask.
LevelMasks:
    .BYTE $01, $02, $04, $08, $10, $20, $40, $80

; ----------------------------------------------------------------------------
; Sets bit 5 ($20) of the current room's flag byte.
; OUT:  A = new flag byte; Y = RoomId; [00:01] = address of the level's room flags.
; NOTE: Dungeon: "visited", shown on the map; set on entering a room and for the room a cellar leads
;       to. OW: set (with RoomId temporarily the rock's room) when a pushed rock or gravestone has moved
;       its full $10 pixels, and for the room shortcut stairs lead to; CheckShortcut tests the bit.
; ----------------------------------------------------------------------------
MarkRoomVisited:
    JSR GetRoomFlags
    ORA #$20
    STA ($00), Y

; Shared RTS of MarkRoomVisited and UpdateTriforcePositionMarker.
Room_MarkVisited_Return:
    RTS

; ----------------------------------------------------------------------------
; Reads the current room's flag byte.
; OUT:  A = flag byte of RoomId; Y = RoomId; [00:01] = address of the current level's 128 room flags
;       (LevelInfo_WorldFlagsAddr), so the caller can store with STA ($00),Y.
; ----------------------------------------------------------------------------
GetRoomFlags:
    LDA LevelInfo_WorldFlagsAddr
    STA $00
    LDA LevelInfo_WorldFlagsAddr+1
    STA $01
    LDY RoomId
    LDA ($00), Y
    RTS
