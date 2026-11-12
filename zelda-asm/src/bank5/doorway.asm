; ============================================================================
; FILE:     bank5/doorway.asm            BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 3656-3907
; SUBJECTS: Doorway, Link, Unused
; TOPIC:    Dungeon doorways: detecting that Link stands at or in one of the four door frames, touching
;           the door there, locking his movement to the doorway's axis, and leaving the room through
;           false and bombed walls.
; SEE ALSO: bank5/touch-door.asm (Door_Touch: what each door type does when touched);
;           bank5/fill-walls.asm (Door_FindTypeByDirBit: door type of a side from room attrs A and B);
;           bank7/walker-move.asm (Walker_Move calls Doorway_CheckLink for Link in dungeons, not in mode 9);
;           bank7/walker-collision-screen-edge.asm (Walker_CheckTileCollision reads DoorwayDir and
;           [0E]; Mode5Play_GoToNextMode); bank5/link-handle-input.asm (calls Link_ModifyDirInDoorway);
;           bank5/enter-room.asm (DoorwayDir is set when Link walks into a room).
; KEY FACTS:
;   - Door frames are at X = $78 (up and down doors) and Y = $8D (left and right doors); Link's other
;     coordinate must match exactly to be at a doorway. Doorway_RequiredCoords, Doorway_CheckLink
;   - From the room, a door is touched when Link is in the wide span of its doorway (Y <= $5D up,
;     Y >= $BD down, X <= $20 left, X >= $CF right) and the input direction is the door's
;     direction. Doorway_BoundsMinOver, Doorway_BoundsMaxOver, Doorway_CheckLink@TestDoorwayDoor
;   - Once DoorwayDir <> 0, the wide spans are used only while Link faces DoorwayDir inside that
;     doorway; otherwise the narrow spans apply (Y <= $5B up, Y >= $BF down, X <= $1E left,
;     X >= $D2 right), so DoorwayDir is cleared 2-3 pixels before he is back on the floor spot.
;     Doorway_BoundsMinUnder, Doorway_BoundsMaxUnder, Doorway_CheckLink@SearchUnderflowDoorway
;   - In a doorway the input direction is forced onto the axis Link faces: forward if held, else
;     backward if held, else (only perpendicular input) forward. Link_ModifyDirInDoorway
;   - Passing a false wall (door types 2, 3) or a bombed wall (type 4) starts the room change at once
;     (Mode5Play_GoToNextMode); other doorways are walked through to the screen edge. Doorway_CheckLink
;   - A blocking door sets [0E] = $FF and leaves DoorwayDir unchanged. The block stops Link only
;     while DoorwayDir = 0: with DoorwayDir <> 0 Walker_CheckTileCollision skips both the [0E]
;     test and the tile test, and Room_BoundDir is not called. Doorway_CheckLink, Door_TouchWall
; CONTENTS: (generated; L = line in this file)
;   L0052 Link_ModifyDirInDoorway           code  Restricts Link's input direction to the axis he faces while he is in a dungeon doorway.
;   L0082 Unused_HideSpritesOverLink_Bank5  code  Unused copy of bank 4's Sprite_HideOverLink: moves sprites $10 and $11 off screen (Y = $F8).
;   L0093 Doorway_RequiredCoords            data  TABLE: coordinate Link must have exactly on the axis across a doorway; index = reverse direction
;   L0100 Doorway_BoundsMinOver             data  TABLE: wide doorway span, lower bound (inclusive); index = reverse direction index; value = Link's
;   L0104 Doorway_BoundsMaxOver             data  TABLE: wide doorway span, upper bound (exclusive); index = reverse direction index; value = pixels.
;   L0111 Doorway_BoundsMinUnder            data  TABLE: narrow doorway span, lower bound (inclusive); index = reverse direction index; value = Link's
;   L0115 Doorway_BoundsMaxUnder            data  TABLE: narrow doorway span, upper bound (exclusive); index = reverse direction index; value = pixels.
;   L0137 Doorway_CheckLink                 code  Finds the dungeon doorway Link is at or in, touches its door, and tracks DoorwayDir.
;   L0260 Doorway_GetLinkCoordsForDir       code  Splits Link's position into the coordinate across a direction's axis and the one along it.
; ============================================================================

; ----------------------------------------------------------------------------
; Restricts Link's input direction to the axis he faces while he is in a dungeon doorway.
; IN:   DoorwayDir; ObjInputDir and ObjDir of Link (slot 0)
; OUT:  ObjInputDir = the facing direction if it is held; else the opposite direction if that is
;       held; else the facing direction. Unchanged when DoorwayDir = 0 or no direction is held.
; NOTE: Holding only a perpendicular direction in a doorway walks Link forward. The axis comes from
;       ObjDir, not from DoorwayDir. Called by Link_HandleInput in dungeons when Link is not shoved.
;       Clobbers A and Y.
; ----------------------------------------------------------------------------
Link_ModifyDirInDoorway:
    ; Not in a doorway, or no direction held: return.
    LDA DoorwayDir
    BEQ @Exit
    LDY ObjInputDir
    BEQ @Exit

    ; The facing direction is held: keep only it.
    LDA ObjDir
    AND ObjInputDir
    BNE :+

    ; The opposite of the facing direction is held: keep only it.
    LDA ObjDir
    JSR Dir_GetOpposite
    AND ObjInputDir

    ; Neither is held: replace the input with the facing direction.
    BNE :+
    LDA ObjDir
:
    STA ObjInputDir

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Unused copy of bank 4's Sprite_HideOverLink: moves sprites $10 and $11 off screen (Y = $F8).
; NOTE: Nothing in the program refers to this label.
; ----------------------------------------------------------------------------
Unused_HideSpritesOverLink_Bank5:
    LDA #$F8
    STA Sprites+64
    STA Sprites+68
    RTS

; The five tables below are indexed by reverse direction index: 0 = up door, 1 = down door,
; 2 = left door, 3 = right door (the order of Dir_ReverseIndexToDir: $08, $04, $02, $01).

; TABLE: coordinate Link must have exactly on the axis across a doorway; index = reverse direction
; index; value = X ($78) for the up and down doors, Y ($8D) for the left and right doors (pixels).
Doorway_RequiredCoords:
    .BYTE $78, $78, $8D, $8D

; TABLE: wide doorway span, lower bound (inclusive); index = reverse direction index; value = Link's
; Y (up, down) or X (left, right) in pixels. With Doorway_BoundsMaxOver the spans are: up Y $3D-$5D,
; down Y $BD-$DD, left X $00-$20, right X $CF-$F0. They include the floor spot in front of the door.
; Used to enter a doorway and while Link faces further into it.
Doorway_BoundsMinOver:
    .BYTE $3D, $BD, $00, $CF

; TABLE: wide doorway span, upper bound (exclusive); index = reverse direction index; value = pixels.
Doorway_BoundsMaxOver:
    .BYTE $5E, $DE, $21, $F1

; TABLE: narrow doorway span, lower bound (inclusive); index = reverse direction index; value = Link's
; Y (up, down) or X (left, right) in pixels. With Doorway_BoundsMaxUnder the spans are: up Y $3D-$5B,
; down Y $BF-$DD, left X $00-$1E, right X $D2-$F0: the wide span less its 2 (up, down, left) or 3
; (right) pixels nearest the room. Used once Link is in a doorway and not facing further into it.
Doorway_BoundsMinUnder:
    .BYTE $3D, $BF, $00, $D2

; TABLE: narrow doorway span, upper bound (exclusive); index = reverse direction index; value = pixels.
Doorway_BoundsMaxUnder:
    .BYTE $5C, $DE, $1F, $F1

; ----------------------------------------------------------------------------
; Finds the dungeon doorway Link is at or in, touches its door, and tracks DoorwayDir.
; IN:   [0F] = moving direction (possibly zeroed by Room_BoundDir); ObjX, ObjY, ObjDir, ObjInputDir of
;       Link; DoorwayDir
; OUT:  DoorwayDir = 0 when Link is in no doorway span; = the doorway's direction bit when he passed
;       its door this frame; otherwise unchanged.
;       [0E] = reverse direction index of the doorway found, or $FF when its door blocked Link.
;       [0F] = the doorway's direction when the door let Link pass; otherwise unchanged.
;       ObjDir = the doorway's direction when the door let Link pass.
;       Clobbers A, X, Y, [00]-[03], [0C], [0D].
; NOTE: A door stops Link only by [0E] = $FF while DoorwayDir = 0. Once DoorwayDir <> 0 (set here
;       when a door let him pass, or on entering the room) the caller neither bounds him by the room
;       nor tests tiles or [0E], so a door that blocks on a later frame no longer stops him.
; NOTE: The door is touched only when the input direction (low 4 bits of ObjInputDir) equals the
;       doorway's direction; with any other input a doorway that was found changes nothing.
;       Passing a door of type 2, 3 (false walls) or 4 (bombable, already open) jumps to
;       Mode5Play_GoToNextMode: the room change starts on that frame.
;       Called by Walker_Move for Link only, in dungeons, in every mode but 9 (cellars).
; ----------------------------------------------------------------------------
Doorway_CheckLink:
    ; Already in a doorway: handled further down.
    LDA DoorwayDir
    BNE @InDoorway

@SearchWideSpans:
    ; Look for a doorway whose wide span holds Link:
    ;   1. [00] (the coordinate across the doorway) equals Doorway_RequiredCoords;
    ;   2. Doorway_BoundsMinOver <= [01] (the coordinate along it) < Doorway_BoundsMaxOver.
    ; Which of X and Y is [00] depends on Link's facing direction, for all four doorways tested.
    LDA ObjDir
    JSR Doorway_GetLinkCoordsForDir
    LDY #$03                    ; Reverse direction index 3 (right) down to 0 (up).

@LoopWideSpans:
    LDA $00                     ; [00] is Link's Y if he faces left or right, else his X.
    CMP Doorway_RequiredCoords, Y
    BNE :+
    LDA $01                     ; [01] is Link's X if he faces left or right, else his Y.
    CMP Doorway_BoundsMinOver, Y
    BCC :+
    CMP Doorway_BoundsMaxOver, Y
    BCC @TestDoorwayDoor        ; In the span: go touch the door.
:
    DEY
    BPL @LoopWideSpans

@NotInDoorway:
    ; In no doorway: clear DoorwayDir and leave [0F] as it is.
    LDA #$00
    STA DoorwayDir
    RTS

@InDoorway:
    ; A = DoorwayDir. The coordinates are split by the doorway's axis this time.
    PHA
    JSR Doorway_GetLinkCoordsForDir
    PLA
    JSR Dir_GetOpposite          ; Only for Y = reverse direction index of DoorwayDir.
    LDA $01                     ; [01] is Link's X in a left or right doorway, else his Y.

    ; If Link is in the wide span of the doorway named by DoorwayDir and faces DoorwayDir, repeat
    ; the wide search that is used to enter a doorway.
    CMP Doorway_BoundsMinOver, Y
    BCC @SearchNarrowSpans
    CMP Doorway_BoundsMaxOver, Y
    BCS @SearchNarrowSpans
    LDA DoorwayDir
    CMP ObjDir
    BEQ @SearchWideSpans

@SearchNarrowSpans:
    ; Link faces another way, or is outside that doorway (he may be in the doorway on the other
    ; side of the room). Search all four again with the narrow spans, which end 2-3 pixels short of
    ; the room: stepping back out of a doorway clears DoorwayDir that much earlier.
    LDY #$03

@LoopNarrowSpans:
    LDA $00                     ; [00]: Link's Y in a left or right doorway (DoorwayDir), else his X.
    CMP Doorway_RequiredCoords, Y
    BNE :+
    LDA $01                     ; [01]: Link's X in a left or right doorway, else his Y.
    CMP Doorway_BoundsMinUnder, Y
    BCC :+
    CMP Doorway_BoundsMaxUnder, Y
    BCC @TestDoorwayDoor
:
    DEY
    BPL @LoopNarrowSpans
    BMI @NotInDoorway           ; In no doorway: clear DoorwayDir and leave [0F] alone.

@TestDoorwayDoor:
    ; [0E] = reverse direction index of the doorway found.
    STY $0E

    ; [02] and [0C] = input direction. [02] is the argument of Door_FindTypeByDirBit; [0C] is the
    ; door direction for Door_Touch.
    LDA ObjInputDir
    AND #$0F
    STA $02
    STA $0C

    ; The input direction is not the doorway's direction: return, nothing changes.
    CMP Dir_ReverseIndexToDir, Y
    BNE @Exit

    ; Touch the door on that side of the room.
    JSR Door_FindTypeByDirBit
    STA $0D                     ; [0D] = door type (0-7; see Door_Touch_JumpTable).
    JSR Door_Touch               ; Sets [0E] to $FF if the door blocks the way.

    ; Blocked: return and leave DoorwayDir as it was.
    LDY $0E
    BMI @Exit

    ; Not blocked: face and move in the doorway's direction, and record being in the doorway.
    LDA Dir_ReverseIndexToDir, Y
    STA ObjDir
    STA $0F                     ; [0F] = moving direction.
    STA DoorwayDir

    ; Door types 2 and 3 (false walls) and 4 (bombable wall, open) have no frame to walk through:
    ; leave the room now.
    LDA $0D
    AND #$07
    CMP #$02
    BEQ :+
    CMP #$03
    BEQ :+
    CMP #$04
    BEQ :+

@Exit:
    RTS
:
    JMP Mode5Play_GoToNextMode

; ----------------------------------------------------------------------------
; Splits Link's position into the coordinate across a direction's axis and the one along it.
; IN:   A = direction (1 right, 2 left, 4 down, 8 up)
; OUT:  [00] = ObjY and [01] = ObjX for left or right; [00] = ObjX and [01] = ObjY for up or down
;       (and for A = 0). Clobbers X and Y.
; ----------------------------------------------------------------------------
Doorway_GetLinkCoordsForDir:
    LDX ObjX
    LDY ObjY
    AND #$03                    ; Left or right?
    BEQ :+
    LDY ObjX                    ; Yes: swap.
    LDX ObjY
:
    STX $00
    STY $01
    RTS
