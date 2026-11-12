; ============================================================================
; FILE:     bank5/spawn-positions.asm    BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 1870-2037
; SUBJECTS: (none)
; TOPIC:    Monster spawn spots, cellar keese, cave person, spawn safety
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0017 CellarKeeseXs            data
;   L0020 CellarKeeseYs            data
;   L0026 AssignObjSpawnPositions  code  Params:
;   L0146 IsSafeToSpawn            code  Params:
;   L0162 IsDistanceSafeToSpawn    code  Params:
;   L0182 ReturnUnsafeToSpawn      code
; ============================================================================

CellarKeeseXs:
    .BYTE $20, $60, $90, $D0

CellarKeeseYs:
    .BYTE $9D, $5D, $7D, $9D

; Params:
; [02]: object template ID
;
AssignObjSpawnPositions:
    LDY RoomObjCount

    ; If object template type = 0 or refers to Zelda, then
    ; skip spawning normal objects.
    LDA $02
    BEQ @AssignSpecialPositions
    CMP #$37
    BEQ @AssignSpecialPositions

    ; If in OW, then see if monsters come in from the edges of the screen.
    LDA CurLevel
    BNE :+
    LDA RoomAttrsF
    AND #$08                    ; Monsters from the edges of the screen
    BNE @AssignSpecialPositions ; If monsters come in from the edges, go skip spawning normal objects.
:
    LDA RoomObjCount
    BEQ @AssignSpecialPositions ; If object count = 0, go skip spawning normal objects.

    ; Get the direction from Link's direction (bit).
    LDA ObjDir
    LDY #$FF
:
    INY
    LSR
    BCC :-

    ; Put the address of the spawn list for Link's direction in [06:07].
    LDA SpawnPosListAddrsLo, Y
    STA $06
    LDA SpawnPosListAddrsHi, Y
    STA $07

    ; Assign spawn coordinates to 9 object slots.
    LDY SpawnCycle
    LDX #$01                    ; Starting at object 1.

@LoopSpawnSpot:
    LDA ($06), Y                ; Get a spawn spot from the list.
    PHA
    ASL                         ; Turn the column component into an X coordinate.
    ASL
    ASL
    ASL
    STA ObjX, X
    PLA                         ; Turn the row component into a Y coordinate.
    AND #$F0
    ORA #$0D                    ; Add $D to the Y coordinate.
    STA ObjY, X
    JSR IsSafeToSpawn
    BCS :+
    INX                         ; Point to the next object.
:
    INY                         ; Point to the next spawn spot.
    CPY #$09
    BCC :+                      ; If we went past the last spawn spot,
    LDY #$00                    ; then reset the cycle.
:
    CPX #$0A
    BCC @LoopSpawnSpot
    STY SpawnCycle

@AssignSpecialPositions:
    ; If in mode 9 (play cellar), then spawn 4 blue keeses.
    LDA GameMode
    CMP #$09
    BNE @CheckCaves             ; If not in mode 9, go check caves.
    LDX #$03
:
    LDA #$1B                    ; Blue Keese
    STA ObjType+1, X
    LDA CellarKeeseXs, X
    STA ObjX+1, X
    LDA CellarKeeseYs, Y
    STA ObjY+1, X
    DEX
    BPL :-
    RTS

@CheckCaves:
    ; If the mode is not $B nor $C (caves), then return.
    CMP #$0B
    BEQ @InCave
    CMP #$0C
    BNE @Exit

@InCave:
    ; We're in a cave.
    ; Reset object types in slots 1 to 8.
    LDX #$07
    LDA #$00
:
    STA ObjType+1, X
    DEX
    BPL :-

    ; cave index = ((cave value) >> 2) - $10
    LDY RoomId
    LDA LevelBlockAttrsB, Y
    AND #$FC                    ; Cave
    SEC
    SBC #$40
    LSR
    LSR

    ; cave dweller object type = $6A + (cave index)
    CLC
    ADC #$6A
    STA ObjType+1               ; Replace first non-Link object type with the cave dweller type.

@Exit:
    RTS

; Params:
; X: object index
;
; Returns:
; C: 1 if unwalkable
;
IsSafeToSpawn:
    TYA                         ; Save %Y.
    PHA
    JSR GetCollidableTileStill
    PLA                         ; Restore %Y.
    TAY
    LDA ObjCollidedTile, X
    CMP RoomFirstUnwalkableTile
    BCS ReturnUnsafeToSpawn     ; If the tile at the object's hotspot is unwalkable, return unwalkable.

; Params:
; X: object index
;
; Returns:
; C: 1 if unwalkable
;
IsDistanceSafeToSpawn:
    ; Get the absolute X distance to Link.
    LDA ObjX
    SEC
    SBC ObjX, X
    JSR Abs
    CMP #$22
    BCS :+                      ; If distance > $22, return walkable.

    ; Get the absolute Y distance to Link.
    LDA ObjY
    SEC
    SBC ObjY, X
    JSR Abs
    CMP #$22
    BCC ReturnUnsafeToSpawn     ; If distance < $22, go return unwalkable.
:
    CLC                         ; Walkable
    RTS

ReturnUnsafeToSpawn:
    SEC                         ; Unwalkable
    RTS
