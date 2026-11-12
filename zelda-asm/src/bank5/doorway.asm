; ============================================================================
; FILE:     bank5/doorway.asm            BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 3656-3907
; SUBJECTS: (none)
; TOPIC:    Doorway detection and movement restriction
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0022 Link_ModifyDirInDoorway           code
;   L0056 Unused_HideSpritesOverLink_Bank5  code  Description:
;   L0070 DoorwayRequiredCoord              data  To be considered within a doorway, one condition is that
;   L0077 DoorwayBoundsMinOver              data  Player coordinate must be >= these bounds.
;   L0084 DoorwayBoundsMaxOver              data  Player coordinate must be < these bounds.
;   L0092 DoorwayBoundsMinUnder             data  Player coordinate must be >= these bounds.
;   L0100 DoorwayBoundsMaxUnder             data  Player coordinate must be < these bounds.
;   L0119 CheckDoorway                      code  Params:
;   L0261 GetPlayerCoordsForDirection       code  Params:
; ============================================================================

    ; End unverified code

Link_ModifyDirInDoorway:
    ; In a doorway (UW), you can only move in the direction
    ; that you entered it or the opposite.
    ; If not in a doorway nor moving, then return.
    LDA DoorwayDir
    BEQ @Exit
    LDY ObjInputDir
    BEQ @Exit

    ; If the facing direction is part of the input direction, then
    ; keep moving in the facing direction.
    LDA ObjDir
    AND ObjInputDir
    BNE :+

    ; If the opposite of the facing direction is part of the input direction, then
    ; face the opposite direction.
    LDA ObjDir
    JSR GetOppositeDir
    AND ObjInputDir

    ; If neither direction matched input direction, then
    ; change input direction to facing direction.
    BNE :+
    LDA ObjDir
:
    STA ObjInputDir

@Exit:
    RTS

; Description:
; This procedure is the same as HideSpritesOverLink in bank 4.
;
Unused_HideSpritesOverLink_Bank5:
    ; Begin unverified code 1515F
    LDA #$F8
    STA Sprites+64
    STA Sprites+68
    RTS
    ; End unverified code

; To be considered within a doorway, one condition is that
; Link's perpendicular coordinate ([00]) has to match the doorway's
; (X=$78 for verticals, Y=$8D for horizontals).
;
; See GetPlayerCoordsForDirection.
;
DoorwayRequiredCoord:
    .BYTE $78, $78, $8D, $8D

; Player coordinate must be >= these bounds.
; Using these bounds; the player will be considered within the
; bounds of a doorway, if strictly inside or 1 pixel outside.
;
DoorwayBoundsMinOver:
    .BYTE $3D, $BD, $00, $CF

; Player coordinate must be < these bounds.
; Using these bounds; the player will be considered within the
; bounds of a doorway, if strictly inside or 1 pixel outside.
;
DoorwayBoundsMaxOver:
    .BYTE $5E, $DE, $21, $F1

; Player coordinate must be >= these bounds.
; Using these bounds; the player will be considered within the
; bounds of a doorway, if strictly inside except for 1 or 2 pixels
; at the edge.
;
DoorwayBoundsMinUnder:
    .BYTE $3D, $BF, $00, $D2

; Player coordinate must be < these bounds.
; Using these bounds; the player will be considered within the
; bounds of a doorway, if strictly inside except for 1 or 2 pixels
; at the edge.
;
DoorwayBoundsMaxUnder:
    .BYTE $5C, $DE, $1F, $F1

; Params:
; [0F]: moving direction
;
; Returns:
; [0E]: reverse index of doorway direction found, or $FF if blocked
; [0F]: untouched, or changed from 0 to a moving direction
;
; When not at or in a doorway, this function leaves
; [0F] moving direction as is.
;
; When blocked by a door, [0E] will be set to $FF. But [0F]
; will be left as is, assuming that it had been reset before this
; routine by BoundByRoom.
;
; Otherwise, [0F] will be changed from 0 to the door's direction.
;
CheckDoorway:
    ; If already in a doorway, go handle it separately.
    LDA DoorwayDir
    BNE @InDoorway

@SearchOverflowBounds:
    ; Not in a doorway. Look for a doorway that Link might be in.
    ;
    ; To match a doorway:
    ; 1. Link's [00] coordinate has to match the doorway's
    ;    (X=$78 for verticals, Y=$8D for horizontals)
    ; 2. Link's [01] coordinate has to >=  min bound, and < max bound
    ;
    ; The "overflow" bounds are used that consider 1 pixel outside
    ; a doorway to be part of it.
    LDA ObjDir
    JSR GetPlayerCoordsForDirection
    LDY #$03

@LoopOverflowBounds:
    LDA $00                     ; [00] is Link's Y, if facing horizontally; else X.
    CMP DoorwayRequiredCoord, Y
    BNE :+
    LDA $01                     ; [01] is Link's X, if facing horizontally; else Y.
    CMP DoorwayBoundsMinOver, Y
    BCC :+
    CMP DoorwayBoundsMaxOver, Y
    BCC @TestDoorwayDoor        ; If found a doorway, go see what it does when you touch it.
:
    DEY
    BPL @LoopOverflowBounds

@NotInDoorway:
    ; Not in any doorway. Reset DoorwayDir.
    ; Leave [0F] moving direction as is.
    LDA #$00
    STA DoorwayDir
    RTS

@InDoorway:
    ; In a doorway. DoorwayDir is in A.
    PHA
    JSR GetPlayerCoordsForDirection
    PLA
    JSR GetOppositeDir          ; Get reverse index of doorway direction.
    LDA $01                     ; [01] is Link's X, if facing horizontally; else Y.

    ; If the player is within the bounds of the doorway for DoorwayDir
    ; (for example the left door way, if DoorwayDir = left;
    ; instead of right doorway while DoorwayDir = left),
    ; and player is facing in DoorwayDir, then go repeat the
    ; original search used to enter the doorway.
    CMP DoorwayBoundsMinOver, Y
    BCC @SearchUnderflowDoorway
    CMP DoorwayBoundsMaxOver, Y
    BCS @SearchUnderflowDoorway
    LDA DoorwayDir
    CMP ObjDir
    BEQ @SearchOverflowBounds

@SearchUnderflowDoorway:
    ; The player might be in the original doorway facing backwards,
    ; outside it, or in the other doorway along the axis.
    ;
    ; Look for a doorway that Link might be in.
    ; The difference between this search and the one above
    ; is that we check the "underflow" bounds that are shorter than
    ; the full doorway length.
    LDY #$03

@LoopUnderflowDoorway:
    LDA $00                     ; [00] is Link's Y, if facing horizontally; else X.
    CMP DoorwayRequiredCoord, Y
    BNE :+
    LDA $01                     ; [01] is Link's X, if facing horizontally; else Y.
    CMP DoorwayBoundsMinUnder, Y
    BCC :+
    CMP DoorwayBoundsMaxUnder, Y
    BCC @TestDoorwayDoor
:
    DEY
    BPL @LoopUnderflowDoorway
    BMI @NotInDoorway           ; Not in a doorway. Go set DoorwayDir to 0, and leave [0F] alone.

@TestDoorwayDoor:
    ;
    ;
    ; [0E] holds reverse index of doorway direction found.
    STY $0E

    ; Store input direction in [02] and [0C].
    LDA ObjInputDir
    AND #$0F
    STA $02
    STA $0C

    ; If input direction is not the doorway direction found, return.
    CMP ReverseDirections, Y
    BNE @Exit

    ; Touch the door in the direction we found.
    JSR FindDoorTypeByDoorBit
    STA $0D                     ; [0D] holds the door type.
    JSR TouchDoor               ; Can set [0E] to $FF, if a door blocks the way.

    ; If blocked, then return and leave DoorwayDir as it was.
    LDY $0E
    BMI @Exit

    ; Set variables to doorway direction found.
    LDA ReverseDirections, Y
    STA ObjDir
    STA $0F                     ; [0F] movement direction
    STA DoorwayDir

    ; Movement was not blocked at a doorway.
    ; If we're at a false wall or bombable, then go pass thru it
    ; and leave the room.
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
    JMP GoToNextModeFromPlay

; Params:
; A: direction
;
; Returns:
; [00]: coordinate on perpendicular axis
; [01]: coordinate on direction's axis
;
; [00]: Y if facing left or right, else X
; [01]: X if facing left or right, else Y
;
GetPlayerCoordsForDirection:
    LDX ObjX
    LDY ObjY
    AND #$03
    BEQ :+
    LDY ObjX
    LDX ObjY
:
    STX $00
    STY $01
    RTS
