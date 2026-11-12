; ============================================================================
; FILE:     bank7/next-room-room-flags.asm BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 737-803
; SUBJECTS: (none)
; TOPIC:    Triforce marker, CalculateNextRoom, MarkRoomVisited, GetRoomFlags
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0020 UpdateTriforcePositionMarker  code
;   L0031 CalculateNextRoom             code
;   L0043 CalculateNextRoom_TableJump   code
;   L0047 CalculateNextRoom_JumpTable   data
;   L0058 CalculateNextRoomOW           code
;   L0063 LevelMasks                    data
;   L0066 MarkRoomVisited               code
;   L0071 Room_MarkVisited_Return       code
;   L0079 GetRoomFlags                  code  Returns:
; ============================================================================

UpdateTriforcePositionMarker:
    LDA CurLevel
    BEQ Room_MarkVisited_Return                    ; If in OW, then return.
    LDA #$05
    JSR SwitchBank
    JSR HasCompass
    BEQ Room_MarkVisited_Return                    ; If player hasn't gotten the compass, then return.
    LDA LevelInfo_TriforceRoomId
    LDX #$04
    JMP UpdatePositionMarker

CalculateNextRoom:
    LDY CurLevel
    BEQ CalculateNextRoomOW     ; If in OW, all directions are open.

    ; Look up door type for player's direction.
    LDA ObjDir
    STA $02
    LDA #$05
    JSR SwitchBank
    JSR FindDoorTypeByDoorBit
    LDY $01                     ; The same as [02].

CalculateNextRoom_TableJump:
    STY $E7                     ; Set [E7] to a door bit/direction.
    JSR TableJump               ; A holds the door type.

CalculateNextRoom_JumpTable:
    .ADDR CalculateNextRoomForDoor
    .ADDR CalculateNoNextRoom
    .ADDR CalculateNextRoomForDoor
    .ADDR CalculateNextRoomForDoor
    .ADDR CalculateNextRoomForDoor
    .ADDR CalculateNextRoomForDoor
    .ADDR CalculateNextRoomForDoor
    .ADDR CalculateNextRoomForDoor
    .ADDR CalculateNoNextRoom

CalculateNextRoomOW:
    LDY ObjDir                  ; Use player's direction bit.
    LDA #$00                    ; Use "open" door type.
    BEQ CalculateNextRoom_TableJump

LevelMasks:
    .BYTE $01, $02, $04, $08, $10, $20, $40, $80

MarkRoomVisited:
    JSR GetRoomFlags
    ORA #$20                    ; Visit state (UW)
    STA ($00), Y

Room_MarkVisited_Return:
    RTS

; Returns:
; A: flags for the room in level block world flags
; Y: room ID
; [$00:01]: address of level block world flags
;
GetRoomFlags:
    LDA LevelInfo_WorldFlagsAddr
    STA $00
    LDA LevelInfo_WorldFlagsAddr+1
    STA $01
    LDY RoomId
    LDA ($00), Y
    RTS
