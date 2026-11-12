; ============================================================================
; FILE:     bank5/create-room-objects.asm BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 8199-8341
; SUBJECTS: (none)
; TOPIC:    Room item, push block, overworld heart container; bank tail
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0014 CreateRoomObjects      code
;   L0126 SetMMC1Control_Local5  code
;   L0138 SwitchBank_Local5      code
; ============================================================================

CreateRoomObjects:
    ; Reset ObjState[$13] to activate room item object.
    LDA #$00
    STA ObjState+19

    ; If in OW, go create heart container in room 5F.
    LDA CurLevel
    BEQ @MakeHeartContainerOW

    ; If the player got the room item already, go deactivate
    ; the room item object.
    JSR GetRoomFlagUWItemState
    BNE @Deactivate

    ; Look up the item for this room, and store it.
    ; If it's item ID 3, then deactivate the object.
    ;
    ; Item ID 3 normally means the master sword. But because the
    ; usual value meaning "no item" ($3F) can't fit in the level
    ; block attribute for a room item (up to $1F); use 3 to
    ; stand in for it.
    LDY RoomId
    LDA LevelBlockAttrsE, Y
    AND #$1F                    ; Room item
    CMP #$03
    BNE :+
    DEC ObjState+19
:
    STA RoomItemId

    ; If secret trigger is 3 "last boss" or 7 "foes for item",
    ; then deactivate room item object. They'll be activated by
    ; the secret action.
    LDA LevelBlockAttrsF, Y
    AND #$07                    ; Secret trigger
    CMP #$03
    BEQ @Deactivate
    CMP #$07
    BNE :+

@Deactivate:
    DEC ObjState+19             ; Deactivate room item object by setting ObjState[$13] to $FF.
:
    ; If there's a push block in the room, then look for where
    ; it goes, and activate it.
    LDA LevelBlockAttrsD, Y
    AND #$40                    ; Push block
    BEQ :+
    JSR FindAndCreatePushBlockObject
:
    ; Set the X and Y for the room item object.
    JSR GetShortcutOrItemXY

@StoreLocation:
    STA ObjX+19
    STY ObjY+19

    ; If the room item is a triforce piece,
    ; the move it left 8 pixels.
    LDY RoomId
    LDA LevelBlockAttrsE, Y
    AND #$1F                    ; Room item
    CMP #$1B                    ; Triforce piece
    BNE :+
    LDA ObjX+19
    SEC
    SBC #$08
    STA ObjX+19
:
    RTS

@MakeHeartContainerOW:
    ; Try to make the heart container in OW.
    LDA #$1A                    ; Heart container
    STA RoomItemId

    ; Load the coordinates of the heart container in
    ; OW room $5F.
    LDA #$C0
    LDY #$90

    ; If in mode 5, and in room $5F, then go store the coordinates
    ; in the object slot.
    LDX GameMode
    CPX #$05
    BNE :+
    LDX RoomId
    CPX #$5F
    BEQ @StoreLocation
:
    ; Else deactivate the object.
    DEC ObjState+19
    RTS

.SEGMENT "BANK_05_ISR"


.EXPORT SetMMC1Control_Local5
.EXPORT SwitchBank_Local5


; Unknown block
    .BYTE $78, $D8, $A9, $00, $8D, $00, $20, $A2
    .BYTE $FF, $9A, $AD, $02, $20, $29, $80, $F0
    .BYTE $F9, $AD, $02, $20, $29, $80, $F0, $F9
    .BYTE $09, $FF, $8D, $00, $80, $8D, $00, $A0
    .BYTE $8D, $00, $C0, $8D, $00, $E0, $A9, $0F
    .BYTE $20, $98, $BF, $A9, $00, $8D, $00, $A0
    .BYTE $4A, $8D, $00, $A0, $4A, $8D, $00, $A0
    .BYTE $4A, $8D, $00, $A0, $4A, $8D, $00, $A0
    .BYTE $A9, $07, $20, $AC, $BF, $4C, $40, $E4

SetMMC1Control_Local5:
    STA $8000
    LSR
    STA $8000
    LSR
    STA $8000
    LSR
    STA $8000
    LSR
    STA $8000
    RTS

SwitchBank_Local5:
    STA $E000
    LSR
    STA $E000
    LSR
    STA $E000
    LSR
    STA $E000
    LSR
    STA $E000
    RTS

.SEGMENT "BANK_05_VEC"




; Unknown block
    .BYTE $84, $E4, $50, $BF, $F0, $BF
