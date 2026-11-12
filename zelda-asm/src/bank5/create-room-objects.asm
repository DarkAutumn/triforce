; ============================================================================
; FILE:     bank5/create-room-objects.asm BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 8199-8341
; SUBJECTS: Mmc1, Room
; TOPIC:    Room setup of the room item (object slot $13), the dungeon push block and the overworld
;           heart container of room $5F; then the bank 5 tail: its copy of the reset handler, the MMC1
;           control and bank switch writers, and the vectors.
; SEE ALSO: bank7/init-mode5-play.asm (Room_RunEntryTasks, the caller);
;           bank1/take-item.asm (Item_TryTakeRoomItem, RoomFlags_GetItemTaken); bank1/bomb-candle-wield.asm
;           (Room_GetShortcutOrItemXY); bank5/uw-floor-push-block.asm (PushBlock_FindAndCreate);
;           bank5/underworld-secrets.asm (Secret_CheckUnderworld: trigger 7 reveals the hidden room
;           item); bank4/ganon-draw-collisions.asm (reveals the room item of Ganon's room);
;           bank4/ghini-armos.asm (an Armos secret turns the room item into the bracelet, $14);
;           bank7/room-item-draw.asm (Item_MoveAndDrawRoomItem);
;           bank7/reset-mmc1-vectors.asm (Reset_Handler, Mmc1_SetControl, Mmc1_SwitchBank: the fixed-bank originals).
; KEY FACTS:
;   - The room item is object slot $13: ObjState+19 = 0 means present, negative means absent;
;     RoomItemId holds its item id. Room_SetUpItemAndPushBlock
;   - Dungeon room item id = LevelBlockAttrsE & $1F; id 3 means "no item". Room_SetUpItemAndPushBlock
;   - A dungeon room item is absent when the room's item-taken flag (room flags bit 4, $10) is set.
;     Room_SetUpItemAndPushBlock
;   - With secret trigger (LevelBlockAttrsF & 7) = 3 (Triforce of Power taken -> shutters open) or 7
;     (all monsters dead -> the item appears) the item starts absent. For 7 Secret_CheckUnderworld
;     makes it appear; Ganon_ActivateRoomItem does so when Ganon dies. Room_SetUpItemAndPushBlock
;   - A dungeon room with LevelBlockAttrsD bit 6 gets a push block object. Room_SetUpItemAndPushBlock
;   - The item position comes from LevelInfo_ShortcutOrItemPosArray (index = attr F bits 5-4); a
;     Triforce piece ($1B) is placed 8 pixels further left. Room_SetUpItemAndPushBlock
;   - Overworld: room setup always makes the room item a heart container ($1A), present only in game
;     mode 5 in room $5F, at X $C0, Y $90. Room_SetUpItemAndPushBlock
; CONTENTS: (generated; L = line in this file)
;   L0051 Room_SetUpItemAndPushBlock  code  Sets up the room item (object slot $13) for the current room, and the push block in a dungeon room.
;   L0165 Mmc1_SetControl_Bank5       code  Writes the low 5 bits of A to the MMC1 control register (bank 5 copy of Mmc1_SetControl).
;   L0184 Mmc1_SwitchBank_Bank5       code  Switches the PRG bank at $8000-$BFFF to bank A (bank 5 copy of Mmc1_SwitchBank).
; ============================================================================

; ----------------------------------------------------------------------------
; Sets up the room item (object slot $13) for the current room, and the push block in a dungeon room.
; OUT:  ObjState+19 = 0 if the room item is present, negative if not; RoomItemId; ObjX+19, ObjY+19;
;       object slot $0B set up as a push block when the room has one.
; NOTE: Called from Room_RunEntryTasks (game modes 4, 5, 6, 9, $B, $C), so it runs
;       more than once per room visit.
;       Dungeon: item id = LevelBlockAttrsE & $1F (3 = none); absent if already taken (room flags
;       bit 4, $10), if the id is 3, or if the secret trigger is 3 or 7 (hidden until revealed). When
;       the item was already taken, RoomItemId is not written and keeps its old value.
;       Overworld: always item $1A (heart container), present only in mode 5 in room $5F at
;       ($C0, $90); in any other room or mode ObjState+19 = $FF and ObjX+19/ObjY+19 are not
;       written. This routine does not test whether it was taken (Item_TryTakeRoomItem and
;       Item_MoveAndDrawRoomItem test the room's item-taken flag).
;       A Triforce piece ($1B) is placed 8 pixels left of the looked-up position.
; ----------------------------------------------------------------------------
Room_SetUpItemAndPushBlock:
    ; ObjState[$13] = 0: room item present, until decided otherwise below.
    LDA #$00
    STA ObjState+19

    ; Overworld: the only room item is the heart container of room $5F.
    LDA CurLevel
    BEQ @MakeHeartContainerOW

    ; Room item already taken (A = $10): absent. Y = RoomId after this call.
    JSR RoomFlags_GetItemTaken
    BNE @Deactivate

    ; Look up the item for this room, and store it.
    ; Item id 3 (elsewhere the magic sword) stands for "no item" here: the usual "no item" value
    ; $3F does not fit in the 5-bit field.
    LDY RoomId
    LDA LevelBlockAttrsE, Y
    AND #$1F                    ; Room item
    CMP #$03
    BNE :+
    DEC ObjState+19             ; No item: ObjState[$13] = $FF.
:
    STA RoomItemId

    ; Secret trigger 3 (Triforce of Power taken -> shutters open) or 7 (all monsters dead -> the
    ; item appears): the item starts absent. Secret_CheckUnderworld reveals it for trigger 7;
    ; Ganon_ActivateRoomItem reveals the room item when Ganon dies.
    LDA LevelBlockAttrsF, Y
    AND #$07                    ; Secret trigger
    CMP #$03
    BEQ @Deactivate
    CMP #$07
    BNE :+

@Deactivate:
    DEC ObjState+19             ; Absent: ObjState[$13] = $FF ($FE if it was already $FF).
:
    ; If the room has a push block (attr D bit 6), find its tile and create the object.
    LDA LevelBlockAttrsD, Y
    AND #$40                    ; Push block
    BEQ :+
    JSR PushBlock_FindAndCreate
:
    ; A = X and Y = Y of the room item, from the level's four item positions.
    JSR Room_GetShortcutOrItemXY

@StoreLocation:
    STA ObjX+19
    STY ObjY+19

    ; If the room's item attribute is a Triforce piece, then move the item left 8 pixels.
    ; (The overworld path for room $5F also runs this test, on a byte that is cave ware data there.)
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
    LDA #$1A                    ; Heart container
    STA RoomItemId

    ; Position of the heart container in overworld room $5F: X = $C0, Y = $90.
    LDA #$C0
    LDY #$90

    ; Present only in game mode 5 (play) in room $5F.
    LDX GameMode
    CPX #$05
    BNE :+
    LDX RoomId
    CPX #$5F
    BEQ @StoreLocation
:
    ; Any other overworld room or mode: absent (ObjState[$13] = $FF).
    DEC ObjState+19
    RTS

.SEGMENT "BANK_05_ISR"


.EXPORT Mmc1_SetControl_Bank5
.EXPORT Mmc1_SwitchBank_Bank5


; DATA: bank 5's copy of the reset handler, at $BF50 (the reset vector below), stored as bytes.
; Same instructions as Reset_Handler in the fixed bank: SEI; CLD; PPU control = 0; stack = $FF; wait for
; the vertical blank flag twice; write $FF to $8000/$A000/$C000/$E000 (reset the MMC1 shift
; registers); A = $0F, JSR $BF98 (Mmc1_SetControl_Bank5); five writes of 0 to $A000 (CHR bank 0);
; A = 7, JSR $BFAC (Mmc1_SwitchBank_Bank5); JMP $E440 (the target of Reset_Handler's final JMP, Reset_RunGame).
    .BYTE $78, $D8, $A9, $00, $8D, $00, $20, $A2
    .BYTE $FF, $9A, $AD, $02, $20, $29, $80, $F0
    .BYTE $F9, $AD, $02, $20, $29, $80, $F0, $F9
    .BYTE $09, $FF, $8D, $00, $80, $8D, $00, $A0
    .BYTE $8D, $00, $C0, $8D, $00, $E0, $A9, $0F
    .BYTE $20, $98, $BF, $A9, $00, $8D, $00, $A0
    .BYTE $4A, $8D, $00, $A0, $4A, $8D, $00, $A0
    .BYTE $4A, $8D, $00, $A0, $4A, $8D, $00, $A0
    .BYTE $A9, $07, $20, $AC, $BF, $4C, $40, $E4

; ----------------------------------------------------------------------------
; Writes the low 5 bits of A to the MMC1 control register (bank 5 copy of Mmc1_SetControl).
; IN:   A = control value ($0E = vertical mirroring from the curtain code; the reset stub passes $0F)
; OUT:  A shifted right 4 times.
; NOTE: One bit per write, low bit first: five writes to $8000. At $BF98. Called by name from the
;       common code that runs from RAM (bank 1's BANK_01_CODE), and by address from the reset stub.
; ----------------------------------------------------------------------------
Mmc1_SetControl_Bank5:
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

; ----------------------------------------------------------------------------
; Switches the PRG bank at $8000-$BFFF to bank A (bank 5 copy of Mmc1_SwitchBank).
; IN:   A = bank number
; OUT:  A shifted right 4 times.
; NOTE: Five one-bit writes to $E000, low bit first. At $BFAC. Called by name from the common code
;       that runs from RAM (bank 1's BANK_01_CODE), and by address from the reset stub (A = 7).
; ----------------------------------------------------------------------------
Mmc1_SwitchBank_Bank5:
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




; DATA: this bank's interrupt vectors at $BFFA-$BFFF, stored as bytes: NMI $E484 (Nmi_Isr's address in
; the fixed bank's vectors), reset $BF50 (the reset handler copy above), IRQ $BFF0.
    .BYTE $84, $E4, $50, $BF, $F0, $BF
