; ============================================================================
; FILE:     bank5/update-doors.asm       BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 5038-5309
; SUBJECTS: (none)
; TOPIC:    Door open/close commands, timers, flags in both rooms
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0015 UpdateDoors                                   code
;   L0145 L_SetOpenedDoorMaskAndResetCmdAndLayOutDoors  code
;   L0148 L_ResetDoorCmdAndLayOutDoors                  code
;   L0210 PrepareWriteHorizontalDoorTransferRecords     code  Params:
; ============================================================================

UpdateDoors:
    ; If mode = $12, or door timer <> 0, or door command = 0,
    ; then return.
    LDA GameMode
    CMP #$12
    BEQ L165D4_Exit
    LDA DoorTimer
    BNE L165D4_Exit
    LDA TriggeredDoorCmd
    BEQ L165D4_Exit

    ; Turn the door command into the desired open or closed state
    ; to store in [08].
    ;
    ; cmd:  7 6 3 2 
    ; [08]: 2 0 0 0
    ;
    ; There are other combinations, but 2, 3, 6, and 7 are the only
    ; commands intended to be used.
    ;
    ; 3 is the end state of 2: close
    ; 7 is the end state of 6: open
    AND #$07
    LDY #$01
    STY $02
    BIT $02
    BEQ :+
    LSR
:
    CMP #$02
    BNE :+

    ; Command 2 sets Link's timer to $30.
    LDY #$30
    STY ObjTimer
:
    AND #$03
    SEC
    SBC #$01
    AND #$02
    STA $08

    ; The command to close a key door or bombable wall does not
    ; do anything.
    LDA TriggeredDoorCmd
    CMP #$05
    BCS :+
    LDA TriggeredDoorDir
    STA $02
    JSR FindDoorTypeByDoorBit
    CMP #$07
    BEQ :+
    JMP L_ResetDoorCmdAndLayOutDoors
:
    ; TriggeredDoorCmd >= 5 or door type = 7
    ;
    ; Shutters and the commands to open a door always change
    ; tiles.
    JSR PrepareWriteHorizontalDoorTransferRecords

@LoopTransferRec:
    ; Copy [06] to [04] for the call to write tiles below.
    ; We need to remember how many tiles need to be copied
    ; in each transfer record.
    LDA $06
    STA $04

    ; Write the transfer record header for the door.
    LDA $00                     ; VRAM address high byte of door
    STA DynTileBuf, X
    INX
    LDA $01                     ; VRAM address low byte of door
    STA DynTileBuf, X
    INX
    LDA $06                     ; 2 for 2 tiles
    STA DynTileBuf, X
    INX
:
    ; Write two tiles in a short loop indexed by [04].
    JSR WriteDoorFaceTileHorizontally
    BNE :-

    ; OR the low VRAM address with $20 to go down one row
    ; in order to work on the second row of door tiles.
    LDA #$20
    ORA $01                     ; [01] low VRAM address
    STA $01

    ; Bottom of the loop.
    ; Loop again to write the second transfer record for the
    ; second pair of door tiles, if [05] has not reached 0.
    DEC $05
    BNE @LoopTransferRec

    ; Write the end marker, and update buffer length.
    LDA #$FF
    STA DynTileBuf, X
    TXA
    STA DynTileBufLen

    ; Increment the command.
    INC TriggeredDoorCmd

    ; If (new command AND 3) = 0, go update door flags and masks,
    ; and resetting the door command.
    ; This ends up catching original states 3 and 7.
    ;
    ; Else set door timer to 8 and return.
    LDA TriggeredDoorCmd
    AND #$03
    BEQ :+
    LDA #$08
    STA DoorTimer
    RTS
:
    ; If the door command = 4 after incrementing it, then
    ; it was 3 (close door). So:
    ; 1. reset this door's flag
    ; 2. remove it from the opened door mask
    ; 3. reset the door command
    ; 4. go lay out doors in the play area map
    LDA TriggeredDoorCmd
    CMP #$04
    BNE :+
    LDX $09                     ; [09] door direction index
    JSR ResetDoorFlag
    LDA TriggeredDoorDir        ; Remove the triggered door from opened door mask.
    EOR #$0F
    AND CurOpenedDoors

L_SetOpenedDoorMaskAndResetCmdAndLayOutDoors:
    STA CurOpenedDoors

L_ResetDoorCmdAndLayOutDoors:
    LDA #$00                    ; Reset door command
    STA TriggeredDoorCmd
    JMP LayOutDoors
:
    ; Door command <> 4.
    ; It must be 8, which means it was 7: open door.
    ;
    ; If the door is a shutter (7), then go add the door to the
    ; opened door mask, reset door command, and lay out doors.
    LDA TriggeredDoorDir
    STA $02
    JSR FindDoorTypeByDoorBit
    CMP #$07
    BEQ :+

    ; Else the door is not a shutter.
    LDX $09                     ; [09] door direction index
    JSR SetDoorFlag

    ; Get the next room's ID.
    TYA
    CLC
    ADC DoorNextRoomIdOffsets, X
    TAY

    ; Flip the door direction index.
    TXA
    EOR #$01
    TAX

    ; Set the door flag for the opposite door in the next room.
    LDA ($00), Y
    ORA LevelMasks, X
    STA ($00), Y
:
    ; Add the door to the opened door mask, reset door command,
    ; and lay out doors.
    LDA TriggeredDoorDir
    ORA CurOpenedDoors
    JMP L_SetOpenedDoorMaskAndResetCmdAndLayOutDoors

; Params:
; [08]: 0 if closed
;
; Returns:
; X: current length of dynamic transfer buf
; [00]: door VRAM address high byte
; [01]: door VRAM address low byte
; [02:03]: address of door tiles for direction and face
; [05]: 2, the number of transfer records to write
; [06]: 2, the number of tiles in each record
; [07]: 0, the first index of a tile to transfer
; [09]: door direction index
;
; Note that this routine is called with triggered doors:
; doors that can change state. Their door type numbers are >= 4.
;
; If this routine were ever called with a fixed door
; ("open", "wall", or "false"), then it would produce non-sensical
; values for the door face.
;
PrepareWriteHorizontalDoorTransferRecords:
    ; If triggered door type >= 5 (keys and shutter), play door sound.
    LDA TriggeredDoorDir
    STA $02
    JSR FindDoorTypeByDoorBit
    CMP #$05
    BCC :+
    PHA
    LDA #$04                    ; Door sound
    JSR PlaySample
    PLA
:
    ; Calculate the door face index in three parts.
    ;
    ; First, turn door type 4 into 8, and 1 into 4
    ; (provisional face index in Y register).
    CMP #$04
    BNE :+
    LDA #$08
:
    CMP #$01
    BNE :+
    ; Begin unverified code 166CD
    LDA #$04
    ; End unverified code
:
    ; Second, subtract 3 from provisional face index.
    SEC
    SBC #$03
    TAY

    ; Lastly, if the door is closed ([08] = 0) and provisional
    ; face index >= 3, then subtract 1 from it.
    ;
    ; But if open ([08] <> 0) and provisional face index <> 5,
    ; then make it 1.
    LDA $08
    BEQ :+
    CPY #$05
    BEQ @MakeForwardIndex
    LDY #$01
:
    CPY #$03
    BCC @MakeForwardIndex
    DEY

@MakeForwardIndex:
    ; The routine that returned the door type also returned its
    ; reverse direction index in [03].
    ;
    ; We want a forward direction index. So, subtract [03] from 3.
    LDA #$03
    SEC
    SBC $03

    ; Call this to put the address of door face tiles in [02:03].
    JSR FetchDoorAddrsFaceTilesSrcAndPlayAreaDst

    ; Return the address of the door in the nametable in [01:00].
    ; Note the order is reversed, as usual with VRAM addresses.
    LDA DoorVramAddrsHi, X
    STA $00
    LDA DoorVramAddrsLo, X
    STA $01

    ; Return the direction of the door in [09], and the dynamic
    ; transfer buf length in X register.
    STX $09
    LDX DynTileBufLen

    ; Return some hardcoded values in [05], [06], [07]. See the comments above.
    LDA #$00
    STA $07
    LDA #$02
    STA $06
    STA $05
    RTS
