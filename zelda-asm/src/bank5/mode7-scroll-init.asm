; ============================================================================
; FILE:     bank5/mode7-scroll-init.asm  BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 723-1037
; SUBJECTS: (none)
; TOPIC:    Mode 7 init: next room layout, dark-room fade
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0028 InitMode7Submodes                              code
;   L0032 InitMode7Submodes_JumpTable                    data
;   L0041 InitMode7_Sub1                                 code
;   L0091 LayOutDoorsPrev                                code
;   L0104 InitMode7_Sub0                                 code
;   L0116 InitMode7_Sub2                                 code
;   L0151 L1433A_IncSubmode                              code
;   L0156 InitMode7_Sub3And4_TransferPlayAreaAttrsToNT2  code
;   L0187 InitMode7_Sub5                                 code
;   L0249 InitMode7_Sub6                                 code
;   L0253 InitMode7_Finish                               code
;   L0260 L143AD_Exit                                    code
;   L0263 Sprite0Descriptor                              data
;   L0266 WriteAndEnableSprite0                          code
;   L0277 SetEnteringDoorwayAsCurOpenedDoors             code
;   L0289 RoomPaletteSelectorToNTAttr                    data
;   L0296 FillPlayAreaAttrs                              code  Params:
; ============================================================================

InitMode7Submodes:
    LDA GameSubmode
    JSR TableJump

InitMode7Submodes_JumpTable:
    .ADDR InitMode7_Sub0
    .ADDR InitMode7_Sub1
    .ADDR InitMode7_Sub2
    .ADDR InitMode7_Sub3And4_TransferPlayAreaAttrsToNT2
    .ADDR InitMode7_Sub3And4_TransferPlayAreaAttrsToNT2
    .ADDR InitMode7_Sub5
    .ADDR InitMode7_Sub6

InitMode7_Sub1:
    JSR DrawSpritesBetweenRooms
    JSR Link_EndMoveAndAnimateBetweenRooms
    LDA CurOpenedDoors          ; Assign the current opened doors to the previous one.
    STA PrevOpenedDoors

    ; If this doorway Link is entering from is not a true door,
    ; then LayOutDoors called from LayOutRoom will clear it.
    JSR SetEnteringDoorwayAsCurOpenedDoors
    DEC PrevRow                 ; TODO: ?
    INC GameSubmode
    JSR CalculateNextRoom
    LDA NextRoomId

    ; If next room ID is invalid, then return because
    ; something went wrong calculating it.
    ; The mode was changed to load the OW.
    BMI @Exit
    LDA RoomId
    PHA                         ; Save current room ID.
    LDY ObjDir
    CPY #$08

    ; If we're not going up, then temporarily set RoomId to next
    ; room; so we can draw next room in NT 2 below and
    ; subsequent submodes.
    ;
    ; Only in up direction do we draw the current room in NT 2.
    BEQ :+
    LDA NextRoomId
    STA RoomId
:
    JSR FillPlayAreaAttrs
    LDA #$15                    ; Start at bottom row.
    STA CurRow
    LDY ObjDir
    CPY #$08
    BEQ :+                      ; If going up, then go lay out the only dynamic elements for the current room: the doors.
    JSR LayOutRoom              ; else lay out the next room.

@RestoreRoomId:
    PLA                         ; Restore current room ID.
    STA RoomId

@Exit:
    RTS
:
    JSR LayOutDoorsPrev         ; Here, "previous" means "current", because we already changed TempDoorDirHeldOpen.
    JMP @RestoreRoomId          ; Go restore current room ID and return.

LayOutDoorsPrev:
    LDA CurLevel
    BEQ :+                      ; If in OW, then return.
    LDA CurOpenedDoors
    PHA
    LDA PrevOpenedDoors
    STA CurOpenedDoors
    JSR LayOutDoors
    PLA
    STA CurOpenedDoors
:
    RTS

InitMode7_Sub0:
    ; If teleporting, then set the room ID to the one that will
    ; make it look like we scrolled to a dungeon entrance.
    LDA WhirlwindTeleportingState
    BEQ :+
    LDA WhirlwindPrevRoomId
    STA RoomId
:
    LDA SecretColorCycle
    BEQ L1433A_IncSubmode       ; If we had no flute secret (pond) or it finished, then go to next submode.
    JMP AnimatePond             ; Go reverse the flute secret (pond colors).

InitMode7_Sub2:
    ; This transfers a room's play area tiles to nametable 2.
    ;
    ; CurRow starts at the bottom ($15).
    JSR CopyRowToTileBuf
    LDA DynTileBuf

    ; Change the target nametable from 0 to 2.
    ;
    ; Because there's only a play area and no status bar in
    ; nametable 2, you have to add 7 to the high byte instead of 8.
    ; So, the top of the play area in NT 0 is at $2100.
    ; But the top of the play area in NT 2 is at $2800.
    ;
    ; This assumes that you want to scroll vertically, where the
    ; two scroll areas must be contiguous vertically.
    AND #$0F
    CLC
    ADC #$27
    STA DynTileBuf
    LDA ObjDir
    CMP #$04
    BCS :+                      ; If the scroll direction is not vertical,

    ; then you DO have to account for the status bar.
    ; So, add $100 (by incrementing high byte) to move
    ; everything down 64 pixels.
    ;
    ; Now the play areas will line up in NT 0 and NT 2 when
    ; we change mirroring to vertical at the end of this mode.
    INC DynTileBuf
:
    DEC CurRow
    BPL :+

L1433A_IncSubmode:
    INC GameSubmode
:
    RTS

InitMode7_Sub3And4_TransferPlayAreaAttrsToNT2:
    ; Transfer half of a room's play area attributes to nametable 2.
    ; The top half in submode 3, and the bottom half in submode 4.
    LDA #$08                    ; Test player's direction for up (8).
    BIT ObjDir
    BNE @Vertical
    LSR                         ; Test player's direction for down (4).
    BIT ObjDir
    BEQ @Horizontal

@Vertical:
    ; Scrolling up or down.
    LDA #$C0                    ; This will mean PPU address $2BC0.

@TransferHalfPlayAreaAttrs:
    LDY #$17                    ; This is the offset of the end of the first half of play area NT attributes.
    LDX GameSubmode
    CPX #$03
    BEQ :+                      ; If in submode 4 instead of 3,
    JSR GetPlayAreaAttrsBottomHalfInfo    ; then refer to bottom half of NT attributes.
:
    JMP CueTransferPlayAreaAttrsHalfAndAdvanceSubmodeNT2

@Horizontal:
    ; Scrolling left or right.
    ; This will mean PPU address $2BD0, 64 pixels down from
    ; the top of NT 2. See submode 2 for an explanation of
    ; offsets and mirroring for vertical scrolling.
    LDA #$D0
    BNE @TransferHalfPlayAreaAttrs    ; Go continue setting up transfer of attributes.

InitMode7_Sub5:
    LDA #$00                    ; Reset fade cycle, in case we don't need to fade.
    STA FadeCycle
    LDA ObjDir
    CMP #$04
    BCS @Vertical               ; If direction is horizontal,

    ; then cue transfer of a row of blanks above the play area.
    ; This should lead to clean vertical scrolling without visual
    ; artifacts.
    ;
    ; TODO: But, maybe it doesn't work, because its address
    ; $28E0 is in NT 2, which will become a mirror of NT 0 below
    ; in this submode.
    LDY #$4E
    STY TileBufSelector

@Vertical:
    CMP #$08
    BNE @CheckDark

    ; Scrolling up.

    ; Lay out the next room now, because we had to transfer
    ; the current one to NT 2 earlier. Up is the only direction
    ; that draws the current room to NT 2.
    LDA RoomId
    PHA                         ; Save current room ID.
    LDA NextRoomId              ; Lay out the next room.
    STA RoomId
    JSR LayOutRoom
    PLA                         ; Restore current room ID.
    STA RoomId

@CheckDark:
    ; Scrolling in any direction.
    LDY NextRoomId
    JSR IsDarkRoom_Bank5
    BEQ InitMode7_Finish        ; If the next room is not dark, then go finish up and start updating.

    ; The next room is dark.
    LDY RoomId
    JSR IsDarkRoom_Bank5
    BNE @CheckIfLit             ; If the current room is dark by default, go see if it was lit up.

@DarkenRoom:
    ; Going from room with light to a dark room.
    LDA #$00                    ; The next room will not be lit yet.
    STA CandleState
    LDA #$40                    ; Start a fade-to-black cycle.
    STA FadeCycle
    INC GameSubmode
    RTS

@CheckIfLit:
    LDA CandleState
    BNE @DarkenRoom             ; If it was lit, then go start a fade-to-black cycle.

    ; Not lit. Dark to dark. Nothing to do.
    ; Go finish initializing, and begin updating the mode.
    BEQ InitMode7_Finish

InitMode7_Sub6:
    JSR AnimateWorldFading
    BNE L143AD_Exit             ; If we're still fading, then return.

InitMode7_Finish:
    ; Finish initializing this mode, and start updating it.
    LDA NextRoomId              ; Set current room to next room.
    STA RoomId
    JSR WriteAndEnableSprite0
    JSR BeginUpdateMode

L143AD_Exit:
    RTS

Sprite0Descriptor:
    .BYTE $27, $61, $20, $58

WriteAndEnableSprite0:
    LDA #$01
    STA IsSprite0CheckActive
    LDY #$03
:
    LDA Sprite0Descriptor, Y
    STA Sprites, Y
    DEY
    BPL :-
    RTS

SetEnteringDoorwayAsCurOpenedDoors:
    LDA ObjDir                  ; Calculate the opposite of the player's direction.
    LSR
    AND #$05
    STA $00
    LDA ObjDir
    ASL
    AND #$0A
    ORA $00
    STA CurOpenedDoors          ; That is the door the player's entering the new room from.
    RTS

RoomPaletteSelectorToNTAttr:
    .BYTE $00, $55, $AA, $FF

; Params:
; A: room ID
;
; Look up room attributes A for the room.
FillPlayAreaAttrs:
    TAY
    LDA LevelBlockAttrsA, Y
    AND #$03                    ; Get the outer palette selector from the byte.
    TAX
    LDA RoomPaletteSelectorToNTAttr, X    ; Get the nametable attributes for the palette selector.
    LDX #$2F                    ; Fill the play area NT attributes.
:
    STA PlayAreaAttrs, X
    DEX
    BPL :-
    LDA LevelBlockAttrsB, Y     ; Look up room attributes B for the room.
    AND #$03                    ; Get the inner palette selector.
    TAX

    ; Fill the inner play area NT attributes (offset 9 to $26).
    LDY #$09

@LoopRow:
    TYA
    AND #$07                    ; Skip left and right edges.
    BEQ @NextLoopRow
    CMP #$07
    BEQ @NextLoopRow
    CPY #$21                    ; For the bottom inner NT attribute row, go combine the inner and outer attributes.
    BCS @CombineInnerOuter
    LDA RoomPaletteSelectorToNTAttr, X
    STA PlayAreaAttrs, Y

@NextLoopRow:
    INY
    CPY #$27
    BCC @LoopRow                ; If we haven't finished the last inner NT attribute row, then go fill more.
    RTS

@CombineInnerOuter:
    ; Combine the NT attributes at current offset with the
    ; new ones we're filling; so that new (inner) attributes
    ; affect the top half of the row.
    LDA RoomPaletteSelectorToNTAttr, X
    AND #$0F
    STA $00
    LDA PlayAreaAttrs, Y
    AND #$F0
    ORA $00
    STA PlayAreaAttrs, Y
    JMP @NextLoopRow            ; Go advance the offset and check if we're done.
