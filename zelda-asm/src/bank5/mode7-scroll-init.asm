; ============================================================================
; FILE:     bank5/mode7-scroll-init.asm  BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 723-1037
; SUBJECTS: (none)
; TOPIC:    Mode 7 (scroll to the next room) init submodes: find the next room, lay it out and send
;           its tiles and attributes to name table 2, fade to black when entering a dark room, then
;           arm the sprite-0 split. Also the play-area attribute fill used by other modes.
; SEE ALSO: bank7/init-mode-dispatch.asm (InitMode7 calls this and sets the mirroring);
;           bank5/mode7-scroll-update.asm and bank5/scroll-world.asm (the scroll itself);
;           bank7/next-room-room-flags.asm and bank5/next-room-calc.asm (CalculateNextRoom, mazes,
;           leaving the map); bank5/mode12-lay-out-room.asm (LayOutRoom, CopyRowToTileBuf);
;           bank5/lay-out-doors.asm (LayOutDoors); bank1/world-fading.asm (AnimateWorldFading);
;           bank5/door-flags-dark-room.asm (IsDarkRoom_Bank5).
; KEY FACTS:
;   - Submodes: 0 whirlwind room / undo pond secret, 1 next room and layout, 2 tile rows to NT 2
;     (22 frames), 3 and 4 attributes to NT 2, 5 dark-room test, 6 fade (InitMode7Submodes_JumpTable).
;   - RoomId becomes NextRoomId only at the end of init, before the scroll is shown
;     (InitMode7_Finish).
;   - A next room id with bit 7 set aborts the scroll: CalculateNextRoom has already switched to
;     mode 2 to load the overworld (InitMode7_Sub1).
;   - The doorway Link comes in through (opposite of his facing) starts out open in the new room
;     (SetEnteringDoorwayAsCurOpenedDoors); the old room's open doors are kept in PrevOpenedDoors.
;   - Scrolling up draws the current room into NT 2 and lays out the next room only in submode 5;
;     other directions lay out the next room in submode 1 (InitMode7_Sub1, InitMode7_Sub5).
;   - Entering a dark room from a lit one (lit by nature or by candle) clears CandleState and fades
;     to black, 4 steps 10 frames apart; dark to dark does nothing (InitMode7_Sub5, InitMode7_Sub6).
;   - In OW no room is dark: IsDarkRoom_Bank5 returns 0 when CurLevel is 0 (InitMode7_Sub5).
;   - Leaving a room waits while SecretColorCycle is nonzero: the flute pond's colour cycle is run
;     back to 0 first, one step on each frame with FrameCounter bit 2 set (InitMode7_Sub0).
;   - Room palettes: attrs A bits 1-0 = outer palette, attrs B bits 1-0 = inner palette
;     (FillPlayAreaAttrs).
; CONTENTS: (generated; L = line in this file)
;   L0055 InitMode7Submodes                              code  Runs the mode 7 init step selected by GameSubmode.
;   L0060 InitMode7Submodes_JumpTable                    data  TABLE: mode 7 init steps; index = GameSubmode (0-6); value = handler address.
;   L0077 InitMode7_Sub1                                 code  Mode 7 init step 1: finds the next room and lays out the room that goes into name table 2.
;   L0135 LayOutDoorsPrev                                code  Lays out the doors of the room in RoomId using PrevOpenedDoors as the open doors (UW only).
;   L0153 InitMode7_Sub0                                 code  Mode 7 init step 0: sets the whirlwind's source room and undoes a flute pond secret.
;   L0170 InitMode7_Sub2                                 code  Mode 7 init step 2: sends tile row CurRow of the laid-out room to name table 2; one row a frame.
;   L0205 L1433A_IncSubmode                              code  Advances GameSubmode (shared by init steps 0 and 2).
;   L0216 InitMode7_Sub3And4_TransferPlayAreaAttrsToNT2  code  Mode 7 init steps 3 and 4: cue the top, then the bottom half of PlayAreaAttrs for name table 2.
;   L0252 InitMode7_Sub5                                 code  Mode 7 init step 5: lays out the next room when going up, then decides on a fade to black.
;   L0314 InitMode7_Sub6                                 code  Mode 7 init step 6: runs the fade to black until it is done, then finishes the init.
;   L0319 InitMode7_Finish                               code  Ends mode 7 init: RoomId = NextRoomId, sprite-0 split armed, mode update begins.
;   L0326 L143AD_Exit                                    code  Shared exit of the mode 7 init steps above.
;   L0331 Sprite0Descriptor                              data  DATA: sprite record 0 used for the sprite-0 hit: Y $27, tile $61, attributes $20 (behind the
;   L0339 WriteAndEnableSprite0                          code  Sets IsSprite0CheckActive and writes Sprite0Descriptor to sprite record 0.
;   L0354 SetEnteringDoorwayAsCurOpenedDoors             code  Sets CurOpenedDoors to the single direction bit opposite Link's facing.
;   L0368 RoomPaletteSelectorToNTAttr                    data  TABLE: name table attribute byte using one palette in all four quadrants; index = palette
;   L0379 FillPlayAreaAttrs                              code  Fills PlayAreaAttrs (6 rows of 8 attribute bytes) with a room's outer and inner palettes.
; ============================================================================

; ----------------------------------------------------------------------------
; Runs the mode 7 init step selected by GameSubmode.
; ----------------------------------------------------------------------------
InitMode7Submodes:
    LDA GameSubmode
    JSR TableJump

; TABLE: mode 7 init steps; index = GameSubmode (0-6); value = handler address.
InitMode7Submodes_JumpTable:
    .ADDR InitMode7_Sub0                                   ; 0 whirlwind room id, undo pond secret
    .ADDR InitMode7_Sub1                                   ; 1 next room, attributes, layout
    .ADDR InitMode7_Sub2                                   ; 2 tile rows to NT 2, one per frame
    .ADDR InitMode7_Sub3And4_TransferPlayAreaAttrsToNT2    ; 3 top half of attributes to NT 2
    .ADDR InitMode7_Sub3And4_TransferPlayAreaAttrsToNT2    ; 4 bottom half of attributes to NT 2
    .ADDR InitMode7_Sub5                                   ; 5 lay out next room (up), dark-room test
    .ADDR InitMode7_Sub6                                   ; 6 fade to black, then finish

; ----------------------------------------------------------------------------
; Mode 7 init step 1: finds the next room and lays out the room that goes into name table 2.
; OUT:  PrevOpenedDoors = old CurOpenedDoors; CurOpenedDoors = the doorway being entered;
;       NextRoomId; PlayAreaAttrs filled; CurRow = $15; tile map laid out; GameSubmode + 1.
; NOTE: If NextRoomId has bit 7 set (no such room), nothing more is done here: CalculateNextRoom
;       has already switched to mode 2 to load the overworld. In OW CalculateNextRoom also applies
;       the mazes.
; ----------------------------------------------------------------------------
InitMode7_Sub1:
    JSR DrawSpritesBetweenRooms
    JSR Link_EndMoveAndAnimateBetweenRooms
    LDA CurOpenedDoors          ; Keep the open doors of the room being left.
    STA PrevOpenedDoors

    ; Mark the doorway Link is entering the new room from as open.
    ; If it is not a true door, then LayOutDoors called from
    ; LayOutRoom will clear it.
    JSR SetEnteringDoorwayAsCurOpenedDoors
    DEC PrevRow                 ; PrevRow is $FF after an earlier scroll, so this leaves $FE: still
                                ; no valid row. No other effect found; purpose (unverified).
    INC GameSubmode
    JSR CalculateNextRoom
    LDA NextRoomId

    ; If next room ID is invalid (bit 7 set), then return.
    ; The mode was already changed to 2, to load the OW.
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
    JSR FillPlayAreaAttrs       ; A = the room to draw (current room if going up, else next room).
    LDA #$15                    ; Submode 2 starts at the bottom row.
    STA CurRow
    LDY ObjDir
    CPY #$08
    BEQ @GoingUp                ; If going up, the current room's tiles stay; only relay its doors.
    JSR LayOutRoom              ; else lay out the next room.

@RestoreRoomId:
    PLA                         ; Restore current room ID.
    STA RoomId

@Exit:
    RTS

@GoingUp:
    ; CurOpenedDoors was just replaced with the entering doorway, so the current
    ; room's doors are laid out from PrevOpenedDoors.
    JSR LayOutDoorsPrev
    JMP @RestoreRoomId          ; Go restore current room ID and return.

; ----------------------------------------------------------------------------
; Lays out the doors of the room in RoomId using PrevOpenedDoors as the open doors (UW only).
; OUT:  CurOpenedDoors unchanged. Does nothing in OW.
; ----------------------------------------------------------------------------
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

; ----------------------------------------------------------------------------
; Mode 7 init step 0: sets the whirlwind's source room and undoes a flute pond secret.
; NOTE: Stays in this step while SecretColorCycle is nonzero: AnimatePond steps it down by 1 on
;       each frame with FrameCounter bit 2 set (4 frames of every 8).
; ----------------------------------------------------------------------------
InitMode7_Sub0:
    ; If the whirlwind is active (WhirlwindTeleportingState nonzero), then make
    ; WhirlwindPrevRoomId the current room, so the next room is found from it.
    LDA WhirlwindTeleportingState
    BEQ :+
    LDA WhirlwindPrevRoomId
    STA RoomId
:
    LDA SecretColorCycle
    BEQ L1433A_IncSubmode       ; No flute secret (pond), or it is fully undone: go to next submode.
    JMP AnimatePond             ; Go run the pond colour cycle one step back; stay in this submode.

; ----------------------------------------------------------------------------
; Mode 7 init step 2: sends tile row CurRow of the laid-out room to name table 2; one row a frame.
; OUT:  CurRow - 1; GameSubmode + 1 after row 0.
; NOTE: 22 frames (rows $15 down to 0).
; ----------------------------------------------------------------------------
InitMode7_Sub2:
    ; CurRow starts at the bottom ($15).
    JSR CopyRowToTileBuf
    LDA DynTileBuf

    ; Change the target nametable from 0 to 2.
    ;
    ; The record's address is $2100 + row * $20 (high byte $21-$23).
    ; Because there's only a play area and no status bar in
    ; nametable 2, add 7 to the high byte instead of 8.
    ; So, the top of the play area in NT 0 is at $2100.
    ; But the top of the play area in NT 2 is at $2800.
    ;
    ; This is for vertical scrolling, where the
    ; two scroll areas must be contiguous vertically.
    AND #$0F
    CLC
    ADC #$27
    STA DynTileBuf
    LDA ObjDir
    CMP #$04
    BCS :+                      ; If the scroll direction is horizontal (1 or 2),

    ; then you DO have to account for the status bar.
    ; So, add $100 (by incrementing high byte) to move
    ; everything down 64 pixels.
    ;
    ; Now the play areas will line up in NT 0 and NT 2 when
    ; InitMode7 changes mirroring to vertical at the end of this mode's init.
    INC DynTileBuf
:
    DEC CurRow
    BPL :+

; Advances GameSubmode (shared by init steps 0 and 2).
L1433A_IncSubmode:
    INC GameSubmode
:
    RTS

; ----------------------------------------------------------------------------
; Mode 7 init steps 3 and 4: cue the top, then the bottom half of PlayAreaAttrs for name table 2.
; OUT:  DynTileBuf holds the attribute record; GameSubmode + 1.
; NOTE: Destination $2BC0 for a vertical scroll, $2BD0 for a horizontal one; the bottom half goes
;       $18 bytes later.
; ----------------------------------------------------------------------------
InitMode7_Sub3And4_TransferPlayAreaAttrsToNT2:
    LDA #$08                    ; Test Link's direction for up (8).
    BIT ObjDir
    BNE @Vertical
    LSR                         ; Test Link's direction for down (4).
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
    BNE @TransferHalfPlayAreaAttrs    ; Always taken.

; ----------------------------------------------------------------------------
; Mode 7 init step 5: lays out the next room when going up, then decides on a fade to black.
; OUT:  FadeCycle = 0, or $40 with CandleState = 0 and GameSubmode + 1 when a lit room leads to a
;       dark one; otherwise falls to InitMode7_Finish.
; NOTE: Fade only when the next room is dark and the current room is not dark or was lit by a
;       candle (CandleState nonzero). No room is dark in OW.
; ----------------------------------------------------------------------------
InitMode7_Sub5:
    LDA #$00                    ; Reset fade cycle, in case we don't need to fade.
    STA FadeCycle
    LDA ObjDir
    CMP #$04
    BCS @Vertical               ; If direction is horizontal,

    ; then cue BlankRowTransferBuf: a row of blank tiles ($24) at $28E0,
    ; the row just above the play area copy at $2900 in NT 2.
    ; Whether this has a visible effect is (unverified).
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
    JSR IsDarkRoom_Bank5        ; A = $80 if dark (UW only), else 0.
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
    BEQ InitMode7_Finish        ; Always taken.

; ----------------------------------------------------------------------------
; Mode 7 init step 6: runs the fade to black until it is done, then finishes the init.
; NOTE: AnimateWorldFading makes 4 palette steps, 10 frames apart.
; ----------------------------------------------------------------------------
InitMode7_Sub6:
    JSR AnimateWorldFading
    BNE L143AD_Exit             ; If we're still fading, then return.

; Ends mode 7 init: RoomId = NextRoomId, sprite-0 split armed, mode update begins.
InitMode7_Finish:
    LDA NextRoomId              ; Set current room to next room.
    STA RoomId
    JSR WriteAndEnableSprite0
    JSR BeginUpdateMode

; Shared exit of the mode 7 init steps above.
L143AD_Exit:
    RTS

; DATA: sprite record 0 used for the sprite-0 hit: Y $27, tile $61, attributes $20 (behind the
; background), X $58.
Sprite0Descriptor:
    .BYTE $27, $61, $20, $58

; ----------------------------------------------------------------------------
; Sets IsSprite0CheckActive and writes Sprite0Descriptor to sprite record 0.
; NOTE: While IsSprite0CheckActive is set the frame handler splits the screen below the status
;       bar. Also called when entering caves and cellars and in the death sequence.
; ----------------------------------------------------------------------------
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

; ----------------------------------------------------------------------------
; Sets CurOpenedDoors to the single direction bit opposite Link's facing.
; OUT:  CurOpenedDoors = opposite of ObjDir (1 <-> 2, 4 <-> 8); uses [00].
; ----------------------------------------------------------------------------
SetEnteringDoorwayAsCurOpenedDoors:
    LDA ObjDir                  ; Swap each pair of direction bits: (dir >> 1) & 5 | (dir << 1) & $A.
    LSR
    AND #$05
    STA $00
    LDA ObjDir
    ASL
    AND #$0A
    ORA $00
    STA CurOpenedDoors          ; That is the door Link is entering the new room from.
    RTS

; TABLE: name table attribute byte using one palette in all four quadrants; index = palette
; selector 0-3 (bits 1-0 of room attrs A or B); value = attribute byte.
RoomPaletteSelectorToNTAttr:
    .BYTE $00, $55, $AA, $FF

; ----------------------------------------------------------------------------
; Fills PlayAreaAttrs (6 rows of 8 attribute bytes) with a room's outer and inner palettes.
; IN:   A = room id.
; OUT:  PlayAreaAttrs; uses [00]; A, X, Y changed.
; NOTE: Outer palette = LevelBlockAttrsA bits 1-0, everywhere. Inner palette = LevelBlockAttrsB
;       bits 1-0, in columns 1-6 of rows 1-3 and the top half of row 4 (tile rows 4-17 of the
;       play area, tile columns 4-27).
; ----------------------------------------------------------------------------
FillPlayAreaAttrs:
    TAY
    LDA LevelBlockAttrsA, Y     ; Room attributes A.
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
    AND #$07                    ; Skip left and right edges (columns 0 and 7).
    BEQ @NextLoopRow
    CMP #$07
    BEQ @NextLoopRow
    CPY #$21                    ; In the bottom inner attribute row, go combine inner and outer.
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
    ; affect the top half of the row (low nibble = the two upper quadrants).
    LDA RoomPaletteSelectorToNTAttr, X
    AND #$0F
    STA $00
    LDA PlayAreaAttrs, Y
    AND #$F0
    ORA $00
    STA PlayAreaAttrs, Y
    JMP @NextLoopRow            ; Go advance the offset and check if we're done.
