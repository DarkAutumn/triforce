; ============================================================================
; FILE:     bank5/mode7-scroll-init.asm  BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 723-1037
; SUBJECTS: Door, Mode7Scroll, Room, Scroll
; TOPIC:    Mode 7 (scroll to the next room) init submodes: find the next room, lay it out and send
;           its tiles and attributes to name table 2, fade to black when entering a dark room, then
;           arm the sprite-0 split. Also the play-area attribute fill used by other modes.
; SEE ALSO: bank7/init-mode-dispatch.asm (Mode7Scroll_Init calls this and sets the mirroring);
;           bank5/mode7-scroll-update.asm and bank5/scroll-world.asm (the scroll itself);
;           bank7/next-room-room-flags.asm and bank5/next-room-calc.asm (Room_CalcNext, mazes,
;           leaving the map); bank5/mode12-lay-out-room.asm (RoomLayout_LayOutRoom, Transfer_CopyRowToTileBuf);
;           bank5/lay-out-doors.asm (Door_LayOut); bank1/world-fading.asm (Palette_AnimateFade);
;           bank5/door-flags-dark-room.asm (Room_IsDark_Bank5).
; KEY FACTS:
;   - Submodes: 0 whirlwind room / undo pond secret, 1 next room and layout, 2 tile rows to NT 2
;     (22 frames), 3 and 4 attributes to NT 2, 5 dark-room test, 6 fade (Mode7Scroll_InitSubmodes_JumpTable).
;   - RoomId becomes NextRoomId only at the end of init, before the scroll is shown
;     (Mode7Scroll_Init_Finish).
;   - A next room id with bit 7 set aborts the scroll: Room_CalcNext has already switched to
;     mode 2 to load the overworld (Mode7Scroll_Init_Sub1).
;   - The doorway Link comes in through (opposite of his facing) starts out open in the new room
;     (Door_SetEnteringDoorwayOpened); the old room's open doors are kept in PrevOpenedDoors.
;   - Scrolling up draws the current room into NT 2 and lays out the next room only in submode 5;
;     other directions lay out the next room in submode 1 (Mode7Scroll_Init_Sub1, Mode7Scroll_Init_Sub5).
;   - Entering a dark room from a lit one (lit by nature or by candle) clears CandleState and fades
;     to black, 4 steps 10 frames apart; dark to dark does nothing (Mode7Scroll_Init_Sub5, Mode7Scroll_Init_Sub6).
;   - In OW no room is dark: Room_IsDark_Bank5 returns 0 when CurLevel is 0 (Mode7Scroll_Init_Sub5).
;   - Leaving a room waits while SecretColorCycle is nonzero: the flute pond's colour cycle is run
;     back to 0 first, one step on each frame with FrameCounter bit 2 set (Mode7Scroll_Init_Sub0).
;   - Room palettes: attrs A bits 1-0 = outer palette, attrs B bits 1-0 = inner palette
;     (Room_FillPlayAreaAttrs).
; CONTENTS: (generated; L = line in this file)
;   L0055 Mode7Scroll_InitSubmodes                      code  Runs the mode 7 init step selected by GameSubmode.
;   L0060 Mode7Scroll_InitSubmodes_JumpTable            data  TABLE: mode 7 init steps; index = GameSubmode (0-6); value = handler address.
;   L0077 Mode7Scroll_Init_Sub1                         code  Mode 7 init step 1: finds the next room and lays out the room that goes into name table 2.
;   L0139 Door_LayOutWithPrevOpened                     code  Lays out the doors of the room in RoomId using PrevOpenedDoors as the open doors (UW only).
;   L0157 Mode7Scroll_Init_Sub0                         code  Mode 7 init step 0: sets the whirlwind's source room and undoes a flute pond secret.
;   L0174 Mode7Scroll_Init_Sub2                         code  Mode 7 init step 2: sends tile row CurRow of the laid-out room to name table 2; one row a frame.
;   L0209 Mode7Scroll_Init_IncSubmode                   code  Advances GameSubmode (shared by init steps 0 and 2).
;   L0220 Mode7Scroll_Init_Sub3And4_TransferAttrsToNT2  code  Mode 7 init steps 3 and 4: cue the top, then the bottom half of PlayAreaAttrs for name table 2.
;   L0256 Mode7Scroll_Init_Sub5                         code  Mode 7 init step 5: lays out the next room when going up, then decides on a fade to black.
;   L0328 Mode7Scroll_Init_Sub6                         code  Mode 7 init step 6: runs the fade to black until it is done, then finishes the init.
;   L0333 Mode7Scroll_Init_Finish                       code  Ends mode 7 init: RoomId = NextRoomId, sprite-0 split armed, mode update begins.
;   L0340 Mode7Scroll_Init_Return                       code  Shared exit of the mode 7 init steps above.
;   L0345 Scroll_Sprite0Record                          data  DATA: sprite record 0 used for the sprite-0 hit: Y $27, tile $61, attributes $20 (behind the
;   L0353 Scroll_WriteAndEnableSprite0                  code  Sets IsSprite0CheckActive and writes Scroll_Sprite0Record to sprite record 0.
;   L0368 Door_SetEnteringDoorwayOpened                 code  Sets CurOpenedDoors to the single direction bit opposite Link's facing.
;   L0382 Room_PaletteSelectorAttrs                     data  TABLE: name table attribute byte using one palette in all four quadrants; index = palette
;   L0393 Room_FillPlayAreaAttrs                        code  Fills PlayAreaAttrs (6 rows of 8 attribute bytes) with a room's outer and inner palettes.
; ============================================================================

; ----------------------------------------------------------------------------
; Runs the mode 7 init step selected by GameSubmode.
; ----------------------------------------------------------------------------
Mode7Scroll_InitSubmodes:
    LDA GameSubmode
    JSR Util_TableJump

; TABLE: mode 7 init steps; index = GameSubmode (0-6); value = handler address.
Mode7Scroll_InitSubmodes_JumpTable:
    .ADDR Mode7Scroll_Init_Sub0                                   ; 0 whirlwind room id, undo pond secret
    .ADDR Mode7Scroll_Init_Sub1                                   ; 1 next room, attributes, layout
    .ADDR Mode7Scroll_Init_Sub2                                   ; 2 tile rows to NT 2, one per frame
    .ADDR Mode7Scroll_Init_Sub3And4_TransferAttrsToNT2    ; 3 top half of attributes to NT 2
    .ADDR Mode7Scroll_Init_Sub3And4_TransferAttrsToNT2    ; 4 bottom half of attributes to NT 2
    .ADDR Mode7Scroll_Init_Sub5                                   ; 5 lay out next room (up), dark-room test
    .ADDR Mode7Scroll_Init_Sub6                                   ; 6 fade to black, then finish

; ----------------------------------------------------------------------------
; Mode 7 init step 1: finds the next room and lays out the room that goes into name table 2.
; OUT:  PrevOpenedDoors = old CurOpenedDoors; CurOpenedDoors = the doorway being entered;
;       NextRoomId; PlayAreaAttrs filled; CurRow = $15; tile map laid out; GameSubmode + 1.
; NOTE: If NextRoomId has bit 7 set (no such room), nothing more is done here: Room_CalcNext
;       has already switched to mode 2 to load the overworld. In OW Room_CalcNext also applies
;       the mazes.
; ----------------------------------------------------------------------------
Mode7Scroll_Init_Sub1:
    JSR Sprite_DrawBetweenRooms
    JSR Link_EndMoveAndAnimateBetweenRooms
    LDA CurOpenedDoors          ; Keep the open doors of the room being left.
    STA PrevOpenedDoors

    ; Mark the doorway Link is entering the new room from as open.
    ; If it is not a true door, then Door_LayOut called from
    ; RoomLayout_LayOutRoom will clear it.
    JSR Door_SetEnteringDoorwayOpened
    DEC PrevRow                 ; PrevRow becomes $FF if it was 0 (as after the RAM clear at reset), $FE
                                ; after an earlier scroll (which leaves $FF): never a valid row. With 0
                                ; left in it, Transfer_CopyColumnOrRowToTileBuf takes row 0, the first
                                ; row of a scroll down, for a row already copied and skips it. (trace)
                                ; PrevRow is 0 when play first starts after power on; with PrevRow
                                ; set to 1 before a scroll down, the row record for $2100 was not sent.
    INC GameSubmode
    JSR Room_CalcNext
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
    JSR Room_FillPlayAreaAttrs       ; A = the room to draw (current room if going up, else next room).
    LDA #$15                    ; Submode 2 starts at the bottom row.
    STA CurRow
    LDY ObjDir
    CPY #$08
    BEQ @GoingUp                ; If going up, the current room's tiles stay; only relay its doors.
    JSR RoomLayout_LayOutRoom              ; else lay out the next room.

@RestoreRoomId:
    PLA                         ; Restore current room ID.
    STA RoomId

@Exit:
    RTS

@GoingUp:
    ; CurOpenedDoors was just replaced with the entering doorway, so the current
    ; room's doors are laid out from PrevOpenedDoors.
    JSR Door_LayOutWithPrevOpened
    JMP @RestoreRoomId          ; Go restore current room ID and return.

; ----------------------------------------------------------------------------
; Lays out the doors of the room in RoomId using PrevOpenedDoors as the open doors (UW only).
; OUT:  CurOpenedDoors unchanged. Does nothing in OW.
; ----------------------------------------------------------------------------
Door_LayOutWithPrevOpened:
    LDA CurLevel
    BEQ :+                      ; If in OW, then return.
    LDA CurOpenedDoors
    PHA
    LDA PrevOpenedDoors
    STA CurOpenedDoors
    JSR Door_LayOut
    PLA
    STA CurOpenedDoors
:
    RTS

; ----------------------------------------------------------------------------
; Mode 7 init step 0: sets the whirlwind's source room and undoes a flute pond secret.
; NOTE: Stays in this step while SecretColorCycle is nonzero: FluteSecret_ReversePondCycle steps it down by 1 on
;       each frame with FrameCounter bit 2 set (4 frames of every 8).
; ----------------------------------------------------------------------------
Mode7Scroll_Init_Sub0:
    ; If the whirlwind is active (WhirlwindTeleportingState nonzero), then make
    ; WhirlwindPrevRoomId the current room, so the next room is found from it.
    LDA WhirlwindTeleportingState
    BEQ :+
    LDA WhirlwindPrevRoomId
    STA RoomId
:
    LDA SecretColorCycle
    BEQ Mode7Scroll_Init_IncSubmode       ; No flute secret (pond), or it is fully undone: go to next submode.
    JMP FluteSecret_ReversePondCycle             ; Go run the pond colour cycle one step back; stay in this submode.

; ----------------------------------------------------------------------------
; Mode 7 init step 2: sends tile row CurRow of the laid-out room to name table 2; one row a frame.
; OUT:  CurRow - 1; GameSubmode + 1 after row 0.
; NOTE: 22 frames (rows $15 down to 0).
; ----------------------------------------------------------------------------
Mode7Scroll_Init_Sub2:
    ; CurRow starts at the bottom ($15).
    JSR Transfer_CopyRowToTileBuf
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
    ; Mode7Scroll_Init changes mirroring to vertical at the end of this mode's init.
    INC DynTileBuf
:
    DEC CurRow
    BPL :+

; Advances GameSubmode (shared by init steps 0 and 2).
Mode7Scroll_Init_IncSubmode:
    INC GameSubmode
:
    RTS

; ----------------------------------------------------------------------------
; Mode 7 init steps 3 and 4: cue the top, then the bottom half of PlayAreaAttrs for name table 2.
; OUT:  DynTileBuf holds the attribute record; GameSubmode + 1.
; NOTE: Destination $2BC0 for a vertical scroll, $2BD0 for a horizontal one; the bottom half goes
;       $18 bytes later.
; ----------------------------------------------------------------------------
Mode7Scroll_Init_Sub3And4_TransferAttrsToNT2:
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
    JSR Transfer_GetPlayAreaAttrsBottomHalf    ; then refer to bottom half of NT attributes.
:
    JMP Transfer_CuePlayAreaAttrsHalfNT2AndAdvanceSubmode

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
;       dark one; otherwise falls to Mode7Scroll_Init_Finish.
; NOTE: Fade only when the next room is dark and the current room is not dark or was lit by a
;       candle (CandleState nonzero). No room is dark in OW.
; ----------------------------------------------------------------------------
Mode7Scroll_Init_Sub5:
    LDA #$00                    ; Reset fade cycle, in case we don't need to fade.
    STA FadeCycle
    LDA ObjDir
    CMP #$04
    BCS @Vertical               ; If direction is horizontal,

    ; then cue Transfer_BlankRow7NT2Buf: a row of blank tiles ($24) at $28E0,
    ; the row just above the play area copy at $2900 in NT 2. During a horizontal scroll the
    ; bottom scanline of that row (line 63) shows in the part of the picture that comes from the
    ; second name table, so the blank row hides what an earlier vertical scroll left there.
    ; It reaches that row only when a fade follows (step 6): the record is then sent while the
    ; mirroring is still horizontal. Without a fade the init finishes in this frame, the mirroring
    ; is vertical when the record is sent on the next frame, and $28E0 is then the same memory as
    ; $20E0, the bottom row of the status bar, which is blank already.
    ; (trace) Scrolls without a fade, overworld and levels 1 and 4: leaving the transfer out changed
    ; no pixel; a visible tile in its place showed on lines 56-63 on all frames but the first, as one written
    ; to $20E0 did, while one written to the other name table's row showed on line 63 only (not
    ; on every frame). Level 1, a scroll down from $63 to $73 and then left or right: line 63
    ; showed tiles that blanking that row by hand removed.
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
    JSR RoomLayout_LayOutRoom
    PLA                         ; Restore current room ID.
    STA RoomId

@CheckDark:
    ; Scrolling in any direction.
    LDY NextRoomId
    JSR Room_IsDark_Bank5        ; A = $80 if dark (UW only), else 0.
    BEQ Mode7Scroll_Init_Finish        ; If the next room is not dark, then go finish up and start updating.

    ; The next room is dark.
    LDY RoomId
    JSR Room_IsDark_Bank5
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
    BEQ Mode7Scroll_Init_Finish        ; Always taken.

; ----------------------------------------------------------------------------
; Mode 7 init step 6: runs the fade to black until it is done, then finishes the init.
; NOTE: Palette_AnimateFade makes 4 palette steps, 10 frames apart.
; ----------------------------------------------------------------------------
Mode7Scroll_Init_Sub6:
    JSR Palette_AnimateFade
    BNE Mode7Scroll_Init_Return             ; If we're still fading, then return.

; Ends mode 7 init: RoomId = NextRoomId, sprite-0 split armed, mode update begins.
Mode7Scroll_Init_Finish:
    LDA NextRoomId              ; Set current room to next room.
    STA RoomId
    JSR Scroll_WriteAndEnableSprite0
    JSR Mode_BeginUpdate

; Shared exit of the mode 7 init steps above.
Mode7Scroll_Init_Return:
    RTS

; DATA: sprite record 0 used for the sprite-0 hit: Y $27, tile $61, attributes $20 (behind the
; background), X $58.
Scroll_Sprite0Record:
    .BYTE $27, $61, $20, $58

; ----------------------------------------------------------------------------
; Sets IsSprite0CheckActive and writes Scroll_Sprite0Record to sprite record 0.
; NOTE: While IsSprite0CheckActive is set the frame handler splits the screen below the status
;       bar. Also called when entering caves and cellars and in the death sequence.
; ----------------------------------------------------------------------------
Scroll_WriteAndEnableSprite0:
    LDA #$01
    STA IsSprite0CheckActive
    LDY #$03
:
    LDA Scroll_Sprite0Record, Y
    STA Sprites, Y
    DEY
    BPL :-
    RTS

; ----------------------------------------------------------------------------
; Sets CurOpenedDoors to the single direction bit opposite Link's facing.
; OUT:  CurOpenedDoors = opposite of ObjDir (1 <-> 2, 4 <-> 8); uses [00].
; ----------------------------------------------------------------------------
Door_SetEnteringDoorwayOpened:
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
Room_PaletteSelectorAttrs:
    .BYTE $00, $55, $AA, $FF

; ----------------------------------------------------------------------------
; Fills PlayAreaAttrs (6 rows of 8 attribute bytes) with a room's outer and inner palettes.
; IN:   A = room id.
; OUT:  PlayAreaAttrs; uses [00]; A, X, Y changed.
; NOTE: Outer palette = LevelBlockAttrsA bits 1-0, everywhere. Inner palette = LevelBlockAttrsB
;       bits 1-0, in columns 1-6 of rows 1-3 and the top half of row 4 (tile rows 4-17 of the
;       play area, tile columns 4-27).
; ----------------------------------------------------------------------------
Room_FillPlayAreaAttrs:
    TAY
    LDA LevelBlockAttrsA, Y     ; Room attributes A.
    AND #$03                    ; Get the outer palette selector from the byte.
    TAX
    LDA Room_PaletteSelectorAttrs, X    ; Get the nametable attributes for the palette selector.
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
    LDA Room_PaletteSelectorAttrs, X
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
    LDA Room_PaletteSelectorAttrs, X
    AND #$0F
    STA $00
    LDA PlayAreaAttrs, Y
    AND #$F0
    ORA $00
    STA PlayAreaAttrs, Y
    JMP @NextLoopRow            ; Go advance the offset and check if we're done.
