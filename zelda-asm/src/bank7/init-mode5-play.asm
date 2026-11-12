; ============================================================================
; FILE:     bank7/init-mode5-play.asm    BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 1491-1647
; SUBJECTS: Mode5Play, Palette, Room
; TOPIC:    Mode 5 (play) init: picks the palette for sprite palette row 7 (boss, tile object or level),
;           then the tasks run whenever a room becomes active: object bounds, visited flag, room
;           objects, room history, boss roar, whirlwind drop-off.
; SEE ALSO: bank7/init-mode-dispatch.asm (Mode_Init, Sprite_DrawBetweenRooms, RoomHistory_Clear);
;           bank5/create-room-objects.asm (Room_SetUpItemAndPushBlock); bank5/room-history-ow.asm and
;           bank5/room-history-uw.asm (how RoomHistory changes a room's monster count);
;           bank5/boss-sound-door-face.asm (Sample_SetBossRoarUW); bank1/whirlwind.asm
;           (Whirlwind_InitDropOffAndBeginUpdate); bank6/transfer-buf-dispatch.asm (Transfer_BufAddrs: what
;           each TileBufSelector value sends); bank1/status-bar-rupees.asm (Mode_BeginUpdate).
; KEY FACTS:
;   - Mode 5 init takes one frame; it ends in Mode_BeginUpdate, so play updates start the next frame.
;     Mode5Play_Init, Room_RunEntryTasks
;   - RoomHistory is a ring of six room ids. A room is added only if it is not already in it; the write
;     index CurRoomHistoryIndex wraps from 5 to 0. Room_RunEntryTasks
;   - The room tasks run at the end of the inits of modes 4, 5, 6, 9, $B and $C, so Room_SetUpItemAndPushBlock
;     runs more than once per room visit. Room_RunEntryTasks
;   - A dungeon room is marked visited (room flag $20) when it is entered (inits of modes 4, 5 and 9),
;     not when it is left. Room_RunEntryTasks_MarkVisited
;   - Dungeon: sprite palette row 7 depends on the object type in slot 1: Aquamentus, Ganon,
;     Digdogger, Dodongo and Gleeok types $43-$45 have their own row; anything else (including Gleeok
;     type $42) gets row 7 of the level palettes. Palette_BossObjTypes, Mode5Play_Init
;   - Overworld: row 7 depends on the tile object type in slot $0B: gravestone $65 ghost palette, armos
;     $66 and rock $62 by bit 0 of the room's inner palette, every other value the red armos palette.
;     Mode5Play_Init
;   - Overworld room $0F plays the "secret found" tune (Tune1Request $04) every time mode 5 starts
;     there with UndergroundExitType = 0 (Link is not coming out of underground). Mode5Play_Init
;   - Only mode 5 starts the dungeon boss roar check (dungeons) or the whirlwind drop-off (overworld).
;     Room_RunEntryTasks
; CONTENTS: (generated; L = line in this file)
;   L0044 Palette_BossTransferBufSelectors  data  TABLE: palette transfer buffer for a boss room; index = same as Palette_BossObjTypes;
;   L0049 Palette_BossObjTypes              data  TABLE: object types that get their own sprite palette row 7; index 0-8; value = object type.
;   L0060 Mode5Play_Init                    code  Mode 5 init: draws Link, cues the palette for sprite palette row 7, then runs the room tasks.
;   L0147 Room_RunEntryTasks_SetBounds      code  Room tasks, entry 1 of 3: sets the object bounds for the overworld or a dungeon (falls through).
;   L0156 Room_RunEntryTasks_MarkVisited    code  Room tasks, entry 2 of 3: in a dungeon marks the room visited and blanks the priority sprites.
;   L0169 Room_RunEntryTasks                code  Room tasks, entry 3 of 3: creates room objects, records the room in RoomHistory, begins updating.
; ============================================================================

; TABLE: palette transfer buffer for a boss room; index = same as Palette_BossObjTypes;
; value = TileBufSelector (offset into Transfer_BufAddrs).
Palette_BossTransferBufSelectors:
    .BYTE $08, $36, $0A, $0A, $0A, $0A, $7C, $7C    ; $08 Aquamentus row 7, $36 Ganon row 7,
    .BYTE $7C                                       ; $0A orange boss row 7, $7C Gleeok row 7

; TABLE: object types that get their own sprite palette row 7; index 0-8; value = object type.
Palette_BossObjTypes:
    .BYTE $3D, $3E, $38, $39, $32, $31, $43, $44    ; Aquamentus, Ganon, Digdogger x2, Dodongo x2,
    .BYTE $45                                       ; Gleeok $43, $44, $45 ($42 is not listed)

; ----------------------------------------------------------------------------
; Mode 5 init: draws Link, cues the palette for sprite palette row 7, then runs the room tasks.
; NOTE: Dungeon: the palette is chosen from the object type in slot 1 (Palette_BossObjTypes),
;       else the level's own row 7. Overworld: from the tile object type in slot $0B. Overworld room
;       $0F plays the "secret found" tune when entered with UndergroundExitType = 0.
;       Falls through into the room tasks below and ends in Mode_BeginUpdate.
; ----------------------------------------------------------------------------
Mode5Play_Init:
    JSR Sprite_DrawBetweenRooms
    JSR Link_EndMoveAndAnimate
    LDA CurLevel
    BEQ @InOW

    ; In UW.
    ;
    ; Certain bosses need their own palette row. If slot 1 holds one of them,
    ; then get the matching transfer buffer selector.
    LDY #$08
    LDA ObjType+1               ; Object type in slot 1.

@FindSpecialBoss:
    CMP Palette_BossObjTypes, Y
    BNE @NextSpecialBoss        ; If the object type doesn't match, go check the next one.
    LDX Palette_BossTransferBufSelectors, Y
    BNE @SelectTransferBufAndFinishInitPlay    ; Always taken: go cue this palette row (selector in X).

@NextSpecialBoss:
    DEY
    BPL @FindSpecialBoss
    BMI @UseLevelPalette        ; No match (always taken): go copy and transfer level palette row 7.

@InOW:
    ; In OW.
    ;
    ; If the room is $0F and Link walked in instead of coming out of
    ; underground, then play the "secret found" tune.
    LDA RoomId
    CMP #$0F
    BNE @ChooseTileObjPalette
    LDA UndergroundExitType
    BNE @ChooseTileObjPalette
    LDA #$04                    ; "Secret found" tune.
    STA Tune1Request

@ChooseTileObjPalette:
    ; Choose a palette transfer buf by the tile object type in slot $0B:
    ; gravestone $65, armos $66, rock $62, anything else.
    LDX #$20                    ; Ghost palette row (Transfer_GhostPaletteRow7Buf).
    LDA ObjType+11              ; Tile object type (slot $0B).
    CMP #$65
    BEQ @SelectTransferBufAndFinishInitPlay    ; Gravestone ($65): cue transfer buf $20.
    CMP #$66
    BEQ @UseChosenOrGreenPalette     ; Armos ($66): $20 or $22, by the room's inner palette.
    CMP #$62
    BNE @UseRedArmosPalette     ; Neither of those nor rock ($62): use the red armos row.

    ; It's a rock tile object.
    ; Choose the palette by the room's inner palette attribute, to match the background.
    LDX #$24                    ; Brown palette row (Transfer_BrownBgPaletteRow7Buf).

@UseChosenOrGreenPalette:
    LDY RoomId
    LDA LevelBlockAttrsB, Y
    AND #$01                    ; Bit 0 of the inner palette (bits 1-0 of attribute B).
    BNE @SelectTransferBufAndFinishInitPlay    ; Odd: cue the buf chosen above ($20 armos, $24 rock).
    LDX #$22                    ; Even: green palette row (Transfer_GreenBgPaletteRow7Buf).
    BNE @SelectTransferBufAndFinishInitPlay    ; Always taken.

@UseRedArmosPalette:
    ; Any other tile object type, or none (0). Use the red armos palette row
    ; (Transfer_DefaultOWPaletteRow7Buf).
    LDX #$7A
    BNE @SelectTransferBufAndFinishInitPlay    ; Always taken.

@UseLevelPalette:
    ; Copy the 4 colours of row 7 of the level palettes into the row 7 transfer buf,
    ; and cue it (selector 6, Transfer_LevelPaletteRow7Buf).
    LDY #$03

@PatchColors:
    LDA LevelInfo_PalettesTransferBuf+31, Y
    STA Transfer_LevelPaletteRow7Buf+3, Y
    DEY
    BPL @PatchColors
    LDX #$06

@SelectTransferBufAndFinishInitPlay:
    STX TileBufSelector

; ----------------------------------------------------------------------------
; Room tasks, entry 1 of 3: sets the object bounds for the overworld or a dungeon (falls through).
; NOTE: Reached from mode 5 init (fall-through) and from ModeBCave_Init_Sub7_EnterCave (submode 7 of the
;       inits of modes $B and $C).
; ----------------------------------------------------------------------------
Room_RunEntryTasks_SetBounds:
    LDA #$05
    JSR Mmc1_SwitchBank
    JSR Room_SetUpObjBounds

; ----------------------------------------------------------------------------
; Room tasks, entry 2 of 3: in a dungeon marks the room visited and blanks the priority sprites.
; NOTE: Reached by fall-through and from Room_EnterCommon (modes 4, 9, $B, $C).
; ----------------------------------------------------------------------------
Room_RunEntryTasks_MarkVisited:
    LDA CurLevel
    BEQ Room_RunEntryTasks    ; Overworld: skip both.
    JSR RoomFlags_MarkVisited         ; Sets room flag $20 for RoomId.
    JSR Sprite_WriteBlankPrioritySprites

; ----------------------------------------------------------------------------
; Room tasks, entry 3 of 3: creates room objects, records the room in RoomHistory, begins updating.
; NOTE: Reached by fall-through, from Mode6Leave_Init (leaving a room) and from ModeBCave_Init_Sub8_WalkIn. The
;       room id is stored at RoomHistory[CurRoomHistoryIndex] only if none of the six entries holds
;       it; the index then advances and wraps from 5 to 0. In mode 5 only: a dungeon checks its boss
;       roar, the overworld checks for a whirlwind drop-off. All paths end in Mode_BeginUpdate.
; ----------------------------------------------------------------------------
Room_RunEntryTasks:
    LDA #$05
    JSR Mmc1_SwitchBank
    JSR Room_SetUpItemAndPushBlock

    ; Count the entries of room history that hold the current room (Y).
    LDY #$00
    LDX #$05
    LDA RoomId

@LoopHistory:
    CMP RoomHistory, X
    BNE :+
    INY
:
    DEX
    BPL @LoopHistory

    ; If it's not found, then store it at the cycling history index.
    CPY #$00
    BNE @RunTasksMode5
    LDX CurRoomHistoryIndex
    STA RoomHistory, X
    INC CurRoomHistoryIndex
    LDA CurRoomHistoryIndex
    CMP #$06
    BCC @RunTasksMode5
    LDA #$00                    ; Wrap the index from 6 to 0.
    STA CurRoomHistoryIndex

@RunTasksMode5:
    LDA GameMode
    CMP #$05
    BNE @BeginUpdate            ; If not in mode 5, go start updating.

    ; Run the tasks that apply only to mode 5.
    LDA CurLevel
    BEQ @CheckWhirlwind
    LDA #$05
    JSR Mmc1_SwitchBank
    JSR Sample_SetBossRoarUW  ; Dungeon: start or silence the boss roar for this room.

@BeginUpdate:
    JMP Mode_BeginUpdate

@CheckWhirlwind:
    ; Overworld: if a whirlwind is carrying Link, set it up in this room; then begin updating.
    LDA #$01
    JSR Mmc1_SwitchBank
    JMP Whirlwind_InitDropOffAndBeginUpdate

; DATA: filler ($FF), 6 bytes.
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF
