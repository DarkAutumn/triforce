; ============================================================================
; FILE:     bank7/init-mode-dispatch.asm BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 1257-1489
; SUBJECTS: Link, Mode, Mode0Demo, Mode13WinGame, Mode1Menu, Mode2Load, Mode3Unfurl, Mode7Scroll,
;          ModeERegister, Palette, RoomHistory, Sprite
; TOPIC:    The init half of the mode machine: one-time game setup, the per-mode init dispatch, the mode
;           inits that live in the fixed bank (0, 1, 2, 3, 7, $E/$F, $13), room history clearing, and
;           helpers run on every room change (sprites between rooms, Link's state and clock reset).
; SEE ALSO: bank7/run-game-nmi.asm (Nmi_Isr calls Mode_InitGameOrMode or Mode_Update once per frame);
;           bank7/update-mode-dispatch.asm (Mode_Update, Mode_GoToNext, Mode_End); bank7/init-mode5-play.asm
;           (Mode5Play_Init, room history recording); bank5/mode3-mode6-init.asm (Mode3Unfurl_Init_Sub2_FillAttrsCuePalettes..Sub8,
;           Mode6Leave_Init); bank5/enter-room.asm (Mode4Enter_Init); bank5/mode7-scroll-init.asm (Mode7Scroll_InitSubmodes);
;           bank6/level-load-mode2.asm (Mode2Load_Init_Submodes); bank1/take-item.asm (ring colour patch).
; KEY FACTS:
;   - Each frame Nmi_Isr runs the mode's init routine while IsUpdatingMode = 0 and its update routine
;     otherwise; an init may take many frames, stepping GameSubmode. Mode_InitGameOrMode, Mode_Init
;   - The very first frame after reset only copies the common code and data to RAM and writes the save
;     RAM markers ($5A at SaveRamBegin, $A5 at SaveRamEnd); no mode runs. Mode_InitGameOrMode
;   - Modes $E and $F share one init routine. Mode_Init_JumpTable
;   - Loading a level (mode 2, submode 0) zeroes the room history and all 128 LevelKillCounts (the
;     kills banked per dungeon room). Overworld kill counts are in the room flags and stay. Mode2Load_Init
;   - Room history is also cleared at the start of mode 3 (unfurl) and of mode 8 (continue menu).
;     Mode3Unfurl_Init_Sub0, RoomHistory_Clear
;   - Mode 3 starts Link in LevelInfo_StartRoomId, except in the overworld when LevelEntranceRoomId is
;     not $FF: then he starts in that room. Whenever the start room equals LevelEntranceRoomId (the
;     test is also made in a dungeon) the variable is set back to $FF. Mode3Unfurl_Init_Sub1
;   - Every room entry (modes 4, 9, $B, $C init), every room exit (mode 6 init) and a choice in the
;     continue menu set Link's state to 0 and turn the clock off (InvClock = 0). Link_ResetStateAndClock
;   - Room scrolling uses vertical mirroring when Link faces left or right and horizontal mirroring
;     when he faces up or down. Mode7Scroll_Init
;   - Link's tunic colour comes from the save slot's row of the menu palettes and is patched into the
;     level palettes before they are transferred. Palette_PatchLinkColorAndCueLevelPalettes
; CONTENTS: (generated; L = line in this file)
;   L0060 Mode_InitGameOrMode                        code  Per-frame entry while a mode is initializing: sets the game up once, then runs the mode's init.
;   L0080 Mode_Init                                  code  Switches to bank 5 and jumps to the init routine of the current game mode.
;   L0087 Mode_Init_JumpTable                        data  TABLE: mode init routines; index = GameMode ($00-$13); value = routine address.
;   L0114 Mode0Demo_InitPatternsOrDemo               code  Mode 0 init: loads the common and the demo pattern tables if missing, else runs the demo init tasks.
;   L0136 Mode1Menu_Init                             code  Mode 1 init: trampoline to the file menu init in bank 2.
;   L0147 Mode2Load_Init                             code  Mode 2 init: on submode 0 forgets room history and kill counts and loads the level's patterns.
;   L0181 Mode7Scroll_Init                           code  Mode 7 init: runs the scroll init submode, then sets name table mirroring for the scroll direction.
;   L0214 ModeERegister_Init                         code  Mode $E and $F init: trampoline to the register/eliminate init in bank 2.
;   L0222 Mode13WinGame_Init                         code  Mode $13 init: sets horizontal mirroring, then runs the ending init in bank 2.
;   L0234 Mode3Unfurl_Init                           code  Mode 3 init: turns video off and runs the unfurl init step selected by GameSubmode (0-8).
;   L0242 Mode3Unfurl_Init_JumpTable                 data  TABLE: mode 3 init steps; index = GameSubmode; value = routine address.
;   L0258 Mode3Unfurl_Init_Sub0                      code  Mode 3 init step 0: blanks the screen, then clears the room history (falls through).
;   L0271 RoomHistory_Clear                          code  Zeroes the six RoomHistory entries and IsSwordShotForced.
;   L0284 Palette_SaveSlotRowOffsets                 data  TABLE: offset of a save slot's palette row within the menu palettes, counted from row 4;
;   L0294 Mode3Unfurl_Init_Sub1                      code  Mode 3 init step 1: chooses the room Link starts in, then cues the level palettes (falls through).
;   L0319 Palette_PatchLinkColorAndCueLevelPalettes  code  Copies Link's tunic colour into the level palettes, cues their transfer and advances the submode.
;   L0339 Sprite_DrawBetweenRooms                    code  Redraws the sprites that stay visible between rooms: map markers, Link, status bar items.
;   L0354 Link_ResetStateAndClock                    code  Makes Link idle (state 0) and turns the clock off.
; ============================================================================

; ----------------------------------------------------------------------------
; Per-frame entry while a mode is initializing: sets the game up once, then runs the mode's init.
; NOTE: Called by Nmi_Isr when IsUpdatingMode = 0. On the first call after reset (IsGameInitialized = 0)
;       it copies the common code (bank 1) and common data (bank 6) to RAM, marks save RAM initialized
;       and returns without running a mode init; later calls fall into Mode_Init.
; ----------------------------------------------------------------------------
Mode_InitGameOrMode:
    LDA IsGameInitialized
    BNE Mode_Init
    LDA #$01
    JSR Mmc1_SwitchBank
    JSR Bank_CopyCommonCodeToRam
    LDA #$06
    JSR Mmc1_SwitchBank
    JSR Bank_CopyCommonDataToRam
    LDA #$5A                    ; Mark save RAM initialized, so it can be recognised after a reset.
    STA SaveRamBegin
    LDA #$A5
    STA SaveRamEnd
    INC IsGameInitialized
    RTS

; ----------------------------------------------------------------------------
; Switches to bank 5 and jumps to the init routine of the current game mode.
; NOTE: Runs once per frame until the init routine calls Mode_BeginUpdate (IsUpdatingMode becomes 1).
; ----------------------------------------------------------------------------
Mode_Init:
    LDA #$05
    JSR Mmc1_SwitchBank
    LDA GameMode
    JSR Util_TableJump

; TABLE: mode init routines; index = GameMode ($00-$13); value = routine address.
Mode_Init_JumpTable:
    .ADDR Mode0Demo_InitPatternsOrDemo             ; $00 demo
    .ADDR Mode1Menu_Init             ; $01 file menu
    .ADDR Mode2Load_Init             ; $02 load level
    .ADDR Mode3Unfurl_Init             ; $03 unfurl
    .ADDR Mode4Enter_Init             ; $04 enter room
    .ADDR Mode5Play_Init         ; $05 play
    .ADDR Mode6Leave_Init             ; $06 leave room
    .ADDR Mode7Scroll_Init             ; $07 scroll
    .ADDR Mode8Continue_Init             ; $08 continue menu
    .ADDR Mode9Cellar_Init             ; $09 enter cellar
    .ADDR ModeALeaveCellar_Init             ; $0A leave cellar
    .ADDR ModeBCave_Init             ; $0B enter cave
    .ADDR ModeCShortcut_Init             ; $0C shortcut stairs
    .ADDR ModeDSave_Init             ; $0D save
    .ADDR ModeERegister_Init         ; $0E register
    .ADDR ModeERegister_Init         ; $0F eliminate (same routine as $0E)
    .ADDR Mode10Stairs_Init            ; $10 stairs
    .ADDR Mode11Death_Init            ; $11 death
    .ADDR Mode12EndLevel_Init            ; $12 end level
    .ADDR Mode13WinGame_Init            ; $13 win game

; ----------------------------------------------------------------------------
; Mode 0 init: loads the common and the demo pattern tables if missing, else runs the demo init tasks.
; NOTE: One of the three jobs per frame: common patterns (bank 2) until HasCommonPatterns = $5A, demo
;       patterns (bank 1) until HasDemoPatterns = $A5, then Mode0Demo_Init (bank 2).
; ----------------------------------------------------------------------------
Mode0Demo_InitPatternsOrDemo:
    LDA HasCommonPatterns
    CMP #$5A
    BEQ :+                      ; Common patterns already transferred.
    LDA #$02
    JSR Mmc1_SwitchBank
    JMP Chr_TransferCommonPatterns
:
    LDA HasDemoPatterns
    CMP #$A5
    BEQ :+                      ; Demo patterns already transferred.
    LDA #$01
    JSR Mmc1_SwitchBank
    JMP Chr_TransferDemoPatterns
:
    LDA #$02
    JSR Mmc1_SwitchBank
    JMP Mode0Demo_Init

; ----------------------------------------------------------------------------
; Mode 1 init: trampoline to the file menu init in bank 2.
; ----------------------------------------------------------------------------
Mode1Menu_Init:
    LDA #$02
    JSR Mmc1_SwitchBank
    JMP Mode1Menu_Init_Full

; ----------------------------------------------------------------------------
; Mode 2 init: on submode 0 forgets room history and kill counts and loads the level's patterns.
; NOTE: Video is turned off every frame of this init. Submode 0 also copies the common code to RAM
;       again. Every frame then continues in Mode2Load_Init_Submodes (bank 6), which copies the level data.
;       Zeroing LevelKillCounts means no room of the newly loaded level remembers banked kills.
; ----------------------------------------------------------------------------
Mode2Load_Init:
    JSR Ppu_TurnOffAllVideo
    LDA GameSubmode
    BNE @InitSubmodes

    ; Submode 0.
    ; Clear the room history, transfer level pattern blocks and copy common code.
    JSR RoomHistory_Clear        ; Returns A = 0.

    ; Clear the level kill counts of all $80 rooms.
    LDY #$7F

@ClearCounts:
    STA LevelKillCounts, Y
    DEY
    BPL @ClearCounts
    LDA #$03
    JSR Mmc1_SwitchBank
    JSR Chr_TransferLevelPatterns
    LDA #$01
    JSR Mmc1_SwitchBank
    JSR Bank_CopyCommonCodeToRam

@InitSubmodes:
    LDA #$06
    JSR Mmc1_SwitchBank
    JMP Mode2Load_Init_Submodes

; ----------------------------------------------------------------------------
; Mode 7 init: runs the scroll init submode, then sets name table mirroring for the scroll direction.
; NOTE: The mirroring is set once, on the first frame on which the sprite-0 check is active:
;       vertical mirroring (MMC1 control $0E) when Link faces right or left, horizontal ($0F) when he
;       faces down or up. Mode7Scroll_Update restores $0F when the scroll ends.
; ----------------------------------------------------------------------------
Mode7Scroll_Init:
    LDA #$05
    JSR Mmc1_SwitchBank
    JSR Mode7Scroll_InitSubmodes
    LDA IsSprite0CheckActive
    BEQ @Exit                   ; If not checking sprite 0, return.

    ; The last init submode turned the sprite-0 check on.
    ; So, set the appropriate mirroring for scrolling during mode update.
    LDA IsScrollMirroringSet
    BNE @Exit                   ; Already set for this scroll.
    INC IsScrollMirroringSet

    ; If Link is facing horizontally, then enable vertical mirroring;
    ; else horizontal mirroring.
    LDA ObjDir
    CMP #$04
    BCC @SetVertical            ; Direction 1 (right) or 2 (left).
    LDA #$0F                    ; Horizontal mirroring (direction 4 down or 8 up).
    BNE @SetMirroring           ; Always taken.

@SetVertical:
    LDA #$0E                    ; Vertical mirroring.

@SetMirroring:
    JSR Mmc1_SetControl

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Mode $E and $F init: trampoline to the register/eliminate init in bank 2.
; ----------------------------------------------------------------------------
ModeERegister_Init:
    LDA #$02
    JSR Mmc1_SwitchBank
    JMP ModeERegister_Init_Full

; ----------------------------------------------------------------------------
; Mode $13 init: sets horizontal mirroring, then runs the ending init in bank 2.
; ----------------------------------------------------------------------------
Mode13WinGame_Init:
    ; Make sure horizontal mirroring is on.
    LDA #$0F
    JSR Mmc1_SetControl
    LDA #$02
    JSR Mmc1_SwitchBank
    JMP Mode13WinGame_Init_Full

; ----------------------------------------------------------------------------
; Mode 3 init: turns video off and runs the unfurl init step selected by GameSubmode (0-8).
; NOTE: One submode per frame. Bank 5 is switched in (submodes 2-8 live there).
; ----------------------------------------------------------------------------
Mode3Unfurl_Init:
    LDA #$05
    JSR Mmc1_SwitchBank
    JSR Ppu_TurnOffAllVideo
    LDA GameSubmode
    JSR Util_TableJump

; TABLE: mode 3 init steps; index = GameSubmode; value = routine address.
Mode3Unfurl_Init_JumpTable:
    .ADDR Mode3Unfurl_Init_Sub0                            ; 0 clear screen and room history
    .ADDR Mode3Unfurl_Init_Sub1                            ; 1 choose the start room, cue level palettes
    .ADDR Mode3Unfurl_Init_Sub2_FillAttrsCuePalettes                            ; 2 play area attributes, cue level palettes
    .ADDR Mode3Unfurl_Init_Sub3_TransferTopHalfAttrs       ; 3 top half of play area attributes
    .ADDR Mode3Unfurl_Init_Sub4_TransferBottomHalfAttrs    ; 4 bottom half of play area attributes
    .ADDR Mode3Unfurl_Init_Sub5_CueStatusBar                            ; 5 static parts of the status bar
    .ADDR Mode3Unfurl_Init_Sub6_CueStatusBarMap                            ; 6 status bar map (dungeon: only with the map)
    .ADDR Mode3Unfurl_Init_Sub7_CueLevelNumber                            ; 7 "LEVEL-n" text (dungeons only)
    .ADDR Mode3Unfurl_Init_Sub8_LayOutRoomPlaceLink                            ; 8 lay out the room, place Link, begin updating

; ----------------------------------------------------------------------------
; Mode 3 init step 0: blanks the screen, then clears the room history (falls through).
; NOTE: Sets IsVideoHeldOff to 1. While it is nonzero, Nmi_Isr does not turn rendering on in frames
;       that send the dynamic transfer buffer; Mode3Unfurl_Init_Sub8_LayOutRoomPlaceLink sets it back to 0.
; ----------------------------------------------------------------------------
Mode3Unfurl_Init_Sub0:
    LDA #$01
    STA IsVideoHeldOff          ; 1: keep video off during the rest of this init (see Nmi_Isr).
    INC GameSubmode
    JSR Ppu_TurnOffVideoAndClearNameTables

; ----------------------------------------------------------------------------
; Zeroes the six RoomHistory entries and IsSwordShotForced.
; OUT:  A = 0; Y = $FF.
; NOTE: Called when a level is loaded (Mode2Load_Init), at the start of mode 3 (fall-through above) and at
;       the start of mode 8 (continue menu). CurRoomHistoryIndex is not reset. The cleared entries
;       hold 0, which is also the id of room $00.
; ----------------------------------------------------------------------------
RoomHistory_Clear:
    LDY #$05
    LDA #$00
    STA IsSwordShotForced

@Loop:
    STA RoomHistory, Y
    DEY
    BPL @Loop
    RTS

; TABLE: offset of a save slot's palette row within the menu palettes, counted from row 4;
; index = save slot (0-2); value = byte offset (4 bytes per row: rows 4, 5, 6).
Palette_SaveSlotRowOffsets:
    .BYTE $00, $04, $08

; ----------------------------------------------------------------------------
; Mode 3 init step 1: chooses the room Link starts in, then cues the level palettes (falls through).
; NOTE: Dungeon (CurLevel <> 0): LevelInfo_StartRoomId. Overworld: LevelEntranceRoomId if it is not
;       $FF, else LevelInfo_StartRoomId. When the chosen room equals LevelEntranceRoomId, that variable
;       is set to $FF, so the dungeon entrance is used as a start room only once. The comparison is
;       also made in a dungeon, against the dungeon's start room.
; ----------------------------------------------------------------------------
Mode3Unfurl_Init_Sub1:
    LDA CurLevel
    BNE @UseStartRoomId         ; If in UW, then go set room ID to StartRoomId.
    LDA LevelEntranceRoomId
    CMP #$FF
    BNE @SetRoomId              ; If it's set to a valid room ID, then start there.

@UseStartRoomId:
    LDA LevelInfo_StartRoomId

@SetRoomId:
    STA RoomId
    CMP LevelEntranceRoomId
    BNE Palette_PatchLinkColorAndCueLevelPalettes    ; Started elsewhere: leave it alone.
    LDA #$FF                    ; Started at LevelEntranceRoomId: consume it (back to $FF = none).
    STA LevelEntranceRoomId

; ----------------------------------------------------------------------------
; Copies Link's tunic colour into the level palettes, cues their transfer and advances the submode.
; OUT:  TileBufSelector = $18 (LevelInfo_PalettesTransferBuf); GameSubmode + 1.
; NOTE: The colour is byte 1 of palette row 4 + save slot of Transfer_MenuPalettesBuf (the ring colour
;       of that slot; Item_Take patches it there when a ring is taken) and goes to byte 1 of row 4 of
;       the level palettes. Also used by mode 8 init, by mode $A init in the overworld (after a cave
;       changed the palettes) and by Item_Take after a ring changed the colour.
; ----------------------------------------------------------------------------
Palette_PatchLinkColorAndCueLevelPalettes:
    LDX CurSaveSlot
    LDY Palette_SaveSlotRowOffsets, X

    ; Get the colour at byte 1 of row 4, 5, or 6 of the menu palettes, according to save slot.
    ; (+20 = 3 header bytes + 4 rows of 4 bytes + 1.)
    LDA Transfer_MenuPalettesBuf+20, Y

    ; Put the value in byte 1 of row 4 of the level palettes that will be transferred.
    STA LevelInfo_PalettesTransferBuf+20
    LDA #$18                    ; Cue transfer of level palettes.
    STA TileBufSelector
    INC GameSubmode
    RTS

; ----------------------------------------------------------------------------
; Redraws the sprites that stay visible between rooms: map markers, Link, status bar items.
; NOTE: Hides all sprites first. Called at the start of the inits of modes 4, 5, 6, 7 and of the
;       cellar and cave modes.
; ----------------------------------------------------------------------------
Sprite_DrawBetweenRooms:
    JSR Sprite_HideAll
    JSR StatusBar_UpdateLinkPositionMarker
    JSR StatusBar_UpdateTriforceMarker
    LDA #$05
    JSR Mmc1_SwitchBank
    JSR Link_DrawBetweenRooms
    JMP StatusBar_DrawItemsAndEnsureSelection

; ----------------------------------------------------------------------------
; Makes Link idle (state 0) and turns the clock off.
; OUT:  A = 0.
; NOTE: Called on every room entry (modes 4, 9, $B, $C), on leaving a room (mode 6) and from the
;       continue menu, so the clock's freeze never lasts beyond the room it was taken in.
; ----------------------------------------------------------------------------
Link_ResetStateAndClock:
    LDA #$00
    STA ObjState
    STA InvClock
    RTS
