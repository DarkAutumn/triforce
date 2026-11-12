; ============================================================================
; FILE:     bank7/init-mode-dispatch.asm BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 1257-1489
; SUBJECTS: (none)
; TOPIC:    The init half of the mode machine: one-time game setup, the per-mode init dispatch, the mode
;           inits that live in the fixed bank (0, 1, 2, 3, 7, $E/$F, $13), room history clearing, and
;           helpers run on every room change (sprites between rooms, Link's state and clock reset).
; SEE ALSO: bank7/run-game-nmi.asm (IsrNmi calls InitializeGameOrMode or UpdateMode once per frame);
;           bank7/update-mode-dispatch.asm (UpdateMode, GoToNextMode, EndGameMode); bank7/init-mode5-play.asm
;           (InitMode5Play, room history recording); bank5/mode3-mode6-init.asm (InitMode3_Sub2..Sub8,
;           InitMode6); bank5/enter-room.asm (InitMode4); bank5/mode7-scroll-init.asm (InitMode7Submodes);
;           bank6/level-load-mode2.asm (InitMode2_Submodes); bank1/take-item.asm (ring colour patch).
; KEY FACTS:
;   - Each frame IsrNmi runs the mode's init routine while IsUpdatingMode = 0 and its update routine
;     otherwise; an init may take many frames, stepping GameSubmode. InitializeGameOrMode, InitMode
;   - The very first frame after reset only copies the common code and data to RAM and writes the save
;     RAM markers ($5A at SaveRamBegin, $A5 at SaveRamEnd); no mode runs. InitializeGameOrMode
;   - Modes $E and $F share one init routine. InitMode_JumpTable
;   - Loading a level (mode 2, submode 0) zeroes the room history and all 128 LevelKillCounts (the
;     kills banked per dungeon room). Overworld kill counts are in the room flags and stay. InitMode2
;   - Room history is also cleared at the start of mode 3 (unfurl) and of mode 8 (continue menu).
;     InitMode3_Sub0, ClearRoomHistory
;   - Mode 3 starts Link in LevelInfo_StartRoomId, except in the overworld when LevelEntranceRoomId is
;     not $FF: then he starts in that room. Whenever the start room equals LevelEntranceRoomId (the
;     test is also made in a dungeon) the variable is set back to $FF. InitMode3_Sub1
;   - Every room entry (modes 4, 9, $B, $C init), every room exit (mode 6 init) and a choice in the
;     continue menu set Link's state to 0 and turn the clock off (InvClock = 0). ResetPlayerState
;   - Room scrolling uses vertical mirroring when Link faces left or right and horizontal mirroring
;     when he faces up or down. InitMode7
;   - Link's tunic colour comes from the save slot's row of the menu palettes and is patched into the
;     level palettes before they are transferred. PatchAndCueLevelPalettesTransferAndAdvanceSubmode
; CONTENTS: (generated; L = line in this file)
;   L0059 InitializeGameOrMode                               code  Per-frame entry while a mode is initializing: sets the game up once, then runs the mode's init.
;   L0079 InitMode                                           code  Switches to bank 5 and jumps to the init routine of the current game mode.
;   L0086 InitMode_JumpTable                                 data  TABLE: mode init routines; index = GameMode ($00-$13); value = routine address.
;   L0113 InitMode0                                          code  Mode 0 init: loads the common and the demo pattern tables if missing, else runs the demo init tasks.
;   L0135 InitMode1                                          code  Mode 1 init: trampoline to the file menu init in bank 2.
;   L0146 InitMode2                                          code  Mode 2 init: on submode 0 forgets room history and kill counts and loads the level's patterns.
;   L0180 InitMode7                                          code  Mode 7 init: runs the scroll init submode, then sets name table mirroring for the scroll direction.
;   L0213 InitModeEandF                                      code  Mode $E and $F init: trampoline to the register/eliminate init in bank 2.
;   L0221 InitMode13                                         code  Mode $13 init: sets horizontal mirroring, then runs the ending init in bank 2.
;   L0233 InitMode3                                          code  Mode 3 init: turns video off and runs the unfurl init step selected by GameSubmode (0-8).
;   L0241 InitMode3_JumpTable                                data  TABLE: mode 3 init steps; index = GameSubmode; value = routine address.
;   L0257 InitMode3_Sub0                                     code  Mode 3 init step 0: blanks the screen, then clears the room history (falls through).
;   L0270 ClearRoomHistory                                   code  Zeroes the six RoomHistory entries and IsSwordShotForced.
;   L0283 SaveSlotToPaletteRowOffset                         data  TABLE: offset of a save slot's palette row within the menu palettes, counted from row 4;
;   L0293 InitMode3_Sub1                                     code  Mode 3 init step 1: chooses the room Link starts in, then cues the level palettes (falls through).
;   L0318 PatchAndCueLevelPalettesTransferAndAdvanceSubmode  code  Copies Link's tunic colour into the level palettes, cues their transfer and advances the submode.
;   L0338 DrawSpritesBetweenRooms                            code  Redraws the sprites that stay visible between rooms: map markers, Link, status bar items.
;   L0353 ResetPlayerState                                   code  Makes Link idle (state 0) and turns the clock off.
; ============================================================================

; ----------------------------------------------------------------------------
; Per-frame entry while a mode is initializing: sets the game up once, then runs the mode's init.
; NOTE: Called by IsrNmi when IsUpdatingMode = 0. On the first call after reset (IsGameInitialized = 0)
;       it copies the common code (bank 1) and common data (bank 6) to RAM, marks save RAM initialized
;       and returns without running a mode init; later calls fall into InitMode.
; ----------------------------------------------------------------------------
InitializeGameOrMode:
    LDA IsGameInitialized
    BNE InitMode
    LDA #$01
    JSR SwitchBank
    JSR CopyCommonCodeToRam
    LDA #$06
    JSR SwitchBank
    JSR CopyCommonDataToRam
    LDA #$5A                    ; Mark save RAM initialized, so it can be recognised after a reset.
    STA SaveRamBegin
    LDA #$A5
    STA SaveRamEnd
    INC IsGameInitialized
    RTS

; ----------------------------------------------------------------------------
; Switches to bank 5 and jumps to the init routine of the current game mode.
; NOTE: Runs once per frame until the init routine calls BeginUpdateMode (IsUpdatingMode becomes 1).
; ----------------------------------------------------------------------------
InitMode:
    LDA #$05
    JSR SwitchBank
    LDA GameMode
    JSR TableJump

; TABLE: mode init routines; index = GameMode ($00-$13); value = routine address.
InitMode_JumpTable:
    .ADDR InitMode0             ; $00 demo
    .ADDR InitMode1             ; $01 file menu
    .ADDR InitMode2             ; $02 load level
    .ADDR InitMode3             ; $03 unfurl
    .ADDR InitMode4             ; $04 enter room
    .ADDR InitMode5Play         ; $05 play
    .ADDR InitMode6             ; $06 leave room
    .ADDR InitMode7             ; $07 scroll
    .ADDR InitMode8             ; $08 continue menu
    .ADDR InitMode9             ; $09 enter cellar
    .ADDR InitModeA             ; $0A leave cellar
    .ADDR InitModeB             ; $0B enter cave
    .ADDR InitModeC             ; $0C shortcut stairs
    .ADDR InitModeD             ; $0D save
    .ADDR InitModeEandF         ; $0E register
    .ADDR InitModeEandF         ; $0F eliminate (same routine as $0E)
    .ADDR InitMode10            ; $10 stairs
    .ADDR InitMode11            ; $11 death
    .ADDR InitMode12            ; $12 end level
    .ADDR InitMode13            ; $13 win game

; ----------------------------------------------------------------------------
; Mode 0 init: loads the common and the demo pattern tables if missing, else runs the demo init tasks.
; NOTE: One of the three jobs per frame: common patterns (bank 2) until HasCommonPatterns = $5A, demo
;       patterns (bank 1) until HasDemoPatterns = $A5, then InitDemo_RunTasks (bank 2).
; ----------------------------------------------------------------------------
InitMode0:
    LDA HasCommonPatterns
    CMP #$5A
    BEQ :+                      ; Common patterns already transferred.
    LDA #$02
    JSR SwitchBank
    JMP TransferCommonPatterns
:
    LDA HasDemoPatterns
    CMP #$A5
    BEQ :+                      ; Demo patterns already transferred.
    LDA #$01
    JSR SwitchBank
    JMP TransferDemoPatterns
:
    LDA #$02
    JSR SwitchBank
    JMP InitDemo_RunTasks

; ----------------------------------------------------------------------------
; Mode 1 init: trampoline to the file menu init in bank 2.
; ----------------------------------------------------------------------------
InitMode1:
    LDA #$02
    JSR SwitchBank
    JMP InitMode1_Full

; ----------------------------------------------------------------------------
; Mode 2 init: on submode 0 forgets room history and kill counts and loads the level's patterns.
; NOTE: Video is turned off every frame of this init. Submode 0 also copies the common code to RAM
;       again. Every frame then continues in InitMode2_Submodes (bank 6), which copies the level data.
;       Zeroing LevelKillCounts means no room of the newly loaded level remembers banked kills.
; ----------------------------------------------------------------------------
InitMode2:
    JSR TurnOffAllVideo
    LDA GameSubmode
    BNE @InitSubmodes

    ; Submode 0.
    ; Clear the room history, transfer level pattern blocks and copy common code.
    JSR ClearRoomHistory        ; Returns A = 0.

    ; Clear the level kill counts of all $80 rooms.
    LDY #$7F

@ClearCounts:
    STA LevelKillCounts, Y
    DEY
    BPL @ClearCounts
    LDA #$03
    JSR SwitchBank
    JSR TransferLevelPatternBlocks
    LDA #$01
    JSR SwitchBank
    JSR CopyCommonCodeToRam

@InitSubmodes:
    LDA #$06
    JSR SwitchBank
    JMP InitMode2_Submodes

; ----------------------------------------------------------------------------
; Mode 7 init: runs the scroll init submode, then sets name table mirroring for the scroll direction.
; NOTE: The mirroring is set once, on the first frame on which the sprite-0 check is active:
;       vertical mirroring (MMC1 control $0E) when Link faces right or left, horizontal ($0F) when he
;       faces down or up. UpdateMode7Scroll restores $0F when the scroll ends.
; ----------------------------------------------------------------------------
InitMode7:
    LDA #$05
    JSR SwitchBank
    JSR InitMode7Submodes
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
    JSR SetMMC1Control

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Mode $E and $F init: trampoline to the register/eliminate init in bank 2.
; ----------------------------------------------------------------------------
InitModeEandF:
    LDA #$02
    JSR SwitchBank
    JMP InitModeEandF_Full

; ----------------------------------------------------------------------------
; Mode $13 init: sets horizontal mirroring, then runs the ending init in bank 2.
; ----------------------------------------------------------------------------
InitMode13:
    ; Make sure horizontal mirroring is on.
    LDA #$0F
    JSR SetMMC1Control
    LDA #$02
    JSR SwitchBank
    JMP InitMode13_Full

; ----------------------------------------------------------------------------
; Mode 3 init: turns video off and runs the unfurl init step selected by GameSubmode (0-8).
; NOTE: One submode per frame. Bank 5 is switched in (submodes 2-8 live there).
; ----------------------------------------------------------------------------
InitMode3:
    LDA #$05
    JSR SwitchBank
    JSR TurnOffAllVideo
    LDA GameSubmode
    JSR TableJump

; TABLE: mode 3 init steps; index = GameSubmode; value = routine address.
InitMode3_JumpTable:
    .ADDR InitMode3_Sub0                            ; 0 clear screen and room history
    .ADDR InitMode3_Sub1                            ; 1 choose the start room, cue level palettes
    .ADDR InitMode3_Sub2                            ; 2 play area attributes, cue level palettes
    .ADDR InitMode3_Sub3_TransferTopHalfAttrs       ; 3 top half of play area attributes
    .ADDR InitMode3_Sub4_TransferBottomHalfAttrs    ; 4 bottom half of play area attributes
    .ADDR InitMode3_Sub5                            ; 5 static parts of the status bar
    .ADDR InitMode3_Sub6                            ; 6 status bar map (dungeon: only with the map)
    .ADDR InitMode3_Sub7                            ; 7 "LEVEL-n" text (dungeons only)
    .ADDR InitMode3_Sub8                            ; 8 lay out the room, place Link, begin updating

; ----------------------------------------------------------------------------
; Mode 3 init step 0: blanks the screen, then clears the room history (falls through).
; NOTE: Sets [17] to 1. While [17] is nonzero, IsrNmi does not turn rendering on in frames that send
;       the dynamic transfer buffer; InitMode3_Sub8 sets it back to 0. [17] has no symbol.
; ----------------------------------------------------------------------------
InitMode3_Sub0:
    LDA #$01
    STA $17                     ; [17] = 1: keep video off during the rest of this init (see IsrNmi).
    INC GameSubmode
    JSR TurnOffVideoAndClearArtifacts

; ----------------------------------------------------------------------------
; Zeroes the six RoomHistory entries and IsSwordShotForced.
; OUT:  A = 0; Y = $FF.
; NOTE: Called when a level is loaded (InitMode2), at the start of mode 3 (fall-through above) and at
;       the start of mode 8 (continue menu). CurRoomHistoryIndex is not reset. The cleared entries
;       hold 0, which is also the id of room $00.
; ----------------------------------------------------------------------------
ClearRoomHistory:
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
SaveSlotToPaletteRowOffset:
    .BYTE $00, $04, $08

; ----------------------------------------------------------------------------
; Mode 3 init step 1: chooses the room Link starts in, then cues the level palettes (falls through).
; NOTE: Dungeon (CurLevel <> 0): LevelInfo_StartRoomId. Overworld: LevelEntranceRoomId if it is not
;       $FF, else LevelInfo_StartRoomId. When the chosen room equals LevelEntranceRoomId, that variable
;       is set to $FF, so the dungeon entrance is used as a start room only once. The comparison is
;       also made in a dungeon, against the dungeon's start room.
; ----------------------------------------------------------------------------
InitMode3_Sub1:
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
    BNE PatchAndCueLevelPalettesTransferAndAdvanceSubmode    ; Started elsewhere: leave it alone.
    LDA #$FF                    ; Started at LevelEntranceRoomId: consume it (back to $FF = none).
    STA LevelEntranceRoomId

; ----------------------------------------------------------------------------
; Copies Link's tunic colour into the level palettes, cues their transfer and advances the submode.
; OUT:  TileBufSelector = $18 (LevelInfo_PalettesTransferBuf); GameSubmode + 1.
; NOTE: The colour is byte 1 of palette row 4 + save slot of MenuPalettesTransferBuf (the ring colour
;       of that slot; TakeItem patches it there when a ring is taken) and goes to byte 1 of row 4 of
;       the level palettes. Also used by mode 8 init, by mode $A init in the overworld (after a cave
;       changed the palettes) and by TakeItem after a ring changed the colour.
; ----------------------------------------------------------------------------
PatchAndCueLevelPalettesTransferAndAdvanceSubmode:
    LDX CurSaveSlot
    LDY SaveSlotToPaletteRowOffset, X

    ; Get the colour at byte 1 of row 4, 5, or 6 of the menu palettes, according to save slot.
    ; (+20 = 3 header bytes + 4 rows of 4 bytes + 1.)
    LDA MenuPalettesTransferBuf+20, Y

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
DrawSpritesBetweenRooms:
    JSR HideAllSprites
    JSR UpdatePlayerPositionMarker
    JSR UpdateTriforcePositionMarker
    LDA #$05
    JSR SwitchBank
    JSR DrawLinkBetweenRooms
    JMP DrawStatusBarItemsAndEnsureItemSelected

; ----------------------------------------------------------------------------
; Makes Link idle (state 0) and turns the clock off.
; OUT:  A = 0.
; NOTE: Called on every room entry (modes 4, 9, $B, $C), on leaving a room (mode 6) and from the
;       continue menu, so the clock's freeze never lasts beyond the room it was taken in.
; ----------------------------------------------------------------------------
ResetPlayerState:
    LDA #$00
    STA ObjState
    STA InvClock
    RTS
