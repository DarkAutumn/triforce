; ============================================================================
; FILE:     bank5/cave-cellar-enter.asm  BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 6588-6803
; SUBJECTS: Mode, Mode9Cellar, ModeALeaveCellar, ModeBCave, World
; TOPIC:    Init submodes shared by the cave and cellar modes (9 enter cellar, $A leave cave or
;           cellar, $B enter cave, $C shortcut cave): sprites, palette fades, attributes, Link's
;           walk in. Also the per-frame heart refill (potion, pond fairy, end of level).
; SEE ALSO: bank5/room-history-ow.asm (ModeBCave_Init, ModeCShortcut_Init, Mode9Cellar_Init and their jump tables);
;           bank1/modeb-cave-save-sprites.asm (ModeBCave_Init_Sub7_EnterCave: Link's start in a cave);
;           bank5/cave-cellar-exits.asm (Warp_CheckSubroomExit: leaving, which starts mode $A);
;           bank5/warps-stairs-caves.asm (Warp_CheckUnderLink: going underground; CellarSourceRoomId);
;           bank5/enter-room.asm (Room_EnterCommon, Palette_SetFadeCycleAndAdvanceSubmode);
;           bank5/layout-room-ow.asm (cave and cellar layouts); bank5/ow-shortcut-map-square.asm
;           (ModeALeaveCellar_Init_Sub4_LayOutRoom and the row copy); bank5/mode3-mode6-init.asm (attribute
;           halves, Obj_ResetWeaponSlotsAndLadder); bank1/world-fading.asm (Palette_AnimateFade);
;           bank7/link-animate-ladder.asm (Link_EndMoveAndAnimate: zeroes the grid offset);
;           bank7/mode5-play-update.asm (World_UpdateHeartsAndRupees, Link_Update);
;           bank1/take-item.asm (Inv_CompareHeartsToContainers).
; KEY FACTS:
;   - Mode $A (leave a cave or cellar) init has 11 submodes; the last sets mode 4, submode 0, with
;     IsUpdatingMode still 0, so the room is then entered through Mode4Enter_Init. ModeALeaveCellar_Init_JumpTable,
;     ModeALeaveCellar_Init_SubA_GoToMode4
;   - Palette fades run only in dungeons (cellars). Entering: series $00 forward, then series $20
;     backward ($A0). Leaving: $20 forward, then $00 backward ($80). Each is 4 palette steps 10
;     frames apart. In the overworld (caves) the fade submodes only advance. Mode9Cellar_Init_Sub1_FadeOutLevel,
;     Mode9Cellar_Init_Sub6_FadeInCellar, ModeALeaveCellar_Init_Sub2_FadeOutCellar, ModeALeaveCellar_Init_Sub8_FadeInLevel, Mode_InitSubroom_AnimateFade
;   - In the overworld submode 0 turns on the sprite-0 split (Nmi_Isr then reads no controller
;     input) until the bottom attribute half is cued. Mode_InitSubroom_Sub0,
;     Mode_InitSubroom_TransferBottomHalfAttrs
;   - Every cave takes its palette selectors from overworld room $44.
;     ModeBCave_Init_Sub5_FillAttrsTransferTopHalf
;   - Cave entry: Link starts at X = $70, Y = $DD facing up with grid offset $30 but walks only 8
;     pixels, to Y = $D5: the end-of-move code zeroes a grid offset that is a multiple of 8.
;     ModeBCave_Init_Sub8_WalkIn
;   - Cellar entry: Link starts at Y = $41 facing down and walks $1C pixels to Y = $5D. X = $30 if
;     he came from the room the cellar's left stairs lead to, else $C0. Mode9Cellar_Init_Sub8_EnterCellar,
;     Mode9Cellar_LinkStartXs, Mode9Cellar_Init_Sub9_WalkIn
;   - UndergroundExitType becomes 1 when the cellar walk ends. Mode9Cellar_Init_Sub9_WalkIn
;   - Heart refill: HeartPartial + 6 per frame; a heart completes on the tick after it reaches $F8
;     or more, 43 frames per empty heart. It ends when full hearts equal containers, and that also
;     clears PauseState and IsSwordBlocked. World_FillHearts
; CONTENTS: (generated; L = line in this file)
;   L0074 ModeALeaveCellar_Init                                code  Mode $A (leave a cave or cellar) init: runs the handler of the current submode.
;   L0079 ModeALeaveCellar_Init_JumpTable                      data  TABLE: mode $A init submode handlers; index = GameSubmode 0-$0A; value = routine address.
;   L0099 Mode_InitSubroom_Sub0                                code  Submode 0 of modes 9, $A, $B and $C: clears CurRow and CurOpenedDoors and redraws the sprites.
;   L0109 Mode_InitSubroom_DrawSpritesAndAdvanceSubmode        code  Goes to the next submode and redraws Link, the map markers and the status bar items.
;   L0116 Mode9Cellar_Init_Sub5_TransferAttrs                  code  Mode 9 init, submode 5: cues the cellar's name table attributes; next submode.
;   L0120 Mode_InitSubroom_SelectTransferBuf                   code  Stores static transfer buffer selector A in TileBufSelector and goes to the next submode.
;   L0124 Mode_InitSubroom_AdvanceSubmode                      code  Submode step that only goes to the next submode (submode 2 of modes $B and $C); shared exit.
;   L0135 Mode9Cellar_Init_Sub1_FadeOutLevel                   code  Mode 9 init, submode 1: starts fade series $00 (the level's colours fade out); next submode.
;   L0147 Mode_InitSubroom_AnimateFade                         code  Submode step: runs the palette fade; goes to the next submode when the fade is done.
;   L0155 Mode9Cellar_Init_Sub6_FadeInCellar                   code  Mode 9 init, submode 6: starts fade series $20 backward (the cellar's colours fade in).
;   L0162 ModeALeaveCellar_Init_Sub2_FadeOutCellar             code  Mode $A init, submode 2: starts fade series $20 forward (the cellar's colours fade out).
;   L0169 ModeALeaveCellar_Init_Sub8_FadeInLevel               code  Mode $A init, submode 8: starts fade series $00 backward (the level's colours fade in).
;   L0176 ModeBCave_Init_Sub1_CueCavePalettes                  code  Modes $B and $C init, submode 1: cues the cave's background palette rows; next submode.
;   L0185 ModeALeaveCellar_Init_Sub1_CueLevelPalettes          code  Mode $A init, submode 1: in the overworld, cues the level palettes again; next submode.
;   L0195 ModeALeaveCellar_Init_SubA_GoToMode4                 code  Mode $A init, last submode: clears the weapon slots and switches to mode 4 (enter room).
;   L0206 ModeALeaveCellar_Init_Sub6_FillAttrsTransferTopHalf  code  Mode $A init, submode 6: builds the room's attribute bytes and cues their top half.
;   L0215 ModeBCave_Init_Sub5_FillAttrsTransferTopHalf         code  Modes $B and $C init, submode 5: builds the cave's attribute bytes and cues their top half.
;   L0225 Mode_InitSubroom_TransferBottomHalfAttrs             code  Submode step of modes $A, $B and $C: cues the bottom attribute half, ends the sprite-0 split.
;   L0246 ModeBCave_Init_Sub8_WalkIn                           code  Modes $B and $C init, submode 8: walks Link up into the cave, then starts the mode's update.
;   L0262 Mode9Cellar_LinkStartXs                              data  TABLE: Link's X at the top of a cellar's stairs; index = 0 he came from the room in
;   L0276 Mode9Cellar_Init_Sub8_EnterCellar                    code  Mode 9 init, submode 8: sets up the cellar room and puts Link at the top of its stairs.
;   L0317 Mode9Cellar_Init_Sub9_WalkIn                         code  Mode 9 init, submode 9: walks Link down the cellar stairs to Y = $5D, then starts the update.
;   L0343 World_FillHearts                                     code  One tick of the heart refill: grows the partial heart, completes hearts, stops when full.
; ============================================================================

; ----------------------------------------------------------------------------
; Mode $A (leave a cave or cellar) init: runs the handler of the current submode.
; NOTE: RoomId is already the room Link returns to (bank5/cave-cellar-exits.asm). Submodes: 0
;       sprites; 1 overworld palettes; 2-3 fade out (dungeon only); 4 lay out the room; 5 copy its
;       22 tile rows; 6-7 attributes; 8-9 fade in (dungeon only); $A switch to mode 4.
; ----------------------------------------------------------------------------
ModeALeaveCellar_Init:
    LDA GameSubmode
    JSR Util_TableJump

; TABLE: mode $A init submode handlers; index = GameSubmode 0-$0A; value = routine address.
ModeALeaveCellar_Init_JumpTable:
    .ADDR Mode_InitSubroom_Sub0                              ; 0: sprites, row and door state
    .ADDR ModeALeaveCellar_Init_Sub1_CueLevelPalettes                                    ; 1: overworld: level palettes again
    .ADDR ModeALeaveCellar_Init_Sub2_FadeOutCellar                                    ; 2: dungeon: start fade series $20
    .ADDR Mode_InitSubroom_AnimateFade                       ; 3: dungeon: run the fade
    .ADDR ModeALeaveCellar_Init_Sub4_LayOutRoom                            ; 4: lay out the room's tile map
    .ADDR Mode_CopyNextRowAndAdvanceSubmodeWhenDone ; 5: copy one tile row per frame
    .ADDR ModeALeaveCellar_Init_Sub6_FillAttrsTransferTopHalf    ; 6: attributes, top half
    .ADDR Mode_InitSubroom_TransferBottomHalfAttrs              ; 7: attributes, bottom half
    .ADDR ModeALeaveCellar_Init_Sub8_FadeInLevel                                    ; 8: dungeon: start series $00 backward
    .ADDR Mode_InitSubroom_AnimateFade                       ; 9: dungeon: run the fade
    .ADDR ModeALeaveCellar_Init_SubA_GoToMode4                          ; $A: go to mode 4

; ----------------------------------------------------------------------------
; Submode 0 of modes 9, $A, $B and $C: clears CurRow and CurOpenedDoors and redraws the sprites.
; OUT:  CurRow = 0; CurOpenedDoors = 0; GameSubmode incremented; in the overworld also sprite 0
;       written and IsSprite0CheckActive = 1.
; NOTE: While IsSprite0CheckActive is set Nmi_Isr splits the screen below the status bar and does
;       not read the controllers. Mode_InitSubroom_TransferBottomHalfAttrs turns it off again.
; ----------------------------------------------------------------------------
Mode_InitSubroom_Sub0:
    LDA #$00
    STA CurRow
    STA CurOpenedDoors
    LDA CurLevel
    BNE Mode_InitSubroom_DrawSpritesAndAdvanceSubmode    ; Dungeon (cellar): no sprite-0 split.
    JSR Mode_InitSubroom_DrawSpritesAndAdvanceSubmode
    JMP Scroll_WriteAndEnableSprite0   ; Overworld (cave): turn the sprite-0 split on.

; Goes to the next submode and redraws Link, the map markers and the status bar items.
Mode_InitSubroom_DrawSpritesAndAdvanceSubmode:
    INC GameSubmode
    JMP Sprite_DrawBetweenRooms

; ----------------------------------------------------------------------------
; Mode 9 init, submode 5: cues the cellar's name table attributes; next submode.
; ----------------------------------------------------------------------------
Mode9Cellar_Init_Sub5_TransferAttrs:
    LDA #$26                    ; Selector $26 = Transfer_CellarAttrsBuf.

; Stores static transfer buffer selector A in TileBufSelector and goes to the next submode.
Mode_InitSubroom_SelectTransferBuf:
    STA TileBufSelector

; Submode step that only goes to the next submode (submode 2 of modes $B and $C); shared exit.
Mode_InitSubroom_AdvanceSubmode:
    INC GameSubmode
    RTS

; ----------------------------------------------------------------------------
; Mode 9 init, submode 1: starts fade series $00 (the level's colours fade out); next submode.
; NOTE: The unnamed label below is the common entry of the four fade-start submodes, with A = the
;       FadeCycle start value. In the overworld (CurLevel = 0) it only goes to the next submode:
;       caves do not fade. In a dungeon it sets FadeCycle, increments GameSubmode and falls into
;       Mode_InitSubroom_AnimateFade, so the first palette step is cued in the same frame.
; ----------------------------------------------------------------------------
Mode9Cellar_Init_Sub1_FadeOutLevel:
    LDA #$00                    ; FadeCycle $00: series $00, forward.
:
    LDY CurLevel
    BEQ Mode_InitSubroom_AdvanceSubmode
    JSR Palette_SetFadeCycleAndAdvanceSubmode

; ----------------------------------------------------------------------------
; Submode step: runs the palette fade; goes to the next submode when the fade is done.
; NOTE: Dungeon only; in the overworld it just goes to the next submode. A fade is 4 palette steps,
;       one every 10 frames (Palette_AnimateFade, timer ObjTimer+12).
; ----------------------------------------------------------------------------
Mode_InitSubroom_AnimateFade:
    LDY CurLevel
    BEQ Mode_InitSubroom_AdvanceSubmode
    JMP Mode11Death_Update_Sub8Fade    ; Palette_AnimateFade; INC GameSubmode when done.

; ----------------------------------------------------------------------------
; Mode 9 init, submode 6: starts fade series $20 backward (the cellar's colours fade in).
; ----------------------------------------------------------------------------
Mode9Cellar_Init_Sub6_FadeInCellar:
    LDA #$A0                    ; FadeCycle $A0: series $20, bit 7 = backward.
    BNE :-                      ; Always taken: to the common entry in Mode9Cellar_Init_Sub1_FadeOutLevel.

; ----------------------------------------------------------------------------
; Mode $A init, submode 2: starts fade series $20 forward (the cellar's colours fade out).
; ----------------------------------------------------------------------------
ModeALeaveCellar_Init_Sub2_FadeOutCellar:
    LDA #$20                    ; FadeCycle $20: series $20, forward.
    BNE :-                      ; Always taken: to the common entry in Mode9Cellar_Init_Sub1_FadeOutLevel.

; ----------------------------------------------------------------------------
; Mode $A init, submode 8: starts fade series $00 backward (the level's colours fade in).
; ----------------------------------------------------------------------------
ModeALeaveCellar_Init_Sub8_FadeInLevel:
    LDA #$80                    ; FadeCycle $80: series $00, bit 7 = backward.
    BNE :-                      ; Always taken: to the common entry in Mode9Cellar_Init_Sub1_FadeOutLevel.

; ----------------------------------------------------------------------------
; Modes $B and $C init, submode 1: cues the cave's background palette rows; next submode.
; ----------------------------------------------------------------------------
ModeBCave_Init_Sub1_CueCavePalettes:
    LDA #$3E                    ; Selector $3E = Transfer_CaveBgPaletteRowsBuf.
    BNE Mode_InitSubroom_SelectTransferBuf    ; Always taken.

; ----------------------------------------------------------------------------
; Mode $A init, submode 1: in the overworld, cues the level palettes again; next submode.
; NOTE: The cave replaced background palette rows (ModeBCave_Init_Sub1_CueCavePalettes); this puts the overworld's
;       back. In a dungeon it only goes to the next submode.
; ----------------------------------------------------------------------------
ModeALeaveCellar_Init_Sub1_CueLevelPalettes:
    LDA CurLevel
    BNE Mode_InitSubroom_AdvanceSubmode
    JMP Palette_PatchLinkColorAndCueLevelPalettes

; ----------------------------------------------------------------------------
; Mode $A init, last submode: clears the weapon slots and switches to mode 4 (enter room).
; OUT:  GameMode = 4; GameSubmode = 0. IsUpdatingMode is still 0, so Mode4Enter_Init runs next frame and
;       Room_EnterCommon places Link (UndergroundExitType is nonzero: he comes out of the ground).
; ----------------------------------------------------------------------------
ModeALeaveCellar_Init_SubA_GoToMode4:
    JSR Obj_ResetWeaponSlotsAndLadder
    LDA #$00
    STA GameSubmode
    LDA #$04
    STA GameMode
    RTS

; ----------------------------------------------------------------------------
; Mode $A init, submode 6: builds the room's attribute bytes and cues their top half.
; ----------------------------------------------------------------------------
ModeALeaveCellar_Init_Sub6_FillAttrsTransferTopHalf:
    LDA RoomId
    JMP :+                      ; To the Room_FillPlayAreaAttrs call just below.

; ----------------------------------------------------------------------------
; Modes $B and $C init, submode 5: builds the cave's attribute bytes and cues their top half.
; NOTE: Every cave uses the palette selectors of overworld room $44 (its attribute bytes A and B,
;       bits 1-0, read by Room_FillPlayAreaAttrs).
; ----------------------------------------------------------------------------
ModeBCave_Init_Sub5_FillAttrsTransferTopHalf:
    LDA #$44                    ; Overworld room $44 stands in for the cave.
:
    JSR Room_FillPlayAreaAttrs
    JMP Mode3Unfurl_Init_Sub3_TransferTopHalfAttrs

; ----------------------------------------------------------------------------
; Submode step of modes $A, $B and $C: cues the bottom attribute half, ends the sprite-0 split.
; OUT:  GameSubmode incremented (by the called step); IsSprite0CheckActive = 0.
; ----------------------------------------------------------------------------
Mode_InitSubroom_TransferBottomHalfAttrs:
    JSR Mode3Unfurl_Init_Sub4_TransferBottomHalfAttrs
    JMP @DisableSprite0Check

    ; Unreachable: it follows an unconditional JMP and has no label.
    INC GameSubmode

@DisableSprite0Check:
    LDA #$00
    STA IsSprite0CheckActive
    RTS

; ----------------------------------------------------------------------------
; Modes $B and $C init, submode 8: walks Link up into the cave, then starts the mode's update.
; NOTE: Submode 7 (ModeBCave_Init_Sub7_EnterCave) put Link at X = $70, Y = $DD facing up with grid
;       offset $30. Each frame here moves him at his q-speed until the grid offset is 0. In these
;       modes Link_EndMoveAndAnimate zeroes a grid offset that is a multiple of 8, so the walk ends
;       when the offset reaches $28: Link moves 8 pixels and stops at Y = $D5. With q-speed $60
;       (1.5 pixels per frame; Room_EnterCommon calls Link_InitSpeed) and the position fraction 0
;       left by Room_EnterCommon that takes 5 frames (2, 1, 2, 1, 2 pixels).
; ----------------------------------------------------------------------------
ModeBCave_Init_Sub8_WalkIn:
    LDA ObjGridOffset
    BEQ @StartUpdate            ; Grid offset 0: the walk is over.

    ; Move Link one frame in his facing direction (up) and draw him.
    LDA ObjDir
    STA ObjInputDir
    STA $0F                     ; [0F] = direction for Obj_MoveByQSpeed.
    LDX #$00
    JSR Obj_MoveByQSpeed
    JMP Link_EndMoveAndAnimateInRoom
@StartUpdate:
    JMP Room_RunEntryTasks

; TABLE: Link's X at the top of a cellar's stairs; index = 0 he came from the room in
; LevelBlockAttrsA of the cellar room (its left stairs), 1 any other room; value = pixels.
Mode9Cellar_LinkStartXs:
    .BYTE $30, $C0

; ----------------------------------------------------------------------------
; Mode 9 init, submode 8: sets up the cellar room and puts Link at the top of its stairs.
; OUT:  Link at Y = $41 facing down, X = $30 or $C0; ObjGridOffset = $E4; DoorwayDir = 0;
;       IsUpdatingMode = 0; GameSubmode = 9.
; NOTE: Room_EnterCommon ends in Mode_BeginUpdate, which zeroes GameSubmode and sets
;       IsUpdatingMode; both are put back here so that the init goes on with submode 9.
; NOTE: X = $30 when CellarSourceRoomId (the room Link took the stairs in) equals LevelBlockAttrsA
;       of the cellar room, the room its left stairs lead to; else $C0. In the level data all 20
;       treasure cellars (layout $3F, both quests) have attributes A and B equal to the one room
;       that leads to them (data/levelblock-uw-*.md), so Link always starts at X = $30 there.
; ----------------------------------------------------------------------------
Mode9Cellar_Init_Sub8_EnterCellar:
    LDA GameSubmode
    PHA                         ; Save the submode.
    JSR Room_EnterCommon
    JSR Obj_ResetWeaponSlotsAndLadder
    PLA                         ; Restore the submode.
    STA GameSubmode

    ; X = 0 (left stairs) if Link came from the room in attribute A of the cellar room, else 1
    ; (right stairs).
    LDY RoomId
    LDX #$00
    LDA CellarSourceRoomId
    CMP LevelBlockAttrsA, Y
    BEQ :+
    INX
:
    LDA Mode9Cellar_LinkStartXs, X
    STA ObjX
    LDA #$41                    ; Link's Y at the top of the stairs.
    STA ObjY
    LDA #$04                    ; Facing down.
    STA ObjDir

    ; Grid offset $E4 = -$1C, the distance to Y = $5D. After 4 pixels it is $E8, a multiple of 8,
    ; which Link_EndMoveAndAnimate zeroes: Link is then on the grid at Y = $45 and goes on in
    ; steps of 8 pixels to $5D.
    LDA #$E4
    STA ObjGridOffset
    LDA #$00
    STA IsUpdatingMode          ; Still initializing (Room_EnterCommon set it).
    STA DoorwayDir
    INC GameSubmode
    RTS

; ----------------------------------------------------------------------------
; Mode 9 init, submode 9: walks Link down the cellar stairs to Y = $5D, then starts the update.
; OUT:  when ObjY = $5D: ObjState = 0, UndergroundExitType = 1, IsUpdatingMode = 1.
; NOTE: Each frame is a normal Link_Update with the input direction set to Link's facing
;       direction (down); the walk is $1C pixels from Y = $41.
; ----------------------------------------------------------------------------
Mode9Cellar_Init_Sub9_WalkIn:
    LDA ObjDir
    STA ObjInputDir
    JSR Link_Update
    LDA ObjY
    CMP #$5D
    BNE @Exit                   ; Not at the bottom of the stairs yet.
    LDA #$00
    STA ObjState                ; Link idle.
    LDA #$01
    STA UndergroundExitType     ; 1 = Link is in a cave or cellar (used when he leaves).
    STA IsUpdatingMode          ; Start updating the mode.
@Exit:
    RTS

; ----------------------------------------------------------------------------
; One tick of the heart refill: grows the partial heart, completes hearts, stops when full.
; IN:   World_IsFillingHearts (0 = nothing to do)
; OUT:  HeartPartial and HeartValues advanced. When full hearts equal heart containers:
;       HeartPartial = $FF and World_IsFillingHearts, PauseState and IsSwordBlocked = 0.
; NOTE: World_UpdateHeartsAndRupees calls it once per frame. HeartPartial rises by 6 per tick; once it
;       is $F8 or more the next tick completes the heart (HeartPartial = 0, full hearts + 1). From
;       an empty heart that is 43 frames per heart: 42 ticks reach $FC, the 43rd completes it.
; NOTE: Every tick while filling requests square-0 tune $10.
; NOTE: The end of a refill clears IsSwordBlocked, whatever started the refill.
; ----------------------------------------------------------------------------
World_FillHearts:
    LDA World_IsFillingHearts
    BEQ @Exit                   ; Not refilling: return.
    LDA #$10                    ; Tune $10: the tick sound, also used for rupee counting and text.
    STA Tune0Request
    LDA HeartPartial
    CMP #$F8
    BCS @CompleteHeart          ; Partial heart >= $F8: complete it.
    CLC
    ADC #$06                    ; Else grow the partial heart by 6.
    STA HeartPartial
    RTS

@CompleteHeart:
    LDA #$00                    ; The next heart starts empty.
    STA HeartPartial
    JSR Inv_CompareHeartsToContainers
    BNE @IncHearts              ; Full hearts (low nibble) <> containers - 1 (high nibble): one more.

    ; The partial heart was the last container. Make it full ($00 - 1 = $FF) and end the refill.
    DEC HeartPartial
    LDA #$00
    STA IsSwordBlocked
    STA World_IsFillingHearts
    STA PauseState              ; Ends the pause a potion started (PauseState 2).

@Exit:
    RTS

@IncHearts:
    INC HeartValues
    RTS
