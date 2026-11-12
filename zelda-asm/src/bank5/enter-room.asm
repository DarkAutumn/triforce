; ============================================================================
; FILE:     bank5/enter-room.asm         BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 1450-1868
; SUBJECTS: Link, Mode4Enter, Palette, Room
; TOPIC:    Mode 4 (enter room) init submodes, and the room-entry routine shared by modes 4, 9, $B and $C:
;           clears per-room RAM, places Link, decodes the room's monster list and count (reduced by the
;           room history), fills the object types, and sets the overworld tile object.
; SEE ALSO: bank5/spawn-positions.asm (Spawn_AssignPositions: where the monsters are put; cellar Keese;
;           cave person); bank5/room-history-ow.asm and bank5/room-history-uw.asm (how remembered kills
;           cut the monster count); bank5/mode-inits-obj-lists.asm (Spawn_ObjLists, Spawn_ObjListAddrs,
;           Mode4Enter_WalkInGridOffsets); bank7/init-mode5-play.asm (room history insert, room item
;           creation after entry); bank7/update-mode-dispatch.asm (mode 4 update: Link's walk-in);
;           bank5/mode7-scroll-update.asm (enters mode 4 at submode 1); bank5/layout-room-ow.asm (finds
;           the tile object).
; KEY FACTS:
;   - Monster list id = attribute byte C bits 5-0, plus $40 when attribute byte D bit 7 is set. Ids
;     below $62 are one object type repeated; ids $62-$7F pick a list of types. Room_EnterCommon.
;   - Monster count = LevelInfo_FoeCounts[attribute byte C bits 7-6], but forced to 1 for ids $32-$61
;     (bosses and other single objects). Room_EnterCommon.
;   - The count is then cut by kills remembered for the room (RoomHistory_ModifyObjCountOW / UW); a
;     result below 0 (or a cleared room) gives count 0 and list id 0. The final count is RoomObjCount.
;     Room_EnterCommon@StoreObjCount.
;   - Types go to object slots 1..count (ObjType+1..); slot 1's type is kept as RoomObjTemplateType.
;     Count 0 or id 0 makes no monsters. Room_EnterCommon@StoreObjTemplate.
;   - In mode 9 (cellar) the room's list and count are discarded (4 blue Keese are made by
;     Spawn_AssignPositions instead). Room_EnterCommon.
;   - All of $0300-$051F is zeroed on every room entry, then object slots $0B..1 get: uninitialized
;     flag, state 0, direction 0, no stun or knockback, metastate 1 (spawn cloud), q-speed $20.
;     Room_EnterCommon@ResetObjLoop.
;   - Coming out of a cave or dungeon onto the overworld: X = attribute byte A high nibble * $10,
;     Y = (attribute byte F bits 2-0) * $10 + $4D, facing down. If Link went in through an entrance
;     tile ($24) he starts $10 pixels lower and walks up to that Y. Room_EnterCommon.
;   - Room to room in a dungeon: X = $00 entering facing right, $F0 facing left; ObjGridOffset =
;     +/-$18 through an open or key door, +/-$28 through any other door type. Overworld: offset 0,
;     position as the scroll left it. Room_EnterCommon@ChooseWalkDistance.
;   - If the door Link came through is open (CurOpenedDoors), a close command (TriggeredDoorCmd = 2)
;     is issued for it. Room_EnterCommon@Method2.
;   - Overworld rooms $3F and $55 get the dock (object type $61) in slot $0B; other rooms get the
;     tile object found by the room layout. Room_SetUpTileObjOW.
;   - Mode 4 submodes: 1 redraw rows (only after scrolling into a dark room), 2 decide whether to
;     fade in after leaving an unlit dark room, 3 run the fade, 0 enter the room. Mode4Enter_Init.
; CONTENTS: (generated; L = line in this file)
;   L0059 Mode4Enter_Init                        code  Mode 4 (enter room) init: runs submode 1, 2 or 3, or (submode 0) the common room entry.
;   L0107 Palette_SetFadeCycleAndAdvanceSubmode  code  Starts a palette fade and advances the submode.
;   L0114 Mode4Enter_Init_Sub3                   code  Mode 4 init submode 3: runs the fade-in each frame; when it is done, falls into the next label.
;   L0119 Mode4Enter_Init_GoToSub0               code  Sets mode 4's submode to 0 (enter the room next frame) and clears CandleState.
;   L0137 Room_EnterCommon                       code  Common room entry: clears per-room state, places Link, creates the room's monsters and tile object.
;   L0402 Link_DrawBetweenRooms                  code  Draws Link for a frame between rooms; behind the background while he is leaving underground.
;   L0423 Room_SetUpTileObjOW                    code  Puts the overworld room's tile object in object slot $0B and activates it.
; ============================================================================

; ----------------------------------------------------------------------------
; Mode 4 (enter room) init: runs submode 1, 2 or 3, or (submode 0) the common room entry.
; NOTE: Mode 7 (scroll) enters at submode 1, and the order is 1 -> 2 -> (3 ->) 0. Other modes enter
;       at submode 0. Submode 1 does nothing unless CurRow >= 0, which mode 7 arranges only after
;       scrolling into a dark dungeon room (CurRow = 0); it then sends all $16 play-area rows to
;       NT 0 again, one per frame. Why that redraw is needed is (unverified).
; ----------------------------------------------------------------------------
Mode4Enter_Init:
    LDX GameSubmode
    BEQ Room_EnterCommon      ; Submode 0: the common room entry.
    DEX
    BNE @CheckSub2

    ; Submode 1: copy the play area to NT 0, one row each frame, if CurRow is valid.
    LDA CurRow
    BMI :+                      ; CurRow = $FF: nothing to copy.
    JSR Transfer_CopyNextPlayAreaRow
    BCC @Exit                   ; C clear: more rows to go.
:
    INC GameSubmode

@Exit:
    RTS

@CheckSub2:
    DEX
    BNE Mode4Enter_Init_Sub3

    ; Submode 2: decide whether the palette must fade back in.
    LDY RoomId
    JSR Room_IsDark_Bank5
    BNE Mode4Enter_Init_GoToSub0      ; This room is dark: no fade-in.

    ; This room is not dark. Find the room Link came from: the room one step from here in the
    ; direction opposite to Link's.
    LDA ObjDir
    JSR Room_CalcNextRoomByDir
    SEC
    SBC RoomId                  ; Room id offset in Link's direction ...
    JSR Math_Negate                  ; ... negated ...
    CLC
    ADC RoomId                  ; ... and applied to this room: the previous room's id.
    TAY
    JSR Room_IsDark_Bank5
    BEQ Mode4Enter_Init_GoToSub0      ; The previous room was not dark.

    ; The previous room was dark.
    LDA CandleState
    BNE Mode4Enter_Init_GoToSub0      ; It had been lit by a candle: the palette is already bright.
    LDA #$C0                    ; Fade series $C: dark to light (the reverse of series 4).

; ----------------------------------------------------------------------------
; Starts a palette fade and advances the submode.
; IN:   A = FadeCycle start value (high nibble = fade series, low nibble = step).
; ----------------------------------------------------------------------------
Palette_SetFadeCycleAndAdvanceSubmode:
    STA FadeCycle
    INC GameSubmode
:
    RTS

; Mode 4 init submode 3: runs the fade-in each frame; when it is done, falls into the next label.
Mode4Enter_Init_Sub3:
    JSR Palette_AnimateFade
    BNE :-                      ; Still fading.

; Sets mode 4's submode to 0 (enter the room next frame) and clears CandleState.
Mode4Enter_Init_GoToSub0:
    LDA #$00
    STA GameSubmode
    STA CandleState
    RTS

; ----------------------------------------------------------------------------
; Common room entry: clears per-room state, places Link, creates the room's monsters and tile object.
; IN:   RoomId; ObjDir (slot 0) = direction of travel; UndergroundExitType; GameMode.
; OUT:  RoomObjCount, RoomObjTemplateType, ObjType+1.., ObjGridOffset (slot 0), CurObjIndex = $0B;
;       the mode switches from init to update (IsUpdatingMode set by the bank 7 routine it calls).
; NOTE: Used by modes 4 (submode 0), 9 (cellar), $B (cave) and $C (shortcut). Zeroes $0300-$051F.
;       Monster list id [02] = attr C bits 5-0 (+ $40 if attr D bit 7). Count [03] =
;       LevelInfo_FoeCounts[attr C bits 7-6], or 1 for ids $32-$61. The room history then lowers the
;       count by remembered kills; in mode 9 both are zeroed. Ids < $62 fill slots 1..count with that
;       object type; ids $62-$7F copy count types from list id - $62.
;       Ends by falling into Link_DrawBetweenRooms.
; ----------------------------------------------------------------------------
Room_EnterCommon:
    JSR Sprite_DrawBetweenRooms
    JSR Link_ResetStateAndClock        ; Link's state = 0; clock item off.

    ; Zero $0300-$051F (per-room object arrays and room variables).
    LDA #$05
    LDY #$1F
    JSR Util_ClearRam0300UpTo

    ; No door is opening or closing.
    LDA #$00
    STA TriggeredDoorCmd
    STA TriggeredDoorDir

    ; Keep a copy of the room's attribute byte F.
    LDY RoomId
    LDA LevelBlockAttrsF, Y
    STA RoomAttrsF
    JSR Obj_ResetWeaponSlotsAndLadder        ; No ladder in use; weapon slots $0D-$12 inactive.

    ; Two ways into a room:
    ;   1. out of a cave, dungeon or cellar (UndergroundExitType <> 0):
    ;      a. walking out of an entrance tile, b. appearing beside stairs;
    ;   2. from the neighbouring room.
    LDA UndergroundExitType
    BEQ @Method2
    LDA CurLevel
    BNE @FaceDown               ; Leaving a cellar inside a dungeon: Link's position is not changed here.

    ; Way 1, overworld: place Link at the room's exit position.
    LDY RoomId
    LDA LevelBlockAttrsA, Y
    AND #$F0                    ; High nibble of attribute byte A: exit column; X = column * $10.
    STA ObjX
    LDA LevelBlockAttrsF, Y
    AND #$07                    ; Bits 2-0 of attribute byte F: exit row, counted from the second row.
    ASL                         ; Times $10: pixels.
    ASL
    ASL
    ASL

    ; $4D = $50 (top of the second square row of the play area) - 3 (Link's Y is 3 above the row top).
    ; C is clear here: the shifted value is at most $70.
    ADC #$4D
    STA ObjY

    ; Way 1-b: Link went underground by stairs (tiles $70-$73), not through an entrance tile ($24):
    ; he appears at the exit position without the walk-out.
    LDY UndergroundEntranceTile
    CPY #$24
    BNE @FaceDown

    ; Way 1-a: Link starts $10 pixels below the exit position and walks up to it (the mode 4 update
    ; moves him 1 pixel every 4 frames until ObjY = StairsTargetY).
    STA StairsTargetY
    CLC
    ADC #$10
    STA ObjY
    LDA #$08                    ; Stairs sound effect.
    STA EffectRequest
@FaceDown:
    ; Way 1, all cases.
    LDA #$04
    STA ObjDir                  ; Link faces down.
    LDA #$00
    STA DoorwayDir              ; Not in a doorway.
    JMP @PlaceObjects           ; A = 0: ObjGridOffset = 0.

@Method2:
    ; Way 2: room to room. Link's direction is the direction he travelled.
    LDA ObjDir
    STA DoorwayDir              ; Link is in the doorway on that axis.
    JSR Dir_GetOpposite          ; The side of this room he came in through.

    ; If the door on that side is one of the room's open true doors, command it to close.
    AND CurOpenedDoors
    STA TriggeredDoorDir
    BEQ :+
    LDA #$02                    ; Door command 2: close.
    STA TriggeredDoorCmd
:
    ; Overworld: A = 0 becomes ObjGridOffset; Link stays where the scroll left him.
    LDA CurLevel
    BEQ @PlaceObjects

    ; Dungeon. Facing right: X = $00 (left edge). Facing left: X = $F0 (right edge).
    ; Facing up or down: X is left alone.
    LDY #$00
    LDA ObjDir
    AND #$03
    BEQ @ChooseWalkDistance
    AND #$01
    BNE :+
    LDY #$F0
:
    STY ObjX

@ChooseWalkDistance:
    ; Choose how far Link walks in. Door_GetPassedType returns the type of the door on the side he
    ; came through (A) and Y = 1 if he faces right or down, else 0.
    ; Door types 0 (open), 5 and 6 (key doors): table entries 0 and 1 ($18, -$18).
    ; Other types (1 wall, 2 and 3 false wall, 4 bombable, 7 shutter): entries 2 and 3 ($28, -$28),
    ; to get clear of the wall or door.
    JSR Door_GetPassedType
    AND #$07
    BEQ :+
    CMP #$05
    BEQ :+
    CMP #$06
    BEQ :+
    INY
    INY
:
    LDA Mode4Enter_WalkInGridOffsets, Y

@PlaceObjects:
    STA ObjGridOffset           ; Link's signed walk-in distance (0 in the overworld and for way 1).
    JSR Room_SetUpObjBounds

    ; The play-mode update walks the object slots from $0B down; start it at $0B.
    LDX #$0B
    STX CurObjIndex
@ResetObjLoop:
    ; Reset object slots $0B..1 (the arrays at $0300-$051F were zeroed above).
    DEC ObjUninitialized, X     ; 0 -> $FF: the type's init routine has not run yet.
    JSR Shove_Reset          ; No knockback; returns A = 0.
    STA ObjState, X
    STA ObjDir, X
    STA ObjStunTimer, X
    INC ObjAnimCounter, X       ; 1.
    INC ObjMetastate, X         ; 1: the first spawn-cloud state.
    LDA #$20                    ; Default q-speed $20 (0.5 pixels per frame).
    STA ObjQSpeedFrac, X
    DEX
    BNE @ResetObjLoop

    ; [02] = monster list id: bits 5-0 of attribute byte C, bit 6 from bit 7 of attribute byte D.
    LDY RoomId
    LDA LevelBlockAttrsC, Y
    PHA
    AND #$3F
    STA $02
    LDA LevelBlockAttrsD, Y
    ASL                         ; Bit 7 into C.
    BCC :+
    LDA $02
    CLC
    ADC #$40
    STA $02
:
    PLA

    ; Y = bits 7-6 of attribute byte C (0-3): which of the level's four monster counts to use.
    AND #$C0
    CLC
    ROL
    ROL
    ROL
    TAY
    LDA LevelInfo_FoeCounts, Y

    ; Ids $32-$61 are bosses and other single objects: the count is 1.
    LDY $02
    CPY #$62
    BCS :+
    CPY #$32
    BCC :+
    LDA #$01
:
    STA $03                     ; [03] = monster count.

    ; Lower the count ([03]) by the kills remembered for this room; these routines also zero the
    ; list id ([02]) when nothing is left.
    LDA CurLevel
    BNE :+
    JSR RoomHistory_ModifyObjCountOW
    JMP @StoreObjCount
:
    JSR RoomHistory_ModifyObjCountUW

    ; Mode 9 (cellar): no monsters from the room's list at all.
    LDA GameMode
    CMP #$09
    BNE @StoreObjCount
    LDA #$00
    STA $02
    STA $03

@StoreObjCount:
    LDA $03
    STA RoomObjCount

    ; Count 0 or list id 0: make no monsters.
    BEQ @SkipObjects
    LDA $02
    BEQ @SkipObjects

    ; Ids $62 and up select a list of object types.
    CMP #$62
    BCS @PlaceList

    ; Ids below $62 are an object type: put it in slots 1..count.
    LDX #$00
:
    LDA $02
    STA ObjType+1, X
    INX
    DEC $03
    BNE :-
    JMP @StoreObjTemplate

@PlaceList:
    ; List index = id - $62.
    LDA $02
    SEC
    SBC #$62

    ; [04:05] = address of the list.
    ASL
    TAY
    LDA Spawn_ObjListAddrs, Y
    STA $04
    INY
    LDA Spawn_ObjListAddrs, Y
    STA $05

    ; Copy the first [03] types of the list to slots 1..count (a type of 0 leaves a slot empty).
    LDY #$00
:
    LDA ($04), Y
    STA ObjType+1, Y
    INY
    CPY $03
    BNE :-

@StoreObjTemplate:
    ; Remember the type in slot 1 (read by the dungeon kill-count save and by the food).
    LDA ObjType+1
    STA RoomObjTemplateType

@SkipObjects:
    JSR Spawn_AssignPositions ; Positions for slots 1-9; cellar Keese; cave person. Uses [02].
    LDA CurLevel
    BNE :+
    JSR Room_SetUpTileObjOW       ; Overworld: dock, or the tile object the layout found.
:
    JSR StatusBar_UpdateLinkPositionMarker

    ; Link: no stun, no knockback, animation counter 4, default speed.
    LDA #$00
    STA ObjStunTimer
    STA ObjShoveDir
    STA ObjShoveDistance
    LDA #$04
    STA ObjAnimCounter
    JSR Link_InitSpeed
    JSR Room_RunEntryTasks_MarkVisited

; ----------------------------------------------------------------------------
; Draws Link for a frame between rooms; behind the background while he is leaving underground.
; NOTE: Skips the draw in modes $B and $C; in a dungeon Link_EndMoveAndAnimateBetweenRooms returns
;       without drawing. When UndergroundExitType is nonzero (leaving a cave, cellar or dungeon)
;       Link's two sprites get background priority. Called from Sprite_DrawBetweenRooms (bank 7),
;       the death init and, by fall-through, the common room entry.
; ----------------------------------------------------------------------------
Link_DrawBetweenRooms:
    JSR Sprite_ResetCurIndex
    LDA GameMode
    CMP #$0B
    BEQ :+
    CMP #$0C
    BEQ :+
    JSR Link_EndMoveAndAnimateBetweenRooms
:
    LDA UndergroundExitType
    BEQ :+
    JSR Link_PutSpritesBehindBackground
:
    RTS

; ----------------------------------------------------------------------------
; Puts the overworld room's tile object in object slot $0B and activates it.
; NOTE: Rooms $3F and $55 always get the dock (type $61; its X and Y are not set here). Other rooms
;       get RoomTileObjType at (RoomTileObjX, RoomTileObjY), found while the room was laid out
;       (type 0 when the room has none). The RoomTileObj variables are cleared afterwards.
; ----------------------------------------------------------------------------
Room_SetUpTileObjOW:
    LDA RoomId
    CMP #$3F
    BEQ :+
    CMP #$55
    BNE @PlaceTileObj
:
    LDA #$61                    ; Dock.
    JMP @SetType

@PlaceTileObj:
    LDA RoomTileObjX
    STA ObjX+11
    LDA RoomTileObjY
    STA ObjY+11
    LDA RoomTileObjType

@SetType:
    STA ObjType+11
    JSR Room_ResetTileObjInfo    ; Returns A = 0.
    STA ObjState+11             ; State 0.
    RTS
