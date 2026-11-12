; ============================================================================
; FILE:     bank1/trap-rupee-stash.asm   BANK: 1 (switchable, $8000-$BFFF)
; UPSTREAM: Z_01.asm 2366-2677
; SUBJECTS: Obj, RupeeStash, Trap
; TOPIC:    Dungeon blade traps (object types $49 and $4A: one generator object becomes six or four
;           traps at fixed spots) and the room of ten rupees (type $35), then $FF padding up to the
;           common code block.
; SEE ALSO: bank7/object-stubs-destroy-monster.asm (Trap_Init, Trap_Update, RupeeStash_Init,
;           RupeeStash_Update: the fixed-bank trampolines that switch in bank 1 and jump here);
;           bank1/monster-collisions.asm (Person_DrawAndCheckCollisions_Common, the trap's collision
;           and draw tail; Damage_ObjTypeDamageToLink); bank1/link-collision-harm.asm (Link_CheckMonsterTouch,
;           Link_HarmByObjType); bank7/walker-move.asm (Obj_MoveByQSpeed); bank1/qspeed-direction-math.asm (q-speed
;           arithmetic, Math_Abs, Dir_GetOpposite); bank5/enter-room.asm (Room_EnterCommon: why the
;           generator is in object slot 1); bank5/room-history-uw.asm (RoomHistory_SaveKillCountUW: why a touched
;           rupee stash never returns); bank1/take-item.asm (Item_TakeOneRupee).
; KEY FACTS:
;   - Monster list id $49 makes six traps, $4A four. The generator sits in object slot 1 (a list id in
;     $32-$61 makes one object, Room_EnterCommon), so the traps are slots 1-6 or 1-4, all of type
;     $49. Trap_Init_Full.
;   - Start spots (X, Y): slots 1-4 the corners ($20,$5D), ($20,$BD), ($D0,$5D), ($D0,$BD); slots 5-6
;     ($40,$8D), ($B0,$8D). Trap_StartXs, Trap_StartYs.
;   - A resting trap charges horizontally when Link's Y is less than $0E pixels from its own, else
;     vertically when Link's X is less than $0E pixels from its own, and only in a direction its slot
;     allows: corner traps toward the room's middle on either axis, slots 5-6 inward horizontally
;     only. Trap_Update_Full, Trap_AllowedDirs.
;   - Charge: q-speed $70 (1.75 pixels a frame at most) until less than 5 pixels from X = $78 or
;     Y = $90. Return: q-speed $20 (0.5 pixels a frame) until exactly back at the start coordinate.
;     Only then can it charge again. Trap_Update_Full.
;   - Link's weapons cannot touch a trap (ObjAttr $A1 has bit $20), so it cannot be killed or stunned.
;     Touching one costs 1 heart before rings (damage byte $01). Trap_Update_Full,
;     Damage_ObjTypeDamageToLink.
;   - A trap still senses and moves while Link holds the clock or is invincible; it only cannot hurt
;     him then (Link_CheckMonsterTouch reports no collision). Trap_Update_Full.
;   - Rupee stash: the object of type $35 in slot 1 fills slots $01-$0A with ten rupees in a diamond.
;     A rupee is taken when Link's X and Y are both less than 9 pixels from its own: +1 rupee each.
;     RupeeStash_Init_Full, RupeeStash_Update_Full.
;   - Taking one rupee zeroes RoomObjCount, so on leaving the room is saved as cleared and the stash
;     is never made again: rupees left behind are lost. RupeeStash_Update_Full.
; CONTENTS: (generated; L = line in this file)
;   L0055 RupeeStash_RupeeXs           data  TABLE: X of each rupee of the rupee stash; index = object slot - 1 (slots $01-$0A); value = ObjX
;   L0061 RupeeStash_RupeeYs           data  TABLE: Y of each rupee of the rupee stash; index = object slot - 1 (slots $01-$0A); value = ObjY
;   L0075 RupeeStash_Init_Full         code  Turns the rupee stash object (type $35) into ten rupee objects in object slots $0A-$01.
;   L0100 Obj_InitSimpleObject         code  Gives object slot X a type and attributes and marks it as already initialized.
;   L0112 Trap_StartXs                 data  TABLE: start X of each trap; index = trap number 0-5 (object slot - 1 with the generator in slot
;   L0117 Trap_StartYs                 data  TABLE: start Y of each trap; index = trap number 0-5; value = ObjY (pixels): $5D top, $BD bottom,
;   L0134 Trap_Init_Full               code  Turns the trap generator (type $49 or $4A) into six or four traps of type $49 at fixed spots.
;   L0167 Trap_AllowedDirs             data  TABLE: directions a trap may charge in; index = object slot - 1 (0-5); value = direction bits
;   L0189 Trap_Update_Full             code  Updates one blade trap: waits for Link to line up, charges to the room's middle, creeps back.
;   L0323 Trap_DrawAndCheckCollisions  code  Shared tail of Trap_Update_Full: tests the collision with Link and draws the trap.
;   L0338 RupeeStash_Update_Full       code  Updates one rupee of the rupee stash: Link takes it by touching it, otherwise it is drawn.
; ============================================================================

; TABLE: X of each rupee of the rupee stash; index = object slot - 1 (slots $01-$0A); value = ObjX
; (pixels). With RupeeStash_RupeeYs the ten rupees form a diamond: rows of 1, 2, 4, 2 and 1.
RupeeStash_RupeeXs:
    .BYTE $78, $70, $80, $60, $70, $80, $90, $70
    .BYTE $80, $78

; TABLE: Y of each rupee of the rupee stash; index = object slot - 1 (slots $01-$0A); value = ObjY
; (pixels): rows at $70, $80, $90, $A0 and $B0.
RupeeStash_RupeeYs:
    .BYTE $70, $80, $80, $90, $90, $90, $90, $A0
    .BYTE $A0, $B0

; ----------------------------------------------------------------------------
; Turns the rupee stash object (type $35) into ten rupee objects in object slots $0A-$01.
; IN:   X = object slot of the stash object (ObjAttr,X already set from the type's attributes)
; OUT:  slots $01-$0A: ObjType = $35, ObjAttr = the stash's, ObjUninitialized = 0, ObjX and ObjY
;       from the tables; X = 0; [00] = $35; [01] = the attributes
; NOTE: The stash is the room's only object and sits in slot 1 (see Room_EnterCommon: a monster
;       list id in $32-$61 makes one object), so it overwrites itself with the first rupee. Every
;       rupee then shows the spawn cloud (room entry set ObjMetastate = 1 in slots $01-$0B) before
;       RupeeStash_Update_Full runs for it.
; ----------------------------------------------------------------------------
RupeeStash_Init_Full:
    LDA ObjAttr, X
    STA $01                     ; [01] = object attributes for every rupee.
    LDA #$35
    STA $00                     ; [00] = object type $35 (rupee stash) for every rupee.

    ; Make ten rupees, from slot $0A down to slot 1.
    LDX #$0A

@LoopRupee:
    JSR Obj_InitSimpleObject
    LDA RupeeStash_RupeeXs-1, X       ; The tables are indexed by slot - 1.
    STA ObjX, X
    LDA RupeeStash_RupeeYs-1, X
    STA ObjY, X
    DEX
    BNE @LoopRupee
    RTS

; ----------------------------------------------------------------------------
; Gives object slot X a type and attributes and marks it as already initialized.
; IN:   X = object slot; [00] = object type; [01] = object attributes (ObjAttr value)
; OUT:  A = [01]; position, state and HP of the slot are not touched
; NOTE: With ObjUninitialized = 0 the type's init routine is not run for this slot (Obj_Update).
; ----------------------------------------------------------------------------
Obj_InitSimpleObject:
    LDA $00
    STA ObjType, X
    LDA #$00
    STA ObjUninitialized, X
    LDA $01
    STA ObjAttr, X
    RTS

; TABLE: start X of each trap; index = trap number 0-5 (object slot - 1 with the generator in slot
; 1); value = ObjX (pixels). Entries 0-3 are the corner traps made by both types, 4-5 the two extra
; traps of type $49.
Trap_StartXs:
    .BYTE $20, $20, $D0, $D0, $40, $B0

; TABLE: start Y of each trap; index = trap number 0-5; value = ObjY (pixels): $5D top, $BD bottom,
; $8D the row of the two extra traps.
Trap_StartYs:
    .BYTE $5D, $BD, $5D, $BD, $8D, $8D

; ----------------------------------------------------------------------------
; Turns the trap generator (type $49 or $4A) into six or four traps of type $49 at fixed spots.
; IN:   X = object slot of the generator; CurObjIndex = the same slot
; OUT:  slots CurObjIndex+0..+5 (generator type $49) or +0..+3 (type $4A): ObjType = $49, ObjAttr =
;       the generator's, ObjUninitialized = 0, ObjX and ObjY from Trap_StartXs and Trap_StartYs; [00] = $49;
;       [01] = the attributes; X = CurObjIndex - 1 (not restored)
; NOTE: The generator sits in slot 1: either the room's only object (a monster list id in $32-$61
;       makes one object, see Room_EnterCommon) or the first entry of a monster list that leaves
;       slots 2-4 empty and puts other monsters in slots 5 and up (lists $6D, $6E, $76, $77), so the
;       traps are slots 1-6 or 1-4. Trap_AllowedDirs is indexed by slot - 1 and fits only that
;       arrangement.
; NOTE: Every trap is type $49, also the four made by a $4A generator. Each shows the spawn cloud
;       first (room entry set ObjMetastate = 1 in slots $01-$0B).
; ----------------------------------------------------------------------------
Trap_Init_Full:
    LDA ObjAttr, X
    STA $01                     ; [01] = object attributes for every trap.

    ; Y = number of the last trap to make: 5 (six traps) for type $49, 3 (four traps) otherwise.
    LDY #$05
    LDA #$49
    STA $00                     ; [00] = object type $49 for every trap.
    CMP ObjType, X
    BEQ @LoopTrap
    LDY #$03                    ; The generator is type $4A.

@LoopTrap:
    ; Trap number Y goes in slot CurObjIndex + Y.
    TYA
    CLC
    ADC CurObjIndex
    TAX
    LDA Trap_StartXs, Y
    STA ObjX, X
    LDA Trap_StartYs, Y
    STA ObjY, X
    JSR Obj_InitSimpleObject
    DEX
    DEY
    BPL @LoopTrap
    RTS

; TABLE: directions a trap may charge in; index = object slot - 1 (0-5); value = direction bits
; (1 right, 2 left, 4 down, 8 up). Each trap may only go toward the middle of the room:
;   slot 1 ($20,$5D) $05 right or down      slot 2 ($20,$BD) $09 right or up
;   slot 3 ($D0,$5D) $06 left or down       slot 4 ($D0,$BD) $0A left or up
;   slot 5 ($40,$8D) $01 right              slot 6 ($B0,$8D) $02 left
Trap_AllowedDirs:
    .BYTE $05, $09, $06, $0A, $01, $02

; ----------------------------------------------------------------------------
; Updates one blade trap: waits for Link to line up, charges to the room's middle, creeps back.
; IN:   X = object slot of the trap (1-6)
; OUT:  ObjState,X = 0 resting, 1 charging, 2 returning; ObjDir,X = direction of the charge or of
;       the return; ObjMovingLimit,X = the trap's start coordinate on the axis it moves along
; NOTE: State 0 compares ObjX and ObjY of Link and the trap (not midpoints). If |Link Y - trap Y| <
;       $0E and the X's differ, the trap wants to go right or left toward Link. Otherwise, if |Link X
;       - trap X| < $0E and the Y's differ, it wants to go down or up toward Link. It charges only if
;       that direction is in Trap_AllowedDirs for its slot; a refused horizontal wish is not followed
;       by the vertical test.
; NOTE: State 1: q-speed $70 (1.75 pixels a frame at most: the sub-steps left in a frame after the
;       grid offset reaches $10 or -$10 move nothing) until the trap is less than 5 pixels from
;       X = $78 (horizontal charge) or Y = $90 (vertical charge); then it turns round. State 2:
;       q-speed $20 (0.5 pixels a frame) until its coordinate equals the start coordinate; then
;       state 0. A trap does not react to Link again before it is back at its start.
; NOTE: Link's weapons cannot touch a trap (type attributes $A1 include bit $20). It harms Link
;       through Link_CheckMonsterTouch: damage byte $01 = 1 heart, halved per ring grade. With InvClock
;       set, or while Link is invincible, there is no collision, but the trap still senses and moves.
; ----------------------------------------------------------------------------
Trap_Update_Full:
    LDA ObjState, X
    BNE @UpdateMovingStates

    ; State 0: resting, sensing Link.
    ;
    ; If Link's Y is $0E or more pixels from the trap's Y, Link is not in the trap's row:
    ; go test its column.
    LDA ObjY
    SEC
    SBC ObjY, X
    JSR Math_Abs
    CMP #$0E
    BCS @CheckColumn

    ; Link is in the trap's row. Y = direction toward Link: 1 (right) if Link's X is greater, 2
    ; (left) if smaller. With equal X's there is no horizontal direction: go test the column.
    LDY #$01
    LDA ObjX
    CMP ObjX, X
    BEQ @CheckColumn
    BCS :+
    LDY #$02
:
    ; A = the trap's X: the coordinate to come back to.
    LDA ObjX, X

@TryCharge:
    ; A = start coordinate on the axis of the charge; Y = wanted direction.
    ; Both are stored even when the charge is refused below.
    STA ObjMovingLimit, X
    TYA
    STA ObjDir, X

    ; If the wanted direction is not allowed for this slot, stay in state 0.
    AND Trap_AllowedDirs-1, X
    BEQ @DrawAndCheckCollisions

    ; Go to state 1 (charging) at q-speed $70 (1.75 pixels a frame).
    INC ObjState, X
    LDA #$70
    STA ObjQSpeedFrac, X

@DrawAndCheckCollisions:
    JMP Trap_DrawAndCheckCollisions

@CheckColumn:
    ; Link is not in the trap's row. If Link's X is $0E or more pixels from the trap's X, Link is
    ; not in its column either: nothing triggers the trap.
    LDA ObjX
    SEC
    SBC ObjX, X
    JSR Math_Abs
    CMP #$0E
    BCS @DrawAndCheckCollisions

    ; Link is in the trap's column. Y = direction toward Link: 4 (down) if Link's Y is greater, 8
    ; (up) if smaller. With equal Y's there is no direction: nothing triggers the trap.
    LDY #$04
    LDA ObjY
    CMP ObjY, X
    BEQ @DrawAndCheckCollisions
    BCS :+
    LDY #$08
:
    ; A = the trap's Y: the coordinate to come back to. Never 0, so the branch is always taken.
    LDA ObjY, X
    BNE @TryCharge

@UpdateMovingStates:
    ; States 1 (charging) and 2 (returning): move in the facing direction at the q-speed set when
    ; the state began.
    LDA ObjDir, X
    STA $0F
    JSR Obj_MoveByQSpeed

    ; Obj_MoveByQSpeed stops moving an object whose grid offset is $10 or -$10. Set the offset back to 0
    ; whenever its low nibble is 0, so the trap keeps going.
    LDA ObjGridOffset, X
    AND #$0F
    BNE :+
    STA ObjGridOffset, X
:
    ; Test the collision with Link here; the shared tail below tests it once more. (Link's weapons
    ; are never tested against a trap: its attributes have bit $20.)
    JSR Link_CheckMonsterTouch

    ; Y = the trap's coordinate on its moving axis, [00] = the turning point on that axis:
    ; X and $78 when moving horizontally, Y and $90 when moving vertically.
    LDY ObjX, X
    LDA #$78
    STA $00
    LDA ObjDir, X
    AND #$0C                    ; Direction bits down and up.
    BEQ @CheckState
    LDY ObjY, X
    LDA #$90
    STA $00

@CheckState:
    ; Odd state = 1 (charging); even = 2 (returning).
    LDA ObjState, X
    AND #$01
    BEQ @FinishState2

    ; State 1: keep charging while the trap is 5 or more pixels from the turning point.
    TYA
    SEC
    SBC $00
    JSR Math_Abs
    CMP #$05
    BCS @Draw

    ; Less than 5 pixels from the turning point: turn round and go to state 2 (returning) at
    ; q-speed $20 (0.5 pixels a frame).
    LDA ObjDir, X
    JSR Dir_GetOpposite
    STA ObjDir, X
    LDA #$20
    STA ObjQSpeedFrac, X
    INC ObjState, X

@Draw:
    JMP Trap_DrawAndCheckCollisions

@FinishState2:
    ; State 2: when the coordinate equals the start coordinate again, go back to state 0.
    TYA
    CMP ObjMovingLimit, X
    BNE Trap_DrawAndCheckCollisions
    LDA #$00
    STA ObjState, X

; Shared tail of Trap_Update_Full: tests the collision with Link and draws the trap.
Trap_DrawAndCheckCollisions:
    JMP Person_DrawAndCheckCollisions_Common

; ----------------------------------------------------------------------------
; Updates one rupee of the rupee stash: Link takes it by touching it, otherwise it is drawn.
; IN:   X = object slot of the rupee ($01-$0A)
; OUT:  X preserved. When taken: InvRupeesToAdd + 1, tune 1 request $01, the slot emptied
;       (ObjType = 0) and RoomObjCount = 0.
; NOTE: Taken when |Link X - rupee X| < 9 and |Link Y - rupee Y| < 9 (ObjX and ObjY of both). No
;       button, no weapon and no other state is tested. Each rupee is worth 1.
; NOTE: RoomObjCount = 0 makes the room count as cleared when Link leaves it (RoomHistory_SaveKillCountUW
;       stores the maximum kill count for a room with RoomObjCount 0), and a cleared room does not
;       make a type $35 object again. So after the first rupee is taken, the rupees still lying
;       there are lost once Link leaves. A stash left untouched is made again in full.
; ----------------------------------------------------------------------------
RupeeStash_Update_Full:
    TXA                         ; Save the current object slot.
    PHA

    ; If Link is 9 or more pixels away on either axis, only draw the rupee.
    LDA ObjY
    SEC
    SBC ObjY, X
    JSR Math_Abs
    CMP #$09
    BCS @DrawRupee
    LDA ObjX
    SEC
    SBC ObjX, X
    JSR Math_Abs
    CMP #$09
    BCS @DrawRupee

    ; Link touches the rupee: post 1 rupee to add, and empty this slot (no drop, no kill counted).
    JSR Item_TakeOneRupee
    JSR Obj_DestroyMonster

    ; From now on the room counts as cleared when Link leaves it (see the header).
    LDA #$00
    STA RoomObjCount
    JMP @ExitRestoreX

@DrawRupee:
    ; Draw the rupee item at the object's position. Item slot $16 is one of the flashing slots:
    ; its palette alternates every 8 frames and the inventory value is not used (Item_DrawBySlot).
    JSR Anim_FetchObjPosForSpriteDescriptor
    LDX #$16                    ; X = item slot $16 (rupee): selects the palette rule.
    LDY #$16                    ; Y = item slot $16 (rupee): selects the sprite tiles.
    JSR Item_DrawInInventory

@ExitRestoreX:
    PLA                         ; Restore the current object slot.
    TAX
    RTS

; DATA: filler ($FF), 221 bytes ($A423-$A4FF), up to where the common code block is stored ($A500).
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF
