; ============================================================================
; FILE:     bank5/underworld-secrets.asm BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 2337-2521
; SUBJECTS: (none)
; TOPIC:    Dungeon room secrets, checked every frame of play: the "no monsters left" test, and the
;           eight secret trigger types (room attribute F bits 2-0) that open shutters, kill a
;           ringleader's followers, turn a pushed block into stairs, or make the room item appear.
; SEE ALSO: bank7/mode5-play-update.asm (calls CheckUnderworldSecrets each frame in a dungeon, then
;           CheckShutters and UpdateDoors); bank5/shutters-door-flags.asm (CheckShutters opens the
;           shutters once ShutterTrigger is set); bank4/push-block.asm (sets BlockPushComplete; the
;           block moves only when IsRoomAllDead is nonzero); bank5/create-room-objects.asm (hides
;           the room item when the trigger is 3 or 7); bank1/life-or-money.asm and
;           bank1/underworld-person-init.asm (set ShutterTrigger themselves);
;           bank7/tile-obj-tiles.asm (ChangeTileObjTiles); bank4/like-like.asm (IsLinkParalyzed).
; KEY FACTS:
;   - Secret trigger = RoomAttrsF & 7: 0 none; 1 all monsters dead -> shutters open; 2 ringleader
;     (slot $01) dead -> all other monsters die; 3 Triforce of Power taken -> shutters open; 4 block
;     pushed -> shutters open; 5 block pushed -> stairs appear; 6 life-or-money man gone -> shutters
;     open; 7 all monsters dead -> shutters open and the room item appears.
;     CheckSecretTrigger_JumpTable, CheckUnderworldSecrets.
;   - "All dead" (IsRoomAllDead) means slots $01-$0C hold no type $01-$2A or $2E-$48. Bubbles
;     ($2B-$2D) and types $49 and up (traps, persons, shots, items, tile objects) do not count.
;     CheckHasLivingMonsters.
;   - IsRoomAllDead is incremented on every frame the room is empty, not set: it wraps and reads 0
;     for one frame in every 256. CheckHasLivingMonsters.
;   - When the room is found empty, IsLinkParalyzed is cleared (frees Link from a Like-Like's hold).
;     CheckHasLivingMonsters.
;   - Ringleader: once slot $01 is empty or holds a type >= $53 (for example its dropped item),
;     every object of type $01-$52 in slots $01-$0C with ObjMetastate 0 gets ObjMetastate $10 (the
;     death sparkle). No shutter is triggered. CheckSecretTriggerRingleader.
;   - Block-to-stairs: on the first frame with BlockPushComplete = 1 it becomes 2, the tile object
;     (slot $0B) is moved to X $D0, Y $60 and its square becomes stairs tiles ($70-$73).
;     CheckSecretTriggerBlockStairs.
;   - Trigger 7's item appears only if it is still hidden (ObjState[$13] nonzero) and the room's
;     item-taken flag ($10) is clear; tune 1 bit $02 plays. CheckUnderworldSecrets.
;   - Opening shutters means ShutterTrigger = 1; the doors themselves open in CheckShutters, one at
;     a time. TriggerShutters.
; CONTENTS: (generated; L = line in this file)
;   L0061 CheckUnderworldSecrets         code  Per-frame dungeon check: updates the all-dead flag, runs the room's secret trigger, reveals item.
;   L0093 CheckSecretTriggerNone         code  Secret trigger 0 (none): returns. Also the shared return of CheckUnderworldSecrets.
;   L0101 CheckSecretTrigger             code  Jumps to the handler of a secret trigger type.
;   L0105 CheckSecretTrigger_JumpTable   data  TABLE: secret trigger handlers; index = RoomAttrsF & 7; value = routine address.
;   L0124 CheckHasLivingMonsters         code  Flags the room as empty of monsters when no counting monster is left in slots $01-$0C.
;   L0152 CheckSecretTriggerAllDead      code  Secret triggers 1 and 7: opens the shutters when no monsters are left.
;   L0158 TriggerShutters                code  Shared tail of the triggers that open shutters: ShutterTrigger = 1, returns C set.
;   L0165 ReturnFalse                    code  Shared "condition not met" exit of the trigger handlers: returns C clear.
;   L0176 CheckSecretTriggerRingleader   code  Secret trigger 2: once the ringleader in slot $01 is dead, kills every other monster in the room.
;   L0215 CheckSecretTriggerBlockDoor    code  Secret trigger 4: opens the shutters once the push block has been pushed all the way.
;   L0226 CheckSecretTriggerBlockStairs  code  Secret trigger 5: turns the tile object into stairs once the push block has been pushed.
;   L0255 CheckSecretTriggerLastBoss     code  Secret trigger 3: opens the shutters once the Triforce of Power has been taken.
;   L0265 CheckSecretTriggerMoneyOrLife  code  Secret trigger 6: opens the shutters once slot $01 (the life-or-money man) is empty.
; ============================================================================

; ----------------------------------------------------------------------------
; Per-frame dungeon check: updates the all-dead flag, runs the room's secret trigger, reveals item.
; IN:   RoomAttrsF bits 2-0 = secret trigger type; CurObjIndex = $0B (as the object loop leaves it).
; OUT:  IsRoomAllDead, IsLinkParalyzed, ShutterTrigger, and for trigger 7 ObjState[$13] = 0 (room
;       item active) with the "item appears" tune.
; NOTE: The trigger handler runs every frame, also after its condition was met.
; ----------------------------------------------------------------------------
CheckUnderworldSecrets:
    JSR CheckHasLivingMonsters

    ; If the room has no secret trigger (type 0), return.
    LDA RoomAttrsF
    AND #$07                    ; Secret trigger
    BEQ CheckSecretTriggerNone

    ; Run the trigger's handler (A = trigger type). If its condition is not met (C clear), return.
    JSR CheckSecretTrigger
    BCC CheckSecretTriggerNone

    ; Only trigger 7 ("kill all monsters for the item") has more to do.
    LDA RoomAttrsF
    AND #$07                    ; Secret trigger
    CMP #$07
    BNE CheckSecretTriggerNone

    ; If the room item is already active (ObjState[$13] = 0), or the room's item-taken flag is
    ; set, then return.
    LDA ObjState+19
    BEQ CheckSecretTriggerNone
    JSR GetRoomFlagUWItemState  ; A = $10 if the item was taken.
    BNE CheckSecretTriggerNone

    ; Else activate the room item, and play the "item appears" tune.
    LDA #$00
    STA ObjState+19
    LDA #$02
    STA Tune1Request

; Secret trigger 0 (none): returns. Also the shared return of CheckUnderworldSecrets.
CheckSecretTriggerNone:
    RTS

; ----------------------------------------------------------------------------
; Jumps to the handler of a secret trigger type.
; IN:   A = trigger type 1-7 (RoomAttrsF & 7).
; OUT:  C set when the trigger's condition is met (handlers 1-7; handler 0 leaves C unchanged).
; ----------------------------------------------------------------------------
CheckSecretTrigger:
    JSR TableJump

; TABLE: secret trigger handlers; index = RoomAttrsF & 7; value = routine address.
CheckSecretTrigger_JumpTable:
    .ADDR CheckSecretTriggerNone            ; 0: no secret
    .ADDR CheckSecretTriggerAllDead         ; 1: all monsters dead -> shutters open
    .ADDR CheckSecretTriggerRingleader      ; 2: ringleader dead -> the other monsters die
    .ADDR CheckSecretTriggerLastBoss        ; 3: Triforce of Power taken -> shutters open
    .ADDR CheckSecretTriggerBlockDoor       ; 4: block pushed -> shutters open
    .ADDR CheckSecretTriggerBlockStairs     ; 5: block pushed -> stairs appear
    .ADDR CheckSecretTriggerMoneyOrLife     ; 6: life-or-money man gone -> shutters open
    .ADDR CheckSecretTriggerAllDead         ; 7: all monsters dead -> shutters open, room item appears

; ----------------------------------------------------------------------------
; Flags the room as empty of monsters when no counting monster is left in slots $01-$0C.
; IN:   CurObjIndex ($0B when called from the play update, so the scan covers slots $0C..$01).
; OUT:  If none is found: IsLinkParalyzed = 0 and IsRoomAllDead incremented. Else nothing changes.
; NOTE: Counting types are $01-$2A and $2E-$48. Bubbles ($2B-$2D) and types $49 and up do not
;       count; neither does a dying monster's slot once its type is 0 or a dropped item ($60).
; NOTE: IsRoomAllDead is incremented on each frame the room is empty, so it wraps to 0 for one
;       frame after 256 such frames; readers that test it for zero see "monsters present" then.
; ----------------------------------------------------------------------------
CheckHasLivingMonsters:
    LDY CurObjIndex

@Loop:
    LDA ObjType+1, Y            ; Type in slot Y + 1.
    BEQ @Next                   ; Empty slot.
    CMP #$2B
    BCC @Exit                   ; Types $01-$2A: a monster is left.
    CMP #$2E
    BCC @Next                   ; Types $2B-$2D (Bubbles) do not count.
    CMP #$49
    BCC @Exit                   ; Types $2E-$48: a monster is left.
                                ; Types $49 and up do not count.

@Next:
    DEY
    BPL @Loop
    LDA #$00                    ; No monsters: release Link from a Like-Like's hold,
    STA IsLinkParalyzed
    INC IsRoomAllDead           ; and flag the room empty.

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Secret triggers 1 and 7: opens the shutters when no monsters are left.
; OUT:  C set and ShutterTrigger = 1 if IsRoomAllDead is nonzero; else C clear.
; ----------------------------------------------------------------------------
CheckSecretTriggerAllDead:
    ; If there are still monsters (Bubbles not counted), return C clear.
    LDA IsRoomAllDead
    BEQ ReturnFalse

; Shared tail of the triggers that open shutters: ShutterTrigger = 1, returns C set.
TriggerShutters:
    LDA #$01
    STA ShutterTrigger
    SEC
    RTS

; Shared "condition not met" exit of the trigger handlers: returns C clear.
ReturnFalse:
    CLC
    RTS

; ----------------------------------------------------------------------------
; Secret trigger 2: once the ringleader in slot $01 is dead, kills every other monster in the room.
; OUT:  C clear while slot $01 holds a type $01-$52; else C set and every object of type $01-$52 in
;       slots $01-$0C with ObjMetastate 0 gets ObjMetastate $10 (start of the death sparkle).
; NOTE: Objects still in their spawn cloud (ObjMetastate 1-4) are skipped until the cloud ends; the
;       handler runs again every frame. Types below $53 include Bubbles, traps and persons.
; ----------------------------------------------------------------------------
CheckSecretTriggerRingleader:
    ; If slot $01 is empty, the ringleader is gone: go kill the rest.
    LDA ObjType+1
    BEQ @KillMonsters

    ; If slot $01 still holds a type below $53 (flying rock), the ringleader lives: return C clear.
    ; Types $53 and up (shots, dead dummy, dropped item) count as gone.
    CMP #$53
    BCC ReturnFalse

@KillMonsters:
    ; For each slot from $0C down to $01: skip it if it is empty, holds a type >= $53, or is
    ; not in the normal metastate (spawning or already dying).
    LDY CurObjIndex

@Loop:
    LDA ObjType+1, Y
    BEQ @Next
    CMP #$53
    BCS @Next
    LDA ObjMetastate+1, Y
    BNE @Next

    ; Start the object's death sparkle.
    LDA #$10
    STA ObjMetastate+1, Y

@Next:
    DEY
    BPL @Loop

    ; Return C set.
    SEC
    RTS

; ----------------------------------------------------------------------------
; Secret trigger 4: opens the shutters once the push block has been pushed all the way.
; OUT:  C set and ShutterTrigger = 1 if BlockPushComplete is nonzero; else C clear.
; ----------------------------------------------------------------------------
CheckSecretTriggerBlockDoor:
    LDA BlockPushComplete
    BEQ ReturnFalse             ; Not pushed (or still moving).
    BNE TriggerShutters         ; Always taken.

; ----------------------------------------------------------------------------
; Secret trigger 5: turns the tile object into stairs once the push block has been pushed.
; OUT:  C set on the one frame the stairs are made (BlockPushComplete 1 -> 2); else C clear.
;       Slot $0B: ObjX = $D0, ObjY = $60; the square there becomes stairs tiles $70-$73.
; NOTE: Dungeon stairs made by a block always appear at X $D0, Y $60.
; ----------------------------------------------------------------------------
CheckSecretTriggerBlockStairs:
    ; If the block has not been pushed completely, return C clear.
    LDA BlockPushComplete
    BEQ ReturnFalse

    ; BlockPushComplete = 2: the stairs were already made. Return C clear.
    LSR
    BCC ReturnFalse

    ; BlockPushComplete = 1: the block has just finished moving. Make it 2.
    INC BlockPushComplete
    LDX #$0B                    ; Tile object slot

    ; Put the tile object at ($D0, $60) and change the square there to stairs.
    LDA #$D0
    STA ObjX, X
    LDA #$60
    STA ObjY, X
    LDA #$70                    ; First stairs tile; the square uses $70-$73.
    JSR ChangeTileObjTiles

    ; Return C set.
    SEC
    RTS

; ----------------------------------------------------------------------------
; Secret trigger 3: opens the shutters once the Triforce of Power has been taken.
; OUT:  C set and ShutterTrigger = 1 if InvPowerTriforce is nonzero; else C clear.
; ----------------------------------------------------------------------------
CheckSecretTriggerLastBoss:
    LDA InvPowerTriforce
    BNE TriggerShutters
    CLC
    RTS

; ----------------------------------------------------------------------------
; Secret trigger 6: opens the shutters once slot $01 (the life-or-money man) is empty.
; OUT:  C set and ShutterTrigger = 1 if ObjType of slot $01 is 0; else C clear.
; ----------------------------------------------------------------------------
CheckSecretTriggerMoneyOrLife:
    LDA ObjType+1
    BEQ TriggerShutters
    CLC
    RTS
