; ============================================================================
; FILE:     bank5/underworld-secrets.asm BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 2337-2521
; SUBJECTS: (none)
; TOPIC:    Dungeon secret triggers: all dead, ringleader, push block, last boss
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0024 CheckUnderworldSecrets         code
;   L0055 CheckSecretTriggerNone         code
;   L0061 CheckSecretTrigger             code  Returns:
;   L0064 CheckSecretTrigger_JumpTable   data
;   L0074 CheckHasLivingMonsters         code
;   L0102 CheckSecretTriggerAllDead      code
;   L0107 TriggerShutters                code
;   L0114 ReturnFalse                    code
;   L0118 CheckSecretTriggerRingleader   code
;   L0155 CheckSecretTriggerBlockDoor    code
;   L0163 CheckSecretTriggerBlockStairs  code
;   L0190 CheckSecretTriggerLastBoss     code
;   L0200 CheckSecretTriggerMoneyOrLife  code
; ============================================================================

CheckUnderworldSecrets:
    JSR CheckHasLivingMonsters

    ; If there's no secret in this room, return.
    LDA LevelBlockAttrsByteF
    AND #$07                    ; Secret trigger
    BEQ CheckSecretTriggerNone

    ; If the secret was not triggered, return.
    JSR CheckSecretTrigger
    BCC CheckSecretTriggerNone

    ; If the secret is not "foes for an item", then we're done.
    LDA LevelBlockAttrsByteF
    AND #$07                    ; Secret trigger
    CMP #$07
    BNE CheckSecretTriggerNone

    ; If the room item was already activated, or the item was
    ; already taken, then return.
    LDA ObjState+19
    BEQ CheckSecretTriggerNone
    JSR GetRoomFlagUWItemState
    BNE CheckSecretTriggerNone

    ; Else activate the room item, and play the "item appears" tune.
    LDA #$00
    STA ObjState+19
    LDA #$02
    STA Tune1Request

CheckSecretTriggerNone:
    RTS

; Returns:
; C: 1 if the condition for the secret was met
;
CheckSecretTrigger:
    JSR TableJump

CheckSecretTrigger_JumpTable:
    .ADDR CheckSecretTriggerNone
    .ADDR CheckSecretTriggerAllDead
    .ADDR CheckSecretTriggerRingleader
    .ADDR CheckSecretTriggerLastBoss
    .ADDR CheckSecretTriggerBlockDoor
    .ADDR CheckSecretTriggerBlockStairs
    .ADDR CheckSecretTriggerMoneyOrLife
    .ADDR CheckSecretTriggerAllDead

CheckHasLivingMonsters:
    ; Look for monsters in slots $C to 1 that are not bubbles.
    ; If one is found, return.
    ;
    ; Else flag Link not paralyzed (to cancel the effects of a like-like),
    ; and set the all-dead-in-room flag.
    LDY CurObjIndex

@Loop:
    LDA ObjType+1, Y
    BEQ @Next
    CMP #$2B
    BCC @Exit
    CMP #$2E
    BCC @Next
    CMP #$49
    BCC @Exit

@Next:
    DEY
    BPL @Loop
    LDA #$00
    STA LinkParalyzed
    INC RoomAllDead

@Exit:
    RTS

CheckSecretTriggerAllDead:
    ; If there are still monsters, not counting bubbles, then return C=0.
    LDA RoomAllDead
    BEQ ReturnFalse

TriggerShutters:
    ; No monsters are left. Trigger shutters to open, and return C=1.
    LDA #$01
    STA ShutterTrigger
    SEC
    RTS

ReturnFalse:
    CLC
    RTS

CheckSecretTriggerRingleader:
    ; If the first monster slot is empty, go kill all monsters.
    LDA ObjType+1
    BEQ @KillMonsters

    ; If the first monster slot has an object other than a monster,
    ; then return C=0.
    CMP #$53
    BCC ReturnFalse

@KillMonsters:
    ; For each object slot from $C to 1:
    ;
    ; If the slot is empty, or has something not a monster, or is
    ; already dying; then go loop again.
    LDY CurObjIndex

@Loop:
    LDA ObjType+1, Y
    BEQ @Next
    CMP #$53
    BCS @Next
    LDA ObjMetastate+1, Y
    BNE @Next

    ; Set the object's metastate to die.
    LDA #$10
    STA ObjMetastate+1, Y

@Next:
    DEY
    BPL @Loop

    ; Return C=1.
    SEC
    RTS

CheckSecretTriggerBlockDoor:
    ; If the block has not been pushed completely, then return C=0.
    LDA BlockPushComplete
    BEQ ReturnFalse

    ; Else go trigger shutters to open, and return C=1.
    BNE TriggerShutters

CheckSecretTriggerBlockStairs:
    ; If the block has not been pushed completely, then return C=0.
    LDA BlockPushComplete
    BEQ ReturnFalse

    ; If BlockPushComplete = 2, then it was pushed, and we already
    ; took action for the secret. So, return C=0.
    LSR
    BCC ReturnFalse

    ; Else BlockPushComplete = 1. It was pushed, but this is the
    ; first time checking it for a secret. Make it 2.
    INC BlockPushComplete
    LDX #$0B                    ; Tile object slot

    ; Stairs in UW always go at ($D0, $60).
    LDA #$D0
    STA ObjX, X
    LDA #$60
    STA ObjY, X
    LDA #$70                    ; Stairs tile
    JSR ChangeTileObjTiles

    ; Return C=1.
    SEC
    RTS

CheckSecretTriggerLastBoss:
    ; If the last boss was defeated, then
    ; go trigger shutters to open, and return C=1.
    LDA LastBossDefeated
    BNE TriggerShutters

    ; Else return C=0.
    CLC
    RTS

CheckSecretTriggerMoneyOrLife:
    ; If the money-or-life man is gone, then
    ; go trigger shutters to open, and return C=1.
    LDA ObjType+1
    BEQ TriggerShutters

    ; Else return C=0.
    CLC
    RTS
