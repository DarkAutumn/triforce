; Aliases of object slot bytes shared by several banks (monster throwers, captors, items, room bookkeeping).
; "per slot" = indexed by object slot (X or Y register). "scalar" = one fixed byte.

; ---- Thrower and shot links (Goriya and its boomerang, monster arrows) ----
ObjRefId := $42C                    ; per slot; slot of the partner object: thrower's shot or shot's thrower.
;   Set on both objects when a monster shoots (goriya code). A dying Goriya uses it to destroy its
;   boomerang; the returning boomerang uses it to find the thrower and restart its timer and state; a
;   monster arrow flies at once when the slot it names is empty. Same byte as ObjCaptureTimer.

; ---- Captors (Wallmaster $27, Like-Like $17) ----
ObjCaptureTimer := $42C             ; per slot; Wallmaster, Like-Like: 0 = Link free, nonzero = Link captured.
;   Collision_CheckMonsterVsWeaponsAndLink increments it when the monster touches Link. Like-Like then adds 1 per frame; at
;   $60 it removes the magic shield (InvMagicShield := 0) and holds the value at $C0. Wallmaster only tests
;   it for nonzero: at the end of its trip it sends Link to the level entrance (mode 3).

; ---- Gohma's eye (types $33, $34; set by the Gohma code in bank 4, read by the knockback code, bank 1) ----
Gohma_ObjEyeState := $46B           ; per slot; Gohma: eye frame image 0-3; an arrow harms Gohma only at 3.
;   0/1 closed (alternating every 8 frames); 2 while the open timer is >= $70 or < $10; 3 in between.

; ---- Items ----
ObjRoomItemId := $98                ; alias of ObjDir; with X = $13 (room item slot) it is RoomItemId ($AB).
Item_ObjItemId := $AC               ; per slot; dropped item: item id; $FF = taken. Alias of ObjState.
;   $23 (fairy) makes the object fly as an item fairy.
Item_ObjItemLifetime := $3A8        ; per slot; dropped or room item: life countdown. Alias of ObjPosFrac.
;   A dropped item starts at $FF, loses 1 every other frame and disappears at 0. While the value is >= $F0
;   the item flashes (drawn on odd values only) and cannot be picked up (Item_TryTake).
Item_ObjMonsterType := $412         ; per slot; object type of the monster that died here, for the drop.
;   Copied from ObjType when the death ends (metastate $14); Drop_SetUpItem compares it with
;   Drop_NoDropTypes and uses it to choose the drop.

; ---- Room bookkeeping ----
RoomKillCount := $34F               ; scalar; monsters killed in the room since it was entered.
;   +1 when a monster's death ends (metastate $14), except dead dummies ($5D), child Gels ($14), red Keese
;   ($1C) and Zoras ($11). On leaving, it is merged into the room's saved kill count (RoomHistory_SaveKillCountOW, the
;   dungeon room history code). It is slot 0's byte of ObjType.
StairsTargetY := $412               ; scalar; Y where Link's walk down or up stairs ends (modes $10 and 4).
;   Link moves 1 pixel every 4 frames until ObjY equals it. Going down: ObjY + $10. Coming out of a cave
;   entrance: the exit Y, with Link starting $10 pixels below. Slot 0's byte of the $412 array.
