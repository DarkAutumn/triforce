; ============================================================================
; FILE:     bank4/shoot-helpers-candle.asm BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 11311-11445
; SUBJECTS: Candle, Room, Shooter
; TOPIC:    How a monster creates a shot or a child object in a free monster slot (with the limit of 4
;           counted shots), and the fade that lights a dark dungeon room when candle fire burns in it;
;           this bank's dark-room test.
; SEE ALSO: bank4/goriya.asm, bank4/moblin-lynel-shooting.asm, bank4/stalfos.asm (callers of
;           Shooter_ShootIfWanted); bank4/zol-gel.asm (Shooter_ShootLimited makes child Gels);
;           bank4/vire.asm (Shooter_Shoot makes red Keese); bank4/monster-shots-fireballs.asm
;           (MonsterShot_DestroyCounted gives the count back); bank7/object-stubs-destroy-monster.asm
;           (Obj_FindEmptyMonsterSlot, Obj_SetTypeAndClear);
;           bank7/fire.asm (Fire_Update calls Candle_UpdateRoomLight); bank7/mode5-play-update.asm (Mode5Play_Update runs
;           only the fade while IsRoomBrightening); bank4/ganon.asm (Ganon_ScenePhase0_LiftInDark lights his room);
;           bank1/world-fading.asm (Palette_AnimateFade); bank5/mode7-scroll-init.asm, bank5/enter-room.asm
;           (CandleState reset and use on room change).
; KEY FACTS:
;   - At most 4 counted monster shots exist at once: a new object of type >= $53 is refused while
;     MonsterShotCount >= 4. Types below $53 (child Gel) are not counted. Shooter_ShootLimited
;   - A shot needs a free object slot in $01-$0B; the highest free one is used. Shooter_ShootLimited
;   - A walking shooter shoots only while Shooter_ObjWantsToShoot,X <> 0. Shooter_ShootIfWanted
;   - The new object starts in state $10 with timer 0, at the position and with the direction of the
;     object in CurObjIndex. Shooter_Shoot
;   - Shooter_Shoot itself tests nothing and counts nothing; the caller must have set
;     EmptyMonsterSlot. Shooter_Shoot
;   - A dark room is a dungeon room (CurLevel <> 0) with bit 7 of LevelBlockAttrsE set; the overworld
;     is never dark. Room_IsDark_Bank4
;   - CandleState: 0 not lit, 1 fading to light, 2 lit. In a dark room the first call starts fade
;     cycle $C0 and sets IsRoomBrightening; in a room that is not dark nothing happens.
;     Candle_State0Begin
;   - The fade has 4 palette steps 10 frames apart; when it ends IsRoomBrightening is cleared and the
;     state becomes 2, after which further fires do nothing. Candle_State1Brightening,
;     Candle_StopBrightening
; CONTENTS: (generated; L = line in this file)
;   L0056 Shooter_ShootIfWanted             code  Makes a shot of type A for the monster in slot X, if that monster currently wants to shoot.
;   L0070 Shooter_ShootLimited              code  Makes an object of type [00] in a free monster slot, enforcing the limit of 4 counted shots.
;   L0094 Shooter_Shoot                     code  Makes an object of type [00] in slot EmptyMonsterSlot at the shooter's position and direction.
;   L0120 Shooter_ReturnDidNotShoot         code  Shared failure exit of the three routines above: returns C = 0.
;   L0132 Candle_UpdateRoomLight            code  Runs the dark-room lighting state machine (CandleState): start the fade, run it, or do nothing.
;   L0137 Candle_UpdateRoomLight_JumpTable  data  TABLE: handlers of Candle_UpdateRoomLight; index = CandleState 0-2.
;   L0147 Candle_State0Begin                code  CandleState 0: in a dark room, starts the fade to light (cycle $C0) and goes to state 1.
;   L0156 Candle_IncState                   code  Advances CandleState to the next state (shared by states 0 and 1).
;   L0160 Candle_State2Done                 code  CandleState 2 (room lit): nothing to do. Also the shared return of the candle states.
;   L0164 Candle_StopBrightening            code  End of the fade: clears IsRoomBrightening and goes to CandleState 2.
;   L0173 Candle_State1Brightening          code  CandleState 1: steps the fade to light; when it is finished, ends the brightening.
;   L0184 Room_IsDark_Bank4                 code  Tests whether a room is a dark dungeon room.
; ============================================================================

; ----------------------------------------------------------------------------
; Makes a shot of type A for the monster in slot X, if that monster currently wants to shoot.
; IN:   A = object type of the shot; X = shooter's object slot (must equal CurObjIndex).
; OUT:  C = 1 and Y = the shot's slot when made; C = 0 when not. [00] = shot type.
; NOTE: Fails when Shooter_ObjWantsToShoot,X = 0; otherwise falls into Shooter_ShootLimited and can fail for
;       its reasons. The flag is not cleared here.
; ----------------------------------------------------------------------------
Shooter_ShootIfWanted:
    STA $00

    ; If the monster does not want to shoot, then return C=0.
    LDA Shooter_ObjWantsToShoot, X
    BEQ Shooter_ReturnDidNotShoot

; ----------------------------------------------------------------------------
; Makes an object of type [00] in a free monster slot, enforcing the limit of 4 counted shots.
; IN:   [00] = object type to make; CurObjIndex = the shooter's object slot.
; OUT:  C = 1 and Y = the new slot when made; C = 0 when not. MonsterShotCount +1 for types >= $53.
; NOTE: Fails when no slot in $01-$0B is free, or when the type is >= $53 (a projectile) and
;       MonsterShotCount is already 4. Types below $53 are neither limited nor counted.
; ----------------------------------------------------------------------------
Shooter_ShootLimited:
    ; Z set = no free slot: return C=0.
    JSR Obj_FindEmptyMonsterSlot
    BEQ Shooter_ReturnDidNotShoot

    ; Types below $53 are not projectiles: make the object without counting it.
    ; For a projectile, return C=0 if 4 counted shots are already in flight.
    LDA $00
    CMP #$53
    BCC Shooter_Shoot
    LDA MonsterShotCount
    CMP #$04
    BCS Shooter_ReturnDidNotShoot

    ; Else count the new shot.
    INC MonsterShotCount

; ----------------------------------------------------------------------------
; Makes an object of type [00] in slot EmptyMonsterSlot at the shooter's position and direction.
; IN:   [00] = object type; EmptyMonsterSlot = free slot; CurObjIndex = the shooter's object slot.
; OUT:  C = 1; Y = the new slot; X = CurObjIndex.
; NOTE: No limit and no counting. The new object starts in state $10 with timer 0; the slot is
;       otherwise cleared by Obj_SetTypeAndClear (flagged uninitialized, metastate 1).
; ----------------------------------------------------------------------------
Shooter_Shoot:
    LDX EmptyMonsterSlot
    LDA $00
    JSR Obj_SetTypeAndClear

    ; Set state to $10 and reset timer, so that the shot moves right away.
    LDY EmptyMonsterSlot
    LDX CurObjIndex
    LDA #$10
    STA a:ObjState, Y
    LDA #$00
    STA a:ObjTimer, Y

    ; Copy the shooter's direction and coordinates to the shot.
    LDA ObjDir, X
    STA a:ObjDir, Y
    LDA ObjX, X
    STA a:ObjX, Y
    LDA ObjY, X
    STA a:ObjY, Y

    ; Return C=1.
    SEC
    RTS

; Shared failure exit of the three routines above: returns C = 0.
Shooter_ReturnDidNotShoot:
    CLC
    RTS

; ----------------------------------------------------------------------------
; Runs the dark-room lighting state machine (CandleState): start the fade, run it, or do nothing.
; OUT:  X and Y changed (callers save X).
; NOTE: Called by a standing candle fire in a dungeon every frame, by the play-mode update on every
;       frame while IsRoomBrightening is set (nothing else of the play update runs on those frames),
;       and by Ganon's opening scene. CandleState is zeroed by the room-change code, so a lit room is
;       dark again on the next visit.
; ----------------------------------------------------------------------------
Candle_UpdateRoomLight:
    LDA CandleState
    JSR Util_TableJump

; TABLE: handlers of Candle_UpdateRoomLight; index = CandleState 0-2.
Candle_UpdateRoomLight_JumpTable:
    .ADDR Candle_State0Begin            ; 0: room not lit yet
    .ADDR Candle_State1Brightening      ; 1: fading to light
    .ADDR Candle_State2Done             ; 2: lit

; ----------------------------------------------------------------------------
; CandleState 0: in a dark room, starts the fade to light (cycle $C0) and goes to state 1.
; NOTE: In a room that is not dark it returns with the state still 0, so it is tested again on
;       every call.
; ----------------------------------------------------------------------------
Candle_State0Begin:
    LDY RoomId
    JSR Room_IsDark_Bank4
    BEQ Candle_State2Done       ; If it's not a dark room, return.
    LDA #$C0                    ; Fade series $40 (dark room) run backwards (bit 7): dark to light.
    STA FadeCycle
    INC IsRoomBrightening

; Advances CandleState to the next state (shared by states 0 and 1).
Candle_IncState:
    INC CandleState

; CandleState 2 (room lit): nothing to do. Also the shared return of the candle states.
Candle_State2Done:
    RTS

; End of the fade: clears IsRoomBrightening and goes to CandleState 2.
Candle_StopBrightening:
    LDA #$00
    STA IsRoomBrightening
    BEQ Candle_IncState       ; Always taken.

; ----------------------------------------------------------------------------
; CandleState 1: steps the fade to light; when it is finished, ends the brightening.
; NOTE: Palette_AnimateFade returns Z = 1 after the 4th palette step; steps are 10 frames apart.
; ----------------------------------------------------------------------------
Candle_State1Brightening:
    JSR Palette_AnimateFade
    BEQ Candle_StopBrightening    ; If done fading, go stop brightening the room.
    RTS

; ----------------------------------------------------------------------------
; Tests whether a room is a dark dungeon room.
; IN:   Y = room id.
; OUT:  A = $80 (Z clear) if dark; A = 0 (Z set) if not, and always in the overworld (CurLevel = 0).
; NOTE: Bank 4's copy of the dark-room test; the flag is bit 7 of LevelBlockAttrsE.
; ----------------------------------------------------------------------------
Room_IsDark_Bank4:
    LDA CurLevel
    BEQ :+
    LDA LevelBlockAttrsE, Y
    AND #$80
:
    RTS
