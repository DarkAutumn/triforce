; ============================================================================
; FILE:     bank4/bubble-keese.asm       BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 1095-1237
; SUBJECTS: Bubble, Flyer, Keese, Obj
; TOPIC:    Bubble (types $2B flashing, $2C, $2D): a wanderer that blocks or unblocks Link's sword on
;           touch. Keese (types $1B blue, $1C red, $1D black): init, update and the flight state table.
;           Also a shared animate-and-draw helper for simple two-frame monsters.
; SEE ALSO: bank4/walker-wanderer-common.asm (Wanderer_Update, Walker_Init: Bubble movement);
;           bank4/fairy-flyer-common.asm (Flyer_State0SpeedUp, Flyer_State2Chase, Flyer_State3Wander, Flyer_State4SlowDown,
;           Flyer_State5Delay, Flyer_Move: the flight states and movement Keese share with other flyers);
;           bank5/link-handle-input.asm (where SwordBlockedLongTimer and IsSwordBlocked stop the sword);
;           bank1/link-collision-harm.asm (Link_CheckMonsterTouch, Link_BeHarmed).
; KEY FACTS:
;   - A Bubble never tests Link's weapons: Bubble_Update calls only Link_CheckMonsterTouch, so nothing
;     harms, stuns or shoves it.
;   - Touching the flashing Bubble ($2B) sets SwordBlockedLongTimer to $10 (10-frame units: 151 to
;     160 frames without the sword; a new touch reloads it) (Bubble_Update).
;   - Touching Bubble $2D sets IsSwordBlocked to 1 (sword disabled until cured); touching Bubble $2C
;     sets it to 0 (Bubble_Update@BlockOrUnblock).
;   - Bubbles walk at q-speed $40 (1 px/frame) with turn rate $40 (Bubble_Init, Bubble_Update).
;   - Keese do not move while InvClock or ItemLiftTimer is nonzero, but still collide (Keese_Update).
;   - Keese top speed is $C0/256 = 0.75 px/frame per axis (FlyerMaxSpeedFrac set by Keese_InitBlue).
;     Blue Keese start at speed $1F, red and black at $7F; the difference lasts only until the first
;     slow-down, which takes every Keese below $20 (Keese_InitBlue, Keese_InitRedOrBlack).
;   - Each time a Keese reaches top speed or finishes its turns it picks by Random+1,X: >= $A0 chase
;     the chase target (ChaseTargetX/Y), $20-$9F wander, < $20 slow down and rest
;     (Flyer_KeeseDecideState).
;   - FlyerMaxSpeedFrac is one byte for the whole room: Keese_InitBlue overwrites it with $C0.
; CONTENTS: (generated; L = line in this file)
;   L0046 Bubble_Init                    code  Initializes a Bubble (types $2B-$2D): q-speed $40 (1 px/frame), then the walker init.
;   L0057 Keese_InitBlue                 code  Initializes a Blue Keese (type $1B): random 8-way direction, speed $1F, top speed $C0.
;   L0077 Keese_InitRedOrBlack           code  Initializes a Red or Black Keese (types $1C, $1D): as the blue one, but start speed $7F.
;   L0093 Bubble_Update                  code  Updates a Bubble: wanders, draws itself, and blocks or unblocks Link's sword when it touches him.
;   L0151 Obj_AnimateAndDrawCommon       code  Advances the object's two-frame animation and draws frame image 0, flipped on its second frame.
;   L0163 Keese_Update                   code  Updates a Keese (types $1B-$1D): flight state, movement, wing animation, collisions.
;   L0186 Keese_ControlFlight            code  Runs the Keese's current flight state handler.
;   L0191 Keese_ControlFlight_JumpTable  data  TABLE: Keese flight state handlers; index = Flyer_ObjFlyingState.
;   L0205 Flyer_KeeseDecideState         code  Keese flight state 1: picks the next state from Random+1,X and loads 6 turns.
;   L0216 Flyer_SetStateAndTurns         code  Sets the flying state from Y and the turn counter to 6 (also used by the flying Gleeok head).
; ============================================================================

; ----------------------------------------------------------------------------
; Initializes a Bubble (types $2B-$2D): q-speed $40 (1 px/frame), then the walker init.
; IN:   X = object slot.
; ----------------------------------------------------------------------------
Bubble_Init:
    LDA #$40                    ; Q-speed $40 (1 pixel a frame)
    STA ObjQSpeedFrac, X
    JMP Walker_Init

; ----------------------------------------------------------------------------
; Initializes a Blue Keese (type $1B): random 8-way direction, speed $1F, top speed $C0.
; IN:   X = object slot.
; NOTE: Also the first step of the red/black Keese init and of the flying Gleeok head's init.
;       Writes FlyerMaxSpeedFrac, which all flyers in the room share.
; ----------------------------------------------------------------------------
Keese_InitBlue:
    ; Choose one of eight directions by Random,X.
    LDA Random, X
    AND #$07
    TAY
    LDA Dir_EightWayDirs, Y
    STA ObjDir, X
    JSR Flyer_ResetState         ; Flying state 0 (speed up); also clears the invincibility timer.
    LDA #$C0                    ; Top speed: $C0/256 = 0.75 pixels a frame.
    STA FlyerMaxSpeedFrac
    LDA #$1F                    ; Start speed; Flyer_State0SpeedUp adds 1 a frame, so $C0 comes $A1 frames later.
    STA Flyer_ObjSpeed, X
    RTS

; ----------------------------------------------------------------------------
; Initializes a Red or Black Keese (types $1C, $1D): as the blue one, but start speed $7F.
; IN:   X = object slot.
; NOTE: They reach top speed $41 frames after init instead of $A1. After a slow-down every Keese
;       restarts below $20, so the colours then fly alike.
; ----------------------------------------------------------------------------
Keese_InitRedOrBlack:
    JSR Keese_InitBlue
    LDA #$7F
    STA Flyer_ObjSpeed, X
    RTS

; ----------------------------------------------------------------------------
; Updates a Bubble: wanders, draws itself, and blocks or unblocks Link's sword when it touches him.
; IN:   X = object slot.
; NOTE: No weapon collision test at all. A touch goes through the normal harm path with damage 0
;       (Damage_ObjTypeDamageToLink is $00 for $2B-$2D): hurt sound, Link is knocked back and gets his
;       invincibility timer, and Link_BeHarmed zeroes WorldKillCount, HelpDropCount and
;       IsHelpDropBomb. Type $2B: SwordBlockedLongTimer = $10 (151 to 160 frames). Type $2D:
;       IsSwordBlocked = 1. Type $2C: IsSwordBlocked = 0. No touch registers while Link is
;       invincible, stunned or holds the clock (Link_CheckMonsterTouch).
; ----------------------------------------------------------------------------
Bubble_Update:
    LDA #$40                    ; Turn rate $40
    JSR Wanderer_Update
    LDA ObjType, X

    ; If the bubble is not type $2B (which blocks the sword temporarily),
    ; then subtract $2B from the type to get sprite palette 1 for type $2C or 2 for type $2D.
    CMP #$2B
    BEQ @Flash
    SEC
    SBC #$2B
    JMP @Animate

@Flash:
    ; Else make it flash by cycling through the four sprite palettes, one a frame.
    LDA FrameCounter
    AND #$03

@Animate:
    JSR Anim_SetSpriteDescriptorAttributes

    ; Animation counter 1 to switch animation frames every screen frame.
    LDA #$01
    JSR Obj_AnimateAndDrawCommon
    JSR Link_CheckMonsterTouch

    ; If Link was not touched ([06] = 0), then return.
    LDA $06
    BEQ @Exit

    ; If not type $2B (flashing), then go block or unblock the sword
    ; depending on the bubble type.
    LDA ObjType, X
    CMP #$2B
    BNE @BlockOrUnblock

    ; Else the flashing bubble blocks the sword temporarily.
    ; Set the long timer to $10 (10-frame units: up to $A0 frames).
    LDA #$10
    STA SwordBlockedLongTimer

@Exit:
    RTS

@BlockOrUnblock:
    ; Subtract $2C from the type to get the value for blocking:
    ; 0 for type $2C (sword allowed again), 1 for type $2D (sword blocked).
    SEC
    SBC #$2C
    STA IsSwordBlocked
    RTS

; ----------------------------------------------------------------------------
; Advances the object's two-frame animation and draws frame image 0, flipped on its second frame.
; IN:   A = animation counter reload (frames per animation frame); X = object slot;
;       [04], [05] = sprite attributes, set by the caller.
; NOTE: Shared by Bubble, Stalfos, Patra and the guard fire.
; ----------------------------------------------------------------------------
Obj_AnimateAndDrawCommon:
    JSR Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor
    JSR Anim_SetObjHFlipForSpriteDescriptor
    LDA #$00
    JMP Anim_DrawObjectNotMirrored

; ----------------------------------------------------------------------------
; Updates a Keese (types $1B-$1D): flight state, movement, wing animation, collisions.
; IN:   X = object slot.
; NOTE: Frozen in place while InvClock or ItemLiftTimer is nonzero; the collision tests still run.
;       Knockback is cancelled every frame, so a Keese is never shoved.
; ----------------------------------------------------------------------------
Keese_Update:
    ; If Link does not have the clock and is not lifting an item,
    ; then fly.
    LDA InvClock
    ORA ItemLiftTimer
    BNE :+
    JSR Keese_ControlFlight
    JSR Flyer_Move
:
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; The wing frame follows the distance flown: it changes every 2 pixels.
    LDA Flyer_ObjDistTraveled, X
    AND #$02
    LSR
    JSR Anim_DrawObjectMirrored
    JSR Collision_CheckMonsterVsWeaponsAndLink
    JMP Shove_Reset

; ----------------------------------------------------------------------------
; Runs the Keese's current flight state handler.
; IN:   X = object slot.
; ----------------------------------------------------------------------------
Keese_ControlFlight:
    LDA Flyer_ObjFlyingState, X
    JSR Util_TableJump

; TABLE: Keese flight state handlers; index = Flyer_ObjFlyingState.
Keese_ControlFlight_JumpTable:
    .ADDR Flyer_State0SpeedUp             ; 0: speed +1 a frame until top speed, then state 1
    .ADDR Flyer_KeeseDecideState    ; 1: pick state 2, 3 or 4 at random
    .ADDR Flyer_State2Chase               ; 2: turn toward the chase target every $10 frames; then state 1
    .ADDR Flyer_State3Wander              ; 3: turn at random every $10 frames; back to 1 when turns run out
    .ADDR Flyer_State4SlowDown            ; 4: speed -1 a frame until below $20, then state 5
    .ADDR Flyer_State5Delay               ; 5: wait for ObjTimer (random $40-$7F frames), then state 0

; ----------------------------------------------------------------------------
; Keese flight state 1: picks the next state from Random+1,X and loads 6 turns.
; IN:   X = object slot.
; NOTE: >= $A0 (96 in 256): state 2, chase. $20-$9F (128 in 256): state 3, wander. < $20 (32 in
;       256): state 4, slow down.
; ----------------------------------------------------------------------------
Flyer_KeeseDecideState:
    LDY #$02
    LDA Random+1, X
    CMP #$A0
    BCS Flyer_SetStateAndTurns  ; If >= $A0, go to state 2.
    INY
    CMP #$20
    BCS Flyer_SetStateAndTurns  ; If >= $20, go to state 3.
    INY                         ; Else go to state 4.

; Sets the flying state from Y and the turn counter to 6 (also used by the flying Gleeok head).
Flyer_SetStateAndTurns:
    TYA
    STA Flyer_ObjFlyingState, X
    LDA #$06                    ; The chase and wander states decrement first: 5 turns, then state 1.
    STA Flyer_ObjTurns, X
    RTS
