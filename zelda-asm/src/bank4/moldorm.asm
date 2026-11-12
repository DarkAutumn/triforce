; ============================================================================
; FILE:     bank4/moldorm.asm            BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 4913-5153
; SUBJECTS: Flyer, Moldorm
; TOPIC:    Moldorm update, chase/wander
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0020 UpdateMoldorm                   code
;   L0111 ControlMoldormFlight            code
;   L0115 ControlMoldormFlight_JumpTable  data
;   L0121 Moldorm_Chase                   code
;   L0137 Moldorm_ChangeFlyingState       code
;   L0165 Moldorm_Wander                  code
;   L0195 Moldorm_PropagateDirs           code
;   L0240 L1156B_Exit                     code
;   L0243 Flyer_MoldormDecideState        code
; ============================================================================

UpdateMoldorm:
    ; If facing direction = 0, return.
    LDA ObjDir, X
    BEQ @Exit

    ; If the magic clock is missing, then move around (using the flying mechanism).
    LDA InvClock
    BNE :+
    JSR ControlMoldormFlight
    JSR MoveFlyer
:
    ; Draw with sprite attributes: normal, red palette row
    LDA #$02
    STA $03
    LDA #$44                    ; Fireball tile
    JSR Anim_WriteSprite

    ; Check collisions, but never let them interfere with movement.
    LDA ObjDir, X
    PHA                         ; Save the direction
    LDA ObjTimer, X
    PHA                         ; Save the timer
    JSR CheckMonsterCollisions
    PLA                         ; Restore the timer
    STA ObjTimer, X
    PLA                         ; Restore the direction
    STA ObjDir, X

    ; If this segment is still alive, then return.
    LDA ObjMetastate, X
    BEQ @Exit

    ; This segment is dead. We'll swap the tail segment with this
    ; dead segment. This way, moldorm heads are always rooted
    ; in slots 5 and $A; and the segments are continguous.
    ;
    ; Start by setting the HP for this dead segment's slot to $20.
    ;
    ; Note that this is not a true swap, because the tail's HP
    ; might be less than $20.
    LDA #$20
    STA ObjHP, X
    JSR CheckBossHitReaction

    ; Look for the tail segment, starting from the low end
    ; of the chain that this dead segment belongs to.
    LDY #$FF
    CPX #$06
    BCC :+
    LDY #$04
:
    INY
    LDA ObjType+1, Y
    CMP #$41
    BNE :-

    ; I don't know why the object timer of the tail segment that will
    ; be made dead is set to $11. By setting its object type to $5D
    ; below, its metastate will become $10 -- dying/dead. From
    ; there, the object timer is set to 6.
    LDA #$11
    STA a:ObjTimer+1, Y

    ; Copy the invincibility timer, X, and Y from the original dead
    ; segment to the tail segment.
    LDA ObjInvincibilityTimer, X
    STA ObjInvincibilityTimer+1, Y
    LDA ObjX, X
    STA a:ObjX+1, Y
    LDA ObjY, X
    STA a:ObjY+1, Y

    ; If the tail segment found is also a head, then
    ; the last segment has died. Return.
    CPY #$04
    BEQ @Exit
    CPY #$09
    BEQ @Exit

    ; Else finish swapping the segments.
    ; Set the tail segment's type to the dead dummy object.
    LDA #$5D
    STA ObjType+1, Y

    ; Reset the metastate of the current segment, so that it comes
    ; back to life.
    JSR ResetObjMetastate

@Exit:
    RTS

ControlMoldormFlight:
    LDA Flyer_ObjFlyingState, X
    JSR TableJump

ControlMoldormFlight_JumpTable:
    .ADDR Moldorm_Chase
    .ADDR Moldorm_Chase
    .ADDR Moldorm_Chase
    .ADDR Moldorm_Wander

Moldorm_Chase:
    ; If the segment is not a head (slot 5 or $A), then return.
    ; Only the head controls the movement direction.
    CPX #$05
    BEQ :+
    CPX #$0A
    BNE L1156B_Exit
:
    ; Chase while flying, as usual.
    JSR Flyer_Chase

    ; If timer has not expired, then
    ; go see if it's time to propagate direction changes down the chain.
    LDA ObjTimer, X
    BNE Moldorm_PropagateDirs

Moldorm_ChangeFlyingState:
    ; Timer = 0. Choose a new flying state, and arm the timer.
    JSR Flyer_MoldormDecideState
    LDA #$10                    ; $10 screen frames
    STA ObjTimer, X

    ; If the next lower slot's direction <> 0, then
    ; go propagate direction changes down the chain.
    ;
    ; But you had to have propagated directions once for the
    ; next lower direction to be set. How does it happen?
    ;
    ; Within a flyer state, the flying routines turn the flyer
    ; and set the object timer to $10 several times. After any of
    ; those times will the test succeed.
    ;
    ; When a flying state ends, timer will be 0, triggering the code
    ; in this block to set the timer to $10, and try to propagate
    ; directions.
    ;
    ; But this is also the case the first time here.
    ;
    ; So, I believe that the point of this test is to delay moving
    ; the lower (not head) segments when starting out.
    LDA ObjDir-1, X
    BNE Moldorm_PropagateDirs
    RTS

Moldorm_Wander:
    ; If the segment is not a head (slot 5 or $A), then return.
    ; Only the head controls the movement direction.
    CPX #$05
    BEQ :+
    CPX #$0A
    BNE L1156B_Exit
:
    ; Turn randomly while flying, as usual.
    JSR Flyer_Wander

    ; After every turn, the common flyer routines set timer to $10.
    ; This can be used as a signal for moldorm to propagate
    ; direction changes down the chain.
    ;
    ; Between each turn, timer <> $10.
    ;
    ; At the end of the delay after the last turn, timer = 0.
    ; This is a signal that that flying state ended, and a new one
    ; must be chosen. Also, because a flying interval ended ($10 frames),
    ; we can propagate direction changes again; and start
    ; a new flying interval by setting timer to $10. This will make
    ; the next flying state delay that amount of time before turning
    ; the first time.
    ;
    ; Here and now, if timer has expired, then go decide a new flying state,
    ; arm the timer again, and shift directions down the chain.
    LDA ObjTimer, X
    BEQ Moldorm_ChangeFlyingState

Moldorm_PropagateDirs:
    ; If timer <> $10 (rearmed in this frame), then return.
    LDA ObjTimer, X
    CMP #$10
    BNE L1156B_Exit

    ; Timer = $10.
    ;
    ; If deferred bounce direction <> 0, then
    ; assign it to facing direction, and reset deferred bounce direction.
    LDA Moldorm_ObjBounceDir, X
    BEQ @SkipBounce
    STA ObjDir, X
    LDA #$00
    STA Moldorm_ObjBounceDir, X

@SkipBounce:
    ; Will loop 4 times, indexed by [00], once for each segment.
    LDA #$04
    STA $00

    ; Starting from the last slot of this moldorm chain (0 or 5):
    LDY #$00
    CPX #$05
    BEQ @Loop
    LDY #$05

@Loop:
    ; Copy old direction from a higher slot to the lower one;
    ; and to the lower one's facing direction.
    LDA Moldorm_ObjOldDir+2, Y
    STA Moldorm_ObjOldDir+1, Y
    STA a:ObjDir+1, Y

    ; Bottom of the loop.
    ; Increment segment index.
    ; Decrement loop counter.
    INY
    DEC $00
    BNE @Loop

    ; Copy current direction to old direction.
    LDA ObjDir, X
    STA Moldorm_ObjOldDir, X

L1156B_Exit:
    RTS

Flyer_MoldormDecideState:
    ; TODO:
    ; Why not call it Moldorm_DecideFlyingState? Same for others.
    ;
    ; If Random >= $40, go to flying state 2, else 3.
    ;
    ; Set up 8 turns.
    LDY #$02
    LDA Random, X
    CMP #$40
    BCS :+
    INY
:
    TYA
    STA Flyer_ObjFlyingState, X
    LDA #$08                    ; Set 8 turns for the next flying state.
    STA Flyer_ObjTurns, X
    RTS
