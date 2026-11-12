; Per-object-type aliases of the object slot arrays, used by the monster code in bank 4.
;
; Each object slot owns one byte in each general-purpose array at $380, $394, $3A8, $3BC and at $412, $41F,
; $42C, $437, $444, $451, $45E, $46B, $478 (slot 0 is Link, so monsters use index 1 and up). What a byte
; means depends on the object type in the slot; the names below give each type's meaning.
; "per slot" = indexed by object slot (X or Y register). "scalar" = one fixed byte.

; ---- Wanderers and shooters (Wanderer_Update; Goriya, Moblin, Lynel, Octorok, Stalfos) ----
Shooter_ObjWantsToShoot := $412     ; per slot; walking shooters: 1 = may shoot now, 0 = may not.
;   Shooter_ShootIfWanted fails when 0. On a grid point Wanderer_Update sets 1 whenever the monster turns
;   (toward the chase target, or the timed perpendicular turn) and 0 when it does not turn; Goriya_Update
;   sets 1 when the longer-axis distance to the target is < $51 pixels. Cleared after a successful shot.
Shooter_ObjShotCounter := $437      ; per slot; walking shooters: decremented once per shot made. Never read.
;   Written by Shooter_ShootAndSetTimer (Moblin, Lynel, Octorok) and the Q2 Stalfos; no code reads it for them.
Wanderer_ObjTurnRate := $41F        ; per slot; wanderers: turn rate $00-$FF given to Wanderer_Update.
;   On each grid square: if rate >= Random+1,X and Link's ObjState is not $FF, the monster may turn toward
;   the chase target when within 8 pixels of it on one axis; otherwise it turns only when
;   Wanderer_ObjTurnTimer is 0.
Wanderer_ObjTurnTimer := $478       ; per slot; wanderers: frames until a perpendicular turn is allowed.
;   Set to Random,X (0-$FF) on every turn (toward the chase target or perpendicular); decremented every
;   frame while > 0.

; ---- Remaining distance (Pols Voice, Blue Wizzrobe, Ganon) ----
ObjRemDistance := $394              ; per slot; pixels left to move. Alias of ObjGridOffset.
;   Pols Voice: walk distance left ($31 or $71 at random after a jump); at 0 it jumps. Blue Wizzrobe:
;   teleport distance left ($20 at start); nonzero = teleporting (drawn every other frame, no shooting).
;   Ganon: set to 1 every frame so that he moves like a teleporting Blue Wizzrobe.

; ---- Dungeon push block (type $68) ----
PushBlock_ObjPushTimer := $412      ; per slot; push block: frames Link has pushed it; it moves at $10.
;   Counts only while Link is aligned with the block (within $10 pixels) and holds the required direction;
;   reset to 0 otherwise.

; ---- Orbiting objects (Patra's children, pond fairy hearts; Obj_MoveAlongOrbit) ----
ObjAngleFrac := $380                ; per slot; orbiters: orbit angle fraction (low byte).
;   Alias of ObjMovingLimit.
ObjAngleWhole := $394               ; per slot; orbiters: angle $00-$1F (32 steps a circle), a direction.
;   Obj_MoveAlongOrbit moves the object up at $00, right at $08, down at $10, left at $18.
;   Obj_DecreaseOrbitAngle subtracts a 16-bit amount each frame (Patra child type $25: $70, type $26: $60) and
;   masks the result with $1F. Alias of ObjGridOffset.
ObjOrbitFracX := $412               ; per slot; orbiters: X position fraction (Obj_MoveAlongOrbit).
ObjOrbitFracY := $41F               ; per slot; orbiters: Y position fraction (Obj_MoveAlongOrbit).

; ---- Monster shots (MonsterShot_Bounce, MonsterShot_CheckLinkCollision) ----
MonsterShot_ObjBounceDir := $380    ; per slot; monster shot: Link's facing direction when it hit his shield.
;   The shot then bounces (state $30), displaced by MonsterShot_BounceStepsX/Heights for the opposite direction.
MonsterShot_ObjBounceDist := $394   ; per slot; monster shot: pixels bounced, +2 per frame; destroyed at $20.
;   Reset to 0 by MonsterShot_CheckLinkCollision.

; ---- Rock and gravestone (tile object types $62, $65) ----
Rock_ObjRoomId := $412              ; per slot; pushed rock or gravestone: RoomId when the push began.
;   Used when the object has moved $10 pixels: that room is marked visited and its stairs position is
;   looked up.

; ---- Fireball (types $55, $56; Fireball_Update) ----
Fireball_ObjDirToTargetX := $412    ; per slot; fireball: horizontal direction toward Link when fired.
Fireball_ObjQSpeedX := $41F         ; per slot; fireball: horizontal q-speed from Fireball_QSpeedsX ($10-$70).
Fireball_ObjDirToTargetY := $437    ; per slot; fireball: vertical direction toward Link when fired.
Fireball_ObjQSpeedY := $444         ; per slot; fireball: vertical q-speed from Fireball_QSpeedsY ($00-$70).
Fireball_ObjPosFracX := $451        ; per slot; fireball: X position fraction.
Fireball_ObjPosFracY := $45E        ; per slot; fireball: Y position fraction.

; ---- Aquamentus (type $3D) ----
Aquamentus_ObjFireballOffset := $478 ; per fireball slot; Y step of an Aquamentus fireball: 0, 1 or $FF.
;   0 middle, 1 lower, $FF upper fireball (type $55). Added to the fireball's ObjY every other frame while
;   Aquamentus's ObjTimer is running.

; ---- Flyers (Keese, Peahat, flying Ghini, Moldorm, Patra, Gleeok head, item fairy; Flyer_Move) ----
Flyer_ObjPosFrac := $412            ; per slot; flyer: position fraction; 1 pixel is moved on each carry.
;   Flyer_Move adds (Flyer_ObjSpeed AND $E0) to it every frame.
Flyer_ObjSpeed := $41F              ; per slot; flyer: speed in 1/256 pixel per frame (bits 5-7 count).
;   +1 per frame in Flyer_State0SpeedUp, -1 per frame in Flyer_State4SlowDown; compared with FlyerMaxSpeedFrac. Start
;   values: blue Keese $1F, red and black Keese $7F, item fairy $7F, Moldorm $80, Gleeok head $BF.
Flyer_ObjTurns := $42C              ; per slot; flyer: turn counter of the chase or wander state.
;   6 for Keese and the item fairy, 8 for Moldorm and Patra. The counter is decremented before each turn
;   and at 0 the flyer goes to flying state 1, so a counter of N gives N - 1 turns, $10 frames apart.
Flyer_ObjDistTraveled := $437       ; per slot; flyer: pixels moved (wraps at $100).
;   Drives the wing animation; the flying Gleeok head may shoot only when bit 0 is clear.
Flyer_ObjFlyingState := $444        ; per slot; flyer: flight state, index into the type's flight table.
;   Keese: 0 speed up, 1 decide, 2 chase Link, 3 wander, 4 slow down, 5 wait for ObjTimer.
;   Item fairy: 0 speed up, 1 decide, 2 nothing, 3 wander.
Flyer_ObjOffsetX := $46B            ; per slot; flyer: signed pixels moved horizontally since last zeroed.
;   Patra zeroes it each frame before moving; its children add Patra's value (slot 1) to their own ObjX.
Flyer_ObjOffsetY := $478            ; per slot; flyer: signed pixels moved vertically since last zeroed.

; ---- Jumpers (Tektite, Boulder) ----
Jumper_ObjSpeedWholeY := $412       ; per slot; jumper: vertical speed, pixels per frame, signed (up < 0).
Jumper_ObjSpeedFracY := $41F        ; per slot; jumper: vertical speed fraction (1/256 pixel per frame).
;   The acceleration from Jumper_YAccelerations is added each frame; the falling speed is capped at
;   2 pixels per frame.
Jumper_ObjTargetY := $444           ; per slot; jumper: Y where the jump ends.
;   The jump ends when the jumper is falling and within 3 pixels of it.
Jumper_ObjReversalCount := $451     ; per slot; jumper: consecutive frames the room boundary reversed it.
;   +1 on each frame of a jump on which Flyer_BoundByRoom reversed the direction (a new jump is then set up);
;   zeroed by every frame that moves.
;   At 2 the horizontal direction is flipped and the count reset, which breaks a loop in the room's corner.

; ---- Statues (dungeon statue fireballs) ----
Statue_FireballTimers := $4E8       ; [4] by fireball number in the statue pattern (not by slot): countdown.
;   Decremented on every statue update; when it was 0, a fireball may be shot (Random,X < $F0) and the
;   timer is reloaded from Statue_FireballStartTimes.

; ---- Wallmaster (type $27) ----
Wallmaster_ObjStep := $412          ; per slot; Wallmaster: index of its step in Wallmaster_StepsLeft.
;   Entry: low nibble = direction, high nibble = sprite attributes. Base 0 for the left and right walls,
;   $20 for top and bottom; +8 when Link faces the decreasing direction; +$10 for the far wall; +1 per $10
;   pixels moved.
Wallmaster_ObjTilesCrossed := $41F  ; per slot; Wallmaster: squares ($10 pixels) crossed; the trip ends at 7.

; ---- Moldorm (type $41; two worms of 5 segments, heads in slots 5 and $A) ----
Moldorm_ObjOldDir := $380           ; per slot; Moldorm: the segment's previous direction.
;   Alias of ObjMovingLimit. When the head's ObjTimer is rearmed to $10, each segment takes the old
;   direction of the segment ahead.
Moldorm_ObjBounceDir := $3BC        ; per slot; Moldorm head: direction to take at the next turn, 0 = none.
;   Set to the reversed 8-way direction when the head hits the room boundary. Alias of ObjQSpeedFrac.

; ---- Digdogger (types $38, $39; child $18) ----
Digdogger_ObjSpeedFrac := $41F      ; per slot; Digdogger: speed low byte (1/256 pixel per frame); starts $3F.
;   The 16-bit speed changes by 1 per frame toward the target speed.
Digdogger_ObjPosFrac := $412        ; per slot; Digdogger: position fraction.
;   Digdogger_Move adds (speed low byte AND $E0) per frame; the carry adds 1 pixel to the movement.
Digdogger_ObjSpeedWhole := $42C     ; per slot; Digdogger: speed high byte (pixels per frame).
Digdogger_ObjTargetSpeedFrac := $437 ; per slot; Digdogger: target speed low byte.
;   $80 while speeding up, $40 while slowing down.
Digdogger_ObjTargetSpeedWhole := $444 ; per slot; Digdogger: target speed high byte.
;   0 for the big Digdogger, 1 for a child.
Digdogger_ObjSpeedFlag := $45E      ; per slot; Digdogger: 0 = speeding up, 1 = slowing down.
Digdogger_ObjIsChild := $46B        ; per slot; Digdogger: 1 = small (child), 0 = big.
Digdogger_ObjCurPart := $478        ; per slot; big Digdogger: corner 0-3 being tested in the collision loop.

; ---- Dodongo (types $31, $32) ----
Dodongo_ObjBloatedSubstate := $42C  ; per slot; Dodongo in state 1 (bloated): substate 0-4.
;   0-2 wait, 3 die, 4 end (back to state 0). When substate 1 ends with fewer than 2 bomb hits, it skips
;   to 4.
Dodongo_ObjBombHits := $437         ; per slot; Dodongo: bombs swallowed; the second one kills it.
;   +1 when bloated substate 0 arms its timer.
Dodongo_ObjBloatedTimer := $45E     ; per slot; Dodongo: frames left in the bloated wait substate.
;   Loaded from Dodongo_BloatedWaitTimes by substate.

; ---- Pols Voice (type $16) ----
PolsVoice_ObjSpeedWhole := $412     ; per slot; Pols Voice: jump speed, pixels per moving frame (up < 0).
;   Pols Voice moves only on even frames (FrameCounter bit 0 clear), so per frame it is half of this.
PolsVoice_ObjLastTile := $41F       ; per slot; Pols Voice: tile under it at the last walkability test.
PolsVoice_ObjTargetY := $42C        ; per slot; Pols Voice: Y where the jump ends.
PolsVoice_ObjSpeedFrac := $444      ; per slot; Pols Voice: jump speed fraction; +$38 per moving frame.

; ---- Wizzrobe (blue $23, red $24) ----
Wizzrobe_ObjLastTile := $41F        ; per slot; Wizzrobe: tile found by its last collision test.
Wizzrobe_ObjBlueTurnCounter := $412 ; per slot; Blue Wizzrobe: step counter that times its turns.
;   It turns toward Link when the low 6 bits are 0: horizontally when bit 6 is clear, vertically when set.
;   Starting a teleport flips bit 6. The draw code reads the same byte as Wizzrobe_ObjRedAnimCounter, so
;   bit 2 of this counter selects the Blue Wizzrobe's frame image too.
Wizzrobe_ObjRedFadeCounter := $394  ; per slot; Red Wizzrobe: +1 per frame while fading; bit 0 = skip drawing.
;   Alias of ObjGridOffset (read through ObjRemDistance).
Wizzrobe_ObjRedAnimCounter := $412  ; per slot; Red Wizzrobe: +1 per frame; bit 2 selects the frame image.

; ---- Manhandla (type $3C; hands in slots 1-4, base in slot 5) ----
Manhandla_ObjPosFrac := $412        ; per slot; Manhandla: position fraction.
;   (Speed low byte AND $E0) is added per frame; the carry adds 1 pixel to the high speed byte's movement.
Manhandla_ObjSpeedFrac := $41F      ; per slot; Manhandla: speed low byte (1/256 pixel per frame); starts $80.
;   Each hand that dies adds $80 to the 16-bit speed of every segment.
Manhandla_ObjSpeedWhole := $42C     ; per slot; Manhandla: speed high byte (pixels per frame).
Manhandla_ObjPrevFrame := $437      ; per slot; Manhandla hand: Manhandla_ObjFrame at its last change.
Manhandla_ObjFrameAccum := $451     ; per slot; Manhandla: animation accumulator; bit 4 = animation frame.
;   Each frame it gains the pixels moved plus a random 0-3.
Manhandla_ObjFrame := $478          ; per slot; Manhandla: frame image and sprite attributes.
;   Bit 0 = animation frame, bit 7 = vertical flip. Start value from Manhandla_BaseFrameImagesAndAttrs.
Manhandla_SegmentJustDied := $383   ; scalar; Manhandla: nonzero = a hand died; the base then speeds up.
;   Cleared by the base (slot 5) after it adds $80 to every segment's speed.
Manhandla_BounceDir := $385         ; scalar; Manhandla: direction the base gives all segments, 0 = none.
;   Set on each turn of the base and when a segment's direction changes on moving (room boundary).
Manhandla_BaseDirBeforeMove := $384 ; scalar; Manhandla: the base's ObjDir before it moved this frame.
;   Every segment compares its own direction after moving with it; a difference (the room boundary
;   reversed the segment) is copied to Manhandla_BounceDir.

; ---- Gohma (types $33, $34) ----
Gohma_ObjShootTimer := $380         ; per slot; Gohma: frames until the next fireball; reload $41.
;   Starts at 1. Alias of ObjMovingLimit.
Gohma_ObjDistTraveled := $412       ; per slot; Gohma: pixels moved in this sprint; reverses at $20.
Gohma_ObjPosFrac := $41F            ; per slot; Gohma: position fraction; +$80 per frame, 1 pixel per carry.
Gohma_ObjNextOpenEyeCounter := $42C ; per slot; Gohma: countdown to the next eye opening.
;   -1 every other frame; at 0 the eye opens and the counter is reloaded with Random,X OR $C0.
Gohma_ObjEyeOpenTimer := $444       ; per slot; Gohma: frames the eye stays open ($80 at opening); 0 = closed.
Gohma_ObjGoStraight := $451         ; per slot; Gohma: 0 = pick a new direction now, nonzero = keep going.
;   The new direction is right, left or down at random. Cleared after every second sprint.
Gohma_ObjSprints := $45E            ; per slot; Gohma: number of $20-pixel sprints completed.
; Gohma_ObjEyeState ($46B) is defined with the symbols shared between banks: the knockback code in bank 1
; reads it too.
Gohma_ObjEyeAnimCounter := $478     ; per slot; Gohma: frames since the closed eye changed image; wraps at 8.

; ---- Lanmola (types $3A, $3B) ----
Lanmola_Speed := $4E6               ; scalar; Lanmola: pixels per frame, object type - $39 (1 or 2).
;   Same address as GleeokAnimationTimer.
Lanmola_Type := $4E7                ; scalar; Lanmola: object type of the room's Lanmola ($3A or $3B).
;   Used to find tail segments. Same address as GleeokBodyAnimationFrame.

; ---- Patra (types $47, $48) ----
Patra_ObjManeuverIndex := $45E      ; per slot; Patra: 0 or 1, the children's orbit pattern.
;   Flipped when Patra's timer is 0 and the child in slot 3 is at angle 0.

; ---- Ganon (type $3E) ----
Ganon_ObjDyingCount := $42C         ; per slot; Ganon: 0 = fighting; nonzero = dying, +1 per frame up to $FF.
;   At $50 he becomes ashes and the burst starts; the burst is drawn while the value is < $A0.
Ganon_ScenePhase := $445            ; scalar; Ganon's room: 0 Triforce lifted in dark, 1 in light, 2 fight.
Ganon_ObjAnimationFrame := $46B     ; per slot; Ganon: body animation frame 0-5; +1 per frame while moving.
Ganon_ObjCloudDist := $478          ; per slot; Ganon: pixel distance of the 8 burst clouds from his position.
;   Set to 8 when the arrow hits; -1 every 8 frames.

; ---- Gleeok (types $42-$45): stored neck data, 6 bytes per neck, index = segment 0-5 ----
; Gleeok_Update copies one neck at a time into ObjX/ObjY of slots 1-6 and Gleeok_CurHeadInfo, updates it,
; and copies it back (Gleeok_FetchNeckAddrs).
Gleeok_NeckXs0 := $438              ; [6] Gleeok neck 0: X of segments 0-5 (all start at $7C).
Gleeok_NeckXs1 := $452              ; [6] Gleeok neck 1: X of segments 0-5.
Gleeok_NeckXs2 := $46C              ; [6] Gleeok neck 2: X of segments 0-5.
Gleeok_NeckXs3 := $395              ; [6] Gleeok neck 3: X of segments 0-5.
Gleeok_NeckYs0 := $445              ; [6] Gleeok neck 0: Y of segments 0-5 (start values Gleeok_SegmentStartYs).
Gleeok_NeckYs1 := $45F              ; [6] Gleeok neck 1: Y of segments 0-5.
Gleeok_NeckYs2 := $479              ; [6] Gleeok neck 2: Y of segments 0-5.
Gleeok_NeckYs3 := $3BD              ; [6] Gleeok neck 3: Y of segments 0-5.
Gleeok_HeadInfo0 := $420            ; [6] Gleeok neck 0: head movement block (fields: GleeokHead_* offsets).
Gleeok_HeadInfo1 := $42D            ; [6] Gleeok neck 1: head movement block.
Gleeok_HeadInfo2 := $381            ; [6] Gleeok neck 2: head movement block.
Gleeok_HeadInfo3 := $3A9            ; [6] Gleeok neck 3: head movement block.

; ---- Gleeok: working copy of the current neck's head movement block (Gleeok_MoveHead) ----
Gleeok_CurHeadInfo := $413          ; [6] Gleeok: head movement block of the neck being updated.
Gleeok_CurHeadDirCounterH := $413   ; scalar; Gleeok head: horizontal counter; at $C it resets and X flips.
;   +1 on every 4th move of the head.
Gleeok_CurHeadDirCounterV := $414   ; scalar; Gleeok head: vertical counter; at 6 it resets and Y flips.
;   +1 on every 4th move of the head.
Gleeok_CurHeadSpeedFlagX := $415    ; scalar; Gleeok head: X step: 0 = +1 pixel per move, nonzero = -1.
Gleeok_CurHeadSpeedFlagY := $416    ; scalar; Gleeok head: Y step: 0 = +1 pixel per move, nonzero = -1.
Gleeok_CurHeadDirChangeCounter := $417 ; scalar; Gleeok head: moves counted; wraps at 4.
Gleeok_CurHeadDelay := $418         ; scalar; Gleeok head: moves to wait before the head starts moving.
;   Necks 0-3 start at 0, $C, $18, $24. A neck moves once every 4 frames.
Gleeok_WrithingCounter := $510      ; scalar; Gleeok body: fast animation steps left after a hit (set to 6).
;   While nonzero the body's animation timer is armed with 6 frames instead of $10. Same address as
;   ActiveRedLeeverCount.

; ---- Gleeok head movement block: field offsets (constants, not addresses) ----
; Added to Gleeok_HeadInfo0-3 in Gleeok_Init. Gleeok_CurHeadInfo + offset gives the Gleeok_CurHead* names.
GleeokHead_DirCounterH = 0          ; offset 0: horizontal direction counter (starts at 6).
GleeokHead_DirCounterV = 1          ; offset 1: vertical direction counter (starts at 3).
GleeokHead_SpeedFlagX = 2           ; offset 2: X step flag (starts $FF on necks 0 and 2, else 0).
GleeokHead_SpeedFlagY = 3           ; offset 3: Y step flag (starts $FF on necks 1 and 3, else 0).
GleeokHead_DirChangeCounter = 4     ; offset 4: move counter (the constant is not referenced).
GleeokHead_Delay = 5                ; offset 5: start delay.
