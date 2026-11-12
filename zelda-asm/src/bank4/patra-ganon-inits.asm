; ============================================================================
; FILE:     bank4/patra-ganon-inits.asm  BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 9526-9585
; SUBJECTS: Ganon, Patra
; TOPIC:    Init routines: Patra (types $47, $48) and its eight children (types $25, $26); Ganon
;           (type $3E), whose init halts Link for the opening scene of the fight.
; SEE ALSO: bank4/patra.asm (Patra_Update, Patra_UpdateChild: orbits, maneuvers); bank4/ganon.asm
;           (Ganon_Update, Ganon_ScenePhase0_LiftInDark: what Link's timer set here drives);
;           bank4/fairy-flyer-common.asm (flyer speed-up and movement used by Patra);
;           bank7/object-update-table.asm (Obj_TypeHpPairs).
; KEY FACTS:
;   - Patra and its children get invincibility mask $FE: only damage type 1 (sword, sword shot, rod
;     swing) can hurt them (Patra_Init).
;   - Patra starts at ($80, $70) facing up, flyer speed $1F, room flyer top speed $40 (Patra_Init).
;   - Patra type $47 makes eight children of type $25, type $48 eight of type $26, in object slots
;     $02-$09 (Patra_Init).
;   - Patra_Init sets the maneuver timer, ObjTimer of the slot after Patra's (ObjTimer+1, X), to $FF
;     frames. With Patra in slot 1 that is slot $02's timer, which the init of the child in slot $02
;     (Obj_ResetMetastateAndTimer) zeroes again; that this happens before Patra's first update follows
;     from the slot order $0B down to $01 but was not traced (unverified) (Patra_Init).
;   - Ganon gets invincibility mask $FA: bits 0 (sword class) and 2 (arrow) are the only damage types
;     not masked (Ganon_Init).
;   - Ganon_Init halts Link (ObjState = $40) and sets Link's timer to $40 frames, the length of the
;     first, dark part of the opening scene (Ganon_Init, Ganon_ScenePhase0_LiftInDark).
; CONTENTS: (generated; L = line in this file)
;   L0041 Patra_Init  code  Sets up Patra and turns object slots $02-$09 into its eight children.
;   L0095 Ganon_Init  code  Sets up Ganon and starts the opening scene of his room by halting Link.
; ============================================================================

; ----------------------------------------------------------------------------
; Sets up Patra and turns object slots $02-$09 into its eight children.
; IN:   X = Patra's object slot. The code reads ObjType+1, so it expects Patra in slot 1.
; OUT:  [00] = child type; ObjType and ObjInvincibilityMask of slots $02-$09 written; Y = $0A.
; NOTE: Mask $FE leaves only bit 0 clear: Patra and the children are hurt by damage type 1 only
;       (sword, sword shot, rod swing).
;       Child type $25 goes with Patra $47, $26 with Patra $48.
;       The roar is sample request $40.
;       The children are initialized on a later update by Obj_ResetMetastateAndTimer (no spawn
;       cloud); for slot $02 that clears the ObjTimer byte set to $FF here.
; ----------------------------------------------------------------------------
Patra_Init:
    ; Invincible to every damage type except 1 (sword).
    LDA #$FE
    STA ObjInvincibilityMask, X

    ; Start at ($80, $70), facing up.
    LDA #$80
    STA ObjX, X
    LDA #$70
    STA ObjY, X
    LDA #$08
    STA ObjDir, X

    ; Flyer speed $1F; the room's flyer top speed $40.
    LDA #$1F
    STA Flyer_ObjSpeed, X
    LDA #$40
    STA FlyerMaxSpeedFrac
    STA SampleRequest           ; Sample $40: the roar shared with Digdogger and Manhandla.

    ; Maneuver timer: $FF frames. Patra_Update flips the children's orbit pattern when it is 0 and
    ; the child in slot 3 is at angle 0. The timer byte is that of the slot after Patra's: with
    ; Patra in slot 1 it is also the ObjTimer of the child in slot $02, whose init zeroes it.
    LDA #$FF
    STA ObjTimer+1, X

    ; Child type: $25 for Patra type $47, else $26 (Patra type $48).
    LDA #$25
    LDY ObjType+1
    CPY #$47
    BEQ :+
    LDA #$26
:
    ; For object slots 2 to 9: set the child type and the same invincibility mask.
    LDY #$02
    STA $00                     ; [00] holds the patra child type.
:
    LDA $00
    STA ObjType, Y
    LDA #$FE                    ; Children, too, are hurt by the sword class only.
    STA ObjInvincibilityMask, Y
    INY
    CPY #$0A
    BNE :-
    RTS

; ----------------------------------------------------------------------------
; Sets up Ganon and starts the opening scene of his room by halting Link.
; IN:   X = Ganon's object slot.
; OUT:  Link's ObjState = $40 (halted) and ObjTimer = $40; Ganon's metastate and timer = 0.
; NOTE: Mask $FA leaves bits 0 and 2 clear: damage types 1 (sword class) and 4 (arrow) are not
;       masked. What a sword or arrow hit does to Ganon is decided in Ganon_CheckCollisions.
;       Link stays halted until Ganon_ScenePhase1_LiftInLight ends. Samples $02 and $10 are requested together.
; ----------------------------------------------------------------------------
Ganon_Init:
    LDA #$FA                    ; Invincible to every damage type except 1 (sword) and 4 (arrow).
    STA ObjInvincibilityMask, X

    ; Halt Link.
    LDA #$40
    STA ObjState

    ; Link's timer: $40 frames for scene phase 0 (the room is still dark).
    STA ObjTimer
    LDA #$02                    ; Sample $02, the boss hit cry, used here as a shout.
    STA SampleRequest
    LDA #$10                    ; Sample $10: the roar shared with Aquamentus and Gleeok.
    JSR Sample_Play              ; ORs it into SampleRequest.

    ; No spawn cloud: metastate 0, and Ganon's own timer 0.
    JMP Obj_ResetMetastateAndTimer
