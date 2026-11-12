; ============================================================================
; FILE:     bank7/sword-and-rod.asm      BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 4362-4679
; SUBJECTS: (none)
; TOPIC:    Rod or arrow, sword or rod update, MakeSwordShot
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0019 UpdateRodOrArrow        code
;   L0033 PlayerToWeaponOffsetsX  data  4 sets of horizontal offsets, one for each state of weapon.
;   L0043 PlayerToWeaponOffsetsY  data  4 sets of vertical offsets, one for each state of weapon.
;   L0053 UpdateSwordOrRod        code  TRIFORCE: Sword melee state machine handler.
;   L0237 SetUpWeaponWithState    code
;   L0270 L1F854_Exit             code
;   L0278 ResetObjState           code  Returns:
;   L0283 MakeSwordShot           code
; ============================================================================

UpdateRodOrArrow:
    ; The rod uses states $3x. Arrow uses $1x and $2x.
    LDA ObjState, X
    AND #$F0
    CMP #$30
    BCS UpdateSwordOrRod
    JMP UpdateArrowOrBoomerang

; 4 sets of horizontal offsets, one for each state of weapon.
; Each set contains 4 offsets in reverse direction index order:
; up, down, left, right
;
; These offsets are added to player X to get weapon X.
;
PlayerToWeaponOffsetsX:
    .BYTE $FF, $01, $00, $F8, $FF, $01, $F5, $0B
    .BYTE $FF, $01, $F9, $07, $FF, $01, $FD, $03

; 4 sets of vertical offsets, one for each state of weapon.
; Each set contains 4 offsets in reverse direction index order:
; up, down, left, right
;
; These offsets are added to player Y to get weapon Y.
;
PlayerToWeaponOffsetsY:
    .BYTE $F7, $F2, $F5, $F5, $F6, $0D, $03, $03
    .BYTE $F7, $09, $03, $03, $FF, $05, $03, $03

; TRIFORCE: Sword melee state machine handler.
; TRIFORCE: States: 1(4f)→2(8f)→3(1f)→4(1f)→5(1f)→6(deactivate) = 15 frames total
; TRIFORCE: ObjAnimCounter is used as the frame timer for each state.
; TRIFORCE: State 2 gets 8 frames; all others get 1 frame (but state 1 gets its timer from caller).
; TRIFORCE: State 3 triggers MakeSwordShot (beam check) in UpdateSwordShotOrMagicShot.
; TRIFORCE: Python: polls sword_animation==0 (state reset at state 6) for controllability.
UpdateSwordOrRod:
    ; If state = 0, the object is not active. So, return.
    LDA ObjState, X
    AND #$0F
    BEQ @Exit

    ; Decrement the state timer.
    ; If it hasn't expired, go move the sword or rod.
    DEC ObjAnimCounter, X
    BNE @DrawSwordOrRod

    ; State 2 will last 8 frames. The higher ones last 1.
    LDA ObjState, X
    AND #$0F
    TAY
    LDA #$08
    DEY
    BEQ :+
    LDA #$01
:
    ; Set both the player's animation counter and the sword's
    ; animation counter/timer to this value.
    STA ObjAnimCounter
    STA ObjAnimCounter, X

    ; Go to the next state.
    INC ObjState, X

    ; Once we reach state 6, deactivate the object.
    ; Otherwise, go draw it.
    LDA ObjState, X
    AND #$0F
    CMP #$06
    BCC @DrawSwordOrRod
    JSR ResetObjState

@Exit:
    RTS

@DrawSwordOrRod:
    ; Reset horizontal flipping in [0F].
    LDA #$00
    STA $0F

    ; If state = 5, return without drawing.
    LDA ObjState, X
    AND #$0F
    TAY
    LDA #$FC
    CPY #$05
    BEQ @Exit

@Add4:
    ; Set [00] to (state - 1) * 4:
    ; the byte offset of the beginning of a set of pixel offsets.
    CLC
    ADC #$04
    DEY
    BNE @Add4
    STA $00

    ; Set object's direction to player's. It might have changed
    ; since the last frame.
    LDA ObjDir
    STA ObjDir, X

    ; Add [00] and the reverse index of the direction to get
    ; the byte offset of the pixel offset.
    JSR GetOppositeDir
    TYA
    CLC
    ADC $00
    TAY

    ; Set object's X to player's X + offset for state and direction.
    ; Copy the result to [00].
    LDA ObjX
    CLC
    ADC PlayerToWeaponOffsetsX, Y
    STA ObjX, X
    STA $00

    ; Set object's Y to player's Y + offset for state and direction.
    ; Copy the result to [01].
    LDA ObjY
    CLC
    ADC PlayerToWeaponOffsetsY, Y
    STA ObjY, X
    STA $01

    ; If state = 1, use up direction.
    ; Else use object direction.
    LDA ObjState, X
    AND #$0F
    TAY
    LDA #$08
    DEY
    BEQ @CalcAttrs
    LDA ObjDir, X

@CalcAttrs:
    ; Store the weapon's frame (horizontal or vertical)
    ; for the direction chosen above in [0C].
    JSR GetOppositeDir
    LDA RDirectionToWeaponFrame, Y
    STA $0C

    ; Calculate sprite attributes.
    ;
    ; If the weapon is the rod, calculate its sprite attributes:
    ;   base attribute OR 1
    ; So, it uses palette row 5.
    LDA RDirectionToWeaponBaseAttribute, Y
    CPX #$0D
    BEQ @CalcSwordAttrs
    ORA #$01
    JMP @SetAttrs

@CalcSwordAttrs:
    ; The weapon is the sword.
    ; To calculate the sprite attributes: (item value - 1) + base attribute
    CLC
    ADC Items
    SEC
    SBC #$01

@SetAttrs:
    ; Set sprite attributes to the calculated value above.
    JSR Anim_SetSpriteDescriptorAttributes

    ; If the direction is left, set [0F] to flip horizontally.
    CPY #$02
    BNE :+
    INC $0F
:
    ; If state = 1, return. Don't show the weapon.
    LDA ObjState, X
    AND #$0F
    CMP #$01
    BEQ L1F854_Exit

    ; Set Y to the right item slot to pass to the routine to write sprites:
    ; sword or rod
    LDY #$00                    ; Sword slot
    CPX #$0D
    BEQ :+
    LDY #$08                    ; Rod slot
:
    JSR Anim_WriteItemSprites

    ; TRIFORCE: Sword melee state machine: 0x01→0x02→0x03→0x04→0x05→0x00
    ; TRIFORCE: State 3 is when MakeSwordShot is called (beam fires here if health is full)
    ; TRIFORCE: Timing: state 1 (4f), 2 (8f), 3 (1f), 4 (1f), 5 (1f) — total ~15 frames
    ; TRIFORCE: Python: frame_skip_wrapper.py ATTACK_COOLDOWN=15 matches this total
    ; If state <> 3, return.
    LDA ObjState, X
    AND #$0F
    CMP #$03
    BNE L1F854_Exit

    ; State = 3. Time to instantiate a shot.
    ;
    ; If object slot is not the rod's, go instantiate the sword shot separately.
    CPX #$12
    BNE MakeSwordShot

    ; Instantiate a magic shot (rod shot).
    ; Switch to the shot slot $E.
    LDX #$0E

    ; If high bit of state is set, return. It's already active.
    LDA ObjState, X
    BEQ @MakeMagicShot
    ASL
    BCS L1F854_Exit

@MakeMagicShot:
    ; Play "magic shot" tune.
    LDA #$04
    STA Tune0Request

    ; Activate the shot object (state $80).
    LDA #$80

SetUpWeaponWithState:
    STA ObjState, X
    LDA #$10
    JSR PlaceWeapon

    ; If the direction is horizontal, and X < $14 or >= $EC, then
    ; deactivate the shot object.
    LDA ObjDir, X
    AND #$03
    BEQ @ChooseSpeed
    LDA ObjX, X
    CMP #$14
    BCC ResetObjState
    CMP #$EC
    BCS ResetObjState

@ChooseSpeed:
    ; If high bit of state is set, use $A0 else $C0 for the q-speed fraction.
    ;
    ; The high bit is set for magic shot, and reset for sword shot.
    LDY #$C0
    LDA ObjState, X
    ASL
    BCC :+
    LDY #$A0
:
    TYA
    STA ObjQSpeedFrac, X

    ; Set the shot object's grid offset the same as the player's.
    LDA ObjGridOffset
    STA ObjGridOffset, X

L1F854_Exit:
    RTS

; Returns:
; A: 0
;
; If it's the room item, then deactivates it.
;
ResetObjState:
    LDA #$00
    STA ObjState, X
    RTS

MakeSwordShot:
    ; TRIFORCE: Beam health check — Python: Link._is_health_full_for_beams (link.py)
    ; TRIFORCE: Beam fires when: filled == containers_minus_one AND partial >= $80
    ; TRIFORCE: BUG-1 was fixed: old Python used float comparison which disagreed at edges.
    ; TRIFORCE: Beam spawns in slot $0E (SLOT_BEAM) with initial state $10.
    ;
    ; Try to activate a sword shot.
    ;
    ; Switch to shot slot $E.
    LDX #$0E

    ; If state <> 0, return. It's already activated.
    LDA ObjState, X
    BNE L1F854_Exit

    ; If [0529] is set, go activate a sword shot regardless of hearts.
    ;
    ; UNKNOWN: But I haven't found any code that sets it.
    LDA ForceSwordShot
    BNE @SetUp

    ; TRIFORCE: HeartValues ($066F) encoding:
    ; TRIFORCE:   High nibble = heart_containers - 1 (e.g., 3 containers → $2x)
    ; TRIFORCE:   Low nibble  = hearts_filled (0-15)
    ; TRIFORCE: For beams: filled must equal containers_minus_one (high nibble)
    ; If full hearts <> (heart containers - 1), return.
    LDA HeartValues
    PHA
    AND #$0F
    STA $00
    PLA
    LSR
    LSR
    LSR
    LSR
    CMP $00
    BNE L1F854_Exit

    ; TRIFORCE: HeartPartial ($0670): $00=empty, $01-$7F=half, $80-$FF=full
    ; TRIFORCE: Beam requires partial >= $80 (at least a full partial heart)
    ; If partial heart is less than half full, return.
    LDA HeartPartial
    CMP #$80
    BCC L1F854_Exit

@SetUp:
    LDA #$01                    ; Sword shot sound effect
    JSR PlaySample

    ; TRIFORCE: Beam initial state is $10 (0x10). Python: ANIMATION_BEAMS_ACTIVE = 0x10
    ; TRIFORCE: After ~22 frames at $10, transitions to $11 (spread). Python: ANIMATION_BEAMS_HIT = 0x11
    ; Go finish setting up a sword shot in initial state $10.
    LDA #$10
    BNE SetUpWeaponWithState
