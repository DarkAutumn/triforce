; ============================================================================
; FILE:     bank1/room-bounds.asm        BANK: 1 (RAM code: stored at $A500, run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 3306-3509
; SUBJECTS: Audio, Link, Room, Sprite
; TOPIC:    Silencing the sound channels; drawing Link behind the background inside the left and right
;           dungeon doorways; the room bound test that stops objects and shots at the room's border.
; SEE ALSO: bank5/mode3-mode6-init.asm (Room_SetUpObjBounds and the bound values);
;           bank7/walker-move.asm (Walker_Move calls Room_BoundDir); bank7/walker-collision-screen-edge.asm
;           (Obj_ResetMovingDir; Link's screen-edge test, which is separate from these bounds);
;           bank1/shot-aim-boomerang-speed.asm (Obj_MoveShot); bank4/wallmaster.asm (other user of
;           Sprite_HalfOffsetsX).
; KEY FACTS:
;   - Link and monsters: moving left is refused when X < RoomBoundLeft ($11 OW, $21 UW), right when
;     X >= RoomBoundRight ($E0 OW, $D0 UW) (Room_BoundDirHorizontally).
;   - Link and monsters: moving up is refused when Y < RoomBoundUp ($4E OW, $5E UW), down when
;     Y >= RoomBoundDown ($CD OW, $BD UW) (Room_BoundDirVertically).
;   - Object slots $0D and up (Link's weapons) and a monster's boomerang (type $5C) get wider bounds:
;     left X < bound - $0B, right X >= bound + $0C, up Y < bound - $0F, down Y >= bound + $12
;     (Room_BoundDirHorizontally, Room_BoundDirVertically).
;   - A refused direction zeroes the whole of [0F], not just the offending bit (Room_BoundDir_Cancel).
;   - [0F] is zeroed only if it holds the direction bit of the bound that failed, so an object
;     outside a bound may still move back in or along it (Room_BoundDir_Cancel).
;   - The test never moves or clamps a coordinate; it only cancels the move (Room_BoundDir).
;   - In a dungeon the sprite at Sprites+$48 gets the behind-background bit while Link's X is < $10
;     or >= $E9, and the one at Sprites+$4C while X + 8 is (Link_PutSpritesBehindSideDoorways).
;   - Audio_SilenceAll requests $80 on Tune0Request (stops the song) and EffectRequest (stops samples
;     and tune 1) and clears CurTune0 and CurTune1.
; CONTENTS: (generated; L = line in this file)
;   L0046 Audio_SilenceAll                   code  Stops the song, the samples and both tune channels.
;   L0056 Sprite_HalfOffsetsX                data  TABLE: X offset of each half of a 16-pixel-wide object; index 0 = right half, 1 = left half; pixels.
;   L0067 Link_PutSpritesBehindSideDoorways  code  Puts Link's sprites behind the background where he overlaps a left or right dungeon doorway.
;   L0116 Room_BoundDirHorizontally          code  Cancels the move in [0F] when it would take the object past the room's left or right bound.
;   L0163 Room_BoundDir_Cancel               code  A bound test failed with Y = that bound's direction bit: zero [0F] if it contains that bit.
;   L0185 Room_BoundDirVertically            code  Cancels the move in [0F] when it would take the object past the room's top or bottom bound.
;   L0232 Room_BoundDir_Return               code  Shared RTS of the bound tests above.
;   L0240 Room_BoundDirA                     code  Stores direction A in [0F], then tests it against the room bounds (falls into Room_BoundDir).
;   L0254 Room_BoundDir                      code  Zeroes the moving direction in [0F] if it leads past any of the room's four bounds.
; ============================================================================

; ----------------------------------------------------------------------------
; Stops the song, the samples and both tune channels.
; OUT:  Tune0Request = EffectRequest = $80; CurTune0 = CurTune1 = 0; A = 0
; NOTE: The sound driver treats $80 in Tune0Request as "silence the song" and $80 in EffectRequest as
;       "silence samples and tune 1" (bank 0, Audio_DriveTune0 and Sfx_Drive).
; ----------------------------------------------------------------------------
Audio_SilenceAll:
    LDA #$80
    STA Tune0Request
    STA EffectRequest
    ASL                         ; A = 0.
    STA CurTune0
    STA CurTune1
    RTS

; TABLE: X offset of each half of a 16-pixel-wide object; index 0 = right half, 1 = left half; pixels.
Sprite_HalfOffsetsX:
    .BYTE $08, $00

; ----------------------------------------------------------------------------
; Puts Link's sprites behind the background where he overlaps a left or right dungeon doorway.
; OUT:  ORs $20 (behind-background priority) into the attribute bytes Sprites+$4A and Sprites+$4E;
;       X = $FF; uses [00]
; NOTE: The sprite at Sprites+$48 is tested with ObjX and the one at Sprites+$4C with ObjX + 8. A
;       sprite is put behind when that X is < $10 or >= $E9. Called every frame in a dungeon after
;       Link's sprites are written; nothing here tests Y or whether a doorway exists.
; ----------------------------------------------------------------------------
Link_PutSpritesBehindSideDoorways:
    LDY #$0A                    ; Sprites+64+$0A = attribute byte of the sprite at Sprites+$48.
    LDX #$00
    LDA ObjX, X
    STA $00                     ; [00] = Link's X.
    LDX #$01                    ; Left half first (offset 0), then the right half (offset 8).

@LoopSprite:
    LDA $00
    CLC
    ADC Sprite_HalfOffsetsX, X
    CMP #$E9
    BCS @AddPriority            ; This half is at X >= $E9: inside the right wall.
    CMP #$10
    BCS @NextSprite             ; X >= $10: in the open, leave it in front.

@AddPriority:
    LDA Sprites+64, Y
    ORA #$20                    ; Sprite attribute bit 5: behind the background.
    STA Sprites+64, Y

@NextSprite:
    INY
    INY
    INY
    INY
    CPY #$00
    BNE :+
    LDY #$20                    ; Never reached: Y is only ever $0E or $12 here.
:
    DEX
    BPL @LoopSprite
    RTS

; ----------------------------------------------------------------------------
; Cancels the move in [0F] when it would take the object past the room's left or right bound.
; IN:   X = object slot; [0F] = moving direction
; OUT:  [0F] = 0 if [0F] had the left bit and the object is left of the left bound, or had the right
;       bit and the object is at or past the right bound; else unchanged. Y = 2 if the left test
;       failed, else 1 (whether or not the right test failed). Uses [00].
; NOTE: Link and monsters: left bound fails when ObjX < RoomBoundLeft ($11 OW, $21 UW); right bound
;       fails when ObjX >= RoomBoundRight ($E0 OW, $D0 UW).
; NOTE: Object slots >= $0D (Link's weapons and the room item slot) and object type $5C (a monster's
;       boomerang) are tested with ObjX + $0B on the left and ObjX - $0C on the right, so they travel
;       $0B pixels farther left and $0C farther right: left fails at ObjX < $06 OW / $16 UW, right at
;       ObjX >= $EC OW / $DC UW. The sums are 8-bit: ObjX >= $F5 wraps and fails the left test, and in
;       the overworld ObjX $06-$0B gives a negative value that fails the right test.
; NOTE: When the left test fails the right test is skipped, whether or not [0F] was cancelled.
; ----------------------------------------------------------------------------
Room_BoundDirHorizontally:
    LDY #$02                    ; Direction bit to cancel if the left test fails.
    LDA ObjX, X
    STA $00                     ; [00] = X coordinate to test.
    CPX #$00
    BEQ @Left                   ; Link: no adjustment.

    ; Object slot >= $0D (weapons) or object type $5C (monster's boomerang): test X + $0B.
    CPX #$0D
    BCS :+
    LDA ObjType, X
    CMP #$5C
    BNE @Left
:
    LDA $00
    CLC
    ADC #$0B
    STA $00

@Left:
    ; If the coordinate is < the left bound, go cancel a leftward move.
    LDA $00
    CMP RoomBoundLeft
    BCC Room_BoundDir_Cancel

    ; Weapons and monster boomerangs again: subtract $17, so the right test uses X - $0C.
    CPX #$00
    BEQ @Right                  ; Link: no adjustment.
    CPX #$0D
    BCS :+
    LDA ObjType, X
    CMP #$5C
    BNE @Right                  ; Not a monster's boomerang: no adjustment.
:
    LDA $00
    SEC
    SBC #$17
    STA $00

@Right:
    ; If the coordinate is < the right bound, return and leave the direction alone.
    LDY #$01                    ; Direction bit to cancel if the right test fails.
    LDA $00
    CMP RoomBoundRight
    BCC Room_BoundDir_Return

; A bound test failed with Y = that bound's direction bit: zero [0F] if it contains that bit.
Room_BoundDir_Cancel:
    ; Shared by the horizontal and vertical tests. If the object is not moving toward the bound it
    ; is beyond, leave [0F] alone; else Obj_ResetMovingDir sets [0F] to 0 (all bits, so a diagonal
    ; direction loses its other axis too).
    TYA
    AND $0F
    BEQ Room_BoundDir_Return
    JMP Obj_ResetMovingDir

; ----------------------------------------------------------------------------
; Cancels the move in [0F] when it would take the object past the room's top or bottom bound.
; IN:   X = object slot; [0F] = moving direction
; OUT:  [0F] = 0 if [0F] had the up bit and the object is above the top bound, or had the down bit
;       and the object is at or below the bottom bound; else unchanged. Y = 8 if the top test failed,
;       else 4 (whether or not the bottom test failed). Uses [00].
; NOTE: Link and monsters: top bound fails when ObjY < RoomBoundUp ($4E OW, $5E UW); bottom bound
;       fails when ObjY >= RoomBoundDown ($CD OW, $BD UW).
; NOTE: Object slots >= $0D and object type $5C (a monster's boomerang) are tested with ObjY + $0F at
;       the top and ObjY - $12 at the bottom: top fails at ObjY < $3F OW / $4F UW, bottom at
;       ObjY >= $DF OW / $CF UW. The sums are 8-bit: ObjY >= $F1 wraps and fails the top test.
; NOTE: When the top test fails the bottom test is skipped, whether or not [0F] was cancelled.
; ----------------------------------------------------------------------------
Room_BoundDirVertically:
    LDY #$08                    ; Direction bit to cancel if the top test fails.
    LDA ObjY, X
    STA $00                     ; [00] = Y coordinate to test.
    CPX #$00
    BEQ @Up                     ; Link: no adjustment.

    ; Object slot >= $0D (weapons) or object type $5C (monster's boomerang): test Y + $0F.
    CPX #$0D
    BCS :+
    LDA ObjType, X
    CMP #$5C
    BNE @Up
:
    LDA $00
    CLC
    ADC #$0F
    STA $00

@Up:
    ; If the coordinate is < the top bound, go cancel an upward move.
    LDA $00
    CMP RoomBoundUp
    BCC Room_BoundDir_Cancel

    ; Weapons and monster boomerangs again: subtract $21, so the bottom test uses Y - $12.
    CPX #$00
    BEQ @Down                   ; Link: no adjustment.
    CPX #$0D
    BCS :+
    LDA ObjType, X
    CMP #$5C
    BNE @Down                   ; Not a monster's boomerang: no adjustment.
:
    LDA $00
    SEC
    SBC #$21
    STA $00

@Down:
    ; If the coordinate is < the bottom bound, return and leave the direction alone.
    LDY #$04                    ; Direction bit to cancel if the bottom test fails.
    LDA $00
    CMP RoomBoundDown
    BCS Room_BoundDir_Cancel    ; At or past (>=) the bottom bound: go cancel a downward move.

; Shared RTS of the bound tests above.
Room_BoundDir_Return:
    RTS

; ----------------------------------------------------------------------------
; Stores direction A in [0F], then tests it against the room bounds (falls into Room_BoundDir).
; IN:   A = moving direction; X = object slot
; OUT:  as Room_BoundDir
; ----------------------------------------------------------------------------
Room_BoundDirA:
    STA $0F

; ----------------------------------------------------------------------------
; Zeroes the moving direction in [0F] if it leads past any of the room's four bounds.
; IN:   X = object slot; [0F] = moving direction
; OUT:  A = [0F] = the direction, or 0 if the move was refused; Z set when A = 0; Y = 8 or 4 (left by
;       the vertical test; it does not tell which bound refused the move). Uses [00].
; NOTE: Z is also set when the caller passed direction 0. No coordinate is changed: an object
;       already outside a bound is not pulled back and may move in any direction except farther out.
; NOTE: For Link, Walker_Move calls this only in a dungeon while DoorwayDir is 0 (and not in mode 9);
;       in the overworld Link is not held by these bounds. Knockback (Shove_Update) and shots use
;       Room_BoundDirA directly.
; ----------------------------------------------------------------------------
Room_BoundDir:
    JSR Room_BoundDirHorizontally
    JSR Room_BoundDirVertically
    LDA $0F
    RTS
