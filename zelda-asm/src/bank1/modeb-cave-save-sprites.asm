; ============================================================================
; FILE:     bank1/modeb-cave-save-sprites.asm BANK: 1 (RAM code: stored at $A500, run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 2978-3151
; SUBJECTS: Boomerang, Dir, ModeBCave, Person, Room, Sample, Save, Sfx, Sprite
; TOPIC:    Small helpers of the common code block: sound requests (sample, effect, boomerang whirr),
;           the cave entry step of modes $B and $C, the save file A pointer table, the rolling sprite
;           index, the reverse direction table, and the line Link cannot cross in front of a person.
; SEE ALSO: bank5/room-history-ow.asm (ModeBCave_Init_JumpTable and ModeCShortcut_Init_JumpTable, submode 7);
;           bank5/cave-cellar-enter.asm (ModeBCave_Init_Sub8_WalkIn, the submode after it);
;           bank7/link-animate-ladder.asm (Link_EndMoveAndAnimate: why the walk-in is 8 pixels);
;           bank5/enter-room.asm (Room_EnterCommon); bank2/moded-save.asm,
;           bank2/mode1-init-save-files.asm and bank2/register-eliminate-init.asm (save files A and
;           B, Save_FileBAddressSets, users of Save_FileBChecksums and Save_FetchFileAAddressSet);
;           bank7/walker-move.asm and bank5/cave-cellar-exits.asm (callers of Person_BlockLinkMovingUp);
;           bank1/qspeed-direction-math.asm (Dir_GetOpposite, which yields the index for
;           Dir_ReverseIndexToDir); bank0/sound-effects.asm and bank0/tune1-and-samples.asm (what the
;           request bits play); bank7/boomerang-return-draw.asm (caller of Boomerang_PlaySfx).
; KEY FACTS:
;   - Entering a cave or the shortcut stairs room puts Link at ($70,$DD) facing up and walks him up
;     8 pixels to Y = $D5 in 5 frames before play starts; UndergroundExitType is 1 from then on.
;     ModeBCave_Init_Sub7_EnterCave.
;   - In a cave (modes $B and $C) Link can never move up past Y = $8D, whether or not a person is
;     there; in a dungeon the same line applies while slot 1 holds a person ($4B-$52) or the hungry
;     Goriya ($36). Person_BlockLinkMovingUp.
;   - Save file B's bookkeeping lives inside this code's RAM copy: checksums at $6D9C-$6DA1
;     (Save_FileBChecksums), flags and counts at $6D90-$6D9B. Saving and registering overwrite 18 bytes
;     of ModeBCave_Init_Sub7_EnterCave; the code copy made at every level load and at reset puts the
;     code back. So file B's checksum does not survive a reset.
;   - Save file A per slot: items $601A/$6042/$606A ($28 bytes), world flags $6092/$6212/$6392
;     ($180 bytes), name $6002/$600A/$6012, then one byte each at $6512+slot (slot in use),
;     $6515+slot (no use: only zeroed, summed and copied), $6518+slot (death count), $651B+slot
;     (quest number). Save_FileAAddressSets.
;   - Dir_ReverseIndexToDir maps the index from Dir_GetOpposite (3 right, 2 left, 1 down, 0 up) back to a
;     single direction bit; the pair reduces a direction byte to its lowest set bit, so right wins
;     over left, down and up.
;   - Sprite records $18-$3F are hidden every unpaused frame of the play update and handed out again
;     through a rolling index that starts one further each frame (40 positions), so an object
;     does not keep the same sprite records from frame to frame. Sprite_HideObjectSprites,
;     Sprite_CycleCurIndex.
;   - The boomerang's sound effect is requested at most once every 10 frames (timer ObjTimer[$13]).
;     Boomerang_PlaySfx.
;   - Sample_Play and Sfx_Play OR a bit into SampleRequest and EffectRequest, so several requests
;     made in one frame are merged into one byte for the sound driver.
; CONTENTS: (generated; L = line in this file)
;   L0069 Sample_Play                    code  Requests a DMC sample by ORing its bit into SampleRequest.
;   L0079 Sfx_Play                       code  Requests a noise-channel sound effect by ORing its bit into EffectRequest.
;   L0104 ModeBCave_Init_Sub7_EnterCave  code  Init submode 7 of modes $B and $C: sets up the cave room and puts Link at its entrance.
;   L0124 Save_FileBChecksums            code  RAM $6D9C-$6DA1 (6 bytes): save file B's checksums, 2 bytes per save slot; not an entry point.
;   L0149 Room_ResetTileObjInfo          code  Forgets the tile object found while laying out the room (type, X, Y = 0).
;   L0159 Dir_ReverseIndexToDir          data  TABLE: single direction bits; index = reverse direction index as returned in Y by Dir_GetOpposite
;   L0166 Save_FileAAddressSets          data  TABLE: save file A pointer sets, 3 save slots x 7 addresses (low byte first); index = slot * $0E +
;   L0171 Save_FileAAddressSet1          data  DATA: save file A pointer set of slot 1: $6042, $6212, $600A, $6513, $6516, $6519, $651C.
;   L0176 Save_FileAAddressSet2          data  DATA: save file A pointer set of slot 2: $606A, $6392, $6012, $6514, $6517, $651A, $651D.
;   L0189 Save_FetchFileAAddressSet      code  Copies the save file A pointer set of the current save slot to [00]-[0D].
;   L0223 Boomerang_PlaySfx              code  Requests a sound effect, but at most once every 10 frames (the boomerang's repeating whirr).
;   L0241 Sprite_HideObjectSprites       code  Hides sprite records $18-$3F and advances the frame's starting sprite index.
;   L0261 Sprite_CycleCurIndex           code  Advances RollingSpriteIndex by 1, wrapping from $27 to 0.
;   L0265 Sprite_CycleIndexInA           code  Second entry: A = a sprite index; stores (A + 1) wrapped at $28 in RollingSpriteIndex and A.
;   L0272 Sprite_ResetCurIndex           code  Third entry (also the wrap case above): RollingSpriteIndex = 0; A = 0.
;   L0289 Person_BlockLinkMovingUp       code  Stops Link from moving up while his Y is less than $8E: the line in front of a person.
; ============================================================================

; ----------------------------------------------------------------------------
; Requests a DMC sample by ORing its bit into SampleRequest.
; IN:   A = sample bit ($01 sword shot, $04 door, $08 Link hurt, $10 boss roar are the values the
;       callers pass)
; OUT:  SampleRequest = SampleRequest OR A; A = that value; Y = 1
; ----------------------------------------------------------------------------
Sample_Play:
    LDY #$01                    ; SongRequest+1 = SampleRequest.
    BNE :+                      ; Always taken.

; ----------------------------------------------------------------------------
; Requests a noise-channel sound effect by ORing its bit into EffectRequest.
; IN:   A = effect bit ($01 sword, $02 boomerang or arrow, $04 flame, $10 bomb, $20 sea; $80 makes
;       the sound driver silence the sample and tune 1 instead)
; OUT:  EffectRequest = EffectRequest OR A; A = that value; Y = 3
; ----------------------------------------------------------------------------
Sfx_Play:
    LDY #$03                    ; SongRequest+3 = EffectRequest.
:
    ORA SongRequest, Y
    STA SongRequest, Y
    RTS

; ----------------------------------------------------------------------------
; Init submode 7 of modes $B and $C: sets up the cave room and puts Link at its entrance.
; (Mode $B is a cave, mode $C the shortcut stairs room; both use this routine.)
; OUT:  Link at ($70,$DD) facing up, ObjGridOffset = $30; GameSubmode = 8 (ModeBCave_Init_Sub8_WalkIn);
;       IsUpdatingMode = 0; UndergroundExitType = 1
; NOTE: Needs bank 5 switched in (Room_EnterCommon, Obj_ResetWeaponSlotsAndLadder); the mode init tables that
;       point here are in bank 5.
; NOTE: Room_EnterCommon and the cross-room tasks both end in Mode_BeginUpdate, which zeroes
;       GameSubmode and sets IsUpdatingMode; this routine saves the submode and undoes both, so
;       the init goes on with submode 8.
; NOTE: The walk-in is 8 pixels, not $30: ModeBCave_Init_Sub8_WalkIn moves Link up until ObjGridOffset is 0,
;       and Link_EndMoveAndAnimate zeroes the offset as soon as it is a multiple of 8 at the end of
;       a frame. At q-speed $60 with ObjPosFrac 0 (Room_EnterCommon cleared it) Link moves 2, 1,
;       2, 1, 2 pixels: offset $28 after 5 frames, Y = $D5. Play starts on the frame after that.
; NOTE: UndergroundExitType stays 1 for the whole stay: Link_EndMoveAndAnimate clears it only in
;       mode 5, and the stay is mode $B or $C. It makes Room_EnterCommon put Link at the room's
;       cave exit spot when he leaves.
; ----------------------------------------------------------------------------
ModeBCave_Init_Sub7_EnterCave:
    ; Save the submode: the calls below reset it.
    LDA GameSubmode
    PHA
    JSR Room_EnterCommon
    JSR Obj_ResetWeaponSlotsAndLadder

    ; Put Link at the cave entrance ($70,$DD), facing up (direction 8).
    LDA #$70
    STA ObjX
    LDA #$DD
    STA ObjY
    LDA #$08

; RAM $6D9C-$6DA1 (6 bytes): save file B's checksums, 2 bytes per save slot; not an entry point.
; The address falls inside the RAM copy of this routine: the save code in bank 2 reads and
; writes file B's checksums over the six code bytes that start here (STA ObjDir, JSR
; Link_EndMoveAndAnimate and the opcode of the next JSR), and file B's other single bytes over
; the twelve bytes before ($6D90-$6D9B, Save_FileBAddressSets). Bank_CopyCommonCodeToRam puts the code
; back at every level load and at reset, so the checksum stored here does not survive a reset.
Save_FileBChecksums:
    STA ObjDir
    JSR Link_EndMoveAndAnimate  ; Draw Link at the entrance.
    JSR Room_RunEntryTasks_SetBounds

    ; Restore the submode, stay in the init phase, and go to the next submode (8, the walk-in).
    PLA
    STA GameSubmode
    LDA #$00
    STA IsUpdatingMode
    INC GameSubmode

    ; Nonzero grid offset = Link still has to walk in. The walk ends after 8 pixels (see header).
    LDA #$30
    STA ObjGridOffset

    ; 1 = Link is in a cave or cellar: used when he leaves to put him at the room's exit spot.
    LDA #$01
    STA UndergroundExitType
    RTS

; ----------------------------------------------------------------------------
; Forgets the tile object found while laying out the room (type, X, Y = 0).
; OUT:  RoomTileObjType = RoomTileObjX = RoomTileObjY = 0; A = 0 (callers store it)
; ----------------------------------------------------------------------------
Room_ResetTileObjInfo:
    LDA #$00
    STA RoomTileObjType
    STA RoomTileObjX
    STA RoomTileObjY
    RTS

; TABLE: single direction bits; index = reverse direction index as returned in Y by Dir_GetOpposite
; (0 up, 1 down, 2 left, 3 right); value = that direction: $08 up, $04 down, $02 left, $01 right.
; Dir_GetOpposite followed by a lookup here reduces a direction byte to its lowest set bit.
Dir_ReverseIndexToDir:
    .BYTE $08, $04, $02, $01

; TABLE: save file A pointer sets, 3 save slots x 7 addresses (low byte first); index = slot * $0E +
; 2 * field; fields in order: items block ($28 bytes), world flags ($180 bytes), name (8 bytes),
; slot-in-use byte, a byte with no use (only zeroed, summed and copied), death count, quest number.
; This label is slot 0's set: $601A, $6092, $6002, $6512, $6515, $6518, $651B.
Save_FileAAddressSets:
    .BYTE $1A, $60, $92, $60, $02, $60, $12, $65
    .BYTE $15, $65, $18, $65, $1B, $65

; DATA: save file A pointer set of slot 1: $6042, $6212, $600A, $6513, $6516, $6519, $651C.
Save_FileAAddressSet1:
    .BYTE $42, $60, $12, $62, $0A, $60, $13, $65
    .BYTE $16, $65, $19, $65, $1C, $65

; DATA: save file A pointer set of slot 2: $606A, $6392, $6012, $6514, $6517, $651A, $651D.
Save_FileAAddressSet2:
    .BYTE $6A, $60, $92, $63, $12, $60, $14, $65
    .BYTE $17, $65, $1A, $65, $1D, $65

; ----------------------------------------------------------------------------
; Copies the save file A pointer set of the current save slot to [00]-[0D].
; IN:   CurSaveSlot = 0-2
; OUT:  [00:01] = items block; [02:03] = world flags; [04:05] = name; [06:07] = slot-in-use byte;
;       [08:09] = the byte with no use ($6515 + slot); [0A:0B] = death count ($6518 + slot);
;       [0C:0D] = quest number; [0E:0F] = address of WorldFlags ($067F, the profile's working
;       copy); X = $FF
; NOTE: No range check: a CurSaveSlot above 2 reads past the three sets.
; ----------------------------------------------------------------------------
Save_FetchFileAAddressSet:
    ; Y = offset of the last byte of the slot's set: $0E * (slot + 1) - 1.
    LDA #$FF
    LDY CurSaveSlot
:
    CLC
    ADC #$0E
    DEY
    BPL :-
    TAY

    ; Copy the set's 14 bytes to [00]-[0D], last byte first.
    LDX #$0D
:
    LDA Save_FileAAddressSets, Y
    STA $00, X
    DEY
    DEX
    BPL :-

    ; [0E:0F] = address of the profile's WorldFlags block.
    LDA #<WorldFlags
    STA $0E
    LDA #>WorldFlags
    STA $0F
    RTS

; ----------------------------------------------------------------------------
; Requests a sound effect, but at most once every 10 frames (the boomerang's repeating whirr).
; IN:   Y = effect bit (the only caller passes $02, the boomerang effect)
; OUT:  when ObjTimer[$13] was 0: effect requested and ObjTimer[$13] = $0A; otherwise nothing
; NOTE: ObjTimer[$13] counts down 1 per frame, so the effect restarts every 10 frames while the
;       caller keeps calling.
; ----------------------------------------------------------------------------
Boomerang_PlaySfx:
    LDA ObjTimer+19
    BNE :+                      ; The interval is still running: return.
    TYA
    JSR Sfx_Play
    LDA #$0A
    STA ObjTimer+19
:
    RTS

; ----------------------------------------------------------------------------
; Hides sprite records $18-$3F and advances the frame's starting sprite index.
; OUT:  Sprites+$60..+$FC: Y byte of each record = $F8 (off screen); FirstSpriteIndex advanced by
;       1, wrapping from $27 to 0; RollingSpriteIndex = the new FirstSpriteIndex; X = 0
; NOTE: Records $00-$17 (offsets $00-$5F; Link's are among them) are left alone. Mode5Play_Update
;       calls this once per frame when the game is not paused (the end-level mode calls it too),
;       so objects take their sprite records from a starting point that moves by one every frame.
; ----------------------------------------------------------------------------
Sprite_HideObjectSprites:
    LDX #$60                    ; Offset of sprite record $18.
    LDA #$F8                    ; Y = $F8 hides a sprite.
:
    STA Sprites, X
    INX
    INX
    INX
    INX
    CPX #$00
    BNE :-                      ; Until X wraps past record $3F.
    LDA FirstSpriteIndex
    JSR Sprite_CycleIndexInA     ; Also stores the new value in RollingSpriteIndex.
    STA FirstSpriteIndex
    RTS

; ----------------------------------------------------------------------------
; Advances RollingSpriteIndex by 1, wrapping from $27 to 0.
; OUT:  RollingSpriteIndex = A = the new index (0-$27)
; ----------------------------------------------------------------------------
Sprite_CycleCurIndex:
    LDA RollingSpriteIndex

; Second entry: A = a sprite index; stores (A + 1) wrapped at $28 in RollingSpriteIndex and A.
Sprite_CycleIndexInA:
    CLC
    ADC #$01
    CMP #$28
    BNE :+

; Third entry (also the wrap case above): RollingSpriteIndex = 0; A = 0.
Sprite_ResetCurIndex:
    LDA #$00
:
    STA RollingSpriteIndex
:
    ; This RTS is also the early exit of Person_BlockLinkMovingUp below.
    RTS

; ----------------------------------------------------------------------------
; Stops Link from moving up while his Y is less than $8E: the line in front of a person.
; IN:   [0F] = Link's moving direction
; OUT:  [0F] = 0 and A = 0 when ObjY < $8E and [0F] contains up (8); else [0F] unchanged
; NOTE: Link's highest position is Y = $8D; left, right and down are never stopped here.
; NOTE: The routine does not look for a person. Its callers decide: in caves (modes $B and $C,
;       Warp_CheckSubroomExit) it always runs; elsewhere Walker_Move calls it for Link while object slot 1
;       holds type $36 (hungry Goriya) or $4B-$52 (dungeon persons).
; ----------------------------------------------------------------------------
Person_BlockLinkMovingUp:
    LDA ObjY
    CMP #$8E
    BCS :-                      ; Y >= $8E: return.
    LDA $0F
    AND #$08
    BEQ :-                      ; Not moving up: return.
    JMP Obj_ResetMovingDir          ; Moving up above the line: [0F] = 0 (no moving direction).
