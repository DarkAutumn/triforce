; ============================================================================
; FILE:     bank5/mode-inits-obj-lists.asm BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 1369-1448
; SUBJECTS: (none)
; TOPIC:    Init routines of modes 8 (continue menu), $D (save) and $10 (stairs); the four monster spawn
;           position lists (one per direction Link faces on entry); Link's walk-in distances through a
;           dungeon door; the room monster lists (ObjLists) and their address table.
; SEE ALSO: bank5/spawn-positions.asm (AssignObjSpawnPositions reads the spawn lists);
;           bank5/enter-room.asm (InitMode_EnterRoom reads ObjListAddrs and the walk-in distances);
;           bank5/mode8-continue-mode10-stairs.asm (update routines of modes 8 and $10);
;           bank5/save-ram-clear-ram.asm (InitSaveRam); bank7/init-mode-dispatch.asm (InitMode table).
; KEY FACTS:
;   - A spawn spot byte is row in the high nibble, column in the low nibble: X = column * $10,
;     Y = row * $10 + $0D. Each list has 9 spots, all away from the side Link enters by: facing
;     right columns 5-$D, left 2-$A, down rows 7-$B, up rows 5-9.
;     SpawnPosList0-3 (decoded by AssignObjSpawnPositions).
;   - The list is chosen by the lowest set bit of Link's direction: 0 right, 1 left, 2 down, 3 up.
;     SpawnPosListAddrsLo, SpawnPosListAddrsHi.
;   - Monster list ids $62-$7F select one of 30 lists in ObjLists (index = id - $62); a list entry is
;     an object type, 0 = leave the slot empty. The room copies as many entries as its monster count,
;     starting at the list's first byte. ObjListAddrs, ObjLists.
;   - Entering a dungeon room through an open or key door Link starts with ObjGridOffset $18 (or -$18);
;     through any other door type (wall, false wall, bombable, shutter) $28 (or -$28). The mode 4 update
;     walks him until the offset is 8, 0 or -8. EnteringRoomRelativePositions.
;   - Mode $10 animates only when Link stands on tile $24 (cave or dungeon entrance): target Y is $10
;     pixels below. On a stairs tile the mode starts with no walk. InitMode10.
;   - Mode 8 init clears UndergroundExitType and the room history. InitMode8.
; CONTENTS: (generated; L = line in this file)
;   L0049 InitMode8                      code  Mode 8 (continue/save/retry menu) init: blanks the screen, then cues the menu text.
;   L0067 InitModeD                      code  Mode $D (save) init: blanks the screen and makes sure save RAM is formatted.
;   L0082 InitMode10                     code  Mode $10 (stairs) init: if Link stands on an entrance tile, starts the walk down into it.
;   L0103 SpawnPosListAddrsLo            data  TABLE: low address bytes of the spawn position lists; index = bit number of Link's direction
;   L0110 SpawnPosListAddrsHi            data  TABLE: high address bytes of the spawn position lists; index as in SpawnPosListAddrsLo.
;   L0119 SpawnPosList0                  data  TABLE: monster spawn spots when Link enters facing right; index = SpawnCycle 0-8; value = row in
;   L0125 SpawnPosList1                  data  TABLE: monster spawn spots when Link enters facing left; same format as SpawnPosList0.
;   L0131 SpawnPosList2                  data  TABLE: monster spawn spots when Link enters facing down; same format as SpawnPosList0.
;   L0137 SpawnPosList3                  data  TABLE: monster spawn spots when Link enters facing up; same format as SpawnPosList0.
;   L0144 EnteringRoomRelativePositions  data  TABLE: Link's starting ObjGridOffset (signed pixels) when he enters a dungeon room; index =
;   L0180 ObjLists                       data  DATA: the 30 room monster lists, 201 bytes of object types (0 = empty slot), addressed through
;   L0185 ObjListAddrs                   data  TABLE: addresses of the monster lists in ObjLists; index = monster list id - $62 (0-29); value =
; ============================================================================

; ----------------------------------------------------------------------------
; Mode 8 (continue/save/retry menu) init: blanks the screen, then cues the menu text.
; NOTE: Submode 0 clears UndergroundExitType, clears the name tables, cues the level palettes
;       (which advances the submode) and clears the room history (ClearRoomHistory). The next
;       frame cues transfer buffer 4 (menu text and attributes) and starts the mode's update.
; ----------------------------------------------------------------------------
InitMode8:
    JSR TurnOffAllVideo
    LDA GameSubmode
    BNE :+
    STA a:UndergroundExitType   ; A = 0: Link is no longer leaving a cave, cellar or dungeon.
    JSR TurnOffVideoAndClearArtifacts
    JSR PatchAndCueLevelPalettesTransferAndAdvanceSubmode
    JMP ClearRoomHistory
:
    LDA #$04                    ; Transfer buffer 4: the text and attributes of the mode 8 menu.
    STA TileBufSelector
    JMP BeginUpdateMode

; ----------------------------------------------------------------------------
; Mode $D (save) init: blanks the screen and makes sure save RAM is formatted.
; NOTE: InitSaveRam returns C set when it had to clear save RAM (its begin/end markers were
;       missing); the game then restarts through the reset handler instead of saving.
; ----------------------------------------------------------------------------
InitModeD:
    JSR TurnOffVideoAndClearArtifacts
    JSR InitSaveRam
    BCS :+                      ; Save RAM had to be cleared: reset the game.
    JMP BeginUpdateMode
:
    JMP IsrReset

; ----------------------------------------------------------------------------
; Mode $10 (stairs) init: if Link stands on an entrance tile, starts the walk down into it.
; OUT:  IsUpdatingMode = 1. On tile $24 also StairsTargetY = ObjY + $10 and the stairs sound.
; NOTE: Tile $24 is the black cave or dungeon entrance; on any other tile (stairs) nothing is set
;       up and the mode's update ends at once. The update moves Link 1 pixel every 4 frames
;       (UpdateMode10Stairs_Full) on frames with FrameCounter AND 3 = 0: $10 pixels in 61-64 frames.
; ----------------------------------------------------------------------------
InitMode10:
    LDX #$00                    ; Tile under Link's hotspot.
    JSR GetCollidableTileStill
    CMP #$24
    BNE :+                      ; Not the entrance tile: no walk.
    LDA #$00                    ; Volume envelope selector of the song = 0 (audible effect (unverified)).
    STA SongEnvelopeSelector
    LDA #$08                    ; Stairs sound effect.
    STA EffectRequest

    ; The walk ends $10 pixels below Link's current Y.
    LDA ObjY
    CLC
    ADC #$10
    STA StairsTargetY
:
    INC IsUpdatingMode
    RTS

; TABLE: low address bytes of the spawn position lists; index = bit number of Link's direction
; (0 right, 1 left, 2 down, 3 up); value = address of a 9-byte list.
SpawnPosListAddrsLo:
    .LOBYTES SpawnPosList0
    .LOBYTES SpawnPosList1
    .LOBYTES SpawnPosList2
    .LOBYTES SpawnPosList3

; TABLE: high address bytes of the spawn position lists; index as in SpawnPosListAddrsLo.
SpawnPosListAddrsHi:
    .HIBYTES SpawnPosList0
    .HIBYTES SpawnPosList1
    .HIBYTES SpawnPosList2
    .HIBYTES SpawnPosList3

; TABLE: monster spawn spots when Link enters facing right; index = SpawnCycle 0-8; value = row in
; the high nibble, column in the low nibble (X = column * $10, Y = row * $10 + $0D).
; (X,Y): ($50,$5D) ($50,$BD) ($80,$7D) ($80,$9D) ($A0,$7D) ($A0,$9D) ($C0,$6D) ($C0,$AD) ($D0,$8D)
SpawnPosList0:
    .BYTE $55, $B5, $78, $98, $7A, $9A, $6C, $AC
    .BYTE $8D

; TABLE: monster spawn spots when Link enters facing left; same format as SpawnPosList0.
; (X,Y): ($20,$8D) ($30,$6D) ($30,$AD) ($50,$7D) ($50,$9D) ($70,$7D) ($70,$9D) ($A0,$5D) ($A0,$BD)
SpawnPosList1:
    .BYTE $82, $63, $A3, $75, $95, $77, $97, $5A
    .BYTE $BA

; TABLE: monster spawn spots when Link enters facing down; same format as SpawnPosList0.
; (X,Y): ($30,$AD) ($50,$7D) ($50,$BD) ($60,$9D) ($70,$8D) ($90,$9D) ($A0,$7D) ($A0,$BD) ($C0,$AD)
SpawnPosList2:
    .BYTE $A3, $75, $B5, $96, $87, $99, $7A, $BA
    .BYTE $AC

; TABLE: monster spawn spots when Link enters facing up; same format as SpawnPosList0.
; (X,Y): ($30,$6D) ($50,$5D) ($50,$9D) ($60,$7D) ($80,$8D) ($90,$7D) ($A0,$5D) ($A0,$9D) ($C0,$6D)
SpawnPosList3:
    .BYTE $63, $55, $95, $76, $88, $79, $5A, $9A
    .BYTE $6C

; TABLE: Link's starting ObjGridOffset (signed pixels) when he enters a dungeon room; index =
; (0 facing left or up, 1 facing right or down) + (0 open or key door, 2 any other door type);
; value = $18 / -$18 / $28 / -$28. The mode 4 update walks Link until the offset is 8, 0 or -8.
EnteringRoomRelativePositions:
    .BYTE $18, $E8, $28, $D8

; DATA: the 30 room monster lists, 201 bytes of object types (0 = empty slot), addressed through
; ObjListAddrs. Decoded (list id: types):
;   $62: $03 $03 $04 $03 $04              red and blue Moblins
;   $63: $03 $04 $03 $04                  Moblins
;   $64: $1A $1A $02 $01 $02 $01          Peahats, Lynels
;   $65: $01 $02 $01 $02                  Lynels
;   $66: $01 $0F $02 $01 $10 $02          Lynels, Blue Leever, Red Leever
;   $67: $0F $1A $10 $1A $0F $1A          Leevers, Peahats
;   $68: $09 $08 $08 $08 $08              Octoroks (blue slow, red fast)
;   $69: $08 $07 $08 $07                  red Octoroks (fast, slow)
;   $6A: $08 $09 $08 $09 $08              Octoroks (red fast, blue slow)
;   $6B: $0A $07 $0A $07 $07              Octoroks (blue fast, red slow)
;   $6C: $03 $0A $04 $0A $04 $04          Moblins, blue fast Octoroks
;   $6D: $4A $00 $00 $00 $13 $13 $00 $13  Trap, Zols
;   $6E: $4A $00 $00 $00 $1B $1B $1B $1B  Trap, blue Keese
;   $6F: $2B $2B $2B $13 $13 $1B $1B $1B  Bubbles, Zols, blue Keese
;   $70: $16 $30 $30 $1B $1B $16 $00 $00  Pols Voices, Gibdos, blue Keese
;   $71: $2B $2B $2B $23 $23 $24 $23 $24  Bubbles, Wizzrobes
;   $72: $2B $2B $12 $12 $12 $00 $00 $00  Bubbles, Vires
;   $73: $2B $2B $13 $13 $17 $17          Bubbles, Zols, Like-Likes
;   $74: $2B $2B $0C $0B $0B $30 $30 $30  Bubbles, Darknuts, Gibdos
;   $75: $2B $2B $05 $05 $05 $1B $1B $1B  Bubbles, red Goriyas, blue Keese
;   $76: $4A $00 $00 $00 $17 $17 $17 $17  Trap, Like-Likes
;   $77: $4A $00 $00 $00 $23 $24 $23 $24  Trap, Wizzrobes
;   $78: $16 $0C $0B $0C $0B $16          Pols Voices, Darknuts
;   $79: $2B $2B $2B $27 $27 $27 $27 $27  Bubbles, Wallmasters
;   $7A: $05 $06 $06 $05 $06 $05 $00 $00  Goriyas
;   $7B: $23 $23 $24 $23 $24              Wizzrobes
;   $7C: $2B $17 $23 $23 $17 $24 $17 $24  Bubble, Like-Likes, Wizzrobes
;   $7D: $2D $2D $2D $2C $23 $24 $23 $24  Bubbles ($2D, $2C), Wizzrobes
;   $7E: $2D $2D $2D $2C $0C $0B $0C $0B  Bubbles ($2D, $2C), Darknuts
;   $7F: $2D $2D $2D $2C $27 $27 $27 $27  Bubbles ($2D, $2C), Wallmasters
; A room with a larger monster count than its list's length reads on into the next list.
ObjLists:
    .INCBIN "dat/ObjLists.dat"

; TABLE: addresses of the monster lists in ObjLists; index = monster list id - $62 (0-29); value =
; address of the list's first object type.
ObjListAddrs:
    .INCLUDE "dat/ObjListAddrs.inc"
