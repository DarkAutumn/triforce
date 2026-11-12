; ============================================================================
; FILE:     bank6/quest2-replacements.asm BANK: 6 (switchable, $8000-$BFFF)
; UPSTREAM: Z_06.asm 265-388
; SUBJECTS: (none)
; TOPIC:    The data the second quest patches into the level data in RAM after a level is loaded: 8
;           overworld room attribute bytes, and for each dungeon level 1-9 a replacement for the level
;           info from offset $29 on (item positions, start/Triforce/boss rooms, cellars, maps).
; SEE ALSO: bank6/level-load-mode2.asm (UpdateMode2Load_Full applies these, plus 7 immediate overworld
;           stores); Variables.inc (LevelInfo_* fields, LevelBlockAttrsA-F bits); data/README.md
;           ("Second-quest patches") and data/levelinfo.md, data/levelblock-ow.md (decoded results).
; KEY FACTS:
;   - Q2 overworld = Q1 block with these rooms changed: $0E and $74 lead to cave $1E, $0F to cave $20,
;     $22 to cave $21 (in Q1 the level 6 entrance), $34 becomes the entrance of level 3 (in Q1 cave $20;
;     in Q1 level 3 is entered from $74), $3C stays the level 2 entrance with inner palette 3, and room
;     $0B gets monster list id $2F (Pond fairy). "Level n" is CurLevel, not the number shown.
;     LevelBlockAttrsBQ2ReplacementOffsets, LevelBlockAttrsBQ2ReplacementValues.
;   - Each dungeon table replaces level info offsets $29 up: 4 item positions, map rotation, map X
;     offset, start room, Triforce room, room flags address, level number, 10 cellar rooms, boss room,
;     16 map mask bytes, status bar map transfer records. LevelInfoUWQ2Replacements1.
;   - The level number in a table is the digit drawn in "LEVEL-n", not CurLevel. Q2 tables for CurLevel
;     1-9 hold 1, 3, 2, 5, 4, 6, 8, 7, 9: the dungeon loaded as CurLevel 2 is shown as LEVEL-3 and so
;     on. LevelInfoUWQ2Replacements2 (and 3, 4, 5, 7, 8).
;   - Q2 start / Triforce / boss rooms by CurLevel: 1 $77/$08/$07; 2 $75/$20/$30; 3 $79/$1B/$2B;
;     4 $72/$00/$10; 5 $7D/$4F/$5F; 6 $74/$16/$26; 7 $7F/$2D/$3D; 8 $79/$1B/$1C; 9 $74/$07/$17.
;     LevelInfoUWQ2Replacements1 .. LevelInfoUWQ2Replacements9.
;   - The room flags address in the tables equals the Q1 value: $06FF for levels 1-6, $077F for 7-9.
;     LevelInfoUWQ2Replacements1, LevelInfoUWQ2Replacements7.
;   - LevelInfoUWQ2ReplacementSizes holds each table's exact length, but the copy loop runs from that
;     index down to 0, so it copies one byte past each table (it lands after the status bar map's end
;     marker; see UpdateMode2Load_Full).
; CONTENTS: (generated; L = line in this file)
;   L0049 LevelBlockAttrsBQ2ReplacementOffsets  data  TABLE: where the Q2 overworld patch writes; index = 0-7; value = offset from LevelBlockAttrsB.
;   L0064 LevelBlockAttrsBQ2ReplacementValues   data  TABLE: what the Q2 overworld patch writes; index = 0-7 (as the offsets table); value = the new byte.
;   L0084 LevelInfoUWQ2Replacements1            data  TABLE: Q2 level info patch for CurLevel 1; index = offset from level info +$29; 57 bytes.
;   L0096 LevelInfoUWQ2Replacements2            data  TABLE: Q2 level info patch for CurLevel 2; same layout; 55 bytes.
;   L0107 LevelInfoUWQ2Replacements3            data  TABLE: Q2 level info patch for CurLevel 3; same layout; 61 bytes.
;   L0119 LevelInfoUWQ2Replacements4            data  TABLE: Q2 level info patch for CurLevel 4; same layout; 57 bytes.
;   L0131 LevelInfoUWQ2Replacements5            data  TABLE: Q2 level info patch for CurLevel 5; same layout; 67 bytes.
;   L0145 LevelInfoUWQ2Replacements6            data  TABLE: Q2 level info patch for CurLevel 6; same layout; 64 bytes.
;   L0158 LevelInfoUWQ2Replacements7            data  TABLE: Q2 level info patch for CurLevel 7; same layout; 63 bytes.
;   L0170 LevelInfoUWQ2Replacements8            data  TABLE: Q2 level info patch for CurLevel 8; same layout; 67 bytes.
;   L0183 LevelInfoUWQ2Replacements9            data  TABLE: Q2 level info patch for CurLevel 9; same layout; 74 bytes.
;   L0197 LevelInfoUWQ2ReplacementAddrs         data  TABLE: address of each level's Q2 level info patch; index = (CurLevel - 1) * 2; value = address.
;   L0211 LevelInfoUWQ2ReplacementSizes         data  TABLE: length of each level's Q2 level info patch; index = CurLevel - 1; value = bytes (55-74).
; ============================================================================

; TABLE: where the Q2 overworld patch writes; index = 0-7; value = offset from LevelBlockAttrsB.
; Pairs with the values table below. $00-$7F is a room id in array B; $8B is array C, room $0B.
LevelBlockAttrsBQ2ReplacementOffsets:
    .BYTE $0E, $0F, $22, $34, $3C, $45, $74, $8B

; TABLE: what the Q2 overworld patch writes; index = 0-7 (as the offsets table); value = the new byte.
; Array B: bits 7-2 cave index (1-9 = entrance of the level with that CurLevel, $10 up = cave), bits
; 1-0 inner palette. Array C: bits 7-6 monster count index, bits 5-0 monster list id.
;   room $0E: $63 -> $7B  cave $18 -> cave $1E, palette 3
;   room $0F: $8B -> $83  cave $22 -> cave $20, palette 3
;   room $22: $18 -> $84  level 6 entrance -> cave $21, palette 0
;   room $34: $83 -> $0F  cave $20 -> level 3 entrance, palette 3
;   room $3C: $0A -> $0B  level 2 entrance, palette 2 -> 3
;   room $45: $12 -> $12  level 4 entrance, palette 2 (same as the Q1 value)
;   room $74: $0F -> $7A  level 3 entrance -> cave $1E, palette 3 -> 2
;   C, room $0B: $10 -> $2F  count index 0, monster list id $10 (Red Leever) -> $2F (Pond fairy)
; (Q1 values are those of the overworld level block, listed in data/levelblock-ow.md.)
LevelBlockAttrsBQ2ReplacementValues:
    .BYTE $7B, $83, $84, $0F, $0B, $12, $7A, $2F

; The nine tables below share one layout; index = offset from LevelInfo_ShortcutOrItemPosArray (level
; info offset $29):
;   $00-$03  LevelInfo_ShortcutOrItemPosArray: 4 room item positions (high nibble X, low nibble Y, * $10)
;   $04      LevelInfo_SubmenuMapRotation
;   $05      LevelInfo_StatusBarMapXOffset (pixels)
;   $06      LevelInfo_StartRoomId
;   $07      LevelInfo_TriforceRoomId
;   $08-$09  LevelInfo_WorldFlagsAddr (low, high)
;   $0A      LevelInfo_LevelNumber (the digit drawn in "LEVEL-n")
;   $0B-$14  LevelInfo_CellarRoomIdArray: 10 room ids, $FF = unused
;   $15      LevelInfo_BossRoomId
;   $16-$25  LevelInfo_SubmenuMapMask: 16 bytes, one per map column
;   $26-     LevelInfo_StatusBarMapTransferBuf: transfer records (address high, low, control, data),
;            ended by $FF

; TABLE: Q2 level info patch for CurLevel 1; index = offset from level info +$29; 57 bytes.
; Start room $77, Triforce room $08, boss room $07, cellar $28, shown as LEVEL-1.
LevelInfoUWQ2Replacements1:
    .BYTE $C9, $AC, $89, $B7, $00, $E0, $77, $08    ; 4 item positions, rotation, X offset, start, Triforce
    .BYTE $FF, $06, $01, $28, $FF, $FF, $FF, $FF    ; flags address $06FF, level number, cellars 0-4
    .BYTE $FF, $FF, $FF, $FF, $FF, $07, $00, $00    ; cellars 5-9, boss room, map mask 0-1
    .BYTE $00, $00, $00, $00, $00, $FF, $DB, $00    ; map mask 2-9
    .BYTE $00, $00, $00, $00, $00, $00, $20, $65    ; map mask 10-15; status bar map records from here
    .BYTE $42, $FF, $20, $85, $02, $FF, $FB, $20
    .BYTE $A5, $02, $FF, $67, $20, $C5, $42, $FF
    .BYTE $FF                                       ; end marker

; TABLE: Q2 level info patch for CurLevel 2; same layout; 55 bytes.
; Start room $75, Triforce room $20, boss room $30, cellar $56, shown as LEVEL-3.
LevelInfoUWQ2Replacements2:
    .BYTE $C9, $AC, $89, $87, $05, $00, $75, $20
    .BYTE $FF, $06, $03, $56, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $30, $00, $00
    .BYTE $00, $00, $00, $30, $00, $00, $00, $00
    .BYTE $7F, $03, $00, $00, $00, $00, $20, $67
    .BYTE $01, $FB, $20, $82, $01, $FF, $20, $87
    .BYTE $C3, $FF, $20, $C8, $01, $FF, $FF

; TABLE: Q2 level info patch for CurLevel 3; same layout; 61 bytes.
; Start room $79, Triforce room $1B, boss room $2B, cellars $09 $0B, shown as LEVEL-2.
LevelInfoUWQ2Replacements3:
    .BYTE $C9, $AC, $89, $37, $0D, $C8, $79, $1B
    .BYTE $FF, $06, $02, $09, $0B, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $2B, $00, $00
    .BYTE $00, $00, $00, $00, $7F, $EC, $7F, $00
    .BYTE $00, $00, $00, $00, $00, $00, $20, $64
    .BYTE $03, $FB, $FF, $FB, $20, $84, $03, $FF
    .BYTE $67, $FF, $20, $A4, $43, $FF, $20, $C4
    .BYTE $03, $FF, $24, $FF, $FF

; TABLE: Q2 level info patch for CurLevel 4; same layout; 57 bytes.
; Start room $72, Triforce room $00, boss room $10, cellars $21 $58 $7A, shown as LEVEL-5.
LevelInfoUWQ2Replacements4:
    .BYTE $C9, $AC, $89, $86, $06, $10, $72, $00
    .BYTE $FF, $06, $05, $21, $58, $7A, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $10, $00, $00
    .BYTE $00, $00, $00, $00, $CF, $DB, $F3, $00
    .BYTE $00, $00, $00, $00, $00, $00, $20, $64
    .BYTE $43, $FF, $20, $85, $02, $FB, $FF, $20
    .BYTE $A4, $02, $FF, $67, $20, $C4, $43, $FF
    .BYTE $FF

; TABLE: Q2 level info patch for CurLevel 5; same layout; 67 bytes.
; Start room $7D, Triforce room $4F, boss room $5F, cellars $0F $6A $7F, shown as LEVEL-4.
LevelInfoUWQ2Replacements5:
    .BYTE $C9, $AC, $89, $87, $0A, $B0, $7D, $4F
    .BYTE $FF, $06, $04, $0F, $6A, $7F, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $5F, $00, $00
    .BYTE $00, $00, $00, $00, $FF, $FF, $E7, $7E
    .BYTE $00, $00, $00, $00, $00, $00, $20, $64
    .BYTE $04, $FF, $FF, $FF, $FB, $20, $84, $04
    .BYTE $FF, $FF, $67, $FF, $20, $A4, $04, $FF
    .BYTE $FF, $FB, $FF, $20, $C4, $04, $FF, $FF
    .BYTE $FF, $67, $FF

; TABLE: Q2 level info patch for CurLevel 6; same layout; 64 bytes.
; Start room $74, Triforce room $16, boss room $26, cellars $03 $73 $46, shown as LEVEL-6.
; Its first two item positions ($49, $79) differ from the other tables' ($C9, $AC).
LevelInfoUWQ2Replacements6:
    .BYTE $49, $79, $89, $56, $04, $00, $74, $16
    .BYTE $FF, $06, $06, $03, $73, $46, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $26, $00, $00
    .BYTE $00, $00, $00, $04, $0C, $7E, $FF, $80
    .BYTE $F0, $00, $00, $00, $00, $00, $20, $65
    .BYTE $03, $FB, $FF, $67, $20, $68, $C2, $FF
    .BYTE $20, $86, $C3, $FF, $20, $85, $83, $FF
    .BYTE $FF, $67, $20, $A3, $02, $FB, $FF, $FF

; TABLE: Q2 level info patch for CurLevel 7; same layout; 63 bytes.
; Start room $7F, Triforce room $2D, boss room $3D, cellars $02 $03 $04 $05 $20 $21 $26 $2B $2C,
; shown as LEVEL-8. Room flags address $077F (levels 7-9).
LevelInfoUWQ2Replacements7:
    .BYTE $C9, $AC, $89, $79, $0C, $C0, $7F, $2D
    .BYTE $7F, $07, $08, $02, $03, $04, $05, $20
    .BYTE $21, $26, $2B, $2C, $FF, $3D, $00, $00
    .BYTE $00, $00, $FE, $FE, $82, $82, $82, $BE
    .BYTE $80, $FF, $00, $00, $00, $00, $20, $62
    .BYTE $C3, $FF, $20, $63, $C3, $FF, $20, $64
    .BYTE $45, $67, $20, $69, $C4, $FF, $20, $87
    .BYTE $C2, $FF, $20, $C2, $46, $67, $FF

; TABLE: Q2 level info patch for CurLevel 8; same layout; 67 bytes.
; Start room $79, Triforce room $1B, boss room $1C, cellars $27 $30 $37 $60 $67 $70, shown as LEVEL-7.
LevelInfoUWQ2Replacements8:
    .BYTE $C9, $AC, $89, $57, $0C, $C0, $79, $1B
    .BYTE $7F, $07, $07, $27, $30, $37, $60, $67
    .BYTE $70, $FF, $FF, $FF, $FF, $1C, $00, $00
    .BYTE $00, $00, $01, $01, $7D, $5D, $5D, $41
    .BYTE $7F, $00, $00, $00, $00, $00, $20, $64
    .BYTE $45, $FB, $20, $84, $05, $FF, $FB, $FB
    .BYTE $24, $FF, $20, $A4, $43, $FF, $20, $A8
    .BYTE $01, $FF, $20, $C2, $46, $FB, $20, $C8
    .BYTE $01, $FF, $FF

; TABLE: Q2 level info patch for CurLevel 9; same layout; 74 bytes.
; Start room $74, Triforce room $07, boss room $17, cellars $71 $72 $75 $76 $77, shown as LEVEL-9.
LevelInfoUWQ2Replacements9:
    .BYTE $C9, $AC, $89, $B6, $04, $00, $74, $07
    .BYTE $7F, $07, $09, $71, $72, $75, $76, $77
    .BYTE $FF, $FF, $FF, $FF, $FF, $17, $00, $00
    .BYTE $00, $00, $CC, $DE, $76, $7F, $7F, $76
    .BYTE $DE, $CC, $00, $00, $00, $00, $20, $62
    .BYTE $48, $FF, $20, $64, $44, $FB, $20, $83
    .BYTE $46, $FB, $20, $84, $44, $FF, $20, $A2
    .BYTE $08, $FF, $FF, $FB, $FF, $FF, $FB, $FF
    .BYTE $FF, $20, $C3, $46, $67, $20, $C5, $42
    .BYTE $FF, $FF

; TABLE: address of each level's Q2 level info patch; index = (CurLevel - 1) * 2; value = address.
; No overworld entry: the code indexes it from LevelInfoUWQ2ReplacementAddrs-2 with CurLevel * 2.
LevelInfoUWQ2ReplacementAddrs:
    .ADDR LevelInfoUWQ2Replacements1
    .ADDR LevelInfoUWQ2Replacements2
    .ADDR LevelInfoUWQ2Replacements3
    .ADDR LevelInfoUWQ2Replacements4
    .ADDR LevelInfoUWQ2Replacements5
    .ADDR LevelInfoUWQ2Replacements6
    .ADDR LevelInfoUWQ2Replacements7
    .ADDR LevelInfoUWQ2Replacements8
    .ADDR LevelInfoUWQ2Replacements9

; TABLE: length of each level's Q2 level info patch; index = CurLevel - 1; value = bytes (55-74).
; The copy loop uses it as its highest index, so it copies length + 1 bytes (indexes length..0); it
; is indexed from LevelInfoUWQ2ReplacementSizes-1 with CurLevel.
LevelInfoUWQ2ReplacementSizes:
    .BYTE $39, $37, $3D, $39, $43, $40, $3F, $43
    .BYTE $4A

; DATA: filler ($FF), 65 bytes, up to LevelBlockOW at $8400.
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF
