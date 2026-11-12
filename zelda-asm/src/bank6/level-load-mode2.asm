; ============================================================================
; FILE:     bank6/level-load-mode2.asm   BANK: 6 (switchable, $8000-$BFFF)
; UPSTREAM: Z_06.asm 15-263
; SUBJECTS: (none)
; TOPIC:    How level data reaches RAM: the tables that pick a level block (768 bytes of room attributes)
;           and a level info block per level and quest, the mode 2 routines that copy them to RAM, the
;           power-on copy of the common data block, and the second-quest patches applied after the copy.
; SEE ALSO: bank6/level-blocks-common-data.asm (the blocks copied here); bank6/quest2-replacements.asm (the
;           Q2 patch tables); bank7/init-mode-dispatch.asm (InitMode2, InitializeGameOrMode: the callers);
;           bank7/update-mode-dispatch.asm (UpdateMode2Load); Variables.inc (LevelBlockAttrsA-F and
;           LevelInfo_* field meanings); bank5/room-layouts-ow.asm (RoomLayoutOWCave2 = overworld layout
;           $7B, given to two rooms by the Q2 patch); data/README.md (decoded formats, patched values).
; KEY FACTS:
;   - The game reads level data only from its RAM copy: the level block at $687E-$6B7D
;     (LevelBlockAttrsA-F, six 128-byte arrays indexed by RoomId), the level info at $6B7E-$6C7D
;     (LevelInfo_*). The ROM blocks are named only by the address tables in this file.
;     FetchLevelBlockDestInfo, FetchLevelInfoDestInfo, CopyBlock.
;   - Mode 2 takes 3 frames with video off: init submode 0 copies the level block (InitMode2_Sub0), init
;     submode 1 copies the level info (InitMode2_Sub1), then one update frame applies the Q2 patches
;     (UpdateMode2Load_Full) and the caller goes to mode 3.
;   - Block choice: overworld = LevelBlockOW in both quests; levels 1-6 share one block and levels 7-9
;     another, each with a Q1 and a Q2 version (LevelBlockAddrsQ1, LevelBlockAddrsQ2). The quest is
;     QuestNumbers[CurSaveSlot]. Level info blocks are per level and the same in both quests
;     (LevelInfoAddrs).
;   - The level info copy moves 256 bytes though a block is 252: the last 4 bytes ($6C7A-$6C7D, covered
;     by no RAM symbol) receive whatever follows the block in ROM. FetchLevelInfoDestInfo.
;   - Q2, levels 1-9: the level info from offset $29 (LevelInfo_ShortcutOrItemPosArray) on is overwritten
;     from a per-level table: item positions, map rotation and X offset, start room, Triforce room, level
;     number, cellar rooms, boss room, map mask, status bar map. Palettes, monster counts and start Y stay.
;     UpdateMode2Load_Full@ReplaceInfoBytes.
;   - Q2, overworld: 15 bytes of the level block are written (rooms $0B, $0E, $0F, $22, $34, $3C, $45,
;     $74): 8 from a table, 7 by immediate stores; 13 of them differ from the Q1 value.
;     UpdateMode2Load_Full@PatchQ2Rooms.
;   - The common data block ($8E bytes of transfer buffers and the overworld column directory) is copied
;     to $67F0-$687D once, at power-on or reset, because code patches it in place. CopyCommonDataToRam.
;   - CopyBlock increments GameSubmode when it finishes: InitMode2_Sub0 relies on that to reach submode
;     1; InitMode2_Sub1 and CopyCommonDataToRam set GameSubmode back to 0 afterwards.
; CONTENTS: (generated; L = line in this file)
;   L0057 LevelBlockAddrsQ1                data  TABLE: ROM address of the Q1 level block; index = CurLevel (0 overworld, 1-9) * 2; value = address.
;   L0071 LevelInfoAddrs                   data  TABLE: ROM address of each level info block; index = CurLevel (0-9) * 2; value = address.
;   L0084 CommonDataBlockAddr_Bank6        data  DATA: ROM address of the common data block that CopyCommonDataToRam copies to $67F0.
;   L0089 LevelBlockAddrsQ2                data  TABLE: ROM address of the Q2 level block; index = CurLevel (0 overworld, 1-9) * 2; value = address.
;   L0108 InitMode2_Submodes               code  Mode 2 (load level) init: runs the submode that copies the level block or the level info to RAM.
;   L0113 InitMode2_Submodes_JumpTable     data  TABLE: mode 2 init submode handlers; index = GameSubmode.
;   L0122 InitMode2_Sub0                   code  Mode 2 init submode 0: copies the level block (768 bytes) for CurLevel and the quest to RAM.
;   L0156 InitMode2_Sub1                   code  Mode 2 init submode 1: copies the level info block for CurLevel to RAM and ends mode 2's init.
;   L0179 CopyCommonDataToRam              code  Copies the common data block (transfer buffers, column directory) from ROM to $67F0-$687D.
;   L0196 FetchLevelBlockDestInfo          code  Sets the copy destination for a level block: $687E (LevelBlockAttrsA) through $6B7D, 768 bytes.
;   L0214 FetchLevelInfoDestInfo           code  Sets the copy destination for a level info block: $6B7E through $6C7D (LevelInfo_*).
;   L0231 FetchDestAddrForCommonDataBlock  code  Sets the copy destination for the common data block: $67F0 through $687D, $8E bytes.
;   L0249 CopyBlock                        code  Copies bytes from [00:01] to [02:03] until the destination reaches [04:05] inclusive.
;   L0297 UpdateMode2Load_Full             code  Mode 2 update: in Q2, patches the level data just copied to RAM; in Q1 it does nothing.
; ============================================================================

; TABLE: ROM address of the Q1 level block; index = CurLevel (0 overworld, 1-9) * 2; value = address.
; Each block is 768 bytes (six 128-byte attribute arrays A-F).
LevelBlockAddrsQ1:
    .ADDR LevelBlockOW          ; 0 overworld
    .ADDR LevelBlockUW1Q1       ; levels 1-6 share one block (each level uses its own rooms of it)
    .ADDR LevelBlockUW1Q1
    .ADDR LevelBlockUW1Q1
    .ADDR LevelBlockUW1Q1
    .ADDR LevelBlockUW1Q1
    .ADDR LevelBlockUW1Q1
    .ADDR LevelBlockUW2Q1       ; levels 7-9 share one block
    .ADDR LevelBlockUW2Q1
    .ADDR LevelBlockUW2Q1

; TABLE: ROM address of each level info block; index = CurLevel (0-9) * 2; value = address.
; Each block is 252 bytes (LevelInfo_* layout). Used in both quests; Q2 patches the RAM copy afterwards.
LevelInfoAddrs:
    .ADDR LevelInfoOW
    .ADDR LevelInfoUW1
    .ADDR LevelInfoUW2
    .ADDR LevelInfoUW3
    .ADDR LevelInfoUW4
    .ADDR LevelInfoUW5
    .ADDR LevelInfoUW6
    .ADDR LevelInfoUW7
    .ADDR LevelInfoUW8
    .ADDR LevelInfoUW9

; DATA: ROM address of the common data block that CopyCommonDataToRam copies to $67F0.
CommonDataBlockAddr_Bank6:
    .ADDR CommonDataBlock_Bank6

; TABLE: ROM address of the Q2 level block; index = CurLevel (0 overworld, 1-9) * 2; value = address.
; Each block is 768 bytes. The overworld entry is the Q1 block (patched in RAM by UpdateMode2Load_Full).
LevelBlockAddrsQ2:
    .ADDR LevelBlockOW          ; 0 overworld: same block as Q1
    .ADDR LevelBlockUW1Q2       ; levels 1-6
    .ADDR LevelBlockUW1Q2
    .ADDR LevelBlockUW1Q2
    .ADDR LevelBlockUW1Q2
    .ADDR LevelBlockUW1Q2
    .ADDR LevelBlockUW1Q2
    .ADDR LevelBlockUW2Q2       ; levels 7-9
    .ADDR LevelBlockUW2Q2
    .ADDR LevelBlockUW2Q2

; ----------------------------------------------------------------------------
; Mode 2 (load level) init: runs the submode that copies the level block or the level info to RAM.
; IN:   GameSubmode = 0 or 1; CurLevel; bank 6 switched in (called from InitMode2 in bank 7, which on
;       submode 0 has already cleared the room history and the level kill counts, loaded the level's
;       patterns and copied the common code to RAM again).
; NOTE: One submode per frame, so loading a level takes 2 init frames and 1 update frame.
; ----------------------------------------------------------------------------
InitMode2_Submodes:
    LDA GameSubmode
    JSR TableJump

; TABLE: mode 2 init submode handlers; index = GameSubmode.
InitMode2_Submodes_JumpTable:
    .ADDR InitMode2_Sub0
    .ADDR InitMode2_Sub1

; ----------------------------------------------------------------------------
; Mode 2 init submode 0: copies the level block (768 bytes) for CurLevel and the quest to RAM.
; OUT:  $687E-$6B7D filled; GameSubmode = 1 (incremented by CopyBlock).
; NOTE: The quest is QuestNumbers[CurSaveSlot]: 0 = Q1 table, nonzero = Q2 table.
; ----------------------------------------------------------------------------
InitMode2_Sub0:
    ; X = CurLevel * 2: offset of the level's entry in the address tables.
    LDA CurLevel
    ASL
    TAX
    LDY CurSaveSlot
    LDA QuestNumbers, Y
    BNE @SecondQuest

    ; Q1: [00:01] = source address.
    LDA LevelBlockAddrsQ1, X
    STA $00
    INX
    LDA LevelBlockAddrsQ1, X
    JMP @Copy

@SecondQuest:
    ; Q2: [00:01] = source address.
    LDA LevelBlockAddrsQ2, X
    STA $00
    INX
    LDA LevelBlockAddrsQ2, X

@Copy:
    STA $01
    JSR FetchLevelBlockDestInfo
    JSR CopyBlock
    RTS

; ----------------------------------------------------------------------------
; Mode 2 init submode 1: copies the level info block for CurLevel to RAM and ends mode 2's init.
; OUT:  $6B7E-$6C7D filled; GameSubmode = 0; IsUpdatingMode = 1 (next frame runs UpdateMode2Load_Full).
; NOTE: The same block is used in Q1 and Q2; Q2 differences are patched in by UpdateMode2Load_Full.
; ----------------------------------------------------------------------------
InitMode2_Sub1:
    ; X = CurLevel * 2: offset of the level's entry in LevelInfoAddrs.
    LDA CurLevel
    ASL
    TAX
    LDA LevelInfoAddrs, X
    STA $00
    INX
    LDA LevelInfoAddrs, X
    STA $01
    JSR FetchLevelInfoDestInfo
    JSR CopyBlock
    LDA #$00                    ; CopyBlock left GameSubmode at 2; the update starts at submode 0.
    STA GameSubmode
    INC IsUpdatingMode
    RTS

; ----------------------------------------------------------------------------
; Copies the common data block (transfer buffers, column directory) from ROM to $67F0-$687D.
; OUT:  $67F0-$687D filled; GameSubmode = 0 (stored after CopyBlock's increment).
; NOTE: Called once per power-on or reset (InitializeGameOrMode, while IsGameInitialized is 0), so the
;       in-place patches made to these buffers during play persist until the next reset.
; ----------------------------------------------------------------------------
CopyCommonDataToRam:
    LDX #$00                    ; [00:01] = address of the common data block in ROM.
    LDA CommonDataBlockAddr_Bank6, X
    STA $00
    INX
    LDA CommonDataBlockAddr_Bank6, X
    STA $01
    JSR FetchDestAddrForCommonDataBlock
    JSR CopyBlock
    LDA #$00
    STA GameSubmode
    RTS

; ----------------------------------------------------------------------------
; Sets the copy destination for a level block: $687E (LevelBlockAttrsA) through $6B7D, 768 bytes.
; OUT:  [02:03] = destination address; [04:05] = address of the last byte to write.
; ----------------------------------------------------------------------------
FetchLevelBlockDestInfo:
    LDA #$7E
    STA $02
    LDA #$68
    STA $03
    LDA #$7D                    ; Last byte: $6B7D (end of LevelBlockAttrsF).
    STA $04
    LDA #$6B
    STA $05
    RTS

; ----------------------------------------------------------------------------
; Sets the copy destination for a level info block: $6B7E through $6C7D (LevelInfo_*).
; OUT:  [02:03] = destination address; [04:05] = address of the last byte to write.
; NOTE: That is 256 bytes, 4 more than a level info block (252). The last 4 ($6C7A-$6C7D, after
;       LevelInfo_DeathPaletteSeries; no RAM symbol covers them) receive the ROM bytes that follow the
;       block: the start of the next level's info, or of the common data block after level 9.
; ----------------------------------------------------------------------------
FetchLevelInfoDestInfo:
    LDA #$7E
    STA $02
    LDA #$6B
    STA $03
    LDA #$7D                    ; Last byte: $6C7D.
    STA $04
    LDA #$6C
    STA $05
    RTS

; ----------------------------------------------------------------------------
; Sets the copy destination for the common data block: $67F0 through $687D, $8E bytes.
; OUT:  [02:03] = destination address; [04:05] = address of the last byte to write.
; NOTE: $67F0 is where the BANK_06_DATA segment is linked to run; $687D is the byte before
;       LevelBlockAttrsA.
; ----------------------------------------------------------------------------
FetchDestAddrForCommonDataBlock:
    LDA #$F0                    ; $67F0 to $687D inclusive.
    STA $02
    LDA #$67
    STA $03
    LDA #$7D
    STA $04
    LDA #$68
    STA $05
    RTS

; ----------------------------------------------------------------------------
; Copies bytes from [00:01] to [02:03] until the destination reaches [04:05] inclusive.
; IN:   [00:01] = source address; [02:03] = destination address; [04:05] = last destination address.
; OUT:  GameSubmode incremented; [00:01] and [02:03] point at the last byte copied; Y = 0.
; NOTE: Y stays 0; both 16-bit pointers are incremented instead. The byte at [04:05] is always
;       written, and at least one byte is copied.
; ----------------------------------------------------------------------------
CopyBlock:
    LDY #$00

@Loop:
    LDA ($00), Y
    STA ($02), Y
    LDA $02                     ; Done when the destination equals the last address (low, then high).
    CMP $04
    BNE @Next
    LDA $03
    CMP $05
    BNE @Next
    INC GameSubmode
    RTS

@Next:
    LDA $02                     ; Increment the destination pointer [02:03].
    CLC
    ADC #$01
    STA $02
    LDA $03
    ADC #$00
    STA $03
    LDA $00                     ; Increment the source pointer [00:01].
    CLC
    ADC #$01
    STA $00
    LDA $01
    ADC #$00
    STA $01
    JMP @Loop

; ----------------------------------------------------------------------------
; Mode 2 update: in Q2, patches the level data just copied to RAM; in Q1 it does nothing.
; IN:   CurLevel; QuestNumbers[CurSaveSlot]; bank 6 switched in (called from UpdateMode2Load in bank 7,
;       which then advances to mode 3).
; OUT:  Q2, levels 1-9: level info bytes from offset $29 (LevelInfo_ShortcutOrItemPosArray) through
;       offset $29 + size are replaced from LevelInfoUWQ2Replacements<level>.
;       Q2, overworld: 8 bytes written through LevelBlockAttrsBQ2ReplacementOffsets/Values and 7 by
;       immediate stores (rooms $0B, $3C, $74).
; NOTE: The level info loop copies indexes size down to 0, that is size + 1 bytes: one byte more than
;       the table holds. The extra byte (first byte of the next table; for level 9 the low byte of
;       LevelInfoUWQ2ReplacementAddrs) lands in LevelInfo_StatusBarMapTransferBuf after the new end
;       marker, where the transfer never reads it.
; NOTE: So in Q2 a dungeon keeps Q1's level info only up to offset $28; item positions, start room,
;       Triforce room, cellar list, boss room and maps come from the table, and the "LEVEL-n" number
;       shown differs from CurLevel for levels 2-5, 7 and 8 (see bank6/quest2-replacements.asm).
; ----------------------------------------------------------------------------
UpdateMode2Load_Full:
    LDY CurSaveSlot
    LDA QuestNumbers, Y
    BEQ @Exit                   ; Q1: nothing to patch.
    LDA CurLevel
    BEQ @PatchQ2Rooms           ; Overworld: go patch the level block.
    TAX
    ASL
    TAY

    ; Levels 1-9. X = level, Y = level * 2.
    ; [00:01] = address of this level's replacement bytes. The address table has no entry for the
    ; overworld, so it is indexed from 2 bytes before its start.
    LDA LevelInfoUWQ2ReplacementAddrs-2, Y
    STA $00
    LDA LevelInfoUWQ2ReplacementAddrs-1, Y
    STA $01

    ; Y = this level's entry of the size table (again no overworld entry): the highest index copied.
    LDY LevelInfoUWQ2ReplacementSizes-1, X

@ReplaceInfoBytes:
    ; Copy replacement bytes [Y]..[0] over the level info, starting at offset $29
    ; (LevelInfo_ShortcutOrItemPosArray) and running on through the fields that follow it.
    LDA ($00), Y
    STA LevelInfo_ShortcutOrItemPosArray, Y
    DEY
    BPL @ReplaceInfoBytes

@Exit:
    RTS

@PatchQ2Rooms:
    ; Overworld in Q2. First the 8 table-driven bytes: the offsets are relative to LevelBlockAttrsB, and
    ; the last one ($8B) reaches past its 128 bytes into LevelBlockAttrsC (room $0B).
    LDY #$07

@ReplaceRoomBytes:
    LDX LevelBlockAttrsBQ2ReplacementOffsets, Y
    LDA LevelBlockAttrsBQ2ReplacementValues, Y
    STA LevelBlockAttrsB, X
    DEY
    BPL @ReplaceRoomBytes
    ; Then 7 immediate stores (decimal offsets 11, 60, 116 = rooms $0B, $3C, $74).
    LDA #$7B
    STA LevelBlockAttrsD+11     ; Room $0B: layout id $0A -> $7B (RoomLayoutOWCave2 in bank 5).
    LDA #$7B
    STA LevelBlockAttrsD+60     ; Room $3C: layout id $0A -> $7B.
    LDA #$5A
    STA LevelBlockAttrsD+116    ; Room $74: layout id $6F -> $5A.
    LDA #$72
    STA LevelBlockAttrsA+60     ; Room $3C: cave exit X $70, outer palette 2 (same as the Q1 value $72).
    LDA #$72
    STA LevelBlockAttrsA+116    ; Room $74: cave exit X $80 -> $70; outer palette 2 kept (Q1 value $82).
    LDA #$01
    STA LevelBlockAttrsF+60     ; Room $3C: cave exit row 3 -> 1 (Q1 value $03; other bits are 0).
    LDA #$00
    STA LevelBlockAttrsF+116    ; Room $74: cave exit row 3 -> 0 (Q1 value $03; other bits are 0).
    RTS
