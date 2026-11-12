; ============================================================================
; FILE:     bank6/level-load-mode2.asm   BANK: 6 (switchable, $8000-$BFFF)
; UPSTREAM: Z_06.asm 15-263
; SUBJECTS: Bank, Level, Mode2Load
; TOPIC:    How level data reaches RAM: the tables that pick a level block (768 bytes of room attributes)
;           and a level info block per level and quest, the mode 2 routines that copy them to RAM, the
;           power-on copy of the common data block, and the second-quest patches applied after the copy.
; SEE ALSO: bank6/level-blocks-common-data.asm (the blocks copied here); bank6/quest2-replacements.asm (the
;           Q2 patch tables); bank7/init-mode-dispatch.asm (Mode2Load_Init, Mode_InitGameOrMode: the callers);
;           bank7/update-mode-dispatch.asm (Mode2Load_Update); Variables.inc (LevelBlockAttrsA-F and
;           LevelInfo_* field meanings); bank5/room-layouts-ow.asm (RoomLayout_RoomOW7B = overworld layout
;           $7B, given to two rooms by the Q2 patch); data/README.md (decoded formats, patched values).
; KEY FACTS:
;   - The game reads level data only from its RAM copy: the level block at $687E-$6B7D
;     (LevelBlockAttrsA-F, six 128-byte arrays indexed by RoomId), the level info at $6B7E-$6C7D
;     (LevelInfo_*). The ROM blocks are named only by the address tables in this file.
;     Level_SetBlockCopyDest, Level_SetInfoCopyDest, Level_CopyBlockAndIncSubmode.
;   - Mode 2 takes 3 frames with video off: init submode 0 copies the level block (Mode2Load_Init_Sub0), init
;     submode 1 copies the level info (Mode2Load_Init_Sub1), then one update frame applies the Q2 patches
;     (Mode2Load_Update_Full) and the caller goes to mode 3.
;   - Block choice: overworld = Level_BlockOW in both quests; levels 1-6 share one block and levels 7-9
;     another, each with a Q1 and a Q2 version (Level_BlockAddrsQ1, Level_BlockAddrsQ2). The quest is
;     QuestNumbers[CurSaveSlot]. Level info blocks are per level and the same in both quests
;     (Level_InfoAddrs).
;   - The level info copy moves 256 bytes though a block is 252: the last 4 bytes ($6C7A-$6C7D, covered
;     by no RAM symbol) receive whatever follows the block in ROM. Level_SetInfoCopyDest.
;   - Q2, levels 1-9: the level info from offset $29 (LevelInfo_ShortcutOrItemPosArray) on is overwritten
;     from a per-level table: item positions, map rotation and X offset, start room, Triforce room, level
;     number, cellar rooms, boss room, map mask, status bar map. Palettes, monster counts and start Y stay.
;     Mode2Load_Update_Full@ReplaceInfoBytes.
;   - Q2, overworld: 15 bytes of the level block are written (rooms $0B, $0E, $0F, $22, $34, $3C, $45,
;     $74): 8 from a table, 7 by immediate stores; 13 of them differ from the Q1 value.
;     Mode2Load_Update_Full@PatchQ2Rooms.
;   - The common data block ($8E bytes of transfer buffers and the overworld column directory) is copied
;     to $67F0-$687D once, at power-on or reset, because code patches it in place. Bank_CopyCommonDataToRam.
;   - Level_CopyBlockAndIncSubmode increments GameSubmode when it finishes: Mode2Load_Init_Sub0 relies on that to reach submode
;     1; Mode2Load_Init_Sub1 and Bank_CopyCommonDataToRam set GameSubmode back to 0 afterwards.
; CONTENTS: (generated; L = line in this file)
;   L0057 Level_BlockAddrsQ1                 data  TABLE: ROM address of the Q1 level block; index = CurLevel (0 overworld, 1-9) * 2; value = address.
;   L0071 Level_InfoAddrs                    data  TABLE: ROM address of each level info block; index = CurLevel (0-9) * 2; value = address.
;   L0084 Bank_CommonDataBlockAddr           data  DATA: ROM address of the common data block that Bank_CopyCommonDataToRam copies to $67F0.
;   L0089 Level_BlockAddrsQ2                 data  TABLE: ROM address of the Q2 level block; index = CurLevel (0 overworld, 1-9) * 2; value = address.
;   L0108 Mode2Load_Init_Submodes            code  Mode 2 (load level) init: runs the submode that copies the level block or the level info to RAM.
;   L0113 Mode2Load_Init_Submodes_JumpTable  data  TABLE: mode 2 init submode handlers; index = GameSubmode.
;   L0122 Mode2Load_Init_Sub0                code  Mode 2 init submode 0: copies the level block (768 bytes) for CurLevel and the quest to RAM.
;   L0156 Mode2Load_Init_Sub1                code  Mode 2 init submode 1: copies the level info block for CurLevel to RAM and ends mode 2's init.
;   L0179 Bank_CopyCommonDataToRam           code  Copies the common data block (transfer buffers, column directory) from ROM to $67F0-$687D.
;   L0196 Level_SetBlockCopyDest             code  Sets the copy destination for a level block: $687E (LevelBlockAttrsA) through $6B7D, 768 bytes.
;   L0214 Level_SetInfoCopyDest              code  Sets the copy destination for a level info block: $6B7E through $6C7D (LevelInfo_*).
;   L0231 Bank_SetCommonDataCopyDest         code  Sets the copy destination for the common data block: $67F0 through $687D, $8E bytes.
;   L0249 Level_CopyBlockAndIncSubmode       code  Copies bytes from [00:01] to [02:03] until the destination reaches [04:05] inclusive.
;   L0297 Mode2Load_Update_Full              code  Mode 2 update: in Q2, patches the level data just copied to RAM; in Q1 it does nothing.
; ============================================================================

; TABLE: ROM address of the Q1 level block; index = CurLevel (0 overworld, 1-9) * 2; value = address.
; Each block is 768 bytes (six 128-byte attribute arrays A-F).
Level_BlockAddrsQ1:
    .ADDR Level_BlockOW          ; 0 overworld
    .ADDR Level_BlockUW1to6Q1       ; levels 1-6 share one block (each level uses its own rooms of it)
    .ADDR Level_BlockUW1to6Q1
    .ADDR Level_BlockUW1to6Q1
    .ADDR Level_BlockUW1to6Q1
    .ADDR Level_BlockUW1to6Q1
    .ADDR Level_BlockUW1to6Q1
    .ADDR Level_BlockUW7to9Q1       ; levels 7-9 share one block
    .ADDR Level_BlockUW7to9Q1
    .ADDR Level_BlockUW7to9Q1

; TABLE: ROM address of each level info block; index = CurLevel (0-9) * 2; value = address.
; Each block is 252 bytes (LevelInfo_* layout). Used in both quests; Q2 patches the RAM copy afterwards.
Level_InfoAddrs:
    .ADDR Level_InfoOW
    .ADDR Level_InfoUW1
    .ADDR Level_InfoUW2
    .ADDR Level_InfoUW3
    .ADDR Level_InfoUW4
    .ADDR Level_InfoUW5
    .ADDR Level_InfoUW6
    .ADDR Level_InfoUW7
    .ADDR Level_InfoUW8
    .ADDR Level_InfoUW9

; DATA: ROM address of the common data block that Bank_CopyCommonDataToRam copies to $67F0.
Bank_CommonDataBlockAddr:
    .ADDR Bank_CommonDataBlock

; TABLE: ROM address of the Q2 level block; index = CurLevel (0 overworld, 1-9) * 2; value = address.
; Each block is 768 bytes. The overworld entry is the Q1 block (patched in RAM by Mode2Load_Update_Full).
Level_BlockAddrsQ2:
    .ADDR Level_BlockOW          ; 0 overworld: same block as Q1
    .ADDR Level_BlockUW1to6Q2       ; levels 1-6
    .ADDR Level_BlockUW1to6Q2
    .ADDR Level_BlockUW1to6Q2
    .ADDR Level_BlockUW1to6Q2
    .ADDR Level_BlockUW1to6Q2
    .ADDR Level_BlockUW1to6Q2
    .ADDR Level_BlockUW7to9Q2       ; levels 7-9
    .ADDR Level_BlockUW7to9Q2
    .ADDR Level_BlockUW7to9Q2

; ----------------------------------------------------------------------------
; Mode 2 (load level) init: runs the submode that copies the level block or the level info to RAM.
; IN:   GameSubmode = 0 or 1; CurLevel; bank 6 switched in (called from Mode2Load_Init in bank 7, which on
;       submode 0 has already cleared the room history and the level kill counts, loaded the level's
;       patterns and copied the common code to RAM again).
; NOTE: One submode per frame, so loading a level takes 2 init frames and 1 update frame.
; ----------------------------------------------------------------------------
Mode2Load_Init_Submodes:
    LDA GameSubmode
    JSR Util_TableJump

; TABLE: mode 2 init submode handlers; index = GameSubmode.
Mode2Load_Init_Submodes_JumpTable:
    .ADDR Mode2Load_Init_Sub0
    .ADDR Mode2Load_Init_Sub1

; ----------------------------------------------------------------------------
; Mode 2 init submode 0: copies the level block (768 bytes) for CurLevel and the quest to RAM.
; OUT:  $687E-$6B7D filled; GameSubmode = 1 (incremented by Level_CopyBlockAndIncSubmode).
; NOTE: The quest is QuestNumbers[CurSaveSlot]: 0 = Q1 table, nonzero = Q2 table.
; ----------------------------------------------------------------------------
Mode2Load_Init_Sub0:
    ; X = CurLevel * 2: offset of the level's entry in the address tables.
    LDA CurLevel
    ASL
    TAX
    LDY CurSaveSlot
    LDA QuestNumbers, Y
    BNE @SecondQuest

    ; Q1: [00:01] = source address.
    LDA Level_BlockAddrsQ1, X
    STA $00
    INX
    LDA Level_BlockAddrsQ1, X
    JMP @Copy

@SecondQuest:
    ; Q2: [00:01] = source address.
    LDA Level_BlockAddrsQ2, X
    STA $00
    INX
    LDA Level_BlockAddrsQ2, X

@Copy:
    STA $01
    JSR Level_SetBlockCopyDest
    JSR Level_CopyBlockAndIncSubmode
    RTS

; ----------------------------------------------------------------------------
; Mode 2 init submode 1: copies the level info block for CurLevel to RAM and ends mode 2's init.
; OUT:  $6B7E-$6C7D filled; GameSubmode = 0; IsUpdatingMode = 1 (next frame runs Mode2Load_Update_Full).
; NOTE: The same block is used in Q1 and Q2; Q2 differences are patched in by Mode2Load_Update_Full.
; ----------------------------------------------------------------------------
Mode2Load_Init_Sub1:
    ; X = CurLevel * 2: offset of the level's entry in Level_InfoAddrs.
    LDA CurLevel
    ASL
    TAX
    LDA Level_InfoAddrs, X
    STA $00
    INX
    LDA Level_InfoAddrs, X
    STA $01
    JSR Level_SetInfoCopyDest
    JSR Level_CopyBlockAndIncSubmode
    LDA #$00                    ; Level_CopyBlockAndIncSubmode left GameSubmode at 2; the update starts at submode 0.
    STA GameSubmode
    INC IsUpdatingMode
    RTS

; ----------------------------------------------------------------------------
; Copies the common data block (transfer buffers, column directory) from ROM to $67F0-$687D.
; OUT:  $67F0-$687D filled; GameSubmode = 0 (stored after Level_CopyBlockAndIncSubmode's increment).
; NOTE: Called once per power-on or reset (Mode_InitGameOrMode, while IsGameInitialized is 0), so the
;       in-place patches made to these buffers during play persist until the next reset.
; ----------------------------------------------------------------------------
Bank_CopyCommonDataToRam:
    LDX #$00                    ; [00:01] = address of the common data block in ROM.
    LDA Bank_CommonDataBlockAddr, X
    STA $00
    INX
    LDA Bank_CommonDataBlockAddr, X
    STA $01
    JSR Bank_SetCommonDataCopyDest
    JSR Level_CopyBlockAndIncSubmode
    LDA #$00
    STA GameSubmode
    RTS

; ----------------------------------------------------------------------------
; Sets the copy destination for a level block: $687E (LevelBlockAttrsA) through $6B7D, 768 bytes.
; OUT:  [02:03] = destination address; [04:05] = address of the last byte to write.
; ----------------------------------------------------------------------------
Level_SetBlockCopyDest:
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
Level_SetInfoCopyDest:
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
Bank_SetCommonDataCopyDest:
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
Level_CopyBlockAndIncSubmode:
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
; IN:   CurLevel; QuestNumbers[CurSaveSlot]; bank 6 switched in (called from Mode2Load_Update in bank 7,
;       which then advances to mode 3).
; OUT:  Q2, levels 1-9: level info bytes from offset $29 (LevelInfo_ShortcutOrItemPosArray) through
;       offset $29 + size are replaced from LevelInfoUWQ2Replacements<level>.
;       Q2, overworld: 8 bytes written through Quest_Q2OverworldPatchOffsets/Values and 7 by
;       immediate stores (rooms $0B, $3C, $74).
; NOTE: The level info loop copies indexes size down to 0, that is size + 1 bytes: one byte more than
;       the table holds. The extra byte (first byte of the next table; for level 9 the low byte of
;       Quest_Q2LevelInfoPatchAddrs) lands in LevelInfo_StatusBarMapTransferBuf after the new end
;       marker, where the transfer never reads it.
; NOTE: So in Q2 a dungeon keeps Q1's level info only up to offset $28; item positions, start room,
;       Triforce room, cellar list, boss room and maps come from the table, and the "LEVEL-n" number
;       shown differs from CurLevel for levels 2-5, 7 and 8 (see bank6/quest2-replacements.asm).
; ----------------------------------------------------------------------------
Mode2Load_Update_Full:
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
    LDA Quest_Q2LevelInfoPatchAddrs-2, Y
    STA $00
    LDA Quest_Q2LevelInfoPatchAddrs-1, Y
    STA $01

    ; Y = this level's entry of the size table (again no overworld entry): the highest index copied.
    LDY Quest_Q2LevelInfoPatchSizes-1, X

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
    LDX Quest_Q2OverworldPatchOffsets, Y
    LDA Quest_Q2OverworldPatchValues, Y
    STA LevelBlockAttrsB, X
    DEY
    BPL @ReplaceRoomBytes
    ; Then 7 immediate stores (decimal offsets 11, 60, 116 = rooms $0B, $3C, $74).
    LDA #$7B
    STA LevelBlockAttrsD+11     ; Room $0B: layout id $0A -> $7B (RoomLayout_RoomOW7B in bank 5).
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
