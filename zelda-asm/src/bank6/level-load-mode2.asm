; ============================================================================
; FILE:     bank6/level-load-mode2.asm   BANK: 6 (switchable, $8000-$BFFF)
; UPSTREAM: Z_06.asm 15-263
; SUBJECTS: (none)
; TOPIC:    Level block/info address tables, mode 2 copy to RAM, quest-2 patch
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0025 LevelBlockAddrsQ1                data
;   L0037 LevelInfoAddrs                   data
;   L0049 CommonDataBlockAddr_Bank6        data
;   L0052 LevelBlockAddrsQ2                data
;   L0064 InitMode2_Submodes               code
;   L0068 InitMode2_Submodes_JumpTable     data
;   L0072 InitMode2_Sub0                   code
;   L0101 InitMode2_Sub1                   code
;   L0118 CopyCommonDataToRam              code
;   L0136 FetchLevelBlockDestInfo          code  Returns:
;   L0152 FetchLevelInfoDestInfo           code  Returns:
;   L0163 FetchDestAddrForCommonDataBlock  code
;   L0181 CopyBlock                        code  Params:
;   L0213 UpdateMode2Load_Full             code
; ============================================================================

LevelBlockAddrsQ1:
    .ADDR LevelBlockOW
    .ADDR LevelBlockUW1Q1
    .ADDR LevelBlockUW1Q1
    .ADDR LevelBlockUW1Q1
    .ADDR LevelBlockUW1Q1
    .ADDR LevelBlockUW1Q1
    .ADDR LevelBlockUW1Q1
    .ADDR LevelBlockUW2Q1
    .ADDR LevelBlockUW2Q1
    .ADDR LevelBlockUW2Q1

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

CommonDataBlockAddr_Bank6:
    .ADDR CommonDataBlock_Bank6

LevelBlockAddrsQ2:
    .ADDR LevelBlockOW
    .ADDR LevelBlockUW1Q2
    .ADDR LevelBlockUW1Q2
    .ADDR LevelBlockUW1Q2
    .ADDR LevelBlockUW1Q2
    .ADDR LevelBlockUW1Q2
    .ADDR LevelBlockUW1Q2
    .ADDR LevelBlockUW2Q2
    .ADDR LevelBlockUW2Q2
    .ADDR LevelBlockUW2Q2

InitMode2_Submodes:
    LDA GameSubmode
    JSR TableJump

InitMode2_Submodes_JumpTable:
    .ADDR InitMode2_Sub0
    .ADDR InitMode2_Sub1

InitMode2_Sub0:
    ; Copy level block for level.
    LDA CurLevel
    ASL
    TAX
    LDY CurSaveSlot
    LDA QuestNumbers, Y
    BNE @SecondQuest

    ; First quest.
    LDA LevelBlockAddrsQ1, X
    STA $00
    INX
    LDA LevelBlockAddrsQ1, X
    JMP @Copy

@SecondQuest:
    ; Second quest.
    LDA LevelBlockAddrsQ2, X
    STA $00
    INX
    LDA LevelBlockAddrsQ2, X

@Copy:
    STA $01
    JSR FetchLevelBlockDestInfo
    JSR CopyBlock
    RTS

InitMode2_Sub1:
    ; Copy level info.
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
    LDA #$00
    STA GameSubmode
    INC IsUpdatingMode
    RTS

CopyCommonDataToRam:
    LDX #$00                    ; Get the source address of common data block in ROM.
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

; Returns:
; [$02:03]: destination address
; [$04:05]: end address
;
; Destination address $687E.
FetchLevelBlockDestInfo:
    LDA #$7E
    STA $02
    LDA #$68
    STA $03
    LDA #$7D                    ; End address $6B7D.
    STA $04
    LDA #$6B
    STA $05
    RTS

; Returns:
; [$02:03]: destination address
; [$04:05]: end address
;
; Destination address $6B7E.
FetchLevelInfoDestInfo:
    LDA #$7E
    STA $02
    LDA #$6B
    STA $03
    LDA #$7D                    ; End address $6C7D.
    STA $04
    LDA #$6C
    STA $05
    RTS

FetchDestAddrForCommonDataBlock:
    LDA #$F0                    ; 67F0 to 687D (inclusive)
    STA $02
    LDA #$67
    STA $03
    LDA #$7D
    STA $04
    LDA #$68
    STA $05
    RTS

; Params:
; [$00:01]: source address
; [$02:03]: destination address
; [$04:05]: end destination address
;
; Also increments submode.
;
CopyBlock:
    LDY #$00

@Loop:
    LDA ($00), Y
    STA ($02), Y
    LDA $02
    CMP $04
    BNE @Next
    LDA $03
    CMP $05
    BNE @Next
    INC GameSubmode
    RTS

@Next:
    LDA $02
    CLC
    ADC #$01
    STA $02
    LDA $03
    ADC #$00
    STA $03
    LDA $00
    CLC
    ADC #$01
    STA $00
    LDA $01
    ADC #$00
    STA $01
    JMP @Loop

UpdateMode2Load_Full:
    ; Make replacements for the second quest.
    LDY CurSaveSlot
    LDA QuestNumbers, Y
    BEQ @Exit                   ; If not second quest, then return.
    LDA CurLevel
    BEQ @PatchQ2Rooms           ; If OW, then go patch rooms.
    TAX
    ASL
    TAY

    ; Get an address for the current level that points
    ; to an array of replacement bytes for Q2 UW level info.
    ;
    ; This address array doesn't access the OW element (0).
    ; So, it overlaps the last two bytes of LevelInfoUWQ2Replacements9.
    LDA LevelInfoUWQ2ReplacementAddrs-2, Y
    STA $00
    LDA LevelInfoUWQ2ReplacementAddrs-1, Y
    STA $01

    ; Get the number of replacement bytes for Q2 UW level info.
    ; This address array doesn't access the OW element (0).
    LDY LevelInfoUWQ2ReplacementSizes-1, X

@ReplaceInfoBytes:
    ; Copy bytes from Q2 replacement array to level info
    ; starting at offset $29 (shortcut position array).
    LDA ($00), Y
    STA LevelInfo_ShortcutOrItemPosArray, Y
    DEY
    BPL @ReplaceInfoBytes

@Exit:
    RTS

@PatchQ2Rooms:
    ; Replace attributes of several rooms in OW in second quest.
    LDY #$07

@ReplaceRoomBytes:
    LDX LevelBlockAttrsBQ2ReplacementOffsets, Y
    LDA LevelBlockAttrsBQ2ReplacementValues, Y
    STA LevelBlockAttrsB, X
    DEY
    BPL @ReplaceRoomBytes
    LDA #$7B
    STA LevelBlockAttrsD+11
    LDA #$7B
    STA LevelBlockAttrsD+60
    LDA #$5A
    STA LevelBlockAttrsD+116
    LDA #$72
    STA LevelBlockAttrsA+60
    LDA #$72
    STA LevelBlockAttrsA+116
    LDA #$01
    STA LevelBlockAttrsF+60
    LDA #$00
    STA LevelBlockAttrsF+116
    RTS
