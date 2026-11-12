; ============================================================================
; FILE:     bank2/moded-save.asm         BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 2915-3201
; SUBJECTS: ModeDSave, Save
; TOPIC:    Mode $D (save): builds file B from the profile, verifies it, copies it over file A, then
;           returns to mode 0 submode 1. Also the slot-info address tables for names and hearts.
; SEE ALSO: bank2/mode1-init-save-files.asm (Save_FormatFileB, checksums, Save_CommitOrRepairFiles);
;           bank2/register-eliminate-init.asm (Save_FetchFileBAddressSet, file B addresses);
;           bank1/modeb-cave-save-sprites.asm (Save_FetchFileAAddressSet, file A addresses);
;           bank2/mode0-demo.asm (Mode0Demo_Update_Sub2, which reloads the slot info after a save).
; KEY FACTS:
;   - Saving takes three updates of mode $D: submode 0 writes file B, submode 1 commits it to file A,
;     submode 2 sets GameMode 0 / GameSubmode 1 (ModeDSave_Update_JumpTable).
;   - Saved: the items block InvItems ($28 bytes), the slot's death count, quest number and name, the
;     world flags ($180 bytes), active flag = 1 (ModeDSave_Update_Sub0BuildFileB).
;   - The items block is copied before the hearts are refilled, so the file holds HeartValues and
;     HeartPartial as they were; only the profile in RAM and SaveSlotHearts get full hearts
;     (ModeDSave_Update_Sub0BuildFileB, Save_StoreSlotHearts).
;   - The copy to file A is guarded: markers and checksum of file A are zeroed first and rewritten
;     last, then IsSaveFileBCommitted = $FF. An interrupted copy leaves file B uncommitted and valid,
;     and the next validation pass repeats the copy (Save_CopyFileBToFileA).
;   - A file B whose checksum does not match is dropped (marked committed without copying)
;     (ModeDSave_Update_Sub1Commit).
; CONTENTS: (generated; L = line in this file)
;   L0040 Save_SlotHeartsAddrsLo           data  TABLE: low bytes of each save slot's entry in SaveSlotHearts; index = save slot.
;   L0046 Save_SlotHeartsAddrsHi           data  TABLE: high bytes of each save slot's entry in SaveSlotHearts; index = save slot.
;   L0052 Save_SlotNameAddrsLo             data  TABLE: low bytes of each save slot's 8-character name in SaveSlotNames; index = save slot.
;   L0058 Save_SlotNameAddrsHi             data  TABLE: high bytes of each save slot's 8-character name in SaveSlotNames; index = save slot.
;   L0066 ModeDSave_Update                 code  Mode $D (save) update: runs the handler selected by GameSubmode.
;   L0071 ModeDSave_Update_JumpTable       data  TABLE: mode $D handlers; index = GameSubmode (0 build file B, 1 commit to file A, 2 leave).
;   L0080 ModeDSave_Update_Sub0BuildFileB  code  Mode $D submode 0: formats file B, copies the profile into it, checksums it, marks it uncommitted.
;   L0155 ModeDSave_Update_Sub1Commit      code  Mode $D submode 1: if file B is pending and its checksum matches, copies it over file A.
;   L0190 Save_CopyFileBToFileA            code  Copies the current slot's file B over file A and its slot info, then marks file B committed.
;   L0287 ModeDSave_Update_Sub2Leave       code  Mode $D submode 2: goes to mode 0 submode 1 (the save-file validation pass), still updating.
;   L0297 Save_FetchSlotNameAddress        code  Points [0C:0D] at the current save slot's name in SaveSlotNames.
;   L0308 Save_StoreSlotHearts             code  Copies HeartValues and HeartPartial to the current save slot's entry in SaveSlotHearts.
; ============================================================================

; TABLE: low bytes of each save slot's entry in SaveSlotHearts; index = save slot.
Save_SlotHeartsAddrsLo:
    .LOBYTES SaveSlotHearts+0
    .LOBYTES SaveSlotHearts+2
    .LOBYTES SaveSlotHearts+4

; TABLE: high bytes of each save slot's entry in SaveSlotHearts; index = save slot.
Save_SlotHeartsAddrsHi:
    .HIBYTES SaveSlotHearts+0
    .HIBYTES SaveSlotHearts+2
    .HIBYTES SaveSlotHearts+4

; TABLE: low bytes of each save slot's 8-character name in SaveSlotNames; index = save slot.
Save_SlotNameAddrsLo:
    .LOBYTES SaveSlotNames+0
    .LOBYTES SaveSlotNames+8
    .LOBYTES SaveSlotNames+16

; TABLE: high bytes of each save slot's 8-character name in SaveSlotNames; index = save slot.
Save_SlotNameAddrsHi:
    .HIBYTES SaveSlotNames+0
    .HIBYTES SaveSlotNames+8
    .HIBYTES SaveSlotNames+16

; ----------------------------------------------------------------------------
; Mode $D (save) update: runs the handler selected by GameSubmode.
; ----------------------------------------------------------------------------
ModeDSave_Update:
    LDA GameSubmode
    JSR Util_TableJump

; TABLE: mode $D handlers; index = GameSubmode (0 build file B, 1 commit to file A, 2 leave).
ModeDSave_Update_JumpTable:
    .ADDR ModeDSave_Update_Sub0BuildFileB
    .ADDR ModeDSave_Update_Sub1Commit
    .ADDR ModeDSave_Update_Sub2Leave

; ----------------------------------------------------------------------------
; Mode $D submode 0: formats file B, copies the profile into it, checksums it, marks it uncommitted.
; NOTE: Also refills Link's hearts in RAM to the number of heart containers.
; ----------------------------------------------------------------------------
ModeDSave_Update_Sub0BuildFileB:
    ; Initialize file B, and copy profile to it.
    ; Calculate and store file B checksum.
    ; Mark file B uncommitted.
    JSR Save_FetchFileBAddressSet
    JSR Save_FormatFileB
    JSR Save_FetchFileBAddressSet
    JSR Save_FetchFileAAddressSet    ; This seems to be called only for $067F put in [$0E:0F].
    LDY #$27                    ; Copy InvItems block ($28 bytes) from profile to file B.

@CopyItems:
    LDA InvItems, Y
    STA ($C0), Y
    DEY
    BPL @CopyItems
    LDY CurSaveSlot
    LDA DeathCounts, Y          ; Copy death count from profile to file B.
    LDY #$00
    STA ($CA), Y
    LDA #$01                    ; We're saving, so make sure the current slot is active.
    STA ($C6), Y
    LDY CurSaveSlot
    STA IsSaveSlotActive, Y     ; Also set the slot active in the save slot info.
    LDA QuestNumbers, Y         ; Copy quest number from profile to file B.
    LDY #$00
    STA ($CC), Y
    JSR Save_FetchSlotNameAddress
    LDY #$07                    ; Copy name from save slot info to file B.

@CopyName:
    LDA ($0C), Y
    STA ($C4), Y
    DEY
    BPL @CopyName
    LDA HeartValues             ; Put in [$0A] a full hearts value for the profile's heart containers.
    AND #$F0
    PHA
    LSR
    LSR
    LSR
    LSR
    STA $0A
    PLA
    ORA $0A
    STA HeartValues
    LDA #$FF                    ; Completely fill the hearts.
    STA HeartPartial
    JSR Save_StoreSlotHearts
    LDY #$00                    ; Copy World Flags ($180 bytes) from profile to file B.

@CopyWorldFlags:
    LDA ($0E), Y
    STA ($C2), Y
    INC $C2
    BNE :+
    INC $C3
:
    INC $0E
    BNE :+
    INC $0F
:
    LDA $0E
    CMP #$FF
    BNE @CopyWorldFlags
    LDA $0F
    CMP #$07
    BNE @CopyWorldFlags
    JSR Save_FetchFileBAddressSet
    JSR Save_StoreFileBChecksumUncommitted
    INC GameSubmode             ; Go to the next submode.
    RTS

; ----------------------------------------------------------------------------
; Mode $D submode 1: if file B is pending and its checksum matches, copies it over file A.
; ----------------------------------------------------------------------------
ModeDSave_Update_Sub1Commit:
    LDY CurSaveSlot
    LDA IsSaveFileBCommitted, Y
    BNE @IncSubmode
    JSR Save_FetchFileBAddressSet
    JSR Save_CalcFileBChecksum
    LDA CurSaveSlot
    ASL
    TAY
    LDA Save_FileBChecksums, Y
    CMP $CE
    BNE @DiscardFileB
    INY
    LDA Save_FileBChecksums, Y
    CMP $CF
    BNE @DiscardFileB
    JSR Save_CopyFileBToFileA        ; The checksum matches, so commit and copy file B to A.

@IncSubmode:
    INC GameSubmode
    RTS

@DiscardFileB:
    ; Discard file B, because it couldn't be validated.
    ; Submode 0 has just stored this checksum, so this path needs file B to change in between.
    LDY CurSaveSlot
    LDA #$FF
    STA IsSaveFileBCommitted, Y
    INC GameSubmode
    RTS

; ----------------------------------------------------------------------------
; Copies the current slot's file B over file A and its slot info, then marks file B committed.
; NOTE: File A's markers and checksum are zeroed during the copy and restored at the end.
; ----------------------------------------------------------------------------
Save_CopyFileBToFileA:
    LDY CurSaveSlot
    LDA #$00                    ; Reset the save file markers.
    STA SaveFileOpenMarkers, Y
    STA SaveFileCloseMarkers, Y
    TYA                         ; Reset the checksum.
    ASL
    TAY
    LDA #$00
    STA SaveFileAChecksums, Y
    INY
    STA SaveFileAChecksums, Y
    JSR Save_FetchFileBAddressSet
    JSR Save_FetchFileAAddressSet    ; This puts $067F in [$0E:0F].
    LDY #$27                    ; Copy InvItems block ($28 bytes) from file B to file A.

@CopyItems:
    LDA ($C0), Y
    STA ($00), Y
    DEY
    BPL @CopyItems
    LDY #$00                    ; Copy these individual bytes from file B to file A.
    LDA ($C6), Y                ; IsSaveSlotActive
    STA ($06), Y
    LDA ($C8), Y                ; The unused per-slot byte ($6D93 + slot to $6515 + slot).
    STA ($08), Y
    LDA ($CA), Y                ; DeathCount
    STA ($0A), Y
    LDA ($CC), Y                ; QuestNumber
    STA ($0C), Y
    LDA ($06), Y                ; Push IsSaveSlotActive from file A to help copy it to save slot info.
    PHA
    LDA ($0A), Y                ; Push death count from file A to help copy it to save slot info.
    PHA
    LDA ($0C), Y                ; Push quest number from file A to help copy it to save slot info.
    PHA
    LDY CurSaveSlot
    PLA                         ; Finish copying quest number from file A to save slot info.
    STA QuestNumbers, Y
    PLA                         ; Finish copying death count from file A to save slot info.
    STA DeathCounts, Y
    PLA                         ; Finish copying IsSaveSlotActive  from file A to save slot info.
    STA IsSaveSlotActive, Y
    LDY #$07                    ; Copy the name from file B to file A.

@CopyName:
    LDA ($C4), Y
    STA ($04), Y
    DEY
    BPL @CopyName

    ; Copy World Flags ($180 bytes) file B to file A.
    ; It counts up from what's in [$0E:0F] ($067F) to $07FF.
    LDY #$00

@CopyWorldFlags:
    LDA ($C2), Y
    STA ($02), Y
    INC $02
    BNE :+
    INC $03
:
    INC $C2
    BNE :+
    INC $C3
:
    INC $0E
    BNE :+
    INC $0F
:
    LDA $0E
    CMP #$FF
    BNE @CopyWorldFlags
    LDA $0F
    CMP #$07
    BNE @CopyWorldFlags
    LDY CurSaveSlot
    LDA #$5A                    ; Write the save file markers.
    STA SaveFileOpenMarkers, Y
    LDA #$A5
    STA SaveFileCloseMarkers, Y
    TYA                         ; Copy the checksum from file B to file A.
    ASL
    TAY
    LDA Save_FileBChecksums, Y
    STA SaveFileAChecksums, Y
    INY
    LDA Save_FileBChecksums, Y
    STA SaveFileAChecksums, Y
    LDY CurSaveSlot
    LDA #$FF                    ; File B has been committed.
    STA IsSaveFileBCommitted, Y
    RTS

; ----------------------------------------------------------------------------
; Mode $D submode 2: goes to mode 0 submode 1 (the save-file validation pass), still updating.
; ----------------------------------------------------------------------------
ModeDSave_Update_Sub2Leave:
    LDA #$00                    ; Go to mode 0 submode 1. Keep it updating.
    STA GameMode
    LDA #$01
    STA GameSubmode
    RTS

; ----------------------------------------------------------------------------
; Points [0C:0D] at the current save slot's name in SaveSlotNames.
; ----------------------------------------------------------------------------
Save_FetchSlotNameAddress:
    LDY CurSaveSlot
    LDA Save_SlotNameAddrsLo, Y
    STA $0C
    LDA Save_SlotNameAddrsHi, Y
    STA $0D
    RTS

; ----------------------------------------------------------------------------
; Copies HeartValues and HeartPartial to the current save slot's entry in SaveSlotHearts.
; ----------------------------------------------------------------------------
Save_StoreSlotHearts:
    LDY CurSaveSlot
    LDA Save_SlotHeartsAddrsLo, Y
    STA $0C
    LDA Save_SlotHeartsAddrsHi, Y
    STA $0D
    LDY #$01                    ; Copy HeartValues and HeartPartial to the slot's SaveSlotHearts entry.

@CopyHearts:
    LDA HeartValues, Y
    STA ($0C), Y
    DEY
    BPL @CopyHearts
    RTS

; Filler ($FF) up to the next routine.
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
