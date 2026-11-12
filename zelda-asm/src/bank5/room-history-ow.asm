; ============================================================================
; FILE:     bank5/room-history-ow.asm    BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 3515-3655
; SUBJECTS: Mode9Cellar, ModeBCave, ModeCShortcut, RoomHistory, Unused
; TOPIC:    Submode dispatch for the init of modes $B (enter cave), $C (shortcut stairs) and 9 (enter
;           cellar); the overworld side of the room history: how many monsters a room makes on entry and
;           how its kills are saved on leaving; two unused sprite helpers.
; SEE ALSO: bank5/room-history-uw.asm (the dungeon versions); bank5/enter-room.asm (Room_EnterCommon
;           builds the monster count in [03] and calls RoomHistory_ModifyObjCountOW);
;           bank5/warps-stairs-caves.asm (RoomHistory_SaveKillCount picks the OW or UW saver);
;           bank7/init-mode5-play.asm (Room_RunEntryTasks adds the room to RoomHistory);
;           bank7/init-mode-dispatch.asm (RoomHistory_Clear; the mode init table that points here).
; KEY FACTS:
;   - Overworld room flags bits 0-2 hold the room's saved kill count, 0-7; 7 means "all dead".
;     RoomHistory_SaveKillCountOW, RoomHistory_ModifyObjCountOW
;   - RoomHistory holds 6 room ids. A room with saved count 7 makes no monsters while it is still in
;     the history; once it has dropped out, entering it clears the count and the full set returns.
;     RoomHistory_ModifyObjCountOW
;   - A saved count of 1-6 is subtracted from the room's monster count on every entry, whether or not
;     the room is in the history: a partly cleared overworld room never refills by itself.
;     RoomHistory_ModifyObjCountOW
;   - On leaving: saved count = old count + RoomKillCount, capped at 7; it is 7 at once when
;     RoomKillCount >= RoomObjCount, which includes a room that made no monsters. RoomHistory_SaveKillCountOW
;   - Cave (mode $B) and shortcut (mode $C) inits differ only in submode 3, the layout used.
;     ModeBCave_Init_JumpTable, ModeCShortcut_Init_JumpTable
; CONTENTS: (generated; L = line in this file)
;   L0043 ModeBCave_Init                              code  Runs the current submode of the init of mode $B (entering an overworld cave).
;   L0048 ModeBCave_Init_JumpTable                    data  TABLE: mode $B init submode handlers; index = GameSubmode (0-8); value = routine address.
;   L0063 ModeCShortcut_Init                          code  Runs the current submode of the init of mode $C (entering the shortcut stairs cave).
;   L0069 ModeCShortcut_Init_JumpTable                data  TABLE: mode $C init submode handlers; index = GameSubmode (0-8); value = routine address.
;   L0093 RoomHistory_ModifyObjCountOW                code  Reduces the monster count of the overworld room being entered by its saved kill count.
;   L0146 RoomHistory_SaveKillCountOW                 code  Adds the kills made in the overworld room being left to the kill count in its room flags.
;   L0173 Mode9Cellar_Init                            code  Runs the current submode of the init of mode 9 (entering a dungeon cellar).
;   L0178 Mode9Cellar_Init_JumpTable                  data  TABLE: mode 9 init submode handlers; index = GameSubmode (0-9); value = routine address.
;   L0196 Unused_DrawObjectNotMirroredOverLink_Bank5  code  Unused copy of bank 4's Sprite_DrawObjNotMirroredOverLink: draws object X unmirrored in sprites $10/$11.
;   L0203 Unused_DrawObjectOverLinkTail_Bank5         code  Unused shared tail of the copy above (bank 4: Sprite_DrawObjMirroredOverLink after its LDY #$01).
; ============================================================================

; ----------------------------------------------------------------------------
; Runs the current submode of the init of mode $B (entering an overworld cave).
; IN:   GameSubmode = 0-8
; ----------------------------------------------------------------------------
ModeBCave_Init:
    LDA GameSubmode
    JSR Util_TableJump

; TABLE: mode $B init submode handlers; index = GameSubmode (0-8); value = routine address.
ModeBCave_Init_JumpTable:
    .ADDR Mode_InitSubroom_Sub0                                  ; 0
    .ADDR ModeBCave_Init_Sub1_CueCavePalettes                                        ; 1
    .ADDR Mode_InitSubroom_AdvanceSubmode                        ; 2
    .ADDR ModeBCave_Init_LayOutCave                            ; 3: lay out RoomLayout_CaveOW
    .ADDR Mode_CopyNextRowAndAdvanceSubmodeWhenDone     ; 4: send the tile rows to the PPU
    .ADDR ModeBCave_Init_Sub5_FillAttrsTransferTopHalf        ; 5
    .ADDR Mode_InitSubroom_TransferBottomHalfAttrs                  ; 6
    .ADDR ModeBCave_Init_Sub7_EnterCave                             ; 7: Link at the cave mouth ($70, $DD)
    .ADDR ModeBCave_Init_Sub8_WalkIn                                     ; 8

; ----------------------------------------------------------------------------
; Runs the current submode of the init of mode $C (entering the shortcut stairs cave).
; IN:   GameSubmode = 0-8
; ----------------------------------------------------------------------------
ModeCShortcut_Init:
    LDA GameSubmode
    JSR Util_TableJump

; TABLE: mode $C init submode handlers; index = GameSubmode (0-8); value = routine address.
; Same as ModeBCave_Init_JumpTable except submode 3.
ModeCShortcut_Init_JumpTable:
    .ADDR Mode_InitSubroom_Sub0                                  ; 0
    .ADDR ModeBCave_Init_Sub1_CueCavePalettes                                        ; 1
    .ADDR Mode_InitSubroom_AdvanceSubmode                        ; 2
    .ADDR ModeCShortcut_Init_LayOutShortcut                       ; 3: lay out RoomLayout_ShortcutCaveOW
    .ADDR Mode_CopyNextRowAndAdvanceSubmodeWhenDone     ; 4: send the tile rows to the PPU
    .ADDR ModeBCave_Init_Sub5_FillAttrsTransferTopHalf        ; 5
    .ADDR Mode_InitSubroom_TransferBottomHalfAttrs                  ; 6
    .ADDR ModeBCave_Init_Sub7_EnterCave                             ; 7
    .ADDR ModeBCave_Init_Sub8_WalkIn                                     ; 8

; ----------------------------------------------------------------------------
; Reduces the monster count of the overworld room being entered by its saved kill count.
; IN:   [02] = monster list id; [03] = monster count from the room attributes; RoomId
; OUT:  [03] = monsters to make; [02] = 0 as well when the saved count exceeds [03], or is 7 with
;       the room in the history;
;       [00:01] = room flags address, Y = RoomId; may clear bits 0-2 of the room's flags
; NOTE: Saved kill count = room flags bits 0-2. Cases:
;         - count 7 and the room is not among the 6 in RoomHistory: the count is cleared and the room
;           makes its full set (this is the only way an overworld room refills);
;         - count 7 and the room is in RoomHistory: nothing is made;
;         - count 0: full set;
;         - count 1-6: [03] - count monsters (none if that is negative), in or out of the history.
; ----------------------------------------------------------------------------
RoomHistory_ModifyObjCountOW:
    ; Look for the room in the history (6 entries).
    LDY #$05
    LDA RoomId
:
    CMP RoomHistory, Y
    BEQ @CheckKillCount         ; Found: the saved kill count applies as it is.
    DEY
    BPL :-

    ; Not in the history. Bits 0-2 of the room flags are the saved kill count.
    JSR RoomFlags_Get
    AND #$07
    CMP #$07
    BNE @CheckKillCount         ; Not "all dead": the saved kill count applies as it is.

    ; Not in the history and all dead: clear the saved kill count, and keep the full count in [03].
    LDA ($00), Y
    AND #$F8
    STA ($00), Y

@Exit:
    RTS

@CheckKillCount:
    ; Saved kill count 0: leave the monster count alone.
    JSR RoomFlags_Get
    AND #$07
    BEQ @Exit

    ; Saved kill count 7: make nothing. Else subtract it from the monster count.
    CMP #$07
    BEQ @ResetObjList
    STA $04
    LDA $03
    SEC
    SBC $04
    BPL :+                      ; Result >= 0: that many monsters are made.

@ResetObjList:
    LDA #$00                    ; No monsters: clear the list id and the count.
    STA $02
:
    STA $03
    RTS

; ----------------------------------------------------------------------------
; Adds the kills made in the overworld room being left to the kill count in its room flags.
; IN:   RoomKillCount; RoomObjCount; RoomId
; OUT:  room flags bits 0-2 = min(7, old count + RoomKillCount), or 7 when everything made was killed
; NOTE: RoomKillCount >= RoomObjCount stores 7 ("all dead"); RoomObjCount is the count after the
;       history reduction, so a room that made no monsters is also stored as 7. [02] is clobbered.
; ----------------------------------------------------------------------------
RoomHistory_SaveKillCountOW:
    JSR RoomFlags_Get
    AND #$07                    ; [02] = kill count saved so far.
    STA $02
    LDA ($00), Y                ; Clear the kill count bits of the room's flags.
    AND #$F8
    STA ($00), Y
    LDA RoomKillCount
    CMP RoomObjCount
    BCS @LimitCount             ; Killed everything the room made: store 7.
    AND #$07
    CLC                         ; Add the kills of this visit to the saved count.
    ADC $02
    CMP #$07
    BCC :+                      ; A total of 7 or more is stored as 7.

@LimitCount:
    LDA #$07
:
    ORA ($00), Y                ; Put the new count into the room's flags.
    STA ($00), Y
    RTS

; ----------------------------------------------------------------------------
; Runs the current submode of the init of mode 9 (entering a dungeon cellar).
; IN:   GameSubmode = 0-9
; ----------------------------------------------------------------------------
Mode9Cellar_Init:
    LDA GameSubmode
    JSR Util_TableJump

; TABLE: mode 9 init submode handlers; index = GameSubmode (0-9); value = routine address.
Mode9Cellar_Init_JumpTable:
    .ADDR Mode_InitSubroom_Sub0                                  ; 0
    .ADDR Mode9Cellar_Init_Sub1_FadeOutLevel                                  ; 1
    .ADDR Mode_InitSubroom_AnimateFade                           ; 2
    .ADDR Mode9Cellar_Init_LayOutCellar                         ; 3: tunnel or treasure cellar layout
    .ADDR Mode_CopyNextRowAndAdvanceSubmodeWhenDone     ; 4: send the tile rows to the PPU
    .ADDR Mode9Cellar_Init_Sub5_TransferAttrs                               ; 5
    .ADDR Mode9Cellar_Init_Sub6_FadeInCellar                                 ; 6
    .ADDR Mode_InitSubroom_AnimateFade                           ; 7
    .ADDR Mode9Cellar_Init_Sub8_EnterCellar                                 ; 8
    .ADDR Mode9Cellar_Init_Sub9_WalkIn                                  ; 9

; ----------------------------------------------------------------------------
; Unused copy of bank 4's Sprite_DrawObjNotMirroredOverLink: draws object X unmirrored in sprites $10/$11.
; IN:   A = frame image; X = object slot
; NOTE: Nothing in the program refers to this label. The LDY #$01 after the branch is the body of the
;       mirrored entry point, which has no label in this copy and cannot be reached.
; ----------------------------------------------------------------------------
Unused_DrawObjectNotMirroredOverLink_Bank5:
    LDY #$00                    ; Not mirrored.
    BEQ Unused_DrawObjectOverLinkTail_Bank5
    LDY #$01                    ; Mirrored (unreachable).

; Unused shared tail of the copy above (bank 4: Sprite_DrawObjMirroredOverLink after its LDY #$01).
; IN: A = frame image; X = object slot; Y = 1 to mirror, 0 not.
Unused_DrawObjectOverLinkTail_Bank5:
    STY $0C                     ; [0C] = mirrored flag.
    LDY ObjType, X              ; Animation index = object type + 1.
    INY
    STA $0D                     ; [0D] = frame image.
    STY $0E                     ; [0E] = animation index.
    STX $08                     ; [08] = object slot.
    LDA #$40                    ; Left half goes to sprite $10 (offset $40 in Sprites),
    STA LeftSpriteOffset
    LDA #$44                    ; right half to sprite $11 (offset $44).
    JMP Anim_DrawObjectWithAnimAndSpecificSprites
