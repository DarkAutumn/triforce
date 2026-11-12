; ============================================================================
; FILE:     bank5/room-history-ow.asm    BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 3515-3655
; SUBJECTS: Unused
; TOPIC:    Submode dispatch for the init of modes $B (enter cave), $C (shortcut stairs) and 9 (enter
;           cellar); the overworld side of the room history: how many monsters a room makes on entry and
;           how its kills are saved on leaving; two unused sprite helpers.
; SEE ALSO: bank5/room-history-uw.asm (the dungeon versions); bank5/enter-room.asm (InitMode_EnterRoom
;           builds the monster count in [03] and calls ModifyObjCountByHistoryOW);
;           bank5/warps-stairs-caves.asm (SaveKillCount picks the OW or UW saver);
;           bank7/init-mode5-play.asm (RunCrossRoomTasksAndBeginUpdateMode adds the room to RoomHistory);
;           bank7/init-mode-dispatch.asm (ClearRoomHistory; the mode init table that points here).
; KEY FACTS:
;   - Overworld room flags bits 0-2 hold the room's saved kill count, 0-7; 7 means "all dead".
;     SaveKillCountOW, ModifyObjCountByHistoryOW
;   - RoomHistory holds 6 room ids. A room with saved count 7 makes no monsters while it is still in
;     the history; once it has dropped out, entering it clears the count and the full set returns.
;     ModifyObjCountByHistoryOW
;   - A saved count of 1-6 is subtracted from the room's monster count on every entry, whether or not
;     the room is in the history: a partly cleared overworld room never refills by itself.
;     ModifyObjCountByHistoryOW
;   - On leaving: saved count = old count + RoomKillCount, capped at 7; it is 7 at once when
;     RoomKillCount >= RoomObjCount, which includes a room that made no monsters. SaveKillCountOW
;   - Cave (mode $B) and shortcut (mode $C) inits differ only in submode 3, the layout used.
;     InitModeB_JumpTable, InitModeC_JumpTable
; CONTENTS: (generated; L = line in this file)
;   L0043 InitModeB                                   code  Runs the current submode of the init of mode $B (entering an overworld cave).
;   L0048 InitModeB_JumpTable                         data  TABLE: mode $B init submode handlers; index = GameSubmode (0-8); value = routine address.
;   L0063 InitModeC                                   code  Runs the current submode of the init of mode $C (entering the shortcut stairs cave).
;   L0069 InitModeC_JumpTable                         data  TABLE: mode $C init submode handlers; index = GameSubmode (0-8); value = routine address.
;   L0093 ModifyObjCountByHistoryOW                   code  Reduces the monster count of the overworld room being entered by its saved kill count.
;   L0146 SaveKillCountOW                             code  Adds the kills made in the overworld room being left to the kill count in its room flags.
;   L0173 InitMode9                                   code  Runs the current submode of the init of mode 9 (entering a dungeon cellar).
;   L0178 InitMode9_JumpTable                         data  TABLE: mode 9 init submode handlers; index = GameSubmode (0-9); value = routine address.
;   L0196 Unused_DrawObjectNotMirroredOverLink_Bank5  code  Unused copy of bank 4's DrawObjectNotMirroredOverLink: draws object X unmirrored in sprites $10/$11.
;   L0203 Unused_DrawObjectMirroredOverLink_Bank5     code  Unused shared tail of the copy above (bank 4: DrawObjectMirroredOverLink after its LDY #$01).
; ============================================================================

; ----------------------------------------------------------------------------
; Runs the current submode of the init of mode $B (entering an overworld cave).
; IN:   GameSubmode = 0-8
; ----------------------------------------------------------------------------
InitModeB:
    LDA GameSubmode
    JSR TableJump

; TABLE: mode $B init submode handlers; index = GameSubmode (0-8); value = routine address.
InitModeB_JumpTable:
    .ADDR InitModeSubroom_Sub0                                  ; 0
    .ADDR InitModeB_Sub1                                        ; 1
    .ADDR InitModeSubroom_AdvanceSubmode                        ; 2
    .ADDR LayoutCaveAndAvanceSubmode                            ; 3: lay out RoomLayoutOWCave0
    .ADDR CopyNextRowToTransferBufAndAdvanceSubmodeWhenDone     ; 4: send the tile rows to the PPU
    .ADDR InitModeB_Sub5_FillTileAttrsAndTransferTopHalf        ; 5
    .ADDR InitModeAOrB_TransferBottomHalfAttrs                  ; 6
    .ADDR InitModeB_EnterCave_Bank5                             ; 7: Link at the cave mouth ($70, $DD)
    .ADDR InitMode_WalkCave                                     ; 8

; ----------------------------------------------------------------------------
; Runs the current submode of the init of mode $C (entering the shortcut stairs cave).
; IN:   GameSubmode = 0-8
; ----------------------------------------------------------------------------
InitModeC:
    LDA GameSubmode
    JSR TableJump

; TABLE: mode $C init submode handlers; index = GameSubmode (0-8); value = routine address.
; Same as InitModeB_JumpTable except submode 3.
InitModeC_JumpTable:
    .ADDR InitModeSubroom_Sub0                                  ; 0
    .ADDR InitModeB_Sub1                                        ; 1
    .ADDR InitModeSubroom_AdvanceSubmode                        ; 2
    .ADDR LayoutShortcutAndAdvanceSubmode                       ; 3: lay out RoomLayoutOWCave1
    .ADDR CopyNextRowToTransferBufAndAdvanceSubmodeWhenDone     ; 4: send the tile rows to the PPU
    .ADDR InitModeB_Sub5_FillTileAttrsAndTransferTopHalf        ; 5
    .ADDR InitModeAOrB_TransferBottomHalfAttrs                  ; 6
    .ADDR InitModeB_EnterCave_Bank5                             ; 7
    .ADDR InitMode_WalkCave                                     ; 8

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
ModifyObjCountByHistoryOW:
    ; Look for the room in the history (6 entries).
    LDY #$05
    LDA RoomId
:
    CMP RoomHistory, Y
    BEQ @CheckKillCount         ; Found: the saved kill count applies as it is.
    DEY
    BPL :-

    ; Not in the history. Bits 0-2 of the room flags are the saved kill count.
    JSR GetRoomFlags
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
    JSR GetRoomFlags
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
SaveKillCountOW:
    JSR GetRoomFlags
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
InitMode9:
    LDA GameSubmode
    JSR TableJump

; TABLE: mode 9 init submode handlers; index = GameSubmode (0-9); value = routine address.
InitMode9_JumpTable:
    .ADDR InitModeSubroom_Sub0                                  ; 0
    .ADDR InitMode9_FadeToDark                                  ; 1
    .ADDR InitModeSubroom_AnimateFade                           ; 2
    .ADDR LayoutCellarAndAdvanceSubmode                         ; 3: tunnel or treasure cellar layout
    .ADDR CopyNextRowToTransferBufAndAdvanceSubmodeWhenDone     ; 4: send the tile rows to the PPU
    .ADDR InitMode9_TransferAttrs                               ; 5
    .ADDR InitMode9_FadeToLight                                 ; 6
    .ADDR InitModeSubroom_AnimateFade                           ; 7
    .ADDR InitMode9_EnterCellar                                 ; 8
    .ADDR InitMode9_WalkCellar                                  ; 9

; ----------------------------------------------------------------------------
; Unused copy of bank 4's DrawObjectNotMirroredOverLink: draws object X unmirrored in sprites $10/$11.
; IN:   A = frame image; X = object slot
; NOTE: Nothing in the program refers to this label. The LDY #$01 after the branch is the body of the
;       mirrored entry point, which has no label in this copy and cannot be reached.
; ----------------------------------------------------------------------------
Unused_DrawObjectNotMirroredOverLink_Bank5:
    LDY #$00                    ; Not mirrored.
    BEQ Unused_DrawObjectMirroredOverLink_Bank5
    LDY #$01                    ; Mirrored (unreachable).

; Unused shared tail of the copy above (bank 4: DrawObjectMirroredOverLink after its LDY #$01).
; IN: A = frame image; X = object slot; Y = 1 to mirror, 0 not.
Unused_DrawObjectMirroredOverLink_Bank5:
    STY $0C                     ; [0C] = mirrored flag.
    LDY ObjType, X              ; Animation index = object type + 1.
    INY
    STA $0D                     ; [0D] = frame image.
    STY $0E                     ; [0E] = animation index.
    STX $08                     ; [08] = object slot.
    LDA #$40                    ; Left half goes to sprite $10 (offset $40 in Sprites),
    STA LeftSpriteOffset
    LDA #$44                    ; right half to sprite $11 (offset $44).
    JMP DrawObjectWithAnimAndSpecificSprites
