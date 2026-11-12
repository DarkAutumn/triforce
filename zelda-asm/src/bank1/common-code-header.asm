; ============================================================================
; FILE:     bank1/common-code-header.asm BANK: 1 (RAM code: stored at $A500, run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 2678-2815
; SUBJECTS: Bank
; TOPIC:    Start of the common code block: the BANK_01_CODE segment, stored in bank 1 but linked to
;           run from cartridge RAM, where any bank can call it. This file only switches the segment
;           and declares the block's imports and exports; the code follows in the next files.
; SEE ALSO: bank1/copy-code-demo-patterns.asm (Bank_CopyCommonCodeToRam copies the block to RAM);
;           bank7/init-mode-dispatch.asm (Mode_InitGameOrMode and Mode2Load_Init call the copy);
;           bank1/status-bar-rupees.asm (first code of the block) through bank1/shove-knockback.asm
;           (last); bank1/modeb-cave-save-sprites.asm (Save_FileBChecksums: save data written over the
;           RAM copy); bank6/level-blocks-common-data.asm (the BANK_06_DATA block copied to RAM at
;           $67F0, home of Transfer_MenuPalettesBuf).
; KEY FACTS:
;   - Everything from here to the end of bank 1's code is stored in ROM at $A500-$B76F and runs at
;     $6C90-$7EFF in cartridge RAM (segment BANK_01_CODE, $1270 bytes). All its labels are RAM
;     addresses. Bank_CommonCodeBlock.
;   - Bank_CopyCommonCodeToRam copies the block at power-on (Mode_InitGameOrMode) and again every time a
;     level is loaded (Mode2Load_Init, submode 0).
;   - Being in RAM, the block is callable whichever ROM bank is switched in: banks 2, 4, 5 and 7
;     import its routines. It holds collision and damage, drawing, item pickup, q-speed movement
;     helpers, the status bar text and sound request helpers (the .EXPORT list below).
;   - The block calls bank 4 and bank 5 routines (Gohma_HandleWeaponCollision, Transfer_CopyColumnToTileBuf,
;     Room_EnterCommon, Obj_ResetWeaponSlotsAndLadder) without checking the bank: such a path is valid only
;     while that bank is switched in.
;   - Mmc1_SwitchBank_Bank2, Mmc1_SwitchBank_Bank5 and Mmc1_SetControl_Bank5 are per-bank copies of the MMC1
;     routines; the copies sit at the same addresses ($BFAC, $BF98) in every switchable bank.
;   - The RAM copy is not protected: saving a game writes file B's bookkeeping bytes into it
;     ($6D90-$6DA1, see Save_FileBChecksums); the next level load restores the code.
; CONTENTS: (generated; L = line in this file)
;   L0037 Bank_CommonCodeBlock  data  DATA: marker without bytes: the ROM address ($A500) where the common code block is stored.
; ============================================================================

; DATA: marker without bytes: the ROM address ($A500) where the common code block is stored.
; The label is still in segment BANK_01_00 and no code refers to it. The segment switch below
; starts the block, whose labels are linked at $6C90 and up.
Bank_CommonCodeBlock:

.SEGMENT "BANK_01_CODE"


; Imports from program bank 02 (a per-bank copy of the bank switch routine)

.IMPORT Mmc1_SwitchBank_Bank2

; Imports from program bank 04 (valid only while bank 4 is switched in)

.IMPORT Gohma_HandleWeaponCollision

; Imports from program bank 05 (the first three are valid only while bank 5 is switched in; the
; last two are per-bank copies of the MMC1 routines)

.IMPORT Transfer_CopyColumnToTileBuf
.IMPORT Room_EnterCommon
.IMPORT Obj_ResetWeaponSlotsAndLadder
.IMPORT Mmc1_SetControl_Bank5
.IMPORT Mmc1_SwitchBank_Bank5

; Imports from bank 06's common data block, which runs from RAM at $67F0 (segment BANK_06_DATA)

.IMPORT Transfer_MenuPalettesBuf

; Imports from program bank 07 (the fixed bank, always mapped)

.IMPORT Anim_FetchObjPosForSpriteDescriptor
.IMPORT Anim_AnimateObjWalking
.IMPORT Mode_End
.IMPORT RoomFlags_Get
.IMPORT SwordShot_HandleBlocked
.IMPORT Util_BitMasks
.IMPORT Link_EndMoveAndAnimate
.IMPORT Obj_MoveByQSpeed
.IMPORT Palette_PatchLinkColorAndCueLevelPalettes
.IMPORT Obj_ResetMovingDir
.IMPORT Room_RunEntryTasks_SetBounds
.IMPORT Palette_SaveSlotRowOffsets
.IMPORT Shove_SetDirAndDistance
.IMPORT DeadDummy_StartDeathSparkle

.EXPORT Dir_CalcAimSpeedIndex
.EXPORT Math_Abs
.EXPORT Math_Add1ToInt16At0
.EXPORT Math_Add1ToInt16At2
.EXPORT Math_Add1ToInt16At4
.EXPORT Obj_AddQSpeedToPosFrac
.EXPORT Math_AddToInt16At0
.EXPORT Math_AddToInt16At2
.EXPORT Math_AddToInt16At4
.EXPORT Anim_EndWriteSprite
.EXPORT Anim_SetSpriteDescriptorAttributes
.EXPORT Anim_SetSpriteDescriptorRedPaletteRow
.EXPORT Anim_WriteItemSprites
.EXPORT Anim_WriteLevelPaletteSprite
.EXPORT Anim_WriteSpecificItemSprites
.EXPORT Anim_WriteSpecificSprite
.EXPORT Anim_WriteSprite
.EXPORT Anim_WriteSpritePairNotFlashing
.EXPORT Anim_WriteStaticItemSpritesWithAttributes
.EXPORT Anim_AnimateAndDrawObjectWalking
.EXPORT Palette_AnimateFade
.EXPORT Shove_Begin
.EXPORT Mode_BeginUpdate
.EXPORT Room_BoundDir
.EXPORT Room_BoundDirA
.EXPORT Room_BoundDirHorizontally
.EXPORT Room_BoundDirVertically
.EXPORT Link_CheckMonsterTouch
.EXPORT Link_CheckMonsterTouch_NoReset
.EXPORT Maze_CheckStep
.EXPORT Arrow_CheckArrowOrRodVsMonster
.EXPORT Bomb_CheckBombOrFireVsMonster
.EXPORT Collision_CheckMonsterVsWeaponsAndLink
.EXPORT Sword_CheckMonsterCollision
.EXPORT SwordShot_CheckMonsterCollision
.EXPORT Person_BlockLinkMovingUp
.EXPORT Inv_CompareHeartsToContainers
.EXPORT Sprite_CycleCurIndex
.EXPORT Damage_DealToMonster
.EXPORT Obj_ResetSlotFields
.EXPORT Collision_TestPoints
.EXPORT Anim_DrawObjectMirrored
.EXPORT Anim_DrawObjectNotMirrored
.EXPORT Anim_DrawObjectWithAnim
.EXPORT Anim_DrawObjectWithAnimAndSpecificSprites
.EXPORT Anim_DrawObjectWithType
.EXPORT Save_FetchFileAAddressSet
.EXPORT Save_FileBChecksums
.EXPORT Text_FormatDecimalByte
.EXPORT Text_FormatHeartRows
.EXPORT StatusBar_FormatText
.EXPORT Dir_GetDirsAndDistancesToTarget
.EXPORT Obj_GetMiddle
.EXPORT Dir_GetOpposite
.EXPORT RoomFlags_GetItemTaken
.EXPORT Room_GetShortcutOrItemXY
.EXPORT Room_GetShortcutOrItemXYForRoomId
.EXPORT Sprite_HideObjectSprites
.EXPORT ModeBCave_Init_Sub7_EnterCave
.EXPORT Item_IdToDescriptor
.EXPORT Item_IdToSlot
.EXPORT Item_SlotToPaletteOffsetsOrValues
.EXPORT Link_BeHarmed
.EXPORT Ppu_MapScreenPosToAddr
.EXPORT Obj_MoveShot
.EXPORT Math_Negate
.EXPORT Person_Draw
.EXPORT Link_PlaceWeaponAhead
.EXPORT Link_UseItem_SetState10
.EXPORT Link_UseItem_SetAnimCounter1
.EXPORT Link_UseItem_SetWeaponState
.EXPORT Boomerang_PlaySfx
.EXPORT Sfx_Play
.EXPORT Sample_Play
.EXPORT Sprite_ResetCurIndex
.EXPORT Room_ResetTileObjInfo
.EXPORT Shove_ResetAndClearInvincibility
.EXPORT Dir_ReverseIndexToDir
.EXPORT Boomerang_SetReturnSpeed
.EXPORT Link_PutSpritesBehindSideDoorways
.EXPORT Audio_SilenceAll
.EXPORT Sprite_RollingOffsets
.EXPORT Sprite_HalfOffsetsX
.EXPORT Math_Sub1FromInt16At4
.EXPORT Obj_SubQSpeedFromPosFrac
.EXPORT Item_TryTake
.EXPORT Item_TryTakeRoomItem
.EXPORT Bomb_UpdateFlashEffect
.EXPORT StatusBar_UpdateLinkPositionMarker
.EXPORT StatusBar_UpdatePositionMarker
.EXPORT Mode_UpdateCurtainEffect
.EXPORT Mode_UpdateCurtainEffect_ReturnToBank2
.EXPORT Bomb_Wield
.EXPORT Candle_Wield
.EXPORT Inv_ChangeRupees
.EXPORT Sprite_WriteBlankPrioritySprites
