; ============================================================================
; FILE:     bank1/common-code-header.asm BANK: 1 (RAM code: stored at $A500, run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 2678-2815
; SUBJECTS: (none)
; TOPIC:    Start of the common code block: the BANK_01_CODE segment, stored in bank 1 but linked to
;           run from cartridge RAM, where any bank can call it. This file only switches the segment
;           and declares the block's imports and exports; the code follows in the next files.
; SEE ALSO: bank1/copy-code-demo-patterns.asm (CopyCommonCodeToRam copies the block to RAM);
;           bank7/init-mode-dispatch.asm (InitializeGameOrMode and InitMode2 call the copy);
;           bank1/status-bar-rupees.asm (first code of the block) through bank1/shove-knockback.asm
;           (last); bank1/modeb-cave-save-sprites.asm (FileBChecksums: save data written over the
;           RAM copy); bank6/level-blocks-common-data.asm (the BANK_06_DATA block copied to RAM at
;           $67F0, home of MenuPalettesTransferBuf).
; KEY FACTS:
;   - Everything from here to the end of bank 1's code is stored in ROM at $A500-$B76F and runs at
;     $6C90-$7EFF in cartridge RAM (segment BANK_01_CODE, $1270 bytes). All its labels are RAM
;     addresses. CommonCodeBlock_Bank1.
;   - CopyCommonCodeToRam copies the block at power-on (InitializeGameOrMode) and again every time a
;     level is loaded (InitMode2, submode 0).
;   - Being in RAM, the block is callable whichever ROM bank is switched in: banks 2, 4, 5 and 7
;     import its routines. It holds collision and damage, drawing, item pickup, q-speed movement
;     helpers, the status bar text and sound request helpers (the .EXPORT list below).
;   - The block calls bank 4 and bank 5 routines (Gohma_HandleWeaponCollision, CopyColumnToTileBuf,
;     InitMode_EnterRoom, ResetInvObjState) without checking the bank: such a path is valid only
;     while that bank is switched in.
;   - SwitchBank_Local2, SwitchBank_Local5 and SetMMC1Control_Local5 are per-bank copies of the MMC1
;     routines; the copies sit at the same addresses ($BFAC, $BF98) in every switchable bank.
;   - The RAM copy is not protected: saving a game writes file B's bookkeeping bytes into it
;     ($6D90-$6DA1, see FileBChecksums); the next level load restores the code.
; CONTENTS: (generated; L = line in this file)
;   L0037 CommonCodeBlock_Bank1  data  DATA: marker without bytes: the ROM address ($A500) where the common code block is stored.
; ============================================================================

; DATA: marker without bytes: the ROM address ($A500) where the common code block is stored.
; The label is still in segment BANK_01_00 and no code refers to it. The segment switch below
; starts the block, whose labels are linked at $6C90 and up.
CommonCodeBlock_Bank1:

.SEGMENT "BANK_01_CODE"


; Imports from program bank 02 (a per-bank copy of the bank switch routine)

.IMPORT SwitchBank_Local2

; Imports from program bank 04 (valid only while bank 4 is switched in)

.IMPORT Gohma_HandleWeaponCollision

; Imports from program bank 05 (the first three are valid only while bank 5 is switched in; the
; last two are per-bank copies of the MMC1 routines)

.IMPORT CopyColumnToTileBuf
.IMPORT InitMode_EnterRoom
.IMPORT ResetInvObjState
.IMPORT SetMMC1Control_Local5
.IMPORT SwitchBank_Local5

; Imports from bank 06's common data block, which runs from RAM at $67F0 (segment BANK_06_DATA)

.IMPORT MenuPalettesTransferBuf

; Imports from program bank 07 (the fixed bank, always mapped)

.IMPORT Anim_FetchObjPosForSpriteDescriptor
.IMPORT AnimateObjectWalking
.IMPORT EndGameMode
.IMPORT GetRoomFlags
.IMPORT HandleShotBlocked
.IMPORT LevelMasks
.IMPORT Link_EndMoveAndAnimate
.IMPORT MoveObject
.IMPORT PatchAndCueLevelPalettesTransferAndAdvanceSubmode
.IMPORT ResetMovingDir
.IMPORT RunCrossRoomTasksAndBeginUpdateMode_PlayModesNoCellar
.IMPORT SaveSlotToPaletteRowOffset
.IMPORT SetShoveInfoWith0
.IMPORT UpdateDeadDummy

.EXPORT _CalcDiagonalSpeedIndex
.EXPORT Abs
.EXPORT Add1ToInt16At0
.EXPORT Add1ToInt16At2
.EXPORT Add1ToInt16At4
.EXPORT AddQSpeedToPositionFraction
.EXPORT AddToInt16At0
.EXPORT AddToInt16At2
.EXPORT AddToInt16At4
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
.EXPORT AnimateAndDrawObjectWalking
.EXPORT AnimateWorldFading
.EXPORT BeginShove
.EXPORT BeginUpdateMode
.EXPORT BoundByRoom
.EXPORT BoundByRoomWithA
.EXPORT BoundDirectionHorizontally
.EXPORT BoundDirectionVertically
.EXPORT CheckLinkCollision
.EXPORT CheckLinkCollisionPreinit
.EXPORT CheckMazes
.EXPORT CheckMonsterArrowOrRodCollision
.EXPORT CheckMonsterBombOrFireCollision
.EXPORT CheckMonsterCollisions
.EXPORT CheckMonsterSwordCollision
.EXPORT CheckMonsterSwordShotOrMagicShotCollision
.EXPORT CheckPersonBlocking
.EXPORT CompareHeartsToContainers
.EXPORT CycleCurSpriteIndex
.EXPORT DealDamage
.EXPORT DestroyObject_WRAM
.EXPORT DoObjectsCollide
.EXPORT DrawObjectMirrored
.EXPORT DrawObjectNotMirrored
.EXPORT DrawObjectWithAnim
.EXPORT DrawObjectWithAnimAndSpecificSprites
.EXPORT DrawObjectWithType
.EXPORT FetchFileAAddressSet
.EXPORT FileBChecksums
.EXPORT FormatDecimalByte
.EXPORT FormatHeartsInTextBuf
.EXPORT FormatStatusBarText
.EXPORT GetDirectionsAndDistancesToTarget
.EXPORT GetObjectMiddle
.EXPORT GetOppositeDir
.EXPORT GetRoomFlagUWItemState
.EXPORT GetShortcutOrItemXY
.EXPORT GetShortcutOrItemXYForRoom
.EXPORT HideObjectSprites
.EXPORT InitModeB_EnterCave_Bank5
.EXPORT ItemIdToDescriptor
.EXPORT ItemIdToSlot
.EXPORT ItemSlotToPaletteOffsetsOrValues
.EXPORT Link_BeHarmed
.EXPORT MapScreenPosToPpuAddr
.EXPORT MoveShot
.EXPORT Negate
.EXPORT Person_Draw
.EXPORT PlaceWeapon
.EXPORT PlaceWeaponForPlayerState
.EXPORT PlaceWeaponForPlayerStateAndAnim
.EXPORT PlaceWeaponForPlayerStateAndAnimAndWeaponState
.EXPORT PlayBoomerangSfx
.EXPORT PlayEffect
.EXPORT PlaySample
.EXPORT ResetCurSpriteIndex
.EXPORT ResetRoomTileObjInfo
.EXPORT ResetShoveInfoAndInvincibilityTimer
.EXPORT ReverseDirections
.EXPORT SetBoomerangSpeed
.EXPORT ShowLinkSpritesBehindHorizontalDoors
.EXPORT SilenceAllSound
.EXPORT SpriteOffsets
.EXPORT SpriteRelativeExtents
.EXPORT Sub1FromInt16At4
.EXPORT SubQSpeedFromPositionFraction
.EXPORT TryTakeItem
.EXPORT TryTakeRoomItem
.EXPORT UpdateBombFlashEffect
.EXPORT UpdatePlayerPositionMarker
.EXPORT UpdatePositionMarker
.EXPORT UpdateWorldCurtainEffect
.EXPORT UpdateWorldCurtainEffect_Bank2
.EXPORT WieldBomb
.EXPORT WieldCandle
.EXPORT World_ChangeRupees
.EXPORT WriteBlankPrioritySprites
