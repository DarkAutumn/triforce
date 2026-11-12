; ============================================================================
; FILE:     bank1/common-code-header.asm BANK: 1 (stored at $A500-$B76F; copied to RAM and run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 2678-2815
; SUBJECTS: (none)
; TOPIC:    BANK_01_CODE segment (runs from RAM): imports and exports
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0012 CommonCodeBlock_Bank1  data
; ============================================================================

CommonCodeBlock_Bank1:

.SEGMENT "BANK_01_CODE"


; Imports from program bank 02

.IMPORT SwitchBank_Local2

; Imports from program bank 04

.IMPORT Gohma_HandleWeaponCollision

; Imports from program bank 05

.IMPORT CopyColumnToTileBuf
.IMPORT InitMode_EnterRoom
.IMPORT ResetInvObjState
.IMPORT SetMMC1Control_Local5
.IMPORT SwitchBank_Local5

; Imports from RAM code bank 06

.IMPORT MenuPalettesTransferBuf

; Imports from program bank 07

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
