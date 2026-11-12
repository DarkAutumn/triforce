.INCLUDE "Variables.inc"
.INCLUDE "BeginEndVars.inc"

.SEGMENT "BANK_02_00"


; Imports from RAM code bank 01

.IMPORT Anim_SetSpriteDescriptorAttributes
.IMPORT Anim_SetSpriteDescriptorRedPaletteRow
.IMPORT Anim_WriteItemSprites
.IMPORT Anim_WriteSpecificItemSprites
.IMPORT Anim_WriteSpritePairNotFlashing
.IMPORT BeginUpdateMode
.IMPORT DrawObjectMirrored
.IMPORT DrawObjectNotMirrored
.IMPORT FetchFileAAddressSet
.IMPORT FileBChecksums
.IMPORT FormatDecimalByte
.IMPORT FormatHeartsInTextBuf
.IMPORT Person_Draw
.IMPORT ResetRoomTileObjInfo
.IMPORT SilenceAllSound
.IMPORT UpdateWorldCurtainEffect_Bank2

; Imports from RAM code bank 06

.IMPORT MenuPalettesTransferBuf

; Imports from program bank 07

.IMPORT Anim_FetchObjPosForSpriteDescriptor
.IMPORT AnimateItemObject
.IMPORT EndGameMode
.IMPORT GoToNextMode
.IMPORT HideAllSprites
.IMPORT Link_EndMoveAndDraw
.IMPORT TableJump
.IMPORT TurnOffAllVideo
.IMPORT TurnOffVideoAndClearArtifacts

.EXPORT InitDemo_RunTasks
.EXPORT InitMode1_Full
.EXPORT InitMode13_Full
.EXPORT InitModeEandF_Full
.EXPORT TransferCommonPatterns
.EXPORT UpdateMode0Demo
.EXPORT UpdateMode13WinGame
.EXPORT UpdateMode1Menu
.EXPORT UpdateModeDSave
.EXPORT UpdateModeERegister
.EXPORT UpdateModeFElimination

.INCLUDE "bank2/common-patterns.asm"
.INCLUDE "bank2/mode0-demo.asm"
.INCLUDE "bank2/demo-tables-init.asm"
.INCLUDE "bank2/demo-story-scroll.asm"
.INCLUDE "bank2/title-animation.asm"
.INCLUDE "bank2/register-eliminate-init.asm"
.INCLUDE "bank2/register-eliminate-update.asm"
.INCLUDE "bank2/register-name-input.asm"
.INCLUDE "bank2/register-cursor-sprites.asm"
.INCLUDE "bank2/mode1-init-save-files.asm"
.INCLUDE "bank2/mode1-slot-tiles.asm"
.INCLUDE "bank2/mode1-menu-update.asm"
.INCLUDE "bank2/moded-save.asm"
.INCLUDE "bank2/mode13-ending-init.asm"
.INCLUDE "bank2/mode13-win-game.asm"
.INCLUDE "bank2/credits.asm"
.INCLUDE "bank2/second-quest-bank-stub.asm"
