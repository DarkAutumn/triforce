.INCLUDE "Variables.inc"
.INCLUDE "BeginEndVars.inc"

.SEGMENT "BANK_02_00"


; Imports from RAM code bank 01

.IMPORT Anim_SetSpriteDescriptorAttributes
.IMPORT Anim_SetSpriteDescriptorRedPaletteRow
.IMPORT Anim_WriteItemSprites
.IMPORT Anim_WriteSpecificItemSprites
.IMPORT Anim_WriteSpritePairNotFlashing
.IMPORT Mode_BeginUpdate
.IMPORT Anim_DrawObjectMirrored
.IMPORT Anim_DrawObjectNotMirrored
.IMPORT Save_FetchFileAAddressSet
.IMPORT Save_FileBChecksums
.IMPORT Text_FormatDecimalByte
.IMPORT Text_FormatHeartRows
.IMPORT Person_Draw
.IMPORT Room_ResetTileObjInfo
.IMPORT Audio_SilenceAll
.IMPORT Mode_UpdateCurtainEffect_ReturnToBank2

; Imports from RAM code bank 06

.IMPORT Transfer_MenuPalettesBuf

; Imports from program bank 07

.IMPORT Anim_FetchObjPosForSpriteDescriptor
.IMPORT Item_DrawObject
.IMPORT Mode_End
.IMPORT Mode_GoToNext
.IMPORT Sprite_HideAll
.IMPORT Link_EndMoveAndDraw
.IMPORT Util_TableJump
.IMPORT Ppu_TurnOffAllVideo
.IMPORT Ppu_TurnOffVideoAndClearNameTables

.EXPORT Mode0Demo_Init
.EXPORT Mode1Menu_Init_Full
.EXPORT Mode13WinGame_Init_Full
.EXPORT ModeERegister_Init_Full
.EXPORT Chr_TransferCommonPatterns
.EXPORT Mode0Demo_Update
.EXPORT Mode13WinGame_Update
.EXPORT Mode1Menu_Update
.EXPORT ModeDSave_Update
.EXPORT ModeERegister_Update
.EXPORT ModeFEliminate_Update

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
