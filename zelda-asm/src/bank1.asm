.INCLUDE "Variables.inc"
.INCLUDE "CommonVars.inc"
.INCLUDE "CaveVars.inc"

.IMPORT __BANK_01_CODE_LOAD__
.IMPORT __BANK_01_CODE_RUN__
.IMPORT __BANK_01_CODE_RUN_END__

.SEGMENT "BANK_01_00"


; Imports from program bank 07

.IMPORT Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor
.IMPORT Anim_FetchObjPosForSpriteDescriptor
.IMPORT Anim_SetObjHFlipForSpriteDescriptor
.IMPORT AnimateItemObject
.IMPORT DestroyMonster
.IMPORT DrawItemInInventory
.IMPORT FindEmptyMonsterSlot
.IMPORT GoToNextModeFromPlay
.IMPORT LevelMasks
.IMPORT Link_EndMoveAndAnimate_Bank1
.IMPORT Link_EndMoveAndDraw_Bank1
.IMPORT MoveObject
.IMPORT ResetObjMetastate
.IMPORT TableJump
.IMPORT TurnOffAllVideo

.EXPORT CheckInitWhirlwindAndBeginUpdate
.EXPORT CheckPassiveTileObjects
.EXPORT CheckPowerTriforceFanfare
.EXPORT CheckTileObjectsBlocking
.EXPORT CopyCommonCodeToRam
.EXPORT InitCave
.EXPORT InitGrumble_Full
.EXPORT InitRupeeStash_Full
.EXPORT InitTrap_Full
.EXPORT InitUnderworldPerson_Full
.EXPORT InitUnderworldPersonLifeOrMoney_Full
.EXPORT SummonWhirlwind
.EXPORT TransferDemoPatterns
.EXPORT UpdateCavePerson
.EXPORT UpdateGrumble_Full
.EXPORT UpdateRupeeStash_Full
.EXPORT UpdateTrap_Full
.EXPORT UpdateUnderworldPerson_Full
.EXPORT UpdateUnderworldPersonLifeOrMoney_Full
.EXPORT UpdateWhirlwind_Full

.INCLUDE "bank1/cave-init-person-text.asm"
.INCLUDE "bank1/cave-person-prices.asm"
.INCLUDE "bank1/person-textbox.asm"
.INCLUDE "bank1/cave-shop.asm"
.INCLUDE "bank1/cave-hint-money-game.asm"
.INCLUDE "bank1/underworld-person-init.asm"
.INCLUDE "bank1/underworld-person-update.asm"
.INCLUDE "bank1/life-or-money.asm"
.INCLUDE "bank1/grumble-goriya.asm"
.INCLUDE "bank1/copy-code-demo-patterns.asm"
.INCLUDE "bank1/whirlwind.asm"
.INCLUDE "bank1/tile-block-power-triforce.asm"
.INCLUDE "bank1/armos-ghini-spawn.asm"
.INCLUDE "bank1/trap-rupee-stash.asm"
.INCLUDE "bank1/common-code-header.asm"
.INCLUDE "bank1/status-bar-rupees.asm"
.INCLUDE "bank1/modeb-cave-save-sprites.asm"
.INCLUDE "bank1/format-decimal-hearts.asm"
.INCLUDE "bank1/room-bounds.asm"
.INCLUDE "bank1/qspeed-direction-math.asm"
.INCLUDE "bank1/shot-aim-boomerang-speed.asm"
.INCLUDE "bank1/bomb-candle-wield.asm"
.INCLUDE "bank1/map-marker-curtain.asm"
.INCLUDE "bank1/take-item.asm"
.INCLUDE "bank1/world-fading.asm"
.INCLUDE "bank1/mazes-lost-woods-hills.asm"
.INCLUDE "bank1/object-animation-draw.asm"
.INCLUDE "bank1/item-sprite-draw.asm"
.INCLUDE "bank1/monster-collisions.asm"
.INCLUDE "bank1/link-collision-harm.asm"
.INCLUDE "bank1/weapon-damage-death.asm"
.INCLUDE "bank1/sword-bomb-arrow-collision.asm"
.INCLUDE "bank1/object-overlap-test.asm"
.INCLUDE "bank1/shove-knockback.asm"
