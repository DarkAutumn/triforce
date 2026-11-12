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
.IMPORT Item_DrawObject
.IMPORT Obj_DestroyMonster
.IMPORT Item_DrawInInventory
.IMPORT Obj_FindEmptyMonsterSlot
.IMPORT Mode5Play_GoToNextMode
.IMPORT Util_BitMasks
.IMPORT Link_EndMoveAndAnimate_ReturnToBank1
.IMPORT Link_EndMoveAndDraw_ReturnToBank1
.IMPORT Obj_MoveByQSpeed
.IMPORT Obj_ResetMetastate
.IMPORT Util_TableJump
.IMPORT Ppu_TurnOffAllVideo

.EXPORT Whirlwind_InitDropOffAndBeginUpdate
.EXPORT Spawn_CheckTouchedArmosOrGravestone
.EXPORT Item_UpdatePowerTriforceFanfare
.EXPORT Link_CheckSlidingTileObjBlocking
.EXPORT Bank_CopyCommonCodeToRam
.EXPORT Cave_Init
.EXPORT Grumble_Init_Full
.EXPORT RupeeStash_Init_Full
.EXPORT Trap_Init_Full
.EXPORT Person_InitUnderworld_Full
.EXPORT Person_InitLifeOrMoney_Full
.EXPORT Whirlwind_Summon
.EXPORT Chr_TransferDemoPatterns
.EXPORT Cave_UpdatePerson
.EXPORT Grumble_Update_Full
.EXPORT RupeeStash_Update_Full
.EXPORT Trap_Update_Full
.EXPORT Person_UpdateUnderworld_Full
.EXPORT Person_UpdateLifeOrMoney_Full
.EXPORT Whirlwind_Update_Full

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
