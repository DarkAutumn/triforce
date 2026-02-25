# pylint: disable=all
"""Assembly address constants from zelda-asm/src/Variables.inc.

These are the canonical NES RAM addresses. Use these instead of magic numbers
in tests so the source of truth is clear.
"""

# Object tables - each is an array indexed by slot number (0-18)
OBJ_X           = 0x070   # ObjX - X position
OBJ_Y           = 0x084   # ObjY - Y position
OBJ_DIR         = 0x098   # ObjDir - direction
OBJ_STATE       = 0x0AC   # ObjState - state byte
OBJ_STUN_TIMER  = 0x03D   # ObjStunTimer
OBJ_TYPE        = 0x34F   # ObjType - object type ID
OBJ_METASTATE   = 0x405   # ObjMetastate - spawn/death state
OBJ_HP          = 0x485   # ObjHP - health
OBJ_POS_FRAC    = 0x3A8   # ObjPosFrac / Item_ObjItemLifetime (union)

# Tile layout
TILE_LAYOUT     = 0xD30   # PlayAreaTiles (retro-remapped from $6530)
TILE_LAYOUT_LEN = 0x2C0   # 704 bytes = 32 x 22

# Game state
CUR_LEVEL       = 0x010   # CurLevel
GAME_MODE       = 0x012   # GameMode
ROOM_ID         = 0x0EB   # RoomId

# Link (slot 0 of object tables)
LINK_X          = 0x070   # ObjX[0]
LINK_Y          = 0x084   # ObjY[0]
LINK_DIR        = 0x098   # ObjDir[0]
LINK_STATE      = 0x0AC   # ObjState[0]

# Weapon slots (slot numbers, add to table base for address)
SLOT_LINK       = 0x00
SLOT_SWORD      = 0x0D    # Sword / Rod
SLOT_BEAM       = 0x0E    # Sword shot / Magic shot
SLOT_BOOMERANG  = 0x0F    # Boomerang / Food
SLOT_BOMB1      = 0x10    # Bomb / Fire 1
SLOT_BOMB2      = 0x11    # Bomb / Fire 2
SLOT_ARROW      = 0x12    # Arrow / Rod shot

# Weapon ObjState addresses (OBJ_STATE + slot)
SWORD_STATE     = OBJ_STATE + SLOT_SWORD     # 0xB9
BEAM_STATE      = OBJ_STATE + SLOT_BEAM      # 0xBA
BOOMERANG_STATE = OBJ_STATE + SLOT_BOOMERANG # 0xBB
BOMB1_STATE     = OBJ_STATE + SLOT_BOMB1     # 0xBC
BOMB2_STATE     = OBJ_STATE + SLOT_BOMB2     # 0xBD
ARROW_STATE     = OBJ_STATE + SLOT_ARROW     # 0xBE

# Hearts / Health
HEART_VALUES    = 0x66F   # HeartValues: hi nibble = containers-1, lo nibble = filled
HEART_PARTIAL   = 0x670   # HeartPartial: 0=empty, 1-7F=half, 80-FF=full

# Inventory / Equipment
INV_SWORD       = 0x657
INV_BOMBS       = 0x658
INV_ARROWS      = 0x659
INV_BOW         = 0x65A
INV_CANDLE      = 0x65B
INV_WHISTLE     = 0x65C
INV_FOOD        = 0x65D
INV_POTION      = 0x65E
INV_MAGIC_ROD   = 0x65F
INV_RAFT        = 0x660
INV_BOOK        = 0x661
INV_RING        = 0x662
INV_STEP_LADDER = 0x663
INV_MAGIC_KEY   = 0x664
INV_POWER_BRACE = 0x665
INV_LETTER      = 0x666
INV_COMPASS     = 0x667
INV_MAP         = 0x668
INV_COMPASS9    = 0x669
INV_MAP9        = 0x66A
INV_CLOCK       = 0x66C
INV_RUPEES      = 0x66D
INV_KEYS        = 0x66E
INV_TRIFORCE    = 0x671
INV_TRIFORCE_P  = 0x672   # LastBossDefeated in assembly
INV_BOOMERANG   = 0x674
INV_MAG_BOOMER  = 0x675
INV_MAG_SHIELD  = 0x676
INV_BOMB_MAX    = 0x67C
INV_RUPEES_ADD  = 0x67D
SELECTED_ITEM   = 0x656

# Sound
TUNE0           = 0x605   # sound_pulse_1

# RNG
RNG_BASE        = 0x018   # rng_0 through rng_11 at 0x018-0x023

# Kill tracking
WORLD_KILL_CNT  = 0x627   # kill_streak
# Note: room_kills is at 0x34F which overlaps ObjType[0]

# Treasure
TREASURE_X      = 0x083
TREASURE_Y      = 0x097
TREASURE_FLAG   = 0x0BF

# NES buttons (indices into MultiBinary(9) action array)
BTN_B           = 0   # B button - use selected item
BTN_SELECT      = 2
BTN_START       = 3
BTN_UP          = 4
BTN_DOWN        = 5
BTN_LEFT        = 6
BTN_RIGHT       = 7
BTN_A           = 8   # A button - sword

# Object table slot count (slots 0-11, 12 entries)
OBJ_SLOT_COUNT  = 0x0C
# Enemy slots are 1-B (11 slots)
ENEMY_SLOT_MIN  = 0x01
ENEMY_SLOT_MAX  = 0x0B
