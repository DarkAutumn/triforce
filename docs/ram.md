# Notes from fiskbit

```
; object variables are the general object arrays, which have space for all of the object indices except possibly Link.
; dynamic variables are the dynamic object arrays. These are stripped-down arrays that tend to only have space for
; dynamic objects (such as enemies and interactive background elements) and sometimes Link. These arrays do not include
; weapon or treasure slots.
object_timer                := $28  ; Timer used for delay before some enemy action, such as spawning or firing.
object_stun_timer           := $3D  ; - $4E (#$12). No entry 0.

object_pos_x                := $70
object_pos_y                := $84
object_face_direction       := $98  ; UDLR. Also used as item type for treasures.
object_status               := $AC  ; For Object IDs >= #$53, status #$1x enables collision.
object_item_id              := $AC  ; Item ID for dropped items. [WIP Remove this extra name, probably]
object_knockback_direction  := $C0  ; UDLR. Bit 7 indicates whether the knockback is fresh and still needs to be
                                    ; validated. Bit 6 is set on hit if the object doesn't reverse, but is never used.
object_knockback_timer      := $D3  ; Decremented once per movement tick (4 times per frame).

dynamic_id                  := $034F  ; No entry 0.
object_unknown_380          := $0380
object_subgrid_offset       := $0394
object_subpos               := $03A8  ; Sub-pixels for BOTH X and Y. You can only have non-zero subpos in one direction at a time.
object_item_timer           := $03A8  ; Despawn timer for dropped items.
object_speed                := $03BC  ; The subpixel speed per movement tick.
object_animation_timer      := $03D0  ; No entry 0.
object_frame                := $03E4  ; No entry 0. [WIPN This is a confusing name for animations that have different
                                      ; graphics for different directions. Maybe object_animation_step or object_animation_state?]
dynamic_move_direction      := $03F8  ; - $404 (#$D).
dynamic_spawn_state         := $0405  ; 0-4. Also applies somehow to enemies entering from the sides.

; AI-related?
dynamic_has_turned          := $0412  ; - $41E (#$D). Used as counter for push block. Contains object ID if object dies.
dynamic_aggressive_chance   := $041F  ; - $42B (#$D). No entry 0? Appears to be a probability for aggressive turns - if the
                                      ; random number is not greater than this, the object will turn aggressively.
dynamic_unknown_42C         := $042C
dyanmic_unknown_437         := $0437
dynamic_unknown_444         := $0444
dynamic_unknown_451         := $0451  ; Maybe projectile subpos? Used for timing when firing projectiles.
dynamic_unknown_45E         := $045E  ; Maybe projectile-related?
dynamic_unknown_46B         := $046B  ; Used by Digdogger related to splitting.
dynamic_turn_timer          := $0478  ; Appears to be a turn timer, making objects turn toward their target when 0.
dynamic_health              := $0485  ; No entry 0. Comes from Bank7_FB4E, a table of nybbles indexed by object ID.
dynamic_needs_init          := $0492  ; If non-zero, the object will be initialized. Otherwise, its AI will run as normal.
object_collision_tile       := $049E  ; The sample tile representing the last thing the object collided with.
dynamic_immunities          := $04B2  ; The weapons an object is immune to.
  kWeaponTypeSwordRod  := #$01
  kWeaponTypeBoomerang := #$02
  kWeaponTypeArrow     := #$04
  kWeaponTypeBomb      := #$08
  kWeaponTypeRodBeam   := #$10
  kWeaponTypeFire      := #$20
dynamic_properties          := $04BF  ; No entry 0? Seems to indicate something about frames and maybe collision behavior
                                      ; Some sort of table at $1FAFF-1FB5D, indexed possibly by enemy ID (note no enemy 0)
                                      ; Bit 6 = collision box is 8x16 instead of 16x16.
                                      ; Maybe not Short?
  kObjectPropertyCustomCollisionAndSprite := #$01
  kObjectPropertyThinFrame                := #$02
  kObjectPropertyCustomSprite             := #$04
  kObjectPropertyOverrideFrameAttributes  := #$08
  kObjectPropertyDisableWeaponCollision   := #$20
  kObjectPropertyThinObject               := #$40
  kObjectPropertyNoReverseOnLinkCollision := #$80
object_iframes              := $04F0  ; [WIP] Invincibility frames indicating how long something is immune to object
                                      ; collision and which palette to use for it.
; Object indices:
; 0 = Link
; 1-B = Dynamic objects
; D = Sword
; E = Sword/Rod Beam
; F = Boomerang/Bait
; 10 = Fire/Bomb 1
; 11 = Fire/Bomb 2
; 12 = Rod/Arrow
; 13 = Treasure
; Short indices:
; 0-C = enemies/items/NPCs
; 5c = boomerang
; also drops - check changed

----

RoomAllDead := $34D
RoomObjCount := $34E
ObjType := $34F
RoomObjTemplateType := $35F
ObjHP := $485
ObjAttr := $4BF
RoomTileObjType := $52B
RoomTileObjX := $52C
RoomTileObjY := $52D
Item_ObjMonsterType := $412
Item_ObjItemLifetime := $3A8
Item_ObjItemId := $AC
LinkX:= $70 link
LinkY := $84  link

97          Dungeon Floor Item
AB          Type of item in screen  The value indicates item type (Bomb, Heart, Rupee, etc)
AD          Type of dropped enemy item #1
AE          Type of dropped enemy item #2
AF          Type of dropped enemy item #3
B0          Type of dropped enemy item #4
B1          Type of dropped enemy item #5
B2          Type of dropped enemy item #6

71          Enemy #1 X-position          Also used for dropped enemy item
72          Enemy #2 X-position          Also used for dropped enemy item
73          Enemy #3 X-position          Also used for dropped enemy item
74          Enemy #4 X-position          Also used for dropped enemy item
75          Enemy #5 X-position          Also used for dropped enemy item
76          Enemy #6 X-position          Also used for dropped enemy item

85          Enemy #1 Y-position          Also used for dropped enemy item
86          Enemy #2 Y-position          Also used for dropped enemy item
87          Enemy #3 Y-position          Also used for dropped enemy item
88          Enemy #4 Y-position          Also used for dropped enemy item
89          Enemy #5 Y-position          Also used for dropped enemy item
8A          Enemy #6 Y-position          Also used for dropped enemy item
```


# Ram map

## Zero Page

```
Address     Function                    Details
----------- --------------------------- -----------------------------------------
00          Multiple Functions          X-coords of sprite, ...
01          Multiple Functions          Y-coords of sprite, ...
02          Multiple Functions          Tile # for sprite (Left Half), ...
03          Multiple Functions          Tile # for sprite (Right Half), ...
04          Multiple Functions          Attributes for sprite (Left Half), ...
05          Multiple Functions          Attributes for sprite (Right Half), ...
06           ?
07          Multiple Functions
08-0C        ?
0D          Multiple Functions          Full Hearts Damage taken, ...
0E          Multiple Functions          Partial Heart Damage taken, ...
0F          Multiple Functions          Link's Move Direction (01,FF), ...
10          Current Level               (0 = Overworld)
11           ?
12          Game Mode                   0=Title/transitory    1=Selection Screen
                                        5=Normal              6=Preparing Scroll
                                        7=Scrolling           4=Finishing Scroll;
                                        E=Registration        F=Elimination
13          Routine Index
14          PPU Loading Index
15          Frame counter
16          Current save slot           $00=first, $01=second, $02=third; On selection/registration/elimination screens: $03=fourth option,  $04=fifth option
17-23       Individual Randomizer
24-27        ?
28          (X to update $00, ...)
29          Timer/Countdown          Used for dialog text and is a Countdown until Enemy #1 can do his action(See Below) again
2A          Countdown until Enemy #2 can do his main action(Shoot a Projectile,Jump,etc.)again
2B          Countdown until Enemy #3 can do his main action(See Above)again
2C          Countdown until Enemy #4 can do his main action(See Above)again
2D          Countdown until Enemy #5 can do his main action(See Above)again
2E          Countdown until Enemy #6 can do his main action(See Above)again
2F           ?
30          Countdown until Rock #1 and Zola's Fireball start to move
31          Countdown until Rock #2 and Zola's Fireball start to move
32          Countdown until Rock #3 and Zola's Fireball start to move and until Zola start his next animation
33          Countdown until Zola start his next animation
34-36        ?
37          Countdown until bait disappears (3 cycle before disappearence)
38          Countdown until bomb explode or flame (Magical Rod or Candle) disappears
39-3B        ?
3C          Countdown until Link and enemies can move again after the recorder was use
3D-4B        ?
4C          Countdown until Link can use his sword again after touching a White Bubble
4D-4F        ?
50          Counts number of enemies killed without taking damage, resets to 0 once it reaches 10 ("10th enemy has the bomb" RAM address)
51-5B        ?
5C          Vertical Scrolling Offset (high byte)
5D-65        ?
66          ROM Offset for start of current Song Data (Low Byte)
67          ROM Offset for start of current Song Data (High Byte)
68-69        ?
6A          Pulse-2 Channel Frequency Offset
6B          Pulse-1 Channel Frequency Offset
6C          Offset related to Music Pointers
6D           ?
6E          Rhythmic Offset             For current note of Pulse 2 Incidental Music
6F          Rhythmic Countdown          For Pulse 2 incidental music (Uses value at $6E)
70          Link's X-position on the screen
71          Enemy #1 X-position          Also used for dropped enemy item
72          Enemy #2 X-position          Also used for dropped enemy item
73          Enemy #3 X-position          Also used for dropped enemy item
74          Enemy #4 X-position          Also used for dropped enemy item
75          Enemy #5 X-position          Also used for dropped enemy item
76          Enemy #6 X-position          Also used for dropped enemy item
77          Enemy #4 Projectile X-position
78          Enemy #1 Projectile X-position
79          Enemy #2 Projectile X-position
7A          Enemy #3 Projectile and Zola(On Some Screen) X-position
7B          Zola X position (On Some screen )
7C           ?
7D          Link's Sword X-position (As a Melee Weapon)
7E          Link's Sword X-position (As a Projectile)
7F          Boomerang/Bait X-position
80          Flame/Bomb #1 X-position
81          Flame/Bomb #2 X-position
82          Link's Arrow X-position
83           ?
84          Link's Y-position on the screen
85          Enemy #1 Y-position          Also used for dropped enemy item
86          Enemy #2 Y-position          Also used for dropped enemy item
87          Enemy #3 Y-position          Also used for dropped enemy item
88          Enemy #4 Y-position          Also used for dropped enemy item
89          Enemy #5 Y-position          Also used for dropped enemy item
8A          Enemy #6 Y-position          Also used for dropped enemy item
8B          Enemy #4 Projectile Y-position
8C          Enemy #1 Projectile Y-position
8D          Enemy #2 Projectile Y-position
8E          Enemy #3 Projectile and Zola (On Some Screen) Y-position
8F          Zola Y-position (On Some Screen)
90           ?
91          Link's Sword Y-position  (As a Melee Weapon)
92          Link's Sword Y-position  (As a Projectile)
93          Boomerang/Bait Y-position
94          Flame/Bomb #1 Y-position
95          Flame/Bomb #2 Y-position
96          Link's Arrow Y-position
97          Dungeon Floor Item
98          Link's Direction             $08=North, $04=South, $01=East, $02=West
99          Enemy #1 Direction (See Above)
9A          Enemy #2 Direction (See Above)
9B          Enemy #3 Direction (See Above)
9C          Enemy #4 Direction (See Above)
9D          Enemy #5 Direction (See Above)
9E          Enemy #6 Direction (See Above)
9F          Enemy #4 Projectile Direction(See Below)
A0          Enemy #1 Projectile Direction             $08=North, $04=South, $01=East, $02=West(Diagonal direction are equal to $(Xdirection+Ydirection))
A1          Enemy #2 Projectile Direction (See Above)
A2          Enemy #3 Projectile Direction (See Above)
A3-A4        ?
A5          Link's Sword Direction (As a Melee weapon)          $08=North, $04=South, $01=East, $02=West
A6          Link's Sword Direction (As a Projectile)
A7          Boomerang Direction
A8          Flame (Candle or Magic Rod) Direction
A9          Second Flame Direction
AA          Arrow and Magic Rod's projectile Direction $08=North, $04=South, $01=East, $02=West
AB          Type of item in screen  The value indicates item type (Bomb, Heart, Rupee, etc)
AC          Link's Animation
AD          Type of dropped enemy item #1
AE          Type of dropped enemy item #2
AF          Type of dropped enemy item #3
B0          Type of dropped enemy item #4
B1          Type of dropped enemy item #5
B2          Type of dropped enemy item #6
B3          State of Enemy #4 Projectile (See Below)
B4          State of Enemy #1 Projectile         $00=Not Existant, $10=In Movement, $20=Start of Blowing Animation, $28=End of Blowing Animation, $30=Being Deflected by Shield
B5          State of Enemy #2 Projectile (See Above)
B6          State of Enemy #3 Projectile (See Above)
B7          State of Enemy #? Projectile (See Above)
B8           ?
B9          Link's Sword Animation
BA          State of Link's Sword(As a Projectile) and Magic             $00=Not Existant, $10=In Movement, $11=Blowing Animation, $80=Magic Onscreen
BB          State of Bait and Link's Boomerang            $10-$17=Rotation of boomerang when going away, $24=Turnaround of Boomerang, $28=Blowing Animation of Boomerang, $50-$57=Rotation of boomerang when coming back, $80=Bait Onscreen(First cycle of 0x0037), $81=Bait Onscreen(Second cycle of 0x0037), $82 Bait Onscreen(Last cycle of 0x0037)
BC          State of Link's Bomb and Flame #1          $12=Bomb Onscreen, $13=Bomb's Smoke Onscreen, $14=Bomb's Smoke Dissappearing, $21=Flame #1 is Moving, $22=Flame #1 is static
BD          State of Link's Flame #2            $21=Flame #2 is Moving, $22=Flame #2 is static
BE          State of Link's Arrow and Magical rod animation          $00=Not Existant, $10=In Movement, $20=Start of Blowing Animation, $28=End of Blowing Animation, $31-$35=Magical Rod Animation
BF-C1        ?
C2          Dropped Enemy Item
C3          Dropped Enemy Item
C4-DF        ?
E0          Game Paused?                $00=No, $01=Yes
E1          Item Menu Scrolling Animation
E2-E7        ?
E8          Screen Scrolling?           $00=No, $08=Northbound, $04=Southbound, $01=Eastbound, $02=Westbound
E9-EA        ?
EB          Map location                Value equals map x location + 0x10 * map y location
EC          Next Location
ED-F7        ?
F8          Player 1 Buttons            (Last Frame)   R = 1, L = 2, D = 4, U = 8, Start = 10, Select = 20, B = 40, A = 80
F9          Player 2 Buttons            (Last Frame)   (See above)
FA          Player 1 Buttons            Pressed    $08=Up, $04=Down, $01=Right, $02=Left $80=A, $40=B, $20=Select, $10=Start
FB          Player 2 Buttons            Pressed    (See Above)
FC          Subscreen Y-scroll position Used for storyboard text and subscreen position
FD          Subscreen X-scroll position This is always zero when not scrolling.
FE          The PPU mask value is set to the value of this address (it's always $1E during gameplay but takes on different values in screen changes of the title screen). Could have had a purpose in some stage of development to hide sprites, since it is used by the item menu and pause routine as well as the screen-transition routine (they call a routine that takes the value and AND's it with $FE, which does nothing today, but could have been set to something else back then).
FF           ?

```

# RAM

```
RAM     Function                    Details
------- --------------------------- -----------------------
0394    Link's Sub-tile             Used for 8 pixel grid for all directions
0395    Enemy #1 Sub-tile?
0396    Enemy #2 Sub-tile?
0397    Enemy #3 Sub-tile?
0398    Enemy #4 Sub-tile?
0399    Enemy #5 Sub-tile?
039A    Enemy #6 Sub-tile?
03A8    Link's Subpixel             Used for all directions
03A9    Enemy #1 Subpixel?
03AA    Enemy #2 Subpixel?
03AB    Enemy #3 Subpixel?
03AC    Enemy #4 Subpixel?
03AD    Enemy #5 Subpixel?
03AE    Enemy #6 Subpixel?
03D0    Link's walking Animation Timer
03E4    Link's Walking Animation State
049E    Link's Current Colliding Tile ($26 is empty)
04CD    Screen options              The entry from ROM:0x18680 for the current room     Overworld: stair placement, quest secrets, Link's V placement, monster entry
0513    Candle used                 Whether Link has used the candle on the current screen  $00=No, $01=Yes
0523				     Randomizer
0526    Cave return screen          Overworld room to return to when exiting the underworld
052A				     Enemies Killed (resets after 9)
052E    Sword disable               Red Bubble Sword Disable switch: $01 if Link's use of his sword has been disabled by a red bubble, $00 otherwise.
052F    Maze counter                Used for screens 0x61 (forest maze) and 0x1B (mountain room south of level 5)
05F0                                Related to current frequency being played by Triangle channel
05F1    Triangle Reverb             Reverb on/off status for Triangle on current song
05F4    Rhythm table pointer        for current song
0600    Song Trigger                Song ID#'s are written here to activate; value is copied over to $609 within a frame. $600 is then zeroed-out.
0602    Pulse 2 I.M. Trigger        Incidental Music ID#'s for Pulse 2 are written here; value is copied over to $607 within a frame.
				     	01 = Rupee Taken
				     	02 = Object Appears
				     	04 = Secret Found
				     	08 = Object Taken
				     	10 = Flute Chime
				     	20 = Enemy Killed
				     	40 = Game Over Tune
				     	80 = Link Death
0604    Pulse 1 I.M. Trigger        Incidental Music ID#'s for Pulse 1 are written here; value is copied over to $605 within a frame.
0605    Pulse 1 I.M. Type           $80=?, $40=1 Heart Warning, $20=Set Bomb, $10=Small Heart Pickup, $08=Key Pickup, $04=Magic Cast, $02=Boomerang Stun, $01=Arrow Deflected
0606    Sound Effects
0607    Pulse 2 I.M. Type           $80=Death Spiral, $40=Continue Screen, $20=Enemy Burst, $10=Whistle, $08=Bomb Pickup, $04=Secret Revealed, $02=Key Appears, $01=Rupee Pickup
0609    Songtype currently active   $80=Title, $40=Dungeon, $20=Level 9, $10=Ending, $08=Item, $04=Triforce, $02=Ganon, $01=Overworld
060A    Pulse 2 pos                 Current position within Pulse 2 part (with respect to overall music program)
060B    Pulse 1 pos                 Current position within Pulse 1 part (with respect to overall music program)
060C    Triangle pos                Current position within Triangle part (with respect to overall music program)
060D    Noise pos                   Current position within Noise part (with respect to overall music program)
060E     ?
060F    Pulse 1 cycle               Current cycle duration for notes played on Pulse 1 channel  Uses value from rhythm lookup table
0610    Pulse 2 cycle               Current cycle duration for notes played on Pulse 2 channel  Uses value from rhythm lookup table
0611    Pulse 2 rhythm              Countdown from Pulse 2's current rhythm offset  Uses value from rhythm lookup table
0612    Pulse 2 countdown           volume fade
0613    Pulse 1 rhythem             Countdown from Pulse 1's current rhythm offset  Uses value from rhythm lookup table
0614    Pulse 1 countdown           volume fade
0615    Triangle cycle              Current cycle duration for notes played on Triangle channel     Uses value from rhythm lookup table
0616    Note countdown              Countdown from value in $615 until next note is to be played
0617     ?
0618    Pulse 2 I.M. offset         Offset into Pulse 2 Incidental Music data
0619    Pulse reverb                Reverb on/off status for BOTH Pulse channels on current song    $80=on, $01=off
061A     ?
061B    Pulse 2 rhythm counter      Countup to Pulse 2 rhythm offset
061C    Pulse 1 rhythm counter      Countup to Pulse 1 rhythm offset
061D                                ??? (seems to count upward as $616 counts down)    Value is always [$615] - [$616] + 1.
061E    Triangle repeat counter     Number of repeats remaining if in Triangle repeat cycle
061F    Triangle Song pos           Position within song program of current Triangle repeat coda (return point)
0620    Screen history write index  Contains the index of the next screen history slot to write to (see below)
0621    Screen history 1            One of five recently visited map locations
0622    Screen history 2            One of five recently visited map locations
0623    Screen history 3            One of five recently visited map locations
0624    Screen history 4            One of five recently visited map locations
0625    Screen history 5            One of five recently visited map locations
0626     ?
0627    Killed enemy count          Number of killed enemies in current screen
0628    Unused
0629    Unused
062A    Unused
062B    Unused
062C    Unused
062D    Current quest               Save slot 1: $00=First, $01=Second
062E    Current quest               Save slot 2: $00=First, $01=Second
062F    Current quest               Save slot 3: $00=First, $01=Second
0630    Number of deaths            Save slot 1
0631    Number of deaths            Save slot 2
0632    Number of deaths            Save slot 3
0633    Slot 1 Present              Save Slot 1 present (Cursor will bypass Slot 1 if set to $00)
0634    Slot 2 Present              Save Slot 2 present (Cursor will bypass Slot 2 if set to $00)
0635    Slot 3 Present              Save Slot 3 present (Cursor will bypass Slot 3 if set to $00)
0636    Register Your Name Present  Register Your Name option present (Cursor will bypass "Register Your Name" if set to $00)
0637    Elimination Mode Present    Elimination Mode option present (Cursor will bypass "Elimination Mode" if set to $00)
0638    Slot 1 Player name          Player Name for Save Slot 1 (8 bytes)
0640    Slot 2 Player name          Player Name for Save Slot 2 (8 bytes)
0648    Slot 3 Player name          Player Name for Save Slot 3 (8 bytes)
0650    Heart Cont. (File Select)   Low Nibble = how many hearts are filled. High Nybble = No. of heart containers - 1
                                    Ex: $22 = 3 Heart Containers with all 3 filled
0651    Fill hearts (File Select)   $FF = Full.
0652    Heart Cont. (File Select)   Low Nibble = how many hearts are filled. High Nybble = No. of heart containers - 1
                                    Ex: $22 = 3 Heart Containers with all 3 filled
0653    Fill hearts (File Select)   $FF = Full.
0654    Heart Cont. (File Select)   Low Nibble = how many hearts are filled. High Nybble = No. of heart containers - 1
                                    Ex: $22 = 3 Heart Containers with all 3 filled
0655    Fill hearts (File Select)   $FF = Full.
0656    Selected item pos           Cursor position for selecting Link's B item
0657    Current sword               $00=None, $01=Sword, $02=White Sword, $03=Magical Sword
0658    Number of Bombs
0659    Arrow status                $00=None, $01=Arrow, $02=Silver Arrow
065A    Bow in Inventory            $00=False, $01=True
065B    Status of candle            $00=None, $01=Blue Candle, $02=Red Candle
065C    Whistle in Inventory        $00=False, $01=True
065D    Food in Inventory           $00=False, $01=True
065E    Potion in Inventory         $00=None/Letter, $01=Life Potion, $02=2nd Potion
065F    Magical Rod in Inventory    $00=False, $01=True
0660    Raft in Inventory           $00=False, $01=True
0661    Magic Book in Inventory     $00=False, $01=True
0662    Ring in Inventory           $00-None, $01-Blue Ring, $02-Red Ring. Note: Changing this value will not change Link's color.
0663    Step Ladder in Inventory    $00=False, $01=True
0664    Magical Key in Inventory    $00=False, $01=True
0665    Power Bracelet in Invenotry $00=False, $01=True
0666    Letter in Inventory         $00=False, $01=True, Link can buy potions from the old woman if $02.
0667    Compass in Inventory        One bit per level
0668    Map in Inventory            One bit per level
0669    Compass in Inventory        (Level 9)
066A    Map in Inventory            (Level 9)
066C    Clock possessed             $00=False, $01=True
066D    Number of Rupees
066E    Number of Keys
066F    Heart Containers            Low Nibble = how many hearts are filled. High Nybble = Number of heart containers - 1
                                    Ex: $10 = 2 Heart Containers with none filled
0670    Partial heart               $00 = empty, $01 to $7F = half full, $80 to $FF = full.
0671    Triforce pieces             One bit per piece
0672                                ??? Related to Ganon's Triforce ???
0673                                ??? Unused ???
0674    Boomerang in Inventory      $00=False, $01=True. Note: 0x0675 overrides this variable.
0675    Magical Boomerang in Inventory
                                    $00=False, $01=True.
0676    Magic Shield in Inventory   $00=False, $01=True.
0677-067B                           ??? Unused ???
067C    Maximum number of bombs     Starts out as $08.
067D    Number of rupees to add
067E    Number of rupees to subtract
067F	                             ?? Screen State (Bit 7 = Secret Discovered)
```

# SRAM

```
RAM     Function                    Details
------- --------------------------- -----------------------
6000				    NameTable Player Name Save #1
600A				    NameTable Player Name Save #2
6010				    NameTable Player Name Save #3
6530				    Tile Mapping Codes for Current Screen (B * 2 * 20 = 2C0 bytes)
67F0				    ..??
6804				    Link's tunic color          $29 = green, $32 = blue, $16 = red
6880				    This is used in Game but also for SaveFunction.
6880				    SaveStatePlayerOne	Name??
6898				    SaveStatePlayerOne 	Item
						PRG $A6C6
							LDY #$27 		TableSize
							LDA $0657,Y     $067E  RAM Location
							STA ($C0),Y     $6997 SRAM Location
							DEY
						PRG $A7B3 			After it will be copied to beginning of SRAM
							LDY #$07
							LDA ($C4),Y    	$6883
							STA ($04),Y     $6005
							DEY
68C0				    ??
6910-6A8F			    End of Player one SaveTable??
6B92    			    Link's tunic color          Overwritten with 0x6804 when Link exits a cave or enters or exits a level
6827	->	19D0F		134E8

Pointer table for Column Definitions
687E -> 8A00	2182		Screen Attributes - Table 0
68FE -> 8A80	2182		Screen Attributes - Table 1
697E -> 8B00	2182		Screen Attributes - Table 2
69FE -> 8B80	2182		Screen Attributes - Table 3
6A7E -> 8C00	2182		Screen Attributes - Table 4
6AFE -> 8C80	2182		Screen Attributes - Table 5
6BA2 -> Enemy Quantities (4 bytes)
6BA6 -> Link's Starting Vertical Position (Various Data)
6BA7 -> Stairs Positions (4 bytes)
6BAF-6BB0 -> 19331	12782

Offset in Cartridge RAM for Screen Status (Overworld Various Data)
6BB1		Level Number
6C90 -> 6500	790		Code
6D80 -> 65F0
6DC7 -> 6637
6E39 -> 66A9
6E6E -> 66DE
6EF9 -> 6769
6F29 -> 6799
6F73 -> 67E3
700F -> 687F
701F -> 688F	790
728E -> 6AFE	790		Code (called at 16A52)
7314 -> 6B84	790
73AA -> 6C1A	790
7613 -> 6E83	790		Table
7A4A -> 72BA	790		Enemy Damage Table
7E26 -> 7696	790
```