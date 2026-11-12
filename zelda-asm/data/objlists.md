# Monster lists

How to read this: a room's level block bytes give a 7-bit monster list id and a count index. This file
lists every list reachable through [`Spawn_ObjListAddrs`](../src/bank5/mode-inits-obj-lists.asm#L185) (blob [`Spawn_ObjLists`](../src/bank5/mode-inits-obj-lists.asm#L180), `dat/ObjLists.dat`) and the object type names used in all listings.

- Monster list id = level block C bits 5-0, plus `$40` when D bit 7 is set. [`Room_EnterCommon`](../src/bank5/enter-room.asm#L137)
- Count = entry (C bits 7-6) of the level info's four monster counts ([`LevelInfo_FoeCounts`](../src/Variables.inc#L715)); `levelinfo.md` has them.
- Id `0`: no monsters. Id `$01`-`$61`: `count` objects of the object type equal to the id; for an id from
  `$32` up the count is forced to 1 (bosses and other single objects). [`Room_EnterCommon`](../src/bank5/enter-room.asm#L137)
- Id `$62`-`$7F`: list number id - `$62` below; the first `count` bytes of the list are the object types of
  object slots `$01` up. A list has no end marker: a count above its stored length runs into the next
  list. [`Room_EnterCommon`](../src/bank5/enter-room.asm#L137)
- Before the objects are made, the count is lowered by the room's kill count, and a cellar or cave
  replaces the list. [`RoomHistory_ModifyObjCountOW`](../src/bank5/room-history-ow.asm#L93), [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50), [`Spawn_AssignPositions`](../src/bank5/spawn-positions.asm#L64)

# Lists

`offset` is the list's offset in the blob; `len` is the distance to the next list. [`Spawn_ObjLists`](../src/bank5/mode-inits-obj-lists.asm#L180)

```
id   offset  len  type bytes                       types
$62  0       5    $03 $03 $04 $03 $04              $03 Moblin, $03 Moblin, $04 Moblin, $03 Moblin, $04 Moblin
$63  5       4    $03 $04 $03 $04                  $03 Moblin, $04 Moblin, $03 Moblin, $04 Moblin
$64  9       6    $1A $1A $02 $01 $02 $01          $1A Peahat, $1A Peahat, $02 Lynel, $01 Lynel, $02 Lynel, $01 Lynel
$65  15      4    $01 $02 $01 $02                  $01 Lynel, $02 Lynel, $01 Lynel, $02 Lynel
$66  19      6    $01 $0F $02 $01 $10 $02          $01 Lynel, $0F Blue Leever, $02 Lynel, $01 Lynel, $10 Red Leever, $02 Lynel
$67  25      6    $0F $1A $10 $1A $0F $1A          $0F Blue Leever, $1A Peahat, $10 Red Leever, $1A Peahat, $0F Blue Leever, $1A Peahat
$68  31      5    $09 $08 $08 $08 $08              $09 blue Octorok slow, $08 Octorok fast, $08 Octorok fast, $08 Octorok fast, $08 Octorok fast
$69  36      4    $08 $07 $08 $07                  $08 Octorok fast, $07 Octorok slow, $08 Octorok fast, $07 Octorok slow
$6A  40      5    $08 $09 $08 $09 $08              $08 Octorok fast, $09 blue Octorok slow, $08 Octorok fast, $09 blue Octorok slow, $08 Octorok fast
$6B  45      5    $0A $07 $0A $07 $07              $0A blue Octorok fast, $07 Octorok slow, $0A blue Octorok fast, $07 Octorok slow, $07 Octorok slow
$6C  50      6    $03 $0A $04 $0A $04 $04          $03 Moblin, $0A blue Octorok fast, $04 Moblin, $0A blue Octorok fast, $04 Moblin, $04 Moblin
$6D  56      8    $4A $00 $00 $00 $13 $13 $00 $13  $4A Trap, $00 (empty), $00 (empty), $00 (empty), $13 Zol, $13 Zol, $00 (empty), $13 Zol
$6E  64      8    $4A $00 $00 $00 $1B $1B $1B $1B  $4A Trap, $00 (empty), $00 (empty), $00 (empty), $1B Keese blue, $1B Keese blue, $1B Keese blue, $1B Keese blue
$6F  72      8    $2B $2B $2B $13 $13 $1B $1B $1B  $2B Bubble, $2B Bubble, $2B Bubble, $13 Zol, $13 Zol, $1B Keese blue, $1B Keese blue, $1B Keese blue
$70  80      8    $16 $30 $30 $1B $1B $16 $00 $00  $16 Pols Voice, $30 Gibdo, $30 Gibdo, $1B Keese blue, $1B Keese blue, $16 Pols Voice, $00 (empty), $00 (empty)
$71  88      8    $2B $2B $2B $23 $23 $24 $23 $24  $2B Bubble, $2B Bubble, $2B Bubble, $23 Blue Wizzrobe, $23 Blue Wizzrobe, $24 Red Wizzrobe, $23 Blue Wizzrobe, $24 Red Wizzrobe
$72  96      8    $2B $2B $12 $12 $12 $00 $00 $00  $2B Bubble, $2B Bubble, $12 Vire, $12 Vire, $12 Vire, $00 (empty), $00 (empty), $00 (empty)
$73  104     6    $2B $2B $13 $13 $17 $17          $2B Bubble, $2B Bubble, $13 Zol, $13 Zol, $17 Like-Like, $17 Like-Like
$74  110     8    $2B $2B $0C $0B $0B $30 $30 $30  $2B Bubble, $2B Bubble, $0C Darknut blue, $0B Darknut red, $0B Darknut red, $30 Gibdo, $30 Gibdo, $30 Gibdo
$75  118     8    $2B $2B $05 $05 $05 $1B $1B $1B  $2B Bubble, $2B Bubble, $05 Goriya, $05 Goriya, $05 Goriya, $1B Keese blue, $1B Keese blue, $1B Keese blue
$76  126     8    $4A $00 $00 $00 $17 $17 $17 $17  $4A Trap, $00 (empty), $00 (empty), $00 (empty), $17 Like-Like, $17 Like-Like, $17 Like-Like, $17 Like-Like
$77  134     8    $4A $00 $00 $00 $23 $24 $23 $24  $4A Trap, $00 (empty), $00 (empty), $00 (empty), $23 Blue Wizzrobe, $24 Red Wizzrobe, $23 Blue Wizzrobe, $24 Red Wizzrobe
$78  142     6    $16 $0C $0B $0C $0B $16          $16 Pols Voice, $0C Darknut blue, $0B Darknut red, $0C Darknut blue, $0B Darknut red, $16 Pols Voice
$79  148     8    $2B $2B $2B $27 $27 $27 $27 $27  $2B Bubble, $2B Bubble, $2B Bubble, $27 Wallmaster, $27 Wallmaster, $27 Wallmaster, $27 Wallmaster, $27 Wallmaster
$7A  156     8    $05 $06 $06 $05 $06 $05 $00 $00  $05 Goriya, $06 Goriya, $06 Goriya, $05 Goriya, $06 Goriya, $05 Goriya, $00 (empty), $00 (empty)
$7B  164     5    $23 $23 $24 $23 $24              $23 Blue Wizzrobe, $23 Blue Wizzrobe, $24 Red Wizzrobe, $23 Blue Wizzrobe, $24 Red Wizzrobe
$7C  169     8    $2B $17 $23 $23 $17 $24 $17 $24  $2B Bubble, $17 Like-Like, $23 Blue Wizzrobe, $23 Blue Wizzrobe, $17 Like-Like, $24 Red Wizzrobe, $17 Like-Like, $24 Red Wizzrobe
$7D  177     8    $2D $2D $2D $2C $23 $24 $23 $24  $2D Bubble, $2D Bubble, $2D Bubble, $2C Bubble, $23 Blue Wizzrobe, $24 Red Wizzrobe, $23 Blue Wizzrobe, $24 Red Wizzrobe
$7E  185     8    $2D $2D $2D $2C $0C $0B $0C $0B  $2D Bubble, $2D Bubble, $2D Bubble, $2C Bubble, $0C Darknut blue, $0B Darknut red, $0C Darknut blue, $0B Darknut red
$7F  193     8    $2D $2D $2D $2C $27 $27 $27 $27  $2D Bubble, $2D Bubble, $2D Bubble, $2C Bubble, $27 Wallmaster, $27 Wallmaster, $27 Wallmaster, $27 Wallmaster
```

# Object type names

Names used for object type ids in these listings (types that share a name differ by id). Type `$6A`
and above are cave persons. [`Obj_Init`](../src/bank7/object-init-table.asm#L45)

```
type  name
$01   Lynel
$02   Lynel
$03   Moblin
$04   Moblin
$05   Goriya
$06   Goriya
$07   Octorok slow
$08   Octorok fast
$09   blue Octorok slow
$0A   blue Octorok fast
$0B   Darknut red
$0C   Darknut blue
$0D   Tektite
$0E   Tektite
$0F   Blue Leever
$10   Red Leever
$11   Zora
$12   Vire
$13   Zol
$14   child Gel
$15   Gel
$16   Pols Voice
$17   Like-Like
$18   small Digdogger
$1A   Peahat
$1B   Keese blue
$1C   Keese red
$1D   Keese black
$1E   Armos
$1F   boulder spawner
$20   Boulder
$21   Ghini
$22   flying Ghini
$23   Blue Wizzrobe
$24   Red Wizzrobe
$25   Patra child
$26   Patra child
$27   Wallmaster
$28   Rope
$2A   Stalfos
$2B   Bubble
$2C   Bubble
$2D   Bubble
$2E   Whirlwind
$2F   Pond fairy
$30   Gibdo
$31   Dodongo
$32   Dodongo
$33   Gohma
$34   Gohma
$35   Rupee stash
$36   Grumble
$37   Zelda
$38   Digdogger
$39   Digdogger
$3A   Lanmola
$3B   Lanmola
$3C   Manhandla
$3D   Aquamentus
$3E   Ganon
$3F   Guard fire
$40   Standing fire
$41   Moldorm
$42   Gleeok
$43   Gleeok
$44   Gleeok
$45   Gleeok
$46   Gleeok head
$47   Patra
$48   Patra
$49   Trap
$4A   Trap
$4B   dungeon person
$4C   dungeon person
$4D   dungeon person
$4E   dungeon person
$4F   dungeon person
$50   dungeon person
$51   life-or-money person
$52   dungeon person
$53   flying rock
$54   unknown shot
$55   fireball
$56   fireball
$57   sword shot
$58   magic shot
$59   magic shot
$5A   shot
$5B   monster arrow
$5C   monster boomerang
$5D   dead dummy
$5E   flute secret
$5F   ladder
$60   dropped item
$61   dock
$62   rock
$63   rock wall
$64   tree
$65   gravestone
$66   armos tile object
$67   rock wall
$68   push block
```
