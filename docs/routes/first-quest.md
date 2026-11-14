# The rest of the first quest: the draft route after Dungeon 4

Nov 13, 2026 (Lee): route the rest of the game, through the overworld to Dungeons 5 to 9 and through each of them,
and show it on the map for feedback. This is a **draft**: `triforce/draft_routes.yaml` (`first-quest`, which extends
`d2first`), read only by `zelda-map.py`. Nothing is trained on it, `routes.yaml` is unchanged, and no model plays any
of it. Every step was then walked in the emulator the same night (below: what "walked" means and what it does not).

    .venv/bin/python zelda-map.py --route first-quest                     # the window; Route and Labels are on the bars
    QT_QPA_PLATFORM=offscreen .venv/bin/python zelda-map.py --route first-quest --level 9 --overlay labels --screenshot d9.png

Order (Lee): **5, 7, the magic sword, 6, 8, 9.** The overworld legs are Lee's, room for room. The dungeon paths are
the shortest ones through `game.yaml`'s room graph that take what the game needs (the recorder, the red candle, the
silver arrow) and what Lee asked for (the red ring), with the keys Link carries (Lee: use them); they leave out
level 6's wand and level 8's book and magic key.

## What Lee decided (Nov 13)

- The overworld legs (the table below), with screen wraps and the recorder's whirlwind.
- Use the carried keys for the shorter dungeon ways (level 7: 19 room entries and 5 bomb walls instead of 25 and 8).
- Take the red ring in level 9 ("one extra Patra fight, but the result should be worth it; we can always drop it
  later").
- No screen wrap in 0x17: the ladder crosses its river.
- The first route through Dungeons 5-9 is the agent's to find; Lee reviews it on the map and adjusts.

## The legs

Step numbers are `zelda-map.py`'s (the draft continues `d2first`'s numbering). "Wrap exit": `CLIP E` wraps Link
into the room's solid W edge and Left changes the room (`game.yaml` `wrap: true`, as 0x75 into 0x74 since round 22).

| Leg | Rooms | What Link does |
|---|---|---|
| D4 to D5 | 45, 55, 54, 53, 52, 51, 50, 60, 5f, 4f, 3f, 2f, 3f, 3e, 3d, 2d, 1d, 1e, 1f, 0f, 1f, 1e, 1d, 0d, 0c, 0b | the raft back to 0x55; a wrap inside 0x55 over its river; wrap exits out of 0x51, 0x60 (round the world into 0x5f), 0x3e and 0x0c; the heart containers of 0x5f (ladder clip) and 0x2f (raft, take-any cave); 0x0f's 100 rupees through 0x1f's wall gap |
| Dungeon 5 | 76, 66, 65, 64, tunnel 07, 06, 05, cellar 04, 05, 06, tunnel, 64, 65, 66, 56, 57, 47, 37, 27, 26, 25, 24, 14 | two bomb walls, the recorder, Digdogger, heart container, triforce |
| D5 to D7 | 0b, (whirlwind) 45, 55, 54, 44, 34, 44, 54, 53, 52, 42 | the recorder twice: the whirlwind to Dungeon 4's island; arrows (80) in 0x44's shop; food (60) in the shop under 0x34's armos; the recorder drains 0x42's pond |
| Dungeon 7 | 79, 69, 59, 49, 39, 38, 28, 18, 19, 1a, cellar 4a, 1a, 1b, 1c, 0c, 0d, tunnel 7b, 29, 2a, 2b | five bomb walls, three key doors, the hungry Goriya, the red candle, three small Digdoggers, Wallmasters, Aquamentus |
| D7 to the magic sword to D6 | 42, 52, 51, 50, 40, 41, 31, 21, 31, 32, 22 | the wrap exit out of 0x51 again; the gravestone of 0x21: the magic sword (12 heart containers, exactly what Link has) |
| Dungeon 6 | 79, 78, 68, 58, 48, 38, 28, 29, 39, 3a, tunnel 08, 1d, 2d, 2c, 1c, 0c | two Wizzrobe rooms to clear, one bomb wall, Gohma (one arrow) |
| D6 to D8 | 22, (whirlwind) 3c, 4c, 4d, 5d, 6d | the recorder twice: Dungeon 2's entrance; the red candle burns 0x6d's tree |
| Dungeon 8 | 7e, 6e, 5e, 4e, 3e, 3f, tunnel 2f, 4c, 3c, 2c | two rooms of Blue Darknuts to clear, two bomb walls, Gleeok with four heads |
| D8 to D9 | 6d, (whirlwind) 37, 38, 48, 47, 48, 38, 28, 27, 17, 07, 06, 05 | the recorder once: Dungeon 1's entrance; 0x47's burned tree: the last heart container; the ladder over 0x17's river; a bomb on 0x05's rock wall |
| Dungeon 9 | 76, 66, 65, 55, tunnel 60, 14, 15, 16, [26, 27, 17, 07, cellar 00, 07, 17, 27, 26, 16], 06, 05, tunnel 70, 63, 62, 61, tunnel 75, 20, 10, cellar 4f, 10, 20, tunnel 75, 61, 51, 41, 31, 30, tunnel 67, 04, 03, tunnel 77, 52, 42, 32 | the red ring detour in brackets; eight bomb walls, three key doors, six tunnel passages, the silver arrow, Ganon, Zelda |

Heart containers: 8 after Dungeon 4, 9 after 0x5f, 10 after 0x2f, 11 after Dungeon 5, 12 after Dungeon 7 (the
magic sword's cave refuses 11 and gives at 12), 13 after Dungeon 6, 14 after Dungeon 8, 15 after 0x47.

## What was walked, and how

Six agents walked the draft in the emulator on Nov 13, one per dungeon and one for the overworld, each in one
continuous game per walk (`triforce/probe.py` puts Link at a level's stairs; the inventory the route has there was
written once).

- **By the game's own rules:** every room change, key door (keys counted), bomb wall (a bomb placed with B), push
  block, stairs, cellar and tunnel, the ladder (laid by walking), every item pickup, the raft, every wrap exit (by
  button presses, on every row), the whirlwind, the shops (bought with rupees), the armos, the gravestone, the
  trees, the pond, 0x05's wall, the level ends, and the end of the game (mode `$13` after Zelda).
- **Fights were not played.** Where a room needs its monsters dead they were put into the game's own death state
  (the kill counts, shutters, room items and push blocks then answer as after real kills). Played for real: the
  small Digdogger (4 white sword hits), Gohma in level 6 (one arrow) and in level 8's side trip (3 arrows), level
  8's Gleeok (9 magic sword hits), Ganon (4 magic sword hits and the silver arrow, with Link placed beside him).
  Hearts were written back in most walks, so the walks say nothing about survival.
- **Link was moved by RAM inside rooms** in most walks (to a door's square, beside a block or an item); every way
  a route step claims inside a room (round a water ring, through blocks, over water with the ladder) was searched
  on the room's tile map and walked by buttons.
- The B item was chosen by RAM, not on the sub-screen (level 7's walk used the sub-screen).

Result: **every step of the draft works in the order written.** The walks changed notes (what must die, where a
key or bomb goes, walls already open on a way back), not rooms, with two exceptions: 0x55 from the dock to its W
edge needs a screen wrap over the river (or the plain way S, 0x65, 0x64, 0x54), and level 9 needs a bomb pickup
(the Patra of 0x16). `game.yaml` matched the game in every room entered: door kinds (also checked for all 236
dungeon rooms by `tests/test_draft_routes.py`), monsters, room items, dark flags. Added from the walks: four wrap
exits, the push blocks of the rooms whose stairs are closed in by a ring of blocks, level 9's one-way shutters.
The level documents (`zelda-asm/levels/level-5.md` to `level-9.md`) carry the room-by-room traces.

## Forced fights

Rooms that do not let Link on until their monsters are dead (a shutter, a push block that only moves in a cleared
room, a boss). Everything else on the route is crossed.

| Dungeon | Forced | Crossed without a fight |
|---|---|---|
| 5 | 0x05 (6 Blue Darknuts), 0x66 on the way back (3 Gibdos), the small Digdogger; without clips also 0x64 (5 Blue Darknuts; with clips a block clip reaches its stairs) | Gibdos, Zols, Dodongos, Red Darknuts, Pols Voices (0x25 is the costliest room to cross) |
| 7 | 0x49 (3 Blue Goriyas, 3 Keese), 0x1c (3 small Digdoggers), 0x0d (5 Wallmasters), Aquamentus; without clips also 0x1a (6 Goriyas; a block clip reaches the candle's stairs) | seven Goriya rooms, the big Digdogger, Dodongos |
| 6 | 0x58 (8 Keese), 0x38 and 0x3a (each 3 Like-Likes, 2 Blue and 2 Red Wizzrobes), 0x39 (5 Vires), Gohma; for a key level 9 needs: 0x2d (3 Vires) | four Wizzrobe rooms (a Red Wizzrobe's shot is 4 hearts), six blade traps |
| 8 | 0x5e (5 Blue Darknuts), 0x3e (6 Blue Darknuts), Gleeok with four heads | a Manhandla, 0x3f (one-way: Darknuts and Gibdos), 8 Pols Voices |
| 9 | 0x55 (2 Red Lanmolas), 0x07, 0x05, 0x20, 0x10 (5 Wizzrobes each), 0x30 (4 Wizzrobes among traps), 0x03 (2 Zols, 2 Like-Likes), three Patras (0x16 for its bombs, 0x61, 0x52), Ganon | 0x65, 0x17, 0x04 (Wizzrobes), Like-Like rooms, traps |

## The ledgers

**Keys** (6 leave Dungeon 4, `d4.md`): Dungeon 5 spends 3 and picks up 2 on the way (5 out); Dungeon 7 spends 3
(2 out); Dungeon 6 spends 2 and takes 0x58's and 0x2d's (2 out); Dungeon 8 takes 0x5e's and 0x4c's and spends 1 (3
out); Dungeon 9 has three key doors before its only key (1 out). So level 9 is entered with exactly the 3 it needs,
because of two pickups made for it: 6:0x2d's (a fight with 3 Vires) and 8:0x4c's (in the open). The magic key
instead costs level 8 eight more room entries, a Gohma (3 arrows) and a bomb.

**Bombs** (the bag holds 8). Walls: Dungeon 5 two, 7 five, 6 one, 8 two, 0x05's rock wall one, 9 eight: 19. Room
items on the path: 5:0x65 (+4, in the way), 9:0x16 (+4, its Patra must die). **Without drops this does not add
up:** leaving Dungeon 5 with a full bag, Link has 3 after Dungeon 7, 2 after 6, 0 after 8, none for 0x05, and
level 9 needs 4 on entry. And Dungeon 5 itself needs one bomb brought in, where today's chains leave Dungeon 4
with none. What can fill the gap, none of it in the draft yet:

| Source | Gives | Costs |
|---|---|---|
| 7:0x1b's room bombs | +4 (up to the bag's 8) | a fight with 6 Blue Goriyas |
| 8:0x3f's room bombs | +4 | a fight with 3 Darknuts and 3 Gibdos in a room with no way back |
| 0x6f's shop (two rooms E of Dungeon 8's entrance), or 0x44's (already visited for the arrows) | +4 per 20 rupees | rooms and rupees (see below) |
| drops | bombs from Gibdos, Red Goriyas, Vires, Red Wizzrobes, Red Darknuts and Blue Moblins (the drop table's bomb row); the forced drop of a tenth damage-free kill made with a bomb | not counted here |

One plan that adds up with no drop at all (Link leaves Dungeon 4 with no bombs), written into the draft's notes
as optional: buy 4 at 0x44 on leg 1 (one room N of 0x54 and back; 20 rupees, which then needs 0x3d's 30 for the
arrows and the food); Dungeon 5: 4, two walls, +4 = 6; Dungeon 7: three walls (3), 0x1b's room bombs (7), two
walls: 5; Dungeon 6: 4; Dungeon 8: 2, and 0x3f's room bombs: 6; 0x05's wall: 5; level 9: four walls (1), the
Patra's bombs (5), four walls: 1 left. Three fights (7:0x1b, 8:0x3f, 9:0x16) and one shop stop.

**Rupees:** 42-54 after `d2first`'s 0x67 cave, +100 at 0x0f, -80 (arrows), -60 (food) leaves **2-14**, and each
arrow shot costs one (one for level 6's Gohma, one for Ganon). Both purchases go through even at the low end; there
is nothing for bombs. On the route: 0x3d's armos hides a cave with 30 rupees (505 frames, in the draft's note as
optional); 0x51's tree 10 (needs the candle: leg 3 only); 0x48's and 0x28's trees 30 each (leg 5).

**Hearts:** no refill is on the route but the triforce pieces (each fills the hearts) and drops. The walks' blind
crossings took about 20 hearts of contact in level 5, 11 in level 7, 12-22 in level 6: the fights and the
Wizzrobe rooms are where a run will be lost. The red ring comes in the middle of level 9.

## What the game does that the draft relies on (from the walks)

- **Wrap exits:** 0x51 into 0x50 (rows Y `$75`-`$9D`), 0x3e into 0x3d (`$85`-`$BD`), 0x0c into 0x0b
  (`$65`-`$AD`): no softlock row. 0x60 into 0x5f works on `$75`-`$9D` and leaves Link **on water**: only rows
  `$85`/`$8D` lead on, with the ladder clip (38 frames from the arrival to the heart container); without the ladder
  it is a softlock.
- **The whirlwind:** a second play of the recorder before the whirlwind reaches Link steps the destination again,
  so two plays facing down (as Link comes out of a dungeon) and one ride take him from 0x0b to 0x45 and from 0x22
  to 0x3c (625 frames each); one play from 0x6d to 0x37. The destination depends on every earlier overworld play:
  an extra one anywhere changes the later counts.
- **Tunnels:** Link always comes up at a fixed spot of the room above, outside its ring of blocks except in 9:0x14
  and 9:0x10 (inside blocks that lead out). The stairs of a ring room open by pushing the ring's W block N or S,
  only in a cleared room; pushed E it closes them until the room is entered again. A block clip gets into the ring
  with the monsters alive (5:0x64, 7:0x1a, 8:0x1f).
- **Banked kills:** a room's kills are kept while it is among the last six rooms added to the room history, so a
  room can be fought in parts through the door Link came in by (level 8's Darknut rooms), and a room cleared on the
  way in is still clear on the way back if the detour is short (9:0x20), but not after a long one (5:0x66).
- **One-way rooms:** 8:0x3f (its shutter shuts behind Link; the stairs are the way on) and level 9 north of 0x41
  (the S shutters of 0x41, 0x31, 0x21 never open): Link must hold 2 bombs when he leaves 9:0x61 north.
- **Items that ride on monsters** (5:0x66's key, 5:0x65's bombs): taken by touching the carrier (2 hearts) or by
  killing it and walking over the spot.
- **The end:** Ganon's room opens when the Triforce of Power is taken, not when he dies; walking up the middle of
  Zelda's room ends the game on the next frame.

## What is not built

Nothing of this route can be played by the agent yet. What it needs beyond rooms, states and families:

- **Actions:** the recorder (three whirlwinds, 0x42's pond, Digdogger twice), food (7:0x28), the candle (two trees);
  ARROW exists but no state has held arrows. Each new action kind makes every trained model unusable: decide them
  together (the version rule's COMPAT bump).
- **The overworld ladder clip** (0x5f's heart container from the wrap arrival): `zelda::clips` offers ladder clips
  in dungeons only. Without it 0x5f's heart is reached the plain way from 0x4f or 0x6f.
- **Clips off:** Lee's rule is that the plain way stays trained. The wrap exits have plain ways of very different
  cost: 0x3e (N, 0x2e, 0x2d: the same room count), 0x0c (the Lost Hills, 1,962 frames against 726), 0x55 (S, 0x65,
  0x64, 0x54), but 0x51 into 0x50 is 23 rooms round the north coast and 0x60 into 0x5f 16 rooms along the south.
- **Route steps the objective does not know:** buying in a shop, the take-any caves' right ware, the armos and the
  gravestone, burning a tree, the whirlwind as an exit, a tunnel (a cellar with two exits), a room left by the door
  it was entered by, the end of the game.
- **Families** (by enemy, not assigned yet): Gibdo, Pols Voice, Digdogger, Blue Darknut (twice the Red one's
  health and damage), Blue Goriya, Wallmasters with a push block, Wizzrobes, Gohma, Lanmola, Patra, Gleeok with
  four heads, Ganon; overworld rooms with Lynels, Peahats, Moblins, Leevers and Ghinis.

## Open questions for Lee

1. **Bombs** (the ledger above): which of the sources goes into the route? The cheapest in rooms is to clear
   7:0x1b and 8:0x3f; the cheapest in fights is a shop, which needs rupees the route does not have.
2. **Rupees:** take 0x3d's 30 in leg 1 (and 0x48's or 0x28's 30 in leg 5)?
3. **Level 9's keys:** two pickups (6:0x2d's Vires, 8:0x4c in the open) or level 8's magic key?
4. **Level 9's bombs:** the Patra of 0x16 (the draft), or the four bombs lying in 0x23 (from 0x14 by 0x24: no
   fight, four more rooms)?
5. **0x60 into 0x5f** needs the overworld ladder clip built (Link arrives on water), and the only other way to
   0x5f's heart is to enter it from 0x4f or 0x6f: 16 rooms along the south coast from 0x60. Build the clip?
6. **Clips off** on legs 1 and 3 (0x51's 23-room plain way): is a clips-off run allowed to be that much longer, or
   do those legs want another route without wraps?
7. **No ring and no magic shield before level 9.** Level 8's eleven Blue Darknuts (2 hearts a touch) and level 6's
   Red Wizzrobes (4 hearts a shot) are fought at full damage. The blue ring is 250 rupees.
8. The wand, the book and the magic key are left out; say if any should come back.

## Tools made for this

- `scripts/level_rooms.py`: a level's `game.yaml` rooms from the ROM's level data (`zelda-asm/tools/romdata.py`).
  Levels 5-9 were written by it; what a person adds is in its docstring.
- `triforce/probe.py`: Link into any dungeon room without a savestate (how `zelda-map.py` pictures levels 5-9, and
  how the walks started).
- `triforce/draft_routes.py`: the draft's loader and its check against `game.yaml` (`problems`).
- `zelda-map.py`: draft routes, door marks on every dungeon level, the Labels overlay with the route's notes.
