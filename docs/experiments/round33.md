# Round 33: the families phase IX left owed

## Wizzrobe and vire: Dungeon 6's 0x78, 0x29, 0x39 and 0x2d

**State of this section.** The cause and the arm; the runs and the choice follow.

wizzrobe plays Dungeon 6's 0x78, 0x38, 0x28, 0x29, 0x3a, 0x2c and level 9's 0x15, 0x17, 0x07, 0x05, 0x20, 0x10, 0x31,
0x30, 0x04 (0x65 has left the route); vire plays Dungeon 4's eight rooms and Dungeon 6's 0x39 and 0x2d. The
route's runs are `r30w-wj-s89` (round 30, a joint run for Dungeons 6 and 9) and `r77-b` (round 27). Phase IX
(round31.md) changed three of their rooms: 6:0x78 is a ringleader room with `drops: first` (step 262), 0x29's key
is taken (step 268), 0x39 and 0x2d are `bombs: tenth` (steps 269, 273).

### The cause

From the chains of record (`r31land-1k-d6`, 1000 runs from `6_79s_0`, seeds 0-999; the old route's
`r29land-1k-d6` beside), and two chains of the landed team with handoffs and replays: `h6` from 0:0x42 after
Dungeon 7 (600 runs, seeds 5000-5599: the route's hearts, Dungeon 6 entered with 10.83 of 12; its triforce 23.2%)
and `h6b` from `6_79s_0` (400 runs, seeds 6000-6399; 33.8%).

| room (step) | new route: entered, hearts in, lost a visit, deaths | old route | below 4 hearts in: deaths, new / old |
|---|---|---|---|
| 0x78 (262) | 1000, 12.00, 0.01, 0 (0 kills, 304 frames) | 1000, 0.00, 0 | |
| 0x38 (266) | 1000, 10.81, 3.94, 89 | 1000, 4.47, 137 | |
| 0x29 (268, the key) | 901, 6.58, **1.52**, **74** | 839, 0.54, 27 | 66 of 188 / 27 of 194 |
| 0x39 (269, vire) | 829, 5.47, −0.13, 36 | 814, −0.17, 26 | 36 of 262 / 26 of 217 |
| 0x3a (270) | 797, 5.78, 2.31, 354 | 801, 2.76, 377 | |
| 0x2d (273, vire) | 437, 4.43, 0.04, **29** | 406, −0.25, 6 | **29 of 202 / 6 of 183** |
| 0x2c (274) | 402, 4.63, 1.23, 89 | 399, 1.03, 56 | |

- **0x78:** the model walks through: no swing in 1000 visits. `drops: first` marks the drops to take; nothing makes
  them.
- **0x29:** every visit that survives takes the key, and no run leaves by a wrong exit (main's 1.9.0.0 wizzrobe on
  the new route: 73 wrong exits, the 9:0x63 cheated key's cause). What the key costs is hearts: 1.52 a visit
  against 0.54 on the old path, so 47 more deaths, nearly all entering below 4 hearts. At the route's hearts the
  room is cheaper (`h6`: 0.85 a visit, 27 deaths of 477), from the segment's start not (`h6b`: 1.6, 23 of 359). The
  room is the same: the walk's door-row way played by a rule over the env's actions from 150 states cut from each
  chain's replays, at 8 hearts, loses 1.17 and 1.12 hearts, 150 of 150 keys, no death. The states differ in what
  the model sees of the rupees (37-40 against 43-44, the observed `rupees_spare`) and the drop cycle; the model
  plays the same room dearer from the segment's start. Training takes 0x29's states from both chains.
- **0x39:** at equal hearts as before (36 of 262 deaths below 4 hearts against 26 of 217); more runs enter it low
  because 0x29 costs more. Not a vire loss.
- **0x2d:** **the Vires cost runs since they are not bombed**: 29 deaths against 6, all entering below 4 hearts
  (14% of those visits against 3%), 0.04 hearts a visit against a gain of 0.25 (the bombs' drops). About 20 runs of
  1000 on `d6`: vire is owed its round.
- 0x2c (wizzrobe) loses 33 more than with the older wizzrobe (round 30 saw it: 88 against 50 on its confirmation
  chains); 0x3a is still the dungeon's killer.

### The arm
