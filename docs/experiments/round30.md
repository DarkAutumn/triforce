# Round 30: the optimisation round over the whole route

## goriya, wallmaster, boss-gleeok

The families of Dungeon 7's losses and Gleeok's two rooms, by the recipe of decisions 2026-11-19 ("One team for
the whole route"). The team is 1.9.0.0's (`r29land-1k`, round29.md §20): goriya `r77-goriya5-s80`, wallmaster
`r76-wall-1`, boss-gleeok `r78t4-gleeok`.

### The cause

From the room records of the chains of record (`r29land-1k-d7`, `-d1`, `-d2`, `-d4`, `-d8`, `-full`, 1000 runs
each, models alone) and the planner beside (`r29land-plan-full`, 300 runs). Hearts lost a visit are counted on the
visits Link survives.

**Dungeon 7: goriya's own rooms spend the hearts 7:0x1b needs.** `d7` starts at 11 hearts and enters 0x1b with
4.51 (the full route 4.08, the planner 9.22). Its deaths are the runs that arrive low:

| hearts entering 0x1b | below 2 | 2-3.9 | 4-5.9 | 6 or more |
|---|---:|---:|---:|---:|
| `d7` (deaths of visits) | 102 of 116 | 107 of 237 | 46 of 283 | 3 of 274 |
| full route | 30 of 37 | 35 of 59 | 10 of 48 | 2 of 55 |
| the planner (full route) | | 0 of 1 | 0 of 9 | 0 of 259 |

The hearts go in goriya's rooms before it (`d7`, a visit; the planner's in brackets, full route): 0x69 1.13
(0.36), 0x59 1.13 (0.12), 0x49 0.69 (−0.19), 0x38 0.37 (0.04), 0x18 0.62 (0.13), 0x19 1.13 (0.30), 0x1a from W 0.41
(0.04), 0x1a from the cellar 0.92 (0.20): some 6.4 hearts against the planner's 1.0. Goriya deaths in `d7`: 0x1b
258, 0x1a from the cellar 58, 0x19 27, 0x29 19, 0x59 18, 0x1a from W 17, 0x18 13, 0x69 9. Every one of goriya's 90
curriculum tracks is complete (nothing to take out). 7:0x1c, the next loss (73 deaths entered with 5.43 hearts),
is boss-digdogger's room.

**Dungeons 1 and 2 at the full route's hearts.** The segments start at full hearts and hide part of goriya's losses
there: 1:0x23 (step 57) 15 deaths and 1:0x44 25 in the full route (entered with 3.3 hearts) against 0 and 1 in the
`d1` segment (4.3-4.7 hearts); 2:0x5e 29 deaths in the full route (2.73 hearts) against 9 in `d2` (2.94); 2:0x1e 41
and 43. The planner beside loses none of them.

**Wallmaster, 7:0x0d.** In `d7` 574 visits: 491 leave right, **64 are captured** (an `exit-wrong` back to 0x79;
59 of those runs die crossing Dungeon 7 again) and 19 end at the room's 2000-step limit: some 8 points of `d7`.
The planner: 9 captured and 6 timeouts in 269.

**Gleeok.** 4:0x13 (`d4`): 76 deaths of 993, entered with 4.56 hearts; deaths entering with 0, 1, 2, 3 hearts 10 of
10, 24 of 38, 19 of 87, 16 of 154, with 4 or more 7 of 704. The yaml's `hp: [0.5, max]` (the owner, Nov 12) never
starts a fight below half the containers. 8:0x3c (`d8`): 70 deaths of 319, entered with 5.39 of 13.
