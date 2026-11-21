# Round 34: the wood-sword dungeons on `d3first`, segment by segment

## Dungeon 7 with the wood sword, and Dungeon 1's boss (goriya, wallmaster, boss-aquamentus; boss-digdogger held)

Branch `r34-d7` (from main 2.0.0.0, 9ee2f6a5). The team is main's promoted runs (stamped 2.0.0.0); `d7` is
`--from 7_79s_0 --stop-at step:194`, `d1` `--from 1_73s_0 --stop-at step:79`, models alone, clips on, 1000 runs.

### 1. Where the hearts before 7:0x1b go

The record (`r32land-1k-d7`, seeds 0-999, 9.4%) against the planner (`temperature:0.5+relax:16+plan:16:1`) on the
same start and seeds 0-299 (`r34d7-plan-d7`, **87.7%**). First visits; hearts on entering, hearts lost a visit
(negative: picked up), deaths per 1000 runs:

| Step | Room (family) | Models: in / lost / deaths | Planner: in / lost / deaths | Bombs a visit (models / planner) |
|---|---|---|---|---|
| 175 | 0x69 (goriya; crossed, the N wall) | 11.00 / 1.01 / 0 | 11.00 / 0.36 / 0 | 3.03 / 2.79 |
| 176 | 0x59 (goriya) | 9.99 / 1.14 / 0 | 10.64 / 0.58 / 0 | 0.01 / 0.05 |
| 177 | 0x49 (goriya; cleared for its N shutter) | 8.84 / 0.85 / 0 | 10.05 / -0.36 / 0 | 1.97 / 2.16 |
| 178 | 0x39 (boss-digdogger; crossed) | 8.00 / 0.34 / 3 | 10.42 / 0.17 / 0 | 0 / 0 |
| 179 | 0x38 (goriya) | 7.68 / 0.37 / 1 | 10.25 / 0.04 / 0 | 0 / 0 |
| 180 | 0x28 (goriya; the Grumble) | 7.32 / 0.00 / 0 | 10.21 / 0.00 / 0 | 0 / 0 |
| 181 | 0x18 (goriya) | 7.32 / 0.61 / 6 | 10.21 / 0.28 / 0 | 0 / 0 |
| 182 | 0x19 (goriya; the E wall) | 6.75 / 1.04 / 19 | 9.94 / 0.39 / 0 | 0.99 / 1.01 |
| 183-185 | 0x1a, the cellar, 0x1a (goriya; the E wall) | 5.82 / 1.43 / 52 | 9.54 / 0.36 / 0 | 0.98 / 1.02 |
| 186 | **0x1b** (goriya; six Blue Goriyas) | **4.64 / 2.57 / 468** | **9.18 / -0.46 / 0** | 0.03 / 0.07 |

Before 0x1b the models lose 6.4 hearts, 6.0 of them in goriya's rooms; the planner loses 1.8. Both throw the bag
down to the entrance's reserve of 3 in 0x69 and 0x49, so both enter 0x1b with 1 bomb (874 of 920; the planner 281 of
301), and with 1 bomb the route wants 0x1b's four (`if_bombs_below: 5`): a wood-sword fight of six Blue Goriyas (5
hits each). The planner wins it at 9.2 hearts; the models die in it at 4.6. The 31 model runs that entered with 5
bombs crossed to the key door instead: 5 died (1.63 hearts a visit) against 460 of 874 (2.62).

Past 0x1b: Digdogger's 0x1c 80 deaths of 450 (4.21 hearts in); wallmaster's 0x0d grabs Link on 235 of 370 runs
that reach it and sends him to the entrance, and no grabbed run wins (the second pass: 0x18 104 deaths, 0x59 41,
0x69 34); the planner is grabbed in 31 of 300 runs and 28 of them win. Aquamentus 7:0x2a 8 of 102.

**The key leak** (backlog): goriya spends a second key in 0x1b in 15 of 939 visits (1.6%; the planner 7 of 301).

Dungeon 1 (`r32land-1k-d1`, 89.6%; the planner's full route, 176 runs at Level 1): Aquamentus 1:0x35 is entered with
3.32 of 5 hearts (73 of 969 die), the planner's with 4.51 (0 of 176). Of the 1.68 hearts the models lose before it,
goriya's 0x23 takes 0.43 and 0x44 0.31 (22 deaths); wallmaster's 1:0x45 none.

### 2. The route data: the bag kept for 0x1b (`ROUTE DATA` 9295f8d0, the lead's decision)

`d3first` writes its own Level 7 entrance and 0x1b (the block `dungeon-7` split into `dungeon-7` and
`dungeon-7-east`; `d2first` writes its own as before): a reserve of **8** from the entrance to 0x1b (the walls of
0x69, 0x19 and 0x1a are the bag's only use there), and 0x1b wants its bombs only **below 2** (0x0c's and 0x29's
walls). An arrival with 8 enters 0x1b with 5 and crosses; with the threshold at 5 (unchanged) only an arrival with 8
would. Arrivals at Level 7 (the record): the leg from Level 2 (`0_3cc_0` holds 8) brings 8 in 1000 of 1000; the `d2`
segment ends with 8 in 567 of 1000; the planner's full route reaches Level 7 in 127 runs, with 8 bombs in 59, 5-7 in
31 and fewer in 37: the threshold of 2 crosses 70% of them where 8 alone would cross 46%. `d2first`: its step list is
identical (354 steps) and a 100-run `d2first` chain from `7_79s_0` to step 243 is identical run for run before and
after (63/100); test `test_d3first_keeps_the_bag_through_dungeon_7_to_0x1b`.

**The arm** (`r34d7-arm-route2`, today's team, seeds 0-999, paired with the record):

| | Record | Route data | Paired |
|---|---|---|---|
| `d7` | 9.4% | **18.4%** | **+9.0 ± 1.5** |
| 0x1b entering, hearts, deaths, hearts lost a visit | 920, 4.64, 468, 2.57 | 899, 4.52, **133**, 1.54 | |
| 0x69 without weapon bombs: bombs, hearts lost, deaths | 3.03, 1.01, 0 | 1.00, 1.00, 0 | |
| 0x49: bombs, hearts lost, deaths | 1.97, 0.85, 0 | 0.00, 1.10, 2 | |
| 0x1c (Digdogger): entering, hearts in, deaths | 450, 4.21, 80 | 766, 3.49, 162 | |
| 0x0d: runs reaching, grabbed (none of them wins) | 370, 235 | 602, **374** | |
| Cheated in Level 7: bombs, keys | 1, 0 | 1, 0 | |
| 0x1b's E door: exits, keys spent, visits spending 2 | 458, 465, 15 | 770, 766, **0** | |

The leak was the fight's: crossing, no visit spends a second key. Wallmaster's grab is now the segment's first loss
(374 runs), then Digdogger 162 and the second pass through goriya's rooms.

### 3. Trainings

(in progress)
