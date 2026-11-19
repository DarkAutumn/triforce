# Round 30: the optimisation round over the whole route

## Wizzrobe and patra (Dungeon 6 and level 9)

**State of this section.** Begun: the cause (below); the arm is being built.

wizzrobe plays Dungeon 6's 0x78, 0x38, 0x28, 0x29, 0x3a, 0x2c and level 9's 0x65, 0x15, 0x17, 0x07, 0x05, 0x20,
0x10, 0x31, 0x30, 0x04; patra level 9's 0x16, 0x27, 0x61, 0x52. The route's runs are `r79c-wj-s85` (round29.md
§18.1: the one joint run of three seeds that held `d6`, 314 against 316 of 1000, while lifting `d9`) and
`r79c-patra`.

### The cause

From the chains of record (`r29land-1k`, 1000 runs a segment, seeds 0-999) and the planner beside
(`r29land-plan-full`, 300 runs from the game start; the Dungeon 9 lane's `r79c-plan-d9`, 300 from Dungeon 9's
entrance). Hearts lost a visit are counted on the visits Link survives.

**Dungeon 6** (`d6`, from 12 hearts):

| room | models: entered, hearts in, lost a visit, deaths | planner (full route): entered, hearts in, lost, deaths |
|---|---|---|
| 0x38 | 1000, 10.83, 4.47, 137 | 254, 10.70, 2.56, 20 |
| 0x28 | 863, 6.40, 0.29, 24 | 234, 8.29, 1.01, 9 |
| 0x29 | 839, 6.23, 0.54, 27 | 237, 7.39, 1.02, 25 |
| 0x3a | 801, 6.16, 2.76, **377** | 204, 6.93, 1.81, 51 |
| 0x2c | 399, 4.77, 1.03, 56 | 141, 6.51, 0.95, 12 |

0x38 costs the models about two hearts a visit more than the planner, so 0x3a is entered with about six. There
they die at every heart count: 148 of 177 entering below 4 hearts, **174 of 374 at 4 to 7.9** (the planner 15 of
71), 55 of 250 at 8 or more (the planner 8 of 90). On the full route (90 runs reach Dungeon 6 with 10.79 of 12
hearts) 0x38 costs 4.16 a visit and 0x3a kills 33 of 67.

**Level 9** (`d9`, from 15 hearts; 690 of 1000 runs lost):

| room | entered | hearts in | lost a visit | deaths | below 4 hearts in: deaths | 4 to 7.9 in: lost, models / planner |
|---|---:|---:|---:|---:|---|---|
| 0x17 to its wall | 998 | 10.85 | 2.42 | 33 | 8 of 12 | 1.71 / 1.00 |
| 0x16, patra's fight | 939 | 7.68 | 0.92 | 20 | 19 of 114 | 0.92 / 0.25 |
| 0x05 | 922 | 6.91 | 1.06 | 69 | 66 of 148 | 1.08 / −0.08 |
| 0x10, fought | 815 | 5.93 | 2.16 | **210** | 156 of 218 | 2.15 / 1.80 |
| 0x30, fought | 537 | 4.20 | 1.21 | **115** | 115 of 271 | 1.47 / 0.48 |
| 0x52 (patra) | 399 | 3.52 | 0.44 | 40 | 40 of 246 | 0.69 / 0.13 |

Level 9's deaths are the runs that enter its fights low: 0x30's 115 deaths are all below 4 hearts, 0x05's 66 of 69,
0x10's 156 of 210. The planner, at the same hearts, loses a tenth (0x05) to a third (0x30) of the models' hearts.

**The draws.** wizzrobe's 60 seed states are 18 of Dungeon 6 and 42 of level 9; its demonstration rows (pooled,
sampled uniformly) are 84 thousand of Dungeon 6 (`bomb28`, `plan38-3a`) among 512 thousand, 300 thousand of them
level 9's scripted rules (`wizzrobe-05`, `-10`, `-30`), which lose more hearts a visit than the model. No
curriculum track is unfinished (106 of 106 complete; patra 24 of 24).
