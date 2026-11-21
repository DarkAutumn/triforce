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

### 3. The trainings (all on `d3first`'s states only)

Each a continuation by the recipe: `state_routes: [d3first]`, `ram_randomization.chain_hp: 1.0` (the hearts the
route's chains bring), harvested handoffs at `harvested_fraction` 0.3, demonstrations at `--demo-coeff 0.05`,
`--sil-coeff 0.1`, 244 iterations (1M steps), `--parallel 8`, checkpoints every 30, `--stop-success 1.01`:

```sh
triforce train --specialist <family yaml> --load <run> --parallel 8 --iterations 244 --checkpoint-every 30 \
    --sil-coeff 0.1 --demos <files> --demo-coeff 0.05 --stop-success 1.01 --seed N --run-id <run>
```

| Family (from) | Runs | Handoffs | Demonstrations |
|---|---|---|---|
| goriya (`r32d1-goriya`) | `r34d7-gor-s1..s3` | Dungeon 1's 495 (round 32), Dungeon 2's 240 and Dungeon 7's 955 (0x38, 0x1a from the cellar, 0x29, 0x2b; half from chains on the old route data, half from the arm's) | round 32's Dungeon 1 planner rows (22,410); the planner on 12 Dungeon 7 stems, 15 runs a state at the chains' hearts (538 of 540 won; 0x1b 41 of 45, 0x49 45 of 45, 0x18 45 of 45): 25,072 rows; the walks' rule on 0x1b (`r32p_rules.py goriya-1b`, new: 66 of 90 at the chains' hearts with 1 bomb, 1.29 hearts lost in the wins): 18,593 rows |
| wallmaster (`r76-wall-1`) | `r34d7-wall-s1..s3` | 1:0x45 161, 7:0x0d 606 (the arm's) | the planner from 7:0x0d and 1:0x45 and 40 handoffs (7:0x0d 82 of 90 won): 21,104 rows |
| boss-aquamentus (`r76-b`) | `r34d7-aqua-s1` | 1:0x35 161, 7:0x2a 204 (the arm's) | the planner (1:0x35 74 of 90, 7:0x2a 120 of 122): 13,057 rows |

The explorer's `d3first` tracks for goriya: 42 of 45 states (Dungeon 7's and Dungeon 2's); 7:0x49 (a forced fight)
is unsolved, so it trains without a track, on the planner's rows.

### 4. The choice (each run's `last`; `--checkpoint FAMILY@d3first=RUN`; the route data in every chain)

Screen: `d7`, seeds 2000-2999, one batch (`r34d7-screen`); confirmation: seeds 3000-3999 (`r34d7-confirm`); paired
with the base (today's team on the route data). Only each run's last weights were measured (one batch for all six
runs, not a `select-checkpoint` per run: the CPU queue).

| Candidate | Screen | Paired | Confirmation | Paired | Rooms |
|---|---|---|---|---|---|
| base | 16.3% | | 16.4% | | |
| wallmaster s1 / s2 / **s3** | 37.7 / 38.5 / **39.3%** | +21.4 / +22.2 / **+23.0 ± 1.6** | **37.8%** | **+21.4 ± 1.6** | 0x0d grabs 387 → 120 of 605 runs (confirmation 397 → 139); the second pass's deaths 0x69 100 → 30, 0x18 128 → 33 |
| goriya s1 / s2 / s3 | 17.6 / 16.8 / 16.3% | +1.3 / +0.5 / 0.0 ± 1.7 | s1 with wallmaster s3: 36.0% | **-1.8 ± 2.1** against wallmaster s3 alone | 0x1b entered with 4.45-4.77 hearts (4.49), deaths 119-149 (146); 0x69, 0x59, 0x49, 0x19 lose 1.0-1.2 hearts a visit as before (the planner 0.4-0.6) |
| boss-aquamentus s1 (screen `r34d7-screen-aqua`) | `d1` 94.2% vs 89.2%; `d7` 17.1% | **+5.0 ± 0.7**; +0.8 ± 0.3 | (the final batch) | | 1:0x35 deaths 79 → 29 of 971 (3.29 hearts in); 7:0x2a 19 → 11 of 182 |

`d1` holds for wallmaster (it plays 1:0x45, step 76): 89.1% against 89.7%, -0.6 ± 0.7; 1:0x45 loses nothing with
either run. Wallmaster and boss-aquamentus play no Dungeon 2 room, and the route data changes only Level 7, so `d2`
is untouched by construction (goriya, the only family here that plays Dungeon 2, is not chosen).

### 5. The chosen runs together (`r34d7-final`, seeds 4000-4999; every chain pins both families by `--checkpoint`)

| Segment | Today (route data, `r76-wall-1`, `r76-b`) | wallmaster `r34d7-wall-s3` | + boss-aquamentus `r34d7-aqua-s1` | Paired |
|---|---|---|---|---|
| `d7` | 12.9% | 32.8% | **34.4%** | team − today **+21.5 ± 1.6**; aquamentus's share +1.6 ± 0.4 |
| `d1` | 90.4% | | **93.6%** | **+3.2 ± 0.7** (1:0x35 deaths 65 → 33 of 969; 1:0x45 0 and 0) |
| `d7`, the planner (300, the team) | | | **98.3%** | |

Against the record (main's data and team, 9.4% on seeds 0-999) `d7` is 34.4%. Where the team still loses `d7` (of
1000): Digdogger 0x1c 207 (entered with 3.33 hearts), goriya's 0x1b 164 (4.32 hearts in, crossing), the second pass
after a grab 106 (grabs 361 → 132 of 540 runs), 0x1a 51, Aquamentus 0x2a 30 of 372, 0x29 24, 0x19 21. Cheated in
Level 7: bombs 0 → 2, keys 0 → 0; Dungeon 1's 0x53 wall bomb in 999 runs as before. The key ledger at 0x1b's E door:
no model visit spends two keys (the planner 1 of 300).

### Verdicts

| Family | Verdict |
|---|---|
| route data (Level 7's bag) | `ROUTE DATA` 9295f8d0: `d7` +9.0 ± 1.5 on today's team; closes goriya's key leak |
| wallmaster | **`r34d7-wall-s3` promoted for `d3first`**: `d7` +21.4 ± 1.6 confirmed, `d1` -0.6 ± 0.7 |
| boss-aquamentus | **`r34d7-aqua-s1` promoted for `d3first`**: `d1` +3.2 ± 0.7, `d7` +1.6 ± 0.4 |
| goriya | not chosen: no seed better beyond noise on `d7` (+1.3, +0.5, 0.0 ± 1.7; s1 -1.8 ± 2.1 beside wallmaster's run). Training at the chains' hearts with the planner's rows did not reduce the hearts goriya loses before 0x1b (about 6.4, against the planner's 1.8). Its next arm is the hearts it loses in 0x69, 0x59, 0x49 and 0x19, judged on `d7` with the team above |
| boss-digdogger | held (`r32late-dig`); now `d7`'s first loss (0x1c 207 of 1000, entered with 3.33 hearts) |

`d2first` keeps every run (`wallmaster` and `boss-aquamentus` promoted only with `--route d3first`).

## Dungeon 7, second pass: goriya's hearts before 0x1b, Digdogger's 0x1c (goriya, boss-digdogger)

Branch `r34-d7`, on the route data of 9295f8d0 and the team of §5 (wallmaster `r34d7-wall-s3`, boss-aquamentus
`r34d7-aqua-s1`; goriya `r32d1-goriya`, boss-digdogger `r32late-dig`).

### 6. The cause, room by room

What the route asks of each goriya room before 0x1b, what the game demands (`zelda-asm/levels/level-7.md`), and how
the team (`r34d7-final-d7-team`, seeds 4000-4999) and the planner (`r34d7-final-d7-plan`, 300) play it. First
visits: hearts lost a visit, decisions on which Link is hit, boomerangs parried (`blocks`), kills.

| Step | Room, way | The route's demand | The game's | Team: lost / hit / parried / kills | Planner |
|---|---|---|---|---|---|
| 175 | 0x69 S→N | bomb the N wall (`treasure: skip`) | the wall only; the room bomb is optional | 0.97 / 5.2 / 0.16 / 0.6 | 0.33 / 1.8 / 0.51 / 0.8 |
| 176 | 0x59 S→N | cross (no item) | nothing | 1.12 / 5.6 / 0.13 / 0 | 0.38 / 1.9 / 0.40 / 0 |
| 177 | 0x49 S→N | the N shutter | **clear** 3 Blue Goriyas and 3 Keese (Bubbles do not count) | 1.11 / 8.5 / 0.95 / 5.2 | -0.12 / 1.7 / 1.67 / 5.6 |
| 181 | 0x18 S→E | the E key door (dark) | a key | 0.61 / 4.3 / 0.17 / 0 | 0.28 / 1.6 / 0.34 / 0.1 |
| 182 | 0x19 W→E | bomb the E wall | the wall; the six need not die | 1.11 / 4.4 / 0.05 / 0 | 0.33 / 1.35 / 0.50 / 0.2 |
| 183-185 | 0x1a W→stairs, cellar 0x4a, 0x1a→E | the red candle, then the E wall | with clips the block clip to the stairs, no kill; the wall | 0.36 + 0.16 + 0.94 | 0.12 + 0.04 + 0.21 |
| 186 | 0x1b W→E | the E key door (`if_bombs_below: 2`, 5 bombs in) | a key | 1.60 / 6.5 / 0.29 / 0 | 0.76 / 3.0 / 1.19 / 0.2 |

On the same start states and seeds (the room's three chains, 40 runs each, at the hearts the chains bring;
`r27_plan_demos.py --route d3first`, the model at its chain act `temperature:0.5+relax:16`), hearts lost in the
wins, model against planner: 0x69 1.00-1.34 / 0.19-0.38, 0x59 0.95-1.37 / 0.67-0.69, 0x49 0.67-0.95 / -0.44 to -0.76,
0x18 0.57-0.72 / 0.14-0.39, 0x19 1.21-1.38 / 0.33-0.72, 0x1a to the stairs 0.42-0.58 / 0.08-0.15, 0x1a from the
cellar 0.58-1.11 / 0.10-0.26, 0x1b 1.52-1.77 / 0.42-0.56 (wins 104 / 114 of 120).

**The route already crosses every goriya room but 0x49, and 0x49 is the game's shutter.** No step fights for a drop
(0x69's room bomb is skipped since round 26; 0x1b's bombs are wanted only below 2 since 9295f8d0), and no kill
opens anything else on the way: the model kills nothing in 0x59, 0x18, 0x19 and 0x1a (with clips the block clip
takes Link to 0x1a's stairs), as the planner. Both stay about as long in each room. The difference is play: in the
same rooms over the same time the model is hit about three times as often and parries a boomerang a third to a
quarter as often; the planner's foresight (`plan:16:1`) replaces exactly the decisions that would be hit.

The ledger on this route: Link brings **5 keys** into Level 7 and spends 3 (0x38 N, 0x18 E, 0x1b E), 2 left, none
cheated; **8 bombs**, 3 walls before 0x1b (0x69 N, 0x19 E, 0x1a E) so 5 at 0x1b, about 2.2 on Digdogger's small ones,
then 0x0c's and 0x29's walls (0x29's cheated in 2 runs of 1000).

### 7. The route data: no arm

The arms the round set out to measure (a fight the route does not need turned into a crossing, an optional detour
dropped) do not exist on these rooms: 0x69 is `treasure: skip`, 0x59, 0x18, 0x19 and 0x1a have no fight on the
route's way, 0x1b is crossed with the bag 9295f8d0 keeps, and 0x49's shutter is the game's. The one detour is the
cellar 0x4a, the red candle, which the route uses after Dungeon 7 (Dungeon 6's dark rooms, the trees of 0:0x6d and
0:0x47 with a free retry; the blue candle burns once a room visit); dropping it would save about 0.5 hearts a run
(0x1a's first visit 0.36 and the cellar 0.16; 0x1a's E wall stays) at the cost of every later segment's candle, so
it is not taken. No `ROUTE DATA` commit; the base stays 9295f8d0.
