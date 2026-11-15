# Round 25: leaving rooms with more hearts

The owner (Nov 15) set the round's direction: "the problem with current death rate is that we enter dangerous rooms
with too few hearts", and fix that, starting from the backlog's hearts item but exploring beyond it. Results are
reported clips on and clips off, but clips on is the version being optimized: "you are optimizing clipping, not
no-clip". He added one concrete fault: in the Manhandla fight Link often leaves without taking the bombs Manhandla
drops ("force manhandla to always drop bombs in training and make it a primary objective to pick them up on win").
The standing limits hold: the route is not changed, and no potions are bought. Every experiment that changed code ran
in its own worktree and branch off the integration branch `r25`, with its own build, models root and trajectories.
The yardstick is the chain, 1000 runs per segment. The baseline is round 24's team as promoted (chains tag `r74-1k`,
clips on | off): D2 89.9 | 89.9, D3 67.3 | 67.3, D4 81.3 | 59.7, the leg 99.8 | 92.0, D5 34.9 | 22.0 (46.4 | 34.0
with `darknut/r74-darknut5`), and the full route to the D5 triforce 4.8 | 0.3. Measurements are paired by seed against
that baseline wherever possible.

## 1. Where the hearts go: the ledger and the value of a heart

Analysis only, with no code and no training. The inputs were round 24's 1000-run chain files: the `r74-1k` and
`r74-1k-noclips` segments and the full route (team as promoted at 02:34 Nov 15, darknut `r74-darknut4`), plus
`r74b-1k` (darknut `r74-darknut5`), 1,275,046 room records in all. The records hold no damage field. A visit's
*hearts lost* is therefore rebuilt as health in + drops healed + container + other gains + triforce refill − health
out, a lower bound wherever a fairy came in the same visit. A *leg* runs from one refill to the next (leg 0 the start
to the D2 triforce, leg 1 to D1's, leg 2 to D3's, leg 3 to D4's, leg 4 to D5's). Natural numbers come from the
full-route file. The segment files start from fixed states whose bombs are not natural (see below).

**The value of a heart.** V(s, h) = P(Link is alive after taking the next triforce | he enters route step s's room
with h hearts). The next triforce refills him, so hearts stop mattering there. This definition is the same for every
model up to the next refill, and it pays nothing for hearts carried into a triforce. Its scope is a choice: a
to-the-end V would scale each leg by one factor, Dungeon 5's skill, so its slope in Dungeon 2 is 5-20 times smaller
for reasons that are not the room's. V is fitted per step from all files of a setting pooled (Dungeon 5 from the
`r74-1k` D5 segment, darknut4). Three parts blend:

- an empirical part: half-heart cells, kernel-smoothed to at least 40 visits and made monotone; at the 57 steps where
  bombs matter (clips on) it is fitted per bombs class and averaged with the natural class shares;
- a model part: a backward pass over the route through each step's death logistic and its survivors' losses;
- a weight on each by the number of visits within half a heart of the cell.

Bootstrap 90% intervals are kept per cell (e.g. Gleeok at 2 hearts 0.57-0.64, the Olive Garden at 6 hearts
0.15-0.18). The check: V averaged over each refill's and dungeon entrance's natural arrivals reproduces the full
route's measured arrival rates within 1-2 standard errors everywhere. The worst case is clips off at 0:0x3c, 69.8
against 66.0 ± 1.8.

V by hearts on entry, at 1 / 2 / 3 / 4 hearts (clips on):

- Gleeok 0.23 / 0.60 / 0.87 / 0.95.
- 4:0x12 0.15 / 0.50 / 0.75 / 0.92.
- 5:0x37 0.04 / 0.55 / 0.82 / 0.94.
- 5:0x66 W→N 0.05 / 0.16 / 0.29 / 0.54 (0.98 at 10).
- D1's entrance 0.38 / 0.68 / 0.87 / 0.96 (1.00 at 5).

**Where the route bleeds.** Hearts lost per 1000 full runs (clips on): the Olive Garden 1344, 3:0x59 N→S 852,
Manhandla 754, 2:0x5e 703, 2:0x1e 656, 2:0x3e 588, Gleeok 538, 4:0x32 (like-like) 462. By family: darknut 3166,
goriya 1898, ow-zora 1265, stalfos-keese-gel 1020. Rooms that kill at least 1% of the full runs: 24 clips on, 25 clips
off. The killer rooms, clips on | off:

| step | room | family | deaths /1000 full runs | dies, of visits | hearts on arrival (mean) | V at arrival (mean) |
|---:|---|---|---|---|---|---|
| 103 | 3:0x4d Manhandla | boss-manhandla | 185 \| 133 | 31% \| 33% | 3.35 \| 3.02 | 0.70 \| 0.69 |
| 192 | 5:0x05 the Olive Garden | darknut | 150 \| 18 | 47% \| 69% | 5.85 \| 3.70 | 0.17 \| 0.17 |
| 132 | 4:0x13 Gleeok | boss-gleeok | 70 \| 48 | 17% \| 20% | 3.78 \| 3.47 | 0.85 \| 0.78 |
| 21 | 2:0x1e S→N | goriya | 48 \| 53 | 5% \| 7% | 2.69 \| 2.59 | 0.94 \| 0.92 |
| 17 | 2:0x5e S→N | goriya | 44 \| 97 | 4% \| 10% | 2.74 \| 2.30 | 0.87 \| 0.76 |
| 197 | 5:0x64 cave→E (way back) | darknut | 37 \| 0 | 22% \| – | 4.34 \| – | 0.29 \| – |
| 57 | 1:0x23 S→W | goriya | 32 \| 33 | 4% \| 6% | 2.92 \| 2.60 | 0.89 \| 0.84 |
| 199 | 5:0x66 W→N | gibdo | 30 \| 1 | 25% \| – | 3.48 \| – | 0.40 \| – |
| 95 | 3:0x69 N→cave | darknut | 24 \| 23 | 4% \| 5% | 3.88 \| 3.60 | 0.63 \| 0.63 |
| 64 | 1:0x44 W→E | goriya | 23 \| 23 | 3% \| 4% | 2.83 \| 2.53 | 0.95 \| 0.92 |
| 19 | 2:0x3e S→N | moldorm | 22 \| 51 | 2% \| 6% | 2.64 \| 2.30 | 0.92 \| 0.86 |
| 42 | 0:0x27 N→E | ow-zora | 18 \| 21 | 2% \| 3% | 3.69 \| 3.23 | 0.80 \| 0.70 |
| 202 | 5:0x47 S→N | gibdo | 18 \| 0 | 20% \| – | 3.41 \| – | 0.51 \| – |
| 203 | 5:0x37 S→N | darknut | 16 \| 1 | 23% \| – | 3.18 \| – | 0.65 \| – |
| 189 | 5:0x64 E→cave (plain way, off) | darknut | 11 \| 135 | 3% \| 84% | 5.86 \| 2.90 | 0.16 \| 0.03 |

Only 333 full runs reach Dungeon 5 clips on. So the Dungeon 5 rows take their arrival hearts from the runs that start
at the D4 triforce (`post-d4`, 1000 runs), and their death shares from the pooled files. "–" marks a room that is not
a killer clips off.

**Manhandla is a bombs problem, not a hearts problem.** V at Manhandla is 0.49 at 1 heart, 0.73 at 3 and 0.77 at 6.
From 3 hearts up, 25-28% of the fights are lost at any health. Each bomb held adds +0.21 to the log-odds, so going from
3 to 7 bombs is worth about +17 points. A bomb there is worth about as much as a heart (§3).

**A heart counts where Link keeps it.** In Dungeon 5 the killers chain. One more heart carried into a room of the way
back is worth +4.1 to +4.7 points of the D5 leg (into the Olive Garden: +3.9). One more heart that only saves Link in
that room is worth +0.1 to +0.2 points there, because the saved Link goes on with half a heart into the next killer
(5:0x37, the last killer, is the exception at +2.8). Gleeok and Manhandla are the opposite case: the triforce is next
door, so a heart there counts in full (Gleeok +10.1 points of leg 3 per heart). Links who arrive low do not play
worse beyond their hearts. Once hearts and bombs are known, the hits a run took in the 12 rooms before a killer add
nothing to its death odds (every coefficient within 2 SE of 0; the Olive Garden +0.54 ± 0.29). This test is weak,
because hits are not the same as a bad state.

**What the low arrivals come from.** Each killer's low arrival traces to a few rooms before it:

- Manhandla (3.35 of 6 hearts): 3:0x59 N→S took 1.12, 3:0x69 0.56, 3:0x5a 0.35.
- Gleeok (3.78 of 7): 4:0x32 (like-like) took 1.09, the vire rooms 0x31 / 0x50 / 0x12 0.49 / 0.48 / 0.34.
- The Olive Garden (5.85 of 10): the leg's 0:0x1f took 1.02, 5:0x65 westbound 0.67, 0:0x2d 0.66.
- 2:0x1e (2.69 of 3): moldorm took 0.61, 2:0x5e 0.60.

**What 20% fewer hearts lost would be worth.** The estimate is first order from V, with interactions between
changed rooms ignored, so the large changes are optimistic. Full route, clips on | off:

| change | full route |
|---|---|
| baseline (full-route file) | 4.8 \| 0.3% |
| every visit loses 20% fewer hearts | 11.9 \| 1.1% (every leg +6.5 to +11 points clips on) |
| visits entered with ≤ 2 hearts take 20% fewer hits | 5.9 \| 0.4% |
| visits entered with ≤ 3 hearts take 20% fewer hits | 6.8 \| 0.5% |
| every heart left on the floor while hurt is picked up | 5.3 \| 0.6% |
| one more heart at every killer, not carried on | 8.9 \| 0.7% |
| arriving at Dungeon 5 with full hearts (V over the D5 arrivals) | D5 leg 16.2 → 33.7 \| 2.4 → 22.0% |

The same full-hearts bound at D1's entrance is 82.4 → 99.6 | 73.8 → 99.4%. Low-health play alone is worth little,
because most hearts are lost at higher health in the rooms before.

**The targets per room and per family** (full route, clips on, if that room or family lost 20% fewer hearts): the
Olive Garden +1.02 points, 5:0x64 cave→E +0.50, 5:0x66 W→N +0.36, Gleeok +0.34, 5:0x47 +0.29, 5:0x37 +0.25, Manhandla
+0.25, 3:0x59 N→S +0.18, 5:0x65 E→W +0.18, 5:0x65 W→E +0.15, 2:0x5e +0.14. (The report's summary lists 0x66 at +0.37
and 2:0x5e at +0.15; its table gives the values above.) One heart less lost per visit at the leg's 0:0x1f alone adds
+3.1 points to the D5 leg. By family:

| family | 20% less loss: full route, clips on | one heart less a visit, clips on | 20% less loss, clips off |
|---|---:|---:|---:|
| darknut | +2.05 | +4.19 | +0.18 |
| gibdo | +1.02 | +1.91 | +0.06 |
| goriya | +0.37 | +0.90 | +0.03 |
| boss-gleeok | +0.34 | +0.62 | +0.03 |
| boss-manhandla | +0.25 | +0.31 | +0.01 |
| vire | +0.17 | +1.09 | +0.02 |
| ow-peahat | +0.13 | +1.15 | +0.02 |
| like-like | +0.13 | +0.46 | +0.01 |
| ow-zora | +0.07 | +2.01 | +0.04 |
| stalfos-keese-gel | +0.06 | +1.11 | +0.01 |

A heart is worth most after the Olive Garden and before Gleeok. It is worth nothing before a triforce, or in rooms
crossed with plenty of hearts (Digdogger, Aquamentus, the D4 entrance).

**The supply is nearly used.** Per full run, clips on | off: 15.99 | 12.83 hearts lost; 8.30 | 6.15 dropped; 6.12 |
4.36 taken. The taken hearts healed 4.71 | 3.52, and 1.41 | 0.84 were wasted at or near full health. Left on the
floor: 0.53 | 0.79 while Link was hurt, 1.47 | 0.88 while he was full. Heart containers give 3.82 | 2.77 hearts and
triforce refills 4.57 | 3.15. Of the hearts lost, drops give back 29 | 27%, refills 29 | 25%, containers 24 | 22%.
Capped at Link's deficit, the hearts left while hurt are worth 0.38 a run. Most of them lie on the overworld: 0:0x1a
N→W 0.081 a run, the 0x2c heart cave 0.071, 0x2c E 0.042, 0:0x4c 0.040, 0:0x0d 0.038. Wallmaster, trap-room,
equipment-room, ow-moblin, ow-lynel and ow-peahat rooms drop nothing.

The ledger's first fairy count was 0.04 a run (gains of 2.5 hearts or more). That undercounts: the direct count of §2
is 1.06 fairies dropped and 0.52 taken a run, healing about 0.76 hearts. The report infers that fairies give back about
5% of the hearts lost (3% by the ledger's own count).

**The segment-start bombs.** Several segment states carry bombs that natural runs do not:

| start | natural arrivals carry | the state carries |
|---|---:|---:|
| Dungeon 4 | 5.0 | 0 |
| Dungeon 5 | 1.2 | 4 (the `post-d4` state: 0) |
| Dungeon 1 | 5.4 | 1 |

From the D3 triforce, runs from the 0-bomb `post-d3` state reach the D4 triforce in 73.3 ± 1.4% of runs; full-route
runs (5 bombs) reach it in 79.7 ± 2.0% (clips off 48.8 ± 1.6 against 69.6 ± 2.8). Within Dungeon 4 the bombs matter:

- runs entering with 4+ bombs die 14.4% of the time, against 22.2% with none (clips off 21.2 against 41.3%);
- the 4+ bomb runs reach Gleeok with 4.06 hearts, against 3.59.

V, fitted for the natural bomb mix, matches the full-route rows and not the 0-bomb `post-d3` rows. So the round 24
D4 segment (81.3 | 59.7%) understates Dungeon 4, most of all clips off, and Dungeon 4 is to be judged from the full
route. This is a comparison between files, not a paired test. The report names the settling measurement: a paired
300-run D4 chain from `4_71s_0` with 0 and 5 bombs, clips on and off. It is not part of this analysis.

**Limits.**

- Dungeon 5's V is darknut4's. The promoted darknut5 had played only from Dungeon 5's entrance, so V should be refit
  once chains from the D4 triforce exist with it.
- Hearts on arrival are not set by experiment. The proposed causal check is single-room evaluations from harvested
  chain states with hearts set to h and h + 1, at Gleeok, 5:0x37, 5:0x66 and 5:0x64; it was not run.
- Leg 4's to-the-end factors rest on 48 full runs that finish, ± 20%.

## 2. Supply: drops as targets

Branch `r25-pickups` (merged into `r25` at `f2ca1e4`).

**What was built:**

- **Supply fields in the chain records** (`132e88c`, a new `zelda::supply` module; the Python `RoomRecord` matches):
  - `kills`;
  - `streak_in` and `streak_max`, the damage-free kill streak (`$627`);
  - `fairies_dropped` and `fairies_taken`;
  - `drops`: one entry per dropped heart or fairy, giving how hurt Link was, his walking distance to the drop and to
    the exit, whether it was taken, expired or left, and whether he was hit meanwhile.
- **`reward.heal_targets`** (off by default): dropped hearts and fairies, and nothing else, become shaping targets
  while Link is hurt. Every dungeon family already had `item_targets` (`crates/env/src/objective.rs`
  `is_item_target`). The overworld families pointed only at the exit.
- **Turned on** for `ow-tektite`, `ow-octorok` and `ow-zora` in its own commit (`8584d90`), with no training.

**The drop rule** (`zelda-asm/enemies.md` "Drop tables", asm:Drop_SetUpItem). The checks run in order:

1. some types never drop;
2. a fairy if the damage-free streak reads exactly 16;
3. the forced 10th-kill drop: bombs or 5 rupees, never a heart;
4. the table item, fixed by the monster's group and a kill counter (`$52A`, 0-9) that every death advances and nothing
   resets. It appears only if a random byte from the RNG on the death's last frame is under the group's rate (31 /
   59 / 41 / 41%).

A drop lasts about 510 frames and cannot be taken for the first ~32. Because the counter is deterministic, the kill
order can in principle choose what drops. Expected hearts per kill (a fairy counted as 3):

| group | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| hearts per kill | 0.22 | 0.12 | 0.12 | 0.49 |

A group-3 monster gives a heart or fairy in 8 of the 10 counter positions. Choosing the order would need `$52A` in
the observation, a schema change, for a few tenths of a heart. Not pursued.

**Streaks and fairies.** The instrumented baseline was 1000 full runs, clips on, with the team including darknut5
(`r75-pk-base1k-full`; 6.1% reached the D5 triforce):

- 102.5 kills a run;
- 93% of runs reach a 10-kill damage-free streak (3.3 times a run), 74% a 16-kill one (1.4 times a run), most often at
  Manhandla (its base counts as a fresh kill on each hit once its health is 0), moldorm, 2:0x4e, 4:0x12 and 2:0x1e;
- 1.06 fairies dropped and 0.52 taken a run, healing about 0.76 hearts, mostly from the drop table, not the streak;
- fairies left while Link was hurt are worth 0.12 hearts a run (Aquamentus' room excluded: the triforce refills Link
  next door).

Of 8.2 hearts dropped a run, 0.27 are left while Link is hurt by a heart or more. These counts use a different team
and threshold from §1 (8.30 dropped, 0.53 left while hurt at all). The left hearts are the overworld families': the
heart lies a median 2 tiles away and the exit 9-17, and the model walks past.

**The zero-training result.** `heal_targets` on the three overworld families, promoted models unchanged, 300 runs from
`0_3cc_0` to the D1 triforce, clips on, paired by seed. The baseline matched round 24's runs seed for seed (300/300).

| | baseline | heal_targets |
|---|---:|---:|
| reached the D1 triforce | 253 | 253 (10 seeds swap each way) |
| hearts at D1's entrance | 3.29 | 3.40 (paired +0.117 ± 0.038) |
| frames to D1's entrance (paired) | | −15 ± 9 |
| hearts healed a visit, 0x2c / 0x2c cave / 0x1a | 0.002 / 0.027 / 0.041 | 0.033 / 0.067 / 0.098 |

0x4c on step 25 is unchanged: Link is at full health there, just after the D2 triforce. By §1's value table the gain
is about +1 point of Dungeon 1 (the report's estimate, not measured).

**Not run:**

- `heal_targets` clips off, and on the leg to Dungeon 5 (0x0d, 0x2d, 0x3f).
- Any 1000-run chain.
- Usable training: the arms `r75-pk-heal` and `r75-pk-base` were stopped when the experiment was cut back after §1,
  and they are unequal (ow-tektite's arm early-stopped at 0.95 success after 58 iterations).
- A wallmaster gathering arm with heart pricing and a `train --gamma` option was prepared on branch
  `r25-pickupswm` (`a9f4ce1`) and never trained. The old gathering model (r56) is two layouts behind and does not
  convert. Its ceiling is Aquamentus' 16 deaths per 1000 full runs, all at under 1 heart.
- Protecting the streak: fairies already fall about once a run.

## 3. Manhandla: the dropped bombs and the fight

Branch `r25-manhandla` (`f2c2917` code, `26809c2` yaml and docs; merged into `r25` at `8e3e99e`). Details are in
`docs/routes/d3.md` "Manhandla's drop" and decisions 2026-11-15 "Manhandla's dropped bombs first".

**What the game does.** A hand that dies while another lives becomes a dead dummy and drops nothing. The last hand is
Manhandla's one counted kill and his one drop. Its group (3) holds hearts, fairies and rupees but no bombs. Bombs come
only from the help drop: `HelpDropCount` (`$50`) at `$0A` with `IsHelpDropBomb` (`$51`) set. A bomb blast over the
base counts as a kill on every frame, so a bomb kill nearly always arms the help drop. Two things override or cancel
it: a fairy wins when the death ends at the 16th damage-free kill, and any hit on Link clears both bytes. The drop
lies where the last hand died, flashes untakeable for 32 frames and lasts about 510.

**Before** (r71-e, clips on). Per kill the drop was bombs 49%, rupees 16%, fairies 5%, hearts 5%, nothing about 25%
(evaluation, 216 kills). Link left the bombs on the floor:

- 5 of 106 in the evaluation, all in the room's west part;
- 15 of 344 in the D3 segment;
- 23 of 246 on the full route to the D3 exit.

That is 4-9% of the time, less often than "often". A pickup costs about 40 frames from the drop appearing, 32 of them
the unavoidable flashing. Link left the fight with 4.26 bombs (evaluation) and entered Dungeon 4 with 4.15 (chains).
The fight was decided by bombs:

- with 0 / 1 / 2-3 / 4-5 / 6-8 bombs it was won 0% (0 of 16) / 11% (2 of 18) / 70% / 79% / 85%;
- at 1-6 hearts it was won 61-83%, with no trend;
- 79 of 86 deaths ended with no bombs left, 42 of them with one hand still alive;
- bomb misses averaged 1.8 a death against 0.7 a win;
- hits came from fireballs (235) and the segments (206).

**What changed** (`specialists/boss-manhandla.yaml`):

- `ram_randomization.drop_bombs: 0.75` arms the game's help drop as bombs at reset and after every step while a
  monster is in the room, in 3 of 4 resets. It applies to training envs only (`EnvOptions::forced_drops`), with about
  0.5-0.7 bomb drops an episode; evaluation and chains keep the natural drop.
- `reward.drops_first`: once the fight is over, a wanted drop is the only shaping target, ahead of the heart container
  and the N exit. Wanted means bombs below the maximum, or a heart or fairy when hurt. Chains build their target the
  same way. It builds on the same drop-target mechanism as §2's `heal_targets`.
- `reward.left_drop_penalty: 3.0` per wanted drop left on the floor.
- Bombs drawn 2-8 instead of 0-8: the route arrives with 2-8, and the old model never won with 0-1.
- Chain records count dropped bombs (dropped, taken, left, expired) and the bombs carried in and out of each room.

`r75-mh-drop` is r71-e + 1M steps; `r75-mh-fight` is r75-mh-drop + 1M steps with bombs 2-8.

**Results.** Chains are 1000 runs from `3_7cs_0` stopped at step 134 (out of Dungeon 4), the same seeds for every
model; ± is the standard error of the paired difference. The family has no clip rooms, so everything up to the
Dungeon 3 exit is identical clips off.

| | r71-e (baseline) | r75-mh-drop | r75-mh-fight |
|---|---|---|---|
| evaluation success (300 episodes, bombs 0-8) | 71.3% | 80.7% | 82.7% |
| evaluation hearts lost | 1.31 | 1.11 | 1.08 |
| evaluation dropped bombs left | 5 / 106 | 0 / 97 | 0 / 88 |
| D3 segment: Manhandla won (930 visits) | 629 | 680 | 758 |
| D3 segment: arrivals out of Dungeon 3 | 62.9% | 68.0% (+5.1 ± 1.7) | 75.8% (+12.9 ± 1.7) |
| D3 segment: dropped bombs left | 15 / 344 | 0 / 339 | 0 / 305 |
| out of Dungeon 4, clips on \| off | 49.7 \| 44.2% | 54.9 \| 45.8% (off +1.6 ± 2.0) | 60.2 (+10.5 ± 2.0) \| 51.9% (+7.7 ± 2.0) |
| full route, Dungeon 3 exit (step 105) | 36.9% | not run | 45.4% (+8.5 ± 1.4) |
| bombs on entering Dungeon 4 (mean) | 4.15 | 3.84 | 3.64 |
| D4 runs whose 0x11 wall opened with real bombs (step 128) | 369 of 618 | 342 of 659 | 402 of 728 |

Clips off, the drop arm's +1.6 ± 2.0 out of Dungeon 4 is not a result.

**What else changed:**

- Kill to exit takes about 300 controlled frames in all three models, so there is no added lingering. The idle
  stretch left in the evaluation fell from 13 steps to 1.
- In chains, by bombs on entry, fights with 2-3 bombs were won 54% → 77% and with 6-8 bombs 76% → 87%.
- The new model also finishes with the sword: it won 6 of 16 fights with 0 bombs and 14 of 18 with 1.
- The fight is now sensitive to health: 95% won from full health, 43% at 1 heart.

`r75-mh-fight` is promoted in `r25`'s models root.

**Open:**

- The fight arm changed two things at once, the 2-8 bomb range and a second million steps. No control arm separates
  them.
- Sword kills arm no bomb drop. Kills drop bombs 34% of the time instead of 49%, so Link enters Dungeon 4 with about
  0.5 fewer bombs, and each Dungeon 4 run uses more cheated bombs. Real-bomb opens of the 0x11 wall still rose in
  count, and so did Dungeon 4 arrivals. A way to keep the bomb kill and its drop while keeping the sword finish is
  open (`docs/routes/d3.md` "Open questions").
- At 1 heart the fight is still lost about half the time.
- A fairy at the 16th kill still overrides the forced bomb drop in training, as the game has it.

## 4. Low-health act rules (rejected)

Branch `r25-act` (`97c75bf`). The idea: act more carefully when Link is low. The mechanism is an act-mode suffix,
`MODE+low:H:LOW`: below H hearts (or a share of the maximum), decisions use the act mode LOW instead of MODE, each part
keeping its own stall rule. It is tested and documented, with all suites green.

The screen used 300 runs per candidate, clips on, on the D5, D3, D4 and D2 segments (D3, D4 and D2 stopping at their
arrival room). The seeds were the same as the standing act mode `temperature:0.5+relax:16`. Cells are the paired
difference in arrivals ± SE, in points:

| low-health part | D5 | D3 | D4 | D2 |
|---|---:|---:|---:|---:|
| standing rule (arrive %) | 47.7 | 64.7 | 83.3 | 88.3 |
| `low:1:temperature:0.25+relax:16` | −1.0 ± 1.1 | −0.7 ± 1.2 | −1.0 ± 1.4 | −2.7 ± 1.9 |
| `low:2:temperature:0.25+relax:16` | +1.3 ± 1.9 | −1.7 ± 1.8 | −2.3 ± 1.9 | +0.3 ± 2.2 |
| `low:3:temperature:0.25+relax:16` | +2.7 ± 2.4 | +0.3 ± 2.6 | −3.3 ± 2.4 | +0.3 ± 2.6 |
| `low:40%:temperature:0.25+relax:16` | −0.3 ± 2.9 | −1.7 ± 1.8 | −2.3 ± 2.3 | −2.7 ± 1.9 |
| `low:2:temperature:0.35+relax:16` | −1.3 ± 1.8 | −1.0 ± 1.7 | −0.3 ± 2.2 | +1.0 ± 2.3 |
| `low:2:escape:16` | +0.0 ± 2.2 | −3.0 ± 2.0 | −2.7 ± 1.9 | +1.3 ± 2.4 |
| `low:2:floor:0.1+relax:16` | +0.3 ± 1.9 | −1.3 ± 1.9 | −1.0 ± 2.0 | +0.3 ± 2.4 |

No candidate passes two standard errors anywhere, and the same rule moves both ways across segments (`low:3` +2.7 on
D5, −3.3 on D4). Hits per decision at 2 hearts or fewer never drop:

| segment | standing rule | candidates |
|---|---:|---:|
| D5 | 1.42% | 1.43-1.54% |
| D3 | 1.87% | 1.79-2.07% |
| D4 | 1.20% | 1.24-1.46% |
| D2 | 1.20% | 1.13-1.26% |

A sharper draw at low health does not get Link hit less, so it does not keep him alive. Deaths and stuck runs follow
the arrivals (D5: 154 deaths and 3 stuck runs for the standing rule, 147-158 deaths for the candidates), and no rule
adds loops. Play in the first room is unchanged. Acting entropy per decision falls by up to 0.06 on D5-D4 (D5 0.45 →
0.39-0.46), and on D2 from 0.48 to 0.26 for `low:3`. `low:40%` equals `low:1` on D2 and `low:2` on D3, which fits 40%
of 3 and of 5 hearts; this was not checked. The result fits §1's estimate that low-health play alone is worth little
(20% fewer hits on visits entered with ≤ 2 hearts: full route 4.8 → 5.9%).

**Verdict:** none of these rules is adopted. The mechanism stays on its branch as tooling. Not run: any 1000-run chain,
any clips-off chain, and a danger-based rule (sharpening the draw is the wrong tool for that).

**The lead it left.** A one-step lookahead, `guard:K` (branch `r25-actguard`, `54d7115`, work in progress; only the
build and the act-mode unit tests had run). Before each guarded decision it tries the chosen action in the emulator and
comes back. If the action would cost health, Link takes the first of the mode's K best actions that would not. It
uses the game's real random numbers, not redrawn ones: it plays with one step of foresight. A 20-run Dungeon 5 smoke
test on the standing rule's seeds (0-19):

| mode | reached the D5 triforce | standing rule | runs better / worse | decisions replaced |
|---|---:|---:|---:|---:|
| guard only at ≤ 2 hearts (`+low:2:…+guard:8`) | 10/20 | 8/20 | 2 / 0 | 74 of 1898 |
| guard on every decision (`temperature:0.5+relax:16+guard:8`) | 16/20 | 8/20 | 9 / 1 | 369 of 12366 |

Guarding every decision does better than guarding only when low: the lever is not being hit at any health, so Link
reaches low health less often. Twenty runs are a lead, not a result. The guard is carried on as its own experiment.

## 5. Rooms that bleed, Dungeon 5: room 5:0x37 and `path_costs`

Branch `r25-bleed-d5` (`da65186` code, data and tests; `3445b00` docs; merged into `r25` at `03f9004`).

**Method.** The baseline was the D5 segment (`5_76s_0`), 300 runs, clips on, with replays: 143/300 = 47.7%. It is
bit-identical, seed for seed, to seeds 0-299 of round 24's `r74b-1k` D5 segment (300 of 300 with the same result and
frames). Every replay was stepped frame by frame through the D5 visits, recording each health drop with Link's
position, the ladder, the monsters and the last decision. The chain's own room-entry machine states fed two kinds of
scripted runner: frame-level paths, and action-level policies through the env's executor (what a model can do).

**The spawn cycles on the way back are fixed by the route.** At each room after the Olive Garden the chain meets 2 of
the game's 9 spawn cycles (0x64 back 7/1, 0x65 W 1/7, 0x66 W 0/4, 0x47 2/3, 0x37 0/1). The spot list walks from
`SpawnCycle` (`$524`), which every spawn advances. Training draws all 9 at every reset, so a room's hits are much more
deterministic in chains than in training.

**5:0x37 S→N (darknut), 180 visits.** The room cost 1.20 hearts a visit (0 / ≤ 1 / > 1 lost: 0 / 77 / 23%) with 24
deaths. Every visit was hit within 50 frames, at X `$78` Y `$95`-`$9F` on the ladder (214 of 221 hits on the ladder),
moving N with no swings. The cause is mechanical. The objective's path search takes the straight column X `$78`
(breadth-first, ties N first; 2 tiles shorter than X `$70`), and that column walks into the Red Darknut that the way
back's spawn cycles 0/1 place in the middle pocket (block row 3). Scripted crossings from the 180 chain entries, in
hearts lost a visit:

| path | frame level | action level (the executor) |
|---|---|---|
| X `$78`, walk | 1.20, 24 deaths (= the models) | 1.21, 24 deaths |
| X `$78`, swing | 0.82 | |
| X `$80`, walk / swing | 0.75 / 0.45 | |
| X `$70`, walk | 0.70 | 0.48, 7 deaths |
| **X `$70`, swing (frame) / SWORD N (action) at a Darknut ahead** | **0.00, 180 of 180 clean, 121 frames** | **0.03, 175 of 180 clean, 1 death** |

The models take 268 frames.

**The fix in data.** A new `game.yaml` room key, `path_costs`, puts costs on Link-tile rectangles: a path-search step
onto them costs 1 + N. The wavefront becomes a Dijkstra, and the target observation's descent counts the costs.
0x37 charges tile columns 15-16 in rows 6-15, so the objective, PBRS and target cross at X `$70`, the owner's column.

**Zero-shot it does nothing.** The promoted darknut ignores the moved target. It still crosses at X `$78` and loses
1.20 hearts a visit there, and reaches the D5 triforce in 138/300 runs against 143/300 before.

**The training arm never ran.** Six hand-made 0x37 tracks (X `$70` and SWORD N at a Darknut ahead; made by a
throwaway script, not committed) replaced the explorer's `5_37s` tracks for a darknut continuation, `r75-bd5-dk`
(`--load darknut --iterations 244 --seed 75 --stop-success 1.01`). These are the same training settings as the
baseline of the exit-value reward experiment, `r75-v1-base` (reported separately). The run ended with an error (a
broken pipe) before its first iteration and left no checkpoint. The control that was measured: `r75-v1-base` itself,
under the path-cost code, on the same seeds 0-299, reached the D5 triforce in 102/300 = 34.0%, against the promoted
`r74-darknut5`'s 143/300. A continuation arm therefore has to beat the promoted model, not only that baseline (§7 has
the cause of the drop). Not done: the arm, a committed track maker, and a 1000-run paired chain clips on and off.

**The other rooms are real fights** (no mechanical cause found):

- **0x47 S→N (gibdo), 202 visits:** 0.84 a visit, 22 deaths. 58 of 91 first hits come within 50 frames, right after
  the ladder: the model stops after the crossing, and a Gibdo of the cycle-2/3 spawn walks into it.
  - Frame-level scripts: straight walk 1.41 (30 deaths), straight with a swing 0.27 (7 deaths, 124 frames).
  - Action level the swing line does not carry: SWORD N 1.50, flash-keeping SWORD 1.81, BOOMERANG at a Gibdo ahead
    1.25, walk 1.84. An action's SWORD stops Link 13+ frames on the floor, and the decision cadence misses the end of
    the flash.
  - The model's 0.84 beats every action-level script. Left to training.
- **0x57 W→N (zol), 205 visits:** 0.08 a visit; the models already cross on row 3 with the ladder.
- **0x64 cave→E (darknut, way back), 274 visits:** 1.50 a visit, 24 deaths, from Blue Darknuts at 50-150 frames.
  Scripted S round the diamond: walk 1.29 (13 deaths), swing 1.44 (34 deaths). A fight.
- **0x65 E→W (gibdo), 300 visits:** 0.74 a visit. Bombs were taken westbound in 40 of 300 visits, only 3 of them
  with a hit: they are touches of a flashing or dead carrier, not the losses. Runs that take them end no worse
  (`r74b-1k` D5 46.8 against 46.3%; `post-d4` 16.0 against 15.1%; Olive Garden won 89 against 91%). Scripted lines
  are far worse (walk 3.08, swing 2.09). Nothing to gain.
- **0x65 W→E (gibdo), 250 visits:** 0.43 a visit, bombs taken in 217 of 250, 10 of those with a hit. A scripted line
  without the take costs 0.46. Nothing to fix.
- **0x66 W→N (gibdo), 245 visits:** 1.11 a visit, 37 deaths, 22.6 swings a visit, hits spread over 100-1000 frames:
  the forced fight.
- **The Olive Garden, 299 visits:** 5.34 hearts a visit gross, all from Blue Darknut touches during the fight. There
  are no entry hits (3 of 801 hits in the first 150 frames), none at the push or the stairs; 15% of hits come with
  Link in the NW corner.

## 6. Rooms that bleed, overworld: spawn spots and 0:0x27

Branch `r25-bleed-ow`. Diagnosis chains were 300 runs each, clips on, with replays. Every frame where Link's health
fell was logged with his position, the action and the nearest objects.

**Spawn spots.** Each entry direction has nine fixed spawn spots (asm:Spawn_PosListFacing*), and unsafe ones are
skipped. Several of them sit right on the route's entry lines:

- (`$80`,`$8D`) straight ahead of an S entry: 0x2d, and in dungeons right in front of the S door (3:0x49's Keese);
- (`$70`,`$8D`) ahead of an N entry: 0x38's Octorok at the corridor junction, 3:0x69;
- (`$80`,`$7D`) and (`$50`,`$5D`) for a W entry: 0x1f.

A hit counts as "at spawn" when the hitting object is within 24 px of Link and within 16 px of where its slot spawned.
That covers 540 of the 1601 hearts lost on the two segments:

| room | 0:0x27 | 0:0x1d | 0:0x2d S→N | 0:0x1f | 0:0x38 |
|---|---:|---:|---:|---:|---:|
| share of hearts lost at spawn | 99% | 93% | 61% | 60% | 47% |

**0:0x27 N→E (ow-zora).** The room cost 0.50 hearts on every visit: 246 of 246 visits took one hit, from a Peahat.
Link enters facing down, and only three spots of the facing-down list are safe in 0x27 ((160,125), (80,125),
(112,141)), with two Peahats on each (asm:Spawn_AssignPositions). There is no spawn cloud; they rise slowly. Link
enters from 0x17 at X `$A0` and walks straight down that column into the Peahat at (160, ~117): 214 of 246 hits
came within 50 frames of entry, at Y 96-111. Scripted walks from `0_27n_0` (60 seeds):

- straight down X `$A0`, then E: 0.54 hearts, always hit;
- down the corridor's middle (X `$88`) to Y `$8D`, then E along the strip's bottom: 0.12 (hit in 17-25%), in about
  the same time.

The cause is the path the objective draws. **The fix** (`03cd333`) uses §5's `path_costs`: cost 3 on columns 14-15
and 18-23, rows 0-9 (the corridor's sides above the strip). Measured with ow-zora `r74-leg` unchanged, with no
training:

- Single room, 100 episodes × 3 states: −0.337 → −0.143 hearts. The states entering at X `$98`/`$A0` like the chain
  went −0.50 → −0.15; the X `$70` state went −0.01 → −0.13. A variant with columns 19-21 only was worse (−0.71).
- Post-d2 chain, 300 runs paired: 0x27 0.50 → 0.19 hearts a visit, 221 → 328 frames, deaths there 6 → 2, arrivals at
  D1 95.7 → 97.3%.

Confirmed at 1000 runs, post-d2 to the D1 triforce, paired by seed with `r74-1k` and `r74-1k-noclips` (the runs are
bit-identical up to 0x27):

| | 0x27 hearts lost a visit | frames a visit | deaths in 0x27 | reach D1's entrance | hearts there | reach the D1 triforce |
|---|---|---|---|---|---|---|
| clips on, round 24 | 0.50 | 223 | 16 | 952 | 3.16 | 805 |
| **clips on, path costs** | **0.21** | 328 | 8 | 963 | **3.45** | **829** |
| clips off, round 24 | 0.50 | 223 | 17 | 923 | 2.76 | 718 |
| clips off, path costs | 0.22 | 328 | 8 | 932 | 3.06 | 758 |

Link reaches Dungeon 1 with about 0.3 hearts more, clips on and off. The D1 triforce is taken 2.4 points more often
clips on (80.5 → 82.9%) and 4.0 points more clips off (71.8 → 75.8%). As independent samples each rate has a standard
error of about 1.8 points. The runs are paired and diverge only from 0x27 on, so the true error is smaller, but it
was not computed. The cost is 105 frames a visit, under 2 seconds. `03cd333` is not on `r25` yet; it needs the
`path_costs` code, which reached `r25` with §5's merge.

**Tweaks that failed zero-shot and were reverted:**

- **0:0x1a's W exit with `wrap_in: [9, 10]`**, so the path search jumps from the E wrap zone to the W landing (today
  18% of visits clip E there: 0.09 hearts and 285 frames, against 0.42 hearts and 453 frames for the 82% that walk W
  through six red Tektites). Post-d2 screen, 300 runs paired: with the target pointing E, 61 of 296 runs walked out of
  0x1a's open E edge into 0x1b (a wrong exit, back and forth), and arrivals fell 95.7 → 92.0%. The wrap zone is next
  to the E exit on those rows, and a model trained without the jump walks out instead of clipping.
- **0:0x1f** (cost 2 on rows 2-6, columns 0-14, to keep off the top strip):
  - single room −0.80 → −0.59 hearts, but 2.3% wrong exits;
  - leg chain, 300 runs paired: 0x1f 0.84 → 0.44 hearts a visit, but 52 of 300 runs walked into 0x1f's money-making
    cave (untagged, so the run fails), and arrivals fell 100 → 82.7%. The model, trained on the top strip, stalls at
    X `$60` on the middle row and steps N into the cave's mouth;
  - variants (rows 2-9: −0.69 with 2.3% wrong exits; plus cost 6 on columns 10-15, rows 7-9: −0.71 with 3.7%) were
    worse.
- **0:0x2d S→N** (cost 3 on columns 15-17, rows 8-12, to keep 16 px off (`$80`,`$8D`)): single room −0.417 → −0.400.
  On the leg screen (300 paired) it was worse: 0.56 → 0.62 hearts a visit, 252 → 324 frames, and 9 runs strayed W into
  0x2c.
- **0:0x37 cave→E** (cost 3 on columns 19-21, rows 7-9; the one Octorok spawns at (`$A0`,`$7D`), and its cloud ends
  as Link passes under it on Y `$85`, 143 of 143 hits at frames 40-50): single room −0.238 → −0.253. The ow-octorok
  model does not follow the moved target.

The conclusion: only ow-zora in 0x27 follows a moved target without training. Elsewhere, path costs need the family
retrained with them.

**Other diagnoses:**

- **0:0x38 N→W:** Octoroks spawn at (112,141) and (160,125) under the entry column; 29% of visits lose half a heart
  walking W along Y `$85` at frames 50-100.
- **0:0x0a S→cave (ow-lynel):** 0.27-0.34 a visit, from Zora fireballs and the Lynel while Link dithers on the E side
  (median 683 frames): behaviour, not a mechanism.
- **Dungeons:**
  - 3:0x49 S→S: 62 of 66 hits within 50 frames, from a Keese spawned at (`$80`,`$8D`) straight ahead of the S door.
  - 3:0x69 N→cave, 0.62: Darknuts spawned at (`$A0`,`$7D`) and (`$70`,`$8D`) walk into Link going S at X `$90`; 48% of
    visits have no hit.
  - 3:0x5a E→W, 0.26: blue Keese (106 hits) and traps (24) near the W door.
  - Fights: 3:0x59 N→S (1.43), 4:0x32 (1.00), 4:0x31 (0.69), 4:0x50 (0.34), 4:0x12 (0.43).

**The Dungeon 4 ladder clip (4:0x21 S→N, 0.22 a visit).** 62 of 132 hits come during CLIP N. The Gels chase Link and
gather on the landing strip (Y `$5D`) above him, and the ladder clip lands in them. The boomerang kills Gels (HP 0)
before the clip; this was noted, not tried.

**Open at the handover** (this work continues as its own experiment):

- The training arm `r75-bow-ow` (branch `r25-bleedow-tr`) finished:
  - ow-tektite with 0x1a's `wrap_in`, at iteration 766, 99% training success, 0x1a 100%;
  - ow-peahat with 0x1f's path costs, 94% training success, its weakest room 0x1d 92%.
- Their paired 300-run screens (post-d2 for 0x1a, the leg for 0x1f) were playing. Neither run is to be promoted until
  they show arrivals held and no new wrong exits (0x1b from 0x1a, 0x1f's cave).
- The suites had not been run on `r25-bleed-ow`.
- The other rooms of the diagnosis are left to training.

## 7. Checkpoint noise (preliminary)

First result; the selection tool and the fresh-seed protocol are reported when that work ends.

The question was whether a darknut checkpoint can be picked by its Dungeon 5 arrivals. The test: a darknut continued
from `r74-darknut5` with `--sil-coeff 0.1` for 977 iterations, saving 17 checkpoints, one every 61 iterations. Each
checkpoint played the D5 segment, clips on, 1000 runs, seeds 0-999, with the promoted team otherwise.

- **No checkpoint differs beyond noise.** Arrivals ranged 42.4-47.7%, mean 45.2, SD 1.6, equal to the binomial
  standard error of 1.6. The trend was −1.7 points per 1000 iterations, not significant.
- The promoted `r74-darknut5` re-measured 46.4%, identical to round 24's `r74b-1k`.
- **On fresh seeds 1000-1999:** the best-screened checkpoint (iteration 9831) 47.4%, the promoted 47.0%, the last
  43.7%.
- Paired differences between two models on the same 1000 seeds have a standard error of 2.2 points: pairing gains
  nothing here.
- Training success, value loss and entropy do not correlate with D5 arrivals across checkpoints.

**The consequence.** A 34-35% darknut is far outside this range, so a continuation that fell to 34% lost it to the
missing SIL term, not to checkpoint luck. The baseline trained without `--sil-coeff 0.1` (`r75-v1-base`, §5) measured
102 of 300 = 34.0% on seeds 0-299, where the promoted model measured 143 of 300 = 47.7%.

## 8. Job coordination (tooling, landed on `main` during the round)

Trainings and evaluations from many worktrees piled up on one machine (a load average of 120). Two commits on `main`,
both merged into `r25`, settled it:

- **The first design** (`a35a849`; decisions 2026-11-15 "Trainings and evaluations wait their turn"): at most
  `$TRIFORCE_MAX_JOBS` (default 2) trainings or evaluations run at once; the rest wait first come, first served, and
  say so. There is no controller. The per-user job registry is the queue, each job checks its turn under a `flock`,
  and liveness comes from `/proc`, so a crashed job stops counting by itself. Every training and evaluation command
  waits and lowers its own priority (nice 15). The only opt-outs are `--no-wait` and `--nice`. Play, replay and build
  tools never wait.
- **The scored line** (`87875eb`; decisions 2026-11-15 "The line drains the fastest jobs first"). First come let long
  jobs hog the line: a three-minute chain waited hours behind two rounds. Each job now estimates its run time from its
  work (training steps, evaluation episodes, chain runs × route steps) at rates learned from the registry. Waiting
  jobs start in order of `estimate − 0.5 × waited`, the owner's rule: short jobs go first, and a long job passes fresh
  short ones after waiting about twice its own length, so it always starts.
- **The express place.** A job estimated at 10 minutes or less takes one extra place, so it does not wait an hour
  behind two trainings.
- **The fallback.** While any job from the earlier code (no estimate) waits, everyone keeps the first-come rule, so
  old and new jobs cannot wait for each other in a cycle.
