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
reaches low health less often. Twenty runs are a lead, not a result. The guard is carried on as its own experiment
(§12).

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

**Follow-up: the column by behaviour cloning** (branch `r25-d537`). The final darknut retrain `r75-f-dk-s75` (the six
tracks, `--sil-coeff 0.1`, the heart-value reward) did not change 0x37 either: 1.19 hearts a visit, 91 deaths per 1000
D5 runs (`B-s75-d5`). *Cause:* in 0x37 alone (3 states × 60 seeds, temperature 0.5) the model crosses at X `$78` in
180 of 180 episodes, and the pocket's Darknut is there in 87 of them (training draws every spawn cycle; the chain meets
it always): hit in all 87, 1.29 hearts a pocket visit, 0.00 in the others. Past the door its probability of any MOVE W
is 0.0000 (max 0.001) at temperature 1: the policy never tries the column. PBRS doesn't push it (with the path cost,
MOVE N up X `$78` earns +0.05 a tile like MOVE W; only the `target` direction moved), the heart-value potential
prices a hit near 0 above 6 hearts (V(0x37, h) ≥ 0.99), and the tracks complete at once (7 stages × 10 episodes;
success is the exit, hit or not). Arms, all continued from `r75-f-dk-s75` (seed 75), in 0x37 alone (hearts a pocket
visit) and the family evaluation (100 episodes per state, clips both; s75 under this code: 0x05 −1.41, 0x64 −1.40):

| arm | what | 0x37 pocket | other rooms |
|---|---|---|---|
| a | 40 it, `start_states` 0x37 ×3 its share, the six tracks only, `--ent-coeff 0.05` (new flag) | 1.29, P(W) 0.0002 | – |
| b | a + `heart_value.potential` 100 + curriculum fraction 1.0 | 1.29, P(W) 0.0003 | – |
| c | 60 it, `train --demos` (new: the rule's clean runs, 17,423 rows, `--demo-coeff 0.2`) | 0.38 (X `$70` 180/180, never swings: swings are 1% of rows) | 0x64 −1.74 |
| c2 | c, swing rows ×40 | 0.10 | 0x05 −1.65, 0x64 −1.66 |
| ctl | 60 it, no demos | = s75 | 0x05 −1.38, 0x64 −1.44 |
| c3 | c2, every duration free | X `$50` overshoots (MOVE W 32), 0.76 a visit | 0x05 −1.45, 0x64 −1.42 |
| c4 | c2, MOVE N durations free unless a monster is within 48 px (`demos.ANY_DURATION`) | 0.14 | 0x05 −1.55, 0x64 −1.54 |
| **c5** | c4 at `--demo-coeff 0.05` | **0.13** | **0x05 −1.38, 0x64 −1.46; all −0.49 vs −0.52** |
| c6 | c5 + one DAgger round (c5 plays, the rule labels) | 0.29 | – |

The cloned rule's MOVE N 4 rows were what spread to the Blue Darknut rooms (c2: MOVE N 4 from 29% to 39% of moves
in 0x64/0x05). D5 chain, 1000 runs, seeds 0-999, clips on, paired with `B-s75-d5`: c2 52.5% (+0.2 ± 2.2; Garden 126
deaths against 76), c4 55.2% (+2.9 ± 2.2; 0x37 0.15 a visit, 10 deaths; Garden 105), **c5 55.5% (+3.2 ± 2.2; 0x37
0.20 a visit, 80% clean, 28 deaths against 1.19, 0%, 91; Garden 79, 0x64 back 45 deaths as before)**, ctl 49.7%
(c5 − ctl +5.8 ± 2.1). The remaining 0x37 hits are swings the clone misses where its own path leaves the rule's (the
rule is hit in 1 of 300 single-room episodes, c5 in 27). Recipe: `scripts/experiments/r25_d5_37_tracks.py --demos`.
Fresh seeds 1000-1999, paired with `fresh-B75-on|off-d5`: clips on 59.9% against 50.7% (+9.2 ± 2.1; 0x37 0.18 a
visit, 18 deaths against 1.18, 89; the Garden 82 deaths against 54), clips off 46.8% against 41.6% (+5.2 ± 2.2; 0x37
0.14, 8 deaths against 0.95, 54; the plain 0x64 fight 159 deaths against 123). The Blue Darknut rooms are the thing
to watch when this recipe is used again. A c7 at `--demo-coeff 0.1` sat between c4 and c5 in 0x37 (0.18 a pocket
visit; chain 0.17 a visit, 14 deaths) but lost the Garden again (105 deaths, single-room 0x64 −1.65): 54.2%
(+1.9 ± 2.1), below c5.

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

**Spawn spots.** Each entry direction has nine fixed spawn spots (asm:Spawn_AssignPositions; the four lists are
asm:Spawn_PosListFacingRight, asm:Spawn_PosListFacingLeft, asm:Spawn_PosListFacingDown and
asm:Spawn_PosListFacingUp), and unsafe ones are
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
was not computed. The cost is 105 frames a visit, under 2 seconds. `03cd333` reached `r25` with the merge of
`r25-bleed-ow` (`09b3136`), after the `path_costs` code had come with §5's merge.

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

## 7. Checkpoint noise, and choosing a checkpoint by the chain

Branch `r25-ckpt`, squashed into `r25` as `3d6bfaa`. Decisions 2026-11-15 "Checkpoints are chosen by the chain and
confirmed on fresh seeds"; `docs/architecture.md` "Numbered checkpoints" and `triforce select-checkpoint`.

The question was whether a checkpoint can be picked by its arrivals on a chain segment. Single-room evaluations
cannot see what the route needs: in round 24, `r74-darknut4` and `r74-darknut5` both scored 97% in single rooms,
yet 34.9% and 46.4% on the Dungeon 5 segment. And the best of N noisy chain measurements is biased upward, so
whatever is chosen has to be measured again on fresh seeds.

**The tool:**

- **Numbered checkpoints:** `train --checkpoint-every N [--checkpoint-keep K]`, passed through by `train_round.py`.
  It saves `<run dir>/checkpoints/it<iteration>/`: a byte copy of `ckpt.pt` as saved at that iteration, with its own
  `version` and an `act` bound to the copy's sha1. A numbered checkpoint therefore loads anywhere a run dir does.
  The option changes nothing else. A test shows it: the same seed on the CPU gives identical weights, optimizer state
  and metrics with and without it.
- **Selection:** `triforce select-checkpoint FAMILY RUN --segment NAME --runs 1000 --confirm 1000 [--confirm-top K]
  [--clips off] [--only …] [--apply]` runs as one batch job.
  - Every numbered checkpoint, the run's last (`ckpt.pt`) and the family's promoted run play the same chain seeds,
    with the rest of the team as promoted.
  - They are ranked by arrivals, then hearts on arrival, with paired differences against the last and the promoted.
  - The winner, the next best K − 1, the last and the promoted then play again on fresh seeds. That confirmation is
    the number to report.
  - It writes `selection.json` in the run dir and chooses the winner only if its confirmed paired difference against
    the run's last checkpoint is above zero. It does not test against the promoted run.
  - `--apply` makes the chosen checkpoint the run's `ckpt.pt`. The old file and its evaluations move to their own
    numbered folder, and the run then needs a fresh canonical evaluation.

**The darknut curve.** The run `r75-ck-darknut` continued `r74-darknut5` with `--sil-coeff 0.1` and `--parallel 8`
for 977 iterations (iterations 9771-10747, 4.0M steps) and kept a checkpoint every 61 iterations. Its canonical
evaluation scored 97% over 9000 episodes, the worst room 5:0x05 at 91%. Every checkpoint played the D5 segment, clips
on, 1000 runs on seeds 0-999, with the promoted team otherwise. Each D5 figure has a standard error of ±1.6.

- "Garden alone" is the Olive Garden by itself with 2 bombs, hp drawn as in training, 300 episodes.
- "Hearts out of the Garden" is the health Link carries into the next room (route step 194's `health_in`) in the
  chain.
- The training metrics are averaged over the 30 iterations before each checkpoint.

| checkpoint | D5 % | hearts out of the Garden | Garden alone: success % / hearts | train success | value loss | entropy |
|---|---:|---:|---|---:|---:|---:|
| promoted `r74-darknut5` | 46.4 | 6.43 | 81.3 / −3.04 | – | – | – |
| it009831 (+61) | **47.7** | 6.35 | 80.0 / −2.86 | .968 | 1.90 | .800 |
| it009892 | 43.1 | 6.18 | 79.7 / −3.11 | .971 | 1.59 | .785 |
| it009953 | 45.3 | 6.21 | 83.3 / −2.98 | .973 | 1.55 | .799 |
| it010014 | 47.2 | 6.21 | 82.7 / −2.93 | .970 | 1.55 | .788 |
| it010075 | 46.1 | 6.31 | 79.0 / −3.23 | .974 | 1.54 | .809 |
| it010136 | 44.9 | 5.96 | 81.0 / −3.33 | .971 | 1.60 | .796 |
| it010197 | 45.5 | 6.29 | 82.3 / −3.06 | .974 | 1.53 | .838 |
| it010258 | 45.7 | 6.36 | 83.3 / −2.81 | .969 | 1.61 | .808 |
| it010319 | 46.3 | 6.38 | 83.7 / −2.79 | .972 | 1.45 | .767 |
| it010380 | 46.6 | 6.36 | 83.7 / −2.79 | .982 | 1.29 | .754 |
| it010441 | 44.3 | 6.11 | 83.3 / −3.00 | .975 | 1.37 | .759 |
| it010502 | 43.4 | 6.07 | 83.0 / −3.05 | .972 | 1.82 | .754 |
| it010563 | 43.2 | 5.93 | 82.7 / −3.11 | .976 | 1.47 | .783 |
| it010624 | 42.4 | 5.92 | 81.3 / −3.12 | .972 | 1.56 | .836 |
| it010685 | 44.1 | 6.03 | 80.7 / −3.08 | .971 | 1.54 | .821 |
| it010746 | 47.1 | 6.09 | 82.3 / −3.00 | .976 | 1.33 | .820 |
| last (10747) | 45.1 | 6.07 | 78.7 / −3.18 | .977 | 1.32 | .819 |

The promoted run's 46.4% is round 24's `r74b-1k` figure exactly (464 of 1000).

**Fresh seeds 1000-1999** (1000 runs each; the screen's first, second and third, the last and the promoted). Clips
off played only these five, on the same fresh seeds:

| checkpoint | screen, seeds 0-999 (on) | confirmed, clips on | confirmed, clips off |
|---|---:|---:|---:|
| it009831 | 47.7 | 47.4 | 36.9 ± 1.5 |
| it010014 | 47.2 | 45.3 | 36.4 |
| it010746 | 47.1 | 42.4 | 34.1 |
| last | 45.1 | 43.7 | 35.6 |
| promoted | 46.4 | 47.0 | 34.3 |

The screen's second and third fell on fresh seeds (47.2 → 45.3, 47.1 → 42.4). This is the upward bias of the best of
N, and it is why the confirmation is the number reported. Paired differences on the fresh seeds:

- it009831 − promoted: +0.4 ± 2.2 clips on, +2.6 ± 2.1 clips off;
- it009831 − last: +3.7 ± 2.2. The tool's rule therefore chose it009831;
- promoted − last: +3.3 ± 2.2. The run's own last checkpoint was the worst of the three.

The promoted run measured 34.0% clips off on seeds 0-999 in round 24 (`r74b-1k-noclips`).

**The noise floor:**

- One model on 1000 runs has a binomial standard error of 1.6 points. The 17 checkpoints spread 42.4-47.7% (mean
  45.2) with an SD of 1.6, so no checkpoint differs from another beyond noise. The trend over the run, −1.7 points per
  1000 iterations, is not significant (residual SD 1.54).
- The paired difference of two models on the same 1000 seeds has a standard error of 2.2 points. The same seeds do not
  make two different models' runs alike, so pairing gains nothing here.

**What makes a checkpoint good.** One thing varies beyond noise: the hearts Link carries out of the Olive Garden.
Across checkpoints it has an SD of 0.165 against a standard error of 0.084. It slides from 6.43 (the promoted run) to
5.92-6.11 over the last seven checkpoints. Correlations across the 17 checkpoints:

| signal | correlation with D5 arrivals |
|---|---:|
| hearts out of the Garden (in the chain) | 0.69 |
| hearts into 0x64 on the way back (the same runs, partly mechanical) | 0.72 |
| the Garden alone: hearts a fight | 0.51 (detrended 0.49) |
| the Garden alone: success | 0.09 |
| the way back at low health (`5_64c` and `5_37s`, hp 0.15-0.5, 300 episodes): success / hearts | −0.40 / −0.12 |
| training: success / value loss / entropy / hearts | 0.04 / −0.07 / 0.0 / 0.30 |

The chain's Garden hearts also correlate 0.62 with the Garden-alone hearts. The way-back evaluations sat at 91.0-93.7%
and −0.74 to −0.89 hearts for every checkpoint. The training signals say nothing. A darknut trained longer loses Garden
hearts while its training metrics stay flat; this went to the backlog as a new item.

**Verdict.** No checkpoint of `r75-ck-darknut` is promoted: it009831 is at best equal to the promoted run. Darknut
stayed `r74-darknut5` until the final retrain (§11). Selection was to be run on gibdo next only after a confirmed
gain on darknut, so it was not run on gibdo.

**What it explains.** A 34-35% darknut is far outside this range. So the continuations that fell to 34-36% lost it to
the missing SIL term, not to checkpoint luck. The value-reward experiment's first screen ran without `--sil-coeff
0.1` (§9), and that run's baseline, `r75-v1-base`, measured 102 of 300 = 34.0% on seeds 0-299 (§5) where the promoted
model measured 143 of 300 = 47.7%. The value-reward experiment confirmed that it had run without SIL, on otherwise the
same code.

**The measurement rules it leaves** (decisions 2026-11-15; applied in §9 and §11):

- **Runs per arm:** 1000 runs per arm per chain, on seeds not used to choose the model. One arm against another needs
  about 5 points to be a result. 3 points need about 2200 runs an arm and 2 points about 5000. A 300-run screen sees
  only effects of 8 points or more.
- **Seeds per arm:** inside one continuation the checkpoint barely matters, but separate trainings can differ by 10
  points or more (darknut4 against darknut5; with and without SIL). So an arm gets 2-3 training seeds, each judged by
  its last checkpoint at 1000 runs, and darknut baselines keep SIL on.
- **A darknut secondary:** for darknut changes, also report the hearts out of the Olive Garden (step 194's
  `health_in`, standard error 0.08 on 1000 runs). It moves where arrivals cannot show it.

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

## 9. Hearts priced by what they are worth to the run (`reward.heart_value`)

Branch `r25-value` (`c2aa3eb` mechanism, `423738b` tests and docs; merged into `r25` at `fda107f`); the arm's yaml on
branch `r25-value-ab`. Decisions 2026-11-15 "Hearts priced by what they are worth to the run"; `docs/architecture.md`
on `reward.heart_value`.

**Why.** The reward paid about 2 per heart anywhere: 20 × w × h / max at the exit, and 0.25 per half heart on a hit or
pickup. A heart's worth to the run is steep and local (§1). At Gleeok V is 0.22 / 0.59 / 0.86 / 0.95 at 1 / 2 / 3 / 4
hearts, and at 5:0x37 it is 0.00 / 0.58 / 0.81 / 0.93. It is flat above 3 hearts at Manhandla and after any boss,
where the triforce refills Link. A potential term also prices the heart at the hit, not 100 steps later at a
discounted exit.

**The table.** `triforce/heart_value.json` sits beside `routes.yaml`. It holds V(step, hearts): the chance that a run
entering a route step's room with that many hearts is alive when it takes the next triforce. It has an entry for
every route step at every half heart, per way (`clips` for clips on, `plain` for off). `python -m triforce.heart_value
<chain jsonl>...` fits it from chain runs. The shipped table is fitted from round 24's chains (`r74-1k` clips on and
off, the full route with the segments, and `r74b-1k-d5`).

- Each arrival bin's own transitions are mixed with 20 pseudo-visits of a prior, a Kaplan-Meier estimate over the
  step's visits that assumes a room's damage does not depend on the hearts brought.
- The fit runs backwards from each triforce and is made non-decreasing in hearts.

It agrees with §1's independently fitted table within 0.01-0.04: Gleeok 0.22 / 0.59 / 0.86 / 0.95 against 0.23 / 0.60
/ 0.87 / 0.95, and 5:0x37 0.00 / 0.58 / 0.81 / 0.93 against 0.04 / 0.55 / 0.82 / 0.94. In Dungeon 5, V jumps at full
health because full health means sword beams: at the D5 entrance (step 186) it is 0.30 at 9 hearts and 0.43 at 10.

The table is data, not a model, and is to be refit after a round that changes the team or the route. It names each
step's room, entry, exit and visit, and a family that reads it refuses to load when they no longer match the route.

**The three terms** (`reward: {heart_value: {exit, potential, low_health}}`; all off by default; off is today's
reward bit for bit, tested):

- **`exit: true`:** the success terminal's health part becomes V(next step, hearts out) / V(next step, max hearts)
  instead of hp / max, still weighted by w.
- **`potential: S`:** every change of hearts in a room (a hit or a pickup, not a refill or a heart container) pays S ×
  (V(step, after) − V(step, before)) at the room's route step. It is paid beside the flat hit and pickup terms, added
  after the step's ±1 clamp (so a heart can be priced above 1), and scaled by `shaping_scale`.
- **`low_health: F`:** adds F × (1 − hearts after / max) × the step's hit penalty.

**The first screen is void.** Darknut and gibdo were continued 244 iterations (1M steps) from their promoted runs,
seed 75, `--parallel 5`: a base arm (`r75-v1-base`, heart value off) and an arm with `{exit: true, potential: 20}`
(`r75-v1-ab`). Both ran with `--sil-coeff 0` by mistake. Round 24 trained with `--sil-coeff 0.1`, and the default
of `train.py` is 0. Darknut-only D5 chains, 300 runs paired, clips on:

| darknut | arrive | Garden deaths a visit | hearts into 0x64 back | 0x66 W→N | 0x37 | darknut hearts lost a visit | frames a visit | WAIT |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| promoted `r74-darknut5` | 47.7 ± 2.9 | 0.084 | 6.06 | 4.66 | 4.03 | 0.86 | 524 | 1.3% |
| `r75-v1-base` | 35.3 ± 2.8 | 0.137 | 5.56 | 4.43 | 3.94 | 0.96 | 536 | 0.7% |
| `r75-v1-ab` | 36.0 ± 2.8 | 0.094 | 5.91 | 4.62 | 3.94 | 0.88 | 557 | 1.5% |

The promoted model re-measured on this code gave 143/300, identical to round 24's seeds 0-299. The two arms do not
differ in arrivals. The priced arm kept 0.35 hearts more out of the Garden and died there less (0.094 against 0.137 a
visit), but the later rooms ate it. Both arms lost 12 points against the promoted model, and that is why the screen is
void: the missing SIL term, not the reward (§7). On different code, §5's control measured `r75-v1-base` at 102/300 =
34.0% on the same seeds. The gibdo runs of this screen were trained, not chained. The rule since then: continuations
of a round 24 model pass `--sil-coeff 0.1`.

**The darknut result.** Two arms, darknut only, continued from `r74-darknut5` for 244 iterations with `--sil-coeff
0.1`, `--parallel 8`, seeds 75 / 76 / 77 per arm:

- plain: `r75-v2-base-s75/76/77`, heart value off;
- priced: `r75-v2-ab-s75/76/77`, `{exit: true, potential: 20}`.

Each checkpoint ran the D5 segment, 1000 runs, seeds 0-999, clips on, with only darknut swapped and the act mode
`temperature:0.5+relax:16`. The other columns are hearts on entering each way-back room, darknut's hearts lost and
deaths a visit, frames a visit, and the share of WAIT decisions. All runs entered the Garden with 9.22 hearts.

| darknut | D5 arrival | vs promoted (paired) | hearts in: 0x64 back | 0x65 | 0x66 N | 0x47 | 0x37 | Digdogger | darknut hearts lost / deaths a visit | frames a visit | WAIT |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| promoted `r74-darknut5` | 46.4 ± 1.6 | | 6.33 | 5.17 | 4.89 | 4.47 | 4.04 | 3.35 | 0.85 / 0.040 | 522 | 1.2% |
| priced, seed 75 | 50.9 ± 1.6 | +4.5 ± 2.2 | 6.43 | 5.52 | 5.20 | 4.79 | 4.37 | 3.57 | 0.80 / 0.037 | 523 | 1.4% |
| priced, seed 76 | 49.9 | +3.5 ± 2.2 | 6.31 | 5.40 | 5.02 | 4.56 | 4.21 | 3.44 | 0.80 / 0.035 | 532 | 1.9% |
| **priced, seed 77** | **51.3** | **+4.9 ± 2.2** | 6.59 | 5.68 | 5.41 | 4.82 | 4.49 | 3.79 | 0.76 / 0.036 | 529 | 0.8% |
| plain, seed 75 | 44.3 ± 1.6 | −2.1 ± 2.2 | 6.06 | 5.26 | 4.91 | 4.50 | 4.14 | 3.58 | 0.86 / 0.045 | 511 | 1.3% |
| plain, seed 76 | 48.5 | +2.1 ± 2.2 | 6.48 | 5.40 | 5.05 | 4.59 | 4.27 | 3.44 | 0.79 / 0.032 | 524 | 1.7% |
| plain, seed 77 | 41.5 | −4.9 ± 2.1 | 5.96 | 4.96 | 4.69 | 4.32 | 3.90 | 3.22 | 0.90 / 0.044 | 530 | 2.5% |

- **Arm means:** priced 50.7% (49.9-51.3, SD over seeds 0.7), plain 44.8% (41.5-48.5, SD over seeds 3.5).
- **Priced minus plain, per seed:** +6.6, +1.4 and +9.8 points. The mean of +5.9 has a standard error of about 2.4
  across seeds and is larger than the plain arm's seed spread, so it passes the rule set for the round (§7).
- **Steadier as well as better:** the priced arm's three seeds lie within 1.4 points of each other, the plain arm's
  within 7.
- **Against the promoted model:** every priced seed beats it, pooled +4.3 points over 3000 paired runs, about 3
  standard errors.
- **Where it comes from:** Link enters every room of the way back with about 0.3 hearts more (0x65, 0x66, 0x47, 0x37),
  and darknut loses fewer hearts a visit (0.76-0.80 against 0.85).
- **It does not play slower or more passively:** frames a visit and the WAIT share are within the plain arm's range.

The canonical single-room evaluation (9000 episodes) agrees. The priced runs keep 0.03-0.05 hearts more per room at
equal or better success:

| run | success | mean hearts change |
|---|---:|---:|
| `r74-darknut5` (promoted) | 97.29% | −0.548 |
| `r75-v1-base` (no SIL) | 96.93% | −0.610 |
| `r75-v1-ab` (no SIL) | 97.12% | −0.581 |
| `r75-v2-base-s75` | 97.32% | −0.558 |
| `r75-v2-ab-s75` | 97.52% | −0.529 |
| `r75-v2-ab-s76` | 97.50% | −0.511 |

**What was kept.** `{exit: true, potential: 20}` became the setting of every dungeon fight family (`fd7cc6e`: darknut,
gibdo, goriya, vire, stalfos-keese-gel, moldorm, zol, like-like, d2-rope, trap-room, equipment-room), and the final
retrain trained them with it (§11). The candidate `r75-v2-ab-s77` was not promoted itself: the final retrain's darknut
replaced it.

**Built and not run:**

- `exit` alone (branch `r25-value-a`) and `low_health: 3` (branch `r25-value-c`). The plan ran them only if the pair
  showed an effect. It did, but neither was started, so which term carries the gain is not known.
- Another scale for the potential (10 or 40).
- The premium the table puts on Dungeon 5's last heart (beams) on its own.
- Clips-off chains of these arms, and the gibdo arm. Gibdo and the other nine families got the setting only through
  the final retrain, with no arm of their own (§11).

## 10. Spawn spots as a general rule, the trained arms, and the ladder clip's landing (rejected and kept apart)

This continues §6. It closes the arm left open there, tries a general spawn rule in place of hand-placed path
costs, and tries a mask on Dungeon 4's ladder clip. Branches `r25-spawns` (the rule `7b786c4`, 0x27's hand costs out
`cbeeb62`, the clear-landing mask `54c7a6e`, docs `b8029fd`), `r25-spawnstr` (the trained arm), `r25-spawnsclip`
(the mask's screen).

**Closing §6.** The suites ran on `r25-bleed-ow` (`path_costs` and the 0x27 costs) and were all green: cargo, pytest
(750 passed, 4 skipped), replay verify (3 ok), the state audit (613 states, 0 problems), `asm.py check --final`. The
training arm `r75-bow-ow` was continued with `--load {name} --iterations 250 --seed 75 --stop-success 1.01`, without
`--sil-coeff 0.1`. It was judged against the promoted models on the same seeds:

- **ow-tektite with 0x1a's `wrap_in`: not promoted.**
  - Screen (post-d2, 300 paired, against the promoted ow-tektite with the 0x27 costs): 0x1a N→W 0.39 → 0.27 hearts a
    visit, 422 → 329 frames, deaths there 1 → 3. One run went E into 0x1b twice and recovered. D1's entrance 97.3 →
    96.0% (−1.3 ± 1.5), hearts there 3.55 → 3.67.
  - 1000 runs paired, clips on: arrivals at D1's entrance 96.3 → 95.6% (−0.7 ± 0.9), hearts there +0.21 ± 0.05,
    0x1a 0.41 → 0.29 a visit, but deaths in 0x1a 9 → 16.
  - 1000 runs paired, clips off: arrivals 93.2 → 94.3%, hearts −0.01 ± 0.05, 0x1a 0.47 → 0.51.
  - Fewer hearts lost in 0x1a with clips on are paid for with more deaths there, and clips off gains nothing.
- **ow-peahat trained with the 0x1f path costs: promoted.** The costs are §6's zero-shot failure, cost 2 on Link-tile
  rows 2-6 west of column 15 (columns 0-14). Entered from W, the Peahats spawn on (`$50`,`$5D`) and (`$80`,`$7D`), so
  the old way N-then-E met both on every visit. With the costs the way goes E along the middle (Y `$85`) and then N up
  X `$80`. Trained with them, the model follows them, and the cave the zero-shot model walked into stays a wrong exit
  it avoids.
  - Screen (the leg, 300 paired): 0x1f 1.00 → 0.28 hearts a visit, 0x1d 0.15 → 0.00, arrivals 100 → 100%, no wrong
    exit into the cave, hearts at Dungeon 5's entrance 6.55 → 7.47 (+0.92 ± 0.10).
  - 1000 runs, the leg (`0_45c_0` → 5:0x76), paired with `r74-1k-post-d4` and `r74-1k-noclips-post-d4`:

| the leg, 1000 runs | clips on | clips off |
|---|---|---|
| arrivals at 5:0x76 | 99.8 → 99.9% | 92.0 → 96.8% (+4.8 ± 1.0) |
| hearts at Dungeon 5's entrance (paired) | 6.57 → 7.48 (+0.91 ± 0.05) | +0.71 ± 0.07 |
| 0:0x1f hearts lost a visit | 1.02 → 0.27 | 0.96 → 0.27 (deaths 15 → 3) |
| 0:0x1d hearts lost a visit | 0.14 → 0.00 | |

The 0x1f costs are on `r25` as `70027cd`, and `ow-peahat/r75-bow-ow` is in the promoted team (§11's "old" team).

**The general rule, as built** (`reward.spawn_cost`, a u8, default 0 = off). It tries to find the spawn spots of §6
without placing costs by hand.

- **What counts as a hazard** (`zelda::spawns::SpawnWatch`, in `GameCompletion`; `Objective::spawn_hazards`, a
  position and a heading). On a visit's first playable frame (mode 5), every monster in slots 1-9 standing within 8 px
  of a spot of the game's spawn list for Link's facing counts (asm:Spawn_AssignPositions; the game has already skipped
  the unsafe spots). Excluded: caves and cellars, overworld rooms whose monsters come in from the edges
  (asm:RoomAttrsF bit 3), and Zoras. A hazard lasts while the monster is alive in its slot and within 16 px of its
  spot. Once it moves off, it never counts again that visit.
- **The cost shape** (`env::objective::spawn_costs`). The rule charges 2 × cost on the 3 × 3 Link tiles that touch the
  monster (under 9 px between middles, as asm:Link_CheckMonsterTouch_NoReset tests) and 1 × cost on the ring round
  them. The cost is placed at the monster's position and 16 px along its heading, taking the larger of the two. It
  does not apply while the objective is a fight. `Room::wavefront_with` adds the costs to the room's `path_costs`, and
  the target's descent reads the wavefront's step costs. With the rule off the wavefront is the room's own, byte for
  byte (tested).
- **Why the heading.** With a plain 5 × 5 square, the cheapest way through 0x27's strip (four of Link's rows) went over
  the X `$A0` Peahat. That Peahat rises, heading N from frame 1, into exactly that corridor, where §6 found the hits at
  Y 96-111. With the heading the way goes down the middle and along the strip's bottom row, the way of the hand-placed
  costs.
- To measure the rule alone, 0x27's hand-placed costs were taken out on this branch (`cbeeb62`).

**Untrained, it hurts.** `spawn_cost: 3` on every overworld family, zol and darknut, with the promoted models; 300
runs paired, clips on (baselines: §6's diagnosis chains, or the first 300 seeds of round 24's `r74-1k` files):

| segment (baseline) | arrivals | hearts on arrival (paired) | rooms that moved (hearts lost a visit) |
|---|---|---|---|
| start → D2 (`r74-1k` full route) | 100 → 100% | −0.02 ± 0.02 | none |
| post-d2 (diagnosis) | 95.7 → 96.3% (+0.7 ± 1.5) | −0.21 ± 0.06 | 0x27 0.50 → 0.58; 20 more returns into 0x17 (wrong exits) |
| post-d1 (diagnosis) | 100 → 100% | +0.04 ± 0.04 | 0x38 0.32 → 0.21, 0x37 0.24 → 0.21, 0x48 0.05 → 0.11 |
| d3 to Manhandla (diagnosis) | 94.3 → 92.0% (−2.3 ± 1.7) | −0.25 ± 0.10 | 3:0x49 0.15 → 0.21, 3:0x69 0.62 → 0.85 (deaths 4 → 9) |
| post-d3 (`r74-1k`) | 100 → 100% | −0.05 ± 0.03 | 0x65 0.08 → 0.14; 8 wrong-exit returns |
| post-d4, the leg (diagnosis) | 100 → 99.7% | −0.10 ± 0.09 | 0x3d 0.06 → 0.20, 0x1d 0.15 → 0.10, 0x4f 0.19 → 0.14; 1 wrong exit |
| d5 (the rule off on this code) | 46.0 → 39.0% | | |

In 0x27 alone (ow-zora `r74-leg`, 100 episodes × 3 states) the room lost 0.34 hearts an episode without costs, 0.49
with the rule, and 0.14 with §6's hand-placed costs.

**Why it fails in 0x27.** A hazard is given up once its monster is 16 px off its spot, which for a Peahat is around
frame 75. The risen Peahat is then in the corridor the path snaps back to. At frame 0 the monster has no heading yet,
so the way first goes over the Peahat and then flips. Two variants that keep the cost after the monster leaves its spot
were built as an experiment switch and never measured: `follow` (the cost follows the live monster) and `spot` (a
static square on the spot for the whole visit).

**Trained, it is uneven.** The arm `r75-sp` trained ow-peahat, ow-zora and ow-octorok with `spawn_cost: 3`, 250
iterations, `--sil-coeff 0.1` (seed 75; ow-peahat also seeds 76 and 77). Each trained checkpoint then played alone
with the rule on, the other families promoted with the rule off. 300 runs paired against the promoted team, clips on:

| run | segment | arrivals | hearts on arrival (paired) | rooms (hearts lost a visit) |
|---|---|---|---|---|
| ow-peahat s75 | the leg | 100 → 100% | +0.57 ± 0.07 | 0x1f 1.00 → 0.61, 0x1d 0.15 → 0.00 |
| ow-peahat s76 | the leg | 100 → 100% | +0.17 ± 0.06 | 0x1f 1.00 → 1.02, 0x1d 0.15 → 0.00 |
| ow-peahat s77 | the leg | 100 → 98.3% | −0.02 ± 0.06 | 0x1f 1.00 → 1.07, 0x1d 0.15 → 0.06; 5 runs failed in Dungeon 4's 0x71/0x61 (the leg starts at 0x45) |
| ow-octorok s75 | after D1 | 100 → 100% | **+0.39 ± 0.04** | 0x37 0.24 → 0.00, 0x38 0.32 → 0.18 |
| ow-octorok s75 | after D2 | 95.7 → 93.3% (−2.3 ± 1.7) | −0.14 ± 0.09 | its own 0x4c unchanged (0.10 → 0.08); the losses are in other families' rooms (0x0a, 0x1a) after the runs diverge |
| ow-zora s75 | after D2 | 95.7 → 96.0% | −0.22 ± 0.07 | 0x27 0.50 → 0.51, 0x38 0.17 → 0.22 |
| ow-zora s75 | after D1 | 100 → 100% | −0.05 ± 0.04 | 0x38 0.32 → 0.38 |
| ow-zora s75 | the leg | 100 → 100% | +0.13 ± 0.10 | 0x2d 0.56 → 0.69, 0x4f 0.19 → 0.04 |
| ow-zora s75 | after D3 | 100 → 100% | +0.06 ± 0.03 | 0x55 0.19 → 0.13 |

- **ow-peahat:** the three seeds spread widely. 0x1f costs 0.61, 1.02 or 1.07 hearts a visit, and the leg gains
  +0.57, +0.17 or −0.02 hearts. The hand-placed 0x1f costs (`r75-bow-ow`) beat every seed: 0x1f 0.27, +0.91 hearts at
  1000 runs.
- **ow-zora:** 0x27 is not fixed (0.50 → 0.51), and 0x38 gets worse.
- **ow-octorok:** the one trained case where the rule pays. Out of 0:0x37's cave the room drops to no loss (0.24 →
  0.00), and Link arrives after Dungeon 1 with 0.39 hearts more. After Dungeon 2 its own room is unchanged and arrivals
  fall 2.3 ± 1.7 points, within noise. The lead was to be checked with 1000 runs after Dungeon 1 and after Dungeon 2,
  clips on and off. That check was not run.

**The ladder clip's landing (4:0x21 S→N).** From 295 visits of the diagnosis chain (300 runs, promoted team, clips
on), 0x21 cost 0.224 hearts a visit, and 41 of the 66 hearts lost there were lost on `CLIP N`.

- At the CLIP N decision a monster was within 24 px of the landing (Link's X, Y `$55`) in 127 of 295 visits. Those
  visits lost 0.232 hearts within 90 frames of the clip (47% hit), against 0.068 (14%) with the landing clear. If
  every occupied clip went like a clear one, the saving would be at most 0.07 hearts a visit.
- Waiting does not clear the landing. Of 18,051 island frames with the landing occupied before the first clip, 16,435
  fall in the last occupied stretch before the clip: the Gels follow Link's X and stay. Where the landing did clear,
  the wait averaged 56 frames (median 38). So a rule must make Link kill (the boomerang kills Gels) or move along.
- Built: `clips: {clear_landing: true}` (`54c7a6e`; `zelda::clips::landing`, `landing_occupied`). The CLIP action is
  not offered while a live monster is within 24 px of the clip's landing.
- Screen: stalfos-keese-gel with `clear_landing`, post-d3 → 4:0x11, 300 runs paired, untrained. 0x21 went from 0.234
  to 0.180 hearts a visit and from 506 to 538 frames, deaths 3 → 2, arrivals 95.3 → 95.7%, hearts at 4:0x11 +0.05 ±
  0.01. No 1000-run check was run.

**The decision: neither lands in round 25.**

- **The general spawn rule** (branch `r25-spawns`) failed its own check. It does not reproduce the hand-placed 0x27
  gain (0.49 against 0.14 hearts in the room alone), and untrained it hurts (D3 −0.25 ± 0.10 hearts, post-d2 −0.21 ±
  0.06, D5 46.0 → 39.0%). Its one trained lead is ambiguous: ow-octorok gains +0.39 ± 0.04 hearts after Dungeon 1 but
  loses 2.3 ± 1.7 points of arrivals after Dungeon 2, at 300 runs.
- **The clear-landing mask** gains little: +0.05 ± 0.01 hearts at 4:0x11 at 300 runs, for 32 more frames a visit.
- Both stay on their branch and go to the backlog.

**Landed from this work and §6:** the 0x27 path costs, the 0x1f path costs, and ow-peahat `r75-bow-ow`.

## 11. The final retrain and the team

**What was trained.** Eleven dungeon families were continued from their promoted runs with the round's heart pricing
(`reward.heart_value: {exit: true, potential: 20}`, §9). Each ran 244 iterations (999,424 steps) with `--sil-coeff
0.1`, `--parallel 8`, `--stop-success 1.01` and seed 75. They trained as four rounds:

- `r75-f-a`: gibdo, goriya, vire;
- `r75-f-b`: stalfos-keese-gel, moldorm, zol;
- `r75-f-c`: like-like, d2-rope, trap-room, equipment-room;
- darknut three times, seeds 75 / 76 / 77 (`r75-f-dk-s75/76/77`). Its curriculum included the six hand-made 0x37
  tracks of §5 (X `$70` and SWORD N at a Darknut ahead).

The canonical evaluation is from each round's own table:

| family | run | continued from | train wall | train success (last 5) | eval success | worst room (rate) |
|---|---|---|---:|---:|---:|---|
| gibdo | `r75-f-a` | `r74-d5new` | 1654 s | 99% | 99% | 5:0x47 (98%) |
| goriya | `r75-f-a` | `r71-goriya-cont` | 846 s | 95% | 97% | 2:0x5e (91%) |
| vire | `r75-f-a` | `r72-vire-cont` | 1402 s | 100% | 100% | 4:0x50 (99%) |
| stalfos-keese-gel | `r75-f-b` | `r72-skg` | 963 s | 100% | 100% | 1:0x53 (99%) |
| moldorm | `r75-f-b` | `r71-c` | 1014 s | 90% | 91% | 2:0x3e (91%) |
| zol | `r75-f-b` | `r74-d5cont` | 1444 s | 100% | 100% | 4:0x40 (99%) |
| like-like | `r75-f-c` | `r72-likelike2` | 1291 s | 100% | 100% | 4:0x32 (100%) |
| d2-rope | `r75-f-c` | `r71-a` | 1429 s | 99% | 99% | 2:0x4e (98%) |
| trap-room | `r75-f-c` | `r72-trap` | 1871 s | 100% | 100% | 3:0x4c (100%) |
| equipment-room | `r75-f-c` | `r74-d5cont` | 1446 s | 100% | 100% | 1:0x7f cave (99%) |
| darknut | `r75-f-dk-s75` / `s76` / `s77` | `r74-darknut5` | 1279 / 1333 / 1382 s | 98% each | 98% each | 5:0x05 (91%) each |

**Darknut on the Dungeon 5 segment.** Each candidate played 1000 runs, seeds 0-999, clips on, with the act mode
`temperature:0.5+relax:16` and the other families as promoted before the retrain. The chains ran on `r25` at
`fd7cc6e`, which has 0x37's path costs. Three groups were compared:

- `ref`: `r74-darknut5`, the promoted darknut;
- `A-s75/76/77`: §9's priced runs `r75-v2-ab-s75/76/77`, trained without the tracks;
- `B-s75/76/77`: `r75-f-dk-s75/76/77`, trained with the tracks.

Arrival ± is the paired difference against `ref`. The room cells give hearts on entry / hearts lost a visit / deaths
(per 1000 runs).

| darknut | D5 arrival | vs ref (paired) | the Olive Garden | 0x64 back | 0x66 W→N | 0x47 | 0x37 | Digdogger |
|---|---:|---|---|---|---|---|---|---|
| ref | 46.8 | | 9.22 / 4.62 / 92 | 6.33 / 1.55 / 68 | 4.89 / 1.20 / 109 | 4.47 / 0.79 / 62 | 4.04 / 1.20 / 94 | 3.39 / 0.07 / 18 |
| A-s75 | 50.8 | +4.0 ± 2.2 | 9.22 / 4.45 / 88 | 6.43 / 1.27 / 59 | 5.20 / 1.23 / 108 | 4.79 / 0.77 / 56 | 4.37 / 1.21 / 86 | 3.61 / 0.06 / 16 |
| A-s76 | 50.8 | +4.0 ± 2.2 | 9.22 / 4.45 / 81 | 6.31 / 1.26 / 60 | 5.02 / 1.20 / 105 | 4.56 / 0.83 / 81 | 4.21 / 1.17 / 84 | 3.45 / 0.04 / 17 |
| A-s77 | 50.5 | +3.7 ± 2.1 | 9.22 / 4.18 / 80 | 6.59 / 1.25 / 55 | 5.41 / 1.33 / 99 | 4.82 / 0.74 / 65 | 4.49 / 1.21 / 99 | 3.77 / 0.08 / 23 |
| **B-s75** | **52.3** | **+5.5 ± 2.2** | 9.22 / 4.06 / 76 | 6.73 / 1.23 / 44 | 5.46 / 1.21 / 93 | 4.96 / 0.81 / 63 | 4.53 / 1.19 / 91 | 3.85 / 0.10 / 29 |
| B-s76 | 51.0 | +4.2 ± 2.1 | 9.22 / 4.15 / 71 | 6.59 / 1.25 / 42 | 5.20 / 1.24 / 106 | 4.70 / 0.87 / 81 | 4.30 / 1.19 / 92 | 3.60 / 0.06 / 20 |
| B-s77 | 50.1 | +3.3 ± 2.2 | 9.22 / 4.13 / 76 | 6.58 / 1.24 / 54 | 5.37 / 1.39 / 112 | 4.76 / 0.83 / 68 | 4.34 / 1.19 / 94 | 3.60 / 0.07 / 24 |

- **The re-measurements differ slightly.** On these seeds, round 24 and §7 measured `ref` at 46.4%, and §9 measured
  the A runs at 50.9 / 49.9 / 51.3% on their own branch's code. Here they read 46.8% and 50.8 / 50.8 / 50.5%. The
  differences are inside the noise (§7).
- **The arms.** Every priced continuation beats `ref` by 3.3-5.5 points, each about 1.5-2.5 standard errors. A ranges
  50.5-50.8% and B 50.1-52.3%. The tracks give no difference between the arms beyond noise.
- **The tracks changed nothing in 0x37:** 1.19 hearts a visit and 91 / 92 / 94 deaths for B, against 1.21 / 1.17 /
  1.21 and 86 / 84 / 99 for A and 1.20 and 94 for `ref`. §5's follow-up gives the cause (the policy never tries the
  column) and the fix that worked: behaviour cloning of the scripted rule, `r75-d537-c5`, continued from
  `r75-f-dk-s75`.
- **Where B differs from `ref`:** fewer deaths in the Garden (71-76 against 92) and in 0x64 on the way back (42-54
  against 68), and more hearts into every room after.

`r75-f-dk-s75`, the best of B, was confirmed on fresh seeds 1000-1999 (1000 runs, paired with `ref`; the files
`fresh-ref-on|off-d5` and `fresh-B75-on|off-d5`):

| fresh seeds 1000-1999 | ref | `r75-f-dk-s75` | paired difference | the Garden: hearts lost a visit, deaths | 0x37: hearts lost a visit, deaths |
|---|---:|---:|---|---|---|
| clips on | 478 (47.8%) | 507 (50.7%) | +2.9 ± 2.2 | 4.54, 75 → 4.07, 54 | 1.20, 87 → 1.18, 89 |
| clips off | 333 (33.3%) | 416 (41.6%) | +8.3 ± 2.1 | 3.35, 288 → 3.17, 234 | 0.93, 32 → 0.95, 54 |

Clips on, the confirmed gain (+2.9 ± 2.2) is smaller than the screen's (+5.5 ± 2.2), as §7 predicts for the best of
three. Clips off it is +8.3 ± 2.1. Clips off, Link also reaches the Garden more often (874 visits against 807) and
with more hearts (6.88 against 6.23). §7 measured the promoted model on these seeds at 47.0 | 34.3% on its branch's
code; here it reads 47.8 | 33.3%.

**The team.** The promoted team before the retrain ("old") played 1000 runs per segment and the full route against the
same team with the eleven retrained families ("new", darknut `r75-f-dk-s75`). Both ran clips on, seeds 0-999, act mode
`temperature:0.5+relax:16`, on `r25` at `70027cd`. The old team already had `boss-manhandla/r75-mh-fight` (§3),
`ow-peahat/r75-bow-ow` (§10) and the 0x27 and 0x1f path costs (§6, §10). Arrivals at each segment's end, ± the paired
difference:

| segment (from → arrives at) | old | new | difference (paired) |
|---|---:|---:|---|
| d2 (`2_7ds_0` → 0:0x3c) | 89.9 | 88.2 | −1.7 ± 1.4 |
| d1 (`1_73s_0` → 0:0x37) | 99.7 | 99.7 | +0.0 ± 0.2 |
| d3 (`3_7cs_0` → 0:0x74) | 75.8 | 81.0 | +5.2 ± 1.8 |
| d4 (`4_71s_0` → 0:0x45) | 81.3 | 92.3 | +11.0 ± 1.5 |
| d5 (`5_76s_0` → the D5 triforce) | 46.8 | 60.3 | +13.5 ± 2.2 |

The full route from the game start, 1000 runs each, seeds 0-999 (the share of runs that reach each point):

| Link reaches | old | new | difference (paired) |
|---|---:|---:|---|
| Dungeon 1 (1:0x73) | 81.5 | 83.9 | +2.4 ± 1.6 |
| Dungeon 4 (4:0x71) | 45.2 | 59.2 | +14.0 ± 2.2 |
| Dungeon 5 (5:0x76) | 36.6 | 53.9 | +17.3 ± 2.2 |
| **the D5 triforce** | **9.5** | **18.6** | **+9.1 ± 1.6** |

**Dungeon 2 did not gain.** It is the one segment that lost (−1.7 ± 1.4, within noise). The deaths in its two Goriya
rooms moved in opposite directions in the segment pair and the full-route pair. Counted per room from the four files
(deaths per 1000 runs; hearts on entry):

| room | D2 segment: old → new | full route: old → new |
|---|---|---|
| 2:0x5e S→N (goriya) | 14 → 28 (2.94 hearts in both) | 42 → 36 (2.75 → 2.76) |
| 2:0x1e S→N (goriya) | 46 → 52 (2.72 → 2.69) | 47 → 38 (2.69 → 2.69) |
| 2:0x3e S→N (moldorm) | 25 → 13 | 20 → 20 |

In the full-route files, Link took the D2 triforce (reached route step 24) in 854 → 876 runs (+2.2 ± 1.5, paired).
So the segment's loss does not appear on the full route, where the same families gained a little, and neither change
is outside the noise.

**The decision: the whole team is adopted.** The eleven retrained families are adopted and promoted in the round's
models, with darknut as `r75-d537-c5` (§5's follow-up, continued from `r75-f-dk-s75`) instead of `r75-f-dk-s75`.
`ow-peahat/r75-bow-ow` (§10) and `boss-manhandla/r75-mh-fight` (§3) are promoted likewise. Every one of these runs
plays with the act mode `temperature:0.5+relax:16`. The chains of record with the final team, models alone and with
the planner of §12, are in §13.

## 12. One action of foresight: the guard and the honest planner

Branch `r25-actguard` (`e0d62b2` code and tests, `c10239a` docs, `f3a5b0a` merge of `main`, `34484cb` the planner as
a labelled act mode, the decisions entry and the teacher document). `docs/rust-actor.md` "Foresight" and "The honest
planner"; decisions 2026-11-16 "The honest planner beside the models".

**Where it came from.** §4's low-health rules did not get Link hit less, and none was adopted. The lead they left was
a one-step lookahead: in a 20-run Dungeon 5 smoke test, `guard:8` on every decision reached the D5 triforce in 16 of
20 runs against the standing rule's 8. Guarding only at 2 hearts or fewer reached it in 10. This section made the
guard exact, built an honest version beside it, screened the settings of both and measured the best of each in
1000-run chains.

**The mechanism.** An act mode may end with one foresight part. Before a decision is taken, the env tries the
decision's action in the emulator (`PoolEnv::trial`, `Session::trial`) and comes back. If the action would lose health
(or Link's life), the next of the mode's K best actions is tried, and the first that would not is taken instead. If
none would be safe, Link takes the decision's own action. The candidates are ranked as the mode draws them: ln q plus
the decision's own Gumbel noise. There are two kinds:

| spelling | the action is tried against | refused when it loses health in | what the numbers claim |
|---|---|---|---|
| `guard:K` (the full guard) | the game's real future: its random numbers as they stand | that future | a valid run of button presses, chosen with knowledge of the future that a player does not have |
| `plan:K:S` (the honest planner) | S futures, the game's random numbers redrawn before each trial (a natural RNG state of a phase keyed by run, decision and future, as the search's `perturb_rng`; the same S futures for every candidate) | any of them | planning with a model of the game: the rules and the room as it stands, not the dice |
| `plan:K:S:R` | the same | more than the share R of them | the same |

The honest planner never sees the real future: its trials play the room on with different random numbers. K = 1
(`guard:1`, `plan:1:S`) only tries the decision's own action and never replaces it. With `low:H:` (§4) the foresight
belongs to the low-health part and applies only at H hearts or fewer.

**Exactness and replays:**

- A Rust test shows that a trial, with the game's own future or a redrawn one, leaves everything as it was: the
  machine-state bytes, the game's random numbers, the counters, the credit ledger, the state and mask, the open clip
  and the pre-entry snapshot. A session that tries actions before every step plays frame for frame like a twin that
  never does (300 steps).
- pytest: `guard:1` and `plan:1:3` give plays and room records identical to the mode without foresight. `guard:4`
  refuses exactly the action the unguarded run took on a step where that run lost health.
- At scale: `guard:1` gave records identical to the standing rule on all 1200 screen runs (about 530k trials).
- A run with foresight is a sequence of button presses like any other, and its replay verifies on the emulator alone.
  pytest records chain runs with `guard:4`, `plan:4:2` and `plan:16:1` and verifies each replay bit for bit. Two
  replays recorded by `zelda-headless.py` with the planner also pass `tf replay verify`. A replaying actor has no
  foresight, and training rollouts never try actions.
- Every room record with foresight carries `decisions.foresight`: kept, replaced and nothing-safe counts, the trials,
  their emulated frames, and per replaced decision the refused and the taken action with their probabilities.
- Suites on the branch: cargo, `tf replay verify`, the state audit and `asm.py check --final` green. The full pytest
  run had one failure, an estimate assertion in `tests/test_training_gui.py` from the job line's code; that file
  passed 56 of 56 run alone.

**The cost.** Emulated frames per run: the full guard about 2.0× the standing rule's, `plan:K:S` about (1 + S)×, the
variants restricted to 2 hearts or fewer 1.1-1.2×. Wall time of the four screen segments on the shared machine (noisy
under load): standing rule 419 s, `guard:8` 701 s (1.7×), `plan:8:1` 989 s (2.4×), `plan:8:4` 1493 s (3.6×),
`plan:8:8` 3866 s (9.2×).

**The screen.** 300 runs per candidate, clips on, on §4's seeds; the standing rule here played record for record like
§4's baseline files. Arrivals in %, with the paired difference ± SE against the standing rule where the report gives
it:

| mode | D5 | D3 | D4 | D2 | emulated frames |
|---|---|---|---|---|---|
| standing rule | 47.7 | 64.7 | 83.3 | 88.3 | 1× |
| `guard:4` | 80.3 (+32.7 ± 3.5) | 73.3 | 97.3 | 97.3 | 2.0× |
| `guard:8` | 89.7 (+42.0 ± 3.1) | 75.3 (+10.7 ± 3.6) | 99.3 (+16.0 ± 2.2) | 98.3 (+10.0 ± 1.9) | 2.0× |
| **`guard:16`** | **92.0** | **77.3** | 99.3 | **99.7** | 2.1× |
| `plan:8:1` | 89.7 | 72.3 | 99.7 | 97.0 | 2.0× |
| `plan:8:4` | 89.0 | 76.3 | 99.7 | 98.7 | 5.0× |
| `plan:8:8` | 88.3 | 76.0 | 99.7 | 98.0 | 9.0× |
| `plan:8:4:0.5` | 89.7 | 74.0 | 99.3 | 97.3 | 3.1× |
| `plan:8:8:0.5` | 89.0 | 73.3 | 99.7 | 97.0 | 5.2× |
| **`plan:16:1`** | 91.0 | **78.7** | 99.7 | 98.7 | about 2× |
| `guard:8` at ≤ 2 hearts | 54.0 (+6.3 ± 1.9) | 68.3 | 93.7 | 96.0 | 1.1-1.2× |
| `plan:8:1` at ≤ 2 hearts | 53.3 | 67.7 | 93.0 | 96.0 | 1.1-1.2× |
| `guard:4` / `plan:8:4` / `plan:8:8` at ≤ 2 hearts | 49.7 / 52.7 / 52.0 | 67.0 / 67.7 / 68.0 | 91.0 / 93.7 / 93.0 | 94.3 / 95.0 / 95.0 | 1.0-2.3× |

- **K matters; S and R do not.** `guard:4` is clearly below `guard:8`. `guard:16` is above `guard:8` on D5 by +2.3 ±
  2.0 (not significant), with more hearts at every D5 room. More futures and the "half" share change nothing beyond
  noise, at 3-9× the emulation. `guard:16` and `plan:16:1` went to the chains.
- **Honest equals full within noise.** Over one action (4 to 32 frames) the redrawn random numbers rarely change the
  outcome: what decides a hit is the visible state.
- **Foresight only at low health helps far less.** Guarding every decision means Link reaches low health less often.

For `guard:8` against the standing rule:

| | D5 | D3 | D4 | D2 |
|---|---|---|---|---|
| hits per decision above 2 hearts | 1.44 → 0.68% | 1.44 → 0.75% | 1.18 → 0.42% | 0.90 → 0.49% |
| hits per decision at 2 hearts or fewer | 1.42 → 0.57% | 1.87 → 2.16% | 1.20 → 1.06% | 1.20 → 0.79% |
| deaths | 154 → 24 | 106 → 74 | 50 → 1 | 34 → 5 |
| stuck runs | 3 → 7 | 0 → 0 | 0 → 1 | 1 → 0 |
| decisions replaced | 2.8% | 2.1% | 1.6% | 0.9% |
| nothing safe found | 0.7% | 0.8% | 0.4% | 0.5% |

D3 at 2 hearts or fewer gets worse because that is Manhandla. Variety holds: acting entropy per decision went 0.45 →
0.48 on D5 and 0.50 → 0.58 on D3. All 50-decision prefixes were distinct in every mode, and two runs from the same
start parted after about 5 decisions on average in every mode.

**The 1000-run chains.** These tables were measured with the team as it stood before the final retrain of §11: the
round 24 team, with darknut `r74-darknut5`, Manhandla `r71-e` and ow-peahat `r74-leg` (the chain files' provenance).
They ran on the branch's own code, which came from `main` and has none of the round's path costs (0x27, 0x1f, 0x37)
or heart pricing. All runs are 1000 runs on seeds 0-999.

- **The "models alone" column:** D3 and the full route were re-measured, because round 24's `r74-1k` D3 and full-route
  files used darknut4; clips off D3, D4, D5 and the full route were re-measured too.
- **Reused round 24 files:** D2, post-d2, D1, post-d1, post-d3, D4 and post-d4 come from `r74-1k`, and D5 from
  `r74b-1k`. Before reuse, the branch's code was shown to play the standing rule record for record like `r74b-1k` D5
  (300 seeds) and like `r74-1k` D2 and D4 up to the arrival room. Clips off, the re-measured D4 and D5 equal round
  24's 59.7% and 34.0%.

Clips on. Arrivals, with the paired difference against the models alone, and deaths (d) and stuck runs (s) per 1000:

| segment | models alone | honest planner (`plan:16:1`) | full guard (`guard:16`) |
|---|---|---|---|
| D2 | 89.9% (d100 s1) | 98.9% (+9.0 ± 1.0; d11 s0) | 99.7% (+9.8 ± 1.0; d3 s0) |
| post-d2 | 95.2% (d48) | 98.4% (+3.2 ± 0.7; d14 s2) | 98.6% (+3.4 ± 0.7; d12 s2) |
| D1 | 99.7% | 99.9% (+0.2 ± 0.2) | 99.9% (+0.2 ± 0.2) |
| post-d1 | 100% | 100% | 100% |
| D3 | 62.9% (d371) | 77.6% (+14.7 ± 1.9; d223) | 75.8% (+12.9 ± 2.0; d241) |
| post-d3 | 100% | 100% | 100% |
| D4 | 81.3% (d186) | 99.5% (+18.2 ± 1.2; d1 s4) | 99.4% (+18.1 ± 1.3; d0 s6) |
| post-d4 (the leg) | 99.8% | 99.4% (−0.4 ± 0.2; d3 s3) | 99.4% (−0.4 ± 0.2; d3 s3) |
| D5 | 46.4% (d529 s7) | 93.5% (+47.1 ± 1.7; d39 s26) | 94.1% (+47.7 ± 1.7; d29 s30) |
| **full route** | **6.1%** (d933 s6) | **60.9%** (+54.8 ± 1.7; d358 s33) | **62.6%** (+56.5 ± 1.7; d335 s39) |

Hearts on entering the killer rooms, full route, clips on:

| room | models alone | honest planner | full guard |
|---|---:|---:|---:|
| 2:0x5e | 2.74 | 2.92 | 2.92 |
| 2:0x1e | 2.69 | 2.92 | 2.92 |
| 1:0x23 | 2.92 | 4.26 | 4.33 |
| 3:0x59 N→S | 4.54 | 5.60 | 5.62 |
| 3:0x5b W→E | 3.02 | 5.03 | 5.12 |
| Manhandla | 3.23 | 5.24 | 5.29 |
| Gleeok | 3.84 | 6.33 | 6.40 |
| the Olive Garden | 5.81 | 8.93 | 9.03 |
| 5:0x64 on the way back | 4.89 | 8.10 | 8.40 |
| 5:0x66 W→N | 3.74 | 7.26 | 7.57 |
| 5:0x47 | 3.98 | 7.25 | 7.65 |
| 5:0x37 | 3.62 | 6.97 | 7.37 |
| Digdogger | 3.07 | 6.11 | 6.49 |

Clips off:

| segment | models alone | honest planner | full guard |
|---|---|---|---|
| D3 | 62.9% | 77.6% (+14.7 ± 1.9) | 75.8% (+12.9 ± 2.0) |
| D4 | 59.7% (d401) | 99.8% (+40.1 ± 1.6; d0 s2) | 99.7% (+40.0 ± 1.5; d1 s2) |
| D5 | 34.0% (d655 s5) | 90.2% (+56.2 ± 1.8; d64 s34) | 91.3% (+57.3 ± 1.7; d58 s29) |
| **full route** | **0.8%** (d990 s2) | **51.4%** (+50.6 ± 1.6; d402 s84) | **53.6%** (+52.8 ± 1.6; d375 s89) |

D3 is the same clips on and off, run for run: no clip way passes through Dungeon 3. Clips off, the hearts rise with
foresight from 2.90 to 6.10 at Gleeok on the D4 segment, and on the full route from 3.82 to 6.98 at the Olive Garden
and from 3.56 to 6.35 at Digdogger.

**What the numbers say:**

- **The honest planner is within about one standard error of the full guard.** It is within about 2 points
  everywhere and slightly ahead on D3. The full guard is ahead by 1.7 points on the full route clips on and 2.2 clips
  off, both about one standard error of the paired difference, so not a clear difference.
- **What still stops it is Manhandla and Dungeon 3.** D3 reaches only 77.6% although Link now arrives at Manhandla
  with 5.24 hearts instead of 3.23. This fits §1 and §3: Manhandla is a bombs problem, not a hearts problem. In the
  screen's bleeding-room counts, Manhandla went from 1.26 to 1.44 hearts lost a visit under the guard, with nothing
  safe for 2.4% of its decisions, and its deaths went from 89 to 74 only.
- **Stuck runs rise with foresight.** On the full route they go from 6 to 33 clips on and from 2 to 84 clips off. On
  the clips-off full route about one in six of the planner's failures is a stuck run, not a death: the planner keeps
  refusing hits and Link stops making progress. This needs a look before the planner becomes the golden-path or play
  mode.
- **The leg loses a little:** −0.4 ± 0.2 on post-d4 with either mode (3 deaths and 3 stuck runs each).

**The decision (the owner, Nov 16; decisions 2026-11-16 "The honest planner beside the models").** "The honest
planner (`plan:K:S`, redrawn random numbers) runs beside the models. Models alone stay the primary number and what
training optimises; the honest planner is a second, labelled act mode: reported beside it in every chain report, used
for the golden path and for play, and to be tried as a teacher. The full guard (real future) was not chosen: keep its
numbers as the comparison, but it is not to become a default anywhere."

- The setting is `temperature:0.5+relax:16+plan:16:1`, `registry.PLAN_ACT`.
- It can be selected by any `--act`: `eval-chain`, `eval-batch`, `r56_chains.sh ACT=`, `zelda-headless.py` and its
  `--golden` path, and `play-zelda.py`, whose Options menu lists it as "honest planner".
- It is no default: `ACT_RULE`, `DEFAULT_ACT`, `PLAY_ACT` and every model's `act` file stay the models alone, and
  training never tries actions.
- The full guard is kept for comparison only and is no default anywhere.

The planner's chains with the final team of §11 are in §13.

**The planner as a teacher** (`docs/experiments/round25-planner-teacher.md`; analysis only, no training). Where the
planner's choice differs from the model's is a label a training could learn from, and every room record of a planner
run already carries it (`decisions.foresight.replaced`).

- **How often:** the planner replaces 0.5-6.6% of a family's decisions (ow-peahat 6.6%, darknut and ow-zora 5.4%,
  Manhandla 4.5%, Gleeok 3.4%).
- **What it replaces:** the refused action is the model's own first choice in 55-87% of cases, and the safe action
  taken instead has a median probability under the model well below 0.1.
- **Where:** most in 5:0x37 (15.4% of its decisions), 0:0x2d, 5:0x64, 3:0x69 and 0:0x1f.
- **The typical correction is small:** a shorter move or a wait, a step back or sideways instead of swinging into a
  touching Darknut, or the boomerang before a CLIP.
- **The proposed order of trials, simplest first:** a penalty on the refused action, labelled during ordinary
  rollouts by `plan:1:1` (which tries but never changes play); then a cross-entropy target toward the planner's action;
  a mask during rollouts only as a curriculum aid. It proposes to start with darknut and ow-peahat.
