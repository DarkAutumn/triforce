# Round 19: bombs (the new baseline)

Lee, Nov 11: bombs should help with the Goriyas and Darknuts. Bombs as a weapon where enemies take more than 3
hits (goriya, darknut), moldorm too, not Aquamentus and not the overworld; save bombs for what the route needs
(D3: 3 for the bomb wall, 2 for Manhandla), with machinery for "from here to there never drop below X"; train every
family to bomb the 10th damage-free kill (it arms a forced drop of 4 bombs), rewarded for arming and for the
pickup; track bomb hits, misses, 10th-kill opportunities and tries. Retrain everything regardless: this is the new
baseline. Code on branch `bombs` (worktree `../triforce2-bombs`), version 1.3.2.0 (observation schema 3).

## Summary

Bombs are in (1.3.2.0, observation schema 3): bomb use per family (`free` for goriya, darknut, moldorm, Dodongo and
Manhandla, `required` where the route needs them, the 10th damage-free kill everywhere else), D3's reserve of 5,
rewards and metrics for all of it, GUI charts and an early stop that also waits on the bomb value. The whole team was
retrained from scratch (§3). Two fixes followed the first chains: half the resets keep the start state's bombs
(Dungeon 2 has none; §4–5), and, Lee's idea, Link now clears 2:0x5e for its Red Goriyas' bomb drops before the
Moldorm (a new route rule `treasure: clear`; §6). Final chains, 1000 runs per segment, against round 18: from the
start to Dungeon 1 55.2% (49.3%), to Dungeon 3 36.3% (29.2%), to the D3 triforce 10.3% (10.6%, within noise). The
D2 segment from its entrance state is 82.1% (87.6%), and the D3 segment 39.5% (40.6%): Darknuts stall on 3:0x59's
kill-all visit (74 runs, bombs masked by the reserve) and Manhandla kills more (123 vs 84). Both are in backlog.md.

## 1. Machinery (`6b97144`, `437e853`)

Design and reasons: `docs/decisions.md` 2026-11-11 "Bombs as a weapon, a bomb reserve and the 10th kill's forced
bomb drop".

- `items.bombs` per family: `free` goriya, darknut, moldorm, boss-dodongo, boss-manhandla; `required` ow-tektite,
  trap-room; `tenth` the other 13. Recomputed every step.
- routes.yaml `bomb_reserve: 5` from Dungeon 3's entrance; 3:0x5b takes its bombs at 4 or fewer.
- Observation schema 3 (`link` 80 wide): `bomb_usable`, `bomb_reserve`, `bombs_spare`, `help_count`,
  `next_kill_arms`, `drop_armed`, `drop_is_bombs`, `fairy_next`.
- `reward-armed-bomb-drop` (+0.75); bomb costs and hit rewards unchanged (−0.25 per bomb, +0.5 per hit).
- Every family: `ram_randomization: {bombs: [0, 8], help_count: [0, 9]}`.
- Metrics `bombs/{placed, hits, misses, tenth_kills, tenth_tries, armed, picked_up}` (training, eval, chains, GUI).

Checked on the emulator (throwaway scripts, `/tmp/bomb_facts.py`, `/tmp/darknut_bombs.py`):

- A bomb kill on the 9th → 10th damage-free kill sets `$51`; the next drop is a bombs item (2 of 2 seeds in 1:0x44;
  also the test `a_bomb_on_the_tenth_kill_arms_a_forced_bomb_drop`).
- Darknuts (3:0x5b, 6 seeds, bombs placed within 40 px, Darknuts that kept one direction during the blast): facing
  opposite to Link's facing when he placed the bomb, 434 cases, 0 hits; otherwise 951 hits of 1,478. Lee's claim
  holds and is broader than "facing Link": only the directions count.

## 2. Pilot: goriya, darknut, moldorm (`r65-pilot-a`, `r65-pilot-b`)

`scripts/train_round.py --steps 8000000 --parallel 8 --eval-parallel 3 --chain-runs 0 --train-args "--stop-success
0.99 --sil-coeff 0.1"`, from scratch (schema 3), goriya in one stream, darknut then moldorm in the other.

goriya (`r65-pilot-a`): ran the whole 8M steps (no early stop; 6,939 s, 1,154 steps/s). Evaluation (1500 episodes,
100 per start state, the default act mode) against the round 17 model on the same rooms (r61-dg, argmax, before
bombs):

| | r61-dg (no bombs) | r65-pilot-a (bombs) |
|---|---:|---:|
| success | 93.0% | 96.1% |
| health change / episode | −0.27 | −0.19 |
| steps / episode | | 39.9 |
| bombs placed / hits / misses | – | 2.03 / 2.28 / 0.53 |
| 10th kills / bomb tries / armed / bombs picked up | – | 0.14 / 0.15 / 0.08 / 0.57 |

Worst start room 2:0x5e (90%). The model bombs about twice a room; a bomb hits 1.1 enemies on average and a
quarter hit nothing. Of the 10th kills it meets, it tries a bomb on about every one and arms the drop about half
the time. No reward change was needed: the pilot's bomb terms stand for the retrain.

darknut and moldorm (`r65-pilot-b`, 8M steps each, no early stop), against the round 18 models in their act modes
(same rooms, same evaluation size):

| | darknut r63a (sample) | darknut r65 | moldorm r63b (argmax) | moldorm r65 |
|---|---:|---:|---:|---:|
| success | 97.4% | 98.4% | 87.0% | 88.3% |
| health change / episode | −0.49 | −0.37 | −0.25 | −0.17 |
| bombs placed / hits / misses | – | 0.24 / 0.19 / 0.10 | – | 3.62 / 7.22 / 0.63 |
| armed / bombs picked up | – | 0.03 / 0.18 | – | 0.27 / 0.30 |

Moldorm bombs the chain hard (two segments per bomb). The Darknut rooms rarely bomb: most of their visits leave
(`treasure: skip`) and D3's reserve of 5 masks bombs at 5 or fewer, which the random 0–8 bombs make about two
thirds of resets. All three pilots are better than the models they replace on success and hearts.

After the pilots, `bombs-dev` (the bomb damage counter, Lee: "how hard we are hitting", and the bombs early-stop
rule) was merged into `bombs`.

## 3. The full retrain

The other 17 families from scratch, same recipe, two streams (`--parallel 8` each): `r66-ow` (the 8 overworld
families, from `../triforce2-bombs-dev` with the damage counter and the bombs early-stop rule; 6.4 h) and `r66-dg`
(the 9 dungeon families, from `../triforce2-bombs` after the merge; 5.6 h). Early stops: everything but ow-moblin,
ow-lynel, d2-rope and boss-manhandla, which used the whole 8M steps. The bombs rule held training on at least once
(ow-octorok: "bombs still improving").

Act modes (`/tmp/actmodes3.py`: 100 episodes per start state, seed 0, both modes; argmax unless its paired 95% CI
on success is below sample's). Cells: success / hearts lost per episode / bombs placed per episode; the old column
is the round 18 promoted run's canonical evaluation (its act mode, round 18 code):

| family | sample | argmax | act | round 18 |
|---|---|---|---|---|
| ow-start | 100.0% / +0.000 / 0.00 | 100.0% / +0.000 / 0.00 | argmax | r61-ow 100.0% / +0.000 |
| ow-octorok | 98.5% / +0.094 / 0.01 | 98.7% / +0.082 / 0.00 | argmax | r63a 98.8% / +0.089 |
| ow-zora | 99.1% / +0.192 / 0.03 | 99.3% / +0.160 / 0.02 | argmax | r63b 99.3% / +0.153 |
| ow-moblin | 97.6% / +0.159 / 0.01 | 97.7% / +0.143 / 0.00 | argmax | r61-ow 98.4% / +0.119 |
| ow-leever | 100.0% / −0.042 / 0.17 | 100.0% / −0.050 / 0.16 | argmax | r63b 100.0% / −0.103 |
| ow-tektite | 99.6% / −0.046 / 0.27 | 99.6% / −0.056 / 0.21 | argmax | r63b 99.3% / −0.031 |
| ow-lynel | 99.1% / +0.218 / 0.00 | 98.2% / +0.217 / 0.00 | argmax | r63a 99.2% / +0.132 |
| ow-boulder | 100.0% / +0.148 / 0.00 | 99.9% / +0.131 / 0.00 | argmax | r63a 99.9% / +0.135 |
| goriya | 96.1% / +0.187 / 2.03 | 96.4% / +0.138 / 1.94 | argmax | r61-dg 93.0% / +0.266 |
| darknut | 98.4% / +0.370 / 0.24 | 98.8% / +0.335 / 0.24 | argmax | r63a 97.4% / +0.489 |
| moldorm | 88.3% / +0.172 / 3.62 | 89.7% / +0.093 / 3.57 | argmax | r63b 87.0% / +0.250 |
| stalfos-keese-gel | 99.8% / −0.054 / 0.01 | 99.8% / −0.056 / 0.00 | argmax | r62-fixed 99.9% / −0.073 |
| d2-rope | 98.9% / −0.262 / 0.00 | 98.7% / −0.268 / 0.00 | argmax | r63a 98.9% / −0.256 |
| wallmaster | 100.0% / +0.000 / 0.00 | 100.0% / +0.000 / 0.00 | argmax | r61-dg 100.0% / +0.000 |
| boss-aquamentus | 99.3% / −0.975 / 0.00 | 100.0% / −1.023 / 0.00 | argmax | r61-dg 100.0% / −1.000 |
| boss-dodongo | 99.3% / −0.938 / 1.87 | 98.3% / −0.900 / 1.83 | argmax | r63b 100.0% / −1.005 |
| equipment-room | 98.5% / +0.244 / 0.04 | 99.0% / +0.170 / 0.02 | argmax | r63a 99.0% / +0.193 |
| trap-room | 99.5% / +0.282 / 0.41 | 99.7% / +0.266 / 0.38 | argmax | r63a 99.5% / +0.263 |
| zol | 99.8% / −0.221 / 0.04 | 97.2% / −0.217 / 0.00 | sample | r62-fixed 99.7% / −0.265 |
| boss-manhandla | 93.3% / +0.448 / 4.01 | 95.0% / +0.238 / 3.74 | argmax | r63b 96.3% / +0.010 |

Clearly better: goriya, darknut, moldorm (the bomb families). About level: the rest of the dungeon families and
most of the overworld. Worse: ow-lynel (+0.09 hearts lost), ow-moblin (+0.02, −0.7 points), boss-manhandla
(+0.23 hearts, −1.3 points) and boss-dodongo (98.3% argmax, 1.83 bombs per fight against 2.5 in round 12).
The whole team was promoted in the worktree's models dir (`../triforce2-bombs/models`; main's models untouched),
eval.json = the chosen mode's evaluation (the round's own is kept as `eval-round.json`).

## 4. Chains, 1000 runs per segment

`RUNS=1000 ACT=model TAG=r66-1k PARALLEL=20 scripts/experiments/r56_chains.sh run` in the worktree (its promoted
team), against the round 18 team (`r63-mix1k`) and the team before round 17 (`old1k`, on its own code), arrivals as
a share of runs that reach each target:

| segment | old1k | r63-mix1k | r66-1k |
|---|---:|---:|---:|
| D2 | 73.1% | 87.6% | 79.2% |
| after D2 | 80.9% | 81.9% | 85.9% |
| D1 | 97.4% | 97.8% | 94.9% |
| after D1 | 100.0% | 100.0% | 100.0% |
| D3 | 23.8% | 40.6% | 41.9% |
| full → D2 | 91.2% | 98.1% | 97.7% |
| full → after D2 | 56.6% | 62.2% | 59.1% |
| full → D1 | 44.2% | 49.3% | 50.4% |
| full → after D1 | 28.8% | 29.2% | 34.6% |
| full route (D3 triforce) | 2.6% | 10.6% | 11.6% |

Where the D2 and D1 segments end before their targets (runs, r63-mix1k → r66-1k): D2 moldorm 2:0x3e deaths 33 → 71,
goriya 2:0x5e 17 → 34 and 2:0x1e 53 → 62, d2-rope 2:0x4e 4 → 19; D1 `stuck` (argmax loops) goriya 1:0x44 8 → 17,
stalfos-keese-gel 1:0x33 0 → 13 and 1:0x53 0 → 10. Over all segments the `stuck` endings were darknut 521,
stalfos-keese-gel 88, goriya 58, d2-rope 43 (the round 17 rule: more than 5 means sampling).

Two causes:

1. **Dungeon 2 has no bombs.** The route carries none until Dodongo (every D2 start state has 0), but training drew
   0–8 bombs every reset, so the D2 families met a bombless room once in nine resets. Evaluated with 0 bombs
   (argmax, 100 per start state): moldorm 89.7% → 83.3% and −0.09 → −0.40 hearts; goriya 96.4% → 91.1% and its D2
   rooms 92/93% → 82/79% (round 18's goriya: 93.0% overall without bombs). Fix: `ram_randomization.bombs_share:
   0.5` (every family): half the resets keep the start state's own count (`74b0a15`), and goriya, moldorm and d2-rope
   continue 4M steps on it (`r67-share`).
2. **Argmax loops.** darknut and stalfos-keese-gel go to sampling (as in round 18); goriya and d2-rope are measured
   again after their continuations.

## 5. Continuations on `bombs_share: 0.5` (`r67-share-a`, `r67-share-b`)

`train_round.py --steps 4000000 --parallel 8 --train-args "--load {name} --stop-success 0.99 --sil-coeff 0.1"`, two
streams. Argmax, 100 episodes per start state, seed 0; hearts lost per episode:

| model | normal resets | 0 bombs | 0 bombs, D2 rooms |
|---|---|---|---|
| goriya r61-dg (round 18, no bombs) | 93.0% / 0.27 | | 0x1e 83.0%, 0x5e 85.7% |
| goriya r65-pilot-a | 96.4% / 0.14 | 91.1% / 0.38 | 0x1e 82.0%, 0x5e 78.7% |
| goriya r67-share-a | 94.8% / 0.19 | 93.1% / 0.29 | 0x1e 84.3%, 0x5e 85.0% |
| moldorm r63b (round 18, no bombs) | 87.0% / 0.25 | | |
| moldorm r65-pilot-b | 89.7% / 0.09 | 83.3% / 0.40 | |
| moldorm r67-share-b | 86.0% / 0.21 | 83.3% / 0.34 | |

goriya recovered its bombless D2 rooms; moldorm did not (4M steps, no early stop). d2-rope `r67-share-b` stopped early
at 98% (sample 98.3%, argmax 98.6%, as r66-dg). A moldorm continuation with bombs in a fifth of resets (`r68-moldorm`)
was stopped when Lee's 0x5e question (§6) superseded it.

## 6. Clearing 2:0x5e for bomb drops (Lee)

Lee: the Red Goriya room before the Moldorm can drop bombs if it is cleared, and early in a run a high-variance play
is fine. Experiment (branches `exp-clear5e`, `exp-sword-moldorm`): a new route rule `treasure: clear` (fight until
the room is empty, no correct exit before; `3c56836`) on 0x5e, goriya continued 4M steps on it (`r68-clear5e`:
sample 95.4% / 0.16 hearts, argmax 95.3% / 0.14; 0x5e 86–87%), against a baseline that runs through 0x5e with a
sword-only moldorm (`r68-sword`: r67-share-b continued 4M steps with `bombs: never` and no bomb randomization; sample
86.3%, argmax 85.7%, 0.32 hearts lost). Chains of 1000 runs from the D2 entrance (`--from 2_7ds_0 --stop-at 2:0x2e`,
`--checkpoint` per plan, `/tmp/moldorm_cut.py`; bombs carried = 4 per pickup − placed, from 0):

| plan | reach 0x2e | left 0x5e | hearts out of 0x5e | with bombs out of 0x5e | Moldorm won (of arrivals) | hearts out of 0x3e | ≤ 1 heart | ≥ 2.5 hearts | with bombs out of 0x3e |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| round 18 team (r63-mix1k, reference) | 94.5% | 98.2% | 2.29 | 0% | 96.6% | 2.29 | 10.4% | 58.2% | 0% |
| r66 team (reference) | 87.3% | 96.5% | 1.98 | 0% | 92.5% | 2.18 | 15.2% | 53.3% | 0% |
| baseline: through 0x5e, sword-only moldorm | 91.9% | 97.3% | 2.11 | 0.1% | 96.5% | 2.32 | 9.7% | 61.0% | 0.1% |
| clear 0x5e, goriya argmax, moldorm r67 | 90.2% | 95.0% | 2.44 | 42.2% | 95.3% | 2.31 | 12.7% | 59.6% | 20.2% |
| clear 0x5e, goriya sample, moldorm r67 | 89.9% | 96.9% | 2.41 | 39.6% | 93.7% | 2.32 | 13.7% | 61.9% | 18.7% |

- Clearing 0x5e leaves it with *more* hearts (2.44 vs 2.11: the kills drop hearts too) and bombs in about 40% of runs
  (414–445 bomb pickups per 1000 runs). Goriya argmax looped in 25 runs (2000-step room budget); sampling: 1.
- The Moldorm with bombs: 96.3–98.5% won, 2.43–2.48 hearts out; without: 92.1–93.1%, 2.21. Bombless, the r67
  moldorm is below the sword-only one (96.5%).
- 0x4e (d2-rope) deaths: baseline 17, clear 3 and 11 (Link arrives healthier).

Decision (Lee's rule: keep the clear plan within 4 points of the baseline, judged on the Moldorm room's exit: hearts
and bombs): **clear 0x5e**. Reaching 0x2e is 1.7–2.0 points below the baseline, hearts out of the Moldorm room are
the same (2.31–2.32; ≥ 2.5 hearts 60–62% in both; ≤ 1 heart 3–4 points more often), and one run in five leaves it
with bombs. Promoted: goriya `r68-clear5e` (sample), moldorm `r67-share-b` (argmax; yaml back to `bombs_share:
0.5`).

## 7. Final team and chains

Promoted in `../triforce2-bombs/models` (act mode in brackets): the 8 overworld families `r66-ow` (argmax);
stalfos-keese-gel `r66-dg` (sample), d2-rope `r67-share-b` (sample), zol `r66-dg` (sample), wallmaster,
boss-aquamentus, boss-dodongo, equipment-room, trap-room and boss-manhandla `r66-dg` (argmax); goriya `r68-clear5e`
(sample), darknut `r65-pilot-b` (sample), moldorm `r67-share-b` (argmax). darknut, stalfos-keese-gel and d2-rope sample
because of their argmax loops in §4 (round 17's rule).

`RUNS=1000 ACT=model TAG=r69-1k PARALLEL=20 scripts/experiments/r56_chains.sh run` (30 minutes), arrivals:

| segment | r63-mix1k (round 18) | r66-1k (§4) | r69-1k (final) |
|---|---:|---:|---:|
| D2 | 87.6% | 79.2% | 82.1% |
| after D2 | 81.9% | 85.9% | 85.9% |
| D1 | 97.8% | 94.9% | 98.4% |
| after D1 | 100.0% | 100.0% | 100.0% |
| D3 | 40.6% | 41.9% | 39.5% |
| full → D2 | 98.1% | 97.7% | 97.7% |
| full → after D2 | 62.2% | 59.1% | 67.6% |
| full → D1 | 49.3% | 50.4% | 55.2% |
| full → D3 | 29.2% | 34.6% | 36.3% |
| full route (D3 triforce) | 10.6% | 11.6% | 10.3% |

Failures by room (runs per 1000, r63-mix1k → r69-1k). D2 segment: 2:0x1e goriya deaths 53 → 66, 2:0x3e moldorm 33 →
55, 2:0x5e goriya 17 → 38 (it now fights the whole room), 2:0x4e d2-rope 4 → 6. The full route converts D2 better
than before (97.7% → 67.6% against 98.1% → 62.2%), so the segment's single entrance state understates the change.
D3 segment: 3:0x59 darknut deaths 153 → 132 and `stuck` 10 → 74, all on its kill-all visit from 0x49 with no bomb
placed (the reserve of 5 masks bombs while Link holds 5 or fewer); 3:0x5b deaths 176 → 127; 3:0x69 103 → 66; 3:0x4d
Manhandla 84 → 123 (§3's single-room evaluation already showed it: 95.0% and 0.24 hearts lost vs round 18's 96.3%
and 0.01). `stuck` endings over all segments: darknut 183 (argmax in §4: 521), trap-room 3,
ow-tektite 3, boss-dodongo 2.

Not fixed (logged in backlog.md): the 0x59 stalls, Manhandla's deaths, the D2 segment's goriya 0x1e deaths, and the
single-room regressions of ow-lynel, ow-moblin and boss-dodongo (§3), none of which shows in the chains as more
than noise.
