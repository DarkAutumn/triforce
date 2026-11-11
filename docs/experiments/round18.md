# Round 18: back to the previous team's level (hearts, not just success)

Lee, Nov 11: the r61 team (round 17) lost Dungeon 1 (115 → 80 of 120). Work on the stalfos-keese-gel model's
problems and whatever else they affect; heart loss can improve even when a room is a "success" (goriya should be
able to reach −0.1 to −0.2 hearts per room with enough training or shaping). Get every chain segment back within 5%
of the previous team before more experiments. Chains are measured with 1000 runs per segment from here on (Lee).

## Summary

- **Cause of the D1 drop:** stalfos-keese-gel early-stopped after 80 iterations (4 minutes) at a confirmed 99% success
  while its hearts per episode were still improving fast (training: −0.5 → −0.1 over those 80 iterations). In the
  full route it lost 0.18 hearts per room visit where the old model gained 0.05, and D1 ends in its rooms.
- **Fix in the early stop** (`0b1a6d4`): a confirmed stop also needs the hearts lost per episode (start − end health
  in the confirmation evaluation) to have stopped improving: not better than every earlier confirmed check by more
  than max(0.02, 2 SE), checks at least 50 iterations apart (`--stop-hearts-delta`, `--stop-hearts-gap`).
- **Continuations:** stalfos-keese-gel and zol 500 more iterations each (`r62-fixed`): stalfos-keese-gel 99.1% →
  99.9% success, hearts lost per episode +0.10 → −0.07 (now a net gain); zol 98.8% → 99.8%, −0.07 → −0.26. Every
  other family continued with the new stop rule (`r63a`, `r63b`, up to 4M steps); 13 improved and were promoted.
- **Result, 1000 runs per segment, against the previous team on its own code and states** (§3): every segment at or
  above the previous team; the full route to the D3 triforce 2.6% → 10.6%; D3 23.8% → 40.6%.
- **Moved to the backlog as projects of their own (Lee):** wallmaster no longer collects the hearts its kills drop
  (old model +0.33 per room, +0.85 in the full route at low health; new 0.0: under γ = 0.99 rushing out is worth
  more than fighting for the drops, the old model had stopped training before it unlearned that, and
  `exit_health_weight 1.0` doesn't bring it back), and goriya's heart loss (+0.26 per room; Lee's target is a net
  gain of 0.1–0.2; more training of the same objective doesn't move it).
- **Tooling (Lee):** `--device` defaults to `cuda` (CPU training is opt-in; a CPU update next to a running training
  took 80–170 s per iteration); the training GUI shows eval-chain throughput and a CPU/GPU/load status bar.

## 1. stalfos-keese-gel and zol

| model | act | success | hearts lost / episode |
|---|---|---:|---:|
| stalfos-keese-gel r61-dg (80 iterations) | sample | 99.1% | +0.103 |
| stalfos-keese-gel r61-dg | argmax | 95.8% | +0.094 |
| stalfos-keese-gel r62-fixed (+500) | sample | 99.9% | −0.073 |
| stalfos-keese-gel r62-fixed | argmax | 99.7% | −0.084 |
| zol r61-dg2 (96 iterations) | sample | 98.8% | −0.069 |
| zol r61-dg2 | argmax | 93.1% | −0.126 |
| zol r62-fixed (+500) | sample | 99.8% | −0.256 |
| zol r62-fixed | argmax | 99.7% | −0.265 |

(100 episodes per start state, seed 0; hearts lost is start − end health, negative = net gain.) Training health per
50 iterations, stalfos-keese-gel: −0.05, +0.02, +0.04, +0.055, +0.06, +0.07, +0.07, +0.07, +0.08, +0.08 (net
change, positive = gain): most of the gain came in the first 200 iterations after the old stop. The new stop rule
would have stopped around there. stalfos-keese-gel acts by sampling (argmax looped 47 times in 6000 chain runs);
zol by argmax.

## 2. Continuations with the hearts-aware stop (r63)

`train_round.py --steps 4000000 --train-args "--load {name} --sil-coeff 0.1 --stop-success 0.99"` for the other 18
families. Promoted where the new run is better (success first, then hearts): darknut, equipment-room, trap-room,
d2-rope, ow-lynel, ow-boulder, ow-octorok, ow-zora, boss-manhandla, moldorm, ow-tektite, boss-dodongo, ow-leever.
Kept: goriya, wallmaster, ow-start, ow-moblin, boss-aquamentus (no better). Single-room numbers, each run in the act
mode the table names (100 episodes per start state, seed 0; hearts lost is start − end health):

| family | r61 (act) success / hearts | r63 (act) success / hearts | promoted |
|---|---|---|---|
| goriya | 93.0% / +0.266 (argmax) | 92.7% / +0.262 (argmax) | r61 |
| darknut | 96.4% / +0.536 (sample) | 97.4% / +0.489 (sample) | r63a |
| equipment-room | 99.0% / +0.227 | 99.0% / +0.193 | r63a |
| trap-room | 99.7% / +0.350 | 99.5% / +0.263 | r63a |
| d2-rope | 98.3% / −0.248 (sample) | 98.9% / −0.256 (sample) | r63a |
| ow-lynel | 98.9% / +0.131 | 99.2% / +0.132 | r63a |
| ow-boulder | 99.9% / +0.170 | 99.9% / +0.135 | r63a |
| ow-octorok | 98.6% / +0.091 | 98.8% / +0.089 | r63a |
| ow-zora | 99.3% / +0.164 | 99.3% / +0.153 | r63b |
| boss-manhandla | 95.7% / +0.057 | 96.3% / +0.010 | r63b |
| moldorm | 86.7% / +0.348 | 87.0% / +0.250 | r63b |
| ow-tektite | 99.4% / +0.008 | 99.3% / −0.031 | r63b |
| boss-dodongo | 99.0% / −0.927 | 100.0% / −1.005 | r63b |
| ow-leever | 100.0% / +0.030 | 100.0% / −0.103 | r63b |
| boss-aquamentus | 100.0% / −1.000 | 99.3% / −1.055 | r61 |
| ow-moblin | 98.4% / +0.119 | 98.2% / +0.108 | r61 |
| wallmaster | 100.0% / +0.000 | 100.0% / +0.000 | r61 |
| ow-start | 100.0% / 0.000 | 100.0% / 0.000 | r61 |

Most families moved by a few hundredths of a heart: the stop rule matters for the families that stopped within
minutes, not for the ones that already used their budget. goriya stays at about +0.26 hearts per room (Lee's
target −0.1 to −0.2 is not reached by more training of the same objective; backlog).

## 3. Chains, 1000 runs per segment

The previous team (r56 + r57-sil-s60, its act modes) on its own code (`ab17fc5`) and states, models through
`/tmp/oldteam` (a models root with the old pointers and one-word act files); the current team on HEAD.

| segment | previous team | r61 (round 17) + r62 stalfos-keese-gel, zol | **final (r63 promotions)** | final / previous |
|---|---:|---:|---:|---:|
| D2 (`2_7ds_0` → 0:0x3c) | 73.1% | 84.3% | **87.6%** | 1.20 |
| after D2 (`0_3cc_0` → 1:0x73) | 80.9% | 77.3% | **81.9%** | 1.01 |
| D1 (`1_73s_0` → 0:0x37) | 97.4% | 97.4% | **97.8%** | 1.00 |
| after D1 (`0_37c_0` → 3:0x7c) | 100.0% | 100.0% | **100.0%** | 1.00 |
| D3 (`3_7cs_0` → triforce) | 23.8% | 27.3% | **40.6%** | 1.71 |
| full → D2 | 91.2% | 97.5% | **98.1%** | 1.08 |
| full → after D2 | 56.6% | 58.2% | **62.2%** | 1.10 |
| full → D1 | 44.2% | 44.3% | **49.3%** | 1.12 |
| full → after D1 | 28.8% | 25.1% | **29.2%** | 1.01 |
| full → D3 | 28.8% | 25.1% | **29.2%** | 1.01 |
| full route (D3 triforce) | 2.6% | 5.7% | **10.6%** | 4.08 |

(Tags: `target/r62-old/old1k-*`, `target/r56-chains/r62-a1k-*`, `target/r56-chains/r63-mix1k-*`; arrivals in order,
`/tmp/arrivals.py`.) Every segment is at or above the previous team. With 1000 runs a 95% interval is about ±3
points at 50% and ±1 point near 98%.

Act modes in chains: goriya loops by argmax (52 `stuck` per 6000 runs, more than the round 17 rule's 5 per 720), so
it was tried with sampling (`r63-mix2-1k`): D1 97.8% → 99.3%, but D2 87.6% → 81.3% (goriya's D2 rooms) and the full
route 10.6% → 9.8%. goriya stays argmax: the rule's loop count is a symptom, the chain is the measure. zol (argmax)
loops 29–38 times and darknut (sample) 34 times per 6000.

Final team: stalfos-keese-gel r62-fixed (sample), zol r62-fixed (argmax), darknut r63a (sample), d2-rope r63a
(sample), equipment-room, trap-room, ow-lynel, ow-boulder, ow-octorok r63a, ow-zora, boss-manhandla, moldorm,
ow-tektite, boss-dodongo, ow-leever r63b, goriya, wallmaster, boss-aquamentus r61-dg, ow-start, ow-moblin r61-ow
(argmax unless named).
