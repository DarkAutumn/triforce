# Round 11: power-on RAM $00 and a fresh retrain (Nov 8)

Lee (Nov 8): use $00 for the power-on RAM fill, regenerate everything, then retrain the whole team from scratch up
to 8M steps per family, through `scripts/train_round.py` with `logs/` for the training GUI, 16 workers.

## 1. Regeneration

- **zcore** (`cade3c0`): `POWER_ON_RAM` $00, state file version 6 (older states are refused, not silently loaded).
  The twelve TetaNES test ROMs that needed zero-filled RAM pass and are no longer ignored. The scripted corpus movies
  re-record byte-identical (they store inputs; every scripted checkpoint landed on the same frame); the old goldens
  diverged only in the RAM and CHR domains at frame 0; the 7 committed goldens and a full-corpus baseline
  (`target/ladder/zero`, 63 movies) were regenerated, 0 diverge.
- **Start states** (`c53ff6a`, `target/zero/build_states.sh`): start + 3 route chains + D1 bow rooms + fairy route +
  D2 0x6d, as plan step 18b. The same three D2 steps (2_2es_0, 2_0es_1, 2_0es_2) needed the 12,000-iteration retry
  as last time, and the fairy lane 0x5c backtracked as before. Audit: 181 states, 0 problems.
- **Trajectories** (`target/zero/explore.sh`, gitignored; the $FF set is in `target/trajectories-ff`): 84/84 solved;
  goriya 2_1es_0..2 and d2-rope 2_3es_0/2 at 12,000 iterations. ow-zora and ow-moblin used exactly the env steps of
  the last exploration.
- Tests: cargo 561 passed (the 12 test ROMs included), pytest 229 + the slow state/explore tests.
- **Old team on the new states** (120 runs per segment, seeds 100–219): D2 74, post-D1 main 43, post-D1 fairy 61,
  full main 27, full fairy 46: exactly round 10's final numbers, so the regenerated states play the same game.

## 2. Fresh retrain (`r44-fresh`)

`scripts/train_round.py --round r44-fresh --steps 8000000 --parallel 16 --no-promote --chain-runs 0 --train-args
"--stop-success 0.99"`; GUI log folder `logs/261108_019_r44-fresh/`. Every trainable family from scratch, in route
order, with the canonical yamls (round 10 §8–§10: `obs.target` everywhere, item targets in the dungeon families).
Early stop only at 0.99 balanced success for 5 updates (the round default 0.95 would stop before health is refined;
round 10 §8). Not promoted automatically: the new team must beat the old one on the same chains first.

**Stopped and resumed (Lee, Nov 8).** The first launch (`logs/261108_019_r44-fresh/`) trained ow-start (early stop
at 0.99 after 180k steps; eval 100%), ow-octorok (8M; eval 98%) and ow-zora (8M; evaluated separately after the
stop). Lee then had the round stopped right after ow-zora to **split the Moldorm room 2:0x3e out of d2-rope**
into its own specialist `moldorm` (`0acabad`: yaml, game.yaml tag, routes.yaml family and step; its three 2_3es
trajectories moved to `trajectories/moldorm`; until its retrained model exists, the promoted moldorm is a copy of
d2-rope r40-obs-s41). The round resumed for the other eight families (stalfos-keese-gel restarted from scratch):
`--specialists stalfos-keese-gel,goriya,wallmaster,boss-aquamentus,ow-leever-peahat,ow-moblin,d2-rope,moldorm`,
same settings, GUI log folder `logs/261108_020_r44-fresh/`.

**Second stop: per-room evaluation (Lee, Nov 8).** After goriya (8M) the round was stopped again to finish the
per-room-visit evaluation started in another session (`cde10ca`: `zelda::visits`, eval `format: 2`) and the
`zelda-map.py` room map (`6eb8a1c`). Evaluation now runs 100 episodes per start state (`20b2074`; 60 for the
re-evals below). Re-evaluated with format 2 at 60 per state: all 11 promoted models and the finished r44 runs
(ow-start 99.7%, ow-octorok 98.7%, ow-zora 97.9%, stalfos-keese-gel 98.1%); goriya r44 at 100 per state: 88.6%,
health change -0.51 (promoted goriya 79.0%, -0.83). The round resumed from wallmaster (wallmaster,
boss-aquamentus, ow-leever-peahat, ow-moblin, d2-rope, moldorm), GUI log folder `logs/261108_021_r44-fresh/`.
`zelda-map.py --run-id r44-fresh` shows the r44 runs where they exist, else the promoted models.
