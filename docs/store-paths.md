# The paths the documents cite, and where each is now (Store41, Nov 24)

Every concrete `models/...`, `trajectories/...`, `triforce/states/harvested...` and `target/...` path cited in
`docs/`, `AGENTS.md` and `README.md` (on `main` at `b66c9322`), looked up on Nov 24 in the shared store
(`~/triforce-shared`: `store.db`'s runs and filed records, `legacy/import-261123/`, the data snapshots), in the
archive (`/backup/triforce`: `models/runs/`, `models/sources/`, `worktree-data/`) and in the main checkout. The forms
(`models/<family>/<run>`, `trajectories/<family>/<track>` ...) and how to read any old path are in
`docs/data-layout.md` "The reconciliation of the main checkout". The documents themselves are history and keep
their paths. 256 paths; 111 forms with placeholders are not listed. Look a run up with `triforce store runs
FAMILY/RUN`, its evaluations and `act` with `triforce store outputs --run FAMILY/RUN`, a record with `triforce store
outputs --kind record`.

## `models/`

| cited path | now | cited in |
|---|---|---|
| `models/boss-aquamentus/r30-exit` (and `/config.yaml`) | store run `boss-aquamentus/r30-exit` | round7.md |
| `models/boss-aquamentus/r30-exit/eval.json` | store run `boss-aquamentus/r30-exit`; `eval.json` filed by job | round7.md |
| `models/boss-aquamentus/r47-base-dg/ckpt.pt` | store run `boss-aquamentus/r47-base-dg` | all-rust.md |
| `models/boss-aquamentus/r66-dg/ckpt.pt` | store run `boss-aquamentus/r66-dg` | runs250/backtest-trees.md |
| `models/boss-aquamentus/r71-e/ckpt.pt`, `/act` | store run `boss-aquamentus/r71-e`; `act` filed by job | runs250/backtest-trees.md |
| `models/boss-aquamentus/r76-b/ckpt.pt` | store run `boss-aquamentus/r76-b` | runs250/backtest-trees.md |
| `models/boss-manhandla/r32d3-manh/ckpt.pt` | store run `boss-manhandla/r32d3-manh` | round35.md |
| `models/d2-rope/r34-east-key` | store run `d2-rope/r34-east-key` | round7.md |
| `models/d2-rope/r35-east-curriculum` | store run `d2-rope/r35-east-curriculum` | round7.md |
| `models/darknut/r32d3-dark2-s44/ckpt.pt` | store run `darknut/r32d3-dark2-s44` | round34.md |
| `models/darknut/r34d3-dark-s341` (and `/ckpt.pt`) | store run `darknut/r34d3-dark-s341` | round34.md |
| `models/darknut/r34d3-dlp-s353` (and `/ckpt.pt`) | store run `darknut/r34d3-dlp-s353` | round34.md |
| `models/darknut/r78-dark37-s85` | store run `darknut/r78-dark37-s85` | round30.md |
| `models/goriya/r31-block1` | store run `goriya/r31-block1` | dependencies.md |
| `models/goriya/r31-quiet005/config.yaml` | store run `goriya/r31-quiet005` | round7.md |
| `models/goriya/r32d1-goriya` | store run `goriya/r32d1-goriya` | round38.md |
| `models/goriya/r38shift-pre-w8` | archive only: `/backup/triforce/models/runs/goriya/r38shift-pre-w8` | round38.md |
| `models/moldorm/r75-f-b` | store run `moldorm/r75-f-b` | round30.md |
| `models/ow-lynel/r74-lynel` | store run `ow-lynel/r74-lynel` | round27.md |
| `models/ow-octorok/r31-oct-s97/ckpt.pt` | store run `ow-octorok/r31-oct-s97` | round31.md |
| `models/ow-octorok/r74-leg2/ckpt.pt` | store run `ow-octorok/r74-leg2` | round31.md |
| `models/ow-zora/r76-zora-arrows` | store run `ow-zora/r76-zora-arrows` | round29.md |
| `models/r36obs-pre` | archive only: `/backup/triforce/models/runs/r36obs-pre` | round38.md |
| `models/r38head-hit/all` | store run `r38head-hit/all` | round38.md, round38-one.md, round39.md |
| `models/r38shift/pre-w4` | archive only: `/backup/triforce/models/runs/r38shift/pre-w4` | round38.md |
| `models/r38shift/pre-w8-frozen` | archive only: `/backup/triforce/models/runs/r38shift/pre-w8-frozen` | round38.md |
| `models/vire/r77-b` | store run `vire/r77-b` | round33.md |
| `models/wizzrobe/r30w-wj-s89` | store run `wizzrobe/r30w-wj-s89` | round33.md |
| `models/wizzrobe/r77-wizz` | not in the store or the archive (removed before either) | round27.md |
| `models/wizzrobe/r78-wizz-s78` | not in the store or the archive (removed before either) | round27.md |
| `models/zol` | store family `runs/zol/` (21 runs) | data-layout.md, round40.md, lab.md |
| `models/zol/r79b-zol` | store run `zol/r79b-zol` | lab.md |
| `models/r29-linger-baseline` | 13 records filed by the import, 4 files in `legacy/import-261123/models/r29-linger-baseline/` | round7.md |
| `models/r29-linger-baseline/aquamentus-through-exit.json`, `/boomerang-decode-probe.json` | records filed by the import | round7.md |
| `models/r32b-octorok-evidence` | 49 records filed by the import | round7.md |
| `models/r33-cutoff-evidence` | 63 records filed by the import | round7.md |
| `models/r34-east-key-evidence` | 36 records filed by the import | round7.md |
| `models/r35-east-curriculum-evidence` | 71 records filed by the import, 5,140 files in `legacy/import-261123/models/r35-east-curriculum-evidence/` | round7.md |
| `models/r35-east-curriculum-evidence/retired-default-d2-curricula` | 10 records, 1,449 files in `legacy/import-261123/models/...` | round7.md |
| `models/r35-east-curriculum-evidence/retired-moblin-curricula` | 7 records, 303 files in `legacy/import-261123/models/...` | round7.md |
| `models/r36-observation-full-retrain-evidence` | 155 records filed by the import, 6,557 files in `legacy/import-261123/models/r36-observation-full-retrain-evidence/` | round7.md |
| `models/r36-observation-full-retrain-evidence/logs/schema1` | 24 records filed by the import | backlog.old.md |
| `models/r31-block-evidence/{chain-stall-replays, corner-replays, corner-replays-quiet005, promoted-eval, seed-after, seed-before}.json` | records filed by the import | round7.md |
| `models/r32-nearby-evidence/{comparison, evaluations, protocol}.json` | records filed by the import | round7.md |
| `models/r33-cutoff-evidence/{bootstrap-smoke, config-override-smoke, current-baseline-cutoff-exposure, shutter-objective-smoke, terminal-health-smoke}.json` | records filed by the import | round7.md |
| `models/r34-d2-key-evidence/{builder-after, builder-before, builder-prototype-after, inventory-before}.json` | records filed by the import | round7.md |
| `models/r3-baseline-summary.json`, `models/r3c-curriculum-summary.json` | records filed by the import | round3.md |
| `models/r30-aquamentus-exit-old.json` | record filed by the import | round7.md |
| `models/round1-chain-summary.json`, `models/round1-chain.jsonl`, `models/round1-summary.json` | records filed by the import | round1.md |
| `models/runs`, `models/runs.jsonl`, `models/sources`, `models/files.tsv` | the archive's own `/backup/triforce/models/...`, not a checkout's | AGENTS.md, architecture.md, data-layout.md |

## `trajectories/`, `triforce/states/harvested/`, `target/demos/`

| cited path | now | cited in |
|---|---|---|
| `trajectories/darknut`, `like-like`, `moldorm`, `ow-moblin`, `ow-start`, `ow-zora`, `pols-voice` | snapshot `trajectories/261122-52fb5b17/<family>` (main's folder of Nov 23) | decisions.md, round38.md, round11.md, round7.md, backlog.old.md, README.md |
| `trajectories.prev-r60/-r70/-r71` | snapshots `trajectories/261109-2dfee7b9`, `261113-2ebb5ce2`, `261113-578619ff` | data-layout.md |
| `trajectories/curricula`, `ow-moblin/ow-moblin`, `ow-start/0_2fcave_0..2`, `r21-trap22-north`, `r21-trap22-south`, `r21-vire50`, `r21-vire50-explore` (and `/vire50-none-nocur`), `r30-exit/boss-aquamentus`, `r34-east-key` | in no snapshot and not in the archive (removed before the store) | routes/d6.md, round6.md, round30.md, round21-trap-room.md, round21-vire-50.md, round7.md |
| `triforce/states/harvested/boss-manhandla`, `darknut`, `moldorm`, `ow-lynel`, `ow-zora` | snapshot `harvested/261123-ad612e94/<family>` (main's folder of Nov 23) | round30.md, round27.md, AGENTS.md, round26.md, round29.md, route-team.md |
| `target/demos` and `target/demos/{r27l/lynel74-0a, r28-demos/darknut-3e, r28-demos/darknut-5e, r28-demos/whirlwind, r29-demos/rock-wall, r31/mouth-37, r78dk/demos37}.npz` | snapshot `demos/261122-453e1975/<path>` (main's folder of Nov 22) | AGENTS.md, data-layout.md, round30.md, round31.md |
| `target/r27-demos/{bomb28, plan38-3a}.npz`, `target/r28-demos/whirlwind.npz`, `target/r29-demos` and its `{ladder-17, rock-wall, tree, wizzrobe-07}.npz` | a file of the same name in snapshot `demos/261122-453e1975/r27-demos/`, `r28-demos/`, `r29-demos/` | round27.md, round29.md, data-layout.md, backlog.old.md, route-team.md |

## `target/` (scratch: main's stays in main)

In main's `target/` (not moved by the reconciliation): `bench-cache`, `bench-runs`, `converted-models`,
`corpus-runs`, `dodongo` (`README.md`, `geometry.py`, `how_killed.py`, `swallow-tracks`), `harvested.prev-1.9.0.1`,
`l6` (`new-metrics.jsonl`, `old-metrics.jsonl`), `ladder/hardware`, `ladder/zero`, `lynel` (`explore/ow-lynel`,
`spawnstates`), `opt/gate.sh`, `overworld/render.py`, `owner-review/CHECKIN.md`, `owner-review/r38-moldorm-key.md`,
`perf/parallel/nov7`, `r23-pilot-clipson`, `r32-walks/states`, `r35-wrap/golden`, `r35-wrap/single`, `r39`,
`r43/final`, `r47/rooms-full-r49.txt`, `r47/rooms.py`, `r56-chains` (`d3-100-land.jsonl`, `r32land-1k-d3.jsonl`,
`r32land-1k-room-stats.json`, `r35land-1k-{d3,d5,d7,full}.jsonl`, `r35land-plan-full.jsonl`,
`r35nobag-1k-{d5,full}.jsonl`, `r35nobag-plan-d5.jsonl`, `r72-1k-both-room-stats.json`), `r61/b1-yardstick.json`,
`r61/promoted-before-r61.txt`, `r61/r61-modes.txt`, `r72/reeval`, `review-guide.md`, `review-ledger.md`,
`review-progress.md`, `runs250/2.0.0.14`, `states-milestone1`, `states-pre-bow`, `step17`, `step18/build_states.sh`,
`step18/logs`, `trajectories-ff`, `trajectories-milestone1`, `trajectories-pre-bow`, `traps`, `traps/path.py`,
`worktrees` (`r35-guard`, `r35-guard/r35`), `zero/build_states.sh`, `zero/explore.sh`.

In the archive, a removed worktree's `target/` (`/backup/triforce/worktree-data/<date>_<branch>/target/...`):

| cited path | now | cited in |
|---|---|---|
| `target/eval250-smoke` | `worktree-data/2026-11-24_eval250-v2.0.0.3/target/eval250-smoke` and 25 more back-test trees | runs250/backtest-trees.md |
| `target/golden-r39d5b` | `worktree-data/2026-11-23_r39-dark5b/target/golden-r39d5b` | round38.md |
| `target/proof` | `worktree-data/2026-11-23_r40-store-proof/target/proof` | decisions.md |
| `target/r38/table2013.py`, `target/r38/yaml/ow-lynel-late.yaml` | `worktree-data/2026-11-24_r38-one/target/r38/...` | round38-one.md, decisions.md |
| `target/r40/bench-final.jsonl` | `worktree-data/2026-11-23_r40-gen-obs/target/r40/bench-final.jsonl` | round40.md |
| `target/r40f`, `target/r40f/base-d3-1k.jsonl` | `worktree-data/2026-11-23_r40-fireline/target/r40f/...` | round40.md |
| `target/route-tracks` | `worktree-data/2026-11-24_r40-trainer/target/route-tracks` and 6 more | data-layout.md, lab.md |
| `target/select/` | `worktree-data/2026-11-23_r40-store-after/target/select/` | data-layout.md |
| `target/trajectories` | `worktree-data/2026-11-24_r40-states/target/trajectories` | data-layout.md |

Not in main and not in the archive (a worktree's scratch, removed before `tidy_worktrees.sh` kept worktree data, or
still in a live worktree): `agent-pool`, `agent-tests2`, `curricula.sh`, `entry-states/{d4-cN, r21-ladder-cN,
r22-d4-plain, r26-d7/c, r26-d7/sources, r27-d6, r28-d8, r29-d9}`, `flag`, `golden-S`, `merge/{gpu-a.sh,
one-chains.sh, student-manifest.json}`, `r23-merge-check`, `r23-pilot` (`chains`, `report-rooms.md`),
`r23-pilot-clipsoff`, `r23-quick`, `r23prep/chains`, `r31-chains`, `r32-team/golden1`,
`r32-turn/screen.sh`, `r34-d4/{final.sh, states-4_71s}`, `r34/{demos/darknut-lastpass.npz, final-s353-d5.jsonl,
harvest-d3.jsonl, joint-d5.jsonl, plan300-d3.jsonl, plan300-d5.jsonl, train/darknut-lastpass.yaml}`,
`r35/{base-d3.jsonl, harv-d3.jsonl, harvest-models, harvest-models/boss-manhandla, harvest-plan, manh-fight,
plan-d3.jsonl}`, `r36/bench.jsonl`, `r37/{bench/r36w768e2-vs-ref.md, one-chains.sh}`, `r38h/bench-all-table.md`,
`r40/chains/replays-s40N`, `r56-chains/{r76-base-room-stats.json, r77-base-room-stats.json, r77a-1k-d6.jsonl,
r77a-1k-post-d7.jsonl, r79t5-1k-d9.jsonl}`, `r79/r79-plan-d6.jsonl`, `r79b/{replays-d9/00000009-win.tfreplay,
replays-postd8/00000012-win.tfreplay, tracks-dropped/lanmola}`, `r79c/{replays-d9/00000000-win.tfreplay,
replays-postd8/00000025-win.tfreplay}`, `replays/{golden, po, state}`, `traj`, `zasm/miniasm.py`.
