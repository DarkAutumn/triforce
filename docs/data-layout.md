# What a run reads and writes that git does not track (inventory, Nov 23)

Round 40, step 0 of the recipe work (the owner, through the lead: "a full, written layout of every file a run reads
or writes that git does not track, and each one's rules", before anything moves). Measured in the main checkout
`/home/leculver/git/triforce` on Nov 23 at commit `23412d57` (sizes by `du -sh`, files by `find -type f`). Line
numbers are that commit's. A claim not read from the code or the disk is marked [inference].

Two facts frame everything below:

- **A worktree holds the tracked files only.** Every row marked "shared by nature" is today re-created in each
  worktree by hand or by `triforce job setup` (`triforce/job.py:754-761`: symlinks `roms`, `trajectories`,
  `triforce/states/harvested`; never `models`). Nothing else checks that it was.
- **Most paths are `<this checkout>/<name>`**, computed in each module (`Path(__file__).parents[1] / ...` or
  `ROOT / ...`), with an environment override for a few: `$TRIFORCE_ROM`, `$TRIFORCE_STATES_DIR`,
  `$TRIFORCE_MODELS_DIR`, `$TRIFORCE_LOGS_DIR`, `$TRIFORCE_QUEUE_DIR`, `$TRIFORCE_BACKUP` (scripts only). There is
  no single answer to "where is the shared data".

## Summary

| thing | size (main) | nature | changed in place? | backed up |
|---|---|---|---|---|
| `roms/zelda.nes` | 136K, 1 file | shared (the game) | never | no (owner's copy) |
| `trajectories/` | 1.5G, 128,295 files | shared (data set) | yes: a family is re-explored over its old tracks | no |
| `trajectories.prev-r60/-r70/-r71` | 185M / 48M / 241M | shared, frozen (hand copies) | no | no |
| `triforce/states/harvested/` | 783M, 66,438 files | shared (data set) | yes: files added, removed, replaced on route changes | only a removed worktree's copy (`worktree-data`) |
| `triforce/states/harvested-rooms/` | 15M, 1,196 files | shared (the lab's bench) | added to | no |
| `models/` | 26G, 21,505 files; 34 families + 17 group/evidence dirs + 174 loose files | shared (the team of record) and per-run output mixed | yes: pointers, `act`, `ckpt.pt` (`--apply`, same-dir continuations), `eval*.json` | yes, `/backup/triforce/models` (when `backup_models.py store` is run) |
| `logs/` | 607M, 3,283 files | per checkout (GUI input), except test timing logs (main's) | appended | no |
| `target/` | 44G, 102,053 files | per checkout (scratch, build) but holds shared inputs (below) | yes | a removed worktree's, `worktree-data/` |
| `runs/` | 95M, 4 files | per checkout (default chain outputs) | appended | no |
| `replays/` | 57M, 691 files (61 tracked) | shared (golden and single runs to watch) | added to | yes, `/backup/triforce/replays` (compressed) |
| `.play/` | 28K | per checkout (one interactive session) | yes | no |
| `screens/` | 1.8M, 390 files | per checkout cache of `zelda-map.py` | yes | no |
| `golden-*.mp4` (repo root) | 2 files | owner's outputs | no | no |
| `triforce/_native.abi3.so` | 7.3M | per checkout (build) | rebuilt | no |
| `.venv/` | 5.6G | per checkout | rebuilt | no |
| `triforce/states/*.png`, `*.yaml`, `**/*.lock` | small | per checkout scratch / lock | yes | no |
| `~/.local/state/triforce/` | 4.2M (`queue.db`, WAL, `locks/`) | shared by nature (one machine queue) | yes | no |
| `/backup/triforce/` | 36G, 180,102 files | shared archive (own 8 TB disk) | added to; files are read-only hard links | it is the backup |

## Each thing

### `roms/zelda.nes`
- **What:** the iNES ROM (PRG SHA-1 checked by `tf`, `crates/tools/src/rom.rs`).
- **Read by:** the Python module, once per process (`crates/triforce-py/src/context.rs:53-81`, `$TRIFORCE_ROM` else
  `<root>/roms/zelda.nes`, `root` = the folder the built module sits in); `tf` (`crates/tools/src/rom.rs:8-27`,
  the cargo workspace's root at compile time); the Rust tests (`crates/test-rom/src/lib.rs:15-22`); the Python tests
  (`tests/conftest.py:30-38`); `scripts/level_rooms.py:152`.
- **Written by:** the owner, once. **In place:** never.
- **Absent:** loud at the first emulator use ("ROM not found: set TRIFORCE_ROM or create roms/zelda.nes"); the Rust
  tests fail unless `TRIFORCE_ALLOW_NO_ROM=1`.
- **Nature:** shared, immutable. **Backup:** none in `/backup/triforce` (it is the owner's dump).

### `trajectories/<family>/<track>/` (curriculum tracks)
- **What:** per seed state a track of savestates along a successful path (`trajectory.json` + `NNNN.state`), from
  `python -m triforce.explore`; rule tracks `<state>~rule<seed>` from the round scripts.
- **Read by:** `triforce train` when the family's yaml has `curriculum.enabled` or `--curriculum DIR`
  (`triforce/train.py:494`: `--curriculum`, else the yaml's `curriculum.trajectories`, else
  `paths.TRAJECTORIES_DIR / <family>`), loaded by Rust `Curriculum::new` (`crates/env/src/curriculum.rs:159-170`,
  `load_tracks` 101-110); `run.json`'s `curriculum_tracks` lists what was found (`train.py:590-596`); the lab's
  route tracks (`triforce/lab/collect.py:66`); `scripts/state_walk.py:66`.
- **Written by:** `triforce.explore --out` (default `paths.TRAJECTORIES_DIR`, `triforce/explore.py:140`), the round
  curriculum scripts (`scripts/experiments/r2*_curricula.sh`). **In place:** yes, a re-exploration rewrites a
  family's tracks (AGENTS.md: re-explore whenever a family's rooms change); a running training reads them at start.
- **Absent:** the whole folder missing or empty: loud, "No successful trajectories under DIR"
  (`curriculum.rs:166`). **One track missing (a seed state never explored): silent**; the state is drawn from the
  seed pool only, visible only in `run.json` `curriculum_tracks`.
- **Nature:** shared data set. **Backup:** none. Three hand-made older copies sit beside it in main
  (`trajectories.prev-r60/-r70/-r71`), read by nothing in the code.

### `triforce/states/harvested/<family>/*.state`
- **What:** chain handoff states (a family's room entered as the route hands it over, with the hearts and items the
  run brought), the start states a family draws at `harvested_fraction`.
- **Read by:** the Rust start-state pool of every training env (`crates/env/src/start_states.rs:275-331`,
  `config.rs:1124-1140`: `<states dir>/harvested/<family>` unless the yaml names `harvested_states`), so by
  `triforce train` (`train.py:609`) **and by the lab's trainings** (`triforce/lab/collect.py:207`, its own
  `TrainTask`); `train.py:574-588` records the source in `run.json` `harvested`; `GameData::drawn_namespaces`
  (`config.rs:1477-1479`).
- **Written by:** `eval-chain --harvest [DIR]` (default `paths.HARVESTED_DIR`, `triforce/cli.py:454`; the files are
  written as runs play, `runner.ChainPool`), the round scripts (`scripts/experiments/r26_arrow_states.py`,
  `r27_lynel_states.py`, `r29_zora_states.py`, `r29_demos.py --harvest`, `r32c_harvest.py`,
  `r34d5_garden_states.py`, ...). **A worktree's `eval-chain --harvest` writes straight into main's folder through
  the setup link.** **In place:** yes; files are added, and on a route change the lead moves a set aside
  (`target/harvested.prev-1.9.0.1` in main) and harvests again.
- **Absent:** `triforce train` refuses since `23412d57` when `harvested_fraction > 0` and none are found. **Silent
  elsewhere:** the Rust pool with seed states and an empty folder draws only seed states
  (`start_states.rs:321-325`), which is what the lab's trainings get; a harvested file deleted while a training
  runs sends its resets back to the seed states (F159, `start_states.rs:337-351`).
- **Nature:** shared data set. **Backup:** none for main's folder (783M); a removed worktree's harvests are under
  `/backup/triforce/worktree-data/<date>_<branch>/`.

### `triforce/states/harvested-rooms/<family>/`
- **What:** room-entry states of the lab's bench (`triforce/lab/bench.py:31,62`; listed by `bench-rooms.json`).
- **Read by:** `triforce.lab.bench` (`bench.py:120,181,320`: `<states dir>/harvested-rooms`). **Written by:**
  `eval-chain --harvest DIR --harvest-every-room` by hand, then `bench rooms` (`bench.py:458`) writes the manifest
  (tracked). **In place:** added to.
- **Absent:** [inference] a bench entry's state fails to load (the store's "no such state"), loud.
- **Nature:** shared data set (the bench is a yardstick). **Backup:** none.

### `models/` (`registry.models_root`: `--models-dir`, else `$TRIFORCE_MODELS_DIR`, else `<repo>/models`, `triforce/registry.py:147-153`)
Four kinds of thing share one folder:
1. **Run dirs** `models/<family>/<run-id>/`: `ckpt.pt`, `metrics.jsonl`, `run.json`, `train_summary.json`, `tb/`,
   `version`, `act`, `eval*.json`, `curriculum.json`, `checkpoints/it*/`, `selection.json`; `<run>.prev-N/` set
   aside by a retraining into the same dir (`train.py:554-572`).
2. **Pointers** `models/<family>/promoted`, `promoted.<route>`, `late.<route>`: the team of record.
3. **Groups and built networks** with no family yaml: `r36merge/`, `r38c/`, `r38head-hit/` (hit networks the
   promoted hit-head networks were built from), `d1-melee`, ... (17 dirs, some "evidence").
4. **Loose round outputs** in the root: 174 files (`<round>-summary.json`, `chain-*.jsonl`, `arm-*.jsonl`), written
   by `scripts/train_round.py` (docstring lines 20-25).
- **Read by:** every tool that plays a network: `train --load` (`train.py:394-420`: a path, else a family's
  promoted), the chain's team (`runner.Policies`, `runner.py:430-487`, through `registry.policy_checkpoint`
  547-574), `evaluate` (promoted by default), `select-checkpoint`, `eval-batch --families`, the golden racer, the
  play tools, the lab (`--teacher`, `--load`), `scripts/full_runs.py:686`, `scripts/parallel_study.py:70`; the
  test suite reads main's team through `$TRIFORCE_MODELS_DIR` (`triforce/job.py:687-697`); the back-test trees'
  `models/` are symlinks into `/backup/triforce/models/runs`.
- **Written by:** `train` (its run dir), `promote` (pointers, `registry.py:389-411`, atomic replace),
  `--act`/`set_act_mode` (`act`, `registry.py:447-470`, renames `eval.json`), `evaluate` (`eval*.json` in the run
  dir, `evaluate.py:398-420`), `select-checkpoint --apply` (replaces the run's `ckpt.pt`,
  `select_checkpoint.py:380`), `convert-model` (a new
  run dir under `--out`), `train_round.py` (root summaries), `backup_models.py team` (a whole team dir), the lead and
  agents by hand (pointer files copied into worktrees, run dirs linked between checkouts: Fireline40's
  `models/zol/` holds copied `promoted`, `promoted.d3first` files and symlinks to every one of main's zol run dirs,
  so a same-dir continuation or an `--apply` there would write into main's runs).
- **In place:** yes, all of: pointers, `act`, `ckpt.pt` (`--apply`; a same-dir `--load` continuation), `eval.json`.
- **Absent:** a family with no pointer: `train --load NAME` fails (FileNotFoundError); a chain records it in
  `provenance.missing` and every run reaching its rooms ends `error`, exit 1 (`runner.py:466-470`,
  `crates/pool/src/zelda/envs.rs:1047`). **Silent:** no `promoted.<route>` plays the family's `promoted` on that
  route (by design, `registry.py:478-486, 547-564`); no `act` file plays `DEFAULT_ACT`, and an `act` file written for
  other weights plays `DEFAULT_ACT` with a warning only (`registry.py:424-445`); a missing `version` file reads as
  None; a promoted hit-head network replaced by `--checkpoint family=RUN` plays the bare policy with no remark
  (the second void result of Nov 23).
- **Nature:** the pointers and the runs they name are shared (the team of record); a worktree's own trainings are
  per checkout until promoted; the loose summaries are scratch. **Backup:** `/backup/triforce/models` by
  `scripts/backup_models.py store` (run dirs once by content, pointers under `sources/<date>_<what>/`, teams per
  version); run by hand at landings, not on a schedule.

### `logs/` (`events.logs_root`: `$TRIFORCE_LOGS_DIR`, else `<this checkout>/logs`, `triforce/events.py:125-131`)
- **What:** one folder per job/tool (`<yymmdd>_<NNN>_<label>/`: `training.json` plan, event streams, `.log`).
- **Read by:** `training-gui.py` (its own checkout's `logs/` or `$TRIFORCE_LOGS_DIR`, `training-gui.py:4851`; a
  worktree's job is found through the queue row's `log_dir`), `triforce test-times`.
- **Written by:** every queued tool (`events.new_log_dir`); the test timing logs go to the main checkout's `logs/`
  from any checkout (`triforce/testlog.py:59-65`). **In place:** appended.
- **Absent:** created. **Nature:** per checkout (test timings shared). **Backup:** a removed worktree's, in
  `worktree-data`.

### `target/` (cargo's and everyone's scratch; 44G in main)
- **Build output:** `release/` (2.2G), `maturin/`: per checkout.
- **Tool defaults under it:** `entry-states/<stamp>` (`entry_states.py:94`, staging before `install`),
  `select/<family>-<run>-<segment>-clips*` (`select_checkpoint.py:531`), `full-runs/`, `runs250/`
  (`scripts/full_runs.py:674,689`), `route-tracks/` (`lab/collect.py:71`), `perf/parallel/`
  (`scripts/parallel_study.py:151`), `leaderboard/` (`scripts/speedrun.py:38,112`), `r56-chains/` (19G; written by
  `scripts/experiments/r56_chains.sh`, read by `room_stats.py:97` and `zelda-map.py:49,2326`).
- **Shared data living in scratch:** `demos/` (986M: the `--demos` files every lineage that used them must keep,
  AGENTS.md), `models.prev-*` (5.7G: teams moved aside at old landings), `harvested.prev-1.9.0.1`, `worktrees/`
  (4.0G: removed worktrees' data before the backup layout), `bench-cache`, `bench-runs`, `corpus-golden`,
  `golden*`, `oracle`, round folders `r10a` ... `r59-harvest`.
- **Absent:** created. **Nature:** per checkout by intent; in practice main's holds data other runs read (next
  section). **Backup:** only a removed worktree's (`tidy_worktrees.sh` -> `worktree-data/`); main's is not.

### `runs/`
- **What:** default outputs: `eval-chain` (`runs/eval-chain-<stamp>.jsonl`, `cli.py:207`), `triforce run`
  (`runs/runner.jsonl`, `run.py:136`), `eval-batch` (`runs/<batch>/`, `eval_batch.py:398`), golden races
  (`runs/golden/<stamp>`, `job.py:658`). Paths are relative to the current directory, so they land in whatever
  checkout the command ran from. **Nature:** per checkout scratch. **Backup:** a removed worktree's.

### `replays/`
- 61 tracked files (golden and single runs checked in) and 630 untracked ones beside them in main (written by
  `scripts/archive_replays.py --repo` (default `<repo>/replays`, line 138) and golden races, not yet committed).
  Read by `play-zelda.py`, `scripts/speedrun.py:183`, `tf replay`. **Nature:** shared (to watch).
  **Backup:** `/backup/triforce/replays/<version>/` (bzip2 / 7z) by `archive_replays.py`.

### `.play/`, `screens/`, `golden-*.mp4`, `triforce/states/*.png|*.yaml|*.lock`
- `.play/session.pkl`: `python -m triforce.play`'s session (`triforce/play.py:28`, relative to the cwd).
- `screens/`: `zelda-map.py`'s room pictures (`zelda-map.py:138`, `--screens-dir`).
- `golden-*.mp4`: `create-video.py` outputs the owner asked for.
- `triforce/states/*.png`, `*.yaml`: scratch of state tools; `**/*.lock` (`entry_states.json.lock`): the entry-state
  manifest's lock. All per checkout, none read by a run.

### `triforce/_native.abi3.so`, `.venv/`
- The built module (`maturin develop`, `triforce job build`) and the venv. Per checkout by nature. The module is
  the code a run executes: a module older than the last change under `crates/` runs old Rust under new Python, and
  nothing records or checks it today.

### `~/.local/state/triforce/` (`jobqueue.queue_dir`: `$TRIFORCE_QUEUE_DIR`, else `$XDG_STATE_HOME/triforce`, else this)
- `queue.db` (every job's row, kept for good), its WAL, `locks/` (one flock per place and per live job).
- Read and written by `triforce job`, the tools (`jobs.progress`, `record_plan`), the GUI. Shared by nature (one
  machine). **Backup:** none.

### `/backup/triforce/` (`$TRIFORCE_BACKUP` in the scripts)
- Layout in its own `README.md`: `models/runs` (every run dir once, hard links, read-only), `models/teams/<version>
  .json`, `models/sources/`, `replays/<version>/`, `worktree-data/<date>_<branch>/`. Written by
  `scripts/backup_models.py store|manifest|index`, `scripts/archive_replays.py`, `scripts/tidy_worktrees.sh`; read
  by `backup_models.py team` (rebuilds a version's models dir) and the back-test trees. Not backed up itself (its
  own disk).

## Outputs that later runs read as inputs (the leaks between scratch and shared)

1. **Harvested states from a chain**: `eval-chain --harvest` writes into the shared `triforce/states/harvested`
   directly (default) or into a scratch folder the lead copies in; every later training of the family draws them.
   No record says which chain, team or code produced a file.
2. **Entry states**: `entry_states` stages under `target/entry-states/<stamp>`, `install` copies into the tracked
   `triforce/states` (that one is reviewed through git).
3. **Heart values and chain hearts**: `python -m triforce.heart_value CHAINS...` and `triforce.chain_hp CHAINS...`
   read chain `.jsonl` files from wherever they are (usually `target/r56-chains/` or a round's `target/` folder) and
   write the tracked `triforce/heart_value*.json`, `chain_hp*.json`, which every training's rewards and hearts read.
4. **Checkpoint selection**: `select-checkpoint` plays chains into `target/select/...`, writes `selection.json` in
   the run dir, and `--apply` replaces the run's `ckpt.pt` (shared once promoted).
5. **Demonstrations**: `scripts/experiments/r2*_demos.py` write `target/demos/*.npz` (or `target/r29-demos/`);
   `train --demos FILE` reads them; a lineage that used them needs them kept.
6. **Room statistics and the map**: `room_stats.py` and `zelda-map.py` read `target/r56-chains/`.
7. **Lab tracks and caches**: `lab/collect.py` writes `target/route-tracks/<route>/<family>` from `trajectories/`
   and the lab's trainings read them; the bench's caches under `target/bench-cache`.
8. **Teams of record**: `backup_models.py manifest VERSION CHAIN.meta.json` reads a chain's sidecar (in `target/`
   or `runs/`) to write `/backup/triforce/models/teams/<version>.json`.
9. **Run dirs between checkouts**: a worktree's `models/` gets main's pointers copied and its runs symlinked by hand
   (round 40: `triforce-r40-fireline/models/zol/r34d3-zol-s341` is a link to main's run dir, a write through it
   lands in main); a worktree's promoted run reaches main at landing the same way.
10. **The 250-run test**: `scripts/full_runs.py runs250` writes `target/runs250/<stamp>`; its numbers go into
    `docs/runs250/` (tracked) and its replays into `/backup/triforce/replays`.
11. **Teams moved aside**: `target/models.prev-*` and `target/harvested.prev-*` hold what an old landing replaced;
    nothing reads them today, but they are the only copy of what they hold outside `/backup`.

## Silent defaults found (missing read as "none")

- A curriculum track missing for a seed state (`curriculum.rs`): the state plays without one; only `run.json`
  lists the tracks.
- Harvested states missing for a lab training (`lab/collect.py:207`): drawn from seed states only.
- A harvested file deleted mid-training (F159): resets go back to seed states.
- A start-state glob of a yaml that matches no file (`config.rs:1155-1170`): dropped; the family trains on the
  others (only all of them matching nothing is an error, `start_states.rs:296-302`).
- No `act` file, or one for other weights: `DEFAULT_ACT` (a warning in the second case only).
- No `promoted.<route>`: the family's `promoted` plays the route's own part (by design, unrecorded at play).
- A `--checkpoint family=RUN` override of a hit-head network with a bare policy: no remark.
- The checkout's built module older than its `crates/`: nothing checks.

## What the layout has to answer (for the lead and the owner)

The owner's direction (Nov 23): shared state in one place under the home directory, outside any checkout, known to
every agent as shared and not to be touched; `target/` per checkout and separate. From the table, the candidates
for that place are the rows marked shared: the ROM, `trajectories/` (and its `.prev` copies), `harvested/`,
`harvested-rooms/`, the team of record in `models/` (pointers and the runs they name, the groups the hit-head
networks were built from), `target/demos/`, the watched `replays/`, and the queue's state (already outside every
checkout). Open questions the inventory raises, each needing a rule:

1. `models/` mixes the team of record, every agent's trainings, and loose round summaries. Where does a worktree's
   training write, and how does a run become part of the shared set (promotion as the only door?).
2. Writers into shared data: `eval-chain --harvest`, `explore --out`, `promote`, `set_act_mode`,
   `select-checkpoint --apply` change shared files in place today. Should a shared set be append-only (new folder
   per harvest / exploration, a pointer naming the current one), so a run can name exactly what it read?
3. `target/` holds data later runs read (the leaks above). Each leak needs a door into the shared place or a
   record of where its input came from.
4. Backups: `harvested/` (783M), `trajectories/` (1.5G), `harvested-rooms/` and `target/demos/` (986M) have no
   backup today.

Whatever the answer, the code will ask one function in `triforce/paths.py` (and its Rust twin) for each location;
the manifest and the refusals of round 40 take every path from there and record where each input came from.
