# Where things are: the shared store and each checkout (the owner's layout, Nov 23)

Everything a run shares with other runs lives in **one place outside every checkout**, the shared store
`~/triforce-shared/` (`$TRIFORCE_SHARED` for tests and other machines), and is written **only through one program**,
`triforce.store` (the owner's decisions of Nov 23; the store's shape is the lead's; `docs/credits.md`). A checkout
holds the tracked files and its scratch, `target/`, which is never an input to another run. Every location is one
function's answer in `triforce/paths.py` (its Rust twin is the `paths` crate); `triforce store paths` prints them all,
and the store keeps them in `layout.json` for tools that are not Python.

## The store

```
~/triforce-shared/
  README.md            the rules, for a person who opens the folder (text: triforce/store_readme.md)
  layout.json          every location (paths.layout()), rewritten by the store whenever a `current` changes
  store.db, locks/     the index: snapshots, runs, teams, jobs, filed outputs, kept summaries, and the audit log
                       (every write: when, job, agent, user, entry point `public` or `lead`, what)
  rom/zelda.nes        read-only
  data/<kind>/<name>/  named snapshots, read-only: harvested, harvested-rooms, trajectories, demos; file count,
                       bytes and content hash recorded at install (no reader re-hashes); data/<kind>/current names
                       the default
  runs/<family>/<run>/ network runs; family may be a group's path (r38head-hit, r36-...-evidence/schema2/goriya)
  teams/<name>.json    immutable teams; teams/current names the team of record
  jobs/<job>-<label>/  what a job filed: chains/, evaluate/<family>/<run>/, selection/, batches/, harvested/,
                       replays, golden races, runs250/, statistics
  legacy/import-<date>/ what the import of an old models folder found no home for, with LIST.txt
```

## The rules

1. **Nothing shared is changed in place**; things are added: a snapshot installed, a run sealed, a team made. What is
   read-only is read-only on disk (files 0444, folders 0555), so an accident fails.
2. **A run** is allocated by the store (unique name in its family, owner job and agent, its parent, the effective
   team and the data snapshots it read), written by its training while **open**, then **sealed**: read-only, its
   checkpoint SHA-1 and default act mode recorded. A continuation (`train --load FAMILY/RUN` or a team slot), a
   checkpoint chosen by `select-checkpoint --apply`, a converted model is a **new run with a parent**. An interrupted
   training leaves its run open; its owner (the same agent) continues it in place with `train --resume FAMILY/RUN`.
   **A numbered checkpoint** (`<run>/checkpoints/it<NNNNNN>/`, round 41) becomes a sealed run of its own,
   `<family>/<run>-it<NNNNNN>` (origin `checkpoint`, parent the run), so a team can play it while the run still
   trains (`triforce store checkpoint FAMILY/RUN@it228|@latest`, `eval-batch --teacher GROUP=FAMILY/RUN@it228`): the
   parent is only read, never written or sealed; the weights are a copy when the parent is open (a hard link would
   share the file's modes with the training's own file) and a hard link when it is sealed; a checkpoint the trainer
   is still writing is refused (the trainer renames `ckpt.pt` into place, then writes `version`, then `act` naming
   the new file's SHA-1: an open run's checkpoint is complete once its `act` names its `ckpt.pt`); the same
   checkpoint again is the same run. Its `run.json`: `checkpoint` (parent, the parent's status then, iteration,
   steps, SHA-1, source, `copied` or `linked`, act mode), `parent` and `manifest` (the parent's, with
   `networks.checkpoint`), so `triforce manifest diff` and a team record name the weights.
3. **What used to be written into a run dir after training** (`eval*.json`, `selection.json`, `act`) is filed by the
   job that made it, indexed by the run (`triforce store outputs --run FAMILY/RUN`); the act mode is the run's sealed
   default and each team slot's. A tool given a team reads a run's filed evaluations as files of the run (the view).
4. **A team** names per slot a sealed run, its checkpoint SHA-1, its act mode and whether it has a hit head. Slots:
   `family` (what `promoted` was), `family@route` (`promoted.<route>`), `family@route:late` (`late.<route>`). A tool
   plays `--team NAME` (default the current one) or an experiment's override, `--team NAME+zol@d3first=r40-x-s401`
   (`+SLOT=RUN`, repeatable, a later one winning; `RUN` alone is of the slot's family; round 40: `+@GROUP=FAMILY/RUN`
   puts one run in every slot of a teacher's families, `family@route` and `family@route:late` each, `store.groups`;
   a family, run or team name holding `+` cannot be written in a spec and is refused when allocated), and records the
   effective team (name, overrides, content hash, every slot) with its result: a chain's provenance `team`, a run's
   `run.json` and index row. `triforce-lead team new NAME --base TEAM --slot @GROUP=FAMILY/RUN` makes it lasting.
5. **Data** is read by name: `train --data harvested=NAME` (default the kind's current snapshot), recorded with the
   run. `eval-chain --harvest` without a folder writes into the job's folder, never into a snapshot; a harvest becomes
   training data only when the lead installs it.
6. **Every evaluation, chain and training filed leaves one summary row** kept for good (what was measured, of which
   run or team, on what data, runs or episodes, successes, rate and its standard error, and the headline numbers of
   the run statistics' compact row); the raw files in the job's folder are what a later trim may remove.
   `triforce store summaries --run FAMILY/RUN | --team NAME` lists them over time.
7. **Missing is an error** that names where it looked (the ROM, a snapshot, a run, a team).
8. **Only the store writes the store.** Agents use `triforce store ...` to look (and `triforce store checkpoint`,
   which only adds a sealed run); the tools call `triforce.store` to allocate, seal and file. Nobody copies, links,
   moves or edits anything there by hand.

## The statistics tables (the aggregate kept for good)

The run statistics (`triforce.run_stats`: every chain, evaluation and training leaves `<output>.stats.json`) are
duplicated into two SQL tables of `store.db` when a job files them (`triforce.store_stats`, in the filing's
transaction): derived, rebuildable (`triforce store reindex`), queryable after the raw files are trimmed or moved to
`/backup`. Column names are the statistics file's fields one to one, nested names joined with `_`.

- `evaluations`: one row per statistics file: who and when (`job_id`, `job_agent`, `job_task`, `job_subtask`,
  `written`), what (`kind`, `name`, `description`, `provenance_route`, `from_step`, `to_step`, `provenance_code`,
  `provenance_version`), played by what (`store_team`, `store_team_sha256`, `store_overrides`, `store_run`,
  `store_data`), the rate (`rate_n`, `rate_successes`, `rate_rate`, `rate_se`), the endings, the beams, hearts lost
  and frames as count, sum and sum of squares, the flags, `raw_path` (the file, which may later be gone) and `runs`:
  every run's outcome (seed, result, furthest step, frames, frames to each step ...) as one compressed JSON blob
  (`store_stats.runs_of`), enough to pair two evaluations by seed.
- `evaluation_rooms`: one row per evaluation and route step (an evaluation's per room): `step`, `room`, `level`,
  `room_id`, `cave`, `family`, `network`, `network_run`, `network_sha1`, the counters (`visits`, `exits`, `deaths`,
  `stuck` ...), and every per-visit figure (hits, hearts lost, health in and out, bombs and keys used and cheated,
  shots, shots that hit, far and near kills, lined-up unfired, held frames, linger ...) as `<name>_n`, `<name>_sum`,
  `<name>_sq`, so a mean and its standard error is a query; frames to enter the step as `enter_*`.
- Indexes on (level, room), family, team, time; `stats_schema` numbers the layout.

```
triforce store query                                   # the examples
triforce store query deaths-by-room-over-time --param level=3 --param room=0x5b
triforce store query bombs-by-family
triforce store query "SELECT store_team, SUM(rate_successes), SUM(rate_n) FROM evaluations GROUP BY 1"
```

## A checkout

- Tracked files, `target/` (scratch: builds and tools' working folders, `paths.scratch()`), `logs/` (the GUI's event
  logs, `paths.logs_dir()`), the venv and the built module.
- **No models folder and no data links.** `triforce job setup` makes none once the store exists.
- Our tools write nothing in a checkout outside `target/` and `logs/` when they play a store team.

## The transition (staged cutover)

Until the lead's reconciliation, today's layout keeps working exactly as before: `--models-dir DIR`,
`$TRIFORCE_MODELS_DIR` and a checkout's own `models/` (in that order, before the store's current team) are read and
written as they always were and recorded as such; a checkout's own `roms/`, `trajectories/`,
`triforce/states/harvested` are read before the store's snapshots. Old versions' trees get an old-style folder of
links: `triforce store view TEAM --into DIR`. What resolves where:

| thing | first | then | then |
|---|---|---|---|
| ROM | `$TRIFORCE_ROM` (only it) | store `rom/zelda.nes` | `<checkout>/roms/zelda.nes` |
| networks | `--models-dir` | `--team` / `$TRIFORCE_TEAM` | `$TRIFORCE_MODELS_DIR`, `<checkout>/models`, the store's current team |
| harvested, harvested-rooms, trajectories, demos | `--data KIND=NAME` (`train`) | the caller's own folder | the snapshot the checkout pins (`triforce/data_pins.yaml`), else the store's current snapshot |
| run written by `train` | `--models-dir` (as before) | the store (a store team plays) | |

One rule for every reader of the data, not only `triforce train`'s command line: `paths::data_dir` in Rust
(`SpecialistConfig::harvested_dir`, so every `StartStatePool`, env and pool; `GameData::trajectories_dir`) and
`paths.data_dir` in Python (the curriculum's default folder, the lab's families and route tracks, the bench's
harvested and harvested-rooms entries). The caller's own folder is `<states dir>/harvested` (and `harvested-rooms`)
for the states dir it passes, `<checkout>/trajectories` for curricula: a caller that passes an explicit states dir
keeps the tracked seed states from it and gets the harvested ones by the rule (its own `harvested/` while it has
one, else the store's snapshot). When neither exists the reader refuses, naming both places (`paths.data_missing`).
`triforce job doctor` prints where each kind resolves for its checkout. `explore` writes new tracks into a
checkout's `trajectories/` only while it has one, else into `target/trajectories`: a new folder would shadow the
snapshot for every reader. The test suites' session store links the machine store's current snapshots, so they
play the same states in a checkout with the old links or without them; `triforce job test` names the store's current
team (`$TRIFORCE_TEAM`) and the Python suite gives its tests a read-only view of it, never a checkout's `models/`.

**Pins** (Nov 25): a branch whose code no longer matches the store's current snapshot of a kind names the one it
reads in `triforce/data_pins.yaml`, committed, so every worktree of the branch reads it (`kind: snapshot` lines,
`#` comments; `paths.pins`, the Rust twin `paths::pins` with the same rule; a malformed line is refused naming it).
The order is the caller's own folder, then the pin, then the store's current snapshot; a pinned snapshot the store
does not hold is refused (`paths.data_missing` names the pin), never quietly replaced by the current one. Python
reads the pins of the checkout `triforce/paths.py` lives in, Rust those of the checkout `crates/paths` was built
from. `triforce job doctor` says `store snapshot NAME (pinned by .../triforce/data_pins.yaml: ...)`, a training's
source line `pinned by triforce/data_pins.yaml`, and its manifest's data record carries `'pinned': true` beside the
snapshot's name; the test suites' session store links the pinned snapshots beside the current ones. No file, no
pins: the default. `main` pins `trajectories: 261124-gen2` (the store's current, `261125-all4`, is the fresh
generation's, without the old `d3first` tracks of the walked doors that main's three-chain states and tasks
draw); the generation's branch `r40-int` has no pins: it deletes the file in one commit after it merges main, and
keeps it deleted on every later merge (git keeps a deletion against an unchanged file; when main changes the file,
the merge stops on a modify/delete conflict, resolved with `git rm triforce/data_pins.yaml`). When a branch catches
up with the current snapshot it deletes its pin.

Every tool that plays or trains says at its start, in one line, which it took and why, and its manifest carries the
same line (`models.why`; a training's `sources`): `models: store team 2.0.0.13 (--team; sha256 ...)`,
`models: --models-dir /path (the layout before the store)`, `harvested: this checkout's folder /path (the layout
before the store)`, `trajectories: store snapshot 261122-52fb5b17 (current: /path)`. A temporary team view a process
left behind (the queue's dry call ends with `os._exit`) is removed by the next tool that makes one.

## The reconciliation of the main checkout: old paths and where the same thing is now

The main checkout kept its own folders after the import of Nov 23, so it read them and not the store. The
reconciliation moves them aside (`python -m triforce.reconcile`, round40.md "Main reads the store"): every folder
below, compared by content first (`reconcile compare`; Nov 24: every one identical to a store item, nothing only in
main), then renamed into one dated folder beside the checkout (`reconcile aside`; `reconcile back` puts them back).
Main's `target/` (but `target/demos`), `logs/`, `runs/`, `replays/` stay: per checkout. A document that cites an old
path (the experiment docs are history and keep them) is read with this table; every concrete path cited in
`docs/`, `AGENTS.md` and `README.md` is looked up in `docs/store-paths.md`.

| old path (a checkout before the store) | what it was | now |
|---|---|---|
| `roms/zelda.nes` | the ROM | `rom/zelda.nes` of the store (`paths.rom()`, the `paths` crate) |
| `trajectories/<family>/<track>/` | curriculum tracks | `data/trajectories/<snapshot>/<family>/<track>/`; main's folder of Nov 23 is snapshot `261122-52fb5b17`, the default is `data/trajectories/current` (`triforce store data`) |
| `trajectories.prev-r60`, `-r70`, `-r71` | hand-made older copies | snapshots `261109-2dfee7b9`, `261113-2ebb5ce2`, `261113-578619ff` |
| `triforce/states/harvested/<family>/` | chain handoff states | `data/harvested/<snapshot>/<family>/`; main's is `261123-ad612e94` |
| `triforce/states/harvested-rooms/<family>/` | the bench's room entries | `data/harvested-rooms/261123-aabd5d07/<family>/` |
| `target/demos/<path>.npz` | demonstration files | `data/demos/<snapshot>/<path>.npz`; main's is `261122-453e1975` |
| `models/<family>/<run>/` (`ckpt.pt`, `run.json`, `metrics.jsonl`, `checkpoints/`, `tb/` ...) | a network run | `runs/<family>/<run>/`, sealed (`triforce store runs FAMILY/RUN`); `<family>` may be a group's path (`r38head-hit/all`) |
| `models/<family>/<run>/eval*.json`, `act`, `selection.json` | written after the training | filed by job, indexed by the run (`triforce store outputs --run FAMILY/RUN`; the import's are under `jobs/475-lead-store-import/evaluate/<family>/<run>/` and `act/<family>/<run>/`); the act mode is the run's sealed default and each team slot's; a tool given a team reads them as files of the run (the view) |
| `models/<family>/promoted`, `promoted.<route>`, `late.<route>` | the team of record | slots `family`, `family@route`, `family@route:late` of a team (`triforce store teams NAME`); `teams/current` names the team of record (2.0.0.15: main's pointers of Nov 23, every slot run and act mode the same, compared Nov 24) |
| `models/<round>-summary.json`, `models/chain-*.jsonl`, `models/<name>-evidence/*.json` | loose records | filed records `models/<path>` (`triforce store outputs --kind record`), at `jobs/475-lead-store-import/record/models/<path>` |
| any other file under `models/` (pictures, logs, retired curricula of the evidence folders) | no home | `legacy/import-261123/models/<path>` (listed in its `LIST.txt`) |
| `models/` as a folder a tool takes (`--models-dir`, an old version's tree) | an old-style team | `triforce store view TEAM --into DIR` |
| a run deleted before the store, kept by a backup | | `/backup/triforce/models/runs/<family>/<run>` (`runs.jsonl`); not in the store |
| `target/<...>` of a worktree, `runs/`, `logs/` | scratch | a removed worktree's: `/backup/triforce/worktree-data/<date>_<branch>/`; main's stay in main |

## Hardcoded locations replaced (each is now `triforce.paths`)

`context.rs` and `crates/tools/src/rom.rs` and `crates/test-rom/src/lib.rs` ROM lookups (the `paths` crate);
`tests/conftest.py` ROM lookup; `scripts/level_rooms.py:152`; `events.logs_root`; `entry_states.STAGING_ROOT`;
`room_stats.CHAINS_DIR`; `select_checkpoint`'s chains folder; the lab's `target/route-tracks` and its trajectories
source; `eval-chain --harvest`'s default; `registry.models_root`'s default; the trajectories folder of `train`.
Left as they were (tracked, or per checkout): `STATES_DIR`, `SPECIALISTS_DIR`, the corpus, `GameData.trajectories_dir`
(informational), the scripts' own `target/` folders.

# Appendix A: before the store (the inventory of Nov 23, Recipe40)

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
- **Stale:** a name outlives the route that wrote it (round 42: 2.0.0.5's reorder left a fifth of `harvested`
  261124-gen1 holding pieces and items of no step of today's `d3first`; `261124-gen2` is it without them). `python -m
  triforce.harvest_audit audit --snapshot NAME --out X.jsonl --invalid X.txt` checks a snapshot against the route as
  it stands (exit 1 on an invalid state); a training reset refuses a harvested start off the route's progress
  (`zelda::objectives::start_on_route`) and a chain's harvest writes none. Run it on a snapshot before it is set
  current and before a retrain.
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

## Silent defaults found (missing read as "none"), and what round 40 made of each

Line numbers are `23412d57`'s. **Refused**: the run stops before it starts and names where it looked
(`docs/architecture.md` "Missing is an error"). **Declared**: "none" stays a legitimate recipe, written in the task.

1. A curriculum track missing for a seed state (`curriculum.rs`): the state played without one; only `run.json`
   listed the tracks. **Refused unless declared** (`curriculum.untracked` in the task; every task of `specialists/`
   declares today's untracked states).
2. Harvested states missing for a lab training (`lab/collect.py:207`): drawn from seed states only. **Refused**
   (`lab.collect.parse_families`, and the native pool itself, 4).
3. A harvested file deleted mid-training (F159): resets went back to the seed states. **Refused** (an error of the
   running pool).
4. A pool drawing harvested states from an empty folder (`start_states.rs:321-325`): the seed states alone.
   **Refused** when the pool is built.
5. A start-state glob of a task that matches no file (`config.rs:1155-1170`): dropped; the family trained on the
   others. **Refused** (train, evaluate, the lab; none of today's tasks has one).
6. No `act` file, or one for other weights: `DEFAULT_ACT` (a warning in the second case only). **Left as a rule**
   (F064, the act played is in every chain's provenance; the store seals it into the run).
7. No `promoted.<route>`: the family's `promoted` plays the route's own part. **Left as a rule** (phase X's team
   rule; the key that played is in the provenance; the store's teams name every slot).
8. A `--checkpoint family=RUN` override of a hit-head network with a bare policy: no remark. **Refused unless
   declared** (`--headless KEY`; `select-checkpoint --headless`).
9. The checkout's built module older than its `crates/`: nothing checked. **Refused** (the build's stamp,
   `triforce.build`).
10. A `--load` continuation's curriculum progress resumed when the loaded run's `run.json` had no `curriculum`
    field (`train.py:475-482`). **Refused unless declared** (`--curriculum-unrecorded resume|fresh`).
11. `--load VALUE` that is no existing path was read as a family name (`train.py:394-398`). **Refused** with what
    was looked for (a directory without `ckpt.pt` too).
12. A worktree with its own `models/` (even a partial one) gave the test suites that team (`job.py:691-697`).
    **Fixed**: the suites read the main checkout's team always; since the reconciliation's preparation (Store41,
    Nov 24) they play the store's current team (`job.test_steps`, `$TRIFORCE_TEAM`), never a checkout's folder.
13. `eval-chain --specialist-config YAML` for a family the runs never play: accepted and unused. **Refused**.
14. `eval-chain --harvest` without DIR wrote into the shared harvested folder (`cli.py:454`). **Refused** with a
    models folder: DIR is required; with a store team it writes into the job's folder of the store, never a snapshot.
15. A family with no promoted run on a chain's route: only the runs that reached its rooms ended `error`.
    **Refused** before anything plays (`runner.check_team`).
16. The training GUI lists log folders of its own checkout only (`training-gui.py:4851`). **Left**: not an input of
    a run; every queued job shows through its row, and the store files job outputs in one place.
17. A missing `version` stamp in a run dir reads as None (`registry.model_version`). **Left**: provenance records
    None; not an input to play.
18. The harvested folder rescanned under a running pool picks up files added since the start (`start_states.rs`
    `HARVEST_RESCAN_RESETS`): an input that grows during the run. **Left** for the store's read-only snapshots.

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
