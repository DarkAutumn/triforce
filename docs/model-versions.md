# Model versions: what each one was and what it changed

Every version of the model, from the hand-written projects to the fresh generation training now, with what each
changed and how far it got. Built from the history: the `VERSION` file's log (`git log -p -- VERSION`), the release
tags `v<VERSION>`, `docs/experiments/round*.md`, `docs/decisions.md`, `docs/experiments/backlog.md`,
`docs/runs250/`, AGENTS.md "Where things stand", README.md "History", the `v1-original-design` and `v2-stable-retro`
branches, the backup's index (`/backup/triforce/INDEX.md`) and the old development checkout `triforce2` (below,
"History not in this repository").

## What a version is here

- **One version for the code and the models it plays**, `MAJOR.MINOR.GEN.REV`, in the `VERSION` file since Nov 11
  (commit `65c1f101`, 1.3.1.0; AGENTS.md "Versioning"). `triforce.__version__`, `_native.VERSION`, `pyproject.toml`
  and the Cargo workspace repeat it; every run dir carries a `version` file (provenance only: whether a model loads is
  decided by its checkpoint's schemas since Nov 23). A release is tagged `v<VERSION>` on `main`.
- **`MAJOR`**: only when the owner says so. 2.0 is the `d3first` route (the speedrunners' order) and, from 2.0.1.0,
  the program of one network.
- **`MINOR`**: through 1.9 it counted the dungeons the route completes (1.3 = three, 1.9 = all nine: Dungeons 8 and 9
  landed together, so there is no 1.8). From Nov 19 a minor bump is a big change, with a fresh generation.
- **The third number**: `COMPAT` until Nov 23 (+1 when old models could not load the new code: an observation or
  action layout change), then `GEN`, the retrain generation: +1 when every network the version plays is trained
  from scratch by that version's pipeline ("breaking the ship of Theseus"; decisions 2026-11-23). No version through
  2.0.0.16 was a fresh generation; none was renumbered.
- **`REV`**: every other change to what plays or how it trains (rewards, env or training code, networks continued or
  replaced, route data, a widened observation).
- **The current numbers** (decisions 2026-11-26): the generation training now is stamped **2.0.1.0** when its finished
  network is measured on the 1000-run chains; changes after that are 2.0.1.1, 2.0.1.2, ...; the next fresh retrain is
  2.0.2.0.

**How to read an entry.** Each version (small REV bumps grouped, every number listed) has a short paragraph saying
what the model was, a list of what it changed against the versions before it, by theme, and **Results**. A "segment"
is a chain from a fixed start state to the next milestone, 1000 runs unless said, models alone; "the full route" is a
single run from the game's start; "clips on | off" gives both switch states where they were measured (from round 26
on, clips only). The "250 test" is 250 runs of `d3first` from console power on (`docs/runs250/`). The honest
planner (from 1.5.0.1) is an act mode that tries each action on redrawn random numbers first; its numbers are
reported beside the models', never as the model's. Where a number was not measured, the entry says so. **Dates**
(2026 unless given) are commit dates: from the commit that set `VERSION` to the number to the release tag's commit
(the round's commits where there is no tag); the old projects' eras are their branches' first and last commits.

## The older models (before this code)

The project's first two versions were written by hand by the owner, to learn reinforcement learning; both are kept
on their own branches (README.md "History").

**v1, `v1-original-design`** (Dec 30, 2023, `ab746be0`, to Jan 4, 2025, `02139ee8`; 199 commits). Built on
stable-retro (NES emulation through
FCEUmm) and stable-baselines3's PPO with its default `MultiInputPolicy`. The network saw a grayscale viewport
centred on Link plus a few vectors (to the nearest enemy, item, projectile and the objective, and whether the beams
were up); actions were four moves and four attacks; an objective selector pointed Link where to go, critics gave
rewards and a separate score judged scenarios. Several models split the game by place and equipment
(`overworld1beams`, `overworld1nobeams`, `dungeon1beams`, `dungeon1nobeams`, `dungeon1boss`, `dungeon1entrance`,
`overworld-sword`), each picked by priority and the conditions it declared. It played from the start through the end
of Dungeon 1, though usually without winning; the README's [video](https://www.youtube.com/watch?v=yERh3IJ54dU) shows
it beating Dungeon 1. Its `models/evaluation.csv` (1000 episodes a scenario) records 93.1% for `dungeon1beams`, 93.7%
for `dungeon1nobeams`, 99.7% for `overworld1beams` and 98.4% for `overworld1nobeams`. Why it was replaced: the
stable-baselines architecture made action masking and frame stacking awkward (the v2 README: the model "simply never
trained" with stacked frames, and masks could not reach the policy), and the owner wanted his own networks and PPO.

**v2, `v2-stable-retro`** (Jan 7, 2025, `62ae9189`, to Aug 4, 2026, `80b26635`; it shares v1's history up to Jan 4,
2025, then 128 commits of
its own). Still stable-retro, still hand-written at first, with a PPO implemented from the original paper and the
owner's own networks: action masking, frames stacked, one model instead of the beams/no-beams split, and a one-hot
objective in place of v1's objective vector ("it felt too much like cheating"). Later work (2026, with LLM agents
helping) added an IMPALA ResNet with CoordConv and multi-head attention over items, enemies and projectiles
(`impala-multihead`), potential-based movement shaping, an all-items action space, weighted training circuits and a
Qt debugger. Its last recorded experiment (`docs/experiments/experiment5-summary.md` and
`dungeon1-completion-plan.md` on that branch, Aug 2026) puts it at: the overworld solved, Dungeon 1 stopping at
Aquamentus; the full-game evaluation 0 of 100 with a median of 13 of 17 milestones, the late Dungeon 1 chain 0%. Why
it was replaced: having learned what he set out to learn by hand, the owner moved to agent-written code to test
ideas quickly and get through the game (README.md "History"), starting from a fresh design (below).

## This code, version by version

This code (called v3 in the README) started on Nov 1, 2026 as a fresh repository, developed in the checkout
`triforce2` and merged into this repository's `main` with its whole history on Nov 12 (`cc26b050`, "Triforce v3: the
Rust emulator rewrite replaces the Python code on main"). Its first commit (`9f5eabae`) calls it "Triforce v2: first
working version", after the design document it was built from; the README's numbering (v3) is the one used here.

### Before the version file (Nov 1, `9f5eabae`, to Nov 11; rounds 1-18; no version number)

Specialist PPO models, one per family of rooms, read structured game state out of RAM (no pixels) and chose timed
options ("move W for 16 frames"); a runner picked the specialist for each room and played a fixed route. The route
was `milestone1` (the game start through Dungeon 1 to Dodongo's room in Dungeon 2) until round 13, then `d2first`
(Dungeon 2 first, then Dungeon 1, later on to Dungeon 3). There was no `VERSION` file; observation and action
schemas were numbered on their own.

- **Rounds 1-7 (Nov 1-3, Python on stable-retro):** every specialist trained once (round 1); Go-Explore with a
  backward curriculum over saved states for the families plain PPO could not learn (rounds 2-3); start states saved
  before room entry; one-room episodes and route fixes; families by enemy rather than by dungeon (the owner, round
  6: goriya, wallmaster, stalfos-keese-gel); the fairy route after Dungeon 1; heart tracking.
- **Rounds 8-11 (Nov 5-8, the Rust port):** `zcore`, a Zelda-only NES core cut down from TetaNES, and the game, env,
  pool and chain runner in Rust, with Python left for the network, PPO and orchestration (round 8); the core moved
  from reproducing FCEUmm to hardware timing, one quirk at a time (round 9); observation schema 2's `target` block
  and items as shaping targets (round 10); power-on RAM `$00`, every state regenerated and the whole team retrained
  from scratch (round 11).
- **Rounds 12-15 (Nov 8-9):** Dodongo beaten the owner's way, bomb-stun and sword (round 12); the reset onto
  `d2first` with every family retrained (round 13); the bow detour, `trap-room` and `equipment-room` (round 14); the
  whole overworld and Dungeon 3 mapped, cheated items in one ledger (round 15).
- **Rounds 16-18 (Nov 9-11):** inference in Rust (the Rust actor), AVX policy kernels, self-imitation (SIL), act
  modes per family (round 16); a 192-finding code review worked through and a fresh team `r61` from scratch with SIL
  (round 17); the hearts-aware early stop and continuations (round 18, whose team became 1.3.1.0).

**Results:** round 1's chain 0 of 20 (every run died in the overworld's Zora room 0x38); round 6 the first full
`milestone1` chain to Dodongo's room, 1 of 40; round 10's target observation took Dodongo-room arrivals per 120 runs
from 44/12/18/3/6 to 74/43/61/27/46 across the D2, post-D1 (two routes) and full (two routes) segments
(`round10.md` §8-§10; the same figures are in
the Nov 8 status note, below); round 12's Dodongo 300 of 300 fights; round 13's `d2first` 25 of 120 (21%) from the
start to the Dungeon 1 triforce; round 15 0 of 120 to the Dungeon 3 triforce; round 16's SIL team 4 of 120; round
17's `r61` 1 of 120 (Dungeon 1 115 → 80 of 120). Chains of 1000 runs a segment start in round 18 (next entry).

### 1.3.1.0 (Nov 11: `65c1f101`, tag `v1.3.1.0` at `5a5d4d23`; round 18's team)

The first numbered version: the team of round 18 (20 networks, `r61`-`r63`) on `d2first` from the game start
through Dungeon 2, the white sword and Dungeon 1 to the Dungeon 3 triforce, versioned when the replays branch
merged.

- Versioning itself: the `VERSION` file, a version stamp in every run dir, incompatible models refused.
- The hearts-aware early stop (round 18): a training stops only when hearts lost an episode have stopped improving
  too, not on success alone; stalfos-keese-gel and zol continued 500 iterations, 13 families continued and
  promoted.
- Starts from console power on, `TFREPLAY` replays, `zelda-headless.py` and the golden path (a race of runs until one
  reaches each segment's end), `play-zelda.py` playing replays from queues.

**Results** (round 18, 1000 runs a segment, against the previous team on its own code): Dungeon 2 87.6% (73.1%),
after Dungeon 2 81.9%, Dungeon 1 97.8%, after Dungeon 1 100%, Dungeon 3 40.6% (23.8%); from the start to Dungeon 1
49.3%, to Dungeon 3 29.2%, **the full route to the Dungeon 3 triforce 10.6%** (2.6%). The golden path from power on
to the Dungeon 3 triforce took 15-19 runs (the corpus's `golden-d2first-to-dungeon3-triforce`, 61,551 frames).

### 1.3.1.1 and 1.3.1.2 (Nov 11: `eb9a9ee4`, `291f4f74`; not tagged)

Two small revisions of 1.3.1.0's team, no network retrained.

- 1.3.1.1: a fix for the endless 0x4c/0x4d bounce (off the route the objective prefers only the route exits Link
  can play with his progress), and chains end `stuck` on the fifth return in a row (`bounce-limit`).
- 1.3.1.2 (branch `d4`): Dungeon 4 scouted (its map, room names, a scouting route `d2first-d4` and start states); no
  training. Merged into 1.3.2.0's line.

**Results:** not measured as versions of their own (the bounce had burned whole step budgets in 120-run tests;
`backlog.old.md`).

### 1.3.2.0 (Nov 11-12: `6b97144e`, tag `v1.3.2.0` at `8ee1fc23`; round 19)

The bombs version: the same route and its 20 families, every one retrained from scratch with bombs (the owner: "this
is the new baseline").

- Bombs by family: `free` as a weapon for goriya, darknut, moldorm, Dodongo and Manhandla, `required` where the route
  needs a wall opened, the 10th damage-free kill's forced bomb drop for everyone else; a route bomb reserve ("from
  here to there never below X", Dungeon 3's 5).
- Observation schema 3 (bomb inputs), bomb rewards and metrics, an early stop that also waits on the bomb value.
- Route: 2:0x5e cleared for its Goriyas' bomb drops before the Moldorm (the owner's idea, `treasure: clear`); half
  the resets keep the start state's bombs.

**Results** (1000 runs, round 18 in brackets): from the start to Dungeon 1 55.2% (49.3%), to Dungeon 3 36.3%
(29.2%), to the Dungeon 3 triforce 10.3% (10.6%, within noise); the Dungeon 2 segment 82.1% (87.6%), Dungeon 3 39.5%
(40.6%). Dungeon 3's Darknuts stalled on 3:0x59 and Manhandla killed more.

### 1.3.3.0 and 1.3.3.1 (Nov 12: `671fa8a7`, `376b6fdd`; branch `clips`, not tagged)

The first clips, on a branch off 1.3.2.0; they reached `main` inside 1.4.1.0.

- 1.3.3.0: the overworld screen wrap as a macro the executor plays, offered to the policy in several designs
  (`clips.action`, `clips.shaping`).
- 1.3.3.1: the dungeon block clip, proven in 1:0x22; only clips that help the route are offered.

**Results:** no chains on these versions; round 21's design experiments measured them (the overworld wrap gauntlet
99.4% against 96.2-96.4% without, `round21-gauntlet.md`).

### 1.4.0.0 (Nov 12: `fb730ef2`, tag `v1.4.0.0` at `04b18b83`; round 20)

Four dungeons: `d2first` runs on through Dungeon 4 (steps 98-122) to its triforce. 24 networks.

- New route stretch: by raft from 0x55 to Dungeon 4's island, its two keys, the ladder, Gleeok.
- New families `vire`, `like-like`, `ow-peahat`, `boss-gleeok`; the old ones continued on Dungeon 4's rooms.
- Engine gaps found in the disassembly closed: the ladder over 0x30's water, Gleeok's heads observable (and his body
  no longer paying boss hits), kills the game does not count resetting the stall penalty; a game-faithful doorway
  mask after the explorer found Link walking through a wall.

**Results** (1000 runs, `r70-1k`): Dungeon 4 21.1% to its triforce (Gleeok: 199 deaths and 188 stalls per 1000, Link
arriving with 2.1 hearts), Dungeon 3 38.6%, after Dungeon 3 99.8%; from the start to Dungeon 3 26.7% (36.3%) and **to
the Dungeon 4 triforce 1.8%**. The Dungeon 2 segment fell to 68.6%: a doorway mask one pixel too strict, found in round
21.

### 1.4.1.0 (Nov 12-13: `10d47bae` on `clips`, merged at `9978e5d`, tag `v1.4.1.0` at `74109a5e`; round 21)

Clips and attacks by buttons, with all 24 families retrained from scratch (`r71`, 93.7M steps).

- New action kind `CLIP <N|S|W|E>` (the speedrunners' grid-point skip: block clip, ladder clip; the overworld wrap),
  legal only where a skip is available and useful, played as a macro; on for vire, stalfos-keese-gel, trap-room and
  the nine overworld families. Observation schema 4 (`clip_usable_*`), action schema 2.
- Attacks played by buttons: no writes of Link's facing or the selected item; B items chosen on the sub-screen.
- New route stretch: Dungeon 4's shortcut onto 0x21's island and off it by the ladder clip.
- The doorway mask's one-pixel error fixed.

**Results** (1000 runs, `r71-1k-dks`; round 20 in brackets): Dungeon 2 85.6% (68.6%), after Dungeon 2 88.6%, Dungeon 1
99.1%, after Dungeon 1 99.4%, Dungeon 3 46.5% (38.6%), after Dungeon 3 100%, Dungeon 4 56.4% (21.1%); from the start
to Dungeon 3 56.7%, **to the Dungeon 4 triforce 12.8%** (1.8%; 15.4% with darknut on argmax, within noise). Clips
learned in every room tried (4:0x50: 2,397 of 2,400 episodes clip; −0.07 to −0.17 hearts against −0.48 to −0.61 on
the zigzag).

### 1.4.1.1 (Nov 13-14: `90c5d13b`, tag `v1.4.1.1` at `637fdde6`; round 22)

The same route with a run-time clips switch, so every number from here to round 25 is given clips on and off. 12
families retrained (`r72`), 12 kept.

- Every start writes a natural RNG state (a phase of the game's own cycle), so start states no longer share one RNG.
- A clips switch any decision can flip; the route holds both ways through each stretch where they differ
  (`d3-approach`, `d4-island`) and re-plans from where Link stands; clip families train half their episodes with
  CLIP masked.
- New route stretch: the overworld wrap exit (`CLIP E` into 0x74 on the way to Dungeon 3).
- The BOMB mask for `required` families (BOMB only where it opens the route's wall).

**Results** (1000 runs, clips on | off): Dungeon 2 87.2 | 87.2%, after Dungeon 2 91.0 | 89.5%, Dungeon 1 98.9 | 99.5%,
after Dungeon 1 99.9 | 98.9%, Dungeon 3 50.4 | 50.4%, after Dungeon 3 100 | 100%, Dungeon 4 73.6 | 56.8%; from the
start to Dungeon 3 59.5 | 41.0%, to the Dungeon 3 triforce 19.5 | 10.2%, **to the Dungeon 4 triforce 13.6 | 4.8%**.

### 1.4.1.2 (Nov 14: `2c676388`, tag `v1.4.1.2` at `9478d1e5`; round 23)

Round 22's team under one act mode.

- Act modes over the joint action (`temperature`, `floor`, `escape`, `relax`, combinations) in every tool that plays;
  the owner chose `temperature:0.5+relax:16` for play, and the same rule became every model's mode for evaluation and
  training (before, argmax for some families and sampling for others).
- One bomb per wall; 3:0x5b's last pass fights first.

**Results** (1000 runs, clips on | off): Dungeon 2 89.9 | 89.9%, Dungeon 3 60.1 | 60.1%, Dungeon 4 78.6 | 64.7%; from
the start to Dungeon 3 63.8 | 48.1%, to the Dungeon 3 triforce 33.5 | 22.7%, **to the Dungeon 4 triforce 28.7 | 15.3%**
(1.4.1.1: 13.6 | 4.8%).

### 1.4.2.0 (Nov 14: `97967058`; not tagged; the start of round 24)

Three item actions added; the 1.4.1 networks widened, not retrained.

- New action kinds `RECORDER`, `FOOD <dir>`, `CANDLE <dir>` (every older index kept), masked unless a family's
  `items:` turns them on; observation schema 5.
- `triforce convert-model`: older checkpoints widened to the new layout, proven to act bit for bit as their sources.

**Results:** play unchanged by construction (the converted models were proven identical); no chains of their own.

### 1.5.0.0 (Nov 15: `5c2b23c4`, tag `v1.5.0.0` at `2b297cb6`; round 24)

Five dungeons: `d2first` runs from the game start to the Dungeon 5 triforce (209 steps). 28 networks.

- New route stretch: after Dungeon 4 the leg (steps 134-185: the raft back to 0x55, 0x5f's heart container round the
  world or by the coast, the Lost Hills by the plain way, 0x2f's take-any cave, 0x0f's 100 rupees), then Dungeon 5
  (steps 186-208: 0x64 by the block clip, the Olive Garden 5:0x05 for the recorder, Digdogger by bomb, recorder and
  sword).
- New families gibdo, pols-voice, dodongo-crossing, boss-digdogger.
- Training starts: state names for later visits, staged starts (`room_kills`), states after a fight the explorer
  could not clear harvested from chain runs.

**Results** (1000 runs, clips on | off): Dungeon 2 89.9 | 89.9%, Dungeon 3 67.3 | 67.3%, Dungeon 4 81.3 | 59.7%, the
leg to Dungeon 5 99.8 | 92.0%, Dungeon 5 34.9 | 22.0% (46.4 | 34.0% with the promoted darknut `r74-darknut5`); from the
start the Dungeon 4 triforce 33.3 | 19.2%, **the Dungeon 5 triforce 4.8 | 0.3%**.

### 1.5.0.1 (Nov 16: `a71ebc4c`, tag `v1.5.0.1` at `b6d4e351`; round 25)

The same route, the round of "leave rooms with more hearts".

- New reward: `reward.heart_value`, hearts priced by the fitted chance to reach the next triforce alive from that
  route step with that many hearts, on for eleven dungeon families (continued with SIL); heal targets for three
  overworld families.
- Manhandla fought with dropped bombs; path costs keep the objective's path off monsters' spawn spots (0:0x27,
  0:0x1f, 5:0x37); `train --demos` clones a scripted rule's clean runs during PPO (5:0x37's column).
- The honest planner, reported beside the models from here on.
- Measuring rules: a 1000-run chain has a standard error of about 1.6 points; an arm needs three seeds and a chain
  comparison; checkpoints chosen by a chain segment (`select-checkpoint`). Jobs go through `triforce job`.

**Results** (1000 runs, round 24 → 25, clips on | off): Dungeon 2 89.9 → 88.2 | 89.9 → 88.2%, Dungeon 3 67.3 → 81.0 |
67.3 → 81.0%, Dungeon 4 81.3 → 92.3 | 59.7 → 79.9%, the leg 99.8 → 99.9 | 92.0 → 97.0%, Dungeon 5 34.9 → **68.0** |
22.0 → 47.5%; **from the start to the Dungeon 5 triforce 4.8 → 19.7 | 0.3 → 2.6%**. The planner beside: the full
route 82.6% (clips on).

### 1.5.1.0 (Nov 16-18: `11438788`, until 1.6.0.0; not tagged)

Observation schema 6, the one input change made for Dungeon 7 to Ganon; the 1.5.0 networks converted, not retrained.

- New observations from per-type object rules (`zelda::object_rules`): whether a touch harms, weapon target,
  visibility, phase and phase time, touch damage, in-line, the Wallmaster lure; 8 entity, 6 Link and 4 room-state
  features; hidden positions read from RAM, last seen or zero.

**Results:** the converted 1.5.0 runs proven identical (round 26); no chains of the version on its own.

### 1.6.0.0 (Nov 18: `3b0b58e9`, tag `v1.6.0.0` at `6b4157f5`; round 26)

Seven dungeons by the route's count: `d2first` runs to the Dungeon 7 triforce (248 steps). 28 networks.

- New route stretch (steps 209-247): 0x0b's recorder and the whirlwind to 0x45, 0x44's arrows, the Armos and the
  food at 0x34, 0x42's pond, Dungeon 7 with the Grumble, the red candle, two Digdoggers, the Wallmaster room and
  Aquamentus.
- Engine and route vocabulary: recorder entries and exits, item waypoints, overworld secrets as targets, shops and a
  rupee reserve, ARROW masked unless a family's items allow it, opened bomb holes as doorways; the granted-bomb rule
  (a granted bomb opens the wall and nothing else); 7:0x0d's Wallmaster objective and clock fixes with demonstrations.
- Two teams trained; the second adds goriya `r76-goriya4-s78` and ow-zora `r76-zora-arrows`. From here on chains are
  clips on only (the owner, Nov 18).

**Results** (1000 runs; round 25 in brackets): Dungeon 2 81.5% (88.2%; goriya's 2:0x5e and 2:0x1e), after Dungeon 2
95.7%, Dungeon 1 99.0%, Dungeon 3 78.2%, Dungeon 4 92.1%, the leg 99.9%, Dungeon 5 68.7%, after Dungeon 5 98.7%,
**Dungeon 7 40.0%** (7:0x1b took 324 of 872); from the start the Dungeon 5 triforce 22.4% (19.7%) and **the Dungeon 7
triforce 6.7%**. The planner beside: the full route 85.0%, Dungeon 7 95.3%.

### 1.7.0.0 (Nov 18: `32c2149d`, tag `v1.7.0.0` at `e7065bae`; round 27)

The magic sword and Dungeon 6: `d2first` runs to the Dungeon 6 triforce (281 steps). 31 networks.

- New route stretch (steps 248-280): out of 0:0x42, the wrap exit of 0x51 or the Lost Woods, the graveyard, 0x21's
  gravestone pushed for the magic sword, Dungeon 6 with its blocks, bomb wall, tunnel and Gohma.
- New families ow-ghini, wizzrobe, boss-gohma; wizzrobe taught by the honest planner's play of 0x38 and 0x3a and
  priced by the heart value; ow-lynel a joint continuation over both its lanes.
- The owner's rules: no potions, no CONTINUE.

**Results** (1000 runs; round 26 in brackets): after Dungeon 2 94.0% (95.7%), Dungeon 1 98.7%, Dungeon 3 78.6%,
Dungeon 4 91.1%, the leg 100%, Dungeon 5 69.1%, **Dungeon 6 30.0%** (6:0x3a killed 376 of 824 entering it with 6.1
hearts); the full route: Dungeon 1 left 64.2%, the Dungeon 5 triforce 23.3%, the Dungeon 7 triforce 6.1%, **the
Dungeon 6 triforce 1.1%**. The planner beside: Dungeon 6 67.3%, the full route 37.7%.

### 1.9.0.0 (Nov 18-19: `106351f8`, tag `v1.9.0.0` at `10d4522a`; rounds 28 and 29)

The whole game: `d2first` runs from the game start to the ending (358 steps), one team of 34 networks for every
room. Dungeons 8 and 9 landed together, so there is no 1.8.

- New route stretches: the leg to Dungeon 8 and Dungeon 8 (the whirlwind out of 0x22, 0x6d's burned tree, the Blue
  Darknut rooms, Pols Voice, the four-headed Gleeok); the leg to Dungeon 9, Dungeon 9 and Ganon (the silver arrow,
  the red ring, `exit: end`).
- New families lanmola, patra, ganon; route vocabulary for burned and pushed secrets, a step's own bomb mode, kills
  the game keeps across visits (`banked_kills`), ARROW only at the brown Ganon.
- Scripted rules from the walks used as demonstrations, harvested handoff states at the chain's hearts, and the
  census (`route-team.md`) that chose one team for the whole route.

**Results** (1000 runs a segment, `r29land-1k`): Dungeon 2 91.6%, after Dungeon 2 93.2%, Dungeon 1 99.8%, Dungeon 3
82.7%, Dungeon 4 91.7%, the leg 100%, Dungeon 5 72.2%, after Dungeon 5 98.0%, Dungeon 7 46.6%, after Dungeon 7 99.1%,
Dungeon 6 31.4%, after Dungeon 6 100%, Dungeon 8 24.8%, after Dungeon 8 100%, **Dungeon 9 to the ending 31.0%**. The
full route: Dungeon 1 left 73.0%, the Dungeon 5 triforce 26.2%, the Dungeon 7 triforce 9.1%, the Dungeon 6 triforce
1.7%, Dungeon 9 entered 0.4%, **the game ended 0 of 1000**. The models end the game from Dungeon 9's entrance (310 of
1000) and from the Dungeon 8 triforce (105 of 1000). The planner beside ends the game in 29.7% of 300 runs from the
start.

### 1.9.0.1 (Nov 20: `e79307cd`, tag `v1.9.0.1` at `8b48df60`; rounds 30 and 31)

The owner's eight phase IX changes to `d2first` (354 steps) and the optimisation round's team.

- Route: Dungeon 1 left by its S door and entered again, rooms dropped (0x72, 0x74, 0x52, 0x42), 0x53's wall bombed
  into 0x43, Dungeon 6's drops and keys, Dungeon 8's 0x4c key left, Dungeon 9's 0x66 and 0x55 changed.
- One family at a time retrained or continued, each by its own gate: wizzrobe, patra, boss-manhandla, gibdo, goriya,
  darknut, ow-lynel, ow-octorok (the Dungeon 1 trip).

**Results** (1000 runs, `r31land-1k`, paired with 1.9.0.0): Dungeon 3 85.3%, Dungeon 5 76.3% (+4.1), Dungeon 7 56.9%
(+10.3), Dungeon 6 28.5% (the new route costs 10.3), Dungeon 8 36.3% (+11.5), Dungeon 9 13.7% (−17.3: 9:0x56, new to
like-like); the full route: the Dungeon 5 triforce 30.4%, the Dungeon 7 triforce 12.5%, the Dungeon 6 triforce
3.0%, Dungeon 9 entered 1.3%, **the game ended 0 of 1000**. The planner beside ends it in 9.0% (9:0x56 again).

### 2.0.0.0 (Nov 19-21: `a5a03a7c` on `r32-routes`, tag `v2.0.0.0` at `9ee2f6a5`; rounds 32 and 33)

The second route, `d3first`, the speedrunners' order, becomes the main one: the wood sword until the magic sword,
Dungeons 3, 4, 1, 5, 2, 7, the magic sword, 6, 8, 9 (304 steps). 49 networks play it (fifteen families with a
`d3first`-only run for the wood-sword half); `d2first` stays playable with 34.

- New route: recorder warps, the blue candle, three secrets, heart rock's containers, the ladder, the raft and
  0x47, the dungeons shared with `d2first` as route blocks.
- Cheated and counted: bomb walls, the bag topped up to 8 at Dungeon 5's entrance, rupees at the shops.
- `d2first` (round 33): new wizzrobe, vire and like-like defaults.

**Results:** the owner's condition for 2.0, a golden path from console power on with the models alone reaching the
ending, met three races of three (88, 95 and 64 runs on the landed code; the first golden replay 153,308 frames).
Segments (1000 runs, `r32land-1k`): the legs 98.7-100%, Dungeon 3 12.6%, 4 6.2% (Gleeok), 1 89.6%, 5 24.0% (the
Olive Garden), 2 100%, 7 9.4% (7:0x1b), 6 35.2%, 8 37.8%, 9 48.9%; from the start the Dungeon 3 triforce 13.6% and no
run past Dungeon 5; the planner ends the game in 17.7% of 300. On `d2first` the full route ended the game for the
first time, 4 of 1000. 250 test (back-tested, `docs/runs250/backtest.md`): 0 of 250 won, the Dungeon 3 triforce in 39.

### The 2.0.0.x versions on the 250 test

Every 2.0.0.x version played back on its own tagged code and team (`docs/runs250/backtest.md`, Nov 23-24; 2.0.0.13
on from `main`): 250 runs of `d3first` from console power on, runs taking each triforce, in route order.

| version | won | D3 | D4 | D1 | D2 | D5 | D7 | D6 | D8 | D9 | golden race (speedrun time) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2.0.0.0 | 0 | 39 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 39:24.789 |
| 2.0.0.1 | 0 | 51 | 9 | 5 | 3 | 3 | 1 | 1 | 0 | 0 | 39:58.167 |
| 2.0.0.2 | 0 | 68 | 7 | 7 | 2 | 2 | 0 | 0 | 0 | 0 | 40:17.352 |
| 2.0.0.3 | 0 | 68 | 7 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 40:01.512 |
| 2.0.0.4 | 0 | 68 | 7 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 39:09.264 |
| 2.0.0.5 | 0 | 68 | 7 | 7 | 5 | 1 | 0 | 0 | 0 | 0 | 40:36.654 |
| 2.0.0.6 | 0 | 52 | 6 | 6 | 3 | 0 | 0 | 0 | 0 | 0 | 40:32.943 |
| 2.0.0.7 | 0 | 52 | 6 | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 39:33.524 |
| 2.0.0.8 | 0 | 71 | 33 | 29 | 28 | 15 | 0 | 0 | 0 | 0 | 42:17.188 |
| 2.0.0.9 | 0 | 71 | 33 | 32 | 31 | 13 | 7 | 0 | 0 | 0 | 39:17.784 |
| 2.0.0.10 | 0 | 94 | 57 | 54 | 53 | 23 | 9 | 0 | 0 | 0 | 38:18.564 |
| 2.0.0.11 | 1 | 94 | 57 | 54 | 53 | 23 | 9 | 2 | 1 | 1 | 37:51.060 |
| 2.0.0.12 | 0 | 94 | 57 | 54 | 53 | 23 | 9 | 2 | 2 | 0 | 38:03.922 |
| 2.0.0.13 | 1 | 94 | 57 | 54 | 53 | 27 | 15 | 2 | 2 | 1 | not raced |
| 2.0.0.14 | 0 | 94 | 57 | 54 | 53 | 27 | 12 | 3 | 2 | 0 | not raced |
| 2.0.0.15 | 1 | 104 | 61 | 58 | 57 | 28 | 13 | 3 | 3 | 1 | not raced |
| 2.0.0.16 | 1 | 104 | 61 | 58 | 57 | 28 | 13 | 3 | 3 | 1 | not raced |

Not measured: the 1.x versions (the back-test's replay stream was not ported to them).

### 2.0.0.1 (Nov 21: `35c453f2`, tag `v2.0.0.1` at `273a1ed5`; round 34)

`d3first`'s wood-sword dungeons trained segment by segment: eight new route runs, 51 networks.

- New route runs for zol, stalfos-keese-gel, darknut (Dungeon 3), vire, like-like, boss-gleeok (Dungeon 4),
  wallmaster and boss-aquamentus (Dungeon 7 and Dungeon 1's boss).
- Dungeon 4's entrance states rebuilt from chain arrivals; Dungeon 7's bag reserve; a training config may count a
  truncation (no damage, max steps, no room progress) as a failure.

**Results** (1000 runs, paired with 2.0.0.0): Dungeon 3 12.6 → 18.0%, 4 6.2 → 18.5%, 1 89.6 → 92.7%, 5 24.0 → 27.4%,
7 9.4 → 38.7%; from the start the Dungeon 3 triforce 22.7% (13.6%), Dungeon 4's 2.8%, three runs past Dungeon 5. The
golden path ends the game three of three (51, 71, 60 runs); the planner 25.0% of 300.

### 2.0.0.2 (Nov 21: `57a2ff02`, tag `v2.0.0.2` at `83d9960c`; round 34)

Two late round-34 runs promoted: darknut `r34d5-dg-s362` (the Olive Garden) and boss-digdogger `r34d7b-dig-s1`.

- Two networks replaced; no code or route data changed.

**Results** (1000 runs, paired with 2.0.0.1): Dungeon 3 18.0 → 23.2%, 5 27.4 → 33.0%, 7 38.7 → 42.3%; from the start
the Dungeon 3 triforce 23.5%, Dungeon 5's 12 of 1000; none ends the game. Golden path three of three (50, 105, 53
runs).

### 2.0.0.3 (Nov 21: `2239019b`, tag `v2.0.0.3` at `7687066f`; round 35)

The owner's checkpoint: the bag of 8 bombs at Dungeon 5's entrance removed ("The goal is a win without a cheated
bomb"); the learned guard (a second network vetoing actions that get Link hit) held back.

- Route: no grant anywhere but a required wall's missing bomb (still cheated and counted).

**Results** (1000 runs): Dungeon 5 from its entrance 33.0 → 7.0% without the bag; from the start 6 runs take Dungeon
5's triforce and **one run of 1000 ends the game, the route's first single run to the ending**
(`single-run-d3first-old-order-51-networks`). Golden path three of three in 161 runs.

### 2.0.0.4 (Nov 22: `cd1499ad`; not tagged; round 36's lab)

The lab's extensions on `main`; no network's play changed.

- Training of networks of any shape over several families, teachers, memory, auxiliary targets and the hit head's
  labels, sampling across families by need; the yardstick bench (`docs/lab.md`). Observation schema 7 (Link-relative
  geometry and contact per entity) available to new networks.

**Results:** play unchanged (the 250 test equals 2.0.0.3's: the Dungeon 3 triforce in 68 of 250).

### 2.0.0.5 (Nov 22: `e64a26da`, tag `v2.0.0.5` at `cde00236`; round 36)

The owner's corrected `d3first` becomes the route: Dungeons 3, 4, 1, 0x2c's heart, 0x2d's rupees, Dungeon 2, 0x0f's
100 rupees, Dungeon 5, two wrong warps over a dock, the arrows, Dungeon 7 with the wood sword, the magic sword,
Dungeons 6, 8, 9 (292 steps). Today's 51 networks, not retrained.

- New route: the recorder's plays found by search; the two dock warps played by the engine as a macro once Link is on
  the dock; 861 entry states.
- The program begins (the owner, Nov 21): this route, played in the end by one network (backlog Track M).

**Results** (1000 runs a segment): Dungeon 3 23.2%, 4 18.5%, 1 93.3%, the leg to Level 2 99.1%, 2 99.8%, 5 31.2%, Level
5 to Level 7 78.9%, 7 13.1%, Level 7 to Level 6 22.4%, 6 36.4%, out of Level 6 32.0%, 8 40.0%, 9 51.3%; from the start
30 runs take Dungeon 2's triforce, 6 Dungeon 5's, none ends the game. The golden race wins three of three (411 runs).

### 2.0.0.6 and 2.0.0.7 (Nov 22: `0c8def57`, tag `v2.0.0.6` at `cbc1dfa9`; `3242772d`, tag `v2.0.0.7` at `1caa6475`; rounds 37 and 38)

Two route-rule changes from the owner on the same 51 networks.

- 2.0.0.6, the owner's bomb rules (`bomb_rules:`): bombs a weapon only in dungeon rooms the route fights, none but the
  wall's in 3:0x5b, a room's bombs fought for with 7 or fewer and then taken, the 10th kill's bomb always; 2:0x5e
  crossed.
- 2.0.0.7: 0:0x2d's bomb-wall rupee cave dropped (290 steps); the states after it rebuilt.

**Results** (1000 runs): 2.0.0.6 took weapon bombs in crossings from 1.94 to 0 a run and cheated wall bombs from 0.66
to 0.30 a run, Dungeon 3 −1.5, 4 +2.6, 2 −1.4 points (the rest within noise). 2.0.0.7: to Level 2 98.3%, Dungeon 2
99.3%, Level 5 to Level 7 83.1%; from the start 28 of 1000 take Dungeon 2's triforce. The golden race still ends the
game three of three on both.

### 2.0.0.8 (Nov 22: `673a1dce`, tag `v2.0.0.8` at `d8596127`; round 38)

Hit heads: eight families play `d3first` with a supervised hit network beside the policy that lowers the logits of
actions it reads as getting Link hit (`hitshift`).

- New networks for boss-gleeok, like-like, vire, stalfos-keese-gel, darknut, boss-manhandla, zol, gibdo (route runs
  `r38head-fam-w8`/`w4`).
- Every replay kept from here (`replays/`, `/backup/triforce`).

**Results** (1000 runs paired with 2.0.0.6's): Dungeon 4 21.1 → 68.4%, Dungeon 3 21.7 → 39.2%, Dungeon 5 31.2 →
66.1%. 250 test: the Dungeon 4 triforce 6 → 33, Dungeon 5's 0 → 15.

### 2.0.0.9 (Nov 22: `1bf27323`, the tag's commit; round 38)

Goriya's hit head and Dungeon 8's three, through a `d3first`-only late pointer (`late.<route>`, so `d2first` plays
as before).

- New networks: goriya `r38ppo-aux-s38`; darknut, boss-gleeok, boss-manhandla for the steps after the magic sword.

**Results** (1000 runs, paired): Dungeon 8 39.1 → 80.9%; goriya on 2.0.0.7: Dungeon 1 92.8 → 97.7%, Dungeon 7 15.2 →
76.9%. 250 test: the Dungeon 7 triforce 0 → 7.

### 2.0.0.10 (Nov 22-23: placeholder `fa924833`, then `f1d67ff5`, the tag's commit; round 38)

The owner's diagnosis from watching the golden replays, retrained: the opening's back-and-forth came from a network
trained on `d2first`'s opening, and 8:0x4c's from a Pols Voice trained when the key was still taken. 53 networks.

- New networks: a fresh ow-start for `d3first`'s opening (`r38t-start-fresh2`), a pols-voice that leaves 8:0x4c's key
  (`r38t-pols`); 3:0x5b's wall-and-bombs order pinned by a test.

**Results:** the opening's aimless frames 661 → 4 a visit; 8:0x4c 92.2 → 100% (5,267 → 2 aimless frames). 250 test: the
Dungeon 3 triforce 71 → 94 of 250, Dungeon 4's 33 → 57, Dungeon 5's 13 → 23.

### 2.0.0.11 and 2.0.0.12 (Nov 23: `0a3e7915`, tag `v2.0.0.11`; `d2a95040`, tag `v2.0.0.12` at `484892bf`; round 38)

The overworld legs after Dungeons 7 and 6 played by networks trained on this route's rooms; new reward terms. 54
networks.

- New networks: ow-lynel `r38rw-lynel` for the leg after Dungeon 7 (0:0x50 E → S, 0x60), a late ow-lynel
  `r38d-lynel-late-s1` for the leg after Dungeon 6 (0x22's recorder out of Level 6).
- Rewards: rupees paid once a pickup by size (not every counted rupee); beam terms (a range-kill bonus, credit for a
  shot projected to meet a monster, a cost for a lined-up shot not taken), at weights that did not yet change how the
  continued networks fought.
- Beside the team, the first one network: `r38c/d2b-all-w8`, a 768-wide `hitnet` copied from the 51 on `d3first`
  states, corrected on its own states (DAgger), with the shared hit head, then fine-tuned on the legs' teachers
  (`round38-one.md`).

**Results:** the leg after Dungeon 7 24.0 → 99.0% (300 paired), the leg after Dungeon 6 33.3 → 100.0% (1000 paired).
250 test 2.0.0.11: **1 of 250 won**, the first back-tested win; the golden race 37:51.060. The one network on
2.0.0.11's code: **6 of 1000 single runs end the game** (the team's 250 test 1 of 250); on 2.0.0.12's code its golden
race wins three of three in 93 runs, the fastest 37:56.351 (211th of 463 on the Any% No Up+A board).

### 2.0.0.13 (Nov 23: `b814f953`, tag `v2.0.0.13` at `fc0ee1a6`; rounds 38 and 39)

`d3first` leaves Moldorm alive in 2:0x3e and takes 9:0x56's key instead (the owner); moldorm and like-like fine-tuned
for it. 55 networks.

- Route: 2:0x3e crossed, 9:0x56 fought for its key; the states after it set to one key fewer.
- New networks: moldorm `r39t-moldorm`, like-like `r39t-like`.

**Results** (1000 runs): Dungeon 9 with the route change alone 6.7% (like-like stuck at the key), with the
fine-tunes 51.2% (53.0% on the old route, fighting the Moldorm). The 2.0.0.13 team against the one network on every
segment (`round38-one.md`): level from the start, 2 against 2 endings of 1000, the product of the 17 segment rates
0.0176 against 0.0175 (one in 57 each); the network ahead on Dungeon 6 (+12.9), 5 (+5.6), 3 (+3.8), behind on Dungeon 4
(−16.0, Gleeok), 7 (−11.0) and 8 (−6.2). 250 test: 1 of 250 won.

### 2.0.0.14, 2.0.0.15 and 2.0.0.16 (Nov 23-24: `453f074a`, tag at `902205ec`; `2b0ccde2`; `2f73da4b`; round 40)

The last revisions of 2.0.0's team: engine and rule changes, no network retrained, while the fresh generation's
pieces were built.

- 2.0.0.14: the engine plays `d3first`'s two whirlwinds (0:0x0b out of Level 5, 0:0x22 out of Level 6) as macros.
- 2.0.0.15: Manhandla's 3:0x4d and Dodongo's 2:0x0e are bomb-weapon rooms: Link is never left without a bomb there
  (the owner).
- 2.0.0.16: a training reset writes Link's hearts as the game holds them (the owner's find playing `emu-zelda.py`: a
  whole number of hearts below full had been written with an empty partial heart, which survives one hit more; 22% of
  zol's and 15% of darknut's resets). Chains and replays carry the game's own hearts, so play is unchanged.

**Results:** 2.0.0.14 the leg out of Dungeon 5 82.3 → 89.3% (1000 paired); 2.0.0.15 Dungeon 3 39.2 → 45.7%. The
baseline the fresh generation is compared with, 2.0.0.15's team at 1000 runs a segment (`r40-2015-1k`; from the
previous triforce | from the entrance): Dungeon 3 43.2 | 45.7%, 4 45.8 | 68.4%, 1 98.5 | 98.9%, 2 93.8 | 99.8%, 5 57.8 |
66.1%, 7 42.5 | 77.3%, 6 18.2 | 35.7%, 8 88.9 | 88.6%, 9 22.6 | 51.2%; **the full route 3 of 1000 (0.3%)**. 250 test: 1 of
250 won on 2.0.0.15 and 2.0.0.16, every run identical between the two.

### 2.0.1.0 (Nov 23 to now: round 40's first commit to Nov 27; in progress, the fresh generation)

The first fresh generation and the first version whose product is one network (decisions 2026-11-23, "2.0.1.0's team
is the one network"): every network trained from scratch by today's pipeline, teachers by group first, then one
student from them. Not stamped yet: the `VERSION` file says 2.0.0.16 until the finished network's 1000-run check
(next section).

- New observations (behind switches in the route's task files): schema 8's `entity_threat` (range-aware contact, the
  shooter's threat lane, the bomb carrier), schema 9's `chase` and `entity_chase` (where the monsters are sent, what a
  grid-point walker may do next), schema 10's `entity_shooter` (when each shooter throws next) for the Goriya,
  Wizzrobe and overworld teachers.
- New network: `hitnet`, Link's token over the monster tokens with a hit head trained in the same pass and fed to the
  policy, 768 wide everywhere (the owner, Nov 24).
- The owner's replay feedback built in: Dungeon 3 by his plan (two rooms crossed, one Darknut fought for its bombs,
  3:0x69 crossed by a line, the reserve of 5 in 3:0x59), Darknuts fought his way (`docs/owner/darknuts.md`), the
  executor's closing no-button frame removed, the docks played with no help from the engine, a cheated wall bomb only
  when the wall is the objective, no weapon bomb in a room fought for its bombs, real spawn layouts and drop counters
  only, Link's hearts as the game holds them.
- Rewards: the aimless-walking penalty, the Bubble touch, the exposed-attack penalty (Goriyas and the walkers), a
  stronger block reward for the Goriyas; rupees per pickup and the beam terms trained from the start.
- Training data: start states walked from power on (one per door), curricula re-explored in the route's layout,
  harvested states audited; clips on in every task (the owner, Nov 24).

**Results:** none yet for the student. The teachers' first reads are in the next section.

## The current workflow: the fresh generation (2.0.1.0)

**Stage 1: group teachers** (`docs/lab.md` "Groups of families", `triforce/groups.yaml`). Each teacher is one
`hitnet` trained from scratch with PPO on its members' route tasks at once (`python -m triforce.lab.train --group
NAME`; the network, the hit loss and the look-ahead labels from the file). The groups: overworld (ten families),
walkers (eight), others (four), rooms (two), and teachers of their own where a group could not learn a family:
darknut (Nov 25), goriya (Nov 26), wizzrobe (Nov 27), and each of the seven bosses. A run is sized by its families
(about 3 million steps a family: 30M for darknut, goriya, wizzrobe and the overworld, 24M walkers, 12M others, 6M
rooms, 3M a boss; `scripts/experiments/r41_stage1.sh`). Seed 1 of each is judged before seeds 2 and 3 are spent (the
owner, Nov 24).

**Judging a teacher.** A quick read the moment a training finishes (a few minutes of chains from each dungeon's
entrance beside the previous team on the same starts), then chains of chosen checkpoints in their families' slots of
2.0.0.15's team (`triforce eval-batch NAME --teacher GROUP=RUN`, `triforce judge checkpoints`), from the previous
triforce and from the entrance, paired by seed against the baseline `r40-2015-1k` and its control on the generation's
new starts (`triforce judge compare`). The generation is judged from the previous triforce (the owner, Nov 24).

**Stage 2: one network from the teachers** (`docs/lab.md` "Stage 2"; round 38 made its one network this way). Record
the named teachers' play on every member's task with hit labels (`r36_merge_record.py`), cache it, imitate it into
one `hitnet` student (KL over the 352 joint actions, the value, the hit head in the same pass), correct it on its own
states (DAgger: the student plays, the teachers label), then PPO one dungeon at a time with the teacher's term fading
(`lab.train --load ... --teacher ... --teacher-fade`).

**The check and landing.** The finished network gets 1000-run chains on every segment and the full route, every run
recorded as a replay and kept on local disk to be watched and mined; that check is officially 2.0.1.0, and it lands
on `main` with its 250 test and golden race like any version. Changes after it are 2.0.1.x; the next fresh retrain is
2.0.2.0 (WAIT with a direction, statues as entities and the backlog's other "next retrain" entries).

**Where it stands (Nov 27; `round40.md`, the queue).** Stage 1's seed 1 is trained for every teacher but the
Wizzrobes, some restarted after fixes (the start-state draw, Nov 25; the overworld's exit table) or continued
(`gen1-walkers/s1r-c2`, `gen1-overworld/s1r2-c2`, `gen1-boss-gleeok/s1r-c3`). Against 2.0.0.15's team on the
generation's new starts (300 runs paired, `round40.md` "Stage 1 restarted"), the Darknut and walkers teachers took
Dungeon 3 from 1.3% to 61.3% from the previous triforce, Dungeon 5 27.0 → 63.0%, Dungeon 2 76.0 → 96.0%, Dungeon 8
87.0 → 99.3%; Dungeon 7 fell (61.0 → 9.3% from the entrance: 7:0x1b's Goriyas, which led to the Goriya teacher).
The first read of the whole new team (100 runs a segment) was good everywhere but Dungeons 6 (0%) and 9 (5%), traced
to the others teacher's Wizzrobes, which learned to shelter in a Like-Like's hold (masked since) and never learned
the two fights: the Wizzrobe teacher trains now in two arms (continued from `gen1-others/s1r4` and from scratch).
Checkpoints of the Darknut and walkers teachers are being confirmed on 300-run chains. Stage 2 has not started.

## History not in this repository

**`triforce2`, the old development checkout.** This code was developed there from Nov 1 (`9f5eabae`) until Nov 12
(`5d6bed8a`, its last commit), then merged
into `main` with its whole history (`cc26b050`): every branch of `triforce2` (`main`, `better-play`, `heart-stop`,
`replays`, `zelda-asm`) is in this repository's history, and its documents differ from this repository's only by
later edits (the owner's name replaced by "the owner", disassembly citations). What git never held there (its
`models/`, `logs/`, `runs/`, `target/` and `trajectories/`) was carried over by hand: its training logs and `target/`
notes (round 17's review ledger, the Dodongo, Lynel and trap research notes, round 7's stale `todo.md`, the round 8-9
gate outputs) are in the main checkout's ignored folders; its run dirs, chain records (rounds 1-7's `chain-*.jsonl`)
and round 17's `trajectories.prev-r60` are in `/backup/triforce`. Checked for this document: every run dir there with
a checkpoint is in the backup or the shared store, and every chain record in it is in the backup's sources. Nothing
in this document's version history comes only from `triforce2`; from it come the confirmation that the pre-merge
history is complete and where its data went.

**Notes kept outside both checkouts** (on the owner's machine, not in git):

- The design documents the owner brought in: the v3 starting design (`triforce-v2-design.md`, Nov 1: specialists by
  enemy family, no vision, a runner, timed options; it calls this code "v2" and the hand-written project "v1") and
  the Rust core and environment architecture (`triforce-rust-env-architecture.md`, Nov 4: `zcore` cut down from
  TetaNES, FCEUmm as ground truth at first); the better-play plan (Nov 8; `docs/experiments/better-play-reply.md` is
  the reply). `docs/credits.md` counts them as outside documents.
- A status note of Nov 8 (`triforce2-status.md`: the Rust stack, the heart series' numbers above, round 11's retrain
  in progress, Dodongo next), a handoff of Nov 8 (work done in `triforce2` from another session), and a research
  note of Nov 13 (`triforce-next-research.md`: directions in tiers at the end of round 21, the route's 13% to the
  Dungeon 4 triforce for scale).

**Missing from every record:** the code commits of the chains that define 1.3.1.0's and 1.3.2.0's teams ("its code
commits are gone from history", the backup's index; the teams themselves are kept by checkpoint), and the 250 test of
any 1.x version.
