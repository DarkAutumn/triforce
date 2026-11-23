# Round 38

## A Bubble's touch

**The question** (the owner, watching a replay): "link does get hit by the blue bubble often, does he get penalized
for that directly? As that's a thing that costs time and usually hearts because it forces him to dodge for a while."
**Until this round, no.** A Bubble's touch costs no health, and every place a hit was counted (the reward's
`penalty-lost-health`, the yardstick's hits a room visit, the lab's `hit` labels) counts a decision in which Link
lost health.

**What a touch does** (zelda-asm `bank4/bubble-keese.asm` `Bubble_Update`, `bank1/link-collision-harm.asm`
`Link_HarmByObjType`). The flashing Bubble (`$2B`, the only one of the first quest; the blue `$2C` and the red `$2D`
are in the second quest's level blocks only, and the route meets neither) runs the monsters' touch test on Link; a
touch loads `SwordBlockedLongTimer` (`$4C`) with `$10`, which loses one every 10 frames: no sword for 151-160 frames
(the observation's `link` `sword_blocked`, `sword_blocked_time`). The touch also goes through `Link_HarmByObjType`
like any monster's, with a damage of 0: Link is shoved `$20` pixels, made invincible for `$18` frames (no monster
touches him meanwhile, so two touches are at least 24 frames apart), hears the hurt sound, and the kill counters
behind the fairy and the 10th-kill drop are zeroed.

### How often, and what it costs

The chains on the branch, today's 51 networks (`--act model`, clips on), 300 runs a segment from its entry state
(seeds 0-299), each stopped at its segment's exit (`d3` at 0:0x74, `d4` at 0:0x45, `d7` at 0:0x42, `d6` at 0:0x22,
`d8` at 0:0x6d; `d9` to the end of the game): every visit of a room whose game.yaml enemies hold Bubbles, at the
hearts the runs bring (`RoomRecord.bubble_touches`, `sword_blocked_frames`: the frames of the steps begun with the
sword blocked). Hearts lost: health in − health out (≥ 0); a death is a visit whose outcome is `death`.

| room (route step, family) | visits | visits touched | touches a visit | frames without the sword a visit | hearts lost (touched / not) | deaths (touched / not) | frames a visit (touched / not) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3:0x49 (12, zol) | 282 | 29% | 0.32 | 62 | 0.15 / 0.24 | 0.0% / 5.5% | 327 / 296 |
| 4:0x30 (36, vire) | 299 | 34% | 0.37 | 97 | 0.07 / 0.01 | 0.0% / 0.0% | 405 / 368 |
| 4:0x32 (38, like-like) | 295 | 97% | 5.77 | 737 | 0.76 / 0.72 | 5.2% / 0.0% | 2098 / 1321 |
| 4:0x32 (40, like-like) | 269 | 45% | 0.54 | 133 | 0.32 / 0.23 | 4.9% / 9.5% | 439 / 365 |
| 7:0x49 (163, goriya) | 305 | 92% | 3.75 | 527 | 1.08 / 0.77 | 0.7% / 4.2% | 1799 / 1191 |
| 7:0x18 (167, goriya) | 293 | 45% | 0.51 | 149 | 0.26 / 0.79 | 0.0% / 5.0% | 394 / 381 |
| 7:0x0d (175, wallmaster) | 56 | 98% | 6.82 | 875 | 0.10 / 0.50 | 0.0% / 0.0% | 3587 / 892 |
| 6:0x38 (204, wizzrobe) | 300 | 70% | 1.63 | 239 | 5.15 / 3.49 | 13.3% / 6.7% | 1403 / 1185 |
| 6:0x3a (208, wizzrobe) | 249 | 65% | 1.51 | 236 | 3.28 / 3.01 | 27.3% / 53.4% | 1380 / 797 |
| 6:0x1d (210, like-like) | 155 | 0% | 0.00 | 0 | — / 0.00 | — / 0.0% | — / 263 |
| 6:0x2d (211, vire) | 155 | 59% | 1.19 | 209 | 0.23 / 0.05 | 2.2% / 4.7% | 718 / 520 |
| 8:0x4e (224, darknut) | 301 | 46% | 0.56 | 111 | 2.04 / 2.98 | 3.6% / 6.7% | 389 / 345 |
| 8:0x3f (226, darknut) | 211 | 89% | 3.03 | 471 | 2.29 / 1.81 | 25.7% / 33.3% | 1207 / 573 |
| 9:0x56 (248, like-like) | 300 | 26% | 0.29 | 71 | 0.01 / 0.01 | 0.0% / 0.0% | 522 / 443 |
| 9:0x17 (256, wizzrobe) | 298 | 33% | 0.41 | 117 | 1.69 / 2.92 | 0.0% / 1.5% | 614 / 612 |
| 9:0x07 (257, wizzrobe) | 295 | 42% | 0.42 | 52 | 0.04 / 0.84 | 0.0% / 0.0% | 254 / 235 |
| 9:0x07 (259, wizzrobe) | 294 | 5% | 0.06 | 12 | 0.20 / 0.11 | 0.0% / 0.0% | 298 / 229 |
| 9:0x17 (261, wizzrobe) | 294 | 18% | 0.19 | 41 | 0.26 / 0.25 | 0.0% / 0.0% | 438 / 411 |
| 9:0x10 (274, wizzrobe) | 280 | 89% | 6.20 | 812 | 2.67 / 1.17 | 13.2% / 20.0% | 1867 / 1136 |
| 9:0x10 (276, wizzrobe) | 239 | 40% | 0.41 | 92 | 0.00 / 0.00 | 0.0% / 0.0% | 362 / 323 |
| 9:0x31 (283, wizzrobe) | 234 | 6% | 0.06 | 15 | 0.54 / 0.26 | 0.0% / 0.5% | 677 / 532 |
| 9:0x03 (287, like-like) | 187 | 0% | 0.00 | 0 | — / 0.00 | — / 0.0% | — / 206 |
| **all** | 5,592 | 46% | 1.40 | 213 | 1.49 / 0.75 | 7.1% / 3.9% | 1171 / 420 |

**Bubbles touch Link 1.40 times a visit of a Bubble room, and in 46% of the visits**; the rooms where Link fights
with Bubbles around for a long time take most of them: 4:0x32 on the way in (the Breakfast Room, 5.8 a visit),
7:0x49 (3.8), 7:0x0d (6.8), 9:0x10 (6.2), 8:0x3f (3.0). A touch costs 152 frames without the sword (the steps begun
blocked over the touches).

**What it costs the run** (each estimate within a room, so the room mix doesn't count: the all row's 1.49 / 0.75
hearts and 7.1% / 3.9% deaths compare harder rooms with easier ones). Hearts: **0.07 to 0.08 hearts a touch** (a
room's touched visits against its untouched ones, 0.077 a touch over 7,454 touches; the visit's hearts lost on its
touches with the room held, 0.084, the same with the visit's length held too; the hearts lost a frame with the sword
blocked against without, times the 152 frames, 0.071). Time: **about 90 frames of the visit a touch** (the visit's
frames on its touches,
room held). Deaths: none measurable (−0.010 a touch with the room held; the dead visits are short and carry fewer
touches). So the owner's "costs time" holds, and "usually hearts" is a fraction of a half heart a touch, a heart in
about thirteen touches.

### The penalty

**`penalty-sword-blocked`**, `reward.sword_blocked_penalty` (`crates/env/src/critic.rs`, `config.rs`), **on in every
family by default, `REWARD_TINY` = 0.05 a touch** (0 turns it off; `docs/architecture.md` "Experimental reward and
cutoff controls"). It is paid the step a touch lands (`zelda::Change::bubble_touched`: `$4C` rose, from 0 to `$10`
or back up to `$10` while it ran down, or the red Bubble's `IsSwordBlocked` set), once a touch and never per frame
of the block. **A second touch while blocked pays again**: it restarts the 160 frames, so it costs what the first
did; Link's 24 invincible frames after a touch keep two touches apart, and two inside one step pay once.

**Its size.** Against `penalty-lost-health`'s 0.25 a half heart, the measured 0.07-0.08 hearts is 0.035-0.04; with
the ~90 frames it costs (nothing in the default reward prices a frame) the nearest named step is `REWARD_TINY`, 0.05,
a fifth of a half heart's price: enough to make a touch something to avoid, too little to make a network run from a
Bubble into a monster. **Flat, not by the heart value of rounds 25-34**: that pricing (`reward.heart_value`) values a
change of hearts by the route step (V(step, hearts)), and a touch changes no hearts; the hits it leads to are priced
by the route step when they land, so pricing the touch by the step as well would count them twice, and what the
touch costs by itself (the sword, the time) is the same at every step.

**Training only.** No network's play changes: the chain identity check (below) holds.

### Where it is counted

Beside every count of hits: the evaluation's visits (`bubble_touches`, `sword_blocked_frames`; `eval.json`
`visited_rooms` `bubble_touches_per_visit`, `sword_blocked_frames_per_visit`), the chains' room records (same two
fields), the yardstick (`python -m triforce.lab.bench`: `bubble_touches` per episode beside `hits`, the key's and the
total's `bubble_touches_per_visit`, paired by `compare`; results written before this round, `reference.json`
included, lack it: `compare` reports it `missing` and pairs the rest), the lab's rollouts (`RolloutBuffer.blocked`, a
touch landed during the action; `family/<key>/bubble_touches_per_episode`), the look-ahead's trials
(`--try-others`: `tried_blocked` beside `tried_lost`, from the trial's RAM at its end) and the hit head's labels by
config: `--aux OUTPUT=hit_or_blocked:COEFF` (a decision that lost health or that a Bubble touched, the tried
candidates likewise, its held-out precision against those labels; `hit` stays health alone) (`docs/lab.md`
"Auxiliary targets", "The yardstick").

### Does the observation show a Bubble as something to avoid?

Yes; nothing to fix. Schema 6's `touch_harms` is 1 for the flashing Bubble and `touch_damage` 0 (a touch acts on
Link and costs no heart), `weapon_target` 0; schema 7's `entity_contact` `harms` is the game's touch rule
(asm:Link_CheckMonsterTouch: live, `ObjMetastate` 0, no stun timer), which a Bubble passes, and rightly: its touch
does run `Link_HarmByObjType` (the shove, the block), so its contact columns extrapolate it like a monster's (a
wanderer that walls stop, not a flier); `entity_rel` is geometry only. `link`'s `sword_blocked` and
`sword_blocked_time` show the block once it is on. `tests/test_bubble.py` pins it on 7:0x0d's Bubbles: the three
columns, and Link put on a Bubble: the step's touch is seen, costs no health, blocks the sword (`$4C` `$10`, SWORD
masked), and the next step (Link invincible) is no new touch. The blue Bubble (`$2C`, its touch gives the sword
back) would read the same harms and touch columns though its touch helps; it is second-quest only, so left as is.

### Checks

- The chain identity check: main's module and the branch's, 100 runs each (seeds 0-99), `d3first` from `3_7cs_0`
  to 0:0x74 and from `9_76s_0` to the end of the game (Dungeon 9: Bubbles in 9:0x56, 0x17, 0x07, 0x10, 0x31, 0x03):
  **100 of 100 runs identical on both** (every room record, the two new fields aside; the branch's 100 runs saw 44
  and 755 touches).
- The suites: pytest 935 passed, 12 skipped, and `tests/test_jobs.py`'s plan test, which failed only after
  `tests/test_hitnet.py` in the same process (its `begin_job(--no-wait)` left `jobs._START['no_wait']` set; so on
  `main` too): the fixture now resets it, and the two pass together; `cargo test --release --workspace --features
  zelda/zcore` ok; the zelda-asm tests 5 passed; `asm.py check` ok.

## The hit head at play: what a predictor is worth and how it must be connected (branch `r38-hit`)

**The question** (backlog Track M, queue 6; round36.md "What the observation lacks" §7-§8, round 36's arms on
`r36-obs`): does a network that predicts hits well take fewer hits, and how must the predictor be connected to the
action choice for that? Round 36 had a supervised hit network that reads "this action gets Link hit" at 0.86 / 0.81
(precision = recall at the planner's refusal rate, the planner's draws / the model's own) but no reinforcement arm
with a head took fewer hits than its teacher.

**Code base.** Every number below is on this branch unless it says otherwise: main at 2.0.0.5 (the corrected route,
observation schema 7, `hitnet`, the lab) plus one route change, d3first's 7:0x1b `treasure: {if_bombs_below: 8}`:
the six-Goriya room is a fight whenever Link comes in with 7 bombs or fewer, what the owner's floor (2.0.0.6's
`bomb_rules: treasure_below 8`) gives it. 2.0.0.6 was not merged. No act mode below looks ahead in the emulator
except the planner (`temperature:0.5+relax:16+plan:16:1`), reported beside.

### 1. Round 36's last two arms (B, P)

On `r36-obs`'s code and order (2.0.0.3's route, its yardstick entries: goriya@d3first, 120 episodes, paired on the
same states and seeds; `bench run --network models/r36obs-{b,p}/r36obs-{b,p}-s36 --keys goriya@d3first` from that
worktree):

| arm (400 iterations, seed 36) | hits a visit | hearts lost | success | decisions | hits − teacher (an episode) |
|---|---:|---:|---:|---:|---:|
| teacher (r32d1-goriya) | 0.783 | 0.204 | 98.3% | 41.3 | |
| A today's shape | 0.825 | 0.229 | 99.2% | 46.6 | +0.042 ± 0.099 |
| B hitnet, no head | 0.925 | 0.275 | 97.5% | 55.1 | +0.142 ± 0.096 |
| C head as an auxiliary loss | 0.883 | 0.300 | 98.3% | 54.6 | +0.100 ± 0.103 |
| D head feeding the policy tower | 0.767 | 0.229 | 97.5% | 46.4 | −0.017 ± 0.089 |
| P pretrained head (0.886 / 0.857), feeding, trained on | 0.917 | 0.246 | 98.3% | 47.6 | +0.133 ± 0.098 |
| planner beside (today's network + look-ahead) | 0.233 | −0.138 | 100% | 42.4 | |

B − teacher: hearts +0.071 ± 0.081, decisions +13.8 ± 2.7; P − teacher: hearts +0.042 ± 0.082, decisions +6.3 ± 1.5;
P − D hits +0.150 ± 0.116. No arm takes fewer hits than the teacher; the pretrained head (P) does not help either.

### 2. The direct test: the supervised head as a fixed part of the network

**The network** (`triforce/nets/hitshift.py`, arch `hitshift`): two bodies side by side, nothing shared and nothing
trained: today's goriya `@d3first` network (r32d1-goriya, `ZeldaPolicy`) gives the factored action logits and the
value; round 36's supervised hit network (`hitnet`, Link's token over type, features, `rel` and contact, joint 352
head; trained on the 2 million recorded decisions of `goriya-all`, 0.886 / 0.857; `models/r36obs-pre/
hitnet-feed-goriya-all` on `r36-obs`) gives one hit logit `h` a joint action. One forward, no emulator: the joint
action played is

    q(k, d, u)  ∝  p(k, d, u) · exp(−w · softplus(h(k, d, u)))  =  p · P(not hit)^w

then the act mode's temperature 0.5. `softplus(h) = −log P(not hit)`: a safe action keeps its probability, a likely
hit loses it; subtracting the raw logit instead would also re-rank the safe actions by how safe they read (the
form `logit` exists, not measured). The Rust actor draws kind, then direction and duration given the kind, so the
network returns the factored heads that match q's marginals q(k), q(d | k), q(u | k): only the coupling of
direction and duration within a kind is lost. Measured offline on round 36's 247,115 teacher-alone decisions
(`r38_hit.py calibrate`): KL(exact ‖ factored) at most 0.0005 nats a decision, and the head's expected hit rate of
the draw equal to the exact joint's to four digits (0.00994 / 0.00992 at w = 4). w = 0 is the teacher exactly (the
served network played every one of the 780 bench episodes as the Rust actor did: identical numbers).

**Calibration of w** (same data; the share of decisions whose temperature-0.5 draw, on the same noise, differs from
the teacher's, and the head's expected hit rate of the draw): the planner refuses a draw when it is hit, 3.19% of
the teacher's decisions; w = 1: 1.6% refused, expected hit 3.23% → 2.08%; w = 2: 2.4%, 1.47%; **w = 4: 3.1%,
0.99%** (about the planner's rate); w = 8: 3.7%, 0.72%.

**The bench** (`r38_hit.py bench`, one queued job; act `temperature:0.5+relax:16`, clips on; paired on states and
seeds): goriya@d3first on 48 of its yardstick entries × 10 seeds (480 episodes: Dungeons 1, 2 and 7), and Dungeon
7's six-Goriya room 7:0x1b on its three d3first entry states × 100 seeds (300 episodes; task
`scripts/experiments/r38_goriya_0x1b.yaml`: 0-7 bombs on every reset, all six Blue Goriyas, hearts drawn as the
family trains, 15% to full, half full; the room a fight by the route).

| goriya@d3first (480) | hits a visit | hearts lost | success | decisions | hits − teacher (an episode) | hearts − teacher |
|---|---:|---:|---:|---:|---:|---:|
| teacher (r32d1-goriya) | 0.808 | 0.151 | 98.3% | 39.9 | | |
| w = 1 | 0.685 | 0.022 | 97.7% | 40.1 | −0.123 ± 0.031 | −0.129 ± 0.027 |
| w = 2 | 0.600 | −0.028 | 98.3% | 40.2 | −0.208 ± 0.036 | −0.179 ± 0.029 |
| **w = 4** | **0.519** | **−0.084** | 99.0% | 40.6 | **−0.290 ± 0.041** | **−0.235 ± 0.033** |
| w = 8 | 0.529 | −0.095 | 99.0% | 40.9 | −0.279 ± 0.044 | −0.246 ± 0.035 |
| planner beside | 0.248 | −0.227 | 99.4% | 40.7 | −0.560 ± 0.044 | −0.378 ± 0.036 |

| 7:0x1b (300) | hits a visit | hearts lost | won | decisions | hits − teacher | hearts − teacher | won − teacher |
|---|---:|---:|---:|---:|---:|---:|---:|
| teacher | 3.833 | 1.942 | 83.3% | 98.7 | | | |
| w = 1 | 3.170 | 1.278 | 90.3% | 101.5 | −0.663 ± 0.104 | −0.663 ± 0.141 | +7.0 ± 1.8 |
| w = 2 | 2.820 | 0.952 | 91.3% | 103.7 | −1.013 ± 0.115 | −0.990 ± 0.149 | +8.0 ± 1.9 |
| w = 4 | 2.320 | 0.588 | 92.7% | 104.2 | −1.513 ± 0.119 | −1.353 ± 0.150 | +9.3 ± 1.9 |
| **w = 8** | **2.083** | **0.385** | **94.3%** | 106.4 | **−1.750 ± 0.118** | **−1.557 ± 0.148** | **+11.0 ± 1.9** |
| planner beside | 1.013 | −0.247 | 98.3% | 106.9 | −2.820 ± 0.115 | −2.188 ± 0.138 | +15.0 ± 2.1 |

(Hearts lost is net of hearts picked up, so it can be negative. The bombs used are unchanged: 0.35-0.36 an episode
on goriya's rooms, 1.62-1.70 in 7:0x1b.)

**What a good head is worth, with nothing trained:** at the planner's refusal rate (w = 4) it takes **36% fewer hits
a visit** in goriya's rooms (0.808 → 0.519) and **39% fewer** in 7:0x1b (3.83 → 2.32), with hearts lost down 0.24
and 1.35 an episode, success level or better, and no stall (+0.7 and +5.5 decisions an episode, the fight lasting
longer while Link refuses the hits). The planner, which tries each action in the emulator, takes 69% / 74% fewer:
**the learned head recovers about half the planner's edge** (52% of the hits it saves on goriya's rooms, 54% on
7:0x1b at w = 4, 62% at w = 8). The curve flattens past w = 4 in goriya's rooms (w = 8 level with it) and still
gains in 7:0x1b. Round 36's arms, the same head trained inside a reinforcement run or fed to the policy tower,
took no fewer hits than the teacher (§1): the head is worth this much only when it is connected to the action
choice directly, as a shift of the action's probability by its own prediction.

### 3. The head on the states it will meet

Labels on the actor's own states, no update (`python -m triforce.lab.record --families goriya@d3first --actor
teacher|student [--student models/r38shift/pre-w4] --try-others 8:1.0 --rollouts 35`: the look-ahead tries the
actor's next 8 ranked candidates at every decision, in collection only; 143,360 decisions each, every 7th shard held
out). Both recordings ran before main's fix of harvested start states by namespace, so they also drew d2first's
goriya harvest (the old order's Dungeon 7 at its hearts and bombs): a broader distribution, not the route's alone.
Precision = recall on the actions taken (their real outcome) / on every judged candidate:

| head | r36's held-out (planner / model draws) | teacher's states (rec-teacher) | the w = 4 combination's own states (rec-w4) |
|---|---|---|---|
| supervised (r36, 2 million decisions) | 0.886 / 0.857 | 0.736 / 0.679 (all) | 0.507 / 0.598 (all) |
| + 4 epochs on rec-teacher (`hit-r0t`) | 0.888 / 0.861 | 0.813 / 0.792 (held out) | 0.619 / 0.722 (all, unseen) |
| + rec-teacher + rec-w4, round 1 (`hit-r1`, 4 epochs) | 0.890 / 0.862 | 0.826 / 0.806 | 0.593 / 0.738 (held out) |

**It is the states, not the training.** The supervised head loses 0.12 on the teacher's states of the corrected
route (0.857 → 0.736) with nothing trained in between; four epochs with those states in the mix win back most of it
(0.813) and lose nothing on the old data. On the combination's own states the drop looks larger (0.507), but the
number on the actions taken is biased there: the combination plays the actions its head passed, so the hits left
among them are the head's blind spots by construction (its taken-lost rate is 1.59% against the teacher's 2.36%).
The fair number is the judged one (the untaken candidates the policy ranked next): 0.598 → 0.722 with the teacher's
states added, 0.738 after round 1 (its own states in the training mix), with no loss on the old data or the teacher's states. Round 36's in-run head (0.46-0.56) was below even this: an in-run
head sees 0.25 trials a decision against 8 here and starts from nothing; the pretrained arm P started at 0.65 and
fell back with the trunk shared under policy losses.

Not finished (the round's budget): round 1's head (`models/r38hit/hit-r1`) combined and benched, round 2 (record the r1 combination's states on main's fixed start states, retrain on
everything), and the `hit_or_blocked` label (main's Bubble touches; `r38_hit.py` reads it as `DIR+blocked` from a
recording made on the merged code).

### 4. Reinforcement with the head protected: not run

Two arms were queued and cancelled before they started, when main's start-state fix showed that a lab run with a
route's states drew d2first's harvest too: (frozen) the combination with its hit body frozen
(`models/r38shift/pre-w8-frozen`), (own loss) the hit body trained only by its own loss on fresh look-ahead labels
(`--aux hit=hit:0.5 --try-others 4:0.25`); both with the fixed combination as the fading teacher, so the teacher
term does not pull the policy to undo the shift (KL to the unshifted teacher would). Commands in the status notes;
they want the merged code (branch `r38-hit-m`).

### 5. 7:0x1b as a fight

The room's row of §2: with the six Blue Goriyas and 0-7 bombs the teacher wins 83.3% at 3.83 hits a visit and 1.94
hearts; the fixed combination (w = 8) 94.3% at 2.08 and 0.39; the planner 98.3% at 1.01. The `d7` chains (1000 runs,
`--checkpoint goriya@d3first=models/r38shift/pre-w8`, and today's team on the same seeds and code; the planner on
300) were queued. Today's team with 7:0x1b a fight (this branch's code, seeds 0-999, from `d3first_7_79s_0` to step 180, out of Dungeon 7 with its triforce): **12.5% of runs finish Dungeon 7**; 517 of the 875 losses are in 7:0x1b (3.51 hits a visit over 737 visits), then 7:0x1a 136, 7:0x19 67, Digdogger's 7:0x1c 31. The planner beside (300 runs, every network with the look-ahead): 96.7%, 1.12 hits a visit in 7:0x1b, 2 runs lost there. With goriya played by the fixed combination (w = 8, the rest of the team today's, the same seeds and code): **67.8% finish Dungeon 7, +55.3 ± 1.8 points paired on the 1000 seeds**; in 7:0x1b 2.45 hits a visit against 3.51 and 0.96 hearts lost a visit against 2.47 (1,087 visits against 737: more runs get there), and 105 runs lost there against 517. The losses left: 7:0x1b 105, 7:0x18 66, Digdogger's 7:0x1c 57, 7:0x1a 21. It is still below round 34's crossing (`d7` 86.9% at 2 bombs, the backlog's baseline) and the planner (96.7%). **The decision for the lead:** with today's goriya network 7:0x1b must stay a crossing (a fight keeps 12.5%); with the head connected as a shift the fight keeps 68%: worth keeping as the fight the owner wants only once the head is stronger (round 1's head, §3, is not yet benched) and the remaining 0x1b and 0x18 losses are worked on; until then the crossing costs 19 fewer runs in 100.

### 6. The threat lane

`scripts/experiments/r38_lane.py` (inputs computed in Python from the recorded RAM; module docstring for the exact
definition). Measured on goriya's collection: the boomerang flies at q-speed `$A0` = 2.5 px a frame (RAM, every
flying one); of 116,082 lost candidates only 888 were hit by a shot fired after the decision (722 boomerangs), most of
them 4-9 frames into the action; 59% of those stood in a ready Goriya's lane at the decision, **12 / 21 / 32 pixels
ahead (10th / 50th / 90th percentile): on goriya the lane's reach is within 2 tiles**, inside the owner's guess of
about 3. Ranked on experiment 2's bench (the hitnet candidate, 10 epochs, two seeds; 136 held-out post-decision
hits):

| inputs | P = R planner / model draws | post-decision shot hits: AUC | their recall at the P = R threshold |
|---|---|---|---|
| no lane (`in_line` zeroed) | 0.868 / 0.820, 0.869 / 0.816 | 0.950, 0.946 | 0.272, 0.243 |
| `in_line` (schema 6, today) | 0.868 / 0.818, 0.867 / 0.821 | 0.949, 0.948 | 0.257, 0.265 |
| **the threat lane** | 0.865 / 0.824, 0.864 / 0.811 | **0.959, 0.960** | **0.338, 0.346** |
| the lane, `in_line` zeroed | 0.863 / 0.818, 0.866 / 0.821 | 0.958, 0.960 | 0.331, 0.331 |

The lane catches a third more of the hits it is for (0.26 → 0.34 recall) and leaves the rest level; `in_line` adds
nothing. On goriya the slice is under 1% of the hits, so the lane cannot move goriya's totals. The shooter family it
is meant for (Octoroks: a 48-frame stand still before the rock, 3 px a frame) had no collection: one was queued with
the probe logger of branch `r36-hit` (ow-octorok's `d3first` rooms 0x37, 0x3c, 0x4c, 0x64; 100 planner and 100
model-alone runs from each of 30 states; queue cost: 83 minutes from queueing to the file, most of it waiting). It does not rank the lane: the model and the planner on those rooms were hit by a rock fired after the decision only 24 times in the whole collection (2 held out), and the three inputs read the rest of its hits alike (P = R 0.65-0.73 / 0.57-0.61, within the seeds' spread on a small set). The shot speed checks out (3 px a frame from RAM). An Octorok ranking needs a collection aimed at the slice: states where Link walks into a standing Octorok's lane.
**The engine fields, if the Octorok ranking agrees**: per entity, 11 inputs: can fire (Goriya: `ObjState` bit 7
clear; Octorok, Moblin, Lynel: shoot timer `$451` zero or above `$10`; not stunned `$3D`, metastate `$405` 0), the
delay to the shot (Goriya `ObjTimer` `$28`; the others' timer − `$10`, or 32 when it has not started) / 64, in the
lane now (ahead along `ObjDir`, middle-to-middle across under 9 px, within the boomerang's `$51` px), the distance
along / 64, across / 16, and danger = in lane × max(0, 1 − frames to arrive / 32) now and after Link stands or walks
N, S, W, E for 8 frames; shot speeds 2.5 (boomerang), 3 (rock, sword shot), 2 (arrow) px a frame.

### 7. Darknut and Gleeok

The same construction with each family's own supervised head (`r38_hit.py retrain --init fresh:POLICY --data
r36:darknut|r36:boss-gleeok`: 0.845 / 0.775 and 0.774 / 0.676, as round 36), weights at the planner's refusal rate
(darknut 1.31%, Gleeok 1.56%: w = 4) and twice, on their yardstick entries × 10 seeds:

| | hits a visit | hearts lost | success | hits − teacher (an episode) |
|---|---:|---:|---:|---:|
| darknut@d3first teacher | 0.779 | 0.602 | 94.2% | |
| w = 4 | 0.671 | 0.490 | 95.4% | −0.108 ± 0.030 |
| w = 8 | 0.671 | 0.471 | 95.8% | −0.108 ± 0.039 |
| planner | 0.354 | 0.144 | 97.9% | −0.425 ± 0.063 |
| boss-gleeok@d3first teacher | 4.358 | 3.340 | 45.8% | |
| w = 4 | 4.025 | 2.940 | 52.1% | −0.333 ± 0.173 |
| w = 8 | 4.100 | 2.996 | 57.1% | −0.258 ± 0.188 |
| planner | 2.154 | 1.244 | 59.2% | −2.204 ± 0.226 |

Darknut −14% hits a visit (a quarter of the planner's edge), Gleeok −8% with +6 to +11 points of wins: smaller than
goriya's −36%, as the heads are weaker (round 36: darknut's misses are late in long actions, Gleeok's fireballs) and
were trained on two rooms (darknut) or one (Gleeok) of the entries benched.

**What one all-families network of this kind needs**: a hit body trained on look-ahead labels from every family's
rooms (the labels are cheap: `lab.record --try-others 8:1.0` adds no network pass), refreshed on the states the
combined network reaches (§3), kept out of the policy's gradient; the shift applied at play with one weight (w = 4-8
fitted goriya, darknut and Gleeok alike); branch `r38-one`'s `h768` shape (a hitnet with a dormant joint head) is
the body to put it on.

## The hit head across families (branch `r38-head`)

**The question:** does §2's recipe (a supervised hit head on look-ahead labels, a fixed shift of the policy's joint
action by `-w * softplus(hit logit)`, nothing else trained) carry to the families that cost Dungeons 4, 3, 5 and 8,
and does one head across families read as well as one a family. Scripts: `scripts/experiments/r38_head.py` (cost,
manifests, combine, bench, sliced precision, paired chains), `r38_hit.py retrain --by-row` (the trunk run once a
decision for its nine candidates: a family's head in 3-10 minutes on the GPU). `lab.record` gained `--state-route`
(a route's states alone, as lab training's) and `--teacher KEY=CKPT` (another actor a family).

**Code.** Recordings and heads: 2.0.0.5 (the teacher's states; 2.0.0.6 changes only the bomb masks of crossing
rooms). Every bench and chain number: 2.0.0.6 (main cbc1dfa9 merged, routes and crates identical to main), today's
team from main's record `r37fix-1k-*` (same seeds), scored by the segment's arrival.

### What the families cost

Per 1000 runs of each segment (today's 51 networks, the 1000-run record `r36base` on the corrected route; 2.0.0.5),
runs lost in the family's rooms (the run's last room) / hearts lost there:

| family | Dungeon 4 | Dungeon 3 | Dungeon 5 | Dungeon 8 | runs lost | hits a visit |
|---|---:|---:|---:|---:|---:|---:|
| darknut | | 323 / 1444 | 521 / 6644 | 483 / 11840 | 1327 | 1.24 |
| boss-gleeok | 515 / 874 | | | 97 / 827 | 612 | 2.26 |
| boss-manhandla | | 258 / 366 | | 0 / 870 | 258 | 0.96 |
| like-like | 123 / 985 | | | | 123 | 0.63 |
| gibdo | | | 122 / 1183 | | 122 | 0.12 |
| vire | 86 / 1098 | | | | 86 | 0.24 |
| zol | 4 / 168 | 53 / 740 | 0 / 7 | | 57 | 0.21 |
| boss-digdogger | | | 37 / 86 | | 37 | 1.18 |
| stalfos-keese-gel | 27 / 347 | 9 / 164 | | 0 / 0 | 36 | 0.21 |
| pols-voice | | | 8 / 45 | 17 / 0 | 25 | 0.02 |
| segment rate | 0.185 | 0.232 | 0.312 | 0.400 | | |

(equipment-room 97 and trap-room 91 are played by the engine, not a network: out of reach of a head.) Wizzrobe and
wallmaster cost Dungeons 6, 9 and 7, not these four.

### Labels and how much of them

`python -m triforce.lab.record --families KEY... --state-route d3first --actor teacher --try-others 8:1.0
--rollouts 100 --parallel 8` in three jobs (Dungeon 4's four families, two envs each; Dungeon 3 and 5's four; Pols
Voice and Digdogger, 50 rollouts, four envs each): 102,400 decisions a family, eight look-ahead candidates each, every
7th shard held out. A 100-rollout job runs 65 minutes at 100 decisions a second (Hit38's rate), plus the line (5 to
25 minutes today; 2 h for the Dungeon 3 job behind others).

**The data curve** (goriya, a fresh head on the two goriya recordings' training shards, P = R on rec-teacher's
held-out judged candidates / rec-w4's): 15k decisions 0.662 / 0.562, 31k 0.708 / 0.600, 61k 0.743 / 0.641, 122k
0.769 / 0.687, 245k 0.794 / 0.719 (taken 0.669 → 0.813). Each doubling buys +0.046, +0.035, +0.026, +0.025: still
rising, flattening; the 2 million old-order decisions plus both recordings (hit-r1) read 0.806 / 0.738. 100k a
family is on the curve's knee; a second 100k (the combination's own states) is worth about +0.025.

### The heads

P = R on the held-out shards of each family's teacher recording, every judged candidate / the actions taken:

| family | own head (100k decisions) | round 36's head on these states |
|---|---|---|
| darknut (+ r36 darknut) | 0.840 / 0.829 | 0.522 / 0.485 |
| boss-gleeok (+ r36 Gleeok) | 0.721 / 0.749 | 0.554 / 0.599 |
| boss-manhandla | 0.774 / 0.757 | |
| like-like | 0.818 / 0.684 | |
| vire | 0.731 / 0.691 | |
| zol | 0.806 / 0.717 | |
| gibdo | 0.829 / 0.765 | |
| stalfos-keese-gel | 0.803 / 0.739 | |
| boss-digdogger | 0.822 / 0.820 | |
| pols-voice | 0.880 / 0.706 (17 taken hits) | |

**Why Darknut's and Gleeok's heads read 0.52-0.57:** the states. Round 36's heads were trained on the old order's
rooms (two Darknut rooms, one Gleeok fight at other hearts and swords); on the corrected route's own rooms they read
0.52 / 0.55, and the same construction trained with the route's states reads 0.84 / 0.72. What is left, from round
36's collections where the hitter is known (`r38_head.py slices36`; recall at P = R on the model's own draws):
Darknut's touches are read 0.98 in an action's first 4 frames, 0.87 at 4-8, **0.53 at 8-16 and 0.18 after**
(goriya's head on 2M decisions: 0.95 / 0.87 / 0.71 / 0.69): the misses are late hits in long actions, the Darknut
turning into Link mid-walk; a Darknut's facing and sword reach are in the inputs, its next turn is not (random). More
data reads late hits better (goriya), and a shorter action (the duration) is the cheap way round them. Gleeok:
touches (the necks and heads) 0.69, fireballs in the air 0.61, fired after the decision 0.43 (7 hits); frames 0.81 /
0.64 / 0.47 / 0.06: the necks' swing is a timed motion the observation shows only as positions (memory or the heads'
phase would carry it), and the fireballs are aimed shots that the threat lane's arithmetic would cover (Gleeok's
heads as shooters). The inputs' own "shot in the air" flag cannot slice Gleeok or Digdogger (it is up on 94-99% of
their hits).

### The bench and the chains

Bench (2.0.0.6, 240 episodes a family: 24 of its `d3first` harvested entries × 10 seeds, paired), hits a visit,
teacher → w2 / w4 / w8 / planner beside:

| family | teacher | w2 | w4 | w8 | planner | best w − teacher (hits, hearts an episode; success) |
|---|---:|---:|---:|---:|---:|---|
| darknut | 0.746 | 0.658 | 0.592 | **0.525** | 0.396 | −0.221 ± 0.063, −0.260 ± 0.064; +3.3 ± 1.4 |
| boss-manhandla | 1.438 | 1.296 | **1.196** | 1.288 | 1.033 | −0.242 ± 0.064, −0.254 ± 0.070; +2.9 ± 2.0 |
| zol | 0.092 | 0.071 | 0.046 | **0.042** | 0.033 | −0.050 ± 0.020, −0.037 ± 0.016 |
| gibdo | 0.096 | 0.083 | 0.071 | **0.067** | 0.050 | −0.029 ± 0.015, −0.056 ± 0.029 |
| boss-gleeok | 4.358 | 3.450 | 3.188 | **2.971** | 2.154 | −1.387 ± 0.247, −1.406 ± 0.247; won 45.8% → 59.2% |
| like-like | 0.329 | 0.246 | 0.221 | **0.171** | 0.092 | −0.158 ± 0.033, −0.144 ± 0.030 |
| vire | 0.233 | 0.188 | 0.133 | **0.108** | 0.050 | −0.125 ± 0.031, −0.106 ± 0.027 |
| stalfos-keese-gel | 0.171 | 0.129 | 0.104 | **0.092** | 0.046 | −0.079 ± 0.025, −0.040 ± 0.012 |
| boss-digdogger | 2.312 | 2.242 | 2.171 | 2.104 | 1.575 | −0.208 ± 0.131 (not shown); success −1.3 |
| pols-voice | 0.100 | 0.100 | 0.100 | 0.096 | 0.037 | none (0.16% of its decisions are hit) |

The best weight takes 56-86% of the planner's saving on the eight families that gain, more than goriya's 52%; no
stall (darknut 33.5 → 39.5 decisions, Gleeok 179 → 212, the others within +5). Digdogger and Pols Voice do not gain.

**One head for every family:** the head trained on all the recordings at once (`r38_hit.py retrain --by-row --batch
256 --epochs 8 --init fresh:models/goriya/r32d1-goriya --data <the three d3first recordings, the two goriya ones,
r36:darknut, r36:boss-gleeok> --save r38head-hit/all`) reads every family as well as or better than its own head
(judged +0.017 to +0.038; taken better on all but Gleeok, 0.734 against 0.749): darknut 0.858 / 0.851, Gleeok 0.738 /
0.734, Manhandla 0.795 / 0.781, like-like 0.856 / 0.747, vire 0.751 / 0.712, zol 0.825 / 0.750, gibdo 0.853 / 0.809,
stalfos-keese-gel 0.839 / 0.788, Digdogger 0.841 / 0.852.

**The shared head at play** (bench, 240 episodes a family; `r38_head.py combine --head all=models/r38head-hit/all
--weights 4 8`; the same one hit network beside each family's own policy; `target/r38h/bench-all-table.md`), hits a
visit at w8 against the family's own head at w8, paired an episode:

| family | own head w8 | shared head w8 | shared − own (hits an episode) | success shared − own |
|---|---:|---:|---|---|
| darknut | 0.525 | **0.442** | −0.083 ± 0.051 | −1.3 ± 0.9 |
| boss-gleeok | 2.971 | **2.587** | −0.383 ± 0.194 | **−6.7 ± 3.0** (52.5% won against 59.2%) |
| boss-manhandla | 1.288 (w4 1.196) | **1.154** | −0.133 ± 0.072 | −1.7 ± 1.7 |
| like-like | **0.171** | 0.179 | +0.008 ± 0.021 | −1.3 ± 0.7 |
| vire | 0.108 | **0.083** | −0.025 ± 0.020 | 0 |
| zol | 0.042 | **0.029** | −0.013 ± 0.009 | +0.4 |
| gibdo | 0.067 | **0.046** | −0.021 ± 0.014 | 0 |
| stalfos-keese-gel | 0.092 | **0.071** | −0.021 ± 0.017 | 0 |
| boss-digdogger | 2.104 | 2.058 | −0.046 ± 0.098 (hearts −0.20 ± 0.10) | +2.5 ± 1.3 |
| pols-voice | 0.096 | **0.050** | −0.046 ± 0.015 | 0 |

The shared head takes fewer hits than the family's own on nine of ten (w4 is weaker than w8 everywhere), Pols Voice
and Digdogger included, where the own heads did nothing. One cost: Gleeok wins fewer fights with it (−6.7 ± 3.0
points) for 0.38 fewer hits an episode, so the bench does not settle it for Gleeok, and a chain has to. Bubble
touches in the like-like rooms (the only bench rooms with Bubbles): teacher 1.14 a visit, own w8 1.39, shared w8
1.28, planner 1.32. The head does not count touches.

**What the one-network line needs from it.** The shared head is one hit network (`models/r38head-hit/all`, a
1.5M-parameter `hitnet` trunk and joint hit head), and nothing about it is family-specific, so it goes beside the
one network as it goes beside each family's policy: `r38_hit.combine(<the student's ckpt>,
'models/r38head-hit/all', [8], ...)`, which builds a `hitshift` of the hitnet-body student `h768-f` (checked: 6.2M
parameters, heads on recorded batches). Offline, on the teacher recordings' own decisions, the head reads the
student's draws as it reads the teachers': the predicted hit chance of the draw at w0 / w8 is Gleeok 1.53 / 0.41%
(teacher 1.29 / 0.39), darknut 2.78 / 0.60% (2.70 / 0.72), Manhandla 3.72 / 1.45% (3.67 / 1.54), like-like, vire,
skg, zol and gibdo 0.66-1.26 / 0.19-0.32%, with 1.0-4.8% of draws refused (`target/r38h/calib-*-all.json`). Four
things are still open: (1) one weight for all keys. w8 is best on nine families, and per-key weights would need the
key in the network. (2) The keys the head never saw: overworld, traps, Wizzrobe, Lanmola, Patra, Ganon. Bench the
combination on all 51 keys (`lab.bench run --network`, and the rooms tier) before any chain, so a refusal does not
stall a walk. (3) Labels on the student's own states (`lab.record --actor <the combination> --try-others 8:1.0`):
the student's states are not the teachers', and its own labels are what a retrained head reads best on (goriya's
second round: +0.034 on its own held-out labels, level on the bench). (4) Train with the head in place and kept on
its own labels (round 38's PPO aux arm, `--aux hit=hit:0.5`: +7.8 points on Dungeon 7 over the fixed shift), not
frozen, because frozen, the policy does not learn to use it.

**Chains (main's serve fix and CUDA graphs, 2.0.0.6; 1000 seeds paired with main's record `r37fix-1k`, scored by
the segment's arrival; `r38_head.py chains ... --out target/r38h/score-<d>-head.json`).** Run twice in separate
processes, the 100-run Dungeon 3 chain with four combinations is identical: all 100 runs, every room field, and the
actions of all 1,575 rooms. Only the diagnostic surprisal and entropy floats differ, by at most 2.4e-5. The team
on the merged module plays the record's 100 runs identically. The void Dungeon 3 chain (bug era) turns out to be
identical to the fixed one in all 1000 runs: no temporary died there. Dungeon 4's void chain (w4) is replaced by w8.

| dungeon | heads (weight) | team → heads | per 1000 runs | rooms: runs lost team → heads | hits / hearts lost a visit, team → heads |
|---|---|---:|---:|---|---|
| 4 | Gleeok, like-like, vire, skg (w8) | 21.1% → **66.6%** | **+455 ± 19** | Gleeok 4:0x13 504 → 280, like-like 116 → 10, vire 87 → 17, skg 27 → 1 | Gleeok 1.92 → 1.64 / 1.26 → 0.80; like-like 0.62 → 0.25 / 0.51 → 0.21; vire 0.22 → 0.11; skg 0.27 → 0.13 |
| 3 | darknut w8; Manhandla, zol, skg w4 | 21.7% → **40.1%** | **+184 ± 20** | darknut 385 → 236 (0x5b 120 → 60, 0x59 62 → 11, 0x5c 47 → 14, 0x69 156 → 151), Manhandla 223 → 236 (of 440 → 637 entering), zol 53 → 12 | darknut 0.32 → 0.24 / 0.30 → 0.23; Manhandla 1.16 → 1.03 / 0.75 → 0.64; zol 0.23 → 0.14 |
| 5 | darknut w8; gibdo, zol w4 | 31.2% → **66.1%** | **+349 ± 20** | darknut 521 → 193 (5:0x05 416 → 105), gibdo 122 → 94 | darknut 0.72 → 0.42 / 1.29 → 0.74; gibdo 0.13 → 0.10 / 0.23 → 0.18 |
| 8 | darknut, Gleeok w8; Manhandla, skg w4 (the bare keys) | 39.1% → **80.9%** | **+418 ± 19** | darknut 474 → 81 (8:0x3e 266 → 43, 0x3f 149 → 37, 0x4e 56 → 1), Gleeok 96 → 40, Pols Voice 8:0x4c 37 → 68 (no hits; 529 → 920 entering) | darknut 3.37 → 2.07 / 3.23 → 1.86; Gleeok 2.65 → 2.23 / 1.58 → 1.08 |

Dungeon 8's rooms play the **bare** keys (`darknut`: r30-dark69-s94, `boss-gleeok`: r78t4-gleeok, and so on; the
record's rooms have no `@d3first` policy), so its combinations are those policies with the family's head
(`models/r38head/<f>-bare-w{4,8}`). Served play ran at 983-1,758 decisions a second, start included (d4 754,712
decisions in 12 min 48 s; d8 903,777 in 8 min 34 s), against about 800 for main's four-network d3 chain.

**The set to promote, played as one** (one run a key, every key in every dungeon; batch `r38h-promo`, same seeds):
`@d3first` Gleeok, like-like, vire, stalfos-keese-gel and darknut at w8, Manhandla, zol and gibdo at w4; bare
darknut and Gleeok at w8, bare Manhandla at w4. Bare stalfos-keese-gel stays as it is, because it also plays Dungeons
6 and 9 (2,364 and 5,897 room visits in the record, unmeasured with a head) and its Dungeon 8 rooms lost no run either
way. Within each segment, the `@d3first` keys play only Dungeons 3, 4 and 5, and the bare darknut, Gleeok and
Manhandla play only Dungeon 8. Results: Dungeon 4 **68.4%** (+47.3 ± 1.9 against the team; +1.8 ± 0.8 against the
arm above, where zol at w4 is the only change in its rooms), Dungeon 3 **39.2%** (+17.5 ± 2.0; −0.9 ± 0.7 against
the arm: skg at w8), Dungeon 8 80.9% (identical to the arm in all 1000 runs), Dungeon 5 as above (the same keys).
Product with this set and goriya's: 1.02e-3.

**What it predicts for one run** (`r38_head.py product`: the product of the segment rates, each by its arrival;
today's team is `r37fix-1k` where main has the segment, round 36's records elsewhere): today's team 6.99e-6 (one in
143,000; round 37's 6.73e-6 by result on 2.0.0.5). With the four dungeons' heads and goriya's (`r38shift-r1-w8`,
Dungeon 7 71.3%, measured on 2.0.0.5 with its team at 12.5%): **1.01e-3, one in 989**, ten times the owner's 1e-4
(−ln p summed 6.90 of the 9.21 allowed). Without Dungeon 7, 1.79e-4. With the PPO aux arm on Dungeon 7 (79.1%),
1.12e-3. This is the team of specialists with heads, not one network. The largest costs left are post-D7 (22.4%,
1.50), post-D6 (1.14), D6 (1.01), D3 (0.91, Manhandla and 3:0x69's darknut) and D9 (0.63).

## Aimless walking

**The question** (the owner, watching the golden replay `corpus/replays/golden-d3first-single-network.tfreplay`):
"the opening room link walks back and forth a while before grabbing the sword. In 0x4c [Dungeon 8] [...] he walks
back and forth after the bombed wall is opened before going north. [...] you could create an 'aimless walk' score,
where there are no enemies and all link has to do is go to the cave, door, heart, triforce, etc (and isn't being
actively pushed to an item [...]). Some amount of missed steps is totally fine, but the opening level and d8 0x4c is
something I'd like toned down, if not eliminated eventually."

**The score** (`crates/zelda/src/aimless.rs`, docs/architecture.md "Aimless walking"): per room visit, the frames of
its calm stretches (nothing in the room can hurt Link, the objective a place that does not go away, no dropped item
pulling him) beyond the walk the ground they gained needed at walking speed, less 16 frames a stretch. Carried in
chain room records, evaluation visits, rollouts, the bench and `triforce room-stats`; `reward.aimless_penalty`
(`penalty-aimless`, default off) charges it.

### The golden replay's two visits

The golden replay carries no decision stream, so its two segments were played again with one
(`Route.load(..., power_on={'name': 'LINK', 'delay': 4587})` with seed 7 to route step 3; the replay's own checkpoint
at step 213 with seed 14042 to step 227; the single network in every slot, `temperature:0.5+relax:16`): **both
reproduce the golden game exactly** (1,493 and 11,160 frames, the two visits' 948 and 2,317 frames), so the
probabilities below are the ones the golden run drew from. The objective's wavefront step at each decision is the
`target` observation of a single-family env replaying the same actions from the same state (0:0x77 from the game
start; 8:0x4c from the replay's keyframe after the Pols Voices died, as `d3first_8_4cc_*` so the env plays d3first's
step), which reproduces the chain's score of the visit to the frame.

| visit | frames | calm frames | needed | aimless | calm decisions | away | reversals |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0:0x77, the opening (step 0) | 948 | 563 | 90 | **457** | 45 | 13 | 26 |
| 8:0x4c after the Pols Voices (step 226) | 2,317 | 1,495 | 180 | **1,282** | 119 | 35 | 11 |

The score sees what the owner saw: the opening's straight walk to the cave needs 90 frames and took 563; 0x4c's
walk out needs 180 and took 1,495. Every other room of the two segments scores 0.

**0:0x77: a confident loop at the cave's mouth.** Link walks N then W to (64, 85), under the cave (the wavefront's
step there is N, 1 tile). The policy's moves at the mouth, per decision (joint probability of the direction, any
duration): at x = 64-66 **E 0.92-0.96** (N 0.03-0.05); at x = 70-71 N 0.57-0.60 against W 0.39-0.42, and N there
only aligns Link to x = 72 (the game slides him to the nearest column; the wavefront still says N); at x = 72 W 0.99
(the wavefront's W); after a W 4 back to x = 66 **E 32 at 0.94** to x = 110, then W 32 at 0.86 back to x = 65. The
same cycle ran about five times. It ended at decision 44 with a **MOVE N 4 drawn at p = 0.05** at x = 64 (by
then `relax:16` had the stall count, 39 decisions at the visit's most, and played plain `sample`; at the
temperature's 0.5 that move was 0.003). The target never flips: it is the cave from the first decision to the
last, pointing N where Link can go N.

**8:0x4c: the opening is the target at once; a confident loop on the top row.** After the BOMB N from (119, 93),
the WAITs while the smoke clears are not calm (the wall has not opened; the wavefront can't reach Link). The first
calm decision, at x = 108, has the opening 2 tiles E as its target; the policy goes **W 8 at 0.98**, then W along
the top row (0.95-0.99) to the W wall (x = 32), E back (0.71-0.98) to x = 76-91, **N at 0.90-0.95** where the top
row's wall blocks N (the wavefront's step alternates N and E along that row: N where the tile grid has room above
Link, which the wall's pixels don't give him), then W again (0.88-0.95): fourteen times to the W wall and back in
about 1,390 frames. It ended with a **MOVE E 32 drawn at p = 0.42** at x = 80 (W 0.58), then N 8 out.

**The cause, in both rooms:** not a draw between near-equal moves and not a flat distribution (the moves of the
loops are 0.85-0.99; the entropy of the kind and direction is 0.04-0.45 nats on most of them), and not a target that
flips: **confident wrong moves that form a loop**, each one the policy's most likely move, ended only when sampling
drew the minority move (0.05, 0.42). Playing the most likely move on calm decisions would make the loops endless
until the stall rule's switch, so no `calm:argmax` rule was built. Two details of the target make the loops easier
to fall into: at the cave's mouth it says N where N only aligns Link (x = 65-71), and on 0x4c's top row it says N
where the wall stops him.

**What would break each loop.** Not the target: in both rooms it is the right goal from the first calm decision
(the cave; the opening, 2 tiles E, the moment the wall is open), and the policy walks against it at 0.94-0.98.
Two corrections to the target would remove the ground the loops stand on (its step at the cave's mouth should be W
until Link is in the cave's column; on 0x4c's top row it should be E, not N, until the opening), but the loops'
longest legs (E 32 from the mouth to x = 110; W from the opening to the W wall) go against a target that is right.
What the policy needs to see is there: the target moves to the opening the moment the wall is open. What is
missing is a cost: the movement shaping is potential-based, so a loop nets 0, and nothing else charges it (below).
Until a policy is trained against it, only sampling breaks a loop.

### Today's numbers (the arms, 0x4c's baseline, Dungeon 3)

One batch on the branch (`r38walk`, 300 runs an entry, seeds 0-299, clips on): the team (today's 51 networks,
`--act model`, every one `temperature:0.5+relax:16`) and the one network (`r36merge/w768e2-z` in every slot). The
opening is played from the game start to route step 3 (0:0x77, its cave, 0:0x77 again, into 0:0x76); 0x4c from
`8_4cc_0` (up the cellar's stairs: the Pols Voices, the bomb wall) to step 227. Aimless frames a visit of 0:0x77
(step 0) and 8:0x4c (step 226): mean, median, 90th percentile; success: the run reached the stop; stuck: the
family's 2,000-step cap.

| arm | team: 0:0x77 aimless (mean / med / p90) | reversals | team: 8:0x4c aimless | success (stuck) | one: 0:0x77 | one: 8:0x4c | success |
|---|---|---:|---|---|---|---|---|
| `temperature:0.5+relax:16` (today's) | **668 / 493 / 1,454** (the route from the start) | 40.2 | **5,239 / 3,723 / 14,483** | 93.0% (21) | | | |
| `temperature:0.25+relax:16` | 710 / 515 / 1,516 | 42.5 | 5,475 / 3,857 / 14,098 | 91.7% | 664 / 503 / 1,345 | 5,918 / 4,443 / 15,079 | 92.3% |
| `temperature:0.1+relax:16` | 707 / 521 / 1,485 | 42.3 | 5,710 / 4,186 / 15,291 | 89.7% | 683 / 503 / 1,345 | 6,365 / 5,093 / 15,632 | 89.7% |
| `escape:16` | 708 / 512 / 1,474 | 42.1 | 6,149 / 4,163 / 15,705 | 88.3% | 659 / 503 / 1,305 | 5,987 / 4,624 / 14,820 | 92.7% |
| `temperature:0.5+relax:4` | **574 / 396 / 1,333** | 34.1 | 5,647 / 3,788 / 15,359 | 89.3% (31) | **537 / 379 / 1,288** | 5,101 / 3,405 / 12,837 | 93.3% |

**Every visit wastes.** All 300 of 300 opening visits score more than 100 aimless frames in every `relax:16` and
`escape` arm, 278 and 268 of 300 under `relax:4` (the straight walk needs 90): the golden run's 457 is below these
arms' medians (503-521), not an unlucky opening. 0x4c is the bigger sink by
far: a median of 3,723 aimless frames (a minute) a visit, 5,416 calm frames, 7% of the runs stuck at the step cap
walking it; the golden run's 1,282 was a lucky one. The opening succeeds in every arm (the run reaches 0:0x76);
no arm lost health in 0x4c.

**Act mode alone.** A lower temperature and `escape:16` make the opening a little worse (707-710 against today's
668: the loop's moves are already near 1, so sharpening them keeps the loop and the stall rule's switch at 16 decides
when it breaks) and 0x4c worse (5,475-6,149 against today's 5,239, success down 1.3-4.7 points). **`relax:4`**
(plain sampling after 4 decisions without progress instead of 16) **trims the opening by about 14%** (574 against
today's 668 for the team, median 396 against 493; 537 against 664-683 for the one network's other arms; reversals 34
against 40; the visit about 100 frames shorter), and **does nothing for 0x4c** (team 5,647 against 5,239, success
89.3% against 93.0%, 31 stuck runs against 21; the one network 5,101 against its other arms' 5,918-6,365). Its
cost elsewhere is not measured here: it switches every room to plain sampling after 4 stalled decisions, fights
included, where today's `relax:16` keeps the temperature's sharper play for 16; a fight that stalls 4 decisions
(waiting out a Darknut, a boss's pattern) would draw from the raw policy four times as soon. Making it the default
needs the route's record under it; a calm-only form (sooner plain sampling on calm decisions only) would leave
fights alone, but 14% is the ceiling it could reach in the opening and 0x4c shows no gain to keep. So: nothing in
act mode tones 0x4c down, and the opening only by a seventh; both need training.

**The route from the start and Dungeon 8, the team** (2.0.0.5 code, like every number in this section; 300 runs
each). From the game start (`--act model`, no stop; no run reached the end of the game): 841 aimless frames a run
(median 647, p90 1,686), 5.9% of the runs' frames, 185 calm decisions a run of which 12.5% moved away, 45.6
reversals a run; the opening holds 668 of the 841. The `d8` segment (from `8_7es_0` to 0:0x6d, 40.0% success):
1,473 aimless frames a run, 20% of its frames; **8:0x4c at step 226 holds 440,306 of the segment's 441,817** (154
visits, 2,859 a visit, median 809, p90 9,723). Over the 600 runs of `d3` and `d8`, the rooms that waste most:
8:0x4c (226) 90%, 3:0x59 (17) 8.7%, 3:0x5b (19) 0.7%, then 8:0x2c (228) and 3:0x3d (23) at 8 frames a visit; 28
room-steps hold all of it.

**Dungeon 3, the team** (the identity check's entry, 300 runs from `3_7cs_0` to 0:0x74, 21.3% success): 156
aimless frames a run (median 24, p90 495), 1.9% of its frames; 3:0x59 at route step 17 holds 91% of them (203 a
visit, median 43, p90 555), 3:0x5b (step 19) 7%.

### `penalty-aimless`, and a first training experiment (written, not run)

**The term** (`reward.aimless_penalty`, default 0): each calm decision pays the frames it added to its stretch's
waste (lasted − needed so far, its rise, never below 0) times the setting times `shaping_scale`. A decision that gains
a tile for each 6 frames pays nothing; nothing is charged while a monster, a shot or a moving hazard is in the room.
`tests/test_aimless.py` pins it in the sword's cave: a walk N, N, E, E, W, W, N pays on each decision exactly
setting × scale × its rise of waste (computed from the env's calm frames and the `target` distance; at 0.01 the four
wrong MOVEs pay 0.16, 0.34, 0 and 0.22), at two settings and scales, and nothing with the setting at 0; and 12
decisions in 8:0x4c with its Pols Voices alive pay nothing at a setting of 1.0.

**Why the shaping doesn't already stop the loops:** `reward-pbrs-movement` is potential-based, so a loop that comes
back to where it began nets 0 (each step toward the cave earns what the step back pays); only the discount and the
episode's end weigh against it. `penalty-aimless` charges the loop each time round.

**The experiment.** Fine-tune from the promoted runs, one arm a family, the families of the two rooms first:
`ow-start` (0:0x77 and the sword cave) and `pols-voice` (8:0x4c; its fight is untouched by the term), then the
families that own the most aimless frames in the table below. Settings 0.002 and 0.005 a frame against 0
(the same steps and seeds): at 0.002 a wasted 16-frame MOVE pays 0.032, under a step of the movement shaping
(0.05-0.15) and an eighth of a half heart's `penalty-lost-health` (0.25); the golden opening's 473 wasted frames
would pay 0.95 over 45 decisions, 0x4c's 1,315 would pay 2.6 over 115. It can't outweigh dodging, since it is never
charged while anything can hurt Link; what it could teach wrongly is hurry after a fight (walking into a room's
next spawn), which the yardstick's hits a visit and the family's success check. Measured: aimless frames a visit in
the family's rooms (the bench, `aimless_frames_per_visit`), success and hits a visit (the yardstick), and the
chain on the route from the start (300 runs, paired seeds) against today's; the term stays off for every other
family until an arm shows fewer aimless frames at no loss of success or hits.

### Checks

- Two faults found on the way, fixed on the branch: `triforce.runner.RoomRecord` lacked the seven fields, so
  `RunResult.from_dict` (every `eval-chain` run) raised on the first record (`tests/test_aimless.py` pins it), and
  `triforce room-stats`' evaluation entries lacked the means the chain entries carry (`tests/test_room_stats.py`'s
  "every scope's room entries carry the same fields"; its by-clips summary gains `aimless_frames_per_run`).
- The chain identity check: the branch's team from `3_7cs_0` to 0:0x74 (`--act model`, clips on, seeds 0-99 of the
  batch's 300) against main's `target/r56-chains/d3-100-land.jsonl` (same seeds, start and stop): **100 of 100
  runs identical** (every room record, the new fields and the Bubble's two aside).
- The suites: pytest 943 passed, 12 skipped; `cargo test --release --workspace --features zelda/zcore` ok; the
  zelda-asm tests 5 passed; `asm.py check` ok.
- The registry's estimate against the real pace: the batch was estimated at 1 h 25 min (`chain_steps` at 0.0142 s
  a run a route step); its entries ran 2-65 times longer than their own estimates: the opening (300 runs × 3 steps,
  13 s) 23-55 s, 0x4c (300 × 1 step, 4 s) 60-262 s, Dungeon 3 (300 × 18 steps, 77 s) 470 s. The unit is the
  fault, not the rate: a route step is no fixed amount of play (0x4c's one step is 47 to 2,000 decisions, the opening's
  three about 60), and an entry's start (loading 34 networks, the pool) is a fixed cost the unit doesn't count, so
  short entries and long rooms are underestimated most. A unit of decisions (or of the route steps' frames from the
  latest chains of record) would fit.

## 0x2d dropped

**The owner's change** (Nov 22, from watching): "Overworld: drop the bomb for rupees in 0xd2, keep the armos rupees in
3d (losing too many hearts in that d2 fight, and it's redundant with 0x3d)... we may have to cheat rupees more, but
that 1 30 secret + the 100 secret + drops is the intended route." Asked which room, he confirmed: "0x2d is indeed
what I meant." Landed as **2.0.0.7** (branch `r38-2d`); `d2first` unchanged.

**What changed.** `d3first`'s three steps at 0:0x2d (in from W, the rock wall bombed, the cave's 30 rupees, out S)
are one crossing, W → S; every step after 90 is two lower (292 → 290 steps). 0x3d's Armos cave stays. `game.yaml`
loses `d3first`'s own 0x2d entries (the wall, the secret cave): both routes play the room alike, and neither bombs
its wall. `ow-zora` loses the cave room (`specialists/ow-zora.yaml`, `specialists/d3first/ow-zora.yaml`). The
entry states after 0x2d were made again: steps 89-153 by the explorer from `d3first_0_2dw_*`, the Dungeon 5
entrance (`5_76s`), the Level 5 exit (`0_0bc`) and Dungeon 7's entrance (`7_79s`) cut from the new chains' arrivals;
the audit is clean (855 states); `heart_value.d3first.json` refit on the 290 steps. The purse in the rebuilt states
(RAM `$66D`, chains 0/1/2), 2.0.0.6 → 2.0.0.7: Dungeon 2's entrance 62/70/74 → 32/40/45 (bombs 2/4/0 → 3/5/0), out
of Dungeon 2 72/79/83 → 36/55/65, Dungeon 5's entrance 179/182/196 → 147/147/150, out of Level 5 172/180/183 →
147/148/148, before the arrows (0_44n) 112/120/123 → 76/96/107.

### Measured (today's 51 networks, 1000 runs a segment, paired by seed)

Batch `r38-2d-1k` against 2.0.0.6's `r37fix-1k` (post-d1, d2, the route from the start); 2.0.0.6 has no post-d2 or
post-d5 record, so those two compare with 2.0.0.5's `r36base-1k-post-d2` and `r36guard-1k-post-d5`:

| segment | before | 2.0.0.7 | paired difference |
|---|---:|---:|---:|
| post-d1 (to Level 2) | 99.1% | 98.3% | −0.8 ± 0.5 |
| d2 | 98.4% | 99.3% | +0.9 ± 0.5 |
| post-d2 (to Level 5) | 100% | 100% | 0 |
| post-d5 (Level 5 to Level 7: the dock warp, the meat, the arrows) | 78.9% | 83.1% | +4.2 ± 1.7 |

The post-d5 gain is the new start state's, not the code's: 2.0.0.7's code from 2.0.0.6's own Level 5 exit state
(`post-d5-206`, `--stop-at 7:0x79`) makes 78.9% again. The leg to Level 2 is **620 frames shorter** (4,719 against
5,339 a run to arrival).

**Bombs.** Runs enter Dungeon 2 with **3.07** bombs (2.02 before): the bomb that opened 0x2d's wall is kept. It
hardly reaches the later walls: from post-d1, Dungeon 5 is entered with 6.98 (6.95), runs with none 22 of about 850
(27); from post-d2 8.0 both ways. No bomb is cheated on the leg to Level 2 either way.

**From the start** (runs of 1000 reaching each triforce, 2.0.0.6 → 2.0.0.7): Dungeon 3 195 → 195, Dungeon 4 36 → 36,
Dungeon 1 31 → 31, **Dungeon 2 21 → 28**, Dungeon 5 4 → 6, Dungeon 7 0 → 1 (that run also entered Dungeon 6). No
run ends the game.

**`d2first`:** the 100-run chain is the same as `d2first-100-land` in result, steps and frames on all 100 runs (the
room records differ only by the two counters `bubble_touches` and `sword_blocked_frames`, which the older record
lacks).

### Rupees cheated at the shops (the owner's question)

Rupees are cheated only at the two shops, the meat at 0x34 (60) and the arrows at 0x44 (80): where Link stands at
the counter short, the shortfall is handed over and counted (`cheated_rupees` of the cave rooms).

- **Before (2.0.0.6): none.** No run that reached the shops was short (779 runs from Level 5's exit, 146 continuing
  from after Dungeon 1).
- **Now, for a run that carries its own purse from after Dungeon 2: short at the arrows about 7 times in 10.** 137 of
  the 196 runs that reached the shops were short at 0x44, by about 3 rupees each (438 rupees in all: 2.2 a run over
  all 196). The meat is never short.
- **Now, from the shop leg's own start state: none** (824 runs): that state was cut with 147 rupees, more than the
  140 the two shops cost.

In plain words: without 0x2d's 30, the purse reaches the shops with about what the meat and the arrows cost; most
runs come up a few rupees short at the arrows and get them cheated, about 3 rupees a short run, never more than the
30 the cave gave. The states after 0x44 (steps 154-157, 159-289) were not made again and hold about 30 rupees more
than runs bring there; rupees matter there only in the arrow rooms, which top the purse up anyway.

### The golden race on 2.0.0.7

`zelda-headless.py OUT 3 --golden --power-on --clips on --act model --parallel 8 --attempts 2000`, today's 51
networks on the branch after `main` (4be3f4c8) was merged: **three of three paths reach the ending**, 325 runs in
all (3 min 36 s: the served networks' batch window). All three verify (`tf replay verify`). Runs a segment (D3, D4,
D1, D2, D5, D7, D6, D8, D9 + Ganon) and the cheated items (`scripts/experiments/r32team_golden.py`; no rupee cheated
on any path):

| path (first seed) | runs a segment | runs | frames | cheated |
|---|---|---:|---:|---|
| 0 | 4 / 4 / 5 / 1 / 7 / 25 / 26 / 12 / 6 | 90 | 160,248 | 2 bombs (3:0x5b, 7:0x1a), 1 key (9:0x63) |
| 18,000 | 16 / 8 / 4 / 1 / 11 / 12 / 32 / 21 / 9 | 114 | 177,038 | 4 bombs (7:0x69, 7:0x19, 7:0x1a, 6:0x28) |
| 36,000 | 7 / 15 / 2 / 1 / 8 / 18 / 35 / 28 / 7 | 121 | 157,553 | 8 bombs (3:0x5b, 4:0x21, 4:0x11, 1:0x53, 0:0x2c, 7:0x19, 7:0x1a, 6:0x28) |

Path 0 replaces `corpus/replays/golden-d3first-51-networks.tfreplay` (fewest cheated bombs, then frames): two
bombs like 2.0.0.6's path, 19,870 frames shorter (180,118 before). It is the first golden path with a cheated key
(Level 9's 0x63; every earlier one cheated none).

### The batch's report step

`r38-2d-1k`'s chains all finished; its last entry, the report (`scripts/yardstick.py report --run promoted`, then
`triforce room-stats`), failed after 32 s, and so did the report of every batch since `r74-1k` (Nov 15: `r37rules`,
`r37fix`, `r37x1b`, `r37land-d2first`). **The inputs, not the tool:** the yardstick's specialist rows read each
promoted network's evaluation in its act mode, and seven promoted networks have none of their current weights:
`d2-rope`, `moldorm`, `ow-peahat`, `ow-start` and `ow-tektite` were widened to schema 6 by `triforce convert-model`
(Nov 16), whose `eval.json` stays the source's by design ("measure the converted run again for a canonical
evaluation of its own"); `boss-dodongo`'s evaluation is of weights older than its conversion's source; `ow-octorok`'s
run has none. The chain rows (arrivals, hearts, cheats) compute for every `d3first` segment, and `room-stats` ran
(`r38-2d-1k-room-stats.json`). The fix is to evaluate the seven in `temperature:0.5+relax:16`
(`python -m triforce.evaluate --specialist S`, or `triforce eval-batch TAG --families ...`), a queue job.

### The lab's reference

`triforce/lab/reference.json`'s `ow-zora` and `ow-zora@d3first` each played 10 episodes on `d3first_0_2dw_h370001_9`
and `_h370075_9`, whose objective was the cave. Played again with the bench's own command (`python -m
triforce.lab.bench run --checkpoint ow-zora=... ow-zora@d3first=...`, the keys' networks, the manifest's seeds; 14 s
through the queue): every other episode of the two keys identical, those 20 replaced: ow-zora 87.5% → 88.3%,
ow-zora@d3first 93.3% → 92.5%; the reference's total 92.9% unchanged, hearts lost 0.249 → 0.248 an episode, hits
0.677 → 0.675 a visit, bombs 0.360 → 0.357 an episode (docs/lab.md "The yardstick").

### Checks

- `entry_states audit --route d3first` exit 0 (855 states); `d2first`'s states and manifest entries untouched.
- The suites after `main` (4be3f4c8) was merged: pytest 946 passed, 12 skipped; `cargo test --release --workspace
  --features zelda/zcore` ok (the corpus test with the new path's pins among them); the zelda-asm tests 5 passed;
  `asm.py check` ok.

## 2.0.0.8: the hit heads promoted for `d3first`

Eight families play `d3first`'s own steps (before `own_models_until`) with their `hitshift` combination: the family's
promoted policy plus a supervised hit network that lowers the logits of actions it reads as getting Link hit (\"The
hit head across families\" above). Measured on 2.0.0.6 with the serve fix (CUDA graphs, one index a served network),
1000 runs paired with `r37fix-1k`: Dungeon 4 21.1% -> 68.4%, Dungeon 3 21.7% -> 39.2%, Dungeon 5 31.2% -> 66.1%
(the set as promoted). Promoted with `triforce promote FAMILY RUN --route d3first`; the runs they replace stay in
`models/`:

| Family | Promoted (`promoted.d3first`) | Replaced |
|---|---|---|
| boss-gleeok | r38head-fam-w8 | r34d4-gleeok4-s1-it2877 |
| like-like | r38head-fam-w8 | r34d4-like |
| vire | r38head-fam-w8 | r34d4-vire-s1 |
| stalfos-keese-gel | r38head-fam-w8 | r34d3-skg-s341 |
| darknut | r38head-fam-w8 | r34d5-dg-s362 |
| boss-manhandla | r38head-fam-w4 | r32d3-manh |
| zol | r38head-fam-w4 | r34d3-zol-s341 |
| gibdo | r38head-fam-w4 | r32late-gibdo |

Not yet: Dungeon 8's darknut, boss-gleeok and boss-manhandla (39.1% -> 80.9% on `d8`) play the bare keys, whose
`promoted` pointer `d2first` also reads: they wait for a `d3first`-only pointer for the steps after
`own_models_until` (branch `r38-late`), so `d2first` stays as it is. goriya's head (Dungeon 7 12.6% -> 79.1%) waits
for its Dungeon 1 and Dungeon 7 chains on today's code (branch `r38-hit`). The chains of record of 2.0.0.8 are
`r38-208-1k` (to come). The product of segment rates with every head is an estimate of about one in 1,000 (Head38b's
`r38_head.py product`); no unbroken run from the start has finished yet.

## Moldorm crossed, 9:0x56's key

**The owner's change** (Nov 22, after the scout's ledger, `target/owner-review/r38-moldorm-key.md`): Moldorm's key
"is only needed for d9? That's interesting because d9 0x56 drops a key and it's super safe to get. Confirm this
works, then drop the moldorm fight (it's now passthrough) and fight in d9 0x56 to get the missing key." Branch
`r39-key`, on 2.0.0.8 (v2.0.0.8, d8596127); `d2first` unchanged.

### Confirmed first

**The ledger** (`scripts/experiments/r39_keys.py ledger`: the route's key events, step numbers of the 290 steps,
checked by the scout against the 1000-run records). A state at step N holds the count after step N − 1.

| Step | Room | Event | Before | Now |
|---:|---|---|---:|---:|
| 96, 99 | 2:0x7e, 0x4e | +1 each (5 Ropes) | 4, 5 | 4, 5 |
| 100 | 2:0x3e | Moldorm's key → crossed | 6 | 5 |
| 127-142 | Level 5 | 0x06 W door, 0x66's Gibdo, 0x27 (+1, door), 0x25 W door | 5 out of Level 5 | 4 |
| 163-170 | Level 7 | 0x38 N, 0x18 E, 0x1b E doors | 2 out of Level 7 | 1 |
| 197 | 6:0x79 | W key door | 1 | **0** |
| 200-210 | Level 6 | 0x58, 0x29, 0x2d (+1 each), 0x2c N door | 3 out of Level 6 | 2 |
| 221, 222 | 8:0x5e, 0x4e | +1 (5 Blue Darknuts), N door | 3 out of Level 8 | 2 |
| **246** | **9:0x56** | **+1 (2 Zols, 2 Like-Likes)** | 3 (crossed) | **3** |
| 249, 262 | 9:0x14, 0x16 | E door, N door | 2, 1 | 2, 1 |
| 266 | 9:0x63 | W key door (into 0x62) | 0 | **0** |
| 268 | 9:0x61 | +1 (Patra) | 1, to the end | 1, to the end |

Never below 0: the count reaches 0 after 6:0x79's door (197; 6:0x58 gives one back at 200) and after 9:0x63's (266),
as before at 266; it ends at 1 both ways. From step 246 on the two ledgers are the same.

**The key is real.** game.yaml 9:0x56: `treasure: key`, `trigger: kill-all-item` (attribute F & 7 = 7). The
disassembly (`Secret_CheckUnderworld`, `Secret_UpdateRoomAllDead`): trigger 7 activates the room item (and plays
the tune) on the first frame no counting monster is left in slots 1-12; Bubbles (types `$2B`-`$2D`) never count,
a split Zol's Gels (`$14`) do, so the 2 Zols, their Gels and the 2 Like-Likes must die. The key lies at
(`$80`,`$90`) between two water rows (level-9.md: the ladder from the N or S door). In the record `r37fix-1k-d9`
(1000 runs from `9_76s_0`, the room a crossing then) 14 of 1030 visits killed all four, 6 took the key (keys 3 → 4)
and left W `exit-correct`: the ladder reaches it and the way out is the same bomb wall.

**What the step cost then:** the crossing took 470 frames and 0.001 hearts a visit (845 visits with no kill); the
visits that cleared the room took 2,170-2,672 frames and 0.14-0.21 hearts (a network trained to cross, not to
fight). Moldorm's fight cost about 1,000 frames and 0.5 hearts a visit (`r38-2d-1k-d2`), 1 death in 1000.

### The change

- routes.yaml: both routes write 2:0x4e, 0x3e and 9:0x56 themselves; the blocks are `dungeon-2-north` (now from
  0x2e), `dungeon-9` (0x76, 0x66) and `dungeon-9-from-55`. `d3first`: 0x3e `treasure: skip` (the Moldorm left
  alive; the room has no shutter), 0x56 with no `treasure:` (its key; `bombs: required` stays for the W wall).
  `d2first`: the Moldorm fought, 0x56 crossed, as before (`test_d2first_is_its_golden_list`). 290 steps.
- States: every `d3first` state after Moldorm's room through 9:0x56's entry (steps 101-246, 3 chains) holds one
  key fewer. They were set to the ledger by writing `InvKeys` (`$66E`) alone (`r39_keys.py patch`: a load and save
  without the write gives the same file byte for byte; the patched files' RAM differs from the old in that byte
  only); their manifest entries carry `keys: {was, now}`. Chain 2's states 215-266 were one key short already (the
  first order's build): 215-246 needed no change, 247-266 were set up to the ledger. 425 states; the audit is
  clean (855). The plain stretch's 4:0x30 (step 44) keeps its key (its door is 0x30's).

### Measured (2.0.0.8's team, not retrained; 1000 runs a segment, paired by seed with `r38-208-1k`)

Batch `r39-key-1k` on the branch at v2.0.0.8 with 2.0.0.8's networks, against the record `r38-208-1k` (the same
code base and team, seeds 0-999, `--act model`), each segment scored by its arrival:

| Segment | 2.0.0.8 | Moldorm crossed, 0x56's key | Paired | Hearts arriving | Frames to arrival (both arrive) |
|---|---:|---:|---:|---:|---:|
| `d2` | 99.3% | 99.0% | −0.3 ± 0.3 | 6.12 / 6.13 | 7,236 / 6,546 (986 seeds) |
| `d6` | 36.4% | 35.5% | −0.9 ± 2.1 | 6.30 / 6.22 | 9,525 / 9,552 (140) |
| `d9` (the ending) | 53.0% | **6.7%** | **−46.3 ± 1.8** | | |
| `full`: Dungeon 2's triforce, of 1000 | 167 | 158 | | | |

No key is cheated anywhere, before or after (`cheated_keys` 0 in every room of the four segments both ways; 6:0x79's
door is opened with the last key in every `d6` run that reaches it).

**Moldorm crossed** (2:0x3e, `d2`, 999 visits each way): 383 frames a visit against 1,008 (625 fewer; the segment 690
fewer to its arrival), but 0.78 hearts lost there against 0.52 (0.40 net of drops before), and **5 deaths against 1**;
0.18 kills a visit (9.5). From the start (175 visits each way): 578 frames against 1,056, 0.92 hearts against 0.60,
**7 deaths against 1**: Dungeon 2's triforce 158 against 167. The moldorm network was trained to fight; crossing, it
walks into the worm. The rooms after it are unchanged (0x2e 0 / 0 deaths, 0x1e 2 / 3, Dodongo 3 / 1).

**9:0x56 fought for its key: today's like-like network cannot take it.** Of 1016 visits, **873 end `stuck`** (the
family's 2,000-step cap; 744 of them after 4 kills, 115 after 3), 143 leave W, 128 with the key (7,162 frames and
0.43 hearts a visit). The cleared room's key lies inside a ring of water rows (rows 6-7 and 14-15 of the room's tile
grid, the bottom row Link enters on below them): the ladder must be walked into, which the network, trained to cross
by the bottom row, never learned; its curriculum tracks (`trajectories/like-like`, `9_56s_*`) cross without the key.
Past 0x56 the route holds: of the 127 runs that got through, 67 end the game (62 of the same seeds on 2.0.0.8).

**`d2first`:** its 100-run chain is the same as `d2first-100-land` in result, steps and frames on all 100 runs.

**Not landable as it is.** Both rooms need their network taught the new step, as round 38 did for 8:0x4c
(`r38t-pols`):

- **like-like for 9:0x56** (step 246 ≥ `own_models_until`: through `late.d3first`): fine-tune the family's
  `promoted` run on `d3first_9_56s_*` with the key objective (`specialists/d3first/like-like.yaml`, `state_routes:
  [d3first]`), new Go-Explore tracks that clear the room and take the key over the water (the explorer crossed this
  room in round 32), SIL as `r38t-pols`; a 1000-run `d9` against `r38-208-1k-d9`.
- **moldorm for 2:0x3e** (step 100: `promoted.d3first`): fine-tune `promoted` on `d3first_2_3es_*` with the crossing
  objective, against the fight's 1 death in 1000 and 0.52 hearts.
- The other key, if the fine-tune is not wanted: 8:0x4c's (step 226, in the open, no kill). Its network
  `r38t-pols` (2.0.0.10's `late.d3first`) was trained to leave it; `r78t4-pols` (d2first's) still takes it (92.2%
  from `8_4cc_0`, round38.md "8:0x4c"), at about 5,000 aimless frames a visit.
