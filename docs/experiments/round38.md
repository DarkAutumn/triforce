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
stalfos-keese-gel 0.839 / 0.788, Digdogger 0.841 / 0.852. It is the head the one-network work wants; it has not
played yet.

**Chains: UNRELIABLE, to be measured again.** A chain that serves two or more torch networks could, until main's fix
(Nov 22), play one family's rooms with another family's network. Both combination chains served four: Dungeon 4
21.1% → 54.7% (+33.6 ± 1.9, 1000 paired seeds against main's 2.0.0.6 record) and Dungeon 3 21.7% → 40.1% (+18.4 ±
2.0) are not results. Reliable beside them: the team on this branch plays the record's runs identically (100 of 100);
the planner reaches 93.7% on Dungeon 4 and 57.0% on Dungeon 3 (300 seeds).
