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
at step 215 with seed 14042 to step 229; the single network in every slot, `temperature:0.5+relax:16`): **both
reproduce the golden game exactly** (1,493 and 11,160 frames, the two visits' 948 and 2,317 frames), so the
probabilities below are the ones the golden run drew from. The objective's wavefront step at each decision is the
`target` observation of a single-family env replaying the same actions from the same state (0:0x77 from the game
start; 8:0x4c from the replay's keyframe after the Pols Voices died, as `d3first_8_4cc_*` so the env plays d3first's
step), which reproduces the chain's score of the visit to the frame.

| visit | frames | calm frames | needed | aimless | calm decisions | away | reversals |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0:0x77, the opening (step 0) | 948 | 563 | 90 | **457** | 45 | 13 | 26 |
| 8:0x4c after the Pols Voices (step 228) | 2,317 | 1,495 | 180 | **1,282** | 119 | 35 | 11 |

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
`8_4cc_0` (up the cellar's stairs: the Pols Voices, the bomb wall) to step 229. Aimless frames a visit of 0:0x77
(step 0) and 8:0x4c (step 228): mean, median, 90th percentile; success: the run reached the stop; stuck: the
family's 2,000-step cap.

| arm | team: 0:0x77 aimless (mean / med / p90) | reversals | team: 8:0x4c aimless | success (stuck) | one: 0:0x77 | one: 8:0x4c | success |
|---|---|---:|---|---|---|---|---|
| `temperature:0.5+relax:16` (today's) | (the route from the start: to come) | | **5,239 / 3,723 / 14,483** | 93.0% (21) | | | |
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

**Act mode alone.** A lower temperature and `escape:16` change nothing in the opening (707-710 against each other;
the loop's moves are already near 1, so sharpening them keeps the loop and the stall rule's switch at 16 decides
when it breaks) and make 0x4c worse (5,475-6,149 against today's 5,239, success down 1.3-4.7 points). **`relax:4`**
(plain sampling after 4 decisions without progress instead of 16) **trims the opening by about 20%** (574 against
710 for the team, 537 against 664-683 for the one network; reversals 34 against 42; the visit 150 frames shorter),
and **does nothing for 0x4c** (team 5,647 against 5,239, success 89.3% against 93.0%, 31 stuck runs against 21;
the one network 5,101 against its other arms' 5,918-6,365). Its cost elsewhere is not measured here: it switches
every room to plain sampling after 4 stalled decisions, fights included, where today's `relax:16` keeps the
temperature's sharper play for 16; a fight that stalls 4 decisions (waiting out a Darknut, a boss's pattern) would
draw from the raw policy four times as soon. Making it the default needs the route's record under it (300 runs from
the start); a calm-only form (sooner plain sampling on calm decisions only) would leave fights alone, but the 20%
is the ceiling it could reach in the opening and 0x4c shows no gain to keep. So: nothing in act mode tones 0x4c
down, and the opening only by a fifth; both need training.

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
