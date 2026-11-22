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
