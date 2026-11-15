# The leg to Dungeon 5 (route `d2first`, steps 134-185, round 24)

`triforce/routes.yaml` `d2first` goes on after the D4 triforce (step 133) with the owner's leg of Nov 13-14 (its clip
way and its plain way, `triforce/draft_routes.yaml` `first-quest` until round 24): from the triforce's warp to 0x45
back by raft to 0x55, the two coast heart containers (0x5f, 0x2f's take-any cave), 0x0f's 100 rupees, the mountain to
Dungeon 5's entrance 0x0b, then Dungeon 5 (steps 186-208, `docs/routes/d5.md`). Three stretches carry a clip way and
a plain way; the run-time clips switch picks one (`routes.yaml` header, `zelda::objectives::GameCompletion::
set_clips_allowed`). The mechanics are round 24's slices: `r24-leg` (round the world, 0x55's wrap over its river,
the overworld ladder and its clip, 0x1f's wall gap, 0x5f's heart as a room objective, the switch held on 0x5f's
water; decisions 2026-11-14 "The leg to Dungeon 5: its clip way"), `r24-maze` (the Lost Hills by `exit: maze`;
decisions "A maze loop is a room transition") and `r24-names` (a later visit's states are `..._v<N>_<chain>`).

## Steps

| Step | Room | Entry → exit | Way | Family | What Link does |
|---|---|---|---|---|---|
| 134 | 0x45 | cave → S | shared | ow-peahat | S off the pier at X `$80`: the raft down to 0x55's dock; `bomb_reserve: 2` |
| 135 | 0x55 | N → W | coast-heart, clips (parts) | ow-zora | CLIP E on Y `$85`/`$8D` over the river onto its W part (game.yaml `wrap_in`), W |
| 136 | 0x54 | E → W | clips | ow-zora | |
| 137 | 0x53 | E → W | clips | ow-moblin | |
| 138 | 0x52 | E → W | clips | ow-moblin | |
| 139 | 0x51 | E → W | clips | ow-moblin | wrap exit (CLIP E out of the solid W edge, Y `$75`-`$9D`) |
| 140 | 0x50 | E → S | clips | ow-lynel | |
| 141 | 0x60 | N → W | clips | ow-lynel | wrap exit round the world into 0x5f (room id − 1), Y `$85`/`$8D` |
| 142 | 0x5f | E → N | clips | ow-zora | on the water: the overworld ladder clip W to the heart container (9 hearts), the plain ladder to land, N |
| 143 | 0x55 | N → E | coast-heart, plain (parts) | ow-zora | E on Y `$85`/`$8D` |
| 144 | 0x56 | W → E | plain | ow-zora | |
| 145 | 0x57 | W → E | plain | ow-octorok | |
| 146 | 0x58 | W → E | plain | ow-octorok | |
| 147 | 0x59 | W → E | plain | ow-zora | `0_59w_v2` |
| 148 | 0x5a | W → E | plain | ow-zora | `0_5aw_v2` |
| 149 | 0x5b | W → E | plain | ow-moblin | upper rows; `0_5bw_v2` |
| 150 | 0x5c | W → E | plain | ow-moblin | `0_5cw_v2` |
| 151 | 0x5d | W → S | plain | ow-moblin | S at X `$C0`; `0_5dw_v2` |
| 152 | 0x6d | N → E | plain | ow-moblin | |
| 153 | 0x6e | W → E | plain | ow-moblin | |
| 154 | 0x6f | W → N | plain | ow-zora | |
| 155 | 0x5f | S → N | plain | ow-zora | E on the land, the plain ladder twice to the heart container (9 hearts), back W, N |
| 156 | 0x4f | S → N | shared (coast-heart rejoins) | ow-zora | |
| 157 | 0x3f | S → N | shared | ow-zora | onto the dock (X `$60`, Y `$7D`): the raft N |
| 158 | 0x2f | S → cave | shared | ow-start | from the pier into the take-any cave |
| 159 | 0x2f cave | S → S | shared, `waypoint: heart` | ow-start | the right ware: the heart container (10 hearts), never the potion |
| 160 | 0x2f | cave → S | shared | ow-start | S off the pier: the raft down |
| 161 | 0x3f | N → W | shared | ow-zora | |
| 162 | 0x3e | E → W | lake-3e, clips (parts) | ow-zora | wrap exit (Y `$85`-`$BD`) |
| 163 | 0x3d | E → N | clips | ow-moblin | |
| 164 | 0x2d | S → N | clips | ow-zora | `0_2ds_v2` |
| 165 | 0x3e | E → N | lake-3e, plain (parts) | ow-zora | |
| 166 | 0x2e | S → W | plain | ow-zora | |
| 167 | 0x2d | E → N | plain | ow-zora | |
| 168 | 0x1d | S → E | shared (lake-3e rejoins) | ow-peahat | into its E part (S lanes X `$70`-`$80`) |
| 169 | 0x1e | W → E | shared | ow-tektite | |
| 170 | 0x1f | W → N | shared | ow-peahat | the wall gap at X `$80` |
| 171 | 0x0f | S → cave | shared | ow-start | the open secret cave |
| 172 | 0x0f cave | S → S | shared, `waypoint: rupees` | ow-start | walking up at X `$78`: +100 rupees |
| 173 | 0x0f | cave → S | shared | ow-start | |
| 174 | 0x1f | N → W | shared | ow-peahat | down through the gap |
| 175 | 0x1e | E → W | shared | ow-tektite | |
| 176 | 0x1d | E → N | mountain, clips (parts) | ow-peahat | N at X `$D0` |
| 177 | 0x0d | S → W | clips | ow-tektite | |
| 178 | 0x0c | E → W | clips | ow-tektite | wrap exit onto the mountain top (Y `$65`-`$AD`) |
| 179 | 0x0b | E → cave | clips | ow-leever | Dungeon 5's mouth (14, 8) |
| 180 | 0x1d | E → S | mountain, plain (parts) | ow-peahat | the S lanes (X `$70`-`$80`) |
| 181 | 0x2d | N → W | plain | ow-zora | |
| 182 | 0x2c | E → N | plain | ow-tektite | its rock wall open since d2first's heart cave (`requires_bombs`, but no cheated bomb once the wall is open: `cheats::bomb_need`; BOMB masked, no bomb target); `0_2ce_v2` |
| 183 | 0x1c | S → W | plain | ow-start | `0_1cs_v2` |
| 184 | 0x1b | E → maze | plain | ow-start | the Lost Hills: N four times at X `$70` (`exit: maze`); `0_1be_v2` |
| 185 | 0x0b | S → cave | plain | ow-leever | straight up at X `$70` into the mouth |
| 186 | 5:0x76 | S → N | shared (mountain rejoins) | gibdo | Dungeon 5 (`docs/routes/d5.md`) |

State names: a room entered again from the same side gets `_v2` (the names rule); the table names those. The
first visits keep the bare name (`0_45c`, `0_55n`, `0_54e`, ...). Both ways of a stretch's first room share one name
(`0_55n`, `0_3ee`, `0_1de`).

## Ledgers

Measured on the three `4_03s_<c>` states (the D4 triforce's room, step 133) and by the running-game walks of
`crates/zelda/tests/d5_leg.rs`.

| | In (step 133) | Along the leg | Out (5:0x76, step 186) |
|---|---|---|---|
| Hearts (max) | 8 | 0x5f's container on either way (9; the audit counts it once), 0x2f's take-any cave's right ware (10) | 10; 11 after Digdogger |
| Rupees | 68, 57, 63 (chains 0-2: Dungeon 4's rupee room 4:0x11 and drops) | +100 in 0x0f's cave (walking up at X `$78`); nothing bought (0x2f's take-any cave costs nothing, the potion is never taken) | 157-168 plus drops |
| Keys | 6 | none used, none found | 6 into Dungeon 5 (spent at 5:0x06 W, 0x27 W, 0x25 W) |
| Bombs | 0, 2, 6 (of 8) | `bomb_reserve: 2` from step 134 (Dungeon 4's value): BOMB masked at 2 or fewer outside `requires_bombs` rooms but for the 10th damage-free kill; no wall on the leg; 0x2c (plain mountain way) grants no bomb, its wall is open | as in; Dungeon 5 sets `bomb_reserve: 8` at 0x76, and an empty bag gets a cheated bomb for each of 0x66 W and 0x65 W |

## States (built, round 24)

150 states in `triforce/states/`, chains 0-2: 93 from the clip way's build (31 a chain: `0_45c` .. `0_0be`, then
`5_76s`) and 57 from the plain ways' walks (19 a chain). Audit: every one clean (hearts 8 to 0x5f on either way, 9
after it, 10 after 0x2f's cave). Chain `c` is played from `4_03s_<c>`, its own Dungeon 4 run (rupees 68/57/63,
bombs 0/2/6, as the ledger above), not all three from `4_03s_0`.

- **Clip way and shared steps**, one build a chain (13 min each, three at once):
  `python -m triforce.entry_states build --route d2first --start-step 133 --source 4_03s_<c> --chains <c>`. The
  build plays on into Dungeon 5 and stops at the Olive Garden (5:0x05, 3 attempts); only the leg and `5_76s` were
  installed (Dungeon 5's states are its second pass's).
- **Plain ways**, by `path` from the installed clip-way sources (their manifest `route_step` names the later
  visits `_v2`): `--source 0_55n_<c> --rooms 0:0x56,0:0x57,0:0x58,0:0x59,0:0x5a,0:0x5b,0:0x5c,0:0x5d,0:0x6d,0:0x6e,
  0:0x6f,0:0x5f,0:0x4f`, `--source 0_3ee_<c> --rooms 0:0x2e,0:0x2d,0:0x1d`, `--source 0_1de_<c> --rooms
  0:0x2d,0:0x2c,0:0x1c,0:0x1b,0:0x0b,5:0x76` (all `--route d2first --chain <c>`). Each way reaches its rejoin
  (`0_4fs`, `0_1ds`, `5_76s`) under the clip way's name; the clip way's are installed, the plain way's stay staged.
  So `5_76s_0..2` are the clip way's (from `0_0be_<c>`); the plain way enters 5:0x76 from `0_0bs_<c>` the same way
  (12-78 steps).

Explorer effort, env steps of the crossing into each state (chains 0 / 1 / 2; one attempt each unless noted):

| Step | State | Env steps | | Step | State | Env steps |
|---|---|---|---|---|---|---|
| 134 | `0_45c` | 4467 / 4166 / 4241 | | 160 | `0_2fc` | **63380 / 21735 / 5807** |
| 135 | `0_55n` | 4807 / 5347 / 4896 | | 161 | `0_3fn` | 4751 / 5699 / 5438 |
| 136 | `0_54e` | 4128 / 3357 / 3550 | | 162 | `0_3ee` | 3602 / 3528 / 3307 |
| 137 | `0_53e` | 4803 / 4153 / 4400 | | 163 | `0_3de` | 4360 / 4020 / 3815 |
| 138 | `0_52e` | 4451 / 5907 / 4909 | | 164 | `0_2ds_v2` | 7524 / 4676 / 5472 |
| 139 | `0_51e` | 4808 / 3971 / 5362 | | 166 | `0_2es` | 3741 / 4821 / 4292 |
| 140 | `0_50e` | 4348 / 4414 / 4552 | | 167 | `0_2de` | 3387 / 3439 / 3032 |
| 141 | `0_60n` | 3131 / 3750 / 3686 | | 168 | `0_1ds` | 5006 / 4638 / 4606 |
| 142 | `0_5fe` | 4386 / 4500 / 4162 | | 169 | `0_1ew` | 3510 / 3563 / 3964 |
| 144 | `0_56w` | 3698 / 3364 / 3292 | | 170 | `0_1fw` | 3470 / 3385 / 3832 |
| 145 | `0_57w` | 4253 / 4110 / 4847 | | 171 | `0_0fs` | 6637 / 5140 / 5109 |
| 146 | `0_58w` | 4086 / 3689 / 3743 | | 172 | `0_0fcave` | 4748 / 6267 / 5543 |
| 147 | `0_59w_v2` | 4104 / 3978 / 4424 | | 173 | `0_0fc` | **18669 / 8875 / 15168** |
| 148 | `0_5aw_v2` | 3780 / 3859 / 4369 | | 174 | `0_1fn` | 4769 / 5255 / 6691 |
| 149 | `0_5bw_v2` | 4130 / 4057 / 4155 | | 175 | `0_1ee` | 4305 / 4744 / 5305 |
| 150 | `0_5cw_v2` | 4114 / 4408 / 4998 (crossed again 1 / 3 / 1 times) | | 176 | `0_1de` | 4380 / 5332 / 4236 |
| 151 | `0_5dw_v2` | 4357 / 5903 / 4862 (after 1 / 3 / 1 failed crossings) | | 177 | `0_0ds` | 5229 / 5391 / 6078 |
| 152 | `0_6dn` | 5361 / 4246 / 6428 (crossed again 1 / 0 / 3 times) | | 178 | `0_0ce` | 4842 / 4986 / 5054 |
| 153 | `0_6ew` | 3073 / 2671 / 2658 (after 1 / 0 / 3 failed crossings) | | 179 | `0_0be` | 4130 / 3984 / 3990 |
| 154 | `0_6fw` | 4800 / 4888 / 4399 | | 181 | `0_2dn` | 3934 / 3701 / 3505 |
| 155 | `0_5fs` | 3871 / 3525 / 3922 | | 182 | `0_2ce_v2` | 4109 / 3835 / 4373 |
| 156 | `0_4fs` | 4919 / 5053 / 4689 | | 183 | `0_1cs_v2` | 4239 / 5025 / 4802 |
| 157 | `0_3fs` | 4261 / 4606 / 4493 | | 184 | `0_1be_v2` | 2741 / 4316 / 3818 |
| 158 | `0_2fs` | 3771 / 3496 / 3803 | | 185 | `0_0bs` | **5444 / 9126 / 15150** (the Lost Hills) |
| 159 | `0_2fcave` | 4754 / 4807 / 5282 | | 186 | `5_76s` | 4764 / 4466 / 4532 |

Every clip-way step was found in one attempt with no help; the slowest crossings are the way out of the two caves.
On the plain coast way `path`'s backtracking was needed twice a chain at most: 0x5c's E exit can't be reached from
the lane 0x5b's lower opening leads into (one failed crossing is 3 attempts, about 120,000 env steps), so 0x5b → 0x5c
was crossed again with a new explore seed (once, three times, once); 0x6d's E exit likewise from some of 0x5d's S
lanes (0x6d crossed again once and three times in chains 0 and 2, about 165,000 env steps a failure).

**The clips switch per state** (`objectives::start_clips`, measured on every state by the families' envs): on (clip
way only) `0_54e`, `0_53e`, `0_52e`, `0_51e`, `0_50e`, `0_60n`, `0_5fe`, `0_3de`, `0_2ds_v2`, `0_0ds`, `0_0ce`,
`0_0be`; off (plain way only) `0_56w`, `0_57w`, `0_58w`, `0_59w_v2`, `0_5aw_v2`, `0_5bw_v2`, `0_5cw_v2`,
`0_5dw_v2`, `0_6dn`, `0_6ew`, `0_6fw`, `0_5fs`, `0_2es`, `0_2de`, `0_2dn`, `0_2ce_v2`, `0_1cs_v2`, `0_1be_v2`,
`0_0bs`; drawn (both ways, or shared) the rest: `0_45c`, `0_55n`, `0_4fs`, `0_3fs`, `0_2fs`, `0_2fcave`, `0_2fc`,
`0_3fn`, `0_3ee`, `0_1ds`, `0_1ew`, `0_1fw`, `0_0fs`, `0_0fcave`, `0_0fc`, `0_1fn`, `0_1ee`, `0_1de`. Each changed
family's env played every new state for 3 × 300 random legal steps: every reset worked, CLIP was never offered with
the switch off, every episode ended (exit-correct, `enter-room:5:0x76` out of 0x0b, exit-wrong, death), and BOMB
in the plain way's 0x2c is offered only for the 10th kill (`$50` = 9).

## Families and curricula

`start_states` globs: ow-zora `0_56*`, `0_54*`, `0_5f*`, `0_4f*`, `0_3f*`, `0_3e*`, `0_2e*`, `0_6f*` (its `0_2d*`,
`0_59*`, `0_5a*`, `0_55*` take the rest); ow-moblin `0_53*`, `0_52*`, `0_51*`, `0_6d*`, `0_6e*`; ow-lynel `0_50*`,
`0_60*`; ow-octorok `0_57*`; ow-peahat `0_1d*`, `0_1f*`; ow-tektite `0_1e*`, `0_0d*`, `0_0c*`; ow-leever `0_0b*`;
ow-start `0_2f*`, `0_0f*`. Success endings were already right (0x0b's `enter-room:5:0x76`; every other leg exit,
cave mouths included, is an `exit-correct`). `max_steps` holds the new rooms' explored lengths (at most 208 steps,
0x5f's plain ladder, against ow-zora's 2000; elsewhere at most 185 against 600).

Curricula (`python -m triforce.explore --specialist NAME --states ... --parallel 8 --out trajectories`; gitignored,
so they live with the states' builder until copied), every state solved, one track a state; lengths in steps
(chains 0 / 1 / 2):

- **ow-zora** (48 tracks): `0_55n` 54/58/39, `0_54e` 75/87/12, `0_5fe` 69/67/35 (the ladder clip and the heart),
  `0_4fs` 37/99/43, `0_3fs` 41/24/40 (onto the dock), `0_3fn` 35/30/23, `0_3ee` 21/3/5, `0_2ds_v2` 71/34/55, `0_56w`
  64/42/73, `0_59w_v2` 17/12/49, `0_5aw_v2` 44/55/42, `0_6fw` 72/15/36, `0_5fs` 120/126/208 (the plain ladder twice
  and the heart), `0_2es` 16/42/39, `0_2de` 58/22/66, `0_2dn` 72/40/58.
- **ow-moblin** (27): `0_53e` 62/41/52, `0_52e` 105/67/57, `0_51e` 24/72/29 (the wrap exit), `0_3de` 91/22/44,
  `0_5bw_v2` 64/38/56, `0_5cw_v2` 47/48/106, `0_5dw_v2` 185/78/71, `0_6dn` 9/11/22, `0_6ew` 52/91/71.
- **ow-peahat** (15): `0_45c` 4/3/4 (off the pier), `0_1ds` 13/27/156, `0_1fw` 72/41/42 (the wall gap), `0_1fn`
  29/98/18, `0_1de` 29/33/98.
- **ow-leever** (6): `0_0be` 12/12/16, `0_0bs` 17/17/17.
- **ow-lynel** (15; curriculum new): `0_50e` 26/41/29, `0_60n` 73/47/139 (CLIP E on Y `$85`/`$8D` only: random play
  offered it 0-2 times in 900 steps a chain), and its first states `0_0as` 83/50/73, `0_0acave` 12, `0_0ac`
  124/109/127 (the white sword cave's entry, inside and exit).
- **ow-start** (40; curriculum new, and one-room episodes): `0_2fs` 6, `0_2fcave` 17 (the heart container, never the
  potion), `0_2fc` 4 (off the pier: the raft), `0_0fs` 7, `0_0fcave` 111 (the rupees and the way out), `0_0fc` 6,
  `0_1cs_v2` 14/33/33, `0_1be_v2` 49/55/45 (the Lost Hills' four Ns), and its first states `start` 7, `0_77n`
  13/8/13, `0_77cave` 12, `0_77c` 10, `0_1cs` 18/17/18, `0_1be` 4/4/3. The cave tracks and the Lost Hills' were
  explored again with `--iterations 8000 --patience 3000` (shorter: 23 → 17, 121 → 111, 78-122 → 45-55).
- **No curriculum** for ow-tektite (0x1e a walk each way, 0x0d a walk, 0x0c's wrap exit one CLIP from the arrival,
  0x2c's plain crossing a walk) and ow-octorok (0x57, 0x58 W → E walks): no step or item order to find, the kind
  of crossing both families learned without tracks. Random play (900 steps a state) crossed 0x1e, 0x0c (68-108
  times), 0x0d and 0x57, but neither 0x58 nor the plain 0x2c: watch those two in training.

Both-ways states are explored with the switch on (their tracks follow the clip way: `0_3ee` wraps W, `0_55n` wraps
onto 0x55's W part); plain-only states with it off. A curriculum frame and an explored cell take the step of their
seed state's entry, so tracks must not leave a room and come back into it: that is why ow-start, whose episodes ran
on across its rooms (0x2f → its cave → 0x2f → the raft), is one-room now (the first explore of `0_0fs` replayed into
the cave a second time, a wrong exit, after the explorer had scored it as correct).

## Tests

`crates/zelda/tests/d5_leg.rs` plays the assembled route in the running game: the clip way by the objective from
`4_03s_0` through every clip-way step into 5:0x76, where Dungeon 5's first step takes over; the plain way the same
way (row 5 and 6, the plain ladder at 0x5f, 0x2e, the Lost Hills by their sequence into 0x0b from S and on into
Dungeon 5), each stretch's rejoin reached as its shared step. The clips switch turned off inside each stretch's clip
way: lake and mountain re-plan to their plain way from the nearest room; the coast stretch past 0x55's river (0x55's
W part, 0x54, 0x53, 0x52, 0x51, 0x50, 0x60, 0x5f's water, each tried) holds the clip way to the rejoin 0x4f, never
turning back, and goes on by the shared steps (0x3f, 0x2f's cave, 0x3f) into the lake's plain way; turned off in
0x55 on the dock side the plain way takes over at once. On 0x5f's water and its platform the switch stays held
until 0x4f. Maze loops: `crates/env/tests/maze.rs`.
