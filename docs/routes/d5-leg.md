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
| 182 | 0x2c | E → N | plain | ow-tektite | its rock wall open since d2first (`requires_bombs`: the cheat grants a bomb here, taken back unused); `0_2ce_v2` |
| 183 | 0x1c | S → W | plain | ow-start | `0_1cs_v2` |
| 184 | 0x1b | E → maze | plain | ow-start | the Lost Hills: N four times at X `$70` (`exit: maze`); `0_1be_v2` |
| 185 | 0x0b | S → cave | plain | ow-leever | straight up at X `$70` into the mouth |
| 186 | 5:0x76 | S → N | shared (mountain rejoins) | gibdo | Dungeon 5 (`docs/routes/d5.md`) |

State names: a room entered again from the same side gets `_v2` (the names rule); the table names those. The
first visits keep the bare name (`0_45c`, `0_55n`, `0_54e`, ...). Both ways of a stretch's first room share one name
(`0_55n`, `0_3ee`, `0_1de`).

## Ledgers

- **Hearts (max):** 8 out of Dungeon 4; 9 after 0x5f (either way); 10 after 0x2f's cave; 11 after Digdogger (the
  audit counts heart containers along the way a step is on).
- **Rupees:** 42-54 in (after d2first's 0x67 cave; nothing in Dungeons 3 and 4 is taken), +100 at 0x0f: 142-154 out.
- **Keys:** 6 in (out of Dungeon 4), none used or found on the leg: 6 into Dungeon 5.
- **Bombs:** `bomb_reserve: 2` from step 134 (the value Dungeon 4's entrance set): BOMB stays masked while Link holds
  2 or fewer outside `requires_bombs` rooms, but for the 10th damage-free kill; those two are Dungeon 5's walls (0x66
  W, 0x65 W; 0x65's own bombs are refused westbound). No wall on the leg; 0x2c (plain mountain way) is a
  `requires_bombs` room whose wall is already open. Dungeon 5 sets 8 at 0x76.

## Building the states

- The clip way and the shared steps: `python -m triforce.entry_states build --route d2first --start-step 133
  --source 4_03s_0 --chains 0,1,2` (the build follows the clip way; it writes `0_45c` through `0_0be`, then
  `5_76s`). The leg slice staged the same 90 states on `d5leg`'s numbering (identical steps 133-163 of its clip way)
  in its worktree (`target/entry-states/d5leg-1`, not installed).
- The plain ways by `path` from the clip way's states (each source's manifest `route_step` gives the walk its step):
  - coast-heart: from `0_55n_<c>` through 0x56 .. 0x5d, 0x6d, 0x6e, 0x6f to 0x5f (from S), and on to 0x4f;
  - lake-3e: from `0_3ee_<c>` to 0x2e, 0x2d (from E), 0x1d (from S);
  - mountain: from `0_1de_<c>` to 0x2d (from N), 0x2c, 0x1c, 0x1b, 0x0b (from S) and 5:0x76 (`path` explores under
    `--route d2first`, so the maze loops follow 0x1b's `exit: maze`).
- Then `entry_states audit`, `install`, the families' `start_states` globs (the existing `0_<rr>*_[0-9]` globs pick
  up `_v2` names), curricula (`python -m triforce.explore --specialist NAME --states ...`) and trainings.

## Tests

`crates/zelda/tests/d5_leg.rs` plays the assembled route in the running game: the clip way by the objective from
`4_03s_0` through every clip-way step into 5:0x76, where Dungeon 5's first step takes over; the plain way the same
way (row 5 and 6, the plain ladder at 0x5f, 0x2e, the Lost Hills by their sequence into 0x0b from S and on into
Dungeon 5), each stretch's rejoin reached as its shared step; the clips switch turned off inside each stretch's clip
way (the plain way from the nearest room) and on 0x5f's water (held until the heart's platform). Maze loops:
`crates/env/tests/maze.rs`.
