# Round 33: the families phase IX left untrained

## like-like at 9:0x56, lanmola at 9:0x55 from E (Dungeon 9)

The owner's phase IX (round31.md) sends Dungeon 9 from 0x66 N into 0x56 (2 Bubbles, 2 Zols, 2 Like-Likes), across
it to its W bomb wall (step 310, `bombs: required`, the room `requires_bombs`) and into 0x55 from E (step 311, the
Lanmolas' room, crossed by the block clip `CLIP N` from (`$A0`,`$9D`)). Both rooms are new to their families:
like-like `r79c-like` never trained on 0x56, lanmola `r79b-lanmola` trained 0x55 from S.

### The cause

From the chains of record (`r31land-1k-d9`, 1000 runs from `9_76s_0`, seeds 0-999):

- **"stuck" is the bounce limit.** All 616 runs that end in 0x56 end `bounce-limit: like-like left into (level 9,
  room 0x66), the 5th return`. like-like walks straight back out of the S door it came in by: 1635 visits end
  `exit-wrong` after 7.6 decisions and 0.02 hearts (878 of the 1000 first visits), stalfos-keese-gel sends Link N
  again from 0x66, five times. 438 visits leave W as the route says (382 bombs laid, none cheated; 1250 frames,
  0.72 hearts a visit).
- **The route data and the engine are right.** At the S entry of `9_56s_0..2` (like-like's environment) the objective
  is `MOVE` with the next room W; the mask offers `MOVE N` and `MOVE S` (Link stands in the doorway). The walk's way
  played as a rule over the environment's own actions (below) is offered `BOMB W` at (`$20`,`$8D`) in 300 of 300
  runs with no masked choice. Nothing is missing in the engine: the model has never seen the room.
- **0x55 from E**: 438 visits: 368 cross by the clip (37 decisions, 0.18 hearts; the walk's rule takes 15 and none),
  54 wrong exits, 7 stuck, 9 deaths (16 runs lost).

### The arm (the recipe of decisions 2026-11-19)

- **The walk's rules as teachers** (`scripts/experiments/r33_d9_demos.py`, round 29's `r29_demos` loop over the
  family's environment with its own hp and bombs draws, `--hearts family`, 100 runs a chain):

  | Rule (family) | Way | Won | Hearts lost in the wins | Decisions | Rows |
  |---|---|---:|---:|---:|---:|
  | `wall-56` (like-like) | up to the bottom row Y `$BD`, W to X `$20`, up over the water to (`$20`,`$8D`), `BOMB W`, W (a held Link stabs) | 300 of 300 | 0.06 | 32.6 | 21,832 |
  | `clip-55e` (lanmola) | from the E door down to Y `$9D`, W to X `$A0`, `CLIP N`, the stairs | 300 of 300 | 0 | 15 | 4,500 |

  Each rule's first three won runs of each chain are curriculum tracks (`9_56s_<c>~rule100N`, `9_55e_<c>~rule100N`).
  Every track of both families finishes (like-like 48, lanmola 21): none is taken out.
- **Handoffs at the route's hearts** (`eval-chain --from 0_6dc_0 --runs 500 --first-seed 5000 --harvest DIR`, the
  landed team: Dungeon 9 entered as the route enters it): 9:0x56 is entered 1680 times with 11.41 hearts (the
  bounces included), 0x55 from E 629 times with 10.15. Drawn at `harvested_fraction: 0.25`: like-like 0x56 400, 0x14
  130, 0x51 66, 0x03 34; lanmola 0x55 from E 300 (`triforce/states/harvested/<family>`, not in git).
- **The runs**: like-like three seeds from `r79c-like` (`r33-like-s81`, `-s82`, `-s83`), lanmola one (`r33-lanmola-s80`
  from `r79b-lanmola`); 244 iterations, `--stop-success 1.01 --sil-coeff 0.1 --checkpoint-every 30`, clips
  `{enabled: true, off_share: 0}` (both yamls already), `--demo-coeff 0.05` over the lineage's files and the new one
  (like-like: `like-like-14`, `clip-03`, `wall-56`; lanmola: `clip-55`, `clip-55-300`, `clip-55e`).

### The selections on `d9`

`select-checkpoint like-like r33-like-sNN --segment d9 --runs 1000 --first-seed 2000 --confirm 1000
--confirm-first-seed 3000 --confirm-top 1 --only it003295,it003325,it003355,last[,promoted] --parallel 10`
(`r79c-like` pinned in s83's batch, `--no-promoted` in s81's and s82's), the rest of the team main's:

| run | it003295 | it003325 | it003355 | last | confirm (3000-3999) |
|---|---:|---:|---:|---:|---|
| `r33-like-s81` | 45.2% | 46.1% | **47.0%** | 45.2% | it003355 48.6% (3.77 hearts at the end), last 47.6% |
| `r33-like-s82` | **47.3%** | 44.9% | 46.7% | 47.3% | it003295 47.8% (3.76 hearts), last 45.5% |
| `r33-like-s83` | 45.1% | 44.9% | **47.7%** | 44.6% | **it003355 49.4% (3.80 hearts)**, last 43.7% |
| `r79c-like` (promoted) | | | | 13.0% | 13.6% (3.45 hearts) |

**On seeds not used to choose it, `r33-like-s83` at it003355 takes `d9` in 49.4% against 13.6%: +35.8 ± 1.9
points paired** (s81's choice 48.6%, s82's 47.8%: the three seeds agree within noise). 9:0x56 is crossed in every
visit (s81 it003295: 1004 of 1004 visits leave W; main's record: 438 of 2689). Main's record of the segment,
13.7% on seeds 0-999, matches the reference here.

**Not yet decided:** the other segments like-like plays, `d4` (4:0x32) and `d6` (6:0x1d), paired with `r79c-like`
on seeds 0-999 (`select-checkpoint like-like r33-like-s83 --segment SEG --runs 1000 --confirm 0 --only
it003355,promoted`), are queued. It is promoted if neither is worse beyond noise; then lanmola `r33-lanmola-s80` is
chosen on `d9` with it.

### Backlog

- like-like: the explorer's 9_56s tracks wander (a `BOMB S` in each) beside the rule's; set them aside if a later
  round sees the model lay bombs off the wall.
- like-like is trained on two lines now (this one for `d2first`'s 9:0x56, `r32d4-like2` for `d3first`'s Dungeon 4):
  the one team needs a joint continuation or a choice on both routes' segments.
