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
