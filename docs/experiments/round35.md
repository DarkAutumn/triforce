# Round 35

## 2.0.0.2: round 34's two late runs land (branch `r35-land`)

The two `d3first` route runs that qualified after 2.0.0.1's chains of record had played, darknut `r34d5-dg-s362`
(it012716, the Olive Garden) and boss-digdogger `r34d7b-dig-s1` (Digdogger at the route's hearts), promoted `--route
d3first` on the 2.0.0.1 team; `VERSION` 2.0.0.2, no code or route data changed (no identity check owed). Their causes
and the full numbers are in round34.md "2.0.0.2"; decisions 2026-11-21 "2.0.0.2".

The chains of record `r35land-1k` (seeds 0-999, models alone, clips on, paired with `r34land-1k`):

| Segment | 2.0.0.1 | 2.0.0.2 | Paired |
|---|---:|---:|---:|
| `d3` (`3_7cs_0` → 24) | 18.0% | 23.2% | **+5.2 ± 1.8** |
| `d5` (`5_76s_0` → 120) | 27.4% | 33.0% | **+5.6 ± 2.0** |
| `d7` (`7_79s_0` → 194) | 38.7% | 42.3% | +3.6 ± 2.0 |

The gate (`d3`, `d5` not worse and one better, `d7` not worse) passed. From the game start (1000): the D3 triforce
23.5% (22.7%, +0.8 ± 1.9), D4's 4.5% (2.8%), D1's 3.8% (2.3%), D5's 12 runs (3), the magic sword 2, D6's 1, the
ending 0. The planner beside (300): the ending 20.3% (25.0%, -4.7 ± 3.5), its D5 triforce 47.7% (59.0%, -11.3 ±
4.1: darknut's 5:0x37 over the new run, 21 of 171 entering lost, 7 of them stalls). The golden path from power on:
three of three in 50, 105 and 53 runs (2.0.0.1: 51, 71, 60).

```sh
# the worktree's team (main's promoted run dirs and pointers + the two runs, promoted --route d3first)
A="--route d3first --clips on"; M="--runs 1000 --act model"
triforce eval-batch r35land-1k --parallel 10 \
  --chain "d3: $A --from 3_7cs_0 --stop-at step:24 $M --output target/r56-chains/r35land-1k-d3.jsonl" \
  --chain "d5: $A --from 5_76s_0 --stop-at step:120 $M --output target/r56-chains/r35land-1k-d5.jsonl" \
  --chain "d7: $A --from 7_79s_0 --stop-at step:194 $M --output target/r56-chains/r35land-1k-d7.jsonl" \
  --chain "full: $A $M --output target/r56-chains/r35land-1k-full.jsonl" \
  --chain "plan-full: $A --runs 300 --act temperature:0.5+relax:16+plan:16:1 --output target/r56-chains/r35land-plan-full.jsonl"
python zelda-headless.py target/golden-S 1 --golden --power-on --clips on --act model --parallel 8 --first-seed S  # S = 0, 700000, 800000
```
