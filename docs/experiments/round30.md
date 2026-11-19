# Round 30: the optimisation round over the whole route

## Darknut, boss-manhandla, gibdo: Dungeon 5 at the route's hearts, then Dungeon 8

darknut plays 3:0x5b, 0x59, 0x69 and 0x5c in Dungeon 3, 5:0x64, 0x06, the Olive Garden 0x05 and 0x37 in Dungeon 5,
8:0x5e, 0x4e, 0x3e and 0x3f in Dungeon 8. Its line: `r75-d537-c5` (round 25, `main` until round 28) →
`r78t4-darknut` → `r78-dark37-s85` (round28.md §15: chosen on `d8`, checked on `d3` +1.4 ± 2.0 and `d5` +6.8 ± 2.0
from full hearts). Every round of the lineage trained half its episodes with CLIP masked; this is its first with
`clips: {enabled: true, off_share: 0}`.

### The cause

The full route (`r29land-1k-full`, 1000 runs, `r78-dark37-s85`) against `r77b-1k-full` (the same seeds, `main`'s
round 27 team with `r75-d537-c5`) and the honest planner beside (`r29land-plan-full`, 300 runs), darknut's visits by
route step: visits, hearts on entering, hearts lost a visit (the visits Link survives), deaths.

| visit | `r78-dark37-s85` | `r75-d537-c5` | the planner (300) |
|---|---|---|---|
| 5:0x05 from E (step 192) | 540, 7.06 in, **1.69**, **141 deaths** | 434, 7.42, 1.31, 80 | 284, 9.17, 0.73, 5 |
| 5:0x64 from the cellar (197) | 398, 5.44, 1.12, 40 | 354, 6.10, 1.15, 24 | 280, 8.39, 0.93, 4 |
| 5:0x37 (203) | 285, 4.05, 0.12, 4 | 264, 4.62, 0.16, 8 | 273, 7.35, 0.07, 1 |
| 3:0x59 from N (94) | 740, 4.89, 0.58, 4 | 641, 4.92, 0.70, 10 | 297, 5.71, −0.05, 1 |
| 3:0x69 from N (95) | 725, 4.32, 0.65, 12 | 631, 4.26, 0.71, 19 | 294, 5.78, 0.49, 0 |
| 3:0x5b from W (100) | 703, 3.48, −0.05, 15 | 603, 3.39, −0.14, 13 | 294, 5.25, −0.17, 0 |

The Olive Garden by the hearts Link brings (full route): 6 to 7.9 hearts in, 63 deaths of 245 visits against 30 of
168; 8 to 9.9, 2.61 hearts lost a visit against 1.65 (32 deaths of 186 against 24 of 187). From full hearts (the
`d5` segment, `r29land-1k-d5`) it loses 2.65 a visit with sword beams and 35 deaths of 691: the segment the run was
checked on starts where the route never arrives.

**Dungeon 5 at the route's hearts.** The leg after Dungeon 4 arrives at 5:0x76 with 7.77 hearts (the full route
7.74), and no darknut room is on it, so `--from 0_45c_0 --stop-at step:209` (the leg and Dungeon 5, to the overworld
after the D5 triforce) plays Dungeon 5 as the full route does. 500 runs of it (seeds 7000-7499, the harvest chain
below): **247 of 500 take the triforce (49.4%)**, the full route 262 of 540 entering (48.5%), the `d5` segment
72.2%; 0x05 entered with 7.27 hearts, 1.87 lost a visit, 116 deaths of 499.

**Dungeon 8** (`r29land-1k-d8`, 1000 runs from its entrance at 13 of 13 hearts; the full route enters with 12.50):
0x5e 1000 visits, 12.18 in, 4.04 lost, 20 deaths; 0x4e 979, 8.15, 2.91, 88; **0x3e 891, 5.68, 1.61, 414**; 0x3f
476, 5.49, 1.98, 155. darknut's four rooms end 677 of the 752 runs `d8` loses. The planner in the same rooms (full
route): 0x5e 0.97 hearts a visit, 0x4e 1.33 (3 deaths of 129), 0x3e −0.29 (none), 0x3f 0.10 (none).

**Dungeon 3**: darknut's rooms kill few runs (0x69 12, 0x5b 15 of 1000 full runs) but lose the hearts Manhandla
needs: 3:0x4d is entered with 3.58 hearts (the planner 5.42) and 81 of its 97 deaths are entries below 4 hearts.

All 73 curriculum tracks of `r78-dark37-s85` are complete: no track idles the resets (the recipe's fourth lever has
nothing to take out).
