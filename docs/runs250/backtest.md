# The 250 test of every version on its own code and team (back-test, Nov 23-24)

Each version 2.0.0.0-2.0.0.12 played on its own tagged code (`~/git/triforce-eval250-v<version>`, branch `eval250/v<version>`, local only; bundle `/backup/triforce/eval250-branches.bundle`) with its own team (`/backup/triforce/models/teams/<version>.json`, by checkpoint SHA-1): 250 runs of d3first from console power on, seeds 0-249, every run recorded (format 2), and a golden race capped at 10 minutes. The replays carry the replay streams: **full** (targets with roles, masks with reasons, every decision) where the round-40 change applied to that version as it was, **minimal** (each decision's objective kind, so whether it was a fight, and the legal actions, so whether BOMB was allowed) where porting the full stream would have meant porting the version's mask and objective code (the owner's rule). **Identity**: the same 5 seeds before and after the port, every room record equal (the decision statistics surprisal and act entropy differ by float rounding, at most 1.7e-05). The records were computed by main's `scripts/full_runs.py stats --from-replays` from the replays alone. 2.0.0.13-2.0.0.15 are main's own records (`docs/runs250/<version>.json`).

| version | stream | identity (5 seeds) | won | triforces D3 D4 D1 D2 D5 D7 D6 D8 D9 | died / stuck | furthest step median / max | stuck kinds | golden (speedrun timing) |
|---|---|---|---|---|---|---|---|---|
| 2.0.0.15 | main (format 2) | main record | 1/250 | 104 61 58 57 28 13 3 3 1 | 243 / 6 | 22.0 / 290 | bounce-limit 3, off-route 1, room-step-cap 2 | not raced |
| 2.0.0.14 | main (format 2) | main record | 0/250 | 94 57 54 53 27 12 3 2 0 | 245 / 5 | 22.0 / 264 | bounce-limit 3, off-route 1, room-step-cap 1 | not raced |
| 2.0.0.13 | main (format 2) | main record | 1/250 | 94 57 54 53 27 15 2 2 1 | 245 / 4 | 22.0 / 290 | bounce-limit 2, off-route 1, room-step-cap 1 | not raced |
| 2.0.0.12 | full | 5 of 5 (float stats drift 2e-05) | 0/250 | 94 57 54 53 23 9 2 2 0 | 243 / 7 | 22.0 / 284 | bounce-limit 4, room-step-cap 3 | 38:03.922 |
| 2.0.0.11 | full | 5 of 5 (float stats drift 1e-05) | 1/250 | 94 57 54 53 23 9 2 1 1 | 242 / 7 | 22.0 / 290 | bounce-limit 4, room-step-cap 3 | 37:51.060 |
| 2.0.0.10 | full | 5 of 5 (float stats drift 2e-05) | 0/250 | 94 57 54 53 23 9 0 0 0 | 238 / 12 | 22.0 / 206 | bounce-limit 9, room-step-cap 3 | 38:18.564 |
| 2.0.0.9 | full | 5 of 5 (float stats drift 1e-05) | 0/250 | 71 33 32 31 13 7 0 0 0 | 241 / 9 | 22.0 / 206 | bounce-limit 8, room-step-cap 1 | 39:17.784 |
| 2.0.0.8 | full | 5 of 5 (float stats drift 8e-06) | 0/250 | 71 33 29 28 15 0 0 0 0 | 247 / 3 | 22.0 / 173 | bounce-limit 2, room-step-cap 1 | 42:17.188 |
| 2.0.0.7 | full | 5 of 5 (float stats drift 0) | 0/250 | 52 6 6 6 0 0 0 0 0 | 248 / 2 | 21.0 / 143 | room-step-cap 2 | 39:33.524 |
| 2.0.0.6 | full | 5 of 5 (float stats drift 0) | 0/250 | 52 6 6 3 0 0 0 0 0 | 247 / 3 | 21.0 / 145 | room-step-cap 3 | 40:32.943 |
| 2.0.0.5 | minimal | 5 of 5 (float stats drift 0) | 0/250 | 68 7 7 5 1 0 0 0 0 | 250 / 0 | 22.0 / 164 | - | 40:36.654 |
| 2.0.0.4 | minimal | 5 of 5 (float stats drift 0) | 0/250 | 68 7 7 0 0 0 0 0 0 | 250 / 0 | 22.0 / 110 | - | 39:09.264 |
| 2.0.0.3 | minimal | 5 of 5 (float stats drift 0) | 0/250 | 68 7 7 0 0 0 0 0 0 | 250 / 0 | 22.0 / 110 | - | 40:01.512 |
| 2.0.0.2 | minimal | 5 of 5 (float stats drift 0) | 0/250 | 68 7 7 2 2 0 0 0 0 | 250 / 0 | 22.0 / 187 | - | 40:17.352 |
| 2.0.0.1 | minimal | 5 of 5 (float stats drift 0) | 0/250 | 51 9 5 3 3 1 1 0 0 | 250 / 0 | 20.0 / 238 | - | 39:58.167 |
| 2.0.0.0 | minimal | 5 of 5 (float stats drift 0) | 0/250 | 39 1 1 0 0 0 0 0 0 | 250 / 0 | 19.0 / 103 | - | 39:24.789 |

Not measured: the 1.x versions (the minimal stream is ported and compiles for 1.9.0.1 - 1.5.0.1, built for 1.9.0.1 only; none played), 1.5.0.0 and older (not ported).

## Detail (`full_runs.py analyze`, the first record is the baseline)

### Runs-250: 2.0.0.15 main vs 2.0.0.14 main vs 2.0.0.13 main vs 2.0.0.12 backtest vs 2.0.0.11 backtest vs 2.0.0.10 backtest vs 2.0.0.9 backtest vs 2.0.0.8 backtest vs 2.0.0.7 backtest vs 2.0.0.6 backtest vs 2.0.0.5 backtest vs 2.0.0.4 backtest vs 2.0.0.3 backtest vs 2.0.0.2 backtest vs 2.0.0.1 backtest vs 2.0.0.0 backtest

| build | record | runs | won % | died % | stuck % | median step | stuck kinds | median frames to the end | wall s | decisions/s |
|---|---|---|---|---|---|---|---|---|---|---|
| 2.0.0.15 main | 2.0.0.15.json | 250 | 0.4 ± 0.4 | 97.2 ± 1.0 | 2.4 ± 1.0 | 22.0 | bounce-limit 3, off-route 1, room-step-cap 2 | 140229 | 690.8 | 556.3 |
| 2.0.0.14 main | 2.0.0.14.json | 250 | 0.0 ± 0.0 | 98.0 ± 0.9 | 2.0 ± 0.9 | 22.0 | bounce-limit 3, off-route 1, room-step-cap 1 | – | 558.5 | 648.3 |
| 2.0.0.13 main | 2.0.0.13.json | 250 | 0.4 ± 0.4 | 98.0 ± 0.9 | 1.6 ± 0.8 | 22.0 | bounce-limit 2, off-route 1, room-step-cap 1 | 144591 | 339.8 | 1073.9 |
| 2.0.0.12 backtest | 2.0.0.12_backtest.json | 250 | 0.0 ± 0.0 | 97.2 ± 1.0 | 2.8 ± 1.0 | 22.0 | bounce-limit 4, room-step-cap 3 | – | 729.4 | 501.7 |
| 2.0.0.11 backtest | 2.0.0.11_backtest.json | 250 | 0.4 ± 0.4 | 96.8 ± 1.1 | 2.8 ± 1.0 | 22.0 | bounce-limit 4, room-step-cap 3 | 147660 | 724.6 | 504.7 |
| 2.0.0.10 backtest | 2.0.0.10_backtest.json | 250 | 0.0 ± 0.0 | 95.2 ± 1.4 | 4.8 ± 1.4 | 22.0 | bounce-limit 9, room-step-cap 3 | – | 618.8 | 580.9 |
| 2.0.0.9 backtest | 2.0.0.9_backtest.json | 250 | 0.0 ± 0.0 | 96.4 ± 1.2 | 3.6 ± 1.2 | 22.0 | bounce-limit 8, room-step-cap 1 | – | 433.6 | 675.0 |
| 2.0.0.8 backtest | 2.0.0.8_backtest.json | 250 | 0.0 ± 0.0 | 98.8 ± 0.7 | 1.2 ± 0.7 | 22.0 | bounce-limit 2, room-step-cap 1 | – | 258.0 | 1124.4 |
| 2.0.0.7 backtest | 2.0.0.7_backtest.json | 250 | 0.0 ± 0.0 | 99.2 ± 0.6 | 0.8 ± 0.6 | 21.0 | room-step-cap 2 | – | 86.0 | 2234.0 |
| 2.0.0.6 backtest | 2.0.0.6_backtest.json | 250 | 0.0 ± 0.0 | 98.8 ± 0.7 | 1.2 ± 0.7 | 21.0 | room-step-cap 3 | – | 88.8 | 2163.3 |
| 2.0.0.5 backtest | 2.0.0.5_backtest.json | 250 | 0.0 ± 0.0 | 100.0 ± 0.0 | 0.0 ± 0.0 | 22.0 | – | – | 97.5 | 2042.6 |
| 2.0.0.4 backtest | 2.0.0.4_backtest.json | 250 | 0.0 ± 0.0 | 100.0 ± 0.0 | 0.0 ± 0.0 | 22.0 | – | – | 90.8 | 2135.9 |
| 2.0.0.3 backtest | 2.0.0.3_backtest.json | 250 | 0.0 ± 0.0 | 100.0 ± 0.0 | 0.0 ± 0.0 | 22.0 | – | – | 87.1 | 2226.6 |
| 2.0.0.2 backtest | 2.0.0.2_backtest.json | 250 | 0.0 ± 0.0 | 100.0 ± 0.0 | 0.0 ± 0.0 | 22.0 | – | – | 100.4 | 1972.1 |
| 2.0.0.1 backtest | 2.0.0.1_backtest.json | 250 | 0.0 ± 0.0 | 100.0 ± 0.0 | 0.0 ± 0.0 | 20.0 | – | – | 90.4 | 2061.3 |
| 2.0.0.0 backtest | 2.0.0.0_backtest.json | 250 | 0.0 ± 0.0 | 100.0 ± 0.0 | 0.0 ± 0.0 | 19.0 | – | – | 74.2 | 2146.6 |

Runs that took each triforce (± binomial standard error, points):

| dungeon | 2.0.0.15 main % | 2.0.0.14 main % | 2.0.0.13 main % | 2.0.0.12 backtest % | 2.0.0.11 backtest % | 2.0.0.10 backtest % | 2.0.0.9 backtest % | 2.0.0.8 backtest % | 2.0.0.7 backtest % | 2.0.0.6 backtest % | 2.0.0.5 backtest % | 2.0.0.4 backtest % | 2.0.0.3 backtest % | 2.0.0.2 backtest % | 2.0.0.1 backtest % | 2.0.0.0 backtest % | 2.0.0.14 main − 2.0.0.15 main | 2.0.0.13 main − 2.0.0.15 main | 2.0.0.12 backtest − 2.0.0.15 main | 2.0.0.11 backtest − 2.0.0.15 main | 2.0.0.10 backtest − 2.0.0.15 main | 2.0.0.9 backtest − 2.0.0.15 main | 2.0.0.8 backtest − 2.0.0.15 main | 2.0.0.7 backtest − 2.0.0.15 main | 2.0.0.6 backtest − 2.0.0.15 main | 2.0.0.5 backtest − 2.0.0.15 main | 2.0.0.4 backtest − 2.0.0.15 main | 2.0.0.3 backtest − 2.0.0.15 main | 2.0.0.2 backtest − 2.0.0.15 main | 2.0.0.1 backtest − 2.0.0.15 main | 2.0.0.0 backtest − 2.0.0.15 main |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D3 | 41.6 ± 3.1 | 37.6 ± 3.1 | 37.6 ± 3.1 | 37.6 ± 3.1 | 37.6 ± 3.1 | 37.6 ± 3.1 | 28.4 ± 2.9 | 28.4 ± 2.9 | 20.8 ± 2.6 | 20.8 ± 2.6 | 27.2 ± 2.8 | 27.2 ± 2.8 | 27.2 ± 2.8 | 27.2 ± 2.8 | 20.4 ± 2.5 | 15.6 ± 2.3 | -4.0 ± 4.4 (z -0.9) | -4.0 ± 4.4 (z -0.9) | -4.0 ± 4.4 (z -0.9) | -4.0 ± 4.4 (z -0.9) | -4.0 ± 4.4 (z -0.9) | -13.2 ± 4.2 (z -3.1) | -13.2 ± 4.2 (z -3.1) | -20.8 ± 4.0 (z -5.2) | -20.8 ± 4.0 (z -5.2) | -14.4 ± 4.2 (z -3.4) | -14.4 ± 4.2 (z -3.4) | -14.4 ± 4.2 (z -3.4) | -14.4 ± 4.2 (z -3.4) | -21.2 ± 4.0 (z -5.3) | -26.0 ± 3.9 (z -6.7) |
| D4 | 24.4 ± 2.7 | 22.8 ± 2.7 | 22.8 ± 2.7 | 22.8 ± 2.7 | 22.8 ± 2.7 | 22.8 ± 2.7 | 13.2 ± 2.1 | 13.2 ± 2.1 | 2.4 ± 1.0 | 2.4 ± 1.0 | 2.8 ± 1.0 | 2.8 ± 1.0 | 2.8 ± 1.0 | 2.8 ± 1.0 | 3.6 ± 1.2 | 0.4 ± 0.4 | -1.6 ± 3.8 (z -0.4) | -1.6 ± 3.8 (z -0.4) | -1.6 ± 3.8 (z -0.4) | -1.6 ± 3.8 (z -0.4) | -1.6 ± 3.8 (z -0.4) | -11.2 ± 3.5 (z -3.2) | -11.2 ± 3.5 (z -3.2) | -22.0 ± 2.9 (z -7.6) | -22.0 ± 2.9 (z -7.6) | -21.6 ± 2.9 (z -7.4) | -21.6 ± 2.9 (z -7.4) | -21.6 ± 2.9 (z -7.4) | -21.6 ± 2.9 (z -7.4) | -20.8 ± 3.0 (z -7.0) | -24.0 ± 2.7 (z -8.7) |
| D1 | 23.2 ± 2.7 | 21.6 ± 2.6 | 21.6 ± 2.6 | 21.6 ± 2.6 | 21.6 ± 2.6 | 21.6 ± 2.6 | 12.8 ± 2.1 | 11.6 ± 2.0 | 2.4 ± 1.0 | 2.4 ± 1.0 | 2.8 ± 1.0 | 2.8 ± 1.0 | 2.8 ± 1.0 | 2.8 ± 1.0 | 2.0 ± 0.9 | 0.4 ± 0.4 | -1.6 ± 3.7 (z -0.4) | -1.6 ± 3.7 (z -0.4) | -1.6 ± 3.7 (z -0.4) | -1.6 ± 3.7 (z -0.4) | -1.6 ± 3.7 (z -0.4) | -10.4 ± 3.4 (z -3.1) | -11.6 ± 3.4 (z -3.5) | -20.8 ± 2.8 (z -7.3) | -20.8 ± 2.8 (z -7.3) | -20.4 ± 2.9 (z -7.1) | -20.4 ± 2.9 (z -7.1) | -20.4 ± 2.9 (z -7.1) | -20.4 ± 2.9 (z -7.1) | -21.2 ± 2.8 (z -7.5) | -22.8 ± 2.7 (z -8.4) |
| D2 | 22.8 ± 2.7 | 21.2 ± 2.6 | 21.2 ± 2.6 | 21.2 ± 2.6 | 21.2 ± 2.6 | 21.2 ± 2.6 | 12.4 ± 2.1 | 11.2 ± 2.0 | 2.4 ± 1.0 | 1.2 ± 0.7 | 2.0 ± 0.9 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.8 ± 0.6 | 1.2 ± 0.7 | 0.0 ± 0.0 | -1.6 ± 3.7 (z -0.4) | -1.6 ± 3.7 (z -0.4) | -1.6 ± 3.7 (z -0.4) | -1.6 ± 3.7 (z -0.4) | -1.6 ± 3.7 (z -0.4) | -10.4 ± 3.4 (z -3.1) | -11.6 ± 3.3 (z -3.5) | -20.4 ± 2.8 (z -7.2) | -21.6 ± 2.7 (z -7.9) | -20.8 ± 2.8 (z -7.4) | -22.8 ± 2.7 (z -8.6) | -22.8 ± 2.7 (z -8.6) | -22.0 ± 2.7 (z -8.1) | -21.6 ± 2.7 (z -7.9) | -22.8 ± 2.7 (z -8.6) |
| D5 | 11.2 ± 2.0 | 10.8 ± 2.0 | 10.8 ± 2.0 | 9.2 ± 1.8 | 9.2 ± 1.8 | 9.2 ± 1.8 | 5.2 ± 1.4 | 6.0 ± 1.5 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.4 ± 0.4 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.8 ± 0.6 | 1.2 ± 0.7 | 0.0 ± 0.0 | -0.4 ± 2.8 (z -0.1) | -0.4 ± 2.8 (z -0.1) | -2.0 ± 2.7 (z -0.7) | -2.0 ± 2.7 (z -0.7) | -2.0 ± 2.7 (z -0.7) | -6.0 ± 2.4 (z -2.5) | -5.2 ± 2.5 (z -2.1) | -11.2 ± 2.0 (z -5.6) | -11.2 ± 2.0 (z -5.6) | -10.8 ± 2.0 (z -5.3) | -11.2 ± 2.0 (z -5.6) | -11.2 ± 2.0 (z -5.6) | -10.4 ± 2.1 (z -5.0) | -10.0 ± 2.1 (z -4.7) | -11.2 ± 2.0 (z -5.6) |
| D7 | 5.2 ± 1.4 | 4.8 ± 1.4 | 6.0 ± 1.5 | 3.6 ± 1.2 | 3.6 ± 1.2 | 3.6 ± 1.2 | 2.8 ± 1.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.4 ± 0.4 | 0.0 ± 0.0 | -0.4 ± 1.9 (z -0.2) | +0.8 ± 2.1 (z +0.4) | -1.6 ± 1.8 (z -0.9) | -1.6 ± 1.8 (z -0.9) | -1.6 ± 1.8 (z -0.9) | -2.4 ± 1.7 (z -1.4) | -5.2 ± 1.4 (z -3.7) | -5.2 ± 1.4 (z -3.7) | -5.2 ± 1.4 (z -3.7) | -5.2 ± 1.4 (z -3.7) | -5.2 ± 1.4 (z -3.7) | -5.2 ± 1.4 (z -3.7) | -5.2 ± 1.4 (z -3.7) | -4.8 ± 1.5 (z -3.3) | -5.2 ± 1.4 (z -3.7) |
| D6 | 1.2 ± 0.7 | 1.2 ± 0.7 | 0.8 ± 0.6 | 0.8 ± 0.6 | 0.8 ± 0.6 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.4 ± 0.4 | 0.0 ± 0.0 | +0.0 ± 1.0 (z +0.0) | -0.4 ± 0.9 (z -0.4) | -0.4 ± 0.9 (z -0.4) | -0.4 ± 0.9 (z -0.4) | -1.2 ± 0.7 (z -1.7) | -1.2 ± 0.7 (z -1.7) | -1.2 ± 0.7 (z -1.7) | -1.2 ± 0.7 (z -1.7) | -1.2 ± 0.7 (z -1.7) | -1.2 ± 0.7 (z -1.7) | -1.2 ± 0.7 (z -1.7) | -1.2 ± 0.7 (z -1.7) | -1.2 ± 0.7 (z -1.7) | -0.8 ± 0.8 (z -1.0) | -1.2 ± 0.7 (z -1.7) |
| D8 | 1.2 ± 0.7 | 0.8 ± 0.6 | 0.8 ± 0.6 | 0.8 ± 0.6 | 0.4 ± 0.4 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | -0.4 ± 0.9 (z -0.4) | -0.4 ± 0.9 (z -0.4) | -0.4 ± 0.9 (z -0.4) | -0.8 ± 0.8 (z -1.0) | -1.2 ± 0.7 (z -1.7) | -1.2 ± 0.7 (z -1.7) | -1.2 ± 0.7 (z -1.7) | -1.2 ± 0.7 (z -1.7) | -1.2 ± 0.7 (z -1.7) | -1.2 ± 0.7 (z -1.7) | -1.2 ± 0.7 (z -1.7) | -1.2 ± 0.7 (z -1.7) | -1.2 ± 0.7 (z -1.7) | -1.2 ± 0.7 (z -1.7) | -1.2 ± 0.7 (z -1.7) |
| D9 | 0.4 ± 0.4 | 0.0 ± 0.0 | 0.4 ± 0.4 | 0.0 ± 0.0 | 0.4 ± 0.4 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | -0.4 ± 0.4 (z -1.0) | +0.0 ± 0.6 (z +0.0) | -0.4 ± 0.4 (z -1.0) | +0.0 ± 0.6 (z +0.0) | -0.4 ± 0.4 (z -1.0) | -0.4 ± 0.4 (z -1.0) | -0.4 ± 0.4 (z -1.0) | -0.4 ± 0.4 (z -1.0) | -0.4 ± 0.4 (z -1.0) | -0.4 ± 0.4 (z -1.0) | -0.4 ± 0.4 (z -1.0) | -0.4 ± 0.4 (z -1.0) | -0.4 ± 0.4 (z -1.0) | -0.4 ± 0.4 (z -1.0) | -0.4 ± 0.4 (z -1.0) |

Median frames to each triforce (the chain's count, from the game's start after the file select; a replay adds the power-on boot and the file select):

| dungeon | 2.0.0.15 main | 2.0.0.14 main | 2.0.0.13 main | 2.0.0.12 backtest | 2.0.0.11 backtest | 2.0.0.10 backtest | 2.0.0.9 backtest | 2.0.0.8 backtest | 2.0.0.7 backtest | 2.0.0.6 backtest | 2.0.0.5 backtest | 2.0.0.4 backtest | 2.0.0.3 backtest | 2.0.0.2 backtest | 2.0.0.1 backtest | 2.0.0.0 backtest |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D3 | 12208 | 12196 | 12196 | 12196 | 12196 | 12196 | 12913 | 12913 | 12847 | 12847 | 12741 | 12741 | 12741 | 12741 | 12713 | 12729 |
| D4 | 26013 | 25999 | 25999 | 25999 | 25999 | 25999 | 27578 | 27578 | 27499 | 27499 | 27026 | 27026 | 27026 | 27026 | 26725 | 26084 |
| D1 | 37060 | 37060 | 37060 | 37060 | 37060 | 37060 | 38446 | 38440 | 38526 | 38526 | 38740 | 38740 | 38740 | 38740 | 36922 | 37082 |
| D2 | 48172 | 48306 | 48306 | 48469 | 48469 | 48469 | 50250 | 50752 | 51672 | 52723 | 52452 | – | – | 72458 | 73344 | – |
| D5 | 67524 | 67935 | 67935 | 68422 | 68422 | 68422 | 68561 | 69813 | – | – | 69834 | – | – | 57880 | 57557 | – |
| D7 | 89090 | 90636 | 88915 | 89299 | 89299 | 89299 | 92157 | – | – | – | – | – | – | – | 96606 | – |
| D6 | 102670 | 112112 | 104640 | 104474 | 104622 | – | – | – | – | – | – | – | – | – | 110159 | – |
| D8 | 113679 | 133906 | 115695 | 115168 | 115445 | – | – | – | – | – | – | – | – | – | – | – |
| D9 | 140229 | – | 144591 | – | 147660 | – | – | – | – | – | – | – | – | – | – | – |

The 12 rooms that lost the most runs (a run lost there died or stuck; reached: runs that entered it at that route step):

| room (step) | family | 2.0.0.15 main: lost/reached, hits/visit, hearts/visit | 2.0.0.14 main: lost/reached, hits/visit, hearts/visit | 2.0.0.13 main: lost/reached, hits/visit, hearts/visit | 2.0.0.12 backtest: lost/reached, hits/visit, hearts/visit | 2.0.0.11 backtest: lost/reached, hits/visit, hearts/visit | 2.0.0.10 backtest: lost/reached, hits/visit, hearts/visit | 2.0.0.9 backtest: lost/reached, hits/visit, hearts/visit | 2.0.0.8 backtest: lost/reached, hits/visit, hearts/visit | 2.0.0.7 backtest: lost/reached, hits/visit, hearts/visit | 2.0.0.6 backtest: lost/reached, hits/visit, hearts/visit | 2.0.0.5 backtest: lost/reached, hits/visit, hearts/visit | 2.0.0.4 backtest: lost/reached, hits/visit, hearts/visit | 2.0.0.3 backtest: lost/reached, hits/visit, hearts/visit | 2.0.0.2 backtest: lost/reached, hits/visit, hearts/visit | 2.0.0.1 backtest: lost/reached, hits/visit, hearts/visit | 2.0.0.0 backtest: lost/reached, hits/visit, hearts/visit | lost % 2.0.0.14 main − 2.0.0.15 main | lost % 2.0.0.13 main − 2.0.0.15 main | lost % 2.0.0.12 backtest − 2.0.0.15 main | lost % 2.0.0.11 backtest − 2.0.0.15 main | lost % 2.0.0.10 backtest − 2.0.0.15 main | lost % 2.0.0.9 backtest − 2.0.0.15 main | lost % 2.0.0.8 backtest − 2.0.0.15 main | lost % 2.0.0.7 backtest − 2.0.0.15 main | lost % 2.0.0.6 backtest − 2.0.0.15 main | lost % 2.0.0.5 backtest − 2.0.0.15 main | lost % 2.0.0.4 backtest − 2.0.0.15 main | lost % 2.0.0.3 backtest − 2.0.0.15 main | lost % 2.0.0.2 backtest − 2.0.0.15 main | lost % 2.0.0.1 backtest − 2.0.0.15 main | lost % 2.0.0.0 backtest − 2.0.0.15 main |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3:0x5b (9) | darknut | 5/250 (2.0%; 5 died, 0 stuck), 0.29, 0.29 | 5/250 (2.0%; 5 died, 0 stuck), 0.29, 0.29 | 5/250 (2.0%; 5 died, 0 stuck), 0.29, 0.29 | 5/250 (2.0%; 5 died, 0 stuck), 0.29, 0.29 | 5/250 (2.0%; 5 died, 0 stuck), 0.29, 0.29 | 5/250 (2.0%; 5 died, 0 stuck), 0.29, 0.29 | 1/250 (0.4%; 1 died, 0 stuck), 0.24, 0.24 | 1/250 (0.4%; 1 died, 0 stuck), 0.24, 0.24 | 10/248 (4.0%; 10 died, 0 stuck), 0.47, 0.46 | 10/248 (4.0%; 10 died, 0 stuck), 0.47, 0.46 | 10/248 (4.0%; 10 died, 0 stuck), 0.47, 0.46 | 10/248 (4.0%; 10 died, 0 stuck), 0.47, 0.46 | 10/248 (4.0%; 10 died, 0 stuck), 0.47, 0.46 | 10/248 (4.0%; 10 died, 0 stuck), 0.47, 0.46 | 10/248 (4.0%; 10 died, 0 stuck), 0.53, 0.53 | 14/247 (5.7%; 14 died, 0 stuck), 0.45, 0.44 | +0.0 ± 1.3 (z +0.0) | +0.0 ± 1.3 (z +0.0) | +0.0 ± 1.3 (z +0.0) | +0.0 ± 1.3 (z +0.0) | +0.0 ± 1.3 (z +0.0) | -1.6 ± 1.0 (z -1.6) | -1.6 ± 1.0 (z -1.6) | +2.0 ± 1.5 (z +1.3) | +2.0 ± 1.5 (z +1.3) | +2.0 ± 1.5 (z +1.3) | +2.0 ± 1.5 (z +1.3) | +2.0 ± 1.5 (z +1.3) | +2.0 ± 1.5 (z +1.3) | +2.0 ± 1.5 (z +1.3) | +3.7 ± 1.7 (z +2.1) |
| 3:0x49 (12) | zol | 1/243 (0.4%; 1 died, 0 stuck), 0.14, 0.08 | 1/243 (0.4%; 1 died, 0 stuck), 0.14, 0.08 | 1/243 (0.4%; 1 died, 0 stuck), 0.14, 0.08 | 1/243 (0.4%; 1 died, 0 stuck), 0.14, 0.08 | 1/243 (0.4%; 1 died, 0 stuck), 0.14, 0.08 | 1/243 (0.4%; 1 died, 0 stuck), 0.14, 0.08 | 2/249 (0.8%; 2 died, 0 stuck), 0.10, 0.06 | 2/249 (0.8%; 2 died, 0 stuck), 0.10, 0.06 | 10/233 (4.3%; 10 died, 0 stuck), 0.20, 0.16 | 10/233 (4.3%; 10 died, 0 stuck), 0.20, 0.16 | 10/233 (4.3%; 10 died, 0 stuck), 0.20, 0.16 | 10/233 (4.3%; 10 died, 0 stuck), 0.20, 0.16 | 10/233 (4.3%; 10 died, 0 stuck), 0.20, 0.16 | 10/233 (4.3%; 10 died, 0 stuck), 0.20, 0.16 | 12/231 (5.2%; 12 died, 0 stuck), 0.26, 0.20 | 19/219 (8.7%; 19 died, 0 stuck), 0.41, 0.28 | +0.0 ± 0.6 (z +0.0) | +0.0 ± 0.6 (z +0.0) | +0.0 ± 0.6 (z +0.0) | +0.0 ± 0.6 (z +0.0) | +0.0 ± 0.6 (z +0.0) | +0.4 ± 0.7 (z +0.6) | +0.4 ± 0.7 (z +0.6) | +3.9 ± 1.4 (z +2.8) | +3.9 ± 1.4 (z +2.8) | +3.9 ± 1.4 (z +2.8) | +3.9 ± 1.4 (z +2.8) | +3.9 ± 1.4 (z +2.8) | +3.9 ± 1.4 (z +2.8) | +4.8 ± 1.5 (z +3.2) | +8.3 ± 1.9 (z +4.2) |
| 3:0x59 (13) | darknut | 3/242 (1.2%; 3 died, 0 stuck), 0.14, 0.14 | 3/242 (1.2%; 3 died, 0 stuck), 0.14, 0.14 | 3/242 (1.2%; 3 died, 0 stuck), 0.14, 0.14 | 3/242 (1.2%; 3 died, 0 stuck), 0.14, 0.14 | 3/242 (1.2%; 3 died, 0 stuck), 0.14, 0.14 | 3/242 (1.2%; 3 died, 0 stuck), 0.14, 0.14 | 3/247 (1.2%; 3 died, 0 stuck), 0.20, 0.19 | 3/247 (1.2%; 3 died, 0 stuck), 0.20, 0.19 | 20/223 (9.0%; 20 died, 0 stuck), 0.39, 0.35 | 20/223 (9.0%; 20 died, 0 stuck), 0.39, 0.35 | 20/223 (9.0%; 20 died, 0 stuck), 0.39, 0.35 | 20/223 (9.0%; 20 died, 0 stuck), 0.39, 0.35 | 20/223 (9.0%; 20 died, 0 stuck), 0.39, 0.35 | 20/223 (9.0%; 20 died, 0 stuck), 0.39, 0.35 | 14/219 (6.4%; 14 died, 0 stuck), 0.31, 0.28 | 16/200 (8.0%; 16 died, 0 stuck), 0.35, 0.33 | +0.0 ± 1.0 (z +0.0) | +0.0 ± 1.0 (z +0.0) | +0.0 ± 1.0 (z +0.0) | +0.0 ± 1.0 (z +0.0) | +0.0 ± 1.0 (z +0.0) | -0.0 ± 1.0 (z -0.0) | -0.0 ± 1.0 (z -0.0) | +7.7 ± 2.0 (z +3.8) | +7.7 ± 2.0 (z +3.8) | +7.7 ± 2.0 (z +3.8) | +7.7 ± 2.0 (z +3.8) | +7.7 ± 2.0 (z +3.8) | +7.7 ± 2.0 (z +3.8) | +5.2 ± 1.8 (z +2.9) | +6.8 ± 2.0 (z +3.3) |
| 3:0x69 (14) | darknut | 40/239 (16.7%; 40 died, 0 stuck), 0.93, 0.91 | 40/239 (16.7%; 40 died, 0 stuck), 0.93, 0.91 | 40/239 (16.7%; 40 died, 0 stuck), 0.93, 0.91 | 40/239 (16.7%; 40 died, 0 stuck), 0.93, 0.91 | 40/239 (16.7%; 40 died, 0 stuck), 0.93, 0.91 | 40/239 (16.7%; 40 died, 0 stuck), 0.93, 0.91 | 39/244 (16.0%; 39 died, 0 stuck), 0.95, 0.91 | 39/244 (16.0%; 39 died, 0 stuck), 0.95, 0.91 | 24/203 (11.8%; 24 died, 0 stuck), 0.65, 0.62 | 24/203 (11.8%; 24 died, 0 stuck), 0.65, 0.62 | 11/203 (5.4%; 11 died, 0 stuck), 0.46, 0.44 | 11/203 (5.4%; 11 died, 0 stuck), 0.46, 0.44 | 11/203 (5.4%; 11 died, 0 stuck), 0.46, 0.44 | 11/203 (5.4%; 11 died, 0 stuck), 0.46, 0.44 | 18/205 (8.8%; 18 died, 0 stuck), 0.61, 0.59 | 21/184 (11.4%; 21 died, 0 stuck), 0.59, 0.57 | +0.0 ± 3.4 (z +0.0) | +0.0 ± 3.4 (z +0.0) | +0.0 ± 3.4 (z +0.0) | +0.0 ± 3.4 (z +0.0) | +0.0 ± 3.4 (z +0.0) | -0.8 ± 3.4 (z -0.2) | -0.8 ± 3.4 (z -0.2) | -4.9 ± 3.3 (z -1.5) | -4.9 ± 3.3 (z -1.5) | -11.3 ± 2.9 (z -3.9) | -11.3 ± 2.9 (z -3.9) | -11.3 ± 2.9 (z -3.9) | -11.3 ± 2.9 (z -3.9) | -8.0 ± 3.1 (z -2.5) | -5.3 ± 3.4 (z -1.6) |
| 3:0x0f:cave (15) | equipment-room | 15/199 (7.5%; 15 died, 0 stuck), 0.45, 0.23 | 15/199 (7.5%; 15 died, 0 stuck), 0.45, 0.23 | 15/199 (7.5%; 15 died, 0 stuck), 0.45, 0.23 | 15/199 (7.5%; 15 died, 0 stuck), 0.45, 0.23 | 15/199 (7.5%; 15 died, 0 stuck), 0.45, 0.23 | 15/199 (7.5%; 15 died, 0 stuck), 0.45, 0.23 | 12/205 (5.9%; 12 died, 0 stuck), 0.47, 0.23 | 12/205 (5.9%; 12 died, 0 stuck), 0.47, 0.23 | 10/179 (5.6%; 10 died, 0 stuck), 0.53, 0.27 | 10/179 (5.6%; 10 died, 0 stuck), 0.53, 0.27 | 13/192 (6.8%; 13 died, 0 stuck), 0.46, 0.23 | 13/192 (6.8%; 13 died, 0 stuck), 0.46, 0.23 | 13/192 (6.8%; 13 died, 0 stuck), 0.46, 0.23 | 13/192 (6.8%; 13 died, 0 stuck), 0.46, 0.23 | 13/187 (7.0%; 13 died, 0 stuck), 0.53, 0.26 | 17/163 (10.4%; 17 died, 0 stuck), 0.61, 0.31 | +0.0 ± 2.6 (z +0.0) | +0.0 ± 2.6 (z +0.0) | +0.0 ± 2.6 (z +0.0) | +0.0 ± 2.6 (z +0.0) | +0.0 ± 2.6 (z +0.0) | -1.7 ± 2.5 (z -0.7) | -1.7 ± 2.5 (z -0.7) | -2.0 ± 2.5 (z -0.8) | -2.0 ± 2.5 (z -0.8) | -0.8 ± 2.6 (z -0.3) | -0.8 ± 2.6 (z -0.3) | -0.8 ± 2.6 (z -0.3) | -0.8 ± 2.6 (z -0.3) | -0.6 ± 2.6 (z -0.2) | +2.9 ± 3.0 (z +1.0) |
| 3:0x5a (18) | trap-room | 9/184 (4.9%; 9 died, 0 stuck), 0.20, 0.10 | 9/184 (4.9%; 9 died, 0 stuck), 0.20, 0.10 | 9/184 (4.9%; 9 died, 0 stuck), 0.20, 0.10 | 9/184 (4.9%; 9 died, 0 stuck), 0.20, 0.10 | 9/184 (4.9%; 9 died, 0 stuck), 0.20, 0.10 | 9/184 (4.9%; 9 died, 0 stuck), 0.20, 0.10 | 14/191 (7.3%; 14 died, 0 stuck), 0.25, 0.13 | 14/191 (7.3%; 14 died, 0 stuck), 0.25, 0.13 | 5/169 (3.0%; 5 died, 0 stuck), 0.21, 0.11 | 5/169 (3.0%; 5 died, 0 stuck), 0.21, 0.11 | 8/179 (4.5%; 8 died, 0 stuck), 0.22, 0.11 | 8/179 (4.5%; 8 died, 0 stuck), 0.22, 0.11 | 8/179 (4.5%; 8 died, 0 stuck), 0.22, 0.11 | 8/179 (4.5%; 8 died, 0 stuck), 0.22, 0.11 | 16/173 (9.2%; 16 died, 0 stuck), 0.24, 0.12 | 9/146 (6.2%; 9 died, 0 stuck), 0.21, 0.11 | +0.0 ± 2.2 (z +0.0) | +0.0 ± 2.2 (z +0.0) | +0.0 ± 2.2 (z +0.0) | +0.0 ± 2.2 (z +0.0) | +0.0 ± 2.2 (z +0.0) | +2.4 ± 2.5 (z +1.0) | +2.4 ± 2.5 (z +1.0) | -1.9 ± 2.1 (z -0.9) | -1.9 ± 2.1 (z -0.9) | -0.4 ± 2.2 (z -0.2) | -0.4 ± 2.2 (z -0.2) | -0.4 ± 2.2 (z -0.2) | -0.4 ± 2.2 (z -0.2) | +4.4 ± 2.7 (z +1.6) | +1.3 ± 2.5 (z +0.5) |
| 3:0x5b (19) | darknut | 10/175 (5.7%; 9 died, 1 stuck), 0.32, 0.30 | 10/175 (5.7%; 9 died, 1 stuck), 0.32, 0.30 | 10/175 (5.7%; 9 died, 1 stuck), 0.32, 0.30 | 10/175 (5.7%; 9 died, 1 stuck), 0.32, 0.30 | 10/175 (5.7%; 9 died, 1 stuck), 0.32, 0.30 | 10/175 (5.7%; 9 died, 1 stuck), 0.32, 0.30 | 13/177 (7.3%; 13 died, 0 stuck), 0.37, 0.36 | 13/177 (7.3%; 13 died, 0 stuck), 0.37, 0.36 | 23/164 (14.0%; 22 died, 1 stuck), 0.55, 0.53 | 23/164 (14.0%; 22 died, 1 stuck), 0.55, 0.53 | 14/171 (8.2%; 14 died, 0 stuck), 0.49, 0.46 | 14/171 (8.2%; 14 died, 0 stuck), 0.49, 0.46 | 14/171 (8.2%; 14 died, 0 stuck), 0.49, 0.46 | 14/171 (8.2%; 14 died, 0 stuck), 0.49, 0.46 | 29/157 (18.5%; 29 died, 0 stuck), 0.65, 0.61 | 33/137 (24.1%; 33 died, 0 stuck), 0.80, 0.75 | +0.0 ± 2.5 (z +0.0) | +0.0 ± 2.5 (z +0.0) | +0.0 ± 2.5 (z +0.0) | +0.0 ± 2.5 (z +0.0) | +0.0 ± 2.5 (z +0.0) | +1.6 ± 2.6 (z +0.6) | +1.6 ± 2.6 (z +0.6) | +8.3 ± 3.2 (z +2.6) | +8.3 ± 3.2 (z +2.6) | +2.5 ± 2.7 (z +0.9) | +2.5 ± 2.7 (z +0.9) | +2.5 ± 2.7 (z +0.9) | +2.5 ± 2.7 (z +0.9) | +12.8 ± 3.6 (z +3.6) | +18.4 ± 4.1 (z +4.5) |
| 3:0x5c (20) | darknut | 7/164 (4.3%; 7 died, 0 stuck), 0.12, 0.11 | 7/164 (4.3%; 7 died, 0 stuck), 0.12, 0.11 | 7/164 (4.3%; 7 died, 0 stuck), 0.12, 0.11 | 7/164 (4.3%; 7 died, 0 stuck), 0.12, 0.11 | 7/164 (4.3%; 7 died, 0 stuck), 0.12, 0.11 | 7/164 (4.3%; 7 died, 0 stuck), 0.12, 0.11 | 3/164 (1.8%; 3 died, 0 stuck), 0.07, 0.07 | 3/164 (1.8%; 3 died, 0 stuck), 0.07, 0.07 | 13/141 (9.2%; 13 died, 0 stuck), 0.32, 0.30 | 13/141 (9.2%; 13 died, 0 stuck), 0.32, 0.30 | 11/157 (7.0%; 11 died, 0 stuck), 0.31, 0.30 | 11/157 (7.0%; 11 died, 0 stuck), 0.31, 0.30 | 11/157 (7.0%; 11 died, 0 stuck), 0.31, 0.30 | 11/157 (7.0%; 11 died, 0 stuck), 0.31, 0.30 | 11/128 (8.6%; 11 died, 0 stuck), 0.29, 0.27 | 10/104 (9.6%; 10 died, 0 stuck), 0.28, 0.25 | +0.0 ± 2.2 (z +0.0) | +0.0 ± 2.2 (z +0.0) | +0.0 ± 2.2 (z +0.0) | +0.0 ± 2.2 (z +0.0) | +0.0 ± 2.2 (z +0.0) | -2.4 ± 1.9 (z -1.3) | -2.4 ± 1.9 (z -1.3) | +5.0 ± 2.9 (z +1.7) | +5.0 ± 2.9 (z +1.7) | +2.7 ± 2.6 (z +1.1) | +2.7 ± 2.6 (z +1.1) | +2.7 ± 2.6 (z +1.1) | +2.7 ± 2.6 (z +1.1) | +4.3 ± 2.9 (z +1.5) | +5.3 ± 3.3 (z +1.6) |
| 3:0x4c (21) | trap-room | 3/157 (1.9%; 3 died, 0 stuck), 0.20, 0.11 | 3/157 (1.9%; 3 died, 0 stuck), 0.20, 0.11 | 3/157 (1.9%; 3 died, 0 stuck), 0.20, 0.11 | 3/157 (1.9%; 3 died, 0 stuck), 0.20, 0.11 | 3/157 (1.9%; 3 died, 0 stuck), 0.20, 0.11 | 3/157 (1.9%; 3 died, 0 stuck), 0.20, 0.11 | 8/161 (5.0%; 8 died, 0 stuck), 0.21, 0.12 | 8/161 (5.0%; 8 died, 0 stuck), 0.21, 0.12 | 10/128 (7.8%; 10 died, 0 stuck), 0.23, 0.15 | 10/128 (7.8%; 10 died, 0 stuck), 0.23, 0.15 | 9/146 (6.2%; 9 died, 0 stuck), 0.29, 0.17 | 9/146 (6.2%; 9 died, 0 stuck), 0.29, 0.17 | 9/146 (6.2%; 9 died, 0 stuck), 0.29, 0.17 | 9/146 (6.2%; 9 died, 0 stuck), 0.29, 0.17 | 9/117 (7.7%; 9 died, 0 stuck), 0.26, 0.14 | 4/94 (4.3%; 4 died, 0 stuck), 0.15, 0.10 | +0.0 ± 1.5 (z +0.0) | +0.0 ± 1.5 (z +0.0) | +0.0 ± 1.5 (z +0.0) | +0.0 ± 1.5 (z +0.0) | +0.0 ± 1.5 (z +0.0) | +3.1 ± 2.0 (z +1.5) | +3.1 ± 2.0 (z +1.5) | +5.9 ± 2.6 (z +2.3) | +5.9 ± 2.6 (z +2.3) | +4.3 ± 2.3 (z +1.9) | +4.3 ± 2.3 (z +1.9) | +4.3 ± 2.3 (z +1.9) | +4.3 ± 2.3 (z +1.9) | +5.8 ± 2.7 (z +2.1) | +2.3 ± 2.4 (z +1.0) |
| 3:0x4d (22) | boss-manhandla | 50/154 (32.5%; 50 died, 0 stuck), 0.92, 0.54 | 60/154 (39.0%; 60 died, 0 stuck), 1.01, 0.63 | 60/154 (39.0%; 60 died, 0 stuck), 1.01, 0.63 | 60/154 (39.0%; 60 died, 0 stuck), 1.01, 0.63 | 60/154 (39.0%; 60 died, 0 stuck), 1.01, 0.63 | 60/154 (39.0%; 60 died, 0 stuck), 1.01, 0.63 | 82/153 (53.6%; 82 died, 0 stuck), 1.16, 0.84 | 82/153 (53.6%; 82 died, 0 stuck), 1.16, 0.84 | 66/118 (55.9%; 66 died, 0 stuck), 1.10, 0.72 | 66/118 (55.9%; 66 died, 0 stuck), 1.10, 0.72 | 69/137 (50.4%; 69 died, 0 stuck), 1.12, 0.76 | 69/137 (50.4%; 69 died, 0 stuck), 1.12, 0.76 | 69/137 (50.4%; 69 died, 0 stuck), 1.12, 0.76 | 69/137 (50.4%; 69 died, 0 stuck), 1.12, 0.76 | 57/108 (52.8%; 57 died, 0 stuck), 1.13, 0.74 | 51/90 (56.7%; 51 died, 0 stuck), 1.16, 0.79 | +6.5 ± 5.4 (z +1.2) | +6.5 ± 5.4 (z +1.2) | +6.5 ± 5.4 (z +1.2) | +6.5 ± 5.4 (z +1.2) | +6.5 ± 5.4 (z +1.2) | +21.1 ± 5.5 (z +3.8) | +21.1 ± 5.5 (z +3.8) | +23.5 ± 5.9 (z +4.0) | +23.5 ± 5.9 (z +4.0) | +17.9 ± 5.7 (z +3.1) | +17.9 ± 5.7 (z +3.1) | +17.9 ± 5.7 (z +3.1) | +17.9 ± 5.7 (z +3.1) | +20.3 ± 6.1 (z +3.3) | +24.2 ± 6.4 (z +3.8) |
| 4:0x13 (51) | boss-gleeok | 35/96 (36.5%; 35 died, 0 stuck), 1.69, 0.89 | 30/87 (34.5%; 30 died, 0 stuck), 1.74, 0.87 | 30/87 (34.5%; 30 died, 0 stuck), 1.74, 0.87 | 30/87 (34.5%; 30 died, 0 stuck), 1.74, 0.87 | 30/87 (34.5%; 30 died, 0 stuck), 1.74, 0.87 | 30/87 (34.5%; 30 died, 0 stuck), 1.74, 0.87 | 28/61 (45.9%; 28 died, 0 stuck), 1.74, 1.02 | 28/61 (45.9%; 28 died, 0 stuck), 1.74, 1.02 | 25/31 (80.6%; 25 died, 0 stuck), 2.03, 1.39 | 25/31 (80.6%; 25 died, 0 stuck), 2.03, 1.39 | 30/37 (81.1%; 30 died, 0 stuck), 1.76, 1.32 | 30/37 (81.1%; 30 died, 0 stuck), 1.76, 1.32 | 30/37 (81.1%; 30 died, 0 stuck), 1.76, 1.32 | 30/37 (81.1%; 30 died, 0 stuck), 1.76, 1.32 | 16/25 (64.0%; 16 died, 0 stuck), 1.92, 1.20 | 13/14 (92.9%; 13 died, 0 stuck), 1.86, 1.50 | -2.0 ± 7.1 (z -0.3) | -2.0 ± 7.1 (z -0.3) | -2.0 ± 7.1 (z -0.3) | -2.0 ± 7.1 (z -0.3) | -2.0 ± 7.1 (z -0.3) | +9.4 ± 8.1 (z +1.2) | +9.4 ± 8.1 (z +1.2) | +44.2 ± 8.6 (z +5.1) | +44.2 ± 8.6 (z +5.1) | +44.6 ± 8.1 (z +5.5) | +44.6 ± 8.1 (z +5.5) | +44.6 ± 8.1 (z +5.5) | +44.6 ± 8.1 (z +5.5) | +27.5 ± 10.8 (z +2.6) | +56.4 ± 8.5 (z +6.7) |
| 5:0x05 (128) | darknut | 14/57 (24.6%; 14 died, 0 stuck), 1.95, 3.04 | 12/53 (22.6%; 12 died, 0 stuck), 1.96, 3.06 | 12/53 (22.6%; 12 died, 0 stuck), 1.96, 3.06 | 14/53 (26.4%; 14 died, 0 stuck), 2.26, 3.67 | 14/53 (26.4%; 14 died, 0 stuck), 2.26, 3.67 | 14/53 (26.4%; 14 died, 0 stuck), 2.26, 3.67 | 10/30 (33.3%; 10 died, 0 stuck), 2.17, 3.62 | 6/28 (21.4%; 6 died, 0 stuck), 2.29, 3.39 | 4/6 (66.7%; 4 died, 0 stuck), 3.17, 6.00 | not reached | not reached | not reached | not reached | not reached | not reached | not reached | -1.9 ± 8.1 (z -0.2) | -1.9 ± 8.1 (z -0.2) | +1.9 ± 8.3 (z +0.2) | +1.9 ± 8.3 (z +0.2) | +1.9 ± 8.3 (z +0.2) | +8.8 ± 10.3 (z +0.8) | -3.1 ± 9.6 (z -0.3) | +42.1 ± 20.1 (z +2.1) | – | – | – | – | – | – | – |

- 2.0.0.15 main kept: seed 106 best (step 290, 140229 frames, 1 bombs/keys and 0 rupees cheated: the game ended in (level 9, room 0x32)); seed 172 worst (step 9, 3819 frames, 0 bombs/keys and 0 rupees cheated: died in (level 3, room 0x5b)); seed 38 stuck-7-0x0d (step 173, 134607 frames, 6 bombs/keys and 0 rupees cheated: wallmaster@d3first exceeded 2000 steps); seed 108 stuck-0-0x45 (step 147, 71554 frames, 4 bombs/keys and 0 rupees cheated: bounce-limit: ow-peahat left into (level 0, room 0x55), the 5th return in a row into the room before the last (at most 4 in a row)); seed 115 stuck-4-0x61 (step 146, 76459 frames, 2 bombs/keys and 0 rupees cheated: vire left the route into untagged (level 4, room 0x62)).
- 2.0.0.14 main kept: seed 89 best (step 264, 129615 frames, 2 bombs/keys and 0 rupees cheated: died in (level 9, room 0x05)); seed 172 worst (step 9, 3819 frames, 0 bombs/keys and 0 rupees cheated: died in (level 3, room 0x5b)); seed 108 stuck-0-0x45 (step 147, 71554 frames, 4 bombs/keys and 0 rupees cheated: bounce-limit: ow-peahat left into (level 0, room 0x55), the 5th return in a row into the room before the last (at most 4 in a row)); seed 115 stuck-4-0x61 (step 146, 76459 frames, 2 bombs/keys and 0 rupees cheated: vire left the route into untagged (level 4, room 0x62)); seed 139 stuck-3-0x5b (step 19, 46858 frames, 0 bombs/keys and 0 rupees cheated: darknut@d3first exceeded 2000 steps).
- 2.0.0.13 main kept: seed 67 best (step 290, 144591 frames, 3 bombs/keys and 0 rupees cheated: the game ended in (level 9, room 0x32)); seed 172 worst (step 9, 3819 frames, 0 bombs/keys and 0 rupees cheated: died in (level 3, room 0x5b)); seed 121 stuck-4-0x61 (step 146, 81231 frames, 1 bombs/keys and 0 rupees cheated: vire left the route into untagged (level 4, room 0x62)); seed 139 stuck-3-0x5b (step 19, 46858 frames, 0 bombs/keys and 0 rupees cheated: darknut@d3first exceeded 2000 steps); seed 140 stuck-0-0x1b (step 145, 106478 frames, 3 bombs/keys and 0 rupees cheated: bounce-limit: ow-start left into (level 0, room 0x0b), the 5th return in a row into the room before the last (at most 4 in a row)).
- 2.0.0.12 backtest kept: seed 238 best (step 284, 142899 frames, 4 bombs/keys and 0 rupees cheated: died in (level 9, room 0x04)); seed 172 worst (step 9, 3819 frames, 0 bombs/keys and 0 rupees cheated: died in (level 3, room 0x5b)); seed 49 stuck-0-0x45 (step 147, 70966 frames, 4 bombs/keys and 0 rupees cheated: bounce-limit: ow-peahat left into (level 0, room 0x55), the 5th return in a row into the room before the last (at most 4 in a row)); seed 0 stuck-0-0x55 (step 147, 69974 frames, 2 bombs/keys and 0 rupees cheated: bounce-limit: ow-zora left into (level 0, room 0x45), the 5th return in a row into the room before the last (at most 4 in a row)); seed 6 stuck-0-0x1b (step 145, 72778 frames, 1 bombs/keys and 0 rupees cheated: bounce-limit: ow-start left into (level 0, room 0x0b), the 5th return in a row into the room before the last (at most 4 in a row)).
- 2.0.0.11 backtest kept: seed 21 best (step 290, 147660 frames, 6 bombs/keys and 0 rupees cheated: the game ended in (level 9, room 0x32)); seed 172 worst (step 9, 3819 frames, 0 bombs/keys and 0 rupees cheated: died in (level 3, room 0x5b)); seed 49 stuck-0-0x45 (step 147, 70966 frames, 4 bombs/keys and 0 rupees cheated: bounce-limit: ow-peahat left into (level 0, room 0x55), the 5th return in a row into the room before the last (at most 4 in a row)); seed 0 stuck-0-0x55 (step 147, 69974 frames, 2 bombs/keys and 0 rupees cheated: bounce-limit: ow-zora left into (level 0, room 0x45), the 5th return in a row into the room before the last (at most 4 in a row)); seed 6 stuck-0-0x1b (step 145, 72778 frames, 1 bombs/keys and 0 rupees cheated: bounce-limit: ow-start left into (level 0, room 0x0b), the 5th return in a row into the room before the last (at most 4 in a row)).
- 2.0.0.10 backtest kept: seed 200 best (step 206, 112013 frames, 0 bombs/keys and 0 rupees cheated: died in (level 6, room 0x3a)); seed 172 worst (step 9, 3819 frames, 0 bombs/keys and 0 rupees cheated: died in (level 3, room 0x5b)); seed 21 stuck-0-0x40 (step 181, 91029 frames, 5 bombs/keys and 0 rupees cheated: bounce-limit: ow-ghini left into (level 0, room 0x50), the 5th return in a row into the room before the last (at most 4 in a row)); seed 49 stuck-0-0x45 (step 147, 70966 frames, 4 bombs/keys and 0 rupees cheated: bounce-limit: ow-peahat left into (level 0, room 0x55), the 5th return in a row into the room before the last (at most 4 in a row)); seed 0 stuck-0-0x55 (step 147, 69974 frames, 2 bombs/keys and 0 rupees cheated: bounce-limit: ow-zora left into (level 0, room 0x45), the 5th return in a row into the room before the last (at most 4 in a row)).
- 2.0.0.9 backtest kept: seed 162 best (step 206, 103173 frames, 7 bombs/keys and 0 rupees cheated: died in (level 6, room 0x3a)); seed 122 worst (step 9, 4699 frames, 0 bombs/keys and 0 rupees cheated: died in (level 3, room 0x5b)); seed 2 stuck-0-0x40 (step 181, 98801 frames, 2 bombs/keys and 0 rupees cheated: bounce-limit: ow-ghini left into (level 0, room 0x50), the 5th return in a row into the room before the last (at most 4 in a row)); seed 118 stuck-7-0x28 (step 164, 94759 frames, 0 bombs/keys and 0 rupees cheated: goriya@d3first exceeded 2000 steps); seed 156 stuck-5-0x65 (step 145, 72365 frames, 1 bombs/keys and 0 rupees cheated: bounce-limit: gibdo left into (level 5, room 0x66), the 5th return in a row into the room before the last (at most 4 in a row)).
- 2.0.0.8 backtest kept: seed 51 best (step 173, 96772 frames, 4 bombs/keys and 0 rupees cheated: died in (level 7, room 0x18)); seed 122 worst (step 9, 4699 frames, 0 bombs/keys and 0 rupees cheated: died in (level 3, room 0x5b)); seed 84 stuck-5-0x37 (step 139, 78655 frames, 2 bombs/keys and 0 rupees cheated: darknut@d3first exceeded 2000 steps); seed 111 stuck-0-0x45 (step 147, 70335 frames, 2 bombs/keys and 0 rupees cheated: bounce-limit: ow-peahat left into (level 0, room 0x55), the 5th return in a row into the room before the last (at most 4 in a row)); seed 238 stuck-0-0x1b (step 145, 73336 frames, 1 bombs/keys and 0 rupees cheated: bounce-limit: ow-start left into (level 0, room 0x0b), the 5th return in a row into the room before the last (at most 4 in a row)).
- 2.0.0.7 backtest kept: seed 50 best (step 143, 66944 frames, 3 bombs/keys and 0 rupees cheated: died in (level 5, room 0x24)); seed 53 worst (step 7, 4139 frames, 0 bombs/keys and 0 rupees cheated: died in (level 3, room 0x7b)); seed 117 stuck-4-0x31 (step 37, 35834 frames, 1 bombs/keys and 0 rupees cheated: vire@d3first exceeded 2000 steps); seed 248 stuck-3-0x5b (step 19, 18984 frames, 0 bombs/keys and 0 rupees cheated: darknut@d3first exceeded 2000 steps).
- 2.0.0.6 backtest kept: seed 210 best (step 145, 70812 frames, 5 bombs/keys and 0 rupees cheated: died in (level 5, room 0x24)); seed 53 worst (step 7, 4139 frames, 0 bombs/keys and 0 rupees cheated: died in (level 3, room 0x7b)); seed 117 stuck-4-0x31 (step 37, 35834 frames, 1 bombs/keys and 0 rupees cheated: vire@d3first exceeded 2000 steps); seed 236 stuck-2-0x5e (step 100, 58562 frames, 0 bombs/keys and 0 rupees cheated: goriya@d3first exceeded 2000 steps); seed 248 stuck-3-0x5b (step 19, 18984 frames, 0 bombs/keys and 0 rupees cheated: darknut@d3first exceeded 2000 steps).
- 2.0.0.5 backtest kept: seed 222 best (step 164, 77914 frames, 2 bombs/keys and 0 rupees cheated: died in (level 7, room 0x39)); seed 53 worst (step 7, 4139 frames, 0 bombs/keys and 0 rupees cheated: died in (level 3, room 0x7b)).
- 2.0.0.4 backtest kept: seed 4 best (step 110, 52714 frames, 1 bombs/keys and 15 rupees cheated: died in (level 5, room 0x66)); seed 53 worst (step 7, 4139 frames, 0 bombs/keys and 0 rupees cheated: died in (level 3, room 0x7b)).
- 2.0.0.3 backtest kept: seed 4 best (step 110, 52714 frames, 1 bombs/keys and 15 rupees cheated: died in (level 5, room 0x66)); seed 53 worst (step 7, 4139 frames, 0 bombs/keys and 0 rupees cheated: died in (level 3, room 0x7b)).
- 2.0.0.2 backtest kept: seed 237 best (step 187, 86857 frames, 12 bombs/keys and 5 rupees cheated: died in (level 7, room 0x1c)); seed 53 worst (step 7, 4139 frames, 0 bombs/keys and 0 rupees cheated: died in (level 3, room 0x7b)).
- 2.0.0.1 backtest kept: seed 30 best (step 238, 116604 frames, 11 bombs/keys and 12 rupees cheated: died in (level 8, room 0x3f)); seed 53 worst (step 7, 4139 frames, 0 bombs/keys and 0 rupees cheated: died in (level 3, room 0x7b)).
- 2.0.0.0 backtest kept: seed 76 best (step 103, 45984 frames, 8 bombs/keys and 13 rupees cheated: died in (level 5, room 0x05)); seed 21 worst (step 8, 3682 frames, 0 bombs/keys and 0 rupees cheated: died in (level 3, room 0x6b)).

#### 2.0.0.12 backtest

Bomb metrics over 230 runs (the owner, Nov 23; 1478 windows):

| Metric A: the 10th-kill bomb | all windows | without a full bag |
|---|---|---|
| opportunities | 318 | 306 |
| armed | 146 (45.9%) | 145 (47.4%) |
| missed | 42 (13.2%) | 38 (12.4%) |
| other | 130 (40.9%) | 123 (40.2%) |

Not counted: 29 lost, 1 paused-out, 6 suspended, 4 open-at-end.

| Metric B: the bomb carrier | all windows | without a full bag |
|---|---|---|
| opportunities | 744 | 744 |
| hit | 18 (2.4%) | 18 (2.4%) |
| miss | 55 (7.4%) | 55 (7.4%) |
| left | 671 (90.2%) | 671 (90.2%) |

Not counted: 343 closed, 33 open-at-end.

Forced bomb drops after `armed`: {'pending': 0, 'fell': 2, 'taken': 98, 'wasted': 4, 'lost': 42}. Bombs the carrier hits dropped: {'none': 13, 'fell': 0, 'taken': 4, 'wasted': 1}.

Rooms with the most opportunities (top 12; the room each window closed in):

| room | A opp | armed | missed | other | A full | B opp | hit | miss | left |
|---|---|---|---|---|---|---|---|---|---|
| 3:0x69 | 6 | 6 | 0 | 0 | 0 | 144 | 0 | 0 | 144 |
| 3:0x5b | 29 | 15 | 11 | 3 | 2 | 97 | 1 | 0 | 96 |
| 3:0x4d | 86 | 83 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3:0x5c | 0 | 0 | 0 | 0 | 0 | 59 | 0 | 0 | 59 |
| 0:0x4c | 2 | 0 | 0 | 2 | 0 | 52 | 4 | 25 | 23 |
| 4:0x31 | 49 | 16 | 8 | 25 | 3 | 0 | 0 | 0 | 0 |
| 0:0x38 | 0 | 0 | 0 | 0 | 0 | 43 | 0 | 16 | 27 |
| 0:0x3d | 0 | 0 | 0 | 0 | 0 | 37 | 0 | 0 | 37 |
| 5:0x66 | 8 | 2 | 4 | 2 | 1 | 22 | 1 | 6 | 15 |
| 0:0x3c | 0 | 0 | 0 | 0 | 0 | 26 | 0 | 0 | 26 |
| 0:0x63 | 0 | 0 | 0 | 0 | 0 | 25 | 0 | 0 | 25 |
| 0:0x64 | 0 | 0 | 0 | 0 | 0 | 25 | 0 | 0 | 25 |


#### 2.0.0.11 backtest

Bomb metrics over 230 runs (the owner, Nov 23; 1477 windows):

| Metric A: the 10th-kill bomb | all windows | without a full bag |
|---|---|---|
| opportunities | 317 | 305 |
| armed | 146 (46.1%) | 145 (47.5%) |
| missed | 42 (13.2%) | 38 (12.5%) |
| other | 129 (40.7%) | 122 (40.0%) |

Not counted: 28 lost, 1 paused-out, 6 suspended, 4 open-at-end.

| Metric B: the bomb carrier | all windows | without a full bag |
|---|---|---|
| opportunities | 745 | 745 |
| hit | 20 (2.7%) | 20 (2.7%) |
| miss | 53 (7.1%) | 53 (7.1%) |
| left | 672 (90.2%) | 672 (90.2%) |

Not counted: 343 closed, 33 open-at-end.

Forced bomb drops after `armed`: {'pending': 0, 'fell': 2, 'taken': 98, 'wasted': 4, 'lost': 42}. Bombs the carrier hits dropped: {'none': 13, 'fell': 0, 'taken': 6, 'wasted': 1}.

Rooms with the most opportunities (top 12; the room each window closed in):

| room | A opp | armed | missed | other | A full | B opp | hit | miss | left |
|---|---|---|---|---|---|---|---|---|---|
| 3:0x69 | 6 | 6 | 0 | 0 | 0 | 144 | 0 | 0 | 144 |
| 3:0x5b | 29 | 15 | 11 | 3 | 2 | 97 | 1 | 0 | 96 |
| 3:0x4d | 86 | 83 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3:0x5c | 0 | 0 | 0 | 0 | 0 | 59 | 0 | 0 | 59 |
| 0:0x4c | 2 | 0 | 0 | 2 | 0 | 53 | 4 | 25 | 24 |
| 4:0x31 | 49 | 16 | 8 | 25 | 3 | 0 | 0 | 0 | 0 |
| 0:0x38 | 0 | 0 | 0 | 0 | 0 | 43 | 0 | 17 | 26 |
| 0:0x3d | 0 | 0 | 0 | 0 | 0 | 37 | 0 | 0 | 37 |
| 5:0x66 | 8 | 2 | 4 | 2 | 1 | 22 | 1 | 6 | 15 |
| 0:0x3c | 0 | 0 | 0 | 0 | 0 | 26 | 0 | 0 | 26 |
| 0:0x63 | 0 | 0 | 0 | 0 | 0 | 25 | 0 | 0 | 25 |
| 0:0x64 | 0 | 0 | 0 | 0 | 0 | 25 | 0 | 0 | 25 |


#### 2.0.0.10 backtest

Bomb metrics over 230 runs (the owner, Nov 23; 1446 windows):

| Metric A: the 10th-kill bomb | all windows | without a full bag |
|---|---|---|
| opportunities | 315 | 303 |
| armed | 146 (46.3%) | 145 (47.9%) |
| missed | 42 (13.3%) | 38 (12.5%) |
| other | 127 (40.3%) | 120 (39.6%) |

Not counted: 28 lost, 1 paused-out, 6 suspended, 4 open-at-end.

| Metric B: the bomb carrier | all windows | without a full bag |
|---|---|---|
| opportunities | 718 | 718 |
| hit | 13 (1.8%) | 13 (1.8%) |
| miss | 49 (6.8%) | 49 (6.8%) |
| left | 656 (91.4%) | 656 (91.4%) |

Not counted: 340 closed, 34 open-at-end.

Forced bomb drops after `armed`: {'pending': 0, 'fell': 2, 'taken': 98, 'wasted': 4, 'lost': 42}. Bombs the carrier hits dropped: {'none': 8, 'fell': 0, 'taken': 5, 'wasted': 0}.

Rooms with the most opportunities (top 12; the room each window closed in):

| room | A opp | armed | missed | other | A full | B opp | hit | miss | left |
|---|---|---|---|---|---|---|---|---|---|
| 3:0x69 | 6 | 6 | 0 | 0 | 0 | 144 | 0 | 0 | 144 |
| 3:0x5b | 29 | 15 | 11 | 3 | 2 | 97 | 1 | 0 | 96 |
| 3:0x4d | 86 | 83 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3:0x5c | 0 | 0 | 0 | 0 | 0 | 59 | 0 | 0 | 59 |
| 0:0x4c | 2 | 0 | 0 | 2 | 0 | 52 | 4 | 25 | 23 |
| 4:0x31 | 49 | 16 | 8 | 25 | 3 | 0 | 0 | 0 | 0 |
| 0:0x38 | 0 | 0 | 0 | 0 | 0 | 41 | 0 | 16 | 25 |
| 0:0x3d | 0 | 0 | 0 | 0 | 0 | 37 | 0 | 0 | 37 |
| 5:0x66 | 8 | 2 | 4 | 2 | 1 | 22 | 1 | 6 | 15 |
| 0:0x3c | 0 | 0 | 0 | 0 | 0 | 26 | 0 | 0 | 26 |
| 0:0x63 | 0 | 0 | 0 | 0 | 0 | 25 | 0 | 0 | 25 |
| 0:0x64 | 0 | 0 | 0 | 0 | 0 | 25 | 0 | 0 | 25 |


#### 2.0.0.9 backtest

Bomb metrics over 220 runs (the owner, Nov 23; 1109 windows):

| Metric A: the 10th-kill bomb | all windows | without a full bag |
|---|---|---|
| opportunities | 254 | 241 |
| armed | 128 (50.4%) | 126 (52.3%) |
| missed | 29 (11.4%) | 29 (12.0%) |
| other | 97 (38.2%) | 86 (35.7%) |

Not counted: 17 lost, 2 paused-out, 4 open-at-end.

| Metric B: the bomb carrier | all windows | without a full bag |
|---|---|---|
| opportunities | 525 | 525 |
| hit | 10 (1.9%) | 10 (1.9%) |
| miss | 26 (5.0%) | 26 (5.0%) |
| left | 489 (93.1%) | 489 (93.1%) |

Not counted: 284 closed, 23 open-at-end.

Forced bomb drops after `armed`: {'pending': 0, 'fell': 2, 'taken': 65, 'wasted': 6, 'lost': 55}. Bombs the carrier hits dropped: {'none': 7, 'fell': 0, 'taken': 2, 'wasted': 1}.

Rooms with the most opportunities (top 12; the room each window closed in):

| room | A opp | armed | missed | other | A full | B opp | hit | miss | left |
|---|---|---|---|---|---|---|---|---|---|
| 3:0x69 | 10 | 7 | 3 | 0 | 0 | 146 | 0 | 0 | 146 |
| 3:0x5b | 20 | 9 | 5 | 6 | 0 | 73 | 0 | 0 | 73 |
| 3:0x4d | 81 | 80 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3:0x5c | 0 | 0 | 0 | 0 | 0 | 70 | 0 | 0 | 70 |
| 4:0x31 | 31 | 12 | 7 | 12 | 2 | 0 | 0 | 0 | 0 |
| 4:0x40 | 26 | 1 | 0 | 25 | 7 | 0 | 0 | 0 | 0 |
| 0:0x4c | 0 | 0 | 0 | 0 | 0 | 25 | 0 | 7 | 18 |
| 0:0x38 | 0 | 0 | 0 | 0 | 0 | 23 | 0 | 11 | 12 |
| 5:0x66 | 3 | 0 | 1 | 2 | 0 | 16 | 4 | 5 | 7 |
| 0:0x3c | 0 | 0 | 0 | 0 | 0 | 17 | 1 | 0 | 16 |
| 0:0x63 | 0 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 17 |
| 0:0x64 | 0 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 17 |


#### 2.0.0.8 backtest

Bomb metrics over 220 runs (the owner, Nov 23; 1143 windows):

| Metric A: the 10th-kill bomb | all windows | without a full bag |
|---|---|---|
| opportunities | 245 | 233 |
| armed | 125 (51.0%) | 124 (53.2%) |
| missed | 26 (10.6%) | 26 (11.2%) |
| other | 94 (38.4%) | 83 (35.6%) |

Not counted: 20 lost, 2 paused-out, 4 open-at-end.

| Metric B: the bomb carrier | all windows | without a full bag |
|---|---|---|
| opportunities | 548 | 548 |
| hit | 6 (1.1%) | 6 (1.1%) |
| miss | 34 (6.2%) | 34 (6.2%) |
| left | 508 (92.7%) | 508 (92.7%) |

Not counted: 297 closed, 27 open-at-end.

Forced bomb drops after `armed`: {'pending': 0, 'fell': 2, 'taken': 63, 'wasted': 6, 'lost': 54}. Bombs the carrier hits dropped: {'none': 2, 'fell': 0, 'taken': 3, 'wasted': 1}.

Rooms with the most opportunities (top 12; the room each window closed in):

| room | A opp | armed | missed | other | A full | B opp | hit | miss | left |
|---|---|---|---|---|---|---|---|---|---|
| 3:0x69 | 10 | 7 | 3 | 0 | 0 | 146 | 0 | 0 | 146 |
| 3:0x5b | 20 | 9 | 5 | 6 | 0 | 73 | 0 | 0 | 73 |
| 3:0x4d | 81 | 80 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3:0x5c | 0 | 0 | 0 | 0 | 0 | 70 | 0 | 0 | 70 |
| 0:0x4c | 0 | 0 | 0 | 0 | 0 | 32 | 3 | 17 | 12 |
| 4:0x31 | 31 | 12 | 7 | 12 | 2 | 0 | 0 | 0 | 0 |
| 0:0x3d | 0 | 0 | 0 | 0 | 0 | 27 | 0 | 0 | 27 |
| 4:0x40 | 26 | 1 | 0 | 25 | 7 | 0 | 0 | 0 | 0 |
| 5:0x66 | 3 | 1 | 1 | 1 | 0 | 23 | 1 | 8 | 14 |
| 0:0x38 | 0 | 0 | 0 | 0 | 0 | 19 | 0 | 9 | 10 |
| 5:0x65 | 0 | 0 | 0 | 0 | 0 | 18 | 1 | 0 | 17 |
| 0:0x63 | 0 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 17 |


#### 2.0.0.7 backtest

Bomb metrics over 211 runs (the owner, Nov 23; 755 windows):

| Metric A: the 10th-kill bomb | all windows | without a full bag |
|---|---|---|
| opportunities | 139 | 135 |
| armed | 76 (54.7%) | 76 (56.3%) |
| missed | 20 (14.4%) | 19 (14.1%) |
| other | 43 (30.9%) | 40 (29.6%) |

Not counted: 22 lost, 2 open-at-end.

| Metric B: the bomb carrier | all windows | without a full bag |
|---|---|---|
| opportunities | 331 | 331 |
| hit | 1 (0.3%) | 1 (0.3%) |
| miss | 3 (0.9%) | 3 (0.9%) |
| left | 327 (98.8%) | 327 (98.8%) |

Not counted: 233 closed, 28 open-at-end.

Forced bomb drops after `armed`: {'pending': 0, 'fell': 1, 'taken': 36, 'wasted': 2, 'lost': 37}. Bombs the carrier hits dropped: {'none': 0, 'fell': 0, 'taken': 1, 'wasted': 0}.

Rooms with the most opportunities (top 12; the room each window closed in):

| room | A opp | armed | missed | other | A full | B opp | hit | miss | left |
|---|---|---|---|---|---|---|---|---|---|
| 3:0x69 | 2 | 2 | 0 | 0 | 0 | 118 | 0 | 0 | 118 |
| 3:0x5b | 12 | 5 | 4 | 3 | 1 | 97 | 0 | 0 | 97 |
| 3:0x4d | 60 | 57 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3:0x5c | 0 | 0 | 0 | 0 | 0 | 52 | 0 | 0 | 52 |
| 4:0x31 | 18 | 5 | 5 | 8 | 0 | 0 | 0 | 0 | 0 |
| 0:0x63 | 0 | 0 | 0 | 0 | 0 | 13 | 0 | 0 | 13 |
| 0:0x64 | 0 | 0 | 0 | 0 | 0 | 13 | 0 | 0 | 13 |
| 0:0x73 | 0 | 0 | 0 | 0 | 0 | 13 | 0 | 0 | 13 |
| 4:0x40 | 13 | 0 | 1 | 12 | 2 | 0 | 0 | 0 | 0 |
| 3:0x59 | 6 | 2 | 3 | 1 | 0 | 0 | 0 | 0 | 0 |
| 0:0x4c | 0 | 0 | 0 | 0 | 0 | 5 | 1 | 2 | 2 |
| 3:0x0f:cave | 5 | 0 | 0 | 5 | 1 | 0 | 0 | 0 | 0 |


#### 2.0.0.6 backtest

Bomb metrics over 211 runs (the owner, Nov 23; 757 windows):

| Metric A: the 10th-kill bomb | all windows | without a full bag |
|---|---|---|
| opportunities | 134 | 130 |
| armed | 73 (54.5%) | 73 (56.2%) |
| missed | 19 (14.2%) | 18 (13.8%) |
| other | 42 (31.3%) | 39 (30.0%) |

Not counted: 22 lost, 2 open-at-end.

| Metric B: the bomb carrier | all windows | without a full bag |
|---|---|---|
| opportunities | 333 | 333 |
| hit | 0 (0.0%) | 0 (0.0%) |
| miss | 6 (1.8%) | 6 (1.8%) |
| left | 327 (98.2%) | 327 (98.2%) |

Not counted: 237 closed, 29 open-at-end.

Forced bomb drops after `armed`: {'pending': 0, 'fell': 1, 'taken': 34, 'wasted': 2, 'lost': 36}. Bombs the carrier hits dropped: {'none': 0, 'fell': 0, 'taken': 0, 'wasted': 0}.

Rooms with the most opportunities (top 12; the room each window closed in):

| room | A opp | armed | missed | other | A full | B opp | hit | miss | left |
|---|---|---|---|---|---|---|---|---|---|
| 3:0x69 | 2 | 2 | 0 | 0 | 0 | 118 | 0 | 0 | 118 |
| 3:0x5b | 12 | 5 | 4 | 3 | 1 | 97 | 0 | 0 | 97 |
| 3:0x4d | 60 | 57 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3:0x5c | 0 | 0 | 0 | 0 | 0 | 52 | 0 | 0 | 52 |
| 4:0x31 | 18 | 5 | 5 | 8 | 0 | 0 | 0 | 0 | 0 |
| 0:0x63 | 0 | 0 | 0 | 0 | 0 | 13 | 0 | 0 | 13 |
| 0:0x64 | 0 | 0 | 0 | 0 | 0 | 13 | 0 | 0 | 13 |
| 0:0x73 | 0 | 0 | 0 | 0 | 0 | 13 | 0 | 0 | 13 |
| 4:0x40 | 13 | 0 | 1 | 12 | 2 | 0 | 0 | 0 | 0 |
| 0:0x2d | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 1 | 6 |
| 0:0x3d | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 6 |
| 3:0x59 | 6 | 2 | 3 | 1 | 0 | 0 | 0 | 0 | 0 |


#### 2.0.0.5 backtest

Bomb metrics over 219 runs (the owner, Nov 23; 947 windows):

| Metric A: the 10th-kill bomb | all windows | without a full bag |
|---|---|---|
| opportunities | 213 | 208 |
| armed | 111 (52.1%) | 110 (52.9%) |
| missed | 26 (12.2%) | 26 (12.5%) |
| other | 76 (35.7%) | 72 (34.6%) |

Not counted: 22 lost, 1 paused-out.

| Metric B: the bomb carrier | all windows | without a full bag |
|---|---|---|
| opportunities | 493 | 493 |
| hit | 113 (22.9%) | 113 (22.9%) |
| miss | 5 (1.0%) | 5 (1.0%) |
| left | 375 (76.1%) | 375 (76.1%) |

Not counted: 201 closed, 17 open-at-end.

Forced bomb drops after `armed`: {'pending': 0, 'fell': 1, 'taken': 62, 'wasted': 12, 'lost': 36}. Bombs the carrier hits dropped: {'none': 66, 'fell': 0, 'taken': 9, 'wasted': 38}.

Rooms with the most opportunities (top 12; the room each window closed in):

| room | A opp | armed | missed | other | A full | B opp | hit | miss | left |
|---|---|---|---|---|---|---|---|---|---|
| 3:0x69 | 10 | 9 | 1 | 0 | 0 | 192 | 94 | 0 | 98 |
| 3:0x5b | 23 | 10 | 9 | 4 | 0 | 94 | 0 | 0 | 94 |
| 3:0x4d | 75 | 73 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4:0x31 | 31 | 10 | 5 | 16 | 0 | 18 | 0 | 0 | 18 |
| 3:0x5c | 0 | 0 | 0 | 0 | 0 | 40 | 0 | 0 | 40 |
| 4:0x50 | 6 | 2 | 2 | 2 | 0 | 34 | 12 | 0 | 22 |
| 4:0x40 | 21 | 1 | 0 | 20 | 2 | 0 | 0 | 0 | 0 |
| 0:0x63 | 0 | 0 | 0 | 0 | 0 | 19 | 0 | 0 | 19 |
| 0:0x64 | 0 | 0 | 0 | 0 | 0 | 19 | 0 | 0 | 19 |
| 0:0x73 | 0 | 0 | 0 | 0 | 0 | 19 | 0 | 0 | 19 |
| 4:0x30 | 0 | 0 | 0 | 0 | 0 | 15 | 0 | 0 | 15 |
| 3:0x0f:cave | 14 | 1 | 0 | 13 | 0 | 0 | 0 | 0 | 0 |


#### 2.0.0.4 backtest

Bomb metrics over 219 runs (the owner, Nov 23; 919 windows):

| Metric A: the 10th-kill bomb | all windows | without a full bag |
|---|---|---|
| opportunities | 206 | 204 |
| armed | 109 (52.9%) | 109 (53.4%) |
| missed | 25 (12.1%) | 25 (12.3%) |
| other | 72 (35.0%) | 70 (34.3%) |

Not counted: 22 lost, 1 paused-out.

| Metric B: the bomb carrier | all windows | without a full bag |
|---|---|---|
| opportunities | 475 | 475 |
| hit | 113 (23.8%) | 113 (23.8%) |
| miss | 3 (0.6%) | 3 (0.6%) |
| left | 359 (75.6%) | 359 (75.6%) |

Not counted: 198 closed, 17 open-at-end.

Forced bomb drops after `armed`: {'pending': 0, 'fell': 1, 'taken': 61, 'wasted': 12, 'lost': 35}. Bombs the carrier hits dropped: {'none': 66, 'fell': 0, 'taken': 9, 'wasted': 38}.

Rooms with the most opportunities (top 12; the room each window closed in):

| room | A opp | armed | missed | other | A full | B opp | hit | miss | left |
|---|---|---|---|---|---|---|---|---|---|
| 3:0x69 | 10 | 9 | 1 | 0 | 0 | 192 | 94 | 0 | 98 |
| 3:0x5b | 23 | 10 | 9 | 4 | 0 | 94 | 0 | 0 | 94 |
| 3:0x4d | 75 | 73 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4:0x31 | 31 | 10 | 5 | 16 | 0 | 18 | 0 | 0 | 18 |
| 3:0x5c | 0 | 0 | 0 | 0 | 0 | 40 | 0 | 0 | 40 |
| 4:0x50 | 6 | 2 | 2 | 2 | 0 | 34 | 12 | 0 | 22 |
| 4:0x40 | 21 | 1 | 0 | 20 | 2 | 0 | 0 | 0 | 0 |
| 0:0x63 | 0 | 0 | 0 | 0 | 0 | 19 | 0 | 0 | 19 |
| 0:0x64 | 0 | 0 | 0 | 0 | 0 | 19 | 0 | 0 | 19 |
| 0:0x73 | 0 | 0 | 0 | 0 | 0 | 19 | 0 | 0 | 19 |
| 4:0x30 | 0 | 0 | 0 | 0 | 0 | 15 | 0 | 0 | 15 |
| 3:0x0f:cave | 14 | 1 | 0 | 13 | 0 | 0 | 0 | 0 | 0 |


#### 2.0.0.3 backtest

Bomb metrics over 219 runs (the owner, Nov 23; 919 windows):

| Metric A: the 10th-kill bomb | all windows | without a full bag |
|---|---|---|
| opportunities | 206 | 204 |
| armed | 109 (52.9%) | 109 (53.4%) |
| missed | 25 (12.1%) | 25 (12.3%) |
| other | 72 (35.0%) | 70 (34.3%) |

Not counted: 22 lost, 1 paused-out.

| Metric B: the bomb carrier | all windows | without a full bag |
|---|---|---|
| opportunities | 475 | 475 |
| hit | 113 (23.8%) | 113 (23.8%) |
| miss | 3 (0.6%) | 3 (0.6%) |
| left | 359 (75.6%) | 359 (75.6%) |

Not counted: 198 closed, 17 open-at-end.

Forced bomb drops after `armed`: {'pending': 0, 'fell': 1, 'taken': 61, 'wasted': 12, 'lost': 35}. Bombs the carrier hits dropped: {'none': 66, 'fell': 0, 'taken': 9, 'wasted': 38}.

Rooms with the most opportunities (top 12; the room each window closed in):

| room | A opp | armed | missed | other | A full | B opp | hit | miss | left |
|---|---|---|---|---|---|---|---|---|---|
| 3:0x69 | 10 | 9 | 1 | 0 | 0 | 192 | 94 | 0 | 98 |
| 3:0x5b | 23 | 10 | 9 | 4 | 0 | 94 | 0 | 0 | 94 |
| 3:0x4d | 75 | 73 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4:0x31 | 31 | 10 | 5 | 16 | 0 | 18 | 0 | 0 | 18 |
| 3:0x5c | 0 | 0 | 0 | 0 | 0 | 40 | 0 | 0 | 40 |
| 4:0x50 | 6 | 2 | 2 | 2 | 0 | 34 | 12 | 0 | 22 |
| 4:0x40 | 21 | 1 | 0 | 20 | 2 | 0 | 0 | 0 | 0 |
| 0:0x63 | 0 | 0 | 0 | 0 | 0 | 19 | 0 | 0 | 19 |
| 0:0x64 | 0 | 0 | 0 | 0 | 0 | 19 | 0 | 0 | 19 |
| 0:0x73 | 0 | 0 | 0 | 0 | 0 | 19 | 0 | 0 | 19 |
| 4:0x30 | 0 | 0 | 0 | 0 | 0 | 15 | 0 | 0 | 15 |
| 3:0x0f:cave | 14 | 1 | 0 | 13 | 0 | 0 | 0 | 0 | 0 |


#### 2.0.0.2 backtest

Bomb metrics over 219 runs (the owner, Nov 23; 942 windows):

| Metric A: the 10th-kill bomb | all windows | without a full bag |
|---|---|---|
| opportunities | 209 | 207 |
| armed | 111 (53.1%) | 111 (53.6%) |
| missed | 25 (12.0%) | 25 (12.1%) |
| other | 73 (34.9%) | 71 (34.3%) |

Not counted: 22 lost, 1 paused-out.

| Metric B: the bomb carrier | all windows | without a full bag |
|---|---|---|
| opportunities | 494 | 494 |
| hit | 113 (22.9%) | 113 (22.9%) |
| miss | 4 (0.8%) | 4 (0.8%) |
| left | 377 (76.3%) | 377 (76.3%) |

Not counted: 199 closed, 17 open-at-end.

Forced bomb drops after `armed`: {'pending': 0, 'fell': 1, 'taken': 63, 'wasted': 12, 'lost': 35}. Bombs the carrier hits dropped: {'none': 66, 'fell': 0, 'taken': 9, 'wasted': 38}.

Rooms with the most opportunities (top 12; the room each window closed in):

| room | A opp | armed | missed | other | A full | B opp | hit | miss | left |
|---|---|---|---|---|---|---|---|---|---|
| 3:0x69 | 10 | 9 | 1 | 0 | 0 | 192 | 94 | 0 | 98 |
| 3:0x5b | 23 | 10 | 9 | 4 | 0 | 94 | 0 | 0 | 94 |
| 3:0x4d | 75 | 73 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4:0x31 | 31 | 10 | 5 | 16 | 0 | 18 | 0 | 0 | 18 |
| 3:0x5c | 0 | 0 | 0 | 0 | 0 | 40 | 0 | 0 | 40 |
| 4:0x50 | 6 | 2 | 2 | 2 | 0 | 34 | 12 | 0 | 22 |
| 4:0x40 | 21 | 1 | 0 | 20 | 2 | 0 | 0 | 0 | 0 |
| 0:0x63 | 0 | 0 | 0 | 0 | 0 | 20 | 0 | 0 | 20 |
| 0:0x73 | 0 | 0 | 0 | 0 | 0 | 20 | 0 | 0 | 20 |
| 0:0x64 | 0 | 0 | 0 | 0 | 0 | 19 | 0 | 0 | 19 |
| 4:0x30 | 0 | 0 | 0 | 0 | 0 | 15 | 0 | 0 | 15 |
| 3:0x0f:cave | 14 | 1 | 0 | 13 | 0 | 0 | 0 | 0 | 0 |


#### 2.0.0.1 backtest

Bomb metrics over 221 runs (the owner, Nov 23; 803 windows):

| Metric A: the 10th-kill bomb | all windows | without a full bag |
|---|---|---|
| opportunities | 157 | 157 |
| armed | 84 (53.5%) | 84 (53.5%) |
| missed | 18 (11.5%) | 18 (11.5%) |
| other | 55 (35.0%) | 55 (35.0%) |

Not counted: 22 lost, 3 open-at-end.

| Metric B: the bomb carrier | all windows | without a full bag |
|---|---|---|
| opportunities | 400 | 400 |
| hit | 84 (21.0%) | 84 (21.0%) |
| miss | 5 (1.2%) | 5 (1.2%) |
| left | 311 (77.8%) | 311 (77.8%) |

Not counted: 200 closed, 21 open-at-end.

Forced bomb drops after `armed`: {'pending': 0, 'fell': 0, 'taken': 42, 'wasted': 3, 'lost': 39}. Bombs the carrier hits dropped: {'none': 52, 'fell': 0, 'taken': 5, 'wasted': 27}.

Rooms with the most opportunities (top 12; the room each window closed in):

| room | A opp | armed | missed | other | A full | B opp | hit | miss | left |
|---|---|---|---|---|---|---|---|---|---|
| 3:0x69 | 7 | 4 | 3 | 0 | 0 | 165 | 74 | 0 | 91 |
| 3:0x5b | 6 | 2 | 3 | 1 | 0 | 106 | 0 | 0 | 106 |
| 3:0x4d | 62 | 62 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3:0x5c | 0 | 0 | 0 | 0 | 0 | 39 | 0 | 0 | 39 |
| 4:0x31 | 20 | 6 | 3 | 11 | 0 | 5 | 0 | 0 | 5 |
| 4:0x50 | 7 | 1 | 2 | 4 | 0 | 12 | 5 | 0 | 7 |
| 4:0x40 | 14 | 1 | 0 | 13 | 0 | 0 | 0 | 0 | 0 |
| 0:0x63 | 0 | 0 | 0 | 0 | 0 | 12 | 0 | 0 | 12 |
| 0:0x64 | 0 | 0 | 0 | 0 | 0 | 12 | 0 | 0 | 12 |
| 0:0x73 | 0 | 0 | 0 | 0 | 0 | 12 | 0 | 0 | 12 |
| 4:0x61 | 3 | 1 | 0 | 2 | 0 | 7 | 2 | 0 | 5 |
| 3:0x0f:cave | 8 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 0 |


#### 2.0.0.0 backtest

Bomb metrics over 211 runs (the owner, Nov 23; 756 windows):

| Metric A: the 10th-kill bomb | all windows | without a full bag |
|---|---|---|
| opportunities | 113 | 111 |
| armed | 65 (57.5%) | 65 (58.6%) |
| missed | 12 (10.6%) | 12 (10.8%) |
| other | 36 (31.9%) | 34 (30.6%) |

Not counted: 19 lost, 4 open-at-end.

| Metric B: the bomb carrier | all windows | without a full bag |
|---|---|---|
| opportunities | 369 | 369 |
| hit | 96 (26.0%) | 96 (26.0%) |
| miss | 2 (0.5%) | 2 (0.5%) |
| left | 271 (73.4%) | 271 (73.4%) |

Not counted: 225 closed, 26 open-at-end.

Forced bomb drops after `armed`: {'pending': 0, 'fell': 1, 'taken': 30, 'wasted': 7, 'lost': 27}. Bombs the carrier hits dropped: {'none': 52, 'fell': 0, 'taken': 2, 'wasted': 42}.

Rooms with the most opportunities (top 12; the room each window closed in):

| room | A opp | armed | missed | other | A full | B opp | hit | miss | left |
|---|---|---|---|---|---|---|---|---|---|
| 3:0x69 | 8 | 7 | 1 | 0 | 0 | 172 | 91 | 0 | 81 |
| 3:0x5b | 5 | 2 | 3 | 0 | 0 | 110 | 0 | 0 | 110 |
| 3:0x4d | 48 | 48 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3:0x5c | 0 | 0 | 0 | 0 | 0 | 29 | 0 | 0 | 29 |
| 4:0x31 | 16 | 3 | 4 | 9 | 1 | 4 | 0 | 0 | 4 |
| 4:0x50 | 7 | 3 | 2 | 2 | 0 | 13 | 5 | 0 | 8 |
| 3:0x0f:cave | 9 | 1 | 0 | 8 | 0 | 0 | 0 | 0 | 0 |
| 0:0x63 | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 8 |
| 0:0x64 | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 8 |
| 0:0x73 | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 8 |
| 4:0x40 | 8 | 0 | 0 | 8 | 1 | 0 | 0 | 0 | 0 |
| 4:0x61 | 1 | 0 | 0 | 1 | 0 | 7 | 0 | 0 | 7 |


