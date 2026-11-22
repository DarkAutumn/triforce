# Round 38: what a hit predictor is worth at play, and how it must be connected (branch `r38-hit`)

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

## 1. Round 36's last two arms (B, P)

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

## 2. The direct test: the supervised head as a fixed part of the network

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
