# Round 6: enemy-based families, rooms toward the bow, pass-through damage

Lee (Nov 3): rooms are rooms; a Red Goriya room needs the same skills in any dungeon. Split the Goriya rooms into
their own model, give Wallmasters their own model, name Dungeon 1's melee family by its enemies, and add the two
off-route Dungeon 1 rooms toward the bow. Also try a damage penalty for rooms Link only has to cross.

Setup as in round 5 (2M-step budget, balanced 95% early stop, 20 sampled episodes per seed state, hp
`[0.15, max]` with half the resets at full health). Chains are 20 runs unless noted.

## Summary (where round 6 ended, Nov 3)

Best chains (40 runs each, to the Dodongo room 2:0x0e):

| segment | result |
|---|---|
| full chain, milestone1 | 1/40 (r23; the first ever) |
| full chain, milestone1-fairy | 2–3/40 |
| after D1 (`--from 0_37c_2`), milestone1-fairy | **8/40** |
| D2 entrance at full hearts (`--from 2_7ds_2`) | 9/40 |

- **Fairy route** (§9, §11): `milestone1-fairy` (Lee) goes south after D1, past the fairy pond 0x39, then east and
  north into D2. It beats milestone1 after D1: 37/40 reach the D2 entrance with 2.5 hearts (milestone1: 27/40 with
  1.7); every pond visit refilled to full. 40 new start states; ow-zora / ow-octorok / ow-start learned the new
  rooms (97–98%; the pond 2% → 98%).
- **The Moblin room Lee flagged** (§9): 0x5b's east edge has two openings and the lower one leads into a pocket of
  0x5c that only exits south; leaving through it counted as success. game.yaml exit `span` limits it to the upper
  opening. The wavefront also had an off-map lane under every overworld room's bottom tree line (fixed). ow-moblin
  failed for a third reason (r9a's habit of leaving Moblin rooms west, fatal when entering from the west);
  warm-started from ow-octorok it went 18% → 97%.
- **Heart tracking** (§8): drops / taken / left / healed / wasted per room in training metrics, eval, chain
  records and the GUI (real vs total heart change). Overworld models left 26 of 34 dropped hearts; dungeon models
  take 55–80%. A "heart is the objective when hurt" option (`collect_hearts`) didn't help the chain (d2-rope:
  12/80 vs 16/80) and was removed; Lee's v1-style variant (nearest of heart or exit) is still to try (todo.md).
- **Dungeon 2** (§12): D2 rooms got 3 start states each (were 1). d2-rope r28 95% (was 90%), promoted; it lifted
  the fairy post-D1 chain from 2/40 to 8/40. goriya r28 lost to r21 in the chain (D2 6/40 vs 9/40), r21 stays.
- **Dropped after testing:** pass-through damage penalty (§5), chain-matched hp (§6), heart objective (§10).
- **What limits the full chain:** the D1 Goriya room (0x44). Link arrives with ~1.1 hearts, it kills 15 of the 28
  runs that reach it, and only 8 of 40 runs leave D1 alive. 0x53 kills 5 more.
- **Promoted at the end:** ow-start / ow-zora / ow-octorok r24-fairy, ow-moblin r26c, ow-leever-peahat and
  boss-aquamentus r23, stalfos-keese-gel r25, goriya r21, wallmaster r10 copy, d2-rope r28.
- Lee's observations from watching `play-zelda.py` after the round (Link lingering in rooms, monsters chasing
  "shadow Link", Aquamentus wandering after the kill, hearts left behind, blocking, stuck runs, the boomerang
  question, D2's 0x7e key) are in todo.md "Next up", in order, and the game mechanics in AGENTS.md.

**Round 7 integrity correction:** the train/evaluate CLIs discarded alternate `--specialist /tmp/*.yaml` configs
and reloaded the default specialist by name. Confirmed affected training arms: r20-rope-pt, r21-goriya-chainhp,
r22-skg-chainhp, r25-skg-hearts and r27-rope-hearts (`run.json` records the temporary paths; round7.md §4).
Their measured scores remain real, but these comparisons do not establish the named treatment's effect. No exact
old main-YAML settings are recoverable from those run files. Lee's rejected ideas stay rejected; do not silently
reintroduce them because of this finding. The CLI bug is fixed for subsequent experiments.

## 1. Families

| family | rooms | was |
|---|---|---|
| `stalfos-keese-gel` | D1 entrance hall, Keese key room, Stalfos key rooms ×2, Stalfos room, Keese room, Gel rooms ×2, triforce room; off-route Stalfos key room north of the Gel map room (0x33) | `d1-melee` (+0x33) |
| `goriya` | D1 Goriya room (3, boomerang prize, 0x44); D2 Goriya rooms (5 each, 0x5e and 0x1e); off-route D1 Goriya key room (3, 0x23) | `d1-wallmaster` (0x44) + `d2-goriya` (+0x23) |
| `wallmaster` | D1 Wallmaster room (0x45) | `d1-wallmaster` (0x45) |

`docs/specialists.md` and `docs/decisions.md` record the change. Promoted models were copied under the new names
(`stalfos-keese-gel/from-d1-melee-r15`, `wallmaster/from-d1-wallmaster-r10`, `goriya/from-d2-goriya-r12`).

## 2. States for the rooms toward the bow

`game.yaml` now has 0x33's N door and 0x23's S/W doors locked (`docs/route.md`) and a new room 0x22 (the bow room:
a push block opens the stairs to the bow cellar). `entry_states path --source 1_43w_<c> --rooms
1:0x33,1:0x23,1:0x22` (new: Go-Explore only has to enter the next room, whatever the objective wants) built all 9
states in 4 minutes: `1_33s_{0,1,2}`, `1_23s_{0,1,2}`, `1_22e_{0,1,2}` (5k–22k env steps per room). Audit: full
monster counts, 4 spawn layouts each; the bow room spawns no monsters.

At 0x33 the objective is the visible key, then S (back toward the route); at 0x23 it is fight, key, then S. The bow
route itself isn't defined yet, so these rooms train their fight + key + exit, with the exit the objective gives.

## 3. Goriya

Neither old model plays the other dungeon's Goriya rooms (on the new `goriya` room set, 20 episodes per state):

| model | D1 Goriya room (0x44) | D1 north Goriya room (0x23) | D2 first (0x5e) | D2 shutter room (0x1e) | all |
|---|---|---|---|---|---|
| `d2-goriya` r12 | 0%, −2.17 hearts | 0%, −2.07 | 80%, −0.70 | 85%, −0.53 | 21% |
| `d1-wallmaster` r10 | 75%, −0.97 | 0%, −2.14 | 50%, −1.90 | 20%, −2.73 | 37% |

In D1 Link has 3 hearts and no boomerang (it is 0x44's prize); in D2 he has 4 hearts and the boomerang.
Trajectories: 0x23 explored 3/3 (76k env steps); 0x44 re-explored under the `goriya` config 3/3 (94k), including
`1_44w_2`, which the old two-room family never solved (its trajectories ran on into 0x45 and crashed the goriya
curriculum: a trajectory's states must stay in the family's rooms).

**r17** (warm start from `d2-goriya` r12, curriculum on, 8 workers, 2M steps): 66% over all four rooms (0x44 68%,
0x23 67%, 0x5e 65%, 0x1e 60%; −0.96 hearts per room). Every trajectory finished the curriculum by iteration ~100;
the balanced training rate then sat at 0.60–0.72 with 25–33% deaths. One model now plays every Goriya room at
60–68%, against 0–85% for the old split models, but it is still below each old model in its own rooms (0x44 75%
for `d1-wallmaster`, 0x5e/0x1e 80/85% for `d2-goriya`).

**r19** (r17 continued, +2M steps, stop at 99%): 72% (0x44 **78%**, 0x23 68%, 0x5e 70%, 0x1e 70%; −1.01 hearts).
0x44 now beats the old `d1-wallmaster` model (75%); the D2 rooms are still below `d2-goriya`'s 80/85% (20 episodes
each). Promoted, then 40-run chains with every promoted model:

| chain (40 runs) | reaches | ends |
|---|---|---|
| full, from `start` | 0x63 34, 0x52 24, D1 Goriya room 24, Wallmaster room **11**, Aquamentus 11, triforce **8**, D2 entrance **5** | 0/40; D1 Goriya room ×13, 0x53 ×7, 0x74 ×6, Aquamentus ×3 |
| after D1, from `0_37c_2` | D2 entrance 20 | 0/40; 0x6d ×7, D2 first Goriya room ×7, 0x4c ×6, Moblin rooms ×8 |
| D2, from `2_7ds_2` | 0x1e 5, Dodongo room 1 | 1/40; 0x4e ×11, 0x2e ×10, 0x3e ×6, 0x5e ×5 |

Round 5's best full chain (20 runs) reached the Wallmaster room in 3 and D2 in 1; with the `goriya` model 11/40
pass the D1 Goriya room and 5/40 reach D2. In D2 the rope rooms now end most runs; the first Goriya room still
costs 1.34 hearts.

## 4. stalfos-keese-gel with the north Stalfos room (r18)

r18 continues the `d1-melee` r15 weights with 0x33 added (stop at 99%, 2M steps, 10 workers). Both models evaluated
under the new config (10 rooms, 34 states):

| model | north Stalfos key room (0x33) | 5-Stalfos key room (0x53) | 3-Stalfos room (0x63) | 5-Stalfos key room (0x74) | Keese key room (0x72) | route rooms | all |
|---|---|---|---|---|---|---|---|
| `d1-melee` r15 | 0% (no-progress) | 92% | 100% | 92% | 90% | 96.9% | 88% |
| r18 | **75%**, −0.53 hearts | 88% | 90% | 95% | 98% | 97.1% | **95%** |

Same on the route, plus the new room. Promoted. Note the noise: r15's 0x53 was 78% in its own eval and 92% here
(60 episodes per room, a different emulator RNG stream after code changes); differences under ~10% in one room are
not meaningful.

## 5. Pass-through damage penalty (r16)

`reward.pass_through_damage: k`: when an episode starts in a room whose objective is MOVE (no fight or pickup
wanted) and ends in success, the terminal reward loses k per heart lost since entry. r16 used k = 4 on
ow-leever-peahat and ow-moblin, trained from scratch like r9a (same config without the penalty).

| specialist | r3 | r9a (no penalty) | r16 (k = 4) |
|---|---|---|---|
| ow-leever-peahat | 96%, −0.19 hearts, 16 steps | 99%, −0.39, 31 steps | 98%, −0.20, 20 steps |
| ow-moblin | 92%, −0.38 | 98%, −0.33 | 92%, −0.19 |

The penalty halved the damage in eval against the same config without it. In the post-D1 chain (40 runs from
`0_37c_2`, only these two specialists swapped) it did not change the outcome: D2 reached 20/40 either way, the
Moblin room's loss fell 0.65 → 0.27 hearts, the Leever rooms' did not fall, deaths moved around (Moblin rooms 8 → 5,
Leever rooms 2 → 4).

Second test on d2-rope, which drains the most in Dungeon 2: two arms from r14, 1M steps each (6 workers, stop at
99%), the only difference being `pass_through_damage: 4`. Both evaluated under the plain config, then the Dungeon 2
chain (40 runs from `2_7ds_2`, only d2-rope swapped):

| arm | eval | eval hearts per room | D2 chain: reach 0x1e | reach the Dodongo room |
|---|---|---|---|---|
| r20-rope-plain | 94% | −0.46 | 10/40 | 4/40 |
| r20-rope-pt (k = 4) | 91% | −0.48 | 7/40 | 6/40 |

No damage reduction in eval, no clear chain difference. **Dropped** (configs and code removed). The plain arm is a
good d2-rope model anyway: 94% (0x4e 85%, 0x3e 90%, 0x2e 90%); promoted. With it the D2 chain reaches the Dodongo
room in 4/40 (r14: 1/40).

## 6. Chain-matched hp (goriya)

`ram_randomization.chain_hp: 0.5`: half the room-entry resets take their hearts from the chain profile
(`triforce/chain_hp.json`, 14 recent chains; the D1 Goriya room's arrivals average 1.14 of 3 hearts). Two arms from
goriya r19, 1M steps each (6 workers, stop at 99%):

| arm | eval (standard hp) | eval at chain hp (`chain_hp: 1.0`) | full chain (40): pass the D1 Goriya room |
|---|---|---|---|
| r19 (start) | 72% | 62% (0x44 53%) | 11/24 |
| r21-goriya-plain | 72% | 61% (0x44 45%) | 8/24 |
| r21-goriya-chainhp | 76% | 64% (0x44 50%) | 10/24 |

All within noise; chain_hp is at best a small gain. r21-goriya-chainhp is promoted (best on both eval
distributions). The trace explains the D1 Goriya room (`1_44w_0`, 20 episodes each, r21-goriya-chainhp):

| start hp | success | mean steps | actions | damage taken from |
|---|---|---|---|---|
| 3 hearts | 95% | 73 | MOVE 65%, SWORD 29%, BEAMS 5% | contact ×27, boomerangs ×10 |
| 1 heart | 45% | 389 | SWORD 38%, WAIT 33%, MOVE 28% | contact ×19, boomerangs ×9 |

At full health the model wins by trading hits; at one heart it can't afford that, turns passive (a third of its
actions are WAIT) and still dies mostly to contact. The chain arrives at this room with about one heart, so the
fix is either less damage upstream or a way of fighting Goriyas without trading hits.

Heart drops are not a way out: over 20 episodes per room, the 5-Stalfos rooms dropped hearts in 0 (0x53) and 8
(0x74) episodes; 4 were picked up.

Second chain-hp test, stalfos-keese-gel: two arms from r18, 1M steps each.

| arm | eval | full chain (40): reach D1 Goriya room | Wallmaster room | Aquamentus | triforce | D2 entrance |
|---|---|---|---|---|---|---|
| r22-skg-plain | **98%** (0x53 93%, 0x33 93%), −0.16 hearts | **29** | **12** | **12** | **8** | 4 (+1 at 0x1e) |
| r22-skg-chainhp | 97% (0x53 90%), −0.14 | 24 | 7 | 7 | 4 | 2 |

Chain hp didn't help here either (fewer runs through the 5-Stalfos key room: 8 deaths vs 4). r22-skg-plain is
promoted: one more million plain steps took the weak rooms to 93% and let 29 of 40 chains reach the D1 Goriya room
(r19 chain: 24). **Chain-matched hp: no gain in two tests** (goriya within noise, stalfos-keese-gel worse); the
option is off everywhere.

## 7. Train longer (r23) and the first full-chain success

Chains with stalfos-keese-gel r22-plain, goriya r21-chainhp and d2-rope r20-plain promoted (40 runs each):
the full chain reaches the D1 Goriya room 29/40, the D2 entrance 4/40, 0x1e once; post-D1 reaches the D2 entrance
20/40; **the D2 segment reaches the Dodongo room 7/40** (r19 chain: 1/40; r20-rope-plain arm: 4/40).

r23 continued four promoted models with `--stop-success 0.99` (up to 2M steps). Evaluated against the promoted
models under the current yamls:

| specialist | promoted before | r23-more | |
|---|---|---|---|
| ow-zora | r3c: 96%, −0.34 hearts | **98%, −0.13** (0x38 92% → 96%) | promoted |
| boss-aquamentus | r7: 93%, +0.12 | **100%, +0.48** (384k steps, early stop) | promoted |
| ow-leever-peahat | r3: 96%, −0.19 | **100%, −0.16** (0x2b 85% → 100%) | promoted |
| wallmaster | r10 | early stop after 5 iterations: the same model | not promoted |

Chains after the promotions (40 runs each; r22 chains in brackets):

| segment | reach | success (Dodongo room) |
|---|---|---|
| full | D1 Goriya room 32 [29], Aquamentus 14 [12], triforce 9 [8], D2 entrance 8 [4], 0x1e 1 [1] | **1/40** [0] |
| post-D1 (`0_37c_2`) | D2 entrance **27** [20], 0x1e 1 | 0 |
| D2 (`2_7ds_2`) | 0x1e 11 [11] | 7/40 [7] |

**First full-chain success:** one run went from the game start into the Dodongo room. Deaths, full chain: D1
Goriya room 16 (of 32 that reached it), 0x53 7, Aquamentus 4, D2 rooms 7. Post-D1: 0x4c 8 (ow-octorok, arriving
with 1.8 hearts), 0x6d 7, 0x6e 6, 0x5e 6, 0x4e 5. D2: 0x3e 11, 0x2e 5, 0x5e 4, 0x1e 4.

## 8. Heart tracking (Lee): do the agents pick up dropped hearts?

New: `hearts.HeartTracker` follows every dropped heart (item 0x22 only; fairies, triforce and containers don't
count) through a room visit: `dropped`, `taken` (by Link or his weapons; a vanished heart whose lifetime couldn't
have run out), `healed` (real hearts: `min(1, max − hp)` per heart, `Z_01 TakeHearts`), `wasted` (taken − healed),
`left` (on the floor when Link left or died), `expired`. Episodes report `health_change` (real) and
`health_change_total` (real + wasted); eval, chain records, training metrics and the training GUI show both.

Survey of the promoted models (evaluate, 20 episodes per start state; rooms with drops):

| specialist | room | drops / episode | taken | left | health real (total) |
|---|---|---|---|---|---|
| stalfos-keese-gel | 1:0x53 | 0.45 | 0.22 | **0.22** | −0.69 (−0.62) |
| stalfos-keese-gel | 1:0x74 | 0.18 | 0.05 | 0.13 | −0.38 |
| stalfos-keese-gel | 1:0x33 | 0.33 | 0.28 | 0.05 | −0.48 |
| goriya | 1:0x44 | 0.28 | 0.17 | 0.03 | −0.91 |
| goriya | 2:0x1e | 0.90 | 0.55 | 0.05 | −0.93 |
| wallmaster | 1:0x45 | 0.78 | 0.63 | 0 | +0.32 (+0.63) |
| d2-rope | 2:0x4e | 1.25 | 0.65 | **0.40** | −0.38 (−0.23) |
| d2-rope | 2:0x3e | 0.45 | 0.20 | 0.15 | −1.05 |
| d2-rope | 2:0x2e | 0.95 | 0.50 | 0.05 | −0.75 (−0.57) |
| ow-octorok | 0:0x4c | 0.12 | 0.02 | 0.10 | −0.45 |
| ow-zora | 0:0x38 | 0.07 | 0 | 0.07 | −0.26 |

Totals: dungeon models take 55–80% of the hearts that drop (the rest are left or time out during long fights);
the overworld models take 8 of 34 and leave 26. The wallmaster model picks up nearly everything, often at full
health (total +0.63 vs real +0.32). Hearts are already worth a lot: success pays 20 × (0.5 + 0.5 × hp/max), so a
heart at the exit adds 2.5–3.3, plus `reward-gained-health`. But PBRS never points at a heart: dropped items are
added to the objective's `targets`, not to `pbrs_targets`, so the shaping always pulls toward the exit.

## 9. Second route past the fairy pond (Lee): `milestone1-fairy`

After D1: 0x38 S 0x48 S 0x58 E 0x59 N 0x49 N 0x39 (fairy pond) S 0x49 S 0x59 E 0x5a E 0x5b E 0x5c E 0x5d N
0x4d W 0x4c, then as milestone1 (`docs/route.md`; mechanics in `docs/decisions.md`). 40 entry states built and
audited (chains 0–2). Baseline chain with the old models (40 runs from `0_37c_2`): every run left 0x38 north
(41 exit-wrong), so no run reached the pond. The overworld models followed the room, not the objective's exit
bit, which they had never seen change there.

**r24-fairy** (from the promoted models, `--stop-success 0.99`, 2M steps; ow-start early stop at 115k):

| specialist | before (current yaml) | r24 | new rooms |
|---|---|---|---|
| ow-zora | 42%, −0.55 | **97%, −0.15** | 0x59 12% → 100%, 0x5a 0% → 98%; 0x38 (both routes) 68% → 94% |
| ow-octorok | 77%, −0.17 | **98%, −0.09** | 0x49 43% → 98%, 0x58 from N 57% → 99%; 0x4c 83% → 92% |
| ow-start | 70% | **98%** | pond 0x39 2% → 98%, +0.93 hearts per visit |
| ow-moblin | 18% | 28% (stopped) | 0x5b / 0x5c / 0x5d 0% |

Promoted: ow-zora, ow-octorok, ow-start r24-fairy. ow-moblin collapsed into 2000-step timeouts (1–4 episodes
per iteration). Lee pointed at the cause: a Moblin room with two tree openings, one leading E and the other S.
0x5b's east edge has two openings. The lower one opens into a pocket of 0x5c that only leads south, yet the
objective counted every tile of that edge as the exit to 0x5c, and a wavefront-greedy walker from the west entry
took exactly the lower one. Also, `Room.can_move` let Link's 2×2 top-left tile run along the off-map last row,
so the wavefront joined every south opening under the bottom tree line and made the pocket look connected to
0x5c's east exit. Fixes (`docs/decisions.md`): game.yaml exit `span` (0x5b E = rows 0–9) used by the objective,
end conditions, critic and runner; no off-map lane. An audit of every overworld entry state on both routes
found no other split edge. ow-moblin now has curriculum (12 new trajectories, 12–70 steps) and `max_steps` 600.
r26 retrains it from r9a.

## 10. Heart objective (`collect_hearts`)

Option (off by default): below max hearts, a dropped heart on the floor makes the objective ITEM (a free
observation slot) and the only PBRS target; exits stay correct. A/B on stalfos-keese-gel, both arms from r22,
1M steps, each evaluated under its own config:

| arm | eval | health real (total) | hearts dropped / taken / left per episode | 0x53 | steps |
|---|---|---|---|---|---|
| r25-skg-plain | 98% | −0.09 (−0.08) | 0.09 / 0.06 / 0.03 | 92%, −0.45 | 31 |
| r25-skg-hearts | 98% | −0.14 (−0.13) | 0.06 / 0.03 / 0.02 | 98%, −0.58 | 24 |

The objective did not raise pickups (half the drops vs two thirds) and the arm rushed exits (fewer kills, fewer
drops). r25-skg-plain beats r22 (−0.09 vs −0.16 hearts; 0x53 −0.45 vs −0.69) and is promoted.

Second test, d2-rope (arms from r20, 1M steps; D2 chain from `2_7ds_2`, 80 runs per arm, each arm's own config):

| arm | eval | health real (total) | dropped / taken / left per episode | D2 chain: reach 0x1e | Dodongo room |
|---|---|---|---|---|---|
| r27-rope-plain | 92% | −0.35 (−0.29) | 0.52 / 0.25 / 0.21 | 20 | **16/80** |
| r27-rope-hearts | 89% | −0.36 (−0.29) | 0.51 / 0.31 / 0.12 | 26 | 12/80 |

Here the objective did what it says (hearts left behind halved; in 0x4e 0.45 → 0.05 per episode) and more runs
got through the rope rooms, but health didn't improve and they died at 0x1e instead (12 deaths vs 3). **Dropped**
after two tests (code removed). r27-rope-plain is promoted (same success as r20 within noise, −0.35 vs −0.46
hearts).

ow-moblin after the fixes (§9): r26 trained without its new trajectories (`explore --out` takes the trajectories
root and appends the specialist; they landed in `trajectories/ow-moblin/ow-moblin`). r26b from r9a with them:
r9a's habit is to leave Moblin rooms W (0x4d), and the new rooms are entered from the W edge, so its first move
was an instant exit-wrong; it unlearned W and then stood at the edge (the direction head never picked E). r26c
from ow-octorok r24 instead (zero-shot on ow-moblin's rooms: ow-octorok 36%, ow-zora 40%, r9a 16%; ow-octorok
best on the mazes): a dip to ~10% while it relearned (action-kind entropy up to 0.96), then 97%.

| room | r9a | r26c |
|---|---|---|
| 0x3d | 95%, −0.55 | 95%, −0.60 |
| 0x4d | 38% (both entries) | 98% |
| 0x5b / 0x5c / 0x5d | 0% | 97% / 97% / 98% |

Promoted.

## 11. Chains on both routes

Promoted set: overworld r24-fairy / r26c (ow-moblin) / r23 (ow-leever-peahat), stalfos-keese-gel r25, goriya r21,
wallmaster r10, boss-aquamentus r23, d2-rope r27. 40 runs per segment:

| segment | milestone1 | milestone1-fairy |
|---|---|---|
| full chain: reach D2 entrance (hearts there) | 5 (2.20) | 8 (3.06) |
| full chain: Dodongo room | 0/40 | **3/40** |
| post-D1 (`0_37c_2`): reach D2 entrance (hearts there) | 27 (1.69) | **37 (2.46)** |
| post-D1: Dodongo room | 0/40 | **2/40** |
| D2 (`2_7ds_2`, full hearts): Dodongo room | 7/40 | (same rooms) |

Every pond visit refilled Link (40 of 40 post-D1, arriving with 3.2 hearts, leaving with 4.0). The fairy route is
now the better road to D2. What ends runs: the D1 Goriya room (15 deaths of 28 arrivals) and 0x53 (5) in the full
chain; in D2 the first Goriya room 0x5e (11 deaths post-D1 on the fairy route), then the rope rooms.

## 12. Three chains of Dungeon 2 states (r28)

D2 specialists had one start state per room (chain 2 only). New: chains 0 and 1 from `0_4ce_0/1` through D2
(`entry_states build --start-step 33`; 22 states incl. `0_3cs` and `2_0es`, audited clean), and goriya
trajectories for the new 0x5e and 0x1e states (0x1e needed 8000 Go-Explore iterations: all 5 Goriyas must die
before the N shutter opens). r28 continued both D2 families for 2M steps on the 3× state set; evaluated against
the promoted models on the same states:

| specialist | promoted | r28 | notes |
|---|---|---|---|
| d2-rope | r27: 90%, −0.44 | **95%, −0.29** | up in every room; 0x3e takes more damage (−1.14 vs −0.77) |
| goriya | r21: 74%, −0.96 | 69%, −0.89 | route rooms equal with less damage; off-route 0x23 77% → 57% |

d2-rope r28 promoted. goriya decided by chain (40 runs each, d2-rope r28 in both):

| goriya | D2 (`2_7ds_2`) → Dodongo room | fairy post-D1 (`0_37c_2`) → Dodongo room |
|---|---|---|
| r21 (kept) | **9/40** | **8/40** |
| r28 | 6/40 | 3/40 |

With d2-rope r28 the fairy route's post-D1 segment went from 2/40 to **8/40**. Full fairy-route chain with the
final set: 2/40 (r27 set: 3/40; same seeds, the same 8 runs leave D1 in both). What limits the full chain is
still Dungeon 1: the D1 Goriya room kills 15 of the 28 runs that reach it, 0x53 five more. In D2 the deaths
spread over 0x1e, 0x3e, 0x4e, 0x5e and 0x2e.

Promoted at the end of round 6: ow-start / ow-zora / ow-octorok r24-fairy, ow-moblin r26c, ow-leever-peahat
and boss-aquamentus r23, stalfos-keese-gel r25, goriya r21, wallmaster r10 copy, d2-rope r28.
