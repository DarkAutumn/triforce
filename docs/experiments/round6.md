# Round 6: enemy-based families, rooms toward the bow, pass-through damage

Lee (Nov 3): rooms are rooms; a Red Goriya room needs the same skills in any dungeon. Split the Goriya rooms into
their own model, give Wallmasters their own model, name Dungeon 1's melee family by its enemies, and add the two
off-route Dungeon 1 rooms toward the bow. Also try a damage penalty for rooms Link only has to cross.

Setup as in round 5 (2M-step budget, balanced 95% early stop, 20 sampled episodes per seed state, hp
`[0.15, max]` with half the resets at full health). Chains are 20 runs unless noted.

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
for `d1-wallmaster`, 0x5e/0x1e 80/85% for `d2-goriya`). r19 continues it.

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
Leever rooms 2 → 4). Not adopted yet: configs back to no penalty, the option stays (off by default) for one more
test where the drain is larger (d2-rope).
