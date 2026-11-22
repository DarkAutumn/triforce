# Round 36: fewer networks, toward one

## The lab

The machinery the round's experiments run on (`docs/lab.md`): any torch network playable by the same evaluations
and chains, runs over several families at once with today's networks as fading teachers, and a fixed yardstick
(`python -m triforce.lab.bench`). Today's networks play as before: 100 chain runs of `d3first` from `3_7cs_0`
(seeds 0-99) give every run's record identical, field for field, between main's module and the lab's (1,847 room
records, 62,586 decisions).

**What the lab now has.** Each decision's hearts lost and monster hits in the rollout and losses on a network's
named outputs (`--aux OUTPUT=TARGET:COEFF`); the policy's next K candidates tried in the emulator on a share P of
the training decisions (`--try-others K:P`, labels for actions not taken; training only: a network trained with it
plays from the observation alone); a hit head inside the network (`--arch hithead`); networks with memory
(`--arch gru`, `--bptt`); rows per family by need (`--sampling need`); towers of any depth on the Rust actor
(`tower_depth`); `--demos`, `--stop-success`, `--checkpoint-every/--checkpoint-keep` in lab runs; each wood-sword
run's own task (`specialists/d3first/<family>.yaml`, rebuilt and checked against each run's `run.json`: the
original yamls were lost with their worktrees); `bench manifest --route`; `bench speed` for any checkpoint and half
precision.

**Speed** (`docs/lab.md` "How an experimental network acts"; the machine at load 40-45). A served network takes
1.6-2.9× the Rust actor's time a decision (217-469 decisions a second on the GPU against 580-1,191), more envs
helping it most; a network four times today's size costs no more a decision; half precision never pays.

**The look-ahead's cost** (`docs/lab.md` "Auxiliary targets"; goriya's `d3first` rooms, 8 envs). `--try-others
4:0.25` emulates 14.7 more frames a decision (1.8× the frames) with no measurable slow-down; `8:1.0` 117 more
(7.6×), the rollout 3.5× slower.

**Smoke of the hit head** (backlog Track M, experiment 6; one run, one seed). A `hithead` network from scratch on
goriya's `d3first` rooms, `--aux hit=hit:0.5 --try-others 4:0.25`, goriya's route run (`r32d1-goriya`) as the
teacher (KL weight 1.0 for 30 iterations, fading to 0 over 50), 80 iterations on 8 envs (327,680 decisions, 18
minutes):

```
python -m triforce.lab.train --name hit --run-id r36hit-smoke --families goriya@d3first --parallel 8 \
    --iterations 80 --arch hithead --aux hit=hit:0.5 --try-others 4:0.25 --teacher-coeff 1.0 --teacher-hold 30 \
    --teacher-fade 50 --seed 36
```

- The hit loss falls: 0.44 a label over iterations 1-10, 0.15 over 36-45, 0.13 over 71-80 (about two labels a
  decision: the action taken and one tried candidate). Training success 6% → 48% → 90% over the same windows.
- Held out (fresh rollouts of the trained network on the same rooms, seed 777, 4,096 decisions, the action taken
  and the network's next 4 candidates each tried: 20,480 labels, 4.1% of them hits): **precision = recall 0.10**,
  AUC 0.65 (the action taken: 0.09 and 0.66; the tried candidates: 0.11 and 0.65). Weak: the head ranks hits a
  little above chance; the guard prototype read the planner's verdicts at AUC 0.95 (backlog experiment 2).
- On the yardstick against goriya's route run (120 paired episodes): success **−5.0 ± 2.0** points, hearts lost
  **+0.70 ± 0.10** an episode, hits **+1.19 ± 0.14**, decisions +16.2 ± 2.7. Eighty iterations from scratch are
  not enough to match a teacher trained for thousands; the smoke shows the pieces train, play and report.

**Smoke of a network with memory** (one run, one seed). A `gru` network from scratch on zol's and
stalfos-keese-gel's `d3first` rooms, their route runs as teachers (KL weight 1.0 for 20 iterations, fading to 0
over 40), sequences of 16, 60 iterations on 8 envs (245,760 decisions, 47 minutes):

```
python -m triforce.lab.train --name mem --run-id r36gru-smoke --families zol@d3first stalfos-keese-gel@d3first \
    --parallel 8 --iterations 60 --arch gru --arch-config '{"hidden_dim": 256, "core_dim": 256}' --bptt 16 \
    --teacher-coeff 1.0 --teacher-hold 20 --teacher-fade 40 --seed 36
```

It trains (training success over iterations 1-10, 26-35, 51-60: zol 24% → 41% → 70%, stalfos-keese-gel 17% → 50%
→ 89%; the teacher KL 1.92 → 1.39 → 0.90), plays served with its state kept per env, and reports: on the yardstick
against the teachers (240 paired episodes) success **−5.8 ± 1.5** points (zol −8.3 ± 2.5, stalfos-keese-gel
−3.3 ± 1.6), hearts lost **+0.33 ± 0.05** an episode, hits +0.51 ± 0.07. Like the hit head, too short to judge
memory itself.
