---
name: triforce-plan-task
description: Use when asked to work the next item from a Triforce living plan document, e.g. 'work on the next proposed experiment', 'do the next task', 'next item from the table', 'continue the dungeon 1 plan', or 'pick up the next blocking task'. Drives one plan task end to end. Selects it, executes it, decides whether the code lands, opens a PR and merges when green, then checks the item off either way.
---

# Triforce Plan Task Skill

Executes **exactly one** task from a living plan document, end to end, then stops and reports.

Default plan: `docs/experiments/dungeon1-completion-plan.md`. Any other plan doc with the same shape
(a `§0.6 TASK INDEX` table of `| [ ] | ID Title | Phase | Line |` rows plus per-task `###` sections
ending in `**Outcome:**`) works the same way; use it if the user names one.

**The contract:** one task per invocation. The item gets checked off and the plan doc lands **whether
or not the code lands**. Never chain into a second task — report and stop.

---

## Step 0 — Orient and verify a clean start

1. Read plan `§0` (all of it: mission, root cause, protocol, rules of engagement, key assets, index).
   It is written to be read every time.
2. Verify you are actually starting clean:

```bash
git status --short                 # must be empty
git log origin/main..HEAD          # must be empty
git branch --show-current
```

**Trap that has already bitten this project once:** a branch created but never committed to reports
`Your branch is up to date with 'origin/main'` — because it has zero commits of its own. That looks
identical to "work landed." `git log origin/main..HEAD` plus `git status --short` is the only honest
check. If you find uncommitted work from a previous session, **stop and report it** rather than
building on top of it or discarding it; ask whether to land or drop it.

3. Start from a fresh branch off `origin/main`:

```bash
git checkout main && git pull origin main
git checkout -b <task-id-lowercase>-<short-slug>      # e.g. p5-primary-exit-criteria
```

Never commit or push to `main`. Merge only via PR.

---

## Step 1 — Select the task

Unless the user named a specific task ID, apply this exactly:

1. Parse the `§0.6 TASK INDEX` rows. Status boxes: `[ ]` open, `[x]` done, `[~]` attempted/inconclusive,
   `[!]` falsified/abandoned. Only `[ ]` rows are candidates.
2. Read each candidate's `###` section heading for its dependency parenthetical and resolve it:
   - `(depends X)` → `X` must be `[x]`.
   - `(after A–B, before C)` → every task in `A..B` must be `[x]`.
   - `(piggybacks on X)` → `X` must be `[x]`; these are designed to be collected *during* `X`.
   - `(independent…)` / no parenthetical → no dependencies.
   - `(before any weighted run)` and similar → a constraint on *other* tasks, not a dependency of this one.
3. Among candidates with satisfied dependencies: take `⛔` rows first in index order, then the rest in
   index order.
4. **Re-read the chosen task's full section before committing to it.** Earlier outcomes may have
   already answered it. If existing recorded evidence fully satisfies its acceptance, do not re-run
   work — go to Step 5 and close it as `[x]` citing that evidence, or `[!]` if the task's premise is
   now void.

### EXPERIMENT-class tasks are gated — do not start one autonomously

Any index row whose title contains `EXPERIMENT` (currently R6, B4, I1, I2, I3) is a multi-hour to
multi-day training run. These consume GPU for days and cannot complete inside one session.

When the selected task is EXPERIMENT-class: **stop and ask the user** for go-ahead, wall-clock budget,
and baseline before launching. On approval, hand off to the `triforce-experiment` skill and follow the
mandatory experiment protocol (plan `§10.4`): read `docs/experiments/experiment-memory.md` and the most
relevant `docs/experiments/<id>-summary.md`, write `training/experiments/<id>/journal.md` **before**
starting, then launch via `triforce_experiment_start`. The checkoff happens when the experiment
*finishes*, not in the session that launches it.

If the user's intent was clearly "make progress now, don't burn days," offer the next non-EXPERIMENT
candidate instead and let them choose.

---

## Step 2 — Load the task's context

Read, in this order:
- The task's own `###` section (Goal / Hypothesis, Change, Acceptance, Falsification).
- Plan `§1` (the verified diagnosis) if you have not read it this session — it exists so you never
  re-derive the root cause.
- Plan `§10` reference card: commands, checkpoint paths, baseline eval JSONs, RAM overrides, demo format.
- For training tasks only: `docs/experiments/experiment-memory.md` (mandatory, `§10.4`).

Then read the actual source you intend to change. File:line references in the plan are load-bearing but
may have drifted — re-verify each before editing.

---

## Step 3 — Execute by task class

Do the task **completely**: code, tests, run, numbers. A task is not done because it compiles.

| Class | Shape | Proof required |
|---|---|---|
| **P** Tooling | Code + tests | Run the tool on a real checkpoint; show its output |
| **V** Probe | Script/diagnose run, often no production code | The verdict, with numbers, on the task's hypothesis |
| **R** Reward spec | Code + unit tests on the reward/ending logic | Tests, plus an accounting check on a real episode |
| **B** Demos / **C** Curriculum | Config/format/code | Validate with `diagnose.py --demo-report` or a short scenario run |
| **S** Structural | Code + tests | Targeted test; do not run a full training leg |
| **X** Backlog | Usually small | Task-appropriate |

Rules that apply to every class:

- **Grep for callsites wider than feels necessary.** `train.py`, `debug.py`, `record.py`, `diagnose.py`,
  `triforce_debugger/`, `tests/`, `scripts/`. A signature change already slipped past a
  `triforce/`-only grep in this repo and was caught only by pylint `E1120`. Run pylint before believing
  a refactor is complete.
- **Never call `evaluate.py` directly for a number you will record.** Use `triforce_evaluation_start`
  (see the `triforce-evaluation` skill) and end the turn; the plugin wakes you. The single narrow
  exception: smoke-testing a change *to `evaluate.py` itself* with a handful of episodes — that is a
  code test, and say so when reporting it.
- **Anything long-running goes in the background**, never a blocking `bash` call: use `hub`
  (`op:"start"`) for local runs, or the evaluation/experiment plugins. A 100-episode late-chain eval is
  ~55 minutes.
- **One emulator instance per process.** `close()` before creating another.
- Keep diagnostics general-purpose (any room, any enemy) — never one-off hacks. Per
  `.github/copilot-instructions.md`, `diagnose.py` improvements are persistent tools.

---

## Step 4 — Decide whether the code lands

Two separate decisions. **The plan-doc update always lands. The code is judged on its own merit.**

### Land the code when
- It is a durable improvement the repo wants regardless of how the hypothesis turned out: tooling,
  diagnostics, tests, bug fixes, a script or demo fixture a later task needs.
- A spec/reward change whose acceptance criteria were met.

### Do not land the code when
- The hypothesis was **falsified** and the code existed only to test it → discard the code, keep the findings.
- Measurements got **worse** → discard, record the numbers so nobody retries it blind.
- It is throwaway scaffolding: temporary prints, hardcoded paths, commented-out experiments.
- It is a checkpoint or training artifact (`training/**` is gitignored — never force-add it).

### Land these even when the fix is discarded
Diagnostic and test improvements developed along the way. They keep their value independently, and
`copilot-instructions` explicitly wants `diagnose.py` improvements committed separately from the fix
being investigated. Split them into their own commit.

If the code lands, **commit tooling/diagnostics separately from the behavior change** — one logical
change per commit, in one PR.

---

## Step 5 — Record the outcome in the plan doc (ALWAYS)

This happens on every path: landed, discarded, inconclusive, or falsified.

### 5a. Pick the status marker honestly

| Marker | Use when |
|---|---|
| `[x]` | Acceptance criteria met. **A probe that refuted its own hypothesis is `[x]`** — the task was "find out," and you found out. |
| `[~]` | Ran it, but the acceptance question is still open. Must state exactly what remains and what would settle it. |
| `[!]` | The task's *premise* is void or it is no longer worth doing. Must say why and what supersedes it. |

Do not use `[x]` for partial work, and do not use `[!]` merely because a hypothesis was disproven.

### 5b. Replace the `**Outcome:** _not attempted_` line

Use this shape (mirrors the P1/P2 entries already in the doc):

```markdown
**Outcome:** ✅ **DONE <YYYY-MM-DD>.** <One line: what was implemented/run, PR # and commit if landed,
or "code discarded, see below" if not.>

<What changed or what was run — specifics, not narration.>

**Numbers:** <the measurements, as a table when comparing. Never "improved" without figures.>

**Verdict:** <direct answer to the task's Hypothesis/Acceptance, in one sentence.>

**What the next agent must know:** <numbered. Traps hit, invalidated assumptions, tasks this
unblocks or makes pointless. This is the highest-value part of the block — write it for someone
who has none of your context.>
```

Rules: dated; every claim carries its number; state anything that changes another task's premise. If
you discarded the code, say so explicitly and say why, so the next agent does not assume it exists.

### 5c. Append to `§9 Decision Log` only if strategy changed

A dated entry when your result changes what the next agent should *believe* or *do* — a falsified
phase, a re-ordered dependency, a ruled-out branch. Routine completions do not need one.

### 5d. Update the index row and refresh line numbers

Flip the status box. Then refresh the `Line` column, which drifts as outcome blocks grow, and verify:

```bash
grep -n '^### ' docs/experiments/dungeon1-completion-plan.md      # actual heading positions
```

```bash
python -c "
import re
lines=open('docs/experiments/dungeon1-completion-plan.md').read().split('\n')
heads={}
for i,l in enumerate(lines,1):
    m=re.match(r'^### ([A-Z]\d+)\.',l)
    if m: heads[m.group(1)]=i
bad=0
for l in lines:
    m=re.match(r'^\| \[([ x~!])\][^|]*\| ([A-Z]\d+) .*\| (\d+) \|\$', l)
    if m and heads.get(m.group(2))!=int(m.group(3)):
        print(f'  MISMATCH {m.group(2)}: index says {m.group(3)}, actual {heads.get(m.group(2))}'); bad+=1
print('all index line numbers correct' if not bad else f'{bad} mismatches')
"
```

Must print `all index line numbers correct` before you commit.

---

## Step 6 — Gate, PR, merge when green

### 6a. Run the gates locally first

Required before any PR touching `triforce/`. Match CI's environment so you do not get a green local run
and a red CI:

```bash
source .venv/bin/activate
SDL_VIDEODRIVER=dummy QT_QPA_PLATFORM=offscreen pytest
pylint triforce/ triforce_debugger/ debug.py evaluate.py train.py record.py
```

CI (`.github/workflows/python-tests.yml`, job `build (3.10)`) runs bare `pytest` with those env vars and
pylint without `record.py`. Running the superset locally is intentional. Pylint must be 10.00/10 —
this repo keeps it clean.

### 6b. Commit

Logical commits, imperative subject, body explaining *why*. Reference the task ID.

```bash
git add <specific files>          # never `git add -A`; training/** is gitignored, keep it that way
git commit
```

### 6c. PR and merge

```bash
git push -u origin <branch>
gh pr create --base main --head <branch> --title "<TASK-ID>: <what changed>" --body "<see below>"
gh pr checks <N> --watch --interval 20
gh pr merge <N> --squash --delete-branch
git checkout main && git pull origin main
```

PR body must contain: which plan task this is, what changed, the verification actually run (with
numbers), and any missed-callsite or trap worth flagging to a reviewer.

**Merge only when CI is green.** If CI fails: fix forward on the branch. If it cannot be fixed, close
the PR, and still land the plan-doc update with an honest `[~]` recording the failure.

### 6d. When the code is discarded

Still open a PR — with the plan-doc update (and any diagnostics worth keeping) only. Discard the rest:

```bash
git checkout -- <files>        # or restore selectively; never leave the tree dirty
```

Say plainly in the PR body that the investigated change was discarded and why. A negative result that
is written down is a real contribution; an unrecorded one costs the next agent the same days.

### 6e. Verify it actually landed

```bash
git checkout main && git pull origin main
git log --oneline -3
git status --short                    # clean
```

Confirm your commit is in `main`'s history. Do not report success off the PR page alone.

---

## Step 7 — Report and stop

Report: task ID and title, what was done, the numbers, land-or-discard decision and why, PR number and
merge status, the status marker you set, and what this unblocks. Then **stop** — one task per
invocation. Do not begin the next task.

---

## Definition of done

- [ ] Exactly one task executed, with proof appropriate to its class (not just "it compiles")
- [ ] Land/discard decided on the code's own merit; discarded code actually removed from the tree
- [ ] Diagnostics/tests kept even if the fix was discarded, in their own commit
- [ ] `**Outcome:**` block written: dated, numbers, verdict, what-the-next-agent-must-know
- [ ] Index status box flipped to `[x]` / `[~]` / `[!]` — honestly
- [ ] Index line numbers refreshed and the validator prints `all index line numbers correct`
- [ ] `§9 Decision Log` appended if strategy changed
- [ ] `pytest` and `pylint` green locally before the PR
- [ ] PR opened, CI green, squash-merged, branch deleted
- [ ] `main` pulled and confirmed to contain the commit; working tree clean
- [ ] Reported and stopped

## Guardrails

- Never commit or push to `main`; never force-push anything but your own feature branch.
- Never edit `.omp/extensions/` during an experiment; record extension pain points in the summary.
- Never commit anything under `training/**` (gitignored checkpoints and run artifacts).
- Treat untracked savestates under `triforce/custom_integrations/Zelda-NES/` as the user's work.
- Single-milestone readings are not evidence (plan `§0.4`): a claim of improvement needs ≥3 consecutive
  milestone readings or a 100-episode eval. A 1.0 micro-scenario eval says nothing about chain
  performance — read every gate on its own scenario.
- Prefer a probe that takes minutes over a run that takes hours whenever the probe can kill the idea.
- If blocked, say exactly what is missing and what you tried. Do not silently narrow the task.
