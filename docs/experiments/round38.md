# Round 38: one network on the corrected route, copied clean, then reinforced

The owner's program (`backlog.md` Track M): one network that beats the corrected `d3first`, judged by one run in
10,000 that completes the game without a restart; combat that looks skilled (hits taken a room visit and hearts lost
are reported before the success rate); never trained on `d2first`'s route or states. Round 37 left a network copied
on data that held `d2first` states (product of its 18 segment rates 2.3e-7 against the 51's 6.7e-6 and the
target's 1e-4). This round: the copy redone on `d3first` states alone, two shapes compared on the chains of the
dungeons that cost most, then reinforcement on top, one dungeon at a time.

## The data: the 51 teachers on `d3first` states only

`scripts/experiments/r36_merge_record.py --state-route d3first --decisions 20480`: every key of the team (34
families, 17 `family@d3first` route parts), from its training resets (its task yaml on the route's states alone,
`lab.collect.route_only`) and from the corrected route's harvested room entries (`d3first_*`, less the bench's): 97
sources, 20,480 decisions each, 1,986,560 rows, every row with observation schema 7's keys. Each source's
`meta.json` now lists the start state of every episode it played (`start_states_used`), the manifest the check reads.

**A leak the manifest found.** In the first record, 1,280 episodes in 21 of the reset sources started from
bare-named harvested states, the `d2first` route's (boss-gleeok and its `@d3first` part, boss-manhandla and its
part, darknut and its part, gibdo@d3first, goriya@d3first, lanmola, like-like and its part, moldorm, ow-lynel 293 of
them, ow-zora and its part, patra, vire and its part, wallmaster and its part, wizzrobe). The cause:
`StartStatePool::scan_harvested` (`crates/env/src/start_states.rs`) drew every file of the family's harvested
folder at `harvested_fraction` and ignored `state_routes:`, so `--state-route d3first` kept the seed states and the
curriculum on the route but not the harvested starts. **Round 36's students and round 37's reinforcement arms drew
`d2first`'s harvested states for this reason too**, not only through the `d2first`-only families. Fixed (the pool
draws only its namespaces' files; a family without `state_routes:` draws every route's as before; Rust test
`pool_draws_only_its_namespaces_harvested_states`; on `main` since). The 21 sources were recorded again with the fix.

**Cost.** The first record ran 83 minutes once it had the machine (about 2 h 20 min in the queue first); the 21
sources again, 9 minutes (22 queued).

## The students by copying, two shapes

Recipe as round 36 (`r36_merge_distill.py train`, 20 epochs, value weight 0.02), first on the 76 clean sources
(1,556,480 rows) while the 21 re-recorded ones waited in the queue, then 5 epochs from there (lr 3e-4) on all 97
(the `-f` students, where the reinforcement starts):

| student | shape | parameters | held-out KL (76 sources) | joint top-1 | bench vs the 51 (6,120 pairs) | hearts lost an episode | hits a room visit (the 51: 0.677) |
|---|---|---:|---:|---:|---:|---:|---:|
| s1x | today's (`ZeldaPolicy`), 1x | 866k | 0.125 | 0.869 | 0.0 ± 0.3 points (92.9%) | -0.001 ± 0.015 | 0.662 |
| w768e2 | today's, 768x3, two entity layers (converted, Rust path) | 4.01M | 0.088 | 0.898 | **+1.2 ± 0.3** (94.1%) | **-0.057 ± 0.015** | 0.623 |
| h768 | `hitnet`: Link's token over the monster tokens (type, features, `entity_rel`, `entity_contact`), 768x3 towers, a dormant hit head (served torch path) | 4.69M | 0.089 | 0.895 | **+1.4 ± 0.3** (94.3%) | **-0.083 ± 0.015** | **0.591** |

The two large shapes copy equally well (held-out KL 0.088 / 0.089), but the shape that sees contact plays the bench's fights with fewer hits: h768 against w768e2, paired, success +0.2 ± 0.3 points, hearts lost -0.025 ± 0.011 an episode, hits -0.032 ± 0.010 a visit (Darknuts -0.37 ± 0.10, Gleeok -0.31 ± 0.15; Goriyas' success -4.2 ± 1.8). Its bench took 48 minutes on the served path against 14 for w768e2 on the Rust actor (both with queue waits on a busy machine). The bench shortlists; the chains decide; the clean data copies better than round 36's
(its 768x3 two-entity-layer student: 0.116). No correction round was run: each is a 2-million-decision record,
hours in the shared queue; the reinforcement's teacher term on the student's own states does its work. The chains
on Dungeons 4, 7, 3, 5 and 8 (1,000 runs each, paired with the 51's chains of record) for w768e2 and h768 are queued.

**The round-1 student on the chains: a warning, not a verdict.** w768e2-z (76 sources) on 1,000 runs, against the
51 (hits a visit / hearts lost a visit / success): D4 0.626 / 0.436 / 1.0% (the 51: 0.354 / 0.236 / 18.5%), D7 1.111 /
0.839 / 2.1% (13.1%), D3 0.587 / 0.402 / 0.0% (23.2%), D5 0.621 / 1.141 / 0.1% (31.2%), D8 2.425 / 2.654 / 0.0% (40.0%
of 300). The runs die where the 21 missing sources' families fight: Manhandla 3:0x4d 1,000 lost of 1,000 entering,
the Olive Garden's Darknuts 5:0x05 999, the Blue Darknuts 8:0x3e / 0x3f 982 / 1,000, 3:0x59's Darknuts 781, the
like-like 4:0x32 628 and vire 4:0x31 339. The 21 sources the round-1 students never saw were exactly the reset tasks
of those families (boss-manhandla, darknut, boss-gleeok, vire, like-like and their route parts); the bench's
entries do not show it because they are other states. The shapes must be compared on the `-f` students (all 97
sources); the h768 round-1 chains still pair the two shapes on the same handicap.
