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
