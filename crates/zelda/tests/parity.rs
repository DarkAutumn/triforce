//! Session parity with the Python game stack on the FCEUmm oracle (env parity fixtures, plan step 10).
//!
//! For every `crates/env/tests/fixtures/parity/*.npz`: restore the recorded room-entry oracle state, load it as
//! the recorder did (`randomize_rng=False, fresh_room=True`, the specialist's disabled kinds), restore a snapshot
//! at every episode start, replay the recorded actions and require exact frames, interrupts, control frames,
//! shield blocks, change fields, masks, RAM, tile grid, objectives, PBRS wavefront, linger and heart counts
//! (floats within 1e-6).  A difference is allowed only when `tests/parity_exceptions.toml` lists it.

mod common;

use common::{Npy, entry_state, fixtures, open_oracle, read_npz, repo};
use rand::SeedableRng;
use std::collections::HashMap;
use std::sync::Arc;
use zelda::enums::{ActionKind, Direction, MapLocation, item};
use zelda::game::GameView;
use zelda::hearts::{HeartTracker, HeartView};
use zelda::linger::{LingerState, LingerStep, LingerTracker, linger_wavefront};
use zelda::objectives::{GameCompletion, Objective};
use zelda::room::{Target, Wavefront};
use zelda::room_state::decode_room_state;
use zelda::tiles::build_grid;
use zelda::{Action, GameMap, LoadOptions, Session};

#[derive(serde::Deserialize)]
struct Exceptions {
    #[serde(default)]
    exception: Vec<Exception>,
}

#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct Exception {
    room: String,
    step: i64,
    field: String,
    reason: String,
}

/// `reward.pbrs` and `reward.nearby_items` of specialists/<name>.yaml.
fn reward_flags(specialist: &str) -> (bool, bool) {
    let path = repo()
        .join("specialists")
        .join(format!("{specialist}.yaml"));
    let text = std::fs::read_to_string(&path).unwrap();
    let v: serde_yaml_ng::Value = serde_yaml_ng::from_str(&text).unwrap();
    let flag = |name: &str, default: bool| -> bool {
        match v.get("reward").and_then(|r| r.get(name)) {
            Some(serde_yaml_ng::Value::Bool(b)) => *b,
            Some(serde_yaml_ng::Value::String(s)) => s == "on" || s == "true",
            _ => default,
        }
    };
    (flag("pbrs", true), flag("nearby_items", false))
}

/// The env's `_attach_objectives`: the objective plus the PBRS wavefront.
fn attach(
    gc: &mut GameCompletion,
    prev: Option<&GameView>,
    state: &GameView,
    pbrs: bool,
    nearby: bool,
) -> (Objective, Option<Arc<Wavefront>>) {
    let mut objective = gc.objectives(prev, state);
    if nearby {
        for it in &state.items {
            if it.id == item::HEART || it.id == item::CLOCK {
                objective
                    .pbrs_targets
                    .extend(it.link_overlap_tiles().into_iter().map(Target::Tile));
            }
        }
    }
    let wavefront = (pbrs && !state.game_over()).then(|| {
        let locked: Vec<Direction> = if state.keys() > 0 && state.level() > 0 {
            [Direction::N, Direction::S, Direction::E, Direction::W]
                .into_iter()
                .filter(|&d| state.is_door_locked(d))
                .collect()
        } else {
            Vec::new()
        };
        let targets: Vec<Target> = objective.pbrs_targets.iter().copied().collect();
        Arc::new(state.room().wavefront(&targets, &[], &locked))
    });
    (objective, wavefront)
}

struct Checker<'a> {
    room: String,
    exceptions: &'a [Exception],
    failures: Vec<String>,
}

impl Checker<'_> {
    fn check<T: PartialEq + std::fmt::Debug>(&mut self, step: i64, field: &str, got: T, want: T) {
        if got != want && !self.excepted(step, field) {
            self.failures.push(format!(
                "{} step {step} {field}: rust {got:?} != python {want:?}",
                self.room
            ));
        }
    }

    fn close(&mut self, step: i64, field: &str, got: f64, want: f64) {
        if (got - want).abs() > 1e-6 && !self.excepted(step, field) {
            self.failures.push(format!(
                "{} step {step} {field}: rust {got} != python {want}",
                self.room
            ));
        }
    }

    fn excepted(&self, step: i64, field: &str) -> bool {
        self.exceptions
            .iter()
            .any(|e| e.room == self.room && e.step == step && e.field == field)
    }
}

fn row<T: Clone>(v: &[T], n: usize, i: usize) -> Vec<T> {
    v[i * n..(i + 1) * n].to_vec()
}

fn wavefront_grid(w: Option<&Arc<Wavefront>>) -> Vec<i64> {
    let mut out = vec![-1i64; 32 * 22];
    if let Some(w) = w {
        for x in 0..32 {
            for y in 0..22 {
                out[(x * 22 + y) as usize] = w.get((x, y)).map_or(-1, |d| d as i64);
            }
        }
    }
    out
}

fn run_fixture(
    nes: common::OracleNes,
    f: &HashMap<String, Npy>,
    map: &Arc<GameMap>,
    c: &mut Checker,
) -> common::OracleNes {
    let specialist = f["specialist"].string();
    let (pbrs, nearby) = reward_flags(&specialist);
    let route = f["route"].string();
    let route_step = f["route_step"].scalar_i64();
    let route_step = (route_step >= 0).then_some(route_step as usize);
    let disabled: Vec<ActionKind> = f["disabled_kinds"]
        .i64()
        .iter()
        .map(|&k| ActionKind::from_index(k as usize).unwrap())
        .collect();

    let mut session = Session::new(nes, map.clone());
    session.set_disabled_kinds(&disabled);
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let opts = LoadOptions {
        randomize_rng: false,
        fresh_room: true,
        ..Default::default()
    };
    session
        .load(&entry_state(f), &opts, &mut rng)
        .expect("load");
    let loc = MapLocation::new(
        f["level"].scalar_i64() as u8,
        f["room"].scalar_i64() as u8,
        f["in_cave"].bools()[0],
    );
    c.check(-1, "location", session.state().full_location(), loc);
    c.check(-1, "ram", session.state().ram().to_vec(), f["ram0"].bytes());
    c.check(
        -1,
        "mask",
        session.mask().to_flat().to_vec(),
        f["mask0"].bools(),
    );

    let steps = f["frames"].len();
    let actions = f["actions"].i64();
    let (episode_start, frames) = (f["episode_start"].bools(), f["frames"].i64());
    let (interrupt, control, blocks) = (
        f["interrupt"].i64(),
        f["control_frames"].i64(),
        f["blocks"].i64(),
    );
    let (lost, gained) = (f["health_lost"].f64(), f["health_gained"].f64());
    let (hit, stunned, items) = (
        f["enemies_hit"].i64(),
        f["enemies_stunned"].i64(),
        f["items_gained"].i64(),
    );
    let (changed, triforce) = (f["changed_location"].bools(), f["gained_triforce"].bools());
    let (masks, rams, grids) = (f["mask"].bools(), f["ram"].bytes(), f["obs_grid"].i64());
    let (terminated, truncated) = (f["terminated"].bools(), f["truncated"].bools());
    let (kinds, next_rooms, wavefronts) = (
        f["objective_kind"].i64(),
        f["next_rooms"].i64(),
        f["wavefront"].i64(),
    );
    let (tail, max) = (f["linger_tail_steps"].i64(), f["linger_max_steps"].i64());
    let hearts_rec = f["hearts"].i64();
    let block_events = f["block_events"].i64();
    let mut events: Vec<i64> = Vec::new();

    let mut gc = GameCompletion::new(map.clone(), &route, route_step).unwrap();
    let mut hearts = HeartTracker::default();
    let mut linger = LingerTracker::default();
    let mut objective = None;
    for i in 0..steps {
        let step = i as i64;
        if episode_start[i] {
            let snap = session.snapshot();
            session.restore(&snap);
            gc = GameCompletion::new(map.clone(), &route, route_step).unwrap();
            let state = session.state().clone();
            let (obj, wf) = attach(&mut gc, None, &state, pbrs, nearby);
            hearts.reset(&HeartView::of(&state));
            linger.reset(&LingerState::of(&state, &obj), || {
                linger_wavefront(&state, &obj, wf.clone())
            });
            objective = Some(obj);
        }
        let a = row(&actions, 3, i);
        let action = Action::from_indices(a[0] as usize, a[1] as usize, a[2] as usize).unwrap();
        c.check(step, "legal", session.mask().is_allowed(&action), true);
        let change = session.act(action).expect("act");

        c.check(step, "frames", change.frames as i64, frames[i]);
        c.check(
            step,
            "interrupt",
            change.interrupt.map_or(-1, |x| x.index() as i64),
            interrupt[i],
        );
        c.check(
            step,
            "control_frames",
            change.control_frames as i64,
            control[i],
        );
        c.check(step, "blocks", change.blocks() as i64, blocks[i]);
        c.close(step, "health_lost", change.health_lost, lost[i]);
        c.close(step, "health_gained", change.health_gained, gained[i]);
        let mut h = vec![0i64; 12];
        for &(slot, d) in &change.enemies_hit {
            h[slot as usize] += d as i64;
        }
        c.check(step, "enemies_hit", h, row(&hit, 12, i));
        let mut s = vec![0i64; 12];
        for &slot in &change.enemies_stunned {
            s[slot as usize] += 1;
        }
        c.check(step, "enemies_stunned", s, row(&stunned, 12, i));
        let mut it = vec![-999i64; 8];
        for (k, &id) in change.items_gained.iter().enumerate().take(8) {
            it[k] = id as i64;
        }
        c.check(step, "items_gained", it, row(&items, 8, i));
        c.check(
            step,
            "changed_location",
            change.changed_location(),
            changed[i],
        );
        c.check(
            step,
            "gained_triforce",
            change.gained_triforce(),
            triforce[i],
        );
        c.check(
            step,
            "mask",
            session.mask().to_flat().to_vec(),
            row(&masks, 91, i),
        );
        c.check(
            step,
            "ram",
            session.state().ram().to_vec(),
            row(&rams, 10240, i),
        );
        let state = change.state.clone();
        let grid: Vec<i64> = build_grid(&state, &decode_room_state(state.ram()))
            .iter()
            .flatten()
            .map(|&x| x as i64)
            .collect();
        c.check(step, "grid", grid, row(&grids, 22 * 32, i));
        for b in &change.projectile_blocks {
            events.extend([step, b.frame as i64, b.slot as i64, b.projectile_id as i64]);
        }

        // The env's step: hearts, objectives, linger.
        let taken = hearts.taken;
        hearts.update(&HeartView::of(&state), change.frames);
        let (obj, wf) = attach(&mut gc, Some(&change.previous), &state, pbrs, nearby);
        let prev_obj = objective.take().unwrap();
        linger.update(
            &LingerState::of(&change.previous, &prev_obj),
            &LingerState::of(&state, &obj),
            &LingerStep::of(&change),
            hearts.taken - taken,
            || linger_wavefront(&state, &obj, wf.clone()),
        );
        if terminated[i] || truncated[i] {
            hearts.finish(Some(&HeartView::of(&state)));
        }
        c.check(step, "objective_kind", obj.kind as i64, kinds[i]);
        let mut rooms = vec![-1i64; 12];
        for (k, r) in obj.next_rooms.iter().take(4).enumerate() {
            rooms[k * 3..k * 3 + 3].copy_from_slice(&[
                r.level as i64,
                r.value as i64,
                r.in_cave as i64,
            ]);
        }
        c.check(step, "next_rooms", rooms, row(&next_rooms, 12, i));
        c.check(
            step,
            "wavefront",
            wavefront_grid(wf.as_ref()),
            row(&wavefronts, 32 * 22, i),
        );
        c.check(
            step,
            "linger_tail_steps",
            linger.s.tail_steps as i64,
            tail[i],
        );
        c.check(step, "linger_max_steps", linger.s.max_steps as i64, max[i]);
        let hs = hearts.summary();
        c.check(
            step,
            "hearts",
            vec![
                hs.dropped as i64,
                hs.taken as i64,
                hs.left as i64,
                hs.expired as i64,
                (hs.healed * 2.0).round() as i64,
            ],
            row(&hearts_rec, 5, i),
        );
        objective = Some(obj);
        if c.failures.len() > 20 {
            break;
        }
    }
    c.check(-1, "block_events", events, block_events);
    session.into_nes()
}

#[test]
fn session_matches_python_fixtures() {
    let _lock = oracle::lock();
    let Some(mut nes) = open_oracle() else { return };
    let map = Arc::new(GameMap::builtin());
    let text =
        std::fs::read_to_string(repo().join("crates/zelda/tests/parity_exceptions.toml")).unwrap();
    let exceptions: Exceptions = toml::from_str(&text).unwrap();
    let files = fixtures();
    assert!(
        files.len() >= 14,
        "expected the 14 parity fixtures, found {}",
        files.len()
    );
    let mut failures = Vec::new();
    for path in files {
        let room = path.file_stem().unwrap().to_string_lossy().into_owned();
        let f = read_npz(&path);
        let mut c = Checker {
            room: room.clone(),
            exceptions: &exceptions.exception,
            failures: Vec::new(),
        };
        nes = run_fixture(nes, &f, &map, &mut c);
        eprintln!(
            "{room}: {} steps, {} differences",
            f["frames"].len(),
            c.failures.len()
        );
        failures.extend(c.failures);
    }
    assert!(
        failures.is_empty(),
        "{} differences:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
