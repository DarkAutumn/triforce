//! F168: a bootstrapped truncation (`no-damage`) through the real TrainEnv in a pool: its row's `bootstrap_value` is
//! the actor's value of the final observation, every other row's is 0. Skipped without the ROM
//! (`TRIFORCE_ALLOW_NO_ROM=1`).

use env::EnvError;
use env::StateStore;
use env::config::GameData;
use env::specialist_env::EnvOptions;
use pool::rollout::{Actor, Decision};
use pool::zelda::{ACTION_MASK, TrainEnv, ZeldaAction};
use pool::{ObsData, Pool, PoolOptions};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use zcore::{MachineState, Rom, ZCore};
use zelda::{Action, ActionMask, MASK_SIZE, SessionSnapshot};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn rom() -> Option<Arc<Rom>> {
    test_rom::bytes().map(|bytes| Rom::from_ines(&bytes).expect("valid Zelda ROM"))
}

struct TfzStore(Arc<Rom>);

impl StateStore<MachineState> for TfzStore {
    fn load(&self, name: &str) -> Result<MachineState, EnvError> {
        let path = repo().join(format!("triforce/states/{name}.state"));
        let bytes = std::fs::read(&path).map_err(|e| EnvError::State(format!("{}: {e}", path.display())))?;
        Ok(zcore::state_file::decode(&self.0, &bytes)
            .map_err(|e| EnvError::State(format!("{}: {e}", path.display())))?
            .machine)
    }

    fn save(&self, _: &Path, _: &SessionSnapshot<MachineState>) -> Result<(), EnvError> {
        Err(EnvError::State("read-only store".into()))
    }
}

/// WAITs; values every observation at `VALUE` (so a bootstrapped row shows it, a zero-value ending doesn't).
struct Waits;

const VALUE: f32 = 1.0;

impl Actor<ZeldaAction> for Waits {
    type Scratch = ();

    fn scratch(&self) {}

    fn act(&self, obs: &ObsData, _: i64, _: i64, _: i64, _: &mut ()) -> Result<(ZeldaAction, Decision), String> {
        let flat: [bool; MASK_SIZE] = obs.bool(ACTION_MASK).try_into().unwrap();
        let mask = ActionMask::from_flat(&flat);
        let action = Action::parse("WAIT 8").map_err(|e| e.to_string())?;
        assert!(mask.is_allowed(&action), "WAIT 8 is legal");
        Ok((action.to_indices().map(|i| i as i64), Decision::default()))
    }

    fn value(&self, _: &ObsData, _: i64, _: &mut ()) -> Result<f32, String> {
        Ok(VALUE)
    }
}

#[test]
fn a_no_damage_truncation_bootstraps_with_the_actors_value() {
    let Some(rom) = rom() else { return };
    let data = GameData::load(&repo()).expect("game data");
    let mut config = (*data.specialists["stalfos-keese-gel"]).clone();
    config.no_damage_steps = Some(3);
    config.max_steps = 100;
    let config = Arc::new(config);
    let opts = || EnvOptions {
        start_state: Some("1_73s_0".into()),
        ..Default::default()
    };
    let pool = Pool::new(PoolOptions::new(1, 1, 0), |init| {
        TrainEnv::new(
            config.clone(),
            data.clone(),
            ZCore::power_on(rom.clone()),
            TfzStore(rom.clone()),
            opts(),
            init,
            None,
        )
    })
    .unwrap();
    let (_, r) = pool.rollout(7, Arc::new(Waits)).unwrap().pop().unwrap();
    let cut: Vec<bool> = (0..7).map(|i| r.truncated[i] && !r.terminated[i]).collect();
    assert_eq!(cut, [false, false, true, false, false, true, false], "no-damage every 3 WAITs");
    for (i, &truncated) in cut.iter().enumerate() {
        let expected = if truncated { VALUE } else { 0.0 };
        assert_eq!(r.bootstrap_value[i], expected, "row {i}");
    }
}
