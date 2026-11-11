//! The checked-in power-on idle states (`triforce/states/power_on/idle_<frames>.state`) are the machines the
//! emulator reaches by idling on the file-select screen.  A failure means the state format or the emulator's
//! behaviour changed: regenerate them with `python -m triforce.entry_states idle`.

mod common;

use zelda::power_on::{IDLE_STATES, IDLE_STEP, idle, idle_state_name, to_file_select};

#[test]
fn the_idle_states_are_the_machines_idling_reaches() {
    let (Some(rom), Some(mut nes)) = (common::rom(), common::open_core()) else { return };
    to_file_select(&mut nes).unwrap();
    for index in 1..=IDLE_STATES {
        let frames = index * IDLE_STEP;
        idle(&mut nes, IDLE_STEP);
        let path = common::states_dir().join(format!("{}.state", idle_state_name(frames)));
        let saved = std::fs::read(&path).unwrap_or_else(|e| {
            panic!("{}: {e} (write it with `python -m triforce.entry_states idle`)", path.display())
        });
        let live = zcore::state_file::encode(&rom, &nes.snapshot(), 0);
        let same = zcore::state_file::same_machine(&rom, &saved, &live).unwrap_or_else(|e| {
            panic!("{}: {e} (regenerate it with `python -m triforce.entry_states idle`)", path.display())
        });
        assert!(same, "{} is not the machine after {frames} idle frames: regenerate it", path.display());
    }
}
