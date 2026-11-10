//! `Explorer::set_stop`: a set stop flag ends `explore` at its next iteration with an error (F155: Ctrl-C).

mod common;

use common::{TfzStore, open_core};
use env::EnvError;
use env::explore::Explorer;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

#[test]
fn a_set_stop_flag_ends_the_search_at_once() {
    let Some(nes) = open_core() else { return };
    let data = common::game_data();
    let config = data.specialists["ow-zora"].clone();
    let mut explorer = Explorer::new(&config, data.clone(), nes, TfzStore::new(), 0).unwrap();
    let stop = Arc::new(AtomicBool::new(true));
    explorer.set_stop(Some(stop.clone()));
    let t0 = Instant::now();
    let result = explorer.explore("0_38n_0", 1_000_000, 25, 1_000_000, None);
    assert!(matches!(&result, Err(EnvError::State(m)) if m.contains("interrupted")), "{result:?}");
    assert!(t0.elapsed() < Duration::from_secs(5));
    // Cleared, it searches again.
    stop.store(false, Ordering::Relaxed);
    assert!(explorer.explore("0_38n_0", 5, 25, 5, None).is_ok());
}
