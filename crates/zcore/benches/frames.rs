//! Frames per second of [`zcore::ZCore`] on Zelda, rendering on and off.
//!
//! `cargo bench -p zcore` (needs the ROM: `$TRIFORCE_ROM`, else `roms/zelda.nes`). Plays `power_on_to_start`, then times
//! `ZCORE_BENCH_FRAMES` (default 20,000) frames of seeded random input with START and SELECT
//! masked, so the game keeps playing rather than pausing.

use std::{hint::black_box, time::Instant};
use zcore::{Rom, ZCore};

fn main() {
    let repo = test_rom::repo();
    let Some(bytes) = test_rom::bytes() else { return };
    let rom = Rom::from_ines(&bytes).expect("Zelda ROM");
    let movie = movie::Movie::load(&repo.join("corpus/movies/power_on_to_start.tfmovie"))
        .expect("power_on_to_start movie");
    let frames = std::env::var("ZCORE_BENCH_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20_000u32);

    for render in [false, true] {
        let mut nes = ZCore::power_on(rom.clone());
        for &buttons in &movie.inputs {
            nes.run_frame(buttons, false);
        }
        let mut seed = 1u64;
        let start = Instant::now();
        for _ in 0..frames {
            // xorshift64: cheap, deterministic input.
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            nes.run_frame(black_box(seed as u8 & !0x0C), render);
        }
        let fps = f64::from(frames) / start.elapsed().as_secs_f64();
        println!(
            "zcore: {fps:.0} frames/s (render {}, 1 core)",
            if render { "on" } else { "off" }
        );
    }
}
