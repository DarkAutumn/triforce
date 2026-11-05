# zcore

A NES core for *The Legend of Zelda* (USA), subtracted from TetaNES's `tetanes-core` one verified
stage at a time (see `UPSTREAM.md`). It reproduces FCEUmm (libretro `fceumm`, commit `e0f3086`)
frame for frame, including the FCEUmm behaviours listed in `FCEUMM_QUIRKS.md`.

The supported interface is [`ZCore`](crate::ZCore) with [`Rom`](crate::Rom),
[`MachineState`](crate::MachineState) and the [`state_file`](crate::state_file) format:

```no_run
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let rom = zcore::Rom::from_ines(&std::fs::read("roms/zelda.nes")?)?;
let mut nes = zcore::ZCore::power_on(rom);
for _ in 0..600 {
    nes.run_frame(0, false); // controller byte, render this frame?
}
let state = nes.snapshot();
nes.run_frame(0x08, true); // START
let _pixels = nes.pixels().expect("rendered"); // 256x240 palette indices
nes.restore(&state);
# Ok(())
# }
```

Everything else in the crate is the emulator behind that interface and changes with the ladder.

Licensed MIT OR Apache-2.0, like TetaNES (`LICENSE-MIT`, `LICENSE-APACHE`).
