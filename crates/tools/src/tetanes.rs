//! Stock/patched TetaNES (`../tetanes/tetanes-core`, branch `bugs`) as a `tf` backend. Dev-only:
//! removed once zcore is imported (plan step 9, Z0).

use crate::backend::{Backend, TraceRow};
use crate::formats::Domain;
use anyhow::{Context, Result};
use std::cell::RefCell;
use tetanes_core::control_deck::{Config, ControlDeck, HeadlessMode};
use tetanes_core::input::{JoypadBtnState, Player};
use tetanes_core::memory::{RamState, Src};
use tetanes_core::video::VideoFilter;

pub struct Tetanes {
    deck: ControlDeck,
}

impl Tetanes {
    pub fn open(rom: &[u8]) -> Result<Tetanes> {
        let mut cfg = Config::default();
        cfg.ram_state = RamState::AllOnes;
        cfg.sram_dir = None;
        cfg.filter = VideoFilter::Pixellate;
        cfg.headless_mode = HeadlessMode::NO_AUDIO;
        cfg.concurrent_dpad = true;
        cfg.fceumm_compat = true;
        let mut deck = ControlDeck::with_config(cfg);
        deck.load_rom("zelda.nes", &mut &rom[..]).context("TetaNES load_rom")?;
        // Power-on spec (TFMOVIE v1): an empty, zero-filled battery, as the oracle does.
        let len = deck.sram().len();
        deck.set_sram(&vec![0u8; len]);
        Ok(Tetanes { deck })
    }

    fn set_buttons(&mut self, buttons: u8) {
        self.deck.joypad_mut(Player::One).buttons = JoypadBtnState::from_bits_truncate(buttons as u16);
    }

    fn row(bus: &tetanes_core::bus::Bus) -> TraceRow {
        let c = &bus.cpu;
        TraceRow {
            pc: c.pc,
            opcode: bus.peek(c.pc),
            a: c.acc,
            x: c.x,
            y: c.y,
            p: c.status.bits(),
            sp: c.sp,
            cycle: c.cycle as u64,
            scanline: bus.ppu.scanline as i16,
        }
    }
}

impl Backend for Tetanes {
    fn name(&self) -> &'static str {
        "tetanes"
    }

    fn run_frame(&mut self, buttons: u8) {
        self.set_buttons(buttons);
        let _ = self.deck.clock_frame().expect("TetaNES clock_frame");
        if std::env::var_os("TF_DBG").is_some() {
            let bus = self.deck.bus();
            eprintln!(
                "frame {} boundary_dot {} sl {} cyc {} render {}",
                self.deck.frame_number(),
                bus.cpu.cycle as i64 * 3 - (bus.ppu.cycle as i64 - 1),
                bus.ppu.scanline,
                bus.ppu.cycle,
                bus.ppu.mask_rendering_enabled
            );
        }
    }

    fn poke(&mut self, addr: u16, value: u8) {
        match addr {
            0x0000..0x0800 => self.deck.wram_mut()[addr as usize] = value,
            0x6000..0x8000 => self.deck.bus_mut().memory.region_mut(Src::PrgRam)[addr as usize - 0x6000] = value,
            _ => panic!("poke address ${addr:04X} not allowed"),
        }
    }

    fn read_domain(&mut self, domain: Domain, out: &mut Vec<u8>) -> bool {
        out.clear();
        let bus = self.deck.bus();
        match domain {
            Domain::Ram => out.extend_from_slice(self.deck.wram()),
            Domain::PrgRam => out.extend_from_slice(&bus.memory.region_ref(Src::PrgRam)[..0x2000]),
            Domain::Chr => out.extend((0..0x2000u16).map(|a| bus.ppu_bus_peek(a))),
            Domain::Ciram => out.extend_from_slice(&bus.memory.region_ref(Src::CiRam)[..0x800]),
            Domain::Palette => out.extend((0x3F00..0x3F20u16).map(|a| bus.ppu_bus_peek(a))),
            Domain::Oam => out.extend((0..256).map(|i| bus.ppu.oamdata[i])),
            Domain::Pixels => out.extend(self.deck.frame_buffer_raw().iter().map(|&p| (p & 0x3F) as u8)),
            _ => return false,
        }
        true
    }

    fn overshoot_dots(&self) -> Option<f64> {
        let fb: i64 = std::env::var("TF_FB_OFF").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
        let ppu = &self.deck.bus().ppu;
        Some((i64::from(ppu.scanline) * 341 + i64::from(ppu.cycle) - (240 * 341 + fb)) as f64)
    }

    fn trace_frame(&mut self, buttons: u8) -> Vec<TraceRow> {
        self.set_buttons(buttons);
        let rows = RefCell::new(vec![Self::row(self.deck.bus())]);
        let start = self.deck.frame_number();
        // The stop hook sees the state each instruction leaves behind, i.e. the next instruction.
        loop {
            let r = self.deck.clock_frame_until(|bus| {
                rows.borrow_mut().push(Self::row(bus));
                false
            });
            let _ = r.expect("TetaNES clock_frame_until");
            if self.deck.frame_number() != start {
                break;
            }
        }
        let mut rows = rows.into_inner();
        rows.pop(); // the first instruction of the next frame
        rows
    }
}
