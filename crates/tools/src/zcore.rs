//! zcore as a `tf` backend (the subtraction ladder, plan step 9), through the `ZCore` API; the
//! ladder-only domains read internals through `ZCore::bus`.

use crate::backend::{Backend, TraceRow};
use crate::formats::Domain;
use anyhow::{Context, Result};
use std::sync::mpsc;
use zcore::memory::Src;
use zcore::{Rom, ZCore};

pub struct Zcore {
    nes: ZCore,
    render: bool,
}

impl Zcore {
    pub fn open(rom: &[u8]) -> Result<Zcore> {
        let rom = Rom::from_ines(rom).context("zcore ROM")?;
        Ok(Zcore { nes: ZCore::power_on(rom), render: true })
    }
}

impl Backend for Zcore {
    fn name(&self) -> &'static str {
        "zcore"
    }

    fn run_frame(&mut self, buttons: u8) {
        self.nes.run_frame(buttons, self.render);
    }

    fn set_render(&mut self, on: bool) -> bool {
        self.render = on;
        true
    }

    fn poke(&mut self, addr: u16, value: u8) {
        self.nes.poke(addr, value);
    }

    fn read_domain(&mut self, domain: Domain, out: &mut Vec<u8>) -> bool {
        out.clear();
        let bus = self.nes.bus();
        match domain {
            Domain::Ram => out.extend_from_slice(self.nes.ram()),
            Domain::PrgRam => out.extend_from_slice(self.nes.prg_ram()),
            Domain::Chr => out.extend((0..0x2000u16).map(|a| bus.ppu_bus_peek(a))),
            Domain::Ciram => out.extend_from_slice(&bus.memory.region_ref(Src::CiRam)[..0x800]),
            Domain::Palette => out.extend((0x3F00..0x3F20u16).map(|a| bus.ppu_bus_peek(a))),
            Domain::Oam => out.extend((0..256).map(|i| bus.ppu.oamdata[i])),
            Domain::Pixels => match self.nes.pixels() {
                Some(px) => out.extend_from_slice(px),
                None => return false,
            },
            d => return crate::ladder_domains!(bus, d, out, zcore::mapper::Mapper::Sxrom),
        }
        true
    }

    fn overshoot_dots(&self) -> Option<f64> {
        // Frame boundary at line 240 dot 1 (FQ-08).
        let ppu = &self.nes.bus().ppu;
        Some((i64::from(ppu.scanline) * 341 + i64::from(ppu.cycle) - (240 * 341 + 1)) as f64)
    }

    fn trace_frame(&mut self, buttons: u8) -> Vec<TraceRow> {
        let (tx, rx) = mpsc::channel();
        self.nes.set_trace(Some(Box::new(move |r: &zcore::TraceRow| {
            let row = TraceRow {
                pc: r.pc,
                opcode: r.opcode,
                a: r.a,
                x: r.x,
                y: r.y,
                p: r.p,
                sp: r.sp,
                cycle: u64::from(r.cycle),
                scanline: r.scanline as i16,
            };
            tx.send(row).expect("trace receiver");
        })));
        self.run_frame(buttons);
        self.nes.set_trace(None);
        rx.try_iter().collect()
    }
}
