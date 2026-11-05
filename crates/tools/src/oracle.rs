//! FCEUmm oracle as a `tf` backend (see the dev-only `oracle` crate).

use crate::backend::{Backend, TraceRow};
use crate::formats::Domain;
use anyhow::Result;

pub struct Oracle(oracle::Oracle);

impl Oracle {
    pub fn open(rom_path: &std::path::Path) -> Result<Oracle> {
        Ok(Oracle(oracle::Oracle::open(rom_path)?))
    }
}

impl Backend for Oracle {
    fn name(&self) -> &'static str {
        "fceumm"
    }

    fn run_frame(&mut self, buttons: u8) {
        self.0.run_frame(buttons);
    }

    fn poke(&mut self, addr: u16, value: u8) {
        self.0.poke(addr, value);
    }

    fn read_domain(&mut self, domain: Domain, out: &mut Vec<u8>) -> bool {
        out.clear();
        match domain {
            Domain::Ram => out.extend_from_slice(self.0.ram()),
            Domain::PrgRam => out.extend_from_slice(self.0.prg_ram()),
            Domain::Chr => out.extend_from_slice(&self.0.chr()),
            Domain::Ciram => out.extend_from_slice(&self.0.ciram()),
            Domain::Palette => out.extend_from_slice(&self.0.palette()),
            // OAM as $2004 reads it: attribute bits 2-4 don't exist in the PPU (FCEUmm stores them).
            Domain::Oam => {
                out.extend(self.0.oam().iter().enumerate().map(|(i, &b)| if i & 3 == 2 { b & 0xE3 } else { b }))
            }
            Domain::Pixels => {
                let mut px = Box::new([0u8; 256 * 240]);
                self.0.pixels(&mut px);
                out.extend_from_slice(&px[..]);
            }
            _ => return false,
        }
        true
    }

    fn overshoot_dots(&self) -> Option<f64> {
        Some(-self.0.budget_dots())
    }

    fn trace_frame(&mut self, buttons: u8) -> Vec<TraceRow> {
        self.0.trace_frame(buttons)
    }
}
