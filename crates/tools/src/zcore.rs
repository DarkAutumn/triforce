//! zcore driven by `tf`: frames, pokes, the 11 digest domains and instruction traces. The
//! machine-state domains (TFGOLDEN bits 7-10) read internals through the hidden `ZCore::bus`.

use crate::formats::Domain;
use anyhow::{Context, Result};
use std::sync::mpsc;
use zcore::bus::Bus;
use zcore::mapper::Mapper;
use zcore::memory::Src;
use zcore::{Rom, TraceRow, ZCore};

pub struct Zcore {
    nes: ZCore,
    render: bool,
}

impl Zcore {
    pub fn open(rom: &[u8]) -> Result<Zcore> {
        let rom = Rom::from_ines(rom).context("zcore ROM")?;
        Ok(Zcore { nes: ZCore::power_on(rom), render: true })
    }

    pub fn run_frame(&mut self, buttons: u8) {
        self.nes.run_frame(buttons, self.render);
    }

    /// Draw pixels on the following frames (default on).
    pub fn set_render(&mut self, on: bool) {
        self.render = on;
    }

    pub fn poke(&mut self, addr: u16, value: u8) {
        self.nes.poke(addr, value);
    }

    /// The machine as a `TFZSTATE` file.
    pub fn save_state(&self) -> Vec<u8> {
        zcore::state_file::encode(self.nes.rom(), &self.nes.snapshot(), 0)
    }

    /// Resume a [`Zcore::save_state`] file.
    pub fn load_state(&mut self, bytes: &[u8]) -> Result<()> {
        let file = zcore::state_file::decode(self.nes.rom(), bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
        self.nes.restore(&file.machine);
        Ok(())
    }

    pub fn ram(&self) -> &[u8; 0x800] {
        self.nes.ram()
    }

    pub fn prg_ram(&self) -> &[u8; 0x2000] {
        self.nes.prg_ram()
    }

    /// Fill `out` with the bytes of a domain. Returns false for pixels of a frame that wasn't drawn.
    pub fn read_domain(&self, domain: Domain, out: &mut Vec<u8>) -> bool {
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
            Domain::Cpu | Domain::Ppu | Domain::Mapper | Domain::Apu => machine_domain(bus, domain, out),
        }
        true
    }

    /// Run one frame while recording every instruction.
    pub fn trace_frame(&mut self, buttons: u8) -> Vec<TraceRow> {
        let (tx, rx) = mpsc::channel();
        self.nes.set_trace(Some(Box::new(move |r: &TraceRow| tx.send(*r).expect("trace receiver"))));
        self.run_frame(buttons);
        self.nes.set_trace(None);
        rx.try_iter().collect()
    }
}

/// Machine-state domains, encoded field by field so the digests survive representation changes
/// inside zcore: a change that alters any of these bytes changes emulated state, not just layout.
fn machine_domain(bus: &Bus, domain: Domain, out: &mut Vec<u8>) {
    match domain {
        Domain::Cpu => {
            let c = &bus.cpu;
            out.extend_from_slice(&c.pc.to_le_bytes());
            out.extend_from_slice(&[c.acc, c.x, c.y, c.status.bits(), c.sp]);
            out.extend_from_slice(&c.cycle.to_le_bytes());
        }
        Domain::Ppu => {
            let p = &bus.ppu;
            out.extend_from_slice(&p.scanline.to_le_bytes());
            out.extend_from_slice(&p.cycle.to_le_bytes());
            out.push((p.frame_count & 1) as u8);
            out.extend_from_slice(&p.scroll.v.to_le_bytes());
            out.extend_from_slice(&p.scroll.t.to_le_bytes());
            out.extend_from_slice(&p.scroll.fine_x.to_le_bytes());
            out.push(u8::from(p.scroll.write_latch));
            out.extend_from_slice(&[p.read_status_bits(), p.oamaddr, p.vram_buffer, p.open_bus]);
        }
        Domain::Mapper => {
            let Mapper::Sxrom(m) = &bus.mapper else { unreachable!("Zelda is MMC1") };
            let m = &m.mmc1;
            out.extend_from_slice(&[
                m.write_buffer,
                m.shift_count,
                u8::from(m.prg_ram_disabled),
                u8::from(m.chr_mode),
                u8::from(m.prg_mode),
                u8::from(m.prg_bank_select),
                m.mirroring as u8,
                m.chr0,
                m.chr1,
                m.prg,
            ]);
        }
        Domain::Apu => {
            let a = &bus.apu;
            let d = &a.dmc;
            out.extend_from_slice(&d.cycle.to_le_bytes());
            out.extend_from_slice(&d.counter.to_le_bytes());
            out.extend_from_slice(&d.period.to_le_bytes());
            out.extend_from_slice(&d.addr.to_le_bytes());
            out.extend_from_slice(&d.sample_addr.to_le_bytes());
            out.extend_from_slice(&d.bytes_remaining.to_le_bytes());
            out.extend_from_slice(&d.sample_length.to_le_bytes());
            out.extend_from_slice(&[
                u8::from(d.irq_enabled),
                u8::from(d.loops),
                u8::from(d.buffer_empty),
                u8::from(d.dma_pending),
                d.init,
                d.bits_remaining,
            ]);
            out.extend_from_slice(&[
                u8::from(a.frame_counter.irq_pending),
                u8::from(a.frame_counter.inhibit_irq),
                u8::from(d.irq_pending),
            ]);
        }
        _ => unreachable!("{} is not a machine-state domain", domain.name()),
    }
}
