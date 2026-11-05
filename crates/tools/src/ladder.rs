//! Ladder-only digest domains (TFGOLDEN bits 7-10): internal machine state, encoded field by field
//! so the encoding survives representation changes between ladder stages. A stage whose change alters
//! any of these bytes is not a pure subtraction.
//!
//! Instantiated by the zcore backend; the encoding must stay stable across ladder stages.

#[macro_export]
macro_rules! ladder_domains {
    ($bus:expr, $domain:expr, $out:expr, $mapper:path) => {{
        let bus = $bus;
        let out: &mut Vec<u8> = $out;
        match $domain {
            $crate::formats::Domain::Cpu => {
                let c = &bus.cpu;
                out.extend_from_slice(&c.pc.to_le_bytes());
                out.extend_from_slice(&[c.acc, c.x, c.y, c.status.bits(), c.sp]);
                out.extend_from_slice(&c.cycle.to_le_bytes());
                true
            }
            $crate::formats::Domain::Ppu => {
                let p = &bus.ppu;
                out.extend_from_slice(&p.scanline.to_le_bytes());
                out.extend_from_slice(&p.cycle.to_le_bytes());
                out.push((p.frame_count & 1) as u8);
                out.extend_from_slice(&p.scroll.v.to_le_bytes());
                out.extend_from_slice(&p.scroll.t.to_le_bytes());
                out.extend_from_slice(&p.scroll.fine_x.to_le_bytes());
                out.push(u8::from(p.scroll.write_latch));
                out.extend_from_slice(&[
                    p.read_status_bits(),
                    p.oamaddr,
                    p.vram_buffer,
                    p.open_bus,
                    p.dead_frames,
                ]);
                true
            }
            $crate::formats::Domain::Mapper => {
                if let $mapper(m) = &bus.mapper {
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
                    true
                } else {
                    false
                }
            }
            $crate::formats::Domain::Apu => {
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
                true
            }
            _ => false,
        }
    }};
}
