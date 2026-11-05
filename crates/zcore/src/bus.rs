//! NES Memory/Data Bus implementation.
//!
//! <https://wiki.nesdev.org/w/index.php/CPU_memory_map>
//!
//! [`State`] is the whole emulated console - the CPU, PPU, APU, input, the cartridge's board and
//! RAM, the work RAM and open bus - in fixed arrays, so it is `Copy` and a snapshot is a memcpy
//! (ladder stage Z5). [`Bus`] is a running console: a `State` plus what is not emulated state -
//! the PRG-ROM it reads (shared, immutable) and the frame buffer pixels go to. `Bus` derefs to
//! its `State`, so the emulation code reads `self.cpu`, `self.ppu`, ... either way.
//!
//! Behavior that needs more than one component lives in an `impl Bus` block in the file that owns
//! the state it reads: the CPU's in [`cpu`](crate::cpu), the instruction set's in
//! [`instr`](crate::cpu::instr), the PPU's in [`ppu`](crate::ppu), the input ports' in
//! [`input`](crate::input). What is here is the CPU-side routing - which address reaches which
//! component.
//!
//! `Bus` carries both address spaces, so its accessors name the one they mean:
//!
//! | | reads | writes |
//! |---|---|---|
//! | CPU, spending a cycle | [`Bus::read`], [`Bus::peek`] | [`Bus::write`] |
//! | PPU address space | [`Bus::ppu_bus_peek`] | |
//! | cartridge, through the page tables | [`Bus::chr_peek`] | |
//!
//! [`Bus::poke`] is the write with no side effects, storing into RAM alone.
//!
//! # Stability
//!
//! [`State`]'s fields are the emulation's internal wiring. They are public so that tools can
//! digest machine state, but they track the implementation rather than any stable surface.

use crate::{
    apu::Apu,
    cart::Cart,
    common::ResetKind,
    cpu::{Cpu, IrqFlags, TraceHook},
    input::{Input, Player},
    mapper::Mapper,
    memory::{ConstArray, Memory, PRG_ROM_SIZE, PrgRom},
    ppu::{self, Ppu},
};
use serde::{Deserialize, Serialize};
use std::{
    ops::{Deref, DerefMut},
    sync::{Arc, LazyLock},
};

/// A pixel frame: one palette index (plus emphasis bits) per pixel, row-major.
pub type FrameBuffer = [u16; ppu::size::FRAME];

/// The PRG-ROM of a console with no cart: all zeroes.
static NO_PRG_ROM: LazyLock<Arc<PrgRom>> = LazyLock::new(|| {
    vec![0; PRG_ROM_SIZE]
        .into_boxed_slice()
        .try_into()
        .map(|rom: Box<PrgRom>| Arc::from(rom))
        .expect("PRG_ROM_SIZE bytes")
});

/// A running console: the emulated [`State`], the PRG-ROM it reads, and the frame its PPU draws.
#[must_use]
pub struct Bus {
    /// Everything emulated.
    pub state: State,
    /// The cart's PRG-ROM, shared and immutable.
    prg_rom: Arc<PrgRom>,
    /// Where the PPU draws, when `render` is set.
    frame: Box<FrameBuffer>,
    /// Whether the PPU draws pixels. Off, it does every bit of the same work except storing them,
    /// so the emulated state is identical either way.
    pub render: bool,
    /// Called before every instruction when set; see [`TraceHook`].
    pub(crate) trace: Option<TraceHook>,
    /// The PPU runs behind the CPU and is caught up only when something can observe it (see
    /// [`Bus::ppu_sync`]). Master clock at which it must be caught up between instructions,
    /// because a frame-boundary or NMI-request dot may be due. Recomputed on every catch-up.
    pub(crate) ppu_deadline: u32,
    /// PPU catch-up target at the start of the current instruction: where FQ-03's
    /// `status_at_instr` snapshot is taken.
    pub(crate) instr_start_clock: u32,
    /// Whether `status_at_instr` has been taken for the current instruction yet.
    pub(crate) status_latched: bool,
    /// The APU's `master_clock` from which a CPU cycle has board or APU work to do
    /// ([`Apu::next_clock`], or every cycle while the board counts down a write). Zero (work on
    /// the next cycle, which recomputes it) after anything that changes either from outside its
    /// cycle.
    pub(crate) tick_deadline: u32,
    /// Whether the interrupt inputs are all low and no interrupt flag is up, so a cycle's
    /// interrupt polling ([`Bus::handle_interrupts`]) would change nothing and is skipped. Cleared
    /// wherever an input can rise; set again between instructions once all is quiet.
    pub(crate) irq_quiet: bool,
}

impl Deref for Bus {
    type Target = State;

    #[inline(always)]
    fn deref(&self) -> &State {
        &self.state
    }
}

impl DerefMut for Bus {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut State {
        &mut self.state
    }
}

impl std::fmt::Debug for Bus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bus")
            .field("state", &self.state)
            .field("render", &self.render)
            .finish_non_exhaustive()
    }
}

impl Default for Bus {
    fn default() -> Self {
        Self::new()
    }
}

impl Bus {
    /// Creates a console with no cart, drawing pixels.
    pub fn new() -> Self {
        Self {
            state: State::new(),
            prg_rom: NO_PRG_ROM.clone(),
            frame: vec![0; ppu::size::FRAME]
                .into_boxed_slice()
                .try_into()
                .expect("FRAME pixels"),
            render: true,
            trace: None,
            ppu_deadline: 0,
            instr_start_clock: 0,
            status_latched: true,
            tick_deadline: 0,
            irq_quiet: false,
        }
    }

    /// Installs a cart: its PRG-ROM, its board and every memory region it came with.
    pub fn load_cart(&mut self, cart: Cart) {
        self.prg_rom = cart.prg_rom;
        self.state.memory = cart.memory;
        self.state.mapper = cart.mapper;
    }

    /// Removes the cart, leaving the console with no board.
    pub fn unload_cart(&mut self) {
        self.prg_rom = NO_PRG_ROM.clone();
        self.state.mapper = Mapper::default();
    }

    /// The PRG-ROM image this console reads at `$8000-$FFFF`.
    pub fn prg_rom(&self) -> &Arc<PrgRom> {
        &self.prg_rom
    }

    /// Share another holder's copy of the same PRG-ROM image.
    pub fn set_prg_rom(&mut self, prg_rom: Arc<PrgRom>) {
        debug_assert!(prg_rom[..] == self.prg_rom[..], "the same PRG-ROM");
        self.prg_rom = prg_rom;
    }

    /// The last frame drawn (pixels of frames run with `render` off are not updated).
    #[must_use]
    pub fn frame(&self) -> &FrameBuffer {
        &self.frame
    }

    /// A CPU-space byte at or above `$4020`: PRG-ROM through the page tables, or cart RAM.
    #[inline(always)]
    pub(crate) fn prg_peek(&self, addr: u16) -> u8 {
        if addr >= 0x8000 {
            self.prg_rom[self.state.memory.rom_index(addr)]
        } else {
            self.state.memory.prg_peek(addr)
        }
    }

    /// The frame buffer, when this frame is being drawn, alongside the PPU state.
    #[inline(always)]
    pub(crate) fn ppu_and_frame(&mut self) -> (&mut Ppu, Option<&mut FrameBuffer>) {
        let frame = self.render.then_some(&mut *self.frame);
        (&mut self.state.ppu, frame)
    }

    /// Resets the console, forwarding it to every component, and clears the frame buffer.
    pub fn reset(&mut self, kind: ResetKind) {
        self.frame.fill(0);
        self.state.reset_components(kind);
        self.tick_deadline = 0;
        self.irq_quiet = false;
        // Read straight off the bus so the components are not clocked during the reset itself.
        let lo = self.cpu_bus_read(Cpu::RESET_VECTOR);
        let hi = self.cpu_bus_read(Cpu::RESET_VECTOR + 1);
        self.cpu.pc = u16::from_le_bytes([lo, hi]);
        // The CPU takes 7 cycles to reset/power on (nesdev "CPU interrupts"), but FQ-06: FCEUmm
        // fetches the vector without spending any cycles.
    }
}

/// The whole emulated console, `Copy`.
///
/// <https://wiki.nesdev.org/w/index.php/CPU_memory_map>
///
/// |-----------------| $FFFF |-----------------|
/// | PRG-ROM         |       |                 |
/// |-----------------| $8000 |-----------------|
/// | PRG-RAM or SRAM |       | PRG-RAM or SRAM |
/// |-----------------| $6000 |-----------------|
/// | Expansion       |       | Expansion       |
/// | Modules         |       | Modules         |
/// |-----------------| $4020 |-----------------|
/// | APU/Input       |       |                 |
/// | Registers       |       |                 |
/// |- - - - - - - - -| $4000 |                 |
/// | PPU Mirrors     |       | I/O Registers   |
/// | $2000-$2007     |       |                 |
/// |- - - - - - - - -| $2008 |                 |
/// | PPU Registers   |       |                 |
/// |-----------------| $2000 |-----------------|
/// | WRAM Mirrors    |       |                 |
/// | $0000-$07FF     |       |                 |
/// |- - - - - - - - -| $0800 |                 |
/// | WRAM            |       | 2K Internal     |
/// |- - - - - - - - -| $0200 | Work RAM        |
/// | Stack           |       |                 |
/// |- - - - - - - - -| $0100 |                 |
/// | Zero Page       |       |                 |
/// |-----------------| $0000 |-----------------|
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[must_use]
#[repr(C)]
pub struct State {
    /// Central Processing Unit registers and cycle counters. What the CPU *does* is an
    /// `impl Bus` block in [`cpu`](crate::cpu), since every access moves the whole console.
    pub cpu: Cpu,
    /// Picture Processing Unit.
    pub ppu: Ppu,
    /// The cartridge's board.
    pub mapper: Mapper,
    /// Page-table addressed cartridge memory and CIRAM.
    pub memory: Memory,
    /// Audio Processing Unit.
    pub apu: Apu,
    /// Joypad inputs.
    pub input: Input,
    /// 2K of work RAM on the console itself, at $0000-$07FF and mirrored to $1FFF.
    pub wram: ConstArray<u8, { size::WRAM }>,
    /// Whatever was last read or written to to the Bus.
    pub open_bus: u8,
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}

pub mod size {
    //! Memory size constants.

    /// 2K of work RAM on the console itself, available to the CPU.
    pub const WRAM: usize = 0x800;
}

impl State {
    /// Power-on fill of console and cart RAM: FCEUmm's, and what the corpus was recorded with.
    pub const POWER_ON_RAM: u8 = 0xFF;

    /// A console with no cart.
    pub fn new() -> Self {
        Self {
            wram: ConstArray::new(),
            cpu: Cpu::new(),
            ppu: Ppu::new(),
            mapper: Mapper::none(),
            memory: Memory::default(),
            apu: Apu::new(),
            input: Input::new(),
            open_bus: 0x00,
        }
    }

    /// Resets every component. A hard reset is a power cycle, so it also re-initialises RAM that a
    /// power cycle would not preserve: the console's work RAM, and the cart's unless a battery is
    /// keeping it.
    fn reset_components(&mut self, kind: ResetKind) {
        self.cpu.reset(kind);
        if kind == ResetKind::Hard {
            self.wram.fill(Self::POWER_ON_RAM);
            if !self.memory.battery_backed() {
                self.memory.fill_ram(Self::POWER_ON_RAM);
            }
        }
        self.ppu.reset(kind);
        self.mapper.reset(kind);
        // Reset can change banking, and the board cannot reach `Memory` from `reset`.
        self.rebuild_mapper_state();
        self.apu.reset(kind);
    }
}

impl Bus {
    /// The console's 2K of work RAM.
    #[must_use]
    #[inline]
    #[allow(clippy::missing_const_for_fn)] // false positive on non-const deref coercion
    pub fn wram(&self) -> &[u8; size::WRAM] {
        &self.wram
    }

    /// The console's 2K of work RAM, for a debugger or a frontend's cheat engine to write.
    #[must_use]
    #[inline]
    #[allow(clippy::missing_const_for_fn)] // false positive on non-const deref coercion
    pub fn wram_mut(&mut self) -> &mut [u8; size::WRAM] {
        &mut self.wram
    }

    /// Store `val` at `addr` if RAM answers there, reporting whether it landed.
    ///
    /// The write a debugger makes, the counterpart of [`Bus::peek`]: it decodes the CPU address
    /// the way [`Bus::write`] does, but reaches only work RAM and whatever the board maps
    /// writable, so poking a byte cannot clock a component or trip a mapper register. `false`
    /// where the address reaches ROM, an unmapped page, or the `$2000-$40FF` registers, none of
    /// which can be written without moving the console.
    pub fn poke(&mut self, addr: u16, val: u8) -> bool {
        match addr {
            0x0000..=0x1FFF => {
                self.wram[usize::from(addr & 0x07FF)] = val;
                true
            }
            0x4100..=0xFFFF => {
                let writable = self.memory.prg_writable(addr);
                if writable {
                    self.memory.prg_write(addr, val);
                }
                writable
            }
            _ => false,
        }
    }

    /// Clocks everything the CPU's cycle drives: the board and the APU (the PPU catches up on its
    /// own, see [`Bus::ppu_sync`]). Until `tick_deadline` only the APU's cycle counters move.
    #[inline(always)]
    pub fn cpu_clock(&mut self) {
        self.state.apu.tick();
        if self.state.apu.master_clock >= self.tick_deadline {
            self.cpu_clock_due();
        }
    }

    /// A cycle on which the board or the APU has work: the board's write countdown, then the
    /// APU's frame counter, length counters and DMC, whose fetch request halts the CPU at its
    /// next read ([`Bus::handle_dma`]).
    #[inline(never)]
    fn cpu_clock_due(&mut self) {
        let State { mapper, apu, cpu, .. } = &mut self.state;
        mapper.clock();
        apu.clock_due();
        if apu.dmc.dma_pending {
            apu.dmc.dma_pending = false;
            cpu.irq_flags
                .insert(IrqFlags::DMA_DMC | IrqFlags::DMA_HALT | IrqFlags::DMA_DUMMY_READ);
        }
        self.tick_deadline = if mapper.clocking() {
            0
        } else {
            apu.next_clock()
        };
        if apu.irq_pending() {
            self.irq_quiet = false;
        }
    }
}

impl Bus {
    /// Route a read to whichever component owns `addr`, without moving the console.
    ///
    /// The address decode alone; [`Bus::read`] spends a CPU cycle and then comes through here.
    #[inline(always)]
    pub(crate) fn cpu_bus_read(&mut self, addr: u16) -> u8 {
        self.open_bus = if addr < 0x2000 {
            self.wram[usize::from(addr & 0x07FF)]
        } else if addr >= 0x4100 {
            self.prg_peek(addr)
        } else if addr < 0x4000 {
            self.ppu_register_read(addr & 0x2007)
        } else {
            match addr {
                0x4015 => {
                    self.tick_deadline = 0;
                    self.irq_quiet = false;
                    self.apu.read_status()
                }
                0x4016 => self.input_read(Player::One),
                0x4017 => self.input_read(Player::Two),
                _ => self.open_bus,
            }
        };
        self.open_bus
    }

    /// A read of `$2000-$2007`, after catching the PPU up to it.
    #[inline(never)]
    fn ppu_register_read(&mut self, addr: u16) -> u8 {
        self.ppu_sync();
        match addr {
            0x2002 => self.ppu.read_status(),
            0x2004 => self.ppu.read_oamdata(),
            0x2007 => self.read_data(),
            _ => self.ppu.open_bus_now(),
        }
    }

    /// Route a read to whichever component owns `addr`, with no side effects at all.
    #[must_use]
    pub(crate) fn cpu_bus_peek(&self, addr: u16) -> u8 {
        let addr = match addr {
            0x0800..=0x1FFF => addr & 0x07FF,
            0x2008..=0x3FFF => addr & 0x2007,
            _ => addr,
        };
        match addr {
            0x0000..=0x07FF => self.wram[usize::from(addr)],
            0x4100..=0xFFFF => self.prg_peek(addr),
            0x2002 => self.ppu.peek_status(),
            0x2004 => self.ppu.peek_oamdata(),
            0x2007 => self.peek_data(),
            0x4015 => self.apu.peek_status(),
            0x4016 => self.input_peek(Player::One),
            0x4017 => self.input_peek(Player::Two),
            0x2000 | 0x2001 | 0x2003 | 0x2005 | 0x2006 => self.ppu.open_bus_now(),
            _ => self.open_bus,
        }
    }

    /// Route a write to whichever component owns `addr`, without moving the console.
    ///
    /// [`Bus::write`] is the cycle-spending form.
    #[inline(always)]
    pub(crate) fn cpu_bus_write(&mut self, addr: u16, val: u8) {
        self.open_bus = val;
        if addr < 0x2000 {
            self.wram[usize::from(addr & 0x07FF)] = val;
        } else if addr >= 0x4100 {
            if addr >= 0x8000 {
                // A board register: banking and mirroring change what the PPU fetches, and the
                // board counts the cycles after a write.
                self.ppu_sync();
                self.tick_deadline = 0;
            }
            // Data store first, then let the board act on any register the write hit.
            // Destructured so both fields can be borrowed at once.
            let State { mapper, memory, .. } = &mut self.state;
            memory.prg_write(addr, val);
            mapper.write_register(memory, addr, val);
        } else if addr < 0x4000 {
            self.ppu_register_write(addr & 0x2007, val);
        } else {
            match addr {
                0x4000..=0x4013 | 0x4015 | 0x4017 => {
                    self.tick_deadline = 0;
                    self.irq_quiet = false;
                    self.apu.write(addr, val);
                }
                0x4016 => self.input.write(val),
                // $4014: OAM DMA is handled by the CPU.
                _ => (),
            }
        }
    }

    /// A write to `$2000-$2007`, after catching the PPU up to it.
    #[inline(never)]
    fn ppu_register_write(&mut self, addr: u16, val: u8) {
        self.ppu_sync();
        match addr {
            0x2000 => self.write_ctrl(val),
            0x2001 => self.write_mask(val),
            0x2002 => self.ppu.write_open_bus(val),
            0x2003 => self.ppu.write_oamaddr(val),
            0x2004 => self.ppu.write_oamdata(val),
            0x2005 => self.ppu.write_scroll(val),
            0x2006 => self.ppu.write_addr(val),
            _ => self.write_data(val),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{input::JoypadBtn, mapper::Nrom, memory::Src};

    #[test]
    fn load_cart_values() {
        let mut bus = Bus::default();
        #[rustfmt::skip]
        let rom: [u8; 16] = [
            0x4E, 0x45, 0x53, 0x1A,
            0x00, 0x00, 0x02, 0x00,
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
        ];
        let cart = Cart::from_rom("load_cart_test", &mut rom.as_slice()).expect("valid cart");

        let expected_mirroring = cart.mirroring();
        bus.load_cart(cart);

        assert!(
            matches!(bus.mapper, Mapper::Nrom(_)),
            "mapper is Nrom: {:?}",
            bus.mapper
        );
        assert_eq!(bus.mirroring(), expected_mirroring, "mirroring");
    }

    #[test]
    fn load_cart_chr_rom() {
        let mut bus = Bus::default();
        // A non-zero CHR-ROM size gives CHR-ROM rather than CHR-RAM.
        let mut cart = Cart::empty_sized(0x4000, 0x2000);
        cart.mapper = Nrom::load(&mut cart).unwrap();
        cart.memory.region_mut(Src::Chr).fill(0x66);
        bus.load_cart(cart);

        bus.cpu_bus_write(0x2006, 0x00);
        bus.cpu_bus_write(0x2006, 0x00);
        bus.cpu_bus_read(0x2007);
        assert_eq!(bus.cpu_bus_read(0x2007), 0x66, "chr_rom start");
        bus.cpu_bus_write(0x2006, 0x1F);
        bus.cpu_bus_write(0x2006, 0xFF);
        bus.cpu_bus_read(0x2007);
        assert_eq!(bus.cpu_bus_read(0x2007), 0x66, "chr_rom end");

        // Writes disallowed
        bus.cpu_bus_write(0x2006, 0x00);
        bus.cpu_bus_write(0x2006, 0x10);
        bus.cpu_bus_write(0x2007, 0x77);

        bus.cpu_bus_write(0x2006, 0x00);
        bus.cpu_bus_write(0x2006, 0x10);
        bus.cpu_bus_read(0x2007);
        assert_eq!(bus.cpu_bus_read(0x2007), 0x66, "chr_rom read-only");
    }

    #[test]
    fn load_cart_chr_ram() {
        let mut bus = Bus::default();
        // A zero-sized CHR-ROM yields CHR-RAM.
        let mut cart = Cart::empty_sized(0x4000, 0);
        cart.mapper = Nrom::load(&mut cart).unwrap();
        cart.memory.region_mut(Src::Chr).fill(0x66);
        bus.load_cart(cart);

        bus.cpu_bus_write(0x2006, 0x00);
        bus.cpu_bus_write(0x2006, 0x00);
        bus.cpu_bus_read(0x2007);
        assert_eq!(bus.cpu_bus_read(0x2007), 0x66, "chr_ram start");
        bus.cpu_bus_write(0x2006, 0x1F);
        bus.cpu_bus_write(0x2006, 0xFF);
        bus.cpu_bus_read(0x2007);
        assert_eq!(bus.cpu_bus_read(0x2007), 0x66, "chr_ram end");

        // Writes allowed
        bus.cpu_bus_write(0x2006, 0x10);
        bus.cpu_bus_write(0x2006, 0x00);
        // PPU writes to $2006 are delayed by 2 PPU clocks
        bus.ppu_clock();
        bus.ppu_clock();
        bus.cpu_bus_write(0x2007, 0x77);

        bus.cpu_bus_write(0x2006, 0x10);
        bus.cpu_bus_write(0x2006, 0x00);
        // PPU writes to $2006 are delayed by 2 PPU clocks
        bus.ppu_clock();
        bus.ppu_clock();
        bus.cpu_bus_read(0x2007);
        assert_eq!(bus.cpu_bus_read(0x2007), 0x77, "chr_ram write");
    }

    #[test]
    fn clock() {
        let mut bus = Bus::default();

        bus.ppu_clock_to(12);
        assert_eq!(bus.ppu.master_clock, 12, "ppu clock");
        bus.cpu_clock();
        assert_eq!(bus.apu.master_clock, 1, "apu clock");
    }

    #[test]
    fn read_write_ram() {
        let mut bus = Bus::default();

        bus.cpu_bus_write(0x0001, 0x66);
        assert_eq!(bus.cpu_bus_peek(0x0001), 0x66, "peek ram");
        assert_eq!(bus.cpu_bus_read(0x0001), 0x66, "read ram");
        assert_eq!(bus.cpu_bus_read(0x0801), 0x66, "peek mirror 1");
        assert_eq!(bus.cpu_bus_read(0x0801), 0x66, "read mirror 1");
        assert_eq!(bus.cpu_bus_read(0x1001), 0x66, "peek mirror 2");
        assert_eq!(bus.cpu_bus_read(0x1001), 0x66, "read mirror 2");
        assert_eq!(bus.cpu_bus_read(0x1801), 0x66, "peek mirror 3");
        assert_eq!(bus.cpu_bus_read(0x1801), 0x66, "read mirror 3");

        bus.cpu_bus_write(0x0802, 0x77);
        assert_eq!(bus.cpu_bus_read(0x0002), 0x77, "write mirror 1");
        bus.cpu_bus_write(0x1002, 0x88);
        assert_eq!(bus.cpu_bus_read(0x0002), 0x88, "write mirror 2");
        bus.cpu_bus_write(0x1802, 0x99);
        assert_eq!(bus.cpu_bus_read(0x0002), 0x99, "write mirror 3");
    }

    /// $2000-$2007 repeat every 8 bytes up to $3FFF, so the mirror mask is what decides which
    /// register a write lands on. Reads of the write-only registers return the PPU's open bus
    /// rather than anything latched.
    #[test]
    #[ignore = "FQ-03: $2002 reads see VBlank as of the instruction start"]
    fn read_write_ppu() {
        let mut bus = Bus::default();

        bus.cpu_bus_write(0x2000, 0x80);
        assert!(bus.ppu.ctrl_nmi_enabled, "$2000 PPUCTRL");
        bus.cpu_bus_write(0x3FF8, 0x00);
        assert!(!bus.ppu.ctrl_nmi_enabled, "$3FF8 mirrors $2000");

        bus.cpu_bus_write(0x2001, 0x1E);
        assert_eq!(bus.ppu.mask_bits.bits(), 0x1E, "$2001 PPUMASK");
        bus.cpu_bus_write(0x3FF9, 0x00);
        assert_eq!(bus.ppu.mask_bits.bits(), 0x00, "$3FF9 mirrors $2001");
        bus.cpu_bus_write(0x2003, 0x42);
        assert_eq!(bus.ppu.oamaddr, 0x42, "$2003 OAMADDR");

        // OAMDATA round-trips through $2004, and the address post-increments on write.
        bus.cpu_bus_write(0x2003, 0x10);
        bus.cpu_bus_write(0x2004, 0x99);
        assert_eq!(bus.ppu.oamaddr, 0x11, "OAMADDR increments on write");
        bus.cpu_bus_write(0x2003, 0x10);
        assert_eq!(bus.cpu_bus_read(0x2004), 0x99, "$2004 OAMDATA");

        // The write-only registers read back the PPU's open bus, not their contents.
        bus.ppu.open_bus = 0xA5;
        for addr in [0x2000, 0x2001, 0x2003, 0x2005, 0x2006] {
            assert_eq!(bus.cpu_bus_read(addr), 0xA5, "${addr:04X} is write-only");
        }

        // $2002 clears the vblank flag as a side effect, so a second read differs - and `peek`
        // must not do it.
        bus.ppu.set_in_vblank(true);
        assert_ne!(bus.cpu_bus_peek(0x2002) & 0x80, 0, "peek sees vblank");
        assert_ne!(bus.cpu_bus_peek(0x2002) & 0x80, 0, "and leaves it set");
        assert_ne!(bus.cpu_bus_read(0x2002) & 0x80, 0, "read sees vblank");
        assert_eq!(bus.cpu_bus_read(0x2002) & 0x80, 0, "and clears it");
    }

    /// $4015 is the APU status register both ways, but $4017 is not symmetric: writing it sets the
    /// APU frame counter while reading it returns controller two. Getting that backwards is silent.
    #[test]
    fn read_write_apu() {
        let mut bus = Bus::default();

        // $4015 write enables length counters; reading it reports which are non-zero.
        bus.cpu_bus_write(0x4015, 0x0F);
        assert!(bus.apu.pulse1.enabled, "pulse1 enabled");
        assert!(bus.apu.pulse2.enabled, "pulse2 enabled");
        assert!(bus.apu.triangle.enabled, "triangle enabled");
        assert!(bus.apu.noise.enabled, "noise enabled");
        bus.cpu_bus_write(0x4015, 0x00);
        assert!(!bus.apu.pulse1.enabled, "pulse1 disabled");

        // $4017 write is the frame counter; bit 6 inhibits the frame IRQ.
        bus.cpu_bus_write(0x4017, 0x40);
        assert!(
            bus.apu.frame_counter.inhibit_irq,
            "$4017 bit 6 inhibits the frame IRQ"
        );

        // $4017 read is controller two, not the frame counter.
        bus.input.joypads[1].press(JoypadBtn::A);
        bus.cpu_bus_write(0x4016, 0x01);
        bus.cpu_bus_write(0x4016, 0x00);
        assert_eq!(
            bus.cpu_bus_read(0x4017) & 0x01,
            0x01,
            "$4017 reads controller two"
        );
    }

    /// Each channel's length load and halt flag reach its own counter only.
    #[test]
    fn write_apu_lengths() {
        let mut bus = Bus::default();
        bus.cpu_bus_write(0x4015, 0x0F);

        // Halt flags: D5 for pulse/noise, D7 for the triangle.
        bus.cpu_bus_write(0x4000, 0x20);
        bus.cpu_bus_write(0x4008, 0x80);
        bus.cpu_bus_write(0x400C, 0x20);
        assert!(bus.apu.pulse1.new_halt, "$4000 D5");
        assert!(!bus.apu.pulse2.new_halt, "pulse2 untouched");
        assert!(bus.apu.triangle.new_halt, "$4008 D7");
        assert!(bus.apu.noise.new_halt, "$400C D5");

        // Length loads latch a reload value; the next APU clock applies it.
        bus.cpu_bus_write(0x4003, 0x08);
        assert_eq!(bus.apu.pulse1.reload, 254, "$4003 length");
        assert_eq!(bus.apu.pulse2.reload, 0, "pulse2 untouched");
        bus.cpu_bus_write(0x4007, 0x10);
        bus.cpu_bus_write(0x400B, 0x18);
        bus.cpu_bus_write(0x400F, 0x20);
        assert_eq!(bus.apu.noise.reload, 40, "$400F length");
        bus.cpu_clock();
        assert_eq!(bus.apu.peek_status() & 0x0F, 0x0F, "all four counting");

        // A disabled channel ignores the write.
        bus.cpu_bus_write(0x4015, 0x00);
        bus.cpu_bus_write(0x400F, 0x10);
        assert_eq!(bus.apu.noise.reload, 0, "a disabled channel ignores the write");

        // $4009 and $400D are not registers.
        bus.cpu_bus_write(0x4015, 0x0F);
        bus.cpu_bus_write(0x4009, 0xFF);
        bus.cpu_bus_write(0x400D, 0xFF);
        assert_eq!(bus.apu.triangle.reload, 0, "$4009 is unmapped");
        assert_eq!(bus.apu.noise.reload, 0, "$400D is unmapped");
    }

    /// $4013 sets the DMC sample length, stored scaled; $4015 bit 4 reports it once enabled.
    #[test]
    fn write_dmc() {
        let mut bus = Bus::default();

        bus.cpu_bus_write(0x4013, 0x02);
        assert_eq!(bus.apu.dmc.sample_length, 0x21, "$4013 is n*16 + 1");
        assert_eq!(bus.apu.peek_status() & 0x10, 0, "not playing");
        bus.cpu_bus_write(0x4015, 0x10);
        assert_eq!(bus.apu.dmc.bytes_remaining, 0x21, "enabled");
        assert_eq!(bus.apu.peek_status() & 0x10, 0x10, "$4015 bit 4");
        bus.cpu_bus_write(0x4015, 0x00);
        assert_eq!(bus.apu.peek_status() & 0x10, 0, "disabled");
    }

    /// $4016 writes strobe both controllers; $4016 and $4017 read them back one bit at a time.
    #[test]
    fn read_write_input() {
        let mut bus = Bus::default();

        bus.input.joypads[0].press(JoypadBtn::A);
        bus.input.joypads[0].press(JoypadBtn::Right);

        // Strobe high then low latches the button state and rewinds to bit 0.
        bus.cpu_bus_write(0x4016, 0x01);
        bus.cpu_bus_write(0x4016, 0x00);

        // A, B, Select, Start, Up, Down, Left, Right - so bit 0 is A and bit 7 is Right.
        let bits: Vec<u8> = (0..8).map(|_| bus.cpu_bus_read(0x4016) & 0x01).collect();
        assert_eq!(bits, [1, 0, 0, 0, 0, 0, 0, 1], "controller one shifts out");

        // Re-strobing rewinds it.
        bus.cpu_bus_write(0x4016, 0x01);
        bus.cpu_bus_write(0x4016, 0x00);
        assert_eq!(
            bus.cpu_bus_read(0x4016) & 0x01,
            0x01,
            "back to the A button"
        );

        // Controller two is a separate shift register on $4017.
        assert_eq!(
            bus.cpu_bus_read(0x4017) & 0x01,
            0x00,
            "controller two is idle"
        );
    }

    /// A hard reset re-fills WRAM with the power-on $FF; a soft reset leaves it alone.
    #[test]
    fn reset() {
        let mut bus = Bus::default();

        bus.cpu_bus_write(0x0001, 0x66);
        bus.cpu_bus_write(0x2000, 0x80);

        bus.reset(ResetKind::Soft);
        assert_eq!(
            bus.cpu_bus_peek(0x0001),
            0x66,
            "a soft reset preserves WRAM"
        );

        bus.reset(ResetKind::Hard);
        assert_eq!(bus.cpu_bus_peek(0x0001), 0xFF, "a hard reset refills WRAM");
    }

    /// A hard reset is a power cycle, so cart RAM comes back up in whatever state the console is
    /// configured for - unless a battery is keeping it, which is the whole point of the battery.
    #[test]
    fn hard_reset_refills_cart_ram_unless_it_is_battery_backed() {
        for battery_backed in [false, true] {
            let mut bus = Bus::default();
            let mut cart = Cart::empty_sized(0x4000, 0x2000);
            cart.mapper = Nrom::load(&mut cart).expect("valid mapper");
            bus.load_cart(cart);
            bus.memory.set_battery_backed(battery_backed);
            bus.memory.region_mut(Src::PrgRam).fill(0x42);

            bus.reset(ResetKind::Soft);
            assert!(
                bus.memory
                    .region_ref(Src::PrgRam)
                    .iter()
                    .all(|&b| b == 0x42),
                "a soft reset leaves cart RAM alone"
            );

            bus.reset(ResetKind::Hard);
            let expected = if battery_backed { 0x42 } else { 0xFF };
            assert!(
                bus.memory
                    .region_ref(Src::PrgRam)
                    .iter()
                    .all(|&b| b == expected),
                "battery_backed {battery_backed}: cart RAM after a power cycle"
            );
        }
    }

    /// A debugger write must not move the console, so it reaches RAM and refuses everything else
    /// rather than falling back on the bus write.
    #[test]
    fn a_poke_reaches_ram_and_refuses_the_rest() {
        let mut bus = Bus::default();
        let mut cart = Cart::empty_sized(0x4000, 0x2000);
        cart.mapper = Nrom::load(&mut cart).expect("valid mapper");
        bus.load_cart(cart);
        bus.reset(ResetKind::Hard);

        assert!(bus.poke(0x0042, 0x5A), "work RAM");
        assert_eq!(bus.peek(0x0042), 0x5A);
        assert!(bus.poke(0x1842, 0xA5), "the mirror of it at $1842");
        assert_eq!(bus.peek(0x0042), 0xA5, "which is the same byte");

        assert!(bus.poke(0x6000, 0x3C), "cart RAM");
        assert_eq!(bus.peek(0x6000), 0x3C);

        assert!(!bus.poke(0x2000, 0xFF), "PPUCTRL");
        assert!(!bus.poke(0x4015, 0xFF), "APU status");
        assert!(!bus.ppu.ctrl_nmi_enabled, "no register took the write");

        let rom = bus.peek(0x8000);
        assert!(!bus.poke(0x8000, rom.wrapping_add(1)), "PRG-ROM");
        assert_eq!(bus.peek(0x8000), rom);
    }
}
