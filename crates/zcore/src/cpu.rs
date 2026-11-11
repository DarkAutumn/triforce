//! 6502 Central Processing Unit (CPU) implementation.
//!
//! <https://wiki.nesdev.org/w/index.php/CPU>
//!
//! [`Cpu`] is the register file and cycle counters - the state a 6502 keeps. Everything it *does*
//! is an inherent method on [`Bus`], since a memory access moves the whole console: reading a byte
//! clocks the PPU, the APU and the board on the way past. Those `impl Bus` blocks are here and in
//! [`instr`].
//!
//! # Stability
//!
//! [`Cpu`]'s fields are the emulation's internal wiring - registers and cycle counters. They are
//! public so that tools can digest them, but they track the implementation, not a stable
//! interface; the stable entry point is [`ZCore`](crate::ZCore).

use crate::{
    bus::Bus,
    common::ResetKind,
    cpu::instr::AddrMode,
};
use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use std::fmt::{self};

pub mod instr;

/// The CPU at an instruction boundary, before the opcode fetch: what a trace hook is handed.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct TraceRow {
    /// Address of the instruction.
    pub pc: u16,
    /// Its opcode.
    pub opcode: u8,
    /// Accumulator.
    pub a: u8,
    /// X index.
    pub x: u8,
    /// Y index.
    pub y: u8,
    /// Status flags.
    pub p: u8,
    /// Stack pointer.
    pub sp: u8,
    /// CPU cycle counter (32-bit, wrapping).
    pub cycle: u32,
    /// PPU scanline.
    pub scanline: u16,
    /// PPU dot within the scanline.
    pub dot: u16,
}

/// A per-instruction callback (see [`ZCore::set_trace`](crate::ZCore::set_trace)).
pub type TraceHook = Box<dyn FnMut(&TraceRow) + Send + Sync>;

bitflags! {
    /// Interrupt and DMA state, kept as one byte because it is tested on every cycle.
    ///
    /// The `PREV_` flags are the previous cycle's values, which is what makes an interrupt take
    /// effect one instruction late the way the hardware does.
    #[derive(Default, Serialize, Deserialize, Debug, Copy, Clone)]
    #[must_use]
    pub struct IrqFlags: u8 {
        /// NMI line asserted.
        const NMI = 1 << 0;
        /// NMI line as of the previous cycle.
        const PREV_NMI = 1 << 1;
        /// NMI was pending as of the previous cycle.
        const PREV_NMI_PENDING = 1 << 2;
        /// An IRQ will be serviced after the current instruction.
        const RUN_IRQ = 1 << 3;
        /// `RUN_IRQ` as of the previous cycle.
        const PREV_RUN_IRQ = 1 << 4;
        /// A DMC sample fetch is stealing cycles.
        const DMA_DMC = 1 << 5;
        /// The CPU is halted for a DMA.
        const DMA_HALT = 1 << 6;
        /// The DMA's alignment dummy read.
        const DMA_DUMMY_READ = 1 << 7;
    }
}

// Status Registers
// https://wiki.nesdev.org/w/index.php/Status_flags
// 7654 3210
// NVUB DIZC
// |||| ||||
// |||| |||+- Carry
// |||| ||+-- Zero
// |||| |+--- Interrupt Disable
// |||| +---- Decimal Mode - Not used in the NES but still has to function
// |||+------ Break - 1 when pushed to stack from PHP/BRK, 0 from IRQ/NMI
// ||+------- Unused - always set to 1 when pushed to stack
// |+-------- Overflow
// +--------- Negative
bitflags! {
    /// CPU Status Registers.
    #[derive(Default, Serialize, Deserialize, Debug, Copy, Clone, PartialEq, Eq)]
    #[must_use]
    pub struct Status: u8 {
        /// Carry.
        const C = 1;
        /// Zero.
        const Z = 1 << 1;
        /// Disable interrupt.
        const I = 1 << 2;
        /// Decimal mode; unused by the NES, but the flag itself still works.
        const D = 1 << 3;
        /// Break: set when pushed by PHP/BRK, clear when pushed by IRQ/NMI.
        const B = 1 << 4;
        /// Unused; always set when pushed to the stack.
        const U = 1 << 5;
        /// Overflow.
        const V = 1 << 6;
        /// Negative.
        const N = 1 << 7;
    }
}
/// The Central Processing Unit status and registers
#[derive(Default, Copy, Clone, Serialize, Deserialize)]
#[must_use]
#[repr(C)]
pub struct Cpu {
    /// Total CPU cycles run.
    pub cycle: u32,
    /// Master clock cycles run, which is what the PPU and APU are caught up against.
    pub master_clock: u32,
    /// Program counter.
    pub pc: u16,
    /// The current instruction's operand.
    pub operand: u16,
    /// Master cycles before a read or write within the current instruction.
    pub start_cycles: u8,
    /// Master cycles after a read or write within the current instruction.
    pub end_cycles: u8,
    /// The current instruction's addressing mode.
    pub addr_mode: AddrMode,
    /// Stack pointer; the stack itself is at $0100-$01FF.
    pub sp: u8,
    /// Accumulator.
    pub acc: u8,
    /// X index register.
    pub x: u8,
    /// Y index register.
    pub y: u8,
    /// Status register.
    pub status: Status,
    /// Interrupt and DMA state.
    pub irq_flags: IrqFlags,
    /// Source page of an OAM DMA in progress.
    pub dma_oam_addr: Option<u16>,
    /// Set when an invalid opcode has jammed the CPU; it keeps cycling but stops fetching.
    pub corrupted: bool,
}

impl Cpu {
    // Represents CPU/PPU alignment and would range from 1..=ppu.clock_divider-1
    // if random PPU alignment was emulated
    // See: https://www.nesdev.org/wiki/PPU_frame_timing#CPU-PPU_Clock_Alignment
    pub(crate) const PPU_OFFSET: u32 = 1;

    /// NMI vector Address.
    pub const NMI_VECTOR: u16 = 0xFFFA;
    /// IRQ vector address
    pub const IRQ_VECTOR: u16 = 0xFFFE;
    /// Reset vector address.
    pub const RESET_VECTOR: u16 = 0xFFFC;
    /// Power on status register values.
    pub const POWER_ON_STATUS: Status = Status::U.union(Status::I);
    /// Power on stack pointer value.
    pub const POWER_ON_SP: u8 = 0xFD;
    /// Stack Pointer base address.
    pub const SP_BASE: u16 = 0x0100;

    /// `JMP` absolute opcode.
    pub const JMP_ABS: u8 = 0x4C;

    /// Create a new CPU.
    pub fn new() -> Self {
        Self {
            cycle: 0,
            master_clock: 0,
            // NTSC_MASTER_CLOCK_DIVIDER / 2
            start_cycles: 6,
            end_cycles: 6,
            pc: 0x0000,
            operand: 0,
            addr_mode: AddrMode::default(),
            sp: 0x00,
            acc: 0x00,
            x: 0x00,
            y: 0x00,
            status: Self::POWER_ON_STATUS,
            irq_flags: IrqFlags::default(),
            dma_oam_addr: None,
            corrupted: false,
        }
    }

    /// Resets the registers.
    ///
    /// Updates the SP and Status values to defined constants. [`Bus::reset`] is what resets the
    /// rest of the console, fetches the reset vector into PC, and runs the seven cycles the reset
    /// itself takes.
    pub fn reset(&mut self, kind: ResetKind) {
        match kind {
            ResetKind::Soft => {
                self.status.set(Status::I, true);
                // Reset pushes to the stack similar to IRQ, but since the read bit is set, nothing is
                // written except the SP being decremented
                self.sp = self.sp.wrapping_sub(0x03);
            }
            ResetKind::Hard => {
                self.acc = 0x00;
                self.x = 0x00;
                self.y = 0x00;
                self.status = Self::POWER_ON_STATUS;
                self.sp = Self::POWER_ON_SP;
            }
        }

        self.cycle = 0;
        self.master_clock = 0;
        self.irq_flags = IrqFlags::default();
        self.corrupted = false;
    }

    /// Start OAM DMA.
    #[inline]
    pub const fn start_oam_dma(&mut self, addr: u16) {
        self.irq_flags = self.irq_flags.union(IrqFlags::DMA_HALT);
        self.dma_oam_addr = Some(addr);
    }

    // Interrupt flag functions

    /// Clear [`IrqFlags`] flags for the given bits.
    #[inline(always)]
    pub(crate) const fn clear_irq_flags(&mut self, flags: IrqFlags) {
        self.irq_flags = self.irq_flags.difference(flags);
    }

    /// Returns `true` if the [`IrqFlags`] register is set.
    #[inline(always)]
    pub(crate) const fn irq_flags(&self, flags: IrqFlags) -> bool {
        self.irq_flags.intersection(flags).bits() == flags.bits()
    }

    // Status Register functions

    /// Set [`Status`] flags for the given bits.
    #[inline(always)]
    pub(crate) const fn set_status(&mut self, status: Status) {
        self.status = status.difference(Status::U).difference(Status::B);
    }

    /// Returns the [`Status`] register as a byte.
    #[inline(always)]
    pub(crate) const fn status_bit(&self, reg: Status) -> u8 {
        self.status.intersection(reg).bits()
    }

    /// Set accumulator and update [`Status`] flags based on value.
    #[inline(always)]
    pub(crate) fn set_acc(&mut self, val: u8) {
        self.set_zn_status(val);
        self.acc = val;
    }

    /// Set x and update [`Status`] flags based on value.
    #[inline(always)]
    pub(crate) fn set_x(&mut self, val: u8) {
        self.set_zn_status(val);
        self.x = val;
    }

    /// Set y and update [`Status`] flags based on value.
    #[inline(always)]
    pub(crate) fn set_y(&mut self, val: u8) {
        self.set_zn_status(val);
        self.y = val;
    }

    /// Set stack pointer.
    #[inline(always)]
    pub(crate) const fn set_sp(&mut self, val: u8) {
        self.sp = val;
    }

    /// Set both [`Status::Z`] and [`Status::N`] flags based on value.
    #[inline(always)]
    pub(crate) fn set_zn_status(&mut self, val: u8) {
        self.status.set(Status::Z, val == 0x00);
        self.status.set(Status::N, val & 0x80 > 0);
    }

    // Utilities

    /// Returns whether two addresses are on different memory pages.
    #[inline(always)]
    #[must_use]
    pub(crate) const fn pages_differ(addr1: u16, addr2: u16) -> bool {
        (addr1 & 0xFF00) != (addr2 & 0xFF00)
    }

    /// Returns whether a memory page is crossed using relative address.
    #[inline(always)]
    #[must_use]
    pub(crate) const fn page_crossed(addr: u16, offset: i16) -> bool {
        ((addr as i16 + offset) as u16 & 0xFF00) != (addr & 0xFF00)
    }
}

/// The CPU's view of the console: every access below moves the whole machine, so it takes the
/// [`Bus`] rather than the [`Cpu`].
impl Bus {
    /// Process an interrupted request.
    ///
    /// <https://wiki.nesdev.org/w/index.php/IRQ>
    ///  #  address R/W description
    /// --- ------- --- -----------------------------------------------
    ///  1    PC     R  fetch PCH
    ///  2    PC     R  fetch PCL
    ///  3  $0100,S  W  push PCH to stack, decrement S
    ///  4  $0100,S  W  push PCL to stack, decrement S
    ///  5  $0100,S  W  push P to stack, decrement S
    ///  6    PC     R  fetch low byte of interrupt vector
    ///  7    PC     R  fetch high byte of interrupt vector
    #[cold]
    #[inline(never)]
    pub fn irq(&mut self) {
        self.read(self.cpu.pc); // Dummy read
        self.read(self.cpu.pc); // Dummy read
        self.push_word(self.cpu.pc);

        // Pushing status to the stack has to happen after checking NMI since it can hijack the BRK
        // IRQ when it occurs between cycles 4 and 5.
        // https://www.nesdev.org/wiki/CPU_interrupts#Interrupt_hijacking
        //
        // Set U and !B during push
        let status = ((self.cpu.status | Status::U) & !Status::B).bits();
        let nmi = self.cpu.irq_flags(IrqFlags::NMI);
        self.push_byte(status);
        self.cpu.status.set(Status::I, true);

        if nmi {
            self.cpu.clear_irq_flags(IrqFlags::NMI);
            self.cpu.pc = self.read_word(Cpu::NMI_VECTOR);
            self.clock_sync();
        } else {
            self.cpu.pc = self.read_word(Cpu::IRQ_VECTOR);
        }
    }

    /// Handle CPU interrupt requests, if any are pending.
    ///
    /// <https://www.nesdev.org/wiki/CPU_interrupts>
    ///
    /// - `PREV_NMI` takes `NMI`: the internal signal goes high during φ1 of the cycle that follows
    ///   the one where the edge is detected, and stays high until the NMI has been handled.
    /// - The edge detector polls the NMI line during φ2 of each cycle (hence here in `end_cycle`)
    ///   and raises `NMI` when it goes from low on the previous cycle to high on this one.
    /// - The IRQ status at the end of the second-to-last cycle is what matters, so `PREV_RUN_IRQ`
    ///   keeps the previous `RUN_IRQ`.
    ///
    /// Done on the flag byte's bits directly: it runs on every CPU cycle.
    #[inline(always)]
    fn handle_interrupts(&mut self) {
        let nmi_pending = u8::from(self.ppu.nmi_pending);
        let run_irq = u8::from(self.apu.irq_pending() & !self.cpu.status.intersects(Status::I));
        let f = self.cpu.irq_flags.bits();
        let nmi = f & IrqFlags::NMI.bits();
        let prev_nmi_pending = (f >> 2) & 1;
        let nmi_edge = !prev_nmi_pending & nmi_pending;
        let kept = IrqFlags::DMA_DMC | IrqFlags::DMA_HALT | IrqFlags::DMA_DUMMY_READ;
        let f = (f & kept.bits())
            | nmi
            | nmi_edge
            | (nmi << 1)
            | (nmi_pending << 2)
            | (run_irq << 3)
            | ((f & IrqFlags::RUN_IRQ.bits()) << 1);
        self.cpu.irq_flags = IrqFlags::from_bits_retain(f);
    }

    /// Start a CPU cycle.
    ///
    /// The PPU is not clocked here: it runs behind and is caught up by [`Bus::ppu_sync`] when
    /// something can observe it. That lands it on exactly the dots per-cycle clocking would have.
    #[inline(always)]
    pub(crate) fn start_cycle(&mut self, increment: u8) {
        self.cpu.master_clock = self.cpu.master_clock.wrapping_add(u32::from(increment));
        self.cpu.cycle = self.cpu.cycle.wrapping_add(1);
        self.cpu_clock();
    }

    /// End a CPU cycle.
    ///
    /// The NMI edge detector polls /NMI here, so the PPU is caught up first if a VBlank edge (one
    /// is also the frame boundary) may have come during the cycle.
    #[inline(always)]
    pub(crate) fn end_cycle(&mut self, increment: u8) {
        self.cpu.master_clock = self.cpu.master_clock.wrapping_add(u32::from(increment));
        if self.cpu.master_clock >= self.ppu_deadline {
            self.ppu_sync();
        }
        if !self.irq_quiet {
            self.handle_interrupts();
        }
    }

    /// Whether [`Bus::handle_interrupts`] would leave everything as it is: no interrupt input
    /// high and no interrupt flag up (the DMA bits are not its business).
    #[inline(always)]
    fn interrupts_quiet(&self) -> bool {
        !self.apu.irq_pending()
            && !self.ppu.nmi_pending
            && self.cpu.irq_flags.bits() & 0x1F == 0
    }

    /// Catches the PPU up to where per-CPU-cycle clocking would have it now.
    ///
    /// Called before anything that observes or changes what the PPU does - a `$2000-$3FFF`
    /// access, a board register write, a VBlank-edge dot falling due, a trace
    /// row - so it sees and changes the same dots it would have clocked one CPU cycle at a time.
    #[inline(always)]
    pub(crate) fn ppu_sync(&mut self) {
        self.ppu_sync_to(self.cpu.master_clock.saturating_sub(Cpu::PPU_OFFSET));
    }

    /// [`Bus::ppu_sync`] to an explicit master clock.
    #[inline(never)]
    pub(crate) fn ppu_sync_to(&mut self, clock: u32) {
        self.ppu_clock_to(clock);
        if self.ppu.nmi_pending {
            self.irq_quiet = false;
        }
        self.set_ppu_deadline();
    }

    /// Sets [`Bus::ppu_deadline`] from the PPU's next VBlank-edge dot.
    #[inline(always)]
    pub(crate) fn set_ppu_deadline(&mut self) {
        self.ppu_deadline = self.ppu.next_event_clock() + Cpu::PPU_OFFSET;
    }

    /// A DMA cycle: the first after the halt clears it, the next the DMC's dummy read; OAM DMA
    /// cycles count as those for a DMC fetch that comes in while it runs.
    #[inline(always)]
    fn start_dma_cycle(&mut self) {
        if self.cpu.irq_flags(IrqFlags::DMA_HALT) {
            self.cpu.clear_irq_flags(IrqFlags::DMA_HALT);
        } else {
            self.cpu.clear_irq_flags(IrqFlags::DMA_DUMMY_READ);
        }
        self.start_cycle(self.cpu.start_cycles - 1);
    }

    /// OAM and DMC DMA (nesdev "DMA"), run when the CPU halts at a read with one pending.
    ///
    /// The halted read is repeated, then DMA runs on get (even) and put (odd) cycles: OAM DMA
    /// copies 256 bytes as get/put pairs after aligning to a get cycle (513 or 514 cycles from a
    /// `$4014` write), a DMC sample fetch reads on a get cycle after its halt and a dummy cycle
    /// (3 or 4 cycles alone, fewer inside an OAM DMA, whose cycles count as its halt and dummy).
    /// The CPU's repeated reads do not repeat for `$4016`/`$4017` (the joypads would shift).
    #[cold]
    #[inline(never)]
    fn handle_dma(&mut self, addr: u16) {
        self.start_cycle(self.cpu.start_cycles - 1);
        self.cpu_bus_read(addr);
        self.end_cycle(self.cpu.start_cycles + 1);
        self.cpu.clear_irq_flags(IrqFlags::DMA_HALT);

        let skip_dummy_reads = addr == 0x4016 || addr == 0x4017;
        let mut oam_offset = 0;
        let mut oam_dma_count = 0;
        let mut read_val = 0;
        loop {
            let dma_dmc = self.cpu.irq_flags(IrqFlags::DMA_DMC);
            let dma_oam_addr = self.cpu.dma_oam_addr;
            if !dma_dmc && dma_oam_addr.is_none() {
                break;
            }
            if self.cpu.cycle & 0x01 == 0x00 {
                if dma_dmc
                    && !self.cpu.irq_flags(IrqFlags::DMA_HALT)
                    && !self.cpu.irq_flags(IrqFlags::DMA_DUMMY_READ)
                {
                    // The DMC's fetch, its halt and dummy cycles done.
                    self.start_dma_cycle();
                    let sample = self.cpu_bus_read(self.apu.dmc.dma_addr());
                    self.end_cycle(self.cpu.start_cycles + 1);
                    if self.apu.dmc.bytes_remaining > 0 {
                        self.sound.sample(self.state.apu.cpu_cycle, sample);
                    }
                    self.apu.dmc.load_buffer();
                    self.cpu.clear_irq_flags(IrqFlags::DMA_DMC);
                    // The buffer moved the DMC's next fetch, and the last byte may raise its IRQ.
                    self.tick_deadline = 0;
                    self.irq_quiet = false;
                } else if let Some(oam_addr) = dma_oam_addr {
                    self.start_dma_cycle();
                    read_val = self.cpu_bus_read(oam_addr + oam_offset);
                    self.end_cycle(self.cpu.start_cycles + 1);
                    oam_offset += 1;
                    oam_dma_count += 1;
                } else {
                    // A DMC fetch still halting or on its dummy cycle, no OAM DMA.
                    self.start_dma_cycle();
                    if !skip_dummy_reads {
                        self.cpu_bus_read(addr);
                    }
                    self.end_cycle(self.cpu.start_cycles + 1);
                }
            } else if dma_oam_addr.is_some() && oam_dma_count & 0x01 == 0x01 {
                // OAM DMA's put cycle after each get.
                self.start_dma_cycle();
                self.cpu_bus_write(0x2004, read_val);
                self.end_cycle(self.cpu.start_cycles + 1);
                oam_dma_count += 1;
                if oam_dma_count == 0x200 {
                    self.cpu.dma_oam_addr = None;
                }
            } else {
                // Align to a get cycle for OAM DMA or the DMC's fetch.
                self.start_dma_cycle();
                if !skip_dummy_reads {
                    self.cpu_bus_read(addr);
                }
                self.end_cycle(self.cpu.start_cycles + 1);
            }
        }
    }

    // Stack Functions

    /// Push a byte to the stack.
    #[inline(always)]
    pub(crate) fn push_byte(&mut self, val: u8) {
        self.write(Cpu::SP_BASE | u16::from(self.cpu.sp), val);
        self.cpu.sp = self.cpu.sp.wrapping_sub(1);
    }

    /// Pull a byte from the stack.
    #[inline(always)]
    #[must_use]
    pub(crate) fn pop_byte(&mut self) -> u8 {
        self.cpu.sp = self.cpu.sp.wrapping_add(1);
        self.read(Cpu::SP_BASE | u16::from(self.cpu.sp))
    }

    /// Push a word (two bytes) to the stack
    #[inline(always)]
    pub(crate) fn push_word(&mut self, val: u16) {
        let [lo, hi] = val.to_le_bytes();
        self.push_byte(hi);
        self.push_byte(lo);
    }

    /// Pull a word (two bytes) from the stack
    #[inline(always)]
    pub(crate) fn pop_word(&mut self) -> u16 {
        let lo = self.pop_byte();
        let hi = self.pop_byte();
        u16::from_le_bytes([lo, hi])
    }

    // Memory accesses

    /// Read a byte, spending a full CPU cycle - which clocks the PPU, the APU and the board.
    #[inline(always)]
    pub fn read(&mut self, addr: u16) -> u8 {
        if self.cpu.irq_flags(IrqFlags::DMA_HALT) {
            self.handle_dma(addr);
        }

        self.start_cycle(self.cpu.start_cycles - 1);
        let val = self.cpu_bus_read(addr);
        self.end_cycle(self.cpu.end_cycles + 1);
        val
    }

    /// Read a byte without side effects, and without moving the console.
    #[inline(always)]
    #[must_use]
    pub fn peek(&self, addr: u16) -> u8 {
        self.cpu_bus_peek(addr)
    }

    /// Write a byte, spending a full CPU cycle - which clocks the PPU, the APU and the board.
    #[inline(always)]
    pub fn write(&mut self, addr: u16, val: u8) {
        self.start_cycle(self.cpu.start_cycles + 1);
        if addr == 0x4014 {
            self.cpu.start_oam_dma(u16::from(val) << 8);
        } else {
            self.cpu_bus_write(addr, val);
        }
        self.end_cycle(self.cpu.end_cycles - 1);
    }

    /// Fetch a byte and increments PC by 1.
    #[inline(always)]
    #[must_use]
    pub(crate) fn fetch_byte(&mut self) -> u8 {
        let val = self.read(self.cpu.pc);
        self.cpu.pc = self.cpu.pc.wrapping_add(1);
        val
    }

    /// Fetch opcode operand based on addressing mode (a constant in each [`Bus::exec`], so this folds away).
    #[inline(always)]
    #[must_use]
    fn fetch_operand(&mut self, mode: AddrMode) -> u16 {
        match mode {
            AddrMode::ACC | AddrMode::IMP => self.acc_imp(),
            AddrMode::IMM | AddrMode::REL | AddrMode::ZP0 => self.imm_rel_zp(),
            AddrMode::ZPX => self.zpx(),
            AddrMode::ZPY => self.zpy(),
            AddrMode::IND => self.ind(),
            AddrMode::IDX => self.idx(),
            AddrMode::IDY => self.idy(false),
            AddrMode::IDYW => self.idy(true),
            AddrMode::ABS => self.abs(),
            AddrMode::ABX => self.abx(false),
            AddrMode::ABXW => self.abx(true),
            AddrMode::ABY => self.aby(false),
            AddrMode::ABYW => self.aby(true),
            AddrMode::OTH => 0,
        }
    }

    /// Fetch a 16-bit word and increments PC by 2.
    #[inline(always)]
    #[must_use]
    pub(crate) fn fetch_word(&mut self) -> u16 {
        let lo = self.fetch_byte();
        let hi = self.fetch_byte();
        u16::from_le_bytes([lo, hi])
    }

    /// Read operand value.
    #[inline(always)]
    #[must_use]
    pub(crate) fn read_operand(&mut self) -> u8 {
        if matches!(
            self.cpu.addr_mode,
            AddrMode::ACC | AddrMode::IMP | AddrMode::IMM | AddrMode::REL
        ) {
            self.cpu.operand as u8
        } else {
            self.read(self.cpu.operand)
        }
    }

    /// Read a 16-bit word.
    #[inline(always)]
    #[must_use]
    pub fn read_word(&mut self, addr: u16) -> u16 {
        let lo = self.read(addr);
        let hi = self.read(addr.wrapping_add(1));
        u16::from_le_bytes([lo, hi])
    }

    /// Hands the instruction about to run to the trace hook.
    #[cold]
    #[inline(never)]
    fn trace_instr(&mut self) {
        let c = &self.cpu;
        let row = TraceRow {
            pc: c.pc,
            opcode: self.peek(c.pc),
            a: c.acc,
            x: c.x,
            y: c.y,
            p: c.status.bits(),
            sp: c.sp,
            cycle: c.cycle,
            scanline: self.ppu.scanline,
            dot: self.ppu.cycle,
        };
        if let Some(hook) = self.trace.as_mut() {
            hook(&row);
        }
    }

    /// Runs all components up to master clock, synchronizing them.
    #[inline(always)]
    pub fn clock_sync(&mut self) {
        self.ppu_sync_to(self.cpu.master_clock);
        self.cpu.master_clock = self.cpu.master_clock.saturating_sub(self.ppu.master_clock);
        self.ppu.master_clock = 0;
        self.apu.clock_sync();
        self.tick_deadline = 0;
        self.irq_quiet = false;
        self.set_ppu_deadline();
    }

    /// Runs instructions until the frame counter advances (VBlank starts), then synchronizes.
    pub fn clock_frame(&mut self) {
        // The PPU is caught up between frames; anything may have changed it, the APU or the
        // interrupt state since.
        self.set_ppu_deadline();
        self.tick_deadline = 0;
        self.irq_quiet = false;
        let frame = self.ppu.frame_number();
        while frame == self.ppu.frame_number() {
            self.step_instr();
        }
        self.clock_sync();
    }

    /// Runs the CPU one instruction, leaving the PPU caught up with it.
    pub fn clock_instr(&mut self) {
        self.set_ppu_deadline();
        self.tick_deadline = 0;
        self.irq_quiet = false;
        self.step_instr();
        self.ppu_sync();
    }

    /// Runs the CPU one instruction, catching the PPU up only as far as the instruction needed.
    #[inline(always)]
    fn step_instr(&mut self) {
        if self.trace.is_some() {
            self.ppu_sync();
            self.trace_instr();
        }

        let pc = self.cpu.pc;
        let opcode = self.fetch_byte(); // Cycle 1
        Cpu::EXEC[usize::from(opcode)](self);

        if self
            .cpu
            .irq_flags
            .intersects(IrqFlags::PREV_RUN_IRQ | IrqFlags::PREV_NMI)
        {
            self.irq();
        }
        if !self.irq_quiet {
            self.irq_quiet = self.interrupts_quiet();
        }
        if opcode == Cpu::JMP_ABS && self.cpu.pc == pc {
            self.skip_idle_loop(pc);
        }
    }

    /// One instruction after its opcode fetch, specialised per opcode: the addressing mode and the operation
    /// are constants here, so the operand fetch is direct and the operation's call can be inlined, leaving
    /// [`Cpu::EXEC`]'s one indirect call per instruction.
    #[inline(always)]
    pub(crate) fn exec<const OP: u8>(&mut self) {
        let op = const { Cpu::OPS[OP as usize] };
        self.cpu.addr_mode = op.addr_mode();
        self.cpu.operand = self.fetch_operand(op.addr_mode());
        op.run(self);
    }

    /// A `JMP` to itself in ROM (Zelda's wait for the next NMI, ~90% of a gameplay frame's
    /// instructions) only reads its own three bytes. While nothing can happen - interrupts quiet,
    /// no DMA, and no PPU or APU/board deadline before the end - each iteration does nothing but
    /// count: 3 CPU cycles, 3 APU ticks, `3 × (start + end)` master clocks, the operand fields and
    /// open bus left at the target's high byte. Called right after one ran (so the next
    /// instruction is the same jump), this applies as many whole iterations as fit before the
    /// nearer deadline at once; the ones that reach a deadline run as ordinary instructions, so
    /// every observable state is the same as stepping them one cycle at a time.
    #[inline(always)]
    fn skip_idle_loop(&mut self, pc: u16) {
        if self.irq_quiet
            && self.trace.is_none()
            && pc >= 0x8000
            && pc <= 0xFFFD
            && self.cpu.irq_flags.bits() == 0
            && !self.cpu.corrupted
        {
            self.skip_jmp_self(pc);
        }
    }

    #[inline(never)]
    fn skip_jmp_self(&mut self, pc: u16) {
        // A read is `start_cycles - 1` master clocks, the bus access, then `end_cycles + 1`.
        let per_iteration = 3 * (u32::from(self.cpu.start_cycles) + u32::from(self.cpu.end_cycles));
        // After its last cycle the APU counter must still be short of `tick_deadline` (checked
        // after each tick) and the master clock short of `ppu_deadline` (checked at each end).
        let by_apu = self
            .tick_deadline
            .saturating_sub(self.apu.master_clock)
            .saturating_sub(1)
            / 3;
        let by_ppu = self
            .ppu_deadline
            .saturating_sub(self.cpu.master_clock)
            .saturating_sub(1)
            / per_iteration;
        let n = by_apu.min(by_ppu);
        if n == 0 {
            return;
        }
        self.cpu.master_clock = self.cpu.master_clock.wrapping_add(n * per_iteration);
        self.cpu.cycle = self.cpu.cycle.wrapping_add(3 * n);
        self.apu.cpu_cycle = self.apu.cpu_cycle.wrapping_add(3 * n);
        self.apu.master_clock += 3 * n;
        self.open_bus = self.prg_peek(pc + 2);
        self.cpu.addr_mode = AddrMode::ABS;
        self.cpu.operand = pc;
    }
}

impl fmt::Debug for Cpu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::result::Result<(), fmt::Error> {
        f.debug_struct("Cpu")
            .field("cycle", &self.cycle)
            .field("pc", &format_args!("${:04X}", self.pc))
            .field("sp", &format_args!("${:02X}", self.sp))
            .field("acc", &format_args!("${:02X}", self.acc))
            .field("x", &format_args!("${:02X}", self.x))
            .field("y", &format_args!("${:02X}", self.y))
            .field("status", &self.status)
            .field("interrupt_flags", &self.irq_flags)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use crate::{cart::Cart, cpu::instr::Instr::*, mapper::Nrom};

    /// An instruction is as long as its addressing mode's operand says.
    #[test]
    fn every_opcodes_length_matches_what_executing_it_consumes() {
        use super::*;
        let mut bus = Bus::default();
        let mut cart = Cart::empty();
        cart.mapper = Nrom::load(&mut cart).unwrap();
        bus.load_cart(cart);

        for instr_ref in Cpu::INSTR_REF.iter() {
            // A jam never leaves the instruction, and a branch or jump moves PC by its own rules.
            if matches!(
                instr_ref.instr,
                HLT | JMP | JSR | RTS | RTI | BRK | BCC | BCS | BEQ | BMI | BNE | BPL | BVC | BVS
            ) {
                continue;
            }
            bus.reset(ResetKind::Hard);
            bus.cpu_bus_write(0x0000, instr_ref.opcode);
            bus.cpu.pc = 0x0000;
            let len = 1 + u16::from(instr_ref.addr_mode.operand_len());
            bus.clock_instr();
            let consumed = bus.cpu.pc;
            assert_eq!(
                len, consumed,
                "${:02X} {:?} #{:?} runs {consumed} bytes",
                instr_ref.opcode, instr_ref.instr, instr_ref.addr_mode
            );
        }
    }

    #[test]
    fn cycle_timing() {
        use super::*;
        let mut bus = Bus::default();
        let mut cart = Cart::empty();
        cart.mapper = Nrom::load(&mut cart).unwrap();
        bus.load_cart(cart);
        bus.reset(ResetKind::Hard);
        bus.clock_instr();

        assert_eq!(bus.cpu.cycle, 14, "cpu after power + one clock");

        for instr_ref in Cpu::INSTR_REF.iter() {
            let extra_cycle = match instr_ref.instr {
                BCC | BNE | BPL | BVC => 1,
                _ => 0,
            };
            // Ignore invalid opcodes
            if instr_ref.instr == HLT {
                continue;
            }
            bus.reset(ResetKind::Hard);
            bus.cpu_bus_write(0x0000, instr_ref.opcode);
            bus.clock_instr();
            let cpu_cyc = u32::from(7 + instr_ref.cycles + extra_cycle);
            assert_eq!(
                bus.cpu.cycle, cpu_cyc,
                "cpu ${:02X} {:?} #{:?}",
                instr_ref.opcode, instr_ref.instr, instr_ref.addr_mode
            );
        }
    }

    /// OAM DMA halts the CPU at the read after `STA $4014` for 513 cycles, 514 when it halts on
    /// a put (odd) cycle (nesdev "DMA").
    #[test]
    fn oam_dma_takes_513_or_514_cycles_from_the_next_read() {
        use super::*;
        let mut lens = Vec::new();
        // With and without a 3-cycle `LDA $00` first, so the halt lands on both cycle parities.
        for prefix in [&[][..], &[0xA5, 0x00][..]] {
            let mut bus = Bus::default();
            let mut cart = Cart::empty();
            cart.mapper = Nrom::load(&mut cart).unwrap();
            bus.load_cart(cart);
            bus.reset(ResetKind::Hard);
            let code: Vec<u8> = [prefix, &[0x8D, 0x14, 0x40, 0xEA][..]].concat();
            for (addr, &val) in (0x0010u16..).zip(&code) {
                bus.cpu_bus_write(addr, val);
            }
            bus.cpu.pc = 0x0010;
            bus.cpu.acc = 0x02;
            if !prefix.is_empty() {
                bus.clock_instr();
            }
            let before = bus.cpu.cycle;
            bus.clock_instr();
            assert_eq!(bus.cpu.cycle - before, 4, "the STA itself");
            let before = bus.cpu.cycle;
            bus.clock_instr();
            lens.push(bus.cpu.cycle - before - 2);
        }
        lens.sort_unstable();
        assert_eq!(lens, [513, 514]);
    }

    /// A playing DMC sample halts the CPU for each byte, 3 or 4 cycles between NOPs, once per
    /// eight timer periods once the first byte is shifting, and its last byte raises the DMC IRQ
    /// (nesdev "APU DMC", "DMA").
    #[test]
    fn dmc_fetches_halt_the_cpu_once_per_byte() {
        use super::*;
        let mut bus = Bus::default();
        let mut cart = Cart::empty();
        cart.mapper = Nrom::load(&mut cart).unwrap();
        bus.load_cart(cart);
        bus.reset(ResetKind::Hard);
        // NOPs from $0010, looping back with `JMP $0010` at $07FD.
        for addr in 0x0010u16..0x07FD {
            bus.cpu_bus_write(addr, 0xEA);
        }
        for (addr, val) in [(0x07FDu16, 0x4Cu8), (0x07FE, 0x10), (0x07FF, 0x00)] {
            bus.cpu_bus_write(addr, val);
        }
        bus.cpu.pc = 0x0010;
        // IRQ on, fastest rate (54 cycles); 17 bytes from $C000.
        for (addr, val) in [(0x4010u16, 0x8Fu8), (0x4012, 0x00), (0x4013, 0x01), (0x4015, 0x10)] {
            bus.cpu_bus_write(addr, val);
        }
        let mut fetches = Vec::new();
        for _ in 0..4000 {
            let base = if bus.cpu.pc == 0x07FD { 3 } else { 2 };
            let before = bus.cpu.cycle;
            bus.clock_instr();
            let stolen = bus.cpu.cycle - before - base;
            if stolen > 0 {
                assert!((3..=4).contains(&stolen), "a fetch stole {stolen} cycles");
                fetches.push(before);
            }
        }
        assert_eq!(fetches.len(), 17, "{fetches:?}");
        // The second fetch waits for the first byte to reach the shifter.
        for gap in fetches.windows(2).skip(1).map(|w| w[1] - w[0]) {
            assert!((8 * 54 - 4..=8 * 54 + 4).contains(&gap), "gap {gap} in {fetches:?}");
        }
        assert!(bus.apu.dmc.irq_pending && bus.apu.peek_status() & 0x80 != 0);
        assert_eq!(bus.apu.dmc.bytes_remaining, 0);
    }

    /// A bus with an empty NROM cart and `program` at $0200, run to its end after `setup` has
    /// set registers and memory. No branches, so "PC past the end" is the end.
    fn run(program: &[u8], setup: impl FnOnce(&mut super::Bus)) -> super::Bus {
        use super::*;
        let mut bus = Bus::default();
        let mut cart = Cart::empty();
        cart.mapper = Nrom::load(&mut cart).unwrap();
        bus.load_cart(cart);
        bus.reset(ResetKind::Hard);
        for (addr, &val) in (0x0200u16..).zip(program) {
            bus.cpu_bus_write(addr, val);
        }
        bus.cpu.pc = 0x0200;
        setup(&mut bus);
        let end = 0x0200 + program.len() as u16;
        while bus.cpu.pc < end {
            bus.clock_instr();
        }
        bus
    }

    /// The NES 6502 has no BCD: ADC and SBC (and the unofficial SBC $EB) add in binary with the
    /// decimal flag set, and carry, overflow, zero and negative follow signed/unsigned overflow.
    #[test]
    fn adc_and_sbc_are_binary_with_carry_and_overflow() {
        use super::*;
        let nzcv = Status::N | Status::Z | Status::C | Status::V;
        #[rustfmt::skip]
        let cases = [
            // opcode, A, operand, carry in, result, flags
            (0x69, 0x50, 0x50, false, 0xA0, Status::V | Status::N),
            (0x69, 0xFF, 0x01, false, 0x00, Status::C | Status::Z),
            (0x69, 0x80, 0x80, false, 0x00, Status::C | Status::V | Status::Z),
            (0x69, 0x7F, 0x00, true,  0x80, Status::V | Status::N),
            (0x69, 0x09, 0x01, false, 0x0A, Status::empty()), // BCD would give $10
            (0xE9, 0x50, 0xF0, true,  0x60, Status::empty()),
            (0xE9, 0x50, 0xB0, true,  0xA0, Status::V | Status::N),
            (0xE9, 0x00, 0x01, true,  0xFF, Status::N),
            (0xE9, 0x10, 0x01, false, 0x0E, Status::C), // borrow in
            (0xE9, 0x80, 0x01, true,  0x7F, Status::C | Status::V),
            (0xEB, 0x80, 0x01, true,  0x7F, Status::C | Status::V),
        ];
        for (opcode, a, operand, carry, result, flags) in cases {
            let bus = run(&[opcode, operand], |bus| {
                bus.cpu.acc = a;
                bus.cpu.status.set(Status::D, true);
                bus.cpu.status.set(Status::C, carry);
            });
            let case = format!("${opcode:02X} A=${a:02X} #${operand:02X} C={carry}");
            assert_eq!(bus.cpu.acc, result, "{case}");
            assert_eq!(bus.cpu.status & nzcv, flags, "{case}");
            assert!(bus.cpu.status.contains(Status::D), "{case}: D untouched");
        }
    }

    /// The stable unofficial read-modify-write and immediate opcodes, each against the official
    /// pair it fuses.
    #[test]
    fn unofficial_opcodes_fuse_their_official_pairs() {
        use super::*;
        let zp = |val: u8, a: u8, x: u8, carry: bool| {
            move |bus: &mut Bus| {
                bus.cpu_bus_write(0x80, val);
                bus.cpu.acc = a;
                bus.cpu.x = x;
                bus.cpu.status.set(Status::C, carry);
            }
        };
        let c = |bus: &Bus| bus.cpu.status.contains(Status::C);

        let b = run(&[0xA7, 0x80], zp(0x8F, 0, 0, false)); // LAX: LDA + TAX
        assert_eq!((b.cpu.acc, b.cpu.x), (0x8F, 0x8F), "LAX");
        assert!(b.cpu.status.contains(Status::N), "LAX sets N");

        let b = run(&[0x87, 0x80], zp(0, 0xF0, 0x3C, false)); // SAX: store A & X
        assert_eq!(b.peek(0x80), 0x30, "SAX");

        let b = run(&[0xC7, 0x80], zp(0x41, 0x40, 0, false)); // DCP: DEC + CMP
        assert_eq!(b.peek(0x80), 0x40, "DCP decrements");
        assert!(b.cpu.status.contains(Status::Z) && c(&b), "DCP compares equal");

        let b = run(&[0xE7, 0x80], zp(0x0F, 0x20, 0, true)); // ISB: INC + SBC
        assert_eq!((b.peek(0x80), b.cpu.acc, c(&b)), (0x10, 0x10, true), "ISB");

        let b = run(&[0x07, 0x80], zp(0x81, 0x01, 0, false)); // SLO: ASL + ORA
        assert_eq!((b.peek(0x80), b.cpu.acc, c(&b)), (0x02, 0x03, true), "SLO");

        let b = run(&[0x27, 0x80], zp(0x81, 0xFF, 0, false)); // RLA: ROL + AND
        assert_eq!((b.peek(0x80), b.cpu.acc, c(&b)), (0x02, 0x02, true), "RLA");

        let b = run(&[0x47, 0x80], zp(0x03, 0x0F, 0, false)); // SRE: LSR + EOR
        assert_eq!((b.peek(0x80), b.cpu.acc, c(&b)), (0x01, 0x0E, true), "SRE");

        // RRA: ROR (carry out 1) + ADC with that carry.
        let b = run(&[0x67, 0x80], zp(0x03, 0x10, 0, false));
        assert_eq!((b.peek(0x80), b.cpu.acc, c(&b)), (0x01, 0x12, false), "RRA");

        let b = run(&[0x0B, 0x80], zp(0, 0xFF, 0, false)); // ANC: AND, C = bit 7
        assert_eq!((b.cpu.acc, c(&b)), (0x80, true), "ANC");

        let b = run(&[0x4B, 0x03], zp(0, 0xFF, 0, false)); // ALR: AND + LSR
        assert_eq!((b.cpu.acc, c(&b)), (0x01, true), "ALR");

        // ARR: AND + ROR, C from bit 6 and V from bit 6 ^ bit 5 of the result.
        let b = run(&[0x6B, 0xFF], zp(0, 0xC0, 0, true));
        assert_eq!((b.cpu.acc, c(&b)), (0xE0, true), "ARR");
        assert!(!b.cpu.status.contains(Status::V), "ARR V = 1 ^ 1");
        let b = run(&[0x6B, 0xFF], zp(0, 0x40, 0, false));
        assert_eq!((b.cpu.acc, c(&b)), (0x20, false), "ARR");
        assert!(b.cpu.status.contains(Status::V), "ARR V = 0 ^ 1");

        let b = run(&[0xCB, 0x02], zp(0, 0x0F, 0x06, false)); // AXS: X = (A & X) - imm
        assert_eq!((b.cpu.x, b.cpu.acc, c(&b)), (0x04, 0x0F, true), "AXS");
        let b = run(&[0xCB, 0x07], zp(0, 0x0F, 0x06, true));
        assert_eq!((b.cpu.x, c(&b)), (0xFF, false), "AXS borrows without carry in");

        // LAS: A, X and SP all get memory & SP.
        let b = run(&[0xBB, 0x80, 0x00], |bus: &mut Bus| {
            bus.cpu_bus_write(0x80, 0x0F);
            bus.cpu.sp = 0xFD;
            bus.cpu.y = 0;
        });
        assert_eq!((b.cpu.acc, b.cpu.x, b.cpu.sp), (0x0D, 0x0D, 0x0D), "LAS");
    }

    /// A jam opcode holds the CPU on itself: cycles keep passing, but PC and the registers stay.
    #[test]
    fn a_jam_opcode_holds_the_cpu_in_place() {
        let mut bus = run(&[], |_| ());
        bus.cpu_bus_write(0x0300, 0x02);
        bus.cpu.pc = 0x0300;
        bus.cpu.acc = 0x42;
        let before = bus.cpu.cycle;
        for _ in 0..10 {
            bus.clock_instr();
        }
        assert_eq!(bus.cpu.pc, 0x0300, "still on the jam");
        assert_eq!(bus.cpu.acc, 0x42);
        assert!(bus.cpu.cycle - before >= 10, "time passes");
        assert!(bus.cpu.corrupted, "the jam is reported");
    }
}
