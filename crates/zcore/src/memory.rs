//! Cartridge and console-side video memory, and the page tables that address it.
//!
//! zcore's memory is a fixed layout (ladder stage Z5), part of the `Copy` machine state:
//!
//! | Region | Size | Where |
//! |---|---|---|
//! | PRG-ROM | 128 KiB | outside the state: the `Bus` holds an `Arc<PrgRom>`; `rom_pages` index it |
//! | zero page | 1 KiB | arena `$0000`: what an unmapped page reads (0) |
//! | PRG-RAM | 8 KiB | arena `$0400` |
//! | CHR (RAM, or ROM copied in for the test carts) | 8 KiB | arena `$2400` |
//! | CIRAM | 2 KiB | arena `$4400` |
//!
//! Address translation goes through page tables of 1 KiB entries, so a read is a table lookup and
//! an indexed load with no dispatch on the mapper:
//!
//! ```ignore
//! let page = self.chr_pages[(addr as usize >> PAGE_SHIFT) & CHR_PAGE_MASK];
//! self.arena()[page.offset() | (addr as usize & PAGE_OFFSET_MASK)]
//! ```
//!
//! Boards are pure register state: on a register write they call [`Memory::map_prg`] /
//! [`Memory::map_chr`] to rewrite a few page entries. CPU `$8000-$FFFF` is always PRG-ROM
//! (`rom_pages`, byte offsets into the image); `$0000-$7FFF` is arena pages (`prg_pages`, only
//! `$6000-$7FFF` is ever mapped). Nametable mirroring is CHR page entries pointing into CIRAM.
//!
//! Alongside this module holds [`ConstArray`], the plain byte storage the rest of the emulator
//! shares.

use crate::ppu::Mirroring;
use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{SeqAccess, Visitor},
    ser::SerializeTuple,
};
use std::{
    fmt,
    marker::PhantomData,
    ops::{Deref, DerefMut, Index, IndexMut, Range, RangeInclusive},
};

/// Address bits translated within a single page. 1 KiB granularity.
pub const PAGE_SHIFT: usize = 10;
/// Size of a single translation page.
pub const PAGE_SIZE: usize = 1 << PAGE_SHIFT;
/// Mask selecting the offset within a page.
pub const PAGE_OFFSET_MASK: usize = PAGE_SIZE - 1;

/// Number of CHR page entries, covering `$0000-$3FFF`.
pub const CHR_PAGES: usize = 0x4000 >> PAGE_SHIFT;
/// Number of PRG-ROM page entries, covering `$8000-$FFFF`.
pub const ROM_PAGES: usize = 0x8000 >> PAGE_SHIFT;
/// Number of arena-backed CPU page entries, covering `$0000-$7FFF`.
pub const RAM_PAGES: usize = 0x8000 >> PAGE_SHIFT;

const CHR_PAGE_MASK: usize = CHR_PAGES - 1;

/// First nametable page in the CHR table, i.e. `$2000`.
const NAMETABLE_PAGE: usize = 0x2000 >> PAGE_SHIFT;

/// Bytes of PRG-ROM in a [`PrgRom`] image. Smaller power-of-two ROMs are mirrored to fill it, which
/// leaves every bank number wrapping onto the same bytes it would in the smaller ROM.
pub const PRG_ROM_SIZE: usize = 0x2_0000;
/// Bytes of PRG-RAM.
pub const PRG_RAM_SIZE: usize = 0x2000;
/// Bytes of CHR.
pub const CHR_SIZE: usize = 0x2000;
/// Bytes of CIRAM (the console's two nametables).
pub const CIRAM_SIZE: usize = 0x800;

const PRG_RAM_START: usize = PAGE_SIZE;
const CHR_START: usize = PRG_RAM_START + PRG_RAM_SIZE;
const CIRAM_START: usize = CHR_START + CHR_SIZE;
/// Bytes of the arena: the zero page, PRG-RAM, CHR and CIRAM.
pub const ARENA_SIZE: usize = CIRAM_START + CIRAM_SIZE;

/// A PRG-ROM image, mirrored up to [`PRG_ROM_SIZE`].
pub type PrgRom = [u8; PRG_ROM_SIZE];

/// Which backing region a mapping refers to.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[must_use]
pub enum Src {
    /// Program ROM.
    PrgRom,
    /// Program RAM.
    PrgRam,
    /// CHR-ROM or CHR-RAM, whichever the cart provides.
    Chr,
    /// Nametable RAM.
    CiRam,
}

impl Src {
    /// Start and length of the region (in the arena, or in the PRG-ROM image for `PrgRom`).
    const fn region(self) -> (usize, usize) {
        match self {
            Self::PrgRom => (0, PRG_ROM_SIZE),
            Self::PrgRam => (PRG_RAM_START, PRG_RAM_SIZE),
            Self::Chr => (CHR_START, CHR_SIZE),
            Self::CiRam => (CIRAM_START, CIRAM_SIZE),
        }
    }
}

/// An arena page table entry: a 1 KiB-aligned byte offset into the arena plus a writable flag.
///
/// An unmapped entry is all zeroes, which points at the zero-filled page at offset 0 and is not
/// writable. That keeps reads branchless - an unmapped read yields 0 - while writes still have to
/// test the writable flag.
#[derive(Default, Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[must_use]
pub struct Page(u16);

impl Page {
    const WRITABLE: u16 = 1 << 15;

    /// An unmapped page: reads as zero, ignores writes.
    pub const UNMAPPED: Self = Self(0);

    const fn new(offset: usize, writable: bool) -> Self {
        let offset = offset as u16 & !Self::WRITABLE;
        Self(if writable {
            offset | Self::WRITABLE
        } else {
            offset
        })
    }

    /// Byte offset of this page within the arena.
    #[inline(always)]
    pub const fn offset(self) -> usize {
        (self.0 & !Self::WRITABLE) as usize
    }

    /// Whether writes to this page are stored.
    #[inline(always)]
    pub const fn is_writable(self) -> bool {
        self.0 & Self::WRITABLE != 0
    }
}

/// Cartridge memory and CIRAM, plus the page tables that address them.
#[derive(Copy, Clone, Serialize, Deserialize)]
#[must_use]
pub struct Memory {
    /// `$8000-$FFFF`: byte offsets into the PRG-ROM image (which the `Bus` holds).
    rom_pages: [u32; ROM_PAGES],
    /// `$0000-$7FFF`: arena pages (work RAM below `$2000` never comes here).
    prg_pages: [Page; RAM_PAGES],
    /// `$0000-$3FFF` of the PPU.
    chr_pages: [Page; CHR_PAGES],
    /// Whether the CHR region is RAM.
    chr_writable: bool,
    /// Whether the cart's RAM is battery-backed, and so survives a power cycle.
    battery_backed: bool,
    /// Zero page, PRG-RAM, CHR, CIRAM. Read through [`Memory::arena`]: `ConstArray`'s own `Index`
    /// masks with `N - 1`, which is only right for a power-of-two `N`.
    data: ConstArray<u8, ARENA_SIZE>,
}

impl Default for Memory {
    fn default() -> Self {
        Self::new(true)
    }
}

/// Reports the layout flags rather than the contents.
impl fmt::Debug for Memory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Memory")
            .field("chr_writable", &self.chr_writable)
            .field("battery_backed", &self.battery_backed)
            .finish_non_exhaustive()
    }
}

impl Memory {
    /// Memory with all pages unmapped except the nametables (horizontal mirroring), and every RAM
    /// byte zero.
    pub fn new(chr_writable: bool) -> Self {
        let mut memory = Self {
            rom_pages: [0; ROM_PAGES],
            prg_pages: [Page::UNMAPPED; RAM_PAGES],
            chr_pages: [Page::UNMAPPED; CHR_PAGES],
            chr_writable,
            battery_backed: false,
            data: ConstArray::filled(0),
        };
        // CIRAM is console-internal, not part of the cart, so the nametables are readable before
        // any board has mapped anything.
        memory.set_mirroring(Mirroring::default());
        memory
    }

    /// Whether every mapped PRG/CHR page lies inside the arena (a decoded state is checked with it).
    pub fn pages_in_arena(&self) -> bool {
        self.prg_pages.iter().chain(&self.chr_pages).all(|p| p.offset() + PAGE_SIZE <= ARENA_SIZE)
    }

    /// Points the first PRG page past the arena (0x7C00), as a corrupt state could.
    #[cfg(test)]
    pub(crate) fn break_page_for_test(&mut self) {
        self.prg_pages[0] = Page(0x7C00);
    }

    #[inline(always)]
    fn arena(&self) -> &[u8; ARENA_SIZE] {
        &self.data
    }

    #[inline(always)]
    fn arena_mut(&mut self) -> &mut [u8; ARENA_SIZE] {
        &mut self.data
    }

    /// Where in the PRG-ROM image the CPU address `addr` (`$8000-$FFFF`) reads.
    #[inline(always)]
    #[must_use]
    pub const fn rom_index(&self, addr: u16) -> usize {
        let addr = addr as usize;
        let offset = self.rom_pages[(addr >> PAGE_SHIFT) & (ROM_PAGES - 1)] as usize;
        (offset | (addr & PAGE_OFFSET_MASK)) & (PRG_ROM_SIZE - 1)
    }

    /// Read a byte from the CPU address space below `$8000` (cart RAM, or 0 where unmapped).
    #[inline(always)]
    pub fn prg_peek(&self, addr: u16) -> u8 {
        let addr = usize::from(addr) & 0x7FFF;
        let page = self.prg_pages[addr >> PAGE_SHIFT];
        self.arena()[page.offset() | (addr & PAGE_OFFSET_MASK)]
    }

    /// Write a byte to the CPU address space. Writes to ROM or unmapped pages are discarded.
    #[inline(always)]
    pub fn prg_write(&mut self, addr: u16, val: u8) {
        let addr = usize::from(addr);
        if addr < 0x8000 {
            let page = self.prg_pages[addr >> PAGE_SHIFT];
            if page.is_writable() {
                self.arena_mut()[page.offset() | (addr & PAGE_OFFSET_MASK)] = val;
            }
        }
    }

    /// Whether a write to the CPU `addr` is stored rather than discarded, which is false for ROM
    /// and for an unmapped page.
    #[inline(always)]
    #[must_use]
    pub const fn prg_writable(&self, addr: u16) -> bool {
        let addr = addr as usize;
        addr < 0x8000 && self.prg_pages[addr >> PAGE_SHIFT].is_writable()
    }

    /// Read a byte from the PPU address space, including the nametable range.
    #[inline(always)]
    pub fn chr_peek(&self, addr: u16) -> u8 {
        let page = self.chr_pages[(addr as usize >> PAGE_SHIFT) & CHR_PAGE_MASK];
        self.arena()[page.offset() | (addr as usize & PAGE_OFFSET_MASK)]
    }

    /// Write a byte to the PPU address space. Writes to CHR-ROM are discarded.
    #[inline(always)]
    pub fn chr_write(&mut self, addr: u16, val: u8) {
        let page = self.chr_pages[(addr as usize >> PAGE_SHIFT) & CHR_PAGE_MASK];
        if page.is_writable() {
            self.arena_mut()[page.offset() | (addr as usize & PAGE_OFFSET_MASK)] = val;
        }
    }

    /// Map a window of the CPU address space to `bank` of `src`: PRG-ROM at or above `$8000`,
    /// PRG-RAM below it.
    ///
    /// `bank` is signed and wraps: `-1` selects the last bank. `size` is rounded down to a whole
    /// number of pages.
    pub fn map_prg(&mut self, addr: u16, size: usize, bank: i32, src: Src) {
        let rom = src == Src::PrgRom;
        let writable = self.is_writable(src);
        let first = usize::from(addr) >> PAGE_SHIFT;
        for (i, offset) in Self::bank_pages(size, bank, src).enumerate() {
            let slot = (first + i) & (RAM_PAGES + ROM_PAGES - 1);
            if slot >= RAM_PAGES {
                assert!(rom, "only PRG-ROM maps at $8000-$FFFF");
                self.rom_pages[slot - RAM_PAGES] = offset as u32;
            } else {
                assert!(!rom, "PRG-ROM maps only at $8000-$FFFF");
                self.prg_pages[slot] = Page::new(offset, writable);
            }
        }
    }

    /// Map a window of the PPU address space to `bank` of `src`.
    ///
    /// Also used for the nametable range; see [`Memory::set_mirroring`] for the common cases.
    pub fn map_chr(&mut self, addr: u16, size: usize, bank: i32, src: Src) {
        assert!(src != Src::PrgRom, "PRG-ROM is not in the PPU's reach");
        let writable = self.is_writable(src);
        let first = usize::from(addr) >> PAGE_SHIFT;
        for (i, offset) in Self::bank_pages(size, bank, src).enumerate() {
            self.chr_pages[(first + i) & CHR_PAGE_MASK] = Page::new(offset, writable);
        }
    }

    /// The offsets of the pages of `bank` of a `size` window of `src`.
    ///
    /// Every page offset is wrapped within its region, so a window at least as large as the region
    /// repeats it instead of running off its end.
    fn bank_pages(size: usize, bank: i32, src: Src) -> impl Iterator<Item = usize> {
        let (start, len) = src.region();
        let count = (size >> PAGE_SHIFT).max(1);
        let banks = (len / size.max(1)).max(1) as i32;
        let base = bank.rem_euclid(banks) as usize * size;
        (0..count).map(move |i| start + (base + i * PAGE_SIZE) % len)
    }

    /// Unmap a window of the CPU address space below `$8000`, so it reads as zero and ignores
    /// writes.
    pub fn unmap_prg(&mut self, addr: u16, size: usize) {
        let slot = usize::from(addr) >> PAGE_SHIFT;
        for i in 0..(size >> PAGE_SHIFT).max(1) {
            self.prg_pages[slot + i] = Page::UNMAPPED;
        }
    }

    /// Unmap a window of the PPU address space, so it reads as zero and ignores writes.
    pub fn unmap_chr(&mut self, addr: u16, size: usize) {
        let slot = (addr as usize >> PAGE_SHIFT) & CHR_PAGE_MASK;
        for i in 0..(size >> PAGE_SHIFT).max(1) {
            self.chr_pages[(slot + i) & CHR_PAGE_MASK] = Page::UNMAPPED;
        }
    }

    /// Override whether a mapped CPU window below `$8000` accepts writes.
    pub fn set_prg_writable(&mut self, addr: u16, size: usize, writable: bool) {
        let slot = usize::from(addr) >> PAGE_SHIFT;
        for page in self.prg_pages.iter_mut().skip(slot).take(size >> PAGE_SHIFT) {
            *page = Page::new(page.offset(), writable);
        }
    }

    /// Point the nametable range at CIRAM according to `mirroring`.
    ///
    /// `$3000-$3EFF` mirrors `$2000-$2EFF`, so all eight pages of the range are written.
    pub fn set_mirroring(&mut self, mirroring: Mirroring) {
        // Which 1 KiB CIRAM bank each of the four nametables selects.
        let banks: [usize; 4] = match mirroring {
            Mirroring::Vertical => [0, 1, 0, 1],
            Mirroring::Horizontal => [0, 0, 1, 1],
            Mirroring::SingleScreenA => [0; 4],
            Mirroring::SingleScreenB => [1; 4],
        };
        for (i, bank) in banks.into_iter().enumerate() {
            let page = Page::new(CIRAM_START + bank * PAGE_SIZE, true);
            self.chr_pages[NAMETABLE_PAGE + i] = page;
            // $3000-$3FFF mirrors $2000-$2FFF.
            self.chr_pages[NAMETABLE_PAGE + 4 + i] = page;
        }
    }

    const fn is_writable(&self, src: Src) -> bool {
        match src {
            Src::PrgRom => false,
            Src::Chr => self.chr_writable,
            Src::PrgRam | Src::CiRam => true,
        }
    }

    /// Bytes of a RAM region (`Src::Chr` even when it holds CHR-ROM).
    ///
    /// # Panics
    ///
    /// For `Src::PrgRom`, which is not part of the state.
    pub fn region_mut(&mut self, src: Src) -> &mut [u8] {
        assert!(src != Src::PrgRom, "PRG-ROM is immutable");
        let (start, len) = src.region();
        &mut self.arena_mut()[start..start + len]
    }

    /// Bytes of a RAM region.
    ///
    /// # Panics
    ///
    /// For `Src::PrgRom`, which is not part of the state.
    pub fn region_ref(&self, src: Src) -> &[u8] {
        assert!(src != Src::PrgRom, "PRG-ROM is not part of the state");
        let (start, len) = src.region();
        &self.arena()[start..start + len]
    }

    /// The cart's PRG-RAM.
    pub fn prg_ram(&self) -> &[u8; PRG_RAM_SIZE] {
        self.arena()[PRG_RAM_START..CHR_START]
            .try_into()
            .expect("PRG_RAM_SIZE bytes")
    }

    /// The cart's PRG-RAM, for writing.
    pub fn prg_ram_mut(&mut self) -> &mut [u8; PRG_RAM_SIZE] {
        (&mut self.arena_mut()[PRG_RAM_START..CHR_START])
            .try_into()
            .expect("PRG_RAM_SIZE bytes")
    }

    /// Whether the cart's RAM is battery-backed, and so survives a power cycle.
    #[must_use]
    pub const fn battery_backed(&self) -> bool {
        self.battery_backed
    }

    /// Record whether the cart's RAM is battery-backed.
    pub const fn set_battery_backed(&mut self, battery_backed: bool) {
        self.battery_backed = battery_backed;
    }

    /// Fill the cart's RAM as if the console had just been powered on: PRG-RAM, and CHR when the
    /// cart provides CHR-RAM rather than CHR-ROM.
    pub fn fill_ram(&mut self, val: u8) {
        self.region_mut(Src::PrgRam).fill(val);
        if self.chr_writable {
            self.region_mut(Src::Chr).fill(val);
        }
    }
}

/// A fixed-size array that serializes as a slice, so a save state does not need `serde`'s
/// const-generic array support.
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct ConstArray<T, const N: usize> {
    data: [T; N],
}

impl<T, const N: usize> ConstArray<T, N> {
    /// Create a new `ConstSlice` instance.
    pub fn new() -> Self
    where
        T: Default + Copy,
    {
        Self::default()
    }

    /// Create a new `ConstSlice` instance filled with `val`.
    pub const fn filled(val: T) -> Self
    where
        T: Copy,
    {
        Self { data: [val; N] }
    }
}

impl<T, const N: usize> fmt::Debug for ConstArray<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConstArray")
            .field("len", &self.data.len())
            .finish()
    }
}

impl<T: Default + Copy, const N: usize> Default for ConstArray<T, N> {
    fn default() -> Self {
        Self {
            data: [T::default(); N],
        }
    }
}

impl<T, const N: usize> From<[T; N]> for ConstArray<T, N> {
    fn from(data: [T; N]) -> Self {
        Self { data }
    }
}

impl<T, const N: usize> Deref for ConstArray<T, N> {
    type Target = [T; N];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T, const N: usize> DerefMut for ConstArray<T, N> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<T, const N: usize> AsRef<[T]> for ConstArray<T, N> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.data.as_ref()
    }
}

impl<T, const N: usize> AsMut<[T]> for ConstArray<T, N> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.data.as_mut()
    }
}

impl<T, const N: usize> Index<usize> for ConstArray<T, N> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        self.data.index(index & (N - 1))
    }
}

impl<T, const N: usize> IndexMut<usize> for ConstArray<T, N> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.data.index_mut(index & (N - 1))
    }
}

impl<T, const N: usize> Index<Range<usize>> for ConstArray<T, N> {
    type Output = [T];

    #[inline]
    fn index(&self, range: Range<usize>) -> &Self::Output {
        self.data.index(range.start & (N - 1)..range.end.min(N))
    }
}

impl<T, const N: usize> IndexMut<Range<usize>> for ConstArray<T, N> {
    #[inline]
    fn index_mut(&mut self, range: Range<usize>) -> &mut Self::Output {
        self.data.index_mut(range.start & (N - 1)..range.end.min(N))
    }
}

impl<T, const N: usize> Index<RangeInclusive<usize>> for ConstArray<T, N> {
    type Output = [T];

    #[inline]
    fn index(&self, range: RangeInclusive<usize>) -> &Self::Output {
        self.data
            .index(range.start() & (N - 1)..=*range.end().min(&(N - 1)))
    }
}

impl<T, const N: usize> IndexMut<RangeInclusive<usize>> for ConstArray<T, N> {
    #[inline]
    fn index_mut(&mut self, range: RangeInclusive<usize>) -> &mut Self::Output {
        self.data
            .index_mut(range.start() & (N - 1)..=*range.end().min(&(N - 1)))
    }
}

impl<T: Serialize, const N: usize> Serialize for ConstArray<T, N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut s = serializer.serialize_tuple(N)?;
        for item in &self.data {
            s.serialize_element(item)?;
        }
        s.end()
    }
}

impl<'de, T, const N: usize> Deserialize<'de> for ConstArray<T, N>
where
    T: Deserialize<'de> + Default + Copy,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ArrayVisitor<T, const N: usize>(PhantomData<T>);

        impl<'de, T, const N: usize> Visitor<'de> for ArrayVisitor<T, N>
        where
            T: Deserialize<'de> + Default + Copy,
        {
            type Value = [T; N];

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str(&format!("an array of length {N}"))
            }

            #[inline]
            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut data = [T::default(); N];
                for data in &mut data {
                    match (seq.next_element())? {
                        Some(val) => *data = val,
                        None => return Err(serde::de::Error::invalid_length(N, &self)),
                    }
                }
                Ok(data)
            }
        }

        deserializer
            .deserialize_tuple(N, ArrayVisitor(PhantomData))
            .map(|data| Self { data })
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    /// A 64 KiB PRG-ROM mirrored to [`PRG_ROM_SIZE`], with each 1 KiB page filled with its own
    /// index so a read identifies which bank it came from.
    fn test_prg_rom() -> Box<PrgRom> {
        let mut rom: Box<PrgRom> = vec![0; PRG_ROM_SIZE]
            .into_boxed_slice()
            .try_into()
            .expect("PRG_ROM_SIZE bytes");
        for (i, page) in rom.chunks_mut(PAGE_SIZE).enumerate() {
            page.fill((i % 64) as u8);
        }
        rom
    }

    /// CHR-ROM, with each 1 KiB page filled with `0x80 | index`.
    fn test_memory() -> Memory {
        let mut memory = Memory::new(false);
        for (i, page) in memory
            .region_mut(Src::Chr)
            .chunks_mut(PAGE_SIZE)
            .enumerate()
        {
            page.fill(0x80 | i as u8);
        }
        memory
    }

    #[test]
    fn unmapped_reads_zero_and_ignores_writes() {
        let mut memory = test_memory();
        assert_eq!(memory.prg_peek(0x6000), 0);
        assert_eq!(memory.chr_peek(0x0000), 0);

        memory.prg_write(0x6000, 0x42);
        memory.chr_write(0x0000, 0x42);
        assert_eq!(memory.prg_peek(0x6000), 0);
        assert_eq!(memory.chr_peek(0x0000), 0);
        assert!(!memory.prg_writable(0x6000));
    }

    #[test]
    fn maps_prg_window_across_all_its_pages() {
        let rom = test_prg_rom();
        let mut memory = test_memory();
        // 16 KiB window at $8000 = PRG pages 0..16.
        memory.map_prg(0x8000, 16 * 1024, 0, Src::PrgRom);
        for i in 0..16 {
            let addr = 0x8000 + (i * PAGE_SIZE) as u16;
            assert_eq!(rom[memory.rom_index(addr)], i as u8, "page {i}");
        }
    }

    #[test]
    fn negative_bank_selects_from_the_end_and_a_mirrored_rom_wraps_like_the_original() {
        let rom = test_prg_rom();
        let mut memory = test_memory();
        // 64 KiB of PRG in 16 KiB banks = 4 banks; -1 is bank 3, starting at page 48.
        memory.map_prg(0xC000, 16 * 1024, -1, Src::PrgRom);
        assert_eq!(rom[memory.rom_index(0xC000)], 48);
        memory.map_prg(0xC000, 16 * 1024, -2, Src::PrgRom);
        assert_eq!(rom[memory.rom_index(0xC000)], 32);
        // Bank 4 wraps to bank 0 and bank 5 to bank 1.
        memory.map_prg(0x8000, 16 * 1024, 4, Src::PrgRom);
        assert_eq!(rom[memory.rom_index(0x8000)], 0);
        memory.map_prg(0x8000, 16 * 1024, 5, Src::PrgRom);
        assert_eq!(rom[memory.rom_index(0x8000)], 16);
    }

    #[test]
    fn window_sizes_from_1k_to_32k() {
        let rom = test_prg_rom();
        let mut memory = test_memory();
        for size in [1, 2, 4, 8, 16, 32] {
            let size = size * 1024;
            memory.map_prg(0x8000, size, 1, Src::PrgRom);
            // Bank 1 of a `size` window starts at byte offset `size`.
            assert_eq!(
                rom[memory.rom_index(0x8000)],
                (size >> PAGE_SHIFT) as u8,
                "{size} byte window"
            );
        }
    }

    #[test]
    #[should_panic(expected = "PRG-ROM maps only at $8000-$FFFF")]
    fn prg_rom_cannot_map_below_8000() {
        test_memory().map_prg(0x6000, 8 * 1024, 0, Src::PrgRom);
    }

    #[test]
    fn chr_rom_is_not_writable_but_chr_ram_is() {
        let mut memory = test_memory();
        memory.map_chr(0x0000, 8 * 1024, 0, Src::Chr);
        memory.chr_write(0x0000, 0x42);
        assert_eq!(memory.chr_peek(0x0000), 0x80, "CHR-ROM must ignore writes");

        let mut memory = Memory::new(true);
        memory.map_chr(0x0000, 8 * 1024, 0, Src::Chr);
        memory.chr_write(0x0000, 0x42);
        assert_eq!(memory.chr_peek(0x0000), 0x42, "CHR-RAM must accept writes");
    }

    #[test]
    fn chr_banks_and_windows_larger_than_the_region_repeat() {
        let mut memory = test_memory();
        memory.map_chr(0x1000, 4 * 1024, 1, Src::Chr);
        assert_eq!(memory.chr_peek(0x1000), 0x84, "4K bank 1");
        memory.map_chr(0x0000, 16 * 1024, 0, Src::Chr);
        let read: Vec<u8> = (0..16).map(|i| memory.chr_peek(i * 1024)).collect();
        assert_eq!(&read[..8], &read[8..], "a 16K window over 8K repeats it");
    }

    #[test]
    fn prg_ram_write_protection() {
        let mut memory = test_memory();
        memory.map_prg(0x6000, 8 * 1024, 0, Src::PrgRam);
        memory.prg_write(0x6000, 0x42);
        assert_eq!(memory.prg_peek(0x6000), 0x42);
        assert_eq!(memory.prg_ram()[0], 0x42);

        memory.set_prg_writable(0x6000, 8 * 1024, false);
        memory.prg_write(0x6000, 0x99);
        assert_eq!(memory.prg_peek(0x6000), 0x42, "writes must be discarded");
    }

    /// Write a distinct value into each nametable slot, then check which slots alias.
    fn nametable_aliases(mirroring: Mirroring) -> [u8; 4] {
        let mut memory = Memory::new(true);
        memory.set_mirroring(mirroring);
        for i in 0..4u16 {
            memory.chr_write(0x2000 + i * 0x400, i as u8 + 1);
        }
        [0, 1, 2, 3].map(|i| memory.chr_peek(0x2000 + i * 0x400))
    }

    #[test]
    fn mirroring_modes() {
        // Vertical: NT0/NT2 alias, NT1/NT3 alias. Writing 1,2,3,4 leaves 3,4,3,4.
        assert_eq!(nametable_aliases(Mirroring::Vertical), [3, 4, 3, 4]);
        // Horizontal: NT0/NT1 alias, NT2/NT3 alias.
        assert_eq!(nametable_aliases(Mirroring::Horizontal), [2, 2, 4, 4]);
        // Single screen: all four alias the same 1 KiB.
        assert_eq!(nametable_aliases(Mirroring::SingleScreenA), [4; 4]);
        assert_eq!(nametable_aliases(Mirroring::SingleScreenB), [4; 4]);
    }

    #[test]
    fn single_screen_a_and_b_use_different_banks() {
        let mut memory = Memory::default();
        memory.set_mirroring(Mirroring::SingleScreenA);
        memory.chr_write(0x2000, 0xAA);
        memory.set_mirroring(Mirroring::SingleScreenB);
        memory.chr_write(0x2000, 0xBB);

        memory.set_mirroring(Mirroring::SingleScreenA);
        assert_eq!(memory.chr_peek(0x2000), 0xAA);
        memory.set_mirroring(Mirroring::SingleScreenB);
        assert_eq!(memory.chr_peek(0x2000), 0xBB);
    }

    #[test]
    fn nametables_mirror_into_3000_range() {
        let mut memory = Memory::default();
        memory.set_mirroring(Mirroring::Horizontal);
        memory.chr_write(0x2000, 0x5A);
        assert_eq!(memory.chr_peek(0x3000), 0x5A, "$3000 mirrors $2000");
        memory.chr_write(0x2800, 0xA5);
        assert_eq!(memory.chr_peek(0x3800), 0xA5, "$3800 mirrors $2800");
    }

    /// Every page offset stays inside the arena, whatever bank a board asks for.
    #[test]
    fn every_addressable_byte_is_in_bounds() {
        let mut memory = test_memory();
        for bank in [-2, -1, 0, 1, 7, 1000] {
            for window in [1024, 8 * 1024, 16 * 1024, 32 * 1024] {
                memory.map_prg(0x8000, window, bank, Src::PrgRom);
                memory.map_prg(0x6000, window.min(8 * 1024), bank, Src::PrgRam);
                memory.map_chr(0x0000, window.min(8 * 1024), bank, Src::Chr);
                for addr in (0x4020..0x8000u32).step_by(64) {
                    let _ = memory.prg_peek(addr as u16);
                }
                for addr in (0x8000..=0xFFFFu32).step_by(64) {
                    assert!(memory.rom_index(addr as u16) < PRG_ROM_SIZE);
                }
                for addr in (0..0x4000u32).step_by(64) {
                    let _ = memory.chr_peek(addr as u16);
                }
            }
        }
    }

    /// The state file carries memory through serde; what comes back must be the same machine.
    #[test]
    fn memory_round_trips_through_postcard() {
        let mut memory = test_memory();
        memory.map_prg(0x8000, 16 * 1024, 3, Src::PrgRom);
        memory.map_prg(0x6000, 8 * 1024, 0, Src::PrgRam);
        memory.prg_write(0x6123, 0x5A);
        memory.set_battery_backed(true);

        let bytes = postcard::to_allocvec(&memory).expect("serializes");
        let restored: Memory = postcard::from_bytes(&bytes).expect("deserializes");
        assert_eq!(restored.prg_peek(0x6123), 0x5A);
        assert_eq!(restored.rom_index(0x8000), memory.rom_index(0x8000));
        assert!(restored.battery_backed());
        assert_eq!(restored.region_ref(Src::Chr), memory.region_ref(Src::Chr));
    }

    #[test]
    fn debug_does_not_print_contents() {
        let debug = format!("{:?}", Memory::default());
        assert!(debug.len() < 512, "Debug must summarize, not dump: {debug}");
    }
}
