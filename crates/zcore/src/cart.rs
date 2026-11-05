//! NES cartridge implementation.
//!
//! A [`Cart`] is what loading an iNES file yields: the header, the board, the RAM regions
//! ([`Memory`]) and the PRG-ROM image, mirrored up to [`PRG_ROM_SIZE`] and shared through an
//! `Arc` (it is not part of the emulated state).

use crate::{
    bus::State,
    mapper::{self, Mapper},
    memory::{CHR_SIZE, Memory, PRG_RAM_SIZE, PRG_ROM_SIZE, PrgRom, Src},
    ppu::Mirroring,
};
use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
    sync::Arc,
};

const PRG_ROM_BANK_SIZE: usize = 0x4000;
const CHR_ROM_BANK_SIZE: usize = 0x2000;
/// A `Result` from loading a cartridge.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors from loading a cartridge.
#[derive(Debug)]
#[must_use]
#[non_exhaustive]
pub enum Error {
    /// The iNES/NES 2.0 header did not describe a cartridge this crate can build.
    InvalidHeader {
        /// Which header byte was rejected.
        byte: u8,
        /// The value found there.
        value: u8,
        /// What was wrong with it.
        message: String,
    },
    /// The header named a mapper no board in the table serves.
    InvalidMapper(mapper::Error),
    /// The ROM could not be read.
    Io {
        /// What was being read when this happened.
        context: String,
        /// The underlying I/O error.
        source: std::io::Error,
    },
}

impl Error {
    /// Wraps an I/O error with a description of what was being read.
    pub fn io(source: std::io::Error, context: impl Into<String>) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidHeader {
                byte,
                value,
                message,
            } => write!(
                f,
                "invalid nes header (found: ${value:04X} at byte: {byte}). {message}"
            ),
            Self::InvalidMapper(err) => write!(f, "mapper: {err}"),
            Self::Io { context, source } => write!(f, "{context}: {source:?}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidMapper(err) => Some(err),
            Self::Io { source, .. } => Some(source),
            Self::InvalidHeader { .. } => None,
        }
    }
}

impl From<mapper::Error> for Error {
    fn from(err: mapper::Error) -> Self {
        Self::InvalidMapper(err)
    }
}

/// An NES cartridge.
#[derive(Debug)]
#[must_use]
pub struct Cart {
    /// The ROM's name, taken from the path it was loaded from.
    pub name: String,
    /// The iNES/NES 2.0 header it was built from.
    pub header: NesHeader,
    /// The board serving this cart.
    pub mapper: Mapper,
    /// The cart's PRG-RAM, CHR and CIRAM, and the page tables that address them.
    pub memory: Memory,
    /// The cart's PRG-ROM, mirrored up to [`PRG_ROM_SIZE`].
    pub prg_rom: Arc<PrgRom>,
    /// Bytes of CHR-ROM; 0 for a cart with CHR-RAM instead.
    pub chr_rom_size: usize,
    /// Bytes of PRG-ROM in the file.
    pub prg_rom_size: usize,
}

impl Default for Cart {
    fn default() -> Self {
        Self::empty()
    }
}

impl Cart {
    /// Creates a cart with one PRG-ROM bank and one CHR-ROM bank of zeros, and no board.
    pub fn empty() -> Self {
        Self::empty_sized(PRG_ROM_BANK_SIZE, CHR_ROM_BANK_SIZE)
    }

    /// An empty `Cart` with the given ROM sizes. A `chr_rom_size` of zero yields CHR-RAM instead.
    pub fn empty_sized(prg_rom_size: usize, chr_rom_size: usize) -> Self {
        Self {
            name: "Empty Cart".to_string(),
            header: NesHeader::default(),
            mapper: Mapper::none(),
            memory: Self::build_memory(chr_rom_size == 0),
            prg_rom: Self::prg_image(&vec![0; prg_rom_size]).expect("a valid PRG-ROM size"),
            chr_rom_size,
            prg_rom_size,
        }
    }

    /// Memory as a cart powers on: PRG-RAM (and CHR-RAM) filled with [`State::POWER_ON_RAM`].
    fn build_memory(chr_writable: bool) -> Memory {
        let mut memory = Memory::new(chr_writable);
        memory.fill_ram(State::POWER_ON_RAM);
        memory
    }

    /// The [`PRG_ROM_SIZE`] image of a PRG-ROM, mirrored (all zero for an empty one). `None`
    /// unless its size is a power of two no larger than the image, which is what keeps every bank
    /// number reading the bytes it would in the ROM itself.
    fn prg_image(prg_rom: &[u8]) -> Option<Arc<PrgRom>> {
        let image = match prg_rom.len() {
            0 => vec![0; PRG_ROM_SIZE],
            len if len.is_power_of_two() && len <= PRG_ROM_SIZE => {
                prg_rom.repeat(PRG_ROM_SIZE / len)
            }
            _ => return None,
        };
        let image: Box<PrgRom> = image.into_boxed_slice().try_into().ok()?;
        Some(Arc::from(image))
    }

    /// Load `Cart` from a ROM path.
    ///
    /// # Errors
    ///
    /// If the NES header is corrupted, the ROM file cannot be read, or the data does not match
    /// the header, then an error is returned.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        Self::from_rom(path.to_string_lossy(), &mut Self::open(path)?)
    }

    fn open(path: &Path) -> Result<BufReader<File>> {
        Ok(BufReader::new(File::open(path).map_err(|err| {
            Error::io(err, format!("failed to open rom {path:?}"))
        })?))
    }

    /// Load `Cart` from ROM data, selecting the board its mapper number calls for.
    ///
    /// Accepts both iNES and NES 2.0 headers. [`ZCore::power_on`](crate::ZCore::power_on) does
    /// this and installs the cart; reach for this to inspect a ROM without running it.
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use zcore::cart::Cart;
    ///
    /// let rom = std::fs::read("some_awesome_game.nes")?;
    /// let cart = Cart::from_rom("some_awesome_game", &mut rom.as_slice())?;
    ///
    /// println!("mapper {}", cart.mapper_num());
    /// println!("{} PRG-ROM bytes, battery: {}", cart.prg_rom().len(), cart.battery_backed());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// If the NES header is invalid, the ROM data does not match the header, or no board
    /// implements the cart's mapper, then an error is returned.
    pub fn from_rom<S, F>(name: S, rom_data: &mut F) -> Result<Self>
    where
        S: ToString,
        F: Read,
    {
        let mut cart = Self::from_rom_unmapped(name, rom_data)?;
        // Which board each mapper number selects lives with the boards, in `Mapper::from_cart`.
        cart.mapper = Mapper::from_cart(&mut cart)?;
        Ok(cart)
    }

    /// Load a cart's ROM and metadata *without* selecting a board.
    ///
    /// The result holds [`Mapper::none`] and cannot be run. Everything it holds comes from the
    /// header and the ROM itself, none of it from the board.
    ///
    /// # Errors
    ///
    /// If the NES header is invalid, or the ROM data does not match the header.
    pub fn from_rom_unmapped<S, F>(name: S, mut rom_data: &mut F) -> Result<Self>
    where
        S: ToString,
        F: Read,
    {
        let name = name.to_string();
        let header = NesHeader::load(&mut rom_data)?;

        let prg_rom_size = (header.prg_rom_banks as usize) * PRG_ROM_BANK_SIZE;
        let mut prg_rom = vec![0u8; prg_rom_size];
        rom_data.read_exact(&mut prg_rom).map_err(|err| {
            if let std::io::ErrorKind::UnexpectedEof = err.kind() {
                Error::InvalidHeader {
                    byte: 4,
                    value: header.prg_rom_banks as u8,
                    message: format!(
                        "expected `{}` prg-rom banks ({prg_rom_size} total bytes)",
                        header.prg_rom_banks
                    ),
                }
            } else {
                Error::io(err, "failed to read prg-rom")
            }
        })?;
        let prg_rom = Self::prg_image(&prg_rom).ok_or_else(|| Error::InvalidHeader {
            byte: 4,
            value: header.prg_rom_banks as u8,
            message: format!("PRG-ROM must be a power of two up to {PRG_ROM_SIZE} bytes"),
        })?;

        let prg_ram_size = Self::calculate_ram_size(header.prg_ram_shift, 10)?;
        if prg_ram_size > PRG_RAM_SIZE {
            return Err(Error::InvalidHeader {
                byte: 10,
                value: header.prg_ram_shift,
                message: format!("PRG-RAM is at most {PRG_RAM_SIZE} bytes"),
            });
        }

        let chr_rom_size = (header.chr_rom_banks as usize) * CHR_ROM_BANK_SIZE;
        let chr_ram_size = if chr_rom_size > 0 {
            0
        } else {
            Self::calculate_ram_size(header.chr_ram_shift, 11)?
        };
        if chr_rom_size.max(chr_ram_size) > CHR_SIZE {
            return Err(Error::InvalidHeader {
                byte: 5,
                value: header.chr_rom_banks as u8,
                message: format!("CHR is at most {CHR_SIZE} bytes"),
            });
        }
        if header.flags & 0x08 == 0x08 {
            return Err(Error::InvalidHeader {
                byte: 6,
                value: header.flags,
                message: "four-screen carts are not supported".to_string(),
            });
        }

        let mut memory = Self::build_memory(chr_rom_size == 0);
        if chr_rom_size > 0 {
            rom_data
                .read_exact(&mut memory.region_mut(Src::Chr)[..chr_rom_size])
                .map_err(|err| {
                    if let std::io::ErrorKind::UnexpectedEof = err.kind() {
                        Error::InvalidHeader {
                            byte: 5,
                            value: header.chr_rom_banks as u8,
                            message: format!(
                                "expected `{}` chr-rom banks ({chr_rom_size} total bytes)",
                                header.chr_rom_banks
                            ),
                        }
                    } else {
                        Error::io(err, "failed to read chr-rom")
                    }
                })?;
        }

        let mut cart = Self {
            name,
            header,
            mapper: Mapper::none(),
            memory,
            prg_rom,
            chr_rom_size,
            prg_rom_size,
        };
        // Header mirroring is the default for every board; only boards that override it - either
        // hard-wired or via a register - touch it again.
        cart.memory.set_mirroring(cart.mirroring());
        cart.memory.set_battery_backed(cart.battery_backed());
        Ok(cart)
    }

    /// The ROM's name.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // false positive on non-const deref coercion
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Whether the header is iNES rather than NES 2.0.
    #[must_use]
    pub const fn is_ines(&self) -> bool {
        matches!(
            self.header.variant,
            NesVariant::ArchaicINes | NesVariant::INes07 | NesVariant::INes
        )
    }

    /// Whether the header is NES 2.0.
    #[must_use]
    pub const fn is_nes2(&self) -> bool {
        matches!(self.header.variant, NesVariant::Nes2)
    }

    /// Returns whether this cartridge has battery-backed Save RAM.
    #[must_use]
    pub const fn battery_backed(&self) -> bool {
        self.header.flags & 0x02 == 0x02
    }

    /// Returns hardware configured `Mirroring`.
    pub const fn mirroring(&self) -> Mirroring {
        if self.header.flags & 0x01 == 0 {
            Mirroring::Horizontal
        } else {
            Mirroring::Vertical
        }
    }

    /// Returns the Mapper number for this Cart.
    #[must_use]
    pub const fn mapper_num(&self) -> u16 {
        self.header.mapper_num
    }

    /// Returns the Sub-Mapper number for this Cart.
    #[must_use]
    pub const fn submapper_num(&self) -> u8 {
        self.header.submapper_num
    }

    /// The cart's PRG-ROM, exactly as the file contained it.
    #[must_use]
    pub fn prg_rom(&self) -> &[u8] {
        &self.prg_rom[..self.prg_rom_size]
    }

    /// The cart's CHR-ROM, exactly as the file contained it. Empty for a CHR-RAM cart.
    #[must_use]
    pub fn chr_rom(&self) -> &[u8] {
        &self.memory.region_ref(Src::Chr)[..self.chr_rom_size]
    }

    /// Total RAM a NES 2.0 size byte asks for, volatile plus battery-backed.
    ///
    /// Bytes 10 and 11 are each **two** nibbles - the low one volatile RAM, the high one
    /// battery-backed NVRAM - and each nibble is a shift, where the size is `64 << shift` and 0
    /// means none. Reading the byte whole instead made a cart declaring 8 KiB of PRG-NVRAM ask for
    /// `64 << 0x70`, which overflowed and was reported as a corrupt header, so **every NES 2.0
    /// cart with a battery save was rejected outright**.
    ///
    /// The two are summed because a cart has one region per kind rather than one per volatility.
    fn calculate_ram_size(byte: u8, header_byte: u8) -> Result<usize> {
        let shift_size = |shift: u8| {
            // `0xF` is reserved rather than a size, and is rejected while parsing the header.
            64usize
                .checked_shl(shift.into())
                .ok_or_else(|| Error::InvalidHeader {
                    byte: header_byte,
                    value: byte,
                    message: format!("invalid ram size shift `{shift}` in header"),
                })
        };

        let volatile = match byte & 0x0F {
            0 => 0,
            shift => shift_size(shift)?,
        };
        let non_volatile = match byte >> 4 {
            0 => 0,
            shift => shift_size(shift)?,
        };

        Ok(volatile + non_volatile)
    }
}

impl std::fmt::Display for Cart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        write!(
            f,
            "{} - mapper {}, CHR-ROM: {}K, PRG-ROM: {}K, Mirroring: {:?}, Battery: {}",
            self.name,
            self.mapper_num(),
            self.chr_rom_size / 0x0400,
            self.prg_rom_size / 0x0400,
            self.mirroring(),
            self.battery_backed(),
        )
    }
}
/// Which cartridge header format a ROM carries.
#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
#[must_use]
pub enum NesVariant {
    /// The original iNES header, before byte 7 was assigned any meaning.
    #[default]
    ArchaicINes,
    /// iNES 0.7, whose byte 7 holds a mapper high nibble but no padding requirement.
    INes07,
    /// iNES, i.e. the format with bytes 8-15 required to be zero.
    INes,
    /// NES 2.0, which adds submappers, sized RAM fields and a region field.
    Nes2,
}

/// An `iNES` or `NES 2.0` formatted header representing hardware specs of a given NES cartridge.
///
/// <https://wiki.nesdev.org/w/index.php/INES>
/// <https://wiki.nesdev.org/w/index.php/NES_2.0>
/// <https://nesdev.org/NESDoc.pdf> (page 28)
#[derive(Default, Copy, Clone, PartialEq, Eq)]
#[must_use]
#[non_exhaustive]
pub struct NesHeader {
    /// Which of the four header formats this is.
    pub variant: NesVariant,
    /// The primary mapper number.
    pub mapper_num: u16,
    /// NES 2.0 submapper number.
    ///
    /// <https://wiki.nesdev.org/w/index.php/NES_2.0_submappers>
    pub submapper_num: u8,
    /// Mirroring, battery, trainer, VS Unisystem, PlayChoice-10 and the NES 2.0 marker.
    pub flags: u8,
    /// Number of 16K PRG-ROM banks.
    pub prg_rom_banks: u16,
    /// Number of 8K CHR-ROM banks.
    pub chr_rom_banks: u16,
    /// NES 2.0 PRG-RAM size, as a shift count.
    pub prg_ram_shift: u8,
    /// NES 2.0 CHR-RAM size, as a shift count.
    pub chr_ram_shift: u8,
    /// NES 2.0 NTSC/PAL indicator.
    pub tv_mode: u8,
    /// NES 2.0 VS System data.
    pub vs_data: u8,
}

impl NesHeader {
    /// Load `NesHeader` from a ROM path.
    ///
    /// # Errors
    ///
    /// If the NES header is corrupted, the ROM file cannot be read, or the data does not match
    /// the header, then an error is returned.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let mut rom = BufReader::new(
            File::open(path)
                .map_err(|err| Error::io(err, format!("failed to open rom {path:?}")))?,
        );
        Self::load(&mut rom)
    }

    /// Load `NesHeader` from ROM data.
    ///
    /// # Errors
    ///
    /// If the NES header is invalid, then an error is returned.
    pub fn load<F: Read>(rom_data: &mut F) -> Result<Self> {
        let mut header = [0u8; 16];
        rom_data.read_exact(&mut header).map_err(|err| {
            if let std::io::ErrorKind::UnexpectedEof = err.kind() {
                Error::InvalidHeader {
                    byte: 0,
                    value: 0,
                    message: "expected 16-byte header".to_string(),
                }
            } else {
                Error::io(err, "failed to read nes header")
            }
        })?;

        // Header checks
        if header[0..4] != *b"NES\x1a" {
            return Err(Error::InvalidHeader {
                byte: 0,
                value: header[0],
                message: "nes header signature not found".to_string(),
            });
        }
        if (header[7] & 0x0C) == 0x04 {
            return Err(Error::InvalidHeader {
                byte: 7,
                value: header[7],
                message: "header is corrupted by `DiskDude!`. repair and try again".to_string(),
            });
        }
        if (header[7] & 0x0C) == 0x0C {
            return Err(Error::InvalidHeader {
                byte: 7,
                value: header[7],
                message: "unrecognized header format. repair and try again".to_string(),
            });
        }

        let mut prg_rom_banks = u16::from(header[4]);
        let mut chr_rom_banks = u16::from(header[5]);
        // Upper 4 bits of flags 6 = D0..D3 and 7 = D4..D7
        let mut mapper_num = u16::from(((header[6] & 0xF0) >> 4) | (header[7] & 0xF0));
        // Lower 4 bits of flag 6 = D0..D3, upper 4 bits of flag 7 = D4..D7
        let flags = (header[6] & 0x0F) | ((header[7] & 0x0F) << 4);

        // NES 2.0 Format
        let mut submapper_num = 0;
        let mut prg_ram_shift = 0;
        let mut chr_ram_shift = 0;
        let mut tv_mode = 0;
        let mut vs_data = 0;
        // If D2..D3 of flag 7 == 2, then NES 2.0 (supports bytes 0-15)
        let variant = if header[7] & 0x0C == 0x08 {
            // lower 4 bits of flag 8 = D8..D11 of mapper num
            mapper_num |= u16::from(header[8] & 0x0F) << 8;
            // upper 4 bits of flag 8 = D0..D3 of submapper
            submapper_num = (header[8] & 0xF0) >> 4;
            // lower 4 bits of flag 9 = D8..D11 of prg_rom_size
            prg_rom_banks |= u16::from(header[9] & 0x0F) << 8;
            // upper 4 bits of flag 9 = D8..D11 of chr_rom_size
            chr_rom_banks |= u16::from(header[9] & 0xF0) << 4;
            prg_ram_shift = header[10];
            chr_ram_shift = header[11];
            tv_mode = header[12];
            vs_data = header[13];

            if prg_ram_shift & 0x0F == 0x0F || prg_ram_shift & 0xF0 == 0xF0 {
                return Err(Error::InvalidHeader {
                    byte: 10,
                    value: prg_ram_shift,
                    message: "invalid prg-ram size in header".to_string(),
                });
            }
            if chr_ram_shift & 0x0F == 0x0F || chr_ram_shift & 0xF0 == 0xF0 {
                return Err(Error::InvalidHeader {
                    byte: 11,
                    value: chr_ram_shift,
                    message: "invalid chr-ram size in header".to_string(),
                });
            }
            NesVariant::Nes2
        } else if header[7] & 0x0C == 0x04 {
            // If D2..D3 of flag 7 == 1, then archaic iNES (supports bytes 0-7)
            for (i, value) in header.iter().enumerate().take(16).skip(8) {
                if *value > 0 {
                    return Err(Error::InvalidHeader {
                        byte: i as u8,
                        value: *value,
                        message: format!(
                            "unrecognized data found at header byte {i}. repair and try again"
                        ),
                    });
                }
            }
            NesVariant::ArchaicINes
        } else if header[7] & 0x0C == 00 && header[12..=15].iter().all(|v| *v == 0) {
            // If D2..D3 of flag 7 == 0 and bytes 12-15 are all 0, then iNES (supports bytes 0-9)
            NesVariant::INes
        } else {
            // Else iNES 0.7 or archaic iNES (supports mapper high nibble)
            NesVariant::INes07
        };

        // Trainer
        if flags & 0x04 == 0x04 {
            return Err(Error::InvalidHeader {
                byte: 6,
                value: header[6],
                message: "trained roms are currently not supported.".to_string(),
            });
        }

        Ok(Self {
            variant,
            mapper_num,
            submapper_num,
            flags,
            prg_rom_banks,
            chr_rom_banks,
            prg_ram_shift,
            chr_ram_shift,
            tv_mode,
            vs_data,
        })
    }
}

impl std::fmt::Debug for NesHeader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        f.debug_struct("NesHeader")
            .field("version", &self.variant)
            .field("mapper_num", &format_args!("{:03}", self.mapper_num))
            .field("submapper_num", &self.submapper_num)
            .field("flags", &format_args!("0b{:08b}", &self.flags))
            .field("prg_rom_banks", &self.prg_rom_banks)
            .field("chr_rom_banks", &self.chr_rom_banks)
            .field("prg_ram_shift", &self.prg_ram_shift)
            .field("chr_ram_shift", &self.chr_ram_shift)
            .field("tv_mode", &self.tv_mode)
            .field("vs_data", &self.vs_data)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! test_headers {
        ($(($test:ident, $data:expr, $header:expr$(,)?)),*$(,)?) => {$(
            #[test]
            fn $test() {
                let header = NesHeader::load(&mut $data.as_slice()).expect("valid header");
                assert_eq!(header, $header);
            }
        )*};
    }

    #[rustfmt::skip]
    test_headers!(
        (
            mapper000_horizontal,
            [0x4E, 0x45, 0x53, 0x1A,
             0x02, 0x01, 0x01, 0x00,
             0x00, 0x00, 0x00, 0x00,
             0x00, 0x00, 0x00, 0x00],
            NesHeader {
                variant: NesVariant::INes,
                mapper_num: 0,
                flags: 0b0000_0001,
                prg_rom_banks: 2,
                chr_rom_banks: 1,
                ..NesHeader::default()
            },
        ),
        (
            mapper001_vertical,
            [0x4E, 0x45, 0x53, 0x1A,
             0x08, 0x00, 0x10, 0x00,
             0x00, 0x00, 0x00, 0x00,
             0x00, 0x00, 0x00, 0x00],
            NesHeader {
                variant: NesVariant::INes,
                mapper_num: 1,
                flags: 0b0000_0000,
                prg_rom_banks: 8,
                chr_rom_banks: 0,
                ..NesHeader::default()
            },
        ),
    );

    /// A minimal NES 2.0 ROM: `ram_byte` is header byte 10, the PRG-RAM/PRG-NVRAM pair.
    fn nes2_rom(ram_byte: u8) -> Vec<u8> {
        let mut rom = vec![
            0x4E, 0x45, 0x53, 0x1A, // NES\x1a
            0x01, // 1 x 16K PRG-ROM
            0x01, // 1 x 8K CHR-ROM
            0x02, // mapper 0, battery-backed
            0x08, // NES 2.0
            0x00, 0x00, ram_byte, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        rom.resize(rom.len() + PRG_ROM_BANK_SIZE + CHR_ROM_BANK_SIZE, 0);
        rom
    }

    /// Header byte 10 is two nibbles - PRG-RAM low, battery-backed PRG-NVRAM high - and each has
    /// to be shifted on its own. Reading the byte whole asks for `64 << 0x70` on a cart with an
    /// 8 KiB battery save, which overflows and reads as a corrupt header. zcore carts have at most
    /// 8 KiB, so a larger declaration is refused.
    #[test]
    fn nes2_prg_nvram_is_the_high_nibble_of_byte_10() {
        for byte in [0x00, 0x07, 0x70] {
            let _cart = Cart::from_rom("test.nes", &mut nes2_rom(byte).as_slice())
                .unwrap_or_else(|err| panic!("byte 10 = {byte:#04X} should load: {err}"));
        }
        for byte in [0x77, 0x0E] {
            let err = Cart::from_rom("test.nes", &mut nes2_rom(byte).as_slice())
                .expect_err("more than 8 KiB of PRG-RAM");
            assert!(
                matches!(err, Error::InvalidHeader { byte: 10, .. }),
                "byte 10 = {byte:#04X} gave {err}"
            );
        }
    }

    /// `0xF` is reserved in either nibble rather than a shift.
    #[test]
    fn nes2_rejects_the_reserved_ram_shift() {
        for byte in [0x0F, 0xF0] {
            let err = Cart::from_rom("test.nes", &mut nes2_rom(byte).as_slice())
                .expect_err("reserved shift should be rejected");
            assert!(
                matches!(err, Error::InvalidHeader { byte: 10, .. }),
                "byte 10 = {byte:#04X} gave {err}"
            );
        }
    }

    /// An iNES image: header bytes 4-7 as given, bytes 8-15 zero, then `prg` 16K banks of PRG-ROM
    /// (bank `n` filled with `n`) and `chr` 8K banks of CHR-ROM (filled with `0xC0 | n`).
    fn ines(prg: u8, chr: u8, flags6: u8, flags7: u8) -> Vec<u8> {
        let mut rom = vec![0x4E, 0x45, 0x53, 0x1A, prg, chr, flags6, flags7];
        rom.resize(16, 0);
        for bank in 0..prg {
            rom.resize(rom.len() + PRG_ROM_BANK_SIZE, bank);
        }
        for bank in 0..chr {
            rom.resize(rom.len() + CHR_ROM_BANK_SIZE, 0xC0 | bank);
        }
        rom
    }

    /// The header byte a refusal names.
    fn refused_byte(rom: &[u8]) -> u8 {
        match Cart::from_rom("test.nes", &mut &rom[..]) {
            Err(Error::InvalidHeader { byte, .. }) => byte,
            Err(err) => panic!("expected an invalid header, got {err}"),
            Ok(cart) => panic!("expected a refusal, loaded {cart}"),
        }
    }

    /// A file too short for a header, or without the `NES\x1a` signature, is refused at byte 0.
    #[test]
    fn a_missing_header_or_signature_is_refused_at_byte_0() {
        assert_eq!(refused_byte(b"NES\x1a\x01\x01"), 0, "short header");
        let mut rom = ines(1, 1, 0, 0);
        rom[3] = 0x1B;
        assert_eq!(refused_byte(&rom), 0, "bad signature");
    }

    /// Byte 7 with bits 2-3 = 1 is the `DiskDude!` signature (garbage in bytes 7-15), and 3 is
    /// no known format: both are refused rather than read as a mapper high nibble.
    #[test]
    fn corrupt_byte_7_formats_are_refused() {
        for flags7 in [0x04, 0x0C, 0x44] {
            assert_eq!(
                refused_byte(&ines(1, 1, 0, flags7)),
                7,
                "byte 7 = {flags7:#04X}"
            );
        }
    }

    /// A trainer would shift every later byte by 512, and four-screen needs VRAM no zcore cart
    /// has: both are refused at byte 6 rather than misloaded.
    #[test]
    fn trainers_and_four_screen_carts_are_refused() {
        let trainer = refused_byte(&ines(1, 1, 0x04, 0));
        assert_eq!(trainer, 6, "trainer");
        let four_screen =
            Cart::from_rom("test.nes", &mut &ines(1, 1, 0x08, 0)[..]).expect_err("four-screen");
        assert!(
            four_screen.to_string().contains("four-screen"),
            "{four_screen}"
        );
    }

    /// PRG and CHR shorter than the header promises name the size byte that lied.
    #[test]
    fn truncated_prg_and_chr_name_their_size_byte() {
        let rom = ines(2, 1, 0, 0);
        assert_eq!(
            refused_byte(&rom[..16 + PRG_ROM_BANK_SIZE]),
            4,
            "PRG cut short"
        );
        assert_eq!(refused_byte(&rom[..rom.len() - 1]), 5, "CHR cut short");
    }

    /// PRG-ROM is mirrored up to the 128K image, so it must be a power of two no larger than it;
    /// CHR-ROM is at most 8K.
    #[test]
    fn rom_sizes_the_image_cannot_hold_are_refused() {
        assert_eq!(refused_byte(&ines(3, 1, 0, 0)), 4, "three PRG banks");
        assert_eq!(refused_byte(&ines(16, 1, 0, 0)), 4, "256K of PRG");
        assert_eq!(refused_byte(&ines(1, 2, 0, 0)), 5, "16K of CHR");
    }

    /// A loaded cart reports the file's PRG/CHR bytes exactly, and mirroring and battery from
    /// byte 6; a smaller PRG-ROM is mirrored through the whole image.
    #[test]
    fn a_loaded_cart_reports_its_file_contents_and_flags() {
        let cart = Cart::from_rom("test.nes", &mut &ines(2, 1, 0x13, 0)[..]).expect("loads");
        assert_eq!(cart.mapper_num(), 1);
        assert!(cart.is_ines() && !cart.is_nes2());
        assert_eq!(cart.mirroring(), Mirroring::Vertical);
        assert!(cart.battery_backed());
        assert_eq!(cart.prg_rom().len(), 2 * PRG_ROM_BANK_SIZE);
        assert_eq!(cart.prg_rom()[PRG_ROM_BANK_SIZE], 1, "second bank");
        assert_eq!(cart.prg_rom[2 * PRG_ROM_BANK_SIZE], 0, "mirrored image");
        assert_eq!(cart.chr_rom(), &[0xC0; CHR_ROM_BANK_SIZE][..]);

        let cart = Cart::from_rom("test.nes", &mut &ines(1, 0, 0x00, 0)[..]).expect("loads");
        assert_eq!(cart.mirroring(), Mirroring::Horizontal);
        assert!(!cart.battery_backed());
        assert!(cart.chr_rom().is_empty(), "CHR-RAM cart");
    }

    /// The mapper number takes its low nibble from byte 6, its high nibble from byte 7 and, in
    /// NES 2.0, bits 8-11 from byte 8: mapper 257 must not alias to 1.
    #[test]
    fn mapper_numbers_use_every_nibble_the_format_defines() {
        let err = Cart::from_rom("test.nes", &mut &ines(1, 1, 0x40, 0)[..]).expect_err("MMC3");
        assert!(
            matches!(err, Error::InvalidMapper(mapper::Error::Unimplemented(4))),
            "{err}"
        );
        assert!(std::error::Error::source(&err).is_some());
        let err = Cart::from_rom("test.nes", &mut &ines(1, 1, 0x00, 0x10)[..]).expect_err("16");
        assert!(
            matches!(err, Error::InvalidMapper(mapper::Error::Unimplemented(16))),
            "{err}"
        );

        let mut rom = ines(1, 1, 0x10, 0x08);
        rom[8] = 0x51; // submapper 5, mapper bit 8
        let header = NesHeader::load(&mut &rom[..]).expect("NES 2.0 header");
        assert_eq!(header.variant, NesVariant::Nes2);
        assert_eq!((header.mapper_num, header.submapper_num), (0x101, 5));
        let err = Cart::from_rom("test.nes", &mut &rom[..]).expect_err("mapper 257");
        assert!(
            matches!(
                err,
                Error::InvalidMapper(mapper::Error::Unimplemented(0x101))
            ),
            "{err}"
        );
    }

    /// NES 2.0 byte 9 extends both ROM sizes, so a PRG count of 0x101 banks is not one bank.
    #[test]
    fn nes2_byte_9_extends_the_rom_sizes() {
        let mut rom = ines(1, 1, 0x10, 0x08);
        rom[9] = 0x21;
        let header = NesHeader::load(&mut &rom[..]).expect("NES 2.0 header");
        assert_eq!((header.prg_rom_banks, header.chr_rom_banks), (0x101, 0x201));
        assert_eq!(
            refused_byte(&rom),
            4,
            "the PRG the header asks for is not in the file"
        );
    }

    /// Byte 11 (CHR-RAM/NVRAM shifts) is checked like byte 10: `0xF` is reserved, and only a
    /// CHR-RAM cart's declared size counts against the 8K of CHR.
    #[test]
    fn nes2_chr_ram_size_is_validated() {
        let mut rom = ines(1, 0, 0x10, 0x08);
        for (byte11, refused_at) in [(0x0F, Some(11)), (0xF0, Some(11)), (0x08, Some(5))] {
            rom[11] = byte11;
            assert_eq!(
                Cart::from_rom("test.nes", &mut &rom[..])
                    .err()
                    .map(|err| match err {
                        Error::InvalidHeader { byte, .. } => byte,
                        err => panic!("{err}"),
                    }),
                refused_at,
                "byte 11 = {byte11:#04X}"
            );
        }
        rom[11] = 0x07; // 8K of CHR-RAM
        let cart = Cart::from_rom("test.nes", &mut &rom[..]).expect("8K of CHR-RAM loads");
        assert!(cart.is_nes2());
    }

    /// With bits 2-3 of byte 7 clear, nonzero bytes 12-15 mean iNES 0.7 (bytes 8-15 may hold
    /// junk), not iNES; the cart still loads from bytes 4-7.
    #[test]
    fn junk_in_bytes_12_to_15_reads_as_ines_0_7() {
        let mut rom = ines(1, 1, 0x10, 0);
        rom[12..16].copy_from_slice(b"Dude");
        let header = NesHeader::load(&mut &rom[..]).expect("iNES 0.7 header");
        assert_eq!(header.variant, NesVariant::INes07);
        assert_eq!(header.mapper_num, 1);
        assert!(Cart::from_rom("test.nes", &mut &rom[..]).is_ok());
    }

    /// A missing file is an I/O error carrying its cause, not a header error.
    #[test]
    fn a_missing_file_is_an_io_error() {
        let path = std::env::temp_dir().join("zcore-no-such-rom.nes");
        for err in [
            Cart::from_path(&path).expect_err("no file"),
            NesHeader::from_path(&path).expect_err("no file"),
        ] {
            assert!(matches!(err, Error::Io { .. }), "{err}");
            assert!(std::error::Error::source(&err).is_some());
            assert!(err.to_string().contains("zcore-no-such-rom.nes"), "{err}");
        }
    }
}
