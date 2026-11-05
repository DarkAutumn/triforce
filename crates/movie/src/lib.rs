//! Portable artifact formats: `TFMOVIE` v1 input movies and `TFGOLDEN` v1 per-frame digests.
//! Shared by the `tf` tools, tests that replay corpus movies and the pool's movie recording.
//!
//! A movie starts from power-on, or — when its metadata names a `base` corpus movie (normally
//! `power_on_to_start`, which ends on the first gameplay frame after registering LINK) — from the
//! state at the end of that base. Every backend reaches the base state by playing the base movie
//! once; nothing replays the title and registration screens per movie.
//!
//! Frame numbering: a movie has `frames` inputs, indexed `0..frames`. A poke tagged `frame = k` is
//! applied after input `k - 1` has run and before input `k` runs (`k = 0`: right at the start).
//! Golden record 0 is the starting state, before any poke or input; record `k` (`k >= 1`) is the
//! state after input `k - 1` has run.

use anyhow::{Context, Result, bail, ensure};
use std::io::{Read, Write};
use std::path::Path;

pub const MOVIE_MAGIC: &[u8; 8] = b"TFMOVIE\0";
pub const GOLDEN_MAGIC: &[u8; 8] = b"TFGOLDEN";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Poke {
    pub frame: u32,
    pub addr: u16,
    pub value: u8,
}

pub fn poke_addr_allowed(addr: u16) -> bool {
    addr < 0x0800 || (0x6000..0x8000).contains(&addr)
}

/// A machine a movie can be played on.
pub trait Playback {
    fn run_frame(&mut self, buttons: u8);
    fn poke(&mut self, addr: u16, value: u8);
}

/// Play `movie` on a freshly powered-on machine: its `base` chain first (loaded from `movies_dir`,
/// `<name>.tfmovie`), then its inputs and pokes, including pokes tagged after the last frame.
pub fn play(p: &mut dyn Playback, movie: &Movie, movies_dir: &Path) -> Result<()> {
    play_base(p, movie, movies_dir)?;
    let mut pokes = movie.pokes.iter().peekable();
    for (k, &x) in movie.inputs.iter().enumerate() {
        while let Some(q) = pokes.next_if(|q| q.frame as usize <= k) {
            p.poke(q.addr, q.value);
        }
        p.run_frame(x);
    }
    for q in pokes {
        p.poke(q.addr, q.value);
    }
    Ok(())
}

/// Bring a freshly powered-on machine to `movie`'s starting state: play its base (recursively), or
/// nothing for a power-on movie.
pub fn play_base(p: &mut dyn Playback, movie: &Movie, movies_dir: &Path) -> Result<()> {
    if let Some(name) = movie.base() {
        let base = Movie::load(&movies_dir.join(format!("{name}.tfmovie")))?;
        ensure!(base.rom_sha1 == movie.rom_sha1, "base movie {name} is for a different ROM");
        play(p, &base, movies_dir)?;
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub struct Movie {
    pub rom_sha1: [u8; 20],
    /// One NES controller byte per frame (A=1, B=2, Select=4, Start=8, Up, Down, Left, Right).
    pub inputs: Vec<u8>,
    /// Pokes in emission order, frames non-decreasing.
    pub pokes: Vec<Poke>,
    pub meta: serde_json::Value,
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(self.pos + n <= self.buf.len(), "truncated file at byte {}", self.pos);
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
}

impl Movie {
    pub fn new(rom_sha1: [u8; 20], meta: serde_json::Value) -> Self {
        Movie { rom_sha1, inputs: Vec::new(), pokes: Vec::new(), meta }
    }

    pub fn name(&self) -> &str {
        self.meta.get("name").and_then(|v| v.as_str()).unwrap_or("")
    }

    /// The corpus movie this one starts after (`meta.base`), if any.
    pub fn base(&self) -> Option<&str> {
        self.meta.get("base").and_then(|v| v.as_str())
    }

    pub fn tags(&self) -> Vec<String> {
        self.meta
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|t| t.as_str().map(str::to_owned)).collect())
            .unwrap_or_default()
    }

    pub fn push_poke(&mut self, addr: u16, value: u8) -> Result<()> {
        ensure!(poke_addr_allowed(addr), "poke address ${addr:04X} is outside $0000-$07FF/$6000-$7FFF");
        self.pokes.push(Poke { frame: self.inputs.len() as u32, addr, value });
        Ok(())
    }

    /// Pokes grouped per frame: `pokes_at(k)` are applied before input `k` runs.
    pub fn poke_index(&self) -> Vec<(u32, std::ops::Range<usize>)> {
        let mut out: Vec<(u32, std::ops::Range<usize>)> = Vec::new();
        for (i, p) in self.pokes.iter().enumerate() {
            match out.last_mut() {
                Some((f, r)) if *f == p.frame => r.end = i + 1,
                _ => out.push((p.frame, i..i + 1)),
            }
        }
        out
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(64 + self.inputs.len() / 4 + self.pokes.len() * 7);
        b.extend_from_slice(MOVIE_MAGIC);
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&0u16.to_le_bytes());
        b.extend_from_slice(&self.rom_sha1);
        b.extend_from_slice(&(self.inputs.len() as u32).to_le_bytes());
        let mut runs: Vec<(u8, u32)> = Vec::new();
        for &x in &self.inputs {
            match runs.last_mut() {
                Some((v, n)) if *v == x => *n += 1,
                _ => runs.push((x, 1)),
            }
        }
        b.extend_from_slice(&(runs.len() as u32).to_le_bytes());
        for (v, n) in runs {
            b.push(v);
            b.extend_from_slice(&n.to_le_bytes());
        }
        b.extend_from_slice(&(self.pokes.len() as u32).to_le_bytes());
        for p in &self.pokes {
            b.extend_from_slice(&p.frame.to_le_bytes());
            b.extend_from_slice(&p.addr.to_le_bytes());
            b.push(p.value);
        }
        let meta = serde_json::to_vec(&self.meta).expect("movie metadata serializes");
        b.extend_from_slice(&(meta.len() as u32).to_le_bytes());
        b.extend_from_slice(&meta);
        b
    }

    pub fn decode(buf: &[u8]) -> Result<Movie> {
        let mut r = Reader { buf, pos: 0 };
        ensure!(r.take(8)? == MOVIE_MAGIC, "not a TFMOVIE file");
        let version = r.u16()?;
        ensure!(version == 1, "unsupported TFMOVIE version {version}");
        r.u16()?;
        let rom_sha1: [u8; 20] = r.take(20)?.try_into().unwrap();
        let frames = r.u32()? as usize;
        // Each count is checked against the bytes (or frames) it describes before anything is
        // allocated for it, so a corrupt count is an error rather than a huge reservation.
        let run_count = r.u32()? as usize;
        let (runs, _) = r.take(run_count * 5)?.as_chunks::<5>();
        let run_len = |&[_, n @ ..]: &[u8; 5]| u32::from_le_bytes(n) as usize;
        let mut covered = 0usize;
        for run in runs {
            covered += run_len(run);
            ensure!(covered <= frames, "button runs exceed the frame count");
        }
        ensure!(covered == frames, "button runs cover {covered} of {frames} frames");
        let mut inputs = Vec::with_capacity(frames);
        for run in runs {
            inputs.resize(inputs.len() + run_len(run), run[0]);
        }
        let poke_count = r.u32()? as usize;
        let (poke_bytes, _) = r.take(poke_count * 7)?.as_chunks::<7>();
        let mut pokes = Vec::with_capacity(poke_count);
        let mut last = 0u32;
        for &[f0, f1, f2, f3, a0, a1, value] in poke_bytes {
            let p = Poke { frame: u32::from_le_bytes([f0, f1, f2, f3]), addr: u16::from_le_bytes([a0, a1]), value };
            ensure!(poke_addr_allowed(p.addr), "poke address ${:04X} not allowed", p.addr);
            ensure!(p.frame >= last && p.frame as usize <= frames, "poke frame {} out of order/range", p.frame);
            last = p.frame;
            pokes.push(p);
        }
        let meta_len = r.u32()? as usize;
        let meta = serde_json::from_slice(r.take(meta_len)?).context("movie metadata is not JSON")?;
        ensure!(r.pos == buf.len(), "trailing bytes after movie metadata");
        Ok(Movie { rom_sha1, inputs, pokes, meta })
    }

    pub fn load(path: &Path) -> Result<Movie> {
        let buf = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        Movie::decode(&buf).with_context(|| format!("decoding {}", path.display()))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        write_atomic(path, &self.encode())
    }
}

pub fn write_atomic(path: &Path, data: &[u8]) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("tmp~");
    let mut f = std::fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
    f.write_all(data)?;
    f.sync_all()?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Digest domains, in record order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Domain {
    Ram = 0,
    PrgRam = 1,
    Chr = 2,
    Ciram = 3,
    Palette = 4,
    Oam = 5,
    Pixels = 6,
    Cpu = 7,
    Ppu = 8,
    Mapper = 9,
    Apu = 10,
}

pub const ALL_DOMAINS: [Domain; 11] = [
    Domain::Ram,
    Domain::PrgRam,
    Domain::Chr,
    Domain::Ciram,
    Domain::Palette,
    Domain::Oam,
    Domain::Pixels,
    Domain::Cpu,
    Domain::Ppu,
    Domain::Mapper,
    Domain::Apu,
];

/// Every domain (ladder self-diffs).
pub const LADDER_MASK: u16 = 0x7FF;

impl Domain {
    pub fn bit(self) -> u16 {
        1 << self as u8
    }
    pub fn name(self) -> &'static str {
        match self {
            Domain::Ram => "ram",
            Domain::PrgRam => "prg_ram",
            Domain::Chr => "chr",
            Domain::Ciram => "ciram",
            Domain::Palette => "palette",
            Domain::Oam => "oam",
            Domain::Pixels => "pixels",
            Domain::Cpu => "cpu",
            Domain::Ppu => "ppu",
            Domain::Mapper => "mapper",
            Domain::Apu => "apu",
        }
    }
    pub fn in_mask(mask: u16) -> impl Iterator<Item = Domain> {
        ALL_DOMAINS.into_iter().filter(move |d| mask & d.bit() != 0)
    }
}

pub fn digest8(data: &[u8]) -> u64 {
    let h = blake3::hash(data);
    u64::from_le_bytes(h.as_bytes()[..8].try_into().unwrap())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Golden {
    pub domain_mask: u16,
    pub movie_hash: [u8; 32],
    /// `records[k][i]` is the digest of the i-th enabled domain after `k` frames.
    pub records: Vec<Vec<u64>>,
}

impl Golden {
    pub fn domains(&self) -> Vec<Domain> {
        Domain::in_mask(self.domain_mask).collect()
    }

    pub fn encode(&self) -> Vec<u8> {
        let per = self.domains().len();
        let mut b = Vec::with_capacity(48 + self.records.len() * per * 8);
        b.extend_from_slice(GOLDEN_MAGIC);
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&self.domain_mask.to_le_bytes());
        b.extend_from_slice(&self.movie_hash);
        b.extend_from_slice(&(self.records.len() as u32).to_le_bytes());
        for r in &self.records {
            assert_eq!(r.len(), per);
            for d in r {
                b.extend_from_slice(&d.to_le_bytes());
            }
        }
        b
    }

    pub fn decode(buf: &[u8]) -> Result<Golden> {
        let mut r = Reader { buf, pos: 0 };
        ensure!(r.take(8)? == GOLDEN_MAGIC, "not a TFGOLDEN file");
        let version = r.u16()?;
        ensure!(version == 1, "unsupported TFGOLDEN version {version}");
        let domain_mask = r.u16()?;
        ensure!(domain_mask & !LADDER_MASK == 0, "unknown domain bits {domain_mask:#x}");
        ensure!(domain_mask != 0, "golden has no digest domains");
        let movie_hash: [u8; 32] = r.take(32)?.try_into().unwrap();
        let count = r.u32()? as usize;
        let per = Domain::in_mask(domain_mask).count();
        // Read the records' bytes before allocating for them, so a corrupt count is refused.
        let data = r.take(count * per * 8)?;
        let records = data
            .chunks_exact(per * 8)
            .map(|rec| rec.as_chunks::<8>().0.iter().map(|&d| u64::from_le_bytes(d)).collect())
            .collect();
        ensure!(r.pos == buf.len(), "trailing bytes after golden records");
        Ok(Golden { domain_mask, movie_hash, records })
    }

    pub fn load(path: &Path) -> Result<Golden> {
        let buf = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        Golden::decode(&buf).with_context(|| format!("decoding {}", path.display()))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        write_atomic(path, &self.encode())
    }
}

pub fn movie_file_hash(path: &Path) -> Result<[u8; 32]> {
    let mut f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    Ok(*blake3::hash(&buf).as_bytes())
}

pub fn parse_hex_u16(s: &str) -> Result<u16> {
    let t = s.trim();
    let t = t.strip_prefix("0x").or_else(|| t.strip_prefix('$')).unwrap_or(t);
    u16::from_str_radix(t, 16).with_context(|| format!("bad hex number {s:?}"))
}

pub fn parse_num_u8(s: &str) -> Result<u8> {
    let t = s.trim();
    if let Some(h) = t.strip_prefix("0x").or_else(|| t.strip_prefix('$')) {
        return u8::from_str_radix(h, 16).with_context(|| format!("bad value {s:?}"));
    }
    match t.parse::<u8>() {
        Ok(v) => Ok(v),
        Err(_) => bail!("bad value {s:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn movie_roundtrip_and_poke_order() {
        let mut m = Movie::new([7; 20], serde_json::json!({"name": "t", "tags": ["short"]}));
        m.inputs.extend([0, 0, 8, 8, 8, 0x41]);
        m.push_poke(0x10, 3).unwrap();
        m.inputs.push(0);
        m.push_poke(0x6001, 9).unwrap();
        m.push_poke(0x6002, 9).unwrap();
        assert!(m.push_poke(0x2000, 1).is_err());
        let enc = m.encode();
        let back = Movie::decode(&enc).unwrap();
        assert_eq!(back, m);
        assert_eq!(back.poke_index(), vec![(6, 0..1), (7, 1..3)]);
        assert_eq!(back.tags(), vec!["short".to_string()]);
    }

    #[test]
    fn movie_rejects_garbage() {
        let m = Movie::new([0; 20], serde_json::json!({}));
        let mut enc = m.encode();
        enc.push(0);
        assert!(Movie::decode(&enc).is_err());
        assert!(Movie::decode(b"TFMOVIE").is_err());
    }

    #[test]
    fn golden_roundtrip() {
        let g = Golden {
            domain_mask: Domain::Ram.bit() | Domain::Pixels.bit(),
            movie_hash: [1; 32],
            records: vec![vec![1, 0], vec![2, 3]],
        };
        assert_eq!(Golden::decode(&g.encode()).unwrap(), g);
        assert_eq!(g.domains(), vec![Domain::Ram, Domain::Pixels]);
    }
}
