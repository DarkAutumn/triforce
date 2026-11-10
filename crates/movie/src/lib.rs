//! Portable artifact formats: `TFMOVIE` v1 input movies and `TFGOLDEN` v1 per-frame digests.
//! Shared by the `tf` tools, tests that replay corpus movies and the pool's movie recording.
//!
//! A movie starts from power-on; or — when its metadata names a `base` corpus movie (normally
//! `power_on_to_start`, which ends on the first gameplay frame after registering LINK) — from the
//! state at the end of that base; or — when its metadata names an `origin` — from the `TFZSTATE`
//! file `<movies_dir>/../states/<origin>.tfzstate` (recorded chain runs, `tf movie import`). Every
//! backend reaches the base state by playing the base movie once; nothing replays the title and
//! registration screens per movie. A movie names at most one of `base` and `origin`.
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
    /// Resume a `TFZSTATE` file (an `origin` movie's start). Players that can't are refused.
    fn load_state(&mut self, _state: &[u8]) -> Result<()> {
        bail!("this player cannot start a movie from a saved state")
    }
}

/// The start state file of an `origin` movie.
pub fn origin_path(movies_dir: &Path, origin: &str) -> std::path::PathBuf {
    movies_dir.parent().unwrap_or(movies_dir).join("states").join(format!("{origin}.tfzstate"))
}

/// Play `movie` on a freshly powered-on machine: its `base` chain first (loaded from `movies_dir`,
/// `<name>.tfmovie`), then its inputs and pokes, including pokes tagged after the last frame.
pub fn play(p: &mut dyn Playback, movie: &Movie, movies_dir: &Path) -> Result<()> {
    play_base(p, movie, movies_dir)?;
    play_inputs(p, movie);
    Ok(())
}

/// Bring a freshly powered-on machine to `movie`'s starting state: play its base (recursively),
/// load its origin state, or nothing for a power-on movie. A base chain that loops is an error.
pub fn play_base(p: &mut dyn Playback, movie: &Movie, movies_dir: &Path) -> Result<()> {
    if let Some(origin) = movie.origin() {
        ensure!(movie.base().is_none(), "movie {} names both a base and an origin", movie.name());
        let path = origin_path(movies_dir, origin);
        let state = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        return p.load_state(&state).with_context(|| format!("loading {}", path.display()));
    }
    let mut chain = Vec::new();
    let mut next = movie.base().map(str::to_owned);
    while let Some(name) = next {
        ensure!(chain.iter().all(|(n, _): &(String, Movie)| *n != name), "base chain loops at {name}");
        let base = Movie::load(&movies_dir.join(format!("{name}.tfmovie")))?;
        ensure!(base.rom_sha1 == movie.rom_sha1, "base movie {name} is for a different ROM");
        ensure!(base.origin().is_none(), "base movie {name} starts from a saved state");
        next = base.base().map(str::to_owned);
        chain.push((name, base));
    }
    for (_, base) in chain.iter().rev() {
        play_inputs(p, base);
    }
    Ok(())
}

/// Play `movie`'s own pokes and inputs (no base).
fn play_inputs(p: &mut dyn Playback, movie: &Movie) {
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
}

/// The longest movie `Movie::decode` accepts: 10M frames (46 hours of play; the longest corpus
/// movie is about 41k frames).
pub const MAX_FRAMES: usize = 10_000_000;

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

    /// The start state this one plays from (`meta.origin`, `origin_path`), if any.
    pub fn origin(&self) -> Option<&str> {
        self.meta.get("origin").and_then(|v| v.as_str())
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
        ensure!(frames <= MAX_FRAMES, "frame count {frames} exceeds the {MAX_FRAMES}-frame limit");
        // The frame count is capped, and the run and poke counts are checked against the bytes
        // they describe before anything is allocated for them, so a corrupt count is an error
        // rather than a huge reservation.
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

/// The BLAKE3 of a movie file followed by each file of its `base` chain (base, its base, ...,
/// loaded from `movies_dir`): a golden's staleness key, so re-recording a base makes the goldens
/// of every movie built on it stale. Equals `movie_file_hash` for a power-on movie.
pub fn movie_chain_hash(path: &Path, movies_dir: &Path) -> Result<[u8; 32]> {
    let mut h = blake3::Hasher::new();
    let mut seen: Vec<String> = Vec::new();
    let mut next = path.to_path_buf();
    loop {
        let buf = std::fs::read(&next).with_context(|| format!("reading {}", next.display()))?;
        h.update(&buf);
        let m = Movie::decode(&buf).with_context(|| format!("decoding {}", next.display()))?;
        if let Some(origin) = m.origin() {
            let state = origin_path(movies_dir, origin);
            h.update(&std::fs::read(&state).with_context(|| format!("reading {}", state.display()))?);
        }
        let Some(name) = m.base() else { break };
        ensure!(!seen.iter().any(|n| n == name), "base chain loops at {name}");
        seen.push(name.to_owned());
        next = movies_dir.join(format!("{name}.tfmovie"));
    }
    Ok(*h.finalize().as_bytes())
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

    /// A movie with inputs `inputs` and one poke per `(frame, addr)`.
    fn movie_with(inputs: &[u8], pokes: &[(u32, u16)]) -> Movie {
        let mut m = Movie::new([3; 20], serde_json::json!({"name": "m"}));
        m.inputs = inputs.to_vec();
        m.pokes = pokes.iter().map(|&(frame, addr)| Poke { frame, addr, value: 0x5A }).collect();
        m
    }

    /// Offset of the run count in an encoded movie (magic, version, flags, ROM SHA-1, frames).
    const RUNS_AT: usize = 8 + 2 + 2 + 20 + 4;

    fn decode_err(buf: &[u8]) -> String {
        format!("{:#}", Movie::decode(buf).expect_err("refused"))
    }

    #[test]
    fn run_length_round_trips_at_its_boundaries() {
        let alternating: Vec<u8> = (0..1000).map(|k| (k % 2) as u8).collect();
        let cases: [(&[u8], u32); 5] = [
            (&[], 0),
            (&[0x80], 1),
            (&[0; 100_000], 1),
            (&alternating, 1000),
            (&[1, 1, 2, 2, 2, 1], 3),
        ];
        for (inputs, runs) in cases {
            let enc = movie_with(inputs, &[]).encode();
            let count = u32::from_le_bytes(enc[RUNS_AT..RUNS_AT + 4].try_into().unwrap());
            assert_eq!(count, runs, "{} frames", inputs.len());
            assert_eq!(Movie::decode(&enc).unwrap().inputs, inputs);
        }
        // Pokes at frame 0 and at frame == frames (after the last input) are both in range.
        let m = movie_with(&[0, 0], &[(0, 0x00), (2, 0x7FFF)]);
        assert_eq!(Movie::decode(&m.encode()).unwrap(), m);
    }

    #[test]
    fn movie_decode_names_what_is_wrong() {
        let good = movie_with(&[1, 1, 2], &[(1, 0x10)]).encode();
        let mut bad = good.clone();
        bad[0] = b'X';
        assert!(decode_err(&bad).contains("not a TFMOVIE"));
        let mut bad = good.clone();
        bad[8] = 2;
        assert!(decode_err(&bad).contains("version 2"));
        for len in 0..good.len() {
            assert!(decode_err(&good[..len]).contains("truncated"), "cut at {len}");
        }
        let mut bad = good.clone();
        bad.push(b' ');
        assert!(decode_err(&bad).contains("trailing"));

        // Runs: two runs, (1, 2) then (2, 1); the first run's count is at RUNS_AT + 5.
        let mut bad = good.clone();
        bad[RUNS_AT + 5] = 3;
        assert!(decode_err(&bad).contains("exceed the frame count"));
        let mut bad = good.clone();
        bad[RUNS_AT + 5] = 1;
        assert!(decode_err(&bad).contains("cover 2 of 3"));

        // The poke follows the runs and their count: frame (4), addr (2), value (1).
        let poke_at = RUNS_AT + 4 + 2 * 5 + 4;
        for (addr, ok) in [(0x07FFu16, true), (0x0800, false), (0x5FFF, false), (0x6000, true), (0x8000, false)] {
            let mut buf = good.clone();
            buf[poke_at + 4..poke_at + 6].copy_from_slice(&addr.to_le_bytes());
            match Movie::decode(&buf) {
                Ok(m) => assert!(ok && m.pokes[0].addr == addr, "${addr:04X}"),
                Err(e) => assert!(!ok && format!("{e}").contains("not allowed"), "${addr:04X}: {e}"),
            }
        }
        let mut bad = good.clone();
        bad[poke_at..poke_at + 4].copy_from_slice(&4u32.to_le_bytes());
        assert!(decode_err(&bad).contains("out of order/range"), "poke past the end");
        let unordered = movie_with(&[0, 0], &[(2, 0x10), (1, 0x10)]).encode();
        assert!(decode_err(&unordered).contains("out of order/range"));

        let mut bad = good.clone();
        let meta_at = bad.len() - serde_json::to_vec(&serde_json::json!({"name": "m"})).unwrap().len();
        bad[meta_at] = b'[';
        assert!(decode_err(&bad).contains("not JSON"));
    }

    /// Counts are read before the data they describe: a corrupt count must be refused as a
    /// truncated file rather than reserve memory for it.
    #[test]
    fn corrupt_counts_are_refused_without_allocating_for_them() {
        let mut huge = movie_with(&[], &[]).encode();
        huge[RUNS_AT - 4..RUNS_AT].copy_from_slice(&u32::MAX.to_le_bytes()); // frames, no runs
        assert!(decode_err(&huge).contains("exceeds the 10000000-frame limit"));
        // One run claiming u32::MAX frames, matching the header: refused before allocating.
        let mut bomb = movie_with(&[], &[]).encode();
        bomb[RUNS_AT - 4..RUNS_AT].copy_from_slice(&u32::MAX.to_le_bytes());
        bomb[RUNS_AT..RUNS_AT + 4].copy_from_slice(&1u32.to_le_bytes());
        bomb.splice(RUNS_AT + 4..RUNS_AT + 4, [0, 0xFF, 0xFF, 0xFF, 0xFF]);
        assert!(decode_err(&bomb).contains("frame limit"));
        let mut at_limit = movie_with(&[], &[]).encode();
        at_limit[RUNS_AT - 4..RUNS_AT].copy_from_slice(&(MAX_FRAMES as u32).to_le_bytes());
        assert!(decode_err(&at_limit).contains("cover 0 of 10000000"));
        let mut huge = movie_with(&[], &[]).encode();
        huge[RUNS_AT..RUNS_AT + 4].copy_from_slice(&u32::MAX.to_le_bytes()); // runs
        assert!(decode_err(&huge).contains("truncated"));
        let mut huge = movie_with(&[], &[]).encode();
        huge[RUNS_AT + 4..RUNS_AT + 8].copy_from_slice(&u32::MAX.to_le_bytes()); // pokes
        assert!(decode_err(&huge).contains("truncated"));

        let g = Golden { domain_mask: LADDER_MASK, movie_hash: [0; 32], records: vec![] };
        let mut huge = g.encode();
        let n = huge.len();
        huge[n - 4..].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(format!("{:#}", Golden::decode(&huge).unwrap_err()).contains("truncated"));
    }

    #[test]
    fn golden_decode_names_what_is_wrong() {
        let g = Golden { domain_mask: Domain::Cpu.bit() | Domain::Apu.bit(), movie_hash: [9; 32], records: vec![vec![1, 2]; 3] };
        let good = g.encode();
        let err = |buf: &[u8]| format!("{:#}", Golden::decode(buf).unwrap_err());
        let mut bad = good.clone();
        bad[..8].copy_from_slice(MOVIE_MAGIC);
        assert!(err(&bad).contains("not a TFGOLDEN"));
        let mut bad = good.clone();
        bad[8] = 9;
        assert!(err(&bad).contains("version 9"));
        let mut bad = good.clone();
        bad[11] = 0x08; // bit 11: no such domain
        assert!(err(&bad).contains("unknown domain bits"));
        let mut bad = good.clone();
        bad[10..12].fill(0); // no domains: records of nothing, any count would "fit"
        assert!(err(&bad).contains("no digest domains"));
        for len in 0..good.len() {
            assert!(err(&good[..len]).contains("truncated"), "cut at {len}");
        }
        let mut bad = good;
        bad.extend_from_slice(&[0; 8]);
        assert!(err(&bad).contains("trailing"));
        assert_eq!(Domain::in_mask(LADDER_MASK).collect::<Vec<_>>(), ALL_DOMAINS);
    }

    /// Records every call so the order of pokes and frames can be checked.
    #[derive(Default)]
    struct Recorder(Vec<String>);

    impl Playback for Recorder {
        fn run_frame(&mut self, buttons: u8) {
            self.0.push(format!("frame {buttons}"));
        }
        fn poke(&mut self, addr: u16, value: u8) {
            self.0.push(format!("poke {addr:04X}={value}"));
        }
        fn load_state(&mut self, state: &[u8]) -> Result<()> {
            self.0.push(format!("state {}", String::from_utf8_lossy(state)));
            Ok(())
        }
    }

    /// A player without `load_state`.
    struct PowerOnOnly;

    impl Playback for PowerOnOnly {
        fn run_frame(&mut self, _: u8) {}
        fn poke(&mut self, _: u16, _: u8) {}
    }

    /// An `origin` movie loads `../states/<origin>.tfzstate` before its frames; its chain hash
    /// covers the state file; a player that can't load states, a missing state and a movie naming
    /// both a base and an origin are refused.
    #[test]
    fn origin_movies_start_from_their_state() {
        let root = std::env::temp_dir().join(format!("movie-origin-{}", std::process::id()));
        let (movies, states) = (root.join("movies"), root.join("states"));
        std::fs::create_dir_all(&states).unwrap();
        std::fs::write(states.join("s0.tfzstate"), b"S0").unwrap();
        let mut m = movie_with(&[5], &[(0, 0x10)]);
        m.meta = serde_json::json!({"name": "run", "origin": "s0"});
        m.save(&movies.join("run.tfmovie")).unwrap();

        let mut r = Recorder::default();
        play(&mut r, &m, &movies).unwrap();
        assert_eq!(r.0, ["state S0", "poke 0010=90", "frame 5"]);
        let err = play(&mut PowerOnOnly, &m, &movies).unwrap_err();
        assert!(format!("{err:#}").contains("cannot start a movie from a saved state"), "{err:#}");

        let path = movies.join("run.tfmovie");
        let before = movie_chain_hash(&path, &movies).unwrap();
        std::fs::write(states.join("s0.tfzstate"), b"S1").unwrap();
        assert_ne!(movie_chain_hash(&path, &movies).unwrap(), before);

        m.meta["origin"] = serde_json::json!("missing");
        assert!(play(&mut Recorder::default(), &m, &movies).is_err());
        m.meta = serde_json::json!({"name": "run", "origin": "s0", "base": "x"});
        assert!(format!("{:#}", play(&mut Recorder::default(), &m, &movies).unwrap_err()).contains("both"));
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// `play` runs the base chain first, applies a frame-`k` poke before input `k`, and applies
    /// pokes tagged after the last input at the end; a base for another ROM is refused.
    #[test]
    fn play_orders_base_pokes_and_frames() {
        let dir = std::env::temp_dir().join(format!("movie-play-{}", std::process::id()));
        let mut root = movie_with(&[1], &[(0, 0x01)]);
        root.meta = serde_json::json!({"name": "root"});
        root.save(&dir.join("root.tfmovie")).unwrap();
        let mut mid = movie_with(&[2], &[]);
        mid.meta = serde_json::json!({"name": "mid", "base": "root"});
        mid.save(&dir.join("mid.tfmovie")).unwrap();
        let mut top = movie_with(&[3, 4], &[(1, 0x02), (1, 0x03), (2, 0x04)]);
        top.meta = serde_json::json!({"name": "top", "base": "mid"});

        let mut r = Recorder::default();
        play(&mut r, &top, &dir).unwrap();
        let expected = [
            "poke 0001=90", "frame 1", "frame 2", "frame 3", "poke 0002=90", "poke 0003=90", "frame 4",
            "poke 0004=90",
        ];
        assert_eq!(r.0, expected);

        let mut r = Recorder::default();
        play_base(&mut r, &top, &dir).unwrap();
        assert_eq!(r.0, expected[..3]);

        top.rom_sha1 = [4; 20];
        let err = play(&mut Recorder::default(), &top, &dir).unwrap_err();
        assert!(format!("{err:#}").contains("different ROM"), "{err:#}");
        top.meta["base"] = serde_json::json!("missing");
        assert!(play(&mut Recorder::default(), &top, &dir).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A base chain that loops (a movie based on itself, or a→b→a) is an error, not a stack overflow.
    #[test]
    fn play_refuses_a_looping_base_chain() {
        let dir = std::env::temp_dir().join(format!("movie-loop-{}", std::process::id()));
        let save = |name: &str, base: &str| {
            let mut m = movie_with(&[1], &[]);
            m.meta = serde_json::json!({"name": name, "base": base});
            m.save(&dir.join(format!("{name}.tfmovie"))).unwrap();
            m
        };
        let selfish = save("s", "s");
        let a = save("a", "b");
        save("b", "a");
        for (m, at) in [(&selfish, "s"), (&a, "b")] {
            let err = play(&mut Recorder::default(), m, &dir).unwrap_err();
            assert!(format!("{err:#}").contains(&format!("base chain loops at {at}")), "{err:#}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A based movie's chain hash changes when its base's bytes change and not otherwise; a
    /// power-on movie's equals its file hash.
    #[test]
    fn chain_hash_covers_the_base_chain() {
        let dir = std::env::temp_dir().join(format!("movie-chain-{}", std::process::id()));
        let save = |name: &str, base: Option<&str>, inputs: &[u8]| {
            let mut m = movie_with(inputs, &[]);
            m.meta = serde_json::json!({"name": name});
            if let Some(b) = base {
                m.meta["base"] = serde_json::json!(b);
            }
            m.save(&dir.join(format!("{name}.tfmovie"))).unwrap();
        };
        save("root", None, &[1]);
        save("mid", Some("root"), &[2]);
        let (root, mid) = (dir.join("root.tfmovie"), dir.join("mid.tfmovie"));
        assert_eq!(movie_chain_hash(&root, &dir).unwrap(), movie_file_hash(&root).unwrap());
        let before = movie_chain_hash(&mid, &dir).unwrap();
        assert_ne!(before, movie_file_hash(&mid).unwrap());
        assert_eq!(movie_chain_hash(&mid, &dir).unwrap(), before, "unchanged files, same hash");
        save("root", None, &[1, 1]);
        assert_ne!(movie_chain_hash(&mid, &dir).unwrap(), before, "re-recorded base");
        save("root", None, &[1]);
        assert_eq!(movie_chain_hash(&mid, &dir).unwrap(), before);
        save("root", Some("mid"), &[1]);
        assert!(format!("{:#}", movie_chain_hash(&mid, &dir).unwrap_err()).contains("loops"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn number_parsers_accept_the_script_forms() {
        for s in ["0x6000", "$6000", "6000", " 6000 "] {
            assert_eq!(parse_hex_u16(s).unwrap(), 0x6000, "{s:?}");
        }
        for s in ["0x10000", "", "0xG", "-1"] {
            assert!(parse_hex_u16(s).is_err(), "{s:?}");
        }
        for (s, v) in [("255", 255), ("0xff", 255), ("$0A", 10), ("10", 10)] {
            assert_eq!(parse_num_u8(s).unwrap(), v, "{s:?}");
        }
        for s in ["256", "0x100", "ten", "-1", ""] {
            assert!(parse_num_u8(s).is_err(), "{s:?}");
        }
    }
}
