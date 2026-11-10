//! Movie replay on zcore and the per-frame digests behind `ladder-golden` and `ladder-diff`.

use crate::formats::{Domain, Golden, Movie, digest8};
use crate::zcore::Zcore;
use anyhow::Result;

/// Digests of the enabled domains. Record 0 has no pixel digest (0).
pub fn digests(b: &Zcore, mask: u16, record: usize, buf: &mut Vec<u8>) -> Result<Vec<u64>> {
    let mut out = Vec::new();
    for d in Domain::in_mask(mask) {
        if record == 0 && d == Domain::Pixels {
            out.push(0);
            continue;
        }
        anyhow::ensure!(b.read_domain(d, buf), "zcore can't read domain {} at record {record}", d.name());
        out.push(digest8(buf));
    }
    Ok(out)
}

struct Play<'a>(&'a mut Zcore);

impl movie::Playback for Play<'_> {
    fn run_frame(&mut self, buttons: u8) {
        self.0.run_frame(buttons);
    }
    fn poke(&mut self, addr: u16, value: u8) {
        self.0.poke(addr, value);
    }
    fn load_state(&mut self, state: &[u8]) -> Result<()> {
        self.0.load_state(state)
    }
}

/// Bring a freshly powered-on `b` to `movie`'s starting state: play its base movie (recursively),
/// load its origin state, or nothing for a power-on movie.
pub fn play_base(b: &mut Zcore, movie: &Movie) -> Result<()> {
    movie::play_base(&mut Play(b), movie, &crate::corpus::corpus_dir().join("movies"))
}

/// Play `movie` (base included) on a freshly powered-on `b`, including pokes tagged after its last
/// frame.
pub fn play_movie(b: &mut Zcore, movie: &Movie) -> Result<()> {
    play_movie_in(b, movie, &crate::corpus::corpus_dir().join("movies"))
}

/// [`play_movie`] with bases and origin states looked up next to `movies_dir` instead of the corpus.
pub fn play_movie_in(b: &mut Zcore, movie: &Movie, movies_dir: &std::path::Path) -> Result<()> {
    movie::play(&mut Play(b), movie, movies_dir)
}

/// Replay `movie` on a freshly powered-on `b`, calling `each(k, &digests)` for records `0..=frames`
/// (record 0 is the movie's starting state: power-on, or the end of its base). Stops early when
/// `each` returns false.
pub fn replay(b: &mut Zcore, movie: &Movie, mask: u16, mut each: impl FnMut(usize, &[u64]) -> bool) -> Result<()> {
    play_base(b, movie)?;
    let mut buf = Vec::with_capacity(256 * 240);
    let d0 = digests(b, mask, 0, &mut buf)?;
    if !each(0, &d0) {
        return Ok(());
    }
    let index = movie.poke_index();
    let mut next = 0usize;
    for k in 0..movie.inputs.len() {
        apply_pokes(b, movie, &index, &mut next, k);
        b.run_frame(movie.inputs[k]);
        let d = digests(b, mask, k + 1, &mut buf)?;
        if !each(k + 1, &d) {
            return Ok(());
        }
    }
    Ok(())
}

pub fn apply_pokes(
    b: &mut Zcore,
    movie: &Movie,
    index: &[(u32, std::ops::Range<usize>)],
    next: &mut usize,
    frame: usize,
) {
    while *next < index.len() && index[*next].0 as usize <= frame {
        let (f, r) = &index[*next];
        if *f as usize == frame {
            for p in &movie.pokes[r.clone()] {
                b.poke(p.addr, p.value);
            }
        }
        *next += 1;
    }
}

pub fn golden_of(b: &mut Zcore, movie: &Movie, key: [u8; 32], mask: u16) -> Result<Golden> {
    let mut records = Vec::with_capacity(movie.inputs.len() + 1);
    replay(b, movie, mask, |_, d| {
        records.push(d.to_vec());
        true
    })?;
    Ok(Golden { domain_mask: mask, movie_hash: key, records })
}

/// First record where `b` diverges from `golden`, with the differing domains.
pub struct Divergence {
    pub record: usize,
    pub domains: Vec<Domain>,
}

pub fn first_divergence(b: &mut Zcore, movie: &Movie, golden: &Golden) -> Result<Option<Divergence>> {
    let domains = golden.domains();
    anyhow::ensure!(
        golden.records.len() == movie.inputs.len() + 1,
        "golden has {} records for a {}-frame movie",
        golden.records.len(),
        movie.inputs.len()
    );
    let mut found = None;
    replay(b, movie, golden.domain_mask, |k, d| {
        let bad: Vec<Domain> =
            domains.iter().zip(d).zip(&golden.records[k]).filter(|((_, a), b)| a != b).map(|((dom, _), _)| *dom).collect();
        if bad.is_empty() {
            true
        } else {
            found = Some(Divergence { record: k, domains: bad });
            false
        }
    })?;
    Ok(found)
}
