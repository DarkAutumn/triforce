//! Corpus layout, manifest, the zcore self-diff (`ladder-golden`/`ladder-diff`) and the coverage gate.
//!
//! - `corpus/movies/<name>.tfmovie` — committed movies: scripted and random ones from power-on or
//!   `power_on_to_start`, and chain runs of the trained team (`chain-*`, `tf movie import`).
//! - `corpus/states/<origin>.tfzstate` — the start states of the chain-run movies.
//! - `corpus/manifest.toml` — `[[movie]]` entries rebuilt by `tf corpus sync`.
//! - `corpus/golden/<name>.tfgolden` — committed zcore goldens (all 11 domains) for the movies tagged
//!   `short`: the reference a fresh clone diffs against (`tf ladder-diff`).
//!
//! `$TF_CORPUS` points every command at another corpus directory (tests).

use crate::formats::{Golden, LADDER_MASK, Movie, movie_chain_hash};
use crate::replay::{first_divergence, golden_of};
use crate::rom::{RomInfo, workspace_root};
use crate::zcore::Zcore;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

pub const SHORT_TAG: &str = "short";
pub const MIN_FRAMES: u64 = 1_000_000;
/// The coverage gate's minimum of gameplay frames (game mode $12 = 5) replayed in each of the
/// overworld and the route's dungeons 1-3.
pub const MIN_PLAY_FRAMES: u64 = 10_000;
pub const PLAY_LEVELS: [u8; 4] = [0, 1, 2, 3];

pub fn corpus_dir() -> PathBuf {
    std::env::var_os("TF_CORPUS").map(PathBuf::from).unwrap_or_else(|| workspace_root().join("corpus"))
}

/// Committed goldens of the `short` movies.
pub fn golden_dir() -> PathBuf {
    corpus_dir().join("golden")
}

pub fn movie_path(name: &str) -> PathBuf {
    corpus_dir().join("movies").join(format!("{name}.tfmovie"))
}

fn name_of(path: &Path) -> String {
    path.file_stem().unwrap().to_string_lossy().into_owned()
}

#[derive(Serialize, Deserialize, Default)]
pub struct Manifest {
    #[serde(default)]
    pub movie: Vec<Entry>,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub name: String,
    pub file: String,
    pub tags: Vec<String>,
    pub frames: u64,
}

impl Manifest {
    pub fn load() -> Result<Manifest> {
        let p = corpus_dir().join("manifest.toml");
        if !p.exists() {
            return Ok(Manifest::default());
        }
        let text = std::fs::read_to_string(&p)?;
        toml::from_str(&text).with_context(|| format!("parsing {}", p.display()))
    }
}

pub fn sync() -> Result<i32> {
    let dir = corpus_dir().join("movies");
    let mut entries = Vec::new();
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "tfmovie"))
        .collect();
    paths.sort();
    for p in paths {
        let m = Movie::load(&p)?;
        let name = name_of(&p);
        ensure!(m.name() == name, "{} metadata name {:?} != file name", p.display(), m.name());
        entries.push(Entry { name, file: format!("movies/{}", p.file_name().unwrap().to_string_lossy()), tags: m.tags(), frames: m.inputs.len() as u64 });
    }
    let total: u64 = entries.iter().map(|e| e.frames).sum();
    let text = toml::to_string_pretty(&Manifest { movie: entries })?;
    crate::formats::write_atomic(&corpus_dir().join("manifest.toml"), text.as_bytes())?;
    println!("manifest: {total} frames");
    Ok(0)
}

/// Run `f` over `items` on `jobs` threads; results in item order.
fn par_map<T: Sync, R: Send>(items: &[T], jobs: usize, f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let next = AtomicUsize::new(0);
    let mut out: Vec<Option<R>> = items.iter().map(|_| None).collect();
    std::thread::scope(|s| {
        let workers: Vec<_> = (0..jobs.clamp(1, items.len().max(1)))
            .map(|_| {
                s.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(item) = items.get(i) else { break done };
                        done.push((i, f(item)));
                    }
                })
            })
            .collect();
        for w in workers {
            for (i, r) in w.join().expect("worker panicked") {
                out[i] = Some(r);
            }
        }
    });
    out.into_iter().map(|r| r.expect("every item ran")).collect()
}

fn load_movie(rom: &RomInfo, path: &Path) -> Result<Movie> {
    let movie = Movie::load(path)?;
    ensure!(movie.rom_sha1 == rom.payload_sha1, "{} was recorded for a different ROM", path.display());
    Ok(movie)
}

/// Every manifest movie, or only those tagged `short`.
fn manifest_movies(short: bool) -> Result<Vec<PathBuf>> {
    let m = Manifest::load()?;
    ensure!(!m.movie.is_empty(), "corpus manifest is empty: run `tf corpus sync`");
    Ok(m.movie
        .iter()
        .filter(|e| !short || e.tags.iter().any(|t| t == SHORT_TAG))
        .map(|e| corpus_dir().join(&e.file))
        .collect())
}

/// Write an all-domain zcore golden for each movie (default: the whole manifest) into `out`.
/// Explicit movies must be corpus movies (`corpus/movies/<name>.tfmovie`): `ladder-diff` finds a
/// golden's movie there by name.
pub fn ladder_golden(rom: &RomInfo, movies: &[PathBuf], short: bool, out: &Path, jobs: usize) -> Result<i32> {
    for path in movies {
        let canon = |p: &Path| std::fs::canonicalize(p).ok();
        ensure!(
            canon(path).is_some() && canon(path) == canon(&movie_path(&name_of(path))),
            "{} is not a corpus movie: ladder goldens are for movies in {} (copy it there first)",
            path.display(),
            corpus_dir().join("movies").display()
        );
    }
    std::fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    let mut movies = if movies.is_empty() { manifest_movies(short)? } else { movies.to_vec() };
    movies.sort_by_key(|p| name_of(p));
    movies.dedup_by_key(|p| name_of(p));
    let results = par_map(&movies, jobs, |path| -> Result<()> {
        let movie = load_movie(rom, path)?;
        let mut b = Zcore::open(&rom.bytes)?;
        let g = golden_of(&mut b, &movie, movie_chain_hash(path, &corpus_dir().join("movies"))?, LADDER_MASK)?;
        g.save(&out.join(format!("{}.tfgolden", name_of(path))))
    });
    let mut bad = 0;
    for (path, r) in movies.iter().zip(results) {
        if let Err(e) = r {
            println!("{}: {e:#}", path.display());
            bad += 1;
        }
    }
    println!("ladder: wrote {} of {} goldens to {}", movies.len() - bad, movies.len(), out.display());
    Ok(if bad == 0 { 0 } else { 1 })
}

/// Replay the movie of every golden in `dir` on zcore and report the first diverging record.
pub fn ladder_diff(rom: &RomInfo, dir: &Path, jobs: usize) -> Result<i32> {
    let mut goldens: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "tfgolden"))
        .collect();
    goldens.sort();
    ensure!(!goldens.is_empty(), "no ladder goldens in {}", dir.display());
    let results = par_map(&goldens, jobs, |g| -> Result<String> {
        let path = movie_path(&name_of(g));
        let movie = load_movie(rom, &path)?;
        let golden = Golden::load(g)?;
        ensure!(
            golden.movie_hash == movie_chain_hash(&path, &corpus_dir().join("movies"))?,
            "{} is stale (recorded for another version of {} or of its base movies): regenerate it with `tf ladder-golden`",
            g.display(),
            path.display()
        );
        let mut b = Zcore::open(&rom.bytes)?;
        Ok(match first_divergence(&mut b, &movie, &golden)? {
            None => String::new(),
            Some(d) => {
                let names: Vec<&str> = d.domains.iter().map(|x| x.name()).collect();
                format!("first divergence at frame {}: {}", d.record, names.join(","))
            }
        })
    });
    let mut bad = 0;
    for (g, r) in goldens.iter().zip(results) {
        match r {
            Ok(msg) if msg.is_empty() => {}
            Ok(msg) => {
                println!("{}: {msg}", name_of(g));
                bad += 1;
            }
            Err(e) => {
                println!("{}: {e:#}", name_of(g));
                bad += 1;
            }
        }
    }
    println!("ladder: {} of {} movies diverge", bad, goldens.len());
    Ok(if bad == 0 { 0 } else { 1 })
}

/// Counts gameplay frames (game mode $12 = 5) by level ($10) while a movie plays.
struct PlayCounter<'a> {
    b: &'a mut Zcore,
    play: [u64; 256],
}

impl movie::Playback for PlayCounter<'_> {
    fn run_frame(&mut self, buttons: u8) {
        self.b.run_frame(buttons);
        let ram = self.b.ram();
        if ram[0x12] == 5 {
            self.play[usize::from(ram[0x10])] += 1;
        }
    }
    fn poke(&mut self, addr: u16, value: u8) {
        self.b.poke(addr, value);
    }
    fn load_state(&mut self, state: &[u8]) -> Result<()> {
        self.b.load_state(state)
    }
}

/// Gameplay frames by level of `movie`'s own frames (its base is played but not counted).
fn play_frames(rom: &RomInfo, movie: &Movie) -> Result<[u64; 256]> {
    let mut b = Zcore::open(&rom.bytes)?;
    b.set_render(false);
    crate::replay::play_base(&mut b, movie)?;
    let mut own = movie.clone();
    own.meta = serde_json::json!({"name": movie.name()});
    let mut c = PlayCounter { b: &mut b, play: [0; 256] };
    movie::play(&mut c, &own, &corpus_dir().join("movies"))?;
    Ok(c.play)
}

/// Coverage gate: every manifest entry matches its movie and the ROM, ≥ 1,000,000 frames, and —
/// measured by replaying every movie, not taken from tags — at least 10,000 gameplay frames in
/// each of the overworld and Dungeons 1, 2 and 3.
pub fn check(rom: &RomInfo, jobs: usize) -> Result<i32> {
    let m = Manifest::load()?;
    let mut ok = true;
    let total: u64 = m.movie.iter().map(|e| e.frames).sum();
    let results = par_map(&m.movie, jobs, |e| -> Result<[u64; 256]> {
        let mv = Movie::load(&corpus_dir().join(&e.file))?;
        ensure!(mv.rom_sha1 == rom.payload_sha1 && mv.inputs.len() as u64 == e.frames, "manifest out of date or wrong ROM");
        play_frames(rom, &mv)
    });
    let mut play = [0u64; 256];
    for (e, r) in m.movie.iter().zip(results) {
        match r {
            Ok(p) => play.iter_mut().zip(p).for_each(|(a, b)| *a += b),
            Err(err) => {
                println!("FAIL {}: {err:#}", e.name);
                ok = false;
            }
        }
    }
    println!("corpus: {} movies, {total} frames (need {MIN_FRAMES})", m.movie.len());
    if total < MIN_FRAMES {
        ok = false;
    }
    for level in PLAY_LEVELS {
        let n = play[usize::from(level)];
        let pass = n >= MIN_PLAY_FRAMES;
        println!("level {level}: {n} gameplay frames (need {MIN_PLAY_FRAMES}){}", if pass { "" } else { " FAIL" });
        ok &= pass;
    }
    println!("{}", if ok { "corpus check passed" } else { "corpus check FAILED" });
    Ok(if ok { 0 } else { 1 })
}
