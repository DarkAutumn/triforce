//! Corpus layout, manifest, fan-out over processes (one oracle per process) and the coverage gate.
//!
//! - `corpus/movies/<name>.tfmovie` — committed movies.
//! - `corpus/golden/<name>.tfgolden` — committed goldens for movies tagged `short`.
//! - `target/corpus-golden/<name>.tfgolden` — generated goldens for long movies.
//! - `corpus/manifest.toml` — `[[movie]]` entries rebuilt by `tf corpus sync`.
//! - `corpus/pixel-allowlist.toml` — `[[allow]] movie, frames = [first, last], rows = [first, last], reason`.

use crate::backend::Backend;
use crate::formats::{Golden, LADDER_MASK, Movie, ORACLE_MASK, movie_file_hash};
use crate::rom::{RomInfo, workspace_root};
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

pub const SHORT_TAG: &str = "short";
pub const D2_TAG: &str = "power-on-to-d2";
pub const MIN_FRAMES: u64 = 1_000_000;

pub fn corpus_dir() -> PathBuf {
    workspace_root().join("corpus")
}

pub fn movie_path(name: &str) -> PathBuf {
    corpus_dir().join("movies").join(format!("{name}.tfmovie"))
}

fn name_of(path: &Path) -> String {
    path.file_stem().unwrap().to_string_lossy().into_owned()
}

pub fn golden_path(name: &str, tags: &[String]) -> PathBuf {
    if tags.iter().any(|t| t == SHORT_TAG) {
        corpus_dir().join("golden").join(format!("{name}.tfgolden"))
    } else {
        workspace_root().join("target/corpus-golden").join(format!("{name}.tfgolden"))
    }
}

pub fn golden_path_for(movie: &Path) -> Result<PathBuf> {
    let name = name_of(movie);
    for p in [
        corpus_dir().join("golden").join(format!("{name}.tfgolden")),
        workspace_root().join("target/corpus-golden").join(format!("{name}.tfgolden")),
    ] {
        if p.exists() {
            return Ok(p);
        }
    }
    bail!("no golden for {name}: pass --golden or run `tf golden --all`")
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

/// Run `tf <args>` jobs in parallel child processes, printing each child's output as it ends.
pub fn fan_out(jobs: Vec<Vec<OsString>>, parallel: usize) -> Result<Vec<i32>> {
    let exe = std::env::current_exe()?;
    let mut codes = vec![0; jobs.len()];
    let mut running: Vec<(usize, std::process::Child)> = Vec::new();
    let mut next = 0;
    while next < jobs.len() || !running.is_empty() {
        while next < jobs.len() && running.len() < parallel.max(1) {
            let child = Command::new(&exe)
                .args(&jobs[next])
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()?;
            running.push((next, child));
            next += 1;
        }
        let mut i = 0;
        let mut progressed = false;
        while i < running.len() {
            if running[i].1.try_wait()?.is_some() {
                let (idx, child) = running.swap_remove(i);
                let out = child.wait_with_output()?;
                print!("{}", String::from_utf8_lossy(&out.stdout));
                eprint!("{}", String::from_utf8_lossy(&out.stderr));
                codes[idx] = out.status.code().unwrap_or(2);
                progressed = true;
            } else {
                i += 1;
            }
        }
        if !progressed {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }
    Ok(codes)
}

fn os(args: &[&dyn AsRef<std::ffi::OsStr>]) -> Vec<OsString> {
    args.iter().map(|a| a.as_ref().to_os_string()).collect()
}

fn golden_current(golden: &Path, movie: &Path) -> bool {
    match (Golden::load(golden), movie_file_hash(movie)) {
        (Ok(g), Ok(h)) => g.movie_hash == h,
        _ => false,
    }
}

pub fn golden_all(rom: &RomInfo, parallel: usize) -> Result<i32> {
    let m = Manifest::load()?;
    ensure!(!m.movie.is_empty(), "corpus manifest is empty: run `tf corpus sync`");
    let mut jobs = Vec::new();
    for e in &m.movie {
        let movie = corpus_dir().join(&e.file);
        let out = golden_path(&e.name, &e.tags);
        if golden_current(&out, &movie) {
            continue;
        }
        jobs.push(os(&[&"--rom", &rom.path, &"golden", &movie, &"-o", &out]));
    }
    println!("generating {} goldens ({} up to date)", jobs.len(), m.movie.len() - jobs.len());
    let codes = fan_out(jobs, parallel)?;
    Ok(if codes.iter().all(|&c| c == 0) { 0 } else { 1 })
}

pub fn diff_all(rom: &RomInfo, backend: &str, parallel: usize) -> Result<i32> {
    let m = Manifest::load()?;
    let mut jobs = Vec::new();
    for e in &m.movie {
        let movie = corpus_dir().join(&e.file);
        let golden = golden_path(&e.name, &e.tags);
        ensure!(golden.exists(), "missing golden {}: run `tf golden --all`", golden.display());
        jobs.push(os(&[&"--rom", &rom.path, &"diff-one", &movie, &backend, &golden, &ORACLE_MASK.to_string()]));
    }
    let codes = fan_out(jobs, parallel)?;
    let bad = codes.iter().filter(|&&c| c != 0).count();
    println!("{} of {} movies diverge", bad, codes.len());
    Ok(if bad == 0 { 0 } else { 1 })
}

pub fn ladder_golden(rom: &RomInfo, movies: &[PathBuf], out: &Path, backend: &str, parallel: usize) -> Result<i32> {
    std::fs::create_dir_all(out)?;
    let movies: Vec<PathBuf> = if movies.is_empty() {
        Manifest::load()?.movie.iter().map(|e| corpus_dir().join(&e.file)).collect()
    } else {
        movies.to_vec()
    };
    let jobs = movies
        .iter()
        .map(|m| {
            let g = out.join(format!("{}.tfgolden", name_of(m)));
            os(&[&"--rom", &rom.path, &"golden-one", &m, &g, &backend, &LADDER_MASK.to_string()])
        })
        .collect();
    let codes = fan_out(jobs, parallel)?;
    Ok(if codes.iter().all(|&c| c == 0) { 0 } else { 1 })
}

pub fn ladder_diff(rom: &RomInfo, dir: &Path, backend: &str, parallel: usize) -> Result<i32> {
    let mut jobs = Vec::new();
    let mut goldens: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "tfgolden"))
        .collect();
    goldens.sort();
    ensure!(!goldens.is_empty(), "no ladder goldens in {}", dir.display());
    for g in goldens {
        let movie = movie_path(&name_of(&g));
        jobs.push(os(&[&"--rom", &rom.path, &"diff-one", &movie, &backend, &g, &LADDER_MASK.to_string()]));
    }
    let codes = fan_out(jobs, parallel)?;
    let bad = codes.iter().filter(|&&c| c != 0).count();
    println!("ladder: {} of {} movies diverge", bad, codes.len());
    Ok(if bad == 0 { 0 } else { 1 })
}

#[derive(Deserialize, Default)]
pub struct PixelAllowlist {
    #[serde(default)]
    allow: Vec<Allow>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Allow {
    movie: String,
    frames: [usize; 2],
    #[allow(dead_code)]
    rows: [usize; 2],
    #[allow(dead_code)]
    reason: String,
}

impl PixelAllowlist {
    pub fn load() -> Result<PixelAllowlist> {
        let p = corpus_dir().join("pixel-allowlist.toml");
        if !p.exists() {
            return Ok(PixelAllowlist::default());
        }
        toml::from_str(&std::fs::read_to_string(&p)?).with_context(|| format!("parsing {}", p.display()))
    }

    pub fn covers(&self, movie: &str, frame: usize) -> bool {
        self.allow.iter().any(|a| a.movie == movie && (a.frames[0]..=a.frames[1]).contains(&frame))
    }
}

/// Coverage gate: ≥ 1,000,000 frames, at least one `power-on-to-d2` movie, and every such movie ends
/// in Dungeon 2's boss room (level $10 == 2, room $EB == 0x0E) in the oracle.
pub fn check(rom: &RomInfo) -> Result<i32> {
    let m = Manifest::load()?;
    let mut ok = true;
    let total: u64 = m.movie.iter().map(|e| e.frames).sum();
    for e in &m.movie {
        let mv = Movie::load(&corpus_dir().join(&e.file))?;
        if mv.rom_sha1 != rom.payload_sha1 || mv.inputs.len() as u64 != e.frames {
            println!("FAIL {}: manifest out of date or wrong ROM", e.name);
            ok = false;
        }
    }
    println!("corpus: {} movies, {total} frames (need {MIN_FRAMES})", m.movie.len());
    if total < MIN_FRAMES {
        ok = false;
    }
    let d2: Vec<&Entry> = m.movie.iter().filter(|e| e.tags.iter().any(|t| t == D2_TAG)).collect();
    if d2.is_empty() {
        println!("FAIL: no {D2_TAG} movie");
        ok = false;
    }
    for e in d2 {
        let jobs = vec![os(&[&"--rom", &rom.path, &"end-room", &corpus_dir().join(&e.file)])];
        let codes = fan_out(jobs, 1)?;
        if codes[0] != 0 {
            ok = false;
        }
    }
    println!("{}", if ok { "corpus check passed" } else { "corpus check FAILED" });
    Ok(if ok { 0 } else { 1 })
}

/// Replay a movie in the oracle and require level 2, room 0x0E at the end.
pub fn end_room(b: &mut dyn Backend, movie_path: &Path) -> Result<i32> {
    let mv = Movie::load(movie_path)?;
    crate::script::play_prefix(b, &mv);
    let ram = b.ram();
    let (level, room) = (ram[0x10], ram[0xEB]);
    let ok = level == 2 && room == 0x0E;
    println!(
        "{} {}: ends at level {level} room {room:#04x}",
        if ok { "ok  " } else { "FAIL" },
        movie_path.display()
    );
    Ok(if ok { 0 } else { 1 })
}

