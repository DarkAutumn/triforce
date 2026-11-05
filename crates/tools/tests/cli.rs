//! End-to-end runs of the `tf` binary on the real ROM and the committed corpus. Every output goes
//! to a temporary directory; the corpus is only read.

use movie::{Golden, Movie};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Whether `tf` can find the ROM; without it every test here is skipped, like zcore's.
fn have_rom() -> bool {
    let found = std::env::var_os("TRIFORCE_ROM").is_some() || repo().join("roms/zelda.nes").exists();
    if !found {
        eprintln!("roms/zelda.nes missing: skipping");
    }
    found
}

fn tf(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tf")).args(args).current_dir(repo()).output().expect("tf runs")
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// A fresh directory under the system temp dir, removed when dropped.
struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> TempDir {
        let dir = std::env::temp_dir().join(format!("tf-cli-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        TempDir(dir)
    }

    fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }

    fn arg(&self, name: &str) -> String {
        self.join(name).to_string_lossy().into_owned()
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// `movie script` records the frames and pokes of a script after its base, checking its asserts
/// as it goes, and `movie info` reads the result back.
#[test]
fn movie_script_records_what_info_reports() {
    if !have_rom() {
        return;
    }
    let tmp = TempDir::new("script");
    std::fs::write(
        tmp.join("walk.txt"),
        "base: power_on_to_start\ntags: e2e, tiny\nassert ram[0x12] == 0x05  # in play\nRIGHT 8\npoke 0x10 0x00\n- 2\n",
    )
    .unwrap();
    let out = tf(&["movie", "script", &tmp.arg("walk.txt"), "-o", &tmp.arg("walk.tfmovie")]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(stdout(&out).contains("10 frames, 1 pokes"), "{}", stdout(&out));

    let m = Movie::load(&tmp.join("walk.tfmovie")).unwrap();
    assert_eq!(m.inputs, [[0x80; 8].as_slice(), &[0; 2]].concat());
    assert_eq!((m.pokes[0].frame, m.pokes[0].addr), (8, 0x10));
    assert_eq!(m.base(), Some("power_on_to_start"));
    assert_eq!(m.tags(), ["e2e", "tiny"]);

    let info = tf(&["movie", "info", &tmp.arg("walk.tfmovie")]);
    assert!(info.status.success(), "{}", stderr(&info));
    let text = stdout(&info);
    for line in ["name:   walk", "frames: 10", "pokes:  1", "rom:    a12d74c73a0481599a5d832361d168f4737bbcf6"] {
        assert!(text.contains(line), "{line:?} missing from\n{text}");
    }
}

/// A failed assert fails the command with the address, expected value and frame, and writes no
/// movie.
#[test]
fn a_failing_script_assert_writes_no_movie() {
    if !have_rom() {
        return;
    }
    let tmp = TempDir::new("assert");
    std::fs::write(tmp.join("bad.txt"), "base: power_on_to_start\n- 3\nassert ram[0x12] == 0x06\n").unwrap();
    let out = tf(&["movie", "script", &tmp.arg("bad.txt"), "-o", &tmp.arg("bad.tfmovie")]);
    assert_eq!(out.status.code(), Some(2));
    assert!(stderr(&out).contains("assert ram[$0012] == 0x06 failed after frame 3"), "{}", stderr(&out));
    assert!(!tmp.join("bad.tfmovie").exists());

    let info = tf(&["movie", "info", &tmp.arg("bad.txt")]);
    assert_eq!(info.status.code(), Some(2));
    assert!(stderr(&info).contains("not a TFMOVIE"), "{}", stderr(&info));
}

/// `movie random` is a pure function of its seed, and `movie rebase` strips the base's frames
/// from a power-on movie, shifting its pokes.
#[test]
fn random_is_seeded_and_rebase_strips_the_base() {
    if !have_rom() {
        return;
    }
    let tmp = TempDir::new("random");
    for (name, seed) in [("a", "7"), ("b", "7"), ("c", "8")] {
        let out = tf(&["movie", "random", "--seed", seed, "--frames", "40", "-o", &tmp.arg(&format!("{name}.tfmovie"))]);
        assert!(out.status.success(), "{}", stderr(&out));
    }
    let load = |name: &str| Movie::load(&tmp.join(&format!("{name}.tfmovie"))).unwrap();
    let (a, b, c) = (load("a"), load("b"), load("c"));
    assert_eq!(a.inputs.len(), 40);
    assert_eq!(a.base(), Some("power_on_to_start"));
    assert_eq!(a.inputs, b.inputs, "same seed, same movie");
    assert_ne!(a.inputs, c.inputs, "another seed, another movie");

    // A power-on movie: power_on_to_start's frames, then three of its own and a poke.
    let base = Movie::load(&repo().join("corpus/movies/power_on_to_start.tfmovie")).unwrap();
    let mut full = Movie::new(base.rom_sha1, serde_json::json!({"name": "full"}));
    full.inputs = base.inputs.clone();
    full.inputs.extend([0x01, 0x02]);
    full.push_poke(0x6000, 9).unwrap();
    full.inputs.push(0x04);
    full.save(&tmp.join("full.tfmovie")).unwrap();
    let out = tf(&["movie", "rebase", &tmp.arg("full.tfmovie"), "-o", &tmp.arg("rebased.tfmovie")]);
    assert!(out.status.success(), "{}", stderr(&out));
    let r = load("rebased");
    assert_eq!(r.inputs, [0x01, 0x02, 0x04]);
    assert_eq!((r.pokes[0].frame, r.pokes[0].addr), (2, 0x6000));
    assert_eq!(r.base(), Some("power_on_to_start"));

    // Rebasing again, or a movie that doesn't start with the base, is refused.
    let out = tf(&["movie", "rebase", &tmp.arg("rebased.tfmovie"), "-o", &tmp.arg("x.tfmovie")]);
    assert!(stderr(&out).contains("already has base"), "{}", stderr(&out));
    full.inputs[0] ^= 1;
    full.save(&tmp.join("full.tfmovie")).unwrap();
    let out = tf(&["movie", "rebase", &tmp.arg("full.tfmovie"), "-o", &tmp.arg("x.tfmovie")]);
    assert!(stderr(&out).contains("does not start with power_on_to_start"), "{}", stderr(&out));
}

/// Fresh goldens of the short movies replay without divergence; a golden whose digest was changed
/// reports the record and domain, one recorded for another version of its movie is stale, and
/// either makes `ladder-diff` exit 1.
#[test]
fn ladder_diff_passes_fresh_goldens_and_reports_tampered_ones() {
    if !have_rom() {
        return;
    }
    let tmp = TempDir::new("ladder");
    let dir = tmp.arg("golden");
    let out = tf(&["ladder-golden", "--short", "-o", &dir]);
    assert!(out.status.success(), "{}{}", stdout(&out), stderr(&out));
    let shorts = std::fs::read_dir(tmp.join("golden")).unwrap().count();
    assert!(stdout(&out).contains(&format!("wrote {shorts} of {shorts} goldens")), "{}", stdout(&out));

    let out = tf(&["ladder-diff", &dir]);
    assert_eq!(out.status.code(), Some(0), "{}", stdout(&out));
    assert!(stdout(&out).contains(&format!("0 of {shorts} movies diverge")), "{}", stdout(&out));

    let path = tmp.join("golden/pause_spam.tfgolden");
    let mut g = Golden::load(&path).unwrap();
    g.records[100][0] ^= 1; // RAM digest after input 99
    g.save(&path).unwrap();
    let path = tmp.join("golden/title_idle_3600.tfgolden");
    let mut g = Golden::load(&path).unwrap();
    g.movie_hash[0] ^= 1;
    g.save(&path).unwrap();

    let out = tf(&["ladder-diff", &dir]);
    let text = stdout(&out);
    assert_eq!(out.status.code(), Some(1), "{text}");
    assert!(text.contains("pause_spam: first divergence at frame 100: ram\n"), "{text}");
    assert!(text.contains("title_idle_3600.tfgolden is stale"), "{text}");
    assert!(text.contains(&format!("2 of {shorts} movies diverge")), "{text}");
}

#[test]
fn corpus_check_passes() {
    if !have_rom() {
        return;
    }
    let out = tf(&["corpus", "check"]);
    assert_eq!(out.status.code(), Some(0), "{}", stdout(&out));
    assert!(stdout(&out).contains("corpus check passed"), "{}", stdout(&out));
}

/// `trace` prints one row per instruction of exactly the requested records, numbered from 1.
#[test]
fn trace_prints_the_requested_records() {
    if !have_rom() {
        return;
    }
    let movie = "corpus/movies/power_on_to_start.tfmovie";
    let out = tf(&["trace", movie, "--frame", "3", "--count", "2"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let text = stdout(&out);
    let records: Vec<&str> = text.lines().map(|l| l.split_whitespace().next().unwrap()).collect();
    let rows_of = |r: &str| records.iter().filter(|&&x| x == r).count();
    // Roughly one instruction per 3 of a frame's 29,780 cycles.
    assert!(rows_of("3") > 5000 && rows_of("4") > 5000, "{} rows", records.len());
    assert_eq!(rows_of("3") + rows_of("4"), records.len(), "only records 3 and 4");
    assert!(records.windows(2).all(|w| w[0] <= w[1]), "records in order");
    let first = text.lines().next().unwrap();
    for field in ["PC:", "OP:", "A:", "X:", "Y:", "P:", "SP:", "SL:", "DOT:", "CYC:"] {
        assert!(first.contains(field), "{field} missing from {first:?}");
    }

    let out = tf(&["trace", movie, "--frame", "0"]);
    assert!(stderr(&out).contains("1-based"), "{}", stderr(&out));
    let out = tf(&["trace", movie, "--frame", "663"]);
    assert!(stderr(&out).contains("movie has only 662 frames"), "{}", stderr(&out));
    // Running into the end stops there.
    let out = tf(&["trace", movie, "--frame", "662", "--count", "5"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(stdout(&out).lines().all(|l| l.trim_start().starts_with("662 ")));
}

#[test]
fn a_missing_rom_is_refused_up_front() {
    let out = tf(&["--rom", "/nonexistent/zelda.nes", "corpus", "check"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(stderr(&out).contains("ROM not found"), "{}", stderr(&out));
}
