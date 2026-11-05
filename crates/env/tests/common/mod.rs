//! Shared test helpers: an oracle-backed `Nes`, an `.npz` reader and the parity fixtures.
#![allow(dead_code)]

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use zelda::Nes;

/// `zelda::Nes` over the dev-only FCEUmm oracle (state = oracle savestate bytes); `.1` counts every frame run.
pub struct OracleNes(pub oracle::Oracle, pub u64);

impl Nes for OracleNes {
    type State = Vec<u8>;

    fn run_frame(&mut self, buttons: u8, _render: bool) {
        self.0.run_frame(buttons);
        self.1 += 1;
    }

    fn ram(&self) -> &[u8; 0x800] {
        self.0.ram()
    }

    fn prg_ram(&self) -> &[u8; 0x2000] {
        self.0.prg_ram()
    }

    fn poke(&mut self, addr: u16, value: u8) {
        self.0.poke(addr, value);
    }

    fn snapshot(&self) -> Vec<u8> {
        self.0.save_state()
    }

    fn restore(&mut self, state: &Vec<u8>) {
        self.0.load_state(state).expect("oracle load_state");
    }
}

pub fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Opens the oracle, or None (with the reason on stderr) when the library or ROM is missing.
pub fn open_oracle() -> Option<OracleNes> {
    if !oracle::available() {
        eprintln!("oracle library not built (oracle/fceumm/build.sh): skipping");
        return None;
    }
    let rom = std::env::var_os("TRIFORCE_ROM")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo().join("roms/zelda.nes"));
    if !rom.exists() {
        eprintln!("ROM not found: set TRIFORCE_ROM or create roms/zelda.nes");
        return None;
    }
    Some(OracleNes(
        oracle::Oracle::open(&rom).expect("open oracle"),
        0,
    ))
}

/// One array of an `.npz`.
#[derive(Clone, Debug)]
pub struct Npy {
    pub shape: Vec<usize>,
    pub data: Data,
}

#[derive(Clone, Debug)]
pub enum Data {
    I64(Vec<i64>),
    F64(Vec<f64>),
    F32(Vec<f32>),
    I16(Vec<i16>),
    U8(Vec<u8>),
    Bool(Vec<bool>),
    Str(Vec<String>),
}

impl Npy {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Elements per leading index.
    pub fn row(&self) -> usize {
        self.shape.iter().skip(1).product()
    }

    pub fn i64(&self) -> Vec<i64> {
        match &self.data {
            Data::I64(v) => v.clone(),
            Data::I16(v) => v.iter().map(|&x| x as i64).collect(),
            Data::U8(v) => v.iter().map(|&x| x as i64).collect(),
            Data::Bool(v) => v.iter().map(|&x| x as i64).collect(),
            d => panic!("not an integer array: {d:?}"),
        }
    }

    pub fn f64(&self) -> Vec<f64> {
        match &self.data {
            Data::F64(v) => v.clone(),
            Data::F32(v) => v.iter().map(|&x| x as f64).collect(),
            _ => self.i64().into_iter().map(|x| x as f64).collect(),
        }
    }

    pub fn bools(&self) -> Vec<bool> {
        match &self.data {
            Data::Bool(v) => v.clone(),
            _ => self.i64().into_iter().map(|x| x != 0).collect(),
        }
    }

    pub fn bytes(&self) -> Vec<u8> {
        match &self.data {
            Data::U8(v) => v.clone(),
            d => panic!("not a uint8 array: {d:?}"),
        }
    }

    pub fn strings(&self) -> Vec<String> {
        match &self.data {
            Data::Str(v) => v.clone(),
            d => panic!("not a string array: {d:?}"),
        }
    }

    pub fn scalar_i64(&self) -> i64 {
        self.i64()[0]
    }

    pub fn string(&self) -> String {
        self.strings()[0].clone()
    }
}

fn le<const N: usize, T>(body: &[u8], n: usize, f: fn([u8; N]) -> T) -> Vec<T> {
    body.as_chunks::<N>()
        .0
        .iter()
        .take(n)
        .map(|c| f(*c))
        .collect()
}

fn parse_npy(bytes: &[u8]) -> Npy {
    assert_eq!(&bytes[..6], b"\x93NUMPY", "not an npy file");
    let major = bytes[6];
    let (header_len, start) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
            12,
        )
    };
    let header = std::str::from_utf8(&bytes[start..start + header_len]).unwrap();
    let body = &bytes[start + header_len..];
    let field = |name: &str| {
        let i = header.find(&format!("'{name}':")).unwrap() + name.len() + 3;
        header[i..].trim_start()
    };
    let descr = field("descr");
    let descr = &descr[1..descr[1..].find('\'').unwrap() + 1];
    assert!(field("fortran_order").starts_with("False"), "fortran order");
    let shape_text = field("shape");
    let shape_text = &shape_text[1..shape_text.find(')').unwrap()];
    let shape: Vec<usize> = shape_text
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let n: usize = shape.iter().product();
    let data = match descr {
        "<i8" => Data::I64(le(body, n, i64::from_le_bytes)),
        "<f8" => Data::F64(le(body, n, f64::from_le_bytes)),
        "<f4" => Data::F32(le(body, n, f32::from_le_bytes)),
        "<i2" => Data::I16(le(body, n, i16::from_le_bytes)),
        "|u1" => Data::U8(body[..n].to_vec()),
        "|b1" => Data::Bool(body[..n].iter().map(|&b| b != 0).collect()),
        d if d.starts_with("<U") => {
            let width: usize = d[2..].parse().unwrap();
            Data::Str(
                body.chunks_exact(width * 4)
                    .take(n)
                    .map(|c| {
                        le(c, width, u32::from_le_bytes)
                            .into_iter()
                            .take_while(|&u| u != 0)
                            .map(|u| char::from_u32(u).unwrap())
                            .collect()
                    })
                    .collect(),
            )
        }
        d => panic!("unsupported npy dtype {d}"),
    };
    Npy { shape, data }
}

/// Reads every array of an `.npz`.
pub fn read_npz(path: &Path) -> HashMap<String, Npy> {
    let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut zip = zip::ZipArchive::new(file).unwrap();
    let mut out = HashMap::new();
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).unwrap();
        let name = entry.name().trim_end_matches(".npy").to_owned();
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes).unwrap();
        out.insert(name, parse_npy(&bytes));
    }
    out
}

pub fn fixture_dir() -> PathBuf {
    repo().join("crates/env/tests/fixtures/parity")
}

/// The fixture files, sorted by name.
pub fn fixtures() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(fixture_dir())
        .expect("fixture dir")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "npz"))
        .collect();
    v.sort();
    v
}

/// Oracle state of a fixture's room entry (FCEUmm body + the oracle's frame counter).
pub fn entry_state(f: &HashMap<String, Npy>) -> Vec<u8> {
    let mut state = f["oracle_state"].bytes();
    state.extend_from_slice(&(f["entry_frame"].scalar_i64() as u64).to_le_bytes());
    state
}

/// In-memory `StateStore` (tests): `load` finds a state by name or by the label of a saved path; `save` keeps
/// the machine state under the path.
#[derive(Default)]
pub struct MemStore(pub std::cell::RefCell<HashMap<String, Vec<u8>>>);

impl env::StateStore<Vec<u8>> for MemStore {
    fn load(&self, name: &str) -> Result<Vec<u8>, env::EnvError> {
        let states = self.0.borrow();
        states
            .get(name)
            .or_else(|| {
                let label = env::start_states::state_label(name);
                states
                    .iter()
                    .find(|(k, _)| env::start_states::state_label(k) == label)
                    .map(|(_, v)| v)
            })
            .cloned()
            .ok_or_else(|| env::EnvError::State(format!("no state {name}")))
    }

    fn save(
        &self,
        path: &Path,
        snapshot: &zelda::SessionSnapshot<Vec<u8>>,
    ) -> Result<(), env::EnvError> {
        std::fs::write(path, b"mem")?;
        self.0
            .borrow_mut()
            .insert(path.display().to_string(), snapshot.machine.clone());
        Ok(())
    }
}

/// `StateStore` over the legacy gzip start states (`triforce/custom_integrations/Zelda-NES/*.state`, FCEUmm
/// bodies the oracle loads directly): names resolve like `GameSession.load` (bare name -> `<dir>/<name>.state`),
/// and `save` writes a gzip body.  Loaded states get a zero frame counter.
pub struct GzipStore {
    pub dir: PathBuf,
}

impl GzipStore {
    pub fn new() -> GzipStore {
        GzipStore {
            dir: repo().join(env::config::STATES_DIR),
        }
    }

    pub fn path(&self, name: &str) -> PathBuf {
        let p = Path::new(name);
        if p.is_absolute() || p.exists() {
            return p.to_path_buf();
        }
        let file = if name.ends_with(".state") {
            name.to_owned()
        } else {
            format!("{name}.state")
        };
        self.dir.join(file)
    }
}

impl env::StateStore<Vec<u8>> for GzipStore {
    fn load(&self, name: &str) -> Result<Vec<u8>, env::EnvError> {
        let path = self.path(name);
        let file = std::fs::File::open(&path)
            .map_err(|e| env::EnvError::State(format!("{}: {e}", path.display())))?;
        let mut body = Vec::new();
        flate2::read::GzDecoder::new(file)
            .read_to_end(&mut body)
            .map_err(|e| env::EnvError::State(format!("{}: {e}", path.display())))?;
        body.extend_from_slice(&0u64.to_le_bytes());
        Ok(body)
    }

    fn save(
        &self,
        path: &Path,
        snapshot: &zelda::SessionSnapshot<Vec<u8>>,
    ) -> Result<(), env::EnvError> {
        use std::io::Write;
        let state = &snapshot.machine;
        let body = &state[..state.len() - 8];
        let tmp = PathBuf::from(format!("{}.{}.tmp", path.display(), std::process::id()));
        let mut enc = flate2::write::GzEncoder::new(
            std::fs::File::create(&tmp)?,
            flate2::Compression::default(),
        );
        enc.write_all(body)?;
        enc.finish()?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }
}
