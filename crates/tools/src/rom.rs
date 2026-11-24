//! ROM location and identity checks shared by every `tf` command.

use anyhow::{Context, Result, ensure};
use std::path::{Path, PathBuf};

pub const PAYLOAD_SHA1: &str = "a12d74c73a0481599a5d832361d168f4737bbcf6";

pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("workspace root exists")
}

pub struct RomInfo {
    pub bytes: Vec<u8>,
    pub payload_sha1: [u8; 20],
}

/// The ROM at `path`, else [`paths::rom_path`] (`$TRIFORCE_ROM`, else the shared store's `rom/zelda.nes`, else
/// `<repo>/roms/zelda.nes`; missing: an error naming every place looked in).
pub fn load(path: Option<&Path>) -> Result<RomInfo> {
    let path = match path {
        Some(path) => {
            ensure!(path.exists(), "ROM not found: {}", path.display());
            path.to_path_buf()
        }
        None => paths::rom_path(&workspace_root()).map_err(anyhow::Error::msg)?,
    };
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    ensure!(bytes.len() == 16 + 8 * 0x4000, "ROM {} is {} bytes, expected 131088", path.display(), bytes.len());
    ensure!(&bytes[..4] == b"NES\x1a", "ROM {} has no iNES header", path.display());
    let mapper = (bytes[6] >> 4) | (bytes[7] & 0xF0);
    ensure!(mapper == 1 && bytes[4] == 8 && bytes[5] == 0, "ROM {} is not MMC1 with 8x16 KiB PRG and CHR-RAM", path.display());
    let digest = sha1_smol::Sha1::from(&bytes[16..]).digest().bytes();
    ensure!(
        hex(&digest) == PAYLOAD_SHA1,
        "ROM {} payload SHA-1 {} != {PAYLOAD_SHA1}",
        path.display(),
        hex(&digest)
    );
    Ok(RomInfo { bytes, payload_sha1: digest })
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
