//! Movie scripts (`corpus/scripts/*.txt`), one directive per line, `#` comments:
//!
//! - `BUTTONS FRAMES` — hold buttons (`A+RIGHT`, `START`, `-` for none) for FRAMES frames.
//! - `poke ADDR VALUE` — write RAM/PRG-RAM before the next frame.
//! - `assert ram[ADDR] == VALUE` — checked after the frames so far have run.
//! - `until ram[ADDR] == VALUE BUTTONS max N` — hold BUTTONS until the condition holds (checked
//!   after each frame), failing after N frames.
//! - `tags: a, b` — movie tags.
//! - `base: NAME` (first directive only) — start from the end of corpus movie NAME (normally
//!   `power_on_to_start`); the recorded movie holds only what follows it.
//!
//! The recorded movie contains only frames and pokes; asserts and conditions are verified while
//! recording.

use crate::backend::Backend;
use crate::formats::{Movie, parse_hex_u16, parse_num_u8, poke_addr_allowed};
use anyhow::{Context, Result, bail, ensure};

#[derive(Clone, Debug, PartialEq)]
pub enum Directive {
    Hold { buttons: u8, frames: u32 },
    Poke { addr: u16, value: u8 },
    Assert { addr: u16, value: u8 },
    Until { addr: u16, value: u8, buttons: u8, max: u32 },
    Tags(Vec<String>),
    /// `base: NAME` (first line only): start from the end of corpus movie NAME.
    Base(String),
}

pub fn parse_buttons(s: &str) -> Result<u8> {
    if s == "-" {
        return Ok(0);
    }
    let mut b = 0u8;
    for part in s.split('+') {
        b |= match part.to_ascii_uppercase().as_str() {
            "A" => 0x01,
            "B" => 0x02,
            "SELECT" => 0x04,
            "START" => 0x08,
            "UP" => 0x10,
            "DOWN" => 0x20,
            "LEFT" => 0x40,
            "RIGHT" => 0x80,
            _ => bail!("unknown button {part:?}"),
        };
    }
    Ok(b)
}

fn parse_ram_ref(s: &str) -> Result<u16> {
    let inner = s
        .strip_prefix("ram[")
        .and_then(|t| t.strip_suffix(']'))
        .with_context(|| format!("expected ram[ADDR], got {s:?}"))?;
    let a = parse_hex_u16(inner)?;
    ensure!(poke_addr_allowed(a), "address ${a:04X} outside RAM/PRG-RAM");
    Ok(a)
}

pub fn parse(text: &str) -> Result<Vec<Directive>> {
    let mut out = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap().trim();
        if line.is_empty() {
            continue;
        }
        let d = parse_line(line).with_context(|| format!("line {}: {raw:?}", i + 1))?;
        out.push(d);
    }
    Ok(out)
}

fn parse_line(line: &str) -> Result<Directive> {
    if let Some(rest) = line.strip_prefix("base:") {
        return Ok(Directive::Base(rest.trim().to_owned()));
    }
    if let Some(rest) = line.strip_prefix("tags:") {
        return Ok(Directive::Tags(rest.split(',').map(|t| t.trim().to_owned()).filter(|t| !t.is_empty()).collect()));
    }
    let w: Vec<&str> = line.split_whitespace().collect();
    match w.as_slice() {
        ["poke", a, v] => {
            let addr = parse_hex_u16(a)?;
            ensure!(poke_addr_allowed(addr), "poke address ${addr:04X} not allowed");
            Ok(Directive::Poke { addr, value: parse_num_u8(v)? })
        }
        ["assert", r, "==", v] => Ok(Directive::Assert { addr: parse_ram_ref(r)?, value: parse_num_u8(v)? }),
        ["until", r, "==", v, b, "max", n] => Ok(Directive::Until {
            addr: parse_ram_ref(r)?,
            value: parse_num_u8(v)?,
            buttons: parse_buttons(b)?,
            max: n.parse()?,
        }),
        [b, n] => Ok(Directive::Hold { buttons: parse_buttons(b)?, frames: n.parse()? }),
        _ => bail!("unrecognized directive"),
    }
}

fn read_addr(b: &mut dyn Backend, addr: u16) -> u8 {
    let mut buf = Vec::new();
    if addr < 0x800 {
        b.ram()[addr as usize]
    } else {
        b.read_domain(crate::formats::Domain::PrgRam, &mut buf);
        buf[addr as usize - 0x6000]
    }
}

/// Run `directives` on `b`, appending to `movie` (which may already hold a prefix that `b` has
/// already played).
pub fn record(b: &mut dyn Backend, movie: &mut Movie, directives: &[Directive]) -> Result<()> {
    for d in directives {
        match d {
            &Directive::Hold { buttons, frames } => {
                for _ in 0..frames {
                    step(b, movie, buttons);
                }
            }
            &Directive::Poke { addr, value } => movie.push_poke(addr, value)?,
            &Directive::Assert { addr, value } => {
                let got = read_addr(b, addr);
                ensure!(
                    got == value,
                    "assert ram[${addr:04X}] == {value:#04x} failed after frame {}: got {got:#04x}",
                    movie.inputs.len()
                );
            }
            &Directive::Until { addr, value, buttons, max } => {
                let mut n = 0;
                while read_addr(b, addr) != value {
                    ensure!(n < max, "until ram[${addr:04X}] == {value:#04x} not reached in {max} frames");
                    step(b, movie, buttons);
                    n += 1;
                }
            }
            Directive::Tags(t) => {
                movie.meta["tags"] = serde_json::json!(t);
            }
            Directive::Base(_) => bail!("base: must be the first directive"),
        }
    }
    Ok(())
}

/// Run one frame on `b`, applying pending pokes for this frame first, and append it to `movie`.
pub fn step(b: &mut dyn Backend, movie: &mut Movie, buttons: u8) {
    let frame = movie.inputs.len() as u32;
    // Pending pokes are the tail of the list tagged with this frame.
    let start = movie.pokes.iter().rposition(|p| p.frame != frame).map_or(0, |i| i + 1);
    for p in &movie.pokes[start..] {
        b.poke(p.addr, p.value);
    }
    b.run_frame(buttons);
    movie.inputs.push(buttons);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_directives() {
        let d = parse("tags: short, x\nSTART 1 # go\n- 60\nA+RIGHT 16\npoke 0x6000 0x10\nassert ram[0x12] == 5\nuntil ram[$12] == 5 - max 900\n").unwrap();
        assert_eq!(d[0], Directive::Tags(vec!["short".into(), "x".into()]));
        assert_eq!(d[1], Directive::Hold { buttons: 8, frames: 1 });
        assert_eq!(d[2], Directive::Hold { buttons: 0, frames: 60 });
        assert_eq!(d[3], Directive::Hold { buttons: 0x81, frames: 16 });
        assert_eq!(d[4], Directive::Poke { addr: 0x6000, value: 0x10 });
        assert_eq!(d[5], Directive::Assert { addr: 0x12, value: 5 });
        assert_eq!(d[6], Directive::Until { addr: 0x12, value: 5, buttons: 0, max: 900 });
        assert!(parse("poke 0x2000 1").is_err());
        assert!(parse("JUMP 3").is_err());
    }
}
