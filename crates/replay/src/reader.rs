//! Reading a replay: the header and table are parsed (and INPUTS and STATES checked) up front, every
//! other stream is decoded as it is iterated.

use crate::format::{
    Cheat, Context, Cursor, Decision, ENDINGS, ENTRY_LEN, EVENT_CONTROL, EVENT_EPISODE, EVENT_OUTCOME,
    EVENT_ROOM, EVENT_SEGMENT, Edit, Event, EventRecord, HEADER_LEN, MAGIC, NO_POLICY, NO_STEP,
    REPLAY_VERSION, ReplayError, Reward, SOURCE_EMBEDDED, SOURCE_FILE, SOURCE_POWER_ON, Source, State,
    StateKind, format_error, section,
};

#[derive(Debug, Clone, Copy)]
pub(crate) struct Section {
    pub kind: u16,
    pub count: u32,
    pub last_frame: u32,
    pub offset: usize,
    pub len: usize,
}

/// A parsed replay file.  Owns the bytes.
#[derive(Debug, Clone)]
pub struct Replay {
    bytes: Vec<u8>,
    version: u32,
    frames: u32,
    ending: u8,
    source: u8,
    rom_sha1: [u8; 20],
    sections: Vec<Section>,
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("4 bytes"))
}

fn read_table(bytes: &[u8]) -> Result<Vec<Section>, ReplayError> {
    let count = u16::from_le_bytes([bytes[18], bytes[19]]) as usize;
    let mut end = HEADER_LEN + count * ENTRY_LEN;
    if bytes.len() < end {
        return format_error("the section table runs past the end of the file");
    }
    let mut sections: Vec<Section> = Vec::with_capacity(count);
    for entry in bytes[HEADER_LEN..end].chunks_exact(ENTRY_LEN) {
        let kind = u16::from_le_bytes([entry[0], entry[1]]);
        if sections.last().is_some_and(|last| last.kind >= kind) {
            return format_error("section kinds are out of order or repeated");
        }
        let offset = u64::from_le_bytes(entry[16..24].try_into().expect("8 bytes"));
        let len = u64::from_le_bytes(entry[24..32].try_into().expect("8 bytes"));
        if offset != end as u64 {
            return format_error(format!("section {kind} does not follow the one before it"));
        }
        let Some(next) = offset.checked_add(len).filter(|&next| next <= bytes.len() as u64) else {
            return format_error(format!("section {kind} runs past the end of the file"));
        };
        sections.push(Section {
            kind,
            count: u32_at(entry, 4),
            last_frame: u32_at(entry, 8),
            offset: end,
            len: len as usize,
        });
        end = next as usize;
    }
    if end != bytes.len() {
        return format_error(format!("{} bytes after the last section", bytes.len() - end));
    }
    Ok(sections)
}

impl Replay {
    /// Parses a replay file.  Refuses anything this build can't replay: another magic, a newer version, a
    /// damaged table, inputs that don't add up to the header's frame count, a missing END (or START) state.
    pub fn parse(bytes: Vec<u8>) -> Result<Replay, ReplayError> {
        if bytes.len() < HEADER_LEN || &bytes[..8] != MAGIC {
            return format_error("not a TFREPLAY file");
        }
        let version = u32_at(&bytes, 8);
        let table = read_table(&bytes);
        if version > REPLAY_VERSION {
            let recorded_by = table.ok().and_then(|sections| {
                let meta = sections.iter().find(|s| s.kind == section::META)?;
                let meta: serde_json::Value =
                    serde_json::from_slice(&bytes[meta.offset..meta.offset + meta.len]).ok()?;
                Some(meta.get("triforce")?.as_str()?.to_owned())
            });
            return Err(ReplayError::Version { found: version, recorded_by });
        }
        let replay = Replay {
            version,
            frames: u32_at(&bytes, 12),
            ending: bytes[16],
            source: bytes[17],
            rom_sha1: bytes[20..40].try_into().expect("20 bytes"),
            sections: table?,
            bytes,
        };
        replay.check()?;
        Ok(replay)
    }

    fn check(&self) -> Result<(), ReplayError> {
        if self.ending as usize >= ENDINGS.len() {
            return format_error(format!("unknown ending {}", self.ending));
        }
        if self.source > SOURCE_EMBEDDED {
            return format_error(format!("unknown source {}", self.source));
        }
        for kind in section::REQUIRED {
            if self.section(kind).is_none() {
                return format_error(format!("no {} section", section::NAMES[kind as usize]));
            }
        }
        let (mut frames, mut runs) = (0u64, 0u32);
        let mut cursor = self.cursor(section::INPUTS);
        while !cursor.done() {
            cursor.u8()?;
            let count = cursor.varint()?;
            if count == 0 {
                return cursor.error("an empty run");
            }
            frames += u64::from(count);
            runs += 1;
        }
        if frames != u64::from(self.frames) || runs != self.count(section::INPUTS) {
            return format_error(format!(
                "the inputs run {frames} frames in {runs} runs, the header says {} in {}",
                self.frames,
                self.count(section::INPUTS)
            ));
        }
        let mut states = Vec::new();
        for state in self.state_records() {
            states.push(state?);
        }
        let ends = states.iter().filter(|s| s.kind == StateKind::End).count();
        match states.last() {
            Some(last) if last.kind == StateKind::End && last.frame == self.frames && ends == 1 => {}
            _ => return format_error("no END state after the last frame"),
        }
        let starts = states.iter().filter(|s| s.kind == StateKind::Start).count();
        let embedded = self.source == SOURCE_EMBEDDED;
        let first_is_start = states[0].kind == StateKind::Start && states[0].frame == 0;
        if starts != usize::from(embedded) || (embedded && !first_is_start) {
            return format_error(if embedded {
                "the replay starts from an embedded state but has no START state at frame 0"
            } else {
                "a START state in a replay that does not start from one"
            });
        }
        let edits = self.count(section::EDITS);
        if states.windows(2).any(|w| w[0].edits > w[1].edits) || states.iter().any(|s| s.edits > edits) {
            return format_error("STATES section: edit counts are out of order");
        }
        Ok(())
    }

    pub(crate) fn section(&self, kind: u16) -> Option<&Section> {
        self.sections.iter().find(|s| s.kind == kind)
    }

    /// A section's bytes (empty when the file has no such section).
    pub(crate) fn section_bytes(&self, kind: u16) -> &[u8] {
        self.section(kind).map_or(&[], |s| &self.bytes[s.offset..s.offset + s.len])
    }

    pub(crate) fn count(&self, kind: u16) -> u32 {
        self.section(kind).map_or(0, |s| s.count)
    }

    pub(crate) fn last_frame(&self, kind: u16) -> u32 {
        self.section(kind).map_or(0, |s| s.last_frame)
    }

    fn cursor(&self, kind: u16) -> Cursor<'_> {
        Cursor::new(self.section_bytes(kind), section::NAMES.get(kind as usize).copied().unwrap_or("unknown"))
    }

    fn records<'a, T>(&'a self, kind: u16, width: usize, read: Read<'a, T>) -> Records<'a, T> {
        Records { cursor: self.cursor(kind), frame: 0, left: self.count(kind), width, read }
    }

    /// The file's bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The file's bytes, without copying.
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    /// Input frames.
    pub fn frames(&self) -> u32 {
        self.frames
    }

    /// The header's ending (indexes [`ENDINGS`]).
    pub fn ending(&self) -> u8 {
        self.ending
    }

    pub fn ending_name(&self) -> &'static str {
        ENDINGS[self.ending as usize]
    }

    /// SHA-1 of the ROM payload the run played on.
    pub fn rom_sha1(&self) -> [u8; 20] {
        self.rom_sha1
    }

    pub(crate) fn source_byte(&self) -> u8 {
        self.source
    }

    /// The table: `(kind, records, bytes)` per section, in file order.
    pub fn sections(&self) -> impl Iterator<Item = (u16, u32, usize)> + '_ {
        self.sections.iter().map(|s| (s.kind, s.count, s.len))
    }

    /// Whether the file has section `kind`.
    pub fn has(&self, kind: u16) -> bool {
        self.section(kind).is_some()
    }

    /// The META object.
    pub fn meta(&self) -> Result<serde_json::Map<String, serde_json::Value>, ReplayError> {
        match serde_json::from_slice(self.section_bytes(section::META)) {
            Ok(serde_json::Value::Object(meta)) => Ok(meta),
            Ok(_) => format_error("META section: not a JSON object"),
            Err(e) => format_error(format!("META section: {e}")),
        }
    }

    /// Where the replay starts.
    pub fn source(&self) -> Result<Source, ReplayError> {
        let mut cursor = self.cursor(section::SOURCE);
        let source = match self.source {
            SOURCE_POWER_ON => {
                let len = cursor.u8()? as usize;
                let Ok(name) = std::str::from_utf8(cursor.take(len)?) else {
                    return cursor.error("the name is not UTF-8");
                };
                Source::PowerOn { name: name.to_owned(), delay: cursor.u32()? }
            }
            SOURCE_FILE => {
                let file_hash = cursor.u64()?;
                Source::File { name: cursor.text()?, file_hash }
            }
            _ => Source::Embedded,
        };
        if !cursor.done() {
            return cursor.error("bytes after the source");
        }
        Ok(source)
    }

    /// The inputs as `(buttons, frames)` runs.
    pub fn runs(&self) -> impl Iterator<Item = (u8, u32)> + '_ {
        let mut cursor = self.cursor(section::INPUTS);
        // `parse` checked the section, so neither read fails.
        std::iter::from_fn(move || Some((cursor.u8().ok()?, cursor.varint().ok()?)))
    }

    /// The inputs, one controller byte per frame.
    pub fn inputs(&self) -> Vec<u8> {
        let mut inputs = Vec::with_capacity(self.frames as usize);
        for (buttons, count) in self.runs() {
            inputs.resize(inputs.len() + count as usize, buttons);
        }
        inputs
    }

    /// Every memory edit, in order.
    pub fn edits(&self) -> Records<'_, Edit> {
        self.records(section::EDITS, 0, |c, frame, _| {
            Ok(Edit { frame, addr: c.u16()?, prev: c.u8()?, value: c.u8()? })
        })
    }

    /// The cheated-item spend ledger.
    pub fn cheats(&self) -> Records<'_, Cheat> {
        self.records(section::CHEATS, 0, |c, frame, _| {
            let item = c.u8()?;
            if item as usize >= crate::format::CHEAT_ITEMS.len() {
                return c.error("an unknown cheat item");
            }
            Ok(Cheat { frame, item, count: c.varint()? })
        })
    }

    pub fn events(&self) -> Records<'_, EventRecord> {
        self.records(section::EVENTS, 0, |c, frame, _| {
            let event = match c.u8()? {
                EVENT_CONTROL => Event::Control,
                EVENT_ROOM => {
                    let (level, room, flags, step, policy) = (c.u8()?, c.u8()?, c.u8()?, c.u16()?, c.u8()?);
                    Event::Room {
                        level,
                        room,
                        in_cave: flags & 1 != 0,
                        step: (step != NO_STEP).then_some(step),
                        policy: (policy != NO_POLICY).then_some(policy),
                    }
                }
                EVENT_OUTCOME => Event::Outcome(c.text()?),
                EVENT_EPISODE => {
                    let (ending, class, flags) = (c.text()?, c.u8()?, c.u8()?);
                    Event::Episode { ending, class, terminated: flags & 1 != 0, truncated: flags & 2 != 0 }
                }
                EVENT_SEGMENT => Event::Segment,
                _ => return c.error("an unknown event"),
            };
            Ok(EventRecord { frame, event })
        })
    }

    /// The reward terms (empty without a REWARDS section).
    pub fn rewards(&self) -> Records<'_, Reward> {
        self.records(section::REWARDS, 0, |c, frame, _| {
            Ok(Reward { frame, term: c.u8()?, value: c.f32()? })
        })
    }

    /// Entries per decision's distribution (META `actions.distribution`; 0 without a DECISIONS section).
    pub fn distribution(&self) -> Result<usize, ReplayError> {
        if !self.has(section::DECISIONS) {
            return Ok(0);
        }
        match self.meta()?.get("actions").and_then(|a| a.get("distribution")).and_then(|n| n.as_u64()) {
            Some(n) if n <= 1 << 16 => Ok(n as usize),
            _ => format_error("META section: no actions.distribution for the DECISIONS section"),
        }
    }

    /// The model decisions (empty without a DECISIONS section).
    pub fn decisions(&self) -> Result<Records<'_, Decision<'_>>, ReplayError> {
        Ok(self.records(section::DECISIONS, self.distribution()?, |c, frame, width| {
            let policy = c.u8()?;
            let action = [c.u8()?, c.u8()?, c.u8()?];
            Ok(Decision { frame, policy, action, value: c.f32()?, dist: c.take(4 * width)? })
        }))
    }

    fn state_records(&self) -> Records<'_, State<'_>> {
        self.records(section::STATES, 0, |c, frame, _| {
            let Some(kind) = StateKind::from_byte(c.u8()?) else {
                return c.error("an unknown state kind");
            };
            let step = c.u16()?;
            let granted = [c.u8()?, c.u8()?, c.u8()?];
            let edits = c.varint()?;
            let len = c.u32()? as usize;
            let context = Context { step: (step != NO_STEP).then_some(step), granted };
            Ok(State { frame, kind, context, edits, blob: c.take(len)? })
        })
    }

    /// The states, in frame order: START (embedded source), keyframes and checkpoints, END last.
    pub fn states(&self) -> impl Iterator<Item = State<'_>> + '_ {
        self.state_records().filter_map(Result::ok)     // `parse` checked the section
    }

    /// The START state of an embedded-source replay.
    pub fn start_state(&self) -> Option<State<'_>> {
        self.states().next().filter(|s| s.kind == StateKind::Start)
    }

    /// The END state.
    pub fn end_state(&self) -> State<'_> {
        self.states().last().expect("parse checked the END state")
    }
}

type Read<'a, T> = fn(&mut Cursor<'a>, u32, usize) -> Result<T, ReplayError>;

/// A stream's records.  A cut-off or malformed record yields one `Err`, then the iterator ends.
pub struct Records<'a, T> {
    cursor: Cursor<'a>,
    frame: u32,
    left: u32,
    width: usize,
    read: Read<'a, T>,
}

impl<'a, T> Records<'a, T> {
    /// Records not read yet.
    pub fn remaining(&self) -> u32 {
        self.left
    }

    /// The frame of the record read last.
    pub(crate) fn frame(&self) -> u32 {
        self.frame
    }

    /// Bytes read so far.
    pub(crate) fn position(&self) -> usize {
        self.cursor.position()
    }

    /// The whole section.
    pub(crate) fn bytes(&self) -> &'a [u8] {
        self.cursor.all()
    }

    fn read(&mut self) -> Result<T, ReplayError> {
        let Some(frame) = self.frame.checked_add(self.cursor.varint()?) else {
            return self.cursor.error("a frame number is out of range");
        };
        self.frame = frame;
        (self.read)(&mut self.cursor, frame, self.width)
    }
}

impl<T> Iterator for Records<'_, T> {
    type Item = Result<T, ReplayError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.left == 0 {
            if self.cursor.done() {
                return None;
            }
            let error = self.cursor.error("bytes after the last record");
            self.cursor.end();
            return Some(error);
        }
        self.left -= 1;
        let item = self.read();
        if item.is_err() {
            self.left = 0;
            self.cursor.end();
        }
        Some(item)
    }
}
