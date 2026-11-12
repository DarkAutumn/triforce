#!/usr/bin/env python3
"""romdata.py - decodes the game's data tables into the markdown listings under data/.

The source includes 48 binary blobs with `.INCBIN "dat/X.dat"`; `src/bins.xml` gives each blob's
offset and length in the PRG image.  This tool reads the gameplay blobs from a ROM image (PRG0),
reads the tables that live in the source as `.BYTE` rows through tools/asm.py, decodes them the
way the game's code does, and writes:

    data/README.md                    formats, with a link to the routine that reads each field
    data/levelblock-ow.md             overworld rooms and caves (first quest, second-quest patches)
    data/levelblock-uw-1to6-q1.md     underworld rooms, levels 1-6, first quest
    data/levelblock-uw-7to9-q1.md     underworld rooms, levels 7-9, first quest
    data/levelblock-uw-1to6-q2.md     the same for the second quest
    data/levelblock-uw-7to9-q2.md
    data/levelinfo.md                 the 252-byte level info block of each level
    data/objlists.md                  monster lists
    data/room-layouts.md              room layouts, column tables, squares
    data/person-text.md               the text strings of cave and dungeon persons

Usage: romdata.py --rom PATH [--out DIR]
Exit status: 0 written, 1 the source and the ROM disagree, 2 the ROM is not PRG0.
The output is the same on every run.  Labels are written here by their upstream names and looked
up at run time, so the listings follow renames.  Standard library only.
"""

import argparse
import bisect
import hashlib
import os
import re
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asm  # noqa: E402

PRG0_SHA1 = "a12d74c73a0481599a5d832361d168f4737bbcf6"
PRG_SIZE = 0x20000
BANK_SIZE = 0x4000

# Object type names (the table in the README; types that share a name differ by id).
OBJ_NAMES = {
    0x01: "Lynel", 0x02: "Lynel", 0x03: "Moblin", 0x04: "Moblin", 0x05: "Goriya", 0x06: "Goriya",
    0x07: "Octorok slow", 0x08: "Octorok fast", 0x09: "blue Octorok slow", 0x0A: "blue Octorok fast",
    0x0B: "Darknut red", 0x0C: "Darknut blue", 0x0D: "Tektite", 0x0E: "Tektite", 0x0F: "Blue Leever",
    0x10: "Red Leever", 0x11: "Zora", 0x12: "Vire", 0x13: "Zol", 0x14: "child Gel", 0x15: "Gel",
    0x16: "Pols Voice", 0x17: "Like-Like", 0x18: "small Digdogger", 0x1A: "Peahat", 0x1B: "Keese blue",
    0x1C: "Keese red", 0x1D: "Keese black", 0x1E: "Armos", 0x1F: "boulder spawner", 0x20: "Boulder",
    0x21: "Ghini", 0x22: "flying Ghini", 0x23: "Blue Wizzrobe", 0x24: "Red Wizzrobe", 0x25: "Patra child",
    0x26: "Patra child", 0x27: "Wallmaster", 0x28: "Rope", 0x2A: "Stalfos", 0x2B: "Bubble", 0x2C: "Bubble",
    0x2D: "Bubble", 0x2E: "Whirlwind", 0x2F: "Pond fairy", 0x30: "Gibdo", 0x31: "Dodongo", 0x32: "Dodongo",
    0x33: "Gohma", 0x34: "Gohma", 0x35: "Rupee stash", 0x36: "Grumble", 0x37: "Zelda", 0x38: "Digdogger",
    0x39: "Digdogger", 0x3A: "Lanmola", 0x3B: "Lanmola", 0x3C: "Manhandla", 0x3D: "Aquamentus",
    0x3E: "Ganon", 0x3F: "Guard fire", 0x40: "Standing fire", 0x41: "Moldorm", 0x42: "Gleeok",
    0x43: "Gleeok", 0x44: "Gleeok", 0x45: "Gleeok", 0x46: "Gleeok head", 0x47: "Patra", 0x48: "Patra",
    0x49: "Trap", 0x4A: "Trap", 0x4B: "dungeon person", 0x4C: "dungeon person", 0x4D: "dungeon person",
    0x4E: "dungeon person", 0x4F: "dungeon person", 0x50: "dungeon person", 0x51: "life-or-money person",
    0x52: "dungeon person", 0x53: "flying rock", 0x54: "unknown shot", 0x55: "fireball", 0x56: "fireball",
    0x57: "sword shot", 0x58: "magic shot", 0x59: "magic shot", 0x5A: "shot", 0x5B: "monster arrow",
    0x5C: "monster boomerang", 0x5D: "dead dummy", 0x5E: "flute secret", 0x5F: "ladder",
    0x60: "dropped item", 0x61: "dock", 0x62: "rock", 0x63: "rock wall", 0x64: "tree", 0x65: "gravestone",
    0x66: "armos tile object", 0x67: "rock wall", 0x68: "push block",
}

ITEM_NAMES = {
    0x00: "bomb", 0x01: "wooden sword", 0x02: "white sword", 0x03: "magic sword", 0x04: "food",
    0x05: "flute", 0x06: "blue candle", 0x07: "red candle", 0x08: "wooden arrow", 0x09: "silver arrow",
    0x0A: "bow", 0x0B: "magic key", 0x0C: "raft", 0x0D: "ladder", 0x0E: "Triforce of Power",
    0x0F: "5 rupees", 0x10: "rod", 0x11: "book", 0x12: "blue ring", 0x13: "red ring", 0x14: "bracelet",
    0x15: "letter", 0x16: "compass", 0x17: "map", 0x18: "rupee", 0x19: "key", 0x1A: "heart container",
    0x1B: "Triforce piece", 0x1C: "magic shield", 0x1D: "wooden boomerang", 0x1E: "magic boomerang",
    0x1F: "blue potion", 0x20: "red potion", 0x21: "clock", 0x22: "heart", 0x23: "fairy",
}

DOOR_NAMES = ["open", "wall", "false", "false2", "bomb", "key", "key2", "shutter"]
SECRET_NAMES = [
    "-",
    "all dead: shutters open",
    "ringleader: first monster dead kills all",
    "last boss dead: shutters open",
    "push block: shutters open",
    "push block: stairs appear",
    "life-or-money person gone: shutters open",
    "all dead: shutters open, room item appears",
]
CAVE_FLAG_NAMES = [
    (0x01, "pick-up"), (0x02, "pay"), (0x04, "show-items"), (0x08, "show-prices"),
    (0x10, "hint"), (0x20, "money"), (0x40, "heart-requirement"), (0x80, "minus-sign"),
]

# Character codes of person text: tile numbers of the common background pattern table.
CHARS = {i: str(i) for i in range(10)}
CHARS.update({10 + i: chr(65 + i) for i in range(26)})
CHARS.update({0x24: " ", 0x25: " ", 0x28: ",", 0x29: "!", 0x2A: "'", 0x2B: "&", 0x2C: ".", 0x2D: '"',
              0x2E: "?", 0x2F: "-"})


class DataError(Exception):
    pass


def hx(v, digits=2):
    return "$%0*X" % (digits, v)


def hexs(data):
    return " ".join(hx(b) for b in data)


def obj_name(t):
    if t == 0:
        return "(empty)"
    if t >= 0x6A:
        return "cave person"
    return OBJ_NAMES.get(t, "type")


def item_name(i):
    return ITEM_NAMES.get(i, "item")


def parse_nums(operand):
    out = []
    for tok in operand.split(","):
        tok = tok.strip()
        if tok.startswith("$"):
            out.append(int(tok[1:], 16))
        elif tok.startswith("%"):
            out.append(int(tok[1:], 2))
        else:
            out.append(int(tok))
    return out


def table(headers, rows, sep="  "):
    """Fixed-width text table (left-aligned columns; the last column is not padded)."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row[:-1]):
            widths[i] = max(widths[i], len(cell))
    out = []
    for row in [headers] + rows:
        cells = [c.ljust(widths[i]) for i, c in enumerate(row[:-1])] + [row[-1]]
        out.append(sep.join(cells).rstrip())
    return out


def fenced(lines):
    return ["```"] + list(lines) + ["```"]


# ---- the source and the ROM ---------------------------------------------------------------------

class Source:
    """The parsed source (through asm.py) plus the ROM's PRG image."""

    def __init__(self, root, prg, outdir):
        self.root, self.prg, self.outdir = root, prg, outdir
        self.prog = asm.Program(root)
        if self.prog.errors:
            f, n, m = self.prog.errors[0]
            raise DataError("the source does not parse: %s:%s: %s" % (f, n, m))
        self.u2c = asm.upstream_to_current(root)
        self.bins = {}
        for b in ET.parse(os.path.join(root, "src", "bins.xml")).getroot():
            self.bins[b.get("FileName")] = (int(b.get("Offset")), int(b.get("Length")))
        self._gpos = {}

    # -- names and links
    def cur(self, up):
        """Current name of an upstream label or symbol."""
        if up not in self.u2c:
            raise DataError("no upstream name %s" % up)
        return self.u2c[up]

    def resolve(self, up):
        try:
            return self.prog.resolve_name(self.cur(up))
        except asm.AsmError as e:
            raise DataError(str(e))

    def link(self, name, file, line):
        rel = os.path.relpath(os.path.join(self.prog.src, file), self.outdir).replace(os.sep, "/")
        return "[`%s`](%s#L%d)" % (name, rel, line)

    def ref(self, up):
        """Markdown reference link to the label or symbol with this upstream name."""
        kind, file, line, obj = self.resolve(up)
        return self.link(self.cur(up), file, line)

    def refs(self, *ups):
        return ", ".join(self.ref(u) for u in ups)

    def ref_label(self, lab):
        return self.link(lab.name, lab.file, lab.no)

    def sym(self, up):
        kind, file, line, obj = self.resolve(up)
        if kind != "symbol":
            raise DataError("%s is not a symbol" % up)
        return obj.value

    def label(self, up):
        kind, file, line, obj = self.resolve(up)
        if kind != "global":
            raise DataError("%s is not a global label" % up)
        return obj

    # -- statements
    def stmts_of(self, lab):
        """Statements from a global label up to the next global label of its bank."""
        mod = lab.mod
        pos = self._gpos.get(mod.id)
        if pos is None:
            pos = self._gpos[mod.id] = sorted({l.pos for l in mod.labels if l.kind == "global"})
        i = bisect.bisect_right(pos, lab.pos)
        end = pos[i] if i < len(pos) else len(mod.stmts)
        return mod.stmts[lab.pos:end]

    def label_at(self, mid, pos):
        mod = self.prog.mods[mid]
        for l in self.prog.labels_at(mod).get(pos, []):
            if l.kind == "global":
                return l
        raise DataError("no global label at statement %d of bank %d" % (pos, mid))

    def blob_file(self, name):
        if name not in self.bins:
            raise DataError("%s is not listed in src/bins.xml" % name)
        off, length = self.bins[name]
        return self.prg[off:off + length]

    def stmt_bytes(self, st):
        """Bytes of one data statement; address entries become zero bytes (their value is not known)."""
        if st.head == ".BYTE":
            return bytes(parse_nums(st.operand))
        if st.head == ".INCBIN":
            return self.blob_file(st.operand.strip().strip('"'))
        n = len(st.operand.split(","))
        if st.head in (".ADDR", ".DBYT"):
            return bytes(2 * n)
        if st.head in (".LOBYTES", ".HIBYTES"):
            return bytes(n)
        raise DataError("%s:%d: not a data statement" % (st.file, st.no))

    def data_of(self, lab):
        """The `.BYTE` rows and blobs directly under a global label."""
        out = bytearray()
        for st in self.stmts_of(lab):
            if st.head not in (".BYTE", ".INCBIN"):
                break
            out += self.stmt_bytes(st)
        return bytes(out)

    def data(self, up):
        return self.data_of(self.label(up))

    def region(self, lab_a, lab_b_end):
        """Bytes of the data statements from label a through the end of label b; {statement: offset}."""
        mod = lab_a.mod
        last = self.stmts_of(lab_b_end)
        end = last[-1].idx + 1 if last else lab_b_end.pos
        out, offs = bytearray(), {}
        for st in mod.stmts[lab_a.pos:end]:
            offs[st.idx] = len(out)
            out += self.stmt_bytes(st)
        offs[end] = len(out)
        return bytes(out), offs

    def addrs(self, up):
        """Entries of an address table: (bank, statement index, offset) per entry."""
        out = []
        for st in self.stmts_of(self.label(up)):
            if st.head != ".ADDR":
                break
            for tok in st.canon.split(None, 1)[1].split(","):
                m = re.match(r"^L(\d):(\d+)(?:([+-])\$([0-9A-F]+))?$", tok)
                if not m:
                    raise DataError("%s:%d: address entry `%s` is not a label" % (st.file, st.no, tok))
                off = int(m.group(4), 16) if m.group(4) else 0
                out.append((int(m.group(1)), int(m.group(2)), -off if m.group(3) == "-" else off))
        return out

    def incbin_of(self, lab):
        st = lab.mod.stmts[lab.pos]
        if st.head != ".INCBIN":
            raise DataError("%s does not include a blob" % lab.name)
        return st.operand.strip().strip('"')

    def rom_find(self, data, bank):
        """PRG offset of a byte string inside one bank; it must occur exactly once."""
        base = bank * BANK_SIZE
        i = self.prg.find(data, base, base + BANK_SIZE)
        if i < 0 or self.prg.find(data, i + 1, base + BANK_SIZE) >= 0:
            raise DataError("cannot locate a %d-byte source table of bank %d in the ROM" % (len(data), bank))
        return i

    def imm_before_store(self, up, sym_up):
        """Immediate values that a routine loads and stores into a symbol (`LDA #v` / `STA sym`)."""
        target = "STA $%X" % self.sym(sym_up)
        out, last = [], None
        for st in self.stmts_of(self.label(up)):
            m = re.match(r"^LDA #\$([0-9A-F]+)$", st.canon)
            if m:
                last = int(m.group(1), 16)
            elif st.canon == target and last is not None:
                out.append(last)
            elif st.head == "LDA":
                last = None
        return out


# ---- decoding -------------------------------------------------------------------------------------

class Game:
    """Everything decoded once; the writers below only format it."""

    def __init__(self, src):
        self.src = s = src
        self.block_base = s.sym("LevelBlockAttrsA")
        self.info_base = s.sym("LevelInfo_PalettesTransferBuf")
        for i, up in enumerate(["LevelBlockAttrsA", "LevelBlockAttrsB", "LevelBlockAttrsC", "LevelBlockAttrsD",
                                "LevelBlockAttrsE", "LevelBlockAttrsF"]):
            if s.sym(up) != self.block_base + 128 * i:
                raise DataError("%s is not at %s" % (up, hx(self.block_base + 128 * i, 4)))

        # level -> level block label per quest, level -> level info label
        self.block_labels = []
        for up in ("LevelBlockAddrsQ1", "LevelBlockAddrsQ2"):
            self.block_labels.append([s.label_at(m, p) for m, p, o in s.addrs(up)])
        self.info_labels = [s.label_at(m, p) for m, p, o in s.addrs("LevelInfoAddrs")]
        if not (len(self.block_labels[0]) == len(self.block_labels[1]) == len(self.info_labels) == 10):
            raise DataError("the level address tables do not have 10 entries")
        self.blocks = {}
        for q in (0, 1):
            for lab in self.block_labels[q]:
                if lab.name not in self.blocks:
                    data = s.data_of(lab)
                    if len(data) != 768:
                        raise DataError("level block %s is %d bytes" % (lab.name, len(data)))
                    self.blocks[lab.name] = data
        self.infos = [[], []]
        for lab in self.info_labels:
            data = s.data_of(lab)
            if len(data) != 252:
                raise DataError("level info %s is %d bytes" % (lab.name, len(data)))
            self.infos[0].append(data)

        # field offsets inside the level info block
        f = lambda up: s.sym(up) - self.info_base
        self.o_pal, self.o_foe, self.o_starty = 0, f("LevelInfo_FoeCounts"), f("LevelInfo_StartY")
        self.o_pos, self.o_rot = f("LevelInfo_ShortcutOrItemPosArray"), f("LevelInfo_SubmenuMapRotation")
        self.o_xoff, self.o_start = f("LevelInfo_StatusBarMapXOffset"), f("LevelInfo_StartRoomId")
        self.o_tri, self.o_flags = f("LevelInfo_TriforceRoomId"), f("LevelInfo_WorldFlagsAddr")
        self.o_num, self.o_cellar = f("LevelInfo_LevelNumber"), f("LevelInfo_CellarRoomIdArray")
        self.o_boss, self.o_mask = f("LevelInfo_BossRoomId"), f("LevelInfo_SubmenuMapMask")
        self.o_sbmap, self.o_cycles = f("LevelInfo_StatusBarMapTransferBuf"), f("LevelInfo_PaletteCycles")
        self.o_death = f("LevelInfo_DeathPaletteSeries")

        self._quest2()
        self._objlists()
        self._layouts()
        self._caves()
        self._text()
        self.dock_rooms, self.dock_type = self._dock_rooms()
        self.boss_sounds = s.data("BossSoundEffects")

    # -- second quest
    def _quest2(self):
        s = self.src
        # level info: (length + 1) bytes copied over the block from the position array on
        sizes = s.data("LevelInfoUWQ2ReplacementSizes")[:9]
        self.q2_info_patches = []
        self.infos[1] = [self.infos[0][0]]
        dst = self.o_pos
        for i, (m, p, o) in enumerate(s.addrs("LevelInfoUWQ2ReplacementAddrs")):
            lab = s.label_at(m, p)
            tbl = s.data_of(lab)
            off = s.rom_find(tbl, m)
            n = sizes[i] + 1
            patch = s.prg[off:off + n]
            info = bytearray(self.infos[0][i + 1])
            info[dst:dst + n] = patch
            self.infos[1].append(bytes(info))
            self.q2_info_patches.append((lab, len(tbl), sizes[i], patch))
        if len(self.infos[1]) != 10:
            raise DataError("expected 9 second-quest level info tables")
        # overworld level block: a table of (offset, value) pairs over array B, then fixed stores
        base_b = s.sym("LevelBlockAttrsB") - self.block_base
        offs = s.data("LevelBlockAttrsBQ2ReplacementOffsets")
        vals = s.data("LevelBlockAttrsBQ2ReplacementValues")
        self.q2_ow_patches = [(base_b + o, v, "table") for o, v in zip(offs, vals)]
        last = None
        for st in s.stmts_of(s.label("UpdateMode2Load_Full")):
            m = re.match(r"^LDA #\$([0-9A-F]+)$", st.canon)
            if m:
                last = int(m.group(1), 16)
                continue
            m = re.match(r"^STA \$([0-9A-F]+)(?:\+\$([0-9A-F]+))?$", st.canon)
            if m and last is not None:
                addr = int(m.group(1), 16) + (int(m.group(2), 16) if m.group(2) else 0)
                if self.block_base <= addr < self.block_base + 768:
                    self.q2_ow_patches.append((addr - self.block_base, last, "store"))
        ow_name = self.block_labels[0][0].name
        if self.block_labels[1][0].name != ow_name:
            raise DataError("the second quest uses another overworld level block")
        ow2 = bytearray(self.blocks[ow_name])
        for off, val, how in self.q2_ow_patches:
            ow2[off] = val
        self.ow = [self.blocks[ow_name], bytes(ow2)]

    # -- monster lists
    def _objlists(self):
        s = self.src
        self.objlists = s.data("ObjLists")
        self.objlist_offs = [o for m, p, o in s.addrs("ObjListAddrs")]
        self.list_base = None
        need = {"CPY #$32": False, "CPY #$62": False}
        for st in s.stmts_of(s.label("InitMode_EnterRoom")):
            m = re.match(r"^SBC #\$([0-9A-F]+)$", st.canon)
            if m:
                self.list_base = int(m.group(1), 16)        # the value the code subtracts from a list id
            if st.canon in need:
                need[st.canon] = True
        if self.list_base != 0x62 or not all(need.values()):
            raise DataError("the monster list thresholds of the room entry code are not $32 and $62")

    def monsters(self, list_id, count):
        """(count used, text) for a monster list id and the count from the level info."""
        if list_id == 0:
            return 0, "none"
        if list_id < self.list_base:
            n = 1 if list_id >= 0x32 else count
            return n, "%s %s x%d" % (hx(list_id), obj_name(list_id), n)
        idx = list_id - self.list_base
        if idx >= len(self.objlist_offs):
            return count, "list %s (no such list)" % hx(list_id)
        off = self.objlist_offs[idx]
        types = list(self.objlists[off:off + count])
        parts, seen = [], []
        for t in types:
            if t not in seen:
                seen.append(t)
        for t in seen:
            parts.append("%s %s x%d" % (hx(t), obj_name(t), types.count(t)))
        return count, "list %s: %s" % (hx(list_id), ", ".join(parts))

    # -- room layouts
    def _layouts(self):
        s = self.src
        # overworld: layouts, cave layouts and the 16 column tables are one contiguous run
        l_first, l_end = s.label("RoomLayoutsOW"), s.label("RoomLayoutsOWAddr")
        heap_labels = [s.label("ColumnHeapOW%X" % i) for i in range(16)]
        reg, offs = s.region(l_first, heap_labels[-1])
        if heap_labels[-1].pos >= l_end.pos:
            raise DataError("overworld column tables are out of order")
        self.ow_layout_bytes = reg[:offs[heap_labels[0].pos]]
        self.ow_heap = reg[offs[heap_labels[0].pos]:]
        self.ow_heap_starts = [offs[l.pos] - offs[heap_labels[0].pos] for l in heap_labels]
        self.ow_heap_labels = heap_labels
        blob_off = s.bins[s.incbin_of(l_first)][0]
        cpu0 = 0x8000 + blob_off % BANK_SIZE + offs[heap_labels[0].pos]
        direc = s.data("ColumnDirectoryOW")
        want = [cpu0 + o for o in self.ow_heap_starts]
        got = [direc[2 * i] | direc[2 * i + 1] << 8 for i in range(16)]
        if got != want:
            raise DataError("the overworld column directory does not point at the 16 column tables")
        self.ow_layout_count = len(s.data_of(l_first)) // 16
        self.ow_extra_layouts = []
        for up in ("RoomLayoutOWCave0", "RoomLayoutOWCave1", "RoomLayoutOWCave2"):
            lab = s.label(up)
            self.ow_extra_layouts.append((offs[lab.pos] // 16, lab))
            if offs[lab.pos] % 16:
                raise DataError("%s is not on a layout boundary" % up)
        self.sq_primary = s.data("PrimarySquaresOW") + s.data("SecondarySquaresOW")
        self.sq_primary_len = len(s.data("PrimarySquaresOW"))
        self.sq_secondary = s.data("SecondarySquaresOW")
        self.tileobj_types = s.data("TileObjectTypes")
        self.tileobj_squares = s.data("TileObjectPrimarySquaresOW")
        # underworld: layouts and the 10 column tables named by the directory
        u_first = s.label("RoomLayoutsUW")
        self.uw_layout_bytes = s.data_of(u_first)
        self.uw_layout_count = len(self.uw_layout_bytes) // 12
        self.uw_heap_labels = [s.label_at(m, p) for m, p, o in s.addrs("ColumnDirectoryUW")]
        reg, offs = s.region(self.uw_heap_labels[0], self.uw_heap_labels[-1])
        self.uw_heap = reg
        self.uw_heap_starts = [offs[l.pos] for l in self.uw_heap_labels]
        self.sq_uw = s.data("PrimarySquaresUW")
        # caves and cellars (laid out by the overworld routine)
        self.sub_layouts = [s.label_at(m, p) for m, p, o in s.addrs("SubroomLayoutAddrs")]
        self.cellar_heap = s.data("ColumnHeapUWCellar")

    def ow_tables(self, cellar=False):
        tabs = [(self.ow_heap, st) for st in self.ow_heap_starts]
        if cellar:
            tabs[0] = (self.cellar_heap, 0)
        return tabs

    def ow_grid(self, desc, tables):
        """11 rows x 16 columns of square indexes, decoded as the overworld layout routine does."""
        grid = [[None] * 16 for _ in range(11)]
        rep = 0
        for c, d in enumerate(desc):
            data, y = tables[d >> 4]
            x = d & 0x0F
            try:
                while True:
                    if data[y] & 0x80:
                        x -= 1
                        if x < 0:
                            break
                    y += 1
                for r in range(11):
                    b = data[y]
                    grid[r][c] = b & 0x3F
                    if b & 0x40:
                        rep ^= 0x40
                        if rep:
                            continue
                    y += 1
            except IndexError:
                pass
        return grid

    def uw_grid(self, desc):
        """7 rows x 12 columns of square indexes, decoded as the underworld floor routine does."""
        grid = [[None] * 12 for _ in range(7)]
        data = self.uw_heap
        for c, d in enumerate(desc):
            y, x = self.uw_heap_starts[d >> 4] if (d >> 4) < len(self.uw_heap_starts) else len(data), d & 0x0F
            try:
                while True:
                    if data[y] & 0x80:
                        x -= 1
                        if x < 0:
                            break
                    y += 1
                rep = 0
                for r in range(7):
                    b = data[y]
                    grid[r][c] = b & 0x07
                    if (b >> 4) & 0x07 == rep:
                        rep = 0
                        y += 1
                    else:
                        rep += 1
            except IndexError:
                pass
        return grid

    def ow_layout(self, lid):
        return self.ow_layout_bytes[lid * 16:lid * 16 + 16]

    def ow_tile_object(self, room, lid):
        """The tile object the overworld layout leaves for a room: (type, x, y) or None."""
        if room in self.dock_rooms:
            return (self.dock_type, None, None)
        desc = self.ow_layout(lid)
        if len(desc) < 16:
            return None
        found = None
        grid = self.ow_grid(desc, self.ow_tables())
        for c in range(16):
            for r in range(11):
                sq = grid[r][c]
                if sq is None:
                    continue
                prim = self.sq_primary[sq]
                if 0xE5 <= prim <= 0xEA:
                    found = (self.tileobj_types[prim - 0xE5], c * 16, r * 16 + 0x40)
        return found

    def _dock_rooms(self):
        rooms, typ = [], None
        for st in self.src.stmts_of(self.src.label("SetupTileObjectOW")):
            m = re.match(r"^(CMP|LDA) #\$([0-9A-F]+)$", st.canon)
            if m and m.group(1) == "CMP":
                rooms.append(int(m.group(2), 16))
            elif m:
                typ = int(m.group(2), 16)
                break
        if typ is None:
            raise DataError("cannot read the dock rooms")
        return rooms, typ

    # -- caves
    def _caves(self):
        s = self.src
        sel = s.data("OverworldPersonTextSelectors")
        e = self.ow[0][4 * 128:5 * 128]
        self.cave_count = len(sel)
        self.price_base = s.sym("LevelBlockAttrsE")
        for st in s.stmts_of(s.label("InitCaveContinue")):
            m = re.match(r"^LDA \$([0-9A-F]+)\+\$([0-9A-F]+),Y$", st.canon)
            if m and int(m.group(1), 16) == s.sym("LevelBlockAttrsE"):
                self.price_off = int(m.group(2), 16)
                break
        else:
            raise DataError("cannot find the price table offset")
        self.caves = []
        for n in range(self.cave_count):
            items = list(e[3 * n:3 * n + 3])
            prices = list(e[self.price_off + 3 * n:self.price_off + 3 * n + 3])
            flags = (items[0] & 0xC0) | (items[1] & 0xC0) >> 2 | (items[2] & 0xC0) >> 4 | sel[n] >> 6
            self.caves.append({"n": n, "index": 0x10 + n, "type": 0x6A + n, "sel": sel[n] & 0x3F,
                               "selbyte": sel[n], "items": items, "prices": prices, "flags": flags})

    def cave_flags(self, flags):
        return ",".join(name for bit, name in CAVE_FLAG_NAMES if flags & bit) or "-"

    def cave_wares(self, cave):
        out = []
        for it, pr in zip(cave["items"], cave["prices"]):
            i = it & 0x3F
            if i == 0x3F:
                out.append("-")
            else:
                out.append("%s %s%s" % (hx(i), item_name(i), " @%d" % pr if pr else ""))
        return out

    # -- person text
    def _text(self):
        s = self.src
        blob = s.data("PersonText")
        self.texts = []
        for i, (m, p, off) in enumerate(s.addrs("PersonTextAddrs")):
            lines, cur, j = [], [], off
            while j < len(blob):
                b = blob[j]
                cur.append(b & 0x3F)
                j += 1
                if b & 0xC0:
                    lines.append((cur, b & 0xC0))
                    cur = []
                    if b & 0xC0 == 0xC0:
                        break
            self.texts.append({"sel": 2 * i, "off": off, "len": j - off, "lines": lines})

    def text_plain(self, sel):
        """The string of a text selector as one line of plain text."""
        if sel % 2 or sel // 2 >= len(self.texts):
            return "?"
        out = []
        for codes, mark in self.texts[sel // 2]["lines"]:
            out.append("".join(CHARS.get(c, "?") for c in codes).strip())
        return " / ".join(out)

    # -- level info
    def info(self, quest, level):
        return self.infos[quest][level]

    def members(self, info):
        """Rooms the level's submenu map shows (the mask is applied after the rotation)."""
        rot, mask = info[self.o_rot], info[self.o_mask:self.o_mask + 16]
        return {r * 16 + c for r in range(8) for c in range(16) if mask[(c + rot) & 15] & (0x80 >> r)}

    def cellars(self, info):
        return [b for b in info[self.o_cellar:self.o_cellar + 10] if b != 0xFF]

    def pos(self, info, idx):
        b = info[self.o_pos + idx]
        return b & 0xF0, (b & 0x0F) << 4

    def levels_of_block(self, quest, name):
        return [lv for lv in range(10) if self.block_labels[quest][lv].name == name]


def decode_transfer(buf):
    """Transfer records: (address, vertical, repeat, count, data bytes) until an address byte >= $80."""
    out, i = [], 0
    while i + 2 < len(buf) and buf[i] < 0x80:
        addr, ctl = buf[i] << 8 | buf[i + 1], buf[i + 2]
        count = (ctl & 0x3F) or 64
        n = 1 if ctl & 0x40 else count
        out.append((addr, bool(ctl & 0x80), bool(ctl & 0x40), count, bytes(buf[i + 3:i + 3 + n])))
        i += 3 + n
    return out, i


# ---- writers --------------------------------------------------------------------------------------

def write_readme(g):
    s = g.src
    R, RS, N = s.ref, s.refs, s.cur
    L = []
    A = L.append
    A("# Game data listings")
    A("")
    A("How to read this: the files in this directory are decoded listings of the game's data tables. They are")
    A("written by `tools/romdata.py --rom <image>` from a PRG0 ROM image and from the tables in `src/`; do not")
    A("edit them by hand. This file describes every format and links each field to the routine that reads it.")
    A("In the listings every row starts with its id as `$xx`, so `grep '^\\$32 '` finds a room. Hex is written")
    A("`$xx`; bit 7 is the high bit; a room id is `$00`-`$7F` (row in the high nibble, column in the low nibble")
    A("of a 16 x 8 map).")
    A("")
    A("| File | Contents |")
    A("| --- | --- |")
    A("| `levelblock-ow.md` | overworld rooms: exits, caves, monsters, layout, secrets; cave wares; Q2 patches |")
    A("| `levelblock-uw-1to6-q1.md`, `levelblock-uw-7to9-q1.md` | underworld rooms of Q1: doors, monsters, item, secret |")
    A("| `levelblock-uw-1to6-q2.md`, `levelblock-uw-7to9-q2.md` | the same for Q2 |")
    A("| `levelinfo.md` | per level: start room, Triforce room, boss room, cellars, positions, map, palettes |")
    A("| `objlists.md` | monster lists and how a room selects one; object type names |")
    A("| `room-layouts.md` | room layouts (overworld, underworld, caves, cellars), column tables, squares |")
    A("| `person-text.md` | every text string of cave and dungeon persons |")
    A("")
    A("# Blobs")
    A("")
    A("The source includes 48 binary blobs with `.INCBIN`; `src/bins.xml` gives the offset of each in the PRG")
    A("image (the ROM file without its 16-byte header) and its length. Bank = offset / `$4000`.")
    A("")
    A("| Blob | PRG offset | Length | Bank | Included at |")
    A("| --- | --- | --- | --- | --- |")
    incl = {}
    for mod in s.prog.mods:
        for st in mod.stmts:
            if st.head == ".INCBIN":
                incl[st.operand.strip().strip('"')] = s.label_at(mod.id, st.idx)
    for name, (off, length) in sorted(s.bins.items(), key=lambda kv: kv[1][0]):
        lab = incl.get(name)
        A("| `%s` | %s | %d | %d | %s |" % (name, hx(off, 5), length, off // BANK_SIZE,
                                         s.ref_label(lab) if lab else "(not included)"))
    A("")

    A("# Level blocks")
    A("")
    A("A level block is 768 bytes: six arrays of 128 bytes, called A to F, each with one byte per room.")
    A("")
    A("- Loading a level copies one block to RAM at %s (arrays B to F follow at 128-byte steps: %s) and one"
      % (R("LevelBlockAttrsA"), RS("LevelBlockAttrsB", "LevelBlockAttrsC", "LevelBlockAttrsD",
                                  "LevelBlockAttrsE", "LevelBlockAttrsF")))
    A("  level info block after it; all code reads the RAM copy. %s" % RS("InitMode2_Sub0", "CopyBlock",
                                                                          "FetchLevelBlockDestInfo"))
    A("- The block is chosen by the level number and the quest of the save slot: %s for Q1, %s for Q2. %s"
      % (R("LevelBlockAddrsQ1"), R("LevelBlockAddrsQ2"), R("InitMode2_Sub0")))
    A("")
    A("| Level | Q1 block | Q2 block | Level info |")
    A("| --- | --- | --- | --- |")
    for lv in range(10):
        A("| %s | %s (`%s`) | %s (`%s`) | %s (`%s`) |" % (
            "overworld" if lv == 0 else str(lv),
            s.ref_label(g.block_labels[0][lv]), s.incbin_of(g.block_labels[0][lv]),
            s.ref_label(g.block_labels[1][lv]), s.incbin_of(g.block_labels[1][lv]),
            s.ref_label(g.info_labels[lv]), s.incbin_of(g.info_labels[lv])))
    A("")
    A("## Overworld rooms")
    A("")
    A("- A bits 7-4: X where Link appears when he comes out of a cave or a level into this room (`A & $F0`). %s"
      % R("InitMode_EnterRoom"))
    A("- A bit 3: a Zora lives in this room. %s" % R("CheckZora"))
    A("- A bit 2: the sea sound plays in this room. %s" % R("BeginUpdateWorld"))
    A("- A bits 1-0: palette selector of the outer ring of the play area; B bits 1-0: palette selector of the")
    A("  inner part. %s" % RS("FillPlayAreaAttrs", "RoomPaletteSelectorToNTAttr"))
    A("- B bit 0 also picks the palette row of a rock (`$62`) or armos (`$66`) tile object. %s" % R("InitMode5Play"))
    A("- B bits 7-2: cave index, read when Link stands on a stairs tile (`$70`-`$73`) or a cave entrance tile")
    A("  (`$24`, `$88`). `1`-`9`: enter that level. `$10` and above: a cave (`$14`: the shortcut cave, mode `$C`;")
    A("  any other: mode `$B`). The code treats every value below `$10` as a level number; the listings show")
    A("  `0` as no destination, assuming such rooms hold no stairs or cave tile (unverified). %s" % R("HandleWarpOW"))
    A("- The person in a cave is object type `$6A + (cave index - $10)`. %s" % R("AssignObjSpawnPositions"))
    A("- C bits 5-0 and D bit 7 (as bit 6) form the monster list id; C bits 7-6 select one of the four")
    A("  monster counts of the level info (%s). See `objlists.md`. %s" % (R("LevelInfo_FoeCounts"),
                                                                       R("InitMode_EnterRoom")))
    A("- D bits 6-0: room layout id (7 bits in the overworld). See `room-layouts.md`. %s" % R("LayoutRoomOW"))
    A("- The layout id that %s returns is only D bits 5-0, also in the overworld. %s"
      % (N("GetUniqueRoomId"), R("GetUniqueRoomId")))
    A("- E is not per room in the overworld: it holds the wares and prices of the caves (below). %s"
      % R("InitCaveContinue"))
    A("- F bits 7-6: the quest in which the room's tile-object secret (pushed rock or gravestone, bombed rock")
    A("  wall, burned tree) works: `0` both, `1` Q1 only, `2` Q2 only (%s). %s"
      % (R("SecretQuestNumbers"), RS("IsQuestSecretMismatch", "UpdateRockOrGravestone", "UpdateRockWall",
                                     "UpdateTree")))
    A("- F bits 5-4: index into the four positions of the level info (%s) where the room's secret stairs"
      % R("LevelInfo_ShortcutOrItemPosArray"))
    A("  appear; a position byte is X in the high nibble (`byte & $F0`) and Y in the low nibble")
    A("  (`(byte & $0F) * 16`). %s" % RS("GetShortcutOrItemXYForRoom", "CheckShortcut"))
    A("- F bit 3: monsters enter from the screen edges instead of starting inside the room. %s"
      % RS("AssignObjSpawnPositions", "InitObject"))
    A("- F bits 2-0: row where Link appears when he comes out of a cave or a level: `Y = row * 16 + $4D`; after")
    A("  a cave entrance tile (`$24`) he starts 16 pixels lower and walks up to that Y. %s" % R("InitMode_EnterRoom"))
    A("- The tile object of a room is not in the level block: it comes from the layout (a square whose primary")
    A("  tile is `$E5`-`$EA`), except in the dock rooms %s, which get object type %s. %s"
      % (" and ".join(hx(r) for r in g.dock_rooms), hx(g.dock_type), RS("CheckTileObject", "SetupTileObjectOW")))
    A("")
    A("## Cave wares (overworld array E)")
    A("")
    A("- Cave `n` = cave index - `$10` (%d caves). Bytes `E[3n]`, `E[3n+1]`, `E[3n+2]` are its three wares, left"
      % g.cave_count)
    A("  to right; `E[%d+3n]` and the two bytes after it are their prices. %s" % (g.price_off, R("InitCaveContinue")))
    A("- A ware byte holds the item id in bits 5-0 (`$3F` = nothing) and two cave flags in bits 7-6. %s"
      % RS("InitCaveContinue", "DrawCaveItems"))
    A("- %s gives each cave its text selector in bits 5-0 (see `person-text.md`) and two more cave flags in"
      % R("OverworldPersonTextSelectors"))
    A("  bits 7-6. %s" % R("InitCaveContinue"))
    A("- The eight flags are packed into %s: bits 7-6 from ware 0, bits 5-4 from ware 1, bits 3-2 from ware 2,"
      % R("CaveFlags"))
    A("  bits 1-0 from the text selector byte. %s" % R("InitCaveContinue"))
    A("- Flag `$01` pick-up: Link can take a ware by touching it; without it the person only talks. %s"
      % R("UpdateCavePersonState_TalkOrShopOrDoorCharge"))
    A("- Flag `$02` pay: taking a ware costs its price. %s" % R("UpdateCavePersonState_TalkOrShopOrDoorCharge"))
    A("- Flag `$04` show-items: the wares are drawn. Flag `$08` show-prices: a rupee and the prices are drawn. %s"
      % RS("DrawCaveItems", "UpdateCavePersonState_TransferPrices"))
    A("- Flag `$10` hint: Link pays the price of the ware he touches and the person says a hint. %s"
      % RS("UpdateCavePersonState_TalkOrShopOrDoorCharge", "UpdateCavePersonState_HintOrMoneyGame"))
    A("- Flag `$20` money: the money game for person types below `$7B`; a person of type `$7B` or above")
    A("  gives Link the middle price as rupees. %s" % R("UpdateCavePersonState_HintOrMoneyGame"))
    A("- Flag `$40` heart-requirement: the ware can be taken only when %s >= `$40` (person type `$6C`) or"
      % R("HeartValues"))
    A("  >= `$B0` (other types). %s" % R("UpdateCavePersonState_TalkOrShopOrDoorCharge"))
    A("- Flag `$80` minus-sign: prices are shown with a minus sign. %s" % R("WritePricesToDynamicTransferBuf"))
    A("- The last 8 bytes of E (`E[$78]`-`E[$7F]`) are not read by any code found. (unverified) %s"
      % R("InitCaveContinue"))
    A("")
    A("## Underworld rooms")
    A("")
    A("- Doors: A bits 7-5 north, A bits 4-2 south, B bits 7-5 west, B bits 4-2 east. %s" % R("FindDoorTypeByDoorBit"))
    A("- Door types: `0` open, `1` wall, `2` and `3` false wall (Link walks through after pushing), `4` bombable")
    A("  wall, `5` and `6` key door, `7` shutter. %s" % RS("TouchDoor_JumpTable", "LayOutDoors",
                                                          "CalculateNextRoom_JumpTable"))
    A("- A bits 1-0 and B bits 1-0: palette selectors of the outer ring and the inner part. %s" % R("FillPlayAreaAttrs"))
    A("- C bits 5-0 and D bit 7 (as bit 6): monster list id; C bits 7-6: index of the monster count. %s"
      % R("InitMode_EnterRoom"))
    A("- D bit 6: the room has a push block; it is put on the block tile (`$B0`) found in the floor row at")
    A("  Y = `$90`. %s" % RS("CreateRoomObjects", "FindAndCreatePushBlockObject"))
    A("- D bits 5-0: room layout id. `$3E` and `$3F` are cellars (below). %s" % RS("GetUniqueRoomId", "LayoutUWFloor"))
    A("- E bit 7: the room is dark. %s" % RS("IsDarkRoom_Bank5", "IsDarkRoom_Bank4"))
    A("- E bits 6-5: index of the sample request (%s) that repeats in the room until the flags of the level's"
      % hexs(g.boss_sounds))
    A("  boss room have both kill-count bits set. %s" % RS("CheckBossSoundEffectUW", "BossSoundEffects"))
    A("- E bits 4-0: room item id; `$03` stands for no item. %s" % R("CreateRoomObjects"))
    A("- F bits 5-4: index into the four positions of the level info where the room item lies (same position")
    A("  bytes as above; a Triforce piece is drawn 8 pixels further left). %s"
      % RS("GetShortcutOrItemXYForRoom", "CreateRoomObjects"))
    A("- F bits 2-0: secret trigger: `1` all monsters dead opens shutters; `2` killing the monster in the first")
    A("  slot kills all; `3` shutters open when the last boss is defeated; `4` pushing the block opens shutters;")
    A("  `5` pushing the block makes stairs appear at (`$D0`, `$60`); `6` shutters open when the life-or-money")
    A("  person is gone; `7` all monsters dead opens shutters and makes the room item appear. %s"
      % RS("CheckSecretTrigger_JumpTable", "CheckUnderworldSecrets"))
    A("- With trigger `3` or `7` the room item starts hidden. %s" % R("CreateRoomObjects"))
    A("- F bits 7-6 and bit 3 are read only by overworld code (the readers test the level number or belong to")
    A("  overworld tile objects); they are 0 in every underworld room. %s"
      % RS("InitObject", "AssignObjSpawnPositions", "IsQuestSecretMismatch"))
    A("")
    A("## Cellar rooms")
    A("")
    A("- A cellar is a room of the level block with layout id `$3E` (tunnel) or `$3F` (treasure room); each")
    A("  level lists its cellars in the level info (%s). %s"
      % (R("LevelInfo_CellarRoomIdArray"), RS("LayoutCellarAndAdvanceSubmode", "InitMode11_Sub1")))
    A("- Stairs in a room lead to the first listed cellar whose A or B byte equals the room id. %s" % R("CheckWarps"))
    A("- A: the room Link returns to when he leaves the cellar at X < `$80`; B: the room when X >= `$80`. Link")
    A("  enters at the left ladder when he comes from room A, else at the right ladder. %s"
      % RS("CheckSubroom", "InitMode9_EnterCellar"))
    A("- C: where Link appears in the room he returns to: `X = C & $F0`, `Y = (C & $0F) * 16 + $0D`. %s"
      % R("CheckSubroom"))
    A("- E and F are read as in other underworld rooms (room item, its position, dark flag). %s"
      % R("CreateRoomObjects"))
    A("- A cellar always holds four %s; its own monster list is ignored. %s"
      % (obj_name(0x1B), RS("AssignObjSpawnPositions", "InitMode_EnterRoom")))
    A("")

    A("# Level info")
    A("")
    A("One block of 252 bytes per level, copied to RAM at %s. Offsets are from the start of the block."
      % R("LevelInfo_PalettesTransferBuf"))
    A("")
    A("- %s chooses the block by level number. The copy runs to `$6C7D`: 256 bytes, the block and the 4 bytes"
      % R("LevelInfoAddrs"))
    A("  that follow it in the ROM. %s" % RS("InitMode2_Sub1", "FetchLevelInfoDestInfo", "CopyBlock"))
    A("- `+%s` (36 bytes): transfer record with the level's 32 palette bytes; byte 20 is replaced with the"
      % hx(g.o_pal))
    A("  save slot's colour for Link before the transfer. %s"
      % RS("LevelInfo_PalettesTransferBuf", "PatchAndCueLevelPalettesTransferAndAdvanceSubmode"))
    A("- `+%s` (4 bytes): the four monster counts a room can select. %s"
      % (hx(g.o_foe), RS("LevelInfo_FoeCounts", "InitMode_EnterRoom")))
    A("- `+%s`: Link's Y when the level starts (his X is `$78`). %s"
      % (hx(g.o_starty), RS("LevelInfo_StartY", "InitMode3_Sub8")))
    A("- `+%s` (4 bytes): positions of secret stairs (overworld) or room items (underworld). %s"
      % (hx(g.o_pos), RS("LevelInfo_ShortcutOrItemPosArray", "GetShortcutOrItemXYForRoom")))
    A("- `+%s`: number of columns the submenu map is rotated right. %s"
      % (hx(g.o_rot), RS("LevelInfo_SubmenuMapRotation", "Submenu_WriteSheetMapRowTransferRecord")))
    A("- `+%s`: pixels added to the X of the status bar map markers. %s"
      % (hx(g.o_xoff), RS("LevelInfo_StatusBarMapXOffset", "UpdatePositionMarker")))
    A("- `+%s`: start room. %s" % (hx(g.o_start), RS("LevelInfo_StartRoomId", "InitMode3_Sub1")))
    A("- `+%s`: Triforce room, shown on the status bar map with the compass (underworld only). %s"
      % (hx(g.o_tri), RS("LevelInfo_TriforceRoomId", "UpdateTriforcePositionMarker")))
    A("- `+%s` (2 bytes, low byte first): RAM address of this level's room flags (one byte per room). %s"
      % (hx(g.o_flags), RS("LevelInfo_WorldFlagsAddr", "GetRoomFlags")))
    A("- `+%s`: level number as written into the `LEVEL-n` text; 0 in the overworld. %s"
      % (hx(g.o_num), RS("LevelInfo_LevelNumber", "InitMode3_Sub7")))
    A("- `+%s` (10 bytes, `$FF` = unused): cellar rooms (underworld); in the overworld the first four are the"
      % hx(g.o_cellar))
    A("  rooms joined by the shortcut cave: stairs 1 to 3 lead to the next 1 to 3 entries after the current")
    A("  room, wrapping after the fourth. %s" % RS("LevelInfo_CellarRoomIdArray", "CheckWarps", "CheckSubroom"))
    A("- `+%s`: boss room; the boss sound of a room stops once this room's flags say the boss is dead"
      % hx(g.o_boss))
    A("  (underworld only). %s" % RS("LevelInfo_BossRoomId", "CheckBossSoundEffectUW"))
    A("- `+%s` (16 bytes): submenu map mask, one byte per map column after the rotation; bit `$80 >> row`"
      % hx(g.o_mask))
    A("  set = the room in that row is drawn. Room (row, column) uses byte `(column + rotation) & 15`. %s"
      % RS("LevelInfo_SubmenuMapMask", "Submenu_WriteSheetMapRowTransferRecord"))
    A("- `+%s` (to `+%s`): transfer records that draw the status bar map. %s"
      % (hx(g.o_sbmap), hx(g.o_cycles - 1), RS("LevelInfo_StatusBarMapTransferBuf", "TransferBufAddrs")))
    A("- `+%s` (96 bytes): three fade series of 4 steps x 8 palette bytes (written to PPU `$3F08`), used by"
      % hx(g.o_cycles))
    A("  fade cycles `$00`, `$20` and `$40` and backwards by `$80`, `$A0` and `$C0`. %s"
      % RS("LevelInfo_PaletteCycles", "AnimateWorldFading"))
    A("- `+%s` (32 bytes): a fourth series, used by fade cycle `$60` when Link dies. %s"
      % (hx(g.o_death), RS("LevelInfo_DeathPaletteSeries", "InitMode11_Sub1", "AnimateWorldFading")))
    A("- A transfer record is: PPU address (high byte, low byte), a control byte (bit 7: write downwards, bit 6:")
    A("  repeat one byte, bits 5-0: count, 0 = 64), then the data; a high byte >= `$80` ends the list. %s"
      % RS("TransferTileBuf", "ContinueTransferTileBuf"))
    A("")

    A("# Second-quest patches")
    A("")
    A("In Q2 the level data is patched in RAM after it is copied. %s" % R("UpdateMode2Load_Full"))
    A("")
    A("- Levels 1-9 use their own Q2 level blocks (table above) and the Q1 level info blocks. %s"
      % RS("LevelBlockAddrsQ2", "LevelInfoAddrs"))
    A("- Level info, levels 1-9: bytes from offset `+%s` on are overwritten with the level's table (%s);"
      % (hx(g.o_pos), R("LevelInfoUWQ2ReplacementAddrs")))
    A("  %s holds each table's length. The loop counts from the length down to 0, so it copies one byte"
      % R("LevelInfoUWQ2ReplacementSizes"))
    A("  more than the table holds (the first byte of whatever follows it). %s" % R("UpdateMode2Load_Full"))
    A("- The tables replace the positions, map rotation and offset, start room, Triforce room, room flags")
    A("  address, level number, cellar rooms, boss room, map mask and status bar map; the palettes, the monster")
    A("  counts and the start Y stay. `levelinfo.md` lists the patched blocks. %s"
      % R("LevelInfo_ShortcutOrItemPosArray"))
    A("- The level number of a table is not always the number of the level it patches: in Q2 the number")
    A("  drawn as `LEVEL-n` differs from the internal level number (%s) for some levels (table below). All"
      % R("CurLevel"))
    A("  listings use the internal number. %s" % RS("LevelInfo_LevelNumber", "InitMode3_Sub7"))
    A("")
    A("| Level | Table | Table bytes | Bytes copied | Last info offset written | Number shown |")
    A("| --- | --- | --- | --- | --- | --- |")
    for i, (lab, tlen, size, patch) in enumerate(g.q2_info_patches):
        A("| %d | %s | %d | %d | `+%s` | %d |" % (i + 1, s.ref_label(lab), tlen, len(patch),
                                               hx(g.o_pos + len(patch) - 1), g.info(1, i + 1)[g.o_num]))
    A("")
    A("- Overworld: the level block is the Q1 block with %d bytes replaced: %d through %s and %s (offsets"
      % (len(g.q2_ow_patches), sum(1 for p in g.q2_ow_patches if p[2] == "table"),
         R("LevelBlockAttrsBQ2ReplacementOffsets"), R("LevelBlockAttrsBQ2ReplacementValues")))
    A("  from array B; an offset above `$7F` lands in array C), the rest by fixed stores. %s"
      % R("UpdateMode2Load_Full"))
    A("")
    A("| Array | Room | Q1 value | Q2 value | Written by |")
    A("| --- | --- | --- | --- | --- |")
    for off, val, how in sorted(g.q2_ow_patches):
        A("| %s | %s | %s | %s | %s |" % ("ABCDEF"[off // 128], hx(off % 128), hx(g.ow[0][off]), hx(val),
                                         "replacement table" if how == "table" else "fixed store"))
    A("")
    A("`levelblock-ow.md` lists the patched rooms decoded.")
    A("")

    A("# Monster lists (`dat/ObjLists.dat`)")
    A("")
    A("- A room's monster list id (7 bits) is either an object type or a list: `0` no monsters; `$01`-`%s`"
      % hx(g.list_base - 1))
    A("  that object type, repeated; `%s`-`$7F` list number id - `%s`. %s"
      % (hx(g.list_base), hx(g.list_base), R("InitMode_EnterRoom")))
    A("- The count comes from the level info; an id from `$32` to `%s` always makes one object. %s"
      % (hx(g.list_base - 1), R("InitMode_EnterRoom")))
    A("- %s holds the address of each list inside %s; a list is a run of object type bytes, of which the first"
      % (R("ObjListAddrs"), R("ObjLists")))
    A("  `count` are used (lists have no end marker). %s" % R("InitMode_EnterRoom"))
    A("- The count is lowered by the room's kill count before the objects are made. %s"
      % RS("ModifyObjCountByHistoryOW", "ModifyObjCountByHistoryUW"))
    A("")

    A("# Room layouts (`dat/RoomLayoutsOW.dat`, `dat/RoomLayoutsUW.dat`)")
    A("")
    A("- Overworld: a layout is 16 column descriptors (the play area is 16 columns x 11 rows of 16-pixel")
    A("  squares); layout `n` starts at byte `16 * n` of %s. %s" % (R("RoomLayoutsOW"), R("LayoutRoomOW")))
    A("- A column descriptor is a column table number (high nibble, an entry of %s) and a column number inside"
      % R("ColumnDirectoryOW"))
    A("  that table (low nibble). Column `k` starts at the `k+1`-th byte with bit 7 set. %s" % R("LayoutRoomOrCaveOW"))
    A("- Overworld square descriptor: bits 5-0 square index, bit 6 = the square fills two rows, bit 7 = first")
    A("  byte of a column. Eleven rows are read. %s" % R("LayoutRoomOrCaveOW"))
    A("- Square index < `$10`: the four tiles are entry `index` of %s (top left, bottom left, top right, bottom"
      % R("SecondarySquaresOW"))
    A("  right). Other indexes: the tile is %s`[index]` and the next three tile numbers. %s"
      % (R("PrimarySquaresOW"), R("WriteSquareOW")))
    A("- A square whose primary tile is `$E5`-`$EA` makes the room's tile object (type from %s, drawn with the"
      % R("TileObjectTypes"))
    A("  tile from %s); the last such square of a layout wins. %s" % (R("TileObjectPrimarySquaresOW"), R("CheckTileObject")))
    A("- When the room's secret was already found, a `$E7` square is drawn as stairs, a `$E6` square as a cave")
    A("  entrance and a `$EA` square as stairs. %s" % R("LayoutRoomOrCaveOW"))
    A("- Caves and cellars use the overworld routine with the layouts of %s (cave, shortcut cave, tunnel"
      % R("SubroomLayoutAddrs"))
    A("  cellar, treasure cellar); in a level, column table 0 is %s instead. %s"
      % (R("ColumnHeapUWCellar"), RS("LayoutCaveAndAvanceSubmode", "LayoutShortcutAndAdvanceSubmode",
                                     "LayoutCellarAndAdvanceSubmode", "PatchColumnDirectoryForCellar")))
    A("- Underworld: a layout is 12 column descriptors (the floor is 12 columns x 7 rows inside the walls);")
    A("  layout `n` starts at byte `12 * n` of %s; descriptors work as above with %s. %s"
      % (R("RoomLayoutsUW"), R("ColumnDirectoryUW"), R("LayoutUWFloor")))
    A("- Underworld square descriptor: bits 2-0 square index into %s, bits 6-4 number of extra rows the square"
      % R("PrimarySquaresUW"))
    A("  repeats, bit 7 = first byte of a column. A primary tile from `$70` to `$F2` stands for four")
    A("  consecutive tiles, any other for one tile used four times. %s" % RS("LayoutUWFloor", "WriteSquareUW"))
    A("- Walls and doors of an underworld room are not in the layout: walls come from %s, doors from the"
      % R("WallTileList"))
    A("  door types of the level block. %s" % RS("FillWalls", "LayOutDoors", "LayOutRoom"))
    A("")

    A("# Person text (`dat/PersonText.dat`)")
    A("")
    A("- %s holds one address per string; a text selector is the byte offset into that table (string"
      % R("PersonTextAddrs"))
    A("  number x 2). %s" % R("UpdatePersonState_Textbox"))
    A("- Each byte is one character: bits 5-0 are the character code; bits 7-6 after the character: `$80` go")
    A("  to the second line, `$40` go to the third line, `$C0` end of the text. %s" % R("UpdatePersonState_Textbox"))
    A("- One character is shown every 6 frames. Code `$25` is a blank that takes no time. %s"
      % R("UpdatePersonState_Textbox"))
    A("- A character code is the number of a background tile; the blob %s is loaded as background tiles"
      % R("CommonBackgroundPatterns"))
    A("  `$00`-`$6F` (PPU `$1000`, the table the background uses). Read from its bitmaps: `$00`-`$09` digits,")
    A("  `$0A`-`$23` letters A-Z, `$24` blank, `$28` comma, `$29` `!`, `$2A` apostrophe, `$2B` `&`, `$2C` full")
    A("  stop, `$2D` double quote, `$2E` `?`, `$2F` dash. %s"
      % RS("CommonPatternVramAddrs", "TransferCommonPatterns", "ResetPpuRegisters"))
    A("- Who says which string: %s (caves), %s and %s (hint caves), %s, %s and %s (dungeon persons). %s"
      % (R("OverworldPersonTextSelectors"), R("HintCaveTextSelectors0"), R("HintCaveTextSelectors1"),
         R("UnderworldPersonTextSelectorsA"), R("UnderworldPersonTextSelectorsB"),
         R("UnderworldPersonTextSelectorsC"), R("InitUnderworldPerson_Full")))
    A("")
    return "\n".join(L)


def ow_rows(g, quest, rooms):
    info = g.info(quest, 0)
    blk = g.ow[quest]
    arr = [blk[i * 128:(i + 1) * 128] for i in range(6)]
    rows = []
    for r in rooms:
        a, b, c, d, e, f = (x[r] for x in arr)
        cave = b >> 2
        if cave == 0:
            dest = "-"
        elif cave < 0x10:
            dest = "level %d" % cave
        elif cave == 0x14:
            dest = "shortcut cave %s" % hx(cave)
        else:
            dest = "cave %s" % hx(cave)
        lid = (c & 0x3F) | (0x40 if d & 0x80 else 0)
        count = info[g.o_foe + (c >> 6)]
        n, mons = g.monsters(lid, count)
        lay = d & 0x7F
        qs = ["-", "Q1", "Q2", "3?"][f >> 6]
        px, py = g.pos(info, (f >> 4) & 3)
        to = g.ow_tile_object(r, lay)
        if to is None:
            tos = "-"
        elif to[1] is None:
            tos = "%s %s" % (hx(to[0]), obj_name(to[0]))
        else:
            tos = "%s %s at %s,%s" % (hx(to[0]), obj_name(to[0]), hx(to[1]), hx(to[2]))
        notes = mons
        if 0x10 <= cave < 0x10 + g.cave_count:
            cv = g.caves[cave - 0x10]
            wares = [w for w in g.cave_wares(cv) if w != "-"]
            notes += "; person %s: %s" % (hx(cv["type"]), ", ".join(wares) if wares else "no wares")
        rows.append([
            hx(r), hx(a), hx(b), hx(c), hx(d), hx(e), hx(f),
            "%s,%s" % (hx(a & 0xF0), hx((f & 7) * 16 + 0x4D)),
            "Z" if a & 8 else "-", "S" if a & 4 else "-", str(a & 3), str(b & 3),
            dest, str(count), hx(lid), hx(lay), qs, "%d:%s,%s" % ((f >> 4) & 3, hx(px), hx(py)),
            "E" if f & 8 else "-", tos, notes])
    return rows


OW_HEADERS = ["room", "A", "B", "C", "D", "E", "F", "exit", "zo", "se", "po", "pi", "dest", "cnt", "list",
              "lay", "qs", "stairs", "ed", "tileobj", "monsters; cave"]


def write_ow(g):
    s = g.src
    R, RS = s.ref, s.refs
    L = []
    A = L.append
    A("# Overworld level block")
    A("")
    A("How to read this: one row per overworld room `$00`-`$7F` of %s (`%s`), first quest. Each row gives the"
      % (s.ref_label(g.block_labels[0][0]), s.incbin_of(g.block_labels[0][0])))
    A("six raw attribute bytes A-F, then the fields the game reads from them. The formats are in `README.md`.")
    A("The second-quest overworld is the same block with a few bytes patched; those rooms are listed again at")
    A("the end. Find a room with `grep '^\\$77 '`.")
    A("")
    A("Columns:")
    A("")
    A("- `exit`: X,Y where Link appears when he comes out of a cave or level (`A & $F0`, `(F & 7) * 16 + $4D`). %s"
      % R("InitMode_EnterRoom"))
    A("- `zo`: `Z` = a Zora lives here (A bit 3). `se`: `S` = sea sound (A bit 2). %s" % RS("CheckZora", "BeginUpdateWorld"))
    A("- `po`, `pi`: outer and inner palette selectors (A bits 1-0, B bits 1-0). %s" % R("FillPlayAreaAttrs"))
    A("- `dest`: where a stairs or cave entrance tile of the room leads (B bits 7-2). %s" % R("HandleWarpOW"))
    A("  `level n` is the internal level number (%s); in Q2 the number shown on screen differs for some"
      % R("CurLevel"))
    A("  levels (see `levelinfo.md`). %s" % R("LevelInfo_LevelNumber"))
    A("- `cnt`: monster count selected by C bits 7-6 from %s (%s); `list`: monster list id (C bits 5-0, D bit 7)."
      % (R("LevelInfo_FoeCounts"), hexs(g.info(0, 0)[g.o_foe:g.o_foe + 4])))
    A("  The `monsters` text applies the rules of `objlists.md`. %s" % R("InitMode_EnterRoom"))
    A("- `lay`: room layout id (D bits 6-0), see `room-layouts.md`. %s" % R("LayoutRoomOW"))
    A("- `qs`: quest in which the room's tile-object secret works (F bits 7-6): `-` both, `Q1`, `Q2`. %s"
      % R("IsQuestSecretMismatch"))
    A("- `stairs`: position index (F bits 5-4) and the X,Y it selects in %s, where secret stairs appear. %s"
      % (R("LevelInfo_ShortcutOrItemPosArray"), R("GetShortcutOrItemXYForRoom")))
    A("- `ed`: `E` = monsters enter from the screen edges (F bit 3). %s" % R("AssignObjSpawnPositions"))
    A("- `tileobj`: the tile object the room's layout creates (type, X, Y), or the dock. %s"
      % RS("CheckTileObject", "SetupTileObjectOW"))
    A("- `monsters; cave`: the monsters made on entry (before the kill count is subtracted), then for a cave")
    A("  room the person's object type and wares (`@n` = price). %s" % RS("InitMode_EnterRoom", "InitCaveContinue"))
    A("- Raw byte E is not a room attribute in the overworld (cave wares, table below). %s" % R("InitCaveContinue"))
    A("")
    A("# Rooms (first quest)")
    A("")
    L.extend(fenced(table(OW_HEADERS, ow_rows(g, 0, range(128)))))
    A("")
    A("# Caves")
    A("")
    A("One row per cave index (B bits 7-2 of a room): the person's object type, the text it says, its cave")
    A("flags and its three wares with prices, from overworld array E and %s. Flags are explained in"
      % R("OverworldPersonTextSelectors"))
    A("`README.md`. `rooms` are the first-quest rooms that lead to the cave. %s" % R("InitCaveContinue"))
    A("")
    b0 = g.ow[0][128:256]
    b1 = g.ow[1][128:256]
    rows = []
    for cv in g.caves:
        rooms0 = [hx(r) for r in range(128) if b0[r] >> 2 == cv["index"]]
        rooms1 = [hx(r) for r in range(128) if b1[r] >> 2 == cv["index"]]
        rows.append([hx(cv["index"]), hx(cv["type"]), hx(cv["selbyte"]), hx(cv["sel"]),
                     hexs(cv["items"]), hexs(cv["prices"]), g.cave_flags(cv["flags"]),
                     " | ".join(g.cave_wares(cv)), g.text_plain(cv["sel"]),
                     "Q1 " + (",".join(rooms0) or "-") + " Q2 " + (",".join(rooms1) or "-")])
    L.extend(fenced(table(["cave", "person", "sel", "text", "ware bytes", "price bytes", "flags",
                           "wares", "says", "rooms"], rows)))
    A("")
    A("# Rooms patched for the second quest")
    A("")
    A("The rooms whose bytes the second-quest patch changes, decoded with the patched values. %s"
      % R("UpdateMode2Load_Full"))
    A("")
    patched = sorted({off % 128 for off, val, how in g.q2_ow_patches})
    L.extend(fenced(table(OW_HEADERS, ow_rows(g, 1, patched))))
    A("")
    return "\n".join(L)


UW_HEADERS = ["room", "A", "B", "C", "D", "E", "F", "lvl", "N", "S", "W", "E", "po", "pi", "cnt", "list",
              "blk", "lay", "dk", "snd", "item", "pos", "secret", "monsters; cellar"]


def write_uw(g, quest, name, levels):
    s = g.src
    R, RS = s.ref, s.refs
    blk = g.blocks[name]
    arr = [blk[i * 128:(i + 1) * 128] for i in range(6)]
    lab = next(l for l in g.block_labels[quest] if l.name == name)
    owner, cellar_of = {}, {}
    for lv in levels:
        info = g.info(quest, lv)
        for r in g.members(info):
            owner.setdefault(r, lv)
        for r in g.cellars(info):
            cellar_of.setdefault(r, lv)
    L = []
    A = L.append
    A("# Underworld level block, levels %d-%d, Q%d" % (levels[0], levels[-1], quest + 1))
    A("")
    A("How to read this: one row per room `$00`-`$7F` of %s (`%s`), the level block that levels %s share in"
      % (s.ref_label(lab), s.incbin_of(lab), ", ".join(str(l) for l in levels)))
    A("the %s quest. Each row gives the six raw attribute bytes A-F, then the fields the game reads from"
      % ("first", "second")[quest])
    A("them. The formats are in `README.md`. Find a room with `grep '^\\$32 '`.")
    A("")
    A("Columns:")
    A("")
    A("- `lvl`: the level whose submenu map shows the room; `cN` = a cellar listed by level N; `-` = neither")
    A("  (the room is drawn on no level's map). Levels are internal level numbers (%s). %s"
      % (R("CurLevel"), RS("LevelInfo_SubmenuMapMask", "LevelInfo_CellarRoomIdArray")))
    A("- `N`, `S`, `W`, `E`: door type per side (A bits 7-5, A bits 4-2, B bits 7-5, B bits 4-2): `open`,")
    A("  `wall`, `false`/`false2` (walk-through wall), `bomb` (bombable wall), `key`/`key2` (key door),")
    A("  `shutter`. %s" % RS("FindDoorTypeByDoorBit", "TouchDoor_JumpTable"))
    A("- `po`, `pi`: outer and inner palette selectors (A bits 1-0, B bits 1-0). %s" % R("FillPlayAreaAttrs"))
    A("- `cnt`: monster count selected by C bits 7-6 from %s; `list`: monster list id (C bits 5-0, D bit 7)."
      % R("LevelInfo_FoeCounts"))
    A("  The `monsters` text applies the rules of `objlists.md`. %s" % R("InitMode_EnterRoom"))
    A("- `blk`: `B` = the room has a push block (D bit 6). %s" % R("CreateRoomObjects"))
    A("- `lay`: room layout id (D bits 5-0), see `room-layouts.md`; `$3E`/`$3F` = cellar. %s" % R("LayoutUWFloor"))
    A("- `dk`: `D` = dark room (E bit 7). %s" % R("IsDarkRoom_Bank5"))
    A("- `snd`: boss sound request heard in the room (E bits 6-5 through %s). %s"
      % (R("BossSoundEffects"), R("CheckBossSoundEffectUW")))
    A("- `item`: room item (E bits 4-0; `$03` = none, shown as `-`). %s" % R("CreateRoomObjects"))
    A("- `pos`: position index (F bits 5-4) and the item's X,Y from the owning level's %s (`?` when the room"
      % R("LevelInfo_ShortcutOrItemPosArray"))
    A("  has no `lvl`). %s" % R("GetShortcutOrItemXYForRoom"))
    A("- `secret`: secret trigger (F bits 2-0). %s" % R("CheckSecretTrigger_JumpTable"))
    A("- `monsters; cellar`: the monsters made on entry (before the kill count is subtracted). For a cellar:")
    A("  the rooms its left and right exits lead to (bytes A and B) and where Link appears there (byte C);")
    A("  its door columns are `-` and its monster list is not used. %s" % RS("CheckSubroom", "InitMode_EnterRoom"))
    A("")
    A("Levels of this block:")
    A("")
    rows = []
    for lv in levels:
        info = g.info(quest, lv)
        rows.append([str(lv), "LEVEL-%d" % info[g.o_num], hx(info[g.o_start]), hx(info[g.o_tri]),
                     hx(info[g.o_boss]), ",".join(hx(c) for c in g.cellars(info)) or "-",
                     hexs(info[g.o_foe:g.o_foe + 4]),
                     " ".join("%s,%s" % (hx(x), hx(y)) for x, y in (g.pos(info, i) for i in range(4)))])
    L.extend(fenced(table(["level", "shown as", "start", "Triforce", "boss", "cellars", "counts",
                           "item positions"], rows)))
    A("")
    A("# Rooms")
    A("")
    rows = []
    for r in range(128):
        a, b, c, d, e, f = (x[r] for x in arr)
        lay = d & 0x3F
        is_cellar = lay >= 0x3E
        if r in owner:
            lv, lvs = owner[r], str(owner[r])
        elif r in cellar_of:
            lv, lvs = cellar_of[r], "c%d" % cellar_of[r]
        else:
            lv, lvs = None, "-"
        info = g.info(quest, lv if lv is not None else levels[0])
        lid = (c & 0x3F) | (0x40 if d & 0x80 else 0)
        count = info[g.o_foe + (c >> 6)]
        n, mons = g.monsters(lid, count)
        item = e & 0x1F
        pi = (f >> 4) & 3
        if lv is None:
            ps = "%d:?" % pi
        else:
            px, py = g.pos(info, pi)
            ps = "%d:%s,%s" % (pi, hx(px), hx(py))
        snd = g.boss_sounds[(e >> 5) & 3]
        if is_cellar:
            doors = ["-"] * 4
            notes = "%s cellar: left exit to %s, right exit to %s, Link appears at %s,%s" % (
                "treasure" if lay & 1 else "tunnel", hx(a), hx(b), hx(c & 0xF0), hx(((c & 0x0F) << 4) | 0x0D))
            cnts, lids, pals = "-", "-", ["-", "-"]
        else:
            doors = [DOOR_NAMES[a >> 5], DOOR_NAMES[(a >> 2) & 7], DOOR_NAMES[b >> 5], DOOR_NAMES[(b >> 2) & 7]]
            notes = mons
            cnts, lids, pals = str(count), hx(lid), [str(a & 3), str(b & 3)]
        rows.append([
            hx(r), hx(a), hx(b), hx(c), hx(d), hx(e), hx(f), lvs] + doors + [
            pals[0], pals[1], cnts, lids, "B" if d & 0x40 else "-", hx(lay),
            "D" if e & 0x80 else "-", hx(snd) if snd else "-",
            "-" if item == 3 else "%s %s" % (hx(item), item_name(item)), ps,
            SECRET_NAMES[f & 7], notes])
    L.extend(fenced(table(UW_HEADERS, rows)))
    A("")
    return "\n".join(L)


def write_levelinfo(g):
    s = g.src
    R, RS, N = s.ref, s.refs, s.cur
    L = []
    A = L.append
    A("# Level info")
    A("")
    A("How to read this: one section per level (`0` = overworld) and quest, decoded from the level info blocks")
    A("(%s) with the second-quest tables applied for Q2 (%s). Every line starts with the RAM symbol of the field;"
      % (R("LevelInfoAddrs"), R("UpdateMode2Load_Full")))
    A("the field formats are in `README.md`. A position is written X,Y. The overworld block is the same in")
    A("both quests. Palettes and transfer records are given as raw bytes.")
    A("")
    A("# Summary")
    A("")
    for q in (0, 1):
        rows = []
        for lv in range(10):
            if q == 1 and lv == 0:
                continue
            info = g.info(q, lv)
            rows.append(["Q%d" % (q + 1), str(lv), "LEVEL-%d" % info[g.o_num] if lv else "-",
                         hx(info[g.o_start]), hx(info[g.o_tri]), hx(info[g.o_boss]),
                         hx(info[g.o_starty]), hx(info[g.o_flags] | info[g.o_flags + 1] << 8, 4),
                         ",".join(hx(c) for c in g.cellars(info)) or "-"])
        L.extend(fenced(table(["quest", "level", "shown as", "start", "Triforce", "boss", "startY", "flags",
                               "cellars"], rows)))
        A("")
    A("- `level`: the internal level number (%s), which selects the block; `shown as`: the number drawn on"
      % R("CurLevel"))
    A("  screen (%s). They differ for some levels in Q2. %s" % (R("LevelInfo_LevelNumber"), R("InitMode3_Sub7")))
    A("- `start`, `Triforce`, `boss`: room ids (%s). In the overworld only the start room is read." %
      RS("LevelInfo_StartRoomId", "LevelInfo_TriforceRoomId", "LevelInfo_BossRoomId"))
    A("- `cellars`: cellar rooms of the level; for the overworld, the four rooms the shortcut cave joins. %s"
      % R("LevelInfo_CellarRoomIdArray"))
    A("")
    for q in (0, 1):
        for lv in range(10):
            if q == 1 and lv == 0:
                continue
            info = g.info(q, lv)
            if lv == 0:
                A("# Overworld (level 0), Q1")
            else:
                A("# Level %d, Q%d (shown as LEVEL-%d)" % (lv, q + 1, info[g.o_num]))
            A("")
            if q == 0:
                A("Block %s (`%s`)." % (s.ref_label(g.info_labels[lv]), s.incbin_of(g.info_labels[lv])))
            else:
                A("Block %s (`%s`) patched with %s." % (s.ref_label(g.info_labels[lv]), s.incbin_of(g.info_labels[lv]),
                                                       s.ref_label(g.q2_info_patches[lv - 1][0])))
            A("")
            rows = []
            K = lambda up, val: rows.append([N(up), val])
            K("LevelInfo_StartRoomId", hx(info[g.o_start]))
            K("LevelInfo_TriforceRoomId", hx(info[g.o_tri]))
            K("LevelInfo_BossRoomId", hx(info[g.o_boss]))
            K("LevelInfo_LevelNumber", hx(info[g.o_num]))
            K("LevelInfo_StartY", hx(info[g.o_starty]))
            K("LevelInfo_FoeCounts", hexs(info[g.o_foe:g.o_foe + 4]))
            K("LevelInfo_ShortcutOrItemPosArray", hexs(info[g.o_pos:g.o_pos + 4]) + "  = " + "  ".join(
                "%d:%s,%s" % (i, hx(g.pos(info, i)[0]), hx(g.pos(info, i)[1])) for i in range(4)))
            K("LevelInfo_CellarRoomIdArray", hexs(info[g.o_cellar:g.o_cellar + 10]))
            K("LevelInfo_WorldFlagsAddr", hx(info[g.o_flags] | info[g.o_flags + 1] << 8, 4))
            K("LevelInfo_SubmenuMapRotation", hx(info[g.o_rot]))
            K("LevelInfo_StatusBarMapXOffset", hx(info[g.o_xoff]))
            K("LevelInfo_SubmenuMapMask", hexs(info[g.o_mask:g.o_mask + 16]))
            recs, end = decode_transfer(info[g.o_sbmap:g.o_cycles])
            K("LevelInfo_StatusBarMapTransferBuf", hexs(info[g.o_sbmap:g.o_sbmap + end + 1]))
            for addr, vert, rep, count, data in recs:
                rows.append(["  record", "%s %s%d: %s" % (hx(addr, 4), "down " if vert else "",
                                                         count, hexs(data) + (" repeated" if rep else ""))])
            pal = info[g.o_pal:g.o_foe]
            K("LevelInfo_PalettesTransferBuf", hexs(pal))
            recs, end = decode_transfer(pal)
            for addr, vert, rep, count, data in recs:
                for i in range(0, len(data), 4):
                    rows.append(["  palette row %d" % (i // 4), hexs(data[i:i + 4])])
            cyc = info[g.o_cycles:g.o_death]
            for i in range(0, len(cyc), 8):
                rows.append(["%s cycle %s step %d" % (N("LevelInfo_PaletteCycles"), hx(i // 32 * 0x20), i // 8 % 4),
                             hexs(cyc[i:i + 8])])
            dth = info[g.o_death:]
            for i in range(0, 32, 8):
                rows.append(["%s step %d" % (N("LevelInfo_DeathPaletteSeries"), i // 8), hexs(dth[i:i + 8])])
            L.extend(fenced(table(["field", "value"], rows)))
            A("")
            if lv:
                mem = g.members(info)
                A("Rooms on the level's submenu map (`#`), by room id; `S` start, `T` Triforce, `B` boss room")
                A("(marked also when the map does not show them):")
                A("")
                grid = ["     " + " ".join("%X" % c for c in range(16))]
                for r in range(8):
                    cells = []
                    for c in range(16):
                        rm = r * 16 + c
                        ch = "#" if rm in mem else "."
                        if rm == info[g.o_start]:
                            ch = "S"
                        elif rm == info[g.o_boss]:
                            ch = "B"
                        elif rm == info[g.o_tri]:
                            ch = "T"
                        cells.append(ch)
                    grid.append("%s  " % hx(r * 16) + " ".join(cells))
                L.extend(fenced(grid))
                A("")
    return "\n".join(L)


def write_objlists(g):
    s = g.src
    R, RS = s.ref, s.refs
    L = []
    A = L.append
    A("# Monster lists")
    A("")
    A("How to read this: a room's level block bytes give a 7-bit monster list id and a count index. This file")
    A("lists every list reachable through %s (blob %s, `%s`) and the object type names used in all listings."
      % (R("ObjListAddrs"), R("ObjLists"), s.incbin_of(s.label("ObjLists"))))
    A("")
    A("- Monster list id = level block C bits 5-0, plus `$40` when D bit 7 is set. %s" % R("InitMode_EnterRoom"))
    A("- Count = entry (C bits 7-6) of the level info's four monster counts (%s); `levelinfo.md` has them."
      % R("LevelInfo_FoeCounts"))
    A("- Id `0`: no monsters. Id `$01`-`%s`: `count` objects of the object type equal to the id; for an id from"
      % hx(g.list_base - 1))
    A("  `$32` up the count is forced to 1 (bosses and other single objects). %s" % R("InitMode_EnterRoom"))
    A("- Id `%s`-`$7F`: list number id - `%s` below; the first `count` bytes of the list are the object types of"
      % (hx(g.list_base), hx(g.list_base)))
    A("  object slots `$01` up. A list has no end marker: a count above its stored length runs into the next")
    A("  list. %s" % R("InitMode_EnterRoom"))
    A("- Before the objects are made, the count is lowered by the room's kill count, and a cellar or cave")
    A("  replaces the list. %s" % RS("ModifyObjCountByHistoryOW", "ModifyObjCountByHistoryUW", "AssignObjSpawnPositions"))
    A("")
    A("# Lists")
    A("")
    A("`offset` is the list's offset in the blob; `len` is the distance to the next list. %s" % R("ObjLists"))
    A("")
    rows = []
    offs = g.objlist_offs
    order = sorted(set(offs)) + [len(g.objlists)]
    for i, off in enumerate(offs):
        end = order[order.index(off) + 1]
        types = g.objlists[off:end]
        rows.append([hx(g.list_base + i), str(off), str(end - off), hexs(types),
                     ", ".join("%s %s" % (hx(t), obj_name(t)) for t in types)])
    L.extend(fenced(table(["id", "offset", "len", "type bytes", "types"], rows)))
    A("")
    A("# Object type names")
    A("")
    A("Names used for object type ids in these listings (types that share a name differ by id). Type `$6A`")
    A("and above are cave persons. %s" % R("InitObject"))
    A("")
    rows = [[hx(t), OBJ_NAMES[t]] for t in sorted(OBJ_NAMES)]
    L.extend(fenced(table(["type", "name"], rows)))
    A("")
    return "\n".join(L)


def grid_lines(grid, width):
    out = ["     " + " ".join("c%X" % c for c in range(width))]
    for r, row in enumerate(grid):
        out.append("r%-2d  " % r + " ".join("??" if v is None else "%02X" % v for v in row))
    return out


def write_layouts(g):
    s = g.src
    R, RS = s.ref, s.refs
    L = []
    A = L.append
    A("# Room layouts")
    A("")
    A("How to read this: every room layout as stored (its column descriptors) and decoded into a grid of")
    A("square indexes (hex, without `$`; column `c0` is the left edge, row `r0` the top). The square tables")
    A("at the end give the tiles of each square index. The formats are in `README.md`. Find a layout with")
    A("`grep '^OW \\$22 '` or `grep '^UW \\$0A '`; its grid follows the line.")
    A("")
    A("- Overworld layouts: 16 columns x 11 rows. %s" % RS("LayoutRoomOW", "LayoutRoomOrCaveOW"))
    A("- Underworld layouts: 12 columns x 7 rows of floor inside the walls. %s" % R("LayoutUWFloor"))
    A("- A column descriptor is `table number, column number`: high and low nibble. %s"
      % RS("ColumnDirectoryOW", "ColumnDirectoryUW"))
    A("- `rooms` lists the rooms whose level block byte D selects the layout (Q2 only where it differs). %s"
      % R("GetUniqueRoomId"))
    A("")
    A("# Overworld layouts")
    A("")
    d0 = g.ow[0][3 * 128:4 * 128]
    d1 = g.ow[1][3 * 128:4 * 128]
    tabs = g.ow_tables()
    total = len(g.ow_layout_bytes) // 16
    extra = {lid: lab for lid, lab in g.ow_extra_layouts}
    A("Layouts `$00`-`%s` are %s; ids %s fall on the tables that follow it in the ROM: %s."
      % (hx(g.ow_layout_count - 1), R("RoomLayoutsOW"),
         ", ".join("`%s`" % hx(lid) for lid, lab in g.ow_extra_layouts),
         ", ".join(s.ref_label(lab) for lid, lab in g.ow_extra_layouts)))
    A("")
    for lid in range(total):
        desc = g.ow_layout(lid)
        r0 = [hx(r) for r in range(128) if d0[r] & 0x7F == lid]
        r1 = [hx(r) for r in range(128) if d1[r] & 0x7F == lid]
        rooms = ",".join(r0) or "-"
        if r1 != r0:
            rooms += " (Q2: %s)" % (",".join(r1) or "-")
        L.append("```")
        A("OW %s  columns: %s  rooms: %s%s" % (hx(lid), hexs(desc), rooms,
                                             "  (%s)" % extra[lid].name if lid in extra else ""))
        L.extend(grid_lines(g.ow_grid(desc, tabs), 16))
        L.append("```")
    A("")
    A("# Cave and cellar layouts")
    A("")
    A("Laid out by the overworld routine from %s: entry 0 the cave, 1 the shortcut cave, 2 the tunnel cellar"
      % R("SubroomLayoutAddrs"))
    A("(room layout id `$3E`), 3 the treasure cellar (`$3F`). Cellars read column table 0 from %s. %s"
      % (R("ColumnHeapUWCellar"), RS("LayoutCaveAndAvanceSubmode", "LayoutShortcutAndAdvanceSubmode",
                                     "LayoutCellarAndAdvanceSubmode")))
    A("")
    kinds = ["cave", "shortcut cave", "tunnel cellar", "treasure cellar"]
    for i, lab in enumerate(g.sub_layouts):
        desc = s.data_of(lab)[:16]
        L.append("```")
        A("SUB %d  %s  columns: %s  (%s)" % (i, kinds[i] if i < len(kinds) else "?", hexs(desc), lab.name))
        L.extend(grid_lines(g.ow_grid(desc, g.ow_tables(cellar=i >= 2)), 16))
        L.append("```")
    A("")
    A("# Underworld layouts")
    A("")
    A("Layouts `$00`-%s of %s. Rooms are listed per level block: `1-6` and `7-9` are the level groups."
      % (hx(g.uw_layout_count - 1), R("RoomLayoutsUW")))
    A("")
    uses = []
    for q in (0, 1):
        for name in dict.fromkeys(l.name for l in g.block_labels[q][1:]):
            lv = g.levels_of_block(q, name)
            d = g.blocks[name][3 * 128:4 * 128]
            uses.append(("%d-%d Q%d" % (lv[0], lv[-1], q + 1), d))
    for lid in range(g.uw_layout_count):
        desc = g.uw_layout_bytes[lid * 12:lid * 12 + 12]
        L.append("```")
        A("UW %s  columns: %s" % (hx(lid), hexs(desc)))
        for tag, d in uses:
            A("  rooms %s: %s" % (tag, ",".join(hx(r) for r in range(128) if d[r] & 0x3F == lid) or "-"))
        L.extend(grid_lines(g.uw_grid(desc), 12))
        L.append("```")
    A("")
    A("# Overworld squares")
    A("")
    A("Square index to tiles. Index < `$10`: the four tiles of %s (top left, bottom left, top right, bottom"
      % R("SecondarySquaresOW"))
    A("right). Other indexes: primary tile from %s and the three tile numbers after it, in the same order."
      % R("PrimarySquaresOW"))
    A("`object` = the tile object type the square creates and the primary tile it is drawn with. %s"
      % RS("WriteSquareOW", "CheckTileObject"))
    A("")
    rows = []
    for i in range(g.sq_primary_len):
        prim = g.sq_primary[i]
        draw = prim
        note = "-"
        if 0xE5 <= prim <= 0xEA:
            t = g.tileobj_types[prim - 0xE5]
            draw = g.tileobj_squares[prim - 0xE5]
            note = "%s %s, drawn with %s" % (hx(t), obj_name(t), hx(draw))
        if i < 0x10:
            tiles = g.sq_secondary[i * 4:i * 4 + 4]
        else:
            tiles = bytes((draw + k) & 0xFF for k in range(4))
        rows.append(["%02X" % i, hx(prim), hexs(tiles), note])
    L.extend(fenced(table(["sq", "primary", "tiles", "object"], rows)))
    A("")
    A("# Underworld squares")
    A("")
    A("Square index to primary tile (%s). A primary from `$70` to `$F2` is the first of four consecutive"
      % R("PrimarySquaresUW"))
    A("tiles; any other fills the square with one tile. Tile `$B0` is the block a push block object is placed")
    A("on; tiles `$70`-`$73` are stairs. %s" % RS("WriteSquareUW", "FindAndCreatePushBlockObject", "CheckWarps"))
    A("")
    rows = []
    for i, prim in enumerate(g.sq_uw):
        if 0x70 <= prim < 0xF3:
            tiles = bytes((prim + k) & 0xFF for k in range(4))
        else:
            tiles = bytes([prim] * 4)
        rows.append(["%02X" % i, hx(prim), hexs(tiles)])
    L.extend(fenced(table(["sq", "primary", "tiles"], rows)))
    A("")
    A("# Column tables")
    A("")
    A("The square descriptors of every column table, one line per column (`table.column`), as stored.")
    A("Overworld descriptor: bit 7 column start, bit 6 two rows, bits 5-0 square index. Underworld descriptor:")
    A("bit 7 column start, bits 6-4 extra rows, bits 2-0 square index. A column that holds fewer rows than the")
    A("room needs (11 or 7) continues with the bytes of the next line. %s" % RS("LayoutRoomOrCaveOW", "LayoutUWFloor"))
    A("")

    def columns(tag, labels):
        rows = []
        for t, lab in enumerate(labels):
            data = s.data_of(lab)
            starts = [i for i, b in enumerate(data) if b & 0x80]
            for k, st in enumerate(starts):
                en = starts[k + 1] if k + 1 < len(starts) else len(data)
                rows.append(["%s %X.%X" % (tag, t, k), lab.name if k == 0 else "", hexs(data[st:en])])
        return rows

    L.extend(fenced(table(["column", "table", "square descriptors"],
                          columns("OWCOL", g.ow_heap_labels)
                          + columns("UWCOL", g.uw_heap_labels)
                          + columns("CELLARCOL", [s.label("ColumnHeapUWCellar")]))))
    A("")
    return "\n".join(L)


def write_text(g):
    s = g.src
    R, RS = s.ref, s.refs
    # who uses which selector
    used = {}
    add = lambda sel, who: used.setdefault(sel, []).append(who)
    for cv in g.caves:
        add(cv["sel"], "cave person %s" % hx(cv["type"]))
    for k, up in enumerate(("HintCaveTextSelectors0", "HintCaveTextSelectors1")):
        for i, sel in enumerate(s.data(up)):
            add(sel, "hint cave, %s, ware %d" % ("person $75" if k == 0 else "other persons", i))
    jt = s.addrs("InitUnderworldPerson_Full_JumpTable")
    for letter in "ABC":
        lab = s.label("InitUnderworldPerson" + letter)
        levels = [str(i) for i, (m, p, o) in enumerate(jt) if p == lab.pos]
        for i, sel in enumerate(s.data("UnderworldPersonTextSelectors" + letter)):
            t = 0x4B + i
            if OBJ_NAMES.get(t) == "dungeon person":
                add(sel, "dungeon person %s in level %s" % (hx(t), ",".join(levels)))
            else:
                add(sel, "(table entry of type %s in level %s, not used: that type sets its own text)"
                    % (hx(t), ",".join(levels)))
    for up, who in (("InitUnderworldPersonLifeOrMoney_Full", "life-or-money person $51"),
                    ("InitGrumble_Full", "Grumble $36")):
        for sel in s.imm_before_store(up, "PersonTextSelector")[:1]:
            add(sel, who)
    L = []
    A = L.append
    A("# Person text")
    A("")
    A("How to read this: one row per text string of %s (`%s`), in the order of %s. `sel` is the text"
      % (R("PersonText"), s.incbin_of(s.label("PersonText")), R("PersonTextAddrs")))
    A("selector (the value stored in %s); `offset` and `len` locate the string in the blob. `text` is"
      % R("PersonTextSelector"))
    A("what the textbox shows, lines separated by ` / `, with the blanks at both ends of each line removed;")
    A("`indent` is the number of leading blank cells of each line (`f` = fast blanks, code `$25`, that take no")
    A("time; `s` = ordinary blanks, code `$24`); `line` is the textbox line each part goes to. `said by` comes")
    A("from the selector tables. The encoding is in `README.md`. %s" % R("UpdatePersonState_Textbox"))
    A("")
    A("- Cave persons: %s, indexed by person type - `$6A`. %s" % (R("OverworldPersonTextSelectors"), R("InitCaveContinue")))
    A("- Hint caves, after Link pays: %s for person type `$75`, %s for the others, indexed by the ware touched. %s"
      % (R("HintCaveTextSelectors0"), R("HintCaveTextSelectors1"), R("UpdateCavePersonState_HintOrMoneyGame")))
    A("- Dungeon persons: selector tables by level through %s, indexed by object type - `$4B`. %s"
      % (R("InitUnderworldPerson_Full_JumpTable"), RS("UnderworldPersonTextSelectorsA", "UnderworldPersonTextSelectorsB",
                                                      "UnderworldPersonTextSelectorsC")))
    A("- The life-or-money person and Grumble set their selector directly. %s"
      % RS("InitUnderworldPersonLifeOrMoney_Full", "InitGrumble_Full"))
    A("")
    rows = []
    for i, t in enumerate(g.texts):
        parts, indents, lines = [], [], []
        line = 1
        for codes, mark in t["lines"]:
            lead = 0
            while lead < len(codes) and codes[lead] in (0x24, 0x25):
                lead += 1
            fast = sum(1 for c in codes[:lead] if c == 0x25)
            indents.append(("%df" % fast if fast else "") + ("%ds" % (lead - fast) if lead - fast else "") or "0")
            parts.append("".join(CHARS.get(c, "?") for c in codes).strip())
            lines.append(str(line))
            line = {0x80: 2, 0x40: 3, 0xC0: 1}[mark]
        rows.append([hx(t["sel"]), str(i), str(t["off"]), str(t["len"]), ",".join(lines), ",".join(indents),
                     " / ".join(parts), "; ".join(used.get(t["sel"], [])) or "-"])
    L.extend(fenced(table(["sel", "n", "offset", "len", "line", "indent", "text", "said by"], rows, sep="  | ")))
    A("")
    return "\n".join(L)


# ---- main -----------------------------------------------------------------------------------------

def load_prg(path):
    with open(path, "rb") as f:
        rom = f.read()
    prg = rom[16:16 + PRG_SIZE]
    if len(rom) < 16 + PRG_SIZE or hashlib.sha1(rom[16:]).hexdigest() != PRG0_SHA1:
        return None
    return prg


def build(root, prg, outdir):
    """{file name: text} of every listing."""
    g = Game(Source(root, prg, outdir))
    out = {"README.md": write_readme(g), "levelblock-ow.md": write_ow(g), "levelinfo.md": write_levelinfo(g),
           "objlists.md": write_objlists(g), "room-layouts.md": write_layouts(g), "person-text.md": write_text(g)}
    for q in (0, 1):
        for name in dict.fromkeys(l.name for l in g.block_labels[q][1:]):
            levels = g.levels_of_block(q, name)
            out["levelblock-uw-%dto%d-q%d.md" % (levels[0], levels[-1], q + 1)] = write_uw(g, q, name, levels)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rom", required=True, help="ROM image (16-byte header followed by the PRG0 image)")
    ap.add_argument("--out", help="output directory (default: data/ beside tools/)")
    ap.add_argument("--root", default=asm.ROOT, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    root = os.path.abspath(args.root)
    outdir = os.path.abspath(args.out) if args.out else os.path.join(root, "data")
    try:
        prg = load_prg(args.rom)
    except OSError as e:
        print("romdata: cannot read the ROM: %s" % e, file=sys.stderr)
        return 2
    if prg is None:
        print("romdata: the ROM is not PRG0 (SHA-1 of the image without its 16-byte header must be %s)"
              % PRG0_SHA1, file=sys.stderr)
        return 2
    try:
        out = build(root, prg, outdir)
    except DataError as e:
        print("romdata: %s" % e, file=sys.stderr)
        return 1
    for name in sorted(out):
        asm.write_text(os.path.join(outdir, name), out[name].rstrip("\n") + "\n")
    print("romdata: wrote %d files to %s" % (len(out), os.path.relpath(outdir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
