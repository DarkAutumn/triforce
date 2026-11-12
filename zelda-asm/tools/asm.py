#!/usr/bin/env python3
"""asm.py - keeps the annotated disassembly honest.

The source under src/ is never assembled here.  Instead every statement (instruction or data
directive) is reduced to a canonical form in which each identifier is replaced by what it stands
for: a RAM symbol by its value, a label by (bank, index of the statement it points at).  The
canonical form of the original import is frozen under index/upstream/.  Splitting files, renaming
labels and editing comments must leave the canonical form identical; `check` proves it.

Subcommands: check, gen, rename, where, split, stale-names, canon-hash, freeze-baseline.
Run `asm.py <subcommand> -h` for options.  Standard library only.
"""

import argparse
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MNEMONICS = set("""ADC AND ASL BCC BCS BEQ BIT BMI BNE BPL BRK BVC BVS CLC CLD CLI CLV CMP CPX CPY DEC DEX
DEY EOR INC INX INY JMP JSR LDA LDX LDY LSR NOP ORA PHA PHP PLA PLP ROL ROR RTI RTS SBC SEC SED SEI STA
STX STY TAX TAY TSX TXA TXS TYA""".split())
DATA_DIRECTIVES = {".BYTE", ".ADDR", ".DBYT", ".LOBYTES", ".HIBYTES", ".INCBIN"}
NUM_MODULES = 8

IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
RE_SYMDEF = re.compile(r"^(%s)\s*(:=|=)\s*(.+)$" % IDENT)
RE_GLOBAL = re.compile(r"^(%s):(.*)$" % IDENT)
RE_CHEAP = re.compile(r"^(@%s):(.*)$" % IDENT)
RE_UNNAMED = re.compile(r"^:(?![+\-])(.*)$")
RE_STMT = re.compile(r"^(\.?[A-Za-z]+)(?:\s+(.*))?$")
RE_TOKEN = re.compile(
    r"""\s*(?:(?P<str>"[^"]*")|(?P<hex>\$[0-9A-Fa-f]+)|(?P<bin>%[01]+)|(?P<dec>[0-9]+)"""
    r"""|(?P<cheap>@[A-Za-z_][A-Za-z0-9_]*)|(?P<apre>a:)|(?P<id>[A-Za-z_][A-Za-z0-9_]*)"""
    r"""|(?P<unnamed>:[+\-]+)|(?P<punct>[#<>(),+\-*/]))""")

RE_NAME_GLOBAL = re.compile(r"^[A-Z][A-Za-z0-9]*(_[A-Za-z0-9]+)+$")
RE_NAME_SYMBOL = re.compile(r"^[A-Z][A-Za-z0-9]*(_[A-Z][A-Za-z0-9]*)?$")
RE_NAME_CHEAP = re.compile(r"^@[A-Z][A-Za-z0-9]*$")
RE_PLAIN_WORD = re.compile(r"^[A-Z]?[a-z]+$")

# Outside references. `TRIFORCE:` is the old comment tag; the game's own word "Triforce" (any case) is
# allowed, the other project's name as a path or module is not.
RE_SELF = re.compile(r"TRIFORCE:|triforce2|triforce/|triforce\._|autosplit|[Pp]ython|frame_skip|data\.json"
                     r"|\.py\b|\.rs\b")
RE_MDLINK = re.compile(r"\[`(%s(?:@%s)?)`\]\(([^)#\s]*)#L(\d+)\)" % (IDENT, IDENT))
RE_ASMREF = re.compile(r"(?<![A-Za-z0-9_\-/.])asm:(%s(?:@%s)?)" % (IDENT, IDENT))
RE_RULE = re.compile(r"^;\s*[=\-]{8,}\s*$")
RE_HRULE = re.compile(r"^;\s*={8,}\s*$")
DOC_FILES = ("enemies.md", "items.md", "mechanics.md")
HEADER_FIELDS = ("FILE", "UPSTREAM", "SUBJECTS", "TOPIC", "SEE ALSO", "KEY FACTS", "CONTENTS")
BAD_SUMMARY = re.compile(r"^(Params|Returns|IN|OUT|NOTE|TODO|Unknown block)\b")


class AsmError(Exception):
    pass


def strip_comment(line):
    """Split a source line into (code, comment); the comment starts at the first ';' outside quotes."""
    inq = False
    for i, c in enumerate(line):
        if c == '"':
            inq = not inq
        elif c == ";" and not inq:
            return line[:i], line[i:]
    return line, ""


class Line:
    __slots__ = ("file", "no", "text", "code", "comment")

    def __init__(self, file, no, text):
        self.file, self.no, self.text = file, no, text
        self.code, self.comment = strip_comment(text)


class Stmt:
    __slots__ = ("idx", "file", "no", "head", "operand", "li", "canon")


class Label:
    __slots__ = ("name", "kind", "pos", "file", "no", "li", "scope", "mod")

    def __init__(self, name, kind, pos, file, no, li, scope, mod):
        self.name, self.kind, self.pos, self.file, self.no = name, kind, pos, file, no
        self.li, self.scope, self.mod = li, scope, mod


class Sym:
    __slots__ = ("name", "value", "file", "no", "comment")

    def __init__(self, name, value, file, no, comment):
        self.name, self.value, self.file, self.no, self.comment = name, value, file, no, comment


class Module:
    def __init__(self, mid, path):
        self.id = mid
        self.path = path            # relative to src/
        self.lines = []             # expanded Line records
        self.stmts = []
        self.labels = []            # global and cheap, in definition order
        self.globals = {}           # name -> Label
        self.cheap = defaultdict(dict)   # scope owner li (or -1) -> {name: Label}
        self.unnamed = []           # (li, pos)
        self.symbols = {}           # name -> Sym (visible to this module)
        self.imports = {}           # name -> (file, no)
        self.exports = {}
        self.files = []             # every file expanded into this module, in order
        self.includes = []          # (including file, no, included path)


class Program:
    """All eight bank modules parsed from <root>/src (or another src directory)."""

    def __init__(self, root=None, src=None, overlay=None):
        self.root = root or ROOT
        self.src = src or os.path.join(self.root, "src")
        self.overlay = overlay or {}
        self.errors = []            # (file relative to root, line, message)
        self.mods = []
        self.all_symbols = {}       # name -> Sym, program wide
        for n in range(NUM_MODULES):
            for cand in ("bank%d.asm" % n, "Z_%02d.asm" % n):
                if self._exists(cand):
                    self.mods.append(self._parse_module(n, cand))
                    break
            else:
                raise AsmError("no source file for bank %d in %s" % (n, self.src))
        self.split_done = self.mods[0].path.startswith("bank")
        self._canonicalize()

    # ---- file access ------------------------------------------------------------------------
    def _abs(self, rel):
        return os.path.join(self.src, rel)

    def _exists(self, rel):
        return self._abs(rel) in self.overlay or os.path.exists(self._abs(rel))

    def read(self, rel):
        p = self._abs(rel)
        if p in self.overlay:
            return self.overlay[p]
        with open(p, encoding="utf-8") as f:
            return f.read()

    def rel(self, srcrel):
        """Path of a src-relative file relative to the root (for messages and indexes)."""
        return os.path.relpath(self._abs(srcrel), self.root)

    def err(self, file, no, msg):
        self.errors.append((self.rel(file), no, msg))

    # ---- parsing ----------------------------------------------------------------------------
    def _parse_module(self, mid, path):
        mod = Module(mid, path)
        scope = [None]              # current global Label (cheap label scope owner)
        pending = []                # labels waiting for the next statement index

        def add_label(name, kind, ln, li):
            lab = Label(name, kind, len(mod.stmts), ln.file, ln.no, li, scope[0], mod)
            mod.labels.append(lab)
            if kind == "global":
                if name in mod.globals:
                    self.err(ln.file, ln.no, "duplicate global label %s (first at %s:%d)" % (
                        name, self.rel(mod.globals[name].file), mod.globals[name].no))
                else:
                    mod.globals[name] = lab
                scope[0] = lab
            else:
                key = scope[0].li if scope[0] else -1
                if name in mod.cheap[key]:
                    self.err(ln.file, ln.no, "duplicate cheap label %s in scope %s" % (
                        name, scope[0].name if scope[0] else "(none)"))
                else:
                    mod.cheap[key][name] = lab
            return lab

        def expand(file, stack):
            if file in stack:
                raise AsmError("recursive .INCLUDE of %s" % file)
            mod.files.append(file)
            text = self.read(file)
            for no, raw in enumerate(text.split("\n"), 1):
                ln = Line(file, no, raw)
                li = len(mod.lines)
                mod.lines.append(ln)
                s = ln.code.strip()
                if not s:
                    continue
                m = RE_SYMDEF.match(s)
                if m:
                    self._define_symbol(mod, m.group(1), m.group(3).strip(), ln)
                    continue
                m = RE_GLOBAL.match(s)
                if m:
                    add_label(m.group(1), "global", ln, li)
                    s = m.group(2).strip()
                else:
                    m = RE_CHEAP.match(s)
                    if m:
                        add_label(m.group(1), "cheap", ln, li)
                        s = m.group(2).strip()
                    else:
                        m = RE_UNNAMED.match(s)
                        if m:
                            mod.unnamed.append((li, len(mod.stmts)))
                            s = m.group(1).strip()
                if not s:
                    continue
                m = RE_STMT.match(s)
                if not m:
                    self.err(file, no, "cannot parse: %s" % s)
                    continue
                head, operand = m.group(1).upper(), (m.group(2) or "").strip()
                if head == ".INCLUDE":
                    inc = operand.strip('"')
                    mod.includes.append((file, no, inc))
                    cand = os.path.normpath(os.path.join(os.path.dirname(file), inc))
                    if not self._exists(cand):
                        cand = inc
                    if not self._exists(cand):
                        self.err(file, no, ".INCLUDE file not found: %s" % inc)
                        continue
                    expand(cand, stack + [file])
                elif head in (".IMPORT", ".EXPORT"):
                    table = mod.imports if head == ".IMPORT" else mod.exports
                    for name in (x.strip() for x in operand.split(",")):
                        if not re.fullmatch(IDENT, name):
                            self.err(file, no, "bad %s operand: %s" % (head, operand))
                        else:
                            table.setdefault(name, (file, no))
                elif head == ".SEGMENT" or head in DATA_DIRECTIVES or head in MNEMONICS:
                    st = Stmt()
                    st.idx, st.file, st.no, st.head, st.operand, st.li = (
                        len(mod.stmts), file, no, head, operand, li)
                    st.canon = None
                    mod.stmts.append(st)
                else:
                    self.err(file, no, "unsupported directive or mnemonic: %s" % head)

        expand(path, [])
        return mod

    def _define_symbol(self, mod, name, expr, ln):
        try:
            value = self._eval_const(mod, expr)
        except AsmError as e:
            self.err(ln.file, ln.no, "symbol %s: %s" % (name, e))
            return
        comment = ln.comment[1:].strip() if ln.comment else ""
        sym = Sym(name, value, ln.file, ln.no, comment)
        if name in mod.symbols:
            self.err(ln.file, ln.no, "duplicate symbol %s" % name)
        mod.symbols[name] = sym
        prev = self.all_symbols.get(name)
        if prev and (prev.file, prev.no) != (ln.file, ln.no):
            self.err(ln.file, ln.no, "symbol %s also defined at %s:%d" % (name, self.rel(prev.file), prev.no))
        self.all_symbols[name] = sym

    def _eval_const(self, mod, expr):
        total, sign, pos, expect_value = 0, 1, 0, True
        expr = expr.strip()
        while pos < len(expr):
            m = RE_TOKEN.match(expr, pos)
            if not m:
                raise AsmError("cannot parse expression %r" % expr)
            pos = m.end()
            kind, text = m.lastgroup, m.group(m.lastgroup)
            if expect_value:
                if kind == "hex":
                    v = int(text[1:], 16)
                elif kind == "bin":
                    v = int(text[1:], 2)
                elif kind == "dec":
                    v = int(text)
                elif kind == "id" and text in mod.symbols:
                    v = mod.symbols[text].value
                else:
                    raise AsmError("unsupported term %r in %r" % (text, expr))
                total += sign * v
                expect_value = False
            else:
                if text not in "+-":
                    raise AsmError("unsupported operator %r in %r" % (text, expr))
                sign = 1 if text == "+" else -1
                expect_value = True
        if expect_value:
            raise AsmError("incomplete expression %r" % expr)
        return total

    # ---- canonical form -----------------------------------------------------------------------
    def _canonicalize(self):
        exporters = defaultdict(list)
        for mod in self.mods:
            for name, (file, no) in mod.exports.items():
                if name not in mod.globals:
                    self.err(file, no, ".EXPORT of %s, which this bank does not define" % name)
                else:
                    exporters[name].append(mod)
        for name, mods in exporters.items():
            if len(mods) > 1:
                file, no = mods[1].exports[name]
                self.err(file, no, "%s is exported by more than one bank" % name)
        self.exporters = exporters
        for mod in self.mods:
            for name, (file, no) in mod.imports.items():
                if name.startswith("__") and name.endswith("__"):
                    continue
                if name not in exporters:
                    self.err(file, no, ".IMPORT of %s, which no bank exports" % name)
            for name in mod.globals:
                if name in mod.symbols:
                    lab = mod.globals[name]
                    self.err(lab.file, lab.no, "%s is both a label and a symbol" % name)
            unnamed_li = [u[0] for u in mod.unnamed]
            glob_li = sorted(l.li for l in mod.labels if l.kind == "global")
            for st in mod.stmts:
                st.canon = self._canon_stmt(mod, st, unnamed_li, glob_li)

    @staticmethod
    def _bisect(seq, x):
        lo, hi = 0, len(seq)
        while lo < hi:
            mid = (lo + hi) // 2
            if seq[mid] <= x:
                lo = mid + 1
            else:
                hi = mid
        return lo                   # number of entries <= x

    def scope_of(self, mod, li):
        """The global Label owning cheap labels at expanded line index li (None before the first)."""
        cache = getattr(mod, "_glabs", None)
        if cache is None:
            cache = mod._glabs = sorted((l for l in mod.labels if l.kind == "global"), key=lambda l: l.li)
            mod._glis = [l.li for l in cache]
        k = self._bisect(mod._glis, li)
        return cache[k - 1] if k else None

    def _canon_stmt(self, mod, st, unnamed_li, glob_li):
        if st.head == ".INCBIN" or st.head == ".SEGMENT":
            return "%s %s" % (st.head, st.operand)
        if not st.operand:
            return st.head
        out, pos, text = [], 0, st.operand
        while pos < len(text):
            m = RE_TOKEN.match(text, pos)
            if not m:
                if not text[pos:].strip():
                    break
                self.err(st.file, st.no, "cannot tokenize operand: %s" % text)
                return "%s ?%s" % (st.head, text)
            pos = m.end()
            kind, tok = m.lastgroup, m.group(m.lastgroup)
            if kind == "hex":
                out.append("$%X" % int(tok[1:], 16))
            elif kind == "bin":
                out.append("$%X" % int(tok[1:], 2))
            elif kind == "dec":
                out.append("$%X" % int(tok))
            elif kind == "cheap":
                owner = self.scope_of(mod, st.li)
                lab = mod.cheap.get(owner.li if owner else -1, {}).get(tok)
                if lab is None:
                    self.err(st.file, st.no, "unresolved cheap label %s" % tok)
                    out.append("?" + tok)
                else:
                    out.append("L%d:%d" % (mod.id, lab.pos))
            elif kind == "unnamed":
                n = len(tok) - 1
                k = self._bisect(unnamed_li, st.li)
                j = k + n - 1 if tok[1] == "+" else k - n
                # (an unnamed label on the statement's own line counts as "previous")
                if j < 0 or j >= len(unnamed_li):
                    self.err(st.file, st.no, "unresolved unnamed label reference %s" % tok)
                    out.append("?" + tok)
                else:
                    out.append("L%d:%d" % (mod.id, mod.unnamed[j][1]))
            elif kind == "id":
                out.append(self._resolve_id(mod, st, tok))
            else:
                out.append(tok)
        return "%s %s" % (st.head, "".join(out))

    def _resolve_id(self, mod, st, name):
        if name in mod.symbols:
            return "$%X" % mod.symbols[name].value
        if name in mod.globals:
            return "L%d:%d" % (mod.id, mod.globals[name].pos)
        if name in mod.imports:
            if name.startswith("__") and name.endswith("__"):
                return name
            mods = self.exporters.get(name, [])
            if len(mods) == 1:
                return "L%d:%d" % (mods[0].id, mods[0].globals[name].pos)
            return "?" + name       # reported once at the .IMPORT line
        if name in ("A", "X", "Y"):
            return name
        others = [m.id for m in self.mods if name in m.globals]
        if others:
            self.err(st.file, st.no, "%s is defined in bank %s but not imported here" % (
                name, ",".join(map(str, others))))
        elif name in self.all_symbols:
            self.err(st.file, st.no, "symbol %s is defined in %s, which this bank does not include" % (
                name, self.all_symbols[name].file))
        else:
            self.err(st.file, st.no, "unresolved identifier %s" % name)
        return "?" + name

    # ---- lookups --------------------------------------------------------------------------------
    def fingerprint(self, mod):
        return hashlib.sha256("\n".join(s.canon for s in mod.stmts).encode()).hexdigest()

    def global_defs(self, name):
        return [m.globals[name] for m in self.mods if name in m.globals]

    def labels_at(self, mod):
        cache = getattr(mod, "_at", None)
        if cache is None:
            cache = mod._at = defaultdict(list)
            for lab in mod.labels:
                cache[lab.pos].append(lab)
        return cache

    def owner_matches(self, mod, owner, gname):
        """True when global label `gname` may stand for cheap-label scope owner `owner`."""
        if owner is None:
            return False
        if owner.name == gname:
            return True
        g = mod.globals.get(gname)
        return bool(g and g.pos == owner.pos and g.li < owner.li)

    def resolve_name(self, name):
        """Name (global, Global@Cheap or symbol) -> (kind, srcfile, line, obj) or raises AsmError."""
        if "@" in name:
            gname, cname = name.split("@", 1)
            hits = []
            for mod in self.mods:
                for key, table in mod.cheap.items():
                    lab = table.get("@" + cname)
                    if lab and self.owner_matches(mod, lab.scope, gname):
                        hits.append(lab)
            if len(hits) == 1:
                return ("cheap", hits[0].file, hits[0].no, hits[0])
            raise AsmError("%s: %s" % (name, "ambiguous" if hits else "no such cheap label"))
        defs = self.global_defs(name)
        count = len(defs) + (name in self.all_symbols)
        if count == 1 and defs:
            return ("global", defs[0].file, defs[0].no, defs[0])
        if count == 1:
            s = self.all_symbols[name]
            return ("symbol", s.file, s.no, s)
        raise AsmError("%s: %s" % (name, "ambiguous (defined %d times)" % count if count else "no such name"))

    def label_kind(self, lab):
        mod = lab.mod
        if lab.pos < len(mod.stmts) and mod.stmts[lab.pos].head in MNEMONICS:
            return "code"
        return "data"

    def summary(self, mod, lab):
        """First meaningful comment line of the block directly above a label."""
        lines, i = mod.lines, lab.li - 1
        if lab.kind == "cheap":
            own = lines[lab.li].comment[1:].strip() if lines[lab.li].comment else ""
            if own:
                return own
            if i >= 0 and lines[i].file == lab.file and not lines[i].code.strip() and lines[i].comment:
                return lines[i].comment[1:].strip()
            return ""
        def label_only(k):
            m = RE_GLOBAL.match(lines[k].code.strip())
            return bool(m and not m.group(2).strip())
        while i >= 0 and lines[i].file == lab.file and label_only(i):
            i -= 1
        block = []
        while i >= 0 and lines[i].file == lab.file and not lines[i].code.strip() and lines[i].comment:
            block.append(lines[i].comment)
            i -= 1
        for c in reversed(block):
            if RE_RULE.match(c):
                continue
            t = c[1:].strip()
            if t:
                return t
        return ""

    def chunk_files(self, mod):
        """Chunk files (src-relative) included by a bank file, in order."""
        return [inc for f, no, inc in mod.includes if f == mod.path and re.match(r"^bank\d/", inc)]

    def segments_of(self, mod, file):
        """Names of the segments that statements of `file` are assembled into."""
        seg, out = "", []
        for st in mod.stmts:
            if st.head == ".SEGMENT":
                seg = st.operand.strip('"')
            if st.file == file and seg not in out:
                out.append(seg)
        return out


# ---- baseline -------------------------------------------------------------------------------------

def upstream_dir(root):
    return os.path.join(root, "index", "upstream")


def load_baseline(root):
    d = upstream_dir(root)
    if not os.path.exists(os.path.join(d, "fingerprints.json")):
        return None
    base = {"stmts": {}, "labels": [], "symbols": {}}
    for n in range(NUM_MODULES):
        with gzip.open(os.path.join(d, "Z_%02d.canon.gz" % n), "rt", encoding="utf-8") as f:
            rows = []
            for line in f.read().split("\n"):
                if line:
                    loc, canon = line.split("\t", 1)
                    file, no = loc.rsplit(":", 1)
                    rows.append((file, int(no), canon))
            base["stmts"][n] = rows
    with open(os.path.join(d, "labels.tsv"), encoding="utf-8") as f:
        for line in f.read().split("\n")[1:]:
            if line:
                m, pos, order, kind, scope, name, file, no = line.split("\t")
                base["labels"].append(dict(mod=int(m), pos=int(pos), order=int(order), kind=kind,
                                           scope=scope, name=name, file=file, no=int(no)))
    with open(os.path.join(d, "symbols.tsv"), encoding="utf-8") as f:
        for line in f.read().split("\n")[1:]:
            if line:
                name, value, file, no = line.split("\t")
                base["symbols"][name] = (int(value[1:], 16), file, int(no))
    with open(os.path.join(d, "fingerprints.json"), encoding="utf-8") as f:
        base["fingerprints"] = json.load(f)
    return base


def cmd_freeze(args):
    d = upstream_dir(args.root)
    if os.path.exists(os.path.join(d, "fingerprints.json")) and not args.force:
        print("baseline already frozen; it is never refreshed", file=sys.stderr)
        return 1
    prog = Program(args.root)
    if report(prog.errors):
        return 1
    os.makedirs(d, exist_ok=True)
    fps = {}
    for mod in prog.mods:
        body = "".join("%s:%d\t%s\n" % (s.file, s.no, s.canon) for s in mod.stmts)
        with open(os.path.join(d, "Z_%02d.canon.gz" % mod.id), "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as f:
                f.write(body.encode())
        fps["Z_%02d" % mod.id] = prog.fingerprint(mod)
    rows = ["module\tposition\torder\tkind\tscope\tname\tfile\tline"]
    for mod in prog.mods:
        order = defaultdict(int)
        for lab in mod.labels:
            rows.append("%d\t%d\t%d\t%s\t%s\t%s\t%s\t%d" % (
                mod.id, lab.pos, order[lab.pos], lab.kind, lab.scope.name if lab.scope else "",
                lab.name, lab.file, lab.no))
            order[lab.pos] += 1
    write_text(os.path.join(d, "labels.tsv"), "\n".join(rows) + "\n")
    rows = ["name\tvalue\tfile\tline"]
    for s in sorted(prog.all_symbols.values(), key=lambda s: (s.file, s.no)):
        rows.append("%s\t$%X\t%s\t%d" % (s.name, s.value, s.file, s.no))
    write_text(os.path.join(d, "symbols.tsv"), "\n".join(rows) + "\n")
    write_text(os.path.join(d, "fingerprints.json"), json.dumps(fps, indent=1, sort_keys=True) + "\n")
    for k in sorted(fps):
        print(k, fps[k])
    return 0


# ---- renames.tsv ------------------------------------------------------------------------------------

class Renames:
    """index/renames.tsv: upstream name -> current name.

    kind `global`: scope `bankN`.  kind `cheap`: scope `bankN:<upstream global scope>`, names with
    their `@`.  kind `symbol`: scope empty.
    """

    def __init__(self, root):
        self.path = os.path.join(root, "index", "renames.tsv")
        self.rows = {}              # (kind, scope, old) -> new
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                for line in f.read().split("\n")[1:]:
                    if line:
                        old, new, kind, scope = line.split("\t")
                        self.rows[(kind, scope, old)] = new

    def glob(self, mod, name):
        return self.rows.get(("global", "bank%d" % mod, name), name)

    def cheap(self, mod, scope, name):
        return self.rows.get(("cheap", "bank%d:%s" % (mod, scope), name), name)

    def sym(self, name):
        return self.rows.get(("symbol", "", name), name)

    def text(self):
        out = ["old\tnew\tkind\tscope"]
        for (kind, scope, old), new in sorted(self.rows.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
            if old != new:
                out.append("%s\t%s\t%s\t%s" % (old, new, kind, scope))
        return "\n".join(out) + "\n"


# ---- helpers ------------------------------------------------------------------------------------------

def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def report(errors, only=None, limit=400):
    """Print problems as `file:line: message`; returns how many were printed."""
    shown = 0
    for file, no, msg in errors:
        if only and not any(file == o or file.startswith(o.rstrip("/") + "/") for o in only):
            continue
        shown += 1
        if shown <= limit:
            print("%s:%s: %s" % (file, no, msg))
    if shown > limit:
        print("... %d more problems" % (shown - limit))
    return shown


def md_files(root):
    out = []
    for base, dirs, files in os.walk(root):
        relbase = os.path.relpath(base, root)
        dirs[:] = sorted(d for d in dirs if not (relbase == "." and d in ("work", "bin", "obj", "ext"))
                         and not (relbase == "index" and d == "upstream"))
        for f in sorted(files):
            if f.endswith(".md"):
                out.append(os.path.join(base, f))
    return out


def src_files(prog):
    out = []
    for base, dirs, files in os.walk(prog.src):
        dirs.sort()
        for f in sorted(files):
            out.append(os.path.join(base, f))
    return out


def load_subjects(root):
    p = os.path.join(root, "tools", "subjects.txt")
    if not os.path.exists(p):
        return None
    subs = set()
    for line in read_text(p).split("\n"):
        line = line.split("#", 1)[0].strip()
        if line:
            subs.add(line.split()[0])
    return subs


def split_header(text):
    """(header lines, rest lines) of a chunk file; header is None when the file has none."""
    lines = text.split("\n")
    if not lines or not RE_HRULE.match(lines[0]):
        return None, lines
    for i in range(1, len(lines)):
        if RE_HRULE.match(lines[i]):
            return lines[:i + 1], lines[i + 1:]
    return None, lines


def header_fields(header):
    """field name -> list of text lines (first line after the colon, then continuation lines)."""
    fields, cur = {}, None
    for line in header[1:-1]:
        m = re.match(r"^; ([A-Z][A-Z ]*[A-Z]):\s*(.*)$", line)
        if m and m.group(1) in HEADER_FIELDS:
            cur = m.group(1)
            first = m.group(2)
            if cur == "FILE":
                first = re.sub(r"\s+BANK:.*$", "", first)
            fields[cur] = [re.sub(r"\s*\((generated[^)]*)\)\s*$", "", first)]
        elif cur:
            fields[cur].append(line[1:].strip() if line.startswith(";") else line)
    return fields


def bank_role(prog, mod, chunk):
    segs = prog.segments_of(mod, chunk)
    if "BANK_01_CODE" in segs:
        return "1 (stored at $A500-$B76F; copied to RAM and run at $6C90-$7EFF)"
    if "BANK_06_DATA" in segs:
        return "6 (switchable, $8000-$BFFF; the BANK_06_DATA part is copied to RAM $67F0-$687D)"
    if mod.id == 7:
        return "7 (fixed, $C000-$FFFF)"
    return "%d (switchable, $8000-$BFFF)" % mod.id


# ---- gen ------------------------------------------------------------------------------------------------

RE_DEFLINE = re.compile(r"^(%s)\s*(:=|=)\s*(\S+)\s*(;.*)?$" % IDENT)


def align_definition(line):
    """`Name := $addr      ; meaning` with the `;` at column 37 (one space when the definition is longer)."""
    m = RE_DEFLINE.match(line)
    if not m:
        return line
    d = "%s %s %s" % (m.group(1), m.group(2), m.group(3))
    if not m.group(4):
        return d
    return (d.ljust(36) if len(d) < 36 else d + " ") + m.group(4)


def wrap_items(prefix, cont, items, width=108):
    lines, cur = [], prefix
    for i, it in enumerate(items):
        piece = it + ("," if i < len(items) - 1 else "")
        if len(cur) + 1 + len(piece) > width and cur.strip("; ") and cur != prefix:
            lines.append(cur.rstrip())
            cur = cont
        cur += (" " if not cur.endswith(" ") else "") + piece
    lines.append(cur.rstrip())
    return lines


def regen_header(prog, mod, chunk, subjects):
    """New text of a chunk file with FILE, SUBJECTS and CONTENTS regenerated (None if no header)."""
    text = prog.read(chunk)
    header, rest = split_header(text)
    if header is None:
        return None
    labs = [l for l in mod.labels if l.file == chunk and l.kind == "global"]
    subs = sorted({l.name.split("_", 1)[0] for l in labs
                   if subjects and "_" in l.name and l.name.split("_", 1)[0] in subjects})
    def field_of(line):
        m = re.match(r"^; ([A-Z][A-Z ]*[A-Z]):", line)
        return m.group(1) if m and m.group(1) in HEADER_FIELDS else None
    blocks = []
    for line in header[1:-1]:
        f = field_of(line)
        if f or not blocks:
            blocks.append((f, [line]))
        else:
            blocks[-1][1].append(line)
    new, seen = [header[0]], set()
    for field, block in blocks:
        if field == "FILE":
            new.append("; FILE:     %-28s BANK: %s" % (chunk, bank_role(prog, mod, chunk)))
        elif field == "SUBJECTS":
            new.extend(wrap_items("; SUBJECTS:", ";          ", subs or ["(none)"]))
        elif field == "CONTENTS":
            new.append("; CONTENTS: (generated; L = line in this file)")
            new.append(None)        # placeholder, filled below
        else:
            new.extend(block)
        if field:
            seen.add(field)
    if "CONTENTS" not in seen:
        new.append("; CONTENTS: (generated; L = line in this file)")
        new.append(None)
    new.append(header[-1])
    shift = (len(new) - 1 + len(labs)) - len(header)
    width = max([len(l.name) for l in labs] or [0])
    contents = []
    for l in labs:
        s = prog.summary(mod, l)
        if len(s) > 110:
            s = s[:107].rstrip() + "..."
        contents.append((";   L%04d %-*s  %s  %s" % (l.no + shift, width, l.name, prog.label_kind(l), s)).rstrip())
    k = new.index(None)
    new[k:k + 1] = contents
    return "\n".join(new + rest)


def upstream_names(prog, base, ren):
    """(mod id, pos, current name) -> upstream name, for globals and cheap labels; symbol map too."""
    lab_up, sym_up = {}, {}
    if base:
        for b in base["labels"]:
            if b["kind"] == "global":
                cur = ren.glob(b["mod"], b["name"])
            else:
                cur = ren.cheap(b["mod"], b["scope"], b["name"])
            lab_up[(b["mod"], b["pos"], cur)] = b["name"]
        for name in base["symbols"]:
            sym_up[ren.sym(name)] = name
    return lab_up, sym_up

def upstream_to_current(root=None):
    """{upstream name: current name} for global labels and `:=` symbols of the original import.

    For tools that must keep working across renames: write the upstream name once, look the current
    one up here.  Names the original defined in more than one bank (`Exit`) are left out.
    """
    root = root or ROOT
    base, ren = load_baseline(root), Renames(root)
    out, dup = {}, set()
    for b in base["labels"]:
        if b["kind"] == "global":
            if b["name"] in out:
                dup.add(b["name"])
            out[b["name"]] = ren.glob(b["mod"], b["name"])
    for name in dup:
        del out[name]
    for name in base["symbols"]:
        out[name] = ren.sym(name)
    return out



def generate(root):
    """Returns {absolute path: text} for every generated or regenerated file."""
    out = {}
    subjects = load_subjects(root)
    prog = Program(root)
    if prog.split_done:
        overlay = {}
        for mod in prog.mods:
            for chunk in prog.chunk_files(mod):
                new = regen_header(prog, mod, chunk, subjects)
                if new is not None:
                    overlay[prog._abs(chunk)] = new
        for path, text in overlay.items():
            out[path] = text
        prog = Program(root, overlay=overlay)
    base, ren = load_baseline(root), Renames(root)
    lab_up, sym_up = upstream_names(prog, base, ren)

    # index/symbols.tsv
    rows = ["name\tkind\tbank\tfile\tline\tupstream_name\tsummary"]
    for mod in prog.mods:
        for lab in mod.labels:
            up = lab_up.get((mod.id, lab.pos, lab.name), "")
            if lab.kind == "global":
                name, kind = lab.name, prog.label_kind(lab)
            else:
                name, kind = "%s%s" % (lab.scope.name if lab.scope else "", lab.name), "cheap"
            rows.append("%s\t%s\t%d\t%s\t%d\t%s\t%s" % (
                name, kind, mod.id, prog.rel(lab.file), lab.no, up,
                prog.summary(mod, lab).replace("\t", " ")))
    for s in sorted(prog.all_symbols.values(), key=lambda s: (s.value, s.file, s.no)):
        rows.append("%s\tvar\t\t%s\t%d\t%s\t%s" % (
            s.name, prog.rel(s.file), s.no, sym_up.get(s.name, ""),
            ("$%04X " % s.value) + s.comment.replace("\t", " ")))
    out[os.path.join(root, "index", "symbols.tsv")] = "\n".join(rows) + "\n"

    # index/ram.md
    rows = ["# RAM symbols by address", "",
            "Generated by `tools/asm.py gen` from the `:=` definitions in `src/*.inc`. Several names on one",
            "address are aliases: the same byte used for different things by different object types or game",
            "modes. `Upstream` is the name in the original disassembly when it differs.", "",
            "| Address | Name | Upstream | Defined | Meaning |", "|---|---|---|---|---|"]
    for s in sorted(prog.all_symbols.values(), key=lambda s: (s.value, s.file, s.no)):
        up = sym_up.get(s.name, "")
        rows.append("| `$%04X` | `%s` | %s | %s:%d | %s |" % (
            s.value, s.name, "`%s`" % up if up and up != s.name else "",
            os.path.basename(s.file), s.no, s.comment.replace("|", "\\|")))
    out[os.path.join(root, "index", "ram.md")] = "\n".join(rows) + "\n"

    # index/files.md
    if prog.split_done:
        rows = ["# Source files", "",
                "Generated by `tools/asm.py gen`. One table per bank, files in ROM order. Each file's own header",
                "lists its labels with line numbers.", ""]
        for mod in prog.mods:
            chunks = prog.chunk_files(mod)
            rows += ["## Bank %d (`src/%s`)" % (mod.id, mod.path), "",
                     "| File | Lines | Labels | Subjects | Topic |", "|---|---|---|---|---|"]
            for chunk in chunks:
                text = prog.read(chunk)
                header, _ = split_header(text)
                f = header_fields(header) if header else {}
                nlab = sum(1 for l in mod.labels if l.file == chunk and l.kind == "global")
                subs = " ".join(x for x in f.get("SUBJECTS", [""]) if x).replace("(none)", "").strip()
                rows.append("| [%s](../src/%s) | %d | %d | %s | %s |" % (
                    chunk, chunk, text.count("\n") + (0 if text.endswith("\n") else 1), nlab, subs,
                    " ".join(x for x in f.get("TOPIC", [""]) if x).replace("|", "\\|")))
            rows.append("")
        out[os.path.join(root, "index", "files.md")] = "\n".join(rows)

    # src/*.inc: align the meaning comment of every definition at column 37
    for f in sorted(os.listdir(prog.src)):
        if f.endswith(".inc"):
            text = prog.read(f)
            new = "\n".join(align_definition(line) for line in text.split("\n"))
            if new != text:
                out[prog._abs(f)] = new

    # markdown: reference links and tables of contents
    for path in md_files(root):
        if os.path.dirname(path) == os.path.join(root, "index"):
            continue
        text = read_text(path)
        new = fix_links(prog, root, path, text)
        if os.path.dirname(path) == root and os.path.basename(path) in DOC_FILES:
            new = fix_toc(new)
        if new != text:
            out[path] = new
    return out, prog


def fix_links(prog, root, path, text):
    def repl(m):
        try:
            kind, file, no, obj = prog.resolve_name(m.group(1))
        except AsmError:
            return m.group(0)
        relp = os.path.relpath(prog._abs(file), os.path.dirname(path))
        return "[`%s`](%s#L%d)" % (m.group(1), relp, no)
    return RE_MDLINK.sub(repl, text)


def md_headings(lines):
    out, fence = [], False
    for i, line in enumerate(lines):
        if line.startswith("```"):
            fence = not fence
        elif not fence and line.startswith("# "):
            out.append((i, line[2:].strip()))
    return out


def fix_toc(text):
    lines = text.split("\n")
    try:
        a, b = lines.index("<!-- toc -->"), lines.index("<!-- /toc -->")
    except ValueError:
        return text
    heads = [(i, h) for i, h in md_headings(lines) if i > b]
    shift = len(heads) - (b - a - 1)
    toc = ["- L%d %s" % (i + 1 + shift, h) for i, h in heads]
    return "\n".join(lines[:a + 1] + toc + lines[b:])


def cmd_gen(args):
    out, prog = generate(args.root)
    if report(prog.errors):
        print("gen: source has errors; nothing written", file=sys.stderr)
        return 1
    n = 0
    for path, text in sorted(out.items()):
        if not os.path.exists(path) or read_text(path) != text:
            write_text(path, text)
            n += 1
    print("gen: %d file(s) written" % n)
    return 0


# ---- check ---------------------------------------------------------------------------------------------

def check(root, final=False, quick=False, naming=False, headers=False, selfc=False):
    """Returns the list of problems as (file, line, message)."""
    naming, headers, selfc = naming or final, headers or final, selfc or final
    prog = Program(root)
    errs = list(prog.errors)                                     # check 1
    E = lambda f, n, m: errs.append((f, n, m))
    base, ren = load_baseline(root), Renames(root)

    if base is None:
        E("index/upstream", 0, "no frozen baseline (run freeze-baseline once on the verbatim import)")
    else:
        for mod in prog.mods:                                    # check 2
            brow = base["stmts"][mod.id]
            for i, st in enumerate(mod.stmts):
                if i >= len(brow):
                    E(prog.rel(st.file), st.no, "statement %d has no upstream counterpart (bank %d has %d "
                      "statements upstream)" % (i, mod.id, len(brow)))
                    break
                if st.canon != brow[i][2]:
                    E(prog.rel(st.file), st.no, "statement differs from upstream %s:%d: now `%s`, upstream `%s`" % (
                        brow[i][0], brow[i][1], st.canon, brow[i][2]))
                    break
            else:
                if len(mod.stmts) < len(brow):
                    b = brow[len(mod.stmts)]
                    E(prog.rel(mod.path), 0, "bank %d ends early: upstream %s:%d `%s` is missing" % (
                        mod.id, b[0], b[1], b[2]))
        for key in ren.rows:                                     # check 3
            kind, scope, old = key
            if kind == "symbol":
                ok = old in base["symbols"]
            else:
                ok = any(b["kind"] == kind and b["name"] == old and
                         ("bank%d" % b["mod"] if kind == "global" else "bank%d:%s" % (b["mod"], b["scope"])) == scope
                         for b in base["labels"])
            if not ok:
                E("index/renames.tsv", 0, "row %s %s %s: no such upstream name" % (kind, scope, old))
        for b in base["labels"]:
            mod = prog.mods[b["mod"]]
            here = prog.labels_at(mod).get(b["pos"], [])
            if b["kind"] == "global":
                want = ren.glob(b["mod"], b["name"])
                ok = any(l.kind == "global" and l.name == want for l in here)
            else:
                want = ren.cheap(b["mod"], b["scope"], b["name"])
                ok = any(l.name == want for l in here)
            if not ok:
                E(prog.rel(here[0].file if here else mod.path), here[0].no if here else 0,
                  "upstream label %s (%s:%d) should be `%s` at statement %d of bank %d; labels there: %s" % (
                      b["name"], b["file"], b["no"], want, b["pos"], b["mod"],
                      ", ".join(l.name for l in here) or "none"))
        for name, (value, file, no) in base["symbols"].items():
            want = ren.sym(name)
            s = prog.all_symbols.get(want)
            if s is None:
                E("src/" + file, no, "upstream symbol %s should exist as `%s`" % (name, want))
            elif s.value != value:
                E(prog.rel(s.file), s.no, "symbol %s is $%X; upstream %s is $%X" % (want, s.value, name, value))

    if prog.split_done:                                          # check 4
        included = defaultdict(list)
        for mod in prog.mods:
            for chunk in prog.chunk_files(mod):
                included[chunk].append(mod.path)
                first = None
                for ln in prog.read(chunk).split("\n"):
                    code = strip_comment(ln)[0].strip()
                    if not code:
                        continue
                    if re.match(r"^\.(SEGMENT|EXPORT|IMPORT)\b", code, re.I):
                        continue
                    first = code
                    break
                if first is None or not RE_GLOBAL.match(first) or RE_SYMDEF.match(first):
                    E(prog.rel(chunk), 1, "chunk file must begin with a global label (found: %s)" % first)
        for base_dir, dirs, files in os.walk(prog.src):
            for f in files:
                relp = os.path.relpath(os.path.join(base_dir, f), prog.src)
                if re.match(r"^bank\d/", relp) and f.endswith(".asm"):
                    if len(included.get(relp, [])) != 1:
                        E(prog.rel(relp), 1, "chunk file is included %d times (must be exactly once)" % len(
                            included.get(relp, [])))
        seen = {}
        for mod in prog.mods:
            for name, lab in mod.globals.items():
                if name in seen:
                    E(prog.rel(lab.file), lab.no, "global label %s is also defined at %s:%d" % (
                        name, prog.rel(seen[name].file), seen[name].no))
                else:
                    seen[name] = lab
                if name in prog.all_symbols:
                    E(prog.rel(lab.file), lab.no, "global label %s is also a symbol" % name)

    if selfc:                                                    # check 5
        for path in src_files(prog):
            for no, line in enumerate(read_text(path).split("\n"), 1):
                m = RE_SELF.search(line)
                if m:
                    E(os.path.relpath(path, root), no, "outside reference `%s` (the tree must be self-contained)" % m.group(0))
        for path in md_files(root):
            for no, line in enumerate(read_text(path).split("\n"), 1):
                m = RE_SELF.search(line.replace("tools/asm.py", "").replace("tools/romdata.py", ""))
                if m:
                    E(os.path.relpath(path, root), no, "outside reference `%s` (the tree must be self-contained)" % m.group(0))

    if naming:                                                   # check 6
        subjects = load_subjects(root)
        if subjects is None:
            E("tools/subjects.txt", 0, "missing (needed for the naming check)")
            subjects = set()
        for mod in prog.mods:
            for lab in mod.labels:
                if lab.kind == "global":
                    if not RE_NAME_GLOBAL.match(lab.name):
                        E(prog.rel(lab.file), lab.no, "global label %s does not match <Subject>_<Rest>" % lab.name)
                    elif lab.name.split("_", 1)[0] not in subjects:
                        E(prog.rel(lab.file), lab.no, "global label %s: subject %s is not in tools/subjects.txt" % (
                            lab.name, lab.name.split("_", 1)[0]))
                elif not RE_NAME_CHEAP.match(lab.name):
                    E(prog.rel(lab.file), lab.no, "cheap label %s is not @PascalCase" % lab.name)
        for s in prog.all_symbols.values():
            if not RE_NAME_SYMBOL.match(s.name):
                E(prog.rel(s.file), s.no, "symbol %s is not PascalCase[_PascalCase]" % s.name)

    if headers:
        for mod in prog.mods:
            for chunk in (prog.chunk_files(mod) if prog.split_done else []):
                header, _ = split_header(prog.read(chunk))
                if header is None:
                    E(prog.rel(chunk), 1, "chunk file has no header block")
                    continue
                f = header_fields(header)
                for name in HEADER_FIELDS:
                    if name not in f:
                        E(prog.rel(chunk), 1, "header lacks the %s field" % name)
                if not " ".join(f.get("TOPIC", [])).strip():
                    E(prog.rel(chunk), 1, "header TOPIC is empty")
                facts = [x for x in f.get("KEY FACTS", []) if x.startswith("- ")]
                if not 1 <= len(facts) <= 12:
                    E(prog.rel(chunk), 1, "header KEY FACTS has %d bullets (1 to 12 wanted)" % len(facts))
            for lab in mod.labels:
                if lab.kind == "global":
                    s = prog.summary(mod, lab)
                    if not s or BAD_SUMMARY.match(s):
                        E(prog.rel(lab.file), lab.no, "global label %s has no summary line above it" % lab.name)

    if not quick:                                                # checks 7 and 8
        for path in md_files(root):
            relp = os.path.relpath(path, root)
            text = read_text(path)
            for no, line in enumerate(text.split("\n"), 1):
                for m in RE_MDLINK.finditer(line):
                    try:
                        prog.resolve_name(m.group(1))
                    except AsmError as e:
                        E(relp, no, "reference does not resolve: %s" % e)
            if os.path.dirname(path) == root and os.path.basename(path) in DOC_FILES:
                lines = text.split("\n")
                in_toc, fence, i = False, False, 0
                while i < len(lines):
                    line = lines[i]
                    if line.startswith("```"):
                        fence = not fence
                    if line == "<!-- toc -->":
                        in_toc = True
                    elif line == "<!-- /toc -->":
                        in_toc = False
                    elif line.startswith("- ") and not in_toc and not fence:
                        j, bullet = i + 1, line
                        while j < len(lines) and lines[j].startswith("  ") and lines[j].strip():
                            bullet += "\n" + lines[j]
                            j += 1
                        if not RE_MDLINK.search(bullet):
                            E(relp, i + 1, "bullet has no source reference link")
                        i = j - 1
                    i += 1
        for relp, no, name in outside_refs(root):
            try:
                prog.resolve_name(name)
            except AsmError as e:
                E(relp, no, "asm: citation does not resolve: %s" % e)
        if not prog.errors:
            out, _ = generate(root)
            for path, text in sorted(out.items()):
                if not os.path.exists(path) or read_text(path) != text:
                    E(os.path.relpath(path, root), 0, "generated content is stale (run `tools/asm.py gen`)")
    return errs


def outside_refs(root):
    """`asm:Name` citations in git-tracked text files outside this tree: (path, line, name)."""
    try:
        top = subprocess.run(["git", "-C", root, "rev-parse", "--show-toplevel"], capture_output=True,
                             text=True, check=True).stdout.strip()
        files = subprocess.run(["git", "-C", top, "ls-files"], capture_output=True, text=True,
                               check=True).stdout.split("\n")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    here = os.path.relpath(root, top) + "/"
    out = []
    for f in files:
        if not f or f.startswith(here) or not f.endswith((".md", ".rs", ".py", ".yaml", ".yml", ".toml", ".txt")):
            continue
        p = os.path.join(top, f)
        try:
            if os.path.getsize(p) > 4 << 20:
                continue
            text = read_text(p)
        except (OSError, UnicodeDecodeError):
            continue
        if "asm:" not in text:
            continue
        for no, line in enumerate(text.split("\n"), 1):
            for m in RE_ASMREF.finditer(line):
                out.append((os.path.relpath(p, root), no, m.group(1)))
    return out


def cmd_check(args):
    errs = check(args.root, final=args.final, quick=args.quick, naming=args.naming,
                 headers=args.headers, selfc=args.self_contained)
    n = report(errs, only=args.only)
    if n:
        print("check: %d problem(s)" % n)
        return 1
    print("check: ok")
    return 0


# ---- where -----------------------------------------------------------------------------------------------

def locate_stmt(prog, mod, idx):
    """(file relative to root, line, enclosing global name, nearest cheap name or '') of a statement."""
    if idx >= len(mod.stmts):
        return (prog.rel(mod.path), 0, "", "")
    st = mod.stmts[idx]
    owner = prog.scope_of(mod, st.li)
    cheap = ""
    if owner:
        cands = [l for l in mod.cheap.get(owner.li, {}).values() if l.li <= st.li]
        if cands:
            cheap = max(cands, key=lambda l: l.li).name
    return (prog.rel(st.file), st.no, owner.name if owner else "", cheap)


def cmd_where(args):
    prog = Program(args.root)
    base, ren = load_baseline(args.root), Renames(args.root)
    q = " ".join(args.query).strip()
    found = 0
    m = re.match(r"^(?:Z_?0?|bank)?([0-7])(?:\.asm)?[:\s]+L?(\d+)$", q)
    if m and base:
        n, line = int(m.group(1)), int(m.group(2))
        up = "Z_%02d.asm" % n
        rows = base["stmts"][n]
        idx = next((i for i, r in enumerate(rows) if r[0] == up and r[1] >= line), None)
        labs = [b for b in base["labels"] if b["mod"] == n and b["file"] == up and b["no"] == line]
        if idx is None:
            print("no statement at or after %s:%d" % (up, line))
            return 1
        file, no, owner, cheap = locate_stmt(prog, prog.mods[n], idx)
        upowner = [b for b in base["labels"] if b["mod"] == n and b["kind"] == "global" and b["pos"] <= idx]
        upname = upowner[-1]["name"] if upowner else ""
        name = owner + (cheap if cheap else "")
        if labs:
            lab = labs[0]
            cur = ren.glob(n, lab["name"]) if lab["kind"] == "global" else ren.cheap(n, lab["scope"], lab["name"])
            name = cur if lab["kind"] == "global" else owner + cur
        print("%s:%d  %s  (upstream %s:%d in %s%s)" % (
            file, no, name, up, line, upname,
            "" if rows[idx][1] == line else "; next statement is upstream line %d" % rows[idx][1]))
        return 0
    lab_up, sym_up = upstream_names(prog, base, ren)
    # current names
    for name in ([q] if q else []):
        try:
            kind, file, no, obj = prog.resolve_name(name)
        except AsmError:
            hits = prog.global_defs(name) if "@" not in name else []
            for lab in hits:
                print("%s:%d  %s  (bank %d)" % (prog.rel(lab.file), lab.no, lab.name, lab.mod.id))
                found += 1
            continue
        if kind == "symbol":
            up = sym_up.get(name, "")
            print("%s:%d  %s = $%04X  (upstream %s)" % (prog.rel(file), no, name, obj.value, up or "none"))
        else:
            up = lab_up.get((obj.mod.id, obj.pos, obj.name), "")
            b = next((b for b in (base["labels"] if base else []) if b["mod"] == obj.mod.id and b["pos"] == obj.pos
                      and b["name"] == up), None)
            print("%s:%d  %s  (upstream %s)" % (prog.rel(file), no, name,
                                                "%s:%d %s" % (b["file"], b["no"], up) if b else "none"))
        found += 1
    # upstream names
    if base:
        for b in base["labels"]:
            if b["name"] == q or (b["kind"] == "cheap" and "%s%s" % (b["scope"], b["name"]) == q):
                cur = ren.glob(b["mod"], q) if b["kind"] == "global" else ren.cheap(b["mod"], b["scope"], b["name"])
                here = [l for l in prog.labels_at(prog.mods[b["mod"]]).get(b["pos"], []) if l.name == cur]
                if here and not (b["kind"] == "global" and cur == q):
                    l = here[0]
                    shown = cur if l.kind == "global" else "%s%s" % (l.scope.name if l.scope else "", cur)
                    print("%s:%d  %s  (upstream %s:%d %s)" % (prog.rel(l.file), l.no, shown, b["file"], b["no"], q))
                    found += 1
        if q in base["symbols"] and ren.sym(q) != q:
            s = prog.all_symbols.get(ren.sym(q))
            if s:
                print("%s:%d  %s = $%04X  (upstream %s)" % (prog.rel(s.file), s.no, s.name, s.value, q))
                found += 1
    if not found:
        print("not found: %s" % q)
        return 1
    return 0


# ---- rename ------------------------------------------------------------------------------------------------

RE_CODE_TOKEN = re.compile(r'"[^"]*"|(?<![A-Za-z0-9_@$/])(%s)(?![A-Za-z0-9_])(?!\.(?:dat|inc|asm)\b)' % IDENT)
RE_CHEAP_TOKEN = re.compile(r"(?<![A-Za-z0-9_])(@%s)(?![A-Za-z0-9_])" % IDENT)
RE_QUAL_CHEAP = re.compile(r"(?<![A-Za-z0-9_@$/])(%s)(@%s)(?![A-Za-z0-9_])" % (IDENT, IDENT))


def sub_idents(text, table, prose):
    """Replace whole identifiers found in `table`; in prose, single plain words are left alone."""
    def repl(m):
        name = m.group(1)
        if name is None or name not in table:
            return m.group(0)
        if prose and RE_PLAIN_WORD.match(name):
            return m.group(0)
        return table[name]
    return RE_CODE_TOKEN.sub(repl, text)


def cmd_rename(args):
    root = args.root
    prog = Program(root)
    if report(prog.errors):
        print("rename: fix the source errors first", file=sys.stderr)
        return 1
    base, ren = load_baseline(root), Renames(root)
    subjects = load_subjects(root)
    problems = []
    glob_all = {}               # old -> new, program wide (globals and symbols)
    glob_mod = defaultdict(dict)    # mod id -> {old: new} for names defined in several banks
    cheap_map = defaultdict(dict)   # (mod id, owner li) -> {old: new}
    ents = []                   # (kind, obj, new)
    rows = []
    for no, line in enumerate(read_text(args.map).split("\n"), 1):
        line = line.rstrip("\r")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = [p.strip() for p in line.split("\t") if p.strip()]
        if len(parts) < 2:
            parts = line.split()
        if len(parts) < 2:
            problems.append("%s:%d: need Old<TAB>New" % (args.map, no))
            continue
        rows.append((no, parts[0], parts[1]))
    for no, old, new in rows:
        where = "%s:%d" % (args.map, no)
        modsel = None
        m = re.match(r"^bank(\d):(.+)$", old)
        if m:
            modsel, old = int(m.group(1)), m.group(2)
        if "@" in old:
            gname, cname = old.split("@", 1)
            cname = "@" + cname
            hits = []
            for mod in prog.mods:
                if modsel is not None and mod.id != modsel:
                    continue
                for table in mod.cheap.values():
                    lab = table.get(cname)
                    if lab and prog.owner_matches(mod, lab.scope, gname):
                        hits.append(lab)
            if len(hits) != 1:
                problems.append("%s: %s%s: %s" % (where, gname, cname, "ambiguous; use bankN:" if hits else "no such cheap label"))
                continue
            if not new.startswith("@"):
                new = "@" + new
            if not RE_NAME_CHEAP.match(new):
                problems.append("%s: %s is not @PascalCase" % (where, new))
                continue
            lab = hits[0]
            key = (lab.mod.id, lab.scope.li if lab.scope else -1)
            if cname in cheap_map[key]:
                problems.append("%s: %s%s renamed twice" % (where, gname, cname))
            cheap_map[key][cname] = new
            ents.append(("cheap", lab, new))
            continue
        defs = [d for d in prog.global_defs(old) if modsel is None or d.mod.id == modsel]
        alldefs = prog.global_defs(old)
        if old in prog.all_symbols and not defs:
            if not RE_NAME_SYMBOL.match(new):
                problems.append("%s: symbol name %s is not PascalCase[_PascalCase]" % (where, new))
                continue
            if old in glob_all:
                problems.append("%s: %s renamed twice" % (where, old))
            glob_all[old] = new
            ents.append(("symbol", prog.all_symbols[old], new))
            continue
        if len(defs) != 1:
            problems.append("%s: %s: %s" % (where, old, "defined in %d banks; use bankN:%s" % (len(defs), old)
                                            if defs else "no such label or symbol"))
            continue
        if not RE_NAME_GLOBAL.match(new) and not (subjects is None and re.fullmatch(IDENT, new)):
            problems.append("%s: label name %s does not match <Subject>_<Rest>" % (where, new))
            continue
        if subjects is not None and new.split("_", 1)[0] not in subjects:
            problems.append("%s: %s: subject %s is not in tools/subjects.txt" % (where, new, new.split("_", 1)[0]))
            continue
        lab = defs[0]
        if len(alldefs) > 1:
            if any(old in m.exports for m in prog.mods):
                problems.append("%s: %s is defined in several banks and exported; not supported" % (where, old))
                continue
            if old in glob_mod[lab.mod.id]:
                problems.append("%s: %s renamed twice" % (where, old))
            glob_mod[lab.mod.id][old] = new
        else:
            if old in glob_all:
                problems.append("%s: %s renamed twice" % (where, old))
            glob_all[old] = new
        ents.append(("global", lab, new))

    # collisions: globals and symbols share one namespace across the program
    taken = defaultdict(int)
    for mod in prog.mods:
        for name in mod.globals:
            taken[name] += 1
    for name in prog.all_symbols:
        taken[name] += 1
    for kind, obj, new in ents:
        if kind != "cheap":
            taken[obj.name] -= 1
    newcount = defaultdict(int)
    for kind, obj, new in ents:
        if kind != "cheap":
            newcount[new] += 1
    for name, n in sorted(newcount.items()):
        if n > 1:
            problems.append("new name %s is given to %d labels or symbols" % (name, n))
        elif taken.get(name, 0) > 0:
            problems.append("new name %s already exists" % name)
    for (mid, owner_li), table in cheap_map.items():
        existing = set(prog.mods[mid].cheap.get(owner_li, {})) - set(table)
        news = list(table.values())
        for n in set(news):
            if news.count(n) > 1 or n in existing:
                owner = next((l for l in prog.mods[mid].labels if l.li == owner_li), None)
                problems.append("cheap name %s collides in scope %s" % (n, owner.name if owner else "?"))
    if problems:
        for p in problems:
            print(p)
        print("rename: %d problem(s); nothing changed" % len(problems))
        return 1
    if args.dry_run:
        print("rename: map is valid (%d labels/symbols, %d cheap labels); dry run, nothing changed" % (
            sum(1 for e in ents if e[0] != "cheap"), sum(1 for e in ents if e[0] == "cheap")))
        return 0

    # rewrite source files
    backup, changed, skipped = {}, {}, []
    file_lines = {}
    shared = set()
    counts = defaultdict(int)
    for mod in prog.mods:
        for f in mod.files:
            counts[f] += 1
    shared = {f for f, c in counts.items() if c > 1}
    done_shared = set()
    for mod in prog.mods:
        local = dict(glob_all)
        local.update(glob_mod.get(mod.id, {}))
        for li, ln in enumerate(mod.lines):
            if ln.file in shared:
                if (ln.file, ln.no) in done_shared:
                    continue
                done_shared.add((ln.file, ln.no))
                table = glob_all
            else:
                table = local
            code = sub_idents(ln.code, table, prose=False)
            comment = sub_idents(ln.comment, table, prose=True)
            for m in RE_CODE_TOKEN.finditer(ln.comment):
                if m.group(1) in table and RE_PLAIN_WORD.match(m.group(1)):
                    skipped.append("%s:%d: %s -> %s" % (prog.rel(ln.file), ln.no, m.group(1), table[m.group(1)]))
            owner = prog.scope_of(mod, li)
            cm = cheap_map.get((mod.id, owner.li if owner else -1))
            new = code + comment
            if cm:
                new = RE_CHEAP_TOKEN.sub(lambda m: cm.get(m.group(1), m.group(1)), new)
            file_lines.setdefault(ln.file, {})[ln.no] = new
    for f, table in file_lines.items():
        path = prog._abs(f)
        old_text = read_text(path)
        lines = old_text.split("\n")
        for no, new in table.items():
            lines[no - 1] = new
        new_text = "\n".join(lines)
        if new_text != old_text:
            backup[path], changed[path] = old_text, new_text

    # rewrite documents and scratch notes
    cheap_by_name = {}
    for kind, obj, new in ents:
        if kind == "cheap" and obj.scope:
            cheap_by_name[(obj.scope.name, obj.name)] = new
    extra = md_files(root)
    work = os.path.join(root, "work")
    for base_dir, dirs, files in os.walk(work):
        if os.path.relpath(base_dir, work).split(os.sep)[0] == "renames":
            continue
        for f in sorted(files):
            extra.append(os.path.join(base_dir, f))
    for path in extra:
        if os.path.dirname(path) == os.path.join(root, "index"):
            continue
        try:
            old_text = read_text(path)
        except UnicodeDecodeError:
            continue
        def qual(m):
            g, c = m.group(1), m.group(2)
            return glob_all.get(g, g) + cheap_by_name.get((g, c), c)
        def line_sub(line):
            parts = RE_QUAL_CHEAP.sub(qual, line).split("`")
            return "`".join(sub_idents(p, glob_all, prose=(k % 2 == 0)) for k, p in enumerate(parts))
        new_text = "\n".join(line_sub(line) for line in old_text.split("\n"))
        if new_text != old_text:
            backup[path], changed[path] = old_text, new_text

    # renames.tsv: key every renamed entity by its upstream identity
    lab_up, sym_up = upstream_names(prog, base, ren)
    bscope = {}
    if base:
        for b in base["labels"]:
            if b["kind"] == "cheap":
                bscope[(b["mod"], b["pos"], b["name"])] = b["scope"]
    for kind, obj, new in ents:
        if kind == "symbol":
            up = sym_up.get(obj.name)
            if up:
                ren.rows[("symbol", "", up)] = new
        else:
            up = lab_up.get((obj.mod.id, obj.pos, obj.name))
            if not up:
                continue
            if kind == "global":
                ren.rows[("global", "bank%d" % obj.mod.id, up)] = new
            else:
                ren.rows[("cheap", "bank%d:%s" % (obj.mod.id, bscope[(obj.mod.id, obj.pos, up)]), up)] = new
    ren_old = read_text(ren.path) if os.path.exists(ren.path) else None
    backup[ren.path], changed[ren.path] = ren_old, ren.text()

    for path, text in changed.items():
        write_text(path, text)
    gen_out, _ = generate(root)
    for path, text in gen_out.items():
        if path not in backup:
            backup[path] = read_text(path) if os.path.exists(path) else None
        write_text(path, text)
    errs = check(root, quick=True)
    if errs:
        report(errs, limit=40)
        for path, text in backup.items():
            if text is None:
                if os.path.exists(path):
                    os.remove(path)
            else:
                write_text(path, text)
        print("rename: check failed; every file restored, nothing changed")
        return 1
    print("rename: %d labels/symbols and %d cheap labels renamed in %d file(s)" % (
        sum(1 for e in ents if e[0] != "cheap"), sum(1 for e in ents if e[0] == "cheap"),
        sum(1 for p in changed if p != ren.path)))
    if skipped:
        note = os.path.join(root, "work", "rename-skipped.txt")
        os.makedirs(os.path.dirname(note), exist_ok=True)
        with open(note, "a", encoding="utf-8") as f:
            f.write("\n".join(skipped) + "\n")
        print("rename: %d comment mention(s) of renamed single-word names were left alone (a word such as "
              "`Exit` may be prose); review %s" % (len(skipped), os.path.relpath(note, root)))
    return 0


# ---- split --------------------------------------------------------------------------------------------------

def cmd_split(args):
    root = args.root
    prog = Program(root)
    if prog.split_done:
        print("split: already done", file=sys.stderr)
        return 1
    if report(prog.errors):
        return 1
    manifest = defaultdict(list)
    for no, line in enumerate(read_text(args.manifest).split("\n"), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 5:
            print("%s:%d: need `bank | label | upstream line | file | topic [| start=N]`" % (args.manifest, no))
            return 1
        start = None
        if len(parts) > 5 and parts[5].startswith("start="):
            start = int(parts[5][6:])
        manifest[int(parts[0])].append(dict(label=parts[1], line=int(parts[2]), file=parts[3],
                                            topic=parts[4], start=start))
    problems, plan = [], {}
    for mod in prog.mods:
        chunks = manifest.get(mod.id)
        if not chunks:
            problems.append("bank %d: no chunks in the manifest" % mod.id)
            continue
        lines = prog.read(mod.path).split("\n")
        is_filler = lambda s: (not strip_comment(s)[0].strip()) or re.match(
            r"^\s*\.(SEGMENT|EXPORT|IMPORT)\b", strip_comment(s)[0], re.I)
        prev_label = 0
        for c in chunks:
            lab = mod.globals.get(c["label"])
            if lab is None or lab.file != mod.path or lab.no != c["line"]:
                problems.append("bank %d: %s is not a global label at line %d%s" % (
                    mod.id, c["label"], c["line"], " (it is at %d)" % lab.no if lab and lab.file == mod.path else ""))
                continue
            if c["start"] is None:
                s = c["line"]
                # The first chunk takes only the comment lines attached to its label: everything
                # above it is the bank header and stays in the bank file.
                while s - 1 > prev_label and is_filler(lines[s - 2]) and (
                        prev_label or lines[s - 2].lstrip().startswith(";")):
                    s -= 1
                while s < c["line"] and not lines[s - 1].strip():
                    s += 1
                c["start"] = s
            elif not (prev_label < c["start"] <= c["line"]) or not all(
                    is_filler(lines[k - 1]) for k in range(c["start"], c["line"])):
                problems.append("bank %d: %s: start=%d is not usable" % (mod.id, c["label"], c["start"]))
            prev_label = c["line"]
        for a, b in zip(chunks, chunks[1:]):
            if a.get("start") and b.get("start") and not a["start"] < b["start"]:
                problems.append("bank %d: chunks %s and %s are out of order" % (mod.id, a["file"], b["file"]))
        if problems:
            continue
        bounds = [c["start"] for c in chunks] + [len(lines) + 1]
        chunk_of = lambda no: max((i for i in range(len(chunks)) if bounds[i] <= no), default=-1)
        unnamed_li = [u[0] for u in mod.unnamed]
        for st in mod.stmts:
            if st.file != mod.path:
                continue
            m = re.fullmatch(r":([+\-]+)", st.operand)
            if not m:
                continue
            n, k = len(m.group(1)), Program._bisect(unnamed_li, st.li)
            j = k + n - 1 if m.group(1)[0] == "+" else k - n
            target = mod.lines[mod.unnamed[j][0]]
            if target.file == mod.path and chunk_of(target.no) != chunk_of(st.no):
                problems.append("bank %d: unnamed label reference at %s:%d reaches line %d across the start of %s" % (
                    mod.id, mod.path, st.no, target.no, chunks[max(chunk_of(st.no), chunk_of(target.no))]["file"]))
        for lab in mod.labels:
            if lab.kind == "cheap" and lab.file == mod.path and lab.scope and lab.scope.file == mod.path \
                    and chunk_of(lab.no) != chunk_of(lab.scope.no):
                problems.append("bank %d: cheap label %s at line %d is cut off from its scope %s" % (
                    mod.id, lab.name, lab.no, lab.scope.name))
        plan[mod.id] = (lines, chunks, bounds)
    if problems:
        for p in problems:
            print(p)
        print("split: %d problem(s); nothing written" % len(problems))
        return 1
    rule = "; " + "=" * 76
    for mod in prog.mods:
        lines, chunks, bounds = plan[mod.id]
        head = lines[:bounds[0] - 1]
        while head and not head[-1].strip():
            head.pop()
        bank = head + [""] + ['.INCLUDE "bank%d/%s.asm"' % (mod.id, c["file"]) for c in chunks] + [""]
        for i, c in enumerate(chunks):
            body = lines[bounds[i] - 1:bounds[i + 1] - 1]
            while body and not body[-1].strip():
                body.pop()
            relp = "bank%d/%s.asm" % (mod.id, c["file"])
            hdr = [rule,
                   "; FILE:     %-28s BANK: %d" % (relp, mod.id),
                   "; UPSTREAM: %s %d-%d" % (os.path.basename(mod.path), bounds[i], bounds[i] + len(body) - 1),
                   "; SUBJECTS: (none)",
                   "; TOPIC:    %s" % c["topic"],
                   "; SEE ALSO:",
                   "; KEY FACTS:",
                   "; CONTENTS: (generated; L = line in this file)",
                   rule, ""]
            write_text(prog._abs(relp), "\n".join(hdr + body) + "\n")
        write_text(prog._abs("bank%d.asm" % mod.id), "\n".join(bank))
        os.remove(prog._abs(mod.path))
    print("split: %d chunk files written" % sum(len(p[1]) for p in plan.values()))
    return 0


# ---- stale-names, canon-hash ----------------------------------------------------------------------------------

def cmd_stale(args):
    ren = Renames(args.root)
    old = {}
    for (kind, scope, o), n in ren.rows.items():
        if kind != "cheap" and o != n and not RE_PLAIN_WORD.match(o):
            old.setdefault(o, set()).add(n)
    prog = Program(args.root)
    current = set(prog.all_symbols)
    for mod in prog.mods:
        current |= set(mod.globals)
    old = {o: n for o, n in old.items() if o not in current}
    n = 0
    for path in args.paths:
        try:
            text = read_text(path)
        except (OSError, UnicodeDecodeError):
            continue
        for no, line in enumerate(text.split("\n"), 1):
            scan = line
            if path.endswith(".rs"):
                scan = line[line.index("//"):] if "//" in line else ""
            elif path.endswith((".yaml", ".yml", ".toml")):
                scan = line[line.index("#"):] if "#" in line else ""
            for m in re.finditer(r"(?<![A-Za-z0-9_@$/])(%s)(?![A-Za-z0-9_])" % IDENT, scan):
                if m.group(1) in old:
                    print("%s:%d: %s -> %s" % (path, no, m.group(1), " | ".join(sorted(old[m.group(1)]))))
                    n += 1
    return 1 if n else 0


def cmd_canon_hash(args):
    prog = Program(args.root, src=os.path.abspath(args.src) if args.src else None)
    report(prog.errors)
    base = load_baseline(args.root)
    rc = 0
    for mod in prog.mods:
        fp = prog.fingerprint(mod)
        tag = ""
        if base:
            same = base["fingerprints"].get("Z_%02d" % mod.id) == fp
            tag = "  same as baseline" if same else "  DIFFERS from baseline"
            rc |= 0 if same else 1
        print("Z_%02d %s %d statements%s" % (mod.id, fp, len(mod.stmts), tag))
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT, help="the zelda-asm directory (default: this tool's parent)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check", help="verify the tree against the frozen baseline and the house rules")
    p.add_argument("--final", action="store_true", help="also naming, headers/summaries, self-containment")
    p.add_argument("--quick", action="store_true", help="skip documents and generated-file currency")
    p.add_argument("--naming", action="store_true")
    p.add_argument("--headers", action="store_true")
    p.add_argument("--self-contained", action="store_true")
    p.add_argument("--only", nargs="+", help="report only problems in these files (paths relative to the root)")
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("gen", help="regenerate indexes, file headers, tables of contents and link targets")
    p.set_defaults(fn=cmd_gen)
    p = sub.add_parser("rename", help="rename labels and symbols everywhere from a TSV map")
    p.add_argument("map")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_rename)
    p = sub.add_parser("where", help="find a current name, an upstream name, or an upstream file:line")
    p.add_argument("query", nargs="+")
    p.set_defaults(fn=cmd_where)
    p = sub.add_parser("split", help="one-time: cut the bank files into chunk files")
    p.add_argument("manifest")
    p.set_defaults(fn=cmd_split)
    p = sub.add_parser("stale-names", help="list renamed upstream names still used in the given files")
    p.add_argument("paths", nargs="+")
    p.set_defaults(fn=cmd_stale)
    p = sub.add_parser("canon-hash", help="print canonical fingerprints of a source directory")
    p.add_argument("--src")
    p.set_defaults(fn=cmd_canon_hash)
    p = sub.add_parser("freeze-baseline", help="one-time: freeze the canonical form of the verbatim import")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_freeze)
    args = ap.parse_args(argv)
    args.root = os.path.abspath(args.root)
    try:
        return args.fn(args)
    except AsmError as e:
        print("error: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
