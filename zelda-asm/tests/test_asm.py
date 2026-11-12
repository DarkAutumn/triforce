"""The annotated disassembly must stay equivalent to the code it was imported as.

`tools/asm.py check` compares every statement with a frozen canonical form; these tests prove the
check passes on the tree and fails when a label is redirected, half-renamed or an operand changes.

They need only the standard library and pytest, and live inside this tree (not under the repository's
`tests/`, whose conftest loads the emulator module), so `pytest zelda-asm/tests` runs anywhere.
"""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_LABEL = "CheckScreenEdge"      # upstream name, bank 7; found through the baseline


def run(root, *args):
    result = subprocess.run([sys.executable, str(root / "tools" / "asm.py"), *args],
                            capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def load_tool(root):
    spec = importlib.util.spec_from_file_location("zelda_asm_tool", root / "tools" / "asm.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def find_label(tool, root):
    """The current label that upstream called CheckScreenEdge, with its module and program."""
    prog = tool.Program(str(root))
    base = tool.load_baseline(str(root))
    row = next(b for b in base["labels"] if b["mod"] == 7 and b["name"] == UPSTREAM_LABEL)
    mod = prog.mods[7]
    label = next(l for l in prog.labels_at(mod)[row["pos"]] if l.kind == "global")
    return prog, mod, label


@pytest.fixture
def tree(tmp_path):
    copy = tmp_path / "zelda-asm"
    shutil.copytree(ROOT, copy, ignore=shutil.ignore_patterns("work", "bin", "obj", "ext", "__pycache__"))
    return copy


def edit_line(path, line_no, old, new):
    lines = path.read_text().split("\n")
    assert old in lines[line_no - 1], lines[line_no - 1]
    lines[line_no - 1] = lines[line_no - 1].replace(old, new, 1)
    path.write_text("\n".join(lines))


def test_tree_passes_check():
    code, out = run(ROOT, "check")
    assert code == 0, out


def test_redirected_branch_is_caught(tree):
    tool = load_tool(tree)
    prog, mod, label = find_label(tool, tree)
    branches = ("BCC", "BCS", "BEQ", "BMI", "BNE", "BPL", "BVC", "BVS")
    stmt = next(s for s in mod.stmts[label.pos:] if s.head in branches)
    edit_line(Path(prog.src) / stmt.file, stmt.no, stmt.operand, label.name)      # branch to the routine's start
    code, out = run(tree, "check", "--quick")
    assert code == 1
    assert "%s:%d:" % (prog.rel(stmt.file), stmt.no) in out, out


def test_half_rename_is_caught(tree):
    tool = load_tool(tree)
    prog, mod, label = find_label(tool, tree)
    edit_line(Path(prog.src) / label.file, label.no, label.name + ":", label.name + "Renamed:")
    code, out = run(tree, "check", "--quick")
    assert code == 1
    assert label.name in out, out


def test_changed_operand_value_is_caught(tree):
    tool = load_tool(tree)
    prog, mod, label = find_label(tool, tree)
    stmt = next(s for s in mod.stmts[label.pos:] if s.operand.startswith("#$"))
    value = int(stmt.operand[2:], 16)
    edit_line(Path(prog.src) / stmt.file, stmt.no, stmt.operand, "#$%02X" % ((value + 1) & 0xFF))
    code, out = run(tree, "check", "--quick")
    assert code == 1
    assert "%s:%d:" % (prog.rel(stmt.file), stmt.no) in out, out


def test_where_translates_an_upstream_line():
    tool = load_tool(ROOT)
    _, _, label = find_label(tool, ROOT)
    code, out = run(ROOT, "where", "Z_%02d:3025" % 7)      # upstream bank 7 file, line 3025
    assert code == 0, out
    assert label.name in out, out
