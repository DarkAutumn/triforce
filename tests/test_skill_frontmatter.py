# pylint: disable=all
"""Every agent skill must have frontmatter that parses as strict YAML.

Skill loaders differ in strictness. OMP accepts a malformed plain scalar that
Copilot rejects outright with "mapping values are not allowed in this context",
which silently drops the whole skill. Both shipped skills hit this via an
unquoted ': ' inside the description, so this test parses every skill's
frontmatter the strict way.
"""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = ROOT / ".agents/skills"
SKILL_FILES = sorted(SKILL_DIR.glob("*/SKILL.md"))


def _split_frontmatter(text):
    """Return the frontmatter block, delimited by the first two '---' lines."""
    lines = text.split("\n")
    assert lines[0].strip() == "---", "SKILL.md must open with a '---' frontmatter delimiter"
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "\n".join(lines[1:i])
    raise AssertionError("SKILL.md frontmatter is not closed by a '---' line")


def test_skills_are_discovered():
    # Guard against this whole module silently passing on an empty glob.
    assert SKILL_FILES, f"no SKILL.md files found under {SKILL_DIR}"


@pytest.mark.parametrize("path", SKILL_FILES, ids=lambda p: p.parent.name)
def test_frontmatter_parses_as_strict_yaml(path):
    frontmatter = _split_frontmatter(path.read_text(encoding="utf-8"))
    try:
        parsed = yaml.safe_load(frontmatter)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" at line {mark.line + 2}, column {mark.column + 1}" if mark else ""
        pytest.fail(
            f"{path.parent.name}/SKILL.md frontmatter is not valid YAML{where}: "
            f"{getattr(exc, 'problem', exc)}. The usual cause is an unquoted ': ' "
            f"(colon followed by a space) inside a plain scalar, which YAML reads as "
            f"a nested mapping. Rephrase to avoid ': ', or quote the whole value."
        )
    assert isinstance(parsed, dict), "frontmatter must be a YAML mapping"


@pytest.mark.parametrize("path", SKILL_FILES, ids=lambda p: p.parent.name)
def test_name_and_description_are_present_and_well_formed(path):
    parsed = yaml.safe_load(_split_frontmatter(path.read_text(encoding="utf-8")))

    name = parsed.get("name")
    assert isinstance(name, str) and name.strip(), "frontmatter needs a non-empty 'name'"
    assert name == path.parent.name, (
        f"frontmatter name {name!r} must match its directory {path.parent.name!r}"
    )

    description = parsed.get("description")
    assert isinstance(description, str) and description.strip(), (
        "frontmatter needs a non-empty 'description'; it drives skill matching"
    )
    # A parsed-but-truncated description means YAML swallowed part of the line.
    assert description.rstrip().endswith((".", "!")), (
        f"description looks truncated, ending {description[-40:]!r}"
    )
