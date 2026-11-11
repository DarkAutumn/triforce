"""One version (the ``VERSION`` file) for the package, the native module, pyproject.toml and the Cargo workspace."""

import tomllib
from pathlib import Path

import triforce
from triforce import _native

ROOT = Path(__file__).resolve().parent.parent


def test_every_version_follows_the_version_file():
    version = (ROOT / 'VERSION').read_text(encoding='utf-8').strip()
    fields = version.split('.')
    assert len(fields) == 4 and all(field.isdigit() for field in fields)
    assert triforce.__version__ == version
    assert _native.VERSION == version
    assert tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']['version'] == version
    cargo = tomllib.loads((ROOT / 'Cargo.toml').read_text(encoding='utf-8'))
    assert cargo['workspace']['package']['version'] == '.'.join(fields[:3])
