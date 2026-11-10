"""The ROM must be present: without it about 40% of the suite cannot run.  ``TRIFORCE_ALLOW_NO_ROM=1`` turns the
ROM tests (this one included) into skips."""

from .conftest import require_rom, rom_path


def test_rom_is_present():
    require_rom()
    assert rom_path().exists()
