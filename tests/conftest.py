# pylint: disable=all
"""Low-level test fixtures for direct emulator access.

ZeldaFixture wraps retro.make() WITHOUT FrameSkipWrapper or StateChangeWrapper,
giving tests single-frame stepping and raw RAM access. This is for verifying
game state interpretation against the actual NES behavior.
"""

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import triforce  # noqa: E402 - registers custom integrations
import retro      # noqa: E402

from triforce.zelda_game import ZeldaGame, ObjectTables
from tests.asm_addresses import *


class ZeldaFixture:
    """Low-level test fixture with direct emulator access.

    Provides single-frame NES stepping and raw RAM read/write without any
    wrapper chain. Use this for verifying RAM interpretation and timing.
    """

    def __init__(self, savestate="start.state"):
        self.env = retro.make(
            game='Zelda-NES',
            state=savestate,
            inttype=retro.data.Integrations.CUSTOM_ONLY,
            render_mode=None
        )
        self._info = None
        self._frame_count = 0
        self.env.reset()
        self._info = {k: self.env.data.lookup_value(k) for k in self._info_keys()}

    def _info_keys(self):
        """Get all variable names from data.json."""
        # Step once to get info dict with all keys
        action = np.zeros(9, dtype=bool)
        _, _, _, _, info = self.env.step(action)
        return list(info.keys())

    @property
    def ram(self):
        """Returns the full 10KB RAM array. Reads are current; writes require set_value."""
        return self.env.get_ram()

    def get(self, name):
        """Read a named variable via the retro data API."""
        return self.env.data.lookup_value(name)

    def set(self, name, value):
        """Write a named variable via the retro data API."""
        self.env.data.set_value(name, value)

    def step(self, buttons=None):
        """Advance exactly one NES frame.

        Args:
            buttons: list of button indices (BTN_B, BTN_UP, etc.) to press,
                     or None for no input.

        Returns:
            (obs, reward, terminated, truncated, info) from the environment.
        """
        action = np.zeros(9, dtype=bool)
        if buttons:
            for b in buttons:
                action[b] = True

        obs, rew, term, trunc, info = self.env.step(action)
        self._info = info
        self._frame_count += 1
        return obs, rew, term, trunc, info

    def step_n(self, n, buttons=None):
        """Advance N NES frames with the same input. Returns the last info."""
        result = None
        for _ in range(n):
            result = self.step(buttons)
        return result

    def game_state(self):
        """Build a ZeldaGame from the current emulator state."""
        return ZeldaGame(self.env, self._info, self._frame_count)

    def object_tables(self):
        """Read the raw object tables from current RAM."""
        return ObjectTables(self.ram)

    def save(self):
        """Save the full emulator state (for mid-test checkpoints)."""
        return self.env.em.get_state()

    def restore(self, state):
        """Restore a previously saved emulator state."""
        self.env.em.set_state(state)

    def close(self):
        """Clean up the environment."""
        self.env.close()


class RAMWatcher:
    """Records changes to specified RAM addresses across frames.

    Use this to trace state machines frame-by-frame and verify transitions.
    """

    def __init__(self, fixture, addresses):
        """
        Args:
            fixture: ZeldaFixture instance
            addresses: dict of {name: ram_address} to watch
        """
        self.fixture = fixture
        self.addresses = addresses
        self.trace = []

    def snapshot(self):
        """Take a snapshot of watched addresses without stepping."""
        ram = self.fixture.ram
        return {name: int(ram[addr]) for name, addr in self.addresses.items()}

    def step(self, buttons=None):
        """Step one frame and record watched addresses."""
        self.fixture.step(buttons)
        snap = self.snapshot()
        self.trace.append(snap)
        return snap

    def run(self, n_frames, buttons=None):
        """Run N frames, recording each. Returns the full trace."""
        for _ in range(n_frames):
            self.step(buttons)
        return self.trace

    def values(self, name):
        """Get all recorded values for a single watched address."""
        return [snap[name] for snap in self.trace]

    def transitions(self, name):
        """Get only the frames where a value changed, as (frame, old, new) tuples."""
        vals = self.values(name)
        changes = []
        for i in range(1, len(vals)):
            if vals[i] != vals[i - 1]:
                changes.append((i, vals[i - 1], vals[i]))
        return changes

    def clear(self):
        """Reset the trace."""
        self.trace.clear()


# -- Pytest fixtures --

@pytest.fixture
def emu():
    """Provide a ZeldaFixture with the start state."""
    fixture = ZeldaFixture("start.state")
    yield fixture
    fixture.close()

@pytest.fixture(params=["1_44e.state", "1_72e.state", "1_73s.state", "1_74w.state"])
def dungeon_emu(request):
    """Provide a ZeldaFixture for various dungeon rooms."""
    fixture = ZeldaFixture(request.param)
    yield fixture
    fixture.close()
