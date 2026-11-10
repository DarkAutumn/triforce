"""The shared fixtures don't depend on what ran before them (F104): ``models_dir``'s weights ignore the global torch
RNG, and the session states dir is every default states dir from the start of the session."""

import os
import subprocess
import sys

import torch

from triforce import _native, paths, registry

from .conftest import promote_untrained, seeded_network


def _hash(root, name='ow-start'):
    return registry.load_model(registry.promoted_checkpoint(name, root))[0].to_native().hash


def test_models_dir_weights_ignore_the_global_rng(tmp_path):
    torch.manual_seed(1)
    first = promote_untrained(tmp_path / 'a')
    torch.manual_seed(2)
    torch.rand(100)
    second = promote_untrained(tmp_path / 'b')
    assert _hash(first) == _hash(second)
    assert _hash(first) != _hash(first, 'goriya')            # one seed per specialist


def test_seeded_network_leaves_the_global_rng_alone():
    torch.manual_seed(3)
    expected = torch.rand(4)
    torch.manual_seed(3)
    seeded_network(5)
    assert torch.equal(torch.rand(4), expected)


def test_states_dir_is_every_default_states_dir(states_dir):
    assert (states_dir / 'start.state').is_file()
    assert _native.paths()['states_dir'] == states_dir
    assert paths.STATES_DIR == states_dir
    child = subprocess.run([sys.executable, '-c', "from triforce import _native; print(_native.paths()['states_dir'])"],
                           check=True, capture_output=True, text=True, env=os.environ.copy())
    assert child.stdout.strip() == str(states_dir)
