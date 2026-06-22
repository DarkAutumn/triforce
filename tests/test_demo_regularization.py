# pylint: disable=all
"""Demo-regularization helper tests."""

import torch

from triforce.demo import stack_demo_observations


def test_stack_demo_observations_batches_dict_tensors():
    observations = [
        {"image": torch.tensor([[1, 2], [3, 4]]), "vector": torch.tensor([1.0, 2.0])},
        {"image": torch.tensor([[5, 6], [7, 8]]), "vector": torch.tensor([3.0, 4.0])},
    ]

    stacked = stack_demo_observations(observations, torch.device("cpu"))

    assert stacked["image"].shape == (2, 2, 2)
    assert stacked["vector"].shape == (2, 2)
    assert torch.equal(stacked["image"][1], torch.tensor([[5, 6], [7, 8]]))
