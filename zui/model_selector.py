
from collections import OrderedDict
import math

import torch
from triforce.action_space import ZeldaActionSpace
from triforce.models import ModelDefinition

class ModelSelector:
    """Selects a model from a list of available models."""
    def __init__(self, env, model_path, model_definition : ModelDefinition):
        self._model_path = model_path
        self._model_definition = model_definition
        self._loaded_models = OrderedDict()
        self.action_space : ZeldaActionSpace = env
        while not isinstance(self.action_space , ZeldaActionSpace):
            self.action_space = self.action_space.env

        models = [(self._model_definition.neural_net(env.observation_space, env.action_space), name, path)
                  for name, path in self._model_definition.find_available_models(self._model_path).items()]

        for network, _, path in models:
            network.load(path)

        models.sort(key=lambda x: x[0].steps_trained)

        name = None
        for network, name, path in models:
            assert name is not None
            self._loaded_models[name] = (network, path)

        network = self._model_definition.neural_net(env.observation_space, env.action_space)
        self._loaded_models["untrained"] = (network, "untrained")

        best = self._find_best_model()
        self._loaded_models["best"] = (self._loaded_models[best][0], "best")
        self._curr_index = name
        self._curr = self._loaded_models[self._curr_index if self._curr_index is not None else "untrained"]

    @property
    def model_name(self):
        """The name of the current model."""
        return self._curr_index

    @property
    def model_path(self):
        """The path to the current model."""
        return self._curr[1]

    @property
    def model(self):
        """The current model."""
        return self._curr[0]

    def next(self):
        """Selects the next model."""
        keys = list(self._loaded_models.keys())
        self._curr_index = keys[(keys.index(self._curr_index) + 1) % len(keys)]
        self._curr = self._loaded_models[self._curr_index]

    def previous(self):
        """Selects the previous model."""
        keys = list(self._loaded_models.keys())
        self._curr_index = keys[(keys.index(self._curr_index) - 1) % len(keys)]
        self._curr = self._loaded_models[self._curr_index]

    def _find_best_model(self):
        best_model = "untrained"
        best_score = -math.inf
        for name, (network, _) in self._loaded_models.items():
            if not network.metrics:
                continue

            score = network.metrics.get("success-rate", None)
            if score is not None and score > best_score:
                best_score = score
                best_model = name

        return best_model

    def get_probabilities(self, obs, mask):
        """Returns the probability of every action for the model."""
        logits, value = self.model.forward(obs)

        if mask is not None:
            assert mask.any(dim=-1).all(), "Mask must contain at least one valid action"
            logits = logits.clone()
            invalid_mask = ~mask
            logits[invalid_mask] = -1e9

        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        result = OrderedDict()
        result['value'] = value

        for i, prob in enumerate(probabilities.squeeze()):
            taken = self.action_space.get_action_taken(i)
            result.setdefault(taken.kind, []).append((taken.direction, prob.item()))

        return result
