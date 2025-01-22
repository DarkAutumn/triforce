from functools import reduce
import inspect
import operator
import pickle
import sys
import json
import os
from typing import List
from pydantic import BaseModel, field_validator
import torch
from torch import nn
import torch.distributions as dist
import numpy as np
from gymnasium.spaces import Dict

class Network(nn.Module):
    """The base class of neural networks used for PPO training."""
    base : nn.Module
    action_net : nn.Module
    value_net : nn.Module

    def __init__(self, base_network : nn.Module, obs_space, action_space):
        super().__init__()
        self.observation_space = obs_space
        self.action_space = action_space
        self.steps_trained = 0
        self.episodes_evaluated = 0
        self.metrics : Dict[str, float] = {}

        self.base = base_network
        self.action_net = self.layer_init(nn.Linear(64, action_space.n), std=0.01)
        self.value_net = self.layer_init(nn.Linear(64, 1), std=1.0)

    def forward(self, obs):
        """Forward pass."""
        obs = self._unsqueeze(obs)

        if isinstance(obs, dict):
            inputs = []
            for key in obs:
                inputs.append(obs[key])
            x = self.base(*inputs)
        else:
            x = self.base(obs)

        action = self.action_net(x)
        value = self.value_net(x)
        return action, value

    def _unsqueeze(self, obs):
        """Unsqueeze the observation."""
        if isinstance(obs, dict):
            for key in obs:
                if obs[key].shape == self.observation_space[key].shape:
                    obs[key] = obs[key].unsqueeze(0)
        else:
            if obs.shape == self.observation_space.shape:
                obs = obs.unsqueeze(0)

        return obs

    def get_action_and_value(self, obs, mask, actions=None, deterministic=False):
        """Gets the action, logprob, entropy, and value."""
        logits, value = self.forward(obs)

        # mask out invalid actions
        if mask is not None:
            assert mask.any(dim=-1).all(), "Mask must contain at least one valid action"
            logits = logits.clone()
            invalid_mask = ~mask
            logits[invalid_mask] = -1e9

            # distribution for entropy calculation
        distribution = dist.Categorical(logits=logits)
        entropy = distribution.entropy()

        # sample an action if not provided
        if actions is None:
            if deterministic:
                actions = logits.argmax(dim=-1)
            else:
                actions = distribution.sample()

        log_prob = distribution.log_prob(actions)

        # value has shape [batch_size, 1], flatten if needed
        return actions, log_prob, entropy, value.view(-1)

    def get_value(self, obs):
        """Get value estimate."""
        _, value = self.forward(obs)
        return value.view(-1)

    def get_action(self, obs, mask = None, deterministic = False):
        """Get the action from the observation."""
        action, _, _, _ = self.get_action_and_value(obs, mask, deterministic=deterministic)
        return action

    @staticmethod
    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        """Initialize a linear layer."""
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def save(self, path):
        """Save the network to a file."""
        save_data = {
            "model_state_dict": self.state_dict(),
            "steps_trained": self.steps_trained,
            "episodes_evaluated" : self.episodes_evaluated,
            "metrics": pickle.dumps(self.metrics) if self.metrics else None,
            "obs_space": self.observation_space,
            "action_space": self.action_space,
        }

        torch.save(save_data, path)

    def load(self, path) -> 'Network':
        """Load the network from a file."""
        save_data = torch.load(path)

        self.load_state_dict(save_data["model_state_dict"])
        self.steps_trained = save_data["steps_trained"]
        self.episodes_evaluated = save_data.get("episodes_evaluated", 0)
        metrics_pickled = save_data.get("metrics")
        self.metrics = pickle.loads(metrics_pickled) if metrics_pickled else {}

        if self.observation_space != save_data["obs_space"]:
            raise ValueError("Mismatch in observation space!")

        if self.action_space != save_data["action_space"]:
            raise ValueError("Mismatch in action space!")

        return self

    @staticmethod
    def load_metrics(path):
        """Load the metrics from a file."""
        save_data = torch.load(path)
        metrics_pickled = save_data.get("metrics")
        metrics = pickle.loads(metrics_pickled) if metrics_pickled else {}
        episodes_evaluated = save_data.get("episodes_evaluated", 0)
        return metrics, episodes_evaluated

    @staticmethod
    def load_spaces(path):
        """Load the observation and action spaces from a file."""
        save_data = torch.load(path)
        return save_data["obs_space"], save_data["action_space"]

class NatureCNN(nn.Module):
    """Simple CNN."""
    def __init__(self, input_channels=1, linear_output_size=256):
        super().__init__()
        self.cnn = nn.Sequential(
            Network.layer_init(nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            Network.layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            Network.layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten()
        )

        dummy_input = torch.zeros(1, input_channels, 128, 128)  # Example input shape
        with torch.no_grad():
            cnn_out = self.cnn(dummy_input)
        cnn_output_size = cnn_out.shape[1]

        self.linear = nn.Sequential(
            Network.layer_init(nn.Linear(cnn_output_size, linear_output_size)),
            nn.ReLU()
        )

    def forward(self, tensor):
        """Forward pass."""
        cnn_out = self.cnn(tensor)
        linear_out = self.linear(cnn_out)
        return linear_out

class CombinedExtractor(nn.Module):
    """Combine image, vectors, and information."""
    def __init__(self, image_channels=1, image_linear_size=256, vectors_size=18, info_size=12):
        super().__init__()
        self.image_extractor = NatureCNN(input_channels=image_channels, linear_output_size=image_linear_size)
        self.flatten_info = nn.Flatten()
        self.flatten_vectors = nn.Flatten()
        self.vectors_size = vectors_size
        self.info_size = info_size

    def forward(self, image, vectors, information):
        """Forward pass."""
        image_features = self.image_extractor(image)
        vectors_features = self.flatten_vectors(vectors)
        info_features = self.flatten_info(information)
        combined_features = torch.cat([image_features, vectors_features, info_features], dim=1)
        return combined_features

class MlpExtractor(nn.Module):
    """MLP for policy and value."""
    def __init__(self, input_size, policy_hidden_size=64, value_hidden_size=64):
        super().__init__()
        self.policy_net = nn.Sequential(
            Network.layer_init(nn.Linear(input_size, policy_hidden_size)),
            nn.Tanh(),
            Network.layer_init(nn.Linear(policy_hidden_size, policy_hidden_size)),
            nn.Tanh()
        )
        self.value_net = nn.Sequential(
            Network.layer_init(nn.Linear(input_size, value_hidden_size)),
            nn.Tanh(),
            Network.layer_init(nn.Linear(value_hidden_size, value_hidden_size)),
            nn.Tanh()
        )

    def forward(self, combined_features):
        """Forward pass."""
        policy_features = self.policy_net(combined_features)
        value_features = self.value_net(combined_features)
        return policy_features, value_features

class SharedNatureAgent(Network):
    """Actor-critic policy with multiple inputs + action masking."""
    def __init__(self, obs_space : Dict, action_space):
        channels, height, width = obs_space["image"].shape
        image_linear_size = height + width

        vector_size = reduce(operator.mul, obs_space["vectors"].shape)
        info_size = reduce(operator.mul, obs_space["information"].shape)

        combined_input_size = image_linear_size + vector_size + info_size

        base = CombinedExtractor(
            image_channels=channels,
            image_linear_size=image_linear_size,
            vectors_size=vector_size,
            info_size=info_size
        )

        super().__init__(base, obs_space, action_space)

        self.mlp_extractor = MlpExtractor(input_size=combined_input_size)
        self.action_net = Network.layer_init(nn.Linear(64, action_space.n), std=0.01)
        self.value_net = Network.layer_init(nn.Linear(64, 1), std=1.0)

    def forward(self, obs):
        obs = self._unsqueeze(obs)
        combined_features = self.base(obs['image'], obs['vectors'], obs['information'])
        policy_features, value_features = self.mlp_extractor(combined_features)

        action_logits = self.action_net(policy_features)
        value = self.value_net(value_features)
        return action_logits, value

def create_network(network, obs_space, action_space):
    """Create a network from a class or instance."""
    if isinstance(network, type) and issubclass(network, Network):
        network = network(obs_space, action_space)
    elif not isinstance(network, Network):
        raise ValueError("network must be a Network or a Network subclass")

    return network

def _init_models():
    # Get all classes defined in this module
    result = {}
    current_module = sys.modules[__name__]
    for cls_name, cls_obj in inspect.getmembers(current_module, inspect.isclass):
        # Check if the class subclasses 'Network' and matches the name
        if issubclass(cls_obj, Network) and cls_obj is not Network:
            result[cls_name] = cls_obj

    return result


NEURAL_NETWORK_DEFINITIONS = _init_models()

def register_neural_network(name, model_class):
    """Register a neural network definition."""
    if not issubclass(model_class, Network):
        raise ValueError("model_class must be a subclass of Network")

    if name in NEURAL_NETWORK_DEFINITIONS:
        raise ValueError(f"Model {name} already exists")

    if model_class == Network:
        raise ValueError("Cannot register the base Network")

    NEURAL_NETWORK_DEFINITIONS[name] = model_class

def get_neural_network(name):
    """Get a model by name."""
    return NEURAL_NETWORK_DEFINITIONS[name]


class ModelDefinition(BaseModel):
    """
    Represents a defined AI model for The Legend of Zelda.  Each ZeldaAIModel will have a set of available models,
    which are a trained version of this defined model.
    """
    name : str
    neural_net : type
    action_space : List[str]

    @field_validator('neural_net', mode='before')
    @classmethod
    def neural_net_validator(cls, value):
        """Gets the class from the name."""
        return get_neural_network(value)

    def find_available_models(self, path):
        """Finds the available models for this model definition in the given path.  Returns a dictionary of name to
        path."""
        available_models = {}
        if path is None:
            return available_models

        if not os.path.exists(path):
            return available_models

        # Check if .pt
        full_path = os.path.join(path, self.name + '.pt')
        if os.path.exists(full_path):
            available_models['default'] = full_path

        # otherwise, it's a training directory of models
        else:
            dir_name = os.path.join(path, self.name)
            if os.path.isdir(dir_name):
                for filename in os.listdir(dir_name):
                    if filename.endswith('.pt'):
                        i = filename.find('_')
                        if i > 0:
                            iterations = int(filename[i+1:-3])
                            available_models[iterations] = os.path.join(dir_name, filename)
                        else:
                            available_models[filename[:-3]] = os.path.join(dir_name, filename)

        return available_models

    @staticmethod
    def get_all_models():
        """Loads the models and scenarios from triforce.json."""
        models = {}
        script_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(script_dir, 'triforce.json'), encoding='utf-8') as f:
            for model in json.load(f)["models"]:
                model = ModelDefinition(**model)
                models[model.name] = model

        return models

    @staticmethod
    def get(name, default=None):
        """Get a model by name."""
        return ModelDefinition.get_all_models().get(name, default)

__all__ = [
    Network.__name__,
    SharedNatureAgent.__name__,
    register_neural_network.__name__,
    get_neural_network.__name__
    ]
