import inspect
import pickle
import sys
import os
from typing import List
import torch
from torch import nn
import torch.distributions as dist
import numpy as np
import yaml
from gymnasium.spaces import Dict

class Network(nn.Module):
    """The base class of neural networks used for PPO training."""
    base : nn.Module
    action_net : nn.Module
    value_net : nn.Module
    is_multihead = False

    def __init__(self, base_network : nn.Module, obs_space, action_space,
                 model_kind=None, action_space_name=None, base_output_size=64):
        super().__init__()
        self.observation_space = obs_space
        self.action_space = action_space
        self.model_kind = model_kind
        self.action_space_name = action_space_name
        self.steps_trained = 0
        self.episodes_evaluated = 0
        self.metrics : dict[str, float] = {}

        self.base = base_network
        self.action_net = self.layer_init(nn.Linear(base_output_size, action_space.n), std=0.01)
        self.value_net = self.layer_init(nn.Linear(base_output_size, 1), std=1.0)

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
            obs = obs.copy()
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
            "model_kind": self.model_kind,
            "action_space_name": self.action_space_name,
        }

        torch.save(save_data, path)

    def load(self, path) -> 'Network':
        """Load the network from a file."""
        save_data = torch.load(path, weights_only=False)

        self.load_state_dict(save_data["model_state_dict"])
        self.steps_trained = save_data["steps_trained"]
        self.episodes_evaluated = save_data.get("episodes_evaluated", 0)
        self.model_kind = save_data.get("model_kind")
        self.action_space_name = save_data.get("action_space_name")
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
        save_data = torch.load(path, weights_only=False)
        metrics_pickled = save_data.get("metrics")
        metrics = pickle.loads(metrics_pickled) if metrics_pickled else {}
        episodes_evaluated = save_data.get("episodes_evaluated", 0)
        return metrics, episodes_evaluated

    @staticmethod
    def load_spaces(path):
        """Load the observation and action spaces from a file."""
        save_data = torch.load(path, weights_only=False)
        return save_data["obs_space"], save_data["action_space"]

    @staticmethod
    def load_metadata(path):
        """Load model_kind and action_space_name from a saved .pt file."""
        save_data = torch.load(path, weights_only=False)
        metrics_pickled = save_data.get("metrics")
        metrics = pickle.loads(metrics_pickled) if metrics_pickled else {}
        return {
            "model_kind": save_data.get("model_kind"),
            "action_space_name": save_data.get("action_space_name"),
            "steps_trained": save_data.get("steps_trained", 0),
            "obs_space": save_data.get("obs_space"),
            "action_space": save_data.get("action_space"),
            "metrics": metrics,
        }

class NatureCNN(nn.Module):
    """Simple CNN that adjusts to input size dynamically."""
    def __init__(self, input_channels=1, input_height=128, input_width=128, linear_output_size=256):
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

        # Use dynamic input size for figuring out output shape
        dummy_input = torch.zeros(1, input_channels, input_height, input_width)
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

class MlpExtractor(nn.Module):
    """MLP for policy and value."""
    def __init__(self, input_size, policy_hidden_size=256, value_hidden_size=256):
        super().__init__()
        self.policy_output_dim = policy_hidden_size
        self.value_output_dim = value_hidden_size

        self.policy_net = nn.Sequential(
            Network.layer_init(nn.Linear(input_size, policy_hidden_size)),
            nn.LayerNorm(policy_hidden_size),
            nn.ReLU(),
            Network.layer_init(nn.Linear(policy_hidden_size, policy_hidden_size)),
            nn.LayerNorm(policy_hidden_size),
            nn.ReLU(),
            Network.layer_init(nn.Linear(policy_hidden_size, policy_hidden_size)),
            nn.LayerNorm(policy_hidden_size),
            nn.Tanh(),
        )
        self.value_net = nn.Sequential(
            Network.layer_init(nn.Linear(input_size, value_hidden_size)),
            nn.LayerNorm(value_hidden_size),
            nn.ReLU(),
            Network.layer_init(nn.Linear(value_hidden_size, value_hidden_size)),
            nn.LayerNorm(value_hidden_size),
            nn.ReLU(),
            Network.layer_init(nn.Linear(value_hidden_size, value_hidden_size)),
            nn.LayerNorm(value_hidden_size),
            nn.ReLU(),
        )

    def forward(self, combined_features):
        """Forward pass."""
        policy_features = self.policy_net(combined_features)
        value_features = self.value_net(combined_features)
        return policy_features, value_features

class EntityAttentionEncoder(nn.Module):
    """Self-attention encoder for a unified entity list.

    Processes a flat list of entity slots (enemies, items, projectiles, treasure)
    through a transformer encoder, masking empty slots, then mean-pools present
    entities into a fixed-size output vector.
    """
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(self, num_entity_types, continuous_features=9, embedding_dim=8,
                 d_model=64, num_heads=4, num_layers=1, ff_dim=128, output_dim=64):
        super().__init__()
        self.type_embedding = nn.Embedding(num_entity_types, embedding_dim)
        input_dim = continuous_features + embedding_dim
        self.input_proj = Network.layer_init(nn.Linear(input_dim, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim,
            batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                                enable_nested_tensor=False)
        self.output_proj = Network.layer_init(nn.Linear(d_model, output_dim))
        self.output_dim = output_dim

    def forward(self, entity_features, entity_types):
        """Forward pass.

        Args:
            entity_features: (batch, slots, continuous_features)
            entity_types: (batch, slots) long tensor of type IDs
        Returns:
            (batch, output_dim) pooled entity representation
        """
        type_embeds = self.type_embedding(entity_types.long())
        combined = torch.cat([entity_features, type_embeds], dim=-1)
        projected = self.input_proj(combined)

        # Mask empty slots (presence == 0) so they don't attend or get attended to
        empty_mask = entity_features[:, :, 0] == 0

        # Prevent all-masked rows: unmask slot 0 so attention softmax never sees all -inf
        all_empty = empty_mask.all(dim=1)
        safe_mask = empty_mask.clone()
        safe_mask[all_empty, 0] = False

        attended = self.transformer(projected, src_key_padding_mask=safe_mask)

        # Mean pool over genuinely present entities (use original mask, not safe_mask)
        present = (~empty_mask).unsqueeze(-1).float()
        num_present = present.sum(dim=1).clamp(min=1)
        pooled = (attended * present).sum(dim=1) / num_present

        return self.output_proj(pooled)


class CombinedExtractor(nn.Module):
    """Combined extractor for CNN image + entity attention + boolean info."""
    # pylint: disable=too-many-positional-arguments, too-many-arguments
    def __init__(
        self,
        image_channels: int,
        input_height: int,
        input_width: int,
        num_entity_types: int,
        entity_features: int = 9,
        embedding_dim: int = 8,
        attention_heads: int = 4,
        attention_layers: int = 1,
        attention_output_dim: int = 64,
        info_size: int = 14,
        image_linear_size: int = 256,
    ):
        super().__init__()

        self.image_extractor = NatureCNN(
            input_channels=image_channels,
            input_height=input_height,
            input_width=input_width,
            linear_output_size=image_linear_size
        )

        self.entity_encoder = EntityAttentionEncoder(
            num_entity_types=num_entity_types,
            continuous_features=entity_features,
            embedding_dim=embedding_dim,
            num_heads=attention_heads,
            num_layers=attention_layers,
            output_dim=attention_output_dim
        )

        self.info_size = info_size
        self.output_dim = image_linear_size + attention_output_dim + info_size

    def forward(self, image: torch.Tensor, entities: torch.Tensor,
                entity_types: torch.Tensor, information: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        img_out = self.image_extractor(image)
        entity_out = self.entity_encoder(entities, entity_types)
        info_float = information.float()
        return torch.cat([img_out, entity_out, info_float], dim=1)

class MultiHeadAgent(Network):
    """Two-head action decomposition: action_type (K) + direction (4).

    Instead of a flat Discrete(N) action space, this agent uses MultiDiscrete([K, 4])
    where K = number of action types. The two heads share the same CNN + MLP backbone
    but output independent logits. Joint log-prob: log π(a|s) = log π_type + log π_dir.
    """
    is_multihead = True

    # pylint: disable=super-init-not-called
    def __init__(self, obs_space: Dict, action_space, model_kind=None, action_space_name=None):
        channels, height, width = obs_space["image"].shape
        image_linear_size = 256

        base_network = CombinedExtractor(
            image_channels=channels,
            input_height=height,
            input_width=width,
            num_entity_types=int(obs_space["entity_types"].nvec[0]),
            entity_features=obs_space["entities"].shape[1],
            info_size=obs_space["information"].n,
            image_linear_size=image_linear_size
        )

        # Bypass Network.__init__ because MultiDiscrete has no .n attribute.
        # Manually set all attributes that Network.__init__ would create.
        # pylint: disable=non-parent-init-called
        nn.Module.__init__(self)
        self.observation_space = obs_space
        self.action_space = action_space
        self.model_kind = model_kind
        self.action_space_name = action_space_name
        self.steps_trained = 0
        self.episodes_evaluated = 0
        self.metrics = {}

        self.base = base_network
        self.mlp_extractor = MlpExtractor(input_size=self.base.output_dim)

        # Two action heads: action_type (K) and direction (4)
        num_action_types = int(action_space.nvec[0])
        self.action_type_net = self.layer_init(
            nn.Linear(self.mlp_extractor.policy_output_dim, num_action_types), std=0.01)
        self.direction_net = self.layer_init(
            nn.Linear(self.mlp_extractor.policy_output_dim, 4), std=0.01)

        # Single shared value head
        self.value_net = self.layer_init(
            nn.Linear(self.mlp_extractor.value_output_dim, 1), std=1.0)

        # Placeholder so Network.base class annotation is satisfied
        self.action_net = nn.Identity()

    def forward(self, obs):
        """Returns (action_type_logits, direction_logits, value)."""
        obs = self._unsqueeze(obs)

        combined_features = self.base(
            image=obs["image"],
            entities=obs["entities"],
            entity_types=obs["entity_types"],
            information=obs["information"]
        )

        policy_features, value_features = self.mlp_extractor(combined_features)

        action_type_logits = self.action_type_net(policy_features)
        direction_logits = self.direction_net(policy_features)
        value = self.value_net(value_features)
        return action_type_logits, direction_logits, value

    def get_action_and_value(self, obs, mask, actions=None, deterministic=False):
        """Gets action, joint log-prob, summed entropy, and value.

        Args:
            obs: Dict observation.
            mask: Concatenated [batch, K+4] mask — first K for action type, last 4 for direction.
            actions: Optional [batch, 2] tensor of (action_type, direction) indices.
            deterministic: If True, use argmax instead of sampling.

        Returns:
            (actions [batch, 2], log_prob [batch], entropy [batch], value [batch])
        """
        action_type_logits, direction_logits, value = self.forward(obs)
        num_action_types = int(self.action_space.nvec[0])

        # Per-head masking from concatenated mask
        if mask is not None:
            action_type_mask = mask[..., :num_action_types]
            direction_mask = mask[..., num_action_types:]

            bad_type = ~action_type_mask.any(dim=-1)
            bad_dir = ~direction_mask.any(dim=-1)
            if bad_type.any() or bad_dir.any():
                idx = (bad_type | bad_dir).nonzero(as_tuple=True)[0][0].item()
                info_vec = obs['info'][idx].tolist() if 'info' in obs else 'N/A'
                raise ValueError(
                    f"Empty action mask in batch element {idx}.\n"
                    f"  action_type_mask={action_type_mask[idx].tolist()}\n"
                    f"  direction_mask={direction_mask[idx].tolist()}\n"
                    f"  full_mask={mask[idx].tolist()}\n"
                    f"  num_action_types={num_action_types}\n"
                    f"  nvec={self.action_space.nvec.tolist()}\n"
                    f"  info={info_vec}"
                )

            action_type_logits = action_type_logits.clone()
            action_type_logits[~action_type_mask] = -1e9

            direction_logits = direction_logits.clone()
            direction_logits[~direction_mask] = -1e9

        type_dist = dist.Categorical(logits=action_type_logits)
        dir_dist = dist.Categorical(logits=direction_logits)

        if actions is None:
            if deterministic:
                type_action = action_type_logits.argmax(dim=-1)
                dir_action = direction_logits.argmax(dim=-1)
            else:
                type_action = type_dist.sample()
                dir_action = dir_dist.sample()
            actions = torch.stack([type_action, dir_action], dim=-1)

        # Joint log-prob: log π(a|s) = log π_type(a_type|s) + log π_dir(a_dir|s)
        log_prob = type_dist.log_prob(actions[..., 0]) + dir_dist.log_prob(actions[..., 1])

        # Entropy: sum of per-head entropies
        entropy = type_dist.entropy() + dir_dist.entropy()

        return actions, log_prob, entropy, value.view(-1)

    def get_value(self, obs):
        """Get value estimate."""
        _, _, value = self.forward(obs)
        return value.view(-1)

    def get_action(self, obs, mask=None, deterministic=False):
        """Get the action from the observation."""
        action, _, _, _ = self.get_action_and_value(obs, mask, deterministic=deterministic)
        return action

    def get_entropy_details(self, obs, mask):
        """Returns per-head entropy means for Tensorboard logging.

        Used by PPO._optimize to log action_type and direction entropy separately,
        enabling diagnosis of per-head entropy collapse.
        """
        action_type_logits, direction_logits, _ = self.forward(obs)
        num_action_types = int(self.action_space.nvec[0])

        if mask is not None:
            action_type_mask = mask[..., :num_action_types]
            direction_mask = mask[..., num_action_types:]
            action_type_logits = action_type_logits.clone()
            action_type_logits[~action_type_mask] = -1e9
            direction_logits = direction_logits.clone()
            direction_logits[~direction_mask] = -1e9

        type_dist = dist.Categorical(logits=action_type_logits)
        dir_dist = dist.Categorical(logits=direction_logits)

        return {
            "entropy/action_type": type_dist.entropy().mean().item(),
            "entropy/direction": dir_dist.entropy().mean().item(),
        }


class SharedNatureAgent(Network):
    """Actor-critic policy with multiple inputs, action masking, and shared CNN."""
    def __init__(self, obs_space: Dict, action_space, model_kind=None, action_space_name=None):
        channels, height, width = obs_space["image"].shape

        # We'll do a straightforward approach: let the user define a 'linear_output_size'.
        # We'll keep 'image_linear_size' for the final MLP size.
        image_linear_size = 256  # or some fixed size, ignoring (height+width) approach

        super().__init__(
            base_network=CombinedExtractor(
                image_channels=channels,
                input_height=height,
                input_width=width,
                num_entity_types=int(obs_space["entity_types"].nvec[0]),
                entity_features=obs_space["entities"].shape[1],
                info_size=obs_space["information"].n,
                image_linear_size=image_linear_size
            ),
            obs_space=obs_space,
            action_space=action_space,
            model_kind=model_kind,
            action_space_name=action_space_name
        )

        # Now we create an MLP for policy/value.
        self.mlp_extractor = MlpExtractor(input_size=self.base.output_dim)

        # Overwrite action_net and value_net to match MLP output dimensions.
        self.action_net = self.layer_init(
            nn.Linear(self.mlp_extractor.policy_output_dim, action_space.n), std=0.01)
        self.value_net = self.layer_init(
            nn.Linear(self.mlp_extractor.value_output_dim, 1), std=1.0)

    def forward(self, obs):
        obs = self._unsqueeze(obs)

        combined_features = self.base(
            image=obs["image"],
            entities=obs["entities"],
            entity_types=obs["entity_types"],
            information=obs["information"]
        )

        policy_features, value_features = self.mlp_extractor(combined_features)

        action_logits = self.action_net(policy_features)
        value = self.value_net(value_features)
        return action_logits, value

def create_network(network, obs_space, action_space, model_kind=None, action_space_name=None):
    """Create a network from a class or instance."""
    if isinstance(network, type) and issubclass(network, Network):
        network = network(obs_space, action_space, model_kind=model_kind, action_space_name=action_space_name)
    elif not isinstance(network, Network):
        raise ValueError("network must be a Network or a Network subclass")

    return network

def _init_models():
    # Get all classes defined in this module
    result = {}
    current_module = sys.modules[__name__]
    for cls_name, cls_obj in inspect.getmembers(current_module, inspect.isclass):
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

def _load_triforce_yaml():
    """Load and return the parsed triforce.yaml."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(script_dir, 'triforce.yaml'), encoding='utf-8') as f:
        return yaml.safe_load(f)


class ActionSpaceDefinition:
    """A named action space from triforce.yaml."""
    def __init__(self, name: str, actions: List[str], default: bool = False):
        self.name = name
        self.actions = actions
        self.default = default

    @staticmethod
    def get_all():
        """Load all action space definitions from triforce.yaml."""
        result = {}
        data = _load_triforce_yaml()
        for name, entry in data["action-spaces"].items():
            result[name] = ActionSpaceDefinition(
                name=name,
                actions=entry["actions"],
                default=entry.get("default", False),
            )
        return result

    @staticmethod
    def get(name):
        """Get an action space by name."""
        return ActionSpaceDefinition.get_all()[name]

    @staticmethod
    def get_default():
        """Get the default action space."""
        for asd in ActionSpaceDefinition.get_all().values():
            if asd.default:
                return asd
        raise ValueError("No default action space defined in triforce.yaml")


class ModelKindDefinition:
    """A named model kind from triforce.yaml mapping to a Network subclass."""
    def __init__(self, name: str, network_class: type, default: bool = False):
        self.name = name
        self.network_class = network_class
        self.default = default

    @staticmethod
    def get_all():
        """Load all model kind definitions from triforce.yaml."""
        result = {}
        data = _load_triforce_yaml()
        for name, entry in data["model-kinds"].items():
            result[name] = ModelKindDefinition(
                name=name,
                network_class=get_neural_network(entry["class"]),
                default=entry.get("default", False),
            )
        return result

    @staticmethod
    def get(name):
        """Get a model kind by name."""
        return ModelKindDefinition.get_all()[name]

    @staticmethod
    def get_default():
        """Get the default model kind."""
        for mkd in ModelKindDefinition.get_all().values():
            if mkd.default:
                return mkd
        raise ValueError("No default model kind defined in triforce.yaml")


__all__ = [
    Network.__name__,
    MultiHeadAgent.__name__,
    SharedNatureAgent.__name__,
    register_neural_network.__name__,
    get_neural_network.__name__,
    ActionSpaceDefinition.__name__,
    ModelKindDefinition.__name__,
    ]
