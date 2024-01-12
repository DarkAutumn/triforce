import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

class ZeldaCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        # CNN architecture for image processing
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate the output size of the CNN
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space['image'].sample()[None]).float()).shape[1]

        # Number of binary features
        num_binary_features = observation_space['features'].shape[0]

        # Linear layer to combine CNN output and binary features
        self.linear = nn.Linear(n_flatten + num_binary_features, features_dim)

    def forward(self, observations):
        image = observations['image']
        features = observations['features']

        # Process the image through the CNN
        image_embedding = self.cnn(image)

        # Convert boolean features to float and reshape
        features = features.float().view(features.size(0), -1)

        # Concatenate the image embedding with the binary features
        combined_features = torch.cat([image_embedding, features], dim=1)

        return self.linear(combined_features)

class ZeldaActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.features_extractor = ZeldaCNNFeatureExtractor(self.observation_space)
