"""Defines the ZeldaAIModel class and loads the models and scenarios from triforce.json."""

import json
import os
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, field_validator
from stable_baselines3 import PPO

class ZeldaScenario(BaseModel):
    """A scenario in the game to train on.  This is a combination of critics and end conditions."""
    name : str
    description : str
    critic : str
    reward_overrides : Optional[Dict[str, Union[int, float, None]]] = {}
    end_conditions : List[str]
    start : List[str]
    data : Optional[Dict[str, int]] = {}
    fixed : Optional[Dict[str, int]] = {}

    @classmethod
    def get_all_scenarios(cls) -> List['ZeldaScenario']:
        """Returns all scenarios."""
        return list(ALL_SCENARIOS.values())

    @classmethod
    def get(cls, name) -> 'ZeldaScenario':
        """Gets the scenario from the name."""
        return ALL_SCENARIOS.get(name)

class ZeldaAIModel(BaseModel):
    """
    Represents a defined AI model for The Legend of Zelda.  Each ZeldaAIModel will have a set of available models,
    which are a trained version of this defined model.
    """
    name : str
    action_space : str
    priority : int

    levels : List[int]
    rooms : Optional[List[int]] = None
    requires_enemies : bool
    equipment_required : Optional[List[str]] = []

    training_scenario : ZeldaScenario
    iterations : int
    available_models : Dict[Union[str, int], str] # version : path

    @field_validator('training_scenario', mode='before')
    @classmethod
    def get(cls, value):
        """Gets the scenario from the name."""
        return ALL_SCENARIOS.get(value)

    @field_validator('levels', 'rooms', mode='before')
    @classmethod
    def list_of_int_validator(cls, value):
        """
        Accepts a list of integers, a single integer, or a string representing a hexadecimal value and returns a list
        of integers.

        Args:
            value: The room value to be validated.

        Returns:
            A list of integers.
        """
        if isinstance(value, int):
            return [value]

        if isinstance(value, str):
            return [int(value, 16)]

        if isinstance(value, list):
            return [int(x, 16) for x in value]

        return value

    def load(self, version):
        """Loads the specified version of the model."""
        assert version in self.available_models, f"Model kind {version} is not available"

        if self.name not in LOADED_MODELS:
            LOADED_MODELS[self.name] = {}

        if version not in LOADED_MODELS[self.name]:
            LOADED_MODELS[self.name][version] = PPO.load(self.available_models[version])

        return LOADED_MODELS[self.name][version]

    def create(self, **values):
        """Creates a new instance of the model."""
        return PPO('MultiInputPolicy', **values)

    @classmethod
    def initialize(cls, path : Optional[str] = None) -> List['ZeldaAIModel']:
        """Creates a ZeldaAIModel from the specified directory.  Meant to be used internally."""
        result = []

        if path is not None and not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist")

        for model_name, defined_model in DEFINED_MODELS.items():
            available_models = cls.__get_available_models(path, model_name)
            result.append(cls(**defined_model, available_models=available_models))

        return result

    @classmethod
    def __get_available_models(cls, path, model_name):
        available_models = {}

        if path is None:
            return available_models

        # Check if model.zip exists
        full_path = os.path.join(path, model_name + '.zip')
        if os.path.exists(full_path):
            available_models['default'] = full_path

        # otherwise, it's a training directory of models
        else:
            dir_name = os.path.join(path, model_name)
            if os.path.isdir(dir_name):
                for filename in os.listdir(dir_name):
                    if filename.endswith('.zip'):
                        if filename.startswith('model_'):
                            iterations = int(filename[6:-4])
                            available_models[iterations] = os.path.join(dir_name, filename)
                        else:
                            available_models[filename[:-4]] = os.path.join(dir_name, filename)

        return available_models


def load_models_and_scenarios():
    """Loads the models and scenarios from triforce.json."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    models = {}
    scenarios = {}
    with open(os.path.join(script_dir, 'triforce.json'), encoding='utf-8') as f:
        all_settings = json.load(f)

    for model in all_settings['models']:
        models[model['name']] = model

    for scenario in all_settings['scenarios']:
        scenario = ZeldaScenario(**scenario)
        scenarios[scenario.name] = scenario

    return models, scenarios

DEFINED_MODELS, ALL_SCENARIOS = load_models_and_scenarios()
LOADED_MODELS = {}

__all__ = [ZeldaAIModel.__name__, ZeldaScenario.__name__]
