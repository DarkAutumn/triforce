"""Defines the ZeldaAIModel class and loads the models and scenarios from triforce.json."""

import json
import os
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, field_validator

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

class ZeldaModelDefinition(BaseModel):
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
    requires_triforce : Optional[int] = None
    equipment_required : Optional[List[str]] = []

    training_scenario : ZeldaScenario
    iterations : int
    @field_validator('training_scenario', mode='before')
    @classmethod
    def training_scenario_validator(cls, value):
        """Gets the scenario from the name."""
        return TRAINING_SCENARIOS.get(value)

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

    def find_available_models(self, path) -> Dict[str, 'ZeldaModelDefinition']:
        """Finds the available models for this model definition in the given path.  Returns a dictionary of name to
        path."""
        available_models = {}
        if path is None:
            return available_models

        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist")

        # Check if model.zip exists
        full_path = os.path.join(path, self.name + '.zip')
        if os.path.exists(full_path):
            available_models['default'] = full_path

        # otherwise, it's a training directory of models
        else:
            dir_name = os.path.join(path, self.name)
            if os.path.isdir(dir_name):
                for filename in os.listdir(dir_name):
                    if filename.endswith('.zip'):
                        if filename.startswith('model_'):
                            iterations = int(filename[6:-4])
                            available_models[iterations] = os.path.join(dir_name, filename)
                        else:
                            available_models[filename[:-4]] = os.path.join(dir_name, filename)

        return available_models

def _load_settings():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(script_dir, 'triforce.json'), encoding='utf-8') as f:
        return json.load(f)

def _load_training_scenarios(settings):
    """Loads the models and scenarios from triforce.json."""
    scenarios = {}
    for scenario in settings['scenarios']:
        scenario = ZeldaScenario(**scenario)
        scenarios[scenario.name] = scenario

    return scenarios

def _load_model_definitions(settings):
    """Loads the models and scenarios from triforce.json."""
    models = {}
    for model in settings['models']:
        models[model['name']] = ZeldaModelDefinition(**model)

    return models

all_settings = _load_settings()
TRAINING_SCENARIOS = _load_training_scenarios(all_settings)
ZELDA_MODELS = _load_model_definitions(all_settings)

__all__ = [ZeldaModelDefinition.__name__, ZeldaScenario.__name__, 'ZELDA_MODELS', 'TRAINING_SCENARIOS']
