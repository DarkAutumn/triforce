
import pytest
from pydantic import BaseModel, field_validator
from typing import Dict, List, Optional, Union

class ZeldaAIModel(BaseModel):
    """
    Represents a defined AI model for The Legend of Zelda.  Each ZeldaAIModel will have a set of available models,
    which are a trained version of this defined model.
    """
    name : str
    action_space : str
    priority : int

    levels : List[int]
    rooms : List[int]
    requires_enemies : bool
    equipment_required : Optional[List[str]] = None

    training_scenario : str
    iterations : int
    available_models : Dict[Union[str, int], str] # version : path

    @classmethod
    @field_validator('levels', mode='before')
    def list_of_int_validator(cls, value):
        """
        Accepts a list of integers, a single integer, or a string representing a hexadecimal value and returns a list
        of integers.

        Args:
            value: The room value to be validated.

        Returns:
            A list of integers.
        """
        print("hello")
        if isinstance(value, int):
            return [value]

        if isinstance(value, str):
            return [int(value, 16)]

        if isinstance(value, list):
            return [int(x, 16) for x in value]

        return value

data = {'name': 'overworld-sword', 'action_space': 'move-only', 'priority': 2, 'levels': [0], 'rooms': [0x77], 'requires_enemies': False, 'training_scenario': 'overworld-sword', 'iterations': 1000000, 'available_models': {'default': '/home/leculver/git/triforce/models/overworld-sword.zip'}}

ZeldaAIModel(**data)
