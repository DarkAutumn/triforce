import json
import os
from typing import List
from dataclasses import dataclass

@dataclass
class ZeldaModelInfo:
    name : str
    path : str
    priority : int

    levels : List[int]
    rooms : List[int]

    training_scenario : str
    iterations : int

class ZeldaModel(ZeldaModelInfo):
    def __init__(self, model_info : ZeldaModelInfo, model):
        super().__init__(**model_info.__dict__)
        self.model = model

def load_model_info():
    model_json_path = os.path.join(os.path.dirname(__file__), "models.json")

    models = []
    with open(model_json_path) as f:
        models_json = json.load(f)
        for model_json in models_json['models']:
            if isinstance(model_json['levels'], int):
                model_json['levels'] = [model_json['levels']]

            if 'rooms' not in model_json:
                model_json['rooms'] = None

            elif isinstance(model_json['rooms'], str):
                model_json['rooms'] = [int(model_json['rooms'], 16)]

            elif isinstance(model_json['rooms'], list):
                model_json['rooms'] = [int(x, 16) for x in model_json['rooms']]

            models.append(ZeldaModelInfo(**model_json))

    return models