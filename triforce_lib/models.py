import json
import os
from typing import List, Optional
from dataclasses import dataclass

from stable_baselines3 import PPO

@dataclass
class ZeldaModelInfo:
    name : str
    path : str
    action_space : bool
    priority : int

    levels : List[int]
    rooms : List[int]
    requires_enemies : bool
    equipment_required : List[str]

    training_scenario : str
    iterations : int

class ZeldaModel(ZeldaModelInfo):
    _models = {}
    _model_infos = []

    def __init__(self, model_info : ZeldaModelInfo, models, model_kinds):
        super().__init__(**model_info.__dict__)
        self.models = models
        self.model_kinds = model_kinds

    @classmethod
    def initialize(cls, models_json):
        model_info = []
        for model_json in models_json:
            if isinstance(model_json['levels'], int):
                model_json['levels'] = [model_json['levels']]

            if 'rooms' not in model_json:
                model_json['rooms'] = None

            elif isinstance(model_json['rooms'], str):
                model_json['rooms'] = [int(model_json['rooms'], 16)]

            elif isinstance(model_json['rooms'], list):
                model_json['rooms'] = [int(x, 16) for x in model_json['rooms']]

            model_info.append(ZeldaModelInfo(**model_json))
        
        cls._model_infos = model_info

    @classmethod
    def load_models(cls, path, **kwargs):
        for model_info in cls._model_infos:
            models = []
            kinds = []
            if not cls.__try_load_model(models, kinds, None, path, model_info.path + '.zip', **kwargs):
                model_path = os.path.join(path, model_info.path)

                if os.path.exists(model_path):
                    cls.__try_load_model(models, kinds, "last", model_path, "last.zip", **kwargs)
                    cls.__try_load_model(models, kinds, "best-score", model_path, "best_score.zip", **kwargs)
                    cls.__try_load_model(models, kinds, "best-reward", model_path, "best_reward.zip", **kwargs)

            cls._models[model_info.name] = ZeldaModel(model_info, models, kinds)
    
    @classmethod
    def __try_load_model(cls, models, kinds, kind, path, subpath, **kwargs):
        fullpath = os.path.join(path, subpath)
        if os.path.exists(fullpath):
            models.append(PPO.load(fullpath, **kwargs))
            kinds.append(kind)
            return True
        
        return False

    @classmethod
    def get(cls, name : str) -> Optional['ZeldaModel']:
        return cls._models.get(name, None)
    
    @classmethod
    def get_loaded_models(cls) -> List['ZeldaModel']:
        return list(cls._models.values())
    
    @classmethod
    def get_model_info(cls) -> List[str]:
        return cls._model_infos