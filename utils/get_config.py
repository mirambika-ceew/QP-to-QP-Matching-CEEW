
# config.py
from dataclasses import dataclass
from typing import Dict, List
import yaml
import logging
import os


@dataclass
class AppConfig:
    DATA_DIR: str
    JOB_DATA_PATH: str
    SIMILARITY_DATA_PATH: Dict
    LOG_PATH: str
    MODEL_CONFIG: Dict
    KEY_NAME: Dict
    CACHE_DIR: str
    SUPPORTED_SECTORS: List[str]

    @classmethod
    def from_yaml(cls, config_path: str) -> 'AppConfig':
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

