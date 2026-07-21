"""JSON / text I/O + RNG seeding. Copied and pruned from utils/util.py (fb-2s)."""
import json
import os
import random

import numpy as np


def _ensure_parent_dir(file_path):
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(file_path, data):
    _ensure_parent_dir(file_path)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def read_released_prompt(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Missing prompt asset: {file_path}. "
            "Please provide the required prompt asset or the generated intermediate data before rerunning this stage."
        )
    return read_txt(file_path)


def write_txt(file_path, data):
    _ensure_parent_dir(file_path)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(data)


def seed_set(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
