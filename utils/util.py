import json
import os
import random
import threading
import numpy as np
import httpx
from openai import OpenAI


LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:4142/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o")
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-placeholder")
EMBED_BASE_URL = os.environ.get("EMBED_BASE_URL", "http://localhost:4142/v1")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")

# Shared singleton clients so all worker threads reuse one httpx connection pool
# (HTTP keep-alive). Without this, every LLM call opens a new TCP socket through
# the ssh tunnel and exhausts the ssh client's fd limit ("accept: Too many open files").
_chat_client = None
_embed_client = None
_client_lock = threading.Lock()


def _build_http_client(pool_size):
    limits = httpx.Limits(
        max_connections=pool_size,
        max_keepalive_connections=pool_size,
        keepalive_expiry=300.0,
    )
    return httpx.Client(limits=limits)


def make_chat_client():
    global _chat_client
    if _chat_client is None:
        with _client_lock:
            if _chat_client is None:
                pool = int(os.environ.get("LLM_MAX_WORKERS", "96"))
                _chat_client = OpenAI(
                    base_url=LLM_BASE_URL,
                    api_key=LLM_API_KEY,
                    timeout=90.0,
                    http_client=_build_http_client(pool),
                )
    return _chat_client


def make_embed_client():
    global _embed_client
    if _embed_client is None:
        with _client_lock:
            if _embed_client is None:
                pool = int(os.environ.get("LLM_MAX_WORKERS", "96"))
                _embed_client = OpenAI(
                    base_url=EMBED_BASE_URL,
                    api_key=LLM_API_KEY,
                    timeout=120.0,
                    http_client=_build_http_client(pool),
                )
    return _embed_client

def _ensure_parent_dir(file_path):
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def write_json(file_path, data):
    _ensure_parent_dir(file_path)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def read_txt(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data

def read_released_prompt(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Missing prompt asset: {file_path}. "
            "Please provide the required prompt asset or the generated intermediate data before rerunning this stage."
        )
    return read_txt(file_path)

def write_txt(file_path, data):
    _ensure_parent_dir(file_path)
    with open(file_path, 'w') as file:
        file.write(data)

def seed_set(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def is_json(text):
    try:
        json.loads(text)
        return True
    except:
        return False

def normalized_sampling(data):
    normalized_data = data / np.sum(data)
    sampled_index = np.random.choice(len(data), size=1, p=normalized_data)
    return sampled_index[0], normalized_data

def store_dsl(feature_data_path, dsl_store_path):
    dsl = {}
    feature_data = read_json(feature_data_path)
    for opcode in feature_data:
        dsl[opcode] = []
        patterns = []
        max_label_1 = 0
        for hierarchy in feature_data[opcode]:
            if "label-1" in hierarchy:
                max_label_1 = max(max_label_1, int(hierarchy["label-1"])+1)
        for i in range(max_label_1):
            patterns.append({
                "Precond": [],
                "Postcond": []
            })
        for hierarchy in feature_data[opcode]:
            if "input_flow_units" in hierarchy["hierarchy-1"] and "label-1" in hierarchy:
                patterns[int(hierarchy["label-1"])]["Precond"].append(hierarchy["hierarchy-1"]["input_flow_units"])
            if "output_flow_units" in hierarchy["hierarchy-1"] and "label-1" in hierarchy:
                patterns[int(hierarchy["label-1"])]["Postcond"].append(hierarchy["hierarchy-1"]["output_flow_units"])
    for pattern in patterns:
        dsl[opcode].append(pattern)
    write_json(dsl_store_path, dsl)
    # Merge label-2 patterns within the same label-1 group.
    for opcode in feature_data:
        new_patterns = []
        for label_1, pattern in enumerate(dsl[opcode]):
            current_label_1_patterns = []
            max_label_2 = 0
            for hierarchy in feature_data[opcode]:
                if "label-1" in hierarchy and int(hierarchy["label-1"]) == label_1 and "label-2" in hierarchy:
                    max_label_2 = max(max_label_2, int(hierarchy["label-2"])+1)
            for i in range(max_label_2):
                current_label_1_patterns.append({
                    "Precond": pattern["Precond"],
                    "Postcond": pattern["Postcond"],
                    "Device": []
                })
            if max_label_2 == 0:
                current_label_1_patterns.append({
                    "Precond": pattern["Precond"],
                    "Postcond": pattern["Postcond"],
                    "Device": []
                })
            for hierarchy in feature_data[opcode]:
                if "label-1" in hierarchy and int(hierarchy["label-1"]) == label_1 and "label-2" in hierarchy:
                    print("hierarchy[hierarchy-2]:", hierarchy["hierarchy-2"])
                    current_label_1_patterns[int(hierarchy["label-2"])]["Device"].append(hierarchy["hierarchy-2"])
            new_patterns.extend(current_label_1_patterns)
        dsl[opcode] = new_patterns
    write_json(dsl_store_path.split(".")[0] + "2.json", dsl)
