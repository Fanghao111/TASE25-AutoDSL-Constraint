import json
import os
import random
import numpy as np

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
    # 在同一个 label-1 下，整合 label-2
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
            # 对 new_patterns 去重
            new_patterns.extend(current_label_1_patterns)
        dsl[opcode] = new_patterns
    write_json(dsl_store_path.split(".")[0] + "2.json", dsl)
