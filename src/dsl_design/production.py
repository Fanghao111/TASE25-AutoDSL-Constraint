import copy
from tqdm import tqdm
from collections import defaultdict, Counter
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import random
from src.dsl_design.cluster import DPMM
from utils.distribution import N_Gaussian_Distribution
from utils.util import read_json, write_json
import math
import itertools

class Production:
    def __init__(self, domain_data, operation_dsl={}, same_components={}):
        self.domain_data = domain_data
        self.operation_dsl = operation_dsl
        self.same_components = same_components
        self.component_mapping = self.__unify()
        self.production_dsl = {}
        self.production_list = []
        self.pattern_example = {
            "Pred": [],
            "FlowUnit": {
                "component": "",
                "component_type": "",
                "container": [],
            },
            "Succ": []
        }
        self.dependency_structure = []

        self.operations = []
        self.EM_results = []
        self.EM_updates = []
        # {
        #     Pred: <Operation.UniqueName>,
        #     FlowUnit: {
        #         component: "E.coli",
        #         ComponentType: BiologicalMaterial,
        #         RefName: <STR>,
        #         UnitArgType: PROD,
        #         Vol: (range),
        #         Container: Flask | Tube,
        #         Cond: {
        #             Tempreature: (range)
        #         }
        #     },
        #     Succ: <Operation.UniqueName>
        # }

    def extract(self):
        num_flowunits = 0
        for job in self.domain_data:
            for idx, step in enumerate(job.get("route_sheet", [])):
                try:
                    meta_flowunits = [metadata for metadata in (step.get("precondition", []) + step.get("postcondition", [])) if metadata]
                except:
                    continue
                num_flowunits += len(meta_flowunits)

                for metadata in meta_flowunits:
                    if not metadata.get("component_type") or metadata["component_type"] == "NONE":
                        continue
                    name = self.component_mapping.get(metadata["component"], metadata["component"])
                    pattern = self.production_dsl.setdefault(name, copy.deepcopy(self.pattern_example))
                    pattern["Pred"].append(step["operation"])
                    pattern["FlowUnit"]["component"] = name
                    pattern["FlowUnit"]["component_type"] = metadata["component_type"]
                    
                    if metadata["container"]:
                        pattern["FlowUnit"]["container"].append(metadata["container"])
                    
                    # pattern["Succ"] 的添加规则：如果当前 step 的 postcondition 将被用于该 domain 内所有 step 中的另一个 step 的 precondition，则将这另一个 step 的 operation 添加到 pattern["Succ"] 中
                    for other_job in self.domain_data:
                        for other_step in other_job.get("route_sheet", []):
                            if other_step["precondition"] and metadata["component_type"] in [other_metadata["component_type"] for other_metadata in other_step["precondition"]]:
                                pattern["Succ"].append(other_step["operation"])
                        
        self.__abstraction()
        num_abstraction = len(self.production_dsl)
        # write_json(production_dsl_store_path, self.production_dsl)
        # print(num_flowunits)
        # print(num_abstraction)
        print(format(num_abstraction / num_flowunits * 100, ".2f"))
    
    def __extract_actual_product_structure(self):
        EM_results = []
        production_dsl = copy.deepcopy(self.production_dsl)
        pred_operation_mapping = {}
        succ_operation_mapping = {}
        pred_operation_template = "Pred_i"
        succ_operation_template = "Succ_i"
        for component_name, dsl in production_dsl.items():
            pred = dsl["Pred"]
            succ = dsl["Succ"]
            if pred not in pred_operation_mapping:
                pred_operation_mapping[pred] = pred_operation_template.replace("i", str(len(pred_operation_mapping)))
            if succ not in succ_operation_mapping:
                succ_operation_mapping[succ] = succ_operation_template.replace("i", str(len(succ_operation_mapping)))
            EM_results.append([pred_operation_mapping[pred], succ_operation_mapping[succ]])
        # # 对 EM_results 去重
        # EM_results = list(set([tuple(result) for result in EM_results]))
        # 对属于同一个 succ 的 pred 进行合并
        # Step 1: Merge preds for the same succ
        succ_to_preds = defaultdict(list)
        for pred, succ in EM_results:
            succ_to_preds[succ].append(pred)

        # Reverse the mapping: preds to succ
        preds_to_succ = defaultdict(list)
        for succ, preds in succ_to_preds.items():
            preds_tuple = tuple(sorted(preds))
            preds_to_succ[preds_tuple].append(succ)

        # Final merge where preds map to multiple succs
        result = []
        for preds, succs in preds_to_succ.items():
            result.append({"preds": list(preds), "succs": succs})
        # Step 2: Filter results where preds length and succs length are the same
        return result

    def EM_extract(self):
        threshold = 0
        structure_candidates = [
            (1, 1), (2, 1), (3, 1), (4, 1), 
            (1, 2), (2, 2), (3, 2), (4, 2),
            (1, 3), (2, 3), (3, 3), (4, 3),
            (1, 4), (2, 4), (3, 4), (4, 4)
            ]
        weights = [1/len(structure_candidates) for _ in range(len(structure_candidates))]
        updates = [[] for _ in range(len(structure_candidates))]

        actual_product_structure_raw = self.__extract_actual_product_structure()
        actual_product_structure = {}

        for i in range(30):
            for j in range(30):
                actual_product_structure[f"{i+1}-{j+1}"] = 0
        
        total_structure_num = 0
        for structure in actual_product_structure_raw:
            pres_count = len(structure["preds"])
            succs_count = len(structure["succs"])
            actual_product_structure[f"{pres_count}-{succs_count}"] += 1
            total_structure_num += 1

        actual_product_structure_weights = [
            actual_product_structure.get(f"{pres}-{succ}", 0) / total_structure_num
            for pres, succ in structure_candidates
        ]
        for _ in range(100):
            # E-step
            weights_new = [weight * actual_weight for weight, actual_weight in zip(weights, actual_product_structure_weights)]
            weights_new_sum = sum(weights_new)
            # print("actual_product_structure_weights: ", actual_product_structure_weights)
            # print("weights_new: ", weights_new, "\n")
            weights_new = [weight / weights_new_sum for weight in weights_new]
            for i in range(len(structure_candidates)):
                update = abs(weights_new[i] - weights[i])
                updates[i].append(update)
            # M-step
            weights = weights_new
        self.EM_updates = updates
        self.EM_results = [structure for i, structure in enumerate(structure_candidates) if weights[i] > threshold]
        
    def syntax_structure(self):
        production_dsl = read_json("outputs/total_production_dsl.json")

    def __abstraction(self):
        to_delete = []
        for name, dsl in self.production_dsl.items():
            try:
                dsl["Pred"] = self.__most_frequent(dsl["Pred"])
                dsl["Succ"] = self.__most_frequent(dsl["Succ"])
                dsl["FlowUnit"]["container"] = self.__sorted_unique(dsl["FlowUnit"]["container"])
            except:
                to_delete.append(name)
        # Delete after the iteration
        for name in to_delete:
            del self.production_dsl[name]
    
    def __most_frequent(self, lst):
        lst = [ele for ele in lst if ele != "NONE"]
        if not lst:
            return ""
        count = Counter(lst)
        return count.most_common(1)[0][0]

    def __sorted_unique(self, lst):
        count = Counter(lst)
        return sorted(count.keys(), key=lambda x: count[x], reverse=True)

    def __unify(self):
        component_mapping = {original: unified for unified, original_list in self.same_components.items() for original in original_list}
        return component_mapping
    
    def operation_clustering(self):
        operation_features = self.__operation_feature_extraction()
        feature_space = self.__create_feature_space(operation_features)
        # operation_vectors = self.__encode_operations_merge(operation_features, feature_space)
        operation_vectors = self.__encode_operations_select(operation_features, feature_space)

        vectorizer = DictVectorizer(sparse=False)
        X = vectorizer.fit_transform(operation_vectors)

        result = DPMM.cluster(data=X, Distribution=N_Gaussian_Distribution, feature_dim=len(feature_space), iter_times=10000, alpha=0.1, regular=0.1)

        clustered_operations = {}
        opcode_to_superclass = read_json("dsl_design/data/demo/opcode_to_superclass.json")
        operation_names = list(operation_features.keys())
        for i, name in enumerate(operation_names):
            ground_truth = opcode_to_superclass[name[0].upper() + name[1:].lower()]
            clustered_operations.setdefault(int(result["label"][i]), []).append(ground_truth)

        for cluster, ops in clustered_operations.items():
            print(f"Cluster {cluster}: {Counter(ops)}")      

    def __operation_feature_extraction(self):
        operation_features = {}
        for opcode, patterns in self.operation_dsl.items():
            if not patterns:
                continue

            opcode_features = []
            for pattern in patterns:
                feature = {"Precond": [], "Postcond": [], "Device": []}
                for slot in pattern["pattern"]["Precond"]:
                    for _ in range(int(slot["SlotArgNum"])):
                        feature["Precond"].append(slot["SlotArg"])
                for slot in pattern["pattern"]["Postcond"]:
                    for _ in range(int(slot["EmitArgNum"])):
                        feature["Postcond"].append(slot["EmitArg"])
                for device in pattern["pattern"]["Execution"]:
                    feature["Device"].append(device["DeviceType"])
                opcode_features.append(feature)
            operation_features[opcode] = opcode_features
        return operation_features
    
    def __create_feature_space(self, data):
        feature_space = set()
        
        for patterns in data.values():
            for pattern in patterns:
                precond_counts = {}
                postcond_counts = {}
                device_counts = {}

                for pre in pattern.get('Precond', []):
                    count = precond_counts.get(pre, 0) + 1
                    precond_counts[pre] = count
                    feature_space.add(f'Precond_{pre}_{count}')

                for post in pattern.get('Postcond', []):
                    count = postcond_counts.get(post, 0) + 1
                    postcond_counts[post] = count
                    feature_space.add(f'Postcond_{post}_{count}')

                for dev in pattern.get('Device', []):
                    count = device_counts.get(dev, 0) + 1
                    device_counts[dev] = count
                    feature_space.add(f'Device_{dev}_{count}')
        
        return sorted(list(feature_space))

    def __encode_operations_merge(self, data, feature_space):
        operation_vectors = []
        
        for operation, patterns in data.items():
            feature_vector = {feature: 0 for feature in feature_space}
            
            for pattern in patterns:
                precond_counts = {}
                postcond_counts = {}
                device_counts = {}

                for pre in pattern.get('Precond', []):
                    count = precond_counts.get(pre, 0) + 1
                    precond_counts[pre] = count
                    feature_vector[f'Precond_{pre}_{count}'] = 1

                for post in pattern.get('Postcond', []):
                    count = postcond_counts.get(post, 0) + 1
                    postcond_counts[post] = count
                    feature_vector[f'Postcond_{post}_{count}'] = 1

                for dev in pattern.get('Device', []):
                    count = device_counts.get(dev, 0) + 1
                    device_counts[dev] = count
                    feature_vector[f'Device_{dev}_{count}'] = 1
            
            operation_vectors.append(feature_vector)
        
        return operation_vectors
    
    def __encode_operations_select(self, data, feature_space):
        operation_vectors = []
        operation_names = []
        
        for operation, patterns in data.items():

            def count_parameters(pattern):
                return len(pattern.get("Precond", [])) + len(pattern.get("Postcond", [])) + len(pattern.get("Device", []))
            
            selected_pattern = max(patterns, key=count_parameters)

            feature_vector = {feature: 0 for feature in feature_space}
            
            precond_counts = {}
            postcond_counts = {}
            device_counts = {}

            for pre in selected_pattern.get('Precond', []):
                count = precond_counts.get(pre, 0) + 1
                precond_counts[pre] = count
                feature_vector[f'Precond_{pre}_{count}'] = 1

            for post in selected_pattern.get('Postcond', []):
                count = postcond_counts.get(post, 0) + 1
                postcond_counts[post] = count
                feature_vector[f'Postcond_{post}_{count}'] = 1

            for dev in selected_pattern.get('Device', []):
                count = device_counts.get(dev, 0) + 1
                device_counts[dev] = count
                feature_vector[f'Device_{dev}_{count}'] = 1
            
            operation_vectors.append(feature_vector)
        
        return operation_vectors

    def __convert_patterns_to_features(self, patterns):
        pattern_dicts = []
        for pattern in patterns:
            pattern_dict = defaultdict(int)
            for pre in pattern.get("Precond", []):
                pattern_dict[f"Precond_{pre}"] += 1
            for post in pattern.get("Postcond", []):
                pattern_dict[f"Postcond_{post}"] += 1
            for dev in pattern.get("Device", []):
                pattern_dict[f"Device_{dev}"] += 1
            pattern_dicts.append(dict(pattern_dict))
        return pattern_dicts
