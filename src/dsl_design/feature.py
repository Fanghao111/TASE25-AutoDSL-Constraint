from __future__ import annotations
import numpy as np
from nltk.stem import WordNetLemmatizer
from collections import defaultdict, Counter

from utils.util import read_json, write_json

class Feature:
    def __init__(self, domain_data, feature_data_path):
        self.domain_data = domain_data
        self.feature_data_path = feature_data_path
        self.stop_words = []
        self.feature_data = {}
        self.lemmatizer = WordNetLemmatizer()
        self.corporas = self.load_data()
        self.feature_data = self.load_feature_data()
        # self.feature_data:
        # {
        #     "operation_1": [
        #         {"index": "0", "hierarchy-1": {}, "hierarchy-2": {}, "hierarchy-3": {}}
        #     ],
        #     "operation_2": [...]
        #     ...
        # } # 一个 domain 下的 feature
        # ...

    def load_data(self):
        # 返回值：
        # {
        #     "operation_1": [{<Route Sheet>}, ...],
        #     "operation_2": [{<Route Sheet>}, ...],
        #     ...
        # }, # 一个 domain 下的数据
        # ...
        domain_data = self.domain_data
        sorted_corporas = defaultdict(dict)
        for job in domain_data:
            for step in job.get("route_sheet", []):
                operation = step["operation"]
                if operation not in sorted_corporas:
                    sorted_corporas[operation] = []
                sorted_corporas[operation].append(step)
        return sorted_corporas
    
    def load_feature_data(self):
        # 返回值：
        # {
        #     "operation_1": [
        #         {"index": "0", "hierarchy-1": {}, "hierarchy-2": {}, "hierarchy-3": {}}
        #     ],
        #     ...
        # }, # 一个 domain 下的 feature
        # ...
        feature_data = {operation: self.__global_feature_extraction(operation) for operation in self.corporas}
        return feature_data

    def dump_feature_data(self):
        write_json(self.feature_data_path, self.feature_data)
        return

    def feature_vector_extraction(self, opcode, hierarchy:int=1, idx_list:list=None, labels:list=None) -> tuple[list, list]:
        '''
        hierarchy=1: 特征空间为该 opcode 下的所有句子的 hierachy-1-feature
        hierarchy=2 or 3: 根据传入的上一层的聚类结果：idx_list -> 层级为1的列表，每个idx对应一个句子；labels -> 层级为1的列表，对应每个句子的类别
                          以每个 cluster 为特征空间，构建下一层的特征向量
        返回两个列表，第一个为 index 列表，标识特征向量对应的句子；第二个是特征向量列表
        '''
        index_list, vectors_list = [], []
        global_meta_features = self.feature_data[opcode]
        if hierarchy == 1:
            meta_features_vault = [meta_feature for meta_feature in global_meta_features if meta_feature.get("hierarchy-1")]    # 筛选出存在 hierarchy-1 特征的数据
            features_vault = [meta_feature["hierarchy-1"] for meta_feature in meta_features_vault]   # 提取出所有 hierarchy-1 特征
            vectors_list = self.__feature_vector_cal_2(features_vault, hierarchy)   # 计算特征向量
            index_list = [int(meta_feature["index"]) for meta_feature in meta_features_vault] # 记录特征向量的索引，将特征向量与句子对应起来
        elif hierarchy in range(2, 4):
            grouped_dict = defaultdict(list)
            for idx, label in zip(idx_list, labels):
                grouped_dict[label].append(idx)
            grouped_idx_list = [grouped_dict[label] for label in sorted(grouped_dict)]  # e.g. [[1, 2, 6], [5, 7, 10], [3, 4, 8, 9]]

            for space_idx in grouped_idx_list:
                meta_features_vault = [global_meta_features[int(i)] for i in space_idx if global_meta_features[int(i)].get(f"hierarchy-{hierarchy}")]
                if not meta_features_vault: # 没有下一层的特征，无须再聚，层次树形成叶子结点
                    continue
                features_vault = [meta_feature[f"hierarchy-{hierarchy}"] for meta_feature in meta_features_vault]
                vectors = self.__feature_vector_cal_2(features_vault, hierarchy)
                indices = [int(meta_feature["index"]) for meta_feature in meta_features_vault]
                index_list.append(indices)
                vectors_list.append(vectors)
        return index_list, vectors_list

    def store_cluster_result(self, opcode, idx_list, n_clusters, labels, hierarchy:int):
        for idx, label in zip(idx_list, labels):
            self.feature_data[opcode][idx][f"label-{hierarchy}"] = str(int(label))
        if hierarchy == 3:
            return None, None
        index_list, vectors_list = self.feature_vector_extraction(opcode, hierarchy+1, idx_list, labels)
        return index_list, vectors_list

    def test_feature_vector_extraction(self, opcode, hierarchy):
        '''
        将全局空间作为特征空间去计算特征向量，而不管在哪个 hierarchy
        '''
        global_meta_features = self.feature_data[opcode]
        meta_features_vault = [meta_feature for meta_feature in global_meta_features if meta_feature.get(f"hierarchy-{hierarchy}")]    # 筛选出存在 hierarchy 特征的数据
        features_vault = [meta_feature[f"hierarchy-{hierarchy}"] for meta_feature in meta_features_vault]   # 提取出所有 hierarchy 特征
        vectors_list = self.__feature_vector_cal_2(features_vault, hierarchy)   # 计算特征向量
        index_list = [int(meta_feature["index"]) for meta_feature in meta_features_vault] # 记录特征向量的索引，将特征向量与句子对应起来
        return index_list, vectors_list

    # 对某个 opcode 下所有的 sentence 的三层特征提取
    def __global_feature_extraction(self, opcode):
        feature_space = []
        for idx, step in enumerate(self.corporas[opcode]):
            metadata = {
                "index": str(idx), 
            }
            for hierarchy in range(1, 4):
                feature = self.__feature_extraction(step, hierarchy) or {}
                metadata[f"hierarchy-{hierarchy}"] = feature
            feature_space.append(metadata)

        return feature_space
    
    # 单句的三个层级的特征提取
    def __feature_extraction(self, step, hierarchy:int):
        features = {}
        if hierarchy == 1:
            for flow_units in ["precondition", "postcondition"]:
                for unit in step.get(flow_units, []):
                    # if unit.get("Name", "") != "":
                    #     features[flow_units].append(unit["Name"])
                    # 使用组分的上位类来构造特征
                    superclass = unit.get("component_type")
                    if superclass and superclass != "NONE":
                        features.setdefault(flow_units, []).append(superclass)
        
        elif hierarchy == 2:
            machine = self.__lemmarize(step.get("machine", ""))
            if machine:
                argkeys = [
                    self.__lemmarize(argkey)
                    for argkey, argvalue in step.get("parameters", {}).items()
                    if argvalue
                ]
                features[machine] = argkeys
        
        elif hierarchy == 3:
            for argkey, argvalue in step.get("parameters", {}).items():
                if argvalue:
                    features.setdefault(self.__lemmarize(argkey), []).append(argvalue)
        else:
            return None
        
        return features
    
    # feature_vault:
    #   hierarchy-1: [{"input_flow_units": [liquid], "output_flow_units": [solid, liquid]}, {...}, ...]
    #   hierarchy-2: [{device_1: [temperature, time], device_2: [], ..., device_i: [argkey_0, ..., argkey_j]}, {...}, ...]
    #   hierarchy-3: [{"time": ["15 mins"], "volume": [50mL, 100mL], ..., argkey_i: [argvalue_0, ..., argvalue_j]}, {...}, ...]
    def __feature_vector_cal(self, features_vault, hierarchy:int):
        '''
        不改变句子顺序，计算 features_vault 中每个句子在此 hierarchy 的特征向量，返回特征向量列表
        '''
        feature_vectors = []

        if hierarchy in [1, 2]:
            feature_keys = {key for item in features_vault for key in item.keys()}
            feature_values = {value for item in features_vault for value_list in item.values() for value in value_list}
            
            feature_keys = sorted(feature_keys)
            if hierarchy in [1, 2]:
                feature_values = sorted(feature_values)
            else:
                feature_values = ["Gas", "Liquid", "Solid", "Semi-Solid", "Mixture", "Chemical Compound", "Biological Material", "Reagent", "Physical Object", "File/Data"]
            
            # print(feature_keys)
            # print(feature_values)

            for item in features_vault:
                feature_matrix = np.zeros((len(feature_keys), len(feature_values) + 1))
                for key, value_list in item.items():
                    key_idx = feature_keys.index(key)
                    feature_matrix[key_idx][len(feature_values)] = 1
                    for value in value_list:
                        value_idx = feature_values.index(value)
                        feature_matrix[key_idx][value_idx] += 1
                # Flatten
                feature_vectors.append(feature_matrix.flatten())
        
        elif hierarchy == 3:
            feature_dict = defaultdict(set)
            for item in features_vault:
                for argkey, argvalue_list in item.items():
                    feature_dict[argkey].update(argvalue_list)
            sorted_feature_dict = {
                key: sorted(value) 
                for key, value in sorted(feature_dict.items(), key=lambda item: len(item[1]), reverse=True)
            }

            keys_list = list(sorted_feature_dict.keys())
            values_list = list(sorted_feature_dict.values())

            # print(keys_list)
            # print(values_list)

            for item in features_vault:
                feature_matrix = [np.zeros(len(argvalue_list)) for argvalue_list in values_list]
                for key, value_list in item.items():
                    key_idx = keys_list.index(key)
                    for value in value_list:
                        value_idx = values_list[key_idx].index(value)
                        feature_matrix[key_idx][value_idx] = 1

                # Flatten
                feature_vectors.append(np.concatenate(feature_matrix))

        return np.array(feature_vectors)
    
    def __feature_vector_cal_2(self, features_vault, hierarchy:int):
        feature_vectors = []

        if hierarchy == 1:
            feature_keys = ["precondition", "postcondition"]
            vector_reference = []
            for key in feature_keys:
                value_count_global = defaultdict(int)
                for item in features_vault:
                    if key in item:
                        value_count = Counter(item[key])
                        for value, num in value_count.items():
                            if value_count_global[value] < num:
                                value_count_global[value] = num
                vector_reference.append(dict(sorted(value_count_global.items())))

            # print(vector_reference)
            feature_values = [list(reference.keys()) for reference in vector_reference]

            feature_dim = [sum(count_dict.values()) for count_dict in vector_reference]
            # print(feature_dim)
            
            for item in features_vault:
                feature_matrix = [np.zeros(dim) for dim in feature_dim]
                for key, value_list in item.items():
                    key_idx = feature_keys.index(key)
                    reference = vector_reference[key_idx]
                    count = Counter(value_list)
                    for value, num in count.items():
                        value_idx = feature_values[key_idx].index(value)
                        start = sum([reference[feature_values[key_idx][k]] for k in range(value_idx)])
                        feature_matrix[key_idx][start:start+num] = 1
                
                feature_vectors.append(np.concatenate(feature_matrix))
        
        elif hierarchy in [2, 3]:
            feature_dict = defaultdict(set)
            for item in features_vault:
                for argkey, argvalue_list in item.items():
                    try:
                        feature_dict[argkey].update(argvalue_list)
                    except Exception as _:
                        continue
            sorted_feature_dict = {
                key: sorted(map(str, value)) 
                for key, value in sorted(feature_dict.items(), key=lambda item: len(item[1]), reverse=True)
            }

            keys_list = list(sorted_feature_dict.keys())
            values_list = list(sorted_feature_dict.values())

            # print(keys_list)
            # print(values_list)

            for item in features_vault:
                feature_matrix = [np.zeros(len(argvalue_list) + 1) for argvalue_list in values_list]
                for key, value_list in item.items():
                    try:
                        key_idx = keys_list.index(key)
                        feature_matrix[key_idx][len(values_list[key_idx])] = 1
                        for value in value_list:
                            value_idx = values_list[key_idx].index(value)
                            feature_matrix[key_idx][value_idx] = 1
                    except Exception as e:
                        continue

                # Flatten
                feature_vectors.append(np.concatenate(feature_matrix))

        return np.array(feature_vectors)
    
    def __lemmarize(self, word, pos="n"):
        return self.lemmatizer.lemmatize(word.lower(), pos=pos)
    
    def __unify(self, data_recognized):
        device_mapping = {original: unified for unified, original_list in self.same_devices.items() for original in original_list}
        for protocol in data_recognized:
            for sentence in protocol:
                for device_dict in sentence.get("recognized", {}).get("devices", []):
                    try:
                        device = device_dict["Name"]
                        device_dict["Name"] = device_mapping.get(device, device)
                    except:
                        continue
        return data_recognized
