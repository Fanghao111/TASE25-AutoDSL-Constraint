from __future__ import annotations
import random 
import copy
import pandas as pd
import torch
import openai
import os
import time
from openai import OpenAI
from collections import defaultdict, Counter
from torch.nn import functional as F
from src.dsl_design.cluster import DPMM
from src.dsl_design.cluster2 import DPMM as DPMM2
from src.dsl_design.feature import Feature
from utils.distribution import N_Gaussian_Distribution
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from utils.util import read_json, write_json, read_txt

class Operation:
    def __init__(self, feature:Feature, operation_dsl_path):
        self.feature = feature
        self.name_mapping = {"precondition": "Precond", "postcondition": "Postcond"}
        self.operation_dsl_path = operation_dsl_path
        self.operation_dsl_tree = {}
        # self.operation_dsl_tree_corpora = {}
        self.operation_dsl = {}
        # self.curve = pd.DataFrame()
        self.curve = {}
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', clean_up_tokenization_spaces=True)
        self.model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

    def recursive_clustering(self, opcode, idx_list, value_list, hierarchy, iter_times=200, alpha=0.1, regular=0.1):
        if value_list is None or len(value_list) == 0:
            return
        
        result = DPMM.cluster(value_list, N_Gaussian_Distribution, len(value_list[0]), iter_times=iter_times, alpha=alpha, regular=regular)
        # result = DPMM2(X=value_list)
        if hierarchy == 1:
            self.curve[opcode] = [float(num) for num in result["log_likelihood_list"].split()]
        
        next_idx_lists, next_value_lists = self.feature.store_cluster_result(
            opcode=opcode,
            idx_list=idx_list,
            n_clusters=result["K"],
            labels=result["label"],
            hierarchy=hierarchy
        )

        if hierarchy < 3:
            for idx_list_next, value_list_next in zip(next_idx_lists, next_value_lists):
                self.recursive_clustering(opcode, idx_list_next, value_list_next, hierarchy + 1, iter_times, alpha, regular)
        
    def analyse(self, opcode):
        self.__hierarchy_tree_construction(opcode)
        self.__dsl_construction_3(opcode)
        self.__dsl_abstraction(opcode)
        self.__pattern_merge(opcode)

    def dsl_regular(self):
        operation_dsl = self.operation_dsl
        # 对 operation_dsl 的 operation 进行同义词合并，将含义相近的 operation 下的 pattern 合并到一个 operation 中。同时对 pattern 内的 parameters 的 value 进行合并，例如 400 mm 和 400 MM 将视为同一个，仅保留其中一个。
        # added_operation_dsl = {}
        # for operation, patterns in operation_dsl.items():
        #     for added_operation, added_patterns in added_operation_dsl.items():
        #         # 判断 operation 与 added_operation 是否同义词：
        #         if operation == added_operation:
        #             added_patterns.extend(copy.deepcopy(patterns))
        #             break
        #         operation_embedding = self.__get_embedding(operation).reshape(1, -1)
        #         added_operation_embedding = self.__get_embedding(added_operation).reshape(1, -1)
        #         if cosine_similarity(operation_embedding, added_operation_embedding) > 0.9:
        #             result = self.__same_operation_judge(operation, added_operation)
        #             if result == "YES" or result == "YES.":
        #                 added_patterns.extend(copy.deepcopy(patterns))
        #                 break
        #     else:
        #         added_operation_dsl[operation] = copy.deepcopy(patterns)
        added_operation_dsl = copy.deepcopy(operation_dsl)
        for added_operation, added_patterns in added_operation_dsl.items():
            # 对 added_patterns 的 parameters 的 value 进行合并
            for pattern in added_patterns:
                for execution in pattern["pattern"]["Execution"]:
                    for argkey, argvalues in execution["parameters"].items():
                        new_argvalues = self.__sorted_unique(argvalues)
                        execution["parameters"][argkey] = self.__sorted_unique(argvalues)
        for added_operation, added_patterns in added_operation_dsl.items():
            most_machine_count = {}
            # 统计当前 patterns 中出现次数最多的 most_machine，仅保留出现 most_machine 的 patterns，且 Execution 数组中也只保留出现 most_machine 的项
            for pattern in added_patterns:
                for execution in pattern["pattern"]["Execution"]:
                    machine = execution["machine"]
                    most_machine_count[machine] = most_machine_count.get(machine, 0) + 1
            most_machine = max(most_machine_count, key=most_machine_count.get)
            for pattern in added_patterns:
                pattern["pattern"]["Execution"] = [execution for execution in pattern["pattern"]["Execution"] if execution["machine"] == most_machine]
            # 删除 added_patterns 中的空 pattern
            added_patterns = [pattern for pattern in added_patterns if pattern["pattern"]["Execution"]]
        self.operation_dsl = added_operation_dsl

    # 将聚类结果构建为层次聚类树
    def __hierarchy_tree_construction(self, opcode):
        feature_data = self.feature.feature_data.get(opcode, [])
        
        tree = {}
        for sentence in feature_data:
            # label
            l1 = sentence.get("label-1")
            l2 = sentence.get("label-2")
            l3 = sentence.get("label-3")

            # feature
            h1 = sentence.get("hierarchy-1", {})
            h2 = sentence.get("hierarchy-2", {})
            h3 = sentence.get("hierarchy-3", {})

            if l1 and l1 not in tree:
                tree[l1] = {"pattern": {"Precond": [], "Postcond": []}, "child": []}
            if l2 and l2 not in tree[l1]:
                tree[l1] = {**tree[l1], l2: {"pattern": {"machine":[]}, "child": []}}
            if l3 and l3 not in tree[l1][l2]:
                tree[l1][l2] = {**tree[l1][l2], l3: {"pattern": {}, "child": []}}
            
            if l1:
                for key, value in h1.items():
                    tree[l1]["pattern"][self.name_mapping[key]].append(value)

            if l2 and h2:
                tree[l1]["child"].append(l2)
                tree[l1][l2]["pattern"]["machine"].append(h2)

            # if l3:
            #     tree[l1][l2]["child"].append(l3)
            #     for key, value in h3.items():
            #         tree[l1][l2][l3]["pattern"].setdefault(key, []).append(value)
            #     tree[l1][l2][l3]["examples"].append(sentence["sentence"])

            if l3:
                tree[l1][l2]["child"].append(l3)
                for device, argkeys in h2.items():
                    for argkey in argkeys:
                        tree[l1][l2][l3]["pattern"].setdefault(device, {}).setdefault(argkey, []).extend(h3.get(argkey, []))
        
        tree = {label: tree[label] for label in sorted(tree)}
        self.operation_dsl_tree[opcode] = tree
        # self.operation_dsl_tree_corpora[opcode] = copy.deepcopy(tree)
        
        return tree

    # 对层次聚类树中的各层 pattern 进行抽象与合并
    def __dsl_tree_abstraction(self, opcode):
        tree = self.operation_dsl_tree[opcode]
        for _, feature1 in tree.items():
            feature1["pattern"] = self.__pattern_abstraction(feature1["pattern"], hierarchy=1)

            for label2, feature2 in feature1.items():
                if label2.isdigit():
                    feature2["pattern"] = self.__pattern_abstraction(feature2["pattern"], hierarchy=2)

                    for label3, feature3 in feature2.items():
                        if label3.isdigit():
                            feature3["pattern"] = self.__pattern_abstraction(feature3["pattern"], hierarchy=3)

    # 对单组 pattern 的抽象与合并
    def __pattern_abstraction(self, pattern, hierarchy):
        abstract_pattern = {}
        if hierarchy == 1:
            for key in ["Precond", "Postcond"]:
                abstract_pattern[key] = []

                if not pattern[key]:
                    continue

                pattern_dict = defaultdict(int)
                
                # 统计同一类中不同的 pattern 计数
                for feature in pattern[key]:
                    pattern_tuple = tuple(sorted(Counter(feature).items()))
                    pattern_dict[pattern_tuple] += 1
                
                # 选择出现次数最多的 pattern
                most_common_pattern = max(pattern_dict, key=pattern_dict.get)

                arg_name = "SlotArg" if key == "Precond" else "EmitArg"
                for phase, num in most_common_pattern:
                    sub_pattern = {
                        f"{arg_name}Num": num,
                        arg_name: phase
                    }
                    abstract_pattern[key].append(sub_pattern)
        
        elif hierarchy == 2:
            pattern_dict = defaultdict(set)
            for feature in pattern["machine"]:
                for device, argkeys in feature.items():
                    pattern_dict[device].update(argkeys)
            sorted_pattern_dict = {
                key: sorted(value)
                for key, value in sorted(pattern_dict.items(), key=lambda item: len(item[1]), reverse=True)
            }
            abstract_pattern["machine"] = sorted_pattern_dict

        elif hierarchy == 3:
            for argkey, argvalues in pattern.items():
                value_range = sorted([value for value_list in argvalues for value in value_list])
                abstract_pattern[argkey] = value_range
        
        return abstract_pattern

    # 提取层次聚类树的所有叶子结点特征的 pattern 组合，生成标准的 DSL instruction
    def __dsl_construction(self, opcode):
        opcode_feature = []

        tree = copy.deepcopy(self.operation_dsl_tree[opcode])
        for _, feature1 in tree.items():
            pattern = {
                "Precond": [],
                "Postcond": [],
                "Execution": []
            }
            pattern1 = feature1["pattern"]
            pattern["Precond"] = pattern1["Precond"]
            pattern["Postcond"] = pattern1["Postcond"]

            for label2, feature2 in feature1.items():
                if label2.isdigit():
                    pattern2 = feature2["pattern"]
                    for device, argkeys in pattern2["machine"].items():
                        sub_pattern = {"machine": device, "parameters": {}}
                        for argkey in argkeys:
                            sub_pattern["parameters"][argkey] = []
                        pattern["Execution"].append(sub_pattern)

                    for label3, feature3 in feature2.items():
                        if label3.isdigit():
                            pattern3 = feature3["pattern"]
                            for argkey, value_range in pattern3.items():
                                for execution in pattern["Execution"]:
                                    if argkey in execution["parameters"]:
                                        execution["parameters"][argkey] = value_range
            
            opcode_feature.append({
                "pattern": pattern, 
            })
        
        self.operation_dsl[opcode] = opcode_feature

        return opcode_feature

    def __dsl_construction_2(self, opcode):
        def combine_patterns(tree, node, ancestor_pattern=None):
            if ancestor_pattern is None:
                ancestor_pattern = {
                    "Precond": [],
                    "Postcond": [],
                    "Execution": {
                        "machine": [],
                        "parameters": {}
                    }
                }
            node_pattern = node.get("pattern", {})
            
            # 合并当前节点的 pattern
            if any(cond in node_pattern for cond in ["Precond", "Postcond"]):
                ancestor_pattern["Precond"].extend(node_pattern.get("Precond", []))
                ancestor_pattern["Postcond"].extend(node_pattern.get("Postcond", [])) 
            elif "machine" in node_pattern:
                for device, argkeys in node_pattern["machine"].items():
                    sub_pattern = {"machine": device, "Argkeys": []}
                    sub_pattern["Argkeys"].extend(argkeys)
                    ancestor_pattern["Execution"]["machine"].append(sub_pattern)
            else:
                ancestor_pattern["Execution"]["parameters"] = node_pattern
            
            # 如果有子节点，递归处理每个子节点
            leaf_patterns = []
            has_child = False
            for label, child_node in node.items():
                if label.isdigit():
                    has_child = True
                    # 递归
                    child_patterns = combine_patterns(tree, child_node, copy.deepcopy(ancestor_pattern))
                    leaf_patterns.extend(child_patterns)

            # 如果是叶子节点
            if not has_child:
                leaf_patterns.append({
                    "pattern": copy.deepcopy(ancestor_pattern),
                })

            return leaf_patterns
        
        opcode_feature = []
        tree = copy.deepcopy(self.operation_dsl_tree[opcode])
        for label, feature in tree.items():
            if label.isdigit(): # 遍历第一层
                leaf_patterns = combine_patterns(tree, feature)
                opcode_feature.extend(leaf_patterns)

        self.operation_dsl[opcode] = opcode_feature

        return opcode_feature

    # 根据聚类树生成 dsl （句子特征堆砌罗列），合并第二第三层的 pattern，每个第一层的 cluster 对应一个 pattern
    def __dsl_construction_3(self, opcode):
        def combine_patterns(node):
            pattern = {
                    "Precond": [],
                    "Postcond": [],
                    "Execution": {
                        "machine": [],
                        "parameters": {}
                }
            }
            sub_pattern_1 = node.get("pattern", {})
            pattern["Precond"] = sub_pattern_1.get("Precond", [])
            pattern["Postcond"] = sub_pattern_1.get("Postcond", [])
            
            for label2, feature2 in node.items():
                if label2.isdigit():
                    sub_pattern_2 = feature2.get("pattern", {})
                    pattern["Execution"]["machine"].extend(sub_pattern_2.get("machine", []))

                    for label3, feature3 in feature2.items():
                        if label3.isdigit():
                            sub_pattern_3 = feature3.get("pattern", {})
                            for device, arg_dict in sub_pattern_3.items():
                                for argkey, argvalues in arg_dict.items():
                                    pattern["Execution"]["parameters"].setdefault(device, {}).setdefault(argkey, []).extend(argvalues)
            
            res_pattern = {
                "pattern": pattern,
            }

            return res_pattern

        opcode_feature = []
        tree = copy.deepcopy(self.operation_dsl_tree[opcode])
        for label, feature in tree.items():
            if label.isdigit():
                pattern = combine_patterns(feature)
                opcode_feature.append(pattern)
        
        self.operation_dsl[opcode] = opcode_feature

        return opcode_feature

    def __dsl_pattern_abstraction(self, pattern):
        abstract_pattern = {
                    "Precond": {},
                    "Postcond": {},
                    "Execution": []
            }
        # 第一层特征
        for key in ["Precond", "Postcond"]:
            if not pattern[key]:
                continue
            pattern_dict = defaultdict(int)
            # 统计同一类中不同的 pattern 计数
            for feature in pattern[key]:
                pattern_tuple = tuple(sorted(Counter(feature).items()))
                pattern_dict[pattern_tuple] += 1
            # 选择出现次数最多的 pattern
            most_common_pattern = max(pattern_dict, key=pattern_dict.get)
            arg_name = "SlotArg" if key == "Precond" else "EmitArg"
            sub_pattern = {
                f"{arg_name}Num": 0,
                arg_name: []
            }
            for phase, num in most_common_pattern:
                for _ in range(num):
                    sub_pattern[f"{arg_name}Num"] += 1
                    sub_pattern[arg_name].append(phase)
            abstract_pattern[key] = sub_pattern
        # 第二层特征
        pattern_dict = defaultdict(set)
        for feature in pattern["Execution"]["machine"]:
            for device, argkeys in feature.items():
                pattern_dict[device].update(argkeys)
        sorted_pattern_dict = {
            key: sorted(value)
            for key, value in sorted(pattern_dict.items(), key=lambda item: len(item[1]), reverse=True)
        }
        for device, argkeys in sorted_pattern_dict.items():
            sub_pattern = {"machine": device, "parameters": {}}
            for argkey in argkeys:
                sub_pattern["parameters"][argkey] = []
            abstract_pattern["Execution"].append(sub_pattern)
        # 第三层特征
        # abstract_pattern["Execution"]["Config"] = {argkey: self.__sorted_unique(argvalues) for argkey, argvalues in pattern["Execution"]["Config"].items()}
        for device, arg_dict in pattern["Execution"]["parameters"].items():
            for execution in abstract_pattern["Execution"]:
                if execution["machine"] == device:
                    for argkey, argvalues in arg_dict.items():
                        execution["parameters"][argkey].extend(self.__sorted_unique(argvalues))
        
        return abstract_pattern
    
    def __dsl_abstraction(self, opcode):
        dsl = self.operation_dsl[opcode]
        for meta_pattern in dsl[:]:
            meta_pattern["pattern"] = self.__dsl_pattern_abstraction(meta_pattern["pattern"])
            if not meta_pattern["pattern"]["Execution"]:
                dsl.remove(meta_pattern)
            meta_pattern["pattern"] = {
                "Precond": meta_pattern["pattern"]["Precond"],
                "Execution": meta_pattern["pattern"]["Execution"],
                "Postcond": meta_pattern["pattern"]["Postcond"]
            }
        if not dsl:
            del self.operation_dsl[opcode]

    def __pattern_merge(self, opcode):
        if opcode not in self.operation_dsl:
            return
        
        def has_same_slot_emit(precond1, precond2, postcond1, postcond2):
            return (set(precond1.get("SlotArg", [])) == set(precond2.get("SlotArg", []))) and (set(postcond1.get("EmitArg", [])) == set(postcond2.get("EmitArg", [])))
        
        def has_same_device_type(exec1, exec2):
            device_types1 = [device["machine"] for device in exec1]
            device_types2 = [device["machine"] for device in exec2]
            return set(device_types1) == set(device_types2)
        
        def choose_pattern(item1, item2):
            return item1

        data = self.operation_dsl[opcode]
        merged_data = []
        # 用于存储已经处理过的字典的索引
        processed_indices = set()
        # 遍历每一个数据字典
        for i, item in enumerate(data):
            if i in processed_indices:
                continue  # 如果这个字典已经合并过，跳过
            # 初始化合并对象
            merged_item = {
                "pattern": {
                    "Precond": item["pattern"]["Precond"],
                    "Execution": [],
                    "Postcond": item["pattern"]["Postcond"]
                },
            }
            # 合并第一个字典的内容
            merged_item["pattern"]["Execution"].extend(item["pattern"]["Execution"])
            # 与后续未处理的数据进行比较和合并
            for j in range(i+1, len(data)):
                if j in processed_indices:
                    continue  # 如果这个字典已经处理过，跳过
                # 如果 precond 和 postcond 相同，则合并
                if (has_same_slot_emit(item["pattern"]["Precond"], data[j]["pattern"]["Precond"],
                                       item["pattern"]["Postcond"], data[j]["pattern"]["Postcond"]) or 
                    has_same_device_type(item["pattern"]["Execution"], data[j]["pattern"]["Execution"])):

                    chosen_pattern = choose_pattern(item, data[j])
                    merged_item["pattern"]["Precond"] = chosen_pattern["pattern"]["Precond"]
                    merged_item["pattern"]["Postcond"] = chosen_pattern["pattern"]["Postcond"]
                    # 合并 Execution
                    for new_device in data[j]["pattern"]["Execution"]:
                        # 检查是否存在相同的 machine
                        for existing_device in merged_item["pattern"]["Execution"]:
                            if existing_device["machine"] == new_device["machine"]:
                                # 如果 machine 相同，合并 parameters
                                for key, value in new_device["parameters"].items():
                                    if key in existing_device["parameters"]:
                                        existing_device["parameters"][key].extend(value)
                                    else:
                                        existing_device["parameters"][key] = value
                                break
                        else:
                            # 如果没有相同的 machine，添加新的 machine
                            merged_item["pattern"]["Execution"].append(new_device)

                    # 标记这个字典已经处理过
                    processed_indices.add(j)

            for device_dict in merged_item["pattern"]["Execution"]:
                device_dict["parameters"] = {key: self.__sorted_unique(value_list) for key, value_list in device_dict["parameters"].items()}
            # 将合并后的字典加入结果列表
            merged_data.append(merged_item)
        self.operation_dsl[opcode] = merged_data

    def __sorted_unique(self, lst):
        try:
            # 将所有字符串转为小写
            processed_lst = [x.lower() if isinstance(x, str) else x for x in lst]
            
            # 统计元素频率
            count = Counter(processed_lst)
            
            # 按频率排序并返回
            return sorted(count.keys(), key=lambda x: count[x], reverse=True)
        except:
            # 异常处理，返回原列表
            return lst

    def dump_result(self):
        self.feature.dump_feature_data()
        self.dump_operation_dsl()
        self.dump_log()

    def dump_operation_dsl(self):
        base_path = self.operation_dsl_path.rsplit(".json", 1)[0]
        write_json(self.operation_dsl_path, self.operation_dsl)
        write_json(f"{base_path}_tree.json", self.operation_dsl_tree)
        # write_json(f"{base_path}_tree_corpora.json", self.operation_dsl_tree_corpora) 

    def dump_log(self):
        metadata = {"Number of opcode": len(self.operation_dsl)}
        count = {opcode: len(operations) for opcode, operations in self.operation_dsl.items()}
        metadata["Average number of pattern"] = round(sum(count.values()) / len(count), 1)
        metadata.update(sorted(count.items(), key=lambda item: item[1], reverse=True))
        base_path = self.operation_dsl_path.rsplit(".json", 1)[0]
        write_json(f"{base_path}_metadata.json", metadata)

        self.curve.to_csv(f"{base_path}_curve.csv", index=True)
        
    def __get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.squeeze().numpy()

    def __same_operation_judge(self, operation_1, operation_2):
        prompt = read_txt("data/prompts/same_operation_judgement.txt")
        pair = (operation_1, operation_2)
        prompt = prompt.replace("---TARGET---", str(pair))
        result = self.__chatgpt_function(prompt)
        return result

    def __chatgpt_function(self, content, gpt_model="gpt-4o"):
        while True:
            try:
                client = OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                )
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "user", "content": content}
                    ],
                    model=gpt_model
                )
                return chat_completion.choices[0].message.content
            except openai.APIError as error:
                print("error: ", error)
                continue
