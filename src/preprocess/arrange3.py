from utils.util import read_json, write_json, read_txt, write_txt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import itertools
from tqdm import tqdm
import networkx as nx
from networkx.algorithms import bipartite

class Arrange3:
    def __init__(self, synthetic_graph_path, jssp_graph_path, arrange_store_path, arrange_apply_path):
        self.synthetic_graph = read_json(synthetic_graph_path)
        self.jssp_graphs = read_json(jssp_graph_path)
        self.arrange_apply_path = arrange_apply_path
        self.arrange_store_path = arrange_store_path
        self.arrange_result = list()

        self.synthetic_dependency_pairs = self.__get_dependency_pairs(self.synthetic_graph["dependence_graph"])
    
    def arrange(self):
        total_matches = 0
        conflict_jssp_num = 0
        for jssp_graph in tqdm(self.jssp_graphs):
            jssp_dependency_pairs = self.__get_dependency_pairs(jssp_graph["dependence_graph"])
            # 匹配
            # print("synthetic_dependency_pairs: ", self.synthetic_dependency_pairs)
            # print("jssp_dependency_pairs: ", jssp_dependency_pairs)
            mapping, matches, conflict = self.__find_best_mapping_3(self.synthetic_dependency_pairs, jssp_dependency_pairs)
            if conflict > 0:
                conflict_jssp_num += 1
            # make it JSON serializable
            if mapping == None:
                mapping = 'None'
            self.arrange_result.append({
                "mapping": mapping,
                "matches": int(matches)
            })
            total_matches += int(matches)
        print("conflict_jssp_num: ", conflict_jssp_num)
        print("total_matches: ", total_matches)
        write_json(self.arrange_store_path, self.arrange_result)
    
    def __get_dependency_pairs(self, matrix):
        pairs = list()
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] == 1:
                    pairs.append((i, j))# 即 j 依赖 i
        return pairs

    def __find_best_mapping(self, A, B):
        GA = self.__build_graph(A)
        GB = self.__build_graph(B)
        
        # 这里我们尝试找到最大公共子图
        GM = nx.algorithms.isomorphism.DiGraphMatcher(GA, GB)
        best_mapping = None
        max_matches = -1
        for subgraph_mapping in GM.subgraph_isomorphisms_iter():
            matches = len(subgraph_mapping)
            if matches > max_matches:
                max_matches = matches
                best_mapping = subgraph_mapping
        return best_mapping, max_matches

    def __build_graph(self, dependencies):
        G = nx.DiGraph()
        for src, dst in dependencies:
            G.add_edge(src, dst)
        return G

    def __find_best_mapping_2(self, A, B):
        pass

    def __find_best_mapping_3(self, A, B):
        from collections import defaultdict
        new_B = B.copy()
        # 提取所有唯一的节点
        nodes_A = set()
        for src, dst in A:
            nodes_A.add(src)
            nodes_A.add(dst)
            
        nodes_B = set()
        for src, dst in B:
            nodes_B.add(src)
            nodes_B.add(dst)

        # 统计 A 中每个节点的入度和出度
        in_degree_A = defaultdict(int)
        out_degree_A = defaultdict(int)
        for src, dst in A:
            out_degree_A[src] += 1
            in_degree_A[dst] += 1

        # 统计 B 中每个节点的入度和出度
        in_degree_B = defaultdict(int)
        out_degree_B = defaultdict(int)
        for src, dst in B:
            out_degree_B[src] += 1
            in_degree_B[dst] += 1

        # 为 B 的节点找到 A 中度数最接近的节点
        mapping = {}
        used_A = set()

        # 为了提高映射效果，可以按 B 节点的总度数（入度 + 出度）降序排序
        sorted_b_nodes = sorted(
            nodes_B, 
            key=lambda b: (out_degree_B[b] + in_degree_B[b]), 
            # key=lambda b: (in_degree_B[b]), 
            reverse=True
        )

        for b_node in sorted_b_nodes:
            candidates = sorted(
                nodes_A - used_A, 
                key=lambda a_node: abs(out_degree_A[a_node] - out_degree_B[b_node]) + abs(in_degree_A[a_node] - in_degree_B[b_node])
            )
            if candidates:
                # 选择度数差异最小的 A 节点进行映射，并检查是否存在相反的依赖关系，如果存在，则选择下一个 A 节点
                for a_node in candidates:
                    current_match = a_node
                    mapping[b_node] = current_match
                    flag = True
                    for src, dst in B:
                        if (mapping.get(dst), mapping.get(src)) in A:
                            # 删除该映射
                            mapping[b_node] = None
                            flag = False
                            break
                    if(flag):
                        used_A.add(current_match)
                        break
                            
            else:
                # 如果没有可用的 A 节点（理论上不会发生，因为 A 的节点 >= B 的节点）
                mapping[b_node] = None

        
        matches = sum(
            1 for src, dst in B 
            if (mapping.get(src), mapping.get(dst)) in A
        )
        conflict = 0
        for src, dst in B:
            if (mapping.get(dst), mapping.get(src)) in A:
                conflict += 1
        # if conflict > 0:
        #     print("conflict: ", conflict, "\n")
        return mapping, matches, conflict

    def __find_best_mapping_4(self, A, B):
        # 将 B 随机打乱，然后 A 和 B 的每一项逐项映射，映射完后检查是否存在与 A 中相反的依赖关系，若存在，则重新映射
        # 重复这个过程直到找到最佳映射
        best_mapping = None
        best_matches = 0
        for _ in range(1000):
            B_shuffled = B.copy()
            np.random.shuffle(B_shuffled)
            mapping = dict(zip(B, B_shuffled))
            matches = sum(1 for src, dst in B if (mapping.get(src), mapping.get(dst)) in A)
            if matches > best_matches:
                best_mapping = mapping
                best_matches = matches
        return best_mapping, best_matches

