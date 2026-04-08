from utils.util import read_json, write_json, read_txt, write_txt
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict

class SyntheticDependencyGraph:
    def __init__(self, machines_data_path, traversal_store_path, graph_store_path):
        self.machines_data = read_json(machines_data_path)
        self.traversal_store_path = traversal_store_path
        self.graph_store_path = graph_store_path
        self.dependence_graph = {}
        self.pre2post = {}
        self.all_paths = []
        self.path_length_counts = defaultdict(int)
    
    def traversal(self):
        # 将每台机器的 pattern 作为节点进行依赖关系的构建
        for machine_data in self.machines_data:
            machine_name = machine_data['machine']
            for pattern in machine_data['patterns']:
                precondition = pattern['preconditions']
                postcondition = pattern['postconditions']
                configuration = pattern['configuration']
                
                # 保存机器和对应的 pattern 信息
                if precondition not in self.pre2post:
                    self.pre2post[precondition] = []
                self.pre2post[precondition].append({
                    'machine': machine_name,
                    'postcondition': postcondition,
                    'configuration': configuration
                })

        # 深度优先遍历依赖图，寻找所有路径
        def find_paths(start_condition, path=[]):
            # 如果当前条件没有依赖项，返回当前路径
            if start_condition not in self.pre2post:
                return [path]
            
            paths = []
            for dep in self.pre2post[start_condition]:
                # 构建当前的路径信息
                current_machine = dep['machine']
                postcondition = dep['postcondition']
                configuration = dep['configuration']
                
                # 构建路径描述
                path_description = {
                    'machine': current_machine,
                    'precondition': start_condition,
                    'postcondition': postcondition,
                    'configuration': configuration
                }
                # 递归寻找后续路径
                new_paths = find_paths(postcondition, path + [path_description])
                paths.extend(new_paths)
            
            return paths

        for machine_data in self.machines_data:
            for pattern in machine_data['patterns']:
                start_condition = pattern['preconditions']
                paths_from_start = find_paths(start_condition)
                self.all_paths.extend(paths_from_start)

        # 统计路径总数和路径长度分布
        path_lengths = [len(path) for path in self.all_paths]
        self.path_length_counts = defaultdict(int)

        for length in path_lengths:
            self.path_length_counts[length] += 1

        # 打印统计指标
        print(f"总路径数: {len(self.all_paths)}")
        print("路径长度分布:")
        for length, count in sorted(self.path_length_counts.items()):
            print(f"长度 {length}: {count} 条路径")
        write_json(self.traversal_store_path, self.all_paths)
        
    def show_path_distribution(self):
        # 绘制路径长度分布的条形图
        plt.figure(figsize=(8, 6))
        plt.bar(self.path_length_counts.keys(), self.path_length_counts.values(), color='skyblue')
        plt.xlabel('path length')
        plt.ylabel('number of paths')
        plt.title('Distribution of path lengths')
        plt.xticks(range(1, max(self.path_length_counts.keys()) + 1))
        plt.show()

    def construct_graph(self):
        '''
        对20种机器构造依赖图，邻接矩阵表示
        dependence_graph[i][j] == 1 表示 j 依赖于 i，即 j 紧跟着 i 之后
        '''
        graph = [[0 for _ in range(20)] for _ in range(20)]
        machines_list = []
        for machine_data in self.machines_data:
            machine_name = machine_data['machine']
            machines_list.append(machine_name)
        for path in self.all_paths:
            for i in range(len(path) - 1):
                current_machine = path[i]['machine']
                next_machine = path[i+1]['machine']
                graph[machines_list.index(current_machine)][machines_list.index(next_machine)] = 1
        self.dependence_graph['machines'] = machines_list
        self.dependence_graph['dependence_graph'] = graph
        print("dependence_graph: ", self.dependence_graph)
        write_json(self.graph_store_path, self.dependence_graph)

    @property
    def dependency_num(self):
        all_num = 0
        for i in range(len(self.dependence_graph['dependence_graph'])):
            for j in range(len(self.dependence_graph['dependence_graph'][0])):
                if self.dependence_graph['dependence_graph'][i][j] == 1:
                    all_num += 1
        return all_num

class JSSPDependencyGraph:
    def __init__(self, raw_data_path, jssp_data_path, store_path):
        self.data = read_txt(raw_data_path)
        self.jssp_data_path = jssp_data_path
        self.store_path = store_path
        self.dependence_graphs = [] # 依赖图，邻接矩阵表示的有向图，[i][j] == 1 表示 i 依赖于 j，[i][j] == 0 表示不存在依赖关系

    def preprocess(self):
        raw_data = self.data.split(' +++++++++++++++++++++++++++++')[1:]
        result = []
        for data in raw_data:
            data = data.strip()
        for i in range(0, len(raw_data), 2):
            description = raw_data[i].strip()
            # print("description:", description)
            jssp_data = raw_data[i+1].split('\n')[1:]
            jobs_num = jssp_data[1].split()[0].strip()
            machines_num = jssp_data[1].split()[1].strip()
            jobs_data = []
            jssp_data = jssp_data[2:]
            for line in jssp_data:
                steps = []
                units = line.split()
                for j in range(0, len(units), 2):
                    machine = units[j].strip()
                    time = units[j+1].strip()
                    steps.append({
                        "machine": int(machine),
                        "time": int(time)
                    })
                jobs_data.append({
                    "job": len(jobs_data) + 1,
                    "steps": steps
                })
            result.append({
                "description": description,
                "jobs_num": int(jobs_num),
                "machines_num": int(machines_num),
                "data": jobs_data
            })
        write_json(self.jssp_data_path, result)

    def construct_graph(self):
        '''
        1. 对一个 job，先尝试在线性依赖的条件下向依赖图中添加依赖关系
        2. 检查是否有相对执行序不单调的机器对，一旦发现两台机器之间的相对执行序不单调，则删除这两台机器之间的依赖关系
        3. 重复上述过程知道遍历完所有 job（边遍历边检查）
        '''
        jssp_data = read_json(self.jssp_data_path)
        for data in jssp_data:
            graph = dict()
            graph["description"] = data["description"]
            graph["machines_num"] = data["machines_num"]
            graph["jobs_num"] = data["jobs_num"]
            dependence_graph = [[0 for _ in range(data["machines_num"])] for _ in range(data["machines_num"])]
            # dependence_graph[i][j] == 1 表示 j 依赖于 i，即 j 紧跟着 i 之后
            pre_order_graph = [[0 for _ in range(data["machines_num"])] for _ in range(data["machines_num"])]
            # pre_order_graph[i][j] == 1 表示 i 在 j 之前执行
            for job in data["data"]:
                for i in range(len(job["steps"]) - 1):
                    current_machine = int(job["steps"][i]["machine"])
                    next_machine = int(job["steps"][i+1]["machine"])
                    if pre_order_graph[current_machine - 1][next_machine - 1] == 0:
                        dependence_graph[current_machine - 1][next_machine - 1] = 1
                        pre_order_graph[current_machine - 1][next_machine - 1] = 1
                        # 递归 pre_order_graph
                        for j in range(data["machines_num"]):
                            if pre_order_graph[j][current_machine - 1] == 1:
                                pre_order_graph[j][next_machine - 1] = 1
                    else:
                        # 如果相对执行序不单调，删除依赖关系
                        dependence_graph[next_machine - 1][current_machine - 1] = 0
                        dependence_graph[current_machine - 1][next_machine - 1] = 0
            graph["dependence_graph"] = dependence_graph
            graph["pre_order_graph"] = pre_order_graph
            self.dependence_graphs.append(graph)
        write_json(self.store_path, self.dependence_graphs)

    def dependence_graph_visualization(self, jssp_index):
        '''
        依赖图可视化
        '''
        dependence_graphs_matrix = read_json(self.store_path)
        description = dependence_graphs_matrix[jssp_index]["description"]
        dependence_graph = dependence_graphs_matrix[jssp_index]["dependence_graph"]
        print("dependence_graph:", dependence_graph)
        colors = ['#FFFFFF', '#90b5da']  # 白色对应 0，#FF8247 对应 1
        cmap = mcolors.ListedColormap(colors)
        bounds = [0, 0.5, 1]  # 确保 0 和 1 的值分别对应到两种颜色
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # 绘制热力图
        plt.figure(figsize=(8, 6))
        plt.imshow(dependence_graph, cmap=cmap, norm=norm)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Dependence graph of {description}. \nGraph[y][x] == 1 means x depends on y.')
        plt.colorbar()  # 显示颜色条
        plt.show()

    def pre_order_graph_visualization(self, jssp_index):
        '''
        相对执行序图可视化
        '''
        dependence_graphs_matrix = read_json(self.store_path)
        description = dependence_graphs_matrix[jssp_index]["description"]
        pre_order_graph = dependence_graphs_matrix[jssp_index]["pre_order_graph"]
        print("pre_order_graph:", pre_order_graph)
        # 权重为 1 用浅橙色绘制，为 0 时用白色绘制。坐标轴要说明机器的编号。pre_order_graph[i][j] == 1 表示 i 在 j 之前执行
        colors = ['#FFFFFF', '#90b5da']  # 白色对应 0，#FF8247 对应 1
        cmap = mcolors.ListedColormap(colors)
        bounds = [0, 0.5, 1]  # 确保 0 和 1 的值分别对应到两种颜色
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # 绘制热力图
        plt.figure(figsize=(8, 6))
        plt.imshow(pre_order_graph, cmap=cmap, norm=norm)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Pre-order graph of {description}. \nGraph[y][x] == 1 means y is executed before x.')
        plt.colorbar()  # 显示颜色条
        plt.show()

    def statistics(self):
        # 统计每个 jssp_problem 下依赖关系的数目，绘制依赖数目分布的条形图，横轴为依赖关系数目，纵轴为依赖关系为给定数量的 jssp_problem 的数量
        statistic = []
        max_matrix = 0
        dependence_graphs_matrix = read_json(self.store_path)
        for jssp_index, dependence_graph in enumerate(dependence_graphs_matrix):
            dependence_graph = dependence_graph["dependence_graph"]
            dependence_num = 0
            for i in range(len(dependence_graph)):
                for j in range(len(dependence_graph[0])):
                    dependence_num += dependence_graph[i][j]
            max_matrix = max(max_matrix, len(dependence_graph)*len(dependence_graph[0]))
            statistic.append(dependence_num)
        print("max_matrix: ", max_matrix)
        print("statistic: ", statistic)
        plt.figure(figsize=(8, 6))
        plt.hist(statistic, bins=max_matrix, color='skyblue')
        plt.xlabel('number of dependencies')
        plt.ylabel('number of jssp problems')
        plt.title('Distribution of number of dependencies')
        plt.show()
        

