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
        # Build dependencies using each machine pattern as a graph node.
        for machine_data in self.machines_data:
            machine_name = machine_data['machine']
            for pattern in machine_data['patterns']:
                precondition = pattern['preconditions']
                postcondition = pattern['postconditions']
                configuration = pattern['configuration']
                
                # Store the machine together with its pattern metadata.
                if precondition not in self.pre2post:
                    self.pre2post[precondition] = []
                self.pre2post[precondition].append({
                    'machine': machine_name,
                    'postcondition': postcondition,
                    'configuration': configuration
                })

        # Traverse the dependency graph with DFS to enumerate all paths.
        def find_paths(start_condition, path=[]):
            # Return the current path when the condition has no outgoing edge.
            if start_condition not in self.pre2post:
                return [path]
            
            paths = []
            for dep in self.pre2post[start_condition]:
                # Build the current path item.
                current_machine = dep['machine']
                postcondition = dep['postcondition']
                configuration = dep['configuration']
                
                # Describe the current transition.
                path_description = {
                    'machine': current_machine,
                    'precondition': start_condition,
                    'postcondition': postcondition,
                    'configuration': configuration
                }
                # Recursively extend downstream paths.
                new_paths = find_paths(postcondition, path + [path_description])
                paths.extend(new_paths)
            
            return paths

        for machine_data in self.machines_data:
            for pattern in machine_data['patterns']:
                start_condition = pattern['preconditions']
                paths_from_start = find_paths(start_condition)
                self.all_paths.extend(paths_from_start)

        # Count total paths and the path-length distribution.
        path_lengths = [len(path) for path in self.all_paths]
        self.path_length_counts = defaultdict(int)

        for length in path_lengths:
            self.path_length_counts[length] += 1

        # Print summary statistics.
        print(f"Total number of paths: {len(self.all_paths)}")
        print("Path length distribution:")
        for length, count in sorted(self.path_length_counts.items()):
            print(f"Length {length}: {count} paths")
        write_json(self.traversal_store_path, self.all_paths)
        
    def show_path_distribution(self):
        # Plot the path-length distribution.
        plt.figure(figsize=(8, 6))
        plt.bar(self.path_length_counts.keys(), self.path_length_counts.values(), color='skyblue')
        plt.xlabel('path length')
        plt.ylabel('number of paths')
        plt.title('Distribution of path lengths')
        plt.xticks(range(1, max(self.path_length_counts.keys()) + 1))
        plt.show()

    def construct_graph(self):
        """
        Construct the 20-machine dependency graph as an adjacency matrix.
        dependence_graph[i][j] == 1 means machine j depends on machine i.
        """
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
        self.dependence_graphs = []  # Directed dependency graphs stored as adjacency matrices.

    def preprocess(self):
        raw_data = self.data.split(' +++++++++++++++++++++++++++++')[1:]
        result = []
        for i in range(0, len(raw_data), 2):
            description = raw_data[i].strip()
            jssp_data = raw_data[i+1].split('\n')[1:]
            jobs_num = int(jssp_data[1].split()[0].strip())
            machines_num = int(jssp_data[1].split()[1].strip())
            jobs_data = []
            # Only consume exactly jobs_num data lines; ignore trailing blank lines
            # that would otherwise be appended as phantom `{"job": N+1, "steps": []}`.
            for line in jssp_data[2:]:
                if len(jobs_data) >= jobs_num:
                    break
                units = line.split()
                if not units:
                    continue
                steps = []
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
                "jobs_num": jobs_num,
                "machines_num": machines_num,
                "data": jobs_data
            })
        write_json(self.jssp_data_path, result)

    def construct_graph(self):
        """
        1. For each job, first add dependencies under the linear execution order.
        2. Remove a dependency if a machine pair shows inconsistent relative order.
        3. Repeat the process while traversing all jobs.
        """
        jssp_data = read_json(self.jssp_data_path)
        for data in jssp_data:
            graph = dict()
            graph["description"] = data["description"]
            graph["machines_num"] = data["machines_num"]
            graph["jobs_num"] = data["jobs_num"]
            dependence_graph = [[0 for _ in range(data["machines_num"])] for _ in range(data["machines_num"])]
            # dependence_graph[i][j] == 1 means machine j depends on machine i.
            pre_order_graph = [[0 for _ in range(data["machines_num"])] for _ in range(data["machines_num"])]
            # pre_order_graph[i][j] == 1 means machine i is executed before machine j.
            for job in data["data"]:
                for i in range(len(job["steps"]) - 1):
                    current_machine = int(job["steps"][i]["machine"])
                    next_machine = int(job["steps"][i+1]["machine"])
                    if pre_order_graph[current_machine - 1][next_machine - 1] == 0:
                        dependence_graph[current_machine - 1][next_machine - 1] = 1
                        pre_order_graph[current_machine - 1][next_machine - 1] = 1
                        # Propagate transitive precedence relations.
                        for j in range(data["machines_num"]):
                            if pre_order_graph[j][current_machine - 1] == 1:
                                pre_order_graph[j][next_machine - 1] = 1
                    else:
                        # Remove the dependency when the relative order is inconsistent.
                        dependence_graph[next_machine - 1][current_machine - 1] = 0
                        dependence_graph[current_machine - 1][next_machine - 1] = 0
            graph["dependence_graph"] = dependence_graph
            graph["pre_order_graph"] = pre_order_graph
            self.dependence_graphs.append(graph)
        write_json(self.store_path, self.dependence_graphs)

    def dependence_graph_visualization(self, jssp_index):
        """Visualize a dependency graph."""
        dependence_graphs_matrix = read_json(self.store_path)
        description = dependence_graphs_matrix[jssp_index]["description"]
        dependence_graph = dependence_graphs_matrix[jssp_index]["dependence_graph"]
        print("dependence_graph:", dependence_graph)
        colors = ['#FFFFFF', '#90b5da']  # White for 0 and blue for 1.
        cmap = mcolors.ListedColormap(colors)
        bounds = [0, 0.5, 1]  # Map 0 and 1 to distinct colors.
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Draw the heatmap.
        plt.figure(figsize=(8, 6))
        plt.imshow(dependence_graph, cmap=cmap, norm=norm)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Dependence graph of {description}. \nGraph[y][x] == 1 means x depends on y.')
        plt.colorbar()  # Show the color bar.
        plt.show()

    def pre_order_graph_visualization(self, jssp_index):
        """Visualize a precedence-order graph."""
        dependence_graphs_matrix = read_json(self.store_path)
        description = dependence_graphs_matrix[jssp_index]["description"]
        pre_order_graph = dependence_graphs_matrix[jssp_index]["pre_order_graph"]
        print("pre_order_graph:", pre_order_graph)
        # pre_order_graph[i][j] == 1 means machine i is executed before machine j.
        colors = ['#FFFFFF', '#90b5da']  # White for 0 and blue for 1.
        cmap = mcolors.ListedColormap(colors)
        bounds = [0, 0.5, 1]  # Map 0 and 1 to distinct colors.
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Draw the heatmap.
        plt.figure(figsize=(8, 6))
        plt.imshow(pre_order_graph, cmap=cmap, norm=norm)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Pre-order graph of {description}. \nGraph[y][x] == 1 means y is executed before x.')
        plt.colorbar()  # Show the color bar.
        plt.show()

    def statistics(self):
        # Plot the distribution of dependency counts across JSSP instances.
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
        
