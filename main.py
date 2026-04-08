from src.preprocess.dependence_graph_traversal import JSSPDependencyGraph, SyntheticDependencyGraph
from concurrent.futures import ThreadPoolExecutor, as_completed
# from src.preprocess.arrange import Arrange
# from src.preprocess.arrange2 import Arrange2
from src.preprocess.arrange3 import Arrange3
from src.preprocess.RouteSheet import RouteSheet
from src.dsl_design.feature import Feature
from src.dsl_design.operation import Operation
from src.dsl_design.production import Production
from src.experiment.baseline import Baseline
from src.experiment.baseline2 import Baseline_2
from src.experiment.dsl_pipeline import DSLPipeline
from src.evaluation.evaluation import Evaluation
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from tqdm import tqdm
from utils.util import read_json, write_json, read_txt, write_txt, seed_set
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='evaluation', choices=['preprocess', 'autodsl_operation', 'autodsl_production', 'baseline', 'baseline_2', 'dsl_pipeline', 'evaluation'])
parser.add_argument('--type', default='CPE_CAE_CSE-2', choices=['CPE_CAE_CSE-2', 'CSE-1', 'SGE', 'DAE'])
parser.add_argument('--evaluation_type', default='CPE', choices=['CPE', 'CAE', 'CSE-1', 'CSE-2', 'SGE', 'DAE'])
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--demo', action='store_true', default=False)
args = parser.parse_args()

legal_instance_description_list = [
    "instance ta71",
    "instance ta72",
    "instance ta73",
    "instance ta74",
    "instance ta75",
    "instance ta76",
    "instance ta77",
    "instance ta78",
    "instance ta79",
    "instance ta80",
]

if __name__ == '__main__':
    seed_set(args.seed)
    if args.mode == "preprocess":
        # 最后的结果: data/route_sheet.json，data/route_sheet_reduce.json，完全结构化
        route_sheet = RouteSheet(
            machines_data_path="data/machines.json", 
            jssp_data_path="data/jssp_data.json", 
            arrange_path="data/arrange.json", 
            jssp_mapped_path="data/jssp_mapped.json", 
            route_sheet_store_path="data/route_sheet.json", route_sheet_reduce_path="data/route_sheet_reduce.json"
        )
        # route_sheet.mapping()
        # route_sheet.create_route_sheet()
        route_sheet.route_sheet_reduce()
    elif args.mode == "autodsl_operation":
        # 最后的结果：outputs/EM_results.json, outputs/operation_dsl.json, outputs/production_dsl.json
        route_sheet_all = read_json("data/route_sheet_reduce.json")
        total_feature = []
        total_operation_dsl = []
        total_likelihood_list = []
        for i, domain_data in enumerate(route_sheet_all):
            if domain_data[0]["instance_description"] not in legal_instance_description_list:
                continue
            print("start process: ", domain_data[0]["instance_description"])
            feature = Feature(
                domain_data=domain_data,
                feature_data_path="data/feature.json"
            )
            operation = Operation(
                feature=feature, 
                operation_dsl_path="data/operation_dsl.json"
            )

            def process_opcode(opcode):
                idx_list_h1, value_list_h1 = feature.feature_vector_extraction(opcode=opcode, hierarchy=1)
                operation.recursive_clustering(opcode, idx_list_h1, value_list_h1, hierarchy=1)
                operation.analyse(opcode)
                return opcode

            # 使用 ThreadPoolExecutor 并行化处理
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_opcode, opcode) for opcode in feature.feature_data]
                
                # 使用 tqdm 来显示进度
                for future in tqdm(as_completed(futures), total=len(futures), desc="Abstraction"):
                    future.result()  # 获取结果，确保异常被捕获

            operation.dsl_regular()

            total_feature.append(feature.feature_data)
            total_operation_dsl.append({
                "instance_description": domain_data[0]["instance_description"],
                "operation_dsl": operation.operation_dsl
            })
            total_likelihood_list.append({
                "instance_description": domain_data[0]["instance_description"],
                "likelihood_list": operation.curve
            })
        write_json("outputs/AutoDSL/feature.json", total_feature)
        write_json("outputs/AutoDSL/total_likelihood_list.json", total_likelihood_list)
        write_json("outputs/AutoDSL/total_operation_dsl.json", total_operation_dsl)

    elif args.mode == "autodsl_production":
        # 最后的结果：outputs/production_dsl.json
        total_EM_results = []
        route_sheet_all = read_json("data/route_sheet_reduce.json")
        total_production_dsl = []
        total_EM_updates = []
        
        for i, domain_data in enumerate(route_sheet_all):
            # 如果 domain_data["instance_description"] 不等于 instance ta71 ~ instance ta80，则跳过
            if(domain_data[0]["instance_description"] not in legal_instance_description_list):
                continue
            print("start process: ", domain_data[0]["instance_description"])
            production = Production(
                domain_data=domain_data
            )
            production.extract()
            production.EM_extract()
            total_production_dsl.append({
                "instance_description": domain_data[0]["instance_description"],
                "production_dsl": production.production_dsl
            })
            total_EM_results.append({
                "instance_description": domain_data[0]["instance_description"],
                "EM_results": production.EM_results
            })
            total_EM_updates.append({
                "instance_description": domain_data[0]["instance_description"],
                "updates": production.EM_updates
            })
        write_json("outputs/AutoDSL/total_EM_updates.json", total_EM_updates)
        write_json("outputs/AutoDSL/total_production_dsl.json", total_production_dsl)
        write_json("outputs/AutoDSL/EM_results.json", total_EM_results)

    elif args.mode == "baseline":
        route_sheet_all = read_json("data/route_sheet_reduce.json")
        route_sheet_choosed = [route_sheet for route_sheet in route_sheet_all if route_sheet[0]["instance_description"] in legal_instance_description_list]
        
        for route_sheet in tqdm(route_sheet_choosed):
            baseline = Baseline(structured_route_sheet=route_sheet, instance_description=route_sheet[0]["instance_description"], experiment_type=args.type)
            baseline.run()

    elif args.mode == "baseline_2":
        route_sheet_all = read_json("data/route_sheet_reduce.json")
        route_sheet_choosed = [route_sheet for route_sheet in route_sheet_all if route_sheet[0]["instance_description"] in legal_instance_description_list]

        for route_sheet in tqdm(route_sheet_choosed):
            baseline_2 = Baseline_2(structured_route_sheet=route_sheet, instance_description=route_sheet[0]["instance_description"], experiment_type=args.type)
            baseline_2.run()

    elif args.mode == "dsl_pipeline":
        route_sheet_all = read_json("data/route_sheet_reduce.json")
        route_sheet_choosed = [route_sheet for route_sheet in route_sheet_all if route_sheet[0]["instance_description"] in legal_instance_description_list]

        dsl_pipeline = DSLPipeline([], production_dsl={}, operation_dsl={}, EM_structure={}, instance_description="", experiment_type=args.type)

        for route_sheet in tqdm(route_sheet_choosed):
            production_dsls = read_json("outputs/AutoDSL/total_production_dsl.json")
            operation_dsls = read_json("outputs/AutoDSL/total_operation_dsl.json")
            EM_structures = read_json("outputs/AutoDSL/EM_results.json")

            production_dsl = [
                production_dsl["production_dsl"] for production_dsl in production_dsls 
                if production_dsl["instance_description"] == route_sheet[0]["instance_description"]
            ][0]
            operation_dsl = [
                operation_dsl["operation_dsl"] for operation_dsl in operation_dsls 
                if operation_dsl["instance_description"] == route_sheet[0]["instance_description"]
            ][0]
            EM_structure = [
                EM_structure["EM_results"] for EM_structure in EM_structures 
                if EM_structure["instance_description"] == route_sheet[0]["instance_description"]
            ][0]

            dsl_pipeline.production_dsl = production_dsl
            dsl_pipeline.operation_dsl = operation_dsl
            dsl_pipeline.EM_structure = EM_structure
            dsl_pipeline.instance_description = route_sheet[0]["instance_description"]
            dsl_pipeline.run()
    
    elif args.mode == "evaluation":
        evaluation = Evaluation()
        evaluation.evaluate(experiment_type=args.evaluation_type)
    
    else:
        raise ValueError("Invalid mode. Please choose from preprocess, autodsl, baseline, dsl_pipeline, evaluate.")
    
    
    
    
    
    
    
    
    
    
    
    # machines_data_path = "data/machines.json"
    # traversal_store_path = "data/synthetic_traversal.json"
    # graph_store_path = "data/synthetic_graph.json"
    # synthetic_dependency_graph = SyntheticDependencyGraph(machines_data_path, traversal_store_path, graph_store_path)
    # synthetic_dependency_graph.traversal()
    # synthetic_dependency_graph.construct_graph()
    # print("dependency_num: ", synthetic_dependency_graph.dependency_num)
    

    # raw_data_path = "data/jssp_data.txt"
    # jssp_data_path = "data/jssp_data.json"
    # jssp_graph_store_path = "data/jssp_graph.json"
    # jssp_dependency_graph = JSSPDependencyGraph(raw_data_path, jssp_data_path, jssp_graph_store_path)
    # jssp_dependency_graph.statistics()

    # synthetic_graph_path = "data/synthetic_graph.json"
    # jssp_graph_path = "data/jssp_graph.json"
    # arrange_store_path = "data/arrange.json"
    # arrange_apply_path = "data/jssp_graph_arrange"
    # arrange = Arrange3(synthetic_graph_path, jssp_graph_path, arrange_store_path, arrange_apply_path)
    # arrange.arrange()
    # # arrange.statistics()

    
