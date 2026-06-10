from src.preprocess.dependence_graph_traversal import JSSPDependencyGraph, SyntheticDependencyGraph
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.preprocess.arrange3 import Arrange3
from src.preprocess.RouteSheet import RouteSheet
from src.dsl_design.feature import Feature
from src.dsl_design.operation import Operation
from src.dsl_design.production import Production
from src.experiment.baseline import Baseline
from src.experiment.baseline2 import Baseline_2
from src.experiment.dsl_pipeline import DSLPipeline
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from tqdm import tqdm
import numpy as np
from utils.util import read_json, write_json, read_txt, write_txt, seed_set
import argparse
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os
import json
import copy
import difflib
from collections import Counter, defaultdict


class Evaluation:
    def __init__(self):
        self.mode = ["Baseline-", "Baseline2-", "DSLPipeline-", "GroundTruth", "UnifiedPipeline-"]
        self.suffix = ["CPE_CAE_CSE-2/", "CSE-1/", "SGE/"]
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.result_path = "outputs/"
        self.groundtruth_dir_path = self.result_path + self.mode[3]

    def evaluate(self, experiment_type: str):
        """
        experiment_type: str, CPE | CAE | CSE-1 | CSE-2 | SGE
        CPE:
            - Input: NL description
            - Output: production plan
            - Methods: ROUGE-L, BLEU
        CAE:
            - Input: NL description
            - Output: fully structural route sheet
            - Methods: BLEU, ROUGE-L
        CSE-1:
            - Input: groundtruth fully structural route sheet
            - Output: or matirx
            - Methods:
                - Constraint-Level Accuracy: evaluate constraint consistency
                - Compiler Error Rate
                - Runtime Error Rate
        CSE-2:
            - Input: NL description
            - Output: or matirx
            - Methods: same as CSE-1
        SGE:
            - Input: JSP solver result (assigned_jobs.json)
            - Output: production plan
            - Methods: BLEU, ROUGE-L
        DAE:
        """
        if experiment_type == "CPE":
            baseline_dir_path = self.result_path + self.mode[0] + self.suffix[0]
            baseline2_dir_path = self.result_path + self.mode[1] + self.suffix[0]
            dsl_pipeline_dir_path = self.result_path + self.mode[2] + self.suffix[0]
            unified_dir_path = self.result_path + self.mode[4] + self.suffix[0]

            baseline_bleu = []
            baseline2_bleu = []
            dsl_bleu = []
            unified_bleu = []

            baseline_rouge = {"Precision": [], "Recall": [], "F1": []}
            baseline2_rouge = {"Precision": [], "Recall": [], "F1": []}
            dsl_rouge = {"Precision": [], "Recall": [], "F1": []}
            unified_rouge = {"Precision": [], "Recall": [], "F1": []}


            # List subfolders, assuming aligned names across methods.
            subfolders = os.listdir(baseline_dir_path)

            for subfolder in tqdm(subfolders):
                baseline_path = os.path.join(baseline_dir_path, subfolder, "production_plan.json")
                baseline2_path = os.path.join(baseline2_dir_path, subfolder, "production_plan.json")
                dsl_pipeline_path = os.path.join(dsl_pipeline_dir_path, subfolder, "production_plan.json")
                unified_path = os.path.join(unified_dir_path, subfolder, "production_plan.json")
                ground_truth_path = os.path.join(self.groundtruth_dir_path, subfolder, "production_plan.json")

                required = [baseline_path, baseline2_path, dsl_pipeline_path, ground_truth_path]
                if all(os.path.exists(path) for path in required):
                    # Read JSON content.
                    baseline_content = json.dumps(read_json(baseline_path))
                    baseline2_content = json.dumps(read_json(baseline2_path))
                    dsl_pipeline_content = json.dumps(read_json(dsl_pipeline_path))
                    ground_truth_content = json.dumps(read_json(ground_truth_path))

                    # Compute BLEU scores.
                    baseline_bleu.append(self.__bleu_score(ground_truth_content, baseline_content))
                    baseline2_bleu.append(self.__bleu_score(ground_truth_content, baseline2_content))
                    dsl_bleu.append(self.__bleu_score(ground_truth_content, dsl_pipeline_content))

                    # Compute ROUGE scores.
                    baseline_rouge_precision, baseline_rouge_recall, baseline_rouge_F1 = self.__rouge_score(ground_truth_content, baseline_content)
                    baseline2_rouge_precision, baseline2_rouge_recall, baseline2_rouge_F1 = self.__rouge_score(ground_truth_content, baseline2_content)
                    dsl_rouge_precision, dsl_rouge_recall, dsl_rouge_F1 = self.__rouge_score(ground_truth_content, dsl_pipeline_content)

                    baseline_rouge["Precision"].append(baseline_rouge_precision)
                    baseline_rouge["Recall"].append(baseline_rouge_recall)
                    baseline_rouge["F1"].append(baseline_rouge_F1)

                    baseline2_rouge["Precision"].append(baseline2_rouge_precision)
                    baseline2_rouge["Recall"].append(baseline2_rouge_precision)
                    baseline2_rouge["F1"].append(baseline2_rouge_precision)

                    dsl_rouge["Precision"].append(dsl_rouge_precision)
                    dsl_rouge["Recall"].append(dsl_rouge_recall)
                    dsl_rouge["F1"].append(dsl_rouge_F1)

                    # Unified pipeline scores
                    if os.path.exists(unified_path):
                        unified_pp = read_json(unified_path)

                        # Name alignment: map unified names to GT names
                        gt_rs_path = os.path.join(self.groundtruth_dir_path, subfolder, "route_sheets.json")
                        u_rs_path = os.path.join(unified_dir_path, subfolder, "route_sheets.json")
                        if os.path.exists(gt_rs_path) and os.path.exists(u_rs_path):
                            name_maps = self._build_name_alignment(read_json(gt_rs_path), read_json(u_rs_path))
                            unified_pp = self._align_production_plan(unified_pp, name_maps)

                        unified_content = json.dumps(unified_pp)
                        unified_bleu.append(self.__bleu_score(ground_truth_content, unified_content))
                        u_p, u_r, u_f = self.__rouge_score(ground_truth_content, unified_content)
                        unified_rouge["Precision"].append(u_p)
                        unified_rouge["Recall"].append(u_r)
                        unified_rouge["F1"].append(u_f)

            print("baseline_bleu: ", baseline_bleu)
            print("baseline2_bleu: ", baseline2_bleu)
            print("dsl_bleu: ", dsl_bleu)
            print("unified_bleu: ", unified_bleu)

            print("baseline_rouge: ", baseline_rouge)
            print("baseline2_rouge: ", baseline2_rouge)
            print("dsl_rouge: ", dsl_rouge)
            print("unified_rouge: ", unified_rouge)

            write_json("outputs/Evaluation/CPE_baseline_bleu.json", baseline_bleu)
            write_json("outputs/Evaluation/CPE_baseline2_bleu.json", baseline2_bleu)
            write_json("outputs/Evaluation/CPE_dsl_bleu.json", dsl_bleu)
            write_json("outputs/Evaluation/CPE_unified_bleu.json", unified_bleu)

            write_json("outputs/Evaluation/CPE_baseline_rouge.json", baseline_rouge)
            write_json("outputs/Evaluation/CPE_baseline2_rouge.json", baseline2_rouge)
            write_json("outputs/Evaluation/CPE_dsl_rouge.json", dsl_rouge)
            write_json("outputs/Evaluation/CPE_unified_rouge.json", unified_rouge)

        elif experiment_type == "CAE":
            baseline_dir_path = self.result_path + self.mode[0] + self.suffix[0]
            dsl_pipeline_dir_path = self.result_path + self.mode[2] + self.suffix[0]
            
            baseline_bleu = []
            dsl_bleu = []

            baseline_rouge = {"Precision": [], "Recall": [], "F1": []}
            dsl_rouge = {"Precision": [], "Recall": [], "F1": []}
            

            # List subfolders, assuming aligned names across methods.
            subfolders = os.listdir(baseline_dir_path)

            for subfolder in tqdm(subfolders):
                baseline_path = os.path.join(baseline_dir_path, subfolder, "structural_info.json")
                dsl_pipeline_path = os.path.join(dsl_pipeline_dir_path, subfolder, "route_sheets.json")
                ground_truth_path = os.path.join(self.groundtruth_dir_path, subfolder, "route_sheets.json")

                if all(os.path.exists(path) for path in [baseline_path, dsl_pipeline_path, ground_truth_path]):
                    # Read JSON content.
                    baseline_content = json.dumps(read_json(baseline_path))
                    dsl_pipeline_content = json.dumps(read_json(dsl_pipeline_path))
                    ground_truth_content = json.dumps(read_json(ground_truth_path))

                    # Compute BLEU scores.
                    baseline_bleu.append(self.__bleu_score(ground_truth_content, baseline_content))
                    dsl_bleu.append(self.__bleu_score(ground_truth_content, dsl_pipeline_content))

                    # Compute ROUGE scores.
                    baseline_rouge_precision, baseline_rouge_recall, baseline_rouge_F1 = self.__rouge_score(ground_truth_content, baseline_content)
                    dsl_rouge_precision, dsl_rouge_recall, dsl_rouge_F1 = self.__rouge_score(ground_truth_content, dsl_pipeline_content)

                    baseline_rouge["Precision"].append(baseline_rouge_precision)
                    baseline_rouge["Recall"].append(baseline_rouge_recall)
                    baseline_rouge["F1"].append(baseline_rouge_F1)

                    dsl_rouge["Precision"].append(dsl_rouge_precision)
                    dsl_rouge["Recall"].append(dsl_rouge_recall)
                    dsl_rouge["F1"].append(dsl_rouge_F1)

            print("baseline_bleu: ", baseline_bleu)
            print("dsl_bleu: ", dsl_bleu)

            print("baseline_rouge: ", baseline_rouge)
            print("dsl_rouge: ", dsl_rouge)


            write_json("outputs/Evaluation/CAE_baseline_bleu.json", baseline_bleu)
            write_json("outputs/Evaluation/CAE_dsl_bleu.json", dsl_bleu)

            write_json("outputs/Evaluation/CAE_baseline_rouge.json", baseline_rouge)
            write_json("outputs/Evaluation/CAE_dsl_rouge.json", dsl_rouge)

        elif experiment_type == "CSE-1":
            baseline_dir_path = self.result_path + self.mode[0] + self.suffix[1]
            dsl_pipeline_dir_path = self.result_path + self.mode[2] + self.suffix[1]
            
            baseline_bleu = []
            dsl_bleu = []

            baseline_result = {"accuracy_rate": [], "compile_err_rate": [], "runtime_err_rate": []}
            dsl_result = {"accuracy_rate": [], "compile_err_rate": [], "runtime_err_rate": []}
            # List subfolders, assuming aligned names across methods.
            subfolders = os.listdir(baseline_dir_path)

            for subfolder in tqdm(subfolders):
                baseline_path = os.path.join(baseline_dir_path, subfolder, "or_matrix.json")
                baseline_route_sheet = os.path.join(os.path.join(baseline_dir_path, subfolder, "route_sheets.json"))
                baseline_machines = os.path.join(os.path.join(baseline_dir_path, subfolder, "machines.json"))

                dsl_pipeline_path = os.path.join(dsl_pipeline_dir_path, subfolder, "or_matrix.json")
                dsl_pipeline_operation_programs_path = os.path.join(dsl_pipeline_dir_path, subfolder, "operation_programs.json")
                dsl_pipeline_production_programs_path = os.path.join(dsl_pipeline_dir_path, subfolder, "production_programs.json")

                ground_truth_path = os.path.join(self.groundtruth_dir_path, subfolder, "or_matrix.json")
                ground_truth_route_sheet = os.path.join(os.path.join(self.groundtruth_dir_path, subfolder, "route_sheets.json"))

                if all(os.path.exists(path) for path in [baseline_path, dsl_pipeline_path, ground_truth_path, baseline_route_sheet, dsl_pipeline_operation_programs_path, dsl_pipeline_production_programs_path, ground_truth_route_sheet]):
                    # Read JSON content.
                    baseline_content = read_json(baseline_path)
                    baseline_route_sheet_content = read_json(baseline_route_sheet)
                    baseline_machines_content = read_json(baseline_machines)

                    dsl_pipeline_content = read_json(dsl_pipeline_path)
                    dsl_pipeline_operation_programs_content = read_json(dsl_pipeline_operation_programs_path)
                    dsl_pipeline_production_programs_content = read_json(dsl_pipeline_production_programs_path)

                    ground_truth_content = read_json(ground_truth_path)
                    ground_truth_route_sheet_content = read_json(ground_truth_route_sheet)

                    baseline_resource_constraints = self.__get_baseline_recourse_constraint_CSE_1(baseline_content, baseline_machines_content, baseline_route_sheet_content)
                    baseline_precedence_constraints = self.__get_baseline_precedence_constraint_CSE_1(baseline_content, baseline_route_sheet_content, baseline_machines_content)

                    dsl_resource_constraints = self.__get_DSL_recourse_constraint_CSE(dsl_pipeline_operation_programs_content)
                    dsl_precedence_constraints = self.__get_DSL_precedence_constraint_CSE(dsl_pipeline_production_programs_content)

                    ground_truth_resource_constraints = self.__get_groundtruth_recourse_constraint(ground_truth_route_sheet_content)
                    ground_truth_operation_precedence_constraints = self.__get_groundtruth_operation_precedence_constraint(ground_truth_content, ground_truth_route_sheet_content)
                    ground_truth_machine_precedence_constraints = self.__get_groundtruth_machine_precedence_constraint(ground_truth_content, ground_truth_route_sheet_content)

                    baseline_result["accuracy_rate"].append(
                        self.__iou(baseline_resource_constraints + baseline_precedence_constraints,  
                        ground_truth_resource_constraints + ground_truth_machine_precedence_constraints)
                    )
                    baseline_result["runtime_err_rate"].append(
                        float(read_txt(os.path.join(baseline_dir_path, subfolder, "err_rate.txt")))
                    )
                    dsl_result["accuracy_rate"].append(self.__iou(
                        dsl_resource_constraints + dsl_precedence_constraints,
                        ground_truth_resource_constraints + ground_truth_operation_precedence_constraints
                    ))
                    dsl_result["runtime_err_rate"].append(
                        float(read_txt(os.path.join(dsl_pipeline_dir_path, subfolder, "err_rate.txt")))
                    )

            print("baseline_result: ", baseline_result)
            print("dsl_result: ", dsl_result)

            write_json("outputs/Evaluation/CSE-1_baseline.json", baseline_result)
            write_json("outputs/Evaluation/CSE-1_dsl.json", dsl_result)

        elif experiment_type == "CSE-2":
            baseline_dir_path = self.result_path + self.mode[0] + self.suffix[0]
            dsl_pipeline_dir_path = self.result_path + self.mode[2] + self.suffix[0]
            unified_dir_path = self.result_path + self.mode[4] + self.suffix[0]

            baseline_bleu = []
            dsl_bleu = []

            baseline_result = {"accuracy_rate": [], "runtime_err_rate": []}
            baseline2_result = {"accuracy_rate": [], "runtime_err_rate": []}
            dsl_result = {"accuracy_rate": [], "runtime_err_rate": []}
            unified_result = {"accuracy_rate": [], "runtime_err_rate": []}
            # List subfolders, assuming aligned names across methods.
            subfolders = os.listdir(baseline_dir_path)

            for subfolder in tqdm(subfolders):
                baseline_path = os.path.join(baseline_dir_path, subfolder, "or_matrix.json")
                baseline_route_sheet = os.path.join(os.path.join(baseline_dir_path, subfolder, "structural_info.json"))
                baseline_machines = os.path.join(os.path.join(baseline_dir_path, subfolder, "machines.json"))

                baseline2_path = os.path.join(baseline_dir_path, subfolder, "or_matrix.json")
                baseline2_machines = os.path.join(os.path.join(baseline_dir_path, subfolder, "machines.json"))

                dsl_pipeline_path = os.path.join(dsl_pipeline_dir_path, subfolder, "or_matrix.json")
                dsl_pipeline_operation_programs_path = os.path.join(dsl_pipeline_dir_path, subfolder, "operation_programs.json")
                dsl_pipeline_production_programs_path = os.path.join(dsl_pipeline_dir_path, subfolder, "production_programs.json")

                unified_or_path = os.path.join(unified_dir_path, subfolder, "or_matrix.json")
                unified_rs_path = os.path.join(unified_dir_path, subfolder, "route_sheets.json")

                ground_truth_path = os.path.join(self.groundtruth_dir_path, subfolder, "or_matrix.json")
                ground_truth_route_sheet = os.path.join(os.path.join(self.groundtruth_dir_path, subfolder, "route_sheets.json"))

                if all(os.path.exists(path) for path in [baseline_path, dsl_pipeline_path, ground_truth_path, baseline_route_sheet, dsl_pipeline_operation_programs_path, dsl_pipeline_production_programs_path, ground_truth_route_sheet]):
                    # Read JSON content.
                    baseline_content = read_json(baseline_path)
                    baseline_route_sheet_content = read_json(baseline_route_sheet)
                    baseline_machines_content = read_json(baseline_machines)

                    baseline2_content = read_json(baseline2_path)
                    baseline2_machines_content = read_json(baseline2_machines)

                    dsl_pipeline_content = read_json(dsl_pipeline_path)
                    dsl_pipeline_operation_programs_content = read_json(dsl_pipeline_operation_programs_path)
                    dsl_pipeline_production_programs_content = read_json(dsl_pipeline_production_programs_path)

                    ground_truth_content = read_json(ground_truth_path)
                    ground_truth_route_sheet_content = read_json(ground_truth_route_sheet)

                    baseline_resource_constraints = self.__get_baseline_recourse_constraint_CSE_2(baseline_content, baseline_machines_content, baseline_route_sheet_content)
                    baseline_precedence_constraints = self.__get_baseline_precedence_constraint_CSE_1(baseline_content, baseline_route_sheet_content, baseline_machines_content)

                    baseline2_resource_constraints = self.__get_baseline_recourse_constraint_CSE_2(baseline2_content, baseline2_machines_content, baseline_route_sheet_content)
                    baseline2_precedence_constraints = self.__get_baseline_precedence_constraint_CSE_1(baseline2_content, baseline_route_sheet_content, baseline2_machines_content)

                    dsl_resource_constraints = self.__get_DSL_recourse_constraint_CSE(dsl_pipeline_operation_programs_content)
                    dsl_precedence_constraints = self.__get_DSL_precedence_constraint_CSE(dsl_pipeline_production_programs_content)

                    ground_truth_resource_constraints = self.__get_groundtruth_recourse_constraint(ground_truth_route_sheet_content)
                    ground_truth_operation_precedence_constraints = self.__get_groundtruth_operation_precedence_constraint(ground_truth_content, ground_truth_route_sheet_content)
                    ground_truth_machine_precedence_constraints = self.__get_groundtruth_machine_precedence_constraint(ground_truth_content, ground_truth_route_sheet_content)

                    baseline_result["accuracy_rate"].append(
                        self.__iou(baseline_resource_constraints + baseline_precedence_constraints,
                        ground_truth_resource_constraints + ground_truth_machine_precedence_constraints)
                    )
                    baseline_result["runtime_err_rate"].append(
                        float(read_txt(os.path.join(baseline_dir_path, subfolder, "err_rate.txt")))
                    )

                    baseline2_result["accuracy_rate"].append(
                        self.__iou(baseline2_resource_constraints + baseline2_precedence_constraints,
                        ground_truth_resource_constraints + ground_truth_machine_precedence_constraints)
                    )
                    baseline2_result["runtime_err_rate"].append(
                        float(read_txt(os.path.join(baseline_dir_path, subfolder, "err_rate.txt")))
                    )

                    dsl_result["accuracy_rate"].append(self.__iou(
                        dsl_resource_constraints + dsl_precedence_constraints,
                        ground_truth_resource_constraints + ground_truth_operation_precedence_constraints
                    ))
                    dsl_result["runtime_err_rate"].append(
                        float(read_txt(os.path.join(dsl_pipeline_dir_path, subfolder, "err_rate.txt")))
                    )

                    # Unified pipeline
                    if os.path.exists(unified_or_path) and os.path.exists(unified_rs_path):
                        unified_or_content = read_json(unified_or_path)
                        unified_rs_content = read_json(unified_rs_path)

                        # Name alignment: map unified names to GT names
                        name_maps = self._build_name_alignment(ground_truth_route_sheet_content, unified_rs_content)
                        aligned_rs = self._align_route_sheets(unified_rs_content, name_maps)
                        write_json(os.path.join(unified_dir_path, subfolder, "name_alignment.json"), name_maps)

                        unified_resource = self.__get_unified_resource_constraint_CSE(aligned_rs)
                        unified_precedence = self.__get_unified_precedence_constraint_CSE(unified_or_content, aligned_rs)
                        unified_result["accuracy_rate"].append(self.__iou(
                            unified_resource + unified_precedence,
                            ground_truth_resource_constraints + ground_truth_operation_precedence_constraints
                        ))
                        unified_result["runtime_err_rate"].append(
                            float(read_txt(os.path.join(unified_dir_path, subfolder, "err_rate.txt")))
                        )

            print("baseline_result: ", baseline_result)
            print("baseline2_result: ", baseline2_result)
            print("dsl_result: ", dsl_result)
            print("unified_result: ", unified_result)

            write_json("outputs/Evaluation/CSE-2_baseline.json", baseline_result)
            write_json("outputs/Evaluation/CSE-2_baseline2.json", baseline2_result)
            write_json("outputs/Evaluation/CSE-2_dsl.json", dsl_result)
            write_json("outputs/Evaluation/CSE-2_unified.json", unified_result)

        elif experiment_type == "SGE":
            baseline_dir_path = self.result_path + self.mode[0] + self.suffix[2]
            baseline2_dir_path = self.result_path + self.mode[1] + self.suffix[2]
            dsl_pipeline_dir_path = self.result_path + self.mode[2] + self.suffix[2]
            
            baseline_bleu = []
            baseline2_bleu = []
            dsl_bleu = []

            baseline_rouge = {"Precision": [], "Recall": [], "F1": []}
            baseline2_rouge = {"Precision": [], "Recall": [], "F1": []}
            dsl_rouge = {"Precision": [], "Recall": [], "F1": []}
            

            # List subfolders, assuming aligned names across methods.
            subfolders = os.listdir(baseline_dir_path)

            for subfolder in tqdm(subfolders):
                baseline_path = os.path.join(baseline_dir_path, subfolder, "production_plan.json")
                baseline2_path = os.path.join(baseline2_dir_path, subfolder, "production_plan.json")
                dsl_pipeline_path = os.path.join(dsl_pipeline_dir_path, subfolder, "production_plan.json")
                ground_truth_path = os.path.join(self.groundtruth_dir_path, subfolder, "production_plan.json")

                if all(os.path.exists(path) for path in [baseline_path, baseline2_path, dsl_pipeline_path, ground_truth_path]):
                    # Read JSON content.
                    baseline_content = json.dumps(read_json(baseline_path))
                    baseline2_content = json.dumps(read_json(baseline2_path))
                    dsl_pipeline_content = json.dumps(read_json(dsl_pipeline_path))
                    ground_truth_content = json.dumps(read_json(ground_truth_path))

                    # Compute BLEU scores.
                    baseline_bleu.append(self.__bleu_score(ground_truth_content, baseline_content))
                    baseline2_bleu.append(self.__bleu_score(ground_truth_content, baseline2_content))
                    dsl_bleu.append(self.__bleu_score(ground_truth_content, dsl_pipeline_content))

                    # Compute ROUGE scores.
                    baseline_rouge_precision, baseline_rouge_recall, baseline_rouge_F1 = self.__rouge_score(ground_truth_content, baseline_content)
                    baseline2_rouge_precision, baseline2_rouge_recall, baseline2_rouge_F1 = self.__rouge_score(ground_truth_content, baseline2_content)
                    dsl_rouge_precision, dsl_rouge_recall, dsl_rouge_F1 = self.__rouge_score(ground_truth_content, dsl_pipeline_content)

                    baseline_rouge["Precision"].append(baseline_rouge_precision)
                    baseline_rouge["Recall"].append(baseline_rouge_recall)
                    baseline_rouge["F1"].append(baseline_rouge_F1)

                    baseline2_rouge["Precision"].append(baseline2_rouge_precision)
                    baseline2_rouge["Recall"].append(baseline2_rouge_precision)
                    baseline2_rouge["F1"].append(baseline2_rouge_precision)

                    dsl_rouge["Precision"].append(dsl_rouge_precision)
                    dsl_rouge["Recall"].append(dsl_rouge_recall)
                    dsl_rouge["F1"].append(dsl_rouge_F1)

            print("baseline_bleu: ", baseline_bleu)
            print("baseline2_bleu: ", baseline2_bleu)
            print("dsl_bleu: ", dsl_bleu)

            print("baseline_rouge: ", baseline_rouge)
            print("baseline2_rouge: ", baseline2_rouge)
            print("dsl_rouge: ", dsl_rouge)

            write_json("outputs/Evaluation/SGE_baseline_bleu.json", baseline_bleu)
            write_json("outputs/Evaluation/SGE_baseline2_bleu.json", baseline2_bleu)
            write_json("outputs/Evaluation/SGE_dsl_bleu.json", dsl_bleu)

            write_json("outputs/Evaluation/SGE_baseline_rouge.json", baseline_rouge)
            write_json("outputs/Evaluation/SGE_baseline2_rouge.json", baseline2_rouge)
            write_json("outputs/Evaluation/SGE_dsl_rouge.json", dsl_rouge)

        elif experiment_type == "DAE":
            DAE_result = {
                "dsl_data": [],
                "baseline_data": [],
                "baseline2_data": []
            }
            result = [[] for _ in range(10)]
            dsl_bleu = read_json("outputs/Evaluation/CPE_dsl_bleu.json")
            dsl_rouge = read_json("outputs/Evaluation/CPE_dsl_rouge.json")
            for i in range(10):
                result[i].append(dsl_bleu[i])
                result[i].append(dsl_rouge["Precision"][i])
                result[i].append(dsl_rouge["Recall"][i])
                result[i].append(dsl_rouge["F1"][i])
            DAE_result["dsl_data"] = result

            result = [[] for _ in range(10)]
            baseline_bleu = read_json("outputs/Evaluation/CPE_baseline_bleu.json")
            baseline_rouge = read_json("outputs/Evaluation/CPE_baseline_rouge.json")
            for i in range(10):
                result[i].append(baseline_bleu[i])
                result[i].append(baseline_rouge["Precision"][i])
                result[i].append(baseline_rouge["Recall"][i])
                result[i].append(baseline_rouge["F1"][i])
            DAE_result["baseline_data"] = result

            result = [[] for _ in range(10)]
            baseline2_bleu = read_json("outputs/Evaluation/CPE_baseline2_bleu.json")
            baseline2_rouge = read_json("outputs/Evaluation/CPE_baseline2_rouge.json")
            for i in range(10):
                result[i].append(baseline2_bleu[i])
                result[i].append(baseline2_rouge["Precision"][i])
                result[i].append(baseline2_rouge["Recall"][i])
                result[i].append(baseline2_rouge["F1"][i])
            DAE_result["baseline2_data"] = result


            DAE_metric_result = {
                "BLEU": {
                    "dsl_data": [],
                    "baseline_data": [],
                    "baseline2_data": []
                },
                "Precision": {
                    "dsl_data": [],
                    "baseline_data": [],
                    "baseline2_data": []
                },
                "Recall": {
                    "dsl_data": [],
                    "baseline_data": [],
                    "baseline2_data": []
                },
                "F1": {
                    "dsl_data": [],
                    "baseline_data": [],
                    "baseline2_data": []
                }
            }
            for i in range(10):
                DAE_metric_result["BLEU"]["dsl_data"].append(DAE_result["dsl_data"][i][0])
                DAE_metric_result["BLEU"]["baseline_data"].append(DAE_result["baseline_data"][i][0])
                DAE_metric_result["BLEU"]["baseline2_data"].append(DAE_result["baseline2_data"][i][0])

                DAE_metric_result["Precision"]["dsl_data"].append(DAE_result["dsl_data"][i][1])
                DAE_metric_result["Precision"]["baseline_data"].append(DAE_result["baseline_data"][i][1])
                DAE_metric_result["Precision"]["baseline2_data"].append(DAE_result["baseline2_data"][i][1])

                DAE_metric_result["Recall"]["dsl_data"].append(DAE_result["dsl_data"][i][2])
                DAE_metric_result["Recall"]["baseline_data"].append(DAE_result["baseline_data"][i][2])
                DAE_metric_result["Recall"]["baseline2_data"].append(DAE_result["baseline2_data"][i][2])

                DAE_metric_result["F1"]["dsl_data"].append(DAE_result["dsl_data"][i][3])
                DAE_metric_result["F1"]["baseline_data"].append(DAE_result["baseline_data"][i][3])
                DAE_metric_result["F1"]["baseline2_data"].append(DAE_result["baseline2_data"][i][3])

            write_json("outputs/Evaluation/DAE.json", DAE_result)
            write_json("outputs/Evaluation/DAE_metric.json", DAE_metric_result)

            # Function to calculate VMR
            def calculate_vmr(values):
                mean = np.mean(values)
                variance = np.var(values)
                return variance / mean if mean != 0 else np.nan

            # Calculating VMR for each group
            vmr_results = {
                metric: {
                    "dsl_vmr": calculate_vmr(values["dsl_data"]),
                    "baseline_vmr": calculate_vmr(values["baseline_data"]),
                    "baseline2_vmr": calculate_vmr(values["baseline2_data"]),
                }
                for metric, values in DAE_metric_result.items()
            }

            write_json("outputs/Evaluation/DAE_vmr.json", vmr_results)
            return

    def __bleu_score(self, reference, candidate):
        return sentence_bleu([reference], candidate, smoothing_function=SmoothingFunction().method4)

    def __rouge_score_old(self, reference, candidate):
        # return: precision, recall, fmeasure
        return self.scorer.score(reference, candidate)["rougeL"] 

    def __rouge_score_similarity(self, reference, candidate):
        # reference: groundtruth
        reference_json = json.loads(reference)
        candidate_json = json.loads(candidate)
        # Flatten both structures before comparison.
        reference_json = self.flatten_structure(reference_json)
        candidate_json = self.flatten_structure(candidate_json)
        X = len(candidate_json)
        Y = len(reference_json)
        C = 0

        if X == 0 or Y == 0:
            return 0, 0, 0
        for key_1, value_1 in reference_json.items():
            actual_key_1 = key_1.split(".")[-1]
            if "start" in actual_key_1 or "end" in actual_key_1 or "job_id" in actual_key_1 or "task_id" in actual_key_1:
                Y -= 1
        for key_2, value_2 in candidate_json.items():
            actual_key_2 = key_2.split(".")[-1]
            if "start" in actual_key_2 or "end" in actual_key_2 or "job_id" in actual_key_2 or "task_id" in actual_key_2:
                X -= 1
        for key_1, value_1 in candidate_json.items():
            actual_key_1 = key_1.split(".")[-1]
            if "start" in actual_key_1 or "end" in actual_key_1 or "job_id" in actual_key_1 or "task_id" in actual_key_1:
                continue
            for key_2, value_2 in reference_json.items():
                actual_key_2 = key_2.split(".")[-1]
                if "start" in actual_key_2 or "end" in actual_key_2 or "job_id" in actual_key_2 or "task_id" in actual_key_2:
                    continue
                word_1 = actual_key_1.lower() + " " + str(value_1)
                word_2 = actual_key_2.lower() + " " + str(value_2)
                vector_1 = self.__get_word_vector(word_1)
                vector_2 = self.__get_word_vector(word_2)
                similarity = self.__cosine_similarity(vector_1, vector_2)
                C += similarity

        print("X: ", X)
        print("Y: ", Y)
        return C/X, C/Y, 2*C/(X+Y)

    def __rouge_score(self, reference, candidate):
        # reference: groundtruth
        reference_json = json.loads(reference)
        candidate_json = json.loads(candidate)
        # Flatten both structures before comparison.
        reference_json = self.flatten_structure(reference_json)
        candidate_json = self.flatten_structure(candidate_json)
        X = len(candidate_json)
        Y = len(reference_json)
        C = 0

        if X == 0 or Y == 0:
            return 0, 0, 0
        for key_1, value_1 in candidate_json.items():
            actual_key_1 = key_1.split(".")[-1]
            if "start" in actual_key_1 or "end" in actual_key_1 or "job_id" in actual_key_1 or "task_id" in actual_key_1:
                continue
            for key_2, value_2 in reference_json.items():
                actual_key_2 = key_2.split(".")[-1]
                if "start" in actual_key_2 or "end" in actual_key_2 or "job_id" in actual_key_2 or "task_id" in actual_key_2:
                    continue
                if actual_key_1.lower() == actual_key_2.lower():
                    if value_1 == value_2:
                        C += 1
                        break
                    elif isinstance(value_1, str) and isinstance(value_2, str):
                        if value_1.lower() == value_2.lower():
                            C += 1
                            break
        print("X: ", X)
        print("Y: ", Y)
        return C/X, C/Y, 2*C/(X+Y)

    def flatten_structure(self, data, parent_key='', sep='.'):
        """
        Flatten nested dictionaries and lists.

        :param data: Structure to flatten, either a list or a dict
        :param parent_key: Parent key used to build recursive key paths
        :param sep: Separator between key segments
        :return: Flattened dictionary
        """
        items = []
        if isinstance(data, dict):
            for k, v in data.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, (dict, list)):  # Recurse on nested dicts and lists.
                    items.extend(self.flatten_structure(v, new_key, sep=sep).items())
                else:  # Append scalar values directly.
                    items.append((new_key, v))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_key = f"{parent_key}[{i}]"
                if isinstance(item, (dict, list)):  # Recurse on nested dicts and lists.
                    items.extend(self.flatten_structure(item, new_key, sep=sep).items())
                else:  # Append scalar values directly.
                    items.append((new_key, item))
        else:
            items.append((parent_key, data))  # Handle a single scalar value.
        return dict(items)

    def __get_word_vector(self, text):
        sentence_embedding = self.model.encode(text)
        return sentence_embedding

    def __cosine_similarity(self, vector1, vector2):
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    def __get_baseline_recourse_constraint_CSE_1(self, matrix, machine_list, route_sheet):
        # Resultant list for storing (operation, machine) pairs
        operations2machines = {}

        for job_index, job_steps in enumerate(matrix):
            for step_index, step in enumerate(job_steps):
                machine_id, duration, dependencies = step
                try:
                    # Retrieve operation details from the route sheet
                    operation_details = route_sheet[job_index]["route_sheet"][step_index]
                    operation_name = operation_details.get("operation", "None")

                    # Retrieve machine name from the machine list
                
                    machine_name = machine_list[machine_id]

                    # Map operation to machine
                    operations2machines[operation_name] = machine_name
                except:
                    continue
        operations2machines = [str(key) + " " + str(val) for key, val in operations2machines.items()]
        return operations2machines

    def __get_baseline_precedence_constraint_CSE_1(self, matrix, route_sheet, machine_list):
        '''
            return machine precedence constraints
        '''
        precedence_constraints = []

        for job_index, job_steps in enumerate(matrix):
            for step_index, step in enumerate(job_steps):
                machine_index, _, dependencies = step
                try:
                    # Retrieve the current operation
                    current_machine = machine_list[machine_index]

                    # Iterate through dependencies and map them to operations
                    for dependency_index in dependencies:
                        pred_machine = machine_list[matrix[job_index][dependency_index][0]]
                        precedence_constraints.append((pred_operation, current_machine))
                except:
                    continue

        precedence_constraints = [str(key) + " " + str(val) for key, val in precedence_constraints]
        return precedence_constraints

    def __get_DSL_recourse_constraint_CSE(self, operation_programs):
        operations2machines = {}
        for operation_program in operation_programs:
            for operation in operation_program:
                operation_name = operation.get("Operation", "None")
                machine_name = operation.get("Execution", {}).get("machine", "None")
                operations2machines[operation_name] = machine_name
        operations2machines = [str(key) + " " + str(val) for key, val in operations2machines.items()]
        return operations2machines

    def __get_DSL_precedence_constraint_CSE(self, production_programs):
        precedence_constraints = []
        for production_program in production_programs:
            for production in production_program:
                Pred = production.get("Pred", "None")
                Succ = production.get("Succ", "None")
                precedence_constraints.append((Pred, Succ))
        precedence_constraints = [str(key) + " " + str(val) for key, val in precedence_constraints]
        return precedence_constraints

    def __get_baseline_recourse_constraint_CSE_2(self, matrix, machine_list, structural_info):
        # Resultant list for storing (operation, machine) pairs
        operations2machines = {}

        for job_index, job_steps in enumerate(matrix):
            for step_index, step in enumerate(job_steps):
                machine_id, duration, dependencies = step
                try:
                    # Retrieve operation details from the route sheet
                    operation_details = structural_info[job_index]["steps"][step_index]
                    operation_name = operation_details.get("operation", "None")

                    # Retrieve machine name from the machine list
                    machine_name = machine_list[machine_id]

                    # Map operation to machine
                    operations2machines[operation_name] = machine_name
                except:
                    continue
        operations2machines = [str(key) + " " + str(val) for key, val in operations2machines.items()]
        return operations2machines

    def __get_baseline_precedence_constraint_CSE_2(self, matrix, machine_list, structural_info):
        '''
            return: machine precedence constraints
        '''
        precedence_constraints = []

        for job_index, job_steps in enumerate(matrix):
            for step_index, step in enumerate(job_steps):
                machine_index, _, dependencies = step
                try:
                    # Retrieve the current operation
                    current_machine = machine_list[machine_index]

                    # Iterate through dependencies and map them to operations
                    for dependency_index in dependencies:
                        pred_machine = machine_list[matrix[job_index][dependency_index][0]]
                        precedence_constraints.append((pred_machine, current_machine))
                except:
                    continue
        precedence_constraints = [str(key) + " " + str(val) for key, val in precedence_constraints]
        return precedence_constraints

    def __get_baseline2_precedence_constraint_CSE_2(self, matrix, machine_list, structural_info):
        '''
            return: machine precedence constraints
        '''
        precedence_constraints = []

        for job_index, job_steps in enumerate(matrix):
            for step_index, step in enumerate(job_steps):
                try:
                    machine_index, _, dependencies = step

                    # Retrieve the current operation
                    current_machine = machine_list[machine_index]

                    # Iterate through dependencies and map them to operations
                    for dependency_index in dependencies:
                        pred_machine = machine_list[matrix[job_index][dependency_index][0]]
                        precedence_constraints.append((pred_machine, current_machine))
                except:
                    continue
        precedence_constraints = [str(key) + " " + str(val) for key, val in precedence_constraints]
        return precedence_constraints

    def __get_groundtruth_recourse_constraint(self, route_sheet):
        # Resultant list for storing (operation, machine) pairs
        operations2machines = {}

        for job_index, job_steps in enumerate(route_sheet):
            for step_index, step in enumerate(job_steps.get("route_sheet", [])):
                try:
                    operation_name = step.get("operation", "None")
                    machine_name = step.get("machine", "None")

                    # Map operation to machine
                    operations2machines[operation_name] = machine_name
                except:
                    continue
        operations2machines = [str(key) + " " + str(val) for key, val in operations2machines.items()]
        return operations2machines

    def __get_groundtruth_operation_precedence_constraint(self, matrix, route_sheet):
        '''
            return operation precedence constraints
        '''
        precedence_constraints = []

        for job_index, job_steps in enumerate(matrix):
            for step_index, step in enumerate(job_steps):
                try:
                    _, _, dependencies = step

                    # Retrieve the current operation
                    current_operation = route_sheet[job_index]["route_sheet"][step_index]["operation"]

                    # Iterate through dependencies and map them to operations
                    for dependency_index in dependencies:
                        pred_operation = route_sheet[job_index]["route_sheet"][dependency_index]["operation"]
                        precedence_constraints.append((pred_operation, current_operation))
                except:
                    continue
        precedence_constraints = [str(key) + " " + str(val) for key, val in precedence_constraints]
        return precedence_constraints

    def __get_groundtruth_machine_precedence_constraint(self, matrix, route_sheet):
        '''
            return machine precedence constraints
        '''
        precedence_constraints = []

        for job_index, job_steps in enumerate(matrix):
            for step_index, step in enumerate(job_steps):
                try:
                    _, _, dependencies = step

                    # Retrieve the current operation
                    current_machine = route_sheet[job_index]["route_sheet"][step_index]["machine"]

                    # Iterate through dependencies and map them to operations
                    for dependency_index in dependencies:
                        pred_machine = route_sheet[job_index]["route_sheet"][dependency_index]["machine"]
                        precedence_constraints.append((pred_machine, current_machine))
                except:
                    continue
        precedence_constraints = [str(key) + " " + str(val) for key, val in precedence_constraints]
        return precedence_constraints

    def __iou(self, list1, list2):

        # Convert nested lists to tuples so they become hashable.
        set1 = set(tuple(item) if isinstance(item, list) else item for item in list1)
        set2 = set(tuple(item) if isinstance(item, list) else item for item in list2)

        # Compute intersection and union.
        intersection = set1 & set2
        union = set1 | set2

        # Avoid division by zero.
        if not union:
            return 0.0

        # Compute IoU.
        iou = len(intersection) / len(union)
        return iou

    def __get_unified_resource_constraint_CSE(self, route_sheets):
        """Extract {operation: machine} pairs from unified route sheets."""
        operations2machines = {}
        for rs_data in route_sheets:
            for step in rs_data.get("route_sheet", []):
                operation_name = step.get("operation", "None")
                machine_name = step.get("machine", "None")
                operations2machines[operation_name] = machine_name
        return [str(key) + " " + str(val) for key, val in operations2machines.items()]

    def __get_unified_precedence_constraint_CSE(self, or_matrix, route_sheets):
        """Extract (pred_operation, succ_operation) pairs from unified OR matrix + route sheets."""
        precedence_constraints = []
        for job_index, job_steps in enumerate(or_matrix):
            for step_index, step in enumerate(job_steps):
                try:
                    _, _, dependencies = step
                    current_operation = route_sheets[job_index]["route_sheet"][step_index]["operation"]
                    for dep_index in dependencies:
                        pred_operation = route_sheets[job_index]["route_sheet"][dep_index]["operation"]
                        precedence_constraints.append((pred_operation, current_operation))
                except:
                    continue
        return [str(key) + " " + str(val) for key, val in precedence_constraints]

    # ------------------------------------------------------------------ #
    #  Name alignment: Unified Pipeline → Ground Truth                    #
    # ------------------------------------------------------------------ #

    def _build_name_alignment(self, gt_route_sheets, unified_route_sheets):
        """Build unified→GT name mappings via step-level positional alignment."""
        # Tier 1: Positional alignment — collect (unified_name, gt_name) pairs
        op_pairs = []
        mach_pairs = []
        comp_pairs = []

        for i in range(min(len(gt_route_sheets), len(unified_route_sheets))):
            gt_steps = gt_route_sheets[i].get("route_sheet", [])
            u_steps = unified_route_sheets[i].get("route_sheet", [])
            for j in range(min(len(gt_steps), len(u_steps))):
                gs, us = gt_steps[j], u_steps[j]
                op_pairs.append((us.get("operation", ""), gs.get("operation", "")))
                mach_pairs.append((us.get("machine", ""), gs.get("machine", "")))
                for cond_key in ("precondition", "postcondition"):
                    gt_conds = gs.get(cond_key, [])
                    u_conds = us.get(cond_key, [])
                    for k in range(min(len(gt_conds), len(u_conds))):
                        comp_pairs.append((
                            u_conds[k].get("component_type", ""),
                            gt_conds[k].get("component_type", "")
                        ))

        op_map = self._majority_vote(op_pairs)
        mach_map = self._majority_vote(mach_pairs)
        comp_map = self._majority_vote(comp_pairs)

        # Parameter key alignment via token-subset matching
        param_pairs = []
        for i in range(min(len(gt_route_sheets), len(unified_route_sheets))):
            gt_steps = gt_route_sheets[i].get("route_sheet", [])
            u_steps = unified_route_sheets[i].get("route_sheet", [])
            for j in range(min(len(gt_steps), len(u_steps))):
                gt_params = gt_steps[j].get("parameters", {})
                u_params = u_steps[j].get("parameters", {})
                param_pairs.extend(self._match_param_keys(gt_params, u_params))
        param_map = self._majority_vote(param_pairs)

        # Collect all GT and unified name pools for fallback tiers
        gt_ops, gt_machines, gt_comps = set(), set(), set()
        u_ops, u_machines, u_comps = set(), set(), set()
        for rs_list, ops, machs, comps in [
            (gt_route_sheets, gt_ops, gt_machines, gt_comps),
            (unified_route_sheets, u_ops, u_machines, u_comps),
        ]:
            for job in rs_list:
                for step in job.get("route_sheet", []):
                    ops.add(step.get("operation", ""))
                    machs.add(step.get("machine", ""))
                    for cond in step.get("precondition", []) + step.get("postcondition", []):
                        comps.add(cond.get("component_type", ""))

        # Tier 2 + 3: fallback for unmapped names
        for u_pool, gt_pool, mapping in [
            (u_ops, gt_ops, op_map),
            (u_machines, gt_machines, mach_map),
            (u_comps, gt_comps, comp_map),
        ]:
            for u_name in u_pool - set(mapping.keys()):
                if not u_name:
                    continue
                match = self._case_insensitive_match(u_name, gt_pool)
                if not match:
                    match = self._fuzzy_match(u_name, gt_pool)
                if match:
                    mapping[u_name] = match

        return {"operations": op_map, "machines": mach_map, "component_types": comp_map, "param_keys": param_map}

    def _majority_vote(self, pairs):
        """Given (unified_name, gt_name) pairs, return {unified→gt} by majority vote."""
        counts = defaultdict(Counter)
        for u_name, gt_name in pairs:
            if u_name and gt_name:
                counts[u_name][gt_name] += 1
        return {u_name: counter.most_common(1)[0][0] for u_name, counter in counts.items()}

    def _case_insensitive_match(self, name, candidates):
        for c in candidates:
            if name.lower() == c.lower():
                return c
        return None

    def _fuzzy_match(self, name, candidates, threshold=0.6):
        best_match, best_ratio = None, 0
        for c in candidates:
            ratio = difflib.SequenceMatcher(None, name.lower(), c.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = c
        return best_match if best_ratio >= threshold else None

    def _match_param_keys(self, gt_params, u_params):
        """Match parameter keys within a step via token-subset matching."""
        pairs = []
        for u_key in u_params:
            u_tokens = set(u_key.lower().replace('_', ' ').split())
            best_gt, best_score = None, 0
            for gt_key in gt_params:
                gt_tokens = set(gt_key.lower().split())
                if gt_tokens <= u_tokens:
                    score = len(gt_tokens) / len(u_tokens)
                    if score > best_score:
                        best_score = score
                        best_gt = gt_key
            if best_gt:
                pairs.append((u_key, best_gt))
        return pairs

    def _align_route_sheets(self, unified_rs, name_maps):
        """Return a deep copy of unified route_sheets with names mapped to GT."""
        aligned = copy.deepcopy(unified_rs)
        op_map = name_maps["operations"]
        mach_map = name_maps["machines"]
        comp_map = name_maps["component_types"]
        param_map = name_maps.get("param_keys", {})
        for job in aligned:
            for step in job.get("route_sheet", []):
                op = step.get("operation", "")
                if op in op_map:
                    step["operation"] = op_map[op]
                m = step.get("machine", "")
                if m in mach_map:
                    step["machine"] = mach_map[m]
                for cond in step.get("precondition", []) + step.get("postcondition", []):
                    ct = cond.get("component_type", "")
                    if ct in comp_map:
                        cond["component_type"] = comp_map[ct]
                if step.get("parameters") and param_map:
                    step["parameters"] = {param_map.get(k, k): v for k, v in step["parameters"].items()}
        return aligned

    def _align_production_plan(self, production_plan, name_maps):
        """Return a deep copy of production_plan with names mapped to GT."""
        aligned = copy.deepcopy(production_plan)
        op_map = name_maps["operations"]
        mach_map = name_maps["machines"]
        comp_map = name_maps["component_types"]
        param_map = name_maps.get("param_keys", {})
        for entry in aligned:
            m = entry.get("machine", "")
            if m in mach_map:
                entry["machine"] = mach_map[m]
            for seq in entry.get("production_sequence", []):
                op = seq.get("operation", "")
                if op in op_map:
                    seq["operation"] = op_map[op]
                m2 = seq.get("machine", "")
                if m2 in mach_map:
                    seq["machine"] = mach_map[m2]
                seq["precondition"] = [comp_map.get(c, c) for c in seq.get("precondition", [])]
                seq["postcondition"] = [comp_map.get(c, c) for c in seq.get("postcondition", [])]
                if seq.get("parameters") and param_map:
                    seq["parameters"] = {param_map.get(k, k): v for k, v in seq["parameters"].items()}
        return aligned
