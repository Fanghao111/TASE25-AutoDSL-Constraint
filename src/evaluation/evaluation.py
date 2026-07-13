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
from src.experiment.fb_pipeline import FBPipeline

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


# Evaluation type names (see MIGRATION.md for old→new mapping):
#   production_plan   — s7 output vs GT production_plan (BLEU / ROUGE-L)
#   route_sheet       — s4 output → build_route_sheets vs GT route_sheets (BLEU / ROUGE-L)
#   graph_from_gt     — GT route_sheet → s5 output vs GT graph (accuracy IoU + err + makespan)
#   graph             — NL → s5 output vs GT graph (accuracy IoU + err + makespan)
#   ground_from_gt    — GT schedule → s7 output vs GT plan (BLEU / ROUGE-L)
#   dae               — DSL diagnostic aggregation (read prior production_plan results)


class Evaluation:
    def __init__(self):
        self.mode = ["Baseline-", "Baseline2-", "DSLPipeline-", "GroundTruth", "FB-2s-"]
        # NOTE: the "full" suffix maps to the FB experiment_type "full" (was "CPE_CAE_CSE-2");
        # DSL/Baseline still ship data under the legacy "CPE_CAE_CSE-2" folder name so we
        # keep that string in `suffix` for DSL/Baseline lookups.
        self.suffix = ["CPE_CAE_CSE-2/", "CSE-1/", "SGE/"]
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.result_path = "outputs/"
        self.groundtruth_dir_path = self.result_path + self.mode[3]

        # FB_EVAL_LABEL lets the wrapper steer FB pipeline path & evaluation output
        # dir per model (e.g. "FB-models/qwen3-4b"). Unset -> original behavior.
        fb_label = os.environ.get("FB_EVAL_LABEL")
        self.fb_dir_override = (self.result_path + fb_label + "/") if fb_label else None
        self.fb_eval_out_dir = ("outputs/Evaluation/" + fb_label.replace("/", "_") + "/") if fb_label else "outputs/Evaluation/"
        if fb_label:
            os.makedirs(self.fb_eval_out_dir, exist_ok=True)

        # DSL_EVAL_LABEL mirrors FB_EVAL_LABEL for the DSL pipeline (e.g. "DSL-models/qwen3-4b").
        dsl_label = os.environ.get("DSL_EVAL_LABEL")
        self.dsl_dir_override = (self.result_path + dsl_label + "/") if dsl_label else None
        self.dsl_eval_out_dir = ("outputs/Evaluation/" + dsl_label.replace("/", "_") + "/") if dsl_label else "outputs/Evaluation/"
        if dsl_label:
            os.makedirs(self.dsl_eval_out_dir, exist_ok=True)

        # Back-compat: existing FB writes use self.eval_out_dir.
        self.eval_out_dir = self.fb_eval_out_dir

    def evaluate(self, experiment_type: str):
        """
        experiment_type:
            production_plan  — NL description → production plan (BLEU / ROUGE-L)
            route_sheet      — NL description → fully-structural route sheet (BLEU / ROUGE-L)
            graph_from_gt    — GT route sheet → OR matrix (accuracy IoU + err_rate + makespan_ratio)
            graph            — NL description → OR matrix (accuracy IoU + err_rate + makespan_ratio)
            ground_from_gt   — GT JSP solver result → production plan (BLEU / ROUGE-L)
            dae              — aggregate variance-to-mean ratio over prior production_plan results
        """
        if experiment_type == "production_plan":
            baseline_dir_path = self.result_path + self.mode[0] + self.suffix[0]
            baseline2_dir_path = self.result_path + self.mode[1] + self.suffix[0]
            dsl_pipeline_dir_path = self.dsl_dir_override or (self.result_path + self.mode[2] + self.suffix[0])
            unified_dir_path = self.fb_dir_override or (self.result_path + self.mode[4] + self.suffix[0])

            baseline_bleu = []
            baseline2_bleu = []
            dsl_bleu = []
            fb_bleu = []

            baseline_rouge = {"Precision": [], "Recall": [], "F1": []}
            baseline2_rouge = {"Precision": [], "Recall": [], "F1": []}
            dsl_rouge = {"Precision": [], "Recall": [], "F1": []}
            fb_rouge = {"Precision": [], "Recall": [], "F1": []}

            # List subfolders from ground truth (always available).
            subfolders = os.listdir(self.groundtruth_dir_path)

            for subfolder in tqdm(subfolders):
                ground_truth_path = os.path.join(self.groundtruth_dir_path, subfolder, "production_plan.json")
                if not os.path.exists(ground_truth_path):
                    continue
                ground_truth_content = json.dumps(read_json(ground_truth_path))

                # Baseline (optional)
                baseline_path = os.path.join(baseline_dir_path, subfolder, "production_plan.json")
                if os.path.exists(baseline_path):
                    baseline_content = json.dumps(read_json(baseline_path))
                    baseline_bleu.append(self.__bleu_score(ground_truth_content, baseline_content))
                    p, r, f = self.__rouge_score(ground_truth_content, baseline_content)
                    baseline_rouge["Precision"].append(p)
                    baseline_rouge["Recall"].append(r)
                    baseline_rouge["F1"].append(f)

                # Baseline2 (optional)
                baseline2_path = os.path.join(baseline2_dir_path, subfolder, "production_plan.json")
                if os.path.exists(baseline2_path):
                    baseline2_content = json.dumps(read_json(baseline2_path))
                    baseline2_bleu.append(self.__bleu_score(ground_truth_content, baseline2_content))
                    p, r, f = self.__rouge_score(ground_truth_content, baseline2_content)
                    baseline2_rouge["Precision"].append(p)
                    baseline2_rouge["Recall"].append(r)
                    baseline2_rouge["F1"].append(f)

                # DSL Pipeline (optional)
                dsl_pipeline_path = os.path.join(dsl_pipeline_dir_path, subfolder, "production_plan.json")
                if os.path.exists(dsl_pipeline_path):
                    dsl_pipeline_content = json.dumps(read_json(dsl_pipeline_path))
                    dsl_bleu.append(self.__bleu_score(ground_truth_content, dsl_pipeline_content))
                    p, r, f = self.__rouge_score(ground_truth_content, dsl_pipeline_content)
                    dsl_rouge["Precision"].append(p)
                    dsl_rouge["Recall"].append(r)
                    dsl_rouge["F1"].append(f)

                # FB Pipeline (optional) — s7 output
                fb_path = os.path.join(unified_dir_path, subfolder, "s7_production_plan.json")
                if os.path.exists(fb_path):
                    fb_content = json.dumps(read_json(fb_path))
                    fb_bleu.append(self.__bleu_score(ground_truth_content, fb_content))
                    p, r, f = self.__rouge_score(ground_truth_content, fb_content)
                    fb_rouge["Precision"].append(p)
                    fb_rouge["Recall"].append(r)
                    fb_rouge["F1"].append(f)

            print("baseline_bleu: ", baseline_bleu)
            print("baseline2_bleu: ", baseline2_bleu)
            print("dsl_bleu: ", dsl_bleu)
            print("fb_bleu: ", fb_bleu)

            print("baseline_rouge: ", baseline_rouge)
            print("baseline2_rouge: ", baseline2_rouge)
            print("dsl_rouge: ", dsl_rouge)
            print("fb_rouge: ", fb_rouge)

            write_json("outputs/Evaluation/production_plan_baseline_bleu.json", baseline_bleu)
            write_json("outputs/Evaluation/production_plan_baseline2_bleu.json", baseline2_bleu)
            write_json(os.path.join(self.dsl_eval_out_dir, "production_plan_dsl_bleu.json"), dsl_bleu)
            write_json(os.path.join(self.eval_out_dir, "production_plan_fb_bleu.json"), fb_bleu)

            write_json("outputs/Evaluation/production_plan_baseline_rouge.json", baseline_rouge)
            write_json("outputs/Evaluation/production_plan_baseline2_rouge.json", baseline2_rouge)
            write_json(os.path.join(self.dsl_eval_out_dir, "production_plan_dsl_rouge.json"), dsl_rouge)
            write_json(os.path.join(self.eval_out_dir, "production_plan_fb_rouge.json"), fb_rouge)

        elif experiment_type == "route_sheet":
            baseline_dir_path = self.result_path + self.mode[0] + self.suffix[0]
            dsl_pipeline_dir_path = self.dsl_dir_override or (self.result_path + self.mode[2] + self.suffix[0])
            unified_dir_path = self.fb_dir_override or (self.result_path + self.mode[4] + self.suffix[0])

            baseline_bleu = []
            dsl_bleu = []
            fb_bleu = []

            baseline_rouge = {"Precision": [], "Recall": [], "F1": []}
            dsl_rouge = {"Precision": [], "Recall": [], "F1": []}
            fb_rouge = {"Precision": [], "Recall": [], "F1": []}

            # List subfolders from ground truth (always available).
            subfolders = os.listdir(self.groundtruth_dir_path)

            for subfolder in tqdm(subfolders):
                ground_truth_path = os.path.join(self.groundtruth_dir_path, subfolder, "route_sheets.json")
                if not os.path.exists(ground_truth_path):
                    continue
                ground_truth_content = json.dumps(read_json(ground_truth_path))

                # Baseline (optional)
                baseline_path = os.path.join(baseline_dir_path, subfolder, "structural_info.json")
                if os.path.exists(baseline_path):
                    baseline_content = json.dumps(read_json(baseline_path))
                    baseline_bleu.append(self.__bleu_score(ground_truth_content, baseline_content))
                    p, r, f = self.__rouge_score(ground_truth_content, baseline_content)
                    baseline_rouge["Precision"].append(p)
                    baseline_rouge["Recall"].append(r)
                    baseline_rouge["F1"].append(f)

                # DSL Pipeline (optional)
                dsl_pipeline_path = os.path.join(dsl_pipeline_dir_path, subfolder, "route_sheets.json")
                if os.path.exists(dsl_pipeline_path):
                    dsl_pipeline_content = json.dumps(read_json(dsl_pipeline_path))
                    dsl_bleu.append(self.__bleu_score(ground_truth_content, dsl_pipeline_content))
                    p, r, f = self.__rouge_score(ground_truth_content, dsl_pipeline_content)
                    dsl_rouge["Precision"].append(p)
                    dsl_rouge["Recall"].append(r)
                    dsl_rouge["F1"].append(f)

                # FB Pipeline (optional) — s4 normalized JSON, adapted to route_sheet format
                fb_nj_path = os.path.join(unified_dir_path, subfolder, "s4_normalized.json")
                if os.path.exists(fb_nj_path):
                    fb_nj = read_json(fb_nj_path)
                    fb_rs = FBPipeline.build_route_sheets(fb_nj, subfolder)
                    fb_content = json.dumps(fb_rs)
                    fb_bleu.append(self.__bleu_score(ground_truth_content, fb_content))
                    p, r, f = self.__rouge_score(ground_truth_content, fb_content)
                    fb_rouge["Precision"].append(p)
                    fb_rouge["Recall"].append(r)
                    fb_rouge["F1"].append(f)

            print("baseline_bleu: ", baseline_bleu)
            print("dsl_bleu: ", dsl_bleu)
            print("fb_bleu: ", fb_bleu)

            print("baseline_rouge: ", baseline_rouge)
            print("dsl_rouge: ", dsl_rouge)
            print("fb_rouge: ", fb_rouge)

            write_json("outputs/Evaluation/route_sheet_baseline_bleu.json", baseline_bleu)
            write_json(os.path.join(self.dsl_eval_out_dir, "route_sheet_dsl_bleu.json"), dsl_bleu)
            write_json(os.path.join(self.eval_out_dir, "route_sheet_fb_bleu.json"), fb_bleu)

            write_json("outputs/Evaluation/route_sheet_baseline_rouge.json", baseline_rouge)
            write_json(os.path.join(self.dsl_eval_out_dir, "route_sheet_dsl_rouge.json"), dsl_rouge)
            write_json(os.path.join(self.eval_out_dir, "route_sheet_fb_rouge.json"), fb_rouge)

        elif experiment_type == "graph_from_gt":
            baseline_dir_path = self.result_path + self.mode[0] + self.suffix[1]
            dsl_pipeline_dir_path = self.dsl_dir_override or (self.result_path + self.mode[2] + self.suffix[1])
            fb_dir_path = self.fb_dir_override or (self.result_path + self.mode[4] + self.suffix[1])

            baseline_result = {"accuracy_rate": [], "compile_err_rate": [], "runtime_err_rate": [], "makespan_ratio": []}
            dsl_result = {"accuracy_rate": [], "compile_err_rate": [], "runtime_err_rate": [], "makespan_ratio": []}
            fb_result = {"accuracy_rate": [], "compile_err_rate": [], "runtime_err_rate": [], "makespan_ratio": []}

            # List subfolders from ground truth (always available).
            subfolders = os.listdir(self.groundtruth_dir_path)

            for subfolder in tqdm(subfolders):
                ground_truth_path = os.path.join(self.groundtruth_dir_path, subfolder, "or_matrix.json")
                ground_truth_route_sheet = os.path.join(self.groundtruth_dir_path, subfolder, "route_sheets.json")
                if not all(os.path.exists(p) for p in [ground_truth_path, ground_truth_route_sheet]):
                    continue

                ground_truth_content = read_json(ground_truth_path)
                ground_truth_route_sheet_content = read_json(ground_truth_route_sheet)
                ground_truth_resource_constraints = self.__get_groundtruth_recourse_constraint(ground_truth_route_sheet_content)
                ground_truth_operation_precedence_constraints = self.__get_groundtruth_operation_precedence_constraint(ground_truth_content, ground_truth_route_sheet_content)
                ground_truth_machine_precedence_constraints = self.__get_groundtruth_machine_precedence_constraint(ground_truth_content, ground_truth_route_sheet_content)

                gt_makespan_path = os.path.join(self.groundtruth_dir_path, subfolder, "makespan.txt")
                gt_makespan = float(read_txt(gt_makespan_path)) if os.path.exists(gt_makespan_path) else 0

                # --- Baseline (optional) — reads legacy baseline output layout ---
                baseline_path = os.path.join(baseline_dir_path, subfolder, "or_matrix.json")
                baseline_route_sheet = os.path.join(baseline_dir_path, subfolder, "route_sheets.json")
                baseline_machines = os.path.join(baseline_dir_path, subfolder, "machines.json")
                if all(os.path.exists(p) for p in [baseline_path, baseline_route_sheet, baseline_machines]):
                    baseline_content = read_json(baseline_path)
                    baseline_route_sheet_content = read_json(baseline_route_sheet)
                    baseline_machines_content = read_json(baseline_machines)
                    baseline_resource_constraints = self.__get_baseline_recourse_constraint_CSE_1(baseline_content, baseline_machines_content, baseline_route_sheet_content)
                    baseline_precedence_constraints = self.__get_baseline_precedence_constraint_CSE_1(baseline_content, baseline_route_sheet_content, baseline_machines_content)
                    baseline_result["accuracy_rate"].append(
                        self.__iou(baseline_resource_constraints + baseline_precedence_constraints,
                        ground_truth_resource_constraints + ground_truth_machine_precedence_constraints)
                    )
                    baseline_result["runtime_err_rate"].append(
                        float(read_txt(os.path.join(baseline_dir_path, subfolder, "err_rate.txt")))
                    )
                    if gt_makespan > 0:
                        bl_ms_path = os.path.join(baseline_dir_path, subfolder, "makespan.txt")
                        if os.path.exists(bl_ms_path):
                            baseline_result["makespan_ratio"].append(float(read_txt(bl_ms_path)) / gt_makespan)

                # --- DSL Pipeline (optional) — reads legacy dsl output layout ---
                dsl_pipeline_path = os.path.join(dsl_pipeline_dir_path, subfolder, "or_matrix.json")
                dsl_op_path = os.path.join(dsl_pipeline_dir_path, subfolder, "operation_programs.json")
                dsl_prod_path = os.path.join(dsl_pipeline_dir_path, subfolder, "production_programs.json")
                if all(os.path.exists(p) for p in [dsl_pipeline_path, dsl_op_path, dsl_prod_path]):
                    dsl_pipeline_content = read_json(dsl_pipeline_path)
                    dsl_op_content = read_json(dsl_op_path)
                    dsl_prod_content = read_json(dsl_prod_path)
                    dsl_resource_constraints = self.__get_DSL_recourse_constraint_CSE(dsl_op_content)
                    dsl_precedence_constraints = self.__get_DSL_precedence_constraint_CSE(dsl_prod_content)
                    dsl_result["accuracy_rate"].append(self.__iou(
                        dsl_resource_constraints + dsl_precedence_constraints,
                        ground_truth_resource_constraints + ground_truth_operation_precedence_constraints
                    ))
                    dsl_err_path = os.path.join(dsl_pipeline_dir_path, subfolder, "err_rate.txt")
                    dsl_result["runtime_err_rate"].append(
                        float(read_txt(dsl_err_path)) if os.path.exists(dsl_err_path) else 0.0
                    )
                    if gt_makespan > 0:
                        dsl_ms_path = os.path.join(dsl_pipeline_dir_path, subfolder, "makespan.txt")
                        if os.path.exists(dsl_ms_path):
                            dsl_result["makespan_ratio"].append(float(read_txt(dsl_ms_path)) / gt_makespan)

                # --- FB Pipeline (optional) — s5/s6 output ---
                fb_or_path = os.path.join(fb_dir_path, subfolder, "s5_or_matrix.json")
                fb_machines_path = os.path.join(fb_dir_path, subfolder, "s5_machines.json")
                if all(os.path.exists(p) for p in [fb_or_path, fb_machines_path]):
                    fb_or_content = read_json(fb_or_path)
                    fb_machines_content = read_json(fb_machines_path)
                    fb_resource = self.__get_baseline_recourse_constraint_CSE_1(fb_or_content, fb_machines_content, ground_truth_route_sheet_content)
                    fb_precedence = self.__get_unified_precedence_constraint_CSE(fb_or_content, ground_truth_route_sheet_content)
                    fb_result["accuracy_rate"].append(self.__iou(
                        fb_resource + fb_precedence,
                        ground_truth_resource_constraints + ground_truth_operation_precedence_constraints
                    ))
                    fb_err_path = os.path.join(fb_dir_path, subfolder, "s6_err_rate.txt")
                    fb_result["runtime_err_rate"].append(
                        float(read_txt(fb_err_path)) if os.path.exists(fb_err_path) else 0.0
                    )
                    if gt_makespan > 0:
                        fb_ms_path = os.path.join(fb_dir_path, subfolder, "s6_makespan.txt")
                        if os.path.exists(fb_ms_path):
                            fb_result["makespan_ratio"].append(float(read_txt(fb_ms_path)) / gt_makespan)

            print("baseline_result: ", baseline_result)
            print("dsl_result: ", dsl_result)
            print("fb_result: ", fb_result)

            if baseline_result["accuracy_rate"]:
                write_json("outputs/Evaluation/graph_from_gt_baseline.json", baseline_result)
            write_json(os.path.join(self.dsl_eval_out_dir, "graph_from_gt_dsl.json"), dsl_result)
            write_json(os.path.join(self.eval_out_dir, "graph_from_gt_fb.json"), fb_result)

        elif experiment_type == "graph":
            baseline_dir_path = self.result_path + self.mode[0] + self.suffix[0]
            dsl_pipeline_dir_path = self.dsl_dir_override or (self.result_path + self.mode[2] + self.suffix[0])
            unified_dir_path = self.fb_dir_override or (self.result_path + self.mode[4] + self.suffix[0])

            baseline_result = {"accuracy_rate": [], "runtime_err_rate": [], "makespan_ratio": []}
            dsl_result = {"accuracy_rate": [], "runtime_err_rate": [], "makespan_ratio": []}
            fb_result = {"accuracy_rate": [], "runtime_err_rate": [], "makespan_ratio": []}

            # List subfolders from ground truth (always available).
            subfolders = os.listdir(self.groundtruth_dir_path)

            for subfolder in tqdm(subfolders):
                ground_truth_path = os.path.join(self.groundtruth_dir_path, subfolder, "or_matrix.json")
                ground_truth_route_sheet = os.path.join(self.groundtruth_dir_path, subfolder, "route_sheets.json")
                if not all(os.path.exists(p) for p in [ground_truth_path, ground_truth_route_sheet]):
                    continue

                ground_truth_content = read_json(ground_truth_path)
                ground_truth_route_sheet_content = read_json(ground_truth_route_sheet)
                ground_truth_resource_constraints = self.__get_groundtruth_recourse_constraint(ground_truth_route_sheet_content)
                ground_truth_operation_precedence_constraints = self.__get_groundtruth_operation_precedence_constraint(ground_truth_content, ground_truth_route_sheet_content)
                ground_truth_machine_precedence_constraints = self.__get_groundtruth_machine_precedence_constraint(ground_truth_content, ground_truth_route_sheet_content)

                gt_makespan_path = os.path.join(self.groundtruth_dir_path, subfolder, "makespan.txt")
                gt_makespan = float(read_txt(gt_makespan_path)) if os.path.exists(gt_makespan_path) else 0

                # --- Baseline (optional) — legacy layout ---
                baseline_path = os.path.join(baseline_dir_path, subfolder, "or_matrix.json")
                baseline_route_sheet = os.path.join(baseline_dir_path, subfolder, "structural_info.json")
                baseline_machines = os.path.join(baseline_dir_path, subfolder, "machines.json")
                if all(os.path.exists(p) for p in [baseline_path, baseline_route_sheet, baseline_machines]):
                    baseline_content = read_json(baseline_path)
                    baseline_route_sheet_content = read_json(baseline_route_sheet)
                    baseline_machines_content = read_json(baseline_machines)
                    baseline_resource_constraints = self.__get_baseline_recourse_constraint_CSE_2(baseline_content, baseline_machines_content, baseline_route_sheet_content)
                    baseline_precedence_constraints = self.__get_baseline_precedence_constraint_CSE_1(baseline_content, baseline_route_sheet_content, baseline_machines_content)
                    baseline_result["accuracy_rate"].append(
                        self.__iou(baseline_resource_constraints + baseline_precedence_constraints,
                        ground_truth_resource_constraints + ground_truth_machine_precedence_constraints)
                    )
                    baseline_result["runtime_err_rate"].append(
                        float(read_txt(os.path.join(baseline_dir_path, subfolder, "err_rate.txt")))
                    )
                    if gt_makespan > 0:
                        ms_path = os.path.join(baseline_dir_path, subfolder, "makespan.txt")
                        if os.path.exists(ms_path):
                            baseline_result["makespan_ratio"].append(float(read_txt(ms_path)) / gt_makespan)

                # --- DSL Pipeline (optional) — legacy layout ---
                dsl_pipeline_path = os.path.join(dsl_pipeline_dir_path, subfolder, "or_matrix.json")
                dsl_op_path = os.path.join(dsl_pipeline_dir_path, subfolder, "operation_programs.json")
                dsl_prod_path = os.path.join(dsl_pipeline_dir_path, subfolder, "production_programs.json")
                if all(os.path.exists(p) for p in [dsl_pipeline_path, dsl_op_path, dsl_prod_path]):
                    dsl_op_content = read_json(dsl_op_path)
                    dsl_prod_content = read_json(dsl_prod_path)
                    dsl_resource_constraints = self.__get_DSL_recourse_constraint_CSE(dsl_op_content)
                    dsl_precedence_constraints = self.__get_DSL_precedence_constraint_CSE(dsl_prod_content)
                    dsl_result["accuracy_rate"].append(self.__iou(
                        dsl_resource_constraints + dsl_precedence_constraints,
                        ground_truth_resource_constraints + ground_truth_operation_precedence_constraints
                    ))
                    dsl_err_path = os.path.join(dsl_pipeline_dir_path, subfolder, "err_rate.txt")
                    dsl_result["runtime_err_rate"].append(
                        float(read_txt(dsl_err_path)) if os.path.exists(dsl_err_path) else 0.0
                    )
                    if gt_makespan > 0:
                        dsl_ms_path = os.path.join(dsl_pipeline_dir_path, subfolder, "makespan.txt")
                        if os.path.exists(dsl_ms_path):
                            dsl_result["makespan_ratio"].append(float(read_txt(dsl_ms_path)) / gt_makespan)

                # --- FB Pipeline (optional) — s5/s6 output ---
                fb_or_path = os.path.join(unified_dir_path, subfolder, "s5_or_matrix.json")
                fb_machines_path = os.path.join(unified_dir_path, subfolder, "s5_machines.json")
                if all(os.path.exists(p) for p in [fb_or_path, fb_machines_path]):
                    fb_or_content = read_json(fb_or_path)
                    fb_machines_content = read_json(fb_machines_path)
                    fb_resource = self.__get_baseline_recourse_constraint_CSE_1(fb_or_content, fb_machines_content, ground_truth_route_sheet_content)
                    fb_precedence = self.__get_unified_precedence_constraint_CSE(fb_or_content, ground_truth_route_sheet_content)
                    fb_result["accuracy_rate"].append(self.__iou(
                        fb_resource + fb_precedence,
                        ground_truth_resource_constraints + ground_truth_operation_precedence_constraints
                    ))
                    fb_err_path = os.path.join(unified_dir_path, subfolder, "s6_err_rate.txt")
                    fb_result["runtime_err_rate"].append(
                        float(read_txt(fb_err_path)) if os.path.exists(fb_err_path) else 0.0
                    )
                    if gt_makespan > 0:
                        fb_ms_path = os.path.join(unified_dir_path, subfolder, "s6_makespan.txt")
                        if os.path.exists(fb_ms_path):
                            fb_result["makespan_ratio"].append(float(read_txt(fb_ms_path)) / gt_makespan)

            print("baseline_result: ", baseline_result)
            print("dsl_result: ", dsl_result)
            print("fb_result: ", fb_result)

            write_json("outputs/Evaluation/graph_baseline.json", baseline_result)
            write_json(os.path.join(self.dsl_eval_out_dir, "graph_dsl.json"), dsl_result)
            write_json(os.path.join(self.eval_out_dir, "graph_fb.json"), fb_result)

        elif experiment_type == "ground_from_gt":
            dsl_pipeline_dir_path = self.dsl_dir_override or (self.result_path + self.mode[2] + self.suffix[2])
            fb_dir_path = self.fb_dir_override or (self.result_path + self.mode[4] + self.suffix[2])

            dsl_bleu = []
            fb_bleu = []
            dsl_rouge = {"Precision": [], "Recall": [], "F1": []}
            fb_rouge = {"Precision": [], "Recall": [], "F1": []}

            # List subfolders from ground truth.
            subfolders = os.listdir(self.groundtruth_dir_path)

            for subfolder in tqdm(subfolders):
                ground_truth_path = os.path.join(self.groundtruth_dir_path, subfolder, "production_plan.json")
                if not os.path.exists(ground_truth_path):
                    continue
                ground_truth_content = json.dumps(read_json(ground_truth_path))

                # DSL — legacy layout
                dsl_path = os.path.join(dsl_pipeline_dir_path, subfolder, "production_plan.json")
                if os.path.exists(dsl_path):
                    dsl_content = json.dumps(read_json(dsl_path))
                    dsl_bleu.append(self.__bleu_score(ground_truth_content, dsl_content))
                    p, r, f = self.__rouge_score(ground_truth_content, dsl_content)
                    dsl_rouge["Precision"].append(p)
                    dsl_rouge["Recall"].append(r)
                    dsl_rouge["F1"].append(f)

                # FB — s7 output
                fb_path = os.path.join(fb_dir_path, subfolder, "s7_production_plan.json")
                if os.path.exists(fb_path):
                    fb_content = json.dumps(read_json(fb_path))
                    fb_bleu.append(self.__bleu_score(ground_truth_content, fb_content))
                    p, r, f = self.__rouge_score(ground_truth_content, fb_content)
                    fb_rouge["Precision"].append(p)
                    fb_rouge["Recall"].append(r)
                    fb_rouge["F1"].append(f)

            print("dsl_bleu: ", dsl_bleu)
            print("dsl_rouge: ", dsl_rouge)
            print("fb_bleu: ", fb_bleu)
            print("fb_rouge: ", fb_rouge)

            write_json(os.path.join(self.dsl_eval_out_dir, "ground_from_gt_dsl_bleu.json"), dsl_bleu)
            write_json(os.path.join(self.dsl_eval_out_dir, "ground_from_gt_dsl_rouge.json"), dsl_rouge)
            write_json(os.path.join(self.eval_out_dir, "ground_from_gt_fb_bleu.json"), fb_bleu)
            write_json(os.path.join(self.eval_out_dir, "ground_from_gt_fb_rouge.json"), fb_rouge)

        elif experiment_type == "dae":
            DAE_result = {
                "dsl_data": [],
                "baseline_data": [],
                "baseline2_data": []
            }
            result = [[] for _ in range(10)]
            dsl_bleu = read_json(os.path.join(self.dsl_eval_out_dir, "production_plan_dsl_bleu.json"))
            dsl_rouge = read_json(os.path.join(self.dsl_eval_out_dir, "production_plan_dsl_rouge.json"))
            for i in range(10):
                result[i].append(dsl_bleu[i])
                result[i].append(dsl_rouge["Precision"][i])
                result[i].append(dsl_rouge["Recall"][i])
                result[i].append(dsl_rouge["F1"][i])
            DAE_result["dsl_data"] = result

            result = [[] for _ in range(10)]
            baseline_bleu = read_json("outputs/Evaluation/production_plan_baseline_bleu.json")
            baseline_rouge = read_json("outputs/Evaluation/production_plan_baseline_rouge.json")
            for i in range(10):
                result[i].append(baseline_bleu[i])
                result[i].append(baseline_rouge["Precision"][i])
                result[i].append(baseline_rouge["Recall"][i])
                result[i].append(baseline_rouge["F1"][i])
            DAE_result["baseline_data"] = result

            result = [[] for _ in range(10)]
            baseline2_bleu = read_json("outputs/Evaluation/production_plan_baseline2_bleu.json")
            baseline2_rouge = read_json("outputs/Evaluation/production_plan_baseline2_rouge.json")
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

            write_json("outputs/Evaluation/dae.json", DAE_result)
            write_json("outputs/Evaluation/dae_metric.json", DAE_metric_result)

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

            write_json("outputs/Evaluation/dae_vmr.json", vmr_results)
            return

    def __bleu_score(self, reference, candidate):
        # NLTK sentence_bleu expects token lists, not raw strings. Passing strings
        # makes it tokenize per character, which inflates scores via the shared
        # JSON template chars ({, ", :, comma, etc.) and squashes model differences.
        # Split on whitespace so each JSON token (e.g. `"op":`, `60},`) is one unit.
        return sentence_bleu([reference.split()], candidate.split(),
                             smoothing_function=SmoothingFunction().method4)

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

        def normalize(item):
            if isinstance(item, list):
                return tuple(str(x).lower() for x in item)
            return str(item).lower()

        set1 = set(normalize(item) for item in list1)
        set2 = set(normalize(item) for item in list2)

        # Compute intersection and union.
        intersection = set1 & set2
        union = set1 | set2

        # Avoid division by zero.
        if not union:
            return 0.0

        # Compute IoU.
        iou = len(intersection) / len(union)
        return iou

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
