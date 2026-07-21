"""sschema evaluate — score pipeline outputs against ground truth.

Nine evaluation types, all comparing sschema pipeline artifacts vs ground truth
built by common/groundtruth.py:

  Structural / NL similarity:
    production_plan       s7 output vs GT production_plan (BLEU / ROUGE)
    route_sheet           s4 output → build_route_sheets vs GT (BLEU / ROUGE)
    graph                 s5 or_matrix vs GT (IoU / err_rate / makespan_ratio)
    graph_from_gt         s5 or_matrix (from_gt_route_sheet path) vs GT
    ground_from_gt        s7 (from_gt_schedule path) vs GT (BLEU / ROUGE)

  Per-field P/R/F1 (isolate individual pipeline stages):
    extract               s1_extracted vs GT_s1
    verify                s2_verified vs GT_s1 + fix/break vs s1
    normalize             s4 vocabulary shrinkage per category (no GT compare)
    route_sheet_fieldlevel  s4_normalized vs GT_s1

  all                     run everything above in sequence

Ground truth is auto-built (via common/groundtruth.py) if the target instance
doesn't yet have artifacts/groundtruth/<inst>/ populated.

Instance discovery: pass --instances explicitly, or omit to auto-discover from
artifacts/pipeline/<experiment_type>/.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List

from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

from common.groundtruth import GroundTruth
from common.io import read_json, read_txt, write_json

# Import SchemaPipeline lazily inside build_route_sheets consumer to avoid
# forcing openai/ortools on people who only run evaluate on cached artifacts.


ALL_INSTANCES = [f"ta{i}" for i in range(71, 81)]

STRUCTURAL_TYPES = ("production_plan", "route_sheet", "graph",
                    "graph_from_gt", "ground_from_gt")
FIELDLEVEL_TYPES = ("extract", "verify", "normalize", "route_sheet_fieldlevel")
ALL_TYPES = STRUCTURAL_TYPES + FIELDLEVEL_TYPES


# --------------------------------------------------------------------- #
#  Small helpers                                                        #
# --------------------------------------------------------------------- #


def _to_full(inst: str) -> str:
    return inst if inst.startswith("instance ") else f"instance {inst}"


def _to_short(inst: str) -> str:
    return inst.replace("instance ", "")


def _norm(value: Any) -> str:
    return str(value).strip().lower()


def _parse_duration_minutes(text: Any) -> int:
    text = str(text).lower().strip()
    m = re.search(r"([\d.]+)", text)
    if not m:
        return 0
    value = float(m.group(1))
    if "hour" in text or "hr" in text:
        return int(value * 60)
    if "second" in text or "sec" in text:
        return max(1, int(value / 60))
    return int(value)


def _fixed_schema_from_route_sheet(rs_job: Dict[str, Any]) -> Dict[str, Any]:
    return {"steps": rs_job.get("route_sheet", [])}


def _step_component_types(step: Dict[str, Any], field: str) -> List[str]:
    out = []
    for item in step.get(field, []) or []:
        if isinstance(item, dict):
            ct = item.get("component_type", "")
            if ct:
                out.append(_norm(ct))
    return out


def _step_parameters(step: Dict[str, Any]) -> Dict[str, str]:
    params = step.get("parameters", {}) or {}
    if not isinstance(params, dict):
        return {}
    return {_norm(k): (_norm(v) if isinstance(v, (str, int, float)) else _norm(json.dumps(v)))
            for k, v in params.items()}


def _build_route_sheets(normalized_jsons, instance_description=""):
    """Same shape as SchemaPipeline.build_route_sheets — duplicated here so evaluate
    can run without importing openai/ortools transitively via pipeline.py."""
    out = []
    for data in normalized_jsons:
        if isinstance(data, dict) and "bad_case" in data:
            out.append({"instance_description": instance_description, "route_sheet": []})
            continue
        out.append({
            "instance_description": instance_description,
            "route_sheet": data.get("steps", []) if isinstance(data, dict) else [],
        })
    return out


# --------------------------------------------------------------------- #
#  Ground truth resolution                                              #
# --------------------------------------------------------------------- #


def _ensure_gt(instance_short: str, artifacts_dir: str,
               preprocess_dir: str, data_dir: str,
               need_schedule: bool) -> str:
    """Build GT for one instance if missing. Returns the GT dump directory."""
    gt_dir = os.path.join(artifacts_dir, "groundtruth", instance_short)
    needed = ["route_sheets.json", "or_matrix.json"]
    if need_schedule:
        needed += ["assigned_jobs.json", "production_plan.json", "makespan.txt"]
    if all(os.path.exists(os.path.join(gt_dir, f)) for f in needed):
        return gt_dir

    gt = GroundTruth(
        instance_description=_to_full(instance_short),
        route_sheet_reduce_path=os.path.join(preprocess_dir, "route_sheet_reduce.json"),
        jssp_mapped_path=os.path.join(preprocess_dir, "jssp_mapped.json"),
        machines_data_path=os.path.join(data_dir, "machines.json"),
        dump_dir_path=gt_dir,
    )
    gt.get_grounded_route_sheet()
    gt.get_grounded_or_matrix()
    if need_schedule:
        gt.get_grounded_assigned_jobs()
        gt.get_grounded_production_plan()
    return gt_dir


# --------------------------------------------------------------------- #
#  Metric primitives                                                    #
# --------------------------------------------------------------------- #


_ROUGE_SCORER = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


def _bleu(reference: str, candidate: str) -> float:
    return sentence_bleu([reference.split()], candidate.split(),
                         smoothing_function=SmoothingFunction().method4)


def _flatten_structure(data, parent_key='', sep='.'):
    items = []
    if isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, (dict, list)):
                items.extend(_flatten_structure(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            new_key = f"{parent_key}[{i}]"
            if isinstance(item, (dict, list)):
                items.extend(_flatten_structure(item, new_key, sep=sep).items())
            else:
                items.append((new_key, item))
    else:
        items.append((parent_key, data))
    return dict(items)


_EXCLUDE_KEYS = ("start", "end", "job_id", "task_id")


def _rouge_like(reference: str, candidate: str):
    """Structural precision/recall/F1 (same definition as fb-2s evaluation.py:__rouge_score).

    Compares flattened key/value pairs; excludes schedule-position fields
    (start/end/job_id/task_id) so structural content dominates the score.
    """
    ref_json = json.loads(reference)
    cand_json = json.loads(candidate)
    ref_flat = _flatten_structure(ref_json)
    cand_flat = _flatten_structure(cand_json)

    X = sum(1 for k in cand_flat if not any(e in k.split(".")[-1] for e in _EXCLUDE_KEYS))
    Y = sum(1 for k in ref_flat if not any(e in k.split(".")[-1] for e in _EXCLUDE_KEYS))
    if X == 0 or Y == 0:
        return 0.0, 0.0, 0.0

    C = 0
    for k1, v1 in cand_flat.items():
        actual_k1 = k1.split(".")[-1]
        if any(e in actual_k1 for e in _EXCLUDE_KEYS):
            continue
        for k2, v2 in ref_flat.items():
            actual_k2 = k2.split(".")[-1]
            if any(e in actual_k2 for e in _EXCLUDE_KEYS):
                continue
            if actual_k1.lower() == actual_k2.lower():
                if v1 == v2:
                    C += 1
                    break
                if isinstance(v1, str) and isinstance(v2, str) and v1.lower() == v2.lower():
                    C += 1
                    break
    return C / X, C / Y, 2 * C / (X + Y)


def _iou(list1, list2):
    def normalize(item):
        if isinstance(item, list):
            return tuple(str(x).lower() for x in item)
        return str(item).lower()
    set1 = {normalize(x) for x in list1}
    set2 = {normalize(x) for x in list2}
    inter = set1 & set2
    union = set1 | set2
    return len(inter) / len(union) if union else 0.0


def _gt_resource_constraint(route_sheet):
    ops2machines = {}
    for job in route_sheet:
        for step in job.get("route_sheet", []) or []:
            try:
                ops2machines[step.get("operation", "None")] = step.get("machine", "None")
            except Exception:
                continue
    return [f"{k} {v}" for k, v in ops2machines.items()]


def _gt_operation_precedence(matrix, route_sheet):
    out = []
    for job_index, job_steps in enumerate(matrix):
        for step_index, step in enumerate(job_steps):
            try:
                _, _, deps = step
                cur = route_sheet[job_index]["route_sheet"][step_index]["operation"]
                for d in deps:
                    pred = route_sheet[job_index]["route_sheet"][d]["operation"]
                    out.append((pred, cur))
            except Exception:
                continue
    return [f"{a} {b}" for a, b in out]


def _pipeline_resource_constraint(matrix, machine_list, route_sheet):
    """Mirror of fb-2s Evaluation.__get_baseline_recourse_constraint_CSE_1 but
    reads from sschema pipeline s5 output + GT route_sheet."""
    ops2machines = {}
    for job_index, job_steps in enumerate(matrix):
        for step_index, step in enumerate(job_steps):
            try:
                machine_id, _, _ = step
                op = route_sheet[job_index]["route_sheet"][step_index].get("operation", "None")
                ops2machines[op] = machine_list[machine_id]
            except Exception:
                continue
    return [f"{k} {v}" for k, v in ops2machines.items()]


def _pipeline_precedence_constraint(or_matrix, route_sheets):
    out = []
    for job_index, job_steps in enumerate(or_matrix):
        for step_index, step in enumerate(job_steps):
            try:
                _, _, deps = step
                cur = route_sheets[job_index]["route_sheet"][step_index]["operation"]
                for d in deps:
                    pred = route_sheets[job_index]["route_sheet"][d]["operation"]
                    out.append((pred, cur))
            except Exception:
                continue
    return [f"{a} {b}" for a, b in out]


# --------------------------------------------------------------------- #
#  Field-level P/R/F1 primitives                                        #
# --------------------------------------------------------------------- #

_FIELDS = ("machine", "operation", "duration",
           "component_type", "param_key", "param_value")


def _prf1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _scalar_cmp(pred, gt):
    if not pred and not gt:
        return {"tp": 0, "fp": 0, "fn": 0}
    if pred == gt:
        return {"tp": 1, "fp": 0, "fn": 0}
    if pred and not gt:
        return {"tp": 0, "fp": 1, "fn": 0}
    if gt and not pred:
        return {"tp": 0, "fp": 0, "fn": 1}
    return {"tp": 0, "fp": 1, "fn": 1}


def _multiset_cmp(pred, gt):
    from collections import Counter
    pc, gc = Counter(pred), Counter(gt)
    tp = sum((pc & gc).values())
    return {"tp": tp, "fp": sum(pc.values()) - tp, "fn": sum(gc.values()) - tp}


def _set_cmp(pred: set, gt: set):
    tp = len(pred & gt)
    return {"tp": tp, "fp": len(pred - gt), "fn": len(gt - pred)}


def _empty_per_field():
    return {f: {"tp": 0, "fp": 0, "fn": 0} for f in _FIELDS}


def _empty_fix_break():
    return {f: {"fix": 0, "break": 0, "still_wrong": 0, "still_right": 0}
            for f in _FIELDS}


def _accumulate(dst, src):
    for f, d in src.items():
        for k, v in d.items():
            dst[f][k] += v


def _prf1_by_field(per_field):
    return {f: _prf1(**per_field[f]) for f in per_field}


def _macro_f1(summary):
    return sum(s["f1"] for s in summary.values()) / len(summary) if summary else 0.0


def _score_step_pure(p_step, g_step):
    out = {}
    out["machine"] = _scalar_cmp(_norm((p_step or {}).get("machine", "")),
                                 _norm((g_step or {}).get("machine", "")))
    out["operation"] = _scalar_cmp(_norm((p_step or {}).get("operation", "")),
                                   _norm((g_step or {}).get("operation", "")))
    out["duration"] = _scalar_cmp(
        str(_parse_duration_minutes((p_step or {}).get("duration", "0"))),
        str(_parse_duration_minutes((g_step or {}).get("duration", "0"))),
    )
    p_cts = ((_step_component_types(p_step, "precondition")
              + _step_component_types(p_step, "postcondition")) if p_step else [])
    g_cts = ((_step_component_types(g_step, "precondition")
              + _step_component_types(g_step, "postcondition")) if g_step else [])
    out["component_type"] = _multiset_cmp(p_cts, g_cts)
    p_params = _step_parameters(p_step) if p_step else {}
    g_params = _step_parameters(g_step) if g_step else {}
    out["param_key"] = _set_cmp(set(p_params), set(g_params))
    common = set(p_params) & set(g_params)
    pv_tp = sum(1 for k in common if p_params[k] == g_params[k])
    pv_fp = sum(1 for k in common if p_params[k] != g_params[k])
    out["param_value"] = {"tp": pv_tp, "fp": pv_fp, "fn": pv_fp}
    return out


def _score_fixed_schema(pred_jobs, gt_jobs):
    per_field = _empty_per_field()
    n = max(len(pred_jobs), len(gt_jobs))
    for j in range(n):
        p_job = pred_jobs[j] if j < len(pred_jobs) else None
        g_job = gt_jobs[j] if j < len(gt_jobs) else None
        p_steps = p_job.get("steps", []) if isinstance(p_job, dict) else []
        g_steps = g_job.get("steps", []) if isinstance(g_job, dict) else []
        for s in range(max(len(p_steps), len(g_steps))):
            p_step = p_steps[s] if s < len(p_steps) else None
            g_step = g_steps[s] if s < len(g_steps) else None
            for f, delta in _score_step_pure(p_step, g_step).items():
                for k, v in delta.items():
                    per_field[f][k] += v
    return per_field


def _fix_break_per_field(s1_jobs, s2_jobs, gt_jobs):
    counts = _empty_fix_break()
    n = max(len(s1_jobs), len(s2_jobs), len(gt_jobs))
    for j in range(n):
        s1_job = s1_jobs[j] if j < len(s1_jobs) else None
        s2_job = s2_jobs[j] if j < len(s2_jobs) else None
        g_job = gt_jobs[j] if j < len(gt_jobs) else None
        s1_steps = s1_job.get("steps", []) if isinstance(s1_job, dict) else []
        s2_steps = s2_job.get("steps", []) if isinstance(s2_job, dict) else []
        g_steps = g_job.get("steps", []) if isinstance(g_job, dict) else []
        for s in range(max(len(s1_steps), len(s2_steps), len(g_steps))):
            s1_step = s1_steps[s] if s < len(s1_steps) else None
            s2_step = s2_steps[s] if s < len(s2_steps) else None
            g_step = g_steps[s] if s < len(g_steps) else None
            s1_scores = _score_step_pure(s1_step, g_step)
            s2_scores = _score_step_pure(s2_step, g_step)
            for f in _FIELDS:
                s1_ok = s1_scores[f]["fp"] == 0 and s1_scores[f]["fn"] == 0
                s2_ok = s2_scores[f]["fp"] == 0 and s2_scores[f]["fn"] == 0
                if not s1_ok and s2_ok:
                    counts[f]["fix"] += 1
                elif s1_ok and not s2_ok:
                    counts[f]["break"] += 1
                elif not s1_ok and not s2_ok:
                    counts[f]["still_wrong"] += 1
                else:
                    counts[f]["still_right"] += 1
    return counts


def _print_prf1(name, out_path, macro, summary):
    print(f"{name} → {out_path}")
    print(f"  macro F1 = {macro:.4f}")
    for f, s in summary.items():
        print(f"    {f:15s} P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f}")


# --------------------------------------------------------------------- #
#  Main evaluator                                                       #
# --------------------------------------------------------------------- #


class SchemaEvaluator:
    def __init__(self, artifacts_dir: str, preprocess_dir: str, data_dir: str):
        self.artifacts_dir = artifacts_dir
        self.preprocess_dir = preprocess_dir
        self.data_dir = data_dir

        self.groundtruth_dir = os.path.join(artifacts_dir, "groundtruth")
        self.pipeline_dir = os.path.join(artifacts_dir, "pipeline")
        self.out_dir = os.path.join(artifacts_dir, "evaluate")
        os.makedirs(self.out_dir, exist_ok=True)

    # ---- path helpers --------------------------------------------------

    def _pipe_dir(self, experiment_type: str, inst_short: str) -> str:
        return os.path.join(self.pipeline_dir, experiment_type, inst_short)

    def _pipe_json(self, experiment_type: str, inst_short: str, filename: str):
        p = os.path.join(self._pipe_dir(experiment_type, inst_short), filename)
        return read_json(p) if os.path.exists(p) else None

    def _gt_dir(self, inst_short: str, need_schedule: bool) -> str:
        return _ensure_gt(inst_short, self.artifacts_dir, self.preprocess_dir,
                          self.data_dir, need_schedule)

    def _load_gt_json(self, inst_short: str, filename: str, need_schedule: bool):
        gt_dir = self._gt_dir(inst_short, need_schedule)
        p = os.path.join(gt_dir, filename)
        return read_json(p) if os.path.exists(p) else None

    def _load_gt_makespan(self, inst_short: str) -> float:
        gt_dir = self._gt_dir(inst_short, need_schedule=True)
        p = os.path.join(gt_dir, "makespan.txt")
        return float(read_txt(p)) if os.path.exists(p) else 0.0

    # ---- discovery -----------------------------------------------------

    def discover_instances(self, experiment_type: str) -> List[str]:
        d = os.path.join(self.pipeline_dir, experiment_type)
        if not os.path.isdir(d):
            return []
        return sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)))

    # ---- dispatch ------------------------------------------------------

    def run(self, evaluation_type: str, instances: List[str]):
        if evaluation_type == "all":
            for t in ALL_TYPES:
                print(f"\n=== evaluation: {t} ===", flush=True)
                self._run_one(t, instances)
            return
        if evaluation_type not in ALL_TYPES:
            raise ValueError(
                f"Unknown --type '{evaluation_type}'. Choose from {ALL_TYPES + ('all',)}."
            )
        self._run_one(evaluation_type, instances)

    def _run_one(self, t: str, instances: List[str]):
        if t == "production_plan":
            self._run_bleu_rouge(instances, "full", "s7_production_plan.json",
                                 "production_plan.json", need_schedule=True,
                                 out_stem="production_plan")
        elif t == "route_sheet":
            self._run_route_sheet_bleu_rouge(instances)
        elif t == "graph":
            self._run_graph(instances, "full", out_stem="graph")
        elif t == "graph_from_gt":
            self._run_graph(instances, "from_gt_route_sheet", out_stem="graph_from_gt")
        elif t == "ground_from_gt":
            self._run_bleu_rouge(instances, "from_gt_schedule", "s7_production_plan.json",
                                 "production_plan.json", need_schedule=True,
                                 out_stem="ground_from_gt")
        elif t == "extract":
            self._run_extract(instances)
        elif t == "verify":
            self._run_verify(instances)
        elif t == "normalize":
            self._run_normalize(instances)
        elif t == "route_sheet_fieldlevel":
            self._run_route_sheet_fieldlevel(instances)

    # ---- structural metrics -------------------------------------------

    def _run_bleu_rouge(self, instances, experiment_type, pipe_filename,
                        gt_filename, need_schedule, out_stem):
        bleu, prec, rec, f1 = [], [], [], []
        skipped = 0
        for inst in tqdm(instances, desc=out_stem):
            pipe = self._pipe_json(experiment_type, inst, pipe_filename)
            gt = self._load_gt_json(inst, gt_filename, need_schedule)
            if pipe is None or gt is None:
                skipped += 1
                continue
            ref = json.dumps(gt)
            cand = json.dumps(pipe)
            bleu.append(_bleu(ref, cand))
            p, r, f = _rouge_like(ref, cand)
            prec.append(p); rec.append(r); f1.append(f)

        report = {
            "instances": instances,
            "skipped": skipped,
            "bleu_mean": (sum(bleu) / len(bleu)) if bleu else 0.0,
            "rouge_mean": {
                "precision": (sum(prec) / len(prec)) if prec else 0.0,
                "recall": (sum(rec) / len(rec)) if rec else 0.0,
                "f1": (sum(f1) / len(f1)) if f1 else 0.0,
            },
            "bleu_per_instance": bleu,
            "rouge_per_instance": {"precision": prec, "recall": rec, "f1": f1},
        }
        out_path = os.path.join(self.out_dir, f"{out_stem}_report.json")
        write_json(out_path, report)
        print(f"{out_stem} → {out_path}  BLEU={report['bleu_mean']:.4f} "
              f"F1={report['rouge_mean']['f1']:.4f}  (skipped {skipped})")

    def _run_route_sheet_bleu_rouge(self, instances):
        """s4_normalized.json → build_route_sheets → compare with GT route_sheets."""
        bleu, prec, rec, f1 = [], [], [], []
        skipped = 0
        for inst in tqdm(instances, desc="route_sheet"):
            pipe = self._pipe_json("full", inst, "s4_normalized.json")
            gt = self._load_gt_json(inst, "route_sheets.json", need_schedule=False)
            if pipe is None or gt is None:
                skipped += 1
                continue
            pipe_rs = _build_route_sheets(pipe, _to_full(inst))
            ref = json.dumps(gt)
            cand = json.dumps(pipe_rs)
            bleu.append(_bleu(ref, cand))
            p, r, f = _rouge_like(ref, cand)
            prec.append(p); rec.append(r); f1.append(f)

        report = {
            "instances": instances,
            "skipped": skipped,
            "bleu_mean": (sum(bleu) / len(bleu)) if bleu else 0.0,
            "rouge_mean": {
                "precision": (sum(prec) / len(prec)) if prec else 0.0,
                "recall": (sum(rec) / len(rec)) if rec else 0.0,
                "f1": (sum(f1) / len(f1)) if f1 else 0.0,
            },
            "bleu_per_instance": bleu,
            "rouge_per_instance": {"precision": prec, "recall": rec, "f1": f1},
        }
        out_path = os.path.join(self.out_dir, "route_sheet_report.json")
        write_json(out_path, report)
        print(f"route_sheet → {out_path}  BLEU={report['bleu_mean']:.4f} "
              f"F1={report['rouge_mean']['f1']:.4f}  (skipped {skipped})")

    def _run_graph(self, instances, experiment_type, out_stem):
        """IoU vs GT operation-precedence + err_rate + makespan_ratio."""
        result = {"accuracy_rate": [], "runtime_err_rate": [], "makespan_ratio": []}
        skipped = 0
        for inst in tqdm(instances, desc=out_stem):
            gt_or = self._load_gt_json(inst, "or_matrix.json", need_schedule=False)
            gt_rs = self._load_gt_json(inst, "route_sheets.json", need_schedule=False)
            or_mat = self._pipe_json(experiment_type, inst, "s5_or_matrix.json")
            machines = self._pipe_json(experiment_type, inst, "s5_machines.json")
            if any(x is None for x in (gt_or, gt_rs, or_mat, machines)):
                skipped += 1
                continue

            gt_res = _gt_resource_constraint(gt_rs)
            gt_op_prec = _gt_operation_precedence(gt_or, gt_rs)

            pipe_res = _pipeline_resource_constraint(or_mat, machines, gt_rs)
            pipe_prec = _pipeline_precedence_constraint(or_mat, gt_rs)

            result["accuracy_rate"].append(_iou(pipe_res + pipe_prec, gt_res + gt_op_prec))

            err_path = os.path.join(self._pipe_dir(experiment_type, inst), "s6_err_rate.txt")
            result["runtime_err_rate"].append(float(read_txt(err_path)) if os.path.exists(err_path) else 0.0)

            gt_makespan = self._load_gt_makespan(inst)
            ms_path = os.path.join(self._pipe_dir(experiment_type, inst), "s6_makespan.txt")
            if gt_makespan > 0 and os.path.exists(ms_path):
                pipe_ms = float(read_txt(ms_path))
                result["makespan_ratio"].append(pipe_ms / gt_makespan)

        def _mean(xs): return sum(xs) / len(xs) if xs else 0.0
        report = {
            "instances": instances,
            "skipped": skipped,
            "accuracy_rate_mean": _mean(result["accuracy_rate"]),
            "runtime_err_rate_mean": _mean(result["runtime_err_rate"]),
            "makespan_ratio_mean": _mean(result["makespan_ratio"]),
            "per_instance": result,
        }
        out_path = os.path.join(self.out_dir, f"{out_stem}_report.json")
        write_json(out_path, report)
        print(f"{out_stem} → {out_path}  IoU={report['accuracy_rate_mean']:.4f} "
              f"err={report['runtime_err_rate_mean']:.4f} "
              f"makespan_ratio={report['makespan_ratio_mean']:.4f}  (skipped {skipped})")

    # ---- field-level metrics ------------------------------------------

    def _iter_gt_s1(self, instances) -> Iterable[tuple]:
        for inst in instances:
            gt_rs = self._load_gt_json(inst, "route_sheets.json", need_schedule=False)
            if gt_rs is None:
                continue
            yield inst, [_fixed_schema_from_route_sheet(job) for job in gt_rs]

    def _run_extract(self, instances):
        totals = _empty_per_field()
        per_instance = {}
        for inst, gt in tqdm(list(self._iter_gt_s1(instances)), desc="extract"):
            pred = self._pipe_json("full", inst, "s1_extracted.json")
            if pred is None:
                continue
            inst_field = _score_fixed_schema(pred, gt)
            _accumulate(totals, inst_field)
            per_instance[inst] = _prf1_by_field(inst_field)
        summary = _prf1_by_field(totals)
        macro = _macro_f1(summary)
        out_path = os.path.join(self.out_dir, "extract_report.json")
        write_json(out_path, {"per_field": summary, "macro_f1": macro, "per_instance": per_instance})
        _print_prf1("extract", out_path, macro, summary)

    def _run_verify(self, instances):
        totals_s1 = _empty_per_field()
        totals_s2 = _empty_per_field()
        totals_fb = _empty_fix_break()
        per_instance = {}
        for inst, gt in tqdm(list(self._iter_gt_s1(instances)), desc="verify"):
            pred_s1 = self._pipe_json("full", inst, "s1_extracted.json")
            pred_s2 = self._pipe_json("full", inst, "s2_verified.json")
            if pred_s1 is None or pred_s2 is None:
                continue
            inst_s1 = _score_fixed_schema(pred_s1, gt)
            inst_s2 = _score_fixed_schema(pred_s2, gt)
            inst_fb = _fix_break_per_field(pred_s1, pred_s2, gt)
            _accumulate(totals_s1, inst_s1)
            _accumulate(totals_s2, inst_s2)
            _accumulate(totals_fb, inst_fb)
            per_instance[inst] = {
                "s1": _prf1_by_field(inst_s1),
                "s2": _prf1_by_field(inst_s2),
                "fix_break": inst_fb,
            }
        s1_summary = _prf1_by_field(totals_s1)
        s2_summary = _prf1_by_field(totals_s2)
        macro_s1 = _macro_f1(s1_summary)
        macro_s2 = _macro_f1(s2_summary)
        delta = {
            f: {
                "f1_delta": s2_summary[f]["f1"] - s1_summary[f]["f1"],
                "precision_delta": s2_summary[f]["precision"] - s1_summary[f]["precision"],
                "recall_delta": s2_summary[f]["recall"] - s1_summary[f]["recall"],
            }
            for f in s1_summary
        }
        result = {
            "s1_per_field": s1_summary,
            "s2_per_field": s2_summary,
            "delta": delta,
            "macro_f1_s1": macro_s1,
            "macro_f1_s2": macro_s2,
            "macro_f1_delta": macro_s2 - macro_s1,
            "fix_break": totals_fb,
            "per_instance": per_instance,
        }
        out_path = os.path.join(self.out_dir, "verify_report.json")
        write_json(out_path, result)
        print(f"verify → {out_path}")
        print(f"  macro F1: s1={macro_s1:.4f}  s2={macro_s2:.4f}  Δ={macro_s2-macro_s1:+.4f}")
        for f in s1_summary:
            fb = totals_fb[f]
            print(f"    {f:15s} s1_F1={s1_summary[f]['f1']:.3f}  s2_F1={s2_summary[f]['f1']:.3f}  "
                  f"Δ={delta[f]['f1_delta']:+.3f}   fix={fb['fix']} break={fb['break']} "
                  f"still_wrong={fb['still_wrong']} still_right={fb['still_right']}")

    def _run_route_sheet_fieldlevel(self, instances):
        totals = _empty_per_field()
        per_instance = {}
        for inst, gt in tqdm(list(self._iter_gt_s1(instances)), desc="route_sheet_fieldlevel"):
            pred = self._pipe_json("full", inst, "s4_normalized.json")
            if pred is None:
                continue
            inst_field = _score_fixed_schema(pred, gt)
            _accumulate(totals, inst_field)
            per_instance[inst] = _prf1_by_field(inst_field)
        summary = _prf1_by_field(totals)
        macro = _macro_f1(summary)
        out_path = os.path.join(self.out_dir, "route_sheet_fieldlevel_report.json")
        write_json(out_path, {"per_field": summary, "macro_f1": macro, "per_instance": per_instance})
        _print_prf1("route_sheet_fieldlevel", out_path, macro, summary)

    def _run_normalize(self, instances):
        cats = ("machine", "operation", "component_type", "param_key", "param_value")
        totals = {c: {"pre": 0, "post": 0, "mapping_members": 0, "distinct_canonicals": 0}
                  for c in cats}
        per_instance = {}
        for inst in tqdm(instances, desc="normalize"):
            s3 = self._pipe_json("full", inst, "s3_formatted.json")
            mapping = self._pipe_json("full", inst, "s4_mappings.json")
            if s3 is None or mapping is None:
                continue
            inst_stats = self._score_normalize_one(s3, mapping)
            for cat, stats in inst_stats.items():
                for k, v in stats.items():
                    totals[cat][k] += v
            per_instance[inst] = inst_stats

        summary = {}
        for cat, s in totals.items():
            pre, post = s["pre"], s["post"]
            summary[cat] = {
                "pre_variant_count": pre,
                "post_variant_count": post,
                "shrinkage": pre - post,
                "shrinkage_rate": (pre - post) / pre if pre else 0.0,
                "mapping_members": s["mapping_members"],
                "distinct_canonicals": s["distinct_canonicals"],
            }
        out_path = os.path.join(self.out_dir, "normalize_report.json")
        write_json(out_path, {"per_category": summary, "per_instance": per_instance})
        print(f"normalize → {out_path}")
        for cat, s in summary.items():
            print(f"  {cat:15s} pre={s['pre_variant_count']:5d} → post={s['post_variant_count']:5d} "
                  f"(shrink {s['shrinkage']:4d}, {s['shrinkage_rate']*100:5.1f}%)  "
                  f"mapped {s['mapping_members']} → {s['distinct_canonicals']}")

    @staticmethod
    def _score_normalize_one(s3_data, mapping):
        cats = ("machine", "operation", "component_type", "param_key", "param_value")
        pre = {c: set() for c in cats}
        for data in s3_data:
            if not isinstance(data, dict) or "bad_case" in data:
                continue
            for step in data.get("steps", []):
                if step.get("machine"): pre["machine"].add(_norm(step["machine"]))
                if step.get("operation"): pre["operation"].add(_norm(step["operation"]))
                for field in ("precondition", "postcondition"):
                    for item in step.get(field, []) or []:
                        if isinstance(item, dict) and item.get("component_type"):
                            pre["component_type"].add(_norm(item["component_type"]))
                for k, v in (step.get("parameters", {}) or {}).items():
                    pre["param_key"].add(_norm(k))
                    if isinstance(v, str):
                        pre["param_value"].add(_norm(v))

        mapping_by_cat = {c: {} for c in cats}
        for member, canonical in (mapping or {}).items():
            m_norm = _norm(member)
            for cat, pre_set in pre.items():
                if m_norm in pre_set:
                    mapping_by_cat[cat][m_norm] = _norm(canonical)

        stats = {}
        for cat, pre_set in pre.items():
            cat_map = mapping_by_cat[cat]
            post_set = {cat_map.get(v, v) for v in pre_set}
            stats[cat] = {
                "pre": len(pre_set),
                "post": len(post_set),
                "mapping_members": len(cat_map),
                "distinct_canonicals": len(set(cat_map.values())),
            }
        return stats


# --------------------------------------------------------------------- #
#  CLI                                                                  #
# --------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--type", dest="etype", default="all",
                    choices=list(ALL_TYPES) + ["all"],
                    help="Which metric to compute (default: all).")
    ap.add_argument("--instances", nargs="+", default=None,
                    help="Instance short-names. Default: all 10.")
    ap.add_argument("--artifacts-dir", default=os.path.join(_HERE, "artifacts"),
                    help="Where pipeline/, groundtruth/, evaluate/ live.")
    ap.add_argument("--preprocess-dir", default=os.path.join(_HERE, "preprocess_out"),
                    help="Where jssp_mapped / route_sheet_reduce live (for GT build).")
    ap.add_argument("--data-dir", default=os.path.join(_HERE, "data"),
                    help="Where machines.json lives (for GT build).")
    args = ap.parse_args()

    insts = [_to_short(i) for i in (args.instances or ALL_INSTANCES)]

    ev = SchemaEvaluator(
        artifacts_dir=args.artifacts_dir,
        preprocess_dir=args.preprocess_dir,
        data_dir=args.data_dir,
    )
    ev.run(args.etype, insts)


if __name__ == "__main__":
    main()
