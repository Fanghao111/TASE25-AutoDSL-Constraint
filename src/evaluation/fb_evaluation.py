"""FB pipeline evaluation harness.

Eight evaluation_types covering the whole FB pipeline (s1..s7):

Reused (delegated to `Evaluation`):
    production_plan       — s7 output vs GT production_plan
    route_sheet           — s4_normalized (via build_route_sheets) vs GT route_sheet
    graph                 — s5 or_matrix (full path) vs GT or_matrix
    graph_from_gt         — s5 or_matrix (from_gt_route_sheet path) vs GT or_matrix
    ground_from_gt        — s7 (from_gt_schedule path) vs GT production_plan

New (isolate the LLM steps; each reads FB pipeline intermediate outputs and
compares against GT — no synthetic data):
    extract               — s1_extracted vs GT_s1 (field-level P/R/F1)
    verify                — s2_verified vs GT_s1 relative to s1_extracted
                            (fix_rate / break_rate / net_improvement)
    normalize             — s4_mappings vs canonical set aggregated from all GT route_sheets
                            (canonical_hit_rate / resolve_rate / post_variant_count)

GT_s1 is reverse-projected from GT route_sheet: {"steps": job["route_sheet"]}.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List

from tqdm import tqdm

from src.evaluation.evaluation import Evaluation
from utils.util import read_json, write_json


# --------------------------------------------------------------------- #
#  Small helpers                                                        #
# --------------------------------------------------------------------- #


def _norm(value: Any) -> str:
    return str(value).strip().lower()


def _parse_duration_minutes(text: Any) -> int:
    """Extract minutes as integer from a duration string. Mirrors FBPipeline._parse_duration."""
    text = str(text).lower().strip()
    match = re.search(r"([\d.]+)", text)
    if not match:
        return 0
    value = float(match.group(1))
    if "hour" in text or "hr" in text:
        return int(value * 60)
    if "second" in text or "sec" in text:
        return max(1, int(value / 60))
    return int(value)


def _fixed_schema_from_route_sheet(rs_job: Dict[str, Any]) -> Dict[str, Any]:
    """{"steps": job.route_sheet} — matches FBPipeline._route_sheet_to_fixed_schema."""
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


# --------------------------------------------------------------------- #
#  Main class                                                           #
# --------------------------------------------------------------------- #


ALL_EVALUATION_TYPES = [
    "production_plan",
    "route_sheet",
    "graph",
    "graph_from_gt",
    "ground_from_gt",
    "extract",
    "verify",
    "normalize",
    "route_sheet_fieldlevel",
]

EXISTING_TYPES = {"production_plan", "route_sheet", "graph", "graph_from_gt", "ground_from_gt"}
NEW_TYPES = {"extract", "verify", "normalize", "route_sheet_fieldlevel"}


class FBEvaluation:
    """FB-focused evaluator. Reuses `Evaluation` for the 5 existing metrics
    and adds 4 isolation metrics for s1 / s2 / s4 / s5.

    Reads FB pipeline outputs from FB_EVAL_LABEL-driven directory (same convention
    as `Evaluation`); writes new-metric outputs alongside existing evaluation outputs.
    """

    def __init__(self):
        self.evaluator = Evaluation()          # delegate for existing 5
        self.groundtruth_dir = "outputs/GroundTruth"

        # FB_EVAL_LABEL=FB-models/<model> → outputs/FB-models/<model>/, outputs/Evaluation/FB-models_<model>/
        fb_label = os.environ.get("FB_EVAL_LABEL")
        self.fb_dir = f"outputs/{fb_label}/" if fb_label else "outputs/FB-2s-full/"
        self.out_dir = (f"outputs/Evaluation/{fb_label.replace('/', '_')}/"
                        if fb_label else "outputs/Evaluation/")
        os.makedirs(self.out_dir, exist_ok=True)

    # ----------------------------------------------------------------- #
    #  Dispatch                                                         #
    # ----------------------------------------------------------------- #

    def run(self, evaluation_type: str):
        if evaluation_type == "all":
            for t in ALL_EVALUATION_TYPES:
                print(f"\n=== FB evaluation: {t} ===", flush=True)
                self._run_one(t)
            return
        if evaluation_type not in ALL_EVALUATION_TYPES:
            raise ValueError(
                f"Unknown evaluation_type '{evaluation_type}'. "
                f"Choose from {ALL_EVALUATION_TYPES + ['all']}."
            )
        self._run_one(evaluation_type)

    def _run_one(self, evaluation_type: str):
        if evaluation_type in EXISTING_TYPES:
            self.evaluator.evaluate(experiment_type=evaluation_type)
        elif evaluation_type == "extract":
            self._run_extract()
        elif evaluation_type == "verify":
            self._run_verify()
        elif evaluation_type == "normalize":
            self._run_normalize()
        elif evaluation_type == "route_sheet_fieldlevel":
            self._run_route_sheet_fieldlevel()

    # ----------------------------------------------------------------- #
    #  Data loading                                                     #
    # ----------------------------------------------------------------- #

    def _iter_instances(self) -> Iterable[str]:
        """Yield subfolder names in outputs/GroundTruth that also have FB s1 output."""
        if not os.path.isdir(self.groundtruth_dir):
            return
        for sub in sorted(os.listdir(self.groundtruth_dir)):
            gt_rs = os.path.join(self.groundtruth_dir, sub, "route_sheets.json")
            if os.path.exists(gt_rs):
                yield sub

    def _load_gt_s1(self, instance: str) -> List[Dict[str, Any]]:
        """Reverse-project GT route_sheet → list of {"steps": [...]}."""
        path = os.path.join(self.groundtruth_dir, instance, "route_sheets.json")
        rs = read_json(path)
        return [_fixed_schema_from_route_sheet(job) for job in rs]

    def _load_fb_json(self, instance: str, filename: str) -> Any:
        """Load a per-instance FB pipeline artifact; returns None if missing."""
        path = os.path.join(self.fb_dir, instance, filename)
        if not os.path.exists(path):
            return None
        return read_json(path)

    # ----------------------------------------------------------------- #
    #  s1 extract — field-level P/R/F1                                  #
    # ----------------------------------------------------------------- #

    def _run_extract(self):
        """Compare pred `s1_extracted.json` against GT_s1 with per-field P/R/F1.

        See `_score_fixed_schema` — the same scoring used by verify and
        route_sheet_fieldlevel so all three metrics are on the same scale.
        """
        totals = _empty_per_field()
        per_instance = {}

        for inst in tqdm(list(self._iter_instances()), desc="extract"):
            pred = self._load_fb_json(inst, "s1_extracted.json")
            if pred is None:
                continue
            gt = self._load_gt_s1(inst)
            inst_field = _score_fixed_schema(pred, gt)
            _accumulate(totals, inst_field)
            per_instance[inst] = _prf1_by_field(inst_field)

        summary = _prf1_by_field(totals)
        macro = _macro_f1(summary)
        result = {"per_field": summary, "macro_f1": macro, "per_instance": per_instance}
        out_path = os.path.join(self.out_dir, "extract_fb.json")
        write_json(out_path, result)
        _print_prf1("extract", out_path, macro, summary)

    # ----------------------------------------------------------------- #
    #  s2 verify — per-field F1 for s1 & s2 + step-level fix/break      #
    # ----------------------------------------------------------------- #

    def _run_verify(self):
        """Score s1 and s2 against GT on the SAME scale as `extract`, then
        report the delta. Fix/break buckets are also computed at per-field
        granularity so you can see whether verify tends to churn or improve.

        Output:
            s1_per_field      — {field: P/R/F1/tp/fp/fn}
            s2_per_field      — same, on pred_s2 vs GT
            delta             — {field: {f1_delta, precision_delta, recall_delta}}
            macro_f1_s1 / macro_f1_s2 / macro_f1_delta
            fix_break         — per-field step-level counts: fix/break/still_wrong/still_right
                                where "matched" is defined per field
                                (see _match_step for how each field is compared)
        """
        totals_s1 = _empty_per_field()
        totals_s2 = _empty_per_field()
        totals_fb = _empty_fix_break()
        per_instance = {}

        for inst in tqdm(list(self._iter_instances()), desc="verify"):
            pred_s1 = self._load_fb_json(inst, "s1_extracted.json")
            pred_s2 = self._load_fb_json(inst, "s2_verified.json")
            if pred_s1 is None or pred_s2 is None:
                continue
            gt = self._load_gt_s1(inst)

            inst_s1 = _score_fixed_schema(pred_s1, gt)
            inst_s2 = _score_fixed_schema(pred_s2, gt)
            inst_fb = _fix_break_per_field(pred_s1, pred_s2, gt)

            _accumulate(totals_s1, inst_s1)
            _accumulate(totals_s2, inst_s2)
            _accumulate_fix_break(totals_fb, inst_fb)

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
        out_path = os.path.join(self.out_dir, "verify_fb.json")
        write_json(out_path, result)
        print(f"verify → {out_path}")
        print(f"  macro F1: s1={macro_s1:.4f}  s2={macro_s2:.4f}  Δ={macro_s2-macro_s1:+.4f}")
        for f in s1_summary:
            fb = totals_fb[f]
            print(f"    {f:15s} s1_F1={s1_summary[f]['f1']:.3f}  s2_F1={s2_summary[f]['f1']:.3f}  "
                  f"Δ={delta[f]['f1_delta']:+.3f}   fix={fb['fix']} break={fb['break']} "
                  f"still_wrong={fb['still_wrong']} still_right={fb['still_right']}")

    # ----------------------------------------------------------------- #
    #  s4 → route_sheet field-level (same scoring as extract/verify)    #
    # ----------------------------------------------------------------- #

    def _run_route_sheet_fieldlevel(self):
        """Field-level P/R/F1 for FB s4 output vs GT route_sheet.

        Uses the same _score_fixed_schema as extract/verify so the three metrics
        (s1_F1, s2_F1, s4_F1) can be read as a cascade to see where quality is
        lost. The `s4_normalized.json` file is already in {"steps": [...]} shape.
        """
        totals = _empty_per_field()
        per_instance = {}

        for inst in tqdm(list(self._iter_instances()), desc="route_sheet_fieldlevel"):
            pred = self._load_fb_json(inst, "s4_normalized.json")
            if pred is None:
                continue
            gt = self._load_gt_s1(inst)
            inst_field = _score_fixed_schema(pred, gt)
            _accumulate(totals, inst_field)
            per_instance[inst] = _prf1_by_field(inst_field)

        summary = _prf1_by_field(totals)
        macro = _macro_f1(summary)
        result = {"per_field": summary, "macro_f1": macro, "per_instance": per_instance}
        out_path = os.path.join(self.out_dir, "route_sheet_fieldlevel_fb.json")
        write_json(out_path, result)
        _print_prf1("route_sheet_fieldlevel", out_path, macro, summary)

    # ----------------------------------------------------------------- #
    #  s4 normalize — pre → post shrinkage per category                 #
    # ----------------------------------------------------------------- #

    def _run_normalize(self):
        """Report how much s4 collapses the vocabulary in each category.

        s4's job is synonym merging; the canonical name it picks does NOT need to
        string-match any GT vocabulary (GT and s4 can both be "correct" while
        using different canonical strings). So we deliberately do not compare to
        GT. We only report per-category shrinkage:

            pre_variant_count   # unique values in s3_formatted (input to s4)
            post_variant_count  # unique values after applying mapping
            shrinkage           # pre - post (absolute reduction)
            shrinkage_rate      # (pre - post) / pre
            mapping_members     # how many s3 values got a remap entry
            distinct_canonicals # how many distinct target names the mapping uses

        Aggregate totals across all instances, plus per-instance breakdown.
        """
        totals = {cat: {"pre": 0, "post": 0, "mapping_members": 0,
                        "distinct_canonicals": 0}
                  for cat in ("machine", "operation", "component_type",
                              "param_key", "param_value")}
        per_instance = {}

        for inst in tqdm(list(self._iter_instances()), desc="normalize"):
            s3 = self._load_fb_json(inst, "s3_formatted.json")
            mapping = self._load_fb_json(inst, "s4_mappings.json")
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

        result = {"per_category": summary, "per_instance": per_instance}
        out_path = os.path.join(self.out_dir, "normalize_fb.json")
        write_json(out_path, result)
        print(f"normalize → {out_path}")
        for cat, s in summary.items():
            print(f"  {cat:15s} pre={s['pre_variant_count']:5d} → post={s['post_variant_count']:5d} "
                  f"(shrink {s['shrinkage']:4d}, {s['shrinkage_rate']*100:5.1f}%)  "
                  f"mapped {s['mapping_members']} → {s['distinct_canonicals']}")

    def _score_normalize_one(self, s3_data, mapping):
        """Per-instance normalize stats (no GT comparison)."""
        pre = {cat: set() for cat in ("machine", "operation", "component_type",
                                      "param_key", "param_value")}
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

        # mapping is a flat {member: canonical}; bucket each entry by which
        # pre-set the MEMBER appears in (the mapping file doesn't carry categories).
        mapping_by_cat = {cat: {} for cat in pre}
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
#  Metric primitives                                                    #
# --------------------------------------------------------------------- #
#  Metric primitives                                                    #
# --------------------------------------------------------------------- #


def _prf1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _scalar_cmp(pred: str, gt: str) -> Dict[str, int]:
    if not pred and not gt:
        return {"tp": 0, "fp": 0, "fn": 0}
    if pred == gt:
        return {"tp": 1, "fp": 0, "fn": 0}
    if pred and not gt:
        return {"tp": 0, "fp": 1, "fn": 0}
    if gt and not pred:
        return {"tp": 0, "fp": 0, "fn": 1}
    # both non-empty, different — count as both FP (wrong prediction) and FN (missed truth)
    return {"tp": 0, "fp": 1, "fn": 1}


def _multiset_cmp(pred: List[str], gt: List[str]) -> Dict[str, int]:
    """Sorted-multiset match: tp = size of intersection with multiplicity."""
    from collections import Counter
    pc, gc = Counter(pred), Counter(gt)
    tp = sum((pc & gc).values())
    fp = sum(pc.values()) - tp
    fn = sum(gc.values()) - tp
    return {"tp": tp, "fp": fp, "fn": fn}


def _set_cmp(pred: set, gt: set) -> Dict[str, int]:
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)
    return {"tp": tp, "fp": fp, "fn": fn}


# --------------------------------------------------------------------- #
#  Shared fixed-schema scoring — used by extract / verify / route_sheet_fieldlevel
# --------------------------------------------------------------------- #

_FIELDS = ("machine", "operation", "duration",
           "component_type", "param_key", "param_value")


def _empty_per_field() -> Dict[str, Dict[str, int]]:
    return {f: {"tp": 0, "fp": 0, "fn": 0} for f in _FIELDS}


def _empty_fix_break() -> Dict[str, Dict[str, int]]:
    return {f: {"fix": 0, "break": 0, "still_wrong": 0, "still_right": 0}
            for f in _FIELDS}


def _accumulate(dst: Dict[str, Dict[str, int]], src: Dict[str, Dict[str, int]]):
    for f, d in src.items():
        for k, v in d.items():
            dst[f][k] += v


def _accumulate_fix_break(dst: Dict[str, Dict[str, int]], src: Dict[str, Dict[str, int]]):
    _accumulate(dst, src)  # same structure — reuse


def _prf1_by_field(per_field: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    return {f: _prf1(**per_field[f]) for f in per_field}


def _macro_f1(summary: Dict[str, Dict[str, float]]) -> float:
    return sum(s["f1"] for s in summary.values()) / len(summary) if summary else 0.0


def _print_prf1(name: str, out_path: str, macro: float,
                summary: Dict[str, Dict[str, float]]):
    print(f"{name} → {out_path}")
    print(f"  macro F1 = {macro:.4f}")
    for f, s in summary.items():
        print(f"    {f:15s} P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f}")


def _score_fixed_schema(pred_jobs: List[Any], gt_jobs: List[Any]
                        ) -> Dict[str, Dict[str, int]]:
    """Score two lists of `{"steps": [...]}` (or entries containing "bad_case").

    Aligns by index at both the job and step level. Missing pred → pure FN;
    extra pred → pure FP. Returns per-field TP/FP/FN aggregated over all jobs
    and steps in this pair.
    """
    per_field = _empty_per_field()
    n_jobs = max(len(pred_jobs), len(gt_jobs))
    for j in range(n_jobs):
        p_job = pred_jobs[j] if j < len(pred_jobs) else None
        g_job = gt_jobs[j] if j < len(gt_jobs) else None
        p_steps = p_job.get("steps", []) if isinstance(p_job, dict) else []
        g_steps = g_job.get("steps", []) if isinstance(g_job, dict) else []
        n_steps = max(len(p_steps), len(g_steps))
        for s in range(n_steps):
            p_step = p_steps[s] if s < len(p_steps) else None
            g_step = g_steps[s] if s < len(g_steps) else None
            for f, delta in _score_step_pure(p_step, g_step).items():
                for k, v in delta.items():
                    per_field[f][k] += v
    return per_field


def _score_step_pure(p_step, g_step) -> Dict[str, Dict[str, int]]:
    """Return per-field TP/FP/FN deltas for a single step pair (no side effects)."""
    out: Dict[str, Dict[str, int]] = {}

    out["machine"] = _scalar_cmp(
        _norm((p_step or {}).get("machine", "")),
        _norm((g_step or {}).get("machine", "")),
    )
    out["operation"] = _scalar_cmp(
        _norm((p_step or {}).get("operation", "")),
        _norm((g_step or {}).get("operation", "")),
    )
    out["duration"] = _scalar_cmp(
        str(_parse_duration_minutes((p_step or {}).get("duration", "0"))),
        str(_parse_duration_minutes((g_step or {}).get("duration", "0"))),
    )

    p_cts = (_step_component_types(p_step, "precondition")
             + _step_component_types(p_step, "postcondition")) if p_step else []
    g_cts = (_step_component_types(g_step, "precondition")
             + _step_component_types(g_step, "postcondition")) if g_step else []
    out["component_type"] = _multiset_cmp(p_cts, g_cts)

    p_params = _step_parameters(p_step) if p_step else {}
    g_params = _step_parameters(g_step) if g_step else {}
    out["param_key"] = _set_cmp(set(p_params), set(g_params))
    common = set(p_params) & set(g_params)
    pv_tp = sum(1 for k in common if p_params[k] == g_params[k])
    pv_fp = sum(1 for k in common if p_params[k] != g_params[k])
    out["param_value"] = {"tp": pv_tp, "fp": pv_fp, "fn": pv_fp}
    return out


def _fix_break_per_field(pred_s1_jobs: List[Any], pred_s2_jobs: List[Any],
                         gt_jobs: List[Any]) -> Dict[str, Dict[str, int]]:
    """For each (job, step, field), classify (s1 vs GT, s2 vs GT) into one of
    {fix, break, still_wrong, still_right} where "matched" means TP == max(pred, gt)
    for that field (i.e. exact match after normalization). Aggregate counts per field.
    """
    counts = _empty_fix_break()
    n_jobs = max(len(pred_s1_jobs), len(pred_s2_jobs), len(gt_jobs))
    for j in range(n_jobs):
        s1_job = pred_s1_jobs[j] if j < len(pred_s1_jobs) else None
        s2_job = pred_s2_jobs[j] if j < len(pred_s2_jobs) else None
        g_job = gt_jobs[j] if j < len(gt_jobs) else None
        s1_steps = s1_job.get("steps", []) if isinstance(s1_job, dict) else []
        s2_steps = s2_job.get("steps", []) if isinstance(s2_job, dict) else []
        g_steps = g_job.get("steps", []) if isinstance(g_job, dict) else []
        n_steps = max(len(s1_steps), len(s2_steps), len(g_steps))
        for s in range(n_steps):
            s1_step = s1_steps[s] if s < len(s1_steps) else None
            s2_step = s2_steps[s] if s < len(s2_steps) else None
            g_step = g_steps[s] if s < len(g_steps) else None
            s1_scores = _score_step_pure(s1_step, g_step)
            s2_scores = _score_step_pure(s2_step, g_step)
            for f in _FIELDS:
                s1_ok = _step_field_ok(s1_scores[f])
                s2_ok = _step_field_ok(s2_scores[f])
                if not s1_ok and s2_ok: counts[f]["fix"] += 1
                elif s1_ok and not s2_ok: counts[f]["break"] += 1
                elif not s1_ok and not s2_ok: counts[f]["still_wrong"] += 1
                else: counts[f]["still_right"] += 1
    return counts


def _step_field_ok(field_delta: Dict[str, int]) -> bool:
    """A step's field is 'ok' iff no FP and no FN for it."""
    return field_delta["fp"] == 0 and field_delta["fn"] == 0
