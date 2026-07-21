"""sschema pipeline — the 7-step FB pipeline (NL → structured JSON → CP-SAT → plan).

This is the CENTRAL script of sschema. Reads pre-generated NL orders under
`{preprocess_dir}/orders/<inst>/orders.json`, runs the 7 stages s1..s7, and
writes each intermediate artifact + the final production plan under
`{artifacts_dir}/pipeline/<experiment_type>/<inst>/`.

Experiment types:
  full                 — s1..s7 end-to-end from NL orders
  from_gt_route_sheet  — skip s1..s4, use GT route sheet directly, run s5..s7
  from_gt_schedule     — skip s1..s6, use GT schedule directly, run s7 only

For the two `from_gt_*` types, the ground truth is built automatically under
`{artifacts_dir}/groundtruth/<inst>/` if not already present.

Environment variables:
  LLM_BASE_URL, LLM_MODEL, OPENAI_API_KEY, LLM_MAX_WORKERS — see common/llm.py
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from common.cpsat import schedule
from common.format_rules import format_data as _code_format_data
from common.groundtruth import GroundTruth
from common.io import read_json, read_txt, write_json, write_txt
from common.llm import LLM_MODEL, make_chat_client
from common.schema_validator import validate_extraction


ALL_INSTANCES = [f"ta{i}" for i in range(71, 81)]
EXPERIMENT_TYPES = ("full", "from_gt_route_sheet", "from_gt_schedule")


def _to_full(inst: str) -> str:
    return inst if inst.startswith("instance ") else f"instance {inst}"


def _to_short(inst: str) -> str:
    return inst.replace("instance ", "")


class SchemaPipeline:
    """Structured-schema pipeline for a single instance.

    Stage map (same as fb-2s src/experiment/fb_pipeline.py):
        s1 extract    NL order → fixed-schema JSON             (LLM, per-order)
        s2 verify     factual verification vs NL                (LLM, per-step)
        s3 format     deterministic Title/Pascal/lower casing   (code)
        s4 normalize  global synonym merging across all orders  (LLM, 1 call + verify)
        s5 graph      build OR matrix / disjunctive graph       (code)
        s6 solve      JSP CP-SAT scheduling                     (OR-Tools)
        s7 ground     reproject schedule back to production plan (code)
    """

    def __init__(
        self,
        instance_description: str,
        preprocess_dir: str,
        artifacts_dir: str,
        prompts_dir: str,
        data_dir: str,
        experiment_type: str = "full",
        force: bool = True,
    ):
        if experiment_type not in EXPERIMENT_TYPES:
            raise ValueError(
                f"experiment_type must be one of {EXPERIMENT_TYPES}, got {experiment_type!r}"
            )
        self.instance_description = _to_full(instance_description)
        self.instance_short = _to_short(self.instance_description)
        self.experiment_type = experiment_type
        self.force = force

        self.preprocess_dir = preprocess_dir
        self.artifacts_dir = artifacts_dir
        self.prompts_dir = prompts_dir
        self.data_dir = data_dir

        self.pipeline_dir = os.path.join(
            artifacts_dir, "pipeline", experiment_type, self.instance_short
        )
        self.groundtruth_dir = os.path.join(
            artifacts_dir, "groundtruth", self.instance_short
        )

        self.orders = []
        self.extracted_jsons = []
        self.verified_jsons = []
        self.formatted_jsons = []
        self.normalized_jsons = []
        self.or_matrix = []
        self.machines = []
        self.assigned_jobs = {}
        self.production_plan = []

        self.compile_error_num = 0
        self.total_num = 0

        self.s1_extract_prompt = read_txt(os.path.join(prompts_dir, "step1_extract.txt"))
        self.s2_verify_prompt = read_txt(os.path.join(prompts_dir, "step1_verify.txt"))
        self.s4_normalize_prompt = read_txt(os.path.join(prompts_dir, "step3_normalize.txt"))
        self.s4_verify_prompt = read_txt(os.path.join(prompts_dir, "step3_verify.txt"))

    # ------------------------------------------------------------------ #
    #  Entry                                                             #
    # ------------------------------------------------------------------ #

    def run(self):
        os.makedirs(self.pipeline_dir, exist_ok=True)
        self._load_intermediates()

        if self.experiment_type == "full":
            orders_path = os.path.join(
                self.preprocess_dir, "orders", self.instance_short, "orders.json"
            )
            if not os.path.exists(orders_path):
                raise FileNotFoundError(
                    f"Missing orders at {orders_path}. "
                    "Run `python sschema/preprocess.py --stage orders` first "
                    "or point --preprocess-dir at a directory that contains them."
                )
            self.orders = read_json(orders_path)

            if not self.extracted_jsons:
                self.extract()
            if not self.verified_jsons:
                self.verify()
            if not self.formatted_jsons:
                self.apply_format()
            if not self.normalized_jsons:
                self.normalize()
            self.build_graph()
            self.solve()
            self.ground()

        elif self.experiment_type == "from_gt_route_sheet":
            gt = self._ensure_groundtruth(need_schedule=False)
            route_sheets = gt.get_grounded_route_sheet()
            self.normalized_jsons = [
                {"steps": rs.get("route_sheet", [])} for rs in route_sheets
            ]
            self.build_graph()
            self.solve()
            self.ground()

        elif self.experiment_type == "from_gt_schedule":
            gt = self._ensure_groundtruth(need_schedule=True)
            self.assigned_jobs = gt.get_grounded_assigned_jobs()
            route_sheets = gt.get_grounded_route_sheet()
            self.normalized_jsons = [
                {"steps": rs.get("route_sheet", [])} for rs in route_sheets
            ]
            self.ground()

    def _ensure_groundtruth(self, need_schedule: bool) -> GroundTruth:
        gt = GroundTruth(
            instance_description=self.instance_description,
            route_sheet_reduce_path=os.path.join(self.preprocess_dir, "route_sheet_reduce.json"),
            jssp_mapped_path=os.path.join(self.preprocess_dir, "jssp_mapped.json"),
            machines_data_path=os.path.join(self.data_dir, "machines.json"),
            dump_dir_path=self.groundtruth_dir,
        )
        gt.get_grounded_route_sheet()
        gt.get_grounded_or_matrix()
        if need_schedule:
            gt.get_grounded_assigned_jobs()
            gt.get_grounded_production_plan()
        return gt

    # ------------------------------------------------------------------ #
    #  s1: Extract                                                       #
    # ------------------------------------------------------------------ #

    def extract(self):
        print("s1 extract: NL → fixed-schema JSON ...", flush=True)
        prompts = [
            self.s1_extract_prompt.replace("---ORDER---", json.dumps(order))
            for order in self.orders
        ]
        results = self._parallel_llm_calls(prompts)

        self.extracted_jsons = []
        for i, result in enumerate(results):
            parsed = self._safe_json_parse(result)
            is_valid, err = validate_extraction(parsed)
            if is_valid:
                self.extracted_jsons.append(parsed)
            else:
                self.extracted_jsons.append({"steps": [], "bad_case": f"s1 extract: {err}"})

        bad_indices = [i for i, d in enumerate(self.extracted_jsons) if "bad_case" in d]
        if bad_indices:
            print(f"  Retrying {len(bad_indices)} failed extractions...", flush=True)
            for idx in bad_indices:
                result = self._chatgpt_function(prompts[idx])
                parsed = self._safe_json_parse(result)
                is_valid, err = validate_extraction(parsed)
                if is_valid:
                    self.extracted_jsons[idx] = parsed
            still_bad = sum(1 for d in self.extracted_jsons if "bad_case" in d)
            print(f"  After retry: {still_bad}/{len(self.extracted_jsons)} still bad", flush=True)

        write_json(os.path.join(self.pipeline_dir, "s1_extracted.json"), self.extracted_jsons)

    # ------------------------------------------------------------------ #
    #  s2: Verify                                                        #
    # ------------------------------------------------------------------ #

    def verify(self):
        """Per-step factual verification against NL. Preserves step count by construction."""
        print("s2 verify: LLM per-step factual verification vs NL ...", flush=True)

        tasks = []
        for i, (order, extracted) in enumerate(zip(self.orders, self.extracted_jsons)):
            if "bad_case" in extracted:
                continue
            nl_steps = order.get("steps", []) if isinstance(order, dict) else []
            for j, s1_step in enumerate(extracted.get("steps", [])):
                if j >= len(nl_steps):
                    continue
                prompt = self.s2_verify_prompt \
                    .replace("---STEP_NL---", json.dumps(nl_steps[j])) \
                    .replace("---STEP_JSON---", json.dumps(s1_step))
                tasks.append((i, j, prompt))

        prompts = [t[2] for t in tasks]
        results = self._parallel_llm_calls(prompts) if prompts else []

        self.verified_jsons = [copy.deepcopy(x) for x in self.extracted_jsons]

        s2_accept = s2_fallback = 0
        for (i, j, prompt), raw in zip(tasks, results):
            step = self._parse_single_step(raw)
            if step is None:
                raw2 = self._chatgpt_function(prompt)
                step = self._parse_single_step(raw2)
            if step is not None:
                self.verified_jsons[i]["steps"][j] = step
                s2_accept += 1
            else:
                s2_fallback += 1

        total = s2_accept + s2_fallback
        print(f"  s2 accepted: {s2_accept}/{total}  fell back to s1 step: {s2_fallback}",
              flush=True)
        write_json(os.path.join(self.pipeline_dir, "s2_verified.json"), self.verified_jsons)

    @staticmethod
    def _single_step_schema_ok(step) -> bool:
        if not isinstance(step, dict):
            return False
        required = {"operation", "machine", "duration", "precondition", "postcondition"}
        if not required.issubset(step.keys()):
            return False
        if not str(step.get("machine", "")).strip():
            return False
        if not any(c.isdigit() for c in str(step.get("duration", ""))):
            return False
        for f in ("precondition", "postcondition"):
            if not isinstance(step.get(f), list):
                return False
        return True

    def _parse_single_step(self, raw):
        parsed = self._safe_json_parse(raw)
        if not parsed:
            return None
        if isinstance(parsed, dict) and "steps" in parsed:
            steps = parsed.get("steps") or []
            if not steps:
                return None
            parsed = steps[0]
        if isinstance(parsed, list):
            if not parsed:
                return None
            parsed = parsed[0]
        return parsed if self._single_step_schema_ok(parsed) else None

    # ------------------------------------------------------------------ #
    #  s3: Format                                                        #
    # ------------------------------------------------------------------ #

    def apply_format(self):
        print("s3 format (code): applying Title/Pascal/lower casing ...", flush=True)
        self.formatted_jsons = [_code_format_data(d) for d in self.verified_jsons]
        write_json(os.path.join(self.pipeline_dir, "s3_formatted.json"), self.formatted_jsons)

    # ------------------------------------------------------------------ #
    #  s4: Normalize                                                     #
    # ------------------------------------------------------------------ #

    def normalize(self):
        print("s4 normalize: global synonym merging ...", flush=True)

        machines = set()
        operations = set()
        component_types = set()
        param_keys = set()
        param_values = set()

        valid_data = [d for d in self.formatted_jsons if "bad_case" not in d]

        for data in valid_data:
            for step in data.get("steps", []):
                if step.get("machine"):
                    machines.add(step["machine"])
                if step.get("operation"):
                    operations.add(step["operation"])
                for field in ("precondition", "postcondition"):
                    for item in step.get(field, []):
                        if isinstance(item, dict) and item.get("component_type"):
                            component_types.add(item["component_type"])
                for k, v in step.get("parameters", {}).items():
                    param_keys.add(k)
                    if isinstance(v, str):
                        param_values.add(v)

        prompt = self.s4_normalize_prompt \
            .replace("---MACHINES---", json.dumps(sorted(machines))) \
            .replace("---OPERATIONS---", json.dumps(sorted(operations))) \
            .replace("---COMPONENT_TYPES---", json.dumps(sorted(component_types))) \
            .replace("---PARAM_KEYS---", json.dumps(sorted(param_keys))) \
            .replace("---PARAM_VALUES---", json.dumps(sorted(param_values)))

        print(f"  Unique values: {len(machines)} machines, {len(operations)} operations, "
              f"{len(component_types)} types, {len(param_keys)} param keys", flush=True)

        result = self._chatgpt_function(prompt)
        normalization = self._safe_json_parse(result)

        def _is_valid_groups(parsed):
            if not isinstance(parsed, dict):
                return False
            for category in ("machine_groups", "operation_groups", "component_type_groups",
                             "param_key_groups", "param_value_groups"):
                groups = parsed.get(category, [])
                if not isinstance(groups, list):
                    return False
                for group in groups:
                    if not isinstance(group, dict):
                        return False
            return True

        def _call_until_valid(call_prompt, label, max_retries=3):
            for attempt in range(max_retries):
                raw = self._chatgpt_function(call_prompt)
                parsed = self._safe_json_parse(raw)
                if _is_valid_groups(parsed):
                    return parsed
                print(f"  [s4] {label} returned invalid shape on attempt {attempt + 1}, retrying...",
                      flush=True)
            print(f"  [s4] {label} still invalid after {max_retries} attempts; using empty groups",
                  flush=True)
            return {}

        if not _is_valid_groups(normalization):
            print(f"  [s4] normalize returned invalid shape on attempt 1, retrying...", flush=True)
            normalization = _call_until_valid(prompt, "normalize", max_retries=2)

        def _flatten_groups(groups_dict):
            flat = {}
            for category in ("machine_groups", "operation_groups", "component_type_groups",
                             "param_key_groups", "param_value_groups"):
                for group in groups_dict.get(category, []):
                    canonical = group.get("canonical", "")
                    for member in group.get("members", []):
                        if member != canonical:
                            flat[member] = canonical
            return flat

        mappings = _flatten_groups(normalization)

        write_json(os.path.join(self.pipeline_dir, "s4_mappings_pre.json"), mappings)

        pre_sets_by_category = {
            "machines": sorted(machines),
            "operations": sorted(operations),
            "component_types": sorted(component_types),
            "param_keys": sorted(param_keys),
            "param_values": sorted(param_values),
        }

        def _apply_set(values, m):
            return sorted({m.get(v, v) for v in values})

        def _merge_groups(values, m):
            groups = {}
            for v in values:
                groups.setdefault(m.get(v, v), []).append(v)
            return {c: sorted(ms) for c, ms in groups.items() if len(ms) > 1}

        diff_by_category = {
            cat: {
                "pre_set": pre_values,
                "post_set": _apply_set(pre_values, mappings),
                "merges": _merge_groups(pre_values, mappings),
            }
            for cat, pre_values in pre_sets_by_category.items()
        }

        verify_prompt = self.s4_verify_prompt.replace(
            "---DIFF---", json.dumps(diff_by_category, indent=2, ensure_ascii=False)
        )
        verified_normalization = _call_until_valid(verify_prompt, "verify", max_retries=3)

        if verified_normalization and any(
            verified_normalization.get(k) for k in
            ("machine_groups", "operation_groups", "component_type_groups",
             "param_key_groups", "param_value_groups")
        ):
            mappings = _flatten_groups(verified_normalization)

        self.normalized_jsons = []
        for data in self.formatted_jsons:
            if "bad_case" in data:
                self.normalized_jsons.append(data)
                continue
            self.normalized_jsons.append(self._apply_normalization(data, mappings))

        write_json(os.path.join(self.pipeline_dir, "s4_normalized.json"), self.normalized_jsons)
        write_json(os.path.join(self.pipeline_dir, "s4_mappings.json"), mappings)

    def _apply_normalization(self, data, mappings):
        result = {"steps": []}
        for step in data.get("steps", []):
            new_step = {
                "operation": mappings.get(step.get("operation", ""), step.get("operation", "")),
                "machine": mappings.get(step.get("machine", ""), step.get("machine", "")),
                "duration": step.get("duration", ""),
                "precondition": [],
                "postcondition": [],
                "parameters": {},
            }
            for field in ("precondition", "postcondition"):
                for item in step.get(field, []):
                    if isinstance(item, dict):
                        new_item = dict(item)
                        ct = new_item.get("component_type", "")
                        new_item["component_type"] = mappings.get(ct, ct)
                        new_step[field].append(new_item)
                    else:
                        new_step[field].append(item)
            for k, v in step.get("parameters", {}).items():
                new_k = mappings.get(k, k)
                new_v = mappings.get(v, v) if isinstance(v, str) else v
                new_step["parameters"][new_k] = new_v
            result["steps"].append(new_step)
        return result

    # ------------------------------------------------------------------ #
    #  s5: Graph                                                         #
    # ------------------------------------------------------------------ #

    def build_graph(self):
        print("s5 graph: building disjunctive graph (linear precedence) ...", flush=True)
        self.or_matrix = []
        self.compile_error_num = 0
        self.total_num = 0

        machine_set = set()
        for data in self.normalized_jsons:
            if "bad_case" in data:
                continue
            for step in data.get("steps", []):
                m = step.get("machine", "")
                if m:
                    machine_set.add(str(m).lower())
        self.machines = sorted(machine_set)

        for data in self.normalized_jsons:
            if "bad_case" in data:
                self.or_matrix.append([])
                continue
            steps = data.get("steps", [])
            row = self._derive_precedence_linear(steps)
            self.or_matrix.append(row)

        write_json(os.path.join(self.pipeline_dir, "s5_or_matrix.json"), self.or_matrix)
        write_json(os.path.join(self.pipeline_dir, "s5_machines.json"), self.machines)

    def _derive_precedence_linear(self, steps):
        row = []
        for i, step in enumerate(steps):
            self.total_num += 1
            machine_idx = self._get_machine_idx(step)
            duration = self._parse_duration(step.get("duration", "0"))
            pre_indexes = [i - 1] if i > 0 else []
            row.append([machine_idx, duration, pre_indexes])
        return row

    def _get_machine_idx(self, step):
        machine_name = str(step.get("machine", "")).lower()
        try:
            return self.machines.index(machine_name)
        except ValueError:
            self.compile_error_num += 1
            return 0

    # ------------------------------------------------------------------ #
    #  s6: Solve                                                         #
    # ------------------------------------------------------------------ #

    def solve(self):
        print("s6 solve: OR-Tools CP-SAT scheduling ...", flush=True)
        or_matrix = copy.deepcopy(self.or_matrix)
        assigned_jobs, solver, err_rate, makespan = schedule(or_matrix)
        if len(assigned_jobs) == 0:
            print("  No solution found.")
            self.compile_error_num += 1
        else:
            self.assigned_jobs = assigned_jobs
        write_json(os.path.join(self.pipeline_dir, "s6_assigned_jobs.json"), self.assigned_jobs)
        write_txt(os.path.join(self.pipeline_dir, "s6_err_rate.txt"), str(err_rate))
        write_txt(os.path.join(self.pipeline_dir, "s6_makespan.txt"), str(makespan))
        write_txt(os.path.join(self.pipeline_dir, "makespan.txt"), str(makespan))

    # ------------------------------------------------------------------ #
    #  s7: Ground                                                        #
    # ------------------------------------------------------------------ #

    def ground(self):
        print("s7 ground: reprojecting schedule to production plan ...", flush=True)
        production_plan = []

        for machine_index_str, production_sequence in self.assigned_jobs.items():
            machine_index = int(machine_index_str)
            if machine_index < len(self.machines):
                machine_name = self.machines[machine_index]
            else:
                machine_name = f"machine_{machine_index}_unknown"

            machine_plan = {"machine": machine_name, "production_sequence": []}
            for step in production_sequence:
                start_time, job_id, task_id, duration = step[0], step[1], step[2], step[3]
                step_info = {
                    "start": start_time,
                    "end": start_time + duration,
                    "job_id": job_id,
                    "task_id": task_id,
                }
                if job_id < len(self.normalized_jsons):
                    data = self.normalized_jsons[job_id]
                    if "bad_case" not in data:
                        steps = data.get("steps", [])
                        if task_id < len(steps):
                            src_step = steps[task_id]
                            step_info["operation"] = src_step.get("operation", "")
                            step_info["machine"] = src_step.get("machine", "")
                            step_info["duration"] = src_step.get("duration", "")
                            step_info["precondition"] = [
                                item.get("component_type", "")
                                for item in src_step.get("precondition", [])
                                if isinstance(item, dict)
                            ]
                            step_info["postcondition"] = [
                                item.get("component_type", "")
                                for item in src_step.get("postcondition", [])
                                if isinstance(item, dict)
                            ]
                            step_info["parameters"] = src_step.get("parameters", {})
                machine_plan["production_sequence"].append(step_info)
            production_plan.append(machine_plan)

        self.production_plan = production_plan
        write_json(os.path.join(self.pipeline_dir, "s7_production_plan.json"), self.production_plan)
        write_txt(os.path.join(self.pipeline_dir, "s7_compile_errors.txt"), str(self.compile_error_num))

    # ------------------------------------------------------------------ #
    #  Helpers                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_route_sheets(normalized_jsons, instance_description=""):
        """Convert normalized JSONs to route_sheet format (for evaluation)."""
        route_sheets = []
        for data in normalized_jsons:
            if "bad_case" in data:
                route_sheets.append({"instance_description": instance_description, "route_sheet": []})
                continue
            route_sheets.append({
                "instance_description": instance_description,
                "route_sheet": data.get("steps", []),
            })
        return route_sheets

    @staticmethod
    def _parse_duration(duration_str):
        text = str(duration_str).lower().strip()
        match = re.search(r'([\d.]+)', text)
        if not match:
            return 0
        value = float(match.group(1))
        if 'hour' in text or 'hr' in text:
            return int(value * 60)
        if 'second' in text or 'sec' in text:
            return max(1, int(value / 60))
        return int(value)

    def _load_intermediates(self):
        if self.force:
            return
        d = self.pipeline_dir
        if os.path.exists(os.path.join(d, "s1_extracted.json")):
            self.extracted_jsons = read_json(os.path.join(d, "s1_extracted.json"))
        if os.path.exists(os.path.join(d, "s2_verified.json")):
            self.verified_jsons = read_json(os.path.join(d, "s2_verified.json"))
        if os.path.exists(os.path.join(d, "s3_formatted.json")):
            self.formatted_jsons = read_json(os.path.join(d, "s3_formatted.json"))
        if os.path.exists(os.path.join(d, "s4_normalized.json")):
            self.normalized_jsons = read_json(os.path.join(d, "s4_normalized.json"))
        if os.path.exists(os.path.join(d, "s5_or_matrix.json")):
            self.or_matrix = read_json(os.path.join(d, "s5_or_matrix.json"))
        if os.path.exists(os.path.join(d, "s5_machines.json")):
            self.machines = read_json(os.path.join(d, "s5_machines.json"))
        if os.path.exists(os.path.join(d, "s6_assigned_jobs.json")):
            self.assigned_jobs = read_json(os.path.join(d, "s6_assigned_jobs.json"))

    def _parallel_llm_calls(self, prompts, max_workers=None):
        if max_workers is None:
            max_workers = int(os.environ.get("LLM_MAX_WORKERS", "64"))
        results = [None] * len(prompts)

        def call_with_index(idx, prompt):
            return idx, self._chatgpt_function(prompt)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(call_with_index, i, p) for i, p in enumerate(prompts)]
            for future in tqdm(as_completed(futures), total=len(futures), desc="LLM calls"):
                idx, result = future.result()
                results[idx] = result
        return results

    def _chatgpt_function(self, content, model=None, max_retries=3):
        if model is None:
            model = LLM_MODEL
        client = make_chat_client()
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are an expert in structured data analysis and process scheduling."},
                        {"role": "user", "content": content},
                    ],
                    model=model,
                    max_tokens=16384,
                )
                result = resp.choices[0].message.content or ""
                if result.strip():
                    return result
                print(f"Empty response (attempt {attempt+1}/{max_retries})", flush=True)
            except Exception as e:
                print(f"API error (attempt {attempt+1}/{max_retries}): {e}", flush=True)
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                continue
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
        print(f"Max retries reached, returning empty response", flush=True)
        return "{}"

    def _safe_json_parse(self, text):
        if not text:
            return {}
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            start = text.find("[")
            end = text.rfind("]") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            print(f"Failed to parse JSON: {text[:200]}")
            return {}


# --------------------------------------------------------------------- #
#  CLI                                                                  #
# --------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--instances", nargs="+", default=None,
                    help="Instance short-names (ta71 ..) or full ('instance ta71'). Default: all 10.")
    ap.add_argument("--experiment-type", choices=EXPERIMENT_TYPES, default="full",
                    help="Which pipeline variant to run (default: full).")
    ap.add_argument("--preprocess-dir", default=os.path.join(_HERE, "preprocess_out"),
                    help="Where orders.json + jssp_mapped.json + route_sheet_reduce.json live.")
    ap.add_argument("--artifacts-dir", default=os.path.join(_HERE, "artifacts"),
                    help="Where pipeline outputs and groundtruth are written.")
    ap.add_argument("--prompts-dir", default=os.path.join(_HERE, "prompts"),
                    help="Where s1/s2/s4 prompt templates live.")
    ap.add_argument("--data-dir", default=os.path.join(_HERE, "data"),
                    help="Where raw jssp_data / arrange / machines JSONs live.")
    ap.add_argument("--force", action="store_true",
                    help="Force re-run of every stage even if outputs already exist.")
    ap.add_argument("--no-force", dest="force", action="store_false",
                    help="Resume from existing intermediate files where possible.")
    ap.set_defaults(force=True)
    args = ap.parse_args()

    insts = args.instances or ALL_INSTANCES

    for inst in insts:
        print(f"\n=== SchemaPipeline: {inst} ({args.experiment_type}) ===", flush=True)
        pipe = SchemaPipeline(
            instance_description=inst,
            preprocess_dir=args.preprocess_dir,
            artifacts_dir=args.artifacts_dir,
            prompts_dir=args.prompts_dir,
            data_dir=args.data_dir,
            experiment_type=args.experiment_type,
            force=args.force,
        )
        pipe.run()


if __name__ == "__main__":
    main()
