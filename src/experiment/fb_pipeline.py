from __future__ import annotations
import copy
import os
import re
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from utils.util import read_json, write_json, read_txt, write_txt, make_chat_client, LLM_MODEL
from src.experiment.schedule import schedule
from src.experiment.groundtruth import GroundTruth
from src.experiment.schema_validator import validate_extraction
from src.experiment.format_rules import format_data as _code_format_data


# Pipeline step names (see MIGRATION.md for old→new mapping):
#   s1 extract    — NL order → fixed-schema JSON             (LLM, per-order)
#   s2 verify     — factual verification vs NL                (LLM, per-order)
#   s3 format     — deterministic Title/Pascal/lower casing   (code)
#   s4 normalize  — global synonym merging across all orders  (LLM, single call + verify)
#   s5 graph      — build OR matrix / disjunctive graph       (code)
#   s6 solve      — JSP CP-SAT scheduling                     (OR-Tools)
#   s7 ground     — reproject schedule back to production plan (code)


class FBPipeline:
    def __init__(self, instance_description: str, experiment_type: str = "full", force: bool = True):
        self.instance_description = instance_description
        self.experiment_type = experiment_type
        self.force = force
        self.dump_dir_path = ""

        self.orders = []
        self.extracted_jsons = []        # s1 output
        self.verified_jsons = []         # s2 output
        self.formatted_jsons = []        # s3 output
        self.normalized_jsons = []       # s4 output (final structured JSON)
        self.or_matrix = []              # s5 output
        self.machines = []               # s5 output
        self.assigned_jobs = {}          # s6 output
        self.production_plan = []        # s7 output

        self.compile_error_num = 0
        self.total_num = 0

        # Load prompts (prompt filenames unchanged — they're data assets)
        self.s1_extract_prompt = read_txt("src/prompts/step1_extract.txt")
        self.s2_verify_prompt = read_txt("src/prompts/step1_verify.txt")
        self.s4_normalize_prompt = read_txt("src/prompts/step3_normalize.txt")
        self.s4_verify_prompt = read_txt("src/prompts/step3_verify.txt")

        self.groundtruth = None

    def run(self):
        output_prefix = os.environ.get("FB_OUTPUT_PREFIX", f"FB-2s-{self.experiment_type}")
        self.dump_dir_path = f"outputs/{output_prefix}/{self.instance_description}/"
        self.groundtruth = GroundTruth(self.instance_description)
        self._load_data()

        if self.experiment_type == "full":
            if len(self.orders) == 0:
                raise RuntimeError(
                    f"Missing orders at preprocess/orders/{self.instance_description.replace('instance ', '')}/orders.json"
                )

            # ---- s1: Extract ----
            if len(self.extracted_jsons) == 0:
                self.extract()
            # ---- s2: Verify (factual vs NL) ----
            if len(self.verified_jsons) == 0:
                self.verify()
            # ---- s3: Format (code) ----
            if len(self.formatted_jsons) == 0:
                self.apply_format()
            # ---- s4: Normalize ----
            if len(self.normalized_jsons) == 0:
                self.normalize()
            # ---- s5..s7: Graph + Solve + Ground ----
            self.build_graph()
            self.solve()
            self.ground()

        elif self.experiment_type == "from_gt_route_sheet":
            route_sheets = read_json(
                f"outputs/GroundTruth/{self.instance_description}/route_sheets.json"
            )
            self.normalized_jsons = [self._route_sheet_to_fixed_schema(rs) for rs in route_sheets]
            self.build_graph()
            self.solve()
            self.ground()

        elif self.experiment_type == "from_gt_schedule":
            self.assigned_jobs = read_json(
                f"outputs/GroundTruth/{self.instance_description}/assigned_jobs.json"
            )
            route_sheets = read_json(
                f"outputs/GroundTruth/{self.instance_description}/route_sheets.json"
            )
            self.normalized_jsons = [self._route_sheet_to_fixed_schema(rs) for rs in route_sheets]
            self.ground()

    # ------------------------------------------------------------------ #
    #  s1: Extract — NL → fixed-schema JSON (LLM, per-order parallel)     #
    # ------------------------------------------------------------------ #

    def extract(self):
        """s1: extract structured JSON AND apply format rules in one LLM call."""
        print("s1 extract: NL → fixed-schema JSON ...", flush=True)
        self.extracted_jsons = []

        prompts = []
        for order in self.orders:
            prompt = self.s1_extract_prompt.replace("---ORDER---", json.dumps(order))
            prompts.append(prompt)

        results = self._parallel_llm_calls(prompts)

        for i, result in enumerate(results):
            parsed = self._safe_json_parse(result)
            is_valid, err = validate_extraction(parsed)
            if is_valid:
                self.extracted_jsons.append(parsed)
            else:
                self.extracted_jsons.append({"steps": [], "bad_case": f"s1 extract: {err}"})

        # Retry bad cases once
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

        write_json(self.dump_dir_path + "s1_extracted.json", self.extracted_jsons)

    # ------------------------------------------------------------------ #
    #  s2: Verify — factual correctness vs NL (LLM)                       #
    # ------------------------------------------------------------------ #

    def verify(self):
        """s2: per-step verify against the matching NL sentence.

        One LLM call per (order, step) pair. The output for each step is a
        single-step dict; we assemble the per-order `{"steps": [...]}` in
        code, so:

        - The output step count for each order EQUALS the input s1 step count
          by construction — no more "s2 dropped to []" or "s2 doubled steps"
          class of failures.
        - Each LLM call has minimal context (one NL sentence + one JSON step),
          so it can't confuse itself by re-emitting neighbors.
        - If a single step's LLM call fails schema after one retry, we fall
          back to the s1 version of THAT step (not the whole order).
        """
        print("s2 verify: LLM per-step factual verification vs NL ...", flush=True)

        # Flatten to per-step tasks. Each task is (order_idx, step_idx, prompt).
        tasks = []  # list of (i, j, prompt)
        for i, (order, extracted) in enumerate(zip(self.orders, self.extracted_jsons)):
            if "bad_case" in extracted:
                continue
            nl_steps = order.get("steps", []) if isinstance(order, dict) else []
            for j, s1_step in enumerate(extracted.get("steps", [])):
                if j >= len(nl_steps):
                    continue  # no NL for this step; will fall back to s1
                prompt = self.s2_verify_prompt \
                    .replace("---STEP_NL---", json.dumps(nl_steps[j])) \
                    .replace("---STEP_JSON---", json.dumps(s1_step))
                tasks.append((i, j, prompt))

        prompts = [t[2] for t in tasks]
        results = self._parallel_llm_calls(prompts) if prompts else []

        # Start from s1 verbatim; overwrite per-step with verified step when accepted.
        self.verified_jsons = [copy.deepcopy(x) for x in self.extracted_jsons]

        s2_accept = s2_fallback = 0
        for (i, j, prompt), raw in zip(tasks, results):
            step = self._parse_single_step(raw)
            if step is None:
                # Retry once
                raw2 = self._chatgpt_function(prompt)
                step = self._parse_single_step(raw2)
            if step is not None:
                self.verified_jsons[i]["steps"][j] = step
                s2_accept += 1
            else:
                # Fallback: leave s1's step in place (already there via deepcopy)
                s2_fallback += 1

        total = s2_accept + s2_fallback
        print(f"  s2 accepted: {s2_accept}/{total}  fell back to s1 step: {s2_fallback}",
              flush=True)
        write_json(self.dump_dir_path + "s2_verified.json", self.verified_jsons)

    @staticmethod
    def _single_step_schema_ok(step) -> bool:
        """A step is well-formed iff it looks like s1's per-step schema.

        Same required fields as validate_extraction expects, but at the single
        step level (no {"steps": ...} wrapper).
        """
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
        """Parse LLM raw output as a SINGLE-step JSON object.

        Handles the case where the model still wraps in {"steps": [...]}
        (extract the first element) or {"steps": [step1, step2, ...]} with
        multiple (take only the first — better than nothing).
        Returns the step dict on success, None on failure.
        """
        parsed = self._safe_json_parse(raw)
        if not parsed:
            return None
        # Unwrap if the model returned {"steps": [...]}
        if isinstance(parsed, dict) and "steps" in parsed:
            steps = parsed.get("steps") or []
            if not steps:
                return None
            parsed = steps[0]
        # Some models occasionally return [step] instead of step
        if isinstance(parsed, list):
            if not parsed:
                return None
            parsed = parsed[0]
        return parsed if self._single_step_schema_ok(parsed) else None

    # ------------------------------------------------------------------ #
    #  s3: Format — deterministic code format pass                        #
    # ------------------------------------------------------------------ #

    def apply_format(self):
        """s3: deterministic code-based format pass on factually-verified JSONs.

        Runs AFTER verify() so any factual corrections that drifted the format
        are normalized back. Idempotent; no LLM calls.
        """
        print("s3 format (code): applying Title/Pascal/lower casing ...", flush=True)
        self.formatted_jsons = [_code_format_data(d) for d in self.verified_jsons]
        write_json(self.dump_dir_path + "s3_formatted.json", self.formatted_jsons)

    # ------------------------------------------------------------------ #
    #  s4: Normalize — global synonym merging (LLM, single call + verify) #
    # ------------------------------------------------------------------ #

    def normalize(self):
        """s4: discover and apply semantic normalization across all valid data."""
        print("s4 normalize: global synonym merging ...", flush=True)

        # Collect all unique values by category
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

        # LLM call for normalization
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

        # Validate the LLM output has the expected groups-of-dicts shape.
        # Re-run the call up to N times instead of crashing on bad shape.
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

        # Retry the normalize call if the first attempt produced an invalid shape.
        if not _is_valid_groups(normalization):
            print(f"  [s4] normalize returned invalid shape on attempt 1, retrying...", flush=True)
            normalization = _call_until_valid(prompt, "normalize", max_retries=2)

        # Build flat mappings from groups (shared helper for pre- and post-verify)
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

        # Persist pre-verify intermediate products (mirrors s1..s3 convention)
        write_json(self.dump_dir_path + "s4_mappings_pre.json", mappings)

        # Build diff-based verify input: pre_set / post_set / merges per category
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

        # Verify the mapping by inspecting the pre/post diff (not random samples)
        verify_prompt = self.s4_verify_prompt.replace(
            "---DIFF---", json.dumps(diff_by_category, indent=2, ensure_ascii=False)
        )
        verified_normalization = _call_until_valid(verify_prompt, "verify", max_retries=3)

        # Rebuild mappings from verified result if valid
        if verified_normalization and any(
            verified_normalization.get(k) for k in
            ("machine_groups", "operation_groups", "component_type_groups",
             "param_key_groups", "param_value_groups")
        ):
            mappings = _flatten_groups(verified_normalization)

        # Apply mappings to all data
        self.normalized_jsons = []
        for data in self.formatted_jsons:
            if "bad_case" in data:
                self.normalized_jsons.append(data)
                continue
            normalized = self._apply_normalization(data, mappings)
            self.normalized_jsons.append(normalized)

        write_json(self.dump_dir_path + "s4_normalized.json", self.normalized_jsons)
        write_json(self.dump_dir_path + "s4_mappings.json", mappings)

    def _apply_normalization(self, data, mappings):
        """Apply normalization mappings to a fixed-schema JSON."""
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
    #  s5: Graph — build disjunctive graph / OR matrix (code)             #
    # ------------------------------------------------------------------ #

    def build_graph(self):
        """s5: derive OR matrix from normalized JSONs using fixed schema fields."""
        print("s5 graph: building disjunctive graph (linear precedence) ...", flush=True)
        self.or_matrix = []
        self.compile_error_num = 0
        self.total_num = 0

        # Collect all machines
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

        write_json(self.dump_dir_path + "s5_or_matrix.json", self.or_matrix)
        write_json(self.dump_dir_path + "s5_machines.json", self.machines)

    def _derive_precedence_linear(self, steps):
        """Classical JSP precedence: each step depends only on the previous one.

        The underlying benchmark (Taillard/ABZ/...) defines a job as a totally
        ordered sequence of (machine, duration) tuples — conjunctive arcs in
        the disjunctive graph model. This matches the OR-Tools jobs_data input
        convention and is robust to component_type naming drift introduced
        during LLM extraction.
        """
        row = []
        for i, step in enumerate(steps):
            self.total_num += 1
            machine_idx = self._get_machine_idx(step)
            duration = self._parse_duration(step.get("duration", "0"))
            pre_indexes = [i - 1] if i > 0 else []
            row.append([machine_idx, duration, pre_indexes])
        return row

    def _get_machine_idx(self, step):
        """Get machine index for a step, tracking compile errors."""
        machine_name = str(step.get("machine", "")).lower()
        try:
            return self.machines.index(machine_name)
        except ValueError:
            self.compile_error_num += 1
            return 0

    # ------------------------------------------------------------------ #
    #  s6: Solve — CP-SAT JSP scheduling (OR-Tools)                       #
    # ------------------------------------------------------------------ #

    def solve(self):
        """s6: run OR-Tools JSP solver."""
        print("s6 solve: OR-Tools CP-SAT scheduling ...", flush=True)
        or_matrix = copy.deepcopy(self.or_matrix)
        assigned_jobs, solver, err_rate, makespan = schedule(or_matrix)
        if len(assigned_jobs) == 0:
            print("  No solution found.")
            self.compile_error_num += 1
        else:
            self.assigned_jobs = assigned_jobs
        write_json(self.dump_dir_path + "s6_assigned_jobs.json", self.assigned_jobs)
        write_txt(self.dump_dir_path + "s6_err_rate.txt", str(err_rate))
        write_txt(self.dump_dir_path + "s6_makespan.txt", str(makespan))

    # ------------------------------------------------------------------ #
    #  s7: Ground — map schedule back to production plan                  #
    # ------------------------------------------------------------------ #

    def ground(self):
        """s7: map solver output back to step details."""
        print("s7 ground: reprojecting schedule to production plan ...", flush=True)
        production_plan = []

        for machine_index_str, production_sequence in self.assigned_jobs.items():
            machine_index = int(machine_index_str)
            if machine_index < len(self.machines):
                machine_name = self.machines[machine_index]
            else:
                machine_name = f"machine_{machine_index}_unknown"

            machine_plan = {
                "machine": machine_name,
                "production_sequence": []
            }

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
        write_json(self.dump_dir_path + "s7_production_plan.json", self.production_plan)
        write_txt(self.dump_dir_path + "s7_compile_errors.txt", str(self.compile_error_num))

    # ------------------------------------------------------------------ #
    #  Route sheet builder (static — used by evaluation as adapter)       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_route_sheets(normalized_jsons, instance_description=""):
        """Convert normalized JSONs (fixed schema) into route_sheet format for evaluation.

        Since the fixed schema matches the ground truth field names, this is a direct extraction.
        """
        route_sheets = []
        for data in normalized_jsons:
            if "bad_case" in data:
                route_sheets.append({"instance_description": instance_description, "route_sheet": []})
                continue
            route_sheet_entry = {
                "instance_description": instance_description,
                "route_sheet": data.get("steps", [])
            }
            route_sheets.append(route_sheet_entry)
        return route_sheets

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _route_sheet_to_fixed_schema(self, route_sheet_data):
        """Convert ground truth route_sheet format to fixed schema."""
        return {"steps": route_sheet_data.get("route_sheet", [])}

    def _parse_duration(self, duration_str):
        """Parse duration string to integer minutes."""
        text = str(duration_str).lower().strip()
        match = re.search(r'([\d.]+)', text)
        if not match:
            return 0
        value = float(match.group(1))
        if 'hour' in text or 'hr' in text:
            return int(value * 60)
        if 'second' in text or 'sec' in text:
            return max(1, int(value / 60))
        return int(value)  # default: minutes

    # ------------------------------------------------------------------ #
    #  Data loading                                                       #
    # ------------------------------------------------------------------ #

    def _load_data(self):
        if not os.path.exists(self.dump_dir_path):
            os.makedirs(self.dump_dir_path)
        # Read orders directly from the canonical location — no per-model copy needed.
        # (Historically generate_orders.py fanned orders into each pipeline's dump dir;
        # since that script was removed, we read canonical directly and keep dump dirs
        # for pipeline outputs only.)
        canonical_orders = f"preprocess/orders/{self.instance_description.replace('instance ', '')}/orders.json"
        if os.path.exists(canonical_orders):
            self.orders = read_json(canonical_orders)
        # When force=True, skip loading intermediate files so all steps re-run
        if self.force:
            return
        if os.path.exists(self.dump_dir_path + "s1_extracted.json"):
            self.extracted_jsons = read_json(self.dump_dir_path + "s1_extracted.json")
        if os.path.exists(self.dump_dir_path + "s2_verified.json"):
            self.verified_jsons = read_json(self.dump_dir_path + "s2_verified.json")
        if os.path.exists(self.dump_dir_path + "s3_formatted.json"):
            self.formatted_jsons = read_json(self.dump_dir_path + "s3_formatted.json")
        if os.path.exists(self.dump_dir_path + "s4_normalized.json"):
            self.normalized_jsons = read_json(self.dump_dir_path + "s4_normalized.json")
        if os.path.exists(self.dump_dir_path + "s5_or_matrix.json"):
            self.or_matrix = read_json(self.dump_dir_path + "s5_or_matrix.json")
        if os.path.exists(self.dump_dir_path + "s5_machines.json"):
            self.machines = read_json(self.dump_dir_path + "s5_machines.json")
        if os.path.exists(self.dump_dir_path + "s6_assigned_jobs.json"):
            self.assigned_jobs = read_json(self.dump_dir_path + "s6_assigned_jobs.json")

    # ------------------------------------------------------------------ #
    #  LLM utilities                                                      #
    # ------------------------------------------------------------------ #

    def _parallel_llm_calls(self, prompts, max_workers=None):
        """Execute LLM calls in parallel, preserving order."""
        if max_workers is None:
            max_workers = int(os.environ.get("LLM_MAX_WORKERS", "64"))
        results = [None] * len(prompts)

        def call_with_index(idx, prompt):
            return idx, self._chatgpt_function(prompt)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(call_with_index, i, p)
                for i, p in enumerate(prompts)
            ]
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
        """Parse JSON from LLM output, handling markdown code blocks."""
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
