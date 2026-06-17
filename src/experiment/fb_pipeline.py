from __future__ import annotations
import copy
import os
import re
import time
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from utils.util import read_json, write_json, read_txt, write_txt
from src.experiment.schedule import schedule
from src.experiment.groundtruth import GroundTruth
from src.experiment.schema_validator import validate_extraction


class FBPipeline:
    def __init__(self, instance_description: str, experiment_type: str = "CPE_CAE_CSE-2", force: bool = True):
        self.instance_description = instance_description
        self.experiment_type = experiment_type
        self.force = force
        self.dump_dir_path = ""

        self.orders = []
        self.extracted_jsons = []      # Step 1 output
        self.verified_jsons = []       # Step 1-verify output
        self.formatted_jsons = []      # Step 2 output
        self.format_verified_jsons = []  # Step 2-verify output
        self.normalized_jsons = []     # Step 3 output (final)
        self.or_matrix = []
        self.machines = []
        self.assigned_jobs = {}
        self.production_plan = []

        self.compile_error_num = 0
        self.total_num = 0

        # Load prompts
        self.step1_extract_prompt = read_txt("src/prompts/step1_extract.txt")
        self.step1_verify_prompt = read_txt("src/prompts/step1_verify.txt")
        self.step2_format_prompt = read_txt("src/prompts/step2_format.txt")
        self.step2_verify_prompt = read_txt("src/prompts/step2_verify.txt")
        self.step3_normalize_prompt = read_txt("src/prompts/step3_normalize.txt")
        self.step3_verify_prompt = read_txt("src/prompts/step3_verify.txt")

        self.groundtruth = None

    def run(self):
        self.dump_dir_path = f"outputs/FB-{self.experiment_type}/{self.instance_description}/"
        self.groundtruth = GroundTruth(self.instance_description)
        self._load_data()

        if self.experiment_type == "CPE_CAE_CSE-2":
            if len(self.orders) == 0:
                raise RuntimeError(
                    f"Missing orders at {self.dump_dir_path}orders.json. "
                    "Please provide NL orders before running the pipeline."
                )
            # ---- Step 1: Extract ----
            if len(self.extracted_jsons) == 0:
                self.extract_all()
            # ---- Step 1-verify ----
            if len(self.verified_jsons) == 0:
                self.verify_extract()
            # ---- Step 2: Format ----
            if len(self.formatted_jsons) == 0:
                self.format_all()
            # ---- Step 2-verify ----
            if len(self.format_verified_jsons) == 0:
                self.verify_format()
            # ---- Step 3: Normalize ----
            if len(self.normalized_jsons) == 0:
                self.normalize_all()
            # ---- Step 4-6: Build OR Matrix + Solve + Ground ----
            self.build_or_matrix()
            self.solve_jsp()
            self.ground_production_plan()

        elif self.experiment_type == "CSE-1":
            route_sheets = read_json(
                f"outputs/GroundTruth/{self.instance_description}/route_sheets.json"
            )
            self.normalized_jsons = [self._route_sheet_to_fixed_schema(rs) for rs in route_sheets]
            self.build_or_matrix()
            self.solve_jsp()
            self.ground_production_plan()

        elif self.experiment_type == "SGE":
            self.assigned_jobs = read_json(
                f"outputs/GroundTruth/{self.instance_description}/assigned_jobs.json"
            )
            route_sheets = read_json(
                f"outputs/GroundTruth/{self.instance_description}/route_sheets.json"
            )
            self.normalized_jsons = [self._route_sheet_to_fixed_schema(rs) for rs in route_sheets]
            self.ground_production_plan()

    # ------------------------------------------------------------------ #
    #  Step 1: Extract information (fixed schema)                         #
    # ------------------------------------------------------------------ #

    def extract_all(self):
        """Extract structured JSON from each order using fixed schema."""
        print("Step 1: Extracting information ...", flush=True)
        self.extracted_jsons = []

        prompts = []
        for order in self.orders:
            prompt = self.step1_extract_prompt.replace("---ORDER---", json.dumps(order))
            prompts.append(prompt)

        results = self._parallel_llm_calls(prompts)

        for i, result in enumerate(results):
            parsed = self._safe_json_parse(result)
            is_valid, err = validate_extraction(parsed)
            if is_valid:
                self.extracted_jsons.append(parsed)
            else:
                self.extracted_jsons.append({"steps": [], "bad_case": f"Step 1: {err}"})

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

        write_json(self.dump_dir_path + "step1_extracted.json", self.extracted_jsons)

    # ------------------------------------------------------------------ #
    #  Step 1-verify: Verify factual accuracy                             #
    # ------------------------------------------------------------------ #

    def verify_extract(self):
        """Verify each extraction against its NL source."""
        print("Step 1-verify: Verifying extractions ...", flush=True)

        prompts = []
        for i, (order, extracted) in enumerate(zip(self.orders, self.extracted_jsons)):
            if "bad_case" in extracted:
                prompts.append(None)
                continue
            prompt = self.step1_verify_prompt \
                .replace("---ORDER---", json.dumps(order)) \
                .replace("---EXTRACTED---", json.dumps(extracted))
            prompts.append(prompt)

        # Only call LLM for valid entries
        valid_indices = [i for i, p in enumerate(prompts) if p is not None]
        valid_prompts = [prompts[i] for i in valid_indices]
        results = self._parallel_llm_calls(valid_prompts)

        self.verified_jsons = list(self.extracted_jsons)  # copy
        for j, idx in enumerate(valid_indices):
            parsed = self._safe_json_parse(results[j])
            is_valid, err = validate_extraction(parsed)
            if is_valid:
                self.verified_jsons[idx] = parsed
            else:
                # Retry once
                result = self._chatgpt_function(valid_prompts[j])
                parsed = self._safe_json_parse(result)
                is_valid, err = validate_extraction(parsed)
                if is_valid:
                    self.verified_jsons[idx] = parsed
                else:
                    self.verified_jsons[idx] = {"steps": [], "bad_case": f"Step 1-verify: {err}"}

        write_json(self.dump_dir_path + "step1_verified.json", self.verified_jsons)

    # ------------------------------------------------------------------ #
    #  Step 2: Format (Title Case, PascalCase, etc.)                      #
    # ------------------------------------------------------------------ #

    def format_all(self):
        """Adjust formatting of all verified JSONs."""
        print("Step 2: Formatting ...", flush=True)

        prompts = []
        for i, data in enumerate(self.verified_jsons):
            if "bad_case" in data:
                prompts.append(None)
                continue
            prompt = self.step2_format_prompt.replace("---INPUT---", json.dumps(data))
            prompts.append(prompt)

        valid_indices = [i for i, p in enumerate(prompts) if p is not None]
        valid_prompts = [prompts[i] for i in valid_indices]
        results = self._parallel_llm_calls(valid_prompts)

        self.formatted_jsons = list(self.verified_jsons)  # copy
        for j, idx in enumerate(valid_indices):
            parsed = self._safe_json_parse(results[j])
            is_valid, err = validate_extraction(parsed)
            if is_valid:
                self.formatted_jsons[idx] = parsed
            else:
                # Retry once
                result = self._chatgpt_function(valid_prompts[j])
                parsed = self._safe_json_parse(result)
                is_valid, err = validate_extraction(parsed)
                if is_valid:
                    self.formatted_jsons[idx] = parsed
                else:
                    self.formatted_jsons[idx] = {"steps": [], "bad_case": f"Step 2: {err}"}

        write_json(self.dump_dir_path + "step2_formatted.json", self.formatted_jsons)

    # ------------------------------------------------------------------ #
    #  Step 2-verify: Verify formatting                                   #
    # ------------------------------------------------------------------ #

    def verify_format(self):
        """LLM check that formatting rules are satisfied."""
        print("Step 2-verify: Verifying format ...", flush=True)

        prompts = []
        for i, data in enumerate(self.formatted_jsons):
            if "bad_case" in data:
                prompts.append(None)
                continue
            prompt = self.step2_verify_prompt.replace("---INPUT---", json.dumps(data))
            prompts.append(prompt)

        valid_indices = [i for i, p in enumerate(prompts) if p is not None]
        valid_prompts = [prompts[i] for i in valid_indices]
        results = self._parallel_llm_calls(valid_prompts)

        self.format_verified_jsons = list(self.formatted_jsons)  # copy
        for j, idx in enumerate(valid_indices):
            parsed = self._safe_json_parse(results[j])
            is_valid, err = validate_extraction(parsed)
            if is_valid:
                self.format_verified_jsons[idx] = parsed
            else:
                # Keep the formatted version if verify fails
                self.format_verified_jsons[idx] = self.formatted_jsons[idx]

        write_json(self.dump_dir_path + "step2_verified.json", self.format_verified_jsons)

    # ------------------------------------------------------------------ #
    #  Step 3: Semantic normalization (synonym merging)                    #
    # ------------------------------------------------------------------ #

    def normalize_all(self):
        """Discover and apply semantic normalization across all valid data."""
        print("Step 3: Normalizing ...", flush=True)

        # Collect all unique values by category
        machines = set()
        operations = set()
        component_types = set()
        param_keys = set()
        param_values = set()

        valid_data = [d for d in self.format_verified_jsons if "bad_case" not in d]

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
        prompt = self.step3_normalize_prompt \
            .replace("---MACHINES---", json.dumps(sorted(machines))) \
            .replace("---OPERATIONS---", json.dumps(sorted(operations))) \
            .replace("---COMPONENT_TYPES---", json.dumps(sorted(component_types))) \
            .replace("---PARAM_KEYS---", json.dumps(sorted(param_keys))) \
            .replace("---PARAM_VALUES---", json.dumps(sorted(param_values)))

        print(f"  Unique values: {len(machines)} machines, {len(operations)} operations, "
              f"{len(component_types)} types, {len(param_keys)} param keys", flush=True)

        result = self._chatgpt_function(prompt)
        normalization = self._safe_json_parse(result)

        # Build flat mappings from groups
        mappings = {}
        for category in ("machine_groups", "operation_groups", "component_type_groups",
                         "param_key_groups", "param_value_groups"):
            groups = normalization.get(category, [])
            for group in groups:
                canonical = group.get("canonical", "")
                for member in group.get("members", []):
                    if member != canonical:
                        mappings[member] = canonical

        # Verify the mapping
        samples = valid_data[:3] if len(valid_data) >= 3 else valid_data
        verify_prompt = self.step3_verify_prompt \
            .replace("---MAPPING---", json.dumps(normalization, indent=2)) \
            .replace("---SAMPLES---", json.dumps(samples, indent=2))
        verify_result = self._chatgpt_function(verify_prompt)
        verified_normalization = self._safe_json_parse(verify_result)

        # Rebuild mappings from verified result if valid
        if verified_normalization and any(
            verified_normalization.get(k) for k in
            ("machine_groups", "operation_groups", "component_type_groups",
             "param_key_groups", "param_value_groups")
        ):
            mappings = {}
            for category in ("machine_groups", "operation_groups", "component_type_groups",
                             "param_key_groups", "param_value_groups"):
                groups = verified_normalization.get(category, [])
                for group in groups:
                    canonical = group.get("canonical", "")
                    for member in group.get("members", []):
                        if member != canonical:
                            mappings[member] = canonical

        # Apply mappings to all data
        self.normalized_jsons = []
        for data in self.format_verified_jsons:
            if "bad_case" in data:
                self.normalized_jsons.append(data)
                continue
            normalized = self._apply_normalization(data, mappings)
            self.normalized_jsons.append(normalized)

        write_json(self.dump_dir_path + "step3_normalized.json", self.normalized_jsons)
        write_json(self.dump_dir_path + "step3_mappings.json", mappings)

        # Also save as CAM-3_normalized_jsons.json for evaluation compatibility
        write_json(self.dump_dir_path + "CAM-3_normalized_jsons.json", self.normalized_jsons)

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
    #  Step 4: Build OR Matrix (deterministic)                            #
    # ------------------------------------------------------------------ #

    def build_or_matrix(self):
        """Derive OR matrix from normalized JSONs using fixed schema fields."""
        print("Step 4: Building OR matrix ...", flush=True)
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
            row = []

            for i, step in enumerate(steps):
                self.total_num += 1
                machine_name = str(step.get("machine", "")).lower()
                try:
                    machine_idx = self.machines.index(machine_name)
                except ValueError:
                    machine_idx = 0
                    self.compile_error_num += 1

                duration = self._parse_duration(step.get("duration", "0"))

                # Determine precedence via component_type matching
                current_input_types = set()
                for item in step.get("precondition", []):
                    if isinstance(item, dict):
                        ct = item.get("component_type", "")
                        if ct:
                            current_input_types.add(ct.lower())

                pre_indexes = []
                for j in range(i):
                    prev_step = steps[j]
                    prev_output_types = set()
                    for item in prev_step.get("postcondition", []):
                        if isinstance(item, dict):
                            ct = item.get("component_type", "")
                            if ct:
                                prev_output_types.add(ct.lower())
                    if current_input_types & prev_output_types:
                        pre_indexes.append(j)

                row.append([machine_idx, duration, pre_indexes])
            self.or_matrix.append(row)

        write_json(self.dump_dir_path + "CGM_or_matrix.json", self.or_matrix)
        write_json(self.dump_dir_path + "CGM_machines.json", self.machines)

    # ------------------------------------------------------------------ #
    #  Step 5: Solve JSP (OR-Tools)                                       #
    # ------------------------------------------------------------------ #

    def solve_jsp(self):
        """Run OR-Tools JSP solver."""
        print("Step 5: Solving JSP ...", flush=True)
        or_matrix = copy.deepcopy(self.or_matrix)
        assigned_jobs, solver, err_rate, makespan = schedule(or_matrix)
        if len(assigned_jobs) == 0:
            print("  No solution found.")
            self.compile_error_num += 1
        else:
            self.assigned_jobs = assigned_jobs
        write_json(self.dump_dir_path + "assigned_jobs.json", self.assigned_jobs)
        write_txt(self.dump_dir_path + "err_rate.txt", str(err_rate))
        write_txt(self.dump_dir_path + "makespan.txt", str(makespan))

    # ------------------------------------------------------------------ #
    #  Step 6: Ground production plan                                     #
    # ------------------------------------------------------------------ #

    def ground_production_plan(self):
        """Map solver output back to step details."""
        print("Step 6: Grounding production plan ...", flush=True)
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
        write_json(self.dump_dir_path + "SGM_production_plan.json", self.production_plan)
        write_txt(self.dump_dir_path + "compile_error_num.txt", str(self.compile_error_num))

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
        # Always load input orders
        if os.path.exists(self.dump_dir_path + "orders.json"):
            self.orders = read_json(self.dump_dir_path + "orders.json")
        # When force=True, skip loading intermediate files so all steps re-run
        if self.force:
            return
        if os.path.exists(self.dump_dir_path + "step1_extracted.json"):
            self.extracted_jsons = read_json(self.dump_dir_path + "step1_extracted.json")
        if os.path.exists(self.dump_dir_path + "step1_verified.json"):
            self.verified_jsons = read_json(self.dump_dir_path + "step1_verified.json")
        if os.path.exists(self.dump_dir_path + "step2_formatted.json"):
            self.formatted_jsons = read_json(self.dump_dir_path + "step2_formatted.json")
        if os.path.exists(self.dump_dir_path + "step2_verified.json"):
            self.format_verified_jsons = read_json(self.dump_dir_path + "step2_verified.json")
        if os.path.exists(self.dump_dir_path + "step3_normalized.json"):
            self.normalized_jsons = read_json(self.dump_dir_path + "step3_normalized.json")
        if os.path.exists(self.dump_dir_path + "CGM_or_matrix.json"):
            self.or_matrix = read_json(self.dump_dir_path + "CGM_or_matrix.json")
        if os.path.exists(self.dump_dir_path + "CGM_machines.json"):
            self.machines = read_json(self.dump_dir_path + "CGM_machines.json")
        if os.path.exists(self.dump_dir_path + "assigned_jobs.json"):
            self.assigned_jobs = read_json(self.dump_dir_path + "assigned_jobs.json")

    # ------------------------------------------------------------------ #
    #  LLM utilities                                                      #
    # ------------------------------------------------------------------ #

    def _parallel_llm_calls(self, prompts, max_workers=1):
        """Execute LLM calls in parallel, preserving order."""
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

    def _chatgpt_function(self, content, model="gpt-4o", max_retries=3):
        client = OpenAI(
            base_url="http://localhost:4141/v1",
            api_key=os.environ.get("OPENAI_API_KEY", "sk-placeholder"),
            timeout=300.0,
        )
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
                err_str = str(e).lower()
                if "rate" in err_str or "429" in err_str or "too many" in err_str:
                    wait = 60 * (attempt + 1)
                    print(f"Rate limited (attempt {attempt+1}/{max_retries}), sleeping {wait}s ...", flush=True)
                    time.sleep(wait)
                else:
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
