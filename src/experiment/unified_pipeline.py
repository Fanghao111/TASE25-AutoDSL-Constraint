from __future__ import annotations
import copy
import openai
import os
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from collections import defaultdict, Counter
from utils.util import read_json, write_json, read_txt, write_txt
from src.experiment.schedule import schedule
from src.experiment.groundtruth import GroundTruth


class UnifiedPipeline:
    def __init__(self, instance_description: str, experiment_type: str = "CPE_CAE_CSE-2"):
        self.instance_description = instance_description
        self.experiment_type = experiment_type
        self.dump_dir_path = ""

        self.orders = []
        self.raw_jsons = []          # Step 1 output: free-form extracted JSONs
        self.field_mapping = {}      # discovered field semantic mapping
        self.normalized_jsons = []   # Step 2 output: verified + normalized JSONs
        self.route_sheets = []       # normalized JSONs reformatted as route sheets
        self.or_matrix = []
        self.machines = []
        self.assigned_jobs = {}
        self.production_plan = []

        self.compile_error_num = 0
        self.total_num = 0

        self.free_extract_prompt = read_txt("src/prompts/free_extract.txt")
        self.verify_normalize_prompt = read_txt("src/prompts/verify_normalize.txt")

        self.groundtruth = None

    def run(self):
        self.dump_dir_path = f"outputs/UnifiedPipeline-{self.experiment_type}/{self.instance_description}/"
        self.groundtruth = GroundTruth(self.instance_description)
        self._load_data()

        if self.experiment_type == "CPE_CAE_CSE-2":
            if len(self.orders) == 0:
                raise RuntimeError(
                    f"Missing orders at {self.dump_dir_path}orders.json. "
                    "Please provide NL orders before running unified_pipeline."
                )
            # Step 1: Free extraction
            if len(self.raw_jsons) == 0:
                self.free_extract_all()
            # Step 2: Verify + normalize
            if len(self.normalized_jsons) == 0:
                self.verify_and_normalize_all()
            # Build route sheets from normalized JSONs
            self._build_route_sheets()
            # Step 3: Derive constraints
            self.derive_constraints()
            # Step 4: Solve + ground
            self.solve_jsp()
            self.ground_production_plan()

        elif self.experiment_type == "CSE-1":
            # Input: ground-truth route sheets. Skip Steps 1-2.
            self.route_sheets = read_json(
                f"outputs/GroundTruth/{self.instance_description}/route_sheets.json"
            )
            self._build_field_mapping_from_gt()
            self.derive_constraints()
            self.solve_jsp()
            self.ground_production_plan()

        elif self.experiment_type == "SGE":
            # Input: ground-truth assigned_jobs. Skip Steps 1-3.
            self.assigned_jobs = read_json(
                f"outputs/GroundTruth/{self.instance_description}/assigned_jobs.json"
            )
            self.route_sheets = read_json(
                f"outputs/GroundTruth/{self.instance_description}/route_sheets.json"
            )
            self._build_field_mapping_from_gt()
            self.ground_production_plan()

    # ------------------------------------------------------------------ #
    #  Step 1: Free extraction                                            #
    # ------------------------------------------------------------------ #

    def free_extract_all(self):
        """Extract structured JSON from each order's NL description."""
        print("Step 1: Free extraction ...")
        self.raw_jsons = []

        prompts = []
        for order in self.orders:
            prompt = self.free_extract_prompt.replace("---ORDER---", json.dumps(order))
            prompts.append(prompt)

        results = self._parallel_llm_calls(prompts)

        for result in results:
            parsed = self._safe_json_parse(result)
            self.raw_jsons.append(parsed)

        write_json(self.dump_dir_path + "raw_jsons.json", self.raw_jsons)

    # ------------------------------------------------------------------ #
    #  Step 2: Verify + normalize                                         #
    # ------------------------------------------------------------------ #

    def verify_and_normalize_all(self):
        """Verify each extracted JSON against its NL source, normalize fields."""
        print("Step 2: Verify + normalize ...")

        # 2a: Discover field vocabulary from raw extractions
        field_vocab = self._discover_field_vocab(self.raw_jsons)
        component_vocab = self._discover_component_vocab(self.raw_jsons)

        # 2b: Per-job verification + normalization
        prompts = []
        for order, raw_json in zip(self.orders, self.raw_jsons):
            prompt = self.verify_normalize_prompt \
                .replace("---ORDER---", json.dumps(order)) \
                .replace("---EXTRACTED---", json.dumps(raw_json)) \
                .replace("---FIELD_VOCAB---", json.dumps(field_vocab)) \
                .replace("---COMPONENT_VOCAB---", json.dumps(component_vocab))
            prompts.append(prompt)

        results = self._parallel_llm_calls(prompts)

        self.normalized_jsons = []
        for result in results:
            parsed = self._safe_json_parse(result)
            self.normalized_jsons.append(parsed)

        # 2c: Build field mapping from the normalized output
        self.field_mapping = self._build_field_mapping(self.normalized_jsons)

        write_json(self.dump_dir_path + "normalized_jsons.json", self.normalized_jsons)
        write_json(self.dump_dir_path + "field_mapping.json", self.field_mapping)

    def _discover_field_vocab(self, jsons):
        """Collect all unique field names across extracted JSONs and propose canonical names."""
        all_fields = Counter()
        for data in jsons:
            self._collect_fields(data, all_fields)

        # Build canonical name suggestions based on frequency
        canonical = {
            "operation": ["operation", "action", "process", "op", "step_name"],
            "machine": ["machine", "device", "equipment", "tool_machine"],
            "duration": ["duration", "time", "processing_time", "minutes"],
            "inputs": ["inputs", "input", "input_materials", "raw_material", "precondition"],
            "outputs": ["outputs", "output", "output_materials", "product", "postcondition"],
            "type": ["type", "material_type", "component_type", "category"],
            "parameters": ["parameters", "params", "settings", "config"],
        }

        # For each semantic role, pick the name that appears most frequently
        vocab = {}
        for role, candidates in canonical.items():
            best = role  # default
            best_count = 0
            for c in candidates:
                if all_fields[c] > best_count:
                    best = c
                    best_count = all_fields[c]
            vocab[role] = best

        return vocab

    def _discover_component_vocab(self, jsons):
        """Collect all unique component/material names across jobs."""
        components = set()
        for data in jsons:
            self._collect_string_values(data, components)
        result = sorted(components)
        # Truncate to avoid overly long prompts
        if len(result) > 200:
            result = result[:200]
        return result

    def _collect_fields(self, obj, counter):
        if isinstance(obj, dict):
            for k, v in obj.items():
                counter[k] += 1
                self._collect_fields(v, counter)
        elif isinstance(obj, list):
            for item in obj:
                self._collect_fields(item, counter)

    def _collect_string_values(self, obj, values):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, str) and len(v) > 1:
                    values.add(v)
                else:
                    self._collect_string_values(v, values)
        elif isinstance(obj, list):
            for item in obj:
                self._collect_string_values(item, values)

    def _build_field_mapping(self, jsons):
        """Analyze normalized JSONs to discover which field names map to which semantic roles."""
        all_fields = Counter()
        for data in jsons:
            self._collect_fields(data, all_fields)

        # Heuristic mapping: match field names to semantic roles
        role_keywords = {
            "operation_field": ["operation", "action", "process", "op"],
            "machine_field": ["machine", "device", "equipment"],
            "duration_field": ["duration", "time", "processing_time", "minutes"],
            "input_field": ["inputs", "input", "input_materials", "precondition", "raw_material"],
            "output_field": ["outputs", "output", "output_materials", "postcondition", "product"],
            "input_type_field": ["type", "material_type", "component_type", "category"],
            "output_type_field": ["type", "material_type", "component_type", "category"],
            "params_field": ["parameters", "params", "settings"],
        }

        mapping = {}
        for role, keywords in role_keywords.items():
            best = keywords[0]
            best_count = 0
            for kw in keywords:
                if all_fields[kw] > best_count:
                    best = kw
                    best_count = all_fields[kw]
            mapping[role] = best

        return mapping

    def _build_field_mapping_from_gt(self):
        """Build field mapping for ground truth route sheet format."""
        self.field_mapping = {
            "operation_field": "operation",
            "machine_field": "machine",
            "duration_field": "duration",
            "input_field": "precondition",
            "output_field": "postcondition",
            "input_type_field": "component_type",
            "output_type_field": "component_type",
            "params_field": "parameters",
        }

    # ------------------------------------------------------------------ #
    #  Build route sheets from normalized JSONs                           #
    # ------------------------------------------------------------------ #

    def _build_route_sheets(self):
        """Convert normalized JSONs into route_sheet format for evaluation compatibility."""
        fm = self.field_mapping
        self.route_sheets = []

        for i, data in enumerate(self.normalized_jsons):
            route_sheet_entry = {
                "instance_description": self.instance_description,
                "route_sheet": []
            }

            steps = self._find_steps(data)
            for step in steps:
                rs_step = {
                    "machine": self._get_field(step, fm.get("machine_field", "machine"), ""),
                    "duration": str(self._get_field(step, fm.get("duration_field", "duration"), 0)),
                    "operation": self._get_field(step, fm.get("operation_field", "operation"), ""),
                    "precondition": self._extract_materials(
                        step, fm.get("input_field", "inputs"), fm.get("input_type_field", "type")
                    ),
                    "postcondition": self._extract_materials(
                        step, fm.get("output_field", "outputs"), fm.get("output_type_field", "type")
                    ),
                    "parameters": self._extract_parameters(step, fm),
                }
                route_sheet_entry["route_sheet"].append(rs_step)

            self.route_sheets.append(route_sheet_entry)

        write_json(self.dump_dir_path + "route_sheets.json", self.route_sheets)

    def _find_steps(self, data):
        """Find the list of steps in a free-form JSON."""
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ["steps", "operations", "process", "manufacturing_steps", "route"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            # Try any list-valued field
            for k, v in data.items():
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                    return v
        return []

    def _get_field(self, step, field_name, default):
        """Get a field value with fallback."""
        if isinstance(step, dict):
            return step.get(field_name, default)
        return default

    def _extract_materials(self, step, field_name, type_field):
        """Extract input/output materials as a list of {component, component_type, container}."""
        raw = step.get(field_name, [])
        if isinstance(raw, str):
            return [{"component": raw, "component_type": raw, "container": ""}]
        if isinstance(raw, list):
            materials = []
            for item in raw:
                if isinstance(item, dict):
                    materials.append({
                        "component": item.get("component", item.get("name", item.get("material", ""))),
                        "component_type": item.get(type_field, item.get("component_type", "")),
                        "container": item.get("container", ""),
                    })
                elif isinstance(item, str):
                    materials.append({"component": item, "component_type": item, "container": ""})
            return materials
        if isinstance(raw, dict):
            return [{
                "component": raw.get("component", raw.get("name", "")),
                "component_type": raw.get(type_field, raw.get("component_type", "")),
                "container": raw.get("container", ""),
            }]
        return []

    def _extract_parameters(self, step, fm):
        """Extract parameters: everything that's not a core field."""
        core_fields = {
            fm.get("operation_field", "operation"),
            fm.get("machine_field", "machine"),
            fm.get("duration_field", "duration"),
            fm.get("input_field", "inputs"),
            fm.get("output_field", "outputs"),
            "step_id", "step_number", "step",
        }
        # Check if there's an explicit parameters field
        params_field = fm.get("params_field", "parameters")
        if params_field in step and isinstance(step[params_field], dict):
            return step[params_field]

        # Otherwise, collect non-core scalar fields as parameters
        params = {}
        for k, v in step.items():
            if k not in core_fields and not isinstance(v, (dict, list)):
                params[k] = v
        return params

    # ------------------------------------------------------------------ #
    #  Step 3: Deterministic constraint derivation                        #
    # ------------------------------------------------------------------ #

    def derive_constraints(self):
        """Derive OR matrix from route sheets using key-value type matching."""
        print("Step 3: Deriving constraints ...")
        self.or_matrix = []
        self.compile_error_num = 0
        self.total_num = 0

        # Collect all machine names
        machine_set = set()
        for rs_data in self.route_sheets:
            for step in rs_data.get("route_sheet", []):
                m = step.get("machine", "")
                if m:
                    machine_set.add(m.lower())
        self.machines = sorted(machine_set)

        # Build OR matrix: one row per job
        for rs_data in self.route_sheets:
            steps = rs_data.get("route_sheet", [])
            row = []

            # Derive precedence constraints via input/output type matching
            for i, step in enumerate(steps):
                self.total_num += 1
                # Machine index
                machine_name = step.get("machine", "").lower()
                try:
                    machine_idx = self.machines.index(machine_name)
                except ValueError:
                    machine_idx = 0
                    self.compile_error_num += 1

                # Duration
                try:
                    duration = int(
                        "".join(c for c in str(step.get("duration", 0)) if c.isdigit() or c == ".")
                        or "0"
                    )
                except (ValueError, TypeError):
                    duration = 0
                    self.compile_error_num += 1

                # Precedence: check if any earlier step's output type matches this step's input type
                current_input_types = {
                    mat.get("component_type", "").lower()
                    for mat in step.get("precondition", [])
                    if mat.get("component_type")
                }
                pre_indexes = []
                for j in range(i):
                    prev_step = steps[j]
                    prev_output_types = {
                        mat.get("component_type", "").lower()
                        for mat in prev_step.get("postcondition", [])
                        if mat.get("component_type")
                    }
                    if current_input_types & prev_output_types:
                        pre_indexes.append(j)

                row.append([machine_idx, duration, pre_indexes])
            self.or_matrix.append(row)

        write_json(self.dump_dir_path + "or_matrix.json", self.or_matrix)
        write_json(self.dump_dir_path + "machines.json", self.machines)

    # ------------------------------------------------------------------ #
    #  Step 4: Solve + ground                                             #
    # ------------------------------------------------------------------ #

    def solve_jsp(self):
        """Run OR-Tools JSP solver."""
        print("Step 4a: Solving JSP ...")
        or_matrix = copy.deepcopy(self.or_matrix)
        assigned_jobs, solver, err_rate = schedule(or_matrix)
        if len(assigned_jobs) == 0:
            print("No solution found.")
            self.compile_error_num += 1
        else:
            self.assigned_jobs = assigned_jobs
        write_json(self.dump_dir_path + "assigned_jobs.json", self.assigned_jobs)
        write_txt(self.dump_dir_path + "err_rate.txt", str(err_rate))

    def ground_production_plan(self):
        """Map solver output back to route sheet details to produce the final plan."""
        print("Step 4b: Grounding production plan ...")
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

                # Look up details from route sheet
                if job_id < len(self.route_sheets):
                    rs_steps = self.route_sheets[job_id].get("route_sheet", [])
                    if task_id < len(rs_steps):
                        rs_step = rs_steps[task_id]
                        step_info["operation"] = rs_step.get("operation", "")
                        step_info["machine"] = rs_step.get("machine", "")
                        step_info["duration"] = rs_step.get("duration", "")
                        step_info["precondition"] = [
                            mat.get("component_type", "")
                            for mat in rs_step.get("precondition", [])
                        ]
                        step_info["postcondition"] = [
                            mat.get("component_type", "")
                            for mat in rs_step.get("postcondition", [])
                        ]
                        step_info["parameters"] = rs_step.get("parameters", {})

                machine_plan["production_sequence"].append(step_info)

            production_plan.append(machine_plan)

        self.production_plan = production_plan
        write_json(self.dump_dir_path + "production_plan.json", self.production_plan)
        write_txt(self.dump_dir_path + "compile_error_num.txt", str(self.compile_error_num))

    # ------------------------------------------------------------------ #
    #  Data loading                                                       #
    # ------------------------------------------------------------------ #

    def _load_data(self):
        if not os.path.exists(self.dump_dir_path):
            os.makedirs(self.dump_dir_path)
        if os.path.exists(self.dump_dir_path + "orders.json"):
            self.orders = read_json(self.dump_dir_path + "orders.json")
        if os.path.exists(self.dump_dir_path + "raw_jsons.json"):
            self.raw_jsons = read_json(self.dump_dir_path + "raw_jsons.json")
        if os.path.exists(self.dump_dir_path + "normalized_jsons.json"):
            self.normalized_jsons = read_json(self.dump_dir_path + "normalized_jsons.json")
        if os.path.exists(self.dump_dir_path + "field_mapping.json"):
            self.field_mapping = read_json(self.dump_dir_path + "field_mapping.json")
        if os.path.exists(self.dump_dir_path + "route_sheets.json"):
            self.route_sheets = read_json(self.dump_dir_path + "route_sheets.json")
        if os.path.exists(self.dump_dir_path + "or_matrix.json"):
            self.or_matrix = read_json(self.dump_dir_path + "or_matrix.json")
        if os.path.exists(self.dump_dir_path + "machines.json"):
            self.machines = read_json(self.dump_dir_path + "machines.json")
        if os.path.exists(self.dump_dir_path + "assigned_jobs.json"):
            self.assigned_jobs = read_json(self.dump_dir_path + "assigned_jobs.json")

    # ------------------------------------------------------------------ #
    #  LLM utilities                                                      #
    # ------------------------------------------------------------------ #

    def _parallel_llm_calls(self, prompts, max_workers=8):
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

    def _chatgpt_function(self, content, model="claude-sonnet-4-6-20250514", max_retries=5):
        client = OpenAI(
            base_url="http://localhost:4141/v1",
            api_key=os.environ.get("OPENAI_API_KEY", "sk-placeholder"),
        )
        for attempt in range(max_retries):
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are an expert in manufacturing processes."},
                        {"role": "user", "content": content},
                    ],
                    model=model,
                    max_tokens=8192,
                )
                return chat_completion.choices[0].message.content
            except openai.APIError as e:
                print(f"API error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    print(f"Max retries reached, returning empty response")
                    return "{}"

    def _safe_json_parse(self, text):
        """Parse JSON from LLM output, handling markdown code blocks."""
        if not text:
            return {}
        text = text.strip()
        # Strip markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            # Try to find JSON array
            start = text.find("[")
            end = text.rfind("]") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            print(f"Failed to parse JSON: {text[:200]}")
            return {}
