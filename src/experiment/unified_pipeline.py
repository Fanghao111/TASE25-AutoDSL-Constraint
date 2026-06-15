from __future__ import annotations
import copy
import openai
import os
import time
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from utils.util import read_json, write_json, read_txt, write_txt
from src.experiment.schedule import schedule
from src.experiment.groundtruth import GroundTruth


class UnifiedPipeline:
    def __init__(self, instance_description: str, experiment_type: str = "CPE_CAE_CSE-2"):
        self.instance_description = instance_description
        self.experiment_type = experiment_type
        self.dump_dir_path = ""

        self.orders = []
        self.raw_jsons = []          # Step 1 output
        self.verified_jsons = []     # Step 1.5 output
        self.normalized_jsons = []   # Step 2 output
        self.semantic_roles = {}     # Step 2 output: LLM-discovered roles
        self.field_mapping = {}      # derived from semantic_roles for downstream
        self.route_sheets = []
        self.or_matrix = []
        self.machines = []
        self.assigned_jobs = {}
        self.production_plan = []

        self.compile_error_num = 0
        self.total_num = 0

        self.free_extract_prompt = read_txt("src/prompts/free_extract.txt")
        self.verify_extraction_prompt = read_txt("src/prompts/verify_extraction.txt")
        self.normalize_and_discover_prompt = read_txt("src/prompts/normalize_and_discover.txt")
        self.fix_mapping_prompt = read_txt("src/prompts/fix_mapping.txt")

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
            # Step 1.5: Verify extractions
            if len(self.verified_jsons) == 0:
                self.verify_extraction_all()
            # Step 2: Normalize + discover semantic roles
            if len(self.normalized_jsons) == 0:
                self.normalize_and_discover()
            # Build route sheets from normalized JSONs
            self._build_route_sheets()
            # Step 3: Derive constraints
            self.derive_constraints()
            # Step 4: Solve + ground
            self.solve_jsp()
            self.ground_production_plan()

        elif self.experiment_type == "CSE-1":
            self.route_sheets = read_json(
                f"outputs/GroundTruth/{self.instance_description}/route_sheets.json"
            )
            self._build_field_mapping_from_gt()
            self.derive_constraints()
            self.solve_jsp()
            self.ground_production_plan()

        elif self.experiment_type == "SGE":
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
        print("Step 1: Free extraction ...", flush=True)
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
    #  Step 1.5: Verify extractions                                       #
    # ------------------------------------------------------------------ #

    def verify_extraction_all(self):
        """Verify each extracted JSON against its NL source, fix errors."""
        print("Step 1.5: Verify extractions ...", flush=True)

        prompts = []
        for order, raw_json in zip(self.orders, self.raw_jsons):
            prompt = self.verify_extraction_prompt \
                .replace("---ORDER---", json.dumps(order)) \
                .replace("---EXTRACTED---", json.dumps(raw_json))
            prompts.append(prompt)

        results = self._parallel_llm_calls(prompts)

        self.verified_jsons = []
        for result in results:
            parsed = self._safe_json_parse(result)
            self.verified_jsons.append(parsed)

        write_json(self.dump_dir_path + "verified_jsons.json", self.verified_jsons)

    # ------------------------------------------------------------------ #
    #  Step 2: Normalize + discover semantic roles                        #
    # ------------------------------------------------------------------ #

    def normalize_and_discover(self):
        """One LLM call for normalization + role discovery, then deterministic apply + validate."""
        print("Step 2: Normalize + discover semantic roles ...", flush=True)

        # 2a: Collect field frequencies and component names
        from collections import Counter
        field_counter = Counter()
        all_components = set()
        all_steps = []
        for data in self.verified_jsons:
            steps = self._find_steps(data)
            all_steps.extend(steps)
            for step in steps:
                if isinstance(step, dict):
                    for k in step:
                        field_counter[k] += 1
            self._collect_string_values(data, all_components)

        # Only send fields that appear in >=5% of steps for normalization
        # This filters out rare parameter fields and keeps core scheduling fields
        threshold = max(len(all_steps) * 0.05, 3)
        frequent_fields = sorted([f for f, c in field_counter.items() if c >= threshold])
        rare_fields = sorted([f for f, c in field_counter.items() if c < threshold])
        unique_components = sorted(all_components)

        print(f"  {len(frequent_fields)} frequent fields (>={threshold:.0f} occurrences), "
              f"{len(rare_fields)} rare fields, {len(unique_components)} unique components", flush=True)

        # 2b: Sample representative JSONs for the LLM (keep compact — proxy has ~6K char limit)
        # Pick the smallest sample that still shows the structure
        sample_sizes = [(len(json.dumps(self.verified_jsons[i])), i) for i in range(len(self.verified_jsons))]
        sample_sizes.sort()
        # Pick a medium-sized sample (not too small to miss fields, not too large for proxy)
        mid_idx = sample_sizes[len(sample_sizes) // 2][1]
        samples = [self.verified_jsons[mid_idx]]

        # 2c: One LLM call — normalize fields + discover roles
        # Skip component names entirely — handle via deterministic normalization
        prompt = self.normalize_and_discover_prompt \
            .replace("---SAMPLES---", json.dumps(samples)) \
            .replace("---FIELD_NAMES---", json.dumps(frequent_fields)) \
            .replace("---COMPONENT_NAMES---", "[]")

        print(f"  Prompt size: {len(prompt)} chars", flush=True)
        result = self._chatgpt_function(prompt)
        print(f"  LLM response length: {len(result) if result else 0} chars", flush=True)
        discovery = self._safe_json_parse(result)

        # Convert groups to flat mappings
        field_mapping_raw = self._groups_to_mapping(discovery.get("field_groups", []))
        self.semantic_roles = discovery.get("semantic_roles", {})
        print(f"  Field groups: {len(discovery.get('field_groups', []))}", flush=True)

        # Component normalization: fully deterministic (CamelCase/PascalCase -> lowercase spaces)
        component_mapping = self._extend_component_mapping({}, unique_components)

        # 2d: Deterministic replacement on all verified_jsons
        self.normalized_jsons = []
        for data in self.verified_jsons:
            normalized = self._apply_field_mapping(data, field_mapping_raw)
            normalized = self._apply_value_mapping(normalized, component_mapping)
            self.normalized_jsons.append(normalized)

        # 2e: Deterministic rule validation
        issues = self._validate_semantic_roles(self.normalized_jsons, self.semantic_roles)

        if issues:
            print(f"  Validation found {len(issues)} issues, requesting fix ...", flush=True)
            fix_prompt = self.fix_mapping_prompt \
                .replace("---MAPPING---", json.dumps(discovery, indent=2)) \
                .replace("---ISSUES---", json.dumps(issues, indent=2)) \
                .replace("---SAMPLES---", json.dumps(samples, indent=2))
            fix_result = self._chatgpt_function(fix_prompt)
            fixed = self._safe_json_parse(fix_result)

            if fixed.get("field_groups"):
                field_mapping_raw = self._groups_to_mapping(fixed["field_groups"])
            if fixed.get("semantic_roles"):
                self.semantic_roles = fixed["semantic_roles"]

            # Re-apply with fixed mappings
            self.normalized_jsons = []
            for data in self.verified_jsons:
                normalized = self._apply_field_mapping(data, field_mapping_raw)
                normalized = self._apply_value_mapping(normalized, component_mapping)
                self.normalized_jsons.append(normalized)
        else:
            print("  Validation passed.", flush=True)

        # 2f: Build field_mapping for downstream from semantic_roles
        self.field_mapping = self._build_field_mapping_from_roles(self.semantic_roles)

        write_json(self.dump_dir_path + "normalized_jsons.json", self.normalized_jsons)
        write_json(self.dump_dir_path + "field_mapping.json", self.field_mapping)
        write_json(self.dump_dir_path + "semantic_roles.json", self.semantic_roles)
        write_json(self.dump_dir_path + "field_mapping_raw.json", field_mapping_raw)
        write_json(self.dump_dir_path + "component_mapping.json", component_mapping)

    def _sample_jsons(self, jsons, n=5):
        """Select representative samples: pick diverse structures."""
        if len(jsons) <= n:
            return jsons
        # Pick first, last, and random middle ones
        indices = [0, len(jsons) - 1]
        middle = random.sample(range(1, len(jsons) - 1), min(n - 2, len(jsons) - 2))
        indices.extend(middle)
        return [jsons[i] for i in sorted(set(indices))]

    def _collect_field_names(self, obj, fields):
        """Recursively collect all field names."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                fields.add(k)
                self._collect_field_names(v, fields)
        elif isinstance(obj, list):
            for item in obj:
                self._collect_field_names(item, fields)

    def _collect_string_values(self, obj, values):
        """Collect string values from nested JSON."""
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, str) and len(v) > 1:
                    values.add(v)
                else:
                    self._collect_string_values(v, values)
        elif isinstance(obj, list):
            for item in obj:
                self._collect_string_values(item, values)

    def _extend_component_mapping(self, base_mapping, all_components):
        """Extend LLM-provided component mapping to cover all components.
        For unmapped components, normalize CamelCase/snake_case to lowercase with spaces."""
        import re
        # Build reverse map: canonical -> set of originals
        canonical_set = set(base_mapping.values())
        mapped_set = set(base_mapping.keys()) | canonical_set

        extended = dict(base_mapping)
        for comp in all_components:
            if comp in mapped_set:
                continue
            # Normalize: CamelCase -> space-separated, underscores -> spaces, lowercase
            normalized = re.sub(r'([a-z])([A-Z])', r'\1 \2', comp)
            normalized = normalized.replace('_', ' ').strip().lower()
            if normalized != comp:
                # Check if normalized form matches an existing canonical
                if normalized in canonical_set:
                    extended[comp] = normalized
                else:
                    extended[comp] = normalized
        return extended

    def _groups_to_mapping(self, groups):
        """Convert group format [{"canonical": "x", "members": ["a", "b"]}] to flat mapping {"a": "x", "b": "x"}."""
        mapping = {}
        for group in groups:
            canonical = group.get("canonical", "")
            for member in group.get("members", []):
                if member != canonical:
                    mapping[member] = canonical
        return mapping

    def _apply_field_mapping(self, obj, mapping):
        """Recursively rename field keys."""
        if isinstance(obj, dict):
            return {mapping.get(k, k): self._apply_field_mapping(v, mapping) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._apply_field_mapping(item, mapping) for item in obj]
        return obj

    def _apply_value_mapping(self, obj, mapping):
        """Recursively replace string values."""
        if isinstance(obj, dict):
            return {k: self._apply_value_mapping(v, mapping) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._apply_value_mapping(item, mapping) for item in obj]
        elif isinstance(obj, str):
            return mapping.get(obj, obj)
        return obj

    def _validate_semantic_roles(self, normalized_jsons, semantic_roles):
        """Deterministic validation of discovered semantic roles."""
        issues = []

        # Collect all steps from normalized JSONs
        all_steps = []
        for data in normalized_jsons:
            steps = self._find_steps(data)
            all_steps.extend(steps)

        if not all_steps or not semantic_roles:
            issues.append("No steps found or no semantic roles discovered")
            return issues

        # Validate duration field: values should be numeric
        duration_field = semantic_roles.get("duration", {}).get("field", "")
        if duration_field:
            non_numeric = 0
            total = 0
            for step in all_steps:
                val = step.get(duration_field)
                if val is not None:
                    total += 1
                    try:
                        float(str(val).replace(",", ""))
                    except (ValueError, TypeError):
                        non_numeric += 1
            if total > 0 and non_numeric / total > 0.3:
                issues.append(
                    f"duration field '{duration_field}': {non_numeric}/{total} values are non-numeric"
                )

        # Validate resource field: values should repeat across jobs (shared resources)
        resource_field = semantic_roles.get("resource", {}).get("field", "")
        if resource_field:
            from collections import Counter
            resource_values = Counter()
            for step in all_steps:
                val = step.get(resource_field)
                if isinstance(val, str) and val:
                    resource_values[val] += 1
            if resource_values:
                unique_ratio = len(resource_values) / sum(resource_values.values())
                if unique_ratio > 0.9:
                    issues.append(
                        f"resource field '{resource_field}': {len(resource_values)} unique values out of "
                        f"{sum(resource_values.values())} total — resources should be shared across steps"
                    )

        # Validate dependency fields: input/output type overlap should exist
        dep_in_field = semantic_roles.get("dependency_in", {}).get("field", "")
        dep_out_field = semantic_roles.get("dependency_out", {}).get("field", "")
        type_field = semantic_roles.get("dependency_type_field", {}).get("field", "")
        if dep_in_field and dep_out_field and type_field:
            all_in_types = set()
            all_out_types = set()
            for step in all_steps:
                for mat in (step.get(dep_in_field) or []):
                    if isinstance(mat, dict):
                        t = mat.get(type_field, "")
                        if t:
                            all_in_types.add(t.lower())
                for mat in (step.get(dep_out_field) or []):
                    if isinstance(mat, dict):
                        t = mat.get(type_field, "")
                        if t:
                            all_out_types.add(t.lower())
            if all_in_types and all_out_types and not (all_in_types & all_out_types):
                issues.append(
                    f"dependency fields: no overlap between input types ({len(all_in_types)}) "
                    f"and output types ({len(all_out_types)}) — precedence constraints won't work"
                )

        # Validate field coverage
        for role_name, role_info in semantic_roles.items():
            field = role_info.get("field", "")
            if not field:
                issues.append(f"role '{role_name}' has no field assigned")
                continue
            present = sum(1 for step in all_steps if field in step)
            if all_steps and present / len(all_steps) < 0.3:
                issues.append(
                    f"role '{role_name}' field '{field}': only present in {present}/{len(all_steps)} steps"
                )

        return issues

    def _build_field_mapping_from_roles(self, semantic_roles):
        """Convert semantic_roles to the field_mapping format used by downstream code."""
        role_to_key = {
            "operation": "operation_field",
            "resource": "machine_field",
            "duration": "duration_field",
            "dependency_in": "input_field",
            "dependency_out": "output_field",
            "dependency_type_field": "input_type_field",
            "extra_params": "params_field",
        }
        mapping = {}
        for role_name, mapping_key in role_to_key.items():
            field = semantic_roles.get(role_name, {}).get("field", "")
            if field:
                mapping[mapping_key] = field

        # If dependency_type_field wasn't discovered, auto-detect from data
        if "input_type_field" not in mapping:
            type_field = self._detect_type_field(semantic_roles)
            mapping["input_type_field"] = type_field

        mapping["output_type_field"] = mapping.get("input_type_field", "type")
        return mapping

    def _detect_type_field(self, semantic_roles):
        """Auto-detect the sub-field within input/output used for type matching."""
        dep_in_field = semantic_roles.get("dependency_in", {}).get("field", "input")
        # Look at actual data to find the type sub-field
        for data in self.normalized_jsons[:10]:
            steps = self._find_steps(data)
            for step in steps:
                materials = step.get(dep_in_field, [])
                if isinstance(materials, list):
                    for mat in materials:
                        if isinstance(mat, dict):
                            # Common type field names
                            for candidate in ["type", "component_type", "material_type", "category"]:
                                if candidate in mat:
                                    return candidate
        return "type"

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
            for k, v in data.items():
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                    return v
        return []

    def _get_field(self, step, field_name, default):
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
        params_field = fm.get("params_field", "parameters")
        if params_field in step and isinstance(step[params_field], dict):
            return step[params_field]

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
        print("Step 3: Deriving constraints ...", flush=True)
        self.or_matrix = []
        self.compile_error_num = 0
        self.total_num = 0

        machine_set = set()
        for rs_data in self.route_sheets:
            for step in rs_data.get("route_sheet", []):
                m = step.get("machine", "")
                if m:
                    machine_set.add(m.lower())
        self.machines = sorted(machine_set)

        for rs_data in self.route_sheets:
            steps = rs_data.get("route_sheet", [])
            row = []

            for i, step in enumerate(steps):
                self.total_num += 1
                machine_name = step.get("machine", "").lower()
                try:
                    machine_idx = self.machines.index(machine_name)
                except ValueError:
                    machine_idx = 0
                    self.compile_error_num += 1

                try:
                    duration = int(
                        "".join(c for c in str(step.get("duration", 0)) if c.isdigit() or c == ".")
                        or "0"
                    )
                except (ValueError, TypeError):
                    duration = 0
                    self.compile_error_num += 1

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
        print("Step 4a: Solving JSP ...", flush=True)
        or_matrix = copy.deepcopy(self.or_matrix)
        assigned_jobs, solver, err_rate, makespan = schedule(or_matrix)
        if len(assigned_jobs) == 0:
            print("No solution found.")
            self.compile_error_num += 1
        else:
            self.assigned_jobs = assigned_jobs
        write_json(self.dump_dir_path + "assigned_jobs.json", self.assigned_jobs)
        write_txt(self.dump_dir_path + "err_rate.txt", str(err_rate))
        write_txt(self.dump_dir_path + "makespan.txt", str(makespan))

    def ground_production_plan(self):
        """Map solver output back to route sheet details to produce the final plan."""
        print("Step 4b: Grounding production plan ...", flush=True)
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
        if os.path.exists(self.dump_dir_path + "verified_jsons.json"):
            self.verified_jsons = read_json(self.dump_dir_path + "verified_jsons.json")
        if os.path.exists(self.dump_dir_path + "normalized_jsons.json"):
            self.normalized_jsons = read_json(self.dump_dir_path + "normalized_jsons.json")
        if os.path.exists(self.dump_dir_path + "field_mapping.json"):
            self.field_mapping = read_json(self.dump_dir_path + "field_mapping.json")
        if os.path.exists(self.dump_dir_path + "semantic_roles.json"):
            self.semantic_roles = read_json(self.dump_dir_path + "semantic_roles.json")
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
            timeout=300.0,
        )
        for attempt in range(max_retries):
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are an expert in structured data analysis and process scheduling."},
                        {"role": "user", "content": content},
                    ],
                    model=model,
                    max_tokens=8192,
                )
                return chat_completion.choices[0].message.content
            except Exception as e:
                print(f"API error (attempt {attempt+1}/{max_retries}): {e}", flush=True)
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
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
