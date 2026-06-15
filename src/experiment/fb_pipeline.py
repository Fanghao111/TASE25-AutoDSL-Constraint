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


class FBPipeline:
    def __init__(self, instance_description: str, experiment_type: str = "CPE_CAE_CSE-2", force: bool = True):
        self.instance_description = instance_description
        self.experiment_type = experiment_type
        self.force = force
        self.dump_dir_path = ""

        self.orders = []
        self.raw_jsons = []          # Step 1 output
        self.verified_jsons = []     # Step 1.5 output
        self.normalized_jsons = []   # Step 2 output
        self.semantic_roles = {}     # Step 2 output: LLM-discovered roles
        self.field_mapping = {}      # derived from semantic_roles for downstream
        self._job_steps = []         # unified internal representation: list of step-lists
        self.or_matrix = []
        self.machines = []
        self.assigned_jobs = {}
        self.production_plan = []

        self.compile_error_num = 0
        self.total_num = 0

        self.free_extract_prompt = read_txt("src/prompts/free_extract.txt")
        self.verify_extraction_prompt = read_txt("src/prompts/verify_extraction.txt")
        self.normalize_prompt = read_txt("src/prompts/normalize.txt")
        self.discover_roles_prompt = read_txt("src/prompts/discover_roles.txt")
        self.fix_mapping_prompt = read_txt("src/prompts/fix_mapping.txt")

        self.groundtruth = None

    def run(self):
        self.dump_dir_path = f"outputs/FB-{self.experiment_type}/{self.instance_description}/"
        self.groundtruth = GroundTruth(self.instance_description)
        self._load_data()

        if self.experiment_type == "CPE_CAE_CSE-2":
            if len(self.orders) == 0:
                raise RuntimeError(
                    f"Missing orders at {self.dump_dir_path}orders.json. "
                    "Please provide NL orders before running unified_pipeline."
                )
            # ---- Stage 1: Extract + Verify ----
            if len(self.raw_jsons) == 0:
                self.free_extract_all()
            if len(self.verified_jsons) == 0:
                self.verify_extraction_all()

            # ---- Stage 2: Normalize + Discover roles ----
            if len(self.normalized_jsons) == 0:
                self.normalize()
            if not self.semantic_roles:
                self.discover_roles()

            # ---- Stage 3: Compile + Solve + Ground ----
            self._job_steps = [self._find_steps(data) for data in self.normalized_jsons]
            self.derive_constraints()
            self.solve_jsp()
            self.ground_production_plan()

        elif self.experiment_type == "CSE-1":
            route_sheets = read_json(
                f"outputs/GroundTruth/{self.instance_description}/route_sheets.json"
            )
            self._job_steps = [rs.get("route_sheet", []) for rs in route_sheets]
            self._build_field_mapping_from_gt()
            # Stage 3
            self.derive_constraints()
            self.solve_jsp()
            self.ground_production_plan()

        elif self.experiment_type == "SGE":
            self.assigned_jobs = read_json(
                f"outputs/GroundTruth/{self.instance_description}/assigned_jobs.json"
            )
            route_sheets = read_json(
                f"outputs/GroundTruth/{self.instance_description}/route_sheets.json"
            )
            self._job_steps = [rs.get("route_sheet", []) for rs in route_sheets]
            self._build_field_mapping_from_gt()
            # Stage 3 (ground only)
            self.ground_production_plan()

    # ------------------------------------------------------------------ #
    #  Stage 1a: Free extraction                                          #
    # ------------------------------------------------------------------ #

    def free_extract_all(self):
        """Extract structured JSON from each order's NL description."""
        print("Stage 1a: Free extraction ...", flush=True)
        self.raw_jsons = []

        prompts = []
        for order in self.orders:
            prompt = self.free_extract_prompt.replace("---ORDER---", json.dumps(order))
            prompts.append(prompt)

        results = self._parallel_llm_calls(prompts)

        for result in results:
            parsed = self._safe_json_parse(result)
            self.raw_jsons.append(parsed)

        # Retry failed extractions
        empty_indices = [i for i, d in enumerate(self.raw_jsons) if d == {}]
        if empty_indices:
            print(f"  Retrying {len(empty_indices)} failed extractions...", flush=True)
            for idx in empty_indices:
                result = self._chatgpt_function(prompts[idx])
                parsed = self._safe_json_parse(result)
                if parsed != {}:
                    self.raw_jsons[idx] = parsed
            still_empty = sum(1 for d in self.raw_jsons if d == {})
            print(f"  After retry: {still_empty}/{len(self.raw_jsons)} still empty", flush=True)

        write_json(self.dump_dir_path + "CAM-1_raw_jsons.json", self.raw_jsons)

    # ------------------------------------------------------------------ #
    #  Stage 1b: Verify extractions                                       #
    # ------------------------------------------------------------------ #

    def verify_extraction_all(self):
        """Verify each extracted JSON against its NL source, fix errors."""
        print("Stage 1b: Verify extractions ...", flush=True)

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

        # Retry failed verifications
        empty_indices = [i for i, d in enumerate(self.verified_jsons) if d == {}]
        if empty_indices:
            print(f"  Retrying {len(empty_indices)} failed verifications...", flush=True)
            for idx in empty_indices:
                prompt = self.verify_extraction_prompt \
                    .replace("---ORDER---", json.dumps(self.orders[idx])) \
                    .replace("---EXTRACTED---", json.dumps(self.raw_jsons[idx]))
                result = self._chatgpt_function(prompt)
                parsed = self._safe_json_parse(result)
                if parsed != {}:
                    self.verified_jsons[idx] = parsed
            still_empty = sum(1 for d in self.verified_jsons if d == {})
            print(f"  After retry: {still_empty}/{len(self.verified_jsons)} still empty", flush=True)

        write_json(self.dump_dir_path + "CAM-2_verified_jsons.json", self.verified_jsons)

    # ------------------------------------------------------------------ #
    #  CAM-3: Normalize field and component names                         #
    # ------------------------------------------------------------------ #

    def normalize(self):
        """LLM-based field name normalization + deterministic component normalization."""
        print("CAM-3: Normalizing field and component names ...", flush=True)

        # Collect field frequencies and component names
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

        threshold = max(len(all_steps) * 0.05, 3)
        frequent_fields = sorted([f for f, c in field_counter.items() if c >= threshold])
        rare_fields = sorted([f for f, c in field_counter.items() if c < threshold])
        unique_components = sorted(all_components)

        print(f"  {len(frequent_fields)} frequent fields (>={threshold:.0f} occurrences), "
              f"{len(rare_fields)} rare fields, {len(unique_components)} unique components", flush=True)

        # Sample representative JSONs
        sample_sizes = [(len(json.dumps(self.verified_jsons[i])), i) for i in range(len(self.verified_jsons))]
        sample_sizes.sort()
        mid_idx = sample_sizes[len(sample_sizes) // 2][1]
        samples = [self.verified_jsons[mid_idx]]

        # LLM call — normalize field and component names
        prompt = self.normalize_prompt \
            .replace("---SAMPLES---", json.dumps(samples)) \
            .replace("---FIELD_NAMES---", json.dumps(frequent_fields)) \
            .replace("---COMPONENT_NAMES---", "[]")

        print(f"  Prompt size: {len(prompt)} chars", flush=True)
        result = self._chatgpt_function(prompt)
        print(f"  LLM response length: {len(result) if result else 0} chars", flush=True)
        normalization = self._safe_json_parse(result)

        # Convert groups to flat mappings
        field_mapping_raw = self._groups_to_mapping(normalization.get("field_groups", []))
        print(f"  Field groups: {len(normalization.get('field_groups', []))}", flush=True)

        # Component normalization: fully deterministic (PascalCase)
        component_mapping = self._extend_component_mapping({}, unique_components)

        # Apply mappings to all verified_jsons
        self.normalized_jsons = []
        for data in self.verified_jsons:
            normalized = self._apply_field_mapping(data, field_mapping_raw)
            normalized = self._apply_value_mapping(normalized, component_mapping)
            self.normalized_jsons.append(normalized)

        write_json(self.dump_dir_path + "CAM-3_normalized_jsons.json", self.normalized_jsons)
        write_json(self.dump_dir_path + "CAM-3_field_mapping_raw.json", field_mapping_raw)
        write_json(self.dump_dir_path + "CAM-3_component_mapping.json", component_mapping)

    # ------------------------------------------------------------------ #
    #  SRD: Semantic Role Discovery                                       #
    # ------------------------------------------------------------------ #

    def discover_roles(self):
        """Discover which fields serve scheduling-specific semantic roles."""
        print("SRD: Discovering semantic roles ...", flush=True)

        # Collect canonical field names from normalized JSONs
        from collections import Counter
        field_counter = Counter()
        all_steps = []
        for data in self.normalized_jsons:
            steps = self._find_steps(data)
            all_steps.extend(steps)
            for step in steps:
                if isinstance(step, dict):
                    for k in step:
                        field_counter[k] += 1

        threshold = max(len(all_steps) * 0.05, 3)
        canonical_fields = sorted([f for f, c in field_counter.items() if c >= threshold])

        # Sample representative normalized JSONs
        sample_sizes = [(len(json.dumps(self.normalized_jsons[i])), i) for i in range(len(self.normalized_jsons))]
        sample_sizes.sort()
        mid_idx = sample_sizes[len(sample_sizes) // 2][1]
        samples = [self.normalized_jsons[mid_idx]]

        # LLM call — discover semantic roles
        prompt = self.discover_roles_prompt \
            .replace("---SAMPLES---", json.dumps(samples)) \
            .replace("---FIELD_NAMES---", json.dumps(canonical_fields))

        print(f"  Prompt size: {len(prompt)} chars", flush=True)
        result = self._chatgpt_function(prompt)
        print(f"  LLM response length: {len(result) if result else 0} chars", flush=True)
        discovery = self._safe_json_parse(result)

        self.semantic_roles = discovery.get("semantic_roles", {})

        # Validate discovered roles
        issues = self._validate_semantic_roles(self.normalized_jsons, self.semantic_roles)

        if issues:
            print(f"  Validation found {len(issues)} issues, requesting fix ...", flush=True)
            fix_prompt = self.fix_mapping_prompt \
                .replace("---MAPPING---", json.dumps({"semantic_roles": self.semantic_roles}, indent=2)) \
                .replace("---ISSUES---", json.dumps(issues, indent=2)) \
                .replace("---SAMPLES---", json.dumps(samples, indent=2))
            fix_result = self._chatgpt_function(fix_prompt)
            fixed = self._safe_json_parse(fix_result)

            if fixed.get("semantic_roles"):
                self.semantic_roles = fixed["semantic_roles"]
        else:
            print("  Validation passed.", flush=True)

        # Build field_mapping for downstream (Stage 3)
        self.field_mapping = self._build_field_mapping_from_roles(self.semantic_roles)

        write_json(self.dump_dir_path + "SRD_semantic_roles.json", self.semantic_roles)
        write_json(self.dump_dir_path + "SRD_field_mapping.json", self.field_mapping)

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
        For unmapped components, normalize to PascalCase (e.g., 'sheet metal' -> 'SheetMetal')."""
        import re
        canonical_set = set(base_mapping.values())
        mapped_set = set(base_mapping.keys()) | canonical_set

        extended = dict(base_mapping)
        for comp in all_components:
            if comp in mapped_set:
                continue
            # Normalize: lowercase/spaces/underscores -> PascalCase
            # "sheet metal" -> "SheetMetal", "lathed_part" -> "LathedPart"
            normalized = re.sub(r'[_\s]+', ' ', comp).strip()
            normalized = ''.join(word.capitalize() for word in normalized.split())
            if normalized != comp:
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
        # Strip bracket notation: "inputs[].type" or "inputs[*].type" → "type"
        if type_field:
            import re
            parts = re.split(r'\[[^\]]*\]\.', type_field)
            if len(parts) > 1:
                type_field = parts[-1]
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
                # Strip array bracket notation: "inputs[].type" or "inputs[*].type" → "type"
                # The LLM sometimes returns full path notation, but downstream
                # code needs just the sub-field name within each list item.
                if mapping_key in ("input_type_field",):
                    import re
                    parts = re.split(r'\[[^\]]*\]\.', field)
                    if len(parts) > 1:
                        field = parts[-1]
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
    #  Route sheet builder (static — used by evaluation as adapter)       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_route_sheets(normalized_jsons, field_mapping, instance_description=""):
        """Convert normalized JSONs into route_sheet format for evaluation compatibility.

        This is NOT a pipeline step — it's an adapter for evaluation to produce
        the route_sheet format needed for comparison with other methods.
        """
        fm = field_mapping
        route_sheets = []

        for data in normalized_jsons:
            route_sheet_entry = {
                "instance_description": instance_description,
                "route_sheet": []
            }

            steps = FBPipeline._find_steps_static(data)
            for step in steps:
                rs_step = {
                    "machine": step.get(fm.get("machine_field", "machine"), ""),
                    "duration": str(step.get(fm.get("duration_field", "duration"), 0)),
                    "operation": step.get(fm.get("operation_field", "operation"), ""),
                    "precondition": FBPipeline._extract_materials_static(
                        step, fm.get("input_field", "inputs"), fm.get("input_type_field", "type")
                    ),
                    "postcondition": FBPipeline._extract_materials_static(
                        step, fm.get("output_field", "outputs"), fm.get("output_type_field", "type")
                    ),
                    "parameters": FBPipeline._extract_parameters_static(step, fm),
                }
                route_sheet_entry["route_sheet"].append(rs_step)

            route_sheets.append(route_sheet_entry)

        return route_sheets

    @staticmethod
    def _find_steps_static(data):
        """Find the list of steps in a free-form JSON, filtering non-dict entries."""
        steps = []
        if isinstance(data, list):
            steps = data
        elif isinstance(data, dict):
            for key in ["steps", "operations", "process", "manufacturing_steps", "route"]:
                if key in data and isinstance(data[key], list):
                    steps = data[key]
                    break
            else:
                for k, v in data.items():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                        steps = v
                        break
        return [s for s in steps if isinstance(s, dict)]

    @staticmethod
    def _extract_materials_static(step, field_name, type_field):
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

    @staticmethod
    def _extract_parameters_static(step, fm):
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

    def _find_steps(self, data):
        """Find the list of steps in a free-form JSON, filtering non-dict entries."""
        return FBPipeline._find_steps_static(data)

    def _get_field(self, step, field_name, default):
        if isinstance(step, dict):
            return step.get(field_name, default)
        return default

    def _get_type_values(self, step, field_name, type_field):
        """Extract component type strings from a material/dependency field, handling various formats."""
        raw = step.get(field_name, [])
        types = set()
        if isinstance(raw, str) and raw:
            types.add(raw)
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    t = item.get(type_field, item.get("component_type", ""))
                    if t:
                        types.add(t)
                elif isinstance(item, str) and item:
                    types.add(item)
        elif isinstance(raw, dict):
            t = raw.get(type_field, raw.get("component_type", ""))
            if t:
                types.add(t)
        return types

    # ------------------------------------------------------------------ #
    #  Stage 3: Compile + Solve + Ground                                  #
    # ------------------------------------------------------------------ #

    def derive_constraints(self):
        """Derive OR matrix from job steps using key-value type matching."""
        print("Stage 3a: Deriving constraints ...", flush=True)
        self.or_matrix = []
        self.compile_error_num = 0
        self.total_num = 0

        fm = self.field_mapping
        machine_field = fm.get("machine_field", "machine")
        duration_field = fm.get("duration_field", "duration")
        input_field = fm.get("input_field", "precondition")
        output_field = fm.get("output_field", "postcondition")
        type_field = fm.get("input_type_field", "component_type")

        machine_set = set()
        for steps in self._job_steps:
            for step in steps:
                m = step.get(machine_field, "")
                if m:
                    machine_set.add(str(m).lower())
        self.machines = sorted(machine_set)

        for steps in self._job_steps:
            row = []

            for i, step in enumerate(steps):
                self.total_num += 1
                machine_name = str(step.get(machine_field, "")).lower()
                try:
                    machine_idx = self.machines.index(machine_name)
                except ValueError:
                    machine_idx = 0
                    self.compile_error_num += 1

                try:
                    duration = int(
                        "".join(c for c in str(step.get(duration_field, 0)) if c.isdigit() or c == ".")
                        or "0"
                    )
                except (ValueError, TypeError):
                    duration = 0
                    self.compile_error_num += 1

                current_input_types = {t.lower() for t in self._get_type_values(step, input_field, type_field)}
                pre_indexes = []
                for j in range(i):
                    prev_step = steps[j]
                    prev_output_types = {t.lower() for t in self._get_type_values(prev_step, output_field, type_field)}
                    if current_input_types & prev_output_types:
                        pre_indexes.append(j)

                row.append([machine_idx, duration, pre_indexes])
            self.or_matrix.append(row)

        write_json(self.dump_dir_path + "CGM_or_matrix.json", self.or_matrix)
        write_json(self.dump_dir_path + "CGM_machines.json", self.machines)

    # ------------------------------------------------------------------ #
    #  Stage 3b/3c: Solve + Ground                                        #
    # ------------------------------------------------------------------ #

    def solve_jsp(self):
        """Run OR-Tools JSP solver."""
        print("Stage 3b: Solving JSP ...", flush=True)
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
        """Map solver output back to step details to produce the final plan."""
        print("Stage 3c: Grounding production plan ...", flush=True)
        production_plan = []

        fm = self.field_mapping
        op_field = fm.get("operation_field", "operation")
        machine_field = fm.get("machine_field", "machine")
        duration_field = fm.get("duration_field", "duration")
        input_field = fm.get("input_field", "precondition")
        output_field = fm.get("output_field", "postcondition")
        type_field = fm.get("input_type_field", "component_type")

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

                if job_id < len(self._job_steps):
                    job_steps = self._job_steps[job_id]
                    if task_id < len(job_steps):
                        src_step = job_steps[task_id]
                        step_info["operation"] = src_step.get(op_field, "")
                        step_info["machine"] = str(src_step.get(machine_field, ""))
                        step_info["duration"] = str(src_step.get(duration_field, ""))
                        step_info["precondition"] = list(self._get_type_values(src_step, input_field, type_field))
                        step_info["postcondition"] = list(self._get_type_values(src_step, output_field, type_field))
                        step_info["parameters"] = FBPipeline._extract_parameters_static(src_step, fm)

                machine_plan["production_sequence"].append(step_info)

            production_plan.append(machine_plan)

        self.production_plan = production_plan
        write_json(self.dump_dir_path + "SGM_production_plan.json", self.production_plan)
        write_txt(self.dump_dir_path + "compile_error_num.txt", str(self.compile_error_num))

    # ------------------------------------------------------------------ #
    #  Data loading                                                       #
    # ------------------------------------------------------------------ #

    def _load_data(self):
        if not os.path.exists(self.dump_dir_path):
            os.makedirs(self.dump_dir_path)
        # Always load input orders
        if os.path.exists(self.dump_dir_path + "orders.json"):
            self.orders = read_json(self.dump_dir_path + "orders.json")
        # When force=True, skip loading intermediate files so all steps re-run and overwrite
        if self.force:
            return
        if os.path.exists(self.dump_dir_path + "CAM-1_raw_jsons.json"):
            self.raw_jsons = read_json(self.dump_dir_path + "CAM-1_raw_jsons.json")
        if os.path.exists(self.dump_dir_path + "CAM-2_verified_jsons.json"):
            self.verified_jsons = read_json(self.dump_dir_path + "CAM-2_verified_jsons.json")
        if os.path.exists(self.dump_dir_path + "CAM-3_normalized_jsons.json"):
            self.normalized_jsons = read_json(self.dump_dir_path + "CAM-3_normalized_jsons.json")
        if os.path.exists(self.dump_dir_path + "SRD_field_mapping.json"):
            self.field_mapping = read_json(self.dump_dir_path + "SRD_field_mapping.json")
        if os.path.exists(self.dump_dir_path + "SRD_semantic_roles.json"):
            self.semantic_roles = read_json(self.dump_dir_path + "SRD_semantic_roles.json")
        if os.path.exists(self.dump_dir_path + "CGM_or_matrix.json"):
            self.or_matrix = read_json(self.dump_dir_path + "CGM_or_matrix.json")
        if os.path.exists(self.dump_dir_path + "CGM_machines.json"):
            self.machines = read_json(self.dump_dir_path + "CGM_machines.json")
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
                stream = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are an expert in structured data analysis and process scheduling."},
                        {"role": "user", "content": content},
                    ],
                    model=model,
                    max_tokens=16384,
                    stream=True,
                )
                chunks = []
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        chunks.append(chunk.choices[0].delta.content)
                result = "".join(chunks)
                if result.strip():
                    return result
                print(f"Empty streaming response (attempt {attempt+1}/{max_retries})", flush=True)
            except Exception as e:
                print(f"API error (attempt {attempt+1}/{max_retries}): {e}", flush=True)
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
