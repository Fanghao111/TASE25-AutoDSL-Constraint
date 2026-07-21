"""GroundTruth: build canonical route sheets / OR matrix / assigned jobs /
production plan for a single instance from the mapped JSSP + reduced route sheet.

Ported from src/experiment/groundtruth.py (fb-2s). Changes:
- All input paths (route_sheet_reduce, jssp_mapped, machines) + the output
  dump dir are constructor arguments (were hardcoded to preprocess/ + data/ +
  outputs/GroundTruth/).
- Imports now come from common.io / common.cpsat.
- DSL-only helpers were never present in fb-2s (already trimmed upstream).
"""
from __future__ import annotations
import copy
import os

from common.cpsat import schedule
from common.io import read_json, write_json, write_txt


class GroundTruth:
    def __init__(
        self,
        instance_description: str,
        route_sheet_reduce_path: str,
        jssp_mapped_path: str,
        machines_data_path: str,
        dump_dir_path: str,
    ):
        self.route_sheets = []
        self.or_matrix = []
        self.assigned_jobs = []
        self.production_plan = []
        self.solver = None
        self.instance_description = instance_description
        self.route_sheet_reduce_path = route_sheet_reduce_path
        self.jssp_mapped_path = jssp_mapped_path
        self.dump_dir_path = dump_dir_path if dump_dir_path.endswith(os.sep) or dump_dir_path.endswith("/") else dump_dir_path + "/"
        self.machines = [ele["machine"] for ele in read_json(machines_data_path)]
        self.load_data()

    def get_grounded_route_sheet(self):
        if len(self.route_sheets) == 0:
            route_sheet_all = read_json(self.route_sheet_reduce_path)
            self.route_sheets = [
                rs for rs in route_sheet_all
                if rs[0]["instance_description"] == self.instance_description
            ][0]
            write_json(self.dump_dir_path + "route_sheets.json", self.route_sheets)
        return self.route_sheets

    def get_grounded_or_matrix(self):
        if len(self.or_matrix) == 0:
            # JSP standard linear precedence: task i depends on task i-1 within
            # each job (conjunctive arcs). Matches Taillard/OR-Tools convention
            # and lets solver find the true benchmark optimum.
            pre_indexes_all = []
            for route_sheet_data in self.route_sheets:
                steps = route_sheet_data.get("route_sheet", [])
                pre_indexes_all.append([[i - 1] if i > 0 else [] for i in range(len(steps))])

            jssp_mapped_all = read_json(self.jssp_mapped_path)
            jssp_mapped = [
                jssp for jssp in jssp_mapped_all
                if jssp["description"] == self.instance_description
            ][0]

            for i, job in enumerate(jssp_mapped["data"]):
                row = []
                for j, step in enumerate(job["steps"]):
                    ele = []
                    ele.append(step["machine"])
                    ele.append(step["time"])
                    pre_indexes = []
                    if len(pre_indexes_all) > i and len(pre_indexes_all[i]) > j:
                        pre_indexes = pre_indexes_all[i][j]
                    else:
                        continue
                    ele.append(pre_indexes)
                    row.append(ele)
                self.or_matrix.append(row)
            write_json(self.dump_dir_path + "or_matrix.json", self.or_matrix)
        return self.or_matrix

    def get_grounded_assigned_jobs(self):
        if len(self.assigned_jobs) == 0:
            or_matrix = copy.deepcopy(self.or_matrix)
            assigned_jobs, solver, _, makespan = schedule(or_matrix)
            if len(assigned_jobs) == 0:
                print("No solver")
                return
            self.assigned_jobs = assigned_jobs
            self.solver = solver
            write_json(self.dump_dir_path + "assigned_jobs.json", self.assigned_jobs)
            write_txt(self.dump_dir_path + "makespan.txt", str(makespan))
        return self.assigned_jobs

    def get_grounded_production_plan(self):
        if len(self.production_plan) == 0:
            production_plan = []
            for machine_index, production_sequence in self.assigned_jobs.items():
                machine_plan = {
                    "machine": self.machines[int(machine_index)],
                    "production_sequence": []
                }
                for step in production_sequence:
                    step_info = {
                        "start": step[0],
                        "end": step[0] + step[3],
                        "job_id": step[1],
                        "task_id": step[2]
                    }
                    if step[2] < len(self.route_sheets[step[1]].get("route_sheet", [])):
                        step_data = self.route_sheets[step[1]].get("route_sheet", [])[step[2]]
                        step_data["precondition"] = [ele["component_type"] for ele in step_data["precondition"]]
                        step_data["postcondition"] = [ele["component_type"] for ele in step_data["postcondition"]]
                        step_info.update(step_data)
                        step_info.pop("d", None)
                    machine_plan["production_sequence"].append(step_info)
                production_plan.append(machine_plan)
            self.production_plan = production_plan
            write_json(self.dump_dir_path + "production_plan.json", self.production_plan)
        return self.production_plan

    def load_data(self):
        os.makedirs(self.dump_dir_path, exist_ok=True)
        if os.path.exists(self.dump_dir_path + "route_sheets.json"):
            self.route_sheets = read_json(self.dump_dir_path + "route_sheets.json")
        if os.path.exists(self.dump_dir_path + "or_matrix.json"):
            self.or_matrix = read_json(self.dump_dir_path + "or_matrix.json")
        if os.path.exists(self.dump_dir_path + "assigned_jobs.json"):
            self.assigned_jobs = read_json(self.dump_dir_path + "assigned_jobs.json")
        if os.path.exists(self.dump_dir_path + "production_plan.json"):
            self.production_plan = read_json(self.dump_dir_path + "production_plan.json")
