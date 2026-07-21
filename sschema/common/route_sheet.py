"""RouteSheet: mapping + LLM route sheet generation + reduction.

Ported from src/preprocess/RouteSheet.py (fb-2s). Changes:
- imports now come from common.io / common.llm
- prompt path is a constructor argument (was hardcoded to src/prompts/...)
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from common.io import read_json, read_txt, write_json
from common.llm import LLM_MODEL, make_chat_client


class RouteSheet:
    def __init__(
        self,
        machines_data_path,
        jssp_data_path,
        arrange_path,
        jssp_mapped_path,
        route_sheet_store_path,
        route_sheet_reduce_path,
        prompt_path,
    ):
        self.machines = read_json(machines_data_path)
        self.jssp_data = read_json(jssp_data_path)
        self.arrange = read_json(arrange_path)
        self.jssp_mapped_path = jssp_mapped_path
        self.route_sheet_reduce_path = route_sheet_reduce_path
        self.route_sheet_store_path = route_sheet_store_path
        self.prompt = read_txt(prompt_path)
        self.sys_prompt = "You are an expert in the field of manufacturing"

    def mapping(self):
        skip_arrange = os.environ.get("SKIP_ARRANGE_MAPPING", "").lower() in ("1", "true", "yes")
        for i in range(len(self.arrange)):
            mapping = self.arrange[i]["mapping"]
            data = self.jssp_data[i]["data"]
            if not skip_arrange:
                for job in data:
                    for step in job["steps"]:
                        key = str(step["machine"])
                        if key in mapping:
                            step["machine"] = mapping[key]
            self.jssp_data[i]["data"] = data
        write_json(self.jssp_mapped_path, self.jssp_data)

    def create_route_sheet(self, target_descriptions=None, target_flat_idxs=None):
        jssp_data = read_json(self.jssp_mapped_path)

        flat_ptr = []
        flat_desc = []
        for inst_idx, inst in enumerate(jssp_data):
            for job_idx in range(len(inst["data"])):
                flat_ptr.append((inst_idx, job_idx))
                flat_desc.append(inst["description"])
        total_jobs = len(flat_ptr)

        if os.path.exists(self.route_sheet_store_path):
            old_flat = read_json(self.route_sheet_store_path)
            if len(old_flat) < total_jobs:
                old_flat = old_flat + [{} for _ in range(total_jobs - len(old_flat))]
            elif len(old_flat) > total_jobs:
                old_flat = old_flat[:total_jobs]
        else:
            old_flat = [{} for _ in range(total_jobs)]

        if target_descriptions is None:
            target_idxs = list(range(total_jobs))
        else:
            target_set = set(target_descriptions)
            target_idxs = [i for i, d in enumerate(flat_desc) if d in target_set]

        if target_flat_idxs is not None:
            fset = set(target_flat_idxs)
            target_idxs = [i for i in target_idxs if i in fset]

        if not target_idxs:
            print("create_route_sheet: no jobs matched target_descriptions; nothing to do", flush=True)
            write_json(self.route_sheet_store_path, old_flat)
            return

        prompts = []
        prompt_flat_idxs = []
        prompt_step_counts = []
        prompt_expected_machines = []
        prompt_expected_durations = []
        skipped = 0
        for flat_i in target_idxs:
            inst_idx, job_idx = flat_ptr[flat_i]
            steps = jssp_data[inst_idx]["data"][job_idx]["steps"]
            if any(s["machine"] is None for s in steps):
                skipped += 1
                continue
            real_job = self.__create_real_job(steps)
            prompts.append(self.prompt.replace("---STEPS---", json.dumps(real_job, indent=2)))
            prompt_flat_idxs.append(flat_i)
            prompt_step_counts.append(len(real_job))
            prompt_expected_machines.append([s["machine"] for s in real_job])
            prompt_expected_durations.append([s["duration"] for s in real_job])

        if skipped:
            print(f"create_route_sheet: skipped {skipped} target jobs with null machine mappings", flush=True)
        if not prompts:
            write_json(self.route_sheet_store_path, old_flat)
            return

        print(f"create_route_sheet: regenerating {len(prompts)} route sheets via LLM ({LLM_MODEL})", flush=True)

        max_workers = int(os.environ.get("LLM_MAX_WORKERS", "100"))
        client = make_chat_client()

        def call_llm(local_idx, prompt, expected_steps, expected_machines, expected_durations):
            for attempt in range(5):
                try:
                    resp = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": self.sys_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        model=LLM_MODEL,
                        max_tokens=16384,
                    )
                    text = resp.choices[0].message.content or ""
                except Exception as e:
                    print(f"LLM error (attempt {attempt+1}/5) local_idx={local_idx}: {e}", flush=True)
                    if attempt < 4:
                        time.sleep(2 ** attempt)
                    continue

                try:
                    parsed = json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    print(f"JSON parse failed local_idx={local_idx} attempt {attempt+1}/5", flush=True)
                    if attempt < 4:
                        time.sleep(2 ** attempt)
                    continue

                rs = parsed.get("route_sheet") if isinstance(parsed, dict) else None
                if not isinstance(rs, list) or len(rs) != expected_steps:
                    got = len(rs) if isinstance(rs, list) else "N/A"
                    print(
                        f"step count mismatch local_idx={local_idx}: expected {expected_steps}, got {got}; attempt {attempt+1}/5",
                        flush=True,
                    )
                    if attempt < 4:
                        time.sleep(2 ** attempt)
                    continue

                bad_at = None
                bad_reason = None
                for i, step in enumerate(rs):
                    if not isinstance(step, dict):
                        bad_at, bad_reason = i, f"step is not a dict (got {type(step).__name__})"
                        break
                    if step.get("machine") != expected_machines[i]:
                        bad_at = i
                        bad_reason = f"machine mismatch: got {step.get('machine')!r} expected {expected_machines[i]!r}"
                        break
                    want_dur = f"{expected_durations[i]} minutes"
                    if step.get("duration") != want_dur:
                        bad_at = i
                        bad_reason = f"duration mismatch: got {step.get('duration')!r} expected {want_dur!r}"
                        break
                if bad_at is not None:
                    print(
                        f"content mismatch local_idx={local_idx} step={bad_at}: {bad_reason}; attempt {attempt+1}/5",
                        flush=True,
                    )
                    if attempt < 4:
                        time.sleep(2 ** attempt)
                    continue

                return local_idx, parsed
            return local_idx, {}

        results = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    call_llm,
                    i,
                    p,
                    prompt_step_counts[i],
                    prompt_expected_machines[i],
                    prompt_expected_durations[i],
                )
                for i, p in enumerate(prompts)
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="route sheet LLM"):
                local_idx, parsed = future.result()
                results[local_idx] = parsed if isinstance(parsed, dict) else {}

        empty_after_retry = []
        for local_i, flat_i in enumerate(prompt_flat_idxs):
            if results[local_i]:
                old_flat[flat_i] = results[local_i]
            else:
                empty_after_retry.append(flat_i)

        write_json(self.route_sheet_store_path, old_flat)
        print(f"create_route_sheet: wrote {len(old_flat)} entries to {self.route_sheet_store_path}", flush=True)
        if empty_after_retry:
            print(
                f"create_route_sheet: {len(empty_after_retry)} slot(s) still empty after 5-attempt inline retries "
                f"(flat_idxs: {empty_after_retry[:20]}{'...' if len(empty_after_retry) > 20 else ''})",
                flush=True,
            )

    def route_sheet_reduce(self):
        route_sheet = read_json(self.route_sheet_store_path)
        result = []
        jssp_len = []
        for jssp in self.jssp_data:
            jssp_len.append(len(jssp["data"]))
        for i in range(len(jssp_len)):
            result.append([])
            for j in range(jssp_len[i]):
                if len(route_sheet) == 0:
                    break
                ele = route_sheet.pop(0)

                if isinstance(ele, dict):
                    ele["instance_description"] = self.jssp_data[i]["description"]
                    result[i].append(ele)
                else:
                    result[i].append({
                        "instance_description": self.jssp_data[i]["description"]
                    })
        while len(result) > 0 and len(result[-1]) == 0:
            result.pop()
        write_json(self.route_sheet_reduce_path, result)
        print("Route sheet reduce finished")

    def __create_real_job(self, steps):
        new_steps = []
        for step in steps:
            machine = self.machines[int(step["machine"])]["machine"]
            patterns = self.machines[int(step["machine"])]["patterns"]
            duration = int(step["time"])
            new_steps.append({
                "machine": machine,
                "duration": duration,
                "patterns": patterns
            })
        return new_steps
