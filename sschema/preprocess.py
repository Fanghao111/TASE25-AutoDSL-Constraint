"""sschema preprocess — reproducibility entry.

Merges what used to be `main.py --mode preprocess` (mapping → route_sheet →
reduce) and `generate_orders.py` (per-step NL orders) into a single CLI. This
is the ONLY LLM-heavy stage in sschema — the definitive canonical outputs are
already shipped under sschema/preprocess_out/, so most users should NOT re-run
this. Use it only when regenerating from raw data.

Environment variables:
  LLM_BASE_URL, LLM_MODEL, OPENAI_API_KEY, LLM_MAX_WORKERS — see common/llm.py
  SKIP_ARRANGE_MAPPING=1 — skip arrange.json remapping in the mapping stage.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Allow running as `python sschema/preprocess.py` from repo root without
# installing sschema as a package.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from tqdm import tqdm

from common.io import read_json, read_txt, write_json
from common.llm import LLM_MODEL, make_chat_client
from common.route_sheet import RouteSheet


ALL_INSTANCES = [f"ta{i}" for i in range(71, 81)]
DUR_RE = re.compile(r"\b(\d+)[\s\-]minutes?\b", re.IGNORECASE)


def _to_full(inst: str) -> str:
    return inst if inst.startswith("instance ") else f"instance {inst}"


def _to_short(inst: str) -> str:
    return inst.replace("instance ", "")


def _parse_int_prefix(s):
    if not isinstance(s, str):
        return None
    m = re.match(r"\s*(\d+)", s)
    return int(m.group(1)) if m else None


def _expected_duration(step):
    return _parse_int_prefix(step.get("duration"))


def _call_step_llm(client, prompt_template, step_dict, expected_dur, max_attempts=5):
    """Per-step LLM call with retries. Returns NL string or None."""
    prompt = prompt_template.replace("---STEP---", json.dumps(step_dict, indent=2))
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert in the field of manufacturing"},
                    {"role": "user", "content": prompt},
                ],
                model=LLM_MODEL,
                max_tokens=2048,
            )
            text = resp.choices[0].message.content or ""
        except Exception:
            if attempt < max_attempts:
                time.sleep(min(2 ** attempt, 30))
            continue

        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            if attempt < max_attempts:
                time.sleep(min(2 ** attempt, 30))
            continue

        if not (isinstance(parsed, dict) and isinstance(parsed.get("step"), str)):
            if attempt < max_attempts:
                time.sleep(min(2 ** attempt, 30))
            continue

        nl = parsed["step"]
        if expected_dur is not None:
            if not any(int(x) == expected_dur for x in DUR_RE.findall(nl)):
                if attempt < max_attempts:
                    time.sleep(min(2 ** attempt, 30))
                continue
        return nl
    return None


# --------------------------------------------------------------------- #
#  Stage entry points                                                   #
# --------------------------------------------------------------------- #


def _make_route_sheet(data_dir, prompts_dir, out_dir):
    return RouteSheet(
        machines_data_path=os.path.join(data_dir, "machines.json"),
        jssp_data_path=os.path.join(data_dir, "jssp_data.json"),
        arrange_path=os.path.join(data_dir, "arrange.json"),
        jssp_mapped_path=os.path.join(out_dir, "jssp_mapped.json"),
        route_sheet_store_path=os.path.join(out_dir, "route_sheet.json"),
        route_sheet_reduce_path=os.path.join(out_dir, "route_sheet_reduce.json"),
        prompt_path=os.path.join(prompts_dir, "route_sheet_prompt.txt"),
    )


def run_mapping(data_dir, prompts_dir, out_dir, targets):
    os.makedirs(out_dir, exist_ok=True)
    rs = _make_route_sheet(data_dir, prompts_dir, out_dir)
    rs.mapping()


def run_route_sheet(data_dir, prompts_dir, out_dir, targets):
    os.makedirs(out_dir, exist_ok=True)
    rs = _make_route_sheet(data_dir, prompts_dir, out_dir)
    rs.create_route_sheet(target_descriptions=set(targets))


def run_reduce(data_dir, prompts_dir, out_dir, targets):
    os.makedirs(out_dir, exist_ok=True)
    rs = _make_route_sheet(data_dir, prompts_dir, out_dir)
    rs.route_sheet_reduce()


def run_orders(data_dir, prompts_dir, out_dir, targets, retry_missing=False):
    prompt_template = read_txt(os.path.join(prompts_dir, "generate_synthetic_step_prompt.txt"))
    reduce_path = os.path.join(out_dir, "route_sheet_reduce.json")
    if not os.path.exists(reduce_path):
        raise FileNotFoundError(
            f"{reduce_path} not found — run --stage reduce first, or copy from preprocess_out/."
        )
    route_sheet_all = read_json(reduce_path)

    orders_out_dir = os.path.join(out_dir, "orders")
    os.makedirs(orders_out_dir, exist_ok=True)
    max_workers = int(os.environ.get("LLM_MAX_WORKERS", "64"))
    client = make_chat_client()

    target_full = set(targets)

    for domain_data in route_sheet_all:
        if not domain_data:
            continue
        instance_description = domain_data[0].get("instance_description")
        if instance_description not in target_full:
            continue

        instance_short = _to_short(instance_description)
        out_path = os.path.join(orders_out_dir, instance_short, "orders.json")

        n_jobs = len(domain_data)
        job_steps = [None] * n_jobs
        job_part_names = [None] * n_jobs
        tasks = []

        existing_orders = None
        if os.path.exists(out_path) and retry_missing:
            existing_orders = read_json(out_path)

        for j, rs_entry in enumerate(domain_data):
            if not (isinstance(rs_entry, dict) and rs_entry.get("route_sheet")):
                continue
            steps = rs_entry["route_sheet"]
            job_part_names[j] = rs_entry.get("part_name", "")
            job_steps[j] = [None] * len(steps)
            for k, step in enumerate(steps):
                if existing_orders is not None:
                    prior = existing_orders[j] if j < len(existing_orders) else None
                    if isinstance(prior, dict) and isinstance(prior.get("steps"), list):
                        if k < len(prior["steps"]) and isinstance(prior["steps"][k], str) and prior["steps"][k]:
                            job_steps[j][k] = prior["steps"][k]
                            continue
                tasks.append((j, k, step, _expected_duration(step)))

        if not tasks:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            if existing_orders is None:
                write_json(out_path, [{} for _ in range(n_jobs)])
                print(f"[EMPTY] {instance_description}: no jobs with populated route_sheet")
            else:
                print(f"[SKIP] {instance_description}: --retry-missing but all slots already filled")
            continue

        print(f"[START] {instance_description}: {len(tasks)} step-calls "
              f"({LLM_MODEL}, {max_workers} workers)")

        def _run_batch(batch, desc):
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                fut_to_key = {
                    ex.submit(_call_step_llm, client, prompt_template, step, exp): (j, k)
                    for (j, k, step, exp) in batch
                }
                for fut in tqdm(as_completed(fut_to_key), total=len(fut_to_key), desc=desc):
                    j, k = fut_to_key[fut]
                    job_steps[j][k] = fut.result()

        _run_batch(tasks, instance_short)

        none_tasks = [
            (j, k, tasks[i][2], tasks[i][3])
            for i, (j, k, *_rest) in enumerate(tasks)
            if job_steps[j] is not None and job_steps[j][k] is None
        ]
        if none_tasks:
            print(f"  auto-retry: {len(none_tasks)} slot(s) still None → second 5-attempt round")
            _run_batch(none_tasks, f"{instance_short} (retry)")

        orders = []
        empty_positions = 0
        for j in range(n_jobs):
            if job_steps[j] is None:
                orders.append({})
                continue
            empty_positions += sum(1 for s in job_steps[j] if s is None)
            orders.append({"part_name": job_part_names[j], "steps": job_steps[j]})

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        write_json(out_path, orders)
        tag = f"({empty_positions} step(s) still None)" if empty_positions else "(all succeeded)"
        print(f"  Saved: {out_path}  {tag}")


# --------------------------------------------------------------------- #
#  CLI                                                                  #
# --------------------------------------------------------------------- #


STAGES = ("mapping", "route_sheet", "reduce", "orders", "all")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=STAGES, default="all",
                    help="Which stage to run (default: all).")
    ap.add_argument("--instances", nargs="+", default=None,
                    help="Instance short-names (ta71 ..) or full ('instance ta71'). Default: all 10.")
    ap.add_argument("--data-dir", default=os.path.join(_HERE, "data"),
                    help="Directory with raw jssp_data / arrange / machines JSONs.")
    ap.add_argument("--prompts-dir", default=os.path.join(_HERE, "prompts"),
                    help="Directory with LLM prompt templates.")
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "artifacts", "preprocess"),
                    help="Where to write regenerated artifacts. Does NOT overwrite preprocess_out/ by default.")
    ap.add_argument("--retry-missing", action="store_true",
                    help="For --stage orders: only re-run positions currently None in an existing orders.json.")
    args = ap.parse_args()

    insts = args.instances or ALL_INSTANCES
    targets = [_to_full(i) for i in insts]

    os.makedirs(args.out_dir, exist_ok=True)

    if args.stage in ("mapping", "all"):
        print(f"=== stage: mapping ({args.out_dir}) ===", flush=True)
        run_mapping(args.data_dir, args.prompts_dir, args.out_dir, targets)
    if args.stage in ("route_sheet", "all"):
        print(f"=== stage: route_sheet ({args.out_dir}) ===", flush=True)
        run_route_sheet(args.data_dir, args.prompts_dir, args.out_dir, targets)
    if args.stage in ("reduce", "all"):
        print(f"=== stage: reduce ({args.out_dir}) ===", flush=True)
        run_reduce(args.data_dir, args.prompts_dir, args.out_dir, targets)
    if args.stage in ("orders", "all"):
        print(f"=== stage: orders ({args.out_dir}) ===", flush=True)
        run_orders(args.data_dir, args.prompts_dir, args.out_dir, targets, retry_missing=args.retry_missing)


if __name__ == "__main__":
    main()
