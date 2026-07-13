"""Per-step generator: for each step in each route sheet, ask the LLM to produce
one NL sentence, then assemble {part_name, steps} per job. All calls for a model
go through ONE global ThreadPoolExecutor to maximize throughput.

Env vars:
  ORDERS_OUTPUT_DIR     — output root (default "preprocess/orders").
  ORDERS_OUTPUT_ROOT    — deprecated alias for ORDERS_OUTPUT_DIR (still honored,
                          logs a warning).
  LLM_MAX_WORKERS       — pool size (default 64).
  LLM_MODEL, LLM_BASE_URL, OPENAI_API_KEY — inherited by make_chat_client().
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from utils.util import LLM_MODEL, make_chat_client, read_json, read_txt, write_json

INSTANCES = [
    "instance ta71", "instance ta72", "instance ta73", "instance ta74", "instance ta75",
    "instance ta76", "instance ta77", "instance ta78", "instance ta79", "instance ta80",
]

DUR_RE = re.compile(r"\b(\d+)[\s\-]minutes?\b", re.IGNORECASE)


def _resolve_output_dir():
    d = os.environ.get("ORDERS_OUTPUT_DIR")
    if d:
        return d
    legacy = os.environ.get("ORDERS_OUTPUT_ROOT")
    if legacy:
        print("WARNING: ORDERS_OUTPUT_ROOT is deprecated, use ORDERS_OUTPUT_DIR", file=sys.stderr)
        return legacy
    return "preprocess/orders"


def _parse_int_prefix(s):
    if not isinstance(s, str):
        return None
    m = re.match(r"\s*(\d+)", s)
    return int(m.group(1)) if m else None


def _expected_duration(step):
    """Get the integer duration expected in the NL sentence for a route_sheet step."""
    return _parse_int_prefix(step.get("duration"))


def call_step_llm(client, prompt_template, step_dict, expected_dur, max_attempts=5):
    """Make one per-step LLM call with retries. Returns NL string or None."""
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
        except Exception as e:
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
        # Loose duration check: expected int must appear as "<N> minute(s?)" (allow "-").
        if expected_dur is not None:
            if not any(int(x) == expected_dur for x in DUR_RE.findall(nl)):
                if attempt < max_attempts:
                    time.sleep(min(2 ** attempt, 30))
                continue

        return nl
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reduce", default="preprocess/route_sheet_reduce.json",
                    help="Path to route_sheet_reduce.json (source of truth for steps + part_name).")
    ap.add_argument("--retry-missing", action="store_true",
                    help="If orders.json exists, only re-run positions that are currently None.")
    args = ap.parse_args()

    prompt_template = read_txt("src/prompts/generate_synthetic_step_prompt.txt")
    route_sheet_all = read_json(args.reduce)

    out_dir = _resolve_output_dir()
    max_workers = int(os.environ.get("LLM_MAX_WORKERS", "64"))
    client = make_chat_client()

    for domain_data in route_sheet_all:
        if not domain_data:
            continue
        instance_description = domain_data[0].get("instance_description")
        if instance_description not in INSTANCES:
            continue

        instance_short = instance_description.replace("instance ", "")
        out_path = os.path.join(out_dir, instance_short, "orders.json")

        # Collect (job_idx, step_idx, step_dict, expected_dur) work items for this instance.
        # Skip jobs whose route_sheet is empty (upstream never generated one).
        n_jobs = len(domain_data)
        job_steps = [None] * n_jobs      # per-job: list of NL strings (or dict template)
        job_part_names = [None] * n_jobs
        tasks = []                        # list of (job_idx, step_idx, step_dict, expected_dur)

        existing_orders = None
        if os.path.exists(out_path) and args.retry_missing:
            existing_orders = read_json(out_path)

        for j, rs_entry in enumerate(domain_data):
            if not (isinstance(rs_entry, dict) and rs_entry.get("route_sheet")):
                continue
            steps = rs_entry["route_sheet"]
            job_part_names[j] = rs_entry.get("part_name", "")
            job_steps[j] = [None] * len(steps)
            for k, step in enumerate(steps):
                # If retry-missing mode and this slot already has content, skip.
                if existing_orders is not None:
                    prior = existing_orders[j] if j < len(existing_orders) else None
                    if isinstance(prior, dict) and isinstance(prior.get("steps"), list):
                        if k < len(prior["steps"]) and isinstance(prior["steps"][k], str) and prior["steps"][k]:
                            job_steps[j][k] = prior["steps"][k]
                            continue
                tasks.append((j, k, step, _expected_duration(step)))

        if not tasks and existing_orders is None:
            # Empty domain — nothing to do (create empty file for consistency).
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            write_json(out_path, [{} for _ in range(n_jobs)])
            print(f"[EMPTY] {instance_description}: no jobs with populated route_sheet")
            continue

        # If no tasks but existing_orders present, we're just re-saving; log and continue.
        if not tasks:
            print(f"[SKIP] {instance_description}: --retry-missing but all slots already filled")
            continue

        print(f"[START] {instance_description}: {len(tasks)} step-calls "
              f"across {sum(1 for j in job_steps if j is not None)} job(s) "
              f"({LLM_MODEL}, {max_workers} workers)")

        def _run_batch(batch, desc):
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                future_to_key = {
                    ex.submit(call_step_llm, client, prompt_template, step, exp): (j, k)
                    for (j, k, step, exp) in batch
                }
                for fut in tqdm(as_completed(future_to_key), total=len(future_to_key), desc=desc):
                    j, k = future_to_key[fut]
                    job_steps[j][k] = fut.result()

        _run_batch(tasks, instance_short)

        # Auto-recover slots the first pass left as None. Each call_step_llm
        # already burns 5 inline attempts (with duration/struct checks), so any
        # None here is a slot where all 5 attempts failed a hard check or hit
        # the API. One extra pass over just those slots gives them another 5
        # attempts. This replaces the previous manual --retry-missing gate for
        # the same use case (fresh runs); --retry-missing itself still works
        # for the "resume from an existing orders.json" case.
        none_tasks = [
            (j, k, tasks[i][2], tasks[i][3])
            for i, (j, k, *_rest) in enumerate(tasks)
            if job_steps[j] is not None and job_steps[j][k] is None
        ]
        if none_tasks:
            print(f"  auto-retry: {len(none_tasks)} slot(s) still None after first pass → second 5-attempt round")
            _run_batch(none_tasks, f"{instance_short} (retry)")

        # Assemble orders array: one entry per job slot.
        orders = []
        empty_positions = 0
        for j in range(n_jobs):
            if job_steps[j] is None:
                orders.append({})
                continue
            # Count None (failed) steps.
            empty_positions += sum(1 for s in job_steps[j] if s is None)
            orders.append({
                "part_name": job_part_names[j],
                "steps": job_steps[j],   # list of str-or-None
            })

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        write_json(out_path, orders)
        if empty_positions:
            print(f"  Saved: {out_path}  ({empty_positions} step(s) still None after retries)")
        else:
            print(f"  Saved: {out_path}  (all step calls succeeded)")
        print(f"[DONE] {instance_description}: {len(orders)} orders generated")


if __name__ == "__main__":
    main()
