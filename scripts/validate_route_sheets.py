"""Cross-check generated route sheets against the mapped input jssp.

Covers:
  Layer 1 — per-model hard constraints
    * struct_ok    : entry is a non-empty dict with a "route_sheet" list
    * len_ok       : len(route_sheet) == len(input steps)
    * machine_ok   : route_sheet[i].machine == expected machine name (after mapping)
    * duration_ok  : route_sheet[i].duration == "<input step time> minutes"

  Layer 3 — cross-model consensus (only over slots where all models are len_ok)
    * machine_agree : all N models agree on the full machine sequence
    * duration_agree: all N models agree on the full duration sequence

Any slot skipped by create_route_sheet (because arrange.mapping produced a
null machine) is excluded from the target set — those slots are legitimately
empty across all models.

Usage:
    python scripts/validate_route_sheets.py
    python scripts/validate_route_sheets.py --models qwen3-4b qwen3-30b
    python scripts/validate_route_sheets.py --show 30
"""

import argparse
import json
import os
import sys
from collections import Counter

DEFAULT_MODELS = ["deepseek-v3", "qwen3-235b", "qwen3-30b", "qwen3-4b"]
TARGET_INSTANCES = {f"instance ta{n}" for n in range(71, 81)}
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_expected(machines_path, jssp_mapped_path):
    """For every flat_idx in the mapped jssp, return the (machines, durations)
    the LLM was expected to reproduce. Slots with any null-machine step get
    None (create_route_sheet skips them, so they never had a chance to fill)."""
    machines = read_json(machines_path)
    jssp = read_json(jssp_mapped_path)

    expected = []          # list aligned with flat_idx: (machines_list, durations_list) or None
    flat_instance = []     # instance description at each flat_idx
    for inst in jssp:
        desc = inst["description"]
        for job in inst["data"]:
            flat_instance.append(desc)
            steps = job["steps"]
            if any(s["machine"] is None for s in steps):
                expected.append(None)
                continue
            machs = [machines[int(s["machine"])]["machine"] for s in steps]
            durs = [int(s["time"]) for s in steps]
            expected.append((machs, durs))
    return expected, flat_instance


def check_entry(entry, exp_machines, exp_durations):
    """Return (struct_ok, len_ok, machine_ok, duration_ok, model_machines, model_durations)."""
    if not (isinstance(entry, dict) and entry.get("route_sheet")):
        return False, False, False, False, None, None
    rs = entry["route_sheet"]
    if not isinstance(rs, list):
        return True, False, False, False, None, None
    if len(rs) != len(exp_machines):
        return True, False, False, False, None, None
    got_machines = [s.get("machine") for s in rs]
    got_durations = [s.get("duration") for s in rs]
    machine_ok = got_machines == exp_machines
    duration_ok = all(
        got_durations[i] == f"{exp_durations[i]} minutes"
        for i in range(len(rs))
    )
    return True, True, machine_ok, duration_ok, got_machines, got_durations


def per_model_scan(model, store_path, target_idxs, expected):
    """Run layer-1 checks for one model. Returns dict of counters + per-slot
    detail for cross-model comparison."""
    if not os.path.exists(store_path):
        return {"missing": True}

    flat = read_json(store_path)
    tallies = Counter()
    per_slot_machines = {}
    per_slot_durations = {}
    machine_mismatch = []    # (flat_idx, first mismatch position or None if length differs)
    duration_mismatch = []

    for fi in target_idxs:
        entry = flat[fi] if fi < len(flat) else None
        exp_m, exp_d = expected[fi]
        struct, length, machine_ok, dur_ok, got_m, got_d = check_entry(entry, exp_m, exp_d)
        tallies["target"] += 1
        tallies["struct_ok"] += int(struct)
        tallies["len_ok"] += int(length)
        tallies["machine_ok"] += int(machine_ok)
        tallies["duration_ok"] += int(dur_ok)
        if length:
            per_slot_machines[fi] = got_m
            per_slot_durations[fi] = got_d
        if length and not machine_ok:
            # first differing position
            pos = next((i for i in range(len(got_m)) if got_m[i] != exp_m[i]), None)
            machine_mismatch.append((fi, pos, got_m[pos] if pos is not None else None, exp_m[pos] if pos is not None else None))
        if length and not dur_ok:
            pos = next(
                (i for i in range(len(got_d)) if got_d[i] != f"{exp_d[i]} minutes"),
                None,
            )
            duration_mismatch.append((fi, pos, got_d[pos] if pos is not None else None, f"{exp_d[pos]} minutes" if pos is not None else None))
    return {
        "missing": False,
        "tallies": tallies,
        "per_slot_machines": per_slot_machines,
        "per_slot_durations": per_slot_durations,
        "machine_mismatch": machine_mismatch,
        "duration_mismatch": duration_mismatch,
    }


def cross_model_consensus(models, results, target_idxs):
    """For each slot, check whether all models with a length-valid entry agree
    on machine seq and duration seq. Slots where any model failed len_ok are
    reported separately (can't compare)."""
    machine_disagree = []   # (flat_idx, {model: seq})
    duration_disagree = []
    incomplete = []         # (flat_idx, [models_missing_len])
    all_agree_machine = 0
    all_agree_duration = 0

    for fi in target_idxs:
        got_m = {m: results[m]["per_slot_machines"].get(fi) for m in models}
        got_d = {m: results[m]["per_slot_durations"].get(fi) for m in models}
        missing = [m for m in models if got_m[m] is None]
        if missing:
            incomplete.append((fi, missing))
            continue
        seqs_m = list(got_m.values())
        if all(s == seqs_m[0] for s in seqs_m):
            all_agree_machine += 1
        else:
            machine_disagree.append((fi, got_m))
        seqs_d = list(got_d.values())
        if all(s == seqs_d[0] for s in seqs_d):
            all_agree_duration += 1
        else:
            duration_disagree.append((fi, got_d))
    return {
        "all_agree_machine": all_agree_machine,
        "all_agree_duration": all_agree_duration,
        "machine_disagree": machine_disagree,
        "duration_disagree": duration_disagree,
        "incomplete": incomplete,
    }


def run_canonical(
    store_path="preprocess/route_sheet.json",
    jssp_mapped_path="preprocess/jssp_mapped.json",
    machines_path="data/machines.json",
    target_instances=None,
    show=5,
):
    """Library entry point for pipeline scripts. Validate a single canonical
    route_sheet.json against a canonical jssp_mapped.json. Returns True iff every
    struct/len/machine/duration check passes for slots in ``target_instances``.

    ``target_instances`` accepts strings like "ta75" / "instance ta75" (or None
    for all ta71..ta80). Prints a compact summary and, on failure, the first
    ``show`` mismatches. Never raises."""
    if not os.path.exists(store_path):
        print(f"[validate_route_sheets] MISSING {store_path}")
        return False
    if not os.path.exists(jssp_mapped_path):
        print(f"[validate_route_sheets] MISSING {jssp_mapped_path}")
        return False

    if target_instances is None:
        targets = TARGET_INSTANCES
    else:
        targets = {t if t.startswith("instance ") else f"instance {t}" for t in target_instances}

    expected, flat_instance = build_expected(machines_path, jssp_mapped_path)
    target_idxs = [i for i, d in enumerate(flat_instance)
                   if d in targets and expected[i] is not None]
    result = per_model_scan("canonical", store_path, target_idxs, expected)
    if result.get("missing"):
        print(f"[validate_route_sheets] MISSING {store_path}")
        return False

    t = result["tallies"]
    n = t["target"]
    passed = (t["struct_ok"] == n and t["len_ok"] == n
              and t["machine_ok"] == n and t["duration_ok"] == n)
    tag = "OK" if passed else "FAIL"
    scope = "all" if target_instances is None else ",".join(sorted(targets))
    print(f"[validate_route_sheets/{scope}] {tag}  "
          f"struct={t['struct_ok']}/{n} len={t['len_ok']}/{n} "
          f"machine={t['machine_ok']}/{n} duration={t['duration_ok']}/{n}")
    if not passed:
        for fi, pos, got, exp in result["machine_mismatch"][:show]:
            print(f"    machine  flat_idx={fi} step={pos}  got={got!r}  expected={exp!r}")
        if len(result["machine_mismatch"]) > show:
            print(f"    ... {len(result['machine_mismatch']) - show} more machine mismatches")
        for fi, pos, got, exp in result["duration_mismatch"][:show]:
            print(f"    duration flat_idx={fi} step={pos}  got={got!r}  expected={exp!r}")
        if len(result["duration_mismatch"]) > show:
            print(f"    ... {len(result['duration_mismatch']) - show} more duration mismatches")
    return passed


def fmt_row(cells, widths):
    return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="Model names to validate (default: 4 LLM set).")
    ap.add_argument("--show", type=int, default=10,
                    help="Show up to N mismatch rows per category (default 10).")
    ap.add_argument("--machines", default="data/machines.json")
    ap.add_argument("--jssp-mapped", default="preprocess/jssp_mapped.json")
    ap.add_argument("--store-tmpl", default="preprocess/{model}/route_sheet.json",
                    help="Template with {model} placeholder for the per-model file.")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)

    expected, flat_instance = build_expected(args.machines, args.jssp_mapped)
    target_idxs = [i for i, d in enumerate(flat_instance)
                   if d in TARGET_INSTANCES and expected[i] is not None]
    total_target = len(target_idxs)
    excluded = sum(1 for i, d in enumerate(flat_instance)
                   if d in TARGET_INSTANCES and expected[i] is None)
    print(f"target slots (ta71-ta80, mappable): {total_target}"
          + (f"  [excluded {excluded} null-machine slots]" if excluded else ""))

    results = {}
    for m in args.models:
        path = args.store_tmpl.format(model=m)
        results[m] = per_model_scan(m, path, target_idxs, expected)

    # --- Layer 1 table --------------------------------------------------
    print()
    print("== Layer 1: per-model hard-constraint checks ==")
    header = ["model", "struct_ok", "len_ok", "machine_ok", "duration_ok",
              "machine_mism", "duration_mism"]
    widths = [16, 11, 8, 12, 13, 14, 15]
    print(fmt_row(header, widths))
    print(fmt_row(["-" * w for w in widths], widths))
    for m in args.models:
        r = results[m]
        if r.get("missing"):
            print(fmt_row([m, "(file missing)"] + [""] * 5, widths)); continue
        t = r["tallies"]
        row = [
            m,
            f"{t['struct_ok']}/{t['target']}",
            f"{t['len_ok']}/{t['target']}",
            f"{t['machine_ok']}/{t['target']}",
            f"{t['duration_ok']}/{t['target']}",
            len(r["machine_mismatch"]),
            len(r["duration_mismatch"]),
        ]
        print(fmt_row(row, widths))

    # --- Layer 1 detail -------------------------------------------------
    for m in args.models:
        r = results[m]
        if r.get("missing"): continue
        mm = r["machine_mismatch"]
        dm = r["duration_mismatch"]
        if not mm and not dm:
            continue
        print(f"\n-- [{m}] first mismatches --")
        for fi, pos, got, exp in mm[:args.show]:
            print(f"   machine  flat_idx={fi:5d}  step={pos}  got={got!r}  expected={exp!r}")
        if len(mm) > args.show:
            print(f"   ... {len(mm) - args.show} more machine mismatches")
        for fi, pos, got, exp in dm[:args.show]:
            print(f"   duration flat_idx={fi:5d}  step={pos}  got={got!r}  expected={exp!r}")
        if len(dm) > args.show:
            print(f"   ... {len(dm) - args.show} more duration mismatches")

    # --- Layer 3: cross-model consensus --------------------------------
    live_models = [m for m in args.models if not results[m].get("missing")]
    if len(live_models) < 2:
        print("\n(skipping cross-model consensus — need >= 2 models present)")
        return

    consensus = cross_model_consensus(live_models, results, target_idxs)
    print()
    print(f"== Layer 3: cross-model consensus (over {len(live_models)} models: {', '.join(live_models)}) ==")
    print(f"slots where all models agree on machine  sequence: "
          f"{consensus['all_agree_machine']}/{total_target}")
    print(f"slots where all models agree on duration sequence: "
          f"{consensus['all_agree_duration']}/{total_target}")
    if consensus["incomplete"]:
        print(f"slots skipped (some model has bad length): {len(consensus['incomplete'])}")
    if consensus["machine_disagree"]:
        print(f"\n-- machine-seq disagreements (first {args.show}) --")
        for fi, per_model in consensus["machine_disagree"][:args.show]:
            print(f"   flat_idx={fi}")
            for mm, seq in per_model.items():
                head = seq[:6] + (["..."] if len(seq) > 6 else [])
                print(f"     {mm:14s}  len={len(seq)}  {head}")
        if len(consensus["machine_disagree"]) > args.show:
            print(f"   ... {len(consensus['machine_disagree']) - args.show} more")
    if consensus["duration_disagree"]:
        print(f"\n-- duration-seq disagreements (first {args.show}) --")
        for fi, per_model in consensus["duration_disagree"][:args.show]:
            print(f"   flat_idx={fi}")
            for mm, seq in per_model.items():
                head = seq[:6] + (["..."] if len(seq) > 6 else [])
                print(f"     {mm:14s}  {head}")
        if len(consensus["duration_disagree"]) > args.show:
            print(f"   ... {len(consensus['duration_disagree']) - args.show} more")

    # --- Exit code: nonzero if any hard-constraint failed. --------------
    total_fail = sum(
        len(results[m]["machine_mismatch"]) + len(results[m]["duration_mismatch"])
        + (results[m]["tallies"]["target"] - results[m]["tallies"]["len_ok"])
        for m in live_models
    )
    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()
