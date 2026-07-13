"""Cross-check generated orders against the canonical route_sheet_reduce input.

Covers:
  Layer 1 — per-model hard constraints
    * struct_ok    : entry is dict with non-empty part_name (str) and steps (list of str)
    * part_name_ok : order.part_name == route_sheet_reduce[.].part_name
    * len_ok       : len(order.steps) == len(input.route_sheet)
    * duration_ok  : each step[j] contains the expected "<N> minutes?" pattern
                     where N == int(input.route_sheet[j].duration)
    * strict_dur_ok: same but requires plural "minutes" (matches prompt's exact wording)

  Layer 3 — cross-model consensus (over slots where all models are len_ok)
    * part_name_agree     : all N models produce the same part_name
    * steps_len_agree     : all N models produce the same steps length
    * duration_seq_agree  : all N models mention the same duration integers,
                            in the same order (first "<int> minutes?" per step)

Skips slots where the canonical route_sheet_reduce entry is empty (upstream
create_route_sheet never generated one).

Usage:
    python scripts/validate_orders.py
    python scripts/validate_orders.py --models qwen3-4b qwen3-30b
    python scripts/validate_orders.py --show 30
    python scripts/validate_orders.py --root-tmpl "preprocess/orders_{model}"
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

DEFAULT_MODELS = ["deepseek-v3", "qwen3-235b", "qwen3-30b", "qwen3-4b"]
INSTANCES = [f"ta{n}" for n in range(71, 81)]
TARGET_INSTANCE_DESCS = {f"instance {ta}" for ta in INSTANCES}
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Any integer followed by "minute" or "minutes" (case-insensitive).
# Accepts both "5 minutes" and adjective form "5-minute" (as in "a 2-minute cycle").
DUR_ANY_RE     = re.compile(r"\b(\d+)[\s\-]minutes?\b", re.IGNORECASE)
# Strict: only plural "minutes" — matches the prompt's CRITICAL UNIT RULE.
# NOTE: generate_synthetic_data.txt actually allows "1 minute" (singular) for
# duration=1, so this metric is informational only — do not treat as a defect.
DUR_STRICT_RE  = re.compile(r"\b(\d+)\s+minutes\b")


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_int_prefix(s):
    """Extract leading integer from a string like '52 minutes' -> 52. None if missing."""
    if not isinstance(s, str):
        return None
    m = re.match(r"\s*(\d+)", s)
    return int(m.group(1)) if m else None


def build_expected(reduce_path):
    """Return dict (ta, idx) -> {'part_name': str, 'durations': [int, ...]}
    from the canonical route_sheet_reduce.json."""
    reduce = read_json(reduce_path)
    expected = {}
    for domain in reduce:
        if not domain:
            continue
        inst_desc = domain[0].get("instance_description")
        if inst_desc not in TARGET_INSTANCE_DESCS:
            continue
        ta = inst_desc.replace("instance ", "")
        for idx, entry in enumerate(domain):
            if not (isinstance(entry, dict) and entry.get("route_sheet")):
                continue
            rs = entry["route_sheet"]
            durs = [parse_int_prefix(step.get("duration")) for step in rs]
            expected[(ta, idx)] = {
                "part_name": entry.get("part_name"),
                "durations": durs,
            }
    return expected


def load_model_orders(model, root_tmpl):
    """Return dict ta -> list-of-orders, or missing_paths list."""
    out, missing = {}, []
    for ta in INSTANCES:
        p = os.path.join(root_tmpl.format(model=model), ta, "orders.json")
        if not os.path.exists(p):
            missing.append(p); continue
        out[ta] = read_json(p)
    return out, missing


def extract_first_duration(step_text):
    """First integer with 'minute(s)' in step; None if not present."""
    if not isinstance(step_text, str):
        return None
    m = DUR_ANY_RE.search(step_text)
    return int(m.group(1)) if m else None


def check_order(order, exp_part_name, exp_durations):
    """Layer-1 checks for one order. Returns (flags dict, extracted list)."""
    flags = {"struct_ok": False, "part_name_ok": False, "len_ok": False,
             "duration_ok": False, "strict_dur_ok": False}
    if not (isinstance(order, dict) and isinstance(order.get("part_name"), str)
            and order["part_name"] and isinstance(order.get("steps"), list)
            and all(isinstance(s, str) for s in order["steps"])):
        return flags, None
    flags["struct_ok"] = True
    flags["part_name_ok"] = (order["part_name"] == exp_part_name)
    if len(order["steps"]) != len(exp_durations):
        return flags, None
    flags["len_ok"] = True

    # Per-step duration checks.
    all_ok = True
    strict_ok = True
    extracted_first = []
    for step, exp in zip(order["steps"], exp_durations):
        extracted_first.append(extract_first_duration(step))
        if exp is None:
            continue
        # Loose: "<exp> minute(s)" appears at least once
        loose_hit = any(int(x) == exp for x in DUR_ANY_RE.findall(step))
        if not loose_hit:
            all_ok = False; strict_ok = False
            continue
        # Strict: "<exp> minutes" (plural s required)
        strict_hit = any(int(x) == exp for x in DUR_STRICT_RE.findall(step))
        if not strict_hit:
            strict_ok = False
    flags["duration_ok"] = all_ok
    flags["strict_dur_ok"] = strict_ok
    return flags, extracted_first


def per_model_scan(model, orders_by_ta, expected, missing_paths):
    """Layer-1 scan for one model. Returns tallies + per-slot dur seq for consensus."""
    if missing_paths:
        return {"missing": missing_paths}

    tallies = Counter()
    duration_mismatch = []   # (ta, idx, [(step_j, expected, extracted_first)])
    partname_mismatch = []   # (ta, idx, got, expected)
    len_mismatch = []        # (ta, idx, got, expected)
    struct_bad = []          # (ta, idx)
    per_slot_partname = {}
    per_slot_len = {}
    per_slot_durseq = {}

    for (ta, idx), exp in expected.items():
        arr = orders_by_ta.get(ta)
        if arr is None or idx >= len(arr):
            tallies["missing"] += 1; continue
        order = arr[idx]
        flags, extracted = check_order(order, exp["part_name"], exp["durations"])
        tallies["target"] += 1
        for k, v in flags.items():
            tallies[k] += int(v)

        if not flags["struct_ok"]:
            struct_bad.append((ta, idx)); continue

        # for consensus we still record even on partial failure
        per_slot_partname[(ta, idx)] = order.get("part_name")
        per_slot_len[(ta, idx)] = len(order["steps"]) if isinstance(order.get("steps"), list) else None
        per_slot_durseq[(ta, idx)] = extracted  # None if len_ok failed

        if not flags["part_name_ok"]:
            partname_mismatch.append((ta, idx, order.get("part_name"), exp["part_name"]))
        if not flags["len_ok"]:
            len_mismatch.append((ta, idx, len(order.get("steps") or []), len(exp["durations"])))
        if flags["len_ok"] and not flags["duration_ok"]:
            bad = []
            for j, (step, want, got) in enumerate(zip(order["steps"], exp["durations"], extracted or [])):
                if want is None: continue
                if not any(int(x) == want for x in DUR_ANY_RE.findall(step)):
                    bad.append((j, want, got))
            duration_mismatch.append((ta, idx, bad))

    return {
        "missing": None,
        "tallies": tallies,
        "duration_mismatch": duration_mismatch,
        "partname_mismatch": partname_mismatch,
        "len_mismatch": len_mismatch,
        "struct_bad": struct_bad,
        "per_slot_partname": per_slot_partname,
        "per_slot_len": per_slot_len,
        "per_slot_durseq": per_slot_durseq,
    }


def cross_model_consensus(models, results, expected):
    partname_disagree = []
    len_disagree = []
    durseq_disagree = []
    incomplete = []
    all_agree_pn = 0
    all_agree_len = 0
    all_agree_dur = 0
    total = 0

    for key in expected:
        total += 1
        pns  = {m: results[m]["per_slot_partname"].get(key) for m in models}
        lens = {m: results[m]["per_slot_len"].get(key) for m in models}
        durs = {m: results[m]["per_slot_durseq"].get(key) for m in models}
        missing = [m for m in models if pns[m] is None or durs[m] is None]
        if missing:
            incomplete.append((key, missing)); continue

        pn_vals = list(pns.values())
        if all(v == pn_vals[0] for v in pn_vals):
            all_agree_pn += 1
        else:
            partname_disagree.append((key, pns))

        len_vals = list(lens.values())
        if all(v == len_vals[0] for v in len_vals):
            all_agree_len += 1
        else:
            len_disagree.append((key, lens))

        dur_vals = list(durs.values())
        if all(v == dur_vals[0] for v in dur_vals):
            all_agree_dur += 1
        else:
            durseq_disagree.append((key, durs))
    return {
        "total": total,
        "all_agree_pn": all_agree_pn,
        "all_agree_len": all_agree_len,
        "all_agree_dur": all_agree_dur,
        "partname_disagree": partname_disagree,
        "len_disagree": len_disagree,
        "durseq_disagree": durseq_disagree,
        "incomplete": incomplete,
    }


def fmt_row(cells, widths):
    return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))


def run_canonical(
    reduce_path="preprocess/route_sheet_reduce.json",
    orders_root="preprocess/orders",
    target_instances=None,
    show=5,
):
    """Library entry point for pipeline scripts. Validate canonical
    ``orders/ta{71..80}/orders.json`` against a canonical
    ``route_sheet_reduce.json``. Returns True iff every struct/part_name/len/
    duration hard check passes for slots in ``target_instances``.

    ``target_instances`` accepts strings like "ta75" (with or without the
    "instance " prefix); None means all ta71..ta80. Prints a compact summary
    and first ``show`` mismatches on failure. Never raises.

    ``strict_dur_ok`` is intentionally excluded from the pass criterion — the
    generate_synthetic_step prompt allows singular "1 minute" for duration=1,
    so the strict check is informational only."""
    if not os.path.exists(reduce_path):
        print(f"[validate_orders] MISSING {reduce_path}")
        return False

    expected = build_expected(reduce_path)
    if target_instances is not None:
        want = {(t.replace("instance ", "") if t.startswith("instance ") else t)
                for t in target_instances}
        expected = {(ta, idx): v for (ta, idx), v in expected.items() if ta in want}
    if not expected:
        print(f"[validate_orders] no target slots to check")
        return True

    orders_by_ta = {}
    missing_paths = []
    tas_needed = sorted({ta for (ta, _) in expected})
    for ta in tas_needed:
        p = os.path.join(orders_root, ta, "orders.json")
        if not os.path.exists(p):
            missing_paths.append(p)
        else:
            orders_by_ta[ta] = read_json(p)
    if missing_paths:
        for p in missing_paths:
            print(f"[validate_orders] MISSING {p}")
        return False

    result = per_model_scan("canonical", orders_by_ta, expected, missing_paths=[])
    t = result["tallies"]
    n = t["target"]
    passed = (t["struct_ok"] == n and t["part_name_ok"] == n
              and t["len_ok"] == n and t["duration_ok"] == n)
    tag = "OK" if passed else "FAIL"
    scope = "all" if target_instances is None else ",".join(tas_needed)
    print(f"[validate_orders/{scope}] {tag}  "
          f"struct={t['struct_ok']}/{n} part_name={t['part_name_ok']}/{n} "
          f"len={t['len_ok']}/{n} duration={t['duration_ok']}/{n} "
          f"strict_dur(info)={t['strict_dur_ok']}/{n}")
    if not passed:
        for ta, idx in result["struct_bad"][:show]:
            print(f"    struct     {ta}[{idx}]  entry not a well-formed order dict")
        if len(result["struct_bad"]) > show:
            print(f"    ... {len(result['struct_bad']) - show} more struct issues")
        for ta, idx, got, exp in result["partname_mismatch"][:show]:
            print(f"    part_name  {ta}[{idx}]  got={got!r}  expected={exp!r}")
        if len(result["partname_mismatch"]) > show:
            print(f"    ... {len(result['partname_mismatch']) - show} more part_name mismatches")
        for ta, idx, got, exp in result["len_mismatch"][:show]:
            print(f"    steps_len  {ta}[{idx}]  got={got}  expected={exp}")
        if len(result["len_mismatch"]) > show:
            print(f"    ... {len(result['len_mismatch']) - show} more length mismatches")
        for ta, idx, bad_steps in result["duration_mismatch"][:show]:
            head = bad_steps[:3]
            more = f"  (+{len(bad_steps)-3} more)" if len(bad_steps) > 3 else ""
            samp = ", ".join(f"step{j}: want {w!r}, got_first={g!r}" for j, w, g in head)
            print(f"    duration   {ta}[{idx}]  {samp}{more}")
        if len(result["duration_mismatch"]) > show:
            print(f"    ... {len(result['duration_mismatch']) - show} more duration mismatches")
    return passed


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--show", type=int, default=10,
                    help="Show up to N mismatch rows per category (default 10).")
    ap.add_argument("--reduce-tmpl", default="preprocess/{model}/route_sheet_reduce.json",
                    help="Per-model route_sheet_reduce template. Self-consistency: each "
                         "model's orders validate against its OWN route_sheet.")
    ap.add_argument("--root-tmpl", default="preprocess/{model}/orders",
                    help="Per-model orders directory template.")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)

    # Build per-model expected — each model's orders validate against its own
    # route_sheet_reduce. Take the first model's expected as the reference for
    # Layer-3 cross-model comparisons (durations are model-independent since
    # they echo the input jssp).
    expected_by_model = {}
    for m in args.models:
        rp = args.reduce_tmpl.format(model=m)
        if not os.path.exists(rp):
            print(f"WARNING: {rp} missing — will skip {m}", file=sys.stderr)
            expected_by_model[m] = None
        else:
            expected_by_model[m] = build_expected(rp)

    ref_expected = next((e for e in expected_by_model.values() if e), None)
    if ref_expected is None:
        print("No reduce files present — nothing to validate.")
        sys.exit(2)
    print(f"expected target slots (10 instances x 100): {len(ref_expected)}")

    results = {}
    for m in args.models:
        exp = expected_by_model[m]
        if exp is None:
            results[m] = {"missing": [args.reduce_tmpl.format(model=m)]}
            continue
        orders_by_ta, missing = load_model_orders(m, args.root_tmpl)
        results[m] = per_model_scan(m, orders_by_ta, exp, missing)

    # --- Layer 1 table --------------------------------------------------
    print()
    print("== Layer 1: per-model hard-constraint checks ==")
    header = ["model", "struct_ok", "part_name_ok", "len_ok", "duration_ok", "strict_dur_ok"]
    widths = [16, 11, 14, 10, 13, 15]
    print(fmt_row(header, widths))
    print(fmt_row(["-" * w for w in widths], widths))
    for m in args.models:
        r = results[m]
        if r.get("missing"):
            print(fmt_row([m, "(files missing: " + str(len(r["missing"])) + ")"] + [""]*5, widths)); continue
        t = r["tallies"]
        row = [
            m,
            f"{t['struct_ok']}/{t['target']}",
            f"{t['part_name_ok']}/{t['target']}",
            f"{t['len_ok']}/{t['target']}",
            f"{t['duration_ok']}/{t['target']}",
            f"{t['strict_dur_ok']}/{t['target']}",
        ]
        print(fmt_row(row, widths))

    # --- Layer 1 detail -------------------------------------------------
    for m in args.models:
        r = results[m]
        if r.get("missing"): continue
        sb, pm, lm, dm = r["struct_bad"], r["partname_mismatch"], r["len_mismatch"], r["duration_mismatch"]
        if not (sb or pm or lm or dm):
            continue
        print(f"\n-- [{m}] first mismatches --")
        for ta, idx in sb[:args.show]:
            print(f"   struct     {ta}[{idx}]  entry not a well-formed order dict")
        if len(sb) > args.show:
            print(f"   ... {len(sb) - args.show} more struct issues")
        for ta, idx, got, exp in pm[:args.show]:
            print(f"   part_name  {ta}[{idx}]  got={got!r:40s}  expected={exp!r}")
        if len(pm) > args.show:
            print(f"   ... {len(pm) - args.show} more part_name mismatches")
        for ta, idx, got, exp in lm[:args.show]:
            print(f"   steps_len  {ta}[{idx}]  got={got}  expected={exp}")
        if len(lm) > args.show:
            print(f"   ... {len(lm) - args.show} more length mismatches")
        for ta, idx, bad_steps in dm[:args.show]:
            head = bad_steps[:3]
            more = f"  (+{len(bad_steps)-3} more)" if len(bad_steps) > 3 else ""
            samp = ", ".join(f"step{j}: want {w!r}, got_first={g!r}" for j,w,g in head)
            print(f"   duration   {ta}[{idx}]  {samp}{more}")
        if len(dm) > args.show:
            print(f"   ... {len(dm) - args.show} more duration mismatches")

    # --- Layer 3 consensus ---------------------------------------------
    live = [m for m in args.models if not results[m].get("missing")]
    if len(live) < 2:
        print("\n(skipping cross-model consensus — need >= 2 models present)")
        return

    con = cross_model_consensus(live, results, ref_expected)
    print()
    print(f"== Layer 3: cross-model consensus (over {len(live)} models: {', '.join(live)}) ==")
    print(f"slots where all models agree on part_name    : {con['all_agree_pn']}/{con['total']}")
    print(f"slots where all models agree on steps length : {con['all_agree_len']}/{con['total']}")
    print(f"slots where all models agree on duration seq : {con['all_agree_dur']}/{con['total']}")
    if con["incomplete"]:
        print(f"slots skipped (some model missing/bad struct): {len(con['incomplete'])}")

    if con["partname_disagree"]:
        print(f"\n-- part_name disagreements (first {args.show}) --")
        for (ta, idx), per_m in con["partname_disagree"][:args.show]:
            print(f"   {ta}[{idx}]  " + " | ".join(f"{m}={v!r}" for m,v in per_m.items()))
        if len(con["partname_disagree"]) > args.show:
            print(f"   ... {len(con['partname_disagree']) - args.show} more")

    if con["len_disagree"]:
        print(f"\n-- steps_len disagreements (first {args.show}) --")
        for (ta, idx), per_m in con["len_disagree"][:args.show]:
            print(f"   {ta}[{idx}]  " + " | ".join(f"{m}={v}" for m,v in per_m.items()))
        if len(con["len_disagree"]) > args.show:
            print(f"   ... {len(con['len_disagree']) - args.show} more")

    if con["durseq_disagree"]:
        print(f"\n-- duration_seq disagreements (first {args.show}) --")
        for (ta, idx), per_m in con["durseq_disagree"][:args.show]:
            # Find first differing step.
            seqs = list(per_m.values())
            L = min(len(s or []) for s in seqs)
            first_diff = None
            for j in range(L):
                vals = [(s or [None])[j] for s in seqs]
                if any(v != vals[0] for v in vals):
                    first_diff = j; break
            print(f"   {ta}[{idx}]  first_diff_step={first_diff}")
            for mm, seq in per_m.items():
                if seq is None:
                    print(f"     {mm:14s}  <no seq>")
                elif first_diff is not None:
                    lo = max(0, first_diff - 1); hi = min(len(seq), first_diff + 2)
                    print(f"     {mm:14s}  ...{seq[lo:hi]} @ step {lo}..{hi-1}")
                else:
                    print(f"     {mm:14s}  len={len(seq)}")
        if len(con["durseq_disagree"]) > args.show:
            print(f"   ... {len(con['durseq_disagree']) - args.show} more")

    # Exit code: nonzero if any hard constraint failed.
    hard_fail = sum(
        (results[m]["tallies"]["target"] - results[m]["tallies"]["duration_ok"])
        + (results[m]["tallies"]["target"] - results[m]["tallies"]["part_name_ok"])
        + (results[m]["tallies"]["target"] - results[m]["tallies"]["len_ok"])
        + (results[m]["tallies"]["target"] - results[m]["tallies"]["struct_ok"])
        for m in live
    )
    sys.exit(0 if hard_fail == 0 else 1)


if __name__ == "__main__":
    main()
