#!/usr/bin/env bash
# One-off: qwen3-235b deterministically fails on flat_idx=7109 (returns 19 steps
# for a 20-step job, 5 attempts). Borrow qwen3-30b's version of that slot
# (verified 20 steps + machine/duration OK) as the tie-breaker source, then
# generate the corresponding order for qwen3-235b via --retry-missing.
set -uo pipefail
cd "D:/FB/TASE25-AutoDSL-Constraint"

export LLM_BASE_URL='http://localhost:8000/v1'
export OPENAI_API_KEY='sk-mi300-local'
export LLM_MODEL='qwen3-235b'
export LLM_MAX_WORKERS=4
unset HTTPS_PROXY HTTP_PROXY https_proxy http_proxy || true

# --- Step 1: patch route_sheet + reduce from qwen3-30b ---------------------
echo "==== [1/2] borrowing flat_idx=7109 from qwen3-30b -> qwen3-235b ===="
./.venv/Scripts/python.exe -u - <<'PY'
import json, os
from utils.util import read_json, write_json
from src.preprocess.RouteSheet import RouteSheet

IDX = 7109

src_flat = read_json("preprocess/qwen3-30b/route_sheet.json")
dst_flat = read_json("preprocess/qwen3-235b/route_sheet.json")
dst_flat[IDX] = src_flat[IDX]
write_json("preprocess/qwen3-235b/route_sheet.json", dst_flat)
print(f"[patch] wrote qwen3-30b's slot {IDX} into qwen3-235b/route_sheet.json")
print(f"  part_name={dst_flat[IDX].get('part_name')!r}  route_sheet_len={len(dst_flat[IDX].get('route_sheet', []))}")

# Rebuild qwen3-235b's route_sheet_reduce from patched flat.
rs = RouteSheet(
    machines_data_path="data/machines.json",
    jssp_data_path="data/jssp_data.json",
    arrange_path="data/arrange.json",
    jssp_mapped_path="preprocess/jssp_mapped.json",
    route_sheet_store_path="preprocess/qwen3-235b/route_sheet.json",
    route_sheet_reduce_path="preprocess/qwen3-235b/route_sheet_reduce.json",
)
rs.route_sheet_reduce()
PY

# --- Step 2: generate the missing order via --retry-missing ---------------
echo ""
echo "==== [2/2] generating missing order for the patched slot ===="
export ORDERS_OUTPUT_DIR="preprocess/qwen3-235b/orders"
./.venv/Scripts/python.exe -u generate_orders.py \
  --reduce preprocess/qwen3-235b/route_sheet_reduce.json \
  --retry-missing 2>&1 | grep -E '^\[(START|DONE|SKIP|EMPTY)\]|Saved' | head -20

# --- Verify ---------------------------------------------------------------
echo ""
echo "==== verify ===="
./.venv/Scripts/python.exe -u scripts/validate_route_sheets.py --models qwen3-235b --show 3 2>&1 | head -10
echo ""
./.venv/Scripts/python.exe -u scripts/validate_orders.py --models qwen3-235b --show 3 2>&1 | head -15