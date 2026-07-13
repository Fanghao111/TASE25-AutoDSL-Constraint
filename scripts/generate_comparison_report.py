"""Scan outputs/Evaluation/FB-models_*/ and DSL-models_*/ and emit a comparison report.

Each FB-models_<model>/ and DSL-models_<model>/ subdir is the eval output for one
pipeline × model combo (set by run_fb_pipeline_4llm.ps1 / run_dsl_pipeline_4llm.ps1
via FB_EVAL_LABEL / DSL_EVAL_LABEL). Aggregates per-instance lists into means
and lays them out side-by-side.

makespan_ratio uses Taillard BKS as denominator (not the GT-derived makespan,
which is confounded by GroundTruth's semantic-precondition precedence logic).
For instances where arrange.json merges machines (ta75, ta78), the LB1 lower
bound is higher than BKS by construction, so a ratio > 1 is expected there.
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone

EVAL_ROOT = "outputs/Evaluation"
OUTPUTS_ROOT = "outputs"
REPORT_PATH = os.path.join(EVAL_ROOT, "FB-models-comparison-report.md")
PLACEHOLDER = "—"
FB_PREFIX = "FB-models_"
DSL_PREFIX = "DSL-models_"

INSTANCES = [f"ta{i}" for i in range(71, 81)]
EXPECTED_TASKS = 2000  # 100 jobs × 20 tasks per Taillard 100×20 instance

# Taillard BKS (best-known solution) for ta71-ta80, 100×20 JSSP.
# For instances where arrange.json merges machines (ta75: 18M, ta78: 15M) the
# physical lower bound LB1 is higher than BKS — expect ratio > 1 there even
# for an optimal solver.
TAILLARD_BKS = {
    "ta71": 5464,
    "ta72": 5181,
    "ta73": 5568,
    "ta74": 5339,
    "ta75": 5392,
    "ta76": 5342,
    "ta77": 5436,
    "ta78": 5394,
    "ta79": 5358,
    "ta80": 5183,
}


def _load(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_txt(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _mean(xs):
    xs = [x for x in (xs or []) if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def _fmt(x, digits=4):
    return PLACEHOLDER if x is None else f"{x:.{digits}f}"


def _pct(x, digits=1):
    return PLACEHOLDER if x is None else f"{x * 100:.{digits}f}%"


def _makespan_ratio_vs_bks(makespan_path_tmpl: str) -> tuple[float | None, int]:
    """Mean(makespan / BKS) over 10 instances; returns (mean, n_present)."""
    ratios = []
    for tag in INSTANCES:
        p = makespan_path_tmpl.format(tag=tag)
        raw = _read_txt(p)
        if raw is None:
            continue
        try:
            ms = float(raw)
        except ValueError:
            continue
        if ms <= 0:
            continue
        ratios.append(ms / TAILLARD_BKS[tag])
    return (_mean(ratios), len(ratios))


def _opcount_stats(or_matrix_path_tmpl: str) -> tuple[float | None, float | None, int]:
    """Mean(total tasks) and mean(task coverage vs 2000); returns (mean_tasks, mean_cov, n_present)."""
    tasks_list, cov_list = [], []
    for tag in INSTANCES:
        p = or_matrix_path_tmpl.format(tag=tag)
        d = _load(p)
        if not isinstance(d, list):
            continue
        n_tasks = sum(len(job) for job in d if isinstance(job, list))
        tasks_list.append(n_tasks)
        cov_list.append(n_tasks / EXPECTED_TASKS)
    return (_mean(tasks_list), _mean(cov_list), len(tasks_list))


def _discover_models() -> list[str]:
    if not os.path.isdir(EVAL_ROOT):
        return []
    out = []
    for name in sorted(os.listdir(EVAL_ROOT)):
        full = os.path.join(EVAL_ROOT, name)
        if os.path.isdir(full) and name.startswith(FB_PREFIX):
            out.append(name[len(FB_PREFIX):])
    return out


def _discover_dsl_models() -> list[str]:
    if not os.path.isdir(EVAL_ROOT):
        return []
    out = []
    for name in sorted(os.listdir(EVAL_ROOT)):
        full = os.path.join(EVAL_ROOT, name)
        if os.path.isdir(full) and name.startswith(DSL_PREFIX):
            out.append(name[len(DSL_PREFIX):])
    return out


def _row_bleu_rouge(model: str, eval_type: str) -> tuple[float | None, float | None, int]:
    base = os.path.join(EVAL_ROOT, f"FB-models_{model}")
    bleu = _load(os.path.join(base, f"{eval_type}_fb_bleu.json"))
    rouge = _load(os.path.join(base, f"{eval_type}_fb_rouge.json"))
    bleu_mean = _mean(bleu) if bleu is not None else None
    rouge_f1_mean = _mean(rouge.get("F1") if isinstance(rouge, dict) else None)
    n = len(bleu) if isinstance(bleu, list) else 0
    return bleu_mean, rouge_f1_mean, n


def _row_graph(model: str) -> tuple[float | None, float | None, float | None, float | None, int, int]:
    """Return acc / err / makespan_ratio_vs_bks / mean_tasks / mean_cov / n."""
    base = os.path.join(EVAL_ROOT, f"FB-models_{model}")
    data = _load(os.path.join(base, "graph_fb.json"))
    if not isinstance(data, dict):
        return None, None, None, None, None, 0
    acc = _mean(data.get("accuracy_rate"))
    err = _mean(data.get("runtime_err_rate"))
    n = max(
        len(data.get("accuracy_rate") or []),
        len(data.get("runtime_err_rate") or []),
    )
    ms_ratio, _ = _makespan_ratio_vs_bks(
        os.path.join(OUTPUTS_ROOT, "FB-models", model, "instance {tag}", "s6_makespan.txt")
    )
    mean_tasks, mean_cov, _ = _opcount_stats(
        os.path.join(OUTPUTS_ROOT, "FB-models", model, "instance {tag}", "s5_or_matrix.json")
    )
    return acc, err, ms_ratio, mean_tasks, mean_cov, n


def _row_dsl_bleu_rouge(model: str, eval_type: str) -> tuple[float | None, float | None, int]:
    base = os.path.join(EVAL_ROOT, f"{DSL_PREFIX}{model}")
    bleu = _load(os.path.join(base, f"{eval_type}_dsl_bleu.json"))
    rouge = _load(os.path.join(base, f"{eval_type}_dsl_rouge.json"))
    bleu_mean = _mean(bleu) if bleu is not None else None
    rouge_f1_mean = _mean(rouge.get("F1") if isinstance(rouge, dict) else None)
    n = len(bleu) if isinstance(bleu, list) else 0
    return bleu_mean, rouge_f1_mean, n


def _row_dsl_graph(model: str) -> tuple[float | None, float | None, float | None, float | None, int, int]:
    base = os.path.join(EVAL_ROOT, f"{DSL_PREFIX}{model}")
    data = _load(os.path.join(base, "graph_dsl.json"))
    if not isinstance(data, dict):
        return None, None, None, None, None, 0
    acc = _mean(data.get("accuracy_rate"))
    err = _mean(data.get("runtime_err_rate"))
    n = max(
        len(data.get("accuracy_rate") or []),
        len(data.get("runtime_err_rate") or []),
    )
    ms_ratio, _ = _makespan_ratio_vs_bks(
        os.path.join(OUTPUTS_ROOT, "DSL-models", model, "instance {tag}", "makespan.txt")
    )
    mean_tasks, mean_cov, _ = _opcount_stats(
        os.path.join(OUTPUTS_ROOT, "DSL-models", model, "instance {tag}", "or_matrix.json")
    )
    return acc, err, ms_ratio, mean_tasks, mean_cov, n


def main() -> None:
    models = _discover_models()
    dsl_models = _discover_dsl_models()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    lines: list[str] = []
    lines.append("# FB Pipeline — 模型对比报告")
    lines.append("")
    lines.append(f"生成时间：{now}")
    lines.append("评测 instance：ta71–ta80（Taillard 100×20 JSSP benchmark）")
    lines.append("评测 evaluation type：production_plan / route_sheet / graph"
                 "（每个模型跑一次 --type=full 得到这三项）")
    lines.append("")
    lines.append("**makespan_ratio 用 Taillard BKS 作为分母**："
                 "`ratio = pipeline_makespan / BKS`。BKS 是 20 台机器上未经合并的已知最优解；"
                 "`data/arrange.json` 会把 ta75、ta78 里的部分机器合并（分别到 18、15 台），"
                 "合并后的物理下界 LB1 高于 BKS，因此这两个 instance 的 ratio 天然大于 1。")
    lines.append("")
    lines.append("**opcount 覆盖率**：`coverage = actual_tasks / 2000`（Taillard 100×20 每 instance 应有 2000 tasks）。"
                 "小于 1 表示 pipeline 生成的 or_matrix 丢失了 task。")
    lines.append("")
    if not models and not dsl_models:
        lines.append("⚠️ 未在 `outputs/Evaluation/` 下找到任何 `FB-models_*` 或 `DSL-models_*` 子目录。")
        os.makedirs(EVAL_ROOT, exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[report] no models found; wrote stub to {REPORT_PATH}")
        return

    if models:
        lines.append(f"FB 覆盖模型：{', '.join(models)}")
    if dsl_models:
        lines.append(f"DSL 覆盖模型：{', '.join(dsl_models)}")
    lines.append("")

    # Union of models present in either FB or DSL eval outputs (stable order).
    all_models: list[str] = []
    for m in models:
        if m not in all_models:
            all_models.append(m)
    for m in dsl_models:
        if m not in all_models:
            all_models.append(m)

    lines.append("## production_plan（NL → production plan）— BLEU / ROUGE-L F1")
    lines.append("")
    lines.append("| 方法 | BLEU 均值 | ROUGE-L F1 均值 | n |")
    lines.append("|------|-----------|-----------------|---|")
    for m in all_models:
        if m in models:
            b, r, n = _row_bleu_rouge(m, "production_plan")
            lines.append(f"| FB / {m} | {_fmt(b)} | {_fmt(r)} | {n} |")
        if m in dsl_models:
            b, r, n = _row_dsl_bleu_rouge(m, "production_plan")
            lines.append(f"| DSL / {m} | {_fmt(b)} | {_fmt(r)} | {n} |")
    lines.append("")

    lines.append("## route_sheet（NL → route sheet）— BLEU / ROUGE-L F1")
    lines.append("")
    lines.append("| 方法 | BLEU 均值 | ROUGE-L F1 均值 | n |")
    lines.append("|------|-----------|-----------------|---|")
    for m in all_models:
        if m in models:
            b, r, n = _row_bleu_rouge(m, "route_sheet")
            lines.append(f"| FB / {m} | {_fmt(b)} | {_fmt(r)} | {n} |")
        if m in dsl_models:
            b, r, n = _row_dsl_bleu_rouge(m, "route_sheet")
            lines.append(f"| DSL / {m} | {_fmt(b)} | {_fmt(r)} | {n} |")
    lines.append("")

    lines.append("## graph（NL → OR matrix）— accuracy / err_rate / makespan_ratio(vs BKS) / task 数 / 覆盖率")
    lines.append("")
    lines.append("| 方法 | accuracy 均值 | err_rate 均值 | makespan_ratio (vs BKS) | 平均 task 数 | opcount 覆盖率 | n |")
    lines.append("|------|---------------|---------------|-------------------------|-------------|----------------|---|")
    for m in all_models:
        if m in models:
            acc, err, ms, mt, cov, n = _row_graph(m)
            lines.append(f"| FB / {m} | {_fmt(acc)} | {_fmt(err)} | {_fmt(ms)} | {_fmt(mt, 1) if mt is not None else PLACEHOLDER} | {_pct(cov)} | {n} |")
        if m in dsl_models:
            acc, err, ms, mt, cov, n = _row_dsl_graph(m)
            lines.append(f"| DSL / {m} | {_fmt(acc)} | {_fmt(err)} | {_fmt(ms)} | {_fmt(mt, 1) if mt is not None else PLACEHOLDER} | {_pct(cov)} | {n} |")
    lines.append("")

    lines.append("## 附：原始 JSON 路径")
    lines.append("")
    for m in models:
        base = f"outputs/Evaluation/{FB_PREFIX}{m}"
        lines.append(f"- `{base}/production_plan_fb_bleu.json`、`{base}/production_plan_fb_rouge.json`、"
                     f"`{base}/route_sheet_fb_bleu.json`、`{base}/route_sheet_fb_rouge.json`、"
                     f"`{base}/graph_fb.json`")
    for m in dsl_models:
        base = f"outputs/Evaluation/{DSL_PREFIX}{m}"
        lines.append(f"- `{base}/production_plan_dsl_bleu.json`、`{base}/production_plan_dsl_rouge.json`、"
                     f"`{base}/route_sheet_dsl_bleu.json`、`{base}/route_sheet_dsl_rouge.json`、"
                     f"`{base}/graph_dsl.json`")
    lines.append("")

    os.makedirs(EVAL_ROOT, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] wrote {REPORT_PATH} (FB: {', '.join(models) or '-'} | DSL: {', '.join(dsl_models) or '-'})")


if __name__ == "__main__":
    main()
