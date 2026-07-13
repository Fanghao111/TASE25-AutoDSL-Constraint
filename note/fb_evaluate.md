# FB Pipeline 与 Evaluation 对照速查

FB pipeline 有 s1..s7 七步(其中 s1/s2/s4 是 LLM 调用,其余是确定性代码 / OR-Tools)。
`src/evaluation/fb_evaluation.py` 提供 9 项评测,把每一步都能单独归因。

**运行入口**

```bash
# 单项
FB_EVAL_LABEL=FB-models/<model> python main.py --mode fb_evaluation --fb_evaluation_type extract
# 全跑 9 项
FB_EVAL_LABEL=FB-models/<model> python main.py --mode fb_evaluation --fb_evaluation_type all
```

`FB_EVAL_LABEL` 决定读哪个 FB pipeline 目录,写哪个 evaluation 输出目录(约定与现有 `Evaluation` 一致)。

## Pipeline↔评测映射

| 步骤 | 类型 | 输出 artifact | 对应 evaluation_type | 考察什么 |
|---|---|---|---|---|
| s1 extract | LLM | `s1_extracted.json` | **`extract`** (新) | 单步 s1 的字段级 NL→JSON 抽取能力 |
| s2 verify | LLM | `s2_verified.json` | **`verify`** (新) | s2 相对 s1 的净修复(同尺度 F1 对比 + 逐字段 fix/break) |
| s3 format | code | `s3_formatted.json` | (由 `normalize` 间接覆盖) | 纯确定性,不单独评 |
| s4 normalize | LLM | `s4_mappings.json`, `s4_normalized.json` | **`normalize`** (统计) + **`route_sheet_fieldlevel`** (新) | 归一化前后词汇量收缩 + s4 输出的字段级 F1 |
| s5 build_graph | code | `s5_or_matrix.json`, `s5_machines.json` | (由 `graph` / `graph_from_gt` 覆盖) | 纯确定性,不单独评 |
| s6 solve | OR-Tools | `s6_assigned_jobs.json`, `s6_makespan.txt` | (间接由 `graph` / `graph_from_gt` 反映) | CP-SAT,不评 |
| s7 ground | code | `s7_production_plan.json` | (间接由 `production_plan` / `ground_from_gt`) | 纯反投影,不单独评 |
| — | 端到端 | s7 output | `production_plan` (老) | s1..s7 累积质量,BLEU + 字段级 ROUGE |
| — | 端到端 legacy | s4 输出转 route_sheet | `route_sheet` (老) | s1+s2+s3+s4 累积,BLEU + 自定义 ROUGE-L(与 `route_sheet_fieldlevel` 尺度不同) |
| — | 端到端 | s5 输出 | `graph` (老) | s1..s5 累积:or_matrix 依赖关系 IoU / err_rate / makespan_ratio |
| — | GT 隔离 | 从 GT route_sheet 喂 s5..s7 | `graph_from_gt` (老) | 拿 GT route_sheet 跑 s5..s7,与 GT or_matrix 比 |
| — | GT 隔离 | 从 GT schedule 喂 s7 | `ground_from_gt` (老) | 拿 GT schedule 跑 s7,与 GT production_plan 比 |

## 每项评测一句话

### 新增 4 项(前 3 项共用同一套字段级 P/R/F1,可以直接级联对比)

`extract` / `verify` / `route_sheet_fieldlevel` **共用同一个** `_score_fixed_schema` 函数:每 (job, step) 对上,把 6 类字段(machine / operation / duration / component_type / param_key / param_value)各自的 TP/FP/FN 汇总,输出 per-field P/R/F1 + macro F1。**这样 s1 F1 → s2 F1 → s4 F1 就是一条统一尺度的级联**,能直接读出"每过一步 F1 掉了多少"。

- **`extract`** — pred `s1_extracted.json` vs GT_s1(GT route_sheet 反投影)→ per-field P/R/F1 + macro F1。**用来判断 s1 抽取是否漏字段、错字段、错值。**

- **`verify`** — 同时对 pred `s1_extracted.json` 与 pred `s2_verified.json` 各跑一遍 `_score_fixed_schema` vs GT_s1,给 `s1_per_field` / `s2_per_field` / per-field `delta`(f1/p/r 各自)+ macro F1 delta。同时按逐字段 step-level 分桶 `fix` / `break` / `still_wrong` / `still_right`(field 判"对"的定义:该字段没有 FP 且没有 FN)。**用来判断 s2 是净修复还是净破坏,以及触动了哪个字段。**

- **`route_sheet_fieldlevel`** — pred `s4_normalized.json` vs GT_s1(与 `route_sheet` 老指标不同尺度,但与 `extract` / `verify` 完全同尺度)→ per-field P/R/F1 + macro F1。**用来判断 s3+s4 相对 s2 又损失了多少 F1;和 `extract`/`verify` 级联对比。**

- **`normalize`** — 只统计 `s3_formatted.json`(归一化前)与"应用 `s4_mappings.json` 后的 s3 值集合"(归一化后)每个 category 的 unique 值数量。字段:`pre_variant_count` / `post_variant_count` / `shrinkage`(pre-post) / `shrinkage_rate` / `mapping_members`(s3 里被映射的值数) / `distinct_canonicals`(mapping 中出现过的不同 canonical 名字数)。**不与 GT 字符串对比** — s4 挑的 canonical 与 GT 的表面串不必一致;这个指标只反映"归一化步骤到底缩减了多少词汇量"。

### 复用现有 5 项(委托 `Evaluation.evaluate`)

- **`production_plan`** — s7 端到端产物 vs GT production_plan,BLEU + 自定义 ROUGE-L (字段级 P/R/F1)。**衡量 s1..s7 累积效果的最终指标。**

- **`route_sheet`** — s4 归一化后经 `FBPipeline.build_route_sheets` 转 route_sheet 结构 vs GT route_sheet,BLEU + 自定义 ROUGE-L。**只截止到 s4,不含 s5..s7,考察前四步累积。** ⚠ 报告中的 ROUGE-L F1 是自定义指标(见 `note/eval_review/route_sheet.md`),对候选字段膨胀敏感。

- **`graph`** — full path 的 s5 输出 vs GT or_matrix,accuracy (IoU) + runtime_err_rate + makespan_ratio(相对 GT makespan)。**s1..s5 累积;是唯一直接影响下游 makespan 的中间产物质量指标。**

- **`graph_from_gt`** — `--type from_gt_route_sheet` 跑 FB pipeline 得到的 s5 输出 vs GT or_matrix,同上三个指标。**跳过 s1..s4,只考察 s5 代码 + s6 求解。**

- **`ground_from_gt`** — `--type from_gt_schedule` 跑 FB pipeline 得到的 s7 输出 vs GT production_plan,BLEU + 自定义 ROUGE-L。**只考察 s7 反投影代码。**

## 输出文件

写入 `outputs/Evaluation/${FB_EVAL_LABEL/.//_}/`(未设 label 时写 `outputs/Evaluation/`):

| evaluation_type | 输出文件 |
|---|---|
| `extract` | `extract_fb.json` |
| `verify` | `verify_fb.json` |
| `normalize` | `normalize_fb.json` |
| `route_sheet_fieldlevel` | `route_sheet_fieldlevel_fb.json` |
| `production_plan` | `production_plan_fb_{bleu,rouge}.json`(由 Evaluation 写) |
| `route_sheet` | `route_sheet_fb_{bleu,rouge}.json`(由 Evaluation 写) |
| `graph` | `graph_fb.json`(由 Evaluation 写) |
| `graph_from_gt` | `graph_from_gt_fb.json`(由 Evaluation 写) |
| `ground_from_gt` | `ground_from_gt_fb_{bleu,rouge}.json`(由 Evaluation 写) |

## 前置条件

| 评测 | 需要预先跑过 | 备注 |
|---|---|---|
| `extract` | `fb_pipeline --type full`(至少到 s1) | 只读 `s1_extracted.json` |
| `verify` | `fb_pipeline --type full`(至少到 s2) | 读 `s1_extracted.json` + `s2_verified.json` |
| `normalize` | `fb_pipeline --type full`(至少到 s4) | 读 `s3_formatted.json` + `s4_mappings.json` |
| `route_sheet_fieldlevel` | `fb_pipeline --type full`(至少到 s4) | 读 `s4_normalized.json`;与 `extract`/`verify` 同尺度可级联 |
| `production_plan` / `route_sheet` / `graph` | `fb_pipeline --type full` 全跑完 | — |
| `graph_from_gt` | `fb_pipeline --type from_gt_route_sheet` 已跑 | — |
| `ground_from_gt` | `fb_pipeline --type from_gt_schedule` 已跑 | — |
