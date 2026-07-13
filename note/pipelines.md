# Repo 流程速查（Preprocess / AutoDSL / DSL Pipeline / FB Pipeline / Evaluate）

> 5 条流程的步骤名、如何运行、输入/输出。以 `main.py` 为主要入口；批量运行走 `scripts/*.ps1`。
> 命名迁移见 `MIGRATION.md`（老 `CPE_CAE_CSE-2` / `CSE-1` / `SGE` → 新 `full` / `from_gt_route_sheet` / `from_gt_schedule`；老 `step*` / `CGM_*` / `SGM_*` → 新 `s1..s7`）。

工作目录固定为 `D:/FB/TASE25-AutoDSL-Constraint/`，运行 python 前先 `source .venv/Scripts/activate`。LLM 相关的流程依赖以下环境变量（一般由 `scripts/*.ps1` 内部 export）：
- `LLM_BASE_URL`（默认 `http://localhost:8000/v1`，走 mi300 tunnel）
- `LLM_MODEL`（`deepseek-v3` / `qwen3-4b` / `qwen3-30b` / `qwen3-235b`）
- `OPENAI_API_KEY`（任意非空即可）
- `LLM_MAX_WORKERS`（默认 64）

---

## 1. Preprocess — 从原始 JSSP 数据到结构化 route sheet + NL orders

### 1.1 `preprocess`（`RouteSheet`）
把原始 JSSP + machines + arrange 组合成结构化 route sheet，并归约为按 domain 分组的形式。用 LLM 但只在 create 阶段。

| 步骤 | 方法 | 输入 | 输出 |
|---|---|---|---|
| mapping | `RouteSheet.mapping()` | `data/machines.json`, `data/jssp_data.json`, `data/arrange.json` | `preprocess/jssp_mapped.json` |
| create_route_sheet (LLM) | `RouteSheet.create_route_sheet(target_descriptions=...)` | `preprocess/jssp_mapped.json`, prompt `src/prompts/route_sheet_prompt.txt` | `preprocess/route_sheet.json`（扁平 list，每 job 一条） |
| route_sheet_reduce | `RouteSheet.route_sheet_reduce()` | `preprocess/route_sheet.json` + `data/jssp_data.json` 的 domain 划分 | `preprocess/route_sheet_reduce.json`（按 domain 分组，每 domain 若干 job） |

运行：
```bash
python main.py --mode preprocess [--instances ta71 ta72 ...]
```
`--instances` 只影响 create_route_sheet 的目标 subset；mapping / reduce 依旧全量。默认目标 10 个 instance（ta71..ta80）。

### 1.2 `generate_orders`（一次性 LLM 造 NL 数据）
把 GT 的结构化 route sheet 反向翻译成自然语言订单，供后续 DSL / FB pipeline 消费。

| 步骤 | 输入 | 输出 |
|---|---|---|
| generate_orders.py (LLM) | `preprocess/route_sheet_reduce.json`, prompt `src/prompts/generate_synthetic_data.txt` | `preprocess/orders/ta{71..80}/orders.json`（每个 instance 一个文件，list of order） |

运行：
```bash
python generate_orders.py
# 或
.\scripts\generate_orders.ps1
```
幂等：文件存在的 instance 会跳过。prompt 已硬约束 duration 必须使用 minutes 单位。

---

## 2. AutoDSL — 从 route sheet 构建 operation / production DSL

两条子流程互相独立，可并行。都读 `preprocess/route_sheet_reduce.json`，都写到 `outputs/AutoDSL/`。

### 2.1 `autodsl_operation`（`Feature` + `Operation`，走 SciBERT + 递归聚类）

| 步骤 | 方法 | 输入 | 输出 |
|---|---|---|---|
| Feature 抽取 | `Feature.__init__(...)` | domain 的所有 route_sheet, `data/feature.json`（读写模型缓存） | 内存中的 `feature.feature_data` |
| 层级聚类 (SciBERT + DPMM，per-opcode 并行) | `Operation.recursive_clustering` + `Operation.analyse` | `feature.feature_data`, `data/operation_dsl.json` | 内存中的 `operation.operation_dsl` 与 likelihood 轨迹 |
| DSL 规整 | `Operation.dsl_regular` | 上一步产物 | 最终 `operation_dsl` |

汇总所有 instance 后一次性落盘。

**运行**：`python main.py --mode autodsl_operation [--instances ...]`

**产物**：
- `outputs/AutoDSL/feature.json`
- `outputs/AutoDSL/total_operation_dsl.json`（10 个 instance 的 operation DSL）
- `outputs/AutoDSL/total_likelihood_list.json`

### 2.2 `autodsl_production`（`Production`，纯 EM）

| 步骤 | 方法 | 输入 | 输出 |
|---|---|---|---|
| 抽取 pred-succ 对 | `Production.extract()` | domain 的 route_sheet | 内存中的 `production.production_dsl` |
| EM 学习结构 | `Production.EM_extract()` | 上一步 | `EM_results`, `EM_updates` |

**运行**：`python main.py --mode autodsl_production [--instances ...]`

**产物**：
- `outputs/AutoDSL/total_production_dsl.json`
- `outputs/AutoDSL/EM_results.json`
- `outputs/AutoDSL/total_EM_updates.json`

### 2.3 `groundtruth`（准 GT，AutoDSL 的评测基线）
基于 GT route_sheet 直接跑 CP-SAT，给出真解和真 or_matrix，供后续 CSE-1 / SGE / from_gt 类实验使用。

| 步骤 | 方法 | 输入 | 输出 |
|---|---|---|---|
| get_grounded_route_sheet | `GroundTruth.get_grounded_route_sheet` | `preprocess/route_sheet_reduce.json` | `outputs/GroundTruth/<instance>/route_sheets.json` |
| get_grounded_or_matrix | `GroundTruth.get_grounded_or_matrix` | `preprocess/jssp_mapped.json` + route_sheet | `outputs/GroundTruth/<instance>/or_matrix.json` |
| get_grounded_assigned_jobs | `GroundTruth.get_grounded_assigned_jobs` | or_matrix + OR-Tools | `outputs/GroundTruth/<instance>/assigned_jobs.json`, `makespan.txt` |
| get_grounded_production_plan | `GroundTruth.get_grounded_production_plan` | assigned_jobs + route_sheet | `outputs/GroundTruth/<instance>/production_plan.json` |

**运行**：`python main.py --mode groundtruth [--instances ...]`

---

## 3. DSL Pipeline — 用 AutoDSL 得到的 DSL 把 NL orders 翻成程序，再求解

需要先跑完 preprocess、generate_orders、autodsl_operation、autodsl_production。

### 步骤（`DSLPipeline`）

| 步骤 | 方法 | 说明 | 落盘文件（在 dump_dir 内） |
|---|---|---|---|
| orders → DSL 程序 (LLM) | `orders2dsl_program_no_batch_parallel` | 每个 order 先做 operation/production 抽取，再拼 prompt 一次翻译成程序。仅 `full` 用 | `operation_programs.json`, `production_programs.json` |
| 从 GT route_sheet → DSL 程序 (LLM) | `route_sheets2dsl_program_no_batch` | 只有 `from_gt_route_sheet` 用 | 同上 |
| DSL → OR matrix + 求解 | `dsl_program2or_matrix` | 抽 op→machine + Pred/Succ 依赖，转 OR matrix；顺手调 OR-Tools | `or_matrix.json`, `machines.json`, `assigned_jobs.json`, `makespan.txt`, `err_rate.txt` |
| DSL → route sheet | `dsl_program2route_sheets` | 把 program 反向 render 出结构化 route sheet | `route_sheets.json` |
| JSP 解 → production plan | `JSP_result2production_plan` | 把 assigned_jobs 映射回 op / precond / postcond / parameters | `production_plan.json`, `compile_error_num.txt` |

**dump_dir**：`outputs/${DSL_OUTPUT_PREFIX:-DSLPipeline-<experiment_type>}/<instance>/`
（多模型场景由 wrapper 设 `DSL_OUTPUT_PREFIX=DSL-models/<model>`）

### 三种 `--type`（老名字保留在 dsl_pipeline 内部）

| `--type`（新） | 老 name（`experiment_type`） | 触发路径 |
|---|---|---|
| `full` | `CPE_CAE_CSE-2` | orders → programs → or_matrix → route_sheets → production_plan |
| `from_gt_route_sheet` | `CSE-1` | GT route_sheets → programs → or_matrix → production_plan |
| `from_gt_schedule` | `SGE` | 直接读 `outputs/GroundTruth/<instance>/assigned_jobs.json` → production_plan |
| `dae` | `DAE` | 空实现（DSL 内做诊断，由 evaluation 阶段读汇总） |

**运行（单实例/单模型）**：
```bash
python main.py --mode dsl_pipeline --type full [--instances ta71 ta72 ...]
```

**批量 4 模型 + evaluation + 对比报告**：
```bash
.\scripts\run_dsl_pipeline_4llm.ps1              # 全部 4 个模型
.\scripts\run_dsl_pipeline_4llm.ps1 qwen3-4b     # 指定子集
```

---

## 4. FB Pipeline — 固定 schema 的 s1..s7，替代 DSL 的两阶段翻译

需要先跑 generate_orders。不依赖 AutoDSL（这是 FB 相对 DSL 的关键差异）。

### 步骤（`FBPipeline`）

| 步骤 | 方法 | 说明 | 落盘文件 |
|---|---|---|---|
| s1 extract (LLM, per-order 并行) | `extract` | NL → 固定 schema JSON；prompt `src/prompts/step1_extract.txt`；`schema_validator` 校验，bad case 重试 1 次 | `s1_extracted.json` |
| s2 verify (LLM, per-order 并行) | `verify` | 用 NL 校对上一步 extraction 的事实一致性；prompt `src/prompts/step1_verify.txt` | `s2_verified.json` |
| s3 format (code) | `apply_format` | 走 `src/experiment/format_rules.py` 的 Title/Pascal/lower casing；纯确定性，不打 LLM | `s3_formatted.json` |
| s4 normalize (LLM, 单次调用 + verify) | `normalize` | 全局跨 order 收集 unique machine/operation/component_type/param 后聚合同义词，diff-based verify；返回归一化 mapping 并作用到所有 order | `s4_normalized.json`, `s4_mappings.json`, `s4_mappings_pre.json` |
| s5 graph (code) | `build_graph` | 从 normalized JSONs 抽 machine 列表 + linear precedence，生成 OR matrix | `s5_or_matrix.json`, `s5_machines.json` |
| s6 solve (OR-Tools CP-SAT) | `solve` | JSP 调度 | `s6_assigned_jobs.json`, `s6_err_rate.txt`, `s6_makespan.txt` |
| s7 ground (code) | `ground` | 把 assigned_jobs 反投影出 op/machine/duration/precond/postcond/parameters | `s7_production_plan.json`, `s7_compile_errors.txt` |

**dump_dir**：`outputs/${FB_OUTPUT_PREFIX:-FB-2s-<experiment_type>}/<instance>/`
（多模型：`FB_OUTPUT_PREFIX=FB-models/<model>`；GT ceiling：`FB-gt-ceiling/from_gt_rs` 或 `FB-gt-ceiling/from_gt_sched`）

### 三种 `--type`（FB 直接用新名字）

| `--type` | 老 name | 触发路径 |
|---|---|---|
| `full` | `CPE_CAE_CSE-2` | orders → s1..s7 全跑 |
| `from_gt_route_sheet` | `CSE-1` | 读 `outputs/GroundTruth/<instance>/route_sheets.json` 直接喂 s5..s7 |
| `from_gt_schedule` | `SGE` | 读 GT `assigned_jobs.json` + `route_sheets.json`，只跑 s7 |

`--force`（默认 `True`）决定是否忽略已有中间文件全量重算；`--no-force` 会加载已有的 sX 文件跳过对应步骤。

**运行（单模型）**：
```bash
python main.py --mode fb_pipeline --type full [--instances ta71 ta72 ...]
# 走 mi300 tunnel 的包装脚本：
.\scripts\run_fb_pipeline.ps1 [ta73 ta74 ...]
```

**批量 4 模型 + GT ceiling + evaluation + 对比报告**：
```bash
.\scripts\run_fb_pipeline_4llm.ps1               # 全部 4 个模型
.\scripts\run_fb_pipeline_4llm.ps1 qwen3-4b      # 指定子集
```
该脚本会顺带跑 `FB-gt-ceiling/{from_gt_rs, from_gt_sched}` 作为 LLM-free 上界。

---

## 5. Evaluate — 6 种评测口径

统一入口：`python main.py --mode evaluation --evaluation_type <TYPE>`

`Evaluation` 会同时读 baseline / baseline2 / DSL / FB 的对应产物（存在的才算）。可通过环境变量把 FB / DSL 的读写目录指到 per-model 目录：
- `FB_EVAL_LABEL=FB-models/<model>` → 读 `outputs/FB-models/<model>/…`，写 `outputs/Evaluation/FB-models_<model>/…`
- `DSL_EVAL_LABEL=DSL-models/<model>` → 读 `outputs/DSL-models/<model>/…`，写 `outputs/Evaluation/DSL-models_<model>/…`

| `--evaluation_type`（新） | 老 name | 输入 | 指标 | 输出（`outputs/Evaluation/`） |
|---|---|---|---|---|
| `production_plan` | `CPE` | FB `s7_production_plan.json` / DSL `production_plan.json` vs GT `production_plan.json` | BLEU + ROUGE-L (P/R/F1) | `production_plan_{fb,dsl,baseline,baseline2}_{bleu,rouge}.json` |
| `route_sheet` | `CAE` | FB `s4_normalized.json`（经 `FBPipeline.build_route_sheets`）/ DSL `route_sheets.json` / baseline `structural_info.json` vs GT `route_sheets.json` | BLEU + ROUGE-L | `route_sheet_{fb,dsl,baseline}_{bleu,rouge}.json` |
| `graph_from_gt` | `CSE-1` | FB `s5_or_matrix.json`+`s5_machines.json`+`s6_*.txt` / DSL `or_matrix.json`+`operation_programs.json`+`production_programs.json`+`err_rate.txt`+`makespan.txt` vs GT | accuracy IoU + compile/runtime err_rate + makespan_ratio | `graph_from_gt_{fb,dsl,baseline}.json` |
| `graph` | `CSE-2` | 同 `graph_from_gt`，但从 `full` 路径的 s5 输出读 | accuracy IoU + runtime err_rate + makespan_ratio | `graph_{fb,dsl,baseline}.json` |
| `ground_from_gt` | `SGE` | FB `s7_production_plan.json`（from_gt_schedule） / DSL `production_plan.json`（SGE） vs GT `production_plan.json` | BLEU + ROUGE-L | `ground_from_gt_{fb,dsl}_{bleu,rouge}.json` |
| `dae` | `DAE` | 复读之前 `production_plan_*` 的 BLEU/ROUGE | VMR（方差/均值） | `dae.json`, `dae_metric.json`, `dae_vmr.json` |

**运行示例**：
```bash
python main.py --mode evaluation --evaluation_type production_plan
python main.py --mode evaluation --evaluation_type graph_from_gt
```

**跨模型对比报告**：
```bash
python scripts/generate_comparison_report.py
# 产物：outputs/Evaluation/FB-models-comparison-report.md
```

---

## 端到端串起来跑一次（4 模型 + 对比）

```powershell
# 前置：起 mi300 tunnel（bash D:/FB/mi300/tunnel.sh），curl http://127.0.0.1:8000/v1/models 应返回可用模型
# 前置：venv 已建、依赖已装、spacy/nltk 数据已下载

# preprocess + autodsl + groundtruth（一次性）
python main.py --mode preprocess
python main.py --mode autodsl_operation
python main.py --mode autodsl_production
python main.py --mode groundtruth
.\scripts\generate_orders.ps1

# 两条 pipeline + evaluation + 报告
.\scripts\run_dsl_pipeline_4llm.ps1
.\scripts\run_fb_pipeline_4llm.ps1
```
