# route_sheet (CAE) 评估审查

评估名：`route_sheet`（论文中的 CAE / Constraint Abstraction Evaluation）
评估入口：`src/evaluation/evaluation.py` 第 159–228 行（`experiment_type == "route_sheet"`）

## 一、测试目标与方法

**目标**：验证「约束抽象」把工艺描述转成完全结构化 route sheet（工序路线：每步 machine / duration / precondition / postcondition / operation / parameters）的能力。对比三条路线（baseline、DSL pipeline、FB pipeline）与专家监督合成的 GT route sheet。

**被评估对象形式**（最终都 `json.dumps` 成字符串再比）：
- **GT**：`groundtruth.py` 第 26–30 行 `get_grounded_route_sheet`，从静态文件 `preprocess/route_sheet_reduce.json` 按 `instance_description` 过滤取出（**非 LLM 现场生成**，无循环污染风险）。实际读取 `outputs/GroundTruth/instance taXX/route_sheets.json`，形如 `[{part_name, route_sheet:[{machine,duration,...}]}]`。
- **FB**：`FBPipeline.build_route_sheets`（`fb_pipeline.py` 516 行）直接把 `s4_normalized.json` 的 `data["steps"]` 包成 `{instance_description, route_sheet:steps}`，字段原样透传。
- **DSL**：`dsl_pipeline.py` 757 行 `dsl_program2route_sheets`，从 operation/production program 组装，顶层是 `[[{step}]]`。
- **baseline**：读 `Baseline-` 目录下 `structural_info.json`。

**指标算法**：
- **BLEU**（`__bleu_score`, 577 行）：GT 与候选各 `json.dumps` 后按空白 `split()` 当 token，nltk `sentence_bleu`，smoothing method4。token 是像 `"machine":`、`60},` 这类含大量 JSON 模板字符的碎片。
- **EMKVP**（代码名 `__rouge_score`, 629 行，**并非标准 ROUGE**，对应论文 Exact Match of Key-Value Pair）：先 `flatten_structure`（662 行）把两边 JSON 拍平成 `路径 -> 标量`；对候选每个 KV，用 `key.split(".")[-1]` 取末段 key 名，在 GT 中找「末段 key 名相同且 value 相等（str 忽略大小写）」的项，命中即 `C+=1; break`；跳过含 `start/end/job_id/task_id` 的 key。返回 P=C/X, R=C/Y, F1=2C/(X+Y)，X/Y 为候选/GT 扣除后的 KV 数。
- **聚合**：`evaluation.py` 只把 per-instance 值 append 进 list 后 `write_json` 原始 list；求均值发生在 `scripts/generate_comparison_report.py`（`_mean(rouge["F1"])`、`_mean(bleu)`），论文表格单值来自该脚本。

## 二、讨论过程

采用说明者 / 质疑者双 agent 三轮讨论，质疑者亲自核对代码与 ta71 三方真实 sample。

- **A0（说明）**：厘清目标、GT 静态来源、三条路线产物形式，并指出 `__rouge_score` 实为 EMKVP 而非标准 ROUGE。
- **B1（质疑，6 问）**：Q1 EMKVP 只取末段 key → 完全 order-invariant；Q2 同名 key 跨步骤/跨字段错配；Q3 duration 类型三方不一致；Q4 顶层结构不同 + KV 量级悬殊；Q5 evaluation.py 无聚合；Q6 baseline 死分支。
- **A1（回答）**：亲自核对 sample，坐实 Q1/Q2/Q3/Q4/Q6；**更正 Q5**——聚合确实存在，只是在 `generate_comparison_report.py` 而非 `evaluation.py`。关键实测：ta71 GT flatten 6632 KV / DSL 26047 KV（约 3.9 倍）；GT/FB duration 为 str `"83 minutes"`、DSL 为 int `83`；全仓 0 个 `structural_info.json`，`route_sheet_baseline_bleu.json` 为 `[]`。
- **B2（收束）**：接受 Q5 更正（撤回「根本没聚合」的说法，降级为「聚合职责分散」实现瑕疵）；其余五项坐实。凝练严重度排序与改动优先级。

## 三、结论

### 合理之处
- GT 由静态专家文件产出而非 LLM 现场生成，**不存在 GT 循环污染**。
- EMKVP 对空/短序列有保护（X==0 或 Y==0 返回 (0,0,0)），`bad_case` 会产出空 route_sheet。
- 跳过 `start/end/job_id/task_id` 等易变字段，避免无意义扣分。
- 聚合逻辑确实存在（在 report 脚本），论文单值有明确来源。

### 存疑（方法层面）—— 致命
1. **度量无位置/字段约束（Q1+Q2，最根本缺陷）**：EMKVP 因 `key.split(".")[-1]` 丢掉 `[i].route_sheet[j]` 索引前缀，退化为「末段 key 名 + 值」的多重集交集。后果：(a) 完全 order-invariant，打乱工序的 route 与正确 route **同分**；(b) 同名 key 跨步骤/跨字段错配——候选某步 `machine` 只要 GT 任意步同值即命中，`precondition.component` 与 `postcondition.component` 因末段同名互相匹配。系统性高估 P/R/F1，且无法反映 route 的核心信息（工序顺序与结构归属）。
2. **三方不可比（Q4）**：顶层结构三方不同（GT `[{part_name,route_sheet}]`、DSL `[[step]]` 无 part_name、FB `[{instance_description,route_sheet}]`）；实测 ta71 DSL flatten KV 量约为 GT 的 3.9 倍，DSL 整包比对不做「取最优 route」对齐。`P=C/X` 被巨大的 X 稀释，F1 被量级差主导，跨方法对比失去意义。
3. **类型不一致导致系统性偏袒（Q3）**：duration 在 GT/FB 是 str `"83 minutes"`、DSL 是 int `83`。EMKVP 中 `83 == "83 minutes"` 为 False 且不进 `isinstance(str)` 分支 → DSL 所有 duration **恒 0 命中**；BLEU 同理。FB 因原样保留 str 而得分，DSL 转了 int 被无辜扣分——这是评测框架用「表面格式」而非「语义」比较造成的不公平。

### 存疑（实现层面）
4. **baseline 死分支（Q6）**：baseline 读 `structural_info.json`，全仓 0 个该文件，`route_sheet_baseline_bleu.json` 为 `[]`，分支永不进入；且 `structural_info` 与 `route_sheets` schema 不同，即使存在也语义不对齐。baseline 一栏既无数据、设计上也不成立。
5. **聚合职责分散（Q5 更正后）**：落盘（evaluation.py）与求均值（report 脚本）分处两文件，易误读、难复现。

### 建议改动（优先级）
- **P0**：EMKVP 改为「先按 step 对齐、再逐字段用 full-path key 比较」，禁止末段名跨行/跨字段错配 —— 修复 Q1/Q2，是结论成立的前提。
- **P0**：评估前统一 duration 等字段类型（str↔int 归一，或解析成分钟数比较）—— 修复 Q3 的系统性不公。
- **P1**：三方先归一到同一 canonical schema 再算指标，DSL 侧做「取最优 route」对齐 —— 修复 Q4 的不可比。
- **P1**：修复或移除 baseline 分支（补齐 `structural_info.json` 或在报告中明确标注 N/A）—— 消除 Q6 的死分支。
- **P2**：把聚合统一进 `evaluation.py`，或在报告中明确标注单值来源脚本 —— 消除 Q5 的职责分散。

**总纲**：P0 两项不修，route_sheet 的 P/R/F1/BLEU 数值不足以支撑任何跨方法结论。
