# DSL / FB Pipeline Evaluation 方法审查（汇总）

> **审查范围**：`src/evaluation/evaluation.py` 全部 6 种 `experiment_type`（`production_plan` / `route_sheet` / `graph_from_gt` / `graph` / `ground_from_gt` / `dae`）
> **审查方式**：每个测试起一对独立 agent（a = 综合论文和代码说明测试；b = 从「评价方法」+「实现漏洞」两角度质疑），三轮讨论后落盘。所有质疑均以 ta71 真实产物或代码位置复核，非空谈。
> **详情**：见 `note/eval_review/{production_plan, route_sheet, graph_from_gt, graph, ground_from_gt, dae}.md`
> **总判定**：**评价框架的分工设计（CPE / CAE / CSE / SGE / DAE）方向合理；但当前实现下几乎每一项 metric 都有系统性抬分或不可比的 bug，且大量 baseline / DSL 分支在本仓库根本未执行。多数「论文数字」在本仓库不可复现。**

---

## 一、跨测试系统性问题（10 条 pattern，均被多个 evaluation 独立命中）

### 1. EMKVP（代码里叫 `__rouge_score`）三重独立 bug，同时污染 CPE / CAE / SGE

三处独立缺陷分别被三个测试的 agent b 命中，共存于同一函数：
- **[CPE]** `C+=1; break` 后不消费 reference 键 → 同一 GT 值被 candidate 多键重复命中 → P/R/F1 系统性虚高（对高频重复值更有利）。
- **[CAE]** `key.split(".")[-1]` 只取末段 key → 完全 order-invariant，跨步骤 / 跨字段错配（`precondition.component` 与 `postcondition.component` 因末段同名互匹）。
- **[SGE]** 匹配循环里 `continue` 跳过 `start/end/job_id/task_id`，但分母 `X/Y` 从不扣减 → P/R/F1 三值恒等且封顶 ≈0.66（`ground_from_gt_fb_rouge.json` 首值 0.6627402…完全由此产生）。
- **附带**：代码/论文里叫「ROUGE-L」，实际无 LCS / n-gram / 顺序概念，是 EMKVP。**命名不实**。

### 2. baseline / 部分 DSL 分支在本仓库**从未执行**，论文数字不可复现

- **[CPE]** `Baseline-*` / `Baseline2-*` 目录不存在 → baseline / baseline2 列表实跑为空。
- **[CAE]** 全仓 0 个 `structural_info.json` → `route_sheet_baseline_bleu.json = []`；即便有，schema 也与 `route_sheets.json` 不对齐。
- **[CSE-1]** `graph_from_gt_dsl.json` 全空、`from_gt` 目录无 `operation_programs.json` → CSE-1 事实上只有 FB 单方数据。
- **[CSE-2]** baseline 三件套缺失，第 362 行 `all(exists)` 恒 False → `graph_baseline.json` 三个 list 全空。
- **[SGE]** `ground_from_gt_dsl_bleu.json = []`、无 DSL production_plan 落地。
- **[DAE]** `production_plan_baseline_bleu.json` / `baseline2_bleu.json` 全空、顶层 `dsl_bleu.json` 不存在 → DAE 分支若真跑会 IndexError 或全 nan。**全仓库无任何 `dae*.json` 产物 → DAE 从未端到端跑通**。

**推论**：论文表格里 MSL / TSL 及部分 DSL 数字在当前 deepseek 副本无法核验，需回溯到原始 GPT-4o 运行。

### 3. 「from_gt」并未真正隔离前段（CSE-1 / SGE）

- **[CSE-1]** FB 的 `s5_or_matrix.dependencies` 是 FB 管线自己重建的（甚至 duration 也是重建的：GT `[83,59,49,84,35,68]` vs FB `[83,70,59,35,25,58]`），`from_gt` 只锚定了 operation 命名。「只考察约束图生成」的前提被击穿。
- **[SGE]** GT、FB、DSL 三方 candidate 都由**同一份 `assigned_jobs` 经同构反投影生成**，结构上是 identity 检验；BLEU 全 ≈0.858、10 instance 差 <0.002 的天花板是这个结构决定的，不反映方法差异。修好 pattern 1 里的 SGE 分母 bug 反而会让 EMKVP 趋近 1.0，从反面坐实 SGE 是 identity 检验。

### 4. CSE-2 「端到端」被 GT 送分（**方法层最严重发现**）

FB precedence 的 `(pred_op, succ_op)` 两端 operation 名从 GT route_sheet 抽取（`evaluation.py:409` 传 `ground_truth_route_sheet_content`），FB 其实自产了 operation 名（`s4_normalized.json` 里的 `"Shearing"`）却弃用。实测 ta71：76/100 job 步数不一致（job0 GT 6 步 vs FB 20 步）；若 resource 用 FB 自己机器名 IoU=0.038，整条 FB IoU=0.35 几乎全靠 precedence 撑起。**端到端最难的 operation 命名被 GT 送分、machine 命名 FB 全错却因 precedence 用 op 名而不受罚**。CSE-2 与 CSE-1 应有的本质差异（是否累积前段命名误差）被抹掉。

### 5. 静默 `except: continue` / 死代码 / 跨测试共用抽取器

- **[CSE-1 & CSE-2 同时]** `__get_baseline_precedence_constraint_CSE_1` 第 736 行 `pred_operation` 未定义（前文只有 `pred_machine`），NameError 被外层 `except: continue` 吞 → baseline precedence 恒空。**同一 bug 污染两个测试**。
- **[CSE-2]** `__get_baseline_precedence_constraint_CSE_2` (785) 与 `__get_baseline2_precedence_constraint_CSE_2` (807) 逻辑等价且从未被调用（死代码）。
- **[CSE-1 & CSE-2]** `__get_DSL_recourse_constraint_CSE` / `__get_DSL_precedence_constraint_CSE` 被两测试共用不带后缀 → 任何 DSL bug 同时污染两测试，二者的 DSL 分数不是独立证据。

### 6. 分母 / 尺子不统一，导致跨方法不可比

- **[CPE / CAE]** DSL 缺 machine/operation/duration 整类键 → `X` 系统性变小 → `P=C/X` 因漏答被抬高，**DSL 与 FB 的 Precision/F1 结构性不可比**。
- **[CAE]** ta71 三方 flatten KV 量：GT 6632 / DSL 26047（≈3.9×），DSL 整包比对不做「取最优 route」对齐 → F1 被量级差主导。
- **[CSE-1 / CSE-2]** FB/DSL 与 GT 比 **operation** precedence、baseline 与 GT 比 **machine** precedence，分母语义不同 → IoU 不可横向比。
- **[CSE-2 / DAE]** 三方数据源不对齐：DAE 里 `dsl` 读 `self.dsl_eval_out_dir` 子目录，`baseline`/`baseline2` 硬编码读顶层 → 即便补齐数据也可能跨模型 / 跨批次。

### 7. 类型 / 单位不一致造成系统性偏袒

- **[CAE]** duration 类型：GT/FB `"83 minutes"` (str) vs DSL `83` (int) → EMKVP 里 `83 == "83 minutes"` 恒 False → **DSL 所有 duration 恒 0 命中**、FB 侥幸得分。「表面格式」而非「语义」比较造成不公平。
- **[CPE]** 值匹配无单位/同义词归一：`"42 mins"` vs `"42 minutes"` 判错、`"1200 RPM"` vs `"1200 rpm"` 过。
- **[SGE 实现 bug]** FB `ground` 463–466 行 `machine_index >= len(self.machines)` 时顶层 machine 反查失败写成 `machine_XX_unknown`，但因顶层 machine 键数少被稀释、metric 对「排程落到哪台机器」这一 grounding 核心输出不敏感（ta71 20/20 全 `unknown`）。

### 8. 名字与实测目标严重错位（**metric 与命题不匹配**）

| 测试 | 名字暗示 | 实测度量 |
|---|---|---|
| CPE | 完整流水线 production plan 质量 | grounding 保真度（CP-SAT 独有 start/end/job/task 恰被过滤） |
| CSE-2 | 端到端 constraint 质量 | 依赖拓扑碰巧命中（op 命名靠 GT 送分） |
| SGE | 独立 grounding 能力 | identity 检验（candidate ≈ reference） |
| DAE | one-approach-multi-domain 自适应 | 10 个 instance 内抽样波动（无 domain 标注） |

### 9. 聚合口径 / 样本数不齐

- **[CSE-1 / CSE-2]** accuracy vs makespan_ratio 样本数不同（e.g. 3 vs 2；baseline 0 / DSL 10 / FB 10），仍算术平均。
- **[CPE / CAE]** `evaluation.py` 只落 per-instance list，均值发生在 `scripts/generate_comparison_report.py::_mean` — 聚合职责分散、易误读。
- **[CSE-2 / SGE]** makespan_ratio 只统计有 `makespan.txt` 的成功样本 → 幸存者偏差；崩溃 / 无 makespan 的样本被静默丢弃，均值系统性偏乐观。
- **[CSE-1 / CSE-2]** `compile_err_rate` 列为指标却**从未填充**（磁盘 `s7_compile_errors.txt` 存在却未接线）。

### 10. 论文声明 vs 代码实现直接矛盾

- **[DAE 致命]** 论文明说 "VMR ... where higher values are more desirable" 且 "Ours ... VMR significantly surpasses that of TSL"；但代码 `calculate_vmr = np.var / np.mean`（标准 VMR / 离散指数），高均值低方差反而**更小**。方向互斥。
- **[CSE-2]** 论文三指标之一 Compiler-ER 在 `graph` 分支结果字典无该键。
- **[DAE]** 论文 IV-G 说 DAE 观测 convergence 并覆盖「all previous experiments」，代码只读 production_plan 一种产物、无收敛曲线；论文 V-E 的 Kruskal-Wallis 检验代码里无实现。

---

## 二、每个测试的最严重问题（一屏摘要）

### CPE — `production_plan` [详见 `eval_review/production_plan.md`]
1. **[真 bug]** EMKVP 命中后 break 但不消费 reference 键 → P/R/F1 系统性虚高。
2. **[公平性]** DSL 缺整类键 → `X` 变小 → `P=C/X` 被漏答抬高 → DSL vs FB Precision/F1 结构性不可比。
3. **[效度]** EMKVP 砍掉路径前缀退化为「末段键名+值」多重集重叠；CPE 实为 grounding 保真度评估，不评排产（属论文声明的分工，但存在名实落差）。

### CAE — `route_sheet` [详见 `eval_review/route_sheet.md`]
1. **[致命·方法]** EMKVP 用 `key.split(".")[-1]` 完全 order-invariant + 跨步骤/跨字段错配（machine 跨步命中、pre/postcondition.component 互配）→ 打乱工序与正确 route 同分。
2. **[偏袒]** duration 类型 GT/FB=str 而 DSL=int → DSL 恒 0 命中。
3. **[不可比]** 三方顶层结构不同 + KV 量级悬殊（ta71 GT 6632 / DSL 26047），DSL 不做「取最优 route」对齐。

### CSE-1 — `graph_from_gt` [详见 `eval_review/graph_from_gt.md`]
1. **[方法·最严重]** `from_gt` 未真正隔离前段——FB 的 `s5_or_matrix.dependencies` 由 FB 自己重建（连 duration 都重建），precedence 差异混入 or_matrix 重建误差。
2. **[实现·最严重]** DSL / baseline 在 `from_gt` 下无数据（`graph_from_gt_dsl.json` 全空、baseline json 不存在）→ CSE-1 事实上只有 FB 单方，"三方对比" 不成立。
3. **[实现]** `__get_baseline_precedence_constraint_CSE_1` 第 736 行 `pred_operation` 未定义、异常被吞 → baseline precedence 恒空。

### CSE-2 — `graph` [详见 `eval_review/graph.md`]
1. **[致命·方法]** FB 端到端被架空：precedence 端点 operation 名借用 GT route_sheet，最难的 op 命名被送分、machine 命名 FB 全错却不受罚，与 CSE-1 的本质差异被抹掉。
2. **[实现]** baseline 恒空未参评（三件套文件缺失 + NameError bug），论文若报 CSE-2 baseline 数字来源存疑。
3. **[方法]** Compiler-ER 完全缺失；baseline 用 machine_precedence、DSL/FB 用 operation_precedence，两把尺子不可横比。

### SGE — `ground_from_gt` [详见 `eval_review/ground_from_gt.md`]
1. **[范式]** SGE 本质是 identity 检验——GT/FB/DSL 三方 candidate 都由同一份 `assigned_jobs` 经同构反投影生成，方法间无区分度空间。改代码无解。
2. **[实现]** EMKVP 分母未扣减 skip 键 → P=R=F1 恒等且封顶 ≈0.66；修好后会趋近 1.0，反证 SGE 是 identity 检验。
3. **[实现]** FB 顶层 machine 反查越界写 `machine_XX_unknown`（20/20 全 unknown），但 metric 对此不敏感。

### DAE [详见 `eval_review/dae.md`]
1. **[致命·矛盾]** 论文「越高越好、Ours surpass TSL」与代码 `var/mean` 越小越好方向互斥，必有一处错。
2. **[致命·实现]** DAE 从未端到端跑通、结果不可复现：依赖的输入产物全空或不存在，全项目无任何 `dae*.json`。
3. **[方法]** 10 个 scenario 是否真 domain shift 无从佐证（无 domain 标注、无 Kruskal-Wallis 实现、无收敛曲线），DAE 与论文 IV-G 声称的范围严重不符。

---

## 三、综合建议（按修复价值排序）

### P0（不修则任何跨方法结论都不成立）
1. **修复 EMKVP 三重 bug**（`evaluation.py::__rouge_score`）：
   - 命中后用 set 记录已消费的 reference key，使 `C ≤ min(X,Y)`；
   - 用 full-path key（而非末段 key）匹配，禁止跨步骤/跨字段错配；
   - 分母 X/Y 按 skip 键实际扣减（仿弃用的 `__rouge_score_similarity`）。
2. **修复 CSE-2 端到端被送分**：FB precedence 命名改用 FB 自产的 `s4_normalized.json` operation 名，resource 用 FB 自产 machine 名并配合命名归一 / 对齐。
3. **修复 `pred_operation` NameError**（`__get_baseline_precedence_constraint_CSE_1:736`）：`pred_operation` → `pred_machine`。
4. **补齐或声明缺失的 baseline / DSL 产物**：`Baseline-*`、`from_gt` 场景的 DSL、DAE 依赖的顶层文件——要么用当前模型重跑，要么在报告中标注 N/A 且不列数字。
5. **DAE 方向自洽**：论文改述 or 代码改公式（改用 CV = std/mean 或 mean/var），二者对齐；补齐 Kruskal-Wallis 与收敛曲线实现。
6. **重新审视 SGE 定位**：明确它只是 CPE 的 ceiling 参照而非独立能力评估；否则改造使 candidate 不再与 GT 同构（例如从中间语义而非直接查 GT 明细取值）。
7. **改述 CSE-1 的「隔离」表述**：或让 FB precedence 直接从 GT dependencies 抽取，做到真正隔离。

### P1
8. 统一数据类型 / 单位（duration str↔int、`"42 mins"`≈`"42 minutes"`），或在 EMKVP 前做归一。
9. IoU 改保留重数的多重集 IoU 或加权 F1；对 CP-SAT 多解做等价类归一（可达性 / 传递闭包比较）。
10. 统一 baseline 与 DSL/FB 的 precedence 尺子（都用 operation 或都用 machine），或在报告中明示尺子差异、禁止横向比较。
11. 补统计 CSE-2 的 Compiler-ER、接线 `s7_compile_errors.txt`；makespan_ratio 对失败样本改为计入惩罚或单列「成功率」，消除幸存者偏差。
12. 精确对齐三方 instance 子集再求均值，禁止样本数不齐仍算术平均。

### P2
13. 重命名 `__rouge_score` → `EMKVP`，代码与论文术语统一。
14. 删除死代码 `__get_baseline_precedence_constraint_CSE_2` / `__get_baseline2_precedence_constraint_CSE_2`。
15. 给 CSE-1 / CSE-2 共用的 DSL 抽取器加显式单测，避免交叉污染。
16. 聚合统一到 `evaluation.py` 或在报告脚本中明示单值来源。
17. DAE：改用样本方差 `ddof=1`；补 10 个 scenario 的 domain 异质性量化（否则改称「跨 instance 稳定性」）。

---

## 四、附：审查方法说明

- 每个测试起一对独立 general-purpose subagent，agent a 综述（论文 + 代码 + 一手 sample），agent b 从「评价方法层面」+「实现漏洞层面」提 3–6 条具体问题，双方最多 3 轮讨论。
- 所有质疑均要求以 ta71 真实产物或代码行号复核，非空谈。协调者对每条论断独立复核后落盘。
- 6 份 md 位于 `note/eval_review/{production_plan, route_sheet, graph_from_gt, graph, ground_from_gt, dae}.md`。每份含完整讨论过程与逐条结论。
