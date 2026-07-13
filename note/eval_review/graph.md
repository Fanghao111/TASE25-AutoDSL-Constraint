# graph (CSE-2) 评估审查

> 评估名（代码内部）：`experiment_type == "graph"`，src/evaluation/evaluation.py 第 331–429 行。
> 论文对应：complete pipeline（端到端）版本的 constraint specification evaluation（Sec. IV，指标 Constraint-Acc / Compiler-ER / Runtime-ER）。
> 审查方式：说明者(agent_a) + 质疑者(agent_b) 双 agent 讨论 + 主审用 ta71 真实产物交叉复核。全程纯分析，未改任何代码。

## 一、测试目标与方法

### 目标
衡量**端到端** constraint quality：三条 pipeline（baseline / DSL / FB）各自从自己产出的中间件（OR-matrix、machines、program）抽取调度约束，与 GroundTruth 约束做 IoU 比对。与 CSE-1（`graph_from_gt`，第 229–329 行）的区别在于：CSE-1 从 GT 侧目录（`suffix[1]`）出发，隔离“图/约束抽取模块”单独评估；CSE-2 走 full 路径（`suffix[0]`），累积前段（NL→中间件）抽取误差，考察整条链路。

### 被评估的三条路径与抽取器
- **baseline**：`or_matrix.json` + `structural_info.json` + `machines.json`；resource=`__get_baseline_recourse_constraint_CSE_2`(763)，precedence=`__get_baseline_precedence_constraint_CSE_1`(697)。
- **DSL**：`operation_programs.json` + `production_programs.json`；resource=`__get_DSL_recourse_constraint_CSE`(743)，precedence=`__get_DSL_precedence_constraint_CSE`(753)。
- **FB**：`s5_or_matrix.json` + `s5_machines.json`；resource=`__get_baseline_recourse_constraint_CSE_1`，precedence=`__get_unified_precedence_constraint_CSE`(915)。

### GT 来源（三种约束）
均从 GT 的 `or_matrix.json` + `route_sheets.json` 抽：resource=(operation, machine)；operation_precedence=(pred_op, succ_op)；machine_precedence=(pred_machine, succ_machine)。

### 指标
- **accuracy_rate** = `__iou`(candidate 约束集, GT 约束集)，字符串集合的交并比（大小写归一），对应论文 Constraint-Acc。
- **runtime_err_rate** = 直接读磁盘 `err_rate.txt`/`s6_err_rate.txt`（评估代码不重算，是上游产物），对应论文 Runtime-ER。
- **makespan_ratio** = pipeline makespan / GT makespan（论文三指标中没有此项，属额外指标）。
- 论文要求的 **Compiler-ER 在本分支完全缺失**（结果字典无该键）。

### IoU 配对（注意 GT 侧不统一）
- baseline：candidate(resource + **机器序** precedence) 配 GT(resource + machine_precedence)。
- DSL/FB：candidate(resource + **工序序** precedence) 配 GT(resource + operation_precedence)。
即 baseline 与 DSL/FB 并非同一把尺子。

## 二、讨论过程

**A0（说明者）** 给出上述结构，并主动点出四处不一致：① Compiler-ER 缺失；② `__get_baseline_precedence_constraint_CSE_1` 第 736 行引用未定义变量 `pred_operation`（函数只算了 `pred_machine`），有依赖时抛 NameError 被 `except: continue` 吞掉，使 baseline precedence 恒空；③ `__get_baseline_precedence_constraint_CSE_2`(785) 与 `__get_baseline2_precedence_constraint_CSE_2`(807) 逻辑等价且主流程均未调用（死代码）；④ FB 的 `__get_unified_precedence_constraint_CSE` 第 409 行传入的是 `ground_truth_route_sheet_content`，即 FB 借 GT 的 route_sheet 给自己 matrix 的 step 命名，存在索引对齐风险。

**B1（质疑者）** 用 ta71 真实产物复核后修正/加重了 A0，并补出更严重的问题：
- 修正：行号有系统性偏移；A0 说“FB 与 GT 只是首步机器号不同”**严重低估**——实测 76/100 个 job 连步数都不一致（job0：GT 6 步 vs FB 20 步）。
- **[最严重] FB 端到端名不副实**：若 resource 用 FB 自己的机器名，IoU=0.0（机器命名体系完全不同："sheet metal shear" vs "Shear (Sheet Metal)"）。FB 的分数几乎全部来自 precedence，而 precedence 的 (pred_op, succ_op) **两端 operation 名都来自 GT route_sheet**，FB 只贡献“哪个 step 依赖哪个 step”的拓扑索引。实测 GT 144 条、FB 155 条、交集 109 条，precedence IoU≈0.72，整条 FB IoU=0.35，与磁盘 `graph_fb.json` 首值 0.384 同量级。
- baseline 不仅有 NameError bug，更根本是 baseline 目录三件套根本不存在，第 362 行 `all(exists)` 为假、整块被跳过；实测 `graph_baseline.json` 三个 list 全空——**baseline 未参评**。
- `__get_unified_precedence_constraint_CSE` 与 `__get_groundtruth_operation_precedence_constraint` 逻辑同构，异常吞在最内层 try，越界只丢单条 pair 不丢整 job。
- `__get_DSL_*_CSE` 被 CSE-1/CSE-2 共用不带后缀区分，任何 bug 同时污染两测试。
- per-instance IoU 只 append 进 list 后算术平均，各路径样本数可不齐（baseline 0 / DSL 10 / FB 10），均值在不同样本集上比、不可比。
- makespan_ratio 只统计有 `makespan.txt` 的成功样本 → 幸存者偏差。

**主审补充复核（收束 A1 视角）**：直接读 FB 的 `s4_normalized.json` 发现——**FB 自己抽出了 operation 名**（"Shearing"、"Turning"…），只是与 GT 命名（"Shear (Sheet Metal)"）不同。这坐实了 B1 第 5 点是**设计缺陷而非无奈之举**：FB 有自产 operation 名可用于命名，但评估代码偏偏改用 GT route_sheet 命名，把 FB 自产 operation 名弃之不用。因此 FB 端到端最难的两件事——step→operation 命名、machine 命名——前者被 GT 送分、后者 FB 做了但因 precedence 用 op 名而不受罚，端到端评估被架空。

## 三、结论

### 合理之处
- 双路径设计（CSE-2 端到端 vs CSE-1 from-GT）的动机正确：论文确实想用二者对比来分离“前段累积误差”与“图抽取误差”。
- IoU 作为 constraint-level accuracy、以字符串集合交并比实现，方法本身对齐论文 Constraint-Acc，思路无误。
- runtime_err_rate 复用上游产物、precedence 异常吞在最内层（丢单条而非整 job），在“尽量多保留可比条目”意义上是可接受的工程折中。

### 存疑（方法）
1. **[致命] FB 端到端评估被架空**：precedence 的 operation 名借用 GT route_sheet，端到端最难的 operation 命名被直接送分；machine 命名 FB 全错却因 precedence 用 op 名不受罚。FB 的 0.35 分几乎只反映“依赖拓扑碰巧命中”，不反映命名质量。**这使 CSE-2 与 CSE-1 的本质区别（是否累积前段命名误差）被抹掉**——正是本次审查最核心的问题。
2. **baseline 与 DSL/FB 用不同尺子**（machine_precedence vs operation_precedence），横向对比不公平（因 baseline 恒空目前是潜在陷阱而非已发生错误）。
3. **Compiler-ER 缺失**：论文三指标之一在本分支无处产出，CSE-2 无法支撑论文关于端到端 Compiler-ER 的任何结论。
4. **makespan_ratio 幸存者偏差**：崩溃/无 makespan 的样本被静默丢弃，均值系统性偏乐观。
5. **样本数不齐仍算术平均**：`mean(dsl)` 与 `mean(fb)` 可能在不同 instance 子集上比较，不可比。
6. runtime_err_rate 是上游黑盒产物，评估代码无法追溯其定义；实测三路径全为 0.0，可信度需向生成端求证。

### 存疑（实现）
1. **baseline 恒空**：baseline 目录三件套缺失使整块被跳过；即便文件在，`__get_baseline_precedence_constraint_CSE_1` 第 736 行 `pred_operation` NameError 也会让 precedence 恒空。**论文若报了 CSE-2 的 baseline 数字，来源存疑。**
2. **死代码**：`__get_baseline_precedence_constraint_CSE_2`(785) 与 `__get_baseline2_precedence_constraint_CSE_2`(807) 逻辑等价且从未被调用。
3. **DSL 抽取器 CSE-1/CSE-2 共用**：`__get_DSL_recourse_constraint_CSE`/`__get_DSL_precedence_constraint_CSE` 任何 bug 会同时污染两个测试，二者的 DSL 分数不是独立证据。
4. baseline 的 runtime_err_rate 读取（373 行）无 `os.path.exists` 保护，缺文件直接崩（DSL/FB 有保护）。

### 建议改动（按优先级）
- **P0**：FB precedence 命名改用 FB 自产的 `s4_normalized.json`（含 operation 名）而非 GT route_sheet，让端到端评估真正端到端；同时 resource 的 machine 命名也应用 FB 自产并配合命名归一/对齐，否则 machine 命名误差应如实惩罚。改后需重跑并重新解读 CSE-2 vs CSE-1 的差距。
- **P0**：修复 `__get_baseline_precedence_constraint_CSE_1` 第 736 行 `pred_operation`→`pred_machine`；补齐/确认 baseline 目录产物，否则明确声明 baseline 在 CSE-2 不参评、不得在论文列出其数字。
- **P1**：补统计 Compiler-ER（与论文三指标对齐）；makespan_ratio 对失败样本改为计入惩罚或单列“成功率”，消除幸存者偏差。
- **P1**：统一 baseline 与 DSL/FB 的 precedence 尺子，或在报告中明确说明尺子差异、禁止直接横向比较。
- **P2**：删除死代码 `_CSE_2`/`baseline2_CSE_2`；给 DSL 抽取器加显式测试，避免 CSE-1/CSE-2 交叉污染；baseline runtime_err_rate 读取加 exists 保护；汇总时对齐三路径的 instance 子集再求均值。
