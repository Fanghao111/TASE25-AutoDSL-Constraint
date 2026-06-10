## 论文测试总结

论文: "Automated Constraint Specification for Job Scheduling by Regulating Generative Model with Domain-Specific Representation" (IEEE T-ASE 2025 Best Paper)

### 一、测试数据集

基于10个经典 Taillard JSP 实例 (ta71-ta80)，每个实例包含 100 个 job 和 20 台 machine。原始 JSP 数据仅包含 machine ID 和 duration，论文使用 GPT-4o 进行数据增强，生成：
- 自然语言生产流程描述
- 半结构化路线表 (route sheet)
- 详细的执行配置参数（材料、产品、设备参数等）

每个 JSP 实例被视为一个独立的生产场景，需要单独的 DSL 进行约束规范。

### 二、对比方法

三种方法使用相同的基础 LLM（GPT-4o）：

| 方法 | 缩写 | 描述 |
|------|------|------|
| **本文方法** | Ours | 通过 DSL 双程序视图 (operation-centric + product-flow-centric) 调控 LLM，包含约束抽象、约束生成、调度落地三个模块 |
| **Multi-Stage-LLM** | MSL | 与本文架构对齐的三阶段纯 LLM 方法：NL->路线表->OR矩阵->生产计划（每步均为 LLM prompt） |
| **Two-Stage-LLM** | TSL | 简化为两阶段：NL->直接生成 OR 矩阵->生产计划（跳过显式路线表中间表示） |

### 三、五组实验及测试内容

#### 实验 1：完整流水线评估 (Complete Pipeline Evaluation, CPE)
- **输入**：NL 描述 / 半结构化路线表
- **输出**：落地的生产计划 (JSON 格式)
- **评估指标**：BLEU、EMKVP-Precision、EMKVP-Recall、EMKVP-F1
- **对比**：Ours vs MSL vs TSL
- **结果**：Ours 在全部 4 项指标上显著优于 MSL 和 TSL（配对 t 检验，所有 p < .0001）。Ours 能准确捕获细粒度执行配置（如 feed rate、tool type）和时间一致性（start/end/duration），MSL 部分丢失，TSL 完全失败。

#### 实验 2：约束抽象评估 (Constraint Abstraction Evaluation, CAE)
- **输入**：与 CPE 相同
- **输出**：完全结构化的路线表
- **评估指标**：BLEU、EMKVP-Precision、EMKVP-Recall、EMKVP-F1
- **对比**：Ours-CAM vs MSL-I
- **结果**：Ours-CAM 显著优于 MSL-I（BLEU: p < .0001, EMKVP-F1: p < .0001）。MSL-I 的 Precision 可能存在虚高（因为大量字段未被指定，低 Recall 导致假阴性被排除）。

#### 实验 3：约束生成评估 (Constraint Specification Evaluation, CSE)

分两个版本：

**CSE-1（隔离版）**：
- **输入**：ground-truth 完全结构化路线表（消除上游误差传播）
- **输出**：JSP 求解器格式的约束集
- **对比**：Ours-CGM vs MSL-II
- **评估指标**：Constraint-Acc (IoU)、Compiler-ER、Runtime-ER
- **结果**：Ours-CGM 在 Constraint-Acc 上显著优于 MSL-II (p < .0001)

**CSE-2（集成版）**：
- **输入**：原始 NL 描述（包含前两个模块的累积误差）
- **对比**：Ours-CAM-CGM vs MSL-I-II vs TSL-I
- **结果**：尽管存在级联误差传播，Ours-CAM-CGM 仍显著优于两个 baseline（Constraint-Acc 和 Runtime-ER 均 p < .0001）。Baseline 方法无法准确指定资源约束和优先约束，生成的 Gantt 图表现为更少的机器行和更短的时间轴。

#### 实验 4：调度落地评估 (Schedule Grounding Evaluation, SGE)
- **输入**：ground-truth 约束集生成的 JSP 调度结果
- **输出**：落地的生产计划
- **评估指标**：BLEU、EMKVP-Precision、EMKVP-Recall、EMKVP-F1
- **对比**：Ours-SGM vs MSL-III vs TSL-II
- **结果**：Ours-SGM 显著优于两个 baseline（所有指标 p < .0001 或 p < .01）。即使给定 ground-truth 调度，baseline 方法仍无法保持 start/end/duration 的一致性。

#### 实验 5：自动化场景适配评估 (Domain Adaptation Evaluation, DAE)

**元研究 (meta-study)**，包含两部分：

**Part A — 收敛性**：
- DPMM 非参数模型（操作 DSL 语义设计）在 10 个场景上均收敛
- EM 算法（产品流 DSL 语法设计）在 10 个场景上均收敛

**Part B — 跨场景一致性**：
- Kruskal-Wallis H 检验：H(9) = 2.605, p = .978，表明 Ours 的性能与场景选择无关
- VMR (Variance-to-Mean Ratio) 指标：Ours 呈现高均值低方差，VMR 显著优于 TSL；MSL 呈现低均值低方差，TSL 呈现低均值高方差（不确定性更大）

### 四、关键结论

1. DSL 双程序视图的结构化中间表示显著优于纯 LLM 直接生成
2. 显式路线表作为中间工作区的收益（MSL > TSL）大于其带来的级联误差成本
3. 自动化场景适配算法使架构无需人工 DSL 设计即可适用于不同制造场景，且性能一致
