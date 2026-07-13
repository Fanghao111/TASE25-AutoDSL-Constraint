# dae (DAE) 评估审查

> 审查对象：`src/evaluation/evaluation.py` 第 479–575 行（DAE 分支 + 内嵌 `calculate_vmr`）
> 论文：`2510.02679v1.pdf` Sec IV-G（Protocol for the Automated Adaptation Evaluation）+ Sec V-E（结果）
> 上游依赖：CPE（production_plan）产物

## 一、测试目标与方法

**目标**：DAE 是一个 meta-study，验证「同一套方法在多个生产场景（10 个 production scenario）下自动适应的稳定性与可扩展性」。论文期望这 10 组「perform uniformly well, without case-specific bias」——即跨场景表现一致、无 case-specific 偏差。论文将此定位为 "one-approach-multiple-domain" 评估。

**指标 VMR（Variance-to-Mean Ratio）**：
- **论文口径**：论文正文明确 "VMR ... where higher values are more desirable"，且 Sec V-E 原文 "Ours demonstrate a high mean and low variance, resulting in a trend where the VMR of Ours significantly surpasses that of TSL"。即：高均值 + 低方差 → 高 VMR → 越好。
- **代码实现**（evaluation.py:559–562）：`calculate_vmr(values) = np.var(values) / np.mean(values)`（`np.var` 为总体方差，除 N；`mean == 0` 返回 `np.nan`）。这是标准统计学 VMR / 离散指数：值越大代表越离散（越不稳定/越不理想）。

**被评估对象与聚合口径**：DAE 不重跑推理，直接读取 CPE（production_plan 分支）写出的 `production_plan_{dsl,baseline,baseline2}_{bleu,rouge}.json`。组织成 4 指标（BLEU, Precision, Recall, F1）× 3 方法（`dsl_data`/`baseline_data`/`baseline2_data`），每组 `for i in range(10)` 取 **10 个 production scenario / instance** 的值，对每组 10 个值求 VMR。三方对应论文的 **Ours / MSL / TSL**。

**依赖关系**：DAE 完全下游依赖 CPE，本身不含任何独立正确性校验或统计检验，只做「读取 → 按指标/方法重排 → 求 VMR」。CPE 的任何问题都会级联到 DAE。

## 二、讨论过程

**说明者（A0）** 梳理了目标、VMR 双口径、聚合维度（10 = scenario）、三方 = Ours/MSL/TSL，并主动标注了「代码 var/mean 越小越好」与「论文越高越好」的方向差异。

**质疑者（B1）** 核对真实数据后提出 6 问，核心两点：
- (A) 代码 `var/mean` 与论文「越高越好 / Ours surpass TSL」方向完全相反，必有一处错；
- (B) DAE 依赖的顶层 `production_plan_baseline_bleu.json` / `baseline2_bleu.json` 是**空数组 `[]`**、顶层 `production_plan_dsl_bleu.json` **不存在**（真实数据只在 `DSL-models_*/`、`FB-models_*/` 子目录），且全项目**无任何 `dae.json`/`dae_metric.json`/`dae_vmr.json` 产物** → DAE 从未端到端跑通。
- 另问：三方数据源不对齐（dsl 读子目录 / baseline 硬编码读顶层）、10 个是否真 domain shift、小样本小均值 VMR 不稳与量纲不可比、只复用一种测试且无收敛曲线 / 无 Kruskal-Wallis 检验实现。

**说明者（A1）** 逐条回应，对 Q1（方向矛盾）、Q2（未跑通、不可复现）如实承认，不作辩护；对 Q3/Q6 证实缺陷存在；对 Q4/Q5 区分「论文声称」与「代码实现」，承认统计选型（总体方差、非无量纲）站不住。

**质疑者（B2）** 收束：Q1、Q2 真实成立且最重（硬矛盾 + 不可复现）；Q3、Q6 成立但较轻（少做 / 做偏）；Q4、Q5 降级为待补证据 / 方法改进项。

## 三、结论

### 合理之处
- 复用 CPE 产物做二次统计、避免重复推理，思路上经济合理。
- 「跨多场景稳定性」这一评估维度本身有价值，VMR 作为离散度指标方向上可用（只要口径统一）。
- 三方对比框架（Ours vs MSL vs TSL）与论文主实验一致。

### 存疑（方法）
- **VMR 语义方向自相矛盾（致命）**：论文要「越高越好、高均值低方差胜出」，但标准 VMR = var/mean 下高均值低方差得到的是**更小**值，与论文 "surpass" 逻辑互斥。要自洽应改用 CV（std/mean）或信噪比（mean/var），至少一方（论文文字 or 代码公式）是错的。
- **10 个 scenario 是否真 domain shift 无从佐证**：代码只对 10 个 index 循环，无任何 domain 标注；若这 10 组只是同一 production_plan 任务的规模 / 实例变体，则「跨 domain 自适应稳定性」命题落空，DAE 度量的实为「同任务内抽样波动」。
- **小样本 + 小均值下 VMR 不稳且不可横向比较**：n=10、BLEU 均值≈0.178，方差被小分母放大；4 个指标分母量级差数倍（BLEU≈0.18 vs Precision≈0.41…），未无量纲化就并入同表横向比较无意义。

### 存疑（实现）
- **DAE 从未端到端跑通、结果不可复现（致命）**：顶层 `production_plan_baseline_bleu.json` / `baseline2_bleu.json` 为空数组，顶层 `production_plan_dsl_bleu.json` 不存在，全项目无任何 dae 产物。按现仓库运行 dae 分支会在 `dsl_bleu[i]` IndexError，或 baseline/baseline2 侧 `np.mean([])`→nan 导致 VMR 全 nan。论文 V-E 的 VMR 数字在本仓库无迹可循、无法复现。
- **三方数据源不对齐**：`dsl` 读 `self.dsl_eval_out_dir`（按模型分的子目录），`baseline`/`baseline2` 却**硬编码**读顶层 `outputs/Evaluation/`。即便补齐数据，三方也可能来自不同模型 / 不同批次，VMR 三方对比在数据源层面不可比。
- **总体方差而非样本方差**：`np.var` 默认 `ddof=0`，小样本下系统性低估离散度，论文未说明。
- **覆盖面远小于论文声称**：论文 IV-G 说 DAE 要观测 convergence 并覆盖「all previous experiments」，代码只读 production_plan 一种产物，无收敛曲线；论文 V-E 的 Kruskal-Wallis H(9)=2.605 检验在代码里**完全无实现**，该数字来源无法追溯。

### 建议改动（按优先级）
- **P0 — 统一 VMR 语义**：改用 CV（std/mean）或 mean/var，使「高均值低方差 = 更优」成立，同步订正论文文字与代码公式，二者对齐。
- **P0 — 补齐可复现产物**：修复读取路径，让 dsl/baseline/baseline2 三方 BLEU/ROUGE 都有真实数据，消除空数组与 IndexError，能端到端跑出 `dae_vmr.json`。
- **P1 — 三方同源**：统一三方的模型 / 批次 / 读取路径（dsl 与 baseline 都从子目录或都从顶层取），保证同一模型、同一批 10 个 scenario。
- **P1 — 补统计检验**：实现 Kruskal-Wallis 检验与收敛曲线，对齐论文 IV-G / V-E 声称的分析范围。
- **P2 — 统计稳健性**：VMR 改用无量纲 CV，方差用 `ddof=1`，标注小样本置信区间。
- **P2 — domain shift 佐证**：补 10 个 scenario 的异质性量化（分布距离 / 任务规模差异指标），否则弱化「跨 domain」表述，改称「跨 instance 稳定性」。

**总判定**：两个 P0（VMR 方向矛盾、从未跑通不可复现）未修复前，论文 V-E 的 DAE 评估既不自洽也不可复现，结论不成立。
