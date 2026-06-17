# Code Review Report: `fb-test` Branch

对比 `origin/main`，由 Challenger（质疑者）和 Defender（解释者）两个 Agent 对抗式审查。

---

## 1. DSL Pipeline 代码改动的影响

| 严重度 | 问题 | 质疑者 | 解释者 |
|--------|------|--------|--------|
| **Critical** | `schedule.py:47` 早期退出返回3个值，调用方期望4个 | 当 `horizon==0` 时会崩溃 (`ValueError: not enough values to unpack`) | 确实是一个bug，需要修复为 `return {}, None, err_rate, -1` |
| **High** | 5次重试后返回`"{}"` 静默降级 | 下游 `json.loads("{}")` → 空dict → 空 operation_programs，整个 job 被跳过但无告警 | 相比原来的无限循环（永远不会终止），这是活性修复。但应增加计数器追踪失败数 |
| **Medium** | 模型/端点切换 (`deepseek-chat` → `gpt-4o`) | DSL 新产出与论文旧结果不可比 | 开发便利性改动，老方法保留在代码中可恢复。CLAUDE.md 已记录此差异 |
| **Medium** | production plan 新增 `operation/machine/duration` 字段 | 改变 ROUGE 分母（候选方 token 数增加） | 实际是**修复了 DSL 评估的不公平**——原来 GT 有这些字段但 DSL 输出没有，导致 DSL 分数被人为压低 |
| **Low** | 大量死代码（batch方法 ~470行） | 代码卫生问题，commit 说"remove dead code"但这些还在 | 保留是为了可恢复，但确实应清理 |

---

## 2. FB Pipeline 的逻辑和实现

| 严重度 | 问题 | 质疑者 | 解释者 |
|--------|------|--------|--------|
| **Medium** | 归一化只用1个样本（中位数大小的JSON） | 如果中位数样本非典型，字段映射不完整 | 字段频率统计覆盖全部数据，只是给LLM看的 *样本* 受限。频率阈值保证只考虑常见字段 |
| **Medium** | `_find_steps()` 硬编码key，fallback取第一个 list-of-dicts | 若LLM输出 `{"metadata": [{}], "steps": [...]}` 会错选 metadata | 提取 prompt 引导 LLM 产出顶层数组，实际触发概率低。可改进优先级 |
| **Medium** | CSE-1 模式绕过整个 FB 管线，直接用 GT 数据 | 只测试了 `derive_constraints()` 在完美输入上的表现，弱化了测试力度 | 这是 CSE-1 的定义——给定正确路线单评估约束编译。DSL 的 CSE-1 也是同样逻辑 |
| **Low** | 前驱检测只看 job 内 (`j < i`) | 无法捕捉跨 job 依赖 | 这是 JSSP 问题定义决定的——经典 JSP 中 job 间独立 |
| **Low** | 验证阈值（30%/90%）看似随意 | 边缘情况（如15台唯一机器）可能误报 | 有 fix-mapping 重试机制兜底，最多多一次LLM调用 |

---

## 3. Evaluation 中对 DSL 改动的影响

| 严重度 | 问题 | 质疑者 | 解释者 |
|--------|------|--------|--------|
| **High** | 子文件夹枚举改为 `groundtruth_dir_path`，各方法独立评估 | 不同方法可能在不同实例子集上评估，均值不可比 | 修复了旧逻辑的问题：原来 baseline 没跑的实例 DSL 也被跳过。但应报告每方法的 N 值 |
| **High** | DAE 硬编码 `range(10)` 但实例列表只有 ta71 | 当前配置必然 `IndexError` 崩溃 | 这是开发阶段暂时缩小实例列表的副作用，部署前需恢复为10个实例或修改 DAE |
| **Medium** | SGE 移除了 baseline/baseline2 | 如论文还报告 baseline SGE 结果，需另行评估 | 设计意图是用 FB 替代 baseline2 作为比较对象。DSL 评估不受影响 |
| **Positive** | baseline2_rouge 存在 bug（main 分支上用了 precision 代替 recall/F1） | — | fb-test 分支**修复了这个 bug** |

---

## 4. DSL 方法的 Evaluate 是否合理

| 严重度 | 问题 | 质疑者 | 解释者 |
|--------|------|--------|--------|
| **Critical** | `evaluation.py:722` — `pred_operation` 未定义 | `__get_baseline_precedence_constraint_CSE_1` 始终返回空列表（bare except 吞掉 NameError）。**Baseline CSE-1 优先约束评估始终为0** | **这是 main 分支就有的 pre-existing bug**，不是 fb-test 引入的。但它意味着论文中 baseline 的 CSE-1 precedence 分数是错误的 |
| **High** | Baseline 与 DSL 对比的 ground truth 不同 | Baseline 对比 machine-level precedence，DSL 对比 operation-level precedence，IoU 分数不直接可比 | 语义上正确——各方法在其自身表示层级做比较。DSL 输出的就是 operation pairs，baseline 输出的是 machine indices |
| **Medium** | dict 去重导致同名 operation 只保留最后一个 machine | 低估了约束集实际大小 | GT 侧也用同样的 dict 去重，所以 IoU 两边一致性有保障（consistent bias） |

---

## 5. FB 方法的 Evaluate 是否合理

| 严重度 | 问题 | 质疑者 | 解释者 |
|--------|------|--------|--------|
| **Critical** | `__get_unified_precedence_constraint_CSE` 用 GT route sheets 按 index 查 operation name | **CSE-2 模式下**，FB 的 `normalized_jsons` 可能与 GT 步骤数/顺序不对齐，导致错误映射或被 except 吞掉 | CSE-1 对齐有保证。CSE-2 确实有位置对齐假设。但效果是 **低估** FB 质量（misalign 导致跳过/错匹配），对 FB 不利而非有利 |
| **Critical** | 同样的 index 对齐问题影响 resource constraint 评估 | `route_sheet[job_index]["route_sheet"][step_index]` 从 GT 取 operation name，与 FB 的 machine 组成嵌合体 | 这与 baseline 使用的方法 (`__get_baseline_recourse_constraint_CSE_1`) 完全一致——是统一的评估模式 |
| **High** | 所有评估辅助方法的 bare `except: continue` | 屏蔽了所有错误类型，无法诊断数据对齐问题 | 确实是代码质量问题。但在论文验证阶段，这些 except 防止了 pipeline 崩溃。应加 logging |
| **Medium** | CAE 用 LLM-discovered field_mapping 构建 route sheets | 如果 SRD 发现了错误字段角色，结果就是 garbage | 这正是端到端评估应有的行为——衡量的是包含字段映射步骤在内的整体质量 |

---

## Summary: 最严重的发现

| # | 严重度 | 位置 | 问题 | 建议 |
|---|--------|------|------|------|
| 1 | **Critical** | `schedule.py:47` | `return {}, None, err_rate` 少返回一个值 | 改为 `return {}, None, err_rate, -1` |
| 2 | **Critical** | `evaluation.py:722` | `pred_operation` 未定义 (pre-existing) | 改为 `pred_machine`（与上下文语义一致）|
| 3 | **Critical** | `evaluation.py:901` | CSE-2 模式下 FB 评估的 index 对齐假设 | 可接受（保守低估），但应在论文中说明 |
| 4 | **High** | `evaluation.py:476` | DAE `range(10)` vs 单实例配置 | 恢复10实例或改为动态长度 |
| 5 | **High** | `evaluation.py:79` | 各方法评估集合大小可能不同 | 报告每方法的评估实例数 N |
