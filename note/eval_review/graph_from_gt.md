# graph_from_gt (CSE-1) 评估审查

评估名：`graph_from_gt`（论文 CSE-1，约束生成模块 CGM 的隔离评估）。
审查方式：说明者 agent_a + 质疑者 agent_b 双人讨论，协调者核实一手数据（ta71 复算）。

## 一、测试目标与方法

**目标**：以 GT 的 route sheet 为输入，屏蔽前段自然语言抽取（CAM），只考察"从一份干净 route sheet 出发生成的约束图（constraint graph）质量"。代码入口 `experiment_type == "graph_from_gt"`（`src/evaluation/evaluation.py` 第 229–330 行），逐个 GT 子文件夹计算。

**指标**：
- **accuracy_rate（Constraint-Acc / IoU）**：`__iou`(893) 把 recourse 约束（operation→machine 映射，元素 `"op machine"`）与 precedence 约束（有向对 `"pred succ"`）拼成一个列表，各自归一化（转小写字符串）成集合后算 `|交集|/|并集|`。
- **compile_err_rate**：本分支从不填充（恒空）。
- **runtime_err_rate**：读各管线目录 `err_rate.txt` / `s6_err_rate.txt`。
- **makespan_ratio**：候选 makespan / GT makespan（仅 GT makespan>0 时计）。

**三方抽取路径**：
- GT（reference）：recourse `__get_groundtruth_recourse_constraint`(830)；precedence 有 operation 对(847) 与 machine 对(870) 两套。
- FB（candidate, s5/s6）：recourse 复用 `__get_baseline_recourse_constraint_CSE_1`(读 s5_or_matrix+s5_machines+**GT** route sheet)；precedence 用 `__get_unified_precedence_constraint_CSE`(915，operation 对)。比较时 reference 用 **operation** precedence。
- DSL（candidate）：recourse `__get_DSL_recourse_constraint_CSE`(743)；precedence `__get_DSL_precedence_constraint_CSE`(753，Pred/Succ)。比较时 reference 用 **operation** precedence。
- baseline：precedence 用 `__get_baseline_precedence_constraint_CSE_1`(720，**machine** 对)。比较时 reference 用 **machine** precedence（与 FB/DSL 不同）。

**GT 来源**：GT or_matrix 的 dependencies（前驱依赖图）由 CP-SAT 求解得到，可能存在多个等价最优解。

## 二、讨论过程

### A0（说明者）
见上"测试目标与方法"，并指出关键事实：FB/DSL 与 GT 比 operation precedence，baseline 与 GT 比 machine precedence，三方 reference 不统一；GT dependencies 依赖 CP-SAT 可能多解。

### B1（质疑者，含 ta71 一手复算）
质疑者亲自复算 ta71，得到 combined IoU=0.66794，与 `graph_from_gt_fb.json` 第一个值逐位吻合（复现忠实）。核实：recourse IoU 实为 1.0（`.lower()` 后 GT `"Shear (Sheet Metal)"` 与 FB `"shear (sheet metal)"` 对齐），拖低总分的是 precedence。提出六点：

- **Q1（方法）** IoU 集合去重丢弃 precedence 重数。ta71：GT operation precedence 原始 418 条→去重 144 条（65% 重复），FB 439→185；两侧压缩比不同，IoU 分子分母构成本就不等价。为何不用多重集 IoU / 加权 F1？
- **Q2（方法）** makespan_ratio=10.04（GT 1411 vs FB 14160）甚至 36.45。且同 job 内 GT 与 FB 每步 duration 都不同（GT job0=[83,59,49,84,35,68]，FB=[83,70,59,35,25,58]）——FB 连时长/结构都重建了。数量级、求解器不一致下用比值衡量"图质量"无解释力。
- **Q3（方法/实现，新证据）** from_gt 并未真隔离前段：同一份 GT route_sheets 下，GT job0 依赖=[[],[0],[],[2],[3],[4]]（第 3 步无前驱、第 4 步跳接第 2 步），FB job0=[[],[0],[1],[2],[3],[4]]（纯链式）。FB 的 s5_or_matrix dependencies 是 FB 管线自己重建的，precedence 差异里混入了 FB 的 or_matrix 重建误差，"只考察约束图生成"的前提被击穿。
- **Q4（方法）** CP-SAT 多最优解使 precedence 正确答案不唯一（串行 vs 并行等价拓扑），Q3 的 `[2]` 跳接 vs `[1]` 链接很可能都合法。逐对精确 IoU 把语义等价解判为错，系统性低估。
- **Q5（实现）** `graph_from_gt_dsl.json` 全空，全仓库 from_gt 目录找不到 `operation_programs.json`；baseline json 因 line 326 门槛未写。CSE-1 实际只有 FB 一方有数，"三方对比"不成立。
- **Q6（实现）** (a) FB/DSL 比 operation precedence，baseline 比 machine precedence，分母语义不同，IoU 不可横向比；(b) `__get_baseline_precedence_constraint_CSE_1` 第 736 行引用未定义 `pred_operation`（前文只有 `pred_machine`）→每次迭代抛 NameError 被 `except: continue` 吞→baseline precedence 恒为空，baseline accuracy 实际只算了 recourse 一半。
- **附** compile_err_rate 三方恒空，评估器从不追加该字段，磁盘 `s7_compile_errors.txt` 存在却未被读。

### 协调者裁决
A0/B1 已充分暴露分歧，B1 的每条质疑均带一手复算证据且可复现。协调者独立核实以下证据全部为真：DSL 产物全空、FB makespan_ratio=[10.03,36.45]、compile_err_rate=[] 恒空、accuracy 3 个而 makespan_ratio 仅 2 个（样本数不齐）、机器 label 大小写差异、第 736 行 `pred_operation` bug。B1 的判定成立。

## 三、结论

### 合理之处
- 隔离前段的**动机**正确：约束图质量应与 NL 抽取误差解耦。
- recourse 约束（operation→machine）方向清晰，`.lower()` 归一化对机器大小写差异有效（recourse IoU 在 ta71 达 1.0）。
- runtime_err_rate 直接读求解器产物，语义明确。

### 存疑（方法）
1. **IoU 集合去重丢弃约束重数**（Q1）：precedence 是可重复的有向对，去重后两侧压缩比不同（GT 65% 重复），IoU 数值不能反映真实覆盖率，横向可比性存疑。
2. **from_gt 未真正隔离前段**（Q3，最严重方法缺陷）：FB 的 s5_or_matrix dependencies 由 FB 自己重建，precedence 差异混入了 or_matrix 重建误差，评估并未做到"只考察约束图生成"。
3. **CP-SAT 多解未处理**（Q4）：precedence ground truth 不唯一，逐对精确匹配把等价拓扑判错，系统性低估。
4. **makespan_ratio 不可解释**（Q2）：比值 10~36 说明 FB 与 GT 的 makespan 在时长量级/求解器/建模上根本不可比，用比值（而非相对误差、且量级如此大）无法解读为"图生成质量"。
5. **recourse 用 dict[op]=machine 覆盖**（A0/代码事实）：同名 operation 后者覆盖前者，若同一 operation 在不同 job 用不同机器则丢信息。

### 存疑（实现）
1. **DSL/baseline 在 from_gt 下无数据**（Q5，最严重实现问题）：`graph_from_gt_dsl.json` 全空、baseline json 不存在，CSE-1 事实上只有 FB 单方，任何"对比"结论不成立。
2. **三方 reference 不一致**（Q6a）：FB/DSL 比 operation precedence、baseline 比 machine precedence，分母语义不同，IoU 不可横向比较。
3. **baseline precedence bug**（Q6b）：第 736 行 `pred_operation` 未定义，异常被 `except` 吞，baseline precedence 恒空。
4. **compile_err_rate 恒空**：列为指标却从不填充，`s7_compile_errors.txt` 未接线。
5. **样本数不齐**：accuracy 3 个 vs makespan_ratio 2 个，聚合（求均值）时分母不一致。

### 建议改动（按优先级）
- **P0** 修复 from_gt 隔离（Q3）：FB/DSL 的 precedence 应直接从 **GT 的 dependencies** 抽取，或明确评估边界并在论文中改述"半隔离"，否则结论无效。
- **P0** 补齐 DSL/baseline 的 from_gt 产物（Q5），否则 CSE-1 不能称"对比"；若无法补齐，报告中删除或明确标注缺失。
- **P1** 修 baseline 第 736 行 `pred_operation`→`pred_machine`（Q6b）；统一三方 precedence 语义（都用 operation 或都用 machine）（Q6a）。
- **P1** IoU 改为保留重数的多重集 IoU 或加权 F1（Q1）；对 CP-SAT 多解做等价类归一或用可达性/传递闭包比较（Q4）。
- **P2** makespan_ratio 改用相对误差并核对 makespan 单位/求解器一致性（Q2）；recourse 改为集合而非 dict 以保留一对多（方法存疑 5）。
- **P2** compile_err_rate 接线读 `s7_compile_errors.txt` 或从指标中删除；对齐 accuracy 与 makespan_ratio 的样本集合。
