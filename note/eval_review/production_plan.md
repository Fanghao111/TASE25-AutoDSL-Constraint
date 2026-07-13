# production_plan (CPE) 评估审查

> 双 agent 对抗式审查。目标：核查 CPE（Complete Pipeline Evaluation，代码 `experiment_type == "production_plan"`）的指标定义、GT 构造、baseline 设置与实现细节是否存在实质漏洞。
> 关键代码：`src/evaluation/evaluation.py` L74–158（production_plan 分支）、`__bleu_score`(577)、`__rouge_score`(629)、`flatten_structure`(662)；GT 生产 `src/experiment/groundtruth.py` L90–117；聚合 `scripts/generate_comparison_report.py`。

## 一、测试目标与方法（agent a 综述）

**目标**：论文 Sec IV-C「完整流水线评估」。检验一整条从原始生产资料到最终「生产计划」(production plan) 的端到端流水线，评价各方法产出的排产计划与专家真值的接近度。production plan 即把 JSP 求解器输出的无语义调度，回填成「某机器在某时间段执行某操作及其参数」的可解释计划。

**被评估对象（4 条流水线）**，均与同一 GT 比：
- `baseline` = 论文 **MSL**（三模块序列，`baseline.py`）→ `outputs/Baseline-CPE_CAE_CSE-2/…/production_plan.json`
- `baseline2` = 论文 **TSL**（两模块序列，`baseline2.py`）→ `outputs/Baseline2-CPE_CAE_CSE-2/…/production_plan.json`
- `dsl` → `outputs/DSLPipeline-…/…/production_plan.json`
- `fb` = 本文方法 (Ours, s7 产物) → `outputs/FB-…/…/s7_production_plan.json`
- **注意**：当前工作副本（deepseek 复现）下 `Baseline-*`/`Baseline2-*` 目录不存在，`os.path.exists` 逐个跳过，baseline/baseline2 列表实跑为空。

**GT 来源**（`groundtruth.py` L90–117 `get_grounded_production_plan`）：CP-SAT 求解 `assigned_jobs` → 按机器组织，每步写 `start/end/job_id/task_id` → 从 route_sheet 回填 `operation/duration/precondition/postcondition/parameters`（`step_info.update(step_data)`）。含调度信息的完整 plan。

**两个指标**（reference=GT plan 的 JSON，candidate=各流水线产物 JSON）：
- **BLEU**（`__bleu_score` 577）：整个 plan `json.dumps` 成字符串，按空白 `split()` 当 token，`nltk.sentence_bleu([ref.split()], cand.split(), SmoothingFunction().method4)`。
- **「ROUGE」/EMKVP**（`__rouge_score` 629）：名叫 ROUGE-L 但实为论文的 **EMKVP（Exact Match of Key-Value Pairs）**，非标准 ROUGE-L（无 LCS/n-gram/顺序）。`flatten_structure` 拍平成 `{key路径: 值}` → 剔除末段名含 `start/end/job_id/task_id` 的项 → candidate 每键去 reference 找「末段键名相同（小写）且值相等（str 时小写相等）」，命中 `C+=1; break`。返回 `P=C/X, R=C/Y, F1=2C/(X+Y)`，X/Y = 过滤后 candidate/reference 字段数。
- 聚合：evaluation.py 只 append 逐 instance 分数并原样 `write_json`；单值平均在 `scripts/generate_comparison_report.py` 的 `_mean`（对 10 instance 简单算术平均，BLEU 取均值、ROUGE 取 F1 列均值，空 list 返回 None 不污染）。

## 二、讨论过程

### 第 1 轮

**agent b 提问**

- **Q1 (实现)**：过滤用子串 `"end" in actual_key`；`split(".")[-1]` 对 `precondition[0]` 返回带中括号末段。后果：precond/postcond 位置绑定；含 end 子串合法字段被误杀。
- **Q2 (方法)**：EMKVP 双重 for 全表扫，砍掉路径前缀只留末段名+值，无视机器/job/工序归属。高频值(LathedPart/MetalBar)几乎必命中，退化为「键值多重集重叠」，与排产正确性无关。
- **Q3 (实现)**：`C+=1; break` 只跳内层、不消费 reference 键。同一 GT 值被多个 candidate 键重复命中 → C 虚高、P/R 齐涨、破坏二部匹配语义。
- **Q4 (方法)**：被评字段全来自 route_sheet 回填，CP-SAT 唯一独有产物(start/end/job/task)恰被过滤。CPE 到底评「排产」还是「route_sheet 抄写」？
- **Q5 (实现)**：GT 内层 machine 被 route_sheet 覆盖(与外层 CP-SAT 分派不一致)；DSL sample 缺 machine/operation/duration 整类键 → X 系统性做小 → `P=C/X` 被漏答抬高。DSL 与 FB 的 Precision 不可比？
- **Q6 (方法+实现)**：BLEU 对 JSON whitespace split 衡量的是结构标点+调度数字；baseline 目录不存在→空 list 原样写→无平均。论文 baseline 数字从哪来？聚合口径？

**agent a 回应**

- **Q1**：属实。list 元素用 `[i]` 拼接无点号，`split(".")[-1]` 对 `precondition[0]` 返回 `precondition[0]`（中括号保留），故 precond/postcond 长度或顺序不同即判错。键名匹配大小写不敏感(650)，值先严格 `==` 再 str 小写降级(651-655)：`"1200 RPM"`vs`"1200 rpm"` 过、`"42 mins"`vs`"42 minutes"` 判错、无单位/数字归一化。「含 end 误杀」机制成立但当前 sample 字段名未触发。
- **Q2**：完全属实。但论文 L726-741 明说 EMKVP **有意**只评字段-值一致性、procedure 级交 BLEU、排产正确性归 CSE。属论文声明过的设计取舍；可批评点是「作为 complete pipeline 主指标的名实落差」。
- **Q3**：属实，是真 bug。内层命中不标记 reference 键，5 个 `postcondition[0]==LathedPart` 会各配同一 GT 键 `C+=5`，C 可逼近 X，P/R/F1 齐虚高，对高频重复值计划更有利。
- **Q4**：属实。`groundtruth.py` L101-111 唯一来自 CP-SAT 的 start/end/job/task 恰被过滤，其余全 route_sheet 回填。CPE 实为评 **grounding 保真度**（把纯数字 schedule 贴回工序语义），论文语境自洽；排产正确性由 CSE 的 IoU/makespan 负责。「CP-SAT 白算」不成立。
- **Q5**：两后果都属实。GT 内层 machine=route_sheet 的(update 覆盖)，三家机器名风格各异易判错。DSL 缺整类键 → X 做小 → `P=C/X` 抬高（漏答提分），DSL vs FB 的 Precision/F1 **结构性不可比**；Recall=C/Y 用 GT 键数相对公平。Q3(抬 P) 与 Q5 缺键(抬 DSL 的 P) 方向叠加不抵消。
- **Q6**：(a) 属实，token 大半是 JSON 标点+调度数字，method4 单参考几千 token，BLEU 反映 JSON 结构+数字 n-gram 重合，弱于语义。(b) 属实，outputs 下无 Baseline-* 目录，production_plan 分支无 `np.mean`（唯一在 560 属 DAE）。**新发现**：聚合在 `scripts/generate_comparison_report.py` 的 `_mean`，对 10 instance 简单算术平均。论文 baseline(MSL/TSL) 数字来自原始 AutoDSL 用 **GPT-4o** 跑 baseline.py 生成 `Baseline-*` 后聚合；本 deepseek 副本未重跑 baseline，故目录空、无法从当前仓库证实论文表内 baseline 具体数值。

### 第 2 轮（收束）

agent b 独立复核 `__rouge_score`(642-660)、`flatten_structure`(676-681)、`_mean`(32-34) 三处关键代码，与 agent a 陈述完全吻合，未发现新的事实性错误。**裁定：全部达成一致，无需追问。** Q1/Q3/Q5 承认为真实缺陷（Q3 为确认的 bug），Q2/Q4/Q6 经论文声明与代码事实澄清了质疑者的部分误判（CPE 是 grounding 保真度评估，EMKVP 归属无视是声明过的分工取舍）。

## 三、结论

### 合理之处
- CPE 定位为 **grounding 保真度**评估（把无语义 schedule 贴回工序语义的还原度）在论文语境自洽；排产正确性由 CSE (IoU/makespan) 承担，分工清晰。
- 过滤 start/end/job_id/task_id 是有意为之——这些是调度产物，不属于「知识回填」的评价范围。
- BLEU 注释中已修正过一次「按字符切词导致模板字符灌水」的问题，改为按空白切词，体现作者对 tokenization 有意识。
- `_mean` 对空 list 返回 None，空 baseline 目录不会污染最终均值。

### 存疑之处（评价方法层面）
- **EMKVP 无归属**（Q2）：砍掉路径前缀只留末段名+值，无视机器/job/工序。高频值(LathedPart/MetalBar)几乎必命中，实测退化为「键值多重集重叠」。虽属论文声明的设计取舍，但作为 complete pipeline 主指标存在**名实落差**——读者易误以为它评「生产计划质量」，实际粒度只到字段值多重集。
- **BLEU 语义弱**（Q6a）：对 `json.dumps` 全串 whitespace split，多数 token 是 JSON 标点与调度数字，实际衡量的是「JSON 结构+数字 n-gram 重合」而非工序语义。
- **值匹配脆弱**（Q1）：无单位/同义词/数字归一化。`"42 mins"`vs`"42 minutes"`、`"medium"`vs`"1200 RPM"` 一律判错，惩罚的是字面而非语义。

### 存疑之处（实现层面）
- **【真 bug】重复命中，非二部匹配**（Q3）：`C+=1; break` 不消费已匹配 reference 键，同一 GT 值可被多个 candidate 键重复计入，`C` 可超二部匹配上限，P/R/F1 系统性虚高，且对高频重复值计划更有利。
- **【公平性】DSL 缺键抬高 Precision**（Q5b）：DSL sample 缺 machine/operation/duration 整类键 → `X` 系统性变小 → `P=C/X` 因漏答被抬高。**DSL vs FB 的 Precision/F1 结构性不可比**；且 Q3(抬 P) 与 Q5(抬 DSL 的 P) 方向叠加不抵消。Recall=C/Y 相对公平。
- **precond/postcond 位置绑定**（Q1a）：`precondition[0]` 只能匹配 `precondition[0]`，列表长度或顺序不同即判错，非集合匹配。
- **GT 内层 machine 被 route_sheet 覆盖**（Q5a）：与外层 CP-SAT 分派机器不一致，三家命名风格各异，machine 字段几乎恒判错。
- **子串过滤隐患**（Q1b）：`"end" in actual_key` 会误杀任何末段名含 `end`/`start` 子串的合法字段（当前 sample 未触发，属潜在陷阱）。
- **baseline 数值不可复现**（Q6b）：本副本未重跑 baseline.py，`Baseline-*` 目录空，无法从当前代码核验论文表内 MSL/TSL 数字（源于原始 GPT-4o 运行）。

### 建议改动（优先级排序）
- **P0** 修复 Q3 二部匹配 bug：内层命中后用 set 记录已消费的 reference key 并跳过，使 `C ≤ min(X,Y)`。这同时缓解 P/R/F1 虚高。
- **P0** 解决 Q5b 可比性：统一 candidate schema（缺失字段按 GT 键集补空或计入 X 分母），或改用只以 GT 键为基准的对齐匹配，确保 DSL/FB 的 Precision 口径一致。
- **P1** 重命名指标：代码/论文里把 `__rouge_score` 明确标为 EMKVP，避免与标准 ROUGE-L 混淆；并在论文中明示 CPE 评的是 grounding 保真度而非排产。
- **P1** precond/postcond 改集合匹配（去下标后按值多重集比对），去除位置绑定脆弱性。
- **P2** 值匹配加单位/数字归一化（如 `"42 mins"`≈`"42 minutes"`）；子串过滤改为末段名精确等于白名单集合，消除 end/start 误杀隐患。
- **P2** 若要在本副本对比 baseline，需用相同模型重跑 baseline.py/baseline2.py 生成 `Baseline-*`，否则报告中应显式标注 baseline 缺失。
