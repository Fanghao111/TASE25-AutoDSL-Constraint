# ground_from_gt (SGE) 评估审查

评估名：`ground_from_gt`（论文中的 SGE / Schedule Grounding Evaluation，Sec. IV-F）
评估入口：`src/evaluation/evaluation.py` 第 431–478 行（`experiment_type == "ground_from_gt"`）

## 一、测试目标与方法

**目标**：单独考察 schedule grounding（排程落地）这一段能力。整条 pipeline 从工艺描述 → 约束抽取 → 转 JSP → CP-SAT 求解 → 把「语义为空的排程解」反投影回细粒度 production plan。SGE 把前段全部固定成 GT——直接读入 GT 那份 `assigned_jobs`（CP-SAT 排程解），让各流水线只从这份**相同**的排程解出发做最后一段落地：把每个 `(machine_index → [step...])` 的 step 反查 job/task 明细（operation / machine / duration / precondition / postcondition / parameters），拼回完整 production plan。理论上度量到的差异只反映 grounding 阶段。

**被评估对象形式**（都 `json.dumps` 成字符串再比）：
- **GT**：`groundtruth.py::get_grounded_production_plan`（第 90–117 行），遍历 `self.assigned_jobs`，`step[0]/step[0]+step[3]/step[1]/step[2]` 拼 start/end/job_id/task_id，再从 route_sheet 补明细。产物 `outputs/GroundTruth/instance taXX/production_plan.json`。
- **FB**：`fb_pipeline.py::ground`（第 456–509 行，s7），遍历同一份 `assigned_jobs`，从 `normalized_jsons[job_id].steps[task_id]` 补明细。产物 `s7_production_plan.json`。
- **DSL**：`dsl_pipeline.py::JSP_result2production_plan`（第 810–844 行），遍历同一份 `assigned_jobs`，从 `operation_programs` 补明细。产物 `production_plan.json`。
- **baseline**：**本分支不接 baseline**（只有 dsl / fb 两路）。

**指标算法**：
- **BLEU**（`__bleu_score`, 577 行）：两边 `json.dumps` 后按空白 `split()` 当 token，nltk `sentence_bleu`，smoothing method4。
- **EMKVP**（代码名 `__rouge_score`, 629 行，**非标准 ROUGE**，对应论文 Exact Match of Key-Value Pair）：先 `flatten_structure`（662 行）把 JSON 拍平成 `路径 → 标量`；对 candidate 每个 KV，取末段 key 名，在 GT 找「末段 key 名相同且 value 相等（str 忽略大小写）」的项，命中 `C+=1; break`；匹配时用 `continue` 跳过含 `start/end/job_id/task_id` 的 key。返回 P=C/X, R=C/Y, F1=2C/(X+Y)。
- **聚合**：`evaluation.py` 只写 per-instance list；求均值在 `scripts/generate_comparison_report.py`（`_mean`），读 `CEILING_GROUND_DIR/ground_from_gt_fb_{bleu,rouge}.json`。

**与 CPE（`production_plan` 分支）的差异**：同一份 GT、同一对 BLEU/EMKVP 方法、同一 tokenize；唯一区别是 candidate 来源——CPE 来自完整 LLM 路径且接 4 路（baseline/baseline2/dsl/fb），SGE 来自读 GT `assigned_jobs` 后的纯 grounding、只 2 路无 baseline。

## 二、讨论过程

采用说明者 / 质疑者双 agent 讨论，质疑者亲自核对代码、两个生成函数与 ta71 真实 sample，协调者对全部论断逐条独立复核。

- **A0（说明）**：厘清 SGE「隔离到只剩 grounding」的目标、GT/FB/DSL 三方均源自同一 `assigned_jobs`、BLEU 与 EMKVP 算法、无 baseline、与 CPE 的差异。指出 `__rouge_score` 实为 EMKVP。
- **B1（质疑，6 问）**：见下，六点经协调者逐条核实**全部成立**。
- **B2（收束）**：区分「设计固有局限」与「可修实现 bug」，凝练严重度。关键量化推论：一旦修好 rouge 分母 bug（扣减跳过键），因 candidate 与 GT 同构近乎相同，EMKVP 会跳到 ≈0.995，SGE 随即退化为 identity 检验、方法间无区分度；即当前「未满分（≈0.66）」完全是分母 bug 的副产品，而非任何 grounding 质量信号。

协调者独立核实的关键证据：
1. `outputs/Evaluation/FB-gt-ceiling_from_gt_sched/ground_from_gt_fb_bleu.json`：10 个 instance BLEU 全 ≈0.858，彼此差 < 0.002。
2. 同目录 rouge.json：Precision / Recall / F1 三数组**逐元素完全相等**（首元素 0.6627402355858648…），即每 instance P==R==F1。数学上意味 X==Y（candidate 与 GT flatten 后扣除跳过键前的 KV 数相等）。
3. `outputs/Evaluation/ground_from_gt_dsl_bleu.json = []`、`ground_from_gt_dsl_rouge.json = {Precision:[],Recall:[],F1:[]}`——**DSL 一路无任何产出**；`find outputs -path "*from_gt*" -name production_plan.json` 无 DSL 命中。
4. GT (`ta71/production_plan.json`) 与 FB (`ta71/s7_production_plan.json`) 逐字段值几乎完全相同（连 "42 mins"、"1200 RPM" 原文都一致），唯一差异是顶层 machine 名（GT `"Shear (Sheet Metal)"` vs FB `"machine_11_unknown"`）与字段书写顺序。
5. `groundtruth.py::get_grounded_production_plan` 与 `fb_pipeline.ground` / `dsl_pipeline.JSP_result2production_plan` 确为**同构的反投影逻辑**，都遍历同一份 `assigned_jobs`。

## 三、结论

### 合理之处
- 「隔离最后一段 grounding、把前段固定成 GT」这个设计意图本身正确，是一种理想的消融/ceiling 思路。
- BLEU 的 whitespace tokenize（577 行注释）比按字符切要合理，避免 JSON 模板字符灌水。
- EMKVP 跳过 start/end/job_id/task_id 的初衷（这些是排程给定、非 grounding 产物）方向是对的——但见下方存疑。
- SGE 作为 CPE 的 ceiling（前段无误差时的上限）在报告脚本中被明确这样使用（`generate_comparison_report.py` 第 183 行注释）。

### 存疑（方法层面）
- **[最严重·设计固有] SGE 本质是在测 identity，而非 grounding 能力**。GT、FB、DSL 三方 candidate 都由**同一份 `assigned_jobs`** 经**同构反投影**生成，明细来源在 from_gt 场景下内容一致。因此 candidate ≈ reference 被设计钉死，BLEU 全 ≈0.858、10 instance 差 < 0.002 的「天花板」是这个结构决定的，不反映任何方法差异。改代码也解决不了——是范式问题。
- **[设计固有] grounding 的真正难度未被评分**。EMKVP 跳过 start/end/job_id/task_id，恰恰扣掉了「排程落地」最核心的时间/顺序/机位内容；剩下参与算分的 operation/duration/precond/postcond/parameters 全是从 step 明细**直接查表拷贝**、不含任何排程决策。metric 度量的是「查表对不对」。
- **[设计固有] EMKVP 精确匹配过脆**。对同义命名、单位写法（"42 mins" vs "42 minutes" 判 0）、list 顺序极敏感；因 SGE 前段全固定，差异被压到最小，这个脆弱性在 SGE 里暂未暴露，但使分数无区分度。
- **SGE 相对 CPE 的增量存疑**：同 GT、同 metric，仅 candidate 源不同；SGE 未新测任何 grounding 独有维度，独立列项的信息量有限（其主要价值仅在于充当 CPE 的 ceiling 参照）。

### 存疑（实现层面）
- **[最严重·可修 bug] EMKVP 分母未扣减跳过键，造成假天花板 ≈0.66**。active 的 `__rouge_score`（636–637 行）令 `X=Y=len(全量 flatten)`，匹配循环里只 `continue` 跳过 start/end/job_id/task_id，**从不扣减 X/Y**。于是即便 candidate 与 GT 完全相同，命中数 C 也只是「非跳过键数」，P=R=F1=C/全量 ≈0.66 被永久钉死。对比：弃用的 `__rouge_score_similarity`（602–609 行）明确会 `X-=1/Y-=1` 扣减。这解释了产物里 0.6627402… 的固定值。**最小修法**：在 active 版按 similarity 版同样扣减 X/Y（或匹配前先过滤跳过键再取 len）。
- **[可修 bug] P==R==F1 恒等 → rouge 退化为单一数、零区分度**（X==Y 所致），三列冗余。修好分母 bug 并让 X/Y 分别只统计各自的可匹配键后可恢复三者独立。
- **[可修 bug] FB 顶层 machine 反查失败写成 `machine_XX_unknown`**（`ground` 463–466 行，`machine_index >= len(self.machines)`），GT 是真实机名。因顶层 machine 键数量远少于 step 级键，错误被稀释、BLEU/EMKVP 几乎不掉分 → metric 对「排程落到哪台机器」这个 grounding 核心输出不敏感。**最小修法**：修 `self.machines` 索引对齐（machine_index 越界），并在 metric 中不把顶层 machine 当可忽略项。
- **[可修 bug] DSL-SGE 空跑**：evaluation 读 `dsl_pipeline_dir/subfolder/production_plan.json` 用 `if os.path.exists` 静默保护，from_gt 场景 DSL 样本从未生成，产物全空。论文表格若出现 DSL-SGE 数字，则口径来源存疑。**最小修法**：生成 from_gt 场景的 DSL 产物，或在缺失时显式报错而非静默跳过。

### 建议改动（优先级）
1. **[P0]** 修 `__rouge_score` 分母 bug（扣减跳过键），否则所有 EMKVP 数字系统性偏低且封顶 ~0.66，不可解释。**注意**：修好后，因 candidate 与 GT 同构近乎相同，EMKVP 会跳到 ≈0.995——这将使 SGE 退化为 identity 检验、作为「区分不同方法 grounding 能力」的评估**彻底失效**（所有方法都近满分），从反面坐实「SGE 在测 identity」这一范式问题，也说明当前 ≈0.66 完全是 bug 副产品、非质量信号。
2. **[P0]** 重新审视 SGE 的定位：明确它只是 CPE 的 ceiling 参照，而非独立能力评估；或改造使 candidate 不再与 GT 同构（例如让 grounding 从中间语义而非直接查 GT 明细取值）。
3. **[P1]** 修 FB machine 反查越界；让 metric 对机位错误敏感。
4. **[P1]** 补齐或显式报错 DSL-SGE 产物，避免静默空跑。
5. **[P2]** 合并冗余的 P/R/F1 三列，或在扣减后恢复三者独立含义；EMKVP 增加单位/同义归一，降低脆弱性。
