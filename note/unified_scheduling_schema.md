# 统一调度问题 Schema 方案 (Unified Scheduling Schema)

> 目标: 让 AutoDSL / DSL Pipeline 从只支持 JSSP,扩展到 JSP / FJSP / FlowShop / OpenShop / RCPSP / MM-RCPSP 等所有主流调度问题,且不牺牲抽取质量。
> 核心思路: **一次抽取(超集 schema) + 后处理判类型 + 类型路由到对应 builder**。
> 关联文档: `pipelines.md`(现有流程)、`note.md`(设计原则)、`test_procedure.md`(测试规范)。

---

## 1. 为什么不"先判类再抽取"

分两步走的方案(先让 LLM 判类,再选对应 schema)看似清晰,实测有三个致命问题:

1. **判类本身要读懂全文**——LLM 判类阶段已经把大半信息看过一遍,再走抽取等于读两遍原文,token 和延迟翻倍。
2. **判错整体崩**——一旦第一步判成 JSP,而实际是 FJSP,第二步用了错 schema,后续 pipeline 全废。
3. **现实描述常混合**——"3 台机床 + 2 个工人共享" 是 JSP + 累积资源,任何硬分类都会丢维度。

推荐方案: **超集 schema 一次抽取,判类放到抽取后用纯代码做**。LLM 只负责"读懂原文填字段",分类和路由用确定性代码处理。

---

## 2. 超集 Schema 设计

设计原则:
- 所有可选维度都设成**可空字段**,缺失即代表该问题不涉及该维度。
- **每个 op 统一用 `candidates` 列表**——JSP 就是长度 1 的特例,FJSP 是长度 >1。
- **`predecessors` + `op_order`** 双字段统一表达 flow / job / open / DAG 结构。
- 沿用项目已有字段命名: `operation` / `machine` / `duration` / `precondition` / `postcondition` / `parameters` / `component_type` / `container`,保证 downstream 复用。
- `duration` 保持字符串 "X minutes" 单位约定(见 `route_sheet_prompt.txt`)。

### 2.1 JSON 结构

```jsonc
{
  "part_name": "<STR>",

  "machines": [
    {
      "id": "<STR>",                          // 机器 id/名称
      "unavailable": [["HH:MM", "HH:MM"]]     // 可选;停机窗口
    }
  ],

  "resources": [                              // RCPSP 用;JSP/FJSP 一般为空
    {"id": "<STR>", "capacity": <INT>}
  ],

  "jobs": [
    {
      "id": "<STR>",
      "release": "<STR>",                     // 可选;例 "0 minutes"
      "due":     "<STR>",                     // 可选;例 "300 minutes"
      "weight":  <NUM>,                        // 可选;默认 1
      "op_order": "sequential | none | dag", // sequential=JSP/Flow, none=OpenShop, dag=RCPSP
      "ops": [
        {
          "id": "<STR>",                      // op 唯一 id,用于 predecessors 引用
          "operation": "<STR>",               // 工序名(Title Case)
          "candidates": [                     // 长度=1 → JSP;>1 → FJSP
            {
              "machine":  "<STR>",
              "duration": "<STR>",            // "X minutes"
              "resources": {"<res_id>": <INT>} // 可选;累积资源占用
            }
          ],
          "predecessors": ["<op_id>"],        // 空 → 走 op_order 顺序;非空 → DAG
          "precondition":  [ {"component": "<STR>", "component_type": "<STR>", "container": "<STR>"} ],
          "postcondition": [ {"component": "<STR>", "component_type": "<STR>", "container": "<STR>"} ],
          "parameters": { "<key>": "<value>" }
        }
      ]
    }
  ],

  "objective": {
    "type": "makespan | weighted_tardiness | weighted_completion | total_flow_time",
    "weights": { "<job_id>": <NUM> }          // 部分 type 需要
  },

  "constraints": {                            // 全局约束,均可选
    "setup_times": [                           // 例: 机器 M0 上从 j1 切到 j2 需要 15 分钟
      {"machine": "<STR>", "from_job": "<STR>", "to_job": "<STR>", "duration": "<STR>"}
    ],
    "no_wait":    false,                       // 相邻 op 之间不能等待
    "preemption": false,                       // 是否允许抢占
    "batching":   null                          // 预留
  }
}
```

### 2.2 字段可选性总结

| 字段 | JSP | FJSP | FlowShop | OpenShop | RCPSP | MM-RCPSP |
|------|-----|------|----------|----------|-------|----------|
| `machines` | ✓ | ✓ | ✓ | ✓ | 可空 | ✓ |
| `resources` | 空 | 空 | 空 | 空 | ✓ | ✓ |
| `ops[].candidates` (len) | =1 | ≥1 | =1 | =1 | =1 | ≥1 |
| `ops[].predecessors` | 空 | 空 | 空 | 空 | ✓ | ✓ |
| `jobs[].op_order` | sequential | sequential | sequential | none | dag | dag |

---

## 3. 后处理判类逻辑 (纯代码,不用 LLM)

```python
def classify(instance: dict) -> tuple[str, list[str]]:
    """
    返回 (base_type, extensions)。
    base_type ∈ {JSP, FJSP, FlowShop, OpenShop, RCPSP, MM-RCPSP}
    extensions ⊂ {setup, due-date, release-time, no-wait, preemption, machine-downtime, weighted}
    """
    ops = [op for j in instance["jobs"] for op in j["ops"]]

    flexible       = any(len(op["candidates"]) > 1 for op in ops)
    has_resources  = bool(instance.get("resources"))
    has_precedence = any(op.get("predecessors") for op in ops)
    orders         = {j.get("op_order", "sequential") for j in instance["jobs"]}

    # base type
    if has_resources and has_precedence:
        base = "MM-RCPSP" if flexible else "RCPSP"
    elif orders == {"none"}:
        base = "OpenShop"
    elif flexible:
        base = "FJSP"
    elif _all_same_machine_sequence(instance):
        base = "FlowShop"
    else:
        base = "JSP"

    # extensions
    ext = []
    constraints = instance.get("constraints", {}) or {}
    if constraints.get("setup_times"):     ext.append("setup")
    if constraints.get("no_wait"):         ext.append("no-wait")
    if constraints.get("preemption"):      ext.append("preemption")
    if any(j.get("due")     for j in instance["jobs"]): ext.append("due-date")
    if any(j.get("release") for j in instance["jobs"]): ext.append("release-time")
    if any((m.get("unavailable") or []) for m in instance.get("machines", [])):
        ext.append("machine-downtime")
    if instance.get("objective", {}).get("type", "").startswith("weighted"):
        ext.append("weighted")

    return base, ext
```

判定表:

| 特征组合 | 类型 |
|---------|-----|
| candidates 全 1 + op_order=sequential + 各 job 机器序列不完全相同 | JSP |
| candidates 有 >1 + op_order=sequential | FJSP |
| candidates 全 1 + 所有 job 机器序列相同 | FlowShop |
| op_order=none | OpenShop |
| resources 非空 + predecessors 非空 + candidates 全 1 | RCPSP |
| resources 非空 + predecessors 非空 + candidates 有 >1 | MM-RCPSP |

---

## 4. Builder 路由

```python
BUILDERS = {
    "JSP":       build_jsp,
    "FJSP":      build_fjsp,
    "FlowShop":  build_flowshop,   # 可直接复用 build_jsp
    "OpenShop":  build_openshop,
    "RCPSP":     build_rcpsp,
    "MM-RCPSP":  build_mm_rcpsp,   # 可复用 build_rcpsp + 可选区间
}

EXTENSION_APPLIERS = {
    "setup":            apply_setup_times,
    "no-wait":          apply_no_wait,
    "preemption":       apply_preemption,
    "due-date":         apply_due_date,
    "release-time":     apply_release_time,
    "machine-downtime": apply_downtime,
    "weighted":         apply_weighted_objective,
}

def solve(instance):
    base, extensions = classify(instance)
    model, vars_ = BUILDERS[base](instance)
    for ext in extensions:
        EXTENSION_APPLIERS[ext](model, vars_, instance)
    return solve_cp_sat(model, vars_)
```

**极简替代方案**: 只写一个通用 builder,因为 JSP ⊂ FJSP ⊂ MM-RCPSP。一个通用 builder 就能覆盖所有类型,连路由都省了。首版建议还是**分 builder + extension applier**,便于单元测试和 debug。

---

## 5. 抽取 Prompt 改造

在 `src/prompts/` 新增 `unified_scheduling_extract.txt`,用来替代/补充 `step1_extract.txt`:

关键要点:
1. 明确告诉 LLM "每个 op 都用 `candidates` 数组包裹,即使只有一个候选机器"。
2. 告知 `op_order` 的取值语义(sequential / none / dag),并给出触发词示例(如 "in any order"→none,"prerequisite"→dag)。
3. 告诉模型 **不要自己判断类型**,只要如实填写字段,缺失字段留空或省略。
4. 沿用项目已有的 formatting rules(Title Case operation、"X minutes" duration、PascalCase component_type 等)。

Prompt 骨架(伪代码):

```
You are given a natural language description of a scheduling problem.
Extract all information into a structured JSON following the EXACT schema below.
DO NOT try to classify the problem type. Just fill fields; leave optional fields
empty or omitted if not mentioned.

<粘贴 §2.1 的 schema>

<粘贴 step1_extract.txt 的 Formatting rules>

Structural rules:
- Every op MUST use "candidates" array, even if there is only one machine candidate.
- If op order is described as "in any order" / "no ordering constraint" → op_order = "none".
- If any op references a specific prerequisite op → op_order = "dag" and fill predecessors.
- Otherwise → op_order = "sequential".
- resources[] only when the description mentions shared limited resources (workers,
  tools, energy) beyond machines.
```

---

## 6. Pipeline 改造点

对照 `pipelines.md` 现有 5 条流程,列出需要动的地方:

### 6.1 Preprocess

- `generate_orders.py` 需要扩展:目前只从 JSSP 反向生成 NL 订单。要支持其他类型,需要额外准备 FJSP / RCPSP 的 GT 数据源(或从 JSSP 数据人工/程序性扰动生成)。
- `RouteSheet.create_route_sheet` 保持不变(它处理的是单条 job 的 route sheet,可继续复用)。

### 6.2 DSL Pipeline (`main.py --mode dsl_pipeline`)

- 新增/替换 step1 使用 `unified_scheduling_extract.txt`。
- 新增 step `classify_and_route`,调用 §3 的 `classify()`,把结果和 base_type、extensions 一起持久化到中间产物(便于 debug)。
- step "solve" 改为读入 base_type + extensions,走 §4 的 BUILDERS + APPLIERS。

### 6.3 Evaluate

- 新增按 base_type 分组的准确率指标。
- 新增判类准确率指标 (`classified_type` vs `gt_type`)。
- 保留原有 makespan / accuracy 指标,分类型汇总。

---

## 7. 落地测试步骤(建议在新分支中执行)

按顺序推进,每步都能独立验证再往下:

1. **代码骨架**
   - 新增 `src/schema/unified.py`(pydantic 定义或 dataclass)
   - 新增 `src/classify.py` 实现 §3
   - 新增 `src/builders/` 目录,先只放 `build_jsp.py`(基线),其余先放 stub

2. **回归 JSP baseline**
   - 用现有 ta71–ta80 数据,新 pipeline 走 JSP 分支,和现有 `main.py --mode dsl_pipeline` 结果 diff。
   - 通过后再动其他 builder。

3. **加 FJSP 支持**
   - 手工构造 3–5 个小 FJSP 实例(可以从 ta71 派生:给几个 op 加候选机器)。
   - 实现 `build_fjsp.py`,回归判类和求解正确性。

4. **加 extension applier**
   - 先做 `due-date`、`release-time`、`weighted`(改目标函数最简单)。
   - 每个 applier 用单元测试覆盖:构造最小实例,验证约束真的加进去了。

5. **接入 LLM 抽取**
   - 写 `unified_scheduling_extract.txt`,先在 5–10 条 NL 订单上对比新旧 prompt 抽取结果。
   - 观察 candidates 数组、op_order 字段的填写正确率。

6. **端到端评估**
   - 混合类型数据集(JSP + FJSP + 带 due date 的 JSP)跑一遍完整 pipeline。
   - 输出分类混淆矩阵 + 求解正确率。

---

## 8. 风险与备注

- **LLM 填 `op_order` 可能不稳定**——尤其"sequential"是隐含默认时,模型可能漏填。首版建议在解析端加兜底:字段缺失一律当 sequential。
- **`candidates` 数组即使长度 1 也要求 LLM 显式包裹**——历史 prompt 是扁平结构,模型可能退化。可以在 prompt 中加 few-shot 示例强化。
- **`predecessors` 依赖 `op.id`**——LLM 生成的 op id 要保证唯一。可以在后处理阶段补一次唯一性校验/重命名。
- **不要在首版就追求通用 builder**——分 builder 结构调试起来定位问题快,后期稳定后再考虑合并。
- **Setup times / preemption 这类扩展 CP-SAT 建模复杂度高**——首版可以先跳过,只保证判类给出正确 extension 标签,builder 抛"not implemented"。
