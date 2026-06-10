# FB Pipeline 架构分析：跳出三模块框架

## 一、DSL 论文核心贡献拆解

论文的三大贡献对应 6 个独立的贡献单元：

| ID | 贡献 | 论文章节 | 技术实体 |
|----|------|---------|---------|
| **C1** | 三模块约束架构 | Sec. II-A | CAM → CGM → SGM 三阶段流水线 |
| **C2** | DSL 双程序视图 | Sec. II-B | Operation-centric view `Lo = {So, Λo}` + Product-flow-centric view `Lp = {Sp, Λp}` |
| **C3** | PDA 形式化约束推导 | Sec. II-C | 基于编译理论的下推自动机遍历产品流，确定可达性和生命周期 |
| **C4** | 自动化 DSL 设计 | Sec. III | DPMM 非参数聚类（操作语义）+ EM 算法（产品流语法） |
| **C5** | 问题形式化 | Sec. II-C | 将约束规范定义为 `R = {(Oi, Mj) | exec}`, `P = {(Oi, Oj) | dep}` |
| **C6** | 评估体系 | Sec. IV | 5 组实验 + EMKVP/Constraint-Acc/Compiler-ER/Runtime-ER + 统计检验 |

---

## 二、FB 对 DSL 六个贡献的关系

### 完全依赖（直接复用）

| 贡献 | FB 中的体现 | 依赖程度 |
|------|-----------|---------|
| **C1 三模块架构** | FB-CAM → FB-CGM → FB-SGM，结构完全对齐 | 完全复用 |
| **C5 问题形式化** | 求解同一个 R, P 约束规范问题 | 完全复用（但 R, P 本身是教科书标准定义） |
| **C6 评估体系** | 同样的 5 组实验、同样的指标 | 完全复用 |

### 完全替换（FB 的革新点）

| 贡献 | DSL 做法 | FB 做法 |
|------|---------|--------|
| **C2 双程序视图** | 预定义结构空间 Lo, Lp，LLM 在 DSL 模板内翻译 | 零 schema 自由提取 + LLM 自我验证 + 语义角色发现 |
| **C3 PDA 约束推导** | PDA 遍历 + define/kill 分析 + 可达性证明 | component_type 集合交集匹配（15 行代码） |
| **C4 DPMM + EM** | 需要预处理阶段，有收敛性证明 | 完全消除，语义角色在运行时一次性发现 |

### FB 独有的新机制

| 机制 | 代码位置 | 本质 |
|------|---------|------|
| 零 schema 自由提取 | `free_extract_all()` | LLM 不受结构约束，自主决定字段名 |
| LLM 自我验证 | `verify_extraction_all()` | 用原始 NL 交叉校验提取结果 |
| 语义角色发现 | `normalize_and_discover()` | LLM 从自由格式数据中识别调度语义 |
| 确定性规则验证 | `_validate_semantic_roles()` | 代码检查 LLM 发现的语义角色是否合理 |
| 验证-修复反馈环 | `fix_mapping.txt` | 验证不过 → LLM 重新修正 → 重新应用 |

---

## 三、FB 当前数据流的真实面貌

```
NL orders
  │
  ▼ [LLM × N]
① free_extract_all() → raw_jsons         ← LLM 创造性工作
  │
  ▼ [LLM × N]
② verify_extraction_all() → verified_jsons ← LLM 校验性工作
  │
  ▼ [代码统计 + LLM × 1 + 代码验证 + LLM修复]
③ normalize_and_discover()                 ← 混合：LLM理解 + 代码验证
  │  → normalized_jsons
  │  → semantic_roles
  │  → field_mapping
  │
  ▼ [纯代码格式转换]
④ _build_route_sheets() → route_sheets     ← 纯机械转换
  │
  ▼ [纯代码集合运算]
⑤ derive_constraints() → or_matrix         ← 纯机械转换
  │
  ▼ [OR-Tools]
⑥ solve_jsp() → assigned_jobs             ← 求解器
  │
  ▼ [纯代码索引回查]
⑦ ground_production_plan() → production_plan ← 纯机械转换
```

---

## 四、三个关键观察

### 观察 1：④ 是多余的中间层

`_build_route_sheets()` 把 normalized_json + field_mapping 转换成 `{machine, duration, operation, precondition, postcondition, parameters}` 格式。

但 ⑤ `derive_constraints()` 完全可以直接从 normalized_jsons + semantic_roles 工作，不需要先转成 route_sheet 格式。route_sheet 格式的**唯一作用**是与 DSL 论文的评估框架对齐。

```python
# 现在 ⑤ 的逻辑：
machine_name = step.get("machine", "")
current_input_types = {mat.get("component_type", "")}

# 完全可以直接：
machine_name = step.get(semantic_roles["resource"]["field"], "")
input_field = semantic_roles["dependency_in"]["field"]
current_input_types = {mat.get(type_field, "")}
```

**④ 存在的原因是继承了 DSL 的 "route_sheet 中间表示"**，但 FB 不需要它。

### 观察 2：⑦ 在 FB 中是平凡的

DSL-SGM 之所以重要，是因为要从双程序视图（operation_programs + production_programs）回查细节，涉及 Precond/Postcond/Execution 的跨视图映射。

FB-SGM 只是：`route_sheets[job_id]["route_sheet"][task_id]` — 一个数组索引。这不值得作为独立模块存在。

### 观察 3：③ 横跨了 CAM 和 CGM 的边界

`normalize_and_discover()` 实际做了两件不同性质的事：
- **归一化**（field/component name alignment）— 属于数据清洗，是 CAM 的事
- **语义角色发现**（哪个字段是 machine、duration、dependency）— 属于理解约束语义，本质上是 CGM 的事

在 DSL 论文中，这两件事不需要分开，因为 DSL 模板同时定义了名称和语义。但在 FB 中，它们被人为合并在了一个函数里。

---

## 五、FB 的自然架构：两层 + action-feedback 统一模式

从信息的本质流向看，FB 的自然结构不是三模块，而是：

```
┌─────────────────────────────────────────────────────────┐
│                    FB 的自然结构                          │
│                                                         │
│  Stage 1: Extract + Verify                              │
│  ┌─────────────┐    ┌──────────────┐                    │
│  │ LLM 自由提取 │───→│ LLM 交叉验证  │──→ verified JSON  │
│  └─────────────┘    └──────┬───────┘                    │
│                      feedback: JSON ↔ NL 比对            │
│                                                         │
│  Stage 2: Discover + Validate                           │
│  ┌─────────────┐    ┌──────────────┐                    │
│  │ LLM 角色发现 │───→│ 规则验证      │──→ semantic roles  │
│  └─────────────┘    └──────┬───────┘                    │
│                      feedback: 确定性规则检查             │
│                                                         │
│  Stage 3: Compile + Solve                               │
│  ┌─────────────────────────────────┐                    │
│  │ 确定性约束推导 → 求解 → 落地     │──→ production plan │
│  └─────────────────────────────────┘                    │
│   (全部纯代码/求解器，无 LLM)                             │
└─────────────────────────────────────────────────────────┘
```

### 与三模块框架的对比

| | DSL 三模块 | FB 自然结构 |
|--|-----------|-----------|
| 组织原则 | 按**功能角色**分 (抽象/生成/落地) | 按**action-feedback 对**分 |
| 模块数 | 3 (CAM, CGM, SGM) | 2+1 (两个带反馈的理解阶段 + 一个确定性编译阶段) |
| 边界标准 | "中间表示"的形态变化 | "是否需要 LLM" + "如何验证" |
| SGM 的地位 | 独立模块 | 并入编译阶段（因为是平凡索引） |
| route_sheet | 核心中间表示 | 不需要（直接操作 normalized_json） |

---

## 六、更激进的重构可能性

### 可能性 A：统一的 action-feedback 架构（推荐）

把 Stage 2 也做成 LLM 反馈式：

```
Stage 2: Discover + LLM-Verify + Rule-Verify
  1. LLM 发现语义角色
  2. LLM 自我验证: "你确定 'equipment' 是 resource 而不是其他含义吗？"
  3. 确定性规则兜底验证
```

三个 Stage 统一模式：**LLM action → (LLM self-check) → deterministic validation → feedback**

发表潜力：
1. 架构创新清晰：不是 "三个功能模块"，而是 "N 层反馈精炼"，每层有统一的 `act → verify → feedback` 模式
2. 与 DSL 论文形成对比叙事：DSL 靠"编译时的结构约束"保证质量，FB 靠"运行时的多层验证反馈"保证质量 —— 两种不同的可靠性保障范式
3. 可以自然扩展：层数可增可减，每层的验证策略可以不同（LLM 自检 / 规则检查 / 求解器检查）
4. 消除 route_sheet 中间表示的依赖

### 可能性 B：LLM 直接推导约束

跳过语义角色发现，LLM 直接识别约束关系：

```
normalized_jsons → [LLM 直接识别约束关系] → constraints
  "这两个 step 共享同一台 machine → resource constraint"
  "这个 step 的 output 类型匹配那个 step 的 input → precedence constraint"
```

完全绕过 route_sheet 和 field_mapping。代价是丧失可解释性和可验证性。

### 可能性 C：渐进式精炼

不分模块，整个流程看作同一个数据对象的逐步精炼：

```
Representation v0: NL text
     ↓ enrich (LLM extract)
Representation v1: free-form JSON
     ↓ refine (LLM verify)
Representation v2: verified JSON
     ↓ enrich (LLM discover + code normalize)
Representation v3: JSON with semantic annotations
     ↓ compile (code)
Representation v4: constraint matrix
     ↓ solve (OR-Tools)
Representation v5: scheduled plan
```

每一步都是 enrich / refine / compile，没有模块边界，只有精炼层级。

---

## 七、结论与建议

**可能性 A（统一的 action-feedback 架构）** 最有发表潜力，原因：

1. 与 DSL 论文形成清晰的范式对比：**结构约束 vs. 反馈验证** 两种不同的 LLM 可靠性保障范式
2. 架构上真正独立，不再是三模块的变体
3. 代码层面需要：去掉 `_build_route_sheets()` 和 `ground_production_plan()` 对 route_sheet 格式的依赖，让 `derive_constraints()` 直接消费 normalized_jsons + semantic_roles
