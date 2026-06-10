# Unified Pipeline 全流程总结

完整实验类型 `CPE_CAE_CSE-2` 包含以下步骤（其他类型跳过前面的步骤）：

---

## Step 1: Free Extraction（自由信息抽取）
**方法: LLM 并行调用** (`free_extract_all`)

- **输入**: 每条自然语言订单 (NL order)
- **处理**: 对每个 order 并行调用 LLM，不指定固定 schema，让 LLM 自由提取
- **Prompt 核心逻辑** (`free_extract.txt`):
  - 从制造流程描述中提取操作名称、机器设备、耗时（纯数字）、输入/输出材料及类型、工艺参数
  - 明确要求 "Do NOT follow a fixed schema"，字段名由 LLM 自行决定
  - 输出纯 JSON
- **输出**: `raw_jsons.json` — 每个 order 对应一个自由格式的 JSON

---

## Step 1.5: Verify Extraction（验证抽取结果）
**方法: LLM 并行调用** (`verify_extraction_all`)

- **输入**: 原始 NL order + Step 1 产出的 raw JSON
- **处理**: 对每对 (order, raw_json) 并行调用 LLM 做交叉校验
- **Prompt 核心逻辑** (`verify_extraction.txt`):
  1. 修正与原文不符的值
  2. 补充原文中有但 JSON 中缺失的信息
  3. 删除 JSON 中凭空捏造的信息
  4. 保持结构和字段名不变，只修值
- **输出**: `verified_jsons.json`

---

## Step 2: Normalize + Discover Semantic Roles（归一化 + 语义角色发现）
**方法: 确定性代码 + LLM 单次调用 + 确定性验证** (`normalize_and_discover`)

这是最复杂的一步，分为 6 个子步骤：

### 2a: 统计字段频率（确定性代码）
- 遍历所有 verified JSON，用 `_find_steps()` 找到 steps 列表
- 统计每个字段名出现次数，区分 frequent fields (≥5% 出现率) 和 rare fields
- 收集所有字符串值（组件/材料名称）

### 2b: 采样代表性 JSON（确定性代码）
- 按 JSON 大小排序，选取中位大小的 1 个样本，避免超出 proxy 的 ~6K 字符限制

### 2c: LLM 单次调用 — 归一化 + 角色发现
- **Prompt 核心逻辑** (`normalize_and_discover.txt`)，三个任务：
  - **Task 1 — Field Name Normalization**: 将同义字段名分组（如 `machine/device/equipment` → `machine`），只输出有 2+ 成员的组
  - **Task 2 — Component Name Normalization**: 将同义组件名分组（如 `SheetMetal/sheet_metal` → `sheet metal`），只输出有 2+ 成员的组
  - **Task 3 — Semantic Role Discovery**: 发现哪些字段承担调度语义角色：
    - `resource`: 共享资源（机器）
    - `duration`: 耗时（数值型）
    - `dependency_in/out`: 输入/输出依赖（实现前驱约束）
    - `dependency_type_field`: 依赖匹配的子字段
    - `operation`: 操作名
    - `extra_params`: 额外参数

### 2d: 确定性替换
- `_groups_to_mapping()`: 将 LLM 返回的分组转为 `{old_name: canonical_name}` 映射
- `_extend_component_mapping()`: 纯代码处理 — CamelCase/snake_case → lowercase spaces
- `_apply_field_mapping()`: 递归替换所有 JSON 的字段名
- `_apply_value_mapping()`: 递归替换所有 JSON 的字符串值

### 2e: 确定性规则验证 (`_validate_semantic_roles`)
- duration 字段的值是否为数值型（>30% 非数值则报错）
- resource 字段的值是否跨 job 共享（>90% 唯一则报错 — 机器应该被多个 step 共用）
- dependency in/out 的类型是否有交集（否则前驱约束无法建立）
- 每个角色字段的覆盖率是否 ≥30%

### 2f: 修复循环（条件触发 LLM 调用）
- 如果验证发现问题，用 `fix_mapping.txt` prompt 发送问题列表 + 原始映射 + 样本给 LLM，请求修正
- 用修正后的映射重新执行 2d 步骤

**最终输出**: `normalized_jsons.json`, `field_mapping.json`, `semantic_roles.json`

---

## Build Route Sheets（构建工艺路线表）
**方法: 纯确定性代码** (`_build_route_sheets`)

- 利用 `field_mapping`（从 semantic roles 导出）将 normalized JSON 转换为统一格式的 route sheet：
  ```
  { machine, duration, operation, precondition, postcondition, parameters }
  ```
- `precondition/postcondition` 格式: `[{component, component_type, container}]`
- `parameters`: 所有非核心字段自动归入此项
- **输出**: `route_sheets.json`

---

## Step 3: Derive Constraints（推导约束）
**方法: 纯确定性代码** (`derive_constraints`)

- 收集所有 machine 名称，排序建立索引
- 对每个 route sheet 的每个 step：
  - 查找 machine 在索引中的位置 → `machine_idx`
  - 提取 duration（纯数字）
  - **前驱约束推导**: 当前 step 的 `precondition.component_type` 与前面所有 step 的 `postcondition.component_type` 做交集匹配 — 有交集即建立前驱依赖
- **输出**: `or_matrix.json` — `[machine_idx, duration, pre_indexes]` 三元组矩阵, `machines.json`

---

## Step 4a: Solve JSP（求解作业车间调度）
**方法: OR-Tools CP-SAT 求解器** (`solve_jsp` → `schedule()`)

- 输入 `or_matrix`
- 建模:
  - 每个 task 创建 `start/end/interval` 变量
  - **机器约束**: 同一机器上的任务不重叠 (`add_no_overlap`)
  - **前驱约束**: `pre_index` 的 `end ≤ 当前 task 的 start`
  - **目标**: 最小化 makespan（最后一个 task 完成时间的最大值）
- **输出**: `assigned_jobs.json` — 按机器分组的调度结果 `(start_time, job_id, task_id, duration)`

---

## Step 4b: Ground Production Plan（生成生产计划）
**方法: 纯确定性代码** (`ground_production_plan`)

- 将 solver 输出的抽象 `(job_id, task_id)` 映射回 route sheet 的具体信息
- 为每条记录补充: operation, machine 名称, duration, precondition/postcondition 类型, parameters
- **输出**: `production_plan.json` — 最终可读的生产计划

---

## 流程总览

```
NL Orders
  │
  ▼ [LLM × N 并行]
Step 1: Free Extract → raw_jsons
  │
  ▼ [LLM × N 并行]
Step 1.5: Verify → verified_jsons
  │
  ▼ [代码统计 + LLM × 1 + 代码替换 + 代码验证 + (LLM修复)]
Step 2: Normalize & Discover → normalized_jsons + semantic_roles + field_mapping
  │
  ▼ [纯代码]
Build Route Sheets → route_sheets
  │
  ▼ [纯代码, component_type 交集匹配]
Step 3: Derive Constraints → or_matrix + machines
  │
  ▼ [OR-Tools CP-SAT]
Step 4a: Solve JSP → assigned_jobs
  │
  ▼ [纯代码映射]
Step 4b: Ground → production_plan
```

**LLM 调用总计**: Step 1 和 1.5 各 N 次并行（N = 订单数），Step 2 固定 1 次 + 可选 1 次修复 = 总共 **2N + 1~2 次**。其余步骤全部是确定性代码或求解器。
