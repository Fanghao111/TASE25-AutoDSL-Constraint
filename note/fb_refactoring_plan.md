# FB Pipeline Refactoring Plan

## Context

当前FBPipeline由于使用free-form LLM提取（不固定schema），导致需要额外的SRD（Semantic Role Discovery）阶段来发现字段含义，以及大量field_mapping逻辑。通过在提取阶段强制使用固定schema，可以消除整个SRD阶段及相关代码（~200行），使pipeline逻辑更清晰。

目标：
1. 消除冗余转换，使每阶段职责单一明确
2. 全面归一化（machine/operation/type/参数key/参数value单位）
3. 加入bad case收集机制，支持人工修正后重跑

---

## New Pipeline Architecture (3 Steps × 2, + 3 Deterministic)

```
NL Description
    │
    ▼
[Step 1: 提取信息] ──→ 固定schema JSON（只管提取内容，不管格式）
    │
    ▼
[Step 1-verify] ──→ LLM对照原文校验事实准确性
    │
    ▼
[Step 2: 调整格式] ──→ 格式化JSON（Title Case/PascalCase/lowercase keys/含单位等）
    │
    ▼
[Step 2-verify] ──→ LLM检查格式规范是否到位
    │
    ▼
[Step 3: 语义归一] ──→ 归一化JSON（同义词合并mapping）
    │
    ▼
[Step 3-verify] ──→ LLM检查归一化mapping是否合理
    │
    ▼
[Step 4: Build OR Matrix] ──→ [[machine_idx, duration, [pre_indexes]], ...] (确定性)
    │
    ▼
[Step 5: Solve] ──→ assigned_jobs (OR-Tools, 确定性)
    │
    ▼
[Step 6: Ground] ──→ Production plan (确定性)
```

### 各步骤职责划分（互不重叠）

| 步骤 | 职责 | 不管什么 |
|---|---|---|
| Step 1 提取 | 从NL中提取所有信息，按固定schema结构输出 | 不管格式（大小写、PascalCase等） |
| Step 1-verify | 对照原文检查：信息是否遗漏/捏造 | 不管格式 |
| Step 2 格式 | 按规范调整值的格式 | 不管语义是否重复 |
| Step 2-verify | 检查格式规范是否全部满足 | 不管语义 |
| Step 3 归一 | 同义词/变体合并为统一名称 | 不管格式（已经处理过了） |
| Step 3-verify | 检查mapping是否合理（不该合并的有没有被错误合并） | 不管格式 |

---

## Fixed Schema Definition

每个order的提取结果必须遵循此结构（Step 1只要求结构正确，不要求格式）：
```json
{
  "steps": [
    {
      "operation": "laser cutting",
      "machine": "laser cutter",
      "duration": "88 minutes",
      "precondition": [
        {"component": "sheet metal", "component_type": "sheet metal"}
      ],
      "postcondition": [
        {"component": "cut sheet metal", "component_type": "cut sheet metal"}
      ],
      "parameters": {
        "laser power": "High",
        "cutting speed": "100 mm/s"
      }
    }
  ]
}
```

格式规范（由Step 2负责调整）：
- operation: Title Case
- machine: Title Case，含括号信息保留（如 "Shear (Sheet Metal)"）
- duration: 字符串含单位（如 "88 minutes"）
- precondition/postcondition: component_type用PascalCase
- parameters: key用lowercase with spaces，value为含单位字符串

---

## Bad Case Mechanism

### 判定条件（任一条件即为bad case）

1. **JSON解析失败**：LLM返回内容无法解析为有效JSON
2. **结构不完整**：缺少`steps`，或step中缺少必须字段（operation/machine/duration/precondition/postcondition）
3. **关键字段值无效**：duration中无数字，machine为空字符串
4. **提取结果为空或明显不完整**：steps数量为0

### 标记方式

直接在原有JSON中加入`bad_case`字段，不单独建文件：

```json
{
  "steps": [...],
  "bad_case": "Step 1: JSON解析失败，LLM返回非JSON内容"
}
```

或解析完全失败时：
```json
{
  "steps": [],
  "bad_case": "Step 1: JSON解析失败，原始返回内容无法解析"
}
```

### 处理策略

- Bad case**照常存在结果JSON数组中**（位置/index不变）
- 通过检查是否存在`bad_case`字段来判断是否跳过
- 后续步骤遇到有`bad_case`字段的order直接跳过，不处理
- 人工修正后，删掉`bad_case`字段即可重跑

### 触发时机

- Step 1 (提取) + verify 后：schema结构不对或信息明显错误 → retry一次 → 仍失败则标记
- Step 2 (格式) + verify 后：格式调整失败或破坏了结构 → retry一次 → 仍失败则标记
- Step 3 (归一) + verify 后：归一化mapping不合理或应用后关键字段无效 → retry一次 → 仍失败则标记

---

## 归一化范围（Step 3 语义归一，不含格式）

Step 3 只处理**语义等价**（同义词/变体合并），格式已由Step 2处理完毕：

| 类别 | 归一化内容 | 示例 |
|---|---|---|
| machine名 | 合并指代同一设备的不同叫法 | "CNC Mill" / "CNC Milling Machine" → "CNC Mill" |
| operation名 | 合并指代同一工序的不同叫法 | "Laser Cutting" / "Laser Cut" → "Laser Cutting" |
| component_type | 合并指代同一物料类型的不同叫法 | "SheetMetal" / "MetalSheet" → "SheetMetal" |
| 参数key | 合并指代同一参数的不同叫法 | "cutting speed" / "cut speed" → "cutting speed" |
| 参数value单位 | 合并同义单位表达 | "100 mm/s" / "100 millimeters per second" → "100 mm/s" |

---

## What Gets Removed (冗余消除)

| 被移除代码 | 原因 |
|---|---|
| `discover_roles()` (lines 238-296) | 固定schema后role已知，无需发现 |
| `_validate_semantic_roles()` (lines 360-451) | 无SRD则无需验证SRD |
| `_build_field_mapping_from_roles()` (lines 453-484) | field_mapping整体移除 |
| `_detect_type_field()` (lines 486-501) | component_type字段固定为`"component_type"` |
| `_find_steps()` instance method (line 621-623) | steps固定在`data["steps"]` |
| `_groups_to_mapping()` (lines 332-340) | 不再需要field name分组 |
| `_apply_field_mapping()` (lines 342-348) | 不再需要field key改名 |
| `self.semantic_roles` / `self.field_mapping` / `self._job_steps` 状态 | 完全移除，无需任何替代 |
| `FIELD_MAPPING` 常量 / `SRD_field_mapping.json` 输出 | 通过修改evaluation.py直接消除 |
| `discover_roles_prompt` / `fix_mapping_prompt` 加载 | 不再使用 |

---

## Implementation Detail

### `__init__` 变更

```python
# 移除
self.semantic_roles = {}
self.field_mapping = {}
self._job_steps = []
self.discover_roles_prompt = ...
self.fix_mapping_prompt = ...

# 新增（无需bad_cases列表，直接通过JSON中的bad_case字段判断）
# 跳过逻辑：if "bad_case" in data: continue
```

### Step 1: `extract_all()` — 提取信息

```python
def extract_all(self):
    # 1. 构建prompt（固定schema结构，不要求格式）
    # 2. 并行LLM调用
    # 3. 解析JSON + schema结构验证（字段是否齐全）
    # 4. 失败的retry一次
    # 5. 仍失败的标记为bad case
    # 6. 保存 step1_extracted.json
```

### Step 1-verify: `verify_extract()` — 校验事实准确性

```python
def verify_extract(self):
    # 1. 对每个valid order：构建校验prompt（原文 + 提取结果）
    # 2. LLM对照原文检查：是否遗漏信息/捏造信息
    # 3. 解析LLM返回的修正结果 + schema验证
    # 4. 失败的retry → 仍失败标记bad case
    # 5. 保存 step1_verified.json
```

### Step 2: `format_all()` — 调整格式

```python
def format_all(self):
    # 1. 构建格式化prompt（附格式规范：Title Case/PascalCase/lowercase keys等）
    # 2. 并行LLM调用
    # 3. 解析JSON + 格式校验（确定性检查：是否满足格式规范）
    # 4. 失败的retry
    # 5. 仍失败标记bad case
    # 6. 保存 step2_formatted.json
```

### Step 2-verify: `verify_format()` — 校验格式

```python
def verify_format(self):
    # 1. LLM检查格式规范是否全部满足
    # 2. 如有格式问题，LLM返回修正版
    # 3. 确定性格式校验
    # 4. 失败标记bad case
    # 5. 保存 step2_verified.json
```

### Step 3: `normalize_all()` — 语义归一

```python
def normalize_all(self):
    # 1. 从所有valid数据中收集5类值（machine/operation/component_type/参数key/参数value）
    # 2. LLM调用：分组同义变体 → canonical mapping
    # 3. 应用mapping到所有数据
    # 4. 保存 step3_normalized.json + mapping文件
```

### Step 3-verify: `verify_normalize()` — 校验归一化mapping

```python
def verify_normalize(self):
    # 1. LLM检查mapping是否合理（不该合并的有没有被错误合并）
    # 2. 如有问题，LLM返回修正后的mapping
    # 3. 重新应用修正后的mapping
    # 4. schema验证（归一化后关键字段是否仍有效）
    # 5. 失败标记bad case
    # 6. 保存 step3_verified.json (即最终的normalized结果)
```

### Stage 4: `build_or_matrix()` (原 `derive_constraints`)

直接字段访问，无field_mapping间接层，归一化后值已统一无需lower：
```python
def build_or_matrix(self):
    # 对每个valid order的 data["steps"]:
    machine_name = step["machine"]
    duration = _parse_duration(step["duration"])
    precondition = {item["component_type"] for item in step.get("precondition", [])}
    # 匹配前置步骤的postcondition中的component_type确定pre_indexes
```

### Stage 5: `solve_jsp()` — 不变

### Stage 6: `ground_production_plan()` — 简化

```python
src_step = self.normalized_jsons[job_id]["steps"][task_id]
step_info["operation"] = src_step["operation"]
step_info["machine"] = src_step["machine"]
step_info["duration"] = src_step["duration"]
step_info["precondition"] = [item["component_type"] for item in src_step.get("precondition", [])]
step_info["postcondition"] = [item["component_type"] for item in src_step.get("postcondition", [])]
step_info["parameters"] = src_step.get("parameters", {})
```

---

## 新增文件

| 文件 | 用途 | 约行数 |
|---|---|---|
| `src/experiment/schema_validator.py` | validate_extraction()：确定性schema检查 | ~50行 |
| **Prompts (6个，每步1个主prompt + 1个verify prompt)** | | |
| `src/prompts/step1_extract.txt` | 提取信息（固定schema结构，不管格式） | ~30行 |
| `src/prompts/step1_verify.txt` | 对照原文校验事实准确性 | ~25行 |
| `src/prompts/step2_format.txt` | 按规范调整格式（Title Case/PascalCase等） | ~30行 |
| `src/prompts/step2_verify.txt` | 检查格式规范是否满足 | ~25行 |
| `src/prompts/step3_normalize.txt` | 语义归一（同义词分组） | ~35行 |
| `src/prompts/step3_verify.txt` | 检查归一化mapping是否合理 | ~25行 |

---

## evaluation.py 修改（彻底去掉 field_mapping）

`src/evaluation/evaluation.py` 第 189-201 行当前逻辑：
```python
fb_nj = read_json(fb_nj_path)          # CAM-3_normalized_jsons.json
fb_fm = read_json(fb_fm_path)          # SRD_field_mapping.json
fb_rs = FBPipeline.build_route_sheets(fb_nj, fb_fm, subfolder)
```

改为直接按固定schema转换，不再依赖 field_mapping 文件：
```python
fb_nj = read_json(fb_nj_path)          # CAM-3_normalized_jsons.json (固定schema)
fb_rs = FBPipeline.build_route_sheets(fb_nj, subfolder)
```

对应 `FBPipeline.build_route_sheets()` 静态方法签名改为：
```python
@staticmethod
def build_route_sheets(normalized_jsons, instance_description=""):
    """直接按固定schema转换为route_sheet格式，无需field_mapping参数。
    由于固定schema与ground truth的字段命名完全一致（precondition/postcondition/component_type），
    本质上只是提取steps层级。"""
    route_sheets = []
    for data in normalized_jsons:
        route_sheet_entry = {
            "instance_description": instance_description,
            "route_sheet": data.get("steps", [])
        }
        route_sheets.append(route_sheet_entry)
    return route_sheets
```

**彻底消除的内容：**
- `SRD_field_mapping.json` 文件不再生成
- `build_route_sheets()` 不再接受 `field_mapping` 参数
- `_extract_materials_static()` / `_extract_parameters_static()` 辅助方法不再需要（逻辑内联到上面）
- `_find_steps_static()` 不再需要（固定schema直接 `data["steps"]`）

---

## 向后兼容

1. ~~evaluation.py兼容：写入硬编码SRD_field_mapping.json~~ → **直接修改evaluation.py，去掉field_mapping依赖**
2. **CSE-1/SGE模式**：新增`_route_sheet_to_fixed_schema()`将ground truth格式转为固定schema
3. **旧prompt文件保留**不删除，只是不再引用

---

## CSE-1/SGE模式处理

```python
elif self.experiment_type == "CSE-1":
    route_sheets = read_json(groundtruth_path)
    # 转换为固定schema格式
    self.normalized_jsons = [self._route_sheet_to_fixed_schema(rs) for rs in route_sheets]
    self.valid_indices = set(range(len(self.normalized_jsons)))
    self.build_or_matrix()
    self.solve_jsp()
    self.ground_production_plan()
```

`_route_sheet_to_fixed_schema()` — 由于命名统一，转换极简：
```python
def _route_sheet_to_fixed_schema(self, route_sheet_data):
    """Ground truth的route_sheet格式与固定schema字段命名一致，直接包装即可。"""
    return {"steps": route_sheet_data.get("route_sheet", [])}
```

---

## 辅助方法

### `_parse_duration(duration_str) -> int`

```python
def _parse_duration(self, duration_str: str) -> int:
    """解析duration字符串为整数分钟"""
    text = str(duration_str).lower().strip()
    match = re.search(r'([\d.]+)', text)
    if not match:
        return 0
    value = float(match.group(1))
    if 'hour' in text or 'hr' in text:
        return int(value * 60)
    if 'second' in text or 'sec' in text:
        return max(1, int(value / 60))
    return int(value)  # default: minutes
```

---

## 关键文件清单

| 文件 | 操作 |
|---|---|
| `src/experiment/fb_pipeline.py` | 重构主文件 |
| `src/evaluation/evaluation.py` | 修改FB评估部分，去掉field_mapping依赖 |
| `src/experiment/schema_validator.py` | 新建 |
| `src/prompts/step1_extract.txt` | 新建 |
| `src/prompts/step1_verify.txt` | 新建 |
| `src/prompts/step2_format.txt` | 新建 |
| `src/prompts/step2_verify.txt` | 新建 |
| `src/prompts/step3_normalize.txt` | 新建 |
| `src/prompts/step3_verify.txt` | 新建 |

---

## Verification Plan

1. `python main.py --mode fb_pipeline --type CPE_CAE_CSE-2` 单instance(ta71)运行
2. 检查输出文件完整性：step1_extracted, step1_verified, step2_formatted, step2_verified, step3_normalized, CGM_or_matrix, assigned_jobs, SGM_production_plan
3. 检查bad case标记：结果JSON中有`bad_case`字段的order被后续步骤跳过
4. 确认**不再生成** `SRD_field_mapping.json`（已彻底去掉）
5. `python main.py --mode evaluation --evaluation_type CSE-2` 确认评估正常
6. 对比makespan与之前的运行结果，确保质量不降
