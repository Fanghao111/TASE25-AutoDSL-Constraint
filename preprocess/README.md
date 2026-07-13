# `preprocess/` 目录说明

> 从 raw JSSP 数据到 pipeline 入口的中间产物集合。全部由 `main.py --mode preprocess` 或 `generate_orders.py` 生成。

## 文件与生成关系

| 路径 | 类型 | 生成方 | LLM? | 说明 |
|---|---|---|---|---|
| `jssp_mapped.json` | canonical | `RouteSheet.mapping()` (`main.py --mode preprocess` 第一步) | ❌ | `data/jssp_data.json` 经 `data/arrange.json` 的 machine id rewrite 后透传。deterministic，所有模型共享。<br>**`SKIP_ARRANGE_MAPPING=1`** 可跳过 arrange 改写，保留 Taillard 原始 20 台机器编号。 |
| `route_sheet.json` | canonical | `RouteSheet.create_route_sheet()` | ✅ | 扁平 list，每 job 一条。LLM 把 `(machine, duration)` 步骤 render 成含 `precondition / postcondition / parameters / duration` 的结构。 |
| `route_sheet_reduce.json` | canonical | `RouteSheet.route_sheet_reduce()` | ❌ | 上一步的 domain 分组版本（按 `data/jssp_data.json` 的 domain 划分）。下游 AutoDSL / GT / generate_orders 都读这个。 |
| `orders/ta{71..80}/orders.json` | canonical | `generate_orders.py` | ✅ | 从 `route_sheet_reduce.json` 出发，per-step LLM 调用生成 NL 订单描述。FB / DSL pipeline 的 NL 输入。 |
| `<model>/route_sheet.json`<br>`<model>/route_sheet_reduce.json`<br>`<model>/orders/…` | per-model | `scripts/run_preprocess_4llm.ps1`<br>`scripts/run_generate_orders_4llm.ps1` | ✅ | 每个模型（deepseek-v3 / qwen3-4b / qwen3-30b / qwen3-235b）独立跑一份产物，供人手工挑选质量最好的一份作为 canonical。 |
| `orders.old.20260701/` | 归档 | — | — | 2026-07-01 之前的旧 orders 备份，不再使用。 |

## Canonical 的来源

`preprocess/` 顶层的 `route_sheet.json` / `route_sheet_reduce.json` / `orders/` **不由 pipeline 自动产生**，而是人工挑选后从某个 `preprocess/<model>/` 复制上来的：

- 当前 canonical = **qwen3-235b**（三个文件字节级完全等于 `preprocess/qwen3-235b/…`，2026-07-07 19:26 复制）
- 选择依据：跑完 4 模型 preprocess 后人工检视 `route_sheet.json` 的字段完整性、命名一致性、duration 单位规范性；qwen3-235b 是 4 个候选里最稳的
- 下游所有 pipeline（AutoDSL / GroundTruth / DSL / FB）都只消费这个 canonical，不读 per-model 目录

## 两步 pipeline 的依赖关系

```
data/jssp_data.json  ─┐
data/arrange.json    ─┼─► RouteSheet.mapping()      ─► jssp_mapped.json     (deterministic)
data/machines.json   ─┤
                      └─► RouteSheet.create_route_sheet()  (per model, LLM)
                                                    ─► <model>/route_sheet.json
                                                    ─► RouteSheet.route_sheet_reduce()
                                                    ─► <model>/route_sheet_reduce.json
                                                          │
                          [人工挑最好的一份] ─────────────┘
                          复制 → preprocess/route_sheet.json + route_sheet_reduce.json
                                                          │
                                                          ▼
                          generate_orders.py (per model, LLM)
                                                    ─► <model>/orders/ta{71..80}/orders.json
                                                          │
                          [同一模型的 orders 也复制上来]  │
                          复制 → preprocess/orders/ta{71..80}/orders.json
```

## 环境变量

| 名字 | 默认 | 作用 |
|---|---|---|
| `PREPROCESS_OUTPUT_DIR` | `preprocess` | `route_sheet.json` / `route_sheet_reduce.json` 的写入根目录。多模型并行时每个 worker 设成 `preprocess/<model>` 避免冲突。 |
| `PREPROCESS_SKIP_MAPPING` | `0` | 设为 `1`：跳过 `RouteSheet.mapping()`（复用现有 `jssp_mapped.json`）。多模型并行时避免并发写 `jssp_mapped.json`。 |
| `SKIP_ARRANGE_MAPPING` | `0` | 设为 `1`：`mapping()` 依然写 `jssp_mapped.json`，但**不应用 `arrange.json` 的 machine id 改写**。用于避开 arrange3.py 图匹配副作用（ta75/ta78 的机器合并），让下游看到干净的 Taillard 20-machine 数据。 |
| `ORDERS_OUTPUT_DIR` | `preprocess/orders` | `generate_orders.py` 的输出目录。多模型并行时设成 `preprocess/<model>/orders`。 |
| `LLM_MODEL` | — | 每次调 LLM 的模型名（`deepseek-v3` / `qwen3-4b` / `qwen3-30b` / `qwen3-235b`）。 |

## 常用命令

**单模型重跑（推荐）**：
```bash
LLM_MODEL=qwen3-235b \
./.venv/Scripts/python.exe -u main.py --mode preprocess [--instances ta75 ta78]

LLM_MODEL=qwen3-235b \
./.venv/Scripts/python.exe -u generate_orders.py [--retry-missing]
```
产物直接写在 `preprocess/`（canonical 位置），不进 per-model 目录。

**4 模型并行 preprocess + orders**（用于选 canonical）：
```powershell
.\scripts\run_preprocess_4llm.ps1
.\scripts\run_generate_orders_4llm.ps1
# 挑最好一份，手动复制到 preprocess/ 顶层
```

**跳过 arrange 的机器合并**：
```bash
SKIP_ARRANGE_MAPPING=1 LLM_MODEL=qwen3-235b \
./.venv/Scripts/python.exe -u main.py --mode preprocess --instances ta75 ta78
```
