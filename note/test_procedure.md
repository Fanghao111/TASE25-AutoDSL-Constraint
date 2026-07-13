# DSL vs FB 完整测试流程（ta71-ta80）

> 在 Windows 上跑 driver，DSL → mi300-1 sglang DeepSeek-V3，FB → mi300-2 sglang DeepSeek-V3，两条 pipeline 并行不互相竞争 LLM 资源。

## 0. 前置条件（一次性）

- venv 已建好：`D:\FB\TASE25-AutoDSL-Constraint\.venv\`
  - torch 必须是 **2.6+ CPU**（transformers 要求 ≥2.6，最新版 Windows DLL 坏）
  - `pip install -r requirements.txt`，再 `pip install --force-reinstall "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu`
  - `python -m spacy download en_core_web_sm`
  - `python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt'); nltk.download('punkt_tab')"`
- 两台机器上 sglang DeepSeek-V3 容器在跑（参见 `D:\FB\mi300\machines.md` 和 `start_sglang_server_v3.sh`）。

## 1. 起两条 SSH 隧道（Windows shell，前台或 `&` 后台都行）

```bash
ssh -i "C:\Users\fanghaozhou\OneDrive - Microsoft\Desktop\deepseek\id_rsa" \
    -N -L 5001:127.0.0.1:5000 -p 50020 azureuser@64.236.30.11 &
ssh -i "C:\Users\fanghaozhou\OneDrive - Microsoft\Desktop\deepseek\id_rsa" \
    -N -L 5002:127.0.0.1:5000 -p 50021 azureuser@64.236.30.11 &
```

检查：
```bash
curl http://127.0.0.1:5001/v1/models   # 应回 deepseek-v3
curl http://127.0.0.1:5002/v1/models
```

## 2. 环境变量（每次开新 shell 都要 export）

```bash
cd D:/FB/TASE25-AutoDSL-Constraint
source .venv/Scripts/activate   # 或 .venv\Scripts\activate.bat (cmd) / activate.ps1 (PowerShell)

export LLM_BASE_URL='http://127.0.0.1:5001/v1'   # 默认指 mi300-1，跑 prereqs 和 dsl 用
export LLM_MODEL='deepseek-v3'
```

跑 fb_pipeline 的那个 shell 单独 `export LLM_BASE_URL='http://127.0.0.1:5002/v1'`。

## 3. Prereqs（4 个，可并行，全部打 mi300-1）

输出位置:
- groundtruth → `outputs/GroundTruth/instance ta7X/`
- autodsl_operation → `outputs/AutoDSL/total_operation_dsl.json` 等（单文件含所有 instance）
- autodsl_production → `outputs/AutoDSL/total_production_dsl.json` 等
- generate_orders → **canonical**: `outputs/Orders/instance ta7X/orders.json`，同时复制到两条 pipeline 的 dump 目录

```bash
# 各 1 个 shell 并发跑，全都 LLM_BASE_URL=5001：
python main.py --mode groundtruth          > logs/gt.log 2>&1 &           # 几秒到 1 分钟
python main.py --mode autodsl_operation    > logs/autodsl_op.log 2>&1 &   # SciBERT + DPMM ~5-10 分钟
python main.py --mode autodsl_production   > logs/autodsl_prod.log 2>&1 & # 纯数学 EM ~1-2 分钟
python generate_orders.py                  > logs/orders.log 2>&1 &       # LLM ~10×7 = ~70 分钟
wait
```

> **注意**：`generate_orders.py` 用的 prompt 已硬性要求所有 duration 用 minutes（修在 `src/prompts/generate_synthetic_data.txt`，2026-06-26）。LLM 单位漂移 bug 修复后请勿回滚。

## 4. 两条 Pipeline 并行（DSL→5001, FB→5002）

```bash
# Shell A
export LLM_BASE_URL='http://127.0.0.1:5001/v1' LLM_MODEL='deepseek-v3'
python main.py --mode dsl_pipeline --type CPE_CAE_CSE-2 > logs/dsl.log 2>&1

# Shell B (同时)
export LLM_BASE_URL='http://127.0.0.1:5002/v1' LLM_MODEL='deepseek-v3'
python main.py --mode fb_pipeline --type CPE_CAE_CSE-2 > logs/fb.log 2>&1
```

单 instance ~15-20 分钟，10 个 instance 并行约 2.5-3 小时。

输出:
- DSL → `outputs/DSLPipeline-CPE_CAE_CSE-2/instance ta7X/{makespan.txt, or_matrix.json, ...}`
- FB → `outputs/FB-2s-CPE_CAE_CSE-2/instance ta7X/{makespan.txt, CGM_or_matrix.json, ...}`

## 5. Evaluation

```bash
python main.py --mode evaluation --evaluation_type CSE-2   # makespan ratio, accuracy
python main.py --mode evaluation --evaluation_type CPE     # BLEU / ROUGE on constraint extraction
python main.py --mode evaluation --evaluation_type CAE     # BLEU / ROUGE on constraint aggregation
```

每个 evaluation 写 `outputs/Evaluation/{type}_{method}_{metric}.json`。

## 6. 关键代码改动（已合入）

| 文件 | 改动 |
|---|---|
| `utils/util.py` | 加 `make_chat_client / make_embed_client / LLM_BASE_URL / LLM_MODEL / EMBED_BASE_URL / EMBED_MODEL`，全部读环境变量 |
| `src/experiment/dsl_pipeline.py` | `__chatgpt_function` 和 embedding client 走新 helper；spacy 模型用 `SPACY_MODEL` 环境变量 |
| `src/experiment/fb_pipeline.py` | `_chatgpt_function` 走新 helper |
| `src/dsl_design/operation.py` | `__chatgpt_function` 走新 helper |
| `generate_orders.py` | canonical 写到 `outputs/Orders/`，并复制到两个 pipeline 的 dump 目录 |
| `src/prompts/generate_synthetic_data.txt` | prompt 加 "All durations MUST be expressed in minutes only" 硬约束 |

## 7. 常见坑

- **torch DLL 加载失败**：用 2.12.x 或最新版可能在 Windows 上崩。固定到 `torch==2.6.0+cpu`。
- **transformers 报 CVE-2025-32434**：要求 torch ≥2.6，所以不能降到 2.5。
- **nltk LookupError 'wordnet'**：跑前 `nltk.download('wordnet','omw-1.4','punkt','punkt_tab')`。
- **spacy `en_core_web_trf` 太重**：默认换成 `en_core_web_sm`（`os.environ.get("SPACY_MODEL","en_core_web_sm")` 覆盖）。
- **10.0.0.x 在 Windows 上不通**：必须 SSH 隧道，不能直连。
- **SSH 隧道端口被占用**：上一轮的 ssh 子进程可能没被父 bash 一起杀掉。`netstat -an | grep :5001` / `5002` 看一下，必要时 `taskkill` 旧 ssh.exe。

## 8. 单 instance smoke test（验证接线）

想先只跑 ta71 验证流程能跑通：在 `main.py` 的 `legal_instance_description_list` 和 `generate_orders.py` 的 `INSTANCES` 里只保留 `"instance ta71"`，其余注释掉。完整流程 ~20-30 分钟。
