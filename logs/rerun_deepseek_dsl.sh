#!/usr/bin/env bash
cd "D:/FB/TASE25-AutoDSL-Constraint"
export LLM_BASE_URL='http://localhost:8000/v1'
export OPENAI_API_KEY='sk-mi300-local'
export LLM_MAX_WORKERS="${LLM_MAX_WORKERS:-32}"
export LLM_MODEL='deepseek-v3'
export DSL_OUTPUT_PREFIX='DSL-models/deepseek-v3'
./.venv/Scripts/python.exe -u main.py --mode dsl_pipeline --type full \
  --instances ta71 ta72 ta73 ta74 ta75 ta76 ta77 ta78 ta79 ta80 \
  > logs/dsl_pipeline_deepseek-v3.log 2>&1
rc=$?
echo "[$(date '+%H:%M:%S')] deepseek-v3 DSL rerun rc=$rc" >> logs/orchestration.log
touch outputs/.phase1_dsl_deepseek-v3.done

# eval
export FB_EVAL_LABEL='FB-models/deepseek-v3'
export DSL_EVAL_LABEL='DSL-models/deepseek-v3'
for t in production_plan route_sheet graph; do
  ./.venv/Scripts/python.exe -u main.py --mode evaluation --evaluation_type "$t" > "logs/eval_deepseek-v3_${t}.log" 2>&1
done
echo "[$(date '+%H:%M:%S')] deepseek-v3 eval rerun done" >> logs/orchestration.log
touch outputs/.phase2_deepseek-v3.done
