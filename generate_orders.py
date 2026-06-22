"""Generate NL order descriptions from structured route sheets for all 10 instances."""

import json
import os
import time
from openai import OpenAI
from tqdm import tqdm
from utils.util import read_json, write_json, read_txt

INSTANCES = [
    "instance ta71", "instance ta72", "instance ta73", "instance ta74", "instance ta75",
    "instance ta76", "instance ta77", "instance ta78", "instance ta79", "instance ta80",
]

OUTPUT_DIRS = [
    "outputs/DSLPipeline-CPE_CAE_CSE-2",
    "outputs/FB-CPE_CAE_CSE-2",
]


def chatgpt_function(content, gpt_model="gpt-4o"):
    for attempt in range(5):
        try:
            client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", "sk-placeholder"),
                base_url="http://localhost:4142/v1"
            )
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": content}],
                model=gpt_model,
                max_tokens=8192,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"error (attempt {attempt+1}/5): ", e)
            if attempt < 4:
                time.sleep(2 ** attempt)
    raise RuntimeError("LLM call failed after 5 attempts")


def main():
    prompt_template = read_txt("src/prompts/generate_synthetic_data.txt")
    route_sheet_all = read_json("data/route_sheet_reduce.json")

    for domain_data in route_sheet_all:
        instance_description = domain_data[0]["instance_description"]
        if instance_description not in INSTANCES:
            continue

        # Check if orders already exist in both dirs
        first_out = os.path.join(OUTPUT_DIRS[0], instance_description, "orders.json")
        if os.path.exists(first_out):
            print(f"[SKIP] {instance_description}: orders.json already exists")
            continue

        print(f"[START] {instance_description}: generating orders for {len(domain_data)} route sheets")
        orders = []
        for structured in tqdm(domain_data, desc=instance_description):
            prompt = prompt_template.replace("---STRUCTURED---", json.dumps(structured))
            result = chatgpt_function(prompt)
            try:
                clean_result = json.loads(result)
            except json.JSONDecodeError:
                print(f"  Warning: JSON parse failed, storing raw")
                clean_result = {}
            orders.append(clean_result)

        # Save to both output directories
        for out_dir in OUTPUT_DIRS:
            out_path = os.path.join(out_dir, instance_description, "orders.json")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            write_json(out_path, orders)
            print(f"  Saved: {out_path}")

        print(f"[DONE] {instance_description}: {len(orders)} orders generated")


if __name__ == "__main__":
    main()
