# NL description -> fully structural route sheet (with LLM)
# fully structural route sheet -> JSP solver formatted matrix (with LLM)
# JSP solver formatted matrix -> JSP solver output
# fully structural route sheet + JSP solver output -> production plans

from __future__ import annotations
import random 
import copy
import openai
import os
import time
import json
from tqdm import tqdm
from openai import OpenAI
from collections import defaultdict, Counter
from utils.util import read_json, write_json, read_txt, read_released_prompt, write_txt
from src.experiment.schedule import schedule
from src.experiment.groundtruth import GroundTruth

class Baseline:
    def __init__(self, structured_route_sheet: list, instance_description: str, experiment_type: str):
        self.structured_route_sheet = structured_route_sheet
        self.orders = [] 
        self.machines = []
        self.structural_info = []
        self.or_matrix = []
        self.production_plan = []
        self.generate_synthetic_data_prompt_path = "src/prompts/generate_synthetic_data.txt"
        self.orders2machines_prompt = read_txt("src/prompts/orders2machines.txt")
        self.orders2structural_info_prompt = read_txt("src/prompts/orders2structural_info.txt")
        self.orders2or_matrix_prompt = read_txt("src/prompts/orders2or_matrix.txt")
        self.structural_info2or_matrix_prompt = read_txt("src/prompts/structural_info2or_matrix.txt")
        self.structural_info2production_plan_prompt = read_txt("src/prompts/structural_info2production_plan.txt")
        self.assigned_jobs = {} # OR-Tools solve result
        self.solver = None # OR-Tools solve result
        self.dump_dir_path = ""
        self.experiment_type = experiment_type # CPE_CAE_CSE-2, CSE-1, SGE, DAE
        self.instance_description = instance_description

        self.ground_route_sheet = {}

        self.compile_error_num = 0

        self.system_prompt = "You are an expert in the field of manufacturing"
        self.batch_input_path = "data/temp_batch/batch_input.jsonl"
        self.batch_output_path = "data/temp_batch/batch_output.jsonl"

    def load_data(self):
        if not os.path.exists(self.dump_dir_path):
            os.makedirs(self.dump_dir_path)
        if os.path.exists(self.dump_dir_path + "orders.json"):
            self.orders = read_json(self.dump_dir_path + "orders.json")
        if os.path.exists(self.dump_dir_path + "machines.json"):
            self.machines = read_json(self.dump_dir_path + "machines.json")
        if os.path.exists(self.dump_dir_path + "structural_info.json"):
            self.structural_info = read_json(self.dump_dir_path + "structural_info.json")
        if os.path.exists(self.dump_dir_path + "or_matrix.json"):
            self.or_matrix = read_json(self.dump_dir_path + "or_matrix.json")
        if os.path.exists(self.dump_dir_path + "assigned_jobs.json"):
            self.assigned_jobs = read_json(self.dump_dir_path + "assigned_jobs.json")

    def run(self):
        self.dump_dir_path = "outputs/Baseline-" + self.experiment_type + "/"+ self.instance_description + "/"
        self.groundtruth = GroundTruth(self.instance_description)
        self.load_data()
        if self.experiment_type == "CPE_CAE_CSE-2":
            if len(self.orders) == 0:
                self.structured2orders()
            self.orders2machines()
            self.orders2structural_info()
            self.structural_info2or_matrix()
            self.or_matrix2JSP_result()
            self.JSP_result2production_plan()

        elif self.experiment_type == "CSE-1":
            # Input: ground-truth fully structured route sheet.
            # Output: OR matrix.
            self.structural_info = read_json(self.dump_dir_path + "route_sheets.json")
            self.structural_info2or_matrix()
            self.or_matrix2JSP_result()
            self.JSP_result2production_plan()

        elif self.experiment_type == "SGE":
            # Input: JSP solver result.
            # Output: production plan.
            self.assigned_jobs = read_json("outputs/GroundTruth/" + self.instance_description + "/" + "assigned_jobs.json")
            self.JSP_result2production_plan()

        elif self.experiment_type == "DAE":
            pass

    # with LLM
    def structured2orders(self):
        print("structured2orders ing...")
        prompt_template = read_released_prompt(self.generate_synthetic_data_prompt_path)
        for structured in tqdm(self.structured_route_sheet):
            prompt = prompt_template.replace("---STRUCTURED---", json.dumps(structured))
            result = self.__chatgpt_function(prompt)
            try:
                clean_result = json.loads(result)
            except:
                print("Error json loads")
                clean_result = {}
            self.orders.append(clean_result)
            time.sleep(3)
        write_json(self.dump_dir_path + "orders.json", self.orders)

    # with LLM
    def orders2machines(self):
        print("orders2machines ing...")
        prompt = self.orders2machines_prompt.replace("---ORDERS---", str(self.orders))
        print("prompt: ", prompt[:30])
        result = self.__chatgpt_function(prompt)
        print("result: ", result[:30], "\n")
        try:
            clean_result = json.loads(result)
        except:
            print("Error json loads")
            clean_result = []
        self.machines = clean_result
        write_json(self.dump_dir_path + "machines.json", self.machines)

    # with LLM
    def orders2structural_info(self):
        print("orders2structural_info ing...")
        self.structural_info = []

        self.__empty_jsonl_contents()

        for index, order in enumerate(self.orders):
            order_copy = copy.deepcopy(order)
            order_copy["machines"] = self.machines
            prompt = self.orders2structural_info_prompt.replace("---ORDERS---", json.dumps(order_copy))
            sys_content = "You are an AI assistant specializing in parsing and structuring industrial order data."
            user_content = prompt
            self.__gpt_batch_store(sys_content, user_content, str(index))

        batch_obj = self.__gpt_batch_call()

        batch_id = batch_obj.id
        results = self.__get_batch_result(batch_id)

        for result in results:
            try:
                clean_result = json.loads(result)
            except json.JSONDecodeError:
                print("Error json loads")
                clean_result = {}
            self.structural_info.append(clean_result)

        write_json(self.dump_dir_path + "structural_info.json", self.structural_info)

    # with LLM
    def structural_info2or_matrix(self):
        print("structural_info2or_matrix ing...")
        self.or_matrix = []
        
        # Step 1: Prepare batch inputs
        self.__empty_jsonl_contents()
        for index, structural_info in enumerate(self.structural_info):
            structural_info_copy = copy.deepcopy(structural_info)
            structural_info_copy["machines"] = self.machines
            prompt = self.structural_info2or_matrix_prompt.replace("---STRUCTURAL_INFO---", json.dumps(structural_info_copy))
            self.__gpt_batch_store(self.system_prompt, prompt, str(index))

        # Step 2: Call the batch API
        batch_obj = self.__gpt_batch_call()
        batch_id = batch_obj.id

        # Step 3: Retrieve and process batch results
        results = self.__get_batch_result(batch_id)
        
        if not results:
            print("No results returned from batch API.")
            return

        for result in results:
            try:
                clean_result = json.loads(result)
            except json.JSONDecodeError:
                print("structural_info2or_matrix Error json loads")
                clean_result = []
                self.compile_error_num += 1

            row = []
            for res in clean_result:
                ele = []
                ele.append(res.get("machine", None))
                ele.append(res.get("duration", None))
                ele.append(res.get("pre_indexs", [i for i in range(len(row))]))
                row.append(ele)
            if len(row) != 0:
                self.or_matrix.append(row)

        # Step 4: Save the resulting OR matrix
        write_json(self.dump_dir_path + "or_matrix.json", self.or_matrix)
        write_json(self.dump_dir_path + "compile_error_rate.json", self.compile_error_num/len(results))
    
    def or_matrix2JSP_result(self):
        print("or_matrix2JSP_result ing...")
        or_matrix = copy.deepcopy(self.or_matrix)
        assigned_jobs, solver, err_rate = schedule(or_matrix)
        if len(assigned_jobs) == 0:
            print("No solver")
        else:
            self.assigned_jobs = assigned_jobs
            self.solver = solver
        write_json(self.dump_dir_path + "assigned_jobs.json", self.assigned_jobs)
        write_txt(self.dump_dir_path + "err_rate.txt", str(err_rate))

    def JSP_result2production_plan(self):
        production_plan = []
        for machine_index, production_sequence in self.assigned_jobs.items():
            print("machine_index: ", machine_index)
            if int(machine_index) < len(self.machines):
                machine_plan = {
                    "machine": self.machines[int(machine_index)],
                    "production_sequence": []
                }
            else:
                machine_plan = {
                    "machine": "machine_" + str(machine_index) + "_unknown",
                    "production_sequence": []
                }
            for step in production_sequence:
                print(step)
                step_info = {
                    "start": step[0],
                    "end": step[0] + step[3],
                    "job_id": step[1],
                    "task_id": step[2]
                }
                try:
                    if step[2] < len(self.structural_info[step[1]].get("steps", [])):
                        step_data = self.structural_info[step[1]].get("steps", [])[step[2]]
                        step_info.update(step_data)
                except:
                    continue
                machine_plan["production_sequence"].append(step_info)
            production_plan.append(machine_plan)
        self.production_plan = production_plan
        write_json(self.dump_dir_path + "production_plan.json", self.production_plan)

    def JSP_result2production_plan_with_llm(self):
        print("JSP_result2production_plan ing...")
        structural_info2production_plan_prompt = self.structural_info2production_plan_prompt.replace("---STRUCTURAL_INFO---", json.dumps(self.structural_info)).replace("---ASSIGNED_JOBS---", json.dumps(self.assigned_jobs)).replace("---MACHINES---", json.dumps(self.machines))
        result = self.__chatgpt_function(structural_info2production_plan_prompt)
        try:
            clean_result = json.loads(result)
        except:
            print("Error json loads")
            clean_result = []
        self.production_plan = clean_result
        write_json(self.dump_dir_path + "production_plan.json", self.production_plan)

    def __chatgpt_function(self, content, gpt_model="gpt-4o"):
        while True:
            try:
                client = OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                )
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "user", "content": content}
                    ],
                    model=gpt_model,
                    max_tokens=15000
                )
                return chat_completion.choices[0].message.content
            except openai.APIError as error:
                print("error: ", error)
                continue

    def __gpt_batch_store(self, sys_content, user_content, index):
        standard = {"custom_id": "", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "system", "content": ""},{"role": "user", "content": ""}],"max_tokens": 10000}}
        prompt_unit = standard.copy()
        prompt_unit["body"]["messages"][0]["content"] = sys_content
        prompt_unit["body"]["messages"][1]["content"] = user_content
        prompt_unit["custom_id"] = index
        with open(self.batch_input_path, 'a') as file:
            json_line = json.dumps(prompt_unit)
            file.write(json_line + '\n')

    def __gpt_batch_call(self):
        client = OpenAI()
        while(True):
            try:
                batch_input_file = client.files.create(
                    file=open(self.batch_input_path, "rb"),
                    purpose="batch"
                )
                break
            except openai.APIError as error:
                print(error)
                time.sleep(1)
        batch_input_file_id = batch_input_file.id
        while(True):
            try:
                batch_obj = client.batches.create(
                    input_file_id=batch_input_file_id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                )
                break
            except openai.APIError as error:
                print(error)
                time.sleep(1)
        write_txt("data/temp_batch/batch_id.txt", batch_obj.id)
        return batch_obj
    
    def __get_batch_result(self, batch_id):
        client = OpenAI()
       
        while True:
            try:
                batch = client.batches.retrieve(batch_id)
            except openai.APIError as error:
                print(error)
                time.sleep(1)
                continue
            if batch.output_file_id:
                print("Batch output_file_id: ", batch.output_file_id)
            if batch.status == "completed":
                results_return = []
                result_file_id = batch.output_file_id
                result = client.files.content(result_file_id).content
                result_file_name = self.batch_output_path
                with open(result_file_name, 'wb') as file:
                    file.write(result)
                results = []
                with open(result_file_name, 'r') as file:
                    for line in file:
                        # Parsing the JSON string into a dict and appending to the list of results
                        json_object = json.loads(line.strip())
                        results.append(json_object)
                for r in results:
                    result = r["response"]["body"]["choices"][0]["message"]["content"]
                    results_return.append(result)
                return results_return
            elif batch.status == "failed" :
                print("Batch failed")
                return []
            elif batch.status == "expired":
                print("Batch expired")
                return []
            elif batch.status == "cancelled":
                print("Batch cancelled")
                return []
            elif batch.status == "cancelling":
                print("Batch cancelling")
                return []
            else:
                time.sleep(1)

    def __empty_jsonl_contents(self):
        if os.path.exists(self.batch_input_path):
            with open(self.batch_input_path, 'w') as file:
                file.write('')
        if os.path.exists(self.batch_output_path):
            with open(self.batch_output_path, 'w') as file:
                file.write('')
