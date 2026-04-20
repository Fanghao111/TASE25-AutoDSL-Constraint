from utils.util import read_json, write_json, read_txt, write_txt
from tqdm import tqdm
from openai import OpenAI
import openai
import time
import os
import json
import re


class RouteSheet:
    def __init__(self, machines_data_path, jssp_data_path, arrange_path, jssp_mapped_path, route_sheet_store_path, route_sheet_reduce_path):
        self.machines = read_json(machines_data_path)
        self.jssp_data = read_json(jssp_data_path)
        self.arrange = read_json(arrange_path)
        self.jssp_mapped_path = jssp_mapped_path
        self.route_sheet_reduce_path = route_sheet_reduce_path
        self.route_sheet_store_path = route_sheet_store_path
        self.batch_input_path = "data/temp_batch/batch_input.jsonl"
        self.batch_output_path = "data/temp_batch/batch_output.jsonl"
        self.prompt = read_txt("src/prompts/route_sheet_prompt.txt")
        self.sys_prompt = "You are an expert in the field of manufacturing"
        self.batch_size = 1500

    def mapping(self):
        for i in range(len(self.arrange)):
            mapping = self.arrange[i]["mapping"]
            machines_num = self.jssp_data[i]["machines_num"]
            data = self.jssp_data[i]["data"]
            for job in data:
                steps = job["steps"]
                for step in steps:
                    machine = step["machine"]
                    step["machine"] = mapping[machine] if machine in mapping else machine
            self.jssp_data[i]["data"] = data
        write_json(self.jssp_mapped_path, self.jssp_data)

    def create_route_sheet(self):
        jssp_data = read_json(self.jssp_mapped_path)
        jssp_jobs_number = [len(j["data"]) for j in jssp_data]
        jobs_data = []
        for single_jssp in jssp_data:
            for job in single_jssp["data"]:
                jobs_data.append(self.__create_real_job(job["steps"]))
        
        for k in tqdm(range(0, len(jobs_data), self.batch_size)):
            print("Current k: ", k)
            batch_route_sheet = []
            jobs_data_batch = jobs_data[k:k+self.batch_size]
            self.__empty_jsonl_contents()
            for i in range(0, len(jobs_data_batch)):
                job = jobs_data_batch[i]
                prompt = self.prompt.replace("---STEPS---", str(job))
                # print("prompt: ", prompt)
                self.__gpt_batch_store(self.sys_prompt, prompt, str(i))
                
            print("Batch stored")
            batch_obj = self.__gpt_batch_call()
            print("Batch called, waiting for results...")
            results = self.__get_batch_result(batch_obj.id)
            print("Results received")
            print("jobs_data_batch len: ", len(jobs_data_batch))
            print("results len: ", len(results))
            # print("results: ", results)
            for result in results:
                try:
                    clean_result = json.loads(result)
                except:
                    print("Error json loads")
                    clean_result = result
                batch_route_sheet.append(clean_result)
            # break
            # 增量更新
            route_sheet = batch_route_sheet
            if os.path.exists(self.route_sheet_store_path):
                old_route_sheet = read_json(self.route_sheet_store_path)
                route_sheet = old_route_sheet + batch_route_sheet
            write_json(self.route_sheet_store_path, route_sheet)

    def route_sheet_reduce(self):
        route_sheet = read_json(self.route_sheet_store_path)
        result = []
        jssp_len = []
        for jssp in self.jssp_data:
            jssp_len.append(len(jssp["data"])) # 该 domain 下的jobs num
        for i in range(len(jssp_len)):
            result.append([])
            for j in range(jssp_len[i]):
                if len(route_sheet) == 0:
                    break
                ele = route_sheet.pop(0)
                
                if isinstance(ele, dict):
                    ele["instance_description"] = self.jssp_data[i]["description"]
                    result[i].append(ele)
                else:
                    result[i].append({
                        "instance_description": self.jssp_data[i]["description"]
                    })
        # 去除 result 末尾多余的空数组
        while len(result) > 0 and len(result[-1]) == 0:
            result.pop()
        write_json(self.route_sheet_reduce_path, result)
        print("Route sheet reduce finished")

    def __create_real_job(self, steps):
        new_steps = []
        for step in steps:
            machine = self.machines[int(step["machine"])]["machine"]
            patterns = self.machines[int(step["machine"])]["patterns"]
            duration = int(step["time"])
            new_steps.append({
                "machine": machine,
                "duration": duration,
                "patterns": patterns
            })
        return new_steps

    def __gpt_batch_store(self, sys_content, user_content, index):
        standard = {"custom_id": "", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "system", "content": ""},{"role": "user", "content": ""}],"max_tokens": 10000}}
        prompt_unit = standard.copy()
        prompt_unit["body"]["messages"][0]["content"] = sys_content
        prompt_unit["body"]["messages"][1]["content"] = user_content
        prompt_unit["custom_id"] = index
        with open(self.batch_input_path, 'a') as file:
            # 将字典转换为JSON字符串并追加到文件
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
                time.sleep(3)
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
                time.sleep(3)
        write_txt("data/temp_batch/batch_id.txt", batch_obj.id)
        return batch_obj
    
    def __get_batch_result(self, batch_id):
        client = OpenAI()
        results_return = []
        while True:
            try:
                batch = client.batches.retrieve(batch_id)
            except openai.APIError as error:
                print(error)
                time.sleep(3)
                continue
            if batch.status == "completed":
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
                # print("results: ", results)
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
                # print("Batch status: ", batch.status)
                time.sleep(3)

    def __empty_jsonl_contents(self):
        if os.path.exists(self.batch_input_path):
            with open(self.batch_input_path, 'w') as file:
                file.write('')
        if os.path.exists(self.batch_output_path):
            with open(self.batch_output_path, 'w') as file:
                file.write('')
