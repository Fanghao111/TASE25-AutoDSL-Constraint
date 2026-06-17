# 1. Raw -> Structured
#   Input: Natural-Language-based/ human description of workflow (pure natural language, including machine name, operation description with execution configurations, duration, precondition, and postcondition)
#   Method: DSL translation (with the help of LLMs)
#   Output: Manufacturing route sheet (highly-structured, represented in the corresponding DSL)

# 2.1 Structured -> Constrained
#   Input: Manufacturing route sheet (highly-structured, represented in the corresponding DSL)
#   Method: DSL program verification (explicitly transform the programs representing the route sheet to the resource constraints and precedence constraints)
#   Output: Constraint representation (represented in programs of the corresponding DSL)

# 2.2 Constrained -> JSP Formatted
#   Input: Constraint representation (represented in programs of the corresponding DSL)
#   Method: Symbolic mapping
#   Output: JSP solver constraint format (matrices, the input format of OR-Tools)

# 3. JSP Formatted -> JSP Scheduled
#   Input: JSP solver constraint format (matrices, the input format of OR-Tools)
#   Method: JSP scheduling solver (Google OR-Tools Scheduling Solver)
#   Output: JSP solved schedule (matrices, the output format of OR-Tools)

# 4. JSP Scheduled -> Grounded
#   Input: JSP solved schedule (matrices, the output format of OR-Tools)
#   Method: Symbolic mapping
#   Output: Grounded production plan (represented in programs of the corresponding DSL with detailed execution configurations)

from __future__ import annotations
import copy
import openai
import os
import time
import json
import spacy
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
import numpy as np
from utils.util import read_json, write_json, read_txt, write_txt
from src.experiment.schedule import schedule
from nltk.stem import WordNetLemmatizer
from src.experiment.groundtruth import GroundTruth

class DSLPipeline:
    def __init__(self, raw_sheet: list, production_dsl: dict, operation_dsl: dict, EM_structure: list, instance_description: str, experiment_type="CPE_CAE"):
        self.groundtruth = None
        self.production_dsl = production_dsl
        self.operation_dsl = operation_dsl
        self.EM_structure = EM_structure
        self.orders = [] 
        self.operation_programs = []
        self.production_programs = []
        self.route_sheets = []
        self.or_matrix = []
        self.machines = []
        self.operation_translation_prompt = read_txt("src/prompts/operation_translation.txt")
        self.production_translation_prompt = read_txt("src/prompts/production_translation.txt")
        self.operation_extraction_prompt = read_txt("src/prompts/operation_extraction.txt")
        self.component_extraction_prompt = read_txt("src/prompts/component_extraction.txt")

        self.operation_production_extraction_prompt = read_txt("src/prompts/operation_production_extraction.txt")
        self.program_translation_prompt = read_txt("src/prompts/program_translation.txt")
        self.program_translation_prompt_2 = read_txt("src/prompts/program_translation_2.txt")

        self.assigned_jobs = [] # OR-Tools solve result
        self.solver = None # OR-Tools solve result
        self.dump_dir_path = ""
        self.experiment_type = experiment_type # CPE_CAE_CSE-2, CSE-1, SGE, DAE
        self.instance_description = instance_description

        self.lemmatizer = WordNetLemmatizer()
        self.nlp = spacy.load("en_core_web_trf")

        self.embedding_dic = {}
        self.sys_content = "You are an expert in the field of manufacturing"
        self.batch_input_path = "data/temp_batch/batch_input.jsonl"
        self.batch_output_path = "data/temp_batch/batch_output.jsonl"

        self.compile_error_num = 0

    def run(self):
        self.dump_dir_path = "outputs/DSLPipeline-" + self.experiment_type + "/"+self.instance_description+"/"
        self.groundtruth = GroundTruth(self.instance_description)
        self.load_data()
        if self.experiment_type == "CPE_CAE_CSE-2":
            if len(self.orders) == 0:
                raise RuntimeError(
                    f"Missing intermediate synthetic orders at {self.dump_dir_path}orders.json. "
                    "Please provide the intermediate orders before running dsl_pipeline."
                )
            self.orders2dsl_program_no_batch_parallel()
            self.dsl_program2or_matrix()
            self.dsl_program2route_sheets()
            self.or_matrix2JSP_result()
            self.JSP_result2production_plan()

        elif self.experiment_type == "CSE-1":
            # Input: ground-truth fully structured route sheet.
            # Output: OR matrix.
            self.route_sheets2dsl_program_no_batch()
            self.dsl_program2or_matrix()
            self.or_matrix2JSP_result()
            self.JSP_result2production_plan()

        elif self.experiment_type == "SGE":
            # Input: JSP solver result.
            # Output: production plan.
            self.assigned_jobs = read_json("outputs/GroundTruth/" + self.instance_description + "/" + "assigned_jobs.json")
            self.JSP_result2production_plan()

        elif self.experiment_type == "DAE":
            pass

    def load_data(self):
        if not os.path.exists(self.dump_dir_path):
            os.makedirs(self.dump_dir_path)
        if os.path.exists(self.dump_dir_path + "orders.json"):
            self.orders = read_json(self.dump_dir_path + "orders.json")
        if os.path.exists(self.dump_dir_path + "operation_programs.json"):
            self.operation_programs = read_json(self.dump_dir_path + "operation_programs.json")
        if os.path.exists(self.dump_dir_path + "production_programs.json"):
            self.production_programs = read_json(self.dump_dir_path + "production_programs.json")
        if os.path.exists(self.dump_dir_path + "route_sheets.json"):
            self.route_sheets = read_json(self.dump_dir_path + "route_sheets.json")
        if os.path.exists(self.dump_dir_path + "or_matrix.json"):
            self.or_matrix = read_json(self.dump_dir_path + "or_matrix.json")
        if os.path.exists(self.dump_dir_path + "assigned_jobs.json"):
            self.assigned_jobs = read_json(self.dump_dir_path + "assigned_jobs.json")
        if os.path.exists("data/embedding_dic.json"):
            self.embedding_dic = read_json("data/embedding_dic.json")

    def orders2dsl_program_old(self):
        print("orders2dsl_program ing...")
        self.operation_programs = []
        self.production_programs = []

        # Prepare prompts for operation extraction batch processing
        operation_batch_index = 0
        production_batch_index = 0
        

        # Create batch prompts for operation extraction
        operation_extraction_prompts = []
        for order in self.orders:
            order_copy = copy.deepcopy(order)
            for sentence in order_copy.get("steps", []):
                operation_prompt = self.operation_extraction_prompt.replace("---SENTENCES---", sentence)
                operation_extraction_prompts.append(operation_prompt)
        self.__empty_jsonl_contents()
        # Add operation extraction prompts to batch
        for idx, prompt in enumerate(operation_extraction_prompts):
            self.__gpt_batch_store(sys_content=self.sys_content, user_content=prompt, index=str(idx))

        # Execute batch processing for operation extraction
        print("Operation extraction batch stored")
        operation_batch_obj = self.__gpt_batch_call()
        print("Batch called, waiting for results...")
        operation_batch_results = self.__get_batch_result(operation_batch_obj.id)
        print("Results received")

        # Process operation extraction results
        operation_results = []
        for result in operation_batch_results:
            try:
                operation_response = result.strip().split(",")
                for i in range(len(operation_response)):
                    operation_response[i] = operation_response[i].strip()
            except:
                print("Error parsing operation result JSON")
                operation_response = ["NONE"]
            operation_results.append(operation_response)

        # Group extracted operations by order
        grouped_operations = []
        idx = 0
        for order in self.orders:
            operations = []
            for _ in order.get("steps", []):
                operations.extend(operation_results[idx])
                idx += 1
            grouped_operations.append(operations)

        # Prepare prompts for operation translation
        self.__empty_jsonl_contents()
        for order, operations in zip(self.orders, grouped_operations):
            oper_repr = {}
            for operation in operations:
                opcodes = self.__similarity_opcode(operation)
                if "NONE" not in opcodes:
                    for opcode in opcodes:
                        oper_repr[opcode] = self.operation_dsl[opcode]

            operation_translation_prompt = self.operation_translation_prompt.replace("---OPERATION_DSL---", json.dumps(oper_repr))\
                .replace("---ORDER---", json.dumps(order))

            self.__gpt_batch_store(sys_content=self.sys_content, user_content=operation_translation_prompt, index=str(operation_batch_index))
            operation_batch_index += 1

        # Execute operation translation batch processing
        print("Operation translation batch stored")
        operation_translation_batch_obj = self.__gpt_batch_call()
        print("Batch called, waiting for results...")
        operation_translation_batch_results = self.__get_batch_result(operation_translation_batch_obj.id)
        print("Results received")

        # Parse operation translation results
        result_index = 0
        for order in self.orders:
            try:
                operation_result = json.loads(operation_translation_batch_results[result_index])
            except:
                print("Error json loads in operation result")
                operation_result = []
            self.operation_programs.append(operation_result)
            result_index += 1

        # Prepare prompts for component extraction batch processing
        component_extraction_prompts = []
        for order in self.orders:
            order_copy = copy.deepcopy(order)
            for sentence in order_copy.get("steps", []):
                component_prompt = self.component_extraction_prompt.replace("---SENTENCES---", sentence)
                component_extraction_prompts.append(component_prompt)

        # Add component extraction prompts to batch
        self.__empty_jsonl_contents()
        for idx, prompt in enumerate(component_extraction_prompts):
            self.__gpt_batch_store(sys_content=self.sys_content, user_content=prompt, index=str(idx))

        # Execute batch processing for component extraction
        print("Component extraction batch stored")
        component_batch_obj = self.__gpt_batch_call()
        print("Batch called, waiting for results...")
        component_batch_results = self.__get_batch_result(component_batch_obj.id)
        print("Results received")

        # Process component extraction results
        component_results = []
        for result in component_batch_results:
            try:
                component_response = result.strip().split(",")
                for i in range(len(component_response)):
                    component_response[i] = component_response[i].strip()
            except:
                print("Error parsing component result JSON")
                component_response = ["NONE"]
            component_results.append(component_response)

        # Group extracted components by order
        grouped_components = []
        idx = 0
        for order in self.orders:
            components = []
            for _ in order.get("steps", []):
                components.extend(component_results[idx])
                idx += 1
            grouped_components.append(components)

        # Prepare prompts for production translation
        self.__empty_jsonl_contents()
        for order, components in zip(self.orders, grouped_components):
            prod_repr = {}
            for component in components:
                comp_matches = self.__similarity_component(component)
                if "NONE" not in comp_matches:
                    for match in comp_matches:
                        prod_repr[match] = self.production_dsl[match]

            operation_list = [ele.get("Operation", "") for ele in self.operation_programs[production_batch_index]]
            production_translation_prompt = self.production_translation_prompt.replace("---PRODUCTION_DSL---", json.dumps(prod_repr))\
                .replace("---EM_STRUCTURE---", json.dumps(self.EM_structure))\
                .replace("---ORDER---", json.dumps(order))\
                .replace("---OPERATION_LIST---", json.dumps(operation_list))

            self.__gpt_batch_store(sys_content=self.sys_content, user_content=production_translation_prompt, index=str(production_batch_index))
            production_batch_index += 1

        # Execute production translation batch processing
        print("Production translation batch stored")
        production_batch_obj = self.__gpt_batch_call()
        print("Batch called, waiting for results...")
        production_batch_results = self.__get_batch_result(production_batch_obj.id)
        print("Results received")

        # Parse production translation results
        result_index = 0
        for order in self.orders:
            try:
                production_result = json.loads(production_batch_results[result_index])
            except:
                print("Error json loads in production result")
                production_result = []
            self.production_programs.append(production_result)
            result_index += 1

        # Write results to files
        write_json(self.dump_dir_path + "operation_programs.json", self.operation_programs)
        write_json(self.dump_dir_path + "production_programs.json", self.production_programs)

    def orders2dsl_program(self):
        print("orders2dsl_program ing...")
        self.operation_programs = []
        self.production_programs = []

        operation_list = [operation for operation, _ in self.operation_dsl.items()]
        production_list = [production for production, _ in self.production_dsl.items()]


        # Prepare prompts for batch processing
        extraction_prompts = []
        translation_prompts = []

        for order in self.orders:
            order_copy = copy.deepcopy(order)

            # Generate extraction prompts for all steps in the order
            for sentence in order_copy.get("steps", []):
                extraction_prompt = self.operation_production_extraction_prompt\
                    .replace("---OPERATION_LIST---", json.dumps(operation_list))\
                    .replace("---PRODUCTION_LIST---", json.dumps(production_list))\
                    .replace("---SENTENCES---", sentence)
                extraction_prompts.append(extraction_prompt)

        # Add extraction prompts to batch
        self.__empty_jsonl_contents()
        for idx, prompt in enumerate(extraction_prompts):
            self.__gpt_batch_store(sys_content=self.sys_content, user_content=prompt, index=str(idx))

        # Execute batch processing for extraction
        print("Extraction batch stored")
        extraction_batch_obj = self.__gpt_batch_call()
        print("Batch called, waiting for results...")
        extraction_batch_results = self.__get_batch_result(extraction_batch_obj.id)
        print("Results received")

        # Parse extraction results and prepare translation prompts
        result_idx = 0
        for order in self.orders:
            oper_repr, prod_repr = {}, {}

            for sentence in order.get("steps", []):
                try:
                    clean_result = json.loads(extraction_batch_results[result_idx])
                except:
                    print("Error json loads")
                    clean_result = {}

                result_idx += 1

                if clean_result.get("operation", "") in operation_list:
                    oper_repr[clean_result["operation"]] = self.operation_dsl[clean_result["operation"]]
                if clean_result.get("component", "") in production_list:
                    prod_repr[clean_result["component"]] = self.production_dsl[clean_result["component"]]

            translation_prompt = self.program_translation_prompt\
                .replace("---OPERATION_DSL---", json.dumps(oper_repr))\
                .replace("---PRODUCTION_DSL---", json.dumps(prod_repr))\
                .replace("---EM_STRUCTURE---", json.dumps(self.EM_structure))\
                .replace("---ORDER---", json.dumps(order))

            translation_prompts.append(translation_prompt)

        # Add translation prompts to batch
        self.__empty_jsonl_contents()
        for idx, prompt in enumerate(translation_prompts):
            self.__gpt_batch_store(sys_content=self.sys_content, user_content=prompt, index=str(idx))

        # Execute batch processing for translation
        print("Translation batch stored")
        translation_batch_obj = self.__gpt_batch_call()
        print("Batch called, waiting for results...")
        translation_batch_results = self.__get_batch_result(translation_batch_obj.id)
        print("Results received")

        # Parse translation results
        clean_result = {}
        for result in translation_batch_results:
            try:
                clean_result = json.loads(result)
            except:
                print("Error json loads")
                clean_result = {}

            self.operation_programs.append(clean_result.get("operation-view programs", []))
            self.production_programs.append(clean_result.get("production-view programs", []))

        # Write results to files
        write_json(self.dump_dir_path + "operation_programs.json", self.operation_programs)
        write_json(self.dump_dir_path + "production_programs.json", self.production_programs)

    def orders2dsl_program_no_batch(self):
        print("orders2dsl_program ing...")
        self.operation_programs = []
        self.production_programs = []

        operation_list = [operation for operation, _ in self.operation_dsl.items()]
        production_list = [production for production, _ in self.production_dsl.items()]

        for order in tqdm(self.orders):
            order_copy = copy.deepcopy(order)
            oper_repr, prod_repr = {}, {}

            # Generate extraction prompts for all steps in the order
            for sentence in tqdm(order_copy.get("steps", [])):
                extraction_prompt = self.operation_production_extraction_prompt\
                    .replace("---OPERATION_LIST---", json.dumps(operation_list))\
                    .replace("---PRODUCTION_LIST---", json.dumps(production_list))\
                    .replace("---SENTENCES---", sentence)

                # Execute extraction prompt using single round conversation
                extraction_result = self.__chatgpt_function(extraction_prompt)
                try:
                    clean_result = json.loads(extraction_result)
                except Exception as e:
                    print("extraction_result: ", extraction_result)
                    print("Error json loads")
                    clean_result = {}

                if clean_result.get("operation", "") in operation_list:
                    oper_repr[clean_result["operation"]] = self.operation_dsl[clean_result["operation"]]
                if clean_result.get("component", "") in production_list:
                    prod_repr[clean_result["component"]] = self.production_dsl[clean_result["component"]]

            # Generate translation prompt for the order
            translation_prompt = self.program_translation_prompt\
                .replace("---OPERATION_DSL---", json.dumps(oper_repr))\
                .replace("---PRODUCTION_DSL---", json.dumps(prod_repr))\
                .replace("---EM_STRUCTURE---", json.dumps(self.EM_structure))\
                .replace("---ORDER---", json.dumps(order))

            # Execute translation prompt using single round conversation
            translation_result = self.__chatgpt_function(translation_prompt)
            try:
                clean_result = json.loads(translation_result)
            except:
                print("Error json loads")
                clean_result = {}

            self.operation_programs.append(clean_result.get("operation-view programs", []))
            self.production_programs.append(clean_result.get("production-view programs", []))

        # Write results to files
        write_json(self.dump_dir_path + "operation_programs.json", self.operation_programs)
        write_json(self.dump_dir_path + "production_programs.json", self.production_programs)

    def orders2dsl_program_no_batch_parallel(self):
        print("orders2dsl_program ing...")
        self.operation_programs = []
        self.production_programs = []

        operation_list = list(self.operation_dsl.keys())
        production_list = list(self.production_dsl.keys())

        # Process a single order.
        def process_single_order(order):
            order_copy = copy.deepcopy(order)
            oper_repr, prod_repr = {}, {}

            # Process all steps in parallel.
            sentences = order_copy.get("steps", [])
            extraction_prompts = [
                self.operation_production_extraction_prompt
                    .replace("---OPERATION_LIST---", json.dumps(operation_list))
                    .replace("---PRODUCTION_LIST---", json.dumps(production_list))
                    .replace("---SENTENCES---", sentence)
                for sentence in sentences
            ]

            # Execute extraction requests in parallel.
            with ThreadPoolExecutor() as inner_executor:
                extraction_results = list(inner_executor.map(self.__chatgpt_function, extraction_prompts))

            # Merge extraction results.
            for result in extraction_results:
                try:
                    clean_result = json.loads(result)
                except:
                    clean_result = {}
                if clean_result.get("operation", "") in operation_list:
                    oper_repr[clean_result["operation"]] = self.operation_dsl[clean_result["operation"]]
                if clean_result.get("component", "") in production_list:
                    prod_repr[clean_result["component"]] = self.production_dsl[clean_result["component"]]

            # Generate and execute the translation request.
            translation_prompt = self.program_translation_prompt\
                .replace("---OPERATION_DSL---", json.dumps(oper_repr))\
                .replace("---PRODUCTION_DSL---", json.dumps(prod_repr))\
                .replace("---EM_STRUCTURE---", json.dumps(self.EM_structure))\
                .replace("---ORDER---", json.dumps(order))
            translation_result = self.__chatgpt_function(translation_prompt)

            try:
                clean_result = json.loads(translation_result)
            except:
                clean_result = {}
            
            return (
                clean_result.get("operation-view programs", []),
                clean_result.get("production-view programs", [])
            )

        # Process all orders in parallel.
        with ThreadPoolExecutor() as outer_executor:
            # Keep the output order aligned with the input order.
            results = list(tqdm(
                outer_executor.map(process_single_order, self.orders),
                total=len(self.orders),
                desc="Processing Orders"
            ))

        # Collect results.
        for op_prog, prod_prog in results:
            self.operation_programs.append(op_prog)
            self.production_programs.append(prod_prog)

        # Write results to disk.
        write_json(self.dump_dir_path + "operation_programs.json", self.operation_programs)
        write_json(self.dump_dir_path + "production_programs.json", self.production_programs)   

    def route_sheets2dsl_program(self):
        print("route_sheets2dsl_program ing...")
        self.operation_programs = []
        self.production_programs = []

        operation_list = [operation for operation, _ in self.operation_dsl.items()]
        production_list = [production for production, _ in self.production_dsl.items()]

        total_operation_list = []
        total_production_list = []

        translation_prompts = []

        for route_sheet in self.route_sheets:
            current_operation_list = []
            current_production_list = []
            print("route_sheet: ", route_sheet)
            route_sheet_copy = copy.deepcopy(route_sheet["route_sheet"])
            for step in route_sheet_copy:
                current_operation_list.append(step["operation"])
                for component in step["precondition"]:
                    current_production_list.append(component["component_type"])
                for component in step["postcondition"]:
                    current_production_list.append(component["component_type"])
            # Deduplicate items.
            current_operation_list = list(set(current_operation_list))
            current_production_list = list(set(current_production_list))
            total_operation_list.append(current_operation_list)
            total_production_list.append(current_production_list)
                    
        # Parse extraction results and prepare translation prompts
        result_idx = 0
        for route_sheet in self.route_sheets:
            oper_repr, prod_repr = {}, {}

            current_operation_list = total_operation_list[result_idx]
            current_production_list = total_production_list[result_idx]

            for operation in current_operation_list:
                if operation in operation_list:
                    oper_repr[operation] = self.operation_dsl[operation]
            for production in current_production_list:
                if production in production_list:
                    prod_repr[production] = self.production_dsl[production]

            translation_prompt = self.program_translation_prompt_2\
                .replace("---OPERATION_DSL---", json.dumps(oper_repr))\
                .replace("---PRODUCTION_DSL---", json.dumps(prod_repr))\
                .replace("---EM_STRUCTURE---", json.dumps(self.EM_structure))\
                .replace("---ROUTE_SHEET---", json.dumps(route_sheet["route_sheet"]))

            translation_prompts.append(translation_prompt)

        # Add translation prompts to batch
        self.__empty_jsonl_contents()
        for idx, prompt in enumerate(translation_prompts):
            self.__gpt_batch_store(sys_content=self.sys_content, user_content=prompt, index=str(idx))

        # Execute batch processing for translation
        print("Translation batch stored")
        translation_batch_obj = self.__gpt_batch_call()
        print("Batch called, waiting for results...")
        translation_batch_results = self.__get_batch_result(translation_batch_obj.id)
        print("Results received")

        # Parse translation results
        clean_result = {}
        for result in translation_batch_results:
            try:
                clean_result = json.loads(result)
            except:
                print("Error json loads")
                clean_result = {}

            self.operation_programs.append(clean_result.get("operation-view programs", []))
            self.production_programs.append(clean_result.get("production-view programs", []))

        # Write results to files
        write_json(self.dump_dir_path + "operation_programs.json", self.operation_programs)
        write_json(self.dump_dir_path + "production_programs.json", self.production_programs)

    def route_sheets2dsl_program_no_batch(self):
        print("route_sheets2dsl_program ing...")
        self.operation_programs = []
        self.production_programs = []

        operation_list = [operation for operation, _ in self.operation_dsl.items()]
        production_list = [production for production, _ in self.production_dsl.items()]

        total_operation_list = []
        total_production_list = []

        translation_prompts = []

        for route_sheet in self.route_sheets:
            current_operation_list = []
            current_production_list = []
            print("route_sheet: ", route_sheet)
            route_sheet_copy = copy.deepcopy(route_sheet["route_sheet"])
            for step in route_sheet_copy:
                current_operation_list.append(step["operation"])
                for component in step["precondition"]:
                    current_production_list.append(component["component_type"])
                for component in step["postcondition"]:
                    current_production_list.append(component["component_type"])
            # Deduplicate items.
            current_operation_list = list(set(current_operation_list))
            current_production_list = list(set(current_production_list))
            total_operation_list.append(current_operation_list)
            total_production_list.append(current_production_list)
                    
        # Parse extraction results and prepare translation prompts
        result_idx = 0
        for route_sheet in self.route_sheets:
            oper_repr, prod_repr = {}, {}

            current_operation_list = total_operation_list[result_idx]
            current_production_list = total_production_list[result_idx]

            for operation in current_operation_list:
                if operation in operation_list:
                    oper_repr[operation] = self.operation_dsl[operation]
            for production in current_production_list:
                if production in production_list:
                    prod_repr[production] = self.production_dsl[production]

            translation_prompt = self.program_translation_prompt_2\
                .replace("---OPERATION_DSL---", json.dumps(oper_repr))\
                .replace("---PRODUCTION_DSL---", json.dumps(prod_repr))\
                .replace("---EM_STRUCTURE---", json.dumps(self.EM_structure))\
                .replace("---ROUTE_SHEET---", json.dumps(route_sheet["route_sheet"]))

            translation_prompts.append(translation_prompt)

        # Execute translation prompts one by one
        print("Starting translation...")
        for idx, prompt in enumerate(translation_prompts):
            print(f"Processing prompt {idx + 1}/{len(translation_prompts)}")
            response = self.__chatgpt_function(prompt)
            try:
                clean_result = json.loads(response)
            except:
                print("Error json loads")
                clean_result = {}

            self.operation_programs.append(clean_result.get("operation-view programs", []))
            self.production_programs.append(clean_result.get("production-view programs", []))

        # Write results to files
        write_json(self.dump_dir_path + "operation_programs.json", self.operation_programs)
        write_json(self.dump_dir_path + "production_programs.json", self.production_programs)

    def dsl_program2or_matrix(self):
        print("dsl_program2or_matrix ing...")
        self.or_matrix = []
        self.compile_error_num = 0
        self.total_num = 0
        operation_programs = copy.deepcopy(self.operation_programs)
        production_programs = copy.deepcopy(self.production_programs)
        operation2machine_dict = {}
        for i in range(len(operation_programs)):
            operation_program = operation_programs[i]
            for step in operation_program:
                self.total_num += 1
                operation = step["Operation"]
                try:
                    machine = step["Execution"]["machine"]
                except:
                    machine = None
                    self.compile_error_num += 1
                    continue
                if operation not in operation2machine_dict:
                    operation2machine_dict[operation] = [machine]
                else:
                    operation2machine_dict[operation].append(machine)
        # Normalize machine names to lowercase.
        self.machines = list(set([machine.lower() for machine_list in operation2machine_dict.values() for machine in machine_list]))
        for i in range(len(operation_programs)):
            # Build the i-th job row.
            row = []
            operation_program = operation_programs[i]
            production_program = production_programs[i]
            for j in range(len(operation_programs[i])):
                ele = []  # machine, duration, pre_indexes
                try:
                    ele.append(operation_program[j]["Execution"]["machine"])
                except:
                    ele.append(0)
                try:
                    ele.append(int(operation_program[j]["Execution"]["duration"]))
                except:
                    ele.append(0)
                pre_indexes = []
                for production_unit in production_program:
                    pred = production_unit.get("Pred", "")
                    succ = production_unit.get("Succ", "")
                    if operation_program[j]["Operation"] == succ:
                        for ele_2 in row:
                            if ele_2[0] in operation2machine_dict.get(pred, []):
                                pre_indexes.append(row.index(ele_2))
                ele.append(pre_indexes)
                row.append(ele)
            # Replace machine names with their indices in self.machines.
            for ele in row:
                self.total_num += 1
                try:
                    ele[0] = self.machines.index(ele[0].lower())
                except:
                    self.compile_error_num += 1
                    ele[0] = 0

            print("row: ", row, "\n")
            self.or_matrix.append(row)
        write_json(self.dump_dir_path + "or_matrix.json", self.or_matrix)
        write_json(self.dump_dir_path + "machines.json", self.machines)
        print("or_matrix2JSP_result ing...")
        or_matrix = copy.deepcopy(self.or_matrix)
        assigned_jobs, solver, err_rate, makespan = schedule(or_matrix)
        if len(assigned_jobs) == 0:
            print("No solver")
            self.compile_error_num += 1
            return
        else:
            self.assigned_jobs = assigned_jobs
            self.solver = solver
        write_json(self.dump_dir_path + "assigned_jobs.json", self.assigned_jobs)
        write_txt(self.dump_dir_path + "err_rate.txt", str(err_rate))
        write_txt(self.dump_dir_path + "makespan.txt", str(makespan))

    def dsl_program2route_sheets(self):
        for i in range(len(self.production_programs)):
            operation_view = self.operation_programs[i]
            production_view = self.production_programs[i]
            route_sheet = []
            # Create a mapping from component_type in production_view to FlowUnit details
            production_mapping = {}
            for step in production_view:
                component_type = step["FlowUnit"]["component_type"]
                production_mapping[component_type] = step["FlowUnit"]

            for operation in operation_view:
                # Extract preconditions and postconditions from operation-view
                precondition_components = operation.get("Precond", {}).get("SlotArg", [])
                postcondition_components = operation.get("Postcond", {}).get("EmitArg", [])

                # Map preconditions and postconditions to production-view details
                preconditions = [production_mapping.get(comp, {
                    "component": comp,
                    "component_type": comp,
                }) for comp in precondition_components]

                postconditions = [production_mapping.get(comp, {
                    "component": comp,
                    "component_type": comp,
                }) for comp in postcondition_components]

                # Create the route sheet entry
                route_sheet_entry = {
                    "machine": operation["Execution"].get("machine", ""),
                    "duration": operation["Execution"].get("duration", ""),
                    "precondition": preconditions,
                    "postcondition": postconditions,
                    "operation": operation["Operation"],
                    "parameters": operation["Execution"].get("parameters", {})
                }

                route_sheet.append(route_sheet_entry)
            self.route_sheets.append(route_sheet)
        write_json(self.dump_dir_path + "route_sheets.json", self.route_sheets)

    def JSP_result2production_plan(self):
        print("JSP_result2production_plan ing")
        self.__extract_machines()
        production_plan = []
        for machine_index, production_sequence in self.assigned_jobs.items():
            if int(machine_index) >= len(self.machines):
                continue
            machine_plan = {
                "machine": self.machines[int(machine_index)],
                "production_sequence": []
            }
            for step in production_sequence:
                if step[2] < len(self.operation_programs[step[1]]):
                    print("self.operation_programs[step[1]][step[2]]: ", self.operation_programs[step[1]][step[2]], "\n")
                    precond = self.operation_programs[step[1]][step[2]].get("Precond", {}).get("SlotArg", [])
                    postcond = self.operation_programs[step[1]][step[2]].get("Postcond", {}).get("EmitArg", [])
                    parameters = self.operation_programs[step[1]][step[2]].get("Execution", {}).get("parameters", {})
                else:
                    precond = []
                    postcond = []
                    parameters = {}
                    self.compile_error_num += 1
                machine_plan["production_sequence"].append({
                    "start": step[0],
                    "end": step[0] + step[3],
                    "job_id": step[1],
                    "task_id": step[2],
                    "precondition": precond,
                    "postcondition": postcond,
                    "parameters": parameters
                })
            production_plan.append(machine_plan)
        self.production_plan = production_plan
        write_json(self.dump_dir_path + "production_plan.json", self.production_plan)
        write_txt(self.dump_dir_path + "compile_error_num.txt", str(self.compile_error_num))

    def __extract_machines(self):
        operation_programs = copy.deepcopy(self.operation_programs)
        production_programs = copy.deepcopy(self.production_programs)
        operation2machine_dict = {}
        for i in range(len(operation_programs)):
            operation_program = operation_programs[i]
            for step in operation_program:
                operation = step["Operation"]
                try:
                    machine = step["Execution"]["machine"]
                except:
                    machine = None
                    self.compile_error_num += 1
                    continue
                if operation not in operation2machine_dict:
                    operation2machine_dict[operation] = [machine]
                else:
                    operation2machine_dict[operation].append(machine)
        # Normalize machine names to lowercase.
        self.machines = list(set([machine.lower() for machine_list in operation2machine_dict.values() for machine in machine_list]))

    def __chatgpt_function(self, content, gpt_model="gpt-4o"):
        for attempt in range(5):
            try:
                client = OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY", "sk-placeholder"),
                    base_url="http://localhost:4141/v1"
                )
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "user", "content": content}
                    ],
                    model=gpt_model,
                    max_tokens=8192,
                )
                return chat_completion.choices[0].message.content
            except Exception as e:
                print(f"error (attempt {attempt+1}/5): ", e)
                if attempt < 4:
                    time.sleep(2 ** attempt)
        raise RuntimeError("LLM call failed after 5 attempts")


    def __EM_normalize(self, EM_structure):
        preds_index = 0
        succs_index = 0
        EM_structure = copy.deepcopy(EM_structure)
        for EM in EM_structure:
            for pred in EM["preds"]:
                pred = "Pred" + str(preds_index)
                preds_index += 1
            for succ in EM["succs"]:
                succ = "Succ" + str(succs_index)
                succs_index += 1
        return EM_structure

    def __cosine_similarity(self, vec1, vec2):
        # Ensure vectors are NumPy arrays.
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        # Compute cosine similarity.
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    def __operation_extraction(self, sentence):
        prompt = self.operation_extraction_prompt.replace("---SENTENCES---", sentence)
        for _ in range(5):
            response = self.__chatgpt_function(prompt).strip().upper()
            if "NONE" in response:
                return ["NONE"]
            operations = [
                self.lemmatizer.lemmatize(op.strip().lower(), pos="v") 
                for op in response.split(",") if op.strip()
            ]
            if all(sum(1 for token in self.nlp(op) if not token.is_punct) == 1 for op in operations):
                return operations
        return ["NONE"]

    def __component_extraction(self, sentence):
        prompt = self.component_extraction_prompt.replace("---SENTENCES---", sentence)
        for _ in range(5):
            response = self.__chatgpt_function(prompt).strip()
            if "NONE" in response:
                return ["NONE"]
            return [flowunit.strip() for flowunit in response.split(",") if flowunit.strip()]

    def __similarity_opcode(self, operation):
        if not operation or operation == "NONE":
            return "NONE"
        operation_lower = operation.lower()
        closest_word_list = []
        closest_word = "NONE"
        max_similarity = -1
        for opcode in self.operation_dsl:
            similarity = self.__cosine_similarity(self.__get_embedding(operation_lower), self.__get_embedding(opcode))
            if similarity > max_similarity:
                max_similarity = similarity
                closest_word = opcode
        return [closest_word]

    def __similarity_component(self, flowunit):
        if not flowunit or flowunit == "NONE":
            return "NONE"
        components = list(self.production_dsl.keys())
        closest_word = "NONE"
        max_similarity = -1
        for i, component_word in enumerate(components):
            similarity = self.__cosine_similarity(self.__get_embedding(flowunit), self.__get_embedding(component_word))
            if similarity > max_similarity:
                max_similarity = similarity
                closest_word = components[i]
        return [closest_word]
    
    def __get_embedding(self, text):
        '''get embedding'''
        if text in self.embedding_dic:
            return self.embedding_dic[text]
        else:
            while True:
                try:
                    client = OpenAI()
                    response = client.embeddings.create(
                        input=text,
                        model="text-embedding-3-small"
                    )
                    self.embedding_dic[text] = response.data[0].embedding
                    write_json("data/embedding_dic.json", self.embedding_dic)
                    return response.data[0].embedding
                except openai.APIError as error:
                    print("error: ", error)
                    continue

    def __gpt_batch_store(self, sys_content, user_content, index, model="deepseek-v3"):
        standard = {"custom_id": "", "method": "POST", "url": "/v1/chat/completions", "body": {"model": model, "messages": [{"role": "system", "content": ""},{"role": "user", "content": ""}],"max_tokens": 8192}}
        prompt_unit = standard.copy()
        prompt_unit["body"]["messages"][0]["content"] = sys_content
        prompt_unit["body"]["messages"][1]["content"] = user_content
        prompt_unit["custom_id"] = index
        with open(self.batch_input_path, 'a') as file:
            # Append the request as one JSONL line.
            json_line = json.dumps(prompt_unit)
            file.write(json_line + '\n')

    def __gpt_batch_call(self):
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # DashScope compatible API endpoint.
        )
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
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1" 
        )
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
                result = client.files.content(result_file_id).text
                result_file_name = self.batch_output_path
                with open(result_file_name, 'wb') as file:
                    file.write(result.encode('utf-8'))
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

    def __run_in_parallel(self, func, N, max_workers=8, has_feedback=False):
        # Run tasks with a bounded thread pool.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for _ in trange(N, total=N):
                futures.append(executor.submit(func, has_feedback=has_feedback))
                if len(futures) >= max_workers:
                    for future in as_completed(futures):
                        futures.remove(future)

            # Wait for the remaining tasks to finish.
            for future in as_completed(futures):
                future.result()


    def store_embedding_dic(self):
        write_json("data/embedding_dic.json", self.embedding_dic)
