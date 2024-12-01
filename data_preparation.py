import shutil, os, time, datetime, io
from datetime import datetime
from random import choice, randint
import zipfile, json
import requests
import time
from typing import List, Optional
#from gpt3 import gpt3_generator
from string import ascii_lowercase
import numpy as np
import json 
import csv 
import pandas as pd
import time
from sklearn.model_selection import train_test_split
#from config import OPENAI_API_KEY
import re
from openai import OpenAI
import json
import re
import argparse
import mlflow

client = OpenAI(api_key='')#Enter your code here

def find_value_in_context(value, context):
    value = re.escape(value)
    result = re.search(fr"{value}", context, flags=re.I)
    if result is not None:
        start_index, end_index = result.span()
    else:
        start_index, end_index = -1, -1
    return start_index

def text_strip_clean(text):
    text = text.strip()
    text_tokens = [token.strip() for token in text.split() if token.strip()]
    text = " ".join(text_tokens)
    return text

def normalize_text(text):
    text = text.lower().strip()
    text_tokens = [token.strip() for token in text.split() if token.strip()]
    text = " ".join(text_tokens)
    return text

def remove_indexes_from_list(input_list):
    output_list = []
    for item in input_list:
        # Check if the string starts with digits followed by a dot and a space
        if re.match(r'^[\d+\.\s*-]*', item):
            # If it matches, remove the index and special characters
            cleaned_item = re.sub(r'^[\d+\.\s*-]*', '', item).lstrip(' -').strip()
            output_list.append(cleaned_item)
        else:
            # If it doesn't match, leave the item unchanged
            output_list.append(item)
    return output_list

#start text
def Question_Generate(parametername, context):
    
    content=f"Please generate 9 questions about the 'str({parametername})', given the following context: "+ str({context}) +". The questions should be simple, concise and direct, aiming to extract the before mentioned parameter from the context. These questions will be used to augment our extractive question answering dataset for fine-tuning."
    
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Please generate 9 questions about the coolant temperature, given the following context: Coolant temperature 25 °C. The questions should be simple, concise and direct, aiming to extract the before mentioned parameter from the context. These questions will be used to augment our extractive question answering dataset for fine-tuning."},
        {"role": "assistant", "content": "what is the value of coolant temperature?\nHow much is the value of coolant temperature\nwhat is the exact value of coolant temperature\nHow much is the coolant temperature"},
        {"role": "user", "content": content},
    ])
    output= response.choices[0].message.content.splitlines()
    output= remove_indexes_from_list(output)
    return output

def Tender_Generate(parametername, parametervalue):
    content=f"Generate semantically meaningful sentences for tender specifications of the given parameter '{parametername}' and the value of that parameter is '{parametervalue}'. Make sure that the generated text is in natural language and is of 2-3 lines at maximum. USE THE NAMES OF PARAMETER AND ITS VALUE EXACTLY SAME AS IT IS!!"
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content":"Generate semantically meaningful tender specifications for the given parameter name speed of rotation and the value of that parameter is 1600 rpm. Make sure that the generated text is in natural language and is a maximum of 2-3 lines at maximum. Keep the names of parameter and its values as it is."},
        {"role": "assistant", "content": "The speed of rotation for the mechanical motors under torque must be more than 1600 rpm."},
        {"role": "user", "content": content},
    ])
    output= response.choices[0].message.content.splitlines()
    output= remove_indexes_from_list(output)
    return output

def process_excel_to_json(args, output_dir):
    unique_combinations = set()
    data_list = []
    row_name_parametername=args.row_name_parametername
    row_name_parametervalue=args.row_name_parametervalue
    row_name_context=args.row_name_context
    excel_files=args.input
    required_columns = {row_name_parametername, row_name_parametervalue}
    
    if not args.create_context_using_chatgpt:
        # Check for required columns when create_context_using_chatgpt is False
        required_columns.add(row_name_context) 
           
    for excel_file in excel_files:
        # Read the Excel file
        df = pd.read_excel(excel_file)
        print(f'the columns in the dataset is{df.columns}')
        
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Invalid format: Missing required columns {required_columns}")
            
        # Filter the DataFrame for non-empty 'parametername' and 'parametervalue'
        filtered_df = df[(df[row_name_parametername].notna()) & (df[row_name_parametervalue].notna())]
        
        # Convert the filtered data into a list of dictionaries
        
        for index, row in filtered_df.iterrows():
            parametername = str(row[row_name_parametername])
            parametervalue = str(row[row_name_parametervalue])
            combination_key = (normalize_text(parametername), normalize_text(parametervalue))
            parametername = text_strip_clean(parametername)
            parametervalue = text_strip_clean(parametervalue)
            # Check if the combination_key is already in the set
            new_dictionary={
                "parameter_name": parametername,
                "parameter_value": parametervalue
            }
            if not args.create_context_using_chatgpt:
                # Add the dictionary to the data_list
                context = str(row['Text'])
                new_dictionary['context']=context
                data_list.append(new_dictionary)
            else:
                if combination_key not in unique_combinations:
                    # Add the combination_key to the set
                    unique_combinations.add(combination_key)
                    data_list.append(new_dictionary)
                
    # Create the output file name based on the suffix
    output_file_name = "data.json"
    output_json_file = os.path.join(output_dir, output_file_name)
    # Write the data from the current Excel file to the JSON file
    print(f"Data extracted has been dumped to {output_json_file}")
    with open(output_json_file, "w", encoding="utf-8") as json_file:
        json.dump(data_list, json_file, indent=4, ensure_ascii=False)

    return output_json_file
    
def update_json_file(json_file_path):
    # Load the JSON data
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Create a new list to hold updated data
    updated_data = []

    # Update the answer_start value for each item
    for item in data:
        parameter_value = item['answers']['text'][0]
        context = item['context']
        answer_start = find_value_in_context(parameter_value, context)
        item['answers']['answer_start'][0] = int(answer_start)
        updated_data.append(item)

    # Save the updated data back to the same JSON file
    with open(json_file_path, 'w', encoding='utf-8') as file:
        json.dump(updated_data, file, indent=4, ensure_ascii=False)

def create_original_augmented_context(input_file, look_up_file):
    # Read the JSON file
    lookup_dict = {}
    new_dataset = []
    if (look_up_file is not None) and os.path.exists(look_up_file):
        with open(look_up_file, 'r',encoding='utf-8') as file:
            data = json.load(file)
        for item in data:
            key = (normalize_text(item["parameter_name"]), normalize_text(item["parameter_value"]))
            lookup_dict[key] = item["context"]

    # Create an empty list to store the new questions
    count=1
    with open(input_file, 'r',encoding='utf-8') as file:
        data = json.load(file)
    # Extract relevant information from the JSON data
    for item in data:
        parameter_name = item['parameter_name']
        parameter_value = item['parameter_value']
        combination_key = (normalize_text(parameter_name), normalize_text(parameter_value))
        # Check if the combination key exists in the lookup_dict
        if combination_key in lookup_dict:
            # If yes, assign the context from lookup_dict
            context = lookup_dict[combination_key]
        else:
            context = Tender_Generate(parameter_name, parameter_value)
            context=context[0]
              # Append each generated question to the list
        new_dictionary={
            "parameter_name": str(parameter_name),
            "parameter_value": str(parameter_value),
            "context": str(context)
            }
      
        print(count)
        count+=1
        new_dataset.append(new_dictionary)
        # Write the new questions to a JSON file

    output_dir = os.path.dirname(input_file)
    # Create the output file name based on the suffix
    output_file_name = "context_data.json"
    output_file = os.path.join(output_dir, output_file_name)

    with open(output_file, 'a', encoding='utf-8') as file:
        json.dump(new_dataset, file, indent=4, ensure_ascii=False)
    return output_file


def change_data_format_to_squad(input_file, output_dir):
    with open(input_file, 'r',encoding='utf-8') as file:
        data = json.load(file)
    # Create lists to store entries for each JSON file
    dataset_with_answer = []
    dataset_without_answer = []
    for count,item in enumerate(data):
        context=item['context']
        parameter_value=item['parameter_value']
        parameter_name=item['parameter_name']
        answer_start=int(find_value_in_context(parameter_value, context))
        new_question = {
                "answers": {
                "answer_start":[answer_start],  # Assuming answer always starts from the beginning of the context
                "text": [parameter_value]
            },
            "context":context,
            "id": str(count),
            "question":f"what is the value of {parameter_name}?",
            "title": parameter_name
            
        }

        #print(f'the answer start is {answer_start}')
        # Append to the appropriate list based on answer_start
        if answer_start == -1:
            dataset_without_answer.append(new_question)
        else:
            dataset_with_answer.append(new_question)
    # Define output directory and file names
    output_file_with_answer = os.path.join(output_dir, "squad_data_with_answer.json")
    output_file_without_answer = os.path.join(output_dir, "squad_data_without_answer.json")
        
    # Write both datasets to separate JSON files
    with open(output_file_with_answer, 'w', encoding='utf-8') as file:
            json.dump(dataset_with_answer, file, indent=4, ensure_ascii=False)
    with open(output_file_without_answer, 'w', encoding='utf-8') as file:
            json.dump(dataset_without_answer, file, indent=4, ensure_ascii=False)
        
    print(f"Data with answer_start saved at {output_file_with_answer}")
    print(f"Data without answer_start saved at {output_file_without_answer}")
    
    return output_file_with_answer

def create_final_json_augmented(input_file, look_up_file):
    lookup_dict={}
    if (look_up_file is not None) and os.path.exists(look_up_file):
        with open(input_file, 'r',encoding='utf-8') as file:
            data = json.load(file)
        for item in data:
            key = (normalize_text(item["parameter_name"]), normalize_text(item["parameter_value"]))
            question=item['question']
            if normalize_text(question)==normalize_text(f"what is the value of {parameter_name}?"):
                continue
            # Check if the key already exists in lookup_dict
            if key in lookup_dict:
                # If the key exists, append the context to the existing list
                lookup_dict[key].append(item["question"])
            else:
                # If the key doesn't exist, create a new list with the context
                lookup_dict[key] = [item["question"]]

    # Read the JSON file
    count=1
    with open(input_file, 'r',encoding='utf-8') as file:
        data = json.load(file)
    
    # Create an empty list to store the new questions
    valid_questions = []
    unvalid_questions=[]
    
    # Extract relevant information from the JSON data
    for item in data:
        parameter_name = item['title']
        context = item['context']
        parameter_value = item['answers']['text']  # Assuming there's only one answer
        combination_key = (normalize_text(parameter_name), normalize_text(parameter_value[0]))
        if combination_key in lookup_dict:
            # If yes, assign the context from lookup_dict
            questions = lookup_dict[combination_key]
        # Generate questions
        else:
            questions = Question_Generate(parameter_name, context)
        
        # Append each generated question to the list
        for question in questions:
            answer_start=item['answers']['answer_start']
            new_question = {
                    "answers": {
                    "answer_start": answer_start,  # Assuming answer always starts from the beginning of the context
                    "text": parameter_value
                },
               
                "context":context,
                "id": item['id'],
                "question":question,
                "title": parameter_name
                
            }
            print(count)
            count+=1
            if answer_start == -1:
                unvalid_questions.append(new_question)
            else:
                valid_questions.append(new_question)
    # Write the new questions to a JSON file
    output_dir = os.path.dirname(input_file)
    # Create the output file name based on the suffix
    output_file_name = "augmented_data.json"
    output_json_file = os.path.join(output_dir, output_file_name)        
    with open(output_json_file, 'a', encoding='utf-8') as file:
        json.dump(valid_questions, file, indent=4, ensure_ascii=False)
    invalid_json_file = os.path.join(output_dir, 'invalid_augmented_data')        
    with open(invalid_json_file, 'a', encoding='utf-8') as file:
        json.dump(unvalid_questions, file, indent=4, ensure_ascii=False)
    print(f"The augmented dataset has been saved to {output_json_file}")
    return output_json_file

def split_and_save_datasets(json_file_path, output_dir):
    # Load the dataset from the JSON file
    print(f'the file path is {json_file_path}')
    with open(json_file_path, "r", encoding='utf-8') as file:
        dataset = json.load(file)

    # Split the dataset into train (80%) and temp (20%)
    train_dataset, temp_dataset = train_test_split(dataset, test_size=0.30, random_state=42, shuffle=True)

    # Split the temp dataset into eval (50% of temp) and test (50% of temp)
    eval_dataset, test_dataset = train_test_split(temp_dataset, test_size=0.60, random_state=42, shuffle=True)

    # Define paths for the output files
    train_path = os.path.join(output_dir, "train.json")
    eval_path = os.path.join(output_dir, "eval.json")
    test_path = os.path.join(output_dir, "test.json")

    # Write each dataset to a separate JSON file
    with open(train_path, "w", encoding='utf-8') as file:
        json.dump(train_dataset, file, indent=4, ensure_ascii=False)

    with open(eval_path, "w", encoding='utf-8') as file:
        json.dump(eval_dataset, file, indent=4, ensure_ascii=False)

    with open(test_path, "w", encoding='utf-8') as file:
        json.dump(test_dataset, file, indent=4, ensure_ascii=False)

    print(f"Train dataset written to {train_path}")
    print(f"Eval dataset written to {eval_path}")
    print(f"Test dataset written to {test_path}")

    return train_path, eval_path, test_path

def extract_ids_from_json(file_name):
    # Load the JSON file
    with open(file_name, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Extract IDs from dictionaries
    ids = [str(entry['id']) for entry in data]
    
    return ids

def filter_and_save_json(file_name, ids, output_file_suffix, output_dir):
    # Load the JSON file
    with open(file_name, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Filter dictionaries with IDs present in the given list
    filtered_data = [entry for entry in data if entry['id'] in ids]
    
    # Create the output file name based on the suffix
    output_file_name = f"augmented_{output_file_suffix}.json"
    output_file_path = os.path.join(output_dir, output_file_name)
    
    # Save the filtered data to the new JSON file
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(filtered_data, file, indent=4, ensure_ascii=False)
    
    print(f"Filtered data saved to {output_file_path}")
    return output_file_path

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

# Define args as an attribute dictionary
args = AttrDict({
    'prepare': AttrDict({'enabled': False, 'input': ['predicted_results _qa_original.xlsx'], 'look_up_file': None, 'create_context_using_chatgpt':False, 'row_name_parametername':'parameter_name', 'row_name_parametervalue':'parameter_Value', 'row_name_context':'Text'}),
    'update': AttrDict({'enabled': False, 'input': ''}),
    'augment': AttrDict({'enabled': False, 'input': '', 'look_up_file': ''}),
    'split': AttrDict({'enabled': True, 'input': '/usr/squad_data_with_answer.json', 'augment_file': ''}),
    'experiments_name':'data_augmentation_preparation'
})


def main(args):
    train_paths=[]
    eval_paths=[]
    test_paths=[]
    artifact_dir = mlflow.get_artifact_uri()
    output_dir=artifact_dir.replace("file://", "")
    if args.prepare.enabled:
        # Determine the directory of the input file
        context_data_path = process_excel_to_json(args.prepare, output_dir)
        if args.prepare.create_context_using_chatgpt:
            context_data_path = create_original_augmented_context(context_data_path, args.prepare.look_up_file)
        args.split.input=change_data_format_to_squad(context_data_path, output_dir)
        #print(original_squad_dataset)
    
    if args.update.enabled:
        update_json_file(args.update.input)
    
    if args.augment.enabled:
            args.split.augment_file=create_final_json_augmented(args.augment.input)
    
    if args.split.enabled:
        train_path, eval_path, test_path = split_and_save_datasets(args.split.input, output_dir)
        train_paths.append(train_path)
        eval_paths.append(eval_path)
        test_paths.append(test_path)
        
        train_ids = extract_ids_from_json(train_path)
        eval_ids = extract_ids_from_json(eval_path)
        test_ids = extract_ids_from_json(test_path)
        print(f"The number of data point in train: {len(train_ids)}")
        print(f"The number of data point in eval: {len(eval_ids)}")
        print(f"The number of data point in test: {len(test_ids)}")
        if args.split.augment_file:
            filter_and_save_json(args.split.augment_file, train_ids, 'train', output_dir)
            filter_and_save_json(args.split.augment_file, eval_ids, 'eval', output_dir)
            filter_and_save_json(args.split.augment_file, test_ids, 'test', output_dir)
   
    return train_paths, eval_paths, test_paths 

if __name__ == "__main__":
    main(args)

    
'''
def main(args):
    if args.command == 'prepare':
        data_path=process_excel_to_json(args.input)
        context_data_path=create_original_augmented_context(data_path, args.look_up_file)
        change_data_format_to_squad(context_data_path)
    elif args.command == 'update':
        update_json_file(args.input)
    elif args.command == 'augment':
        create_final_json_augmented(args.input)
    elif args.command == 'split':
        train_path,eval_path,test_path=split_and_save_datasets(args.input)
        train_ids=extract_ids_from_json(train_path)
        eval_ids=extract_ids_from_json(eval_path)
        test_ids=extract_ids_from_json(test_path)
        print(f"The number of data point in train: {len(train_ids)}")
        print(f"The number of data point in eval: {len(eval_ids)}")
        print(f"The number of data point in test: {len(test_ids)}")
        filter_and_save_json(args.augment_file,train_ids,'train')
        filter_and_save_json(args.augment_file, eval_ids,'eval')
        filter_and_save_json(args.augment_file,test_ids,'test')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run different data processing and augmentation tasks on your data")
    subparsers = parser.add_subparsers(dest='command',help="Sub-commands")

    # Prepare parser
    parser_prepare = subparsers.add_parser('prepare', help="Prepare the dataset")
    parser_prepare.add_argument('--input', type=str, default=['kumar-suprabhat/final_dataset_eval.xlsx'], help="Path to the input excel file")
    parser_prepare.add_argument('--look_up_file', type=str, default=None, help="Path to the optional lookup JSON file")

    # Update parser
    parser_update = subparsers.add_parser('update', help="Update the JSON file")
    parser_update.add_argument('input', type=str, help="Path to the input JSON file")
    #TODO Optional configuration paramerter for augmenting the dataset 
    #TODO Those models hving -1 as the answer they should be dumped in an another file, before going to gpt
    #TODO optioal parameter to add SQUAD data with the current training data

    # Augment parser
    parser_augment = subparsers.add_parser('augment', help="Create the final augmented JSON")
    parser_augment.add_argument('input', type=str, help="Path to the input JSON file")
    parser_augment.add_argument('--look_up_file', type=str, default=None, help="Path to the optional lookup JSON file")
    
    # Split parser
    parser_split = subparsers.add_parser('split', help="Split the JSON file")
    parser_split.add_argument('input', type=str, help="Path to the input JSON file")
    parser_split.add_argument('augment_file', type=str, help="Path to the output JSON file")

    args = parser.parse_args()
    main(args)
'''
