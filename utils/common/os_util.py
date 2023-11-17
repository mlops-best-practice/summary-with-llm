import os
import json
import pandas as pd


def save_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w', encoding="utf-8") as json_file:
        print(f"upload file to: {file_path}")
        json.dump(data, json_file, indent=4, ensure_ascii=False)
    print(f"SAVE COMPLATED | PATH ={file_path}")

def save_txt(file_path, data_str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding="utf-8") as f:
        f.write(data_str)
    print(f"SAVE COMPLATED | PATH = {file_path}")

def save_docx(file_path: str, docx_bytes: bytes):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        f.write(docx_bytes)
    print(f"SAVE COMPLATED | PATH = {file_path}")

def append_to_file(file_path, data_str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        with open(file_path, 'a', encoding="utf-8") as f:
            f.write(data_str)
        print(f"Data added to file: {file_path}")
    except IOError as e:
        print(f"An error occurred while adding data to {file_path}: {str(e)}")

def load_file_txt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return "".join(lines)


def load_file_json(path):
    with open(path, 'r') as f:
        return  json.load(f)

def load_all_json_file(folder_path: str):
    outputs = []
    paths = get_all_file_paths(folder_path)
    for path in paths:
        if ".json" in path:
            outputs.append(load_file_json(path))
        else:
            print(f"WARN: {path} isn't json file")
    return outputs

def get_all_file_paths(folder_path):
    file_paths = []  
    if os.path.exists(folder_path):
        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                file_paths.append(file_path)

    return file_paths


def load_file_csv(path):
    return pd.read_csv(path)

def load_all_csv_file(folder_path: str):
    outputs = pd.DataFrame()

    paths = get_all_file_paths(folder_path)
    for path in paths:
        if ".csv" in path:
            df = pd.read_csv(path)
            outputs = pd.concat([outputs, df], ignore_index=True)
        else:
            print(f"WARN: {path} isn't csv file")
    return outputs


            

def load_all_txt(folder_path, *args, **kwargs):
    paths = get_all_file_paths(folder_path)
    output = []
    for path in paths:
        if ".txt" in path:
            output.append(load_file_txt(path, *args, **kwargs))
    return output
    