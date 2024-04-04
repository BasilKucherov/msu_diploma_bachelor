import argparse
import os
import pandas as pd
import numpy as np
import json
import re


SPEECH_COMMANDS_TEST_SPLIT = "testing_list.txt"
SPEECH_COMMANDS_VALIDATION_SPLIT = "validation_list.txt"


def is_class_dir(path: str) -> bool:
    return os.path.isdir(path) and not path.split("/")[-1].startswith("_")


def is_wav_file(path: str) -> bool:
    return os.path.isfile(path) and path.endswith(".wav")


def get_all_links(dataset_path: str) -> pd.DataFrame:
    result = {
        "LINK": [],
        "WORD": [],
    }

    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if is_class_dir(item_path):
            cur_class = item_path.split("/")[-1]
            
            for f in os.listdir(item_path):
                f_path = os.path.join(item_path, f)
                if is_wav_file(f_path):
                    result["LINK"].append(os.path.join(cur_class, f))
                    result["WORD"].append(cur_class)
    
    return pd.DataFrame(result)


def load_file_lines(path: str) -> list[str]:
    with open(path, "r") as fp:
        lines = fp.read().splitlines()
    
    return lines

# WORD, LINK, SET
# SET = {"TRAIN", "TEST", "VALID"}
def get_original_split(dataset_path: str, test_split: str, validation_split: str):
    df = get_all_links(dataset_path)

    test_links = load_file_lines(os.path.join(dataset_path, test_split))
    validation_links = load_file_lines(os.path.join(dataset_path, validation_split))

    df['SET'] = df.apply(lambda row: "TEST" if row.LINK in test_links else ("VALID" if row.LINK in validation_links else "TRAIN"), axis=1)
    return df


def fill_filename(filename_format: int, packs_number: int, spc_number: int):
    # Find the pack numeric part using regex
    pack_pattern = r'pack(\d+)'
    match = re.search(pack_pattern, filename_format)
    
    if not match:
        return None
    
    numeric_part = match.group(1)
    replaced_string = re.sub(pack_pattern, str(packs_number).zfill(len(numeric_part)) + "pack", filename_format)

    # Find the spc numeric part using regex
    spc_pattern = r'(\d+)spc'
    match = re.search(spc_pattern, replaced_string)
    
    if not match:
        return None
    
    numeric_part = match.group(1)
    replaced_string = re.sub(spc_pattern, str(spc_number).zfill(len(numeric_part)) + "spc", replaced_string)

    return replaced_string
    

def create_split(split_description: dict, original_split: pd.DataFrame):
    os.makedirs(split_description["dir"], exist_ok=True)

    if split_description["type"] == "usual":
        result_split = {
            "WORD": [],
            "LINK": []
        }
        
        for word in split_description["classes"]:
            word_links_list = list(original_split[original_split["SET"].isin(split_description["original_set"]) & (original_split["WORD"] == word)]["LINK"])
            result_split["WORD"] += [word] * split_description["samples_per_class"]
            result_split["LINK"] += list(np.random.choice(word_links_list, size=split_description["samples_per_class"], replace=False))
    
        save_path = os.path.join(split_description["dir"], split_description["name"])
        pd.DataFrame(result_split).to_csv(save_path, index=False)
    elif split_description["type"] == "fsl":
        for samples_per_class in split_description["samples_per_class"]:
            for pack_number in range(split_description["packs_number"]):
                result_split = {
                    "WORD": [],
                    "LINK": []
                }
                
                for word in split_description["classes"]:
                    word_links_list = list(original_split[original_split["SET"].isin(split_description["original_set"]) & (original_split["WORD"] == word)]["LINK"])
                    result_split["WORD"] += [word] * samples_per_class
                    result_split["LINK"] += list(np.random.choice(word_links_list, size=samples_per_class, replace=False))
                
                save_name = fill_filename(split_description["name"], pack_number, samples_per_class)
                save_path = os.path.join(split_description["dir"], save_name)
                pd.DataFrame(result_split).to_csv(save_path, index=False)


if __name__ == "__main__":
    np.random.seed(42)

    parser = argparse.ArgumentParser(description="Command line arguments")
    parser.add_argument("--dataset", type=str, help="Path to the speech commands dataset directory", required=True)
    parser.add_argument("--config", type=str, help="Path to the config file with splits description", required=True)

    args = parser.parse_args()


    original_split = get_original_split(args.dataset, SPEECH_COMMANDS_TEST_SPLIT, SPEECH_COMMANDS_VALIDATION_SPLIT)

    with open(args.config, "r") as fp:
        splits_descriptions = json.load(fp)
    
    for name, split_description in splits_descriptions.items():
        print(f"Creating {name}")
        create_split(split_description, original_split)