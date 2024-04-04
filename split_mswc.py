import argparse
import os
import pandas as pd
import numpy as np
import json
import re

def is_opus_file(path: str) -> bool:
    return os.path.isfile(path) and path.endswith(".opus")

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
    
def create_split(split_description: dict, original_splits: pd.DataFrame):
    os.makedirs(split_description["dir"], exist_ok=True)
    original_split = original_splits[split_description["language"]]

    if split_description["type"] == "usual":
        result_split = {
            "WORD": [],
            "LINK": []
        }
        
        for word in split_description["classes"]:
            chosen_rows = original_split["SET"].isin(split_description["original_set"])
            word_links_list = list(original_split[chosen_rows & (original_split["WORD"] == word)]["LINK"])
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
                    chosen_rows = original_split["SET"].isin(split_description["original_set"])
                    word_links_list = list(original_split[chosen_rows & (original_split["WORD"] == word)]["LINK"])
                    result_split["WORD"] += [word] * samples_per_class
                    result_split["LINK"] += list(np.random.choice(word_links_list, size=samples_per_class, replace=False))
                
                save_name = fill_filename(split_description["name"], pack_number, samples_per_class)
                save_path = os.path.join(split_description["dir"], save_name)
                pd.DataFrame(result_split).to_csv(save_path, index=False)


def config_to_original_splits(split_configs: dict, dataset_path: str):
    original_splits = {}
    for name, config in split_configs.items():
        lang = config["language"]
        if lang not in original_splits.keys():
            print(f"LOAD: {os.path.join(dataset_path, 'splits', f'{lang}_splits.csv')}")
            original_splits[lang] = pd.read_csv(os.path.join(dataset_path, "splits", f"{lang}_splits.csv"))

    return original_splits


if __name__ == "__main__":
    np.random.seed(42)

    parser = argparse.ArgumentParser(description="Command line arguments")
    parser.add_argument("--dataset", type=str, help="Path to the speech commands dataset directory", required=True)
    parser.add_argument("--config", type=str, help="Path to the config file with splits description", required=True)

    args = parser.parse_args()

    with open(args.config, "r") as fp:
        split_configs = json.load(fp)
    
    original_splits = config_to_original_splits(split_configs, args.dataset)
    
    for name, split_description in split_configs.items():
        print(f"Creating {name}")
        create_split(split_description, original_splits)