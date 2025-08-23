import os
import json
import pandas as pd
from pathlib import Path

input_folder = "../data/raw_dataset"
output_csv = "../data/flat_dataset/merged_output.csv"

def json_tocsv(input_path,output_path):
    all_data = []

    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            filepath = os.path.join(input_folder, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = [data]

            # Flatten one level deep
            flattened = []
            for item in data:
                flat_item = {}
                for key, value in item.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            flat_item[sub_key] = sub_value
                    else:
                        flat_item[key] = value
                flat_item["file_name"] = Path(filename).stem
                flattened.append(flat_item)

            df = pd.DataFrame(flattened)
            all_data.append(df)

    merged_df = pd.concat(all_data, ignore_index=True)
merged_df.to_csv(output_csv, index=False)
