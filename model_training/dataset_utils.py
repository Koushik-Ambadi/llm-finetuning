# model_training/dataset_utils.py

import pandas as pd
from datasets import Dataset

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    dataset = Dataset.from_pandas(df[['requirement_description', 'test_steps']])
    return dataset

def format_prompt(example):
    return {
        "input": f"Write a test case for the following requirement:\n{example['requirement_description']}",
        "output": example["test_steps"]
    }

def preprocess_dataset(dataset):
    dataset = dataset.map(format_prompt)
    return dataset.train_test_split(test_size=0.01, seed=42)
