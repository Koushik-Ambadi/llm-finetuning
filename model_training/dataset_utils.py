# model_training/dataset_utils.py

import pandas as pd
from datasets import Dataset
from model_training.config import prompt_prefix


def format_prompt(batch):
    inputs = [f"{prompt_prefix}{req}" for req in batch['requirement_description']]
    outputs = batch['test_steps']
    return {
        "input": inputs,
        "output": outputs
    }

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df.dropna(subset=['requirement_description', 'test_steps'], inplace=True) #should be handled in data prep
    #df[['requirement_description', 'test_steps']] = df[['requirement_description', 'test_steps']].fillna("")  '''for now we remove data in later sessions we can fill empty string'''
    dataset = Dataset.from_pandas(df[['requirement_description', 'test_steps']])
    dataset = dataset.map(format_prompt, batched=True)
    return dataset


def preprocess_dataset(dataset):
    
    return dataset.train_test_split(test_size=0.01, seed=42)


