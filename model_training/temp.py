'''
import pandas as pd
from config import csv_path
df = pd.read_csv(csv_path)
df.dropna(subset=['requirement_description', 'test_steps'], inplace=True)
print(df.notna().sum())
'''
import torch
print(f"Device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
