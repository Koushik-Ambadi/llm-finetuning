# model_training/config.py

import os


CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.normpath(os.path.join(CONFIG_DIR, "..", "data", "flat_dataset", "final_dataset.csv"))
model_path = "/home/navpc24/Desktop/llm-finetuning/Mistral-3B-Instruct-v0.2-init"
output_dir = "mistral3b-finetuned"


device_map="auto"
train_batch_size = 1
gradient_accumulation = 4
num_epochs = 3
learning_rate = 2e-5
target_modules = ["q_proj", "v_proj"]

token_batchsize= 64
max_seq_length = 2048
padding_type = "max_length"   # or "dynamic" etc.
truncation_side = "right"  # or "left", or alternate
padding_side = "right"



# trining prefix instruction
prompt_prefix = "Write a test case for the following requirement:\n"
