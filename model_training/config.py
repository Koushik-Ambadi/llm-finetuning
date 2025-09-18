# model_training/config.py

import os


CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.normpath(os.path.join(CONFIG_DIR, "..", "data", "flat_dataset", "temp.csv"))
model_path = "/home/navpc24/Desktop/llm-finetuning/Mistral-3B-Instruct-v0.2-init"
output_dir = "temp_mistral3b-finetuned" #sanity check using temp path


device_map="auto"
train_batch_size = 1
gradient_accumulation = 1
num_epochs = 50 #sanity check using 50
split_test_size=1 #sanity check using 0.999
learning_rate = 2e-5
target_modules = ["q_proj", "v_proj", "k_proj"] #sanity check adding extra layer for training ['down_proj', 'gate_proj', 'k_proj', 'lm_head', 'o_proj', 'q_proj', 'up_proj', 'v_proj']
rank = 16 #sanity check very high as of now improves accuracy
lora_alpha = 32 #sanity check like learning rate for adapter keeping it high for check
lora_dropout = 0.01 #sanity check regularisation step keeping it vey low as of now

    

token_batchsize= 64
max_seq_length = 256
padding_type = "max_length"   # or "dynamic" etc.
truncation_side = "right"  # or "left", or alternate
padding_side = "right"



# trining prefix instruction
prompt_prefix = "Write a test case for the following requirement:\n"
