# model_training/config.py

csv_path = "data/flat_dataset/final_dataset.csv"
model_path = "Mistral-3B-Instruct-v0.2-init"
output_dir = "mistral3b-finetuned"

device_map="auto"
train_batch_size = 1
gradient_accumulation = 4
num_epochs = 3
learning_rate = 2e-5
target_modules = ["q_proj", "v_proj"]


max_seq_length = 2048
padding_type = "max_length"   # or "dynamic" etc.
truncation_side = "right"  # or "left", or alternate
padding_side = "right"



# trining prefix instruction
prompt_prefix = "Write a test case for the following requirement:\n"
