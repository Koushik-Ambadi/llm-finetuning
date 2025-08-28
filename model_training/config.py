# model_training/config.py

csv_path = "data/final_dataset.csv"
model_path = "Mistral-3B-Instruct-v0.2-init"
output_dir = "mistral3b-finetuned"

train_batch_size = 1
gradient_accumulation = 4
num_epochs = 3
learning_rate = 2e-5
target_modules = ["q_proj", "v_proj"]
