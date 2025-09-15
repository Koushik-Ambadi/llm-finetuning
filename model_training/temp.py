from dataset_utils import load_dataset, preprocess_dataset
dataset = load_dataset(csv_path)
#logger.info("Loaded dataset")
print(f"Dataset sample:\n{dataset[0]}")