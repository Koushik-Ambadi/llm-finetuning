# train.py

from model_training.training_pipeline import train_model
from model_training.config import output_dir
import os

def main():
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Call the training pipeline
    train_model()

if __name__ == "__main__":
    main()
