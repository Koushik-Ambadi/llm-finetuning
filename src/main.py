# main.py

import numpy as np
import pandas as pd

from data_loader import load_data
from pipelines.data_cleaning_pipeline1 import pipeline

# Constants (uppercase to indicate these should not change)
READ_PATH = "C:/Users/koushik/Desktop/project/data/flat_dataset/initial_data1.csv"
WRITE_PATH = "C:/Users/koushik/Desktop/project/data/flat_dataset/final_dataset1.csv"


def main():
    # Load the raw dataset
    df = load_data(READ_PATH)

    # Process the data through the cleaning pipeline
    pipeline(df, WRITE_PATH)


if __name__ == "__main__":
    main()
