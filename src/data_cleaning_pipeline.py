import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    parse_column,
    normalize_column,
    clean_column,
    merge_df
)


def pipeline(df, column_name, cols_to_drop=None):
    """
    Full pipeline: parse, expand, clean, and merge parameter column.
    """
    temp_dataframe = parse_column(df, column_name)
    temp_dataframe = normalize_column(temp_dataframe)
    temp_dataframe = clean_column(temp_dataframe,cols_to_drop)
    df = merge_df(df, temp_dataframe)
    return df
