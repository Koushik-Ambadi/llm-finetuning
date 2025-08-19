import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    parse_column,
    normalize_column,
    clean_column,
    merge_df
)
from utils.html_cleaner import clean_html_columns



def pipeline(df):
    """
    Full pipeline: parse, expand, clean, and merge parameter column.
    """
    #drop irrelavent fields in testcases column
    cols_to_drop = ['refId','stepData','id','type','extensionsHideInfomation','imageHideInformation','extnId']

    #remove html tags from these coulmns
    columns_to_clean = ['TestCaseFormid']  # replace with your actual columns
    
    temp_dataframe = parse_column(df)
    temp_dataframe = normalize_column(temp_dataframe)
    temp_dataframe = clean_column(temp_dataframe,cols_to_drop)
    df = merge_df(df, temp_dataframe)
    df = clean_html_columns(df, columns_to_clean)
    return df


