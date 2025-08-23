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


def pipeline(df,write_path):
    """
    Full pipeline: parse, expand, clean, and merge parameter column.
    """
    #drop irrelavent fields in testcases column
    cols_to_drop = ['TestCaseFormid.refId','TestCaseFormid.stepData','TestCaseFormid.id',
                    'TestCaseFormid.type','TestCaseFormid.extensionsHideInfomation',
                    'TestCaseFormid.imageHideInformation','TestCaseFormid.extnId',

                    'ParametersList.datatype','ParametersList.commonname','ParametersList.longshortname',
                    'ParametersList.referencenumber','ParametersList.labeltype',
                    'ParametersList.reloadflag',
                    
                    'Parameter.description','Parameter.valuedescription','Parameter.referencenumber'
                    ]

    #remove unnecessary objects from these coulmns
    columns_to_clean = ['TestCaseFormid','ParametersList','Parameter'] 
    
    for col in columns_to_clean:
        temp_dataframe = parse_column(df,col)
        temp_dataframe = normalize_column(temp_dataframe,col)
        temp_dataframe = clean_column(temp_dataframe,col,cols_to_drop)
        df = merge_df(df, temp_dataframe,col)

    #clean html tags
    df = clean_html_columns(df, 'TestCaseFormid')
    df.to_csv(write_path, index=False, encoding='utf-8')


