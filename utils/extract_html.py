import pandas as pd
import re
def extract_html_tags(text):

#Extract all HTML tags from a string.
    
    if pd.isna(text):
        return []
    return re.findall(r'<\s*([a-zA-Z0-9]+)', str(text))


def extract_html_from_column(df, column_name):
    """
    Apply `extract_html_tags` on a specific column of a DataFrame.

    Returns:
        pd.Series: Series with lists of tags from each cell.
    """
    return df[column_name].apply(extract_html_tags)