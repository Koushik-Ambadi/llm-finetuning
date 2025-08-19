import pandas as pd
import re

def clean_html(text):
    if pd.isna(text):
        return text

    # Remove <a>, <font>, <br> entirely
    text = re.sub(r'</?(a|font|br)[^>]*>', '', text, flags=re.IGNORECASE)

    # Remove only <p> tags, keep content
    text = re.sub(r'</?p[^>]*>', '', text, flags=re.IGNORECASE)

    return text.strip()

def clean_html_columns(df, columns):
    """
    Apply HTML cleaning to specific columns in a DataFrame.
    """
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_html)
    return df
