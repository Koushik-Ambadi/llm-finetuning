import pandas as pd
import ast


def parse_column(df):
    """
    Returns a new DataFrame with non-null rows from `column_name`,
    parsing the column safely into lists of dicts.
    Does NOT modify the original DataFrame.
    """
    column_name='TestCaseFormid'
    def _parse(x):
        if pd.isna(x):
            return x
        if isinstance(x, str):
            try:
                parsed = ast.literal_eval(x)
            except (SyntaxError, ValueError):
                return x  # return as-is if parsing fails
        else:
            parsed = x

        if isinstance(parsed, dict):
            return [parsed]  # wrap single dict into list
        return parsed

    # Only operate on rows where the column is not null
    valid_dataframe = df[df[column_name].notna()].copy()
    valid_dataframe[column_name] = valid_dataframe[column_name].apply(_parse)
    return valid_dataframe


def normalize_column(valid_dataframe, column_name='TestCaseFormid', id_column='file_name'):
    """
    Extracts list-of-dict entries from a parsed column into flat rows,
    tagging each with its associated ID (like file_name).
    Prints a quick summary of how many unique IDs are present before and after.
    """
    extracted_rows = []

    for _, row in valid_dataframe.iterrows():
        item_id = row[id_column]
        params = row[column_name]

        if isinstance(params, list):
            for param in params:
                if isinstance(param, dict):
                    param_copy = param.copy()
                    param_copy[id_column] = item_id
                    extracted_rows.append(param_copy)

    normalized_df = pd.DataFrame(extracted_rows)

    '''
    # === Print summary of normalization ===
    input_count = valid_dataframe[id_column].nunique()
    output_count = normalized_df[id_column].nunique()
    print(f"[NORMALIZED] {output_count} unique `{id_column}` values (from {input_count} in input)")
    '''

    return normalized_df


def clean_column(normalized_df, cols_to_drop=[],column_name='TestCaseFormid', id_column='file_name'):
    """
    Clean extracted parameter DataFrame:
    - Drop unwanted columns
    - Group back into list-of-dicts per id_column
    """
    normalized_df = normalized_df[
    (normalized_df["type"] != "Pre Conditions") &
    (normalized_df["type"] != "Post Conditions") &
    (normalized_df["action"] != "Test Sequence")]
    
    if cols_to_drop: 
        normalized_df = normalized_df.drop(columns=[c for c in cols_to_drop if c in normalized_df.columns])

    # Group and convert to list-of-dicts
    grouped_df = (
        normalized_df
        .groupby(id_column)
        .apply(lambda g: g.drop(columns=id_column).to_dict(orient='records'))
        .reset_index(name=f'Cleaned{column_name}')
    )

    return grouped_df


def merge_df(df, grouped_df,column_name='TestCaseFormid', id_column='file_name'):
    """
    Merge the cleaned parameter list back into the original DataFrame.
    Dynamically detects the cleaned column name from grouped_df.
    Drops the original nested column before merging.
    """
    # Drop the original column if it exists
    #df = df.drop(column_name)

    # Merge the cleaned column back into the main DataFrame
    df = df.drop(columns=[column_name], errors='ignore')  # drop old column if exists
    df = df.merge(grouped_df, on=id_column, how='left')
    return df



