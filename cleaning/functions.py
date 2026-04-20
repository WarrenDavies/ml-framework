from cleaning.registry import register
import numpy as np


@register
def drop_blank_rows(df):
    df_cleaned = df.copy()
    df_cleaned = df.dropna(how='all')
    return df_cleaned


@register
def convert_values_to_null(df_input, old_col_name, new_col_name, values):
    df_output = df_input.copy()
    df_output[new_col_name] = df_output[old_col_name].replace(values, np.nan)
    return df_output