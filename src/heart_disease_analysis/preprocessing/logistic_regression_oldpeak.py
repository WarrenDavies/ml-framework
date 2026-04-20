import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from heart_disease_analysis.preprocessing import functions as ppf
from heart_disease_analysis.utils import utils
from heart_disease_analysis.preprocessing.registry import register


@register
def logistic_regression_baseline(df, config):
    print(f"Number of rows before preprocessing: {len(df)}")

    # convert target col to binary
    df = ppf.make_int_col_binary(df, "num", "num", 0)

    # label encode string values
    for col, new_value_map in config["cols_to_label_encode"].items():
        df = ppf.label_encode(df, col, col, new_value_map)

    # one hot encoding
    categorical_variables = list(config["cols_to_label_encode"].keys())
    for col in config["cols_to_one_hot_encode"]:
        df = pd.get_dummies(df, columns=[col], prefix=col)
        new_columns = list(df.columns[df.columns.str.startswith(col + "_")])
        categorical_variables = categorical_variables + new_columns

    # fill missing continuous values with column mean
    for col in config["continuous_variables"]:
        df[col] = df[col].fillna(df[col].mean())

    # fill missing categorical values with column mean
    for col in categorical_variables:
        df[col] = df[col].fillna(df[col].mode())

    # fill binary coded cols with column mean
    for col in list(config["cols_to_label_encode"].keys()):
        df[col] = df[col].fillna(df[col].mode()[0]).astype(int)

    print(f"Number of rows after preprocessing: {len(df)}")

    return df

if __name__ == "__main__":
    run()