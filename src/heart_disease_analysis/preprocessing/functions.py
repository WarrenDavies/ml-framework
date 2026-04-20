import csv
import pandas as pd


def remove_blank_lines(input_file, output_file):
    with open(input_file, "r") as infile, open(output_file, "w", newline="") as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        for row in reader:
            if any(field.strip() for field in row):
                writer.writerow(row)


def bool_to_int(df_input, old_col_name, new_col_name):
    df_output = df_input.copy()
    df_output[new_col_name] = df_output[old_col_name].astype(int)
    return df_output


def make_int_col_binary(df_input, old_col_name, new_col_name, threshold):
    df_output = df_input.copy()
    df_output[new_col_name] = (df_output[old_col_name] > threshold).astype(int)
    return df_output


def label_encode(df_input, old_col_name, new_col_name, new_value_map):
    df_output = df_input.copy()
    df_output[new_col_name] = df_output[old_col_name].map(new_value_map)
    return df_output


def get_one_hot_encoded_features_from_df(df, cols_to_one_hot_encode):
    one_hot_encoded_features = []
    for col in cols_to_one_hot_encode:
        new_columns = list(df.columns[df.columns.str.startswith(col + "_")])
        one_hot_encoded_features = one_hot_encoded_features + new_columns
    return one_hot_encoded_features


def get_one_hot_encoded_features_from_csv(csv_path, cols_to_one_hot_encode):
    df = pd.read_csv(csv_path)

    one_hot_encoded_features = get_one_hot_encoded_features_from_df(
        df, 
        cols_to_one_hot_encode
    )
    
    return one_hot_encoded_features