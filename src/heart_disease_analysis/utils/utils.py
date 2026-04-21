import yaml
import datetime
from pathlib import Path

import pandas as pd


def print_header(
    text,
    border_size = 4,
    character="#",
    blank_lines_before=2,
    blank_lines_after=1
):
    text_length = len(text)
    border = character * (text_length + 2 + (border_size * 2))
    middle = character * border_size + f' {text} ' + character * border_size
    if blank_lines_before > 0:
        print("\n" * blank_lines_before, end="")
    print(border)
    print(middle.center(len(border)))
    print(border)
    if blank_lines_after > 0:
        print("\n" * blank_lines_after, end="")


def load_yaml(path):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def create_outputs(config, results):
    ts = datetime.datetime.now()
    title = ts.strftime("Model result from %y-%m-%d %H:%M:%S")
    file_name = ts.strftime("%y%m%d_%H%M%S")
    report_output_path = Path(config["outputs"]["report_path"]) / (file_name + ".md")
    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_output_path = Path(config["outputs"]["config_dump_path"]) / (file_name + ".yaml")
    config_output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_output_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)

    with open(report_output_path, 'a') as file:
        file.write(f"# {title} \n\n")

        file.write(f"## Cleaning functions \n\n")
        for cleaning_function in config["cleaning"]["functions"]:
            file.write(f"* {cleaning_function["name"]}\n")
            if cleaning_function["params"]:
                for param in cleaning_function["params"]:
                    param_value = cleaning_function["params"][param]
                    file.write(f"  * **{param}**: {param_value}\n")
        file.write(f"\n\n")


        file.write(f"## Preprocessing functions \n\n")
        for preprocessing_function in config["preprocessing"]["functions"]:
            file.write(f"* {preprocessing_function["name"]}\n")
            if preprocessing_function["params"]:
                for param in preprocessing_function["params"]:
                    param_value = preprocessing_function["params"][param]
                    file.write(f"  * **{param}**: {param_value}\n")
        file.write(f"\n\n")


        for i, result in enumerate(results):
            file.write(f"## {config["modelling"]["functions"][i]["name"]} \n\n")
            print(result)
            if "classification_report" in result.keys():
                file.write(f"### Classification Report \n\n")
                df_report = pd.DataFrame(result["classification_report"]).transpose()
                md_table = df_report.to_markdown()
                file.write(md_table)
                file.write("* **Precision:** Of the cases predicted as a given class (0 or 1), the fraction that are actually that class (fewer false positives).\n")
                file.write("* **Recall:** Of the cases that truly belong to a given class (0 or 1), the fraction the model correctly identifies (fewer false negatives).\n")
                file.write("* **F1-score:** The harmonic mean of precision and recall for a class, balancing false positives and false negatives.\n")
                file.write("* **Support:** The number of true samples for each class in the evaluation set.\n")
                file.write("* **Accuracy:** The overall fraction of predictions (across both classes) that are correct.\n")
                file.write("* **Macro avg:** The unweighted average of the per-class scores, treating classes 0 and 1 equally regardless of size.\n")
                file.write("* **Weighted avg:** The average of the per-class scores weighted by support, so larger classes contribute more.\n")

                file.write(f"\n\n")


            if "positive_class_results" in result:
                file.write(f"### Validation Results \n\n")
                for positive_class_result in result["positive_class_results"]:
                    metric = positive_class_result
                    mean = result["positive_class_results"][positive_class_result]["mean"]
                    std = result["positive_class_results"][positive_class_result]["std"]
                    file.write(f"* **{metric}**: {mean} ± {std}\n")
                file.write(f"\n\n")


            if "feature_importance" in result:
                file.write(f"### Feature importance \n\n")
                file.write(result["feature_importance"].to_markdown(index=False))
                file.write('\n\n')


            if "chart_paths" in result:
                file.write(f"### Charts \n\n")
                for chart_path in result["chart_paths"]:
                    file.write(f'!["Chart"]({chart_path.replace("outputs", "../")})\n\n')
                file.write(f"\n\n")


