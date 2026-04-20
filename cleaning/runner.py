import pandas as pd
from cleaning.registry import REGISTRY


class Runner():

    def __init__(self, config):
        self.config = config
        self.df_input = pd.read_csv(self.config["input_path"])
    

    def run(self):
        df_output = self.df_input.copy()    

        for function in self.config["functions"]:
            func = REGISTRY[function["name"]]
            df_output = func(df_output, **function["params"])

        self.df_output = df_output.copy()
        self.df_output.to_csv(self.config["output_path"])

        return self.df_output
