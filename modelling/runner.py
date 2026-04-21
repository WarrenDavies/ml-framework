import pandas as pd
from modelling.registry import REGISTRY
from sklearn.model_selection import train_test_split

class Runner():

    def __init__(self, config, run_ts, features, target):
        self.config = config
        self.run_ts = run_ts
        self.features = features
        print(features)
        self.target = target
        self.df_input = pd.read_csv(self.config["input_path"])
        self.create_test_train_split()
        self.results = []


    def create_test_train_split(self):
        X = self.df_input[self.features]
        y = self.df_input[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, 
            y, 
            test_size=self.config["test_size"], 
            random_state=self.config["random_state"]
        )


    def get_x_and_y_datasets(self): 
        if self.config["mode"].lower() in ["train", "cv"]:
            return self.X_train, self.y_train

        if self.config["mode"].lower() in ["test"]:
            return self.X_test, self.y_test


    def run(self):
        x_values, y_values = self.get_x_and_y_datasets()

        for function in self.config["functions"]:
            func = REGISTRY[function["name"]]
            results = func(
                self.run_ts,
                x_values,
                y_values,
                self.X_train.columns,
                **function["params"]
            )
            self.results.append(results)

        return self.results
