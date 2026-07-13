import pandas as pd

class CountNulls(BaseCheck):
    
    def __init__(self, df, cols_to_check):
        super().__init__()
        self.output = self.run_check(df, cols_to_check)


    @staticmethod        
    def run_check(df, cols):
        df_cols_selected = df.copy()
        df_cols_selected = df_cols_selected[cols]

        df_null_counts = pd.DataFrame({
            "nulls": df[cols].isnull().sum(),
            "non_nulls": df[cols].notnull().sum(),
            "proportion_null": df[cols].isnull().mean()
        })
        df_null_counts = df_null_counts.reset_index()
        df_null_counts.columns = ["col_name", "nulls", "non_nulls", "proportion_null"]

        return df_null_counts
