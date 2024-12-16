import pandas as pd


def process_input_data(df: pd.DataFrame):
    return df.to_dict(orient='records')
