import pandas as pd

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)

    # Drop missing values
    df.dropna(inplace=True)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    return df
