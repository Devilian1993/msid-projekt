import numpy as np
import pandas as pd
from typing import NamedTuple, cast
from sklearn.model_selection import train_test_split

class DiabetesData(NamedTuple):
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

def get_raw_data():
    # noinspection PyArgumentList
    df = cast(pd.DataFrame, pd.read_csv('../data/diabetes.csv'))

    # replace 0 in every column (excl. pregnancies and outcome) with median
    cols_to_fix = df.columns[1:-1]
    df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)
    df[cols_to_fix] = df[cols_to_fix].fillna(df[cols_to_fix].median())

    x = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    return DiabetesData(x_train, x_test, y_train, y_test)

if __name__ == "__main__":
    data = get_raw_data()
    data.x_train.info()
