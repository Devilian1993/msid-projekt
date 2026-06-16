import numpy as np
import pandas as pd
from typing import NamedTuple, cast


class DiabetesData(NamedTuple):
    x: pd.DataFrame
    y: pd.Series

# this one is the raw data frame with null values present
def get_raw_data():
    # noinspection PyArgumentList
    df = cast(pd.DataFrame, pd.read_csv('../data/diabetes.csv'))
    cols_to_fix = df.columns[1:-1]
    df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)
    # print(df[cols_to_fix].isnull().sum())
    # print(df.shape)
    # 768 total rows, data not fully presented in half of these!

    return df


# this one gets rid of all rows with null values
def clean_df_null(df):
    df = df.dropna()
    # print(df.shape)
    # dropping rows with missing values leaves us with 392 rows - good enough for a sample

    return split_data(df)


# this one leaves the median in place of null values
def clean_df_median(df):
    # replace 0 in every column (excl. pregnancies and outcome) with median
    cols_to_fix = df.columns[1:-1]
    df[cols_to_fix] = df[cols_to_fix].fillna(df[cols_to_fix].median())

    return split_data(df)


def split_data(df):
    x = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    return DiabetesData(x, y)


if __name__ == "__main__":
    df_raw = get_raw_data()
    # df_raw.info() # overall info for base raw data frame

    df_with_null = split_data(df_raw.copy()) # subset of raw data that swaps 0 to nulls
    df_no_null = clean_df_null(df_raw.copy()) # subset of raw data that drops rows with missing values
    df_fill_median = clean_df_median(df_raw.copy()) # subset of raw data that swaps nulls to median of given column
    # 3 different samples (null values, no null values, null converted to median values) to run tests on

    print("df_with_null:")
    print(df_with_null.x.shape) # this one might not work with logistic regression
    # df_with_null.x_train.info()
    # print(df_with_null.x_test.Insulin.describe()) # insulin contains the most missing values

    print("df_no_null:")
    print(df_no_null.x.shape) # subset is smaller, DiabetesData test data will differ!
    # df_no_null.x_train.info()
    # print(df_no_null.x_test.Insulin.describe())

    print("df_fill_median:")
    print(df_fill_median.x.shape)
    # df_fill_median.x_train.info()
    # print(df_fill_median.x_test.Insulin.describe())

    print(df_raw.corr())
    # correlation matrix - might take into consideration during defining fuzzy logic inputs
    # highest correlation with the outcome for glucose and bmi, later age, pregnancies, skin thickness and insulin

