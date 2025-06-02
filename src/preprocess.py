import pandas as pd
import numpy as np
from itertools import product
from pandas import Timestamp
import sys
from pathlib import Path
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_seq_items', None)
pd.set_option('display.width', None)
sys.path.append(str(Path("/Users/nk/Acedemic/Masters/KULEUVEN/Thesis/project_second/survival_project/src").resolve()))

# Import the function after adding the correct path
from functions import drop_columns_with_nan_greater_than_threshold, get_columns_with_nan_percentage, get_features_info



def load_data():
    raw_data = pd.read_csv("/Users/nk/Acedemic/Masters/KULEUVEN/Thesis/project_second/survival_project/raw_data 1.csv")
    clean_data = raw_data.copy()
    return clean_data

def prepare_column_names(clean_data):
    column_names_df = get_features_info(clean_data)
    # merging columns Before Yer and After Year Columns into one column (quarters)]
    column_names_df['quarter'] = column_names_df["Before Year"] + column_names_df["After Year"]
    # Fill NaN values with a default value (e.g., -1) before converting to integers
    column_names_df["Year"] = column_names_df["Year"].fillna(-1).astype(int)
    return column_names_df

def process_data(clean_data):
    clean_data["listing_date"] = pd.to_datetime(clean_data["First.listing.date"])
    clean_data["bankrupcty_date"] = pd.to_datetime(clean_data["risk.time"])
    clean_data["status"] = clean_data["risk.time"].apply(lambda x: 0 if pd.isna(x) else 1)
    clean_data["bankrupcty_date"] = clean_data["bankrupcty_date"].apply(lambda x: Timestamp("2023-12-31") if pd.isna(x) else x)
    clean_data["survival_time"] = (clean_data["bankrupcty_date"] - clean_data["listing_date"]).dt.days/365.25
    clean_data["censoring_time"] = (Timestamp("2023-12-31") - clean_data["listing_date"]).dt.days/365.25
    clean_data_copy = clean_data.copy()
    # Precompute first_quarter and first_year for all rows in clean_data
    clean_data_copy["first_year"] = clean_data_copy["listing_date"].apply(lambda x: x.year)
    clean_data_copy["last_year"] = clean_data_copy["bankrupcty_date"].apply(lambda x: x.year)
    clean_data_copy["first_quarter"] = clean_data_copy["listing_date"].apply(lambda x: x.quarter)
    clean_data_copy["last_quarter"] = clean_data_copy.apply(
        lambda row: (
            (((row["last_year"] - row["first_year"] - 1) * 4) + (5-row["first_quarter"]) + row["bankrupcty_date"].quarter)
            if (row["last_year"] != row["first_year"])
            else ((1-row["first_quarter"])+ row["bankrupcty_date"].quarter)
        ),
        axis=1
    )
    return clean_data_copy

def create_data(start_date,end_date, clean_data_copy,column_names_df):
    print(clean_data_copy.shape)
    clean_data_copy = clean_data_copy[(clean_data_copy["listing_date"] >= start_date) & (clean_data_copy["listing_date"] <= end_date)]
    print(clean_data_copy.shape)
    # Select the row(s) where survival_time is equal to its maximum value
    # last_quarter = clean_data_copy[clean_data_copy["survival_time"] == clean_data_copy["survival_time"].max()]["last_quarter"].values[0]
    df_final = pd.DataFrame()
    # add a column for Securities.code with out any values
    df_final["Securities.code"] = pd.Series(dtype='str')
    df_final["Securities.code"] = clean_data_copy["Securities.code"]
    list = column_names_df["Feature"].dropna().unique().tolist()
    for feature in list:
        print(feature)
        # # Predefine all columns for the feature
        # new_columns = [f"{feature}Q{i}" for i in range(1, last_quarter)]
        # for col in new_columns:
        #     df_temp[col] = pd.Series(dtype='float')  # Add only if the column doesn't already exist
        # Filter columns for the current feature
        columns_with_features = column_names_df[column_names_df["Feature"] == feature]
        rows = []
        # Iterate over rows in clean_data
        for _, row in clean_data_copy.iterrows():
            first_quarter = row["first_quarter"]
            first_year = row["first_year"]
            id = row["Securities.code"]
            
            # Find the index of the first year for the feature
            index_of_first_year = columns_with_features[columns_with_features["Year"] == first_year].index[0]
            print(index_of_first_year)
            # Slice the relevant columns from the row
            # relevant_columns = row.iloc[index_of_first_year+first_quarter-1 :-7]  # Adjust slicing as needed
            # df_temp.iloc[:,:len(relevant_columns)]= relevant_columns.values
            # df_temp["Securities.code"] = id
            new_row = {"Securities.code": id}
            # Assign the values to the corresponding columns in df
            # df.loc[df["Securities.code"] == id, df.columns[10:10 + len(relevant_columns)]] = relevant_columns.values
            
            # Slice the rows starting from the first year and quarter
            relevant_rows = columns_with_features.loc[index_of_first_year + first_quarter-1:, :]
            print(relevant_rows)
            # Assign values to the df DataFrame
            for index_int, (_, row1) in enumerate(relevant_rows.iterrows(), start=1):
                column_name = f"{row1['Feature']}Q{index_int}"
                new_row[column_name] = row[row1["Column Names"]]
                # Append the new row to df_temp
            rows.append(new_row)
            print(new_row)
        print(rows)
        df_temp =  pd.DataFrame(rows)
        df_final = df_final.merge(df_temp, on="Securities.code", how="left")
    return df_final
        # df_temp = pd.concat([df_temp, pd.DataFrame([new_row])], ignore_index=True)

if __name__ == "__main__":
    clean_data = load_data()
    column_names_df = prepare_column_names(clean_data)
    clean_data_copy = process_data(clean_data)
    start_date = Timestamp("1990-01-01")
    end_date = Timestamp("2023-12-31")
    df_final = create_data(start_date, end_date, clean_data_copy, column_names_df)
    print(df_final.shape)
    print(df_final.head())