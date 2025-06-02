import pandas as pd
import re

def drop_columns_with_nan_greater_than_threshold(df, threshold):
    current_columns = df.columns
    current_columns_df = pd.DataFrame(current_columns, columns=["current_columns"])
    current_columns_df["percentage_of_nans"] = current_columns_df["current_columns"].apply(lambda x: df[x].isna().sum() / df.shape[0] * 100)
    return df.drop(columns=current_columns_df[current_columns_df["percentage_of_nans"] > threshold]["current_columns"].tolist())

def get_columns_with_nan_percentage(df, threshold):
    current_columns = df.columns
    current_columns_df = pd.DataFrame(current_columns, columns=["current_columns"])
    current_columns_df["percentage_of_nans"] = current_columns_df["current_columns"].apply(lambda x: df[x].isna().sum() / df.shape[0] * 100)
    no_rows,no_cols =df.shape
    count = (current_columns_df.iloc[6:-6,1] >= threshold).sum()
    print(f"data with nans percentage less than {threshold}: (",no_rows,",", no_cols-count, ")")


def get_features_info(clean_data: pd.DataFrame) -> pd.DataFrame:
    columns = clean_data.columns
    column_names_df = pd.DataFrame(columns, columns=["Column Names"])

    def split_case_insensitive(value):
        match = re.split(r"\[Reporting\.Period\]", value, flags=re.IGNORECASE)
        return match if len(match) == 2 else [value, None]
    column_names_df[["Feature", "Suffix"]] = (
        column_names_df["Column Names"]
        .iloc[6:]
        .apply(split_case_insensitive)
        .apply(pd.Series)
    )
    column_names_df[["Feature", "Suffix"]] = column_names_df[["Feature", "Suffix"]].reindex(column_names_df.index)

    def split_by_year(value):
        match = re.search(r"(19|20)\d{2}", value) 
        if match:
            year = match.group(0) 
            before_year = value[:match.start()].strip()  
            after_year = value[match.end():].strip() 
            return [before_year, year, after_year]
        else:
            return [value, None, None]  

    column_names_df[["Before Year", "Year", "After Year"]] = (
        column_names_df["Suffix"]
        .iloc[6:] 
        .apply(split_by_year)
        .apply(pd.Series) 
    )

    column_names_df[["Before Year", "Year", "After Year"]] = column_names_df[["Before Year", "Year", "After Year"]].reindex(column_names_df.index)
    return column_names_df

def create_data(start_date,end_date, clean_data_copy,column_names_df):
    clean_data_copy = clean_data_copy[(clean_data_copy["listing_date"] >= start_date) & (clean_data_copy["listing_date"] <= end_date)]
    # Select the row(s) where survival_time is equal to its maximum value
    df_final = pd.DataFrame()
    # add a column for Securities.code with out any values
    df_final = clean_data_copy[["Securities.code","listing_date", "bankrupcty_date", "status", "survival_time", "censoring_time","first_year","last_year","first_quarter","last_quarter"]]
    list = column_names_df["Feature"].dropna().unique().tolist()
    for feature in list:
        # # Predefine all columns for the feature
        columns_with_features = column_names_df[column_names_df["Feature"] == feature]
        rows = []
        # Iterate over rows in clean_data
        for _, row in clean_data_copy.iterrows():
            first_quarter = row["first_quarter"]
            first_year = row["first_year"]
            id = row["Securities.code"]
            
            # Find the index of the first year for the feature
            index_of_first_year = columns_with_features[columns_with_features["Year"] == first_year].index[0]
            # Slice the relevant columns from the row
            new_row = {"Securities.code": id}
            # Assign the values to the corresponding columns in df
            # Slice the rows starting from the first year and quarter
            relevant_rows = columns_with_features.loc[index_of_first_year + first_quarter-1:, :]
            # Assign values to the df DataFrame
            for index_int, (_, row1) in enumerate(relevant_rows.iterrows(), start=1):
                column_name = f"{row1['Feature']}_Q{index_int}"
                new_row[column_name] = row[row1["Column Names"]]
                # Append the new row to df_temp
            rows.append(new_row)
        df_temp =  pd.DataFrame(rows)
        df_final = df_final.merge(df_temp, on="Securities.code", how="left")
    return df_final


def get_features_info_with_quarters(clean_data: pd.DataFrame) -> pd.DataFrame:
    columns = clean_data.columns
    column_names_df = pd.DataFrame(columns, columns=["Column Names"])

    def split_quarter(value):
        match = re.split(r"_Q", value)
        return match if len(match) == 2 else [value, None]
    column_names_df[["Feature", "Quarter"]] = (
        column_names_df["Column Names"]
        .iloc[10:]
        .apply(split_quarter)
        .apply(pd.Series)
    )
    column_names_df[["Feature", "Quarter"]] = column_names_df[["Feature", "Quarter"]].reindex(column_names_df.index)

    return column_names_df


def dynamic_columns_func(row,column_names_quarters):
    last_quarter = row["last_quarter"]
    columns = column_names_quarters[column_names_quarters["Quarter"]<= last_quarter]
    return columns["Column Names"].tolist()

def calculate_row_nan_percentage(df):
    """
    Calculate the percentage of NaN values for each row, considering only dynamic columns.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - dynamic_columns_func (function): A function that takes a row and returns a list of column names to consider.

    Returns:
    - pd.Series: A Series with the percentage of NaN values for each row.
    """
    nan_percentages = []
    column_names_quarters = get_features_info_with_quarters(df)
    column_names_quarters["Quarter"] = column_names_quarters["Quarter"].fillna(999999999).astype(int)

    for index, row in df.iterrows():
        # Get the dynamic columns for this row
        columns_to_consider = dynamic_columns_func(row,column_names_quarters)

        # Calculate the percentage of NaN values for the selected columns
        if columns_to_consider:
            nan_count = row[columns_to_consider].isna().sum()
            total_count = len(columns_to_consider)
            nan_percentage = (nan_count / total_count) * 100
        else:
            nan_percentage = 0  # If no columns are selected, set NaN percentage to 0

        nan_percentages.append(nan_percentage)

    df["Row_NaN_Percentage"] = pd.Series(nan_percentages, index=df.index)

    return df

def reset_data(df):
    """
    Calculate the percentage of NaN values for each row, considering only dynamic columns.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - dynamic_columns_func (function): A function that takes a row and returns a list of column names to consider.

    Returns:
    - pd.Series: A Series with the percentage of NaN values for each row.
    """
    column_names_quarters = get_features_info_with_quarters(df)
    column_names_quarters["Quarter"] = column_names_quarters["Quarter"].fillna(0).astype(int)

    for index, row in df.iterrows():
        # Get the dynamic columns for this row
        columns = column_names_quarters[column_names_quarters["Quarter"]> row["last_quarter"]]
        columns_to_consider = columns["Column Names"].tolist()
        # Calculate the percentage of NaN values for the selected columns
        if columns_to_consider:
            df.loc[index, columns_to_consider] = None
    return df