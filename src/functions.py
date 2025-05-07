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
        .iloc[6:-6]
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
        .iloc[6:-6] 
        .apply(split_by_year)
        .apply(pd.Series) 
    )

    column_names_df[["Before Year", "Year", "After Year"]] = column_names_df[["Before Year", "Year", "After Year"]].reindex(column_names_df.index)
    return column_names_df