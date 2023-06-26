import streamlit as st
import pandas as pd

def check_column_type(column):    
    # Check if the column data type is continuous
    if pd.api.types.is_numeric_dtype(column):
        return 'Continuous'

    # Check if the column data type is categorical
    unique_values = column.unique()
    if len(unique_values) <= (0.4 * len(column)) and pd.api.types.is_object_dtype(column):
        return 'Categorical'

    # Check if the column data type is datetime
    if pd.api.types.is_datetime64_dtype(column):
        return 'Datetime'
    
    # If none of the above, consider it as other/unknown type
    return 'Other'