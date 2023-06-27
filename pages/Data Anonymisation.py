import streamlit as st
import pandas as pd
from anonymizer.auto_detect import *
import os
import sys


# Adding parent directory into working directory for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(1, parent_dir)


st.title("Data Anonymization")

st.subheader("File Upload")

try: 
    uploaded_data = st.file_uploader("Upload your data here", type=["csv"])
    df = pd.read_csv(uploaded_data)
    cols = df.columns
    st.dataframe(df.head(10),
                 hide_index=True) # Show preview of DataFrame

    st.subheader('Data Tagging')
    data_types = ["Continous", "Categorical", "Datetime"]
    information_types = ["NRIC", "Email", "Others"]
    sensitivity_types = ["Direct Identifier, Indirect Identifier, Sensitive, Non-Sensitive"]

    # Inference and Pre-selection
    inferred_dtypes = cols.map(lambda col : check_column_type(df[col])) #datatypes inferred
    preselected_information_types = cols.map(lambda row : "Others") # Others preselected
    preselected_sensitivity_types = cols.map(lambda row : "Non-Sensitive") # Non-sensitive preselected

    initial_tagging_df = pd.DataFrame({'Column Names': cols, 
                               'Datatype' : inferred_dtypes,
                               'Information Type' : preselected_information_types,
                               'Sensitivity Type' : preselected_sensitivity_types
                               })
    
    # Select Box Table for Data Tagging
    editted_df = st.data_editor(initial_tagging_df,
                                hide_index = True,
                                column_config = {
                                    "Datatype" : st.column_config.SelectboxColumn(
                                        "Pick Datatype",
                                        help = "Select the datatype of the Column",
                                        width = "medium",
                                        required = True,
                                        options = data_types
                                    ),
                                    'Information Type' : st.column_config.SelectboxColumn(
                                        "Pick Information Type",
                                        help = "Select the Information Type of the column",
                                        width = 'medium',
                                        required = True,
                                        options = information_types
                                    ),
                                    'Sensitivity Type' : st.column_config.SelectboxColumn(
                                        'Pick Sensitivity Type',
                                        help = "Select the Sensitivity Type of the column",
                                        width = 'medium',
                                        required = True,
                                        options = sensitivity_types
                                    )
                                }
                            )

    st.subheader('Data Transformation')
    col1,col2 = st.columns([1,3]) # 0.25 / 0.75 ratio

    with col1:
        st.selectbox(label = 'Select Column Name',
                     options = df.columns)
        
    # with col2:
        # Input bar for transformation options

        # Information Type


        # Sensitivity Type



        # Before Dataset


        # After Dataset
        
   





except ValueError:
    st.text("Please upload data to continue!")



