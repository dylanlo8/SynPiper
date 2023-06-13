import streamlit as st
import os
import pandas as pd

st.title("Create a New Synthesizer")

st.subheader("File Upload")

uploaded_data = st.file_uploader("Upload your file here...", type=['csv'])

try:
    # Navigates to Parent Directory
    uploaded_data = pd.read_csv(uploaded_data)
    cwd = os.getcwd()
    workingpath = os.path.join(cwd, "workingfolder")
    os.makedirs(workingpath, exist_ok=True)
    path_of_df_real = os.path.join(workingpath, "df_real.csv")
    uploaded_data.to_csv(path_or_buf = path_of_df_real)

    avail_cols = uploaded_data.columns

    cat_cols = st.multiselect(label = "Pick categorical", options = avail_cols)
    num_cols = st.multiselect(label = "Pick numerical", options = [col for col in avail_cols if col not in cat_cols])

    st.subheader("Select Synthesizer")

    model_name = st.selectbox(label = "synthesizer", 
                            label_visibility= 'collapsed', 
                            options = ["Differentially Private Synthesizer"])

    st.text(model_name)

    model_dict = {"Differentially Private Synthesizer" : "dpsynthesizer"}

    st.subheader("Choose Parameters")

except:
    st.text("Please upload a datafile to proceed")

#display_params(model_dict[model_name])
