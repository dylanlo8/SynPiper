import streamlit as st
import os
import pandas as pd
from run import *
#streamlit run c:/Users/User/Desktop/SynPiper/about.py
st.title("Create a New Synthesizer")

st.subheader("File Upload")

uploaded_data = st.file_uploader("Upload your file here...", type=["csv"])

try:
    # Navigates to Parent Directory
    uploaded_data = pd.read_csv(uploaded_data)
    cwd = os.getcwd()
    workingpath = os.path.join(cwd, "workingfolder")
    os.makedirs(
        workingpath, exist_ok=True
    )  # create a workingpath directory, if not already existed
    path_of_df_real = os.path.join(workingpath, "df_real.csv")
    uploaded_data.to_csv(path_or_buf=path_of_df_real)

    avail_cols = uploaded_data.columns

    cat_cols = st.multiselect(label="Pick categorical", options=avail_cols)
    num_cols = st.multiselect(
        label="Pick numerical",
        options=[col for col in avail_cols if col not in cat_cols],
    )

    st.subheader("Select Synthesizer")

    model_name = st.selectbox(
        label="synthesizer",
        label_visibility="collapsed",
        options=["Differentially Private Synthesizer"],
    )

    st.caption(f"{model_name} has been selected.")

    model_dict = {
        "Differentially Private Synthesizer": "dpsynthesizer"
    }  # Include other synthesizers in future
    synthesizer_name = model_dict[model_name]

    st.subheader("Choose Parameters")
    ready_to_train = False

    if synthesizer_name == "dpsynthesizer":
        # st.text("Recommended bayesian network value: 3")
        degree_of_bayesian_networks = st.number_input(
            label="Number of Bayesian Networks", min_value=2, max_value=10
        )

        # st.text("For more privatised data: Pick a lower epsilon value")
        epsilon = st.slider(label="Epsilon Value", min_value=0.0, max_value=100.0)
        st.caption("Pick 0 to disable Differential Privacy")
        ready_to_train = True

        params_required = {
            "categorical_attributes": {},
            "epsilon": epsilon,
            "degree_of_bayesian_network": degree_of_bayesian_networks,
        }

        # Inputting categorical attributes
        for col in cat_cols:
            params_required["categorical_attributes"][col] = True

    # MARK: Implement other synthesizers here
    else:
        pass
    
    if ready_to_train:
        st.subheader("Training of Synthesizer")

        # Number of Rows input and Train Button
        st.caption("Number of Rows to Generate")
        col1, col2 = st.columns(2)
        with col1:
            n_rows_input = st.number_input(
                label="nrows", min_value=1, max_value=100000, label_visibility="collapsed"
            )

        with col2:
            if synthesizer_name == "dpsynthesizer":
                st.button(label="Train")


    
        
except:
    st.text("Please upload a datafile to proceed")

# display_params(model_dict[model_name])
