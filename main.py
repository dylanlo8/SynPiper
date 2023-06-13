from Processor import SDVProcessor, DataSynthesizerProcessor
from SynPiper import SynPiper
import pandas as pd
import os
import streamlit as st
from synthetic_evaluation import *
from timer import Timer

# streamlit run c:/Users/User/Desktop/SynPiper/main.py

if __name__ == "__main__":
    timer = Timer()
    timer.start()
    data_path = os.path.join(os.getcwd(), "datasets", "heartprocessed.csv")
    df_real = pd.read_csv(data_path).drop("Unnamed: 0", axis=1)

    # User Input 1: Name of Synthesizer
    synthesizer_name = "dpsynthesizer"

    # User Input 2: Parameter Dictionary
    categorical_data = [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
        "target",
    ]
    numerical_data = [col for col in df_real.columns if col not in categorical_data]

    params_required = {
        "categorical_attributes": {
            "sex": True,
            "cp": True,
            "fbs": True,
            "restecg": True,
            "exang": True,
            "slope": True,
            "ca": True,
            "thal": True,
            "target": True,
        },
        "epsilon": 80,
        "degree_of_bayesian_network": 3,
    }

    piper = SynPiper(
        data_path, param_dict = params_required, synthesizer_name = synthesizer_name
    )

    # User Input 3: Number of Rows to be generated in Synthetic Dataset
    piper.generate_dpsynthesizer(num_tuples_to_generate= len(df_real))

    df_syn = pd.read_csv(piper.synthetic_filepath).drop("Unnamed: 0", axis=1)

    # Running of Streamlit App
    st.title("Synthetic Data Quality Report")

    st.text(f"Total time taken {timer.stop()}")

    st.subheader("Total Variational Difference (TVD) Analysis")
    df_tvd, plot = get_all_variational_differences(df_real, df_syn, categorical_data)
    st.plotly_chart(plot)
    st.dataframe(
        df_tvd,
        hide_index=True,
        column_config = {
            "categorical_columns": "Categorical Columns",
            "tvd_scores": "TVD Scores (0 to 1)",
        },
    )

    st.subheader("KS-Statistic Analysis")
    df_ks, plot = get_all_ks_scores(df_real, df_syn, numerical_data)
    st.plotly_chart(plot, hide_index=True)
    st.dataframe(
        df_ks,
        hide_index = True,
        column_config={
            "numerical_columns": "Numerical Columns",
            "ks_scores": "KS-Scores (0 to 1)",
        },
    )

    st.subheader("Pairwise Correlation Comparison")
    st.pyplot(plot_corr_matrix(df_real, df_syn))

    st.subheader("Pairwise Mutual Information Score Comparison")
    st.pyplot(plot_mi_matrix(df_real, df_syn))

    # List of Plots
    st.subheader("Column Specific Distribution Comparison")
    for col in df_real.columns:
        st.subheader(f"{col} Distribution")
        st.plotly_chart(plot_real_synthetic(df_real, df_syn, col))