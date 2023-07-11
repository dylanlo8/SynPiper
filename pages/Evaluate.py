import streamlit as st
import pandas as pd
from synthetic_evaluation import *
from utils import count_exact_match_rows
import numpy as np

df_train = st.file_uploader("Upload Real Training Data", type=["csv"])
df_syn = st.file_uploader("Upload Synthetic Data", type = ["csv"])
 
df_train = pd.read_csv(df_train)
df_syn = pd.read_csv(df_syn)

# Running of Streamlit App
st.title("Synthetic Data Quality Assurance Report")

# Categorical / Numerical Features selector
cat_cols = st.multiselect(label="Categorical Features", 
                            options=df_train.columns)

num_cols = st.multiselect(label = "Numerical Features",
    options=[col for col in df_train.columns if col not in cat_cols])

# Headline Metrics
st.subheader("Data Summary")
col1, col2, col3 = st.columns(3)
col1.metric(label = "Dataset Columns", value = len(df_train.columns))
col2.metric(label = "Training Samples", value = df_train.shape[0])
col3.metric(label = "Synthetic Samples", value = df_syn.shape[0])

#if 'time' in st.session_state.keys():
#   st.text(f"Total Time: {st.session_state['time']} seconds.")

st.subheader("Categorical Data Comparison")
df_tvd, plot = get_all_variational_differences(df_train, df_syn, cat_cols)
st.metric(label = "Average Similarity Score", value = f"{round(np.mean(df_tvd.iloc[:,1]) * 100, 1)}%")
st.plotly_chart(plot)
st.dataframe(
    df_tvd,
    hide_index= True,
    column_config = {
        "categorical_columns" : "Categorical Column Names",
        "tvd_scores" : "Similarity Score"
    }
)

st.subheader("Numerical Data Comparison")
df_ks, plot = get_all_ks_scores(df_train, df_syn, num_cols)
st.metric(label = "Average Similarity Score", value = f"{round(np.mean(df_ks.iloc[:,1]) * 100, 1)}%")
st.plotly_chart(plot)
st.dataframe(
    df_ks,
    hide_index= True,
    column_config = {
        "numerical_columns" : "Numerical Column Names",
        "ks_scores" : "Similarity Score"
    }
)

st.subheader("Pairwise Correlation Comparison")
st.pyplot(plot_corr_matrix(df_train[num_cols], df_syn[num_cols]))

st.subheader("Pairwise Mutual Information Score Comparison")
fig, mi_score, n_pass_threshold = plot_mi_matrix(df_train, df_syn)

st.metric(label = "Dataset Mutual Information Score",
        value = f"{mi_score}%")

st.metric(label = "Proportion of Relationships with negligible MI difference",
        value = f"{n_pass_threshold}%")

st.pyplot(fig)


# Privacy Metrics
st.subheader("Privacy Metrics")
st.metric(label = "Exact Row Match Privacy Score", 
        value = f"{count_exact_match_rows(df_train, df_syn)}%")

# List of Plots
st.subheader("Column Distribution Comparison")

# Plot individual distributions for each column
for col in df_train.columns:
    st.plotly_chart(plot_real_synthetic(df_train, df_syn, col))

st.caption("Please upload both datasets to proceed")



