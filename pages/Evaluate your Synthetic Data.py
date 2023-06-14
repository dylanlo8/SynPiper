import streamlit as st
from synthetic_evaluation import *

### Retrieving information from Session State
synthetic_filepath = st.session_state['synthetic_filepath']
real_filepath = st.session_state['real_filepath']
cat_cols = st.session_state['cat_cols']
num_cols = st.session_state['num_cols']


df_real = pd.read_csv(real_filepath, index_col = 0)
df_syn = pd.read_csv(synthetic_filepath, index_col = 0)

# Running of Streamlit App
st.title("Synthetic Data Quality Report")

st.text(f"The generator took {st.session_state['time']} seconds.")

st.subheader("Total Variational Difference (TVD) Analysis")
df_tvd, plot = get_all_variational_differences(df_real, df_syn, cat_cols)
st.plotly_chart(plot)
st.dataframe(
    df_tvd,
    hide_index= True
)

st.subheader("KS-Statistic Analysis")
df_ks, plot = get_all_ks_scores(df_real, df_syn, num_cols)
st.plotly_chart(plot)
st.dataframe(
    df_ks,
    hide_index= True
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