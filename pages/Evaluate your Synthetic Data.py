import streamlit as st
from synthetic_evaluation import *

### Retrieving information from Session State
synthetic_filepath = st.session_state['synthetic_filepath']
train_filepath = st.session_state['train_filepath']
val_filepath = st.session_state['val_filepath']

cat_cols = st.session_state['cat_cols']
num_cols = st.session_state['num_cols']

df_train = pd.read_csv(train_filepath)
df_syn = pd.read_csv(synthetic_filepath)

# Running of Streamlit App
st.title("Synthetic Data Quality Assurance Report")

# Headline Metrics
st.subheader("Data Summary")
col1, col2, col3 = st.columns(3)
col1.metric(label = "No. of columns", value = len(df_train.columns))
col2.metric(label = "No. of rows in Real Data", value = df_train.shape[0])
col3.metric(label = "No. of rows in Synthetic Data", value = df_syn.shape[0])

if 'time' in st.session_state.keys():
    st.text(f"The generator took {st.session_state['time']} seconds.")

st.subheader("Categorical Data Comparison")
df_tvd, plot = get_all_variational_differences(df_train, df_syn, cat_cols)
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
st.plotly_chart(plot)
st.dataframe(
    df_ks,
    hide_index= True
)

st.subheader("Pairwise Correlation Comparison")
st.pyplot(plot_corr_matrix(df_train[num_cols], df_syn[num_cols]))

st.subheader("Pairwise Mutual Information Score Comparison")
st.pyplot(plot_mi_matrix(df_train, df_syn))

# List of Plots
st.subheader("Column Distribution Comparison")

# Plot individual distributions for each column
for col in df_train.columns:
    st.session_state[col] = plot_real_synthetic(df_train, df_syn, col)

selected_col = st.selectbox(label = "Select Column",
             options = df_train.columns)

st.plotly_chart(plot_real_synthetic(df_train, df_syn, selected_col))

