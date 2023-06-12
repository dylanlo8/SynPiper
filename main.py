from Processor import SDVProcessor, DataSynthesizerProcessor
from SynPiper import SynPiper
import pandas as pd
import os
import streamlit as st
from synthetic_evaluation import *

#streamlit run c:/Users/User/Desktop/SynPiper/main.py

if __name__ == '__main__':
    data_path = os.path.join(os.getcwd(), "datasets", "heartprocessed.csv")
    df_real = pd.read_csv(data_path)

    # User Input 1: Name of Synthesizer
    synthesizer_name = "dpsynthesizer"

    # User Input 2: Parameter Dictionary
    categorical_data = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
    numerical_data = [col for col in df_real.columns if col not in categorical_data]

    params_required = {"categorical_attributes" : 
                        {'sex': True, 'cp': True, 
                            'fbs': True, 'restecg': True, 
                            'exang': True, 'slope': True, 
                            'ca' : True, 'thal' : True, 'target' : True
                        },
                    "epsilon" : 0,
                    "degree_of_bayesian_network" : 3}

    piper = SynPiper(data_path, param_dict= params_required, synthesizer_name = synthesizer_name)

    # User Input 3: Number of Rows to be generated in Synthetic Dataset
    #piper.generate_dpsynthesizer(num_tuples_to_generate= 297)

    df_real = pd.read_csv(piper.data_path)
    df_syn = pd.read_csv(piper.synthetic_filepath)

    # Running of Streamlit App
    st.title('Synthetic Data Quality Report')

    # List of Plots
    st.subheader("Distribution Plots")

    st.subheader("Total Variational Difference (TVD) Analysis")
    st.pyplot(get_all_variational_differences(df_real, df_syn, categorical_data))

    st.subheader("KS-Statistic Analysis")
    st.pyplot(get_all_ks_scores(df_real, df_syn, numerical_data))

    st.subheader("Pairwise Correlation Comparison")
    st.pyplot(plot_corr_matrix(df_real, df_syn))

    st.subheader("Pairwise Mutual Information Score Comparison")
    st.pyplot(plot_mi_matrix(df_real, df_syn))
