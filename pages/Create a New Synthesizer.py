import streamlit as st
import os
import pandas as pd
from run import *
from utils import train_val_split
#streamlit run c:/Users/User/Desktop/SynPiper-master/about.py

if __name__ == '__main__':
    st.title("Create a New Synthesizer")

    st.subheader("File Upload")

    uploaded_data = st.file_uploader("Upload your file here...", type=["csv"])

    cwd = os.getcwd()
    workingpath = os.path.join(cwd, "workingfolder")
    os.makedirs(workingpath, exist_ok=True)

    # Creation of Filepaths
    synthetic_filepath = os.path.join(workingpath, "synthetic.csv")
    path_of_df_train = os.path.join(workingpath, "df_train.csv")
    path_of_df_val = os.path.join(workingpath, "df_val.csv")
    
    try: 
    # Navigates to Parent Directory
        uploaded_data = pd.read_csv(uploaded_data)
        
        st.subheader("Preview of Dataset")
        st.dataframe(uploaded_data.head())

        # Train - Holdout set Split
        df_train, df_val = train_val_split(uploaded_data, 
                                           uploaded_data.columns[-1], 
                                           ratio = 0.2) # 20% Validation set
        df_train.to_csv(path_or_buf = path_of_df_train, index = False)
        df_val.to_csv(path_or_buf = path_of_df_val, index = False)

        # Saves the respective filepaths into streamlit's session
        st.session_state['synthetic_filepath'] = synthetic_filepath 
        st.session_state['train_filepath'] = path_of_df_train
        st.session_state['val_filepath'] = path_of_df_val

        ### COLUMNS ###
        avail_cols = df_train.columns

        cat_cols = st.multiselect(label="Categorical Features", 
                                    options=avail_cols)

        num_cols = st.multiselect(
            label = "Numerical Features",
            options=[col for col in avail_cols if col not in cat_cols]
        )

        # Saves the list of categorical and numerical data into st session state
        st.session_state['cat_cols'] = cat_cols
        st.session_state['num_cols'] = num_cols

        all_cols = cat_cols + num_cols
        datatypes_lst = ["Categorical"] * len(cat_cols) + ["Numerical"] * len(num_cols)
        unique_vals_lst = [len(df_train[col].unique()) for col in all_cols]
        df_summary = pd.DataFrame(
            {
                "Column Names": all_cols,
                "Datatypes": datatypes_lst,
                "Number of Unique Values" : unique_vals_lst
            }
        )

        st.dataframe(df_summary)

        ### SYNTHESIZER ###
        st.subheader("Select Synthesizer")

        model_name = st.selectbox(
            label="synthesizer",
            options=["Differentially Private Synthesizer",
                        "CTGAN",
                        "TVAE"],
            label_visibility= "collapsed"
        )

        st.caption(f"{model_name} has been selected.")

        model_dict = {
            "Differentially Private Synthesizer": "dpsynthesizer",
            "CTGAN" : "ctgan",
            "TVAE" : "tvae"   
        } 
            
        synthesizer_name = model_dict[model_name]

        st.subheader("Choose Parameters")
        ready_to_train = False # When True: activate the Train Button

        ### DPSynthesizer
        if synthesizer_name == "dpsynthesizer":
            degree_of_bayesian_networks = st.number_input(
                label="Number of Bayesian Networks", min_value=2, max_value=10
            )
            network_expander = st.expander("See bayesian network parameter configuration")
            network_expander.write("""
                Advised number of networks to pick: 3 \n
                Picking a high number of networks could lead to a more overfitted synthetic data generator.
            """)

            epsilon = st.slider(label="Epsilon Value", min_value=0.0, max_value=100.0)
            epsilon_expander = st.expander("See epsilon parameter configuration")
            epsilon_expander.write("""
                To turn off Differential Privacy, pick 0. \n
                For more privatised synthetic data, pick a small epsilon value. \n
                For a less privatised synthetic data with higher utility, pick a larger epsilon value.
            """)
            ready_to_train = True

            params_required = {
                "categorical_attributes": {},
                "epsilon": epsilon,
                "degree_of_bayesian_network": degree_of_bayesian_networks,
            }

            # Inputting categorical attributes
            for col in cat_cols:
                params_required["categorical_attributes"][col] = True

        elif synthesizer_name == "ctgan" or synthesizer_name == "tvae":
            epochs = st.number_input(
                label="Number of Epochs", min_value = 300, max_value=1500
            )

            ready_to_train = True

            params_required = {
                'categorical_attributes' : cat_cols,
                'epochs' : epochs
            }

        # Implement other Synthesizers here if any.
        else:
            pass

        if ready_to_train:
            st.subheader("Training of Synthesizer")

            # Number of Rows input and Train Button
            st.caption("Number of Rows to Generate")
            
            col1, col2 = st.columns(2)
            with col1: # Nrow Input
                n_rows_input = st.number_input(
                    label="", 
                    min_value=1, 
                    max_value=100000,
                    label_visibility= "collapsed")

            with col2: # Train Button 
                if synthesizer_name == "dpsynthesizer":
                    if st.button(label = "Generate"): 
                        timer = Timer()
                        timer.start()
                        
                        # Runs generator and saves synthetic csv into cwd
                        run_dpsyn(params_required, 
                                    num_tuples_to_generate = n_rows_input, 
                                    data_path = path_of_df_train,
                                    synthetic_filepath= synthetic_filepath)
                        
                        time = timer.stop()
                        st.sesion_state['time'] = time
                        
                elif synthesizer_name == "ctgan":
                    if st.button(label = "Generate"): 
                        timer = Timer()
                        timer.start()
                        
                        # Runs generator and saves synthetic csv into cwd
                        run_ctgan(params_required, 
                                    num_tuples_to_generate = n_rows_input, 
                                    data_path = path_of_df_train,
                                    synthetic_filepath= synthetic_filepath)
                        
                        time = timer.stop()
                        st.session_state['time'] = time
                
                elif synthesizer_name == "tvae":
                    if st.button(label = "Generate"): 
                        timer = Timer()
                        timer.start()
                        
                        # Runs generator and saves synthetic csv into cwd
                        run_tvae(params_required, 
                                    num_tuples_to_generate = n_rows_input, 
                                    data_path = path_of_df_train,
                                    synthetic_filepath= synthetic_filepath)
                        
                        time = timer.stop()
                        st.session_state['time'] = time

    except:
        st.caption("Upload Real Data csv to proceed.")
