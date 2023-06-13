from Processor import SDVProcessor, DataSynthesizerProcessor
from SynPiper import SynPiper
import pandas as pd
import os
import streamlit as st
from synthetic_evaluation import *
from timer import Timer

def run_dpsyn(params_required, num_tuples_to_generate, data_path):
    synthetic_filepath = os.path.join(os.getcwd(), "synthetic.csv")
    piper = SynPiper(
        data_path, param_dict = params_required, synthesizer_name = "dpsynthesizer"
    )

    # User Input 3: Number of Rows to be generated in Synthetic Dataset
    piper.generate_dpsynthesizer(num_tuples_to_generate= num_tuples_to_generate)

    