from Processor import SDVProcessor, DataSynthesizerProcessor
from SynPiper import SynPiper
import pandas as pd
import os
import streamlit as st
from synthetic_evaluation import *
from timer import Timer
"""
    These are runnable functions that will be executed when all 
    required parameters have been collected. 

    These functions are executed by the Generate Button.
"""
def run_dpsyn(params_required, num_tuples_to_generate, data_path, synthetic_filepath):
    piper = SynPiper(
        data_path, 
        param_dict = params_required, 
        synthesizer_name = "dpsynthesizer", 
        synthetic_filepath=synthetic_filepath
    )

    piper.generate_dpsynthesizer(num_tuples_to_generate= num_tuples_to_generate)


def run_ctgan(params_required, num_tuples_to_generate, data_path, synthetic_filepath):
    piper = SynPiper(
        data_path = data_path, 
        param_dict = params_required, 
        synthesizer_name = "ctgan", 
        synthetic_filepath= synthetic_filepath
    )
    
    piper.generate_sdv(num_tuples_to_generate = num_tuples_to_generate)

def run_tvae(params_required, num_tuples_to_generate, data_path, synthetic_filepath):
    piper = SynPiper(
        data_path = data_path, 
        param_dict = params_required, 
        synthesizer_name = "tvae", 
        synthetic_filepath= synthetic_filepath
    )
    
    piper.generate_sdv(num_tuples_to_generate = num_tuples_to_generate)
