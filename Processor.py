import os
from DataSynthesizer.DataDescriber import DataDescriber
from synthetic_evaluation import sdv_metadata_manual_processing
import pandas as pd

class SDVProcessor:
    """ Processes the training data used for synthetic data generation into an appropriate
    format for SDV (Synthetic Data Vault) Synthesizers (CTGAN and TVAE)

    Attributes
        params_dict (Dictionary): 
            categorical_attributes (List): List of categorical column names
            epochs : Number of epochs to run the model training on (min. >= 500)
        data_path (Path): Path of training data
    """
    
    def __init__(self, data_path, param_dict):
        params_required = ["categorical_attributes", 
                           "epochs"]
        self.data_path = data_path

        # Check for inappropriate parameters
        for param in param_dict.keys():
            if param not in params_required:
                raise ValueError("Unspecified Parameter")

        self.param_dict = param_dict

    def process(self):
        """ Processes the training data into an appropriate format for CTGAN / TVAE.
        """
        real_data = pd.read_csv(self.data_path)
        
        # Generates sdv metadata
        metadata = sdv_metadata_manual_processing(real_data, self.param_dict['categorical_attributes'])
        
        return metadata

class DataSynthesizerProcessor:
    """ Processes the training data used for synthetic data generation into an appropriate
    format for DataSynthesizer's Differentially Private Synthetic Data Generator (DP Synthesizer).

    Parameters: 
        1) categorical_attributes : List of categorical attributes in the dataset
        2) epsilon : Privacy Budget (integer). Lower epsilon means more privatised data (less resemblance). 
            0 to turn off differential privacy.
        3) degree_of_bayesian_network: (integer) Higher degree means a more complex Bayesian Network model which could lead to overfitting.
            Recommended value: 3
    """
    
    # Set description file directory
    description_file = os.path.join(os.getcwd(), "description.json")

    def __init__(self, data_path, param_dict):
        params_required = [
            "categorical_attributes",
            "epsilon",
            "degree_of_bayesian_network",
        ]

        # Check for inappropriate parameters
        for param in param_dict.keys():
            if param not in params_required:
                raise ValueError(
                    "Unspecified Parameter, Please follow the correct naming convention"
                )

        self.data_path = data_path
        self.param_dict = param_dict

    def process(self):
        describer = DataDescriber()
        cat_cols = self.param_dict["categorical_attributes"]
        cat_dict = {}

        for cat in cat_cols:
            cat_dict[cat] = True

        describer.describe_dataset_in_correlated_attribute_mode(
            dataset_file = self.data_path,
            epsilon=self.param_dict["epsilon"],
            k=self.param_dict["degree_of_bayesian_network"],
            attribute_to_is_categorical=cat_dict,
        )

        print("Saving Dataset Description File")
        describer.save_dataset_description_to_file(self.description_file)
        return self.description_file