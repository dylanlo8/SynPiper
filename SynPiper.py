from Processor import SDVProcessor, DataSynthesizerProcessor
from DataSynthesizer.DataGenerator import DataGenerator
import os
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
import pandas as pd

class SynPiper:
    """
    Attributes:
        synthesizer_name: Name of Synthesizer
        data_path: File Path of where the Real Data csv is found
        synthetic_filepath: File Path where Synthetic Data csv will be stored
        param_dict: Dictionary of parameters required for synthesizer
    """

    # Initialise appropriate Pre-Processors, check for correct param_dict.
    def __init__(self, data_path, synthesizer_name, param_dict, synthetic_filepath):
        # SDV Pre-Processor
        if synthesizer_name == "ctgan" or synthesizer_name == "tvae":
            print("Initialising SDV Processor")
            self.processor = SDVProcessor(data_path, param_dict)

        # DataSynthesizer Pre-processor
        elif synthesizer_name == "dpsynthesizer":
            print("Initialising DataSynthesizer Processor")
            self.processor = DataSynthesizerProcessor(data_path, param_dict)

        # Unspecified Synthesizer Name Error
        else:
            raise ValueError("Unspecified Synthesizer Name inputted.")

        self.synthesizer_name = synthesizer_name
        self.data_path = data_path
        self.synthetic_filepath = synthetic_filepath
        self.param_dict = param_dict

    # CTGAN and TVAE (belonging to Synthetic Data Vault (sdv) library)
    def generate_sdv(self, num_tuples_to_generate):
        metadata = self.processor.process()
        real_data = pd.read_csv(self.data_path)
        
        if self.synthesizer_name == "ctgan": 
            synthesizer = CTGANSynthesizer(metadata,
                                        verbose = True,
                                        epochs = self.param_dict["epochs"])

        elif self.synthesizer_name == "tvae":
            synthesizer = TVAESynthesizer(metadata,
                                        epochs = self.param_dict["epochs"])

        print("Starting Generator Training")
        synthesizer.fit(real_data)
        
        print(f"Generator Training Completed, Generating {num_tuples_to_generate} of data")
        synthetic_data = synthesizer.sample(num_tuples_to_generate)
        
        # Saves the Synthetic Data csv file to the designated filepath
        synthetic_data.to_csv(self.synthetic_filepath)
        print("Successfully saved the synthetic dataset to", self.synthetic_filepath)

    # DataSynthesizer's Library
    def generate_dpsynthesizer(self, num_tuples_to_generate):
        # Processing input data
        description_file = self.processor.process()
        # description_file = os.path.join(os.getcwd(), "description.json")
        print("DP Synthesizer Processing Completed!")
        # Generating Synthetic Data
        generator = DataGenerator()

        print(f"Generating {num_tuples_to_generate} rows of Synthetic Data.")
        generator.generate_dataset_in_correlated_attribute_mode(
            num_tuples_to_generate, description_file
        )
        
        # Saves the generated synthetic data (csv) to Current Working Directory filepath
        generator.save_synthetic_data(self.synthetic_filepath)
        print("Successfully saved the synthetic dataset to", self.synthetic_filepath)
