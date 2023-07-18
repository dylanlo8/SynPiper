from Processor import SDVProcessor, DataSynthesizerProcessor
from DataSynthesizer.DataGenerator import DataGenerator
import os
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
import pandas as pd
import time

class SynPiper:
    """
    Uninitialised Attributes:
        processor: Allocated input data processor based on chosen synthesizer
        generated_samples: synthetic data (pandas Dataframe format)
    
    Initialised Attributes:
        synthesizer_name: Name of Synthesizer
        data_path: File Path of where the Real Data csv is found
        synthetic_filepath: File Path where Synthetic Data csv will be stored
        param_dict: Dictionary of parameters required for synthesizer
    """

    # Initialise appropriate Pre-Processors, check for correct param_dict.
    def __init__(self, data_path, synthesizer_name, param_dict, synthetic_filepath):
        """ Initialiser for Synthesizer
        Args:
            synthesizer_name: Name of Synthesizer
            data_path: File Path of where the Real Data csv is found
            synthetic_filepath: File Path where Synthetic Data csv will be stored
            param_dict: Dictionary of parameters required for synthesizer
        """

        # SDV Pre-Processor
        if synthesizer_name == "ctgan" or synthesizer_name == "tvae":
            print("Initialising SDV Processor")
            self.processor = SDVProcessor(data_path, param_dict)
            print("Processor initialised!")

        # DataSynthesizer Pre-processor
        elif synthesizer_name == "dpsynthesizer":
            print("Initialising DataSynthesizer Processor")
            self.processor = DataSynthesizerProcessor(data_path, param_dict)
            print("Processor initialised!")

        # Unspecified Synthesizer Name Error
        else:
            raise ValueError("Unspecified Synthesizer Name inputted.")

        self.synthesizer_name = synthesizer_name
        self.data_path = data_path
        self.synthetic_filepath = synthetic_filepath
        self.param_dict = param_dict

    def generate(self, num_tuples_to_generate):
        """ General generate function which calls the appropriate generating function
        based on the name of synthesizer initialised.

        Args:
            num_tuples_to_generate: Number of samples to generate

        Returns:
            None
        """

        timer = Timer()
        timer.start()

        if self.synthesizer_name == "ctgan" or self.synthesizer_name == "tvae":
            self.generate_sdv(num_tuples_to_generate=num_tuples_to_generate)

        elif self.synthesizer_name == "dpsynthesizer":
            self.generate_dpsynthesizer(num_tuples_to_generate=num_tuples_to_generate)

        else:
            raise ValueError("Unknown Synthesizer Name found.")
        
        self.elapsed_time = timer.stop()
    
    # CTGAN and TVAE (belonging to Synthetic Data Vault (sdv) library)
    def generate_sdv(self, num_tuples_to_generate):
        print("Processing input data...")
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
        synthetic_data.to_csv(index = 0, path_or_buf= self.synthetic_filepath)
        print("Successfully saved the synthetic dataset to", self.synthetic_filepath)

        # Store the synthetic samples as an attribute
        self.generated_samples = synthetic_data

    # DataSynthesizer's Library
    def generate_dpsynthesizer(self, num_tuples_to_generate):
        
        print("Processing input data...")
        # Processing input data
        description_file = self.processor.process()
        print("DP Synthesizer Processing Complete")

        # Generating Synthetic Data
        generator = DataGenerator()
        print(f"Generating {num_tuples_to_generate} rows of Synthetic Data.")
        generator.generate_dataset_in_correlated_attribute_mode(
            num_tuples_to_generate, description_file
        )
        
        # Saves the generated synthetic data (csv) to synthetic filepath
        generator.save_synthetic_data(self.synthetic_filepath)
        print("Successfully saved the synthetic dataset to", self.synthetic_filepath)
        print("Access the synthetic samples by calling .generated_samples")

        # Store the synthetic samples as an attribute
        self.generated_samples = pd.read_csv(self.synthetic_filepath)


class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        
        return elapsed_time
    
class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""