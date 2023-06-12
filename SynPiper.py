from Processor import SDVProcessor, DataSynthesizerProcessor
from DataSynthesizer.DataGenerator import DataGenerator
import os

class SynPiper:
    # Saves the generated synthetic data into current path
    synthetic_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '/synthetic.csv')
    
    #synthesizer_name : name of synthesizer 
    #data : input data to train synthetic data generator
    
    ## Initialise appropriate Pre-Processors, the pre-processors will check if a correct param_dict has been specified.
    def __init__(self, data, synthesizer_name, param_dict):
        # SDV Pre-Processor
        if synthesizer_name == "ctgan" or synthesizer_name == "tvae":
            print("Initialising SDV Processor")
            self.processor = SDVProcessor(data, param_dict)
        
        # DataSynthesizer Pre-processor
        elif synthesizer_name == "dpsynthesizer":
            print("Initialising DataSynthesizer Processor")
            self.processor = DataSynthesizerProcessor(data, param_dict)
            
        # Unspecified Synthesizer Name Error    
        else:
            raise ValueError("Unspecified Synthesizer Name inputted.")
        
        self.synthesizer_name = synthesizer_name
        self.data = data
    
    def generate(self):
        pass
    
    def generate_ctgan(self):
        pass
        
        # Saves the Synthetic Data csv file to the designated filepath
        
    def generate_tvae(self):
        pass
    
        # Saves the Synthetic Data csv file to the designated filepath
    
    def generate_dpsynthesizer(self, num_tuples_to_generate):
        description_file = self.processor.process()
        generator = DataGenerator()
        generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)
        
        # Saves the generated synthetic data to designated filepath
        generator.save_synthetic_data(self.synthetic_filepath) # Saves the Synthetic Data csv file to the designated filepath
    
    