import os
from DataSynthesizer.DataDescriber import DataDescriber

class SDVProcessor:
    def __init__(self, data, param_dict):
        params_required = []
        self.data = data
        
        for param in param_dict.keys:
            if param not in params_required:
                raise ValueError("Unspecified Parameter")
            
        self.param_dict = param_dict
          
    def process(self):
       pass
    
class DataSynthesizerProcessor:
    description_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '/workingfolder/description.json')
    
    def __init__(self, data, param_dict):
        params_required = ["categorical_attributes", 
                           "epsilon", 
                           "degree_of_bayesian_network", 
                           "num_tuples_to_generate"]
        self.data = data
        
        for param in param_dict.keys:
            if param not in params_required:
                raise ValueError("Unspecified Parameter, Please follow the correct naming convention")
            
        self.param_dict = param_dict
           
    def process(self):
        describer = DataDescriber()
        describer.describe_dataset_in_correlated_attribute_mode(dataset_file = self.data, 
                                                                epsilon = self.param_dict["epsilon"], 
                                                                k= self.param_dict["degree_of_bayesian_network"],
                                                                attribute_to_is_categorical = self.param_dict["categorical_attributes"])
        describer.save_dataset_description_to_file(self.description_file)
