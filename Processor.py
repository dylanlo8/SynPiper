import os
from DataSynthesizer.DataDescriber import DataDescriber

class SDVProcessor:
    def __init__(self, data_path, param_dict):
        params_required = []
        self.data_path = data_path

        for param in param_dict.keys():
            if param not in params_required:
                raise ValueError("Unspecified Parameter")

        self.param_dict = param_dict

    def process(self):
        pass


class DataSynthesizerProcessor:
    description_file = os.path.join(os.getcwd(), "description.json")

    def __init__(self, data_path, param_dict):
        # List of Required Parameters for DataSynthesizer Model
        params_required = [
            "categorical_attributes",
            "epsilon",
            "degree_of_bayesian_network",
        ]

        # Checks if User has keyed in an appropriate Parameter Dictionary
        for param in param_dict.keys():
            if param not in params_required:
                raise ValueError(
                    "Unspecified Parameter, Please follow the correct naming convention"
                )

        self.data_path = data_path
        self.param_dict = param_dict

    def process(self):
        describer = DataDescriber()
        describer.describe_dataset_in_correlated_attribute_mode(
            dataset_file=self.data_path,
            epsilon=self.param_dict["epsilon"],
            k=self.param_dict["degree_of_bayesian_network"],
            attribute_to_is_categorical=self.param_dict["categorical_attributes"],
        )

        print("Saving Dataset Description File")
        describer.save_dataset_description_to_file(self.description_file)
        return self.description_file
