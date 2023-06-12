if __name__ == '__main__':
    from Processor import SDVProcessor, DataSynthesizerProcessor
    from SynPiper import SynPiper
    import pandas as pd
    import os

    data_path = os.path.join(os.getcwd(), "datasets", "heartprocessed.csv")

    params_required = {"categorical_attributes" : {'sex': True, 'cp': True, 'fbs': True, 'restecg': True, 
                                                'exang': True, 'slope': True, 'ca' : True, 'thal' : True, 'target' : True},
                    "epsilon" : 0,
                    "degree_of_bayesian_network" : 3}

    piper = SynPiper(data_path, param_dict= params_required, synthesizer_name = "dpsynthesizer")
    piper.generate_dpsynthesizer(num_tuples_to_generate= 297)