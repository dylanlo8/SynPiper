from Processor import SDVProcessor, DataSynthesizerProcessor
from SynPiper import SynPiper
import pandas as pd
import os

data_file = os.path.join(os.getcwd(), "datasets", "heartprocessed.csv")
pd.read_csv(data_file)

piper = SynPiper()
