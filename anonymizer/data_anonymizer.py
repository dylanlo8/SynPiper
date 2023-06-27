import pandas as pd
from masking_funcs import Masker
from auto_detect import DataAutoDetecter

class DataAnonymizer:
    def __init__(self, df):
        self.data_masker = Masker()
        self.auto_detector = DataAutoDetecter(df)
        self.data = df
        self.properties_frame = self.auto_detector.construct_column_mapper()

    def change_property(self, colname, property_type, new_property):
        if colname not in self.data.columns:
            raise AttributeError("Unknown Column Name")

        if property_type == "column_type":
            approved_column_types = self.auto_detector.list_approved_column_types()
            if new_property not in approved_column_types:
                raise AttributeError(f"Unknown column type specified. Please specify one of the approved types in {approved_column_types}")
            
            self.properties_frame.loc[colname, 'Column Type'] = new_property
            return self.properties_frame

        elif property_type == "information_type":
            approved_information_types = self.auto_detector.list_approved_information_types()
            if new_property not in approved_information_types:
                raise AttributeError(f"Unknown Information type specified. Please specify one of the approved types in {approved_information_types}")
            
            self.properties_frame.loc[colname, 'Information Type'] = new_property
            return self.properties_frame

        elif property_type == "sensitivity_type":
            approved_sensitivity_types = self.auto_detector.list_approved_sensitivity_types()
            if new_property not in approved_sensitivity_types:
                raise AttributeError(f"Unknown sensitivity type specified. Please specify one of the approved types in {approved_sensitivity_types}")
            
            # Update Column Type
            self.properties_frame.loc[colname, 'Sensitivity Type'] = new_property
            return self.properties_frame

        else:
            raise AttributeError("""Unknown property type found. 
                                 Please specify one of the following, 
                                 [column_type, information_type, sensitivity_type]""")


    def apply_masking() -> pd.DataFrame:
        pass