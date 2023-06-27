import pandas as pd

class DataAutoDetecter:
    def __init__(self, df):
        # Check if column names are unique
        if len(df.columns.unique()) != len(df.columns):
            raise ValueError("Duplicated Column Names Found.")
        
        self.data = df
        self.data_types = ["Continous", "Categorical", "Datetime"]
        self.information_types = ["NRIC", "Email", "Others"]
        self.sensitivity_types = ["Direct Identifier, Indirect Identifier, Sensitive, Non-Sensitive"]

    def __detect_column_type(self, column, unique_val_ratio = 0.4):    
        # Check if the column data type is continuous
        if pd.api.types.is_numeric_dtype(column):
            return 'Continous'

        # Check if the column data type is categorical
        unique_values = column.unique()
        if len(unique_values) <= (unique_val_ratio * len(column)) and pd.api.types.is_object_dtype(column):
            return 'Categorical'

        # Check if the column data type is datetime
        if pd.api.types.is_datetime64_dtype(column):
            return 'Datetime'
        
        # If none of the above, consider it as other/unknown type
        return 'Other'
    
    def __detect_all_column_type(self):
        col_types = []
        for col_name in self.data.columns:
            col_types.append(self.__detect_column_type(self.data[col_name]))

        return col_types
    
    def __detect_information_type(self):
        cols = self.data.columns
        information_types = cols.map(lambda col : "Others") # Others preselected
        return information_types

    def __detect_senesitivity_type(self):
        cols = self.data.columns
        sensitivity_types = cols.map(lambda col : "Non-Sensitive") # Non-sensitive preselected
        return sensitivity_types
    
    def construct_column_mapper(self):
        print("Auto Detecting Data Properties...")
        column_types = self.__detect_all_column_type()
        information_types = self.__detect_information_type()
        sensitivity_types = self.__detect_senesitivity_type()

        print("Constructing Properties Frame")
        self.properties_frame = pd.DataFrame(self.data.columns, columns = ['Column Name'])
        self.properties_frame = self.properties_frame.set_index("Column Name") #set column name as index
        self.properties_frame['Column Type'] = column_types
        self.properties_frame['Information Type'] = information_types
        self.properties_frame['Sensitivity Type'] = sensitivity_types 
        
        return self.properties_frame
    
    # For Checking of Implemented Types
    def list_approved_column_types(self):
        return self.data_types
    
    def list_approved_information_types(self):
        return self.information_types
    
    def list_approved_sensitivity_types(self):
        return self.sensitivity_types


    