import pandas as pd
import re

class DataAutoDetecter:
    def __init__(self, df):
        # Check if column names are unique
        if len(df.columns.unique()) != len(df.columns):
            raise ValueError("Duplicated Column Names Found.")

        """
        Implemented Properties (UPDATE HERE WHEN NEW PROPERTIES ARE IMPLEMENTED)
        
        Update masking_funcs mapper functions when new properties are being added. 
        This is to allow the program to auto recommend transformers to a particular property set.
        
        """
        self.data_types = ["Continuous", "Categorical", "Datetime", "Unique/Sparse"]
        self.information_types = ["NRIC", "Email", "Phone Number", "Salary", "Others"]
        self.sensitivity_types = ["Direct Identifier", "Indirect Identifier", "Sensitive", "Non-Sensitive"]

        self.data = df
        
    def __detect_column_type(self, column, unique_val_ratio = 0.2):    
        unique_values = column.unique()
        if len(unique_values) == column.shape[0]:
            return 'Unique/Sparse'
        
        if pd.api.types.is_numeric_dtype(column):
            return 'Continuous'
        
        if len(unique_values) <= (unique_val_ratio * len(column)) or pd.api.types.is_object_dtype(column):
            return 'Categorical'
        
        if pd.api.types.is_categorical_dtype(column):
            return 'Categorical'

        if pd.api.types.is_datetime64_dtype(column):
            return 'Datetime'
        
        # If none of the above, consider it as other/unknown type
        return 'Other'
    
    def __detect_information_type(self, column, colname):
        # Column Name Checking
        if col_name_information_checker(colname) != "Others":
            # If others returned -> pass
            return col_name_information_checker(colname)
        
        # Column Elements Checking
        if nric_checker(column):
            return "NRIC"
        
        if phone_checker(column):
            return "Phone Number"
        
        if email_checker(column):
            return "Email"
        
        return "Others"
    
    def __detect_sensitivity_type(self, column, colname):
        # Column Name Checking
        if col_name_sensitivity_checker(colname) != "Others":
            # If others returned -> pass
            return col_name_sensitivity_checker(colname)
        
        # Column Values Checking
        if nric_checker(column):
            return "Direct Identifier"
        
        if phone_checker(column):
            return "Sensitive"
        
        if email_checker(column):
            return "Sensitive"
        
        return "Non-Sensitive"
    
    def __detect_all_column_type(self):
        print("Running Column Type Checker...")
        col_types = []
        for col_name in self.data.columns:
            col_types.append(self.__detect_column_type(self.data[col_name]))

        return col_types
    
    def __detect_all_information_type(self):
        print("Running Information Type Checker...")
        information_types_lst = []
        for col_name in self.data.columns:
            information_types_lst.append(self.__detect_information_type(self.data[col_name], col_name))
        return information_types_lst

    def __detect_all_sensitivity_type(self):
        print("Running Sensitivity Type Checker...")
        sensitivity_types_lst = []
        for col_name in self.data.columns:
            sensitivity_types_lst.append(self.__detect_sensitivity_type(self.data[col_name], col_name))
        return sensitivity_types_lst
    
    def construct_column_mapper(self):
        print("Auto Detecting all Data Properties...")
        column_types = self.__detect_all_column_type()
        information_types = self.__detect_all_information_type()
        sensitivity_types = self.__detect_all_sensitivity_type()

        print("Constructing Properties Frame")
        self.properties_frame = pd.DataFrame(self.data.columns, columns = ['Column Name'])
        self.properties_frame = self.properties_frame.set_index("Column Name") #set column name as index

        self.properties_frame['Column Type'] = column_types
        self.properties_frame['Information Type'] = information_types
        self.properties_frame['Sensitivity Type'] = sensitivity_types 
        
        print("Auto-Generated Properties Frame. To change a property, use the change_property method.")
        return self.properties_frame
    
    # For Checking of Implemented Types
    def list_approved_column_types(self):
        return self.data_types
    
    def list_approved_information_types(self):
        return self.information_types
    
    def list_approved_sensitivity_types(self):
        return self.sensitivity_types
    
"""
    Auto Detecting Properties from Column Names
"""

# Information Type
def col_name_information_checker(colname : str):
    colname = colname.lower()
    if colname == "nric" or colname == "fin":
        return "NRIC"
    
    if colname == "salary" or colname == "pay" or colname == "income":
        return "Salary"
    
    if colname == "Email" or colname == "email":
        return "Email"
    
    return "Others"

# Sensitivity Type
def col_name_sensitivity_checker(colname : str):
    colname = colname.lower()
    if colname == "nric" or colname == "fin":
        return "Direct Identifier"
    
    if colname == "salary" or colname == "pay" or colname == "income":
        return "Sensitive"
    
    if colname == "Email" or colname == "email":
        return "Sensitive"
    
    return "Others"

"""
    Utilities for Auto Dectection of Information Types from Column Values
"""

def nric_checker(col : pd.Series):
    def nric_pattern_checker(row):
        string = str(row) # Convert to string
        pattern = r'^[TS]\d{7}[A-Z]$'
        if re.match(pattern, string):
            return True
        return False
    
    bool_col = col.map(lambda string : nric_pattern_checker(string))

    # More than 90% of rows matches NRIC Pattern
    return True if bool_col.mean() > 0.9 else False


def phone_checker(col : pd.Series):
    def phone_pattern_checker(row):
        string = str(row) # Convert to string
        pattern1 = r'\d{4} \d{4}' #9111 1111
        pattern2 = r'\d{8}' #91111111
        pattern3 = r'\+65 \d{4} \d{4}' # +65 9111 1111
        pattern4 = r'\+65 \d{8}' # +65 91111111

        if re.match(pattern1, string) or re.match(pattern2, string) or re.match(pattern3, string) or re.match(pattern4, string):
            return True

        return False
    
    bool_col = col.map(lambda row : phone_pattern_checker(row))

    # More than 90% of rows matches NRIC Pattern
    return True if bool_col.mean() > 0.9 else False

def email_checker(col):
    def email_pattern_checker(row):
        string = str(row) # Convert to string
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if re.match(pattern, string):
            return True
        return False
    
    bool_col = col.map(lambda row : email_pattern_checker(row))
    return True if bool_col.mean() > 0.9 else False


            

