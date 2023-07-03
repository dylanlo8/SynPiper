import pandas as pd
from masking_funcs import Masker
from auto_detect import DataAutoDetecter

class DataAnonymizer:
    def __init__(self, df):
        self.data_masker = Masker()
        self.auto_detector = DataAutoDetecter(df)
        self.data = df
        self.properties_frame = self.auto_detector.construct_column_mapper()
        #self.transformed_data

    def change_property(self, colname : str, property_type : str, new_property : str):
        """
        Changes the property at a cellular level. 
        
        Additional: Detects special information types like NRIC and 
            auto update Sensitivity Types accordingly

        Args:
            colname (String): Name of Column
            property_type (String): Either one from ["column_type, "information_type", "sensitivity_type"]
            new_property (String): _description_

        Returns:
            self.properties_frame: Updated property frame
        """
        if colname not in self.data.columns:
            raise AttributeError("Unknown Column Name")

        if property_type == "column_type":
            approved_column_types = self.auto_detector.list_approved_column_types()
            if new_property not in approved_column_types:
                raise AttributeError(f"Unknown column type specified. Please specify one of the approved types in {approved_column_types}")
            
            self.properties_frame.loc[colname, 'Column Type'] = new_property

            # Implement Special Changes to other property types here.
            
            return self.properties_frame

        elif property_type == "information_type":
            approved_information_types = self.auto_detector.list_approved_information_types()
            if new_property not in approved_information_types:
                raise AttributeError(f"Unknown Information type specified. Please specify one of the approved types in {approved_information_types}")
            
            self.properties_frame.loc[colname, 'Information Type'] = new_property

            # Implement Special Changes to other property types here.
            if new_property == "NRIC":
                self.properties_frame.loc[colname, 'Sensitivity Type'] = "Direct Identifier"
                self.properties_frame.loc[colname, 'Column Type'] = "Unique/Sparse"

            return self.properties_frame

        elif property_type == "sensitivity_type":
            approved_sensitivity_types = self.auto_detector.list_approved_sensitivity_types()
            if new_property not in approved_sensitivity_types:
                raise AttributeError(f"Unknown sensitivity type specified. Please specify one of the approved types in {approved_sensitivity_types}")
            
            # Implement concurrent changes to other property types here.

            self.properties_frame.loc[colname, 'Sensitivity Type'] = new_property
            return self.properties_frame

        else:
            raise AttributeError("Unknown property type found. Please specify one of the following, [column_type, information_type, sensitivity_type]")

   
    def change_property_rowwise(self, colname: str, property_list : list):
        """Changes the property row-wise

        Args:
            property_list (List): ["Categorical", "Others", "Sensitive"]
            
        Returns:
            self.properties_frame: Updated property frame
        """
        new_column_type = property_list[0]
        new_information_type = property_list[1]
        new_sensitivity_type = property_list[2]
        
        # Check if User Specified inputs are within approved list of properties
        if new_column_type not in self.auto_detector.list_approved_column_types():
            raise ValueError("Column Type not found. Please specify value from approved list.")
        
        if new_information_type not in self.auto_detector.list_approved_information_types():
            raise ValueError("Information Type not found. Please specify value from approved list.")
        
        if new_sensitivity_type not in self.auto_detector.list_approved_sensitivity_types():
            raise ValueError("Sensitivity Type not found. Please specify value from approved list.")
        
        # Perform Modifications
        self.properties_frame.loc[colname, 'Column Type'] = new_column_type
        self.properties_frame.loc[colname, 'Information Type'] = new_information_type
        self.properties_frame.loc[colname, 'Sensitivity Type'] = new_sensitivity_type
        return self.properties_frame
        
    def get_mask_table(self):
        self.col_allowed_funcs = {} # Dictionary Holding all allowed Transformation Names
        # e.g. age : ["Generalise (Numerical Bin)", "Retain", "Suppress"]

        # For each column, store the get the list of functions.
        for col in self.properties_frame.index:
            self.col_allowed_funcs[col] = self.data_masker.generate_list_trans_functions(col_type = self.properties_frame.loc[col, "Column Type"],
                                                                                sensitivity_type= self.properties_frame.loc[col, "Sensitivity Type"],
                                                                                information_type= self.properties_frame.loc[col, "Information Type"])

        optimal_transformers = []
        for optimal_transformer in list(self.col_allowed_funcs.values()):
            optimal_transformers.append(optimal_transformer[0]) # Select the first recommended transformation function.
        
        self.transformer_table = pd.DataFrame({"Column Name" : self.data.columns, 
                                        "Transformer" : optimal_transformers})
        
        self.transformer_table = self.transformer_table.set_index("Column Name")

        return self.transformer_table
    
    def list_allowed_transformations(self, colname):
        return self.col_allowed_funcs[colname]

    def change_masking(self, colname, masking_name):
        if colname not in self.data.columns:
            raise ValueError("Column name not found.")
        if masking_name not in self.col_allowed_funcs[colname]:
            raise IndexError(f"{masking_name} not found in {colname} list of allowed transformations.")
        
        # Update Mask Table with new Masking Name
        self.transformer_table.loc[colname, 'Transformer'] = masking_name
        return self.transformer_table
    
    # Could include extra args if the function allows for it
    def apply_masking(self):
        self.transformed_data = pd.DataFrame()

        for colname in self.data.columns:
            # Retrieve selected Transformer
            selected_transformer_name = self.transformer_table.loc[colname, 'Transformer']
            # Retrieve Transformer function
            selected_transformer_func = self.data_masker.get_transformer_from_name(selected_transformer_name)

            # Apply function on column
            transformed_col = selected_transformer_func(self.data[colname])

            # Add column to transformed data DataFrame
            self.transformed_data[colname] = transformed_col

        return self.transformed_data 
        
    def get_quasi_identifiers(self):
        property_frame = self.properties_frame
        series = property_frame['Sensitivity Type'] == 'Indirect Identifier'
        return series[series].index

    def get_quasi_masked_table(self):
        return self.transformed_data.loc[:, self.get_quasi_identifiers()]