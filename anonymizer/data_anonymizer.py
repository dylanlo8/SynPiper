import pandas as pd
from masking_funcs import Masker
from auto_detect import DataAutoDetecter
import matplotlib.pyplot as plt
import numpy as np

class DataAnonymizer:
    def __init__(self, df):
        self.data_masker = Masker()
        self.auto_detector = DataAutoDetecter(df)
        self.data = df
        self.properties_frame = self.auto_detector.construct_column_mapper()

        """
            Properties not initialised yet.

            # self.transformed_data : Transformed dataset 
            # self.default_transformed_data : (Default) transformed dataset

            # self.reidentification_table : Reidentification table of transformed dataset
            # self.default_reidentification_table : Reidentification table of default transformed dataset
        
        """
        
        self.default_transformer_table : pd.DataFrame()

    def change_property(self, colname : str, property_type : str, new_property : str):
        """
        Changes the property at a cellular level. 
        
        Additional: Detects special information types like NRIC and 
            auto update Sensitivity Types accordingly

        Args:
            colname (String): Name of Column
            property_type (String): Either one from ["column_type, "information_type", "sensitivity_type"]
            new_property (String): Name of new property type

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

        # Saves a deep copy of the default transformer table
        self.default_transformer_table = self.transformer_table.copy(deep = True)
        return self.transformer_table
    
    def list_allowed_transformations(self, colname):
        """
            Retrieves the list of allowed masking functions according to the property type of the column.
        """
        return self.col_allowed_funcs[colname]

    def change_masking(self, colname, masking_name):
        """
        Changes the masking function of a column.

        When a new masking function name is specified, check if the masking table is an allowed
        transformation of the column. Apply change if allowed.

        Args:
            colname (String): Name of column
            masking_name (String) : Name of masking function
            
        Returns:
            self.transformer_table (Dataframe): Updated masking table
        """
        try: 
            if colname not in self.data.columns:
                raise ValueError("Column name not found.")
            
            if masking_name not in self.col_allowed_funcs[colname]:
                raise IndexError(f"{masking_name} not found in {colname} list of allowed transformations.")
            
            # Update Mask Table with new Masking Name
            self.transformer_table.loc[colname, 'Transformer'] = masking_name
            return self.transformer_table
        
        # Missing masking table
        except AttributeError:
            print("Masking Table not Found. Call get_mask_table() to load transformer table first.")
    
    # Could include extra args if the function allows for it
    def apply_masking(self):
        """
            Applies the masking functions on every column of the dataset, according to the functions
            specified in the Masking Table. 

            For comparison, the default (auto-suggested) masking functions are also applied to the dataset. 
            The default transformed data is stored in self.default_transformed_data

            Args: None

            Returns:
                self.transformed_data: The transformed dataset after masking functions have been applied.

            
        """
        try:
            self.transformed_data = pd.DataFrame()
            self.default_transformed_data = pd.DataFrame()

            for colname in self.data.columns:
                # Retrieve selected Transformer
                default_transformer = self.default_transformer_table.loc[colname, 'Transformer']
                selected_transformer_name = self.transformer_table.loc[colname, 'Transformer']

                # Retrieve Transformer function
                default_transformer_func = self.data_masker.get_transformer_from_name(default_transformer)
                selected_transformer_func = self.data_masker.get_transformer_from_name(selected_transformer_name)

                # Apply function on column
                default_transformed_col = default_transformer_func(self.data[colname])
                transformed_col = selected_transformer_func(self.data[colname])

                # Add column to transformed data DataFrame
                self.default_transformed_data[colname] = default_transformed_col
                self.transformed_data[colname] = transformed_col

            return self.transformed_data 
        
        except AttributeError:
            print("Masking Table not Found. Call get_mask_table() to load transformer table first.")
    
        
    def get_quasi_identifiers(self):
        """
            Returns the column names of the Quasi-Identifiers (Indirect Identifiers).
        """
        property_frame = self.properties_frame
        series = property_frame['Sensitivity Type'] == 'Indirect Identifier'
        return list(series[series].index)
    
    def get_quasi_original_table(self):
        """
            Returns the subset of original data containing only the quasi-identifiers.
        """
        return self.data.loc[:, self.get_quasi_identifiers()]

    def get_quasi_masked_table(self):
        """
            Args: None

            Returns:
                The subset of transformed data containing only the quasi-identifiers.
                This will be equivalent to the default masked table if no masking functions have been changed.
        """
        return self.transformed_data.loc[:, self.get_quasi_identifiers()]
    
    def get_default_quasi_masked_table(self):
        """
            Args: None

            Returns:
                The subset of transformed data (using recommended maskers) containing only the quasi-identifiers.
        """
        return self.default_transformed_data.loc[:, self.get_quasi_identifiers()]
    
    """
    Evaluation Metrics
    """

    def get_re_identification_table(self):
        # Retrieve Indirect Identifier Masked Table
        quasi_masked_table = self.get_quasi_masked_table()
        default_quasi_masked_table = self.get_default_quasi_masked_table()

        # Generating equivalence class size from duplicated rows
        duplicate_rows = quasi_masked_table[quasi_masked_table.duplicated(keep = False)].value_counts().reset_index()
        default_duplicate_rows = default_quasi_masked_table[default_quasi_masked_table.duplicated(keep = False)].value_counts().reset_index()
        
        # Search unique rows
        non_duplicate_rows = quasi_masked_table[~quasi_masked_table.duplicated(keep=False)]
        default_non_duplicate_rows = default_quasi_masked_table[~default_quasi_masked_table.duplicated(keep=False)]

        # if unique rows exist, store the unique rows as count 1
        if not non_duplicate_rows.empty:
            non_duplicate_rows.loc[:, "count"] = 1
            count_table = pd.concat([duplicate_rows, non_duplicate_rows])
        else:
            count_table = duplicate_rows
        
        if not default_non_duplicate_rows.empty:
            default_non_duplicate_rows.loc[:, "count"] = 1
            default_count_table = pd.concat([default_duplicate_rows, default_non_duplicate_rows])
        else:
            default_count_table = default_duplicate_rows
        
        # Calculate reidentification probabilities (1 * 100 / equivalence class size)
        reidentification_prob = count_table['count'].map(lambda x : 100 / x) 
        default_reidentification_prob = default_count_table['count'].map(lambda x : 100 / x) 

        count_table['reidentifiability proba'] = reidentification_prob
        default_count_table['reidentifiability proba'] = default_reidentification_prob

        # Save as a property
        self.reidentification_table = count_table
        self.default_reidentification_table = default_count_table

        return self.reidentification_table
    
    def avg_re_identification_prob(self):
        """
        Calculate the average reidentification probability of the transformed dataset 
        """
        
        df = self.reidentification_table
        avg_proba = np.mean(df['reidentifiability proba'])
        return avg_proba
    
    def default_avg_re_identification_prob(self):
        """
        Calculate the average reidentification probability of the (default) transformed dataset 
        """

        df = self.default_reidentification_table
        avg_proba = np.mean(df['reidentifiability proba'])
        return avg_proba

    def percentage_rows_above_k_threshold(self, k):
        try:
            df = self.reidentification_table
            return sum(df[df['count'] >= k]['count']) * 100 / sum(df['count'])
        except:
            print("Generate the reidentification table using get_re_identification_table before calling this function again.")

    def default_percentage_rows_above_k_threshold(self, k):
        try:
            df = self.default_reidentification_table
            return sum(df[df['count'] >= k]['count']) * 100 / sum(df['count'])
        except:
            print("Generate the reidentification table using `get_re_identification_table` before calling this function again.")

    def unique_row_proportion(self):
        # Calculate the proportion of rows that are Unique
        # rows of equivalence class size 1 (i.e. score of 100)
        df = self.reidentification_table
        unique_proportion = sum(df[df['count'] == 1]['count']) * 100 / sum(df['count'])
        return unique_proportion
    
    def default_unique_row_proportion(self):
        df = self.default_reidentification_table
        unique_proportion = sum(df[df['count'] == 1]['count']) * 100 / sum(df['count'])
        return unique_proportion
    
    def generate_k_threshold_plot(self):
        k = 2
        percentage = 0
        percentage_default = 0

        lst_percentage = []
        lst_percentage_default = []
        # Cap k at 5
        while k <= 5:
            percentage = self.percentage_rows_above_k_threshold(k)
            percentage_default = self.default_percentage_rows_above_k_threshold(k)
            lst_percentage.append(percentage)
            lst_percentage_default.append(percentage_default)
            k += 1
        
        plt.xticks(range(2, k))
        plt.xlabel("K Threshold")
        plt.ylabel("Proportion of Rows above K Threshold (%)")
        plt.plot(range(2, k), lst_percentage, label = "User")
        plt.plot(range(2, k), lst_percentage_default, label = "Default")
        plt.legend(loc="lower left")
        

        

        








    
