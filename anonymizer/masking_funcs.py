import pandas as pd
import numpy as np
from hashlib import sha256
from sklearn.preprocessing import LabelEncoder

class Masker:
    def __init__(self):
        # Maps a Name to its Transformer
        self.transform_name_mapper = {
            "Shuffle" : self.shuffle,
            "Retain" : self.retention,
            "Surpress" : self.surpression,
            "Mask Email" : self.email_masking,
            "Mask NRIC" : self.nric_masking,
            "Pseudonymise" : self.pseudonymize_sha256,
            "Full Masking" : self.full_masking,
            "Generalise (Numerical Bin)" : self.generalise_num_bin,
            "Generalise (Numerical Bin Mean)" : self.generalise_num_bin_mean,
            "Generalise (Date Bin)" : self.generalise_date_bin,
            "Generalise (Date Bin Median)" : self.generalise_date_median,
            "Encode" : self.encode,
            "Transpose" : self.transpose
        }

        """
            Mapping Information, Sensitivity, and Column Types to recommended Functions.
            
            Make sure that key values (the property type) must belong to the approved list
            of property types in auto_detect.py file.
        """
        self.information_type_mask_mapper = {
            "NRIC" : ["Mask NRIC"],
            "Email" : ["Mask Email"],
            "Salary" : ["Generalise (Numerical Bin)"],
            "Phone Number" : ["Pseudonymise"],
            "Others" : []
        }

        self.sensitivity_type_mask_mapper = {
            "Direct Identifier" : ["Pseudonymise", "Surpress", "Full Masking"],
            "Indirect Identifier" : [], 
            "Sensitive" : [], 
            "Non-Sensitive" : ["Retain"] 
        }

        self.col_type_mask_mapper = {
            "Categorical" : ["Encode"],
            "Continuous" : ["Generalise (Numerical Bin)", "Generalise (Numerical Bin Mean)"],
            "DateTime" : ["Generalise (Date Bin)", "Generalise (Date Bin Median)"],
            "Unique/Sparse" : ["Pseudonymise"],
            "Other" : []
        }

        self.general_type_funcs = ["Retain", "Surpress", "Pseudonymise", "Full Masking", "Transpose", "Shuffle"]

    
    def generate_list_trans_functions(self, information_type, sensitivity_type, col_type):
        """
        Generates a recommended list of transformers in order of priority.
        [a, b] : order of priority (from left to right a > b).

        Information Type -> Sensitivity Type -> Column Type
        """
        
        result = []
 
        # Information Type
        result += self.information_type_mask_mapper[information_type]

        # Sensitivity Type
        result += self.sensitivity_type_mask_mapper[sensitivity_type]

        if sensitivity_type == "Direct Identifier":
            # early break for direct indentifiers to limit available transformation functions
            return result 

        # Column Type
        result += self.col_type_mask_mapper[col_type]

        # General Type
        for func in self.general_type_funcs:
            # Check if already exist
            if func not in result:
                result.append(func)
        
        return result # Ordered List of Transformation Names

    def get_transformer_from_name(self, transformer_name):
        if transformer_name not in self.transform_name_mapper:
            raise ValueError("Transformer Name not found or added to the list of transformers")
        
        return self.transform_name_mapper[transformer_name]
    """
    DATA TRANSFORMATION METHODS
    """
    # General 
    
    def shuffle(self, col) -> pd.Series:
        new_col = pd.Series(np.random.permutation(col))
        return new_col

    def retention(self, col) -> pd.Series:
        return col

    def surpression(self, col) -> pd.Series:
        # Let user know that is the existence of this column but info is removed.
        surpress = lambda x : '-'
        return col.apply(surpress)

    # Information Type Transformations

    def email_masking(self, email_col, n_chars_to_retain = 0) -> pd.Series:
        def mask_email(email):
            username, domain = email.split('@')
            masked_username = username[:n_chars_to_retain] + '*' * (len(username) - n_chars_to_retain)
            return masked_username + '@' + domain
        
        return email_col.apply(mask_email)

    def nric_masking(self, nric_col) -> pd.Series:
        def mask_nric(nric):
            return (len(nric) - 4) * '*' + nric[-4:]
        
        return nric_col.apply(mask_nric)

    # Direct Identifiers Transformation
    def pseudonymize_sha256(self, col) -> pd.Series:
        def hash_sha256(record):
            return sha256(record.encode('utf-8')).hexdigest()
        
        # Convert elements to String
        string_col = col.map(str)

        # SHA 256 Hashing
        return string_col.apply(hash_sha256)
    
    def full_masking(self, string_col) -> pd.Series:
        mask_char = lambda x : len(x) * '-'
        return string_col.apply(mask_char)

    # Numerical Transformation
    def generalise_num_bin(self, num_col, n_bins = 10) -> pd.Series:
        if num_col.dtype == 'int64':
            return pd.cut(num_col, bins = n_bins, precision = 0)
        elif num_col.dtype == 'float':
            return pd.cut(num_col, bins = n_bins, precision = 3)
        else:
            return None

    def generalise_num_bin_mean(self, num_col, n_bins = 10) -> pd.Series:
        df_cut = self.generalise_num_bin(num_col, n_bins)
        if num_col.dtype == 'int64':
            return df_cut.map(lambda x : round((x.left + x.right) / 2, 0))
        else: # if float
            return df_cut.map(lambda x : (x.left + x.right) / 2)

    # DateTime Transformation
    def generalise_date_bin(self, date_col, n_bins = 10) -> pd.Series:
        return pd.cut(date_col, n_bins).apply(lambda x : pd.Interval(x.left.normalize(), x.right.normalize()))

    def generalise_date_median(date_col, n_bins=10):
        intervals = pd.cut(date_col, n_bins).apply(lambda x: pd.Interval(x.left.normalize(), x.right.normalize()))
        median_dates = intervals.apply(lambda x: x.left + (x.right - x.left) / 2)
        return median_dates
    
    

    # Categorical Values
    def encode(self, cat_col):
        le = LabelEncoder()
        return pd.Series(le.fit_transform(cat_col))

    # Transposition
    def transpose(self, col):
        return col

    # 