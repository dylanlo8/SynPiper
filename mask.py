import pandas as pd
import numpy as np
from hashlib import sha256



# Dataset Manipulation
def shuffle():
    pass


# Column Manipulation

def swapping(col) -> pd.Series:
    new_col = pd.Series(np.random.permutation(col))
    return new_col

def retention(col) -> pd.Series:
    return col

# Let user know that is the existence of this column but info is removed.
def surpression(col) -> pd.Series:
    surpress = lambda x : '-'
    return col.apply(surpress)

def full_masking(string_col, index) -> pd.Series:
    mask_char = lambda x : len(x) * '-'
    return string_col.apply(mask_char)

def email_masking(email_col, n_chars_to_retain) -> pd.Series:
    def mask_email(email):
        username, domain = email.split('@')
        masked_username = username[:n_chars_to_retain] + '*' * (len(username) - n_chars_to_retain)
        return masked_username + '@' + domain
    
    return email_col.apply(mask_email)

def nric_masking(nric_col) -> pd.Series:
    def mask_nric(nric):
        return (len(nric) - 4) * '*' + nric[-4:]
    
    return nric_col.apply(mask_nric)

def pseudonymize_sha256(col) -> pd.Series:
    def hash_sha256(record):
        return sha256(record.encode('utf-8')).hexdigest()
     
    # Convert elements to String
    string_col = col.map(str)

    # SHA 256 Hashing
    return string_col.apply(hash_sha256)

# Generationsation
def generalise_num_bin(num_col, n_bins = 10) -> pd.Series:
    if num_col.dtype == 'int64':
      return pd.cut(num_col, bins = n_bins, precision = 0)
    elif num_col.dtype == 'float':
      return pd.cut(num_col, bins = n_bins, precision = 3)
    else:
      return None

def generalise_num_bin_mean(num_col) -> pd.Series:
    df_cut = generalise_num_bin(num_col)
    if num_col.dtype == 'int64':
      return df_cut.map(lambda x : round((x.left + x.right) / 2, 0))
    else: # if float
      return df_cut.map(lambda x : (x.left + x.right) / 2)

def generalise_date_bin(date_col, n_bins = 10) -> pd.Series:
    return pd.cut(date_col, n_bins).apply(lambda x : pd.Interval(x.left.normalize(), x.right.normalize()))

def generalise_date_median(date_col):
    pass

# Transposition
def transpose(col):
    pass





