from sklearn.model_selection import train_test_split
import pandas as pd

def train_val_split(df, label_col, ratio):
    """ Splits the dataframe (df) into a train and validation set based on the specified ratio
        The validation set acts as a holdout for evaluation of synthetic data.
        
    Args:
        df (Dataframe): Real dataframe
        label_col (String): Column name of label / target
        ratio (Float) : Ratio from 0 to 1 on the train-val split
    
    Returns:
        df_train (Dataframe) : Training data split for synthetic generation
        df_val (Dataframe) : Hold out validation split for evaluating synthetic data
    """
    X = df.drop(label_col, axis = 1)
    y = df[label_col]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size= 1 - ratio)

    df_train = pd.concat([X_train, y_train], axis = 1)
    df_val = pd.concat([X_val, y_val], axis = 1)
    return df_train, df_val

def count_exact_match_rows(df_real, df_syn):
    """ Count the number of exact match rows between `df1` and `df2`.
    
    Args:
        df_real (DataFrame): Real Dataframe.
        df_syn (DataFrame): Synthetic dataframe.
    
    Returns:
        Exact Math score: The score of exact match rows from the Synthetic df in Real df.
            100% = 0 Exact Matches
            0% = All Synthetic Samples matches a value in the real dataframe
    """

    percent_match = sum(df_syn.equals(row) for _, row in df_real.iterrows()) / df_syn.shape[0]
    score_match = (1 - percent_match) * 100
    return score_match