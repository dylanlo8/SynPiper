from sklearn.model_selection import train_test_split
import pandas as pd

def train_val_split(df, label_col, ratio):
    X = df.drop(label_col, axis = 1)
    y = df[label_col]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=ratio)

    df_train = pd.concat([X_train, y_train], axis = 1)
    df_val = pd.concat([X_val, y_val], axis = 1)
    return df_train, df_val