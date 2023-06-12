import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import normalized_mutual_info_score

from sdv.evaluation.single_table import get_column_plot
from sdv.metadata import SingleTableMetadata
from sdmetrics.single_column import KSComplement
from sdmetrics.single_column import TVComplement

# Getting all the KS-Statistic Scores (works only for numerical columns)
def get_all_ks_scores(real_table, synthetic_table, numerical_columns):
  results = []
  series_results = {}
  for num_col in numerical_columns:
    ks_score = KSComplement.compute(
      real_data=real_table[num_col],
      synthetic_data=synthetic_table[num_col]
    )

    results.append(ks_score)
    series_results[num_col] = ks_score
  
  plt.figure(figsize = (15,6))
  plots = plt.bar(numerical_columns, results)
  plt.title("Kolmogorov-Smirnov statistic Scores")
  # print(pd.Series(series_results))

#get_all_ks_scores(df, synthetic_data, numerical_columns)

# Total Variational Difference (For Categorical Columns comparison)
def get_all_variational_differences(real_table, synthetic_table, categorical_columns):
  from sdmetrics.single_column import TVComplement

  results = []
  series_results = {}

  for category in categorical_columns:
    tv_score = TVComplement.compute(
        real_data = real_table[category],
        synthetic_data = synthetic_table[category])
    results.append(tv_score)
    series_results[category] = tv_score
  
  plt.figure(figsize = (15,6))
  plots = plt.bar(categorical_columns, results)
  plt.title("Total Variational Difference Scores")
  # print(pd.Series(series_results))

def plot_corr_matrix(real, synthetic):
    fig, (ax1, ax2) = plt.subplots(figsize=(15,6), ncols=2)
    plt.suptitle("Pairwise Correlation Score", fontsize = 25)
    sns.heatmap(real.corr(), ax = ax1)
    ax1.title.set_text("Ground Truth Correlation Matrix")
    sns.heatmap(synthetic.corr(), ax = ax2)
    ax2.title.set_text("Synthetic Data Correlation Matrix")

# Plots a Pairwise Mutual Information Matrix
def plot_mi_matrix(df, df_syn):
    matMI = pd.DataFrame(index = df.columns, columns = df.columns, dtype=float)
    matMI_syn = pd.DataFrame(index = df.columns, columns = df.columns, dtype=float)


    for row in df.columns:
        for col in df.columns:
            matMI.loc[row, col] = normalized_mutual_info_score(df[row],
                                                        df[col],
                                                        average_method = 'arithmetic')
            matMI_syn.loc[row, col] = normalized_mutual_info_score(df_syn[row],
                                                        df_syn[col],
                                                        average_method = 'arithmetic')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    plt.suptitle("Pairwise Mutual Information Score (Normalised)", fontsize = 25)

    #Real
    sns.heatmap(matMI, 
                cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True),
                ax = ax1)
    ax1.set_title('Ground Truth, max=1', fontsize=15)
            
    
    #Synthetic
    sns.heatmap(matMI_syn, 
                cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True),
                ax = ax2)
    ax2.set_title('Synthetic Data, max=1', fontsize=15)
    
    plt.tight_layout()