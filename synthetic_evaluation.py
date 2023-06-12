# General
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import normalized_mutual_info_score

# SDV Metrics Libraries
from sdv.evaluation.single_table import get_column_plot
from sdv.metadata import SingleTableMetadata
from sdmetrics.single_column import KSComplement
from sdmetrics.single_column import TVComplement

""" Compute the KS-Statistic Scores numerical columns.

Args:
    real_table: Real Data
    synthetic_table: Synthetic Data (in the same format as Real Data)
    numerical_columns: List of numerical column names

"""
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
  
  fig = plt.figure(figsize = (15,6))
  plots = plt.bar(numerical_columns, results)
  plt.title("Kolmogorov-Smirnov statistic Scores")
  # print(pd.Series(series_results))
  return fig

""" Plots both the real and synthetic distribution of the columns.
Works for both categorical and continuous data.

Args:
    real: Real Data
    synthetic: Synthetic Data (in the same format as Real Data)
    colname: Name of the Column (categorical or continuous allowed)
"""

def plot_real_synthetic(real, synthetic, colname):
  metadata = SingleTableMetadata()
  metadata.detect_from_dataframe(data=real)

  fig = get_column_plot(
      real_data=real,
      synthetic_data=synthetic,
      column_name=colname,
      metadata=metadata
  )
  fig.show()

""" Plots all Distributions using the plot_real_synthetic function.
  Args:
    real: Real Data
    synthetic: Synthetic Data (in the same format as Real Data)
"""
def plot_all_real_synthetic(real, synthetic):
  for col in real.columns:
    plot_real_synthetic(real, synthetic, col)

""" Plots the Total Variational Difference (TVD) 
between the real and synthetic categorical columns.
Args:
  real_table: Real Data
  synthetic_table: Synthetic Data (in the same format as Real Data)
  categorical_columns: A list of categorical columns names.

"""
def get_all_variational_differences(real_table, synthetic_table, categorical_columns):
  results = []
  series_results = {}

  
  for category in categorical_columns:
    tv_score = TVComplement.compute(
        real_data = real_table[category],
        synthetic_data = synthetic_table[category])
    results.append(tv_score)
    series_results[category] = tv_score
  
  fig = plt.figure(figsize = (15,6))
  plots = plt.bar(categorical_columns, results)
  plt.title("Total Variational Difference Scores")

  return fig
  # print(pd.Series(series_results))


""" Plots the Pairwise Correlation Matrix.
Args:
  Real: Real Data
  Synthetic: Synthetic Data (in the same format as Real Data)

"""
def plot_corr_matrix(real, synthetic):
  fig, (ax1, ax2) = plt.subplots(figsize=(15,6), ncols=2)
  plt.suptitle("Pairwise Correlation Score", fontsize = 25)
  sns.heatmap(real.corr(), 
              cmap = sns.color_palette("mako", as_cmap=True).reversed(),
              ax = ax1)
  ax1.title.set_text("Ground Truth Correlation Matrix")
  sns.heatmap(synthetic.corr(), 
              cmap = sns.color_palette("mako", as_cmap=True).reversed(),
              ax = ax2)
  ax2.title.set_text("Synthetic Data Correlation Matrix")
  return fig

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
              cmap = sns.color_palette("mako", as_cmap=True).reversed(),
              ax = ax1)
  ax1.set_title('Ground Truth, max=1', fontsize=15)
  
  #Synthetic
  sns.heatmap(matMI_syn, 
              cmap = sns.color_palette("mako", as_cmap=True).reversed(),
              ax = ax2)
  ax2.set_title('Synthetic Data, max=1', fontsize=15)
  
  plt.tight_layout()
  return fig