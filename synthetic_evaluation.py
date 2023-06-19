# General
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import normalized_mutual_info_score
import plotly.express as px
import plotly.figure_factory as ff
import plotly.subplots
import json
from pandas.api.types import is_numeric_dtype


# SDV Metrics Libraries
from sdv.evaluation.single_table import get_column_plot
from sdv.metadata import SingleTableMetadata
from sdmetrics.single_column import KSComplement
from sdmetrics.single_column import TVComplement


def sdv_metadata_auto_processing(real_data, categorical_threshold=10):
    """Metadata Processing for sdv_metrics library
    Processes input data into numerical or categorical data based on the
    threshold of the number of unique values in the column.

    Args:
        real_data : Input Dataframe
        categorical_threshold : Threshold (Integer)
    Returns:
        metadata: Synthetic Data Vault metadata for plotting
    """
    metadata = SingleTableMetadata()

    for col in real_data.columns:
        if (len(real_data[col].unique()) <= categorical_threshold) or not (is_numeric_dtype(real_data[col])):
            metadata.add_column(column_name=col, sdtype="categorical")
        else:
            metadata.add_column(column_name=col, sdtype="numerical")
    return metadata

def sdv_metadata_manual_processing(real_data, categorical_attributes):
    """MANUAL Metadata Processing for sdv_metrics library
    Processes input data into numerical or categorical data based on the
    threshold of the number of unique values in the column.

    Args:
        real_data : Input Dataframe
        categorical_threshold : Threshold (Integer)
    Returns:
        metadata: Synthetic Data Vault metadata for plotting
    """
    
    metadata = SingleTableMetadata()

    for col in real_data.columns:
        if col in categorical_attributes:
            metadata.add_column(column_name=col, sdtype="categorical")
        else:
            metadata.add_column(column_name=col, sdtype="numerical")
            
    return metadata


def plot_real_synthetic(real_data, synthetic, colname):
    """Plots both the real and synthetic distribution of the columns.
    Works for both categorical and continuous data.

    Args:
        real: Real Data
        synthetic: Synthetic Data (in the same format as Real Data)
        colname: Name of the Column (categorical or continuous allowed)

    Returns:
        fig: Plotly Figure of Real and Synthetic Distribution
    """

    # Synthetic Data Vault Processing for get_column_plot function
    metadata = sdv_metadata_auto_processing(real_data)

    # Synthetic Data Vault's custom get_column_plot function
    fig = get_column_plot(
        real_data=real_data,
        synthetic_data=synthetic,
        column_name=colname,
        metadata=metadata,  # Uses sdv custom metadata
    )

    return fig


def get_all_ks_scores(real_table, synthetic_table, numerical_columns):
    """Compute the KS-Statistic Scores numerical columns.
    Args:
        real_table: Real Data
        synthetic_table: Synthetic Data (in the same format as Real Data)
        numerical_columns: List of numerical column names

    Returns:
        df_ks : Pandas DataFrame of the KS-Scores for each numerical column
        fig : A plotly barplot of KS-Scores
    """
    results = []
    series_results = {}

    for num_col in numerical_columns:
        ks_score = KSComplement.compute(
            real_data=real_table[num_col], synthetic_data=synthetic_table[num_col]
        )
        results.append(ks_score)
        series_results[num_col] = ks_score

    # Compile KS-Scores into dataframe
    df_ks = pd.DataFrame(columns=["numerical_columns", "ks_scores"])
    df_ks["numerical_columns"] = numerical_columns
    df_ks["ks_scores"] = results

    # Plot a plotly figure of the different KS Scores across all categorical columns
    fig = px.bar(
        data_frame=df_ks,
        x="numerical_columns",
        y="ks_scores",
        title="Kolmogorov-Smirnov statistic Scores",
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_traces(textposition='inside')

    return df_ks, fig


def get_all_variational_differences(real_table, synthetic_table, categorical_columns):
    """Plots the Total Variational Difference (TVD)
    between the real and synthetic categorical columns.

    Args:
        real_table: Pandas DataFrame of Real Data
        synthetic_table: Pandas DataFrame of Synthetic Data
            (in the same format as Real Data)
        categorical_columns: A List of categorical columns names.

    Returns:
        df_tvd: Pandas DataFrame of the TVD Scores for each categorical column
        fig: A plotly barplot of TVD-Scores

    """
    results = []
    series_results = {}

    for category in categorical_columns:
        tv_score = TVComplement.compute(
            real_data=real_table[category], synthetic_data=synthetic_table[category]
        )
        results.append(tv_score)
        series_results[category] = tv_score

    df_tvd = pd.DataFrame(columns=["categorical_columns", "tvd_scores"])
    df_tvd["categorical_columns"] = categorical_columns
    df_tvd["tvd_scores"] = results

    fig = px.bar(
        data_frame=df_tvd,
        x="categorical_columns",
        y="tvd_scores",
        title="Total Variational Difference Scores",
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_traces(textposition='inside')

    return df_tvd, fig


def plot_corr_matrix(real, synthetic):
    """Plots the Pairwise Correlation Matrix of Real and Synthetic data.
    Args:
        real: Real Data
        synthetic: Synthetic Data (in the same format as Real Data)

    Returns:
        fig: A (1,2) subplot containing the pairwise correlation matrix
        of both real and synthetic data for comparison.
    """
    fig, (ax1, ax2) = plt.subplots(figsize=(15, 6), ncols=2)
    plt.suptitle("Pairwise Correlation Score", fontsize=25)
    sns.heatmap(
        real.corr(), cmap=sns.color_palette("mako", as_cmap=True).reversed(), ax=ax1
    )

    ax1.title.set_text("Ground Truth Correlation Matrix")
    sns.heatmap(
        synthetic.corr(),
        cmap=sns.color_palette("mako", as_cmap=True).reversed(),
        ax=ax2,
    )
    ax2.title.set_text("Synthetic Data Correlation Matrix")
    return fig


# Plots a Pairwise Mutual Information Matrix
def plot_mi_matrix(df, df_syn):
    """Plots the Pairwise Mutual Information Matrix of Real and Synthetic data.
    Args:
        df: Real Data
        df_syn: Synthetic Data (in the same format as Real Data)

    Returns:
        fig: A (1,2) subplot containing the pairwise mutual information matrix
        of both real and synthetic data for comparison.
    """

    matMI = pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)
    matMI_syn = pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)

    # Computing Pairwise Mutual Information Score
    for row in df.columns:
        for col in df.columns:
            matMI.loc[row, col] = normalized_mutual_info_score(
                df[row], df[col], average_method="arithmetic"
            )
            matMI_syn.loc[row, col] = normalized_mutual_info_score(
                df_syn[row], df_syn[col], average_method="arithmetic"
            )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    plt.suptitle("Pairwise Mutual Information Score (Normalised)", fontsize=25)

    # Real
    sns.heatmap(matMI, cmap=sns.color_palette("mako", as_cmap=True).reversed(), ax=ax1)
    ax1.set_title("Ground Truth, max=1", fontsize=15)

    # Synthetic
    sns.heatmap(
        matMI_syn, cmap=sns.color_palette("mako", as_cmap=True).reversed(), ax=ax2
    )
    ax2.set_title("Synthetic Data, max=1", fontsize=15)

    plt.tight_layout()
    return fig

