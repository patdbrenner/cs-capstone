from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix


def build_confusion_matrix_figure(y_test: pd.Series, y_pred, labels: list[str]) -> Figure:
    '''
    Build a confusion matrix figure.
    
    Args:
        y_test: True labels for the test set.
        y_pred: Predicted labels for the test set.
        labels: Ordered list of Class labels for axis ticks.
        
    Returns:
        Matplotlib Figure with heatmap.
    '''
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,)
    ax.set_xlabel('Predicted Category')
    ax.set_ylabel('Actual Category')
    ax.set_title('Confusion Matrix: IT Ticket Classification')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.tight_layout()
    return fig

def build_feature_importance_figure(
    feature_names: np.ndarray, importances: np.ndarray, top_n: int = 20
) -> Figure:
    '''
    Build a bar chart of the top N most important features.
    
    Args:
        feature_names: Array of all feature names.
        importances: Array of importance scores.
        top_n: Number of top features to display.

    Returns:
        Matplotlib Figure with bar chart.    
    '''
    top_indices = np.argsort(importances)[-top_n:][::-1]
    top_features = feature_names[top_indices]
    top_importances = importances[top_indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=top_importances, y=top_features, orient='h', ax=ax)
    ax.set_xlabel('Feature Importance')
    ax.set_ylabel('Term (stemmed)')
    ax.set_title(f'Top {top_n} Most Important Features')
    fig.tight_layout()
    return fig

def build_distribution_figure(counts: pd.Series) -> Figure:
    '''
    Build a bar chart of class distribution.
    
    Args:
        counts: Series of class counts.
        
    Returns:
        Matplotlib Figure with bar chart.
    '''
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=counts.values, y=counts.index, ax=ax)
    ax.set_xlabel('Count')
    ax.set_ylabel('Category')
    fig.tight_layout()
    return fig